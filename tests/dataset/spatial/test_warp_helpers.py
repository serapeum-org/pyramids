"""Tests for :mod:`pyramids.dataset.engines._warp` (ARC-66).

The module exists so the five warp sites across `spatial` and `georef` share one
`dstSRS` derivation and one place that pins a warped result's source. These
tests pin both contracts directly rather than through a caller.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset
from pyramids.dataset.engines._warp import dst_srs_arg, warp_to_dataset

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def source() -> Dataset:
    """A 4x4 EPSG:4326 raster to warp."""
    return Dataset.create_from_array(
        np.arange(16, dtype="float32").reshape(4, 4),
        top_left_corner=(0.0, 4.0),
        cell_size=1.0,
        epsg=4326,
    )


class TestDstSrsArg:
    """The `dstSRS` argument prefers an authority string over raw WKT."""

    def test_an_epsg_crs_yields_the_authority_form(self):
        """A CRS with an authority is passed to GDAL as `AUTHORITY:code`.

        Test scenario:
            The authority form makes GDAL write its own canonical WKT for the
            code, matching historical output bytes and avoiding a warning when
            the authority is ESRI rather than EPSG.
        """
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(3857)
        assert dst_srs_arg(sr) == "EPSG:3857", (
            f"expected the authority form, got {dst_srs_arg(sr)!r}"
        )

    def test_an_authority_less_crs_falls_back_to_wkt(self):
        """A custom projection with no authority code is passed as WKT.

        Test scenario:
            A custom orthographic proj4 string resolves to no EPSG code, so
            there is no authority string to send and the full WKT is the only
            lossless form.
        """
        sr = osr.SpatialReference()
        sr.ImportFromProj4("+proj=ortho +lat_0=30 +lon_0=30 +datum=WGS84 +units=m")
        arg = dst_srs_arg(sr)
        assert arg.startswith("PROJCS") or arg.startswith("PROJCRS"), (
            f"an authority-less CRS must fall back to WKT, got {arg[:40]!r}"
        )


class TestWarpToDataset:
    """The warp result pins the source handle, not the pyramids wrapper."""

    def test_the_result_pins_the_source_raster(self, source):
        """`_warp_source` holds the GDAL handle itself.

        Test scenario:
            Engines reach their parent through a `weakref.proxy`, which keeps
            nothing alive. Pinning the wrapper would leave the VRT reading freed
            memory as soon as the source went out of scope.
        """
        result = warp_to_dataset(
            source, gdal.WarpOptions(format="VRT", dstSRS="EPSG:3857")
        )
        assert result._warp_source is source.raster, (
            "the pin must be the source's gdal.Dataset, not the wrapper"
        )

    def test_the_result_reads_after_the_source_wrapper_is_dropped(self, source):
        """The warped VRT survives its source wrapper being collected.

        Test scenario:
            A warped VRT reads through to the raster it was built from on every
            access, so dropping the only other reference to that raster must not
            invalidate the result.
        """
        result = warp_to_dataset(
            source, gdal.WarpOptions(format="VRT", dstSRS="EPSG:3857")
        )
        del source
        gc.collect()
        array = np.asarray(result.read_array())
        assert np.isfinite(array).any(), (
            "the warped VRT must still read after its source wrapper is gone"
        )

    def test_the_wrapper_class_can_be_overridden(self, source):
        """`dataset_class` decides what wraps the result.

        Test scenario:
            The cutline crop wants a plain `Dataset` even when the source is a
            subclass, because its output is no longer a view of anything. The
            default -- the source's own class -- is what every reprojection
            wants.
        """
        result = warp_to_dataset(
            source,
            gdal.WarpOptions(format="VRT", dstSRS="EPSG:3857"),
            dataset_class=Dataset,
        )
        assert type(result) is Dataset, (
            f"expected a plain Dataset, got {type(result).__name__}"
        )

    def test_pin_false_leaves_no_reference(self, source):
        """A materialised result does not read through, so it is not pinned.

        Test scenario:
            Holding the source alive for a result that has already copied every
            pixel would keep an entire raster in memory for nothing.
        """
        result = warp_to_dataset(
            source,
            gdal.WarpOptions(format="MEM", dstSRS="EPSG:3857"),
            pin=False,
        )
        assert getattr(result, "_warp_source", None) is None, (
            "a materialised result must not pin its source"
        )

    def test_the_access_mode_is_forwarded(self, source):
        """`access` reaches the wrapper constructor.

        Test scenario:
            `orthorectify` asks for a read-only result; without forwarding, the
            wrapper would come back writable and its guards would not fire.
        """
        result = warp_to_dataset(
            source,
            gdal.WarpOptions(format="VRT", dstSRS="EPSG:3857"),
            access="read_only",
        )
        assert result.access == "read_only", (
            f"expected a read-only wrapper, got {result.access!r}"
        )

    def test_a_failed_warp_raises_the_caller_s_message(self, source, monkeypatch):
        """GDAL returning `None` becomes a RuntimeError the caller worded.

        Test scenario:
            Each call site describes its own operation -- "could not
            orthorectify", "could not reproject" -- so the shared helper must
            not flatten them into one generic message.
        """
        options = gdal.WarpOptions(format="VRT", dstSRS="EPSG:3857")
        monkeypatch.setattr(gdal, "Warp", lambda *args, **kwargs: None)
        with pytest.raises(RuntimeError, match="could not do the thing"):
            warp_to_dataset(
                source, options, error_message="GDAL could not do the thing."
            )
