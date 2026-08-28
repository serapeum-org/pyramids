"""Tests for :func:`pyramids.dataset.merge.merge_rasters` ``method=`` (PB-7) and
``dst_crs=`` cross-CRS auto-reproject (PY-M).

Covers the overlap-resolution rule added to ``merge_rasters``: the z-order
``first`` / ``last`` paths and the ``min`` / ``max`` / ``sum`` reduction paths
(via ``_merge_reduce``), no-coverage fill, the ``n`` source-nodata knob,
multi-band reduction, and the guard / error branches. The ``dst_crs=`` suite
covers the reproject-before-composite behaviour and its ``_prepare_sources`` /
``_as_srs`` helpers. Source rasters are written to ``tmp_path``.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

import pyramids.dataset.merge as merge_mod
from pyramids.base.crs import reproject_coordinates
from pyramids.base.remote import CloudConfig
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.merge import (
    _as_srs,
    _cloud_config,
    _merge_reduce,
    _prepare_sources,
    _source_bounds,
    merge_rasters,
    stack_bands,
)
from tests._helpers import traced_peak, write_raster

pytestmark = pytest.mark.core


class _FakeSigner:
    """Minimal signer stand-in exposing the ``sign_href`` and ``gdal_env`` hooks.

    Mirrors the two read-time hooks of :class:`pyramids.stac.signers.Signer`
    without pulling in the optional STAC dependency. ``sign_href`` records every
    href it is handed (in ``seen``) and returns it with ``suffix`` appended, so
    tests can assert the signer was applied to each source; the default empty
    suffix keeps local paths openable.

    Args:
        env: The GDAL config mapping the signer advertises.
        suffix: String appended to every href by ``sign_href`` (default ``""``
            → identity rewrite).
    """

    def __init__(self, env, *, suffix=""):
        self._env = dict(env)
        self.suffix = suffix
        self.seen: list[str] = []

    def sign_href(self, href):
        """Record ``href`` and return it with the configured suffix appended."""
        self.seen.append(href)
        return f"{href}{self.suffix}"

    def gdal_env(self):
        """Return the GDAL config mapping (fed into ``CloudConfig.extra``)."""
        return dict(self._env)


@pytest.fixture(scope="function")
def overlapping_pair(tmp_path):
    """Two 4x4 rasters overlapping in a 2-column strip on a shared 6x4 grid.

    Raster A (value 10) sits at columns 0..3; raster B (value 20) at columns
    2..5. Their union is 6 wide × 4 tall, with columns 2..3 overlapping.

    Returns:
        tuple[str, str]: (path_a, path_b).
    """
    a = np.full((4, 4), 10.0, dtype="float32")
    b = np.full((4, 4), 20.0, dtype="float32")
    pa = write_raster(tmp_path / "a.tif", a, (0, 4))
    pb = write_raster(tmp_path / "b.tif", b, (2, 4))
    return pa, pb


class TestMergeMethod:
    """Tests for the ``method=`` overlap rule of ``merge_rasters``."""

    @pytest.mark.parametrize(
        "method, expected_overlap",
        [("last", 20.0), ("first", 10.0), ("min", 10.0), ("max", 20.0), ("sum", 30.0)],
    )
    def test_overlap_resolution(
        self, overlapping_pair, tmp_path, method, expected_overlap
    ):
        """Each method resolves the overlap strip to the expected value.

        Args:
            method: The merge method under test.
            expected_overlap: The value the overlapping columns should hold.

        Test scenario:
            Columns 2..3 are covered by both A(10) and B(20); the non-overlap
            columns keep their single source's value for every method.
        """
        pa, pb = overlapping_pair
        out = tmp_path / f"out_{method}.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, method=method)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.shape == (4, 6), f"Expected union shape (4, 6), got {arr.shape}"
        assert arr[0, 2] == expected_overlap and arr[0, 3] == expected_overlap, (
            f"{method} overlap should be {expected_overlap}, got {arr[0, 2]} / {arr[0, 3]}"
        )
        assert arr[0, 0] == pytest.approx(10.0), f"A-only column changed: {arr[0, 0]}"
        assert arr[0, 5] == pytest.approx(20.0), f"B-only column changed: {arr[0, 5]}"

    @pytest.mark.parametrize("method", ["min", "max", "sum"])
    def test_reduction_byte_identical_across_strip_sizes(
        self, tmp_path, monkeypatch, method
    ):
        """Striping the reduction (and its latitude prune) matches a single-pass merge.

        Test scenario:
            Two vertically-offset sources overlap in a middle row band. Merging with a
            1-row strip — forcing many strips and skipping each source over the strips
            it does not cover — must produce a byte-identical raster to the default
            single-pass merge.
        """
        top = write_raster(
            tmp_path / "top.tif", np.full((4, 4), 10.0, dtype="float32"), (0, 6)
        )
        bottom = write_raster(
            tmp_path / "bottom.tif", np.full((4, 4), 20.0, dtype="float32"), (0, 4)
        )
        single = tmp_path / f"single_{method}.tif"
        merge_rasters([top, bottom], single, no_data_value=-9999.0, method=method)
        monkeypatch.setattr(merge_mod, "_MERGE_STRIP_ROWS", 1)
        stripped = tmp_path / f"stripped_{method}.tif"
        merge_rasters([top, bottom], stripped, no_data_value=-9999.0, method=method)
        assert np.array_equal(
            Dataset.read_file(str(single)).read_array(),
            Dataset.read_file(str(stripped)).read_array(),
        ), f"{method}: 1-row-strip result differs from the single-pass merge"

    @pytest.mark.parametrize("method", ["min", "max", "sum"])
    def test_multiband_reduction_byte_identical_across_strip_sizes(
        self, tmp_path, monkeypatch, method
    ):
        """A multi-band striped reduction matches the single-pass merge on every band.

        Test scenario:
            Two vertically-offset 2-band sources; merging with a 1-row strip must equal
            the single-pass merge across both bands, exercising the per-band strip write
            (`reduced[band_index]` at the strip's row offset).
        """
        top = write_raster(
            tmp_path / "top.tif",
            np.stack(
                [np.full((4, 4), 10.0, "float32"), np.full((4, 4), 11.0, "float32")]
            ),
            (0, 6),
        )
        bottom = write_raster(
            tmp_path / "bottom.tif",
            np.stack(
                [np.full((4, 4), 20.0, "float32"), np.full((4, 4), 21.0, "float32")]
            ),
            (0, 4),
        )
        single = tmp_path / f"single_{method}.tif"
        merge_rasters([top, bottom], single, no_data_value=-9999.0, method=method)
        monkeypatch.setattr(merge_mod, "_MERGE_STRIP_ROWS", 1)
        stripped = tmp_path / f"stripped_{method}.tif"
        merge_rasters([top, bottom], stripped, no_data_value=-9999.0, method=method)
        single_arr = Dataset.read_file(str(single)).read_array()
        stripped_arr = Dataset.read_file(str(stripped)).read_array()
        assert single_arr.shape[0] == 2, f"expected 2 bands, got {single_arr.shape}"
        assert np.array_equal(single_arr, stripped_arr), (
            f"{method}: multi-band striped merge diverged from the single-pass merge"
        )

    def test_reduction_peak_memory_is_bounded_by_the_strip(self, tmp_path, monkeypatch):
        """The min/max/sum merge peaks far below a whole-union pass, proving the strip reduction.

        Test scenario:
            Merge two overlapping 8000x250 sources with 128-row strips and assert the traced
            Python peak stays well below the whole-union float64 byte size (~16 MB). A strip
            reduction holds a few 128-row strips; reading both sources whole and reducing the
            full union cube would peak at least the union size. The bound is the deterministic
            union byte size, not a second measured whole-union peak, so the verdict does not
            depend on the GDAL build's one-time read buffer landing in one measurement (see
            #1049; #1047 fixed the same flake for the reduce test).
        """
        rows, cols = 8000, 250
        pa = write_raster(
            tmp_path / "a.tif", np.ones((rows, cols), dtype="float32"), (0, rows)
        )
        pb = write_raster(
            tmp_path / "b.tif", np.full((rows, cols), 2.0, dtype="float32"), (0, rows)
        )

        # Warm the build's one-time GDAL read buffer with a small windowed read of each
        # source OUTSIDE the traced region so it is already live during the measurement; the
        # absolute ceiling below is the load-bearing backstop (see #1047 / #1049).
        Dataset.read_file(str(pa)).read_array(window=[0, 0, cols, 128])
        Dataset.read_file(str(pb)).read_array(window=[0, 0, cols, 128])

        # Stripped merge.
        monkeypatch.setattr(merge_mod, "_MERGE_STRIP_ROWS", 128)
        with traced_peak() as sp:
            merge_rasters(
                [pa, pb], tmp_path / "big.tif", no_data_value=-1.0, method="max"
            )
        stripped_peak = sp[0]

        # A strip reduction must peak far below a whole-union float64 pass
        # (rows * cols * float64 = ~16 MB); reading both sources whole would peak >= that.
        union_bytes = rows * cols * np.dtype("float64").itemsize
        assert stripped_peak < union_bytes // 2, (
            f"merge peaked at {stripped_peak / 1e6:.1f} MB, not far below the "
            f"{union_bytes / 1e6:.0f} MB whole-union pass — it did not stay under a full "
            "materialisation"
        )

    def test_default_method_is_last(self, overlapping_pair, tmp_path):
        """Omitting method defaults to last-wins (backward compatible).

        Test scenario:
            No method argument yields the same overlap as method='last'.
        """
        pa, pb = overlapping_pair
        out = tmp_path / "default.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(20.0), (
            f"Default should be last-wins (20), got {arr[0, 2]}"
        )

    def test_reduce_fills_uncovered_with_nodata(self, tmp_path):
        """Reduction methods write nodata where no source covers a pixel.

        Test scenario:
            A occupies the top-left 2x2, B the bottom-right 2x2 of a 4x4 union;
            the off-diagonal quadrants are covered by neither and become nodata
            even for 'sum' (which would otherwise yield 0).
        """
        a = np.full((2, 2), 5.0, dtype="float32")
        b = np.full((2, 2), 7.0, dtype="float32")
        pa = write_raster(tmp_path / "tl.tif", a, (0, 4))
        pb = write_raster(tmp_path / "br.tif", b, (2, 2))
        out = tmp_path / "gappy.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, method="sum")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 0] == pytest.approx(5.0), (
            f"Top-left should be A=5, got {arr[0, 0]}"
        )
        assert arr[3, 3] == pytest.approx(7.0), (
            f"Bottom-right should be B=7, got {arr[3, 3]}"
        )
        assert arr[0, 3] == -1.0, (
            f"Uncovered top-right should be nodata -1, got {arr[0, 3]}"
        )
        assert arr[3, 0] == -1.0, (
            f"Uncovered bottom-left should be nodata -1, got {arr[3, 0]}"
        )

    def test_reduce_multiband(self, tmp_path):
        """Reduction operates per band on multi-band sources.

        Test scenario:
            Two 2-band rasters fully overlapping: max picks the larger value in
            each band independently.
        """
        a = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 8.0)]).astype("float32")
        b = np.stack([np.full((3, 3), 4.0), np.full((3, 3), 2.0)]).astype("float32")
        pa = write_raster(tmp_path / "ma.tif", a, (0, 3))
        pb = write_raster(tmp_path / "mb.tif", b, (0, 3))
        out = tmp_path / "mmax.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, method="max")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 1, 1] == pytest.approx(4.0), (
            f"Band 0 max should be 4, got {arr[0, 1, 1]}"
        )
        assert arr[1, 1, 1] == pytest.approx(8.0), (
            f"Band 1 max should be 8, got {arr[1, 1, 1]}"
        )

    def test_n_ignores_source_value_in_reduction(self, tmp_path):
        """The n knob makes a source pixel value count as no-data in reduction.

        Test scenario:
            A is all 10; B is all 20 but with n=20 ignored, so min over the
            overlap is 10 (B's 20 is excluded), not 10-vs-20.
        """
        a = np.full((4, 4), 10.0, dtype="float32")
        b = np.full((4, 4), 20.0, dtype="float32")
        pa = write_raster(tmp_path / "na.tif", a, (0, 4), nodata=-9999.0)
        pb = write_raster(tmp_path / "nb.tif", b, (2, 4), nodata=-9999.0)
        out = tmp_path / "n_min.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, n=20, method="min")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(10.0), (
            f"Overlap min ignoring 20 should be 10, got {arr[0, 2]}"
        )
        assert arr[0, 5] == -1.0, (
            f"B-only column was all-ignored -> nodata, got {arr[0, 5]}"
        )

    def test_invalid_method_raises(self, overlapping_pair, tmp_path):
        """An unknown method raises ValueError.

        Test scenario:
            'mean' is not a supported merge method.
        """
        pa, pb = overlapping_pair
        with pytest.raises(ValueError, match="method must be one of"):
            merge_rasters([pa, pb], tmp_path / "x.tif", method="mean")

    def test_failed_vrt_zorder_raises(self, overlapping_pair, tmp_path, monkeypatch):
        """A None from BuildVRT on the z-order path raises RuntimeError.

        Test scenario:
            Monkeypatching gdal.BuildVRT to return None triggers the defensive
            guard in the last/first path.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = overlapping_pair
        monkeypatch.setattr(merge_mod.gdal, "BuildVRT", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="gdal.BuildVRT returned None"):
            merge_rasters([pa, pb], tmp_path / "x.tif", method="last")

    def test_failed_vrt_reduce_raises(self, overlapping_pair, tmp_path, monkeypatch):
        """A None from BuildVRT on the reduce path raises RuntimeError.

        Test scenario:
            Monkeypatching gdal.BuildVRT to return None triggers the defensive
            guard inside _merge_reduce.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = overlapping_pair
        monkeypatch.setattr(merge_mod.gdal, "BuildVRT", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="gdal.BuildVRT returned None"):
            merge_rasters([pa, pb], tmp_path / "x.tif", method="sum")


@pytest.fixture(scope="function")
def disjoint_pair(tmp_path):
    """Two 4x4 int32 rasters with a 4-column gap between them.

    Raster A (value 10) covers columns 0..3 and raster B (value 20) covers
    columns 8..11 of the 12-wide union grid, leaving columns 4..7 with no
    source coverage.

    Returns:
        tuple[str, str]: (path_a, path_b).
    """
    a = np.full((4, 4), 10, dtype="int32")
    b = np.full((4, 4), 20, dtype="int32")
    pa = write_raster(tmp_path / "left.tif", a, (0, 4))
    pb = write_raster(tmp_path / "right.tif", b, (8, 4))
    return pa, pb


class TestMergeRastersInputContracts:
    """Input/output contracts of ``merge_rasters`` beyond the overlap rule."""

    def test_zorder_init_fills_uncovered_pixels(self, disjoint_pair, tmp_path):
        """``init`` fills pixels no source covers on the z-order path.

        Test scenario:
            Two disjoint tiles leave columns 4..7 uncovered; with
            ``init=-1.0`` / ``no_data_value=-1.0`` those pixels read -1 and
            the output advertises -1 as its nodata marker.
        """
        pa, pb = disjoint_pair
        out = tmp_path / "gap.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, init=-1.0, method="last")
        ds = Dataset.read_file(str(out))
        arr = ds.read_array()
        assert arr.shape == (4, 12), f"Expected union shape (4, 12), got {arr.shape}"
        assert arr[0, 5] == pytest.approx(-1), (
            f"Uncovered pixel should hold init=-1, got {arr[0, 5]}"
        )
        assert ds.no_data_value[0] == pytest.approx(-1.0), (
            f"Output nodata should be -1.0, got {ds.no_data_value[0]}"
        )

    def test_zorder_preserves_source_dtype(self, disjoint_pair, tmp_path):
        """The z-order path keeps the sources' integer dtype.

        Test scenario:
            int32 sources merged with method='last' produce an int32 output
            (BuildVRT + Translate copy the band type through).
        """
        pa, pb = disjoint_pair
        out = tmp_path / "dtype_zorder.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, init=-1.0, method="last")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.dtype == np.int32, f"z-order should preserve int32, got {arr.dtype}"

    def test_reduce_promotes_to_float64(self, disjoint_pair, tmp_path):
        """The reduction path writes Float64 regardless of the source dtype.

        Test scenario:
            int32 sources merged with method='max' produce a float64 output —
            the documented dtype contract of the NaN-aware reducer.
        """
        pa, pb = disjoint_pair
        out = tmp_path / "dtype_reduce.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, method="max")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.dtype == np.float64, f"reduce should write float64, got {arr.dtype}"

    def test_n_ignores_source_value_in_zorder(self, overlapping_pair, tmp_path):
        """``n`` marks a source value as nodata on the z-order path too.

        Test scenario:
            With n=20 every pixel of raster B (all 20s) is treated as source
            nodata: the overlap strip falls back to A's 10 and B-only columns
            become the init fill.
        """
        pa, pb = overlapping_pair
        out = tmp_path / "n_zorder.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, init=-1.0, n=20, method="last")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(10.0), (
            f"Overlap should fall back to A=10, got {arr[0, 2]}"
        )
        assert arr[0, 5] == pytest.approx(-1.0), (
            f"B-only column should be init=-1, got {arr[0, 5]}"
        )
        assert arr[0, 0] == pytest.approx(10.0), f"A-only column changed: {arr[0, 0]}"

    def test_path_object_inputs(self, disjoint_pair, tmp_path):
        """``src`` entries and ``dst`` may be ``pathlib.Path`` objects.

        Test scenario:
            The signature accepts str | Path; passing Path for every argument
            produces the same mosaic as the str form.
        """
        pa, pb = disjoint_pair
        out = tmp_path / "path_objects.tif"
        merge_rasters([Path(pa), Path(pb)], Path(out), no_data_value=-1.0, init=-1.0)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.shape == (4, 12), f"Expected union shape (4, 12), got {arr.shape}"
        assert arr[0, 0] == pytest.approx(10) and arr[0, 11] == pytest.approx(20), (
            f"Tile values lost: left={arr[0, 0]}, right={arr[0, 11]}"
        )


class TestDatasetCollectionMergeMethod:
    """Tests that DatasetCollection.merge threads method through."""

    def test_collection_merge_method(self, overlapping_pair, tmp_path):
        """DatasetCollection.merge forwards method= to merge_rasters.

        Test scenario:
            A file-backed collection of the two overlapping rasters merged with
            method='sum' yields 30 in the overlap.
        """
        from pyramids.dataset.collection import DatasetCollection

        pa, pb = overlapping_pair
        collection = DatasetCollection.read_multiple_files(
            [pa, pb], with_order=False, date=False
        )
        out = tmp_path / "coll_sum.tif"
        collection.merge(out, no_data_value=-9999.0, method="sum")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(30.0), (
            f"Collection sum overlap should be 30, got {arr[0, 2]}"
        )


@pytest.fixture(scope="function")
def shared_crs_pair(tmp_path):
    """Two 4x4 EPSG:4326 rasters overlapping in a 2-column strip (a shared CRS).

    Returns:
        tuple[str, str]: (path_a value 10, path_b value 20).
    """
    a = np.full((4, 4), 10.0, dtype="float32")
    b = np.full((4, 4), 20.0, dtype="float32")
    pa = write_raster(tmp_path / "sa.tif", a, (0, 4), epsg=4326)
    pb = write_raster(tmp_path / "sb.tif", b, (2, 4), epsg=4326)
    return pa, pb


@pytest.fixture(scope="function")
def disagree_pair(tmp_path):
    """Two overlapping rasters in *different* CRSs (EPSG:4326 and EPSG:3857).

    The first is written natively in 4326; the second is the same footprint
    reprojected to 3857 on disk, so the pair genuinely disagrees on CRS.

    Returns:
        tuple[str, str]: (path_4326, path_3857).
    """
    a = np.full((4, 4), 10.0, dtype="float32")
    b = np.full((4, 4), 20.0, dtype="float32")
    pa = write_raster(tmp_path / "da_4326.tif", a, (0, 4), epsg=4326)
    pb_4326 = write_raster(tmp_path / "db_4326.tif", b, (2, 4), epsg=4326)
    pb = str(tmp_path / "db_3857.tif")
    Dataset.read_file(pb_4326).to_crs(3857).to_file(pb)
    return pa, pb


class TestMergeRastersDstCrs:
    """Tests for the ``dst_crs=`` cross-CRS auto-reproject of ``merge_rasters`` (PY-M)."""

    def test_dst_crs_epsg_int_reprojects(self, shared_crs_pair, tmp_path):
        """An EPSG int ``dst_crs`` reprojects sources and stamps the target CRS.

        Test scenario:
            Two EPSG:4326 tiles merged with ``dst_crs=3857`` produce a mosaic
            whose CRS is EPSG:3857.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "int_crs.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, dst_crs=3857)
        result = Dataset.read_file(str(out))
        assert result.epsg == 3857, f"Expected output EPSG 3857, got {result.epsg}"

    def test_dst_crs_string_reprojects(self, shared_crs_pair, tmp_path):
        """A CRS string ``dst_crs`` is parsed and applied like the int form.

        Test scenario:
            ``dst_crs="EPSG:3857"`` yields the same EPSG:3857 mosaic as the int
            ``3857``.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "str_crs.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, dst_crs="EPSG:3857")
        result = Dataset.read_file(str(out))
        assert result.epsg == 3857, f"Expected output EPSG 3857, got {result.epsg}"

    def test_default_shared_crs_no_reproject(self, shared_crs_pair, tmp_path):
        """``dst_crs=None`` with a shared CRS keeps the previous behaviour.

        Test scenario:
            Omitting ``dst_crs`` leaves both EPSG:4326 tiles untouched: the
            output stays in EPSG:4326 and the last-wins overlap is 20.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "default_crs.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0)
        result = Dataset.read_file(str(out))
        arr = result.read_array()
        assert result.epsg == 4326, f"Expected output EPSG 4326, got {result.epsg}"
        assert arr[0, 2] == pytest.approx(20.0), (
            f"Last-wins overlap should be 20, got {arr[0, 2]}"
        )

    def test_disagree_reprojects_onto_first_source_crs(self, disagree_pair, tmp_path):
        """Mismatched CRSs with ``dst_crs=None`` reproject onto the first source.

        Test scenario:
            A 4326 source and a 3857 source merged without ``dst_crs`` are
            composited in the first source's CRS, EPSG:4326.
        """
        pa, pb = disagree_pair
        out = tmp_path / "disagree.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0)
        result = Dataset.read_file(str(out))
        assert result.epsg == 4326, (
            f"Disagreeing sources should mosaic in the first source CRS 4326, got {result.epsg}"
        )

    @pytest.mark.parametrize("method", ["last", "first", "min", "max", "sum"])
    def test_dst_crs_with_each_method(self, shared_crs_pair, tmp_path, method):
        """Reproject composes with every overlap-resolution method.

        Args:
            method: The merge method combined with the reproject.

        Test scenario:
            ``dst_crs=3857`` plus each method produces a readable EPSG:3857
            mosaic (the z-order and reduce paths both honour the reproject).
        """
        pa, pb = shared_crs_pair
        out = tmp_path / f"crs_{method}.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, method=method, dst_crs=3857)
        result = Dataset.read_file(str(out))
        assert result.epsg == 3857, f"{method}: expected EPSG 3857, got {result.epsg}"
        assert result.read_array().size > 0, f"{method}: mosaic is empty"

    def test_invalid_dst_crs_string_raises(self, shared_crs_pair, tmp_path):
        """An unparseable ``dst_crs`` string raises ValueError.

        Test scenario:
            ``dst_crs="not-a-crs"`` cannot be parsed and raises ValueError before
            any compositing happens.
        """
        pa, pb = shared_crs_pair
        with pytest.raises(ValueError, match="Could not parse dst_crs"):
            merge_rasters([pa, pb], tmp_path / "bad.tif", dst_crs="not-a-crs")

    def test_invalid_resampling_raises(self, shared_crs_pair, tmp_path):
        """An unsupported ``resampling`` value raises ValueError.

        Test scenario:
            ``resampling="sinc"`` is not in INTERPOLATION_METHODS and is rejected
            before any compositing.
        """
        pa, pb = shared_crs_pair
        with pytest.raises(ValueError, match="does not exist"):
            merge_rasters(
                [pa, pb], tmp_path / "bad.tif", dst_crs=3857, resampling="sinc"
            )

    @pytest.mark.parametrize("resampling", ["nearest neighbor", "bilinear", "cubic"])
    def test_resampling_methods_reproject(self, shared_crs_pair, tmp_path, resampling):
        """Each supported resampling method reprojects to ``dst_crs`` successfully.

        Args:
            resampling: The resampling method under test.

        Test scenario:
            A reproject to EPSG:3857 with each method produces a 3857 mosaic.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / f"r_{resampling.split()[0]}.tif"
        merge_rasters(
            [pa, pb], out, dst_crs=3857, resampling=resampling, no_data_value=-9999.0
        )
        assert Dataset.read_file(str(out)).epsg == 3857, (
            f"{resampling} did not reproject"
        )

    def test_warp_failure_raises(self, shared_crs_pair, tmp_path, monkeypatch):
        """A None from gdal.Warp during reproject raises RuntimeError.

        Test scenario:
            With ``dst_crs=3857`` forcing a reproject, monkeypatching gdal.Warp
            to return None trips the defensive guard in ``_prepare_sources``.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = shared_crs_pair
        monkeypatch.setattr(merge_mod.gdal, "Warp", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="gdal.Warp returned None"):
            merge_rasters([pa, pb], tmp_path / "x.tif", dst_crs=3857)

    def test_open_failure_raises(self, shared_crs_pair, tmp_path, monkeypatch):
        """A None from gdal.Open while reading source CRS raises RuntimeError.

        Test scenario:
            Monkeypatching gdal.Open to return None trips the guard in the
            CRS-probe loop of ``_prepare_sources``.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = shared_crs_pair
        monkeypatch.setattr(merge_mod.gdal, "Open", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="gdal.Open returned None"):
            merge_rasters([pa, pb], tmp_path / "x.tif")


class TestSourceBounds:
    """Tests for the ``_source_bounds`` extent helper used by the strip reduction."""

    def test_from_path(self, tmp_path):
        """A path resolves to its ``(west, south, east, north)`` extent.

        Test scenario:
            A 4x4 raster at top-left ``(0, 4)`` with unit cells spans x/y ``[0, 4]``.
        """
        path = write_raster(
            tmp_path / "s.tif", np.ones((4, 4), dtype="float32"), (0, 4)
        )
        assert _source_bounds(str(path)) == (0.0, 0.0, 4.0, 4.0), "wrong path bounds"

    def test_from_open_dataset(self, tmp_path):
        """An already-open ``gdal.Dataset`` is used directly, not reopened.

        Test scenario:
            Passing an open handle returns the same extent as passing its path.
        """
        path = write_raster(
            tmp_path / "s.tif", np.ones((4, 4), dtype="float32"), (0, 4)
        )
        assert _source_bounds(gdal.Open(str(path))) == (0.0, 0.0, 4.0, 4.0), (
            "wrong open-dataset bounds"
        )

    def test_unopenable_source_raises(self):
        """A source that cannot be opened raises a clear ``RuntimeError``.

        Test scenario:
            A non-existent path cannot be opened, so the extent lookup fails loudly
            rather than returning a bogus extent.
        """
        with pytest.raises(RuntimeError):
            _source_bounds("/no/such/raster/does-not-exist.tif")


class TestPrepareSources:
    """Tests for the ``_prepare_sources`` reproject helper."""

    def test_shared_crs_reuses_open_handles_no_reproject(self, shared_crs_pair):
        """A shared CRS with no ``dst_crs`` reuses the open handles (no reproject).

        Test scenario:
            When sources agree and ``dst_crs`` is None, no reproject happens.
            Each source is opened once and that same handle is returned as the
            compositor input (and held in keepalive) — so no path is opened
            twice. The handles are plain opens, not warped VRTs.
        """
        pa, pb = shared_crs_pair
        sources, keepalive = _prepare_sources([pa, pb], None)
        assert len(sources) == 2, f"Expected two sources, got {len(sources)}"
        assert all(isinstance(s, gdal.Dataset) for s in sources), (
            f"Cheap path should reuse open datasets, got {[type(s) for s in sources]}"
        )
        assert sources is keepalive, (
            "sources and keepalive should be the same held handles"
        )

    def test_dst_crs_materialises_all_as_datasets(self, shared_crs_pair):
        """An explicit ``dst_crs`` materialises every source as a dataset.

        Test scenario:
            With ``dst_crs`` set, BuildVRT cannot mix paths and datasets, so all
            sources are returned as gdal.Dataset objects and held in keepalive.
        """
        pa, pb = shared_crs_pair
        sources, keepalive = _prepare_sources([pa, pb], 3857)
        assert all(isinstance(s, gdal.Dataset) for s in sources), (
            f"All sources should be gdal.Dataset, got {[type(s) for s in sources]}"
        )
        assert len(keepalive) == len(sources), (
            f"Keepalive should hold every dataset, got {len(keepalive)} vs {len(sources)}"
        )

    def test_disagree_no_dst_crs_materialises_all(self, disagree_pair):
        """Disagreeing CRSs with no ``dst_crs`` still materialise all as datasets.

        Test scenario:
            The matching first source is opened and the mismatched one warped, so
            the returned list is homogeneous gdal.Dataset objects.
        """
        pa, pb = disagree_pair
        sources, keepalive = _prepare_sources([pa, pb], None)
        assert all(isinstance(s, gdal.Dataset) for s in sources), (
            f"Disagree path should yield datasets, got {[type(s) for s in sources]}"
        )
        assert len(keepalive) == 2, (
            f"Both datasets should be held, got {len(keepalive)}"
        )

    def test_crs_less_source_raises(self, shared_crs_pair, tmp_path):
        """A source with no CRS raises a clear ValueError.

        Test scenario:
            A bare GeoTIFF written without a projection is rejected (rather than
            silently mis-aligning the mosaic) when merged with a CRS-bearing one.
        """
        pa, _ = shared_crs_pair
        nocrs = str(tmp_path / "nocrs.tif")
        ds = gdal.GetDriverByName("GTiff").Create(nocrs, 4, 4, 1, gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(np.zeros((4, 4), dtype="float32"))
        ds.FlushCache()
        ds = None
        with pytest.raises(ValueError, match="has no CRS"):
            _prepare_sources([pa, nocrs], None)


class TestAsSrs:
    """Tests for the ``_as_srs`` CRS-parsing helper."""

    def test_epsg_int(self):
        """An EPSG int builds a spatial reference with that authority code.

        Test scenario:
            ``_as_srs(4326)`` returns an SRS whose authority code is 4326.
        """
        srs = _as_srs(4326)
        assert srs.GetAuthorityCode(None) == "4326", (
            f"Expected authority code 4326, got {srs.GetAuthorityCode(None)}"
        )

    def test_crs_string(self):
        """A ``"EPSG:nnnn"`` string parses to the matching spatial reference.

        Test scenario:
            ``_as_srs("EPSG:3857")`` returns an SRS with authority code 3857.
        """
        srs = _as_srs("EPSG:3857")
        assert srs.GetAuthorityCode(None) == "3857", (
            f"Expected authority code 3857, got {srs.GetAuthorityCode(None)}"
        )

    def test_invalid_string_raises(self):
        """An unparseable CRS string raises ValueError.

        Test scenario:
            ``_as_srs("not-a-crs")`` cannot be parsed and raises ValueError.
        """
        with pytest.raises(ValueError, match="Could not parse dst_crs"):
            _as_srs("not-a-crs")

    def test_invalid_epsg_int_raises(self):
        """An invalid EPSG code raises ValueError.

        Test scenario:
            ``_as_srs(999999)`` is not a real EPSG code and raises ValueError.
        """
        with pytest.raises(ValueError, match="Could not parse dst_crs"):
            _as_srs(999999)


@pytest.fixture(scope="function")
def same_grid_bands(tmp_path):
    """Two single-band 4x4 EPSG:4326 rasters sharing one grid (for stacking).

    Returns:
        tuple[str, str]: (band_a value 1, band_b value 2) on identical grids.
    """
    a = np.full((4, 4), 1.0, dtype="float32")
    b = np.full((4, 4), 2.0, dtype="float32")
    pa = write_raster(tmp_path / "band_a.tif", a, (0, 4))
    pb = write_raster(tmp_path / "band_b.tif", b, (0, 4))
    return pa, pb


class TestCloudConfigHelper:
    """Tests for the ``_cloud_config`` signer-to-context helper (PY-N)."""

    def test_none_returns_nullcontext(self):
        """A ``None`` signer yields a no-op nullcontext.

        Test scenario:
            ``_cloud_config(None)`` installs no GDAL config, so callers see the
            previous behaviour unchanged.
        """
        ctx = _cloud_config(None)
        assert isinstance(ctx, nullcontext), f"Expected nullcontext, got {type(ctx)}"

    def test_signer_returns_seeded_cloudconfig(self):
        """A signer yields a CloudConfig carrying its ``gdal_env()`` mapping.

        Test scenario:
            ``_cloud_config(signer)`` returns a CloudConfig whose GDAL config
            equals the signer's advertised environment.
        """
        signer = _FakeSigner({"AWS_REGION": "us-west-2"})
        ctx = _cloud_config(signer)
        assert isinstance(ctx, CloudConfig), f"Expected CloudConfig, got {type(ctx)}"
        assert ctx.as_gdal_config() == {"AWS_REGION": "us-west-2"}, (
            f"CloudConfig should carry the signer env, got {ctx.as_gdal_config()}"
        )


class TestMergeRastersSigner:
    """Tests for the ``signer=`` cloud-config kwarg of ``merge_rasters`` (PY-N)."""

    def test_signer_none_unchanged(self, shared_crs_pair, tmp_path):
        """``signer=None`` leaves the merge result unchanged.

        Test scenario:
            Omitting ``signer`` produces the same last-wins overlap (20) as the
            signer-free baseline.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "no_signer.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, signer=None)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(20.0), (
            f"signer=None overlap should be 20, got {arr[0, 2]}"
        )

    def test_signer_produces_correct_output(self, shared_crs_pair, tmp_path):
        """A signer does not change the merge result for local inputs.

        Test scenario:
            Passing a signer (harmless local-read config) still yields the
            last-wins overlap of 20 — the config only affects cloud access.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "signed.tif"
        merge_rasters(
            [pa, pb],
            out,
            no_data_value=-9999.0,
            signer=_FakeSigner({"GDAL_HTTP_TIMEOUT": "30"}),
        )
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(20.0), (
            f"Signed merge overlap should be 20, got {arr[0, 2]}"
        )

    def test_signer_config_active_during_merge(
        self, shared_crs_pair, tmp_path, monkeypatch
    ):
        """The signer's GDAL config is live while the mosaic is composited.

        Test scenario:
            A spy on gdal.BuildVRT reads the signer's sentinel config option at
            call time; it must see the value, proving CloudConfig was entered.
        """
        from pyramids.dataset import merge as merge_mod

        seen = {}
        real_build_vrt = merge_mod.gdal.BuildVRT

        def spy(*args, **kwargs):
            seen["value"] = gdal.GetConfigOption("PYRAMIDS_TEST_KEY")
            return real_build_vrt(*args, **kwargs)

        monkeypatch.setattr(merge_mod.gdal, "BuildVRT", spy)
        pa, pb = shared_crs_pair
        merge_rasters(
            [pa, pb],
            tmp_path / "active.tif",
            no_data_value=-9999.0,
            signer=_FakeSigner({"PYRAMIDS_TEST_KEY": "on"}),
        )
        assert seen["value"] == "on", (
            f"Signer config should be active during BuildVRT, got {seen.get('value')!r}"
        )

    def test_no_signer_config_absent_during_merge(
        self, shared_crs_pair, tmp_path, monkeypatch
    ):
        """Without a signer no extra config is installed for the merge.

        Test scenario:
            With ``signer=None`` the sentinel config option is unset (None) when
            gdal.BuildVRT runs — the nullcontext path installs nothing.
        """
        from pyramids.dataset import merge as merge_mod

        seen = {}
        real_build_vrt = merge_mod.gdal.BuildVRT

        def spy(*args, **kwargs):
            seen["value"] = gdal.GetConfigOption("PYRAMIDS_TEST_KEY")
            return real_build_vrt(*args, **kwargs)

        monkeypatch.setattr(merge_mod.gdal, "BuildVRT", spy)
        pa, pb = shared_crs_pair
        merge_rasters([pa, pb], tmp_path / "plain.tif", no_data_value=-9999.0)
        assert seen["value"] is None, (
            f"No signer should leave the sentinel unset, got {seen.get('value')!r}"
        )

    @pytest.mark.parametrize("method", ["last", "first", "min", "max", "sum"])
    def test_signer_with_each_method(self, shared_crs_pair, tmp_path, method):
        """Signer composes with every overlap-resolution method.

        Args:
            method: The merge method combined with the signer.

        Test scenario:
            Passing a signer with each method still yields a readable mosaic of
            the union shape (4, 6).
        """
        pa, pb = shared_crs_pair
        out = tmp_path / f"signer_{method}.tif"
        merge_rasters(
            [pa, pb], out, no_data_value=-9999.0, method=method, signer=_FakeSigner({})
        )
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.shape == (
            4,
            6,
        ), f"{method}: expected union shape (4, 6), got {arr.shape}"

    def test_signer_with_dst_crs(self, shared_crs_pair, tmp_path):
        """Signer composes with ``dst_crs`` reprojection.

        Test scenario:
            ``signer`` plus ``dst_crs=3857`` reprojects and stamps EPSG:3857
            while the signer config is applied.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "signer_crs.tif"
        merge_rasters(
            [pa, pb], out, no_data_value=-9999.0, dst_crs=3857, signer=_FakeSigner({})
        )
        result = Dataset.read_file(str(out))
        assert result.epsg == 3857, f"Expected EPSG 3857, got {result.epsg}"

    def test_signer_sign_href_applied_to_each_source(self, shared_crs_pair, tmp_path):
        """H2: ``signer.sign_href`` is called once per source before compositing.

        Test scenario:
            An identity-rewrite signer records each href it signs; after the
            merge its ``seen`` list must equal the two source paths, proving the
            ``sign_href`` hook fires for every source (not only ``gdal_env``).
        """
        pa, pb = shared_crs_pair
        signer = _FakeSigner({})
        merge_rasters(
            [pa, pb], tmp_path / "signed_each.tif", no_data_value=-9999.0, signer=signer
        )
        assert signer.seen == [
            pa,
            pb,
        ], f"sign_href should see each source once, got {signer.seen}"

    def test_signed_href_reaches_mosaic(self, shared_crs_pair, tmp_path, monkeypatch):
        """H2: the *signed* href (not the raw path) is what reaches the mosaic.

        Test scenario:
            A signer that appends ``?sig=tok`` is used; ``_prepare_sources`` is
            stubbed to capture the paths it receives and abort. The captured
            paths must carry the suffix, proving signing happens before the
            GDAL mosaic step — the exact gap H2 fixes (a SAS signer whose
            credential rides the URL would otherwise be dropped).
        """
        captured: dict[str, list[str]] = {}

        class _Stop(Exception):
            pass

        def fake_prepare(src_paths, dst_crs, resampling):
            captured["paths"] = list(src_paths)
            raise _Stop()

        monkeypatch.setattr("pyramids.dataset.merge._prepare_sources", fake_prepare)
        pa, pb = shared_crs_pair
        signer = _FakeSigner({}, suffix="?sig=tok")
        with pytest.raises(_Stop):
            merge_rasters(
                [pa, pb], tmp_path / "x.tif", no_data_value=-9999.0, signer=signer
            )
        assert captured["paths"] == [
            f"{pa}?sig=tok",
            f"{pb}?sig=tok",
        ], f"signed hrefs should reach the mosaic step, got {captured['paths']}"

    def test_url_only_signer_empty_gdal_env(self, shared_crs_pair, tmp_path):
        """H2: a URL-signing signer with an empty ``gdal_env()`` still authenticates.

        Test scenario:
            A signer whose ``gdal_env()`` is ``{}`` (credential rides the href)
            must still have its ``sign_href`` applied to every source — this is
            the case that silently read unauthenticated before H2.
        """
        pa, pb = shared_crs_pair
        signer = _FakeSigner({})
        assert signer.gdal_env() == {}, "precondition: URL-only signer has no env"
        merge_rasters(
            [pa, pb], tmp_path / "url_only.tif", no_data_value=-9999.0, signer=signer
        )
        assert signer.seen == [
            pa,
            pb,
        ], f"URL-only signer's sign_href must still fire per source, got {signer.seen}"


class TestStackBandsSigner:
    """Tests for the ``signer=`` cloud-config kwarg of ``stack_bands`` (PY-N)."""

    def test_signer_none_band_count(self, same_grid_bands):
        """``signer=None`` stacks the inputs into one band per file.

        Test scenario:
            Two single-band rasters stack into a 2-band dataset with no signer.
        """
        pa, pb = same_grid_bands
        result = stack_bands([pa, pb], signer=None)
        assert result.band_count == 2, f"Expected 2 bands, got {result.band_count}"

    def test_signer_band_count(self, same_grid_bands):
        """A signer does not change the stacked band count.

        Test scenario:
            Passing a signer still yields one band per input file.
        """
        pa, pb = same_grid_bands
        result = stack_bands([pa, pb], signer=_FakeSigner({"GDAL_HTTP_TIMEOUT": "30"}))
        assert result.band_count == 2, f"Expected 2 bands, got {result.band_count}"

    def test_signer_config_active_during_stack(self, same_grid_bands, monkeypatch):
        """The signer's GDAL config is live while the bands are stacked.

        Test scenario:
            A spy on Dataset.from_band_files reads the sentinel config option at
            call time and must see it, proving CloudConfig was entered.
        """
        from pyramids.dataset import merge as merge_mod

        seen = {}
        real_from_band_files = merge_mod.Dataset.from_band_files

        def spy(*args, **kwargs):
            seen["value"] = gdal.GetConfigOption("PYRAMIDS_TEST_KEY")
            return real_from_band_files(*args, **kwargs)

        monkeypatch.setattr(merge_mod.Dataset, "from_band_files", spy)
        pa, pb = same_grid_bands
        stack_bands([pa, pb], signer=_FakeSigner({"PYRAMIDS_TEST_KEY": "on"}))
        assert seen["value"] == "on", (
            f"Signer config should be active during stacking, got {seen.get('value')!r}"
        )

    def test_signer_sign_href_applied_to_each_file(self, same_grid_bands):
        """H2: ``signer.sign_href`` fires once per input file before stacking.

        Test scenario:
            An identity-rewrite signer records each href; after the stack its
            ``seen`` list must equal the two input paths, proving ``stack_bands``
            applies the ``sign_href`` hook too (not only ``gdal_env``).
        """
        pa, pb = same_grid_bands
        signer = _FakeSigner({})
        result = stack_bands([pa, pb], signer=signer)
        assert result.band_count == 2, f"Expected 2 bands, got {result.band_count}"
        assert signer.seen == [
            pa,
            pb,
        ], f"sign_href should see each input once, got {signer.seen}"


@pytest.fixture
def uint16_mixed_res_bands(tmp_path):
    """A 10 m and a 20 m uint16 band on the same origin/CRS (nodata 0).

    Mirrors the Sentinel-2 case from issue #362: same unsigned dtype, mismatched
    resolution, so stacking requires align=True.

    Returns:
        tuple[str, str]: (path_10m, path_20m).
    """
    a = Dataset.create_from_array(
        np.arange(16, dtype="uint16").reshape(4, 4),
        top_left_corner=(0.0, 40.0),
        cell_size=10.0,
        epsg=32630,
        no_data_value=0,
    )
    b = Dataset.create_from_array(
        (np.arange(4, dtype="uint16") + 1).reshape(2, 2),
        top_left_corner=(0.0, 40.0),
        cell_size=20.0,
        epsg=32630,
        no_data_value=0,
    )
    pa, pb = str(tmp_path / "b10.tif"), str(tmp_path / "b20.tif")
    a.to_file(pa)
    b.to_file(pb)
    return pa, pb


class TestStackBandsUint16Align:
    """#362: align=True must not overflow on unsigned-dtype bands."""

    def test_stack_bands_uint16_align(self, uint16_mixed_res_bands):
        """stack_bands(align=True) stacks mixed-resolution uint16 bands.

        Test scenario:
            A 10 m + 20 m uint16 pair (nodata 0) stacks into one 2-band uint16
            dataset without the OverflowError from the -9999 template default.
        """
        pa, pb = uint16_mixed_res_bands
        result = stack_bands([pa, pb], align=True, no_data_value=0)
        assert result.band_count == 2, f"expected 2 bands, got {result.band_count}"
        assert result.dtype[0] == "uint16", f"expected uint16, got {result.dtype}"
        assert result.no_data_value[0] == 0, (
            f"nodata should be 0, got {result.no_data_value[0]}"
        )

    def test_from_band_files_uint16_align(self, uint16_mixed_res_bands):
        """from_band_files(align=True) (the underlying API) also succeeds.

        Test scenario:
            The same uint16 mixed-resolution stack via Dataset.from_band_files.
        """
        pa, pb = uint16_mixed_res_bands
        result = Dataset.from_band_files([pa, pb], align=True, no_data_value=0)
        assert result.band_count == 2, f"expected 2 bands, got {result.band_count}"
        assert result.dtype[0] == "uint16", f"expected uint16, got {result.dtype}"

    def test_uint16_align_grid_matches_first(self, uint16_mixed_res_bands):
        """The stacked grid matches the first (10 m) band, not the coarse one.

        Test scenario:
            align resamples the 20 m band onto the 4x4 10 m grid.
        """
        pa, pb = uint16_mixed_res_bands
        result = Dataset.from_band_files([pa, pb], align=True, no_data_value=0)
        assert (result.rows, result.columns) == (
            4,
            4,
        ), f"grid: {(result.rows, result.columns)}"

    def test_uint16_align_inherited_nodata(self, uint16_mixed_res_bands):
        """align=True works when nodata is inherited (not passed) from uint16 sources.

        Test scenario:
            Omitting no_data_value inherits 0 from the sources; the template must
            still not default to -9999 and overflow.
        """
        pa, pb = uint16_mixed_res_bands
        result = Dataset.from_band_files([pa, pb], align=True)
        assert result.band_count == 2, f"expected 2 bands, got {result.band_count}"
        assert result.no_data_value[0] == 0, (
            f"inherited nodata should be 0, got {result.no_data_value[0]}"
        )


class TestMergeNoneGuards:
    """gdal.Translate / gdal.Warp returning None raises a clear RuntimeError (ARC-22)."""

    def test_translate_none_raises(self, overlapping_pair, tmp_path, monkeypatch):
        """A None from the mosaic gdal.Translate raises RuntimeError, not AttributeError."""
        pa, pb = overlapping_pair
        monkeypatch.setattr(gdal, "Translate", lambda *a, **k: None)
        out = str(tmp_path / "o.tif")
        with pytest.raises(RuntimeError, match="Translate returned None"):
            merge_rasters([pa, pb], out, no_data_value=-1.0, method="last")

    def test_reduce_warp_none_raises(self, overlapping_pair, tmp_path, monkeypatch):
        """A None from the per-source gdal.Warp raises RuntimeError in _merge_reduce."""
        pa, pb = overlapping_pair
        monkeypatch.setattr(gdal, "Warp", lambda *a, **k: None)
        out = str(tmp_path / "o.tif")
        with pytest.raises(RuntimeError, match="Warp returned None"):
            _merge_reduce([pa, pb], out, "min", -1.0, "nan")


class TestMergeRastersBbox:
    """Tests for the ``bbox=`` / ``bbox_crs=`` window on ``merge_rasters`` (issue #1064).

    The fixture pair spans a 6x4 union grid on EPSG:4326 at 1.0-degree cells with
    its top-left at ``(0, 4)``, so a native window is easy to state in pixels: the
    bbox ``(1, 1, 4, 3)`` selects columns 1..4 and rows 1..3, i.e. 3x2.
    """

    WINDOW = (1.0, 1.0, 4.0, 3.0)

    @staticmethod
    def _grid(path):
        """Return ``(x_size, y_size, geotransform)`` of a written raster."""
        ds = gdal.Open(str(path))
        return ds.RasterXSize, ds.RasterYSize, ds.GetGeoTransform()

    @pytest.mark.parametrize("method", ["last", "first", "min", "max", "sum"])
    def test_bbox_restricts_the_output_grid(self, overlapping_pair, tmp_path, method):
        """Every method writes only the windowed sub-grid.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: The overlap-resolution rule under test.

        Test scenario:
            Both code paths are covered - z-order via ``projWin`` and the reduction
            path via the clipped union grid - and both must yield the 3x2 window
            rather than the full 6x4 mosaic.
        """
        out = tmp_path / f"win_{method}.tif"
        merge_rasters(list(overlapping_pair), out, method=method, bbox=self.WINDOW)
        x_size, y_size, _ = self._grid(out)
        assert (x_size, y_size) == (3, 2), (
            f"{method}: expected the 3x2 window, got {x_size}x{y_size}"
        )

    @pytest.mark.parametrize("method", ["last", "min"])
    def test_bbox_output_matches_the_same_slice_of_the_full_merge(
        self, overlapping_pair, tmp_path, method
    ):
        """The window holds the same pixels the full merge puts there.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: One z-order and one reduction method.

        Test scenario:
            Restricting the read must not shift or resample anything - a shape-only
            assertion would pass even if the window were taken from the wrong place.
        """
        full = tmp_path / f"full_{method}.tif"
        windowed = tmp_path / f"win_{method}.tif"
        merge_rasters(list(overlapping_pair), full, method=method)
        merge_rasters(list(overlapping_pair), windowed, method=method, bbox=self.WINDOW)

        full_arr = gdal.Open(str(full)).ReadAsArray()
        win_arr = gdal.Open(str(windowed)).ReadAsArray()
        assert np.allclose(win_arr, full_arr[1:3, 1:4], equal_nan=True), (
            f"{method}: window {win_arr.tolist()} != full slice "
            f"{full_arr[1:3, 1:4].tolist()}"
        )

    @pytest.mark.parametrize("method", ["last", "max"])
    def test_windowed_grid_stays_aligned_to_the_full_grid(
        self, overlapping_pair, tmp_path, method
    ):
        """The window's origin lands on a full-merge pixel edge, at the same scale.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: One z-order and one reduction method.

        Test scenario:
            A window that shifted the origin off-grid or changed the cell size would
            silently resample; the snap is what keeps a windowed merge a strict
            sub-grid of the unwindowed one.
        """
        full = tmp_path / f"a_{method}.tif"
        windowed = tmp_path / f"b_{method}.tif"
        merge_rasters(list(overlapping_pair), full, method=method)
        merge_rasters(list(overlapping_pair), windowed, method=method, bbox=self.WINDOW)

        _, _, full_gt = self._grid(full)
        _, _, win_gt = self._grid(windowed)
        assert win_gt[1] == full_gt[1], f"{method}: x cell size changed"
        assert win_gt[5] == full_gt[5], f"{method}: y cell size changed"
        col = (win_gt[0] - full_gt[0]) / full_gt[1]
        row = (win_gt[3] - full_gt[3]) / full_gt[5]
        assert col == pytest.approx(round(col)), f"{method}: x origin off-grid ({col})"
        assert row == pytest.approx(round(row)), f"{method}: y origin off-grid ({row})"

    @pytest.mark.parametrize("method", ["last", "max"])
    def test_bbox_in_another_crs_selects_the_same_area(
        self, overlapping_pair, tmp_path, method
    ):
        """``bbox_crs=`` lets the bbox stay in the caller's own CRS.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared EPSG:4326 union grid.
            tmp_path: pytest temp directory.
            method: One z-order and one reduction method.

        Test scenario:
            The same window is given in EPSG:3857 and must select the same area. A
            reprojected rectangle is a curved quadrilateral, so its envelope can be
            a pixel wider than the native one - the assertion allows that but not a
            full-extent result.
        """
        xs, ys = reproject_coordinates(
            [self.WINDOW[0], self.WINDOW[2]],
            [self.WINDOW[1], self.WINDOW[3]],
            from_crs=4326,
            to_crs=3857,
            precision=None,
        )
        bbox_3857 = (xs[0], ys[0], xs[1], ys[1])
        out = tmp_path / f"m_{method}.tif"
        merge_rasters(
            list(overlapping_pair), out, method=method, bbox=bbox_3857, bbox_crs=3857
        )
        x_size, y_size, _ = self._grid(out)
        assert 3 <= x_size <= 4, f"{method}: expected ~3 cols, got {x_size}"
        assert 2 <= y_size <= 3, f"{method}: expected ~2 rows, got {y_size}"

    @pytest.mark.parametrize("method", ["last", "first", "min", "max", "sum"])
    def test_disjoint_bbox_raises_for_every_method(
        self, overlapping_pair, tmp_path, method
    ):
        """A window that misses the mosaic fails loudly on every path.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: The overlap-resolution rule under test.

        Test scenario:
            GDAL does not treat a disjoint ``projWin`` as an error - it writes a 1x1
            no-data raster at the window's origin, which reads back as a successful
            merge of nothing. Both paths must reject it instead.
        """
        with pytest.raises(ValueError, match="does not overlap"):
            merge_rasters(
                list(overlapping_pair),
                tmp_path / f"x_{method}.tif",
                method=method,
                bbox=(100.0, 100.0, 101.0, 101.0),
            )

    @pytest.mark.parametrize("method", ["last", "sum"])
    def test_no_bbox_is_unchanged(self, overlapping_pair, tmp_path, method):
        """Omitting ``bbox`` merges the full extent, as before.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: One z-order and one reduction method.

        Test scenario:
            The window is opt-in; the default path must not change.
        """
        out = tmp_path / f"f_{method}.tif"
        merge_rasters(list(overlapping_pair), out, method=method)
        x_size, y_size, _ = self._grid(out)
        assert (x_size, y_size) == (6, 4), (
            f"{method}: default merge should stay 6x4, got {x_size}x{y_size}"
        )


class TestRestrictGrid:
    """Unit tests for the grid-clipping helper behind the reduction path."""

    GEOTRANSFORM = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)

    def test_snaps_outward_onto_the_grid(self):
        """A window inside a pixel grows to cover whole pixels.

        Test scenario:
            Rounding inward would silently drop a partially-covered edge pixel the
            caller asked for, so the clip floors the near edges and ceils the far
            ones.
        """
        gt, x_size, y_size = merge_mod._restrict_grid(
            self.GEOTRANSFORM, 6, 4, "", (1.4, 1.4, 3.6, 2.6), None
        )
        assert (x_size, y_size) == (3, 2), f"expected 3x2, got {x_size}x{y_size}"
        assert gt[0] == 1.0, f"x origin not snapped: {gt[0]}"
        assert gt[3] == 3.0, f"y origin not snapped: {gt[3]}"

    def test_clamps_to_the_grid_extent(self):
        """A window larger than the mosaic clips to the mosaic.

        Test scenario:
            Asking for more than exists must not produce a grid larger than the
            union, which would read outside every source.
        """
        _, x_size, y_size = merge_mod._restrict_grid(
            self.GEOTRANSFORM, 6, 4, "", (-50.0, -50.0, 50.0, 50.0), None
        )
        assert (x_size, y_size) == (6, 4), f"expected 6x4, got {x_size}x{y_size}"

    def test_disjoint_window_raises(self):
        """A non-overlapping window raises rather than returning an empty grid.

        Test scenario:
            A zero-sized grid would be written as a valid-looking empty raster.
        """
        with pytest.raises(ValueError, match="does not overlap"):
            merge_mod._restrict_grid(
                self.GEOTRANSFORM, 6, 4, "", (100.0, 100.0, 101.0, 101.0), None
            )


class TestBboxInProjection:
    """Unit tests for the bbox reprojection helper."""

    def test_passthrough_when_no_bbox_crs_given(self):
        """With ``bbox_crs=None`` the bbox is already in the target CRS.

        Test scenario:
            No reprojection should occur, and no CRS is needed to decide that.
        """
        result = merge_mod._bbox_in_projection((1.0, 2.0, 3.0, 4.0), None, "")
        assert result == (1.0, 2.0, 3.0, 4.0), f"passthrough changed the bbox: {result}"

    def test_unprojectable_bbox_raises(self):
        """A bbox outside the target CRS's domain raises instead of yielding inf.

        Test scenario:
            An orthographic projection can only represent the hemisphere it faces;
            the antipodal side reprojects to non-finite coordinates. Left unchecked
            those would flow into the grid arithmetic and produce a nonsense window
            rather than an error.
        """
        ortho = "+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        with pytest.raises(ValueError, match="does not project"):
            merge_mod._bbox_in_projection((175.0, -5.0, 179.0, 5.0), 4326, ortho)


class TestBboxValidation:
    """`bbox` is validated once, up front, for both merge paths."""

    @pytest.mark.parametrize(
        ("bad", "exc", "why"),
        [
            ("1234", TypeError, "a 4-character string is not four coordinates"),
            (b"1234", TypeError, "bytes are not four coordinates"),
            (12.0, TypeError, "a scalar is not a sequence"),
            ((1.0, 2.0, 3.0), ValueError, "three values is not a bbox"),
            ((1.0, 2.0, 3.0, 4.0, 5.0), ValueError, "five values is not a bbox"),
            ((1.0, "south", 3.0, 4.0), TypeError, "a non-numeric element"),
            ((1.0, None, 3.0, 4.0), TypeError, "None is not a coordinate"),
            ((1.0, float("nan"), 3.0, 4.0), ValueError, "NaN is not a coordinate"),
            ((1.0, 2.0, float("inf"), 4.0), ValueError, "inf is not a coordinate"),
            ((4.0, 1.0, 1.0, 3.0), ValueError, "west > east is inverted"),
            ((1.0, 3.0, 4.0, 1.0), ValueError, "south > north is inverted"),
            ((1.0, 1.0, 1.0, 3.0), ValueError, "zero width selects nothing"),
            ((1.0, 1.0, 4.0, 1.0), ValueError, "zero height selects nothing"),
        ],
    )
    def test_rejects_malformed_bbox(self, bad, exc, why):
        """A malformed bbox is refused with a typed error rather than opaque fallout.

        Args:
            bad: The malformed bbox.
            exc: The exception type expected.
            why: What makes it malformed.

        Test scenario:
            Unvalidated, `"1234"` unpacked into four coordinates and silently became
            a window, and a NaN surfaced from deep in the grid arithmetic as
            "cannot convert float NaN to integer".
        """
        with pytest.raises(exc):
            merge_mod._validated_bbox(bad)

    @pytest.mark.parametrize(
        "accepted",
        [
            np.array([1.0, 1.0, 4.0, 3.0]),
            [1.0, 1.0, 4.0, 3.0],
            (1, 1, 4, 3),
        ],
        ids=["ndarray", "list", "ints"],
    )
    def test_accepts_any_iterable_of_four_numbers(self, accepted):
        """A bbox need not be a `Sequence` to be accepted.

        Args:
            accepted: A well-formed bbox in a container that should be allowed.

        Test scenario:
            `np.ndarray` does not register as a `collections.abc.Sequence`, so an
            isinstance check on that ABC rejected `GeoDataFrame.total_bounds` — the
            most natural way a caller in this codebase produces a bbox.
        """
        assert merge_mod._validated_bbox(accepted) == (1.0, 1.0, 4.0, 3.0), (
            f"{type(accepted).__name__} should be accepted as a bbox"
        )

    @pytest.mark.parametrize("method", ["last", "max"])
    def test_malformed_bbox_rejected_on_both_paths(
        self, overlapping_pair, tmp_path, method
    ):
        """Both merge paths reject the same malformed bbox the same way.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: One z-order and one reduction method.

        Test scenario:
            An inverted bbox used to be rejected on one path and silently normalised
            by GDAL on the other.
        """
        with pytest.raises(ValueError):
            merge_rasters(
                list(overlapping_pair),
                tmp_path / f"bad_{method}.tif",
                method=method,
                bbox=(4.0, 1.0, 1.0, 3.0),
            )


class TestBboxPathAgreement:
    """The z-order and reduction paths must select the identical window."""

    @staticmethod
    def _grid(path):
        """Return ``(x_size, y_size, origin_x, origin_y)`` of a written raster."""
        ds = gdal.Open(str(path))
        gt = ds.GetGeoTransform()
        return ds.RasterXSize, ds.RasterYSize, round(gt[0], 6), round(gt[3], 6)

    @pytest.mark.parametrize(
        ("label", "kwargs"),
        [
            ("native", {"bbox": (1.0, 1.0, 4.0, 3.0)}),
            ("reprojected", {"bbox": (1.0, 1.0, 4.0, 3.0), "bbox_crs": 4326}),
            (
                "dst_crs_reprojected",
                {"dst_crs": 3857, "bbox": (1.0, 1.0, 4.0, 3.0), "bbox_crs": 4326},
            ),
        ],
    )
    def test_paths_return_the_same_grid(
        self, overlapping_pair, tmp_path, label, kwargs
    ):
        """`last` and `max` produce the same output grid for the same window.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            label: Names the window form under test.
            kwargs: The window arguments passed to both methods.

        Test scenario:
            The two paths once reprojected the window independently and rounded to
            opposite sides of a pixel edge, returning different rasters for identical
            arguments (3x3 at x=111364.8 against 4x3 at x=0.0 under `dst_crs=3857`).
            This is the regression guard for that.
        """
        z_order = tmp_path / f"z_{label}.tif"
        reduce_ = tmp_path / f"r_{label}.tif"
        merge_rasters(list(overlapping_pair), z_order, method="last", **kwargs)
        merge_rasters(list(overlapping_pair), reduce_, method="max", **kwargs)
        assert self._grid(z_order) == self._grid(reduce_), (
            f"{label}: z-order {self._grid(z_order)} != reduce {self._grid(reduce_)}"
        )
        # Matching geometry is not enough: two paths can agree on the grid and still
        # disagree on which source won a pixel. Comparing content is meaningful for
        # this fixture because the later raster also holds the larger value (B=20
        # over A=10), so `last` and `max` must resolve the overlap identically.
        z_values = gdal.Open(str(z_order)).ReadAsArray()
        r_values = gdal.Open(str(reduce_)).ReadAsArray()
        assert np.array_equal(z_values, r_values, equal_nan=True), (
            f"{label}: the two paths agree on the grid but not on its content\n"
            f"z-order:\n{z_values}\nreduce:\n{r_values}"
        )


class TestBboxReprojectionCurvature:
    """The reprojected window must cover the whole requested area, not a chord."""

    def test_envelope_covers_a_curved_edge(self):
        """The envelope of a strongly curved reprojection is not the corner envelope.

        Test scenario:
            EPSG:3035 (Lambert azimuthal) into lon/lat bows each edge outward, so the
            extreme latitude lies in an edge's interior. A four-corner envelope fell
            0.035 deg (~4 km) short of the true north edge. Densely sampling the edges
            gives the bound the implementation must at least reach.
        """
        bbox = (3000000.0, 3000000.0, 4500000.0, 4000000.0)
        computed = merge_mod._bbox_in_projection(bbox, 3035, "EPSG:4326")

        samples_x, samples_y = [], []
        for step in np.linspace(0.0, 1.0, 200):
            samples_x += [bbox[0] + step * (bbox[2] - bbox[0])] * 2 + [bbox[0], bbox[2]]
            samples_y += [bbox[1], bbox[3]] + [bbox[1] + step * (bbox[3] - bbox[1])] * 2
        lons, lats = reproject_coordinates(
            samples_x, samples_y, from_crs=3035, to_crs=4326, precision=None
        )
        corner_north = max(
            reproject_coordinates(
                [bbox[0], bbox[2], bbox[0], bbox[2]],
                [bbox[1], bbox[3], bbox[3], bbox[1]],
                from_crs=3035,
                to_crs=4326,
                precision=None,
            )[1]
        )
        assert max(lats) - corner_north > 0.01, (
            "precondition: this CRS pair must actually curve, otherwise the test "
            "cannot distinguish a corner envelope from a densified one"
        )
        assert computed[3] > corner_north, (
            f"north edge {computed[3]} is only the corner bound {corner_north}; the "
            "envelope is not densified"
        )
        assert computed[3] == pytest.approx(max(lats), abs=0.005), (
            f"north edge {computed[3]} does not reach the densified bound {max(lats)}"
        )


class TestBboxGridGeometry:
    """`_restrict_grid` on grids that are not the north-up, axis-aligned default."""

    def test_south_up_grid_is_not_rejected(self):
        """A positive pixel height still resolves a window — a helper invariant only.

        Test scenario:
            Ordering the row offsets by value, rather than assuming north-up, is what
            stops a south-up grid raising a spurious "does not overlap".

            This pins `_restrict_grid` on its own terms and is NOT evidence that
            pyramids merges south-up rasters. The case cannot be reached end-to-end:
            the union grid always comes from `gdal.BuildVRT`, which skips south-up
            sources outright ("does not support positive NS resolution"), and
            `_merge_reduce`'s strip loop would hand `gdal.Warp` an `outputBounds` with
            `minY > maxY` if one ever did arrive. The ordering stays as defensive code
            so the helper remains correct in isolation.
        """
        gt, x_size, y_size = merge_mod._restrict_grid(
            (0.0, 1.0, 0.0, 0.0, 0.0, 1.0), 6, 4, "", (1.0, 1.0, 4.0, 3.0), None
        )
        assert (x_size, y_size) == (3, 2), f"expected 3x2, got {x_size}x{y_size}"
        assert gt[3] == 1.0, f"south-up origin should be the low edge, got {gt[3]}"

    @pytest.mark.parametrize(
        "geotransform",
        [
            (0.0, 1.0, 0.5, 4.0, 0.0, -1.0),
            (0.0, 1.0, 0.0, 4.0, 0.2, -1.0),
        ],
        ids=["row_skew", "col_skew"],
    )
    def test_rotated_or_sheared_grid_is_refused(self, geotransform):
        """A skewed geotransform is refused rather than silently mis-georeferenced.

        Args:
            geotransform: A grid carrying a non-zero skew term.

        Test scenario:
            The window arithmetic assumes an axis-aligned grid; applying it to a
            rotated one would place the output in the wrong location.
        """
        with pytest.raises(ValueError, match="rotated or sheared"):
            merge_mod._restrict_grid(geotransform, 6, 4, "", (1.0, 1.0, 4.0, 3.0), None)

    def test_zero_pixel_size_is_refused(self):
        """A degenerate geotransform is refused rather than dividing by zero.

        Test scenario:
            A zero pixel size would raise ZeroDivisionError from inside the offset
            arithmetic.
        """
        with pytest.raises(ValueError, match="zero pixel size"):
            merge_mod._restrict_grid(
                (0.0, 0.0, 0.0, 4.0, 0.0, -1.0), 6, 4, "", (1.0, 1.0, 4.0, 3.0), None
            )

    def test_degenerately_thin_window_is_diagnosed_as_thin(self):
        """A sub-tolerance window inside the mosaic is not called disjoint.

        Test scenario:
            The tolerance is applied inward at both edges, so a box starting on a
            pixel boundary and spanning less than it snaps to zero width while
            sitting squarely inside the mosaic. Reporting "does not overlap" sent
            the caller to inspect their extents instead of their box width.
        """
        with pytest.raises(ValueError, match="selects no whole pixel"):
            merge_mod._restrict_grid(
                (0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
                6,
                4,
                "",
                (1.0, 1.0, 1.0000001, 3.0),
                None,
            )

    def test_disjoint_window_is_still_diagnosed_as_disjoint(self):
        """A box that genuinely misses the mosaic keeps the overlap message.

        Test scenario:
            The thin-window check runs first, so it must not swallow the disjoint
            case it was split out from.
        """
        with pytest.raises(ValueError, match="does not overlap"):
            merge_mod._restrict_grid(
                (0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
                6,
                4,
                "",
                (100.0, 100.0, 104.0, 103.0),
                None,
            )

    def test_edge_on_a_pixel_boundary_adds_no_extra_pixel(self):
        """A window landing exactly on grid lines yields exactly that many pixels.

        Test scenario:
            Float noise puts an edge a few ulps past a boundary; snapping outward on
            that noise costs a spurious row or column and shifts the origin, which is
            how the two merge paths came to disagree.
        """
        nudged = (1.0 + 1e-12, 1.0 - 1e-12, 4.0 - 1e-12, 3.0 + 1e-12)
        _, x_size, y_size = merge_mod._restrict_grid(
            (0.0, 1.0, 0.0, 4.0, 0.0, -1.0), 6, 4, "", nudged, None
        )
        assert (x_size, y_size) == (3, 2), (
            f"a boundary-aligned window should be 3x2, got {x_size}x{y_size}"
        )


class TestCollectionMergeBbox:
    """`DatasetCollection.merge` forwards the window to `merge_rasters`."""

    @pytest.fixture
    def two_day_collection(self, tmp_path):
        """A file-backed collection of two 4x4 tiles on a shared 6x4 union grid."""
        for index, left in enumerate((0, 2)):
            write_raster(
                tmp_path / f"2024-01-0{index + 1}.tif",
                np.full((4, 4), 10.0 + index, dtype="float32"),
                (left, 4),
            )
        return DatasetCollection.from_files(
            str(tmp_path), glob="*.tif", date_format="%Y-%m-%d"
        )

    def test_merge_without_bbox_is_the_full_union(self, two_day_collection, tmp_path):
        """The default still merges the whole union grid.

        Args:
            two_day_collection: Collection of two overlapping tiles.
            tmp_path: pytest temp directory.

        Test scenario:
            The window is opt-in; adding the parameter must not change the default.
        """
        out = tmp_path / "full.tif"
        two_day_collection.merge(out)
        ds = gdal.Open(str(out))
        assert (ds.RasterXSize, ds.RasterYSize) == (6, 4), (
            f"expected the full 6x4 union, got {ds.RasterXSize}x{ds.RasterYSize}"
        )

    def test_merge_with_bbox_restricts_the_output(self, two_day_collection, tmp_path):
        """A bbox reaches `merge_rasters` and restricts the merge.

        Args:
            two_day_collection: Collection of two overlapping tiles.
            tmp_path: pytest temp directory.

        Test scenario:
            Without this the motivating STAC workflow could not use the window at
            all — the collection is how those mosaics are actually built.
        """
        out = tmp_path / "win.tif"
        two_day_collection.merge(out, bbox=(1.0, 1.0, 4.0, 3.0))
        ds = gdal.Open(str(out))
        assert (ds.RasterXSize, ds.RasterYSize) == (3, 2), (
            f"expected the 3x2 window, got {ds.RasterXSize}x{ds.RasterYSize}"
        )

    def test_merge_bbox_crs_is_forwarded(self, two_day_collection, tmp_path):
        """`bbox_crs` reaches `merge_rasters` rather than being dropped.

        Args:
            two_day_collection: Collection of two overlapping tiles.
            tmp_path: pytest temp directory.

        Test scenario:
            A silently-ignored `bbox_crs` would read the window as if it were in the
            mosaic's CRS, selecting the wrong area instead of failing.
        """
        xs, ys = reproject_coordinates(
            [1.0, 4.0], [1.0, 3.0], from_crs=4326, to_crs=3857, precision=None
        )
        out = tmp_path / "win_crs.tif"
        two_day_collection.merge(out, bbox=(xs[0], ys[0], xs[1], ys[1]), bbox_crs=3857)
        ds = gdal.Open(str(out))
        assert 3 <= ds.RasterXSize <= 4, f"expected ~3 cols, got {ds.RasterXSize}"
        assert 2 <= ds.RasterYSize <= 3, f"expected ~2 rows, got {ds.RasterYSize}"

    def test_merge_rejects_a_malformed_bbox(self, two_day_collection, tmp_path):
        """Validation is not bypassed by going through the collection.

        Args:
            two_day_collection: Collection of two overlapping tiles.
            tmp_path: pytest temp directory.

        Test scenario:
            The collection forwards the window unchanged, so `merge_rasters`' checks
            must still apply.
        """
        with pytest.raises(ValueError):
            two_day_collection.merge(tmp_path / "bad.tif", bbox=(4.0, 1.0, 1.0, 3.0))


class TestBboxAntimeridian:
    """A window that wraps the antimeridian must be refused, not inverted."""

    ANTIMERIDIAN_BBOX = (700000.0, 100000.0, 950000.0, 300000.0)
    UTM_60N = 32660
    # A constant rather than a helper call: inside a `pytest.raises` block a second
    # invocation could itself be the one that raises, which is what the assertion is
    # meant to pin down.
    LONLAT = "EPSG:4326"

    def test_wrapped_envelope_is_refused(self):
        """A reprojection that wraps past 180 deg raises instead of inverting.

        Test scenario:
            `pyproj.transform_bounds` signals an antimeridian crossing by returning
            west > east rather than by widening the envelope. This UTM 60N window is
            about 2.25 deg wide and reprojects to (178.797, ..., -178.955); read at
            face value that is a box spanning the long way round.
        """
        with pytest.raises(ValueError, match="crosses the antimeridian"):
            merge_mod._bbox_in_projection(
                self.ANTIMERIDIAN_BBOX, self.UTM_60N, self.LONLAT
            )

    def test_wrapped_window_does_not_become_its_complement(self):
        """The wrapped window never resolves to the rest of the world.

        Test scenario:
            `_restrict_grid` sorts the two column offsets, so before this was caught
            the ~2.25 deg request resolved to 35776 columns of a global 0.01 deg grid
            — a 357.76 deg window, the complement of the one asked for. That is both
            the wrong area and a near-global read from a call made to bound one.
        """
        global_grid = (-180.0, 0.01, 0.0, 90.0, 0.0, -0.01)
        with pytest.raises(ValueError, match="crosses the antimeridian"):
            merge_mod._restrict_grid(
                global_grid,
                36000,
                18000,
                self.LONLAT,
                self.ANTIMERIDIAN_BBOX,
                self.UTM_60N,
            )

    def test_an_ordinary_reprojected_window_still_passes(self):
        """The guard rejects only wrapped envelopes, not every reprojection.

        Test scenario:
            A guard keyed on west > east would break every ordinary cross-CRS window
            if reprojection could produce that ordering for other reasons; this
            pins that it does not.
        """
        west, south, east, north = merge_mod._bbox_in_projection(
            (500000.0, 100000.0, 600000.0, 300000.0), 32633, self.LONLAT
        )
        assert west < east, f"west {west} should stay below east {east}"
        assert south < north, f"south {south} should stay below north {north}"


class TestBboxReprojectionFailure:
    """A CRS the transformer cannot build is reported against the bbox."""

    @pytest.mark.parametrize(
        "bad_crs",
        ["not-a-crs", 999999],
        ids=["garbage_string", "unknown_epsg"],
    )
    def test_unusable_bbox_crs_raises_a_clear_error(self, bad_crs):
        """An unusable `bbox_crs` raises rather than escaping as a pyproj error.

        Args:
            bad_crs: A CRS pyproj cannot resolve.

        Test scenario:
            The handler around the transform exists so the caller learns which bbox
            and which CRS failed. Left uncaught they would get a bare pyproj CRSError
            naming neither.
        """
        with pytest.raises(ValueError, match="could not be reprojected"):
            merge_mod._bbox_in_projection((1.0, 1.0, 4.0, 3.0), bad_crs, "EPSG:4326")


class TestBboxLongitudeConvention:
    """A lon/lat window is read in the mosaic's own longitude convention."""

    SIGNED_GRID = (-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    WRAPPED_GRID = (0.0, 1.0, 0.0, 90.0, 0.0, -1.0)

    def test_signed_window_on_a_wrapped_mosaic(self):
        """A -180..180 window resolves against a 0..360 mosaic.

        Test scenario:
            Global grids from climate NetCDF commonly run 0..360 while callers write
            signed bboxes. Untranslated, the two overlap only partially and the clamp
            returned that sliver as a success — the eastern part of the requested
            area, with no error.
        """
        clipped, x_size, y_size = merge_mod._restrict_grid(
            self.WRAPPED_GRID, 360, 180, "EPSG:4326", (-10.0, -5.0, -2.0, 5.0), None
        )
        assert (x_size, y_size) == (8, 10), f"expected 8x10, got {x_size}x{y_size}"
        assert clipped[0] == 350.0, f"expected origin 350.0, got {clipped[0]}"

    def test_wrapped_window_on_a_signed_mosaic(self):
        """A 0..360 window resolves against a -180..180 mosaic.

        Test scenario:
            The translation has to work in both directions, not just the one the
            climate-grid case motivates.
        """
        clipped, x_size, y_size = merge_mod._restrict_grid(
            self.SIGNED_GRID, 360, 180, "EPSG:4326", (350.0, -5.0, 358.0, 5.0), None
        )
        assert (x_size, y_size) == (8, 10), f"expected 8x10, got {x_size}x{y_size}"
        assert clipped[0] == -10.0, f"expected origin -10.0, got {clipped[0]}"

    def test_window_across_the_seam_is_refused(self):
        """A window spanning the mosaic's longitude seam raises.

        Test scenario:
            A signed window over the prime meridian becomes 350..10 in 0..360 — west
            past east. Sorting those edges would resolve it into the complement, the
            same trap the antimeridian guard exists for.
        """
        with pytest.raises(ValueError, match="crosses the seam"):
            merge_mod._restrict_grid(
                self.WRAPPED_GRID, 360, 180, "EPSG:4326", (-10.0, -5.0, 10.0, 5.0), None
            )

    def test_projected_mosaic_is_untouched(self):
        """Longitude rewriting never applies to a projected CRS.

        Test scenario:
            Easting values can legitimately be negative or exceed 180; treating them
            as longitudes would corrupt every projected window.
        """
        _, x_size, y_size = merge_mod._restrict_grid(
            (0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
            6,
            4,
            "EPSG:3857",
            (1.0, 1.0, 4.0, 3.0),
            None,
        )
        assert (x_size, y_size) == (3, 2), f"expected 3x2, got {x_size}x{y_size}"


class TestReduceWindowPrunesSources:
    """The reduction path skips sources the window does not touch."""

    def test_narrow_window_does_not_warp_every_source(self, tmp_path, monkeypatch):
        """A narrow window warps only the sources it overlaps.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: Used to count `gdal.Warp` calls.

        Test scenario:
            A strip spans the windowed grid's width, but the per-source prune tested
            latitude only. On a wide east-west mosaic every source shares the same
            latitude band, so all of them were warped per strip however narrow the
            window — the exact cost the window exists to avoid.
        """
        paths = []
        for index in range(6):
            path = tmp_path / f"tile{index}.tif"
            write_raster(
                path, np.full((4, 4), float(index), dtype="float32"), (index * 4, 4)
            )
            paths.append(str(path))

        calls = []
        real_warp = gdal.Warp

        def counting_warp(*args, **kwargs):
            calls.append(kwargs.get("outputBounds"))
            return real_warp(*args, **kwargs)

        monkeypatch.setattr(merge_mod.gdal, "Warp", counting_warp)
        merge_rasters(
            paths, tmp_path / "win.tif", method="max", bbox=(1.0, 1.0, 3.0, 3.0)
        )
        assert len(calls) == 1, (
            f"a window inside one tile should warp one source, got {len(calls)} warps"
        )

    def test_pruning_does_not_drop_needed_sources(self, tmp_path):
        """A window spanning several tiles still reduces all of them.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            An east/west prune that is too eager would silently drop contributing
            sources, leaving no-data where real values belong.
        """
        for index in range(3):
            write_raster(
                tmp_path / f"t{index}.tif",
                np.full((4, 4), float(index + 1), dtype="float32"),
                (index * 4, 4),
            )
        out = tmp_path / "spanning.tif"
        merge_rasters(
            sorted(str(p) for p in tmp_path.glob("t*.tif")),
            out,
            method="max",
            bbox=(0.0, 0.0, 12.0, 4.0),
        )
        values = Dataset.read_file(str(out)).read_array()
        assert np.nanmax(values) == 3.0, (
            f"the easternmost tile must still contribute, got max {np.nanmax(values)}"
        )
