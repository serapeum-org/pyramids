"""Tests for GCP/RPC georeferencing (the Georef engine, issue-driven GR-* tasks).

Fixtures are synthetic and offline: a small in-memory raster with four corner
ground-control points, generated via ``set_gcps`` rather than shipping a binary.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset
from pyramids.dataset._gcp import GroundControlPoint
from pyramids.dataset.engines import georef as georef_module
from pyramids.dataset.engines.georef import Georef

pytestmark = pytest.mark.core


def _rpc_coeff(term_index: int) -> str:
    """A 20-term RPC coefficient string with a single 1 at ``term_index``."""
    coeffs = ["0"] * 20
    coeffs[term_index] = "1"
    return " ".join(coeffs)


# A near-identity RPC: sample tracks longitude (term L = index 1), line tracks
# latitude (term P = index 2), both denominators are the constant 1. Maps the
# 8x8 image onto roughly [10, 11]E x [49, 50]N at HEIGHT_OFF elevation.
RPC_SAMPLE: dict[str, str] = {
    "HEIGHT_OFF": "100",
    "HEIGHT_SCALE": "50",
    "LAT_OFF": "49.5",
    "LAT_SCALE": "0.5",
    "LONG_OFF": "10.5",
    "LONG_SCALE": "0.5",
    "LINE_OFF": "4",
    "LINE_SCALE": "4",
    "SAMP_OFF": "4",
    "SAMP_SCALE": "4",
    "SAMP_NUM_COEFF": _rpc_coeff(1),
    "SAMP_DEN_COEFF": _rpc_coeff(0),
    "LINE_NUM_COEFF": _rpc_coeff(2),
    "LINE_DEN_COEFF": _rpc_coeff(0),
}


@pytest.fixture
def corner_gcps() -> list[GroundControlPoint]:
    """Four corner control points of an 8x8 raster in EPSG:4326.

    Returns:
        list[GroundControlPoint]: top-left, top-right, bottom-left, bottom-right.
    """
    return [
        GroundControlPoint(row=0, col=0, x=10.0, y=50.0, id="tl"),
        GroundControlPoint(row=0, col=8, x=11.0, y=50.0, id="tr"),
        GroundControlPoint(row=8, col=0, x=10.0, y=49.0, id="bl"),
        GroundControlPoint(row=8, col=8, x=11.0, y=49.0, id="br"),
    ]


@pytest.fixture
def writable_dataset() -> Dataset:
    """A writable in-memory 8x8 float32 dataset.

    Returns:
        Dataset: MEM-backed (always writable), no GCPs yet.
    """
    return Dataset.create_from_array(
        np.ones((8, 8), dtype="float32"), top_left_corner=(0.0, 8.0), cell_size=1.0
    )


class TestGroundControlPoint:
    """Tests for the GroundControlPoint value object."""

    def test_to_gdal_maps_pixel_and_map_coords(self):
        """`to_gdal` puts col/row on pixel/line and keeps the map coordinate.

        Test scenario:
            A point at (col=7, row=3) -> (x=1, y=2) becomes a gdal.GCP with the
            same pixel/line and X/Y.
        """
        g = GroundControlPoint(row=3.0, col=7.0, x=1.0, y=2.0).to_gdal()
        assert (g.GCPPixel, g.GCPLine, g.GCPX, g.GCPY) == (7.0, 3.0, 1.0, 2.0)

    def test_round_trip_through_gdal(self):
        """`from_gdal(to_gdal())` preserves all fields.

        Test scenario:
            A fully-populated point survives the GDAL round-trip.
        """
        original = GroundControlPoint(
            row=4.0, col=2.0, x=11.5, y=46.2, z=3.0, id="p1", info="note"
        )
        back = GroundControlPoint.from_gdal(original.to_gdal())
        assert back == original

    def test_empty_id_info_become_none(self):
        """Empty GDAL Id/Info come back as None, not empty strings.

        Test scenario:
            A point with no id/info round-trips to None id/info.
        """
        back = GroundControlPoint.from_gdal(
            GroundControlPoint(row=0, col=0, x=0.0, y=0.0).to_gdal()
        )
        assert back.id is None and back.info is None


class TestReadGCPs:
    """Tests for the GCP read properties (gcps / gcp_count / gcp_projection / has_gcps)."""

    def test_plain_raster_has_no_gcps(self, writable_dataset):
        """A raster with no GCPs reports zero / empty / None.

        Test scenario:
            Fresh dataset: gcp_count 0, gcps [], gcp_projection None, has_gcps False.
        """
        assert writable_dataset.gcp_count == 0
        assert writable_dataset.gcps == []
        assert writable_dataset.gcp_projection is None
        assert writable_dataset.has_gcps is False

    def test_reads_attached_gcps(self, writable_dataset, corner_gcps):
        """After set_gcps the read properties return the points and CRS.

        Test scenario:
            4 corner points attached -> count 4, has_gcps True, projection mentions
            4326, and the first point's pixel/map coords match the input.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        assert writable_dataset.gcp_count == 4
        assert writable_dataset.has_gcps is True
        assert "4326" in writable_dataset.gcp_projection
        first = writable_dataset.gcps[0]
        assert (first.col, first.row, first.x, first.y) == (0.0, 0.0, 10.0, 50.0)

    def test_round_trip_preserves_points(self, writable_dataset, corner_gcps):
        """The read-back points equal the value objects passed to set_gcps.

        Test scenario:
            set_gcps then gcps returns equal GroundControlPoint records.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        assert writable_dataset.gcps == corner_gcps


class TestReadRPC:
    """Tests for the RPC read properties (rpcs / has_rpcs)."""

    def test_plain_raster_has_no_rpcs(self, writable_dataset):
        """A raster without RPC metadata reports None / False.

        Test scenario:
            Fresh dataset: rpcs is None, has_rpcs is False.
        """
        assert writable_dataset.rpcs is None
        assert writable_dataset.has_rpcs is False

    def test_reads_rpc_metadata(self, writable_dataset):
        """rpcs returns the RPC domain set on the underlying raster.

        Test scenario:
            After SetMetadata(RPC_SAMPLE, "RPC"), rpcs["HEIGHT_OFF"] matches and
            has_rpcs is True.
        """
        writable_dataset.raster.SetMetadata(RPC_SAMPLE, "RPC")
        assert writable_dataset.has_rpcs is True
        assert writable_dataset.rpcs["HEIGHT_OFF"] == "100"


class TestSetRPC:
    """Tests for Georef.set_rpcs."""

    def test_round_trip(self, writable_dataset):
        """set_rpcs writes the RPC domain so rpcs reads it back.

        Test scenario:
            A complete RPC dict set then read returns equal values.
        """
        writable_dataset.set_rpcs(RPC_SAMPLE)
        assert writable_dataset.rpcs["HEIGHT_OFF"] == "100"
        assert writable_dataset.rpcs["LINE_DEN_COEFF"].split()[0] == "1"

    def test_stringifies_numeric_values(self, writable_dataset):
        """Numeric RPC values are stringified before writing.

        Test scenario:
            HEIGHT_OFF given as a float comes back as its string form.
        """
        rpc = dict(RPC_SAMPLE)
        rpc["HEIGHT_OFF"] = 123.5
        writable_dataset.set_rpcs(rpc)
        assert writable_dataset.rpcs["HEIGHT_OFF"] == "123.5"

    def test_missing_keys_raise(self, writable_dataset):
        """A dict missing required keys is rejected, listing them.

        Test scenario:
            Dropping HEIGHT_OFF raises ValueError naming the missing key.
        """
        rpc = dict(RPC_SAMPLE)
        del rpc["HEIGHT_OFF"]
        with pytest.raises(ValueError, match="HEIGHT_OFF"):
            writable_dataset.set_rpcs(rpc)

    def test_read_only_raises(self, tmp_path):
        """A read-only dataset rejects set_rpcs.

        Test scenario:
            read_only=True raises ReadOnlyError.
        """
        path = tmp_path / "plain.tif"
        Dataset.create_from_array(
            np.ones((4, 4), dtype="float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
        ).to_file(str(path))
        ds = Dataset.read_file(str(path), read_only=True)
        with pytest.raises(ReadOnlyError):
            ds.set_rpcs(RPC_SAMPLE)


class TestOrthorectify:
    """Tests for Georef.orthorectify (warp from RPCs)."""

    @pytest.fixture
    def rpc_dataset(self) -> Dataset:
        """An 8x8 raster carrying the near-identity RPC sensor model."""
        ds = Dataset.create_from_array(
            np.arange(64).reshape(8, 8).astype("float32"),
            top_left_corner=(0.0, 8.0),
            cell_size=1.0,
        )
        ds.raster.SetMetadata(RPC_SAMPLE, "RPC")
        return ds

    def test_no_rpc_raises(self, writable_dataset):
        """orthorectify without RPC metadata is rejected.

        Test scenario:
            A dataset with no RPC raises ValueError mentioning RPC.
        """
        with pytest.raises(ValueError, match="no RPC"):
            writable_dataset.orthorectify(rpc_height=100)

    def test_constant_height_produces_map_grid(self, rpc_dataset):
        """A constant-height ortho yields a finite, map-projected raster.

        Test scenario:
            rpc_height at HEIGHT_OFF -> non-empty raster with a real EPSG and a
            finite bbox bracketing the RPC extent.
        """
        out = rpc_dataset.orthorectify(rpc_height=100)
        assert out.columns > 0 and out.rows > 0
        assert out.epsg == 4326
        xmin, ymin, xmax, ymax = out.bbox
        assert all(np.isfinite([xmin, ymin, xmax, ymax]))

    def test_no_dem_no_height_warns(self, rpc_dataset, caplog):
        """Omitting both dem and rpc_height logs a height-0 warning.

        Test scenario:
            orthorectify() with no elevation emits a WARNING and still returns.
        """
        import logging as _logging

        with caplog.at_level(_logging.WARNING):
            out = rpc_dataset.orthorectify()
        assert out.columns > 0
        assert any("height 0" in record.message for record in caplog.records)

    def test_resolve_dem_path_none_str_path(self):
        """_resolve_dem_path passes None / str / Path through to a path string.

        Test scenario:
            None -> None; a str and a Path both become the str path. (The full
            RPC_DEM warp is GDAL-version-fragile with synthetic coefficients, so
            the DEM *handling* is unit-tested here rather than warped.)
        """
        assert Georef._resolve_dem_path(None) is None
        assert Georef._resolve_dem_path("dem.tif") == "dem.tif"
        assert Georef._resolve_dem_path(Path("a/dem.tif")) == str(Path("a/dem.tif"))

    def test_resolve_dem_path_file_backed_dataset(self, tmp_path):
        """A file-backed DEM Dataset resolves to its on-disk path.

        Test scenario:
            A Dataset read from disk yields a path ending in the file name.
        """
        dem_path = tmp_path / "dem.tif"
        Dataset.create_from_array(
            np.ones((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
        ).to_file(str(dem_path))
        resolved = Georef._resolve_dem_path(Dataset.read_file(str(dem_path)))
        assert resolved.endswith("dem.tif")

    def test_resolve_dem_path_mem_dataset_staged_to_vsimem(self):
        """An in-memory DEM Dataset is staged to a /vsimem/ path.

        Test scenario:
            A MEM-backed Dataset (no on-disk description) resolves to /vsimem/.
        """
        dem = Dataset.create_from_array(
            np.ones((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
        )
        assert Georef._resolve_dem_path(dem).startswith("/vsimem/")


class TestWarpFromGCPs:
    """Tests for Georef.georeference (warp from GCPs)."""

    def test_no_gcps_raises(self, writable_dataset):
        """georeference without GCPs is rejected.

        Test scenario:
            A dataset with no GCPs raises ValueError mentioning GCPs.
        """
        with pytest.raises(ValueError, match="no GCPs"):
            writable_dataset.georeference()

    def test_polynomial_warp_into_gcp_crs(self, writable_dataset, corner_gcps):
        """Default warp lands in the GCP CRS and brackets the GCP extent.

        Test scenario:
            4 corner GCPs (10-11E, 49-50N) -> epsg 4326 and bbox covers that box.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        out = writable_dataset.georeference()
        assert out.epsg == 4326
        xmin, ymin, xmax, ymax = out.bbox
        assert xmin <= 10.0 + 1e-6 and xmax >= 11.0 - 1e-6
        assert ymin <= 49.0 + 1e-6 and ymax >= 50.0 - 1e-6

    def test_tps_transform(self, writable_dataset, corner_gcps):
        """A thin-plate-spline warp also produces a 4326 raster.

        Test scenario:
            transform="tps" warps into the GCP CRS without error.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        out = writable_dataset.georeference(transform="tps")
        assert out.epsg == 4326

    def test_reproject_in_same_pass(self, writable_dataset, corner_gcps):
        """to_epsg reprojects the georeferenced result in one pass.

        Test scenario:
            georeference(to_epsg=3857) yields a Web-Mercator raster.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        out = writable_dataset.georeference(to_epsg=3857)
        assert out.epsg == 3857

    def test_lazy_equals_eager(self, writable_dataset, corner_gcps):
        """A lazy VRT view reads the same pixels as the eager result.

        Test scenario:
            lazy=True and lazy=False produce allclose arrays.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        eager = writable_dataset.georeference(lazy=False)
        lazy = writable_dataset.georeference(lazy=True)
        assert np.allclose(
            np.asarray(eager.read_array()), np.asarray(lazy.read_array())
        )

    def test_invalid_transform_raises(self, writable_dataset, corner_gcps):
        """An unsupported transform name is rejected.

        Test scenario:
            transform="bogus" raises ValueError.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        with pytest.raises(ValueError, match="polynomial.*tps|transform must"):
            writable_dataset.georeference(transform="bogus")

    def test_invalid_order_raises(self, writable_dataset, corner_gcps):
        """A polynomial order outside 1-3 is rejected.

        Test scenario:
            order=7 raises ValueError.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        with pytest.raises(ValueError, match="order must"):
            writable_dataset.georeference(order=7)


class TestSetGCPs:
    """Tests for Georef.set_gcps (and the Dataset facade)."""

    def test_attaches_gcps_and_projection(self, writable_dataset, corner_gcps):
        """set_gcps writes the points and an EPSG:4326 projection to the raster.

        Test scenario:
            After set_gcps the underlying GDAL dataset reports 4 GCPs and a
            4326 projection.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        raster = writable_dataset.raster
        assert raster.GetGCPCount() == 4
        assert "4326" in raster.GetGCPProjection()

    def test_empty_list_raises_value_error(self, writable_dataset):
        """An empty GCP list is rejected.

        Test scenario:
            set_gcps([], 4326) raises ValueError.
        """
        with pytest.raises(ValueError, match="at least one"):
            writable_dataset.set_gcps([], 4326)

    def test_read_only_raises(self, corner_gcps, tmp_path):
        """A read-only dataset rejects set_gcps.

        Test scenario:
            A dataset opened read_only=True raises ReadOnlyError.
        """
        path = tmp_path / "plain.tif"
        Dataset.create_from_array(
            np.ones((8, 8), dtype="float32"), top_left_corner=(0.0, 8.0), cell_size=1.0
        ).to_file(str(path))
        ds = Dataset.read_file(str(path), read_only=True)
        with pytest.raises(ReadOnlyError):
            ds.set_gcps(corner_gcps, 4326)



class TestStagedDemLifetime:
    """ARC-8: a DEM staged into /vsimem by orthorectify is freed exactly once.

    The warp itself is stubbed out. A real RPC warp against a DEM is
    GDAL-version-fragile with the synthetic coefficients this module uses (the
    same reason `_resolve_dem_path` is unit-tested rather than warped), and the
    fix under test is the cleanup around the warp, not the warp.
    """

    @staticmethod
    def _vsimem_dems() -> set[str]:
        """The staged-DEM entries currently live in /vsimem.

        Returns:
            set[str]: full `/vsimem/` paths of every staged orthorectify DEM.
        """
        entries = gdal.ReadDir("/vsimem/") or []
        return {
            f"/vsimem/{name}"
            for name in entries
            if name.startswith("orthorectify_dem_")
        }

    @pytest.fixture
    def rpc_dataset(self) -> Dataset:
        """An 8x8 raster carrying the near-identity RPC sensor model."""
        ds = Dataset.create_from_array(
            np.arange(64).reshape(8, 8).astype("float32"),
            top_left_corner=(0.0, 8.0),
            cell_size=1.0,
        )
        ds.raster.SetMetadata(RPC_SAMPLE, "RPC")
        return ds

    @pytest.fixture
    def mem_dem(self) -> Dataset:
        """A MEM-backed DEM, which orthorectify has to stage to /vsimem."""
        return Dataset.create_from_array(
            np.full((8, 8), 100.0, "float32"),
            top_left_corner=(0.0, 8.0),
            cell_size=1.0,
        )

    @staticmethod
    def _stub_warp(monkeypatch, outcome):
        """Replace `warp_to_dataset` with `outcome`, a callable of no arguments.

        Args:
            monkeypatch: pytest's monkeypatch fixture.
            outcome: Called in place of the warp; its return value (or the
                exception it raises) becomes the warp's.
        """
        monkeypatch.setattr(
            georef_module, "warp_to_dataset", lambda *args, **kwargs: outcome()
        )

    def test_the_eager_path_frees_the_staged_dem(
        self, rpc_dataset, mem_dem, monkeypatch
    ):
        """A materialised result no longer references the DEM, so it is unlinked.

        Test scenario:
            The staged copy used to survive for the lifetime of the process. An
            eager warp has read everything it needs by the time orthorectify
            returns, so nothing should be left behind.
        """
        before = self._vsimem_dems()
        self._stub_warp(
            monkeypatch,
            lambda: Dataset.create_from_array(
                np.zeros((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
            ),
        )
        rpc_dataset.orthorectify(dem=mem_dem)
        assert self._vsimem_dems() == before, (
            "the eager path must unlink the DEM it staged; leftover: "
            f"{self._vsimem_dems() - before}"
        )

    def test_the_error_path_frees_the_staged_dem(
        self, rpc_dataset, mem_dem, monkeypatch
    ):
        """A failing warp unlinks the staged DEM instead of leaking it.

        Test scenario:
            The failure paths returned without unlinking. Checks both that the
            original error reaches the caller -- a VSI error from the cleanup
            must not replace it -- and that no staged DEM is left behind.
        """
        before = self._vsimem_dems()

        def explode():
            raise RuntimeError("synthetic warp failure")

        self._stub_warp(monkeypatch, explode)
        with pytest.raises(RuntimeError, match="synthetic warp failure"):
            rpc_dataset.orthorectify(dem=mem_dem)
        assert self._vsimem_dems() == before, (
            "the error path must unlink the DEM it staged; leftover: "
            f"{self._vsimem_dems() - before}"
        )

    def test_a_second_failure_does_not_replace_the_original_error(
        self, rpc_dataset, mem_dem, monkeypatch
    ):
        """Cleanup uses silent_unlink, so a VSI error cannot mask the warp error.

        Test scenario:
            The old handler called raw gdal.Unlink on the error path. Under
            UseExceptions an Unlink of an already-gone path raises, and that new
            exception replaced the propagating one -- the caller saw a VSI error
            instead of the warp failure. Unlinks the staged DEM out from under
            the handler to force exactly that race.
        """

        def explode():
            for path in self._vsimem_dems():
                gdal.Unlink(path)
            raise RuntimeError("synthetic warp failure")

        self._stub_warp(monkeypatch, explode)
        with pytest.raises(RuntimeError, match="synthetic warp failure"):
            rpc_dataset.orthorectify(dem=mem_dem)

    def test_the_lazy_path_keeps_the_dem_while_the_handle_lives(
        self, rpc_dataset, mem_dem, monkeypatch
    ):
        """A VRT result reads the DEM on every access, so it must survive.

        Test scenario:
            Unlinking eagerly on the lazy path would leave the VRT pointing at a
            path that no longer exists. The finalizer is keyed on the GDAL
            handle rather than the pyramids wrapper, so a derived view that
            outlives the wrapper keeps the DEM alive too.
        """
        before = self._vsimem_dems()
        self._stub_warp(
            monkeypatch,
            lambda: Dataset.create_from_array(
                np.zeros((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
            ),
        )
        view = rpc_dataset.orthorectify(dem=mem_dem, lazy=True)
        staged = self._vsimem_dems() - before
        assert len(staged) == 1, f"expected one staged DEM, got {staged}"
        handle = view.raster
        del view
        gc.collect()
        assert staged <= self._vsimem_dems(), (
            "the DEM must outlive the wrapper while the GDAL handle is alive"
        )
        assert handle.RasterXSize > 0, "the pinned handle must still be readable"
        del handle
        gc.collect()
        assert self._vsimem_dems() == before, (
            "once the GDAL handle is collected the DEM must be unlinked; "
            f"leftover: {self._vsimem_dems() - before}"
        )

    def test_an_unresolvable_crs_frees_the_staged_dem(self, rpc_dataset, mem_dem):
        """A bad `to_epsg` is caught before the DEM is staged.

        Test scenario:
            The CRS lookup used to sit between the staging and the block that
            cleans it up, so the most likely failure of the method -- an
            ordinary typo in the target CRS -- stranded the staged DEM in
            /vsimem for the lifetime of the process. No warp stub here: the
            call must fail before it reaches one.
        """
        before = self._vsimem_dems()
        with pytest.raises(Exception, match="(?i)crs|not.*interpret"):
            rpc_dataset.orthorectify(dem=mem_dem, to_epsg="definitely-not-a-crs")
        assert self._vsimem_dems() == before, (
            "a CRS failure must not strand a staged DEM; leftover: "
            f"{self._vsimem_dems() - before}"
        )

    def test_an_unknown_resampling_frees_the_staged_dem(self, rpc_dataset, mem_dem):
        """A bad `method` is caught before the DEM is staged, too.

        Test scenario:
            The resampling lookup sat in the same uncovered window as the CRS
            one.
        """
        before = self._vsimem_dems()
        with pytest.raises(Exception):
            rpc_dataset.orthorectify(dem=mem_dem, method="not-a-resampling")
        assert self._vsimem_dems() == before, (
            "a resampling failure must not strand a staged DEM; leftover: "
            f"{self._vsimem_dems() - before}"
        )

    def test_a_caller_supplied_dem_is_never_unlinked(self, rpc_dataset, tmp_path):
        """Only a DEM this call staged is ours to free.

        Test scenario:
            The cleanup keys on the /vsimem prefix _resolve_dem_path writes. A
            caller's own file must survive a failed orthorectify untouched.
        """
        dem_path = tmp_path / "caller_dem.tif"
        Dataset.create_from_array(
            np.full((8, 8), 100.0, "float32"),
            top_left_corner=(0.0, 8.0),
            cell_size=1.0,
        ).to_file(str(dem_path))
        with pytest.raises(Exception):
            rpc_dataset.orthorectify(dem=str(dem_path), to_epsg="not-a-crs")
        assert dem_path.exists(), "a caller-supplied DEM must never be removed"
