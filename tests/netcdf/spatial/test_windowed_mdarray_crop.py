"""The rectilinear crop reads only the mask's window from the MDArray (#1071).

The affine crop reads its whole source before clipping. That is cheap locally and expensive over a
remote store: a variable is backed by an `AsClassicDataset` view, and when the file stores its axes
in a different order than the raster presents them (e.g. `(time, longitude, latitude)` shown as
rows=latitude/cols=longitude) every raster row becomes a strided gather. Measured against a 14 GB
`/vsicurl` NetCDF-4, the same 16x10x10 window cost ~7.5 s through the classic view and ~0.85 s
through `MDArray.Read`, and the end-to-end crop fell from 33.6 s to 1.1 s.

The shortcut must be invisible: these tests pin that a crop taking it produces exactly what the
full-read path produces, and that the window read matches the full array cell for cell — including
the Y flip, since the fixtures store latitude ascending and the raster is north-up.
"""

from __future__ import annotations

import gc

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal, osr
from shapely.geometry import box

import pyramids.netcdf.engines.selection as selection_module
from pyramids.base.crs import crs_equal
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

CELL = 0.25
N_LAT, N_LON = 8, 12
LAT_FIRST, LON_FIRST = -89.875, -179.875


@pytest.fixture
def grid_path(tmp_path) -> str:
    """A small netCDF-4 grid with CF coordinates and a distinct value per cell.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written file.
    """
    path = str(tmp_path / "grid.nc")
    ds = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path)
    rg = ds.GetRootGroup()
    d_lat = rg.CreateDimension("latitude", "", "", N_LAT)
    d_lon = rg.CreateDimension("longitude", "", "", N_LON)
    lat = rg.CreateMDArray(
        "latitude", [d_lat], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lat.Write(LAT_FIRST + CELL * np.arange(N_LAT))
    lon = rg.CreateMDArray(
        "longitude", [d_lon], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lon.Write(LON_FIRST + CELL * np.arange(N_LON))
    for arr, std, axis, units in (
        (lat, "latitude", "Y", "degrees_north"),
        (lon, "longitude", "X", "degrees_east"),
    ):
        for key, value in (("standard_name", std), ("axis", axis), ("units", units)):
            attr = arr.CreateAttribute(key, [], gdal.ExtendedDataType.CreateString())
            attr.Write(value)
    var = rg.CreateMDArray(
        "v", [d_lat, d_lon], gdal.ExtendedDataType.Create(gdal.GDT_Float32)
    )
    var.Write(np.arange(N_LAT * N_LON, dtype="float32").reshape(N_LAT, N_LON))
    lat = lon = var = d_lat = d_lon = rg = None
    ds.Close()
    del ds
    gc.collect()
    return path


N_TIME = 3


@pytest.fixture
def cube_path(tmp_path) -> str:
    """A `(time, latitude, longitude)` grid with longitude stored east-to-west.

    Exercises the two shapes a 2-D ascending-latitude grid cannot: a non-spatial dimension that
    the window read must pin to index 0 and flatten into bands, and a reversed X axis.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written file.
    """
    path = str(tmp_path / "cube.nc")
    ds = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path)
    rg = ds.GetRootGroup()
    d_time = rg.CreateDimension("time", "", "", N_TIME)
    d_lat = rg.CreateDimension("latitude", "", "", N_LAT)
    d_lon = rg.CreateDimension("longitude", "", "", N_LON)
    time = rg.CreateMDArray(
        "time", [d_time], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    time.Write(np.arange(N_TIME, dtype="float64"))
    lat = rg.CreateMDArray(
        "latitude", [d_lat], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lat.Write(LAT_FIRST + CELL * np.arange(N_LAT))
    lon = rg.CreateMDArray(
        "longitude", [d_lon], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lon.Write((LON_FIRST + CELL * np.arange(N_LON))[::-1])
    for arr, std, axis, units in (
        (lat, "latitude", "Y", "degrees_north"),
        (lon, "longitude", "X", "degrees_east"),
    ):
        for key, value in (("standard_name", std), ("axis", axis), ("units", units)):
            attr = arr.CreateAttribute(key, [], gdal.ExtendedDataType.CreateString())
            attr.Write(value)
    var = rg.CreateMDArray(
        "v", [d_time, d_lat, d_lon], gdal.ExtendedDataType.Create(gdal.GDT_Float32)
    )
    var.Write(
        np.arange(N_TIME * N_LAT * N_LON, dtype="float32").reshape(N_TIME, N_LAT, N_LON)
    )
    time = lat = lon = var = d_time = d_lat = d_lon = rg = None
    ds.Close()
    del ds
    gc.collect()
    return path


N_LEVEL = 3


@pytest.fixture
def hypercube_path(tmp_path) -> str:
    """A `(time, level, latitude, longitude)` grid — two non-spatial dimensions.

    Three dimensions only ever collapse one axis into bands, which cannot tell a correct
    flattening from one that transposes the two. Here `time` x `level` must flatten to
    `N_TIME * N_LEVEL` bands in storage order, time slowest.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written file.
    """
    path = str(tmp_path / "hypercube.nc")
    ds = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path)
    rg = ds.GetRootGroup()
    d_time = rg.CreateDimension("time", "", "", N_TIME)
    d_level = rg.CreateDimension("level", "", "", N_LEVEL)
    d_lat = rg.CreateDimension("latitude", "", "", N_LAT)
    d_lon = rg.CreateDimension("longitude", "", "", N_LON)
    time = rg.CreateMDArray(
        "time", [d_time], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    time.Write(np.arange(N_TIME, dtype="float64"))
    level = rg.CreateMDArray(
        "level", [d_level], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    level.Write(np.arange(N_LEVEL, dtype="float64"))
    lat = rg.CreateMDArray(
        "latitude", [d_lat], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lat.Write(LAT_FIRST + CELL * np.arange(N_LAT))
    lon = rg.CreateMDArray(
        "longitude", [d_lon], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lon.Write(LON_FIRST + CELL * np.arange(N_LON))
    for arr, std, axis, units in (
        (lat, "latitude", "Y", "degrees_north"),
        (lon, "longitude", "X", "degrees_east"),
    ):
        for key, value in (("standard_name", std), ("axis", axis), ("units", units)):
            attr = arr.CreateAttribute(key, [], gdal.ExtendedDataType.CreateString())
            attr.Write(value)
    var = rg.CreateMDArray(
        "v",
        [d_time, d_level, d_lat, d_lon],
        gdal.ExtendedDataType.Create(gdal.GDT_Float32),
    )
    var.Write(
        np.arange(N_TIME * N_LEVEL * N_LAT * N_LON, dtype="float32").reshape(
            N_TIME, N_LEVEL, N_LAT, N_LON
        )
    )
    time = level = lat = lon = var = None
    d_time = d_level = d_lat = d_lon = rg = None
    ds.Close()
    del ds
    gc.collect()
    return path


def _as_bands(array) -> np.ndarray:
    """Normalise a read to `(bands, rows, cols)` so 2-D and 3-D reads compare alike."""
    values = np.asarray(array)
    return values[None, ...] if values.ndim == 2 else values


class TestWindowViaMdArray:
    """`_window_via_mdarray` returns exactly the cells the full read holds."""

    @pytest.mark.parametrize(
        "x_off, y_off, x_size, y_size",
        [(0, 0, N_LON, N_LAT), (2, 1, 5, 4), (7, 3, 4, 3), (0, 5, 3, 3)],
        ids=["whole", "interior", "corner", "edge"],
    )
    def test_window_matches_the_full_read(
        self, grid_path, x_off, y_off, x_size, y_size
    ):
        """A window equals the same slice of the full array.

        Args:
            grid_path: The written grid fixture.
            x_off: Column offset of the window.
            y_off: Row offset of the window.
            x_size: Window width.
            y_size: Window height.

        Test scenario:
            The fixture stores latitude ascending, so the raster is Y-flipped; a window read must
            undo that flip to line up with the north-up full read. An off-by-one or a transposed
            axis shows up as a mismatch rather than merely wrong-looking numbers.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        full = _as_bands(var.read_array())
        window = var._window_via_mdarray(x_off, y_off, x_size, y_size)
        assert window is not None, "the window read must be served"
        got = _as_bands(window.ReadAsArray())
        expected = full[:, y_off : y_off + y_size, x_off : x_off + x_size]
        assert np.array_equal(got, expected), (
            f"window ({x_off},{y_off},{x_size},{y_size}) does not match the full read"
        )

    def test_window_prefers_the_wrapper_srs(self, grid_path):
        """A CRS already on the wrapper is carried onto the window verbatim.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            The EPSG fallback exists only for a view that carries no SRS of its own. When the
            wrapper does have one — the VRT `_georeference_index_subset` installs, say — that exact
            SRS must win, or a crop against a cutline would clip the wrong region. A deliberately
            different code (3035, against the variable's 4326) is used so the assertion tells the
            two branches apart; an `AsClassicDataset` view silently ignores `SetSpatialRef`, so the
            wrapper is stood in for by a raster that does keep one.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        stand_in = gdal.GetDriverByName("MEM").Create(
            "", N_LON, N_LAT, 1, gdal.GDT_Float32
        )
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3035)
        stand_in.SetSpatialRef(srs)
        var._raster = stand_in
        window = var._window_via_mdarray(1, 1, 4, 4)
        assert window is not None
        assert window.GetSpatialRef() is not None, "the window must carry a CRS"
        assert window.GetSpatialRef().GetAuthorityCode(None) == "3035", (
            "the window must carry the wrapper's own SRS, not the EPSG fallback"
        )

    def test_window_carries_the_sub_affine(self, grid_path):
        """The window's geotransform is the parent's, shifted to the window origin.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            The clip that follows relies on this affine, so a wrong origin would silently move
            the cropped region.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        parent = var.geotransform
        window = var._window_via_mdarray(3, 2, 4, 4)
        assert window is not None
        got = window.GetGeoTransform()
        assert got[0] == pytest.approx(parent[0] + 3 * parent[1])
        assert got[3] == pytest.approx(parent[3] + 2 * parent[5])
        assert got[1] == pytest.approx(parent[1])
        assert got[5] == pytest.approx(parent[5])

    @pytest.mark.parametrize(
        "x_off, y_off, x_size, y_size",
        [
            (0, 0, N_LON + 1, N_LAT),
            (0, 0, N_LON, N_LAT + 1),
            (-1, 0, 2, 2),
            (0, 0, 0, 2),
        ],
        ids=["too-wide", "too-tall", "negative-offset", "empty"],
    )
    def test_out_of_range_window_declines(
        self, grid_path, x_off, y_off, x_size, y_size
    ):
        """A window the array cannot serve returns None rather than a partial read.

        Args:
            grid_path: The written grid fixture.
            x_off: Column offset of the window.
            y_off: Row offset of the window.
            x_size: Window width.
            y_size: Window height.

        Test scenario:
            The caller treats `None` as "use the full-read path", so declining must be explicit
            rather than a truncated or wrapped read.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        assert var._window_via_mdarray(x_off, y_off, x_size, y_size) is None


class TestWindowViaMdArrayMultiBand:
    """The window read over a `(time, y, x)` cube with a reversed X axis."""

    @pytest.mark.parametrize(
        "x_off, y_off, x_size, y_size",
        [(0, 0, N_LON, N_LAT), (3, 2, 4, 4), (8, 0, 4, 3)],
        ids=["whole", "interior", "edge"],
    )
    def test_window_matches_the_full_read(
        self, cube_path, x_off, y_off, x_size, y_size
    ):
        """A window of a 3-D cube equals the same slice of the full read.

        Args:
            cube_path: The written cube fixture.
            x_off: Column offset of the window in raster space.
            y_off: Row offset of the window in raster space.
            x_size: Window width.
            y_size: Window height.

        Test scenario:
            Both axes are reversed here — latitude ascending and longitude east-to-west — and the
            time dimension must be pinned to index 0 and flattened into bands in storage order. A
            band read in the wrong order, or an axis un-reversed on the wrong side, shows up as a
            mismatch against the classic view's own full read.
        """
        var = NetCDF.read_file(cube_path).get_variable("v")
        full = _as_bands(var.read_array())
        window = var._window_via_mdarray(x_off, y_off, x_size, y_size)
        assert window is not None, "the window read must be served"
        got = _as_bands(window.ReadAsArray())
        expected = full[:, y_off : y_off + y_size, x_off : x_off + x_size]
        assert got.shape == expected.shape, (
            f"expected shape {expected.shape}, got {got.shape}"
        )
        assert np.array_equal(got, expected), (
            f"window ({x_off},{y_off},{x_size},{y_size}) does not match the full read"
        )

    @pytest.mark.parametrize(
        "x_off, y_off, x_size, y_size",
        [(0, 0, N_LON, N_LAT), (2, 1, 4, 3)],
        ids=["whole", "interior"],
    )
    def test_two_non_spatial_dimensions_keep_band_order(
        self, hypercube_path, x_off, y_off, x_size, y_size
    ):
        """A `(time, level, y, x)` variable flattens to bands in the classic view's own order.

        Args:
            hypercube_path: The written 4-D fixture.
            x_off: Column offset of the window in raster space.
            y_off: Row offset of the window in raster space.
            x_size: Window width.
            y_size: Window height.

        Test scenario:
            With a single non-spatial axis, a transposed flattening is indistinguishable from a
            correct one — both yield the same band sequence. Two non-spatial axes separate them:
            `time` must vary slowest and `level` fastest, exactly as `AsClassicDataset` orders
            them, or band `k` of the window holds a different slice than band `k` of a full read.
        """
        var = NetCDF.read_file(hypercube_path).get_variable("v")
        full = _as_bands(var.read_array())
        window = var._window_via_mdarray(x_off, y_off, x_size, y_size)
        assert window is not None, "the window read must be served"
        got = _as_bands(window.ReadAsArray())
        assert got.shape[0] == N_TIME * N_LEVEL, (
            f"expected {N_TIME * N_LEVEL} bands, got {got.shape[0]}"
        )
        expected = full[:, y_off : y_off + y_size, x_off : x_off + x_size]
        assert np.array_equal(got, expected), (
            "band order or window placement differs from the classic view for a 4-D variable"
        )

    def test_window_keeps_every_band(self, cube_path):
        """The window carries one band per non-spatial slice, not just the first.

        Args:
            cube_path: The written cube fixture.

        Test scenario:
            Pinning the time dimension to index 0 must select the whole axis for the band stack;
            reading only one slice would silently drop the other time steps.
        """
        var = NetCDF.read_file(cube_path).get_variable("v")
        window = var._window_via_mdarray(2, 2, 3, 3)
        assert window is not None
        assert window.RasterCount == N_TIME, (
            f"expected {N_TIME} bands, got {window.RasterCount}"
        )


class TestWindowViaMdArrayDeclines:
    """Every guard that makes the window read return `None` rather than a wrong raster."""

    def test_missing_root_group_declines(self, grid_path):
        """A variable whose multidim references were dropped cannot be windowed.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            `_materialize_md_view` nulls `_gdal_rg_ref`; the shortcut must then decline instead of
            dereferencing `None`.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        var._gdal_rg_ref = None
        assert var._window_via_mdarray(0, 0, 4, 4) is None

    def test_unopenable_array_declines(self, grid_path, monkeypatch):
        """A failing `OpenMDArray` falls back rather than propagating.

        Args:
            grid_path: The written grid fixture.
            monkeypatch: Used to make the open raise.

        Test scenario:
            The caller treats `None` as "use the full-read path", so a GDAL error here must not
            escape to the user.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")

        def _raise(_name):
            raise RuntimeError("cannot open")

        monkeypatch.setattr(var._gdal_rg_ref, "OpenMDArray", _raise)
        assert var._window_via_mdarray(0, 0, 4, 4) is None

    @pytest.mark.parametrize(
        "outcome", ["raise", "none"], ids=["read-raises", "read-returns-none"]
    )
    def test_failed_read_declines(self, grid_path, monkeypatch, outcome):
        """A read that fails or yields nothing declines instead of building an empty raster.

        Args:
            grid_path: The written grid fixture.
            monkeypatch: Used to break the array read.
            outcome: Whether the read raises or returns `None`.

        Test scenario:
            Both failure shapes must reach the same fallback; a `None` block reaching NumPy would
            raise far from the cause.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        real_open = var._gdal_rg_ref.OpenMDArray

        def _open(name):
            array = real_open(name)

            def _read(*args, **kwargs):
                if outcome == "raise":
                    raise RuntimeError("read failed")
                return None

            monkeypatch.setattr(array, "ReadAsArray", _read)
            return array

        monkeypatch.setattr(var._gdal_rg_ref, "OpenMDArray", _open)
        assert var._window_via_mdarray(0, 0, 4, 4) is None


class TestCropUsesTheWindow:
    """The crop shortcut is invisible: same result as reading the whole variable."""

    @pytest.mark.parametrize(
        "bounds",
        [
            (-179.6, -89.6, -178.9, -89.1),
            (-179.9, -89.9, -179.2, -89.4),
            (-178.0, -88.6, -177.2, -88.1),
        ],
        ids=["interior", "south-west-corner", "north-east"],
    )
    def test_crop_matches_the_full_read_path(self, grid_path, monkeypatch, bounds):
        """Cropping through the window equals cropping the whole variable.

        Args:
            grid_path: The written grid fixture.
            monkeypatch: Used to disable the shortcut for the reference crop.
            bounds: The mask box to crop with.

        Test scenario:
            Shape, geotransform and every cell must agree — the shortcut is an optimisation, so
            any difference at all is a defect. The run is asserted to have *taken* the shortcut:
            without that, a change to the window/variable size threshold would silently compare
            the full-read path against itself and the test would pass while covering nothing.
        """
        mask = gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:4326")
        real_source = selection_module.Selection._mask_window_source
        taken: list[bool] = []

        def _spy(self, mask):
            source = real_source(self, mask)
            taken.append(source is not None)
            return source

        monkeypatch.setattr(selection_module.Selection, "_mask_window_source", _spy)
        fast = NetCDF.read_file(grid_path).get_variable("v").crop(mask=mask, touch=True)
        assert taken, "the crop never consulted the window shortcut at all"
        assert any(taken), (
            "the crop did not take the window shortcut, so this comparison proves nothing"
        )
        monkeypatch.setattr(
            selection_module.Selection, "_mask_window_source", lambda self, mask: None
        )
        reference = (
            NetCDF.read_file(grid_path).get_variable("v").crop(mask=mask, touch=True)
        )
        assert (fast.rows, fast.columns) == (reference.rows, reference.columns), (
            f"shape differs: {(fast.rows, fast.columns)} vs "
            f"{(reference.rows, reference.columns)}"
        )
        assert np.allclose(fast.geotransform, reference.geotransform), (
            f"geotransform differs: {fast.geotransform} vs {reference.geotransform}"
        )
        assert np.array_equal(
            _as_bands(fast.read_array()), _as_bands(reference.read_array())
        ), "cropped values differ between the windowed and full-read paths"

    def test_shortcut_declines_for_a_mask_in_another_crs(self, grid_path):
        """A cutline in a different CRS falls back, so the warp reprojects it.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            The window is computed in the raster's own coordinates, so adopting a mask stated in
            another CRS would window the wrong region. Covers the `FeatureCollection` mask type;
            the bare-`GeoDataFrame` case is covered separately, since the two reach the guard by
            different attributes.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        mask = FeatureCollection(
            geometry=[box(-19970000, -19970000, -19900000, -19900000)],
            crs="EPSG:3857",
        )
        assert int(mask.epsg) != int(var.epsg), "the mask must be in a different CRS"
        assert var.selection._mask_window_source(mask) is None

    def test_bare_geodataframe_in_another_crs_declines(self, grid_path):
        """A plain `GeoDataFrame` in a different CRS declines — it has no `.epsg` to read.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            `crop(mask=...)` accepts a bare `GeoDataFrame` and passes it through unchanged, so
            this is the *common* mask type, not an edge case. An `epsg`-based guard is skipped
            entirely for it, and the mask's unreprojected coordinates then get divided through the
            raster's affine — a plausible window over the wrong part of the grid, returned as data
            rather than an error.

            The mask's coordinates are deliberately chosen to fall *inside* the grid's numeric
            range while being stated in another CRS. A far-away box would decline for an unrelated
            reason — an out-of-range window — and the test would pass without ever reaching the
            CRS guard.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        mask = gpd.GeoDataFrame(
            geometry=[box(-179.6, -89.6, -178.9, -89.1)], crs="EPSG:3857"
        )
        assert not hasattr(mask, "epsg"), (
            "a bare GeoDataFrame must have no .epsg — that is the point of this test"
        )
        assert var.selection._mask_window_source(mask) is None

    def test_bare_geodataframe_in_the_same_crs_still_uses_the_shortcut(self, grid_path):
        """Tightening the CRS guard must not disable the optimisation for ordinary masks.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            The guard declines on *unknown or differing* CRS. A bare `GeoDataFrame` carrying the
            raster's own CRS is neither, so it must still take the fast path — otherwise the fix
            for the fail-open would have quietly removed the feature.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        mask = gpd.GeoDataFrame(
            geometry=[box(-179.6, -89.6, -178.9, -89.1)], crs="EPSG:4326"
        )
        assert var.selection._mask_window_source(mask) is not None

    def test_empty_mask_declines_instead_of_raising(self, grid_path):
        """An empty mask keeps the full-read path's error, not a numeric conversion error.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            An empty or all-null-geometry frame has `total_bounds == [nan] * 4`, and
            `math.floor(nan)` raises `ValueError: cannot convert float NaN to integer`. Turning a
            clear domain error into that, from an optimisation the caller never asked for, is a
            regression in the error contract.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        assert var.selection._mask_window_source(empty) is None
        with pytest.raises(RuntimeError, match="cutline"):
            var.crop(mask=empty, touch=True)

    @pytest.mark.parametrize("epsg, expected", [(4326, True), (3857, False)])
    def test_raster_mask_crs_is_read_from_a_wkt_string(self, grid_path, epsg, expected):
        """A `Dataset` mask states its CRS as WKT, not as a pyproj object.

        Args:
            grid_path: The written grid fixture.
            epsg: CRS to stamp on the raster mask.
            expected: Whether the shortcut should be offered for it.

        Test scenario:
            `crop(mask=...)` also accepts a raster, whose `crs` is a plain WKT string — a guard
            written for `GeoDataFrame`'s pyproj `crs` raises `AttributeError` on it. Matching CRS
            must be offered the window and a differing one declined, without either raising.

            Deliberately exercises `_mask_window_source` rather than `crop()`: cropping a NetCDF
            variable with a raster mask is broken independently of this shortcut (it raises
            `TypeError: create_from_array() got an unexpected keyword argument 'geo'` on `main`
            with the shortcut disabled), so an end-to-end assertion here would be testing that
            unrelated defect.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        template = gdal.GetDriverByName("MEM").Create("", 4, 3, 1, gdal.GDT_Float32)
        template.SetGeoTransform([-179.75, CELL, 0.0, -88.25, 0.0, -CELL])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        template.SetSpatialRef(srs)
        mask = Dataset(template)
        assert isinstance(mask.crs, str), "a Dataset mask must state its CRS as WKT"
        source = var.selection._mask_window_source(mask)
        assert (source is not None) is expected, (
            f"EPSG:{epsg} mask: expected shortcut offered={expected}"
        )

    def test_crop_after_an_in_place_mutation_sees_the_mutation(
        self, grid_path, monkeypatch
    ):
        """`apply(inplace=True)` then `crop()` must return the transformed values.

        Args:
            grid_path: The written grid fixture.
            monkeypatch: Used to disable the shortcut for the reference crop.

        Test scenario:
            An in-place update swaps in a raster of the *same size* holding different values while
            keeping the multidim references, so every size and shape guard still passes. Reading
            the window from the MDArray then returns the original on-disk data and discards the
            mutation silently — wrong data from a public API, for the ordinary sequence
            read -> transform -> crop.
        """
        mask = gpd.GeoDataFrame(
            geometry=[box(-179.6, -89.6, -178.9, -89.1)], crs="EPSG:4326"
        )

        def _cropped():
            var = NetCDF.read_file(grid_path).get_variable("v")
            var.apply(lambda values: values + 1000.0, inplace=True)
            return np.asarray(var.crop(mask=mask, touch=True).read_array())

        fast = _cropped()
        monkeypatch.setattr(
            selection_module.Selection, "_mask_window_source", lambda self, mask: None
        )
        reference = _cropped()
        assert fast.shape == reference.shape, (
            f"shape differs: {fast.shape} vs {reference.shape}"
        )
        assert np.array_equal(fast, reference), (
            "the crop discarded the in-place mutation and returned the pre-mutation data"
        )

    def test_shortcut_declines_for_a_rotated_affine(self, grid_path):
        """A rotated or degenerate affine is not windowable by row/column arithmetic.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            The window is derived by dividing through `gt[1]`/`gt[5]`, which only describes an
            axis-aligned grid; a rotation term would place the window somewhere else entirely.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        gt = list(var.geotransform)
        gt[2] = 0.1
        var._geotransform = tuple(gt)
        mask = gpd.GeoDataFrame(
            geometry=[box(-179.6, -89.6, -178.9, -89.1)], crs="EPSG:4326"
        )
        assert var.selection._mask_window_source(mask) is None

    def test_shortcut_declines_for_a_mask_off_the_grid(self, grid_path):
        """A mask that misses the variable entirely yields no window.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            An empty window must decline rather than clamp to a zero-sized read, which the
            MDArray could not serve anyway.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        mask = gpd.GeoDataFrame(
            geometry=[box(120.0, 40.0, 121.0, 41.0)], crs="EPSG:4326"
        )
        assert var.selection._mask_window_source(mask) is None

    def test_shortcut_declines_for_an_unusable_mask(self, grid_path):
        """A mask with no readable bounds declines instead of raising.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            `total_bounds` is read defensively; anything that is not four numbers must reach the
            ordinary crop path rather than surface a `TypeError` from the shortcut. The stand-in
            must carry a *matching* CRS, or it declines at the CRS guard and never reaches the
            `total_bounds` unpacking this test exists to cover.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")

        class _NoBounds:
            crs = var.crs
            total_bounds = None

        assert crs_equal(_NoBounds.crs, var.crs), (
            "the stand-in must pass the CRS guard, or this test covers the wrong branch"
        )
        assert var.selection._mask_window_source(_NoBounds()) is None

    def test_shortcut_declines_when_the_window_is_most_of_the_grid(self, grid_path):
        """A mask covering the grid falls back — the shortcut must earn its second code path.

        Args:
            grid_path: The written grid fixture.

        Test scenario:
            Reading nearly everything through the window path only adds risk, so the whole-grid
            case keeps the ordinary read.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        mask = gpd.GeoDataFrame(
            geometry=[box(-180.0, -90.0, -176.0, -87.0)], crs="EPSG:4326"
        )
        assert var.selection._mask_window_source(mask) is None
