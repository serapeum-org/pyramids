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
from osgeo import gdal
from shapely.geometry import box

import pyramids.netcdf.engines.selection as selection_module
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
            any difference at all is a defect.
        """
        mask = gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:4326")
        fast = NetCDF.read_file(grid_path).get_variable("v").crop(mask=mask, touch=True)
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
            another CRS would window the wrong region.
        """
        var = NetCDF.read_file(grid_path).get_variable("v")
        mask = gpd.GeoDataFrame(
            geometry=[box(-19970000, -19970000, -19900000, -19900000)],
            crs="EPSG:3857",
        )
        assert var.selection._mask_window_source(mask) is None

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
