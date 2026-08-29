"""Regression tests for georeferencing a variable whose driver supplies no affine (#1071).

`NetCDF._georeference_index_subset` rescues a variable that came back in index space by deriving
the affine from the file's 1-D coordinate variables. It only matters when the driver that opened
the file does not build the affine itself: GDAL's **netCDF** driver does, so the rescue is a no-op
there, but the **HDF5** driver does not — and HDF5 is what serves any `/vsi` NetCDF-4 read on
Windows, because the netCDF driver needs Linux `userfaultfd` for those paths.

The bug: the rescue looked its coordinates up under `lon`/`x` and `lat`/`y` only, so a file naming
them the way CF's own standard does — `longitude` / `latitude` — was never found, and the variable
kept the index-space placeholder `(0, 1, 0, rows, 0, -1)`. Any `crop(bbox=...)` on it then read the
bbox as pixel indices.

The driver cannot be swapped per-open (`GDAL_SKIP` is read when drivers register, so setting it
from a fixture does nothing), so rather than fake the driver these tests drive
`_coordinate_derived_geotransform` directly with a cube that is already in index space — the exact
state the HDF5 path produces, and the input the rescue exists to handle.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

CELL = 0.25
N_LAT, N_LON = 8, 12
# Cell-centre coordinates -> the affine anchors half a cell outside them.
LAT_FIRST, LON_FIRST = -89.875, -179.875
EXPECTED_GT = (-180.0, CELL, 0.0, LAT_FIRST + (N_LAT - 1) * CELL + CELL / 2, 0.0, -CELL)


def _write_grid(path: str, lat_name: str, lon_name: str) -> str:
    """Write a netCDF-4 grid whose coordinates carry CF axis attributes.

    Args:
        path: Output ``.nc`` path.
        lat_name: Name to give the latitude coordinate (and its dimension).
        lon_name: Name to give the longitude coordinate (and its dimension).

    Returns:
        str: ``path``, for chaining.
    """
    ds = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path)
    rg = ds.GetRootGroup()
    d_lat = rg.CreateDimension(lat_name, "", "", N_LAT)
    d_lon = rg.CreateDimension(lon_name, "", "", N_LON)
    lat = rg.CreateMDArray(
        lat_name, [d_lat], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lat.Write(LAT_FIRST + CELL * np.arange(N_LAT))
    lon = rg.CreateMDArray(
        lon_name, [d_lon], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
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


def _index_space_cube(dim_names: list[str]) -> SimpleNamespace:
    """A stand-in variable subset carrying the index-space affine the HDF5 path produces.

    `_md_y_flipped` is ``True`` because the file stores latitude ascending (south-to-north), which
    `_read_md_array` reverses to north-up — so row 0 holds the *last* stored latitude, and the
    affine must anchor there. Getting this wrong flips the grid, not just the numbers.

    Args:
        dim_names: The variable's dimension names, as `_md_array_dims` records them.

    Returns:
        SimpleNamespace: The attributes `_coordinate_derived_geotransform` reads off a cube.
    """
    mem = gdal.GetDriverByName("MEM").Create("", N_LON, N_LAT, 1, gdal.GDT_Float32)
    mem.SetGeoTransform([0.0, 1.0, 0.0, float(N_LAT), 0.0, -1.0])
    return SimpleNamespace(
        _raster=mem,
        rows=N_LAT,
        columns=N_LON,
        _md_array_dims=dim_names,
        _md_x_flipped=False,
        _md_y_flipped=True,
        # Read only by the curvilinear fallback, which a shape mismatch drops into.
        crs="",
        _source_var_name="v",
    )


class TestCoordinateDerivedGeotransform:
    """The affine is derived whatever the coordinates are named (#1071)."""

    @pytest.mark.parametrize(
        "lat_name, lon_name",
        [("latitude", "longitude"), ("lat", "lon"), ("Latitude", "LONGITUDE")],
        ids=["cf-standard", "short", "mixed-case"],
    )
    def test_affine_is_derived_for_any_coordinate_spelling(
        self, tmp_path, lat_name, lon_name
    ):
        """An index-space cube is re-georeferenced from the container's coordinates.

        Args:
            tmp_path: pytest temp directory.
            lat_name: Latitude coordinate/dimension name written to the file.
            lon_name: Longitude coordinate/dimension name written to the file.

        Test scenario:
            Before the fix only the `lat`/`lon` spelling was looked up, so the CF-standard and
            mixed-case files returned `None` and the variable kept the index-space placeholder.
        """
        path = _write_grid(str(tmp_path / f"{lat_name}.nc"), lat_name, lon_name)
        nc = NetCDF.read_file(path)
        try:
            cube = _index_space_cube([lat_name, lon_name])
            derived = nc._coordinate_derived_geotransform(cube)
            assert derived is not None, (
                f"{lat_name}/{lon_name} coordinates must yield an affine, got None"
            )
            assert derived == pytest.approx(EXPECTED_GT), (
                f"unexpected affine for {lat_name}/{lon_name}: {derived}"
            )
        finally:
            nc.close()

    def test_already_georeferenced_cube_is_left_alone(self, tmp_path):
        """A cube whose affine already matches the coordinates is not re-derived.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            The rescue must stay a no-op on the netCDF-driver path, where GDAL already built
            the affine — otherwise every read pays for a redundant correction.
        """
        path = _write_grid(str(tmp_path / "ok.nc"), "latitude", "longitude")
        nc = NetCDF.read_file(path)
        try:
            cube = _index_space_cube(["latitude", "longitude"])
            cube._raster.SetGeoTransform(list(EXPECTED_GT))
            assert nc._coordinate_derived_geotransform(cube) is None, (
                "a correctly georeferenced cube must not be re-derived"
            )
        finally:
            nc.close()

    def test_shape_mismatch_declines(self, tmp_path):
        """Coordinates that do not index the cube's grid are not adopted.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Guards against a same-named but differently-sized axis being adopted, which would
            georeference the variable to the wrong extent.
        """
        path = _write_grid(str(tmp_path / "shape.nc"), "latitude", "longitude")
        nc = NetCDF.read_file(path)
        try:
            cube = _index_space_cube(["latitude", "longitude"])
            cube.columns = N_LON + 3
            assert nc._coordinate_derived_geotransform(cube) is None, (
                "a coordinate whose length does not match the grid must be declined"
            )
        finally:
            nc.close()


class TestCoordinateCandidates:
    """`_coordinate_candidates` ordering rules that the rescue depends on."""

    def test_legacy_names_are_tried_first(self, tmp_path):
        """`lon`/`lat` keep priority, so a file that already resolved resolves identically.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            The lookup is first-match, so ordering is behaviour. Anything that previously
            resolved via `lon`/`x` must not start resolving to a different coordinate.
        """
        path = _write_grid(str(tmp_path / "order.nc"), "latitude", "longitude")
        nc = NetCDF.read_file(path)
        try:
            assert nc._coordinate_candidates("X")[:2] == ("lon", "x"), (
                "the legacy X names must stay first"
            )
            assert nc._coordinate_candidates("Y")[:2] == ("lat", "y"), (
                "the legacy Y names must stay first"
            )
        finally:
            nc.close()

    def test_cf_named_coordinates_are_offered(self, tmp_path):
        """The file's actual CF-named coordinates appear as candidates.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            `longitude`/`latitude` are absent from the legacy pair, so they must come from the
            CF-attribute or well-known-name stage — otherwise the grid is unrescuable.
        """
        path = _write_grid(str(tmp_path / "cands.nc"), "latitude", "longitude")
        nc = NetCDF.read_file(path)
        try:
            assert "longitude" in nc._coordinate_candidates("X")
            assert "latitude" in nc._coordinate_candidates("Y")
        finally:
            nc.close()

    def test_candidates_are_unique(self, tmp_path):
        """No name is offered twice, so a miss is never retried.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            The stages overlap by construction (`lon` is both legacy and well-known), so the
            builder must de-duplicate while preserving order.
        """
        path = _write_grid(str(tmp_path / "uniq.nc"), "lat", "lon")
        nc = NetCDF.read_file(path)
        try:
            for axis in ("X", "Y"):
                names = nc._coordinate_candidates(axis)
                assert len(names) == len(set(names)), (
                    f"duplicate candidate for {axis}: {names}"
                )
        finally:
            nc.close()
