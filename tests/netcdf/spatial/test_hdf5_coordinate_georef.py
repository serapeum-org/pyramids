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
from pyramids.netcdf.netcdf import _X_DIM_NAMES

pytestmark = pytest.mark.core

CELL = 0.25
N_LAT, N_LON = 8, 12
# Cell-centre coordinates -> the affine anchors half a cell outside them.
LAT_FIRST, LON_FIRST = -89.875, -179.875
EXPECTED_GT = (-180.0, CELL, 0.0, LAT_FIRST + (N_LAT - 1) * CELL + CELL / 2, 0.0, -CELL)


def _write_grid(
    path: str,
    lat_name: str,
    lon_name: str,
    with_bounds: bool = False,
    with_aliases: bool = False,
) -> str:
    """Write a netCDF-4 grid whose coordinates carry CF axis attributes.

    Args:
        path: Output ``.nc`` path.
        lat_name: Name to give the latitude coordinate (and its dimension).
        lon_name: Name to give the longitude coordinate (and its dimension).
        with_bounds: Also write ``<name>_bnds`` arrays, created *before* the coordinates so they
            enumerate first, and carrying the same CF attributes real files give them.
        with_aliases: Also write legacy-spelled ``lat``/``lon`` copies over the same dimensions,
            so a file carries both spellings and the candidate ordering is observable.

    Returns:
        str: ``path``, for chaining.
    """
    ds = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path)
    rg = ds.GetRootGroup()
    d_lat = rg.CreateDimension(lat_name, "", "", N_LAT)
    d_lon = rg.CreateDimension(lon_name, "", "", N_LON)
    bounds = []
    if with_bounds:
        d_nv = rg.CreateDimension("nv", "", "", 2)
        for name, dim, size in ((lat_name, d_lat, N_LAT), (lon_name, d_lon, N_LON)):
            arr = rg.CreateMDArray(
                f"{name}_bnds",
                [dim, d_nv],
                gdal.ExtendedDataType.Create(gdal.GDT_Float64),
            )
            arr.Write(np.zeros((size, 2)))
            bounds.append(arr)
    lat = rg.CreateMDArray(
        lat_name, [d_lat], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lat.Write(LAT_FIRST + CELL * np.arange(N_LAT))
    lon = rg.CreateMDArray(
        lon_name, [d_lon], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lon.Write(LON_FIRST + CELL * np.arange(N_LON))
    tagged = [
        (lat, "latitude", "Y", "degrees_north"),
        (lon, "longitude", "X", "degrees_east"),
    ]
    if bounds:
        # Real files tag bounds with the parent's units/standard_name, which is exactly why they
        # are indistinguishable from the coordinate by attributes alone.
        tagged += [
            (bounds[0], "latitude", "Y", "degrees_north"),
            (bounds[1], "longitude", "X", "degrees_east"),
        ]
    for arr, std, axis, units in tagged:
        for key, value in (("standard_name", std), ("axis", axis), ("units", units)):
            attr = arr.CreateAttribute(key, [], gdal.ExtendedDataType.CreateString())
            attr.Write(value)
    aliases = []
    if with_aliases:
        for alias, dim, values in (
            ("lat", d_lat, LAT_FIRST + CELL * np.arange(N_LAT)),
            ("lon", d_lon, LON_FIRST + CELL * np.arange(N_LON)),
        ):
            arr = rg.CreateMDArray(
                alias, [dim], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
            )
            arr.Write(values)
            aliases.append(arr)
    var = rg.CreateMDArray(
        "v", [d_lat, d_lon], gdal.ExtendedDataType.Create(gdal.GDT_Float32)
    )
    var.Write(np.arange(N_LAT * N_LON, dtype="float32").reshape(N_LAT, N_LON))
    tagged = bounds = aliases = None
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
            The lookup is first-match, so ordering is behaviour. On a file carrying *both*
            spellings the legacy one must win, so a grid that resolved before the #1071 change
            resolves to exactly the same coordinate after it. Asserting the literal tuple
            `("lon", "x")` instead would only restate the code, and would keep passing if the
            stage that consumes the order broke.
        """
        path = _write_grid(
            str(tmp_path / "order.nc"), "latitude", "longitude", with_aliases=True
        )
        nc = NetCDF.read_file(path)
        try:
            _, x_name = nc._first_coordinate(nc._coordinate_candidates("X"))
            _, y_name = nc._first_coordinate(nc._coordinate_candidates("Y"))
            assert (x_name, y_name) == ("lon", "lat"), (
                f"the legacy spelling must win when both are present, got {x_name}/{y_name}"
            )
        finally:
            nc.close()

    def test_cf_attributes_alone_identify_a_coordinate(self, tmp_path):
        """A coordinate under an unrecognised name is found from its CF attributes.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Deliberately uses names in neither the legacy pair nor `_X_DIM_NAMES`/`_Y_DIM_NAMES`,
            so the only stage that can supply them is the `axis`/`standard_name` detection. Asking
            for `latitude`/`longitude` instead would prove nothing: the well-known tail offers
            those unconditionally, whether or not the file contains them.
        """
        path = _write_grid(str(tmp_path / "cf_only.nc"), "ordinate", "abscissa")
        nc = NetCDF.read_file(path)
        try:
            assert "abscissa" not in _X_DIM_NAMES, (
                "the fixture name must not be well-known"
            )
            assert "abscissa" in nc._coordinate_candidates("X"), (
                "the CF-attribute stage must offer a coordinate under any name"
            )
            assert "ordinate" in nc._coordinate_candidates("Y")
            cube = _index_space_cube(["ordinate", "abscissa"])
            assert nc._coordinate_derived_geotransform(cube) == pytest.approx(
                EXPECTED_GT
            ), "a CF-attribute-only coordinate must still yield the affine"
        finally:
            nc.close()

    def test_bounds_variables_do_not_defeat_the_rescue(self, tmp_path):
        """A CF bounds array is not mistaken for the axis coordinate.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            CF says a bounds variable inherits its parent's `units` and `standard_name`, so
            `latitude_bnds` is indistinguishable from `latitude` by attributes alone and enters the
            candidate list. Written here *before* the real coordinates so it enumerates first: the
            lookup must scan past the 2-D array rather than commit to it, or the affine comes back
            `None` and the variable stays in index space — the very #1071 symptom, on an ordinary
            CF layout.
        """
        path = _write_grid(
            str(tmp_path / "bounds.nc"), "latitude", "longitude", with_bounds=True
        )
        nc = NetCDF.read_file(path)
        try:
            _, x_name = nc._first_coordinate(nc._coordinate_candidates("X"))
            assert x_name == "longitude", f"expected the 1-D coordinate, got {x_name!r}"
            cube = _index_space_cube(["latitude", "longitude"])
            assert nc._coordinate_derived_geotransform(cube) == pytest.approx(
                EXPECTED_GT
            ), "bounds variables must not defeat the affine rescue"
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
