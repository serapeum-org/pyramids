"""`to_xarray()` then `from_xarray()` must come back holding the same arrays.

The export promotes CF `bounds`, auxiliary-coordinate, cell-measure and
grid-mapping arrays out of `data_vars` and into `coords`, which is where xarray
expects them. The writer behind `from_xarray` only ever wrote the coordinates
that are also *dimensions*, so a promoted non-dimension coordinate landed in
neither mapping and was dropped without a word -- the arrays the export was
widened to preserve were the exact ones the promotion then lost.

For a curvilinear ROMS store the pair it lost (`lat_rho` / `lon_rho`) is the
only georeferencing the file has, so the round trip returned a cube that could
no longer be placed on the earth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"
BOUNDED = "cf__7v__1d3-2d3-3d1__y-asc.nc"
CURVILINEAR = "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc"
CURVILINEAR_PLAIN = "none__4v__1d1-2d2-3d1__curv.nc"
GEOS = "cf__9v__1d7-2d2__geos__y-desc.nc"


def store_array_names(path: str) -> set[str]:
    """Every MDArray name a netCDF file holds, read straight from GDAL.

    Args:
        path: Path to the `.nc` file.

    Returns:
        set[str]: The root group's MDArray names.
    """
    dataset = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
    return set(dataset.GetRootGroup().GetMDArrayNames() or [])


def round_trip(fixture: str, destination: Path) -> tuple[set[str], set[str]]:
    """Export a fixture through xarray and write it back, reporting both name sets.

    Args:
        fixture: File name of the netCDF fixture under `tests/data/netcdf`.
        destination: Path the round-tripped file is written to.

    Returns:
        tuple[set[str], set[str]]: `(source array names, round-tripped array names)`.
    """
    source = str(DATA / fixture)
    exported = NetCDF.read_file(source).to_xarray()
    NetCDF.from_xarray(exported, path=str(destination))
    return store_array_names(source), store_array_names(str(destination))


class TestARoundTripLosesNoArray:
    """Whatever the export moves into `coords`, the writer still has to write."""

    @pytest.mark.parametrize(
        ("fixture", "promoted"),
        [
            (BOUNDED, ("lat_bnds", "lon_bnds", "time_bnds")),
            (CURVILINEAR, ("Cs_r", "h", "lat_rho", "lon_rho")),
            (CURVILINEAR_PLAIN, ("xc", "yc")),
            (GEOS, ("band_id", "band_wavelength")),
        ],
    )
    def test_the_promoted_arrays_survive_the_write(self, fixture, promoted, tmp_path):
        """The arrays the CF promotion moves are the ones that used to vanish.

        Args:
            fixture: The netCDF fixture to round-trip.
            promoted: Array names the export promotes into `coords`.
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            Each of these is a real array in the source store and a
            non-dimension coordinate in the export, which is exactly the
            combination the writer filtered out.
        """
        source_names, written_names = round_trip(fixture, tmp_path / "round_trip.nc")

        assert set(promoted) <= source_names, "the fixture no longer holds these"
        assert set(promoted) <= written_names, (
            f"lost in the round trip: {sorted(set(promoted) - written_names)}"
        )

    @pytest.mark.parametrize(
        "fixture", [BOUNDED, CURVILINEAR, CURVILINEAR_PLAIN, GEOS], ids=lambda f: f
    )
    def test_nothing_at_all_is_lost(self, fixture, tmp_path):
        """The general form: the round trip is not allowed to drop anything.

        Args:
            fixture: The netCDF fixture to round-trip.
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            Naming the arrays one by one catches today's loss; asserting the
            whole set catches the next one.
        """
        source_names, written_names = round_trip(fixture, tmp_path / "round_trip.nc")

        assert source_names - written_names == set()


class TestTheCurvilinearGeoreferencingSurvives:
    """A ROMS store's only georeferencing is its 2-D `lat_rho` / `lon_rho` pair."""

    def test_the_values_come_back_unchanged(self, tmp_path):
        """Present is not enough -- the coordinates have to still be the coordinates.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            A writer that carried the names but wrote a placeholder, or that
            transposed the 2-D grid, would satisfy a name-only assertion and
            still put every cell in the wrong place.
        """
        source = str(DATA / CURVILINEAR)
        destination = tmp_path / "roms.nc"
        NetCDF.from_xarray(NetCDF.read_file(source).to_xarray(), path=str(destination))

        original = NetCDF.read_file(source)
        written = NetCDF.read_file(str(destination))
        for name in ("lat_rho", "lon_rho"):
            expected = np.asarray(original.get_variable(name).read_array())
            actual = np.asarray(written.get_variable(name).read_array())
            np.testing.assert_allclose(actual, expected, err_msg=name)

    def test_the_dimension_coordinates_are_still_dimensions(self, tmp_path):
        """Carrying the aux coordinates must not demote the real axes.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            `s_rho` and `ocean_time` index the cube. Writing them as ordinary
            arrays instead of indexing variables would leave the file without
            a usable vertical or time axis.
        """
        destination = tmp_path / "roms.nc"
        NetCDF.from_xarray(
            NetCDF.read_file(str(DATA / CURVILINEAR)).to_xarray(),
            path=str(destination),
        )

        dataset = gdal.OpenEx(str(destination), gdal.OF_MULTIDIM_RASTER)
        indexed = {
            dimension.GetName()
            for dimension in dataset.GetRootGroup().GetDimensions()
            if dimension.GetIndexingVariable() is not None
        }

        assert {"s_rho", "ocean_time"} <= indexed
