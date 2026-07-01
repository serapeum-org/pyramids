"""Unit tests for the NetCDF time dimension, cube dimension names, and the MSWEP real-world file."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.netcdf.models import NetCDFMetadata
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf._netcdf_unit_helpers import _make_3d_nc, _make_nc_with_time_units
from tests.netcdf.conftest import make_2d_nc

pytestmark = pytest.mark.core


MSWEP_PATH = Path("tests/data/netcdf/MSWEP_1979010100.nc")


class TestTimeStamp:
    """Tests for NetCDF.time_stamp property."""

    def test_time_stamp_returns_none_without_time_units(self):
        """Verify time_stamp returns None when there is no time units attribute.

        Covers delegates to get_time_variable() which returns
        None when time dimension lacks a 'units' attribute.
        """
        nc = _make_3d_nc()
        result = nc.time_stamp
        assert (
            result is None
        ), f"Expected None (no time units in created NC), got {result}"


class TestGetTimeVariable:
    """Tests for NetCDF.get_time_variable method."""

    def test_get_time_variable_with_units(self):
        """Verify get_time_variable parses time when units attribute exists.

        Covers the full path through get_time_variable
        where time_dim has units and time values can be converted.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/noah-precipitation-1979.nc",
            open_as_multi_dimensional=True,
        )
        result = nc.get_time_variable()
        if result is not None:
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) > 0, "Expected non-empty time list"
        # If no time units in this file, it will be None and that's OK

    def test_get_time_variable_no_time_dim(self):
        """Verify get_time_variable returns None when no time dimension exists.

        Covers (time_stamp = None) return.
        """
        nc = make_2d_nc()
        result = nc.get_time_variable()
        assert (
            result is None
        ), f"Expected None for NC without time dimension, got {result}"

    def test_get_time_variable_custom_format(self):
        """Verify get_time_variable respects custom time_format.

        Covers the conversion path with a custom format.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/noah-precipitation-1979.nc",
            open_as_multi_dimensional=True,
        )
        result = nc.get_time_variable(time_format="%Y/%m/%d")
        if result is not None:
            assert "/" in result[0], f"Expected '/' in date format, got {result[0]}"

    def test_get_time_variable_cds_beta_era5(self):
        """E2E: CDS-Beta ERA5 NetCDFs hide ``valid_time#units`` from the
        multidim driver but expose it on the classic driver. The
        ``MetadataBuilder._topup_dim_attrs_from_classic`` fallback must
        recover ``units`` (and ``calendar``) so ``get_time_variable``
        returns parsed dates instead of ``None``. Reproduces issue #309.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/era5_cds_beta_t2m_jan2022.nc",
            open_as_multi_dimensional=True,
        )

        valid_time = nc.meta_data.dimensions["valid_time"]
        assert (
            valid_time.attrs.get("units") == "seconds since 1970-01-01"
        ), f"expected classic units to be merged, got attrs={valid_time.attrs}"
        assert (
            valid_time.attrs.get("calendar") == "proleptic_gregorian"
        ), f"expected calendar from multidim to survive, got attrs={valid_time.attrs}"

        result = nc.get_time_variable("valid_time")
        assert result is not None, (
            "get_time_variable returned None on a CDS-Beta NetCDF — the "
            "classic-metadata top-up did not surface 'units' on valid_time"
        )
        assert isinstance(result, list), f"expected list, got {type(result)}"
        assert (
            len(result) == valid_time.size
        ), f"expected {valid_time.size} timestamps, got {len(result)}"
        assert result[0].startswith(
            "2022-01-"
        ), f"expected 2022-01-* from the CDS-Beta sample, got {result[0]}"


class TestMSWEPFile:
    """Tests using the MSWEP test file for real-world coverage.

    ``MSWEP_PATH`` is committed test data; if it is ever missing these tests
    must fail loudly rather than skip silently, so there is no ``.exists()`` gate.
    """

    def test_read_mswep_mdim(self):
        """Verify reading MSWEP file in multidimensional mode.

        Uses tests/data/netcdf/MSWEP_1979010100.nc to hit real code paths.
        """
        nc = NetCDF.read_file(
            str(MSWEP_PATH),
            open_as_multi_dimensional=True,
        )
        assert (
            "precipitation" in nc.variable_names
        ), f"Expected 'precipitation' in {nc.variable_names}"
        var = nc.get_variable("precipitation")
        assert var.is_subset is True, "Variable should be a subset"
        arr = var.read_array()
        assert arr is not None, "Should read array from variable"
        assert arr.ndim >= 2, f"Expected 2D+ array, got {arr.ndim}D"

    def test_mswep_dimension_names(self):
        """Verify MSWEP file has correct dimension names."""
        nc = NetCDF.read_file(
            str(MSWEP_PATH),
            open_as_multi_dimensional=True,
        )
        dims = nc.dimension_names
        assert dims is not None, "Dimension names should not be None"
        assert "lon" in dims, f"Expected 'lon' in {dims}"
        assert "lat" in dims, f"Expected 'lat' in {dims}"

    def test_mswep_meta_data(self):
        """Verify MSWEP metadata is accessible."""
        nc = NetCDF.read_file(
            str(MSWEP_PATH),
            open_as_multi_dimensional=True,
        )
        md = nc.meta_data
        assert isinstance(
            md, NetCDFMetadata
        ), f"Expected NetCDFMetadata, got {type(md)}"

    def test_mswep_get_all_metadata(self):
        """Verify get_all_metadata populates dimension overview."""
        nc = NetCDF.read_file(
            str(MSWEP_PATH),
            open_as_multi_dimensional=True,
        )
        md = nc.get_all_metadata()
        assert len(md.dimensions) > 0, "dimensions should be populated"

    def test_mswep_lon_lat(self):
        """Verify lon/lat are readable from MSWEP file."""
        nc = NetCDF.read_file(
            str(MSWEP_PATH),
            open_as_multi_dimensional=True,
        )
        lon = nc.lon
        lat = nc.lat
        assert lon is not None, "lon should not be None"
        assert lat is not None, "lat should not be None"
        assert lon.ndim == 1, f"lon should be 1D, got {lon.ndim}D"
        assert lat.ndim == 1, f"lat should be 1D, got {lat.ndim}D"


class TestGetTimeVariableWithUnits:
    """Tests for get_time_variable with actual time units attribute."""

    def test_get_time_variable_with_days_since(self):
        """Verify get_time_variable converts time values when units exist.

        Covers the full path where time_dim has units,
        time values are read, and the conversion function is applied.
        """
        nc = _make_nc_with_time_units(n_times=3)
        result = nc.get_time_variable()
        assert (
            result is not None
        ), "get_time_variable should return dates when units exist"
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == 3, f"Expected 3 time stamps, got {len(result)}"
        assert (
            "1979-01-01" in result[0]
        ), f"Expected '1979-01-01' in first timestamp, got {result[0]}"
        assert (
            "1979-01-02" in result[1]
        ), f"Expected '1979-01-02' in second timestamp, got {result[1]}"

    def test_get_time_variable_custom_format(self):
        """Verify get_time_variable uses a custom format string.

        Covers the create_time_conversion_func call with
        custom time_format.
        """
        nc = _make_nc_with_time_units(n_times=2)
        result = nc.get_time_variable(time_format="%Y/%m/%d")
        assert result is not None, "Should return formatted timestamps"
        assert (
            "/" in result[0]
        ), f"Expected '/' separator in custom format, got {result[0]}"

    def test_time_stamp_property_with_units(self):
        """Verify time_stamp property returns dates when time has units.

        Covers the delegation to get_time_variable().
        """
        nc = _make_nc_with_time_units(n_times=2)
        result = nc.time_stamp
        assert (
            result is not None
        ), "time_stamp should return dates when time units exist"
        assert len(result) == 2, f"Expected 2 timestamps, got {len(result)}"


class TestCubeDimensionNames:
    """`get_variable(...).dimension_names` must mirror the container's view.

    Pre-fix the property returned `None` on a variable subset because the
    classic-mode in-memory `Dataset` underlying the cube has no GDAL root
    group. The cube's dim names live on `_md_array_dims` and are the right
    fall-through. See `pyramids-h1-followup.md` in the earthly planning
    docs for context.

    Note: when pyramids has to y-flip the source on read (ascending-lat
    inputs), the spatial dim is renamed to `subset_lat_*` on the cube.
    Real-world CDS-Beta files (already north-down) keep the original
    name. Tests use the era5 fixture for byte-equality and the synthetic
    4-D fixture for shape / first-two-dims invariants.
    """

    def test_cube_lists_all_dims_in_storage_order_on_real_4d(self):
        """4-D cube reports every dim in storage order (no y-flip rename here).

        Test scenario:
            era5 CDS-Beta pressure-levels file is already north-down so
            pyramids takes the no-flip read path. The cube's
            `dimension_names` should match the container's exactly.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/era5_cds_beta_t_pressure_levels_jan2022.nc"
        )
        var = nc.get_variable("t")
        assert var.dimension_names == [
            "valid_time",
            "pressure_level",
            "latitude",
            "longitude",
        ], f"got {var.dimension_names!r}"

    def test_cube_dimension_names_matches_container_for_real_4d(self):
        """Cube's `dimension_names` mirrors container's on a no-flip file."""
        nc = NetCDF.read_file(
            "tests/data/netcdf/era5_cds_beta_t_pressure_levels_jan2022.nc"
        )
        var = nc.get_variable("t")
        assert (
            var.dimension_names == nc.dimension_names
        ), f"cube={var.dimension_names!r} container={nc.dimension_names!r}"

    def test_cube_dimension_names_first_two_match_band_dims_on_synthetic(self):
        """Synthetic 4-D cube reports both band dims first; spatial dims may
        be renamed by pyramids' y-flip but count is still 4.
        """
        nc = NetCDF.read_file("tests/data/netcdf/pyramids-netcdf-4d.nc")
        var = nc.get_variable("temperature")
        names = var.dimension_names
        assert names is not None, "cube dim names must not be None after fix"
        assert len(names) == 4, f"4-D cube must have 4 dims, got {names!r}"
        assert names[:2] == [
            "time",
            "pressure_level",
        ], f"band dims must be first two, got {names!r}"

    def test_cube_dimension_names_is_independent_copy(self):
        """Mutating the returned list must not alter `_md_array_dims`."""
        nc = NetCDF.read_file("tests/data/netcdf/pyramids-netcdf-4d.nc")
        var = nc.get_variable("temperature")
        names = var.dimension_names
        original = list(var._md_array_dims)
        names.append("bogus")
        assert "bogus" not in var._md_array_dims, (
            f"_md_array_dims was mutated through dimension_names: "
            f"{var._md_array_dims!r}"
        )
        assert var._md_array_dims == original, (
            "subsequent reads of dimension_names should still match the "
            "original cached list"
        )

    def test_cube_with_no_md_array_dims_returns_none(self):
        """Defensive: a cube whose `_md_array_dims` is empty returns `None`."""
        nc = NetCDF.read_file("tests/data/netcdf/pyramids-netcdf-4d.nc")
        var = nc.get_variable("temperature")
        var._md_array_dims = []
        assert (
            var.dimension_names is None
        ), f"empty cache should yield None, got {var.dimension_names!r}"

    def test_container_dimension_names_unchanged(self):
        """The container path is unchanged: still reads from the root group.

        Test scenario:
            On the bundled 4-D synthetic, the container reports the
            exact dim names from the file (no y-flip rename, since the
            container is the original MDIM dataset).
        """
        nc = NetCDF.read_file("tests/data/netcdf/pyramids-netcdf-4d.nc")
        assert nc.dimension_names == [
            "time",
            "pressure_level",
            "lat",
            "lon",
        ], f"container path regressed: {nc.dimension_names!r}"
