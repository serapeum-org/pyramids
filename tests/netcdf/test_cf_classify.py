"""Tests for CF variable classification, axis detection, conventions parsing,
cell methods parsing, and valid range masking (CF-5, CF-6, CF-7, CF-10, CF-11).
"""

import numpy as np
import pytest

from pyramids.netcdf.cf import (
    apply_valid_range_mask,
    classify_variables,
    detect_axis,
    parse_cell_methods,
    parse_conventions,
)
from pyramids.netcdf.netcdf import NetCDF
from pyramids.netcdf.utils import create_time_conversion_func

pytestmark = pytest.mark.core

GEO = (30.0, 0.5, 0, 35.0, 0, -0.5)
SEED = 42


class TestDetectAxis:
    """Tests for cf.detect_axis."""

    def test_explicit_axis_attribute(self):
        """Explicit axis='X' takes highest priority."""
        result = detect_axis("foo", {"axis": "X"})
        assert result == "X", f"Expected X, got {result}"

    def test_standard_name_latitude(self):
        """standard_name=latitude returns Y."""
        result = detect_axis("foo", {"standard_name": "latitude"})
        assert result == "Y", f"Expected Y, got {result}"

    def test_standard_name_time(self):
        """standard_name=time returns T."""
        result = detect_axis("foo", {"standard_name": "time"})
        assert result == "T", f"Expected T, got {result}"

    def test_units_degrees_north(self):
        """units=degrees_north returns Y."""
        result = detect_axis("foo", {"units": "degrees_north"})
        assert result == "Y", f"Expected Y, got {result}"

    def test_units_since(self):
        """units containing 'since' returns T."""
        result = detect_axis("foo", {"units": "days since 1970-01-01"})
        assert result == "T", f"Expected T, got {result}"

    def test_name_pattern_lat(self):
        """Name 'lat' returns Y via name pattern."""
        result = detect_axis("lat", {})
        assert result == "Y", f"Expected Y, got {result}"

    def test_name_pattern_time(self):
        """Name 'time' returns T via name pattern."""
        result = detect_axis("time", {})
        assert result == "T", f"Expected T, got {result}"

    def test_unknown_returns_none(self):
        """Unknown name and no attrs returns None."""
        result = detect_axis("ensemble", {})
        assert result is None, f"Expected None, got {result}"


class TestClassifyVariables:
    """Tests for cf.classify_variables."""

    def _make_mock_var(self, attrs):
        """Create a simple object with .attributes."""

        class MockVar:
            def __init__(self, a):
                self.attributes = a
                self.name = ""
                self.full_name = ""

        return MockVar(attrs)

    def _make_mock_dim(self, name):
        class MockDim:
            def __init__(self, n):
                self.name = n
                self.full_name = f"/{n}"

        return MockDim(name)

    def test_coordinate_by_dimension_name(self):
        """Variable matching a dimension name is 'coordinate'."""
        dims = {"x": self._make_mock_dim("x")}
        vars_ = {"x": self._make_mock_var({})}
        roles = classify_variables(vars_, dims)
        assert roles["x"] == "coordinate", f"Expected coordinate, got {roles['x']}"

    def test_grid_mapping(self):
        """Variable with grid_mapping_name is 'grid_mapping'."""
        dims = {}
        vars_ = {
            "crs": self._make_mock_var({"grid_mapping_name": "transverse_mercator"})
        }
        roles = classify_variables(vars_, dims)
        assert (
            roles["crs"] == "grid_mapping"
        ), f"Expected grid_mapping, got {roles['crs']}"

    def test_bounds(self):
        """Variable referenced by bounds attribute is 'bounds'."""
        dims = {"time": self._make_mock_dim("time")}
        vars_ = {
            "time": self._make_mock_var({"bounds": "time_bnds"}),
            "time_bnds": self._make_mock_var({}),
            "temp": self._make_mock_var({}),
        }
        roles = classify_variables(vars_, dims)
        assert (
            roles["time_bnds"] == "bounds"
        ), f"Expected bounds, got {roles['time_bnds']}"
        assert roles["temp"] == "data", f"Expected data, got {roles['temp']}"

    def test_data_default(self):
        """Variables not matching any role are 'data'."""
        dims = {"x": self._make_mock_dim("x")}
        vars_ = {
            "x": self._make_mock_var({}),
            "temperature": self._make_mock_var({}),
        }
        roles = classify_variables(vars_, dims)
        assert (
            roles["temperature"] == "data"
        ), f"Expected data, got {roles['temperature']}"

    def test_mesh_topology(self):
        """Variable with cf_role=mesh_topology is 'mesh_topology'."""
        dims = {}
        vars_ = {"mesh2d": self._make_mock_var({"cf_role": "mesh_topology"})}
        roles = classify_variables(vars_, dims)
        assert (
            roles["mesh2d"] == "mesh_topology"
        ), f"Expected mesh_topology, got {roles['mesh2d']}"

    def test_connectivity(self):
        """Variable with cf_role containing connectivity."""
        dims = {}
        vars_ = {
            "face_nodes": self._make_mock_var({"cf_role": "face_node_connectivity"})
        }
        roles = classify_variables(vars_, dims)
        assert (
            roles["face_nodes"] == "connectivity"
        ), f"Expected connectivity, got {roles['face_nodes']}"


class TestParseConventions:
    """Tests for cf.parse_conventions."""

    def test_single_convention(self):
        """Parse 'CF-1.8'."""
        result = parse_conventions("CF-1.8")
        assert result == {"CF": "1.8"}, f"Expected {{'CF': '1.8'}}, got {result}"

    def test_multiple_conventions(self):
        """Parse 'CF-1.8 UGRID-1.0 Deltares-0.10'."""
        result = parse_conventions("CF-1.8 UGRID-1.0 Deltares-0.10")
        assert result["CF"] == "1.8", f"CF version: {result.get('CF')}"
        assert result["UGRID"] == "1.0", f"UGRID version: {result.get('UGRID')}"
        assert (
            result["Deltares"] == "0.10"
        ), f"Deltares version: {result.get('Deltares')}"

    def test_none_returns_empty(self):
        """None input returns empty dict."""
        result = parse_conventions(None)
        assert result == {}, f"Expected empty dict, got {result}"

    def test_empty_string_returns_empty(self):
        """Empty string returns empty dict."""
        result = parse_conventions("")
        assert result == {}, f"Expected empty dict, got {result}"


class TestParseCellMethods:
    """Tests for cf.parse_cell_methods."""

    def test_simple_mean(self):
        """Parse 'time: mean'."""
        result = parse_cell_methods("time: mean")
        assert len(result) == 1, f"Expected 1 entry, got {len(result)}"
        assert result[0]["dimensions"] == "time", f"Got {result[0]['dimensions']}"
        assert result[0]["method"] == "mean", f"Got {result[0]['method']}"

    def test_multiple_methods(self):
        """Parse 'time: mean area: sum'."""
        result = parse_cell_methods("time: mean area: sum")
        assert len(result) == 2, f"Expected 2 entries, got {len(result)}"
        assert result[0]["method"] == "mean", f"Got {result[0]['method']}"
        assert result[1]["method"] == "sum", f"Got {result[1]['method']}"

    def test_where_clause(self):
        """Parse 'area: mean where sea_ice'."""
        result = parse_cell_methods("area: mean where sea_ice")
        assert result[0]["where"] == "sea_ice", f"Got {result[0].get('where')}"


class TestApplyValidRangeMask:
    """Tests for cf.apply_valid_range_mask."""

    def test_valid_min(self):
        """Values below valid_min are replaced with NaN."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_valid_range_mask(arr, valid_min=2.5)
        assert np.isnan(result[0]), "1.0 should be masked"
        assert np.isnan(result[1]), "2.0 should be masked"
        assert result[2] == 3.0, "3.0 should be preserved"

    def test_valid_max(self):
        """Values above valid_max are replaced with NaN."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_valid_range_mask(arr, valid_max=3.5)
        assert result[2] == 3.0, "3.0 should be preserved"
        assert np.isnan(result[3]), "4.0 should be masked"
        assert np.isnan(result[4]), "5.0 should be masked"

    def test_valid_range(self):
        """valid_range sets both min and max."""
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = apply_valid_range_mask(arr, valid_range=[1.0, 3.0])
        assert np.isnan(result[0]), "0.0 should be masked"
        assert result[1] == 1.0, "1.0 should be preserved"
        assert result[3] == 3.0, "3.0 should be preserved"
        assert np.isnan(result[4]), "4.0 should be masked"

    def test_custom_fill_value(self):
        """Custom fill_value replaces out-of-range values."""
        arr = np.array([1.0, 5.0])
        result = apply_valid_range_mask(arr, valid_max=3.0, fill_value=-9999.0)
        assert result[1] == -9999.0, f"Expected -9999.0, got {result[1]}"

    def test_no_masking(self):
        """No valid_min/max/range means no changes."""
        arr = np.array([1.0, 2.0, 3.0])
        result = apply_valid_range_mask(arr)
        np.testing.assert_array_equal(result, arr)


class TestCalendarSupport:
    """Tests for non-Gregorian calendar in create_time_conversion_func (CF-6)."""

    def test_standard_calendar_unchanged(self):
        """Standard calendar should produce same results as before."""
        func = create_time_conversion_func("days since 1979-01-01", calendar="standard")
        result = func(0)
        assert result == "1979-01-01 00:00:00", f"Expected 1979-01-01, got {result}"

    def test_gregorian_alias(self):
        """'gregorian' alias should work like standard."""
        func = create_time_conversion_func(
            "days since 2000-01-01", calendar="gregorian"
        )
        result = func(1)
        assert "2000-01-02" in result, f"Expected 2000-01-02, got {result}"

    def test_non_standard_without_cftime_raises(self):
        """Non-standard calendar without cftime raises ImportError.

        Test scenario:
            If cftime is not installed, requesting 360_day should fail
            with a helpful message. We skip this test if cftime IS
            installed (since the import would succeed).
        """
        try:
            import cftime  # noqa: F401

            pytest.skip("cftime is installed, cannot test ImportError")
        except ImportError:
            with pytest.raises(ImportError, match="cftime"):
                create_time_conversion_func("days since 2000-01-01", calendar="360_day")

    def test_360_day_calendar(self):
        """360_day calendar: 30 days per month."""
        cftime = pytest.importorskip("cftime")
        func = create_time_conversion_func(
            "days since 2000-01-01", out_format="%Y-%m-%d", calendar="360_day"
        )
        result = func(30)
        assert result == "2000-02-01", f"Expected 2000-02-01, got {result}"

    def test_noleap_calendar(self):
        """noleap calendar: no Feb 29."""
        cftime = pytest.importorskip("cftime")
        func = create_time_conversion_func(
            "days since 2000-01-01", out_format="%Y-%m-%d", calendar="noleap"
        )
        result_59 = func(59)
        assert result_59 == "2000-03-01", f"Day 59 should be Mar 1, got {result_59}"


class TestCFInfoOnMetadata:
    """Tests for CFInfo on NetCDFMetadata (CF-8)."""

    def test_meta_data_has_cf(self):
        """NetCDFMetadata.cf is not None after reading."""
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO, variable_name="temp")
        md = nc.meta_data
        assert md.cf is not None, "cf should be populated"
        assert md.cf.cf_version == "1.8", f"Expected CF 1.8, got {md.cf.cf_version}"

    def test_cf_classifications(self):
        """CFInfo.classifications contains correct roles."""
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO, variable_name="temp")
        md = nc.meta_data
        assert (
            "temp" in md.cf.data_variable_names
        ), f"temp should be in data_variable_names: {md.cf.data_variable_names}"

    def test_cf_conventions_parsed(self):
        """CFInfo.conventions contains parsed Conventions attribute."""
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO, variable_name="temp")
        md = nc.meta_data
        assert (
            "CF" in md.cf.conventions
        ), f"CF should be in conventions: {md.cf.conventions}"
