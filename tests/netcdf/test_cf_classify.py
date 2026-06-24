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

    @pytest.mark.parametrize(
        "attrs, expected",
        [
            ({"Axis": "Y"}, "Y"),
            ({"AXIS": "x"}, "X"),
            ({"Standard_Name": "latitude"}, "Y"),
            ({"Units": "degrees_east"}, "X"),
            ({"UNITS": "days since 1970-01-01"}, "T"),
        ],
        ids=["Axis", "AXIS", "Standard_Name", "Units", "UNITS"],
    )
    def test_attribute_key_matching_is_case_insensitive(self, attrs, expected):
        """Capitalized CF attribute *names* are still matched (M1 regression guard).

        Args:
            attrs: Attribute dict whose key uses non-lowercase casing.
            expected: The axis role that detection should still return.

        Test scenario:
            GDAL preserves the on-disk attribute-name casing, so a coordinate written
            with ``Axis``/``Standard_Name``/``Units`` (capitalized) must still be
            classified — ``detect_axis`` lower-cases keys before matching. Pins that the
            attribute-based detection is not silently skipped for such keys.
        """
        result = detect_axis("foo", attrs)
        assert result == expected, f"Expected {expected} for {attrs}, got {result}"

    def test_units_degrees_east(self):
        """``units=degrees_east`` returns X (longitude)."""
        result = detect_axis("foo", {"units": "degrees_east"})
        assert result == "X", f"Expected X, got {result}"

    @pytest.mark.parametrize(
        "axis_value, expected",
        [("Z", "Z"), ("T", "T"), ("z", "Z"), ("t", "T")],
        ids=["Z", "T", "z-lower", "t-lower"],
    )
    def test_explicit_axis_z_and_t(self, axis_value, expected):
        """An explicit vertical/time ``axis`` value is returned and upper-cased.

        Args:
            axis_value: The raw ``axis`` attribute value.
            expected: The normalized (upper-cased) axis role.

        Test scenario:
            ``detect_axis`` accepts the full CF axis set (X/Y/Z/T) and normalizes the
            case of the value, so ``axis="z"`` and ``axis="Z"`` both yield ``"Z"``.
        """
        result = detect_axis("foo", {"axis": axis_value})
        assert result == expected, f"Expected {expected} for axis={axis_value!r}, got {result}"

    def test_units_passed_as_separate_parameter(self):
        """The standalone ``units=`` parameter is honored when attrs omit units.

        Test scenario:
            ``detect_axis`` accepts ``units`` separately from ``attrs`` (for callers that
            hold the unit string outside the attribute dict); a ``since`` epoch there
            still classifies the coordinate as a time axis.
        """
        result = detect_axis("foo", {}, units="hours since 2000-01-01")
        assert result == "T", f"Expected T from units= param, got {result}"

    def test_name_pattern_lon(self):
        """Name 'lon' returns X via the name-pattern fallback (no attrs)."""
        result = detect_axis("lon", {})
        assert result == "X", f"Expected X, got {result}"

    @pytest.mark.parametrize(
        "attrs, expected",
        [
            ({"axis": ["X"]}, "X"),
            ({"axis": ("Y",)}, "Y"),
            ({"standard_name": ["latitude"]}, "Y"),
            ({"units": ["degrees_east"]}, "X"),
        ],
        ids=["axis-list", "axis-tuple", "standard_name-list", "units-list"],
    )
    def test_length_one_sequence_attribute_is_unwrapped(self, attrs, expected):
        """A length-1 array-valued attribute classifies like its scalar form (review L1).

        Args:
            attrs: Attribute dict whose value is a one-element list/tuple.
            expected: The axis role detection should still return.

        Test scenario:
            GDAL can store an attribute as a 1-element array (e.g. ``axis = ["X"]``), which
            ``_read_attributes`` surfaces as a length-1 list. ``detect_axis`` must unwrap it
            so attribute-based detection is not silently skipped for array-stored values.
        """
        result = detect_axis("foo", attrs)
        assert result == expected, f"Expected {expected} for {attrs}, got {result}"

    @pytest.mark.parametrize(
        "axis_value, expected",
        [("X ", "X"), (" y ", "Y"), ("\tZ", "Z")],
        ids=["trailing-space", "padded-lower", "leading-tab"],
    )
    def test_axis_value_is_whitespace_stripped(self, axis_value, expected):
        """A whitespace-padded ``axis`` value still classifies (review L2).

        Args:
            axis_value: The raw, whitespace-padded ``axis`` attribute value.
            expected: The normalized axis role.

        Test scenario:
            ``detect_axis`` strips surrounding whitespace before matching, so ``axis="X "``
            is treated as ``"X"`` rather than falling through to the weaker heuristics.
        """
        result = detect_axis("foo", {"axis": axis_value})
        assert result == expected, f"Expected {expected} for axis={axis_value!r}, got {result}"

    def test_explicit_axis_is_authoritative_over_standard_name(self):
        """An explicit ``axis`` wins over a conflicting ``standard_name`` (review M1 contract).

        Test scenario:
            CF treats the explicit ``axis`` attribute as authoritative. When a coordinate
            carries a self-contradictory ``axis="T"`` together with
            ``standard_name="longitude"``, ``detect_axis`` returns the declared ``"T"`` rather
            than silently overriding it with the longitude heuristic. Pins the deliberate
            precedence (the spatial callers then treat a non-X/Y role as non-spatial).
        """
        result = detect_axis("foo", {"axis": "T", "standard_name": "longitude"})
        assert result == "T", f"explicit axis should win; expected T, got {result}"


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
        assert result[2] == pytest.approx(3.0), "3.0 should be preserved"

    def test_valid_max(self):
        """Values above valid_max are replaced with NaN."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_valid_range_mask(arr, valid_max=3.5)
        assert result[2] == pytest.approx(3.0), "3.0 should be preserved"
        assert np.isnan(result[3]), "4.0 should be masked"
        assert np.isnan(result[4]), "5.0 should be masked"

    def test_valid_range(self):
        """valid_range sets both min and max."""
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = apply_valid_range_mask(arr, valid_range=[1.0, 3.0])
        assert np.isnan(result[0]), "0.0 should be masked"
        assert result[1] == pytest.approx(1.0), "1.0 should be preserved"
        assert result[3] == pytest.approx(3.0), "3.0 should be preserved"
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

    def test_360_day_calendar(self):
        """360_day calendar: 30 days per month."""
        func = create_time_conversion_func(
            "days since 2000-01-01", out_format="%Y-%m-%d", calendar="360_day"
        )
        result = func(30)
        assert result == "2000-02-01", f"Expected 2000-02-01, got {result}"

    def test_noleap_calendar(self):
        """noleap calendar: no Feb 29."""
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
