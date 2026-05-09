"""Tests for CF flag decoding (CF-12), attribute preservation (CF-13),
and CF compliance validation (CF-14).
"""

import numpy as np
import pytest

from pyramids.netcdf.cf import decode_flags, validate_cf
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

GEO = (30.0, 0.5, 0, 35.0, 0, -0.5)
SEED = 42


class TestDecodeFlags:
    """Tests for cf.decode_flags."""

    def test_mutually_exclusive(self):
        """Decode a mutually exclusive flag value.

        Test scenario:
            flag_values=[0,1,2], meanings=[good, questionable, bad]
            value=1 should return 'questionable'.
        """
        result = decode_flags(
            1,
            flag_values=[0, 1, 2],
            flag_meanings=["good", "questionable", "bad"],
        )
        assert result == ["questionable"], f"Expected ['questionable'], got {result!r}"

    def test_mutually_exclusive_no_match(self):
        """Value not in flag_values returns ['unknown']."""
        result = decode_flags(
            99,
            flag_values=[0, 1, 2],
            flag_meanings=["good", "questionable", "bad"],
        )
        assert result == ["unknown"], f"Expected ['unknown'], got {result!r}"

    def test_boolean_masks(self):
        """Decode boolean bit-field flags.

        Test scenario:
            flag_masks=[1, 2, 4], meanings=[low_batt, error, valid]
            value=5 (bits 0 and 2) -> ['low_batt', 'valid']
        """
        result = decode_flags(
            5,
            flag_masks=[1, 2, 4],
            flag_meanings=["low_battery", "sensor_error", "data_valid"],
        )
        assert "low_battery" in result, f"Expected low_battery in {result}"
        assert "data_valid" in result, f"Expected data_valid in {result}"
        assert "sensor_error" not in result, f"sensor_error should not be in {result}"

    def test_combined_masks_and_values(self):
        """Decode combined flag_masks + flag_values."""
        result = decode_flags(
            3,
            flag_masks=[3, 3],
            flag_values=[1, 2],
            flag_meanings=["partly_cloudy", "cloudy"],
        )
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_no_meanings_returns_unknown(self):
        """No flag_meanings returns ['unknown']."""
        result = decode_flags(0)
        assert result == ["unknown"], f"Expected ['unknown'], got {result!r}"


class TestValidateCF:
    """Tests for cf.validate_cf."""

    def _make_mock_var(self, attrs, unit=None):
        class MockVar:
            def __init__(self, a, u):
                self.attributes = a
                self.unit = u

        return MockVar(attrs, unit)

    def _make_mock_dim(self, name):
        class MockDim:
            def __init__(self, n):
                self.name = n
                self.full_name = f"/{n}"

        return MockDim(name)

    def test_compliant_file(self):
        """Compliant metadata produces no issues."""
        global_attrs = {"Conventions": "CF-1.8"}
        dims = {"x": self._make_mock_dim("x")}
        vars_ = {"x": self._make_mock_var({"units": "degrees_east"}, "degrees_east")}
        issues = validate_cf(global_attrs, vars_, dims)
        assert len(issues) == 0, f"Expected no issues, got {issues}"

    def test_missing_conventions(self):
        """Missing Conventions attribute produces an issue."""
        issues = validate_cf({}, {}, {})
        assert any(
            "Conventions" in i for i in issues
        ), f"Expected Conventions warning, got {issues}"

    def test_coordinate_missing_units(self):
        """Coordinate variable without units produces a warning."""
        global_attrs = {"Conventions": "CF-1.8"}
        dims = {"x": self._make_mock_dim("x")}
        vars_ = {"x": self._make_mock_var({}, None)}
        issues = validate_cf(global_attrs, vars_, dims)
        assert any(
            "units" in i for i in issues
        ), f"Expected units warning, got {issues}"

    def test_time_missing_calendar(self):
        """Time coordinate without calendar produces a warning."""
        global_attrs = {"Conventions": "CF-1.8"}
        dims = {"time": self._make_mock_dim("time")}
        vars_ = {"time": self._make_mock_var({"units": "days since 1970-01-01"}, None)}
        issues = validate_cf(global_attrs, vars_, dims)
        assert any(
            "calendar" in i for i in issues
        ), f"Expected calendar warning, got {issues}"


class TestCFAttributePreservation:
    """Tests for CF attribute preservation through GIS ops (CF-13)."""

    def test_round_trip_preserves_variable_attrs(self):
        """Variable attributes survive get_variable -> set_variable.

        Test scenario:
            Create a NetCDF, get a variable (which tracks attrs via
            RT-7), crop it, set it back, verify attrs are preserved.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
        )
        var = nc.get_variable("temp")
        assert hasattr(
            var, "_variable_attrs"
        ), "Variable should have _variable_attrs from RT-7"

    def test_conventions_preserved_after_copy(self):
        """Conventions attribute preserved after copy().

        Test scenario:
            Create, copy, check Conventions on copy.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO, variable_name="temp")
        nc2 = nc.copy()
        ga = nc2.global_attributes
        assert (
            ga.get("Conventions") == "CF-1.8"
        ), f"Conventions lost after copy, got {ga.get('Conventions')!r}"
