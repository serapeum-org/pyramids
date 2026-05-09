"""Comprehensive unit tests for pyramids.netcdf.utils module.

Tests all public and private utility functions used by the NetCDF
multidimensional metadata pipeline, using mocked GDAL/OGR objects
to avoid I/O dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from pyramids.netcdf.utils import (
    _dtype_to_str,
    _export_srs,
    _full_name_with_fallback,
    _get_array_nodata,
    _get_array_scale_offset,
    _get_block_size,
    _get_coord_variable_names,
    _get_driver_name,
    _get_group_name,
    _get_root_group,
    _normalize_attr_value,
    _normalize_origin_string,
    _parse_units_origin,
    _read_attribute_value,
    _read_attributes,
    _safe_array_names,
    _safe_group_names,
    _to_py_scalar,
    create_time_conversion_func,
)

pytestmark = pytest.mark.core


def _make_group(
    full_name: str | None = None,
    name: str | None = None,
    array_names: list[str] | None = None,
    group_names: list[str] | None = None,
    fail_full_name: bool = False,
    fail_name: bool = False,
    fail_arrays: bool = False,
    fail_groups: bool = False,
) -> MagicMock:
    """Build a mock ``gdal.Group`` with configurable behaviour."""
    grp = MagicMock()
    if fail_full_name:
        grp.GetFullName.side_effect = RuntimeError("no full name")
    else:
        grp.GetFullName.return_value = full_name

    if fail_name:
        grp.GetName.side_effect = RuntimeError("no name")
    else:
        grp.GetName.return_value = name

    if fail_arrays:
        grp.GetMDArrayNames.side_effect = RuntimeError("no arrays")
    elif array_names is not None:
        grp.GetMDArrayNames.return_value = array_names
    else:
        grp.GetMDArrayNames.return_value = None

    if fail_groups:
        grp.GetGroupNames.side_effect = RuntimeError("no groups")
    elif group_names is not None:
        grp.GetGroupNames.return_value = group_names
    else:
        grp.GetGroupNames.return_value = None

    return grp


def _make_attribute(name: str, value, fail_read: bool = False) -> MagicMock:
    """Build a mock ``gdal.Attribute``."""
    att = MagicMock()
    att.GetName.return_value = name
    if fail_read:
        att.Read.side_effect = RuntimeError("cannot read")
        # Make all six fallback readers fail by default
        att.ReadAsInt64.side_effect = RuntimeError("nope")
        att.ReadAsInt64Array.side_effect = RuntimeError("nope")
        att.ReadAsDouble.side_effect = RuntimeError("nope")
        att.ReadAsDoubleArray.side_effect = RuntimeError("nope")
        att.ReadAsString.side_effect = RuntimeError("nope")
        att.ReadAsStringArray.side_effect = RuntimeError("nope")
    else:
        att.Read.return_value = value
    return att


class TestFullNameWithFallback:
    """Tests for _full_name_with_fallback."""

    def test_returns_full_name_when_available(self):
        """GetFullName succeeds and its value is returned."""
        grp = _make_group(full_name="/root/sub")
        result = _full_name_with_fallback(grp)
        assert result == "/root/sub", "Should return the value from GetFullName"

    def test_falls_back_to_get_name(self):
        """GetFullName fails but GetName succeeds."""
        grp = _make_group(name="child", fail_full_name=True)
        result = _full_name_with_fallback(grp)
        assert result == "/child", "Should construct '/<name>' from GetName"

    def test_falls_back_to_default_name(self):
        """Both GetFullName and GetName fail; uses default_name."""
        grp = _make_group(fail_full_name=True, fail_name=True)
        result = _full_name_with_fallback(grp, default_name="fallback")
        assert result == "/fallback", "Should use the provided default_name"

    def test_returns_slash_when_all_fail_no_default(self):
        """All name methods fail and no default is given."""
        grp = _make_group(fail_full_name=True, fail_name=True)
        result = _full_name_with_fallback(grp)
        assert result == "/", "Should return '/' for unnamed root groups"

    def test_returns_slash_when_name_is_empty(self):
        """GetFullName fails and GetName returns empty string."""
        grp = _make_group(name="", fail_full_name=True)
        result = _full_name_with_fallback(grp)
        assert result == "/", "Empty name should produce '/'"


class TestGetGroupName:
    """Tests for _get_group_name."""

    def test_returns_name(self):
        """GetName succeeds."""
        grp = _make_group(name="temperature")
        result = _get_group_name(grp)
        assert result == "temperature", "Should return the group name"

    def test_returns_empty_on_exception(self):
        """GetName raises and function returns ''."""
        grp = _make_group(fail_name=True)
        result = _get_group_name(grp)
        assert result == "", "Should return empty string on failure"


class TestSafeArrayNames:
    """Tests for _safe_array_names."""

    def test_returns_sorted_names(self):
        """Array names are returned sorted."""
        grp = _make_group(array_names=["z", "a", "m"])
        result = _safe_array_names(grp)
        assert result == ["a", "m", "z"], "Names should be sorted alphabetically"

    def test_returns_empty_list_on_none(self):
        """GetMDArrayNames returns None."""
        grp = _make_group(array_names=None)
        grp.GetMDArrayNames.return_value = None
        result = _safe_array_names(grp)
        assert result == [], "Should return empty list when None"

    def test_returns_empty_list_on_exception(self):
        """GetMDArrayNames raises."""
        grp = _make_group(fail_arrays=True)
        result = _safe_array_names(grp)
        assert result == [], "Should return empty list on exception"

    def test_returns_empty_list_for_empty_group(self):
        """Group has zero arrays."""
        grp = _make_group(array_names=[])
        result = _safe_array_names(grp)
        assert result == [], "Should return empty list for empty group"


class TestSafeGroupNames:
    """Tests for _safe_group_names."""

    def test_returns_sorted_names(self):
        """Sub-group names are returned sorted."""
        grp = _make_group(group_names=["beta", "alpha"])
        result = _safe_group_names(grp)
        assert result == ["alpha", "beta"], "Names should be sorted alphabetically"

    def test_returns_empty_list_on_none(self):
        """GetGroupNames returns None."""
        grp = _make_group(group_names=None)
        grp.GetGroupNames.return_value = None
        result = _safe_group_names(grp)
        assert result == [], "Should return empty list when None"

    def test_returns_empty_list_on_exception(self):
        """GetGroupNames raises."""
        grp = _make_group(fail_groups=True)
        result = _safe_group_names(grp)
        assert result == [], "Should return empty list on exception"


class TestGetRootGroup:
    """Tests for _get_root_group."""

    def test_returns_root_group(self):
        """GetRootGroup succeeds."""
        ds = MagicMock()
        sentinel = MagicMock()
        ds.GetRootGroup.return_value = sentinel
        result = _get_root_group(ds)
        assert result is sentinel, "Should return the root group object"

    def test_returns_none_on_exception(self):
        """GetRootGroup raises."""
        ds = MagicMock()
        ds.GetRootGroup.side_effect = RuntimeError("not MDIM")
        result = _get_root_group(ds)
        assert result is None, "Should return None when GetRootGroup fails"


class TestGetDriverName:
    """Tests for _get_driver_name."""

    def test_returns_driver_short_name(self):
        """Normal dataset returns driver short name."""
        ds = MagicMock()
        driver = MagicMock()
        driver.ShortName = "netCDF"
        ds.GetDriver.return_value = driver
        result = _get_driver_name(ds)
        assert result == "netCDF", "Should return the driver short name"

    def test_returns_unknown_on_exception(self):
        """GetDriver raises."""
        ds = MagicMock()
        ds.GetDriver.side_effect = RuntimeError("no driver")
        result = _get_driver_name(ds)
        assert result == "UNKNOWN", "Should return 'UNKNOWN' on failure"


class TestExportSrs:
    """Tests for _export_srs."""

    def test_returns_none_tuple_for_none_input(self):
        """None SRS yields (None, None)."""
        wkt, projjson = _export_srs(None)
        assert wkt is None, "WKT should be None for None SRS"
        assert projjson is None, "PROJJSON should be None for None SRS"

    def test_exports_both_wkt_and_projjson(self):
        """Both exports succeed."""
        srs = MagicMock()
        srs.ExportToWkt.return_value = "GEOGCS[...]"
        srs.ExportToJSON.return_value = '{"type":"GeographicCRS"}'
        wkt, projjson = _export_srs(srs)
        assert wkt == "GEOGCS[...]", "WKT should match ExportToWkt result"
        assert (
            projjson == '{"type":"GeographicCRS"}'
        ), "PROJJSON should match ExportToJSON result"

    def test_wkt_fails_projjson_succeeds(self):
        """ExportToWkt raises but ExportToJSON succeeds."""
        srs = MagicMock()
        srs.ExportToWkt.side_effect = RuntimeError("wkt fail")
        srs.ExportToJSON.return_value = '{"type":"GeographicCRS"}'
        wkt, projjson = _export_srs(srs)
        assert wkt is None, "WKT should be None when ExportToWkt fails"
        assert projjson == '{"type":"GeographicCRS"}', "PROJJSON should still succeed"

    def test_both_exports_fail(self):
        """Both exports raise."""
        srs = MagicMock()
        srs.ExportToWkt.side_effect = RuntimeError("fail")
        srs.ExportToJSON.side_effect = RuntimeError("fail")
        wkt, projjson = _export_srs(srs)
        assert wkt is None, "WKT should be None"
        assert projjson is None, "PROJJSON should be None"

    def test_falsy_srs_treated_as_none(self):
        """Falsy (but not None) SRS returns (None, None).

        The function checks ``if not srs`` so 0, False, empty
        strings, etc. should all result in (None, None).
        """
        wkt, projjson = _export_srs(0)
        assert (wkt, projjson) == (None, None), "Falsy SRS should yield (None, None)"


class TestGetArrayNodata:
    """Tests for _get_array_nodata."""

    def test_fill_value_from_attrs(self):
        """_FillValue attribute takes precedence."""
        mdarr = MagicMock(spec=[])
        attrs = {"_FillValue": -9999.0}
        result = _get_array_nodata(mdarr, attrs)
        assert result == -9999.0, "Should use _FillValue from attributes"

    def test_missing_value_from_attrs(self):
        """missing_value used when _FillValue is absent."""
        mdarr = MagicMock(spec=[])
        attrs = {"missing_value": -1}
        result = _get_array_nodata(mdarr, attrs)
        assert result == -1, "Should fall back to missing_value"

    def test_fill_value_list_returns_first_element(self):
        """List-valued _FillValue returns first element."""
        mdarr = MagicMock(spec=[])
        attrs = {"_FillValue": [1e20, 1e20]}
        result = _get_array_nodata(mdarr, attrs)
        assert result == 1e20, "Should return first element of list _FillValue"

    def test_fill_value_empty_list_returns_none(self):
        """Empty list _FillValue returns None."""
        mdarr = MagicMock(spec=[])
        attrs = {"_FillValue": []}
        result = _get_array_nodata(mdarr, attrs)
        assert result is None, "Empty list _FillValue should return None"

    def test_gdal_api_nodata_as_double(self):
        """Falls back to GetNoDataValueAsDouble."""
        mdarr = MagicMock()
        mdarr.GetNoDataValueAsDouble.return_value = -999.0
        mdarr.GetNoDataValueAsInt64.return_value = None
        mdarr.GetNoDataValueAsString.return_value = None
        attrs: dict = {}
        result = _get_array_nodata(mdarr, attrs)
        assert result == -999.0, "Should fall back to GDAL API double"

    def test_gdal_api_returns_tuple_with_flag(self):
        """GDAL returns (value, has_value) tuple."""
        mdarr = MagicMock()
        mdarr.GetNoDataValueAsDouble.return_value = (-9999.0, True)
        attrs: dict = {}
        result = _get_array_nodata(mdarr, attrs)
        assert result == -9999.0, "Should unpack (value, True) tuple"

    def test_gdal_api_returns_tuple_with_false_flag(self):
        """GDAL returns (value, False) meaning no nodata; should skip."""
        mdarr = MagicMock()
        mdarr.GetNoDataValueAsDouble.return_value = (0.0, False)
        mdarr.GetNoDataValueAsInt64.return_value = (0, False)
        mdarr.GetNoDataValueAsString.return_value = ("", False)
        attrs: dict = {}
        result = _get_array_nodata(mdarr, attrs)
        assert result is None, "Should return None when GDAL flag is False"

    def test_gdal_api_all_methods_fail(self):
        """All GDAL API methods raise."""
        mdarr = MagicMock()
        mdarr.GetNoDataValueAsDouble.side_effect = RuntimeError
        mdarr.GetNoDataValueAsInt64.side_effect = RuntimeError
        mdarr.GetNoDataValueAsString.side_effect = RuntimeError
        attrs: dict = {}
        result = _get_array_nodata(mdarr, attrs)
        assert result is None, "Should return None when all API methods fail"

    def test_returns_none_when_no_attrs_no_api(self):
        """No attrs and no GDAL API methods at all."""
        mdarr = MagicMock(spec=[])
        attrs: dict = {}
        result = _get_array_nodata(mdarr, attrs)
        assert result is None, "Should return None with no info available"

    def test_fill_value_string_attr(self):
        """String _FillValue attribute."""
        mdarr = MagicMock(spec=[])
        attrs = {"_FillValue": "N/A"}
        result = _get_array_nodata(mdarr, attrs)
        assert result == "N/A", "Should return string _FillValue"

    def test_fill_value_takes_precedence_over_missing_value(self):
        """_FillValue has higher precedence than missing_value."""
        mdarr = MagicMock(spec=[])
        attrs = {"_FillValue": -1, "missing_value": -999}
        result = _get_array_nodata(mdarr, attrs)
        assert result == -1, "_FillValue should have higher precedence"


class TestGetArrayScaleOffset:
    """Tests for _get_array_scale_offset."""

    def test_scale_and_offset_from_attrs(self):
        """CF attributes provide scale and offset."""
        mdarr = MagicMock(spec=[])
        attrs = {"scale_factor": 0.1, "add_offset": 273.15}
        scale, offset = _get_array_scale_offset(mdarr, attrs)
        assert scale == pytest.approx(
            0.1
        ), "Scale should come from scale_factor attribute"
        assert offset == pytest.approx(
            273.15
        ), "Offset should come from add_offset attribute"

    def test_none_when_no_info(self):
        """No attributes and no GDAL API methods."""
        mdarr = MagicMock(spec=[])
        attrs: dict = {}
        scale, offset = _get_array_scale_offset(mdarr, attrs)
        assert scale is None, "Scale should be None"
        assert offset is None, "Offset should be None"

    def test_gdal_api_overrides_attrs(self):
        """GDAL API values override CF attributes when present."""
        mdarr = MagicMock()
        mdarr.GetScale.return_value = 0.01
        mdarr.GetOffset.return_value = 100.0
        attrs = {"scale_factor": 0.1, "add_offset": 200.0}
        scale, offset = _get_array_scale_offset(mdarr, attrs)
        assert scale == pytest.approx(
            0.01
        ), "GDAL API scale should override CF attribute"
        assert offset == pytest.approx(
            100.0
        ), "GDAL API offset should override CF attribute"

    def test_gdal_api_returns_none_keeps_attr_value(self):
        """GDAL API returns None; CF attribute value is kept."""
        mdarr = MagicMock()
        mdarr.GetScale.return_value = None
        mdarr.GetOffset.return_value = None
        attrs = {"scale_factor": 0.5, "add_offset": 10.0}
        scale, offset = _get_array_scale_offset(mdarr, attrs)
        assert scale == pytest.approx(
            0.5
        ), "Should keep CF scale when GDAL returns None"
        assert offset == pytest.approx(
            10.0
        ), "Should keep CF offset when GDAL returns None"

    def test_gdal_api_raises_keeps_attr_value(self):
        """GDAL API raises; CF attribute value is kept."""
        mdarr = MagicMock()
        mdarr.GetScale.side_effect = RuntimeError("fail")
        mdarr.GetOffset.side_effect = RuntimeError("fail")
        attrs = {"scale_factor": 2, "add_offset": 5}
        scale, offset = _get_array_scale_offset(mdarr, attrs)
        assert scale == pytest.approx(2.0), "Should keep CF scale when GDAL API fails"
        assert offset == pytest.approx(5.0), "Should keep CF offset when GDAL API fails"

    def test_integer_attrs_converted_to_float(self):
        """Integer CF attrs are converted to float."""
        mdarr = MagicMock(spec=[])
        attrs = {"scale_factor": 1, "add_offset": 0}
        scale, offset = _get_array_scale_offset(mdarr, attrs)
        assert isinstance(scale, float), "Integer scale should be converted to float"
        assert isinstance(offset, float), "Integer offset should be converted to float"

    def test_non_numeric_attrs_ignored(self):
        """String-typed CF attributes are ignored."""
        mdarr = MagicMock(spec=[])
        attrs = {"scale_factor": "bad", "add_offset": "also_bad"}
        scale, offset = _get_array_scale_offset(mdarr, attrs)
        assert scale is None, "Non-numeric scale_factor should be ignored"
        assert offset is None, "Non-numeric add_offset should be ignored"


class TestGetBlockSize:
    """Tests for _get_block_size."""

    def test_returns_block_size(self):
        """Normal block size returned as list of ints."""
        mdarr = MagicMock()
        mdarr.GetBlockSize.return_value = [256, 256]
        result = _get_block_size(mdarr)
        assert result == [256, 256], "Should return block sizes as list of ints"

    def test_returns_none_on_empty(self):
        """GetBlockSize returns empty list or falsy value."""
        mdarr = MagicMock()
        mdarr.GetBlockSize.return_value = []
        result = _get_block_size(mdarr)
        assert result is None, "Should return None for empty block size"

    def test_returns_none_on_exception(self):
        """GetBlockSize raises."""
        mdarr = MagicMock()
        mdarr.GetBlockSize.side_effect = RuntimeError("fail")
        result = _get_block_size(mdarr)
        assert result is None, "Should return None on exception"

    def test_returns_none_on_none_value(self):
        """GetBlockSize returns None."""
        mdarr = MagicMock()
        mdarr.GetBlockSize.return_value = None
        result = _get_block_size(mdarr)
        assert result is None, "Should return None when GetBlockSize returns None"

    def test_converts_elements_to_int(self):
        """Float block sizes are converted to int."""
        mdarr = MagicMock()
        mdarr.GetBlockSize.return_value = [64.0, 128.0, 32.0]
        result = _get_block_size(mdarr)
        assert result == [64, 128, 32], "Should convert elements to int"
        assert all(isinstance(x, int) for x in result), "All elements should be int"


class TestGetCoordVariableNames:
    """Tests for _get_coord_variable_names."""

    def test_returns_names_via_get_full_name(self):
        """Coordinate variables with GetFullName method."""
        cv1 = MagicMock()
        cv1.GetFullName.return_value = "/lat"
        cv2 = MagicMock()
        cv2.GetFullName.return_value = "/lon"
        mdarr = MagicMock()
        mdarr.GetCoordinateVariables.return_value = [cv1, cv2]
        result = _get_coord_variable_names(mdarr)
        assert result == [
            "/lat",
            "/lon",
        ], "Should use GetFullName for coordinate variables"

    def test_falls_back_to_get_name(self):
        """Coordinate variable without GetFullName but with GetName."""
        cv = MagicMock(spec=["GetName"])
        cv.GetName.return_value = "time"
        mdarr = MagicMock()
        mdarr.GetCoordinateVariables.return_value = [cv]
        result = _get_coord_variable_names(mdarr)
        assert result == ["time"], "Should fall back to GetName"

    def test_falls_back_to_str(self):
        """Coordinate variable without GetFullName or GetName."""
        cv = SimpleNamespace()
        mdarr = MagicMock()
        mdarr.GetCoordinateVariables.return_value = [cv]
        result = _get_coord_variable_names(mdarr)
        assert len(result) == 1, "Should return one name via str() fallback"

    def test_returns_empty_on_exception(self):
        """GetCoordinateVariables raises."""
        mdarr = MagicMock()
        mdarr.GetCoordinateVariables.side_effect = RuntimeError
        result = _get_coord_variable_names(mdarr)
        assert result == [], "Should return empty list on exception"

    def test_returns_empty_for_none_cvs(self):
        """GetCoordinateVariables returns None."""
        mdarr = MagicMock()
        mdarr.GetCoordinateVariables.return_value = None
        result = _get_coord_variable_names(mdarr)
        assert result == [], "Should return empty list for None"

    def test_returns_empty_for_empty_cvs(self):
        """GetCoordinateVariables returns empty list."""
        mdarr = MagicMock()
        mdarr.GetCoordinateVariables.return_value = []
        result = _get_coord_variable_names(mdarr)
        assert result == [], "Should return empty list for empty list"

    def test_get_full_name_raises_falls_to_str(self):
        """GetFullName exists but raises; falls to str() in except."""
        cv = MagicMock()
        cv.GetFullName.side_effect = RuntimeError("fail")
        mdarr = MagicMock()
        mdarr.GetCoordinateVariables.return_value = [cv]
        result = _get_coord_variable_names(mdarr)
        assert len(result) == 1, "Should still produce a name via str() fallback"


class TestNormalizeOriginString:
    """Tests for _normalize_origin_string."""

    def test_minimal_origin(self):
        """Pad a minimal '1-1-1 0:0:0' origin."""
        result = _normalize_origin_string("1-1-1 0:0:0")
        assert (
            result == "0001-01-01 00:00:00"
        ), "Should zero-pad year, month, day, hour, min, sec"

    def test_iso_t_separator(self):
        """T separator is replaced with space."""
        result = _normalize_origin_string("1979-1-1T0:0:0")
        assert result == "1979-01-01 00:00:00", "T separator should be replaced"

    def test_date_only(self):
        """Date-only origin gets midnight time."""
        result = _normalize_origin_string("2000-6-15")
        assert result == "2000-06-15 00:00:00", "Date-only should append 00:00:00"

    def test_already_padded(self):
        """Already zero-padded input is unchanged."""
        result = _normalize_origin_string("1979-01-01 00:00:00")
        assert (
            result == "1979-01-01 00:00:00"
        ), "Already padded origin should be unchanged"

    def test_fractional_seconds(self):
        """Fractional seconds are preserved."""
        result = _normalize_origin_string("2000-1-1 0:0:0.5")
        assert (
            result == "2000-01-01 00:00:0.5"
        ), "Fractional seconds should be preserved"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped."""
        result = _normalize_origin_string("  1979-01-01 00:00:00  ")
        assert result == "1979-01-01 00:00:00", "Whitespace should be stripped"

    def test_partial_date_pads_missing_parts(self):
        """Year-only or year-month date is padded."""
        result = _normalize_origin_string("2000")
        assert (
            result == "2000-01-01 00:00:00"
        ), "Missing month and day should default to 01"

    def test_partial_time_pads_missing_parts(self):
        """Partial time (hour only) is padded."""
        result = _normalize_origin_string("2000-01-01 12")
        assert (
            result == "2000-01-01 12:00:00"
        ), "Missing minutes and seconds should be 00"

    def test_year_month_only(self):
        """Year-month only with no day."""
        result = _normalize_origin_string("2000-6")
        assert result == "2000-06-01 00:00:00", "Missing day should default to 01"


class TestParseUnitsOrigin:
    """Tests for _parse_units_origin."""

    def test_days_since(self):
        """Standard 'days since' string."""
        unit, origin = _parse_units_origin("days since 1979-01-01")
        assert unit == "days", "Unit should be 'days'"
        assert origin == datetime(1979, 1, 1), "Origin should be 1979-01-01"

    def test_hours_since(self):
        """Hours-based unit string."""
        unit, origin = _parse_units_origin("hours since 2000-01-01 00:00:00")
        assert unit == "hours", "Unit should be 'hours'"
        assert origin == datetime(2000, 1, 1), "Origin should be 2000-01-01"

    def test_abbreviated_origin(self):
        """Abbreviated origin '1-1-1 0:0:0'."""
        unit, origin = _parse_units_origin("hours since 1-1-1 0:0:0")
        assert unit == "hours", "Unit should be 'hours'"
        assert origin.year == 1, "Year should be 1"

    def test_invalid_format_raises(self):
        """Non-matching string raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized"):
            _parse_units_origin("not a valid string")

    def test_case_insensitive(self):
        """'Days SINCE' works case-insensitively."""
        unit, origin = _parse_units_origin("Days SINCE 1979-01-01")
        assert unit == "days", "Unit should be lowercased"

    def test_extra_whitespace(self):
        """Extra whitespace around the string."""
        unit, origin = _parse_units_origin("  minutes   since   2020-06-15  ")
        assert unit == "minutes", "Unit should be 'minutes'"
        assert origin.year == 2020, "Year should be 2020"

    def test_seconds_since(self):
        """Seconds-based unit string."""
        unit, origin = _parse_units_origin("seconds since 1970-01-01")
        assert unit == "seconds", "Unit should be 'seconds'"
        assert origin == datetime(1970, 1, 1), "Origin should be Unix epoch"

    def test_fromisoformat_fallback_to_strptime(self, monkeypatch):
        """When fromisoformat raises ValueError, strptime is used.

        In Python 3.11+ fromisoformat is very permissive, so
        this path is only reachable by forcing fromisoformat to
        fail via monkeypatch on the module-level datetime class.
        """
        import pyramids.netcdf.utils as utils_mod

        real_datetime = datetime

        class _PatchedDatetime(real_datetime):
            """datetime subclass whose fromisoformat always fails."""

            @classmethod
            def fromisoformat(cls, s):
                raise ValueError("forced failure for test")

        monkeypatch.setattr(utils_mod, "datetime", _PatchedDatetime)
        unit, origin = _parse_units_origin("days since 1979-01-01 00:00:00")
        assert unit == "days", "Unit should be 'days'"
        assert origin == real_datetime(
            1979, 1, 1
        ), "Should fall back to strptime when fromisoformat fails"


class TestCreateTimeConversionFunc:
    """Tests for create_time_conversion_func."""

    def test_days_conversion(self):
        """Convert day offsets from 1979 origin."""
        convert = create_time_conversion_func("days since 1979-01-01")
        assert convert(0) == "1979-01-01 00:00:00", "Day 0 should be origin"
        assert convert(365) == "1980-01-01 00:00:00", "Day 365 should be next year"

    def test_hours_conversion(self):
        """Convert hour offsets."""
        convert = create_time_conversion_func("hours since 2000-01-01")
        assert convert(24) == "2000-01-02 00:00:00", "24 hours should be next day"

    def test_minutes_conversion(self):
        """Convert minute offsets."""
        convert = create_time_conversion_func("minutes since 2000-01-01")
        assert convert(60) == "2000-01-01 01:00:00", "60 minutes should be 1 hour"

    def test_seconds_conversion(self):
        """Convert second offsets."""
        convert = create_time_conversion_func("seconds since 2000-01-01")
        assert convert(3600) == "2000-01-01 01:00:00", "3600 seconds should be 1 hour"

    def test_custom_format(self):
        """Custom output format."""
        convert = create_time_conversion_func(
            "hours since 2000-01-01",
            out_format="%Y-%m-%d",
        )
        assert (
            convert(24) == "2000-01-02"
        ), "Custom format should produce date-only string"

    def test_unsupported_unit_raises(self):
        """Unsupported unit string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            create_time_conversion_func("months since 2000-01-01")

    def test_fractional_days(self):
        """Fractional day offsets."""
        convert = create_time_conversion_func("days since 2000-01-01")
        assert convert(0.5) == "2000-01-01 12:00:00", "0.5 days should be noon"

    def test_negative_offset(self):
        """Negative offset goes before origin."""
        convert = create_time_conversion_func("days since 2000-01-02")
        assert (
            convert(-1) == "2000-01-01 00:00:00"
        ), "Negative offset should go before origin"

    def test_invalid_units_string_raises(self):
        """Completely invalid string raises ValueError."""
        with pytest.raises(ValueError):
            create_time_conversion_func("garbage")


class TestDtypeToStr:
    """Tests for _dtype_to_str."""

    def test_returns_lowercase_name_from_get_name(self):
        """GetName returns a valid name, lowercased to numpy convention."""
        dt = MagicMock()
        dt.GetName.return_value = "Float64"
        result = _dtype_to_str(dt)
        assert result == "float64", f"Should return lowercased name, got {result}"

    def test_falls_back_to_numeric_dtype(self):
        """GetName fails; falls back to GetNumericDataType."""
        dt = MagicMock()
        dt.GetName.side_effect = AttributeError("no GetName")
        dt.GetNumericDataType.return_value = 3  # GDT_Int16
        result = _dtype_to_str(dt)
        assert result == "int16", f"Should fall back to numeric dtype, got {result}"

    def test_returns_unknown_when_all_fail(self):
        """Both GetName and str() fail."""
        dt = MagicMock()
        dt.GetName.side_effect = RuntimeError("fail")
        dt.__str__ = MagicMock(side_effect=RuntimeError("also fail"))
        result = _dtype_to_str(dt)
        assert result == "unknown", "Should return 'unknown' as last resort"

    def test_empty_name_falls_to_numeric(self):
        """GetName returns empty; falls back to GetNumericDataType."""
        dt = MagicMock()
        dt.GetName.return_value = ""
        dt.GetNumericDataType.return_value = 6  # GDT_Float32
        result = _dtype_to_str(dt)
        assert (
            result == "float32"
        ), f"Empty name should fall through to numeric, got {result}"

    def test_no_numeric_returns_unknown(self):
        """GetName empty and GetNumericDataType fails; returns 'unknown'."""
        dt = MagicMock()
        dt.GetName.return_value = None
        dt.GetNumericDataType.side_effect = Exception("no numeric")
        result = _dtype_to_str(dt)
        assert (
            result == "unknown"
        ), f"Should return 'unknown' when all paths fail, got {result}"


class TestToPyScalar:
    """Tests for _to_py_scalar."""

    def test_int_passthrough(self):
        """Native int passes through unchanged."""
        result = _to_py_scalar(42)
        assert result == 42, "Int should pass through"
        assert isinstance(result, int), "Type should be int"

    def test_float_passthrough(self):
        """Native float passes through unchanged."""
        result = _to_py_scalar(3.14)
        assert result == pytest.approx(3.14), "Float should pass through"

    def test_str_passthrough(self):
        """Native str passes through unchanged."""
        result = _to_py_scalar("hello")
        assert result == "hello", "String should pass through"

    def test_bool_passthrough(self):
        """Native bool passes through unchanged."""
        result = _to_py_scalar(True)
        assert result is True, "Bool should pass through"

    def test_none_passthrough(self):
        """None passes through unchanged."""
        result = _to_py_scalar(None)
        assert result is None, "None should pass through"

    def test_numpy_int(self):
        """numpy int scalar is converted via .item()."""
        result = _to_py_scalar(np.int32(7))
        assert result == 7, "numpy int32 should be converted to Python int"
        assert isinstance(result, int), "Type should be int"

    def test_numpy_float(self):
        """numpy float scalar is converted via .item()."""
        result = _to_py_scalar(np.float64(2.5))
        assert result == pytest.approx(2.5), "numpy float64 should be converted"
        assert isinstance(result, float), "Type should be float"

    def test_numpy_bool(self):
        """numpy bool scalar is converted via .item()."""
        result = _to_py_scalar(np.bool_(True))
        assert result is True, "numpy bool should convert to Python True"

    def test_bytes_decoded(self):
        """Bytes are decoded to str."""
        result = _to_py_scalar(b"hello")
        assert result == "hello", "Bytes should be decoded to string"

    def test_bytes_with_bad_encoding(self):
        """Non-UTF-8 bytes are decoded with errors='ignore'."""
        result = _to_py_scalar(b"\xff\xfe")
        assert isinstance(result, str), "Should still produce a string"

    def test_unknown_type_stringified(self):
        """Non-convertible type is converted via str()."""
        result = _to_py_scalar({"key": "val"})
        assert isinstance(result, str), "Unknown type should be stringified"

    def test_item_raises_falls_through(self):
        """Object with .item() that raises falls through."""
        obj = MagicMock()
        obj.item.side_effect = RuntimeError("broken item")
        result = _to_py_scalar(obj)
        assert isinstance(result, str), "Should fall back to str() when .item() fails"


class TestNormalizeAttrValue:
    """Tests for _normalize_attr_value."""

    def test_scalar_int(self):
        """Integer scalar passes through."""
        result = _normalize_attr_value(42)
        assert result == 42, "Integer scalar should be returned as-is"

    def test_scalar_string(self):
        """String scalar passes through."""
        result = _normalize_attr_value("units")
        assert result == "units", "String scalar should be returned as-is"

    def test_list_converted(self):
        """List elements are individually converted."""
        result = _normalize_attr_value([np.float32(1.0), np.float32(2.0)])
        assert isinstance(result, list), "Should return a list"
        assert result == [
            pytest.approx(1.0),
            pytest.approx(2.0),
        ], "List elements should be converted"

    def test_tuple_converted_to_list(self):
        """Tuple is treated as vector and returned as list."""
        result = _normalize_attr_value((1, 2, 3))
        assert isinstance(result, list), "Tuple should be converted to list"
        assert result == [1, 2, 3], "Values should be preserved"

    def test_none_value(self):
        """None is passed through _to_py_scalar."""
        result = _normalize_attr_value(None)
        assert result is None, "None should pass through"

    def test_bytes_value(self):
        """Bytes scalar is decoded."""
        result = _normalize_attr_value(b"CF-1.6")
        assert result == "CF-1.6", "Bytes should be decoded to string"

    def test_empty_list(self):
        """Empty list is returned as empty list."""
        result = _normalize_attr_value([])
        assert result == [], "Empty list should stay empty"


class TestReadAttributeValue:
    """Tests for _read_attribute_value."""

    def test_read_succeeds(self):
        """attr.Read() returns a value."""
        att = _make_attribute("units", "kg m-2")
        result = _read_attribute_value(att)
        assert result == "kg m-2", "Should return the Read() value"

    def test_read_fails_falls_to_read_as_string(self):
        """Read() fails; ReadAsString succeeds."""
        att = _make_attribute("units", None, fail_read=True)
        # Re-enable only ReadAsString (it comes after Int64,
        # Int64Array, Double, DoubleArray in the iteration order)
        att.ReadAsString = MagicMock(return_value="meters")
        result = _read_attribute_value(att)
        assert result == "meters", "Should fall back to ReadAsString"

    def test_read_fails_falls_to_read_as_double(self):
        """Read() fails; ReadAsDouble succeeds."""
        att = _make_attribute("scale", None, fail_read=True)
        # Re-enable only ReadAsDouble
        att.ReadAsDouble = MagicMock(return_value=0.1)
        result = _read_attribute_value(att)
        assert result == pytest.approx(0.1), "Should fall back to ReadAsDouble"

    def test_read_fails_falls_to_read_as_int64(self):
        """Read() fails; ReadAsInt64 succeeds."""
        att = _make_attribute("count", None, fail_read=True)
        # Re-enable only ReadAsInt64 (first in iteration order)
        att.ReadAsInt64 = MagicMock(return_value=42)
        result = _read_attribute_value(att)
        assert result == 42, "Should fall back to ReadAsInt64"

    def test_all_readers_fail(self):
        """All read methods fail; returns None."""
        att = _make_attribute("broken", None, fail_read=True)
        result = _read_attribute_value(att)
        assert result is None, "Should return None when all readers fail"

    def test_read_returns_list(self):
        """Read() returns a list value."""
        att = _make_attribute("bounds", [0.0, 1.0])
        result = _read_attribute_value(att)
        assert result == [
            pytest.approx(0.0),
            pytest.approx(1.0),
        ], "Should return list value"

    def test_read_returns_numpy_array(self):
        """Read() returns a numpy array (treated as scalar path)."""
        arr = np.array([10, 20, 30])
        att = _make_attribute("data", arr)
        result = _read_attribute_value(att)
        # numpy arrays have .item() but it only works for size-1 arrays;
        # for multi-element arrays it will fail and fall to str()
        assert result is not None, "Should not be None for numpy array"


class TestReadAttributes:
    """Tests for _read_attributes."""

    def test_reads_multiple_attributes(self):
        """Multiple attributes are read into a dict."""
        att1 = _make_attribute("units", "kg m-2")
        att2 = _make_attribute("long_name", "precipitation")
        obj = MagicMock()
        obj.GetAttributes.return_value = [att1, att2]
        result = _read_attributes(obj)
        assert result == {
            "units": "kg m-2",
            "long_name": "precipitation",
        }, "Should read all attributes into a dict"

    def test_empty_attributes(self):
        """GetAttributes returns empty list."""
        obj = MagicMock()
        obj.GetAttributes.return_value = []
        result = _read_attributes(obj)
        assert result == {}, "Should return empty dict for no attributes"

    def test_get_attributes_returns_none(self):
        """GetAttributes returns None."""
        obj = MagicMock()
        obj.GetAttributes.return_value = None
        result = _read_attributes(obj)
        assert result == {}, "Should return empty dict for None"

    def test_get_attributes_raises(self):
        """GetAttributes raises exception."""
        obj = MagicMock()
        obj.GetAttributes.side_effect = RuntimeError("fail")
        result = _read_attributes(obj)
        assert result == {}, "Should return empty dict on exception"

    def test_attribute_get_name_fails_skipped(self):
        """Attribute whose GetName raises is skipped."""
        att_good = _make_attribute("units", "m/s")
        att_bad = MagicMock()
        att_bad.GetName.side_effect = RuntimeError("no name")
        obj = MagicMock()
        obj.GetAttributes.return_value = [att_bad, att_good]
        result = _read_attributes(obj)
        assert "units" in result, "Good attribute should be present"
        assert len(result) == 1, "Bad attribute should be skipped"

    def test_attribute_read_fails_gets_normalized_none(self):
        """Attribute whose read fails gets normalized None."""
        att = MagicMock()
        att.GetName.return_value = "broken_attr"
        att.Read = MagicMock(side_effect=RuntimeError("read fail"))
        # Make _read_attribute_value raise so it hits the outer
        # except clause in _read_attributes
        obj = MagicMock()
        obj.GetAttributes.return_value = [att]
        result = _read_attributes(obj)
        # The attribute will be present (either from
        # _read_attribute_value fallback or _normalize_attr_value(None))
        assert "broken_attr" in result, "Attribute should still be present in the dict"

    def test_read_attribute_value_raises_caught_by_outer_except(self, monkeypatch):
        """When _read_attribute_value raises, the outer except
        in _read_attributes catches it and stores normalized None.

        This covers lines 680-682 in utils.py.
        """
        import pyramids.netcdf.utils as utils_mod

        att = MagicMock()
        att.GetName.return_value = "explosive_attr"

        def _exploding_read(a):
            raise TypeError("unexpected explosion")

        monkeypatch.setattr(utils_mod, "_read_attribute_value", _exploding_read)

        obj = MagicMock()
        obj.GetAttributes.return_value = [att]
        result = _read_attributes(obj)
        assert (
            "explosive_attr" in result
        ), "Attribute key should be present even when reader explodes"
        assert (
            result["explosive_attr"] is None
        ), "Value should be normalized None from the except branch"
