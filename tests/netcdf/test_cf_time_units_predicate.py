"""One rule for "is this units string a CF time axis".

Four places asked that question and answered differently. Axis *detection* in
`cf.py` matched a bare lowercased `"since"` substring; *decoding* in `utils.py`
and `labeled.py` required a case-sensitive `" since "`.

A file whose units read `"Days SINCE 1970-01-01"` was therefore reported as a
time axis and then handed back as raw numbers -- no exception, no warning, just
`1.0` where a date belonged. `cftime` parses that string perfectly well; only
pyramids' own guard rejected it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyramids.netcdf.cf import _check_coordinate_variable, detect_axis
from pyramids.netcdf.utils import decode_cf_time, is_cf_time_units

pytestmark = pytest.mark.core

TIME_UNITS = [
    "days since 1970-01-01",
    "Days SINCE 1970-01-01",
    "hours Since 2000-01-01",
    "DAYS SINCE 1970-01-01 00:00:00",
    "seconds since 1979-01-01T00:00:00Z",
]

NOT_TIME_UNITS = [
    "degrees_north",
    "degrees_east",
    "m s-1",
    "K",
    "",
]
NOT_TIME_UNIT_IDS = ["degrees_north", "degrees_east", "m s-1", "K", "empty"]


class TestIsCfTimeUnits:
    """The shape CF actually specifies: `<period> since <timestamp>`."""

    @pytest.mark.parametrize("units", TIME_UNITS)
    def test_a_time_unit_is_recognised_in_any_case(self, units: str):
        """CF does not mandate lower case, and real files vary.

        Args:
            units: A CF time-units string in some capitalisation.

        Test scenario:
            Each of these parses as a time axis in `cftime`. The predicate has
            to agree, or the axis is detected and then silently left numeric.
        """
        assert is_cf_time_units(units) is True

    @pytest.mark.parametrize("units", NOT_TIME_UNITS)
    def test_a_non_time_unit_is_not(self, units: str):
        """Spatial and physical units must not be decoded as dates.

        Args:
            units: A units string that is not a time axis.
        """
        assert is_cf_time_units(units) is False

    def test_a_missing_unit_is_not_a_time_axis(self):
        """`None` is the common case for a variable with no units at all."""
        assert is_cf_time_units(None) is False

    @pytest.mark.parametrize(
        "units",
        ["sincerity", "since", "quiescence", "insincere"],
        ids=["sincerity", "bare-since", "quiescence", "insincere"],
    )
    def test_the_letters_alone_do_not_make_a_time_axis(self, units: str):
        """The old detection matched a bare substring and would take these.

        Args:
            units: A string containing the letters but not the CF form.

        Test scenario:
            `since` has to be a whitespace-separated word with a period before
            it and an epoch after. A substring match called `"sincerity"` a
            time axis, and a bare `"since"` has nothing to count from.
        """
        assert is_cf_time_units(units) is False


class TestDetectionAndDecodingAgree:
    """The regression: they disagreed, and the gap was silent."""

    @pytest.mark.parametrize("units", TIME_UNITS)
    def test_an_axis_detected_as_time_actually_decodes(self, units: str):
        """The invariant that was broken for every uppercase spelling.

        Args:
            units: A CF time-units string in some capitalisation.

        Test scenario:
            `detect_axis` reporting `"T"` is a promise that the values are
            dates. If `decode_cf_time` then returns the input unchanged, the
            caller gets floats presented as a time coordinate -- wrong data,
            with nothing raised to say so.
        """
        assert detect_axis("t", {"units": units}) == "T"

        decoded = decode_cf_time(np.array([1.0]), units)

        assert decoded.dtype.kind in "MO", f"{units!r} detected as T but not decoded"

    def test_the_uppercase_spelling_decodes_to_the_same_date(self):
        """Case changes the spelling, not the instant.

        Test scenario:
            `"Days SINCE 1970-01-01"` and `"days since 1970-01-01"` name the
            same axis, so offset 1 must decode to 2 January either way.
        """
        lower = decode_cf_time(np.array([1.0]), "days since 1970-01-01")
        upper = decode_cf_time(np.array([1.0]), "Days SINCE 1970-01-01")

        assert str(lower[0]) == str(upper[0])
        assert str(lower[0]).startswith("1970-01-02")

    @pytest.mark.parametrize("units", NOT_TIME_UNITS, ids=NOT_TIME_UNIT_IDS)
    def test_a_non_time_axis_is_left_numeric(self, units: str):
        """The other half of the invariant, so the fix did not over-reach.

        Args:
            units: A units string that is not a time axis.

        Test scenario:
            Widening the predicate must not start decoding latitudes as dates.
            These values have to come back exactly as passed: the same numbers,
            still floating point, rather than instants counted from whatever
            origin the string was mistaken for. Every string the predicate
            rejects is exercised, the empty one included -- taking a slice of
            the list here let a reordering silently change what was covered.
        """
        values = np.array([1.0, 2.0])

        decoded = decode_cf_time(values, units)

        assert np.array_equal(decoded, values), (
            f"{units!r} came back as {decoded!r} rather than untouched"
        )
        assert decoded.dtype == values.dtype, (
            f"{units!r} was decoded away from {values.dtype} to {decoded.dtype}"
        )


class TestThePredicateIsTotal:
    """A malformed file must be reported on, not crash the reporter.

    GDAL attributes are normalised into scalars *or lists*, so a `units`
    attribute written as a one-element array is a real input. The predicate
    reaches `re.search`, which raises on anything that is not a string --
    and `check_cf_compliance` exists precisely to describe files like that.
    """

    @pytest.mark.parametrize(
        "units",
        [
            ["days since 1970-01-01"],
            [],
            1,
            3.5,
            {"units": "days since 1970-01-01"},
            bytes([0xFF, 0xFE]) + b" not utf-8",
        ],
        ids=["list", "empty-list", "int", "float", "dict", "undecodable-bytes"],
    )
    def test_a_non_string_units_is_not_a_time_axis(self, units):
        """Every one of these raised `TypeError` out of the predicate.

        Args:
            units: A `units` attribute value that is not text.

        Test scenario:
            The answer is False -- none of these is a CF time unit -- and the
            important part is that answering does not raise. `bytes` are the
            one non-`str` input that *is* text, so they moved to the class
            below; bytes that are not valid UTF-8 stay here, being text in
            name only.
        """
        assert is_cf_time_units(units) is False

    def test_the_compliance_checker_reports_rather_than_crashing(self):
        """The consumer that made this matter.

        Test scenario:
            `_check_coordinate_variable` runs over whatever a file contains. A
            coordinate whose `units` is a one-element list is exactly the sort
            of malformation the checker is meant to describe, so it has to
            survive reading it.
        """
        variable = SimpleNamespace(
            name="time",
            attributes={"units": ["days since 1970-01-01"]},
            unit=None,
        )

        issues = _check_coordinate_variable("time", variable)

        assert isinstance(issues, list)
