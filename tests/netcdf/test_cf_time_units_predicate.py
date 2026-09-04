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

import numpy as np
import pytest

from pyramids.netcdf.cf import detect_axis
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

    @pytest.mark.parametrize("units", NOT_TIME_UNITS[:4])
    def test_a_non_time_axis_is_left_numeric(self, units: str):
        """The other half of the invariant, so the fix did not over-reach.

        Args:
            units: A units string that is not a time axis.

        Test scenario:
            Widening the predicate must not start decoding latitudes as dates.
            These values have to come back exactly as passed.
        """
        values = np.array([1.0, 2.0])

        decoded = decode_cf_time(values, units)

        assert decoded is values or np.array_equal(decoded, values)
        assert decoded.dtype.kind == "f"
