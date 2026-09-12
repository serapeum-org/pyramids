"""Unit tests for the label / nearest matching primitives behind ``NetCDF.sel``.

Pure functions over a synthetic axis — no GDAL, no fixtures. The end-to-end
behaviour on a real file lives in
``tests/netcdf/selection/test_sel_nearest_and_labels.py``.

Style: Google-style docstrings, <=120 char lines, no inline imports,
descriptive assertion messages.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from pyramids.netcdf._label_select import (
    FULL_FORMAT,
    has_label,
    label_format,
    label_indices,
    nearest_indices,
    normalise_label,
    pad_label,
    probe_format,
)

pytestmark = pytest.mark.core

LEVELS = [1000.0, 925.0, 850.0, 700.0]
STEPS = [datetime(2024, 1, 1) + timedelta(hours=hours) for hours in (0, 6, 12, 18)]


def _decode(fmt: str) -> list[str]:
    """Decode the synthetic 6-hourly axis at ``fmt`` — stands in for the CF decoder."""
    return [step.strftime(fmt) for step in STEPS]


class TestNormaliseLabel:
    """A label is accepted in the ISO spellings a user is likely to paste."""

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("2024-01-01T06:00:00Z", "2024-01-01 06:00:00"),
            ("2024-01-01T06:00:00", "2024-01-01 06:00:00"),
            ("  2024-01-01 06:00:00  ", "2024-01-01 06:00:00"),
            ("2024-01", "2024-01"),
        ],
    )
    def test_spellings_collapse_to_one_form(self, given: str, expected: str):
        """Every accepted spelling normalises to the space-separated, zone-free form.

        Test scenario:
            The ISO ``T`` separator, a trailing ``Z``, and surrounding whitespace are
            all removed; a partial label is otherwise untouched.
        """
        assert normalise_label(given) == expected, (
            f"{given!r} -> {normalise_label(given)!r}"
        )


class TestLabelFormat:
    """A label's length picks the strftime format that decodes the axis to match it."""

    @pytest.mark.parametrize(
        ("label", "fmt"),
        [
            ("2024", "%Y"),
            ("2024-01", "%Y-%m"),
            ("2024-01-01", "%Y-%m-%d"),
            ("2024-01-01 06", "%Y-%m-%d %H"),
            ("2024-01-01 06:00", "%Y-%m-%d %H:%M"),
            ("2024-01-01 06:00:00", FULL_FORMAT),
        ],
    )
    def test_each_precision_maps_to_its_format(self, label: str, fmt: str):
        """Each supported precision maps to the format decoding the axis at that precision."""
        assert label_format(label) == fmt, f"{label!r} -> {label_format(label)!r}"

    def test_unsupported_precision_is_rejected(self):
        """A label of an unsupported length raises, listing the precisions that work.

        Test scenario:
            ``"2024-01-01 06:0"`` sits between two supported precisions.
        """
        with pytest.raises(ValueError, match="Write one of"):
            label_format("2024-01-01 06:0")


class TestPadLabel:
    """A partial label names a period, so it pads to that period's first or last instant."""

    @pytest.mark.parametrize(
        ("label", "lower", "upper"),
        [
            ("2024", "2024-01-01 00:00:00", "2024-12-31 23:59:59"),
            ("2024-06", "2024-06-01 00:00:00", "2024-06-31 23:59:59"),
            ("2024-06-15", "2024-06-15 00:00:00", "2024-06-15 23:59:59"),
        ],
    )
    def test_bounds_cover_the_whole_period(self, label: str, lower: str, upper: str):
        """The lower bound is the period's first instant, the upper its last.

        Test scenario:
            The upper template pads to day 31 whatever the month's length — it is only
            ever compared as text, and no real label for that month sorts above it.
        """
        assert pad_label(label, upper=False) == lower, f"lower of {label!r}"
        assert pad_label(label, upper=True) == upper, f"upper of {label!r}"


class TestHasLabel:
    """A selector carrying a string selects by label; anything else by stored value."""

    @pytest.mark.parametrize(
        ("selector", "expected"),
        [
            ("2024-01-01", True),
            (["2024-01-01", "2024-01-02"], True),
            (slice("2024-01-01", None), True),
            (slice(None, "2024-01-01"), True),
            (850.0, False),
            ([1000.0, 850.0], False),
            (slice(500, 1000), False),
            (slice(None, None), False),
        ],
    )
    def test_string_anywhere_means_label_selection(self, selector, expected: bool):
        """A string anywhere in the selector flags it as a label selection."""
        assert has_label(selector) is expected, f"{selector!r} -> {has_label(selector)}"


class TestLabelIndices:
    """A label matches at its own precision, so a partial one takes a whole period."""

    def test_full_precision_label_matches_one_step(self):
        """A fully-qualified label pins a single step."""
        assert label_indices(_decode, "2024-01-01 12:00:00") == [2]

    def test_date_only_label_matches_every_step_that_day(self):
        """A date-only label matches every step inside that day, as xarray's partial indexing does."""
        assert label_indices(_decode, "2024-01-01") == [0, 1, 2, 3]

    def test_year_label_matches_the_whole_year(self):
        """A year-only label matches the whole axis when it all falls in that year."""
        assert label_indices(_decode, "2024") == [0, 1, 2, 3]

    def test_list_unions_its_labels_in_axis_order(self):
        """A list of labels unions their matches and reports them in ascending axis order."""
        selector = ["2024-01-01 18:00:00", "2024-01-01 00:00:00"]
        assert label_indices(_decode, selector) == [0, 3], (
            "union should be axis-ordered"
        )

    def test_list_may_mix_precisions(self):
        """Each label in a list is matched at its own precision, so mixing them is fine."""
        selector = ["2024-01-01 06", "2024-01-01 12:00:00"]
        assert label_indices(_decode, selector) == [1, 2]

    def test_slice_bounds_are_inclusive(self):
        """A label slice takes an inclusive range, padding each bound to its period edge."""
        assert label_indices(
            _decode, slice("2024-01-01 06:00", "2024-01-01 12:00")
        ) == [1, 2]

    def test_open_slice_runs_to_the_axis_end(self):
        """An open bound runs to the corresponding end of the axis."""
        assert label_indices(_decode, slice("2024-01-01 12:00:00", None)) == [2, 3]
        assert label_indices(_decode, slice(None, "2024-01-01 06:00:00")) == [0, 1]

    def test_reversed_slice_bounds_are_normalised(self):
        """Bounds given newest-first still select the range, matching the stored-value path."""
        reversed_bounds = slice("2024-01-01 12:00", "2024-01-01 06:00")
        assert label_indices(_decode, reversed_bounds) == [1, 2]

    def test_label_outside_the_axis_matches_nothing(self):
        """A label the axis does not carry matches nothing rather than raising."""
        assert label_indices(_decode, "2025-01-01") == []

    def test_an_empty_axis_matches_nothing_rather_than_raising(self):
        """A slice against an axis with no decoded labels answers empty.

        Test scenario:
            The open-bound path takes `min(labels)` / `max(labels)`, which used to raise
            a bare "min() iterable argument is empty" on an axis the caller had not
            pre-checked.
        """
        assert label_indices(lambda fmt: [], slice(None, None)) == []
        assert label_indices(lambda fmt: [], "2024-01-01") == []


class TestNearestIndices:
    """``method="nearest"`` snaps a numeric request to the closest coordinate."""

    def test_snaps_to_the_closer_neighbour(self):
        """A value between two coordinates snaps to the closer one."""
        assert nearest_indices(LEVELS, 900.0) == [1], "900 is closer to 925 than to 850"

    def test_exact_value_snaps_to_itself(self):
        """A value already on the axis snaps to itself."""
        assert nearest_indices(LEVELS, 925.0) == [1]

    def test_out_of_range_value_snaps_to_the_end(self):
        """A value beyond either end snaps to that end rather than failing."""
        assert nearest_indices(LEVELS, 10.0) == [3], (
            "below the axis -> its lowest level"
        )
        assert nearest_indices(LEVELS, 5000.0) == [0], (
            "above the axis -> its highest level"
        )

    def test_each_value_in_a_list_snaps_independently(self):
        """Every value in a list snaps on its own, and the result is axis-ordered."""
        assert nearest_indices(LEVELS, [990.0, 710.0]) == [0, 3]

    def test_collisions_are_deduplicated(self):
        """Two requests snapping to the same coordinate yield one index."""
        assert nearest_indices(LEVELS, [995.0, 1005.0]) == [0]

    def test_numpy_scalars_are_numeric(self):
        """A numpy scalar off a coordinate array counts as numeric."""
        assert nearest_indices(list(np.array(LEVELS)), np.float64(900.0)) == [1]

    def test_slice_is_rejected(self):
        """A slice has no nearest value, so it is rejected with a pointer to the fix."""
        with pytest.raises(ValueError, match="does not accept a slice"):
            nearest_indices(LEVELS, slice(700, 1000))

    def test_non_numeric_selector_is_rejected(self):
        """A non-numeric selector cannot be snapped."""
        with pytest.raises(ValueError, match="numeric selector values"):
            nearest_indices(LEVELS, "850")

    def test_non_numeric_axis_is_rejected(self):
        """An axis of non-numeric coordinates cannot be snapped against."""
        with pytest.raises(ValueError, match="numeric coordinate axis"):
            nearest_indices(["a", "b"], 1.0)

    def test_a_fill_value_in_the_axis_is_never_snapped_to(self):
        """A NaN coordinate is skipped rather than winning the distance scan.

        Test scenario:
            NaN compares false against everything, so a plain `min` over the distances
            used to hand back the fill value's slot — the caller asked for the level
            nearest 900 and got the hole.
        """
        assert nearest_indices([float("nan"), 850.0, 1000.0], 900.0) == [1]

    def test_an_all_fill_axis_is_rejected(self):
        """An axis with nothing finite to snap to says so instead of guessing."""
        with pytest.raises(ValueError, match="no finite coordinate"):
            nearest_indices([float("nan"), float("nan")], 900.0)

    def test_a_non_finite_selector_is_rejected(self):
        """A NaN or infinite request has no nearest coordinate."""
        with pytest.raises(ValueError, match="finite selector values"):
            nearest_indices([1000.0, 850.0], float("nan"))

    @pytest.mark.parametrize(
        "axis",
        [[700.0, 850.0, 925.0, 1000.0], [1000.0, 925.0, 850.0, 700.0]],
        ids=["ascending", "descending"],
    )
    def test_a_tie_resolves_to_the_same_coordinate_either_direction(self, axis):
        """An exact midpoint snaps to the smaller coordinate whichever way the axis is stored.

        Test scenario:
            887.5 is equidistant from 850 and 925. Taking the lowest *index* made the
            answer depend on storage order — 850 on an ascending file, 925 on a
            descending one — for the same physical request.
        """
        index = nearest_indices(axis, 887.5)[0]
        assert axis[index] == 850.0, f"tie resolved to {axis[index]}"


class TestLabelShapeRecognition:
    """A label is recognised by its shape, so an arbitrary string is a stored value."""

    @pytest.mark.parametrize("text", ["control", "member01", "hi", "20240101"])
    def test_a_non_date_string_is_not_a_label(self, text: str):
        """A string that is not shaped like a date resolves no probe format.

        Test scenario:
            ``"control"`` is seven characters like ``"2024-01"``, so a length-only test
            classified it as a month and sent an ensemble member's name down the date
            path. Shape decides instead.
        """
        assert probe_format(text) is None, f"{text!r} should not read as a date label"

    @pytest.mark.parametrize(
        "text", ["2024", "2024-01", "2024-01-01", "2024-01-01 06:00:00"]
    )
    def test_a_date_shaped_string_resolves_a_format(self, text: str):
        """Every supported precision still resolves to its format."""
        assert probe_format(text) is not None, f"{text!r} should read as a date label"

    @pytest.mark.parametrize(
        "selector",
        [slice("control", "x"), slice("850", "1000"), ["control", "x"]],
    )
    def test_a_non_date_container_is_not_a_label_selection(self, selector):
        """The shape test governs lists and slices, not only scalars.

        Test scenario:
            The slice arm short-circuited to the full format before consulting the
            shape, so `slice("control", "x")` read as a date range and failed deep
            inside the padding instead of matching stored values.
        """
        assert probe_format(selector) is None, f"{selector!r} should not read as labels"

    @pytest.mark.parametrize("selector", [850.0, [1000.0, 850.0], slice(500, 1000)])
    def test_a_selector_with_no_string_has_no_probe_format(self, selector):
        """A purely numeric selector needs no decode, so it resolves no format.

        Test scenario:
            The engine only probes a selector `has_label` accepted, but the primitive is
            shared and doctested, so it has to answer for a numeric one too.
        """
        assert probe_format(selector) is None

    def test_a_non_date_string_is_rejected_by_label_format(self):
        """``label_format`` refuses it too, so the two agree on what a label is."""
        with pytest.raises(ValueError, match="not a supported date label"):
            label_format("control")


class TestNonStandardCalendars:
    """A label axis from a non-Gregorian CF calendar selects like any other.

    The decoding itself is cftime's; what is exercised here is that nothing in the
    matcher assumes Gregorian month lengths — a ``360_day`` axis has a real 30th of
    February, and ``pad_label``'s day-31 upper bound has to cover it.
    """

    THIRTY_DAY_FEBRUARY = [
        "2024-02-28 00:00:00",
        "2024-02-29 00:00:00",
        "2024-02-30 00:00:00",
        "2024-03-01 00:00:00",
    ]

    def _decode_360(self, fmt: str) -> list[str]:
        """Decode the synthetic 360-day axis at ``fmt`` by truncating the full labels."""
        widths = {"%Y": 4, "%Y-%m": 7, "%Y-%m-%d": 10, FULL_FORMAT: 19}
        return [label[: widths[fmt]] for label in self.THIRTY_DAY_FEBRUARY]

    def test_a_thirtieth_of_february_selects(self):
        """A date that exists only in a 360-day calendar matches exactly."""
        assert label_indices(self._decode_360, "2024-02-30") == [2]

    def test_the_month_covers_its_thirtieth_day(self):
        """A month label covers day 30, which the Gregorian calendar has no February for."""
        assert label_indices(self._decode_360, "2024-02") == [0, 1, 2]

    def test_a_slice_reaching_day_thirty_includes_it(self):
        """A range ending on the 30th keeps it — no Gregorian month length is assumed."""
        assert label_indices(self._decode_360, slice("2024-02-29", "2024-02-30")) == [
            1,
            2,
        ]

    def test_a_month_bound_pads_past_its_last_day(self):
        """A month as the upper bound covers day 30, which `pad_label` pads past."""
        assert label_indices(self._decode_360, slice("2024-02", "2024-03")) == [
            0,
            1,
            2,
            3,
        ]
