"""Tests for ``sel(method="nearest")`` and date-label selection (issue #1125).

Two gaps this covers end to end on real fixtures:

- ``sel`` matched a coordinate exactly, so asking for "the level nearest 100 m"
  meant knowing the axis values up front.
- A CF time axis is stored as raw offsets, so ``sel(time="2024-01-01")`` — the form
  ``get_time_variable()`` hands back, and the form the plotting reference documents —
  found nothing.

Fixtures:
- ``tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc`` — synthetic 4-D
  ``(time=4, pressure_level=3, lat=5, lon=6)``, ``hours since 2024-01-01``, so the
  time axis decodes and the level axis has gaps to snap into.
- ``tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc`` — a COARDS cube whose
  ``units`` do not parse, standing in for an axis with no labels at all.

Style: Google-style docstrings, <=120 char lines, no inline imports,
descriptive assertion messages.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf._plot import NetCDFPlot
from pyramids.netcdf.netcdf import NetCDF
from pyramids.netcdf.plot_options import Selectors

pytestmark = pytest.mark.core

CF_PATH = "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
COARDS_PATH = "tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc"

TIME_VALUES = [0.0, 6.0, 12.0, 18.0]
LEVEL_VALUES = [1000.0, 850.0, 500.0]


@pytest.fixture(scope="module")
def cf_var() -> NetCDF:
    """The ``temperature`` variable of the synthetic 4-D CF fixture."""
    return NetCDF.read_file(CF_PATH).get_variable("temperature")


@pytest.fixture(scope="module")
def coards_var() -> NetCDF:
    """The ``rhum`` variable of the COARDS fixture, whose time units do not parse."""
    return NetCDF.read_file(COARDS_PATH).get_variable("rhum")


class TestSelNearest:
    """``method="nearest"`` snaps a value onto the axis instead of demanding an exact one."""

    def test_value_between_levels_snaps_to_the_closer_one(self, cf_var):
        """A level the cube does not carry snaps to its closest neighbour.

        Test scenario:
            ``sel(pressure_level=900, method="nearest")`` on ``[1000, 850, 500]`` →
            the 850 hPa plane (|900-850| < |900-1000|).
        """
        result = cf_var.sel(pressure_level=900, method="nearest")
        assert result._band_dim_values_map["pressure_level"] == [850.0], (
            f"expected [850.0], got {result._band_dim_values_map['pressure_level']}"
        )

    def test_the_chosen_coordinate_is_readable_off_the_result(self, cf_var):
        """The snapped coordinate is reported through the public accessor, not just internals.

        Test scenario:
            ``sel(..., method="nearest").get_dimension_values("pressure_level")`` → ``[850.]``.
        """
        result = cf_var.sel(pressure_level=900, method="nearest")
        assert_array_equal(
            result.get_dimension_values("pressure_level"),
            np.array([850.0]),
            err_msg="nearest should report which coordinate it chose",
        )

    def test_snapped_data_matches_the_exact_selection(self, cf_var):
        """Snapping to a level reads the same data as selecting that level exactly."""
        snapped = cf_var.sel(pressure_level=900, method="nearest").read_array()
        exact = cf_var.sel(pressure_level=850).read_array()
        assert_array_equal(snapped, exact, err_msg="nearest must not shift the data")

    def test_exact_value_is_unchanged_by_nearest(self, cf_var):
        """A value already on the axis snaps to itself, so ``method`` is a no-op there."""
        result = cf_var.sel(pressure_level=500, method="nearest")
        assert result._band_dim_values_map["pressure_level"] == [500.0]

    def test_list_of_values_snaps_elementwise(self, cf_var):
        """Every value in a list snaps independently and the result keeps axis order."""
        result = cf_var.sel(pressure_level=[990, 520], method="nearest")
        assert result._band_dim_values_map["pressure_level"] == [1000.0, 500.0], (
            f"got {result._band_dim_values_map['pressure_level']}"
        )

    def test_nearest_works_on_the_time_axis_by_raw_offset(self, cf_var):
        """Snapping applies to any numeric axis, the raw CF time offsets included."""
        result = cf_var.sel(time=7, method="nearest")
        assert result._band_dim_values_map["time"] == [6.0], (
            f"got {result._band_dim_values_map['time']}"
        )

    def test_default_stays_exact(self, cf_var):
        """Without ``method`` a missing value still raises — snapping is opt-in."""
        with pytest.raises(ValueError, match="No bands match"):
            cf_var.sel(pressure_level=900)

    def test_slice_selector_is_rejected(self, cf_var):
        """A range has no nearest value, so the combination is refused."""
        with pytest.raises(ValueError, match="does not accept a slice"):
            cf_var.sel(pressure_level=slice(500, 1000), method="nearest")

    def test_date_label_is_rejected(self, cf_var):
        """Snapping a date label is refused, pointing at exact label selection instead."""
        with pytest.raises(ValueError, match="needs a numeric selector"):
            cf_var.sel(time="2024-01-01", method="nearest")

    def test_unknown_method_is_rejected(self, cf_var):
        """Only ``None`` and ``"nearest"`` are accepted, and the error says so."""
        with pytest.raises(ValueError, match="must be None \\(exact\\) or 'nearest'"):
            cf_var.sel(pressure_level=850, method="pad")


class TestSelByDateLabel:
    """A CF time axis is selectable by the labels ``get_time_variable`` reports."""

    def test_full_precision_label_pins_one_step(self, cf_var):
        """A fully-qualified label selects the single step it names.

        Test scenario:
            ``sel(time="2024-01-01 12:00:00")`` on ``hours since 2024-01-01`` → offset 12.
        """
        result = cf_var.sel(time="2024-01-01 12:00:00")
        assert result._band_dim_values_map["time"] == [12.0], (
            f"got {result._band_dim_values_map['time']}"
        )

    def test_iso_spelling_is_accepted(self, cf_var):
        """The ISO ``T`` separator and a trailing ``Z`` resolve the same as the plain form."""
        result = cf_var.sel(time="2024-01-01T12:00:00Z")
        assert result._band_dim_values_map["time"] == [12.0]

    def test_partial_label_takes_the_whole_period(self, cf_var):
        """A date-only label matches every step inside that day.

        Test scenario:
            The fixture's four steps all fall on 2024-01-01, so ``sel(time="2024-01-01")``
            keeps the whole time axis.
        """
        result = cf_var.sel(time="2024-01-01")
        assert result._band_dim_values_map["time"] == TIME_VALUES, (
            f"got {result._band_dim_values_map['time']}"
        )

    def test_label_slice_is_inclusive(self, cf_var):
        """A slice of labels takes an inclusive range of steps."""
        result = cf_var.sel(time=slice("2024-01-01 06:00", "2024-01-01 12:00"))
        assert result._band_dim_values_map["time"] == [6.0, 12.0], (
            f"got {result._band_dim_values_map['time']}"
        )

    def test_list_of_labels_selects_each(self, cf_var):
        """A list of labels selects every step it names, in axis order."""
        result = cf_var.sel(time=["2024-01-01 18:00:00", "2024-01-01 00:00:00"])
        assert result._band_dim_values_map["time"] == [0.0, 18.0]

    def test_label_selection_reads_the_same_data_as_the_raw_offset(self, cf_var):
        """Selecting by label and by stored offset return the same plane."""
        by_label = cf_var.sel(time="2024-01-01 12:00:00").read_array()
        by_offset = cf_var.sel(time=12).read_array()
        assert_array_equal(by_label, by_offset, err_msg="label and offset must agree")

    def test_stored_offsets_still_select(self, cf_var):
        """The raw-offset vocabulary keeps working — labels are an addition, not a swap."""
        result = cf_var.sel(time=18)
        assert result._band_dim_values_map["time"] == [18.0]

    def test_missing_label_reports_the_available_labels(self, cf_var):
        """A label the axis lacks lists the axis' labels, not its raw offsets.

        Test scenario:
            The error a caller sees must be in the vocabulary they selected with.
        """
        with pytest.raises(
            ValueError, match="Available values: \\['2024-01-01 00:00:00'"
        ):
            cf_var.sel(time="2025-06-01")

    def test_axis_without_parseable_units_falls_back_to_stored_values(self, coards_var):
        """An axis whose CF units do not parse has no labels, so the raw vocabulary answers.

        Test scenario:
            The COARDS fixture's ``units`` defeat the time parser, so a label selector
            finds nothing and the error quotes the stored offsets.
        """
        with pytest.raises(ValueError, match="Available values: \\[17549208.0"):
            coards_var.sel(time="2024-01-01")

    def test_unsupported_label_precision_is_rejected(self, cf_var):
        """A label at an unsupported precision raises rather than silently matching nothing."""
        with pytest.raises(ValueError, match="Write one of"):
            cf_var.sel(time="2024-01-01 06:0")


class TestSelectorsMethodPlumbing:
    """``Selectors.method`` reaches the same resolver the eager ``sel`` path uses."""

    def test_default_is_exact(self):
        """``Selectors`` keeps exact matching unless the caller asks for nearest."""
        assert Selectors().method is None

    def test_flat_band_index_snaps_with_nearest(self, cf_var):
        """The lazy render path resolves a snapped selector to the same band as ``sel``.

        Test scenario:
            ``pressure_level=900`` with ``method="nearest"`` → the 850 hPa plane of the
            first time step, which is flat band 1 of the ``(time=4, level=3)`` cube.
        """
        index = NetCDFPlot(cf_var)._flat_band_index(
            cf_var, {"pressure_level": 900}, "nearest"
        )
        assert index == LEVEL_VALUES.index(850.0), f"got band {index}"

    def test_flat_band_index_resolves_a_date_label(self, cf_var):
        """The lazy render path resolves a date label, matching the documented plot example.

        Test scenario:
            ``time="2024-01-01 06:00:00"`` is time index 1, whose first band on a
            ``(time=4, level=3)`` cube is 3.
        """
        index = NetCDFPlot(cf_var)._flat_band_index(
            cf_var, {"time": "2024-01-01 06:00:00"}
        )
        assert index == len(LEVEL_VALUES), f"got band {index}"


class TestSelSelectorVocabularyFallback:
    """A string on an axis that is not a decodable CF time axis stays a stored value."""

    def test_string_on_a_non_time_axis_matches_stored_values(self, cf_var):
        """A string selector on a numeric non-time axis falls through to exact matching.

        Test scenario:
            ``pressure_level`` carries pressure units, not a CF time origin, so there
            are no labels to decode against; the error quotes the stored values.
        """
        with pytest.raises(ValueError, match=r"Available values: \[1000.0"):
            cf_var.sel(pressure_level="850")

    def test_explicit_method_none_is_exact(self, cf_var):
        """Passing ``method=None`` explicitly behaves exactly as omitting it."""
        result = cf_var.sel(pressure_level=850, method=None)
        assert result._band_dim_values_map["pressure_level"] == [850.0]


class TestSelUndecodableCoordinateValues:
    """A coordinate value the CF converter cannot decode must not abort the selection."""

    def test_fill_value_in_the_time_axis_falls_back_to_stored_values(self, cf_var):
        """A NaN in the time axis leaves the stored-value path working.

        Test scenario:
            ``_decode_time_labels`` converts each value outside its own guard, so a
            ``_FillValue`` / NaN entry used to abort `sel` with "cannot convert float NaN
            to integer". Selecting by stored offset must still work, and a label selector
            must degrade to "no bands match" rather than crash.
        """
        holed = cf_var.copy()
        holed._band_dim_values_map = dict(cf_var._band_dim_values_map)
        holed._band_dim_values_map["time"] = [0.0, float("nan"), 12.0, 18.0]
        assert holed.sel(time=12.0)._band_dim_values_map["time"] == [12.0]
        with pytest.raises(ValueError, match="No bands match"):
            holed.sel(time="2024-01-01 12:00:00")

    def test_string_valued_coordinates_still_match_exactly(self, cf_var):
        """A coordinate variable holding date strings keeps matching as it did before.

        Test scenario:
            Some stores carry the dates as strings in the coordinate variable itself. The
            axis then cannot be decoded from its offsets, so the selector has to fall
            through to an exact string match against the stored values — the behaviour
            this file's decoding must not take away.
        """
        stringy = cf_var.copy()
        stringy._band_dim_values_map = dict(cf_var._band_dim_values_map)
        stringy._band_dim_values_map["time"] = [
            "2024-01-13",
            "2024-01-14",
            "2024-01-15",
            "2024-01-16",
        ]
        result = stringy.sel(time="2024-01-15")
        assert result._band_dim_values_map["time"] == ["2024-01-15"], (
            f"got {result._band_dim_values_map['time']}"
        )


class TestGetTimeVariableRoundTrip:
    """A label handed back by ``get_time_variable`` selects the period it names."""

    def test_default_format_label_selects_the_whole_day(self):
        """``get_time_variable()``'s default label is date-only, so it keeps every step that day.

        Test scenario:
            The documented round-trip has to say which precision it round-trips at —
            ``"%Y-%m-%d"`` is a day, and on this 6-hourly fixture that is four steps.
        """
        nc = NetCDF.read_file(CF_PATH)
        var = nc.get_variable("temperature")
        label = nc.get_time_variable("time")[1]
        assert label == "2024-01-01", f"got {label!r}"
        assert var.sel(time=label)._band_dim_values_map["time"] == TIME_VALUES

    def test_full_precision_format_label_pins_one_step(self):
        """Asking ``get_time_variable`` for the finer format gives labels that pin one step."""
        nc = NetCDF.read_file(CF_PATH)
        var = nc.get_variable("temperature")
        labels = nc.get_time_variable("time", "%Y-%m-%d %H:%M:%S")
        assert labels[1] == "2024-01-01 06:00:00", f"got {labels[1]!r}"
        assert var.sel(time=labels[1])._band_dim_values_map["time"] == [6.0]

    def test_every_full_precision_label_round_trips_to_its_own_step(self):
        """Each finer-format label selects exactly the offset it was decoded from."""
        nc = NetCDF.read_file(CF_PATH)
        var = nc.get_variable("temperature")
        labels = nc.get_time_variable("time", "%Y-%m-%d %H:%M:%S")
        for label, offset in zip(labels, TIME_VALUES, strict=True):
            assert var.sel(time=label)._band_dim_values_map["time"] == [offset], (
                f"{label!r} should select {offset}"
            )


class TestSelMixedVocabularySelectors:
    """A selector must be all labels or all stored values, and says so when it is not."""

    @pytest.mark.parametrize(
        "selector",
        [
            ["2024-01-01 06:00:00", 12],
            slice("2024-01-01", 12),
            slice(12, "2024-01-01"),
        ],
    )
    def test_mixed_selector_raises_value_error(self, cf_var, selector):
        """Mixing the two vocabularies raises the documented ``ValueError``.

        Test scenario:
            ``has_label`` is true when any part is a string while the matcher assumes
            every part is, so these used to leak an ``AttributeError`` ('int' object has
            no attribute 'strip') from inside a private helper.
        """
        with pytest.raises(ValueError, match="mixes date labels with stored values"):
            cf_var.sel(time=selector)

    def test_an_open_bound_is_not_a_mixed_selector(self, cf_var):
        """A half-open label slice is still a pure label selection."""
        result = cf_var.sel(time=slice("2024-01-01 12:00:00", None))
        assert result._band_dim_values_map["time"] == [12.0, 18.0]

    def test_a_non_date_string_on_a_time_axis_is_still_a_malformed_label(self, cf_var):
        """On an axis that does decode, a non-label string is a typo, and says so."""
        with pytest.raises(ValueError, match="Write one of"):
            cf_var.sel(time="control")


class TestSelDecodesTheAxisOncePerPrecision:
    """A label selection must not walk the whole coordinate axis more times than it needs."""

    def _count_decodes(self, monkeypatch, var, selector):
        """Run one `sel` and return how many times the axis was decoded."""
        calls = {"n": 0}
        original = NetCDF._decode_time_labels

        def counted(self, *args, **kwargs):
            calls["n"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(NetCDF, "_decode_time_labels", counted)
        var.sel(time=selector)
        return calls["n"]

    def test_a_list_of_labels_decodes_once(self, cf_var, monkeypatch):
        """Three labels of one precision cost one pass, not one per label plus a probe.

        Test scenario:
            The resolver used to decode at full precision to test the axis, again per
            label to match, and the first result was only ever used to build an error
            message that a successful match never raises.
        """
        selector = [
            "2024-01-01 00:00:00",
            "2024-01-01 06:00:00",
            "2024-01-01 12:00:00",
        ]
        assert self._count_decodes(monkeypatch, cf_var, selector) == 1

    def test_a_label_slice_decodes_once(self, cf_var, monkeypatch):
        """A slice compares full-precision labels, so one pass answers it."""
        selector = slice("2024-01-01 00:00:00", "2024-01-01 12:00:00")
        assert self._count_decodes(monkeypatch, cf_var, selector) == 1


class TestSelectionErrorMessages:
    """What a failed selection tells the caller."""

    def test_a_non_numeric_selector_is_not_called_a_date_label(self, cf_var):
        """A string on an axis with no labels is reported as non-numeric, not as a date.

        Test scenario:
            ``pressure_level="850"`` with ``method="nearest"`` used to be told it was a
            date label and advised about partial labels — nonsense for a pressure axis.
        """
        with pytest.raises(ValueError, match="is not a number"):
            cf_var.sel(pressure_level="850", method="nearest")

    def test_a_long_axis_is_elided_in_the_message(self):
        """A miss on a long axis shows both ends and a count, not every value.

        Test scenario:
            The COARDS cube has 12 time steps; a 128k-step cloud axis would otherwise
            put every decoded timestamp into the exception string.
        """
        var = NetCDF.read_file(COARDS_PATH).get_variable("rhum")
        with pytest.raises(ValueError, match=r"\.\.\..*\(12 values\)"):
            var.sel(time=-1)

    def test_a_short_axis_is_listed_in_full(self, cf_var):
        """A short axis still lists every value — eliding only pays off when there are many."""
        with pytest.raises(
            ValueError, match=r"Available values: \[1000.0, 850.0, 500.0\]"
        ):
            cf_var.sel(pressure_level=999)
