"""A text-coordinate band dimension carried through a rebuild (#1181).

The WRF fixture `none__17v__1d1-2d5-3d6-4d5__stag-str.nc` has `SMOIS` on two band
dimensions, `Time` (text stamps like `'2000-01-24_12:00:00'`) and `soil_layers_stag`
(numeric). Reducing over the numeric dimension has to carry the text one through the
rebuild, and `NetCDF.from_array` / `set_variable` used to coerce every carried coordinate
to float64, so the carry raised `could not convert string to float`. Reducing over the
text dimension always worked because it is consumed, not carried.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.netcdf import GeoReference, NetCDF
from pyramids.netcdf.engines.combine import (
    _check_other_dimensions,
    _comparable,
    _stamps,
)
from pyramids.netcdf.engines.variables import _is_text_axis

pytestmark = pytest.mark.core

STORE = (
    Path(__file__).parents[2]
    / "data"
    / "netcdf"
    / "none__17v__1d1-2d5-3d6-4d5__stag-str.nc"
)
GEO = (30.0, 0.5, 0, 35.0, 0, -0.5)
TIME_STAMPS = [
    "2000-01-24_12:00:00",
    "2000-01-24_13:00:00",
    "2000-01-24_14:00:00",
]


def _smois() -> NetCDF:
    """The `SMOIS` variable: band dims `(Time, soil_layers_stag)`, `Time` text.

    Returns:
        NetCDF: The variable.
    """
    return NetCDF.read_file(str(STORE)).get_variable("SMOIS")


def _carried_time(result: NetCDF) -> list:
    """The `Time` coordinates a rebuilt result carries.

    Args:
        result: The rebuilt variable.

    Returns:
        list: The stamps, as strings.
    """
    return [str(one) for one in result._band_dim_values_map.get("Time", [])]


class TestReducingOverTheNumericDimensionCarriesTheTextAxis:
    """#1181 — the reported failure and its mirror that always worked."""

    def test_reducing_over_the_text_dimension_consumes_it(self):
        """Reducing over `Time` never carried it, so it always worked."""
        assert _smois().reduce("Time", how="mean").shape == (5, 60, 73)

    def test_reducing_over_the_numeric_dimension_now_works(self):
        """The reported bug: carrying the text `Time` raised on the float coercion."""
        assert _smois().reduce("soil_layers_stag", how="mean").shape == (3, 60, 73)

    def test_the_text_stamps_come_through_unchanged(self):
        """The carried coordinates are the source's text, not floats or indices."""
        result = _smois().reduce("soil_layers_stag", how="mean")
        assert _carried_time(result) == TIME_STAMPS


class TestEveryCarryingMemberKeepsTheTextAxis:
    """Every member that rebuilds along the numeric dimension carries `Time` intact."""

    @pytest.mark.parametrize(
        ("member", "call", "length"),
        [
            ("coarsen", lambda v: v.coarsen("soil_layers_stag", 2, boundary="trim"), 3),
            ("rolling", lambda v: v.rolling("soil_layers_stag", 2), 3),
            ("cumsum", lambda v: v.cumsum("soil_layers_stag"), 3),
            ("diff", lambda v: v.diff("soil_layers_stag"), 3),
            ("shift", lambda v: v.shift("soil_layers_stag", 1), 3),
        ],
    )
    def test_the_member_carries_the_text_time_axis(self, member, call, length):
        """The result keeps `Time` as its text stamps.

        Args:
            member: The member under test.
            call: How to call it on the variable.
            length: The expected `Time` length (unchanged by a soil-axis operation).
        """
        result = call(_smois())
        carried = _carried_time(result)
        assert carried == TIME_STAMPS[:length], (
            f"{member} lost the text axis: {carried}"
        )


class TestMergeComparesTextCoordinatesWithoutCoercion:
    """The `combine._stamps` half of #1181."""

    def test_stamps_reads_a_text_axis_as_strings(self):
        """`_stamps` no longer coerces to float, so it does not raise on WRF's `Time`."""
        assert _stamps(_smois(), "Time") == tuple(TIME_STAMPS)

    def test_merge_of_two_variables_sharing_the_text_time_dim(self):
        """Merge compares the shared `Time` via `_stamps`; the coercion raised before.

        Test scenario:
            `IVGTYP` (`Time`) and `SMOIS` (`Time`, `soil_layers_stag`) share the text
            `Time` dimension on the same grid, so `merge` compares their stamps and then
            builds each variable — every step of which coerced the text to float.
        """
        store = NetCDF.read_file(str(STORE))
        merged = NetCDF.merge(
            [store.get_variable("IVGTYP"), store.get_variable("SMOIS")]
        )
        assert sorted(merged.variable_names) == ["IVGTYP", "SMOIS"]
        assert _carried_time(merged.get_variable("SMOIS")) == TIME_STAMPS


class TestSetVariableCreatesATextBandDimension:
    """`set_variable` writes a text band axis via `create_main_dimension` (#1181)."""

    def test_a_text_band_axis_is_stored_as_strings(self):
        """Storing a raster on a text `Time` axis carries the stamps, not floats.

        Test scenario:
            An in-memory container gains a three-band variable whose band dimension is
            WRF's text `Time`; `create_main_dimension` used to cast the stamps to float64
            and raise `could not convert string to float`, so the write now goes through
            the string channel instead.
        """
        geo = GeoReference(geo=GEO)
        base = NetCDF.from_array(
            arr=np.zeros((3, 4), dtype="float32"),
            geo_ref=geo,
            variable_name="base",
            path=None,
        )
        raster = Dataset.from_array(
            np.arange(3 * 3 * 4, dtype="float32").reshape(3, 3, 4), geo_ref=geo
        )
        base.set_variable(
            "SM", raster, band_dim_name="Time", band_dim_values=TIME_STAMPS
        )
        assert _carried_time(base.get_variable("SM")) == TIME_STAMPS

    def test_a_numeric_band_axis_still_writes_through_the_float_channel(self):
        """A numeric band axis keeps the non-string write path, unchanged by the fix.

        Test scenario:
            The same `set_variable` call with numeric band coordinates stores them as
            floats, confirming the string branch is taken only for a text axis.
        """
        geo = GeoReference(geo=GEO)
        base = NetCDF.from_array(
            arr=np.zeros((3, 4), dtype="float32"),
            geo_ref=geo,
            variable_name="base",
            path=None,
        )
        raster = Dataset.from_array(
            np.arange(3 * 3 * 4, dtype="float32").reshape(3, 3, 4), geo_ref=geo
        )
        base.set_variable(
            "SM", raster, band_dim_name="level", band_dim_values=[1.0, 2.0, 3.0]
        )
        carried = [
            float(one) for one in base.get_variable("SM")._band_dim_values_map["level"]
        ]
        assert carried == [1.0, 2.0, 3.0]

    def test_an_object_dtype_text_axis_is_stored_as_strings(self):
        """A `pandas.Index` / xarray / `dtype=object` text axis is text, not float.

        Test scenario:
            The stamps arrive as an object array (`kind == 'O'`), the shape a
            `pandas.Index` of strings and an xarray string coordinate's `.values` take.
            Classifying only `U` / `S` left this falling through to float64, reproducing
            `could not convert string to float` — the exact defect #1181 fixes.
        """
        geo = GeoReference(geo=GEO)
        base = NetCDF.from_array(
            arr=np.zeros((3, 4), dtype="float32"),
            geo_ref=geo,
            variable_name="base",
            path=None,
        )
        raster = Dataset.from_array(
            np.arange(3 * 3 * 4, dtype="float32").reshape(3, 3, 4), geo_ref=geo
        )
        stamps = np.array(TIME_STAMPS, dtype=object)
        base.set_variable("SM", raster, band_dim_name="Time", band_dim_values=stamps)
        assert _carried_time(base.get_variable("SM")) == TIME_STAMPS

    def test_an_integer_band_axis_keeps_its_integer_type(self):
        """An integer band axis is stored as native `int`, not widened to float (#1181 L1).

        Test scenario:
            Routing these paths through `_coordinate_dtype` preserves an integer axis'
            own type, matching `_create_extra_dimensions` — an int64 epoch stamp loses
            precision as float64. `origin/main` stored `[1.0, 2.0, 3.0]` here.
        """
        geo = GeoReference(geo=GEO)
        base = NetCDF.from_array(
            arr=np.zeros((3, 4), dtype="float32"),
            geo_ref=geo,
            variable_name="base",
            path=None,
        )
        raster = Dataset.from_array(
            np.arange(3 * 3 * 4, dtype="float32").reshape(3, 3, 4), geo_ref=geo
        )
        base.set_variable(
            "SM", raster, band_dim_name="level", band_dim_values=[1, 2, 3]
        )
        carried = list(base.get_variable("SM")._band_dim_values_map["level"])
        assert carried == [1, 2, 3]
        assert all(isinstance(one, (int, np.integer)) for one in carried), carried


class TestConcatComparesTheTextOtherDimension:
    """concat's `_check_other_dimensions` compares a text 'other' axis via `_comparable`."""

    def test_an_agreeing_text_other_dimension_passes(self):
        """Two cubes joined along the numeric axis agree on the text `Time`, so it lines up.

        Test scenario:
            Joining along `soil_layers_stag`, `Time` (text) is the *other* dimension, and
            `_check_other_dimensions` compares it through `_comparable`; identical stamps
            must not raise.
        """
        assert (
            _check_other_dimensions([_smois(), _smois()], "soil_layers_stag", "SMOIS")
            is None
        )

    def test_a_mismatched_text_other_dimension_is_refused(self):
        """Different text stamps on the other axis are refused, not silently joined.

        Test scenario:
            The comparison ran through `float(one)` before and raised on the text; it now
            compares the strings and refuses a genuine mismatch with a readable message.
        """
        other = _smois()
        other._band_dim_values_map = dict(other._band_dim_values_map)
        other._band_dim_values_map["Time"] = ["1999-01-01_00:00:00"] * 3
        with pytest.raises(ValueError, match="agree on every dimension"):
            _check_other_dimensions([_smois(), other], "soil_layers_stag", "SMOIS")


class TestClassifyingACoordinateAxis:
    """`_is_text_axis` decides which axes are stored as strings (#1181)."""

    @pytest.mark.parametrize(
        ("values", "text"),
        [
            (np.array(["a", "b"]), True),
            (np.array(["a", "b"], dtype=object), True),
            (np.array([b"a", b"b"]), True),
            (np.array([1, 2.0], dtype=object), False),
            (np.array([1.0, 2.0]), False),
            (np.array([1, 2]), False),
            (np.array([], dtype=object), False),
        ],
        ids=[
            "unicode",
            "object-str",
            "bytes",
            "object-num",
            "float",
            "int",
            "empty-obj",
        ],
    )
    def test_only_text_axes_are_classified_as_text(self, values, text):
        """A numeric or empty axis is never treated as text.

        Args:
            values: The coordinate array under test.
            text: Whether it should be classified as text.
        """
        assert _is_text_axis(values) is text


class TestTheComparableHelper:
    """`_comparable` carries values as they are, unwrapping NumPy scalars."""

    def test_text_values_stay_text(self):
        """A text axis compares as strings, where `float()` raised."""
        assert _comparable(np.array(["a", "b"])) == ("a", "b")

    def test_numeric_values_are_plain_python_scalars(self):
        """A numeric axis unwraps to Python `float` / `int`, so equality is clean."""
        assert _comparable(np.array([1.0, 2.0])) == (1.0, 2.0)
        assert _comparable(np.array([1, 2])) == (1, 2)

    def test_plain_python_values_pass_through_without_item(self):
        """A plain list has no `.item()`, so its scalars are carried as they are."""
        assert _comparable(["a", 2, 3.5]) == ("a", 2, 3.5)
