"""Unit tests for the affine unit-conversion helpers in :mod:`pyramids.dataset.ops.units`."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset.ops.units import (
    _AFFINE,
    convert_array,
    supported_conversions,
)

pytestmark = pytest.mark.core


class TestSupportedConversions:
    """Tests for :func:`pyramids.dataset.ops.units.supported_conversions`."""

    def test_returns_sorted_pairs(self):
        """supported_conversions returns the affine table keys, sorted.

        Test scenario:
            The returned list equals the sorted ``_AFFINE`` keys, so callers can
            introspect exactly which conversions are available.
        """
        result = supported_conversions()
        assert result == sorted(_AFFINE.keys()), f"Unexpected pairs: {result}"

    @pytest.mark.parametrize(
        "pair",
        [("K", "celsius"), ("Pa", "hPa"), ("m s-1", "knots"), ("m", "mm")],
    )
    def test_known_pairs_present(self, pair):
        """supported_conversions advertises every seeded conversion direction.

        Args:
            pair: A ``(source, target)`` tuple expected in the table.

        Test scenario:
            Each documented forward conversion appears in the returned list.
        """
        assert pair in supported_conversions(), f"{pair} missing from table"


class TestConvertArray:
    """Tests for :func:`pyramids.dataset.ops.units.convert_array`."""

    @pytest.mark.parametrize(
        "values, source, target, expected",
        [
            ([273.15, 283.15, 303.15], "K", "celsius", [0.0, 10.0, 30.0]),
            ([0.0, 10.0, 30.0], "celsius", "K", [273.15, 283.15, 303.15]),
            ([100.0, 200.0], "Pa", "hPa", [1.0, 2.0]),
            ([1.0, 2.0], "hPa", "Pa", [100.0, 200.0]),
            ([1.0], "m", "mm", [1000.0]),
            ([1000.0], "mm", "m", [1.0]),
        ],
    )
    def test_affine_conversions(self, values, source, target, expected):
        """convert_array applies the correct affine transform per known pair.

        Args:
            values: Input values to convert.
            source: Source unit label.
            target: Target unit label.
            expected: Hand-computed converted values.

        Test scenario:
            Each supported pair maps inputs to the analytically expected outputs
            within floating-point tolerance.
        """
        result = convert_array(np.array(values), source, target)
        np.testing.assert_allclose(
            result, expected, rtol=1e-6, err_msg=f"{source}->{target} wrong"
        )

    def test_speed_conversion_value(self):
        """convert_array converts m/s to knots with the documented factor.

        Test scenario:
            1 m/s equals ~1.943844 knots; check the scalar factor explicitly.
        """
        result = convert_array(np.array([1.0]), "m s-1", "knots")
        assert result[0] == pytest.approx(1.943844), f"Got {result[0]}"

    def test_knots_roundtrip(self):
        """convert_array round-trips m/s -> knots -> m/s back to the original.

        Test scenario:
            Converting forward then backward recovers the source values, proving
            the reverse factor is the exact inverse.
        """
        original = np.array([0.0, 5.0, 12.5])
        knots = convert_array(original, "m s-1", "knots")
        back = convert_array(knots, "knots", "m s-1")
        np.testing.assert_allclose(back, original, rtol=1e-6, err_msg="roundtrip drift")

    def test_same_unit_is_noop_identity(self):
        """convert_array returns the input object unchanged when source == target.

        Test scenario:
            With matching units no arithmetic is performed; the same array object is
            returned (no copy) and values are untouched.
        """
        arr = np.array([1.0, 2.0, 3.0])
        result = convert_array(arr, "K", "K")
        assert result is arr, "Same-unit conversion should return the input object"

    def test_empty_source_unit_raises(self):
        """convert_array rejects an empty source unit.

        Test scenario:
            A band with no recorded unit cannot be converted; ValueError mentions
            'no source unit'.
        """
        arr = np.array([1.0])
        with pytest.raises(ValueError, match="no source unit") as exc:
            convert_array(arr, "", "celsius")
        assert "no source unit" in str(exc.value), f"Unexpected: {exc.value}"

    def test_unknown_pair_raises(self):
        """convert_array rejects an unsupported (source, target) pair.

        Test scenario:
            A target with no table entry raises ValueError listing the supported
            pairs.
        """
        arr = np.array([1.0])
        with pytest.raises(ValueError, match="No unit conversion") as exc:
            convert_array(arr, "K", "furlongs")
        msg = str(exc.value)
        assert "No unit conversion" in msg, f"Unexpected: {msg}"
        assert "Supported pairs" in msg, f"Message should list supported pairs: {msg}"

    def test_does_not_mutate_input(self):
        """convert_array leaves the input array unmodified for real conversions.

        Test scenario:
            After a K->celsius conversion the original array still holds its
            Kelvin values (the function returns a new array).
        """
        arr = np.array([273.15, 283.15])
        snapshot = arr.copy()
        convert_array(arr, "K", "celsius")
        np.testing.assert_array_equal(arr, snapshot, err_msg="input was mutated")
