"""Unit tests for the shared coordinate-matching predicates in ``_coord_match``.

Pure numpy predicates used across ``NetCDFPlot``'s coordinate-resolution paths and
the CF candidate value object. End-to-end coverage lives in
``tests/netcdf/plot/test_plot_coords.py`` and ``tests/netcdf/samples/test_curvilinear_crop.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf import _coord_match

_SHAPE = (4, 5)  # (rows, cols)


class TestSqueezeLeadingAxes:
    """Tests for ``squeeze_leading_axes``."""

    def test_drops_leading_axis_of_3d_matching_slice(self):
        """A 3-D ``(extra, rows, cols)`` array is reduced to its first plane.

        Test scenario:
            ``(2, 4, 5)`` with slice ``(4, 5)`` returns element 0, shape ``(4, 5)``.
        """
        arr = np.arange(2 * 4 * 5).reshape(2, 4, 5)
        result = _coord_match.squeeze_leading_axes(arr, _SHAPE)
        assert result.shape == _SHAPE, f"expected {_SHAPE}, got {result.shape}"
        assert np.array_equal(result, arr[0]), "must return the time-step-0 slice"

    def test_passes_2d_through_unchanged(self):
        """A 2-D array already matching the slice is returned unchanged.

        Test scenario:
            ``(4, 5)`` is returned as-is (identity).
        """
        arr = np.zeros(_SHAPE)
        assert _coord_match.squeeze_leading_axes(arr, _SHAPE) is arr, (
            "2-D must pass through"
        )

    def test_passes_1d_through_unchanged(self):
        """A 1-D coordinate array is returned unchanged.

        Test scenario:
            A 1-D array is not 3-D, so it passes through.
        """
        arr = np.arange(5)
        assert _coord_match.squeeze_leading_axes(arr, _SHAPE) is arr, (
            "1-D must pass through"
        )

    def test_3d_not_matching_trailing_shape_passes_through(self):
        """A 3-D array whose trailing 2 axes don't match the slice is not squeezed.

        Test scenario:
            ``(2, 3, 3)`` with slice ``(4, 5)`` fails the trailing-shape guard, so it
            is returned unchanged.
        """
        arr = np.zeros((2, 3, 3))
        assert _coord_match.squeeze_leading_axes(arr, _SHAPE) is arr, (
            "no-match 3-D passes through"
        )


class TestMatchesXAxis:
    """Tests for ``matches_x_axis`` (1-D cols or 2-D slice)."""

    @pytest.mark.parametrize(
        "arr, expected",
        [
            (np.arange(5), True),
            (np.arange(4), False),
            (np.zeros((4, 5)), True),
            (np.zeros((5, 4)), False),
        ],
    )
    def test_x_axis_shape_rules(self, arr, expected):
        """1-D length must equal cols, or 2-D shape must equal the slice.

        Args:
            arr: Candidate array.
            expected: Whether it can serve as the x axis for ``(4, 5)``.
        """
        assert _coord_match.matches_x_axis(arr, _SHAPE) is expected, (
            f"matches_x_axis({arr.shape}) should be {expected}"
        )


class TestMatchesYAxis:
    """Tests for ``matches_y_axis`` (1-D rows or 2-D slice)."""

    @pytest.mark.parametrize(
        "arr, expected",
        [
            (np.arange(4), True),
            (np.arange(5), False),
            (np.zeros((4, 5)), True),
            (np.zeros((5, 4)), False),
        ],
    )
    def test_y_axis_shape_rules(self, arr, expected):
        """1-D length must equal rows, or 2-D shape must equal the slice.

        Args:
            arr: Candidate array.
            expected: Whether it can serve as the y axis for ``(4, 5)``.
        """
        assert _coord_match.matches_y_axis(arr, _SHAPE) is expected, (
            f"matches_y_axis({arr.shape}) should be {expected}"
        )


class TestCoordShapesMatch:
    """Tests for ``coord_shapes_match``."""

    def test_none_data_shape_returns_false(self):
        """A ``None`` data shape cannot be validated, so returns False.

        Test scenario:
            ``coord_shapes_match(x, y, None)`` is ``False`` without inspecting arrays.
        """
        assert (
            _coord_match.coord_shapes_match(np.arange(5), np.arange(4), None) is False
        ), "None data_shape must yield False"

    def test_matching_pair(self):
        """A 1-D-cols x and 1-D-rows y line up with the slice.

        Test scenario:
            ``(cols,)`` x and ``(rows,)`` y match ``(4, 5)``.
        """
        assert (
            _coord_match.coord_shapes_match(np.arange(5), np.arange(4), _SHAPE) is True
        ), "a cols/rows 1-D pair must match"

    def test_x_mismatch(self):
        """A wrong-length x fails the match.

        Test scenario:
            An x of length rows (not cols) does not match.
        """
        assert (
            _coord_match.coord_shapes_match(np.arange(4), np.arange(4), _SHAPE) is False
        ), "x of length rows must not match the x axis"


class TestLooksLikeXThenY:
    """Tests for ``looks_like_x_then_y`` (lon-then-lat name heuristic)."""

    @pytest.mark.parametrize(
        "x_name, y_name, expected",
        [
            ("lon", "lat", True),
            ("LONGITUDE", "LATITUDE", True),
            ("lat", "lon", False),
            ("xc", "yc", False),
            ("lon", "xc", False),
        ],
    )
    def test_name_heuristic(self, x_name, y_name, expected):
        """x must read as a longitude and y as a latitude (case-insensitive).

        Args:
            x_name: Candidate x name.
            y_name: Candidate y name.
            expected: Whether the names follow the lon/lat convention.
        """
        assert _coord_match.looks_like_x_then_y(x_name, y_name) is expected, (
            f"looks_like_x_then_y({x_name!r}, {y_name!r}) should be {expected}"
        )


class TestValuesWithinLatitude:
    """Tests for ``values_within_latitude``."""

    @pytest.mark.parametrize(
        "values, expected",
        [
            ([-89.0, 0.0, 89.0], True),
            ([0.0, 180.0, 360.0], False),
            ([np.nan, np.nan], False),
            ([-90.5, 90.5], True),
            ([-91.0], False),
        ],
    )
    def test_latitude_bounds(self, values, expected):
        """All finite values must fall within ``[-90.5, 90.5]`` and at least one be finite.

        Args:
            values: The coordinate values.
            expected: Whether the array reads as a latitude.
        """
        arr = np.array(values, dtype=float)
        assert _coord_match.values_within_latitude(arr) is expected, (
            f"values_within_latitude({values}) should be {expected}"
        )
