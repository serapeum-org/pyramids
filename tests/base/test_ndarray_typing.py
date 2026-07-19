"""Tests for the numpy.typing array-return aliases in `pyramids.base.protocols` (ARC-19).

The module re-exports `NDArray` and a dtype-precise `FloatArray`
(`NDArray[np.float64]`) for typed eager array returns. These tests tie the
annotations to runtime reality: the coordinate/dimension helpers annotated
`FloatArray` actually return float64 arrays, and `as_numpy` returns a numpy
array.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.protocols import ArrayLike, FloatArray, NDArray, as_numpy
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestArrayTypingAliases:
    """`NDArray` / `FloatArray` are importable and usable as annotations."""

    def test_aliases_importable(self):
        """Both aliases import from the single typing seed module."""
        assert NDArray is not None
        assert FloatArray is not None

    def test_float_array_is_subscriptable_ndarray(self):
        """`FloatArray` is the float64 specialisation of `NDArray`."""
        assert FloatArray == NDArray[np.float64]

    def test_array_like_still_dtype_agnostic(self):
        """`ArrayLike` remains the eager-or-lazy union (numpy in its args)."""
        assert np.ndarray in getattr(ArrayLike, "__args__", ())


class TestFloatArrayMatchesRuntime:
    """Helpers annotated `FloatArray` actually return float64 arrays."""

    def test_x_dimension_array_is_float64(self):
        """`get_x_lon_dimension_array` returns a 1-D float64 array."""
        arr = Dataset.get_x_lon_dimension_array(0.0, 10.0, 5)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (5,)

    def test_y_dimension_array_is_float64(self):
        """`get_y_lat_dimension_array` returns a 1-D float64 array."""
        arr = Dataset.get_y_lat_dimension_array(100.0, 10.0, 4)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (4,)

    def test_integer_inputs_still_float64(self):
        """Integer pivot/cell_size still yield float64 (locks the FloatArray annotation).

        The `cell_size / 2` true-division forces float regardless of input dtype;
        this is the non-obvious case the float64 annotation depends on.
        """
        x = Dataset.get_x_lon_dimension_array(0, 10, 5)
        y = Dataset.get_y_lat_dimension_array(100, 10, 4)
        assert x.dtype == np.float64, (
            f"int inputs must still give float64 x, got {x.dtype}"
        )
        assert y.dtype == np.float64, (
            f"int inputs must still give float64 y, got {y.dtype}"
        )


class TestAsNumpyReturn:
    """`as_numpy` returns a concrete numpy array for an eager input."""

    def test_as_numpy_returns_ndarray(self):
        """An eager numpy input round-trips to a numpy array."""
        out = as_numpy(np.arange(4))
        assert isinstance(out, np.ndarray)
        assert out.tolist() == [0, 1, 2, 3]
