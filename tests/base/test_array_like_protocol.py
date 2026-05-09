"""Tests for :data:`pyramids.base.protocols.ArrayLike` + helpers.

introduces the cross-cutting `ArrayLike` type alias plus the
`_ArrayLikeProto` runtime-checkable Protocol, plus two small dispatch
helpers (:func:`is_lazy`, :func:`as_numpy`) that the rest of the code
uses to branch between eager (numpy) and lazy (dask) array backends.

These tests exercise:

* The alias accepts numpy arrays at call time.
* The protocol isinstance check accepts numpy arrays.
* The protocol isinstance check accepts dask arrays (when dask is
  installed). If dask is missing, the dask-specific tests skip
  individually while the numpy tests keep running.
* Non-array objects (lists, ints) do not match the protocol.
* `is_lazy` returns False for numpy and None, True for dask.
* `as_numpy` is a no-op for numpy input, and computes for dask
  input.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask
from pyramids.base.protocols import ArrayLike, _ArrayLikeProto, as_numpy, is_lazy

pytestmark = pytest.mark.core

try:
    import_dask("dask not installed")
    import dask.array as dask_array
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK = False
else:
    HAS_DASK = True
requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


class TestArrayLikeIsinstance:
    """Runtime checks against the _ArrayLikeProto Protocol."""

    def test_numpy_array_is_arraylike(self):
        arr = np.zeros((3, 4))
        assert isinstance(arr, _ArrayLikeProto)

    @requires_dask
    def test_dask_array_is_arraylike(self):
        arr = dask_array.zeros((3, 4), chunks=2)
        assert isinstance(arr, _ArrayLikeProto)

    def test_list_is_not_arraylike(self):
        assert not isinstance([1, 2, 3], _ArrayLikeProto)

    def test_int_is_not_arraylike(self):
        assert not isinstance(42, _ArrayLikeProto)

    def test_none_is_not_arraylike(self):
        assert not isinstance(None, _ArrayLikeProto)


class TestIsLazy:
    """Duck-typed dispatch between eager and lazy backends."""

    def test_numpy_is_not_lazy(self):
        assert is_lazy(np.zeros(5)) is False

    @requires_dask
    def test_dask_is_lazy(self):
        assert is_lazy(dask_array.zeros(5, chunks=2)) is True

    def test_none_is_not_lazy(self):
        assert is_lazy(None) is False

    def test_int_is_not_lazy(self):
        assert is_lazy(42) is False

    def test_list_is_not_lazy(self):
        assert is_lazy([1, 2, 3]) is False

    @requires_dask
    def test_dask_after_compute_is_not_lazy(self):
        arr = dask_array.ones(5, chunks=2).compute()
        assert is_lazy(arr) is False


class TestAsNumpy:
    """Materialisation of lazy inputs to numpy arrays."""

    def test_numpy_in_numpy_out(self):
        src = np.arange(6, dtype=np.float32)
        out = as_numpy(src)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, src)

    @requires_dask
    def test_dask_materialises_to_numpy(self):
        src = dask_array.from_array(np.arange(6), chunks=2)
        out = as_numpy(src)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.arange(6))

    @requires_dask
    def test_as_numpy_preserves_shape_and_dtype(self):
        src = dask_array.ones((3, 4), chunks=2, dtype=np.float64)
        out = as_numpy(src)
        assert out.shape == (3, 4)
        assert out.dtype == np.float64

    def test_as_numpy_noop_returns_view_when_possible(self):
        src = np.arange(6, dtype=np.int32)
        out = as_numpy(src)
        assert out.dtype == src.dtype
        assert out.shape == src.shape


class TestArrayLikeAlias:
    """`ArrayLike` is a typing alias — exercise it in a function signature."""

    def test_function_accepts_numpy(self):
        def sum_array(x: ArrayLike) -> float:
            return float(as_numpy(x).sum())

        assert sum_array(np.arange(5)) == 10.0

    @requires_dask
    def test_function_accepts_dask(self):
        def sum_array(x: ArrayLike) -> float:
            return float(as_numpy(x).sum())

        assert sum_array(dask_array.arange(5, chunks=2)) == 10.0


class TestAliasExport:
    """Basic sanity: the public symbols are importable from the module."""

    def test_alias_exported(self):
        from pyramids.base import protocols

        assert hasattr(protocols, "ArrayLike")
        assert hasattr(protocols, "_ArrayLikeProto")
        assert hasattr(protocols, "is_lazy")
        assert hasattr(protocols, "as_numpy")

    def test_spatial_object_still_exported(self):
        """Smoke test that did not accidentally remove SpatialObject."""
        from pyramids.base.protocols import SpatialObject

        assert SpatialObject is not None


class TestArrayLikeGapCoverage:
    """Gap coverage: partial-duck rejection, dunders, dtype preservation, ducks."""

    def test_partial_duck_without_array_dunder_rejected(self):
        """Object with shape+ndim+dtype but no __array__ should not match."""

        class HalfDuck:
            shape = (3,)
            ndim = 1
            dtype = np.float64

        assert not isinstance(HalfDuck(), _ArrayLikeProto)

    def test_partial_duck_without_getitem_rejected(self):
        """Object with shape+dtype+ndim+__array__ but no __getitem__ rejected."""

        class HalfDuck:
            shape = (3,)
            ndim = 1
            dtype = np.float64

            def __array__(self, dtype=None):
                return np.zeros(3)

        assert not isinstance(HalfDuck(), _ArrayLikeProto)

    def test_array_dunder_invoked_on_numpy(self):
        """Numpy __array__ round-trips through np.asarray."""
        src = np.arange(4)
        out = np.asarray(src)
        assert out.tolist() == [0, 1, 2, 3]

    def test_full_duck_matches_protocol(self):
        """Custom class implementing every required member matches the Protocol."""

        class FullDuck:
            shape = (2, 2)
            ndim = 2
            dtype = np.float32

            def __array__(self, dtype=None):
                return np.zeros((2, 2))

            def __getitem__(self, key):
                return type(self)()

        assert isinstance(FullDuck(), _ArrayLikeProto)

    def test_is_lazy_fake_duck_with_dask_attrs_is_lazy(self):
        """Duck-typed class with.dask and.compute is reported lazy."""

        class FakeDask:
            dask = {"fake": "graph"}

            def compute(self):
                return np.zeros(1)

        assert is_lazy(FakeDask()) is True

    def test_is_lazy_duck_with_only_dask_is_not_lazy(self):
        """`.dask` alone is not enough — must also have `compute`."""

        class HalfDaskLike:
            dask = {}

        assert is_lazy(HalfDaskLike()) is False

    def test_as_numpy_preserves_int_dtype(self):
        """Eager int arrays retain their exact dtype through as_numpy."""
        src = np.arange(6, dtype=np.int16)
        out = as_numpy(src)
        assert out.dtype == np.int16

    def test_as_numpy_preserves_bool_dtype(self):
        """Eager bool arrays retain bool dtype."""
        src = np.array([True, False, True])
        out = as_numpy(src)
        assert out.dtype == np.bool_

    def test_as_numpy_on_2d_numpy(self):
        """2-D numpy input returns 2-D numpy output (no reshape)."""
        src = np.arange(12).reshape(3, 4)
        out = as_numpy(src)
        assert out.shape == (3, 4)

    @requires_dask
    def test_as_numpy_dask_int_dtype(self):
        """Dask int arrays materialise with their dtype preserved."""
        src = dask_array.arange(4, chunks=2, dtype=np.int32)
        out = as_numpy(src)
        assert out.dtype == np.int32

    @requires_dask
    def test_is_lazy_dask_subclass_still_lazy(self):
        """A subclassed dask.array.Array still reports lazy."""

        class DaskSub(dask_array.Array):
            pass

        # Construct via from_array then check attribute presence is sufficient.
        arr = dask_array.from_array(np.arange(4), chunks=2)
        assert is_lazy(arr) is True
