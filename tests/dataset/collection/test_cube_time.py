"""Unit tests for the :class:`pyramids.dataset._cube_time.TimeAxis` value object.

``TimeAxis`` bundles a datacube's resolved time-coordinate values with their CF
attributes and owns the resolution / validation / CF-encoding logic. These tests
exercise it in isolation (pure numpy, no NetCDF I/O); the end-to-end round-trip
through ``to_netcdf`` is covered by ``tests/dataset/collection/test_to_netcdf.py``.
"""

from __future__ import annotations

import datetime
import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyramids.dataset._cube_time import TimeAxis

_EPOCH = np.datetime64("1970-01-01", "ns")


class TestTimeAxis:
    """Tests for the ``TimeAxis`` value object and its class methods."""

    def test_default_attrs_is_empty_dict(self):
        """A TimeAxis built with only values gets an empty attrs dict.

        Test scenario:
            The ``attrs`` field defaults to ``{}`` via ``default_factory``.
        """
        axis = TimeAxis(np.arange(3))
        assert axis.attrs == {}, f"expected empty attrs, got {axis.attrs}"

    def test_is_frozen(self):
        """TimeAxis is immutable — assigning a field raises FrozenInstanceError.

        Test scenario:
            ``frozen=True`` dataclass rejects attribute assignment.
        """
        axis = TimeAxis(np.arange(3), {"units": "x"})
        with pytest.raises(FrozenInstanceError):
            axis.values = np.arange(4)

    def test_resolve_explicit_numeric_passthrough(self):
        """resolve() with explicit numeric coords returns them unencoded.

        Test scenario:
            Integer ``time_coords`` are passed through as values with empty attrs
            (no datetime encoding, no positional-index note).
        """
        axis = TimeAxis.resolve([10, 20, 30], length=3, collection_time=None)
        assert np.array_equal(axis.values, [10, 20, 30]), (
            f"numeric values must pass through, got {axis.values}"
        )
        assert axis.attrs == {}, f"numeric axis must have no CF attrs, got {axis.attrs}"

    def test_resolve_float_passthrough(self):
        """resolve() passes a float axis through with no CF attrs.

        Test scenario:
            Float ``time_coords`` are numeric and are not CF-encoded.
        """
        axis = TimeAxis.resolve([0.5, 1.5], length=2, collection_time=None)
        assert axis.values.dtype.kind == "f", (
            f"expected float axis, got {axis.values.dtype}"
        )
        assert axis.attrs == {}, "float axis must carry no CF attrs"

    def test_resolve_none_falls_back_to_collection_time(self):
        """resolve() uses the collection's own axis when time_coords is None.

        Test scenario:
            ``time_coords=None`` with a dated ``collection_time`` encodes the
            collection axis (here a numeric one, passed through).
        """
        axis = TimeAxis.resolve(None, length=2, collection_time=[7, 8])
        assert np.array_equal(axis.values, [7, 8]), (
            f"must fall back to collection_time, got {axis.values}"
        )

    def test_resolve_explicit_overrides_collection_time(self):
        """resolve() prefers explicit time_coords over the collection axis.

        Test scenario:
            When both are given, ``time_coords`` wins and ``collection_time`` is
            ignored.
        """
        axis = TimeAxis.resolve([1, 2], length=2, collection_time=[100, 200])
        assert np.array_equal(axis.values, [1, 2]), (
            f"explicit coords must override collection_time, got {axis.values}"
        )

    def test_resolve_positional_index_when_all_none(self):
        """resolve() emits a positional index when nothing is supplied.

        Test scenario:
            ``time_coords`` and ``collection_time`` both None => a ``0..length-1``
            int64 index with the positional-index note.
        """
        axis = TimeAxis.resolve(None, length=4, collection_time=None)
        assert np.array_equal(axis.values, [0, 1, 2, 3]), (
            f"expected positional index, got {axis.values}"
        )
        assert axis.values.dtype == np.dtype("int64"), "positional index must be int64"
        assert axis.attrs["long_name"] == "time index", "missing positional long_name"
        assert "positional index" in axis.attrs["note"], "missing positional note"

    def test_encode_length_mismatch_raises(self):
        """_encode() raises ValueError when the coord count != length.

        Test scenario:
            2 coords for a 3-timestep cube must raise, naming both counts.
        """
        with pytest.raises(
            ValueError, match="2 entries but the collection has 3"
        ) as exc:
            TimeAxis._encode([1, 2], length=3)
        assert "3 timesteps" in str(exc.value), f"unexpected message: {exc.value}"

    def test_encode_materialises_generator(self):
        """_encode() materialises a generator (no __len__) before np.asarray.

        Test scenario:
            A generator has no ``__len__``; it must be listed so ``np.asarray``
            gets a sized 1-D sequence rather than a 0-d object array.
        """
        axis = TimeAxis._encode((v for v in [3, 4, 5]), length=3)
        assert np.array_equal(axis.values, [3, 4, 5]), (
            f"generator must be materialised to a 1-D axis, got {axis.values}"
        )

    def test_encode_datetime64_to_cf_int64(self):
        """_encode() CF-encodes a datetime64 axis to int64 ns-since-epoch.

        Test scenario:
            A ``datetime64[ns]`` axis becomes an ``int64`` offset with CF
            ``units`` / ``calendar`` attributes so GDAL can round-trip it.
        """
        dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
        axis = TimeAxis._encode(dates, length=2)
        expected = (dates - _EPOCH).astype("int64")
        assert np.array_equal(axis.values, expected), (
            f"datetime64 must encode to int64 ns offsets, got {axis.values}"
        )
        assert axis.values.dtype == np.dtype("int64"), "encoded values must be int64"
        assert axis.attrs["units"] == "nanoseconds since 1970-01-01 00:00:00", (
            f"wrong CF units, got {axis.attrs.get('units')}"
        )
        assert axis.attrs["calendar"] == "proleptic_gregorian", "wrong CF calendar"

    def test_encode_object_datetimes_are_coerced(self):
        """_encode() coerces a list of datetime objects to a CF-encoded axis.

        Test scenario:
            ``[datetime, datetime]`` arrives as dtype=object; it is coerced to
            ``datetime64`` and then encoded to int64 with CF attrs.
        """
        coords = [datetime.datetime(2021, 6, 1), datetime.datetime(2021, 6, 2)]
        axis = TimeAxis._encode(coords, length=2)
        assert axis.values.dtype == np.dtype("int64"), (
            f"object datetimes must coerce then encode to int64, got {axis.values.dtype}"
        )
        assert axis.attrs["units"].startswith("nanoseconds since"), "missing CF units"

    def test_encode_uncoercible_object_passes_through(self):
        """_encode() leaves a non-datetime object array unencoded (coercion swallowed).

        Test scenario:
            Strings cannot coerce to ``datetime64``; the ``TypeError``/``ValueError``
            is swallowed, the axis stays object dtype, and no CF attrs are added.
        """
        axis = TimeAxis._encode(["a", "b"], length=2)
        assert axis.values.dtype.kind == "U" or axis.values.dtype.kind == "O", (
            f"non-datetime strings must not be CF-encoded, got {axis.values.dtype}"
        )
        assert axis.attrs == {}, (
            f"uncoercible axis must carry no CF attrs, got {axis.attrs}"
        )

    def test_encode_warns_on_non_monotonic(self):
        """_encode() warns when the axis is not monotonically increasing.

        Test scenario:
            Descending integer coords trigger the non-monotonic warning.
        """
        with pytest.warns(UserWarning, match="not monotonically increasing"):
            TimeAxis._encode([3, 1, 2], length=3)

    def test_encode_warns_on_duplicates(self):
        """_encode() warns when the axis contains duplicate values.

        Test scenario:
            A repeated value triggers the duplicate-values warning.
        """
        with pytest.warns(UserWarning, match="duplicate values"):
            TimeAxis._encode([1, 1, 2], length=3)

    def test_encode_single_value_does_not_warn(self):
        """_encode() skips the order/duplicate checks for a length-1 axis.

        Test scenario:
            The ``shape[0] > 1`` guard means a single coord never warns; with the
            filter set to error, any warning would fail the test.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            axis = TimeAxis._encode([5], length=1)
        assert np.array_equal(axis.values, [5]), "single-value axis must pass through"
