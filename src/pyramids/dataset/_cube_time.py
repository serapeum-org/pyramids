"""``TimeAxis`` — a resolved, CF-encoded time axis for datacube writers.

A small value object bundling a cube's 1-D time-coordinate values with their CF
attributes, plus the resolution / validation / encoding logic that produces them.
It is shared by the collection cube writers (``to_netcdf`` and, prospectively,
``to_zarr`` / ``aggregate``) so the time-axis handling lives in one place instead
of being inlined into each writer.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimeAxis:
    """A datacube's time axis: 1-D coordinate values plus their CF attributes.

    ``values`` is the encoded axis — a positional integer index, a numeric axis
    passed through as-is, or a ``datetime64`` axis CF-encoded to ``int64``
    nanoseconds-since-epoch. ``attrs`` are the matching CF coordinate attributes
    (``units`` / ``calendar`` for an encoded datetime axis, or the positional-index
    ``note``).

    The two fields are one concept: they are always produced together and always
    consumed together by the schema builder, so they belong in one object rather
    than as a ``(values, attrs)`` pair threaded through every writer.

    Attributes:
        values: 1-D array of time-coordinate values.
        attrs: CF attributes for the time coordinate. Defaults to empty.
    """

    values: np.ndarray
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        time_coords: Sequence[Any] | None,
        length: int,
        collection_time: Sequence[Any] | None,
    ) -> TimeAxis:
        """Resolve a cube's time axis to a :class:`TimeAxis`.

        Explicit ``time_coords`` win; otherwise the collection's own dated axis is
        used when it has one; otherwise a positional ``0..length-1`` integer index
        is emitted (marked as positional, not calendar).

        Args:
            time_coords: Explicit time-axis values, or ``None`` to auto-resolve.
            length: Number of timesteps the axis must have.
            collection_time: The collection's own time axis (used when
                ``time_coords`` is ``None`` and the collection is dated), else
                ``None``.

        Returns:
            TimeAxis: the resolved, CF-encoded axis.
        """
        if time_coords is None and collection_time is not None:
            # A dated collection (time parsed from file names) exports with its
            # own calendar axis by default; an explicit time_coords overrides it.
            time_coords = collection_time
        if time_coords is not None:
            axis = cls._encode(time_coords, length)
        else:
            axis = cls(
                np.arange(length, dtype="int64"),
                {
                    "long_name": "time index",
                    "note": "positional index, not a calendar time",
                },
            )
        return axis

    @classmethod
    def _encode(cls, time_coords: Sequence[Any], length: int) -> TimeAxis:
        """Validate explicit ``time_coords`` and CF-encode a ``datetime64`` axis.

        Materialises generators, coerces object arrays of datetime / Timestamp to
        ``datetime64``, warns on non-monotonic / duplicate values, and encodes a
        ``datetime64`` axis as ``int64`` "nanoseconds since 1970-01-01" so GDAL's
        multidim writer (which has no datetime type) can round-trip it.

        Args:
            time_coords: Explicit time-axis values (any sized sequence, a
                generator, a ``pd.DatetimeIndex``, or a list of datetime objects).
            length: Number of timesteps the axis must match.

        Returns:
            TimeAxis: the encoded axis.

        Raises:
            ValueError: When ``len(time_coords) != length``.
        """
        # Materialise generators / iterators up front so np.asarray gets a sized
        # sequence (an iterator yields a 0-d object array, tripping a cryptic
        # IndexError below).
        if not hasattr(time_coords, "__len__"):
            time_coords = list(time_coords)
        values = np.asarray(time_coords)
        if values.dtype.kind == "O":
            # pd.DatetimeIndex → datetime64 via asarray, but lists of datetime /
            # Timestamp objects come through as dtype=object; coerce so the
            # datetime branch below picks them up.
            try:
                values = np.asarray(values, dtype="datetime64[ns]")
            except (TypeError, ValueError):
                pass
        if values.shape[0] != length:
            raise ValueError(
                f"time_coords has {values.shape[0]} entries but "
                f"the collection has {length} timesteps"
            )
        attrs: dict[str, Any] = {}
        if values.shape[0] > 1 and values.dtype.kind in "iufM":
            # stacklevel=5: warn -> _encode -> resolve -> CubeNetCDFWriter.write ->
            # DatasetCollection.to_netcdf -> user's call site, so the warning is
            # attributed to the caller of to_netcdf rather than to pyramids.
            if not np.array_equal(values, np.sort(values)):
                warnings.warn(
                    "time_coords is not monotonically increasing; some "
                    "downstream CF readers may reorder or refuse the axis",
                    stacklevel=5,
                )
            if np.unique(values).size != values.size:
                warnings.warn(
                    "time_coords contains duplicate values; downstream "
                    "indexers may pick an arbitrary timestep",
                    stacklevel=5,
                )
        if values.dtype.kind == "M":
            # GDAL's multidim writer has no native datetime64 type; encode as an
            # int64 offset with CF `units` so a CF-aware reader can decode it back
            # to a calendar axis. Nanosecond resolution keeps the datetime64[ns]
            # round-trip lossless.
            epoch = np.datetime64("1970-01-01", "ns")
            values = (values.astype("datetime64[ns]") - epoch).astype("int64")
            attrs["units"] = "nanoseconds since 1970-01-01 00:00:00"
            attrs["calendar"] = "proleptic_gregorian"
        return cls(values, attrs)
