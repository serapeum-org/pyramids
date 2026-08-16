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


@dataclass(frozen=True, eq=False)
class TimeAxis:
    """A datacube's time axis: 1-D coordinate values plus their CF attributes.

    ``eq=False`` (so instances compare/hash by identity): the auto-generated
    ``__eq__`` / ``__hash__`` over an ``np.ndarray`` field would raise (array truth
    value is ambiguous / arrays are unhashable). Nothing compares or hashes a
    ``TimeAxis`` today; identity semantics avoid that footgun.

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

    Examples:
        - Build a positional axis and read its values and CF note:

            ```python
            >>> from pyramids.dataset._cube_time import TimeAxis
            >>> axis = TimeAxis.resolve(None, length=3, collection_time=None)
            >>> axis.values.tolist()
            [0, 1, 2]
            >>> axis.attrs["long_name"]
            'time index'

            ```
    """

    values: np.ndarray
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        time_coords: Sequence[Any] | None,
        length: int,
        collection_time: Sequence[Any] | None,
        *,
        warn_stacklevel: int = 5,
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
            warn_stacklevel: ``stacklevel`` for the non-monotonic / duplicate
                ``time_coords`` warnings, so they point at the caller's own call
                site. The default (5) is tuned for the ``to_netcdf`` chain
                (``user -> to_netcdf -> CubeNetCDFWriter.write -> resolve ->
                _encode -> warn``); a caller at a different depth (e.g. ``to_zarr``)
                passes its own.

        Returns:
            TimeAxis: the resolved, CF-encoded axis.

        Examples:
            - Explicit numeric coordinates pass through unencoded:

                ```python
                >>> from pyramids.dataset._cube_time import TimeAxis
                >>> axis = TimeAxis.resolve([10, 20, 30], length=3, collection_time=None)
                >>> axis.values.tolist()
                [10, 20, 30]
                >>> axis.attrs
                {}

                ```
            - A datetime64 axis is CF-encoded to int64 nanoseconds-since-epoch:

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset._cube_time import TimeAxis
                >>> dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
                >>> axis = TimeAxis.resolve(dates, length=2, collection_time=None)
                >>> axis.values.tolist()
                [1577836800000000000, 1577923200000000000]
                >>> axis.attrs["units"]
                'nanoseconds since 1970-01-01 00:00:00'

                ```
            - With no coordinates, a positional ``0..length-1`` index is emitted:

                ```python
                >>> from pyramids.dataset._cube_time import TimeAxis
                >>> TimeAxis.resolve(None, length=2, collection_time=None).values.tolist()
                [0, 1]

                ```
        """
        if time_coords is None and collection_time is not None:
            # A dated collection (time parsed from file names) exports with its
            # own calendar axis by default; an explicit time_coords overrides it.
            time_coords = collection_time
        if time_coords is not None:
            axis = cls._encode(time_coords, length, warn_stacklevel=warn_stacklevel)
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
    def _encode(
        cls, time_coords: Sequence[Any], length: int, *, warn_stacklevel: int = 5
    ) -> TimeAxis:
        """Validate explicit ``time_coords`` and CF-encode a ``datetime64`` axis.

        Materialises generators, coerces object arrays of datetime / Timestamp to
        ``datetime64``, warns on non-monotonic / duplicate values, and encodes a
        ``datetime64`` axis as ``int64`` "nanoseconds since 1970-01-01" so GDAL's
        multidim writer (which has no datetime type) can round-trip it.

        Args:
            time_coords: Explicit time-axis values (any sized sequence, a
                generator, a ``pd.DatetimeIndex``, or a list of datetime objects).
            length: Number of timesteps the axis must match.
            warn_stacklevel: ``stacklevel`` for the non-monotonic / duplicate
                warnings (see :meth:`resolve`).

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
            # warn_stacklevel is caller-supplied so the warning points at the
            # caller's own call site (default 5 for the to_netcdf chain: warn ->
            # _encode -> resolve -> CubeNetCDFWriter.write -> to_netcdf -> user).
            if not np.array_equal(values, np.sort(values)):
                warnings.warn(
                    "time_coords is not monotonically increasing; some "
                    "downstream CF readers may reorder or refuse the axis",
                    stacklevel=warn_stacklevel,
                )
            if np.unique(values).size != values.size:
                warnings.warn(
                    "time_coords contains duplicate values; downstream "
                    "indexers may pick an arbitrary timestep",
                    stacklevel=warn_stacklevel,
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
