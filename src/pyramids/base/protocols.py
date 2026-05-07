"""Structural-typing protocols shared across the pyramids package.

This module exposes two cross-cutting structural types:

* :class:`SpatialObject` — the surface shared by
  :class:`pyramids.dataset.Dataset` (raster) and
  :class:`pyramids.feature.FeatureCollection` (vector), so callers can
  write generic utilities that accept either without importing both
  concrete classes (and without creating import cycles).
* :class:`ArrayLike` — the structural type matching both
  :class:`numpy.ndarray` and :class:`dask.array.Array`, used to annotate
  array-returning methods that may be either eager or lazy.

The module also exports two small dispatch helpers — :func:`is_lazy` and
:func:`as_numpy` — so the rest of the codebase has a single place to
branch between the eager and lazy paths.

Importing this module does **not** import `dask`; the dask reference
is string-forwarded via :data:`typing.TYPE_CHECKING`, so this file is
cheap to import in environments where the `[lazy]` extra is not
installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    import dask.array as da  # noqa: F401


# Type alias covering both numpy arrays and (optionally-installed) dask arrays.
# String-forwarded so importing `pyramids.base.protocols` never triggers a
# dask import. Use this alias for function signatures that accept or return
# either backend; use the :class:`_ArrayLikeProto` Protocol below for runtime
# isinstance checks.
ArrayLike = Union[np.ndarray, "da.Array"]


@runtime_checkable
class SpatialObject(Protocol):
    """Minimum surface shared by pyramids raster and vector objects.

    Both :class:`pyramids.dataset.Dataset` (raster) and
    :class:`pyramids.feature.FeatureCollection` (vector) implement
    this protocol, so callers can write generic geospatial utilities
    that accept either.

    Attributes / properties:
        epsg (int | None):
            EPSG code of the CRS; `None` when the CRS is unset.
        total_bounds:
            Array-like `[minx, miny, maxx, maxy]` in the object's
            CRS. FeatureCollection inherits this from
            :class:`geopandas.GeoDataFrame`; Dataset exposes the
            same shape via the same attribute.
        top_left_corner:
            Sequence `[minx, maxy]` — the NW corner of the
            bounding box.

    Methods:
        read_file(path) (classmethod):
            Construct an instance from a file path.
        to_file(path,...):
            Serialize the object to `path`.
        plot(...):
            Render a matplotlib view of the object.

    Because this is :func:`typing.runtime_checkable`, you can use it
    with :func:`isinstance`:

    >>> from pyramids.base.protocols import SpatialObject
    >>> def describe(obj: SpatialObject) -> int | None:
    ...     return obj.epsg

    Runtime isinstance checks verify method/attribute presence only
    (PEP 544 — they do not verify signatures or return types).
    """

    epsg: int | None
    total_bounds: Any
    top_left_corner: Any

    @classmethod
    def read_file(cls, path: str | Path, *args: Any, **kwargs: Any) -> SpatialObject:
        """Read an on-disk representation into an instance.

        Protocol stub — see :meth:`pyramids.dataset.Dataset.read_file` and
        :meth:`pyramids.feature.FeatureCollection.read_file` for runnable
        examples. The `...` body here is a structural-type marker, not
        a callable implementation.
        """
        ...

    def to_file(self, path: str | Path, *args: Any, **kwargs: Any) -> None:
        """Serialize this object to `path` (protocol stub; see concrete impls)."""
        ...

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Render a matplotlib view of this object (protocol stub; see concrete impls)."""
        ...


@runtime_checkable
class LazySpatialObject(Protocol):
    """Lazy variant of :class:`SpatialObject` for dask-backed vectors.

    a separate protocol for dask-backed objects whose
    `total_bounds` / geometry attributes are not cheap to read. On an
    eager :class:`pyramids.feature.FeatureCollection`, `total_bounds`
    is a materialised 4-element numpy array — cheap, safe to expose as
    a property. On a :class:`pyramids.feature.LazyFeatureCollection`,
    `total_bounds` is a `dask.Scalar` that requires `.compute()`
    (an O(partitions) reduction) to resolve. Hiding that compute behind
    an eager-looking property is a leak; this protocol makes the
    laziness explicit.

    Consumers that genuinely want to accept either eager or lazy
    objects should type-check against `SpatialObject | LazySpatialObject`
    and branch on :func:`pyramids.feature.is_lazy_fc` before touching
    the bounds attributes.

    Attributes / properties:
        epsg (int | None):
            EPSG code of the CRS; cheap to read (pure metadata).
        total_bounds:
            A lazy object (dask Scalar for dask-geopandas backed
            frames). Consumers must call `.compute()` to materialise.
        npartitions (int):
            Number of dask partitions. Cheap (metadata only).

    Methods:
        compute(**kwargs):
            Materialise the graph; returns a corresponding
            :class:`SpatialObject` (eager twin).
        persist(**kwargs):
            Materialise graph into worker memory, stay lazy.
        to_file(path,...):
            May raise :class:`NotImplementedError` for drivers with
            no lazy write path — callers should `.compute().to_file(...)`.

    Because this is :func:`typing.runtime_checkable`, you can use it
    with :func:`isinstance`:

    >>> from pyramids.base.protocols import LazySpatialObject
    >>> def get_parts(obj: LazySpatialObject) -> int:
    ...     return obj.npartitions

    Runtime isinstance checks verify attribute / method presence only
    (PEP 544 — they do not verify signatures or return types).
    """

    epsg: int | None
    total_bounds: Any
    npartitions: int

    def compute(self, *args: Any, **kwargs: Any) -> SpatialObject:
        """Materialise this lazy object into its eager twin (protocol stub).

        See :meth:`pyramids.feature.LazyFeatureCollection.compute` for
        the concrete implementation and runnable examples.
        """
        ...

    def persist(self, *args: Any, **kwargs: Any) -> LazySpatialObject:
        """Force the graph into worker memory; keep laziness (protocol stub)."""
        ...

    def to_file(self, path: str | Path, *args: Any, **kwargs: Any) -> None:
        """Serialize this object (may raise :class:`NotImplementedError`; stub)."""
        ...


@runtime_checkable
class _ArrayLikeProto(Protocol):
    """Runtime-checkable structural type for eager-or-lazy arrays.

    Matches any object that has the attributes / methods numpy exposes on
    its ndarray and that dask exposes on `dask.array.Array`. Used for
    :func:`isinstance` branches that dispatch on "do we have an array
    backend at all?" — *not* for static type annotations, which should
    use the :data:`ArrayLike` type alias instead.

    PEP 544 runtime checks verify attribute presence only, not
    signatures, so extra guards (for example comparing `ndim` or
    checking `hasattr(x, "dask")`) may be needed for precise dispatch.

    Examples:
        - Numpy `ndarray` satisfies the structural type:
            ```python
            >>> import numpy as np
            >>> from pyramids.base.protocols import _ArrayLikeProto
            >>> isinstance(np.zeros(5), _ArrayLikeProto)
            True

            ```
        - Plain Python containers do not satisfy it:
            ```python
            >>> from pyramids.base.protocols import _ArrayLikeProto
            >>> isinstance([1, 2, 3], _ArrayLikeProto)
            False

            ```
    """

    shape: tuple[int, ...]
    ndim: int
    dtype: Any

    def __array__(
        self, dtype: Any = None
    ) -> np.ndarray:  # pragma: no cover - protocol stub
        """Return a numpy representation of the array."""
        ...

    def __getitem__(
        self, key: Any
    ) -> _ArrayLikeProto:  # pragma: no cover - protocol stub
        """Return a sliced view or copy."""
        ...


def is_lazy(x: Any) -> bool:
    """Return True if `x` is a dask-backed array, False if eager.

    The check is duck-typed rather than isinstance-based, so any
    object exposing a `dask` graph attribute plus a `compute`
    method (for example custom dask subclasses) is reported as lazy.
    `None` and non-array inputs return False.

    Args:
        x: Any object — typically a numpy `ndarray` or a
            `dask.array.Array`.

    Returns:
        bool: `True` when `x` is lazy (has dask graph and a
        `compute` method), `False` otherwise.

    Examples:
        >>> import numpy as np
        >>> from pyramids.base.protocols import is_lazy
        >>> is_lazy(np.zeros(5))
        False
        >>> is_lazy(None)
        False
    """
    return hasattr(x, "dask") and hasattr(x, "compute")


def as_numpy(x: ArrayLike) -> np.ndarray:
    """Return a numpy ndarray view/copy of `x`, computing if lazy.

    Eager :class:`numpy.ndarray` inputs are returned via
    :func:`numpy.asarray` (a zero-copy view when the dtype matches).
    Dask-backed inputs are materialized via `x.compute()`.

    Use this at the boundary where pyramids needs to hand an array
    to code that is not dask-aware (for example a GDAL
    `WriteArray` call), so every lazy-vs-eager branch in the
    codebase funnels through one helper.

    Args:
        x: The input array. Must satisfy :class:`_ArrayLikeProto`.

    Returns:
        np.ndarray: The materialized numpy array.

    Examples:
        >>> import numpy as np
        >>> from pyramids.base.protocols import as_numpy
        >>> arr = np.arange(4)
        >>> as_numpy(arr).tolist()
        [0, 1, 2, 3]
    """
    if is_lazy(x):
        result = x.compute()
    else:
        result = np.asarray(x)
    return result
