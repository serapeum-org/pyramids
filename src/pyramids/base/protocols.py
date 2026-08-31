"""Structural-typing protocols shared across the pyramids package.

This module exposes three cross-cutting structural types:

* :class:`SpatialObject` — the surface shared by
  :class:`pyramids.dataset.Dataset` (raster) and
  :class:`pyramids.feature.FeatureCollection` (vector), so callers can
  write generic utilities that accept either without importing both
  concrete classes (and without creating import cycles).
* :class:`RasterLike` — the raster-specific surface shared by
  :class:`pyramids.dataset.Dataset` and its :class:`pyramids.netcdf.NetCDF`
  subclass (extends :class:`SpatialObject`), for raster-only generic
  utilities that should accept either without importing the concrete classes.
* :class:`ArrayLike` — the structural type matching both
  :class:`numpy.ndarray` and :class:`dask.array.Array`, used to annotate
  array-returning methods that may be either eager or lazy.

For dtype-precise *eager* array returns the module re-exports
:data:`numpy.typing.NDArray` and a :data:`FloatArray` alias
(``NDArray[np.float64]``) — used to annotate typed numpy returns such as
coordinate / dimension arrays from a single place. (The :data:`ArrayLike`
union stays dtype-agnostic because dask arrays do not compose with
``NDArray[...]``.)

**Spelling convention:** this module imports `NDArray` and defines the
precise `FloatArray` here, but consumer modules across the package spell a
generic array return inline as ``np.typing.NDArray`` (resolved off their
existing ``import numpy as np``, so no extra import / isort churn). Reserve
`FloatArray` for returns whose element type is provably float64; leave
dtype-variable returns (data reads of arbitrary dtype) as the Any-dtype
``np.typing.NDArray`` / :data:`ArrayLike`.

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
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeGuard,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from pyramids.base.georeference import GeoReference

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    import dask.array as da  # noqa: F401


# Type alias covering both numpy arrays and (optionally-installed) dask arrays.
# String-forwarded so importing `pyramids.base.protocols` never triggers a
# dask import. Use this alias for function signatures that accept or return
# either backend; use the :class:`_ArrayLikeProto` Protocol below for runtime
# isinstance checks.
ArrayLike = Union[np.ndarray, "da.Array"]

# Dtype-precise alias for *eager* array returns whose element type is known
# (e.g. coordinate / dimension arrays, which are always float64). `NDArray` is
# re-exported from `numpy.typing` so callers annotate typed numpy returns from a
# single place (ARC-19); `ArrayLike` above stays the eager-or-lazy union and is
# deliberately dtype-agnostic (dask arrays do not compose with `NDArray[...]`).
FloatArray = NDArray[np.float64]


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
class RasterLike(SpatialObject, Protocol):
    """Structural type for the full pyramids raster surface (`Dataset` / `NetCDF`).

    The protocol-based type constraint for the dataset surface (ARC-18). It
    extends :class:`SpatialObject` with the raster **read** surface
    (geotransform-derived geometry, grid shape, band / no-data metadata, the
    GDAL handle, array reads) **and** the public raster **operations**
    (`crop`, `to_crs`, `overlay`, `extract`, `change_no_data_value`, the
    overview family) plus the `from_array` constructor — i.e. the
    structural mirror of the
    :class:`pyramids.dataset.abstract_dataset.RasterBase` abstract contract.
    Use it to annotate code that accepts or returns "a pyramids raster"
    (`Dataset`, `NetCDF`, or a future raster type) **without importing the
    concrete classes**, which also lets `base`-layer code below `dataset` in
    the import graph reference a raster without an import cycle.

    This is a typing aid, not a replacement for the ABC: a `Protocol` cannot
    hold the instance state (`_raster`, the geotransform) or the shared method
    bodies that `Dataset` / `NetCDF` inherit, so `RasterBase` remains the
    concrete implementation base and nominal `isinstance(x, RasterBase)`
    checks are unaffected. `RasterLike` is the *type/contract* layer;
    `RasterBase` is the *implementation* layer.

    Intended use is as an **exported, consumer-facing contract** — downstream
    code (and `base`-layer helpers) annotate against it; pyramids' own internal
    call sites generally use the concrete `Dataset` / `RasterBase` types and
    their nominal `isinstance` checks, so `RasterLike` is deliberately not
    forced onto internal signatures (e.g. it is *not* `align`'s parameter type:
    a 0-band `NetCDF` *container* structurally satisfies this protocol yet is
    not a usable alignment source). It is kept `runtime_checkable` so consumers
    *can* `isinstance`-check it — at the cost of a 26-attribute presence probe,
    so prefer it for annotations over hot-loop `isinstance`.

    Because this is :func:`typing.runtime_checkable`, you can use it with
    :func:`isinstance` (PEP 544 — attribute/method presence only, not
    signatures or return types):

    >>> from pyramids.base.protocols import RasterLike
    >>> def cell_count(r: RasterLike) -> int:
    ...     return r.rows * r.columns
    """

    # Geo / grid read surface (in addition to SpatialObject's epsg /
    # total_bounds / top_left_corner).
    cell_size: Any
    rows: int
    columns: int
    band_count: int
    no_data_value: Any
    geotransform: Any
    shape: Any
    crs: Any
    raster: Any

    def read_array(
        self, *args: Any, **kwargs: Any
    ) -> ArrayLike:  # pragma: no cover - protocol stub
        """Read band data as a numpy or dask array (protocol stub; see concrete impls)."""
        ...

    @classmethod
    def from_array(  # pragma: no cover - protocol stub
        cls,
        arr: Any,
        *,
        geo_ref: GeoReference,
        no_data_value: Any = ...,
        path: Any = None,
    ) -> RasterLike:
        """Construct a raster from an array.

        Declared with the real parameters rather than ``*args, **kwargs``: the
        bare form made an incompatible override invisible to static checkers,
        which is how the concrete implementations came to disagree.
        """
        ...

    def crop(
        self, *args: Any, **kwargs: Any
    ) -> RasterLike:  # pragma: no cover - protocol stub
        """Crop to a mask / bounds (protocol stub; see concrete impls)."""
        ...

    def to_crs(
        self, *args: Any, **kwargs: Any
    ) -> RasterLike:  # pragma: no cover - protocol stub
        """Reproject to a target CRS (protocol stub; see concrete impls)."""
        ...

    def overlay(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - protocol stub
        """Zonal/overlay extraction against another object (protocol stub)."""
        ...

    def extract(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - protocol stub
        """Extract cell values (protocol stub; see concrete impls)."""
        ...

    def change_no_data_value(  # pragma: no cover - protocol stub
        self, *args: Any, **kwargs: Any
    ) -> Any:
        """Change the no-data sentinel (protocol stub; see concrete impls)."""
        ...

    def create_overviews(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - protocol stub
        """Build reduced-resolution overviews (protocol stub)."""
        ...

    def recreate_overviews(  # pragma: no cover - protocol stub
        self, *args: Any, **kwargs: Any
    ) -> Any:
        """Rebuild overviews (protocol stub; see concrete impls)."""
        ...

    def get_overview(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - protocol stub
        """Return an overview level (protocol stub; see concrete impls)."""
        ...

    def get_overview_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - protocol stub
        """Return an overview level as a Dataset (protocol stub; see concrete impls)."""
        ...

    def read_overview_array(  # pragma: no cover - protocol stub
        self, *args: Any, **kwargs: Any
    ) -> ArrayLike:
        """Read an overview level as an array (protocol stub)."""
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
    ) -> NDArray:  # pragma: no cover - protocol stub
        """Return a numpy representation of the array."""
        ...

    def __getitem__(
        self, key: Any
    ) -> _ArrayLikeProto:  # pragma: no cover - protocol stub
        """Return a sliced view or copy."""
        ...


def is_lazy(x: Any) -> TypeGuard[da.Array]:
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
        `compute` method), `False` otherwise. Typed as a
        :data:`typing.TypeGuard`, so a truthy result narrows `x` to
        `dask.array.Array` for the type checker.

    Examples:
        >>> import numpy as np
        >>> from pyramids.base.protocols import is_lazy
        >>> is_lazy(np.zeros(5))
        False
        >>> is_lazy(None)
        False
    """
    return hasattr(x, "dask") and hasattr(x, "compute")


def as_numpy(x: ArrayLike) -> NDArray:
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
        # `is_lazy` narrows `x` to `dask.array.Array`; `.compute()` returns an
        # untyped value, so cast the materialized result back to a numpy array.
        result = cast("NDArray", x.compute())
    else:
        result = np.asarray(x)
    return result
