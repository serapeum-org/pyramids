"""Pickle-safe file-handle managers for GDAL / OGR datasets.

Two concrete shapes, both subclasses of :class:`FileManager`:

* :class:`CachingFileManager` — process-global LRU handle cache guarded
  by a user-supplied lock (`SerializableLock` is the default). One
  handle per cache key, shared by every caller that produces the same
  key. On LRU eviction or explicit :meth:`close` the underlying
  `gdal.Dataset` is released. Pattern copied from xarray's
  `xarray.backends.file_manager.CachingFileManager`.

* :class:`ThreadLocalFileManager` — per-thread handles, no locking.
  Each worker thread opens its own handle the first time it calls
  :meth:`acquire`. Pattern copied from rioxarray's
  `rioxarray._io.URIManager`.

**Pickle rule** — `__getstate__` returns only the recipe
(`opener`, `path`, `access`, `kwargs`). The live handle, the
cache, the lock's underlying :class:`threading.Lock` and the ref
counter are never serialized. On unpickle the manager reconstructs
with an empty cache and opens fresh on first :meth:`acquire`.

This module does not import `dask`. The :data:`SerializableLock`
default is re-exported from :mod:`pyramids.base._locks`. If that
module is unavailable, pass `lock=threading.Lock()` or
`lock=False` explicitly.
"""

from __future__ import annotations

import atexit
import os
import threading
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Hashable, Iterator, MutableMapping

import numpy as np # noqa: F401 - imported so type checkers see np.ndarray refs
from osgeo import gdal, ogr

from pyramids.base.remote import _to_vsi

_DEFAULT_MAXSIZE = int(os.environ.get("PYRAMIDS_FILE_CACHE_MAXSIZE", "128"))


_ACCESS_FLAGS = {
    "read_only": gdal.GA_ReadOnly,
    "r": gdal.GA_ReadOnly,
    "write": gdal.GA_Update,
    "w": gdal.GA_Update,
    "update": gdal.GA_Update,
    "a": gdal.GA_Update,
}


def _resolve_access(access: str) -> int:
    """Normalize a pyramids `access` string to the matching GDAL flag.

    Args:
        access: One of `"read_only"`, `"r"`, `"write"`, `"w"`,
            `"update"`, `"a"`.

    Returns:
        int: The corresponding :data:`osgeo.gdal.GA_*` constant.

    Raises:
        ValueError: If `access` is not a recognized mode string.

    Examples:
        - Read-only aliases all resolve to `GA_ReadOnly`:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._file_manager import _resolve_access
            >>> _resolve_access("read_only") == gdal.GA_ReadOnly
            True
            >>> _resolve_access("r") == gdal.GA_ReadOnly
            True

            ```
        - Unknown access string raises a descriptive ValueError:
            ```python
            >>> from pyramids.base._file_manager import _resolve_access
            >>> _resolve_access("bogus")
            Traceback (most recent call last):
              ...
            ValueError: Unknown access mode 'bogus'; expected one of ['a', 'r', 'read_only', 'update', 'w', 'write']

            ```
    """
    try:
        flag = _ACCESS_FLAGS[access]
    except KeyError as exc:
        raise ValueError(
            f"Unknown access mode {access!r}; expected one of "
            f"{sorted(_ACCESS_FLAGS)}"
        ) from exc
    return flag


def gdal_raster_open(path: str, access: str = "read_only", **_: Any) -> gdal.Dataset:
    """Open a classic-mode raster (GeoTIFF, COG, PNG,...) via :func:`gdal.Open`.

    The `path` is rewritten through :func:`pyramids.base.remote._to_vsi`
    first, so callers can pass URL-scheme paths (`s3://bucket/file.tif`,
    `https://example.com/file.tif`) directly.

    Args:
        path: File path or URL.
        access: Access mode string — see :func:`_resolve_access`.
        **_: Extra keyword arguments are accepted and ignored so that a
            single uniform opener signature can be used as a
            `FileManager` `opener` callable.

    Returns:
        osgeo.gdal.Dataset: The opened dataset handle.
    """
    return gdal.Open(_to_vsi(path), _resolve_access(access))


def gdal_mdarray_open(path: str, access: str = "read_only", **_: Any) -> gdal.Dataset:
    """Open a multidimensional raster (NetCDF, HDF5, Zarr) via :func:`gdal.OpenEx`.

    Equivalent to :func:`gdal_raster_open` but uses
    :data:`gdal.OF_MULTIDIM_RASTER`, which is required for group /
    :class:`gdal.MDArray` access on NetCDF and HDF5 files.

    Args:
        path: File path or URL.
        access: Access mode string — see :func:`_resolve_access`.
        **_: Extra keyword arguments accepted and ignored for signature
            uniformity.

    Returns:
        osgeo.gdal.Dataset: The opened MDIM dataset.
    """
    flags = gdal.OF_MULTIDIM_RASTER
    flags |= gdal.OF_UPDATE if access not in {"read_only", "r"} else gdal.OF_READONLY
    return gdal.OpenEx(_to_vsi(path), flags)


def ogr_open(path: str, access: str = "read_only", **_: Any) -> ogr.DataSource:
    """Open a vector datasource via :func:`ogr.Open`.

    Args:
        path: File path or URL.
        access: `"read_only"` / `"r"` opens read-only; any other
            value opens for update.
        **_: Extra keyword arguments accepted and ignored for signature
            uniformity.

    Returns:
        osgeo.ogr.DataSource: The opened vector datasource.
    """
    update = 0 if access in {"read_only", "r"} else 1
    return ogr.Open(_to_vsi(path), update)


class _LRUCache(MutableMapping):
    """Tiny LRU cache with `on_evict` callback.

    Thin wrapper around :class:`collections.OrderedDict`. Keys are
    moved to the end of the insertion order on every access, and
    `popitem(last=False)` removes the least-recently-used entry
    when :attr:`maxsize` would otherwise be exceeded. An `on_evict`
    callable (if provided) is invoked on eviction so cached file
    handles can be closed cleanly.

    Modeled on :class:`xarray.backends.lru_cache.LRUCache`.

    Examples:
        - Basic set / get / eviction:
            ```python
            >>> from pyramids.base._file_manager import _LRUCache
            >>> cache = _LRUCache(maxsize=2)
            >>> cache["a"] = 1
            >>> cache["b"] = 2
            >>> cache["a"]
            1
            >>> cache["c"] = 3
            >>> "b" in cache
            False

            ```
        - `on_evict` fires when a key is pushed out:
            ```python
            >>> from pyramids.base._file_manager import _LRUCache
            >>> evicted = []
            >>> cache = _LRUCache(maxsize=1, on_evict=lambda k, v: evicted.append(k))
            >>> cache["x"] = 1
            >>> cache["y"] = 2
            >>> evicted
            ['x']

            ```
    """

    def __init__(
        self, maxsize: int, on_evict: Callable[[Hashable, Any], None] | None = None
    ):
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._cache: OrderedDict[Hashable, Any] = OrderedDict()
        self._maxsize = maxsize
        self._on_evict = on_evict
        self._lock = threading.RLock()

    @property
    def maxsize(self) -> int:
        """Maximum number of entries held simultaneously."""
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"maxsize must be >= 1, got {value}")
        self._maxsize = value
        self._enforce_size_limit(value)

    def _enforce_size_limit(self, target: int) -> None:
        """Evict LRU entries until `len(self) <= target`.

        `on_evict` runs OUTSIDE the cache lock so the
        callback is free to take any other lock (including a
        :class:`CachingFileManager` per-handle mutex) without
        risking a deadlock against concurrent `acquire()` calls
        on other threads.
        """
        to_evict: list[tuple[Hashable, Any]] = []
        with self._lock:
            while len(self._cache) > target:
                to_evict.append(self._cache.popitem(last=False))
        if self._on_evict is not None:
            for key, value in to_evict:
                self._on_evict(key, value)

    def __getitem__(self, key: Hashable) -> Any:
        with self._lock:
            value = self._cache[key]
            self._cache.move_to_end(key)
            return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        to_evict: list[tuple[Hashable, Any]] = []
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
                return
            while len(self._cache) >= self._maxsize:
                to_evict.append(self._cache.popitem(last=False))
            self._cache[key] = value
        if self._on_evict is not None:
            for evicted_key, evicted_value in to_evict:
                self._on_evict(evicted_key, evicted_value)

    def __delitem__(self, key: Hashable) -> None:
        with self._lock:
            del self._cache[key]

    def __iter__(self) -> Iterator[Hashable]:
        with self._lock:
            return iter(list(self._cache))

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        """Evict every entry, calling `on_evict` for each one.

        `on_evict` runs with the cache lock released so callback
        code can take other locks without deadlock.
        """
        with self._lock:
            items = list(self._cache.items())
            self._cache.clear()
        if self._on_evict is not None:
            for key, value in items:
                self._on_evict(key, value)


def _close_handle(_key: Hashable, handle: Any) -> None:
    """Close a cached GDAL/OGR handle if it has a `Close` method."""
    close = getattr(handle, "Close", None)
    if close is not None:
        try:
            close()
        except Exception: # pragma: no cover - closing a freed handle is harmless
            pass


FILE_CACHE: _LRUCache = _LRUCache(_DEFAULT_MAXSIZE, on_evict=_close_handle)
"""Process-global LRU cache shared by every :class:`CachingFileManager`.

Keyed by a :class:`_HashedSequence` tuple of
`(opener, path, access, sorted_kwargs, manager_id)`. Default size
128; override via the `PYRAMIDS_FILE_CACHE_MAXSIZE` env var or by
setting :attr:`FILE_CACHE.maxsize` at runtime.
"""


class _HashedSequence(list):
    """List subclass with a cached hash value.

    The cache key must be hashable (for dict lookup) and must include
    the opener callable + opener args + kwargs. We spell it as a list
    subclass rather than a tuple so callers can mutate the sequence
    for debugging without disturbing the pre-computed hash.

    Examples:
        - Hash matches the tuple of contents:
            ```python
            >>> from pyramids.base._file_manager import _HashedSequence
            >>> hs = _HashedSequence([1, "x"])
            >>> hash(hs) == hash((1, "x"))
            True

            ```
        - Usable as a dict key:
            ```python
            >>> from pyramids.base._file_manager import _HashedSequence
            >>> d = {_HashedSequence(["a", 1]): "value"}
            >>> d[_HashedSequence(["a", 1])]
            'value'

            ```
    """

    __slots__ = ("hashvalue",)

    def __init__(self, iterable: Iterator[Any]):
        super().__init__(iterable)
        self.hashvalue = hash(tuple(self))

    def __hash__(self) -> int:  # type: ignore[override]
        return self.hashvalue


def _make_cache_key(
    opener: Callable, path: str, access: str, kwargs: dict, manager_id: Hashable
) -> _HashedSequence:
    """Build the `FILE_CACHE` key for a :class:`CachingFileManager`.

    Kwargs are sorted so the same logical configuration always
    produces the same key regardless of dict ordering.
    """
    kwargs_key = tuple(sorted(kwargs.items())) if kwargs else ()
    return _HashedSequence([opener, path, access, kwargs_key, manager_id])


class FileManager(ABC):
    """Abstract base class for pickle-safe GDAL/OGR file-handle managers.

    Subclasses implement :meth:`acquire`, :meth:`acquire_context`, and
    :meth:`close`. The base class is intentionally minimal — it does
    not own the handle, the cache, or the lock. Those concerns are
    pushed into concrete subclasses so that alternative shapes
    (thread-local, ref-counted, test fakes) can coexist without a
    shared implementation.
    """

    @abstractmethod
    def acquire(self) -> Any:
        """Return an open GDAL/OGR handle. Opens the file on first call."""

    @abstractmethod
    def acquire_context(self) -> contextmanager[Any]:  # type: ignore[type-arg]
        """Context manager yielding an open handle; releases on exit."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying handle and remove it from any cache."""


class CachingFileManager(FileManager):
    """Pickle-safe, LRU-cached, lockable file-handle manager.

    Args:
        opener: Callable opening the file — for example
            :func:`pyramids.base._openers.gdal_raster_open`. Must have
            signature `opener(path, access, **kwargs) -> handle`.
        path: File path or URL passed to `opener`.
        access: Access mode string passed to `opener`.
        kwargs: Extra keyword arguments passed to `opener`.
        lock: A `threading.Lock`-like object guarding access to the
            cached handle, or `False` to skip locking. Defaults to
            a fresh :class:`threading.Lock`.
        cache: The cache to store handles in. Defaults to the
            module-level :data:`FILE_CACHE`.
        manager_id: Distinguishes different managers that would
            otherwise hash identically. Defaults to a fresh UUID so
            two managers built from identical arguments do not share
            a cache slot; pass an explicit value to *share* one.

    The manager is picklable. On unpickle, the new instance starts
    with no cached handle and opens fresh on first :meth:`acquire`;
    if two unpickled clones use the same `manager_id`, they both
    resolve to the same cache slot.
    """

    def __init__(
        self,
        opener: Callable[..., Any],
        path: str,
        access: str = "read_only",
        kwargs: dict | None = None,
        *,
        lock: Any = None,
        cache: _LRUCache | None = None,
        manager_id: Hashable | None = None,
    ) -> None:
        self._opener = opener
        self._path = path
        self._access = access
        self._kwargs = dict(kwargs or {})
        self._use_default_lock = lock is None
        if lock is False:
            self._lock = _NULL_LOCK
        elif lock is None:
            self._lock = threading.Lock()
        else:
            self._lock = lock
        self._cache = cache if cache is not None else FILE_CACHE
        self._manager_id = manager_id if manager_id is not None else str(uuid.uuid4())
        self._key = _make_cache_key(
            opener, path, access, self._kwargs, self._manager_id
        )

    def __getstate__(self) -> tuple:
        lock = None if self._use_default_lock else self._lock
        return (
            self._opener,
            self._path,
            self._access,
            self._kwargs,
            lock,
            self._manager_id,
        )

    def __setstate__(self, state: tuple) -> None:
        opener, path, access, kwargs, lock, manager_id = state
        self.__init__(
            opener,
            path,
            access,
            kwargs,
            lock=lock,
            manager_id=manager_id,
        )

    def acquire(self) -> Any:
        """Return the handle, opening it if not already cached."""
        with self._lock:
            try:
                handle = self._cache[self._key]
            except KeyError:
                handle = self._opener(self._path, self._access, **self._kwargs)
                self._cache[self._key] = handle
        return handle

    @contextmanager
    def acquire_context(self) -> Iterator[Any]:
        """Context manager yielding the handle; lock is held inside `with`.

        On any exception raised inside the `with` block, the handle
        is preserved in the cache (other callers may still need it);
        only explicit :meth:`close` removes it.
        """
        with self._lock:
            try:
                handle = self._cache[self._key]
                was_cached = True
            except KeyError:
                handle = self._opener(self._path, self._access, **self._kwargs)
                self._cache[self._key] = handle
                was_cached = False
            try:
                yield handle
            except Exception:
                if not was_cached:
                    self._drop()
                raise

    def _drop(self) -> None:
        """Remove the handle from the cache without calling `on_evict`."""
        try:
            del self._cache[self._key]
        except KeyError:
            pass

    def close(self) -> None:
        """Remove the handle from the cache and close it."""
        with self._lock:
            try:
                handle = self._cache[self._key]
            except KeyError:
                return
            del self._cache[self._key]
        _close_handle(self._key, handle)


class _NullLock:
    """Drop-in lock that never blocks. Used when `lock=False`."""

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return True

    def release(self) -> None:
        pass

    def __enter__(self) -> _NullLock:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


_NULL_LOCK = _NullLock()


class ThreadLocalFileManager(FileManager):
    """Lock-free, per-thread file-handle manager.

    Each thread calling :meth:`acquire` opens its own handle on first
    access and reuses it for the life of the thread. No lock is held
    so concurrent readers on different threads never contend. The
    trade-off: no handle-count bound — don't use for datacubes with
    thousands of distinct files unless the thread count is small.

    Args:
        opener: Callable opening the file. Same signature as for
            :class:`CachingFileManager`.
        path: File path or URL.
        access: Access mode string.
        kwargs: Extra keyword arguments for `opener`.
    """

    def __init__(
        self,
        opener: Callable[..., Any],
        path: str,
        access: str = "read_only",
        kwargs: dict | None = None,
    ) -> None:
        self._opener = opener
        self._path = path
        self._access = access
        self._kwargs = dict(kwargs or {})
        self._local = threading.local()

    def __getstate__(self) -> tuple:
        return (self._opener, self._path, self._access, self._kwargs)

    def __setstate__(self, state: tuple) -> None:
        opener, path, access, kwargs = state
        self.__init__(opener, path, access, kwargs)

    def acquire(self) -> Any:
        """Return this thread's handle, opening one on first call."""
        handle = getattr(self._local, "handle", None)
        if handle is None:
            handle = self._opener(self._path, self._access, **self._kwargs)
            self._local.handle = handle
        return handle

    @contextmanager
    def acquire_context(self) -> Iterator[Any]:
        """Context manager yielding this thread's handle."""
        yield self.acquire()

    def close(self) -> None:
        """Close this thread's handle (does not affect other threads)."""
        handle = getattr(self._local, "handle", None)
        if handle is not None:
            _close_handle(None, handle)
            self._local.handle = None


def _close_all_cached_handles() -> None:  # pragma: no cover - invoked at exit
    """Close every handle in :data:`FILE_CACHE` at interpreter shutdown."""
    try:
        FILE_CACHE.clear()
    except Exception:
        pass


atexit.register(_close_all_cached_handles)
