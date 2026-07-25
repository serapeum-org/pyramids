"""Pickle-safe file-handle managers for GDAL / OGR datasets.

Two concrete shapes, both subclasses of :class:`FileManager`:

* :class:`CachingFileManager` — process-global LRU handle cache guarded
  by a user-supplied lock (`SerializableLock` is the default). One
  handle per cache key, shared by every caller that produces the same
  key. On LRU eviction or explicit :meth:`close` the underlying
  `gdal.Dataset` is released.

* :class:`ThreadLocalFileManager` — per-thread handles, no locking.
  Each worker thread opens its own handle the first time it calls
  :meth:`acquire`.

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
import logging
import os
import threading
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Iterator, MutableMapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any, cast

import numpy as np  # noqa: F401 - imported so type checkers see np.ndarray refs
from osgeo import gdal, ogr

from pyramids.base.remote import _to_vsi

logger = logging.getLogger(__name__)

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
            f"Unknown access mode {access!r}; expected one of {sorted(_ACCESS_FLAGS)}"
        ) from exc
    return cast(int, flag)


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
        - A pinned key is never the eviction victim; the cache grows
          past `maxsize` instead of closing a handle mid-read:
            ```python
            >>> from pyramids.base._file_manager import _LRUCache
            >>> evicted = []
            >>> cache = _LRUCache(maxsize=1, on_evict=lambda k, v: evicted.append(k))
            >>> cache["x"] = 1
            >>> cache.pin("x")
            >>> cache["y"] = 2
            >>> evicted
            []
            >>> sorted(cache)
            ['x', 'y']
            >>> cache.unpin("x")
            >>> cache["z"] = 3
            >>> evicted
            ['x', 'y']

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
        # Number of live managers interested in each key. The handle is closed only when the last
        # manager for a key is finalized, so a manager whose array is dropped never evicts a handle
        # another manager (sharing the same `manager_id`) is still reading through.
        self._refcounts: dict[Hashable, int] = {}
        # Number of in-flight reads holding each key. Distinct from `_refcounts`: a pin is scoped to
        # one `acquire_context()` block and only makes the slot un-evictable for that window, it
        # never closes anything. Without it a manager's insert can LRU-evict and `Close()` a handle
        # another manager is mid-read through, since every manager carries its own lock.
        #
        # EVERY path that removes or replaces a cached value must consult `_pins` before handing the
        # old value to `on_evict` -- closing a handle a reader still holds is undefined behaviour in
        # GDAL, whereas deferring the close only delays reclaiming a descriptor. The size-driven
        # paths (`_select_evictions`), the overwrite path (`__setitem__`), the finalizer
        # (`release`) and explicit teardown (`discard`, used by `CachingFileManager.close`) all do.
        # `clear()` is the sole exception and says so in its docstring.
        self._pins: dict[Hashable, int] = {}
        # Values pulled from the cache while pinned, waiting for the last reader to finish. The
        # final `unpin()` closes them, so an explicit `close()` mid-read still reclaims the handle
        # deterministically instead of either leaking it or closing it under the reader.
        self._pending_close: dict[Hashable, Any] = {}

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

    def _select_evictions(self, target: int) -> list[tuple[Hashable, Any]]:
        """Pop least-recently-used, unpinned entries until `len(self) <= target`.

        The caller must already hold :attr:`_lock`; the popped
        `(key, value)` pairs are returned so `on_evict` can run
        outside it. Pinned keys are skipped: a slot with an in-flight
        read must not be closed underneath the reader, so when every
        remaining candidate is pinned the cache is allowed to sit
        above `target` until those reads finish.

        Args:
            target: Maximum number of entries to leave behind. Callers
                inserting a new entry pass `maxsize - 1` so the cache
                lands exactly at `maxsize` once the insert completes.

        Returns:
            list[tuple[Hashable, Any]]: The evicted `(key, value)`
            pairs, in eviction order, for the caller to pass to
            `on_evict` once the lock is released.
        """
        evicted: list[tuple[Hashable, Any]] = []
        if len(self._cache) > target:
            for key in list(self._cache):
                if len(self._cache) <= target:
                    break
                if self._pins.get(key):
                    continue
                evicted.append((key, self._cache.pop(key)))
        if len(self._cache) > target:
            # Report the configured limit, not `target`: an insert passes `maxsize - 1`,
            # which would otherwise render as "over its 127 limit" on a 128-entry cache.
            logger.debug(
                "file cache is %d entr(ies) over its %d limit: every eviction "
                "candidate is pinned by an in-flight read",
                len(self._cache) - target,
                self._maxsize,
            )
        return evicted

    def _enforce_size_limit(self, target: int) -> None:
        """Evict LRU entries until `len(self) <= target`.

        `on_evict` runs OUTSIDE the cache lock, so the callback never
        deadlocks against another thread waiting on :attr:`_lock`.

        That is the whole guarantee, and it covers the *cache* lock
        only. It says nothing about the caller's own locks, and one
        eviction path does hold one: an insert from
        :meth:`CachingFileManager.acquire` / `acquire_context` reaches
        `__setitem__` with that manager's mutex held, so `on_evict`
        runs under it. The `unpin` path deliberately does not (see
        `acquire_context`). An `on_evict` that takes a
        `CachingFileManager` mutex is therefore NOT safe — keep
        callbacks lock-free, as :func:`_close_handle` is.
        """
        with self._lock:
            to_evict = self._select_evictions(target)
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
                displaced = self._cache[key]
                self._cache.move_to_end(key)
                self._cache[key] = value
                if displaced is not value and not self._pins.get(key):
                    # Overwriting a live key drops the old handle; hand it to `on_evict` or the
                    # file descriptor leaks. Reachable whenever two managers share a `manager_id`
                    # with `lock=False` (e.g. `_read_time_step`'s `manager_id=path`): both can
                    # miss, both open, and both assign.
                    #
                    # A pinned key is exactly that interleaving with a reader inside
                    # `acquire_context()`, and the handle being displaced is the one it is reading
                    # through -- closing it there is a use-after-close, strictly worse than the
                    # descriptor leak. Leave it to the reader's own reference (SWIG closes the
                    # orphan when the last one drops), the same trade-off `_select_evictions` makes.
                    to_evict.append((key, displaced))
            else:
                # Trim to `maxsize - 1` first so the cache lands exactly at `maxsize` after the
                # insert, and so the entry being added is never itself an eviction candidate.
                to_evict.extend(self._select_evictions(self._maxsize - 1))
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
        code can take other locks without deadlock. Unlike LRU
        eviction this ignores pins — `clear()` is the documented hard
        reset (interpreter exit, test fixtures), not a size-driven
        reclaim.
        """
        with self._lock:
            items = list(self._cache.items())
            self._cache.clear()
            self._refcounts.clear()
            self._pins.clear()
        if self._on_evict is not None:
            for key, value in items:
                self._on_evict(key, value)

    def pin(self, key: Hashable) -> None:
        """Protect `key` from LRU eviction until the matching :meth:`unpin`.

        Pins nest: N `pin()` calls need N `unpin()` calls before the
        slot is evictable again. Pinning a key that is not (yet) in
        the cache is allowed — :meth:`CachingFileManager.acquire_context`
        pins before opening so the slot is covered from the moment it
        lands.

        Args:
            key: The cache key to protect.
        """
        with self._lock:
            self._pins[key] = self._pins.get(key, 0) + 1

    def unpin(self, key: Hashable) -> None:
        """Drop one pin from `key`; the slot is evictable again at zero.

        Dropping the last pin re-applies the size limit immediately.
        Pinned reads are the one thing allowed to push the cache over
        `maxsize`, so without this the overflow would persist until
        some *other* key happened to be inserted — on a workload that
        finishes its reads and stops, that is never, leaving handles
        open on a cache configured for far fewer.

        An untracked key -- e.g. wiped by :meth:`clear` while a read
        was in flight -- is a no-op rather than an underflow.

        Args:
            key: The cache key to release.
        """
        over_limit = False
        deferred = None
        with self._lock:
            pinned = self._pins.get(key)
            if pinned is not None:
                if pinned > 1:
                    self._pins[key] = pinned - 1
                else:
                    self._pins.pop(key, None)
                    # The last reader is leaving, so anything an explicit `close()` or a
                    # finalizer parked for this key can finally be released.
                    deferred = self._pending_close.pop(key, None)
                    # Only worth a sweep when pins actually pushed the cache over the
                    # limit. Checking here keeps the steady state -- cache at or under
                    # `maxsize`, which is every read on a warm cache -- free of the
                    # snapshot allocation and the second lock round trip, and keeps
                    # `on_evict` off the hot path entirely.
                    over_limit = len(self._cache) > self._maxsize
        # Outside the lock: `_enforce_size_limit` runs `on_evict` (which closes GDAL
        # handles) with the cache lock released, and re-taking it here would nest.
        if deferred is not None and self._on_evict is not None:
            try:
                self._on_evict(key, deferred)
            except Exception as exc:  # noqa: BLE001 - a teardown path must not raise
                logger.warning(
                    "handle close failed for deferred key %r: %s", key, exc, exc_info=True
                )
        if over_limit:
            self._enforce_size_limit(self._maxsize)

    def discard(self, key: Hashable) -> Any | None:
        """Remove `key` and return its value for the caller to close, if it may.

        Explicit teardown that stays safe against an in-flight read: an
        unpinned entry is returned so the caller closes it immediately,
        while a pinned one is parked in `_pending_close` and released
        by the last :meth:`unpin`. Either way the handle is reclaimed
        deterministically — never left to the garbage collector, and
        never closed while a reader is still going through it.

        Args:
            key: The cache key to remove.

        Returns:
            Any | None: The removed value when the caller should close
            it now, or `None` when the entry was absent or its close
            has been deferred to the last reader.
        """
        with self._lock:
            value = self._cache.pop(key, None)
            if value is not None and self._pins.get(key):
                self._pending_close[key] = value
                value = None
        return value

    def retain(self, key: Hashable) -> None:
        """Register one live referent for `key` (see the `_refcounts` note in `__init__`)."""
        with self._lock:
            self._refcounts[key] = self._refcounts.get(key, 0) + 1

    def release(self, key: Hashable) -> None:
        """Drop one referent for `key`; evict and close the handle when the last one is gone.

        Safe to call from a `weakref.finalize` callback on any thread: it holds no reference to the
        releasing manager and closes the handle outside the cache lock (like `on_evict`). A key that
        is no longer tracked -- e.g. wiped by `clear()` while this manager was still alive -- is
        treated as a no-op, so a stale finalizer never closes a handle the refcount is no longer
        accounting for (which could otherwise be a re-opened handle a live array is still reading). A
        manager that survives a `clear()` therefore falls back to pure LRU / interpreter-exit lifetime;
        that is an accepted trade-off, since `clear()` is an explicit hard reset.

        A pinned key has its close deferred rather than skipped. This path is driven by garbage
        collection rather than by a caller saying "I am done", so it can fire at an arbitrary moment --
        including while another manager sharing the slot is mid-read inside `acquire_context()`.
        Closing there would be a use-after-close, so the entry moves to `_pending_close` and the last
        `unpin()` releases it; the deterministic-release guarantee this path exists for is preserved
        rather than downgraded to "whenever LRU pressure happens to arrive".
        """
        handle = None
        with self._lock:
            tracked = self._refcounts.get(key)
            if tracked is not None:
                if tracked - 1 > 0:
                    self._refcounts[key] = tracked - 1
                else:
                    self._refcounts.pop(key, None)
                    handle = self._cache.pop(key, None)
                    if handle is not None and self._pins.get(key):
                        self._pending_close[key] = handle
                        handle = None
        if handle is not None and self._on_evict is not None:
            # release() runs from a `weakref.finalize` callback, which has no caller to surface a close
            # failure to -- so log any error (e.g. an OSError flushing a remote `/vsi` handle, which
            # `_close_handle` deliberately re-raises on a normal call stack) rather than let it become
            # an "unraisable" exception reported during GC.
            try:
                self._on_evict(key, handle)
            except Exception as exc:  # noqa: BLE001 - a finalizer path must not raise
                logger.warning(
                    "handle close failed during finalizer release for key %r: %s",
                    key,
                    exc,
                    exc_info=True,
                )


def _close_handle(_key: Hashable | None, handle: Any) -> None:
    """Close a cached GDAL/OGR handle if it has a `Close` method.

    Eviction-time close failures are logged at DEBUG and swallowed
    so that one stuck handle cannot abort the LRU's eviction loop
    (which would leave the cache in an inconsistent state). Only
    GDAL's `RuntimeError` is swallowed; other exception classes
    indicate programming bugs in the caller and propagate.

    Notes:
        Exceptions raised by GDAL through `Close()` on remote-FS
        backends — `OSError` and `socket.timeout` from
        `/vsis3/`, `/vsigs/`, `/vsiaz/` flush failures — are
        intentionally NOT caught here. They surface to the caller
        of the LRU operation that triggered the eviction
        (`cache[key] = value` / `cache.clear()` / explicit
        `manager.close()`) so the operator sees the I/O failure
        rather than silent data loss. If a downstream pipeline
        cannot tolerate that surface, wrap the LRU call in its
        own retry/log shim — do not widen the catch here.
    """
    close = getattr(handle, "Close", None)
    if close is None:
        return
    try:
        close()
    except RuntimeError as exc:
        logger.debug(
            "close handle failed for cache key %r: %s",
            _key,
            exc,
            exc_info=True,
        )


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

    def __init__(self, iterable: Iterable[Any]):
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
    def acquire_context(self) -> AbstractContextManager[Any]:
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
        auto_release: When `True`, register a `weakref.finalize` on
            the manager that releases its cached handle (evicting the
            slot and closing the handle) as soon as the manager is
            garbage-collected, refcounted so a shared slot closes only
            when its last manager is finalized. Use this when the
            manager's lifetime should own the handle — e.g. a lazy
            dask read whose manager is kept alive by the graph. The
            default `False` keeps the handle under pure LRU lifetime,
            reusable by later managers with the same `manager_id`. Do
            not share one `manager_id` between an `auto_release` and a
            non-`auto_release` manager: the refcount counts only the
            auto-release ones, so dropping one could close a handle the
            other still relies on.

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
        auto_release: bool = False,
    ) -> None:
        self._opener = opener
        self._path = path
        self._access = access
        self._kwargs = dict(kwargs or {})
        self._use_default_lock = lock is None
        self._lock: Any
        if lock is False:
            self._lock = _NULL_LOCK
        elif lock is None:
            self._lock = threading.Lock()
        else:
            self._lock = lock
        self._cache = cache if cache is not None else FILE_CACHE
        self._manager_id = manager_id if manager_id is not None else str(uuid.uuid4())
        self._auto_release = auto_release
        self._key = _make_cache_key(
            opener, path, access, self._kwargs, self._manager_id
        )
        if auto_release:
            # Release the parked handle deterministically when this manager is garbage-collected --
            # i.e. when the lazy array holding its chunk readers is dropped -- instead of relying on
            # LRU pressure or interpreter exit (#727). Bound to the manager, not the returned array,
            # so it still fires for derived arrays (unpack / mfdataset / plot) whose graph keeps only
            # the readers -> manager alive. The callback holds no reference to `self`, so it cannot
            # keep the manager alive; the refcount stops it evicting a slot another auto-release
            # manager still shares. Managers without `auto_release` keep the pure LRU lifetime.
            self._cache.retain(self._key)
            weakref.finalize(self, self._cache.release, self._key)

    def __getstate__(self) -> tuple:
        lock = None if self._use_default_lock else self._lock
        return (
            self._opener,
            self._path,
            self._access,
            self._kwargs,
            lock,
            self._manager_id,
            self._auto_release,
        )

    def __setstate__(self, state: tuple) -> None:
        # Tolerate the pre-`auto_release` 6-tuple so a manager pickled by an older pyramids (or a
        # differently-versioned dask worker) still unpickles, defaulting `auto_release` to False. The
        # reverse -- an old worker receiving the new 7-tuple -- cannot be helped from here and is
        # intentionally unsupported (mixed-version dask.distributed clusters are not a target).
        opener, path, access, kwargs, lock, manager_id, *rest = state
        auto_release = rest[0] if rest else False
        type(self).__init__(
            self,
            opener,
            path,
            access,
            kwargs,
            lock=lock,
            manager_id=manager_id,
            auto_release=auto_release,
        )

    def acquire(self) -> Any:
        """Return the handle, opening it if not already cached.

        The returned handle is **not** pinned: it is only guaranteed
        live for as long as it stays in the cache, and a concurrent
        `acquire()` on a *different* manager can push it out of the
        shared LRU and close it. Use :meth:`acquire_context` for any
        read that outlives this call — it pins the slot for the
        duration of the `with` block so eviction cannot reclaim it.
        """
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

        The cache slot is pinned for the whole block, so nothing can
        `Close()` the handle mid-read: LRU eviction, an overwrite by a
        manager that lost the open race, the `auto_release` finalizer
        and an explicit :meth:`close` all either skip the slot or defer
        their close to the final unpin (per-manager locks do not
        protect against any of them on their own).

        :meth:`_LRUCache.clear` is the one exception — a hard reset
        that closes everything — and is only used at interpreter exit,
        after CPython has joined the worker threads.

        On any exception raised inside the `with` block, the handle
        is preserved in the cache (other callers may still need it);
        only explicit :meth:`close` removes it.
        """
        # Pin and unpin OUTSIDE the manager mutex. `unpin()` can trigger an eviction
        # sweep, and `on_evict` closes GDAL handles -- a `/vsis3` flush can take
        # seconds. Doing that while this manager's lock is held would block every
        # other caller of the same manager on an eviction unrelated to them, and would
        # deadlock outright for any `on_evict` that takes a manager mutex (the very
        # pattern `_enforce_size_limit` documents as safe). The pin itself only needs
        # the cache's own lock, which `pin()` takes.
        self._cache.pin(self._key)
        try:
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
        finally:
            self._cache.unpin(self._key)

    def _drop(self) -> None:
        """Remove the handle from the cache without calling `on_evict`.

        Safe against an in-flight read regardless of pins: the entry
        leaves the cache but the handle is never closed, so a reader
        holding it keeps a live object and later callers simply miss
        and re-open. The reader's own `unpin` clears the pin entry
        afterwards, so nothing is left behind.
        """
        try:
            del self._cache[self._key]
        except KeyError:
            pass

    def close(self) -> None:
        """Remove the handle from the cache and close it.

        Safe to call while another thread is reading through the same
        cache slot: the entry is removed immediately, so no later
        caller can reach it, but the actual `Close()` is deferred to
        the reader's final `unpin` when the slot is pinned. This
        matters because `close()` is reachable from public API —
        `NetCDF.close()` walks its lazy managers and calls it — while
        dask workers may still be inside `acquire_context()`, and a
        GDAL use-after-close is a segfault rather than an exception.
        """
        handle = self._cache.discard(self._key)
        if handle is not None:
            _close_handle(self._key, handle)


class _NullLock:
    """Drop-in lock that never blocks. Used when `lock=False`."""

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return True

    def release(self) -> None:
        """No-op: a null lock is never held, so there is nothing to release."""

    def __enter__(self) -> _NullLock:
        return self

    def __exit__(self, *_: Any) -> None:
        """No-op: a null lock holds nothing, so context exit releases nothing."""


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
        # Every opened handle is tracked here so close() can release the ones
        # other threads opened (threading.local storage is unreachable from the
        # closing thread). The generation counter lets a thread whose handle was
        # closed by close() reopen on its next acquire() instead of reusing a
        # now-dead handle.
        self._handles_lock = threading.Lock()
        self._handles: list[Any] = []
        self._generation = 0

    def __getstate__(self) -> tuple:
        return (self._opener, self._path, self._access, self._kwargs)

    def __setstate__(self, state: tuple) -> None:
        opener, path, access, kwargs = state
        type(self).__init__(self, opener, path, access, kwargs)

    def acquire(self) -> Any:
        """Return this thread's handle, opening one on first call (or after close)."""
        entry = getattr(self._local, "entry", None)
        if entry is not None and entry[1] == self._generation:
            return entry[0]
        handle = self._opener(self._path, self._access, **self._kwargs)
        with self._handles_lock:
            self._handles.append(handle)
            generation = self._generation
        self._local.entry = (handle, generation)
        return handle

    @contextmanager
    def acquire_context(self) -> Iterator[Any]:
        """Context manager yielding this thread's handle."""
        yield self.acquire()

    def close(self) -> None:
        """Close every open handle across all threads and reset for reuse."""
        with self._handles_lock:
            handles, self._handles = self._handles, []
            self._generation += 1
        for handle in handles:
            _close_handle(None, handle)
        # Drop the calling thread's cached entry; other threads reopen on their
        # next acquire() because the generation no longer matches.
        self._local.entry = None


def _close_all_cached_handles() -> None:  # pragma: no cover - invoked at exit
    """Close every handle in :data:`FILE_CACHE` at interpreter shutdown."""
    try:
        FILE_CACHE.clear()
    except Exception:  # nosec B110 - best-effort cache clear; must not raise
        pass


atexit.register(_close_all_cached_handles)
