"""Lock primitives for safe concurrent access to GDAL/OGR handles.

Three shapes are exported:

* :class:`SerializableLock` — a pickleable lock keyed by a UUID token
  plus a process-wide `WeakValueDictionary` of `token -> Lock`.
  Pickling only transmits the token, so unpickled copies in the same
  process map back to the same underlying :class:`threading.Lock`
  while different processes get independent locks. This is the shape
  rioxarray / xarray use for dask-backed IO and is the correct
  default for pyramids' lazy read paths.
* :class:`DummyLock` — a no-op lock that never blocks. Used when a
  caller passes `lock=False` to opt into lock-free reads (per-thread
  handle, no cross-thread mutex — good for cloud COG reads where
  GDAL's block cache is per-handle anyway).
* :func:`default_lock` — one-line factory that returns a fresh
  :class:`SerializableLock` in a single-process context, or a
  :class:`dask.distributed.Lock` when a running dask client is
  detected. Prefer this over direct instantiation when a method
  accepts `lock=None` as "use the right thing".

The dask import is gated: importing this module does not pull
`dask.distributed`. `default_lock` only imports it when it
actually needs to probe for a running client.
"""

from __future__ import annotations

import threading
import uuid
import weakref
from typing import Any

_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()


class SerializableLock:
    """Pickleable lock keyed by a UUID token.

    Each instance holds a single `threading.Lock` plus a token
    string. On pickle, only the token is serialized; on unpickle, if
    the same token is already resident in the current process's
    `_LOCKS` dict, the unpickled copy maps to the same underlying
    lock (safe for in-process dask schedulers). Otherwise it creates
    a new lock in the target process (correct for cross-process dask
    because different processes should not share a Python-level
    mutex).

    Modeled on :class:`xarray.backends.locks.SerializableLock`.

    Examples:
        - Two instances with the same token share the underlying lock:
            ```python
            >>> from pyramids.base._locks import SerializableLock
            >>> import pickle
            >>> a = SerializableLock()
            >>> b = pickle.loads(pickle.dumps(a))
            >>> a.lock is b.lock
            True

            ```
        - Different instances have different underlying locks:
            ```python
            >>> from pyramids.base._locks import SerializableLock
            >>> a = SerializableLock()
            >>> b = SerializableLock()
            >>> a.lock is b.lock
            False

            ```
    """

    def __init__(self, token: str | None = None):
        self.token = token or str(uuid.uuid4())
        lock = _LOCKS.get(self.token)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[self.token] = lock
        self.lock = lock

    def __getstate__(self) -> str:
        return self.token

    def __setstate__(self, token: str) -> None:
        self.__init__(token)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self.lock.acquire(blocking, timeout)

    def release(self) -> None:
        self.lock.release()

    def __enter__(self) -> SerializableLock:
        self.lock.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self.lock.release()

    def locked(self) -> bool:
        """Return True if the underlying lock is held by any thread."""
        return self.lock.locked()


class DummyLock:
    """No-op lock that never blocks.

    Drop-in substitute when a caller passes `lock=False`. Callers
    get the same context-manager / acquire / release surface without
    paying any synchronization cost.

    Examples:
        - Can be used as a context manager:
            ```python
            >>> from pyramids.base._locks import DummyLock
            >>> with DummyLock():
            ... value = 42
            >>> value
            42

            ```
        - `acquire` always succeeds, even under "contention":
            ```python
            >>> from pyramids.base._locks import DummyLock
            >>> lk = DummyLock()
            >>> lk.acquire(blocking=False)
            True
            >>> lk.locked()
            False

            ```
    """

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return True

    def release(self) -> None:
        pass

    def __enter__(self) -> DummyLock:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def locked(self) -> bool:
        return False


def default_lock() -> Any:
    """Return the right lock for the current execution context.

    Probes for a running :class:`dask.distributed.Client`; if one is
    active, returns a `dask.distributed.Lock` (cross-process mutex).
    Otherwise returns a fresh :class:`SerializableLock` (in-process,
    pickle-safe).

    Does **not** import `dask.distributed` unless a client is
    actually probed, so calling this in a dask-free environment is
    free.

    Returns:
        A lock-protocol object supporting `acquire`, `release`,
        and context-manager use.

    Examples:
        - In a single-process environment, returns a SerializableLock:
            ```python
            >>> from pyramids.base._locks import default_lock, SerializableLock
            >>> lock = default_lock()
            >>> isinstance(lock, SerializableLock)
            True

            ```
    """
    try:
        from dask.distributed import Lock as _DistributedLock
        from dask.distributed import get_client

        client = get_client()
    except (ImportError, ValueError):
        lock: Any = SerializableLock()
    else:
        lock = _DistributedLock(name=f"pyramids-{uuid.uuid4()}", client=client)
    return lock
