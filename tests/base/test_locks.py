"""Tests for :mod:`pyramids.base._locks`.

locking primitives:

* :class:`SerializableLock` — pickleable, UUID-keyed, same-process lock
  sharing via `WeakValueDictionary`.
* :class:`DummyLock` — no-op lock used when `lock=False`.
* :func:`default_lock` — returns the right lock type for the current
  execution context (SerializableLock in single-process; dask
  distributed Lock when a client is running).
"""

from __future__ import annotations

import pickle
import threading

import pytest

from pyramids.base._locks import DummyLock, SerializableLock, default_lock

pytestmark = pytest.mark.core


class TestSerializableLock:
    """Pickleable lock that shares underlying threading.Lock across pickle."""

    def test_acquire_release(self):
        lk = SerializableLock()
        assert not lk.locked()
        lk.acquire()
        assert lk.locked()
        lk.release()
        assert not lk.locked()

    def test_context_manager(self):
        lk = SerializableLock()
        with lk:
            assert lk.locked()
        assert not lk.locked()

    def test_pickle_shares_underlying_lock(self):
        lk1 = SerializableLock()
        lk2 = pickle.loads(pickle.dumps(lk1))
        assert lk1.lock is lk2.lock

    def test_different_instances_have_different_locks(self):
        lk1 = SerializableLock()
        lk2 = SerializableLock()
        assert lk1.lock is not lk2.lock

    def test_token_stable_across_pickle(self):
        lk1 = SerializableLock()
        lk2 = pickle.loads(pickle.dumps(lk1))
        assert lk1.token == lk2.token

    def test_explicit_token_reuses_lock(self):
        lk1 = SerializableLock()
        lk2 = SerializableLock(token=lk1.token)
        assert lk1.lock is lk2.lock

    def test_non_blocking_acquire(self):
        lk = SerializableLock()
        lk.acquire()
        try:
            # Non-blocking second acquire from same thread should not
            # succeed (threading.Lock is not reentrant).
            acquired = lk.acquire(blocking=False)
            assert acquired is False
        finally:
            lk.release()


class TestDummyLock:
    """No-op lock; acquire never blocks, release is a no-op."""

    def test_acquire_always_true(self):
        lk = DummyLock()
        assert lk.acquire() is True
        assert lk.acquire(blocking=False) is True

    def test_release_no_op(self):
        DummyLock().release()

    def test_context_manager(self):
        with DummyLock() as lk:
            assert isinstance(lk, DummyLock)

    def test_locked_is_always_false(self):
        lk = DummyLock()
        lk.acquire()
        assert lk.locked() is False


class TestDefaultLock:
    """`default_lock` returns SerializableLock in single-process context."""

    def test_returns_serializable_lock_without_dask_client(self):
        lk = default_lock()
        assert isinstance(lk, SerializableLock)

    def test_returned_lock_is_usable(self):
        lk = default_lock()
        with lk:
            assert lk.locked()
        assert not lk.locked()


class TestCrossThread:
    """SerializableLock genuinely protects across threads."""

    def test_lock_protects_concurrent_access(self):
        lk = SerializableLock()
        counter = {"n": 0}

        def worker():
            for _ in range(200):
                with lk:
                    counter["n"] += 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert counter["n"] == 800
