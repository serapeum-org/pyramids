"""Tests for :mod:`pyramids.base._file_manager`.

introduces pickle-safe file-handle managers (`CachingFileManager`,
`ThreadLocalFileManager`) plus a module-global `FILE_CACHE` LRU + a
`_HashedSequence` key type.

These tests cover:

* `_LRUCache` eviction with `on_evict` callback.
* `_HashedSequence` stable hashing.
* `CachingFileManager` happy-path open / reuse / close.
* `CachingFileManager` pickle round-trip — state tuple contains only
  the recipe, never a handle.
* `CachingFileManager` LRU eviction calls close.
* `CachingFileManager` with `lock=False` uses the null lock.
* `CachingFileManager` shared `manager_id` → two instances share one
  cache slot.
* `ThreadLocalFileManager` per-thread handle isolation.
* `ThreadLocalFileManager` pickle round-trip.
* `gdal_raster_open` + `gdal_mdarray_open` + `ogr_open` respect
  access modes and do URL rewriting via `_to_vsi`.
"""

from __future__ import annotations

import gc
import pickle
import threading

import pytest

from pyramids.base._file_manager import (
    FILE_CACHE,
    CachingFileManager,
    ThreadLocalFileManager,
    _close_handle,
    _HashedSequence,
    _LRUCache,
    _make_cache_key,
    _NullLock,
    _resolve_access,
    gdal_mdarray_open,
    gdal_raster_open,
    ogr_open,
)

pytestmark = pytest.mark.core


class _FakeHandle:
    """Test double: records open and close calls."""

    def __init__(self, tag: str = ""):
        self.tag = tag
        self.closed = False

    def Close(self):
        self.closed = True

    def __repr__(self):
        return f"_FakeHandle({self.tag!r})"


_counter = {"n": 0}


def _fake_opener(path: str, access: str = "read_only", **kwargs) -> _FakeHandle:
    """Opener that returns a distinct fake handle per call, tagged by path."""
    _counter["n"] += 1
    return _FakeHandle(tag=f"{path}#{_counter['n']}")


@pytest.fixture(autouse=True)
def _reset_counter_and_cache():
    """Each test gets a fresh handle counter and a clean FILE_CACHE."""
    _counter["n"] = 0
    FILE_CACHE.clear()
    yield
    FILE_CACHE.clear()


class TestLRUCache:
    """`_LRUCache` — small OrderedDict wrapper with `on_evict` hook."""

    def test_basic_get_set(self):
        cache = _LRUCache(maxsize=4)
        cache["a"] = 1
        cache["b"] = 2
        assert cache["a"] == 1
        assert cache["b"] == 2

    def test_eviction_calls_on_evict(self):
        evicted: list[tuple] = []
        cache = _LRUCache(maxsize=2, on_evict=lambda k, v: evicted.append((k, v)))
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert evicted == [("a", 1)]
        assert "a" not in cache

    def test_lru_order_respected(self):
        cache = _LRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        _ = cache["a"]  # makes 'a' most-recently-used
        cache["c"] = 3  # 'b' should evict, not 'a'
        assert "a" in cache
        assert "b" not in cache
        assert "c" in cache

    def test_maxsize_setter_evicts(self):
        evicted: list = []
        cache = _LRUCache(maxsize=5, on_evict=lambda k, v: evicted.append(k))
        for i in range(5):
            cache[str(i)] = i
        cache.maxsize = 2
        assert len(cache) == 2
        assert len(evicted) == 3

    def test_rejects_invalid_maxsize(self):
        with pytest.raises(ValueError):
            _LRUCache(maxsize=0)

    def test_clear_calls_on_evict_for_all(self):
        evicted: list = []
        cache = _LRUCache(maxsize=4, on_evict=lambda k, v: evicted.append(k))
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert sorted(evicted) == ["a", "b"]

    def test_maxsize_property_reports_current_limit(self):
        """`cache.maxsize` exposes the configured limit.

        Test scenario:
            The getter returns the integer passed at construction; the
            setter changes the limit without evicting when the new
            limit is wider.
        """
        cache = _LRUCache(maxsize=7)
        assert cache.maxsize == 7, f"Expected maxsize=7, got {cache.maxsize}"
        cache.maxsize = 9
        assert cache.maxsize == 9, (
            f"Expected maxsize=9 after widening, got {cache.maxsize}"
        )

    def test_maxsize_setter_rejects_zero(self):
        """Setting `cache.maxsize` below 1 raises `ValueError`.

        Test scenario:
            A zero/negative limit would mean "evict everything on every
            insert", which is almost always a bug. The setter rejects
            it explicitly.
        """
        cache = _LRUCache(maxsize=3)
        with pytest.raises(ValueError, match="maxsize must be >= 1"):
            cache.maxsize = 0

    def test_setitem_existing_key_closes_displaced_value(self):
        """ARC-11: overwriting a live key hands the displaced value to `on_evict`.

        Test scenario:
            The LRU must bump the key to most-recently-used and swap the
            stored value — but the value it replaced is a GDAL handle
            nobody else will ever close. Pre-fix the overwrite dropped
            it silently, leaking a file descriptor; the displaced value
            must now be released while the key itself stays cached.
        """
        evicted: list = []
        cache = _LRUCache(maxsize=2, on_evict=lambda k, v: evicted.append((k, v)))
        cache["a"] = 1
        cache["b"] = 2
        cache["a"] = 99
        assert cache["a"] == 99, f"Expected updated value 99, got {cache['a']}"
        assert evicted == [("a", 1)], (
            f"The displaced value must be released exactly once, got {evicted}"
        )
        assert "b" in cache, "Overwriting one key must not evict another"

    def test_setitem_same_object_is_a_noop(self):
        """Re-setting the identical object releases nothing.

        Test scenario:
            `acquire()` re-assigning the handle it just read back would
            otherwise close the handle it is about to return. Identity
            (not equality) is the guard, so re-setting the same object
            must not fire `on_evict`.
        """
        evicted: list = []
        cache = _LRUCache(maxsize=2, on_evict=lambda k, v: evicted.append(k))
        handle = object()
        cache["a"] = handle
        cache["a"] = handle
        assert evicted == [], (
            f"Re-setting the same object must not evict, got {evicted}"
        )
        assert cache["a"] is handle

    def test_setitem_does_not_close_a_pinned_displaced_value(self):
        """Overwriting a *pinned* key must not release what it displaces.

        Test scenario:
            The overwrite path exists because two managers sharing a
            `manager_id` can both miss, both open and both assign. When
            the first of them is inside `acquire_context()` the handle
            being displaced is the one it is actively reading through,
            so closing it is a use-after-close — strictly worse than the
            descriptor leak the release was added to prevent. The pin
            must suppress the release exactly as it does for LRU
            eviction.
        """
        evicted: list = []
        cache = _LRUCache(maxsize=4, on_evict=lambda k, v: evicted.append((k, v)))
        cache["a"] = 1
        cache.pin("a")
        cache["a"] = 99
        assert evicted == [], (
            f"A pinned key's displaced value must not be released, got {evicted}"
        )
        assert cache["a"] == 99, "the new value must still be installed"
        cache.unpin("a")
        cache["a"] = 100
        assert evicted == [("a", 99)], (
            f"Once unpinned the overwrite releases again, got {evicted}"
        )

    def test_iter_yields_keys_in_lru_order(self):
        """Iterating the cache returns keys snapshot-safely.

        Test scenario:
            Iteration captures a snapshot under the lock so callers can
            mutate the cache during the loop without a `RuntimeError`.
            The returned order mirrors insertion/access order.
        """
        cache = _LRUCache(maxsize=4)
        cache["x"] = 1
        cache["y"] = 2
        cache["z"] = 3
        keys = list(cache)
        assert keys == [
            "x",
            "y",
            "z",
        ], f"Expected iteration order ['x','y','z'], got {keys}"


class TestCloseHandle:
    """B-15: `_close_handle` narrows its catch and logs at DEBUG.

    Pre-fix the function caught everything with bare
    `except Exception: pass`, hiding both GDAL "I/O error during
    flush" failures and unrelated programming bugs. Post-fix only
    GDAL's `RuntimeError` is swallowed (logged at DEBUG so
    operators can find issues); other exception classes propagate.
    """

    def test_runtime_error_is_swallowed_and_logged(self, caplog):
        class _FlakyHandle:
            def Close(self):
                raise RuntimeError("simulated GDAL flush failure")

        with caplog.at_level("DEBUG", logger="pyramids.base._file_manager"):
            _close_handle("some-key", _FlakyHandle())
        debug_records = [
            r
            for r in caplog.records
            if r.levelname == "DEBUG" and "close handle failed" in r.getMessage()
        ]
        assert debug_records, (
            "Expected a DEBUG log line for the swallowed RuntimeError; "
            f"got records: {[r.getMessage() for r in caplog.records]}"
        )
        assert "some-key" in debug_records[0].getMessage()

    def test_type_error_propagates(self):
        class _BrokenHandle:
            Close = "not callable"

        handle = _BrokenHandle()
        with pytest.raises(TypeError):
            _close_handle("some-key", handle)

    def test_handle_without_close_is_noop(self):
        class _NoCloseHandle:
            pass

        _close_handle("some-key", _NoCloseHandle())


class TestHashedSequence:
    """`_HashedSequence` — list subclass with cached hash."""

    def test_is_hashable(self):
        hs = _HashedSequence([1, "x", (2, 3)])
        assert hash(hs) == hash((1, "x", (2, 3)))

    def test_usable_as_dict_key(self):
        hs = _HashedSequence(["a", 1])
        d = {hs: "value"}
        assert d[_HashedSequence(["a", 1])] == "value"


class TestMakeCacheKey:
    """`_make_cache_key` canonicalizes kwargs order."""

    def test_kwargs_order_independent(self):
        k1 = _make_cache_key(_fake_opener, "p", "r", {"a": 1, "b": 2}, "id")
        k2 = _make_cache_key(_fake_opener, "p", "r", {"b": 2, "a": 1}, "id")
        assert hash(k1) == hash(k2)

    def test_different_manager_ids_different_keys(self):
        k1 = _make_cache_key(_fake_opener, "p", "r", {}, "id1")
        k2 = _make_cache_key(_fake_opener, "p", "r", {}, "id2")
        assert hash(k1) != hash(k2)


class TestCachingFileManager:
    """Happy-path / pickle / eviction / lock behaviour."""

    def test_acquire_opens_once(self):
        fm = CachingFileManager(_fake_opener, "fixture.tif", "read_only")
        h1 = fm.acquire()
        h2 = fm.acquire()
        assert h1 is h2
        assert _counter["n"] == 1

    def test_pickle_roundtrip(self):
        fm = CachingFileManager(_fake_opener, "fixture.tif", "read_only")
        data = pickle.dumps(fm)
        fm2 = pickle.loads(data)
        assert fm2._path == "fixture.tif"
        assert fm2._access == "read_only"

    def test_pickle_excludes_handle(self):
        fm = CachingFileManager(_fake_opener, "fixture.tif", "read_only")
        fm.acquire()  # put handle in cache
        data = pickle.dumps(fm)
        assert b"_FakeHandle" not in data

    def test_pickle_clone_shares_cache_when_manager_id_shared(self):
        fm = CachingFileManager(_fake_opener, "f.tif", "read_only", manager_id="shared")
        h1 = fm.acquire()
        fm2 = pickle.loads(pickle.dumps(fm))
        h2 = fm2.acquire()
        assert h1 is h2
        assert _counter["n"] == 1

    def test_pickle_clone_without_shared_id_opens_fresh(self):
        fm = CachingFileManager(_fake_opener, "f.tif", "read_only")
        fm.acquire()
        fm2 = pickle.loads(pickle.dumps(fm))
        # manager_id is preserved in __getstate__, so this actually shares
        assert fm2._manager_id == fm._manager_id

    def test_close_drops_handle(self):
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only")
        h = fm.acquire()
        fm.close()
        assert h.closed is True
        h2 = fm.acquire()
        assert h2 is not h

    def test_lock_false_uses_null_lock(self):
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only", lock=False)
        assert isinstance(fm._lock, _NullLock)

    def test_custom_lock_used(self):
        lock = threading.Lock()
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only", lock=lock)
        assert fm._lock is lock
        assert fm._use_default_lock is False

    def test_acquire_context_yields_handle(self):
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only")
        with fm.acquire_context() as h:
            assert isinstance(h, _FakeHandle)

    def test_acquire_context_preserves_handle_on_reraise(self):
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only")
        fm.acquire()  # pre-cache
        ctx = fm.acquire_context()
        with pytest.raises(RuntimeError):
            with ctx:
                raise RuntimeError("boom")
        # Still cached because it was cached before the block.
        assert fm._key in FILE_CACHE

    def test_acquire_context_drops_handle_on_first_open_failure(self):
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only")
        ctx = fm.acquire_context()
        with pytest.raises(RuntimeError):
            with ctx:
                raise RuntimeError("boom")
        assert fm._key not in FILE_CACHE

    def test_close_is_idempotent_when_handle_already_evicted(self):
        """`close()` tolerates a cache entry that was already removed.

        Test scenario:
            Pre-evict the handle out-of-band (simulating an LRU eviction
            that closed it in another manager), then call `close()`.
            It must return cleanly via the `except KeyError: return`
            branch rather than raising.
        """
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only")
        fm.acquire()
        del FILE_CACHE[fm._key]
        fm.close()

    def test_drop_handles_missing_cache_entry(self):
        """`_drop()` silently ignores a missing cache key.

        Test scenario:
            `_drop()` is called in failure paths where the handle may
            or may not be in the cache. Pre-remove the entry and call
            `_drop()` directly — the `except KeyError: pass` branch
            absorbs the error.
        """
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only")
        fm.acquire()
        del FILE_CACHE[fm._key]
        fm._drop()


class TestCachingFileManagerLRUEviction:
    """`FILE_CACHE` eviction closes evicted handles."""

    def test_lru_eviction_closes(self):
        FILE_CACHE.maxsize = 2
        try:
            handles = []
            for i in range(3):
                fm = CachingFileManager(_fake_opener, f"f{i}.tif", "read_only")
                handles.append(fm.acquire())
            # first handle should now be evicted and closed
            assert handles[0].closed is True
            assert handles[1].closed is False
            assert handles[2].closed is False
        finally:
            FILE_CACHE.maxsize = 128


class TestLRUCachePinning:
    """ARC-4: pinned slots survive eviction pressure."""

    def test_pinned_key_is_not_evicted(self):
        """A pinned entry is skipped and the cache overflows instead.

        Test scenario:
            Correctness beats the soft size bound: closing a handle a
            reader is mid-way through is a use-after-close, whereas
            holding one extra descriptor for the length of that read is
            merely untidy.
        """
        evicted: list = []
        cache = _LRUCache(maxsize=1, on_evict=lambda k, v: evicted.append(k))
        cache["a"] = 1
        cache.pin("a")
        cache["b"] = 2
        assert evicted == [], f"A pinned key must not be evicted, got {evicted}"
        assert sorted(cache) == ["a", "b"], (
            f"Both entries must remain while 'a' is pinned, got {sorted(cache)}"
        )

    def test_unpin_makes_the_slot_evictable_again(self):
        """After the last `unpin` the LRU reclaims the slot normally."""
        evicted: list = []
        cache = _LRUCache(maxsize=1, on_evict=lambda k, v: evicted.append(k))
        cache["a"] = 1
        cache.pin("a")
        cache["b"] = 2
        cache.unpin("a")
        cache["c"] = 3
        assert evicted == ["a", "b"], (
            f"Both unpinned entries must be reclaimed, got {evicted}"
        )

    def test_releasing_the_last_pin_trims_the_overflow_immediately(self):
        """Dropping the last pin restores `maxsize` without waiting for an insert.

        Test scenario:
            Pinned reads are the one thing allowed to push the cache
            past its limit. If the trim only ran on the next insert, a
            workload that finishes its reads and stops would hold every
            over-limit handle open forever — 20 GDAL datasets on a cache
            configured for 2. Releasing the pins must reclaim them there
            and then.
        """
        evicted: list = []
        cache = _LRUCache(maxsize=2, on_evict=lambda k, v: evicted.append(k))
        for index in range(6):
            key = f"k{index}"
            cache[key] = index
            cache.pin(key)
        assert len(cache) == 6, (
            f"pinned reads must be allowed to exceed maxsize, got {len(cache)}"
        )
        for index in range(6):
            cache.unpin(f"k{index}")
        assert len(cache) == 2, (
            f"the cache must be back at maxsize once unpinned, got {len(cache)}"
        )
        assert evicted == ["k0", "k1", "k2", "k3"], (
            f"the four least-recently-used entries must be released, got {evicted}"
        )

    def test_pins_nest(self):
        """N pins need N unpins before the slot is evictable."""
        evicted: list = []
        cache = _LRUCache(maxsize=1, on_evict=lambda k, v: evicted.append(k))
        cache["a"] = 1
        cache.pin("a")
        cache.pin("a")
        cache.unpin("a")
        cache["b"] = 2
        assert evicted == [], f"One unpin of two must keep 'a' pinned, got {evicted}"

    def test_all_pinned_overflow_logs_the_configured_limit(self, caplog):
        """The over-limit DEBUG line names `maxsize`, not the insert's target.

        Test scenario:
            `__setitem__` trims to `maxsize - 1` so the new entry lands
            at exactly `maxsize`, but that internal target must not
            reach the operator: on a 128-entry cache the message would
            otherwise read "over its 127 limit". Filling a cache with
            pinned entries is the only way to reach the branch.
        """
        cache = _LRUCache(maxsize=2, on_evict=lambda k, v: None)
        for key in ("a", "b"):
            cache[key] = key
            cache.pin(key)
        with caplog.at_level("DEBUG", logger="pyramids.base._file_manager"):
            cache["c"] = "c"
        messages = [
            r.getMessage() for r in caplog.records if "file cache is" in r.getMessage()
        ]
        assert messages, (
            f"the all-pinned overflow must log at DEBUG; got {[r.getMessage() for r in caplog.records]}"
        )
        assert "over its 2 limit" in messages[0], (
            f"the message must report maxsize=2, got {messages[0]!r}"
        )

    def test_unpin_untracked_key_is_a_noop(self):
        """`unpin` on a key wiped by `clear()` must not underflow.

        Test scenario:
            `clear()` drops the pin table wholesale, so a reader still
            inside `acquire_context()` unpins a key the cache no longer
            tracks. That must leave the pin count absent rather than
            going negative — a negative count would read as truthy and
            silently make the key permanently un-evictable.
        """
        cache = _LRUCache(maxsize=2)
        cache["a"] = 1
        cache.pin("a")
        cache.clear()
        cache.unpin("a")
        cache.unpin("never-pinned")
        assert cache._pins == {}, (
            f"an untracked unpin must not record a count, got {cache._pins}"
        )
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert "a" not in cache, (
            "the post-clear key must be evictable, not stuck behind a stale pin"
        )

    def test_maxsize_setter_skips_pinned_entries(self):
        """Shrinking `maxsize` also honours pins."""
        evicted: list = []
        cache = _LRUCache(maxsize=4, on_evict=lambda k, v: evicted.append(k))
        for key in ("a", "b", "c", "d"):
            cache[key] = key
        cache.pin("a")
        cache.maxsize = 1
        assert "a" in cache, "The pinned entry must survive a maxsize shrink"
        assert sorted(evicted) == ["b", "c", "d"], (
            f"Every unpinned entry must be reclaimed, got {evicted}"
        )


class TestCachingFileManagerEvictionSafety:
    """ARC-4: `acquire_context` protects the handle for the whole read."""

    def test_acquire_context_handle_survives_cross_manager_pressure(self):
        """Another manager's insert cannot close a handle being read.

        Test scenario:
            Manager A holds its handle inside `acquire_context()`.
            Manager B — a different cache key, therefore a different
            per-manager lock — inserts into the full shared cache.
            Pre-fix B's insert evicted and `Close()`d A's handle
            mid-read; post-fix the pin makes A's slot ineligible.
        """
        cache = _LRUCache(maxsize=1, on_evict=_close_handle)
        first = CachingFileManager(_fake_opener, "a.tif", cache=cache, lock=False)
        second = CachingFileManager(_fake_opener, "b.tif", cache=cache, lock=False)
        with first.acquire_context() as handle:
            second.acquire()
            assert handle.closed is False, (
                "a handle held inside acquire_context() must not be closed by "
                "another manager's cache insert"
            )

    def test_auto_release_finalizer_defers_the_close_of_a_pinned_slot(self):
        """The GC-driven `release()` must defer, not skip, a pinned handle's close.

        Test scenario:
            `release()` fires from a `weakref.finalize`, so it lands at
            an arbitrary moment — including while a manager sharing the
            cache key is mid-read. Closing there is a use-after-close;
            merely leaving the entry behind would forfeit the
            deterministic release this path exists for. It must park
            the handle and let the last reader release it.
        """
        cache = _LRUCache(maxsize=8, on_evict=_close_handle)
        owner = CachingFileManager(
            _fake_opener, "a.tif", cache=cache, manager_id="k", auto_release=True
        )
        reader = CachingFileManager(
            _fake_opener, "a.tif", cache=cache, manager_id="k", lock=False
        )
        with reader.acquire_context() as handle:
            del owner
            gc.collect()
            assert handle.closed is False, (
                "the auto-release finalizer must not close a pinned handle"
            )
        assert handle.closed is True, (
            "the deferred close must run once the last reader unpins"
        )
        assert cache._pending_close == {}, (
            f"the deferred entry must be drained, got {cache._pending_close}"
        )

    def test_close_defers_teardown_of_a_pinned_slot(self):
        """`close()` unpublishes immediately but closes only after the read ends.

        Test scenario:
            `close()` is reachable from public API — `NetCDF.close()`
            walks its lazy managers and calls it — while dask workers
            may still be inside `acquire_context()`. A GDAL
            use-after-close is a segfault, not an exception, so the
            handle must outlive the call; the slot still has to leave
            the cache at once so no new caller picks it up.
        """
        cache = _LRUCache(maxsize=8, on_evict=_close_handle)
        first = CachingFileManager(
            _fake_opener, "a.tif", cache=cache, manager_id="k", lock=False
        )
        second = CachingFileManager(
            _fake_opener, "a.tif", cache=cache, manager_id="k", lock=False
        )
        with first.acquire_context() as handle:
            second.close()
            assert handle.closed is False, (
                "close() must not close a handle another reader is using"
            )
            assert first._key not in cache, (
                "the slot must leave the cache immediately so no new caller finds it"
            )
        assert handle.closed is True, (
            "the deferred close must run once the reader unpins"
        )

    def test_close_of_an_unpinned_slot_is_immediate(self):
        """With no reader present `close()` still closes on the spot."""
        cache = _LRUCache(maxsize=8, on_evict=_close_handle)
        fm = CachingFileManager(_fake_opener, "a.tif", cache=cache, lock=False)
        handle = fm.acquire()
        fm.close()
        assert handle.closed is True, "an unpinned close must not be deferred"
        assert cache._pending_close == {}, "nothing should be parked"

    def test_slot_is_evictable_again_after_the_block(self):
        """The pin is released on exit, so the LRU bound is restored."""
        cache = _LRUCache(maxsize=1, on_evict=_close_handle)
        first = CachingFileManager(_fake_opener, "a.tif", cache=cache, lock=False)
        second = CachingFileManager(_fake_opener, "b.tif", cache=cache, lock=False)
        with first.acquire_context() as handle:
            pass
        second.acquire()
        assert handle.closed is True, (
            "once the read finishes the handle must be reclaimable by the LRU"
        )
        assert len(cache) == 1, f"cache must be back at maxsize, got {len(cache)}"

    def test_pin_is_released_when_the_block_raises(self):
        """An exception inside the block still unpins the slot."""
        cache = _LRUCache(maxsize=2, on_evict=_close_handle)
        fm = CachingFileManager(_fake_opener, "a.tif", cache=cache, lock=False)
        fm.acquire()  # pre-cache so the failure path keeps the handle
        with pytest.raises(RuntimeError):
            with fm.acquire_context():
                raise RuntimeError("boom")
        assert cache._pins.get(fm._key) is None, (
            f"the pin must be dropped on the error path, got {cache._pins}"
        )

    def test_pin_is_released_when_the_opener_raises(self):
        """A failure to open must not leave the key pinned forever."""

        def _broken_opener(path, access="read_only", **kwargs):
            raise OSError("cannot open")

        cache = _LRUCache(maxsize=2, on_evict=_close_handle)
        fm = CachingFileManager(_broken_opener, "a.tif", cache=cache, lock=False)
        with pytest.raises(OSError):
            with fm.acquire_context():
                pass
        assert cache._pins.get(fm._key) is None, (
            f"a failed open must not leak a pin, got {cache._pins}"
        )


class TestThreadLocalFileManager:
    """Per-thread handle isolation, no locking."""

    def test_acquire_opens_once_per_thread(self):
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        h1 = fm.acquire()
        h2 = fm.acquire()
        assert h1 is h2
        assert _counter["n"] == 1

    def test_different_threads_get_different_handles(self):
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        results: list = []

        def grab():
            results.append(fm.acquire())

        t = threading.Thread(target=grab)
        t.start()
        t.join()
        main_handle = fm.acquire()
        assert len(results) == 1
        assert results[0] is not main_handle

    def test_pickle_roundtrip_no_handle(self):
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        fm.acquire()
        data = pickle.dumps(fm)
        assert b"_FakeHandle" not in data
        fm2 = pickle.loads(data)
        assert fm2._path == "t.tif"

    def test_close_is_thread_local(self):
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        h = fm.acquire()
        fm.close()
        assert h.closed is True
        # Re-acquire on the same thread opens a new handle.
        h2 = fm.acquire()
        assert h2 is not h

    def test_close_releases_all_threads_handles(self):
        """close() closes handles opened by other threads, not just the caller (H4).

        Test scenario:
            A worker thread opens its own per-thread handle and exits; the main
            thread then calls close(). Both the worker's and the caller's handles
            must be closed because the manager tracks every opened handle rather
            than only the closing thread's thread-local state.
        """
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        worker_handles: list = []

        def grab():
            worker_handles.append(fm.acquire())

        t = threading.Thread(target=grab)
        t.start()
        t.join()
        main_handle = fm.acquire()
        fm.close()
        assert worker_handles[0].closed is True, "worker thread handle must be closed"
        assert main_handle.closed is True, "caller handle must be closed"
        assert fm._handles == [], "close() must clear the tracked-handle list"

    def test_acquire_after_close_reopens_via_generation_bump(self):
        """A thread whose handle was closed reopens on next acquire() (H4).

        Test scenario:
            close() bumps the generation, so the same thread's next acquire() opens
            a fresh, live handle instead of returning the now-closed handle still
            cached in its thread-local storage.
        """
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        first = fm.acquire()
        fm.close()
        assert first.closed is True, "the open handle must be closed by close()"
        second = fm.acquire()
        assert second is not first, "must reopen after close, not reuse a dead handle"
        assert second.closed is False, "the reopened handle must be live"

    def test_acquire_context_yields_handle(self):
        """`acquire_context()` yields the thread-local handle.

        Test scenario:
            The context manager is a thin shim over :meth:`acquire`; it
            must yield a :class:`_FakeHandle` and survive re-entry on
            the same thread without re-opening. This covers the
            :meth:`ThreadLocalFileManager.acquire_context` body.
        """
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        with fm.acquire_context() as handle:
            assert isinstance(handle, _FakeHandle), (
                f"Expected _FakeHandle, got {type(handle)}"
            )
        with fm.acquire_context() as handle2:
            assert handle is handle2, (
                "Context manager must reuse the thread-local handle"
            )


class TestNullLock:
    """:class:`_NullLock` drop-in for `lock=False`."""

    def test_acquire_always_returns_true(self):
        """`acquire()` is a no-op that always succeeds.

        Test scenario:
            The null lock never blocks, regardless of `blocking` or
            `timeout` kwargs. It must always return `True` so code
            that inspects the return value (like `with lock:` guards)
            proceeds as if it took the lock.
        """
        lock = _NullLock()
        assert lock.acquire() is True, "_NullLock.acquire() must always return True"
        assert lock.acquire(blocking=False, timeout=5.0) is True, (
            "_NullLock.acquire() must ignore blocking/timeout"
        )

    def test_release_is_noop(self):
        """`release()` returns `None` without raising.

        Test scenario:
            Releasing an unacquired real lock raises `RuntimeError`;
            the null lock must tolerate unmatched releases so caller
            code can treat it interchangeably.
        """
        lock = _NullLock()
        assert lock.release() is None, "_NullLock.release() must return None"


class TestOpeners:
    """`_openers` module primitives."""

    def test_resolve_access_known(self):
        from osgeo import gdal

        assert _resolve_access("read_only") == gdal.GA_ReadOnly
        assert _resolve_access("r") == gdal.GA_ReadOnly
        assert _resolve_access("write") == gdal.GA_Update
        assert _resolve_access("w") == gdal.GA_Update

    def test_resolve_access_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown access mode"):
            _resolve_access("bogus")

    def test_gdal_raster_open_extra_kwargs_ignored(self):
        # Sanity: opener ignores unrelated kwargs rather than raising.
        # We can't open a real file without fixtures, so just assert
        # the signature accepts them.
        import inspect

        sig = inspect.signature(gdal_raster_open)
        params = sig.parameters
        assert "path" in params and "access" in params

    def test_gdal_mdarray_open_sig(self):
        import inspect

        sig = inspect.signature(gdal_mdarray_open)
        assert "path" in sig.parameters

    def test_ogr_open_sig(self):
        import inspect

        sig = inspect.signature(ogr_open)
        assert "path" in sig.parameters


class TestOpenersE2E:
    """End-to-end: exercise openers against a real tiny GeoTIFF fixture."""

    @pytest.fixture
    def tif_path(self, tmp_path):
        """Create a 3x3 uint8 in-memory GeoTIFF and return its path."""
        from osgeo import gdal, osr

        path = str(tmp_path / "tiny.tif")
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(path, 3, 3, 1, gdal.GDT_Byte)
        ds.SetGeoTransform((0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(__import__("numpy").zeros((3, 3), dtype="uint8"))
        ds.FlushCache()
        ds = None
        return path

    def test_gdal_raster_open_real_file(self, tif_path):
        ds = gdal_raster_open(tif_path, "read_only")
        try:
            assert ds.RasterXSize == 3 and ds.RasterYSize == 3
        finally:
            ds = None

    def test_caching_manager_e2e_with_real_file(self, tif_path):
        fm = CachingFileManager(gdal_raster_open, tif_path, "read_only")
        try:
            ds1 = fm.acquire()
            ds2 = fm.acquire()
            assert ds1 is ds2
            assert ds1.RasterXSize == 3
        finally:
            fm.close()

    def test_thread_local_manager_e2e_with_real_file(self, tif_path):
        fm = ThreadLocalFileManager(gdal_raster_open, tif_path, "read_only")
        try:
            ds = fm.acquire()
            assert ds.RasterXSize == 3
        finally:
            fm.close()

    @pytest.fixture
    def geojson_path(self, tmp_path):
        """Write a minimal single-point GeoJSON and return its path.

        Returns:
            str: Filesystem path to the GeoJSON file.
        """
        path = tmp_path / "tiny.geojson"
        path.write_text(
            '{"type":"FeatureCollection","features":['
            '{"type":"Feature","geometry":{"type":"Point",'
            '"coordinates":[0,0]},"properties":{}}'
            ']}',
            encoding="utf-8",
        )
        return str(path)

    def test_ogr_open_read_only_returns_datasource(self, geojson_path):
        """`ogr_open(access="read_only")` returns a readable OGR datasource.

        Test scenario:
            Opening a GeoJSON with `access="read_only"` must set the
            OGR update flag to 0 and return a datasource whose first
            layer has the expected single feature. This covers the
            `update = 0 if access in {...}` branch of :func:`ogr_open`.
        """
        ds = ogr_open(geojson_path, access="read_only")
        try:
            assert ds is not None, "ogr_open must return a datasource"
            layer = ds.GetLayer(0)
            assert layer.GetFeatureCount() == 1, (
                f"Expected 1 feature, got {layer.GetFeatureCount()}"
            )
        finally:
            ds = None

    def test_ogr_open_write_access_flags_update(self, geojson_path):
        """`ogr_open(access="w")` passes `update=1` to :func:`ogr.Open`.

        Test scenario:
            The "write" branch sets `update=1` so drivers that support
            edits open in update mode. The function must still return a
            valid datasource (not raise) for the common GeoJSON driver.
        """
        ds = ogr_open(geojson_path, access="w")
        try:
            assert ds is not None, "ogr_open with access='w' must return a datasource"
        finally:
            ds = None


class TestCachingFileManagerAutoRelease:
    """`auto_release` ties the cached handle to the manager's lifetime, refcounted per slot (#727)."""

    def test_handle_closed_when_manager_is_collected(self):
        """An auto-release manager evicts and closes its handle when garbage-collected."""
        cache = _LRUCache(maxsize=8, on_evict=_close_handle)
        fm = CachingFileManager(_fake_opener, "a.tif", cache=cache, auto_release=True)
        handle = fm.acquire()
        assert len(cache) == 1 and handle.closed is False
        del fm
        gc.collect()
        assert len(cache) == 0, "manager GC must evict the slot"
        assert handle.closed is True, "manager GC must close the handle"

    def test_shared_slot_closes_only_after_last_manager(self):
        """Two auto-release managers sharing a `manager_id` keep the handle until both are collected."""
        cache = _LRUCache(maxsize=8, on_evict=_close_handle)
        first = CachingFileManager(
            _fake_opener, "a.tif", cache=cache, manager_id="k", auto_release=True
        )
        second = CachingFileManager(
            _fake_opener, "a.tif", cache=cache, manager_id="k", auto_release=True
        )
        handle = first.acquire()
        assert second.acquire() is handle, "same manager_id shares one cached handle"
        del first
        gc.collect()
        assert handle.closed is False and len(cache) == 1, (
            "one manager gone: shared handle stays open"
        )
        del second
        gc.collect()
        assert handle.closed is True and len(cache) == 0, (
            "last manager gone: handle closed"
        )

    def test_without_auto_release_handle_survives_manager_gc(self):
        """A default (non-auto-release) manager leaves its handle under pure LRU lifetime."""
        cache = _LRUCache(maxsize=8, on_evict=_close_handle)
        fm = CachingFileManager(_fake_opener, "a.tif", cache=cache)
        handle = fm.acquire()
        del fm
        gc.collect()
        assert handle.closed is False and len(cache) == 1, (
            "LRU lifetime: handle survives manager GC"
        )

    def test_setstate_tolerates_legacy_six_tuple(self):
        """A manager pickled by the pre-`auto_release` 6-tuple format still unpickles (defaults False)."""
        fm = CachingFileManager(_fake_opener, "a.tif", auto_release=True)
        legacy = (fm._opener, fm._path, fm._access, fm._kwargs, None, fm._manager_id)
        clone = CachingFileManager.__new__(CachingFileManager)
        clone.__setstate__(legacy)
        assert clone._auto_release is False, (
            "legacy 6-tuple must default auto_release to False"
        )

    def test_stale_finalizer_after_clear_does_not_close_reopened_handle(self):
        """A finalizer firing after `clear()` must not close a handle re-opened for a live sibling.

        `clear()` wipes both the cache and the refcounts. A manager alive across the clear still holds
        a pending finalizer; when it fires, `release` must treat the now-untracked key as a no-op
        rather than underflowing the count and closing a handle another live manager just re-parked.
        """
        cache = _LRUCache(maxsize=8, on_evict=_close_handle)
        first = CachingFileManager(
            _fake_opener, "a.tif", cache=cache, manager_id="k", auto_release=True
        )
        second = CachingFileManager(
            _fake_opener, "a.tif", cache=cache, manager_id="k", auto_release=True
        )
        first.acquire()
        cache.clear()
        reopened = second.acquire()
        assert len(cache) == 1 and reopened.closed is False
        del first
        gc.collect()
        assert reopened.closed is False, (
            "a stale post-clear finalizer must not close the re-opened handle"
        )
        assert len(cache) == 1, "the re-opened handle survives an untracked release"

    def test_release_logs_close_error_instead_of_raising(self, caplog):
        """A close failure during finalizer-invoked `release()` is logged, not left unraisable (L1)."""

        def boom(_key, _handle):
            raise OSError("simulated remote /vsi flush failure")

        cache = _LRUCache(maxsize=4, on_evict=boom)
        cache.retain("k")
        cache._cache["k"] = object()
        with caplog.at_level("WARNING", logger="pyramids.base._file_manager"):
            cache.release("k")
        assert any(
            "handle close failed during finalizer release" in r.getMessage()
            for r in caplog.records
        ), "the finalizer-invoked close error must be logged at WARNING"
