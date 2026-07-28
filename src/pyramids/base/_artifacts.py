"""Process-scoped scratch artefacts for readers that must outlive their call.

Several STAC readers must materialise intermediate rasters that have to outlive
the call because the returned object is *file-backed* by them: multi-asset
``from_stac`` and ``groupby="solar_day"`` write per-item / per-day GeoTIFFs, and
``build_vrt_from_stac`` writes an in-memory ``/vsimem`` VRT. Previously each call
made its own ``tempfile.mkdtemp`` (or a fresh ``/vsimem`` path) that was **never
cleaned up**, so a long-running process leaked disk / memory indefinitely.

This module centralises that scratch space:

* :func:`artifact_dir` returns a fresh unique directory under **one** shared,
  process-level temp root (so N calls create N small subdirs under a single
  root, not N independent roots).
* :func:`register_vsimem` tracks a ``/vsimem`` path for unlink.
* both the root and the tracked ``/vsimem`` paths are removed by an ``atexit``
  hook, so everything is reclaimed at interpreter shutdown.

The artefacts still live for the rest of the process (the file-backed
collections need them), but they no longer accumulate one orphaned root per
call and are guaranteed to be cleaned up on exit.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import threading

from osgeo import gdal

_ROOT: str | None = None
_VSIMEM_PATHS: list[str] = []
_CLEANUP_ARMED = False

_LOCK = threading.Lock()
"""Guards the module globals.

Two threads building a VRT each mutate `_VSIMEM_PATHS` and race `_CLEANUP_ARMED`,
and `cleanup()` pops from the same list — so a test-then-mutate on it (checking
membership, then removing) can lose the race and raise. That matters because
`unregister_vsimem` runs from a failure handler, where a `ValueError` from the
cleanup would replace the real build error."""


def _arm_cleanup() -> None:
    """Register the exit sweep, once per process.

    Both artefact kinds need it and a process may use only one of them: hanging
    the registration off the temp-root creation meant a process that only ever
    called :func:`build_vrt_from_stac` — which uses `/vsimem`, never the temp
    root — never armed the sweep, so its tracked VRTs were never reclaimed.
    """
    global _CLEANUP_ARMED
    with _LOCK:
        if not _CLEANUP_ARMED:
            atexit.register(cleanup)
            _CLEANUP_ARMED = True


def _root() -> str:
    """Return the shared process artefact root, creating it on first use."""
    global _ROOT
    if _ROOT is None:
        # Generic prefix: the registry started out serving the STAC readers but
        # is now the shared scratch space for anything whose artefacts must
        # outlive the call (zip extraction in pyramids.io among them), so the
        # directory name should not imply a single consumer.
        _ROOT = tempfile.mkdtemp(prefix="pyramids_scratch_")
        _arm_cleanup()
    return _ROOT


def artifact_dir() -> str:
    """Return a fresh unique directory under the shared process artefact root.

    Each call creates a new subdirectory (so artefacts from different calls do
    not collide), but all subdirectories live under one root that is removed at
    process exit.

    Returns:
        The path of a newly created, empty directory.

    Examples:
        - Two calls return distinct directories under one shared parent:
            ```python
            >>> import os
            >>> from pyramids.base._artifacts import artifact_dir
            >>> a, b = artifact_dir(), artifact_dir()
            >>> a != b and os.path.dirname(a) == os.path.dirname(b)
            True
            >>> os.path.isdir(a)
            True

            ```
    """
    return tempfile.mkdtemp(dir=_root())


def register_vsimem(path: str) -> None:
    """Track a ``/vsimem`` path to be unlinked at process exit.

    Arms the exit sweep, so a process that produces only in-memory artefacts
    (one ``build_vrt_from_stac`` per request, say) still reclaims them at
    shutdown rather than growing an in-memory VRT per call for its lifetime.

    Args:
        path: The in-memory GDAL path (e.g. ``/vsimem/foo.vrt``).
    """
    with _LOCK:
        _VSIMEM_PATHS.append(path)
    _arm_cleanup()


def unregister_vsimem(path: str) -> None:
    """Stop tracking a ``/vsimem`` path, for a caller that unlinked it early.

    Keeps the registry from accruing dead entries when a build fails and
    reclaims its own artefact — otherwise :func:`cleanup` later unlinks paths
    that no longer exist.

    Never raises: it is called from a failure handler, where a `ValueError` from
    losing a race against `cleanup()` would replace the error the caller is
    actually reporting.

    Args:
        path: The in-memory GDAL path to forget. Unknown paths are ignored.
    """
    with _LOCK:
        # Rebuild rather than test-then-remove: one atomic assignment, and no
        # window in which a concurrent `cleanup()` pop makes `remove` raise.
        _VSIMEM_PATHS[:] = [tracked for tracked in _VSIMEM_PATHS if tracked != path]


def cleanup() -> None:
    """Remove the artefact root and unlink every tracked ``/vsimem`` path.

    Registered as an ``atexit`` hook the first time either artefact kind is
    used. Best-effort — errors are swallowed so a locked file at shutdown
    never raises.

    Calling this directly removes the root shared by *every* consumer in the
    process (the STAC readers and ``pyramids.io``'s zip extraction among
    them), so a test that invokes it must isolate itself first.
    """
    global _ROOT
    with _LOCK:
        pending = list(_VSIMEM_PATHS)
        _VSIMEM_PATHS.clear()
    while pending:
        path = pending.pop()
        try:
            gdal.Unlink(path)
        except Exception:  # noqa: BLE001  # nosec B110 - best-effort shutdown cleanup
            pass
    if _ROOT is not None and os.path.isdir(_ROOT):
        shutil.rmtree(_ROOT, ignore_errors=True)
    _ROOT = None
