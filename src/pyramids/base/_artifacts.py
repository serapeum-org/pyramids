"""Process-scoped scratch artefacts for materialising readers (M1).

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

from osgeo import gdal

_ROOT: str | None = None
_VSIMEM_PATHS: list[str] = []


def _root() -> str:
    """Return the shared process artefact root, creating it on first use."""
    global _ROOT
    if _ROOT is None:
        _ROOT = tempfile.mkdtemp(prefix="pyramids_stac_")
        atexit.register(cleanup)
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

    Args:
        path: The in-memory GDAL path (e.g. ``/vsimem/foo.vrt``).
    """
    _VSIMEM_PATHS.append(path)


def cleanup() -> None:
    """Remove the artefact root and unlink every tracked ``/vsimem`` path.

    Registered as an ``atexit`` hook the first time the root is created; safe to
    call directly (e.g. from tests). Best-effort — errors are swallowed so a
    locked file at shutdown never raises.
    """
    global _ROOT
    while _VSIMEM_PATHS:
        path = _VSIMEM_PATHS.pop()
        try:
            gdal.Unlink(path)
        except Exception:  # noqa: BLE001 - best-effort shutdown cleanup
            pass
    if _ROOT is not None and os.path.isdir(_ROOT):
        shutil.rmtree(_ROOT, ignore_errors=True)
    _ROOT = None
