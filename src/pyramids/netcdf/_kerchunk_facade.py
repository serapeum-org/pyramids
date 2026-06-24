"""Kerchunk reference-manifest emit for NetCDF files.

serialise a NetCDF (or a list of NetCDFs) into a kerchunk
JSON reference manifest so downstream consumers can open the archive
as a lazy Zarr-backed xarray cube with **zero rewrite**. The manifest
is a small JSON document containing byte-range pointers into each
source file; no pixel data is moved.

Two helpers:

* :func:`to_kerchunk` — single-file manifest, wraps
  :class:`kerchunk.hdf.SingleHdf5ToZarr`.
* :func:`combine_kerchunk` — multi-file manifest, wraps
  :class:`kerchunk.combine.MultiZarrToZarr`, concatenating along a
  user-specified dimension (usually `"time"`).

Kerchunk is not a hard dependency — it lives in the
`[lazy]` optional extra. Helpers raise a clear
:class:`ImportError` when kerchunk is missing.
"""

from __future__ import annotations

import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from pyramids.base._utils import lazy_extra_hint
from pyramids.netcdf._kerchunk_builder import build_single_manifest, combine_manifests

_KERCHUNK_IMPORT_ERROR = lazy_extra_hint(
    "kerchunk is required for NetCDF → Zarr reference manifests."
)


@contextmanager
def _scalar_fill_value_shim() -> Iterator[None]:
    """Make kerchunk tolerate a non-scalar ``_FillValue`` under numpy >= 2.

    GDAL's netCDF driver writes ``_FillValue`` as a **1-element array**, not a
    scalar. kerchunk's HDF translator passes that straight to
    ``zarr.meta.encode_fill_value``, which calls ``float()`` on it — harmless on
    numpy 1.x (a deprecation warning) but a hard ``TypeError`` on numpy >= 2.x
    ("only 0-dimensional arrays can be converted to Python scalars"). kerchunk
    catches that, warns, and **drops the whole variable** from the manifest, so a
    later open raises ``KeyError``. This context manager squeezes a size-1 fill
    value to a Python scalar before encoding so the variable survives; it is a
    no-op for scalar / ``None`` fill values and is fully reverted on exit.

    The patch targets the ``encode_fill_value`` symbol that kerchunk and zarr
    actually call, restoring the originals in a ``finally`` so global state is
    never left modified.
    """
    # Optional-dependency imports (kerchunk/zarr live in the [lazy]
    # extra); resolved lazily, mirroring _require_kerchunk_single below.
    modules: list[Any] = []
    try:
        from kerchunk import hdf as _khdf

        if hasattr(_khdf, "encode_fill_value"):
            modules.append(_khdf)
    except ImportError:
        pass
    try:
        from zarr import meta as _zmeta

        if hasattr(_zmeta, "encode_fill_value"):
            modules.append(_zmeta)
    except ImportError:
        pass
    if not modules:
        yield
        return

    original = modules[0].encode_fill_value

    def _scalarized(value: Any, dtype: Any, object_codec: Any = None) -> Any:
        if (
            value is not None
            and getattr(value, "ndim", 0)
            and np.asarray(value).size == 1
        ):
            value = np.asarray(value).reshape(()).item()
        return original(value, dtype, object_codec)

    for module in modules:
        module.encode_fill_value = _scalarized
    try:
        yield
    finally:
        for module in modules:
            module.encode_fill_value = original


def _require_kerchunk_single() -> Any:
    """Lazy-import :class:`kerchunk.hdf.SingleHdf5ToZarr`."""
    try:
        from kerchunk.hdf import SingleHdf5ToZarr
    except ImportError as exc:
        raise ImportError(_KERCHUNK_IMPORT_ERROR) from exc
    return SingleHdf5ToZarr


def _require_kerchunk_combine() -> Any:
    """Lazy-import :class:`kerchunk.combine.MultiZarrToZarr`."""
    try:
        from kerchunk.combine import MultiZarrToZarr
    except ImportError as exc:
        raise ImportError(_KERCHUNK_IMPORT_ERROR) from exc
    return MultiZarrToZarr


def _kerchunk_translate_single(
    src_str: str, *, inline_threshold: int, vlen_encode: str
) -> dict[str, Any]:
    """Generate a single-file manifest via kerchunk's HDF5 translator.

    Fallback path for files the native builder cannot handle. Under zarr v3 this
    can flakily deadlock (#530), so it is only used when the native builder
    raises on an unsupported HDF5 feature.
    """
    single_hdf5_to_zarr = _require_kerchunk_single()
    with _scalar_fill_value_shim():
        refs = single_hdf5_to_zarr(
            src_str,
            src_str,
            inline_threshold=inline_threshold,
            error="warn",
            vlen_encode=vlen_encode,
        ).translate()
    return refs


def _native_or_fallback(
    native_fn: Callable[[], dict[str, Any]],
    legacy_fn: Callable[[], dict[str, Any]],
    *,
    local_inputs_exist: bool,
    subject: str,
) -> dict[str, Any]:
    """Run the native manifest builder, falling back to the kerchunk translator on failure.

    Shared by :func:`to_kerchunk` and :func:`combine_kerchunk`. An ``OSError`` is re-raised
    when every input is a local file that exists — that signals a genuinely corrupt /
    unreadable HDF5 file, not a remote URL the local-only native builder cannot open (which
    kerchunk's fsspec-backed translator can). Any ``ValueError`` (unsupported HDF5 feature, or
    a combine layout the native path doesn't support) always falls back.

    Args:
        native_fn: Zero-arg callable producing the manifest via the native builder.
        legacy_fn: Zero-arg callable producing the manifest via the kerchunk translator.
        local_inputs_exist: ``True`` when every source path is an existing local file.
        subject: Lead clause of the fallback warning (the ``({exc}); falling back …`` tail
            is appended here).

    Returns:
        The manifest dict from whichever path succeeded.
    """
    try:
        return native_fn()
    except (ValueError, OSError) as exc:
        if isinstance(exc, OSError) and local_inputs_exist:
            raise
        warnings.warn(
            f"{subject} ({exc}); falling back to the kerchunk translator.",
            stacklevel=3,
        )
        return legacy_fn()


def to_kerchunk(
    src_path: str | Path,
    output_path: str | Path,
    *,
    inline_threshold: int = 500,
    vlen_encode: str = "embed",
    backend: str = "native",
) -> dict[str, Any]:
    """Emit a single-file kerchunk reference manifest.

    Args:
        src_path: Path or URL of the source NetCDF / HDF5 file.
        output_path: Path where the JSON manifest is written.
        inline_threshold: Chunks smaller than this many bytes are
            embedded directly in the manifest rather than referenced
            by offset.
        vlen_encode: One of `"embed" | "null" | "leave" | "encode"`.
            Controls how VLEN (variable-length) strings are handled.
            Default `"embed"` inlines string values; other modes
            trade compatibility vs fidelity — see the kerchunk docs.
        backend: `"native"` (default) builds the manifest directly with
            h5py — no live zarr group, so it cannot hit the zarr-v3
            `sync()` deadlock (#530). `"kerchunk"` forces the legacy
            `kerchunk.hdf.SingleHdf5ToZarr` translator. The native
            backend falls back to kerchunk for files using HDF5 features
            it does not support (e.g. vlen/compound dtypes).

    Returns:
        The manifest dict that was written — useful for inspection
        or further programmatic use.

    Raises:
        ImportError: When the chosen backend's dependency is missing
            (h5py for `"native"`, kerchunk for `"kerchunk"`).
        ValueError: When `backend` is not `"native"` or `"kerchunk"`.

    Examples:
        - Emit a manifest for one NetCDF file (requires the
          `[lazy]` extra):
            ```python
            >>> from pathlib import Path  # doctest: +SKIP
            >>> from pyramids.netcdf._kerchunk_facade import to_kerchunk  # doctest: +SKIP
            >>> manifest = to_kerchunk(
            ...     "noah_20240101.nc", "noah_20240101.kerchunk.json",
            ... )  # doctest: +SKIP
            >>> "refs" in manifest or "version" in manifest  # doctest: +SKIP
            True

            ```
    """
    if backend not in ("native", "kerchunk"):
        raise ValueError(
            f"backend must be 'native' or 'kerchunk'; got {backend!r}"
        )
    src_str = str(src_path)

    def _legacy() -> dict[str, Any]:
        return _kerchunk_translate_single(
            src_str, inline_threshold=inline_threshold, vlen_encode=vlen_encode
        )

    if backend == "kerchunk":
        refs = _legacy()
    else:
        refs = _native_or_fallback(
            lambda: build_single_manifest(
                src_str, inline_threshold=inline_threshold, vlen_encode=vlen_encode
            ),
            _legacy,
            local_inputs_exist=os.path.exists(src_str),
            subject=f"native kerchunk builder could not handle {src_str!r}",
        )
    Path(output_path).write_text(json.dumps(refs))
    return refs


def _kerchunk_combine(
    src_paths: Sequence[str | Path],
    *,
    concat_dims: Sequence[str],
    identical_dims: Sequence[str],
    inline_threshold: int,
) -> dict[str, Any]:
    """Combine via kerchunk's translator + ``MultiZarrToZarr`` (legacy path)."""
    single_hdf5_to_zarr = _require_kerchunk_single()
    multi_zarr_to_zarr = _require_kerchunk_combine()

    per_file = []
    with _scalar_fill_value_shim():
        for path in src_paths:
            src_str = str(path)
            refs = single_hdf5_to_zarr(
                src_str,
                src_str,
                inline_threshold=inline_threshold,
                error="warn",
            ).translate()
            per_file.append(refs)

    return multi_zarr_to_zarr(
        per_file,
        concat_dims=list(concat_dims),
        identical_dims=list(identical_dims),
    ).translate()


def combine_kerchunk(
    src_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    concat_dims: Sequence[str] = ("time",),
    identical_dims: Sequence[str] = ("lat", "lon"),
    inline_threshold: int = 500,
    backend: str = "native",
) -> dict[str, Any]:
    """Emit a combined kerchunk manifest spanning many source files.

    The default `"native"` backend scans each file with :func:`build_single_manifest`
    (h5py, no live zarr group) and concatenates the per-file manifests along the
    single `concat_dims` entry — every variable carrying that dimension is stacked,
    the rest are carried from the first file. This avoids the zarr-v3 `sync()`
    deadlock (#530). `"kerchunk"` forces the legacy
    `SingleHdf5ToZarr` + `MultiZarrToZarr` path; the native backend also falls back
    to it for inputs it cannot handle (e.g. multiple concat dims, unsupported HDF5
    features).

    Args:
        src_paths: Sequence of source NetCDF / HDF5 paths or URLs, in
            concatenation order.
        output_path: Path where the combined JSON manifest is written.
        concat_dims: Dimension name(s) along which to concatenate. The native
            backend supports exactly one; more than one falls back to kerchunk.
            Default `("time",)`.
        identical_dims: Dimension name(s) expected to be identical across files
            (e.g. shared `lat`/`lon`). Used by the kerchunk backend; the native
            backend treats every non-concat variable as identical implicitly.
            Default `("lat", "lon")`.
        inline_threshold: Same semantics as :func:`to_kerchunk`.
        backend: `"native"` (default) or `"kerchunk"`.

    Returns:
        The combined manifest dict that was written.

    Raises:
        ImportError: When the chosen backend's dependency is missing.
        ValueError: When `backend` is not `"native"` or `"kerchunk"`.

    Examples:
        - Combine a year's worth of daily NetCDFs into one manifest:
            ```python
            >>> from pathlib import Path  # doctest: +SKIP
            >>> from pyramids.netcdf._kerchunk_facade import combine_kerchunk  # doctest: +SKIP
            >>> srcs = sorted(Path("/data/noah").glob("noah_*.nc"))  # doctest: +SKIP
            >>> manifest = combine_kerchunk(
            ...     srcs, "noah_combined.json",
            ...     concat_dims=("time",),
            ... )  # doctest: +SKIP
            >>> "refs" in manifest or "version" in manifest  # doctest: +SKIP
            True

            ```
    """
    if backend not in ("native", "kerchunk"):
        raise ValueError(
            f"backend must be 'native' or 'kerchunk'; got {backend!r}"
        )
    def _legacy() -> dict[str, Any]:
        return _kerchunk_combine(
            src_paths,
            concat_dims=concat_dims,
            identical_dims=identical_dims,
            inline_threshold=inline_threshold,
        )

    def _native() -> dict[str, Any]:
        if len(concat_dims) != 1:
            raise ValueError(
                "native combine supports exactly one concat dimension; got "
                f"{tuple(concat_dims)}"
            )
        per_file = [
            build_single_manifest(str(path), inline_threshold=inline_threshold)
            for path in src_paths
        ]
        return combine_manifests(per_file, concat_dim=concat_dims[0])

    if backend == "kerchunk":
        combined = _legacy()
    else:
        combined = _native_or_fallback(
            _native,
            _legacy,
            local_inputs_exist=all(os.path.exists(str(path)) for path in src_paths),
            subject="native kerchunk combine could not handle these inputs",
        )
    Path(output_path).write_text(json.dumps(combined))
    return combined


__all__ = ["to_kerchunk", "combine_kerchunk"]
