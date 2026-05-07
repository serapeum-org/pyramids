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
`[netcdf-lazy]` optional extra. Helpers raise a clear
:class:`ImportError` when kerchunk is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

_KERCHUNK_IMPORT_ERROR = (
    "kerchunk is required for NetCDF → Zarr reference manifests. "
    "Install it with: pip install 'pyramids-gis[netcdf-lazy]'"
)


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


def to_kerchunk(
    src_path: str | Path,
    output_path: str | Path,
    *,
    inline_threshold: int = 500,
    vlen_encode: str = "embed",
) -> dict[str, Any]:
    """Emit a single-file kerchunk reference manifest.

    Args:
        src_path: Path or URL of the source NetCDF / HDF5 file.
        output_path: Path where the JSON manifest is written.
        inline_threshold: Chunks smaller than this many bytes are
            embedded directly in the manifest rather than referenced
            by offset. Forwarded to
            :class:`kerchunk.hdf.SingleHdf5ToZarr`.
        vlen_encode: One of `"embed" | "null" | "leave" | "encode"`.
            Controls how VLEN (variable-length) strings are handled.
            Default `"embed"` inlines string values; other modes
            trade compatibility vs fidelity — see the kerchunk docs.

    Returns:
        The manifest dict that was written — useful for inspection
        or further programmatic use.

    Raises:
        ImportError: When kerchunk is not installed.

    Examples:
        - Emit a manifest for one NetCDF file (requires the
          `[netcdf-lazy]` extra):
            ```python
            >>> from pathlib import Path  # doctest: +SKIP
            >>> from pyramids.netcdf._kerchunk import to_kerchunk  # doctest: +SKIP
            >>> manifest = to_kerchunk(
            ...     "noah_20240101.nc", "noah_20240101.kerchunk.json",
            ... )  # doctest: +SKIP
            >>> "refs" in manifest or "version" in manifest  # doctest: +SKIP
            True

            ```
    """
    SingleHdf5ToZarr = _require_kerchunk_single()
    src_str = str(src_path)
    refs = SingleHdf5ToZarr(
        src_str,
        src_str,
        inline_threshold=inline_threshold,
        error="warn",
        vlen_encode=vlen_encode,
    ).translate()
    Path(output_path).write_text(json.dumps(refs))
    return refs


def combine_kerchunk(
    src_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    concat_dims: Sequence[str] = ("time",),
    identical_dims: Sequence[str] = ("lat", "lon"),
    inline_threshold: int = 500,
) -> dict[str, Any]:
    """Emit a combined kerchunk manifest spanning many source files.

    Each source is first scanned via
    :class:`kerchunk.hdf.SingleHdf5ToZarr` (in memory, no per-file
    sidecar JSON is written), then merged via
    :class:`kerchunk.combine.MultiZarrToZarr`.

    Args:
        src_paths: Sequence of source NetCDF / HDF5 paths or URLs.
        output_path: Path where the combined JSON manifest is written.
        concat_dims: Dimension name(s) along which to concatenate
            per-file coordinates. Default `("time",)`.
        identical_dims: Dimension name(s) expected to be identical
            across every file (for example shared `lat`/`lon`
            coordinates). Default `("lat", "lon")`.
        inline_threshold: Same semantics as :func:`to_kerchunk`.

    Returns:
        The combined manifest dict that was written.

    Raises:
        ImportError: When kerchunk is not installed.

    Examples:
        - Combine a year's worth of daily NetCDFs into one manifest:
            ```python
            >>> from pathlib import Path  # doctest: +SKIP
            >>> from pyramids.netcdf._kerchunk import combine_kerchunk  # doctest: +SKIP
            >>> srcs = sorted(Path("/data/noah").glob("noah_*.nc"))  # doctest: +SKIP
            >>> manifest = combine_kerchunk(
            ...     srcs, "noah_combined.json",
            ...     concat_dims=("time",),
            ... )  # doctest: +SKIP
            >>> "refs" in manifest or "version" in manifest  # doctest: +SKIP
            True

            ```
    """
    SingleHdf5ToZarr = _require_kerchunk_single()
    MultiZarrToZarr = _require_kerchunk_combine()

    per_file = []
    for path in src_paths:
        src_str = str(path)
        refs = SingleHdf5ToZarr(
            src_str,
            src_str,
            inline_threshold=inline_threshold,
            error="warn",
        ).translate()
        per_file.append(refs)

    combined = MultiZarrToZarr(
        per_file,
        concat_dims=list(concat_dims),
        identical_dims=list(identical_dims),
    ).translate()
    Path(output_path).write_text(json.dumps(combined))
    return combined


__all__ = ["to_kerchunk", "combine_kerchunk"]
