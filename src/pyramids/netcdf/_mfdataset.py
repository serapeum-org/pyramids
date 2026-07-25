"""Multi-file NetCDF opener with optional parallel metadata fan-out.

`NetCDF.open_mfdataset(paths, variable,...)` opens many
NetCDF files and stacks a single named variable into a single lazy
:class:`dask.array.Array` of shape
`(n_files, *variable_shape)` — the canonical shape for a set of
daily / hourly gridded files.

Unlike :func:`xarray.open_mfdataset`, this helper is deliberately
narrow: one variable at a time, no "by_coords" inference, no combine
strategies. For the common hydrology / meteorology case (open 365
noah_YYYYMMDD.nc files, stack the `precipitation` variable, reduce
along time) that narrowness is the whole point — no metadata
inference means no failure modes when one file has a different
schema.

`parallel=True` wraps each file's metadata read in
:func:`dask.delayed`, so opening 500 files on a distributed cluster
fans out over workers rather than blocking sequentially.
"""

from __future__ import annotations

import glob
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pyramids.base._utils import lazy_extra_hint

if TYPE_CHECKING:
    from pyramids.netcdf import NetCDF


_LAZY_IMPORT_ERROR = lazy_extra_hint(
    "open_mfdataset requires the optional 'dask' dependency."
)


def _resolve_paths(paths: str | Sequence[str | Path]) -> list[str]:
    """Normalize `paths` to a sorted list of absolute string paths."""
    if isinstance(paths, (str, Path)):
        resolved = [str(p) for p in sorted(glob.glob(str(paths)))]
        if not resolved:
            # Fall back to treating the input as a single explicit path —
            # lets callers pass one filename without it being glob-filtered
            # to nothing.
            resolved = [str(paths)]
    else:
        resolved = [str(p) for p in paths]
    return resolved


def _open_and_extract(
    path: str,
    variable: str,
    preprocess: Callable | None,
    chunks: Any,
) -> Any:
    """Open one NetCDF + extract one variable as a dask array.

    Called synchronously by :func:`open_mfdataset` when
    `parallel=False` and wrapped in :func:`dask.delayed` when
    `parallel=True`.
    """
    from pyramids.netcdf import NetCDF

    nc = NetCDF.read_file(path)
    variable_subset = nc.get_variable(variable)
    if preprocess is not None:
        variable_subset = preprocess(variable_subset)
    return variable_subset.read_array(chunks=chunks)


def open_mfdataset(
    paths: str | Sequence[str | Path],
    variable: str,
    *,
    chunks: dict | str | None = None,
    parallel: bool = False,
    preprocess: Callable[[NetCDF], NetCDF] | None = None,
) -> Any:
    """Open many NetCDFs; stack `variable` into one lazy dask array.

    Args:
        paths: Glob string (`"noah_*.nc"`), an explicit file path, or
            a sequence of paths. Glob inputs are expanded and sorted
            alphabetically so the stack order is deterministic.
        variable: Name of the NetCDF variable to extract from each
            file. Must exist in every file with the same spatial
            shape + dtype; mismatches produce a :class:`ValueError`
            from :func:`dask.array.stack` at graph-construction time.
        chunks: Chunk specification forwarded to
            :meth:`NetCDF.read_array` when reading each file. `None`
            (the default) reads each file lazily with `"auto"`
            (native-ish) chunking so the files are not all materialised
            into RAM before stacking; pass an explicit spec to override.
        parallel: Retained for API compatibility. Under the default lazy
            per-file read the per-file arrays are already dask arrays whose
            chunk reads the scheduler parallelises at compute time, so this
            flag has no effect there. It only wraps the per-file open in
            :func:`dask.delayed` when the reads are eager — which the current
            lazy default never produces.
        preprocess: Optional callable applied to each
            :class:`NetCDF` subset before its array is extracted —
            for example to unpack scale/offset, crop, or drop
            auxiliary variables.

    Returns:
        dask.array.Array: A stack of shape `(n_files, *var_shape)`.
        Element `[i,...]` corresponds to `paths[i]` (after glob
        expansion + sort).

    Raises:
        ImportError: When dask is not installed.
        FileNotFoundError: When `paths` is an explicit path that
            does not exist.

    Examples:
        - Stack a single file into a 1-element dask array:
            ```python
            >>> from pyramids.netcdf._mfdataset import open_mfdataset
            >>> path = "tests/data/netcdf/cf__4v__1d3-3d1__proj__y-desc.nc"
            >>> stack = open_mfdataset([path], "values")
            >>> stack.shape[0]
            1

            ```
    """
    try:
        import dask.array as da
    except ImportError as exc:
        raise ImportError(_LAZY_IMPORT_ERROR) from exc

    resolved = _resolve_paths(paths)
    # Default to a lazy per-file read (native-ish `"auto"` chunking) so the stack does not
    # materialise every file into RAM up front -- opening hundreds of files used to load them all
    # eagerly before stacking (ARC-48). An explicit `chunks` is honoured as given.
    effective_chunks = "auto" if chunks is None else chunks

    # Each per-file read already returns a lazy dask array (chunks are never eager here), so the
    # stack is built directly and dask parallelises the per-chunk reads at compute time. The
    # `parallel` flag -- which used to wrap eager reads in `dask.delayed` -- is inert now that the
    # default read is lazy (wrapping a dask array in `from_delayed` would nest a synchronous inner
    # compute per file); it is retained only for backward-compatible call signatures (ARC-48).
    arrays = [_open_and_extract(p, variable, preprocess, effective_chunks) for p in resolved]
    arrays = [
        a if hasattr(a, "dask") else da.from_array(np.asarray(a), chunks="auto")
        for a in arrays
    ]
    return da.stack(arrays, axis=0)
