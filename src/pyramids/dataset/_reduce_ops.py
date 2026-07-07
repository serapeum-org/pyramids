"""Shared dispatcher for dask-array nan-aware reductions.

Used by :meth:`DatasetCollection._reduce` (time-axis reductions)
and :func:`_grouped_reduce` (per-label grouped reductions) so both
call sites agree on the supported ops and the nan-aware naming
convention.
"""

from __future__ import annotations

from typing import Any, Callable

_SUPPORTED_OPS = ("mean", "sum", "min", "max", "std", "var")
_NAN_TABLE = {op: f"nan{op}" for op in _SUPPORTED_OPS}


def resolve_dask_op(op_name: str, *, skipna: bool) -> Callable[..., Any]:
    """Return the :mod:`dask.array` callable for a reduction.

    Centralises the `nan{op} if skipna else op` name mangling and the
    `getattr(dask.array, name)` resolution so every call site agrees on
    the supported ops and the nan-aware naming convention.

    Args:
        op_name: One of `mean`, `sum`, `min`, `max`, `std`, `var`.
        skipna: When True, return the nan-aware variant
            (e.g. `nanmean`); otherwise the plain op.

    Returns:
        The resolved :mod:`dask.array` callable (e.g. `dask.array.nanmean`).

    Raises:
        ValueError: If `op_name` is outside the supported set.
        ImportError: If the optional `dask` dependency is not
            installed.
    """
    if op_name not in _SUPPORTED_OPS:
        raise ValueError(
            f"Unsupported reduction {op_name!r}; supported: {_SUPPORTED_OPS}"
        )
    try:
        import dask.array as da
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DatasetCollection reductions require the optional "
            "'dask' dependency. Install with one of:\n"
            "  - PyPI:        pip install 'pyramids-gis[lazy]'\n"
            "  - conda-forge: conda install -c conda-forge pyramids-lazy"
        ) from exc
    name = _NAN_TABLE[op_name] if skipna else op_name
    return getattr(da, name)
