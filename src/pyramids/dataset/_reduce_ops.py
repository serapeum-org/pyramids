"""Shared dispatcher for dask-array nan-aware reductions.

Used by :meth:`DatasetCollection._reduce` (time-axis reductions)
and the two `_GroupedCollection` reduction paths
(:func:`_flox_groupby_reduce`, :func:`_fallback_groupby_reduce`)
so all three call sites agree on the supported ops and the
nan-aware naming convention.
"""

from __future__ import annotations

from typing import Any, Callable

_SUPPORTED_OPS = ("mean", "sum", "min", "max", "std", "var")
_NAN_TABLE = {op: f"nan{op}" for op in _SUPPORTED_OPS}


def resolve_dask_op(
    op_name: str, *, skipna: bool
) -> tuple[Callable[..., Any], str]:
    """Return `(callable, name-string)` for a dask-array reduction.

    Centralises both the `nan{op} if skipna else op` name mangling
    and the `getattr(dask.array, name)` resolution. Callers that
    only need the string form (e.g.
    :func:`flox.groupby_reduce`, which takes `func=<str>`) use the
    second tuple element; callers that need the callable directly
    (e.g. `func(arr, axis=0)`) use the first.

    Args:
        op_name: One of `mean`, `sum`, `min`, `max`, `std`, `var`.
        skipna: When True, return the nan-aware variant
            (e.g. `nanmean`); otherwise the plain op.

    Returns:
        `(func, name)` where `func` is the resolved
        :mod:`dask.array` callable and `name` is its attribute
        name on `dask.array`.

    Raises:
        ValueError: If `op_name` is outside the supported set.
        ImportError: If the optional `dask` dependency is not
            installed.
    """
    if op_name not in _SUPPORTED_OPS:
        raise ValueError(
            f"Unsupported reduction {op_name!r}; supported: "
            f"{_SUPPORTED_OPS}"
        )
    try:
        import dask.array as da
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DatasetCollection reductions require the optional "
            "'dask' dependency. Install with: "
            "pip install 'pyramids-gis[lazy]'"
        ) from exc
    name = _NAN_TABLE[op_name] if skipna else op_name
    return getattr(da, name), name
