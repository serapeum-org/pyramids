"""Per-variable summary tables shared by the raster and mesh datasets.

The reduction math lives in :mod:`pyramids.base._reductions` (numpy-only). This module adds the one
pandas-aware assembly step that turns a mapping of already-masked, already-unpacked arrays into a
``DataFrame`` with one row per variable — the single source of truth behind ``NetCDF.summary`` and
``UgridDataset.summary`` so the two never drift apart.

The helper is geometry-agnostic: it never knows whether a sample is a raster cell or a mesh element.
Each caller owns the concerns that *are* geometry-specific — reading the full array, applying CF
unpacking, and blanking no-data to NaN — before handing the values here.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from pyramids.base._reductions import REDUCERS

#: The default statistics, matching both classes' existing stat set (min/max/mean/std, plus count).
DEFAULT_METRICS: tuple[str, ...] = ("count", "min", "max", "mean", "std")

#: Metrics that need a parameter (``quantile`` needs ``q``) or answer a flag, excluded from a summary.
_EXCLUDED: frozenset[str] = frozenset({"quantile"})

#: Every metric a summary accepts: ``count`` plus the float reducers (minus the excluded ones).
ALLOWED_METRICS: tuple[str, ...] = ("count",) + tuple(
    m for m in REDUCERS if m not in _EXCLUDED
)


def _one_metric(values: np.ndarray, metric: str, *, skipna: bool, ddof: int) -> float:
    """One statistic of a flat float64 array; NaN when nothing valid remains."""
    if metric == "count":
        result = float(int(np.isfinite(values).sum()))
    elif values.size == 0 or (skipna and not np.isfinite(values).any()):
        result = float("nan")
    else:
        nan_fn, plain_fn = REDUCERS[metric]
        fn = nan_fn if skipna else plain_fn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = float(
                fn(values, ddof=ddof) if metric in ("std", "var") else fn(values)
            )
    return result


def variable_summary(
    arrays: Mapping[str, np.ndarray],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    skipna: bool = True,
    ddof: int = 0,
) -> pd.DataFrame:
    """Reduce each array over *all* its axes into one summary row.

    Args:
        arrays: ``{variable name: values}``. Each array is already masked (no-data is NaN) and,
            where relevant, CF-unpacked by the caller — this function only reduces.
        metrics: Which statistics to compute, in output-column order. ``count`` reports the number
            of valid (non-NaN) samples; the rest come from the shared ``REDUCERS`` so the numbers
            match ``reduce`` / ``rolling``. Defaults to ``("count", "min", "max", "mean", "std")``.
        skipna: Reduce with the NaN-aware reducer (the default) or the plain one.
        ddof: Delta degrees of freedom for ``std`` / ``var``. Defaults to 0 (numpy's default).

    Returns:
        pandas.DataFrame: One row per variable (index name ``"variable"``), one column per metric.
        ``count`` is ``int64``; every other column is ``float64``. An all-NaN variable yields NaN
        statistics and ``count`` 0. An empty ``arrays`` yields an empty frame with the columns.

    Raises:
        ValueError: A requested metric is unknown or not allowed in a summary (e.g. ``quantile``).
    """
    columns = list(metrics)
    bad = [m for m in columns if m not in ALLOWED_METRICS]
    if bad:
        allowed = ", ".join(ALLOWED_METRICS)
        raise ValueError(
            f"unknown summary metric(s) {bad}; allowed metrics are: {allowed}"
        )
    rows = {
        name: {
            metric: _one_metric(
                np.asarray(arr, dtype="float64").ravel(),
                metric,
                skipna=skipna,
                ddof=ddof,
            )
            for metric in columns
        }
        for name, arr in arrays.items()
    }
    frame = pd.DataFrame(rows.values(), index=list(rows.keys()), columns=columns)
    frame.index.name = "variable"
    if "count" in frame.columns:
        frame["count"] = frame["count"].fillna(0).astype("int64")
    return frame
