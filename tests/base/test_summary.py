"""Unit tests for the shared ``variable_summary`` assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyramids.base._summary import DEFAULT_METRICS, variable_summary


def test_default_metrics_and_values():
    df = variable_summary({"a": np.array([1.0, 2.0, 3.0, 4.0])})
    assert list(df.columns) == list(DEFAULT_METRICS)
    assert df.index.name == "variable"
    row = df.loc["a"]
    assert row["count"] == 4
    assert row["min"] == 1.0
    assert row["max"] == 4.0
    assert row["mean"] == pytest.approx(2.5)
    assert row["std"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]))


def test_count_is_int64_and_excludes_nan():
    df = variable_summary({"a": np.array([1.0, np.nan, 3.0])})
    assert df["count"].dtype == np.int64
    assert df.loc["a", "count"] == 2
    assert df.loc["a", "mean"] == pytest.approx(2.0)


def test_inf_counted_like_the_reducers():
    # `count` counts non-NaN samples (inf included), consistent with the NaN-aware reducers,
    # which fold inf in: a [2, inf] variable has count 2 and an infinite max/mean.
    df = variable_summary({"a": np.array([2.0, np.inf])})
    assert df.loc["a", "count"] == 2
    assert np.isinf(df.loc["a", "max"])
    assert np.isinf(df.loc["a", "mean"])


def test_all_nan_variable_is_nan_with_zero_count():
    df = variable_summary({"a": np.array([np.nan, np.nan])})
    assert df.loc["a", "count"] == 0
    assert np.isnan(df.loc["a", "mean"])
    assert np.isnan(df.loc["a", "min"])


def test_multiple_variables_one_row_each_over_all_axes():
    df = variable_summary(
        {"a": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([10.0])}
    )
    assert list(df.index) == ["a", "b"]
    assert df.loc["a", "count"] == 4
    assert df.loc["a", "max"] == 4.0
    assert df.loc["b", "mean"] == 10.0


def test_empty_mapping_yields_empty_framed_columns():
    df = variable_summary({})
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list(DEFAULT_METRICS)
    assert df.index.name == "variable"
    assert len(df) == 0
    assert df["count"].dtype == np.int64
    assert df["mean"].dtype == np.float64


def test_metric_selection_order_preserved():
    df = variable_summary(
        {"a": np.array([1.0, 2.0])}, metrics=("mean", "sum", "median")
    )
    assert list(df.columns) == ["mean", "sum", "median"]
    assert df.loc["a", "sum"] == 3.0
    assert df.loc["a", "median"] == 1.5


def test_unknown_or_excluded_metric_raises():
    with pytest.raises(ValueError, match="unknown summary metric"):
        variable_summary({"a": np.array([1.0])}, metrics=("mean", "bogus"))
    with pytest.raises(ValueError, match="unknown summary metric"):
        # quantile needs a `q` and is deliberately excluded from a summary
        variable_summary({"a": np.array([1.0])}, metrics=("quantile",))


def test_ddof_controls_std():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    pop = variable_summary({"a": values}, metrics=("std",)).loc["a", "std"]
    sample = variable_summary({"a": values}, metrics=("std",), ddof=1).loc["a", "std"]
    assert pop == pytest.approx(np.std(values, ddof=0))
    assert sample == pytest.approx(np.std(values, ddof=1))


def test_skipna_false_propagates_nan():
    df = variable_summary({"a": np.array([1.0, np.nan, 3.0])}, skipna=False)
    assert np.isnan(df.loc["a", "mean"])
    # count is always the finite-sample count, independent of skipna
    assert df.loc["a", "count"] == 2
