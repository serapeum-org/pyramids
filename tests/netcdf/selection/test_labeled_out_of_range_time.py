"""Exporting a store whose time axis leaves `datetime64[ns]`'s range (#1087).

`decode_cf_time` no longer wraps such a date into a plausible-looking wrong one; it keeps
whatever `cftime` decoded. What that means downstream depends on which object `cftime`
chose, and these pin both halves through the public `LabeledDataset` exporters:

- a date Python's `datetime` can represent comes back as `cftime.real_datetime`, which is a
  `datetime` subclass, so `pandas` gives `datetime64[us]` and Parquet stores the *correct*
  far-future date;
- a date it cannot -- in practice a pre-1582 origin on a mixed calendar -- comes back as a
  true `cftime` datetime, which Parquet has no type for, so `to_parquet` raises a pyramids
  error naming the columns instead of a bare `ArrowInvalid` from inside pyarrow.

The store is built with GDAL directly rather than through xarray, so these stay core tests
rather than being gated on the interop extra.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal

from pyramids.base._errors import FailedToSaveError
from pyramids.netcdf import LabeledDataset
from pyramids.netcdf.labeled import _cftime_columns

pytestmark = pytest.mark.core


def _time_store(path: Path, unit: str, offsets: list[float]) -> Path:
    """Write a one-dimensional NetCDF store whose only axis is a CF time coordinate.

    Args:
        path: Output ``.nc`` path.
        unit: The CF ``"<period> since <origin>"`` string to stamp on the axis.
        offsets: The numeric offsets to store.

    Returns:
        Path: ``path``, for chaining.
    """
    dataset = gdal.GetDriverByName("netCDF").CreateMultiDimensional(
        str(path), [], ["FORMAT=NC4"]
    )
    root = dataset.GetRootGroup()
    dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    dim = root.CreateDimension("time", "", "", len(offsets))
    axis = root.CreateMDArray("time", [dim], dtype)
    axis.Write(np.asarray(offsets, dtype="float64"))
    axis.SetUnit(unit)
    for key, value in (
        ("standard_name", "time"),
        ("axis", "T"),
        ("calendar", "standard"),
    ):
        attribute = axis.CreateAttribute(key, [], gdal.ExtendedDataType.CreateString())
        attribute.Write(value)
    return path


class TestOutOfRangeTimeExport:
    """The public exporters, on a time axis outside `datetime64[ns]`'s range."""

    def test_a_far_future_axis_exports_the_real_date(self, tmp_path: Path):
        """A year-3065 axis reaches Parquet as the date it actually is.

        Test scenario:
            This is the case the issue reported, end to end. `cftime` returns
            `real_datetime` here, which `pandas` coerces to `datetime64[us]` -- wide enough
            for year 3065 -- so the export both succeeds and is correct. Before the fix it
            succeeded with 1896.
        """
        pytest.importorskip("pyarrow")
        store = _time_store(
            tmp_path / "future.nc", "days since 1970-01-01", [400_000.0, 400_001.0]
        )
        dataset = LabeledDataset.read_file(store)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                written = dataset.to_parquet(tmp_path / "future.parquet")
        finally:
            dataset.close()
        assert written.exists(), "the far-future axis should still export"
        back = pd.read_parquet(written)
        assert back["time"].iloc[0].year == 3065, back["time"].iloc[0]

    def test_a_cftime_axis_raises_a_pyramids_error(self, tmp_path: Path):
        """A true `cftime` axis fails the Parquet write with an explanation, not `ArrowInvalid`.

        Test scenario:
            A pre-1582 origin decodes to `DatetimeGregorian`, which Parquet has no type for.
            Left alone, pyarrow raises `ArrowInvalid` from several frames deep with no
            mention of the time axis or of why the objects are there. It is chained rather
            than dropped, since it is the only thing that says which value pyarrow choked on.
        """
        pytest.importorskip("pyarrow")
        store = _time_store(
            tmp_path / "ancient.nc", "days since 0001-01-01", [0.0, 1.0]
        )
        dataset = LabeledDataset.read_file(store)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(
                    FailedToSaveError, match="cftime datetimes"
                ) as raised:
                    dataset.to_parquet(tmp_path / "ancient.parquet")
        finally:
            dataset.close()
        assert raised.value.__cause__ is not None, "the pyarrow error should be chained"

    def test_csv_still_writes_a_cftime_axis(self, tmp_path: Path):
        """CSV has no such limit, so it remains the way out for these stores.

        Test scenario:
            The Parquet error tells the caller to use CSV instead; that advice has to work.
        """
        store = _time_store(
            tmp_path / "ancient.nc", "days since 0001-01-01", [0.0, 1.0]
        )
        dataset = LabeledDataset.read_file(store)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                written = dataset.to_csv(tmp_path / "ancient.csv")
        finally:
            dataset.close()
        assert written.exists(), "CSV should still write a cftime axis"
        assert "0001-01-01" in written.read_text(encoding="utf-8")

    def test_an_in_range_axis_is_untouched(self, tmp_path: Path):
        """An ordinary axis keeps `datetime64[ns]` and exports as it always did."""
        pytest.importorskip("pyarrow")
        store = _time_store(tmp_path / "now.nc", "days since 1970-01-01", [0.0, 1.0])
        dataset = LabeledDataset.read_file(store)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                frame = dataset.to_dataframe()
                written = dataset.to_parquet(tmp_path / "now.parquet")
        finally:
            dataset.close()
        assert frame["time"].dtype == np.dtype("datetime64[ns]"), frame["time"].dtype
        assert written.exists()

    def test_an_unrelated_parquet_failure_is_not_blamed_on_the_time_axis(
        self, tmp_path: Path
    ):
        """A write that fails for another reason raises that error, not the #1087 one.

        Test scenario:
            The write is wrapped in a bare `except Exception`, so every failure reaches the
            cftime check. On an ordinary `datetime64[ns]` axis there is nothing to blame, and
            the original error -- here a missing output directory -- has to come back out
            unchanged. Re-labelling it would send the caller hunting a range problem that
            does not exist.
        """
        pytest.importorskip("pyarrow")
        store = _time_store(tmp_path / "now.nc", "days since 1970-01-01", [0.0, 1.0])
        dataset = LabeledDataset.read_file(store)
        try:
            with pytest.raises(OSError) as raised:
                dataset.to_parquet(tmp_path / "absent" / "now.parquet")
        finally:
            dataset.close()
        assert not isinstance(raised.value, FailedToSaveError), (
            f"an unrelated failure was re-labelled: {raised.value}"
        )
        assert "cftime" not in str(raised.value), raised.value

    def test_the_warning_names_the_axis_it_was_read_from(self, tmp_path: Path):
        """Reading the store warns with the coordinate's own name, not just its units.

        Test scenario:
            `decode_cf_time` can name the axis only if it is told, and `LabeledDataset` is
            the only caller that knows the name. A store with several time axes would
            otherwise warn identically for each, leaving the reader nothing to tell them
            apart.
        """
        store = _time_store(
            tmp_path / "future.nc", "days since 1970-01-01", [400_000.0]
        )
        dataset = LabeledDataset.read_file(store)
        try:
            with pytest.warns(UserWarning, match="'time'") as caught:
                dataset.to_dataframe()
        finally:
            dataset.close()
        assert any("outside the" in str(record.message) for record in caught), [
            str(record.message) for record in caught
        ]


class TestCftimeColumnDetection:
    """`_cftime_columns` names the columns Parquet has no type for, and only those."""

    def test_only_a_true_cftime_column_is_named(self):
        """Numeric, string and `real_datetime` columns are all storable, so none is blamed.

        Test scenario:
            The helper decides which columns a failed Parquet write should blame, so a false
            positive would misreport an unrelated failure as a range problem. Only
            `DatetimeGregorian` is unstorable here: a `real_datetime` is a `datetime`
            subclass Parquet takes (that is what makes the year-3065 export work), and a
            string column is an object column that holds no dates at all.
        """
        frame = pd.DataFrame(
            {
                "value": np.array([1.0, 2.0]),
                "label": np.array(["a", "b"], dtype=object),
                "future": np.array(
                    [
                        cftime.real_datetime(3065, 1, 1),
                        cftime.real_datetime(3065, 1, 2),
                    ],
                    dtype=object,
                ),
                "ancient": np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 1, 2),
                    ],
                    dtype=object,
                ),
            }
        )
        assert _cftime_columns(frame) == ["ancient"], _cftime_columns(frame)

    def test_a_later_cftime_value_still_names_the_column(self):
        """Any `cftime` element names the column, not only a leading one.

        Test scenario:
            `cftime` picks one class per array from the units origin, so a column decoded in
            one go is uniform and a first-element check would do. A frame assembled from more
            than one decode need not be, and inspecting only the first value would then name
            no column -- leaving pyarrow's own opaque error to surface instead.
        """
        frame = pd.DataFrame(
            {"time": [datetime(2000, 1, 1), cftime.DatetimeGregorian(1000, 1, 1)]}
        )
        assert _cftime_columns(frame) == ["time"], _cftime_columns(frame)

    def test_a_duplicate_column_label_does_not_raise(self):
        """A repeated column label must not turn the write failure into an `AttributeError`.

        Test scenario:
            `frame[name]` returns a DataFrame when the label is duplicated, and `.dtype` on
            that raises -- inside the `except` block, so it would replace the failure the
            caller needs to see. The scan is positional for that reason.
        """
        frame = pd.DataFrame(
            [[datetime(2000, 1, 1), cftime.DatetimeGregorian(1000, 1, 1)]],
            columns=["time", "time"],
        )
        assert _cftime_columns(frame) == ["time"], _cftime_columns(frame)

    def test_an_empty_object_column_names_nothing(self):
        """A zero-row object column has nothing to find, so it is not named.

        Test scenario:
            An empty selection is an ordinary frame, not a contrived one, and this runs
            while another error is already being handled -- so anything raised here would
            replace the failure the caller actually needs to see.
        """
        frame = pd.DataFrame({"time": pd.Series([], dtype=object)})
        assert _cftime_columns(frame) == [], _cftime_columns(frame)
