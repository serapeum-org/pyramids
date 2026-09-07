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
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import FailedToSaveError
from pyramids.netcdf import LabeledDataset

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
        pandas = pytest.importorskip("pandas")
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
        back = pandas.read_parquet(written)
        assert back["time"].iloc[0].year == 3065, back["time"].iloc[0]

    def test_a_cftime_axis_raises_a_pyramids_error(self, tmp_path: Path):
        """A true `cftime` axis fails the Parquet write with an explanation, not `ArrowInvalid`.

        Test scenario:
            A pre-1582 origin decodes to `DatetimeGregorian`, which Parquet has no type for.
            Left alone, pyarrow raises `ArrowInvalid` from several frames deep with no
            mention of the time axis or of why the objects are there.
        """
        pytest.importorskip("pyarrow")
        store = _time_store(
            tmp_path / "ancient.nc", "days since 0001-01-01", [0.0, 1.0]
        )
        dataset = LabeledDataset.read_file(store)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(FailedToSaveError, match="cftime datetimes"):
                    dataset.to_parquet(tmp_path / "ancient.parquet")
        finally:
            dataset.close()

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
