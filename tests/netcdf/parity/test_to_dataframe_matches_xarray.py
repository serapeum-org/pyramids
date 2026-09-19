"""`to_dataframe` returns the frame `xarray.Dataset.to_dataframe` returns.

The acceptance test the plan asks for is `assert_frame_equal(nc.to_dataframe(),
xr_ds.to_dataframe())` — index names, index order, column names and values all at once.

**One deliberate difference from the plan, not from xarray.** The plan specified
`dropna=True` as the default, calling it "xarray's default". It is not: `to_dataframe` takes
no `dropna` at all and keeps a row for every cell, which is what `TestTheDefaultKeepsEveryRow`
pins. Defaulting to `True` here would make the `assert_frame_equal` above fail, so the
default is `False` and `dropna=True` is the opt-in convenience.
"""

from __future__ import annotations

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.interop

xr = pytest.importorskip("xarray")

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
STAMPS = [0.0, 6.0]
CELLS = np.arange(8.0).reshape(2, 2, 2)


def _container(
    values: np.ndarray = CELLS, no_data_value: float | None = None
) -> NetCDF:
    """A `(time, y, x)` container holding one variable.

    Args:
        values: The cells.
        no_data_value: The sentinel to declare.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        values,
        geo_ref=GEO,
        variable_name="t",
        no_data_value=no_data_value,
        dims=ExtraDimensions(name="time", values=STAMPS),
    )


def _exported(values: np.ndarray = CELLS) -> xr.Dataset:
    """The same cube as xarray holds it, on the same coordinates.

    Args:
        values: The cells.

    Returns:
        xr.Dataset: The cube.
    """
    return xr.Dataset(
        {"t": (("time", "y", "x"), values)},
        coords={"time": STAMPS, "y": [1.5, 0.5], "x": [0.5, 1.5]},
    )


class TestTheFrameMatchesXarray:
    """The whole frame, compared with `assert_frame_equal`."""

    def test_the_frames_are_equal(self):
        """Index names, index order, columns and values all agree.

        Test scenario:
            This is the plan's "done when" for the task — the strongest statement that the
            two describe the same cube.
        """
        assert_frame_equal(_container().to_dataframe(), _exported().to_dataframe())

    def test_a_cube_with_gaps_matches_too(self):
        """A sentinel becomes NaN, which is how xarray spells the same missing cell."""
        values = CELLS.copy()
        values[0, 0, 0] = -9999.0
        expected = CELLS.copy()
        expected[0, 0, 0] = np.nan
        assert_frame_equal(
            _container(values, no_data_value=-9999.0).to_dataframe(),
            _exported(expected).to_dataframe(),
        )

    def test_the_index_names_and_order(self):
        """The dimensions are named outermost first, as the array is laid out."""
        frame = _container().to_dataframe()
        assert list(frame.index.names) == ["time", "y", "x"]
        assert frame.index[0] == (0.0, 1.5, 0.5)
        assert frame.index[1] == (0.0, 1.5, 1.5)

    def test_one_column_per_variable(self):
        """A single-variable cube yields one column, named after it."""
        assert list(_container().to_dataframe().columns) == ["t"]


class TestTheDefaultKeepsEveryRow:
    """`dropna=False` is the default, because xarray keeps every cell."""

    def test_xarray_has_no_dropna_and_keeps_the_missing_row(self):
        """The premise behind the default, measured rather than assumed.

        Test scenario:
            The plan called `dropna=True` "xarray's default". `Dataset.to_dataframe` takes
            no such argument and keeps the row, so defaulting to `True` would have broken
            the frame comparison the task exists for.
        """
        values = CELLS.copy()
        values[0, 0, 0] = np.nan
        assert len(_exported(values).to_dataframe()) == 8

    def test_the_default_keeps_the_missing_row(self):
        """One row per cell, whether or not the cell holds data."""
        values = CELLS.copy()
        values[0, 0, 0] = -9999.0
        frame = _container(values, no_data_value=-9999.0).to_dataframe()
        assert len(frame) == 8
        assert np.isnan(frame["t"].iloc[0])

    def test_dropna_true_drops_it(self):
        """The opt-in convenience drops the rows that are missing in every column."""
        values = CELLS.copy()
        values[0, 0, 0] = -9999.0
        frame = _container(values, no_data_value=-9999.0).to_dataframe(dropna=True)
        assert len(frame) == 7

    def test_the_row_count_is_the_product_of_the_sizes(self):
        """Every cell of every step is a row."""
        assert len(_container().to_dataframe()) == 2 * 2 * 2


class TestTheReceiversAndRefusals:
    """Which variables become columns, and what is refused."""

    def test_a_variable_answers_its_own_frame(self):
        """Calling it on a variable gives that variable's column."""
        frame = _container().get_variable("t").to_dataframe()
        assert list(frame.columns) == ["t"]
        assert len(frame) == 8

    def test_both_receivers_agree(self):
        """The container's frame and the variable's hold the same numbers."""
        container = _container()
        assert_frame_equal(
            container.to_dataframe(), container.get_variable("t").to_dataframe()
        )

    def test_choosing_the_variables(self):
        """`variables=` picks the columns, by name or as a sequence."""
        container = _container()
        assert list(container.to_dataframe(variables="t").columns) == ["t"]
        assert list(container.to_dataframe(variables=["t"]).columns) == ["t"]

    def test_an_unknown_variable_is_refused(self):
        """The refusal lists the gridded variables there are."""
        container = _container()
        with pytest.raises(ValueError, match="gridded variables are"):
            container.to_dataframe(variables="nope")

    def test_variables_on_different_band_dimensions_are_refused(self):
        """Cells that do not line up cannot share one index.

        Test scenario:
            A container may hold a variable over `time` beside one without it; their cells
            do not correspond row for row, so one frame cannot describe both.
        """
        container = _container()
        flat = Dataset.from_array(
            np.ones((2, 2)),
            geo_ref=GEO,
            no_data_value=None,
        )
        container.set_variable("static", flat)
        with pytest.raises(ValueError, match="share their band dimensions"):
            container.to_dataframe()
