"""T11's no-data toolkit: `fillna`, `isnull`, `notnull`, `ffill`, `bfill` and `dropna`.

The cell-wise three are `Analysis` members on any raster; the dimension-wise three run along
a band dimension through the same loop `reduce` and `rolling` use.

**Where the expectations come from.** `fillna`, `dropna`, `isnull` and `notnull` are compared
against xarray. `ffill` and `bfill` are compared against **pandas**: xarray delegates its own
`push` to `bottleneck`, which is not a dependency here, so asking xarray for the answer raises
`ModuleNotFoundError`. pandas implements the same semantics xarray mirrors, and is already a
core dependency.

**One deliberate difference from xarray.** `isnull` / `notnull` answer `uint8` `0` / `1`
flags, not booleans: GDAL has no boolean band type, and the flags then read as a condition for
`where`, which is the point of having them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.base.georeference import GeoReference as DatasetGeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
RASTER_GEO = DatasetGeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
NDV = -9999.0
TIMES = [0.0, 6.0, 12.0, 18.0]


def _column(values: list[float], no_data_value: float | None = None) -> NetCDF:
    """A one-cell variable over `time`, holding `values`.

    Args:
        values: One value per step, NaN for a gap.
        no_data_value: The sentinel to declare; `None` leaves the gaps as NaN.

    Returns:
        NetCDF: The variable.
    """
    array = np.array(values, dtype="float64")
    if no_data_value is not None:
        array = np.where(np.isnan(array), no_data_value, array)
    return NetCDF.from_array(
        array.reshape(len(values), 1, 1),
        geo_ref=GEO,
        variable_name="t",
        no_data_value=no_data_value,
        dims=ExtraDimensions(name="time", values=TIMES[: len(values)]),
    ).get_variable("t")


def _read(result: NetCDF) -> list[float]:
    """A one-cell result's values as a flat list, its gaps as NaN.

    Args:
        result: The variable to read.

    Returns:
        list[float]: One value per step.
    """
    values = np.asarray(result.read_array(), dtype="float64").ravel()
    sentinel = result.no_data_value[0]
    if sentinel is not None and not np.isnan(sentinel):
        values = np.where(values == sentinel, np.nan, values)
    return values.tolist()


class TestFillna:
    """`fillna` writes to the gaps, where `fill` writes to the cells that hold data."""

    @staticmethod
    def _raster() -> Dataset:
        """A 2x2 raster with one gap.

        Returns:
            Dataset: The raster.
        """
        values = np.array([[1.0, NDV], [3.0, 4.0]])
        return Dataset.from_array(values, geo_ref=RASTER_GEO, no_data_value=NDV)

    def test_it_fills_the_gaps(self):
        """The gap takes the value; the cells that held data keep theirs."""
        read = np.asarray(self._raster().fillna(0.0).read_array(), dtype="float64")
        assert_allclose(read, np.array([[1.0, 0.0], [3.0, 4.0]]))

    def test_it_is_the_inverse_of_fill(self):
        """`fill` writes to exactly the cells `fillna` leaves alone.

        Test scenario:
            The two names are one letter apart and do opposite things, which is the whole
            reason the plan called the naming a risk.
        """
        filled = np.asarray(self._raster().fill(0.0).read_array(), dtype="float64")
        assert_allclose(filled, np.array([[0.0, NDV], [0.0, 0.0]]))

    def test_the_grid_survives(self):
        """Filling changes values, never georeferencing."""
        source = self._raster()
        result = source.fillna(0.0)
        assert result.geotransform == source.geotransform
        assert result.epsg == source.epsg

    def test_a_raster_without_gaps_is_unchanged(self):
        """Nothing is missing, so nothing is written."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        raster = Dataset.from_array(values, geo_ref=RASTER_GEO, no_data_value=NDV)
        assert_allclose(
            np.asarray(raster.fillna(0.0).read_array(), dtype="float64"), values
        )


class TestNullFlags:
    """`isnull` and `notnull` answer the `uint8` flags a comparison answers."""

    @staticmethod
    def _raster() -> Dataset:
        """A 2x2 raster with one gap.

        Returns:
            Dataset: The raster.
        """
        values = np.array([[1.0, NDV], [3.0, 4.0]])
        return Dataset.from_array(values, geo_ref=RASTER_GEO, no_data_value=NDV)

    def test_isnull_flags_the_gap(self):
        """`1` at the gap, `0` everywhere else."""
        assert_array_equal(
            np.asarray(self._raster().isnull().read_array()),
            np.array([[0, 1], [0, 0]], dtype="uint8"),
        )

    def test_notnull_is_the_complement(self):
        """Every cell is one or the other, and never both."""
        raster = self._raster()
        assert_array_equal(
            np.asarray(raster.isnull().read_array())
            + np.asarray(raster.notnull().read_array()),
            np.ones((2, 2), dtype="uint8"),
        )

    def test_the_flags_are_uint8(self):
        """GDAL has no boolean band type, so the flags come back as a comparison's do."""
        assert np.asarray(self._raster().isnull().read_array()).dtype == np.uint8

    def test_the_flags_declare_no_no_data_value(self):
        """Every cell is either missing or not, so no flag could fail to be judged."""
        assert self._raster().isnull().no_data_value[0] is None

    def test_they_read_as_a_condition_for_where(self):
        """`where(notnull())` keeps exactly the cells that hold data."""
        raster = self._raster()
        kept = np.asarray(raster.where(raster.notnull()).read_array(), dtype="float64")
        assert_allclose(kept, np.array([[1.0, NDV], [3.0, 4.0]]))

    def test_isnull_selects_the_gaps_for_where(self):
        """`where(isnull(), 0.0)` writes into the cells that hold data, keeping the gaps."""
        raster = self._raster()
        read = np.asarray(
            raster.where(raster.isnull(), 0.0).read_array(), dtype="float64"
        )
        assert read[0, 1] == pytest.approx(NDV)
        assert read[0, 0] == pytest.approx(0.0)


class TestFfillAndBfill:
    """Carrying a valid value into the gaps, compared against pandas."""

    COLUMN = [np.nan, 2.0, np.nan, np.nan]

    @pytest.mark.parametrize("limit", [None, 1, 2])
    def test_ffill_matches_pandas(self, limit):
        """Every limit answers what `pandas.Series.ffill` answers.

        Args:
            limit: How many consecutive gaps one value may fill.
        """
        result = _column(self.COLUMN).ffill("time", limit=limit)
        assert_allclose(
            _read(result),
            pd.Series(self.COLUMN).ffill(limit=limit).tolist(),
            equal_nan=True,
        )

    @pytest.mark.parametrize("limit", [None, 1, 2])
    def test_bfill_matches_pandas(self, limit):
        """Every limit answers what `pandas.Series.bfill` answers.

        Args:
            limit: How many consecutive gaps one value may fill.
        """
        result = _column(self.COLUMN).bfill("time", limit=limit)
        assert_allclose(
            _read(result),
            pd.Series(self.COLUMN).bfill(limit=limit).tolist(),
            equal_nan=True,
        )

    def test_a_leading_gap_is_not_filled_forwards(self):
        """`ffill` carries data forward; it does not invent a start."""
        assert np.isnan(_read(_column(self.COLUMN).ffill("time"))[0])

    def test_a_trailing_gap_is_not_filled_backwards(self):
        """`bfill` reads the other way, and stops at the last valid cell."""
        assert np.isnan(_read(_column(self.COLUMN).bfill("time"))[-1])

    def test_bfill_is_ffill_reversed(self):
        """Filling a reversed column forwards gives the reverse of filling it backwards."""
        forward = _read(_column(list(reversed(self.COLUMN))).ffill("time"))
        assert_allclose(
            list(reversed(forward)),
            _read(_column(self.COLUMN).bfill("time")),
            equal_nan=True,
        )

    def test_a_declared_sentinel_is_a_gap_too(self):
        """A band that marks its gaps with a sentinel fills the same cells."""
        result = _column(self.COLUMN, no_data_value=NDV).ffill("time")
        assert_allclose(_read(result), [np.nan, 2.0, 2.0, 2.0], equal_nan=True)
        assert result.no_data_value[0] == pytest.approx(NDV)

    def test_the_dimension_keeps_its_length_and_stamps(self):
        """Only the values change: `ffill` fills gaps, it does not drop steps."""
        result = _column(self.COLUMN).ffill("time")
        assert result.band_count == 4
        assert result._band_dim_values_map["time"] == TIMES

    @pytest.mark.parametrize("member", ["ffill", "bfill"])
    def test_a_bad_limit_is_refused(self, member):
        """A limit has to be a whole number of steps, and at least one.

        Args:
            member: The member called.
        """
        variable = _column(self.COLUMN)
        with pytest.raises(TypeError):
            getattr(variable, member)("time", limit=1.5)
        with pytest.raises(ValueError):
            getattr(variable, member)("time", limit=0)


class TestDropna:
    """`dropna` removes the steps that hold too little data."""

    COLUMN = [1.0, np.nan, 3.0, np.nan]

    def test_how_any_drops_every_step_holding_a_gap(self):
        """One cell per step, so any gap is the whole step."""
        result = _column(self.COLUMN).dropna("time")
        assert_allclose(_read(result), [1.0, 3.0])

    def test_the_coordinates_are_cut_to_what_survived(self):
        """The stamps of the dropped steps go with them."""
        assert _column(self.COLUMN).dropna("time")._band_dim_values_map["time"] == [
            0.0,
            12.0,
        ]

    def test_how_all_keeps_a_step_with_any_valid_cell(self):
        """A step is dropped only when nothing in it is data."""
        values = np.array([[[1.0, np.nan]], [[np.nan, np.nan]], [[3.0, 4.0]]])
        variable = NetCDF.from_array(
            values,
            geo_ref=GEO,
            variable_name="t",
            dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
        ).get_variable("t")
        assert variable.dropna("time", how="all")._band_dim_values_map["time"] == [
            0.0,
            12.0,
        ]

    def test_how_any_on_the_same_values_keeps_only_the_full_step(self):
        """`any` is stricter than `all`, and drops the half-empty step too."""
        values = np.array([[[1.0, np.nan]], [[np.nan, np.nan]], [[3.0, 4.0]]])
        variable = NetCDF.from_array(
            values,
            geo_ref=GEO,
            variable_name="t",
            dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
        ).get_variable("t")
        assert variable.dropna("time", how="any")._band_dim_values_map["time"] == [12.0]

    def test_thresh_counts_the_valid_cells(self):
        """`thresh=2` keeps only the step with two valid cells."""
        values = np.array([[[1.0, np.nan]], [[np.nan, np.nan]], [[3.0, 4.0]]])
        variable = NetCDF.from_array(
            values,
            geo_ref=GEO,
            variable_name="t",
            dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
        ).get_variable("t")
        assert variable.dropna("time", thresh=2)._band_dim_values_map["time"] == [12.0]

    def test_dropping_everything_is_refused(self):
        """A variable with no bands cannot be built, so the refusal says so."""
        variable = _column([np.nan, np.nan])
        with pytest.raises(ValueError, match="no steps"):
            variable.dropna("time")

    def test_an_unknown_how_is_refused(self):
        """`how` is xarray's vocabulary, and nothing else."""
        variable = _column(self.COLUMN)
        with pytest.raises(ValueError, match="how="):
            variable.dropna("time", how="some")

    def test_the_values_keep_their_own_type(self):
        """Nothing is computed, only selected, so an integer band stays an integer band."""
        values = np.array([1, 2, 3, 4], dtype="int16").reshape(4, 1, 1)
        variable = NetCDF.from_array(
            values,
            geo_ref=GEO,
            variable_name="t",
            no_data_value=2,
            dims=ExtraDimensions(name="time", values=TIMES),
        ).get_variable("t")
        result = variable.dropna("time")
        assert np.asarray(result.read_array()).dtype == np.int16
        assert result._band_dim_values_map["time"] == [0.0, 12.0, 18.0]


class TestTheReceivers:
    """Each dimension-wise member works on a container and on a variable."""

    @staticmethod
    def _container() -> NetCDF:
        """A container holding one gridded variable over `time`.

        Returns:
            NetCDF: The container.
        """
        values = np.array([np.nan, 2.0, np.nan, np.nan]).reshape(4, 1, 1)
        return NetCDF.from_array(
            values,
            geo_ref=GEO,
            variable_name="t",
            dims=ExtraDimensions(name="time", values=TIMES),
        )

    @pytest.mark.parametrize("member", ["ffill", "bfill", "dropna"])
    def test_a_container_answers_a_container(self, member):
        """The container receiver runs the member on every variable that has the dimension.

        Args:
            member: The member called.
        """
        result = getattr(self._container(), member)("time")
        assert "t" in result.variable_names

    @pytest.mark.parametrize("member", ["ffill", "bfill", "dropna"])
    def test_both_receivers_agree(self, member):
        """A variable taken from the filled container holds what filling the variable holds.

        Args:
            member: The member called.
        """
        container = self._container()
        from_container = getattr(container, member)("time").get_variable("t")
        from_variable = getattr(container.get_variable("t"), member)("time")
        assert_allclose(_read(from_container), _read(from_variable), equal_nan=True)

    @pytest.mark.parametrize("member", ["ffill", "bfill", "dropna"])
    def test_a_dimension_no_variable_has_is_refused(self, member):
        """A container refuses a dimension none of its gridded variables carries.

        Args:
            member: The member called.
        """
        call = getattr(self._container(), member)
        with pytest.raises(ValueError, match="not a non-spatial dimension"):
            call("level")
