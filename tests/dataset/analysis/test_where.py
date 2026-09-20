"""Tests for `Analysis.where` — keep the cells a condition selects, mask the rest.

Every expectation is xarray's, measured on the same values: `da.where(cond)` masks where the
condition is false, `da.where(cond, other)` writes `other` there instead, and `drop=True`
trims the raster to the bounding box of what survived. A condition cell that is itself no-data
reads as false, which is what xarray's NaN-propagating comparison does.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.base._errors import AlignmentError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, NetCDF
from pyramids.netcdf import GeoReference as NCGeoReference

pytestmark = pytest.mark.core

GEO_REF = GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326)
NDV = -9999.0
VALUES = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, NDV, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
CLEAN = np.arange(1.0, 13.0).reshape(3, 4)
"""`VALUES` with the gap filled in, for the rasters that declare no sentinel at all."""


def _raster(
    array: np.ndarray | None = None, no_data_value: float | None = NDV
) -> Dataset:
    """A 3x4 raster on a unit grid.

    Args:
        array: The values; `VALUES` when omitted.
        no_data_value: The sentinel to declare.

    Returns:
        Dataset: The raster.
    """
    values = VALUES if array is None else array
    return Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=no_data_value)


def _read(raster: Dataset) -> np.ndarray:
    """The raster's values as float64 with its gaps as NaN.

    Args:
        raster: The raster to read.

    Returns:
        np.ndarray: The values, no-data as NaN.
    """
    values = np.asarray(raster.read_array(), dtype="float64")
    sentinel = raster.no_data_value[0]
    if sentinel is not None and not np.isnan(sentinel):
        values = np.where(values == sentinel, np.nan, values)
    return values


class TestWhereMasks:
    """A false cell is masked; a true cell is kept exactly."""

    def test_an_array_condition_masks_the_false_cells(self):
        """`where(values > 5)` keeps 7, 8, 9, 10, 11 and 12 and masks the rest."""
        result = _raster().where(VALUES > 5)
        expected = np.where(VALUES > 5, VALUES, np.nan)
        expected[1, 1] = np.nan
        assert_allclose(_read(result), expected, equal_nan=True)

    def test_a_kept_cell_is_untouched(self):
        """The values that survive are the source's, bit for bit."""
        read = _read(_raster().where(VALUES > 5))
        assert read[2, 3] == pytest.approx(12.0)
        assert read[1, 2] == pytest.approx(7.0)

    def test_other_replaces_the_false_cells(self):
        """`other=-1.0` writes -1.0 where the condition is false, as xarray does."""
        read = np.asarray(
            _raster().where(VALUES > 5, -1.0).read_array(), dtype="float64"
        )
        assert read[0, 0] == pytest.approx(-1.0)
        assert read[2, 0] == pytest.approx(9.0)

    def test_a_gap_stays_a_gap_under_other(self):
        """The source's own no-data cell is false, so `other` claims it too.

        Test scenario:
            xarray's comparison is false at a NaN, so `where(cond, -1)` writes `-1` there;
            the cell does not survive merely because it was already missing.
        """
        read = np.asarray(
            _raster().where(VALUES > 5, -1.0).read_array(), dtype="float64"
        )
        assert read[1, 1] == pytest.approx(-1.0)

    def test_the_grid_and_crs_survive(self):
        """Masking changes values, never georeferencing."""
        source = _raster()
        result = source.where(VALUES > 5)
        assert result.geotransform == source.geotransform
        assert result.epsg == source.epsg
        assert (result.rows, result.columns) == (source.rows, source.columns)

    def test_a_raster_declaring_no_sentinel_masks_to_nan_and_declares_it(self):
        """With nothing declared to mean "missing", the mask writes NaN and says so.

        Test scenario:
            `where` derives `other` from the raster's own no-data value, and a raster that
            declares none has nothing to derive. The masked cells must still read as gaps,
            so NaN is written *and* declared — a result holding NaN while declaring no
            sentinel would say those cells hold data.
        """
        raster = _raster(CLEAN, no_data_value=None)
        assert raster.no_data_value[0] is None
        result = raster.where(CLEAN > 9)
        read = np.asarray(result.read_array(), dtype="float64")
        assert np.isnan(result.no_data_value[0])
        assert np.isnan(read[0, 0])
        assert read[2, 1] == pytest.approx(10.0)

    def test_a_raster_declaring_no_sentinel_keeps_declaring_none_under_other(self):
        """Writing a number into the masked cells leaves the result with no gaps at all.

        Test scenario:
            The companion of the test above: the derived NaN is declared only because the
            masked cells really are missing. Once `other` fills them with a number there is
            nothing missing, so no sentinel is invented.
        """
        result = _raster(CLEAN, no_data_value=None).where(CLEAN > 9, 0.0)
        read = np.asarray(result.read_array(), dtype="float64")
        assert result.no_data_value[0] is None
        assert read[0, 0] == pytest.approx(0.0)
        assert read[2, 1] == pytest.approx(10.0)


class TestConditionForms:
    """A condition is an array, a raster on the same grid, or a callable."""

    def test_a_boolean_array(self):
        """A plain boolean array selects cell by cell."""
        mask = np.zeros((3, 4), dtype=bool)
        mask[0, 0] = True
        read = _read(_raster().where(mask))
        assert read[0, 0] == pytest.approx(1.0)
        assert np.isnan(read[0, 1])

    def test_a_callable_receives_the_values(self):
        """The callable is handed the raster's array and returns the condition."""
        read = _read(_raster().where(lambda values: values > 5))
        assert_allclose(read, _read(_raster().where(VALUES > 5)), equal_nan=True)

    def test_a_raster_condition(self):
        """A `Dataset` of flags on the same grid selects the same cells as its array."""
        flags = _raster((VALUES > 5).astype("uint8"), no_data_value=None)
        assert_allclose(
            _read(_raster().where(flags)),
            _read(_raster().where(VALUES > 5)),
            equal_nan=True,
        )

    def test_a_comparison_result_is_a_condition(self):
        """`raster > 5` is the natural condition, and reads as one."""
        source = _raster()
        assert_allclose(
            _read(source.where(source > 5)),
            _read(_raster().where(VALUES > 5)),
            equal_nan=True,
        )

    def test_a_no_data_condition_cell_reads_as_false(self):
        """A flag raster's own gap selects nothing, rather than selecting everything.

        Test scenario:
            `raster > 5` declares 255 for a cell it could not compare, so the condition has
            gaps of its own. Reading one as true would keep a cell the comparison never
            judged.
        """
        source = _raster()
        condition = source > 5
        assert condition.no_data_value[0] == pytest.approx(255.0)
        assert np.isnan(_read(source.where(condition))[1, 1])

    def test_a_condition_on_another_grid_is_refused(self):
        """A condition raster must share the grid; resampling silently would be worse."""
        other_grid = GeoReference(
            top_left_corner=(100.0, 3.0), cell_size=1.0, epsg=4326
        )
        flags = Dataset.from_array(
            (VALUES > 5).astype("uint8"), geo_ref=other_grid, no_data_value=None
        )
        raster = _raster()
        with pytest.raises(AlignmentError):
            raster.where(flags)

    def test_a_condition_of_the_wrong_shape_is_refused(self):
        """An array condition has to describe this raster's cells."""
        raster = _raster()
        wrong_shape = np.ones((2, 2), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            raster.where(wrong_shape)


class TestACubeShapedCondition:
    """A condition carrying the cube's leading axis works whatever the cube's length."""

    @staticmethod
    def _variable(steps: int) -> tuple:
        """A `(steps, 3, 3)` variable and the cube-shaped array of its own values.

        Args:
            steps: How many steps the `time` dimension has.

        Returns:
            tuple: The variable and its values as `_materialize_variable_array` gives them.
        """
        container = NetCDF.from_array(
            np.arange(9.0 * steps).reshape(steps, 3, 3),
            geo_ref=NCGeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=[6.0 * i for i in range(steps)]),
        )
        variable = container.get_variable("t")
        return variable, np.asarray(container._materialize_variable_array(variable))

    def test_a_one_step_cube_accepts_its_own_layout(self):
        """The refusal depended on the cube's length, which is not a property of the call.

        Test scenario:
            `_operand_arrays` squeezes a single-band variable to `(rows, cols)` while the
            cube layout stays `(1, rows, cols)`, and `broadcast_to` cannot drop a leading
            axis — so the same construction was refused on a one-step cube and accepted
            on a two-step one.
        """
        variable, values = self._variable(1)
        assert values.shape == (1, 3, 3), "precondition: the cube layout has the axis"
        kept = variable.where(values > 4.0)
        assert np.asarray(kept.read_array()).ravel().tolist()[5:] == [
            5.0,
            6.0,
            7.0,
            8.0,
        ]

    def test_a_two_step_cube_is_unaffected(self):
        """The case that already worked still works."""
        variable, values = self._variable(2)
        kept = variable.where(values > 4.0)
        assert np.asarray(kept.read_array()).shape == (2, 3, 3)

    def test_a_genuinely_wrong_shape_is_still_refused(self):
        """Dropping a leading singleton must not swallow a real mismatch."""
        variable, _ = self._variable(1)
        wrong = np.ones((1, 2, 2), dtype=bool)
        with pytest.raises(ValueError, match="does not broadcast"):
            variable.where(wrong)


class TestDrop:
    """`drop=True` trims the raster to the bounding box of what survived."""

    def test_it_trims_the_fully_masked_edges(self):
        """`where(values > 9, drop=True)` leaves the 1x3 block holding 10, 11 and 12.

        Test scenario:
            Measured on xarray: `da.where(da > 9, drop=True)` answers shape `(1, 3)` with
            `y == [0.5]` and `x == [1.5, 2.5, 3.5]`.
        """
        result = _raster().where(VALUES > 9, drop=True)
        assert (result.rows, result.columns) == (1, 3)
        assert_allclose(_read(result), np.array([[10.0, 11.0, 12.0]]))

    def test_the_trimmed_result_is_georeferenced_to_what_it_kept(self):
        """The origin moves to the first surviving column, and the cell size is unchanged."""
        source = _raster()
        result = source.where(VALUES > 9, drop=True)
        assert result.geotransform[0] == pytest.approx(source.geotransform[0] + 1.0)
        assert result.geotransform[3] == pytest.approx(source.geotransform[3] - 2.0)
        assert result.geotransform[1] == pytest.approx(source.geotransform[1])
        assert result.geotransform[5] == pytest.approx(source.geotransform[5])

    def test_drop_false_keeps_the_full_grid(self):
        """The default leaves every row and column in place."""
        assert (
            _raster().where(VALUES > 9).rows,
            _raster().where(VALUES > 9).columns,
        ) == (
            3,
            4,
        )

    def test_an_all_true_condition_trims_nothing(self):
        """Nothing is masked, so nothing is trimmed."""
        result = _raster().where(np.ones((3, 4), dtype=bool), drop=True)
        assert (result.rows, result.columns) == (3, 4)

    def test_an_all_false_condition_is_refused(self):
        """Nothing survives, and a raster of no cells cannot be built."""
        raster = _raster()
        mask = np.zeros((3, 4), dtype=bool)
        with pytest.raises(ValueError, match="no cells"):
            raster.where(mask, drop=True)

    def test_drop_trims_by_the_condition_even_when_other_fills_the_rest(self):
        """`other` does not save a row the condition was false across.

        Test scenario:
            The box was read off the *result*, where a cell `other` wrote a number into
            is data — so a numeric `other` left nothing to trim and the full grid came
            back. xarray trims by the condition whatever `other` is: measured,
            `da.where(da > 9, -1.0, drop=True)` answers shape `(1, 3)` holding
            `[[10.0, 11.0, 12.0]]`, the same block as without `other`.
        """
        result = _raster().where(VALUES > 9, -1.0, drop=True)
        assert (result.rows, result.columns) == (1, 3)
        assert_allclose(_read(result), np.array([[10.0, 11.0, 12.0]]))

    def test_a_pre_existing_gap_is_not_trimmed_away(self):
        """A gap the condition selected stays inside the box.

        Test scenario:
            Reading the result's domain also trimmed the cells that were *already*
            missing, so an all-true condition on a raster whose edges are gaps came back
            1x1. Measured on xarray, the same call keeps the full 3x3 and its NaNs.
        """
        edges = np.full((3, 4), NDV)
        edges[1, 1] = 5.0
        result = _raster(edges).where(np.ones((3, 4), dtype=bool), drop=True)
        assert (result.rows, result.columns) == (3, 4)
        assert np.isnan(_read(result)).sum() == 11


class TestDropTrimsTheBandDimensionToo:
    """xarray drops labels in every dimension, not only the two spatial ones."""

    @staticmethod
    def _variable() -> tuple:
        """A `(time=2, y=2, x=2)` variable and its cells in cube layout.

        Returns:
            tuple: The variable and the `(2, 2, 2)` array of its values.
        """
        container = NetCDF.from_array(
            np.arange(8.0).reshape(2, 2, 2),
            geo_ref=NCGeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
        )
        variable = container.get_variable("t")
        return variable, np.asarray(container._materialize_variable_array(variable))

    def test_a_step_the_condition_is_false_across_is_dropped(self):
        """The empty step goes, and the stamps go with it.

        Test scenario:
            The trim folded the band axis away with `np.any(selected, axis=0)` and cut
            rows and columns only, so a cube kept every step however empty. Measured on
            xarray: `da.where(da > 4, drop=True)` answers shape `(1, 2, 2)` with
            `time == [6.0]` and values `[[nan, 5.0], [6.0, 7.0]]`.
        """
        variable, values = self._variable()
        kept = variable.where(values > 4.0, drop=True)
        assert (kept.band_count, kept.rows, kept.columns) == (1, 2, 2)
        assert kept._band_dim_values_map["time"] == [6.0]
        assert_allclose(
            _read(kept), np.array([[np.nan, 5.0], [6.0, 7.0]]), equal_nan=True
        )

    def test_the_spatial_trim_still_applies_alongside_it(self):
        """Both halves at once, which is the shape xarray answers.

        Test scenario:
            `da.where(da > 6, drop=True)` is `(1, 1, 1)` at `time [6.0]`, `y [0.5]`,
            `x [1.5]`.
        """
        variable, values = self._variable()
        kept = variable.where(values > 6.0, drop=True)
        assert (kept.band_count, kept.rows, kept.columns) == (1, 1, 1)
        assert kept._band_dim_values_map["time"] == [6.0]

    def test_a_condition_true_in_every_step_keeps_them_all(self):
        """Trimming must not shorten a cube the condition did not empty."""
        variable, values = self._variable()
        kept = variable.where(values >= 0.0, drop=True)
        assert kept.band_count == 2
        assert kept._band_dim_values_map["time"] == [0.0, 6.0]

    def test_a_plain_stack_drops_its_empty_band(self):
        """A stack has bands without stamps, and the same rule applies to them."""
        cells = np.arange(8.0).reshape(2, 2, 2)
        stack = Dataset.from_array(cells, geo_ref=GEO_REF, no_data_value=NDV)
        assert stack.where(cells > 4.0, drop=True).band_count == 1

    def test_two_band_dimensions_keep_every_band(self):
        """A product of two dimensions has no rectangular subset to cut to.

        Test scenario:
            The bands are the flattened product of `time` and `level`, and the steps the
            condition survives in are not a rectangle of that product in general — so
            those variables are trimmed spatially only, which the docstring states.
        """
        container = NetCDF.from_array(
            np.arange(16.0).reshape(2, 2, 2, 2),
            geo_ref=NCGeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(
                dims=[("time", [0.0, 6.0]), ("level", [1000.0, 850.0])]
            ),
        )
        variable = container.get_variable("t")
        # The bands are the flattened product, which is the layout a condition for this
        # variable has to be in — the `(2, 2, 2, 2)` store layout is a separate gap.
        cells = np.arange(16.0).reshape(4, 2, 2)
        kept = variable.where(cells > 12.0, drop=True)
        assert kept.band_count == 4
        assert kept._band_dim_values_map["time"] == [0.0, 6.0]
        assert kept._band_dim_values_map["level"] == [1000.0, 850.0]


class TestBandsAndLayout:
    """A multi-band raster is masked band by band."""

    @staticmethod
    def _stack() -> np.ndarray:
        """Two bands of 3x4, the second ten greater than the first.

        Returns:
            np.ndarray: The stack.
        """
        return np.stack([VALUES, VALUES + 10.0])

    def test_every_band_is_masked_by_a_two_dimensional_condition(self):
        """One condition plane broadcasts across the bands."""
        raster = Dataset.from_array(self._stack(), geo_ref=GEO_REF, no_data_value=NDV)
        read = np.asarray(raster.where(VALUES > 9).read_array(), dtype="float64")
        assert read.shape == (2, 3, 4)
        assert read[1, 2, 1] == pytest.approx(20.0)
        assert np.isnan(_read(raster.where(VALUES > 9))[1, 2, 0])
        assert read[0, 0, 0] != pytest.approx(1.0)

    def test_a_condition_per_band(self):
        """A condition shaped like the stack selects per band."""
        raster = Dataset.from_array(self._stack(), geo_ref=GEO_REF, no_data_value=NDV)
        read = np.asarray(
            raster.where(self._stack() > 11).read_array(), dtype="float64"
        )
        assert read[1, 0, 1] == pytest.approx(12.0)
        assert np.isnan(_read(raster.where(self._stack() > 11))[1, 0, 0])
        assert read[0, 0, 0] != pytest.approx(1.0)

    def test_the_band_count_is_unchanged(self):
        """Masking never adds or drops a band."""
        raster = Dataset.from_array(self._stack(), geo_ref=GEO_REF, no_data_value=NDV)
        assert raster.where(VALUES > 9).band_count == 2

    def test_drop_trims_a_stack_to_the_block_that_survived_in_any_band(self):
        """A row or column is kept when *any* band still holds data there.

        Test scenario:
            The trim reads one plane, folded across the bands, so a cell surviving in the
            second band alone keeps its row and column. Every band is cut to the same
            rectangle, since they share one grid.
        """
        raster = Dataset.from_array(self._stack(), geo_ref=GEO_REF, no_data_value=NDV)
        mask = np.zeros((3, 4), dtype=bool)
        mask[1, 2] = True
        result = raster.where(mask, drop=True)
        assert (result.band_count, result.rows, result.columns) == (2, 1, 1)
        read = np.asarray(result.read_array(), dtype="float64")
        assert read.ravel().tolist() == [7.0, 17.0]

    def test_drop_on_a_stack_keeps_the_grid_of_what_it_kept(self):
        """The origin moves to the surviving cell and the cell size is untouched."""
        raster = Dataset.from_array(self._stack(), geo_ref=GEO_REF, no_data_value=NDV)
        mask = np.zeros((3, 4), dtype=bool)
        mask[1, 2] = True
        result = raster.where(mask, drop=True)
        assert result.geotransform[0] == pytest.approx(2.0)
        assert result.geotransform[3] == pytest.approx(2.0)
        assert result.geotransform[1] == pytest.approx(1.0)
        assert result.geotransform[5] == pytest.approx(-1.0)


class TestTheResultKeepsItsIdentity:
    """A masked raster is still the same band of the same scene."""

    @staticmethod
    def _labelled() -> Dataset:
        """A raster carrying a band name and a dataset tag.

        Returns:
            Dataset: The raster.
        """
        raster = _raster()
        raster.band_names = ["reflectance"]
        raster.meta_data = {"source": "sentinel"}
        return raster

    def test_where_keeps_the_band_names(self):
        """The docstring promises the band names travel, as they do through an operator.

        Test scenario:
            `r + 1` answered `['reflectance']` while `r.where(r > 2)` answered `['Band_1']`:
            `_combine` assigns the names and the tags before labelling, and `where` skipped
            both assignments.
        """
        assert self._labelled().where(VALUES > 2).band_names == ["reflectance"]

    def test_where_keeps_the_metadata(self):
        """A result that has forgotten its scene is harder to use than the array it came from."""
        assert self._labelled().where(VALUES > 2).meta_data == {"source": "sentinel"}

    def test_the_operator_path_is_the_reference(self):
        """Whatever an operator carries, `where` carries — that is the promise."""
        labelled = self._labelled()
        assert labelled.where(VALUES > 2).band_names == (labelled + 1).band_names
        assert labelled.where(VALUES > 2).meta_data == (labelled + 1).meta_data

    def test_fillna_keeps_them_too(self):
        """`fillna` writes to the gaps; it does not rename the band."""
        assert self._labelled().fillna(0.0).band_names == ["reflectance"]
        assert self._labelled().fillna(0.0).meta_data == {"source": "sentinel"}

    def test_the_null_flags_keep_them_too(self):
        """A flag band still describes the band it flags."""
        assert self._labelled().isnull().band_names == ["reflectance"]
        assert self._labelled().notnull().meta_data == {"source": "sentinel"}

    @pytest.mark.parametrize(
        "member",
        ["where", "fillna", "isnull", "notnull"],
    )
    def test_a_lazy_read_of_a_rebuilt_variable_is_refused_in_words(self, member: str):
        """Carrying the name must not make the result look like it still reads its store.

        Test scenario:
            `copy()` clears `_source_var_name` precisely so the lazy read — which reopens
            `file::name` — cannot reach for a store that no longer holds these cells.
            Carrying the name for labelling put the result back in that position, and the
            guard stopped firing: the caller got a bare `RuntimeError: No such file or
            directory` from inside GDAL instead of a sentence.

        Args:
            member: The `Analysis` member to check.
        """
        variable = NetCDF.from_array(
            np.arange(8.0).reshape(2, 2, 2),
            geo_ref=NCGeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
        ).get_variable("t")
        arguments = {"where": (np.ones((2, 2, 2), dtype=bool),), "fillna": (0.0,)}
        result = getattr(variable, member)(*arguments.get(member, ()))
        with pytest.raises(ValueError, match="rebuilt in memory"):
            result.read_array(chunks="auto")

    def test_a_rebuilt_variable_still_reads_eagerly(self):
        """The refusal is about the lazy path only; the cells are right there."""
        variable = NetCDF.from_array(
            np.arange(8.0).reshape(2, 2, 2),
            geo_ref=NCGeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
        ).get_variable("t")
        filled = variable.fillna(0.0)
        assert np.asarray(filled.read_array()).ravel().tolist() == list(range(8))

    @pytest.mark.parametrize(
        "member",
        ["where", "fillna", "isnull", "notnull"],
    )
    def test_a_variable_keeps_the_name_it_answers_to(self, member: str):
        """A masked variable is still `t`, and `to_dataframe` must still call it that.

        Test scenario:
            The members along a dimension (`ffill`, `dropna`, ...) carry
            `_source_var_name` through; these four did not, so the result came back as the
            placeholder `variable`. That was cosmetic until `variables=` began to be read
            on a variable receiver, at which point `var.fillna(0.0).to_dataframe(
            variables="t")` started refusing the variable's own name.

        Args:
            member: The `Analysis` member to check.
        """
        variable = NetCDF.from_array(
            np.arange(8.0).reshape(2, 2, 2),
            geo_ref=NCGeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
        ).get_variable("t")
        arguments = {"where": (np.ones((2, 2, 2), dtype=bool),), "fillna": (0.0,)}
        result = getattr(variable, member)(*arguments.get(member, ()))
        assert list(result.to_dataframe().columns) == ["t"]
        assert list(result.to_dataframe(variables="t").columns) == ["t"]


class TestANanOtherIsDeclared:
    """Filling with NaN must leave the result honest about what is missing."""

    @staticmethod
    def _flagged() -> Dataset:
        """A `uint8` raster declaring 255 for its gaps.

        Returns:
            Dataset: The raster.
        """
        values = np.arange(1, 10, dtype="uint8").reshape(3, 3)
        return Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=255)

    def test_the_result_declares_nan(self):
        """A raster whose gaps are NaN says so, rather than naming a value it never holds.

        Test scenario:
            The declared sentinel was replaced by NaN only when the source declared
            nothing, so a `uint8` raster declaring `255` answered a float64 result still
            declaring `255` — a value absent from it — and then claimed nothing was
            missing: `isnull` all zero, `fillna` filling nothing, and the result not even
            equal to its own copy.
        """
        result = self._flagged().where(np.arange(1, 10).reshape(3, 3) > 4, np.nan)
        assert np.isnan(result.no_data_value[0])

    def test_it_knows_its_own_gaps(self):
        """`isnull` finds the cells the mask removed."""
        result = self._flagged().where(np.arange(1, 10).reshape(3, 3) > 4, np.nan)
        assert np.asarray(result.isnull().read_array()).sum() == 4

    def test_it_equals_its_own_copy(self):
        """A raster that does not equal its own copy is broken by definition."""
        result = self._flagged().where(np.arange(1, 10).reshape(3, 3) > 4, np.nan)
        assert result.equals(result.copy())

    def test_fillna_can_reach_those_cells(self):
        """The gaps are real gaps, so `fillna` writes to them."""
        result = self._flagged().where(np.arange(1, 10).reshape(3, 3) > 4, np.nan)
        filled = np.asarray(result.fillna(0.0).read_array(), dtype="float64")
        assert not np.isnan(filled).any()

    def test_a_gap_the_condition_selected_stays_a_gap(self):
        """The cells that were already missing must not become `-9999.0` measurements.

        Test scenario:
            The fixture above holds no gap at all, so it never exercised the cell this is
            about: a cell the condition *selected* that was already missing. The result
            declares NaN while those cells still held the old numeric sentinel, so every
            one of them was reclassified as data — `isnull` moved from `[[0, 1], [0, 0]]`
            to `[[0, 0], [0, 1]]`. xarray answers `[[1.0, nan], [3.0, nan]]` here.
        """
        raster = _raster(np.array([[1.0, NDV], [3.0, 4.0]]))
        assert np.asarray(raster.isnull().read_array()).tolist() == [[0, 1], [0, 0]]
        result = raster.where(np.array([[True, True], [True, False]]), np.nan)
        assert np.isnan(result.no_data_value[0])
        assert np.asarray(result.isnull().read_array()).tolist() == [[0, 1], [0, 1]]
        assert_allclose(
            _read(result), np.array([[1.0, np.nan], [3.0, np.nan]]), equal_nan=True
        )

    def test_the_old_sentinel_is_nowhere_in_the_result(self):
        """A raster declaring NaN must not still be carrying `-9999.0` cells."""
        raster = _raster(np.array([[1.0, NDV], [3.0, 4.0]]))
        result = raster.where(np.array([[True, True], [True, False]]), np.nan)
        assert not (np.asarray(result.read_array()) == NDV).any()

    def test_other_none_takes_the_same_path(self):
        """`other=None` resolves to NaN, so it must clear the old sentinel too."""
        raster = _raster(np.array([[1.0, NDV], [3.0, 4.0]]))
        result = raster.where(np.array([[True, True], [True, False]]), None)
        assert np.isnan(result.no_data_value[0])
        assert not (np.asarray(result.read_array()) == NDV).any()


class TestDropOnASouthUpRaster:
    """A raster whose rows run south to north trims like any other."""

    def test_it_does_not_refuse_what_crop_accepts(self):
        """`drop=True` builds its bbox from the edges, whichever way the rows run.

        Test scenario:
            The north edge was computed as `geo[3] + rows[0] * geo[5]`, which assumes a
            negative row height. On a south-up geotransform that put south above north and
            the call died inside `crop` with
            `ValueError: bbox must satisfy south < north` — on a raster `crop(bbox=...)`
            accepts directly.
        """
        south_up = GeoReference(geo=(0.0, 1.0, 0.0, 0.0, 0.0, 1.0), epsg=4326)
        values = np.arange(1, 10, dtype="float64").reshape(3, 3)
        raster = Dataset.from_array(values, geo_ref=south_up, no_data_value=NDV)
        result = raster.where(values > 6.0, drop=True)
        assert (result.rows, result.columns) == (1, 3)


class TestDropOnARotatedGrid:
    """The trim is an index slice, so a skewed geotransform moves its corner too."""

    ROTATED = (100.0, 2.0, 0.5, 500.0, 0.25, -2.0)
    """A grid with both rotation terms non-zero, so neither can be dropped unnoticed."""

    @staticmethod
    def _trimmed() -> Dataset:
        """The 3x4 rotated raster trimmed to the interior 2x2 block.

        Returns:
            Dataset: The trimmed raster.
        """
        raster = Dataset.from_array(
            CLEAN,
            geo_ref=GeoReference(geo=TestDropOnARotatedGrid.ROTATED, epsg=4326),
            no_data_value=NDV,
        )
        condition = np.zeros(CLEAN.shape, dtype=bool)
        condition[1:3, 1:3] = True
        return raster.where(condition, drop=True)

    def test_the_origin_moves_through_both_skews(self):
        """The corner is `origin + left * column_terms + top * row_terms`, both axes.

        Test scenario:
            The trim starts one row down and one column across, so the new corner picks
            up a skew term on each axis: `100 + 1*2.0 + 1*0.5` east and
            `500 + 1*0.25 + 1*-2.0` north. Arithmetic that ignored the skews would answer
            `102.0` and `498.0` and place the block off its own grid.
        """
        geo = self._trimmed().geotransform
        assert geo[0] == pytest.approx(102.5)
        assert geo[3] == pytest.approx(498.25)

    def test_the_rotation_terms_survive(self):
        """Only the corner moves: the cell size and both skews are the raster's own."""
        geo = self._trimmed().geotransform
        assert (geo[1], geo[2], geo[4], geo[5]) == (
            self.ROTATED[1],
            self.ROTATED[2],
            self.ROTATED[4],
            self.ROTATED[5],
        )

    def test_it_keeps_the_block_the_condition_selected(self):
        """The values are the interior block, not a resampled or re-cropped one."""
        trimmed = self._trimmed()
        assert (trimmed.rows, trimmed.columns) == (2, 2)
        assert_allclose(_read(trimmed), [[6.0, 7.0], [10.0, 11.0]])


class TestTheResultKeepsItsType:
    """Masking must not quietly double the raster's footprint."""

    @pytest.mark.parametrize(
        ("dtype", "sentinel"),
        [
            ("float32", None),
            ("float32", -9999.0),
            ("float64", -9999.0),
            ("uint8", 255),
            ("int16", -1),
        ],
    )
    def test_the_dtype_survives_a_mask(self, dtype, sentinel):
        """`where` preserves the band type, as `+`, `fill` and `fillna` do.

        Test scenario:
            The fill went in as a Python float, so `np.where` promoted the whole result to
            float64: a `float32` raster doubled and a `uint8` one octupled, undocumented.
            It also made `raster.where(raster.notnull())` — advertised as a no-op — change
            the type.

            A float band needs no sentinel, since it can hold NaN at its own width; an
            integer band can only mark a gap with a declared value, which is why these two
            declare one. An integer band without one is the case in
            `test_an_integer_band_without_a_sentinel_must_widen`.

        Args:
            dtype: The band type under test.
            sentinel: The no-data value to declare, or `None`.
        """
        values = np.arange(1, 10).reshape(3, 3).astype(dtype)
        raster = Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=sentinel)
        masked = raster.where(values > 4)
        assert np.asarray(masked.read_array()).dtype == np.dtype(dtype)

    def test_an_integer_band_without_a_sentinel_must_widen(self):
        """There is no integer that means "missing", so the gaps need a float band."""
        values = np.arange(1, 10, dtype="uint8").reshape(3, 3)
        raster = Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=None)
        masked = raster.where(values > 4)
        assert np.asarray(masked.read_array()).dtype == np.float64
        assert np.isnan(masked.no_data_value[0])

    def test_where_notnull_is_really_a_no_op(self):
        """The docstring calls it one, so it has to answer the same raster.

        Test scenario:
            `equals` reads the values and not the band type, so it answered `True` while
            the result was a `float64` copy of a `float32` raster — the dtype is asserted
            separately for that reason.
        """
        values = np.arange(1, 10, dtype="float32").reshape(3, 3)
        raster = Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=-9999.0)
        result = raster.where(raster.notnull())
        assert result.equals(raster)
        assert np.asarray(result.read_array()).dtype == np.float32

    def test_a_fractional_other_still_widens_an_integer_band(self):
        """A fill the band cannot hold is a real reason to promote, and still does."""
        values = np.arange(1, 10, dtype="int16").reshape(3, 3)
        raster = Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=None)
        assert (
            np.asarray(raster.where(values > 4, 0.5).read_array()).dtype == np.float64
        )


class TestAContainerIsRefusedByName:
    """A container has no raster of its own, and all six members say so the same way."""

    @staticmethod
    def _container() -> NetCDF:
        """A container holding one gridded variable.

        Returns:
            NetCDF: The container.
        """
        return NetCDF.from_array(
            np.arange(8.0).reshape(2, 2, 2),
            geo_ref=NCGeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
        )

    @pytest.mark.parametrize(
        "member", ["where", "fillna", "isnull", "notnull", "equals", "identical"]
    )
    def test_the_refusal_names_the_member_the_caller_called(self, member: str):
        """Five of the six answered a message about `read_array`, which nobody called.

        Test scenario:
            `where` got a bespoke refusal in round 1 and the other five fell through to
            the generic container guard: "Spatial operations are not supported on the
            NetCDF container. Use nc.get_variable('var_name').read_array(...) instead."
            The documentation advertises all six together, so they refuse alike.

        Args:
            member: The member under test.
        """
        container = self._container()
        arguments = {
            "where": (np.ones((2, 2, 2), dtype=bool),),
            "fillna": (0.0,),
            "equals": (container,),
            "identical": (container,),
        }
        call = getattr(container, member)
        with pytest.raises(ValueError, match=rf"^{member}\(\) works on a raster"):
            call(*arguments.get(member, ()))

    def test_the_refusal_names_a_variable_to_call_it_on(self):
        """A refusal that does not say what to do instead is half a refusal."""
        container = self._container()
        with pytest.raises(ValueError, match=r"get_variable\('t'\).fillna"):
            container.fillna(0.0)

    def test_a_variable_compared_with_a_container_still_answers_false(self):
        """The guard is about the receiver; anything may be the other operand."""
        container = self._container()
        assert not container.get_variable("t").equals(container)


class TestTheRefusalsAreSentences:
    """A bad argument is refused in words, not by whatever numpy happened to raise."""

    def test_a_non_numeric_other(self):
        """`other="x"` is a mistake worth naming.

        Test scenario:
            It reached `np.where` and answered numpy's
            `ValueError: could not convert string to float: 'x'`, which names neither the
            member nor the argument.
        """
        raster = _raster()
        with pytest.raises(TypeError, match="where.. needs a number"):
            raster.where(VALUES > 5, "x")

    @pytest.mark.parametrize(
        "value",
        [complex(1, 2), np.complex128(1 + 2j), np.complex64(1 + 2j)],
    )
    def test_a_complex_other_is_refused_however_it_is_spelled(self, value):
        """A complex fill cannot be a cell value under a real no-data value.

        Test scenario:
            The check read `isinstance(fill, (Real, np.number))`, and a `np.complex128`
            is an `np.number` without being a `Real` — so the numpy spelling was accepted
            where the Python one was refused, and produced a complex band declaring a
            real sentinel, with two `ComplexWarning`s on the way.

        Args:
            value: The complex fill under test.
        """
        raster = _raster()
        with pytest.raises(TypeError, match="needs a number"):
            raster.where(VALUES > 5, value)

    @pytest.mark.parametrize(
        "value", [2, 2.5, np.float32(0.5), np.int16(7), np.float64(-1.0)]
    )
    def test_every_real_spelling_is_still_accepted(self, value):
        """Refusing complex must not refuse the numpy reals alongside it.

        Args:
            value: The real fill under test.
        """
        assert _raster().where(VALUES > 5, value) is not None

    def test_other_none_means_nan_whatever_the_raster_declares(self):
        """An explicit `None` is NaN, not the raster's own `-9999.0`.

        Test scenario:
            The refusal offered `None` as "this raster's own no-data value", which it is
            not: leaving `other` out does that, and `None` asks for NaN whatever the
            raster declares. The result declares NaN too, so it can find those cells.
        """
        raster = _raster()
        assert raster.no_data_value[0] == NDV, "precondition: it declares a number"
        result = raster.where(VALUES > 5, None)
        assert np.isnan(result.no_data_value[0])
        assert np.isnan(_read(result)[0, 0])
