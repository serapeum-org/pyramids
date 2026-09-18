"""The edges of `weighted` — the weight forms, the axis resolution and the rebuilt footprint.

`test_weighted.py` covers the statistics against numpy and `test_weighted_dispatch.py` covers
which variables and axes take part. This module pins what those two reach but do not distinguish:
every form `weights` may take (a name, a list, a scalar, a multi-band raster, a variable), every
arm of the axis resolution (a band dimension named twice, a variable that declares no dimensions
of its own, a resolved plane that cannot be read), the refusals of an empty container and of a
grid with no CRS, the dropped auxiliary variable, and what the one-cell result's grid reports.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines._weighted import (
    _area_weights,
    _placed_shape,
    _spatial_names,
    _takes_part,
    _weighted_axes,
    _weighted_geotransform,
    _weighted_names,
    _weights_for,
)

pytestmark = pytest.mark.core

NT, NY, NX = 3, 4, 5
TIMES = [0.0, 6.0, 12.0]
GEO = GeoReference(geo=(-2.0, 1.0, 0.0, 10.0, 0.0, -1.0), epsg=4326)
PROJECTED = GeoReference(
    geo=(500000.0, 1000.0, 0.0, 4000000.0, 0.0, -1000.0), epsg=32618
)
NDV = -9999.0
ERA5_T2M = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-3d1__geog__y-desc.nc"
)
SPATIAL_BOUNDS = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__7v__1d3-2d3-3d1__y-asc.nc"
)


@pytest.fixture
def era5_latitude_variable() -> NetCDF:
    """The ERA5 `t2m` variable, whose store names its axes `latitude` / `longitude`.

    Returns:
        NetCDF: The variable.
    """
    return NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")


def _stack() -> np.ndarray:
    """The `(time, y, x)` values, one cell of the first step a gap.

    Returns:
        np.ndarray: A float64 stack holding `NDV` in the one gap.
    """
    values = np.arange(NT * NY * NX, dtype="float64").reshape(NT, NY, NX)
    values[0, 1, 2] = NDV
    return values


def _container(geo_ref: GeoReference = GEO) -> NetCDF:
    """An in-memory container holding the stack as variable `v` over `time`.

    Args:
        geo_ref: The georeference; geographic by default.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        _stack(),
        geo_ref=geo_ref,
        variable_name="v",
        no_data_value=NDV,
        dims=ExtraDimensions(name="time", values=TIMES),
    )


def _variable(geo_ref: GeoReference = GEO) -> NetCDF:
    """The container's variable `v`.

    Args:
        geo_ref: The georeference.

    Returns:
        NetCDF: The variable.
    """
    return _container(geo_ref).get_variable("v")


def _expected(weights: np.ndarray, how: str = "mean") -> np.ndarray:
    """The weighted statistic of the stack over its two spatial axes, one value per step.

    Args:
        weights: Weights broadcastable to `(NY, NX)`.
        how: `"mean"` or `"sum"`.

    Returns:
        np.ndarray: One value per step.
    """
    masked = np.where(_stack() == NDV, np.nan, _stack())
    spread = np.broadcast_to(np.asarray(weights, dtype="float64"), (NY, NX))
    answers = []
    for step in masked:
        valid = ~np.isnan(step)
        total = np.sum(spread[valid])
        weighted_sum = np.sum(spread[valid] * step[valid])
        answers.append(weighted_sum if how == "sum" else weighted_sum / total)
    return np.asarray(answers)


def _read(result: NetCDF) -> np.ndarray:
    """A container result's values as a flat float64 array.

    Args:
        result: A container holding `v`.

    Returns:
        np.ndarray: The values.
    """
    return np.asarray(result.get_variable("v").read_array(), dtype="float64").ravel()


class TestWeightForms:
    """Anything that resolves to an array of weights is accepted, however it was written."""

    @pytest.mark.parametrize(
        ("label", "weights"),
        [
            pytest.param(
                "per-row-list", [[1.0], [2.0], [3.0], [4.0]], id="list-of-rows"
            ),
            pytest.param("per-column-list", [1.0, 2.0, 3.0, 4.0, 5.0], id="flat-list"),
            pytest.param("scalar", 2.0, id="python-scalar"),
            pytest.param("zero-d", np.asarray(2.0), id="zero-d-array"),
            pytest.param(
                "int-array", np.ones((NY, NX), dtype="int32"), id="integer-array"
            ),
        ],
    )
    def test_the_weights_are_read_as_an_array(self, label, weights):
        """A list, a scalar or a 0-d array weights exactly as the equivalent array does.

        Args:
            label: What the form is, for the failure message.
            weights: The weights as passed.
        """
        result = _read(_container().weighted(weights))
        assert_allclose(
            result,
            _expected(np.asarray(weights, dtype="float64")),
            err_msg=f"{label} weights",
        )

    def test_a_scalar_is_the_plain_mean(self):
        """One weight for every cell is the arithmetic mean, whatever that weight is."""
        assert_allclose(
            _read(_container().weighted(7.0)), _read(_container().weighted(1.0))
        )

    def test_a_variable_of_weights_on_the_same_grid(self):
        """A `Variable` is read directly, without looking for a container's first variable."""
        weights = NetCDF.from_array(
            np.full((NY, NX), 3.0), geo_ref=GEO, variable_name="w"
        ).get_variable("w")
        assert_allclose(
            _read(_container().weighted(weights)), _expected(np.full((NY, NX), 3.0))
        )

    def test_a_multi_band_raster_uses_its_first_band(self):
        """A stack of weights is read through its first band, the rest ignored."""
        stack = NetCDF.from_array(
            np.stack([np.full((NY, NX), 3.0), np.full((NY, NX), 99.0)]),
            geo_ref=GEO,
            variable_name="w",
            dims=ExtraDimensions(name="band", values=[0.0, 1.0]),
        )
        assert_allclose(
            _read(_container().weighted(stack)), _expected(np.full((NY, NX), 3.0))
        )

    def test_the_second_band_is_not_what_is_read(self):
        """Reading the last band instead would answer the mean weighted by 99, which it must not."""
        stack = NetCDF.from_array(
            np.stack(
                [
                    np.arange(NY * NX, dtype="float64").reshape(NY, NX) + 1.0,
                    np.full((NY, NX), 99.0),
                ]
            ),
            geo_ref=GEO,
            variable_name="w",
            dims=ExtraDimensions(name="band", values=[0.0, 1.0]),
        )
        first = np.arange(NY * NX, dtype="float64").reshape(NY, NX) + 1.0
        assert_allclose(_read(_container().weighted(stack)), _expected(first))

    def test_weights_that_only_broadcast_onto_the_variable(self):
        """Grid-shaped weights weight one axis by falling back to the variable's own shape."""
        weights = np.arange(1.0, NY * NX + 1.0).reshape(NY, NX)
        variable = _container().weighted(weights, "x").get_variable("v")
        masked = np.where(_stack() == NDV, np.nan, _stack())
        valid = ~np.isnan(masked)
        total = np.sum(np.where(valid, weights, 0.0), axis=2)
        weighted_sum = np.sum(
            np.where(valid, weights * np.where(valid, masked, 0.0), 0.0), axis=2
        )
        expected = weighted_sum / np.where(total == 0, 1.0, total)
        assert_allclose(
            np.asarray(variable.read_array(), dtype="float64").reshape(NT, NY),
            expected,
        )

    def test_weights_for_spreads_onto_the_weighted_axes(self):
        """Per-row weights are spread over both weighted axes and placed at their positions."""
        spread = _weights_for(
            _variable(), np.arange(1.0, NY + 1.0).reshape(NY, 1), (NT, NY, NX), (1, 2)
        )
        assert spread.shape == (1, NY, NX), f"unexpected shape {spread.shape}"

    def test_weights_for_repeats_a_row_weight_across_the_columns(self):
        """Every column of a row carries that row's weight, which is what `"area"` relies on."""
        spread = _weights_for(
            _variable(), np.arange(1.0, NY + 1.0).reshape(NY, 1), (NT, NY, NX), (1, 2)
        )
        assert_allclose(
            np.asarray(spread)[0],
            np.broadcast_to(np.arange(1.0, NY + 1.0).reshape(NY, 1), (NY, NX)),
        )

    def test_weights_for_places_one_weighted_axis_alone(self):
        """Weighting the band dimension alone places its length at axis 0 and `1` elsewhere."""
        spread = _weights_for(_variable(), np.arange(1.0, NT + 1.0), (NT, NY, NX), (0,))
        assert spread.shape == (NT, 1, 1), f"unexpected shape {spread.shape}"

    def test_weights_for_reads_a_scalar_onto_the_weighted_axes(self):
        """A scalar spreads over the weighted axes alone, not over the band dimension."""
        spread = _weights_for(_variable(), 2.0, (NT, NY, NX), (1, 2))
        assert spread.shape == (1, NY, NX), f"unexpected shape {spread.shape}"

    def test_weights_for_falls_back_to_the_variable_shape(self):
        """Weights describing every cell keep the variable's shape when one axis is weighted."""
        spread = _weights_for(_variable(), np.ones((NY, NX)), (NT, NY, NX), (1,))
        assert spread.shape == (NT, NY, NX), f"unexpected shape {spread.shape}"


class TestPlacedShape:
    """`_placed_shape` puts each weighted axis' length at its own position."""

    @pytest.mark.parametrize(
        ("shape", "axes", "weighted_shape", "expected"),
        [
            pytest.param((3, 4, 5), (1, 2), (4, 5), [1, 4, 5], id="both-spatial"),
            pytest.param((3, 4, 5), (0,), (3,), [3, 1, 1], id="band-only"),
            pytest.param((3, 4, 5), (2,), (5,), [1, 1, 5], id="columns-only"),
            pytest.param((2, 3, 4, 5), (0, 1), (2, 3), [2, 3, 1, 1], id="two-bands"),
        ],
    )
    def test_the_placement(self, shape, axes, weighted_shape, expected):
        """The lengths land on the weighted axes and every other axis is `1`.

        Args:
            shape: The variable's unflattened shape.
            axes: The weighted axes.
            weighted_shape: Their lengths.
            expected: The shape the weights take.
        """
        assert _placed_shape(shape, axes, weighted_shape) == expected, (
            f"axes {axes} of {shape}"
        )


class TestSpatialNames:
    """The row and column axis names come from the axes the read resolved, or from the rebuild's.

    `test_weighted_dispatch.py` covers the three store shapes (declared out of order, declared
    last, built in memory). What is left here is the variable that declares nothing at all and
    the `y` / `x` aliases it answers, which is what an operator result weights through.
    """

    def test_a_variable_declaring_nothing_falls_back(self):
        """An operator result declares no dimensions, so the `y` / `x` a rebuild gives it stand."""
        doubled = _variable() * 2
        assert list(doubled._md_array_dims) == [], list(doubled._md_array_dims)

    def test_the_fallback_names_are_used(self):
        """`_spatial_names` answers `y` / `x` for a variable that declares no dimensions."""
        assert _spatial_names(_variable() * 2) == ("y", "x")

    def test_the_fallback_lets_the_default_dims_resolve(self):
        """`weighted` with no `dims` still collapses an operator result's grid to one cell."""
        result = (_variable() * 2).weighted(np.ones((NY, NX)))
        assert (result.rows, result.columns) == (1, 1), (result.rows, result.columns)

    def test_the_fallback_values_are_the_weighted_mean(self):
        """The operator result's weighted mean is twice the operand's, cell for cell."""
        assert_allclose(
            np.asarray(
                (_variable() * 2).weighted(np.ones((NY, NX))).read_array()
            ).ravel(),
            2.0 * _expected(np.ones((NY, NX))),
        )

    def test_the_read_resolved_plane_names_the_pair(self):
        """A store variable answers the axes its read resolved, recorded as `_md_spatial_dims`."""
        variable = _variable()
        column, row = variable._md_spatial_dims
        declared = list(variable._md_array_dims)
        assert _spatial_names(variable) == (declared[row], declared[column]), declared

    def test_without_that_record_the_last_two_declared_axes_are_taken(self, mocker):
        """A variable declaring dimensions but no resolved plane falls back to its last two.

        Args:
            mocker: The `pytest-mock` fixture, used to blank the resolved-plane record.

        Test scenario:
            Every variable a read produces records the plane, so the documented fallback is
            reached only when that record is missing. Blanking it leaves the declared
            `('time', 'y', 'x')`, whose last two are the pair.
        """
        variable = _variable()
        mocker.patch.object(variable, "_md_spatial_dims", None)
        assert _spatial_names(variable) == ("y", "x"), list(variable._md_array_dims)

    def test_the_fallback_reads_the_declared_order(self, mocker):
        """The fallback really reads the declared names, not the aliases the last resort answers.

        Args:
            mocker: The `pytest-mock` fixture, used to blank the record and rename the axes.
        """
        variable = _variable()
        mocker.patch.object(variable, "_md_spatial_dims", None)
        mocker.patch.object(variable, "_md_array_dims", ["time", "lat", "lon"])
        assert _spatial_names(variable) == ("lat", "lon"), (
            "the declared pair is the answer"
        )

    def test_a_record_the_declared_names_are_too_short_for_is_ignored(self, mocker):
        """A plane pointing past the declared names cannot be read, so the fallback takes over.

        Args:
            mocker: The `pytest-mock` fixture, used to set an out-of-range record.
        """
        variable = _variable()
        mocker.patch.object(variable, "_md_spatial_dims", (7, 6))
        assert _spatial_names(variable) == ("y", "x"), list(variable._md_array_dims)


class TestTakesPart:
    """`_takes_part` decides whether a container's variable is weighted or carried over."""

    def test_a_band_dimension_the_variable_has(self):
        """A variable carrying the weighted band dimension takes part."""
        assert _takes_part(_variable(), ("time",)) is True, (
            "time is a band dimension of v"
        )

    def test_a_band_dimension_the_variable_lacks(self):
        """A variable without the weighted band dimension is carried over instead."""
        assert _takes_part(_variable(), ("level",)) is False, "v has no level dimension"

    def test_the_store_names_of_the_spatial_axes(self, era5_latitude_variable):
        """A spatial axis named as the store names it is one the variable has.

        Args:
            era5_latitude_variable: The ERA5 `t2m` variable fixture.
        """
        assert _takes_part(era5_latitude_variable, ("latitude", "longitude")) is True

    def test_the_aliases_of_the_spatial_axes(self, era5_latitude_variable):
        """`y` / `x` reach the grid whatever the store calls its axes.

        Args:
            era5_latitude_variable: The ERA5 `t2m` variable fixture.
        """
        assert _takes_part(era5_latitude_variable, ("y", "x")) is True

    def test_one_unknown_name_is_enough_to_stand_aside(self):
        """Every name has to be one the variable has, not just one of them."""
        assert _takes_part(_variable(), ("time", "depth")) is False, "depth is unknown"


class TestWeightedNames:
    """`dims` is a name, a sequence of names, or `None` for both spatial axes."""

    def test_none_asks_for_both_spatial_axes(self):
        """`None` resolves to the variable's row and column axis names."""
        assert _weighted_names(_variable(), None) == ("y", "x")

    def test_a_string_is_one_name(self):
        """A single string is not iterated character by character."""
        assert _weighted_names(_variable(), "time") == ("time",)

    def test_a_sequence_is_taken_as_given(self):
        """A list or tuple keeps its order, which decides nothing but reads back the same."""
        assert _weighted_names(_variable(), ["x", "y"]) == ("x", "y")

    def test_a_generator_is_consumed_once(self):
        """A generator is materialised into a tuple, so the names can be read more than once."""
        assert _weighted_names(_variable(), (name for name in ("y", "x"))) == ("y", "x")


class TestWeightedAxes:
    """Names resolve to the axes of the unflattened `(*band_dims, rows, columns)` array."""

    @pytest.mark.parametrize(
        ("dims", "axes", "names"),
        [
            pytest.param(None, (1, 2), ("y", "x"), id="default"),
            pytest.param("y", (1,), ("y",), id="rows"),
            pytest.param("x", (2,), ("x",), id="columns"),
            pytest.param(("x", "y"), (1, 2), ("x", "y"), id="out-of-order"),
            pytest.param("time", (0,), ("time",), id="band"),
        ],
    )
    def test_the_axes_are_sorted_and_the_names_are_not(self, dims, axes, names):
        """The axes come back ascending while the names come back as they were given.

        Args:
            dims: The `dims` as passed.
            axes: The expected axes.
            names: The expected names.
        """
        assert _weighted_axes(_variable(), dims) == (axes, names), (
            f"dims={dims!r} resolved wrongly"
        )

    def test_a_generator_resolves_to_both_spatial_axes(self):
        """A generator of names resolves like the tuple it yields, not to an empty `dims`."""
        resolved = _weighted_axes(_variable(), (name for name in ("y", "x")))
        assert resolved[0] == (1, 2), resolved

    def test_a_generator_weights_a_variable(self):
        """A variable resolves `dims` once, so a one-shot iterable reaches the statistic."""
        result = _variable().weighted(np.ones((NY, NX)), (n for n in ("y", "x")))
        assert_allclose(
            np.asarray(result.read_array(), dtype="float64").ravel(),
            _expected(np.ones((NY, NX))),
        )

    def test_a_generator_weights_a_container_the_same_way(self):
        """A container resolves `dims` as a variable does, whatever iterable carries it.

        Test scenario:
            A container reads `dims` once to decide which variables take part and again for each
            variable's axes, so a one-shot iterable used to be exhausted by the first read and
            the call was refused with `weighted() needs at least one dimension to weight over.`
            — blaming the caller for an empty `dims` it never passed.
        """
        result = _container().weighted(np.ones((NY, NX)), (n for n in ("y", "x")))
        assert_allclose(_read(result), _expected(np.ones((NY, NX))))

    def test_a_band_dimension_named_twice(self):
        """The same band dimension twice would weight it against itself, so it is refused."""
        variable = _variable()
        with pytest.raises(ValueError, match=r"was given 'time' twice"):
            _weighted_axes(variable, ("time", "time"))

    def test_a_column_axis_named_by_alias_and_by_store_name(
        self, era5_latitude_variable
    ):
        """`latitude` and `y` name the same axis, so asking for both is asking twice.

        Args:
            era5_latitude_variable: The ERA5 `t2m` variable fixture.
        """
        with pytest.raises(ValueError, match="twice"):
            _weighted_axes(era5_latitude_variable, ("latitude", "y"))

    def test_an_empty_dims_is_refused(self):
        """An empty sequence leaves nothing to weight over."""
        variable = _variable()
        with pytest.raises(ValueError, match="at least one dimension"):
            _weighted_axes(variable, [])

    def test_an_unknown_name_lists_what_there_is(self):
        """The refusal names the band dimensions and the two spatial axes."""
        variable = _variable()
        with pytest.raises(ValueError, match=r"\['time', 'y', 'x'\]"):
            _weighted_axes(variable, "depth")

    def test_a_band_dimension_and_a_spatial_axis_cannot_mix(self):
        """One call weights band dimensions or spatial axes, never both."""
        variable = _variable()
        with pytest.raises(ValueError, match="not both"):
            _weighted_axes(variable, ("time", "x"))

    def test_two_spatial_axes_do_mix(self):
        """Two spatial axes are not a mix, so the guard does not fire on the default."""
        assert _weighted_axes(_variable(), ("y", "x"))[0] == (1, 2)

    def test_two_band_dimensions_do_mix(self):
        """Two band dimensions are not a mix either, so a 4-D variable can weight both."""
        four = NetCDF.from_array(
            np.arange(2 * NT * NY * NX, dtype="float64").reshape(2, NT, NY, NX),
            geo_ref=GEO,
            variable_name="v",
            dims=ExtraDimensions(dims=[("level", [1000.0, 850.0]), ("time", TIMES)]),
        ).get_variable("v")
        assert _weighted_axes(four, ("level", "time"))[0] == (0, 1)


class TestWeightedTwoBandDimensions:
    """Weighting two band dimensions keeps the grid and removes them both."""

    @staticmethod
    def _four_dimensional() -> NetCDF:
        """A `(level, time, y, x)` variable of ones.

        Returns:
            NetCDF: The variable.
        """
        return NetCDF.from_array(
            np.arange(2 * NT * NY * NX, dtype="float64").reshape(2, NT, NY, NX),
            geo_ref=GEO,
            variable_name="v",
            dims=ExtraDimensions(dims=[("level", [1000.0, 850.0]), ("time", TIMES)]),
        ).get_variable("v")

    def test_the_grid_survives(self):
        """Neither spatial axis is touched, so the result keeps every cell of the grid."""
        result = self._four_dimensional().weighted(np.ones((2, NT)), ("level", "time"))
        assert (result.rows, result.columns) == (NY, NX), (result.rows, result.columns)

    def test_both_dimensions_are_removed(self):
        """The two weighted dimensions go, as a collapsing `reduce` removes one."""
        result = self._four_dimensional().weighted(np.ones((2, NT)), ("level", "time"))
        assert tuple(result._band_dim_names) == (), tuple(result._band_dim_names)

    def test_the_values_are_the_mean_over_both(self):
        """Equal weights over both dimensions give the plain mean of all six steps."""
        result = self._four_dimensional().weighted(np.ones((2, NT)), ("level", "time"))
        source = np.arange(2 * NT * NY * NX, dtype="float64").reshape(2, NT, NY, NX)
        assert_allclose(
            np.asarray(result.read_array(), dtype="float64"),
            source.mean(axis=(0, 1)),
        )


class TestAreaWeights:
    """`"area"` is `cos(latitude)` per row, which needs a geographic CRS to read a latitude from."""

    def test_the_weights_are_one_per_row(self):
        """The weights are shaped `(rows, 1)`, so they broadcast across the columns."""
        assert _area_weights(_variable()).shape == (NY, 1), _area_weights(
            _variable()
        ).shape

    def test_the_weights_are_the_row_centres_cosine(self):
        """Each weight is the cosine of its row's centre latitude, not of its edge."""
        top, step = GEO.geo[3], GEO.geo[5]
        centres = np.asarray([top + (row + 0.5) * step for row in range(NY)])
        assert_allclose(_area_weights(_variable()).ravel(), np.cos(np.deg2rad(centres)))

    def test_a_projected_grid_is_refused_by_name(self):
        """The refusal names the CRS, so the caller can see what it actually has."""
        variable = _variable(PROJECTED)
        with pytest.raises(ValueError, match="geographic"):
            _area_weights(variable)

    def test_a_grid_with_no_crs_at_all_is_refused(self, mocker):
        """With neither an EPSG code nor a WKT there is no CRS to ask whether it is geographic.

        Args:
            mocker: The `pytest-mock` fixture, used to blank both CRS properties.

        Test scenario:
            `from_array` always stamps a CRS, so the two properties are blanked to reach the
            `require_crs_spec` guard, which must name the operation rather than let a `None`
            spec reach pyproj.
        """
        variable = _variable()
        mocker.patch.object(
            type(variable), "epsg", new_callable=mocker.PropertyMock, return_value=None
        )
        mocker.patch.object(
            type(variable), "crs", new_callable=mocker.PropertyMock, return_value=""
        )
        with pytest.raises(ValueError, match="area weights"):
            variable.weighted("area")


class TestWeightedSkipna:
    """`skipna=False` weights the stored values, so a declared sentinel takes part as a number."""

    @staticmethod
    def _nan_gapped() -> NetCDF:
        """A container whose first step holds one NaN cell and which declares NaN as its gap.

        Returns:
            NetCDF: The container.
        """
        values = _stack()
        values[0, 1, 2] = np.nan
        return NetCDF.from_array(
            values,
            geo_ref=GEO,
            variable_name="v",
            no_data_value=np.nan,
            dims=ExtraDimensions(name="time", values=TIMES),
        )

    def test_a_stored_nan_is_skipped_whatever_skipna_says(self):
        """A NaN leaves both sums either way, so the flag changes nothing for a NaN-gapped band.

        Test scenario:
            The sums are masked on `~isnan` after the `skipna` switch, so `skipna` only decides
            whether the declared sentinel becomes a NaN first. This is a deliberate divergence
            from xarray, whose `skipna=False` makes the whole answer NaN, and from
            `reduce("time", "mean", skipna=False)`, which answers NaN for a column holding one;
            both `Selection.weighted` and `_weighted_statistic` document it. Pinned here so the
            two answers cannot drift apart unnoticed.
        """
        skipping = _read(self._nan_gapped().weighted(np.ones((NY, NX))))
        keeping = _read(self._nan_gapped().weighted(np.ones((NY, NX)), skipna=False))
        assert_allclose(
            keeping, skipping, err_msg="skipna must not change a NaN-gapped band"
        )

    def test_the_nan_cell_is_left_out_of_both_sums(self):
        """The gapped step's answer is the mean of the cells that are not NaN, not NaN itself."""
        source = np.asarray(
            self._nan_gapped().get_variable("v").read_array(), dtype="float64"
        )
        result = _read(self._nan_gapped().weighted(np.ones((NY, NX)), skipna=False))
        assert result[0] == pytest.approx(np.nanmean(source[0])), result[0]

    def test_the_other_slices_are_still_the_plain_mean(self):
        """A step with no gap is unaffected by `skipna=False`."""
        source = np.asarray(
            self._nan_gapped().get_variable("v").read_array(), dtype="float64"
        )
        result = _read(self._nan_gapped().weighted(np.ones((NY, NX)), skipna=False))
        assert result[1] == pytest.approx(source[1].mean()), result[1]

    def test_a_sentinel_is_weighted_as_a_number(self):
        """Without skipping, the declared `-9999.0` takes part as the value it is."""
        result = _read(_container().weighted(np.ones((NY, NX)), skipna=False))
        assert result[0] == pytest.approx(_stack()[0].mean()), result[0]

    def test_skipping_is_the_default(self):
        """The default answers the mean of the cells there are, not the sentinel's arithmetic."""
        result = _read(_container().weighted(np.ones((NY, NX))))
        assert result[0] == pytest.approx(_expected(np.ones((NY, NX)))[0]), result[0]


class TestWeightedEmptyContainer:
    """A container with no data variables has nothing to weight."""

    def test_the_refusal_names_the_operation(self):
        """The message reads `"Cannot weight an empty container"`, matching the other members."""
        container = _container()
        container.remove_variable("v")
        with pytest.raises(ValueError, match="Cannot weight an empty container"):
            container.weighted("area")

    def test_an_empty_container_is_refused_before_the_weights_are_read(self):
        """Unusable weights are not what the caller hears about: the empty container comes first."""
        container = _container()
        container.remove_variable("v")
        with pytest.raises(ValueError, match="empty container"):
            container.weighted(np.ones((99, 99)))


class TestWeightedAuxiliaries:
    """A weighted band dimension is a removed one, so an auxiliary spanning it cannot be carried."""

    @staticmethod
    def _era5() -> tuple[NetCDF, int]:
        """The ERA5 container and the length of its `valid_time` dimension.

        Returns:
            tuple: The container and the number of steps.
        """
        container = NetCDF.read_file(str(ERA5_T2M))
        return container, container.get_variable("t2m").band_count

    def test_weighting_a_band_dimension_drops_a_spanning_auxiliary(self):
        """`expver` spans `valid_time`, which the weighting removes, so it is dropped."""
        container, steps = self._era5()
        with pytest.warns(
            UserWarning, match=r"weighted\(\) dropped auxiliary variable"
        ):
            container.weighted(np.ones(steps), "valid_time")

    def test_the_warning_names_the_removed_dimension(self):
        """The message names `valid_time`, the dimension the weighting removed."""
        container, steps = self._era5()
        with pytest.warns(UserWarning, match=r"the reduced dimension 'valid_time'"):
            container.weighted(np.ones(steps), "valid_time")

    def test_the_dropped_variable_is_gone_from_the_result(self):
        """Carrying it would leave an inconsistent `valid_time` length, so it is not carried."""
        container, steps = self._era5()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = container.weighted(np.ones(steps), "valid_time")
        assert "expver" not in result.variable_names, result.variable_names

    def test_weighting_the_grid_keeps_it(self):
        """A spatial weighting does not touch `valid_time`, so the auxiliary comes along."""
        container, _ = self._era5()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = container.weighted("area")
        assert "expver" in result.variable_names, result.variable_names

    @staticmethod
    def _bounded() -> NetCDF:
        """A CF store whose auxiliary variables span its spatial axes.

        Returns:
            NetCDF: `tos(time, lat, lon)` at 170x180, beside `lat_bnds(lat, bnds)`,
            `lon_bnds(lon, bnds)` and `time_bnds(time, bnds)`.
        """
        return NetCDF.read_file(str(SPATIAL_BOUNDS))

    def test_weighting_the_grid_drops_an_auxiliary_spanning_it(self):
        """`lat_bnds` spans `lat`, which the weighting leaves one cell long, so it is dropped.

        Test scenario:
            The dropped list held band dimensions only, so a spatial weighting dropped nothing:
            the result carried `tos` at 1x1 beside `lat_bnds` at 170x2 and `lon_bnds` at 180x2
            — one container, three grids — and it survived `to_file`.
        """
        with pytest.warns(UserWarning, match="dropped auxiliary variable"):
            result = self._bounded().weighted("area")
        assert "lat_bnds" not in result.variable_names, result.variable_names
        assert "lon_bnds" not in result.variable_names, result.variable_names

    def test_nothing_left_spans_a_reduced_axis(self):
        """No variable in the result still declares `lat` or `lon`, which are now one cell.

        Test scenario:
            A carried auxiliary keeps its own shape — `time_bnds(time, bnds)` is untouched, as
            `reduce` leaves it — so what makes the container consistent is that nothing spans a
            dimension whose length changed.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self._bounded().weighted("area")
        group = result._working_group()
        for name in result.variable_names:
            spanned = result._variable_dim_names(group, name)
            assert "lat" not in spanned, (name, spanned)
            assert "lon" not in spanned, (name, spanned)

    def test_an_auxiliary_on_an_untouched_dimension_survives(self):
        """`time_bnds` spans `time`, which a spatial weighting does not touch."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self._bounded().weighted("area")
        assert "time_bnds" in result.variable_names, result.variable_names

    def test_the_warning_names_the_callers_line(self):
        """The `UserWarning` is attributed to this file, not to a frame inside pyramids.

        Test scenario:
            `_carry_auxiliaries` counted the frames of the along-dimension loop, and
            `weighted` reaches it one frame deeper, so the warning was reported against
            `netcdf.py` — invisible to a user filtering by module or running
            `-W error::UserWarning`.
        """
        container, steps = self._era5()
        with pytest.warns(UserWarning) as caught:
            container.weighted(np.ones(steps), "valid_time")
        assert caught[0].filename == __file__, caught[0].filename

    def test_a_variable_receivers_warning_names_the_callers_line(self):
        """A weighted container reached through `reduce`'s loop reports the caller too."""
        container, steps = self._era5()
        with pytest.warns(UserWarning) as caught:
            container.reduce("valid_time", "mean")
        assert caught[0].filename == __file__, caught[0].filename

    def test_weighting_one_axis_drops_only_that_axis_auxiliary(self):
        """Weighting `x` leaves `lat` alone, so `lat_bnds` comes along and `lon_bnds` does not."""
        with pytest.warns(UserWarning, match="the reduced dimension 'lon'"):
            result = self._bounded().weighted(np.ones((170, 180)), "x")
        assert "lat_bnds" in result.variable_names, result.variable_names
        assert "lon_bnds" not in result.variable_names, result.variable_names


class TestWeightedGeotransform:
    """A reduced spatial axis becomes one cell spanning the source's whole extent along it."""

    @pytest.mark.parametrize(
        ("rows", "columns", "expected"),
        [
            pytest.param(False, False, (-2.0, 1.0, 0.0, 10.0, 0.0, -1.0), id="neither"),
            pytest.param(True, False, (-2.0, 1.0, 0.0, 10.0, 0.0, -4.0), id="rows"),
            pytest.param(False, True, (-2.0, 5.0, 0.0, 10.0, 0.0, -1.0), id="columns"),
            pytest.param(True, True, (-2.0, 5.0, 0.0, 10.0, 0.0, -4.0), id="both"),
        ],
    )
    def test_the_widened_cell(self, rows, columns, expected):
        """Only the reduced axis' spacing grows, and the origin never moves.

        Args:
            rows: Whether the row axis was reduced.
            columns: Whether the column axis was reduced.
            expected: The geotransform.
        """
        assert _weighted_geotransform(_variable(), rows, columns) == expected, (
            f"rows={rows}, columns={columns}"
        )

    def test_the_single_cell_sits_at_the_extent_centre(self):
        """The container stores the reduced axes as one coordinate each, the extent's centre."""
        result = _container().weighted("area")
        centre_x = GEO.geo[0] + GEO.geo[1] * NX / 2
        assert float(result.get_dimension_values("x")[0]) == pytest.approx(centre_x)

    def test_a_kept_axis_keeps_its_coordinates(self):
        """Weighting the columns alone leaves the row coordinates exactly where they were."""
        source = _container().get_dimension_values("y")
        result = _container().weighted(np.ones((NY, NX)), "x")
        assert_allclose(
            np.asarray(result.get_dimension_values("y"), dtype="float64"),
            np.asarray(source, dtype="float64"),
        )

    def test_the_kept_axis_geotransform_degrades_with_the_reduced_one(self):
        """A single-cell axis costs the whole geotransform its spacing, the kept axis included.

        Test scenario:
            Weighting the columns leaves four rows, whose coordinates the container still stores
            exactly (pinned above). The variable's geotransform is recomputed from the stored
            coordinates, and the one-cell column axis carries no spacing, so the recompute falls
            back to index space for both axes: the rows read back at `0, -1, -2, -3` rather than
            at their real latitudes. Documented on `weighted` as the single cell's footprint
            limit; pinned here so a change to the rebuild is noticed.
        """
        variable = _container().weighted(np.ones((NY, NX)), "x").get_variable("v")
        assert variable.geotransform[5] == pytest.approx(-1.0), variable.geotransform


class TestWeightedContainerMixedVariables:
    """A container whose gridded variables do not all carry the weighted band dimension.

    `test_weighted_dispatch.py` pins the band-dimension carry and the refusal naming one string.
    What is left here is the spatial weighting of such a container, which reaches every variable,
    and the refusal's other arm, where `dims` is a sequence rather than a name.
    """

    @staticmethod
    def _mixed() -> NetCDF:
        """A container holding `v` over `time` and `flat`, which has no `time` dimension.

        Returns:
            NetCDF: The container.
        """
        container = _container()
        container.set_variable(
            "flat",
            Dataset.from_array(
                np.arange(NY * NX, dtype="float64").reshape(NY, NX), geo_ref=GEO
            ),
        )
        return container

    def test_a_spatial_weighting_reaches_every_variable(self):
        """Every gridded variable has the grid, so both are collapsed to one cell."""
        result = self._mixed().weighted("area")
        assert (
            result.get_variable("flat").rows,
            result.get_variable("flat").columns,
        ) == (1, 1), "the variable without a band dimension should still be weighted"

    def test_the_variable_with_the_dimension_is_still_weighted(self):
        """The carry does not turn the whole call into a no-op: `v` loses its `time` dimension."""
        result = self._mixed().weighted(np.ones(NT), "time")
        assert tuple(result.get_variable("v")._band_dim_names) == (), (
            result.get_variable("v")._band_dim_names
        )

    @pytest.mark.parametrize(
        ("dims", "named"),
        [
            pytest.param(["level"], r"\['level'\]", id="one-in-a-list"),
            pytest.param(("level", "depth"), r"\['level', 'depth'\]", id="two-names"),
        ],
    )
    def test_a_sequence_naming_no_dimension_is_listed_in_the_refusal(self, dims, named):
        """A `dims` sequence no variable carries is listed, where a single name is quoted.

        Args:
            dims: The `dims` as passed.
            named: The pattern the refusal must hold.
        """
        container = self._mixed()
        with pytest.raises(ValueError, match=named):
            container.weighted(np.ones(NT), dims)

    def test_a_single_name_is_quoted_not_listed(self):
        """One name is shown as `'level'`, so the message reads as the caller wrote it."""
        container = self._mixed()
        with pytest.raises(ValueError, match=r"Dimension 'level' is not"):
            container.weighted(np.ones(NT), "level")
