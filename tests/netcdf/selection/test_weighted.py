"""`weighted` — area-weighted statistics, which collapse the spatial axes onto one cell.

Every expectation is numpy: a weighted mean is `sum(w * x) / sum(w)` over the valid cells, and
`"area"` weights are `cos(latitude)` per row of a geographic grid. The result stays a raster —
one cell spanning the source extent when both spatial axes go — so it still selects, reduces
and writes like any other.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.netcdf import Variable

pytestmark = pytest.mark.core

NDV = -9999.0
NT, NY, NX = 3, 4, 5
TIMES = [0.0, 6.0, 12.0]
GEO = GeoReference(geo=(-2.0, 1.0, 0.0, 10.0, 0.0, -1.0), epsg=4326)
PROJECTED = GeoReference(
    geo=(500000.0, 1000.0, 0.0, 4000000.0, 0.0, -1000.0), epsg=32618
)
ERA5_T2M = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-3d1__geog__y-desc.nc"
)


def _values() -> np.ndarray:
    """The `(time, y, x)` stack with one gap cell and one all-gap row at the last step.

    Returns:
        np.ndarray: A float64 stack holding `NDV` in the gaps.
    """
    rng = np.random.default_rng(31)
    values = np.round(rng.uniform(0.0, 20.0, size=(NT, NY, NX)), 2)
    values[0, 1, 2] = NDV
    values[2] = NDV
    return values


def _masked() -> np.ndarray:
    """`_values()` with its gaps as NaN.

    Returns:
        np.ndarray: The stack with NaN gaps.
    """
    values = _values()
    return np.where(values == NDV, np.nan, values)


def _container(geo_ref: GeoReference = GEO) -> NetCDF:
    """An in-memory container holding the stack as variable `v` over `time`.

    Args:
        geo_ref: The georeference; geographic by default.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        _values(),
        geo_ref=geo_ref,
        variable_name="v",
        no_data_value=NDV,
        dims=ExtraDimensions(name="time", values=TIMES),
    )


def _latitudes(geo_ref: GeoReference = GEO) -> np.ndarray:
    """The latitude of each row centre of the test grid.

    Args:
        geo_ref: The georeference.

    Returns:
        np.ndarray: One latitude per row.
    """
    top, step = geo_ref.geo[3], geo_ref.geo[5]
    return np.asarray([top + (row + 0.5) * step for row in range(NY)])


def _area_weights() -> np.ndarray:
    """`cos(latitude)` per row, shaped `(NY, 1)` so it broadcasts over the columns.

    Returns:
        np.ndarray: The weights.
    """
    return np.cos(np.deg2rad(_latitudes())).reshape(NY, 1)


def _weighted(masked: np.ndarray, weights: np.ndarray, how: str = "mean") -> np.ndarray:
    """The weighted statistic over the two spatial axes, one value per step.

    Args:
        masked: The `(time, y, x)` stack with NaN gaps.
        weights: Weights broadcastable to `(y, x)`.
        how: `"mean"`, `"sum"`, `"sum_of_weights"`, `"var"` or `"std"`.

    Returns:
        np.ndarray: One value per step, NaN where no weight is left.
    """
    spread = np.broadcast_to(weights, masked.shape[1:])
    results = []
    for step in masked:
        valid = ~np.isnan(step)
        total = np.sum(spread[valid])
        if how == "sum_of_weights":
            results.append(total if valid.any() else np.nan)
            continue
        if not valid.any() or total == 0:
            results.append(np.nan)
            continue
        weighted_sum = np.sum(spread[valid] * step[valid])
        if how == "sum":
            results.append(weighted_sum)
        elif how == "mean":
            results.append(weighted_sum / total)
        else:
            mean = weighted_sum / total
            variance = np.sum(spread[valid] * (step[valid] - mean) ** 2) / total
            results.append(variance if how == "var" else np.sqrt(variance))
    return np.asarray(results)


def _read(result: NetCDF) -> np.ndarray:
    """A result's values as a flat float64 array, its no-data value read as NaN.

    Args:
        result: A container holding `v`, or a variable.

    Returns:
        np.ndarray: The values.
    """
    variable = result if isinstance(result, Variable) else result.get_variable("v")
    values = np.asarray(variable.read_array(), dtype=np.float64).ravel()
    ndv = variable.no_data_value[0]
    if ndv is not None and not np.isnan(ndv):
        values = np.where(values == ndv, np.nan, values)
    return values


class TestWeightedValues:
    """The statistics are the weighted formulas over the valid cells."""

    @pytest.mark.parametrize("how", ["mean", "sum", "sum_of_weights", "var", "std"])
    def test_area_weighted_statistics(self, how):
        """Each statistic with `cos(latitude)` weights, per step.

        Args:
            how: The statistic.
        """
        result = _container().weighted("area", how=how)
        assert_allclose(
            _read(result), _weighted(_masked(), _area_weights(), how), equal_nan=True
        )

    def test_uniform_weights_are_the_plain_mean(self):
        """With equal weights the weighted mean is the arithmetic mean of the valid cells."""
        weights = np.ones((NY, NX))
        result = _container().weighted(weights)
        masked = _masked()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmean(masked.reshape(NT, -1), axis=1)
        assert_allclose(_read(result), expected, equal_nan=True)

    def test_a_gap_drops_out_of_both_sums(self):
        """The one gap cell of step 0 leaves numerator and denominator alike."""
        weights = _area_weights()
        masked = _masked()
        result = _read(_container().weighted(weights))
        spread = np.broadcast_to(weights, masked.shape[1:])
        valid = ~np.isnan(masked[0])
        expected = np.sum(spread[valid] * masked[0][valid]) / np.sum(spread[valid])
        assert result[0] == pytest.approx(expected)

    def test_an_all_gap_step_is_no_data(self):
        """The last step is a gap throughout, so it has no weighted mean."""
        result = _container().weighted("area")
        assert np.isnan(_read(result)[2])

    def test_zero_weights_leave_no_data(self):
        """Weights that are all zero leave nothing to average."""
        result = _container().weighted(np.zeros((NY, NX)))
        assert np.all(np.isnan(_read(result)))

    def test_negative_weights_are_allowed(self):
        """A negative weight is used as given, as xarray uses it."""
        weights = np.full((NY, NX), -1.0)
        result = _container().weighted(weights)
        assert_allclose(_read(result), _weighted(_masked(), weights), equal_nan=True)

    def test_weights_from_a_netcdf_on_the_same_grid(self):
        """A `NetCDF` of weights on the same grid is read as the weights."""
        weights = np.abs(np.round(np.cos(np.deg2rad(_latitudes())), 3)).reshape(NY, 1)
        spread = np.broadcast_to(weights, (NY, NX)).copy()
        operand = NetCDF.from_array(spread, geo_ref=GEO, variable_name="w")
        result = _container().weighted(operand)
        assert_allclose(_read(result), _weighted(_masked(), spread), equal_nan=True)

    @pytest.mark.parametrize(
        "shape", [(NY, 1), (1, NX), (NY, NX)], ids=["per-row", "per-column", "per-cell"]
    )
    def test_weight_shapes_that_broadcast(self, shape):
        """Weights per row, per column or per cell all broadcast over the grid.

        Args:
            shape: The weights' shape.
        """
        weights = np.abs(np.random.default_rng(7).uniform(0.5, 2.0, size=shape))
        result = _container().weighted(weights)
        assert_allclose(_read(result), _weighted(_masked(), weights), equal_nan=True)


class TestWeightedDimensions:
    """`dims` picks which axes go; the result stays a raster."""

    def test_both_spatial_axes_leave_one_cell(self):
        """The default collapses the grid to one cell, centred on the source's extent."""
        result = _container().weighted("area")
        variable = result.get_variable("v")
        assert (variable.rows, variable.columns) == (1, 1)
        assert variable.epsg == 4326
        centre_x = GEO.geo[0] + GEO.geo[1] * NX / 2
        centre_y = GEO.geo[3] + GEO.geo[5] * NY / 2
        assert float(result.get_dimension_values("x")[0]) == pytest.approx(centre_x)
        assert float(result.get_dimension_values("y")[0]) == pytest.approx(centre_y)

    def test_a_single_cell_axis_cannot_carry_its_width(self):
        """The cell is centred right, but its width reads back as one index unit.

        Test scenario:
            The rebuilt store keeps a spatial axis as coordinate values, and one value carries no
            spacing, so `_compute_geotransform` falls back to the index-space cell size — the
            same limit a one-column `reduce` result has. The weighted values are exact; only the
            cell's footprint is understated.
        """
        variable = _container().weighted("area").get_variable("v")
        assert variable.geotransform[1] == pytest.approx(1.0)
        assert variable.geotransform[5] == pytest.approx(-1.0)
        assert float(variable.lat[0]) == pytest.approx(float(variable.lon[0]))

    @pytest.mark.parametrize(
        ("dims", "shape"),
        [
            pytest.param("y", (1, NX), id="rows-only"),
            pytest.param("x", (NY, 1), id="columns-only"),
            pytest.param(("y", "x"), (1, 1), id="both-named"),
        ],
    )
    def test_one_spatial_axis(self, dims, shape):
        """Collapsing one axis leaves the other in place.

        Args:
            dims: The axes to collapse.
            shape: The expected `(rows, columns)`.
        """
        variable = _container().weighted(np.ones((NY, NX)), dims).get_variable("v")
        assert variable.rows == shape[0], (variable.rows, shape)
        assert variable.columns == shape[1], (variable.columns, shape)

    def test_weights_over_the_whole_variable(self):
        """Weights shaped like the grid work when only one axis is weighted.

        Test scenario:
            The weighted axis alone is `(NY,)`, but weights of `(NY, NX)` describe every cell,
            so they broadcast against the variable itself and each column is weighted in turn.
        """
        weights = np.abs(np.random.default_rng(13).uniform(0.5, 2.0, size=(NY, NX)))
        variable = _container().weighted(weights, "y").get_variable("v")
        masked = _masked()
        valid = ~np.isnan(masked)
        total = np.sum(np.where(valid, weights, 0.0), axis=1)
        weighted_sum = np.sum(
            np.where(valid, weights * np.where(valid, masked, 0.0), 0.0), axis=1
        )
        expected = np.where(
            total > 0, weighted_sum / np.where(total == 0, 1.0, total), np.nan
        )
        read = np.asarray(variable.read_array(), dtype=np.float64).reshape(NT, 1, NX)
        assert_allclose(
            np.where(read == variable.no_data_value[0], np.nan, read)[:, 0, :],
            expected,
            equal_nan=True,
        )

    def test_a_spatial_axis_by_its_store_name(self):
        """ERA5 names its axes `latitude` / `longitude`, and those names work too."""
        container = NetCDF.read_file(str(ERA5_T2M))
        variable = container.get_variable("t2m").weighted(
            "area", ("latitude", "longitude")
        )
        assert (variable.rows, variable.columns) == (1, 1)
        assert variable.band_count == container.get_variable("t2m").band_count

    def test_a_band_dimension(self):
        """Weighting over `time` keeps the grid and removes the dimension."""
        weights = np.asarray([1.0, 3.0, 0.0])
        result = _container().weighted(weights, "time")
        variable = result.get_variable("v")
        assert (variable.rows, variable.columns) == (NY, NX)
        assert tuple(variable._band_dim_names) == ()
        masked = _masked()
        spread = weights.reshape(NT, 1, 1)
        valid = ~np.isnan(masked)
        total = np.sum(np.where(valid, spread, 0.0), axis=0)
        weighted_sum = np.nansum(np.where(valid, masked * spread, 0.0), axis=0)
        expected = np.where(
            total > 0, weighted_sum / np.where(total == 0, 1, total), np.nan
        )
        assert_allclose(
            np.asarray(variable.read_array(), dtype=np.float64),
            expected,
            equal_nan=True,
        )

    def test_the_band_dimensions_and_units_are_kept(self):
        """ERA5 `t2m` weighted over its grid keeps `valid_time` and still selects by date."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        result = variable.weighted("area")
        assert tuple(result._band_dim_names) == ("valid_time",)
        assert result.band_count == variable.band_count
        assert result.sel(valid_time="2022-01-01").band_count == 4


class TestWeightedReceivers:
    """A container and a variable weight the same cells."""

    def test_a_variable_weights_like_its_container(self):
        """`get_variable("v").weighted(...)` holds what `weighted(...).get_variable("v")` does."""
        container = _container()
        from_variable = container.get_variable("v").weighted("area")
        from_container = container.weighted("area")
        assert isinstance(from_variable, Variable), type(from_variable).__name__
        assert_allclose(_read(from_variable), _read(from_container), equal_nan=True)

    def test_an_auxiliary_variable_is_carried(self):
        """ERA5's `expver` is not gridded, so it is carried over unchanged."""
        container = NetCDF.read_file(str(ERA5_T2M))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = container.weighted("area")
        assert "expver" in result.variable_names, result.variable_names


class TestWeightedRefusals:
    """Unusable weights, dimensions or statistics are refused before any cell is read."""

    def test_area_on_a_projected_grid(self):
        """`"area"` is cos-latitude, which a projected grid has no latitudes for."""
        container = _container(PROJECTED)
        with pytest.raises(ValueError, match="geographic"):
            container.weighted("area")

    def test_an_unknown_weights_name(self):
        """Only `"area"` is a named weighting."""
        container = _container()
        with pytest.raises(ValueError, match="area"):
            container.weighted("cosine")

    def test_weights_holding_a_gap(self):
        """A NaN weight would make every statistic NaN, so it is refused, as xarray refuses it."""
        weights = np.ones((NY, NX))
        weights[0, 0] = np.nan
        container = _container()
        with pytest.raises(ValueError, match="missing values"):
            container.weighted(weights)

    def test_weights_that_do_not_broadcast(self):
        """Weights shaped for another grid are refused, naming both shapes."""
        container = _container()
        with pytest.raises(ValueError, match="broadcast"):
            container.weighted(np.ones((NY + 1, NX)))

    def test_weights_on_another_grid(self):
        """A `NetCDF` of weights on a different grid is refused."""
        other = NetCDF.from_array(
            np.ones((NY, NX)),
            geo_ref=GeoReference(geo=(0.0, 2.0, 0.0, 10.0, 0.0, -2.0), epsg=4326),
            variable_name="w",
        )
        container = _container()
        with pytest.raises(ValueError, match="same grid"):
            container.weighted(other)

    def test_an_unknown_statistic(self):
        """`how` is one of the weighted statistics; a quantile is not among them."""
        container = _container()
        with pytest.raises(ValueError, match="how must be one of"):
            container.weighted("area", how="quantile")

    def test_an_unknown_dimension(self):
        """A name that is neither a band dimension nor a spatial axis is refused."""
        container = _container()
        with pytest.raises(ValueError, match="level"):
            container.weighted("area", "level")

    def test_no_dimensions(self):
        """An empty `dims` leaves nothing to weight over."""
        container = _container()
        with pytest.raises(ValueError, match="at least one dimension"):
            container.weighted("area", ())

    def test_the_options_are_keyword_only(self):
        """`weighted("area", "y", "sum")` is a `TypeError`: `how` must be named."""
        container = _container()
        with pytest.raises(TypeError):
            container.weighted("area", "y", "sum")

    def test_mixing_a_spatial_axis_with_a_band_dimension(self):
        """Weighting a band dimension keeps the grid and weighting a spatial axis reduces it."""
        container = _container()
        with pytest.raises(ValueError, match="not both"):
            container.weighted("area", ("time", "y"))

    def test_a_dimension_named_twice(self):
        """The same axis twice would weight it against itself."""
        container = _container()
        with pytest.raises(ValueError, match="twice"):
            container.weighted("area", ("y", "y"))
