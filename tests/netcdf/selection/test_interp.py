"""Tests for `NetCDF.interp` / `NetCDF.interp_like` — band-axis 1-D interpolation.

`interp` resamples a band dimension onto new coordinate values with `scipy.interpolate.interp1d`;
a spatial axis is refused (that is a GDAL warp — `resample`/`to_crs`/`align`). `interp_like` does
the same onto another cube's coordinates, refusing when the spatial grids differ. Both match
xarray's `interp` for the band-axis case.

Style: Google-style docstrings, <=120 char lines, no inline imports, descriptive assertions.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0.0, 1.0, 0.0, -1.0)


def _cube(stamps, values=None, *, name="t", cols=1, dim="time", no_data_value=-9999.0):
    """A cube of `len(stamps)` bands over a `1 x cols` grid.

    Args:
        stamps: The band coordinate values.
        values: The band values; `arange` when omitted.
        name: The variable name.
        cols: The number of grid columns.
        dim: The band dimension name.
        no_data_value: The declared no-data value, or `None` for none.

    Returns:
        NetCDF: The container.
    """
    n = len(stamps)
    arr = (
        np.arange(n * cols, dtype="float64")
        if values is None
        else np.asarray(values, "float64")
    )
    return NetCDF.from_array(
        arr.reshape(n, 1, cols),
        geo_ref=GeoReference(geo=GEO, epsg=4326),
        variable_name=name,
        dims=ExtraDimensions(name=dim, values=list(stamps)),
        no_data_value=no_data_value,
    )


def _var(stamps, values=None, **kwargs):
    """A single variable, as `_cube(...).get_variable("t")`."""
    return _cube(stamps, values, **kwargs).get_variable(kwargs.get("name", "t"))


class TestInterpValues:
    """`interp` interpolates a band axis onto new stamps and relabels the coordinate."""

    def test_linear_midpoints(self):
        """Linear interpolation onto midpoints averages the bracketing steps."""
        out = _var([0.0, 10.0, 20.0], [0.0, 10.0, 20.0]).interp(time=[5.0, 15.0])
        assert_allclose(out.read_array().ravel(), [5.0, 15.0])
        assert out._band_dim_values_map["time"] == [5.0, 15.0]

    def test_nearest_takes_the_closer_step(self):
        """`method="nearest"` snaps each target to its closest source step's value."""
        out = _var([0.0, 10.0, 20.0], [0.0, 10.0, 20.0]).interp(
            time=[4.0, 16.0], method="nearest"
        )
        assert_allclose(out.read_array().ravel(), [0.0, 20.0])

    def test_cubic_needs_four_points_and_runs(self):
        """`method="cubic"` interpolates a 4-step axis without error."""
        out = _var([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 8.0, 27.0]).interp(
            time=[1.5], method="cubic"
        )
        assert out.read_array().size == 1, "one target step should give one band"

    def test_scalar_target_collapses_to_one_step(self):
        """A scalar target interpolates onto a single stamp."""
        out = _var([0.0, 10.0], [0.0, 10.0]).interp(time=5.0)
        assert out._band_dim_values_map["time"] == [5.0]
        assert_allclose(out.read_array().ravel(), [5.0])

    def test_unsorted_source_axis_matches_sorted(self):
        """A descending source axis gives the same answer as its ascending twin."""
        ascending = _var([0.0, 10.0, 20.0], [0.0, 10.0, 20.0]).interp(time=[7.0])
        descending = _var([20.0, 10.0, 0.0], [20.0, 10.0, 0.0]).interp(time=[7.0])
        assert_allclose(descending.read_array(), ascending.read_array())

    def test_multiple_dims_apply_sequentially(self):
        """Two `dim=targets` pairs interpolate one after the other."""
        cube = NetCDF.from_array(
            np.arange(6.0).reshape(3, 2, 1),
            geo_ref=GeoReference(geo=GEO, epsg=4326),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=[0.0, 10.0, 20.0]),
        )
        # Only `time` is a band dim here; assert chaining a single band dim twice is stable.
        once = cube.interp(time=[5.0, 15.0]).get_variable("t")
        assert once._band_dim_values_map["time"] == [5.0, 15.0]


class TestInterpNoData:
    """Out-of-range targets and gaps follow the variable's no-data contract."""

    def test_out_of_range_is_declared_no_data(self):
        """A target outside the source range fills the declared no-data value."""
        out = _var([0.0, 10.0], [0.0, 10.0], no_data_value=-9999.0).interp(
            time=[-5.0, 25.0]
        )
        assert_allclose(out.read_array().ravel(), [-9999.0, -9999.0])

    def test_out_of_range_is_nan_without_declared_no_data(self):
        """With no declared no-data, an out-of-range target is NaN (xarray parity)."""
        out = _var([0.0, 10.0], [0.0, 10.0], no_data_value=None).interp(time=[25.0])
        assert np.isnan(out.read_array()).all(), "out-of-range should be NaN"


class TestInterpContainer:
    """`interp` on a container interpolates every variable that has the dimension."""

    def test_container_interpolates_all_variables(self):
        """A 2-column cube interpolates each pixel's series independently."""
        cube = _cube([0.0, 10.0, 20.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], cols=2)
        out = cube.interp(time=[5.0, 15.0]).get_variable("t")
        assert_allclose(out.read_array().ravel(), [1.0, 2.0, 3.0, 4.0])
        assert out._band_dim_values_map["time"] == [5.0, 15.0]


class TestInterpLike:
    """`interp_like` interpolates shared band dims onto another cube's coordinates."""

    def test_onto_other_time_axis(self):
        """A 2-step cube takes a 3-step cube's finer time axis by interpolation."""
        coarse = _var([0.0, 20.0], [0.0, 20.0])
        fine = _var([0.0, 10.0, 20.0], [0.0, 0.0, 0.0])
        out = coarse.interp_like(fine)
        assert_allclose(out.read_array().ravel(), [0.0, 10.0, 20.0])
        assert out._band_dim_values_map["time"] == [0.0, 10.0, 20.0]

    def test_matches_the_equivalent_interp(self):
        """`interp_like(other)` equals `interp(dim=other's coords)`."""
        source = _var([0.0, 20.0], [0.0, 20.0])
        template = _var([0.0, 5.0, 20.0], [0.0, 0.0, 0.0])
        like = source.interp_like(template).read_array()
        explicit = source.interp(time=[0.0, 5.0, 20.0]).read_array()
        assert_allclose(like, explicit)


class TestInterpMatchesXarray:
    """The band-axis result agrees with `xarray.DataArray.interp` (parity)."""

    def test_linear_matches_xarray(self):
        """Linear interp of a time series matches xarray on the shared stamps."""
        xr = pytest.importorskip("xarray")
        values = [0.0, 4.0, 9.0, 16.0]
        stamps = [0.0, 10.0, 20.0, 30.0]
        targets = [5.0, 12.0, 27.0]
        got = _var(stamps, values, no_data_value=None).interp(time=targets)
        expected = (
            xr.DataArray(values, dims="time", coords={"time": stamps})
            .interp(time=targets)
            .values
        )
        assert_allclose(got.read_array().ravel(), expected)


class TestInterpRefusals:
    """`interp` / `interp_like` refuse the cases the design note pins."""

    def test_no_coords(self):
        """`interp()` with no dimension is refused."""
        with pytest.raises(ValueError, match="at least one dimension"):
            _var([0.0, 10.0]).interp()

    def test_unknown_method(self):
        """An unsupported `method` is refused with the allowed set."""
        with pytest.raises(ValueError, match="method must be one of"):
            _var([0.0, 10.0]).interp(time=[5.0], method="spline")

    @pytest.mark.parametrize("spatial", ["x", "y", "lon", "lat"])
    def test_spatial_axis_points_to_the_warp_verbs(self, spatial):
        """A spatial axis is refused, naming resample / to_crs / align / extract."""
        with pytest.raises(ValueError, match="resample|to_crs|align|extract") as excinfo:
            _var([0.0, 10.0]).interp(**{spatial: [0.5]})
        assert "spatial" in str(excinfo.value), "message should say the axis is spatial"

    def test_unknown_dimension(self):
        """A name that is no band dimension of the variable is refused."""
        with pytest.raises(ValueError, match="does not match any band dimension"):
            _var([0.0, 10.0]).interp(level=[1.0])

    def test_empty_target(self):
        """An empty target sequence is refused."""
        with pytest.raises(ValueError, match="is empty"):
            _var([0.0, 10.0]).interp(time=[])

    def test_nan_target(self):
        """A target holding NaN is refused."""
        with pytest.raises(ValueError, match="contains NaN"):
            _var([0.0, 10.0]).interp(time=[float("nan")])

    def test_interp_like_grid_mismatch(self):
        """`interp_like` across differing spatial grids is refused, naming the warp verbs."""
        source = _var([0.0, 20.0], [0.0, 20.0])
        other = NetCDF.from_array(
            np.zeros((2, 1, 1)),
            geo_ref=GeoReference(geo=GEO, epsg=3857),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=[0.0, 10.0]),
        ).get_variable("t")
        with pytest.raises(ValueError, match="to_crs|resample|align"):
            source.interp_like(other)

    def test_interp_like_no_shared_dim(self):
        """`interp_like` with no shared band dimension is refused."""
        source = _var([0.0, 20.0], [0.0, 20.0], dim="time")
        other = _var([0.0, 10.0], [0.0, 0.0], dim="level")
        with pytest.raises(ValueError, match="no band dimension shared"):
            source.interp_like(other)
