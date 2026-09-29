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


def _cube2(time_vals, level_vals, values=None, *, cols=1):
    """A 4-D cube with band dimensions `time` then `level` over a `1 x cols` grid.

    Args:
        time_vals: The `time` coordinate values.
        level_vals: The `level` coordinate values.
        values: The band values; `arange` when omitted.
        cols: The number of grid columns.

    Returns:
        NetCDF: The container.
    """
    nt, nl = len(time_vals), len(level_vals)
    arr = (
        np.arange(nt * nl * cols, dtype="float64")
        if values is None
        else np.asarray(values, "float64")
    )
    return NetCDF.from_array(
        arr.reshape(nt, nl, 1, cols),
        geo_ref=GeoReference(geo=GEO, epsg=4326),
        variable_name="t",
        dims=ExtraDimensions(dims=[("time", list(time_vals)), ("level", list(level_vals))]),
    )


class TestInterpKinds:
    """Every `interp1d` kind `interp` advertises runs and shapes the output correctly."""

    @pytest.mark.parametrize("kind", ["quadratic", "slinear", "previous", "next", "zero"])
    def test_kind_runs_on_a_four_step_axis(self, kind):
        """A supported non-default kind interpolates a 4-step axis to the target length.

        Args:
            kind: The interpolation kind under test.
        """
        out = _var([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 8.0, 27.0], no_data_value=None).interp(
            time=[0.5, 1.5, 2.5], method=kind
        )
        assert out.read_array().size == 3, f"{kind} should give three output bands"
        assert out._band_dim_values_map["time"] == [0.5, 1.5, 2.5]


class TestInterpSourceGaps:
    """A gap in the source series interpolates to a gap."""

    def test_declared_no_data_gap_propagates(self):
        """A target in a clean segment interpolates; one touching a no-data step stays no-data."""
        out = _var(
            [0.0, 10.0, 20.0, 30.0], [0.0, 10.0, -9999.0, 30.0], no_data_value=-9999.0
        ).interp(time=[5.0, 15.0])
        result = out.read_array().ravel()
        assert result[0] == 5.0, f"clean segment [0, 10] interpolates, got {result[0]}"
        assert result[1] == -9999.0, f"segment touching the gap stays no-data, got {result[1]}"

    def test_nan_gap_propagates_without_declared_no_data(self):
        """Without a declared no-data value, a segment touching a gap interpolates to NaN."""
        out = _var(
            [0.0, 10.0, 20.0, 30.0], [0.0, 10.0, float("nan"), 30.0], no_data_value=None
        ).interp(time=[15.0])
        assert np.isnan(out.read_array()).all(), "gap segment should be NaN"


class TestInterpLikeShared:
    """`interp_like` interpolates only the band dimensions shared with `other`."""

    def test_skips_a_dimension_other_lacks(self):
        """A dim present on this cube but not on `other` is left untouched."""
        source = _cube2([0.0, 10.0], [0.0, 100.0]).get_variable("t")
        other = _var([0.0, 5.0, 10.0], [0.0, 0.0, 0.0])
        out = source.interp_like(other)
        assert out._band_dim_values_map["time"] == [0.0, 5.0, 10.0], "time follows other"
        assert out._band_dim_values_map["level"] == [0.0, 100.0], "level is untouched"


class TestInterpMultiDim:
    """`interp` applies several `dim=targets` pairs one after another."""

    def test_two_band_dims_chained(self):
        """Interpolating `time` then `level` composes the two 1-D interpolations."""
        out = _cube2([0.0, 10.0], [0.0, 100.0]).interp(
            time=[5.0], level=[50.0]
        ).get_variable("t")
        assert out._band_dim_values_map["time"] == [5.0]
        assert out._band_dim_values_map["level"] == [50.0]
        assert_allclose(out.read_array().ravel(), [1.5])


class TestInterpMoreRefusals:
    """The remaining coordinate and target refusals."""

    def test_text_axis_is_refused(self):
        """A non-numeric band axis cannot be interpolated."""
        cube = NetCDF.from_array(
            np.arange(2.0).reshape(2, 1, 1),
            geo_ref=GeoReference(geo=GEO, epsg=4326),
            variable_name="t",
            dims=ExtraDimensions(name="scenario", values=["rcp45", "rcp85"]),
        )
        with pytest.raises(ValueError, match="numeric"):
            cube.interp(scenario=[0.5])

    def test_coordinate_less_axis_is_refused(self):
        """A band axis carrying no coordinates has nothing to interpolate from."""
        var = _var([0.0, 10.0, 20.0], [0.0, 10.0, 20.0])
        var._band_dim_values_map["time"] = None
        with pytest.raises(ValueError, match="coordinate-less axis"):
            var.interp(time=[5.0])

    def test_nan_source_coordinate_is_refused(self):
        """A source coordinate holding NaN is refused before interpolating."""
        var = _var([0.0, 10.0, 20.0], [0.0, 10.0, 20.0])
        var._band_dim_values_map["time"] = [0.0, float("nan"), 20.0]
        with pytest.raises(ValueError, match="contain NaN"):
            var.interp(time=[5.0])

    def test_two_dimensional_target_is_refused(self):
        """A 2-D target array is refused; targets must be 1-D."""
        with pytest.raises(ValueError, match="one-dimensional"):
            _var([0.0, 10.0], [0.0, 10.0]).interp(time=[[5.0], [6.0]])

    def test_interp_like_column_mismatch(self):
        """Differing column counts (same EPSG) refuse `interp_like` too."""
        source = _var([0.0, 20.0], [0.0, 20.0], cols=1)
        other = _cube([0.0, 10.0], [0.0, 0.0, 0.0, 0.0], cols=2).get_variable("t")
        with pytest.raises(ValueError, match="to_crs|resample|align"):
            source.interp_like(other)

    def test_container_unknown_dimension(self):
        """A name that is no dimension of a container is refused by the container path."""
        cube = _cube([0.0, 10.0], [0.0, 1.0])
        with pytest.raises(ValueError, match="not a dimension of this container"):
            cube.interp(depth=[5.0])


def _two_var_cube(stamps):
    """A container with two variables `a`, `b` both spanning `time` on the same 2x2 grid.

    Args:
        stamps: The shared `time` coordinate values.

    Returns:
        NetCDF: The two-variable container.
    """
    ref = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
    n = len(stamps)
    a = NetCDF.from_array(
        np.arange(n * 4, dtype="float64").reshape(n, 2, 2),
        geo_ref=ref,
        variable_name="a",
        dims=ExtraDimensions(name="time", values=list(stamps)),
    )
    b = NetCDF.from_array(
        (np.arange(n * 4, dtype="float64") + 100.0).reshape(n, 2, 2),
        geo_ref=ref,
        variable_name="b",
        dims=ExtraDimensions(name="time", values=list(stamps)),
    )
    a.set_variable("b", b.get_variable("b"))
    return a


class TestInterpMultiVariableContainer:
    """A multi-variable container interpolated to a single band keeps every variable's band (H1)."""

    def test_single_band_result_keeps_all_variables_metadata(self):
        """Every variable, not just the first, keeps the interpolated dim's name and coordinates."""
        cube = _two_var_cube([0.0, 10.0])
        assert sorted(cube.variable_names) == ["a", "b"], cube.variable_names
        out = cube.interp(time=[5.0])
        for name in ("a", "b"):
            v = out.get_variable(name)
            assert v._band_dim_names == ("time",), (
                f"{name} lost its band dim: {v._band_dim_names}"
            )
            assert v._band_dim_values_map["time"] == [5.0], (
                f"{name} lost its time coordinate: {v._band_dim_values_map}"
            )
            v.sel(time=5.0)


class TestInterpSplineGapPropagation:
    """Spline kinds fit globally, so any gap makes the whole axis no-data (review M1)."""

    @pytest.mark.parametrize("kind", ["cubic", "quadratic"])
    def test_spline_gap_propagates_to_whole_axis(self, kind):
        """A single interior no-data cell makes a cubic/quadratic interp all no-data (scipy/xarray).

        Args:
            kind: The global-spline interpolation kind under test.
        """
        out = _var(
            [0.0, 10.0, 20.0, 30.0, 40.0],
            [0.0, 10.0, -9999.0, 30.0, 40.0],
            no_data_value=-9999.0,
        ).interp(time=[5.0, 35.0], method=kind)
        assert (out.read_array().ravel() == -9999.0).all(), (
            f"{kind} spline should make the whole axis no-data on a single gap"
        )

    def test_linear_keeps_the_gap_local(self):
        """Contrast: `linear` interpolates the clean segments and only gaps the gap-touched one."""
        out = _var(
            [0.0, 10.0, 20.0, 30.0, 40.0],
            [0.0, 10.0, -9999.0, 30.0, 40.0],
            no_data_value=-9999.0,
        ).interp(time=[5.0, 35.0])
        result = out.read_array().ravel()
        assert result[0] == 5.0, f"clean segment [0, 10] interpolates, got {result[0]}"
        assert result[1] == 35.0, f"clean segment [30, 40] interpolates, got {result[1]}"


class TestInterpAxisTooShort:
    """A source axis shorter than the kind needs is refused with a friendly message (review L1, L2)."""

    def test_single_point_axis_is_refused(self):
        """A one-step axis cannot be interpolated; refuse instead of returning NaN + a scipy warning."""
        with pytest.raises(ValueError, match="at least 2 source steps"):
            _var([5.0], [5.0], no_data_value=None).interp(time=[5.0])

    def test_cubic_on_three_points_is_refused(self):
        """`cubic` needs four steps; a three-step axis is refused, not left to raw scipy."""
        with pytest.raises(ValueError, match="'cubic' needs at least 4 source steps"):
            _var([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], no_data_value=None).interp(
                time=[1.5], method="cubic"
            )

    def test_quadratic_on_two_points_is_refused(self):
        """`quadratic` needs three steps; a two-step axis is refused."""
        with pytest.raises(ValueError, match="'quadratic' needs at least 3 source steps"):
            _var([0.0, 1.0], [0.0, 1.0], no_data_value=None).interp(
                time=[0.5], method="quadratic"
            )
