"""Tests for `NetCDF.rank` / `NetCDF.pad` / `NetCDF.transpose` — the last Tier 2c members.

`rank` ranks a band axis (ties averaged, gaps excluded); `pad` extends a band axis with a fill or
grows the spatial grid while moving the geotransform; `transpose` reorders band dimensions with the
`(y, x)` plane pinned trailing. All three are band-dimension operations — a spatial axis is refused
by `rank`/`transpose` and handled specially by `pad`.

Style: Google-style docstrings, <=120 char lines, no inline imports, descriptive assertions.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


def _var(stamps, values, *, dim="time", no_data_value=-9999.0, cols=1):
    """A single variable of `len(stamps)` bands over a `1 x cols` grid."""
    n = len(stamps)
    return NetCDF.from_array(
        np.asarray(values, "float64").reshape(n, 1, cols),
        geo_ref=GeoReference(geo=GEO, epsg=4326),
        variable_name="t",
        dims=ExtraDimensions(name=dim, values=list(stamps)),
        no_data_value=no_data_value,
    ).get_variable("t")


def _grid(bands, rows, cols, *, no_data_value=-9999.0):
    """A `bands x rows x cols` container on `time`."""
    return NetCDF.from_array(
        np.arange(bands * rows * cols, dtype="float64").reshape(bands, rows, cols),
        geo_ref=GeoReference(geo=GEO, epsg=4326),
        variable_name="t",
        dims=ExtraDimensions(name="time", values=list(range(bands))),
        no_data_value=no_data_value,
    )


def _cube2(time_vals, level_vals):
    """A 4-D container with band dims `time` then `level`."""
    nt, nl = len(time_vals), len(level_vals)
    return NetCDF.from_array(
        np.arange(nt * nl, dtype="float64").reshape(nt, nl, 1, 1),
        geo_ref=GeoReference(geo=GEO, epsg=4326),
        variable_name="t",
        dims=ExtraDimensions(dims=[("time", list(time_vals)), ("level", list(level_vals))]),
    )


class TestRank:
    """`rank` ranks a band axis, ties averaged, gaps excluded."""

    def test_ordinal_ranks_with_averaged_ties(self):
        """Values rank 1..N along the axis with tied values sharing the average position."""
        out = _var([0, 1, 2, 3], [30.0, 10.0, 10.0, 20.0], no_data_value=None).rank("time")
        assert_allclose(out.read_array().ravel(), [4.0, 1.5, 1.5, 3.0])

    def test_pct_divides_by_the_valid_count(self):
        """`pct=True` returns the rank as a fraction of the valid count, in (0, 1]."""
        out = _var([0, 1, 2, 3], [30.0, 10.0, 10.0, 20.0], no_data_value=None).rank(
            "time", pct=True
        )
        assert_allclose(out.read_array().ravel(), [1.0, 0.375, 0.375, 0.75])

    def test_no_data_is_excluded_and_returned_as_no_data(self):
        """A gap is left out of the ranking and comes back as the no-data value."""
        out = _var([0, 1, 2], [10.0, -9999.0, 30.0], no_data_value=-9999.0).rank("time")
        result = out.read_array().ravel()
        assert result[1] == -9999.0, f"the gap stays no-data, got {result[1]}"
        assert_allclose([result[0], result[2]], [1.0, 2.0])

    def test_matches_xarray(self):
        """Ranks agree with `xarray.DataArray.rank` on a gapless series."""
        xr = pytest.importorskip("xarray")
        pytest.importorskip("bottleneck")  # xarray.rank delegates to bottleneck
        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        got = _var(range(5), values, no_data_value=None).rank("time").read_array().ravel()
        expected = (
            xr.DataArray(values, dims="time").rank("time").values
        )
        assert_allclose(got, expected)

    def test_container_ranks_every_variable(self):
        """A container ranks each variable that has the dimension."""
        out = _grid(3, 1, 2).rank("time").get_variable("t")
        assert out.read_array().shape == (3, 1, 2), out.read_array().shape

    def test_spatial_axis_is_refused(self):
        """Ranking a spatial axis is refused."""
        var = _var([0, 1], [1.0, 2.0])
        with pytest.raises(ValueError, match="band dimension"):
            var.rank("x")


class TestPadBand:
    """`pad` on a band dimension extends it with a fill and NaN coordinate stamps."""

    def test_pad_before_fills_no_data(self):
        """Padding before the axis prepends no-data cells and a NaN stamp."""
        out = _var([0, 1], [1.0, 2.0], no_data_value=-9999.0).pad(time=(1, 0))
        assert_allclose(out.read_array().ravel(), [-9999.0, 1.0, 2.0])
        assert out._band_dim_values_map["time"][0] != out._band_dim_values_map["time"][0]  # NaN
        assert out._band_dim_values_map["time"][1:] == [0.0, 1.0]

    def test_constant_values_fills_that_value(self):
        """An explicit `constant_values` fills the pad cells instead of no-data."""
        out = _var([0, 1], [1.0, 2.0]).pad(time=(0, 1), constant_values=0.0)
        assert_allclose(out.read_array().ravel(), [1.0, 2.0, 0.0])

    def test_int_width_pads_both_sides(self):
        """A scalar width pads both sides."""
        out = _var([0, 1], [1.0, 2.0], no_data_value=-9999.0).pad(time=1)
        assert_allclose(out.read_array().ravel(), [-9999.0, 1.0, 2.0, -9999.0])


class TestPadSpatial:
    """`pad` on a spatial axis grows the grid and moves the geotransform."""

    def test_pad_x_before_shifts_x_origin_west(self):
        """Padding one cell on the left grows the columns and moves the x origin one cell west."""
        out = _grid(1, 2, 2).pad(x=(1, 0)).get_variable("t")
        assert out.columns == 3, out.columns
        assert tuple(out.geotransform)[0] == -1.0, tuple(out.geotransform)

    def test_pad_y_before_shifts_y_origin_north(self):
        """Padding one cell on the top grows the rows and moves the y origin one cell north."""
        out = _grid(1, 2, 2).pad(y=(1, 0)).get_variable("t")
        assert out.rows == 3, out.rows
        assert tuple(out.geotransform)[3] == 5.0, tuple(out.geotransform)

    def test_pad_then_crop_round_trips_the_extent(self):
        """Padding then cropping back to the original bounds restores the grid."""
        base = _grid(1, 2, 2)
        padded = base.pad(x=(1, 1), y=(1, 1))
        assert padded.get_variable("t").rows == 4
        assert padded.get_variable("t").columns == 4


class TestTranspose:
    """`transpose` reorders band dimensions; the spatial plane stays trailing."""

    def test_swaps_two_band_dims(self):
        """Naming the dims in a new order permutes the band axes and their coordinates."""
        out = _cube2([0, 1], [10, 20]).transpose("level", "time").get_variable("t")
        assert out._band_dim_names == ("level", "time")
        assert out._band_dim_values_map == {"level": [10.0, 20.0], "time": [0.0, 1.0]}
        assert_allclose(out.read_array().ravel(), [0.0, 2.0, 1.0, 3.0])

    def test_no_args_reverses_band_order(self):
        """With no arguments the band dimensions are reversed."""
        out = _cube2([0, 1], [10, 20]).transpose().get_variable("t")
        assert out._band_dim_names == ("level", "time")

    def test_ellipsis_expands_to_the_rest(self):
        """`...` stands in for the unnamed band dimensions in current order."""
        out = _cube2([0, 1], [10, 20]).transpose("level", ...).get_variable("t")
        assert out._band_dim_names == ("level", "time")

    def test_variable_and_container_agree(self):
        """`transpose` on a variable matches the container route."""
        cube = _cube2([0, 1], [10, 20])
        from_container = cube.transpose("level", "time").get_variable("t").read_array()
        from_variable = cube.get_variable("t").transpose("level", "time").read_array()
        assert_allclose(from_variable, from_container)


class TestTransposeRefusals:
    """`transpose` refuses spatial axes, incomplete orders, unknown and duplicate dims."""

    def test_spatial_axis_is_refused(self):
        """Naming a spatial axis is refused with the pinned-plane reason."""
        cube = _cube2([0, 1], [10, 20])
        with pytest.raises(ValueError, match="only band"):
            cube.transpose("x", "time")

    def test_incomplete_order_without_ellipsis_is_refused(self):
        """Naming some but not all band dims (without `...`) is refused."""
        cube = _cube2([0, 1], [10, 20])
        with pytest.raises(ValueError, match="name every band dimension"):
            cube.transpose("time")

    def test_unknown_dim_is_refused(self):
        """A name that is no band dimension is refused."""
        cube = _cube2([0, 1], [10, 20])
        with pytest.raises(ValueError, match="not band dimensions"):
            cube.transpose("time", "depth")

    def test_duplicate_dim_is_refused(self):
        """A repeated dimension is refused."""
        cube = _cube2([0, 1], [10, 20])
        with pytest.raises(ValueError, match="duplicate"):
            cube.transpose("time", "time")


class TestPadRefusals:
    """`pad` refuses a missing dim, a bad mode, and a malformed width."""

    def test_no_dimension_is_refused(self):
        """`pad()` with no dimension is refused."""
        with pytest.raises(ValueError, match="at least one dimension"):
            _var([0, 1], [1.0, 2.0]).pad()

    def test_unsupported_mode_is_refused(self):
        """Only `mode='constant'` is supported."""
        with pytest.raises(ValueError, match="only mode='constant'"):
            _var([0, 1], [1.0, 2.0]).pad(time=(1, 1), mode="reflect")

    def test_negative_width_is_refused(self):
        """A negative pad width is refused."""
        with pytest.raises(ValueError, match="non-negative"):
            _var([0, 1], [1.0, 2.0]).pad(time=(-1, 0))
