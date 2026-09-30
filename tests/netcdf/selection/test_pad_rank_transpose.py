"""Tests for `NetCDF.rank` / `NetCDF.pad` / `NetCDF.transpose` — the last Tier 2c members.

`rank` ranks a band axis (ties averaged, gaps excluded); `pad` extends a band axis with a fill or
grows the spatial grid while moving the geotransform; `transpose` reorders band dimensions with the
`(y, x)` plane pinned trailing. All three are band-dimension operations — a spatial axis is refused
by `rank`/`transpose` and handled specially by `pad`.

Style: Google-style docstrings, <=120 char lines, no inline imports, descriptive assertions.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines.selection import _transpose_order

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)

ERA5_T2M = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-3d1__geog__y-desc.nc"
)


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
        dims=ExtraDimensions(
            dims=[("time", list(time_vals)), ("level", list(level_vals))]
        ),
    )


def _cube3(time_vals, level_vals, depth_vals):
    """A 5-D container with band dims `time`, `level`, `depth`, filled `arange`."""
    nt, nl, nd = len(time_vals), len(level_vals), len(depth_vals)
    return NetCDF.from_array(
        np.arange(nt * nl * nd, dtype="float64").reshape(nt, nl, nd, 1, 1),
        geo_ref=GeoReference(geo=GEO, epsg=4326),
        variable_name="t",
        dims=ExtraDimensions(
            dims=[
                ("time", list(time_vals)),
                ("level", list(level_vals)),
                ("depth", list(depth_vals)),
            ]
        ),
    )


class TestRank:
    """`rank` ranks a band axis, ties averaged, gaps excluded."""

    def test_ordinal_ranks_with_averaged_ties(self):
        """Values rank 1..N along the axis with tied values sharing the average position."""
        out = _var([0, 1, 2, 3], [30.0, 10.0, 10.0, 20.0], no_data_value=None).rank(
            "time"
        )
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

    def test_container_ranks_every_variable(self):
        """A container ranks each variable that has the dimension."""
        out = _grid(3, 1, 2).rank("time").get_variable("t")
        assert out.read_array().shape == (3, 1, 2), out.read_array().shape

    def test_spatial_axis_is_refused(self):
        """Ranking a spatial axis is refused."""
        var = _var([0, 1], [1.0, 2.0])
        with pytest.raises(ValueError, match="band dimension"):
            var.rank("x")

    def test_container_and_single_variable_agree(self):
        """`rank` on a container matches ranking its extracted variable directly."""
        grid = _grid(3, 1, 2)
        from_container = grid.rank("time").get_variable("t").read_array()
        from_variable = grid.get_variable("t").rank("time").read_array()
        assert_allclose(from_variable, from_container)

    def test_pct_all_gap_pixel_returns_no_data_without_a_warning(self):
        """An all-gap pixel dodges the divide-by-zero guard: it returns no-data and warns nothing."""
        var = _var(
            [0, 1, 2],
            [-9999.0, 10.0, -9999.0, 30.0, -9999.0, 20.0],
            no_data_value=-9999.0,
            cols=2,
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = var.rank("time", pct=True).read_array().reshape(3, 2)
        assert not any("divide" in str(w.message) for w in record), (
            f"the all-gap guard must not emit a divide warning, got {[str(w.message) for w in record]}"
        )
        assert np.all(result[:, 0] == -9999.0), (
            f"the all-gap column must stay no-data, got {result[:, 0]}"
        )
        assert_allclose(
            result[:, 1],
            [1.0 / 3.0, 1.0, 2.0 / 3.0],
            err_msg="the valid column must rank as fractions",
        )


class TestPadBand:
    """`pad` on a band dimension extends it with a fill and NaN coordinate stamps."""

    def test_pad_before_fills_no_data(self):
        """Padding before the axis prepends no-data cells and a NaN stamp."""
        out = _var([0, 1], [1.0, 2.0], no_data_value=-9999.0).pad(time=(1, 0))
        assert_allclose(out.read_array().ravel(), [-9999.0, 1.0, 2.0])
        assert np.isnan(out._band_dim_values_map["time"][0]), (
            "the prepended stamp must be NaN"
        )
        assert out._band_dim_values_map["time"][1:] == [0.0, 1.0]

    def test_constant_values_fills_that_value(self):
        """An explicit `constant_values` fills the pad cells instead of no-data."""
        out = _var([0, 1], [1.0, 2.0]).pad(time=(0, 1), constant_values=0.0)
        assert_allclose(out.read_array().ravel(), [1.0, 2.0, 0.0])

    def test_int_width_pads_both_sides(self):
        """A scalar width pads both sides."""
        out = _var([0, 1], [1.0, 2.0], no_data_value=-9999.0).pad(time=1)
        assert_allclose(out.read_array().ravel(), [-9999.0, 1.0, 2.0, -9999.0])

    def test_before_and_after_in_one_call_with_distinct_widths(self):
        """A `(before, after)` pair with different widths pads each side and stamps each pad NaN."""
        out = _var([0, 1], [1.0, 2.0], no_data_value=-9999.0).pad(time=(2, 1))
        assert_allclose(out.read_array().ravel(), [-9999.0, -9999.0, 1.0, 2.0, -9999.0])
        stamps = out._band_dim_values_map["time"]
        assert stamps[2:4] == [0.0, 1.0], (
            f"the original stamps must survive in place, got {stamps}"
        )
        assert all(np.isnan(s) for s in stamps[:2] + stamps[4:]), (
            f"every pad stamp must be NaN, got {stamps}"
        )

    def test_no_data_none_variable_pads_with_nan(self):
        """A variable declaring no no-data value pads its band cells with NaN."""
        out = _var([0, 1], [1.0, 2.0], no_data_value=None).pad(time=(1, 0))
        result = out.read_array().ravel()
        assert np.isnan(result[0]), (
            f"the pad cell must be NaN when no no-data is declared, got {result[0]}"
        )
        assert_allclose(result[1:], [1.0, 2.0])

    def test_container_band_pad_extends_every_variable(self):
        """`pad` on a container band axis grows the dimension of each gridded variable."""
        out = _grid(2, 1, 2, no_data_value=-9999.0).pad(time=(1, 0)).get_variable("t")
        assert out.read_array().shape == (3, 1, 2), out.read_array().shape
        assert_allclose(
            out.read_array().ravel(), [-9999.0, -9999.0, 0.0, 1.0, 2.0, 3.0]
        )

    def test_pad_a_dimension_without_coordinate_stamps_keeps_it_unstamped(self):
        """A band dim carrying no stamps (an operator result) pads its values but stays unlabelled."""
        summed = _var([0, 1], [1.0, 2.0]) + _var([5, 6], [3.0, 4.0])
        assert summed._band_dim_values_map["time"] is None, (
            "the operator result should drop its stamps"
        )
        out = summed.pad(time=(1, 0))
        result = out.read_array().ravel()
        assert np.isnan(result[0]), f"the pad cell must be NaN, got {result[0]}"
        assert_allclose(result[1:], [4.0, 6.0])
        assert out._band_dim_values_map["time"] is None, (
            "an unstamped dimension must stay unstamped after pad"
        )


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

    def test_pad_x_before_fills_the_new_column_with_no_data(self):
        """Padding one cell on the left fills the whole new first column with the no-data value."""
        plane = np.asarray(
            _grid(1, 2, 2, no_data_value=-9999.0)
            .pad(x=(1, 0))
            .get_variable("t")
            .read_array()
        ).reshape(2, 3)
        assert np.all(plane[:, 0] == -9999.0), (
            f"the new left column must hold the no-data fill, got {plane[:, 0]}"
        )
        assert_allclose(plane[:, 1:].ravel(), [0.0, 1.0, 2.0, 3.0])

    def test_pad_y_before_constant_values_fills_the_new_row(self):
        """`constant_values` fills the new top row of a `y` pad instead of the no-data value."""
        plane = np.asarray(
            _grid(1, 2, 2, no_data_value=-9999.0)
            .pad(y=(1, 0), constant_values=7.0)
            .get_variable("t")
            .read_array()
        ).reshape(3, 2)
        assert np.all(plane[0, :] == 7.0), (
            f"the new top row must hold the constant fill, got {plane[0, :]}"
        )
        assert_allclose(plane[1:, :].ravel(), [0.0, 1.0, 2.0, 3.0])

    def test_pad_spatial_fills_nan_when_no_no_data_is_declared(self):
        """With no declared no-data value the new spatial cells are filled with NaN."""
        plane = np.asarray(
            _grid(1, 2, 2, no_data_value=None)
            .pad(x=(1, 0))
            .get_variable("t")
            .read_array()
        ).reshape(2, 3)
        assert np.all(np.isnan(plane[:, 0])), (
            f"the new column must be NaN with no no-data declared, got {plane[:, 0]}"
        )

    def test_pad_grows_the_grid_and_shifts_the_origin_with_data(self):
        """A one-cell border on every side grows a 2x2 grid to 4x4, moves the origin north-west,
        keeps the original cells in the interior, and fills the new border with no-data.

        This is the invariant a pad/crop round-trip would protect: an origin shifted the wrong way
        (east/south instead of west/north) or data placed at the wrong corner fails here.
        """
        base = _grid(1, 2, 2, no_data_value=-9999.0)
        base_plane = np.asarray(base.get_variable("t").read_array()).reshape(2, 2)
        padded = base.pad(x=(1, 1), y=(1, 1)).get_variable("t")
        assert (padded.rows, padded.columns) == (4, 4), (padded.rows, padded.columns)
        gt = tuple(padded.geotransform)
        assert gt[0] == -1.0, f"x origin should shift one cell west, got {gt[0]}"
        assert gt[3] == 5.0, f"y origin should shift one cell north, got {gt[3]}"
        assert (gt[1], gt[5]) == (1.0, -1.0), f"pixel sizes must be unchanged, got {gt}"
        plane = np.asarray(padded.read_array()).reshape(4, 4)
        assert_allclose(plane[1:3, 1:3].ravel(), base_plane.ravel())
        border = np.ones((4, 4), dtype=bool)
        border[1:3, 1:3] = False
        assert np.all(plane[border] == -9999.0), (
            "the padded border must hold the no-data fill"
        )


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

    def test_ellipsis_in_the_middle_expands_to_the_unnamed_dims(self):
        """`...` between two named dims stands in for the band dimensions left unnamed, in order."""
        out = (
            _cube3([0, 1], [10, 20], [100, 200])
            .transpose("depth", ..., "time")
            .get_variable("t")
        )
        assert out._band_dim_names == ("depth", "level", "time"), out._band_dim_names
        source = np.arange(8.0).reshape(2, 2, 2, 1, 1)
        assert_allclose(
            out.read_array().ravel(), np.transpose(source, (2, 1, 0, 3, 4)).ravel()
        )

    def test_single_band_transpose_is_a_no_op(self):
        """`transpose()` on a one-band-dimension variable leaves its axis and values untouched."""
        out = _var([0, 1], [1.0, 2.0]).transpose()
        assert out._band_dim_names == ("time",), out._band_dim_names
        assert_allclose(out.read_array().ravel(), [1.0, 2.0])

    def test_heterogeneous_container_reorders_each_variable_by_its_own_dims(self):
        """With `...`, a container variable lacking a named dim is reordered by only the dims it has."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cube = _cube2([0, 1], [10, 20])
            other = NetCDF.from_array(
                np.arange(4.0).reshape(2, 2, 1, 1),
                geo_ref=GeoReference(geo=GEO, epsg=4326),
                variable_name="v",
                dims=ExtraDimensions(
                    dims=[("time", [0.0, 1.0]), ("depth", [100.0, 200.0])]
                ),
            )
            cube.set_variable("v", other.get_variable("v"))
            out = cube.transpose("level", ...)
        assert out.get_variable("t")._band_dim_names == ("level", "time"), (
            out.get_variable("t")._band_dim_names
        )
        assert out.get_variable("v")._band_dim_names == ("time", "depth"), (
            out.get_variable("v")._band_dim_names
        )


class TestTransposeOrderHelper:
    """`_transpose_order` builds one variable's new band order for the requested `dims`."""

    def test_ellipsis_skips_a_named_dim_the_variable_lacks(self):
        """An explicit name absent from a variable is skipped while `...` expands to the dims it has."""
        order = _transpose_order(["time", "depth"], ("level", ...))
        assert order == ["time", "depth"], (
            f"the absent 'level' must be skipped, got {order}"
        )

    def test_ellipsis_places_named_dims_around_the_expanded_rest(self):
        """Named dims keep their positions and `...` fills the gap with the unnamed dims, in order."""
        order = _transpose_order(["time", "level", "depth"], ("depth", ..., "time"))
        assert order == ["depth", "level", "time"], (
            f"the middle `...` must expand to 'level', got {order}"
        )

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

    def test_duplicate_ellipsis_is_refused(self):
        """Passing `...` more than once is refused clearly, not with numpy's cryptic axes error."""
        cube = _cube2([0, 1], [10, 20])
        with pytest.raises(ValueError, match="ellipsis"):
            cube.transpose(..., ...)

    def test_non_string_dim_is_refused(self):
        """A dimension that is neither a string nor `...` is refused."""
        cube = _cube2([0, 1], [10, 20])
        with pytest.raises(ValueError, match="strings or"):
            cube.transpose("time", 5)

    def test_empty_container_is_refused(self):
        """A container with no data variables cannot be transposed."""
        container = _grid(2, 1, 2)
        container.remove_variable("t")
        with pytest.raises(ValueError, match="Cannot transpose an empty container"):
            container.transpose()


class TestPadRefusals:
    """`pad` refuses a missing dim, a bad mode, and a malformed width."""

    def test_no_dimension_is_refused(self):
        """`pad()` with no dimension is refused."""
        var = _var([0, 1], [1.0, 2.0])
        with pytest.raises(ValueError, match="at least one dimension"):
            var.pad()

    def test_unsupported_mode_is_refused(self):
        """Only `mode='constant'` is supported."""
        var = _var([0, 1], [1.0, 2.0])
        with pytest.raises(ValueError, match="only mode='constant'"):
            var.pad(time=(1, 1), mode="reflect")

    def test_negative_width_is_refused(self):
        """A negative pad width is refused."""
        var = _var([0, 1], [1.0, 2.0])
        with pytest.raises(ValueError, match="non-negative"):
            var.pad(time=(-1, 0))

    def test_malformed_width_tuple_is_refused(self):
        """A width that is neither a 2-tuple nor an int (a 3-tuple) is refused."""
        var = _var([0, 1], [1.0, 2.0])
        with pytest.raises(ValueError, match=r"\(before, after\)"):
            var.pad(time=(1, 2, 3))


class TestPadContainerAuxiliary:
    """`pad` on a container drops an auxiliary variable spanning the padded band dimension."""

    def test_spanning_auxiliary_is_dropped_with_a_padded_dimension_warning(self):
        """The ERA5 `expver` auxiliary spans `valid_time`, so padding it drops it and warns."""
        container = NetCDF.read_file(str(ERA5_T2M))
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = container.pad(valid_time=(1, 1))
        messages = [str(w.message) for w in record]
        assert any("padded dimension" in m for m in messages), (
            f"the drop warning must name the padded dimension, got {messages}"
        )
        assert "expver" not in out.variable_names, (
            f"the spanning auxiliary must be dropped, got {out.variable_names}"
        )
