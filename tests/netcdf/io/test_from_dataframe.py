"""Tests for `NetCDF.from_dataframe` — rebuilding a cube from a `MultiIndex` DataFrame.

`from_dataframe` is the inverse of `to_dataframe`: it reads the dimensions off the index (the
two innermost levels are the `(y, x)` grid, any outer levels are band dimensions), rebuilds the
array per column, and infers the geotransform from the `x` / `y` cell centres. The round trip
`from_dataframe(nc.to_dataframe(), crs=nc.epsg)` must reproduce `nc`.

Style: Google-style docstrings, <=120 char lines, no inline imports, descriptive assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

GEO_2D = (10.0, 2.0, 0.0, 20.0, 0.0, -2.0)
GEO_CUBE = (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)


def _cube(values: np.ndarray, *, name: str = "t", stamps=(0.0, 6.0)) -> NetCDF:
    """A two-step in-memory cube over a 2x2 grid.

    Args:
        values: The `(time, y, x)` cells.
        name: The variable name.
        stamps: The `time` coordinate values.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        values,
        geo_ref=GeoReference(geo=GEO_CUBE, epsg=4326),
        variable_name=name,
        dims=ExtraDimensions(name="time", values=list(stamps)),
    )


class TestRoundTripReproducesTheCube:
    """`from_dataframe(nc.to_dataframe(), crs=nc.epsg)` reproduces the source cube."""

    def test_a_three_dim_cube_round_trips(self):
        """Values, band names, band coordinates, geotransform and CRS all come back."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        back = NetCDF.from_dataframe(nc.to_dataframe(), crs=nc.epsg)
        var = back.get_variable("t")
        assert var._band_dim_names == ("time",)
        assert var._band_dim_values_map["time"] == [0.0, 6.0]
        assert_array_equal(var.read_array(), nc.get_variable("t").read_array())
        assert_allclose(var.geotransform, GEO_CUBE)
        assert back.epsg == 4326

    def test_a_two_dim_raster_round_trips(self):
        """A `[y, x]` frame rebuilds a single-band raster with the right grid."""
        raster = NetCDF.from_array(
            np.arange(6.0).reshape(2, 3),
            geo_ref=GeoReference(geo=GEO_2D, epsg=4326),
            variable_name="z",
        )
        back = NetCDF.from_dataframe(raster.to_dataframe(), crs=4326)
        var = back.get_variable("z")
        assert var._band_dim_names == ()
        assert_array_equal(var.read_array(), raster.get_variable("z").read_array())
        assert_allclose(var.geotransform, GEO_2D)

    def test_two_columns_rebuild_two_variables(self):
        """Every non-index column becomes a data variable on the shared grid and dims."""
        first = _cube(np.arange(8.0).reshape(2, 2, 2), name="a")
        second = _cube(np.arange(8.0).reshape(2, 2, 2) * 10.0, name="b")
        frame = first.to_dataframe().join(second.to_dataframe())
        back = NetCDF.from_dataframe(frame, crs=4326)
        assert sorted(back.variable_names) == ["a", "b"]
        assert_array_equal(
            back.get_variable("b").read_array(),
            second.get_variable("b").read_array(),
        )

    def test_variables_selects_a_subset_of_columns(self):
        """`variables=` keeps only the named columns as variables."""
        first = _cube(np.arange(8.0).reshape(2, 2, 2), name="a")
        second = _cube(np.arange(8.0).reshape(2, 2, 2) * 10.0, name="b")
        frame = first.to_dataframe().join(second.to_dataframe())
        back = NetCDF.from_dataframe(frame, crs=4326, variables="a")
        assert back.variable_names == ["a"]

    def test_variables_as_a_list_keeps_those_columns(self):
        """A list of names selects exactly those columns as variables."""
        first = _cube(np.arange(8.0).reshape(2, 2, 2), name="a")
        second = _cube(np.arange(8.0).reshape(2, 2, 2) * 10.0, name="b")
        frame = first.to_dataframe().join(second.to_dataframe())
        back = NetCDF.from_dataframe(frame, crs=4326, variables=["a", "b"])
        assert sorted(back.variable_names) == ["a", "b"], (
            f"expected both listed columns as variables, got {back.variable_names}"
        )

    def test_a_written_store_round_trips(self, tmp_path):
        """With `path=`, the cube is written to netCDF and read back with its coordinates.

        Args:
            tmp_path: pytest's temporary directory.
        """
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        out = tmp_path / "from_df.nc"
        back = NetCDF.from_dataframe(nc.to_dataframe(), crs=nc.epsg, path=str(out))
        assert out.exists()
        assert back.get_variable("t")._band_dim_values_map["time"] == [0.0, 6.0]


class TestOrientationIsAlwaysNorthUp:
    """The result is north-up whatever order the frame's y level runs in."""

    def test_an_ascending_y_frame_gives_the_same_raster_as_a_descending_one(self):
        """Sorting the frame's rows south-first does not flip the rebuilt raster."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        frame = nc.to_dataframe()
        shuffled = frame.sort_index(level="y", ascending=True)
        north_up = NetCDF.from_dataframe(frame, crs=4326).get_variable("t").read_array()
        from_shuffled = (
            NetCDF.from_dataframe(shuffled, crs=4326).get_variable("t").read_array()
        )
        assert_array_equal(north_up, from_shuffled)
        assert_array_equal(north_up, nc.get_variable("t").read_array())


class TestGapsSurviveTheRoundTrip:
    """A `NaN` cell rebuilds as a gap that reads back as `NaN`."""

    def test_a_nan_cell_reads_back_as_a_gap(self):
        """A frame with a `NaN` value rebuilds a raster whose gap is `NaN` in `to_dataframe`."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        frame = nc.to_dataframe()
        frame.iloc[0] = np.nan
        back = NetCDF.from_dataframe(frame, crs=4326)
        assert np.isnan(back.to_dataframe()["t"].to_numpy()).sum() == 1

    def test_a_missing_row_becomes_a_gap(self):
        """A cell absent from the frame is reindexed in as a `NaN` gap, not an error."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        frame = nc.to_dataframe().iloc[1:]
        back = NetCDF.from_dataframe(frame, crs=4326)
        assert np.isnan(back.to_dataframe()["t"].to_numpy()).sum() == 1


class TestGeoreferencingRules:
    """`crs=` supplies the CRS; without it the CRS is left unset."""

    def test_crs_none_leaves_the_crs_unset(self):
        """A DataFrame carries no CRS, so `crs=None` builds a store with no EPSG."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        back = NetCDF.from_dataframe(nc.to_dataframe())
        assert back.epsg is None

    def test_crs_given_sets_the_crs(self):
        """An explicit `crs=` is applied to the result."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        back = NetCDF.from_dataframe(nc.to_dataframe(), crs=3857)
        assert back.epsg == 3857

    def test_named_axes_select_non_default_levels(self):
        """`x=` / `y=` name the grid levels when they are not the innermost two."""
        idx = pd.MultiIndex.from_product(
            [[20.0, 18.0], [0.0, 1.0], [0.0, 6.0]], names=["y", "x", "time"]
        )
        frame = pd.DataFrame({"v": np.arange(8.0)}, index=idx)
        back = NetCDF.from_dataframe(frame, crs=4326, x="x", y="y")
        assert back.get_variable("v")._band_dim_names == ("time",)


class TestFromDataframeRefusals:
    """Frames that cannot honestly become a georeferenced cube are refused."""

    def test_an_irregular_grid_is_refused(self):
        """A jittered spatial axis has no affine transform, so it raises naming the axis."""
        idx = pd.MultiIndex.from_product(
            [[20.0, 18.0, 10.0], [0.0, 1.0]], names=["y", "x"]
        )
        frame = pd.DataFrame({"v": np.arange(6.0)}, index=idx)
        with pytest.raises(ValueError, match="regular y axis"):
            NetCDF.from_dataframe(frame)

    def test_a_single_cell_axis_is_refused(self):
        """One coordinate on an axis gives no spacing to infer, so it raises."""
        idx = pd.MultiIndex.from_product([[20.0], [0.0, 1.0]], names=["y", "x"])
        frame = pd.DataFrame({"v": np.arange(2.0)}, index=idx)
        with pytest.raises(ValueError, match="single y coordinate"):
            NetCDF.from_dataframe(frame)

    def test_duplicate_index_rows_are_refused(self):
        """More than one value for a cell is ambiguous, so it raises."""
        idx = pd.MultiIndex.from_tuples(
            [(20.0, 0.0), (20.0, 0.0), (18.0, 0.0), (18.0, 1.0)], names=["y", "x"]
        )
        frame = pd.DataFrame({"v": np.arange(4.0)}, index=idx)
        with pytest.raises(ValueError, match="duplicate index rows"):
            NetCDF.from_dataframe(frame)

    def test_a_tidy_frame_is_refused(self):
        """A plain `RangeIndex` frame is scattered rows, not a grid, so it raises."""
        frame = pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0], "v": [3.0, 4.0]})
        with pytest.raises(ValueError, match="indexed by its dimensions"):
            NetCDF.from_dataframe(frame)

    def test_an_unknown_variable_is_refused(self):
        """A requested column that is not in the frame raises naming the available ones."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        with pytest.raises(ValueError, match="nope"):
            NetCDF.from_dataframe(nc.to_dataframe(), variables="nope")

    def test_a_missing_named_axis_is_refused(self):
        """`x=` naming a level the index does not have raises."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        with pytest.raises(ValueError, match="not an\\s+index level"):
            NetCDF.from_dataframe(nc.to_dataframe(), x="lon")

    def test_a_one_level_multiindex_is_refused(self):
        """A one-level MultiIndex cannot hold both grid axes, so it raises."""
        idx = pd.MultiIndex.from_arrays([[0.0, 1.0]], names=["x"])
        frame = pd.DataFrame({"v": [3.0, 4.0]}, index=idx)
        with pytest.raises(ValueError, match="indexed by its dimensions"):
            NetCDF.from_dataframe(frame)

    def test_an_unnamed_index_level_is_refused(self):
        """An unnamed index level cannot be addressed as an axis, so it raises."""
        idx = pd.MultiIndex.from_product([[20.0, 18.0], [0.0, 1.0]], names=["y", None])
        frame = pd.DataFrame({"v": np.arange(4.0)}, index=idx)
        with pytest.raises(ValueError, match="every index level named"):
            NetCDF.from_dataframe(frame)

    def test_x_and_y_naming_the_same_level_is_refused(self):
        """Pointing both the y and x axes at one level collapses the grid, so it raises."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        with pytest.raises(ValueError, match="both the y and x axes"):
            NetCDF.from_dataframe(nc.to_dataframe(), x="y", y="y")

    def test_a_frame_with_no_value_columns_is_refused(self):
        """An index-only frame has nothing to become a data variable, so it raises."""
        idx = pd.MultiIndex.from_product([[20.0, 18.0], [0.0, 1.0]], names=["y", "x"])
        frame = pd.DataFrame(index=idx)
        with pytest.raises(ValueError, match="at least one value column"):
            NetCDF.from_dataframe(frame)

    def test_a_repeated_variable_is_refused(self):
        """Asking for the same column twice would build one variable twice, so it raises."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        with pytest.raises(ValueError, match="more than once"):
            NetCDF.from_dataframe(nc.to_dataframe(), variables=["t", "t"])

    def test_an_empty_variable_selection_is_refused(self):
        """An empty `variables=` selects no column, so no cube can be built and it raises."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        with pytest.raises(ValueError, match="empty selection"):
            NetCDF.from_dataframe(nc.to_dataframe(), variables=[])

    def test_an_irregular_x_axis_is_refused(self):
        """A jittered x axis has no affine transform, so it raises naming the x axis."""
        idx = pd.MultiIndex.from_product(
            [[20.0, 18.0], [0.0, 1.0, 3.0]], names=["y", "x"]
        )
        frame = pd.DataFrame({"v": np.arange(6.0)}, index=idx)
        with pytest.raises(ValueError, match="regular x axis"):
            NetCDF.from_dataframe(frame)

    def test_a_single_cell_x_axis_is_refused(self):
        """One x coordinate gives no spacing to infer, so it raises naming the x axis."""
        idx = pd.MultiIndex.from_product([[20.0, 18.0], [0.0]], names=["y", "x"])
        frame = pd.DataFrame({"v": np.arange(2.0)}, index=idx)
        with pytest.raises(ValueError, match="single x coordinate"):
            NetCDF.from_dataframe(frame)


def _two_band_cube() -> NetCDF:
    """A `(time=2, level=3, y=2, x=2)` cube with distinct cells, for the 4-D reshape path.

    Returns:
        NetCDF: The container, variable `t`, over `GEO_CUBE`.
    """
    return NetCDF.from_array(
        np.arange(2 * 3 * 2 * 2, dtype="float64").reshape(2, 3, 2, 2),
        geo_ref=GeoReference(geo=GEO_CUBE, epsg=4326),
        variable_name="t",
        dims=ExtraDimensions(
            dims=[("time", [0.0, 6.0]), ("level", [1000.0, 850.0, 500.0])]
        ),
    )


class TestReshapeAndReorderAreValueVerified:
    """The hardest paths — multi-band, reordered axes, odd coordinates — check placement."""

    def test_a_four_dim_two_band_cube_round_trips(self):
        """A `(time, level, y, x)` cube reproduces its cells, band names, and coordinates."""
        nc = _two_band_cube()
        back = NetCDF.from_dataframe(nc.to_dataframe(), crs=nc.epsg)
        var = back.get_variable("t")
        assert var._band_dim_names == ("time", "level")
        assert var._band_dim_values_map["time"] == [0.0, 6.0]
        assert var._band_dim_values_map["level"] == [1000.0, 850.0, 500.0]
        assert_array_equal(var.read_array(), nc.get_variable("t").read_array())

    def test_non_innermost_named_axes_place_values_correctly(self):
        """With `x=` / `y=` naming non-innermost levels, the cells land where they belong."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        reordered = nc.to_dataframe().reorder_levels(["y", "x", "time"])
        back = NetCDF.from_dataframe(reordered, crs=4326, x="x", y="y")
        assert back.get_variable("t")._band_dim_names == ("time",)
        assert_array_equal(
            back.get_variable("t").read_array(),
            nc.get_variable("t").read_array(),
        )

    def test_descending_band_stamps_keep_their_order(self):
        """A band axis stored descending comes back in the same order, not sorted."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2), stamps=(18.0, 6.0))
        back = NetCDF.from_dataframe(nc.to_dataframe(), crs=4326)
        assert back.get_variable("t")._band_dim_values_map["time"] == [18.0, 6.0]
        assert_array_equal(
            back.get_variable("t").read_array(), nc.get_variable("t").read_array()
        )

    def test_an_all_nan_column_becomes_an_all_gap_variable(self):
        """A column that is entirely `NaN` rebuilds a variable whose every cell is a gap."""
        nc = _cube(np.arange(8.0).reshape(2, 2, 2))
        frame = nc.to_dataframe()
        frame["t"] = np.nan
        back = NetCDF.from_dataframe(frame, crs=4326)
        assert np.isnan(back.to_dataframe()["t"].to_numpy()).all()

    def test_a_text_band_axis_round_trips(self):
        """A non-numeric band axis keeps its labels through the round trip."""
        nc = NetCDF.from_array(
            np.arange(8.0).reshape(2, 2, 2),
            geo_ref=GeoReference(geo=GEO_CUBE, epsg=4326),
            variable_name="t",
            dims=ExtraDimensions(name="scenario", values=["rcp45", "rcp85"]),
        )
        back = NetCDF.from_dataframe(nc.to_dataframe(), crs=4326)
        assert list(back.get_variable("t")._band_dim_values_map["scenario"]) == [
            "rcp45",
            "rcp85",
        ]
        assert_array_equal(
            back.get_variable("t").read_array(), nc.get_variable("t").read_array()
        )


class TestNonStandardColumns:
    """Value columns whose labels are not strings, or are non-numeric / duplicated."""

    def test_an_integer_column_label_becomes_a_string_variable(self):
        """A non-string column label indexes the frame and is stringified for the variable.

        The frame is read by the real (integer) label; only the NetCDF variable name is
        stringified. Before the fix this raised `KeyError: '7'` (#1203 L1).
        """
        idx = pd.MultiIndex.from_product(
            [[0.0, 6.0], [20.0, 19.0], [0.0, 1.0]], names=["time", "y", "x"]
        )
        frame = pd.DataFrame({7: np.arange(8.0)}, index=idx)
        back = NetCDF.from_dataframe(frame, crs=4326)
        assert back.variable_names == ["7"]
        assert_array_equal(
            back.get_variable("7").read_array(), frame[7].to_numpy().reshape(2, 2, 2)
        )

    def test_an_integer_variable_selector_is_honoured(self):
        """`variables=` given a non-string label is matched against the real column labels."""
        idx = pd.MultiIndex.from_product([[20.0, 19.0], [0.0, 1.0]], names=["y", "x"])
        frame = pd.DataFrame({7: np.arange(4.0), 8: np.arange(4.0)}, index=idx)
        back = NetCDF.from_dataframe(frame, crs=4326, variables=7)
        assert back.variable_names == ["7"]

    def test_a_non_numeric_column_is_refused_by_name(self):
        """A text value column raises a from_dataframe() message naming the column (#1203 L2)."""
        idx = pd.MultiIndex.from_product([[20.0, 19.0], [0.0, 1.0]], names=["y", "x"])
        frame = pd.DataFrame({"v": ["a", "b", "c", "d"]}, index=idx)
        with pytest.raises(ValueError, match="from_dataframe.*'v'.*numeric"):
            NetCDF.from_dataframe(frame)

    def test_duplicate_column_labels_are_refused_by_name(self):
        """Two columns of one name cannot become one variable, so it raises naming it (#1203 L2)."""
        idx = pd.MultiIndex.from_product([[20.0, 19.0], [0.0, 1.0]], names=["y", "x"])
        frame = pd.DataFrame(
            np.arange(8.0).reshape(4, 2), index=idx, columns=["v", "v"]
        )
        with pytest.raises(
            ValueError, match="from_dataframe.*more than one column labelled 'v'"
        ):
            NetCDF.from_dataframe(frame)
