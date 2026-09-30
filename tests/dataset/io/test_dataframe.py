"""Tests for `Dataset.to_dataframe` / `Dataset.from_dataframe` — the pandas round trip.

`to_dataframe` hands the raster to pandas as one row per cell on a `(band, y, x)` MultiIndex
with a single `values` column; `from_dataframe` is its inverse, reading the grid off the index
(the two innermost levels are `(y, x)`, an optional outer level is the positional `band` axis),
rebuilding the array, and inferring the geotransform from the `x` / `y` cell centres. The round
trip `Dataset.from_dataframe(ds.to_dataframe(), crs=ds.epsg)` must reproduce `ds`.

The grid-inference and reshape helpers are shared with `NetCDF.from_dataframe`
(`pyramids.base._dataframe_grid`), so the refusal messages match the NetCDF suite; the
raster-only rules (a single value column, at most one band level) are tested here.

Style: Google-style docstrings, <=120 char lines, no inline imports, descriptive assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.core

GEO_2D = (10.0, 2.0, 0.0, 20.0, 0.0, -2.0)
GEO_CUBE = (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)


def _raster(
    values: np.ndarray, *, geo=GEO_CUBE, epsg=4326, no_data_value=-9999.0
) -> Dataset:
    """An in-memory raster over the given grid.

    Args:
        values: The `(y, x)` or `(band, y, x)` cells.
        geo: The affine geotransform.
        epsg: The CRS code.
        no_data_value: The nodata sentinel.

    Returns:
        Dataset: The raster.
    """
    return Dataset.from_array(
        values, geo_ref=GeoReference(geo=geo, epsg=epsg), no_data_value=no_data_value
    )


class TestToDataframeShape:
    """`to_dataframe` produces the `(band, y, x)` / single-`values`-column frame."""

    def test_index_names_and_single_values_column(self):
        """The frame is a `(band, y, x)` MultiIndex with one `values` column."""
        df = _raster(np.arange(8.0).reshape(2, 2, 2)).to_dataframe()
        assert list(df.index.names) == ["band", "y", "x"]
        assert list(df.columns) == ["values"]

    def test_the_band_level_is_kept_for_a_one_band_raster(self):
        """A single-band raster still carries the `band` level, value `0`."""
        df = _raster(np.array([[1.0, 2.0], [3.0, 4.0]])).to_dataframe()
        assert list(df.index.names) == ["band", "y", "x"]
        assert df.index.get_level_values("band").unique().tolist() == [0]

    def test_values_are_float64_from_an_integer_raster(self):
        """Whatever the raster's dtype, the `values` column is `float64`."""
        df = _raster(
            np.arange(4, dtype="int32").reshape(2, 2), no_data_value=None
        ).to_dataframe()
        assert df["values"].dtype == np.float64

    def test_layout_is_north_up_row_major(self):
        """Values are laid out band, then y descending, then x ascending."""
        df = _raster(np.array([[1.0, 2.0], [3.0, 4.0]])).to_dataframe()
        assert df["values"].tolist() == [1.0, 2.0, 3.0, 4.0]
        assert df.index.get_level_values("y").tolist() == [1.5, 1.5, 0.5, 0.5]
        assert df.index.get_level_values("x").tolist() == [0.5, 1.5, 0.5, 1.5]

    def test_nodata_cells_become_nan(self):
        """A cell holding the sentinel is reported as `NaN`, not the sentinel."""
        arr = np.array([[-9999.0, 2.0], [3.0, 4.0]])
        df = _raster(arr).to_dataframe()
        assert bool(np.isnan(df["values"].iloc[0]))
        assert df["values"].iloc[1:].tolist() == [2.0, 3.0, 4.0]

    def test_a_nan_sentinel_reads_gaps_without_extra_masking(self):
        """A raster whose nodata is `NaN` keeps its gaps as `NaN` (no sentinel to swap)."""
        arr = np.array([[np.nan, 2.0], [3.0, 4.0]])
        df = _raster(arr, no_data_value=np.nan).to_dataframe()
        assert bool(np.isnan(df["values"].iloc[0]))
        assert df["values"].iloc[1:].tolist() == [2.0, 3.0, 4.0]

    def test_dropna_omits_the_nodata_rows(self):
        """``dropna=True`` drops the rows whose ``values`` is ``NaN`` (a nodata cell)."""
        ds = _raster(np.array([[-9999.0, 2.0], [3.0, 4.0]]))
        assert ds.to_dataframe().shape == (4, 1)
        dropped = ds.to_dataframe(dropna=True)
        assert dropped.shape == (3, 1)
        assert not dropped["values"].isna().any()
        assert dropped["values"].tolist() == [2.0, 3.0, 4.0]

    def test_unstack_band_gives_the_wide_view(self):
        """`df["values"].unstack("band")` is the one-column-per-band view."""
        wide = (
            _raster(np.arange(8.0).reshape(2, 2, 2))
            .to_dataframe()["values"]
            .unstack("band")
        )
        assert list(wide.columns) == [0, 1]
        assert wide.shape == (4, 2)


class TestBandSelector:
    """`bands=` chooses which bands become rows, and in what order."""

    def test_a_single_int_selects_one_band(self):
        """`bands=1` keeps only that band, still under the `band` level."""
        df = _raster(np.arange(8.0).reshape(2, 2, 2)).to_dataframe(bands=1)
        assert df.index.get_level_values("band").unique().tolist() == [1]
        assert df["values"].tolist() == [4.0, 5.0, 6.0, 7.0]

    def test_a_sequence_selects_and_orders(self):
        """A sequence of indices takes those bands in the given order."""
        df = _raster(np.arange(8.0).reshape(2, 2, 2)).to_dataframe(bands=[1, 0])
        assert df.index.get_level_values("band").unique().tolist() == [1, 0]

    def test_an_out_of_range_band_is_refused(self):
        """A band index beyond the raster raises naming the offending index."""
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        with pytest.raises(ValueError, match="out of range"):
            ds.to_dataframe(bands=5)

    def test_an_empty_band_selection_is_refused(self):
        """An empty ``bands=`` selection has nothing to put in rows, so it raises."""
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        with pytest.raises(ValueError, match="empty bands"):
            ds.to_dataframe(bands=[])

    def test_a_boolean_band_selection_is_refused(self):
        """A boolean is a type mistake, not a band index (``bool`` subclasses ``int``), so it raises."""
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        with pytest.raises(TypeError, match="boolean"):
            ds.to_dataframe(bands=True)


class TestRoundTripReproducesTheRaster:
    """`Dataset.from_dataframe(ds.to_dataframe(), crs=ds.epsg)` reproduces the source."""

    def test_a_multi_band_raster_round_trips(self):
        """Values, band count, geotransform and CRS all come back."""
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        back = Dataset.from_dataframe(ds.to_dataframe(), crs=ds.epsg)
        assert back.band_count == 2
        assert_array_equal(back.read_array(), ds.read_array())
        assert_allclose(back.geotransform, GEO_CUBE)
        assert back.epsg == 4326

    def test_a_single_band_raster_round_trips(self):
        """A one-band raster rebuilds a one-band raster with the same grid."""
        ds = _raster(np.arange(6.0).reshape(2, 3), geo=GEO_2D)
        back = Dataset.from_dataframe(ds.to_dataframe(), crs=4326)
        assert back.band_count == 1
        assert_array_equal(back.read_array(), ds.read_array())
        assert_allclose(back.geotransform, GEO_2D)

    def test_a_two_level_frame_builds_a_single_band_raster(self):
        """A plain `(y, x)` frame with no band level rebuilds a single-band raster."""
        ds = _raster(np.arange(6.0).reshape(2, 3), geo=GEO_2D)
        flat = ds.to_dataframe().droplevel("band")
        back = Dataset.from_dataframe(flat, crs=4326)
        assert back.band_count == 1
        assert_array_equal(back.read_array(), ds.read_array())

    def test_a_written_raster_round_trips(self, tmp_path):
        """With `path=`, the raster is written to a GeoTIFF and read back.

        Args:
            tmp_path: pytest's temporary directory.
        """
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        out = tmp_path / "from_df.tif"
        back = Dataset.from_dataframe(ds.to_dataframe(), crs=ds.epsg, path=str(out))
        assert out.exists()
        assert back.band_count == 2
        assert_array_equal(back.read_array(), ds.read_array())


class TestSingleCellAxisIsNotRoundTrippable:
    """A 1-wide / 1-tall raster tabulates fine, but ``from_dataframe`` cannot invert it.

    ``to_dataframe`` places every cell regardless of grid shape, but ``from_dataframe`` needs
    at least two coordinates per axis to infer a cell size, so the round trip is not universally
    invertible — a documented asymmetry, not a bug.
    """

    def test_a_single_column_raster_refuses_on_the_way_back(self):
        """A 1-column raster tabulates, then ``from_dataframe`` refuses its single-cell x axis."""
        ds = _raster(np.arange(2.0).reshape(2, 1))
        df = ds.to_dataframe()
        assert df.shape == (2, 1)
        with pytest.raises(ValueError, match="single x coordinate"):
            Dataset.from_dataframe(df, crs=4326)

    def test_a_single_row_raster_refuses_on_the_way_back(self):
        """A 1-row raster tabulates, then ``from_dataframe`` refuses its single-cell y axis."""
        ds = _raster(np.arange(2.0).reshape(1, 2))
        df = ds.to_dataframe()
        assert df.shape == (2, 1)
        with pytest.raises(ValueError, match="single y coordinate"):
            Dataset.from_dataframe(df, crs=4326)


class TestOrientationIsAlwaysNorthUp:
    """The result is north-up whatever order the frame's `y` level runs in."""

    def test_a_south_first_frame_rebuilds_the_same_raster(self):
        """Sorting the frame's rows y-ascending does not flip the rebuilt raster."""
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        frame = ds.to_dataframe()
        shuffled = frame.sort_index(level="y", ascending=True)
        north_up = Dataset.from_dataframe(frame, crs=4326).read_array()
        from_shuffled = Dataset.from_dataframe(shuffled, crs=4326).read_array()
        assert_array_equal(north_up, from_shuffled)
        assert_array_equal(north_up, ds.read_array())


class TestGapsSurviveTheRoundTrip:
    """A `NaN` cell rebuilds as a gap that reads back as `NaN`."""

    def test_a_nan_cell_reads_back_as_a_gap(self):
        """A `NaN` value rebuilds a raster whose gap is `NaN` in `to_dataframe`."""
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        frame = ds.to_dataframe()
        frame.iloc[0] = np.nan
        back = Dataset.from_dataframe(frame, crs=4326)
        assert np.isnan(back.to_dataframe()["values"].to_numpy()).sum() == 1

    def test_a_missing_row_becomes_a_gap(self):
        """A cell absent from the frame is reindexed in as a `NaN` gap, not an error."""
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        frame = ds.to_dataframe().iloc[1:]
        back = Dataset.from_dataframe(frame, crs=4326)
        assert np.isnan(back.to_dataframe()["values"].to_numpy()).sum() == 1


class TestNodataFillsWithNaNByDefault:
    """`from_dataframe` stamps gaps with `NaN` by default, so it cannot invent or lose a gap."""

    def test_a_real_value_equal_to_the_old_default_is_not_a_gap(self):
        """A real ``-9999.0`` in a frame stays a value; the ``NaN`` default cannot collide."""
        idx = pd.MultiIndex.from_product(
            [[0], [2.0, 1.0], [0.0, 1.0]], names=["band", "y", "x"]
        )
        frame = pd.DataFrame({"values": [-9999.0, 2.0, 3.0, 4.0]}, index=idx)
        back = Dataset.from_dataframe(frame, crs=4326)
        values = back.to_dataframe()["values"]
        assert not np.isnan(values.iloc[0]), "a real -9999.0 must not become a gap"
        assert values.iloc[0] == -9999.0

    def test_a_float_raster_with_nan_gaps_round_trips_losslessly(self):
        """A float raster whose gaps are ``NaN`` comes back cell-for-cell identical."""
        ds = _raster(np.array([[np.nan, 2.0], [3.0, 4.0]]), no_data_value=np.nan)
        back = Dataset.from_dataframe(ds.to_dataframe(), crs=ds.epsg)
        assert_array_equal(back.read_array(), ds.read_array())

    def test_a_non_default_sentinel_returns_as_nan_but_can_be_restored(self):
        """A ``255``-nodata gap returns as ``NaN`` by default, or as ``255`` when asked."""
        ds = _raster(np.array([[255.0, 2.0], [3.0, 4.0]]), no_data_value=255.0)
        frame = ds.to_dataframe()
        default_back = Dataset.from_dataframe(frame, crs=ds.epsg)
        assert np.isnan(default_back.to_dataframe()["values"].iloc[0])
        restored = Dataset.from_dataframe(frame, crs=ds.epsg, no_data_value=255.0)
        assert restored.no_data_value[0] == 255.0


class TestGeoreferencingRules:
    """`crs=` supplies the CRS; without it the CRS is left unset."""

    def test_crs_none_leaves_the_crs_unset(self):
        """A DataFrame carries no CRS, so `crs=None` builds a raster with no EPSG."""
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        back = Dataset.from_dataframe(ds.to_dataframe())
        assert back.epsg in (None, 0)

    def test_crs_given_sets_the_crs(self):
        """An explicit `crs=` is applied to the result."""
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        back = Dataset.from_dataframe(ds.to_dataframe(), crs=3857)
        assert back.epsg == 3857

    def test_named_axes_select_non_default_levels(self):
        """`x=` / `y=` name the grid levels when they are not the innermost two."""
        ds = _raster(np.arange(8.0).reshape(2, 2, 2))
        reordered = ds.to_dataframe().reorder_levels(["y", "x", "band"])
        back = Dataset.from_dataframe(reordered, crs=4326, x="x", y="y")
        assert back.band_count == 2
        assert_array_equal(back.read_array(), ds.read_array())


class TestFromDataframeRefusals:
    """Frames that cannot honestly become a georeferenced raster are refused."""

    def test_a_multi_column_frame_points_at_netcdf(self):
        """More than one value column is a cube's job, so it raises pointing at NetCDF."""
        frame = _raster(np.arange(8.0).reshape(2, 2, 2)).to_dataframe()
        frame["extra"] = frame["values"]
        with pytest.raises(
            ValueError, match="single value column.*NetCDF.from_dataframe"
        ):
            Dataset.from_dataframe(frame)

    def test_more_than_one_band_level_points_at_netcdf(self):
        """A frame with two band levels is a multi-dim cube, so it raises pointing at NetCDF."""
        idx = pd.MultiIndex.from_product(
            [[0, 1], [0, 1], [2.0, 1.0], [0.0, 1.0]], names=["b1", "b2", "y", "x"]
        )
        frame = pd.DataFrame({"values": np.arange(16.0)}, index=idx)
        with pytest.raises(ValueError, match="at most one band index level"):
            Dataset.from_dataframe(frame)

    def test_an_irregular_y_axis_is_refused(self):
        """A jittered y axis has no affine transform, so it raises naming the axis."""
        idx = pd.MultiIndex.from_product(
            [[20.0, 18.0, 10.0], [0.0, 1.0]], names=["y", "x"]
        )
        frame = pd.DataFrame({"values": np.arange(6.0)}, index=idx)
        with pytest.raises(ValueError, match="regular y axis"):
            Dataset.from_dataframe(frame)

    def test_an_irregular_x_axis_is_refused(self):
        """A jittered x axis has no affine transform, so it raises naming the x axis."""
        idx = pd.MultiIndex.from_product(
            [[20.0, 18.0], [0.0, 1.0, 3.0]], names=["y", "x"]
        )
        frame = pd.DataFrame({"values": np.arange(6.0)}, index=idx)
        with pytest.raises(ValueError, match="regular x axis"):
            Dataset.from_dataframe(frame)

    def test_a_single_cell_y_axis_is_refused(self):
        """One y coordinate gives no spacing to infer, so it raises naming the axis."""
        idx = pd.MultiIndex.from_product([[20.0], [0.0, 1.0]], names=["y", "x"])
        frame = pd.DataFrame({"values": np.arange(2.0)}, index=idx)
        with pytest.raises(ValueError, match="single y coordinate"):
            Dataset.from_dataframe(frame)

    def test_duplicate_index_rows_are_refused(self):
        """More than one value for a cell is ambiguous, so it raises."""
        idx = pd.MultiIndex.from_tuples(
            [(20.0, 0.0), (20.0, 0.0), (18.0, 0.0), (18.0, 1.0)], names=["y", "x"]
        )
        frame = pd.DataFrame({"values": np.arange(4.0)}, index=idx)
        with pytest.raises(ValueError, match="duplicate index rows"):
            Dataset.from_dataframe(frame)

    def test_a_tidy_frame_is_refused(self):
        """A single value column on a plain `RangeIndex` is not a grid, so it raises."""
        frame = pd.DataFrame({"values": [3.0, 4.0, 5.0, 6.0]})
        with pytest.raises(ValueError, match="indexed by its dimensions"):
            Dataset.from_dataframe(frame)

    def test_a_missing_named_axis_is_refused(self):
        """`x=` naming a level the index does not have raises."""
        frame = _raster(np.arange(8.0).reshape(2, 2, 2)).to_dataframe()
        with pytest.raises(ValueError, match="not an\\s+index level"):
            Dataset.from_dataframe(frame, x="lon")

    def test_x_and_y_naming_the_same_level_is_refused(self):
        """Pointing both the y and x axes at one level collapses the grid, so it raises."""
        frame = _raster(np.arange(8.0).reshape(2, 2, 2)).to_dataframe()
        with pytest.raises(ValueError, match="both the y and x axes"):
            Dataset.from_dataframe(frame, x="y", y="y")

    def test_an_unnamed_index_level_is_refused(self):
        """An unnamed index level cannot be addressed as an axis, so it raises."""
        idx = pd.MultiIndex.from_product([[20.0, 18.0], [0.0, 1.0]], names=["y", None])
        frame = pd.DataFrame({"values": np.arange(4.0)}, index=idx)
        with pytest.raises(ValueError, match="every index level named"):
            Dataset.from_dataframe(frame)

    def test_a_non_numeric_value_column_is_refused_by_name(self):
        """A text value column raises a message naming the column."""
        idx = pd.MultiIndex.from_product([[20.0, 18.0], [0.0, 1.0]], names=["y", "x"])
        frame = pd.DataFrame({"values": ["a", "b", "c", "d"]}, index=idx)
        with pytest.raises(ValueError, match="from_dataframe.*'values'.*numeric"):
            Dataset.from_dataframe(frame)
