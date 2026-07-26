"""Tests for :meth:`Dataset.sample` / ``Analysis.sample`` and ``_points_to_xy`` (PB-9).

Covers the windowed, vectorised, out-of-bounds-safe point-sampling path: band
selection (all / single / subset), the three ``on_out_of_bounds`` policies,
masked output, no-data handling and int→float promotion, every input type
accepted by ``_points_to_xy``, and the guard clauses. All fixtures are in-memory
``create_from_array`` rasters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame
from osgeo import gdal, osr
from shapely.geometry import Point

from pyramids.base._errors import OutOfBoundsError
from pyramids.dataset import Dataset
from pyramids.dataset.engines import analysis
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def two_band() -> Dataset:
    """A 2-band 5x5 raster, values 0..49, extent x:[0,5] y:[0,5], nodata -9999.

    Band 0 holds 0..24, band 1 holds 25..49 (row-major). Top-left corner (0, 5),
    cell size 1, EPSG:4326.

    Returns:
        Dataset: the test raster.
    """
    arr = np.arange(2 * 5 * 5, dtype="float32").reshape(2, 5, 5)
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326, no_data_value=-9999.0
    )


@pytest.fixture(scope="function")
def inside_points() -> GeoDataFrame:
    """Two points inside ``two_band``: cell (0,0) and the centre cell (2,2).

    Returns:
        GeoDataFrame: EPSG:4326 points at (0.5, 4.5) and (2.5, 2.5).
    """
    return GeoDataFrame(geometry=[Point(0.5, 4.5), Point(2.5, 2.5)], crs=4326)


@pytest.fixture(scope="function")
def mixed_points() -> GeoDataFrame:
    """One inside point (centre cell) and one far outside the extent.

    Returns:
        GeoDataFrame: points at (2.5, 2.5) [inside] and (100, 100) [outside].
    """
    return GeoDataFrame(geometry=[Point(2.5, 2.5), Point(100, 100)], crs=4326)


class TestSample:
    """Tests for ``Analysis.sample`` via the ``Dataset.sample`` facade."""

    def test_all_bands_default(self, two_band, inside_points):
        """Default samples every band, returning (n_bands, n_points).

        Test scenario:
            Cell (0,0)=0 / (2,2)=12 on band 0; +25 on band 1.
        """
        result = two_band.sample(inside_points)
        assert result.shape == (2, 2), f"Expected (2, 2), got {result.shape}"
        assert result.tolist() == [
            [0.0, 12.0],
            [25.0, 37.0],
        ], f"Wrong values: {result.tolist()}"

    def test_single_band_is_1d(self, two_band, inside_points):
        """A single int band returns a flat (n_points,) array.

        Test scenario:
            bands=0 yields the band-0 values as a 1-D array.
        """
        result = two_band.sample(inside_points, bands=0)
        assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
        assert result.tolist() == [0.0, 12.0], f"Wrong values: {result.tolist()}"

    def test_band_subset_order_preserved(self, two_band, inside_points):
        """A band list samples in the requested order.

        Test scenario:
            bands=[1, 0] returns band 1 first, then band 0.
        """
        result = two_band.sample(inside_points, bands=[1, 0])
        assert result.tolist() == [
            [25.0, 37.0],
            [0.0, 12.0],
        ], f"Wrong order/values: {result.tolist()}"

    def test_out_of_bounds_nodata(self, two_band, mixed_points):
        """Default policy fills out-of-bounds points with the band no-data value.

        Test scenario:
            The (100,100) point yields -9999 while the inside point is read.
        """
        result = two_band.sample(mixed_points, bands=0)
        assert result.tolist() == [12.0, -9999.0], f"Wrong values: {result.tolist()}"

    def test_out_of_bounds_snap(self, two_band, mixed_points):
        """The 'snap' policy clamps out-of-bounds points to the nearest edge.

        Test scenario:
            (100,100) snaps to the top-right cell (0,4) = value 4.
        """
        result = two_band.sample(mixed_points, bands=0, on_out_of_bounds="snap")
        assert result.tolist() == [12.0, 4.0], f"Wrong values: {result.tolist()}"

    def test_out_of_bounds_raise(self, two_band, mixed_points):
        """The 'raise' policy raises OutOfBoundsError listing the count.

        Test scenario:
            One of two points is outside, so the error reports 1 of 2.
        """
        with pytest.raises(OutOfBoundsError, match="1 of 2 points"):
            two_band.sample(mixed_points, on_out_of_bounds="raise")

    def test_masked_output(self, two_band, mixed_points):
        """masked=True returns a MaskedArray masking out-of-bounds points.

        Test scenario:
            The outside point is masked; the inside value is preserved.
        """
        result = two_band.sample(mixed_points, bands=0, masked=True)
        assert np.ma.isMaskedArray(result), "Expected a masked array"
        assert result.mask.tolist() == [
            False,
            True,
        ], f"Wrong mask: {result.mask.tolist()}"
        assert result[0] == pytest.approx(12.0), f"Inside value lost: {result[0]}"

    def test_masked_output_multiband(self, two_band, mixed_points):
        """masked=True broadcasts the mask across all bands.

        Test scenario:
            For 2 bands the outside column is masked in both rows.
        """
        result = two_band.sample(mixed_points, masked=True)
        assert result.shape == (2, 2), f"Expected (2, 2), got {result.shape}"
        assert result.mask.tolist() == [
            [False, True],
            [False, True],
        ], f"Wrong mask: {result.mask.tolist()}"

    def test_int_band_without_nodata_promotes_to_float(self, mixed_points):
        """An int band with no no-data value fills out-of-bounds with NaN (float).

        Test scenario:
            int32 raster, no nodata: result dtype is float64 and the outside
            point is NaN.
        """
        arr = np.arange(25, dtype="int32").reshape(1, 5, 5)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326, no_data_value=None
        )
        result = ds.sample(mixed_points, bands=0)
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"
        assert result[0] == pytest.approx(12.0), f"Inside value wrong: {result[0]}"
        assert np.isnan(result[1]), f"Outside point should be NaN, got {result[1]}"

    def test_dataframe_input(self, two_band):
        """A DataFrame with x/y columns is accepted.

        Test scenario:
            Two coordinates given as a DataFrame return the same values as the
            equivalent point geometries.
        """
        df = pd.DataFrame({"x": [0.5, 2.5], "y": [4.5, 2.5]})
        result = two_band.sample(df, bands=0)
        assert result.tolist() == [0.0, 12.0], f"Wrong values: {result.tolist()}"

    def test_feature_collection_input(self, two_band, inside_points):
        """A FeatureCollection is accepted just like a GeoDataFrame.

        Test scenario:
            Wrapping the points in a FeatureCollection yields identical values.
        """
        fc = FeatureCollection(inside_points)
        result = two_band.sample(fc, bands=0)
        assert result.tolist() == [0.0, 12.0], f"Wrong values: {result.tolist()}"

    def test_matches_extract_for_inbounds(self, two_band, inside_points):
        """sample agrees with extract(mask=…) for in-bounds points.

        Test scenario:
            For points inside the extent both APIs return the same band-0 values.
        """
        sampled = two_band.sample(inside_points, bands=0)
        extracted = two_band.extract(band=0, mask=inside_points)
        assert sampled.tolist() == extracted.tolist(), (
            f"sample {sampled.tolist()} != extract {extracted.tolist()}"
        )

    def test_empty_band_list(self, two_band, inside_points):
        """An empty band list yields a (0, n_points) array.

        Test scenario:
            bands=[] selects no bands.
        """
        result = two_band.sample(inside_points, bands=[])
        assert result.shape == (0, 2), f"Expected (0, 2), got {result.shape}"

    def test_invalid_on_out_of_bounds_raises(self, two_band, inside_points):
        """An unknown on_out_of_bounds value raises ValueError.

        Test scenario:
            'clamp' is not a recognised policy.
        """
        with pytest.raises(ValueError, match="on_out_of_bounds must be one of"):
            two_band.sample(inside_points, on_out_of_bounds="clamp")

    def test_band_out_of_range_raises(self, two_band, inside_points):
        """A band index beyond the raster raises ValueError.

        Test scenario:
            band=9 on a 2-band raster is out of range.
        """
        with pytest.raises(ValueError, match="out of range"):
            two_band.sample(inside_points, bands=9)

    def test_bad_points_type_raises(self, two_band):
        """A non-point input type raises TypeError.

        Test scenario:
            A bare list is not a FeatureCollection / GeoDataFrame / DataFrame.
        """
        with pytest.raises(TypeError, match="must be a FeatureCollection"):
            two_band.sample([(0.5, 4.5)], bands=0)

    def test_dataframe_without_xy_raises(self, two_band):
        """A DataFrame lacking x/y columns raises ValueError.

        Test scenario:
            A DataFrame with only a 'lon' column is rejected.
        """
        df = pd.DataFrame({"lon": [0.5], "lat": [4.5]})
        with pytest.raises(ValueError, match="must have 'x' and 'y' columns"):
            two_band.sample(df, bands=0)


class TestHistogramEdgesAndStatsPrecision:
    """ARC-27: bucket edges must describe the buckets that were counted."""

    @staticmethod
    def _ramp() -> Dataset:
        """A 10x10 raster holding 0..99.

        Returns:
            Dataset: In-memory single-band dataset.
        """
        array = np.arange(100, dtype="float32").reshape(10, 10)
        raster = gdal.GetDriverByName("MEM").Create("", 10, 10, 1, gdal.GDT_Float32)
        raster.SetGeoTransform((0.0, 1.0, 0.0, 10.0, 0.0, -1.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        raster.SetProjection(srs.ExportToWkt())
        band = raster.GetRasterBand(1)
        band.WriteArray(array)
        band.SetNoDataValue(-9999.0)
        return Dataset(raster)

    def test_edges_follow_the_requested_range(self):
        """Narrowing the range moves the reported edges with it.

        Test scenario:
            `bin_width` came from the caller's `min_value`/`max_value` but the
            edges were anchored at the raster minimum, so a narrowed request
            returned edges describing buckets `GetHistogram` never filled.
        """
        dataset = self._ramp()
        _, ranges = dataset.get_histogram(band=0, bins=5, min_value=50, max_value=100)
        assert ranges[0][0] == pytest.approx(50.0), (
            f"first edge should be the requested min_value 50, got {ranges[0][0]}"
        )
        assert ranges[-1][1] == pytest.approx(100.0), (
            f"last edge should be the requested max_value 100, got {ranges[-1][1]}"
        )

    def test_default_range_still_spans_the_raster(self):
        """With no narrowing the edges still start at the raster minimum."""
        dataset = self._ramp()
        _, ranges = dataset.get_histogram(band=0, bins=5)
        assert ranges[0][0] == pytest.approx(0.0), (
            f"the default range must start at the raster min, got {ranges[0][0]}"
        )

    def test_stats_exposes_approx_ok(self):
        """`stats` lets the caller demand exact figures.

        Test scenario:
            `GetStatistics(True, True)` was hard-coded, so the returned
            min/max/mean/std could come from overviews or a subsample with no
            way to ask for the exact values.
        """
        dataset = self._ramp()
        exact = dataset.stats(approx_ok=False)
        assert list(exact.columns) == ["min", "max", "mean", "std"], (
            f"unexpected columns: {list(exact.columns)}"
        )
        assert float(exact["min"].iloc[0]) == pytest.approx(0.0), (
            f"exact min over 0..99 should be 0, got {exact['min'].iloc[0]}"
        )
        assert float(exact["max"].iloc[0]) == pytest.approx(99.0), (
            f"exact max over 0..99 should be 99, got {exact['max'].iloc[0]}"
        )


def _points(coords: list[tuple[float, float]]) -> GeoDataFrame:
    """Build an EPSG:4326 point GeoDataFrame from ``(x, y)`` pairs."""
    return GeoDataFrame(geometry=[Point(x, y) for x, y in coords], crs=4326)


class TestWindowedAndPerPointStrategiesAgree:
    """ARC-61: the windowed read is an optimisation, not a behaviour change."""

    @pytest.fixture(scope="function")
    def wide(self) -> Dataset:
        """A 1-band 100x100 raster of distinct values, nodata -9999.

        Large enough that two far-apart points span a bounding box past the
        windowed-read threshold, so the per-point branch is reachable without
        touching the module constants.

        Returns:
            Dataset: the test raster.
        """
        arr = np.arange(100 * 100, dtype="float64").reshape(100, 100)
        return Dataset.create_from_array(
            arr,
            top_left_corner=(0, 100),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )

    def test_a_sparse_batch_takes_the_per_point_branch(self, wide, monkeypatch):
        """Two corner points span too large a box to be worth one read.

        Test scenario:
            The windowed read is taken only while the points' bounding box stays
            a small multiple of the point count. Two opposite corners of a
            100x100 raster span 10000 pixels for 2 points, past the 4096-pixel
            floor, so the sparse branch runs. Counts `ReadAsArray` calls to
            prove which branch executed rather than inferring it.
        """
        calls: list[tuple] = []
        original = gdal.Band.ReadAsArray

        def counting_read(self, *args, **kwargs):
            calls.append(args)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(gdal.Band, "ReadAsArray", counting_read)
        wide.sample(_points([(0.5, 99.5), (99.5, 0.5)]))
        assert len(calls) == 2, (
            f"a sparse batch must issue one read per point, got {len(calls)}"
        )
        assert all(args[2:] == (1, 1) for args in calls), (
            f"each per-point read must be a 1x1 window, got {calls}"
        )

    def test_a_dense_batch_takes_the_windowed_branch(self, wide, monkeypatch):
        """Neighbouring points collapse into a single read.

        Test scenario:
            Four adjacent cells span a 2x2 box, far under the threshold, so one
            windowed read covers them all.
        """
        calls: list[tuple] = []
        original = gdal.Band.ReadAsArray

        def counting_read(self, *args, **kwargs):
            calls.append(args)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(gdal.Band, "ReadAsArray", counting_read)
        wide.sample(_points([(10.5, 50.5), (11.5, 50.5), (10.5, 49.5), (11.5, 49.5)]))
        assert len(calls) == 1, (
            f"a dense batch must collapse into one windowed read, got {len(calls)}"
        )

    def test_both_strategies_return_identical_values(self, wide):
        """The two branches sample the same cells and return the same numbers.

        Test scenario:
            Samples one dense batch through the windowed branch, then the same
            points again with the threshold forced to zero so the per-point
            branch runs, and compares. The windowed branch indexes into a block
            read at an offset; an off-by-one there would go unnoticed because
            nothing else compares the two paths.
        """
        points = _points(
            [(10.5, 50.5), (11.5, 50.5), (12.5, 50.5), (10.5, 49.5), (11.5, 49.5)]
        )
        windowed = wide.sample(points)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(analysis, "_POINT_WINDOW_MIN_PIXELS", 0)
            patch.setattr(analysis, "_POINT_WINDOW_MAX_WASTE", 0)
            per_point = wide.sample(points)
        np.testing.assert_array_equal(windowed, per_point)

    def test_the_strategies_agree_on_an_out_of_bounds_mix(self, wide):
        """Out-of-bounds points get the fill value on both branches.

        Test scenario:
            The in-bounds index set drives the window offset, so a batch that
            mixes in- and out-of-bounds points is where an indexing slip would
            surface as a value landing in the wrong slot.
        """
        points = _points(
            [(10.5, 50.5), (-5.0, 50.5), (11.5, 50.5), (500.0, 50.5), (12.5, 50.5)]
        )
        windowed = wide.sample(points)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(analysis, "_POINT_WINDOW_MIN_PIXELS", 0)
            patch.setattr(analysis, "_POINT_WINDOW_MAX_WASTE", 0)
            per_point = wide.sample(points)
        np.testing.assert_array_equal(windowed, per_point)
        assert windowed[0][1] == -9999.0 and windowed[0][3] == -9999.0, (
            f"out-of-bounds points must carry the fill value, got {windowed[0]}"
        )


class TestStatsRecoveryPath:
    """H1: the no-cached-statistics recovery must be exact, not approximate."""

    @pytest.fixture(scope="function")
    def sparse_on_disk(self, tmp_path) -> str:
        """A 2048x2048 GTiff with overviews and two valid pixels.

        The shape any clipped catchment, sparse mask or `to_cog` output takes:
        so much no-data that GDAL's overview-based estimate sees only the fill
        and either fails outright or averages the two valid pixels into one.

        Args:
            tmp_path: pytest's temporary directory fixture.

        Returns:
            str: path to the written raster.
        """
        path = tmp_path / "sparse.tif"
        array = np.full((2048, 2048), -9999.0, dtype="float32")
        array[0, 0] = 1.0
        array[0, 1] = 4.0
        driver = gdal.GetDriverByName("GTiff")
        raster = driver.Create(str(path), 2048, 2048, 1, gdal.GDT_Float32)
        raster.GetRasterBand(1).WriteArray(array)
        raster.GetRasterBand(1).SetNoDataValue(-9999.0)
        raster.BuildOverviews("average", [2, 4, 8, 16])
        raster = None
        return str(path)

    @staticmethod
    def _force_the_recovery(monkeypatch):
        """Make GDAL's cached/approximate statistics unavailable.

        Reproduces the state the recovery exists for: `GetStatistics` raises
        "Failed to compute statistics, no valid pixels found in sampling",
        which the engine turns into the `sum(vals) == 0` fallback.

        Args:
            monkeypatch: pytest's monkeypatch fixture.
        """

        def refuse(self, *args, **kwargs):
            raise RuntimeError(
                "Failed to compute statistics, no valid pixels found in sampling."
            )

        monkeypatch.setattr(gdal.Band, "GetStatistics", refuse)

    def test_the_recovery_does_not_re_run_the_route_that_just_failed(
        self, sparse_on_disk, monkeypatch
    ):
        """The fallback is a full scan, so it survives what the estimate could not.

        Test scenario:
            The recovery only runs because the approximate route already failed.
            Repeating it approximately raises the very same error, straight out
            of the public `stats()` -- the default call, with no caller opting
            in. Honouring `approx_ok` here made the recovery a no-op.
        """
        self._force_the_recovery(monkeypatch)
        stats = Dataset.read_file(sparse_on_disk).stats()
        assert float(stats["min"].iloc[0]) == 1.0, (
            f"expected the true minimum 1.0, got {stats['min'].iloc[0]}"
        )
        assert float(stats["max"].iloc[0]) == 4.0, (
            f"expected the true maximum 4.0, got {stats['max'].iloc[0]}"
        )
        assert float(stats["std"].iloc[0]) > 0.0, (
            "a band with two distinct values must not report a zero deviation"
        )

    def test_the_recovery_is_exact_for_either_approx_ok(
        self, sparse_on_disk, monkeypatch
    ):
        """`approx_ok` does not reach the recovery, so both callers agree.

        Test scenario:
            Both routes end in the same full scan once the estimate is gone.
        """
        self._force_the_recovery(monkeypatch)
        dataset = Dataset.read_file(sparse_on_disk)
        np.testing.assert_array_equal(
            dataset.stats(approx_ok=False).to_numpy(), dataset.stats().to_numpy()
        )

    def test_a_healthy_band_still_answers_from_the_estimate(self, two_band):
        """The recovery is not on the happy path.

        Test scenario:
            A band whose statistics GDAL can compute must not pay a full scan;
            the fallback is reached only when the estimate returns nothing.
        """
        stats = two_band.stats()
        assert len(stats) == 2, f"expected one row per band, got {len(stats)}"
        assert float(stats["min"].iloc[0]) == 0.0, (
            f"band 0 spans 0..24, got a minimum of {stats['min'].iloc[0]}"
        )
