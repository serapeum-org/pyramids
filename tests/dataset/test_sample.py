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
from shapely.geometry import Point

from pyramids.base._errors import OutOfBoundsError
from pyramids.dataset import Dataset
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
        assert result.tolist() == [[0.0, 12.0], [25.0, 37.0]], f"Wrong values: {result.tolist()}"

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
        assert result.tolist() == [[25.0, 37.0], [0.0, 12.0]], f"Wrong order/values: {result.tolist()}"

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
        assert result.mask.tolist() == [False, True], f"Wrong mask: {result.mask.tolist()}"
        assert result[0] == 12.0, f"Inside value lost: {result[0]}"

    def test_masked_output_multiband(self, two_band, mixed_points):
        """masked=True broadcasts the mask across all bands.

        Test scenario:
            For 2 bands the outside column is masked in both rows.
        """
        result = two_band.sample(mixed_points, masked=True)
        assert result.shape == (2, 2), f"Expected (2, 2), got {result.shape}"
        assert result.mask.tolist() == [[False, True], [False, True]], (
            f"Wrong mask: {result.mask.tolist()}"
        )

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
        assert result[0] == 12.0, f"Inside value wrong: {result[0]}"
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
