"""Tests for :meth:`Dataset.proximity` / ``Analysis.proximity`` (PB-6).

Covers ``gdal.ComputeProximity``-based distance-to-target rasters: distance 0 at
targets growing outward, GEO vs PIXEL units, ``target_values`` selection,
``max_distance`` clamping with ``nodata`` fill, no-data band tagging, band
selection, geo/CRS round-trip, and the guard clauses. All fixtures are in-memory
``create_from_array`` rasters.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def single_target() -> Dataset:
    """A 5x5 raster of zeros with a single target pixel (value 1) at (2, 2).

    Returns:
        Dataset: int32 EPSG:4326 raster, top-left (0, 5), cell size 1.
    """
    arr = np.zeros((5, 5), dtype="int32")
    arr[2, 2] = 1
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326
    )


@pytest.fixture(scope="function")
def two_classes() -> Dataset:
    """A 5x5 raster with target value 1 at (2,2) and value 5 at (0,0).

    Returns:
        Dataset: int32 EPSG:4326 raster, top-left (0, 5), cell size 1.
    """
    arr = np.zeros((5, 5), dtype="int32")
    arr[2, 2] = 1
    arr[0, 0] = 5
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326
    )


class TestProximity:
    """Tests for ``Analysis.proximity`` via the ``Dataset.proximity`` facade."""

    def test_zero_at_target(self, single_target):
        """The target pixel itself has distance 0.

        Test scenario:
            Pixel (2,2) is the target, so its proximity is 0.
        """
        result = single_target.proximity(distance_units="PIXEL").read_array()
        assert (
            float(result[2, 2]) == pytest.approx(0.0)
        ), f"Target pixel should be 0, got {result[2, 2]}"

    def test_euclidean_growth_pixel_units(self, single_target):
        """Distances grow Euclidean-ally outward in pixel units.

        Test scenario:
            Two cells left of the target is 2.0; the diagonal corner is 2*sqrt2.
        """
        result = single_target.proximity(distance_units="PIXEL").read_array()
        assert float(result[2, 0]) == pytest.approx(2.0), f"Got {result[2, 0]}"
        assert float(result[4, 4]) == pytest.approx(
            2 * math.sqrt(2)
        ), f"Got {result[4, 4]}"

    def test_geo_units_scale_by_cell_size(self):
        """GEO units multiply pixel distance by the cell size.

        Test scenario:
            With cell size 2, two cells from the target reads 4.0 in GEO units.
        """
        arr = np.zeros((5, 5), dtype="int32")
        arr[2, 2] = 1
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 10), cell_size=2.0, epsg=4326
        )
        result = ds.proximity(distance_units="GEO").read_array()
        assert float(result[2, 0]) == pytest.approx(4.0), f"Got {result[2, 0]}"

    def test_nonzero_targets_default(self, two_classes):
        """Without target_values every non-zero pixel is a target.

        Test scenario:
            Both (2,2) and (0,0) are targets, so both read 0.
        """
        result = two_classes.proximity(distance_units="PIXEL").read_array()
        assert (
            float(result[2, 2]) == pytest.approx(0.0)
            and float(result[0, 0]) == pytest.approx(0.0)
        ), f"Both non-zero pixels should be targets, got {result[2, 2]} / {result[0, 0]}"

    def test_target_values_selects_class(self, two_classes):
        """target_values restricts which pixel values count as targets.

        Test scenario:
            With target_values=[1] the value-5 pixel at (0,0) is NOT a target;
            its distance is measured to (2,2) = 2*sqrt2.
        """
        result = two_classes.proximity(
            distance_units="PIXEL", target_values=[1]
        ).read_array()
        assert float(result[0, 0]) == pytest.approx(
            2 * math.sqrt(2)
        ), f"(0,0) should measure to the value-1 target, got {result[0, 0]}"

    def test_max_distance_with_nodata_fill(self, single_target):
        """Pixels beyond max_distance are filled with nodata.

        Test scenario:
            max_distance=1 leaves the far corner unreachable -> nodata -1, while
            an adjacent cell stays 1.0.
        """
        result = single_target.proximity(
            distance_units="PIXEL", max_distance=1.0, nodata=-1
        ).read_array()
        assert (
            float(result[4, 4]) == -1.0
        ), f"Far corner should be nodata -1, got {result[4, 4]}"
        assert float(result[2, 1]) == pytest.approx(
            1.0
        ), f"Adjacent cell should be 1, got {result[2, 1]}"

    def test_nodata_tagged_on_band(self, single_target):
        """The nodata value is recorded on the output band.

        Test scenario:
            proximity(nodata=-1) yields a band reporting no-data -1.
        """
        result = single_target.proximity(nodata=-1)
        assert (
            result.no_data_value[0] == -1.0
        ), f"Expected nodata -1, got {result.no_data_value[0]}"

    def test_output_is_float32_single_band(self, single_target):
        """The output is a single-band Float32 Dataset.

        Test scenario:
            proximity returns one float32 band regardless of source dtype.
        """
        result = single_target.proximity()
        assert isinstance(result, Dataset), f"Expected a Dataset, got {type(result)}"
        assert result.band_count == 1, f"Expected 1 band, got {result.band_count}"
        assert result.dtype[0] == "float32", f"Expected float32, got {result.dtype[0]}"

    def test_geotransform_and_crs_preserved(self, single_target):
        """The output keeps the source geotransform and CRS.

        Test scenario:
            Shape, geotransform, and EPSG match the source.
        """
        result = single_target.proximity()
        assert (result.rows, result.columns) == (
            5,
            5,
        ), f"Wrong shape: {result.rows}x{result.columns}"
        assert (
            result.geotransform == single_target.geotransform
        ), "Geotransform not preserved"
        assert result.epsg == 4326, f"Expected EPSG 4326, got {result.epsg}"

    def test_band_selection(self):
        """A non-default band is used as the target source.

        Test scenario:
            For a 2-band raster whose band 1 holds the target, band=1 measures
            distance to it.
        """
        zeros = np.zeros((5, 5), dtype="int32")
        target = np.zeros((5, 5), dtype="int32")
        target[2, 2] = 1
        arr = np.stack([zeros, target])
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326
        )
        result = ds.proximity(band=1, distance_units="PIXEL").read_array()
        assert (
            float(result[2, 2]) == pytest.approx(0.0)
        ), f"Band-1 target should be 0, got {result[2, 2]}"

    def test_invalid_distance_units_raises(self, single_target):
        """An unknown distance_units value raises ValueError.

        Test scenario:
            'metres' is not a recognised unit.
        """
        with pytest.raises(ValueError, match="distance_units must be"):
            single_target.proximity(distance_units="metres")

    def test_band_out_of_range_raises(self, single_target):
        """A band index beyond the raster raises ValueError.

        Test scenario:
            band=9 on a 1-band raster is out of range.
        """
        with pytest.raises(ValueError, match="out of range"):
            single_target.proximity(band=9)

    def test_negative_max_distance_raises(self, single_target):
        """A negative max_distance raises ValueError.

        Test scenario:
            max_distance=-5 is invalid.
        """
        with pytest.raises(ValueError, match="max_distance must be >= 0"):
            single_target.proximity(max_distance=-5)
