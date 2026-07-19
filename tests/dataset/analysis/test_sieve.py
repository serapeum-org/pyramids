"""Tests for :meth:`Dataset.sieve` / ``Analysis.sieve`` (PB-5).

Covers ``gdal.SieveFilter``-based speckle removal: small clumps dissolved into
their neighbour, large clumps preserved, 4- vs 8-connectivity, no-data
preservation, band selection, the optional mask band, geo/CRS round-trip, and
the guard clauses. All fixtures are in-memory ``create_from_array`` rasters.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def speckled() -> Dataset:
    """A 6x6 raster of 1s with a 3x3 clump of 2s and one isolated 2 pixel.

    The 9-pixel clump survives a small threshold; the lone pixel does not.

    Returns:
        Dataset: int32 EPSG:4326 raster, top-left (0, 6), cell size 1.
    """
    arr = np.ones((6, 6), dtype="int32")
    arr[0:3, 0:3] = 2
    arr[5, 5] = 2
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326
    )


@pytest.fixture(scope="function")
def diagonal() -> Dataset:
    """A 5x5 raster of 1s with two 2-pixels touching only diagonally.

    Returns:
        Dataset: int32 EPSG:4326 raster with 2s at (1,1) and (2,2).
    """
    arr = np.ones((5, 5), dtype="int32")
    arr[1, 1] = 2
    arr[2, 2] = 2
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326
    )


class TestSieve:
    """Tests for ``Analysis.sieve`` via the ``Dataset.sieve`` facade."""

    def test_small_clump_removed(self, speckled):
        """A clump below threshold is merged into its neighbour.

        Test scenario:
            The isolated 1-pixel '2' at (5,5) becomes background '1' at
            threshold=4.
        """
        result = speckled.sieve(threshold=4).read_array()
        assert result[5, 5] == 1, (
            f"Isolated pixel should be merged to 1, got {result[5, 5]}"
        )

    def test_large_clump_preserved(self, speckled):
        """A clump at or above threshold is preserved.

        Test scenario:
            The 9-pixel '2' block survives threshold=4 unchanged.
        """
        result = speckled.sieve(threshold=4).read_array()
        assert result[0, 0] == 2 and result[2, 2] == 2, (
            f"Large clump should survive, got corners {result[0, 0]} / {result[2, 2]}"
        )

    def test_connectedness_4_removes_diagonal(self, diagonal):
        """With 4-connectivity, diagonal-only pixels are separate size-1 clumps.

        Test scenario:
            Each lone '2' is size 1 < threshold=2, so both are removed.
        """
        result = diagonal.sieve(threshold=2, connectedness=4).read_array()
        assert result[1, 1] == 1 and result[2, 2] == 1, (
            f"4-connectivity should remove diagonal singles, got {result[1, 1]} / {result[2, 2]}"
        )

    def test_connectedness_8_keeps_diagonal(self, diagonal):
        """With 8-connectivity, diagonal-touching pixels form one size-2 clump.

        Test scenario:
            The two '2's join into a size-2 clump that meets threshold=2.
        """
        result = diagonal.sieve(threshold=2, connectedness=8).read_array()
        assert result[1, 1] == 2 and result[2, 2] == 2, (
            f"8-connectivity should keep the joined clump, got {result[1, 1]} / {result[2, 2]}"
        )

    def test_single_band_output(self, speckled):
        """The result is always a single-band Dataset.

        Test scenario:
            Sieving a single-band raster returns a 1-band Dataset.
        """
        result = speckled.sieve(threshold=4)
        assert isinstance(result, Dataset), f"Expected a Dataset, got {type(result)}"
        assert result.band_count == 1, f"Expected 1 band, got {result.band_count}"

    def test_geotransform_and_crs_preserved(self, speckled):
        """The output keeps the source geotransform and CRS.

        Test scenario:
            Shape, geotransform, and EPSG match the source.
        """
        result = speckled.sieve(threshold=4)
        assert (result.rows, result.columns) == (
            6,
            6,
        ), f"Wrong shape: {result.rows}x{result.columns}"
        assert result.geotransform == speckled.geotransform, (
            "Geotransform not preserved"
        )
        assert result.epsg == 4326, f"Expected EPSG 4326, got {result.epsg}"

    def test_nodata_preserved(self):
        """A source no-data value is carried onto the output band.

        Test scenario:
            A raster with nodata=-1 yields an output reporting nodata -1.
        """
        arr = np.ones((6, 6), dtype="int32")
        arr[5, 5] = 2
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326, no_data_value=-1
        )
        result = ds.sieve(threshold=4)
        assert result.no_data_value[0] == -1, (
            f"Expected nodata -1, got {result.no_data_value[0]}"
        )

    def test_no_nodata_value(self):
        """A source without a no-data value yields an output band with none.

        Test scenario:
            no_data_value=None is not back-filled on the output.
        """
        arr = np.ones((6, 6), dtype="int32")
        arr[5, 5] = 2
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326, no_data_value=None
        )
        result = ds.sieve(threshold=4)
        assert result.raster.GetRasterBand(1).GetNoDataValue() is None, (
            "Output should have no nodata"
        )

    def test_band_selection(self):
        """A non-default band is sieved when band= is given.

        Test scenario:
            A 2-band raster whose band 1 has a removable speckle is cleaned on
            band 1.
        """
        clean = np.ones((6, 6), dtype="int32")
        speck = np.ones((6, 6), dtype="int32")
        speck[5, 5] = 2
        arr = np.stack([clean, speck])
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326
        )
        result = ds.sieve(threshold=4, band=1).read_array()
        assert result[5, 5] == 1, f"Band-1 speckle not removed, got {result[5, 5]}"

    def test_mask_band(self, speckled):
        """An all-valid mask behaves like no mask (sieving still occurs).

        Test scenario:
            Passing a mask of all-ones removes the small clump just as the
            default path does.
        """
        mask_arr = np.ones((6, 6), dtype="int32")
        mask = Dataset.create_from_array(
            mask_arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326
        )
        result = speckled.sieve(threshold=4, mask=mask).read_array()
        assert result[5, 5] == 1, (
            f"Small clump should be removed with mask, got {result[5, 5]}"
        )

    def test_threshold_below_one_raises(self, speckled):
        """A threshold below 1 raises ValueError.

        Test scenario:
            threshold=0 is rejected.
        """
        with pytest.raises(ValueError, match="threshold must be >= 1"):
            speckled.sieve(threshold=0)

    def test_invalid_connectedness_raises(self, speckled):
        """A connectedness other than 4 or 8 raises ValueError.

        Test scenario:
            connectedness=5 is invalid.
        """
        with pytest.raises(ValueError, match="connectedness must be 4 or 8"):
            speckled.sieve(threshold=4, connectedness=5)

    def test_band_out_of_range_raises(self, speckled):
        """A band index beyond the raster raises ValueError.

        Test scenario:
            band=9 on a 1-band raster is out of range.
        """
        with pytest.raises(ValueError, match="out of range"):
            speckled.sieve(threshold=4, band=9)
