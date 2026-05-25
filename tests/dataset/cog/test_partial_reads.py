"""Tests for COG partial reads: read_part / preview / point / read_tile.

Covers overview-decimated geographic reads, whole-image previews, single-point
sampling, Web-Mercator XYZ tiles, the `_xyz_bounds_3857` helper, CRS
reprojection of the request window, and out-of-bounds / invalid-arg handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._errors import OutOfBoundsError
from pyramids.dataset import Dataset
from pyramids.dataset.engines.cog import _xyz_bounds_3857

pytestmark = pytest.mark.core

_GT_4326 = (0.0, 0.01, 0.0, 10.0, 0.0, -0.01)


@pytest.fixture
def ramp_4326() -> Dataset:
    """A 100x100 Float32 ramp on EPSG:4326 where value == row*100 + col.

    Returns:
        Dataset: An in-memory dataset with a deterministic ramp.
    """
    arr = np.arange(100 * 100, dtype="float32").reshape(100, 100)
    return Dataset.create_from_array(arr, geo=_GT_4326, epsg=4326)


@pytest.fixture
def tile_3857(tmp_path) -> Dataset:
    """A 256x256 Float32 ramp on EPSG:3857 covering exactly XYZ tile (1, 0, 0).

    Args:
        tmp_path: pytest temp directory.

    Returns:
        Dataset: An in-memory dataset aligned to the zoom-1 NW tile bounds.
    """
    west, south, east, north = _xyz_bounds_3857(1, 0, 0)
    cell = (east - west) / 256.0
    gt = (west, cell, 0.0, north, 0.0, -cell)
    mem = gdal.GetDriverByName("MEM").Create("", 256, 256, 1, gdal.GDT_Float32)
    mem.SetGeoTransform(gt)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(3857)
    mem.SetProjection(sr.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(
        np.arange(256 * 256, dtype="float32").reshape(256, 256)
    )
    mem.FlushCache()
    return Dataset(mem)


class TestXyzBounds3857:
    """Tests for the _xyz_bounds_3857 helper."""

    def test_zoom0_world(self):
        """Zoom 0 tile (0,0) spans the whole Web-Mercator world.

        Test scenario:
            west/south are the negative half-extent; east/north the positive.
        """
        w, s, e, n = _xyz_bounds_3857(0, 0, 0)
        assert round(w) == -20037508 and round(s) == -20037508, (w, s)
        assert round(e) == 20037508 and round(n) == 20037508, (e, n)

    def test_zoom1_nw_quadrant(self):
        """Zoom 1 tile (0,0) is the north-west quadrant.

        Test scenario:
            east and south meet at the origin (0, 0).
        """
        w, s, e, n = _xyz_bounds_3857(1, 0, 0)
        assert round(e) == 0 and round(s) == 0, (e, s)
        assert round(w) == -20037508 and round(n) == 20037508, (w, n)


class TestReadPart:
    """Tests for COG.read_part."""

    def test_full_window_native_size(self, ramp_4326):
        """Reading the full bbox at native size returns the full array.

        Args:
            ramp_4326: Fixture ramp Dataset.

        Test scenario:
            The dataset bbox at no explicit output size yields (100, 100).
        """
        arr = ramp_4326.read_part(tuple(ramp_4326.bbox), bbox_crs=4326, band=0)
        assert arr.shape == (100, 100), f"unexpected shape {arr.shape}"

    def test_decimated_output_shape(self, ramp_4326):
        """An explicit dst size decimates the read to that shape.

        Args:
            ramp_4326: Fixture ramp Dataset.

        Test scenario:
            Requesting 25x25 over the full bbox returns a 25x25 array.
        """
        arr = ramp_4326.read_part(
            tuple(ramp_4326.bbox), dst_width=25, dst_height=25, bbox_crs=4326, band=0
        )
        assert arr.shape == (25, 25), f"unexpected shape {arr.shape}"

    def test_all_bands_shape(self, ramp_4326):
        """Reading all bands of a single-band raster returns a 2-D array.

        Args:
            ramp_4326: Fixture ramp Dataset (single band).

        Test scenario:
            band=None reads every band; GDAL's ReadAsArray collapses a
            single-band read to 2-D (rows, cols) — a multiband source would
            instead yield (bands, rows, cols).
        """
        arr = ramp_4326.read_part(tuple(ramp_4326.bbox), bbox_crs=4326)
        assert arr.shape == (100, 100), f"unexpected shape {arr.shape}"

    def test_invalid_resampling_raises(self, ramp_4326):
        """An unknown resampling name raises ValueError.

        Args:
            ramp_4326: Fixture ramp Dataset.

        Test scenario:
            resampling='bogus' is rejected before any read.
        """
        with pytest.raises(ValueError, match="unknown resampling"):
            ramp_4326.read_part(tuple(ramp_4326.bbox), resampling="bogus")

    def test_non_intersecting_bbox_raises(self, ramp_4326):
        """A bbox outside the raster extent raises OutOfBoundsError.

        Args:
            ramp_4326: Fixture ramp Dataset (lon 0..1, lat 9..10).

        Test scenario:
            A far-away bbox does not intersect the raster.
        """
        with pytest.raises(OutOfBoundsError):
            ramp_4326.read_part((50.0, 50.0, 51.0, 51.0), bbox_crs=4326)


class TestPreview:
    """Tests for COG.preview."""

    def test_max_size_long_edge(self, ramp_4326):
        """preview caps the long edge at max_size.

        Args:
            ramp_4326: Fixture ramp Dataset (100x100).

        Test scenario:
            max_size=40 on a square raster yields a 40x40 thumbnail.
        """
        thumb = ramp_4326.preview(max_size=40, band=0)
        assert max(thumb.shape) == 40, f"unexpected shape {thumb.shape}"

    def test_small_raster_not_upsampled(self, ramp_4326):
        """A raster already smaller than max_size is returned at native size.

        Args:
            ramp_4326: Fixture ramp Dataset (100x100).

        Test scenario:
            max_size=512 leaves the 100x100 raster unchanged.
        """
        thumb = ramp_4326.preview(max_size=512, band=0)
        assert thumb.shape == (100, 100), f"unexpected shape {thumb.shape}"


class TestPoint:
    """Tests for COG.point."""

    def test_samples_expected_value(self, ramp_4326):
        """point samples the ramp value at a known pixel.

        Args:
            ramp_4326: Fixture ramp Dataset where value == row*100 + col.

        Test scenario:
            A coordinate at the centre of pixel (col=20, row=30) returns
            30*100 + 20 == 3020.
        """
        x = (20 + 0.5) * 0.01
        y = 10.0 - (30 + 0.5) * 0.01
        value = ramp_4326.point(x, y, point_crs=4326, band=0)
        assert float(value) == pytest.approx(3020.0), f"got {value}"

    def test_all_bands_returns_vector(self, ramp_4326):
        """point with band=None returns a 1-D per-band vector.

        Args:
            ramp_4326: Fixture ramp Dataset (single band).

        Test scenario:
            A valid coordinate yields a length-1 array (one band).
        """
        x = (10 + 0.5) * 0.01
        y = 10.0 - (10 + 0.5) * 0.01
        vec = ramp_4326.point(x, y, point_crs=4326)
        assert vec.shape == (1,), f"unexpected shape {vec.shape}"

    def test_out_of_bounds_raises(self, ramp_4326):
        """A coordinate outside the extent raises OutOfBoundsError.

        Args:
            ramp_4326: Fixture ramp Dataset.

        Test scenario:
            A far-away lon/lat is rejected.
        """
        with pytest.raises(OutOfBoundsError):
            ramp_4326.point(80.0, -10.0, point_crs=4326)

    def test_reprojects_point_crs(self, tile_3857):
        """point reprojects from point_crs to the dataset CRS.

        Args:
            tile_3857: Fixture EPSG:3857 dataset covering the NW quadrant.

        Test scenario:
            A 4326 lon/lat inside the NW quadrant (e.g. -90, 45) samples a
            finite value after reprojection to 3857.
        """
        value = tile_3857.point(-90.0, 45.0, point_crs=4326, band=0)
        assert np.isfinite(float(value)), f"expected a finite sample, got {value}"


class TestReadTile:
    """Tests for COG.read_tile."""

    def test_aligned_tile_shape(self, tile_3857):
        """Reading the aligned XYZ tile returns a tilesize square.

        Args:
            tile_3857: Fixture dataset aligned to zoom-1 tile (0,0).

        Test scenario:
            read_tile(1, 0, 0) over the matching 3857 dataset yields 256x256.
        """
        tile = tile_3857.read_tile(1, 0, 0, tilesize=256, band=0)
        assert tile.shape == (256, 256), f"unexpected shape {tile.shape}"

    def test_custom_tilesize(self, tile_3857):
        """read_tile honours a custom tilesize.

        Args:
            tile_3857: Fixture dataset aligned to zoom-1 tile (0,0).

        Test scenario:
            tilesize=128 yields a 128x128 tile.
        """
        tile = tile_3857.read_tile(1, 0, 0, tilesize=128, band=0)
        assert tile.shape == (128, 128), f"unexpected shape {tile.shape}"

    def test_non_overlapping_tile_raises(self, tile_3857):
        """A tile that does not overlap the raster raises OutOfBoundsError.

        Args:
            tile_3857: Fixture dataset covering only the NW quadrant.

        Test scenario:
            Tile (1, 1, 1) is the south-east quadrant and does not intersect.
        """
        with pytest.raises(OutOfBoundsError):
            tile_3857.read_tile(1, 1, 1, tilesize=256, band=0)
