"""Tests for pyramids.dataset.cog.inspect (cog_info / COGInfo / OverviewLevel).

Covers the GDAL-only structured COG inspection: the profile fields
(compression, predictor, blocksize, dtype, CRS/bounds/resolution), the overview
pyramid, band tags, colour-table detection, the ``is_cog`` flag (delegated to
the validator), and the ``COG.info`` / ``Dataset.cog_info`` engine facade.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset
from pyramids.dataset.cog import COGInfo, OverviewLevel, cog_info

pytestmark = pytest.mark.core

_GEOTRANSFORM = (0.0, 0.01, 0.0, 10.0, 0.0, -0.01)


@pytest.fixture
def big_float_cog(tmp_path) -> str:
    """A 600x600 float32 COG on disk (large enough to carry overviews).

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written COG.
    """
    rng = np.random.default_rng(seed=11)
    arr = (rng.random((600, 600)) * 100.0).astype("float32")
    ds = Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)
    out = ds.to_cog(tmp_path / "big.tif")
    return str(out)


@pytest.fixture
def plain_geotiff(tmp_path) -> str:
    """A large untiled (stripped) plain GeoTIFF (not a COG).

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written plain GeoTIFF.

    Note:
        The sample COG validator is lenient about stripped/un-tiled layout
        (it only *warns*), so to get a definitive non-COG we attach an
        **external** overview sidecar (``.ovr``) — which the validator
        rejects as an error.
    """
    path = str(tmp_path / "plain.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 600, 600, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(_GEOTRANSFORM)
    ds.GetRasterBand(1).WriteArray(np.ones((600, 600), dtype="float32"))
    ds.FlushCache()
    ds = None
    # Read-only BuildOverviews writes an external .ovr sidecar, which a COG
    # must not have — guarantees the validator reports an error.
    ovr_ds = gdal.Open(path, gdal.GA_ReadOnly)
    ovr_ds.BuildOverviews("NEAREST", [2, 4])
    ovr_ds = None
    return path


class TestCogInfo:
    """Tests for the cog_info function."""

    def test_profile_fields_on_valid_cog(self, big_float_cog):
        """cog_info reports the band/geo profile of a valid COG.

        Args:
            big_float_cog: Fixture path to a 600x600 float COG.

        Test scenario:
            Compression DEFLATE, predictor 3 (float), 512 tiles, Float32, EPSG
            4326, single band, and is_cog True.
        """
        info = cog_info(big_float_cog)
        assert info.is_cog is True, f"expected a valid COG, errors via validate"
        assert info.driver == "GTiff", f"unexpected driver {info.driver}"
        assert (
            info.compression == "DEFLATE"
        ), f"unexpected compression {info.compression}"
        assert (
            info.predictor == "3"
        ), f"float COG should have predictor 3, got {info.predictor}"
        assert info.blocksize == (512, 512), f"unexpected blocksize {info.blocksize}"
        assert info.dtype == "Float32", f"unexpected dtype {info.dtype}"
        assert info.crs_epsg == 4326, f"unexpected epsg {info.crs_epsg}"
        assert info.band_count == 1, f"unexpected band count {info.band_count}"

    def test_resolution_and_bounds(self, big_float_cog):
        """cog_info derives resolution and bounds from the geotransform.

        Args:
            big_float_cog: Fixture path to a 600x600 float COG.

        Test scenario:
            Pixel size (0.01, 0.01); bounds span 600 px * 0.01 = 6.0 from the
            top-left origin (0, 10).
        """
        info = cog_info(big_float_cog)
        assert info.resolution == pytest.approx((0.01, 0.01)), info.resolution
        min_x, min_y, max_x, max_y = info.bounds
        assert min_x == pytest.approx(0.0), f"min_x {min_x}"
        assert max_x == pytest.approx(6.0), f"max_x {max_x}"
        assert max_y == pytest.approx(10.0), f"max_y {max_y}"
        assert min_y == pytest.approx(4.0), f"min_y {min_y}"

    def test_overview_pyramid(self, big_float_cog):
        """cog_info enumerates the overview pyramid with decimation factors.

        Args:
            big_float_cog: Fixture path to a 600x600 float COG.

        Test scenario:
            A 600x600 COG with 512 tiles has at least one overview; the first
            level decimates by >= 2 and every entry is an OverviewLevel.
        """
        info = cog_info(big_float_cog)
        assert info.overview_count >= 1, "a 600px COG should carry overviews"
        first = info.overviews[0]
        assert isinstance(first, OverviewLevel), f"unexpected type {type(first)}"
        assert (
            first.decimation >= 2
        ), f"first overview should decimate >=2, got {first.decimation}"
        assert first.width < info.width, "overview must be smaller than full-res"

    def test_plain_geotiff_is_not_cog(self, plain_geotiff):
        """cog_info flags a plain untiled GeoTIFF as not a COG.

        Args:
            plain_geotiff: Fixture path to an untiled plain GeoTIFF.

        Test scenario:
            is_cog False for a GeoTIFF carrying an external .ovr sidecar
            (external overviews are forbidden in a COG).
        """
        info = cog_info(plain_geotiff)
        assert info.is_cog is False, "a GeoTIFF with external overviews is not a COG"

    def test_colormap_detection(self, tmp_path):
        """cog_info detects a colour table on band 1.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A Byte raster with a palette reports colormap True.
        """
        path = str(tmp_path / "palette.tif")
        ds = gdal.GetDriverByName("GTiff").Create(path, 16, 16, 1, gdal.GDT_Byte)
        ds.SetGeoTransform(_GEOTRANSFORM)
        ct = gdal.ColorTable()
        ct.SetColorEntry(0, (0, 0, 0, 255))
        ct.SetColorEntry(1, (255, 0, 0, 255))
        ds.GetRasterBand(1).SetColorTable(ct)
        ds.FlushCache()
        ds = None
        info = cog_info(path)
        assert info.colormap is True, "palette raster should report colormap True"

    def test_band_tags(self, tmp_path):
        """cog_info captures per-band metadata tags.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A band tag written to the source is reported under its 1-based key.
        """
        path = str(tmp_path / "tagged.tif")
        ds = gdal.GetDriverByName("GTiff").Create(path, 16, 16, 1, gdal.GDT_Float32)
        ds.SetGeoTransform(_GEOTRANSFORM)
        ds.GetRasterBand(1).SetMetadataItem("BAND_NAME", "NDVI")
        ds.FlushCache()
        ds = None
        info = cog_info(path)
        assert info.band_tags[1].get("BAND_NAME") == "NDVI", info.band_tags

    def test_missing_file_raises(self, tmp_path):
        """cog_info raises FileNotFoundError for an unopenable path.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A non-existent path cannot be opened by GDAL.
        """
        with pytest.raises(FileNotFoundError):
            cog_info(str(tmp_path / "nope.tif"))


class TestCogInfoFacade:
    """Tests for the COG.info engine method and Dataset.cog_info facade."""

    def test_dataset_cog_info_matches_function(self, big_float_cog):
        """Dataset.cog_info returns the same structured info as cog_info().

        Args:
            big_float_cog: Fixture path to a 600x600 float COG.

        Test scenario:
            Opening the COG and calling ds.cog_info() yields a COGInfo with the
            same compression and overview count as the module-level function.
        """
        ds = Dataset.read_file(big_float_cog)
        info = ds.cog_info()
        assert isinstance(info, COGInfo), f"unexpected type {type(info)}"
        ref = cog_info(big_float_cog)
        assert info.compression == ref.compression, "compression mismatch"
        assert info.overview_count == ref.overview_count, "overview count mismatch"

    def test_mem_dataset_raises(self):
        """Dataset.cog_info raises for a MEM-only dataset with no backing file.

        Test scenario:
            An in-memory Dataset has no on-disk file to inspect.
        """
        arr = np.ones((8, 8), dtype="float32")
        ds = Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)
        with pytest.raises(FileNotFoundError):
            ds.cog_info()
