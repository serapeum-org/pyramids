"""End-to-end / integration tests for the COG feature (Task 11).

Covers the public user journey:

1. :meth:`Dataset.create_from_array` -> :meth:`Dataset.to_cog` ->
   :meth:`Dataset.read_file` round-trip array equality.
2. Every supported compression type (skip if unavailable in the build).
3. CRS, no-data, and multi-band preservation across a round trip.
4. Web-optimized COG (GoogleMapsCompatible tiling scheme).
5. LERC lossy compression tolerance.
6. Validator agreement with :attr:`Dataset.is_cog`.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture
def float_dataset_128() -> Dataset:
    arr = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    return Dataset.create_from_array(
        arr, top_left_corner=(0.0, 0.0), cell_size=0.001, epsg=4326
    )


@pytest.fixture
def multiband_dataset() -> Dataset:
    rng = np.random.default_rng(42)
    arr = rng.random((3, 64, 64), dtype=np.float32)
    return Dataset.create_from_array(
        arr, top_left_corner=(0.0, 0.0), cell_size=0.001, epsg=4326
    )


@pytest.fixture
def compression_support() -> set[str]:
    """Compression algorithms available in the current GTiff driver."""
    meta = gdal.GetDriverByName("GTiff").GetMetadataItem("DMD_CREATIONOPTIONLIST") or ""
    algos = set()
    for alg in [
        "NONE",
        "LZW",
        "DEFLATE",
        "ZSTD",
        "WEBP",
        "LERC",
        "LERC_DEFLATE",
        "LERC_ZSTD",
        "JPEG",
    ]:
        if f">{alg}<" in meta:
            algos.add(alg)
    return algos


class TestCanonicalRoundtrip:
    """The acceptance-criteria round-trip from the plan."""

    def test_array_equality_float(self, float_dataset_128, tmp_path):
        ds = float_dataset_128
        out = ds.to_cog(tmp_path / "rt.tif")
        reopened = Dataset.read_file(out)
        assert reopened.is_cog is True
        assert np.array_equal(ds.read_array(), reopened.read_array())
        reopened.close()

    def test_preserves_crs(self, float_dataset_128, tmp_path):
        ds = float_dataset_128
        out = ds.to_cog(tmp_path / "rt.tif")
        reopened = Dataset.read_file(out)
        assert reopened.epsg == 4326
        reopened.close()

    def test_preserves_dimensions(self, float_dataset_128, tmp_path):
        ds = float_dataset_128
        out = ds.to_cog(tmp_path / "rt.tif")
        reopened = Dataset.read_file(out)
        assert reopened.rows == 128
        assert reopened.columns == 128
        reopened.close()


class TestMultiBandRoundtrip:
    def test_band_count_preserved(self, multiband_dataset, tmp_path):
        out = multiband_dataset.to_cog(tmp_path / "mb.tif")
        reopened = Dataset.read_file(out)
        assert reopened.band_count == 3
        reopened.close()

    def test_per_band_array_equality(self, multiband_dataset, tmp_path):
        out = multiband_dataset.to_cog(tmp_path / "mb.tif")
        reopened = Dataset.read_file(out)
        orig = multiband_dataset.read_array()
        got = reopened.read_array()
        assert orig.shape == got.shape
        assert np.allclose(orig, got)
        reopened.close()


class TestEveryCompression:
    @pytest.mark.parametrize("method", ["DEFLATE", "LZW", "ZSTD", "NONE", "LERC"])
    def test_round_trip(self, float_dataset_128, tmp_path, compression_support, method):
        if method not in compression_support:
            pytest.skip(f"GDAL build lacks {method}")
        out = float_dataset_128.to_cog(tmp_path / f"{method}.tif", compress=method)
        reopened = Dataset.read_file(out)
        reopened_arr = reopened.read_array()
        if method == "LERC":
            # Default LERC max_z_error=0 -> near-lossless
            assert np.allclose(float_dataset_128.read_array(), reopened_arr, atol=1e-3)
        elif method in {"WEBP", "JPEG"}:
            # Lossy — only compare shape
            assert reopened_arr.shape == float_dataset_128.read_array().shape
        else:
            assert np.array_equal(float_dataset_128.read_array(), reopened_arr)
        reopened.close()


class TestLercTolerance:
    def test_lerc_with_max_z_error(
        self, float_dataset_128, tmp_path, compression_support
    ):
        if "LERC" not in compression_support:
            pytest.skip("LERC unavailable")
        out = float_dataset_128.to_cog(
            tmp_path / "lerc.tif",
            compress="LERC",
            extra={"MAX_Z_ERROR": 0.5},
        )
        reopened = Dataset.read_file(out)
        # With MAX_Z_ERROR=0.5, round-trip error must be <= 0.5 per pixel
        diff = np.abs(
            float_dataset_128.read_array().astype(np.float64)
            - reopened.read_array().astype(np.float64)
        )
        assert diff.max() <= 0.5
        reopened.close()


class TestWebOptimized:
    def test_google_maps_compatible_is_web_mercator(self, float_dataset_128, tmp_path):
        out = float_dataset_128.to_cog(
            tmp_path / "web.tif", tiling_scheme="GoogleMapsCompatible"
        )
        reopened = Dataset.read_file(out)
        assert reopened.epsg == 3857
        assert reopened.is_cog is True
        reopened.close()

    def test_google_maps_with_blocksize_256(self, float_dataset_128, tmp_path):
        """GoogleMapsCompatible + explicit blocksize=256 honored."""
        out = float_dataset_128.to_cog(
            tmp_path / "web.tif",
            tiling_scheme="GoogleMapsCompatible",
            blocksize=256,
        )
        reopened = gdal.Open(str(out))
        bx, _ = reopened.GetRasterBand(1).GetBlockSize()
        assert bx == 256
        reopened = None


class TestValidatorAgreesWithIsCog:
    def test_cog_both_agree_true(self, float_dataset_128, tmp_path):
        out = float_dataset_128.to_cog(tmp_path / "x.tif")
        reopened = Dataset.read_file(out)
        report = reopened.validate_cog()
        assert reopened.is_cog == report.is_valid == True
        reopened.close()

    def test_gtiff_both_agree_false(self, float_dataset_128, tmp_path):
        """A large plain stripped GTiff: is_cog False AND validate fails."""
        # Make a large enough stripped GTiff to trigger validator errors
        big = np.arange(2048 * 2048, dtype=np.float32).reshape(2048, 2048)
        ds = Dataset.create_from_array(
            big, top_left_corner=(0.0, 0.0), cell_size=0.001, epsg=4326
        )
        out = tmp_path / "plain.tif"
        ds.to_file(out)  # default GTiff
        reopened = Dataset.read_file(out)
        report = reopened.validate_cog()
        assert reopened.is_cog == report.is_valid == False
        reopened.close()


class TestNoDataPreservation:
    def test_nodata_round_trips(self, tmp_path):
        arr = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
        arr[0:10, 0:10] = np.nan
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.001,
            epsg=4326,
            no_data_value=-1.0,
        )
        out = ds.to_cog(tmp_path / "nd.tif")
        reopened = Dataset.read_file(out)
        assert reopened.no_data_value[0] == pytest.approx(-1.0)
        reopened.close()


class TestPipelineWithReadFile:
    def test_reopen_via_path_string(self, float_dataset_128, tmp_path):
        out = float_dataset_128.to_cog(tmp_path / "x.tif")
        reopened = Dataset.read_file(str(out))
        assert reopened.is_cog is True
        reopened.close()

    def test_reopen_via_path_object(self, float_dataset_128, tmp_path):
        out = float_dataset_128.to_cog(tmp_path / "x.tif")
        reopened = Dataset.read_file(out)
        assert reopened.is_cog is True
        reopened.close()
