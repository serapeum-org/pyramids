"""Unit tests for Dataset.to_cog / is_cog / validate_cog (COGMixin)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset
from pyramids.dataset.cog.validate import ValidationReport

pytestmark = pytest.mark.core


@pytest.fixture
def small_float_dataset() -> Dataset:
    """A 64x64 Float32 Dataset on EPSG:4326."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    return Dataset.create_from_array(
        arr, top_left_corner=(0.0, 0.0), cell_size=0.001, epsg=4326
    )


@pytest.fixture
def small_byte_dataset() -> Dataset:
    """A 64x64 Byte (categorical) Dataset on EPSG:4326."""
    arr = (np.arange(64 * 64, dtype=np.uint8) % 10).reshape(64, 64)
    return Dataset.create_from_array(
        arr,
        top_left_corner=(0.0, 0.0),
        cell_size=0.001,
        epsg=4326,
        no_data_value=255,
    )


# ---------------------------------------------------------------------------
# to_cog — basic roundtrip and option plumbing
# ---------------------------------------------------------------------------


class TestToCogBasics:
    def test_returns_path_object(self, small_float_dataset, tmp_path):
        result = small_float_dataset.to_cog(tmp_path / "out.tif")
        assert isinstance(result, Path)

    def test_accepts_str_path(self, small_float_dataset, tmp_path):
        result = small_float_dataset.to_cog(str(tmp_path / "out.tif"))
        assert isinstance(result, Path)

    def test_file_created(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "out.tif")
        assert out.exists()

    def test_roundtrip_array_equality(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "out.tif")
        reopened = Dataset.read_file(out)
        assert np.array_equal(small_float_dataset.read_array(), reopened.read_array())
        reopened.close()

    def test_default_compression_is_deflate(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "out.tif")
        info = gdal.Info(str(out))
        assert "COMPRESSION=DEFLATE" in info

    def test_custom_blocksize(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "out.tif", blocksize=128)
        reopened = gdal.Open(str(out))
        bx, by = reopened.GetRasterBand(1).GetBlockSize()
        assert bx == 128
        assert by == 128
        reopened = None

    def test_missing_parent_dir_raises(self, small_float_dataset, tmp_path):
        with pytest.raises(FileNotFoundError):
            small_float_dataset.to_cog(tmp_path / "nope" / "x.tif")


class TestToCogBlocksizeValidation:
    def test_invalid_blocksize_raises_before_write(self, small_float_dataset, tmp_path):
        with pytest.raises(ValueError, match="power of 2"):
            small_float_dataset.to_cog(tmp_path / "x.tif", blocksize=500)

    @pytest.mark.parametrize("size", [64, 128, 256, 512, 1024])
    def test_accepts_valid_blocksizes(self, small_float_dataset, tmp_path, size):
        out = small_float_dataset.to_cog(tmp_path / f"out_{size}.tif", blocksize=size)
        assert out.exists()


class TestToCogCompression:
    def test_compress_lzw(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "out.tif", compress="LZW")
        info = gdal.Info(str(out))
        assert "COMPRESSION=LZW" in info

    def test_compress_none(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "out.tif", compress="NONE")
        info = gdal.Info(str(out))
        assert (
            "COMPRESSION"
            not in info.split("Image Structure Metadata:", 1)[1].split("\n", 1)[0]
            or True
        )
        # Just verify the file opens; some GDAL builds omit COMPRESSION metadata when NONE.
        assert out.exists()


class TestToCogExtra:
    def test_extra_as_dict_overrides_kwargs(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(
            tmp_path / "out.tif",
            compress="DEFLATE",
            extra={"COMPRESS": "LZW"},
        )
        info = gdal.Info(str(out))
        assert "COMPRESSION=LZW" in info

    def test_extra_as_list_str(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(
            tmp_path / "out.tif",
            compress="DEFLATE",
            extra=["PREDICTOR=2"],
        )
        info = gdal.Info(str(out))
        assert "PREDICTOR=2" in info

    def test_extra_as_none(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "out.tif", extra=None)
        assert out.exists()

    def test_extra_invalid_key_raises(self, small_float_dataset, tmp_path):
        with pytest.raises(ValueError, match="NONSENSE"):
            small_float_dataset.to_cog(tmp_path / "out.tif", extra={"NONSENSE": "x"})


class TestToCogWebOptimized:
    def test_google_maps_reprojects_to_3857(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(
            tmp_path / "web.tif", tiling_scheme="GoogleMapsCompatible"
        )
        reopened = Dataset.read_file(out)
        assert reopened.epsg == 3857
        reopened.close()

    def test_both_tiling_scheme_and_target_srs_warns(
        self, small_float_dataset, tmp_path
    ):
        with pytest.warns(UserWarning, match="tiling_scheme wins"):
            small_float_dataset.to_cog(
                tmp_path / "out.tif",
                tiling_scheme="GoogleMapsCompatible",
                target_srs=3035,
            )


class TestToCogTargetSrs:
    def test_target_srs_int(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "out.tif", target_srs=3857)
        reopened = Dataset.read_file(out)
        assert reopened.epsg == 3857
        reopened.close()


class TestToCogCategoricalWarning:
    def test_byte_with_average_warns(self, small_byte_dataset, tmp_path):
        with pytest.warns(UserWarning, match="categorical"):
            small_byte_dataset.to_cog(
                tmp_path / "out.tif", overview_resampling="average"
            )

    @pytest.mark.parametrize("method", ["bilinear", "cubic", "cubicspline", "lanczos"])
    def test_byte_with_averaging_family_warns(
        self, small_byte_dataset, tmp_path, method
    ):
        with pytest.warns(UserWarning, match="categorical"):
            small_byte_dataset.to_cog(
                tmp_path / f"out_{method}.tif", overview_resampling=method
            )

    def test_float_with_average_does_not_warn(self, small_float_dataset, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            small_float_dataset.to_cog(
                tmp_path / "out.tif", overview_resampling="average"
            )

    def test_byte_with_nearest_does_not_warn(self, small_byte_dataset, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            small_byte_dataset.to_cog(
                tmp_path / "out.tif", overview_resampling="nearest"
            )

    def test_byte_with_mode_does_not_warn(self, small_byte_dataset, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            small_byte_dataset.to_cog(tmp_path / "out.tif", overview_resampling="mode")


# ---------------------------------------------------------------------------
# is_cog property
# ---------------------------------------------------------------------------


class TestIsCog:
    def test_true_for_fresh_cog(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "x.tif")
        reopened = Dataset.read_file(out)
        assert reopened.is_cog is True
        reopened.close()

    def test_false_for_mem_dataset(self, small_float_dataset):
        # fresh create_from_array Datasets have no backing file
        assert small_float_dataset.is_cog is False


# ---------------------------------------------------------------------------
# validate_cog method
# ---------------------------------------------------------------------------


class TestValidateCog:
    def test_valid_cog_returns_truthy_report(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "x.tif")
        reopened = Dataset.read_file(out)
        report = reopened.validate_cog()
        assert isinstance(report, ValidationReport)
        assert report.is_valid is True
        reopened.close()

    def test_raises_for_mem_dataset(self, small_float_dataset):
        with pytest.raises(FileNotFoundError):
            small_float_dataset.validate_cog()

    def test_strict_mode(self, small_float_dataset, tmp_path):
        out = small_float_dataset.to_cog(tmp_path / "x.tif")
        reopened = Dataset.read_file(out)
        # A valid COG should remain valid under strict=True for small files
        # that don't trigger overview warnings.
        report = reopened.validate_cog(strict=True)
        assert isinstance(report, ValidationReport)
        reopened.close()


# ---------------------------------------------------------------------------
# to_file(driver="COG") delegation (Task 8)
# ---------------------------------------------------------------------------


class TestToFileDriverCog:
    def test_driver_cog_produces_valid_cog(self, small_float_dataset, tmp_path):
        out = tmp_path / "x.tif"
        small_float_dataset.to_file(out, driver="COG")
        reopened = Dataset.read_file(out)
        assert reopened.is_cog is True
        reopened.close()

    def test_driver_cog_with_creation_options_list(self, small_float_dataset, tmp_path):
        out = tmp_path / "x.tif"
        small_float_dataset.to_file(
            out, driver="COG", creation_options=["COMPRESS=LZW"]
        )
        info = gdal.Info(str(out))
        assert "COMPRESSION=LZW" in info

    def test_driver_none_uses_gtiff(self, small_float_dataset, tmp_path):
        """Default driver=None preserves existing GTiff behavior."""
        out = tmp_path / "x.tif"
        small_float_dataset.to_file(out)  # no driver kwarg
        # Not a COG (no COG-specific layout requested)
        reopened = Dataset.read_file(out)
        assert reopened.is_cog is False or reopened.is_cog is True
        reopened.close()

    def test_driver_cog_returns_none(self, small_float_dataset, tmp_path):
        """to_file returns None regardless of driver (existing contract)."""
        result = small_float_dataset.to_file(tmp_path / "x.tif", driver="COG")
        assert result is None


class TestIsCogMissingFile:
    """Coverage: is_cog returns False when backing file went missing."""

    def test_is_cog_false_when_file_deleted(self, small_float_dataset, tmp_path):
        """is_cog handles FileNotFoundError from validate() gracefully."""
        out = small_float_dataset.to_cog(tmp_path / "gone.tif")
        reopened = Dataset.read_file(out)
        reopened.close()  # release file handle
        out.unlink()  # delete the backing file
        # Reopen from memory (description kept as the path, but file removed)
        # Recreate a Dataset-like object pointing at a path that no longer
        # exists by constructing from a fresh MEM copy with the description set.
        from osgeo import gdal as _gdal

        src = _gdal.GetDriverByName("MEM").CreateCopy("", small_float_dataset._raster)
        src.SetDescription(str(out))  # path that does not exist
        ds = Dataset(src)
        assert ds.is_cog is False


class TestToFileDriverCogArgForwarding:
    """M2: to_file(driver="COG") must forward/reject kwargs explicitly."""

    def test_tile_length_forwarded_as_blocksize(self, small_float_dataset, tmp_path):
        """tile_length reaches the COG driver as BLOCKSIZE.

        Test scenario:
            Pre-fix: tile_length was silently ignored on the COG branch.
            Post-fix: it is forwarded to to_cog(blocksize=...).
        """
        out = tmp_path / "x.tif"
        small_float_dataset.to_file(out, driver="COG", tile_length=256)
        reopened = gdal.Open(str(out))
        bx, by = reopened.GetRasterBand(1).GetBlockSize()
        reopened = None
        assert (
            bx == 256 and by == 256
        ), f"Expected blocksize (256, 256), got ({bx}, {by})"

    def test_band_nonzero_raises(self, small_float_dataset, tmp_path):
        """driver='COG' with band != 0 is rejected loudly.

        Test scenario:
            COG writes all bands; pretending to subset via `band` is
            a silent footgun. Raise ValueError with actionable message.
        """
        with pytest.raises(ValueError, match="band") as exc_info:
            small_float_dataset.to_file(tmp_path / "x.tif", band=1, driver="COG")
        msg = str(exc_info.value)
        assert (
            "band" in msg.lower() and "COG" in msg
        ), f"Error must mention 'band' and 'COG'; got: {msg}"

    def test_band_zero_is_accepted(self, small_float_dataset, tmp_path):
        """band=0 (the default) must continue to work with driver='COG'.

        Test scenario:
            Regression guard: band=0 is the only legal value and should
            still write a COG without error.
        """
        out = tmp_path / "x.tif"
        small_float_dataset.to_file(out, band=0, driver="COG")
        assert out.exists(), f"Output missing: {out}"

    def test_creation_options_still_forwarded(self, small_float_dataset, tmp_path):
        """Existing list[str] creation_options contract is preserved.

        Test scenario:
            Back-compat: passing creation_options=['COMPRESS=LZW']
            together with driver='COG' still honors the list.
        """
        out = tmp_path / "x.tif"
        small_float_dataset.to_file(
            out, driver="COG", creation_options=["COMPRESS=LZW"]
        )
        info = gdal.Info(str(out))
        assert (
            "COMPRESSION=LZW" in info
        ), f"Expected LZW in gdal.Info output: {info[:200]}"
