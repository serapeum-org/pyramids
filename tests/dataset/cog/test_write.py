"""Unit tests for pyramids.dataset.cog.write.translate_to_cog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import DriverNotExistError, FailedToSaveError
from pyramids.dataset.cog.write import translate_to_cog

pytestmark = pytest.mark.core


class TestRoundtrip:
    def test_defaults_write_and_reopen(self, mem_dataset, tmp_path):
        out = tmp_path / "out.tif"
        dst = translate_to_cog(mem_dataset, out, {"COMPRESS": "DEFLATE"})
        dst.FlushCache()
        dst = None

        assert out.exists()
        reopened = gdal.Open(str(out))
        assert reopened is not None
        assert reopened.RasterXSize == 512
        assert reopened.RasterYSize == 512
        # Verify array equality
        original_arr = mem_dataset.GetRasterBand(1).ReadAsArray()
        reopened_arr = reopened.GetRasterBand(1).ReadAsArray()
        assert np.array_equal(original_arr, reopened_arr)
        reopened = None

    def test_default_blocksize_is_512(self, mem_dataset, tmp_path):
        out = tmp_path / "out.tif"
        dst = translate_to_cog(mem_dataset, out, {"BLOCKSIZE": 512})
        dst.FlushCache()
        dst = None

        reopened = gdal.Open(str(out))
        bx, by = reopened.GetRasterBand(1).GetBlockSize()
        assert bx == 512
        assert by == 512
        reopened = None

    def test_custom_blocksize_256(self, mem_dataset, tmp_path):
        out = tmp_path / "out.tif"
        dst = translate_to_cog(mem_dataset, out, {"BLOCKSIZE": 256})
        dst.FlushCache()
        dst = None

        reopened = gdal.Open(str(out))
        bx, by = reopened.GetRasterBand(1).GetBlockSize()
        assert bx == 256
        assert by == 256
        reopened = None

    def test_compress_zstd_if_supported(
        self, mem_dataset, tmp_path, gtiff_compression_list
    ):
        if "ZSTD" not in gtiff_compression_list:
            pytest.skip("GDAL build lacks ZSTD support")
        out = tmp_path / "out.tif"
        dst = translate_to_cog(mem_dataset, out, {"COMPRESS": "ZSTD"})
        dst.FlushCache()
        dst = None
        assert out.exists()

    def test_compress_none_vs_deflate_size(self, mem_dataset, tmp_path):
        """NONE compression produces materially larger file than DEFLATE."""
        p_none = tmp_path / "none.tif"
        p_deflate = tmp_path / "deflate.tif"
        dst = translate_to_cog(mem_dataset, p_none, {"COMPRESS": "NONE"})
        dst.FlushCache()
        dst = None
        dst = translate_to_cog(mem_dataset, p_deflate, {"COMPRESS": "DEFLATE"})
        dst.FlushCache()
        dst = None
        # DEFLATE on a ramp array should substantially beat NONE.
        assert p_none.stat().st_size > p_deflate.stat().st_size


class TestValidation:
    def test_invalid_option_key_raises(self, mem_dataset, tmp_path):
        with pytest.raises(ValueError, match="NONSENSE"):
            translate_to_cog(mem_dataset, tmp_path / "out.tif", {"NONSENSE": "x"})

    def test_missing_parent_dir_raises(self, mem_dataset, tmp_path):
        target = tmp_path / "does_not_exist" / "out.tif"
        with pytest.raises(FileNotFoundError):
            translate_to_cog(mem_dataset, target, {})

    def test_driver_missing_raises(self, mem_dataset, tmp_path, monkeypatch):
        original = gdal.GetDriverByName

        def fake(name):
            result = None if name == "COG" else original(name)
            return result

        monkeypatch.setattr(gdal, "GetDriverByName", fake)
        with pytest.raises(DriverNotExistError):
            translate_to_cog(mem_dataset, tmp_path / "out.tif", {})


class TestSparseOk:
    def test_sparse_ok_yes_embedded(self, mem_dataset, tmp_path):
        out = tmp_path / "sparse.tif"
        dst = translate_to_cog(mem_dataset, out, {"SPARSE_OK": True})
        dst.FlushCache()
        dst = None
        # SPARSE_OK=YES affects storage; we just verify the file opens.
        reopened = gdal.Open(str(out))
        assert reopened is not None
        reopened = None


class TestReturnValue:
    def test_returns_gdal_dataset(self, mem_dataset, tmp_path):
        out = tmp_path / "out.tif"
        dst = translate_to_cog(mem_dataset, out, {})
        assert isinstance(dst, gdal.Dataset)
        dst.FlushCache()
        dst = None

    def test_accepts_str_and_path(self, mem_dataset, tmp_path):
        """Both str and Path accepted."""
        out_str = str(tmp_path / "a.tif")
        out_path = tmp_path / "b.tif"

        dst = translate_to_cog(mem_dataset, out_str, {})
        dst.FlushCache()
        dst = None

        dst = translate_to_cog(mem_dataset, out_path, {})
        dst.FlushCache()
        dst = None

        assert Path(out_str).exists()
        assert out_path.exists()


class TestFailedToSave:
    def test_createcopy_returns_none_raises(self, mem_dataset, tmp_path, monkeypatch):
        """If CreateCopy returns None we surface FailedToSaveError."""
        original_driver = gdal.GetDriverByName("COG")

        class FakeDriver:
            def CreateCopy(self, *a, **kw):  # noqa: N802
                return None

        def fake_get(name):
            return FakeDriver() if name == "COG" else original_driver

        monkeypatch.setattr(gdal, "GetDriverByName", fake_get)
        with pytest.raises(FailedToSaveError, match="returned None"):
            translate_to_cog(mem_dataset, tmp_path / "x.tif", {})


class TestCreateCopyRuntimeError:
    def test_runtime_error_wrapped_in_failed_to_save(
        self, mem_dataset, tmp_path, monkeypatch
    ):
        """GDAL CreateCopy RuntimeError is translated to FailedToSaveError."""
        original = gdal.GetDriverByName("COG")

        class FakeDriver:
            def CreateCopy(self, *a, **kw):  # noqa: N802
                raise RuntimeError("simulated write failure")

        def fake_get(name):
            return FakeDriver() if name == "COG" else original

        monkeypatch.setattr(gdal, "GetDriverByName", fake_get)
        from pyramids.base._errors import FailedToSaveError

        with pytest.raises(FailedToSaveError, match="simulated write failure"):
            translate_to_cog(mem_dataset, tmp_path / "x.tif", {})
