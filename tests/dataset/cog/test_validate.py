"""Unit tests for pyramids.dataset.cog.validate."""

from __future__ import annotations

import pytest
from osgeo import gdal

from pyramids.dataset.cog.validate import (
    ValidationReport,
    _fallback_validate,
    _osgeo_validate,
    validate,
)
from pyramids.dataset.cog.write import translate_to_cog

pytestmark = pytest.mark.core

# --- helper: write a plain stripped GTiff that's large enough to fail COG check ---


def _write_plain_stripped_gtiff(path, size: int = 2048) -> None:
    """Write a plain non-tiled, no-overview GTiff > 512px (invalid COG)."""
    ds = gdal.GetDriverByName("MEM").Create("", size, size, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((0, 0.01, 0, 0, 0, -0.01))
    gdal.GetDriverByName("GTiff").CreateCopy(str(path), ds, 0)
    ds = None


# --- tests ---


class TestValidationReport:
    def test_bool_true_when_valid(self):
        r = ValidationReport(is_valid=True)
        assert bool(r) is True

    def test_bool_false_when_invalid(self):
        r = ValidationReport(is_valid=False, errors=["e"])
        assert bool(r) is False

    def test_defaults(self):
        r = ValidationReport(is_valid=True)
        assert r.errors == []
        assert r.warnings == []
        assert r.details == {}

    def test_frozen(self):
        r = ValidationReport(is_valid=True)
        with pytest.raises(Exception):  # dataclass(frozen=True) -> FrozenInstanceError
            r.is_valid = False  # type: ignore[misc]


class TestValidate:
    def test_valid_cog(self, mem_dataset, tmp_path):
        p = tmp_path / "valid.tif"
        dst = translate_to_cog(mem_dataset, p, {"COMPRESS": "DEFLATE"})
        dst.FlushCache()
        dst = None

        report = validate(p)
        assert report.is_valid is True
        assert report.errors == []
        assert isinstance(report.details, dict)

    def test_valid_cog_report_is_truthy(self, mem_dataset, tmp_path):
        p = tmp_path / "valid.tif"
        dst = translate_to_cog(mem_dataset, p, {})
        dst.FlushCache()
        dst = None

        assert bool(validate(p)) is True

    def test_invalid_stripped_gtiff(self, tmp_path):
        p = tmp_path / "plain.tif"
        _write_plain_stripped_gtiff(p, size=2048)

        report = validate(p)
        assert report.is_valid is False
        assert any("tiled" in e.lower() or "strip" in e.lower() for e in report.errors)

    def test_tiled_no_overviews_warns(self, tmp_path):
        """A tiled TIFF without overviews: warning, not error."""
        ds = gdal.GetDriverByName("MEM").Create("", 2048, 2048, 1, gdal.GDT_Byte)
        ds.SetGeoTransform((0, 0.01, 0, 0, 0, -0.01))
        p = tmp_path / "tiled_no_ovr.tif"
        gdal.GetDriverByName("GTiff").CreateCopy(
            str(p),
            ds,
            0,
            options=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"],
        )
        ds = None

        report = validate(p)
        # Either passes with a warning, or is invalid because of overviews.
        # At minimum: non-empty warnings OR a soft error mentioning overviews.
        has_ovr_msg = any(
            "overview" in m.lower() for m in (report.warnings + report.errors)
        )
        assert has_ovr_msg

    def test_strict_promotes_warnings_to_errors(self, tmp_path):
        """strict=True turns warnings into errors."""
        ds = gdal.GetDriverByName("MEM").Create("", 2048, 2048, 1, gdal.GDT_Byte)
        ds.SetGeoTransform((0, 0.01, 0, 0, 0, -0.01))
        p = tmp_path / "tiled_no_ovr.tif"
        gdal.GetDriverByName("GTiff").CreateCopy(
            str(p),
            ds,
            0,
            options=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"],
        )
        ds = None

        loose = validate(p, strict=False)
        strict = validate(p, strict=True)
        if loose.warnings:
            # warnings moved into errors under strict
            assert len(strict.errors) >= len(loose.errors)
            assert strict.warnings == []
            assert strict.is_valid is False

    def test_file_not_found_local(self, tmp_path):
        missing = tmp_path / "does_not_exist.tif"
        with pytest.raises(FileNotFoundError):
            validate(missing)

    def test_vsi_path_uses_vsistatl_not_path_exists(self, monkeypatch):
        """VSI paths are pre-checked via gdal.VSIStatL, NOT Path.exists.

        Test scenario:
            Post-H1 contract: `_raise_if_missing` (called from
            `validate`) uses :func:`gdal.VSIStatL` for `/vsi*`
            paths and :func:`pathlib.Path.exists` only for local
            paths. Confirm that `Path.exists` is never consulted
            for a `/vsi*` path even when VSIStatL reports the
            target as missing.
        """
        from pathlib import Path as _Path

        path_exists_calls: list[str] = []
        original_exists = _Path.exists

        def spy_exists(self):
            path_exists_calls.append(str(self))
            return original_exists(self)

        monkeypatch.setattr(_Path, "exists", spy_exists)

        # Force VSIStatL to report "missing" without any network I/O.
        from osgeo import gdal as gdal_mod

        monkeypatch.setattr(gdal_mod, "VSIStatL", lambda p: None)

        with pytest.raises(FileNotFoundError):
            validate("/vsicurl/https://127.0.0.1:1/nope.tif")

        vsi_calls = [
            p
            for p in path_exists_calls
            if p.startswith("/vsi") or p.startswith("\\vsi")
        ]
        assert vsi_calls == [], (
            f"Path.exists must not be called on /vsi* paths; "
            f"observed calls: {vsi_calls}"
        )

    def test_accepts_str_and_path(self, mem_dataset, tmp_path):
        p = tmp_path / "x.tif"
        dst = translate_to_cog(mem_dataset, p, {})
        dst.FlushCache()
        dst = None

        assert validate(str(p)).is_valid is True
        assert validate(p).is_valid is True


class TestOsgeoValidate:
    def test_returns_triple(self, mem_dataset, tmp_path):
        p = tmp_path / "x.tif"
        dst = translate_to_cog(mem_dataset, p, {})
        dst.FlushCache()
        dst = None

        errors, warnings, details = _osgeo_validate(str(p))
        assert isinstance(errors, list)
        assert isinstance(warnings, list)
        assert isinstance(details, dict)

    def test_file_not_found_translates_to_oserror(self, tmp_path):
        """Missing file raises FileNotFoundError from the pre-check.

        Test scenario:
            After the H1 refactor, file-existence is checked up-front
            via _raise_if_missing (locale-independent), not by
            substring-matching GDAL's error message.
        """
        missing = tmp_path / "nonexistent_file_xyz_12345.tif"
        with pytest.raises(FileNotFoundError) as exc_info:
            _osgeo_validate(str(missing))
        assert str(missing) in str(
            exc_info.value
        ), f"FileNotFoundError must name the missing path; got: {exc_info.value}"


class TestRaiseIfMissing:
    """Tests for the H1 _raise_if_missing helper (locale-independent FNF)."""

    def test_existing_local_file_silent(self, tmp_path):
        """Existing local file returns silently.

        Test scenario:
            A real file on disk must not raise.
        """
        from pyramids.dataset.cog.validate import _raise_if_missing

        p = tmp_path / "real.txt"
        p.write_text("hello")
        _raise_if_missing(str(p))

    def test_missing_local_file_raises(self, tmp_path):
        """Missing local file raises FileNotFoundError.

        Test scenario:
            Path.exists() returns False -> FileNotFoundError named
            with the input path.
        """
        from pyramids.dataset.cog.validate import _raise_if_missing

        missing = tmp_path / "nope.tif"
        with pytest.raises(FileNotFoundError, match="nope.tif"):
            _raise_if_missing(str(missing))

    def test_existing_vsimem_file_silent(self):
        """Existing /vsimem/ file returns silently.

        Test scenario:
            Write a file into the GDAL in-memory filesystem and confirm
            the VSIStatL path accepts it.
        """
        from osgeo import gdal

        from pyramids.dataset.cog.validate import _raise_if_missing

        p = "/vsimem/raise_if_missing_test.bin"
        gdal.FileFromMemBuffer(p, b"x" * 64)
        try:
            _raise_if_missing(p)
        finally:
            gdal.Unlink(p)

    def test_missing_vsi_path_raises(self):
        """Non-existent /vsi* path raises FileNotFoundError.

        Test scenario:
            VSIStatL returns None for a non-existent VSI path.
        """
        from pyramids.dataset.cog.validate import _raise_if_missing

        with pytest.raises(FileNotFoundError, match="unreachable"):
            _raise_if_missing("/vsimem/unreachable_xyz_12345.tif")


class TestFallbackValidate:
    def test_on_real_cog(self, mem_dataset, tmp_path):
        p = tmp_path / "x.tif"
        dst = translate_to_cog(mem_dataset, p, {})
        dst.FlushCache()
        dst = None
        errors, warnings, details = _fallback_validate(str(p))
        assert errors == []
        assert "blocksize" in details

    def test_on_stripped_gtiff(self, tmp_path):
        p = tmp_path / "plain.tif"
        _write_plain_stripped_gtiff(p, size=2048)
        errors, warnings, details = _fallback_validate(str(p))
        assert any("tiled" in e or "strip" in e for e in errors)


class TestValidateCoverageFill:
    """Scenarios that cover specific branch / error paths in validate.py."""

    def test_osgeo_validate_exception_is_not_file_not_found(
        self, monkeypatch, mem_dataset, tmp_path
    ):
        """A ValidateCloudOptimizedGeoTIFFException unrelated to a
        missing file becomes a non-fatal error entry (not an exception)."""
        from osgeo_utils.samples import validate_cloud_optimized_geotiff as v

        # Write a valid COG first so the Path.exists() pre-check passes.
        p = tmp_path / "x.tif"
        dst = translate_to_cog(mem_dataset, p, {})
        dst.FlushCache()
        dst = None

        def fake_validate(*args, **kwargs):
            raise v.ValidateCloudOptimizedGeoTIFFException("some non-file error")

        monkeypatch.setattr(v, "validate", fake_validate)
        report = validate(p)
        assert report.is_valid is False
        assert any("some non-file error" in e for e in report.errors)

    def test_osgeo_validate_runtime_error_non_fnf(
        self, monkeypatch, mem_dataset, tmp_path
    ):
        """A RuntimeError not about a missing file returns an error entry."""
        from osgeo_utils.samples import validate_cloud_optimized_geotiff as v

        p = tmp_path / "x.tif"
        dst = translate_to_cog(mem_dataset, p, {})
        dst.FlushCache()
        dst = None

        def fake_validate(*args, **kwargs):
            raise RuntimeError("some other error")

        monkeypatch.setattr(v, "validate", fake_validate)
        report = validate(p)
        assert report.is_valid is False
        assert any("some other error" in e for e in report.errors)

    def test_fallback_validate_gdal_open_returns_none(self, monkeypatch, tmp_path):
        """_fallback_validate reports "cannot open" when gdal.Open returns None."""
        import osgeo.gdal as gdal_mod

        from pyramids.dataset.cog.validate import _fallback_validate

        # Create a placeholder file so the validate() pre-check passes
        p = tmp_path / "not_a_tiff.txt"
        p.write_text("garbage")

        # Force gdal.Open to return None
        monkeypatch.setattr(gdal_mod, "Open", lambda *a, **kw: None)
        errors, warnings, details = _fallback_validate(str(p))
        assert any("cannot open" in e for e in errors)
