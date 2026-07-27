"""Tests for resource-handle hygiene in `pyramids._io` and `pyramids.io`.

Covers the three defects fixed on this branch: the zip file-descriptor leak and
the shared update-mode GDAL handle in `pyramids._io`, the leaked extraction
directory in `pyramids.io.sniff`, and the lazy `pyramids.io` exports.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids._io import _get_zip_path, read_file
from pyramids.base import _artifacts


@pytest.fixture(scope="function")
def asc_zip(tmp_path: Path) -> Path:
    """Create a zip holding two ASCII-grid members.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path: Path to the created `.zip` archive.
    """
    archive = tmp_path / "grids.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("1.asc", "ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999\n1\n")
        handle.writestr("2.asc", "ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999\n2\n")
    return archive


@pytest.fixture(scope="function")
def tiny_raster(tmp_path: Path) -> str:
    """Create a small on-disk GeoTIFF.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        str: Path to the created GeoTIFF.
    """
    path = str(tmp_path / "tiny.tif")
    dataset = gdal.GetDriverByName("GTiff").Create(path, 4, 4, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    dataset.GetRasterBand(1).WriteArray(np.zeros((4, 4), dtype="uint8"))
    dataset = None
    return path


class TestGetZipPath:
    """Tests for `_get_zip_path`."""

    def test_returns_first_member_vsi_path(self, asc_zip: Path):
        """`_get_zip_path` resolves a bare archive to its first member.

        Test scenario:
            A zip with `1.asc` and `2.asc` resolves to the `/vsizip/.../1.asc` path.
        """
        result = _get_zip_path(str(asc_zip))
        assert result.startswith("/vsizip/"), f"Expected a /vsizip/ path, got: {result}"
        assert result.endswith("/1.asc"), f"Expected the first member, got: {result}"

    def test_selects_member_by_index(self, asc_zip: Path):
        """`_get_zip_path` honours `file_i` when selecting the member.

        Test scenario:
            `file_i=1` resolves to the second member rather than the first.
        """
        result = _get_zip_path(str(asc_zip), file_i=1)
        assert result.endswith("/2.asc"), f"Expected the second member, got: {result}"

    def test_internal_path_is_only_prefixed(self, asc_zip: Path):
        """An archive path that already names a member is only prefixed.

        Test scenario:
            `<archive>.zip/1.asc` must not be re-opened; it is passed through with
            the `/vsizip/` prefix attached.
        """
        result = _get_zip_path(f"{asc_zip}/1.asc")
        assert result == f"/vsizip/{asc_zip}/1.asc", f"Unexpected passthrough result: {result}"

    def test_does_not_hold_the_archive_open(self, asc_zip: Path):
        """`_get_zip_path` closes the archive before returning.

        Test scenario:
            Listing members must not leak the file descriptor. On Windows an open
            handle blocks deletion, so a successful `os.remove` proves the handle
            was released rather than left to garbage collection.
        """
        _get_zip_path(str(asc_zip))
        os.remove(asc_zip)
        assert not asc_zip.exists(), "archive should be deletable, so its handle must already be closed"


class TestReadFileAccessMode:
    """Tests for GDAL handle sharing in `read_file`."""

    def test_read_only_uses_shared_handle(self, tiny_raster: str, monkeypatch: pytest.MonkeyPatch):
        """Read-only opens go through `gdal.OpenShared`.

        Test scenario:
            Handle reuse is the intended benefit for repeated read-only access, so
            the shared opener must still be used when `read_only=True`.
        """
        calls: list[str] = []
        monkeypatch.setattr(gdal, "OpenShared", lambda *a, **k: calls.append("shared"))
        monkeypatch.setattr(gdal, "Open", lambda *a, **k: calls.append("plain"))
        read_file(tiny_raster, read_only=True)
        assert calls == ["shared"], f"read-only should call OpenShared, got: {calls}"

    def test_update_mode_uses_unshared_handle(self, tiny_raster: str, monkeypatch: pytest.MonkeyPatch):
        """Update-mode opens go through `gdal.Open`, not the shared cache.

        Test scenario:
            GDAL returns one shared handle per path+access+thread, so two
            update-mode datasets would otherwise share a mutable handle with
            independent finalizers and corrupt each other's writes.
        """
        calls: list[str] = []
        monkeypatch.setattr(gdal, "OpenShared", lambda *a, **k: calls.append("shared"))
        monkeypatch.setattr(gdal, "Open", lambda *a, **k: calls.append("plain"))
        read_file(tiny_raster, read_only=False)
        assert calls == ["plain"], f"update mode must not use OpenShared, got: {calls}"

    def test_update_mode_returns_independent_datasets(self, tiny_raster: str):
        """Two update-mode opens yield independently usable datasets.

        Test scenario:
            Writing through one handle and dropping it must leave the second handle
            usable, which a shared mutable handle would not guarantee.
        """
        first = read_file(tiny_raster, read_only=False)
        second = read_file(tiny_raster, read_only=False)
        first.GetRasterBand(1).WriteArray(np.full((4, 4), 7, dtype="uint8"))
        first.FlushCache()
        first = None
        value = int(np.asarray(second.GetRasterBand(1).ReadAsArray()).flat[0])
        second = None
        assert value in (0, 7), f"second handle should stay readable after the first closed, got {value}"


class TestLoadZipExtractionDir:
    """Tests for the extraction directory used by `pyramids.io.sniff._load_zip`."""

    def test_default_extraction_lands_under_artifact_root(self, tmp_path: Path):
        """A zip loaded without `extract_to` extracts under the tracked root.

        Test scenario:
            The default destination must come from `base._artifacts.artifact_dir()`
            so the extraction is swept at interpreter exit instead of leaking.
        """
        from pyramids.io.sniff import load_resource

        source = tmp_path / "table.csv"
        source.write_text("a,b\n1,2\n", encoding="utf-8")
        archive = tmp_path / "one.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.write(source, arcname="table.csv")

        load_resource(archive)
        assert _artifacts._ROOT is not None, "loading a zip should create the shared artefact root"
        assert os.path.isdir(_artifacts._ROOT), f"artefact root should exist, got: {_artifacts._ROOT}"

    def test_explicit_extract_to_is_honoured(self, tmp_path: Path):
        """An explicit `extract_to` still wins over the artefact root.

        Test scenario:
            Callers that pass a destination keep full control of where members land.
        """
        from pyramids.io.sniff import load_resource

        source = tmp_path / "table.csv"
        source.write_text("a,b\n3,4\n", encoding="utf-8")
        archive = tmp_path / "two.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.write(source, arcname="table.csv")

        destination = tmp_path / "chosen"
        load_resource(archive, extract_to=destination)
        assert (destination / "table.csv").exists(), f"member should extract into {destination}"


class TestLazyIoExports:
    """Tests for the PEP 562 lazy exports on `pyramids.io`."""

    def test_import_does_not_pull_the_reader_stack(self):
        """`import pyramids.io` leaves the heavy readers unimported.

        Test scenario:
            Run in a clean subprocess (the test session already imports the readers)
            and assert none of Dataset/FeatureCollection/NetCDF modules were loaded.
        """
        import pyramids

        # Point the child at the very package this session imported, so the test
        # measures the source under test rather than whatever is pip-installed.
        src_root = str(Path(pyramids.__file__).parent.parent)
        env = {**os.environ, "PYTHONPATH": src_root}
        code = (
            "import sys; import pyramids.io; "
            "print([m for m in ('pyramids.dataset','pyramids.feature','pyramids.netcdf') if m in sys.modules])"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
        )
        assert completed.stdout.strip() == "[]", f"import pyramids.io should stay light, got: {completed.stdout!r}"

    @pytest.mark.parametrize("name", ["load_resource", "sniff_format"])
    def test_exported_names_resolve(self, name: str):
        """Both public names resolve through the lazy `__getattr__`.

        Args:
            name: Exported attribute to resolve.

        Test scenario:
            Attribute access must return the callable from `pyramids.io.sniff`.
        """
        import pyramids.io

        resolved = getattr(pyramids.io, name)
        assert callable(resolved), f"{name} should resolve to a callable, got: {type(resolved)}"

    def test_unknown_attribute_raises(self):
        """An unknown attribute still raises `AttributeError`.

        Test scenario:
            The lazy hook must not mask typos or swallow missing names.
        """
        import pyramids.io

        with pytest.raises(AttributeError, match="no attribute"):
            _ = pyramids.io.does_not_exist

    def test_dir_lists_lazy_names(self):
        """`dir(pyramids.io)` advertises the lazily-exported names.

        Test scenario:
            Tab-completion and introspection must show both names before they are
            first accessed.
        """
        import pyramids.io

        listed = dir(pyramids.io)
        assert {"load_resource", "sniff_format"} <= set(listed), f"lazy names missing from dir(): {listed}"
