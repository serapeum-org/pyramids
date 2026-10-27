import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from pyramids.base.config import Config

pytestmark = pytest.mark.core


class TestConfigEndToEnd(unittest.TestCase):
    """
    End-to-end tests for all methods in the Config class.

    These tests validate the actual functionality of the Config class without using mocks.
    The tests are executed across different operating systems and environments (e.g., Conda, non-Conda).
    """

    def setUp(self):
        """Set up a Config instance for tests."""
        self.config = Config()

    def test_set_env_conda(self):
        """End-to-end test for setting up GDAL in a Conda environment."""
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            # Conda environment
            gdal_plugins_path = Path(conda_prefix) / "Library/lib/gdalplugins"
            result = self.config.set_env_conda()
            if os.path.exists(gdal_plugins_path):
                self.assertEqual(result, gdal_plugins_path)
                self.assertEqual(
                    os.environ.get("GDAL_DRIVER_PATH"), str(gdal_plugins_path)
                )
            else:
                self.assertIsNone(result)
        else:
            # not a conda environment
            result = self.config.set_env_conda()
            self.assertIsNone(result)

    def test_dynamic_env_variables(self):
        """End-to-end test for setting dynamic environment variables for GDAL."""
        gdal_plugins_path = self.config.dynamic_env_variables()
        if gdal_plugins_path:
            self.assertTrue(gdal_plugins_path.exists())
            self.assertEqual(os.environ.get("GDAL_DRIVER_PATH"), str(gdal_plugins_path))
        else:
            self.assertIsNone(gdal_plugins_path)


class TestConfigMock(unittest.TestCase):
    def setUp(self):
        # Snapshot GDAL_DRIVER_PATH so tests in this class can mutate
        # `os.environ` freely without leaking a bogus path into later
        # tests — multiprocessing-spawn tests inherit os.environ, and a
        # stale `GDAL_DRIVER_PATH=/fake/...` breaks every downstream
        # subprocess NetCDF read with "gdal_HDF5.so not available".
        self._saved_gdal_driver_path = os.environ.get("GDAL_DRIVER_PATH")
        self.config = Config()

    def tearDown(self):
        if self._saved_gdal_driver_path is None:
            os.environ.pop("GDAL_DRIVER_PATH", None)
        else:
            os.environ["GDAL_DRIVER_PATH"] = self._saved_gdal_driver_path

    @patch("os.environ", new_callable=dict)
    @patch("pyramids.base.config.Config.set_env_conda")
    @patch("pyramids.base.config.Config.dynamic_env_variables")
    @patch("osgeo.gdal.AllRegister")
    def test_initialize_gdal(self, mock_register, mock_dynamic, mock_conda, mock_env):
        mock_dynamic.return_value = Path("/usr/lib/gdalplugins")
        mock_conda.return_value = Path("/usr/lib/gdalplugins")
        self.config.initialize_gdal()

        mock_register.assert_called_once()

    @patch("os.getenv", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    def test_set_env_conda_no_conda(self, mock_exists, mock_getenv):
        result = self.config.set_env_conda()
        self.assertIsNone(result)
        mock_getenv.assert_called()

    @patch("os.getenv", return_value="/fake/conda/prefix")
    @patch("pathlib.Path.exists", return_value=True)
    def test_set_env_conda_success(self, mock_exists, mock_getenv):
        # Env leakage is guarded by `setUp`/`tearDown` on the class.
        result = self.config.set_env_conda()
        self.assertEqual(result, Path("/fake/conda/prefix/Library/lib/gdalplugins"))
        path = Path(os.environ["GDAL_DRIVER_PATH"])
        self.assertEqual(path, Path("/fake/conda/prefix/Library/lib/gdalplugins"))

    @patch("os.getenv", return_value="fake/conda/prefix")
    @patch("pathlib.Path.exists", return_value=False)
    def test_set_env_conda_plugins_path_not_exist(self, mock_exists, mock_getenv):
        result = self.config.set_env_conda()
        self.assertIsNone(result)
        mock_exists.assert_called()

    @patch("site.getsitepackages", return_value=["C:/Python/site-packages"])
    @patch("pathlib.Path.exists", return_value=True)
    def test_dynamic_env_variables_windows(self, mock_exists, mock_site):
        with (
            patch("pyramids.base.config.Config.set_env_conda", return_value=None),
            patch("sys.platform", new="win32"),
        ):
            result = self.config.dynamic_env_variables()
        self.assertEqual(
            result, Path("C:/Python/site-packages/Library/lib/gdalplugins")
        )
        path = Path(os.environ["GDAL_DRIVER_PATH"])
        self.assertEqual(path, Path("C:/Python/site-packages/Library/lib/gdalplugins"))

    @patch("pathlib.Path.exists", side_effect=[True, False])
    def test_dynamic_env_variables_linux(self, mock_exists):
        with (
            patch("pyramids.base.config.Config.set_env_conda", return_value=None),
            patch("sys.platform", new="linux"),
        ):
            result = self.config.dynamic_env_variables()
        self.assertEqual(result, Path("/usr/local/lib/gdalplugins"))
        path = Path(os.environ["GDAL_DRIVER_PATH"])
        self.assertEqual(path, Path("/usr/local/lib/gdalplugins"))


class TestExportGdalLibraryEnv:
    """Direct tests for the shared GDAL env exporter both bootstrap branches call (#1325).

    Each test replaces ``os.environ`` with an isolated dict (auto-restored by monkeypatch)
    and shapes a temp directory like a conda-forge ``Library/`` tree, so the exporter is
    exercised directly instead of only through the conda or win32 branch.
    """

    @staticmethod
    def _make_library(root: Path, *, plugins: bool = True) -> Path:
        """Build a conda-style ``Library/`` tree under root; return the ``Library`` dir.

        Args:
            root: Directory to create the tree in.
            plugins: Whether to create the ``lib/gdalplugins`` directory.

        Returns:
            Path: The created ``Library`` directory.
        """
        lib = root / "Library"
        if plugins:
            (lib / "lib" / "gdalplugins").mkdir(parents=True)
        (lib / "bin").mkdir(parents=True)
        (lib / "share" / "gdal").mkdir(parents=True)
        (lib / "share" / "proj").mkdir(parents=True)
        return lib

    def test_exports_all_four_from_a_library_tree(self, tmp_path, monkeypatch):
        """A complete tree sets GDAL_DRIVER_PATH, prepends bin to PATH, sets GDAL_DATA/PROJ_LIB.

        Test scenario:
            A ``Library`` tree with all four sub-paths present is exported; the plugins dir
            is returned and every env var points at the matching sub-path.
        """
        cfg = Config()
        lib = self._make_library(tmp_path)
        monkeypatch.setattr(os, "environ", {"PATH": "/pre-existing"})
        result = cfg._export_gdal_library_env(lib)
        assert result == lib / "lib" / "gdalplugins", (
            f"unexpected plugins dir: {result}"
        )
        assert os.environ["GDAL_DRIVER_PATH"] == str(lib / "lib" / "gdalplugins")
        assert os.environ["GDAL_DATA"] == str(lib / "share" / "gdal")
        assert os.environ["PROJ_LIB"] == str(lib / "share" / "proj")
        assert os.environ["PATH"].split(os.pathsep)[0] == str(lib / "bin"), (
            f"bin not prepended to PATH: {os.environ['PATH']!r}"
        )

    def test_missing_plugins_returns_none_but_still_exports_present_dirs(
        self, tmp_path, monkeypatch
    ):
        """No gdalplugins dir returns None, yet existing bin/share dirs still export.

        Test scenario:
            The unified helper attempts bin/GDAL_DATA/PROJ_LIB regardless of the plugins
            dir (matching the conda branch), so an absent gdalplugins yields None but the
            present share dirs are still set.
        """
        cfg = Config()
        lib = self._make_library(tmp_path, plugins=False)
        monkeypatch.setattr(os, "environ", {"PATH": ""})
        result = cfg._export_gdal_library_env(lib)
        assert result is None, f"expected None when plugins absent, got {result}"
        assert "GDAL_DRIVER_PATH" not in os.environ, (
            "driver path set without a plugins dir"
        )
        assert os.environ["GDAL_DATA"] == str(lib / "share" / "gdal")

    def test_absent_library_is_a_no_op(self, tmp_path, monkeypatch):
        """A non-existent Library root sets nothing and returns None.

        Test scenario:
            None of the four sub-paths exist, so PATH is untouched and no GDAL env var
            is written.
        """
        cfg = Config()
        monkeypatch.setattr(os, "environ", {"PATH": "/x"})
        result = cfg._export_gdal_library_env(tmp_path / "Library")
        assert result is None, f"expected None, got {result}"
        assert "GDAL_DRIVER_PATH" not in os.environ, (
            "driver path set for an absent tree"
        )
        assert os.environ["PATH"] == "/x", "PATH mutated when no bin dir exists"

    def test_bin_is_not_prepended_twice(self, tmp_path, monkeypatch):
        """The PATH-prepend dedup guard keeps bin on PATH exactly once.

        Test scenario:
            With bin already on PATH, the exporter must not add a second copy.
        """
        cfg = Config()
        lib = self._make_library(tmp_path)
        bin_str = str(lib / "bin")
        monkeypatch.setattr(os, "environ", {"PATH": bin_str})
        cfg._export_gdal_library_env(lib)
        assert os.environ["PATH"].split(os.pathsep).count(bin_str) == 1, (
            f"bin duplicated on PATH: {os.environ['PATH']!r}"
        )


class TestProbeWindowsPlugins:
    """Direct tests for the Windows site-packages probe (#1325)."""

    def test_last_matching_library_wins_and_empty_roots_are_skipped(
        self, tmp_path, monkeypatch
    ):
        """Probing several site-packages picks the last with a Library tree; empty ones skip.

        Test scenario:
            Two site-packages roots are probed: the first has no ``Library`` (the exporter
            returns None, so the per-root check is falsy and the loop continues), the second
            carries a full ``Library`` tree. The probe returns the second root's gdalplugins
            dir, exercising both arms of the per-root check and the last-match-wins loop.
        """
        empty_root = tmp_path / "a"
        empty_root.mkdir()
        full_root = tmp_path / "b"
        plugins = full_root / "Library" / "lib" / "gdalplugins"
        plugins.mkdir(parents=True)
        (full_root / "Library" / "bin").mkdir(parents=True)
        cfg = Config()
        monkeypatch.setattr(os, "environ", {"PATH": ""})
        monkeypatch.setattr(
            "site.getsitepackages", lambda: [str(empty_root), str(full_root)]
        )
        result = cfg._probe_windows_plugins()
        assert result == plugins, f"expected the last root's plugins dir, got {result}"

    def test_no_library_anywhere_returns_none(self, tmp_path, monkeypatch):
        """When no probed root has a Library tree, the probe returns None.

        Test scenario:
            Every site-packages root is empty, so the exporter returns None for each and the
            probe reports no plugins dir found.
        """
        root = tmp_path / "empty"
        root.mkdir()
        cfg = Config()
        monkeypatch.setattr(os, "environ", {"PATH": ""})
        monkeypatch.setattr("site.getsitepackages", lambda: [str(root)])
        result = cfg._probe_windows_plugins()
        assert result is None, f"expected None when no Library is found, got {result}"
