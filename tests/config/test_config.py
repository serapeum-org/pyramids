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

        # self.assertIn("GDAL_DRIVER_PATH", os.environ)
        # self.assertEqual(os.environ["GDAL_DRIVER_PATH"], "/usr/lib/gdalplugins")
        mock_register.assert_called_once()

    @patch("os.getenv", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    def test_set_env_conda_no_conda(self, mock_exists, mock_getenv):
        result = self.config.set_env_conda()
        self.assertIsNone(result)

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

    @patch("site.getsitepackages", return_value=["C:/Python/site-packages"])
    @patch("pathlib.Path.exists", return_value=True)
    def test_dynamic_env_variables_windows(self, mock_exists, mock_site):
        with (
            patch("pyramids.base.config.Config.set_env_conda", return_value=None),
            patch("sys.platform", new="win32"),
        ):
            result = self.config.dynamic_env_variables()
        self.assertEqual(
            result, Path("C:/Python/site-packages/Library/Lib/gdalplugins")
        )
        path = Path(os.environ["GDAL_DRIVER_PATH"])
        self.assertEqual(path, Path("C:/Python/site-packages/Library/Lib/gdalplugins"))

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
