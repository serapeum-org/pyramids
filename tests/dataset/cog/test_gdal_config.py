"""Tests for GDAL config-option support on COG read/write (PC-1).

Covers COG_READ_DEFAULTS, the remote-detection / config-resolution helpers, and
applying an explicit `config` dict to to_cog / validate / cog_info.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset
from pyramids.dataset.cog import COG_READ_DEFAULTS, cog_info, validate
from pyramids.dataset.cog.validate import (
    _is_remote,
    _resolve_read_config,
    config_context,
)
from tests.dataset.cog.conftest import COG_GEOTRANSFORM

pytestmark = pytest.mark.core


@pytest.fixture
def float_cog(tmp_path) -> str:
    """A small valid COG on disk.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the COG.
    """
    arr = (np.random.default_rng(seed=8).random((64, 64)) * 100).astype("float32")
    ds = Dataset.create_from_array(arr, geo=COG_GEOTRANSFORM, epsg=4326)
    return str(ds.to_cog(tmp_path / "c.tif"))


class TestRemoteDetection:
    """Tests for _is_remote and _resolve_read_config."""

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("/vsicurl/https://x/y.tif", True),
            ("/vsis3/bucket/key.tif", True),
            ("https://example.com/a.tif", True),
            ("/local/path.tif", False),
            ("C:/data/a.tif", False),
        ],
    )
    def test_is_remote(self, path, expected):
        """_is_remote flags network-backed paths.

        Args:
            path: The path to classify.
            expected: Expected remote flag.

        Test scenario:
            /vsicurl, cloud-VSI and HTTP(S) are remote; local paths are not.
        """
        assert _is_remote(path) is expected, f"{path}: expected {expected}"

    def test_resolve_explicit_config_wins(self):
        """An explicit config is returned unchanged for any path.

        Test scenario:
            A caller-supplied config overrides the remote defaults.
        """
        cfg = {"GDAL_NUM_THREADS": "2"}
        assert _resolve_read_config("/vsicurl/x.tif", cfg) == cfg, "explicit wins"

    def test_resolve_remote_default(self):
        """A remote path with no config gets COG_READ_DEFAULTS.

        Test scenario:
            The returned dict equals COG_READ_DEFAULTS.
        """
        out = _resolve_read_config("/vsicurl/x.tif", None)
        assert out == COG_READ_DEFAULTS, f"expected read defaults, got {out}"

    def test_resolve_local_none(self):
        """A local path with no config resolves to None (no extra config).

        Test scenario:
            Local reads apply no implicit config.
        """
        assert _resolve_read_config("/local/x.tif", None) is None, "local => None"


class TestConfigApplication:
    """Tests that config is applied to write/validate/inspect without error."""

    def test_to_cog_with_config(self, tmp_path):
        """to_cog accepts and applies a config dict.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Passing config={"GDAL_NUM_THREADS": "1"} still writes a valid COG.
        """
        arr = np.ones((64, 64), dtype="float32")
        ds = Dataset.create_from_array(arr, geo=COG_GEOTRANSFORM, epsg=4326)
        out = ds.to_cog(tmp_path / "cfg.tif", config={"GDAL_NUM_THREADS": "1"})
        assert Dataset.read_file(str(out)).validate_cog().is_valid, (
            "config write invalid"
        )

    def test_validate_with_config(self, float_cog):
        """validate accepts a config dict and still validates.

        Args:
            float_cog: Fixture path to a valid COG.

        Test scenario:
            An explicit config does not change a local validation result.
        """
        report = validate(float_cog, config={"GDAL_NUM_THREADS": "1"})
        assert report.is_valid, f"expected valid, errors {report.errors}"

    def test_cog_info_with_config(self, float_cog):
        """cog_info accepts a config dict and returns metadata.

        Args:
            float_cog: Fixture path to a valid COG.

        Test scenario:
            An explicit config does not change the reported compression.
        """
        info = cog_info(float_cog, config={"GDAL_NUM_THREADS": "1"})
        assert info.compression == "DEFLATE", (
            f"unexpected compression {info.compression}"
        )

    def test_config_context_applies_and_restores(self):
        """config_context applies options inside and restores them after (L3).

        Test scenario:
            A sentinel option is unset before, set inside the block, and unset
            again afterwards.
        """
        key = "PYRAMIDS_L3_SENTINEL"
        assert gdal.GetConfigOption(key, "unset") == "unset"
        with config_context({key: "on"}):
            assert gdal.GetConfigOption(key, "unset") == "on", "should apply inside"
        assert gdal.GetConfigOption(key, "unset") == "unset", "should restore after"

    def test_config_context_none_is_noop(self):
        """config_context(None) is a no-op context (L3).

        Test scenario:
            Passing None/empty yields without touching GDAL config.
        """
        with config_context(None):
            # None yields a no-op context; nothing to do inside
            pass
        with config_context({}):
            # empty mapping yields a no-op context; nothing to do inside
            pass

    def test_config_context_fallback_without_config_options(self, monkeypatch):
        """config_context falls back to set/restore if gdal.config_options absent (L3).

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            Simulate an old GDAL by removing gdal.config_options; the helper must
            still apply the option inside the block and restore it afterwards via
            SetConfigOption.
        """
        key = "PYRAMIDS_L3_FALLBACK"
        monkeypatch.delattr(gdal, "config_options", raising=False)
        assert gdal.GetConfigOption(key, "unset") == "unset"
        with config_context({key: "yes"}):
            assert gdal.GetConfigOption(key, "unset") == "yes", "fallback should apply"
        assert gdal.GetConfigOption(key, "unset") == "unset", "fallback should restore"

    def test_config_is_restored_after_call(self, float_cog):
        """The config context is scoped — options are restored afterwards.

        Args:
            float_cog: Fixture path to a valid COG.

        Test scenario:
            A sentinel option set only inside the call is not leaked globally.
        """
        before = gdal.GetConfigOption("PYRAMIDS_PC1_SENTINEL", "unset")
        validate(float_cog, config={"PYRAMIDS_PC1_SENTINEL": "on"})
        after = gdal.GetConfigOption("PYRAMIDS_PC1_SENTINEL", "unset")
        assert before == "unset" and after == "unset", "config leaked outside context"
