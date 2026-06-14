"""Unit tests for pyramids.stac.download.download_item (PC-3).

stac-asset ships via the optional [stac] extra (heavy async deps), so these tests
do not require it installed: the missing-dependency guard is exercised by
mocking the import helper, and the wiring is exercised with an injected fake
stac_asset module.
"""

from __future__ import annotations

import sys
import types

import pytest

import pyramids.stac.download as dl_mod
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.stac.download import download_item

pytestmark = pytest.mark.core


class TestDownloadItemGuard:
    """The missing-stac-asset guard fires before any download."""

    def test_missing_dependency_raises(self, monkeypatch):
        """download_item raises OptionalPackageDoesNotExist when absent.

        Test scenario:
            import_stac_asset raises -> the error points at the [stac] extra.
        """

        def _raise(*_a, **_k):
            raise OptionalPackageDoesNotExist("download_item requires 'stac-asset'")

        monkeypatch.setattr(dl_mod, "import_stac_asset", _raise)
        with pytest.raises(OptionalPackageDoesNotExist, match="stac-asset"):
            download_item("ITEM", "out/")


class TestDownloadItemWiring:
    """download_item builds a Config from kwargs and delegates to stac_asset."""

    @pytest.fixture
    def fake_stac_asset(self, monkeypatch):
        """Inject a fake stac_asset module + blocking + Config (no real dep).

        Returns:
            dict: ``captured`` recording the Config args and the download call.
        """
        captured: dict = {}

        fake = types.ModuleType("stac_asset")
        fake_blocking = types.ModuleType("stac_asset.blocking")

        class FakeConfig:
            def __init__(self, *, include, exclude, s3_requester_pays):
                captured["config"] = (include, exclude, s3_requester_pays)

        def fake_download(item, directory, config=None):
            captured["item"] = item
            captured["directory"] = directory
            captured["config_obj"] = config
            return "LOCAL_ITEM"

        fake.Config = FakeConfig
        fake.blocking = fake_blocking
        fake_blocking.download_item = fake_download
        monkeypatch.setitem(sys.modules, "stac_asset", fake)
        monkeypatch.setitem(sys.modules, "stac_asset.blocking", fake_blocking)
        monkeypatch.setattr(dl_mod, "import_stac_asset", lambda *a, **k: None)
        return captured

    def test_config_built_and_delegated(self, fake_stac_asset, tmp_path):
        """Config is built from kwargs and the blocking downloader is called.

        Test scenario:
            include/exclude/s3_requester_pays flow into Config; the item and
            directory reach stac_asset.blocking.download_item; its result is
            returned.
        """
        out = download_item(
            "ITEM",
            tmp_path,
            include=["B04"],
            exclude=["thumbnail"],
            s3_requester_pays=True,
        )
        assert out == "LOCAL_ITEM", f"should return the downloader result, got {out}"
        assert fake_stac_asset["config"] == (
            ["B04"],
            ["thumbnail"],
            True,
        ), f"Config args mismatch: {fake_stac_asset['config']}"
        assert fake_stac_asset["item"] == "ITEM", "item should be forwarded"
        assert fake_stac_asset["directory"] == str(
            tmp_path
        ), "directory should be stringified"

    def test_defaults_empty_filters(self, fake_stac_asset, tmp_path):
        """Omitted include/exclude become empty lists in the Config.

        Test scenario:
            No include/exclude -> ([], [], False).
        """
        download_item("ITEM", tmp_path)
        assert fake_stac_asset["config"] == (
            [],
            [],
            False,
        ), f"default Config mismatch: {fake_stac_asset['config']}"
