"""Unit tests for the pyramids.stac.download wrappers (PC-3, STAC-15).

stac-asset ships via the optional [stac] extra (heavy async deps), so these tests
do not require it installed: the missing-dependency guard is exercised by
mocking the import helper, and the wiring is exercised with an injected fake
stac_asset module. Nothing here touches the network.
"""

from __future__ import annotations

import enum
import sys
import types

import pytest

import pyramids.stac.download as dl_mod
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.stac.download import (
    download_collection,
    download_item,
    download_item_collection,
)

pytestmark = pytest.mark.core

DOWNLOADERS = [
    (download_item, "download_item"),
    (download_item_collection, "download_item_collection"),
    (download_collection, "download_collection"),
]


class FakeFileNameStrategy(enum.Enum):
    """Stand-in for `stac_asset.FileNameStrategy` (same member names)."""

    FILE_NAME = 1
    KEY = 2


class FakeErrorStrategy(enum.Enum):
    """Stand-in for `stac_asset.ErrorStrategy` (same member names)."""

    KEEP = 1
    DELETE = 2


class TestDownloadGuards:
    """The missing-stac-asset guard fires before any download."""

    @pytest.mark.parametrize(
        "func, name", DOWNLOADERS, ids=[name for _, name in DOWNLOADERS]
    )
    def test_missing_dependency_raises(self, monkeypatch, func, name):
        """Every wrapper raises OptionalPackageDoesNotExist when absent.

        Test scenario:
            import_stac_asset raises -> the error points at the [stac] extra.
        """

        def _raise(*_a, **_k):
            raise OptionalPackageDoesNotExist("download_item requires 'stac-asset'")

        monkeypatch.setattr(dl_mod, "import_stac_asset", _raise)
        with pytest.raises(OptionalPackageDoesNotExist, match="stac-asset"):
            func("TARGET", "out/")

    def test_install_hint_names_conda_forge_caveat(self):
        """The composed hint still carries the stac-asset conda-forge caveat.

        Test scenario:
            The three wrappers share one hint; it must keep naming the extra.
        """
        assert "pip install stac-asset" in dl_mod._STAC_ASSET_INSTALL_HINT, (
            "the hint should keep the conda-forge caveat line"
        )


class TestDownloadItemWiring:
    """download_item builds a Config from kwargs and delegates to stac_asset."""

    @pytest.fixture
    def fake_stac_asset(self, monkeypatch):
        """Inject a fake stac_asset whose Config accepts only the legacy fields.

        The strict signature is the point: it fails loudly if a default call
        ever starts passing widened options through to `stac_asset.Config`.

        Returns:
            dict: ``captured`` recording the Config args and the download call.
        """
        captured: dict = {}

        fake = types.ModuleType("stac_asset")
        fake_blocking = types.ModuleType("stac_asset.blocking")

        class FakeConfig:
            def __init__(self, *, include, exclude, s3_requester_pays):
                captured["config"] = (include, exclude, s3_requester_pays)

        def make_download(name):
            def fake_download(target, directory, config=None, **kwargs):
                captured["function"] = name
                captured["item"] = target
                captured["directory"] = directory
                captured["config_obj"] = config
                captured["kwargs"] = kwargs
                return "LOCAL_ITEM"

            return fake_download

        fake.Config = FakeConfig
        fake.blocking = fake_blocking
        for _, name in DOWNLOADERS:
            setattr(fake_blocking, name, make_download(name))
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
        assert fake_stac_asset["directory"] == str(tmp_path), (
            "directory should be stringified"
        )

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

    def test_defaults_pass_no_downloader_kwargs(self, fake_stac_asset, tmp_path):
        """A default call adds no extra downloader arguments.

        Test scenario:
            max_concurrent omitted -> no max_concurrent_downloads kwarg.
        """
        download_item("ITEM", tmp_path)
        assert fake_stac_asset["kwargs"] == {}, (
            f"no downloader kwargs expected, got {fake_stac_asset['kwargs']}"
        )

    @pytest.mark.parametrize(
        "func, name", DOWNLOADERS, ids=[name for _, name in DOWNLOADERS]
    )
    def test_each_wrapper_calls_its_own_downloader(
        self, fake_stac_asset, tmp_path, func, name
    ):
        """Each wrapper targets the matching stac_asset.blocking function.

        Test scenario:
            The item-collection wrapper must call download_item_collection, not
            the single-item one.
        """
        out = func("TARGET", tmp_path)
        assert fake_stac_asset["function"] == name, (
            f"expected stac_asset.blocking.{name}, got {fake_stac_asset['function']}"
        )
        assert fake_stac_asset["item"] == "TARGET", "the target should be forwarded"
        assert out == "LOCAL_ITEM", f"should return the downloader result, got {out}"


class TestWidenedOptions:
    """The widened Config / downloader options reach stac_asset correctly."""

    @pytest.fixture
    def fake_stac_asset(self, monkeypatch):
        """Inject a fake stac_asset recording every Config and downloader kwarg.

        Returns:
            dict: ``captured`` with the Config kwargs and the download call.
        """
        captured: dict = {}

        fake = types.ModuleType("stac_asset")
        fake_blocking = types.ModuleType("stac_asset.blocking")

        class FakeConfig:
            def __init__(self, **kwargs):
                captured["config"] = kwargs

        def make_download(name):
            def fake_download(target, directory, config=None, **kwargs):
                captured["function"] = name
                captured["target"] = target
                captured["directory"] = directory
                captured["kwargs"] = kwargs
                return "LOCAL"

            return fake_download

        fake.Config = FakeConfig
        fake.FileNameStrategy = FakeFileNameStrategy
        fake.ErrorStrategy = FakeErrorStrategy
        fake.blocking = fake_blocking
        for _, name in DOWNLOADERS:
            setattr(fake_blocking, name, make_download(name))
        monkeypatch.setitem(sys.modules, "stac_asset", fake)
        monkeypatch.setitem(sys.modules, "stac_asset.blocking", fake_blocking)
        monkeypatch.setattr(dl_mod, "import_stac_asset", lambda *a, **k: None)
        return captured

    @pytest.mark.parametrize(
        "func, name", DOWNLOADERS, ids=[name for _, name in DOWNLOADERS]
    )
    def test_every_option_forwarded(self, fake_stac_asset, tmp_path, func, name):
        """All widened options land in Config, except the concurrency cap.

        Test scenario:
            alternate_assets/file_name_strategy/error_strategy/fail_fast/warn
            go to Config; max_concurrent goes to the downloader itself.
        """
        func(
            "TARGET",
            tmp_path,
            include=["B04"],
            exclude=["thumbnail"],
            alternate_assets=["s3"],
            s3_requester_pays=True,
            file_name_strategy="key",
            error_strategy="keep",
            fail_fast=True,
            warn=True,
            max_concurrent=4,
        )
        assert fake_stac_asset["config"] == {
            "include": ["B04"],
            "exclude": ["thumbnail"],
            "s3_requester_pays": True,
            "alternate_assets": ["s3"],
            "file_name_strategy": FakeFileNameStrategy.KEY,
            "error_strategy": FakeErrorStrategy.KEEP,
            "fail_fast": True,
            "warn": True,
        }, f"Config kwargs mismatch: {fake_stac_asset['config']}"
        assert fake_stac_asset["kwargs"] == {"max_concurrent_downloads": 4}, (
            f"max_concurrent should be a downloader kwarg, got {fake_stac_asset['kwargs']}"
        )

    def test_omitted_options_stay_out_of_config(self, fake_stac_asset, tmp_path):
        """Unset widened options are not passed, so stac_asset's defaults win.

        Test scenario:
            A bare call -> exactly the three legacy Config fields.
        """
        download_item_collection("ITEMS", tmp_path)
        assert set(fake_stac_asset["config"]) == {
            "include",
            "exclude",
            "s3_requester_pays",
        }, f"only the legacy fields expected, got {sorted(fake_stac_asset['config'])}"

    def test_empty_alternate_assets_is_not_passed(self, fake_stac_asset, tmp_path):
        """An empty alternate_assets list leaves the Config field untouched.

        Test scenario:
            alternate_assets=[] equals the stac_asset default -> omitted.
        """
        download_item("ITEM", tmp_path, alternate_assets=[])
        assert "alternate_assets" not in fake_stac_asset["config"], (
            "an empty alternate_assets should not be forwarded"
        )

    def test_iterables_are_copied_to_lists(self, fake_stac_asset, tmp_path):
        """Tuples given for the key filters reach Config as lists.

        Test scenario:
            stac_asset.Config declares list[str] fields.
        """
        download_item(
            "ITEM",
            tmp_path,
            include=("B04", "B03"),
            exclude=("thumbnail",),
            alternate_assets=("s3",),
        )
        config = fake_stac_asset["config"]
        assert config["include"] == ["B04", "B03"], (
            f"include should be a list, got {config['include']}"
        )
        assert config["exclude"] == ["thumbnail"], (
            f"exclude should be a list, got {config['exclude']}"
        )
        assert config["alternate_assets"] == ["s3"], (
            f"alternate_assets should be a list, got {config['alternate_assets']}"
        )

    def test_enum_members_pass_through(self, fake_stac_asset, tmp_path):
        """Strategies given as enum members are forwarded untouched.

        Test scenario:
            Callers with the extra installed may pass the real enum members.
        """
        download_item(
            "ITEM",
            tmp_path,
            file_name_strategy=FakeFileNameStrategy.FILE_NAME,
            error_strategy=FakeErrorStrategy.DELETE,
        )
        config = fake_stac_asset["config"]
        assert config["file_name_strategy"] is FakeFileNameStrategy.FILE_NAME, (
            f"enum member should pass through, got {config['file_name_strategy']}"
        )
        assert config["error_strategy"] is FakeErrorStrategy.DELETE, (
            f"enum member should pass through, got {config['error_strategy']}"
        )

    def test_strategy_strings_are_case_insensitive(self, fake_stac_asset, tmp_path):
        """Strategy names are matched regardless of case or padding.

        Test scenario:
            " Key " resolves to FileNameStrategy.KEY.
        """
        download_item("ITEM", tmp_path, file_name_strategy=" Key ")
        assert (
            fake_stac_asset["config"]["file_name_strategy"] is FakeFileNameStrategy.KEY
        ), "a padded, mixed-case name should resolve"

    @pytest.mark.parametrize("option", ["file_name_strategy", "error_strategy"])
    def test_unknown_strategy_name_raises(self, fake_stac_asset, tmp_path, option):
        """An unknown strategy name fails with the allowed names listed.

        Test scenario:
            file_name_strategy="nope" -> ValueError naming the valid members.
        """
        with pytest.raises(ValueError, match="nope"):
            download_item("ITEM", tmp_path, **{option: "nope"})

    def test_false_flags_are_forwarded(self, fake_stac_asset, tmp_path):
        """fail_fast=False / warn=False are explicit, not treated as unset.

        Test scenario:
            Passing False must still override a stac_asset default of True.
        """
        download_item("ITEM", tmp_path, fail_fast=False, warn=False)
        config = fake_stac_asset["config"]
        assert config["fail_fast"] is False, (
            f"fail_fast should be forwarded, got {config.get('fail_fast')}"
        )
        assert config["warn"] is False, (
            f"warn should be forwarded, got {config.get('warn')}"
        )
