"""Unit tests for the pyramids.stac.download wrappers (PC-3, STAC-15).

stac-asset ships via the optional [stac] extra (heavy async deps), so most of
these tests do not require it installed: the missing-dependency guard is
exercised by mocking the import helper, and the wiring is exercised with an
injected fake stac_asset module. Nothing here touches the network.

A fake `Config` that accepts anything cannot notice an upstream field rename,
so `TestAgainstTheRealConfig` drives the **installed** `stac_asset` instead and
is skipped when it is absent. The import below is guarded rather than inline:
the repo forbids imports inside functions, and a bare module-level import would
stop the whole file collecting without the extra.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
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

try:  # pragma: no cover - exercised by whether the [stac] extra is installed
    import stac_asset
    import stac_asset.blocking
except ImportError:  # pragma: no cover - same
    stac_asset = None

pytestmark = pytest.mark.core

requires_stac_asset = pytest.mark.skipif(
    stac_asset is None,
    reason="stac-asset (pyramids-gis[stac], PyPI only) is not installed",
)

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

    @pytest.mark.parametrize("count", [0, -1])
    def test_non_positive_max_concurrent_raises(self, fake_stac_asset, tmp_path, count):
        """A cap below 1 is rejected instead of stalling the download.

        Test scenario:
            `0` is falsy but not `None`, so it clears the `is None` guard and
            reaches stac-asset's semaphore as a budget of zero permits — a
            download that never starts and never errors.
        """
        with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
            download_item("ITEM", tmp_path, max_concurrent=count)
        assert "kwargs" not in fake_stac_asset, (
            f"the downloader should not have run: {fake_stac_asset}"
        )

    @pytest.mark.parametrize(
        "func, name", DOWNLOADERS, ids=[name for _, name in DOWNLOADERS]
    )
    def test_every_wrapper_validates_max_concurrent(
        self, fake_stac_asset, tmp_path, func, name
    ):
        """All three wrappers share the guard, since they share the body.

        Test scenario:
            The validation belongs to `_download`, so no entry point can be
            left behind.
        """
        with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
            func("TARGET", tmp_path, max_concurrent=0)

    def test_max_concurrent_of_one_is_accepted(self, fake_stac_asset, tmp_path):
        """A serial download is a legitimate request, not a rejected one.

        Test scenario:
            1 is the lowest workable budget and must reach the downloader.
        """
        download_item("ITEM", tmp_path, max_concurrent=1)
        assert fake_stac_asset["kwargs"] == {"max_concurrent_downloads": 1}, (
            f"a cap of 1 should be forwarded, got {fake_stac_asset['kwargs']}"
        )

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


@requires_stac_asset
class TestAgainstTheRealConfig:
    """The forwarded names are fields of the installed `stac_asset.Config`.

    Test scenario:
        Every other test here drives a `FakeConfig` whose `__init__` takes
        `**kwargs`, so it accepts any spelling pyramids invents and any
        spelling stac-asset later abandons. Renaming a real field (say
        `alternate_assets` to `alternates`) would therefore leave the suite
        green while breaking every actual download. These assert against the
        real dataclass and the real downloader signature, so the rename shows
        up here instead of in production.
    """

    _ALL_OPTIONS = {
        "include": ["B04"],
        "exclude": ["thumbnail"],
        "alternate_assets": ["s3"],
        "s3_requester_pays": True,
        "file_name_strategy": "key",
        "error_strategy": "keep",
        "fail_fast": True,
        "warn": False,
    }

    def test_every_forwarded_name_is_a_declared_config_field(self):
        """No kwarg pyramids builds is unknown to the real Config."""
        declared = {field.name for field in dataclasses.fields(stac_asset.Config)}
        forwarded = set(dl_mod._config_kwargs(stac_asset, dict(self._ALL_OPTIONS)))
        unknown = sorted(forwarded - declared)
        assert unknown == [], (
            f"stac_asset.Config declares no such field(s): {unknown}; "
            f"it has {sorted(declared)}"
        )

    def test_the_real_config_is_built_from_every_option(self, monkeypatch, tmp_path):
        """A full call constructs a genuine Config carrying every value.

        Test scenario:
            Only the blocking downloader is replaced, so the Config itself is
            the real dataclass — a renamed or removed field raises TypeError
            here rather than being silently accepted.
        """
        captured: dict = {}

        def fake_download(target, directory, config=None, **kwargs):
            """Record the real Config and the downloader kwargs."""
            captured["config"] = config
            captured["kwargs"] = kwargs
            return "LOCAL_ITEM"

        monkeypatch.setattr(stac_asset.blocking, "download_item", fake_download)
        out = download_item("ITEM", tmp_path, max_concurrent=3, **self._ALL_OPTIONS)
        config = captured["config"]
        assert out == "LOCAL_ITEM", f"should return the downloader result, got {out}"
        assert isinstance(config, stac_asset.Config), (
            f"not a real stac_asset.Config: {type(config)}"
        )
        assert config.include == ["B04"], f"include not set: {config.include}"
        assert config.exclude == ["thumbnail"], f"exclude not set: {config.exclude}"
        assert config.alternate_assets == ["s3"], (
            f"alternate_assets not set: {config.alternate_assets}"
        )
        assert config.s3_requester_pays is True, "s3_requester_pays not set"
        assert config.fail_fast is True and config.warn is False, (
            f"the explicit flags were lost: fail_fast={config.fail_fast}, "
            f"warn={config.warn}"
        )
        assert captured["kwargs"] == {"max_concurrent_downloads": 3}, (
            f"the concurrency cap is a downloader kwarg: {captured['kwargs']}"
        )

    def test_the_strategies_resolve_to_real_enum_members(self):
        """The strategy names pyramids documents exist in the real enums."""
        file_name = dl_mod._coerce_strategy(stac_asset, "key", "FileNameStrategy")
        error = dl_mod._coerce_strategy(stac_asset, "keep", "ErrorStrategy")
        assert file_name is stac_asset.FileNameStrategy.KEY, (
            f"'key' should resolve to the real FileNameStrategy member, got {file_name}"
        )
        assert error is stac_asset.ErrorStrategy.KEEP, (
            f"'keep' should resolve to the real ErrorStrategy member, got {error}"
        )
        assert (
            dl_mod._coerce_strategy(stac_asset, "file_name", "FileNameStrategy")
            is stac_asset.FileNameStrategy.FILE_NAME
        ), "'file_name' no longer resolves"
        assert (
            dl_mod._coerce_strategy(stac_asset, "delete", "ErrorStrategy")
            is stac_asset.ErrorStrategy.DELETE
        ), "'delete' no longer resolves"

    @pytest.mark.parametrize(
        "name", [name for _, name in DOWNLOADERS], ids=[name for _, name in DOWNLOADERS]
    )
    def test_the_real_downloaders_accept_the_concurrency_kwarg(self, name):
        """`max_concurrent_downloads` is a parameter of each real downloader."""
        signature = inspect.signature(getattr(stac_asset.blocking, name))
        assert "max_concurrent_downloads" in signature.parameters, (
            f"stac_asset.blocking.{name} no longer takes max_concurrent_downloads: "
            f"{list(signature.parameters)}"
        )
