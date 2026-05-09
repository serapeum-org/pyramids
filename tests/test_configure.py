"""Tests for :func:`pyramids.configure`.

DASK-5 entry point. Sets GDAL config options in the current process
and, when a dask client is given, broadcasts to every worker.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from osgeo import gdal

from pyramids import configure
from pyramids._configure import GDAL_CLOUD_DEFAULTS, _expand_credentials

pytestmark = pytest.mark.core


class TestConfigureReturnValue:
    """`configure` returns the effective env dict that was applied."""

    def test_cloud_defaults_included(self):
        applied = configure(cloud_defaults=True)
        for key, value in GDAL_CLOUD_DEFAULTS.items():
            assert applied[key] == value

    def test_override_applied_after_cloud_defaults(self):
        applied = configure(cloud_defaults=True, GDAL_HTTP_MAX_RETRY="3")
        assert applied["GDAL_HTTP_MAX_RETRY"] == "3"

    def test_extra_kwargs_stringified(self):
        applied = configure(GDAL_CACHEMAX=512)
        assert applied["GDAL_CACHEMAX"] == "512"

    def test_no_args_returns_empty(self):
        applied = configure()
        assert applied == {}


class TestConfigureAppliesToGdal:
    """Each key in the returned dict is visible via gdal.GetConfigOption."""

    def test_single_option_round_trip(self):
        configure(GDAL_CACHEMAX="64")
        assert gdal.GetConfigOption("GDAL_CACHEMAX") == "64"

    def test_cloud_defaults_applied_to_gdal(self):
        configure(cloud_defaults=True)
        assert gdal.GetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN") == "EMPTY_DIR"
        assert gdal.GetConfigOption("GDAL_HTTP_MULTIRANGE") == "YES"


class TestCredentialExpansion:
    """``aws=``, ``gs=``, ``azure=`` expand via ``_expand_credentials``."""

    def test_aws_unsigned_shortcut(self):
        out = _expand_credentials("AWS", {"aws_unsigned": True})
        assert out == {"AWS_NO_SIGN_REQUEST": "YES"}

    def test_aws_keys_upcased(self):
        out = _expand_credentials("AWS", {"access_key_id": "AKIA123"})
        assert out == {"AWS_ACCESS_KEY_ID": "AKIA123"}

    def test_none_input_empty(self):
        assert _expand_credentials("AWS", None) == {}

    def test_empty_dict_empty(self):
        assert _expand_credentials("GS", {}) == {}

    def test_gs_prefix(self):
        out = _expand_credentials("GS", {"project_id": "my-proj"})
        assert out == {"GS_PROJECT_ID": "my-proj"}

    def test_azure_prefix(self):
        out = _expand_credentials("AZURE", {"storage_account": "acct"})
        assert out == {"AZURE_STORAGE_ACCOUNT": "acct"}


class TestConfigureAwsEndToEnd:
    """`configure(aws=...)` applies expanded keys to GDAL."""

    def test_aws_unsigned_visible_in_gdal(self):
        configure(aws={"aws_unsigned": True})
        assert gdal.GetConfigOption("AWS_NO_SIGN_REQUEST") == "YES"


class TestImportSurface:
    """`pyramids.configure` is reachable from the package root."""

    def test_importable_from_pyramids_root(self):
        import pyramids

        assert callable(pyramids.configure)


class TestWorkerPlugin:
    """Tests for :func:`pyramids._configure._register_worker_plugin`.

    ``configure(..., client=<dask client>)`` replays the chosen GDAL
    options on every worker via a :class:`WorkerPlugin` so distributed
    reads see the same config as the driver. These tests cover the
    client-broadcast branch of :func:`configure`.

    Both client-broadcast tests import :mod:`dask.distributed`
    internally through the ``configure`` code path, so they carry the
    ``lazy`` marker to route the test into the ``[lazy]`` extras env.
    The module-level ``pytestmark = pytest.mark.core`` still applies to
    every other test in this file because ``configure`` without a
    client is pure GDAL.
    """

    @pytest.mark.lazy
    def test_client_triggers_plugin_registration(self):
        """Passing ``client`` calls ``client.register_plugin`` once.

        Test scenario:
            With a mock client, a single :func:`configure` call with a
            non-empty env must fan out to exactly one
            ``register_plugin`` invocation, passing the well-known
            plugin name.
        """
        fake_client = MagicMock()
        configure(client=fake_client, GDAL_CACHEMAX="64")
        fake_client.register_plugin.assert_called_once()
        call = fake_client.register_plugin.call_args
        assert call.kwargs.get("name") == "pyramids-configure", (
            f"Expected plugin name 'pyramids-configure', got "
            f"{call.kwargs.get('name')!r}"
        )

    @pytest.mark.lazy
    def test_client_fallback_to_register_worker_plugin(self):
        """Old dask API (no ``register_plugin``) falls back cleanly.

        Test scenario:
            A client whose :meth:`register_plugin` raises
            :class:`AttributeError` (emulating a legacy dask build)
            must fall through to :meth:`register_worker_plugin` instead
            of crashing the caller.
        """
        fake_client = MagicMock()
        fake_client.register_plugin.side_effect = AttributeError
        configure(client=fake_client, GDAL_CACHEMAX="128")
        fake_client.register_worker_plugin.assert_called_once()

    def test_no_client_skips_plugin(self):
        """Default ``client=None`` must not touch any plugin API.

        Test scenario:
            When no dask client is passed, :func:`configure` is a pure
            in-process operation — no registration calls should fire.
        """
        fake_client = MagicMock()
        configure(GDAL_CACHEMAX="64")
        fake_client.register_plugin.assert_not_called()
        fake_client.register_worker_plugin.assert_not_called()
