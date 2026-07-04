"""ARC-V7: ``pyramids.configure_lazy_vector`` — vector-side dask defaults.

Mirror of :func:`pyramids.configure` for the lazy-vector path. Pin:

* Explicit ``scheduler`` is applied to ``dask.config``.
* ``target_bytes_per_partition`` updates the module-level constant used
  by :func:`_resolve_lazy_partitioning`.
* Both are returned in the applied-settings dict.
* ``client`` kwarg registers a worker plugin (best-effort test —
  constructor success only; actual worker replay is tested upstream
  by dask itself).
* ``None`` kwargs leave state unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pyramids import configure_lazy_vector
from pyramids.feature import collection as _fc_mod

# The whole module exercises dask internals. The ``lazy`` marker gates
# the tests to the env where dask is installed; we therefore cannot
# ``import dask`` at module scope because pytest collects every test
# file in every env (core, xarray, parquet, ...) before it evaluates
# markers, and a missing-dask ImportError at that point fails
# collection outright. Every function / fixture that touches dask
# imports it locally — no skip, no global import, just deferred
# lookup until the lazy env actually runs the test body.
pytestmark = pytest.mark.lazy


@pytest.fixture
def restore_lazy_target():
    """Snapshot + restore the module-level bytes-per-partition constant."""
    saved = _fc_mod._LAZY_TARGET_BYTES_PER_PARTITION
    yield
    _fc_mod._LAZY_TARGET_BYTES_PER_PARTITION = saved


@pytest.fixture
def restore_dask_scheduler():
    """Snapshot + restore the dask scheduler global config."""
    import dask

    saved = dask.config.get("scheduler", default=None)
    yield
    if saved is None:
        dask.config.refresh()
    else:
        dask.config.set(scheduler=saved)


class TestSchedulerApply:
    def test_scheduler_applied_to_dask_config(self, restore_dask_scheduler):
        """Passing ``scheduler='processes'`` updates ``dask.config``."""
        import dask

        configure_lazy_vector(scheduler="processes")
        assert dask.config.get("scheduler") == "processes"

    def test_scheduler_synchronous_applied(self, restore_dask_scheduler):
        """``scheduler='synchronous'`` is accepted and applied."""
        import dask

        configure_lazy_vector(scheduler="synchronous")
        assert dask.config.get("scheduler") == "synchronous"

    def test_none_scheduler_leaves_config_alone(
        self,
        restore_dask_scheduler,
    ):
        """``scheduler=None`` must not touch dask.config."""
        import dask

        dask.config.set(scheduler="synchronous")
        configure_lazy_vector(scheduler=None)
        assert dask.config.get("scheduler") == "synchronous"


class TestTargetBytes:
    def test_target_bytes_updates_module_constant(self, restore_lazy_target):
        """Passing ``target_bytes_per_partition`` patches the module constant."""
        configure_lazy_vector(target_bytes_per_partition=64 * 1024 * 1024)
        assert _fc_mod._LAZY_TARGET_BYTES_PER_PARTITION == 64 * 1024 * 1024

    def test_none_leaves_constant_alone(self, restore_lazy_target):
        """``target_bytes_per_partition=None`` does not touch the constant."""
        before = _fc_mod._LAZY_TARGET_BYTES_PER_PARTITION
        configure_lazy_vector(target_bytes_per_partition=None)
        assert _fc_mod._LAZY_TARGET_BYTES_PER_PARTITION == before

    def test_integer_coercion(self, restore_lazy_target):
        """Float inputs are coerced to int (bytes-count is always integral)."""
        configure_lazy_vector(target_bytes_per_partition=42.9)
        assert _fc_mod._LAZY_TARGET_BYTES_PER_PARTITION == 42


class TestReturnValue:
    def test_empty_call_returns_empty_dict(
        self,
        restore_lazy_target,
        restore_dask_scheduler,
    ):
        """No kwargs → no settings applied → empty return."""
        assert configure_lazy_vector() == {}

    def test_returns_applied_settings(
        self,
        restore_lazy_target,
        restore_dask_scheduler,
    ):
        """The return dict reports exactly what was applied."""
        result = configure_lazy_vector(
            scheduler="processes",
            target_bytes_per_partition=256 * 1024 * 1024,
        )
        assert result == {
            "scheduler": "processes",
            "target_bytes_per_partition": 256 * 1024 * 1024,
        }


class TestWorkerPlugin:
    def test_client_triggers_plugin_registration(
        self,
        restore_lazy_target,
        restore_dask_scheduler,
    ):
        """Passing ``client`` calls ``client.register_plugin``."""
        fake_client = MagicMock()
        configure_lazy_vector(
            scheduler="processes",
            client=fake_client,
        )
        fake_client.register_plugin.assert_called_once()
        # The plugin is registered under our unique name.
        call = fake_client.register_plugin.call_args
        assert call.kwargs.get("name") == "pyramids-configure-lazy-vector"

    def test_client_fallback_to_register_worker_plugin(
        self,
        restore_lazy_target,
        restore_dask_scheduler,
    ):
        """Old dask API (no ``register_plugin``) falls back cleanly."""
        fake_client = MagicMock()
        fake_client.register_plugin.side_effect = AttributeError
        configure_lazy_vector(
            scheduler="processes",
            client=fake_client,
        )
        fake_client.register_worker_plugin.assert_called_once()
