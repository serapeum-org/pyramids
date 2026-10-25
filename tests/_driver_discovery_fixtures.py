"""Shared fixture helpers for the raster and vector driver-discovery test suites.

Both hooks' tests inject fake driver classes two ways — straight into the registry cache, or via
in-memory entry points so real discovery runs. The bodies are identical apart from the drivers
module and the published-attribute prefix, so they live here and each suite wraps them in thin
``@pytest.fixture`` functions.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint
from typing import Any, Callable


def make_register(monkeypatch: Any, module: Any) -> Callable[..., Any]:
    """Build a callable that publishes driver classes straight into ``module._CACHE``.

    Args:
        monkeypatch: The test's pytest ``monkeypatch`` fixture.
        module: The hook's ``_drivers`` module whose ``_CACHE`` is set.

    Returns:
        A callable taking driver classes; it returns the single driver, or the tuple of drivers.
    """

    def _register(*drivers: Any) -> Any:
        monkeypatch.setattr(module, "_CACHE", {driver.name: driver for driver in drivers})
        return drivers[0] if len(drivers) == 1 else drivers

    return _register


def make_publish_entry_points(
    monkeypatch: Any, module: Any, prefix: str
) -> tuple[Callable[..., None], list[str]]:
    """Build a publisher that registers in-memory entry points on ``module``, plus its cleanup list.

    The caller yields the publisher from its fixture, then deletes each name in the returned list
    from ``module`` on teardown.

    Args:
        monkeypatch: The test's pytest ``monkeypatch`` fixture.
        module: The hook's ``_drivers`` module whose ``entry_points`` is patched.
        prefix: A per-suite attribute-name prefix for the published driver factories.

    Returns:
        A ``(publish, published)`` pair — ``publish(*(name, class))`` registers the entry points, and
        ``published`` is the list of module attributes to delete on teardown.
    """
    published: list[str] = []
    group = module.ENTRY_POINT_GROUP

    def _publish(*pairs: Any) -> None:
        entries = []
        for index, (name, factory) in enumerate(pairs):
            attribute = f"{prefix}{index}"
            published.append(attribute)
            setattr(module, attribute, factory)
            entries.append(
                EntryPoint(
                    name=name,
                    value=f"{module.__name__}:{attribute}",
                    group=group,
                )
            )

        def _entry_points(*, group: str | None = None) -> tuple:
            return tuple(entries) if group == module.ENTRY_POINT_GROUP else ()

        monkeypatch.setattr(module, "entry_points", _entry_points)

    return _publish, published
