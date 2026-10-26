"""A reusable conformance kit for ``pyramids.raster_drivers`` plugins.

A downstream distribution that registers a raster driver runs these checks against its driver
*class* in its own test suite — under any test runner, since the kit raises plain
:class:`AssertionError` and needs no pytest. It verifies the same contract
:mod:`pyramids.dataset._drivers` enforces at discovery (so a driver that conforms here is never
silently skipped at runtime), plus the behavioural promises that static validation cannot see:
:meth:`~pyramids.dataset._drivers.RasterDriver.can_open` is callable on the *class* (never needing
an instance), returns a ``bool``, and never raises on a hostile input; and the driver builds with
no arguments (as :meth:`Dataset.read_file` dispatches it) and exposes a callable ``read``.

The checks themselves live in :mod:`pyramids.base._driver_discovery` and are shared with the vector
conformance kit; this module binds them to the raster hook's static validator.

Usage in a plugin's tests::

    from pyramids.dataset.driver_conformance import assert_raster_driver_conforms
    from my_plugin import FciDriver

    def test_driver_conforms():
        assert_raster_driver_conforms(FciDriver)
"""

from __future__ import annotations

from typing import Any

from pyramids.base._driver_discovery import assert_driver_conforms, check_driver
from pyramids.dataset._drivers import _validate

__all__ = ["check_raster_driver", "assert_raster_driver_conforms"]


def check_raster_driver(driver: Any) -> list[str]:
    """Return every way a raster driver class departs from the contract.

    Args:
        driver: The driver class to check (the class itself, not an instance).

    Returns:
        A list of human-readable problems; empty when the driver conforms. The first entry, if
        any, is the static-claims problem that discovery itself would reject on.

    Examples:
        - A well-formed driver class has no problems:
            ```python
            >>> from pyramids.dataset.driver_conformance import check_raster_driver
            >>> class FciDriver:
            ...     name, long_name, api_version = "fci", "EUMETSAT FCI", 1
            ...     suffixes, schemes = (".nat",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            ...     @classmethod
            ...     def can_open(cls, source): return str(source).endswith(".nat")
            ...     def read(self, source, **kw): ...
            >>> check_raster_driver(FciDriver)
            []

            ```
        - A missing capability is reported:
            ```python
            >>> from pyramids.dataset.driver_conformance import check_raster_driver
            >>> class Bad:
            ...     name, long_name, api_version = "bad", "Bad", 1
            ...     suffixes, schemes = (".x",), ()
            ...     capabilities, extra = frozenset(), None
            ...     @classmethod
            ...     def can_open(cls, source): return False
            ...     def read(self, source, **kw): ...
            >>> check_raster_driver(Bad)[0]
            "does not declare the required 'read' capability"

            ```
    """
    return check_driver(driver, validate=_validate)


def assert_raster_driver_conforms(driver: Any) -> None:
    """Raise :class:`AssertionError` when a raster driver class violates the contract.

    Args:
        driver: The driver class to check.

    Raises:
        AssertionError: The driver departs from the contract; the message lists every problem.

    Examples:
        - A conforming driver passes silently:
            ```python
            >>> from pyramids.dataset.driver_conformance import assert_raster_driver_conforms
            >>> class FciDriver:
            ...     name, long_name, api_version = "fci", "EUMETSAT FCI", 1
            ...     suffixes, schemes = (".nat",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            ...     @classmethod
            ...     def can_open(cls, source): return str(source).endswith(".nat")
            ...     def read(self, source, **kw): ...
            >>> assert_raster_driver_conforms(FciDriver)

            ```
    """
    assert_driver_conforms(driver, validate=_validate, kind="raster")
