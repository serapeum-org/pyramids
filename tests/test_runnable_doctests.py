"""Lock the runnable docstring examples so they cannot silently rot.

These modules' docstring `>>>` examples run offline today; this test re-runs them
in CI so a future edit that breaks an example is caught. Examples that need
external data are marked ``# doctest: +SKIP`` in the source and are skipped here.

To add a module: confirm ``pytest --doctest-modules <path>`` passes, then list it.
"""

import doctest
import importlib

import pytest

DOCTEST_MODULES = [
    "pyramids.dataset.engines.spatial",
    "pyramids.base.crs",
    "pyramids.base.config",
    "pyramids.io.sniff",
]


@pytest.mark.parametrize("module_name", DOCTEST_MODULES)
def test_module_doctests(module_name: str) -> None:
    module = importlib.import_module(module_name)
    result = doctest.testmod(
        module,
        verbose=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert result.failed == 0, f"{module_name}: {result.failed} doctest failure(s)"
