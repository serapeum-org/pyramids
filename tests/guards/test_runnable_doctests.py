"""Lock the runnable docstring examples so they cannot silently rot.

These modules' docstring `>>>` examples run offline today; this test re-runs them
in CI so a future edit that breaks an example is caught. Examples that need
external data are marked ``# doctest: +SKIP`` in the source and are skipped here.

To add a module: confirm ``pytest --doctest-modules <path>`` passes, then list it.
"""

import doctest
import importlib
import os

import pytest

DOCTEST_MODULES = [
    "pyramids.dataset.engines.spatial",
    "pyramids.dataset.engines.vectorize",
    "pyramids.base.crs",
    "pyramids.base.config",
    "pyramids.io.sniff",
]


@pytest.fixture(autouse=True)
def _restore_path():
    """Snapshot PATH around each doctest run.

    The ``EnvironmentVariables.prepend`` example mutates ``os.environ['PATH']``
    and restores it in its last line — but a failure earlier in the example
    would leak the mutation into the rest of the test session.
    """
    original = os.environ.get("PATH", "")
    yield
    os.environ["PATH"] = original


@pytest.mark.parametrize("module_name", DOCTEST_MODULES)
def test_module_doctests(module_name: str) -> None:
    module = importlib.import_module(module_name)
    result = doctest.testmod(
        module,
        verbose=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert result.failed == 0, f"{module_name}: {result.failed} doctest failure(s)"
