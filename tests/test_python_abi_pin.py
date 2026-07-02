"""Guard the Python 3.14 ABI pin against non-deterministic free-threaded drift.

conda-forge ships two Python 3.14 builds at the same version: ``*_cp314`` (standard, GIL) and
``*_cp314t`` (free-threaded / no-GIL, PEP 703). A bare ``python = "3.14.*"`` lets the conda solver
pick either, and it did so inconsistently per platform — the ``py314`` test env resolved ``*_cp314``
while the ``wheel-build`` env picked ``*_cp314t`` on ``osx-64`` / ``win-64`` — so the 3.14 legs were
not comparing like for like. pyramids has thread-sensitive code (per-thread GDAL handle management,
``base/_locks.py``, the engine ``weakref`` back-references), so the 3.14 leg must test one intended
ABI (standard GIL). These tests enforce that the manifest pins ``*_cp314`` in every feature whose
Python resolves to 3.14, and that the committed ``pixi.lock`` carries no free-threaded ``*_cp314t``
build on any platform.
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PIXI_LOCK = REPO_ROOT / "pixi.lock"

# Features whose Python resolves to 3.14 and must therefore pin the standard-GIL ABI. ``py314`` is
# the 3.14 test matrix env; ``wheel-build`` is the ``no-default-feature`` cibuildwheel env that
# floats Python to the newest available (3.14) with no ABI constraint of its own.
PY314_FEATURES = ["py314", "wheel-build"]


def _load_pixi() -> dict:
    """Return the ``[tool.pixi]`` table from ``pyproject.toml``."""
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["pixi"]


@pytest.mark.parametrize("feature", PY314_FEATURES)
def test_py314_feature_pins_standard_gil_abi(feature: str):
    """Each 3.14 feature must pin python to the standard-GIL ``*_cp314`` build."""
    python = _load_pixi()["feature"][feature]["dependencies"]["python"]
    # pixi accepts a table ({version=, build=}) or a bare matchspec string ("3.14.* *_cp314").
    build = python.get("build", "") if isinstance(python, dict) else python
    assert "cp314t" not in build and "*_cp314" in build, (
        f"feature {feature!r} python spec {python!r} does not pin the standard-GIL '*_cp314' "
        "ABI; a bare '3.14.*' lets the solver pick the free-threaded '*_cp314t' build."
    )


def test_lockfile_has_no_free_threaded_python():
    """The committed ``pixi.lock`` must carry no free-threaded ``cp314t`` Python 3.14 build."""
    lock = PIXI_LOCK.read_text(encoding="utf-8")
    offenders = sorted(set(re.findall(r"python(?:_abi)?-3\.14[^\s/]*_cp314t", lock)))
    assert not offenders, (
        f"pixi.lock contains free-threaded (cp314t) Python 3.14 builds: {offenders}. Re-pin the "
        "ABI in pyproject.toml and regenerate the lock with `pixi update`."
    )
