"""Guard the Python 3.14 ABI against non-deterministic free-threaded drift.

conda-forge ships two Python 3.14 builds at the same version: ``*_cp314`` (standard, GIL) and
``*_cp314t`` (free-threaded / no-GIL, PEP 703). A bare ``python = "3.14.*"`` lets the conda solver
pick either, and it did so inconsistently per platform — the ``wheel-build`` env picked ``*_cp314t``
on ``osx-64`` / ``win-64`` while the ``py314`` env used ``*_cp314`` everywhere. pyramids has
thread-sensitive code (per-thread GDAL handle management, ``base/_locks.py``), so the 3.14 legs must
run one intended ABI (standard GIL).

Two complementary guards, because the ABI cannot be constrained the same way everywhere:

- :func:`test_feature_targeting_314_pins_gil_abi` — every pixi *feature* that explicitly targets
  Python 3.14 (currently ``py314`` and ``wheel-build``) must pin the standard-GIL ``*_cp314`` build.
  The feature list is derived from the manifest, so a newly added 3.14 feature is checked
  automatically.
- :func:`test_lockfile_has_no_free_threaded_python` — the committed ``pixi.lock`` must carry no
  ``cp314t`` build on any platform. This is the **universal** backstop: the envs that share the
  ``default`` / ``docs`` solve-groups float Python to the newest (3.14) with no ABI pin of their own
  and cannot be ABI-constrained at solve time without freezing their version (a shared pin would also
  break the ``py311`` / ``py312`` / ``py313`` envs). This whole-lock scan is therefore what
  guarantees none of those envs drifts to the free-threaded interpreter on the next re-solve, and it
  also backstops the per-feature guard against manifest forms it does not parse (see below).
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PIXI_LOCK = REPO_ROOT / "pixi.lock"


def _load_pixi() -> dict:
    """Return the ``[tool.pixi]`` table from ``pyproject.toml``."""
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["pixi"]


def _python_spec(feature: dict):
    """Return a feature's top-level ``python`` dependency spec (bare string, table, or ``None``).

    Only ``[...dependencies]`` is inspected, not per-platform
    ``target.<platform>.dependencies`` tables. A 3.14 pin placed under a ``target`` table would be
    missed by the per-feature check, but :func:`test_lockfile_has_no_free_threaded_python` scans the
    whole lock and is the universal backstop for any form this helper does not parse.
    """
    return feature.get("dependencies", {}).get("python")


def _spec_tokens(python) -> list[str]:
    """Tokens of a bare matchspec string, split on whitespace or ``=`` (``"3.14.* *_cp314"`` or
    ``"3.14.*=*_cp314"``)."""
    return [t for t in re.split(r"[ =]+", str(python).strip()) if t] if python else []


def _spec_version(python) -> str:
    """Version half of a pixi python spec — table ``{version=}`` or a bare matchspec string."""
    if isinstance(python, dict):
        return str(python.get("version", ""))
    tokens = _spec_tokens(python)
    return tokens[0] if tokens else ""


def _spec_build(python) -> str:
    """Build half of a pixi python spec (``""`` if the spec carries no build string)."""
    if isinstance(python, dict):
        return str(python.get("build", ""))
    tokens = _spec_tokens(python)
    return tokens[1] if len(tokens) > 1 else ""


def _features_targeting_314() -> list[str]:
    """Names of pixi features whose top-level ``python`` spec explicitly targets 3.14."""
    features = _load_pixi().get("feature", {})
    return sorted(
        name
        for name, feat in features.items()
        if _spec_version(_python_spec(feat)).startswith("3.14")
    )


def test_known_314_features_are_detected():
    """py314 and wheel-build must still be detected as 3.14-targeting features."""
    found = set(_features_targeting_314())
    assert {"py314", "wheel-build"} <= found, (
        "expected 'py314' and 'wheel-build' among the features that target Python 3.14; found "
        f"{sorted(found)}. If a pin was dropped, the ABI is no longer constrained for that feature."
    )


@pytest.mark.parametrize("feature", _features_targeting_314())
def test_feature_targeting_314_pins_gil_abi(feature: str):
    """Each feature targeting Python 3.14 must pin the standard-GIL ``*_cp314`` build."""
    python = _python_spec(_load_pixi()["feature"][feature])
    # Reject the free-threaded ABI anywhere in the spec (so an =-delimited or oddly-formatted
    # cp314t pin can't hide in the version half), and require the standard-GIL cp314 build pin.
    assert "cp314t" not in str(python), (
        f"feature {feature!r} python spec {python!r} allows the free-threaded 'cp314t' ABI."
    )
    assert "cp314" in _spec_build(python), (
        f"feature {feature!r} python spec {python!r} does not pin the standard-GIL 'cp314' build; a "
        "bare '3.14.*' lets the solver pick either ABI."
    )


def test_lockfile_has_no_free_threaded_python():
    """The committed ``pixi.lock`` must carry no free-threaded ``cp314t`` build, in any env/platform."""
    lock = PIXI_LOCK.read_text(encoding="utf-8")
    # Any token containing ``cp314t`` is a free-threaded build — catches both the artifact filenames
    # (``python-3.14.6-..._cp314t.conda``) and the space-delimited matchspec lines (``python_abi
    # 3.14.* *_cp314t``). ``cp314t`` appears nowhere else in the lock.
    offenders = sorted(set(re.findall(r"\S*cp314t\S*", lock)))
    assert not offenders, (
        f"pixi.lock contains free-threaded (cp314t) Python builds: {offenders}. Re-pin the ABI in "
        "pyproject.toml and regenerate the lock with `pixi update python python_abi`."
    )
