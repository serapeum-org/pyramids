"""Guard the shared GDAL pin and its wiring against silent breakage.

The conda GDAL stack is declared once, in ``[tool.pixi.feature.gdal.dependencies]``, and
shared into every pixi environment by listing the ``gdal`` feature. The SWIG ``osgeo``
bindings are ABI-locked to ``libgdal``, so that single pin must reach every env — dev / CI
and the lean ``wheel-build`` container alike (the latter is ``no-default-feature``, so it
cannot inherit the pin from ``[tool.pixi.dependencies]``).

Single-sourcing removes version drift, but introduces one new way to break it: an
environment that forgets to list the ``gdal`` feature resolves with no GDAL at all. These
tests enforce the invariants that keep the wiring sound — every environment references the
feature, the feature pins all four packages, and ``ci/gdal-pin.py`` (which CI uses to
install gdal in the conda-forge-shape wheel test) reads that same table.
"""

import importlib.util
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
GDAL_PIN_SCRIPT = REPO_ROOT / "ci" / "gdal-pin.py"

# The conda packages the shared gdal feature must pin. swig is build-only and lives in
# the wheel-build feature, so it is intentionally excluded here.
GDAL_PACKAGES = ["gdal", "libgdal-netcdf", "libgdal-hdf4", "libgdal-grib"]


def _load_pixi() -> dict:
    """Return the ``[tool.pixi]`` table from ``pyproject.toml``."""
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["pixi"]


def _env_features(env_spec) -> list:
    """Return an environment's feature list (pixi allows a bare list or a table)."""
    return env_spec["features"] if isinstance(env_spec, dict) else env_spec


# The packages ci/setup-gdal-micromamba.sh asks ci/gdal-pin.py to resolve — the shared
# gdal libs plus build-only swig. gdal-pin.py must serve every one of them.
MICROMAMBA_PACKAGES = ["gdal", "libgdal-netcdf", "libgdal-hdf4", "swig"]


def _load_gdal_pin():
    """Import ``ci/gdal-pin.py`` as a module (its hyphenated path is not importable)."""
    spec = importlib.util.spec_from_file_location("gdal_pin", GDAL_PIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gdal_feature_pins_every_package():
    """The shared gdal feature must pin exactly the four conda GDAL packages."""
    deps = _load_pixi()["feature"]["gdal"]["dependencies"]
    assert sorted(deps) == sorted(GDAL_PACKAGES)


@pytest.mark.parametrize("env", sorted(_load_pixi()["environments"]))
def test_every_environment_references_gdal_feature(env: str):
    """Every pixi env must list the ``gdal`` feature, or it resolves with no GDAL at all."""
    features = _env_features(_load_pixi()["environments"][env])
    assert "gdal" in features, (
        f"environment {env!r} does not list the 'gdal' feature; it would resolve without "
        "the shared GDAL pin. Add 'gdal' to its features list in [tool.pixi.environments]."
    )


def test_gdal_pin_script_matches_pyproject():
    """``ci/gdal-pin.py`` must emit the shared gdal pin CI installs the wheel against."""
    assert _load_gdal_pin().gdal_spec(PYPROJECT) == _load_pixi()["feature"]["gdal"]["dependencies"]["gdal"]


def test_gdal_pin_script_resolves_micromamba_packages():
    """``ci/gdal-pin.py`` must resolve every pin ci/setup-gdal-micromamba.sh requests."""
    pins = _load_gdal_pin().feature_pins()
    missing = [pkg for pkg in MICROMAMBA_PACKAGES if pkg not in pins]
    assert not missing, f"ci/gdal-pin.py cannot resolve {missing} that setup-gdal-micromamba.sh requests"
