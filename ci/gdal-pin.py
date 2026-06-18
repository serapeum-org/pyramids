"""Print the conda pins the wheels are built against (single source of truth).

The native build/test stack is pinned once in two ``pyproject.toml`` features:
``[tool.pixi.feature.gdal.dependencies]`` (the shared gdal / libgdal-* libs every
environment lists) and ``[tool.pixi.feature.wheel-build.dependencies]`` (build-only
``swig``). This script reads those features and prints each requested package's version
spec verbatim, one per line, so every consumer installs the exact pins the published
wheels vendor instead of re-encoding them:

- ``.github/workflows/pure-wheel-test.yml`` — the conda-forge-shape wheel test
- ``ci/setup-gdal-micromamba.sh`` — the macOS x86_64 cross-compile env

``pyproject.toml`` is located relative to this file, so the script works from any cwd.

Usage::

    spec=$(python ci/gdal-pin.py)                    # default: gdal ->  >=3.13,<3.14
    python ci/gdal-pin.py gdal libgdal-netcdf swig   # one spec per line
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def feature_pins(pyproject: Path = PYPROJECT) -> dict[str, str]:
    """Return the conda pins from the gdal and wheel-build features, merged by name."""
    with pyproject.open("rb") as fh:
        feature = tomllib.load(fh)["tool"]["pixi"]["feature"]
    return {**feature["gdal"]["dependencies"], **feature["wheel-build"]["dependencies"]}


def gdal_spec(pyproject: Path = PYPROJECT) -> str:
    """Return the shared gdal version spec from ``pyproject``."""
    return feature_pins(pyproject)["gdal"]


def main(argv: list[str]) -> int:
    """Print the spec for each requested package (default: gdal), one per line."""
    pins = feature_pins()
    print("\n".join(pins[name] for name in (argv or ["gdal"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
