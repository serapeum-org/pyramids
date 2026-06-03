"""Print the gdal version spec the wheels are built against (single source of truth).

Reads ``[tool.pixi.feature.gdal.dependencies].gdal`` from ``pyproject.toml`` and prints
it verbatim (e.g. ``>=3.12,<3.13``). That feature is shared by every pixi environment,
so the value is the one the published wheels vendor. CI's conda-forge-shape wheel test
(``.github/workflows/pure-wheel-test.yml``) uses the output to ``conda install`` gdal at
that exact pin, so the test environment can never drift to a different gdal than the one
we build and ship against.

Usage (from the repo root)::

    spec=$(python ci/gdal-pin.py)        # ->  >=3.12,<3.13
    conda install -y "gdal${spec}" "libgdal-netcdf${spec}" ...
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def gdal_spec(pyproject: Path) -> str:
    """Return the shared gdal version spec from ``pyproject``."""
    with pyproject.open("rb") as fh:
        cfg = tomllib.load(fh)
    return cfg["tool"]["pixi"]["feature"]["gdal"]["dependencies"]["gdal"]


def main() -> int:
    """Print the spec; the project root is the current working directory."""
    print(gdal_spec(Path("pyproject.toml")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
