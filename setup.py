"""Minimal setup.py: conditional platform-wheel tagging.

pyramids has no C extensions of its own, but the cibuildwheel path vendors the
`osgeo` package (compiled SWIG bindings) + bundled native libs into the wheel,
which needs platform tags (cp3NN-cp3NN-<plat>) — otherwise cibuildwheel rejects
the wheel.

Gate that behavior on `PACKAGE_DATA=1`, the same env var cibuildwheel exports
via `[tool.cibuildwheel.{linux,macos,windows}.environment]` blocks in
pyproject.toml (also keys ci/install-and-vendor-osgeo.py). Outside cibuildwheel
the var is unset, so plain `python -m build --wheel` (wheel-test.yml,
conda-forge sdist build) emits a `py3-none-any` wheel as expected.

All other config lives in pyproject.toml.
"""
from __future__ import annotations

import os

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


if os.environ.get("PACKAGE_DATA") == "1":
    setup(distclass=BinaryDistribution)
else:
    setup()
