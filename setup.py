"""Minimal setup.py that forces setuptools to produce a platform-specific wheel.

pyramids has no C extensions of its own, but when built via cibuildwheel we
vendor the `osgeo` package (compiled SWIG bindings) and bundled native
libraries into the wheel. Without this override, setuptools produces a
``py3-none-any`` wheel and cibuildwheel rightly refuses it.

See planning/bundle/option-1-implementation-plan.md Task 1.10 for context.
All other config lives in pyproject.toml.
"""
from __future__ import annotations

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Tell setuptools this wheel is platform-specific.

    We return True unconditionally because the wheel always ships vendored
    C extensions from the GDAL SWIG bindings + bundled shared libraries.
    """

    def has_ext_modules(self) -> bool:  # noqa: D401
        return True


setup(distclass=BinaryDistribution)
