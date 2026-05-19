"""pyramids - GIS utility package"""
from __future__ import annotations

from pathlib import Path

# The vendor bootstrap MUST run before any `from osgeo import …`
# anywhere downstream. `pyramids.base._configure` does that import at
# module load (line below), and `_configure` would fail to import the
# SWIG ext unless `_vendor/` is on sys.path. Bootstrap mechanism +
# rationale live in `pyramids.base._bootstrap`.
from pyramids.base._bootstrap import activate_vendored_osgeo

activate_vendored_osgeo(Path(__path__[0]))

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

from pyramids.base._configure import configure, configure_lazy_vector
from pyramids.base.config import Config

try:
    __version__ = _get_version("pyramids-gis")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

Config()


__all__ = [
    "configure",
    "configure_lazy_vector",
    "__version__",
]
