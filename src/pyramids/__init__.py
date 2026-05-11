"""pyramids - GIS utility package"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

from pyramids._configure import configure, configure_lazy_vector
from pyramids.base.config import Config
from pyramids.netcdf._plot_options import ColourOpts, FacetSpec, Selectors

try:
    __version__ = _get_version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

config = Config()


__all__ = [
    "configure",
    "configure_lazy_vector",
    "config",
    "__version__",
    "Selectors",
    "ColourOpts",
    "FacetSpec",
]
