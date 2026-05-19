"""pyramids - GIS utility package"""
from __future__ import annotations

from pathlib import Path

# The vendor bootstrap MUST run before any `from osgeo import …`
# anywhere downstream. `pyramids.base._configure` does that import at
# module load (line below), and `_configure` would fail to import the
# SWIG ext unless `_vendor/` is on sys.path. Bootstrap mechanism +
# rationale live in `pyramids.base._bootstrap`.
#
# pkg_dir is resolved through `__path__` rather than `__file__` so the
# bootstrap module can move anywhere within the package without the
# resolution breaking. Python sets `__path__` on the partial pyramids
# module before any code in this file runs, so it's safe to read here
# even though we're still mid-import.
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

config = Config()


# NetCDF plot-config dataclasses (Selectors, ColourOpts, FacetSpec)
# used to be re-exported from this module. They're now in
# `pyramids.netcdf.plot_options` (no longer underscore-prefixed) and
# re-exported by `pyramids.netcdf` itself — so the canonical import is
#   from pyramids.netcdf import Selectors, ColourOpts, FacetSpec
# rather than `from pyramids import Selectors, ...`. The top-level
# re-export was an inconsistency: every other class in the package
# (Dataset, NetCDF, FeatureCollection, DatasetCollection) has always
# required its full sub-module path. Aligning these three with that
# convention.
__all__ = [
    "configure",
    "configure_lazy_vector",
    "config",
    "__version__",
]
