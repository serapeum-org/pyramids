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
from pyramids.netcdf._plot_options import ColourOpts, FacetSpec, Selectors

try:
    # The distribution name on PyPI is `pyramids-gis`, but the import
    # name is `pyramids`. importlib.metadata.version() needs the dist
    # name; passing __name__ would always raise PackageNotFoundError
    # and quietly set __version__ to "unknown".
    __version__ = _get_version("pyramids-gis")
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
