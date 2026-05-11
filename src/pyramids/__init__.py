"""pyramids - GIS utility package"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

# Bootstrap vendored GDAL if present (platform wheel installs).
# When installed from a platform wheel, the _vendor/osgeo/ directory contains
# the GDAL SWIG Python bindings and _data/ contains GDAL_DATA and PROJ_DATA.
# This block must run BEFORE any `from osgeo import ...` statement.
_pkg_dir = _Path(__file__).parent
_vendored_osgeo = _pkg_dir / "_vendor" / "osgeo"

if _vendored_osgeo.is_dir():
    _vendor_str = str(_pkg_dir / "_vendor")
    if _vendor_str not in _sys.path:
        _sys.path.insert(0, _vendor_str)

    _data_dir = _pkg_dir / "_data"
    _gdal_data = _data_dir / "gdal_data"
    _proj_data = _data_dir / "proj_data"
    _gdal_plugins = _data_dir / "gdalplugins"
    if _gdal_data.is_dir():
        _os.environ.setdefault("GDAL_DATA", str(_gdal_data))
    if _proj_data.is_dir():
        _os.environ.setdefault("PROJ_DATA", str(_proj_data))
        _os.environ.setdefault("PROJ_LIB", str(_proj_data))
    if _gdal_plugins.is_dir():
        # GDAL loads NetCDF / HDF4 / HDF5 drivers from this dir.
        _os.environ.setdefault("GDAL_DRIVER_PATH", str(_gdal_plugins))

    if _sys.platform == "win32":  # pragma: no cover
        _libs_dir = _pkg_dir / ".libs"
        if _libs_dir.is_dir():
            _os.add_dll_directory(str(_libs_dir))

    if _os.environ.get("PYRAMIDS_DEBUG_BOOTSTRAP"):  # pragma: no cover
        print(f"[pyramids] vendor dir: {_vendor_str}")
        print(f"[pyramids] GDAL_DATA: {_os.environ.get('GDAL_DATA')}")
        print(f"[pyramids] PROJ_DATA: {_os.environ.get('PROJ_DATA')}")

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
