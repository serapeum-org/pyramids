"""pyramids - GIS utility package"""

from __future__ import annotations

import os as _os
import sys as _sys
import sysconfig as _sysconfig
import warnings as _warnings
from pathlib import Path as _Path

# Bootstrap vendored GDAL if present (platform wheel installs).
# When installed from a platform wheel, the _vendor/osgeo/ directory contains
# the GDAL SWIG Python bindings and _data/ contains GDAL_DATA and PROJ_DATA.
# This block must run BEFORE any `from osgeo import ...` statement.
_pkg_dir = _Path(__file__).parent
_vendored_osgeo = _pkg_dir / "_vendor" / "osgeo"

# Activate the vendored osgeo only if its SWIG extension matches the
# current Python ABI. A mismatch typically means `_vendor/` is
# leftover from a local `cibuildwheel --only cp3XX-...` run targeting
# a different interpreter than the one currently importing pyramids
# (e.g. dev runs cibuildwheel for cp312, then later `pip install -e .`
# in a cp313 env). Better to fall back to the system osgeo than to
# ImportError inside the SWIG module with a confusing trace.
_vendor_abi_match = False
if _vendored_osgeo.is_dir():
    _ext_suffix = _sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    _vendor_abi_match = (_vendored_osgeo / f"_gdal{_ext_suffix}").is_file()
    if not _vendor_abi_match:
        _py_ver = f"{_sys.version_info.major}.{_sys.version_info.minor}"
        _warnings.warn(
            f"pyramids: ignoring src/pyramids/_vendor/ — no "
            f"_gdal{_ext_suffix} found (vendored for a different Python "
            f"ABI). Remove src/pyramids/_vendor and src/pyramids/_data "
            f"to silence this warning, or rebuild for Python {_py_ver}.",
            stacklevel=2,
        )

if _vendor_abi_match:
    _vendor_str = str(_pkg_dir / "_vendor")
    if _vendor_str not in _sys.path:
        _sys.path.insert(0, _vendor_str)

    _data_dir = _pkg_dir / "_data"
    _gdal_data = _data_dir / "gdal_data"
    _proj_data = _data_dir / "proj_data"
    _gdal_plugins = _data_dir / "gdalplugins"
    # setdefault is intentional: user-set GDAL_DATA / PROJ_DATA /
    # PROJ_LIB / GDAL_DRIVER_PATH always win over the wheel's bundled
    # data dirs. This lets advanced users override (e.g. point PROJ at
    # a custom grid bundle) without having to unset our values first.
    # Side effect: changing these env vars after import has no effect
    # — the bootstrap reads them once at first `import pyramids`.
    if _gdal_data.is_dir():
        _os.environ.setdefault("GDAL_DATA", str(_gdal_data))
    if _proj_data.is_dir():
        _os.environ.setdefault("PROJ_DATA", str(_proj_data))
        _os.environ.setdefault("PROJ_LIB", str(_proj_data))
    if _gdal_plugins.is_dir():
        # GDAL loads NetCDF / HDF4 / HDF5 drivers from this dir.
        _os.environ.setdefault("GDAL_DRIVER_PATH", str(_gdal_plugins))

    if _sys.platform == "win32":  # pragma: no cover
        # delvewheel places DLLs at <site-packages>/pyramids_gis.libs/
        # (one level up from this package) and injects its own
        # add_dll_directory call near the top of this file. The block
        # below is a safety net for both that layout and the older
        # pyramids/.libs/ convention; the vendored osgeo/__init__.py
        # also sets the DLL directory on import so spawn workers that
        # import osgeo without pyramids first still resolve gdal.dll.
        # The returned handle is stored module-level so GC can't
        # silently remove the directory from the search path.
        #
        # We also prepend the libs dir to PATH because GDAL's native
        # plugin loader uses raw LoadLibrary (no SEARCH_USER_DIRS
        # flag), which doesn't honor add_dll_directory. PATH is the
        # only env-controlled fallback in the legacy DLL search order.
        _PYRAMIDS_DLL_HANDLE = None
        for _candidate in (_pkg_dir / ".libs", _pkg_dir.parent / "pyramids_gis.libs"):
            if _candidate.is_dir():
                _PYRAMIDS_DLL_HANDLE = _os.add_dll_directory(str(_candidate))
                _candidate_str = str(_candidate)
                _path = _os.environ.get("PATH", "")
                if _candidate_str not in _path.split(_os.pathsep):
                    _os.environ["PATH"] = _candidate_str + _os.pathsep + _path
                break

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
