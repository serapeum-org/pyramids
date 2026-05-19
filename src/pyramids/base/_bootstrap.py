"""Runtime vendor bootstrap for the pyramids platform wheel.

When `pyramids` is installed from a platform wheel,
`pyramids/_vendor/osgeo/` ships GDAL's Python SWIG bindings and
`pyramids/_data/` ships GDAL_DATA + PROJ_DATA.
`activate_vendored_osgeo(pkg_dir)` injects `_vendor/` into `sys.path`
and configures the env vars + Windows DLL directories that the bundled
GDAL needs. It is called once from `pyramids/__init__.py` BEFORE any
`from osgeo import ...` anywhere downstream — `pyramids.base._configure`
does that import at module load and would fail without the bootstrap.

The bootstrap is a no-op on editable / dev installs that don't have
`_vendor/` populated (the normal `pixi shell -e dev` setup) and skips
with a warning if the vendored `_gdal.<EXT>` doesn't match the
current Python ABI (defensive against stale leftover dirs from a
prior local cibuildwheel run).

Lives under `pyramids/base/` (not at the package root) so internal
infrastructure modules are grouped together. `pkg_dir` is passed
explicitly by the caller (resolved via `pyramids.__path__`) rather
than derived from `__file__`, so this module can move anywhere within
the package without the vendor-dir resolution breaking.
"""
from __future__ import annotations

import os
import sys
import sysconfig
import warnings
from pathlib import Path

# Module-level handle from `os.add_dll_directory()` on Windows.
# add_dll_directory returns a context-manager-style handle; if the
# handle gets garbage-collected the directory falls off the DLL
# search path mid-process. Anchor it at module scope so the bootstrap
# module's lifetime (= the Python process's lifetime) keeps the
# directory registered.
_DLL_HANDLE = None


def activate_vendored_osgeo(pkg_dir: Path) -> bool:
    """Activate the vendored osgeo if `_vendor/osgeo/` exists + ABI-matches.

    Args:
        pkg_dir: directory of the `pyramids` package (containing
            `_vendor/` and `_data/`). The caller is responsible for
            resolving this — typically ``Path(pyramids.__path__[0])``
            from `pyramids/__init__.py`. No default is provided so the
            resolution stays at the caller and this module never has
            to guess its own location (which would be wrong now that
            it lives under `pyramids/base/`).

    Returns:
        ``True`` if the vendor bootstrap activated (sys.path was
        modified, env vars were set). ``False`` if no `_vendor/`
        directory exists or if its `_gdal.<EXT>` doesn't match the
        current Python ABI.
    """
    global _DLL_HANDLE
    vendored_osgeo = pkg_dir / "_vendor" / "osgeo"

    if not vendored_osgeo.is_dir():
        return False

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    if not (vendored_osgeo / f"_gdal{ext_suffix}").is_file():
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        warnings.warn(
            f"pyramids: ignoring src/pyramids/_vendor/ — no "
            f"_gdal{ext_suffix} found (vendored for a different Python "
            f"ABI). Remove src/pyramids/_vendor and src/pyramids/_data "
            f"to silence this warning, or rebuild for Python {py_ver}.",
            stacklevel=2,
        )
        return False

    vendor_str = str(pkg_dir / "_vendor")
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)

    data_dir = pkg_dir / "_data"
    gdal_data = data_dir / "gdal_data"
    proj_data = data_dir / "proj_data"
    gdal_plugins = data_dir / "gdalplugins"
    # setdefault is intentional: user-set GDAL_DATA / PROJ_DATA /
    # PROJ_LIB / GDAL_DRIVER_PATH always win over the wheel's bundled
    # data dirs. This lets advanced users override (e.g. point PROJ at
    # a custom grid bundle) without having to unset our values first.
    # Side effect: changing these env vars after import has no effect
    # — the bootstrap reads them once at first `import pyramids`.
    if gdal_data.is_dir():
        os.environ.setdefault("GDAL_DATA", str(gdal_data))
    if proj_data.is_dir():
        os.environ.setdefault("PROJ_DATA", str(proj_data))
        os.environ.setdefault("PROJ_LIB", str(proj_data))
    if gdal_plugins.is_dir():
        # GDAL loads NetCDF / HDF4 / HDF5 drivers from this dir on
        # Windows. The directory is populated by
        # install-and-vendor-osgeo only on Windows builds —
        # conda-forge's libgdal on Linux/macOS links those drivers in
        # statically, so the dir is absent there and the is_dir()
        # guard makes the bootstrap cross-platform.
        os.environ.setdefault("GDAL_DRIVER_PATH", str(gdal_plugins))

    if sys.platform == "win32":  # pragma: no cover
        # delvewheel places DLLs at <site-packages>/pyramids_gis.libs/
        # (one level up from this package) and injects its own
        # add_dll_directory call near the top of the vendored
        # osgeo/__init__.py. The block below is a safety net for both
        # that layout and the older pyramids/.libs/ convention; the
        # vendored osgeo/__init__.py also sets the DLL directory on
        # import so spawn workers that import osgeo without pyramids
        # first still resolve gdal.dll.
        #
        # We also prepend the libs dir to PATH because GDAL's native
        # plugin loader uses raw LoadLibrary (no SEARCH_USER_DIRS
        # flag), which doesn't honor add_dll_directory. PATH is the
        # only env-controlled fallback in the legacy DLL search order.
        for candidate in (pkg_dir / ".libs", pkg_dir.parent / "pyramids_gis.libs"):
            if candidate.is_dir():
                _DLL_HANDLE = os.add_dll_directory(str(candidate))
                candidate_str = str(candidate)
                path = os.environ.get("PATH", "")
                if candidate_str not in path.split(os.pathsep):
                    os.environ["PATH"] = candidate_str + os.pathsep + path
                break

    if os.environ.get("PYRAMIDS_DEBUG_BOOTSTRAP"):  # pragma: no cover
        print(f"[pyramids] vendor dir: {vendor_str}")
        print(f"[pyramids] GDAL_DATA: {os.environ.get('GDAL_DATA')}")
        print(f"[pyramids] PROJ_DATA: {os.environ.get('PROJ_DATA')}")

    return True
