# pyramids vendor patch: ensure GDAL's bundled DLLs are loadable when
# osgeo is imported directly (i.e. without pyramids being imported
# first). The delvewheel patch in pyramids/__init__.py only runs when
# the parent pyramids package is imported, but multiprocessing.spawn
# workers replay sys.path from the parent so osgeo at
# pyramids/_vendor/osgeo/ becomes importable without pyramids itself
# being imported — and os.add_dll_directory state is process-local
# so the parent's call doesn't carry across. Run the DLL setup here
# unconditionally on Windows so every interpreter that imports osgeo
# gets it.
#
# Both mechanisms are needed:
#   - os.add_dll_directory: Python's import machinery uses
#     LoadLibraryEx(LOAD_LIBRARY_SEARCH_USER_DIRS), which honors it
#     when loading _gdal.pyd and friends.
#   - PATH prepend: GDAL's native plugin loader uses raw LoadLibrary
#     (no SEARCH_USER_DIRS flag), so add_dll_directory is invisible
#     to it. The legacy DLL search order still walks PATH, so the
#     bundled netcdf/hdf5/hdf4 deps of the GDAL plugins become
#     findable.
#
# This file is read at wheel-build time by
# ci/install-and-vendor-osgeo.py and prepended into the vendored
# `osgeo/__init__.py` after any leading comments and `from __future__`
# imports. It is NEVER imported as a module — the `_pyramids_*`
# identifiers below would clash with user code if they were.
import os as _pyramids_os
import sys as _pyramids_sys

if _pyramids_sys.platform == "win32":
    _pyramids_libs = _pyramids_os.path.abspath(
        _pyramids_os.path.join(
            _pyramids_os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "pyramids_gis.libs",
        )
    )
    if _pyramids_os.path.isdir(_pyramids_libs):
        _pyramids_os.add_dll_directory(_pyramids_libs)
        _pyramids_path = _pyramids_os.environ.get("PATH", "")
        if _pyramids_libs not in _pyramids_path.split(_pyramids_os.pathsep):
            _pyramids_os.environ["PATH"] = (
                _pyramids_libs + _pyramids_os.pathsep + _pyramids_path
            )
        del _pyramids_path
    del _pyramids_libs
del _pyramids_os, _pyramids_sys
