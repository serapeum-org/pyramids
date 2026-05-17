"""Install GDAL Python bindings and vendor them into src/pyramids/_vendor/.

Runs once per target Python version in ``CIBW_BEFORE_BUILD``:

1. ``pip install GDAL==<version>`` against the pixi-extracted libgdal
   sitting under ``$BUILD_PREFIX/lib`` (populated by
   ``ci/setup-gdal-from-pixi.{sh,ps1}`` in ``CIBW_BEFORE_ALL``). The
   concrete version is read from ``${BUILD_PREFIX}/GDAL_VERSION`` which
   those scripts write at the resolution of pixi.lock / micromamba.
2. Copy the freshly-built ``osgeo`` package from the target Python's
   ``site-packages/`` into ``src/pyramids/_vendor/osgeo/``.
3. Copy ``$BUILD_PREFIX/share/gdal`` and ``$BUILD_PREFIX/share/proj``
   into ``src/pyramids/_data/`` so setuptools includes them as
   package-data in the wheel.

Activation is gated by ``PACKAGE_DATA=1`` to avoid accidentally running
during local editable installs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _gdal_version() -> str:
    """Return the concrete GDAL version that BEFORE_ALL resolved.

    ``ci/setup-gdal-from-pixi.{sh,ps1}`` writes the version pixi /
    micromamba actually installed into ``${BUILD_PREFIX}/GDAL_VERSION``
    so this script can ``pip install GDAL==X.Y.Z`` against the exact
    libgdal binary we bundle. Falls back to the ``GDAL_VERSION`` env
    var for transitional / out-of-band invocations.
    """
    version_file = _build_prefix() / "GDAL_VERSION"
    if version_file.is_file():
        version = version_file.read_text().strip()
        if version:
            return version
    version = os.environ.get("GDAL_VERSION", "").strip()
    if version:
        return version
    raise RuntimeError(
        f"GDAL version not resolved: expected {version_file} written by "
        "ci/setup-gdal-from-pixi.{sh,ps1} or GDAL_VERSION env var set."
    )


def _build_prefix() -> Path:
    return Path(os.environ.get("BUILD_PREFIX", "/usr/local"))


def _data_layout_roots(prefix: Path) -> tuple[Path, Path, Path]:
    """Return (bin_dir, share_dir, lib_dir) for the current OS.

    Conda Windows packages nest under ``<prefix>/Library/`` (Anaconda's
    Windows convention). Linux and macOS use the standard Unix layout
    directly under ``<prefix>/``.
    """
    if sys.platform == "win32" or os.name == "nt":
        win_lib_root = prefix / "Library"
        return win_lib_root / "bin", win_lib_root / "share", win_lib_root / "lib"
    return prefix / "bin", prefix / "share", prefix / "lib"


def install_gdal_python_bindings() -> None:
    """pip install ``GDAL==<resolved version>`` linking against $BUILD_PREFIX.

    GDAL 3.12.x's setup.py uses two discovery mechanisms:

    * Unix (Linux/macOS, ``unix`` compiler): runs ``gdal-config`` via PATH
      lookup. So we prepend ``${BUILD_PREFIX}/bin`` to PATH.
    * Windows (``msvc`` compiler): skips ``gdal-config`` and reads the
      MSVC ``INCLUDE`` / ``LIB`` env vars. Conda's Windows packages nest
      under ``${BUILD_PREFIX}/Library/`` so we point INCLUDE/LIB there.

    On macOS we cap parallel build jobs to keep peak memory below the
    macos-14 runner's ~7 GB ceiling, and on arm64 we pre-install a
    setuptools/numpy build venv (see the inline comment below the
    parallelism caps for the Accelerate-ILP64 carve-out).
    """
    version = _gdal_version()
    prefix = _build_prefix()
    is_windows = sys.platform == "win32" or os.name == "nt"
    is_macos = sys.platform == "darwin"

    env = os.environ.copy()

    if is_windows:
        bin_dir, _, lib_dir = _data_layout_roots(prefix)
        include_dir = prefix / "Library" / "include"
        # MSVC convention: ';'-separated INCLUDE / LIB env vars
        env["INCLUDE"] = f"{include_dir};{env.get('INCLUDE', '')}"
        env["LIB"] = f"{lib_dir};{env.get('LIB', '')}"
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    else:
        bin_dir, _, _ = _data_layout_roots(prefix)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"

    # macos-14 GitHub runners SIGKILL xcodebuild on every /usr/bin shim
    # invocation (clang, otool, install_name_tool, codesign, ...).
    # CIBW_BEFORE_ALL (setup-gdal-from-pixi.sh) plants symlinks in
    # /usr/local/bin pointing at the real Xcode toolchain binaries, so
    # PATH resolution skips the /usr/bin shims entirely. No CC / CXX /
    # DEVELOPER_DIR plumbing is needed here — just cap parallel build
    # jobs to keep peak RAM under the runner ceiling.
    if is_macos:
        env.setdefault("MAKEFLAGS", "-j2")
        env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
        env.setdefault("NPY_NUM_BUILD_JOBS", "2")

    # macOS arm64 needs special handling for the build-venv numpy:
    #
    # * We can't use numpy 1.x — wheels compiled against numpy 1.x raise
    #   "A module that was compiled using NumPy 1.x cannot be run in
    #   NumPy 2.x" on any end-user machine with numpy 2.x.
    # * We can't let pip pick a numpy 2.x wheel for the host either —
    #   cibuildwheel's framework Python identifies as macosx-14, so pip
    #   downloads numpy-X.Y.Z-cpNNN-cpNNN-macosx_14_0_arm64.whl. That
    #   wheel uses Accelerate ILP64 symbols that aren't actually present
    #   on the runner, so numpy fails to import and GDAL's setup.py
    #   then reports "numpy not available".
    #
    # Force-download the macosx_11_0_arm64 numpy wheel (built against
    # the older non-ILP64 Accelerate) and install it without deps. That
    # gives the build venv a working numpy 2.x ABI, and the resulting
    # _gdal_array.so is forward-compatible with any numpy 2.x runtime.
    #
    # x86_64 cross-compile takes the standard path: numpy 2.x ships cp
    # wheels for osx-64 and meson refuses to cross-build numpy 1.x.
    extra_pip_args: list[str] = []
    target_arch = (
        os.environ.get("CIBW_ARCHS")
        or os.environ.get("CIBW_ARCHS_MACOS")
        or ""
    )
    print(f"[install-and-vendor-osgeo] target arch: {target_arch!r}", flush=True)
    if is_macos and target_arch == "arm64":
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "setuptools>=77.0.3", "wheel"],
            env=env,
        )
        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        download_dir = Path(tempfile.mkdtemp(prefix="numpy-dl-"))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "download",
             "--no-deps", "--only-binary=:all:",
             "--platform", "macosx_11_0_arm64",
             "--python-version", f"{sys.version_info.major}.{sys.version_info.minor}",
             "--implementation", "cp",
             "--abi", py_tag,
             "-d", str(download_dir),
             "numpy>=2.1,<3"],
            env=env,
        )
        numpy_whl = next(download_dir.glob("numpy-*.whl"))
        print(f"[install-and-vendor-osgeo] using {numpy_whl.name}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", str(numpy_whl)],
            env=env,
        )
        extra_pip_args.append("--no-build-isolation")

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir",
        *extra_pip_args,
        f"GDAL=={version}",
    ]
    print(f"[install-and-vendor-osgeo] platform: {sys.platform}", flush=True)
    print(f"[install-and-vendor-osgeo] BUILD_PREFIX: {prefix}", flush=True)
    print(f"[install-and-vendor-osgeo] PATH (head): {env.get('PATH', '')[:300]}", flush=True)
    if is_windows:
        print(f"[install-and-vendor-osgeo] INCLUDE: {env.get('INCLUDE', '')}", flush=True)
        print(f"[install-and-vendor-osgeo] LIB: {env.get('LIB', '')}", flush=True)
    print(f"[install-and-vendor-osgeo] running: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, env=env)


def _copy_tree_replacing(src: Path, dst: Path) -> None:
    """Copy a directory, removing the destination first if it exists."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[install-and-vendor-osgeo] copy {src} -> {dst}", flush=True)
    shutil.copytree(src, dst)


_OSGEO_DLL_BOOTSTRAP = '''\
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
import os as _pyramids_os
import sys as _pyramids_sys
if _pyramids_sys.platform == "win32":
    _pyramids_libs = _pyramids_os.path.abspath(
        _pyramids_os.path.join(
            _pyramids_os.path.dirname(__file__),
            "..", "..", "..", "pyramids_gis.libs",
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
'''


def _patch_vendored_osgeo_init(init_path: Path) -> None:
    """Inject a Windows DLL bootstrap into the vendored osgeo/__init__.py.

    Without this, importing ``osgeo`` in a multiprocessing.spawn worker
    fails with ``ImportError: DLL load failed while importing _gdal``
    because the parent's ``os.add_dll_directory`` call doesn't carry to
    spawn children, and the worker imports osgeo before pyramids.

    The bootstrap is spliced AFTER any leading comments, blank lines,
    and ``from __future__`` imports so the patch stays valid if
    upstream osgeo ever adds a future-import (PEP 236 requires
    __future__ imports to precede any other statement). The current
    conda-forge osgeo doesn't use __future__, but ``gdal=3.12.*`` is
    intentionally loose and a 3.12.x bump could add one without our
    pin moving.
    """
    original = init_path.read_text(encoding="utf-8")
    if "pyramids vendor patch" in original:
        return
    lines = original.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (not stripped
                or stripped.startswith("#")
                or stripped.startswith("from __future__")):
            insert_at = i + 1
            continue
        break
    patched = (
        "".join(lines[:insert_at])
        + _OSGEO_DLL_BOOTSTRAP
        + "".join(lines[insert_at:])
    )
    init_path.write_text(patched, encoding="utf-8")
    print(
        f"[install-and-vendor-osgeo] patched {init_path} with Windows DLL bootstrap",
        flush=True,
    )


def _locate_site_packages_dir(name: str) -> Path | None:
    """Return the on-disk directory of an installed top-level package.

    We deliberately do NOT ``import`` the package: on Windows the GDAL
    SWIG extension loads ``gdal.dll`` and its transitive dep DLLs at
    import time, and a runtime symbol mismatch between Python 3.13's
    bundled vcruntime and conda-forge's bundled vcruntime triggers
    "DLL load failed: specified procedure could not be found" (cp311/
    cp312 don't hit this). The vendoring step only needs the on-disk
    location of the package — no Python code from it actually runs —
    so we look it up via the active environment's purelib path.
    """
    purelib = Path(sysconfig.get_paths()["purelib"])
    candidate = purelib / name
    if candidate.is_dir():
        return candidate
    return None


def vendor_osgeo_into_package() -> None:
    """Copy osgeo and osgeo_utils modules + GDAL/PROJ data files into src/pyramids/."""
    prefix = _build_prefix()

    osgeo_src = _locate_site_packages_dir("osgeo")
    if osgeo_src is None:
        raise RuntimeError(
            "osgeo/ not found in site-packages after pip install GDAL. "
            f"Searched {sysconfig.get_paths()['purelib']}."
        )

    src_pyramids = REPO_ROOT / "src" / "pyramids"

    # 1. Vendor osgeo/
    vendor_dir = src_pyramids / "_vendor"
    _copy_tree_replacing(osgeo_src, vendor_dir / "osgeo")
    (vendor_dir / "__init__.py").touch()
    _patch_vendored_osgeo_init(vendor_dir / "osgeo" / "__init__.py")

    # 1b. Vendor osgeo_utils/ — GDAL's pip package ships a sibling
    # top-level package for utility scripts (gdal_polygonize, etc.).
    # Some of pyramids' tests / third-party code imports it.
    osgeo_utils_src = _locate_site_packages_dir("osgeo_utils")
    if osgeo_utils_src is not None:
        _copy_tree_replacing(osgeo_utils_src, vendor_dir / "osgeo_utils")
    else:
        print("[install-and-vendor-osgeo] osgeo_utils not found; skipping", flush=True)

    # Conda Windows packages nest data under <prefix>/Library/share and
    # plugins under <prefix>/Library/lib/gdalplugins. Linux/macOS use
    # <prefix>/share and <prefix>/lib/gdalplugins directly.
    _, share_dir, lib_dir = _data_layout_roots(prefix)

    # 2. Vendor GDAL_DATA
    gdal_data_src = share_dir / "gdal"
    if not gdal_data_src.is_dir():
        raise RuntimeError(f"GDAL_DATA not found at {gdal_data_src}")
    _copy_tree_replacing(gdal_data_src, src_pyramids / "_data" / "gdal_data")

    # 3. Vendor PROJ_DATA
    proj_data_src = share_dir / "proj"
    if not proj_data_src.is_dir():
        raise RuntimeError(f"PROJ_DATA not found at {proj_data_src}")
    _copy_tree_replacing(proj_data_src, src_pyramids / "_data" / "proj_data")

    # 4. Vendor GDAL plugins (NetCDF / HDF4 / HDF5 drivers).
    # GDAL loads these at runtime when GDAL_DRIVER_PATH points here.
    plugins_src = lib_dir / "gdalplugins"
    if plugins_src.is_dir():
        _copy_tree_replacing(plugins_src, src_pyramids / "_data" / "gdalplugins")


def main() -> None:
    if os.environ.get("PACKAGE_DATA") != "1":
        print("[install-and-vendor-osgeo] PACKAGE_DATA != 1; skipping.", flush=True)
        return
    install_gdal_python_bindings()
    vendor_osgeo_into_package()
    print("[install-and-vendor-osgeo] done.", flush=True)


if __name__ == "__main__":
    main()
