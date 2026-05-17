"""Install GDAL Python bindings and vendor them into src/pyramids/_vendor/.

Runs once per target Python version in ``CIBW_BEFORE_BUILD``:

1. ``pip install GDAL==$GDAL_VERSION`` against the pixi-extracted libgdal
   sitting under ``$BUILD_PREFIX/lib`` (populated by
   ``ci/setup-gdal-from-pixi.sh`` in ``CIBW_BEFORE_ALL``).
2. Copy the freshly-built ``osgeo`` package from the target Python's
   ``site-packages/`` into ``src/pyramids/_vendor/osgeo/``.
3. Copy ``$BUILD_PREFIX/share/gdal`` and ``$BUILD_PREFIX/share/proj``
   into ``src/pyramids/_data/`` so setuptools includes them as
   package-data in the wheel.

Activation is gated by ``PACKAGE_DATA=1`` to avoid accidentally running
during local editable installs.

See planning/bundle/option-1-implementation-plan.md Task 1.5.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
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

    # macOS arm64: pin numpy>=2.1 in the build venv so the SWIG bindings
    # are compiled against the numpy 2.x C API. Wheels built against
    # numpy 1.x raise
    #   "A module that was compiled using NumPy 1.x cannot be run in
    #    NumPy 2.x"
    # at import time on any end-user machine that has numpy 2.x — which
    # is everyone, since numpy 2.0 shipped in mid-2024. numpy >=2.1 is
    # the earliest 2.x version that avoids the macos-14 Accelerate
    # ILP64 symbol bug (_cblas_caxpy$NEWLAPACK$ILP64 not found) that
    # made earlier 2.x wheels fail to import on this runner.
    # We use --no-build-isolation so pip uses the ambient build venv
    # (where we've just installed numpy>=2.1) instead of pulling
    # whatever GDAL's build-system.requires lists.
    # x86_64 cross-compile is unaffected: numpy 2.x ships cp313 wheels
    # and meson refuses to cross-build numpy 1.x from source.
    extra_pip_args: list[str] = []
    target_arch = (
        os.environ.get("CIBW_ARCHS")
        or os.environ.get("CIBW_ARCHS_MACOS")
        or ""
    )
    print(f"[install-and-vendor-osgeo] target arch: {target_arch!r}", flush=True)
    if is_macos and target_arch == "arm64":
        pre = [
            sys.executable, "-m", "pip", "install", "--no-cache-dir",
            "setuptools>=77.0.3", "wheel", "numpy>=2.1,<3",
        ]
        print(f"[install-and-vendor-osgeo] pre-install (macOS arm64): {' '.join(pre)}", flush=True)
        subprocess.check_call(pre, env=env)
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
