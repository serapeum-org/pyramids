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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _gdal_version() -> str:
    """Return the GDAL_VERSION env var, failing fast if unset."""
    version = os.environ.get("GDAL_VERSION")
    if not version:
        raise RuntimeError(
            "GDAL_VERSION env var is required. Set it in "
            "[tool.cibuildwheel.linux.environment] to the version pixi/"
            "conda-forge delivered (check `gdal-config --version`)."
        )
    return version


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
    """pip install ``GDAL==$GDAL_VERSION`` linking against $BUILD_PREFIX.

    GDAL 3.12.x setup.py uses two different discovery mechanisms:

    * Unix (Linux/macOS, ``unix`` compiler): runs ``gdal-config`` via PATH
      lookup. GDAL does NOT read the ``GDAL_CONFIG`` env var — only the
      ``--gdal-config=PATH`` setup option or the binary on PATH. So we
      prepend ``${BUILD_PREFIX}/bin`` to PATH.

    * Windows (``msvc`` compiler): SKIPS gdal-config entirely and relies
      on MSVC conventions — ``INCLUDE`` env var for headers, ``LIB`` env
      var for library dirs. Conda Windows packages nest under
      ``${BUILD_PREFIX}/Library/`` so we point INCLUDE/LIB there.

    We let pip use build isolation so it pulls in the right setuptools
    (>=77 per GDAL's build-system.requires). On macOS we additionally
    apply a PIP_CONSTRAINT pinning numpy<2, because GDAL's build-system
    requires numpy>=2 by default and numpy 2.x on macos-14 runners has
    the Accelerate `_cblas_caxpy$NEWLAPACK$ILP64` symbol bug — the
    build venv's numpy fails to import, and GDAL's setup.py then
    raises "numpy not available".
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

    # On macOS, the build venv that pip's isolation creates installs
    # numpy>=2 per GDAL's build-system.requires. numpy 2.x on macos-14
    # has the Accelerate ILP64 symbol bug — fails to import — and then
    # GDAL's setup.py errors with "numpy not available". Constrain the
    # build env to numpy<2 via PIP_CONSTRAINT, which pip respects for
    # build-isolation envs too.
    constraint_file: Path | None = None
    if is_macos:
        constraint_file = REPO_ROOT / "ci" / "_macos_build_constraints.txt"
        constraint_file.write_text("numpy<2\n", encoding="utf-8")
        env["PIP_CONSTRAINT"] = str(constraint_file)

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir",
        f"GDAL=={version}",
    ]
    print(f"[install-and-vendor-osgeo] platform: {sys.platform}", flush=True)
    print(f"[install-and-vendor-osgeo] BUILD_PREFIX: {prefix}", flush=True)
    print(f"[install-and-vendor-osgeo] PATH (head): {env.get('PATH', '')[:300]}", flush=True)
    if is_windows:
        print(f"[install-and-vendor-osgeo] INCLUDE: {env.get('INCLUDE', '')}", flush=True)
        print(f"[install-and-vendor-osgeo] LIB: {env.get('LIB', '')}", flush=True)
    if is_macos:
        print(f"[install-and-vendor-osgeo] PIP_CONSTRAINT: {env.get('PIP_CONSTRAINT', '')}", flush=True)
    print(f"[install-and-vendor-osgeo] running: {' '.join(cmd)}", flush=True)
    try:
        subprocess.check_call(cmd, env=env)
    finally:
        if constraint_file is not None and constraint_file.exists():
            constraint_file.unlink()


def _copy_tree_replacing(src: Path, dst: Path) -> None:
    """Copy a directory, removing the destination first if it exists."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[install-and-vendor-osgeo] copy {src} -> {dst}", flush=True)
    shutil.copytree(src, dst)


def vendor_osgeo_into_package() -> None:
    """Copy osgeo and osgeo_utils modules + GDAL/PROJ data files into src/pyramids/."""
    import osgeo  # imported lazily so install step runs first

    src_pyramids = REPO_ROOT / "src" / "pyramids"
    osgeo_src = Path(osgeo.__file__).parent
    prefix = _build_prefix()

    # 1. Vendor osgeo/
    vendor_dir = src_pyramids / "_vendor"
    _copy_tree_replacing(osgeo_src, vendor_dir / "osgeo")
    (vendor_dir / "__init__.py").touch()

    # 1b. Vendor osgeo_utils/ — GDAL's pip package ships a sibling
    # top-level package for utility scripts (gdal_polygonize, etc.).
    # Some of pyramids' tests / third-party code imports it.
    try:
        import osgeo_utils
        osgeo_utils_src = Path(osgeo_utils.__file__).parent
        _copy_tree_replacing(osgeo_utils_src, vendor_dir / "osgeo_utils")
    except ImportError:
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
