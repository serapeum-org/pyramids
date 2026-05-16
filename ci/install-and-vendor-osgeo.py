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


def install_gdal_python_bindings() -> None:
    """pip install ``GDAL==$GDAL_VERSION`` linking against $BUILD_PREFIX.

    Uses environment variables (CPPFLAGS / LDFLAGS / GDAL_CONFIG) rather
    than pip's deprecated ``--global-option``. GDAL's setup.py consults
    ``gdal-config`` first, so as long as that points at our extracted
    binary, headers and libs are discovered correctly.
    """
    version = _gdal_version()
    prefix = _build_prefix()

    env = os.environ.copy()
    env["GDAL_CONFIG"] = str(prefix / "bin" / "gdal-config")
    env["CPPFLAGS"] = f"-I{prefix}/include " + env.get("CPPFLAGS", "")
    env["LDFLAGS"] = f"-L{prefix}/lib " + env.get("LDFLAGS", "")

    # numpy's include path is needed by GDAL's swig/python/setup.py.
    # We invoke sys.executable to ask numpy where its headers live
    # rather than importing numpy in this script — that's because on
    # macOS, the script's `import numpy` can fail even when numpy IS
    # installed in the same venv (the macOS Python framework loader
    # has subtle search-path quirks). Going via the same interpreter
    # that pip will use ensures we get a consistent answer.
    numpy_include = subprocess.run(
        [sys.executable, "-c", "import numpy; print(numpy.get_include())"],
        capture_output=True, text=True, check=False,
    )
    if numpy_include.returncode == 0:
        env["CPPFLAGS"] = f"-I{numpy_include.stdout.strip()} " + env["CPPFLAGS"]
    else:
        print(
            f"[install-and-vendor-osgeo] WARNING: numpy not importable via "
            f"{sys.executable}: {numpy_include.stderr.strip()}",
            flush=True,
        )

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir",
        "--no-build-isolation",
        f"GDAL=={version}",
    ]
    print(f"[install-and-vendor-osgeo] running: {' '.join(cmd)}", flush=True)
    print(f"[install-and-vendor-osgeo] GDAL_CONFIG={env['GDAL_CONFIG']}", flush=True)
    print(f"[install-and-vendor-osgeo] CPPFLAGS={env['CPPFLAGS']}", flush=True)
    print(f"[install-and-vendor-osgeo] LDFLAGS={env['LDFLAGS']}", flush=True)
    subprocess.check_call(cmd, env=env)


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

    # 2. Vendor GDAL_DATA
    gdal_data_src = prefix / "share" / "gdal"
    if not gdal_data_src.is_dir():
        raise RuntimeError(f"GDAL_DATA not found at {gdal_data_src}")
    _copy_tree_replacing(gdal_data_src, src_pyramids / "_data" / "gdal_data")

    # 3. Vendor PROJ_DATA
    proj_data_src = prefix / "share" / "proj"
    if not proj_data_src.is_dir():
        raise RuntimeError(f"PROJ_DATA not found at {proj_data_src}")
    _copy_tree_replacing(proj_data_src, src_pyramids / "_data" / "proj_data")

    # 4. Vendor GDAL plugins (NetCDF / HDF4 / HDF5 drivers).
    # GDAL loads these at runtime when GDAL_DRIVER_PATH points here.
    plugins_src = prefix / "lib" / "gdalplugins"
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
