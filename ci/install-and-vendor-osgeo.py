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

import glob
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

    # On macOS, /usr/bin/clang(++) is a stub that asks xcrun to dispatch
    # via xcodebuild for every invocation. On the macos-14 GitHub runner
    # we've observed xcodebuild getting SIGKILLed even on a single
    # upfront probe, with xcrun then reporting
    #   "unable to find utility 'clang', not a developer tool or in PATH"
    # and downstream compiles failing with
    #   "xcode-select: Failed to locate 'clang++'".
    # The fix is to bypass xcrun/xcodebuild entirely: glob the Xcode app
    # bundle for the toolchain clang directly and pin CC/CXX/SDKROOT to
    # those absolute paths. Falls back to CommandLineTools and then to
    # /usr/bin if no Xcode bundle is present. Also caps parallel build
    # jobs to keep peak memory below the runner ceiling.
    if is_macos:
        toolchain_clang_candidates = sorted(
            glob.glob(
                "/Applications/Xcode*.app/Contents/Developer/Toolchains/"
                "XcodeDefault.xctoolchain/usr/bin/clang"
            ),
            reverse=True,
        )
        toolchain_clang_candidates.append(
            "/Library/Developer/CommandLineTools/usr/bin/clang"
        )
        cc_path: str | None = None
        for candidate in toolchain_clang_candidates:
            if Path(candidate).is_file():
                cc_path = candidate
                break
        if cc_path is not None:
            cxx_path = cc_path + "++"
            env["CC"] = cc_path
            env["CXX"] = cxx_path
            developer_dir = cc_path.split("/Toolchains/")[0]
            if developer_dir.endswith("/Developer"):
                env["DEVELOPER_DIR"] = developer_dir
                sdk_candidates = sorted(
                    glob.glob(
                        f"{developer_dir}/Platforms/MacOSX.platform/"
                        "Developer/SDKs/MacOSX*.sdk"
                    ),
                    reverse=True,
                )
                if sdk_candidates:
                    env["SDKROOT"] = sdk_candidates[0]
            print(f"[install-and-vendor-osgeo] resolved CC={env['CC']}", flush=True)
            print(f"[install-and-vendor-osgeo] resolved CXX={env['CXX']}", flush=True)
            if "DEVELOPER_DIR" in env:
                print(
                    f"[install-and-vendor-osgeo] DEVELOPER_DIR={env['DEVELOPER_DIR']}",
                    flush=True,
                )
            if "SDKROOT" in env:
                print(f"[install-and-vendor-osgeo] SDKROOT={env['SDKROOT']}", flush=True)
        else:
            print(
                "[install-and-vendor-osgeo] no Xcode/CLT toolchain found; "
                "leaving CC/CXX unset",
                flush=True,
            )
        env.setdefault("MAKEFLAGS", "-j2")
        env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
        env.setdefault("NPY_NUM_BUILD_JOBS", "2")

    # On macOS arm64, pip's build isolation installs numpy>=2 (per
    # GDAL's build-system.requires) into a fresh build venv. numpy 2.x
    # on macos-14 arm64 has the Accelerate ILP64 symbol bug —
    # _multiarray_umath fails to dlopen — and then GDAL's setup.py
    # errors with "numpy not available".
    #
    # PIP_CONSTRAINT can't override build-system.requires (it can only
    # ADD constraints, not replace deps), so we pre-install
    # setuptools>=77 + numpy<2 + wheel in the build venv and then
    # install GDAL with --no-build-isolation.
    #
    # This is arm64-specific. For x86_64 cross-compile builds (macos-14
    # host targeting x86_64) we leave build isolation on: numpy 1.x has
    # no cp313 wheels, and pip would try to compile it from source in a
    # cross-environment where meson refuses ("Can not run test
    # applications in this cross environment"). numpy 2.x ships cp313
    # x86_64 wheels and the Accelerate ILP64 bug doesn't affect x86_64.
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
            "setuptools>=77.0.3", "wheel", "numpy<2",
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
