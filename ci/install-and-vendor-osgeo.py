"""Install GDAL Python bindings and vendor them into src/pyramids/_vendor/.

Runs once per target Python version in `CIBW_BEFORE_BUILD`:

1. `pip install GDAL==<version>` against the pixi-extracted libgdal
   sitting under `$BUILD_PREFIX/lib` (populated by
   `ci/setup-gdal-from-pixi.{sh,ps1}` in `CIBW_BEFORE_ALL`). The
   concrete version is read from `${BUILD_PREFIX}/GDAL_VERSION` which
   those scripts write at the resolution of pixi.lock / micromamba.
2. Copy the freshly-built `osgeo` package from the target Python's
   `site-packages/` into `src/pyramids/_vendor/osgeo/`.
3. Copy `$BUILD_PREFIX/share/gdal` and `$BUILD_PREFIX/share/proj`
   into `src/pyramids/_data/` so setuptools includes them as
   package-data in the wheel.

Activation is gated by `PACKAGE_DATA=1` to avoid accidentally running
during local editable installs.
"""
from __future__ import annotations

import json
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

    `ci/setup-gdal-from-pixi.{sh,ps1}` writes the version pixi /
    micromamba actually installed into `${BUILD_PREFIX}/GDAL_VERSION`
    so this script can `pip install GDAL==X.Y.Z` against the exact
    libgdal binary we bundle. Falls back to the `GDAL_VERSION` env
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

    Conda Windows packages nest under `<prefix>/Library/` (Anaconda's
    Windows convention). Linux and macOS use the standard Unix layout
    directly under `<prefix>/`.
    """
    if sys.platform == "win32" or os.name == "nt":
        win_lib_root = prefix / "Library"
        return win_lib_root / "bin", win_lib_root / "share", win_lib_root / "lib"
    return prefix / "bin", prefix / "share", prefix / "lib"


def _ca_bundle_src(prefix: Path) -> Path:
    """Return the conda-forge curl CA bundle path inside the build prefix.

    conda-forge's `ca-certificates` ships `ssl/cacert.pem` under the env
    prefix (nested under `Library/` on Windows, as with share/ and lib/).
    conda-forge's libcurl is compiled with this as its baked-in default,
    so the vendored GDAL would otherwise look for it at the build-time
    prefix (`.pixi/envs/wheel-build/ssl/cacert.pem`) — a path absent in
    the consuming env. We copy it into the wheel and re-point GDAL/curl
    at the bundled copy from the runtime bootstrap.
    """
    if sys.platform == "win32" or os.name == "nt":
        return prefix / "Library" / "ssl" / "cacert.pem"
    return prefix / "ssl" / "cacert.pem"


def install_gdal_python_bindings() -> None:
    """pip install `GDAL==<resolved version>` linking against $BUILD_PREFIX.

    GDAL 3.12.x's setup.py uses two discovery mechanisms:

    * Unix (Linux/macOS, `unix` compiler): runs `gdal-config` via PATH
      lookup. So we prepend `${BUILD_PREFIX}/bin` to PATH.
    * Windows (`msvc` compiler): skips `gdal-config` and reads the
      MSVC `INCLUDE` / `LIB` env vars. Conda's Windows packages nest
      under `${BUILD_PREFIX}/Library/` so we point INCLUDE/LIB there.

    On Linux and macOS we cap parallel build jobs (MAKEFLAGS=-j2) to
    keep peak memory below the GitHub runner's ~16 GB ceiling on
    ubuntu-latest / ubuntu-24.04-arm and the macos-14 runner's ~7 GB
    ceiling. On macOS arm64 we additionally pre-install a
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
    # jobs to keep peak RAM under the runner ceiling (~7 GB on macos-14).
    #
    # Linux runners (ubuntu-latest, ubuntu-24.04-arm) have the same
    # 4-core / 16 GB shape and the GDAL build's peak RAM can reach 6-8
    # GB with `-j$(nproc)`. Apply the same -j2 cap so a future GDAL
    # release with higher parallel-build memory pressure doesn't OOM.
    # Windows uses MSVC and doesn't honor MAKEFLAGS; leave it alone.
    if is_macos or sys.platform == "linux":
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
    # Only macOS uses target_arch here: the arm64 carve-out below
    # pre-installs an Accelerate-friendly numpy. CIBW_ARCHS_MACOS is
    # the cibuildwheel-emitted env var; CIBW_ARCHS is the user-facing
    # one set via workflow inputs. Linux / Windows builds simply leave
    # this empty and skip the arm64 branch.
    macos_target_arch = (
        os.environ.get("CIBW_ARCHS_MACOS")
        or os.environ.get("CIBW_ARCHS")
        or ""
    ) if is_macos else ""
    print(
        f"[install-and-vendor-osgeo] macos_target_arch: {macos_target_arch!r}",
        flush=True,
    )
    if is_macos and macos_target_arch == "arm64":
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "setuptools>=77.0.3", "wheel"],
            env=env, check=True,
        )
        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        download_dir = Path(tempfile.mkdtemp(prefix="numpy-dl-"))
        # numpy>=2.1 is tighter than [project.dependencies]'s numpy>=2.0
        # on purpose: 2.0 had bugs in the macosx_11_0_arm64 wheel that
        # interact badly with GDAL's setup.py. The build-pin can be
        # tighter than the runtime-pin since the build numpy never
        # ships in the wheel. Bump the cap when numpy 3.x lands.
        subprocess.run(
            [sys.executable, "-m", "pip", "download",
             "--no-deps", "--only-binary=:all:",
             "--platform", "macosx_11_0_arm64",
             "--python-version", f"{sys.version_info.major}.{sys.version_info.minor}",
             "--implementation", "cp",
             "--abi", py_tag,
             "-d", str(download_dir),
             "numpy>=2.1,<3"],
            env=env, check=True,
        )
        numpy_whl = next(download_dir.glob("numpy-*.whl"))
        print(f"[install-and-vendor-osgeo] using {numpy_whl.name}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", str(numpy_whl)],
            env=env, check=True,
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
    subprocess.run(cmd, env=env, check=True)


# Python bytecode is regenerated on first import; shipping it only bloats
# the wheel (~2 MB of `_vendor/**/__pycache__`). Skip it on every copy (T1.1).
_IGNORE_BYTECODE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def _copy_tree_replacing(src: Path, dst: Path) -> None:
    """Copy a directory, removing the destination first if it exists.

    Excludes Python bytecode (`__pycache__`/`*.pyc`) — it is regenerated on
    first import and only inflates the wheel (T1.1).
    """
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[install-and-vendor-osgeo] copy {src} -> {dst}", flush=True)
    shutil.copytree(src, dst, ignore=_IGNORE_BYTECODE)


def _strip_vendored_extensions(osgeo_dir: Path) -> None:
    """Strip debug symbols from the vendored SWIG extension modules (T1.2).

    The `osgeo/_gdal._osr._ogr…` extensions arrive un-stripped from the pip
    GDAL build; `strip --strip-unneeded` shaves several MB. Verified safe:
    local (non-network) NetCDF multidim `ReadAsArray` round-trips still pass
    in the wheel-test matrix with the extensions stripped. No-op on Windows
    (`.pyd` linkage is delvewheel's job and `strip` is GNU/macOS only).
    """
    if sys.platform.startswith("win") or os.name == "nt":
        return
    strip_args = ["-x"] if sys.platform == "darwin" else ["--strip-unneeded"]
    for so in sorted(osgeo_dir.glob("*.so")):
        try:
            subprocess.run(["strip", *strip_args, str(so)], check=True)
            print(f"[install-and-vendor-osgeo] stripped {so.name}", flush=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[install-and-vendor-osgeo] strip skipped {so.name}: {exc}", flush=True)


def _prune_unused_bindings(osgeo_dir: Path) -> None:
    """Drop the GNM (Geographic Network Model) bindings — unused by pyramids (T1.3).

    Removes `_gnm.<ext>` + `gnm.py` (+ `gnmconst.py`). pyramids has no GNM code
    path and `osgeo/__init__.py` does not auto-import it, so the ~1.2 MB binding
    is pure wheel bloat. Other vendored modules (`osgeo_utils`, used by COG
    validation + `gdal2xyz`) are intentionally kept.
    """
    for pattern in ("_gnm*.so", "_gnm*.pyd", "gnm.py", "gnmconst.py"):
        for f in osgeo_dir.glob(pattern):
            f.unlink()
            print(f"[install-and-vendor-osgeo] pruned {f.name}", flush=True)


def _prune_hdf4_plugin(plugins_dir: Path) -> None:
    """Drop the HDF4 GDAL driver plugin — unused by pyramids (T1.4, issue #474).

    `gdal_HDF4.{so,dll,dylib}` is a dlopen'd driver plugin (not linked into
    libgdal). The wheel-repair tools (auditwheel / delocate / delvewheel) walk
    every shared object in the wheel and bundle its dependency closure, so
    shipping this plugin is what drags `libdf` + `libmfhdf` (~1.2 MB) into the
    wheel — libs nothing else links. pyramids reads no HDF4 rasters (no code
    path or test fixture uses the driver; audit in #474), so removing the
    plugin here keeps the two libs out of the repaired wheel. netCDF / HDF5 /
    GRIB plugins are deliberately kept.
    """
    for pattern in ("gdal_HDF4.so", "gdal_HDF4.dll", "gdal_HDF4.dylib", "gdal_HDF4Image.*"):
        for f in plugins_dir.glob(pattern):
            f.unlink()
            print(f"[install-and-vendor-osgeo] pruned HDF4 plugin {f.name}", flush=True)


# Driver-specific GDAL_DATA support files for formats pyramids does not target
# (niche OGR vector + a few niche raster drivers). Removing each only disables
# that one driver's auxiliary data — geometry/raster reads of common formats are
# unaffected, and NOTHING here is CRS data. This is a deliberate denylist (drop
# only known-niche files) rather than an allowlist, so every `.wkt` / datum /
# ellipsoid / schema / TileMatrixSet / EPSG table is kept untouched (T1.5, #474).
_GDAL_DATA_DROP = (
    "default.rsc",          # MapInfo symbology (TAB/MIF geometry reads don't need it)
    "nitf_spec.xml",        # NITF (defense imagery)
    "nitf_spec.xsd",
    "ruian_*.gfs",          # RUIAN — Czech cadastre (OGR)
    "s57*.csv",             # S-57 — ENC nautical charts (OGR)
    "jpfgdgml_*.gfs",       # Japanese FGD GML (OGR)
    "inspire_cp_*.gfs",     # INSPIRE cadastral (OGR)
    "gmlasconf.xsd",        # GMLAS — GML application schemas (OGR)
    "gmlasconf.xml",
    "plscenesconf.json",    # Planet PLScenes
    "eedaconf.json",        # Earth Engine Data API
    "vdv452.xml",           # VDV-452 public-transport (OGR)
    "vdv452.xsd",
    "MM_m_idofic.csv",      # MiraMon (OGR)
    "pdfcomposition.xsd",   # PDF composition
    "seed_2d.dgn",          # DGN write seeds (Microstation)
    "seed_3d.dgn",
    "bag_template.xml",     # BAG bathymetry
    "pds4_template.xml",    # PDS4 planetary
    "vicar.json",           # VICAR planetary
    "template_tiles.mapml",  # MapML output template
    "leaflet_template.html",  # gdal2tiles leaflet template
)


def _trim_gdal_data(gdal_data_dir: Path) -> None:
    """Drop niche-driver GDAL_DATA support files pyramids never exercises (T1.5).

    See :data:`_GDAL_DATA_DROP`. Keeps every CRS / datum / ellipsoid / schema /
    TileMatrixSet file, so coordinate-system resolution is untouched; only the
    auxiliary data for formats pyramids does not target is removed (~1.4 MB).
    """
    removed = 0
    for pattern in _GDAL_DATA_DROP:
        for f in gdal_data_dir.glob(pattern):
            f.unlink()
            removed += 1
    print(f"[install-and-vendor-osgeo] trimmed {removed} niche GDAL_DATA files", flush=True)


_BOOTSTRAP_TEMPLATE_PATH = Path(__file__).resolve().parent / "_osgeo_bootstrap.py"


def _read_osgeo_bootstrap() -> str:
    """Load the Windows DLL bootstrap source from the sibling template file.

    Kept as a separate ``.py`` file (not an embedded triple-quoted string)
    so the bootstrap code is syntax-highlighted, statically analysable,
    and unit-testable. The file is never imported as a module — it's
    read at vendor time and spliced into the vendored
    ``osgeo/__init__.py``.
    """
    if not _BOOTSTRAP_TEMPLATE_PATH.is_file():
        raise RuntimeError(
            f"osgeo bootstrap template missing: {_BOOTSTRAP_TEMPLATE_PATH}"
        )
    return _BOOTSTRAP_TEMPLATE_PATH.read_text(encoding="utf-8")


def _patch_vendored_osgeo_init(init_path: Path) -> None:
    """Inject a Windows DLL bootstrap into the vendored osgeo/__init__.py.

    Without this, importing `osgeo` in a multiprocessing.spawn worker
    fails with `ImportError: DLL load failed while importing _gdal`
    because the parent's `os.add_dll_directory` call doesn't carry to
    spawn children, and the worker imports osgeo before pyramids.

    The bootstrap is spliced AFTER any leading comments, blank lines,
    and `from __future__` imports so the patch stays valid if
    upstream osgeo ever adds a future-import (PEP 236 requires
    __future__ imports to precede any other statement). The current
    conda-forge osgeo doesn't use __future__, but `gdal=3.12.*` is
    intentionally loose and a 3.12.x bump could add one without our
    pin moving.
    """
    original = init_path.read_text(encoding="utf-8")
    if "pyramids vendor patch" in original:
        return
    bootstrap = _read_osgeo_bootstrap()
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
        + bootstrap
        + "".join(lines[insert_at:])
    )
    init_path.write_text(patched, encoding="utf-8")
    print(
        f"[install-and-vendor-osgeo] patched {init_path} with Windows DLL bootstrap",
        flush=True,
    )


def _locate_site_packages_dir(name: str) -> Path | None:
    """Return the on-disk directory of an installed top-level package.

    We deliberately do NOT `import` the package: on Windows the GDAL
    SWIG extension loads `gdal.dll` and its transitive dep DLLs at
    import time, and a runtime symbol mismatch between Python 3.13's
    bundled vcruntime and conda-forge's bundled vcruntime triggers
    "DLL load failed: specified procedure could not be found" (cp311/
    cp312 don't hit this). The vendoring step only needs the on-disk
    location of the package — no Python code from it actually runs —
    so we look it up via the active environment's purelib + platlib
    paths. For venvs purelib == platlib (single site-packages dir),
    but packages with C extensions like osgeo land under platlib in
    custom layouts (PEP 668, Debian's split site-packages, …).
    """
    sysconfig_paths = sysconfig.get_paths()
    candidates: list[Path] = []
    for key in ("purelib", "platlib"):
        if key in sysconfig_paths:
            candidate = Path(sysconfig_paths[key]) / name
            if candidate.is_dir() and candidate not in candidates:
                candidates.append(candidate)
    return candidates[0] if candidates else None


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
    _prune_unused_bindings(vendor_dir / "osgeo")      # T1.3 — drop unused GNM bindings
    _strip_vendored_extensions(vendor_dir / "osgeo")  # T1.2 — strip SWIG .so debug symbols

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
    gdal_data_dst = src_pyramids / "_data" / "gdal_data"
    _copy_tree_replacing(gdal_data_src, gdal_data_dst)
    _trim_gdal_data(gdal_data_dst)  # T1.5 — drop niche-driver support files

    # 3. Vendor PROJ_DATA
    proj_data_src = share_dir / "proj"
    if not proj_data_src.is_dir():
        raise RuntimeError(f"PROJ_DATA not found at {proj_data_src}")
    _copy_tree_replacing(proj_data_src, src_pyramids / "_data" / "proj_data")

    # 4. Vendor GDAL plugins (NetCDF / HDF4 / HDF5 drivers). On
    # Windows, conda-forge ships these as separate .dll files under
    # `lib/gdalplugins/` (or `Library/lib/gdalplugins/` on the
    # nested layout) and libgdal loads them at runtime via
    # `GDAL_DRIVER_PATH`. On Linux/macOS the same drivers are linked
    # statically into libgdal, so `lib/gdalplugins/` doesn't exist
    # and the `is_dir()` guard turns this into a no-op. The runtime
    # bootstrap in `src/pyramids/__init__.py` mirrors this guard.
    plugins_src = lib_dir / "gdalplugins"
    if plugins_src.is_dir():
        plugins_dst = src_pyramids / "_data" / "gdalplugins"
        _copy_tree_replacing(plugins_src, plugins_dst)
        _prune_hdf4_plugin(plugins_dst)  # T1.4 — drop unused HDF4 driver (+ libdf/libmfhdf)

    # 5. Vendor the curl CA bundle. conda-forge's libcurl bakes its
    # default CA path to `<build-prefix>/ssl/cacert.pem`, which does not
    # exist in the consuming env — so every GDAL `/vsicurl` HTTPS read
    # fails to load trust anchors. Ship the bundle in the wheel; the
    # runtime bootstrap points GDAL/curl at this copy via GDAL_HTTP_CAINFO
    # / CURL_CA_BUNDLE / SSL_CERT_FILE. See issue #412.
    ca_bundle_src = _ca_bundle_src(prefix)
    if ca_bundle_src.is_file():
        ca_dst = src_pyramids / "_data" / "ssl" / "cacert.pem"
        ca_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ca_bundle_src, ca_dst)
    else:
        print(
            f"[install-and-vendor-osgeo] CA bundle not found at {ca_bundle_src}; "
            "vendored GDAL HTTPS reads may fail (#412)",
            flush=True,
        )

    # 6. Vendor third-party license texts. Each conda-forge package
    # ships its LICENSE under `info/licenses/` inside its extracted
    # package dir; the MIT / BSD / LGPL / Apache licenses all require
    # the copyright + permission notice to travel with the binary
    # wherever it's redistributed. Mirror them under
    # `pyramids/_licenses/<pkg>/` so the wheel physically ships each
    # license alongside the libgdal / libproj / libgeos / … binaries
    # it bundles.
    _vendor_license_texts(
        REPO_ROOT / ".pixi" / "envs" / "wheel-build",
        src_pyramids / "_licenses",
    )

    # 7. Defense-in-depth `.gitignore` markers. The repo .gitignore
    # already excludes `src/pyramids/_vendor/` and `src/pyramids/_data/`,
    # but a dev who runs `cibuildwheel` locally and then
    # `git add -f` (force-add bypasses .gitignore) could still
    # accidentally commit the vendored payload. A directory-local
    # .gitignore that says `*` makes git refuse the add even with -f
    # unless the user passes -f twice.
    for marker_dir in (vendor_dir, src_pyramids / "_data", src_pyramids / "_licenses"):
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / ".gitignore").write_text("*\n", encoding="utf-8")


def _vendor_license_texts(pixi_env: Path, dst: Path) -> None:
    """Mirror each conda-forge package's `info/licenses/` under `dst/<pkg>/`.

    Walks every `${pixi_env}/conda-meta/*.json` file, extracts the
    package's name + extracted-package directory, and copies the contents
    of `<extracted>/info/licenses/` into `dst/<pkg-name>/` so the
    wheel can ship the LICENSE text alongside the binary it applies to.
    Skips packages with no `info/licenses/` directory (typically pure
    python helpers that don't ship third-party native code).
    """
    if not pixi_env.is_dir():
        print(
            f"[install-and-vendor-osgeo] pixi env {pixi_env} missing; "
            "skipping license vendoring",
            flush=True,
        )
        return
    conda_meta = pixi_env / "conda-meta"
    if not conda_meta.is_dir():
        print(
            f"[install-and-vendor-osgeo] {conda_meta} missing; "
            "skipping license vendoring",
            flush=True,
        )
        return

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    count = 0
    skipped = 0
    for meta_file in sorted(conda_meta.glob("*.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[install-and-vendor-osgeo] could not parse {meta_file.name}: "
                f"{exc!r}",
                flush=True,
            )
            continue
        pkg_name = meta.get("name")
        extracted = meta.get("extracted_package_dir")
        if not pkg_name or not extracted:
            continue
        licenses_src = Path(extracted) / "info" / "licenses"
        if not licenses_src.is_dir():
            skipped += 1
            continue
        pkg_dst = dst / pkg_name
        pkg_dst.mkdir(parents=True, exist_ok=True)
        for src_file in licenses_src.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(licenses_src)
                tgt = pkg_dst / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, tgt)
        count += 1
    print(
        f"[install-and-vendor-osgeo] vendored licenses for {count} "
        f"conda-forge packages ({skipped} had no info/licenses) -> {dst}",
        flush=True,
    )


def main() -> None:
    if os.environ.get("PACKAGE_DATA") != "1":
        print("[install-and-vendor-osgeo] PACKAGE_DATA != 1; skipping.", flush=True)
        return
    install_gdal_python_bindings()
    vendor_osgeo_into_package()
    print("[install-and-vendor-osgeo] done.", flush=True)


if __name__ == "__main__":
    main()
