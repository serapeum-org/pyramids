#!/bin/bash
#
# Install GDAL + native dependencies via pixi (conda-forge), then extract
# the shared libraries, headers, and data files into ${BUILD_PREFIX} so
# downstream steps (install-and-vendor-osgeo.py, setuptools,
# auditwheel/delocate/delvewheel) can find them without knowing about
# pixi's internal layout.
#
# Runs once per cibuildwheel platform invocation (CIBW_BEFORE_ALL), i.e.
# shared across all Python versions in the matrix.
#
# See docs/how-to/wheel-build-flow.md for the end-to-end pipeline.
set -euo pipefail

BUILD_PREFIX="${BUILD_PREFIX:-/usr/local}"

echo "=== setup-gdal-from-pixi.sh ==="
echo "BUILD_PREFIX=${BUILD_PREFIX}"

# 0. (macOS) Bypass the /usr/bin xcrun shims.
#
# On the macos-14 runner, xcodebuild is reliably SIGKILLed regardless of
# which Xcode is selected (both 15.4 and 16.2 die identically). Every
# /usr/bin/{clang,otool,install_name_tool,codesign,lipo,strip,...} is a
# stub that calls `xcrun -find <tool>` which spawns xcodebuild — so each
# one fails:
#   - GDAL compile: "xcode-select: Failed to locate 'clang++'"
#   - delocate-wheel: "InstallNameError: Unexpected first line:
#     sh: ... Killed: 9 ... -find otool"
# Install symlinks pointing directly at the real binaries inside the
# Xcode toolchain (no /usr/bin indirection) into /usr/local/bin, which
# is on PATH ahead of /usr/bin on macOS. Every subsequent PATH lookup
# for those tools resolves to the real binary and never spawns
# xcodebuild.
#
# This block uses sudo to write into /usr/local/bin. Refuse to run
# outside CI (or unless the user explicitly opts in) — a developer
# running this locally on their own Mac would otherwise be prompted
# for a sudo password (and hang in non-tty contexts) plus pollute
# /usr/local/bin with clang wrappers pointing at their personal Xcode.
# GitHub Actions sets CI=true automatically, so the guard is invisible
# on runners. Set FORCE_LOCAL_SUDO=1 to bypass when you really mean it.
if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ "${CI:-}" != "true" ]] && [[ -z "${FORCE_LOCAL_SUDO:-}" ]]; then
        echo "ERROR: setup-gdal-from-pixi.sh's macOS path requires sudo and is" >&2
        echo "intended for CI use. Re-run with FORCE_LOCAL_SUDO=1 if you really" >&2
        echo "want to install Xcode symlinks into /usr/local/bin locally." >&2
        exit 1
    fi
    # Pick the newest /Applications/Xcode*.app by semver, not lexically.
    # `sort -r` would sort Xcode_15.10.app before Xcode_15.9.app — fine
    # today because Xcode minor versions stay single-digit (15.4, 16.2),
    # broken the moment 15.10+ ships. `sort -V` is version-aware.
    NEWEST_XCODE=""
    shopt -s nullglob
    _xcodes=(/Applications/Xcode*.app)
    shopt -u nullglob
    if (( ${#_xcodes[@]} )); then
        NEWEST_XCODE=$(printf '%s\n' "${_xcodes[@]}" | sort -V | tail -1)
    fi
    if [ -n "${NEWEST_XCODE:-}" ] && [ -d "${NEWEST_XCODE}/Contents/Developer" ]; then
        echo "--- Switching active Xcode to ${NEWEST_XCODE} ---"
        sudo xcode-select -s "${NEWEST_XCODE}/Contents/Developer"
        xcode-select -p

        DEVELOPER_DIR="${NEWEST_XCODE}/Contents/Developer"
        SDKROOT="${DEVELOPER_DIR}/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
        TOOLCHAIN_BIN="${DEVELOPER_DIR}/Toolchains/XcodeDefault.xctoolchain/usr/bin"
        DEVELOPER_USR_BIN="${DEVELOPER_DIR}/usr/bin"
        CLT_BIN="/Library/Developer/CommandLineTools/usr/bin"

        # clang / clang++: install wrapper scripts (not plain symlinks)
        # that export SDKROOT + DEVELOPER_DIR before exec'ing the real
        # toolchain binary. Plain symlinks invoke the toolchain clang
        # directly without xcrun's SDKROOT auto-detection, and clang
        # then can't find system headers ('stdlib.h' file not found).
        # Wrappers fix that without leaking env vars into the rest of
        # the build.
        echo "--- Installing clang/clang++ wrappers into /usr/local/bin ---"
        sudo mkdir -p /usr/local/bin
        for compiler in clang clang++; do
            sudo tee "/usr/local/bin/${compiler}" >/dev/null <<EOF
#!/bin/bash
export SDKROOT="\${SDKROOT:-${SDKROOT}}"
export DEVELOPER_DIR="\${DEVELOPER_DIR:-${DEVELOPER_DIR}}"
exec "${TOOLCHAIN_BIN}/${compiler}" "\$@"
EOF
            sudo chmod +x "/usr/local/bin/${compiler}"
            echo "  /usr/local/bin/${compiler} -> ${TOOLCHAIN_BIN}/${compiler} (with SDKROOT)"
        done

        # Everything else (otool, install_name_tool, codesign, ...) is
        # binary-only — no header lookup, so plain symlinks suffice.
        echo "--- Installing direct toolchain symlinks into /usr/local/bin ---"
        for tool in otool install_name_tool codesign lipo strip ld ar ranlib nm libtool dsymutil; do
            for src in "${TOOLCHAIN_BIN}/${tool}" "${DEVELOPER_USR_BIN}/${tool}" "${CLT_BIN}/${tool}"; do
                if [ -x "${src}" ]; then
                    sudo ln -sf "${src}" "/usr/local/bin/${tool}"
                    echo "  /usr/local/bin/${tool} -> ${src}"
                    break
                fi
            done
        done
    fi
fi

# 1. Install pixi (static binary, ~50 MB, ~5 seconds).
#
# Pin the version via .pixi-version at the repo root so CI, local
# developers, and the cibuildwheel container all agree. PIXI_VERSION
# is consumed by pixi.sh/install.sh — the install script verifies the
# downloaded binary's SHA256 against its built-in checksum table, so
# pinning makes the installer reproducible. Override locally with
# `PIXI_VERSION=X.Y.Z bash ci/setup-gdal-from-pixi.sh` if you need to
# test against a different pixi version.
if ! command -v pixi >/dev/null 2>&1; then
    PIXI_VERSION_FILE="$(cd "$(dirname "$0")/.." && pwd)/.pixi-version"
    if [[ -z "${PIXI_VERSION:-}" && -f "${PIXI_VERSION_FILE}" ]]; then
        PIXI_VERSION="$(tr -d '[:space:]' < "${PIXI_VERSION_FILE}")"
    fi
    PIXI_VERSION="${PIXI_VERSION:?PIXI_VERSION not set and .pixi-version missing}"
    echo "--- Installing pixi ${PIXI_VERSION} ---"
    export PIXI_HOME="${BUILD_PREFIX}"
    export PIXI_NO_PATH_UPDATE=1
    export PIXI_VERSION
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="${BUILD_PREFIX}/bin:${PATH}"
fi
pixi --version

# 2. Install the wheel-build environment.
#
# Native (host arch == target arch): use pixi with --frozen against
# pixi.lock for full reproducibility.
#
# Cross-compile (currently only macos-14 / arm64 host targeting
# osx-64): delegate to ci/setup-gdal-micromamba.sh. pixi has no
# --platform install flag; the dedicated micromamba script
# re-resolves the same dependency range pin against conda-forge.
PIXI_ENV="$(pwd)/.pixi/envs/wheel-build"
TARGET_ARCH="${CIBW_ARCHS:-${CIBW_ARCHS_MACOS:-}}"

if [[ "$(uname -s)" == "Darwin" ]] && [[ "${TARGET_ARCH}" == "x86_64" ]]; then
    export PIXI_ENV BUILD_PREFIX
    export TARGET_PLATFORM="osx-64"
    bash "$(dirname "$0")/setup-gdal-micromamba.sh"
else
    echo "--- Resolving wheel-build environment (host platform) ---"
    pixi install -e wheel-build --frozen
fi

if [ ! -d "${PIXI_ENV}" ]; then
    echo "ERROR: ${PIXI_ENV} does not exist after env install" >&2
    exit 1
fi
echo "wheel-build env: ${PIXI_ENV}"

# Resolve the concrete GDAL version from the env we just materialized
# and persist it for install-and-vendor-osgeo.py (which needs an exact
# version to pin `pip install GDAL==X.Y.Z`). Single source of truth =
# pixi.lock / micromamba solver output — no more hardcoded duplicates
# in pyproject.toml or build-wheels.yml.
GDAL_CONFIG="${PIXI_ENV}/bin/gdal-config"
if [ ! -x "${GDAL_CONFIG}" ]; then
    echo "ERROR: ${GDAL_CONFIG} missing or not executable" >&2
    exit 1
fi
GDAL_VERSION="$("${GDAL_CONFIG}" --version)"
mkdir -p "${BUILD_PREFIX}"
printf "%s" "${GDAL_VERSION}" > "${BUILD_PREFIX}/GDAL_VERSION"
echo "resolved GDAL_VERSION=${GDAL_VERSION}"

# 3. Extract native artifacts into ${BUILD_PREFIX}.
echo "--- Extracting native artifacts into ${BUILD_PREFIX} ---"
mkdir -p "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64" \
         "${BUILD_PREFIX}/include" "${BUILD_PREFIX}/share" \
         "${BUILD_PREFIX}/bin"

# Shared libraries — preserve symlinks with -a.
# Linux uses .so, macOS uses .dylib; glob both so the script is cross-platform.
# nullglob lets a non-matching glob expand to nothing (instead of being
# left as a literal string), so the conditional `cp` only runs when
# there's something to copy. Previous form was
# `cp -a glob 2>/dev/null || true` which also hid real failures
# (permission denied, disk full, etc.) — the explicit length-check
# preserves those failure modes.
shopt -s nullglob
_so_files=( "${PIXI_ENV}/lib/"*.so* )
_dylib_files=( "${PIXI_ENV}/lib/"*.dylib* )
(( ${#_so_files[@]} )) && cp -a "${_so_files[@]}" "${BUILD_PREFIX}/lib/"
(( ${#_dylib_files[@]} )) && cp -a "${_dylib_files[@]}" "${BUILD_PREFIX}/lib/"
if [ -d "${PIXI_ENV}/lib64" ]; then
    _so64_files=( "${PIXI_ENV}/lib64/"*.so* )
    _dylib64_files=( "${PIXI_ENV}/lib64/"*.dylib* )
    (( ${#_so64_files[@]} )) && cp -a "${_so64_files[@]}" "${BUILD_PREFIX}/lib64/"
    (( ${#_dylib64_files[@]} )) && cp -a "${_dylib64_files[@]}" "${BUILD_PREFIX}/lib64/"
fi
shopt -u nullglob

# Headers
cp -a "${PIXI_ENV}/include/." "${BUILD_PREFIX}/include/"

# GDAL_DATA + PROJ_DATA — required at runtime
cp -a "${PIXI_ENV}/share/gdal" "${BUILD_PREFIX}/share/"
cp -a "${PIXI_ENV}/share/proj" "${BUILD_PREFIX}/share/"

# GDAL plugins (libgdal-netcdf / libgdal-hdf4) live in a separate
# subdirectory and are loaded at runtime via GDAL_DRIVER_PATH. These
# MUST be bundled or NetCDF/HDF4/HDF5 drivers will be unavailable.
if [ -d "${PIXI_ENV}/lib/gdalplugins" ]; then
    mkdir -p "${BUILD_PREFIX}/lib/gdalplugins"
    cp -a "${PIXI_ENV}/lib/gdalplugins/." "${BUILD_PREFIX}/lib/gdalplugins/"
fi

# Build tooling needed downstream
for tool in gdal-config swig ogrinfo gdalinfo; do
    src="${PIXI_ENV}/bin/${tool}"
    if [ -f "${src}" ]; then
        cp "${src}" "${BUILD_PREFIX}/bin/"
    fi
done

# pkg-config files (some build tools consult pkg-config)
if [ -d "${PIXI_ENV}/lib/pkgconfig" ]; then
    mkdir -p "${BUILD_PREFIX}/lib/pkgconfig"
    shopt -s nullglob
    _pc_files=( "${PIXI_ENV}/lib/pkgconfig/"*.pc )
    (( ${#_pc_files[@]} )) && cp "${_pc_files[@]}" "${BUILD_PREFIX}/lib/pkgconfig/"
    shopt -u nullglob
fi

# 4. Strip debug symbols to reduce wheel size.
#
# `strip --strip-unneeded` is Linux/GNU. macOS strip uses `-S` for the
# equivalent "strip debug symbols only, keep the dynamic symbol table".
echo "--- Stripping shared libraries ---"
if [[ "$(uname -s)" == "Darwin" ]]; then
    find "${BUILD_PREFIX}/lib" -name '*.dylib*' -type f \
        -exec strip -S {} + 2>/dev/null || true
else
    find "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64" -name '*.so*' -type f \
        -exec strip --strip-unneeded {} + 2>/dev/null || true
fi

# 5. Diagnostic output.
echo "=== setup-gdal-from-pixi.sh complete ==="
echo "GDAL version: $("${BUILD_PREFIX}/bin/gdal-config" --version)"
echo "libgdal: $(ls "${BUILD_PREFIX}/lib/libgdal.so"* 2>/dev/null | head -1)"
echo "libproj: $(ls "${BUILD_PREFIX}/lib/libproj.so"* 2>/dev/null | head -1)"
echo "libgeos: $(ls "${BUILD_PREFIX}/lib/libgeos.so"* 2>/dev/null | head -1)"
echo "Total .so files: $(find "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64" -name '*.so*' -type f 2>/dev/null | wc -l)"
echo "Total size: $(du -sh "${BUILD_PREFIX}/lib" 2>/dev/null | cut -f1)"
