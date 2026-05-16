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
# See planning/bundle/option-1-implementation-plan.md Tasks 1.3–1.4.
set -euo pipefail

BUILD_PREFIX="${BUILD_PREFIX:-/usr/local}"

echo "=== setup-gdal-from-pixi.sh ==="
echo "BUILD_PREFIX=${BUILD_PREFIX}"

# ---------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------
if [[ "$(uname -s)" == "Darwin" ]]; then
    NEWEST_XCODE="$(ls -d /Applications/Xcode*.app 2>/dev/null | sort -r | head -1)"
    if [ -n "${NEWEST_XCODE:-}" ] && [ -d "${NEWEST_XCODE}/Contents/Developer" ]; then
        echo "--- Switching active Xcode to ${NEWEST_XCODE} ---"
        sudo xcode-select -s "${NEWEST_XCODE}/Contents/Developer"
        xcode-select -p

        TOOLCHAIN_BIN="${NEWEST_XCODE}/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin"
        DEVELOPER_USR_BIN="${NEWEST_XCODE}/Contents/Developer/usr/bin"
        CLT_BIN="/Library/Developer/CommandLineTools/usr/bin"
        echo "--- Installing direct toolchain symlinks into /usr/local/bin ---"
        sudo mkdir -p /usr/local/bin
        for tool in clang clang++ otool install_name_tool codesign lipo strip ld ar ranlib nm libtool dsymutil; do
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

# ---------------------------------------------------------------------------
# 1. Install pixi (static binary, ~50 MB, ~5 seconds)
# ---------------------------------------------------------------------------
if ! command -v pixi >/dev/null 2>&1; then
    echo "--- Installing pixi ---"
    export PIXI_HOME="${BUILD_PREFIX}"
    export PIXI_NO_PATH_UPDATE=1
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="${BUILD_PREFIX}/bin:${PATH}"
fi
pixi --version

# ---------------------------------------------------------------------------
# 2. Install the wheel-build environment.
#
# Native (host arch == target arch): use pixi with --frozen for
# reproducibility — exactly what's in pixi.lock, no re-solving.
#
# Cross-compile (e.g. macos-14 / arm64 host building an x86_64 wheel
# because GitHub's macos-13 queue is unusable): pixi can't install for
# a non-host platform, so fall back to micromamba, which natively
# supports `--platform osx-64`. Packages and versions track
# [tool.pixi.feature.wheel-build.dependencies] in pyproject.toml.
# ---------------------------------------------------------------------------
PIXI_ENV="$(pwd)/.pixi/envs/wheel-build"
TARGET_ARCH="${CIBW_ARCHS:-${CIBW_ARCHS_MACOS:-}}"
CROSS_COMPILE=0
if [[ "$(uname -s)" == "Darwin" ]] && [[ "${TARGET_ARCH}" == "x86_64" ]]; then
    CROSS_COMPILE=1
fi

if [ "${CROSS_COMPILE}" = "1" ]; then
    echo "--- Cross-compile: installing osx-64 env via micromamba ---"
    MAMBA_BIN="${BUILD_PREFIX}/bin/micromamba"
    curl -fsSL https://micro.mamba.pm/api/micromamba/osx-arm64/latest \
        | tar -xj -C "${BUILD_PREFIX}" --strip-components=1 bin/micromamba
    chmod +x "${MAMBA_BIN}"
    "${MAMBA_BIN}" --version
    export MAMBA_ROOT_PREFIX="${HOME}/micromamba-root"
    mkdir -p "${MAMBA_ROOT_PREFIX}"
    rm -rf "${PIXI_ENV}"
    mkdir -p "${PIXI_ENV}"
    "${MAMBA_BIN}" create -p "${PIXI_ENV}" \
        --platform osx-64 \
        -c conda-forge \
        -y \
        "gdal=3.12.*" \
        "libgdal-netcdf=3.12.*" \
        "libgdal-hdf4=3.12.*" \
        "swig>=4.3,<5"
else
    echo "--- Resolving wheel-build environment (host platform) ---"
    pixi install -e wheel-build --frozen
fi

if [ ! -d "${PIXI_ENV}" ]; then
    echo "ERROR: ${PIXI_ENV} does not exist after env install" >&2
    exit 1
fi
echo "wheel-build env: ${PIXI_ENV}"

# ---------------------------------------------------------------------------
# 3. Extract native artifacts into ${BUILD_PREFIX}
# ---------------------------------------------------------------------------
echo "--- Extracting native artifacts into ${BUILD_PREFIX} ---"
mkdir -p "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64" \
         "${BUILD_PREFIX}/include" "${BUILD_PREFIX}/share" \
         "${BUILD_PREFIX}/bin"

# Shared libraries — preserve symlinks with -a.
# Linux uses .so, macOS uses .dylib; glob both so the script is cross-platform.
cp -a "${PIXI_ENV}/lib/"*.so* "${BUILD_PREFIX}/lib/" 2>/dev/null || true
cp -a "${PIXI_ENV}/lib/"*.dylib* "${BUILD_PREFIX}/lib/" 2>/dev/null || true
if [ -d "${PIXI_ENV}/lib64" ]; then
    cp -a "${PIXI_ENV}/lib64/"*.so* "${BUILD_PREFIX}/lib64/" 2>/dev/null || true
    cp -a "${PIXI_ENV}/lib64/"*.dylib* "${BUILD_PREFIX}/lib64/" 2>/dev/null || true
fi

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
    cp "${PIXI_ENV}/lib/pkgconfig/"*.pc "${BUILD_PREFIX}/lib/pkgconfig/" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 4. Strip debug symbols to reduce wheel size
#
# `strip --strip-unneeded` is Linux/GNU. macOS strip uses `-S` for the
# equivalent "strip debug symbols only, keep the dynamic symbol table".
# ---------------------------------------------------------------------------
echo "--- Stripping shared libraries ---"
if [[ "$(uname -s)" == "Darwin" ]]; then
    find "${BUILD_PREFIX}/lib" -name '*.dylib*' -type f \
        -exec strip -S {} + 2>/dev/null || true
else
    find "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64" -name '*.so*' -type f \
        -exec strip --strip-unneeded {} + 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 5. Diagnostic output
# ---------------------------------------------------------------------------
echo "=== setup-gdal-from-pixi.sh complete ==="
echo "GDAL version: $("${BUILD_PREFIX}/bin/gdal-config" --version)"
echo "libgdal: $(ls "${BUILD_PREFIX}/lib/libgdal.so"* 2>/dev/null | head -1)"
echo "libproj: $(ls "${BUILD_PREFIX}/lib/libproj.so"* 2>/dev/null | head -1)"
echo "libgeos: $(ls "${BUILD_PREFIX}/lib/libgeos.so"* 2>/dev/null | head -1)"
echo "Total .so files: $(find "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64" -name '*.so*' -type f 2>/dev/null | wc -l)"
echo "Total size: $(du -sh "${BUILD_PREFIX}/lib" 2>/dev/null | cut -f1)"
