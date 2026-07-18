#!/bin/bash
#
# Install the wheel-build environment via micromamba for a cross-platform
# target. Used by ci/setup-gdal-from-pixi.sh when cibuildwheel builds for
# a target arch different from the host (currently: macos-14 / arm64 host
# producing an osx-64 wheel because GitHub's macos-13 Intel runner queue
# is unusable in practice).
#
# Why not pixi: pixi installs envs for the host arch only — there's no
# --platform flag on `pixi install`. micromamba ships a single static
# binary that supports `--platform osx-64` natively.
#
# Why we re-resolve instead of using pixi.lock: pixi.lock has osx-64
# entries (we added the platform), but pixi can't materialize them on
# an osx-arm64 host. micromamba re-solves from conda-forge against the
# same range pin declared in
# [tool.pixi.feature.wheel-build.dependencies]. The trade-off is we
# lose lock-file reproducibility on this one branch — acceptable scope
# until GitHub's Intel runners come back.
#
# Inputs (must be exported by caller):
#   BUILD_PREFIX     destination for the micromamba binary
#   PIXI_ENV         path where the env will be created (kept as
#                    .pixi/envs/wheel-build/ so downstream extraction
#                    logic is identical to the pixi branch)
#   TARGET_PLATFORM  conda platform tag for the wheel (currently only
#                    "osx-64" is reached — the cross-compile target
#                    for macos-14 / arm64 host)
set -euo pipefail

: "${BUILD_PREFIX:?BUILD_PREFIX must be set}"
: "${PIXI_ENV:?PIXI_ENV must be set}"
: "${TARGET_PLATFORM:?TARGET_PLATFORM must be set}"

echo "=== setup-gdal-micromamba.sh ==="
echo "BUILD_PREFIX=${BUILD_PREFIX}"
echo "PIXI_ENV=${PIXI_ENV}"
echo "TARGET_PLATFORM=${TARGET_PLATFORM}"

# Pick the micromamba binary for the host arch (separate from the
# TARGET_PLATFORM we're installing packages for).
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64)      HOST_PLATFORM="osx-arm64" ;;
    Darwin/x86_64)     HOST_PLATFORM="osx-64" ;;
    Linux/x86_64)      HOST_PLATFORM="linux-64" ;;
    Linux/aarch64)     HOST_PLATFORM="linux-aarch64" ;;
    *) echo "ERROR: unsupported host $(uname -s)/$(uname -m)" >&2; exit 1 ;;
esac

# Pin micromamba version via the MICROMAMBA_VERSION env var (set in
# build-wheels.yml's env block; this script only runs on the macOS
# cross-compile path, where cibuildwheel runs before-all on the host
# and inherits it). The previous `…/latest` endpoint was a rolling
# target — a new micromamba release with a regressed `--platform osx-64`
# would have silently broken CI.
# TODO: pinning the version doesn't fully close the supply-chain gap;
# micro.mamba.pm + the S3 backing store could still serve a tampered
# tarball. A follow-up should add SHA256 verification per platform
# (4 SHAs: linux-64, linux-aarch64, osx-arm64, osx-64) committed to a
# `.micromamba-sha256.txt` manifest.
: "${MICROMAMBA_VERSION:?MICROMAMBA_VERSION not set}"

MAMBA_BIN="${BUILD_PREFIX}/bin/micromamba"
mkdir -p "${BUILD_PREFIX}/bin"
MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/${HOST_PLATFORM}/${MICROMAMBA_VERSION}"
echo "--- Installing micromamba ${MICROMAMBA_VERSION} for ${HOST_PLATFORM} ---"
echo "    from ${MICROMAMBA_URL}"
curl -fsSL "${MICROMAMBA_URL}" \
    | tar -xj -C "${BUILD_PREFIX}" bin/micromamba
chmod +x "${MAMBA_BIN}"
"${MAMBA_BIN}" --version

# MAMBA_ROOT_PREFIX is where micromamba stashes its package cache.
# Use a BUILD_PREFIX-local dir rather than $HOME/micromamba-root so a
# developer running this script locally doesn't end up with a stray
# ~/micromamba-root. The CI runner is ephemeral so location doesn't
# matter there.
export MAMBA_ROOT_PREFIX="${BUILD_PREFIX}/micromamba-root"
mkdir -p "${MAMBA_ROOT_PREFIX}"
# Remove ${PIXI_ENV} cleanly — micromamba 2.x's `create -p` creates
# the prefix dir itself and refuses if the path already exists as a
# non-conda directory ("Non-conda folder exists at prefix"). Do NOT
# pre-mkdir.
rm -rf "${PIXI_ENV}"

# All four native build/test pins live once in pyproject.toml — the three gdal libs
# in [tool.pixi.feature.gdal.dependencies], build-only swig in
# [tool.pixi.feature.wheel-build.dependencies]. ci/gdal-pin.py is the single reader of
# those tables; calling it here keeps this cross-compile branch from re-encoding (or
# drifting from) the pins. One subprocess emits all four specs, newline-separated,
# mapped onto the bash vars via `read`. micromamba accepts a conda match-spec
# concatenated as ``<name><spec>`` (e.g. ``gdal>=3.13,<3.14``).
GDAL_PIN="$(cd "$(dirname "$0")" && pwd)/gdal-pin.py"
if [[ ! -f "${GDAL_PIN}" ]]; then
    echo "ERROR: gdal-pin.py not found at ${GDAL_PIN}" >&2
    exit 1
fi

{ read -r GDAL_SPEC; read -r LIBGDAL_NETCDF_SPEC; \
  read -r LIBGDAL_HDF4_SPEC; read -r LIBGDAL_GRIB_SPEC; \
  read -r LIBGDAL_JP2_SPEC; read -r SWIG_SPEC; } \
  < <(python3 "${GDAL_PIN}" gdal libgdal-netcdf libgdal-hdf4 libgdal-grib libgdal-jp2openjpeg swig)

echo "--- Wheel-build pins (from pyproject.toml) ---"
echo "  gdal${GDAL_SPEC}"
echo "  libgdal-netcdf${LIBGDAL_NETCDF_SPEC}"
echo "  libgdal-hdf4${LIBGDAL_HDF4_SPEC}"
echo "  libgdal-grib${LIBGDAL_GRIB_SPEC}"
echo "  libgdal-jp2openjpeg${LIBGDAL_JP2_SPEC}"
echo "  swig${SWIG_SPEC}"

echo "--- Creating ${TARGET_PLATFORM} env at ${PIXI_ENV} ---"
"${MAMBA_BIN}" create -p "${PIXI_ENV}" \
    --platform "${TARGET_PLATFORM}" \
    -c conda-forge \
    -y \
    "gdal${GDAL_SPEC}" \
    "libgdal-netcdf${LIBGDAL_NETCDF_SPEC}" \
    "libgdal-hdf4${LIBGDAL_HDF4_SPEC}" \
    "libgdal-grib${LIBGDAL_GRIB_SPEC}" \
    "libgdal-jp2openjpeg${LIBGDAL_JP2_SPEC}" \
    "swig${SWIG_SPEC}"

echo "=== setup-gdal-micromamba.sh complete ==="
