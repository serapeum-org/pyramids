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
#   TARGET_PLATFORM  conda platform tag for the wheel (e.g. "osx-64",
#                    "osx-arm64")
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

MAMBA_BIN="${BUILD_PREFIX}/bin/micromamba"
mkdir -p "${BUILD_PREFIX}/bin"
echo "--- Installing micromamba for ${HOST_PLATFORM} ---"
curl -fsSL "https://micro.mamba.pm/api/micromamba/${HOST_PLATFORM}/latest" \
    | tar -xj -C "${BUILD_PREFIX}" bin/micromamba
chmod +x "${MAMBA_BIN}"
"${MAMBA_BIN}" --version

export MAMBA_ROOT_PREFIX="${HOME}/micromamba-root"
mkdir -p "${MAMBA_ROOT_PREFIX}"
rm -rf "${PIXI_ENV}"
mkdir -p "${PIXI_ENV}"

# Package set is the single source of truth in
# [tool.pixi.feature.wheel-build.dependencies] in pyproject.toml; read
# the four pins at runtime so a tightening of the pyproject range can
# never drift away from this cross-compile branch unnoticed. micromamba
# accepts a conda match-spec when concatenated as ``<name><spec>``
# (e.g. ``gdal>=3.12,<3.13``) — the same form pyproject uses.
PYPROJECT="$(cd "$(dirname "$0")/.." && pwd)/pyproject.toml"
if [[ ! -f "${PYPROJECT}" ]]; then
    echo "ERROR: pyproject.toml not found at ${PYPROJECT}" >&2
    exit 1
fi

read_pin() {
    python3 - "${PYPROJECT}" "$1" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    data = tomllib.load(f)
print(data["tool"]["pixi"]["feature"]["wheel-build"]["dependencies"][sys.argv[2]])
PY
}

GDAL_SPEC=$(read_pin gdal)
LIBGDAL_NETCDF_SPEC=$(read_pin libgdal-netcdf)
LIBGDAL_HDF4_SPEC=$(read_pin libgdal-hdf4)
SWIG_SPEC=$(read_pin swig)

echo "--- Wheel-build pins (from pyproject.toml) ---"
echo "  gdal${GDAL_SPEC}"
echo "  libgdal-netcdf${LIBGDAL_NETCDF_SPEC}"
echo "  libgdal-hdf4${LIBGDAL_HDF4_SPEC}"
echo "  swig${SWIG_SPEC}"

echo "--- Creating ${TARGET_PLATFORM} env at ${PIXI_ENV} ---"
"${MAMBA_BIN}" create -p "${PIXI_ENV}" \
    --platform "${TARGET_PLATFORM}" \
    -c conda-forge \
    -y \
    "gdal${GDAL_SPEC}" \
    "libgdal-netcdf${LIBGDAL_NETCDF_SPEC}" \
    "libgdal-hdf4${LIBGDAL_HDF4_SPEC}" \
    "swig${SWIG_SPEC}"

echo "=== setup-gdal-micromamba.sh complete ==="
