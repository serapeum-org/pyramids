#!/bin/bash
#
# Phase-1 spike (#332): build the curated GDAL stack FROM SOURCE inside the
# manylinux_2_28 container, replacing ci/setup-gdal-from-pixi.sh as
# CIBW_BEFORE_ALL. Everything compiles with the image's own toolchain, so no
# GLIBCXX symbol exceeds the 2.28 baseline and auditwheel can tag
# manylinux_2_28 (vs the conda-extract model's forced 2_39).
#
# Contract with the rest of the pipeline (mirrors setup-gdal-from-pixi.sh):
#   - installs the stack into ${BUILD_PREFIX} (default /usr/local): libs,
#     gdal-config, share/gdal, share/proj
#   - writes ${BUILD_PREFIX}/GDAL_VERSION for ci/install-and-vendor-osgeo.py
#   - stages the curl CA bundle at ${BUILD_PREFIX}/ssl/cacert.pem (#412)
# Known spike gaps (Phase 2 work, not blockers for the feasibility gate):
#   - third-party license texts are NOT vendored (no conda-meta to mirror);
#     _vendor_license_texts warns and continues
set -euo pipefail

export BUILD_PREFIX="${BUILD_PREFIX:-/usr/local}"
export GDAL_VERSION="${GDAL_VERSION:-3.13.1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== from-source GDAL stack: GDAL ${GDAL_VERSION} -> ${BUILD_PREFIX} ==="

# Toolchain prerequisites. manylinux_2_28 (AlmaLinux 8) ships gcc-toolset +
# make; cmake/wget vary by image revision, so ensure them explicitly.
if ! command -v wget >/dev/null 2>&1; then
    (dnf install -y wget || yum install -y wget) >/dev/null
fi
# OpenSSL's Configure requires full perl (IPC::Cmd, FindBin, Pod::*); the
# AlmaLinux base image ships a minimal perl that aborts at BEGIN.
if ! perl -MIPC::Cmd -e1 >/dev/null 2>&1; then
    (dnf install -y perl-core || yum install -y perl-core) >/dev/null
fi
if ! command -v cmake >/dev/null 2>&1; then
    pipx install cmake >/dev/null 2>&1 || /opt/python/cp312-cp312/bin/pip install --quiet cmake
    command -v cmake >/dev/null 2>&1 || export PATH="/opt/python/cp312-cp312/bin:${PATH}"
fi
echo "cmake: $(command -v cmake) ($(cmake --version | head -1))"

# Build in a scratch dir: config.sh downloads + extracts every source tarball
# into its CWD and drops stamp files there.
work="/tmp/gdal-src-build"
mkdir -p "${work}"
cd "${work}"
bash "${SCRIPT_DIR}/config.sh"

# Post-build wiring the vendor step expects.
echo "${GDAL_VERSION}" > "${BUILD_PREFIX}/GDAL_VERSION"

# curl CA bundle (#412): our from-source curl bakes a container path for its
# default CA file; ship a real bundle in the wheel and let the runtime
# bootstrap point GDAL/curl at it.
mkdir -p "${BUILD_PREFIX}/ssl"
for ca in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt; do
    if [[ -f "${ca}" ]]; then
        cp "${ca}" "${BUILD_PREFIX}/ssl/cacert.pem"
        echo "CA bundle staged from ${ca}"
        break
    fi
done
[[ -f "${BUILD_PREFIX}/ssl/cacert.pem" ]] || echo "WARNING: no CA bundle found; /vsicurl TLS will fail (#412)" >&2

# Driver-presence gate (the F0.2 allow-list + the raster formats pyramids
# ships). A missing driver here means a cmake flag or dep regressed — fail
# loudly now, not in the wheel tests.
echo "=== driver-presence gate ==="
_ogr_formats=$("${BUILD_PREFIX}/bin/ogrinfo" --formats)
_gdal_formats=$("${BUILD_PREFIX}/bin/gdalinfo" --formats)
_missing=0
for drv in "GeoJSON" "ESRI Shapefile" "GPKG" "GPX" "PMTiles" "MVT" "GML" "KML" "WFS" "OAPIF" "FlatGeobuf"; do
    if ! grep -qi -- "${drv}" <<<"${_ogr_formats}"; then
        echo "MISSING OGR driver: ${drv}" >&2; _missing=1
    fi
done
for drv in "GTiff" "COG" "netCDF" "GRIB" "HDF5" "JP2OpenJPEG" "Zarr" "PNG" "JPEG" "WCS" "VRT"; do
    if ! grep -qi -- "${drv}" <<<"${_gdal_formats}"; then
        echo "MISSING raster driver: ${drv}" >&2; _missing=1
    fi
done
if (( _missing )); then
    echo "ERROR: required drivers missing from the source-built GDAL (see above)" >&2
    exit 1
fi
echo "all required drivers present"
"${BUILD_PREFIX}/bin/gdal-config" --version
echo "=== from-source GDAL stack complete ==="
