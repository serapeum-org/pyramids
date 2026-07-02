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

# Dep-stack cache: the full compile is ~50 min; a tar of the installed
# prefix keyed on config.sh's hash makes iteration cheap. The tar lives
# under the mounted project dir so the host job can persist it via
# actions/cache across runs.
PROJECT_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
CACHE_DIR="${PROJECT_DIR}/.srcbuild-cache"
_cfg_hash=$(sha256sum "${SCRIPT_DIR}/config.sh" | cut -c1-16)
CACHE_TAR="${CACHE_DIR}/gdal-stack-${_cfg_hash}-$(uname -m).tar"

if [[ -f "${CACHE_TAR}" ]]; then
    echo "=== restoring cached stack ($(basename "${CACHE_TAR}")) ==="
    tar -C / -xf "${CACHE_TAR}"
else
    # Build in a scratch dir: config.sh downloads + extracts every source
    # tarball into its CWD and drops stamp files there.
    work="/tmp/gdal-src-build"
    mkdir -p "${work}"
    cd "${work}"
    bash "${SCRIPT_DIR}/config.sh"
    cd /
    mkdir -p "${CACHE_DIR}"
    echo "=== caching built stack to $(basename "${CACHE_TAR}") ==="
    tar -C / -cf "${CACHE_TAR}" usr/local
fi

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

# Capability flags for the record ('v' after rw = virtual-IO support).
echo "--- driver capability flags ---"
grep -E "netCDF|GTiff|HDF5|Zarr|GRIB" <<<"${_gdal_formats}" || true

# /vsizip netCDF probe (informational). GDAL 3.13's netCDF driver opens
# non-/vsimem VSI paths (nc inside /vsizip, /vsitar, ...) ONLY via Linux
# userfaultfd (netcdfdataset.cpp: /vsimem -> nc_open_mem; other /vsi ->
# uffd; no in-memory fallback). Two acceptable outcomes here:
#   - read OK: uffd compiled AND permitted in this container;
#   - FAILED with a "requires Linux userfaultfd" message: uffd compiled but
#     blocked by docker's default seccomp — expected in-container; the wheel
#     works on real hosts / seccomp=unconfined containers. Same behavior as
#     the conda-extract wheel (parity).
# A FAILURE WITHOUT the userfaultfd message means ENABLE_UFFD was not even
# compiled (missing linux/userfaultfd.h at configure) — that's a real build
# defect: install kernel-headers and rebuild.
echo "--- /vsizip netCDF probe ---"
"${BUILD_PREFIX}/bin/gdal_create" -of netCDF -outsize 8 8 -bands 1 /tmp/probe.nc
(cd /tmp && /opt/python/cp312-cp312/bin/python -m zipfile -c probe.zip probe.nc)
if CPL_DEBUG=ON "${BUILD_PREFIX}/bin/gdalinfo" /vsizip//tmp/probe.zip/probe.nc >/tmp/probe.out 2>&1; then
    echo "vsizip netCDF read OK (userfaultfd available in this container)"
elif grep -qi "userfaultfd" /tmp/probe.out; then
    echo "vsizip netCDF blocked by container seccomp (uffd COMPILED — expected in docker; OK)"
else
    echo "ERROR: vsizip netCDF failed WITHOUT the userfaultfd message — ENABLE_UFFD likely" >&2
    echo "       not compiled (missing linux/userfaultfd.h). CPL_DEBUG tail:" >&2
    tail -40 /tmp/probe.out >&2
    exit 1
fi

"${BUILD_PREFIX}/bin/gdal-config" --version
echo "=== from-source GDAL stack complete ==="
