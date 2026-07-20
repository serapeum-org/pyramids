#!/bin/bash
#
# Build the curated GDAL stack FROM SOURCE inside the wheel-build container,
# replacing ci/setup-gdal-from-pixi.sh as CIBW_BEFORE_ALL (#332 / #333).
# Everything compiles with the image's own toolchain, so the wheel tags at
# the image's floor:
#   - quay.io/pypa/manylinux_2_28_*  -> manylinux_2_28 (glibc >= 2.28)
#   - quay.io/pypa/musllinux_1_2_*   -> musllinux_1_2 (Alpine/musl — a
#     platform conda-forge cannot serve at all)
#
# Contract with the rest of the pipeline (mirrors setup-gdal-from-pixi.sh):
#   - installs the stack into ${BUILD_PREFIX} (default /usr/local): libs,
#     gdal-config, share/gdal, share/proj
#   - writes ${BUILD_PREFIX}/GDAL_VERSION for ci/install-and-vendor-osgeo.py
#   - stages the curl CA bundle at ${BUILD_PREFIX}/ssl/cacert.pem (#412)
#   - collects each dep's LICENSE/COPYING into
#     ${BUILD_PREFIX}/share/pyramids-bundled-licenses/<dep>/ so
#     install-and-vendor-osgeo.py can ship them in _licenses/ (no conda-meta
#     to mirror under the from-source model)
set -euo pipefail

export BUILD_PREFIX="${BUILD_PREFIX:-/usr/local}"
export GDAL_VERSION="${GDAL_VERSION:-3.13.1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# glibc (manylinux/AlmaLinux, dnf) vs musl (musllinux/Alpine, apk).
if command -v apk >/dev/null 2>&1; then
    LIBC_FLAVOR="musl"
else
    LIBC_FLAVOR="glibc"
fi
echo "=== from-source GDAL stack: GDAL ${GDAL_VERSION} -> ${BUILD_PREFIX} (${LIBC_FLAVOR} $(uname -m)) ==="

# Toolchain prerequisites.
#   - wget: config.sh's fetch helper.
#   - perl: OpenSSL's Configure needs full perl (IPC::Cmd, FindBin, Pod::*);
#     both base images ship a minimal or no perl.
#   - linux-headers (musl only): linux/userfaultfd.h so ENABLE_UFFD compiles
#     into the netCDF driver (manylinux gets it via kernel-headers already).
if [[ "${LIBC_FLAVOR}" == "musl" ]]; then
    apk add --no-cache wget perl linux-headers >/dev/null
else
    if ! command -v wget >/dev/null 2>&1; then
        (dnf install -y wget || yum install -y wget) >/dev/null
    fi
    if ! perl -MIPC::Cmd -e1 >/dev/null 2>&1; then
        (dnf install -y perl-core || yum install -y perl-core) >/dev/null
    fi
fi
# Any of the image's CPython installs works for the helper tasks below
# (cmake fallback, zipfile probe) — don't hardcode one ABI dir.
_PYBIN=$(ls -d /opt/python/cp3*/bin 2>/dev/null | head -1)
if [[ -z "${_PYBIN}" ]]; then
    echo "ERROR: no /opt/python/cp3*/bin interpreter found in this image" >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    pipx install cmake >/dev/null 2>&1 || "${_PYBIN}/pip" install --quiet cmake
    command -v cmake >/dev/null 2>&1 || export PATH="${_PYBIN}:${PATH}"
fi
# Fail fast with a clear message instead of an opaque "cmake: command not
# found" deep inside the first cmake-based dep (pipx can "succeed" into a
# bin dir that is not on PATH).
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake unavailable after bootstrap (pipx/pip install did not land on PATH)" >&2
    exit 1
fi
echo "cmake: $(command -v cmake) ($(cmake --version | head -1))"

# Dep-stack cache: the full compile is ~50 min; a tar of the installed
# prefix keyed on config.sh's hash + libc flavor + arch makes iteration
# cheap. CRITICAL: cibuildwheel COPIES the project into its container (no
# bind mount), so a tar written under /project silently dies with the
# container — actions/cache then warns "Path Validation Error ... no cache
# is being saved" and every build stays cold (observed: zero srcbuild
# caches ever created). cibuildwheel DOES bind-mount the host root at
# /host, so persist the tar through that using the runner workspace path
# forwarded via HOST_WORKSPACE (CIBW_ENVIRONMENT_PASS_LINUX). The project
# dir remains the fallback for non-cibuildwheel invocations.
PROJECT_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
if [[ -d /host && -n "${HOST_WORKSPACE:-}" ]]; then
    CACHE_DIR="/host${HOST_WORKSPACE}/.srcbuild-cache"
else
    CACHE_DIR="${PROJECT_DIR}/.srcbuild-cache"
fi
# Key on everything that shapes the installed prefix: config.sh (the dep
# pins + build flags), this script (license collector, prereqs), and the
# GDAL version, which is injected via the environment rather than
# config.sh's text and would otherwise restore a stale stack.
_cfg_hash=$(cat "${SCRIPT_DIR}/config.sh" "${BASH_SOURCE[0]}" | sha256sum | cut -c1-16)
# gzip the tar: four flavors x several key generations of uncompressed
# ~1 GB tars blow past GitHub's 10 GB per-repo cache pool and evict each
# other (observed 2026-07-03); compressed they coexist for many generations.
CACHE_TAR="${CACHE_DIR}/gdal-stack-${_cfg_hash}-gdal${GDAL_VERSION}-${LIBC_FLAVOR}-$(uname -m).tar.gz"

if [[ -f "${CACHE_TAR}" ]]; then
    echo "=== restoring cached stack ($(basename "${CACHE_TAR}")) ==="
    tar -C / -xzf "${CACHE_TAR}"
else
    # Build in a scratch dir: config.sh downloads + extracts every source
    # tarball into its CWD and drops stamp files there.
    work="/tmp/gdal-src-build"
    mkdir -p "${work}"
    cd "${work}"
    bash "${SCRIPT_DIR}/config.sh"

    # License collection (from-source replacement for the conda-meta mirror):
    # every dependency's source tree is still extracted in ${work}; copy each
    # LICENSE/COPYING variant into the prefix so it (a) ships in the wheel via
    # install-and-vendor-osgeo.py and (b) rides inside the cache tar.
    LIC_DST="${BUILD_PREFIX}/share/pyramids-bundled-licenses"
    mkdir -p "${LIC_DST}"
    for d in "${work}"/*/; do
        dep=$(basename "${d}")
        for f in LICENSE LICENSE.TXT LICENSE.txt LICENSE.md LICENSES.txt COPYING \
            COPYING.txt COPYING.LIB COPYING.LESSER COPYRIGHT LICENCE license.txt; do
            if [[ -f "${d}${f}" ]]; then
                mkdir -p "${LIC_DST}/${dep}"
                cp "${d}${f}" "${LIC_DST}/${dep}/${f}"
            fi
        done
    done
    echo "collected licenses for: $(ls "${LIC_DST}" | tr '\n' ' ')"

    cd /
    mkdir -p "${CACHE_DIR}"
    echo "=== caching built stack to $(basename "${CACHE_TAR}") ==="
    tar -C / -czf "${CACHE_TAR}" "${BUILD_PREFIX#/}"
fi

# Post-build wiring the vendor step expects.
echo "${GDAL_VERSION}" > "${BUILD_PREFIX}/GDAL_VERSION"

# curl CA bundle (#412): our from-source curl bakes a container path for its
# default CA file; ship a real bundle in the wheel and let the runtime
# bootstrap point GDAL/curl at it.
mkdir -p "${BUILD_PREFIX}/ssl"
for ca in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt \
    /etc/ssl/cert.pem; do
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
# Anchor on the "  <ShortName> -vector-/-raster-" line shape so a short name
# cannot be satisfied by a different driver that merely contains it (KML by
# KMLSuperOverlay/LIBKML, GML by GMLAS, ...).
for drv in "GeoJSON" "ESRI Shapefile" "GPKG" "GPX" "PMTiles" "MVT" "GML" "KML" "WFS" "OAPIF" "FlatGeobuf"; do
    if ! grep -q "^ *${drv} -" <<<"${_ogr_formats}"; then
        echo "MISSING OGR driver: ${drv}" >&2; _missing=1
    fi
done
for drv in "GTiff" "COG" "netCDF" "GRIB" "HDF5" "JP2OpenJPEG" "Zarr" "PNG" "JPEG" "WCS" "OGCAPI" "VRT"; do
    if ! grep -q "^ *${drv} -" <<<"${_gdal_formats}"; then
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

# License gate: every dep in the stack must have a collected license dir, so
# a partial collection (an upstream rename of LICENSE/COPYING, a path drift)
# fails the build instead of silently shipping a wheel missing notices. The
# list mirrors config.sh's pins; verified against each pinned tarball's root
# (2026-07-04). sqlite is the one deliberate absentee: it is public domain
# and its autoconf tarball ships no license file at all.
echo "=== license-completeness gate ==="
_lic_missing=0
for dep in c-blosc curl expat gdal geos giflib hdf5 json-c lcms2 lerc libaec libdeflate \
    libjpeg-turbo libpng libwebp netcdf-c nghttp2 openjpeg openssl pcre2 proj \
    tiff xz zlib zstd; do
    if ! ls -d "${BUILD_PREFIX}/share/pyramids-bundled-licenses/${dep}-"[0-9]* >/dev/null 2>&1; then
        echo "MISSING bundled license dir for: ${dep}" >&2; _lic_missing=1
    fi
done
if (( _lic_missing )); then
    echo "ERROR: license collection incomplete (see above) — the wheel would ship without" >&2
    echo "       legally required third-party notices" >&2
    exit 1
fi
echo "all $(ls "${BUILD_PREFIX}/share/pyramids-bundled-licenses" | wc -l) expected license dirs present"

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
# defect: install kernel-headers / linux-headers and rebuild.
echo "--- /vsizip netCDF probe ---"
"${BUILD_PREFIX}/bin/gdal_create" -of netCDF -outsize 8 8 -bands 1 /tmp/probe.nc
(cd /tmp && "${_PYBIN}/python" -m zipfile -c probe.zip probe.nc)
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
