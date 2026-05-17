#!/bin/bash
#
# =============================================================================
# DEPRECATED AS OF 2026-04-13
# =============================================================================
# This from-source build approach has been SUPERSEDED by
# ci/setup-gdal-from-pixi.sh, which installs conda-forge binaries via
# pixi. The new path is ~6x faster (6–10 min vs 45–60 min), doesn't trigger
# Docker OOM crashes, and uses pixi.lock for reproducibility.
#
# See docs/how-to/wheel-build-flow.md for the active pipeline that
# replaced this script.
#
# This script is kept as a fallback in case conda-forge ever drops a
# feature we need, or if we need tight control over compile flags that
# conda-forge's packages don't expose. DO NOT remove without moving the
# working library-version pins elsewhere — they represent hours of
# compatibility research.
# =============================================================================
#
# Build GDAL and all native dependencies from source for Linux manylinux wheels.
# Modeled on rasterio's ci/config.sh with additions for HDF4, NetCDF, and SWIG Python bindings.
#
# Usage: GDAL_VERSION=3.12.1 PYTHON_VERSION=3.12 bash ci/build-gdal-linux.sh
#
set -euo pipefail

BUILD_PREFIX="${BUILD_PREFIX:-/usr/local}"

# ---------------------------------------------------------------------------
# Library versions
# ---------------------------------------------------------------------------
ZLIB_VERSION=1.3.2
XZ_VERSION=5.8.2
OPENSSL_VERSION=3.6.1
NGHTTP2_VERSION=1.68.0
CURL_VERSION=8.18.0
SQLITE_VERSION=3530000
SQLITE_YEAR=2026
PROJ_VERSION=9.7.1
GEOS_VERSION=3.14.1
JSONC_VERSION=0.18
TIFF_VERSION=4.7.1
OPENJPEG_VERSION=2.5.4
LIBPNG_VERSION=1.6.54
JPEGTURBO_VERSION=3.1.3
LIBWEBP_VERSION=1.6.0
ZSTD_VERSION=1.5.7
LERC_VERSION=4.0.0
LIBDEFLATE_VERSION=1.24
GIFLIB_VERSION=5.2.2
PCRE_VERSION=10.47
EXPAT_VERSION=2.7.4
HDF4_VERSION=4.3.0
HDF5_VERSION=1.14.6
LIBAEC_VERSION=1.1.6
NETCDF_VERSION=4.10.0
BLOSC_VERSION=1.21.6
SWIG_VERSION=4.3.0
GDAL_VERSION="${GDAL_VERSION:?Set GDAL_VERSION}"

# ---------------------------------------------------------------------------
# Compiler flags for manylinux compatibility
# ---------------------------------------------------------------------------
export CFLAGS="-O2 -Wl,-strip-all -fPIC"
export CXXFLAGS="-O2 -Wl,-strip-all -fPIC"
export FFLAGS="-O2 -fPIC"
export CPPFLAGS="-I${BUILD_PREFIX}/include"
export LDFLAGS="-L${BUILD_PREFIX}/lib -L${BUILD_PREFIX}/lib64"
export LD_LIBRARY_PATH="${BUILD_PREFIX}/lib:${BUILD_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${BUILD_PREFIX}/lib/pkgconfig:${BUILD_PREFIX}/lib64/pkgconfig"
export PATH="${BUILD_PREFIX}/bin:${PATH}"
export CMAKE_PREFIX_PATH="${BUILD_PREFIX}"

NPROC=$(nproc)
SRC_DIR="/tmp/gdal-build-src"
mkdir -p "${SRC_DIR}"

# ---------------------------------------------------------------------------
# Helper: download + extract
#
# Uses wget as primary (the system one from yum, which has its own trusted
# CA bundle and works reliably with all host certificates). Falls back to
# curl only if wget is unavailable. We avoid relying on our freshly-built
# curl because it's linked against a from-source OpenSSL that doesn't always
# have a usable default CA bundle path.
# ---------------------------------------------------------------------------
fetch() {
    local url="$1" dest="$2"
    if [ ! -f "${dest}" ]; then
        echo ">>> Downloading ${url}"
        if command -v wget >/dev/null 2>&1; then
            wget --no-verbose --tries=3 --timeout=60 -O "${dest}" "${url}"
        else
            /usr/bin/curl -fsSL -o "${dest}" "${url}"
        fi
    fi
}

# ---------------------------------------------------------------------------
# zlib
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libz.so" ]; then
    echo "=== Building zlib ${ZLIB_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.gz" \
        "zlib-${ZLIB_VERSION}.tar.gz"
    tar xzf "zlib-${ZLIB_VERSION}.tar.gz"
    cd "zlib-${ZLIB_VERSION}"
    ./configure --prefix="${BUILD_PREFIX}"
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# xz (lzma)
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/liblzma.so" ]; then
    echo "=== Building xz ${XZ_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/tukaani-project/xz/releases/download/v${XZ_VERSION}/xz-${XZ_VERSION}.tar.gz" \
        "xz-${XZ_VERSION}.tar.gz"
    tar xzf "xz-${XZ_VERSION}.tar.gz"
    cd "xz-${XZ_VERSION}"
    ./configure --prefix="${BUILD_PREFIX}" --disable-static
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# OpenSSL
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib64/libssl.so" ] && [ ! -f "${BUILD_PREFIX}/lib/libssl.so" ]; then
    echo "=== Building OpenSSL ${OPENSSL_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VERSION}/openssl-${OPENSSL_VERSION}.tar.gz" \
        "openssl-${OPENSSL_VERSION}.tar.gz"
    tar xzf "openssl-${OPENSSL_VERSION}.tar.gz"
    cd "openssl-${OPENSSL_VERSION}"
    ./config --prefix="${BUILD_PREFIX}" --openssldir="${BUILD_PREFIX}/ssl" shared
    make -j"${NPROC}" && make install_sw
fi

# ---------------------------------------------------------------------------
# nghttp2
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libnghttp2.so" ]; then
    echo "=== Building nghttp2 ${NGHTTP2_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/nghttp2/nghttp2/releases/download/v${NGHTTP2_VERSION}/nghttp2-${NGHTTP2_VERSION}.tar.gz" \
        "nghttp2-${NGHTTP2_VERSION}.tar.gz"
    tar xzf "nghttp2-${NGHTTP2_VERSION}.tar.gz"
    cd "nghttp2-${NGHTTP2_VERSION}"
    ./configure --prefix="${BUILD_PREFIX}" --disable-static --enable-lib-only
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# curl
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libcurl.so" ]; then
    echo "=== Building curl ${CURL_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://curl.se/download/curl-${CURL_VERSION}.tar.gz" \
        "curl-${CURL_VERSION}.tar.gz"
    tar xzf "curl-${CURL_VERSION}.tar.gz"
    cd "curl-${CURL_VERSION}"
    ./configure --prefix="${BUILD_PREFIX}" \
        --disable-static \
        --with-openssl="${BUILD_PREFIX}" \
        --with-nghttp2="${BUILD_PREFIX}" \
        --without-libpsl \
        --without-brotli \
        --without-zstd \
        --without-libidn2
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libsqlite3.so" ]; then
    echo "=== Building SQLite ${SQLITE_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://www.sqlite.org/${SQLITE_YEAR}/sqlite-autoconf-${SQLITE_VERSION}.tar.gz" \
        "sqlite-autoconf-${SQLITE_VERSION}.tar.gz"
    tar xzf "sqlite-autoconf-${SQLITE_VERSION}.tar.gz"
    cd "sqlite-autoconf-${SQLITE_VERSION}"
    CFLAGS="${CFLAGS} -DSQLITE_ENABLE_RTREE=1 -DSQLITE_ENABLE_COLUMN_METADATA=1" \
        ./configure --prefix="${BUILD_PREFIX}" --disable-static
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# PROJ (with symbol renaming to avoid conflicts with system PROJ / pyproj)
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libproj.so" ]; then
    echo "=== Building PROJ ${PROJ_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://download.osgeo.org/proj/proj-${PROJ_VERSION}.tar.gz" \
        "proj-${PROJ_VERSION}.tar.gz"
    tar xzf "proj-${PROJ_VERSION}.tar.gz"
    cd "proj-${PROJ_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF \
        -DENABLE_CURL=ON \
        -DSQLITE3_INCLUDE_DIR="${BUILD_PREFIX}/include" \
        -DSQLITE3_LIBRARY="${BUILD_PREFIX}/lib/libsqlite3.so" \
        -DCURL_INCLUDE_DIR="${BUILD_PREFIX}/include" \
        -DCURL_LIBRARY="${BUILD_PREFIX}/lib/libcurl.so" \
        -DENABLE_TIFF=OFF
    # Note: PROJ_RENAME_SYMBOLS / PROJ_INTERNAL_CPP_NAMESPACE disabled —
    # they broke GDAL's FindPROJ. Revisit in Phase 6 if pyproj conflicts hit.
    # TIFF disabled in PROJ to avoid picking up the system libtiff's broken
    # CMake config in manylinux2014. PROJ only uses TIFF for some grid files.
    # NOTE: iso19111/factory.cpp is a ~120K-line generated file. Parallel compile
    # can OOM on constrained (WSL2) hosts. Limit PROJ build to 2 jobs.
    PROJ_JOBS=$(( NPROC > 2 ? 2 : NPROC ))
    make -j"${PROJ_JOBS}" && make install
    echo "=== PROJ post-install diagnostics ==="
    ls -la "${BUILD_PREFIX}/include/" | head -30 || true
    echo "--- proj headers ---"
    find "${BUILD_PREFIX}" -name "proj.h" 2>/dev/null
    echo "--- proj pkg-config ---"
    find "${BUILD_PREFIX}" -name "proj*.pc" 2>/dev/null
    echo "--- proj cmake config ---"
    find "${BUILD_PREFIX}" -name "proj*config*.cmake" 2>/dev/null
    echo "--- proj libs ---"
    find "${BUILD_PREFIX}" -name "libproj*" 2>/dev/null
    echo "=== End PROJ diagnostics ==="
fi

# ---------------------------------------------------------------------------
# GEOS
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libgeos.so" ]; then
    echo "=== Building GEOS ${GEOS_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://download.osgeo.org/geos/geos-${GEOS_VERSION}.tar.bz2" \
        "geos-${GEOS_VERSION}.tar.bz2"
    tar xjf "geos-${GEOS_VERSION}.tar.bz2"
    cd "geos-${GEOS_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# json-c
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libjson-c.so" ] && [ ! -f "${BUILD_PREFIX}/lib64/libjson-c.so" ]; then
    echo "=== Building json-c ${JSONC_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://s3.amazonaws.com/json-c_releases/releases/json-c-${JSONC_VERSION}.tar.gz" \
        "json-c-${JSONC_VERSION}.tar.gz"
    tar xzf "json-c-${JSONC_VERSION}.tar.gz"
    cd "json-c-${JSONC_VERSION}"
    mkdir -p build && cd build
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# libjpeg-turbo
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libjpeg.so" ] && [ ! -f "${BUILD_PREFIX}/lib64/libjpeg.so" ]; then
    echo "=== Building libjpeg-turbo ${JPEGTURBO_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${JPEGTURBO_VERSION}/libjpeg-turbo-${JPEGTURBO_VERSION}.tar.gz" \
        "libjpeg-turbo-${JPEGTURBO_VERSION}.tar.gz"
    tar xzf "libjpeg-turbo-${JPEGTURBO_VERSION}.tar.gz"
    cd "libjpeg-turbo-${JPEGTURBO_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_STATIC=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# libpng
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libpng.so" ]; then
    echo "=== Building libpng ${LIBPNG_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/pnggroup/libpng/archive/refs/tags/v${LIBPNG_VERSION}.tar.gz" \
        "libpng-${LIBPNG_VERSION}.tar.gz"
    tar xzf "libpng-${LIBPNG_VERSION}.tar.gz"
    cd "libpng-${LIBPNG_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPNG_SHARED=ON \
        -DPNG_STATIC=OFF \
        -DPNG_TESTS=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# libdeflate
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libdeflate.so" ]; then
    echo "=== Building libdeflate ${LIBDEFLATE_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/ebiggers/libdeflate/releases/download/v${LIBDEFLATE_VERSION}/libdeflate-${LIBDEFLATE_VERSION}.tar.gz" \
        "libdeflate-${LIBDEFLATE_VERSION}.tar.gz"
    tar xzf "libdeflate-${LIBDEFLATE_VERSION}.tar.gz"
    cd "libdeflate-${LIBDEFLATE_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBDEFLATE_BUILD_SHARED_LIB=ON \
        -DLIBDEFLATE_BUILD_STATIC_LIB=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# zstd
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libzstd.so" ]; then
    echo "=== Building zstd ${ZSTD_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/facebook/zstd/releases/download/v${ZSTD_VERSION}/zstd-${ZSTD_VERSION}.tar.gz" \
        "zstd-${ZSTD_VERSION}.tar.gz"
    tar xzf "zstd-${ZSTD_VERSION}.tar.gz"
    cd "zstd-${ZSTD_VERSION}/build/cmake"
    mkdir -p builddir && cd builddir
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DZSTD_BUILD_SHARED=ON \
        -DZSTD_BUILD_STATIC=OFF \
        -DZSTD_BUILD_PROGRAMS=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# libwebp
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libwebp.so" ]; then
    echo "=== Building libwebp ${LIBWEBP_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/webmproject/libwebp/archive/refs/tags/v${LIBWEBP_VERSION}.tar.gz" \
        "libwebp-${LIBWEBP_VERSION}.tar.gz"
    tar xzf "libwebp-${LIBWEBP_VERSION}.tar.gz"
    cd "libwebp-${LIBWEBP_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DWEBP_BUILD_ANIM_UTILS=OFF \
        -DWEBP_BUILD_CWEBP=OFF \
        -DWEBP_BUILD_DWEBP=OFF \
        -DWEBP_BUILD_GIF2WEBP=OFF \
        -DWEBP_BUILD_IMG2WEBP=OFF \
        -DWEBP_BUILD_WEBPINFO=OFF \
        -DWEBP_BUILD_VWEBP=OFF \
        -DWEBP_BUILD_EXTRAS=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# LERC
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libLerc.so" ] && [ ! -f "${BUILD_PREFIX}/lib64/libLerc.so" ]; then
    echo "=== Building LERC ${LERC_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/Esri/lerc/archive/refs/tags/v${LERC_VERSION}.tar.gz" \
        "lerc-${LERC_VERSION}.tar.gz"
    tar xzf "lerc-${LERC_VERSION}.tar.gz"
    cd "lerc-${LERC_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# libtiff
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libtiff.so" ]; then
    echo "=== Building libtiff ${TIFF_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://download.osgeo.org/libtiff/tiff-${TIFF_VERSION}.tar.gz" \
        "tiff-${TIFF_VERSION}.tar.gz"
    tar xzf "tiff-${TIFF_VERSION}.tar.gz"
    cd "tiff-${TIFF_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -Dtiff-tests=OFF \
        -Dtiff-tools=OFF \
        -Dtiff-contrib=OFF \
        -Dtiff-docs=OFF \
        -Dzstd=ON \
        -Dwebp=ON \
        -Djpeg=ON \
        -Dlzma=ON \
        -Dlerc=ON \
        -Dlibdeflate=ON
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# OpenJPEG
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libopenjp2.so" ]; then
    echo "=== Building OpenJPEG ${OPENJPEG_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/uclouvain/openjpeg/archive/refs/tags/v${OPENJPEG_VERSION}.tar.gz" \
        "openjpeg-${OPENJPEG_VERSION}.tar.gz"
    tar xzf "openjpeg-${OPENJPEG_VERSION}.tar.gz"
    cd "openjpeg-${OPENJPEG_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_CODEC=OFF \
        -DBUILD_TESTING=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# giflib
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libgif.so" ]; then
    echo "=== Building giflib ${GIFLIB_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://sourceforge.net/projects/giflib/files/giflib-${GIFLIB_VERSION}.tar.gz/download" \
        "giflib-${GIFLIB_VERSION}.tar.gz"
    tar xzf "giflib-${GIFLIB_VERSION}.tar.gz"
    cd "giflib-${GIFLIB_VERSION}"
    # Patch Makefile to skip the `doc` subdirectory — it needs ImageMagick's
    # `convert` tool which isn't in manylinux2014 and isn't needed for GDAL.
    sed -i -e 's/\bdoc\b//g' Makefile
    make -j"${NPROC}" PREFIX="${BUILD_PREFIX}" libgif.so libgif.a libutil.so libutil.a
    make install-include install-lib PREFIX="${BUILD_PREFIX}"
fi

# ---------------------------------------------------------------------------
# PCRE2
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libpcre2-8.so" ]; then
    echo "=== Building PCRE2 ${PCRE_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-${PCRE_VERSION}/pcre2-${PCRE_VERSION}.tar.gz" \
        "pcre2-${PCRE_VERSION}.tar.gz"
    tar xzf "pcre2-${PCRE_VERSION}.tar.gz"
    cd "pcre2-${PCRE_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_STATIC_LIBS=OFF \
        -DPCRE2_BUILD_PCRE2GREP=OFF \
        -DPCRE2_BUILD_TESTS=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# Expat
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libexpat.so" ]; then
    echo "=== Building Expat ${EXPAT_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/libexpat/libexpat/releases/download/R_$(echo ${EXPAT_VERSION} | tr . _)/expat-${EXPAT_VERSION}.tar.gz" \
        "expat-${EXPAT_VERSION}.tar.gz"
    tar xzf "expat-${EXPAT_VERSION}.tar.gz"
    cd "expat-${EXPAT_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DEXPAT_BUILD_EXAMPLES=OFF \
        -DEXPAT_BUILD_TESTS=OFF \
        -DEXPAT_BUILD_TOOLS=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# blosc
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libblosc.so" ]; then
    echo "=== Building blosc ${BLOSC_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/Blosc/c-blosc/archive/refs/tags/v${BLOSC_VERSION}.tar.gz" \
        "blosc-${BLOSC_VERSION}.tar.gz"
    tar xzf "blosc-${BLOSC_VERSION}.tar.gz"
    cd "c-blosc-${BLOSC_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED=ON \
        -DBUILD_STATIC=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# libaec (for HDF5 szip compression)
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libaec.so" ] && [ ! -f "${BUILD_PREFIX}/lib64/libaec.so" ]; then
    echo "=== Building libaec ${LIBAEC_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://gitlab.dkrz.de/k202009/libaec/-/archive/v${LIBAEC_VERSION}/libaec-v${LIBAEC_VERSION}.tar.gz" \
        "libaec-${LIBAEC_VERSION}.tar.gz"
    tar xzf "libaec-${LIBAEC_VERSION}.tar.gz"
    cd "libaec-v${LIBAEC_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# HDF4 (pyramids-specific: needed for libgdal-hdf4)
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libhdf4.so" ] && [ ! -f "${BUILD_PREFIX}/lib/libmfhdf.so" ]; then
    echo "=== Building HDF4 ${HDF4_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/HDFGroup/hdf4/releases/download/hdf${HDF4_VERSION}/hdf${HDF4_VERSION}.tar.gz" \
        "hdf4-${HDF4_VERSION}.tar.gz"
    # HDF4 tarball extracts to ./hdfsrc/ (not a versioned dir)
    rm -rf hdfsrc
    tar xzf "hdf4-${HDF4_VERSION}.tar.gz"
    cd hdfsrc
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_STATIC_LIBS=OFF \
        -DHDF4_BUILD_FORTRAN=OFF \
        -DHDF4_BUILD_UTILS=OFF \
        -DHDF4_BUILD_TOOLS=OFF \
        -DHDF4_BUILD_EXAMPLES=OFF \
        -DHDF4_ENABLE_SZIP_SUPPORT=ON \
        -DHDF4_ENABLE_Z_LIB_SUPPORT=ON
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libhdf5.so" ]; then
    echo "=== Building HDF5 ${HDF5_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/HDFGroup/hdf5/releases/download/hdf5_${HDF5_VERSION}/hdf5-${HDF5_VERSION}.tar.gz" \
        "hdf5-${HDF5_VERSION}.tar.gz"
    tar xzf "hdf5-${HDF5_VERSION}.tar.gz"
    cd "hdf5-${HDF5_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_STATIC_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_TOOLS=OFF \
        -DHDF5_BUILD_EXAMPLES=OFF \
        -DHDF5_BUILD_FORTRAN=OFF \
        -DHDF5_ENABLE_Z_LIB_SUPPORT=ON \
        -DHDF5_ENABLE_SZIP_SUPPORT=ON
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# NetCDF
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libnetcdf.so" ]; then
    echo "=== Building NetCDF ${NETCDF_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/Unidata/netcdf-c/archive/refs/tags/v${NETCDF_VERSION}.tar.gz" \
        "netcdf-${NETCDF_VERSION}.tar.gz"
    tar xzf "netcdf-${NETCDF_VERSION}.tar.gz"
    cd "netcdf-c-${NETCDF_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DENABLE_TESTS=OFF \
        -DENABLE_DAP=ON \
        -DENABLE_NETCDF_4=ON \
        -DENABLE_HDF4=ON \
        -DHDF5_DIR="${BUILD_PREFIX}" \
        -DCURL_INCLUDE_DIR="${BUILD_PREFIX}/include" \
        -DCURL_LIBRARY="${BUILD_PREFIX}/lib/libcurl.so"
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# SWIG (needed to build GDAL Python bindings)
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/bin/swig" ]; then
    echo "=== Building SWIG ${SWIG_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://github.com/swig/swig/archive/refs/tags/v${SWIG_VERSION}.tar.gz" \
        "swig-${SWIG_VERSION}.tar.gz"
    tar xzf "swig-${SWIG_VERSION}.tar.gz"
    cd "swig-${SWIG_VERSION}"
    ./autogen.sh
    ./configure --prefix="${BUILD_PREFIX}" --without-pcre
    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# GDAL (with Python SWIG bindings)
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_PREFIX}/lib/libgdal.so" ]; then
    echo "=== Building GDAL ${GDAL_VERSION} ==="
    cd "${SRC_DIR}"
    fetch "https://download.osgeo.org/gdal/${GDAL_VERSION}/gdal-${GDAL_VERSION}.tar.gz" \
        "gdal-${GDAL_VERSION}.tar.gz"
    tar xzf "gdal-${GDAL_VERSION}.tar.gz"
    cd "gdal-${GDAL_VERSION}"
    mkdir -p build && cd build
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_APPS=OFF \
        -DBUILD_TESTING=OFF \
        \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DBUILD_JAVA_BINDINGS=OFF \
        -DBUILD_CSHARP_BINDINGS=OFF \
        -DSWIG_EXECUTABLE="${BUILD_PREFIX}/bin/swig" \
        -DPython_EXECUTABLE="/opt/python/cp312-cp312/bin/python" \
        -DPython3_EXECUTABLE="/opt/python/cp312-cp312/bin/python" \
        \
        -DGDAL_BUILD_OPTIONAL_DRIVERS=ON \
        -DOGR_BUILD_OPTIONAL_DRIVERS=ON \
        \
        -DGDAL_USE_CURL=ON \
        -DGDAL_USE_GEOS=ON \
        -DGDAL_USE_GEOTIFF_INTERNAL=ON \
        -DGDAL_USE_ICONV=ON \
        -DGDAL_USE_JSONC=ON \
        -DGDAL_USE_ZLIB=ON \
        -DGDAL_USE_ZLIB_INTERNAL=OFF \
        -DGDAL_USE_TIFF=ON \
        -DGDAL_USE_TIFF_INTERNAL=OFF \
        -DGDAL_USE_SQLITE3=ON \
        -DGDAL_USE_PCRE2=ON \
        -DGDAL_USE_LERC=ON \
        -DGDAL_USE_LERC_INTERNAL=OFF \
        -DGDAL_USE_HDF5=ON \
        -DGDAL_USE_NETCDF=ON \
        -DGDAL_USE_HDF4=ON \
        \
        -DGDAL_ENABLE_DRIVER_GTIFF=ON \
        -DGDAL_ENABLE_DRIVER_GIF=ON \
        -DGDAL_ENABLE_DRIVER_GRIB=ON \
        -DGDAL_ENABLE_DRIVER_JPEG=ON \
        -DGDAL_ENABLE_DRIVER_PNG=ON \
        -DGDAL_ENABLE_DRIVER_OPENJPEG=ON \
        -DGDAL_ENABLE_DRIVER_HDF5=ON \
        -DGDAL_ENABLE_DRIVER_NETCDF=ON \
        -DGDAL_ENABLE_DRIVER_HDF4=ON \
        -DOGR_ENABLE_DRIVER_SQLITE=ON \
        -DOGR_ENABLE_DRIVER_GPKG=ON \
        -DOGR_ENABLE_DRIVER_GEOJSON=ON \
        -DOGR_ENABLE_DRIVER_SHAPE=ON \
        \
        -DGDAL_USE_SFCGAL=OFF \
        -DGDAL_USE_XERCESC=OFF \
        -DGDAL_USE_LIBXML2=OFF \
        -DGDAL_USE_POSTGRESQL=OFF \
        -DGDAL_USE_ODBC=OFF \
        -DGDAL_USE_JXL=OFF \
        -DGDAL_USE_OPENEXR=OFF \
        -DGDAL_USE_HEIF=OFF \
        \
        -DPROJ_INCLUDE_DIR="${BUILD_PREFIX}/include" \
        -DPROJ_LIBRARY="${BUILD_PREFIX}/lib/libproj.so"

    make -j"${NPROC}" && make install
fi

# ---------------------------------------------------------------------------
# Strip shared libraries to reduce wheel size
# ---------------------------------------------------------------------------
echo "=== Stripping shared libraries ==="
find "${BUILD_PREFIX}/lib" -name "*.so*" -exec strip --strip-unneeded {} \; 2>/dev/null || true
find "${BUILD_PREFIX}/lib64" -name "*.so*" -exec strip --strip-unneeded {} \; 2>/dev/null || true

echo "=== GDAL build complete ==="
echo "GDAL_CONFIG: ${BUILD_PREFIX}/bin/gdal-config"
"${BUILD_PREFIX}/bin/gdal-config" --version
echo "GDAL_DATA: ${BUILD_PREFIX}/share/gdal"
echo "PROJ_DATA: ${BUILD_PREFIX}/share/proj"
python3 -c "from osgeo import gdal; print(f'GDAL Python bindings OK: {gdal.__version__}')" || \
    echo "WARNING: GDAL Python bindings not importable yet (may need PYTHONPATH adjustment)"
