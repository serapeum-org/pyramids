#!/bin/bash
# From-source native stack for the pyramids Linux wheels (#332 glibc, #333 musl).
#
# Invoked by ci/source-build/build-gdal-stack.sh inside the cibuildwheel
# container (manylinux_2_28 or musllinux_1_2). Compiles GDAL and its whole
# dependency tree from SHA256-pinned release tarballs so the wheel tags at
# the container image's own libc floor.
#
# Environment contract:
#   BUILD_PREFIX  install prefix (default /usr/local)
#   GDAL_VERSION  GDAL release to build (required; the SHA256 below pins the
#                 default 3.13.1 — bump both together or the fetch fails)
#
# Contract with build-gdal-stack.sh:
#   - runs in a scratch dir; every tarball extracts here and the source
#     trees are LEFT IN PLACE (the license collector reads each tree's
#     LICENSE/COPYING afterwards, keyed on the <name>-<version> dir names)
#   - installs everything into ${BUILD_PREFIX}
#
# Driver policy (evidence: planning/bundle/from-source-phase0-audit.md):
#   - OGR vector allow-list only (OGR_BUILD_OPTIONAL_DRIVERS=OFF + explicit
#     enables): GeoJSON/ESRIJSON, SHAPE, GPKG, GPX, PMTiles, MVT, FlatGeobuf,
#     GML, KML, WFS, OAPIF, SQLite, OSM — the set FeatureCollection uses.
#   - OGCAPI stays ON: Dataset.from_ogc_coverages hard-requires it
#     (src/pyramids/dataset/_ogc_coverages.py); it needs only curl.
#   - Optional raster drivers stay auto-ON (Zarr via blosc/zstd, WCS via
#     curl; GRIB/netCDF/HDF5/JP2 enabled explicitly).
#   - Deliberately absent (verified unused by the test suite): ICU, xerces,
#     libxml2, spatialite, jxl, libkml, muparser, HDF4, postgres, poppler.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: this script builds the Linux wheel stack only (got $(uname -s));" >&2
    echo "       macOS and Windows wheels use the conda-extract / vcpkg paths." >&2
    exit 1
fi

BUILD_PREFIX="${BUILD_PREFIX:-/usr/local}"
GDAL_VERSION="${GDAL_VERSION:?GDAL_VERSION must be set (e.g. 3.13.1)}"
ARCH="$(uname -m)"

case "${ARCH}" in
    x86_64)  OPENSSL_TARGET="linux-x86_64" ;;
    aarch64) OPENSSL_TARGET="linux-aarch64" ;;
    *) echo "ERROR: unsupported architecture ${ARCH}" >&2; exit 1 ;;
esac

# musl (Alpine) removed the LFS64 aliases (off64_t/pread64/pwrite64) in
# 1.2.4+ — its plain pread/pwrite are already 64-bit. Only glibc builds may
# define HAVE_PREAD64 (sqlite); detect the libc by the package manager.
if command -v apk >/dev/null 2>&1; then
    LIBC_FLAVOR="musl"
else
    LIBC_FLAVOR="glibc"
fi

# Toolchain: optimized, debug-info-free binaries. -Wl,-strip-all keeps the
# intermediate libraries small; build_gdal additionally strips libgdal
# itself after install.
export CFLAGS="${CFLAGS:--Wl,-strip-all} -g -O2"
export CXXFLAGS="${CXXFLAGS:--Wl,-strip-all} -g -O2"
export FFLAGS="${FFLAGS:--Wl,-strip-all}"
export CPPFLAGS="-I${BUILD_PREFIX}/include ${CPPFLAGS:-}"
export LIBRARY_PATH="${BUILD_PREFIX}/lib:${LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${BUILD_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export PATH="${BUILD_PREFIX}/bin:${PATH}"
export GDAL_CONFIG="${BUILD_PREFIX}/bin/gdal-config"
export PROJ_DATA="${BUILD_PREFIX}/share/proj"

echo "=== pyramids source stack: ${LIBC_FLAVOR}/${ARCH} -> ${BUILD_PREFIX} (GDAL ${GDAL_VERSION}) ==="

# Every dependency, pinned by version AND content hash. TARBALL is the local
# file name; its stem is also the extraction dir the license collector reads.
# GDAL's hash pins the default GDAL_VERSION.
declare -A VERSION SHA256 URL TARBALL

define() {
    local dep="$1"
    VERSION[$dep]="$2"
    SHA256[$dep]="$3"
    URL[$dep]="$4"
    TARBALL[$dep]="$5"
}

# define <dep> <version> <sha256> <url> <tarball>
define zlib       1.3.2     bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16 \
    "https://github.com/madler/zlib/releases/download/v1.3.2/zlib-1.3.2.tar.gz"                  zlib-1.3.2.tar.gz
define xz         5.8.2     ce09c50a5962786b83e5da389c90dd2c15ecd0980a258dd01f70f9e7ce58a8f1 \
    "https://github.com/tukaani-project/xz/releases/download/v5.8.2/xz-5.8.2.tar.gz"             xz-5.8.2.tar.gz
define nghttp2    1.68.0    2c16ffc588ad3f9e2613c3fad72db48ecb5ce15bc362fcc85b342e48daf51013 \
    "https://github.com/nghttp2/nghttp2/releases/download/v1.68.0/nghttp2-1.68.0.tar.gz"         nghttp2-1.68.0.tar.gz
define openssl    3.6.1     b1bfedcd5b289ff22aee87c9d600f515767ebf45f77168cb6d64f231f518a82e \
    "https://github.com/openssl/openssl/releases/download/openssl-3.6.1/openssl-3.6.1.tar.gz"    openssl-3.6.1.tar.gz
define curl       8.18.0    e9274a5f8ab5271c0e0e6762d2fce194d5f98acc568e4ce816845b2dcc0cf88f \
    "https://curl.se/download/curl-8.18.0.tar.gz"                                                curl-8.18.0.tar.gz
define libpng     1.6.54    ba7efce137409079989df4667706c339bebfbb10e9f413474718012a13c8cd4c \
    "https://github.com/pnggroup/libpng/archive/refs/tags/v1.6.54.tar.gz"                        libpng-1.6.54.tar.gz
define giflib     5.2.2     be7ffbd057cadebe2aa144542fd90c6838c6a083b5e8a9048b8ee3b66b29d5fb \
    "https://sourceforge.net/projects/giflib/files/giflib-5.2.2.tar.gz/download"                 giflib-5.2.2.tar.gz
define libwebp    1.6.0     93a852c2b3efafee3723efd4636de855b46f9fe1efddd607e1f42f60fc8f2136 \
    "https://github.com/webmproject/libwebp/archive/refs/tags/v1.6.0.tar.gz"                     libwebp-1.6.0.tar.gz
define zstd       1.5.7     37d7284556b20954e56e1ca85b80226768902e2edabd3b649e9e72c0c9012ee3 \
    "https://github.com/facebook/zstd/archive/v1.5.7.tar.gz"                                     zstd-1.5.7.tar.gz
define libdeflate 1.24      ad8d3723d0065c4723ab738be9723f2ff1cb0f1571e8bfcf0301ff9661f475e8 \
    "https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.24.tar.gz"                      libdeflate-1.24.tar.gz
define jpegturbo  3.1.3     075920b826834ac4ddf97661cc73491047855859affd671d52079c6867c1c6c0 \
    "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.3/libjpeg-turbo-3.1.3.tar.gz" \
    libjpeg-turbo-3.1.3.tar.gz
define lerc       4.0.0     91431c2b16d0e3de6cbaea188603359f87caed08259a645fd5a3805784ee30a0 \
    "https://github.com/Esri/lerc/archive/refs/tags/v4.0.0.tar.gz"                               lerc-4.0.0.tar.gz
define tiff       4.7.1     f698d94f3103da8ca7438d84e0344e453fe0ba3b7486e04c5bf7a9a3fabe9b69 \
    "https://download.osgeo.org/libtiff/tiff-4.7.1.tar.gz"                                       tiff-4.7.1.tar.gz
define lcms2      2.17      d11af569e42a1baa1650d20ad61d12e41af4fead4aa7964a01f93b08b53ab074 \
    "https://github.com/mm2/Little-CMS/releases/download/lcms2.17/lcms2-2.17.tar.gz"             lcms2-2.17.tar.gz
define openjpeg   2.5.4     a695fbe19c0165f295a8531b1e4e855cd94d0875d2f88ec4b61080677e27188a \
    "https://github.com/uclouvain/openjpeg/archive/refs/tags/v2.5.4.tar.gz"                      openjpeg-2.5.4.tar.gz
define jsonc      0.18      876ab046479166b869afc6896d288183bbc0e5843f141200c677b3e8dfb11724 \
    "https://s3.amazonaws.com/json-c_releases/releases/json-c-0.18.tar.gz"                       json-c-0.18.tar.gz
define sqlite     3510200   fbd89f866b1403bb66a143065440089dd76100f2238314d92274a082d4f2b7bb \
    "https://www.sqlite.org/2026/sqlite-autoconf-3510200.tar.gz"  sqlite-autoconf-3510200.tar.gz
define proj       9.7.1     6c097dc803c561929cdfcc46e4bf9945ea977611fb31493ad14e88edaeae260f \
    "https://download.osgeo.org/proj/proj-9.7.1.tar.gz"                                          proj-9.7.1.tar.gz
define expat      2.7.4     e6af11b01e32e5ef64906a5cca8809eabc4beb7ff2f9a0e6aabbd42e825135d0 \
    "https://github.com/libexpat/libexpat/releases/download/R_2_7_4/expat-2.7.4.tar.bz2"         expat-2.7.4.tar.bz2
define geos       3.14.1    3c20919cda9a505db07b5216baa980bacdaa0702da715b43f176fb07eff7e716 \
    "https://download.osgeo.org/geos/geos-3.14.1.tar.bz2"                                        geos-3.14.1.tar.bz2
define libaec     1.1.6     a469be4d835127e358c4f97de74943a54fbcb870aaf03cd2303c1dcc9fd4af4b \
    "https://github.com/MathisRosenhauer/libaec/releases/download/v1.1.6/libaec-1.1.6.tar.gz"    libaec-1.1.6.tar.gz
define hdf5       2.1.0     ce7f5515a95d588b8606c3fb50643f8b88ac52ffbbde9c63bb1edca6a256e964 \
    "https://github.com/HDFGroup/hdf5/releases/download/2.1.0/hdf5-2.1.0.tar.gz"                 hdf5-2.1.0.tar.gz
define netcdf     4.10.0    ce160f9c1483b32d1ba8b7633d7984510259e4e439c48a218b95a023dc02fd4c \
    "https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.10.0.tar.gz"                       netcdf-c-4.10.0.tar.gz
define blosc      1.21.6    9fcd60301aae28f97f1301b735f966cc19e7c49b6b4321b839b4579a0c156f38 \
    "https://github.com/Blosc/c-blosc/archive/refs/tags/v1.21.6.tar.gz"                          c-blosc-1.21.6.tar.gz
define pcre2      10.47     47fe8c99461250d42f89e6e8fdaeba9da057855d06eb7fc08d9ca03fd08d7bc7 \
    "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.47/pcre2-10.47.tar.bz2"    pcre2-10.47.tar.bz2
define gdal       "${GDAL_VERSION}" e04e9813bd215b56753d5554330c53be25f3df2d7ed7e6413a19e6b66751c675 \
    "https://download.osgeo.org/gdal/${GDAL_VERSION}/gdal-${GDAL_VERSION}.tar.gz"  "gdal-${GDAL_VERSION}.tar.gz"

# Build order: leaves first, GDAL last. Each entry is a src_<dep> function.
BUILD_ORDER=(
    zlib xz nghttp2 openssl curl
    libpng giflib libwebp zstd libdeflate jpegturbo lerc tiff lcms2 openjpeg
    jsonc sqlite proj expat geos
    libaec hdf5 netcdf blosc pcre2
    gdal
)

# Notes pinned by past CI failures — keep these with the mechanism they guard:
#   - xz downloads from its GitHub release mirror: tukaani.org is a single
#     host and timed out hard in CI (2026-07-03); the asset is the identical
#     official tarball and is SHA256-verified either way.
#   - the giflib tarball comes through sourceforge's redirector, hence the
#     /download suffix and the explicit local file name.

fetch() {
    # fetch <dep>: download the pinned tarball, verify its SHA256, extract.
    local dep="$1"
    local tarball="${TARBALL[$dep]}"
    wget --retry-connrefused --waitretry=30 --dns-timeout=20 \
        --connect-timeout=20 --read-timeout=300 --timeout=300 -t 5 \
        -O "${tarball}" "${URL[$dep]}"
    local actual
    actual="$(sha256sum "${tarball}" | cut -d ' ' -f1)"
    if [[ "${actual}" != "${SHA256[$dep]}" ]]; then
        echo "ERROR: SHA256 mismatch for ${dep} (${tarball})" >&2
        echo "       expected ${SHA256[$dep]}" >&2
        echo "       actual   ${actual}" >&2
        exit 1
    fi
    case "${tarball}" in
        *.tar.gz)  tar -xzf "${tarball}" ;;
        *.tar.bz2) tar -xjf "${tarball}" ;;
        *) echo "ERROR: unhandled archive type: ${tarball}" >&2; exit 1 ;;
    esac
}

src_dir() {
    # src_dir <dep>: the directory the dep's tarball extracts to.
    local dep="$1"
    local tarball="${TARBALL[$dep]}"
    tarball="${tarball%.tar.gz}"
    echo "${tarball%.tar.bz2}"
}

cmake_install() {
    # cmake_install <source-subdir> <parallelism> [cmake args...]
    # Configure in a fresh build dir, build, install. The common cache
    # arguments every dependency wants come first so callers pass deltas.
    local src="$1" jobs="$2"
    shift 2
    (
        cd "${src}"
        mkdir -p _pyramids_build && cd _pyramids_build
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
            -DCMAKE_PREFIX_PATH="${BUILD_PREFIX}" \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            "$@"
        cmake --build . -j "${jobs}"
        cmake --install .
    )
}

src_zlib() {
    (cd "$(src_dir zlib)" && ./configure --prefix="${BUILD_PREFIX}" && make && make install)
}

src_xz() {
    (cd "$(src_dir xz)" && ./configure --prefix="${BUILD_PREFIX}" && make && make install)
}

src_nghttp2() {
    (cd "$(src_dir nghttp2)" &&
        ./configure --enable-lib-only --prefix="${BUILD_PREFIX}" &&
        make -j "$(nproc)" && make install)
}

src_openssl() {
    (cd "$(src_dir openssl)" &&
        ./config "${OPENSSL_TARGET}" -fPIC --prefix="${BUILD_PREFIX}" &&
        make -j "$(nproc)" && make install)
}

src_curl() {
    # The container image may ship its own libcurl in the prefix; ours must
    # be the only one the linker can find.
    rm -rf "${BUILD_PREFIX}"/lib/libcurl* || true
    (cd "$(src_dir curl)" &&
        ./configure --prefix="${BUILD_PREFIX}" \
            --with-nghttp2="${BUILD_PREFIX}" \
            --with-zlib="${BUILD_PREFIX}" \
            --with-ssl="${BUILD_PREFIX}" \
            --enable-shared --without-libidn2 --without-libpsl &&
        make -j "$(nproc)" && make install)
}

src_libpng() {
    (cd "$(src_dir libpng)" && ./configure --prefix="${BUILD_PREFIX}" && make && make install)
}

src_giflib() {
    # giflib's default `all` target also renders documentation images via
    # ImageMagick, which the build images don't ship — build and install
    # only the library targets GDAL links against.
    (cd "$(src_dir giflib)" &&
        make libgif.a libgif.so &&
        make install-include install-lib PREFIX="${BUILD_PREFIX}")
}

src_libwebp() {
    (cd "$(src_dir libwebp)" &&
        ./autogen.sh &&
        ./configure --prefix="${BUILD_PREFIX}" --enable-libwebpmux --enable-libwebpdemux &&
        make && make install)
}

src_zstd() {
    # zstd's cmake project lives in a subdirectory of the source tree.
    (
        cd "$(src_dir zstd)/build/cmake"
        cmake . \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
            -DCMAKE_PREFIX_PATH="${BUILD_PREFIX}" \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            -DZSTD_LEGACY_SUPPORT=0 \
            -DSED_ERE_OPT=-r
        cmake --build .
        cmake --install .
    )
}

src_libdeflate() {
    cmake_install "$(src_dir libdeflate)" 4 -DBUILD_SHARED_LIBS=ON
}

src_jpegturbo() {
    # CMAKE_INSTALL_LIBDIR pinned so the library never lands in lib64 —
    # downstream configure flags point at ${BUILD_PREFIX}/lib.
    cmake_install "$(src_dir jpegturbo)" "$(nproc)" \
        -DCMAKE_INSTALL_LIBDIR="${BUILD_PREFIX}/lib" \
        -DWITH_JPEG8=1
}

src_lerc() {
    cmake_install "$(src_dir lerc)" 4 -DBUILD_SHARED_LIBS=ON -DENABLE_IPO=ON
}

src_tiff() {
    (cd "$(src_dir tiff)" &&
        ./configure --prefix="${BUILD_PREFIX}" --libdir="${BUILD_PREFIX}/lib" \
            --enable-zstd --enable-webp --enable-lerc \
            --with-jpeg-include-dir="${BUILD_PREFIX}/include" \
            --with-jpeg-lib-dir="${BUILD_PREFIX}/lib" &&
        make -j "$(nproc)" && make install)
}

src_lcms2() {
    (cd "$(src_dir lcms2)" &&
        ./configure --prefix="${BUILD_PREFIX}" &&
        make -j "$(nproc)" && make install)
}

src_openjpeg() {
    cmake_install "$(src_dir openjpeg)" "$(nproc)"
}

src_jsonc() {
    (
        cd "$(src_dir jsonc)"
        cmake . \
            -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        make -j "$(nproc)"
        make install
    )
}

src_sqlite() {
    # HAVE_PREAD64/HAVE_PWRITE64 are glibc-only (see the LIBC_FLAVOR note at
    # the top); scoped to this build so no other dependency sees the defines.
    local sqlite_cflags="${CFLAGS}"
    if [[ "${LIBC_FLAVOR}" == "glibc" ]]; then
        sqlite_cflags="${CFLAGS} -DHAVE_PREAD64 -DHAVE_PWRITE64"
    fi
    (cd "$(src_dir sqlite)" &&
        CFLAGS="${sqlite_cflags}" ./configure \
            --enable-rtree --enable-threadsafe --prefix="${BUILD_PREFIX}" &&
        make && make install)
}

src_proj() {
    # PROJ_RENAME_SYMBOLS namespaces PROJ's symbols so the wheel's copy can
    # coexist in-process with any other PROJ (pyproj bundles its own).
    # GDAL's build repeats the same defines when it consumes this PROJ.
    (
        cd "$(src_dir proj)"
        CFLAGS="${CFLAGS} -DPROJ_RENAME_SYMBOLS" \
        CXXFLAGS="${CXXFLAGS} -DPROJ_RENAME_SYMBOLS -DPROJ_INTERNAL_CPP_NAMESPACE" \
        cmake . \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
            -DCMAKE_PREFIX_PATH="${BUILD_PREFIX}" \
            -DCMAKE_INCLUDE_PATH="${BUILD_PREFIX}/include" \
            -DSQLite3_INCLUDE_DIR="${BUILD_PREFIX}/include" \
            -DSQLite3_LIBRARY="${BUILD_PREFIX}/lib/libsqlite3.so" \
            -DBUILD_SHARED_LIBS=ON \
            -DENABLE_IPO=ON \
            -DBUILD_APPS:BOOL=OFF \
            -DBUILD_TESTING:BOOL=OFF
        cmake --build . -j "$(nproc)"
        cmake --install .
    )
}

src_expat() {
    # Static + PIC, and it MUST be expat's CMake build: auditwheel's
    # manylinux policy whitelists libexpat.so.1 instead of vendoring it, so
    # a shared expat silently made the wheel depend on the host package
    # (absent on python:*-slim). Static linking removes the runtime dep.
    # The autotools build is unusable for this — even under
    # --disable-shared it installs CMake package files hardcoded to a
    # SHARED imported target, and GDAL's find_package(EXPAT) then fatals
    # on the missing libexpat.so (observed: run 28716182717).
    cmake_install "$(src_dir expat)" "$(nproc)" \
        -DEXPAT_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DEXPAT_BUILD_TOOLS=OFF \
        -DEXPAT_BUILD_EXAMPLES=OFF \
        -DEXPAT_BUILD_TESTS=OFF \
        -DEXPAT_BUILD_DOCS=OFF \
        -DEXPAT_BUILD_FUZZERS=OFF
}

src_geos() {
    cmake_install "$(src_dir geos)" 4 \
        -DBUILD_SHARED_LIBS=ON \
        -DENABLE_IPO=ON \
        -DBUILD_APPS:BOOL=OFF \
        -DBUILD_TESTING:BOOL=OFF
}

src_libaec() {
    # libaec is the szip-compatible compression HDF5 links against.
    cmake_install "$(src_dir libaec)" 4 \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_LIBDIR=lib
}

src_hdf5() {
    cmake_install "$(src_dir hdf5)" 4 \
        -DBUILD_SHARED_LIBS=ON \
        -DHDF5_ENABLE_ZLIB_SUPPORT=ON \
        -DZLIB_ROOT="${BUILD_PREFIX}" \
        -DHDF5_ENABLE_SZIP_SUPPORT:BOOL=ON \
        -Dlibaec_DIR="${BUILD_PREFIX}/lib/cmake/libaec" \
        -DSZIP_LIBRARY:PATH="${BUILD_PREFIX}/lib/libsz.so" \
        -DSZIP_INCLUDE_DIR="${BUILD_PREFIX}/include"
}

src_netcdf() {
    cmake_install "$(src_dir netcdf)" "$(nproc)" \
        -DBUILD_SHARED_LIBS=ON \
        -DENABLE_DAP=ON
}

src_blosc() {
    # Library only: GDAL's Zarr driver links libblosc, and the project's
    # bench/tests need feature macros musl doesn't define implicitly
    # (clock_gettime/CLOCK_MONOTONIC without _GNU_SOURCE).
    (
        cd "$(src_dir blosc)"
        cmake . \
            -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            -DBUILD_BENCHMARKS=OFF \
            -DBUILD_TESTS=OFF \
            -DBUILD_FUZZERS=OFF
        make install
    )
}

src_pcre2() {
    (cd "$(src_dir pcre2)" &&
        ./configure --prefix="${BUILD_PREFIX}" &&
        make -j "$(nproc)" && make install)
}

src_gdal() {
    (
        cd "$(src_dir gdal)"
        mkdir -p _pyramids_build && cd _pyramids_build
        CFLAGS="${CFLAGS} -DPROJ_RENAME_SYMBOLS" \
        CXXFLAGS="${CXXFLAGS} -DPROJ_RENAME_SYMBOLS -DPROJ_INTERNAL_CPP_NAMESPACE" \
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="${BUILD_PREFIX}" \
            -DCMAKE_PREFIX_PATH="${BUILD_PREFIX}" \
            -DCMAKE_INCLUDE_PATH="${BUILD_PREFIX}/include" \
            -DCMAKE_LIBRARY_PATH="${BUILD_PREFIX}/lib" \
            -DCMAKE_PROGRAM_PATH="${BUILD_PREFIX}/bin" \
            -DBUILD_SHARED_LIBS=ON \
            -DBUILD_PYTHON_BINDINGS=OFF \
            -DBUILD_JAVA_BINDINGS=OFF \
            -DBUILD_CSHARP_BINDINGS=OFF \
            -DGDAL_BUILD_OPTIONAL_DRIVERS=ON \
            -DOGR_BUILD_OPTIONAL_DRIVERS=OFF \
            -DGDAL_USE_CURL=ON \
            -DGDAL_USE_GEOS=ON \
            -DGDAL_USE_TIFF=ON \
            -DGDAL_USE_TIFF_INTERNAL=OFF \
            -DGDAL_USE_GEOTIFF_INTERNAL=ON \
            -DGDAL_USE_ICONV=ON \
            -DGDAL_USE_JSONC=ON \
            -DGDAL_USE_JSONC_INTERNAL=OFF \
            -DGDAL_USE_ZLIB=ON \
            -DGDAL_USE_ZLIB_INTERNAL=OFF \
            -DGDAL_USE_HDF5=ON \
            -DGDAL_USE_NETCDF=ON \
            -DGDAL_USE_SQLITE3=ON \
            -DGDAL_USE_PCRE2=ON \
            -DGDAL_USE_LERC=ON \
            -DGDAL_USE_LERC_INTERNAL=OFF \
            -DGDAL_USE_JXL=OFF \
            -DGDAL_USE_SFCGAL=OFF \
            -DGDAL_USE_XERCESC=OFF \
            -DGDAL_USE_LIBXML2=OFF \
            -DGDAL_USE_POSTGRESQL=OFF \
            -DGDAL_USE_OPENEXR=OFF \
            -DGDAL_USE_HEIF=OFF \
            -DGDAL_USE_ODBC=OFF \
            -DSQLite3_INCLUDE_DIR="${BUILD_PREFIX}/include" \
            -DSQLite3_LIBRARY="${BUILD_PREFIX}/lib/libsqlite3.so" \
            -DHDF5_INCLUDE_DIRS="${BUILD_PREFIX}/include" \
            -DPCRE2_INCLUDE_DIR="${BUILD_PREFIX}/include" \
            -DPCRE2-8_LIBRARY="${BUILD_PREFIX}/lib/libpcre2-8.so" \
            -DGDAL_ENABLE_DRIVER_GIF=ON \
            -DGDAL_ENABLE_DRIVER_GRIB=ON \
            -DGDAL_ENABLE_DRIVER_JPEG=ON \
            -DGDAL_ENABLE_DRIVER_PNG=ON \
            -DGDAL_ENABLE_DRIVER_HDF5=ON \
            -DGDAL_ENABLE_DRIVER_NETCDF=ON \
            -DGDAL_ENABLE_DRIVER_OPENJPEG=ON \
            -DGDAL_ENABLE_DRIVER_OGCAPI=ON \
            -DGDAL_ENABLE_DRIVER_MBTILES=ON \
            -DGDAL_ENABLE_DRIVER_AIGRID=ON \
            -DGDAL_ENABLE_DRIVER_AAIGRID=ON \
            -DGDAL_ENABLE_POSTGISRASTER=OFF \
            -DGDAL_ENABLE_EXR=OFF \
            -DGDAL_ENABLE_HEIF=OFF \
            -DOGR_ENABLE_DRIVER_SQLITE=ON \
            -DOGR_ENABLE_DRIVER_GPKG=ON \
            -DOGR_ENABLE_DRIVER_MVT=ON \
            -DOGR_ENABLE_DRIVER_OSM=ON \
            -DOGR_ENABLE_DRIVER_GEOJSON=ON \
            -DOGR_ENABLE_DRIVER_SHAPE=ON \
            -DOGR_ENABLE_DRIVER_GPX=ON \
            -DOGR_ENABLE_DRIVER_PMTILES=ON \
            -DOGR_ENABLE_DRIVER_FLATGEOBUF=ON \
            -DOGR_ENABLE_DRIVER_GML=ON \
            -DOGR_ENABLE_DRIVER_KML=ON \
            -DOGR_ENABLE_DRIVER_WFS=ON \
            -DOGR_ENABLE_DRIVER_OAPIF=ON \
            -DOGR_ENABLE_DRIVER_AVC=ON
        cmake --build . -j 4
        cmake --install .
    )
    # The intermediate -Wl,-strip-all only covers what the linker emits;
    # strip the installed libgdal again to drop everything else.
    strip -v --strip-unneeded "${BUILD_PREFIX}"/lib/libgdal.so.* 2>/dev/null || true
    strip -v --strip-unneeded "${BUILD_PREFIX}"/lib64/libgdal.so.* 2>/dev/null || true
}

echo "=== downloading + verifying ${#BUILD_ORDER[@]} pinned tarballs ==="
for dep in "${BUILD_ORDER[@]}"; do
    fetch "${dep}"
done

for dep in "${BUILD_ORDER[@]}"; do
    stamp=".pyramids-built-${dep}"
    if [[ -e "${stamp}" ]]; then
        echo "=== ${dep}: already built, skipping ==="
        continue
    fi
    echo "=== building ${dep} ${VERSION[$dep]} ==="
    "src_${dep}"
    touch "${stamp}"
done

echo "=== stack complete ==="
ls "${BUILD_PREFIX}/lib"
if [[ -d "${BUILD_PREFIX}/lib64" ]]; then
    ls "${BUILD_PREFIX}/lib64"
fi
"${GDAL_CONFIG}" --version
