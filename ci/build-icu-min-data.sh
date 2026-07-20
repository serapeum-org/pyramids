#!/bin/bash
#
# Build a charset-only ICU data library and swap it for the full one conda
# ships, cutting ~30 MB of unused ICU data from the wheel (T2 of the wheel-size
# plan / issue #472). conda's libicudata is ~33 MB of ICU locale / collation /
# timezone / transliteration tables. The bundled GDAL stack only ever touches
# ICU's charset CONVERTERS (pulled transitively by libxml2 / libxerces-c); the
# rest is dead weight. We rebuild ICU's data with a filter that keeps converter
# mappings and excludes everything else, then overwrite the libicudata that
# auditwheel / delocate will bundle from ${BUILD_PREFIX}/lib.
#
# Build from the GIT-ARCHIVE source (the full repo at the release tag), NOT the
# packaged `-sources.tgz`: the packaged tarball ships a prebuilt full-data
# archive (data/in/*.dat) that ICU repackages verbatim, ignoring the filter;
# the git archive has the full data tree and no prebuilt .dat, so `make` runs
# the filter-aware ICU Data Build Tool.
#
# Linux (any arch, native runner) and macOS, including the macOS x86_64 wheel
# cross-built on the arm64 runner (target arch != host -> ICU --with-cross-build,
# building host tools + filtered data first). Windows ships no libicudata, so
# there is nothing to do there.
#
# Usage: build-icu-min-data.sh <BUILD_PREFIX> <PIXI_ENV> [TARGET_ARCH]
#   TARGET_ARCH defaults to the build host arch; pass the cibuildwheel target
#   arch (e.g. x86_64 on the macOS arm64 runner) to trigger the cross-build.
set -euo pipefail

BUILD_PREFIX="$1"
PIXI_ENV="$2"
TARGET_ARCH="${3:-$(uname -m)}"   # cibuildwheel target arch (may differ from host on macOS)
HOST_ARCH="$(uname -m)"

case "$(uname -s)" in
    Darwin) OS=macos; ICU_CFG=MacOSX ;;
    Linux)  OS=linux; ICU_CFG=Linux  ;;
    *) echo "build-icu-min-data: unsupported OS $(uname -s); skipping"; exit 0 ;;
esac

# Cross-compile when the target arch differs from the build host (macOS
# x86_64 wheels built on the arm64 runner). Normalize x86_64/amd64 spelling.
_norm_arch() {
    local arch="$1"
    case "${arch}" in
        x86_64 | amd64 | AMD64) echo x86_64 ;;
        arm64 | aarch64) echo arm64 ;;
        *) echo "${arch}" ;;
    esac
}
CROSS=0
[[ "$(_norm_arch "${TARGET_ARCH}")" != "$(_norm_arch "${HOST_ARCH}")" ]] && CROSS=1

# Locate the bundled libicudata (real files, not symlinks).
shopt -s nullglob
if [[ "${OS}" == "macos" ]]; then
    icudata_files=( "${BUILD_PREFIX}"/lib/libicudata.*.dylib "${BUILD_PREFIX}"/lib64/libicudata.*.dylib )
else
    icudata_files=( "${BUILD_PREFIX}"/lib/libicudata.so.* "${BUILD_PREFIX}"/lib64/libicudata.so.* )
fi
shopt -u nullglob
if (( ${#icudata_files[@]} == 0 )); then
    echo "build-icu-min-data: no libicudata under ${BUILD_PREFIX}; nothing to do"
    exit 0
fi

# Resolve the concrete ICU version conda installed (e.g. 78.3).
icu_meta=$(ls "${PIXI_ENV}"/conda-meta/icu-*.json 2>/dev/null | head -1 || true)
if [[ -n "${icu_meta}" ]]; then
    icu_ver=$(basename "${icu_meta}" | sed -E 's/^icu-([0-9]+\.[0-9]+).*/\1/')
else
    base=$(basename "${icudata_files[0]}")
    icu_ver=$(echo "${base}" | sed -E 's/^libicudata\.(so\.)?([0-9]+\.[0-9]+).*/\2/')
fi
icu_major=${icu_ver%%.*}
echo "build-icu-min-data: ${OS} ICU ${icu_ver} (major ${icu_major})"

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT

# Download helper: wget (manylinux curl lacks HTTPS) -> curl -> python.
_fetch() {  # _fetch <url> <dest>
    local url="$1" dest="$2"
    if command -v wget >/dev/null 2>&1; then
        wget --retry-connrefused --tries=5 --timeout=120 -qO "${dest}" "${url}" && return 0
    fi
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 "${url}" -o "${dest}" && return 0
    fi
    for py in python3 python; do
        if command -v "${py}" >/dev/null 2>&1; then
            "${py}" -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "${url}" "${dest}" \
                && return 0
        fi
    done
    return 1
}

# Pinned SHA256 of the ICU git-archive source tarballs. The tarball is compiled
# into the shipped wheel's libicudata, so verify it (supply chain). github's
# auto-generated archives are normally byte-stable per tag; if github ever
# recompresses and this mismatches, the build fails loudly — re-verify + update.
_icu_sha256() {
    local ver="$1"
    case "${ver}" in
        78.3) echo "f06bcab72736ee9d55689033b8198a178562354128cf38edb2afc2e67e3fd931" ;;
        *)    echo "" ;;
    esac
}

tag="release-${icu_ver}"
url="https://github.com/unicode-org/icu/archive/refs/tags/${tag}.tar.gz"
echo "build-icu-min-data: fetching ${url}"
_fetch "${url}" "${work}/icu.tgz"

_expected_sha="$(_icu_sha256 "${icu_ver}")"
if [[ -n "${_expected_sha}" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
        _actual_sha=$(sha256sum "${work}/icu.tgz" | awk '{print $1}')
    else
        _actual_sha=$(shasum -a 256 "${work}/icu.tgz" | awk '{print $1}')
    fi
    if [[ "${_actual_sha}" != "${_expected_sha}" ]]; then
        echo "build-icu-min-data: ERROR ICU source SHA256 mismatch for ${icu_ver}" >&2
        echo "  expected ${_expected_sha}" >&2
        echo "  actual   ${_actual_sha}" >&2
        exit 1
    fi
    echo "build-icu-min-data: ICU source SHA256 verified"
elif [[ "${PYRAMIDS_ICU_ALLOW_UNPINNED:-0}" == "1" ]]; then
    echo "build-icu-min-data: WARNING no pinned SHA256 for ICU ${icu_ver}; source unverified" \
        "(PYRAMIDS_ICU_ALLOW_UNPINNED=1)" >&2
else
    # No pin for this ICU version (e.g. an ICU bump). Refuse to compile an
    # unverified source tarball into the shipped wheel. Pin the new hash in
    # _icu_sha256 (compute it with `sha256sum` on the release-<ver> archive),
    # or set PYRAMIDS_ICU_ALLOW_UNPINNED=1 to deliberately opt out. The size
    # gate guards size, not integrity, so a missing pin must fail loudly here.
    echo "build-icu-min-data: ERROR no pinned SHA256 for ICU ${icu_ver}; refusing to build an" >&2
    echo "  unverified source tarball into the wheel. Pin its sha256 in _icu_sha256(), or set" >&2
    echo "  PYRAMIDS_ICU_ALLOW_UNPINNED=1 to opt out." >&2
    exit 1
fi

tar -xzf "${work}/icu.tgz" -C "${work}"
src=$(find "${work}" -maxdepth 3 -type d -path '*/icu4c/source' | head -1)
if [[ -z "${src}" || ! -d "${src}" ]]; then
    echo "build-icu-min-data: ERROR icu4c/source not found after extract" >&2
    exit 1
fi
echo "build-icu-min-data: source = ${src}"

# Filter: keep converter mappings + aliases; drop locales, collation
# (coll_tree), break iterators, transliterators, currency, region/lang/zone/
# unit trees, RBNF, char names. Excluding coll_tree also avoids a genrb
# segfault seen when collation is built.
cat > "${work}/filter.json" <<'JSON'
{
  "featureFilters": {
    "brkitr_dictionaries": "exclude",
    "brkitr_rules": "exclude",
    "brkitr_tree": "exclude",
    "coll_tree": "exclude",
    "curr_tree": "exclude",
    "lang_tree": "exclude",
    "locales_tree": "exclude",
    "misc": "exclude",
    "rbnf_tree": "exclude",
    "region_tree": "exclude",
    "translit": "exclude",
    "unames": "exclude",
    "unit_tree": "exclude",
    "zone_tree": "exclude"
  }
}
JSON

chmod +x "${src}/configure" "${src}/runConfigureICU" || true
_ncpu="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"

# Configure + build ICU in <build_dir> (in-source if build_dir == src, else
# a VPATH build). Extra configure args are passed through. ICU_ARCH_FLAGS adds
# a target -arch for the cross build.
_run_icu_build() {
    local bd="$1"; shift
    mkdir -p "${bd}"
    (
        cd "${bd}"
        export ICU_DATA_FILTER_FILE="${work}/filter.json"
        if [[ "${OS}" == "macos" ]]; then
            # System toolchain: conda's libiconv on the DYLD path otherwise
            # breaks configure's test binary (dyld: _iconv), and configure
            # needs SDKROOT for the macOS SDK.
            export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path 2>/dev/null || true)}"
            unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH 2>/dev/null || true
        fi
        if [[ -n "${ICU_ARCH_FLAGS:-}" ]]; then
            export CFLAGS="${CFLAGS:-} ${ICU_ARCH_FLAGS}"
            export CXXFLAGS="${CXXFLAGS:-} ${ICU_ARCH_FLAGS}"
            export LDFLAGS="${LDFLAGS:-} ${ICU_ARCH_FLAGS}"
        fi
        "${src}/runConfigureICU" "${ICU_CFG}" \
            --disable-tests --disable-samples --disable-extras --disable-layoutex "$@"
        make -j"${_ncpu}"
    )
}

if [[ "${CROSS}" == "1" ]]; then
    echo "build-icu-min-data: cross-compiling ${HOST_ARCH} -> ${TARGET_ARCH}"
    # 1) Host build (native): tools (genrb/pkgdata) + the filtered data the
    #    cross build reuses. 2) Cross build: target-arch libs via the host tools.
    _run_icu_build "${work}/host"
    ICU_ARCH_FLAGS="-arch ${TARGET_ARCH}" _run_icu_build "${work}/cross" \
        --with-cross-build="${work}/host" \
        --enable-shared --disable-static --with-data-packaging=library
    out_root="${work}/cross"
else
    echo "build-icu-min-data: configuring + building ICU data (filtered)"
    _run_icu_build "${src}" --enable-shared --disable-static --with-data-packaging=library
    out_root="${src}"
fi

if [[ "${OS}" == "macos" ]]; then
    built_pat="libicudata.*.dylib"   # libicudata.78.3.dylib
else
    built_pat="libicudata.so.*"      # libicudata.so.78.3
fi
built=$(find "${out_root}/lib" "${out_root}/data/out" -name "${built_pat}" -type f 2>/dev/null | head -1 || true)
if [[ -z "${built}" ]]; then
    echo "build-icu-min-data: ERROR built libicudata not found under ${src}" >&2
    exit 1
fi
echo "build-icu-min-data: filtered libicudata = ${built} ($(du -h "${built}" | cut -f1))"

# Overwrite the REAL libicudata files (keep SONAME / version symlinks). On
# macOS also match the original dylib's install id so delocate keeps the link.
replaced=0
for dst_dir in "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64"; do
    [[ -d "${dst_dir}" ]] || continue
    shopt -s nullglob
    if [[ "${OS}" == "macos" ]]; then
        dst_libs=( "${dst_dir}"/libicudata.*.dylib )
    else
        dst_libs=( "${dst_dir}"/libicudata.so.* )
    fi
    shopt -u nullglob
    for f in "${dst_libs[@]}"; do
        [[ -e "${f}" ]] || continue
        [[ -L "${f}" ]] && continue   # leave symlinks alone
        if [[ "${OS}" == "macos" ]]; then
            orig_id=$(otool -D "${f}" 2>/dev/null | tail -1 || true)
            cp -f "${built}" "${f}"
            [[ -n "${orig_id}" ]] && install_name_tool -id "${orig_id}" "${f}" 2>/dev/null || true
        else
            cp -f "${built}" "${f}"
        fi
        echo "build-icu-min-data: replaced ${f} -> $(du -h "${f}" | cut -f1)"
        replaced=$((replaced + 1))
    done
done
if (( replaced == 0 )); then
    echo "build-icu-min-data: ERROR replaced no libicudata files" >&2
    exit 1
fi
echo "build-icu-min-data: done (${replaced} file(s) replaced)"
