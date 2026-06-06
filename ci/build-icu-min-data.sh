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
# Linux (any arch, native runner) and macOS (native arch only — the
# x86_64-on-arm64 cross-build is tracked in #472). Windows ships no libicudata,
# so there is nothing to do there.
#
# Usage: build-icu-min-data.sh <BUILD_PREFIX> <PIXI_ENV>
set -euo pipefail

BUILD_PREFIX="$1"
PIXI_ENV="$2"

case "$(uname -s)" in
    Darwin) OS=macos; LIBEXT=dylib; ICU_CFG=MacOSX ;;
    Linux)  OS=linux; LIBEXT=so;    ICU_CFG=Linux  ;;
    *) echo "build-icu-min-data: unsupported OS $(uname -s); skipping"; exit 0 ;;
esac

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
    if command -v wget >/dev/null 2>&1; then
        wget --retry-connrefused --tries=5 --timeout=120 -qO "$2" "$1" && return 0
    fi
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 "$1" -o "$2" && return 0
    fi
    for py in python3 python; do
        if command -v "${py}" >/dev/null 2>&1; then
            "${py}" -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$1" "$2" \
                && return 0
        fi
    done
    return 1
}

tag="release-${icu_ver}"
url="https://github.com/unicode-org/icu/archive/refs/tags/${tag}.tar.gz"
echo "build-icu-min-data: fetching ${url}"
_fetch "${url}" "${work}/icu.tgz"
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

echo "build-icu-min-data: configuring + building ICU data (filtered)"
(
    cd "${src}"
    chmod +x configure runConfigureICU || true
    export ICU_DATA_FILTER_FILE="${work}/filter.json"
    if [[ "${OS}" == "macos" ]]; then
        # ICU must build against the system toolchain: configure's test binary
        # otherwise dies with "dyld: Symbol not found: _iconv" because conda's
        # libiconv is on the DYLD path, and configure needs SDKROOT to find the
        # macOS SDK ("C compiler cannot create executables").
        export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path 2>/dev/null || true)}"
        unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH 2>/dev/null || true
    fi
    ./runConfigureICU "${ICU_CFG}" \
        --disable-tests --disable-samples --disable-extras --disable-layoutex \
        --enable-shared --disable-static --with-data-packaging=library
    make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
)

built=$(find "${src}/lib" "${src}/data/out" -name "libicudata.*${LIBEXT}" -type f 2>/dev/null \
        | grep -E "libicudata\.(so\.|[0-9])" | head -1 || true)
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
