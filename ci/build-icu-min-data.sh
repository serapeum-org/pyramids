#!/bin/bash
#
# Build a charset-only ICU data library and swap it for the full one conda
# ships, cutting ~30 MB of unused ICU data from the wheel (T2 of the wheel-size
# plan). conda's libicudata is ~33 MB of ICU locale / collation / timezone /
# transliteration tables. The bundled GDAL stack only ever touches ICU's
# charset CONVERTERS (pulled transitively by libxml2 / libxerces-c); the rest
# is dead weight. We rebuild ICU's data with a filter that keeps the converter
# mappings and excludes everything else, then overwrite the libicudata that
# auditwheel will bundle from ${BUILD_PREFIX}/lib.
#
# IMPORTANT: build from the GIT-ARCHIVE source (the full repo at the release
# tag), NOT the packaged `-sources.tgz`. The packaged tarball ships a prebuilt
# full-data archive (data/in/*.dat) that ICU repackages verbatim, ignoring
# ICU_DATA_FILTER_FILE. The git archive has the full `icu4c/source/data/` tree
# and no prebuilt .dat, so `make` runs the filter-aware ICU Data Build Tool.
#
# See https://unicode-org.github.io/icu/userguide/icu_data/buildtool.html and
# planning/bundle/size/wheel-size-optimization-plan.md / issue #472.
#
# Usage: build-icu-min-data.sh <BUILD_PREFIX> <PIXI_ENV>
set -euo pipefail

BUILD_PREFIX="$1"
PIXI_ENV="$2"

shopt -s nullglob
icudata_files=( "${BUILD_PREFIX}/lib/libicudata.so."* "${BUILD_PREFIX}/lib64/libicudata.so."* )
shopt -u nullglob
if (( ${#icudata_files[@]} == 0 )); then
    echo "build-icu-min-data: no libicudata under ${BUILD_PREFIX}; nothing to do"
    exit 0
fi

# Resolve the concrete ICU version conda installed (e.g. 78.3) — the data
# format is version-stamped, so the rebuilt lib must be the same ICU release.
icu_meta=$(ls "${PIXI_ENV}"/conda-meta/icu-*.json 2>/dev/null | head -1 || true)
if [[ -n "${icu_meta}" ]]; then
    icu_ver=$(basename "${icu_meta}" | sed -E 's/^icu-([0-9]+\.[0-9]+).*/\1/')
else
    base=$(basename "$(ls "${BUILD_PREFIX}"/lib/libicudata.so.*.* 2>/dev/null | head -1)")
    icu_ver=${base#libicudata.so.}
fi
icu_major=${icu_ver%%.*}
echo "build-icu-min-data: ICU ${icu_ver} (major ${icu_major})"

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT

# Download helper. The manylinux container's curl is built without an HTTPS
# backend (curl: (4) ...), so prefer wget, then fall back to curl, then python.
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

# Fetch the GIT-ARCHIVE source for this release (full repo: icu4c/source/data
# present, no prebuilt .dat). Extracts to e.g. icu-release-78.3/icu4c/source.
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

# Data filter: keep only converter mappings + their aliases; drop everything
# else (locales, collation, break iterators, transliterators, currency,
# region/lang/zone/unit trees, RBNF, char names). This is the ~30 MB win.
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
    # Export the filter so BOTH configure and the data build (`make`, where the
    # ICU Data Build Tool applies it) see it.
    export ICU_DATA_FILTER_FILE="${work}/filter.json"
    ./runConfigureICU Linux \
        --disable-tests --disable-samples --disable-extras --disable-layoutex \
        --enable-shared --disable-static --with-data-packaging=library
    make -j"$(nproc)"
)

built=$(find "${src}/lib" "${src}/data/out" -name 'libicudata.so.*' -type f 2>/dev/null | head -1 || true)
if [[ -z "${built}" ]]; then
    echo "build-icu-min-data: ERROR built libicudata not found under ${src}" >&2
    exit 1
fi
built_sz=$(du -h "${built}" | cut -f1)
echo "build-icu-min-data: filtered libicudata = ${built} (${built_sz})"

# Overwrite the REAL libicudata files conda extracted (keep the SONAME symlinks
# so libicuuc's DT_NEEDED still resolves through them).
replaced=0
for dst_dir in "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64"; do
    [[ -d "${dst_dir}" ]] || continue
    for f in "${dst_dir}"/libicudata.so.*; do
        [[ -e "${f}" ]] || continue
        [[ -L "${f}" ]] && continue   # leave SONAME symlinks alone
        cp -f "${built}" "${f}"
        echo "build-icu-min-data: replaced ${f} -> $(du -h "${f}" | cut -f1)"
        replaced=$((replaced + 1))
    done
done
if (( replaced == 0 )); then
    echo "build-icu-min-data: ERROR replaced no libicudata files" >&2
    exit 1
fi
echo "build-icu-min-data: done (${replaced} file(s) replaced)"
