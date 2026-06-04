#!/bin/bash
#
# Build a charset-only ICU data library and swap it for the full one conda
# ships, cutting ~30 MB of unused ICU data from the wheel (T2 of the wheel-size
# plan). conda's libicudata is ~33 MB of locale / collation / timezone /
# transliteration tables. The bundled GDAL stack only ever touches ICU's
# charset CONVERTERS (pulled transitively by libxml2 / libxerces-c); the rest
# is dead weight. We rebuild ICU's data with a filter that keeps the converter
# mappings and excludes everything else, then overwrite the libicudata that
# auditwheel will bundle from ${BUILD_PREFIX}/lib.
#
# See https://unicode-org.github.io/icu/userguide/icu_data/buildtool.html and
# planning/bundle/size/wheel-size-optimization-plan.md.
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
    # Fall back to the soname: libicudata.so.<major>.<minor>
    base=$(basename "$(ls "${BUILD_PREFIX}"/lib/libicudata.so.*.* 2>/dev/null | head -1)")
    icu_ver=${base#libicudata.so.}
fi
icu_major=${icu_ver%%.*}
echo "build-icu-min-data: ICU ${icu_ver} (major ${icu_major})"

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT

# Download helper. The manylinux container's curl is built without an HTTPS
# backend (curl: (4) ...), so prefer wget (what rasterio uses here), then fall
# back to curl, then python urllib.
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

# Fetch the prepared ICU4C source tarball for this release. Asset naming is
# `icu4c-<ver>-sources.tgz` under tag `release-<ver>` (dots, not dashes), e.g.
# release-78.3/icu4c-78.3-sources.tgz. It extracts to `icu/source/`.
tag="release-${icu_ver}"
tgz="icu4c-${icu_ver}-sources.tgz"
url="https://github.com/unicode-org/icu/releases/download/${tag}/${tgz}"
echo "build-icu-min-data: fetching ${url}"
_fetch "${url}" "${work}/icu.tgz"
tar -xzf "${work}/icu.tgz" -C "${work}"
src="${work}/icu/source"
if [[ ! -d "${src}" ]]; then
    echo "build-icu-min-data: ERROR ${src} missing after extract" >&2
    exit 1
fi

# Data filter: keep only converter mappings + their aliases; drop everything
# else (locales, collation, break iterators, transliterators, currency,
# region/lang/zone/unit trees, RBNF, char names). This is the ~30 MB win.
cat > "${work}/filter.json" <<'JSON'
{
  "featureFilters": {
    "brkitr_dictionaries": "exclude",
    "brkitr_rules": "exclude",
    "brkitr_tree": "exclude",
    "collation_tree": "exclude",
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
    # NOTE: the release `-sources.tgz` ships a prebuilt full data archive
    # (data/in/*.dat) that ICU repackages verbatim, ignoring the filter;
    # deleting it breaks the build ("No rule to make target out/tmp/icudata.lst")
    # because the tarball's data Makefile doesn't wire up the filter-aware Data
    # Build Tool for a bare from-source rebuild. Applying the filter needs the
    # git-archive source (full data tree + buildtool) — tracked as a follow-up.
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
echo "build-icu-min-data: filtered libicudata = ${built} ($(du -h "${built}" | cut -f1))"

# Overwrite the REAL libicudata files conda extracted (keep the symlinks so
# libicuuc's SONAME-based DT_NEEDED still resolves through them).
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
