#!/bin/bash
#
# Verify that every wheel in the given directory is within the size budgets
# and ships no build-only artifacts. Emits a GitHub Actions ::notice:: line
# per wheel (compressed + uncompressed size) and ::error::/exits non-zero on
# the first wheel that:
#   - exceeds WHEEL_SIZE_BUDGET_MB        (compressed .whl, what PyPI stores)
#   - exceeds WHEEL_INSTALLED_BUDGET_MB   (uncompressed, the install footprint)
#   - ships a header / static lib / pkgconfig file (a BUILD_PREFIX/include leak)
#
# Usage:
#   WHEEL_SIZE_BUDGET_MB=120 WHEEL_INSTALLED_BUDGET_MB=250 ci/check-wheel-size.sh wheelhouse
#
# Called from .github/workflows/bundle-pypi-wheels.yml's "Report wheel sizes"
# step. See planning/bundle/wheel-size-analysis.md for the rationale behind the
# compressed ceiling, and planning/bundle/size/wheel-size-optimization-plan.md
# (T3.1 / T3.3, issue #474) for the uncompressed gate + leak assertion.
set -euo pipefail

: "${WHEEL_SIZE_BUDGET_MB:?WHEEL_SIZE_BUDGET_MB must be set}"
# Uncompressed (install-footprint) ceiling. Defaults generous so it backstops
# a regression (e.g. an un-shrunk ICU or a re-added driver doubling the
# footprint) without false-failing across platforms; the per-wheel ::notice::
# below surfaces the real numbers so the ceiling can be tightened later.
WHEEL_INSTALLED_BUDGET_MB="${WHEEL_INSTALLED_BUDGET_MB:-250}"

wheel_dir="${1:-wheelhouse}"
if [ ! -d "${wheel_dir}" ]; then
    echo "ERROR: ${wheel_dir} is not a directory" >&2
    exit 1
fi

shopt -s nullglob
wheels=( "${wheel_dir}"/*.whl )
shopt -u nullglob

if [ "${#wheels[@]}" -eq 0 ]; then
    echo "ERROR: no wheels found in ${wheel_dir}" >&2
    exit 1
fi

compressed_budget=$(( WHEEL_SIZE_BUDGET_MB * 1024 * 1024 ))
installed_budget=$(( WHEEL_INSTALLED_BUDGET_MB * 1024 * 1024 ))

# Resolve the interpreter: CI Linux/macOS have `python3`; a Windows/local
# invocation may only expose `python`. Used for the zip introspection below.
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "ERROR: neither python3 nor python found — needed to inspect wheels" >&2
    exit 1
fi

# stat flag differs by platform: GNU stat is `-c%s`, BSD/macOS stat is `-f%z`.
case "$(uname -s)" in
    Darwin) stat_size_flag=("-f%z") ;;
    *)      stat_size_flag=("-c%s") ;;
esac

# Build-only artifacts that must never ship in a wheel: a leaked C/C++ header
# tree (BUILD_PREFIX/include lands as an `include/` subtree), static archives
# (.a), and pkgconfig files (.pc). We flag the `include/` subtree rather than a
# bare `*.h`, so a legitimate data file that merely ends in `.h` does not
# false-trip the gate (only a real header *tree* does). `.lib` is deliberately
# NOT flagged: it collides with the LGPL license texts the wheel ships
# (`COPYING.LIB` / `COPYING3.LIB`), and delvewheel bundles DLLs, not static
# libs. A Python member walk is more reliable than unzip text parsing.
_leak_members() {  # _leak_members <wheel> -> prints any leaking archive members
    local whl="$1"
    "${PY}" - "${whl}" <<'PY'
import sys, zipfile
# Extensions no shipped data/license file legitimately uses (unambiguous build
# output). Note: NOT ".lib" — that matches the COPYING.LIB / COPYING3.LIB
# license texts under _licenses/.
build_exts = (".a", ".pc")
with zipfile.ZipFile(sys.argv[1]) as zf:
    for name in zf.namelist():
        low = name.lower()
        if low.endswith(build_exts) or "/include/" in low or low.startswith("include/"):
            print(name)
PY
}

_uncompressed_bytes() {  # _uncompressed_bytes <wheel> -> total uncompressed size
    local whl="$1"
    "${PY}" - "${whl}" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    print(sum(i.file_size for i in zf.infolist()))
PY
}

for whl in "${wheels[@]}"; do
    base=$(basename "${whl}")
    compressed=$(stat "${stat_size_flag[@]}" "${whl}")
    uncompressed=$(_uncompressed_bytes "${whl}")
    compressed_mb=$(awk -v s="${compressed}" 'BEGIN {printf "%.1f", s/1048576}')
    uncompressed_mb=$(awk -v s="${uncompressed}" 'BEGIN {printf "%.1f", s/1048576}')
    echo "::notice::${base}: ${compressed_mb} MB compressed / ${uncompressed_mb} MB installed"

    if [[ "${compressed}" -gt "${compressed_budget}" ]]; then
        echo "::error::${base} exceeds ${WHEEL_SIZE_BUDGET_MB} MB compressed budget" \
            "— see planning/bundle/wheel-size-analysis.md" >&2
        exit 1
    fi
    if [[ "${uncompressed}" -gt "${installed_budget}" ]]; then
        echo "::error::${base} exceeds ${WHEEL_INSTALLED_BUDGET_MB} MB installed (uncompressed)" \
            "budget — install-footprint regression; see planning/bundle/size/" \
            "wheel-size-optimization-plan.md (T3.1)" >&2
        exit 1
    fi

    leaks=$(_leak_members "${whl}")
    if [[ -n "${leaks}" ]]; then
        echo "::error::${base} ships build-only artifacts (headers/.a/.pc leak, T3.3):" >&2
        echo "${leaks}" | sed 's/^/  /' >&2
        exit 1
    fi
done
