#!/bin/bash
#
# Verify that every wheel in the given directory is at or under the
# WHEEL_SIZE_BUDGET_MB ceiling. Emits a GitHub Actions ::notice:: line
# for each wheel with its size in MB, and ::error::/exits non-zero on
# the first wheel that exceeds the budget.
#
# Usage:
#   WHEEL_SIZE_BUDGET_MB=120 ci/check-wheel-size.sh wheelhouse
#
# Called from .github/workflows/build-wheels.yml's "Report wheel sizes"
# step. See planning/bundle/wheel-size-analysis.md for the rationale
# behind the current 120 MB ceiling.
set -euo pipefail

: "${WHEEL_SIZE_BUDGET_MB:?WHEEL_SIZE_BUDGET_MB must be set}"

wheel_dir="${1:-wheelhouse}"
if [ ! -d "${wheel_dir}" ]; then
    echo "ERROR: ${wheel_dir} is not a directory" >&2
    exit 1
fi

budget_bytes=$(( WHEEL_SIZE_BUDGET_MB * 1024 * 1024 ))

shopt -s nullglob
wheels=( "${wheel_dir}"/*.whl )
shopt -u nullglob

if [ "${#wheels[@]}" -eq 0 ]; then
    echo "ERROR: no wheels found in ${wheel_dir}" >&2
    exit 1
fi

# stat flag differs by platform: GNU stat is `-c%s`, BSD/macOS stat is
# `-f%z`. Linux-only callsite today (build-wheels.yml only invokes
# this script from the build-linux-wheels job), so plain `-c%s` works,
# but the per-platform branch is here for local invocation.
case "$(uname -s)" in
    Linux)  stat_size_flag=("-c%s") ;;
    Darwin) stat_size_flag=("-f%z") ;;
    *)      stat_size_flag=("-c%s") ;;
esac

for whl in "${wheels[@]}"; do
    size=$(stat "${stat_size_flag[@]}" "${whl}")
    # Pass size via -v rather than embedding it in the awk script
    # source. The current input is digit-only (stat output) so injection
    # isn't a real risk today, but -v keeps it disciplined.
    size_mb=$(awk -v size="${size}" 'BEGIN {printf "%.1f", size/1048576}')
    echo "::notice::$(basename "${whl}"): ${size_mb} MB"
    if [ "${size}" -gt "${budget_bytes}" ]; then
        echo "::error::Wheel exceeds ${WHEEL_SIZE_BUDGET_MB} MB size budget — see planning/bundle/wheel-size-analysis.md"
        exit 1
    fi
done
