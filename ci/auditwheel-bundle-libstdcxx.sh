#!/bin/bash
#
# Wrapper around `auditwheel repair` that forces libstdc++.so.6 to be
# bundled into the wheel instead of treated as a system-provided
# library. Called from .github/workflows/build-wheels.yml's Linux
# `repair-wheel-command` via cibuildwheel.
#
# Why this exists: conda-forge's GDAL is compiled with GCC 13 and
# references the symbol GLIBCXX_3.4.32, which is only in glibc 2.39's
# system libstdc++. auditwheel's built-in manylinux_2_28 policy lists
# libstdc++.so.6 in `lib_whitelist` (= "expect the host to provide
# this"), so when we ask for the manylinux_2_28 tag, auditwheel
# refuses with "cannot repair: too-recent versioned symbols".
#
# The libstdcxx-ng pin in [tool.pixi.feature.wheel-build.target.linux-*]
# already places a GCC-13 libstdc++.so.6 at ${BUILD_PREFIX}/lib/. All we
# need is to make auditwheel BUNDLE that one — i.e. drop libstdc++.so.6
# from the policy's lib_whitelist.
#
# auditwheel's --user-policy-json only ADDS policies (can't override
# built-ins by name without changing the --plat tag), so we mutate
# auditwheel's bundled policy file in place. Hacky but deterministic.
# The mutation only lasts for the lifetime of the cibuildwheel
# container — no persistent state.
#
# Usage (from cibuildwheel's repair-wheel-command):
#   bash /project/ci/auditwheel-bundle-libstdcxx.sh {dest_dir} {wheel}
#
# See planning/bundle/m8-lower-glibc-floor-plan.md for the full story.
set -euo pipefail

DEST_DIR="$1"
WHEEL="$2"

echo "=== auditwheel-bundle-libstdcxx.sh ==="
echo "wheel:    ${WHEEL}"
echo "dest_dir: ${DEST_DIR}"
echo "arch:     ${AUDITWHEEL_ARCH:-?}"

# cibuildwheel's manylinux image installs auditwheel via pipx into its
# own private virtualenv, so plain `python` on PATH (which is some
# generic CPython) doesn't import auditwheel. Read auditwheel's
# shebang to find the interpreter that DOES.
AUDITWHEEL_BIN="$(command -v auditwheel)"
AUDITWHEEL_PYTHON="$(head -1 "${AUDITWHEEL_BIN}" | sed -E 's|^#!\s*||')"
echo "auditwheel:        ${AUDITWHEEL_BIN}"
echo "auditwheel python: ${AUDITWHEEL_PYTHON}"

"${AUDITWHEEL_PYTHON}" <<'PY'
"""Mutate auditwheel's bundled manylinux policy to drop libstdc++.so.6.

Loads every JSON file shipped alongside auditwheel.policy and removes
`libstdc++.so.6` from EVERY manylinux entry's `lib_whitelist`.

Why every entry, not just manylinux_2_28: auditwheel's PEP 600
compliance validator (`_validate_pep600_compliance`) requires each
newer manylinux policy's `lib_whitelist` to be a SUPERSET of older
ones. Removing libstdc++.so.6 from manylinux_2_28 alone trips that
check ("Missing whitelist libraries in manylinux_2_28 compared to
previous policies"). Remove from all so the policies stay consistent.

We don't touch musllinux entries — they're unrelated to our use case
and removing libs there would be over-reach.
"""
import json
import sys
from pathlib import Path

import auditwheel.policy

policy_root = Path(auditwheel.policy.__file__).parent
policy_files = sorted(policy_root.rglob('*.json'))
if not policy_files:
    sys.exit(f"ERROR: no policy JSON files under {policy_root}")

mutations = 0
for pf in policy_files:
    text = pf.read_text()
    data = json.loads(text)
    # Schema: file may be a top-level list, or a dict whose only value is a list.
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        # Pick the first list-valued field.
        list_vals = [v for v in data.values() if isinstance(v, list)]
        entries = list_vals[0] if list_vals else []
    else:
        continue
    file_muts = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get('name', '')
        if not name.startswith('manylinux'):
            continue
        whitelist = entry.get('lib_whitelist')
        if isinstance(whitelist, list) and 'libstdc++.so.6' in whitelist:
            whitelist.remove('libstdc++.so.6')
            file_muts += 1
            print(f"  removed libstdc++.so.6 from {name}'s lib_whitelist")
    if file_muts:
        pf.write_text(json.dumps(data, indent=2))
        mutations += file_muts

if not mutations:
    sys.exit("ERROR: did not find any manylinux policy with libstdc++.so.6 "
             "in its lib_whitelist. auditwheel policy schema may have changed.")

print(f"Applied {mutations} mutation(s) to auditwheel's bundled policy")
PY

echo "--- Running auditwheel repair ---"
LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 \
    auditwheel repair --plat "manylinux_2_28_${AUDITWHEEL_ARCH}" \
                      -w "${DEST_DIR}" \
                      "${WHEEL}"
echo "=== auditwheel-bundle-libstdcxx.sh complete ==="
