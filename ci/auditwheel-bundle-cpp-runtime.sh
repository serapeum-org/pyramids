#!/usr/bin/env bash
# Repair a cibuildwheel Linux wheel and tag it manylinux_2_28 while
# bundling the conda-forge GCC-13 C++ runtime.
#
# conda-forge's GDAL is built with GCC 13, so the SWIG ext + the native
# lib chain reference GLIBCXX_3.4.32 / CXXABI_1.3.15 / GCC_12.0.0 —
# symbols absent from the system libstdc++/libgcc on glibc < 2.39 hosts.
# auditwheel rejects any tag below manylinux_2_39 by default. We extend
# auditwheel's policy + bundle libstdc++.so.6 / libgcc_s.so.1 so the
# wheel can be tagged manylinux_2_28 (built in the manylinux_2_28 image).
#
# Two policy changes are both required (independent axes in auditwheel):
#   1. symbol_versions: ADD the GCC-13 GLIBCXX/CXXABI/GCC versions.
#   2. lib_whitelist: REMOVE libstdc++.so.6 + libgcc_s.so.1 so auditwheel
#      BUNDLES them. (Investigating whether bundling libgcc_s is the
#      source of the runtime segfault — see #332.)
# Applied to every manylinux policy (PEP 600 superset rule).
#
# See planning/bundle/review/gdal-bundling-diagnosis.md and #332/#338.
set -euo pipefail

DEST_DIR="$1"
WHEEL="$2"

AUDITWHEEL_BIN="$(command -v auditwheel)"
AUDITWHEEL_PY="$(head -1 "${AUDITWHEEL_BIN}" | sed -E 's|^#!\s*||')"

"${AUDITWHEEL_PY}" - <<'PY'
import json
import pathlib

import auditwheel

NEW_GLIBCXX = ["3.4.20", "3.4.21", "3.4.22", "3.4.23", "3.4.24",
               "3.4.25", "3.4.26", "3.4.27", "3.4.28",
               "3.4.29", "3.4.30", "3.4.31", "3.4.32"]
NEW_CXXABI = ["1.3.8", "1.3.9", "1.3.10", "1.3.11",
              "1.3.12", "1.3.13", "1.3.14", "1.3.15"]
NEW_GCC = ["4.9.0", "5.0.0", "6.0.0", "7.0.0", "8.0.0",
           "9.0.0", "10.0.0", "11.0.0", "12.0.0"]
DROP_LIBS = ("libstdc++.so.6", "libgcc_s.so.1")

root = pathlib.Path(auditwheel.__file__).parent
matches = list(root.rglob("manylinux-policy.json"))
if not matches:
    raise SystemExit("manylinux-policy.json not found under auditwheel package")
policy_file = matches[0]
policies = json.loads(policy_file.read_text())


def extend_symbol_versions(sv: dict) -> int:
    sample = next(iter(sv.values()), None)
    per_arch = list(sv.values()) if isinstance(sample, dict) else [sv]
    changed = 0
    for table in per_arch:
        for tag, news in (("GLIBCXX", NEW_GLIBCXX),
                          ("CXXABI", NEW_CXXABI),
                          ("GCC", NEW_GCC)):
            cur = table.setdefault(tag, [])
            for v in news:
                if v not in cur:
                    cur.append(v)
                    changed += 1
    return changed


mutations = 0
for p in policies:
    if p.get("name") == "linux":
        continue
    for lib in DROP_LIBS:
        if lib in p.get("lib_whitelist", []):
            p["lib_whitelist"].remove(lib)
            mutations += 1
    mutations += extend_symbol_versions(p.get("symbol_versions", {}))

policy_file.write_text(json.dumps(policies, indent=2))
print(f"auditwheel policy ({policy_file}): applied {mutations} mutation(s)")
PY

LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 \
    "${AUDITWHEEL_BIN}" -v repair \
    --plat "manylinux_2_28_${AUDITWHEEL_ARCH}" \
    -w "${DEST_DIR}" "${WHEEL}"
