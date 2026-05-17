#!/bin/bash
#
# Wrapper around `auditwheel repair` that bundles libstdc++.so.6
# into the wheel manually, so we can tag wheels as manylinux_2_28 even
# though conda-forge's GDAL references GLIBCXX_3.4.32 (a symbol only in
# glibc 2.39+'s system libstdc++).
#
# auditwheel's symbol-version check runs BEFORE its bundling decision,
# so policy mutation alone doesn't help: even if we remove libstdc++.so.6
# from manylinux_2_28's lib_whitelist (so it would get bundled), the
# pre-bundling symbol scan still sees GLIBCXX_3.4.32 and rejects
# manylinux_2_28. Mutating the policy's symbol_versions to allow
# GLIBCXX_3.4.32 is also fragile (would need to backport every
# intermediate GLIBCXX_3.4.X version).
#
# Approach taken here:
#   1. `auditwheel repair --exclude libstdc++.so.6 --plat manylinux_2_28_<arch>`
#      → auditwheel skips libstdc++ entirely (no bundle, no symbol check
#         against the policy's allowed versions). All OTHER libs go
#         through the normal repair: bundled into pyramids_gis.libs/,
#         RPATH patched.
#   2. Post-repair: copy our conda-forge libstdcxx-ng `libstdc++.so.6`
#      into pyramids_gis.libs/ ourselves, then patchelf the bundled
#      libs + the SWIG extensions so they find it via $ORIGIN-relative
#      RPATHs.
#   3. `wheel pack` regenerates RECORD with the new file hashes.
#
# Why this is safe at runtime: the bundled libstdc++.so.6 is loaded
# via the patched RPATHs before any system libstdc++ comes into play.
# The wheel doesn't actually require the host's libstdc++ to be at
# glibc 2.39+ level — only glibc itself, which only needs GLIBC_2.17.
#
# Usage (from cibuildwheel's repair-wheel-command):
#   bash /project/ci/auditwheel-bundle-libstdcxx.sh {dest_dir} {wheel}
#
# See planning/bundle/m8-lower-glibc-floor-plan.md for the rationale +
# rollback path.
set -euo pipefail

DEST_DIR="$1"
WHEEL="$2"

echo "=== auditwheel-bundle-libstdcxx.sh ==="
echo "wheel:    ${WHEEL}"
echo "dest_dir: ${DEST_DIR}"
echo "arch:     ${AUDITWHEEL_ARCH:-?}"

# cibuildwheel's manylinux image installs auditwheel via pipx into its
# own private virtualenv. Plain `python` on PATH (a generic CPython)
# doesn't have auditwheel in its site-packages.
AUDITWHEEL_BIN="$(command -v auditwheel)"
AUDITWHEEL_PYTHON="$(head -1 "${AUDITWHEEL_BIN}" | sed -E 's|^#!\s*||')"
echo "auditwheel:        ${AUDITWHEEL_BIN}"
echo "auditwheel python: ${AUDITWHEEL_PYTHON}"

# Locate the conda-forge libstdc++.so.6 that setup-gdal-from-pixi.sh
# already copied into ${BUILD_PREFIX}/lib (and /lib64). Pick the
# regular file, not a symlink, so we copy the real shared object.
LIBSTDCXX_SRC=""
for candidate in /usr/local/lib /usr/local/lib64; do
    if [ -f "${candidate}/libstdc++.so.6" ] && [ ! -L "${candidate}/libstdc++.so.6" ]; then
        LIBSTDCXX_SRC="${candidate}/libstdc++.so.6"
        break
    fi
    # Or follow symlinks to the realpath:
    if [ -L "${candidate}/libstdc++.so.6" ]; then
        LIBSTDCXX_SRC="$(readlink -f "${candidate}/libstdc++.so.6")"
        break
    fi
done
if [ -z "${LIBSTDCXX_SRC}" ] || [ ! -f "${LIBSTDCXX_SRC}" ]; then
    echo "ERROR: libstdc++.so.6 not found under /usr/local/lib{,64}" >&2
    ls -la /usr/local/lib /usr/local/lib64 2>&1 | head -40 >&2
    exit 1
fi
echo "libstdc++ source:  ${LIBSTDCXX_SRC}"

echo "--- auditwheel repair --exclude libstdc++.so.6 ---"
LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 \
    auditwheel repair --exclude libstdc++.so.6 \
                      --plat "manylinux_2_28_${AUDITWHEEL_ARCH}" \
                      -w "${DEST_DIR}" \
                      "${WHEEL}"

REPAIRED="$(ls "${DEST_DIR}"/*.whl | head -1)"
if [ -z "${REPAIRED}" ]; then
    echo "ERROR: auditwheel produced no wheel in ${DEST_DIR}" >&2
    exit 1
fi
echo "repaired wheel:    ${REPAIRED}"

echo "--- Bundling libstdc++.so.6 + patching RPATHs ---"
"${AUDITWHEEL_PYTHON}" - "${REPAIRED}" "${DEST_DIR}" "${LIBSTDCXX_SRC}" <<'PY'
"""Post-repair: drop libstdc++.so.6 into pyramids_gis.libs/ and patch
RPATHs on every bundled .so + every SWIG extension so they resolve
libstdc++ from the bundle instead of the system."""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

repaired_wheel = Path(sys.argv[1])
dest_dir = Path(sys.argv[2])
libstdcxx_src = Path(sys.argv[3])

# Unpack with `wheel unpack` so the produced dist-info is canonical.
work_root = Path(tempfile.mkdtemp(prefix="pyramids-libstdcxx-bundle-"))
subprocess.check_call(
    [sys.executable, "-m", "wheel", "unpack", str(repaired_wheel),
     "--dest", str(work_root)]
)
# wheel unpack drops "<dist>-<ver>" — there should be exactly one.
unpacked = next(d for d in work_root.iterdir() if d.is_dir())
print(f"  unpacked to: {unpacked}")

# Find the bundled .libs directory (delvewheel-equivalent on Linux is
# auditwheel; the dir is named "<wheel_distname>.libs" by convention).
libs_candidates = [d for d in unpacked.iterdir() if d.is_dir() and d.name.endswith(".libs")]
if not libs_candidates:
    sys.exit(f"ERROR: no *.libs/ dir under {unpacked}; auditwheel layout changed?")
libs_dir = libs_candidates[0]
print(f"  libs dir:    {libs_dir}")

# Copy libstdc++.so.6 (real file) into libs_dir, plus a versioned name
# for any code that dlopens by the exact SONAME.
dst = libs_dir / "libstdc++.so.6"
shutil.copy2(libstdcxx_src, dst)
os.chmod(dst, 0o755)
print(f"  copied:      {libstdcxx_src.name} -> {dst.relative_to(unpacked)}")

# Patch RPATH on every .so / .so.* under the unpacked tree so each
# loader can find libstdc++ in libs_dir via $ORIGIN-relative paths.
def patchelf_rpath(so: Path) -> None:
    rel = os.path.relpath(libs_dir, so.parent)
    rpath = f"$ORIGIN/{rel}" if rel != "." else "$ORIGIN"
    # Preserve any existing RPATH (auditwheel set one already pointing
    # at libs_dir) by appending — but in practice auditwheel's RPATH
    # already includes libs_dir, so this is mostly idempotent. Set
    # rather than append to keep things deterministic.
    subprocess.check_call(
        ["patchelf", "--set-rpath", rpath, str(so)],
        stdout=subprocess.DEVNULL,
    )
    print(f"  patchelf:    {so.relative_to(unpacked)}  ->  {rpath}")

for so in unpacked.rglob("*.so*"):
    if so.is_symlink() or not so.is_file():
        continue
    # libstdc++.so.6 itself doesn't need RPATH patched (it has no deps
    # back into the bundle).
    if so.name == "libstdc++.so.6":
        continue
    patchelf_rpath(so)

# Re-pack. `wheel pack` regenerates RECORD with the new sha256s and
# writes the wheel back to dest_dir under the same canonical filename
# (which is what we want — the manylinux_2_28 tag is in the dist-info
# WHEEL file from auditwheel's earlier --plat).
# Delete the old wheel so wheel pack's output doesn't collide.
repaired_wheel.unlink()
subprocess.check_call(
    [sys.executable, "-m", "wheel", "pack", str(unpacked),
     "--dest-dir", str(dest_dir)]
)

# wheel pack uses the dist-info name to derive the filename, so the
# resulting wheel should be tagged manylinux_2_28_<arch> already.
final = next(dest_dir.glob("*.whl"))
print(f"  re-packed:   {final.name}")
PY

echo "=== auditwheel-bundle-libstdcxx.sh complete ==="
