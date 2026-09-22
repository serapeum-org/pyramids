"""Strip the vendored vector-stack deps (geopandas, Shapely) from a musl wheel.

The musllinux wheel vendors shapely + pyogrio + geopandas under
`pyramids/_vendor/` (see ci/install-and-vendor-osgeo.py). On win_arm64 a PEP 508
marker (`sys_platform == 'win32' and platform_machine == 'ARM64'`) drops those
deps at install time; on musl there is NO environment marker that distinguishes
glibc from musl, so pip would otherwise fetch geopandas — and its transitive
pyogrio, which ships no musllinux wheel — on Alpine. Remove the geopandas +
Shapely `Requires-Dist` lines from the built wheel's METADATA (and fix the
matching RECORD row) so the musl wheel is self-contained.

Usage: python ci/strip-vendored-deps-from-wheel.py <wheel-or-dir> [...]
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

# PEP 503 normalized names of the deps the musl wheel vendors and must not
# declare. pyogrio is not a direct dep — it is dropped transitively by removing
# geopandas (which hard-requires it). cftime is dropped because it has no
# musllinux-aarch64 wheel, so the wheel vendors its own copy (see
# ci/install-and-vendor-osgeo.py).
_DROP = {"geopandas", "shapely", "cftime"}
_NAME_RE = re.compile(r"^Requires-Dist:\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _normalize(name: str) -> str:
    """Return the PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_drop_names(metadata: str) -> set[str]:
    """Return the normalized `_DROP` names declared as `Requires-Dist` in `metadata`.

    Independent of `_NAME_RE` (the removal matcher) on purpose: it audits the
    strip's result so an incomplete strip is caught even if the removal regex or
    `_DROP` ever drifts and leaves a vendored dep behind.
    """
    found = set()
    for line in metadata.split("\n"):
        if not line.strip().lower().startswith("requires-dist:"):
            continue
        rest = line.split(":", 1)[1].strip()
        name = re.split(r"[\s;(<>=!~\[]", rest, maxsplit=1)[0]
        norm = _normalize(name)
        if norm in _DROP:
            found.add(norm)
    return found


def _record_row(path: str, data: bytes) -> list[str]:
    """Return the RECORD row (path, sha256=<b64>, size) for `data`."""
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )
    return [path, f"sha256={digest}", str(len(data))]


def strip_wheel(wheel: Path) -> int:
    """Drop geopandas/Shapely Requires-Dist from `wheel` in place.

    Returns the number of `Requires-Dist` lines removed; raises if none matched,
    since the whole point of the step is to remove them and their absence means
    the metadata is not what this step expects (a marker or dep list changed).
    """
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        infos = {n: zf.getinfo(n) for n in names}
        data = {n: zf.read(n) for n in names}

    meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
    record_name = next(n for n in names if n.endswith(".dist-info/RECORD"))

    kept, removed = [], 0
    for line in data[meta_name].decode("utf-8").split("\n"):
        match = _NAME_RE.match(line)
        if match and _normalize(match.group(1)) in _DROP:
            removed += 1
            continue
        kept.append(line)
    if removed == 0:
        raise SystemExit(
            f"{wheel.name}: found no geopandas/Shapely/cftime Requires-Dist to strip — "
            "the metadata is not what this step expects (marker or deps changed?)"
        )
    new_meta = "\n".join(kept).encode("utf-8")
    survivors = _declared_drop_names(new_meta.decode("utf-8"))
    if survivors:
        raise SystemExit(
            f"{wheel.name}: vendored dep(s) {sorted(survivors)} still declared after "
            "the strip — the wheel would try to install a nonexistent musl wheel; the "
            "strip is incomplete (removal matcher or _DROP drifted?)"
        )
    data[meta_name] = new_meta

    # Refresh the METADATA row in RECORD (its hash + size changed); every other
    # row — including RECORD's own hash-less row — passes through unchanged.
    out_rows = []
    for row in csv.reader(io.StringIO(data[record_name].decode("utf-8"))):
        if not row:
            continue
        out_rows.append(
            _record_row(meta_name, new_meta) if row[0] == meta_name else row
        )
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(out_rows)
    data[record_name] = buf.getvalue().encode("utf-8")

    tmp = wheel.with_name(wheel.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            info = infos[name]
            new_info = zipfile.ZipInfo(name, date_time=info.date_time)
            new_info.compress_type = info.compress_type
            new_info.external_attr = info.external_attr
            zf.writestr(new_info, data[name])
    tmp.replace(wheel)
    return removed


def _iter_wheels(args: list[str]):
    """Yield every `.whl` named directly or contained in a named directory."""
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            yield from sorted(path.glob("*.whl"))
        elif path.suffix == ".whl" and path.is_file():
            yield path
        else:
            raise SystemExit(f"not a wheel or directory: {path}")


def main(argv: list[str]) -> None:
    """Strip every wheel named on the command line (files or directories)."""
    wheels = list(_iter_wheels(argv))
    if not wheels:
        raise SystemExit("no wheels found to process")
    for wheel in wheels:
        removed = strip_wheel(wheel)
        print(f"stripped {removed} vendored dep line(s) from {wheel.name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
