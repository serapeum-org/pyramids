#!/usr/bin/env python3
"""Report each freshly-built wheel's compressed size vs the previous PyPI release.

T3.2 of the wheel-size plan (issue #474). Best-effort and **non-fatal**: it
emits a GitHub Actions ``::notice::`` per wheel with the size delta against the
same platform/ABI tag from the latest already-published ``pyramids-gis``
version, so a driver/lib creep (or a win) is visible in the PR. Any failure
(offline, PyPI shape change, first-ever release of a tag) downgrades to a
single notice and exits 0 — this never blocks a build.

Usage:
    python3 ci/wheel-size-delta.py [wheelhouse]
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

PYPI_JSON = "https://pypi.org/pypi/pyramids-gis/json"


def _notice(msg: str) -> None:
    print(f"::notice::{msg}", flush=True)


def _tag(filename: str) -> str | None:
    """Return the ``{python}-{abi}-{platform}`` tag of a wheel filename.

    ``pyramids_gis-0.31.0-cp312-cp312-manylinux_2_39_x86_64.whl`` ->
    ``cp312-cp312-manylinux_2_39_x86_64``. Platform tags use underscores, not
    dashes, so a well-formed wheel name splits into exactly 5 dash parts.
    """
    stem = filename[:-4] if filename.endswith(".whl") else filename
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    return "-".join(parts[-3:])


def _version(filename: str) -> str | None:
    parts = filename[:-4].split("-")
    return parts[1] if len(parts) >= 5 else None


def _version_key(version: str) -> tuple[int, ...]:
    """Order versions numerically, not lexically (so 0.10.0 > 0.9.0).

    Splits on every run of digits; a pre-release/local suffix (``rc1``,
    ``.post0``) just contributes its own numbers, which is good enough for
    picking the newest release of a tag in this best-effort report.
    """
    return tuple(int(n) for n in re.findall(r"\d+", version))


def _fetch_releases() -> dict:
    with urllib.request.urlopen(PYPI_JSON, timeout=30) as resp:
        return json.load(resp).get("releases", {})


def main() -> int:
    wheel_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "wheelhouse")
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        _notice(f"wheel-size-delta: no wheels in {wheel_dir}; nothing to compare")
        return 0

    try:
        releases = _fetch_releases()
    except Exception as exc:  # noqa: BLE001 - best-effort, never fail the build
        _notice(f"wheel-size-delta: skipped (could not read PyPI: {exc})")
        return 0

    # Map each published platform/ABI tag -> {version: compressed_size}.
    by_tag: dict[str, dict[str, int]] = {}
    for version, files in releases.items():
        for f in files:
            if f.get("packagetype") != "bdist_wheel" or f.get("yanked"):
                continue
            tag = _tag(f.get("filename", ""))
            if tag:
                by_tag.setdefault(tag, {})[version] = f.get("size", 0)

    for whl in wheels:
        cur_bytes = whl.stat().st_size
        cur_mb = cur_bytes / 1048576
        tag = _tag(whl.name)
        cur_ver = _version(whl.name)
        prev = by_tag.get(tag or "", {})
        # Compare against the newest published version of this tag that isn't
        # the one we're building now.
        candidates = {v: s for v, s in prev.items() if v != cur_ver and s}
        if not candidates:
            _notice(f"{whl.name}: {cur_mb:.1f} MB (no prior PyPI wheel for this tag)")
            continue
        prev_ver = max(candidates, key=_version_key)  # numeric order: 0.10.0 > 0.9.0
        prev_bytes = candidates[prev_ver]
        delta_mb = (cur_bytes - prev_bytes) / 1048576
        sign = "+" if delta_mb >= 0 else "-"
        _notice(
            f"{whl.name}: {cur_mb:.1f} MB vs {prev_ver} {prev_bytes / 1048576:.1f} MB "
            f"(delta {sign}{abs(delta_mb):.1f} MB)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
