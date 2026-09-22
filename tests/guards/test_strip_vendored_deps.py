"""Guard ci/strip-vendored-deps-from-wheel.py, the musl wheel's self-containment step.

The musllinux wheel vendors shapely + pyogrio + geopandas under `pyramids/_vendor/`
(ci/install-and-vendor-osgeo.py). Because no PEP 508 marker distinguishes musl from
glibc, the vendored deps cannot be dropped with a dependency marker the way win_arm64
does; instead this step removes the geopandas + Shapely `Requires-Dist` lines from the
built wheel's METADATA so `pip install pyramids-gis` on Alpine does not fetch geopandas
(and its transitive, musl-wheel-less pyogrio). These tests pin that the strip removes
exactly those two lines, keeps every other dependency, keeps the RECORD consistent, and
fails loudly when there is nothing to strip.
"""

import base64
import csv
import hashlib
import importlib.util
import io
import re
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STRIP_SCRIPT = REPO_ROOT / "ci" / "strip-vendored-deps-from-wheel.py"


def _load_strip_module():
    """Load the hyphenated ci strip script as an importable module."""
    spec = importlib.util.spec_from_file_location("strip_vendored_deps", STRIP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_row(path: str, data: bytes) -> list:
    """Return the RECORD row (path, sha256=<b64>, size) for `data`."""
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )
    return [path, f"sha256={digest}", str(len(data))]


def _write_wheel(path: Path, metadata: bytes) -> None:
    """Write a minimal but valid wheel carrying `metadata` and a matching RECORD."""
    dist_info = "pyramids_gis-0.0.0.dist-info"
    init = b"__version__ = '0.0.0'\n"
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    writer.writerow(_record_row(f"{dist_info}/METADATA", metadata))
    writer.writerow(_record_row("pyramids/__init__.py", init))
    writer.writerow([f"{dist_info}/RECORD", "", ""])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pyramids/__init__.py", init)
        zf.writestr(f"{dist_info}/METADATA", metadata)
        zf.writestr(f"{dist_info}/RECORD", record.getvalue().encode("utf-8"))


_MUSL_MARKER = "; sys_platform != 'win32' or platform_machine != 'ARM64'"
_METADATA = (
    "Metadata-Version: 2.1\nName: pyramids-gis\nVersion: 0.0.0\n"
    "Requires-Dist: numpy>=2.0.0\n"
    f"Requires-Dist: geopandas>=1.0.0 {_MUSL_MARKER}\n"
    f"Requires-Dist: Shapely>=2.1.0 {_MUSL_MARKER}\n"
    "Requires-Dist: cftime>=1.6.4\n"
    "Requires-Dist: pyproj>=3.7.0\n\nProject description.\n"
).encode("utf-8")


def test_strip_removes_the_vendored_deps(tmp_path):
    """The strip drops exactly the geopandas + Shapely + cftime Requires-Dist lines."""
    module = _load_strip_module()
    wheel = tmp_path / "pyramids_gis-0.0.0-cp312-cp312-musllinux_1_2_x86_64.whl"
    _write_wheel(wheel, _METADATA)
    removed = module.strip_wheel(wheel)
    assert removed == 3
    with zipfile.ZipFile(wheel) as zf:
        metadata = zf.read("pyramids_gis-0.0.0.dist-info/METADATA").decode("utf-8")
    assert "geopandas" not in metadata
    assert "Shapely" not in metadata
    assert "cftime" not in metadata


def test_strip_keeps_the_other_dependencies(tmp_path):
    """Dependencies other than the vendored ones survive the strip untouched."""
    module = _load_strip_module()
    wheel = tmp_path / "pyramids_gis-0.0.0-cp312-cp312-musllinux_1_2_x86_64.whl"
    _write_wheel(wheel, _METADATA)
    module.strip_wheel(wheel)
    with zipfile.ZipFile(wheel) as zf:
        metadata = zf.read("pyramids_gis-0.0.0.dist-info/METADATA").decode("utf-8")
    assert "Requires-Dist: numpy>=2.0.0" in metadata
    assert "Requires-Dist: pyproj>=3.7.0" in metadata


def test_strip_rewrites_the_record_hash(tmp_path):
    """The RECORD's METADATA row is refreshed to the stripped file's hash + size."""
    module = _load_strip_module()
    wheel = tmp_path / "pyramids_gis-0.0.0-cp312-cp312-musllinux_1_2_x86_64.whl"
    _write_wheel(wheel, _METADATA)
    module.strip_wheel(wheel)
    with zipfile.ZipFile(wheel) as zf:
        metadata = zf.read("pyramids_gis-0.0.0.dist-info/METADATA")
        record = zf.read("pyramids_gis-0.0.0.dist-info/RECORD").decode("utf-8")
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(metadata).digest())
        .rstrip(b"=")
        .decode()
    )
    row = next(
        r for r in csv.reader(io.StringIO(record)) if r and r[0].endswith("/METADATA")
    )
    assert row[1] == f"sha256={expected}"
    assert row[2] == str(len(metadata))


def test_strip_fails_when_nothing_to_remove(tmp_path):
    """A wheel with no geopandas/Shapely lines is a signal of drift, so the strip fails."""
    module = _load_strip_module()
    wheel = tmp_path / "pyramids_gis-0.0.0-cp312-cp312-musllinux_1_2_x86_64.whl"
    _write_wheel(
        wheel,
        b"Metadata-Version: 2.1\nName: pyramids-gis\nRequires-Dist: numpy>=2.0.0\n",
    )
    with pytest.raises(SystemExit):
        module.strip_wheel(wheel)


def test_strip_rejects_a_surviving_vendored_dep(tmp_path, monkeypatch):
    """An incomplete strip that leaves a vendored dep declared is a hard failure.

    The post-strip audit is independent of the removal regex, so even if the
    line matcher drifts and misses `cftime`, the surviving `Requires-Dist:
    cftime` is caught and the step fails loudly rather than shipping a wheel pip
    would try to satisfy from a nonexistent aarch64-musl cftime wheel.
    """
    module = _load_strip_module()
    # Narrow the removal regex so it matches geopandas/Shapely but never cftime,
    # simulating drift that leaves one vendored dep behind after removal.
    monkeypatch.setattr(
        module, "_NAME_RE", re.compile(r"^Requires-Dist:\s*([gGsS][A-Za-z0-9._-]*)")
    )
    wheel = tmp_path / "pyramids_gis-0.0.0-cp312-cp312-musllinux_1_2_x86_64.whl"
    _write_wheel(wheel, _METADATA)
    with pytest.raises(SystemExit, match="cftime"):
        module.strip_wheel(wheel)


def test_main_strips_every_wheel_in_a_directory(tmp_path, capsys):
    """main() strips each wheel found under a directory argument and reports it."""
    module = _load_strip_module()
    wheel = tmp_path / "pyramids_gis-0.0.0-cp312-cp312-musllinux_1_2_x86_64.whl"
    _write_wheel(wheel, _METADATA)
    module.main([str(tmp_path)])
    assert "stripped 3 vendored dep line(s)" in capsys.readouterr().out
    with zipfile.ZipFile(wheel) as zf:
        metadata = zf.read("pyramids_gis-0.0.0.dist-info/METADATA").decode("utf-8")
    assert "geopandas" not in metadata


def test_main_strips_a_wheel_file_argument(tmp_path, capsys):
    """main() accepts a .whl file path directly, not only a directory."""
    module = _load_strip_module()
    wheel = tmp_path / "pyramids_gis-0.0.0-cp312-cp312-musllinux_1_2_x86_64.whl"
    _write_wheel(wheel, _METADATA)
    module.main([str(wheel)])
    assert "stripped 3 vendored dep line(s)" in capsys.readouterr().out


def test_main_fails_when_no_wheels_found(tmp_path):
    """main() exits nonzero when the arguments name no wheels at all."""
    module = _load_strip_module()
    with pytest.raises(SystemExit, match="no wheels"):
        module.main([str(tmp_path)])


def test_iter_wheels_rejects_a_non_wheel_argument(tmp_path):
    """A path that is neither a directory nor a .whl is a hard error."""
    module = _load_strip_module()
    junk = tmp_path / "notes.txt"
    junk.write_text("not a wheel")
    with pytest.raises(SystemExit, match="not a wheel or directory"):
        list(module._iter_wheels([str(junk)]))
