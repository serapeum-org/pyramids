"""Tests for JSON safety of the `pyramids bounds --json` payload."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyramids import cli
from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def wgs84_raster(tmp_path: Path) -> str:
    """Create a small WGS84 raster on disk.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        str: Path to the created GeoTIFF.
    """
    path = str(tmp_path / "bounds.tif")
    Dataset.create_from_array(
        np.ones((8, 8), dtype="float32"),
        top_left_corner=(-10.0, 50.0),
        cell_size=0.5,
        epsg=4326,
        driver_type="GTiff",
        path=path,
    )
    return path


@pytest.fixture(scope="function")
def failing_transform(monkeypatch: pytest.MonkeyPatch):
    """Force the corner reprojection to return GDAL's failure sentinel.

    GDAL documents `HUGE_VAL` as the coordinate returned when a point cannot be
    transformed. Current PROJ extrapolates rather than failing for most
    out-of-domain inputs, so no real CRS pair reliably produces it — patch the
    transformer instead, which exercises the guard deterministically and on
    every platform.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    """

    class _Failing:
        def TransformPoints(self, points):
            """Return one non-finite corner and three ordinary ones.

            GDAL hands back 3-tuples (x, y, z), so mirror that shape rather than
            the 2-tuples the caller happens to index.
            """
            failed = (float("inf"), float("inf"), 0.0)
            return [failed] + [(1.0, 2.0, 0.0)] * (len(points) - 1)

    monkeypatch.setattr(cli.osr, "CoordinateTransformation", lambda *a, **k: _Failing())


class TestBoundsJson:
    """Tests for `_cmd_bounds` JSON output."""

    def test_payload_is_valid_json(
        self,
        wgs84_raster: str,
        capsys: pytest.CaptureFixture,
    ):
        """`bounds --json` emits a document a strict parser accepts.

        Test scenario:
            The four bounds are finite here, so the payload round-trips through
            `json.loads` unchanged.
        """
        main(["bounds", wgs84_raster, "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["bounds"]) == 4, (
            f"expected four bounds, got {payload['bounds']}"
        )

    def test_non_finite_corner_becomes_null(
        self,
        wgs84_raster: str,
        capsys: pytest.CaptureFixture,
        failing_transform,
    ):
        """Out-of-domain reprojected corners serialise as null, not `Infinity`.

        Test scenario:
            Reprojecting a global-ish WGS84 extent into a CRS with a limited
            domain can push a corner to HUGE_VAL. `json.dumps` would write that
            as a bare `Infinity`, which is not valid JSON — the guard maps it to
            null so `json.loads(..., parse_constant=...)` never sees a constant.
        """
        main(["bounds", wgs84_raster, "--json", "--crs", "EPSG:3857"])
        raw = capsys.readouterr().out

        def _reject(constant: str):
            raise AssertionError(f"payload contained the bare JSON constant {constant!r}: {raw!r}")

        payload = json.loads(raw, parse_constant=_reject)
        for value in payload["bounds"]:
            assert value is None or isinstance(value, (int, float)), (
                f"unexpected bound {value!r}"
            )

    def test_output_parses_with_a_strict_external_parser(
        self,
        wgs84_raster: str,
        capsys: pytest.CaptureFixture,
        failing_transform,
    ):
        """The emitted text is accepted by a strict, non-Python JSON reader.

        Test scenario:
            Python's own `json.loads` accepts `Infinity` by default, which is
            exactly why the bug went unnoticed; re-parse in a subprocess with
            constants rejected to prove the text is portable.
        """
        main(["bounds", wgs84_raster, "--json", "--crs", "EPSG:3857"])
        raw = capsys.readouterr().out.strip()
        code = (
            "import json,sys\n"
            "def reject(c):\n"
            "    raise SystemExit('bare constant: ' + c)\n"
            "json.loads(sys.stdin.read(), parse_constant=reject)\n"
            "print('ok')\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], input=raw, capture_output=True, text=True
        )
        assert completed.returncode == 0, (
            f"strict parse failed: {completed.stdout}{completed.stderr}"
        )
