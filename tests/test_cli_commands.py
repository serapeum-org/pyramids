"""Smoke tests for the general-purpose CLI subcommands.

Each command is a thin wrapper over a library primitive; these tests pin the
wiring: exit codes, JSON output shapes, written artifacts, and the
one-line-error contract for expected user mistakes. The COG group has its own
suite in `tests/dataset/cog/test_cli.py`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def src_raster(tmp_path) -> str:
    """An 8x8 float32 ramp GeoTIFF on EPSG:4326 with nodata -9999.

    Returns:
        str: Path to the written raster.
    """
    path = str(tmp_path / "src.tif")
    Dataset.create_from_array(
        np.arange(64, dtype="float32").reshape(8, 8),
        top_left_corner=(0, 8), cell_size=1.0, epsg=4326, no_data_value=-9999.0,
    ).to_file(path)
    return path


class TestInfoCommand:
    """`pyramids info`."""

    def test_json_payload(self, src_raster, capsys):
        """--json emits parseable metadata with the key facts.

        Test scenario:
            epsg, shape, dtype, nodata, and bounds are all present and right.
        """
        assert main(["info", src_raster, "--json"]) == 0, "info must exit 0"
        payload = json.loads(capsys.readouterr().out)
        assert payload["epsg"] == 4326, f"epsg wrong: {payload['epsg']}"
        assert (payload["bands"], payload["rows"], payload["columns"]) == (1, 8, 8)
        assert payload["dtype"] == ["float32"], f"dtype wrong: {payload['dtype']}"
        assert payload["no_data_value"] == [-9999.0], "nodata wrong"
        assert payload["bounds"] == [0.0, 0.0, 8.0, 8.0], "bounds wrong"

    def test_plain_output_lists_keys(self, src_raster, capsys):
        """The human-readable form prints key: value lines."""
        assert main(["info", src_raster]) == 0
        out = capsys.readouterr().out
        assert "epsg: 4326" in out, f"missing epsg line in: {out}"

    def test_missing_file_one_line_error(self, tmp_path, capsys):
        """A missing path exits 1 with a one-line error (no traceback)."""
        rc = main(["info", str(tmp_path / "nope.tif")])
        err = capsys.readouterr().err
        assert rc == 1, "missing file must exit 1"
        assert err.startswith("error: "), f"unexpected stderr: {err}"
        assert "Traceback" not in err, "tracebacks must not leak to users"


class TestBoundsCommand:
    """`pyramids bounds`."""

    def test_native_bounds(self, src_raster, capsys):
        """The native bbox is printed space-separated."""
        assert main(["bounds", src_raster]) == 0
        assert capsys.readouterr().out.split() == ["0.0", "0.0", "8.0", "8.0"]

    def test_reprojected_bounds_json(self, src_raster, capsys):
        """--crs reprojects the corners; --json wraps them.

        Test scenario:
            4326 -> 3857: longitude 8 deg is ~890 km east.
        """
        assert main(["bounds", src_raster, "--crs", "EPSG:3857", "--json"]) == 0
        bounds = json.loads(capsys.readouterr().out)["bounds"]
        assert bounds[0] == pytest.approx(0.0, abs=1e-6), "min_x wrong"
        assert bounds[2] == pytest.approx(890_555.9, rel=1e-3), "max_x wrong"


class TestClipCommand:
    """`pyramids clip`."""

    def test_bbox_clip_writes_subset(self, src_raster, tmp_path, capsys):
        """--bbox crops to the requested extent."""
        out_path = str(tmp_path / "clip.tif")
        assert main(["clip", src_raster, out_path, "--bbox", "1", "1", "5", "5"]) == 0
        clipped = Dataset.read_file(out_path)
        assert (clipped.rows, clipped.columns) == (4, 4), "clip extent wrong"

    def test_bbox_and_vector_mutually_exclusive(self, src_raster, tmp_path, capsys):
        """Passing both --bbox and --vector is rejected by argparse."""
        with pytest.raises(SystemExit):
            main([
                "clip", src_raster, str(tmp_path / "o.tif"),
                "--bbox", "1", "1", "5", "5", "--vector", "mask.geojson",
            ])


class TestWarpCommand:
    """`pyramids warp`."""

    def test_warp_writes_target_crs(self, src_raster, tmp_path, capsys):
        """--crs reprojects and writes the output."""
        out_path = str(tmp_path / "warp.tif")
        assert main(["warp", src_raster, out_path, "--crs", "3857"]) == 0
        assert Dataset.read_file(out_path).epsg == 3857, "output CRS wrong"

    def test_invalid_resampling_one_line_error(self, src_raster, tmp_path, capsys):
        """An unknown resampling exits 1 with a clean message."""
        rc = main([
            "warp", src_raster, str(tmp_path / "w.tif"),
            "--crs", "3857", "--resampling", "sinc",
        ])
        assert rc == 1, "bad resampling must exit 1"
        assert "error: " in capsys.readouterr().err, "expected one-line error"


class TestMergeCommand:
    """`pyramids merge`."""

    def test_merge_two_tiles(self, src_raster, tmp_path, capsys):
        """Two overlapping tiles mosaic into one readable output."""
        second = str(tmp_path / "b.tif")
        Dataset.create_from_array(
            np.ones((8, 8), dtype="float32"),
            top_left_corner=(4, 8), cell_size=1.0, epsg=4326, no_data_value=-9999.0,
        ).to_file(second)
        out_path = str(tmp_path / "merged.tif")
        assert main(["merge", src_raster, second, out_path]) == 0
        merged = Dataset.read_file(out_path)
        assert merged.columns > 8, "mosaic must span both tiles"


class TestOverviewCommand:
    """`pyramids overview`."""

    def test_builds_requested_levels(self, src_raster, capsys):
        """--levels builds that many overview levels in place."""
        assert main(["overview", src_raster, "--levels", "2", "4"]) == 0
        assert Dataset.read_file(src_raster).overview_count[0] == 2, (
            "expected 2 overview levels"
        )


class TestSampleCommand:
    """`pyramids sample`."""

    def test_json_per_point_values(self, src_raster, capsys):
        """--json emits one row of band values per point.

        Test scenario:
            (0.5, 7.5) is cell (0, 0) = 0; (3.5, 4.5) is cell (3, 3) = 27.
        """
        assert main([
            "sample", src_raster, "--points", "0.5,7.5;3.5,4.5", "--json",
        ]) == 0
        values = json.loads(capsys.readouterr().out)["values"]
        assert values == [[0.0], [27.0]], f"per-point values wrong: {values}"

    def test_empty_points_rejected(self, src_raster, capsys):
        """An empty --points exits 1 with a clean message."""
        rc = main(["sample", src_raster, "--points", ";"])
        assert rc == 1, "empty points must exit 1"
        assert "at least one" in capsys.readouterr().err, "expected guidance"


class TestConvertCommand:
    """`pyramids convert`."""

    def test_extension_inferred_conversion(self, src_raster, tmp_path, capsys):
        """The output driver is inferred from the extension (.nc -> NetCDF)."""
        out_path = str(tmp_path / "out.nc")
        assert main(["convert", src_raster, out_path]) == 0
        converted = Dataset.read_file(out_path)
        np.testing.assert_array_equal(
            converted.read_array(), Dataset.read_file(src_raster).read_array(),
            err_msg="conversion must preserve values",
        )


class TestHelpSurface:
    """--help wiring for every new command."""

    @pytest.mark.parametrize(
        "command",
        ["info", "bounds", "clip", "warp", "merge", "overview", "sample", "convert"],
    )
    def test_help_exits_zero(self, command, capsys):
        """Each subcommand exposes --help (argparse exits 0).

        Args:
            command: Subcommand under test.
        """
        with pytest.raises(SystemExit) as exc:
            main([command, "--help"])
        assert exc.value.code == 0, f"{command} --help must exit 0"
        assert command in capsys.readouterr().out, "help text must name the command"
