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
        top_left_corner=(0, 8),
        cell_size=1.0,
        epsg=4326,
        no_data_value=-9999.0,
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
        assert payload["no_data_value"] == pytest.approx([-9999.0]), "nodata wrong"
        assert payload["bounds"] == pytest.approx([0.0, 0.0, 8.0, 8.0]), "bounds wrong"

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
            main(
                [
                    "clip",
                    src_raster,
                    str(tmp_path / "o.tif"),
                    "--bbox",
                    "1",
                    "1",
                    "5",
                    "5",
                    "--vector",
                    "mask.geojson",
                ]
            )


class TestWarpCommand:
    """`pyramids warp`."""

    def test_warp_writes_target_crs(self, src_raster, tmp_path, capsys):
        """--crs reprojects and writes the output."""
        out_path = str(tmp_path / "warp.tif")
        assert main(["warp", src_raster, out_path, "--crs", "3857"]) == 0
        assert Dataset.read_file(out_path).epsg == 3857, "output CRS wrong"

    def test_invalid_resampling_one_line_error(self, src_raster, tmp_path, capsys):
        """An unknown resampling exits 1 with a clean message."""
        rc = main(
            [
                "warp",
                src_raster,
                str(tmp_path / "w.tif"),
                "--crs",
                "3857",
                "--resampling",
                "sinc",
            ]
        )
        assert rc == 1, "bad resampling must exit 1"
        assert "error: " in capsys.readouterr().err, "expected one-line error"


class TestMergeCommand:
    """`pyramids merge`."""

    def test_merge_two_tiles(self, src_raster, tmp_path, capsys):
        """Two overlapping tiles mosaic into one readable output."""
        second = str(tmp_path / "b.tif")
        Dataset.create_from_array(
            np.ones((8, 8), dtype="float32"),
            top_left_corner=(4, 8),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        ).to_file(second)
        out_path = str(tmp_path / "merged.tif")
        assert main(["merge", src_raster, second, out_path]) == 0
        merged = Dataset.read_file(out_path)
        assert merged.columns > 8, "mosaic must span both tiles"

    def test_single_input_rejected(self, src_raster, tmp_path, capsys):
        """One source raster exits 1 with a clean message.

        Test scenario:
            `merge a.tif out.tif` parses as one input + the output; the
            command must reject it instead of silently copying the file.
        """
        rc = main(["merge", src_raster, str(tmp_path / "m.tif")])
        assert rc == 1, "single-input merge must exit 1"
        assert "at least two" in capsys.readouterr().err, "expected guidance"


class TestOverviewCommand:
    """`pyramids overview`."""

    def test_builds_requested_levels(self, src_raster, capsys):
        """--levels builds that many overview levels in place."""
        assert main(["overview", src_raster, "--levels", "2", "4"]) == 0
        assert (
            Dataset.read_file(src_raster).overview_count[0] == 2
        ), "expected 2 overview levels"


class TestSampleCommand:
    """`pyramids sample`."""

    def test_json_per_point_values(self, src_raster, capsys):
        """--json emits one row of band values per point.

        Test scenario:
            (0.5, 7.5) is cell (0, 0) = 0; (3.5, 4.5) is cell (3, 3) = 27.
        """
        assert (
            main(
                [
                    "sample",
                    src_raster,
                    "--points",
                    "0.5,7.5;3.5,4.5",
                    "--json",
                ]
            )
            == 0
        )
        values = json.loads(capsys.readouterr().out)["values"]
        expected = [pytest.approx([0.0]), pytest.approx([27.0])]
        assert values == expected, f"per-point values wrong: {values}"

    def test_empty_points_rejected(self, src_raster, capsys):
        """An empty --points exits 1 with a clean message."""
        rc = main(["sample", src_raster, "--points", ";"])
        assert rc == 1, "empty points must exit 1"
        assert "at least one" in capsys.readouterr().err, "expected guidance"

    def test_malformed_point_rejected(self, src_raster, capsys):
        """A point that is not 'x,y' exits 1 naming the bad chunk."""
        rc = main(["sample", src_raster, "--points", "1,2,3"])
        assert rc == 1, "malformed point must exit 1"
        assert "expected 'x,y'" in capsys.readouterr().err, "expected format hint"

    def test_non_numeric_point_rejected(self, src_raster, capsys):
        """A non-numeric coordinate exits 1 naming the bad chunk."""
        rc = main(["sample", src_raster, "--points", "a,2"])
        err = capsys.readouterr().err
        assert rc == 1, "non-numeric point must exit 1"
        assert "'a,2'" in err, f"error must name the bad chunk: {err}"
        assert "numeric" in err, f"expected a numeric hint: {err}"

    def test_out_of_bounds_point_is_json_null(self, tmp_path, capsys):
        """An outside point emits JSON null, keeping the payload parseable.

        Test scenario:
            On a raster without nodata, (100, 100) is outside the 8x8
            extent and samples as NaN, which `json.dumps` would emit as
            bare `NaN` (invalid JSON) — the CLI must map it to null.
        """
        path = str(tmp_path / "no_nodata.tif")
        Dataset.create_from_array(
            np.arange(64, dtype="float32").reshape(8, 8),
            top_left_corner=(0, 8),
            cell_size=1.0,
            epsg=4326,
            no_data_value=None,
        ).to_file(path)
        assert main(["sample", path, "--points", "100,100", "--json"]) == 0
        values = json.loads(capsys.readouterr().out)["values"]
        assert values == [[None]], f"out-of-bounds must be null: {values}"


class TestConvertCommand:
    """`pyramids convert`."""

    def test_extension_inferred_conversion(self, src_raster, tmp_path, capsys):
        """The output driver is inferred from the extension (.nc -> NetCDF)."""
        out_path = str(tmp_path / "out.nc")
        assert main(["convert", src_raster, out_path]) == 0
        converted = Dataset.read_file(out_path)
        np.testing.assert_array_equal(
            converted.read_array(),
            Dataset.read_file(src_raster).read_array(),
            err_msg="conversion must preserve values",
        )

    def test_explicit_driver_overrides_extension(self, src_raster, tmp_path, capsys):
        """--driver wins over the output extension.

        Test scenario:
            `.nc` would infer NetCDF; `--driver geotiff` must produce a
            GTiff file regardless.
        """
        out_path = str(tmp_path / "tiff_in_disguise.nc")
        assert main(["convert", src_raster, out_path, "--driver", "geotiff"]) == 0
        converted = Dataset.read_file(out_path)
        driver = converted.raster.GetDriver().ShortName
        assert driver == "GTiff", f"--driver geotiff ignored; wrote {driver}"

    def test_unknown_driver_one_line_error(self, src_raster, tmp_path, capsys):
        """An unknown --driver exits 1 with a clean message."""
        rc = main(
            [
                "convert",
                src_raster,
                str(tmp_path / "o.tif"),
                "--driver",
                "bogus",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1, "unknown driver must exit 1"
        assert err.startswith("error: "), f"unexpected stderr: {err}"
        assert "Traceback" not in err, "tracebacks must not leak to users"

    def test_unknown_extension_one_line_error(self, src_raster, tmp_path, capsys):
        """An extension absent from the catalog exits 1 with a clean message."""
        rc = main(["convert", src_raster, str(tmp_path / "o.unknown")])
        err = capsys.readouterr().err
        assert rc == 1, "unknown extension must exit 1"
        assert err.startswith("error: "), f"unexpected stderr: {err}"
        assert "Traceback" not in err, "tracebacks must not leak to users"


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
