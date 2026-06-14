"""Smoke tests for the general-purpose CLI subcommands.

Each command is a thin wrapper over a library primitive; these tests pin the
wiring: exit codes, JSON output shapes, written artifacts, and the
one-line-error contract for expected user mistakes. The COG group has its own
suite in `tests/dataset/cog/test_cli.py`.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from osgeo import gdal

from pyramids.cli import main
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


def _crsless_raster(tmp_path) -> str:
    """Write a tiny GeoTIFF with a geotransform but no CRS.

    Returns:
        str: Path to the CRS-less raster.
    """
    path = str(tmp_path / "no_crs.tif")
    out = gdal.GetDriverByName("GTiff").Create(path, 8, 8, 1, gdal.GDT_Float32)
    out.SetGeoTransform((0.0, 1.0, 0.0, 8.0, 0.0, -1.0))
    out.GetRasterBand(1).WriteArray(np.arange(64, dtype="float32").reshape(8, 8))
    out.FlushCache()
    out = None
    return path


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

    def test_invalid_crs_one_line_error(self, src_raster, capsys):
        """An uninterpretable --crs exits 1 with a clean message."""
        rc = main(["bounds", src_raster, "--crs", "not-a-crs"])
        err = capsys.readouterr().err
        assert rc == 1, "bad CRS must exit 1"
        assert err.startswith("error: "), f"unexpected stderr: {err}"
        assert "Traceback" not in err, "tracebacks must not leak to users"

    def test_crs_on_crsless_raster_clear_error(self, tmp_path, capsys):
        """bounds --crs on a raster without a CRS names the real cause (L10)."""
        path = _crsless_raster(tmp_path)
        rc = main(["bounds", path, "--crs", "EPSG:3857"])
        err = capsys.readouterr().err
        assert rc == 1, "reprojecting bounds of a CRS-less raster must exit 1"
        assert "source CRS" in err, f"error should name the missing source CRS: {err}"


class TestClipCommand:
    """`pyramids clip`."""

    def test_bbox_clip_writes_subset(self, src_raster, tmp_path, capsys):
        """--bbox crops to the requested extent."""
        out_path = str(tmp_path / "clip.tif")
        assert main(["clip", src_raster, out_path, "--bbox", "1", "1", "5", "5"]) == 0
        clipped = Dataset.read_file(out_path)
        assert (clipped.rows, clipped.columns) == (4, 4), "clip extent wrong"

    def test_vector_clip_writes_subset(self, src_raster, tmp_path, capsys):
        """--vector crops to the mask polygon's extent.

        Test scenario:
            A rectangular GeoJSON mask covering (1, 1, 5, 5) yields the
            same 4x4 subset as the equivalent --bbox clip.
        """
        mask_path = str(tmp_path / "mask.geojson")
        FeatureCollection.from_bbox((1.0, 1.0, 5.0, 5.0), epsg=4326).to_file(mask_path)
        out_path = str(tmp_path / "clip_vec.tif")
        assert main(["clip", src_raster, out_path, "--vector", mask_path]) == 0
        clipped = Dataset.read_file(out_path)
        assert (clipped.rows, clipped.columns) == (4, 4), "vector clip extent wrong"

    def test_bbox_on_crsless_raster_clear_error(self, tmp_path, capsys):
        """clip --bbox on a CRS-less raster gives a clear error, not a low-level one (L10)."""
        path = _crsless_raster(tmp_path)
        out = str(tmp_path / "out.tif")
        rc = main(["clip", path, out, "--bbox", "1", "1", "5", "5"])
        err = capsys.readouterr().err
        assert rc == 1, "clip on a CRS-less raster must exit 1"
        assert "has none" in err, f"error should name the missing CRS: {err}"

    def test_bbox_disjoint_from_raster_clear_error(self, src_raster, tmp_path, capsys):
        """clip --bbox outside the raster extent gives a clear error, not an IndexError (L4)."""
        out = str(tmp_path / "out.tif")
        rc = main(["clip", src_raster, out, "--bbox", "100", "100", "110", "110"])
        err = capsys.readouterr().err
        assert rc == 1, "a disjoint clip bbox must exit 1"
        assert (
            "does not intersect" in err
        ), f"error should name the disjoint bbox: {err}"
        assert not os.path.exists(
            out
        ), "no output should be written for a disjoint clip"

    def test_vector_disjoint_from_raster_clear_error(
        self, src_raster, tmp_path, capsys
    ):
        """clip --vector with a mask outside the raster extent gives a clear error (L3)."""
        mask_path = str(tmp_path / "far_mask.geojson")
        FeatureCollection.from_bbox((100.0, 100.0, 110.0, 110.0), epsg=4326).to_file(
            mask_path
        )
        out = str(tmp_path / "out.tif")
        rc = main(["clip", src_raster, out, "--vector", mask_path])
        err = capsys.readouterr().err
        assert rc == 1, "a disjoint vector mask must exit 1"
        assert (
            "does not intersect" in err
        ), f"error should name the disjoint mask: {err}"
        assert not os.path.exists(
            out
        ), "no output should be written for a disjoint clip"

    def test_refuses_to_overwrite_without_flag(self, src_raster, tmp_path, capsys):
        """clip refuses to clobber an existing output unless --overwrite (N5)."""
        out = str(tmp_path / "exists.tif")
        assert main(["clip", src_raster, out, "--bbox", "1", "1", "5", "5"]) == 0
        rc = main(["clip", src_raster, out, "--bbox", "1", "1", "5", "5"])
        err = capsys.readouterr().err
        assert rc == 1, "a second write without --overwrite must be refused"
        assert "already exists" in err, f"error should mention the existing file: {err}"
        assert (
            main(["clip", src_raster, out, "--bbox", "1", "1", "5", "5", "--overwrite"])
            == 0
        ), "--overwrite must allow replacing the output"

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

    def test_nearest_neighbor_alias_accepted(self, src_raster, capsys):
        """The warp-family 'nearest neighbor' spelling is accepted here too (L9)."""
        rc = main(
            [
                "overview",
                src_raster,
                "--levels",
                "2",
                "--resampling",
                "nearest neighbor",
            ]
        )
        assert rc == 0, f"'nearest neighbor' should be accepted, got rc={rc}"

    def test_non_power_of_two_levels_rejected(self, src_raster, capsys):
        """A non-power-of-two --levels is rejected up front with a clear error (L6)."""
        rc = main(["overview", src_raster, "--levels", "3", "5"])
        err = capsys.readouterr().err
        assert rc == 1, "non-power-of-two levels must exit 1"
        assert "power-of-two" in err, f"error should name the constraint: {err}"
        assert "[3, 5]" in err, f"error should list the offending levels: {err}"


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


class TestUnexpectedErrors:
    """An internal error not in main()'s expected set still gets one line, not a traceback (L5)."""

    def test_unexpected_exception_one_line(self, src_raster, monkeypatch, capsys):
        """A non-listed exception (e.g. KeyError) exits 1 with a one-line message."""

        def boom(_args):
            raise KeyError("internal")

        monkeypatch.setattr("pyramids.cli._cmd_raster_info", boom)
        rc = main(["info", src_raster])
        err = capsys.readouterr().err
        assert rc == 1, "an unexpected internal error must exit 1"
        assert err.startswith("error: unexpected failure"), f"unexpected stderr: {err}"
        assert "Traceback" not in err, "tracebacks must not leak to users"

    def test_debug_env_reraises_for_stack(self, src_raster, monkeypatch):
        """With PYRAMIDS_DEBUG set, the original exception propagates for the full stack."""

        def boom(_args):
            raise KeyError("internal")

        monkeypatch.setattr("pyramids.cli._cmd_raster_info", boom)
        monkeypatch.setenv("PYRAMIDS_DEBUG", "1")
        with pytest.raises(KeyError):
            main(["info", src_raster])


class TestGeoreferenceCLI:
    """Tests for `pyramids georeference` and `pyramids orthorectify`."""

    def test_georeference_writes_output(self, src_raster, tmp_path):
        """georeference warps the input from --gcp points into a 4326 raster.

        Test scenario:
            Four corner GCPs (10-11E, 49-50N) write a georeferenced GeoTIFF.
        """
        out = str(tmp_path / "geo.tif")
        rc = main(
            [
                "georeference", src_raster, out,
                "--gcp", "0", "0", "10", "50",
                "--gcp", "8", "0", "11", "50",
                "--gcp", "0", "8", "10", "49",
                "--gcp", "8", "8", "11", "49",
                "--gcp-crs", "4326",
            ]
        )
        assert rc == 0, "georeference must exit 0"
        assert os.path.exists(out), "output raster must be written"
        assert Dataset.read_file(out).epsg == 4326

    def test_georeference_refuses_existing(self, src_raster, tmp_path):
        """Without --overwrite, an existing output is refused (exit 1).

        Test scenario:
            georeference onto an existing path returns 1 and writes nothing new.
        """
        out = str(tmp_path / "exists.tif")
        Dataset.create_from_array(
            np.ones((2, 2), "float32"), top_left_corner=(0, 2), cell_size=1.0
        ).to_file(out)
        rc = main(
            [
                "georeference", src_raster, out,
                "--gcp", "0", "0", "10", "50",
                "--gcp-crs", "4326",
            ]
        )
        assert rc == 1, "must refuse an existing output without --overwrite"

    def test_orthorectify_without_rpc_errors_cleanly(self, src_raster, tmp_path):
        """orthorectify on a raster with no RPC metadata exits 1 (clean error).

        Test scenario:
            A plain raster has no RPCs; the command reports the error and exits 1
            rather than crashing.
        """
        out = str(tmp_path / "ortho.tif")
        rc = main(["orthorectify", src_raster, out, "--rpc-height", "100"])
        assert rc == 1, "must exit 1 when the input has no RPC metadata"
        assert not os.path.exists(out), "no output on a failed orthorectify"


class TestEditInfo:
    """Tests for `pyramids edit-info`."""

    def test_sets_crs_and_nodata(self, src_raster):
        """edit-info rewrites the CRS and no-data value in place.

        Test scenario:
            --crs 3857 --nodata 0 on a 4326/-9999 raster; re-read reflects both.
        """
        rc = main(["edit-info", src_raster, "--crs", "3857", "--nodata", "0"])
        assert rc == 0, "edit-info must exit 0"
        ds = Dataset.read_file(src_raster)
        assert ds.epsg == 3857
        assert ds.no_data_value[0] == 0

    def test_sets_tag(self, src_raster):
        """edit-info writes a metadata tag.

        Test scenario:
            --tag AREA=test sets the AREA metadata item.
        """
        rc = main(["edit-info", src_raster, "--tag", "AREA=test"])
        assert rc == 0
        assert Dataset.read_file(src_raster).raster.GetMetadataItem("AREA") == "test"

    def test_no_flags_prints_notice(self, src_raster, capsys):
        """edit-info with no edit flags prints a notice and exits 0.

        Test scenario:
            A bare edit-info call no-ops with a helpful message.
        """
        rc = main(["edit-info", src_raster])
        assert rc == 0
        assert "no edits" in capsys.readouterr().out


class TestCalc:
    """Tests for `pyramids calc` (safe band-expression evaluation)."""

    def _band(self, tmp_path, name, value):
        """Write a 2x2 constant-value GeoTIFF and return its path."""
        path = str(tmp_path / name)
        Dataset.create_from_array(
            np.full((2, 2), value, "float32"), top_left_corner=(0, 2), cell_size=1.0
        ).to_file(path)
        return path

    def test_ndvi_correctness(self, tmp_path):
        """calc evaluates (A - B) / (A + B) element-wise.

        Test scenario:
            A=4, B=2 -> NDVI 2/6 = 0.3333 across the output.
        """
        a = self._band(tmp_path, "a.tif", 4.0)
        b = self._band(tmp_path, "b.tif", 2.0)
        out = str(tmp_path / "ndvi.tif")
        rc = main(["calc", "(A - B) / (A + B)", a, b, out])
        assert rc == 0, "calc must exit 0"
        result = np.asarray(Dataset.read_file(out).read_array())
        assert np.allclose(result, (4.0 - 2.0) / (4.0 + 2.0))

    def test_np_where_allowed(self, tmp_path):
        """A whitelisted np.where call is evaluated.

        Test scenario:
            np.where(A > 3, 1, 0) on A=4 yields all ones.
        """
        a = self._band(tmp_path, "a.tif", 4.0)
        out = str(tmp_path / "w.tif")
        rc = main(["calc", "np.where(A > 3, 1, 0)", a, out])
        assert rc == 0
        assert np.allclose(np.asarray(Dataset.read_file(out).read_array()), 1)

    def test_disallowed_expression_rejected(self, src_raster, tmp_path):
        """A hostile expression is rejected and writes nothing.

        Test scenario:
            __import__('os') exits 1 (ValueError) and creates no output.
        """
        out = str(tmp_path / "evil.tif")
        rc = main(["calc", "__import__('os')", src_raster, out])
        assert rc == 1, "disallowed expression must exit 1"
        assert not os.path.exists(out), "nothing is written on a rejected expression"

    def test_dtype_flag(self, src_raster, tmp_path):
        """--dtype casts the result.

        Test scenario:
            A * 2 with --dtype float64 writes a float64 raster.
        """
        out = str(tmp_path / "d.tif")
        rc = main(["calc", "A * 2", src_raster, out, "--dtype", "float64"])
        assert rc == 0
        assert np.asarray(Dataset.read_file(out).read_array()).dtype == np.float64

    def test_sets_crs_from_authority_string(self, src_raster):
        """edit-info accepts EPSG:NNNN / PROJ4, not just a bare integer (review M1).

        Test scenario:
            --crs "EPSG:3857" sets the CRS without a cryptic error.
        """
        rc = main(["edit-info", src_raster, "--crs", "EPSG:3857"])
        assert rc == 0, "authority-string CRS must be accepted"
        assert Dataset.read_file(src_raster).epsg == 3857

    def test_invalid_crs_leaves_file_unchanged(self, src_raster):
        """An invalid --crs fails cleanly and does not mutate the file (review M1).

        Test scenario:
            --crs not-a-crs exits 1 and the EPSG stays 4326 (no partial write).
        """
        rc = main(["edit-info", src_raster, "--crs", "not-a-crs"])
        assert rc == 1, "invalid CRS must exit 1"
        assert Dataset.read_file(src_raster).epsg == 4326, "no partial write"


class TestShapesRasterizeCLI:
    """Tests for `pyramids shapes` and `pyramids rasterize`."""

    def test_shapes_writes_vector(self, src_raster, tmp_path):
        """shapes vectorizes the raster into a readable vector file.

        Test scenario:
            shapes src.tif out.geojson writes per-cell polygons (64 cells).
        """
        out = str(tmp_path / "shapes.geojson")
        rc = main(["shapes", src_raster, out])
        assert rc == 0, "shapes must exit 0"
        assert os.path.exists(out)
        assert np.all(np.isfinite(FeatureCollection.read_file(out).total_bounds))

    def test_rasterize_round_trips(self, src_raster, tmp_path):
        """rasterize burns a vector (from shapes) back into a raster.

        Test scenario:
            shapes -> vector, then rasterize --cell-size 1 -> a raster file.
        """
        vector = str(tmp_path / "v.geojson")
        assert main(["shapes", src_raster, vector]) == 0
        out = str(tmp_path / "r.tif")
        rc = main(["rasterize", vector, out, "--cell-size", "1"])
        assert rc == 0, "rasterize must exit 0"
        assert os.path.exists(out)
        assert Dataset.read_file(out).cell_size == 1

    def test_rasterize_without_grid_errors(self, src_raster, tmp_path):
        """rasterize without --cell-size or --like exits 1.

        Test scenario:
            A missing output grid is a clean user error.
        """
        vector = str(tmp_path / "v.geojson")
        assert main(["shapes", src_raster, vector]) == 0
        rc = main(["rasterize", vector, str(tmp_path / "r.tif")])
        assert rc == 1, "must exit 1 without --cell-size/--like"
