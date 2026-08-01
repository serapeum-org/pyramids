"""CLI smoke tests for the processing subcommands (tools / tool / run)."""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.cli import main
from pyramids.feature import FeatureCollection
from pyramids.processing import Pipeline


@pytest.fixture()
def points_geojson(tmp_path):
    """Write a small point layer to GeoJSON and return its path.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written .geojson.
    """
    gdf = gpd.GeoDataFrame(
        {"elevation": [1.0, 2.0, 3.0, 4.0, 5.0]},
        geometry=[Point(0, 0), Point(4, 0), Point(0, 4), Point(4, 4), Point(2, 2)],
        crs="EPSG:4326",
    )
    path = tmp_path / "pts.geojson"
    FeatureCollection(gdf).to_file(str(path))
    return str(path)


class TestProcessingCli:
    """Tests for `pyramids tools|tool|run`."""

    def test_tools_lists_registry(self, capsys):
        """`pyramids tools` lists every registered tool.

        Test scenario:
            Output contains slope and interpolate_to_raster and exits 0.
        """
        rc = main(["tools"])
        out = capsys.readouterr().out
        assert rc == 0 and "slope" in out and "interpolate_to_raster" in out, out

    def test_tool_prints_schema(self, capsys):
        """`pyramids tool slope` prints the tool's parameter schema.

        Test scenario:
            Output includes the receiver header and the 'band' param; exit 0.
        """
        rc = main(["tool", "slope"])
        out = capsys.readouterr().out
        assert rc == 0 and "Dataset -> Array" in out and "band" in out, out

    def test_tool_unknown_returns_error(self, capsys):
        """`pyramids tool <unknown>` reports an error and exits non-zero.

        Test scenario:
            An unknown tool name prints 'unknown tool' and returns 1.
        """
        rc = main(["tool", "nope"])
        out = capsys.readouterr().out
        assert rc == 1 and "unknown tool" in out, out

    def test_run_end_to_end_writes_output(self, points_geojson, tmp_path):
        """`pyramids run` executes a pipeline YAML and writes an output.

        Args:
            points_geojson: input point layer path.
            tmp_path: pytest temp directory.

        Test scenario:
            An interpolate_to_raster pipeline over the input writes a .tif and
            exits 0.
        """
        yaml_path = tmp_path / "pipe.yaml"
        Pipeline([("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0})]).to_yaml(
            str(yaml_path)
        )
        out_dir = tmp_path / "out"
        rc = main(["run", str(yaml_path), "--inputs", points_geojson, "--out", str(out_dir)])
        assert rc == 0, "run should exit 0"
        assert list(out_dir.glob("*.tif")), list(out_dir.iterdir())

    def test_run_bad_input_skip_reports_failure(self, tmp_path, capsys):
        """`pyramids run` with a bad input under skip reports the failure, exits 1.

        Args:
            tmp_path: pytest temp directory.
            capsys: stdout capture.

        Test scenario:
            A nonexistent input is skipped-and-reported; exit code is 1.
        """
        yaml_path = tmp_path / "pipe.yaml"
        Pipeline([("slope", {})]).to_yaml(str(yaml_path))
        out_dir = tmp_path / "out"
        rc = main(
            ["run", str(yaml_path), "--inputs", "C:/no/such.tif", "--out", str(out_dir)]
        )
        out = capsys.readouterr().out
        assert rc == 1 and "FAILED" in out, out
