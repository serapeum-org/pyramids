"""Unit tests for :mod:`pyramids.processing.runner`.

The serial runner, receiver-type dispatch, error policy, output writing, and the
parallel guardrails are covered deterministically here. Real process-pool
execution (``parallel=True`` over multiple files) is exercised by manual smoke
runs — it is kept out of the default suite to avoid per-worker spawn + GDAL
initialization overhead and Windows ``spawn`` flakiness.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from pyramids.dataset import Dataset
from pyramids.processing import Pipeline, run


@pytest.fixture(scope="module")
def points_fc():
    """A five-point FeatureCollection with a numeric 'elevation' column.

    Returns:
        FeatureCollection: EPSG:4326 points suitable for interpolate_to_raster.
    """
    from pyramids.feature import FeatureCollection

    gdf = gpd.GeoDataFrame(
        {"elevation": [1.0, 2.0, 3.0, 4.0, 5.0]},
        geometry=[Point(0, 0), Point(4, 0), Point(0, 4), Point(4, 4), Point(2, 2)],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


class TestRun:
    """Tests for run() — serial execution, dispatch, policy, and output."""

    def test_cross_receiver_pipeline(self, points_fc):
        """A FeatureCollection->Dataset->Array chain runs end-to-end.

        Test scenario:
            interpolate_to_raster (FC->Dataset) then slope (Dataset->Array) yields
            one array output with no failures.
        """
        pipe = Pipeline(
            [("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0}), ("slope", {})]
        )
        result = run(pipe, points_fc, on_error="raise")
        assert len(result.outputs) == 1 and result.ok, result.failures
        assert isinstance(result.outputs[0], Dataset), type(result.outputs[0])

    def test_wrong_receiver_is_collected(self, points_fc):
        """A Dataset op applied to a FeatureCollection fails with a clear error.

        Test scenario:
            slope on a FeatureCollection raises TypeError, collected under skip.
        """
        result = run(Pipeline([("slope", {})]), points_fc, on_error="skip")
        assert len(result.failures) == 1, result.outputs
        source, exc = result.failures[0]
        assert isinstance(exc, TypeError) and "expects a Dataset" in str(exc), exc

    def test_array_op_output_is_materialized_and_chainable(self, points_fc):
        """A terminal array op is materialized to a Dataset, so it can be chained.

        Test scenario:
            interpolate -> slope -> aspect: slope's array is re-wrapped into a
            georeferenced Dataset, so aspect runs on it and the run succeeds.
        """
        pipe = Pipeline(
            [
                ("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0}),
                ("slope", {}),
                ("aspect", {}),
            ]
        )
        result = run(pipe, points_fc, on_error="raise")
        assert result.ok and isinstance(result.outputs[0], Dataset), result.failures

    def test_terminal_array_op_writes_tif(self, points_fc, tmp_path):
        """A pipeline ending in a terminal array op writes a georeferenced .tif.

        Args:
            points_fc: point-collection fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            interpolate -> slope with out=dir writes output_0.tif (regression for
            the array-output-not-writable bug).
        """
        pipe = Pipeline(
            [("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0}), ("slope", {})]
        )
        result = run(pipe, points_fc, out=str(tmp_path), on_error="raise")
        assert result.ok, result.failures
        assert (tmp_path / "output_0.tif").exists(), list(tmp_path.iterdir())

    def test_error_policy_skip_collects(self, points_fc):
        """A bad input under skip yields zero outputs and one failure.

        Test scenario:
            A nonexistent raster path is collected rather than raised.
        """
        result = run(Pipeline([("slope", {})]), ["C:/no/such/file.tif"], on_error="skip")
        assert len(result.outputs) == 0 and len(result.failures) == 1, result

    def test_error_policy_raise_propagates(self):
        """A bad input under raise propagates the exception.

        Test scenario:
            A nonexistent path raises rather than being collected.
        """
        with pytest.raises(Exception):
            run(Pipeline([("slope", {})]), ["C:/no/such/file.tif"], on_error="raise")

    def test_invalid_on_error_raises(self, points_fc):
        """An invalid on_error policy is rejected up front.

        Test scenario:
            on_error='bogus' raises ValueError before any work.
        """
        with pytest.raises(ValueError, match="on_error must be"):
            run(Pipeline([("slope", {})]), points_fc, on_error="bogus")

    def test_out_writes_file(self, points_fc, tmp_path):
        """With out set, a Dataset output is written as a .tif.

        Args:
            points_fc: point-collection fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            interpolate_to_raster with out=dir writes output_0.tif and still
            returns the in-memory Dataset in serial mode.
        """
        pipe = Pipeline([("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0})])
        result = run(pipe, points_fc, out=str(tmp_path))
        assert isinstance(result.outputs[0], Dataset), type(result.outputs[0])
        assert (tmp_path / "output_0.tif").exists(), list(tmp_path.iterdir())

    def test_glob_inputs(self, points_fc, tmp_path):
        """A glob input is expanded and each match is processed.

        Args:
            points_fc: point-collection fixture (written twice as geojson).
            tmp_path: pytest temp directory.

        Test scenario:
            Two point files matched by a glob produce two Dataset outputs.
        """
        for name in ("a.geojson", "b.geojson"):
            points_fc.to_file(str(tmp_path / name))
        pipe = Pipeline([("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0})])
        result = run(pipe, str(tmp_path / "*.geojson"), on_error="raise")
        assert len(result.outputs) == 2, [type(o) for o in result.outputs]

    def test_parallel_requires_out(self, points_fc):
        """parallel=True without an out directory is rejected.

        Test scenario:
            run(..., parallel=True, out=None) raises ValueError.
        """
        with pytest.raises(ValueError, match="requires an 'out' directory"):
            run(Pipeline([("slope", {})]), ["a.tif"], parallel=True, out=None)

    def test_parallel_rejects_non_path_inputs(self, points_fc, tmp_path):
        """parallel=True with an in-memory object input is rejected.

        Args:
            points_fc: an in-memory FeatureCollection (not a path).
            tmp_path: a valid out directory.

        Test scenario:
            Passing a FeatureCollection under parallel raises ValueError naming the
            GDAL-handle constraint.
        """
        with pytest.raises(ValueError, match="file-path inputs"):
            run(Pipeline([("slope", {})]), [points_fc], parallel=True, out=str(tmp_path))
