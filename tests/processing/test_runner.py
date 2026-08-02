"""Unit tests for :mod:`pyramids.processing.runner`.

The serial runner, input-type dispatch, error policy, output writing, and the
parallel guardrails are covered deterministically here. Real process-pool
execution (``parallel=True`` over multiple files) is covered by the ``slow``-marked
tests (``test_parallel_run_executes_and_returns_paths``,
``test_parallel_skip_collects_failure_in_input_order``, and
``test_parallel_raise_propagates``): they run in the default suite and are
deselectable with ``-m "not slow"``. These are the tests whose stability depends on
OS process spawning + per-worker GDAL init.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

import pyramids.processing.registry as reg
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.processing import Pipeline, run
from pyramids.processing.schema import Parameter, ToolMetadata


@pytest.fixture(scope="module")
def points_fc():
    """A five-point FeatureCollection with a numeric 'elevation' column.

    Returns:
        FeatureCollection: EPSG:4326 points suitable for interpolate_to_raster.
    """
    gdf = gpd.GeoDataFrame(
        {"elevation": [1.0, 2.0, 3.0, 4.0, 5.0]},
        geometry=[Point(0, 0), Point(4, 0), Point(0, 4), Point(4, 4), Point(2, 2)],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


@pytest.fixture(scope="module")
def raster_ds():
    """An 8x8 single-band EPSG:4326 raster for the Dataset-input tools.

    Returns:
        Dataset: a small in-memory float raster.
    """
    arr = np.arange(64, dtype="float32").reshape(1, 8, 8)
    return Dataset.create_from_array(arr, geo=(0, 1, 0, 8, 0, -1), epsg=4326)


class TestRun:
    """Tests for run() — serial execution, dispatch, policy, and output."""

    def test_cross_receiver_pipeline(self, points_fc):
        """A FeatureCollection->Dataset->Array chain runs end-to-end.

        Test scenario:
            interpolate_to_raster (FC->Dataset) then slope (Dataset->Array) yields
            one array output with no failures.
        """
        pipe = Pipeline(
            [
                ("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0}),
                ("slope", {}),
            ]
        )
        result = run(pipe, points_fc, on_error="raise")
        assert len(result.outputs) == 1, result.failures
        assert result.ok, result.failures
        assert isinstance(result.outputs[0], Dataset), type(result.outputs[0])

    def test_wrong_receiver_is_collected(self, points_fc):
        """A Dataset op applied to a FeatureCollection fails with a clear error.

        Test scenario:
            slope on a FeatureCollection raises TypeError, collected under skip.
        """
        result = run(Pipeline([("slope", {})]), points_fc, on_error="skip")
        assert len(result.failures) == 1, result.outputs
        source, exc = result.failures[0]
        assert isinstance(exc, TypeError), exc
        assert "expects a Dataset" in str(exc), exc

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
        assert result.ok, result.failures
        assert isinstance(result.outputs[0], Dataset), result.failures

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
            [
                ("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0}),
                ("slope", {}),
            ]
        )
        result = run(pipe, points_fc, out=str(tmp_path), on_error="raise")
        assert result.ok, result.failures
        assert (tmp_path / "output_0.tif").exists(), list(tmp_path.iterdir())

    def test_error_policy_skip_collects(self, points_fc):
        """A bad input under skip yields zero outputs and one failure.

        Test scenario:
            A nonexistent raster path is collected rather than raised.
        """
        result = run(
            Pipeline([("slope", {})]), ["C:/no/such/file.tif"], on_error="skip"
        )
        assert len(result.outputs) == 0, result.outputs
        assert len(result.failures) == 1, result.failures

    def test_error_policy_raise_propagates(self, points_fc):
        """A per-item error under raise propagates rather than being collected.

        Test scenario:
            A Dataset op on a FeatureCollection raises TypeError under raise.
        """
        pipe = Pipeline([("slope", {})])
        with pytest.raises(TypeError):
            run(pipe, points_fc, on_error="raise")

    def test_invalid_on_error_raises(self, points_fc):
        """An invalid on_error policy is rejected up front.

        Test scenario:
            on_error='bogus' raises ValueError before any work.
        """
        pipe = Pipeline([("slope", {})])
        with pytest.raises(ValueError, match="on_error must be"):
            run(pipe, points_fc, on_error="bogus")

    def test_out_writes_file(self, points_fc, tmp_path):
        """With out set, a Dataset output is written as a .tif.

        Args:
            points_fc: point-collection fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            interpolate_to_raster with out=dir writes output_0.tif and still
            returns the in-memory Dataset in serial mode.
        """
        pipe = Pipeline(
            [("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0})]
        )
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
        pipe = Pipeline(
            [("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0})]
        )
        result = run(pipe, str(tmp_path / "*.geojson"), on_error="raise")
        assert len(result.outputs) == 2, [type(o) for o in result.outputs]

    def test_datasetcollection_like_input(self):
        """An object exposing `.datasets` is treated as a batch of Datasets.

        Test scenario:
            A collection-like stub (duck-typed `.datasets`) runs slope over each
            member, yielding one materialized Dataset per member.
        """
        arr = np.arange(64, dtype="float32").reshape(1, 8, 8)
        members = [
            Dataset.create_from_array(arr, geo=(0, 1, 0, 8, 0, -1), epsg=4326),
            Dataset.create_from_array(arr, geo=(0, 1, 0, 8, 0, -1), epsg=4326),
        ]

        class _StubCollection:
            datasets = members

        result = run(Pipeline([("slope", {})]), _StubCollection(), on_error="raise")
        assert len(result.outputs) == 2, result.failures
        assert all(isinstance(o, Dataset) for o in result.outputs), result.outputs

    @pytest.mark.parametrize(
        "tool, parameters",
        [
            ("to_crs", {"to_epsg": 3857}),
            ("to_crs", {"to_epsg": 3857, "method": "bilinear", "cell_size": 200000.0}),
            ("resample", {"cell_size": 2.0}),
            ("resample", {"cell_size": 2.0, "method": "average"}),
            ("slope", {"band": 0, "units": "radians"}),
            ("hillshade", {}),
            ("hillshade", {"azimuth": 300.0, "altitude": 30.0, "band": 0}),
            ("fill", {"value": 1.0}),
            ("sieve", {"threshold": 2}),
            ("sieve", {"threshold": 2, "band": 0, "connectedness": 8}),
            ("focal_mean", {"radius": 1}),
            ("focal_std", {"radius": 1}),
        ],
    )
    def test_dataset_tools_run_through_runner(self, raster_ds, tool, parameters):
        """Each Dataset-input tool dispatches and returns a Dataset.

        Args:
            raster_ds: small raster fixture.
            tool: the tool name.
            parameters: its parameters.

        Test scenario:
            Every Dataset tool runs end-to-end (catches param/method-name drift the
            metadata tests would miss); array ops materialize back to a Dataset.
        """
        result = run(Pipeline([(tool, parameters)]), raster_ds, on_error="raise")
        assert result.ok, result.failures
        assert isinstance(result.outputs[0], Dataset), result.outputs

    @pytest.mark.parametrize(
        "tool, parameters",
        [
            ("to_h3", {"resolution": 5}),
            ("voronoi", {}),
            ("voronoi", {"values": "elevation"}),
            ("quadtree", {"column": "elevation", "nmax": 3}),
            ("quadtree", {"column": "elevation", "agg": "max", "nmax": 3, "nmin": 0}),
            ("with_centroid", {}),
            ("with_coordinates", {}),
        ],
    )
    def test_feature_tools_run_through_runner(self, points_fc, tool, parameters):
        """Each FeatureCollection-input tool dispatches and returns one.

        Args:
            points_fc: point-collection fixture.
            tool: the tool name.
            parameters: its parameters.

        Test scenario:
            Every FeatureCollection tool runs end-to-end and yields a
            FeatureCollection.
        """
        result = run(Pipeline([(tool, parameters)]), points_fc, on_error="raise")
        assert result.ok, result.failures
        assert isinstance(result.outputs[0], FeatureCollection), result.outputs

    def test_interpolate_secondary_params_run(self, points_fc):
        """interpolate_to_raster accepts its optional IDW parameters end-to-end.

        Test scenario:
            power/n_neighbors/nodata pass through to the op by name and yield a
            Dataset, guarding the FC->Dataset tool against param-name drift.
        """
        pipe = Pipeline(
            [
                (
                    "interpolate_to_raster",
                    {
                        "column": "elevation",
                        "method": "idw",
                        "cell_size": 1.0,
                        "power": 3.0,
                        "n_neighbors": 3,
                        "nodata": -1.0,
                    },
                )
            ]
        )
        result = run(pipe, points_fc, on_error="raise")
        assert isinstance(result.outputs[0], Dataset), result.outputs

    def test_materialized_output_uses_processed_band_nodata(self):
        """A band>0 terrain op carries that band's no-data into the output (L2).

        Test scenario:
            slope(band=1) on a 2-band raster with distinct per-band no-data yields a
            Dataset advertising band 1's sentinel, not band 0's.
        """
        arr = np.arange(2 * 8 * 8, dtype="float32").reshape(2, 8, 8)
        ds = Dataset.create_from_array(arr, geo=(0, 1, 0, 8, 0, -1), epsg=4326)
        ds.no_data_value = [-1.0, -2.0]
        out = run(Pipeline([("slope", {"band": 1})]), ds, on_error="raise").outputs[0]
        assert out.no_data_value[0] == -2.0, out.no_data_value

    def test_list_of_path_inputs_serial(self, points_fc, tmp_path):
        """A list of pathlib.Path inputs is accepted (M1/L3 parity).

        Args:
            points_fc: point-collection fixture (written to two files).
            tmp_path: pytest temp directory.

        Test scenario:
            Passing [Path(a), Path(b)] runs both, like the equivalent string paths.
        """
        paths = [tmp_path / "a.geojson", tmp_path / "b.geojson"]
        for p in paths:
            points_fc.to_file(str(p))
        pipe = Pipeline(
            [("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0})]
        )
        result = run(pipe, paths, on_error="raise")
        assert len(result.outputs) == 2, result.failures

    def test_materialized_output_preserves_crs(self, raster_ds):
        """A materialized terrain output keeps the source CRS (M1 regression).

        Test scenario:
            slope on an EPSG:4326 raster yields a Dataset still reporting epsg 4326.
        """
        result = run(Pipeline([("slope", {})]), raster_ds, on_error="raise")
        assert result.outputs[0].epsg == 4326, result.outputs[0].epsg

    def test_same_basename_inputs_do_not_collide(self, points_fc, tmp_path):
        """Same-basename inputs from different dirs write distinct outputs (M2).

        Args:
            points_fc: point-collection fixture (written to two dirs).
            tmp_path: pytest temp directory.

        Test scenario:
            Two `pts.geojson` inputs produce two indexed .tif outputs, not one.
        """
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        p1 = tmp_path / "a" / "pts.geojson"
        p2 = tmp_path / "b" / "pts.geojson"
        points_fc.to_file(str(p1))
        points_fc.to_file(str(p2))
        out = tmp_path / "out"
        pipe = Pipeline(
            [("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0})]
        )
        result = run(pipe, [str(p1), str(p2)], out=str(out), on_error="raise")
        assert result.ok, result.failures
        assert len(list(out.glob("*.tif"))) == 2, list(out.iterdir())

    def test_empty_list_raises(self):
        """An empty resolved input set is an error, like an empty glob (L5).

        Test scenario:
            run(pipe, []) raises ValueError rather than silently doing nothing.
        """
        pipe = Pipeline([("slope", {})])
        with pytest.raises(ValueError, match="no inputs to process"):
            run(pipe, [])

    def test_empty_glob_raises(self, tmp_path):
        """A glob that matches nothing is an error, not a silent success.

        Args:
            tmp_path: pytest temp directory (empty).

        Test scenario:
            A *.tif glob over an empty directory raises ValueError rather than
            reporting zero-work success.
        """
        pipe = Pipeline([("slope", {})])
        with pytest.raises(ValueError, match="no inputs matched glob"):
            run(pipe, str(tmp_path / "*.tif"))

    @pytest.mark.slow
    def test_parallel_run_executes_and_returns_paths(self, points_fc, tmp_path):
        """A real process-pool batch writes outputs and returns their paths (M1).

        Args:
            points_fc: point-collection fixture (written to two files).
            tmp_path: pytest temp directory.

        Test scenario:
            Two point files run under parallel=True produce two written .tif paths,
            with one provenance record per input that re-emits its pipeline.
        """
        for name in ("a.geojson", "b.geojson"):
            points_fc.to_file(str(tmp_path / name))
        out = tmp_path / "out"
        pipe = Pipeline(
            [("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0})]
        )
        result = run(
            pipe,
            str(tmp_path / "*.geojson"),
            out=str(out),
            parallel=True,
            max_workers=2,
        )
        assert len(result.outputs) == 2, result.failures
        assert all(isinstance(p, str) and p.endswith(".tif") for p in result.outputs), (
            result.outputs
        )
        assert len(result.provenance) == 2, result.provenance
        assert all(pr.to_pipeline() == pipe for pr in result.provenance), (
            result.provenance
        )

    def test_parallel_requires_out(self, points_fc):
        """parallel=True without an out directory is rejected.

        Test scenario:
            run(..., parallel=True, out=None) raises ValueError.
        """
        pipe = Pipeline([("slope", {})])
        with pytest.raises(ValueError, match="requires an 'out' directory"):
            run(pipe, ["a.tif"], parallel=True, out=None)

    def test_parallel_rejects_runtime_registered_tool(self, tmp_path):
        """parallel=True rejects a tool not in the import-time allowlist.

        Args:
            tmp_path: a valid out directory.

        Test scenario:
            A pipeline using a runtime-registered tool raises up front under
            parallel (workers would not see it), before any process spawns.
        """
        reg.register(
            ToolMetadata(
                "__rt_tool__", "Dataset", "Dataset", (Parameter("x", "Integer"),)
            )
        )
        try:
            pipe = Pipeline([("__rt_tool__", {"x": 1})])
            with pytest.raises(ValueError, match="runtime-registered tools"):
                run(pipe, ["a.tif"], parallel=True, out=str(tmp_path))
        finally:
            reg._REGISTRY.pop("__rt_tool__", None)

    def test_parallel_rejects_overridden_builtin(self, tmp_path):
        """parallel=True rejects a builtin whose spec was overridden in-process.

        Args:
            tmp_path: a valid out directory.

        Test scenario:
            Overriding a builtin via register() then running parallel raises up
            front, because workers re-import the registry and resolve the original
            builtin (the override would be silently ignored).
        """
        original = reg.resolve("slope")
        reg.register(ToolMetadata("slope", "Dataset", "Array", ()))
        try:
            pipe = Pipeline([("slope", {})])
            with pytest.raises(ValueError, match="overridden builtin"):
                run(pipe, ["a.tif"], parallel=True, out=str(tmp_path))
        finally:
            reg.register(original)

    @pytest.mark.slow
    def test_parallel_skip_collects_failure_in_input_order(self, raster_ds, tmp_path):
        """parallel skip collects a bad input's failure, aligned to input order.

        Args:
            raster_ds: raster fixture written to a real file.
            tmp_path: pytest temp directory.

        Test scenario:
            A good raster then a missing path under parallel skip: one written output
            and one collected failure, with the failure carrying the second source.
        """
        good = tmp_path / "good.tif"
        raster_ds.to_file(str(good))
        bad = tmp_path / "missing.tif"
        out = tmp_path / "out"
        result = run(
            Pipeline([("slope", {})]),
            [str(good), str(bad)],
            parallel=True,
            out=str(out),
            on_error="skip",
        )
        assert len(result.outputs) == 1, result.outputs
        assert len(result.failures) == 1, result.failures
        assert result.failures[0][0] == str(bad), result.failures

    @pytest.mark.slow
    def test_parallel_raise_propagates(self, tmp_path):
        """parallel raise re-raises the first worker error instead of collecting it.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A missing input under parallel raise surfaces the worker's
            FileNotFoundError (not an up-front guard ValueError).
        """
        pipe = Pipeline([("slope", {})])
        inputs = [str(tmp_path / "missing.tif")]
        out = str(tmp_path / "out")
        with pytest.raises(FileNotFoundError):
            run(pipe, inputs, parallel=True, out=out, on_error="raise")

    def test_parallel_rejects_non_path_inputs(self, points_fc, tmp_path):
        """parallel=True with an in-memory object input is rejected.

        Args:
            points_fc: an in-memory FeatureCollection (not a path).
            tmp_path: a valid out directory.

        Test scenario:
            Passing a FeatureCollection under parallel raises ValueError naming the
            GDAL-handle constraint.
        """
        pipe = Pipeline([("slope", {})])
        with pytest.raises(ValueError, match="file-path inputs"):
            run(pipe, [points_fc], parallel=True, out=str(tmp_path))
