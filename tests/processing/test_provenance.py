"""Unit tests for :mod:`pyramids.processing.provenance` and run() provenance."""

import geopandas as gpd
from shapely.geometry import Point

from pyramids.feature import FeatureCollection
from pyramids.processing import Pipeline, run
from pyramids.processing.provenance import Provenance, StepRecord


def _points_fc():
    """Build a small point FeatureCollection for interpolation."""
    gdf = gpd.GeoDataFrame(
        {"elevation": [1.0, 2.0, 3.0, 4.0, 5.0]},
        geometry=[Point(0, 0), Point(4, 0), Point(0, 4), Point(4, 4), Point(2, 2)],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


class TestProvenance:
    """Tests for the Provenance / StepRecord records."""

    def test_total_seconds_sums_steps(self):
        """total_seconds is the sum of per-step durations.

        Test scenario:
            Two step records of 0.1 and 0.2 seconds total 0.3.
        """
        prov = Provenance("<x>", [StepRecord("a", {}, 0.1), StepRecord("b", {}, 0.2)])
        assert abs(prov.total_seconds - 0.3) < 1e-9, prov.total_seconds

    def test_to_pipeline_reemits_equal_pipeline(self):
        """to_pipeline re-emits a pipeline equal to the recorded steps.

        Test scenario:
            A provenance built from (tool, params) records re-emits an equal
            Pipeline.
        """
        prov = Provenance(
            "<x>", [StepRecord("slope", {"band": 0, "units": "degrees"}, 0.0)]
        )
        assert prov.to_pipeline() == Pipeline(
            [("slope", {"band": 0, "units": "degrees"})]
        )


class TestRunProvenance:
    """Tests for provenance collected by run()."""

    def test_run_records_provenance_that_reproduces_pipeline(self):
        """run() records provenance whose re-emit equals the original pipeline.

        Test scenario:
            After running a cross-receiver pipeline, provenance[0].to_pipeline()
            equals the pipeline that was run, and total_seconds is non-negative.
        """
        pipe = Pipeline(
            [
                ("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0}),
                ("slope", {}),
            ]
        )
        result = run(pipe, _points_fc(), on_error="raise")
        assert len(result.provenance) == 1, result.provenance
        prov = result.provenance[0]
        assert prov.to_pipeline() == pipe, (
            "re-emitted pipeline should equal the original"
        )
        assert prov.total_seconds >= 0.0, prov.total_seconds
        assert [s.tool for s in prov.steps] == ["interpolate_to_raster", "slope"], (
            prov.steps
        )

    def test_failed_input_records_no_provenance(self):
        """A failed input contributes no provenance entry.

        Test scenario:
            A wrong-receiver run under skip yields one failure and zero provenance.
        """
        result = run(Pipeline([("slope", {})]), _points_fc(), on_error="skip")
        assert len(result.provenance) == 0, result.provenance
        assert len(result.failures) == 1, result.failures
