"""Unit tests for :mod:`pyramids.processing.pipeline` and validate_parameters."""

import numpy as np
import pytest

import pyramids.processing.registry as reg
from pyramids.processing.pipeline import Pipeline, Step
from pyramids.processing.schema import Parameter, ToolMetadata, validate_parameters


class TestValidateParams:
    """Tests for schema.validate_parameters."""

    def test_valid_params_pass(self):
        """Well-formed parameters for a known tool validate cleanly.

        Test scenario:
            slope with a valid band + units raises nothing.
        """
        validate_parameters(reg.resolve("slope"), {"band": 1, "units": "radians"})

    def test_unknown_param_raises(self):
        """An unknown parameter name is rejected.

        Test scenario:
            slope with a 'nope' param raises ValueError listing valid names.
        """
        spec = reg.resolve("slope")
        with pytest.raises(ValueError, match="unknown parameter"):
            validate_parameters(spec, {"nope": 1})

    def test_type_mismatch_rejects_array_value(self):
        """A non-serializable value handed to a scalar param is rejected.

        Test scenario:
            A numpy array for slope's Integer band raises ValueError.
        """
        spec = reg.resolve("slope")
        bad = np.zeros(3)
        with pytest.raises(ValueError, match="expects Integer"):
            validate_parameters(spec, {"band": bad})

    def test_missing_required_raises(self):
        """A missing required parameter is reported.

        Test scenario:
            interpolate_to_raster without its required 'column' raises.
        """
        spec = reg.resolve("interpolate_to_raster")
        with pytest.raises(ValueError, match="missing required parameter"):
            validate_parameters(spec, {})

    def test_for_serialization_rejects_nonserializable_param(self):
        """for_serialization flags a non-serializable declared param.

        Test scenario:
            A Raster param passes normal validation but fails under
            for_serialization=True.
        """
        spec = ToolMetadata("t", "Dataset", "Dataset", (Parameter("mask", "Raster"),))
        validate_parameters(spec, {"mask": "r.tif"})
        with pytest.raises(
            ValueError, match="not\n?.*serializable|not pipeline-serializable"
        ):
            validate_parameters(spec, {"mask": "r.tif"}, for_serialization=True)


class TestPipeline:
    """Tests for the Pipeline object."""

    def test_construct_validates_and_iterates(self):
        """A valid chain constructs and iterates as Steps.

        Test scenario:
            A two-step chain has len 2 and yields Step objects with tool+parameters.
        """
        p = Pipeline([("interpolate_to_raster", {"column": "z"}), ("slope", {})])
        assert len(p) == 2, f"expected 2 steps, got {len(p)}"
        first = next(iter(p))
        assert isinstance(first, Step), first
        assert first.tool == "interpolate_to_raster", first

    def test_unknown_tool_raises_at_construction(self):
        """An unknown tool fails at construction, not at run.

        Test scenario:
            A step naming 'nope' raises ValueError from __init__.
        """
        with pytest.raises(ValueError, match="unknown tool"):
            Pipeline([("nope", {})])

    def test_invalid_param_raises_at_construction(self):
        """A schema-invalid param fails at construction.

        Test scenario:
            slope with band=1.5 raises ValueError from __init__.
        """
        with pytest.raises(ValueError, match="expects Integer"):
            Pipeline([("slope", {"band": 1.5})])

    def test_malformed_step_raises(self):
        """A step that is not a (tool, parameters) pair is rejected.

        Test scenario:
            A bare string step raises a clear ValueError.
        """
        with pytest.raises(ValueError, match="must be a .tool, parameters. pair"):
            Pipeline(["slope"])  # type: ignore[list-item]

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict/from_dict round-trips to an equal pipeline.

        Test scenario:
            from_dict(to_dict(p)) == p.
        """
        p = Pipeline([("slope", {"band": 0, "units": "degrees"})])
        assert Pipeline.from_dict(p.to_dict()) == p, "dict round-trip should be equal"

    def test_from_dict_rejects_non_mapping(self):
        """from_dict rejects data without a 'pipeline' key.

        Test scenario:
            A mapping missing 'pipeline' raises ValueError.
        """
        with pytest.raises(ValueError, match="'pipeline' key"):
            Pipeline.from_dict({"nope": []})

    def test_to_yaml_from_yaml_roundtrip(self, tmp_path):
        """to_yaml/from_yaml round-trips to an equal pipeline.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A scalar-param pipeline written to YAML loads back equal.
        """
        p = Pipeline(
            [
                ("interpolate_to_raster", {"column": "z", "cell_size": 1000.0}),
                ("slope", {}),
            ]
        )
        path = tmp_path / "pipe.yaml"
        p.to_yaml(str(path))
        assert Pipeline.from_yaml(str(path)) == p, "YAML round-trip should be equal"

    def test_to_yaml_rejects_nonserializable(self, tmp_path):
        """to_yaml refuses to write a non-serializable param.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A tool with a Raster param raises on to_yaml and writes no file.
        """
        reg.register(
            ToolMetadata(
                "__ns_tool__", "Dataset", "Dataset", (Parameter("mask", "Raster"),)
            )
        )
        try:
            p = Pipeline([("__ns_tool__", {"mask": "r.tif"})])
            path = tmp_path / "q.yaml"
            with pytest.raises(ValueError, match="not pipeline-serializable"):
                p.to_yaml(str(path))
            assert not path.exists(), "no file should be written on rejection"
        finally:
            reg._REGISTRY.pop("__ns_tool__", None)

    def test_from_yaml_rejects_malformed(self, tmp_path):
        """from_yaml rejects a file that is not a pipeline mapping.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A YAML scalar/mapping without 'pipeline' raises ValueError.
        """
        bad = tmp_path / "bad.yaml"
        bad.write_text("just: a-scalar\n", encoding="utf-8")
        with pytest.raises(ValueError, match="'pipeline' key"):
            Pipeline.from_yaml(str(bad))

    def test_eq_non_pipeline_is_false(self):
        """Comparing a Pipeline to a non-Pipeline is not equal.

        Test scenario:
            __eq__ returns NotImplemented for a str, so == falls back to False.
        """
        assert Pipeline([("slope", {})]) != "not a pipeline", "should not equal a str"

    def test_to_dict_params_are_copied(self):
        """Mutating to_dict()'s parameters does not affect the pipeline (L1).

        Test scenario:
            Editing the parameters dict returned by to_dict leaves the pipeline's own
            step parameters unchanged.
        """
        p = Pipeline([("slope", {"band": 0})])
        d = p.to_dict()
        d["pipeline"][0]["parameters"]["band"] = 99
        assert p.steps[0].parameters["band"] == 0, (
            "to_dict must not leak internal parameters"
        )

    def test_from_dict_pipeline_not_list(self):
        """from_dict rejects a 'pipeline' that is not a list.

        Test scenario:
            {'pipeline': 'nope'} raises ValueError naming the list requirement.
        """
        with pytest.raises(ValueError, match="must be a list"):
            Pipeline.from_dict({"pipeline": "nope"})

    def test_from_dict_step_missing_tool(self):
        """from_dict rejects a step without a 'tool' key.

        Test scenario:
            A step mapping lacking 'tool' raises ValueError.
        """
        with pytest.raises(ValueError, match="'tool' key"):
            Pipeline.from_dict({"pipeline": [{"parameters": {}}]})

    def test_steps_property_returns_copy(self):
        """steps returns a fresh list that does not alias internal state.

        Test scenario:
            Mutating the returned list does not change the pipeline's length.
        """
        p = Pipeline([("slope", {"band": 0})])
        got = p.steps
        got.append("x")
        got[0].parameters["band"] = 99
        assert len(p) == 1, "mutating the steps copy must not affect the pipeline"
        assert p.steps[0].parameters["band"] == 0, (
            "mutating a step's parameters must not leak back"
        )
