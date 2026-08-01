"""Unit tests for :mod:`pyramids.processing.pipeline` and validate_params."""

import numpy as np
import pytest

import pyramids.processing.registry as reg
from pyramids.processing.pipeline import Pipeline, Step
from pyramids.processing.schema import ParamSpec, ToolSpec, validate_params


class TestValidateParams:
    """Tests for schema.validate_params."""

    def test_valid_params_pass(self):
        """Well-formed params for a known tool validate cleanly.

        Test scenario:
            slope with a valid band + units raises nothing.
        """
        validate_params(reg.resolve("slope"), {"band": 1, "units": "radians"})

    def test_unknown_param_raises(self):
        """An unknown parameter name is rejected.

        Test scenario:
            slope with a 'nope' param raises ValueError listing valid names.
        """
        with pytest.raises(ValueError, match="unknown parameter"):
            validate_params(reg.resolve("slope"), {"nope": 1})

    def test_type_mismatch_rejects_array_value(self):
        """A non-serializable value handed to a scalar param is rejected.

        Test scenario:
            A numpy array for slope's Integer band raises ValueError.
        """
        with pytest.raises(ValueError, match="expects Integer"):
            validate_params(reg.resolve("slope"), {"band": np.zeros(3)})

    def test_missing_required_raises(self):
        """A missing required parameter is reported.

        Test scenario:
            interpolate_to_raster without its required 'column' raises.
        """
        with pytest.raises(ValueError, match="missing required parameter"):
            validate_params(reg.resolve("interpolate_to_raster"), {})

    def test_for_serialization_rejects_nonserializable_param(self):
        """for_serialization flags a non-serializable declared param.

        Test scenario:
            A Raster param passes normal validation but fails under
            for_serialization=True.
        """
        spec = ToolSpec("t", "Dataset", "Dataset", (ParamSpec("mask", "Raster"),))
        validate_params(spec, {"mask": "r.tif"})
        with pytest.raises(ValueError, match="not\n?.*serializable|not pipeline-serializable"):
            validate_params(spec, {"mask": "r.tif"}, for_serialization=True)


class TestPipeline:
    """Tests for the Pipeline object."""

    def test_construct_validates_and_iterates(self):
        """A valid chain constructs and iterates as Steps.

        Test scenario:
            A two-step chain has len 2 and yields Step objects with tool+params.
        """
        p = Pipeline([("interpolate_to_raster", {"column": "z"}), ("slope", {})])
        assert len(p) == 2, f"expected 2 steps, got {len(p)}"
        first = next(iter(p))
        assert isinstance(first, Step) and first.tool == "interpolate_to_raster", first

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
        """A step that is not a (tool, params) pair is rejected.

        Test scenario:
            A bare string step raises a clear ValueError.
        """
        with pytest.raises(ValueError, match="must be a .tool, params. pair"):
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
        p = Pipeline([("interpolate_to_raster", {"column": "z", "cell_size": 1000.0}), ("slope", {})])
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
        reg.register(ToolSpec("__ns_tool__", "Dataset", "Dataset", (ParamSpec("mask", "Raster"),)))
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
