"""Unit tests for :mod:`pyramids.processing.registry`."""

import pytest

import pyramids.processing.registry as reg
from pyramids.processing.schema import (
    INPUT_TYPES,
    OUTPUT_TYPES,
    Parameter,
    ToolMetadata,
)

EXPECTED_ALLOWLIST = {
    "slope",
    "aspect",
    "hillshade",
    "to_crs",
    "resample",
    "fill",
    "sieve",
    "focal_mean",
    "focal_std",
    "interpolate_to_raster",
    "to_h3",
    "voronoi",
    "quadtree",
    "with_centroid",
    "with_coordinates",
}


class TestRegistry:
    """Tests for the tool registry surface (register/resolve/tool_names/catalog)."""

    def test_allowlist_names(self):
        """The curated v1 allowlist is exactly the expected set of tools.

        Test scenario:
            tool_names() returns the registered ops (the ~15 v1 allowlist).
        """
        assert set(reg.tool_names()) == EXPECTED_ALLOWLIST, reg.tool_names()

    def test_resolve_returns_spec(self):
        """resolve returns the ToolMetadata for a known tool.

        Test scenario:
            resolve('slope') is a Dataset->Array ToolMetadata named 'slope'.
        """
        spec = reg.resolve("slope")
        assert isinstance(spec, ToolMetadata), spec
        assert spec.name == "slope", spec

    def test_resolve_unknown_raises_and_lists_tools(self):
        """resolve raises for an unknown tool and lists the registered names.

        Test scenario:
            resolve('nope') raises ValueError whose message includes a known tool.
        """
        with pytest.raises(ValueError, match="unknown tool") as exc:
            reg.resolve("nope")
        assert "slope" in str(exc.value), f"error should list tools: {exc.value}"

    def test_registry_view_is_readonly(self):
        """catalog() returns a read-only mapping.

        Test scenario:
            Assigning into the returned mapping raises TypeError.
        """
        view = reg.catalog()
        with pytest.raises(TypeError):
            view["x"] = None  # type: ignore[index]

    @pytest.mark.parametrize(
        "name, input_type, output_type",
        [
            ("slope", "Dataset", "Array"),
            ("interpolate_to_raster", "FeatureCollection", "Dataset"),
            ("to_h3", "FeatureCollection", "FeatureCollection"),
        ],
    )
    def test_input_and_output_type_metadata(self, name, input_type, output_type):
        """Each tool declares the right input and return types.

        Args:
            name: Tool name.
            input_type: Expected input type.
            output_type: Expected return type.

        Test scenario:
            The cross-input op (interpolate_to_raster) is FeatureCollection->Dataset.
        """
        spec = reg.resolve(name)
        assert (spec.input_type, spec.output_type) == (input_type, output_type), spec

    def test_all_specs_have_valid_types(self):
        """Every registered spec has valid input/return types.

        Test scenario:
            Iterating the registry, each spec's input and returns are in INPUT_TYPES.
        """
        for name, spec in reg.catalog().items():
            assert spec.input_type in INPUT_TYPES, f"{name} bad input {spec.input_type}"
            assert spec.output_type in OUTPUT_TYPES, (
                f"{name} bad returns {spec.output_type}"
            )

    def test_register_and_resolve_roundtrip(self):
        """register adds a tool that resolve can then return.

        Test scenario:
            A throwaway tool registers and resolves; cleaned up afterwards so the
            global registry is left as found.
        """
        temp = ToolMetadata(
            "__tmp_tool__", "Dataset", "Dataset", (Parameter("x", "Integer"),)
        )
        reg.register(temp)
        try:
            assert reg.resolve("__tmp_tool__") is temp, "registered tool should resolve"
        finally:
            reg._REGISTRY.pop("__tmp_tool__", None)
        assert "__tmp_tool__" not in reg.tool_names(), (
            "cleanup should remove the temp tool"
        )

    def test_interpolate_required_column(self):
        """interpolate_to_raster declares a required 'column' Field parameter.

        Test scenario:
            The 'column' param exists, is a Field, and is not optional.
        """
        column = reg.resolve("interpolate_to_raster").param("column")
        assert column is not None, "column param should exist"
        assert column.parameter_type == "Field", column
        assert not column.optional, column
