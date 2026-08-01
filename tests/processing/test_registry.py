"""Unit tests for :mod:`pyramids.processing.registry`."""

import pytest

import pyramids.processing.registry as reg
from pyramids.processing.schema import (
    RECEIVER_TYPES,
    RETURN_TYPES,
    ParamSpec,
    ToolSpec,
)

EXPECTED_ALLOWLIST = {
    "slope",
    "aspect",
    "hillshade",
    "to_crs",
    "resample",
    "interpolate_to_raster",
    "to_h3",
}


class TestRegistry:
    """Tests for the tool registry surface (register/resolve/tool_names/registry)."""

    def test_allowlist_names(self):
        """The curated v1 allowlist is exactly the seven expected tools.

        Test scenario:
            tool_names() returns the seven registered ops.
        """
        assert set(reg.tool_names()) == EXPECTED_ALLOWLIST, reg.tool_names()

    def test_resolve_returns_spec(self):
        """resolve returns the ToolSpec for a known tool.

        Test scenario:
            resolve('slope') is a Dataset->Dataset ToolSpec named 'slope'.
        """
        spec = reg.resolve("slope")
        assert isinstance(spec, ToolSpec) and spec.name == "slope", spec

    def test_resolve_unknown_raises_and_lists_tools(self):
        """resolve raises for an unknown tool and lists the registered names.

        Test scenario:
            resolve('nope') raises ValueError whose message includes a known tool.
        """
        with pytest.raises(ValueError, match="unknown tool") as exc:
            reg.resolve("nope")
        assert "slope" in str(exc.value), f"error should list tools: {exc.value}"

    def test_registry_view_is_readonly(self):
        """registry() returns a read-only mapping.

        Test scenario:
            Assigning into the returned mapping raises TypeError.
        """
        view = reg.get_registry()
        with pytest.raises(TypeError):
            view["x"] = None  # type: ignore[index]

    @pytest.mark.parametrize(
        "name, receiver, returns",
        [
            ("slope", "Dataset", "Array"),
            ("interpolate_to_raster", "FeatureCollection", "Dataset"),
            ("to_h3", "FeatureCollection", "FeatureCollection"),
        ],
    )
    def test_receiver_and_returns_metadata(self, name, receiver, returns):
        """Each tool declares the right receiver and return types.

        Args:
            name: Tool name.
            receiver: Expected receiver type.
            returns: Expected return type.

        Test scenario:
            The cross-receiver op (interpolate_to_raster) is FeatureCollection->Dataset.
        """
        spec = reg.resolve(name)
        assert (spec.receiver, spec.returns) == (receiver, returns), spec

    def test_all_specs_have_valid_types(self):
        """Every registered spec has valid receiver/return types.

        Test scenario:
            Iterating the registry, each spec's receiver and returns are in RECEIVER_TYPES.
        """
        for name, spec in reg.get_registry().items():
            assert spec.receiver in RECEIVER_TYPES, f"{name} bad receiver {spec.receiver}"
            assert spec.returns in RETURN_TYPES, f"{name} bad returns {spec.returns}"

    def test_register_and_resolve_roundtrip(self):
        """register adds a tool that resolve can then return.

        Test scenario:
            A throwaway tool registers and resolves; cleaned up afterwards so the
            global registry is left as found.
        """
        temp = ToolSpec("__tmp_tool__", "Dataset", "Dataset", (ParamSpec("x", "Integer"),))
        reg.register(temp)
        try:
            assert reg.resolve("__tmp_tool__") is temp, "registered tool should resolve"
        finally:
            reg._REGISTRY.pop("__tmp_tool__", None)
        assert "__tmp_tool__" not in reg.tool_names(), "cleanup should remove the temp tool"

    def test_interpolate_required_column(self):
        """interpolate_to_raster declares a required 'column' Field parameter.

        Test scenario:
            The 'column' param exists, is a Field, and is not optional.
        """
        column = reg.resolve("interpolate_to_raster").param("column")
        assert column is not None and column.param_type == "Field" and not column.optional, column
