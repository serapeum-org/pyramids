"""Tests for `pyramids.plot`, the lazy re-export of cleopatra plot specs."""

import builtins
import importlib
from unittest.mock import patch

import pytest

import pyramids.plot as plot_specs
from pyramids.base._errors import OptionalPackageDoesNotExist

# No module-scope cleopatra import: most tests here exercise the lazy re-export
# without cleopatra (and so run in the extras-free core job); only the
# resolve-to-cleopatra test needs it, gated with a function-body importorskip.


class TestPlotSpecReExports:
    """`pyramids.plot` lazily re-exports cleopatra's plot spec classes."""

    def test_all_and_dir_list_the_specs(self):
        """`__all__` / `dir()` advertise every re-exported spec, sorted."""
        expected = [
            "Basemap",
            "CellValues",
            "Classify",
            "ColorBar",
            "ColorScaling",
            "Contour",
            "DataStyle",
            "Feature",
            "FrameLabel",
            "PanelLabels",
            "PointOverlay",
        ]
        assert plot_specs.__all__ == expected
        assert dir(plot_specs) == expected

    @pytest.mark.parametrize(
        "name, module_name, attribute",
        [
            ("ColorBar", "cleopatra.styling.colorbar", "ColorBar"),
            ("FrameLabel", "cleopatra.glyphs.gridded.array_glyph", "FrameLabel"),
            ("PointOverlay", "cleopatra.glyphs.gridded.array_glyph", "PointOverlay"),
            ("PanelLabels", "cleopatra.glyphs.gridded.array_glyph", "PanelLabels"),
            ("Basemap", "cleopatra.basemap.geo", "Basemap"),
            ("Feature", "cleopatra.basemap.geo", "Feature"),
            ("ColorScaling", "cleopatra.styling.scaling", "ColorScaling"),
            ("Contour", "cleopatra.styling.params", "Contour"),
            ("CellValues", "cleopatra.styling.params", "CellValues"),
            ("DataStyle", "cleopatra.styling.params", "DataStyle"),
            ("Classify", "cleopatra.styling.params", "Classify"),
        ],
    )
    def test_spec_resolves_to_cleopatra_class(self, name, module_name, attribute):
        """Each name resolves to the identical cleopatra class object."""
        pytest.importorskip(module_name, reason="cleopatra not installed")
        module = importlib.import_module(module_name)
        assert getattr(plot_specs, name) is getattr(module, attribute)

    def test_unknown_attribute_raises_attribute_error(self):
        """An unknown attribute raises AttributeError, not the install hint."""
        with pytest.raises(AttributeError, match="Nonexistent"):
            plot_specs.Nonexistent

    def test_missing_cleopatra_raises_the_viz_install_hint(self):
        """Accessing a spec without cleopatra raises the [viz] install hint."""
        real_import = builtins.__import__

        def _no_cleopatra(name, *args, **kwargs):
            if name.startswith("cleopatra"):
                raise ImportError("simulated missing cleopatra")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_no_cleopatra):
            with pytest.raises(OptionalPackageDoesNotExist, match="viz"):
                plot_specs.ColorBar

    def test_too_old_cleopatra_raises_upgrade_hint(self):
        """An importable-but-old cleopatra lacking a spec raises the upgrade hint."""

        class _OldCleopatraModule:
            """Stands in for a cleopatra too old to carry the spec."""

        with patch(
            "pyramids.plot.require_optional", return_value=_OldCleopatraModule()
        ):
            with pytest.raises(OptionalPackageDoesNotExist, match="0.29"):
                plot_specs.ColorBar
