"""cleopatra 0.27 render params (colorbar=, full_bleed=) reach Dataset.plot.

These params are explicit on `ArrayGlyph.plot`, so pyramids forwards them through
its `**kwargs` pass-through; cleopatra rejects an unknown kwarg, so a clean render
proves the param reached the glyph rather than being dropped.
"""

from unittest.mock import patch

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.dataset._plot_helpers import render_array

pytestmark = pytest.mark.plot

# Version-gate first: the module binds a 0.28-only spec (ColorBar) at module scope,
# so an installed-but-older cleopatra must skip cleanly, not error at collection.
pytest.importorskip("cleopatra", minversion="0.28", reason="needs cleopatra >= 0.28")
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
_cleo_config.Config.set_matplotlib_backend("agg")
_cleo_array = pytest.importorskip(
    "cleopatra.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
ColorBar = _cleo_array.ColorBar
plt = pytest.importorskip("matplotlib.pyplot", reason="cleopatra not installed")


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close figures after each test so the suite does not leak them."""
    yield
    plt.close("all")


class TestNewRenderParams:
    """cleopatra 0.27 render params pass through the pyramids plot facade."""

    @staticmethod
    def _dataset():
        """A small single-band dataset to render."""
        arr = np.random.default_rng(0).random((1, 8, 8)).astype("float32")
        return Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.1, epsg=4326
        )

    def test_colorbar_spec_reaches_cleopatra(self):
        """`colorbar=ColorBar(...)` renders (reaches ArrayGlyph.plot)."""
        glyph = self._dataset().plot(
            band=0, colorbar=ColorBar(location="bottom", label_color="black")
        )
        assert isinstance(glyph, ArrayGlyph)

    def test_colorbar_false_renders(self):
        """`colorbar=False` suppresses the colour bar and still renders."""
        glyph = self._dataset().plot(band=0, colorbar=False)
        assert isinstance(glyph, ArrayGlyph)

    def test_full_bleed_reaches_cleopatra(self):
        """`full_bleed=True` (chrome-free layout) renders."""
        glyph = self._dataset().plot(band=0, full_bleed=True)
        assert isinstance(glyph, ArrayGlyph)

    def test_colorbar_spec_reaches_animate(self):
        """`colorbar=ColorBar(...)` forwards on the animate path (clean render)."""
        stack = np.random.default_rng(1).random((3, 6, 6)).astype("float32")
        result = render_array(
            arr=stack,
            mode="animate",
            animation_axis_values=[0, 1, 2],
            colorbar=ColorBar(location="bottom"),
            extent=[0.0, 0.0, 1.0, 1.0],
        )
        assert isinstance(result, ArrayGlyph)

    def test_full_bleed_reaches_animate(self):
        """`full_bleed=True` forwards on the animate path (clean render)."""
        stack = np.random.default_rng(2).random((3, 6, 6)).astype("float32")
        result = render_array(
            arr=stack,
            mode="animate",
            animation_axis_values=[0, 1, 2],
            full_bleed=True,
            extent=[0.0, 0.0, 1.0, 1.0],
        )
        assert isinstance(result, ArrayGlyph)

    def test_deprecated_cbar_kwarg_routes_to_the_render_call(self):
        """A loose `cbar_*` kwarg reaches `ArrayGlyph.plot`, not the constructor.

        The loose colour-bar kwargs are in `option_keys()`, so the default split would
        route them to the constructor — where cleopatra's own `cbar_*` deprecation never
        fires (it warns only from the render method). `render_array` forces them to the
        render call so the deprecation surfaces uniformly; assert they get there and the
        label still renders (behaviour preserved). Prefer `colorbar=ColorBar(...)`.
        """
        captured: dict = {}
        original = ArrayGlyph.plot

        def spy(self, *args, **kwargs):
            captured.update(kwargs)
            return original(self, *args, **kwargs)

        with patch.object(ArrayGlyph, "plot", spy):
            glyph = self._dataset().plot(band=0, cbar_label="mm")
        assert "cbar_label" in captured, "loose cbar_label must reach the render call"
        label = glyph.cbar.ax.get_ylabel() or glyph.cbar.ax.get_xlabel()
        assert label == "mm", f"colour-bar label should still render, got {label!r}"
