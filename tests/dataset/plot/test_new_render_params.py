"""cleopatra 0.27 render params (colorbar=, full_bleed=) reach Dataset.plot.

These params are explicit on `ArrayGlyph.plot`, so pyramids forwards them through
its `**kwargs` pass-through; cleopatra rejects an unknown kwarg, so a clean render
proves the param reached the glyph rather than being dropped.
"""

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.plot

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
