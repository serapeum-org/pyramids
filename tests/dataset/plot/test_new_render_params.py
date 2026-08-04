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

    def test_deprecated_cbar_kwarg_folds_into_colorbar_and_renders(self):
        """A loose `cbar_*` kwarg is folded into `colorbar=ColorBar(...)` and still renders.

        `render_array` translates the deprecated loose colour-bar kwargs into a typed
        ColorBar before the cleopatra call (so cleopatra never sees the loose form); the
        folded spec still renders the label.
        """
        captured: dict = {}
        original = ArrayGlyph.plot

        def spy(self, *args, **kwargs):
            captured.update(kwargs)
            return original(self, *args, **kwargs)

        with patch.object(ArrayGlyph, "plot", spy):
            with pytest.warns(DeprecationWarning, match="cbar_"):
                glyph = self._dataset().plot(band=0, cbar_label="mm", cbar_length=0.85)
        assert "cbar_label" not in captured, (
            "loose cbar_label must be folded away, not passed loose"
        )
        assert isinstance(captured.get("colorbar"), ColorBar), (
            "cbar_* must fold into a ColorBar"
        )
        label = glyph.cbar.ax.get_ylabel() or glyph.cbar.ax.get_xlabel()
        assert label == "mm", f"folded ColorBar label should render, got {label!r}"

    def test_explicit_colorbar_wins_over_loose_cbar(self):
        """An explicit `colorbar=ColorBar` wins; the loose `cbar_*` are dropped (deprecated)."""
        with pytest.warns(
            DeprecationWarning, match="ignored because an explicit colorbar"
        ):
            glyph = self._dataset().plot(
                band=0, colorbar=ColorBar(label="TYPED"), cbar_label="LOOSE"
            )
        label = glyph.cbar.ax.get_ylabel() or glyph.cbar.ax.get_xlabel()
        assert label == "TYPED", f"explicit ColorBar should win, got {label!r}"

    def test_colorbar_true_still_folds_loose_cbar(self):
        """`colorbar=True` + a loose `cbar_*` folds the styling in (not dropped).

        `True` / `None` carry no caption of their own, so the loose kwargs must still
        render — only a typed `ColorBar` (or `colorbar=False`) suppresses them.
        """
        with pytest.warns(DeprecationWarning, match="cbar_"):
            glyph = self._dataset().plot(band=0, colorbar=True, cbar_label="DEPTH")
        label = glyph.cbar.ax.get_ylabel() or glyph.cbar.ax.get_xlabel()
        assert label == "DEPTH", (
            f"colorbar=True must keep the loose label, got {label!r}"
        )

    def test_deprecated_cbar_kwarg_renders_on_the_facet_path(self):
        """A loose `cbar_*` kwarg still renders the shared colour-bar label on the facet path.

        cleopatra's `facet` does not accept `colorbar=ColorBar` (only the loose `cbar_*`
        kwargs), so the translation is skipped on the facet path — the loose form is kept,
        and the returned grid's shared colour bar still carries the label.
        """
        stack = np.random.default_rng(3).random((3, 6, 6)).astype("float32")
        result = render_array(
            arr=stack,
            mode="facet",
            facet_kwargs={"col": "time", "col_coords": [0, 1, 2]},
            cbar_label="mm",
            extent=[0.0, 0.0, 1.0, 1.0],
        )
        label = result.cbar.ax.get_ylabel() or result.cbar.ax.get_xlabel()
        assert label == "mm", f"facet colour-bar label should render, got {label!r}"
