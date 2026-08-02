"""Tests for the `basemap=` type-dispatch in `render_array` (cleopatra 0.27).

A string provider name (or `True`) draws pyramids' own web-tile basemap under
the raster; a `cleopatra.geo.Basemap` forwards to the glyph's relief/features
reference layer instead.
"""

from unittest.mock import patch

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.dataset._plot_helpers import render_array

pytestmark = pytest.mark.plot

_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
_cleo_config.Config.set_matplotlib_backend("agg")
from cleopatra.array_glyph import ArrayGlyph  # noqa: E402
from cleopatra.geo import Basemap  # noqa: E402


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close figures after each test so the suite does not leak them."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


class TestBasemapDispatch:
    """`basemap=` routes by type: str/True to tiles, Basemap to cleopatra."""

    @staticmethod
    def _dataset():
        """A small single-band EPSG:4326 dataset (basemap needs a CRS)."""
        arr = np.random.default_rng(0).random((1, 8, 8)).astype("float32")
        return Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.1, epsg=4326
        )

    @pytest.mark.parametrize("basemap", ["CartoDB.Positron", True])
    def test_string_or_true_draws_pyramids_tiles(self, basemap):
        """A str provider or `True` routes to pyramids' web-tile `add_basemap`."""
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            self._dataset().plot(band=0, basemap=basemap)
        assert mock_add.called, "str/True basemap must call pyramids add_basemap"

    def test_cleopatra_basemap_is_forwarded_not_tiled(self):
        """A `Basemap` forwards to the glyph and skips the pyramids tile path."""
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            glyph = self._dataset().plot(band=0, basemap=Basemap(relief=False))
        assert not mock_add.called, "a Basemap must not go through the tile path"
        assert glyph is not None

    def test_dict_basemap_is_forwarded_not_tiled(self):
        """A dict (the equivalent of a Basemap spec) also forwards, not tiled."""
        captured, spy = self._spy_on("plot")
        basemap = {"relief": False}
        with patch.object(ArrayGlyph, "plot", spy):
            with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
                self._dataset().plot(band=0, basemap=basemap)
        assert captured.get("basemap") == basemap
        assert not mock_add.called

    def test_no_basemap_draws_no_tiles(self):
        """Omitting `basemap` draws no tile layer."""
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            self._dataset().plot(band=0)
        assert not mock_add.called

    def test_empty_string_basemap_draws_nothing(self):
        """An empty-string `basemap` is treated as no basemap (like None/False).

        It is a falsy string, so it must not reach the tile path (which would
        otherwise leak an AssertionError with no CRS, or draw a default tile
        provider) — it behaves the same as omitting the argument.
        """
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            glyph = self._dataset().plot(band=0, basemap="")
        assert not mock_add.called
        assert glyph is not None

    def test_empty_dict_basemap_draws_nothing(self):
        """An empty dict is falsy, so it is neither tiled nor forwarded."""
        captured, spy = self._spy_on("plot")
        with patch.object(ArrayGlyph, "plot", spy):
            with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
                self._dataset().plot(band=0, basemap={})
        assert "basemap" not in captured
        assert not mock_add.called

    def test_cleopatra_basemap_on_facet_raises(self):
        """A `Basemap` on the faceted path raises a clear error (facet has none)."""
        stack = np.random.default_rng(1).random((3, 6, 6)).astype("float32")
        basemap = Basemap(relief=False)
        with pytest.raises(ValueError, match="faceted plot path"):
            render_array(
                arr=stack,
                mode="facet",
                facet_kwargs={"col": "time", "col_coords": [0, 1, 2]},
                basemap=basemap,
                basemap_epsg=4326,
                extent=[0.0, 0.0, 1.0, 1.0],
            )

    @staticmethod
    def _spy_on(method_name):
        """Wrap an ArrayGlyph method to record the kwargs it was called with."""
        captured: dict = {}
        original = getattr(ArrayGlyph, method_name)

        def spy(self, *args, **kwargs):
            captured.update(kwargs)
            return original(self, *args, **kwargs)

        return captured, spy

    def test_basemap_reaches_the_plot_render_call(self):
        """A Basemap is forwarded to `cleo.plot(basemap=...)`, not just skipped."""
        captured, spy = self._spy_on("plot")
        basemap = Basemap(relief=False)
        with patch.object(ArrayGlyph, "plot", spy):
            self._dataset().plot(band=0, basemap=basemap)
        assert captured.get("basemap") is basemap

    def test_basemap_reaches_the_animate_render_call(self):
        """A Basemap on the animate path forwards to `cleo.animate(basemap=...)`."""
        captured, spy = self._spy_on("animate")
        basemap = Basemap(relief=False)
        stack = np.random.default_rng(2).random((3, 6, 6)).astype("float32")
        with patch.object(ArrayGlyph, "animate", spy):
            with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
                render_array(
                    arr=stack,
                    mode="animate",
                    animation_axis_values=[0, 1, 2],
                    basemap=basemap,
                    basemap_epsg=4326,
                    extent=[0.0, 0.0, 1.0, 1.0],
                )
        assert captured.get("basemap") is basemap
        assert not mock_add.called

    def test_basemap_reaches_the_rgb_animate_render_call(self):
        """A Basemap survives the RGB-animate compositor and reaches animate."""
        captured, spy = self._spy_on("animate")
        basemap = Basemap(relief=False)
        # (time, bands, rows, cols) so the RGB compositor path runs first.
        stack = np.random.default_rng(3).random((3, 3, 6, 6)).astype("float32")
        with patch.object(ArrayGlyph, "animate", spy):
            render_array(
                arr=stack,
                mode="animate",
                animation_axis_values=[0, 1, 2],
                rgb=[0, 1, 2],
                basemap=basemap,
                basemap_epsg=4326,
                extent=[0.0, 0.0, 1.0, 1.0],
            )
        assert captured.get("basemap") is basemap
