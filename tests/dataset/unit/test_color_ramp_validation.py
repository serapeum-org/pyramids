"""Validation guards for `set_color_ramp` (#911) — viz-free, so they run in the main lane.

Every case here raises before `require_cleopatra()`, so these need neither cleopatra nor
matplotlib; the palette-building tests that do live in `tests/dataset/plot/test_plot_color.py`.
"""

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


def _dataset(bands: int = 1) -> Dataset:
    """A writable in-memory raster with `bands` bands and values 1..5."""
    arr = np.random.default_rng(0).integers(1, 6, size=(bands, 10, 10))
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
    )


class TestSetColorRampValidation:
    """`set_color_ramp` rejects bad inputs with a clear message before touching GDAL."""

    def test_band_out_of_range_raises(self):
        """A 1-based band beyond the count is a clear ValueError, not a raw GDAL error."""
        with pytest.raises(ValueError, match="band 99 is out of range"):
            _dataset().set_color_ramp(
                band=99, start_value=1, end_value=5,
                start_color="#000000", end_color="#ffffff",
            )

    def test_band_zero_raises(self):
        """Band 0 is rejected because bands are 1-based."""
        with pytest.raises(ValueError, match="out of range"):
            _dataset().set_color_ramp(
                band=0, start_value=1, end_value=5,
                start_color="#000000", end_color="#ffffff",
            )

    def test_negative_start_value_raises(self):
        """A negative start_value is rejected — GDAL colour indices are non-negative."""
        with pytest.raises(ValueError, match="must be >= 0"):
            _dataset().set_color_ramp(
                band=1, start_value=-1, end_value=5,
                start_color="#000000", end_color="#ffffff",
            )

    def test_non_integer_value_raises(self):
        """A fractional value is a TypeError before any range is built."""
        with pytest.raises(TypeError, match="must be integers"):
            _dataset().set_color_ramp(
                band=1, start_value=1.5, end_value=5,
                start_color="#000000", end_color="#ffffff",
            )

    def test_end_not_greater_than_start_raises(self):
        """A non-increasing range is rejected."""
        with pytest.raises(ValueError, match="must be greater than start_value"):
            _dataset().set_color_ramp(
                band=1, start_value=5, end_value=5,
                start_color="#000000", end_color="#ffffff",
            )

    def test_a_partial_colour_pair_raises(self):
        """Only one of start_color / end_color is rejected."""
        with pytest.raises(ValueError, match="both be given"):
            _dataset().set_color_ramp(
                band=1, start_value=1, end_value=5, start_color="#000000"
            )

    def test_a_blank_colour_raises(self):
        """A blank colour string is treated as missing, not passed to cleopatra."""
        with pytest.raises(ValueError, match="both be given"):
            _dataset().set_color_ramp(
                band=1, start_value=1, end_value=5, start_color="", end_color="#ffffff"
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"start_color": "#000000", "end_color": "#ffffff", "colormap": "viridis"},
            {"colormap": ""},
            {},
        ],
        ids=["both-modes", "blank-colormap", "neither-mode"],
    )
    def test_ambiguous_mode_raises(self, kwargs):
        """Both a colour pair and a colormap, a blank colormap, or neither, is rejected."""
        with pytest.raises(ValueError, match="exactly one"):
            _dataset().set_color_ramp(band=1, start_value=1, end_value=5, **kwargs)
