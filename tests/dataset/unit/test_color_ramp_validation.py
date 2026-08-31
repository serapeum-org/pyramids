"""Validation guards for `set_color_ramp` (#911) — viz-free, so they run in the main lane.

Every case here raises before `require_cleopatra()`, so these need neither cleopatra nor
matplotlib; the palette-building tests that do live in `tests/dataset/plot/test_plot_color.py`.
"""

import numpy as np
import pytest

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset
from pyramids.base.georeference import GeoReference

pytestmark = pytest.mark.core


def _dataset(bands: int = 1) -> Dataset:
    """A writable in-memory raster with `bands` bands and values 1..5."""
    arr = np.random.default_rng(0).integers(1, 6, size=(bands, 10, 10))
    return Dataset.from_array(
               arr,
               geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
           )


class TestSetColorRampValidation:
    """`set_color_ramp` rejects bad inputs with a clear message before touching GDAL."""

    def test_band_out_of_range_raises(self):
        """A 1-based band beyond the count is a clear ValueError, not a raw GDAL error."""
        dataset = _dataset()
        with pytest.raises(ValueError, match="band 99 is out of range"):
            dataset.set_color_ramp(
                band=99,
                start_value=1,
                end_value=5,
                start_color="#000000",
                end_color="#ffffff",
            )

    def test_band_zero_raises(self):
        """Band 0 is rejected because bands are 1-based."""
        dataset = _dataset()
        with pytest.raises(ValueError, match="out of range"):
            dataset.set_color_ramp(
                band=0,
                start_value=1,
                end_value=5,
                start_color="#000000",
                end_color="#ffffff",
            )

    def test_negative_start_value_raises(self):
        """A negative start_value is rejected — GDAL colour indices are non-negative."""
        dataset = _dataset()
        with pytest.raises(ValueError, match="must be >= 0"):
            dataset.set_color_ramp(
                band=1,
                start_value=-1,
                end_value=5,
                start_color="#000000",
                end_color="#ffffff",
            )

    def test_non_integer_value_raises(self):
        """A fractional value is a TypeError before any range is built."""
        dataset = _dataset()
        with pytest.raises(TypeError, match="whole number"):
            dataset.set_color_ramp(
                band=1,
                start_value=1.5,
                end_value=5,
                start_color="#000000",
                end_color="#ffffff",
            )

    def test_non_numeric_value_raises_type_error(self):
        """A non-numeric value gets the documented TypeError, not a cryptic int() error."""
        dataset = _dataset()
        with pytest.raises(TypeError, match="must be an integer"):
            dataset.set_color_ramp(
                band=1,
                start_value="1",
                end_value=5,
                start_color="#000000",
                end_color="#ffffff",
            )

    def test_boolean_value_raises_type_error(self):
        """A bool is not a meaningful colour index and is rejected."""
        dataset = _dataset()
        with pytest.raises(TypeError, match="must be an integer"):
            dataset.set_color_ramp(
                band=1,
                start_value=True,
                end_value=5,
                start_color="#000000",
                end_color="#ffffff",
            )

    def test_end_not_greater_than_start_raises(self):
        """A non-increasing range is rejected."""
        dataset = _dataset()
        with pytest.raises(ValueError, match="must be greater than start_value"):
            dataset.set_color_ramp(
                band=1,
                start_value=5,
                end_value=5,
                start_color="#000000",
                end_color="#ffffff",
            )

    def test_a_partial_colour_pair_raises(self):
        """Only one of start_color / end_color is rejected."""
        dataset = _dataset()
        with pytest.raises(ValueError, match="both be given"):
            dataset.set_color_ramp(
                band=1, start_value=1, end_value=5, start_color="#000000"
            )

    def test_a_blank_colour_raises(self):
        """A blank colour string is treated as missing, not passed to cleopatra."""
        dataset = _dataset()
        with pytest.raises(ValueError, match="both be given"):
            dataset.set_color_ramp(
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
        dataset = _dataset()
        with pytest.raises(ValueError, match="exactly one"):
            dataset.set_color_ramp(band=1, start_value=1, end_value=5, **kwargs)

    def test_read_only_on_disk_dataset_raises(self, tmp_path):
        """The facade guard rejects a read-only on-disk raster before spilling a sidecar."""
        path = tmp_path / "ro.tif"
        Dataset.from_array(
            np.ones((3, 3), dtype="float32"),
            path=str(path),
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
        )
        ro_ds = Dataset.read_file(str(path), read_only=True)
        with pytest.raises(ReadOnlyError, match="read-only"):
            ro_ds.set_color_ramp(
                band=1,
                start_value=1,
                end_value=5,
                start_color="#000000",
                end_color="#ffffff",
            )
