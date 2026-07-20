"""Tests for decimated (out_shape) reads in the core read_array.

Covers `read_array(out_shape=..., resampling=...)` on any Dataset (not just
the COG engine): down/up-sampling, method sensitivity, window composition,
multi-band stacking, overview usage, and the validation contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import OutOfBoundsError
from pyramids.dataset import Dataset, Window

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def ramp_dataset() -> Dataset:
    """A 64x64 float32 ramp on a unit grid.

    Returns:
        Dataset: Single-band in-memory dataset, value == row*64 + col.
    """
    arr = np.arange(64 * 64, dtype="float32").reshape(64, 64)
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 64), cell_size=1.0, epsg=4326
    )


class TestOutShapeReads:
    """read_array(out_shape=...) behaviour."""

    def test_downsample_halves_shape(self, ramp_dataset):
        """out_shape=(32, 32) returns exactly that shape."""
        result = ramp_dataset.read_array(band=0, out_shape=(32, 32))
        assert result.shape == (32, 32), f"unexpected shape {result.shape}"

    def test_constant_raster_survives_nearest(self):
        """Nearest decimation of a constant raster stays constant."""
        ds = Dataset.create_from_array(
            np.full((16, 16), 5.0, dtype="float32"),
            top_left_corner=(0, 16),
            cell_size=1.0,
            epsg=4326,
        )
        result = ds.read_array(band=0, out_shape=(8, 8))
        assert np.isclose(result, 5.0).all(), "constant values must survive decimation"

    def test_average_differs_from_nearest_on_checkerboard(self):
        """The resampling string actually reaches GDAL.

        Test scenario:
            Averaging a 0/100 checkerboard 2x gives 50 everywhere; nearest
            keeps the original values.
        """
        cb = (np.indices((16, 16)).sum(axis=0) % 2 * 100).astype("float32")
        ds = Dataset.create_from_array(
            cb, top_left_corner=(0, 16), cell_size=1.0, epsg=4326
        )
        averaged = ds.read_array(band=0, out_shape=(8, 8), resampling="average")
        nearest = ds.read_array(band=0, out_shape=(8, 8), resampling="nearest")
        assert np.allclose(averaged, 50.0), f"average of 0/100 must be 50: {averaged}"
        assert set(np.unique(nearest)) <= {
            0.0,
            100.0,
        }, f"nearest must keep original values: {np.unique(nearest)}"

    def test_window_composition(self, ramp_dataset):
        """out_shape decimates a sub-window, not the whole raster.

        Test scenario:
            A 32x32 window decimated to 16x16 equals decimating the
            equivalent slice — pinned by the window's top-left value.
        """
        result = ramp_dataset.read_array(
            band=0, window=Window(0, 0, 32, 32), out_shape=(16, 16)
        )
        assert result.shape == (16, 16), f"unexpected shape {result.shape}"
        window_values = ramp_dataset.read_array(band=0)[:32, :32]
        assert result.min() >= window_values.min(), (
            "values leaked from outside the window"
        )
        assert result.max() <= window_values.max(), (
            "values leaked from outside the window"
        )
        slice_ds = Dataset.create_from_array(
            window_values, top_left_corner=(0, 32), cell_size=1.0, epsg=4326
        )
        np.testing.assert_array_equal(
            result,
            slice_ds.read_array(band=0, out_shape=(16, 16)),
            err_msg="window+out_shape must equal decimating the same slice",
        )

    def test_list_window_matches_window_object(self, ramp_dataset):
        """The legacy x-first list window composes with out_shape too."""
        via_object = ramp_dataset.read_array(
            band=0, window=Window(0, 0, 32, 32), out_shape=(16, 16)
        )
        via_list = ramp_dataset.read_array(
            band=0, window=[0, 0, 32, 32], out_shape=(16, 16)
        )
        np.testing.assert_array_equal(
            via_list,
            via_object,
            err_msg="list and Window forms must decimate identically",
        )

    def test_bbox_composes_with_out_shape(self, ramp_dataset):
        """A bbox sub-window decimates the same pixels the native bbox read sees.

        Test scenario:
            The ramp grid spans x 0..64, y 0..64 (origin top-left at
            (0, 64), unit cells); bbox (0, 32, 32, 64) is the top-left
            quadrant. The decimated bbox read must equal decimating the
            native-resolution bbox read of the same area.
        """
        via_bbox = ramp_dataset.read_array(
            band=0, bbox=(0.0, 32.0, 32.0, 64.0), out_shape=(16, 16)
        )
        native = ramp_dataset.read_array(band=0, bbox=(0.0, 32.0, 32.0, 64.0))
        slice_ds = Dataset.create_from_array(
            native, top_left_corner=(0, 64), cell_size=1.0, epsg=4326
        )
        np.testing.assert_array_equal(
            via_bbox,
            slice_ds.read_array(band=0, out_shape=(16, 16)),
            err_msg="bbox+out_shape must decimate the native bbox window",
        )

    def test_upsampling(self, ramp_dataset):
        """An out_shape larger than the source enlarges the array."""
        result = ramp_dataset.read_array(band=0, out_shape=(128, 128))
        assert result.shape == (128, 128), f"unexpected shape {result.shape}"

    def test_multi_band_stacks(self):
        """An all-bands decimated read returns (bands, rows, cols)."""
        base = np.arange(64 * 64, dtype="float32").reshape(64, 64)
        ds = Dataset.create_from_array(
            np.stack([base, base + 1.0]),
            top_left_corner=(0, 64),
            cell_size=1.0,
            epsg=4326,
        )
        result = ds.read_array(out_shape=(16, 16))
        assert result.shape == (2, 16, 16), f"unexpected shape {result.shape}"
        np.testing.assert_array_equal(
            result[1] - result[0],
            np.ones((16, 16), dtype="float32"),
            err_msg="per-band values must decimate independently",
        )

    def test_multi_band_window_composition(self):
        """An all-bands read composes a window with out_shape."""
        base = np.arange(64 * 64, dtype="float32").reshape(64, 64)
        ds = Dataset.create_from_array(
            np.stack([base, base + 1.0]),
            top_left_corner=(0, 64),
            cell_size=1.0,
            epsg=4326,
        )
        result = ds.read_array(window=Window(0, 0, 32, 32), out_shape=(8, 8))
        assert result.shape == (2, 8, 8), f"unexpected shape {result.shape}"
        single = ds.read_array(band=0, window=Window(0, 0, 32, 32), out_shape=(8, 8))
        np.testing.assert_array_equal(
            result[0],
            single,
            err_msg="stacked band 0 must equal the single-band windowed read",
        )

    def test_overview_backed_read(self, ramp_dataset, tmp_path):
        """A raster with overviews serves a matching decimated read.

        Test scenario:
            With a level-2 overview present, out_shape at exactly half
            resolution matches the overview's pixels (GDAL pulls from it).
        """
        path = str(tmp_path / "ovr.tif")
        ramp_dataset.to_file(path)
        writable = Dataset.read_file(path, read_only=False)
        writable.create_overviews(overview_levels=[2])
        writable = None
        ds = Dataset.read_file(path)
        decimated = ds.read_array(band=0, out_shape=(32, 32))
        overview = ds.get_overview(band=0, overview_index=0).ReadAsArray()
        np.testing.assert_array_equal(
            decimated,
            overview,
            err_msg="half-resolution read must come from the level-2 overview",
        )

    def test_case_insensitive_resampling(self, ramp_dataset):
        """The resampling name is normalised before lookup."""
        result = ramp_dataset.read_array(
            band=0, out_shape=(32, 32), resampling=" Average "
        )
        assert result.shape == (32, 32), "normalised name must be accepted"

    @pytest.mark.parametrize(
        "bad_shape",
        [(0, 5), (5, -1), (5,), "32", (True, True), (None, 4), (32.5, 16), ("3", "2")],
    )
    def test_invalid_out_shape_rejected(self, ramp_dataset, bad_shape):
        """Malformed out_shape values raise ValueError.

        Bools, None, floats (silent truncation), and numeric strings are
        rejected alongside non-positive sizes and wrong-length inputs.

        Args:
            bad_shape: The invalid shape under test.
        """
        with pytest.raises(ValueError, match="out_shape"):
            ramp_dataset.read_array(band=0, out_shape=bad_shape)

    def test_short_list_window_rejected(self, ramp_dataset):
        """A bare window list of the wrong length raises ValueError."""
        with pytest.raises(ValueError, match="window"):
            ramp_dataset.read_array(band=0, window=[0, 0, 32], out_shape=(16, 16))

    def test_out_of_bounds_window_raises(self, ramp_dataset):
        """A window beyond the raster raises OutOfBoundsError, not GDAL's error."""
        window = Window(48, 48, 32, 32)
        with pytest.raises(OutOfBoundsError):
            ramp_dataset.read_array(band=0, window=window, out_shape=(8, 8))

    def test_band_index_validated(self, ramp_dataset):
        """An out-of-range band raises the shared band-index ValueError."""
        with pytest.raises(ValueError, match="band index"):
            ramp_dataset.read_array(band=3, out_shape=(8, 8))

    def test_unknown_resampling_rejected(self, ramp_dataset):
        """An unregistered algorithm raises ValueError listing the choices."""
        with pytest.raises(ValueError, match="unknown resampling"):
            ramp_dataset.read_array(band=0, out_shape=(8, 8), resampling="sinc")

    def test_chunks_with_out_shape_rejected(self, ramp_dataset):
        """out_shape with chunks= raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="out_shape"):
            ramp_dataset.read_array(band=0, chunks=8, out_shape=(8, 8))

    def test_resampling_without_out_shape_rejected(self, ramp_dataset):
        """A non-default resampling without out_shape raises, not silently ignored (L13)."""
        with pytest.raises(ValueError, match="only applies to out_shape"):
            ramp_dataset.read_array(band=0, resampling="bilinear")

    def test_default_resolution_unchanged(self, ramp_dataset):
        """out_shape=None keeps the historical native-resolution read."""
        result = ramp_dataset.read_array(band=0)
        assert result.shape == (64, 64), "default read must stay native-resolution"
