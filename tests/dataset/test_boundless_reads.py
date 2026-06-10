"""Tests for boundless windowed reads with fill values.

Covers `read_array(window=..., boundless=True, fill_value=...)`: edge and
corner hang-offs, fully-outside windows, fill precedence (explicit >
no-data > dtype zero), multi-band stacking, the Window/list forms, and the
validation contract.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.base._errors import OutOfBoundsError
from pyramids.dataset import Dataset, Window

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def ramp_dataset() -> Dataset:
    """A 6x6 float32 ramp with nodata -9999 on a unit grid.

    Returns:
        Dataset: Single-band in-memory dataset, value == row*6 + col.
    """
    arr = np.arange(36, dtype="float32").reshape(6, 6)
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326, no_data_value=-9999.0
    )


class TestBoundlessReads:
    """read_array(boundless=True) behaviour."""

    def test_top_left_corner_hangoff(self, ramp_dataset):
        """A window hanging off the top-left keeps its shape; outside is nodata.

        Test scenario:
            Window(-2, -2, 4, 4): rows/cols 0-1 are fill, the 2x2 inside part
            equals the raster's top-left corner.
        """
        arr = ramp_dataset.read_array(band=0)
        result = ramp_dataset.read_array(
            band=0, window=Window(-2, -2, 4, 4), boundless=True
        )
        assert result.shape == (4, 4), f"shape must stay (4, 4), got {result.shape}"
        assert np.isclose(result[:2, :], -9999.0).all(), "rows above must be fill"
        assert np.isclose(result[:, :2], -9999.0).all(), "cols left must be fill"
        np.testing.assert_array_equal(
            result[2:, 2:], arr[:2, :2], err_msg="inside part must be real data"
        )

    def test_bottom_right_edge_hangoff(self, ramp_dataset):
        """A window past the bottom-right is filled beyond the data."""
        arr = ramp_dataset.read_array(band=0)
        result = ramp_dataset.read_array(
            band=0, window=Window(4, 4, 4, 4), boundless=True
        )
        np.testing.assert_array_equal(
            result[:2, :2], arr[4:, 4:], err_msg="inside corner must be real data"
        )
        assert np.isclose(result[2:, :], -9999.0).all(), "rows below must be fill"
        assert np.isclose(result[:, 2:], -9999.0).all(), "cols right must be fill"

    def test_fully_outside_window_is_all_fill(self, ramp_dataset):
        """A window with no raster overlap returns pure fill."""
        result = ramp_dataset.read_array(
            band=0, window=Window(10, 10, 2, 2), boundless=True
        )
        assert np.isclose(result, -9999.0).all(), "disjoint window must be all fill"

    def test_explicit_fill_value_wins(self, ramp_dataset):
        """fill_value= overrides the band's no-data value."""
        result = ramp_dataset.read_array(
            band=0, window=Window(-1, -1, 3, 3), boundless=True, fill_value=7.0
        )
        assert np.isclose(result[0, :], 7.0).all(), "explicit fill must win over nodata"

    def test_multi_band_boundless(self):
        """An all-bands boundless read stacks filled planes per band."""
        base = np.arange(36, dtype="float32").reshape(6, 6)
        ds = Dataset.create_from_array(
            np.stack([base, base + 100.0]),
            top_left_corner=(0, 6),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.read_array(window=Window(-1, -1, 3, 3), boundless=True)
        assert result.shape == (2, 3, 3), f"3-D shape wrong: {result.shape}"
        assert result[0, 1, 1] == pytest.approx(base[0, 0]), "band-0 inside value wrong"
        assert result[1, 1, 1] == pytest.approx(
            base[0, 0] + 100.0
        ), "band-1 inside value wrong"
        assert np.isclose(result[:, 0, :], -9999.0).all(), "outside rows must be fill"

    def test_legacy_list_window_form(self, ramp_dataset):
        """The x-first list window form works boundlessly too."""
        via_list = ramp_dataset.read_array(
            band=0, window=[-2, -2, 4, 4], boundless=True
        )
        via_window = ramp_dataset.read_array(
            band=0, window=Window(-2, -2, 4, 4), boundless=True
        )
        np.testing.assert_array_equal(
            via_list, via_window, err_msg="list and Window forms must agree"
        )

    def test_in_bounds_window_matches_normal_read(self, ramp_dataset):
        """boundless=True on an in-bounds window equals the normal read."""
        window = Window(1, 1, 3, 3)
        np.testing.assert_array_equal(
            ramp_dataset.read_array(band=0, window=window, boundless=True),
            ramp_dataset.read_array(band=0, window=window),
            err_msg="in-bounds boundless read must be identical",
        )

    def test_default_out_of_range_still_raises(self, ramp_dataset):
        """Without boundless=True, an out-of-range window keeps raising."""
        with pytest.raises(OutOfBoundsError):
            ramp_dataset.read_array(band=0, window=[4, 4, 4, 4])

    def test_boundless_without_window_rejected(self, ramp_dataset):
        """boundless=True without a window raises a clear ValueError."""
        with pytest.raises(ValueError, match="requires a window"):
            ramp_dataset.read_array(band=0, boundless=True)

    def test_geometry_window_rejected(self, ramp_dataset):
        """A GeoDataFrame window with boundless=True is rejected."""
        gdf = gpd.GeoDataFrame(geometry=[box(1.0, 1.0, 3.0, 3.0)], crs=4326)
        with pytest.raises(ValueError, match="pixel window"):
            ramp_dataset.read_array(band=0, window=gdf, boundless=True)

    def test_bbox_with_boundless_rejected(self, ramp_dataset):
        """A bbox (a geometry window internally) with boundless=True is rejected."""
        with pytest.raises(ValueError, match="pixel window"):
            ramp_dataset.read_array(band=0, bbox=(1.0, 1.0, 3.0, 3.0), boundless=True)

    def test_dtype_zero_fallback_without_nodata(self):
        """No fill_value and no band nodata falls back to the dtype's zero."""
        arr = np.arange(36, dtype="uint8").reshape(6, 6)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326, no_data_value=None
        )
        result = ds.read_array(band=0, window=Window(-1, -1, 3, 3), boundless=True)
        assert result.dtype == np.uint8, f"band dtype must be kept, got {result.dtype}"
        assert (result[0, :] == 0).all(), "outside row must fall back to dtype zero"
        np.testing.assert_array_equal(
            result[1:, 1:], arr[:2, :2], err_msg="inside part must be real data"
        )

    def test_unrepresentable_fill_value_rejected(self):
        """A fill the integer band dtype cannot hold raises instead of wrapping."""
        arr = np.arange(36, dtype="uint8").reshape(6, 6)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326, no_data_value=None
        )
        for bad_fill in (-9999.0, -9999, 0.5, float("nan")):
            with pytest.raises(ValueError, match="not representable"):
                ds.read_array(
                    band=0,
                    window=Window(-1, -1, 3, 3),
                    boundless=True,
                    fill_value=bad_fill,
                )

    def test_nan_fill_value_on_float_band(self, ramp_dataset):
        """NaN is a valid explicit fill for a float band."""
        result = ramp_dataset.read_array(
            band=0,
            window=Window(-1, -1, 3, 3),
            boundless=True,
            fill_value=float("nan"),
        )
        assert np.isnan(result[0, :]).all(), "outside row must be NaN"
        assert not np.isnan(result[1:, 1:]).any(), "inside part must stay real data"

    def test_single_band_dataset_band_none_is_2d(self, ramp_dataset):
        """band=None on a single-band dataset returns a 2-D filled plane."""
        result = ramp_dataset.read_array(window=Window(-1, -1, 3, 3), boundless=True)
        assert result.shape == (3, 3), f"single band must stay 2-D, got {result.shape}"
        assert np.isclose(result[0, :], -9999.0).all(), "outside row must be fill"

    def test_fill_value_without_boundless_rejected(self, ramp_dataset):
        """fill_value without boundless=True raises instead of being ignored."""
        with pytest.raises(ValueError, match="boundless"):
            ramp_dataset.read_array(band=0, window=[1, 1, 3, 3], fill_value=7.0)

    def test_boundless_with_chunks_rejected(self, ramp_dataset):
        """boundless=True with chunks raises instead of being ignored."""
        with pytest.raises(ValueError, match="not supported"):
            ramp_dataset.read_array(band=0, chunks="auto", boundless=True)
