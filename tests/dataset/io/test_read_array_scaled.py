"""`read_array(scaled=True)` applies per-band GDAL scale/offset (#1031)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


def _make(array: np.ndarray, scale=None, offset=None, no_data=None) -> Dataset:
    """Build an in-memory Dataset, optionally with per-band scale/offset/no-data."""
    ds = Dataset.create_from_array(
        array, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
    )
    if scale is not None:
        ds.scale = scale
    if offset is not None:
        ds.offset = offset
    if no_data is not None:
        ds.no_data_value = no_data
    return ds


@pytest.fixture
def scaled_single() -> Dataset:
    """A 1-band int16 raster with scale 0.1 and offset 5.0."""
    return _make(np.array([[0, 1], [2, 3]], dtype="int16"), scale=[0.1], offset=[5.0])


@pytest.fixture
def scaled_multi() -> Dataset:
    """A 3-band int16 raster with per-band differing scale/offset."""
    arr = np.arange(3 * 4).reshape(3, 2, 2).astype("int16")
    return _make(arr, scale=[0.1, 2.0, 1.0], offset=[0.0, -1.0, 0.0])


class TestReadArrayScaled:
    """`IO.read_array(scaled=True)` and its `_apply_scale_offset` helper."""

    def test_single_band_scaled(self, scaled_single):
        """A scaled single-band read returns `raw * scale + offset` as float64."""
        raw = scaled_single.read_array(band=0)
        out = scaled_single.read_array(band=0, scaled=True)
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        np.testing.assert_allclose(out, raw * 0.1 + 5.0)

    def test_identity_unset_returns_raw(self):
        """A band with no scale/offset is returned unchanged, keeping its dtype."""
        ds = _make(np.array([[0, 1]], dtype="int16"))
        out = ds.read_array(band=0, scaled=True)
        assert out.dtype == np.int16, "an unset band must not be promoted to float"
        np.testing.assert_array_equal(out, ds.read_array(band=0))

    def test_all_bands_per_band_factors(self, scaled_multi):
        """An all-bands scaled read applies each band's own scale/offset."""
        raw = scaled_multi.read_array()
        out = scaled_multi.read_array(scaled=True)
        assert out.shape == raw.shape and out.dtype == np.float64
        expected = raw.astype(np.float64)
        expected[0] = raw[0] * 0.1 + 0.0
        expected[1] = raw[1] * 2.0 - 1.0
        expected[2] = raw[2] * 1.0 + 0.0
        np.testing.assert_allclose(out, expected)

    def test_all_bands_all_unset_returns_raw(self):
        """An all-bands read where no band declares scale/offset returns raw ints."""
        ds = _make(np.arange(2 * 4).reshape(2, 2, 2).astype("int16"))
        out = ds.read_array(scaled=True)
        assert out.dtype == np.int16, "no promotion when nothing is declared"
        np.testing.assert_array_equal(out, ds.read_array())

    def test_masked_mask_preserved(self):
        """`scaled=True` with `masked=True` keeps the mask and scales only data."""
        ds = _make(
            np.array([[0, -9999], [2, 3]], dtype="int16"),
            scale=[0.1],
            offset=[5.0],
            no_data=[-9999],
        )
        out = ds.read_array(band=0, masked=True, scaled=True)
        assert isinstance(out, np.ma.MaskedArray), "expected a masked array"
        base_mask = ds.read_array(band=0, masked=True).mask
        np.testing.assert_array_equal(out.mask, base_mask)
        assert out[0, 0] == pytest.approx(5.0), "unmasked cell scaled"

    def test_unmasked_sentinel_is_scaled(self):
        """Without masking, the no-data sentinel is scaled like any other value."""
        ds = _make(
            np.array([[-9999, 1]], dtype="int16"),
            scale=[0.1],
            offset=[5.0],
            no_data=[-9999],
        )
        out = ds.read_array(band=0, scaled=True)
        assert out[0, 0] == pytest.approx(-9999 * 0.1 + 5.0), "sentinel scaled"

    def test_out_shape_then_scaled(self, scaled_single):
        """A decimated (out_shape) read is scaled to float64 at the requested size."""
        out = scaled_single.read_array(band=0, out_shape=(1, 1), scaled=True)
        assert out.shape == (1, 1) and out.dtype == np.float64

    def test_window_then_scaled(self, scaled_single):
        """A windowed read is scaled over just the window."""
        out = scaled_single.read_array(band=0, window=[0, 0, 2, 1], scaled=True)
        raw = scaled_single.read_array(band=0, window=[0, 0, 2, 1])
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, raw * 0.1 + 5.0)

    def test_boundless_scaled(self, scaled_single):
        """A boundless scaled read scales the whole padded window (fill included)."""
        out = scaled_single.read_array(
            band=0, window=[-1, -1, 2, 2], boundless=True, fill_value=0, scaled=True
        )
        assert out.dtype == np.float64 and out.shape == (2, 2)

    def test_masked_3d_scaled(self, scaled_multi):
        """An all-bands masked scaled read keeps the 3-D mask and scales the data."""
        scaled_multi.no_data_value = [0, 0, 0]
        out = scaled_multi.read_array(masked=True, scaled=True)
        assert isinstance(out, np.ma.MaskedArray) and out.ndim == 3
        assert out.dtype == np.float64
        np.testing.assert_array_equal(
            out.mask, scaled_multi.read_array(masked=True).mask
        )

    def test_threadsafe_scaled(self, scaled_single, tmp_path):
        """A threadsafe scaled read returns scaled float64."""
        path = tmp_path / "ts.tif"
        scaled_single.to_file(str(path))
        ds = Dataset.read_file(str(path))
        out = ds.read_array(band=0, threadsafe=True, scaled=True)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, ds.read_array(band=0) * 0.1 + 5.0)

    def test_scaled_false_is_unchanged(self, scaled_single, scaled_multi):
        """`scaled=False` (default) is byte-identical to a plain read."""
        for ds in (scaled_single, scaled_multi):
            np.testing.assert_array_equal(ds.read_array(scaled=False), ds.read_array())

    def test_lazy_scaled_matches_eager(self, scaled_multi, tmp_path):
        """A lazy (chunks) scaled read computes to the same values as the eager one."""
        pytest.importorskip("dask.array")
        # The chunked path reopens the source by path, so back it with a real file.
        path = tmp_path / "scaled.tif"
        scaled_multi.to_file(str(path))
        ds = Dataset.read_file(str(path))
        lazy = ds.read_array(chunks="auto", scaled=True)
        assert hasattr(lazy, "compute"), "chunks= must return a lazy array"
        np.testing.assert_allclose(lazy.compute(), ds.read_array(scaled=True))
