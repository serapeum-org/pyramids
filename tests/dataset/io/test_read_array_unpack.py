"""`read_array` applies per-band GDAL scale/offset, and `unpack=False` reaches past it.

Added for #1031, when the transform was opt-in behind `scaled=True`. #1124 made it the
default and renamed the keyword to `unpack`, so what these pin now is the other way round:
the physical values come back without being asked for, an unpacked band still costs
nothing, and `unpack=False` is the way to the stored counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


def _make(array: np.ndarray, scale=None, offset=None, no_data=None) -> Dataset:
    """Build an in-memory Dataset, optionally with per-band scale/offset/no-data."""
    ds = Dataset.from_array(
        array,
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
    )
    if scale is not None:
        ds.scale = scale
    if offset is not None:
        ds.offset = offset
    if no_data is not None:
        ds.no_data_value = no_data
    return ds


@pytest.fixture
def packed_single() -> Dataset:
    """A 1-band int16 raster with scale 0.1 and offset 5.0."""
    return _make(np.array([[0, 1], [2, 3]], dtype="int16"), scale=[0.1], offset=[5.0])


@pytest.fixture
def packed_multi() -> Dataset:
    """A 3-band int16 raster with per-band differing scale/offset."""
    arr = np.arange(3 * 4).reshape(3, 2, 2).astype("int16")
    return _make(arr, scale=[0.1, 2.0, 1.0], offset=[0.0, -1.0, 0.0])


class TestReadArrayUnpack:
    """`IO.read_array`'s unpacking and its `_apply_scale_offset` helper."""

    def test_single_band_unpacks_by_default(self, packed_single):
        """A packed single-band read returns `raw * scale + offset` as float64."""
        raw = packed_single.read_array(band=0, unpack=False)
        out = packed_single.read_array(band=0)
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        np.testing.assert_allclose(out, raw * 0.1 + 5.0)

    def test_identity_unset_returns_raw(self):
        """A band with no scale/offset is returned unchanged, keeping its dtype."""
        ds = _make(np.array([[0, 1]], dtype="int16"))
        out = ds.read_array(band=0)
        assert out.dtype == np.int16, "an unset band must not be promoted to float"
        np.testing.assert_array_equal(out, ds.read_array(band=0, unpack=False))

    def test_all_bands_per_band_factors(self, packed_multi):
        """An all-bands read applies each band's own scale/offset."""
        raw = packed_multi.read_array(unpack=False)
        out = packed_multi.read_array()
        assert out.shape == raw.shape
        assert out.dtype == np.float64
        expected = raw.astype(np.float64)
        expected[0] = raw[0] * 0.1 + 0.0
        expected[1] = raw[1] * 2.0 - 1.0
        expected[2] = raw[2] * 1.0 + 0.0
        np.testing.assert_allclose(out, expected)

    def test_all_bands_all_unset_returns_raw(self):
        """An all-bands read where no band declares scale/offset returns raw ints."""
        ds = _make(np.arange(2 * 4).reshape(2, 2, 2).astype("int16"))
        out = ds.read_array()
        assert out.dtype == np.int16, "no promotion when nothing is declared"
        np.testing.assert_array_equal(out, ds.read_array(unpack=False))

    def test_masked_mask_preserved(self):
        """Unpacking with `masked=True` keeps the mask and transforms only the data."""
        ds = _make(
            np.array([[0, -9999], [2, 3]], dtype="int16"),
            scale=[0.1],
            offset=[5.0],
            no_data=[-9999],
        )
        out = ds.read_array(band=0, masked=True)
        assert isinstance(out, np.ma.MaskedArray), "expected a masked array"
        base_mask = ds.read_array(band=0, masked=True, unpack=False).mask
        np.testing.assert_array_equal(out.mask, base_mask)
        assert out[0, 0] == pytest.approx(5.0), "unmasked cell unpacked"

    def test_unmasked_sentinel_is_unpacked(self):
        """Without masking, the no-data sentinel is transformed like any other value."""
        ds = _make(
            np.array([[-9999, 1]], dtype="int16"),
            scale=[0.1],
            offset=[5.0],
            no_data=[-9999],
        )
        out = ds.read_array(band=0)
        assert out[0, 0] == pytest.approx(-9999 * 0.1 + 5.0), "sentinel unpacked"

    def test_out_shape_then_unpacked(self, packed_single):
        """A decimated (out_shape) read is unpacked to float64 at the requested size."""
        out = packed_single.read_array(band=0, out_shape=(1, 1))
        assert out.shape == (1, 1)
        assert out.dtype == np.float64

    def test_window_then_unpacked(self, packed_single):
        """A windowed read is unpacked over just the window."""
        out = packed_single.read_array(band=0, window=[0, 0, 2, 1])
        raw = packed_single.read_array(band=0, window=[0, 0, 2, 1], unpack=False)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, raw * 0.1 + 5.0)

    def test_boundless_unpacked(self, packed_single):
        """A boundless read unpacks the whole padded window (fill included)."""
        out = packed_single.read_array(
            band=0, window=[-1, -1, 2, 2], boundless=True, fill_value=0
        )
        assert out.dtype == np.float64
        assert out.shape == (2, 2)

    def test_masked_3d_unpacked(self, packed_multi):
        """An all-bands masked read keeps the 3-D mask and unpacks the data."""
        packed_multi.no_data_value = [0, 0, 0]
        out = packed_multi.read_array(masked=True)
        assert isinstance(out, np.ma.MaskedArray)
        assert out.ndim == 3
        assert out.dtype == np.float64
        np.testing.assert_array_equal(
            out.mask, packed_multi.read_array(masked=True, unpack=False).mask
        )

    def test_threadsafe_unpacked(self, packed_single, tmp_path):
        """A threadsafe read returns unpacked float64."""
        path = tmp_path / "ts.tif"
        packed_single.to_file(str(path))
        ds = Dataset.read_file(str(path))
        out = ds.read_array(band=0, threadsafe=True)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, ds.read_array(band=0, unpack=False) * 0.1 + 5.0)

    def test_unpack_false_reaches_the_stored_counts(self, packed_single, packed_multi):
        """`unpack=False` answers in the band's own dtype, below the transform.

        Test scenario:
            The escape hatch has to be a real one: not merely a different number but the
            stored integer, so a caller who needs the bytes -- a copy, a checksum, a
            re-write -- can still reach them now that unpacking is the default.
        """
        for ds in (packed_single, packed_multi):
            raw = ds.read_array(unpack=False)
            assert raw.dtype == np.int16, f"expected the stored dtype, got {raw.dtype}"
            assert not np.allclose(raw, ds.read_array()), (
                "unpack=False returned the unpacked values"
            )

    def test_lazy_matches_eager(self, packed_multi, tmp_path):
        """A lazy (chunks) read computes to the same values as the eager one."""
        pytest.importorskip("dask.array")
        # The chunked path reopens the source by path, so back it with a real file.
        path = tmp_path / "packed.tif"
        packed_multi.to_file(str(path))
        ds = Dataset.read_file(str(path))
        lazy = ds.read_array(chunks="auto")
        assert hasattr(lazy, "compute"), "chunks= must return a lazy array"
        np.testing.assert_allclose(lazy.compute(), ds.read_array())
