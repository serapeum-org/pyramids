"""`Dataset.bands.select` band subsetting into a new Dataset (#1032)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

MULTI_BANDS = "tests/data/geotiff/multi_bands.tif"


@pytest.fixture
def rich() -> Dataset:
    """A 4-band raster carrying per-band names/units/scale/offset/no-data/metadata."""
    arr = np.arange(4 * 4).reshape(4, 2, 2).astype("float32")
    ds = Dataset.create_from_array(
        arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
    )
    ds.band_names = ["a", "b", "c", "d"]
    ds.band_units = ["m", "s", "kg", "K"]
    ds.scale = [0.1, 0.2, 0.3, 0.4]
    ds.offset = [1.0, 2.0, 3.0, 4.0]
    ds.no_data_value = [-1.0, -2.0, -3.0, -4.0]
    return ds


class TestSelect:
    """`Bands.select` — construction, ordering, and error handling."""

    def test_select_by_index(self):
        """Selecting 1-based indices yields those bands' pixels in order."""
        ds = Dataset.read_file(MULTI_BANDS)
        sub = ds.bands.select([1, 3])
        assert type(sub) is Dataset and sub.band_count == 2
        np.testing.assert_array_equal(sub.read_array(band=0), ds.read_array(band=0))
        np.testing.assert_array_equal(sub.read_array(band=1), ds.read_array(band=2))

    def test_select_by_name(self, rich):
        """Selecting by band name resolves to the right bands."""
        sub = rich.bands.select(["c", "a"])
        assert sub.band_names == ["c", "a"], sub.band_names

    def test_reorder(self, rich):
        """Selection preserves the requested order (subset + reorder)."""
        sub = rich.bands.select([4, 1])
        assert sub.band_names == ["d", "a"], sub.band_names

    def test_single_band(self):
        """Selecting one band yields a 1-band Dataset."""
        ds = Dataset.read_file(MULTI_BANDS)
        assert ds.bands.select([2]).band_count == 1

    def test_duplicates_allowed(self, rich):
        """Duplicate selectors are allowed (e.g. gray -> RGB expansion)."""
        assert rich.bands.select([1, 1, 2]).band_count == 3

    def test_facade_parity(self, rich):
        """`Dataset.select_bands` mirrors `Dataset.bands.select`."""
        a = rich.select_bands([1, 2]).band_names
        b = rich.bands.select([1, 2]).band_names
        assert a == b == ["a", "b"]


class TestSelectCarryAcross:
    """`Bands.select` carries per-band state across the copy."""

    def test_carries_all_per_band_state(self, rich):
        """Names, units, scale, offset, and no-data follow the selected bands."""
        sub = rich.select_bands([4, 2])
        assert sub.band_names == ["d", "b"]
        assert sub.band_units == ["K", "s"], "units must survive the MEM re-apply"
        assert sub.scale == [0.4, 0.2]
        assert sub.offset == [4.0, 2.0]
        assert list(sub.no_data_value) == [-4.0, -2.0]

    def test_carries_metadata(self, rich):
        """Band metadata follows the selected bands."""
        rich.bands.set_metadata_item("k", "v0", band=0)
        rich.bands.set_metadata_item("k", "v3", band=3)
        sub = rich.select_bands([4, 1])
        assert sub.band_meta_data[0].get("k") == "v3"
        assert sub.band_meta_data[1].get("k") == "v0"

    def test_carries_attribute_table(self):
        """The raster attribute table (category names) survives selection (#1024)."""
        ds = Dataset.create_from_array(
            np.array([[0, 1], [1, 0]], dtype="int32"),
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
        )
        ds.set_attribute_table(
            pd.DataFrame({"value": [0, 1], "label": ["sea", "land"]}), band=0
        )
        rat = ds.bands.select([1]).get_attribute_table(band=0)
        assert rat is not None and "label" in rat.columns, "RAT must survive select"


class TestSelectErrors:
    """`Bands.select` rejects invalid selectors before any GDAL call."""

    @pytest.mark.parametrize(
        "bands, exc",
        [
            ([99], ValueError),
            (["nope"], ValueError),
            ([], ValueError),
            ("a", TypeError),
            (1, TypeError),
            ([True], TypeError),
            ([1.5], TypeError),
        ],
    )
    def test_invalid_selectors_raise(self, rich, bands, exc):
        """Out-of-range, unknown-name, empty, non-list, bool, and float raise."""
        with pytest.raises(exc):
            rich.bands.select(bands)


class TestSelectLazy:
    """`Bands.select(lazy=...)` eager vs VRT-backed."""

    def test_lazy_reads_same_pixels(self):
        """A lazy selection reads the same pixels as the eager one."""
        ds = Dataset.read_file(MULTI_BANDS)
        eager = ds.bands.select([2, 1])
        lazy = ds.bands.select([2, 1], lazy=True)
        assert lazy.band_count == 2
        np.testing.assert_array_equal(lazy.read_array(), eager.read_array())


class TestSelectNetCDF:
    """A NetCDF variable subset returns a base Dataset, not a NetCDF."""

    def test_netcdf_variable_returns_base_dataset(self):
        """Selecting bands of a NetCDF variable yields a plain raster."""
        from pyramids.netcdf import NetCDF

        var = NetCDF.read_file(
            "tests/data/netcdf/none__4v__1d1-2d2-3d1__curv.nc"
        ).get_variable("Tair")
        sub = var.bands.select([1])
        assert type(sub) is Dataset, f"expected a base Dataset, got {type(sub)}"
        assert sub.band_count == 1
