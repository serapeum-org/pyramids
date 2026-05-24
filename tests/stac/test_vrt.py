"""Unit tests for pyramids.stac._vrt.build_vrt_from_stac (PB-5).

Builds a lazy GDAL VRT mosaic over one STAC asset across items. Tests use local
GeoTIFF tiles (no network); the VRT references them and reads lazily.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.stac._vrt import build_vrt_from_stac

pytestmark = pytest.mark.core


def _write(path, arr, top_left, *, cell_size=1.0, epsg=4326, nodata=-9999.0):
    """Write `arr` to `path` as a GeoTIFF and return the path string."""
    ds = Dataset.create_from_array(
        arr, top_left_corner=top_left, cell_size=cell_size, epsg=epsg, no_data_value=nodata
    )
    ds.to_file(str(path))
    return str(path)


@pytest.fixture
def adjacent_tiles(tmp_path):
    """Two 4x4 EPSG:4326 tiles abutting horizontally (union is 4x8).

    Tile A (value 10) covers columns 0..3; tile B (value 20) columns 4..7.

    Returns:
        list[dict]: two raw STAC items each exposing a "data" asset.
    """
    a = np.full((4, 4), 10.0, dtype="float32")
    b = np.full((4, 4), 20.0, dtype="float32")
    pa = _write(tmp_path / "a.tif", a, (0.0, 4.0))
    pb = _write(tmp_path / "b.tif", b, (4.0, 4.0))
    return [{"assets": {"data": {"href": pa}}}, {"assets": {"data": {"href": pb}}}]


class TestBuildVrtFromStac:
    """Tests for build_vrt_from_stac."""

    def test_returns_dataset_over_union(self, adjacent_tiles):
        """The VRT mosaics the two tiles into one Dataset over their union.

        Test scenario:
            Two 4x4 tiles abutting horizontally -> a 4x8 single-band Dataset.
        """
        ds = build_vrt_from_stac(adjacent_tiles, asset="data")
        assert isinstance(ds, Dataset), f"expected a Dataset, got {type(ds)}"
        arr = ds.read_array()
        assert arr.shape == (4, 8), f"expected union shape (4, 8), got {arr.shape}"

    def test_mosaic_values_lazy_read(self, adjacent_tiles):
        """Reading the VRT pulls each source's pixels into the right place.

        Test scenario:
            Left half reads as tile A (10), right half as tile B (20).
        """
        ds = build_vrt_from_stac(adjacent_tiles, asset="data")
        arr = ds.read_array()
        assert float(arr[0, 0]) == 10.0, f"left half should be tile A=10, got {arr[0, 0]}"
        assert float(arr[0, 7]) == 20.0, f"right half should be tile B=20, got {arr[0, 7]}"

    def test_separate_stacks_bands(self, tmp_path):
        """separate=True makes one band per source (same-grid sources).

        Test scenario:
            Two same-grid tiles -> a 2-band VRT.
        """
        a = np.full((3, 3), 1.0, dtype="float32")
        b = np.full((3, 3), 2.0, dtype="float32")
        pa = _write(tmp_path / "sa.tif", a, (0.0, 3.0))
        pb = _write(tmp_path / "sb.tif", b, (0.0, 3.0))
        items = [{"assets": {"d": {"href": pa}}}, {"assets": {"d": {"href": pb}}}]
        ds = build_vrt_from_stac(items, asset="d", separate=True)
        assert ds.band_count == 2, f"separate=True should give 2 bands, got {ds.band_count}"

    def test_signer_applied_to_each_source(self, adjacent_tiles):
        """signer.sign_href is applied to every source href before the build.

        Test scenario:
            An identity-recording signer sees both source hrefs; the build still
            succeeds on the local files.
        """

        class _RecordingSigner:
            def __init__(self):
                self.seen = []

            def sign_href(self, href):
                self.seen.append(href)
                return href

            def gdal_env(self):
                return {}

        signer = _RecordingSigner()
        build_vrt_from_stac(adjacent_tiles, asset="data", signer=signer)
        assert len(signer.seen) == 2, f"sign_href should fire per source, got {signer.seen}"

    def test_empty_items_raises(self):
        """No items raises a clear ValueError.

        Test scenario:
            An empty iterable cannot build a mosaic.
        """
        with pytest.raises(ValueError, match="no items"):
            build_vrt_from_stac([], asset="data")

    def test_missing_asset_raises(self, adjacent_tiles):
        """A missing asset surfaces StacAssetError from resolved_href.

        Test scenario:
            Requesting an absent asset key fails before the build.
        """
        from pyramids.base._errors import StacAssetError

        with pytest.raises(StacAssetError, match="not found"):
            build_vrt_from_stac(adjacent_tiles, asset="nope")
