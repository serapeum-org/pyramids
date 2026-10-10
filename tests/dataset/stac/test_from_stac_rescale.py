"""Tests for from_stac(rescale=...) and from_stac(cfg=...) (STAC-04 / STAC-07)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, DatasetCollection, Grid
from pyramids.dataset._stac import from_stac
from pyramids.stac._config import AssetMetadataWarning

pytestmark = pytest.mark.core

_PHYSICAL = [[np.nan, 1.0], [2.0, 3.0]]


def _timestep(collection, index=0):
    """Return one timestep of a cube as a 2-D float array.

    Args:
        collection: The built :class:`DatasetCollection`.
        index: Which timestep to read.

    Returns:
        np.ndarray: The timestep's first band as floats.
    """
    values = np.asarray(collection.iloc(index).read_array(), dtype="float64")
    return values if values.ndim == 2 else values[0]


def _assert_physical(values, label):
    """Assert a timestep carries the packed fixture's physical values.

    Args:
        values: A 2-D array of the timestep's pixels.
        label: What is being checked, for the assertion message.
    """
    assert np.isnan(values[0, 0]), f"{label}: no-data must stay no-data, got {values}"
    assert values[1, 1] == pytest.approx(3.0), f"{label}: 300 * 0.01 != {values[1, 1]}"
    assert values[0, 1] == pytest.approx(1.0), f"{label}: 100 * 0.01 != {values[0, 1]}"


@pytest.fixture
def packed_items(packed_item):
    """Two copies of the packed item, so the cube has two timesteps."""
    second = {**packed_item, "id": "packed-2"}
    second["properties"] = {"datetime": "2023-06-02T00:00:00Z"}
    return [packed_item, second]


@pytest.fixture
def nodataless_item(tmp_path):
    """An item whose asset declares neither a no-data value nor a unit."""
    path = str(tmp_path / "nodataless.tif")
    Dataset.from_array(
        np.array([[1, 2], [3, 4]], dtype="int16"),
        no_data_value=None,
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return {
        "type": "Feature",
        "id": "thin",
        "collection": "c",
        "bbox": [0.0, 0.0, 2.0, 2.0],
        "properties": {"datetime": "2023-06-01T00:00:00Z"},
        "assets": {"B05": {"href": path, "type": "image/tiff"}},
        "stac_extensions": [],
    }


class TestSingleAssetRescale:
    """Tests for the single-asset path under `rescale=True`."""

    def test_default_keeps_stored_counts(self, packed_items):
        """Without `rescale` the cube still carries the stored counts.

        Test scenario:
            Two packed items, read with today's defaults.
        """
        collection = from_stac(packed_items, asset="data")
        values = _timestep(collection)
        assert values[1, 1] == pytest.approx(300.0), (
            f"rescale=False must keep the stored counts, got {values}"
        )

    def test_rescale_gives_physical_values(self, packed_items):
        """`rescale=True` applies the asset's raster:bands packing per timestep.

        Test scenario:
            Two packed items declaring scale 0.01.
        """
        collection = from_stac(packed_items, asset="data", rescale=True)
        assert collection.time_length == 2, (
            f"expected 2 timesteps, got {collection.time_length}"
        )
        _assert_physical(_timestep(collection, 0), "timestep 0")
        _assert_physical(_timestep(collection, 1), "timestep 1")

    def test_rescale_materialises_off_the_asset_url(
        self, packed_items, packed_raster_path
    ):
        """A rescaled timestep is backed by a local copy, not the asset href.

        Test scenario:
            The cube's backing files are compared against the asset path.
        """
        collection = from_stac(packed_items, asset="data", rescale=True)
        backing = str(collection.iloc(0).file_name)
        assert packed_raster_path not in backing, (
            f"a rescaled timestep must not be backed by the raw asset: {backing}"
        )

    def test_rescale_keeps_the_url_when_nothing_is_declared(self, three_local_items):
        """An asset without raster:bands keeps its lazy href.

        Test scenario:
            Items whose assets declare no packing at all.
        """
        collection = from_stac(three_local_items, asset="data", rescale=True)
        backing = str(collection.iloc(0).file_name)
        expected = three_local_items[0]["assets"]["data"]["href"]
        assert backing.replace("/", "\\") == expected.replace("/", "\\"), (
            f"nothing to rescale must keep the asset href, got {backing}"
        )

    def test_unreadable_asset_raises_while_materialising(self, packed_item, tmp_path):
        """A rescale that cannot open its asset raises instead of staying silent.

        Test scenario:
            The item declares a scale, so the materialiser must open the asset —
            but the href points at a missing file and `errors_as_nodata` is off.
        """
        asset = {**packed_item["assets"]["data"], "href": str(tmp_path / "gone.tif")}
        item = {**packed_item, "assets": {"data": asset}}
        with pytest.raises(FileNotFoundError):
            from_stac([item], asset="data", rescale=True)

    def test_unreadable_asset_is_tolerated_under_errors_as_nodata(
        self, packed_item, tmp_path
    ):
        """With errors_as_nodata the failed materialisation is left to the caller.

        Test scenario:
            The second item's href is missing; the materialiser stays silent and
            the timestep is substituted with a no-data plane instead.
        """
        asset = {**packed_item["assets"]["data"], "href": str(tmp_path / "gone.tif")}
        second = {
            **packed_item,
            "id": "packed-2",
            "properties": {"datetime": "2023-06-02T00:00:00Z"},
            "assets": {"data": asset},
        }
        with pytest.warns(RuntimeWarning, match="errors_as_nodata"):
            collection = from_stac(
                [packed_item, second],
                asset="data",
                rescale=True,
                errors_as_nodata=True,
            )
        assert collection.time_length == 2, (
            f"the unreadable timestep should be kept, got {collection.time_length}"
        )
        assert np.isnan(_timestep(collection, 1)).all(), (
            "the substituted timestep should be entirely no-data"
        )

    def test_rescale_result_declares_identity_packing(self, packed_items):
        """A rescaled timestep carries no packing of its own.

        Test scenario:
            The backing raster's scale/offset slots.
        """
        collection = from_stac(packed_items, asset="data", rescale=True)
        timestep = collection.iloc(0)
        assert timestep.scale == [1.0], f"expected identity scale, got {timestep.scale}"
        assert timestep.offset == [0.0], (
            f"expected identity offset, got {timestep.offset}"
        )

    def test_rescale_combines_with_a_target_grid(self, packed_items):
        """`rescale` and `grid=` apply together.

        Test scenario:
            The packed items matched onto a finer 4x4 template.
        """
        template = Dataset.from_array(
            np.zeros((4, 4), dtype="float32"),
            no_data_value=np.nan,
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=0.5, epsg=4326),
        )
        collection = from_stac(
            packed_items, asset="data", rescale=True, grid=Grid(like=template)
        )
        values = _timestep(collection)
        assert values.shape == (4, 4), f"expected the template grid, got {values.shape}"
        assert np.nanmax(values) == pytest.approx(3.0), (
            f"the aligned cube must still be in physical units, got {values}"
        )


class TestGroupedRescale:
    """Tests for the grouped (mosaicking) path under `rescale=True`."""

    def test_grouped_mosaic_is_physical(self, packed_items):
        """Each group is mosaicked from physical-unit sources.

        Test scenario:
            Two items grouped by id -> two timesteps, both rescaled.
        """
        collection = from_stac(packed_items, asset="data", groupby="id", rescale=True)
        assert collection.time_length == 2, (
            f"expected 2 id groups, got {collection.time_length}"
        )
        _assert_physical(_timestep(collection, 0), "group 0")

    def test_grouped_default_is_unchanged(self, packed_items):
        """Without `rescale` the grouped mosaic keeps the stored counts.

        Test scenario:
            The same items grouped by id, no rescale.
        """
        collection = from_stac(packed_items, asset="data", groupby="id")
        values = _timestep(collection)
        assert values[1, 1] == pytest.approx(300.0), (
            f"the grouped default must keep the counts, got {values}"
        )

    def test_grouped_fuse_func_sees_physical_values(self, packed_items):
        """A `fuse_func` is handed rescaled values when `rescale=True`.

        Test scenario:
            A max-fuser over one group of two packed items.
        """

        def _fuse(dst, src):
            """Keep the larger of the two planes, in place."""
            np.copyto(dst, np.fmax(dst, src))

        collection = from_stac(
            packed_items,
            asset="data",
            groupby="solar_day",
            rescale=True,
            fuse_func=_fuse,
        )
        values = _timestep(collection)
        assert values[1, 1] == pytest.approx(3.0), (
            f"the fuser must see physical values, got {values}"
        )


class TestMultiAssetRescale:
    """Tests for the multi-asset (band-stacking) path under `rescale=True`."""

    def test_each_asset_is_rescaled_before_stacking(self, packed_raster_path):
        """Per-asset packings are applied before the bands are stacked.

        Test scenario:
            Two assets over the same counts with scales 0.01 and 10.
        """
        item = {
            "id": "two-assets",
            "bbox": [0.0, 0.0, 2.0, 2.0],
            "properties": {"datetime": "2023-06-01T00:00:00Z"},
            "assets": {
                "a": {
                    "href": packed_raster_path,
                    "type": "image/tiff",
                    "raster:bands": [{"scale": 0.01, "nodata": 0}],
                },
                "b": {
                    "href": packed_raster_path,
                    "type": "image/tiff",
                    "raster:bands": [{"scale": 10.0, "nodata": 0}],
                },
            },
        }
        collection = from_stac([item], asset=["a", "b"], rescale=True)
        timestep = collection.iloc(0)
        assert timestep.band_count == 2, (
            f"expected a 2-band timestep, got {timestep.band_count}"
        )
        values = np.asarray(timestep.read_array(), dtype="float64")
        assert values[0][1, 1] == pytest.approx(3.0), (
            f"band a must use scale 0.01, got {values[0]}"
        )
        assert values[1][1, 1] == pytest.approx(3000.0), (
            f"band b must use scale 10, got {values[1]}"
        )


class TestFromStacConfig:
    """Tests for from_stac(cfg=...) — overrides and band aliases."""

    def test_alias_names_the_asset(self, nodataless_item):
        """An alias is resolved per item before the href is looked up.

        Test scenario:
            cfg maps "rededge" -> "B05" for collection "c".
        """
        cfg = {"c": {"aliases": {"rededge": "B05"}}}
        collection = from_stac([nodataless_item], asset="rededge", cfg=cfg)
        values = _timestep(collection)
        assert values[1, 1] == pytest.approx(4.0), (
            f"the aliased asset must be the one read, got {values}"
        )

    def test_unknown_alias_still_raises_for_a_missing_asset(self, nodataless_item):
        """An unaliased, absent asset key fails as it does today.

        Test scenario:
            A key that is neither an alias nor an asset.
        """
        cfg = {"c": {"aliases": {"rededge": "B05"}}}
        with pytest.raises(KeyError):
            from_stac([nodataless_item], asset="nir", cfg=cfg)

    def test_missing_nodata_is_supplied(self, nodataless_item):
        """A cfg nodata fills the gap the item leaves.

        Test scenario:
            The asset declares no no-data; cfg supplies 1.
        """
        cfg = {"c": {"assets": {"*": {"nodata": 1}}}}
        collection = from_stac([nodataless_item], asset="B05", cfg=cfg)
        assert collection.iloc(0).no_data_value[0] == pytest.approx(1.0), (
            f"expected the configured no-data, got {collection.iloc(0).no_data_value}"
        )

    def test_wildcard_collection_applies_without_an_item_collection(
        self, three_local_items
    ):
        """The '*' cfg section reaches items that declare no collection.

        Test scenario:
            Items without a `collection` member and a '*' unit override.
        """
        cfg = {"*": {"assets": {"*": {"unit": "K"}}}}
        collection = from_stac(three_local_items, asset="data", cfg=cfg)
        assert collection.iloc(0).band_units == ["K"], (
            f"expected the configured unit, got {collection.iloc(0).band_units}"
        )

    def test_declared_value_is_not_replaced(self, packed_items):
        """A nodata the asset already declares survives, with a warning.

        Test scenario:
            The packed items declare nodata 0; cfg asks for 7.
        """
        cfg = {"*": {"assets": {"*": {"nodata": 7}}}}
        with pytest.warns(AssetMetadataWarning, match="already declares"):
            collection = from_stac(packed_items, asset="data", cfg=cfg)
        assert collection.iloc(0).no_data_value[0] == pytest.approx(0.0), (
            "an override must not replace a value the asset declares"
        )

    def test_cfg_none_is_byte_identical(self, nodataless_item):
        """Omitting cfg leaves the asset backing the timestep directly.

        Test scenario:
            The default read keeps the href as the backing file.
        """
        collection = from_stac([nodataless_item], asset="B05")
        backing = str(collection.iloc(0).file_name).replace("/", "\\")
        expected = nodataless_item["assets"]["B05"]["href"].replace("/", "\\")
        assert backing == expected, (
            f"the default read must back the timestep with the href, got {backing}"
        )

    def test_classmethod_default_path_still_works(self, three_local_items):
        """The public classmethod is unaffected by the new keywords.

        Test scenario:
            DatasetCollection.from_stac with no new arguments.
        """
        collection = DatasetCollection.from_stac(three_local_items, asset="data")
        assert collection.time_length == 3, (
            f"expected 3 timesteps, got {collection.time_length}"
        )
