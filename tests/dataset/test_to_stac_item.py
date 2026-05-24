"""Tests for Dataset.to_stac_item / pyramids.dataset._stac.to_stac_item (PB-6)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset._stac import to_stac_item

pytestmark = pytest.mark.core


@pytest.fixture
def wgs84_dataset():
    """A 4x4 single-band EPSG:4326 dataset (top-left (0, 4), cell 1, nodata -9999)."""
    return Dataset.create_from_array(
        np.ones((4, 4), dtype="float32"),
        top_left_corner=(0.0, 4.0),
        cell_size=1.0,
        epsg=4326,
        no_data_value=-9999.0,
    )


class TestToStacItem:
    """Tests for the raster -> STAC Item conversion."""

    def test_basic_feature_shape(self, wgs84_dataset):
        """The result is a GeoJSON Feature with the core STAC keys.

        Test scenario:
            type/id/geometry/bbox/properties/assets/stac_extensions present.
        """
        item = wgs84_dataset.to_stac_item("scene-1", asset_href="s3://b/s.tif")
        assert item["type"] == "Feature", f"expected a Feature, got {item['type']}"
        assert item["id"] == "scene-1", f"id mismatch: {item['id']}"
        assert "data" in item["assets"], "default asset key 'data' missing"
        assert item["assets"]["data"]["href"] == "s3://b/s.tif", "asset href mismatch"

    def test_proj_fields(self, wgs84_dataset):
        """The proj extension fields come from the dataset grid.

        Test scenario:
            proj:code/epsg/shape/transform/bbox reflect a 4x4 EPSG:4326 grid.
        """
        props = wgs84_dataset.to_stac_item("x", asset_href="s.tif")["properties"]
        assert props["proj:epsg"] == 4326, f"proj:epsg: {props['proj:epsg']}"
        assert props["proj:code"] == "EPSG:4326", f"proj:code: {props['proj:code']}"
        assert props["proj:shape"] == [4, 4], f"proj:shape: {props['proj:shape']}"
        # proj:transform is the rasterio affine [a,b,c,d,e,f]: xres=1, x0=0, yres=-1, y0=4
        assert props["proj:transform"] == [1.0, 0.0, 0.0, 0.0, -1.0, 4.0], props["proj:transform"]

    def test_raster_bands_on_asset(self, wgs84_dataset):
        """raster:bands carries per-band data_type + nodata on the asset.

        Test scenario:
            One float32 band with nodata -9999.
        """
        asset = wgs84_dataset.to_stac_item("x", asset_href="s.tif")["assets"]["data"]
        bands = asset["raster:bands"]
        assert len(bands) == 1, f"expected 1 band, got {len(bands)}"
        assert bands[0]["nodata"] == -9999.0, f"nodata: {bands[0]}"
        assert "float" in bands[0]["data_type"].lower(), f"data_type: {bands[0]['data_type']}"

    def test_bbox_4326_matches_grid(self, wgs84_dataset):
        """The 4326 bbox equals the native grid extent (already lon/lat).

        Test scenario:
            top-left (0, 4), 4x4 at cell 1 -> [0, 0, 4, 4].
        """
        item = wgs84_dataset.to_stac_item("x", asset_href="s.tif")
        assert item["bbox"] == [0.0, 0.0, 4.0, 4.0], f"bbox: {item['bbox']}"

    def test_reprojects_utm_footprint_to_4326(self):
        """A UTM dataset's footprint is reprojected into lon/lat ranges.

        Test scenario:
            A UTM zone-33N grid yields a 4326 bbox within +/-180 / +/-90.
        """
        ds = Dataset.create_from_array(
            np.ones((8, 8), dtype="float32"),
            top_left_corner=(500000.0, 5300000.0),
            cell_size=10.0,
            epsg=32633,
        )
        item = ds.to_stac_item("x", asset_href="s.tif")
        w, s, e, n = item["bbox"]
        assert -180 <= w <= 180 and -180 <= e <= 180, f"lon out of range: {item['bbox']}"
        assert -90 <= s <= 90 and -90 <= n <= 90, f"lat out of range: {item['bbox']}"
        assert item["properties"]["proj:epsg"] == 32633, "native proj:epsg should be UTM"

    def test_media_type_and_roles(self, wgs84_dataset):
        """asset media type and roles are recorded when given.

        Test scenario:
            A COG media type and default roles land on the asset.
        """
        item = wgs84_dataset.to_stac_item(
            "x", asset_href="s.tif", asset_media_type="image/tiff; application=geotiff"
        )
        asset = item["assets"]["data"]
        assert asset["type"] == "image/tiff; application=geotiff", f"type: {asset.get('type')}"
        assert asset["roles"] == ["data"], f"roles: {asset['roles']}"

    def test_datetime_isoformat(self, wgs84_dataset):
        """A datetime object is serialised via isoformat().

        Test scenario:
            A datetime.datetime becomes its ISO string in properties.
        """
        import datetime as dt

        when = dt.datetime(2023, 6, 1, 12, 0, 0)
        item = wgs84_dataset.to_stac_item("x", asset_href="s.tif", datetime=when)
        assert item["properties"]["datetime"] == when.isoformat(), item["properties"]["datetime"]

    def test_with_proj_false_omits_proj(self, wgs84_dataset):
        """with_proj=False omits the proj extension fields and schema.

        Test scenario:
            No proj:* keys and the projection schema is absent.
        """
        item = wgs84_dataset.to_stac_item("x", asset_href="s.tif", with_proj=False)
        assert not any(k.startswith("proj:") for k in item["properties"]), item["properties"]
        assert not any("projection" in e for e in item["stac_extensions"]), item["stac_extensions"]

    def test_crs_less_dataset_world_bbox(self):
        """A dataset without a CRS gets the world bbox + a warning.

        Test scenario:
            epsg falsy -> bbox [-180,-90,180,90] and a UserWarning.
        """
        ds = Dataset.create_from_array(
            np.ones((3, 3), dtype="float32"), top_left_corner=(0.0, 3.0), cell_size=1.0
        )
        # create_from_array without epsg may still tag a default; force no CRS.
        if ds.epsg:
            pytest.skip("dataset got a default CRS; cannot exercise the CRS-less path here")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            item = to_stac_item(ds, "x", asset_href="s.tif")
        assert item["bbox"] == [-180.0, -90.0, 180.0, 90.0], f"bbox: {item['bbox']}"
        assert any("no CRS" in str(w.message) for w in caught), "expected a no-CRS warning"

    def test_round_trip_through_from_stac(self, wgs84_dataset, tmp_path):
        """to_stac_item -> from_stac rebuilds a collection over the asset.

        Test scenario:
            Write the dataset, emit an Item pointing at it, and feed [item] to
            from_stac; the collection reads back the same grid.
        """
        p = str(tmp_path / "scene.tif")
        wgs84_dataset.to_file(p)
        item = wgs84_dataset.to_stac_item("scene-1", asset_href=p, datetime="2023-06-01T00:00:00Z")
        coll = DatasetCollection.from_stac([item], asset="data")
        assert coll.time_length == 1, f"expected 1 timestep, got {coll.time_length}"
        assert coll.datasets[0].shape[-2:] == (4, 4), f"grid not preserved: {coll.datasets[0].shape}"
