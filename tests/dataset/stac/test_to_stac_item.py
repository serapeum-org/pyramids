"""Tests for Dataset.to_stac_item / pyramids.dataset._stac.to_stac_item (PB-6)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from osgeo import gdal
from shapely.geometry import shape

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset._stac import to_stac_item

pytestmark = pytest.mark.core


def _wgs84_from_array(array, nodata=-9999.0, top_left=(0.0, 4.0)):
    """Build a single-band EPSG:4326 dataset from `array` at cell size 1."""
    return Dataset.from_array(
        array,
        no_data_value=nodata,
        geo_ref=GeoReference(top_left_corner=top_left, cell_size=1.0, epsg=4326),
    )


@pytest.fixture
def wgs84_dataset():
    """A 4x4 single-band EPSG:4326 dataset (top-left (0, 4), cell 1, nodata -9999)."""
    return Dataset.from_array(
        np.ones((4, 4), dtype="float32"),
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    )


@pytest.fixture
def ramp_dataset():
    """A 4x4 EPSG:4326 dataset holding 0..15, so it has a real value range."""
    return _wgs84_from_array(np.arange(16, dtype="float32").reshape(4, 4))


@pytest.fixture
def corner_dataset():
    """A 4x4 EPSG:4326 dataset whose only valid pixels are the top-left 2x2."""
    array = np.full((4, 4), -9999.0, dtype="float32")
    array[:2, :2] = 1.0
    return _wgs84_from_array(array)


@pytest.fixture
def all_nodata_dataset():
    """A 4x4 EPSG:4326 dataset with no valid pixel at all."""
    return _wgs84_from_array(np.full((4, 4), -9999.0, dtype="float32"))


@pytest.fixture
def antimeridian_dataset():
    """A 4x8 EPSG:4326 dataset whose x extent runs 178 -> 186, past the seam."""
    return _wgs84_from_array(np.ones((4, 8), dtype="float32"), top_left=(178.0, 2.0))


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
        assert props["proj:transform"] == [1.0, 0.0, 0.0, 0.0, -1.0, 4.0], props[
            "proj:transform"
        ]

    def test_raster_bands_on_asset(self, wgs84_dataset):
        """raster:bands carries per-band data_type + nodata on the asset.

        Test scenario:
            One float32 band with nodata -9999.
        """
        asset = wgs84_dataset.to_stac_item("x", asset_href="s.tif")["assets"]["data"]
        bands = asset["raster:bands"]
        assert len(bands) == 1, f"expected 1 band, got {len(bands)}"
        assert bands[0]["nodata"] == -9999.0, f"nodata: {bands[0]}"
        assert "float" in bands[0]["data_type"].lower(), (
            f"data_type: {bands[0]['data_type']}"
        )

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
        ds = Dataset.from_array(
            np.ones((8, 8), dtype="float32"),
            geo_ref=GeoReference(
                top_left_corner=(500000.0, 5300000.0), cell_size=10.0, epsg=32633
            ),
        )
        item = ds.to_stac_item("x", asset_href="s.tif")
        w, s, e, n = item["bbox"]
        assert -180 <= w <= 180 and -180 <= e <= 180, (
            f"lon out of range: {item['bbox']}"
        )
        assert -90 <= s <= 90 and -90 <= n <= 90, f"lat out of range: {item['bbox']}"
        assert item["properties"]["proj:epsg"] == 32633, (
            "native proj:epsg should be UTM"
        )

    def test_media_type_and_roles(self, wgs84_dataset):
        """asset media type and roles are recorded when given.

        Test scenario:
            A COG media type and default roles land on the asset.
        """
        item = wgs84_dataset.to_stac_item(
            "x", asset_href="s.tif", asset_media_type="image/tiff; application=geotiff"
        )
        asset = item["assets"]["data"]
        assert asset["type"] == "image/tiff; application=geotiff", (
            f"type: {asset.get('type')}"
        )
        assert asset["roles"] == ["data"], f"roles: {asset['roles']}"

    def test_datetime_isoformat(self, wgs84_dataset):
        """A datetime object is serialised via isoformat().

        Test scenario:
            A datetime.datetime becomes its ISO string in properties.
        """
        import datetime as dt

        when = dt.datetime(2023, 6, 1, 12, 0, 0)
        item = wgs84_dataset.to_stac_item("x", asset_href="s.tif", datetime=when)
        assert item["properties"]["datetime"] == when.isoformat(), item["properties"][
            "datetime"
        ]

    def test_with_proj_false_omits_proj(self, wgs84_dataset):
        """with_proj=False omits the proj extension fields and schema.

        Test scenario:
            No proj:* keys and the projection schema is absent.
        """
        item = wgs84_dataset.to_stac_item("x", asset_href="s.tif", with_proj=False)
        assert not any(k.startswith("proj:") for k in item["properties"]), item[
            "properties"
        ]
        assert not any("projection" in e for e in item["stac_extensions"]), item[
            "stac_extensions"
        ]

    def test_crs_less_dataset_world_bbox(self, tmp_path):
        """A dataset without a CRS gets the world bbox + a warning.

        Test scenario:
            A raster with a geotransform but no projection -> bbox
            [-180,-90,180,90] and a no-CRS UserWarning. ``dataset.epsg`` softly
            defaults to 4326 even with no CRS, so the only way to build a genuinely
            CRS-less raster is raw GDAL with no ``SetProjection`` call.
        """
        path = str(tmp_path / "no_crs.tif")
        out = gdal.GetDriverByName("GTiff").Create(path, 3, 3, 1, gdal.GDT_Float32)
        out.SetGeoTransform((0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
        out.GetRasterBand(1).WriteArray(np.ones((3, 3), dtype="float32"))
        out.FlushCache()
        out = None

        ds = Dataset.read_file(path)
        assert not ds.crs, "test precondition: the raster must have no CRS"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            item = to_stac_item(ds, "x", asset_href="s.tif")
        assert item["bbox"] == [-180.0, -90.0, 180.0, 90.0], f"bbox: {item['bbox']}"
        assert any("no EPSG code" in str(w.message) for w in caught), (
            "expected a no-EPSG-code warning"
        )
        assert not any(k.startswith("proj:") for k in item["properties"]), (
            f"a CRS-less item must not advertise proj:* fields: {item['properties']}"
        )

    def test_round_trip_through_from_stac(self, wgs84_dataset, tmp_path):
        """to_stac_item -> from_stac rebuilds a collection over the asset.

        Test scenario:
            Write the dataset, emit an Item pointing at it, and feed [item] to
            from_stac; the collection reads back the same grid.
        """
        p = str(tmp_path / "scene.tif")
        wgs84_dataset.to_file(p)
        item = wgs84_dataset.to_stac_item(
            "scene-1", asset_href=p, datetime="2023-06-01T00:00:00Z"
        )
        coll = DatasetCollection.from_stac([item], asset="data")
        assert coll.time_length == 1, f"expected 1 timestep, got {coll.time_length}"
        assert coll.datasets[0].shape[-2:] == (
            4,
            4,
        ), f"grid not preserved: {coll.datasets[0].shape}"


class TestToStacItemDatetime:
    """L1: datetime handling produces only STAC-valid Items."""

    def test_default_datetime_is_now_not_null(self, wgs84_dataset):
        """Omitting datetime defaults to a non-null UTC timestamp.

        Test scenario:
            No datetime / range -> properties.datetime is a non-null ISO string
            (rio-stac behaviour), never a null that would be STAC-invalid.
        """
        item = wgs84_dataset.to_stac_item("x", asset_href="s.tif")
        when = item["properties"]["datetime"]
        assert when is not None, "datetime must not be null without a start/end range"
        assert when.startswith("20"), f"expected an ISO timestamp, got {when!r}"

    def test_null_datetime_with_range(self, wgs84_dataset):
        """datetime=None plus a start/end range writes a null datetime + range.

        Test scenario:
            The STAC-valid null-datetime form: datetime null, start/end present.
        """
        item = wgs84_dataset.to_stac_item(
            "x",
            asset_href="s.tif",
            datetime=None,
            start_datetime="2023-06-01T00:00:00Z",
            end_datetime="2023-06-30T00:00:00Z",
        )
        props = item["properties"]
        assert props["datetime"] is None, (
            f"datetime should be null with a range, got {props['datetime']}"
        )
        assert props["start_datetime"] == "2023-06-01T00:00:00Z", props
        assert props["end_datetime"] == "2023-06-30T00:00:00Z", props

    def test_range_datetimes_isoformat(self, wgs84_dataset):
        """datetime objects in the range are serialised via isoformat().

        Test scenario:
            A datetime.datetime start/end becomes its ISO string.
        """
        import datetime as dt

        item = wgs84_dataset.to_stac_item(
            "x",
            asset_href="s.tif",
            datetime=None,
            start_datetime=dt.datetime(2023, 1, 1),
            end_datetime=dt.datetime(2023, 12, 31),
        )
        assert item["properties"]["start_datetime"] == "2023-01-01T00:00:00", item[
            "properties"
        ]


class TestToStacItemBandMetadata:
    """STAC-01: the opt-in raster/eo band metadata on the emitted asset."""

    def test_defaults_emit_only_data_type_and_nodata(self, wgs84_dataset):
        """With no new kwargs a band carries exactly data_type + nodata.

        Test scenario:
            The default item's raster:bands entry has no statistics, histogram,
            scale or offset, and no eo extension is advertised.
        """
        item = wgs84_dataset.to_stac_item("x", asset_href="s.tif")
        band = item["assets"]["data"]["raster:bands"][0]
        assert set(band) == {"data_type", "nodata"}, f"extra band keys: {band}"
        assert band["nodata"] == -9999.0, f"nodata changed: {band}"
        assert "eo:bands" not in item["assets"]["data"], "eo:bands must be opt-in"
        assert not any("/eo/" in e for e in item["stac_extensions"]), item[
            "stac_extensions"
        ]

    def test_with_stats_emits_statistics(self, wgs84_dataset):
        """with_stats=True adds the raster-extension statistics object.

        Test scenario:
            An all-ones band yields minimum/maximum/mean 1.0 and stddev 0.0
            under the schema's key spellings.
        """
        band = wgs84_dataset.to_stac_item("x", asset_href="s.tif", with_stats=True)[
            "assets"
        ]["data"]["raster:bands"][0]
        stats = band["statistics"]
        assert set(stats) == {"minimum", "maximum", "mean", "stddev"}, (
            f"statistics keys: {sorted(stats)}"
        )
        assert stats["minimum"] == 1.0 and stats["maximum"] == 1.0, (
            f"an all-ones band should be 1..1, got {stats}"
        )
        assert stats["stddev"] == 0.0, f"stddev of a constant band: {stats}"

    def test_with_stats_exact_mode_matches_dataset_stats(self, ramp_dataset):
        """stats_approx_ok=False reports the exact figures Dataset.stats gives.

        Test scenario:
            A 0..15 ramp read exactly reports the same minimum/maximum as
            Dataset.stats(approx_ok=False).
        """
        band = ramp_dataset.to_stac_item(
            "x", asset_href="s.tif", with_stats=True, stats_approx_ok=False
        )["assets"]["data"]["raster:bands"][0]
        row = ramp_dataset.stats(band=0, approx_ok=False).iloc[0]
        assert band["statistics"]["minimum"] == float(row["min"]), band["statistics"]
        assert band["statistics"]["maximum"] == float(row["max"]), band["statistics"]

    def test_with_stats_all_nodata_band_warns_and_omits(self, all_nodata_dataset):
        """A band with no valid pixels is skipped with a warning, not an error.

        Test scenario:
            with_stats=True on an all-nodata band still emits the band, without
            a statistics key, and warns about the missing pixels.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            item = all_nodata_dataset.to_stac_item(
                "x", asset_href="s.tif", with_stats=True
            )
        band = item["assets"]["data"]["raster:bands"][0]
        assert "statistics" not in band, f"statistics must be omitted: {band}"
        assert band["data_type"].startswith("float"), f"band lost: {band}"
        assert any("no valid pixels" in str(w.message) for w in caught), (
            "expected a no-valid-pixels warning"
        )

    def test_with_histogram_emits_spec_shape(self, ramp_dataset):
        """with_histogram=True adds count/min/max/buckets with count == #buckets.

        Test scenario:
            A 0..15 ramp bucketed into 5 bins reports 5 buckets and count 5.
        """
        band = ramp_dataset.to_stac_item(
            "x", asset_href="s.tif", with_histogram=True, histogram_bins=5
        )["assets"]["data"]["raster:bands"][0]
        histogram = band["histogram"]
        assert set(histogram) == {"count", "min", "max", "buckets"}, (
            f"histogram keys: {sorted(histogram)}"
        )
        assert histogram["count"] == len(histogram["buckets"]) == 5, (
            f"count must be the bucket count: {histogram}"
        )
        assert histogram["min"] < histogram["max"], f"edges: {histogram}"
        assert all(isinstance(c, int) for c in histogram["buckets"]), histogram[
            "buckets"
        ]

    def test_with_histogram_constant_band_warns_and_omits(self, wgs84_dataset):
        """A band with no value range is skipped with a warning.

        Test scenario:
            An all-ones band has min == max, so GDAL cannot bucket it; the band
            is emitted without a histogram key.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            item = wgs84_dataset.to_stac_item(
                "x", asset_href="s.tif", with_histogram=True
            )
        band = item["assets"]["data"]["raster:bands"][0]
        assert "histogram" not in band, f"histogram must be omitted: {band}"
        assert any("no value range" in str(w.message) for w in caught), (
            "expected a no-value-range warning"
        )

    def test_with_eo_emits_bands_and_schema(self, wgs84_dataset):
        """with_eo=True adds eo:bands to the asset and the eo schema URI.

        Test scenario:
            A single-band dataset yields [{"name": "Band_1"}].
        """
        item = wgs84_dataset.to_stac_item("x", asset_href="s.tif", with_eo=True)
        assert item["assets"]["data"]["eo:bands"] == [{"name": "Band_1"}], item[
            "assets"
        ]["data"].get("eo:bands")
        assert any("/eo/" in e for e in item["stac_extensions"]), item[
            "stac_extensions"
        ]

    def test_scale_offset_emitted_only_when_non_identity(self, ramp_dataset):
        """scale/offset appear only for a band with real CF packing.

        Test scenario:
            An unpacked band emits neither; after setting scale 0.01 /
            offset 5.0 both are emitted as floats.
        """
        band = ramp_dataset.to_stac_item("x", asset_href="s.tif")["assets"]["data"][
            "raster:bands"
        ][0]
        assert "scale" not in band and "offset" not in band, (
            f"an unpacked band must emit no packing: {band}"
        )
        ramp_dataset.scale = [0.01]
        ramp_dataset.offset = [5.0]
        packed = ramp_dataset.to_stac_item("x", asset_href="s.tif")["assets"]["data"][
            "raster:bands"
        ][0]
        assert packed["scale"] == 0.01, f"scale: {packed}"
        assert packed["offset"] == 5.0, f"offset: {packed}"

    def test_non_finite_nodata_is_stringified(self):
        """A NaN nodata sentinel is emitted as the string "nan".

        Test scenario:
            The raster extension spells non-finite nodata as a string so the
            emitted JSON stays valid.
        """
        ds = _wgs84_from_array(np.ones((4, 4), dtype="float32"), nodata=np.nan)
        band = ds.to_stac_item("x", asset_href="s.tif")["assets"]["data"][
            "raster:bands"
        ][0]
        assert band["nodata"] == "nan", f"nodata: {band}"

    def test_with_raster_false_omits_bands_but_keeps_eo(self, wgs84_dataset):
        """with_raster=False drops raster:bands while with_eo still applies.

        Test scenario:
            The asset carries eo:bands and no raster:bands.
        """
        item = wgs84_dataset.to_stac_item(
            "x", asset_href="s.tif", with_raster=False, with_eo=True, with_stats=True
        )
        asset = item["assets"]["data"]
        assert "raster:bands" not in asset, f"raster:bands should be absent: {asset}"
        assert asset["eo:bands"] == [{"name": "Band_1"}], asset.get("eo:bands")


class TestToStacItemDataFootprint:
    """STAC-02: the nodata-aware footprint mode."""

    def test_default_footprint_is_bbox_unchanged(self, corner_dataset):
        """The default mode still emits the full bounding rectangle.

        Test scenario:
            A dataset with only a valid corner still reports the grid bbox and
            a single Polygon when footprint is left at its default.
        """
        item = corner_dataset.to_stac_item("x", asset_href="s.tif")
        assert item["bbox"] == [0.0, 0.0, 4.0, 4.0], f"bbox: {item['bbox']}"
        assert item["geometry"]["type"] == "Polygon", item["geometry"]["type"]

    def test_data_footprint_tighter_than_bbox(self, corner_dataset):
        """footprint="data" traces the valid pixels only.

        Test scenario:
            Only the top-left 2x2 of a 4x4 grid is valid, so the data geometry
            is smaller than the bbox geometry and the bbox shrinks to it.
        """
        data_item = corner_dataset.to_stac_item(
            "x", asset_href="s.tif", footprint="data"
        )
        bbox_item = corner_dataset.to_stac_item("x", asset_href="s.tif")
        data_area = shape(data_item["geometry"]).area
        bbox_area = shape(bbox_item["geometry"]).area
        assert data_area < bbox_area, (
            f"data footprint {data_area} should be tighter than {bbox_area}"
        )
        assert data_item["bbox"] == [0.0, 2.0, 2.0, 4.0], f"bbox: {data_item['bbox']}"

    def test_data_footprint_all_nodata_falls_back_to_bbox(self, all_nodata_dataset):
        """An all-nodata band falls back to the bbox rectangle with a warning.

        Test scenario:
            Dataset.footprint returns None, so the emitted bbox is the grid
            extent and a fallback warning is raised.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            item = all_nodata_dataset.to_stac_item(
                "x", asset_href="s.tif", footprint="data"
            )
        assert item["bbox"] == [0.0, 0.0, 4.0, 4.0], f"bbox: {item['bbox']}"
        assert item["geometry"]["type"] == "Polygon", item["geometry"]["type"]
        assert any("no valid pixels" in str(w.message) for w in caught), (
            "expected a fallback warning"
        )

    def test_data_footprint_keeps_multi_part_coverage(self):
        """Disjoint valid blocks stay a MultiPolygon (no convex hull).

        Test scenario:
            Two separated 2x2 valid blocks produce a two-part geometry.
        """
        array = np.full((6, 6), -9999.0, dtype="float32")
        array[:2, :2] = 1.0
        array[4:, 4:] = 1.0
        ds = _wgs84_from_array(array, top_left=(0.0, 6.0))
        item = ds.to_stac_item("x", asset_href="s.tif", footprint="data")
        geom = shape(item["geometry"])
        assert geom.geom_type == "MultiPolygon", f"geom type: {geom.geom_type}"
        assert len(geom.geoms) == 2, f"expected 2 parts, got {len(geom.geoms)}"

    def test_data_footprint_reprojects_from_utm(self):
        """A UTM data footprint lands in lon/lat, matching the bbox mode.

        Test scenario:
            A UTM zone-33N grid's data footprint sits inside the reprojected
            bbox footprint, proving both use the same transform path.
        """
        ds = Dataset.from_array(
            np.ones((8, 8), dtype="float32"),
            geo_ref=GeoReference(
                top_left_corner=(500000.0, 5300000.0), cell_size=10.0, epsg=32633
            ),
        )
        data_geom = shape(
            ds.to_stac_item("x", asset_href="s.tif", footprint="data")["geometry"]
        )
        bbox_geom = shape(ds.to_stac_item("x", asset_href="s.tif")["geometry"])
        assert data_geom.intersection(bbox_geom).area > 0, (
            "the data footprint must overlap the bbox footprint"
        )
        assert data_geom.difference(bbox_geom.buffer(1e-6)).area == pytest.approx(
            0.0, abs=1e-9
        ), "the data footprint must sit inside the bbox footprint"

    def test_densify_adds_vertices_before_reprojection(self):
        """densify bends long edges before they are reprojected.

        Test scenario:
            A UTM footprint densified at 10 m yields more vertices than the
            undensified one, which has only the grid corners.
        """
        ds = Dataset.from_array(
            np.ones((8, 8), dtype="float32"),
            geo_ref=GeoReference(
                top_left_corner=(500000.0, 5300000.0), cell_size=10.0, epsg=32633
            ),
        )
        plain = ds.to_stac_item("x", asset_href="s.tif", footprint="data")
        dense = ds.to_stac_item("x", asset_href="s.tif", footprint="data", densify=10.0)
        plain_ring = plain["geometry"]["coordinates"][0]
        dense_ring = dense["geometry"]["coordinates"][0]
        assert len(dense_ring) > len(plain_ring), (
            f"densify should add vertices: {len(dense_ring)} vs {len(plain_ring)}"
        )

    def test_simplify_tolerance_drops_vertices(self):
        """simplify_tolerance thins the reprojected footprint.

        Test scenario:
            A staircase-shaped coverage simplified at 1 degree keeps fewer
            vertices than the exact polygon.
        """
        array = np.full((6, 6), -9999.0, dtype="float32")
        for row in range(6):
            array[row, : row + 1] = 1.0
        ds = _wgs84_from_array(array, top_left=(0.0, 6.0))
        exact = ds.to_stac_item("x", asset_href="s.tif", footprint="data")
        simple = ds.to_stac_item(
            "x", asset_href="s.tif", footprint="data", simplify_tolerance=1.0
        )
        exact_ring = exact["geometry"]["coordinates"][0]
        simple_ring = simple["geometry"]["coordinates"][0]
        assert len(simple_ring) < len(exact_ring), (
            f"simplify should drop vertices: {len(simple_ring)} vs {len(exact_ring)}"
        )

    def test_footprint_max_samples_still_produces_a_geometry(self, corner_dataset):
        """footprint_max_samples trades accuracy for speed but still emits one.

        Test scenario:
            A decimated mask read yields a non-empty polygonal geometry.
        """
        item = corner_dataset.to_stac_item(
            "x", asset_href="s.tif", footprint="data", footprint_max_samples=4
        )
        geom = shape(item["geometry"])
        assert geom.geom_type in ("Polygon", "MultiPolygon"), geom.geom_type
        assert geom.area > 0, f"empty footprint: {item['geometry']}"

    def test_data_mode_on_crs_less_dataset_uses_world_bbox(self, tmp_path):
        """A CRS-less dataset keeps the world-extent branch in data mode.

        Test scenario:
            Dataset.footprint needs a CRS, so footprint="data" falls through to
            the bbox path and its no-EPSG warning.
        """
        path = str(tmp_path / "no_crs.tif")
        out = gdal.GetDriverByName("GTiff").Create(path, 3, 3, 1, gdal.GDT_Float32)
        out.SetGeoTransform((0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
        out.GetRasterBand(1).WriteArray(np.ones((3, 3), dtype="float32"))
        out.FlushCache()
        out = None

        ds = Dataset.read_file(path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            item = to_stac_item(ds, "x", asset_href="s.tif", footprint="data")
        assert item["bbox"] == [-180.0, -90.0, 180.0, 90.0], f"bbox: {item['bbox']}"
        assert any("no EPSG code" in str(w.message) for w in caught), (
            "expected a no-EPSG-code warning"
        )

    def test_unknown_footprint_mode_raises(self, wgs84_dataset):
        """An unsupported footprint mode is rejected up front.

        Test scenario:
            footprint="convex" raises ValueError naming the valid modes.
        """
        with pytest.raises(ValueError, match="footprint must be 'bbox' or 'data'"):
            wgs84_dataset.to_stac_item("x", asset_href="s.tif", footprint="convex")

    def test_data_footprint_round_trips_through_from_stac(
        self, corner_dataset, tmp_path
    ):
        """An item with a data footprint is still readable by from_stac.

        Test scenario:
            Emit a data-footprint item over a written raster and rebuild a
            collection from it.
        """
        path = str(tmp_path / "corner.tif")
        corner_dataset.to_file(path)
        item = corner_dataset.to_stac_item(
            "scene-1",
            asset_href=path,
            datetime="2023-06-01T00:00:00Z",
            footprint="data",
        )
        coll = DatasetCollection.from_stac([item], asset="data")
        assert coll.time_length == 1, f"expected 1 timestep, got {coll.time_length}"


class TestToStacItemAntimeridian:
    """STAC-03: emitted geometry split at the antimeridian."""

    def test_data_footprint_splits_at_the_seam(self, antimeridian_dataset):
        """A data footprint running past +180 becomes a two-part MultiPolygon.

        Test scenario:
            A grid spanning 178 -> 186 yields two polygons and a west > east
            bbox.
        """
        item = antimeridian_dataset.to_stac_item(
            "x", asset_href="s.tif", footprint="data"
        )
        geom = shape(item["geometry"])
        assert geom.geom_type == "MultiPolygon", f"geom type: {geom.geom_type}"
        assert len(geom.geoms) == 2, f"expected 2 parts, got {len(geom.geoms)}"
        west, south, east, north = item["bbox"]
        assert west > east, f"a crossing bbox must have west > east: {item['bbox']}"
        assert (west, east) == (178.0, -174.0), f"bbox lons: {item['bbox']}"
        assert (south, north) == (-2.0, 2.0), f"bbox lats: {item['bbox']}"

    def test_bbox_footprint_splits_at_the_seam(self, antimeridian_dataset):
        """The bbox mode is split at the seam too.

        Test scenario:
            The same crossing grid in the default footprint mode also yields a
            MultiPolygon and a west > east bbox.
        """
        item = antimeridian_dataset.to_stac_item("x", asset_href="s.tif")
        geom = shape(item["geometry"])
        assert geom.geom_type == "MultiPolygon", f"geom type: {geom.geom_type}"
        west, _, east, _ = item["bbox"]
        assert west > east, f"a crossing bbox must have west > east: {item['bbox']}"

    def test_split_pieces_stay_inside_the_lon_range(self, antimeridian_dataset):
        """Every emitted coordinate is a valid longitude.

        Test scenario:
            No vertex of the split geometry falls outside [-180, 180].
        """
        item = antimeridian_dataset.to_stac_item(
            "x", asset_href="s.tif", footprint="data"
        )
        lons = [
            point[0]
            for polygon in item["geometry"]["coordinates"]
            for ring in polygon
            for point in ring
        ]
        assert all(-180.0 <= lon <= 180.0 for lon in lons), f"longitudes: {lons}"

    def test_geometry_wholly_past_the_seam_is_wrapped_not_split(self):
        """A grid entirely beyond +180 is wrapped back, not split.

        Test scenario:
            x spanning 181 -> 185 is the same place as -179 -> -175, so one
            Polygon with a normal west < east bbox is emitted.
        """
        ds = _wgs84_from_array(np.ones((2, 4), dtype="float32"), top_left=(181.0, 1.0))
        item = ds.to_stac_item("x", asset_href="s.tif", footprint="data")
        geom = shape(item["geometry"])
        assert geom.geom_type == "Polygon", f"geom type: {geom.geom_type}"
        west, _, east, _ = item["bbox"]
        assert (west, east) == (-179.0, -175.0), f"bbox lons: {item['bbox']}"

    def test_non_crossing_geometry_untouched(self, wgs84_dataset):
        """A normal grid is never turned into a MultiPolygon.

        Test scenario:
            Both footprint modes keep a single Polygon and a west <= east bbox.
        """
        for mode in ("bbox", "data"):
            item = wgs84_dataset.to_stac_item("x", asset_href="s.tif", footprint=mode)
            geom = shape(item["geometry"])
            assert geom.geom_type == "Polygon", f"{mode}: {geom.geom_type}"
            west, _, east, _ = item["bbox"]
            assert west <= east, f"{mode} produced a false crossing: {item['bbox']}"

    def test_world_extent_is_not_split(self, tmp_path):
        """The CRS-less world extent is left alone, not read as a crossing.

        Test scenario:
            A -180..180 geometry spans 360 degrees but is global, so it stays a
            single Polygon with the world bbox.
        """
        path = str(tmp_path / "no_crs.tif")
        out = gdal.GetDriverByName("GTiff").Create(path, 3, 3, 1, gdal.GDT_Float32)
        out.SetGeoTransform((0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
        out.GetRasterBand(1).WriteArray(np.ones((3, 3), dtype="float32"))
        out.FlushCache()
        out = None

        ds = Dataset.read_file(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            item = to_stac_item(ds, "x", asset_href="s.tif")
        assert item["geometry"]["type"] == "Polygon", item["geometry"]["type"]
        assert item["bbox"] == [-180.0, -90.0, 180.0, 90.0], f"bbox: {item['bbox']}"
