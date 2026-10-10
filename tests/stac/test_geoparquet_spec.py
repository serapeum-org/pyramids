"""Tests for the spec-compliant STAC-GeoParquet layout (STAC-05).

The JSON-blob layout is covered by `test_geoparquet.py`; everything here
exercises `to_geoparquet_spec` / `from_geoparquet_spec` (and the `spec=` flag on
the shared entry points), which produce the columnar STAC-GeoParquet 1.1 shape
with pyarrow alone.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from pyramids.stac._geoparquet import (
    from_geoparquet,
    from_geoparquet_spec,
    to_geoparquet,
    to_geoparquet_spec,
)

pytestmark = pytest.mark.core


def _item(item_id, lon, lat):
    """A minimal but complete STAC item dict with a Point geometry."""
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "bbox": [lon, lat, lon, lat],
        "collection": "sentinel-2-l2a",
        "properties": {
            "datetime": "2023-06-01T00:00:00Z",
            "eo:cloud_cover": 12,
            "platform": "sentinel-2a",
        },
        "assets": {"data": {"href": f"s3://b/{item_id}.tif", "type": "image/tiff"}},
        "links": [{"href": "https://api/collections/s2", "rel": "collection"}],
        "stac_extensions": ["https://stac-extensions.github.io/eo/v1.1.0/schema.json"],
    }


class TestSpecGuards:
    """Argument guards that fire before any Parquet write."""

    def test_empty_items_raises(self):
        """An empty item sequence raises the same ValueError as the blob path.

        Test scenario:
            Nothing to serialise, so the write is refused up front.
        """
        pytest.importorskip("pyarrow")
        with pytest.raises(ValueError, match="no items"):
            to_geoparquet_spec([], "x.parquet")

    def test_empty_items_raises_through_the_flag(self):
        """`to_geoparquet(..., spec=True)` keeps the empty-items guard.

        Test scenario:
            The flag must not bypass the guard the blob path applies.
        """
        pytest.importorskip("pyarrow")
        with pytest.raises(ValueError, match="no items"):
            to_geoparquet([], "x.parquet", spec=True)

    def test_bad_item_type_raises(self):
        """A non-dict, non-pystac item raises TypeError.

        Test scenario:
            An int is neither a dict nor exposes to_dict().
        """
        pytest.importorskip("pyarrow")
        with pytest.raises(TypeError, match="to_dict"):
            to_geoparquet_spec([123], "x.parquet")

    def test_reserved_property_name_raises(self):
        """A property that would collide with a spec column is refused.

        Test scenario:
            Flattening a property named `assets` onto the reserved `assets`
            column would make the item unreadable, so the write raises.
        """
        pytest.importorskip("pyarrow")
        item = _item("a", 1.0, 2.0)
        item["properties"]["assets"] = {"nope": 1}
        with pytest.raises(ValueError, match="reserves the column names"):
            to_geoparquet_spec([item], "x.parquet")


@pytest.mark.parquet
class TestSpecSchema:
    """The on-disk file is columnar and carries the spec's metadata."""

    def test_columns_are_flattened_not_a_json_blob(self, tmp_path):
        """Properties become top-level typed columns, with no `stac_item` blob.

        Test scenario:
            Two items are written; the Parquet schema is inspected directly.
        """
        pq = pytest.importorskip("pyarrow.parquet")
        path = str(tmp_path / "spec.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0), _item("b", 3.0, 4.0)], path)

        table = pq.read_table(path)
        columns = set(table.column_names)
        assert "stac_item" not in columns, f"blob column leaked: {columns}"
        for required in ("id", "geometry", "bbox", "datetime", "assets", "links"):
            assert required in columns, f"{required} missing from {columns}"
        assert "eo:cloud_cover" in columns, f"properties not flattened: {columns}"
        assert "platform" in columns, f"properties not flattened: {columns}"

    def test_column_types_follow_the_spec(self, tmp_path):
        """Geometry is WKB binary, bbox a struct, datetime a UTC timestamp.

        Test scenario:
            The Arrow schema of a written file is checked field by field.
        """
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        path = str(tmp_path / "types.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0)], path)

        schema = pq.read_table(path).schema
        assert schema.field("geometry").type == pa.binary(), schema.field("geometry")
        assert schema.field("id").type == pa.string(), schema.field("id")
        assert pa.types.is_struct(schema.field("bbox").type), schema.field("bbox")
        datetime_type = schema.field("datetime").type
        assert pa.types.is_timestamp(datetime_type), datetime_type
        assert datetime_type.tz == "UTC", f"datetime tz: {datetime_type.tz}"
        assert pa.types.is_struct(schema.field("assets").type), schema.field("assets")
        assert pa.types.is_list(schema.field("links").type), schema.field("links")
        extensions = schema.field("stac_extensions").type
        assert extensions.value_type == pa.string(), extensions

    def test_file_metadata_declares_geoparquet_and_stac(self, tmp_path):
        """Both the `geo` and `stac-geoparquet` metadata keys are written.

        Test scenario:
            The Parquet key-value metadata is read without pyramids' help.
        """
        pq = pytest.importorskip("pyarrow.parquet")
        path = str(tmp_path / "meta.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0)], path)

        metadata = pq.read_metadata(path).metadata or {}
        assert b"geo" in metadata, f"metadata keys: {sorted(metadata)}"
        assert b"stac-geoparquet" in metadata, f"metadata keys: {sorted(metadata)}"
        geo = json.loads(metadata[b"geo"])
        assert geo["primary_column"] == "geometry", geo
        geometry = geo["columns"]["geometry"]
        assert geometry["encoding"] == "WKB", geometry
        assert geometry["geometry_types"] == ["Point"], geometry
        assert geometry["covering"]["bbox"]["xmin"] == ["bbox", "xmin"], geometry
        stac = json.loads(metadata[b"stac-geoparquet"])
        assert stac["version"] == "1.1.0", stac

    def test_bbox_struct_is_queryable(self, tmp_path):
        """The bbox struct holds the item's own bbox under the spec field names.

        Test scenario:
            One item's bbox is read straight out of the struct column.
        """
        pq = pytest.importorskip("pyarrow.parquet")
        path = str(tmp_path / "bbox.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0)], path)

        bbox = pq.read_table(path).column("bbox").to_pylist()[0]
        assert bbox == {
            "xmin": 1.0,
            "ymin": 2.0,
            "xmax": 1.0,
            "ymax": 2.0,
        }, f"bbox struct: {bbox}"


@pytest.mark.parquet
class TestSpecRoundTrip:
    """Items survive the spec layout unchanged in substance."""

    def test_round_trip_is_lossless(self, tmp_path):
        """Items written and read back equal the originals exactly.

        Test scenario:
            Two fully populated items go through the spec writer and reader.
        """
        pytest.importorskip("pyarrow")
        items = [_item("a", 1.0, 2.0), _item("b", 3.0, 4.0)]
        path = str(tmp_path / "round.parquet")
        to_geoparquet_spec(items, path)

        restored = from_geoparquet_spec(path)
        assert restored == items, f"round trip changed the items: {restored}"

    def test_round_trip_through_the_flag(self, tmp_path):
        """`spec=True` on both entry points is the same round trip.

        Test scenario:
            The flag form writes and reads the same file as the direct form.
        """
        pytest.importorskip("pyarrow")
        items = [_item("a", 1.0, 2.0)]
        path = str(tmp_path / "flag.parquet")
        to_geoparquet(items, path, spec=True)

        restored = from_geoparquet(path, spec=True)
        assert restored == items, f"round trip changed the items: {restored}"

    def test_heterogeneous_items_keep_their_own_keys(self, tmp_path):
        """Arrow's rectangular structs do not leak nulls into sparse items.

        Test scenario:
            Two items with disjoint properties and disjoint asset keys; neither
            may come back carrying the other's keys set to None.
        """
        pytest.importorskip("pyarrow")
        items = [
            {
                "type": "Feature",
                "id": "left",
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                "properties": {"datetime": "2024-01-02T03:04:05.123456Z", "gsd": 10},
                "assets": {"data": {"href": "a.tif", "roles": ["data"]}},
            },
            {
                "type": "Feature",
                "id": "right",
                "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                "bbox": [1.0, 1.0, 1.0, 1.0],
                "properties": {"datetime": "2024-02-02T00:00:00Z", "platform": "s2a"},
                "assets": {"thumbnail": {"href": "t.png"}},
            },
        ]
        path = str(tmp_path / "sparse.parquet")
        to_geoparquet_spec(items, path)

        restored = from_geoparquet_spec(path)
        assert restored == items, f"sparse items did not survive: {restored}"

    def test_fractional_and_whole_second_datetimes_both_survive(self, tmp_path):
        """Microseconds are kept and whole seconds stay free of a `.000000`.

        Test scenario:
            One item with sub-second precision, one without.
        """
        pytest.importorskip("pyarrow")
        first = _item("a", 1.0, 2.0)
        first["properties"]["datetime"] = "2024-01-02T03:04:05.123456Z"
        second = _item("b", 3.0, 4.0)
        path = str(tmp_path / "times.parquet")
        to_geoparquet_spec([first, second], path)

        restored = from_geoparquet_spec(path)
        stamps = [item["properties"]["datetime"] for item in restored]
        assert stamps == [
            "2024-01-02T03:04:05.123456Z",
            "2023-06-01T00:00:00Z",
        ], f"datetimes: {stamps}"

    def test_datetime_range_properties_become_timestamps(self, tmp_path):
        """`start_datetime` / `end_datetime` are timestamp columns too.

        Test scenario:
            An item carrying a time range round-trips and the columns are typed.
        """
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        item = _item("a", 1.0, 2.0)
        item["properties"]["start_datetime"] = "2023-05-31T00:00:00Z"
        item["properties"]["end_datetime"] = "2023-06-02T00:00:00Z"
        path = str(tmp_path / "range.parquet")
        to_geoparquet_spec([item], path)

        schema = pq.read_table(path).schema
        for name in ("start_datetime", "end_datetime"):
            assert pa.types.is_timestamp(schema.field(name).type), schema.field(name)
        assert from_geoparquet_spec(path) == [item], "range item did not survive"

    def test_mixed_typed_property_falls_back_to_json(self, tmp_path):
        """A property Arrow cannot type is JSON-encoded, not dropped.

        Test scenario:
            One item types the property as an int, the other as a string; the
            column lands as a string and both values come back as themselves.
        """
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        first = _item("a", 1.0, 2.0)
        first["properties"]["odd"] = 1
        second = _item("b", 3.0, 4.0)
        second["properties"]["odd"] = "one"
        path = str(tmp_path / "mixed.parquet")
        to_geoparquet_spec([first, second], path)

        schema = pq.read_table(path).schema
        assert schema.field("odd").type == pa.string(), schema.field("odd")
        restored = from_geoparquet_spec(path)
        assert [item["properties"]["odd"] for item in restored] == [
            1,
            "one",
        ], f"json fallback lost values: {restored}"
        assert restored == [first, second], f"mixed items did not survive: {restored}"

    def test_extra_root_keys_stay_at_the_root(self, tmp_path):
        """A non-spec item-root key comes back at the root, not in properties.

        Test scenario:
            An item carries a top-level `pyramids:note` key.
        """
        pytest.importorskip("pyarrow")
        item = _item("a", 1.0, 2.0)
        item["pyramids:note"] = "kept at the root"
        path = str(tmp_path / "root.parquet")
        to_geoparquet_spec([item], path)

        restored = from_geoparquet_spec(path)[0]
        assert restored["pyramids:note"] == "kept at the root", restored
        assert "pyramids:note" not in restored["properties"], restored["properties"]

    def test_bbox_is_absent_when_the_item_had_none(self, tmp_path):
        """An item without a bbox does not gain one on the way back.

        Test scenario:
            The bbox column is null for that row, so the key stays absent.
        """
        pytest.importorskip("pyarrow")
        item = _item("a", 1.0, 2.0)
        del item["bbox"]
        path = str(tmp_path / "nobbox.parquet")
        to_geoparquet_spec([item], path)

        restored = from_geoparquet_spec(path)[0]
        assert "bbox" not in restored, f"bbox was invented: {restored}"
        assert restored == item, f"item changed: {restored}"

    def test_three_dimensional_bbox_keeps_its_z_bounds(self, tmp_path):
        """A 6-element bbox round-trips through the zmin/zmax struct fields.

        Test scenario:
            An item with a 3D bbox is written and read back.
        """
        pytest.importorskip("pyarrow")
        item = _item("a", 1.0, 2.0)
        item["bbox"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        path = str(tmp_path / "bbox3d.parquet")
        to_geoparquet_spec([item], path)

        restored = from_geoparquet_spec(path)[0]
        assert restored["bbox"] == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ], f"3D bbox: {restored['bbox']}"


@pytest.mark.parquet
class TestSpecInteropWithTheBlobLayout:
    """The two layouts coexist: the blob default is untouched and detectable."""

    def test_blob_layout_is_still_the_default(self, tmp_path):
        """Without `spec=`, the writer still emits the `stac_item` blob column.

        Test scenario:
            A default write is inspected with pyarrow.
        """
        pq = pytest.importorskip("pyarrow.parquet")
        path = str(tmp_path / "blob.parquet")
        items = [_item("a", 1.0, 2.0)]
        to_geoparquet(items, path)

        columns = set(pq.read_table(path).column_names)
        assert "stac_item" in columns, f"blob column missing: {columns}"
        assert from_geoparquet(path) == items, "blob round trip changed"

    def test_spec_file_is_auto_detected_without_the_flag(self, tmp_path):
        """`from_geoparquet` reads a spec file even when `spec` is not passed.

        Test scenario:
            The `stac-geoparquet` file metadata routes the read to the spec
            reader instead of failing on the missing blob column.
        """
        pytest.importorskip("pyarrow")
        items = [_item("a", 1.0, 2.0), _item("b", 3.0, 4.0)]
        path = str(tmp_path / "auto.parquet")
        to_geoparquet_spec(items, path)

        restored = from_geoparquet(path)
        assert restored == items, f"auto-detected read changed the items: {restored}"

    def test_spec_items_feed_from_stac(self, tmp_path):
        """Items restored from the spec layout can still drive from_stac.

        Test scenario:
            Two items pointing at real local rasters go through the spec
            round trip and build a DatasetCollection.
        """
        pytest.importorskip("pyarrow")
        import numpy as np

        from pyramids.base.georeference import GeoReference
        from pyramids.dataset import Dataset, DatasetCollection

        items = []
        for index in range(2):
            raster = str(tmp_path / f"r{index}.tif")
            Dataset.from_array(
                np.ones((3, 3), "float32"),
                geo_ref=GeoReference(
                    top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326
                ),
            ).to_file(raster)
            items.append(
                {
                    "type": "Feature",
                    "id": f"r{index}",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0], [0.0, 0.0]]
                        ],
                    },
                    "bbox": [0.0, 0.0, 3.0, 3.0],
                    "properties": {"datetime": f"2023-06-0{index + 1}T00:00:00Z"},
                    "assets": {"data": {"href": raster, "type": "image/tiff"}},
                }
            )
        path = str(tmp_path / "scenes.parquet")
        to_geoparquet_spec(items, path)

        collection = DatasetCollection.from_stac(
            from_geoparquet_spec(path), asset="data"
        )
        assert collection.time_length == 2, (
            f"expected 2 timesteps, got {collection.time_length}"
        )

    def test_dataset_stac_item_round_trips(self, tmp_path):
        """A `Dataset.to_stac_item` dict survives the spec layout.

        Test scenario:
            The proj-extension properties land in their own columns and come
            back unchanged. `to_stac_item` spells its datetime with a `+00:00`
            offset, which the timestamp column normalises to `Z`, so the
            datetime is compared as an instant and the rest verbatim.
        """
        pytest.importorskip("pyarrow")
        import numpy as np

        from pyramids.base.georeference import GeoReference
        from pyramids.dataset import Dataset

        dataset = Dataset.from_array(
            np.ones((4, 4), "float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        )
        item = dataset.to_stac_item("scene-1", asset_href="s3://b/s.tif")
        path = str(tmp_path / "one.parquet")
        to_geoparquet_spec([item], path)

        restored = from_geoparquet_spec(path)[0]
        assert restored["properties"]["proj:code"] == "EPSG:4326", restored[
            "properties"
        ]
        written = restored["properties"].pop("datetime")
        expected = item["properties"].pop("datetime")
        assert written.endswith("Z"), f"datetime not normalised to Z: {written}"
        assert datetime.fromisoformat(
            written.replace("Z", "+00:00")
        ) == datetime.fromisoformat(expected), f"{written} != {expected}"
        assert restored == item, f"to_stac_item dict changed: {restored}"
