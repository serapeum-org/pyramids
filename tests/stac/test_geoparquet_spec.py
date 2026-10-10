"""Tests for the spec-compliant STAC-GeoParquet layout (STAC-05).

The JSON-blob layout is covered by `test_geoparquet.py`; everything here
exercises `to_geoparquet_spec` / `from_geoparquet_spec` (and the `spec=` flag on
the shared entry points), which produce the columnar STAC-GeoParquet 1.1 shape
with pyarrow alone.
"""

from __future__ import annotations

import json
from datetime import datetime

import geopandas
import pyproj
import pytest

from pyramids.stac._geoparquet import (
    _is_spec_geoparquet,
    _spec_column,
    from_geoparquet,
    from_geoparquet_spec,
    to_geoparquet,
    to_geoparquet_spec,
)

pytestmark = pytest.mark.parquet

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


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
        with pytest.raises(ValueError, match="no items"):
            to_geoparquet_spec([], "x.parquet")

    def test_empty_items_raises_through_the_flag(self):
        """`to_geoparquet(..., spec=True)` keeps the empty-items guard.

        Test scenario:
            The flag must not bypass the guard the blob path applies.
        """
        with pytest.raises(ValueError, match="no items"):
            to_geoparquet([], "x.parquet", spec=True)

    def test_bad_item_type_raises(self):
        """A non-dict, non-pystac item raises TypeError.

        Test scenario:
            An int is neither a dict nor exposes to_dict().
        """
        with pytest.raises(TypeError, match="to_dict"):
            to_geoparquet_spec([123], "x.parquet")

    def test_reserved_property_name_raises(self):
        """A property that would collide with a spec column is refused.

        Test scenario:
            Flattening a property named `assets` onto the reserved `assets`
            column would make the item unreadable, so the write raises.
        """
        item = _item("a", 1.0, 2.0)
        item["properties"]["assets"] = {"nope": 1}
        with pytest.raises(ValueError, match="reserves the column names"):
            to_geoparquet_spec([item], "x.parquet")


class TestSpecSchema:
    """The on-disk file is columnar and carries the spec's metadata."""

    def test_columns_are_flattened_not_a_json_blob(self, tmp_path):
        """Properties become top-level typed columns, with no `stac_item` blob.

        Test scenario:
            Two items are written; the Parquet schema is inspected directly.
        """
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
        path = str(tmp_path / "bbox.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0)], path)

        bbox = pq.read_table(path).column("bbox").to_pylist()[0]
        assert bbox == {
            "xmin": 1.0,
            "ymin": 2.0,
            "xmax": 1.0,
            "ymax": 2.0,
        }, f"bbox struct: {bbox}"


class TestSpecRoundTrip:
    """Items survive the spec layout unchanged in substance."""

    def test_round_trip_is_lossless(self, tmp_path):
        """Items written and read back equal the originals exactly.

        Test scenario:
            Two fully populated items go through the spec writer and reader.
        """
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


class TestSpecTimestampCoercion:
    """The timestamp columns accept every spelling, and refuse the rest cleanly."""

    def test_datetime_object_is_normalised_to_utc(self, tmp_path):
        """A naive `datetime` property is read as UTC and emitted with a `Z`.

        Test scenario:
            The item carries a `datetime` object rather than an RFC 3339 string,
            with no tzinfo, so the writer must stamp UTC on it itself.
        """
        item = _item("a", 1.0, 2.0)
        item["properties"]["datetime"] = datetime(2024, 3, 4, 5, 6, 7)
        path = str(tmp_path / "object.parquet")
        to_geoparquet_spec([item], path)

        stamp = from_geoparquet_spec(path)[0]["properties"]["datetime"]
        assert stamp == "2024-03-04T05:06:07Z", f"datetime not normalised: {stamp}"

    def test_aware_datetime_object_keeps_its_instant(self, tmp_path):
        """A `datetime` that already carries a tzinfo is converted, not restamped.

        Test scenario:
            A `+02:00` datetime must come back as the same instant in UTC.
        """
        item = _item("a", 1.0, 2.0)
        item["properties"]["datetime"] = datetime.fromisoformat(
            "2024-03-04T05:06:07+02:00"
        )
        path = str(tmp_path / "aware.parquet")
        to_geoparquet_spec([item], path)

        stamp = from_geoparquet_spec(path)[0]["properties"]["datetime"]
        assert stamp == "2024-03-04T03:06:07Z", f"instant not preserved: {stamp}"

    def test_item_without_a_datetime_still_gets_the_column(self, tmp_path):
        """`datetime` is always a column, even when no item carries the property.

        Test scenario:
            The only item has no `datetime` at all: the schema still holds the
            column (all null) and the key does not reappear on the way back.
        """
        item = _item("a", 1.0, 2.0)
        del item["properties"]["datetime"]
        path = str(tmp_path / "nodatetime.parquet")
        to_geoparquet_spec([item], path)

        columns = pq.read_table(path).column_names
        assert "datetime" in columns, f"datetime column missing: {columns}"
        restored = from_geoparquet_spec(path)[0]
        assert "datetime" not in restored["properties"], (
            f"datetime was invented: {restored['properties']}"
        )
        assert restored == item, f"item changed: {restored}"

    def test_unparseable_datetime_string_falls_back_to_a_string_column(self, tmp_path):
        """A `datetime` that is not RFC 3339 is kept verbatim as a string.

        Test scenario:
            `"last tuesday"` cannot be parsed, so the timestamp column is
            abandoned and the text survives the round trip unchanged.
        """
        item = _item("a", 1.0, 2.0)
        item["properties"]["datetime"] = "last tuesday"
        path = str(tmp_path / "badtext.parquet")
        to_geoparquet_spec([item], path)

        field = pq.read_table(path).schema.field("datetime")
        assert field.type == pa.string(), f"expected a string fallback, got {field}"
        restored = from_geoparquet_spec(path)[0]
        assert restored["properties"]["datetime"] == "last tuesday", (
            f"unparseable datetime was altered: {restored['properties']}"
        )

    def test_non_timestamp_datetime_type_falls_back_to_its_own_type(self, tmp_path):
        """A `datetime` that is neither a string nor a datetime is kept as-is.

        Test scenario:
            An integer epoch is not a timestamp spelling the writer accepts, so
            the column is built from the value's own Arrow type instead.
        """
        item = _item("a", 1.0, 2.0)
        item["properties"]["datetime"] = 1700000000
        path = str(tmp_path / "epoch.parquet")
        to_geoparquet_spec([item], path)

        field = pq.read_table(path).schema.field("datetime")
        assert not pa.types.is_timestamp(field.type), (
            f"an int epoch must not be typed as a timestamp, got {field}"
        )
        restored = from_geoparquet_spec(path)[0]
        assert restored["properties"]["datetime"] == 1700000000, (
            f"int datetime was altered: {restored['properties']}"
        )


class TestSpecTypedColumnFallbacks:
    """A spec column whose values reject its declared type still round-trips."""

    def test_mixed_typed_id_falls_back_to_json(self, tmp_path):
        """An `id` that is an int in one item and a string in another survives.

        Test scenario:
            `id` is declared as a string column, which an int rejects, and the
            inferred type cannot unify the two either — so the column is
            JSON-encoded and both values come back as themselves.
        """
        first = _item(7, 1.0, 2.0)
        second = _item("b", 3.0, 4.0)
        path = str(tmp_path / "mixedid.parquet")
        to_geoparquet_spec([first, second], path)

        field = pq.read_table(path).schema.field("id")
        assert field.type == pa.string(), f"expected a string column, got {field}"
        restored = from_geoparquet_spec(path)
        assert [item["id"] for item in restored] == [7, "b"], (
            f"the json fallback lost an id: {restored}"
        )
        assert restored == [first, second], f"mixed ids did not survive: {restored}"

    def test_out_of_range_int_property_falls_back_to_json(self, tmp_path):
        """An integer too large for Arrow becomes a JSON column, not an error.

        Test scenario:
            `2 ** 70` overflows every Arrow integer type, so the property is
            JSON-encoded and decoded back to the same Python int.
        """
        first = _item("a", 1.0, 2.0)
        first["properties"]["huge"] = 2**70
        second = _item("b", 3.0, 4.0)
        second["properties"]["huge"] = 1
        path = str(tmp_path / "huge.parquet")
        to_geoparquet_spec([first, second], path)

        restored = from_geoparquet_spec(path)
        assert [item["properties"]["huge"] for item in restored] == [2**70, 1], (
            f"the overflowing int did not survive: {restored}"
        )


class TestSpecInteropWithTheBlobLayout:
    """The two layouts coexist: the blob default is untouched and detectable."""

    def test_blob_layout_is_still_the_default(self, tmp_path):
        """Without `spec=`, the writer still emits the `stac_item` blob column.

        Test scenario:
            A default write is inspected with pyarrow.
        """
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
        items = [_item("a", 1.0, 2.0), _item("b", 3.0, 4.0)]
        path = str(tmp_path / "auto.parquet")
        to_geoparquet_spec(items, path)

        restored = from_geoparquet(path)
        assert restored == items, f"auto-detected read changed the items: {restored}"

    def test_non_parquet_file_reports_the_blob_readers_error(self, tmp_path):
        """The auto-detect probe stays silent and lets the blob reader complain.

        Test scenario:
            A text file named `.parquet` makes the metadata probe fail; the read
            must still surface the "not a parquet file" error rather than a
            crash inside the probe.
        """
        path = tmp_path / "text.parquet"
        path.write_text("this is not a parquet file\n", encoding="utf-8")

        with pytest.raises(ValueError, match="parquet"):
            from_geoparquet(str(path))

    def test_missing_path_reports_the_blob_readers_error(self, tmp_path):
        """A path that does not exist fails as a missing file, not in the probe.

        Test scenario:
            The probe's `read_metadata` raises an OSError, which is swallowed so
            the blob reader reports the missing file itself.
        """

        with pytest.raises(FileNotFoundError):
            from_geoparquet(str(tmp_path / "absent.parquet"))

    def test_spec_items_feed_from_stac(self, tmp_path):
        """Items restored from the spec layout can still drive from_stac.

        Test scenario:
            Two items pointing at real local rasters go through the spec
            round trip and build a DatasetCollection.
        """
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


class TestSpecEmptyStructs:
    """An empty dict is a legal STAC value, and Parquet has no empty struct."""

    def test_item_without_assets_round_trips(self, tmp_path):
        """An item whose `assets` is an empty dict writes and reads back.

        Test scenario:
            `pystac.Item.to_dict()` always emits `"assets": {}` for an
            asset-less item, and `pyarrow` infers a zero-field struct for it,
            which the Parquet writer cannot encode.
        """
        item = _item("a", 1.0, 2.0)
        item["assets"] = {}
        path = str(tmp_path / "noassets.parquet")
        to_geoparquet_spec([item], path)

        restored = from_geoparquet_spec(path)
        assert restored == [item], f"asset-less item did not survive: {restored}"

    def test_asset_without_fields_round_trips(self, tmp_path):
        """An asset whose own dict is empty survives too.

        Test scenario:
            `assets={"red": {}}` infers `struct<red: struct<>>`, so the empty
            struct is nested rather than top level.
        """
        item = _item("a", 1.0, 2.0)
        item["assets"] = {"red": {}}
        path = str(tmp_path / "emptyasset.parquet")
        to_geoparquet_spec([item], path)

        restored = from_geoparquet_spec(path)
        assert restored == [item], f"empty asset did not survive: {restored}"

    def test_empty_struct_inside_a_list_round_trips(self, tmp_path):
        """A list of empty dicts (`list<struct<>>`) survives as well.

        Test scenario:
            `links=[{}]` is the list-valued spelling of the same inferred
            zero-field struct.
        """
        item = _item("a", 1.0, 2.0)
        item["links"] = [{}]
        path = str(tmp_path / "emptylink.parquet")
        to_geoparquet_spec([item], path)

        restored = from_geoparquet_spec(path)
        assert restored == [item], f"empty link did not survive: {restored}"

    def test_the_docstring_example_runs(self, tmp_path):
        """The items in `to_geoparquet_spec`'s own example can be written.

        Test scenario:
            The documented example carries `"assets": {}`; it is `+SKIP`-ed in
            the doctest run (the module needs the optional `[parquet]` extra),
            so it is executed here instead.
        """
        items = [
            {
                "id": "a",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "bbox": [1.0, 2.0, 1.0, 2.0],
                "assets": {},
                "properties": {"datetime": "2023-01-01T00:00:00Z"},
            }
        ]
        path = str(tmp_path / "spec.parquet")
        to_geoparquet_spec(items, path)

        columns = pq.read_table(path).column_names
        assert "datetime" in columns, f"the documented column is missing: {columns}"


class TestSpecCrs:
    """The geometry column is advertised as OGC:CRS84, not as CRS-less."""

    def test_geometry_column_declares_crs84(self, tmp_path):
        """The `geo` metadata carries the OGC:CRS84 PROJJSON, not a null crs.

        Test scenario:
            GeoParquet 1.1 reads an explicit `"crs": null` as "no CRS assigned",
            which is not the same as the CRS84 default the docstring promises.
        """
        path = str(tmp_path / "crs.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0)], path)

        geo = json.loads(pq.read_metadata(path).metadata[b"geo"])
        crs = geo["columns"]["geometry"]["crs"]
        assert crs is not None, f"the crs is still an explicit null: {geo}"
        assert crs["id"] == {"authority": "OGC", "code": "CRS84"}, f"crs id: {crs}"

    def test_geopandas_reads_the_crs_back(self, tmp_path):
        """A third-party reader recovers a CRS from the spec file.

        Test scenario:
            geopandas is the reference consumer; a CRS-less GeoDataFrame cannot
            be reprojected, which is the user-visible cost of the null crs.
        """
        path = str(tmp_path / "gpd.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0)], path)

        frame = geopandas.read_parquet(path)
        assert frame.crs is not None, "geopandas read the spec file with no CRS"
        assert frame.crs.equals(pyproj.CRS.from_user_input("OGC:CRS84")), (
            f"expected OGC:CRS84, got {frame.crs}"
        )


class TestSpecCovering:
    """The `covering` is only advertised when it covers every row."""

    def test_covering_is_advertised_when_every_row_has_a_bbox(self, tmp_path):
        """A file whose rows all carry a bbox keeps the covering.

        Test scenario:
            The covering is what lets DuckDB and geopandas push a spatial
            predicate down, so it must survive the stricter rule.
        """
        path = str(tmp_path / "covered.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0), _item("b", 3.0, 4.0)], path)

        geo = json.loads(pq.read_metadata(path).metadata[b"geo"])
        covering = geo["columns"]["geometry"].get("covering")
        assert covering is not None, f"covering dropped for a full column: {geo}"
        assert covering["bbox"]["xmin"] == ["bbox", "xmin"], f"covering: {covering}"

    def test_covering_is_omitted_when_a_row_has_no_bbox(self, tmp_path):
        """A null bbox row means the covering would hide a real geometry.

        Test scenario:
            The second item has a geometry but no bbox, so a covering-based
            spatial filter would skip it; the covering must not be advertised.
        """
        second = _item("b", 3.0, 4.0)
        del second["bbox"]
        path = str(tmp_path / "partial.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0), second], path)

        geo = json.loads(pq.read_metadata(path).metadata[b"geo"])
        column = geo["columns"]["geometry"]
        assert "covering" not in column, f"covering advertised over a null bbox: {geo}"
        boxes = pq.read_table(path).column("bbox").to_pylist()
        assert boxes[1] is None, f"expected a null bbox for the second row: {boxes}"


class TestSpecNumericWidening:
    """A column that mixes integers and floats must not silently widen."""

    def test_mixed_int_and_float_property_keeps_both_types(self, tmp_path):
        """An `int` in one item and a `float` in another both come back as-is.

        Test scenario:
            pyarrow unifies int and float into `double` instead of raising, so
            `proj:epsg`-style integer properties would come back as floats.
        """
        first = _item("a", 1.0, 2.0)
        first["properties"]["n"] = 1
        second = _item("b", 3.0, 4.0)
        second["properties"]["n"] = 2.5
        path = str(tmp_path / "widen.parquet")
        to_geoparquet_spec([first, second], path)

        field = pq.read_table(path).schema.field("n")
        assert field.type == pa.string(), f"expected the json fallback, got {field}"
        restored = from_geoparquet_spec(path)
        values = [item["properties"]["n"] for item in restored]
        assert values == [1, 2.5], f"values widened: {values}"
        assert [type(value) for value in values] == [int, float], (
            f"types widened: {[type(value).__name__ for value in values]}"
        )

    def test_mixed_numerics_nested_in_assets_keep_their_types(self, tmp_path):
        """The same widening inside an `assets` struct is also avoided.

        Test scenario:
            Two items share an asset key whose `gsd` is an int in one and a
            float in the other, one nesting level below the column.
        """
        first = _item("a", 1.0, 2.0)
        first["assets"] = {"data": {"href": "a.tif", "gsd": 10}}
        second = _item("b", 3.0, 4.0)
        second["assets"] = {"data": {"href": "b.tif", "gsd": 10.5}}
        path = str(tmp_path / "nested.parquet")
        to_geoparquet_spec([first, second], path)

        restored = from_geoparquet_spec(path)
        gsds = [item["assets"]["data"]["gsd"] for item in restored]
        assert gsds == [10, 10.5], f"nested values widened: {gsds}"
        assert [type(value) for value in gsds] == [int, float], (
            f"nested types widened: {[type(value).__name__ for value in gsds]}"
        )

    def test_a_uniform_integer_property_stays_an_integer_column(self, tmp_path):
        """The fallback does not fire for a column that needs no widening.

        Test scenario:
            Both items type the property as an int, so the column must stay a
            queryable Arrow integer rather than becoming JSON text.
        """
        path = str(tmp_path / "ints.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0), _item("b", 3.0, 4.0)], path)

        field = pq.read_table(path).schema.field("eo:cloud_cover")
        assert pa.types.is_integer(field.type), f"expected an int column, got {field}"


class TestSpecDatetimeColumnType:
    """`datetime` is a timestamp column in every case, as documented."""

    def test_datetime_column_is_a_timestamp_with_no_datetimes(self, tmp_path):
        """An all-null `datetime` column is still `timestamp[us, UTC]`.

        Test scenario:
            The only item carries no datetime at all, which is the case that
            used to land a string column and defeat the stable schema.
        """
        item = _item("a", 1.0, 2.0)
        del item["properties"]["datetime"]
        path = str(tmp_path / "nulldatetime.parquet")
        to_geoparquet_spec([item], path)

        field = pq.read_table(path).schema.field("datetime")
        assert pa.types.is_timestamp(field.type), f"datetime column type: {field}"
        assert field.type.tz == "UTC", f"datetime tz: {field.type.tz}"
        restored = from_geoparquet_spec(path)[0]
        assert restored == item, f"item changed: {restored}"


class TestSpecProbeRobustness:
    """The auto-detect probe answers False for anything it cannot read."""

    def test_probe_swallows_every_arrow_exception(self, tmp_path, monkeypatch):
        """An `ArrowException` out of the probe is reported as "not a spec file".

        Test scenario:
            `ArrowNotImplementedError` and `ArrowCapacityError` derive from
            `ArrowException` but from neither `OSError` nor `ValueError`, so they
            used to escape a default `from_geoparquet(path)` call.
        """
        path = tmp_path / "probe.parquet"
        path.write_text("not parquet\n", encoding="utf-8")
        for failure in (pa.ArrowNotImplementedError, pa.ArrowCapacityError):

            def _raise(_path, _failure=failure):
                raise _failure("probe exploded")

            monkeypatch.setattr(pq, "read_metadata", _raise)
            assert _is_spec_geoparquet(str(path)) is False, (
                f"{failure.__name__} escaped the probe"
            )

    def test_probe_still_detects_a_real_spec_file(self, tmp_path):
        """The narrowed handler does not break the detection it exists for.

        Test scenario:
            A genuine spec file must still answer True.
        """
        path = str(tmp_path / "real.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0)], path)

        assert _is_spec_geoparquet(path) is True, "a real spec file went undetected"


class TestSpecMixedDimensionBboxes:
    """A 3D bbox keeps its Z extent even next to a 2D one."""

    def test_mixed_2d_and_3d_bboxes_keep_their_z_bounds(self, tmp_path):
        """One 2D bbox must not demote every 3D bbox in the file.

        Test scenario:
            The first item is 2D and the second 3D; the Z extent of the second
            used to be dropped with no warning.
        """
        first = _item("a", 1.0, 2.0)
        first["bbox"] = [1.0, 2.0, 1.0, 2.0]
        second = _item("b", 3.0, 4.0)
        second["bbox"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        path = str(tmp_path / "mixeddim.parquet")
        to_geoparquet_spec([first, second], path)

        boxes = [item["bbox"] for item in from_geoparquet_spec(path)]
        assert boxes == [
            [1.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ], f"mixed-dimension bboxes: {boxes}"

    def test_short_bbox_names_the_item(self, tmp_path):
        """A malformed bbox is refused with the item's id in the message.

        Test scenario:
            A 2-element bbox is neither 2D nor 3D; it used to surface as a bare
            `IndexError` from the struct builder.
        """
        item = _item("broken", 1.0, 2.0)
        item["bbox"] = [1.0, 2.0]
        with pytest.raises(ValueError, match="broken"):
            to_geoparquet_spec([item], str(tmp_path / "shortbbox.parquet"))


def _truncate_then_explode(_table, where, *_args, **_kwargs):
    """Stand in for a `write_table` that creates its target and then fails.

    That is what the Parquet writer does when the table holds a type it cannot
    encode: the file exists, with zero usable bytes, by the time it raises.
    """
    with open(where, "wb"):
        pass
    raise pa.ArrowNotImplementedError("write exploded")


class TestSpecWriteIsAtomic:
    """A failed write leaves no half-made file behind."""

    def test_a_failed_write_leaves_no_target_file(self, tmp_path, monkeypatch):
        """A write that raises must not leave a 0-byte target.

        Test scenario:
            The leftover used to defeat the auto-detect on the next read, which
            then reported `Parquet file size is 0 bytes` from the blob reader.
        """
        target = tmp_path / "failed.parquet"
        monkeypatch.setattr(pq, "write_table", _truncate_then_explode)
        with pytest.raises(pa.ArrowNotImplementedError, match="write exploded"):
            to_geoparquet_spec([_item("a", 1.0, 2.0)], str(target))

        assert not target.exists(), f"a failed write left {target} behind"
        leftovers = sorted(child.name for child in tmp_path.iterdir())
        assert leftovers == [], f"a failed write left temporary files: {leftovers}"

    def test_a_failed_rewrite_keeps_the_previous_file(self, tmp_path, monkeypatch):
        """A failed overwrite leaves the existing file readable.

        Test scenario:
            The target already holds a good spec file; the replacement write
            fails and the old contents must still be there.
        """
        target = tmp_path / "kept.parquet"
        items = [_item("a", 1.0, 2.0)]
        to_geoparquet_spec(items, str(target))

        monkeypatch.setattr(pq, "write_table", _truncate_then_explode)
        with pytest.raises(pa.ArrowNotImplementedError, match="write exploded"):
            to_geoparquet_spec([_item("b", 3.0, 4.0)], str(target))

        assert from_geoparquet_spec(str(target)) == items, (
            "a failed rewrite destroyed the previous file"
        )


class TestSpecColumnErrorHandling:
    """The type probes catch Arrow's own errors and nothing else."""

    def test_a_plain_type_error_is_not_swallowed(self):
        """A `TypeError` that is not an `ArrowTypeError` propagates.

        Test scenario:
            A bug inside the probe (a wrong-arity call, a `None` where the
            module is expected) raises a plain `TypeError`; turning that into a
            silent JSON column would hide it.
        """

        class _BrokenArrow:
            """A pyarrow stand-in whose `array()` fails with a plain TypeError."""

            ArrowInvalid = pa.ArrowInvalid
            ArrowTypeError = pa.ArrowTypeError
            ArrowNotImplementedError = pa.ArrowNotImplementedError

            @staticmethod
            def string():
                return pa.string()

            @staticmethod
            def array(*_args, **_kwargs):
                raise TypeError("array() got an unexpected keyword argument")

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _spec_column(_BrokenArrow(), "odd", [1, "one"])

    def test_an_unserialisable_value_raises_instead_of_stringifying(self, tmp_path):
        """A value Arrow rejects and JSON cannot encode raises a clear error.

        Test scenario:
            `json.dumps(..., default=str)` used to turn such a value into its
            `repr`, silently contradicting the "no data is lost" promise.
        """
        first = _item("a", 1.0, 2.0)
        first["properties"]["odd"] = object()
        second = _item("b", 3.0, 4.0)
        second["properties"]["odd"] = "text"
        with pytest.raises(TypeError, match="odd"):
            to_geoparquet_spec([first, second], str(tmp_path / "odd.parquet"))


class TestSpecReaderRejectsForeignLayouts:
    """`spec=True` on a file that is not a spec file errors instead of guessing."""

    def test_spec_reader_refuses_the_blob_layout(self, tmp_path):
        """A JSON-blob file read with `spec=True` raises, not returns garbage.

        Test scenario:
            The blob layout used to come back as
            `{"properties": {"stac_item": "<raw json>"}}` with no exception.
        """
        path = str(tmp_path / "blob.parquet")
        to_geoparquet([_item("a", 1.0, 2.0)], path)

        with pytest.raises(ValueError, match="stac_item"):
            from_geoparquet(path, spec=True)

    def test_spec_reader_refuses_a_plain_parquet_file(self, tmp_path):
        """A Parquet file with none of the spec columns is refused too.

        Test scenario:
            A two-column table written by plain pyarrow carries neither the
            STAC file metadata nor any reserved spec column.
        """
        path = str(tmp_path / "plain.parquet")
        pq.write_table(pa.table({"a": [1], "b": ["x"]}), path)

        with pytest.raises(ValueError, match="STAC-GeoParquet"):
            from_geoparquet_spec(path)

    def test_a_foreign_spec_writer_is_still_accepted(self, tmp_path):
        """A spec file without pyramids' own hint block still reads.

        Test scenario:
            Only the reserved spec columns are present — no
            `pyramids:stac-geoparquet` metadata — which is what the upstream
            `stac-geoparquet` package produces.
        """
        path = str(tmp_path / "foreign.parquet")
        to_geoparquet_spec([_item("a", 1.0, 2.0)], path)
        stripped = pq.read_table(path).replace_schema_metadata({})
        pq.write_table(stripped, path)

        restored = from_geoparquet_spec(path)
        assert restored[0]["id"] == "a", f"a foreign spec file was refused: {restored}"


class TestSpecDocumentedLosses:
    """The normalisations `from_geoparquet_spec` documents, each measured."""

    def test_an_explicit_null_property_comes_back_absent(self, tmp_path):
        """A property written as `None` is not recoverable.

        Test scenario:
            The column cannot tell "null" from "not set", so the key is dropped
            — the first documented normalisation.
        """
        item = _item("a", 1.0, 2.0)
        item["properties"]["cloud"] = None
        path = str(tmp_path / "null.parquet")
        to_geoparquet_spec([item], path)

        properties = from_geoparquet_spec(path)[0]["properties"]
        assert "cloud" not in properties, f"an explicit null survived: {properties}"

    def test_an_item_without_properties_gains_an_empty_dict(self, tmp_path):
        """`properties` is injected, so the round trip is not an identity.

        Test scenario:
            The item carries no `properties` key; it comes back with an empty
            one — the second documented normalisation.
        """
        item = {
            "type": "Feature",
            "id": "a",
            "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
        }
        path = str(tmp_path / "noprops.parquet")
        to_geoparquet_spec([item], path)

        restored = from_geoparquet_spec(path)[0]
        assert restored["properties"] == {}, f"properties: {restored['properties']}"
        assert restored != item, "the round trip is documented as non-identity here"

    def test_a_null_geometry_comes_back_absent(self, tmp_path):
        """An explicit `"geometry": None` is dropped, like any other null cell.

        Test scenario:
            A STAC Item may legally carry a null geometry; the column stores it
            as a null cell, which the reader turns into an absent key.
        """
        item = {
            "type": "Feature",
            "id": "a",
            "geometry": None,
            "properties": {"datetime": "2023-06-01T00:00:00Z"},
        }
        path = str(tmp_path / "nogeom.parquet")
        to_geoparquet_spec([item], path)

        restored = from_geoparquet_spec(path)[0]
        assert "geometry" not in restored, f"a null geometry survived: {restored}"

    def test_sub_microsecond_timestamps_are_truncated(self, tmp_path):
        """Nanosecond precision is lost to the microsecond column.

        Test scenario:
            The column is `timestamp[us]`, so the last three digits of a
            nanosecond timestamp are dropped — the third documented loss.
        """
        item = _item("a", 1.0, 2.0)
        item["properties"]["datetime"] = "2024-01-02T03:04:05.123456789Z"
        path = str(tmp_path / "nanos.parquet")
        to_geoparquet_spec([item], path)

        stamp = from_geoparquet_spec(path)[0]["properties"]["datetime"]
        assert stamp == "2024-01-02T03:04:05.123456Z", f"truncation changed: {stamp}"


class TestSpecFlagRaises:
    """The `Raises` section of `to_geoparquet(..., spec=True)` is accurate."""

    def test_the_spec_writer_names_itself_in_the_empty_error(self, tmp_path):
        """The empty-input error names the function that actually raised.

        Test scenario:
            `to_geoparquet_spec([])` used to report "to_geoparquet received no
            items."
        """
        with pytest.raises(ValueError, match="to_geoparquet_spec received no items"):
            to_geoparquet_spec([], str(tmp_path / "x.parquet"))

    def test_the_blob_writer_still_names_itself(self, tmp_path):
        """The blob path's own message is unchanged.

        Test scenario:
            Only the spec writer's message was wrong.
        """
        with pytest.raises(ValueError, match="to_geoparquet received no items"):
            to_geoparquet([], str(tmp_path / "x.parquet"))

    def test_the_flag_raises_on_a_reserved_property_name(self, tmp_path):
        """`spec=True` surfaces the reserved-column ValueError.

        Test scenario:
            The documented `Raises` for `to_geoparquet` must cover the
            delegated branch.
        """
        item = _item("a", 1.0, 2.0)
        item["properties"]["bbox"] = [0, 0, 1, 1]
        with pytest.raises(ValueError, match="reserves the column names"):
            to_geoparquet([item], str(tmp_path / "x.parquet"), spec=True)

    def test_the_flag_raises_on_a_bad_item_type(self, tmp_path):
        """`spec=True` surfaces the `TypeError` from the item normaliser.

        Test scenario:
            The blob path raises the same `TypeError`, but only the spec path's
            docstring omitted it.
        """
        with pytest.raises(TypeError, match="to_dict"):
            to_geoparquet([123], str(tmp_path / "x.parquet"), spec=True)
