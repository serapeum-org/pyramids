"""Tests for pyramids.stac._geoparquet (STAC <-> GeoParquet round-trip, PD-3)."""

from __future__ import annotations

import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.stac._geoparquet import from_geoparquet, to_geoparquet

pytestmark = pytest.mark.core

try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

requires_pyarrow = pytest.mark.skipif(
    not HAS_PYARROW, reason="pyarrow ([parquet] extra) not installed"
)


def _item(item_id, lon, lat):
    """A minimal STAC item dict with a Point geometry."""
    return {
        "type": "Feature",
        "id": item_id,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "bbox": [lon, lat, lon, lat],
        "properties": {"datetime": "2023-06-01T00:00:00Z", "eo:cloud_cover": 12},
        "assets": {"data": {"href": f"s3://b/{item_id}.tif", "type": "image/tiff"}},
        "stac_extensions": [],
    }


class TestToGeoparquetGuards:
    """Argument/format guards that need no pyarrow."""

    def test_empty_items_raises(self):
        """No items raises a clear ValueError (before any Parquet write).

        Test scenario:
            An empty list cannot be serialised.
        """
        with pytest.raises(ValueError, match="no items"):
            to_geoparquet([], "x.parquet")

    def test_bad_item_type_raises(self):
        """A non-dict, non-pystac item raises TypeError.

        Test scenario:
            An int is neither a dict nor exposes to_dict().
        """
        with pytest.raises(TypeError, match="to_dict"):
            to_geoparquet([123], "x.parquet")


@requires_pyarrow
class TestRoundTrip:
    """Round-trip items through GeoParquet (needs pyarrow)."""

    def test_round_trip_preserves_items(self, tmp_path):
        """Items survive a to_geoparquet -> from_geoparquet round-trip.

        Test scenario:
            Two items written and read back equal the originals (id,
            properties, assets, geometry).
        """
        items = [_item("a", 1.0, 2.0), _item("b", 3.0, 4.0)]
        path = str(tmp_path / "items.parquet")
        to_geoparquet(items, path)
        restored = from_geoparquet(path)
        assert [r["id"] for r in restored] == [
            "a",
            "b",
        ], f"ids: {[r['id'] for r in restored]}"
        assert restored[0]["properties"]["eo:cloud_cover"] == 12, restored[0][
            "properties"
        ]
        assert restored[0]["assets"]["data"]["href"] == "s3://b/a.tif", restored[0][
            "assets"
        ]
        assert restored[0]["geometry"]["coordinates"] == [1.0, 2.0], restored[0][
            "geometry"
        ]

    def test_round_trip_feeds_from_stac(self, tmp_path):
        """Restored items can drive from_stac (the intended consumer).

        Test scenario:
            Write items pointing at real local rasters, read back, and build a
            collection from the restored dicts.
        """
        import numpy as np

        from pyramids.dataset import Dataset, DatasetCollection

        items = []
        for i in range(2):
            p = str(tmp_path / f"r{i}.tif")
            Dataset.create_from_array(
                np.ones((3, 3), "float32"),
                top_left_corner=(0.0, 3.0),
                cell_size=1.0,
                epsg=4326,
            ).to_file(p)
            items.append(
                {
                    "id": f"r{i}",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]],
                    },
                    "bbox": [0.0, 0.0, 3.0, 3.0],
                    "properties": {"datetime": "2023-06-0%dT00:00:00Z" % (i + 1)},
                    "assets": {"data": {"href": p, "type": "image/tiff"}},
                }
            )
        path = str(tmp_path / "scenes.parquet")
        to_geoparquet(items, path)
        coll = DatasetCollection.from_stac(from_geoparquet(path), asset="data")
        assert coll.time_length == 2, f"expected 2 timesteps, got {coll.time_length}"

    def test_from_stac_item_round_trip(self, tmp_path):
        """A Dataset.to_stac_item dict round-trips through GeoParquet.

        Test scenario:
            to_stac_item -> to_geoparquet -> from_geoparquet preserves proj:code.
        """
        import numpy as np

        from pyramids.dataset import Dataset

        ds = Dataset.create_from_array(
            np.ones((4, 4), "float32"),
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        item = ds.to_stac_item("scene-1", asset_href="s3://b/s.tif")
        path = str(tmp_path / "one.parquet")
        to_geoparquet([item], path)
        restored = from_geoparquet(path)[0]
        assert restored["properties"]["proj:code"] == "EPSG:4326", restored[
            "properties"
        ]
