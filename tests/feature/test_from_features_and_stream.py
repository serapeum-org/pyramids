"""ARC-25 + ARC-28: streaming iteration + from-records constructors.

ARC-28 adds two classmethods:

* ``FeatureCollection.from_features(iterable, *, crs=None, columns=None)``
* ``FeatureCollection.from_records(records, *, geometry='geometry', crs=None)``

ARC-25 adds:

* ``FeatureCollection.iter_features(path, *, layer=None, bbox=None,
  where=None, chunksize=None)``
  — generator that either yields GeoJSON-style dicts (default) or
    FeatureCollection chunks (``chunksize=N``) without loading the
    whole file up front.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base._errors import FeatureError
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture
def small_geojson(tmp_path: Path) -> Path:
    """6 points with id + score attributes."""
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        geometry=[
            Point(0, 0),
            Point(1, 1),
            Point(2, 2),
            Point(3, 3),
            Point(4, 4),
            Point(5, 5),
        ],
        crs="EPSG:4326",
    )
    p = tmp_path / "six.geojson"
    gdf.to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def larger_geojson(tmp_path: Path) -> Path:
    """50 points to exercise batching."""
    gdf = gpd.GeoDataFrame(
        {"i": list(range(50))},
        geometry=[Point(x, x) for x in range(50)],
        crs="EPSG:4326",
    )
    p = tmp_path / "fifty.geojson"
    gdf.to_file(p, driver="GeoJSON")
    return p


# ── ARC-28 : from_features ──────────────────────────────────────────


class TestFromFeatures:
    """Build a FeatureCollection from GeoJSON-feature dicts."""

    def test_basic_round_trip(self):
        feats = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {"name": "a"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1, 1]},
                "properties": {"name": "b"},
            },
        ]
        fc = FeatureCollection.from_features(feats, crs=4326)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 2
        assert fc.epsg == 4326
        assert list(fc["name"]) == ["a", "b"]

    def test_accepts_generator(self):
        """Iterator input (not just a list) is accepted."""

        def gen():
            for i in range(3):
                yield {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [i, i]},
                    "properties": {"k": i},
                }

        fc = FeatureCollection.from_features(gen(), crs=4326)
        assert len(fc) == 3
        assert list(fc["k"]) == [0, 1, 2]

    def test_without_crs(self):
        feats = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {"v": 1},
            }
        ]
        fc = FeatureCollection.from_features(feats)
        assert fc.epsg is None

    def test_empty_iterable_raises(self):
        """C9: empty iterable → ``ValueError``.

        Test scenario:
            A caller passing an empty list (or an exhausted iterator)
            previously got an empty GeoDataFrame with no ``geometry``
            column, which later broke every pyramids method that
            assumes the column exists. The method now raises
            ``ValueError`` up front.
        """
        with pytest.raises(ValueError, match="at least one feature"):
            FeatureCollection.from_features([])

    def test_exhausted_iterator_raises(self):
        """C9: an exhausted iterator is equivalent to an empty list."""

        def empty_gen():
            return
            yield  # pragma: no cover — makes this a generator

        with pytest.raises(ValueError, match="at least one feature"):
            FeatureCollection.from_features(empty_gen())

    def test_empty_tuple_raises(self):
        """C9: an empty tuple is rejected the same as an empty list."""
        with pytest.raises(ValueError, match="at least one feature"):
            FeatureCollection.from_features(())

    def test_error_message_mentions_reason(self):
        """C9: the error names ``geometry column`` so the cause is visible."""
        with pytest.raises(ValueError) as exc_info:
            FeatureCollection.from_features([])
        msg = str(exc_info.value)
        assert "geometry column" in msg, f"error message should explain why; got: {msg}"

    def test_columns_order(self):
        feats = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {"b": 2, "a": 1},
            }
        ]
        fc = FeatureCollection.from_features(
            feats, crs=4326, columns=["a", "b", "geometry"]
        )
        assert list(fc.columns)[:2] == ["a", "b"]

    def test_roundtrip_via_geo_interface(self, small_geojson: Path):
        """fc.from_features(fc.__geo_interface__['features']) reconstructs."""
        original = FeatureCollection.read_file(small_geojson)
        feats = original.__geo_interface__["features"]
        rebuilt = FeatureCollection.from_features(feats, crs=original.crs)
        assert len(rebuilt) == len(original)
        assert rebuilt.epsg == original.epsg


# ── ARC-28 : from_records ───────────────────────────────────────────


class TestFromRecords:
    """Build a FeatureCollection from dict records with shapely geoms."""

    def test_basic(self):
        records = [
            {"geometry": Point(0, 0), "n": 1, "k": "a"},
            {"geometry": Point(1, 1), "n": 2, "k": "b"},
        ]
        fc = FeatureCollection.from_records(records, crs=4326)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 2
        assert fc.epsg == 4326
        assert list(fc["n"]) == [1, 2]

    def test_custom_geometry_column(self):
        records = [
            {"geom": Point(0, 0), "v": 10},
            {"geom": Point(1, 1), "v": 20},
        ]
        fc = FeatureCollection.from_records(records, geometry="geom", crs=4326)
        assert len(fc) == 2
        # With a non-default geometry column name, that column IS the
        # active geometry column.
        assert fc.geometry.name == "geom"

    def test_missing_geometry_raises(self):
        records = [{"v": 1}, {"v": 2}]
        with pytest.raises(FeatureError, match="geometry"):
            FeatureCollection.from_records(records, crs=4326)

    def test_empty_records(self):
        fc = FeatureCollection.from_records([], crs=4326)
        assert len(fc) == 0

    def test_orient_list_basic(self):
        """C26: columnar dict input via ``orient="list"``."""
        fc = FeatureCollection.from_records(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            },
            orient="list",
            crs=4326,
        )
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 3
        assert list(fc["id"]) == [1, 2, 3]
        assert fc.epsg == 4326

    def test_orient_list_custom_geometry_column(self):
        """C26: ``orient="list"`` honours a non-default geometry column."""
        fc = FeatureCollection.from_records(
            {
                "v": [10, 20],
                "geom": [Point(0, 0), Point(1, 1)],
            },
            orient="list",
            geometry="geom",
            crs=4326,
        )
        assert fc.geometry.name == "geom"
        assert len(fc) == 2

    def test_orient_list_empty(self):
        """C26: empty columnar dict yields a zero-row FC."""
        fc = FeatureCollection.from_records(
            {"id": [], "geometry": []},
            orient="list",
            crs=4326,
        )
        assert len(fc) == 0

    def test_orient_list_rejects_non_dict(self):
        """C26: passing a list under ``orient="list"`` raises clearly."""
        with pytest.raises(ValueError, match="dict of column"):
            FeatureCollection.from_records(
                [{"geometry": Point(0, 0)}],
                orient="list",
            )

    def test_invalid_orient_raises(self):
        """C26: unknown ``orient`` value is rejected."""
        with pytest.raises(ValueError, match=r"records.*list"):
            FeatureCollection.from_records(
                [{"geometry": Point(0, 0)}],
                orient="index",
            )

    def test_orient_list_missing_geometry_column_raises(self):
        """C26: columnar dict without the geometry key raises ``FeatureError``."""
        with pytest.raises(FeatureError, match="geometry"):
            FeatureCollection.from_records(
                {"v": [1, 2]},
                orient="list",
                crs=4326,
            )

    def test_orient_list_mismatched_lengths_raises(self):
        """C26: pandas surfaces mismatched-length columns as ValueError."""
        with pytest.raises(ValueError):
            FeatureCollection.from_records(
                {"v": [1, 2, 3], "geometry": [Point(0, 0)]},
                orient="list",
            )


# ── ARC-25 : iter_features dict mode ────────────────────────────────


class TestIterFeaturesDictMode:
    """``chunksize=None`` yields per-feature dicts."""

    def test_yields_dicts(self, small_geojson: Path):
        feats = list(FeatureCollection.iter_features(small_geojson))
        assert len(feats) == 6
        assert all(isinstance(f, dict) for f in feats)
        assert feats[0]["geometry"]["type"] == "Point"

    def test_total_matches_full_read(self, small_geojson: Path):
        total_streamed = sum(1 for _ in FeatureCollection.iter_features(small_geojson))
        total_loaded = len(FeatureCollection.read_file(small_geojson))
        assert total_streamed == total_loaded

    def test_bbox_filter_streamed(self, small_geojson: Path):
        feats = list(
            FeatureCollection.iter_features(small_geojson, bbox=(0.0, 0.0, 2.5, 2.5))
        )
        # points at (0,0), (1,1), (2,2) fall inside
        assert len(feats) == 3

    def test_where_filter_streamed(self, small_geojson: Path):
        feats = list(
            FeatureCollection.iter_features(small_geojson, where="score > 0.3")
        )
        assert len(feats) == 3  # scores 0.4, 0.5, 0.6


# ── ARC-25 : iter_features chunked mode ─────────────────────────────


class TestIterFeaturesChunked:
    """``chunksize=N`` yields FeatureCollection batches of up to N rows."""

    def test_yields_feature_collections(self, larger_geojson: Path):
        chunks = list(FeatureCollection.iter_features(larger_geojson, chunksize=10))
        assert all(isinstance(c, FeatureCollection) for c in chunks)
        # 50 features / 10 per chunk → 5 chunks.
        assert len(chunks) == 5
        assert all(len(c) == 10 for c in chunks)

    def test_last_chunk_can_be_short(self, larger_geojson: Path):
        """50 features at chunksize=15 → 15 + 15 + 15 + 5."""
        chunks = list(FeatureCollection.iter_features(larger_geojson, chunksize=15))
        sizes = [len(c) for c in chunks]
        assert sizes == [15, 15, 15, 5]

    def test_chunk_concat_matches_full(self, larger_geojson: Path):
        """Concatenating every chunk yields the full dataset."""
        import pandas as pd

        chunks = list(FeatureCollection.iter_features(larger_geojson, chunksize=7))
        combined = pd.concat(chunks)
        assert len(combined) == 50
        # Subclass identity survives pd.concat via _constructor.
        assert isinstance(combined, FeatureCollection)

    def test_chunksize_less_than_one_raises(self, small_geojson: Path):
        with pytest.raises(ValueError, match="chunksize"):
            list(FeatureCollection.iter_features(small_geojson, chunksize=0))

    def test_chunked_with_filters(self, larger_geojson: Path):
        """bbox / where compose with chunking."""
        chunks = list(
            FeatureCollection.iter_features(
                larger_geojson,
                bbox=(0.0, 0.0, 9.5, 9.5),
                chunksize=4,
            )
        )
        # Points (0,0)..(9,9) → 10 features; chunks of 4 → 4+4+2.
        total = sum(len(c) for c in chunks)
        assert total == 10


class TestIterFeaturesIncludeIndex:
    """C14: ``include_index=True`` attaches source row indices.

    Users streaming features often need to correlate a yielded feature
    back to its on-disk row (for logging, error reporting, or writing
    a result file at the same row). The ``include_index=True`` flag
    adds an ``"id"`` key to each yielded dict (unchunked) or a
    ``"_row_index"`` column to each yielded FC (chunked).
    """

    def test_per_feature_include_index_adds_id(self, small_geojson: Path):
        """Unchunked + include_index injects sequential ``id`` keys."""
        feats = list(FeatureCollection.iter_features(small_geojson, include_index=True))
        ids = [f["id"] for f in feats]
        assert ids == list(range(len(feats)))

    def test_per_feature_default_id_is_not_row_index(self, small_geojson: Path):
        """include_index=False preserves whatever ``id`` geopandas emits.

        Test scenario:
            ``geopandas.GeoDataFrame.iterfeatures`` always injects an
            ``"id"`` key per GeoJSON convention, but its value is not
            the absolute source-row index; the ``include_index=True``
            branch overrides it to 0-based sequential. With the flag
            off, the key (if present) must NOT already be the
            absolute row index.
        """
        feats = list(FeatureCollection.iter_features(small_geojson))
        ids = [f.get("id") for f in feats]
        # Not every geopandas version sets "id" in every feature; the
        # only invariant we care about is that without the flag, we do
        # NOT overwrite the value to the absolute row index.
        assert ids != list(range(len(feats))) or all(i is None for i in ids)

    def test_chunked_include_index_adds_row_index_column(self, larger_geojson: Path):
        """Chunked + include_index adds a ``_row_index`` column per chunk."""
        chunks = list(
            FeatureCollection.iter_features(
                larger_geojson, chunksize=10, include_index=True
            )
        )
        assert all("_row_index" in c.columns for c in chunks)
        first = chunks[0]
        assert list(first["_row_index"]) == list(range(10))
        second = chunks[1]
        assert list(second["_row_index"]) == list(range(10, 20))

    def test_include_index_survives_python_bbox_filter(self, larger_geojson: Path):
        """With tile_strategy='none' the bbox filter drops rows in Python;
        yielded indices must match the surviving source rows.
        """
        feats = list(
            FeatureCollection.iter_features(
                larger_geojson,
                bbox=(0.0, 0.0, 4.5, 4.5),
                tile_strategy="none",
                include_index=True,
            )
        )
        ids = [f["id"] for f in feats]
        # Points (0,0)..(4,4) are at indices 0..4 in the file.
        assert ids == [0, 1, 2, 3, 4]

    def test_chunked_include_index_with_python_bbox_filter(self, larger_geojson: Path):
        """C14: Python-side bbox filter drops rows; surviving ``_row_index``
        values still match the absolute source positions.
        """
        chunks = list(
            FeatureCollection.iter_features(
                larger_geojson,
                bbox=(0.0, 0.0, 4.5, 4.5),
                tile_strategy="none",
                chunksize=3,
                include_index=True,
            )
        )
        all_ids: list[int] = []
        for c in chunks:
            all_ids.extend(int(x) for x in c["_row_index"].tolist())
        assert all_ids == [0, 1, 2, 3, 4]

    def test_chunksize_one_include_index(self, larger_geojson: Path):
        """C14 boundary: chunksize=1 still attaches sequential indices."""
        chunks = list(
            FeatureCollection.iter_features(
                larger_geojson, chunksize=1, include_index=True
            )
        )
        ids = [int(c["_row_index"].iloc[0]) for c in chunks]
        assert ids == list(range(len(ids)))


class TestIterFeaturesEnginePin:
    """D-M3: iter_features pins ``engine="pyogrio"`` on gpd.read_file.

    ``skip_features`` / ``max_features`` are pyogrio-specific kwargs;
    fiona silently ignores them, which would make every chunk a full
    scan. The function must force the engine explicitly so a
    ``geopandas.options.io_engine = "fiona"`` global doesn't quietly
    break pagination.
    """

    def test_read_file_receives_engine_pyogrio(self, larger_geojson: Path, monkeypatch):
        captured: list = []
        import geopandas

        real_read_file = geopandas.read_file

        def _spy(path, **kwargs):
            captured.append(kwargs)
            return real_read_file(path, **kwargs)

        monkeypatch.setattr(geopandas, "read_file", _spy)
        list(FeatureCollection.iter_features(larger_geojson, chunksize=10))
        assert captured, "gpd.read_file must be invoked at least once"
        assert all(
            k.get("engine") == "pyogrio" for k in captured
        ), f"expected engine='pyogrio' in every call; got {captured}"
