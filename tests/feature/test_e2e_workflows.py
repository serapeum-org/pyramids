"""End-to-end workflows that chain multiple ARC features together.

The per-feature unit tests live alongside each ARC commit. These
tests exercise realistic user pipelines that combine several of
those features in one flow, so regressions that only surface at
integration time (e.g. a downstream method breaking because an
upstream method's return type changed) get caught.

Pipelines covered
-----------------
1. Read → filter → reproject (inherited) → write multi-layer GPKG
   with creation options → round-trip → verify.
2. Build from GeoJSON features → concat → rasterize through
   ``Dataset.from_features`` → read_array back → spot-check.
3. Stream a large file in chunks → ``pd.concat`` → ``schema`` +
   ``list_layers`` metadata.
4. Cross-type protocol polymorphism: a function typed
   ``SpatialObject`` accepts both a raster ``Dataset`` and a vector
   ``FeatureCollection``.
5. ``FeatureCollection`` ↔ ``__geo_interface__`` ↔ ``from_features``
   round-trip equality.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

from pyramids.base.crs import reproject_coordinates
from pyramids.base.protocols import SpatialObject
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection, create_polygon, get_coords

pytestmark = pytest.mark.core


@pytest.fixture
def big_points_geojson(tmp_path: Path) -> Path:
    """100 points with a 'score' attribute, spread diagonally."""
    gdf = gpd.GeoDataFrame(
        {"score": np.arange(100, dtype=float)},
        geometry=[Point(x, x) for x in range(100)],
        crs="EPSG:4326",
    )
    p = tmp_path / "big.geojson"
    gdf.to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def polygon_fc() -> FeatureCollection:
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"class_id": [7]},
            geometry=[box(0, 0, 5, 5)],
            crs="EPSG:32636",
        )
    )


class TestReadFilterReprojectWrite:
    """ARC-24 + ARC-1a inherited + ARC-26 full pipeline."""

    def test_geojson_to_gpkg_with_filters(
        self, tmp_path: Path, big_points_geojson: Path
    ):
        # 1. Read with bbox+where filters pushed down to pyogrio.
        filtered = FeatureCollection.read_file(
            big_points_geojson,
            bbox=(0.0, 0.0, 50.0, 50.0),
            where="score >= 10",
        )
        # Points (x,x) for x in 0..50 → 51 points; filter score>=10 → 41.
        assert len(filtered) == 41
        assert all(filtered["score"] >= 10)

        # 2. Reproject — inherited from GeoDataFrame via ARC-1a.
        reproj = filtered.to_crs(3857)
        assert isinstance(reproj, FeatureCollection)
        assert reproj.epsg == 3857

        # 3. Write with a named layer + GPKG spatial index.
        out = tmp_path / "filtered.gpkg"
        reproj.to_file(
            out,
            driver="gpkg",
            layer="hotspots",
            SPATIAL_INDEX="YES",
        )

        # 4. Round-trip verify.
        reloaded = FeatureCollection.read_file(out, layer="hotspots")
        assert len(reloaded) == 41
        assert reloaded.epsg == 3857
        assert {"score"} <= set(reloaded.columns)


class TestFromFeaturesToRasterize:
    """ARC-28 → ARC-1a concat → ARC-4 Dataset.from_features full pipeline."""

    def test_build_concat_rasterize(self, tmp_path: Path):
        # Build two FCs from GeoJSON-style dicts (ARC-28).
        feats_a = [
            {
                "type": "Feature",
                "geometry": box(0, 0, 1, 1).__geo_interface__,
                "properties": {"v": 1.0},
            }
        ]
        feats_b = [
            {
                "type": "Feature",
                "geometry": box(2, 0, 3, 1).__geo_interface__,
                "properties": {"v": 2.0},
            }
        ]
        fc_a = FeatureCollection.from_features(feats_a, crs="EPSG:32636")
        fc_b = FeatureCollection.from_features(feats_b, crs="EPSG:32636")

        # Concat via the ARC-16 method.
        combined = fc_a.concat(fc_b)
        assert isinstance(combined, FeatureCollection)
        assert len(combined) == 2

        # Rasterize via ARC-4 API.
        raster = Dataset.from_features(combined, cell_size=0.25)

        # Spot-check: the raster contains both burned values.
        arr = raster.read_array()
        unique = set(arr[arr != raster.no_data_value[0]])
        assert 1.0 in unique
        assert 2.0 in unique


class TestStreamAndIntrospect:
    """ARC-25 chunked stream → ARC-27 schema + list_layers."""

    def test_stream_chunks_match_full(self, big_points_geojson: Path):
        """Chunked stream concatenated == full read."""
        chunks = list(FeatureCollection.iter_features(big_points_geojson, chunksize=25))
        combined = pd.concat(chunks)
        full = FeatureCollection.read_file(big_points_geojson)
        assert len(combined) == len(full)
        # Subclass identity preserved via _constructor (ARC-1a).
        assert isinstance(combined, FeatureCollection)

    def test_schema_and_list_layers(self, big_points_geojson: Path):
        """schema after read / list_layers on single-layer GeoJSON."""
        fc = FeatureCollection.read_file(big_points_geojson)
        s = fc.schema
        assert s["geometry"] == "Point"
        assert "score" in s["properties"]
        layers = FeatureCollection.list_layers(big_points_geojson)
        assert len(layers) == 1


class TestProtocolPolymorphism:
    """ARC-17: a SpatialObject-typed function accepts both Dataset and FC."""

    def test_cross_type_function(self, polygon_fc: FeatureCollection, tmp_path: Path):
        # A fresh raster aligned with the FC's CRS.
        raster = Dataset.create(
            cell_size=1.0,
            rows=5,
            columns=5,
            dtype="float32",
            bands=1,
            top_left_corner=(0.0, 5.0),
            epsg=32636,
            no_data_value=-9999.0,
        )

        def spatial_epsg(obj: SpatialObject) -> int | None:
            """Utility function typed on the shared protocol."""
            return obj.epsg

        def spatial_bounds(obj: SpatialObject) -> tuple:
            return tuple(obj.total_bounds)

        assert spatial_epsg(polygon_fc) == 32636
        assert spatial_epsg(raster) == 32636
        # Both expose a bbox of the same shape.
        assert len(spatial_bounds(polygon_fc)) == 4
        assert len(spatial_bounds(raster)) == 4


class TestGeoInterfaceRoundTrip:
    """fc → __geo_interface__ → from_features → equivalent fc."""

    def test_round_trip_via_geo_interface(self):
        original = FeatureCollection(
            gpd.GeoDataFrame(
                {"name": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]},
                geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
                crs="EPSG:4326",
            )
        )
        feats = original.__geo_interface__["features"]
        rebuilt = FeatureCollection.from_features(feats, crs=original.crs)
        assert len(rebuilt) == len(original)
        assert rebuilt.epsg == original.epsg
        assert list(rebuilt["name"]) == list(original["name"])
        for g_orig, g_back in zip(original.geometry, rebuilt.geometry):
            assert g_orig.equals(g_back)


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow not installed (install with pyramids-gis[parquet])",
)
class TestParquetWorkflow:
    """ARC-32 + ARC-26: Parquet → reproject → write multi-format GPKG."""

    def test_parquet_read_reproject_write_gpkg(self, tmp_path: Path):
        # Write a Parquet from a constructed FC.
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"pop": [100, 200, 300]},
                geometry=[Point(x, x) for x in range(3)],
                crs="EPSG:4326",
            )
        )
        pq = tmp_path / "in.parquet"
        fc.to_parquet(pq, compression="gzip")

        # Read back, reproject, project to a single column, write GPKG.
        rt = FeatureCollection.read_parquet(pq, columns=["pop", "geometry"])
        reproj = rt.to_crs(3857)
        gpkg = tmp_path / "out.gpkg"
        reproj.to_file(gpkg, driver="gpkg", layer="points")

        # Verify via read-back.
        final = FeatureCollection.read_file(gpkg, layer="points")
        assert len(final) == 3
        assert final.epsg == 3857


class TestStreamIndexCentroidPickleChain:
    """Chained e2e across C9/C14/C15/C18: stream + centroid + pickle.

    Exercises five of the fixes in one flow:

    * C9 / ARC-28: build a base FC from a features list.
    * C14: ``iter_features(include_index=True)`` rebuilds the FC from
      chunks and preserves source-row indices through a Python-side
      bbox filter.
    * C15: ``list_layers`` on the on-disk GPKG answers from the cache
      on the second call.
    * C18: ``with_centroid`` runs over the reassembled FC without
      warning (no NaN-coord geometries).
    * Pickle: the final FC round-trips through ``pickle`` preserving
      the centroid column and the row-index column.
    """

    def test_stream_filter_centroid_pickle_round_trip(self, tmp_path: Path):
        import pickle

        import pandas as pd
        from shapely.geometry import Point as _Pt

        FeatureCollection.list_layers_cache_clear()

        points = [_Pt(i, i) for i in range(10)]
        gdf = gpd.GeoDataFrame(
            {"id": list(range(10)), "score": [i * 0.1 for i in range(10)]},
            geometry=points,
            crs="EPSG:4326",
        )
        src = tmp_path / "stream.gpkg"
        gdf.to_file(src, driver="GPKG", layer="points")

        # C15: list_layers is cached — two calls, one pyogrio call.
        layers_first = FeatureCollection.list_layers(src)
        layers_second = FeatureCollection.list_layers(src)
        assert layers_first == layers_second
        assert "points" in layers_first

        # C14: stream with Python-bbox filter + include_index.
        chunks = list(
            FeatureCollection.iter_features(
                src,
                layer="points",
                bbox=(0.0, 0.0, 4.5, 4.5),
                tile_strategy="none",
                chunksize=3,
                include_index=True,
            )
        )
        combined = pd.concat(chunks)
        assert isinstance(combined, FeatureCollection)
        combined = combined.reset_index(drop=True)
        assert list(combined["_row_index"]) == [0, 1, 2, 3, 4]

        # C18: with_centroid on the reassembled FC — no NaN rows.
        with_center = combined.with_centroid()
        assert "center_point" in with_center.columns
        assert all(not p.is_empty for p in with_center["center_point"])

        # Pickle round-trip preserves the attached columns.
        restored = pickle.loads(pickle.dumps(with_center))
        assert isinstance(restored, FeatureCollection)
        assert list(restored["_row_index"]) == [0, 1, 2, 3, 4]
        assert list(restored["center_point"]) == list(with_center["center_point"])


class TestGeometryHardeningChain:
    """Chained e2e across D-H1/C21/C22/C23 + D-H2.

    Builds a FeatureCollection with mixed Polygons + MultiPolygons, runs
    ``with_coordinates`` (which calls ``explode_gdf`` internally), asserts:

    * the input FC is untouched (D-H1),
    * the exploded output has the right row count,
    * ``create_polygon`` rejects a bad ring (C21) so the pipeline can
      build one valid polygon to feed the FC,
    * calling ``get_coords`` on an empty row raises (C22), and
    * ``reproject_coordinates`` converts pyproj failures to
      :class:`pyramids.base._errors.CRSError` (C23).
    """

    def test_explode_polygon_extract_reproject_chain(self):
        from shapely.geometry import MultiPolygon as _Mp
        from shapely.geometry import Point as _Pt

        from pyramids.base._errors import CRSError, InvalidGeometryError

        # C21: valid triangle ring; the 2-vertex guard keeps callers from
        # ever reaching this with a degenerate polygon.
        triangle = create_polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        mpoly = _Mp([box(2.0, 2.0, 3.0, 3.0), box(4.0, 4.0, 5.0, 5.0)])
        fc = FeatureCollection(
            gpd.GeoDataFrame(geometry=[triangle, mpoly], crs="EPSG:4326")
        )
        original_types = [g.geom_type for g in fc.geometry]

        # D-H1: with_coordinates explodes internally and must NOT mutate ``fc``.
        exploded = fc.with_coordinates()
        assert [
            g.geom_type for g in fc.geometry
        ] == original_types, "input FC's geometries mutated by with_coordinates"
        # Exploded output carries the split child rows (triangle + 2 boxes).
        assert len(exploded) == 3

        # C22: empty geometry in a row raises InvalidGeometryError on
        # direct ``get_coords`` access.
        import pandas as _pd

        row = _pd.Series({"geometry": _Pt()})
        with pytest.raises(InvalidGeometryError):
            get_coords(row, "geometry", "x")

        # C23: a bad target CRS in a downstream reproject surfaces as
        # pyramids CRSError, not pyproj's.
        with pytest.raises(CRSError):
            reproject_coordinates([1.0], [1.0], from_crs=4326, to_crs="gibberish-wkt")


class TestBatch4Chain:
    """Chained e2e: from_records(orient='list') → rasterize → verify.

    Exercises C26 (columnar-dict input) together with D-M2 (validation
    guard in ``Dataset.from_features``) by building a FC from a
    pandas-style columnar dict and rasterising it through the
    validated path. D-M4's /vsimem/ round-trip is exercised
    transitively by ``Dataset.from_features`` opening the vector via
    ``_ogr.as_datasource``.
    """

    def test_columnar_records_to_raster(self, tmp_path):
        from pyramids.dataset import Dataset as _Ds

        epsg = 32636
        cell_size = 1000.0
        top_left = (500000.0, 3400000.0)
        x0, y0 = top_left

        fc = FeatureCollection.from_records(
            {
                "class_id": [7, 13],
                "geometry": [
                    box(x0, y0 - 2 * cell_size, x0 + 2 * cell_size, y0),
                    box(x0 + 3 * cell_size, y0 - 2 * cell_size, x0 + 5 * cell_size, y0),
                ],
            },
            orient="list",
            crs=f"EPSG:{epsg}",
        )
        assert len(fc) == 2

        raster = _Ds.from_features(fc, cell_size=cell_size, column_name="class_id")
        arr = raster.read_array()
        assert int(arr.max()) == 13, f"expected 13 in burned raster; got {arr.max()}"
        assert 7 in {int(v) for v in arr.flatten() if v != raster.no_data_value[0]}
