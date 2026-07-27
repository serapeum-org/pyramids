"""Regression tests for the arc-feature architecture-review fixes.

Each test pins a specific finding from ``planning/architecture-review/25-july/arc-feature.md`` so the fix cannot
silently regress. Grouped by ARC id.
"""

from __future__ import annotations

import base64

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from pyramids.feature import FeatureCollection, _wfs
from pyramids.feature.geometry import create_points, explode_gdf
from pyramids.feature.tessellation import fishnet_cells


class TestArc16ExplodeGdf:
    """ARC-16: explode_gdf must select rows by mask, not index label."""

    def test_non_unique_index_preserves_all_rows(self):
        """A non-matching row sharing a duplicate index label is not dropped.

        Test scenario:
            A dup-index frame [MultiPolygon, Polygon] both labelled 0 — the old label-drop deleted the plain
            polygon; the mask-based selection keeps it (preserved rows first, then the exploded children).
        """
        multi = MultiPolygon(
            [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), Polygon([(5, 5), (7, 5), (7, 7), (5, 7)])]
        )
        plain = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
        gdf = gpd.GeoDataFrame({"name": ["multi", "plain"]}, geometry=[multi, plain], crs="EPSG:4326")
        gdf.index = [0, 0]
        result = explode_gdf(gdf, "multipolygon")
        assert list(result["name"]) == ["plain", "multi", "multi"], f"lost data: {list(result['name'])}"
        assert len(result) == 3, f"expected 3 rows, got {len(result)}"

    def test_input_not_mutated(self):
        """The caller's frame is untouched by the explode."""
        multi = MultiPolygon(
            [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])]
        )
        gdf = gpd.GeoDataFrame(geometry=[multi], crs="EPSG:4326")
        _ = explode_gdf(gdf, "multipolygon")
        assert gdf.iloc[0].geometry.geom_type == "MultiPolygon", "input frame was mutated"

    def test_explode_on_featurecollection_uses_geopandas_explode(self):
        """A FeatureCollection input does not shadow geopandas' explode with its own override.

        Test scenario:
            FeatureCollection.explode → explode_gdf(self, …); the internal `.explode(index_parts=…)` must reach
            GeoDataFrame.explode, not FeatureCollection.explode(geometry=…) (the FC is-a GeoDataFrame trap).
        """
        multi = MultiPolygon(
            [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), Polygon([(5, 5), (7, 5), (7, 7), (5, 7)])]
        )
        fc = FeatureCollection(gpd.GeoDataFrame({"n": ["m"]}, geometry=[multi], crs="EPSG:4326"))
        result = fc.explode("multipolygon")
        assert len(result) == 2, f"expected 2 exploded polygons, got {len(result)}"
        assert set(result.geom_type) == {"Polygon"}


class TestArc31Schema:
    """ARC-31: schema excludes the active geometry column by name, not the literal 'geometry'."""

    def test_renamed_geometry_column_excluded(self):
        """A renamed active geometry column is excluded; a real non-geometry column stays."""
        fc = FeatureCollection(
            gpd.GeoDataFrame({"v": [1]}, geometry=gpd.GeoSeries([Point(0, 0)], name="geom"), crs=4326)
        )
        props = fc.schema["properties"]
        assert props == {"v": "int64"}, f"renamed geometry column leaked or column dropped: {props}"


class TestArc31IterFeaturesIndexGuard:
    """ARC-31: include_index is incompatible with driver-side filtering (it would emit wrong ids)."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"where": "a=1"},
            {"bbox": (0.0, 0.0, 1.0, 1.0)},
            {"bbox": (0.0, 0.0, 1.0, 1.0), "tile_strategy": "rtree"},
        ],
    )
    def test_include_index_with_driver_filter_raises(self, kwargs):
        """include_index combined with a where / pushed-down bbox raises ValueError on first iteration.

        Args:
            kwargs: A driver-side-filter combination that would misalign the emitted id.

        Test scenario:
            The generator raises when first advanced, rather than emitting wrong absolute row positions.
        """
        gen = FeatureCollection.iter_features("x", include_index=True, **kwargs)
        with pytest.raises(ValueError, match="incompatible with driver-side"):
            next(gen)


class TestArc56Vectorized:
    """ARC-56: vectorized fishnet / create_points keep the exact prior semantics."""

    def test_fishnet_cells_grid(self):
        """fishnet_cells returns the same grid, cell bounds and row/col order as the loop version."""
        polys, rows, cols = fishnet_cells((0.0, 0.0, 1.0, 1.0), 0.5)
        assert len(polys) == 4, f"expected 4 cells, got {len(polys)}"
        assert (rows, cols) == ([0, 0, 1, 1], [0, 1, 0, 1]), f"row/col order changed: {rows} {cols}"
        assert polys[0].bounds == (0.0, 0.0, 0.5, 0.5), f"first cell bounds wrong: {polys[0].bounds}"

    def test_create_points_parity(self):
        """create_points matches map(Point, …) for lists, generators and the empty case."""
        assert [p.wkt for p in create_points([(0, 0), (1, 1)])] == ["POINT (0 0)", "POINT (1 1)"]
        assert create_points((c for c in [(10.5, -3.25)]))[0].x == 10.5, "generator input broke"
        assert create_points([]) == [], "empty input should return []"


_WFS_CAPS = (
    b'<wfs:WFS_Capabilities version="2.0.0" xmlns:wfs="x">'
    b"<FeatureType><Name>topp:states</Name></FeatureType></wfs:WFS_Capabilities>"
)


class TestArc34WfsPreemptiveAuth:
    """ARC-34: WFS GetCapabilities must send Basic credentials preemptively, not reactively."""

    def test_get_capabilities_sends_preemptive_basic_and_real_ua(self, monkeypatch):
        """_get_capabilities builds an Authorization: Basic request with a non-urllib User-Agent.

        Test scenario:
            A server that 403s without a 401 challenge still receives credentials, and the default
            'Python-urllib' UA (which some servers block) is replaced.
        """
        captured: dict[str, dict[str, str]] = {}

        def fake_get(request, timeout):
            captured["headers"] = {k.lower(): v for k, v in request.header_items()}
            return _WFS_CAPS

        monkeypatch.setattr(_wfs, "http_get_with_retry", fake_get)
        _wfs._get_capabilities.cache_clear()
        versions, typenames = _wfs._get_capabilities("https://demo/wfs", None, ("ada", "s3cret"), 60.0)
        expected = "Basic " + base64.b64encode(b"ada:s3cret").decode()
        assert captured["headers"].get("authorization") == expected, "credentials not sent preemptively"
        assert "urllib" not in captured["headers"].get("user-agent", "").lower(), "default urllib UA not replaced"
        assert typenames == frozenset({"topp:states"})


class TestArc42ListLayersCache:
    """ARC-42: list_layers must not return a stale layer set after this class's own writes."""

    def test_to_file_invalidates_list_layers_cache(self, tmp_path):
        """Appending a layer via to_file invalidates the cached list_layers result.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            list_layers is LRU-cached; writing a second layer to the same GPKG must drop the cache so the
            follow-up list_layers sees both layers, not the stale single-layer set.
        """
        path = tmp_path / "layers.gpkg"
        fc = FeatureCollection(gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"))
        fc.to_file(path, driver="gpkg", layer="a", mode="w")
        assert FeatureCollection.list_layers(path) == ["a"], "first write should list one layer"
        fc.to_file(path, driver="gpkg", layer="b", mode="a")
        layers = FeatureCollection.list_layers(path)
        assert set(layers) == {"a", "b"}, f"stale list_layers cache after write: {layers}"
