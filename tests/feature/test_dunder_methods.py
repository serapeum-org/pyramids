"""Dunder behavior tests for FeatureCollection (GeoDataFrame-subclass design).

Under the ARC-1a refactor, ``FeatureCollection`` subclasses
``geopandas.GeoDataFrame`` and inherits the GeoDataFrame / pandas
semantics for ``__len__``, ``__iter__``, ``__getitem__``,
``__contains__``, ``__eq__``. The previous pyramids-specific
``(index, row)`` iteration and bool-casting behaviors were dropped
intentionally so the class behaves as a GeoDataFrame everywhere a
GeoDataFrame is expected.

Retained pyramids-specific dunders: ``__str__`` and ``__repr__`` —
they produce the ``FeatureCollection(...)`` branding.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from geopandas.geodataframe import GeoDataFrame
from shapely.geometry import Point, Polygon

from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture
def point_fc():
    """FeatureCollection with 3 Point features."""
    gdf = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "value": [10, 20, 30]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


@pytest.fixture
def polygon_fc():
    """FeatureCollection with 2 Polygon features."""
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2], "area_m2": [100.0, 200.0]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


@pytest.fixture
def empty_fc():
    """Empty FeatureCollection."""
    gdf = gpd.GeoDataFrame({"name": [], "value": []}, geometry=[])
    return FeatureCollection(gdf)


@pytest.fixture
def single_fc():
    """FeatureCollection with exactly 1 Point feature."""
    gdf = gpd.GeoDataFrame({"label": ["only"]}, geometry=[Point(5, 5)], crs="EPSG:4326")
    return FeatureCollection(gdf)


class TestLen:
    def test_len_multiple_features(self, point_fc):
        assert len(point_fc) == 3

    def test_len_empty(self, empty_fc):
        assert len(empty_fc) == 0

    def test_len_single(self, single_fc):
        assert len(single_fc) == 1


class TestContains:
    def test_contains_existing_column(self, point_fc):
        assert "name" in point_fc

    def test_contains_missing_column(self, point_fc):
        assert "missing_column" not in point_fc

    def test_contains_geometry(self, point_fc):
        assert "geometry" in point_fc


class TestGetitem:
    def test_getitem_string_column(self, point_fc):
        names = point_fc["name"]
        assert list(names) == ["A", "B", "C"]

    def test_getitem_list_of_columns(self, point_fc):
        subset = point_fc[["name", "value"]]
        # The column subset is a FeatureCollection by virtue of _constructor,
        # but it loses the geometry column and therefore is not
        # a valid GeoDataFrame anymore — that's standard pandas behavior.
        assert "name" in subset.columns and "value" in subset.columns

    def test_boolean_mask(self, point_fc):
        subset = point_fc[point_fc["value"] > 10]
        assert len(subset) == 2
        assert isinstance(subset, FeatureCollection)


class TestSlice:
    def test_slice_rows(self, point_fc):
        subset = point_fc[0:2]
        assert len(subset) == 2
        assert isinstance(subset, FeatureCollection)


class TestStr:
    def test_str_contains_feature_count(self, point_fc):
        s = str(point_fc)
        assert "3 features" in s

    def test_str_contains_columns(self, point_fc):
        s = str(point_fc)
        assert "name" in s

    def test_str_contains_epsg(self, point_fc):
        s = str(point_fc)
        assert "4326" in s

    def test_str_empty(self, empty_fc):
        s = str(empty_fc)
        # Empty collection still produces a valid string
        assert "FeatureCollection" in s


class TestRepr:
    def test_repr_format(self, point_fc):
        r = repr(point_fc)
        assert "n_features=3" in r

    def test_repr_includes_columns(self, point_fc):
        r = repr(point_fc)
        assert "name" in r


class TestIterrows:
    def test_iterrows_yields_three_rows(self, point_fc):
        rows = list(point_fc.iterrows())
        assert len(rows) == 3

    def test_iterrows_tuple_structure(self, point_fc):
        for idx, row in point_fc.iterrows():
            assert "name" in row.index
            assert hasattr(row, "geometry")
            break


class TestEndToEnd:
    def test_create_check_contains(self):
        gdf = gpd.GeoDataFrame(
            {"country": ["A", "B"], "pop": [100, 200]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)
        assert len(fc) == 2
        assert "country" in fc
        assert fc.epsg == 4326

    def test_inherited_to_crs_keeps_subclass(self, point_fc):
        out = point_fc.to_crs(3857)
        assert isinstance(out, FeatureCollection)
        assert out.crs.to_epsg() == 3857
