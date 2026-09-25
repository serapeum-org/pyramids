"""Issue #1200: count / inspect a vector file without loading its geometry.

``FeatureCollection.feature_count(path)`` returns the layer feature count
straight from the driver header, and ``FeatureCollection.feature_info(path)``
returns a :class:`VectorInfo` with the count, geometry type, CRS, extent,
fields, layer name and driver — both via :func:`pyogrio.read_info`, so no
feature rows are materialised.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import geopandas as gpd
import pyogrio
import pytest
from pyproj.exceptions import CRSError
from shapely.geometry import Point, box

from pyramids.feature import FeatureCollection, VectorInfo, _read

pytestmark = pytest.mark.core

BASIN = Path("tests/data/basin.geojson")


@pytest.fixture
def points_geojson(tmp_path: Path) -> Path:
    p = tmp_path / "pts.geojson"
    gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    ).to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def two_layer_gpkg(tmp_path: Path) -> Path:
    rivers = gpd.GeoDataFrame(
        {"name": ["r1", "r2"]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    lakes = gpd.GeoDataFrame(
        {"name": ["l1", "l2", "l3"]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)],
        crs="EPSG:4326",
    )
    p = tmp_path / "multi.gpkg"
    rivers.to_file(p, driver="GPKG", layer="rivers")
    lakes.to_file(p, driver="GPKG", layer="lakes")
    return p


@pytest.fixture
def csv_with_wkt(tmp_path: Path) -> Path:
    """A CSV whose extent OGR only computes on a forced scan (no cached header)."""
    p = tmp_path / "pts.csv"
    p.write_text('id,WKT\n1,"POINT (0 0)"\n2,"POINT (2 2)"\n')
    return p


class TestFeatureCount:
    """``FeatureCollection.feature_count(path)`` returns the layer count."""

    def test_counts_geojson(self, points_geojson: Path):
        assert FeatureCollection.feature_count(points_geojson) == 3

    def test_matches_full_read(self):
        """The header count equals the row count of a full read."""
        count = FeatureCollection.feature_count(BASIN)
        loaded = len(FeatureCollection.read_file(BASIN))
        assert count == loaded

    def test_accepts_str_path(self, points_geojson: Path):
        assert FeatureCollection.feature_count(str(points_geojson)) == 3

    def test_does_not_load_geometry(self, points_geojson: Path, monkeypatch):
        """The count must come from ``read_info``, never a full ``read_dataframe``.

        Guards against a regression to ``len(read_file(path))``, which would
        materialise every feature.
        """
        called = False

        def _boom(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("feature_count triggered a full read")

        monkeypatch.setattr(pyogrio, "read_dataframe", _boom)
        assert FeatureCollection.feature_count(points_geojson) == 3
        assert not called

    def test_layer_selects_the_right_count(self, two_layer_gpkg: Path):
        assert FeatureCollection.feature_count(two_layer_gpkg, layer="rivers") == 2
        assert FeatureCollection.feature_count(two_layer_gpkg, layer="lakes") == 3

    def test_missing_path_raises_file_not_found(self, tmp_path: Path):
        missing = tmp_path / "nope.geojson"
        with pytest.raises(FileNotFoundError, match=str(missing.name)):
            FeatureCollection.feature_count(missing)

    def test_remote_path_skips_local_existence_check(self, monkeypatch):
        """A remote URL must not be probed for local existence.

        ``is_remote`` short-circuits the ``Path.exists`` check, so a cloud URL
        is handed straight to the driver rather than raising
        ``FileNotFoundError``.
        """
        captured = {}

        def _fake_read_info(resolved, layer=None, **kwargs):
            captured["resolved"] = resolved
            return {"features": 7}

        monkeypatch.setattr(_read, "is_remote", lambda _p: True)
        monkeypatch.setattr(_read.pyogrio, "read_info", _fake_read_info)
        count = FeatureCollection.feature_count("s3://bucket/missing.geojson")
        assert count == 7
        assert "resolved" in captured, "read_info was never reached"


class TestFeatureInfo:
    """``FeatureCollection.feature_info(path)`` returns a ``VectorInfo``."""

    def test_returns_vector_info(self, points_geojson: Path):
        info = FeatureCollection.feature_info(points_geojson)
        assert isinstance(info, VectorInfo)

    def test_fields_are_populated(self, points_geojson: Path):
        info = FeatureCollection.feature_info(points_geojson)
        assert info.feature_count == 3
        assert info.geometry_type == "Point"
        assert info.crs_epsg == 4326
        assert info.fields == ["id"]
        assert info.driver == "GeoJSON"

    def test_bounds_match_geometry(self, points_geojson: Path):
        info = FeatureCollection.feature_info(points_geojson)
        assert info.bounds == (0.0, 0.0, 2.0, 2.0)

    def test_count_matches_feature_count(self):
        info = FeatureCollection.feature_info(BASIN)
        assert info.feature_count == FeatureCollection.feature_count(BASIN)

    def test_extent_computed_for_extentless_driver(self, csv_with_wkt: Path):
        """A driver that caches no extent (CSV) still yields real bounds.

        ``feature_info`` forces the extent, so the metadata is complete
        regardless of whether the driver caches a bounding box.
        """
        info = FeatureCollection.feature_info(csv_with_wkt)
        assert info.bounds == (0.0, 0.0, 2.0, 2.0)
        assert info.feature_count == 2

    def test_count_forced_for_extentless_driver(self, csv_with_wkt: Path):
        """``feature_count`` returns a real count (never OGR's -1) for a CSV."""
        assert FeatureCollection.feature_count(csv_with_wkt) == 2

    def test_layer_selects_the_right_layer(self, two_layer_gpkg: Path):
        rivers = FeatureCollection.feature_info(two_layer_gpkg, layer="rivers")
        lakes = FeatureCollection.feature_info(two_layer_gpkg, layer="lakes")
        assert rivers.feature_count == 2
        assert rivers.layer == "rivers"
        assert lakes.feature_count == 3
        assert lakes.layer == "lakes"

    def test_crs_epsg_none_when_undefined(self, tmp_path: Path):
        """A CRS-less file yields ``crs_epsg=None`` rather than raising.

        A shapefile with no ``.prj`` sidecar carries no CRS; GeoJSON cannot
        be used here because the format mandates WGS84 (EPSG:4326).
        """
        p = tmp_path / "crsless.shp"
        gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs=None).to_file(p)
        info = FeatureCollection.feature_info(p)
        assert info.crs_epsg is None

    def test_crs_epsg_none_when_unresolvable(self, points_geojson: Path, monkeypatch):
        """An unparseable CRS yields ``crs_epsg=None`` instead of propagating.

        When ``pyproj`` cannot interpret the driver's CRS string it raises
        ``CRSError``; ``feature_info`` swallows it and reports ``None``.
        """

        def _raise(*_args, **_kwargs):
            raise CRSError("unparseable")

        monkeypatch.setattr(_read.pyproj.CRS, "from_user_input", _raise)
        info = FeatureCollection.feature_info(points_geojson)
        assert info.crs_epsg is None

    def test_bounds_none_when_driver_reports_none(
        self, points_geojson: Path, monkeypatch
    ):
        """A driver that reports no extent yields ``bounds=None``.

        Empty layers report ``total_bounds=None``; ``feature_info`` must pass
        that through rather than crashing on the unpack.
        """
        real_read_info = _read.pyogrio.read_info

        def _no_bounds(*args, **kwargs):
            raw = dict(real_read_info(*args, **kwargs))
            raw["total_bounds"] = None
            return raw

        monkeypatch.setattr(_read.pyogrio, "read_info", _no_bounds)
        info = FeatureCollection.feature_info(points_geojson)
        assert info.bounds is None
        assert info.feature_count == 3

    def test_does_not_load_geometry(self, points_geojson: Path, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("feature_info triggered a full read")

        monkeypatch.setattr(pyogrio, "read_dataframe", _boom)
        info = FeatureCollection.feature_info(points_geojson)
        assert info.feature_count == 3

    def test_missing_path_raises_file_not_found(self, tmp_path: Path):
        missing = tmp_path / "nope.geojson"
        with pytest.raises(FileNotFoundError, match=str(missing.name)):
            FeatureCollection.feature_info(missing)


class TestVectorInfo:
    """The ``VectorInfo`` value object itself."""

    def test_is_frozen(self, points_geojson: Path):
        info = FeatureCollection.feature_info(points_geojson)
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.feature_count = 99  # type: ignore[misc]

    def test_exported_from_feature_package(self):
        from pyramids import feature

        assert feature.VectorInfo is VectorInfo
