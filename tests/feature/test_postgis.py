"""Tests for the PostGIS reader/writer (`pyramids.feature._postgis`).

Network-free and DB-free: the OGR PostgreSQL driver and `gpd.read_file` /
`FeatureCollection.to_file` are monkeypatched so `from_postgis` / `to_postgis`
logic — driver gating, connection normalisation, table/sql validation, filter
forwarding, write options, and error normalisation — is covered offline. A real
round-trip against a live PostGIS is a gated test (see the issue's DoD); it needs
a database, so it is not part of this unit suite.
"""

from __future__ import annotations

import os

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.feature import FeatureCollection
from pyramids.feature import _postgis
from pyramids.errors import PostGISError


def _sample_fc() -> FeatureCollection:
    """A tiny two-feature FeatureCollection for write tests."""
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"name": ["a", "b"]},
            geometry=[Point(5.0, 52.0), Point(6.0, 51.0)],
            crs="EPSG:4326",
        )
    )


def _driver_present(monkeypatch):
    monkeypatch.setattr(_postgis.ogr, "GetDriverByName", lambda name: object())


def _driver_absent(monkeypatch):
    monkeypatch.setattr(_postgis.ogr, "GetDriverByName", lambda name: None)


class TestPureHelpers:
    def test_pg_connection_prefixes(self):
        assert _postgis._pg_connection("host=db dbname=gis") == "PG:host=db dbname=gis"

    def test_pg_connection_passthrough(self):
        assert _postgis._pg_connection("PG:host=db") == "PG:host=db"

    def test_qualified_table(self):
        assert _postgis._qualified_table("parcels", "public") == "public.parcels"
        assert _postgis._qualified_table("parcels", None) == "parcels"

    def test_read_kwargs(self):
        kw = _postgis._read_kwargs((1.0, 2.0, 3.0, 4.0), "x>1", ["a", "b"], 5)
        assert kw == {"bbox": (1.0, 2.0, 3.0, 4.0), "where": "x>1", "columns": ["a", "b"], "rows": 5}
        assert _postgis._read_kwargs(None, None, None, None) == {}

    def test_read_kwargs_rejects_bad_bbox_length(self):
        with pytest.raises(ValueError, match="minx, miny, maxx, maxy"):
            _postgis._read_kwargs((1.0, 2.0, 3.0), None, None, None)

    def test_read_kwargs_rejects_inverted_bbox(self):
        with pytest.raises(ValueError, match="minx < maxx"):
            _postgis._read_kwargs((3.0, 2.0, 1.0, 4.0), None, None, None)

    def test_read_kwargs_rejects_negative_max_features(self):
        with pytest.raises(ValueError, match="max_features"):
            _postgis._read_kwargs(None, None, None, -1)

    def test_require_pg_driver_present(self, monkeypatch):
        _driver_present(monkeypatch)
        _postgis._require_pg_driver()  # does not raise

    def test_require_pg_driver_absent(self, monkeypatch):
        _driver_absent(monkeypatch)
        with pytest.raises(PostGISError, match="libgdal-pg"):
            _postgis._require_pg_driver()


class TestFromPostgis:
    def test_requires_exactly_one_of_table_or_sql(self):
        with pytest.raises(ValueError, match="exactly one"):
            FeatureCollection.from_postgis("PG:x")
        with pytest.raises(ValueError, match="exactly one"):
            FeatureCollection.from_postgis("PG:x", table="t", sql="select 1")

    def test_missing_driver_raises_postgiserror(self, monkeypatch):
        _driver_absent(monkeypatch)
        with pytest.raises(PostGISError, match="libgdal-pg"):
            FeatureCollection.from_postgis("PG:x", table="t")

    def test_reads_table_into_featurecollection(self, monkeypatch):
        _driver_present(monkeypatch)
        captured = {}

        def fake_read(conn, **kwargs):
            captured["conn"] = conn
            captured["kwargs"] = kwargs
            return gpd.GeoDataFrame({"n": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")

        monkeypatch.setattr(_postgis.gpd, "read_file", fake_read)
        fc = FeatureCollection.from_postgis(
            "host=db dbname=gis", table="public.parcels", bbox=(1.0, 2.0, 3.0, 4.0),
            where="n > 0", columns=["n"], max_features=10,
        )
        assert isinstance(fc, FeatureCollection)
        assert captured["conn"] == "PG:host=db dbname=gis"
        assert captured["kwargs"]["layer"] == "public.parcels"
        assert captured["kwargs"]["bbox"] == (1.0, 2.0, 3.0, 4.0)
        assert captured["kwargs"]["where"] == "n > 0"
        assert captured["kwargs"]["rows"] == 10

    def test_reads_sql(self, monkeypatch):
        _driver_present(monkeypatch)
        captured = {}

        def fake_read(conn, **kwargs):
            captured.update(kwargs)
            return gpd.GeoDataFrame({"n": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")

        monkeypatch.setattr(_postgis.gpd, "read_file", fake_read)
        FeatureCollection.from_postgis(
            "PG:x", sql="SELECT * FROM t WHERE n > 0", columns=["n"]
        )
        assert captured["sql"] == "SELECT * FROM t WHERE n > 0"
        assert "layer" not in captured
        assert "columns" not in captured  # mutually exclusive with sql in pyogrio

    def test_read_failure_raises_postgiserror(self, monkeypatch):
        _driver_present(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(_postgis.gpd, "read_file", boom)
        with pytest.raises(PostGISError, match="PostGIS read failed"):
            FeatureCollection.from_postgis("PG:x", table="t")


class TestToPostgis:
    def test_rejects_bad_if_exists(self):
        with pytest.raises(ValueError, match="if_exists"):
            _sample_fc().to_postgis("PG:x", table="t", if_exists="upsert")

    def test_missing_driver_raises_postgiserror(self, monkeypatch):
        _driver_absent(monkeypatch)
        with pytest.raises(PostGISError, match="libgdal-pg"):
            _sample_fc().to_postgis("PG:x", table="t")

    def test_writes_with_layer_and_options(self, monkeypatch):
        _driver_present(monkeypatch)
        fc = _sample_fc()
        captured = {}

        def fake_to_file(conn, **kwargs):
            captured["conn"] = conn
            captured["kwargs"] = kwargs

        monkeypatch.setattr(fc, "to_file", fake_to_file)
        fc.to_postgis(
            "host=db dbname=gis", table="parcels", schema="public",
            if_exists="append", geometry_column="the_geom", srid=4326,
        )
        kw = captured["kwargs"]
        assert captured["conn"] == "PG:host=db dbname=gis"
        assert kw["driver"] == "PostgreSQL"
        assert kw["layer"] == "public.parcels"
        assert kw["mode"] == "a"  # append -> mode "a" (to_file's contract)
        assert kw["GEOMETRY_NAME"] == "the_geom"
        assert kw["SRID"] == "4326"
        assert "append" not in kw and "layer_options" not in kw  # not to_file params

    def test_write_replace_uses_mode_w(self, monkeypatch):
        _driver_present(monkeypatch)
        fc = _sample_fc()
        captured = {}
        monkeypatch.setattr(fc, "to_file", lambda conn, **kw: captured.update(kw))
        fc.to_postgis("PG:x", table="t", if_exists="replace")
        assert captured["mode"] == "w"

    def test_write_failure_raises_postgiserror(self, monkeypatch):
        _driver_present(monkeypatch)
        fc = _sample_fc()

        def boom(*a, **k):
            raise RuntimeError("permission denied")

        monkeypatch.setattr(fc, "to_file", boom)
        with pytest.raises(PostGISError, match="PostGIS write failed"):
            fc.to_postgis("PG:x", table="t")


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("PYRAMIDS_POSTGIS_DSN"),
    reason="live PostGIS round-trip; set PYRAMIDS_POSTGIS_DSN to a writable PG: "
    "connection (and a build with the OGR PostgreSQL driver) to run",
)
class TestLivePostGISRoundTrip:
    """Round-trip a FeatureCollection through a real PostGIS database."""

    def test_write_then_read(self):
        """Writing then reading a table preserves geometry, attributes, and CRS."""
        dsn = os.environ["PYRAMIDS_POSTGIS_DSN"]
        table = os.environ.get("PYRAMIDS_POSTGIS_TABLE", "pyramids_roundtrip_test")
        fc = _sample_fc()
        fc.to_postgis(dsn, table=table, if_exists="replace", srid=4326)
        back = FeatureCollection.from_postgis(dsn, table=table)
        assert len(back) == len(fc)
        assert set(back["name"]) == {"a", "b"}
        assert back.crs.to_epsg() == 4326
