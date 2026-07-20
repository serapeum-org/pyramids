"""ARC-22: cloud / virtual-filesystem reads in FeatureCollection.read_file.

After ARC-23 wired ``pyramids._io._parse_path`` into
``FeatureCollection.read_file``, URL-scheme paths (``s3://``, ``gs://``,
``az://``, ``http(s)://``, ``file://``) are rewritten to GDAL
``/vsi*`` form before the file is opened. These tests cover that
behavior without any real network I/O:

* The ``http://`` rewrite is tested by mocking ``geopandas.read_file``
  — we assert that the mock receives the rewritten ``/vsicurl/...``
  path. This is the ARC-22 behavior; actually fetching from HTTP is
  GDAL's job and doesn't need to be re-tested here.
* ``file://`` paths are exercised against a real tmp_path file — the
  rewrite is a no-op string operation, no network.
* ``s3://`` is the same string rewrite, so it is tested the same way —
  by mocking ``geopandas.read_file`` and asserting the ``/vsis3/...``
  path. No network, no credentials, no skip.

Why mock instead of a local HTTP server: GDAL ``/vsicurl/`` on Windows
loopback can hang indefinitely against ``http.server``'s default
HTTP/1.0 handler (no keep-alive, no client-side read timeout). The
behavior under test is string rewriting, not curl semantics — so we
mock at the read boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


class TestHttpRewrite:
    """Assert that ``http://`` URLs reach ``gpd.read_file`` as ``/vsicurl/...``.

    Mocks ``geopandas.read_file`` so no network traffic is issued. The
    point of the test is the rewrite, not GDAL's curl behavior.
    """

    def _fake_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"id": [1, 2, 3], "name": ["a", "b", "c"]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326",
        )

    def test_http_url_is_rewritten_to_vsicurl(self, monkeypatch):
        """Given an ``http://`` URL, ``gpd.read_file`` receives ``/vsicurl/...``."""
        captured: dict[str, object] = {}

        def fake_read_file(path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return self._fake_gdf()

        monkeypatch.setattr("pyramids.feature.collection.gpd.read_file", fake_read_file)

        url = "http://example.invalid/points.geojson"
        fc = FeatureCollection.read_file(url)

        assert captured["path"] == f"/vsicurl/{url}"
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 3

    def test_https_url_is_rewritten_to_vsicurl(self, monkeypatch):
        """``https://`` also maps to ``/vsicurl/``."""
        captured: dict[str, object] = {}

        def fake_read_file(path, **kwargs):
            captured["path"] = path
            return self._fake_gdf()

        monkeypatch.setattr("pyramids.feature.collection.gpd.read_file", fake_read_file)

        url = "https://example.invalid/points.geojson"
        FeatureCollection.read_file(url)

        assert captured["path"] == f"/vsicurl/{url}"

    def test_rewrite_emits_log_message(self, monkeypatch, caplog):
        """The ``pyramids.base.remote`` rewrite log fires on the code path."""

        def fake_read_file(path, **kwargs):
            return self._fake_gdf()

        monkeypatch.setattr("pyramids.feature.collection.gpd.read_file", fake_read_file)

        url = "http://example.invalid/points.geojson"
        with caplog.at_level(logging.DEBUG, logger="pyramids.base.remote"):
            FeatureCollection.read_file(url)

        messages = [rec.getMessage() for rec in caplog.records]
        assert any("rewritten" in m and "/vsicurl/" in m for m in messages), (
            f"expected a /vsicurl/ rewrite log; got: {messages}"
        )


class TestFileUrlRead:
    """``file://`` URLs are rewritten to plain local paths (no network)."""

    def test_read_file_url(self, tmp_path: Path):
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        p = tmp_path / "one.geojson"
        gdf.to_file(p, driver="GeoJSON")

        fc = FeatureCollection.read_file(p.resolve().as_uri())
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 1


class TestS3Rewrite:
    """``s3://`` URLs reach ``gpd.read_file`` as ``/vsis3/...``.

    Mirrors :class:`TestHttpRewrite`: the s3 surface is the same string rewrite,
    so it is tested by mocking ``geopandas.read_file`` (no network, no
    credentials) rather than by an always-skipping placeholder.
    """

    def _fake_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326",
        )

    def test_s3_url_is_rewritten_to_vsis3(self, monkeypatch):
        """An ``s3://`` URL reaches ``gpd.read_file`` as ``/vsis3/...``."""
        captured: dict[str, object] = {}

        def fake_read_file(path, **kwargs):
            captured["path"] = path
            return self._fake_gdf()

        monkeypatch.setattr("pyramids.feature.collection.gpd.read_file", fake_read_file)

        fc = FeatureCollection.read_file("s3://my-bucket/path/points.geojson")

        assert captured["path"] == "/vsis3/my-bucket/path/points.geojson"
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 3
