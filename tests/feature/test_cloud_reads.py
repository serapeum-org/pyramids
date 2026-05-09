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
* Real cloud-service tests (``s3://`` etc.) are marked ``vfs`` and
  skipped by default — they run only when real credentials are in
  the environment. They document the intended surface.

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
        assert any(
            "rewritten" in m and "/vsicurl/" in m for m in messages
        ), f"expected a /vsicurl/ rewrite log; got: {messages}"


class TestFileUrlRead:
    """``file://`` URLs are rewritten to plain local paths (no network)."""

    def test_read_file_url(self, tmp_path: Path):
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        p = tmp_path / "one.geojson"
        gdf.to_file(p, driver="GeoJSON")

        fc = FeatureCollection.read_file(p.resolve().as_uri())
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 1


@pytest.mark.vfs
class TestCloudReads:
    """Cloud-service reads (s3:// etc.) — gated behind the ``vfs`` marker.

    Skipped by default. Run with ``pytest -m vfs`` when you have real
    AWS / GS / Azure credentials available in the environment. Each
    test opens a well-known public or user-controlled object; replace
    the URLs as needed for your setup.
    """

    def test_s3_read_requires_credentials(self):
        """Placeholder: s3:// URL round-trip.

        Fill in with a bucket/object you control. The default is
        skipped unless the AWS_* env vars are set AND the ``vfs``
        marker is selected via ``-m vfs``. This test documents the
        intended public surface; it is not expected to run in CI.
        """
        import os

        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            pytest.skip("AWS credentials not in env; skipping s3:// test")
        pytest.skip("no concrete s3:// fixture; this is a documentation stub")
