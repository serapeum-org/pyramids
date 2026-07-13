"""Tests for the OGC API – Features reader (`pyramids.feature._oapif`).

Network-free. The OGR ``OAPIF`` driver's ``next``-link paging is proven against a
protocol-faithful local mock in ``TestOapifDriverPaging``. Elsewhere the OGR read
is monkeypatched so ``from_ogc_features``'s own logic — collection validation,
the read filters, FeatureCollection wrapping, ``output_crs`` reproject and error
normalisation — is covered without a live service, plus the pure helpers and the
``/collections`` fetch/parse/cache.
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import socketserver
import threading
from collections import Counter

import geopandas as gpd
import pyogrio
import pytest
from osgeo import gdal
from shapely.geometry import Point

from pyramids.base import _ogc_api
from pyramids.feature import FeatureCollection
from pyramids.feature import _oapif
from pyramids.errors import OGCAPIError
from tests.http_mock import make_fixed_body_server

COLLECTIONS_DOC = json.dumps(
    {
        "links": [{"rel": "self", "href": "https://x/collections"}],
        "collections": [
            {"id": "lakes", "title": "Lakes"},
            {"id": "roads", "title": "Roads"},
        ],
    }
)

ERROR_DOC = json.dumps({"code": "NoApplicableCode", "description": "OGC API service error."})


@pytest.fixture(autouse=True)
def _clear_collections_cache():
    """Isolate the module-level collections LRU cache between tests."""
    _oapif._get_collections.cache_clear()
    yield
    _oapif._get_collections.cache_clear()


def _make_server(body: str, content_type: str = "application/json"):
    """Local HTTP server returning `body` for every GET; returns (url, counter, httpd)."""
    return make_fixed_body_server(body, content_type)


@pytest.fixture
def collections_server():
    """A mock service serving the /collections document."""
    url, counter, httpd = _make_server(COLLECTIONS_DOC)
    yield url, counter
    httpd.shutdown()
    httpd.server_close()


def _sample_gdf(crs="EPSG:4326") -> gpd.GeoDataFrame:
    """A tiny two-feature GeoDataFrame standing in for an /items response."""
    return gpd.GeoDataFrame(
        {"name": ["a", "b"], "scalerank": [1, 4]},
        geometry=[Point(5.0, 52.0), Point(6.0, 51.0)],
        crs=crs,
    )


class TestPureHelpers:
    def test_collections_url(self):
        assert _ogc_api.collections_url("https://h/api") == "https://h/api/collections?f=json"
        assert _ogc_api.collections_url("https://h/api/") == "https://h/api/collections?f=json"

    def test_collections_url_preserves_query_auth(self):
        """A query-string-auth endpoint keeps its query; /collections goes before it."""
        assert (
            _ogc_api.collections_url("https://h/ogc?api_key=XYZ")
            == "https://h/ogc/collections?api_key=XYZ&f=json"
        )

    def test_oapif_connection(self):
        assert _oapif._oapif_connection("https://h/api") == "OAPIF:https://h/api"

    def test_gdal_http_config(self):
        assert _oapif._gdal_http_config(None, 60.0) == {"GDAL_HTTP_TIMEOUT": "60"}
        cfg = _oapif._gdal_http_config(("u", "p"), 30.0)
        assert cfg["GDAL_HTTP_USERPWD"] == "u:p" and cfg["GDAL_HTTP_TIMEOUT"] == "30"

    def test_gdal_http_config_clamps_subsecond_timeout(self):
        assert _oapif._gdal_http_config(None, 0.5)["GDAL_HTTP_TIMEOUT"] == "1"

    def test_read_kwargs(self):
        assert _oapif._read_kwargs(None, None, None) == {}
        kw = _oapif._read_kwargs((1.0, 2.0, 3.0, 4.0), "x>1", 10)
        assert kw == {"bbox": (1.0, 2.0, 3.0, 4.0), "where": "x>1", "rows": 10}

    def test_read_kwargs_rejects_negative_max_features(self):
        with pytest.raises(ValueError, match="max_features"):
            _oapif._read_kwargs(None, None, -1)

    def test_read_kwargs_rejects_zero_max_features(self):
        """0 is rejected: pyogrio reads rows=0 as 'no limit', so a 0 cap would fetch everything."""
        with pytest.raises(ValueError, match="max_features must be >= 1"):
            _oapif._read_kwargs(None, None, 0)

    def test_read_kwargs_rejects_bad_bbox_length(self):
        with pytest.raises(ValueError, match="minx, miny, maxx, maxy"):
            _oapif._read_kwargs((1.0, 2.0, 3.0), None, None)

    def test_read_kwargs_rejects_inverted_bbox(self):
        with pytest.raises(ValueError, match="minx < maxx"):
            _oapif._read_kwargs((3.0, 2.0, 1.0, 4.0), None, None)

    def test_collection_ids_prefers_id_then_name(self):
        doc = {"collections": [{"id": "a"}, {"name": "b"}, {"title": "no-id"}, "junk"]}
        assert _ogc_api.collection_ids(doc) == {"a", "b"}

    def test_error_text(self):
        assert _ogc_api.error_text({"description": "boom"}) == "boom"
        assert _ogc_api.error_text({"title": "bad"}) == "bad"  # description/detail absent -> title
        assert _ogc_api.error_text({}) == "no message provided"
        assert _ogc_api.error_text("not a dict") == "no message provided"

    def test_http_error_detail_unreadable_body(self):
        """An HTTPError whose body cannot be read falls back to the reason phrase."""
        class _Unreadable:
            reason = "Server Error"

            def read(self):
                raise OSError("connection gone")

        assert _ogc_api.http_error_detail(_Unreadable()) == "Server Error"

    def test_http_error_detail_non_json_body(self):
        """A non-JSON HTTPError body is returned as truncated plain text."""
        class _Plain:
            reason = "Bad Gateway"

            def read(self):
                return b"upstream exploded"

        assert _ogc_api.http_error_detail(_Plain()) == "upstream exploded"

    def test_read_http_error_returns_code_and_body(self):
        """read_http_error returns the status code and the decoded, stripped body."""
        class _Err:
            code = 422
            reason = "Unprocessable Entity"

            def read(self):
                return b'  {"message": "nope"}  '

        assert _ogc_api.read_http_error(_Err()) == (422, '{"message": "nope"}')

    def test_read_http_error_falls_back_to_reason(self):
        """An empty or unreadable body falls back to the reason phrase."""
        class _Empty:
            code = 500
            reason = "Server Error"

            def read(self):
                return b""

        class _Unreadable:
            code = 503
            reason = "Service Unavailable"

            def read(self):
                raise OSError("connection gone")

        assert _ogc_api.read_http_error(_Empty()) == (500, "Server Error")
        assert _ogc_api.read_http_error(_Unreadable()) == (503, "Service Unavailable")


class TestCollections:
    def test_parses_collection_ids(self, collections_server):
        url, _ = collections_server
        ids = _oapif._get_collections(url, None, 30.0)
        assert ids == {"lakes", "roads"}

    def test_lru_cache_one_fetch_per_endpoint(self, collections_server):
        url, counter = collections_server
        _oapif._get_collections(url, None, 30.0)
        _oapif._get_collections(url, None, 30.0)
        assert counter["GET"] == 1

    @staticmethod
    def _challenging_server():
        """Start a server that 401-challenges until correct Basic credentials arrive."""
        payload = COLLECTIONS_DOC.encode("utf-8")
        expected = "Basic " + base64.b64encode(b"user:secret").decode()
        attempts: Counter[str] = Counter()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                attempts["GET"] += 1
                if self.headers.get("Authorization") != expected:
                    self.send_response(401)
                    self.send_header("WWW-Authenticate", 'Basic realm="oapif"')
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args, **kwargs):  # noqa: N802
                return

        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd, f"http://127.0.0.1:{httpd.server_address[1]}", attempts

    def test_auth_sends_preemptive_basic_credentials(self):
        """Given credentials, the reader sends them preemptively and reads in one request."""
        httpd, url, attempts = self._challenging_server()
        try:
            ids = _oapif._get_collections(url, ("user", "secret"), 30.0)
            assert ids == {"lakes", "roads"}
            assert attempts["GET"] == 1  # preemptive: succeeds without a 401 round-trip
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_missing_credentials_against_challenge_raises(self):
        """Without credentials the 401-challenging service yields an OGCAPIError."""
        httpd, url, _ = self._challenging_server()
        try:
            with pytest.raises(OGCAPIError, match="request failed"):
                _oapif._get_collections(url, None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_error_document_raises_ogcapierror(self):
        url, _, httpd = _make_server(ERROR_DOC)
        try:
            with pytest.raises(OGCAPIError, match="OGC API service error"):
                _oapif._get_collections(url, None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_non_json_body_raises_ogcapierror(self):
        url, _, httpd = _make_server("not json", content_type="text/plain")
        try:
            with pytest.raises(OGCAPIError, match="non-JSON"):
                _oapif._get_collections(url, None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_non_object_json_body_raises_ogcapierror(self):
        """A valid JSON array (not an object with `collections`) is rejected."""
        url, _, httpd = _make_server("[]")
        try:
            with pytest.raises(OGCAPIError, match="no collections"):
                _oapif._get_collections(url, None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_http_error_problem_document_message_extracted(self):
        """A 4xx/5xx problem document's description is surfaced, not just the status."""
        body = json.dumps({"description": "collection backend down"}).encode("utf-8")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(503)
                self.send_header("Content-Type", "application/problem+json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args, **kwargs):  # noqa: N802
                return

        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            with pytest.raises(OGCAPIError, match="collection backend down"):
                _oapif._get_collections(f"http://127.0.0.1:{httpd.server_address[1]}", None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_transport_failure_raises_ogcapierror(self, monkeypatch):
        def boom(self, *args, **kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr(_ogc_api.urllib.request.OpenerDirector, "open", boom)
        with pytest.raises(OGCAPIError, match="request failed"):
            _oapif._get_collections("https://oapif.invalid/x", None, 5.0)

    def test_discovery_sends_json_accept_and_useragent(self):
        """The /collections pre-check negotiates JSON and sends a real User-Agent."""
        seen = {}
        payload = COLLECTIONS_DOC.encode("utf-8")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                seen["accept"] = self.headers.get("Accept")
                seen["user_agent"] = self.headers.get("User-Agent")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args, **kwargs):  # noqa: N802
                return

        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            _oapif._get_collections(f"http://127.0.0.1:{httpd.server_address[1]}", None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()
        assert seen["accept"] == "application/json"
        assert seen["user_agent"] and "Python-urllib" not in seen["user_agent"]


class TestFromOgcApiFeatures:
    def _patch_collections(self, monkeypatch, ids=("lakes",)):
        monkeypatch.setattr(_oapif, "_get_collections", lambda *a, **k: frozenset(ids))

    def test_returns_featurecollection(self, monkeypatch):
        """A successful read is wrapped into a FeatureCollection."""
        self._patch_collections(monkeypatch)
        monkeypatch.setattr(_oapif.gpd, "read_file", lambda *a, **k: _sample_gdf())
        fc = FeatureCollection.from_ogc_features("https://h/api", collection="lakes")
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 2 and fc.crs.to_epsg() == 4326

    def test_passes_filters_to_read(self, monkeypatch):
        """bbox / where / max_features are forwarded to the OGR read as filters."""
        self._patch_collections(monkeypatch)
        captured = {}

        def fake_read(connection, **kwargs):
            captured["connection"] = connection
            captured["kwargs"] = kwargs
            return _sample_gdf()

        monkeypatch.setattr(_oapif.gpd, "read_file", fake_read)
        FeatureCollection.from_ogc_features(
            "https://h/api", collection="lakes", bbox=(1.0, 2.0, 3.0, 4.0),
            where="scalerank <= 2", max_features=5,
        )
        assert captured["connection"] == "OAPIF:https://h/api"
        assert captured["kwargs"]["layer"] == "lakes"
        assert captured["kwargs"]["bbox"] == (1.0, 2.0, 3.0, 4.0)
        assert captured["kwargs"]["where"] == "scalerank <= 2"
        assert captured["kwargs"]["rows"] == 5

    def test_output_crs_reprojects(self, monkeypatch):
        self._patch_collections(monkeypatch)
        monkeypatch.setattr(_oapif.gpd, "read_file", lambda *a, **k: _sample_gdf())
        fc = FeatureCollection.from_ogc_features(
            "https://h/api", collection="lakes", output_crs="EPSG:3857"
        )
        assert fc.crs.to_epsg() == 3857

    def test_output_crs_without_result_crs_raises(self, monkeypatch):
        """output_crs on a CRS-less result raises OGCAPIError instead of silently dropping it."""
        self._patch_collections(monkeypatch)
        crsless = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Point(5.0, 52.0)])
        monkeypatch.setattr(_oapif.gpd, "read_file", lambda *a, **k: crsless)
        with pytest.raises(OGCAPIError, match="without a CRS"):
            FeatureCollection.from_ogc_features(
                "https://h/api", collection="lakes", output_crs="EPSG:3857"
            )

    def test_unknown_collection_raises_valueerror(self, monkeypatch):
        self._patch_collections(monkeypatch, ids=("lakes",))
        with pytest.raises(ValueError, match="not advertised"):
            FeatureCollection.from_ogc_features("https://h/api", collection="missing")

    def test_read_failure_raises_ogcapierror(self, monkeypatch):
        self._patch_collections(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("driver said no")

        monkeypatch.setattr(_oapif.gpd, "read_file", boom)
        with pytest.raises(OGCAPIError, match="items request failed"):
            FeatureCollection.from_ogc_features("https://h/api", collection="lakes")

    def test_auth_and_timeout_active_during_read(self, monkeypatch):
        """The items read runs inside a GDAL config context carrying auth + timeout."""
        self._patch_collections(monkeypatch)
        seen = {}

        def fake_read(connection, **kwargs):
            seen["userpwd"] = _oapif.gdal.GetConfigOption("GDAL_HTTP_USERPWD")
            seen["timeout"] = _oapif.gdal.GetConfigOption("GDAL_HTTP_TIMEOUT")
            return _sample_gdf()

        monkeypatch.setattr(_oapif.gpd, "read_file", fake_read)
        FeatureCollection.from_ogc_features(
            "https://h/api", collection="lakes", auth=("u", "p"), timeout=42.0
        )
        assert seen["userpwd"] == "u:p"
        assert seen["timeout"] == "42"


class TestReadKwargsContract:
    """The filter kwargs `_read_kwargs` builds must be accepted by the *real* `gpd.read_file`.

    The `TestFromOgcApiFeatures` cases stub `read_file`, so they cannot catch a
    kwarg the installed reader rejects — the blind spot that let the PostGIS PR
    ship a broken write path. These feed the exact `_read_kwargs` output to the
    unmocked reader on a local file. (`layer=` acceptance is covered separately by
    `test_read_file_filters.py`.)
    """

    @pytest.fixture
    def sample_file(self, tmp_path):
        """A two-feature GeoJSON written by the OGR writer, readable by the reader."""
        path = tmp_path / "sample.geojson"
        gpd.GeoDataFrame(
            {"name": ["a", "b"], "v": [1, 9]},
            geometry=[Point(5.0, 52.0), Point(6.0, 51.0)],
            crs="EPSG:4326",
        ).to_file(path, driver="GeoJSON")
        return path

    def test_rows_kwarg_caps_features(self, sample_file):
        """`rows` (from max_features) is a valid read kwarg and caps the result."""
        gdf = gpd.read_file(sample_file, **_oapif._read_kwargs(None, None, 1))
        assert len(gdf) == 1

    def test_where_kwarg_filters(self, sample_file):
        """`where` is a valid read kwarg and pushes the attribute filter down."""
        gdf = gpd.read_file(sample_file, **_oapif._read_kwargs(None, "v > 5", None))
        assert list(gdf["name"]) == ["b"]

    def test_bbox_kwarg_filters(self, sample_file):
        """`bbox` is a valid read kwarg and restricts to intersecting features."""
        gdf = gpd.read_file(sample_file, **_oapif._read_kwargs((0.0, 0.0, 5.5, 53.0), None, None))
        assert list(gdf["name"]) == ["a"]

    def test_oapif_driver_is_available_to_the_reader(self):
        """The OAPIF driver is registered in the installed pyogrio reader.

        The wrapper reads via `gpd.read_file("OAPIF:…")` (pyogrio); this guards against a
        pyogrio/GDAL build lacking the driver, which would break the feature wholesale — the
        one wrapper-read property that *can* be checked offline (pyogrio cannot reach a
        localhost mock here, so the full read is covered by the gated live test).
        """
        assert "OAPIF" in pyogrio.list_drivers()


def _feature(fid: str, x: float, y: float, name: str) -> dict:
    """A single GeoJSON point feature."""
    return {
        "type": "Feature",
        "id": fid,
        "geometry": {"type": "Point", "coordinates": [x, y]},
        "properties": {"name": name},
    }


class _OapifHandler(http.server.BaseHTTPRequestHandler):
    """A minimal, protocol-faithful OGC API – Features service with a paged collection.

    Serves the landing page, ``/conformance``, ``/collections``, the collection
    metadata, and a two-page ``/items`` response: page 1 carries a ``rel="next"``
    link to page 2 (``offset=2``), exercising the OAPIF driver's paging.
    """

    def _json(self, doc: dict, content_type: str = "application/json"):
        payload = json.dumps(doc).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        base = f"http://{self.headers.get('Host')}"
        path = self.path.split("?", 1)[0]
        query = self.path[len(path):]
        if path in ("", "/"):
            self._json({"title": "Mock OAPIF", "links": [
                {"rel": "self", "href": f"{base}/", "type": "application/json"},
                {"rel": "conformance", "href": f"{base}/conformance", "type": "application/json"},
                {"rel": "data", "href": f"{base}/collections", "type": "application/json"},
            ]})
        elif path == "/conformance":
            self._json({"conformsTo": [
                "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
                "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson",
                "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/oas30",
            ]})
        elif path == "/collections":
            self._json({"links": [{"rel": "self", "href": f"{base}/collections"}], "collections": [{
                "id": "lakes", "title": "Lakes",
                "extent": {"spatial": {"bbox": [[-180, -90, 180, 90]]}},
                "links": [
                    {"rel": "items", "href": f"{base}/collections/lakes/items",
                     "type": "application/geo+json"},
                    {"rel": "self", "href": f"{base}/collections/lakes"},
                ],
            }]})
        elif path == "/collections/lakes":
            self._json({
                "id": "lakes", "title": "Lakes",
                "extent": {"spatial": {"bbox": [[-180, -90, 180, 90]]}},
                "links": [{"rel": "items", "href": f"{base}/collections/lakes/items",
                           "type": "application/geo+json"}],
            })
        elif path == "/collections/lakes/items" and "offset=2" not in query:
            self._json({
                "type": "FeatureCollection", "numberMatched": 3, "numberReturned": 2,
                "features": [_feature("1", 5.0, 52.0, "a"), _feature("2", 6.0, 51.0, "b")],
                "links": [
                    {"rel": "self", "href": f"{base}/collections/lakes/items"},
                    {"rel": "next", "href": f"{base}/collections/lakes/items?offset=2",
                     "type": "application/geo+json"},
                ],
            }, content_type="application/geo+json")
        elif path == "/collections/lakes/items":
            self._json({
                "type": "FeatureCollection", "numberMatched": 3, "numberReturned": 1,
                "features": [_feature("3", 7.0, 50.0, "c")],
                "links": [{"rel": "self", "href": f"{base}/collections/lakes/items?offset=2"}],
            }, content_type="application/geo+json")
        else:
            self.send_error(404)

    def log_message(self, *args, **kwargs):  # noqa: N802
        return


class TestOapifDriverPaging:
    """Drive the real OAPIF driver against a local two-page mock service.

    ``from_ogc_features`` reads through GDAL's OGR ``OAPIF`` driver (via
    ``gpd.read_file``); this test exercises that same driver directly with
    ``gdal.OpenEx`` so the ``rel="next"`` paging is proven offline. (The bundled
    pyogrio reader cannot reach a localhost mock reliably in CI, so the
    ``from_ogc_features`` wrapper itself is covered end-to-end only by the
    gated live test below.)
    """

    @pytest.fixture
    def oapif_service(self):
        """Start the paged mock OGC API – Features service; yield its base URL."""
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _OapifHandler)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
        httpd.shutdown()
        httpd.server_close()

    def test_driver_follows_next_link(self, oapif_service):
        """The OAPIF driver follows the rel=next link, so both pages' features are read."""
        gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", "15")
        try:
            ds = gdal.OpenEx(f"OAPIF:{oapif_service}", gdal.OF_VECTOR)
            layer = ds.GetLayerByName("lakes")
            names = sorted(f.GetField("name") for f in layer)
            layer = None
            ds = None  # release the dataset's HTTP handle before the server tears down
        finally:
            gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", None)
        assert names == ["a", "b", "c"]  # 2 from page 1 + 1 from page 2 (rel=next)


@pytest.mark.slow
@pytest.mark.live
class TestLiveOapif:
    ENDPOINT = "https://demo.pygeoapi.io/master"

    def test_live_read(self):
        """Exercise the real OGR OAPIF driver against a public endpoint (override via env)."""
        endpoint = os.environ.get("PYRAMIDS_OAPIF_ENDPOINT", self.ENDPOINT)
        collection = os.environ.get("PYRAMIDS_OAPIF_COLLECTION", "lakes")
        fc = FeatureCollection.from_ogc_features(endpoint, collection=collection, max_features=5)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) <= 5
