"""Tests for the OGC API – Coverages reader (`pyramids.dataset._ogc_coverages`).

Network-free. The happy path drives GDAL's native ``OGCAPI`` raster driver
end-to-end against an in-process ``ThreadingHTTPServer`` that speaks just enough
OGC API – Coverages (``/collections`` discovery, the collection document, the
``coverage-domainset`` / ``coverage-rangetype`` sub-documents, and a
``/coverage?subset=…&scaleSize=…`` endpoint that returns an on-the-fly GeoTIFF
georeferenced to the requested window). That proves the full wrapper read —
bounded ``projWin`` + size cap, native-CRS projection, Dataset wrapping,
``output_crs`` / ``output`` — runs offline with the real driver, no skip.

The error / argument-validation paths and the pure helpers are covered without a
server (a couple monkeypatch ``gdal``). A gated ``@pytest.mark.live`` test hits a
public service.
"""

from __future__ import annotations

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base import _ogc_api
from pyramids.dataset import Dataset, _ogc_coverages
from pyramids.errors import OGCAPIError, WCSError


@pytest.fixture(autouse=True)
def _clear_collections_cache():
    """Isolate the shared /collections LRU cache between tests."""
    _ogc_api.get_collections.cache_clear()
    yield
    _ogc_api.get_collections.cache_clear()


class TestPureHelpers:
    def test_coverage_connection_basic(self):
        assert (
            _ogc_coverages._coverage_connection("https://h/ogc", "cov")
            == "OGCAPI:https://h/ogc/collections/cov"
        )

    def test_coverage_connection_strips_trailing_slash(self):
        assert (
            _ogc_coverages._coverage_connection("https://h/ogc/", "cov")
            == "OGCAPI:https://h/ogc/collections/cov"
        )

    def test_coverage_connection_inserts_path_before_query(self):
        """A query-string endpoint keeps its query; /collections/{cov} lands before it."""
        conn = _ogc_coverages._coverage_connection("https://h/ogc?api_key=secret", "cov")
        assert conn == "OGCAPI:https://h/ogc/collections/cov?api_key=secret"
        # the coverage path segment precedes the query string
        assert conn.index("/collections/cov") < conn.index("?api_key=secret")

    def test_coverage_connection_url_encodes_coverage(self):
        """A coverage id with reserved characters becomes a single encoded segment."""
        assert (
            _ogc_coverages._coverage_connection("https://h/ogc", "a/b c")
            == "OGCAPI:https://h/ogc/collections/a%2Fb%20c"
        )

    def test_resolution_pair(self):
        assert _ogc_coverages._resolution_pair(None) is None
        assert _ogc_coverages._resolution_pair(250) == (250.0, 250.0)
        assert _ogc_coverages._resolution_pair((250, 500)) == (250.0, 500.0)

    def test_validate_bbox_ok(self):
        assert _ogc_coverages._validate_bbox((5.0, 51.0, 6.0, 52.0)) == (5.0, 51.0, 6.0, 52.0)

    @pytest.mark.parametrize(
        "bad",
        [(1, 2, 3), (6.0, 51.0, 5.0, 52.0), (5.0, 52.0, 6.0, 51.0)],
    )
    def test_validate_bbox_rejects(self, bad):
        with pytest.raises(ValueError):
            _ogc_coverages._validate_bbox(bad)

    def test_read_size_from_resolution(self):
        # projWin span is 2.0 x 2.0 native units; 0.01 res -> 200 x 200 px
        assert _ogc_coverages._read_size([2.0, 5.0, 4.0, 3.0], (0.01, 0.01)) == (200, 200)

    def test_read_size_nonsquare_resolution(self):
        assert _ogc_coverages._read_size([2.0, 5.0, 4.0, 3.0], (0.01, 0.02)) == (200, 100)

    def test_read_size_default_cap_wide(self):
        # span 4 (x) x 2 (y): longer side x -> width capped at 1024, height halved
        assert _ogc_coverages._read_size([0.0, 2.0, 4.0, 0.0], None) == (1024, 512)

    def test_read_size_default_cap_tall(self):
        assert _ogc_coverages._read_size([0.0, 4.0, 2.0, 0.0], None) == (512, 1024)

    def test_read_size_clamps_to_one(self):
        w, h = _ogc_coverages._read_size([2.0, 5.0, 4.0, 3.0], (1000.0, 1000.0))
        assert (w, h) == (1, 1)

    def test_read_size_rejects_oversize_window(self):
        """A fine resolution over a wide bbox exceeds the hard ceiling and is rejected."""
        with pytest.raises(ValueError, match="px limit"):
            _ogc_coverages._read_size([0.0, 2.0, 2.0, 0.0], (1e-5, 1e-5))


class TestOpenCoverage:
    def test_missing_driver_raises_ogcapierror(self, monkeypatch):
        monkeypatch.setattr(_ogc_coverages.gdal, "GetDriverByName", lambda name: None)
        with pytest.raises(OGCAPIError, match="OGCAPI driver is not available"):
            _ogc_coverages._open_coverage("OGCAPI:x", "cov")

    def test_gdal_runtimeerror_raises_ogcapierror(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("gdal could not open")

        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", boom)
        with pytest.raises(OGCAPIError, match="could not open OGC API coverage"):
            _ogc_coverages._open_coverage("OGCAPI:x", "cov")

    def test_gdal_none_raises_ogcapierror(self, monkeypatch):
        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", lambda *a, **k: None)
        with pytest.raises(OGCAPIError, match="no dataset"):
            _ogc_coverages._open_coverage("OGCAPI:x", "cov")


class TestTranslateWindow:
    def test_runtimeerror_raises_ogcapierror(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("translate blew up")

        monkeypatch.setattr(_ogc_coverages.gdal, "Translate", boom)
        with pytest.raises(OGCAPIError, match="read failed"):
            _ogc_coverages._translate_window(object(), [2.0, 5.0, 4.0, 3.0], (8, 8), "cov")

    def test_none_raises_ogcapierror(self, monkeypatch):
        monkeypatch.setattr(_ogc_coverages.gdal, "Translate", lambda *a, **k: None)
        with pytest.raises(OGCAPIError, match="no raster"):
            _ogc_coverages._translate_window(object(), [2.0, 5.0, 4.0, 3.0], (8, 8), "cov")


class TestFromOgcCoveragesValidation:
    """Argument / error handling that needs no live driver (monkeypatched)."""

    def _patch_collections(self, monkeypatch, ids=("cov",)):
        monkeypatch.setattr(_ogc_coverages, "_get_collections", lambda *a, **k: frozenset(ids))

    def test_bbox_is_required(self):
        with pytest.raises(TypeError):
            Dataset.from_ogc_coverages("https://h/ogc", coverage="cov")

    def test_unknown_coverage_raises_valueerror(self, monkeypatch):
        self._patch_collections(monkeypatch, ids=("other",))
        with pytest.raises(ValueError, match="not advertised"):
            Dataset.from_ogc_coverages(
                "https://h/ogc", coverage="cov", bbox=(5.0, 51.0, 6.0, 52.0)
            )

    def test_empty_collections_does_not_block(self, monkeypatch):
        """An empty /collections set (service advertises none) still attempts the open."""
        self._patch_collections(monkeypatch, ids=())
        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", lambda *a, **k: None)
        with pytest.raises(OGCAPIError, match="no dataset"):
            Dataset.from_ogc_coverages(
                "https://h/ogc", coverage="cov", bbox=(5.0, 51.0, 6.0, 52.0)
            )

    def test_bad_bbox_raises_before_network(self, monkeypatch):
        """An inverted bbox is rejected before any /collections or OpenEx call."""
        def fail(*a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("network must not be touched")

        monkeypatch.setattr(_ogc_coverages, "_get_collections", fail)
        with pytest.raises(ValueError, match="minx < maxx"):
            Dataset.from_ogc_coverages(
                "https://h/ogc", coverage="cov", bbox=(6.0, 51.0, 5.0, 52.0)
            )

    def test_openex_none_raises_ogcapierror(self, monkeypatch):
        self._patch_collections(monkeypatch)
        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", lambda *a, **k: None)
        with pytest.raises(OGCAPIError, match="no dataset"):
            Dataset.from_ogc_coverages(
                "https://h/ogc", coverage="cov", bbox=(5.0, 51.0, 6.0, 52.0)
            )

    def test_openex_runtimeerror_raises_ogcapierror(self, monkeypatch):
        self._patch_collections(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("driver said no")

        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", boom)
        with pytest.raises(OGCAPIError, match="could not open"):
            Dataset.from_ogc_coverages(
                "https://h/ogc", coverage="cov", bbox=(5.0, 51.0, 6.0, 52.0)
            )

    def test_auth_and_timeout_active_during_open(self, monkeypatch):
        """The coverage read runs inside a GDAL config context carrying auth + timeout."""
        self._patch_collections(monkeypatch)
        seen = {}

        def fake_openex(*a, **k):
            seen["userpwd"] = _ogc_coverages.gdal.GetConfigOption("GDAL_HTTP_USERPWD")
            seen["timeout"] = _ogc_coverages.gdal.GetConfigOption("GDAL_HTTP_TIMEOUT")
            raise RuntimeError("stop after capturing config")

        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", fake_openex)
        with pytest.raises(OGCAPIError):
            Dataset.from_ogc_coverages(
                "https://h/ogc",
                coverage="cov",
                bbox=(5.0, 51.0, 6.0, 52.0),
                auth=("u", "p"),
                timeout=42.0,
            )
        assert seen["userpwd"] == "u:p"
        assert seen["timeout"] == "42"

    def test_crs_less_coverage_raises_ogcapierror(self, monkeypatch):
        """A coverage with no resolvable CRS surfaces OGCAPIError, not the reused WCSError."""
        self._patch_collections(monkeypatch)
        mem = gdal.GetDriverByName("MEM").Create("", 4, 4, 1)
        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", lambda *a, **k: mem)

        def no_srs(*a, **k):
            raise WCSError("the WCS coverage has no resolvable spatial reference")

        monkeypatch.setattr(_ogc_coverages, "_resolve_native_srs", no_srs)
        with pytest.raises(OGCAPIError, match="no resolvable spatial reference"):
            Dataset.from_ogc_coverages(
                "https://h/ogc", coverage="cov", bbox=(5.0, 51.0, 6.0, 52.0)
            )

    def test_coverage_crs_forwarded_to_resolver(self, monkeypatch):
        """`coverage_crs` is handed to the CRS resolver so a non-PROJ coverage is shimmable."""
        self._patch_collections(monkeypatch)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        monkeypatch.setattr(
            _ogc_coverages.gdal, "OpenEx",
            lambda *a, **k: gdal.GetDriverByName("MEM").Create("", 4, 4, 1),
        )
        seen = {}

        def fake_resolve(src, coverage_crs):
            seen["coverage_crs"] = coverage_crs
            return srs

        monkeypatch.setattr(_ogc_coverages, "_resolve_native_srs", fake_resolve)
        monkeypatch.setattr(_ogc_coverages, "_native_projwin", lambda *a, **k: [5.0, 52.0, 6.0, 51.0])
        monkeypatch.setattr(
            _ogc_coverages, "_translate_window",
            lambda src, projwin, size, coverage: gdal.GetDriverByName("MEM").Create("", size[0], size[1], 1),
        )
        Dataset.from_ogc_coverages(
            "https://h/ogc", coverage="cov", bbox=(5.0, 51.0, 6.0, 52.0),
            coverage_crs="+proj=igh +datum=WGS84 +units=m +no_defs",
        )
        assert seen["coverage_crs"] == "+proj=igh +datum=WGS84 +units=m +no_defs"

    def test_resolution_sizes_the_native_read(self, monkeypatch):
        """`resolution` (native-CRS units) drives the windowed read size, not a post-warp cell size."""
        self._patch_collections(monkeypatch)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        monkeypatch.setattr(
            _ogc_coverages.gdal, "OpenEx",
            lambda *a, **k: gdal.GetDriverByName("MEM").Create("", 4, 4, 1),
        )
        monkeypatch.setattr(_ogc_coverages, "_resolve_native_srs", lambda *a, **k: srs)
        monkeypatch.setattr(_ogc_coverages, "_native_projwin", lambda *a, **k: [5.0, 52.0, 6.0, 51.0])
        seen = {}

        def fake_translate(src, projwin, size, coverage):
            seen["size"] = size
            return gdal.GetDriverByName("MEM").Create("", size[0], size[1], 1)

        monkeypatch.setattr(_ogc_coverages, "_translate_window", fake_translate)
        Dataset.from_ogc_coverages(
            "https://h/ogc", coverage="cov", bbox=(5.0, 51.0, 6.0, 52.0), resolution=0.01,
        )
        assert seen["size"] == (100, 100)  # 1° span / 0.01° → 100 px each side


# Coverage geo definition (EPSG:4326, Lat/Long order like GNOSIS).
_MINX, _MINY, _MAXX, _MAXY = 0.0, 0.0, 10.0, 8.0
_RES = 0.01
_NX = int(round((_MAXX - _MINX) / _RES))  # 1000
_NY = int(round((_MAXY - _MINY) / _RES))  # 800

_COLLECTION = {
    "id": "demo",
    "title": "Demo coverage",
    "extent": {
        "spatial": {
            "bbox": [[_MINX, _MINY, _MAXX, _MAXY]],
            "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
        }
    },
    "crs": ["http://www.opengis.net/def/crs/OGC/1.3/CRS84"],
    "links": [
        {"rel": "self", "type": "application/json", "href": "http://HOST/collections/demo"},
        {
            "rel": "http://www.opengis.net/def/rel/ogc/1.0/coverage",
            "type": "image/tiff; application=geotiff",
            "href": "http://HOST/collections/demo/coverage",
        },
        {
            "rel": "http://www.opengis.net/def/rel/ogc/1.0/coverage-domainset",
            "type": "application/json",
            "href": "http://HOST/collections/demo/coverage/domainset",
        },
        {
            "rel": "http://www.opengis.net/def/rel/ogc/1.0/coverage-rangetype",
            "type": "application/json",
            "href": "http://HOST/collections/demo/coverage/rangetype",
        },
    ],
}
_DISCOVERY = {
    "links": [{"rel": "self", "type": "application/json", "href": "http://HOST/collections"}],
    "collections": [_COLLECTION],
}
_DOMAINSET = {
    "type": "DomainSet",
    "generalGrid": {
        "type": "GeneralGridCoverage",
        "srsName": "http://www.opengis.net/def/crs/EPSG/0/4326",
        "axisLabels": ["Lat", "Long"],
        "axis": [
            {
                "type": "RegularAxis", "axisLabel": "Lat", "lowerBound": _MINY,
                "upperBound": _MAXY, "resolution": _RES, "uomLabel": "deg",
            },
            {
                "type": "RegularAxis", "axisLabel": "Long", "lowerBound": _MINX,
                "upperBound": _MAXX, "resolution": _RES, "uomLabel": "deg",
            },
        ],
        "gridLimits": {
            "type": "GridLimits",
            "axisLabels": ["i", "j"],
            "axis": [
                {"type": "IndexAxis", "axisLabel": "i", "lowerBound": 0, "upperBound": _NY - 1},
                {"type": "IndexAxis", "axisLabel": "j", "lowerBound": 0, "upperBound": _NX - 1},
            ],
        },
    },
}
_RANGETYPE = {
    "type": "DataRecord",
    "field": [
        {
            "type": "Quantity", "name": "band1", "encodingInfo": {"dataType": "FLOAT32"},
            "definition": "http://www.opengis.net/def/dataType/OGC/0/float32",
        }
    ],
}

_SUBSET_LON_RE = re.compile(r"(?:Long|Lon)\(([-\d.]+):([-\d.]+)\)")
_SUBSET_LAT_RE = re.compile(r"Lat\(([-\d.]+):([-\d.]+)\)")
_SCALE_RE = re.compile(r"(?:Long|Lon)\((\d+)\),Lat\((\d+)\)")


def _make_geotiff(width, height, minx, miny, maxx, maxy):
    """GeoTIFF bytes of the given pixel size, georeferenced to the window, gradient values."""
    drv = gdal.GetDriverByName("GTiff")
    path = f"/vsimem/ogccov_{threading.get_ident()}_{width}x{height}.tif"
    ds = drv.Create(path, width, height, 1, gdal.GDT_Float32)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds.SetProjection(sr.ExportToWkt())
    ds.SetGeoTransform([minx, (maxx - minx) / width, 0, maxy, 0, -(maxy - miny) / height])
    yy, xx = np.mgrid[0:height, 0:width]
    ds.GetRasterBand(1).WriteArray((xx + yy).astype(np.float32) + 100.0)
    ds.FlushCache()
    ds = None
    f = gdal.VSIFOpenL(path, "rb")
    gdal.VSIFSeekL(f, 0, 2)
    n = gdal.VSIFTellL(f)
    gdal.VSIFSeekL(f, 0, 0)
    data = gdal.VSIFReadL(1, n, f)
    gdal.VSIFCloseL(f)
    gdal.Unlink(path)
    return data


class _CoverageHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence the default stderr logging
        pass

    def _json(self, obj):
        host = f"127.0.0.1:{self.server.server_port}"
        body = json.dumps(obj).replace("HOST", host).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/collections":
            return self._json(_DISCOVERY)
        if path in ("/collections/demo", "/collections/demo/"):
            return self._json(_COLLECTION)
        if path == "/collections/demo/coverage/domainset":
            return self._json(_DOMAINSET)
        if path == "/collections/demo/coverage/rangetype":
            return self._json(_RANGETYPE)
        if path == "/collections/demo/coverage":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            mlon = _SUBSET_LON_RE.search(query)
            mlat = _SUBSET_LAT_RE.search(query)
            msc = _SCALE_RE.search(query)
            if mlon and mlat and msc:
                x0, x1 = sorted(map(float, mlon.groups()))
                y0, y1 = sorted(map(float, mlat.groups()))
                w, h = int(msc.group(1)), int(msc.group(2))
            else:
                x0, y0, x1, y1, w, h = _MINX, _MINY, _MAXX, _MAXY, _NX, _NY
            tif = _make_geotiff(max(w, 1), max(h, 1), x0, y0, x1, y1)
            self.send_response(200)
            self.send_header("Content-Type", "image/tiff")
            self.send_header("Content-Length", str(len(tif)))
            self.end_headers()
            self.wfile.write(tif)
            return
        self.send_error(404, "not found: " + path)


@pytest.fixture(scope="class")
def coverage_server():
    """A threaded in-process OGC API – Coverages mock; yields its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CoverageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestRealDriverOffline:
    """Drive the real GDAL OGCAPI driver end-to-end against the in-process mock."""

    def test_bounded_read_at_resolution(self, coverage_server):
        ds = Dataset.from_ogc_coverages(
            coverage_server, coverage="demo", bbox=(2.0, 3.0, 4.0, 5.0), resolution=_RES
        )
        assert isinstance(ds, Dataset)
        assert ds.epsg == 4326
        # projWin span 2x2 deg at 0.01 deg/px -> a bounded 200x200 read, not the planet
        assert ds.shape == (1, 200, 200)
        arr = np.asarray(ds.read_array())
        assert np.isfinite(arr).any()
        assert float(np.nanstd(arr)) > 0.0

    def test_default_cap_keeps_read_bounded(self, coverage_server):
        """With no resolution the longer side is capped at 1024 px (aspect preserved)."""
        ds = Dataset.from_ogc_coverages(
            coverage_server, coverage="demo", bbox=(2.0, 3.0, 4.0, 7.0)
        )
        # span 2 (lon) x 4 (lat): tall window -> height 1024, width 512
        assert ds.shape == (1, 1024, 512)

    def test_output_crs_reprojects(self, coverage_server):
        ds = Dataset.from_ogc_coverages(
            coverage_server,
            coverage="demo",
            bbox=(2.0, 3.0, 4.0, 5.0),
            resolution=_RES,
            output_crs="EPSG:3857",
        )
        assert ds.epsg == 3857

    def test_output_writes_reopenable_file(self, coverage_server, tmp_path):
        out = tmp_path / "coverage_out.tif"
        ds = Dataset.from_ogc_coverages(
            coverage_server,
            coverage="demo",
            bbox=(2.0, 3.0, 4.0, 5.0),
            resolution=_RES,
            output=out,
        )
        assert isinstance(ds, Dataset)
        assert out.exists()
        assert Dataset.read_file(str(out)).shape == (1, 200, 200)


@pytest.mark.slow
@pytest.mark.live
class TestLiveOgcCoverages:
    """Gated live test against a public service.

    GNOSIS (``maps.gnosis.earth``) currently answers the coverage read with HTTP
    401 for anonymous clients, so this is expected to require credentials; it is
    kept behind ``-m live`` and skipped in the default suite.
    """

    ENDPOINT = "https://maps.gnosis.earth/ogcapi"

    def test_live_read(self):
        coverage = os.environ.get("PYRAMIDS_OGC_COVERAGES_NAME", "SRTM_ViewFinderPanorama")
        ds = Dataset.from_ogc_coverages(
            self.ENDPOINT, coverage=coverage, bbox=(5.0, 51.0, 6.0, 52.0)
        )
        assert isinstance(ds, Dataset)
        assert ds.shape[0] >= 1
        assert ds.shape[1] > 0 and ds.shape[2] > 0
