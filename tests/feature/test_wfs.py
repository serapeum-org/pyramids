"""Tests for the OGC WFS reader (`pyramids.feature._wfs` / `FeatureCollection.from_wfs`).

Network-free. The successful ``GetFeature`` path drives GDAL's native OGR WFS
driver, exercised against a protocol-faithful mock in ``test_wfs_driver.py``.
Here the OGR read is monkeypatched so ``from_wfs``'s own logic — capabilities
validation, the read filters, FeatureCollection wrapping, ``output_crs`` reproject
and error normalisation — is covered without a live server, plus the pure helpers
and the capabilities fetch/parse/cache.
"""

from __future__ import annotations

import os

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.feature import FeatureCollection
from pyramids.feature import _wfs
from pyramids.errors import WFSError
from tests.http_mock import make_fixed_body_server

CAPS_2_0_0 = """<?xml version="1.0" encoding="UTF-8"?>
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0"
    xmlns:ows="http://www.opengis.net/ows/1.1" version="2.0.0">
  <ows:ServiceIdentification>
    <ows:ServiceTypeVersion>2.0.0</ows:ServiceTypeVersion>
    <ows:ServiceTypeVersion>1.1.0</ows:ServiceTypeVersion>
  </ows:ServiceIdentification>
  <FeatureTypeList>
    <FeatureType><Name>topp:states</Name></FeatureType>
    <FeatureType><Name>topp:roads</Name></FeatureType>
  </FeatureTypeList>
</wfs:WFS_Capabilities>
"""

EXCEPTION_REPORT = """<?xml version="1.0" encoding="UTF-8"?>
<ows:ExceptionReport xmlns:ows="http://www.opengis.net/ows/1.1" version="2.0.0">
  <ows:Exception exceptionCode="NoApplicableCode">
    <ows:ExceptionText>WFS server error.</ows:ExceptionText>
  </ows:Exception>
</ows:ExceptionReport>
"""

SERVICE_EXCEPTION_1X = """<?xml version="1.0" encoding="UTF-8"?>
<ServiceExceptionReport version="1.1.0">
  <ServiceException code="InvalidParameterValue">legacy WFS error.</ServiceException>
</ServiceExceptionReport>
"""


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    """Isolate the module-level capabilities LRU cache between tests."""
    _wfs._get_capabilities.cache_clear()
    yield
    _wfs._get_capabilities.cache_clear()


def _make_server(body: str, content_type: str = "application/xml"):
    """Local HTTP server returning `body` for every GET; returns (url, counter, httpd)."""
    return make_fixed_body_server(body, content_type, path="/ows")


@pytest.fixture
def caps_server():
    """A mock server serving the 2.0.0 GetCapabilities document."""
    url, counter, httpd = _make_server(CAPS_2_0_0)
    yield url, counter
    httpd.shutdown()
    httpd.server_close()


def _sample_gdf(crs="EPSG:4326") -> gpd.GeoDataFrame:
    """A tiny two-feature GeoDataFrame standing in for a GetFeature response."""
    return gpd.GeoDataFrame(
        {"name": ["a", "b"], "persons": [2_000_000, 500_000]},
        geometry=[Point(5.0, 52.0), Point(6.0, 51.0)],
        crs=crs,
    )


class TestPureHelpers:
    def test_capabilities_url(self):
        with_q = _wfs._capabilities_url("https://h/ows?map=x", None)
        assert with_q == "https://h/ows?map=x&SERVICE=WFS&REQUEST=GetCapabilities"
        no_q = _wfs._capabilities_url("https://h/ows", "2.0.0")
        assert no_q == "https://h/ows?SERVICE=WFS&REQUEST=GetCapabilities&VERSION=2.0.0"

    def test_localname(self):
        assert _wfs._localname("{http://www.opengis.net/wfs/2.0}Name") == "Name"
        assert _wfs._localname("Name") == "Name"

    def test_wfs_connection(self):
        assert _wfs._wfs_connection("https://h/ows", None) == "WFS:https://h/ows"
        assert _wfs._wfs_connection("https://h/ows", "1.1.0") == "WFS:https://h/ows?VERSION=1.1.0"
        assert _wfs._wfs_connection("https://h/ows?a=b", "2.0.0") == "WFS:https://h/ows?a=b&VERSION=2.0.0"

    def test_gdal_http_config(self):
        assert _wfs._gdal_http_config(None, 60.0) == {"GDAL_HTTP_TIMEOUT": "60"}
        cfg = _wfs._gdal_http_config(("u", "p"), 30.0)
        assert cfg["GDAL_HTTP_USERPWD"] == "u:p" and cfg["GDAL_HTTP_TIMEOUT"] == "30"

    def test_gdal_http_config_clamps_subsecond_timeout(self):
        assert _wfs._gdal_http_config(None, 0.5)["GDAL_HTTP_TIMEOUT"] == "1"

    def test_read_kwargs(self):
        assert _wfs._read_kwargs(None, None, None) == {}
        kw = _wfs._read_kwargs((1.0, 2.0, 3.0, 4.0), "x>1", 10)
        assert kw == {"bbox": (1.0, 2.0, 3.0, 4.0), "where": "x>1", "rows": 10}

    def test_read_kwargs_rejects_negative_max_features(self):
        with pytest.raises(ValueError, match="max_features"):
            _wfs._read_kwargs(None, None, -1)

    def test_read_kwargs_rejects_zero_max_features(self):
        """0 is rejected: pyogrio reads rows=0 as 'no limit', so a 0 cap would fetch everything."""
        with pytest.raises(ValueError, match="max_features must be >= 1"):
            _wfs._read_kwargs(None, None, 0)

    def test_read_kwargs_rejects_bad_bbox_length(self):
        with pytest.raises(ValueError, match="minx, miny, maxx, maxy"):
            _wfs._read_kwargs((1.0, 2.0, 3.0), None, None)

    def test_read_kwargs_rejects_inverted_bbox(self):
        with pytest.raises(ValueError, match="minx < maxx"):
            _wfs._read_kwargs((3.0, 2.0, 1.0, 4.0), None, None)

    def test_extract_typenames_only_under_featuretype(self):
        root = _wfs.ET.fromstring(
            "<Caps><Service><Name>svc</Name></Service>"
            "<FeatureTypeList>"
            "<FeatureType><Title>t</Title><Name>a:b</Name></FeatureType>"
            "<FeatureType><Name>c:d</Name></FeatureType></FeatureTypeList></Caps>"
        )
        assert _wfs._extract_typenames(root) == {"a:b", "c:d"}  # service <Name> + <Title> excluded

    def test_exception_text(self):
        root = _wfs.ET.fromstring("<R><ExceptionText>boom</ExceptionText></R>")
        assert _wfs._exception_text(root) == "boom"
        svc = _wfs.ET.fromstring("<R><ServiceException>legacy boom</ServiceException></R>")
        assert _wfs._exception_text(svc) == "legacy boom"  # WFS 1.x form
        assert _wfs._exception_text(_wfs.ET.fromstring("<R></R>")) == "no message provided"


class TestCapabilities:
    def test_parses_versions_and_typenames(self, caps_server):
        url, _ = caps_server
        versions, typenames = _wfs._get_capabilities(url, None, None, 30.0)
        assert "2.0.0" in versions and "1.1.0" in versions
        assert typenames == {"topp:states", "topp:roads"}

    def test_lru_cache_one_fetch_per_endpoint(self, caps_server):
        url, counter = caps_server
        _wfs._get_capabilities(url, None, None, 30.0)
        _wfs._get_capabilities(url, None, None, 30.0)
        assert counter["GET"] == 1

    def test_auth_wires_a_basic_auth_handler(self, caps_server):
        url, _ = caps_server
        _, typenames = _wfs._get_capabilities(url, None, ("user", "secret"), 30.0)
        assert "topp:states" in typenames

    def test_exception_report_raises_wfserror(self):
        url, _, httpd = _make_server(EXCEPTION_REPORT)
        try:
            with pytest.raises(WFSError, match="WFS server error"):
                _wfs._get_capabilities(url, None, None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_service_exception_report_1x_raises_wfserror(self):
        """A WFS 1.x ServiceExceptionReport body surfaces as WFSError with its message."""
        url, _, httpd = _make_server(SERVICE_EXCEPTION_1X)
        try:
            with pytest.raises(WFSError, match="legacy WFS error"):
                _wfs._get_capabilities(url, None, None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_non_xml_body_raises_wfserror(self):
        url, _, httpd = _make_server("not xml", content_type="text/plain")
        try:
            with pytest.raises(WFSError, match="non-XML"):
                _wfs._get_capabilities(url, None, None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_transport_failure_raises_wfserror(self, monkeypatch):
        def boom(self, *args, **kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr(_wfs.urllib.request.OpenerDirector, "open", boom)
        with pytest.raises(WFSError, match="request failed"):
            _wfs._get_capabilities("https://wfs.invalid/x", None, None, 5.0)


class TestFromWfs:
    def _patch_caps(self, monkeypatch, typenames=("topp:states",)):
        monkeypatch.setattr(_wfs, "_get_capabilities", lambda *a, **k: ((), frozenset(typenames)))

    def test_returns_featurecollection(self, monkeypatch):
        """A successful read is wrapped into a FeatureCollection."""
        self._patch_caps(monkeypatch)
        monkeypatch.setattr(_wfs.gpd, "read_file", lambda *a, **k: _sample_gdf())
        fc = FeatureCollection.from_wfs("https://h/ows", typename="topp:states")
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 2 and fc.crs.to_epsg() == 4326

    def test_passes_filters_to_read(self, monkeypatch):
        """bbox / where / max_features are forwarded to the OGR read as filters."""
        self._patch_caps(monkeypatch)
        captured = {}

        def fake_read(connection, **kwargs):
            captured["connection"] = connection
            captured["kwargs"] = kwargs
            return _sample_gdf()

        monkeypatch.setattr(_wfs.gpd, "read_file", fake_read)
        FeatureCollection.from_wfs(
            "https://h/ows", typename="topp:states", bbox=(1.0, 2.0, 3.0, 4.0),
            where="persons > 1000000", max_features=5, version="2.0.0",
        )
        assert captured["connection"] == "WFS:https://h/ows?VERSION=2.0.0"
        assert captured["kwargs"]["layer"] == "topp:states"
        assert captured["kwargs"]["bbox"] == (1.0, 2.0, 3.0, 4.0)
        assert captured["kwargs"]["where"] == "persons > 1000000"
        assert captured["kwargs"]["rows"] == 5

    def test_output_crs_reprojects(self, monkeypatch):
        self._patch_caps(monkeypatch)
        monkeypatch.setattr(_wfs.gpd, "read_file", lambda *a, **k: _sample_gdf())
        fc = FeatureCollection.from_wfs(
            "https://h/ows", typename="topp:states", output_crs="EPSG:3857"
        )
        assert fc.crs.to_epsg() == 3857

    def test_output_crs_without_result_crs_raises(self, monkeypatch):
        """output_crs on a CRS-less result raises WFSError instead of silently dropping it."""
        self._patch_caps(monkeypatch)
        crsless = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Point(5.0, 52.0)])
        monkeypatch.setattr(_wfs.gpd, "read_file", lambda *a, **k: crsless)
        with pytest.raises(WFSError, match="without a CRS"):
            FeatureCollection.from_wfs(
                "https://h/ows", typename="topp:states", output_crs="EPSG:3857"
            )

    def test_unsupported_version_raises_valueerror(self, monkeypatch):
        """A version the server does not advertise raises a clear ValueError."""
        monkeypatch.setattr(
            _wfs, "_get_capabilities",
            lambda *a, **k: (("1.1.0", "2.0.0"), frozenset({"topp:states"})),
        )
        with pytest.raises(ValueError, match="version '3.0.0' is not advertised"):
            FeatureCollection.from_wfs("https://h/ows", typename="topp:states", version="3.0.0")

    def test_advertised_version_passes(self, monkeypatch):
        """A version the server advertises is accepted and the read proceeds."""
        monkeypatch.setattr(
            _wfs, "_get_capabilities",
            lambda *a, **k: (("1.1.0", "2.0.0"), frozenset({"topp:states"})),
        )
        monkeypatch.setattr(_wfs.gpd, "read_file", lambda *a, **k: _sample_gdf())
        fc = FeatureCollection.from_wfs("https://h/ows", typename="topp:states", version="2.0.0")
        assert isinstance(fc, FeatureCollection)

    def test_unknown_typename_raises_valueerror(self, monkeypatch):
        self._patch_caps(monkeypatch, typenames=("topp:states",))
        with pytest.raises(ValueError, match="not advertised"):
            FeatureCollection.from_wfs("https://h/ows", typename="topp:missing")

    def test_read_failure_raises_wfserror(self, monkeypatch):
        self._patch_caps(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("driver said no")

        monkeypatch.setattr(_wfs.gpd, "read_file", boom)
        with pytest.raises(WFSError, match="GetFeature failed"):
            FeatureCollection.from_wfs("https://h/ows", typename="topp:states")


@pytest.mark.slow
@pytest.mark.live
class TestLiveWfs:
    def test_live_read(self):
        """Exercise the real OGR WFS driver against a caller-supplied public endpoint."""
        endpoint = os.environ["PYRAMIDS_WFS_ENDPOINT"]
        typename = os.environ["PYRAMIDS_WFS_TYPENAME"]
        fc = FeatureCollection.from_wfs(endpoint, typename=typename, max_features=5)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) <= 5
