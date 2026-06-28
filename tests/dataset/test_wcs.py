"""Tests for the OGC WCS reader (`pyramids.dataset._wcs` / `Dataset.from_wcs`).

Network-free. The successful ``GetCoverage`` path drives GDAL's native WCS
driver, which needs a protocol-faithful ``DescribeCoverage`` + ``GetCoverage``
mock; that fixture is a follow-up. What is covered here without GDAL's driver:

* every pure helper (bbox / resolution / descriptor / CRS shim / projWin),
* the capabilities fetch + parse + LRU cache (via a stdlib mock server),
* the unknown-coverage ``ValueError`` and the ``<ows:ExceptionReport>``
  ``WCSError`` — both reachable through ``Dataset.from_wcs`` because coverage
  validation runs before GDAL is opened.

A gated live end-to-end against SoilGrids runs only when ``PYRAMIDS_WCS_LIVE``
is set, so normal CI stays offline.
"""

from __future__ import annotations

import os

import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset
from pyramids.dataset import _wcs
from pyramids.errors import WCSError
from tests.dataset.wcs_mock_server import WcsMock
from tests.http_mock import make_fixed_body_server

CAPS_2_0_1 = """<?xml version="1.0" encoding="UTF-8"?>
<wcs:Capabilities xmlns:wcs="http://www.opengis.net/wcs/2.0"
                  xmlns:ows="http://www.opengis.net/ows/2.0" version="2.0.1">
  <ows:ServiceIdentification>
    <ows:ServiceTypeVersion>2.0.1</ows:ServiceTypeVersion>
    <ows:ServiceTypeVersion>1.0.0</ows:ServiceTypeVersion>
  </ows:ServiceIdentification>
  <wcs:Contents>
    <wcs:CoverageSummary><wcs:CoverageId>nitrogen_0-5cm_mean</wcs:CoverageId></wcs:CoverageSummary>
    <wcs:CoverageSummary><wcs:CoverageId>nitrogen_5-15cm_mean</wcs:CoverageId></wcs:CoverageSummary>
  </wcs:Contents>
</wcs:Capabilities>
"""

EXCEPTION_REPORT = """<?xml version="1.0" encoding="UTF-8"?>
<ows:ExceptionReport xmlns:ows="http://www.opengis.net/ows/2.0" version="2.0.1">
  <ows:Exception exceptionCode="NoApplicableCode">
    <ows:ExceptionText>msWCSGetCapabilities(): WCS server error.</ows:ExceptionText>
  </ows:Exception>
</ows:ExceptionReport>
"""


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    """Isolate the module-level capabilities LRU cache between tests."""
    _wcs._get_capabilities.cache_clear()
    yield
    _wcs._get_capabilities.cache_clear()


def _make_server(body: str, content_type: str = "application/xml"):
    """Local HTTP server returning `body` for every GET; returns (url, counter, httpd)."""
    return make_fixed_body_server(body, content_type, path="/mapserv?map=/map/nitrogen.map")


@pytest.fixture
def caps_server():
    """A mock server serving the 2.0.1 GetCapabilities document."""
    url, counter, httpd = _make_server(CAPS_2_0_1)
    yield url, counter
    httpd.shutdown()
    httpd.server_close()


class TestPureHelpers:
    def test_validate_bbox_ok(self):
        assert _wcs._validate_bbox((5.0, 51.0, 6.0, 52.0)) == (5.0, 51.0, 6.0, 52.0)

    @pytest.mark.parametrize(
        "bad",
        [(1, 2, 3), (6.0, 51.0, 5.0, 52.0), (5.0, 52.0, 6.0, 51.0)],
    )
    def test_validate_bbox_rejects(self, bad):
        with pytest.raises(ValueError):
            _wcs._validate_bbox(bad)

    def test_resolution_pair(self):
        assert _wcs._resolution_pair(None) is None
        assert _wcs._resolution_pair(250) == (250.0, 250.0)
        assert _wcs._resolution_pair((250, 500)) == (250.0, 500.0)

    def test_localname(self):
        assert _wcs._localname("{http://www.opengis.net/wcs/2.0}CoverageId") == "CoverageId"
        assert _wcs._localname("name") == "name"

    def test_capabilities_url_appends_correctly(self):
        with_q = _wcs._capabilities_url("https://h/mapserv?map=/m.map", None)
        assert with_q == "https://h/mapserv?map=/m.map&SERVICE=WCS&REQUEST=GetCapabilities"
        no_q = _wcs._capabilities_url("https://h/wcs", "2.0.1")
        assert no_q == "https://h/wcs?SERVICE=WCS&REQUEST=GetCapabilities&VERSION=2.0.1"

    def test_service_descriptor(self):
        xml = _wcs._service_descriptor(
            "https://h/mapserv?map=/m.map", "cov", "2.0.1", "GEOTIFF_INT16", {"FOO": "bar"}
        )
        assert "<CoverageName>cov</CoverageName>" in xml
        assert "<Version>2.0.1</Version>" in xml
        assert "<PreferredFormat>GEOTIFF_INT16</PreferredFormat>" in xml
        assert "&amp;FOO=bar" in xml
        # the ampersand in the ServiceURL query string is escaped
        assert "&amp;SERVICE" not in xml  # no SERVICE param injected here
        assert "map=/m.map" in xml

    def test_service_descriptor_minimal(self):
        xml = _wcs._service_descriptor("https://h", "cov", None, None, None)
        assert "<Version>" not in xml
        assert "<PreferredFormat>" not in xml
        assert "<GetCoverageExtra>" not in xml

    def test_gdal_http_config(self):
        assert _wcs._gdal_http_config(None, 60.0) == {"GDAL_HTTP_TIMEOUT": "60"}
        cfg = _wcs._gdal_http_config(("u", "p"), 30.0)
        assert cfg["GDAL_HTTP_USERPWD"] == "u:p"
        assert cfg["GDAL_HTTP_TIMEOUT"] == "30"


class TestNativeCrsShim:
    def _mem(self, with_srs: bool):
        ds = gdal.GetDriverByName("MEM").Create("", 4, 4, 1, gdal.GDT_Int16)
        if with_srs:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            ds.SetSpatialRef(srs)
        return ds

    def test_uses_server_srs_when_present(self):
        srs = _wcs._resolve_native_srs(self._mem(with_srs=True), None)
        assert srs.GetAuthorityCode(None) == "4326"

    def test_requires_coverage_crs_when_missing(self):
        with pytest.raises(WCSError, match="coverage_crs"):
            _wcs._resolve_native_srs(self._mem(with_srs=False), None)

    def test_applies_coverage_crs_shim(self):
        igh = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        srs = _wcs._resolve_native_srs(self._mem(with_srs=False), igh)
        assert "igh" in srs.ExportToProj4()

    def test_rejects_bad_coverage_crs(self):
        with pytest.raises(ValueError, match="coverage_crs"):
            _wcs._resolve_native_srs(self._mem(with_srs=False), "not-a-crs!!!")

    def test_native_projwin_reprojects_bbox(self):
        igh = osr.SpatialReference()
        igh.ImportFromProj4("+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs")
        ulx, uly, lrx, lry = _wcs._native_projwin((5.0, 51.0, 6.0, 52.0), "EPSG:4326", igh)
        # Netherlands lands ~1.4-1.6 Mm east, ~5.6-5.7 Mm north in IGH metres
        assert 1.4e6 < ulx < 1.6e6
        assert 5.6e6 < uly < 5.8e6
        assert ulx < lrx and uly > lry  # ul/lr ordering


class TestCapabilities:
    def test_parses_versions_and_coverages(self, caps_server):
        url, _ = caps_server
        versions, coverages = _wcs._get_capabilities(url, None, None, 30.0)
        assert "2.0.1" in versions and "1.0.0" in versions
        assert "nitrogen_0-5cm_mean" in coverages
        assert "nitrogen_5-15cm_mean" in coverages

    def test_lru_cache_one_fetch_per_endpoint(self, caps_server):
        url, counter = caps_server
        _wcs._get_capabilities(url, None, None, 30.0)
        _wcs._get_capabilities(url, None, None, 30.0)
        assert counter["GET"] == 1

    def test_exception_report_raises_wcserror(self):
        url, _, httpd = _make_server(EXCEPTION_REPORT)
        try:
            with pytest.raises(WCSError, match="WCS server error"):
                _wcs._get_capabilities(url, None, None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_non_xml_body_raises_wcserror(self):
        url, _, httpd = _make_server("not xml at all", content_type="text/plain")
        try:
            with pytest.raises(WCSError, match="non-XML"):
                _wcs._get_capabilities(url, None, None, 30.0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_auth_wires_a_basic_auth_handler(self, caps_server):
        """Passing auth installs an HTTP Basic-auth handler and still parses caps."""
        url, _ = caps_server
        _, coverages = _wcs._get_capabilities(url, None, ("user", "secret"), 30.0)
        assert "nitrogen_0-5cm_mean" in coverages

    def test_transport_failure_raises_wcserror(self, monkeypatch):
        """A transport-level OSError from the opener surfaces as WCSError."""

        def boom(self, *args, **kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr(_wcs.urllib.request.OpenerDirector, "open", boom)
        with pytest.raises(WCSError, match="request failed"):
            _wcs._get_capabilities("https://wcs.invalid/x", None, None, 5.0)

    def test_exception_text_falls_back_without_exception_element(self):
        """_exception_text returns body text, or a default when nothing is present."""
        body = _wcs.ET.fromstring("<ExceptionReport>broken upstream</ExceptionReport>")
        assert _wcs._exception_text(body) == "broken upstream"
        empty = _wcs.ET.fromstring("<ExceptionReport></ExceptionReport>")
        assert _wcs._exception_text(empty) == "no message provided"


class TestOpenService:
    def test_gdal_runtimeerror_raises_wcserror(self, monkeypatch):
        """A GDAL RuntimeError while opening the service becomes WCSError."""

        def boom(_descriptor):
            raise RuntimeError("gdal could not open")

        monkeypatch.setattr(_wcs.gdal, "Open", boom)
        with pytest.raises(WCSError, match="could not open WCS coverage"):
            _wcs._open_service("<WCS_GDAL/>", "cov")

    def test_gdal_none_raises_wcserror(self, monkeypatch):
        """A None return from gdal.Open becomes WCSError."""
        monkeypatch.setattr(_wcs.gdal, "Open", lambda _descriptor: None)
        with pytest.raises(WCSError, match="no dataset"):
            _wcs._open_service("<WCS_GDAL/>", "cov")


class TestFromWcsValidation:
    def test_unknown_coverage_raises_valueerror(self, caps_server):
        url, _ = caps_server
        with pytest.raises(ValueError, match="not advertised"):
            Dataset.from_wcs(url, coverage="does_not_exist", bbox=(5.0, 51.0, 6.0, 52.0))

    def test_bad_bbox_raises_before_network(self):
        with pytest.raises(ValueError, match="minx < maxx"):
            Dataset.from_wcs(
                "http://127.0.0.1:1/wcs", coverage="cov", bbox=(6.0, 51.0, 5.0, 52.0)
            )

    def test_translate_window_none_raises(self, monkeypatch):
        monkeypatch.setattr(_wcs.gdal, "Translate", lambda *a, **k: None)
        with pytest.raises(WCSError, match="no raster"):
            _wcs._translate_window(object(), [0, 1, 1, 0], "cov")

    def test_translate_window_runtimeerror_raises(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("server said no")

        monkeypatch.setattr(_wcs.gdal, "Translate", boom)
        with pytest.raises(WCSError, match="GetCoverage failed"):
            _wcs._translate_window(object(), [0, 1, 1, 0], "cov")


class TestDriverFullCycle:
    """Offline end-to-end against a mock that drives GDAL's WCS driver fully."""

    def test_1_0_0_returns_dataset_and_uses_bbox_shape(self):
        with WcsMock(version="1.0.0") as server:
            ds = Dataset.from_wcs(
                server.url, coverage="test_cov", bbox=(2.0, 2.0, 4.0, 4.0), version="1.0.0"
            )
            assert ds.shape == (1, 20, 20)
            assert ds.epsg == 4326
            assert [round(v, 3) for v in ds.geotransform] == [2.0, 0.1, 0.0, 4.0, 0.0, -0.1]
            getcov = server.getcoverage_requests()[-1].lower()
            assert "bbox=" in getcov  # 1.0.0 call shape
            assert "subset=" not in getcov

    def test_2_0_1_returns_dataset_and_uses_subset_shape(self):
        with WcsMock(version="2.0.1") as server:
            ds = Dataset.from_wcs(
                server.url, coverage="test_cov", bbox=(2.0, 2.0, 4.0, 4.0), version="2.0.1"
            )
            assert ds.shape == (1, 20, 20)
            getcov = server.getcoverage_requests()[-1].lower()
            assert "coverageid=test_cov" in getcov  # 2.0.x call shape
            assert "subset=" in getcov
            assert "bbox=" not in getcov

    def test_returned_raster_is_readable(self):
        with WcsMock(version="2.0.1") as server:
            ds = Dataset.from_wcs(
                server.url, coverage="test_cov", bbox=(2.0, 2.0, 4.0, 4.0), version="2.0.1"
            )
            arr = ds.read_array()
            assert arr.shape == (20, 20)
            assert arr[0, 0] == 0 and arr[1, 1] == 2  # synthetic (row+col) pattern

    def test_output_writes_a_reopenable_file(self, tmp_path):
        out = tmp_path / "wcs_out.tif"
        with WcsMock(version="1.0.0") as server:
            Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=(2.0, 2.0, 4.0, 4.0),
                version="1.0.0",
                output=out,
            )
        assert out.exists()
        assert Dataset.read_file(str(out)).shape == (1, 20, 20)

    def test_exception_report_on_getcoverage_raises_and_writes_nothing(self, tmp_path):
        out = tmp_path / "should_not_exist.tif"
        with WcsMock(version="2.0.1", getcoverage_body=EXCEPTION_REPORT) as server:
            with pytest.raises(WCSError):
                Dataset.from_wcs(
                    server.url,
                    coverage="test_cov",
                    bbox=(2.0, 2.0, 4.0, 4.0),
                    version="2.0.1",
                    output=out,
                )
        assert not out.exists()

    def test_resolution_without_output_crs_resamples_in_native(self):
        """A resolution with no output_crs resamples within the native CRS to a coarser grid."""
        with WcsMock(version="1.0.0") as server:
            ds = Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=(2.0, 2.0, 4.0, 4.0),
                version="1.0.0",
                resolution=0.2,
            )
            assert ds.epsg == 4326
            assert ds.shape[1] < 20 and ds.shape[2] < 20  # coarser than the 0.1deg native 20x20

    def test_output_crs_reprojects_offline(self):
        """output_crs reprojects the fetched window into the requested CRS."""
        with WcsMock(version="1.0.0") as server:
            ds = Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=(2.0, 2.0, 4.0, 4.0),
                version="1.0.0",
                output_crs="EPSG:3857",
            )
            assert ds.epsg == 3857


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("PYRAMIDS_WCS_LIVE"),
    reason="live network test; set PYRAMIDS_WCS_LIVE=1 to run",
)
class TestLiveSoilGrids:
    ENDPOINT = "https://maps.isric.org/mapserv?map=/map/nitrogen.map"
    IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"

    def test_native_read(self):
        ds = Dataset.from_wcs(
            self.ENDPOINT,
            coverage="nitrogen_0-5cm_mean",
            bbox=(5.0, 51.0, 6.0, 52.0),
            coverage_crs=self.IGH,
        )
        assert ds.shape[0] == 1
        assert "igh" in ds.raster.GetSpatialRef().ExportToProj4()

    def test_output_crs_reproject(self):
        ds = Dataset.from_wcs(
            self.ENDPOINT,
            coverage="nitrogen_0-5cm_mean",
            bbox=(5.0, 51.0, 6.0, 52.0),
            coverage_crs=self.IGH,
            output_crs="EPSG:4326",
        )
        assert ds.epsg == 4326
