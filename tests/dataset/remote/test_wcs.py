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


import copy
import io
import pickle
import urllib.error

import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset
from pyramids.dataset import _wcs
from pyramids.errors import WCSError
from tests.dataset.remote.wcs_mock_server import WcsMock
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

    @pytest.mark.parametrize("bad", [0, -1, (0, 10), (10, -5)])
    def test_resolution_pair_rejects_non_positive(self, bad):
        with pytest.raises(ValueError, match="strictly positive"):
            _wcs._resolution_pair(bad)

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

    def test_get_capabilities_http_error_propagates_status_and_body(self, monkeypatch):
        """A 4xx GetCapabilities carries status_code/response_body through the real call site."""
        body = b'{"message": "forbidden zone", "code": "NOPE"}'

        def boom(self, *args, **kwargs):
            raise urllib.error.HTTPError(
                "https://wcs.invalid/caps-403", 403, "Forbidden", {}, io.BytesIO(body)
            )

        monkeypatch.setattr(_wcs.urllib.request.OpenerDirector, "open", boom)
        with pytest.raises(WCSError) as excinfo:
            _wcs._get_capabilities("https://wcs.invalid/caps-403", None, None, 5.0)
        err = excinfo.value
        assert err.status_code == 403
        assert "forbidden zone" in err.response_body
        assert "forbidden zone" in str(err)

    def test_http_error_surfaces_body_and_attributes(self, monkeypatch):
        """A 4xx GetCoverage surfaces the server body text and carries status + body on WCSError."""
        body = (
            b'{"message": "Requested date 2035-06-11 is outside the available '
            b'coverage range for SPI ERA5 Short Term.", "code": "DATE_OUT_OF_RANGE"}'
        )

        def boom(self, *args, **kwargs):
            raise urllib.error.HTTPError(
                "https://wcs.invalid/x", 422, "Unprocessable Entity", {}, io.BytesIO(body)
            )

        monkeypatch.setattr(_wcs.urllib.request.OpenerDirector, "open", boom)
        with pytest.raises(WCSError) as excinfo:
            _wcs._http_get("https://wcs.invalid/x", None, 5.0, "GetCoverage")
        err = excinfo.value
        assert "outside the available coverage range" in str(err)
        assert "HTTP 422" in str(err)
        assert err.status_code == 422
        assert "DATE_OUT_OF_RANGE" in err.response_body

    def test_http_error_body_truncated_in_message_but_full_on_attribute(self, monkeypatch):
        """A large error body is truncated in the message yet kept whole on response_body."""
        body = b"x" * 5000

        def boom(self, *args, **kwargs):
            raise urllib.error.HTTPError(
                "https://wcs.invalid/x", 500, "Server Error", {}, io.BytesIO(body)
            )

        monkeypatch.setattr(_wcs.urllib.request.OpenerDirector, "open", boom)
        with pytest.raises(WCSError) as excinfo:
            _wcs._http_get("https://wcs.invalid/x", None, 5.0, "GetCapabilities")
        err = excinfo.value
        assert "…" in str(err)  # truncation marker present
        assert len(str(err)) < 1000  # message is bounded
        assert len(err.response_body) == 5000  # full body preserved on the attribute

    def test_non_http_transport_error_leaves_attributes_none(self, monkeypatch):
        """A non-HTTP transport failure keeps its message and leaves status_code/body None."""

        def boom(self, *args, **kwargs):
            raise urllib.error.URLError("connection reset")

        monkeypatch.setattr(_wcs.urllib.request.OpenerDirector, "open", boom)
        with pytest.raises(WCSError) as excinfo:
            _wcs._http_get("https://wcs.invalid/x", None, 5.0, "GetCoverage")
        err = excinfo.value
        assert "request failed" in str(err)
        assert err.status_code is None
        assert err.response_body is None

    def test_wcserror_survives_pickle_and_copy(self):
        """status_code / response_body round-trip through pickle and copy (cross-process safe)."""
        err = WCSError("boom", status_code=422, response_body='{"code": "X"}')
        restored = pickle.loads(pickle.dumps(err))
        assert str(restored) == "boom"
        assert restored.status_code == 422
        assert restored.response_body == '{"code": "X"}'
        cloned = copy.copy(err)
        assert cloned.status_code == 422
        assert cloned.response_body == '{"code": "X"}'

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
@pytest.mark.live
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


class TestDefaultSubsetAxes:
    def test_geographic_crs_gets_long_lat(self):
        assert _wcs._default_subset_axes("EPSG:4326") == ("Long", "Lat")

    def test_projected_crs_gets_x_y(self):
        assert _wcs._default_subset_axes("EPSG:3857") == ("X", "Y")

    def test_unparseable_crs_falls_back_without_raising(self):
        # a garbage CRS must not raise; it falls back to the non-geographic default
        assert _wcs._default_subset_axes("not-a-real-crs") == ("X", "Y")


class TestGetCoverageUrl:
    """The direct-mode KVP GetCoverage URL builder (no network)."""

    def test_2_0_x_builds_coverageid_subset_and_subsettingcrs(self):
        url = _wcs._getcoverage_url(
            "https://x/mapserv?map=GDO_WCS", "spaST", "EPSG:4326",
            (-10.0, 35.0, 5.0, 45.0), "2.0.0", "GEOTIFF", None, None,
            {"TIME": "2024-06-01"},
        )
        assert "REQUEST=GetCoverage" in url and "COVERAGEID=spaST" in url
        assert "SUBSET=Long(-10.0,5.0)" in url  # lon range on the Long axis
        assert "SUBSET=Lat(35.0,45.0)" in url  # lat range on the Lat axis
        assert "SUBSETTINGCRS=EPSG:4326" in url  # colon kept literal for shims
        assert "FORMAT=GEOTIFF" in url and "TIME=2024-06-01" in url

    def test_subset_axes_override_replaces_default_labels(self):
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:4326", (0.0, 1.0, 2.0, 3.0), "2.0.1",
            None, None, ("x", "y"), None,
        )
        assert "SUBSET=x(0.0,2.0)" in url and "SUBSET=y(1.0,3.0)" in url

    def test_1_0_0_builds_bbox_and_resx_resy(self):
        url = _wcs._getcoverage_url(
            "https://x/wcs", "c", "EPSG:4326", (-10.0, 35.0, 5.0, 45.0),
            "1.0.0", "GEOTIFF", 0.1, None, None,
        )
        assert "COVERAGE=c" in url and "BBOX=-10.0,35.0,5.0,45.0" in url
        assert "RESX=0.1" in url and "RESY=0.1" in url

    def test_1_0_0_without_resolution_raises_valueerror(self):
        with pytest.raises(ValueError, match="resolution"):
            _wcs._getcoverage_url(
                "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "1.0.0",
                None, None, None, None,
            )

    def test_unsupported_version_raises_valueerror(self):
        with pytest.raises(ValueError, match="1.0.0 and 2.0.x"):
            _wcs._getcoverage_url(
                "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "1.1.0",
                None, None, None, None,
            )

    def test_malformed_version_raises_valueerror(self):
        with pytest.raises(ValueError, match="x.y.z"):
            _wcs._getcoverage_url(
                "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "2.0",
                None, None, None, None,
            )

    def test_projected_crs_uses_x_y_subset_labels(self):
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:3857", (0.0, 1.0, 2.0, 3.0), "2.0.0",
            None, None, None, None,
        )
        assert "SUBSET=X(0.0,2.0)" in url and "SUBSET=Y(1.0,3.0)" in url

    def test_extra_params_value_with_space_is_encoded(self):
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "2.0.0",
            None, None, None, {"TIME": "2024-06-01 00:00"},
        )
        assert "TIME=2024-06-01%2000" in url  # the space is percent-encoded

    def test_extra_params_key_with_ampersand_is_encoded(self):
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "2.0.0",
            None, None, None, {"a&b": "v"},
        )
        assert "a%26b=v" in url  # '&' in the key cannot split the query

    @pytest.mark.parametrize("key", ["VERSION", "SERVICE", "REQUEST", "SUBSET"])
    def test_extra_params_protocol_key_raises(self, key):
        with pytest.raises(ValueError, match="fixed WCS protocol parameter"):
            _wcs._getcoverage_url(
                "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "2.0.0",
                None, None, None, {key: "x"},
            )

    def test_extra_params_lowercase_coverageid_overrides_builtin(self):
        # A shim that is case-sensitive on the key (Copernicus EDO/GDO) needs the
        # lowercase spelling; the override replaces COVERAGEID, not duplicates it.
        url = _wcs._getcoverage_url(
            "https://x", "spaST", "EPSG:4326", (10.0, 45.0, 15.0, 48.0), "2.0.0",
            "GEOTIFF", None, None, {"coverageID": "spaST"},
        )
        assert "coverageID=spaST" in url
        assert "COVERAGEID=spaST" not in url

    def test_extra_params_crs_overrides_subsettingcrs(self):
        # WCS-1.x CRS= on a 2.0 request: the override replaces SUBSETTINGCRS and no
        # SUBSETTINGCRS token survives (the two share one CRS slot).
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:4326", (10.0, 45.0, 15.0, 48.0), "2.0.0",
            None, None, None, {"CRS": "EPSG:4326"},
        )
        assert "&CRS=EPSG:4326" in url
        assert "SUBSETTINGCRS" not in url

    def test_extra_params_edo_dialect_reproduces_shim_request(self):
        # Both quirks together must reproduce the exact request EDO/GDO answers 200:
        # lowercase coverageID + CRS= + WCS-2.0 SUBSET syntax, extras preserved.
        url = _wcs._getcoverage_url(
            "https://drought.emergency.copernicus.eu/api/wcs?map=DO_WCS",
            "spaST", "EPSG:4326", (10.0, 45.0, 15.0, 48.0), "2.0.0", "GEOTIFF",
            None, None,
            {"coverageID": "spaST", "CRS": "EPSG:4326",
             "TIME": "2023-06-01", "SELECTED_TIMESCALE": "01"},
        )
        assert "coverageID=spaST" in url and "COVERAGEID=spaST" not in url
        assert "&CRS=EPSG:4326" in url and "SUBSETTINGCRS" not in url
        assert "SUBSET=Long(10.0,15.0)" in url and "SUBSET=Lat(45.0,48.0)" in url
        assert "TIME=2023-06-01" in url and "SELECTED_TIMESCALE=01" in url

    def test_extra_params_conflicting_slot_raises(self):
        # Two keys collapsing to one slot (CRS + SUBSETTINGCRS) is contradictory
        # input and must fail loud rather than silently drop one.
        with pytest.raises(ValueError, match="same GetCoverage parameter"):
            _wcs._getcoverage_url(
                "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "2.0.0",
                None, None, None,
                {"CRS": "EPSG:4326", "SUBSETTINGCRS": "EPSG:3857"},
            )

    def test_1_0_0_crs_override_replaces_not_duplicates(self):
        # On a 1.0.0 request the built-in already emits CRS; an override must
        # replace it in place, leaving exactly one CRS token.
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "1.0.0",
            None, 0.1, None, {"CRS": "EPSG:3857"},
        )
        assert url.count("CRS=") == 1 and "CRS=EPSG:3857" in url

    def test_1_0_0_coverageid_override_replaces_coverage(self):
        # The coverage id spelling collapses across versions: a 2.0 `coverageID`
        # override replaces the 1.0.0 built-in `COVERAGE`, not appends beside it.
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "1.0.0",
            None, 0.1, None, {"coverageID": "c"},
        )
        assert "coverageID=c" in url and "COVERAGE=c" not in url

    def test_extra_params_appended_keep_caller_order(self):
        # Non-matching extras (and overridable-but-unmatched keys) append after the
        # built-ins in caller order; regression guard for the ordering guarantee.
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "2.0.0",
            None, None, None, {"BB": "2", "coverageID": "c", "AA": "1"},
        )
        tail = url.split("REQUEST=GetCoverage", 1)[1]
        assert tail.index("BB=2") < tail.index("AA=1")  # caller order preserved
        assert tail.index("coverageID=c") < tail.index("BB=2")  # built-ins first

    def test_extra_params_non_matching_key_is_appended(self):
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "2.0.0",
            None, None, None, {"SELECTED_TIMESCALE": "03"},
        )
        assert "SELECTED_TIMESCALE=03" in url
        assert "COVERAGEID=c" in url  # built-ins untouched

    def test_extra_params_override_case_insensitive(self):
        # A mixed-case override key still matches the built-in it targets ('/' is kept
        # literal in values, like CRS URIs, so the MIME type is not percent-encoded).
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "2.0.0",
            "GEOTIFF", None, None, {"Format": "image/tiff"},
        )
        assert "Format=image/tiff" in url
        assert "FORMAT=GEOTIFF" not in url

    def test_extra_params_overridable_key_without_builtin_is_appended(self):
        # RESX is overridable but a 2.0 request emits no RESX built-in, so it must be
        # kept as a plain extra rather than silently dropped.
        url = _wcs._getcoverage_url(
            "https://x", "c", "EPSG:4326", (0.0, 0.0, 1.0, 1.0), "2.0.0",
            None, None, None, {"RESX": "0.1"},
        )
        assert "RESX=0.1" in url


@pytest.fixture
def geotiff_bytes():
    """Bytes of a tiny 4x3 EPSG:4326 GeoTIFF — stands in for a GetCoverage body."""
    path = "/vsimem/_wcs_direct_fixture.tif"
    src = gdal.GetDriverByName("GTiff").Create(path, 4, 3, 1, gdal.GDT_Byte)
    src.SetGeoTransform([-10.0, 3.75, 0.0, 45.0, 0.0, -10.0 / 3.0])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    src.SetProjection(srs.ExportToWkt())
    src.GetRasterBand(1).Fill(7)
    src.FlushCache()
    src = None
    stat = gdal.VSIStatL(path)
    handle = gdal.VSIFOpenL(path, "rb")
    data = gdal.VSIFReadL(1, stat.size, handle)
    gdal.VSIFCloseL(handle)
    gdal.Unlink(path)
    return bytes(data)


@pytest.fixture
def crsless_geotiff_bytes():
    """Bytes of a tiny GeoTIFF with a geotransform but NO CRS (a shim response)."""
    path = "/vsimem/_wcs_crsless_fixture.tif"
    src = gdal.GetDriverByName("GTiff").Create(path, 4, 3, 1, gdal.GDT_Byte)
    src.SetGeoTransform([-10.0, 3.75, 0.0, 45.0, 0.0, -10.0 / 3.0])
    src.GetRasterBand(1).Fill(7)
    src.FlushCache()
    src = None
    stat = gdal.VSIStatL(path)
    handle = gdal.VSIFOpenL(path, "rb")
    data = gdal.VSIFReadL(1, stat.size, handle)
    gdal.VSIFCloseL(handle)
    gdal.Unlink(path)
    return bytes(data)


class TestDirectGetCoverage:
    """The ``direct=True`` path: no capabilities/describe, KVP GetCoverage only."""

    ENDPOINT = "https://shim.invalid/mapserv?map=GDO_WCS"
    BBOX = (-10.0, 35.0, 5.0, 45.0)

    def test_direct_success_returns_dataset(self, monkeypatch, geotiff_bytes):
        monkeypatch.setattr(_wcs, "_http_get", lambda *a, **k: geotiff_bytes)
        ds = Dataset.from_wcs(
            self.ENDPOINT, coverage="spaST", bbox=self.BBOX, crs="EPSG:4326",
            version="2.0.0", wcs_format="GEOTIFF", direct=True,
        )
        assert ds.shape == (1, 3, 4)
        assert ds.epsg == 4326

    def test_direct_skips_capabilities(self, monkeypatch, geotiff_bytes):
        def boom(*a, **k):
            raise AssertionError("GetCapabilities must not be called in direct mode")

        monkeypatch.setattr(_wcs, "_get_capabilities", boom)
        monkeypatch.setattr(_wcs, "_http_get", lambda *a, **k: geotiff_bytes)
        ds = Dataset.from_wcs(
            self.ENDPOINT, coverage="anything", bbox=self.BBOX, direct=True,
        )
        assert ds.epsg == 4326

    def test_direct_exception_report_raises_wcserror(self, monkeypatch):
        monkeypatch.setattr(
            _wcs, "_http_get", lambda *a, **k: EXCEPTION_REPORT.encode("utf-8")
        )
        with pytest.raises(WCSError, match="exception"):
            Dataset.from_wcs(
                self.ENDPOINT, coverage="spaST", bbox=self.BBOX, direct=True,
            )

    def test_direct_non_raster_body_raises_wcserror(self, monkeypatch):
        monkeypatch.setattr(_wcs, "_http_get", lambda *a, **k: b"not a raster at all")
        with pytest.raises(WCSError, match="no raster|could not be read"):
            Dataset.from_wcs(
                self.ENDPOINT, coverage="c", bbox=(0.0, 0.0, 1.0, 1.0), direct=True,
            )

    def test_direct_coverage_crs_shim_sets_epsg_on_crsless_raster(
        self, monkeypatch, crsless_geotiff_bytes
    ):
        monkeypatch.setattr(_wcs, "_http_get", lambda *a, **k: crsless_geotiff_bytes)
        ds = Dataset.from_wcs(
            self.ENDPOINT, coverage="spaST", bbox=self.BBOX, direct=True,
            coverage_crs="EPSG:4326",
        )
        assert ds.epsg == 4326

    def test_direct_1_0_0_end_to_end_no_client_resample(self, monkeypatch, geotiff_bytes):
        # 1.0.0 sends RESX/RESY, so the server grids server-side and pyramids does
        # NOT resample again client-side. The mock returns the native-res fixture,
        # so the result is the fixture grid unchanged (not resampled to 1.0).
        monkeypatch.setattr(_wcs, "_http_get", lambda *a, **k: geotiff_bytes)
        ds = Dataset.from_wcs(
            self.ENDPOINT, coverage="c", bbox=self.BBOX, crs="EPSG:4326",
            version="1.0.0", resolution=1.0, direct=True,
        )
        assert ds.epsg == 4326
        assert ds.shape == (1, 3, 4)  # native fixture grid, no client resample

    def test_direct_forwards_params_into_the_url(self, monkeypatch, geotiff_bytes):
        captured: dict[str, str] = {}

        def capture(url, *a, **k):
            captured["url"] = url
            return geotiff_bytes

        monkeypatch.setattr(_wcs, "_http_get", capture)
        Dataset.from_wcs(
            self.ENDPOINT, coverage="spaST", bbox=self.BBOX, crs="EPSG:4326",
            version="2.0.1", direct=True, subset_axes=("x", "y"),
            extra_params={"TIME": "2024-06-01"},
        )
        url = captured["url"]
        assert "COVERAGEID=spaST" in url and "VERSION=2.0.1" in url
        assert "SUBSET=x(-10.0,5.0)" in url and "SUBSET=y(35.0,45.0)" in url
        assert "TIME=2024-06-01" in url

    def test_direct_output_crs_without_crs_raises(
        self, monkeypatch, crsless_geotiff_bytes
    ):
        monkeypatch.setattr(_wcs, "_http_get", lambda *a, **k: crsless_geotiff_bytes)
        with pytest.raises(WCSError, match="no CRS"):
            Dataset.from_wcs(
                self.ENDPOINT, coverage="c", bbox=self.BBOX,
                output_crs="EPSG:3857", direct=True,
            )

    def test_direct_resolution_resamples_output(self, monkeypatch, geotiff_bytes):
        monkeypatch.setattr(_wcs, "_http_get", lambda *a, **k: geotiff_bytes)
        ds = Dataset.from_wcs(
            self.ENDPOINT, coverage="c", bbox=self.BBOX, version="2.0.0",
            resolution=1.0, direct=True,
        )
        assert ds.cell_size == pytest.approx(1.0, abs=0.01)

    def test_direct_resolution_without_crs_raises(
        self, monkeypatch, crsless_geotiff_bytes
    ):
        monkeypatch.setattr(_wcs, "_http_get", lambda *a, **k: crsless_geotiff_bytes)
        with pytest.raises(WCSError, match="no CRS"):
            Dataset.from_wcs(
                self.ENDPOINT, coverage="c", bbox=self.BBOX, resolution=1.0,
                direct=True,
            )

    @pytest.mark.parametrize(
        ("dialect", "extra_params"),
        [
            pytest.param("compliant", None, id="compliant-COVERAGEID-SUBSETTINGCRS"),
            pytest.param(
                "edo",
                {"coverageID": "spaST", "CRS": "EPSG:4326"},
                id="edo-coverageID-CRS",
            ),
        ],
    )
    def test_direct_serves_both_wcs_dialects(
        self, monkeypatch, geotiff_bytes, dialect, extra_params
    ):
        # A fake server that returns a raster only for its own KVP dialect (else an
        # exception-report body -> WCSError). direct=True must retrieve a raster from
        # each: the spec spellings with no override, and the EDO shim spellings via
        # extra_params. Proves the override reproduces the shim dialect without
        # regressing the compliant path.
        monkeypatch.setattr(
            _wcs, "_http_get",
            lambda url, *a, **k: _dialect_body(url, geotiff_bytes, dialect),
        )
        ds = Dataset.from_wcs(
            self.ENDPOINT, coverage="spaST", bbox=self.BBOX, crs="EPSG:4326",
            version="2.0.0", wcs_format="GEOTIFF", direct=True,
            extra_params=extra_params,
        )
        assert ds.shape == (1, 3, 4)
        assert ds.epsg == 4326


def _dialect_body(url: str, raster: bytes, dialect: str) -> bytes:
    """Fake GetCoverage body: the raster iff ``url`` matches ``dialect``, else an error.

    Models the two WCS KVP dialects a direct request can target — the spec-compliant
    ``COVERAGEID`` + ``SUBSETTINGCRS`` and the Copernicus EDO/GDO shim's lowercase
    ``coverageID`` + WCS-1.x ``CRS=`` — returning an ``<ows:ExceptionReport>`` (which
    surfaces as :class:`WCSError`) when the request does not match the served dialect.
    """
    if dialect == "edo":
        matched = (
            "coverageID=" in url
            and "COVERAGEID=" not in url
            and "&CRS=" in url
            and "SUBSETTINGCRS" not in url
        )
    else:
        matched = "COVERAGEID=" in url and "SUBSETTINGCRS=" in url
    return raster if matched else EXCEPTION_REPORT.encode("utf-8")


@pytest.mark.slow
@pytest.mark.live
class TestLiveEdoDirect:
    """Live direct GetCoverage against the Copernicus EDO/GDO WCS shim (#713).

    The shim ``500``s on the spec ``COVERAGEID`` / ``SUBSETTINGCRS`` spellings, so
    the request overrides both via ``extra_params`` to the lowercase ``coverageID``
    and the WCS-1.x ``CRS=`` it accepts. ``TIME=2023-06-01`` is a published EDO dekad.
    """

    def test_direct_edo_override_returns_reference_raster(self):
        ds = Dataset.from_wcs(
            "https://drought.emergency.copernicus.eu/api/wcs?map=DO_WCS",
            coverage="spaST",
            bbox=(10.0, 45.0, 15.0, 48.0),
            crs="EPSG:4326",
            version="2.0.0",
            wcs_format="GEOTIFF",
            direct=True,
            extra_params={
                "coverageID": "spaST",
                "CRS": "EPSG:4326",
                "TIME": "2023-06-01",
                "SELECTED_TIMESCALE": "01",
            },
            timeout=45,
        )
        assert ds.shape == (1, 12, 1440)
        assert ds.geotransform == pytest.approx((-180.0, 0.25, 0.0, 48.0, 0.0, -0.25))
        assert ds.epsg == 4326
        arr = ds.read_array()
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 3.0
