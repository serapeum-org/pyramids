"""A protocol-faithful in-process OGC WCS server for driving GDAL's WCS driver.

Used by ``tests/dataset/test_wcs.py`` to exercise the full
``GetCapabilities → DescribeCoverage → GetCoverage`` cycle of
:meth:`pyramids.dataset.Dataset.from_wcs` offline — no live network. The mock
speaks both WCS dialects this matters for:

* **1.0.0** — ``GetCoverage`` uses ``BBOX`` + ``WIDTH``/``HEIGHT``,
* **2.0.1** — ``GetCoverage`` uses ``COVERAGEID`` + named-axis ``SUBSET``.

The synthetic coverage ``test_cov`` is a 100×100 grid at 0.1° over lon/lat
``[0, 10]`` (declared in CRS84 so GDAL builds a north-up geotransform). Each
``GetCoverage`` is answered with a freshly generated GeoTIFF for the requested
window, so the returned raster is real and openable.

Every request path is recorded on the handler class, so a test can assert which
dialect's ``GetCoverage`` shape GDAL emitted.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np
from osgeo import gdal, osr

COVERAGE = "test_cov"
_RES = 0.1

CAPS_100 = """<?xml version="1.0" encoding="UTF-8"?>
<WCS_Capabilities version="1.0.0" xmlns="http://www.opengis.net/wcs"
    xmlns:gml="http://www.opengis.net/gml" xmlns:xlink="http://www.w3.org/1999/xlink">
  <Service><name>mock</name><label>mock</label></Service>
  <Capability/>
  <ContentMetadata>
    <CoverageOfferingBrief>
      <name>test_cov</name>
      <label>Test Coverage</label>
      <lonLatEnvelope srsName="urn:ogc:def:crs:OGC:1.3:CRS84">
        <gml:pos>0 0</gml:pos>
        <gml:pos>10 10</gml:pos>
      </lonLatEnvelope>
    </CoverageOfferingBrief>
  </ContentMetadata>
</WCS_Capabilities>
"""

DESCRIBE_100 = """<?xml version="1.0" encoding="UTF-8"?>
<CoverageDescription version="1.0.0" xmlns="http://www.opengis.net/wcs"
    xmlns:gml="http://www.opengis.net/gml" xmlns:xlink="http://www.w3.org/1999/xlink">
  <CoverageOffering>
    <name>test_cov</name>
    <label>Test Coverage</label>
    <lonLatEnvelope srsName="urn:ogc:def:crs:OGC:1.3:CRS84">
      <gml:pos>0 0</gml:pos>
      <gml:pos>10 10</gml:pos>
    </lonLatEnvelope>
    <domainSet>
      <spatialDomain>
        <gml:Envelope srsName="EPSG:4326">
          <gml:pos>0 0</gml:pos>
          <gml:pos>10 10</gml:pos>
        </gml:Envelope>
        <gml:RectifiedGrid dimension="2">
          <gml:limits>
            <gml:GridEnvelope>
              <gml:low>0 0</gml:low>
              <gml:high>99 99</gml:high>
            </gml:GridEnvelope>
          </gml:limits>
          <gml:axisName>x</gml:axisName>
          <gml:axisName>y</gml:axisName>
          <gml:origin><gml:pos>0.05 9.95</gml:pos></gml:origin>
          <gml:offsetVector>0.1 0</gml:offsetVector>
          <gml:offsetVector>0 -0.1</gml:offsetVector>
        </gml:RectifiedGrid>
      </spatialDomain>
    </domainSet>
    <rangeSet>
      <RangeSet><name>bands</name><label>bands</label></RangeSet>
    </rangeSet>
    <supportedCRSs>
      <requestResponseCRSs>EPSG:4326</requestResponseCRSs>
      <nativeCRSs>EPSG:4326</nativeCRSs>
    </supportedCRSs>
    <supportedFormats nativeFormat="GeoTIFF">
      <formats>GeoTIFF</formats>
    </supportedFormats>
  </CoverageOffering>
</CoverageDescription>
"""

CAPS_201 = """<?xml version="1.0" encoding="UTF-8"?>
<wcs:Capabilities xmlns:wcs="http://www.opengis.net/wcs/2.0"
    xmlns:ows="http://www.opengis.net/ows/2.0" version="2.0.1">
  <ows:ServiceIdentification>
    <ows:ServiceTypeVersion>2.0.1</ows:ServiceTypeVersion>
  </ows:ServiceIdentification>
  <wcs:ServiceMetadata>
    <wcs:formatSupported>image/tiff</wcs:formatSupported>
  </wcs:ServiceMetadata>
  <wcs:Contents>
    <wcs:CoverageSummary>
      <wcs:CoverageId>test_cov</wcs:CoverageId>
      <wcs:CoverageSubtype>RectifiedGridCoverage</wcs:CoverageSubtype>
    </wcs:CoverageSummary>
  </wcs:Contents>
</wcs:Capabilities>
"""

DESCRIBE_201 = """<?xml version="1.0" encoding="UTF-8"?>
<wcs:CoverageDescriptions xmlns:wcs="http://www.opengis.net/wcs/2.0"
    xmlns:gml="http://www.opengis.net/gml/3.2"
    xmlns:gmlcov="http://www.opengis.net/gmlcov/1.0"
    xmlns:swe="http://www.opengis.net/swe/2.0">
  <wcs:CoverageDescription gml:id="test_cov">
    <gml:boundedBy>
      <gml:Envelope srsName="http://www.opengis.net/def/crs/OGC/1.3/CRS84"
          axisLabels="Long Lat" uomLabels="deg deg" srsDimension="2">
        <gml:lowerCorner>0 0</gml:lowerCorner>
        <gml:upperCorner>10 10</gml:upperCorner>
      </gml:Envelope>
    </gml:boundedBy>
    <wcs:CoverageId>test_cov</wcs:CoverageId>
    <gml:domainSet>
      <gml:RectifiedGrid dimension="2" gml:id="grid_test_cov">
        <gml:limits>
          <gml:GridEnvelope>
            <gml:low>0 0</gml:low>
            <gml:high>99 99</gml:high>
          </gml:GridEnvelope>
        </gml:limits>
        <gml:axisLabels>Long Lat</gml:axisLabels>
        <gml:origin>
          <gml:Point gml:id="p_test_cov" srsName="http://www.opengis.net/def/crs/OGC/1.3/CRS84">
            <gml:pos>0.05 9.95</gml:pos>
          </gml:Point>
        </gml:origin>
        <gml:offsetVector srsName="http://www.opengis.net/def/crs/OGC/1.3/CRS84">0.1 0</gml:offsetVector>
        <gml:offsetVector srsName="http://www.opengis.net/def/crs/OGC/1.3/CRS84">0 -0.1</gml:offsetVector>
      </gml:RectifiedGrid>
    </gml:domainSet>
    <gmlcov:rangeType>
      <swe:DataRecord>
        <swe:field name="band1">
          <swe:Quantity>
            <swe:uom code="unity"/>
            <swe:constraint><swe:AllowedValues>
              <swe:interval>-32768 32767</swe:interval>
            </swe:AllowedValues></swe:constraint>
          </swe:Quantity>
        </swe:field>
      </swe:DataRecord>
    </gmlcov:rangeType>
    <wcs:ServiceParameters>
      <wcs:CoverageSubtype>RectifiedGridCoverage</wcs:CoverageSubtype>
      <wcs:nativeFormat>image/tiff</wcs:nativeFormat>
    </wcs:ServiceParameters>
  </wcs:CoverageDescription>
</wcs:CoverageDescriptions>
"""

EXCEPTION_REPORT = """<?xml version="1.0" encoding="UTF-8"?>
<ows:ExceptionReport xmlns:ows="http://www.opengis.net/ows/2.0" version="2.0.1">
  <ows:Exception exceptionCode="NoApplicableCode">
    <ows:ExceptionText>coverage extraction failed.</ows:ExceptionText>
  </ows:Exception>
</ows:ExceptionReport>
"""

_CAPS = {"1.0.0": CAPS_100, "2.0.1": CAPS_201}
_DESCRIBE = {"1.0.0": DESCRIBE_100, "2.0.1": DESCRIBE_201}


def _make_geotiff(width: int, height: int, minx: float, maxy: float) -> bytes:
    """Generate a native-resolution GeoTIFF tile for a window; return its bytes."""
    path = "/vsimem/wcs_tile.tif"
    ds = gdal.GetDriverByName("GTiff").Create(path, width, height, 1, gdal.GDT_Int16)
    ds.SetGeoTransform((minx, _RES, 0, maxy, 0, -_RES))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.fromfunction(lambda r, c: (r + c).astype("int16"), (height, width)))
    ds.GetRasterBand(1).SetNoDataValue(-32768)
    ds.FlushCache()
    ds = None
    f = gdal.VSIFOpenL(path, "rb")
    gdal.VSIFSeekL(f, 0, 2)
    size = gdal.VSIFTellL(f)
    gdal.VSIFSeekL(f, 0, 0)
    data = gdal.VSIFReadL(1, size, f)
    gdal.VSIFCloseL(f)
    gdal.Unlink(path)
    return data


def _window_from_subsets(query: str) -> tuple[int, int, float, float]:
    """Derive (width, height, minx, maxy) from WCS 2.0.x named-axis SUBSETs."""
    raw = parse_qs(query)
    subsets = [v for k, vs in raw.items() if k.lower() == "subset" for v in vs]
    bounds: dict[str, tuple[float, float]] = {}
    for s in subsets:
        axis, rng = s.split("(")
        lo, hi = (float(v) for v in rng.rstrip(")").split(","))
        bounds[axis.lower()] = (lo, hi)
    lon = bounds.get("long", bounds.get("lon", (0.0, 10.0)))
    lat = bounds.get("lat", (0.0, 10.0))
    width = max(1, round((lon[1] - lon[0]) / _RES))
    height = max(1, round((lat[1] - lat[0]) / _RES))
    return width, height, lon[0], lat[1]


def _window_from_bbox(qs: dict[str, str]) -> tuple[int, int, float, float]:
    """Derive (width, height, minx, maxy) from a WCS 1.0.0 BBOX + WIDTH/HEIGHT."""
    minx, miny, maxx, maxy = (float(v) for v in qs["bbox"].split(","))
    width = int(qs.get("width", str(max(1, round((maxx - minx) / _RES)))))
    height = int(qs.get("height", str(max(1, round((maxy - miny) / _RES)))))
    return width, height, minx, maxy


def make_handler(version: str, getcoverage_body: str | None):
    """Build a request handler class for `version`, recording every request path.

    When `getcoverage_body` is given, ``GetCoverage`` returns that XML body with
    HTTP 200 instead of a raster — used to test ``<ows:ExceptionReport>`` handling.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        requests: list[str] = []

        def do_GET(self):  # noqa: N802
            type(self).requests.append(self.path)
            query = urlparse(self.path).query
            qs = {k.lower(): v[0] for k, v in parse_qs(query).items()}
            request = qs.get("request", "").lower()
            if request == "getcapabilities":
                self._send(_CAPS[version])
            elif request == "describecoverage":
                self._send(_DESCRIBE[version])
            elif request == "getcoverage":
                if getcoverage_body is not None:
                    self._send(getcoverage_body)
                    return
                if version == "1.0.0":
                    w, h, minx, maxy = _window_from_bbox(qs)
                else:
                    w, h, minx, maxy = _window_from_subsets(query)
                self._send_bytes(_make_geotiff(w, h, minx, maxy), "image/tiff")
            else:
                self.send_error(400, f"unknown request {request!r}")

        def _send(self, body: str, content_type: str = "application/xml"):
            self._send_bytes(body.encode("utf-8"), content_type)

        def _send_bytes(self, payload: bytes, content_type: str):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args, **kwargs):  # noqa: N802
            return

    Handler.requests = []
    return Handler


class WcsMock:
    """A running mock WCS server. Use as a context manager; `url` is the endpoint."""

    def __init__(self, version: str = "2.0.1", getcoverage_body: str | None = None):
        self._handler = make_handler(version, getcoverage_body)
        self._httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), self._handler)
        port = self._httpd.server_address[1]
        self.url = f"http://127.0.0.1:{port}/wcs"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def requests(self) -> list[str]:
        return self._handler.requests

    def getcoverage_requests(self) -> list[str]:
        return [r for r in self.requests if "getcoverage" in r.lower()]

    def __enter__(self) -> "WcsMock":
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
