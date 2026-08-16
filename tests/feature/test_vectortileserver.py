"""Unit tests for ``FeatureCollection.from_vectortileserver`` (ArcGIS VectorTileServer reader).

The reader is exercised offline against a tiny recorded fixture tileset
(``tests/data/vectortileserver/``): a couple of real gzip-compressed MVT ``.pbf``
tiles plus a VectorTileServer-shaped ``metadata.json``. The two network seams
(``_fetch_vectortileserver_metadata`` / ``_fetch_vectortileserver_tile``) are
monkeypatched to serve those fixtures, so no live endpoint is needed. One live
smoke test is marked ``live`` and skipped by the default ``-m "not live"`` run.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import warnings
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base._errors import VectorTileServerError
from pyramids.feature import _read
from pyramids.feature.collection import FeatureCollection

_DATA = Path(__file__).resolve().parents[1] / "data" / "vectortileserver"


def _load_metadata() -> dict:
    return json.loads((_DATA / "metadata.json").read_text())


def _load_fixture_info() -> dict:
    return json.loads((_DATA / "fixture_info.json").read_text())


def _tile_bytes_for_url(tile_url: str) -> bytes | None:
    """Map a ``.../tile/{z}/{y}/{x}.pbf`` URL to the fixture tile bytes (``None`` if absent)."""
    z, y, x = (part for part in tile_url.rsplit("/", 3)[1:])
    path = _DATA / "tiles" / z / x.removesuffix(".pbf") / f"{y}.pbf"
    return path.read_bytes() if path.exists() else None


@pytest.fixture
def served(monkeypatch):
    """Serve the fixture tileset through the two VectorTileServer fetch seams."""
    metadata = _load_metadata()
    monkeypatch.setattr(
        FeatureCollection,
        "_fetch_vectortileserver_metadata",
        classmethod(lambda cls, url, auth, timeout: metadata),
    )
    monkeypatch.setattr(
        FeatureCollection,
        "_fetch_vectortileserver_tile",
        classmethod(lambda cls, tile_url, auth, timeout: _tile_bytes_for_url(tile_url)),
    )
    return _load_fixture_info()


class TestFromVectorTileServer:
    """End-to-end reads against the fixture tileset."""

    def test_reads_features_across_both_tiles(self, served):
        """A bbox spanning two tiles returns the polygon (clipped in each) and the point."""
        fc = FeatureCollection.from_vectortileserver(
            "https://host/VectorTileServer",
            bbox=tuple(served["bbox_4326"]),
            zoom=served["zoom"],
        )
        assert isinstance(fc, FeatureCollection), "should return a FeatureCollection"
        assert str(fc.crs) == "EPSG:3857", (
            f"native tile CRS should be 3857, got {fc.crs}"
        )
        assert len(fc) == 3, (
            f"polygon clipped into 2 tiles + 1 point = 3 rows, got {len(fc)}"
        )
        kinds = list(fc["kind"])
        assert kinds.count("polygon") == 2 and kinds.count("point") == 1, kinds
        assert set(fc["layer"]) == {served["layer_name"]}, (
            "the source sub-layer is tagged"
        )
        expected = _read._vts_bbox_3857(tuple(served["bbox_4326"]), {})
        lods = _read._resolve_vts_tiling(_load_metadata())[3]
        pad = 512 * lods[served["zoom"]]  # one tile span of slack for MVT edge clipping
        minx, miny, maxx, maxy = fc.total_bounds
        assert expected[0] - pad <= minx and maxx <= expected[2] + pad, (
            minx,
            maxx,
            expected,
        )
        assert expected[1] - pad <= miny and maxy <= expected[3] + pad, (
            miny,
            maxy,
            expected,
        )

    def test_output_crs_reprojects(self, served):
        """``output_crs`` reprojects the result off the native 3857 tiles."""
        fc = FeatureCollection.from_vectortileserver(
            "https://host/VectorTileServer",
            bbox=tuple(served["bbox_4326"]),
            zoom=served["zoom"],
            output_crs="EPSG:4326",
        )
        assert fc.crs is not None and fc.crs.to_epsg() == 4326, (
            f"expected EPSG:4326, got {fc.crs}"
        )

    def test_layer_filter_selects_sublayer(self, served):
        """A matching ``layer`` reads it; a missing one yields an empty collection."""
        hit = FeatureCollection.from_vectortileserver(
            "https://host/VectorTileServer",
            bbox=tuple(served["bbox_4326"]),
            zoom=served["zoom"],
            layer=served["layer_name"],
        )
        assert len(hit) == 3, (
            "the real sub-layer reads all 3 rows (2 polygon pieces + point)"
        )
        miss = FeatureCollection.from_vectortileserver(
            "https://host/VectorTileServer",
            bbox=tuple(served["bbox_4326"]),
            zoom=served["zoom"],
            layer="does-not-exist",
        )
        assert len(miss) == 0, "an unknown sub-layer yields no features"

    def test_zoom_none_autopicks_highest_lod(self, served):
        """``zoom=None`` picks the highest advertised LOD that fits ``max_tiles``."""
        fc = FeatureCollection.from_vectortileserver(
            "https://host/VectorTileServer", bbox=tuple(served["bbox_4326"])
        )
        assert len(fc) == 3, "auto-picked zoom covers the fixture tiles (3 rows)"

    def test_bbox_none_reads_service_full_extent(self, served):
        """``bbox=None`` falls back to the service ``fullExtent`` and still reads the tiles."""
        fc = FeatureCollection.from_vectortileserver("https://host/VectorTileServer")
        assert len(fc) == 3, "the fullExtent fallback covers the fixture tiles (3 rows)"

    def test_max_tiles_cap_warns_and_truncates(self, served):
        """Exceeding ``max_tiles`` emits a UserWarning and reads only the capped count."""
        with pytest.warns(UserWarning, match="max_tiles"):
            fc = FeatureCollection.from_vectortileserver(
                "https://host/VectorTileServer",
                bbox=tuple(served["bbox_4326"]),
                zoom=served["zoom"],
                max_tiles=1,
            )
        assert len(fc) == 1, "only the first of the two covering tiles is read"

    def test_tile_urls_carry_an_existing_query(self, monkeypatch):
        """A ``?token=…`` on the service URL rides on every tile request."""
        metadata, info = _load_metadata(), _load_fixture_info()
        monkeypatch.setattr(
            FeatureCollection,
            "_fetch_vectortileserver_metadata",
            classmethod(lambda cls, url, auth, timeout: metadata),
        )
        seen: list[str] = []

        def _tile(cls, tile_url, auth, timeout):
            seen.append(tile_url)
            return _tile_bytes_for_url(tile_url.split("?")[0])

        monkeypatch.setattr(
            FeatureCollection, "_fetch_vectortileserver_tile", classmethod(_tile)
        )
        FeatureCollection.from_vectortileserver(
            "https://h/VectorTileServer?token=abc",
            bbox=tuple(info["bbox_4326"]),
            zoom=info["zoom"],
        )
        assert seen and all(url.endswith("?token=abc") for url in seen), seen

    def test_absent_tile_is_skipped(self, monkeypatch):
        """A covering tile that comes back absent (``None``) is skipped, not fatal."""
        metadata, info = _load_metadata(), _load_fixture_info()
        monkeypatch.setattr(
            FeatureCollection,
            "_fetch_vectortileserver_metadata",
            classmethod(lambda cls, url, auth, timeout: metadata),
        )
        calls = {"n": 0}

        def _tile(cls, tile_url, auth, timeout):
            calls["n"] += 1
            return _tile_bytes_for_url(tile_url) if calls["n"] == 1 else None

        monkeypatch.setattr(
            FeatureCollection, "_fetch_vectortileserver_tile", classmethod(_tile)
        )
        fc = FeatureCollection.from_vectortileserver(
            "https://host/VectorTileServer",
            bbox=tuple(info["bbox_4326"]),
            zoom=info["zoom"],
        )
        assert calls["n"] == 2, "both covering tiles are requested"
        assert len(fc) == 1, "only the present tile's single polygon fragment survives"

    def test_bad_zoom_raises_value_error(self, served):
        """A ``zoom`` the service does not advertise is a plain ValueError."""
        with pytest.raises(ValueError, match="not an advertised LOD"):
            FeatureCollection.from_vectortileserver(
                "https://host/VectorTileServer",
                bbox=tuple(served["bbox_4326"]),
                zoom=99,
            )


class TestVectorTileServerValidation:
    """Argument and metadata validation (no fixtures / network needed)."""

    def test_max_tiles_below_one_raises_before_any_fetch(self):
        """``max_tiles < 1`` is rejected before any network call."""
        with pytest.raises(ValueError, match="max_tiles must be >= 1"):
            FeatureCollection.from_vectortileserver(
                "https://host/VectorTileServer", max_tiles=0
            )

    def test_resolve_tiling_parses_metadata(self):
        """The tiling scheme is read out of ``tileInfo`` (origin, size, LODs, template)."""
        origin_x, origin_y, tile_size, lods, template = _read._resolve_vts_tiling(
            _load_metadata()
        )
        assert (round(origin_x), round(origin_y)) == (-20037508, 20037508)
        assert tile_size == 512
        assert 0 in lods and 10 in lods, "levels of detail should be parsed"
        assert template == "tile/{z}/{y}/{x}.pbf"

    def test_non_web_mercator_tiling_raises(self):
        """A non-Web-Mercator tiling CRS is rejected (GDAL MVT georeferencing needs 3857)."""
        meta = _load_metadata()
        meta["tileInfo"]["spatialReference"] = {"wkid": 4326, "latestWkid": 4326}
        with pytest.raises(ValueError, match="Web Mercator"):
            _read._resolve_vts_tiling(meta)

    def test_missing_lods_raises_service_error(self):
        """Metadata without any LODs is a VectorTileServerError, not a silent empty read."""
        meta = _load_metadata()
        meta["tileInfo"]["lods"] = []
        with pytest.raises(VectorTileServerError, match="no tileInfo.lods"):
            _read._resolve_vts_tiling(meta)

    def test_bbox_lonlat_to_3857_and_ordering(self):
        """A lon/lat bbox is reprojected to 3857; a bad ordering raises ValueError."""
        minx, miny, maxx, maxy = _read._vts_bbox_3857((-1.0, -1.0, 1.0, 1.0), {})
        assert minx < 0 < maxx and miny < 0 < maxy, (
            "bbox should straddle the 3857 origin"
        )
        with pytest.raises(ValueError, match="west < east"):
            _read._vts_bbox_3857((1.0, 1.0, -1.0, -1.0), {})

    def test_auth_is_sent_as_preemptive_basic_header(self):
        """``auth`` is encoded into a preemptive ``Authorization: Basic`` header."""
        request = _read._vts_request(
            "https://host/x", ("ada", "s3cret"), accept_json=True
        )
        expected = "Basic " + base64.b64encode(b"ada:s3cret").decode()
        assert request.get_header("Authorization") == expected, (
            "Basic auth should be preemptive"
        )

    def test_requests_use_a_consistent_user_agent(self):
        """Metadata and tile requests advertise the same VectorTileServer User-Agent."""
        meta_req = _read._vts_request("https://h/x?f=json", None, accept_json=True)
        tile_req = _read._vts_request(
            "https://h/tile/1/1/1.pbf", None, accept_json=False
        )
        ua = "pyramids-gis VectorTileServer client"
        assert meta_req.get_header("User-agent") == ua, meta_req.header_items()
        assert tile_req.get_header("User-agent") == ua, tile_req.header_items()
        assert meta_req.get_header("Accept") == "application/json", (
            "metadata negotiates JSON"
        )
        assert tile_req.get_header("Accept") is None, (
            "a tile request does not force JSON"
        )

    def test_bbox_none_uses_full_extent(self):
        """``bbox=None`` reads the read extent from the service ``fullExtent`` (already 3857)."""
        meta = {
            "fullExtent": {"xmin": -100.0, "ymin": -50.0, "xmax": 100.0, "ymax": 50.0}
        }
        assert _read._vts_bbox_3857(None, meta) == (-100.0, -50.0, 100.0, 50.0)

    def test_bbox_none_without_full_extent_raises(self):
        """``bbox=None`` and no usable ``fullExtent`` is a clear ValueError."""
        with pytest.raises(ValueError, match="no bbox given"):
            _read._vts_bbox_3857(None, {})

    def test_full_extent_in_4326_is_reprojected_to_3857(self):
        """A ``fullExtent`` reported in EPSG:4326 is reprojected, not read as metres."""
        meta = {
            "fullExtent": {
                "xmin": -1.0,
                "ymin": -1.0,
                "xmax": 1.0,
                "ymax": 1.0,
                "spatialReference": {"wkid": 4326, "latestWkid": 4326},
            }
        }
        minx, miny, maxx, maxy = _read._vts_bbox_3857(None, meta)
        assert abs(maxx - 111319.49) < 1.0, (
            f"1 deg lon should be ~111 km in 3857, got {maxx}"
        )
        assert minx < 0 < maxx and miny < 0 < maxy, (
            "the reprojected extent straddles the origin"
        )

    def test_full_extent_in_unknown_crs_raises_service_error(self):
        """A ``fullExtent`` in an unrecognised CRS is a VectorTileServerError, not misread metres."""
        meta = {
            "fullExtent": {
                "xmin": 0.0,
                "ymin": 0.0,
                "xmax": 1.0,
                "ymax": 1.0,
                "spatialReference": {"wkid": 999999, "latestWkid": 999999},
            }
        }
        with pytest.raises(VectorTileServerError, match="unrecognised CRS"):
            _read._vts_bbox_3857(None, meta)

    def test_pick_zoom_falls_back_to_coarsest_when_all_exceed_cap(self):
        """When every LOD exceeds ``max_tiles`` the coarsest advertised level is used."""
        lods = {2: 9784.0, 5: 1223.0}
        level = _read._pick_vts_zoom(
            (-1000.0, -1000.0, 1000.0, 1000.0),
            lods,
            -_read._WEBMERC_ORIGIN,
            _read._WEBMERC_ORIGIN,
            512,
            max_tiles=0,
        )
        assert level == 2, "the coarsest (minimum) LOD is the fallback"

    def test_assemble_drops_exact_duplicate_rows(self):
        """Byte-identical duplicate rows are dropped; distinct rows are kept."""
        a = gpd.GeoDataFrame(
            {"name": ["x"], "layer": ["places"]},
            geometry=[Point(0, 0)],
            crs="EPSG:3857",
        )
        dup = gpd.GeoDataFrame(
            {"name": ["x"], "layer": ["places"]},
            geometry=[Point(0, 0)],
            crs="EPSG:3857",
        )
        assert len(_read._assemble_vts_frames(FeatureCollection, [a, dup])) == 1, (
            "exact duplicate dropped"
        )
        other = gpd.GeoDataFrame(
            {"name": ["y"], "layer": ["places"]},
            geometry=[Point(1, 1)],
            crs="EPSG:3857",
        )
        assert len(_read._assemble_vts_frames(FeatureCollection, [a, other])) == 2, (
            "distinct rows kept"
        )

    def test_read_tile_frame_skips_empty_sublayer(self, tmp_path, monkeypatch):
        """A sub-layer that reads back with no features emits no frame."""
        tile_bytes = (_DATA / "tiles" / "10" / "511" / "512.pbf").read_bytes()
        monkeypatch.setattr(
            _read.gpd, "read_file", lambda *a, **k: gpd.GeoDataFrame(geometry=[])
        )
        frames = _read._read_vts_tile_frame(
            tile_bytes, 10, 511, 512, None, str(tmp_path)
        )
        assert frames == [], "an empty sub-layer yields no frame"

    def test_tile_count_matches_covering_list(self):
        """The count that drives zoom-pick / max_tiles equals the actual covering-tile list."""
        origin_x, origin_y, tile_size, lods, _ = _read._resolve_vts_tiling(
            _load_metadata()
        )
        info = _load_fixture_info()
        bbox_3857 = _read._vts_bbox_3857(tuple(info["bbox_4326"]), {})
        level = info["zoom"]
        tile_span = tile_size * lods[level]
        count = _read._vts_tile_count(bbox_3857, origin_x, origin_y, tile_span)
        tiles = _read._covering_vts_tiles(
            bbox_3857, level, origin_x, origin_y, tile_span, max_tiles=1000
        )
        assert count == len(tiles) == 2, "pick/cap count and fetched list must agree"

    def test_tile_range_clamped_to_valid_grid(self):
        """A world-spanning bbox clamps to the valid grid (no negative or overflow indices)."""
        origin = _read._WEBMERC_ORIGIN
        span = (2 * origin) / 8  # metres per tile for an 8x8 grid
        huge = (-3 * origin, -3 * origin, 3 * origin, 3 * origin)
        col_min, col_max, row_min, row_max = _read._vts_tile_range(
            huge, -origin, origin, span
        )
        assert (col_min, row_min) == (0, 0), "lower indices clamped to 0"
        assert (col_max, row_max) == (7, 7), "upper indices clamped to grid size - 1"

    def test_grid_dimension_derived_from_tile_span_not_level(self):
        """The grid size comes from ``tile_span`` (world / span), robust to non-canonical LOD numbering."""
        origin = _read._WEBMERC_ORIGIN
        span = (2 * origin) / 4  # a 4x4 grid — regardless of any advertised LOD integer
        huge = (-3 * origin, -3 * origin, 3 * origin, 3 * origin)
        _, col_max, _, row_max = _read._vts_tile_range(huge, -origin, origin, span)
        assert (col_max, row_max) == (3, 3), (
            "grid dimension (4) is derived from tile_span"
        )

    def test_base_and_query_splits_url(self):
        """A URL with a query is split into base + query so the token survives."""
        base, query = _read._vts_base_and_query("https://h/VectorTileServer/?token=abc")
        assert base == "https://h/VectorTileServer", (
            f"trailing slash + query stripped from base, got {base}"
        )
        assert query == "token=abc", f"the query is preserved, got {query}"

    def test_covering_tiles_matches_fixture(self):
        """The covering-tile math reproduces exactly the fixture's two tiles."""
        info = _load_fixture_info()
        origin_x, origin_y, tile_size, lods, _ = _read._resolve_vts_tiling(
            _load_metadata()
        )
        bbox_3857 = _read._vts_bbox_3857(tuple(info["bbox_4326"]), {})
        tile_span = tile_size * lods[info["zoom"]]
        tiles = _read._covering_vts_tiles(
            bbox_3857, info["zoom"], origin_x, origin_y, tile_span, max_tiles=1000
        )
        assert sorted(tuple(t) for t in tiles) == sorted(
            tuple(t) for t in info["tiles"]
        )


class TestVectorTileServerFetch:
    """The metadata/tile fetch seams surface service faults as VectorTileServerError."""

    def test_metadata_without_tileinfo_raises(self, monkeypatch):
        """A JSON body that is not a VectorTileServer (no ``tileInfo``) is rejected."""
        monkeypatch.setattr(
            _read, "http_get_with_retry", lambda request, timeout: b'{"foo": 1}'
        )
        with pytest.raises(
            VectorTileServerError, match="does not describe an ArcGIS VectorTileServer"
        ):
            _read.fetch_vectortileserver_metadata(
                "https://host/VectorTileServer", None, 30.0
            )

    def test_metadata_non_json_raises(self, monkeypatch):
        """A non-JSON metadata body is a VectorTileServerError."""
        monkeypatch.setattr(
            _read, "http_get_with_retry", lambda request, timeout: b"<html>nope</html>"
        )
        with pytest.raises(VectorTileServerError, match="non-JSON body"):
            _read.fetch_vectortileserver_metadata(
                "https://host/VectorTileServer", None, 30.0
            )

    def test_tile_404_returns_none(self, monkeypatch):
        """A 404 tile is treated as an empty cell (``None``), not an error."""

        def _raise(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)

        monkeypatch.setattr(_read, "http_get_with_retry", _raise)
        assert (
            _read.fetch_vectortileserver_tile(
                "https://host/tile/10/1/1.pbf", None, 30.0
            )
            is None
        )

    def test_metadata_success_returns_parsed_dict(self, monkeypatch):
        """A valid metadata body is parsed and returned as a dict."""
        body = json.dumps({"tileInfo": {"lods": []}, "name": "svc"}).encode()
        monkeypatch.setattr(_read, "http_get_with_retry", lambda request, timeout: body)
        doc = _read.fetch_vectortileserver_metadata(
            "https://host/VectorTileServer", None, 30.0
        )
        assert doc["name"] == "svc" and "tileInfo" in doc, (
            "the parsed service document is returned"
        )

    def test_metadata_url_preserves_existing_query(self, monkeypatch):
        """``?f=json`` is merged into an existing query rather than clobbering it."""
        seen = {}

        def _capture(request, timeout):
            seen["url"] = request.full_url
            return json.dumps({"tileInfo": {"lods": []}}).encode()

        monkeypatch.setattr(_read, "http_get_with_retry", _capture)
        _read.fetch_vectortileserver_metadata(
            "https://h/VectorTileServer?token=abc", None, 30.0
        )
        assert seen["url"] == "https://h/VectorTileServer?token=abc&f=json", seen["url"]

    def test_metadata_http_error_wraps_as_service_error(self, monkeypatch):
        """An HTTP error on the metadata fetch surfaces as VectorTileServerError."""

        def _raise(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, None)

        monkeypatch.setattr(_read, "http_get_with_retry", _raise)
        with pytest.raises(VectorTileServerError, match="HTTP 500"):
            _read.fetch_vectortileserver_metadata(
                "https://host/VectorTileServer", None, 30.0
            )

    def test_metadata_transport_error_wraps_as_service_error(self, monkeypatch):
        """A transport (OSError) failure on the metadata fetch surfaces as VectorTileServerError."""

        def _raise(request, timeout):
            raise OSError("connection reset")

        monkeypatch.setattr(_read, "http_get_with_retry", _raise)
        with pytest.raises(VectorTileServerError, match="metadata request failed"):
            _read.fetch_vectortileserver_metadata(
                "https://host/VectorTileServer", None, 30.0
            )

    def test_tile_non_404_http_error_wraps_as_service_error(self, monkeypatch):
        """A non-404 HTTP error on a tile fetch surfaces as VectorTileServerError (not None)."""

        def _raise(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

        monkeypatch.setattr(_read, "http_get_with_retry", _raise)
        with pytest.raises(VectorTileServerError, match="tile request failed"):
            _read.fetch_vectortileserver_tile(
                "https://host/tile/10/1/1.pbf", None, 30.0
            )

    def test_tile_transport_error_wraps_as_service_error(self, monkeypatch):
        """A transport (OSError) failure on a tile fetch surfaces as VectorTileServerError."""

        def _raise(request, timeout):
            raise OSError("connection reset")

        monkeypatch.setattr(_read, "http_get_with_retry", _raise)
        with pytest.raises(VectorTileServerError, match="tile request failed"):
            _read.fetch_vectortileserver_tile(
                "https://host/tile/10/1/1.pbf", None, 30.0
            )


@pytest.mark.live
def test_live_public_vectortileserver():  # pragma: no cover - network, opt-in via -m live
    """Smoke-read a small bbox from a public ArcGIS VectorTileServer (opt-in, network)."""
    url = "https://basemaps.arcgis.com/arcgis/rest/services/World_Basemap_v2/VectorTileServer"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = FeatureCollection.from_vectortileserver(
            url, bbox=(-122.45, 37.75, -122.40, 37.80), zoom=12, max_tiles=16
        )
    assert isinstance(fc, FeatureCollection)
    assert str(fc.crs) == "EPSG:3857"
