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

import pytest

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
        assert len(fc) >= 2, f"expected features from both fixture tiles, got {len(fc)}"
        assert set(fc["kind"]) == {"polygon", "point"}, (
            "both source features should appear"
        )
        assert set(fc["layer"]) == {served["layer_name"]}, (
            "the source sub-layer is tagged"
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
        assert len(hit) >= 2, "the real sub-layer should read features"
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
        assert len(fc) >= 2, "auto-picked zoom should still cover the fixture tiles"

    def test_bbox_none_reads_service_full_extent(self, served):
        """``bbox=None`` falls back to the service ``fullExtent`` and still reads the tiles."""
        fc = FeatureCollection.from_vectortileserver("https://host/VectorTileServer")
        assert len(fc) >= 2, "the fullExtent fallback should cover the fixture tiles"

    def test_max_tiles_cap_warns_and_truncates(self, served):
        """Exceeding ``max_tiles`` emits a UserWarning and reads only the capped count."""
        with pytest.warns(UserWarning, match="max_tiles"):
            fc = FeatureCollection.from_vectortileserver(
                "https://host/VectorTileServer",
                bbox=tuple(served["bbox_4326"]),
                zoom=served["zoom"],
                max_tiles=1,
            )
        assert len(fc) >= 1, "the one read tile should still yield features"

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

    def test_read_tile_frame_skips_empty_sublayer(self, tmp_path, monkeypatch):
        """A sub-layer that reads back with no features emits no frame."""
        import geopandas as gpd

        tile_bytes = (_DATA / "tiles" / "10" / "511" / "512.pbf").read_bytes()
        monkeypatch.setattr(
            _read.gpd, "read_file", lambda *a, **k: gpd.GeoDataFrame(geometry=[])
        )
        frames = _read._read_vts_tile_frame(
            tile_bytes, 10, 511, 512, None, str(tmp_path)
        )
        assert frames == [], "an empty sub-layer yields no frame"

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
