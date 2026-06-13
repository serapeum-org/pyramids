"""Tests for DatasetCollection.from_point / _stac.from_point grid math (PB-7)."""

from __future__ import annotations

import sys

import pytest

import pyramids.stac.search  # noqa: F401  (ensure the submodule is in sys.modules)
from pyramids.dataset import DatasetCollection
from pyramids.dataset._stac import _point_aoi_bbox, _utm_epsg, from_point

pytestmark = pytest.mark.core

# Real module object (the package exports a `search` function that shadows the
# `pyramids.stac.search` submodule attribute; see test_search.py).
_SEARCH_MOD = sys.modules["pyramids.stac.search"]
_STAC_MOD = sys.modules["pyramids.dataset._stac"]


class TestUtmEpsg:
    """Tests for local UTM zone selection."""

    @pytest.mark.parametrize(
        "lon, lat, expected",
        [
            (11.0, 46.0, 32632),  # Italian Alps, zone 32N
            (-76.2, 4.31, 32618),  # Colombia, zone 18N
            (-58.0, -34.0, 32721),  # Buenos Aires, zone 21S
            (-179.0, 0.0, 32601),  # zone 1N
            (179.0, -10.0, 32760),  # zone 60S
        ],
    )
    def test_zone_selection(self, lon, lat, expected):
        """Each lon/lat maps to the right UTM EPSG (326xx N / 327xx S).

        Args:
            lon: Longitude.
            lat: Latitude.
            expected: Expected UTM EPSG code.

        Test scenario:
            Northern points use 326xx, southern 327xx, with the right zone.
        """
        assert _utm_epsg(lon, lat) == expected, f"{lon},{lat} -> {_utm_epsg(lon, lat)}"


class TestPointAoiBbox:
    """Tests for the point -> UTM-snapped square -> 4326 bbox math."""

    def test_local_utm_selected(self):
        """The computed AOI uses the point's local UTM zone.

        Test scenario:
            (46N, 11E) selects EPSG:32632.
        """
        epsg, _ = _point_aoi_bbox(46.0, 11.0, edge_size=64, resolution=10.0, units="px")
        assert epsg == 32632, f"expected UTM 32632, got {epsg}"

    def test_bbox_brackets_the_point(self):
        """The 4326 AOI bbox contains the center point.

        Test scenario:
            (46N, 11E) lies within the returned (w, s, e, n).
        """
        _, (w, s, e, n) = _point_aoi_bbox(
            46.0, 11.0, edge_size=64, resolution=10.0, units="px"
        )
        assert w < 11.0 < e and s < 46.0 < n, f"point not bracketed by {(w, s, e, n)}"

    def test_edge_size_px_extent(self):
        """edge_size px * resolution sets the UTM square width (≈ in 4326).

        Test scenario:
            64 px * 10 m = 640 m square; in degrees its width is ~640/111320/cos(lat).
            Assert the lon span is within 10% of that expectation.
        """
        import math

        _, (w, s, e, n) = _point_aoi_bbox(
            46.0, 11.0, edge_size=64, resolution=10.0, units="px"
        )
        expected_deg = 640.0 / (111_320.0 * math.cos(math.radians(46.0)))
        assert (
            abs((e - w) - expected_deg) < 0.1 * expected_deg
        ), f"lon span {(e - w)} vs {expected_deg}"

    def test_units_metres(self):
        """units='m' uses edge_size directly as metres.

        Test scenario:
            1000 m square is wider than a 64 px * 10 m = 640 m square.
        """
        _, m_bbox = _point_aoi_bbox(
            0.0, 0.0, edge_size=1000, resolution=10.0, units="m"
        )
        _, px_bbox = _point_aoi_bbox(
            0.0, 0.0, edge_size=64, resolution=10.0, units="px"
        )
        assert (m_bbox[2] - m_bbox[0]) > (
            px_bbox[2] - px_bbox[0]
        ), "metres AOI should be wider"

    def test_invalid_units_raises(self):
        """An unsupported units value raises ValueError.

        Test scenario:
            units='furlongs' is rejected.
        """
        with pytest.raises(ValueError, match="units must be"):
            _point_aoi_bbox(0.0, 0.0, edge_size=10, resolution=10.0, units="furlongs")


class TestFromPoint:
    """Tests for from_point wiring (search + from_stac stubbed; no network)."""

    def test_composes_search_and_from_stac(self, monkeypatch):
        """from_point searches the AOI then stacks the bands.

        Test scenario:
            search receives the collection + a 4326 bbox bracketing the point +
            the datetime range; its items + bands flow into from_stac.
        """
        captured: dict = {}

        def fake_search(stac, collection, *, bbox, datetime, query, signer):
            captured["stac"] = stac
            captured["collection"] = collection
            captured["bbox"] = bbox
            captured["datetime"] = datetime
            return ["ITEM1", "ITEM2"]

        def fake_from_stac(items, asset, *, signer=None, align=True, **kw):
            captured["items"] = items
            captured["asset"] = asset
            return "CUBE"

        monkeypatch.setattr(_SEARCH_MOD, "search", fake_search)
        monkeypatch.setattr(_STAC_MOD, "from_stac", fake_from_stac)

        result = from_point(
            46.0,
            11.0,
            collection="sentinel-2-l2a",
            bands=["B04", "B03"],
            start_date="2021-06-01",
            end_date="2021-06-10",
            edge_size=64,
            resolution=10.0,
        )
        assert result == "CUBE", "from_point should return the from_stac result"
        assert (
            captured["collection"] == "sentinel-2-l2a"
        ), f"collection: {captured['collection']}"
        assert (
            captured["datetime"] == "2021-06-01/2021-06-10"
        ), f"datetime: {captured['datetime']}"
        w, s, e, n = captured["bbox"]
        assert (
            w < 11.0 < e and s < 46.0 < n
        ), f"AOI bbox should bracket the point: {captured['bbox']}"
        assert captured["items"] == [
            "ITEM1",
            "ITEM2",
        ], "searched items should flow into from_stac"
        assert captured["asset"] == [
            "B04",
            "B03",
        ], f"bands should pass through: {captured['asset']}"

    def test_classmethod_forwards(self, monkeypatch):
        """DatasetCollection.from_point forwards to the _stac implementation.

        Test scenario:
            The classmethod calls _from_point with lat/lon + kwargs.
        """
        seen: dict = {}

        def fake_from_point(lat, lon, **kwargs):
            seen["lat"] = lat
            seen["lon"] = lon
            seen["kwargs"] = kwargs
            return "CUBE"

        monkeypatch.setattr("pyramids.dataset.collection._from_point", fake_from_point)
        out = DatasetCollection.from_point(
            46.0,
            11.0,
            collection="c",
            bands="B04",
            start_date="2021-01-01",
            end_date="2021-01-31",
            edge_size=32,
            resolution=10.0,
        )
        assert out == "CUBE", "classmethod should return the forwarded result"
        assert (seen["lat"], seen["lon"]) == (
            46.0,
            11.0,
        ), f"lat/lon not forwarded: {seen}"
        assert (
            seen["kwargs"]["collection"] == "c"
        ), f"collection not forwarded: {seen['kwargs']}"
