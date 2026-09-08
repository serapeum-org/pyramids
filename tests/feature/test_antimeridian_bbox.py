"""A ``west > east`` bbox on the vector readers, and why ``fishnet`` still refuses one.

`Dataset.crop` has read a `west > east` bbox as a box crossing the 180 degree seam
for a while: it splits at the meridian, reads both sides and stitches them. The
vector readers refused the same box, from a second, hand-written copy of the
`minx < maxx` rule inside `pyramids.feature._ogc.read_kwargs`:

    ValueError: bbox must have minx < maxx and miny < maxy, got (170.0, -10.0, -170.0, 10.0)

The refusal fired before any request went out, so `from_wfs` / `from_ogc_features`
could not be pointed at Fiji, the Aleutians, Chukotka or the Chatham Islands at all.

What the split has to be is settled by where the filter goes. It is **not** applied
to a datasource pyramids holds: pyogrio turns `bbox=` into
`OGR_L_SetSpatialFilterRect`, and the `WFS` / `OAPIF` drivers turn that into a
request parameter. `TestWhatTheFilterActuallyIs` drives the real OAPIF driver
against a local recording stub to pin all three consequences:

* the bbox leaves as `…/items?limit=1000&bbox=170,-10,180,10`, so a wrap is two
  requests;
* a two-part `MultiPolygon` mask is sent as its *envelope* — the whole globe in
  longitude — so a multipart filter cannot be pushed into one request;
* a wrapping rect handed straight to OGR is silently normalised into its
  **complement**, which is the failure mode the old refusal was standing in front of.

`fishnet` keeps its refusal; `TestFishnetRefusesTheWrapOnPurpose` pins the reasoning
recorded in its docstring.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading

import geopandas as gpd
import pytest
from osgeo import gdal, ogr
from shapely.geometry import LineString, Point, box

from pyramids.base._coverage import seam_halves
from pyramids.errors import OGCAPIError, WFSError
from pyramids.feature import FeatureCollection, _oapif, _ogc, _wfs
from pyramids.feature import tessellation as _tess

WRAP = (170.0, -10.0, -170.0, 10.0)
"""A ten-degree-wide box straddling the antimeridian — the Fiji / Chatham case."""

EAST_HALF, WEST_HALF = seam_halves(WRAP)


def _world() -> gpd.GeoDataFrame:
    """Features either side of the seam, one of which straddles it.

    ``crosses`` is the duplicate hazard: its planar envelope spans -179..179, so it
    intersects both halves and comes back from both requests.
    """
    return gpd.GeoDataFrame(
        {"name": ["east", "on-180", "just-west", "far-west", "elsewhere", "crosses"]},
        geometry=[
            Point(172.0, 0.0),
            Point(180.0, 0.0),
            Point(-179.0, 0.0),
            Point(-172.0, 0.0),
            Point(0.0, 0.0),
            LineString([(179.0, 0.0), (-179.0, 0.0)]),
        ],
        crs="EPSG:4326",
    )


def _recording_reader(source: gpd.GeoDataFrame):
    """A stand-in for ``gpd.read_file`` that filters like a service and records calls.

    ``box(*bbox)`` on a wrapping bbox degenerates exactly the way OGR's envelope does,
    so a caller that forgot to split gets the same wrong answer here as against a real
    driver.
    """
    calls: list[dict] = []

    def fake_read(connection, layer=None, bbox=None, where=None, rows=None):
        calls.append(
            {
                "connection": connection,
                "layer": layer,
                "bbox": bbox,
                "where": where,
                "rows": rows,
            }
        )
        result = source
        if bbox is not None:
            result = result[result.intersects(box(*bbox))]
        if where is not None:
            result = result[result["name"] != where]
        if rows is not None:
            result = result.iloc[:rows]
        return result.reset_index(drop=True)

    return fake_read, calls


def _patch_oapif(monkeypatch, source: gpd.GeoDataFrame):
    """Point ``from_ogc_features`` at the recording reader; return the call log."""
    monkeypatch.setattr(_oapif, "_get_collections", lambda *a, **k: frozenset(["pts"]))
    fake_read, calls = _recording_reader(source)
    monkeypatch.setattr(_ogc.gpd, "read_file", fake_read)
    return calls


def _patch_wfs(monkeypatch, source: gpd.GeoDataFrame):
    """Point ``from_wfs`` at the recording reader; return the call log."""
    monkeypatch.setattr(
        _wfs, "_get_capabilities", lambda *a, **k: ((), frozenset(["pts"]))
    )
    fake_read, calls = _recording_reader(source)
    monkeypatch.setattr(_ogc.gpd, "read_file", fake_read)
    return calls


class TestTheWrapIsNoLongerRefused:
    """`read_kwargs` records a wrapping box instead of raising on it."""

    def test_it_records_the_box_the_caller_wrote(self):
        """Test scenario: the split belongs to `read_ogc_layer`, so nothing is rewritten here."""
        assert _ogc.read_kwargs(WRAP, None, None) == {"bbox": WRAP}

    @pytest.mark.parametrize("reader", [_wfs, _oapif])
    def test_both_readers_reach_the_same_assembler(self, reader):
        """Args: reader: The reader module under test.

        Test scenario:
            The two readers import `read_kwargs` under a private alias each; a fix
            applied to one and not the other would leave the refusal depending on
            which reader the caller reached for.
        """
        assert reader._read_kwargs(WRAP, None, None) == {"bbox": WRAP}

    def test_latitude_is_still_ordered(self):
        """Test scenario: there is no seam in latitude, so `miny >= maxy` is still a mistake."""
        with pytest.raises(ValueError, match="minx < maxx and miny < maxy"):
            _ogc.read_kwargs((1.0, 4.0, 3.0, 2.0), None, None)

    def test_a_wrap_reaching_past_the_seam_is_refused(self):
        """An out-of-range wrap splits into an inverted rect, which OGR normalises.

        Test scenario:
            `(190, -10, -170, 10)` yields a first half of `(190, ..., 180)` --
            west > east. Handing that to OGR is precisely the silent inversion the
            split exists to prevent: the filter goes out as `180, -10, 190, 10`,
            matches nothing on a CRS84 service, and the caller quietly receives
            only the eastern half. The raster readers already refused it; there is
            no reason for the two sides to disagree.
        """
        with pytest.raises(ValueError, match="within -180..180"):
            _ogc.read_kwargs((190.0, -10.0, -170.0, 10.0), None, None)

    def test_a_wrap_in_a_projected_crs_is_refused_by_its_coordinates(self):
        """A projected bbox has no 180 degree seam, and its numbers say so.

        Test scenario:
            `read_kwargs` never learns the CRS, but it does not need to: a
            wrapping bbox in metres carries coordinates far outside -180..180, so
            the corner-range half of the guard refuses it -- which is the right
            answer, since splitting it at 180 would be meaningless.
        """
        with pytest.raises(ValueError, match="within -180..180"):
            _ogc.read_kwargs(
                (2_000_000.0, 6_000_000.0, 1_000_000.0, 6_100_000.0), None, None
            )

    def test_a_zero_width_box_is_still_refused(self):
        """Test scenario: `minx == maxx` is empty whichever way it is read, wrap or not."""
        with pytest.raises(ValueError, match="minx < maxx"):
            _ogc.read_kwargs((3.0, 2.0, 3.0, 4.0), None, None)

    def test_a_non_finite_corner_is_still_refused(self):
        """Test scenario: allowing the wrap must not reopen the hole finiteness closed."""
        with pytest.raises(ValueError, match="four finite numbers"):
            _ogc.read_kwargs((1.0, 2.0, float("nan"), 4.0), None, None)


class TestTheReadIsSplitAtTheSeam:
    """One request per half, and the union of what they return."""

    def test_two_requests_are_made_one_per_half(self, monkeypatch):
        """Test scenario: the filter is a request parameter, so the wrap costs two of them."""
        calls = _patch_oapif(monkeypatch, _world())
        FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=WRAP
        )
        assert [call["bbox"] for call in calls] == [EAST_HALF, WEST_HALF]

    def test_a_plain_bbox_still_makes_exactly_one(self, monkeypatch):
        """Test scenario: the common path must not pay for the seam it does not cross."""
        calls = _patch_oapif(monkeypatch, _world())
        FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=(1.0, 2.0, 3.0, 4.0)
        )
        assert [call["bbox"] for call in calls] == [(1.0, 2.0, 3.0, 4.0)]

    def test_no_bbox_still_makes_exactly_one_unfiltered_read(self, monkeypatch):
        """Test scenario: a reader with no bbox must not acquire a spatial filter."""
        calls = _patch_oapif(monkeypatch, _world())
        FeatureCollection.from_ogc_features("https://h/api", collection="pts")
        assert len(calls) == 1 and calls[0]["bbox"] is None

    def test_the_union_of_both_halves_is_returned(self, monkeypatch):
        """Test scenario: everything inside the wrap, from either side of the seam."""
        _patch_oapif(monkeypatch, _world())
        fc = FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=WRAP
        )
        assert sorted(fc["name"]) == [
            "crosses",
            "east",
            "far-west",
            "just-west",
            "on-180",
        ]

    def test_nothing_outside_the_wrap_comes_back(self, monkeypatch):
        """Test scenario: the split must not widen the box into the far hemisphere."""
        _patch_oapif(monkeypatch, _world())
        fc = FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=WRAP
        )
        assert "elsewhere" not in list(fc["name"])

    def test_a_feature_straddling_the_seam_is_not_duplicated(self, monkeypatch):
        """Test scenario: `crosses` matches both halves; a plain concat would return it twice."""
        _patch_oapif(monkeypatch, _world())
        fc = FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=WRAP
        )
        assert list(fc["name"]).count("crosses") == 1

    def test_the_result_is_exactly_the_union_of_the_two_half_reads(self, monkeypatch):
        """Test scenario: no losses either — compare against reading each half by hand."""
        world = _world()
        _patch_oapif(monkeypatch, world)
        fc = FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=WRAP
        )
        by_hand = set()
        for half in (EAST_HALF, WEST_HALF):
            by_hand |= set(world[world.intersects(box(*half))]["name"])
        assert set(fc["name"]) == by_hand

    def test_the_index_is_contiguous_from_zero(self, monkeypatch):
        """Test scenario: both halves are indexed from 0, so a concat would repeat labels."""
        _patch_oapif(monkeypatch, _world())
        fc = FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=WRAP
        )
        assert list(fc.index) == list(range(len(fc)))

    def test_the_crs_survives_the_merge(self, monkeypatch):
        """Test scenario: a concat that dropped the CRS would break `to_crs` downstream."""
        _patch_oapif(monkeypatch, _world())
        fc = FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=WRAP
        )
        assert fc.crs.to_epsg() == 4326

    def test_output_crs_reprojects_the_merged_result(self, monkeypatch):
        """Test scenario: the reproject runs once, after the halves are unioned."""
        _patch_oapif(monkeypatch, _world())
        fc = FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=WRAP, output_crs="EPSG:3857"
        )
        assert fc.crs.to_epsg() == 3857 and len(fc) == 5

    def test_the_attribute_filter_is_applied_to_both_halves(self, monkeypatch):
        """Test scenario: `where` is not a spatial filter, so both requests must carry it."""
        calls = _patch_oapif(monkeypatch, _world())
        fc = FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=WRAP, where="crosses"
        )
        assert [call["where"] for call in calls] == ["crosses", "crosses"]
        assert "crosses" not in list(fc["name"])

    def test_max_features_caps_the_union_not_each_half(self, monkeypatch):
        """Test scenario: a per-half cap of N would hand back up to 2N features."""
        calls = _patch_oapif(monkeypatch, _world())
        fc = FeatureCollection.from_ogc_features(
            "https://h/api", collection="pts", bbox=WRAP, max_features=3
        )
        assert [call["rows"] for call in calls] == [3, 3], (
            "each half needs the full cap"
        )
        assert len(fc) == 3, "but the promise is about the union"

    @pytest.mark.parametrize(
        ("failing_half", "label"), [(0, "the first"), (1, "the second")]
    )
    def test_a_failure_on_either_half_is_branded(
        self, monkeypatch, failing_half, label
    ):
        """Args: failing_half: Which request fails. label: Which one, for the id.

        Test scenario:
            A wrapping read is two requests but still one call; either failing must
            surface as the reader's own error, worded as a single failed read is.
        """
        monkeypatch.setattr(
            _oapif, "_get_collections", lambda *a, **k: frozenset(["pts"])
        )
        seen: list[int] = []

        def flaky(connection, **kwargs):
            seen.append(1)
            if len(seen) - 1 == failing_half:
                raise RuntimeError("driver said no")
            return _world()

        monkeypatch.setattr(_ogc.gpd, "read_file", flaky)
        with pytest.raises(OGCAPIError, match="items request failed for 'pts'"):
            FeatureCollection.from_ogc_features(
                "https://h/api", collection="pts", bbox=WRAP
            )

    def test_the_wfs_reader_splits_the_same_way(self, monkeypatch):
        """Test scenario: both readers share `read_ogc_layer`; the fix must reach both."""
        calls = _patch_wfs(monkeypatch, _world())
        fc = FeatureCollection.from_wfs("https://h/ows", typename="pts", bbox=WRAP)
        assert [call["bbox"] for call in calls] == [EAST_HALF, WEST_HALF]
        assert sorted(fc["name"]) == [
            "crosses",
            "east",
            "far-west",
            "just-west",
            "on-180",
        ]

    def test_the_wfs_reader_brands_a_half_failure_as_wfserror(self, monkeypatch):
        """Test scenario: the per-reader error class must survive the split."""
        monkeypatch.setattr(
            _wfs, "_get_capabilities", lambda *a, **k: ((), frozenset(["pts"]))
        )

        def boom(*a, **k):
            raise RuntimeError("driver said no")

        monkeypatch.setattr(_ogc.gpd, "read_file", boom)
        with pytest.raises(WFSError, match="GetFeature failed for 'pts'"):
            FeatureCollection.from_wfs("https://h/ows", typename="pts", bbox=WRAP)


class TestMergeSeamHalves:
    """The union step on its own."""

    def test_it_drops_a_row_both_halves_returned(self):
        """Test scenario: identical geometry and attributes means one feature, seen twice."""
        east = _world().iloc[[0, 5]].reset_index(drop=True)
        west = _world().iloc[[5, 3]].reset_index(drop=True)
        merged = _ogc.merge_seam_halves([east, west], None)
        assert list(merged["name"]) == ["east", "crosses", "far-west"]

    def test_it_keeps_rows_that_only_look_alike(self):
        """Test scenario: same name, different geometry — two features, not one."""
        east = gpd.GeoDataFrame(
            {"name": ["twin"]}, geometry=[Point(172.0, 0.0)], crs="EPSG:4326"
        )
        west = gpd.GeoDataFrame(
            {"name": ["twin"]}, geometry=[Point(-172.0, 0.0)], crs="EPSG:4326"
        )
        assert len(_ogc.merge_seam_halves([east, west], None)) == 2

    def test_it_survives_an_unhashable_attribute(self):
        """Test scenario: a GeoJSON property can be a list; keying on it must not raise."""
        listy = gpd.GeoDataFrame(
            {"tags": [["a", "b"]]}, geometry=[Point(179.0, 0.0)], crs="EPSG:4326"
        )
        merged = _ogc.merge_seam_halves([listy, listy], None)
        assert len(merged) == 1

    def test_it_handles_a_half_that_returned_nothing(self):
        """Test scenario: a service with nothing east of the seam is normal, not an error."""
        world = _world()
        merged = _ogc.merge_seam_halves([world.iloc[0:0], world.iloc[[0]]], None)
        assert list(merged["name"]) == ["east"] and merged.crs.to_epsg() == 4326

    def test_it_handles_both_halves_empty(self):
        """Test scenario: an empty result must stay an empty frame, not a crash."""
        empty = _world().iloc[0:0]
        assert len(_ogc.merge_seam_halves([empty, empty], None)) == 0

    def test_it_reindexes_from_zero(self):
        """Test scenario: both inputs are 0-indexed, so labels would otherwise repeat."""
        world = _world()
        merged = _ogc.merge_seam_halves([world.iloc[[0, 1]], world.iloc[[2, 3]]], None)
        assert list(merged.index) == [0, 1, 2, 3]

    def test_the_cap_is_applied_after_de_duplication(self):
        """Test scenario: a duplicate must not eat a slot the cap could have given a feature."""
        east = _world().iloc[[5, 0]].reset_index(drop=True)
        west = _world().iloc[[5, 3]].reset_index(drop=True)
        merged = _ogc.merge_seam_halves([east, west], 3)
        assert list(merged["name"]) == ["crosses", "east", "far-west"]


def _feature(fid: str, geometry: dict, name: str) -> dict:
    """A single GeoJSON feature for the recording stub."""
    return {
        "type": "Feature",
        "id": fid,
        "geometry": geometry,
        "properties": {"name": name},
    }


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": [x, y]}


STUB_FEATURES = [
    (_feature("1", _point(172.0, 0.0), "east"), (172.0, 0.0, 172.0, 0.0)),
    (_feature("2", _point(-172.0, 0.0), "far-west"), (-172.0, 0.0, -172.0, 0.0)),
    (_feature("3", _point(0.0, 0.0), "elsewhere"), (0.0, 0.0, 0.0, 0.0)),
    (
        _feature(
            "4",
            {"type": "LineString", "coordinates": [[179.0, 0.0], [-179.0, 0.0]]},
            "crosses",
        ),
        (-179.0, 0.0, 179.0, 0.0),
    ),
]
"""The stub's features, each with the envelope it is filtered by."""

REQUESTS: list[str] = []
"""Every path the OAPIF driver asked the stub for, in order."""


class _RecordingOapifHandler(http.server.BaseHTTPRequestHandler):
    """A minimal OGC API – Features service that records paths and honours ``bbox``.

    Faithful enough for GDAL's ``OAPIF`` driver to negotiate: landing page,
    ``/conformance``, ``/collections``, the collection document and ``/items``.
    ``/items`` filters by envelope intersection, which is what a ``bbox`` means.
    """

    def _json(self, doc: dict, content_type: str = "application/json"):
        payload = json.dumps(doc).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _items(self, base: str, query: str):
        requested = None
        for part in query.lstrip("?").split("&"):
            if part.startswith("bbox="):
                requested = [float(v) for v in part[len("bbox=") :].split(",")]
        selected = [
            feature
            for feature, bounds in STUB_FEATURES
            if requested is None
            or (
                bounds[0] <= requested[2]
                and bounds[2] >= requested[0]
                and bounds[1] <= requested[3]
                and bounds[3] >= requested[1]
            )
        ]
        self._json(
            {
                "type": "FeatureCollection",
                "numberMatched": len(selected),
                "numberReturned": len(selected),
                "features": selected,
                "links": [{"rel": "self", "href": f"{base}/collections/pts/items"}],
            },
            content_type="application/geo+json",
        )

    def do_GET(self):  # noqa: N802
        REQUESTS.append(self.path)
        base = f"http://{self.headers.get('Host')}"
        path, _, query = self.path.partition("?")
        items_link = {
            "rel": "items",
            "href": f"{base}/collections/pts/items",
            "type": "application/geo+json",
        }
        extent = {"spatial": {"bbox": [[-180, -90, 180, 90]]}}
        if path in ("", "/"):
            self._json(
                {
                    "title": "Recording OAPIF stub",
                    "links": [
                        {"rel": "self", "href": f"{base}/", "type": "application/json"},
                        {
                            "rel": "conformance",
                            "href": f"{base}/conformance",
                            "type": "application/json",
                        },
                        {
                            "rel": "data",
                            "href": f"{base}/collections",
                            "type": "application/json",
                        },
                    ],
                }
            )
        elif path == "/conformance":
            self._json(
                {
                    "conformsTo": [
                        "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
                        "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson",
                    ]
                }
            )
        elif path == "/collections":
            self._json(
                {
                    "links": [{"rel": "self", "href": f"{base}/collections"}],
                    "collections": [
                        {
                            "id": "pts",
                            "title": "Points",
                            "extent": extent,
                            "links": [
                                items_link,
                                {"rel": "self", "href": f"{base}/collections/pts"},
                            ],
                        }
                    ],
                }
            )
        elif path == "/collections/pts":
            self._json({"id": "pts", "extent": extent, "links": [items_link]})
        elif path == "/collections/pts/items":
            self._items(base, query)
        else:
            self.send_error(404)

    def log_message(self, *args, **kwargs):  # noqa: N802
        return


class TestWhatTheFilterActuallyIs:
    """Drive the real OAPIF driver against a recording stub.

    This is the evidence behind the design: the vector `bbox` is not a local OGR
    predicate pyramids can apply twice for free, it is a request parameter. The
    driver is exercised directly through `gdal.OpenEx` because the bundled pyogrio
    reader cannot reach a localhost mock reliably — the same reason
    `test_oapif.py::TestOapifDriverPaging` does it this way.
    """

    @pytest.fixture
    def service(self):
        """Start the recording stub, clear the request log, yield its base URL."""
        httpd = socketserver.ThreadingTCPServer(
            ("127.0.0.1", 0), _RecordingOapifHandler
        )
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        REQUESTS.clear()
        gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", "15")
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}"
        finally:
            gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", None)
            httpd.shutdown()
            httpd.server_close()

    @staticmethod
    def _read(url: str, apply_filter=None) -> list[str]:
        """Read the stub's layer through the OAPIF driver, optionally filtered."""
        ds = gdal.OpenEx(f"OAPIF:{url}", gdal.OF_VECTOR)
        layer = ds.GetLayerByName("pts")
        if apply_filter is not None:
            apply_filter(layer)
        names = sorted(feature.GetField("name") for feature in layer)
        layer = None
        ds = None  # release the HTTP handle before the server tears down
        return names

    @staticmethod
    def _item_requests() -> list[str]:
        """The ``/items`` paths the driver issued, discovery noise dropped."""
        return [path for path in REQUESTS if path.startswith("/collections/pts/items")]

    def test_the_bbox_leaves_as_a_request_parameter(self, service):
        """Test scenario: this is the whole reason a wrap costs two requests.

        If the filter were local, both halves could be applied to one downloaded
        result. It is not: the half box is in the URL, so the server decides.
        """
        names = self._read(
            service, lambda layer: layer.SetSpatialFilterRect(*EAST_HALF)
        )
        assert any("bbox=170,-10,180,10" in path for path in self._item_requests()), (
            f"the half box never reached the service: {self._item_requests()}"
        )
        assert names == ["crosses", "east"]

    def test_a_multipart_mask_is_sent_as_its_envelope(self, service):
        """Test scenario: the "one multipart query" option, measured rather than assumed.

        OGR takes a `MultiPolygon` spatial filter, and the answer it computes is
        right — it filters exactly on the client. What it *asks the server for* is
        the geometry's envelope, which for two halves either side of the seam is
        the entire globe in longitude. Correct, and a full download.
        """
        both = ogr.CreateGeometryFromWkt(
            "MULTIPOLYGON(((170 -10, 180 -10, 180 10, 170 10, 170 -10)),"
            "((-180 -10, -170 -10, -170 10, -180 10, -180 -10)))"
        )
        names = self._read(service, lambda layer: layer.SetSpatialFilter(both))
        assert any("bbox=-180,-10,180,10" in path for path in self._item_requests()), (
            f"expected the whole-globe envelope: {self._item_requests()}"
        )
        assert names == ["crosses", "east", "far-west"]

    def test_a_wrapping_rect_handed_to_ogr_asks_for_the_complement(self, service):
        """Test scenario: what "just pass the wrap through" would actually have done.

        OGR normalises the envelope, so `(170, -10, -170, 10)` goes out as
        `bbox=-170,-10,170,10` — everything *except* the requested strip — and it
        raises nothing on the way. `east` and `far-west`, the two features actually
        inside the requested box, are the ones missing from the answer. Splitting
        before the driver is not a convenience.
        """
        names = self._read(service, lambda layer: layer.SetSpatialFilterRect(*WRAP))
        assert any("bbox=-170,-10,170,10" in path for path in self._item_requests()), (
            f"expected the inverted envelope: {self._item_requests()}"
        )
        assert "east" not in names and "far-west" not in names, (
            f"the features inside the wrap were dropped, silently: {names}"
        )
        assert "elsewhere" in names, (
            f"and a feature 170 degrees away was returned instead: {names}"
        )

    def test_the_two_halves_together_cover_the_wrap(self, service):
        """Test scenario: end to end, the split is the right answer against a real driver."""
        east = self._read(service, lambda layer: layer.SetSpatialFilterRect(*EAST_HALF))
        west = self._read(service, lambda layer: layer.SetSpatialFilterRect(*WEST_HALF))
        assert east == ["crosses", "east"]
        assert west == ["crosses", "far-west"]
        assert sorted(set(east) | set(west)) == ["crosses", "east", "far-west"]


class TestFishnetRefusesTheWrapOnPurpose:
    """A wrapping fishnet has no coherent answer; the refusal is the decision."""

    def test_it_refuses_a_wrapping_extent(self):
        """Test scenario: unlike the readers, `fishnet` does not read `minx > maxx` as a seam."""
        with pytest.raises(
            ValueError, match="fishnet: bounds must satisfy minx < maxx"
        ):
            _tess.fishnet_cells(WRAP, 1.0)

    def test_the_public_classmethod_refuses_it_too(self):
        """Test scenario: the refusal must be reachable where callers actually are."""
        with pytest.raises(ValueError, match="fishnet: bounds must satisfy"):
            FeatureCollection.fishnet(WRAP, 1.0, crs="EPSG:4326")

    def test_it_refuses_regardless_of_crs(self):
        """Test scenario: the reason it refuses — `bounds` is in the units of `crs`.

        There is no 180 degree seam in metres and none at all in a CRS-less grid,
        so `minx > maxx` cannot be read as a wrap without guessing what the numbers
        mean. `crs` is not even passed to `fishnet_cells`, which settles it.
        """
        with pytest.raises(ValueError, match="fishnet: bounds must satisfy"):
            FeatureCollection.fishnet((5e6, -1e6, -5e6, 1e6), 1e5, crs="EPSG:3857")
        with pytest.raises(ValueError, match="fishnet: bounds must satisfy"):
            FeatureCollection.fishnet((5.0, 0.0, 1.0, 1.0), 0.5)

    def test_the_message_names_fishnet_and_bounds(self):
        """Test scenario: the refusal is about a grid extent, not about a request bbox.

        That is why this rule is not collapsed onto `validate_bbox` the way
        `read_kwargs` was — the shared validator speaks about a lon/lat `bbox` and
        has an antimeridian mode this one must not inherit.
        """
        with pytest.raises(ValueError) as excinfo:
            _tess.fishnet_cells(WRAP, 1.0)
        message = str(excinfo.value)
        assert message.startswith("fishnet: bounds")
        assert "bbox" not in message

    def test_the_documented_workaround_produces_two_usable_grids(self):
        """Test scenario: the docstring tells callers to grid each half; that has to work.

        It also shows why one grid is not on offer: `col` restarts at 0 in the
        second half, because each half's columns are laid from its own `minx`.
        """
        grids = [
            FeatureCollection.fishnet(half, 5.0, crs="EPSG:4326")
            for half in seam_halves(WRAP)
        ]
        assert [len(grid) for grid in grids] == [8, 8]
        assert all(min(grid["col"]) == 0 for grid in grids), (
            "each half numbers its own columns from zero — the reason a single "
            "wrapping fishnet has no coherent numbering"
        )
        east_cells, west_cells = (
            [geom.bounds for geom in grid.geometry] for grid in grids
        )
        assert max(bounds[2] for bounds in east_cells) == 180.0
        assert min(bounds[0] for bounds in west_cells) == -180.0

    def test_an_ordinary_extent_is_untouched(self):
        """Test scenario: recording the refusal must not narrow what already worked."""
        polygons, rows, cols = _tess.fishnet_cells((0.0, 0.0, 1.0, 1.0), 0.5)
        assert len(polygons) == 4 and (rows, cols) == ([0, 0, 1, 1], [0, 1, 0, 1])
