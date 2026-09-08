"""Tests for the OGC WMS / WMTS reader (`pyramids.dataset._wms`).

Network-free except the gated live class. The pure helpers — output-size
resolution, the ``<GDAL_WMS>`` descriptor, the ``WMTS:`` connection string, the
layers normalisation, and the ``from_wms`` "needs a size" guard — are covered
offline; a live end-to-end against public OSM-WMS and NASA GIBS WMTS runs only
under ``-m live``.

The antimeridian split is covered offline too, against a stand-in server whose
pixels carry their own centre longitude (``_longitude_raster``). That is what
makes the *geometry* checkable without a network: the stitched raster is compared
column-for-column against the same two regions fetched as ordinary non-wrapping
requests, and its pixel values must step by exactly one cell across the seam. What
it cannot prove is how a real service renders the two ``GetMap`` calls — styling,
label placement and tile-cache seams either side of 180 are a live-server
question, covered only by ``-m live``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, _wms
from pyramids.errors import WMSError

pytestmark = pytest.mark.core

BBOX = (5.0, 51.0, 6.0, 52.0)
WRAP = (170.0, -10.0, -170.0, 10.0)
ENDPOINT = "https://host/wms?"


def _tiny_dataset(epsg: int = 4326) -> Dataset:
    """A 3-band 4x5 in-memory raster for offline reprojection tests."""
    arr = np.ones((3, 4, 5), dtype="float32")
    return Dataset.from_array(
        arr,
        geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.1, epsg=epsg),
    )


def _descriptor_values(descriptor: str) -> dict[str, str]:
    """The simple text elements of a ``<GDAL_WMS>`` descriptor, by tag name."""
    return dict(re.findall(r"<(\w+)>([^<]*)</\1>", descriptor))


def _lonlat_srs() -> osr.SpatialReference:
    """EPSG:4326 in lon/lat order, as the readers stamp it."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _longitude_raster(
    ulx: float,
    uly: float,
    x_res: float,
    y_res: float,
    size: tuple[int, int],
    bands: int,
) -> gdal.Dataset:
    """A MEM raster whose every pixel carries its own centre longitude (folded to 0..360).

    Encoding the geometry in the *pixel values* is what lets an offline test check a
    stitch against the geometry it claims: a constant step between neighbouring
    columns means one uniform ground resolution, and any jump at the seam means a
    half landed in the wrong columns. Each band after the first is offset by a
    further 1000, so a band mix-up cannot pass unnoticed either.
    """
    width, height = size
    src = gdal.GetDriverByName("MEM").Create("", width, height, bands, gdal.GDT_Float64)
    src.SetGeoTransform((ulx, x_res, 0.0, uly, 0.0, y_res))
    src.SetSpatialRef(_lonlat_srs())
    lon = np.mod(ulx + (np.arange(width) + 0.5) * x_res, 360.0)
    for band in range(1, bands + 1):
        src.GetRasterBand(band).WriteArray(
            np.tile(lon, (height, 1)) + 1000.0 * (band - 1)
        )
    return src


def _fake_wms_source(descriptor: str) -> gdal.Dataset:
    """Render what a WMS would return for `descriptor`: the window it asked for."""
    values = _descriptor_values(descriptor)
    ulx, uly = float(values["UpperLeftX"]), float(values["UpperLeftY"])
    lrx, lry = float(values["LowerRightX"]), float(values["LowerRightY"])
    width, height = int(values["SizeX"]), int(values["SizeY"])
    return _longitude_raster(
        ulx,
        uly,
        (lrx - ulx) / width,
        (lry - uly) / height,
        (width, height),
        int(values["BandsCount"]),
    )


@pytest.fixture
def fake_wms(monkeypatch):
    """Serve every ``GetMap`` offline, returning the list of descriptors requested."""
    requests: list[str] = []

    def _fake_open(descriptor, layer, hint):
        requests.append(descriptor)
        return _fake_wms_source(descriptor)

    monkeypatch.setattr(_wms, "_open", _fake_open)
    return requests


def _bands_rows_columns(ds: Dataset) -> np.ndarray:
    """`ds.read_array()` as ``(bands, rows, columns)`` whatever the band count."""
    return np.asarray(ds.read_array()).reshape(-1, ds.rows, ds.columns)


class TestOutputSize:
    def test_explicit_size_used_verbatim(self):
        assert _wms._output_size(BBOX, (640, 480), None) == (640, 480)

    def test_resolution_divides_the_extent(self):
        # 1 deg extent / 0.01 deg = 100 px on each axis
        assert _wms._output_size(BBOX, None, 0.01) == (100, 100)

    def test_non_square_resolution(self):
        assert _wms._output_size(BBOX, None, (0.01, 0.02)) == (100, 50)

    def test_requires_size_or_resolution(self):
        with pytest.raises(ValueError, match="needs the output size"):
            _wms._output_size(BBOX, None, None)

    @pytest.mark.parametrize("bad", [(0, 100), (100, 0), (-1, 100)])
    def test_rejects_non_positive_size(self, bad):
        with pytest.raises(ValueError, match="two positive integers"):
            _wms._output_size(BBOX, bad, None)


class TestOutputSizeCeiling:
    """A GetMap is bounded like every other network read.

    Before a wrap was accepted, a `minx > maxx` bbox never reached the sizing path.
    Now it does, and a transposed pair spans 359 degrees -- so without a ceiling an
    ordinary typo becomes two requests of a few hundred thousand columns each.
    """

    def test_a_transposed_bbox_is_refused_rather_than_sized(self):
        """The case the ceiling exists for.

        Test scenario:
            `(6, 51, 5, 52)` reads as a 359 degree wrap. At 0.001 degree pixels
            that is 359,000 columns -- roughly half a gigabyte per half, and
            another full copy to stitch them. It raises instead, and the message
            names the way out.
        """
        with pytest.raises(
            ValueError, match=r"exceeds the 25000 px limit: 359000x1000"
        ):
            _wms._output_size((6.0, 51.0, 5.0, 52.0), None, 0.001)

    def test_an_ordinary_fine_read_is_still_allowed(self):
        """The ceiling is generous enough not to bother a real request."""
        assert _wms._output_size((5.0, 51.0, 6.0, 52.0), None, 0.001) == (1000, 1000)

    def test_an_explicit_size_is_still_taken_verbatim(self):
        """The cap applies to sizing from a resolution, not to a size the caller set."""
        assert _wms._output_size((5.0, 51.0, 6.0, 52.0), (4096, 2048), None) == (
            4096,
            2048,
        )


class TestLayersValue:
    def test_string_passthrough(self):
        assert _wms._layers_value("OSM-WMS") == "OSM-WMS"

    def test_list_joined_with_commas(self):
        assert _wms._layers_value(["a", "b", "c"]) == "a,b,c"

    def test_tuple_joined_with_commas(self):
        assert _wms._layers_value(("a", "b")) == "a,b"

    @pytest.mark.parametrize("bad", ["", "  ", [], (), ["", "L"], ["A", "  "]])
    def test_rejects_empty_or_blank_entries(self, bad):
        with pytest.raises(ValueError, match="at least one non-empty layer"):
            _wms._layers_value(bad)


class TestWmsDescriptor:
    def test_carries_service_and_window(self):
        xml = _wms._wms_descriptor(
            "https://host/wms?",
            "L1,L2",
            "EPSG:4326",
            "image/png",
            "1.3.0",
            BBOX,
            (512, 256),
            4,
        )
        assert '<Service name="WMS">' in xml
        assert "<Version>1.3.0</Version>" in xml
        assert "<ServerUrl>https://host/wms?</ServerUrl>" in xml
        assert "<Layers>L1,L2</Layers>" in xml
        assert "<CRS>EPSG:4326</CRS>" in xml
        assert "<ImageFormat>image/png</ImageFormat>" in xml
        # DataWindow: upper-left = (minx, maxy), lower-right = (maxx, miny)
        assert "<UpperLeftX>5.0</UpperLeftX>" in xml
        assert "<UpperLeftY>52.0</UpperLeftY>" in xml
        assert "<LowerRightX>6.0</LowerRightX>" in xml
        assert "<LowerRightY>51.0</LowerRightY>" in xml
        assert "<SizeX>512</SizeX>" in xml and "<SizeY>256</SizeY>" in xml
        assert "<BandsCount>4</BandsCount>" in xml

    def test_escapes_ampersand_in_url(self):
        xml = _wms._wms_descriptor(
            "https://host/wms?token=a&b",
            "L",
            "EPSG:3857",
            "image/jpeg",
            "1.1.1",
            BBOX,
            (10, 10),
            3,
        )
        assert "token=a&amp;b" in xml
        assert "&b" not in xml.replace("&amp;", "")


class TestCrsElementTag:
    @pytest.mark.parametrize(
        "version, tag",
        [
            ("1.3.0", "CRS"),
            ("1.1.1", "SRS"),
            ("1.1.0", "SRS"),
            ("1.0.0", "SRS"),
            ("bogus", "CRS"),
        ],
    )
    def test_tag_tracks_version(self, version, tag):
        assert _wms._crs_element_tag(version) == tag

    def test_descriptor_uses_srs_below_1_3_0(self):
        xml = _wms._wms_descriptor(
            "https://x?",
            "L",
            "EPSG:4326",
            "image/png",
            "1.1.1",
            BBOX,
            (10, 10),
            3,
        )
        assert "<SRS>EPSG:4326</SRS>" in xml and "<CRS>" not in xml

    def test_descriptor_gdal_open_accepts_all_versions(self):
        """GDAL must accept the descriptor GDAL-side for every WMS version (the M1 gap).

        gdal.Open only parses the descriptor (the GetMap fetch is deferred), so this
        is network-free; before the SRS/CRS fix it raised for 1.1.1 / 1.0.0.
        """
        for version in ("1.0.0", "1.1.1", "1.3.0"):
            xml = _wms._wms_descriptor(
                "https://example.invalid/wms?",
                "L",
                "EPSG:4326",
                "image/png",
                version,
                BBOX,
                (16, 16),
                3,
            )
            src = _wms.gdal.Open(xml)
            assert src is not None, f"GDAL rejected the {version} descriptor"


class TestWmtsConnection:
    def test_layer_only(self):
        conn = _wms._wmts_connection("https://c.xml", "TC", None)
        assert conn == "WMTS:https://c.xml,layer=TC"

    def test_with_tile_matrix_set(self):
        conn = _wms._wmts_connection("https://c.xml", "TC", "GMC")
        assert conn == "WMTS:https://c.xml,layer=TC,tilematrixset=GMC"


class TestFromWmsGuards:
    def test_from_wms_without_size_or_resolution_raises(self):
        """The size guard fires before any network call."""
        with pytest.raises(ValueError, match="needs the output size"):
            Dataset.from_wms("https://host/wms?", layers="L", bbox=BBOX)

    def test_from_wms_with_both_size_and_resolution_raises(self):
        """size and resolution are mutually exclusive."""
        with pytest.raises(ValueError, match="not both"):
            Dataset.from_wms(
                "https://host/wms?",
                layers="L",
                bbox=BBOX,
                size=(10, 10),
                resolution=0.1,
            )

    @pytest.mark.parametrize("empty", ["", [], (), ["", "L"]])
    def test_from_wms_rejects_empty_layers(self, empty):
        """An empty or partial-empty layers argument fails fast, before any network."""
        with pytest.raises(ValueError, match="at least one non-empty layer"):
            Dataset.from_wms(
                "https://host/wms?", layers=empty, bbox=BBOX, size=(10, 10)
            )

    def test_from_wms_rejects_malformed_bbox(self):
        """An inverted *latitude* range is still refused.

        Longitude no longer is: ``minx > maxx`` is now read as an antimeridian wrap
        (see TestFromWmsAntimeridian), so only the y axis is still one-way.
        """
        with pytest.raises(ValueError, match="miny < maxy"):
            Dataset.from_wms(
                "https://host/wms?",
                layers="L",
                bbox=(5.0, 52.0, 6.0, 51.0),
                size=(10, 10),
            )


class TestRenderWmsErrorWrapping:
    def test_translate_runtimeerror_becomes_wmserror(self, monkeypatch):
        """A GetMap server error (RuntimeError during Translate) surfaces as WMSError."""

        def boom(*_a, **_k):
            raise RuntimeError("HTTP error 500")

        monkeypatch.setattr(_wms.gdal, "Translate", boom)
        with pytest.raises(WMSError, match="GetMap failed"):
            _wms._render_wms(object(), "OSM-WMS")

    def test_translate_none_becomes_wmserror(self, monkeypatch):
        """A None return (no raster) also surfaces as WMSError."""
        monkeypatch.setattr(_wms.gdal, "Translate", lambda *_a, **_k: None)
        with pytest.raises(WMSError, match="no raster"):
            _wms._render_wms(object(), "OSM-WMS")


class TestWmtsPixelCeiling:
    """The WMTS windowed read is bounded by the shared pixel ceiling (ARC-74)."""

    def test_finest_level_over_wide_bbox_raises(self):
        """A finest-level (resolution=None) read over a wide bbox is rejected."""
        src = gdal.GetDriverByName("MEM").Create("", 4, 4, 1)
        src.SetGeoTransform((0.0, 0.001, 0.0, 100.0, 0.0, -0.001))
        with pytest.raises(ValueError, match="limit"):
            _wms._translate_window(src, [0.0, 100.0, 100.0, 0.0], "layer", None, "near")


class TestReprojectTail:
    def test_noop_without_output_crs(self):
        """No output_crs -> the dataset is returned unchanged."""
        ds = _tiny_dataset()
        assert _wms._reproject_tail(ds, None, "nearest") is ds

    def test_reprojects_to_output_crs(self):
        """output_crs reprojects; no cell_size is forced (no unit mismatch)."""
        out = _wms._reproject_tail(_tiny_dataset(4326), "EPSG:3857", "nearest")
        assert out.epsg == 3857


class TestAvailableWmtsLayers:
    def test_parses_layer_ids_from_subdatasets(self, monkeypatch):
        """Layer ids are extracted (and de-duplicated) from the WMTS subdatasets."""

        class _Caps:
            @staticmethod
            def GetSubDatasets():
                # (name, description) pairs, as GDAL parses the SUBDATASETS domain —
                # the same surface Dataset.subdatasets consumes.
                return [
                    ("WMTS:https://x,layer=B", "layer B"),
                    ("WMTS:https://x,layer=A,tilematrixset=t", "layer A"),
                    ("WMTS:https://x,layer=B", "duplicate -> de-duplicated"),
                    ("WMTS:https://x", "no ,layer= key -> skipped"),
                    (
                        "WMTS:https://x",
                        "desc has ,layer=Z but the name does not -> skipped",
                    ),
                ]

        monkeypatch.setattr(_wms.gdal, "Open", lambda _c: _Caps())
        assert _wms._available_wmts_layers("https://x") == ["A", "B"]

    def test_returns_empty_when_open_fails(self, monkeypatch):
        """A capabilities open failure yields [] (never masks the real error)."""

        def boom(_c):
            raise RuntimeError("offline")

        monkeypatch.setattr(_wms.gdal, "Open", boom)
        assert _wms._available_wmts_layers("https://x") == []


class TestSeamBboxGuard:
    """`_check_seam_bbox` refuses the wraps this reader cannot honestly serve."""

    def test_ordinary_bbox_is_untouched_in_any_crs(self):
        """A minx < maxx box is never a wrap, so the guard never looks at the CRS."""
        assert _wms._check_seam_bbox((0.0, 0.0, 1e6, 1e6), "EPSG:3857") is None

    def test_projected_crs_refuses_a_wrapping_bbox(self):
        """There is no 180 degree seam in a projected CRS - minx > maxx is inverted."""
        with pytest.raises(ValueError, match="not a geographic"):
            Dataset.from_wms(
                ENDPOINT, layers="L", bbox=WRAP, crs="EPSG:3857", size=(10, 10)
            )

    @pytest.mark.parametrize(
        "bbox", [(200.0, -10.0, -170.0, 10.0), (170.0, -10.0, -200.0, 10.0)]
    )
    def test_corner_outside_the_lonlat_range_is_refused(self, bbox):
        """A wrap needs both corners in -180..180; outside it, the split is guesswork."""
        with pytest.raises(ValueError, match=r"-180\.\.180"):
            Dataset.from_wms(ENDPOINT, layers="L", bbox=bbox, size=(10, 10))

    def test_wmts_applies_the_same_guard(self):
        with pytest.raises(ValueError, match="not a geographic"):
            Dataset.from_wmts(
                "https://c.xml", layer="L", bbox=WRAP, crs="EPSG:3857", resolution=0.5
            )


class TestSeamWindows:
    """The pixel-width division rule (`_seam_windows`)."""

    def test_ordinary_bbox_is_one_untouched_request(self):
        assert _wms._seam_windows(BBOX, (512, 256)) == [(BBOX, (512, 256))]

    @pytest.mark.parametrize("width", [400, 401, 333, 2, 4097])
    def test_half_widths_always_sum_to_the_requested_width(self, width):
        windows = _wms._seam_windows(WRAP, (width, 16))
        assert sum(size[0] for _, size in windows) == width

    @pytest.mark.parametrize(
        "bbox, width",
        [(WRAP, 400), ((175.0, -10.0, -170.0, 10.0), 400), (WRAP, 401)],
    )
    def test_every_half_is_rendered_at_the_one_shared_resolution(self, bbox, width):
        """Each half's span / columns is the whole wrapping span / the whole width."""
        expected = ((bbox[2] + 360.0) - bbox[0]) / width
        for box, size in _wms._seam_windows(bbox, (width, 16)):
            assert (box[2] - box[0]) / size[0] == pytest.approx(expected)

    @pytest.mark.parametrize(
        "bbox, width", [(WRAP, 400), ((175.0, -10.0, -170.0, 10.0), 400), (WRAP, 401)]
    )
    def test_snapping_moves_the_window_by_under_half_a_pixel(self, bbox, width):
        """The seam is snapped to the nearest pixel edge, so nothing shifts further.

        Half a pixel exactly is reachable — a seam landing on a column midpoint
        (400 columns over 20 degrees split 10/10 at width 401) is the tie case.
        """
        res = ((bbox[2] + 360.0) - bbox[0]) / width
        windows = _wms._seam_windows(bbox, (width, 16))
        assert abs(windows[0][0][0] - bbox[0]) <= 0.5 * res + 1e-12

    def test_uneven_halves_split_in_proportion_to_their_span(self):
        """5 of 15 degrees is a third of the width, not half of it."""
        windows = _wms._seam_windows((175.0, -10.0, -170.0, 10.0), (400, 16))
        assert [size[0] for _, size in windows] == [133, 267]

    def test_halves_meet_exactly_at_the_seam(self):
        (west_box, _), (east_box, _) = _wms._seam_windows(WRAP, (401, 16))
        assert west_box[2] == 180.0
        assert east_box[0] == -180.0

    def test_a_wrap_needs_two_pixels_of_width(self):
        """One pixel cannot straddle the seam, so asking for one is refused.

        Test scenario:
            At width 1 over a 20 degree wrap the resolution is 20 degrees and each
            half is exactly half a pixel. `round()` sent the west half to zero
            through banker's rounding, dropping 10 degrees that were requested and
            adding 10 that were not -- the single window came back as
            `-180 .. -160`. There is no honest one-pixel answer, so it raises.
        """
        with pytest.raises(ValueError, match="at least 2 pixels of width"):
            _wms._seam_windows((170.0, -10.0, -170.0, 10.0), (1, 4))

    def test_two_pixels_give_each_side_of_the_seam_one(self):
        """The smallest width a wrap can be rendered at, split evenly."""
        windows = _wms._seam_windows((170.0, -10.0, -170.0, 10.0), (2, 4))
        assert [size for _, size in windows] == [(1, 4), (1, 4)]
        assert [window[0] for window, _ in windows] == [170.0, -180.0]

    def test_an_exact_half_pixel_half_is_kept_not_dropped(self):
        """Rounding half up, so the boundary case keeps the half rather than losing it.

        Test scenario:
            A three-pixel wrap over equal spans puts each half at 1.5 pixels.
            Banker's rounding would send one to 2 and leave the other at 1 by
            accident of parity; rounding half up makes the west half the wider one
            deterministically, and neither is dropped.
        """
        windows = _wms._seam_windows((170.0, -10.0, -170.0, 10.0), (3, 4))
        assert [size[0] for _, size in windows] == [2, 1]
        assert sum(size[0] for _, size in windows) == 3

    def test_sub_half_pixel_west_sliver_collapses_to_one_request(self):
        """A west side under half a pixel wide *is* the half-pixel snap: drop it."""
        windows = _wms._seam_windows((179.999, -10.0, -170.0, 10.0), (2000, 16))
        assert len(windows) == 1
        assert windows[0][1][0] == 2000
        assert windows[0][0][0] == -180.0

    def test_sub_half_pixel_east_sliver_collapses_to_one_request(self):
        windows = _wms._seam_windows((170.0, -10.0, -179.999, 10.0), (2000, 16))
        assert len(windows) == 1
        assert windows[0][1][0] == 2000
        assert windows[0][0][2] == 180.0


class TestFromWmsAntimeridian:
    """`from_wms` splits, fetches and stitches a west > east bbox, like `crop`."""

    def test_wrapping_bbox_is_no_longer_refused(self, fake_wms):
        ds = Dataset.from_wms(ENDPOINT, layers="L", bbox=WRAP, size=(400, 200))
        assert ds.shape == (3, 200, 400)

    def test_two_getmap_requests_are_issued_either_side_of_the_seam(self, fake_wms):
        Dataset.from_wms(ENDPOINT, layers="L", bbox=WRAP, size=(400, 200))
        assert len(fake_wms) == 2
        west, east = (_descriptor_values(d) for d in fake_wms)
        assert float(west["UpperLeftX"]) == pytest.approx(170.0)
        assert float(west["LowerRightX"]) == pytest.approx(180.0)
        assert float(east["UpperLeftX"]) == pytest.approx(-180.0)
        assert float(east["LowerRightX"]) == pytest.approx(-170.0)
        assert (int(west["SizeX"]), int(east["SizeX"])) == (200, 200)
        assert {west["SizeY"], east["SizeY"]} == {"200"}

    @pytest.mark.parametrize("width", [400, 401, 333])
    def test_output_width_is_exactly_what_was_requested(self, fake_wms, width):
        ds = Dataset.from_wms(ENDPOINT, layers="L", bbox=WRAP, size=(width, 100))
        assert ds.columns == width

    def test_ground_resolution_is_uniform_across_the_seam(self, fake_wms):
        """Neighbouring columns step by one cell everywhere, the seam column included."""
        ds = Dataset.from_wms(ENDPOINT, layers="L", bbox=WRAP, size=(400, 200))
        steps = np.diff(_bands_rows_columns(ds)[0, 0])
        assert steps == pytest.approx(ds.geotransform[1])
        assert steps[199] == pytest.approx(ds.geotransform[1])

    def test_pixels_match_the_same_two_regions_fetched_separately(self, fake_wms):
        """The stitched raster is the two non-wrapping requests, side by side."""
        wrapped = _bands_rows_columns(
            Dataset.from_wms(ENDPOINT, layers="L", bbox=WRAP, size=(400, 200))
        )
        west = _bands_rows_columns(
            Dataset.from_wms(
                ENDPOINT, layers="L", bbox=(170.0, -10.0, 180.0, 10.0), size=(200, 200)
            )
        )
        east = _bands_rows_columns(
            Dataset.from_wms(
                ENDPOINT,
                layers="L",
                bbox=(-180.0, -10.0, -170.0, 10.0),
                size=(200, 200),
            )
        )
        assert wrapped[:, :, :200] == pytest.approx(west)
        assert wrapped[:, :, 200:] == pytest.approx(east)

    def test_longitude_continues_past_the_seam(self, fake_wms):
        """The result keeps the west half's geotransform, as a stitched crop does."""
        ds = Dataset.from_wms(ENDPOINT, layers="L", bbox=WRAP, size=(400, 200))
        assert ds.geotransform[0] == pytest.approx(170.0)
        assert ds.geotransform[1] == pytest.approx(0.05)
        assert ds.bbox[0] == pytest.approx(170.0)
        assert ds.bbox[2] == pytest.approx(190.0)

    def test_uneven_split_keeps_one_resolution_and_the_full_width(self, fake_wms):
        ds = Dataset.from_wms(
            ENDPOINT, layers="L", bbox=(175.0, -10.0, -170.0, 10.0), size=(400, 200)
        )
        cell = 15.0 / 400
        assert ds.columns == 400
        assert ds.geotransform[1] == pytest.approx(cell)
        assert [int(_descriptor_values(d)["SizeX"]) for d in fake_wms] == [133, 267]
        assert abs(ds.geotransform[0] - 175.0) < 0.5 * cell
        steps = np.diff(_bands_rows_columns(ds)[0, 0])
        assert steps == pytest.approx(cell)

    def test_resolution_sizes_the_wrapping_span(self, fake_wms):
        """`resolution=` divides the 20 degree wrap, not the -340 the corners subtract to."""
        ds = Dataset.from_wms(ENDPOINT, layers="L", bbox=WRAP, resolution=0.05)
        assert ds.shape == (3, 400, 400)
        assert ds.geotransform[1] == pytest.approx(0.05)

    def test_sub_half_pixel_sliver_makes_one_request(self, fake_wms):
        ds = Dataset.from_wms(
            ENDPOINT, layers="L", bbox=(179.999, -10.0, -170.0, 10.0), size=(2000, 200)
        )
        assert len(fake_wms) == 1
        assert ds.columns == 2000
        assert abs(ds.geotransform[0] - (179.999 - 360.0)) < 0.5 * ds.geotransform[1]

    def test_non_wrapping_bbox_still_makes_a_single_request(self, fake_wms):
        ds = Dataset.from_wms(ENDPOINT, layers="L", bbox=BBOX, size=(512, 256))
        assert len(fake_wms) == 1
        assert ds.shape == (3, 256, 512)
        assert ds.bbox == pytest.approx([5.0, 51.0, 6.0, 52.0])

    def test_every_requested_band_survives_the_stitch(self, fake_wms):
        """Band n of the fake source is offset by 1000n, so a mix-up cannot pass."""
        ds = Dataset.from_wms(ENDPOINT, layers="L", bbox=WRAP, size=(400, 20), bands=4)
        arr = _bands_rows_columns(ds)
        assert arr.shape == (4, 20, 400)
        for band in range(4):
            assert arr[band, 0] - arr[0, 0] == pytest.approx(1000.0 * band)


class TestSeamOffset:
    def test_geographic_layer_spans_360_degrees(self):
        assert _wms._seam_offset(WRAP, "EPSG:4326", _lonlat_srs()) == pytest.approx(
            360.0
        )

    def test_web_mercator_layer_spans_the_world_width(self):
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        assert _wms._seam_offset(WRAP, "EPSG:4326", srs) == pytest.approx(
            40075016.6856, rel=1e-9
        )


class TestMergeLonHalves:
    """The stitch and the invariant it refuses to stitch without."""

    @staticmethod
    def _half(ulx, width, *, x_res=0.5, y_res=-0.5, uly=10.0, rows=4, bands=2):
        src = gdal.GetDriverByName("MEM").Create("", width, rows, bands, gdal.GDT_Int16)
        src.SetGeoTransform((ulx, x_res, 0.0, uly, 0.0, y_res))
        for band in range(1, bands + 1):
            src.GetRasterBand(band).WriteArray(
                np.full((rows, width), band * 10, dtype="int16")
            )
        return src

    def test_places_each_half_in_its_own_columns(self):
        west, east = self._half(170.0, 20), self._half(-180.0, 12)
        merged = _wms._merge_lon_halves(west, east, 360.0)
        assert merged.RasterXSize == 32
        assert merged.RasterYSize == 4
        assert merged.RasterCount == 2
        assert merged.GetGeoTransform() == west.GetGeoTransform()

    def test_keeps_the_data_type_and_band_metadata(self):
        west, east = self._half(170.0, 20), self._half(-180.0, 20)
        west.GetRasterBand(1).SetNoDataValue(-999.0)
        west.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        merged = _wms._merge_lon_halves(west, east, 360.0)
        assert merged.GetRasterBand(1).DataType == gdal.GDT_Int16
        assert merged.GetRasterBand(1).GetNoDataValue() == -999.0
        assert merged.GetRasterBand(1).GetColorInterpretation() == gdal.GCI_RedBand
        assert merged.GetRasterBand(2).ReadAsArray().min() == 20

    def test_keeps_the_colour_table_and_band_metadata(self):
        """A paletted map must not lose its palette because the bbox crossed 180.

        Test scenario:
            WMS is the rendered-map reader, so a single-band paletted PNG is an
            ordinary response. Copying `ColorInterpretation` without the table left
            the merged band declaring `GCI_PaletteIndex` with nothing to look up --
            worse than dropping both -- and the band description, units and
            metadata went with it. A non-wrapping read kept all of it, so the same
            layer rendered differently either side of the seam.
        """
        west = gdal.GetDriverByName("MEM").Create("", 20, 40, 1, gdal.GDT_Byte)
        west.SetGeoTransform((170.0, 0.5, 0.0, 10.0, 0.0, -0.5))
        table = gdal.ColorTable()
        table.SetColorEntry(0, (10, 20, 30, 255))
        table.SetColorEntry(1, (40, 50, 60, 255))
        band = west.GetRasterBand(1)
        band.SetRasterColorTable(table)
        band.SetColorInterpretation(gdal.GCI_PaletteIndex)
        band.SetDescription("landcover")
        band.SetUnitType("class")
        band.SetMetadata({"legend": "corine"})
        west.SetMetadata({"source": "wms"})

        east = gdal.GetDriverByName("MEM").Create("", 10, 40, 1, gdal.GDT_Byte)
        east.SetGeoTransform((-180.0, 0.5, 0.0, 10.0, 0.0, -0.5))

        merged = _wms._merge_lon_halves(west, east, 360.0)
        merged_band = merged.GetRasterBand(1)

        assert merged_band.GetRasterColorTable() is not None, (
            "the palette must survive the stitch; without it the band declares "
            "GCI_PaletteIndex with nothing to look up"
        )
        assert merged_band.GetRasterColorTable().GetColorEntry(1) == (40, 50, 60, 255)
        assert merged_band.GetDescription() == "landcover"
        assert merged_band.GetUnitType() == "class"
        assert merged_band.GetMetadata() == {"legend": "corine"}
        assert merged.GetMetadata() == {"source": "wms"}

    def test_refuses_mismatched_rows_or_bands(self):
        with pytest.raises(ValueError, match="not concatenable"):
            _wms._merge_lon_halves(
                self._half(170.0, 20), self._half(-180.0, 20, rows=3), 360.0
            )

    def test_refuses_different_resolutions(self):
        with pytest.raises(ValueError, match="different resolutions"):
            _wms._merge_lon_halves(
                self._half(170.0, 20), self._half(-180.0, 40, x_res=0.25), 360.0
            )

    def test_refuses_halves_that_do_not_share_a_top_edge(self):
        with pytest.raises(ValueError, match="top edge"):
            _wms._merge_lon_halves(
                self._half(170.0, 20), self._half(-180.0, 20, uly=20.0), 360.0
            )

    def test_refuses_halves_that_do_not_meet_at_the_seam(self):
        """A west half stopping a pixel short of 180 is not stitchable."""
        with pytest.raises(ValueError, match="apart at the seam"):
            _wms._merge_lon_halves(self._half(170.0, 18), self._half(-180.0, 18), 360.0)

    def test_a_wrong_seam_offset_is_caught(self):
        """A projected layer checked against 360 degrees fails rather than stitching."""
        with pytest.raises(ValueError, match="apart at the seam"):
            _wms._merge_lon_halves(self._half(170.0, 20), self._half(-180.0, 20), 1.0)


class _Part:
    """A stand-in for a fetched half that records whether it was closed."""

    def __init__(self, name, closed):
        self.name = name
        self._closed = closed

    def Close(self):  # noqa: N802 - mirrors gdal.Dataset.Close
        self._closed.append(self.name)


class TestCollectHalves:
    """Ownership: what `_collect_halves` closes and what it hands back open."""

    def test_a_single_window_is_handed_back_open(self):
        closed: list[str] = []
        part = _Part("only", closed)
        assert _wms._collect_halves(lambda w: part, ["only"], 360.0) is part
        assert closed == []

    def test_an_empty_window_list_is_refused_by_name(self):
        """The invariant the callers rely on, stated rather than assumed.

        Test scenario:
            With no windows the loop never runs, the single-window branch is false
            and the merge indexes an empty list -- an `IndexError` naming nothing.
            Now that the WMTS reader filters halves by overlap, an empty list is
            reachable, so it fails with a message instead.
        """
        with pytest.raises(ValueError, match="at least one window"):
            _wms._collect_halves(lambda window: None, [], 360.0)

    def test_both_halves_are_closed_once_merged(self, monkeypatch):
        closed: list[str] = []
        monkeypatch.setattr(_wms, "_merge_lon_halves", lambda w, e, off: "stitched")
        result = _wms._collect_halves(
            lambda w: _Part(w, closed), ["west", "east"], 360.0
        )
        assert result == "stitched"
        assert closed == ["west", "east"]

    def test_a_failed_second_fetch_closes_the_first(self):
        closed: list[str] = []

        def fetch(window):
            if window == "east":
                raise WMSError("GetMap failed")
            return _Part(window, closed)

        with pytest.raises(WMSError):
            _wms._collect_halves(fetch, ["west", "east"], 360.0)
        assert closed == ["west"]

    def test_a_failed_merge_still_closes_both(self, monkeypatch):
        closed: list[str] = []

        def boom(*_a):
            raise ValueError("not concatenable")

        monkeypatch.setattr(_wms, "_merge_lon_halves", boom)
        with pytest.raises(ValueError, match="not concatenable"):
            _wms._collect_halves(lambda w: _Part(w, closed), ["west", "east"], 360.0)
        assert closed == ["west", "east"]


def _global_pyramid(res: float = 0.5) -> gdal.Dataset:
    """A whole-world lon/lat raster standing in for an opened WMTS layer."""
    return _longitude_raster(
        -180.0, 90.0, res, -res, (int(360 / res), int(180 / res)), 1
    )


class TestFromWmtsAntimeridian:
    """`from_wmts` windows the pyramid either side of the seam and stitches."""

    @pytest.fixture
    def fake_wmts(self, monkeypatch):
        src = _global_pyramid()
        monkeypatch.setattr(_wms, "_open", lambda *_a: src)
        return src

    def test_wrapping_bbox_is_stitched(self, fake_wmts):
        ds = Dataset.from_wmts("https://c.xml", layer="L", bbox=WRAP, resolution=0.5)
        assert ds.columns == 40
        assert ds.geotransform[0] == pytest.approx(170.0)
        assert ds.geotransform[1] == pytest.approx(0.5)
        row = _bands_rows_columns(ds)[0, 0]
        assert row == pytest.approx(170.0 + (np.arange(40) + 0.5) * 0.5)

    def test_native_resolution_read_also_stitches(self, fake_wmts):
        """`resolution=None` pins both halves to the layer's own grid before splitting."""
        ds = Dataset.from_wmts("https://c.xml", layer="L", bbox=WRAP)
        assert ds.columns == 40
        assert ds.geotransform[1] == pytest.approx(0.5)

    def test_a_half_that_misses_the_layer_is_not_requested(self, monkeypatch):
        """A regional pyramid returns only the half it actually covers.

        Test scenario:
            A layer reaching the seam from the west only. Without the overlap
            filter the eastern half comes back as a block of no-data and is
            concatenated in as though it were data -- while the WCS and OGC API
            Coverages readers drop it. All three now answer the same way.
        """
        regional = _longitude_raster(170.0, 10.0, 0.5, -0.5, (20, 40), 1)
        monkeypatch.setattr(_wms, "_open", lambda *_a: regional)
        ds = Dataset.from_wmts("https://c.xml", layer="L", bbox=WRAP, resolution=0.5)
        assert ds.columns == 20, (
            f"only the western half overlaps, so the result should be 20 columns "
            f"wide, got {ds.columns}"
        )

    def test_a_layer_nowhere_near_the_seam_is_refused(self, monkeypatch):
        """Neither half overlaps, so there is nothing honest to return.

        Test scenario:
            The same refusal `from_wcs` and `from_ogc_coverages` already give. A
            European pyramid cannot serve a wrap; before the filter it returned two
            no-data blocks stitched together.
        """
        regional = _longitude_raster(0.0, 10.0, 0.5, -0.5, (20, 40), 1)
        monkeypatch.setattr(_wms, "_open", lambda *_a: regional)
        with pytest.raises(ValueError, match="neither half overlaps"):
            Dataset.from_wmts("https://c.xml", layer="L", bbox=WRAP, resolution=0.5)

    def test_non_wrapping_bbox_is_unchanged(self, fake_wmts):
        ds = Dataset.from_wmts(
            "https://c.xml", layer="L", bbox=(0.0, -10.0, 10.0, 10.0), resolution=0.5
        )
        assert ds.columns == 20
        assert ds.geotransform[0] == pytest.approx(0.0)

    def test_pixels_match_the_same_two_regions_fetched_separately(self, fake_wmts):
        wrapped = _bands_rows_columns(
            Dataset.from_wmts("https://c.xml", layer="L", bbox=WRAP, resolution=0.5)
        )
        west = _bands_rows_columns(
            Dataset.from_wmts(
                "https://c.xml",
                layer="L",
                bbox=(170.0, -10.0, 180.0, 10.0),
                resolution=0.5,
            )
        )
        east = _bands_rows_columns(
            Dataset.from_wmts(
                "https://c.xml",
                layer="L",
                bbox=(-180.0, -10.0, -170.0, 10.0),
                resolution=0.5,
            )
        )
        assert wrapped[:, :, :20] == pytest.approx(west)
        assert wrapped[:, :, 20:] == pytest.approx(east)


@pytest.mark.slow
@pytest.mark.live
class TestLiveWms:
    """Live end-to-end against public OSM-WMS and NASA GIBS WMTS."""

    OSM = "https://ows.terrestris.de/osm/service?"
    GIBS = (
        "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml"
    )
    TRUECOLOR = "MODIS_Terra_CorrectedReflectance_TrueColor"

    def test_wms_by_size(self):
        ds = Dataset.from_wms(self.OSM, layers="OSM-WMS", bbox=BBOX, size=(256, 256))
        assert ds.shape == (3, 256, 256)
        assert ds.bbox == pytest.approx([5.0, 51.0, 6.0, 52.0])
        assert ds.epsg == 4326

    def test_wms_by_resolution(self):
        ds = Dataset.from_wms(self.OSM, layers="OSM-WMS", bbox=BBOX, resolution=0.02)
        assert ds.shape == (3, 50, 50)

    def test_wmts_crops_bbox_and_resolves_crs84(self):
        ds = Dataset.from_wmts(
            self.GIBS,
            layer=self.TRUECOLOR,
            bbox=BBOX,
            resolution=0.01,
        )
        assert ds.shape[-2:] == (100, 100)
        assert ds.epsg == 4326  # CRS84 is unresolvable -> epsg_from_wkt default 4326
        assert ds.bbox[0] == pytest.approx(5.0, abs=0.05)

    def test_wmts_unknown_layer_lists_available(self):
        with pytest.raises(ValueError, match="not advertised"):
            Dataset.from_wmts(
                self.GIBS,
                layer="NOT_A_REAL_LAYER",
                bbox=BBOX,
                resolution=0.1,
            )

    # A real service is the only thing that can answer whether two GetMap calls
    # either side of 180 come back consistently styled and cache-aligned - the
    # offline suite proves the geometry, not the rendering.
    FIJI = (179.0, -18.5, -179.0, -17.5)

    def test_wms_across_the_antimeridian(self):
        ds = Dataset.from_wms(
            self.OSM, layers="OSM-WMS", bbox=self.FIJI, size=(512, 256)
        )
        assert ds.shape == (3, 256, 512)
        assert ds.bbox[0] == pytest.approx(179.0, abs=0.01)
        assert ds.bbox[2] == pytest.approx(181.0, abs=0.01)

    def test_wmts_across_the_antimeridian(self):
        ds = Dataset.from_wmts(
            self.GIBS, layer=self.TRUECOLOR, bbox=self.FIJI, resolution=0.01
        )
        assert ds.shape[-2:] == (100, 200)
        assert ds.bbox[0] == pytest.approx(179.0, abs=0.05)
