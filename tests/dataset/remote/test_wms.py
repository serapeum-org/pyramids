"""Tests for the OGC WMS / WMTS reader (`pyramids.dataset._wms`).

Network-free except the gated live class. The pure helpers — output-size
resolution, the ``<GDAL_WMS>`` descriptor, the ``WMTS:`` connection string, the
layers normalisation, and the ``from_wms`` "needs a size" guard — are covered
offline; a live end-to-end against public OSM-WMS and NASA GIBS WMTS runs only
under ``-m live``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, _wms
from pyramids.errors import WMSError

pytestmark = pytest.mark.core

BBOX = (5.0, 51.0, 6.0, 52.0)


def _tiny_dataset(epsg: int = 4326) -> Dataset:
    """A 3-band 4x5 in-memory raster for offline reprojection tests."""
    arr = np.ones((3, 4, 5), dtype="float32")
    return Dataset.create_from_array(
        arr, top_left_corner=(0.0, 0.0), cell_size=0.1, epsg=epsg
    )


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
        with pytest.raises(ValueError, match="minx < maxx"):
            Dataset.from_wms(
                "https://host/wms?",
                layers="L",
                bbox=(6.0, 51.0, 5.0, 52.0),
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
            def GetMetadata(_key):
                return {
                    "SUBDATASET_1_NAME": "WMTS:https://x,layer=B",
                    "SUBDATASET_2_NAME": "WMTS:https://x,layer=A,tilematrixset=t",
                    "SUBDATASET_1_DESC": "ignored, not a _NAME",
                }

        monkeypatch.setattr(_wms.gdal, "Open", lambda _c: _Caps())
        assert _wms._available_wmts_layers("https://x") == ["A", "B"]

    def test_returns_empty_when_open_fails(self, monkeypatch):
        """A capabilities open failure yields [] (never masks the real error)."""

        def boom(_c):
            raise RuntimeError("offline")

        monkeypatch.setattr(_wms.gdal, "Open", boom)
        assert _wms._available_wmts_layers("https://x") == []


@pytest.mark.slow
@pytest.mark.live
class TestLiveWms:
    """Live end-to-end against public OSM-WMS and NASA GIBS WMTS."""

    OSM = "https://ows.terrestris.de/osm/service?"
    GIBS = (
        "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/"
        "WMTSCapabilities.xml"
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
