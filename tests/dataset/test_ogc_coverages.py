"""Tests for the OGC API – Coverages reader (`pyramids.dataset._ogc_coverages`).

Network-free. The successful read drives GDAL's native ``OGCAPI`` raster driver,
which needs a live service; that path is covered only by the gated live test.
Elsewhere ``gdal.OpenEx`` is monkeypatched to return a small in-memory raster so
``from_ogc_coverages``'s own logic — coverage validation, the connection string,
the open options, the Dataset wrapping, ``output_crs`` / ``resolution`` warps and
error normalisation — is covered without a live service, plus the pure helpers.
"""

from __future__ import annotations

import os

import pytest
from osgeo import gdal, osr

from pyramids.base import _ogc_api
from pyramids.dataset import Dataset
from pyramids.dataset import _ogc_coverages
from pyramids.errors import OGCAPIError


@pytest.fixture(autouse=True)
def _clear_collections_cache():
    """Isolate the shared /collections LRU cache between tests."""
    _ogc_api.get_collections.cache_clear()
    yield
    _ogc_api.get_collections.cache_clear()


def _mem_ds(epsg: int | None = 4326, cols: int = 10, rows: int = 10) -> gdal.Dataset:
    """A small in-memory raster standing in for a fetched coverage."""
    ds = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((5.0, 0.1, 0.0, 52.0, 0.0, -0.1))
    if epsg is not None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        ds.SetSpatialRef(srs)
    return ds


class TestPureHelpers:
    def test_coverage_connection_strips_trailing_slash(self):
        assert (
            _ogc_coverages._coverage_connection("https://h/ogc/", "cov")
            == "OGCAPI:https://h/ogc/collections/cov"
        )
        assert (
            _ogc_coverages._coverage_connection("https://h/ogc", "cov")
            == "OGCAPI:https://h/ogc/collections/cov"
        )

    def test_open_options_without_bbox(self):
        assert _ogc_coverages._open_options(None) == ["API=COVERAGE", "IMAGE_FORMAT=GEOTIFF"]

    def test_open_options_with_bbox(self):
        opts = _ogc_coverages._open_options((5.0, 51.0, 6.0, 52.0))
        assert opts[:2] == ["API=COVERAGE", "IMAGE_FORMAT=GEOTIFF"]
        assert opts[2:] == ["MINX=5.0", "MINY=51.0", "MAXX=6.0", "MAXY=52.0"]

    def test_resolution_pair(self):
        assert _ogc_coverages._resolution_pair(None) is None
        assert _ogc_coverages._resolution_pair(250) == (250.0, 250.0)
        assert _ogc_coverages._resolution_pair((250, 500)) == (250.0, 500.0)

    def test_validate_bbox_ok(self):
        assert _ogc_coverages._validate_bbox((5.0, 51.0, 6.0, 52.0)) == (5.0, 51.0, 6.0, 52.0)

    @pytest.mark.parametrize(
        "bad",
        [(1, 2, 3), (6.0, 51.0, 5.0, 52.0), (5.0, 52.0, 6.0, 51.0)],
    )
    def test_validate_bbox_rejects(self, bad):
        with pytest.raises(ValueError):
            _ogc_coverages._validate_bbox(bad)


class TestOpenCoverage:
    def test_gdal_runtimeerror_raises_ogcapierror(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("gdal could not open")

        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", boom)
        with pytest.raises(OGCAPIError, match="could not open OGC API coverage"):
            _ogc_coverages._open_coverage("OGCAPI:x", ["API=COVERAGE"], "cov")

    def test_gdal_none_raises_ogcapierror(self, monkeypatch):
        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", lambda *a, **k: None)
        with pytest.raises(OGCAPIError, match="no dataset"):
            _ogc_coverages._open_coverage("OGCAPI:x", ["API=COVERAGE"], "cov")


class TestFromOgcCoverages:
    def _patch_collections(self, monkeypatch, ids=("cov",)):
        monkeypatch.setattr(_ogc_coverages, "_get_collections", lambda *a, **k: frozenset(ids))

    def _patch_openex(self, monkeypatch, captured, ds_factory=_mem_ds):
        def fake_openex(connection, *flags, **kwargs):
            captured["connection"] = connection
            captured["flags"] = flags
            captured["open_options"] = kwargs.get("open_options")
            return ds_factory()

        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", fake_openex)

    def test_returns_dataset(self, monkeypatch):
        """A successful read is wrapped into a Dataset in the coverage's native CRS."""
        self._patch_collections(monkeypatch)
        self._patch_openex(monkeypatch, {})
        ds = Dataset.from_ogc_coverages("https://h/ogc", coverage="cov")
        assert isinstance(ds, Dataset)
        assert ds.epsg == 4326

    def test_connection_and_default_open_options(self, monkeypatch):
        """Without a bbox the connection + the two pinned open options reach OpenEx."""
        self._patch_collections(monkeypatch)
        captured = {}
        self._patch_openex(monkeypatch, captured)
        Dataset.from_ogc_coverages("https://h/ogc", coverage="cov")
        assert captured["connection"] == "OGCAPI:https://h/ogc/collections/cov"
        assert captured["flags"] == (gdal.OF_RASTER,)
        assert captured["open_options"] == ["API=COVERAGE", "IMAGE_FORMAT=GEOTIFF"]

    def test_bbox_adds_subset_open_options(self, monkeypatch):
        """A bbox becomes the MINX/MINY/MAXX/MAXY subset open options."""
        self._patch_collections(monkeypatch)
        captured = {}
        self._patch_openex(monkeypatch, captured)
        Dataset.from_ogc_coverages(
            "https://h/ogc", coverage="cov", bbox=(5.0, 51.0, 6.0, 52.0)
        )
        assert captured["open_options"] == [
            "API=COVERAGE", "IMAGE_FORMAT=GEOTIFF",
            "MINX=5.0", "MINY=51.0", "MAXX=6.0", "MAXY=52.0",
        ]

    def test_auth_and_timeout_active_during_open(self, monkeypatch):
        """The coverage read runs inside a GDAL config context carrying auth + timeout."""
        self._patch_collections(monkeypatch)
        seen = {}

        def fake_openex(connection, *flags, **kwargs):
            seen["userpwd"] = _ogc_coverages.gdal.GetConfigOption("GDAL_HTTP_USERPWD")
            seen["timeout"] = _ogc_coverages.gdal.GetConfigOption("GDAL_HTTP_TIMEOUT")
            return _mem_ds()

        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", fake_openex)
        Dataset.from_ogc_coverages(
            "https://h/ogc", coverage="cov", auth=("u", "p"), timeout=42.0
        )
        assert seen["userpwd"] == "u:p"
        assert seen["timeout"] == "42"

    def test_unknown_coverage_raises_valueerror(self, monkeypatch):
        self._patch_collections(monkeypatch, ids=("other",))
        with pytest.raises(ValueError, match="not advertised"):
            Dataset.from_ogc_coverages("https://h/ogc", coverage="cov")

    def test_empty_collections_skips_validation(self, monkeypatch):
        """An empty /collections set (service advertises none) does not block the read."""
        self._patch_collections(monkeypatch, ids=())
        self._patch_openex(monkeypatch, {})
        ds = Dataset.from_ogc_coverages("https://h/ogc", coverage="cov")
        assert isinstance(ds, Dataset)

    def test_bad_bbox_raises_before_network(self, monkeypatch):
        """An inverted bbox is rejected before any /collections or OpenEx call."""
        def fail(*a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("network must not be touched")

        monkeypatch.setattr(_ogc_coverages, "_get_collections", fail)
        with pytest.raises(ValueError, match="minx < maxx"):
            Dataset.from_ogc_coverages(
                "https://h/ogc", coverage="cov", bbox=(6.0, 51.0, 5.0, 52.0)
            )

    def test_openex_none_raises_ogcapierror(self, monkeypatch):
        self._patch_collections(monkeypatch)
        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", lambda *a, **k: None)
        with pytest.raises(OGCAPIError, match="no dataset"):
            Dataset.from_ogc_coverages("https://h/ogc", coverage="cov")

    def test_openex_runtimeerror_raises_ogcapierror(self, monkeypatch):
        self._patch_collections(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("driver said no")

        monkeypatch.setattr(_ogc_coverages.gdal, "OpenEx", boom)
        with pytest.raises(OGCAPIError, match="could not open"):
            Dataset.from_ogc_coverages("https://h/ogc", coverage="cov")

    def test_output_crs_reprojects(self, monkeypatch):
        """output_crs warps the fetched coverage into the requested CRS."""
        self._patch_collections(monkeypatch)
        self._patch_openex(monkeypatch, {})
        ds = Dataset.from_ogc_coverages(
            "https://h/ogc", coverage="cov", output_crs="EPSG:3857"
        )
        assert ds.epsg == 3857

    def test_resolution_resamples_in_native(self, monkeypatch):
        """A resolution with no output_crs resamples within the native CRS to a coarser grid."""
        self._patch_collections(monkeypatch)
        self._patch_openex(monkeypatch, {})
        ds = Dataset.from_ogc_coverages(
            "https://h/ogc", coverage="cov", resolution=0.5
        )
        assert ds.epsg == 4326
        assert ds.shape[1] < 10 and ds.shape[2] < 10  # coarser than the 0.1deg 10x10 native grid

    def test_no_srs_no_warp_returns_dataset(self, monkeypatch):
        """A coverage without a CRS and no reprojection still returns a Dataset."""
        self._patch_collections(monkeypatch)
        self._patch_openex(monkeypatch, {}, ds_factory=lambda: _mem_ds(epsg=None))
        ds = Dataset.from_ogc_coverages("https://h/ogc", coverage="cov")
        assert isinstance(ds, Dataset)

    def test_output_writes_a_reopenable_file(self, monkeypatch, tmp_path):
        self._patch_collections(monkeypatch)
        self._patch_openex(monkeypatch, {})
        out = tmp_path / "coverage_out.tif"
        Dataset.from_ogc_coverages("https://h/ogc", coverage="cov", output=out)
        assert out.exists()
        assert Dataset.read_file(str(out)).shape == (1, 10, 10)


@pytest.mark.slow
@pytest.mark.live
class TestLiveOgcCoverages:
    ENDPOINT = "https://maps.gnosis.earth/ogcapi"

    def test_live_read(self):
        """Exercise the real GDAL OGCAPI raster driver against a public coverage."""
        coverage = os.environ.get("PYRAMIDS_OGC_COVERAGES_NAME", "SRTM_ViewFinderPanorama")
        ds = Dataset.from_ogc_coverages(
            self.ENDPOINT, coverage=coverage, bbox=(5.0, 51.0, 6.0, 52.0)
        )
        assert isinstance(ds, Dataset)
        assert ds.shape[0] >= 1
        assert ds.shape[1] > 0 and ds.shape[2] > 0
