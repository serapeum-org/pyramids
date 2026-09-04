"""ARC-331: one GDAL open and one window-into-MEM, branded per protocol at the call site.

The WCS, WMS / WMTS and OGC API – Coverages readers each wrote the same two GDAL
sequences by hand. Opening was `gdal.Open` (or `OpenEx`) guarded against *both* a
`RuntimeError` and a `None` return; reading was `gdal.Translate("", src, format="MEM")`
guarded the same way. Seven copies of two four-line shapes, and the guard against a
`None` return is exactly the kind of half that goes missing when the eighth is written.

`open_network_dataset` and `translate_to_mem` in `pyramids.base._coverage` now own the
sequence and the classification. What must NOT be shared is the wording: each reader
raises its own error class and names its own protocol, so a caller still learns whether
a WMTS tile read or a WCS GetCoverage failed. These tests pin both halves — the shared
mechanism, and the fact that the four translate sites and three open sites still say
four and three different things.
"""

from __future__ import annotations

import pytest
from osgeo import gdal

from pyramids.base._coverage import open_network_dataset, translate_to_mem
from pyramids.dataset import _ogc_coverages, _wcs, _wms
from pyramids.errors import OGCAPIError, WCSError, WMSError

pytestmark = pytest.mark.core


class _DemoError(Exception):
    """Stand-in for a reader's branded error class (WCSError / WMSError / ...)."""


def _mem_source(size: int = 8) -> gdal.Dataset:
    """A square MEM raster with a north-up, one-unit-per-pixel geotransform."""
    src = gdal.GetDriverByName("MEM").Create("", size, size, 1)
    src.SetGeoTransform((0.0, 1.0, 0.0, float(size), 0.0, -1.0))
    return src


def _raise_runtime(*_args, **_kwargs):
    """Stand in for GDAL failing under `gdal.UseExceptions()`."""
    raise RuntimeError("service said no")


def _return_none(*_args, **_kwargs):
    """Stand in for a driver that declines the source without raising."""
    return None


def _refuse(*_args, **_kwargs):
    """Fail loudly if the wrong GDAL entry point is reached."""
    raise AssertionError("the wrong GDAL entry point was called")


class TestOpenNetworkDataset:
    """The open half: one call, two failure shapes, one branded error."""

    def test_an_openable_connection_comes_back_as_a_dataset(self):
        """Test scenario: the happy path must still hand back GDAL's own handle.

        A ``/vsimem`` GeoTIFF stands in for the network source, so the assertion is
        about the wrapper, not about reaching a server.
        """
        path = "/vsimem/arc331_open_ok.tif"
        writer = gdal.GetDriverByName("GTiff").Create(path, 2, 3, 1)
        writer = None
        try:
            src = open_network_dataset(
                path, error=_DemoError, subject="coverage 'demo'"
            )
            size = (src.RasterXSize, src.RasterYSize)
            src = None
        finally:
            gdal.Unlink(path)

        assert size == (2, 3), f"a different dataset came back than was written: {size}"

    def test_a_gdal_runtimeerror_becomes_the_readers_own_error(self, monkeypatch):
        """Test scenario: under `gdal.UseExceptions()` a refused open raises.

        The raw `RuntimeError` names neither the coverage nor the protocol, so it is
        re-branded — and chained, so the GDAL text is still reachable.
        """
        monkeypatch.setattr(gdal, "Open", _raise_runtime)

        with pytest.raises(_DemoError) as excinfo:
            open_network_dataset("x", error=_DemoError, subject="coverage 'demo'")

        message = str(excinfo.value)
        assert message == "could not open coverage 'demo': service said no", (
            f"the branded open message changed: {message!r}"
        )
        assert isinstance(excinfo.value.__cause__, RuntimeError), (
            "GDAL's own RuntimeError was not chained as __cause__"
        )

    def test_a_none_return_becomes_the_readers_own_error(self, monkeypatch):
        """Test scenario: a driver may decline a source by returning None, not raising.

        This is the half that goes missing when the sequence is hand-written: without
        it the None escapes and fails an attribute access one frame later.
        """
        monkeypatch.setattr(gdal, "Open", _return_none)

        with pytest.raises(_DemoError) as excinfo:
            open_network_dataset("x", error=_DemoError, subject="coverage 'demo'")

        message = str(excinfo.value)
        assert message == "GDAL returned no dataset for coverage 'demo'", (
            f"the branded no-dataset message changed: {message!r}"
        )

    def test_without_open_options_the_plain_gdal_open_is_used(self, monkeypatch):
        """Test scenario: WCS / WMS / WMTS pass a descriptor with no open options."""
        seen: list[str] = []

        def record(connection):
            seen.append(connection)
            return "opened"

        monkeypatch.setattr(gdal, "Open", record)
        monkeypatch.setattr(gdal, "OpenEx", _refuse)

        result = open_network_dataset("<WCS_GDAL/>", error=_DemoError, subject="s")

        assert result == "opened", f"the opened handle was not returned: {result!r}"
        assert seen == ["<WCS_GDAL/>"], (
            f"gdal.Open did not receive the connection string verbatim: {seen!r}"
        )

    def test_open_options_route_through_openex_as_a_raster(self, monkeypatch):
        """Test scenario: OGC API – Coverages needs open options to pick the API mode.

        They must arrive as a list alongside ``OF_RASTER``; a vector-capable open would
        let the driver hand back something that is not a raster at all.
        """
        seen: list[tuple] = []

        def record(connection, flags, open_options):
            seen.append((connection, flags, open_options))
            return "opened"

        monkeypatch.setattr(gdal, "OpenEx", record)
        monkeypatch.setattr(gdal, "Open", _refuse)

        result = open_network_dataset(
            "OGCAPI:x",
            error=_DemoError,
            subject="s",
            open_options=("API=COVERAGE", "CACHE=NO"),
        )

        assert result == "opened", f"the opened handle was not returned: {result!r}"
        assert seen == [("OGCAPI:x", gdal.OF_RASTER, ["API=COVERAGE", "CACHE=NO"])], (
            f"OpenEx was not called with OF_RASTER and the options list: {seen!r}"
        )


class TestTranslateToMem:
    """The read half: the window lands in memory, never on the caller's path."""

    def test_the_projwin_window_is_materialised_in_memory(self):
        """Test scenario: a bounded window of an 8x8 source at native resolution."""
        mem = translate_to_mem(
            _mem_source(),
            error=_DemoError,
            action="demo read",
            subject="'demo'",
            projWin=[2.0, 6.0, 6.0, 2.0],
        )

        assert (mem.RasterXSize, mem.RasterYSize) == (4, 4), (
            f"the projWin window was not honoured: {mem.RasterXSize}x{mem.RasterYSize}"
        )
        assert mem.GetDriver().ShortName == "MEM", (
            f"the window was not materialised in memory: {mem.GetDriver().ShortName}"
        )

    def test_an_explicit_size_bounds_the_allocation(self):
        """Test scenario: the OGC API path caps the read with width / height.

        Without the cap the unbounded virtual raster would allocate petabytes, so the
        two keywords reaching `TranslateOptions` is the whole point.
        """
        mem = translate_to_mem(
            _mem_source(),
            error=_DemoError,
            action="demo read",
            subject="'demo'",
            projWin=[0.0, 8.0, 8.0, 0.0],
            width=3,
            height=5,
        )

        assert (mem.RasterXSize, mem.RasterYSize) == (3, 5), (
            f"width / height did not reach Translate: {mem.RasterXSize}x{mem.RasterYSize}"
        )

    def test_a_gdal_runtimeerror_becomes_the_readers_own_error(self, monkeypatch):
        """Test scenario: an error body reaches GDAL and the translate raises."""
        monkeypatch.setattr(gdal, "Translate", _raise_runtime)

        with pytest.raises(_DemoError) as excinfo:
            translate_to_mem(
                _mem_source(), error=_DemoError, action="demo read", subject="'demo'"
            )

        message = str(excinfo.value)
        assert message == "demo read failed for 'demo': service said no", (
            f"the branded translate-failure message changed: {message!r}"
        )
        assert isinstance(excinfo.value.__cause__, RuntimeError), (
            "GDAL's own RuntimeError was not chained as __cause__"
        )

    def test_a_none_return_becomes_the_readers_own_error(self, monkeypatch):
        """Test scenario: Translate answers None rather than raising."""
        monkeypatch.setattr(gdal, "Translate", _return_none)

        with pytest.raises(_DemoError) as excinfo:
            translate_to_mem(
                _mem_source(), error=_DemoError, action="demo read", subject="'demo'"
            )

        message = str(excinfo.value)
        assert message == "demo read returned no raster for 'demo'", (
            f"the branded no-raster message changed: {message!r}"
        )


class TestEachReaderKeepsItsOwnWording:
    """The sequence is shared. The exception class and the protocol words are not."""

    def test_wcs_open_failure_names_the_wcs_coverage(self, monkeypatch):
        """Test scenario: a WCS service descriptor GDAL refuses."""
        monkeypatch.setattr(gdal, "Open", _raise_runtime)

        with pytest.raises(WCSError) as excinfo:
            _wcs._open_service("<WCS_GDAL/>", "cov")

        message = str(excinfo.value)
        assert message == "could not open WCS coverage 'cov': service said no", (
            f"the WCS open message changed: {message!r}"
        )

    @pytest.mark.parametrize("hint", ["WMS", "WMTS"], ids=["wms", "wmts"])
    def test_wms_open_failure_names_the_protocol_hint_and_layer(
        self, monkeypatch, hint: str
    ):
        """One opener serves both WMS and WMTS, so the hint must reach the message.

        Args:
            hint: The protocol word `from_wms` / `from_wmts` passes in.
        """
        monkeypatch.setattr(gdal, "Open", _raise_runtime)

        with pytest.raises(WMSError) as excinfo:
            _wms._open("conn", "layer", hint)

        message = str(excinfo.value)
        assert message == f"could not open {hint} layer 'layer': service said no", (
            f"the {hint} open message changed: {message!r}"
        )

    def test_ogc_open_failure_names_the_ogc_api_coverage(self, monkeypatch):
        """Test scenario: the OGCAPI driver is present but refuses the connection."""
        monkeypatch.setattr(gdal, "GetDriverByName", lambda _name: object())
        monkeypatch.setattr(gdal, "OpenEx", _raise_runtime)

        with pytest.raises(OGCAPIError) as excinfo:
            _ogc_coverages._open_coverage("OGCAPI:x", "cov")

        message = str(excinfo.value)
        assert message == "could not open OGC API coverage 'cov': service said no", (
            f"the OGC API open message changed: {message!r}"
        )

    def test_a_missing_ogcapi_driver_is_still_reported_before_the_open(
        self, monkeypatch
    ):
        """The driver-presence check is OGC-API-only and stays at that call site."""
        monkeypatch.setattr(gdal, "GetDriverByName", lambda _name: None)
        monkeypatch.setattr(gdal, "OpenEx", _refuse)

        with pytest.raises(OGCAPIError, match="OGCAPI driver is not available"):
            _ogc_coverages._open_coverage("OGCAPI:x", "cov")

    def test_wcs_window_failure_says_getcoverage(self, monkeypatch):
        """Test scenario: the windowed GetCoverage fails inside Translate."""
        src = _mem_source()
        monkeypatch.setattr(gdal, "Translate", _raise_runtime)

        with pytest.raises(WCSError) as excinfo:
            _wcs._translate_window(src, [0.0, 8.0, 8.0, 0.0], "cov")

        message = str(excinfo.value)
        assert message == "WCS GetCoverage failed for 'cov': service said no", (
            f"the WCS window message changed: {message!r}"
        )

    def test_wmts_window_failure_says_tile_read(self, monkeypatch):
        """Test scenario: the WMTS crop fails inside Translate."""
        src = _mem_source()
        monkeypatch.setattr(gdal, "Translate", _raise_runtime)

        with pytest.raises(WMSError) as excinfo:
            _wms._translate_window(src, [0.0, 8.0, 8.0, 0.0], "layer", None, "near")

        message = str(excinfo.value)
        assert message == "WMTS tile read failed for 'layer': service said no", (
            f"the WMTS window message changed: {message!r}"
        )

    def test_wms_render_failure_says_getmap(self, monkeypatch):
        """Test scenario: the GetMap request fires during Translate and fails."""
        src = _mem_source()
        monkeypatch.setattr(gdal, "Translate", _raise_runtime)

        with pytest.raises(WMSError) as excinfo:
            _wms._render_wms(src, "OSM-WMS")

        message = str(excinfo.value)
        assert message == "WMS GetMap failed for 'OSM-WMS': service said no", (
            f"the WMS render message changed: {message!r}"
        )

    def test_ogc_window_failure_says_coverage_read(self, monkeypatch):
        """Test scenario: the size-capped OGC API read fails inside Translate."""
        src = _mem_source()
        monkeypatch.setattr(gdal, "Translate", _raise_runtime)

        with pytest.raises(OGCAPIError) as excinfo:
            _ogc_coverages._translate_window(src, [0.0, 8.0, 8.0, 0.0], (4, 4), "cov")

        message = str(excinfo.value)
        assert message == "OGC API coverage read failed for 'cov': service said no", (
            f"the OGC API window message changed: {message!r}"
        )


class TestTheWindowArgumentsSurvivedTheMove:
    """Each reader's own Translate keywords still reach GDAL after the consolidation."""

    def test_wmts_resolution_is_forwarded_as_xres_yres(self):
        """A WMTS resolution picks the overview level, so it must not be dropped.

        Test scenario:
            An 8-unit-wide window at a 2-unit resolution is 4 px. Reading 8 px would
            mean `xRes` / `yRes` never reached `TranslateOptions`.
        """
        mem = _wms._translate_window(
            _mem_source(), [0.0, 8.0, 8.0, 0.0], "layer", (2.0, 2.0), "near"
        )

        assert (mem.RasterXSize, mem.RasterYSize) == (4, 4), (
            f"the WMTS resolution was dropped: {mem.RasterXSize}x{mem.RasterYSize}"
        )

    def test_ogc_size_cap_is_forwarded_as_width_height(self):
        """The OGC API read is unusable without its explicit size cap.

        Test scenario:
            A full-extent window read with a (3, 5) cap must come back 3x5, not at the
            source's own 8x8.
        """
        mem = _ogc_coverages._translate_window(
            _mem_source(), [0.0, 8.0, 8.0, 0.0], (3, 5), "cov"
        )

        assert (mem.RasterXSize, mem.RasterYSize) == (3, 5), (
            f"the OGC API size cap was dropped: {mem.RasterXSize}x{mem.RasterYSize}"
        )

    def test_wcs_window_reads_at_the_sources_native_resolution(self):
        """WCS passes only a projWin; the size follows from the coverage's own grid.

        Test scenario:
            The 8x8 one-unit source windowed to 4 units must come back 4x4 — a size
            keyword leaking in from a neighbouring reader would change that.
        """
        mem = _wcs._translate_window(_mem_source(), [2.0, 6.0, 6.0, 2.0], "cov")

        assert (mem.RasterXSize, mem.RasterYSize) == (4, 4), (
            f"the WCS native-resolution window changed: "
            f"{mem.RasterXSize}x{mem.RasterYSize}"
        )
