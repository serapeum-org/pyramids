"""Tests for the private OGR bridge ``pyramids.feature._ogr``.

The module exposes four internal helpers used by ``FeatureCollection``
methods that need a GDAL/OGR object without leaking one into the
public API:

- ``_new_vsimem_path`` — generates unique ``/vsimem/`` paths.
- ``as_datasource`` — context manager yielding an ``ogr.DataSource`` or
  ``gdal.Dataset`` for a ``GeoDataFrame``.
- ``as_vsimem_path`` — context manager yielding a ``/vsimem/`` path a
  GDAL API can read from.
- ``datasource_to_gdf`` — materializes an OGR DataSource back into a
  ``GeoDataFrame``.

These tests cover shape/uniqueness invariants, both ``gdal_dataset``
toggles on ``as_datasource``, cleanup guarantees (paths unlinked even
when the ``with`` block raises), path validity inside the ``with``
block for ``as_vsimem_path``, and round-trip fidelity via a Polygonize-
style OGR DataSource for ``datasource_to_gdf``.
"""

from __future__ import annotations

import re

import geopandas as gpd
import numpy as np
import pytest
from geopandas import GeoDataFrame
from osgeo import gdal, ogr, osr
from shapely.geometry import Point, Polygon

from pyramids.feature import FeatureCollection
from pyramids.feature._ogr import (
    _new_vsimem_path,
    as_datasource,
    as_vsimem_path,
    datasource_to_gdf,
)

pytestmark = pytest.mark.core


@pytest.fixture
def point_gdf() -> GeoDataFrame:
    """GeoDataFrame with three points and an integer attribute.

    Returns:
        GeoDataFrame: Points at (0,0), (1,1), (2,2) with EPSG:4326.
    """
    return gpd.GeoDataFrame(
        {"score": [10, 20, 30]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )


@pytest.fixture
def polygon_gdf() -> GeoDataFrame:
    """GeoDataFrame with a single square polygon.

    Returns:
        GeoDataFrame: One unit square with EPSG:4326 and a ``value`` column.
    """
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    return gpd.GeoDataFrame({"value": [42]}, geometry=[poly], crs="EPSG:4326")


@pytest.fixture
def polygonize_datasource() -> ogr.DataSource:
    """Produce a Polygonize-style in-memory OGR DataSource.

    Mirrors the internal pattern used by ``Dataset._band_to_polygon``:
    an in-memory OGR DataSource holding a single-layer polygon
    FeatureCollection produced by running :func:`gdal.Polygonize` over
    a tiny raster band.

    Returns:
        ogr.DataSource: In-memory DataSource with one layer named
        ``"poly"`` and a single ``value`` integer field. Two polygons
        are created from a 2×2 band where the two columns have
        different constant values.
    """
    # Build a 2-row × 2-col Byte raster in memory with values 1 and 2.
    mem_driver = gdal.GetDriverByName("MEM")
    raster = mem_driver.Create("", 2, 2, 1, gdal.GDT_Byte)
    band = raster.GetRasterBand(1)
    band.WriteArray(np.array([[1, 2], [1, 2]], dtype=np.uint8))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    ds = ogr.GetDriverByName("Memory").CreateDataSource("scratch")
    layer = ds.CreateLayer("poly", srs=srs)
    layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))
    gdal.Polygonize(band, band, layer, 0, [], callback=None)
    return ds


class TestNewVsimemPath:
    """Tests for :func:`_new_vsimem_path`."""

    def test_starts_with_vsimem_prefix(self):
        """The path must start with ``/vsimem/`` so GDAL routes it to VFS.

        Test scenario:
            Without the ``/vsimem/`` prefix GDAL would attempt to touch
            the real filesystem, which would defeat the zero-disk-I/O
            guarantee of the bridge.
        """
        path = _new_vsimem_path()
        assert path.startswith(
            "/vsimem/"
        ), f"Path must start with /vsimem/, got {path!r}"

    def test_ends_with_geojson_extension(self):
        """The path must end with ``.geojson`` so GDAL picks the GeoJSON driver.

        Test scenario:
            GDAL infers the driver from the extension when none is
            explicitly specified.
        """
        path = _new_vsimem_path()
        assert path.endswith(".geojson"), f"Path must end with .geojson, got {path!r}"

    def test_stem_format(self):
        """The stem is ``<time_ns>_<rand>`` (C35 — UUID4 replaced).

        Test scenario:
            Shorter than UUID4 but still collision-proof per-process:
            nanosecond timestamp + 20-bit random tie-break.
        """
        path = _new_vsimem_path()
        stem = path[len("/vsimem/") : -len(".geojson")]
        stem_re = re.compile(r"^\d+_\d+$")
        assert stem_re.match(stem), f"unexpected stem format: {stem!r}"

    def test_unique_across_calls(self):
        """Successive calls must return distinct paths.

        Test scenario:
            Uniqueness is the guard against concurrent conversions
            clobbering each other's backing files.
        """
        paths = {_new_vsimem_path() for _ in range(50)}
        assert len(paths) == 50, f"Expected 50 unique paths, got {len(paths)}"


class TestAsDatasource:
    """Tests for :func:`as_datasource`."""

    def test_yields_ogr_datasource_by_default(self, point_gdf):
        """Default ``gdal_dataset=False`` must yield an ``ogr.DataSource``.

        Test scenario:
            The default shape is ``ogr.DataSource`` because most internal
            callers (``gdal.Warp`` cutline, OGR SQL) expect it.
        """
        with as_datasource(point_gdf) as ds:
            assert isinstance(
                ds, ogr.DataSource
            ), f"Expected ogr.DataSource, got {type(ds).__name__}"

    def test_yields_gdal_dataset_when_requested(self, point_gdf):
        """``gdal_dataset=True`` must yield a ``gdal.Dataset``.

        Test scenario:
            :func:`gdal.Rasterize` requires ``gdal.Dataset`` when its
            vector argument is a Python GDAL object rather than a path.
        """
        with as_datasource(point_gdf, gdal_dataset=True) as ds:
            assert isinstance(
                ds, gdal.Dataset
            ), f"Expected gdal.Dataset, got {type(ds).__name__}"

    def test_preserves_feature_count(self, point_gdf):
        """The yielded DataSource must expose every input feature.

        Test scenario:
            Three input points → three OGR features.
        """
        with as_datasource(point_gdf) as ds:
            n = ds.GetLayer(0).GetFeatureCount()
        assert n == 3, f"Expected 3 features, got {n}"

    def test_preserves_attribute_values(self, point_gdf):
        """Per-feature attribute values must survive the bridge.

        Test scenario:
            Read back the ``score`` attribute from the OGR layer and
            confirm we see {10, 20, 30}.
        """
        with as_datasource(point_gdf) as ds:
            layer = ds.GetLayer(0)
            scores = [feat.GetField("score") for feat in layer]
        assert sorted(scores) == [
            10,
            20,
            30,
        ], f"Attribute round-trip changed values: {scores}"

    def test_accepts_featurecollection_subclass(self, point_gdf):
        """A ``FeatureCollection`` (GDF subclass) must work unchanged.

        Test scenario:
            The bridge type-annotates ``GeoDataFrame`` but
            ``FeatureCollection`` IS a ``GeoDataFrame`` after ARC-1a.
        """
        fc = FeatureCollection(point_gdf)
        with as_datasource(fc) as ds:
            n = ds.GetLayer(0).GetFeatureCount()
        assert n == 3, f"FeatureCollection bridge failed: {n} features"

    def test_vsimem_path_unlinked_on_normal_exit(self, point_gdf, monkeypatch):
        """After a normal ``with`` exit the ``/vsimem/`` file is unlinked.

        Test scenario:
            Capture the path generated by the context manager, let the
            block exit normally, then confirm the path is no longer
            resolvable in ``osgeo.gdal``'s VFS.
        """
        captured: list[str] = []
        real_new = _new_vsimem_path

        def _capture() -> str:
            path = real_new()
            captured.append(path)
            return path

        monkeypatch.setattr("pyramids.feature._ogr._new_vsimem_path", _capture)

        with as_datasource(point_gdf):
            pass

        assert len(captured) == 1, "Expected exactly one path generation"
        # After unlink, VSIStatL returns None.
        assert (
            gdal.VSIStatL(captured[0]) is None
        ), f"/vsimem/ path was not unlinked: {captured[0]}"

    def test_vsimem_path_unlinked_on_exception(self, point_gdf, monkeypatch):
        """The ``/vsimem/`` file is unlinked even if the body raises.

        Test scenario:
            Capture the path, raise inside the ``with`` block, and
            confirm the path is still cleaned up afterwards.
        """
        captured: list[str] = []
        real_new = _new_vsimem_path

        def _capture() -> str:
            path = real_new()
            captured.append(path)
            return path

        monkeypatch.setattr("pyramids.feature._ogr._new_vsimem_path", _capture)

        with pytest.raises(RuntimeError, match="boom"):
            with as_datasource(point_gdf):
                raise RuntimeError("boom")

        assert len(captured) == 1
        assert (
            gdal.VSIStatL(captured[0]) is None
        ), f"/vsimem/ path leaked after exception: {captured[0]}"

    def test_preserves_crs(self, point_gdf):
        """CRS of the yielded OGR layer must match the input GeoDataFrame.

        Test scenario:
            EPSG:4326 input → EPSG:4326 on the OGR layer (resolved via
            ``AutoIdentifyEPSG``).
        """
        with as_datasource(point_gdf) as ds:
            layer = ds.GetLayer(0)
            srs = layer.GetSpatialRef()
            srs.AutoIdentifyEPSG()
            code = int(srs.GetAuthorityCode(None))
        assert code == 4326, f"Expected EPSG 4326, got {code}"


class TestAsDatasourceExceptionSafety:
    """C4: ``as_datasource`` is exception-safe on every failure path.

    The context manager must (1) raise :class:`VectorDriverError` with a
    typed message rather than yielding ``None`` when GDAL fails to open
    the in-memory GeoJSON, and (2) only call ``gdal.Unlink`` when the
    in-memory file was actually written. A raise before the
    ``FileFromMemBuffer`` call must not leave a leaked path or trigger a
    spurious Unlink on a non-existent path.
    """

    def test_none_open_becomes_vector_driver_error(self, point_gdf, monkeypatch):
        """If ``ogr.Open`` returns ``None``, raise ``VectorDriverError``.

        Test scenario:
            Simulate a GDAL open failure by monkey-patching
            ``ogr.Open`` to return ``None``. The context manager must
            raise ``VectorDriverError`` rather than yielding ``None``
            (which would surface later as an opaque
            ``AttributeError``).
        """
        from osgeo import ogr as _ogr_mod

        from pyramids.base._errors import VectorDriverError

        monkeypatch.setattr(_ogr_mod, "Open", lambda _: None)

        with pytest.raises(VectorDriverError, match="could not open"):
            with as_datasource(point_gdf):
                pass  # pragma: no cover - unreachable when the open fails

    def test_no_unlink_when_serialization_fails(self, monkeypatch):
        """If ``gdf.to_json`` raises, no vsimem file is ever written.

        Test scenario:
            Simulate a serialization failure before the vsimem file is
            written. The context manager must propagate the exception
            without calling ``gdal.Unlink`` on a path that was never
            created.
        """
        unlinked_paths: list[str] = []
        from osgeo import gdal as _gdal_mod

        real_unlink = _gdal_mod.Unlink
        monkeypatch.setattr(
            _gdal_mod,
            "Unlink",
            lambda p: unlinked_paths.append(p) or real_unlink(p),
        )

        class _BadGDF:
            @property
            def crs(self):
                return None

            def to_json(self):
                raise RuntimeError("simulated serialization failure")

        with pytest.raises(RuntimeError, match="simulated"):
            with as_datasource(_BadGDF()):
                pass  # pragma: no cover - unreachable when to_json fails

        assert unlinked_paths == [], (
            f"gdal.Unlink was called on {unlinked_paths}; expected no "
            f"cleanup calls when the in-memory file was never written."
        )

    def test_vector_driver_error_message_names_vsimem_path(
        self, point_gdf, monkeypatch
    ):
        """C4: the ``VectorDriverError`` message includes the vsimem path.

        Test scenario:
            On an OGR open failure, the typed exception message must
            include the ``/vsimem/<uuid>.geojson`` path so debuggers can
            grep it out of a traceback. Tight coupling test — change
            the wording and this fails loudly.
        """
        from osgeo import ogr as _ogr_mod

        from pyramids.base._errors import VectorDriverError

        monkeypatch.setattr(_ogr_mod, "Open", lambda _: None)

        with pytest.raises(VectorDriverError) as exc_info:
            with as_datasource(point_gdf):
                pass  # pragma: no cover
        msg = str(exc_info.value)
        assert (
            "/vsimem/" in msg
        ), f"VectorDriverError message must name the vsimem path; got: {msg}"
        assert ".geojson" in msg, f"expected .geojson in message; got: {msg}"

    def test_gdal_dataset_none_open_also_becomes_vector_driver_error(
        self, point_gdf, monkeypatch
    ):
        """C4: the ``gdal_dataset=True`` branch also raises on None-open.

        Test scenario:
            The OGR fallback path and the ``gdal.OpenEx`` path go
            through the same None-check. This pins that the
            ``gdal_dataset=True`` toggle does not bypass the guard.
        """
        from osgeo import gdal as _gdal_mod

        from pyramids.base._errors import VectorDriverError

        monkeypatch.setattr(_gdal_mod, "OpenEx", lambda _: None)

        with pytest.raises(VectorDriverError):
            with as_datasource(point_gdf, gdal_dataset=True):
                pass  # pragma: no cover

    def test_unlinks_on_exception_inside_with_body(self, point_gdf, monkeypatch):
        """C4: the vsimem file is unlinked even when user code raises.

        Test scenario:
            The context manager's cleanup is the single authoritative
            release point. When the ``with`` block raises, the vsimem
            path must still be unlinked exactly once.
        """
        from osgeo import gdal as _gdal_mod

        calls: list[str] = []
        real_unlink = _gdal_mod.Unlink
        monkeypatch.setattr(
            _gdal_mod,
            "Unlink",
            lambda p: calls.append(p) or real_unlink(p),
        )

        with pytest.raises(ValueError, match="user-code failure"):
            with as_datasource(point_gdf):
                raise ValueError("user-code failure")

        assert len(calls) == 1, f"expected exactly one Unlink call; got {calls}"
        assert calls[0].startswith("/vsimem/"), calls[0]


class TestAsVsimemPath:
    """Tests for :func:`as_vsimem_path`."""

    def test_yields_str_path_with_prefix(self, polygon_gdf):
        """The yielded value is a string beginning with ``/vsimem/``.

        Test scenario:
            Callers pass the yielded string straight into GDAL APIs
            like ``cutlineDSName`` which expect a path string.
        """
        with as_vsimem_path(polygon_gdf) as path:
            assert isinstance(path, str), f"Expected str, got {type(path).__name__}"
            assert path.startswith(
                "/vsimem/"
            ), f"Path must start with /vsimem/, got {path!r}"

    def test_path_is_openable_by_osgeo(self, polygon_gdf):
        """Inside the ``with`` block, ``osgeo.ogr.Open`` must succeed.

        Test scenario:
            The bridge uses ``gdal.FileFromMemBuffer`` so the file
            lands in ``osgeo.gdal``'s VFS (not pyogrio's). We read it
            back with ``osgeo.ogr.Open`` to confirm the write target.
        """
        with as_vsimem_path(polygon_gdf) as path:
            ds = ogr.Open(path)
            assert ds is not None, "ogr.Open returned None for yielded path"
            n = ds.GetLayer(0).GetFeatureCount()
            ds = None
        assert n == 1, f"Expected 1 feature, got {n}"

    def test_path_unlinked_after_normal_exit(self, polygon_gdf):
        """The ``/vsimem/`` path is unlinked after the block exits.

        Test scenario:
            After a normal exit, ``VSIStatL`` on the captured path
            returns ``None`` — the file is gone.
        """
        captured: str | None = None
        with as_vsimem_path(polygon_gdf) as path:
            captured = path
        assert captured is not None
        assert (
            gdal.VSIStatL(captured) is None
        ), f"/vsimem/ path was not unlinked: {captured}"

    def test_path_unlinked_on_exception(self, polygon_gdf):
        """The ``/vsimem/`` path is unlinked even when the body raises.

        Test scenario:
            Raise inside the block, confirm the path is still gone.
        """
        captured: list[str] = []
        with pytest.raises(ValueError, match="bad"):
            with as_vsimem_path(polygon_gdf) as path:
                captured.append(path)
                raise ValueError("bad")
        assert len(captured) == 1
        assert (
            gdal.VSIStatL(captured[0]) is None
        ), f"/vsimem/ path leaked after exception: {captured[0]}"

    def test_accepts_featurecollection_subclass(self, polygon_gdf):
        """A ``FeatureCollection`` is accepted transparently.

        Test scenario:
            FeatureCollection inherits from GeoDataFrame, so the bridge
            must handle it without any type branching.
        """
        fc = FeatureCollection(polygon_gdf)
        with as_vsimem_path(fc) as path:
            ds = ogr.Open(path)
            n = ds.GetLayer(0).GetFeatureCount()
            ds = None
        assert n == 1


class TestDatasourceToGdf:
    """Tests for :func:`datasource_to_gdf`."""

    def test_returns_geodataframe(self, point_gdf):
        """The return type is a plain ``GeoDataFrame``, never a subclass.

        Test scenario:
            The docstring promises a plain GDF so callers who want a
            ``FeatureCollection`` can decide whether to wrap.
        """
        with as_datasource(point_gdf) as ds:
            result = datasource_to_gdf(ds)
        assert isinstance(
            result, GeoDataFrame
        ), f"Expected GeoDataFrame, got {type(result).__name__}"

    def test_preserves_row_count_round_trip(self, point_gdf):
        """Row count survives a GeoDataFrame → DataSource → GeoDataFrame trip.

        Test scenario:
            3-row input → 3-row output.
        """
        with as_datasource(point_gdf) as ds:
            result = datasource_to_gdf(ds)
        assert len(result) == 3, f"Row count changed: {len(result)} != 3"

    def test_preserves_attributes_round_trip(self, point_gdf):
        """Attribute column values survive the OGR round-trip.

        Test scenario:
            ``score`` column values {10, 20, 30} must come back intact.
        """
        with as_datasource(point_gdf) as ds:
            result = datasource_to_gdf(ds)
        assert sorted(result["score"].tolist()) == [
            10,
            20,
            30,
        ], f"Attribute values changed: {result['score'].tolist()}"

    def test_preserves_crs_round_trip(self, point_gdf):
        """CRS (EPSG:4326) survives the OGR round-trip.

        Test scenario:
            ``gdf.crs.to_epsg()`` must still report 4326 on the
            materialized GDF.
        """
        with as_datasource(point_gdf) as ds:
            result = datasource_to_gdf(ds)
        assert result.crs.to_epsg() == 4326, f"CRS changed: {result.crs.to_epsg()}"

    def test_polygonize_style_source(self, polygonize_datasource):
        """Materialize a Polygonize-style OGR DataSource.

        Test scenario:
            ``gdal.Polygonize`` on a 2-column band with two distinct
            values produces two polygons; the bridge must materialize
            a 2-row GeoDataFrame with a ``value`` integer column.
        """
        result = datasource_to_gdf(polygonize_datasource)
        assert isinstance(
            result, GeoDataFrame
        ), f"Expected GeoDataFrame, got {type(result).__name__}"
        assert len(result) == 2, f"Expected 2 polygons, got {len(result)}"
        assert (
            "value" in result.columns
        ), f"value column missing from materialized GDF: {result.columns.tolist()}"
        assert sorted(result["value"].tolist()) == [
            1,
            2,
        ], f"Polygon values changed: {result['value'].tolist()}"

    def test_not_a_featurecollection(self, point_gdf):
        """The helper must NOT return a ``FeatureCollection``.

        Test scenario:
            Users of the private bridge are internal code. Returning a
            ``FeatureCollection`` would force them to unwrap it before
            handing the result up to public APIs.
        """
        with as_datasource(point_gdf) as ds:
            result = datasource_to_gdf(ds)
        assert not isinstance(result, FeatureCollection), (
            f"Result must be plain GeoDataFrame, not a FeatureCollection: "
            f"got {type(result).__name__}"
        )

    def test_vector_translate_failure_raises(self, point_gdf, monkeypatch):
        """A ``None`` return from ``gdal.VectorTranslate`` raises RuntimeError.

        Test scenario:
            The real ``VectorTranslate`` returns ``None`` on failure.
            The helper must convert that sentinel into an explicit
            ``RuntimeError`` so callers cannot ignore it.
        """
        monkeypatch.setattr(
            "pyramids.feature._ogr.gdal.VectorTranslate",
            lambda *a, **kw: None,
        )
        with as_datasource(point_gdf) as ds:
            with pytest.raises(RuntimeError, match="VectorTranslate failed"):
                datasource_to_gdf(ds)

    def test_no_filesystem_tempfile_on_success(self, point_gdf, monkeypatch):
        """D-M4: the /vsimem/ path replaces the old filesystem temp file.

        Test scenario:
            The previous implementation created
            ``<tempdir>/pyramids_ogr_<uuid>.geojson`` on disk for every
            call. The new implementation stays in osgeo.gdal's
            ``/vsimem/`` VFS and reads the bytes out via
            ``gdal.VSIFOpenL``. Patch ``tempfile.gettempdir`` to raise
            if it is called during the round-trip — if it's called,
            the old disk path has crept back in.
        """
        import tempfile

        def _fail_gettempdir():
            raise AssertionError(
                "tempfile.gettempdir was called — the /vsimem/ path "
                "should handle the round-trip in memory."
            )

        monkeypatch.setattr(tempfile, "gettempdir", _fail_gettempdir)

        with as_datasource(point_gdf) as ds:
            gdf_back = datasource_to_gdf(ds)
        assert len(gdf_back) == len(point_gdf)

    def test_empty_datasource_round_trip(self, monkeypatch):
        """D-M4: zero-feature DataSource round-trips without error."""
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with as_datasource(empty) as ds:
            back = datasource_to_gdf(ds)
        assert len(back) == 0
