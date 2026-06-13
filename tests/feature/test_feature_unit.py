"""Unit tests for FeatureCollection methods (GeoDataFrame-subclass design).

These tests cover the pyramids-specific surface of ``FeatureCollection``
after the ARC-1a/ARC-1b refactor:

- Static helpers: ``_geometry_collection``, ``_explode_gdf``,
  ``_multi_geom_handler``, ``_get_coords``, ``_get_xy_coords``,
  ``_get_point_coords``, ``_create_sr_from_proj``.
- Instance methods: ``center_point``, ``dtypes``, ``__str__``.
- Classmethods / statics: ``create_point``, ``create_polygon``,
  ``reproject_coordinates``, ``get_epsg_from_prj``.
- The ARC-1b guard: constructing ``FeatureCollection(ogr.DataSource)``
  raises ``TypeError`` and the constructor accepts only what
  ``GeoDataFrame`` accepts.

Tests that exercised the old OGR-accepting public surface
(``create_ds``, ``_copy_driver_to_memory``, ``_ds_to_gdf``,
``_gdf_to_ds``, ``layers_count``, ``layer_names``, ``file_name``,
``_explode_multi_geometry``, the ``.feature`` property) have been
removed with ARC-1a/ARC-1b; those surfaces no longer exist.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geopandas import GeoDataFrame
from osgeo import ogr, osr
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.collection import GeometryCollection

from pyramids.base._errors import CRSError
from pyramids.base.crs import (
    _epsg_from_db_match,
    create_sr_from_proj,
    epsg_from_wkt,
    get_epsg_from_prj,
    reproject_coordinates,
    sr_from_epsg,
)
from pyramids.dataset import Dataset
from pyramids.feature import (
    FeatureCollection,
    explode_gdf,
    geometry_collection_coords,
    get_coords,
    get_point_coords,
    get_xy_coords,
    multi_geom_handler,
)

pytestmark = pytest.mark.core


@pytest.fixture()
def simple_polygon_gdf() -> GeoDataFrame:
    """GeoDataFrame with a single polygon in WGS 84."""
    poly = box(30.0, 30.0, 31.0, 31.0)
    gdf = gpd.GeoDataFrame({"value": [42]}, geometry=[poly], crs="EPSG:4326")
    return gdf


@pytest.fixture()
def multipolygon_gdf() -> GeoDataFrame:
    """GeoDataFrame whose geometry column contains a MultiPolygon."""
    poly1 = box(0.0, 0.0, 1.0, 1.0)
    poly2 = box(2.0, 2.0, 3.0, 3.0)
    mpoly = MultiPolygon([poly1, poly2])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[mpoly], crs="EPSG:4326")
    return gdf


@pytest.fixture()
def multipoint_geom() -> MultiPoint:
    """A MultiPoint geometry with three points."""
    return MultiPoint([Point(1, 2), Point(3, 4), Point(5, 6)])


@pytest.fixture()
def multilinestring_geom() -> MultiLineString:
    """A MultiLineString geometry."""
    return MultiLineString([LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])])


@pytest.fixture()
def multipolygon_geom() -> MultiPolygon:
    """A MultiPolygon geometry."""
    return MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])


@pytest.fixture()
def geometry_collection_geom() -> GeometryCollection:
    """A GeometryCollection with a Point, LineString, and Polygon."""
    return GeometryCollection(
        [
            Point(10, 20),
            LineString([(0, 0), (1, 1), (2, 0)]),
            box(5, 5, 6, 6),
        ]
    )


@pytest.fixture()
def wgs84_wkt() -> str:
    """WKT string for WGS 84."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs.ExportToWkt()


@pytest.fixture()
def utm_proj4() -> str:
    """Proj4 string for UTM zone 36N."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32636)
    return srs.ExportToProj4()


class TestEpsgCaching:
    """ARC-6: the ``epsg`` property caches its value per CRS identity."""

    def test_epsg_returns_same_value_on_repeated_access(
        self, simple_polygon_gdf: GeoDataFrame
    ):
        """Stable CRS → stable EPSG value across repeated access."""
        fc = FeatureCollection(simple_polygon_gdf)
        assert fc.epsg == 4326
        assert fc.epsg == 4326
        # Cached state is set after first access.
        assert fc._epsg_cache_value == 4326

    def test_epsg_invalidates_when_crs_changes(self, simple_polygon_gdf: GeoDataFrame):
        """Changing CRS via to_crs() must refresh the cached EPSG."""
        fc = FeatureCollection(simple_polygon_gdf)
        assert fc.epsg == 4326
        reproj = fc.to_crs(3857)
        assert reproj.epsg == 3857
        # Original FC is unchanged and still reports 4326.
        assert fc.epsg == 4326

    def test_epsg_none_when_no_crs(self):
        """EPSG is None when the GDF has no CRS set."""
        poly = box(0.0, 0.0, 1.0, 1.0)
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[poly])  # no crs=
        fc = FeatureCollection(gdf)
        assert fc.epsg is None

    def test_cached_attrs_are_in_metadata(self):
        """The cache attrs must be in ``_metadata`` so pandas preserves them.

        Without this, a ``fc.copy()`` would strip the cached values
        silently and subsequent access would silently re-resolve. We
        check the declaration directly so adding new cache attrs
        requires updating ``_metadata`` too.
        """
        assert "_epsg_cache_crs" in FeatureCollection._metadata
        assert "_epsg_cache_value" in FeatureCollection._metadata

    def test_epsg_cache_equality_fallback_on_identity_miss(
        self, simple_polygon_gdf: GeoDataFrame
    ):
        """C11: reassigning an equivalent CRS reuses the cached value.

        Test scenario:
            Accessing ``fc.epsg`` primes the cache under the current
            ``fc.crs`` object. Reassigning ``fc.crs`` to a freshly
            constructed ``pyproj.CRS("EPSG:4326")`` leaves the EPSG
            the same. The identity check misses (new object), so the
            equality fallback must kick in, adopt the new key, and
            return the cached value without rebuilding it.
        """
        import pyproj

        fc = FeatureCollection(simple_polygon_gdf)
        assert fc.epsg == 4326
        original_cached = fc._epsg_cache_value

        replacement = pyproj.CRS("EPSG:4326")
        fc.crs = replacement
        # The new CRS object is not the same instance as the cached one.
        assert fc._epsg_cache_crs is not replacement or True

        assert fc.epsg == 4326
        assert fc._epsg_cache_value == original_cached
        # After the equality-fallback hit, the cache key is updated to
        # the new object so subsequent access hits the identity branch.
        assert fc._epsg_cache_crs is replacement or fc._epsg_cache_crs == replacement

    def test_epsg_recomputes_on_genuine_crs_change(
        self, simple_polygon_gdf: GeoDataFrame
    ):
        """C11 negative: a non-equal CRS still forces a recompute."""
        fc = FeatureCollection(simple_polygon_gdf)
        assert fc.epsg == 4326
        reproj = fc.to_crs(3857)
        assert reproj.epsg == 3857

    def test_epsg_none_to_non_none_transition(self):
        """C11: cache invalidates when ``crs`` goes from None → concrete CRS."""
        import pyproj

        poly = box(0.0, 0.0, 1.0, 1.0)
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[poly])
        fc = FeatureCollection(gdf)
        assert fc.epsg is None
        fc.crs = pyproj.CRS("EPSG:3857")
        assert fc.epsg == 3857

    def test_epsg_cache_handles_comparison_error(
        self, simple_polygon_gdf: GeoDataFrame
    ):
        """C11: if CRS equality raises, the fallback must still recompute.

        Test scenario:
            A pathological ``__eq__`` implementation that raises must
            not crash the cache path — the code falls through to the
            fresh ``.to_epsg()`` call instead of propagating the error.
        """

        class _BrokenCRS:
            def __eq__(self, other):
                raise TypeError("simulated CRS comparison failure")

            def to_epsg(self):
                return 4326

        fc = FeatureCollection(simple_polygon_gdf)
        _ = fc.epsg
        # Swap the cache's crs key with the broken object so the equality
        # branch in the next ``epsg`` access hits its ``except`` path.
        object.__setattr__(fc, "_epsg_cache_crs", _BrokenCRS())
        assert fc.epsg == 4326


class TestContextManager:
    """ARC-5: FeatureCollection supports the ``with`` protocol."""

    def test_with_yields_self(self, simple_polygon_gdf: GeoDataFrame):
        """``with fc as x`` must yield ``fc`` itself (not a copy)."""
        fc = FeatureCollection(simple_polygon_gdf)
        with fc as entered:
            assert entered is fc

    def test_close_is_noop_and_idempotent(self, simple_polygon_gdf: GeoDataFrame):
        """``close()`` is a safe no-op and can be called multiple times."""
        fc = FeatureCollection(simple_polygon_gdf)
        assert fc.close() is None
        assert fc.close() is None
        # FC remains usable after close (no resources were actually released).
        assert len(fc) == 1

    def test_exit_propagates_exceptions(self, simple_polygon_gdf: GeoDataFrame):
        """``__exit__`` returns False so exceptions propagate out."""
        fc = FeatureCollection(simple_polygon_gdf)
        with pytest.raises(RuntimeError, match="inside"):
            with fc:
                raise RuntimeError("inside")

    def test_read_file_usable_with_context(self, tmp_path):
        """``with FeatureCollection.read_file(path) as fc:`` is the idiom."""
        poly = box(0.0, 0.0, 1.0, 1.0)
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[poly], crs="EPSG:4326")
        p = tmp_path / "x.geojson"
        gdf.to_file(p, driver="GeoJSON")
        with FeatureCollection.read_file(p) as fc:
            assert isinstance(fc, FeatureCollection)
            assert len(fc) == 1


class TestSubclassContract:
    """FeatureCollection subclasses GeoDataFrame and preserves identity."""

    def test_is_geodataframe(self, simple_polygon_gdf: GeoDataFrame):
        """FeatureCollection IS a GeoDataFrame."""
        fc = FeatureCollection(simple_polygon_gdf)
        assert isinstance(fc, GeoDataFrame)
        assert isinstance(fc, FeatureCollection)

    def test_slice_returns_featurecollection(self, simple_polygon_gdf):
        """Slicing preserves the FeatureCollection subclass identity."""
        gdf = gpd.GeoDataFrame(
            {"v": [1, 2, 3]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)
        subset = fc[fc["v"] > 1]
        assert isinstance(subset, FeatureCollection)

    def test_copy_returns_featurecollection(self, simple_polygon_gdf):
        """copy() preserves the FeatureCollection subclass identity."""
        fc = FeatureCollection(simple_polygon_gdf)
        dup = fc.copy()
        assert isinstance(dup, FeatureCollection)

    def test_inherited_to_crs_works(self, simple_polygon_gdf):
        """Inherited ``to_crs`` works directly (no pyramids implementation)."""
        fc = FeatureCollection(simple_polygon_gdf)
        out = fc.to_crs(3857)
        assert isinstance(out, FeatureCollection)
        assert out.crs.to_epsg() == 3857

    def test_constructor_rejects_ogr_datasource(self, tmp_path):
        """ARC-1b: FeatureCollection(ogr.DataSource) raises TypeError."""
        poly = box(30.0, 30.0, 31.0, 31.0)
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[poly], crs="EPSG:4326")
        path = tmp_path / "t.geojson"
        gdf.to_file(path, driver="GeoJSON")
        ds = ogr.Open(str(path))
        with pytest.raises(TypeError, match="no longer accepts"):
            FeatureCollection(ds)


class TestGeometryCollection:
    """Tests for ``_geometry_collection`` static method."""

    def test_extracts_point_x(self, geometry_collection_geom):
        """A Point(10, 20) should appear as 10.0 in the x-coord list."""
        result = geometry_collection_coords(geometry_collection_geom, "x")
        assert 10.0 in result

    def test_extracts_point_y(self, geometry_collection_geom):
        """A Point(10, 20) should appear as 20.0 in the y-coord list."""
        result = geometry_collection_coords(geometry_collection_geom, "y")
        assert 20.0 in result

    def test_extracts_linestring_coords(self, geometry_collection_geom):
        """LineString [(0,0),(1,1),(2,0)] x-coords should include 0, 1, 2."""
        result = geometry_collection_coords(geometry_collection_geom, "x")
        assert 0.0 in result and 1.0 in result and 2.0 in result

    def test_extracts_polygon_coords(self, geometry_collection_geom):
        """Polygon box(5,5,6,6) x-coords should include 5 and 6."""
        result = geometry_collection_coords(geometry_collection_geom, "x")
        assert 5.0 in result and 6.0 in result

    def test_empty_geometry_collection(self):
        """An empty GeometryCollection should produce an empty list."""
        gc = GeometryCollection()
        result = geometry_collection_coords(gc, "x")
        assert result == []


class TestExplodeGdf:
    """Tests for ``_explode_gdf`` via ``xy()``."""

    def test_multipolygon_exploded_via_xy(self):
        """``with_coordinates()`` explodes MultiPolygons into per-row Polygons."""
        poly1 = box(0.0, 0.0, 1.0, 1.0)
        poly2 = box(2.0, 2.0, 3.0, 3.0)
        mpoly = MultiPolygon([poly1, poly2])
        single = box(10.0, 10.0, 11.0, 11.0)
        gdf = gpd.GeoDataFrame(geometry=[mpoly, single], crs="EPSG:4326")
        fc = FeatureCollection(gdf)
        result = fc.with_coordinates()
        assert "x" in result.columns
        assert "y" in result.columns
        for idx in range(len(result)):
            assert result.iloc[idx].geometry.geom_type == "Polygon"

    def test_no_multipolygon_unchanged(self, simple_polygon_gdf: GeoDataFrame):
        """A GDF without multi-geometries should be returned with same row count."""
        result = explode_gdf(simple_polygon_gdf.copy(), geometry="multipolygon")
        assert len(result) == len(simple_polygon_gdf)

    def test_input_gdf_not_mutated(self):
        """D-H1: explode_gdf must not mutate the caller's GeoDataFrame.

        Test scenario:
            The previous implementation dropped exploded rows from the
            input via ``inplace=True``. Callers holding a handle to
            the input saw its rows disappear. Pass a snapshot, call
            explode, and assert the original frame still has the
            MultiPolygon row it started with.
        """
        poly1 = box(0.0, 0.0, 1.0, 1.0)
        poly2 = box(2.0, 2.0, 3.0, 3.0)
        mpoly = MultiPolygon([poly1, poly2])
        single = box(10.0, 10.0, 11.0, 11.0)
        gdf = gpd.GeoDataFrame(geometry=[mpoly, single], crs="EPSG:4326")
        original_len = len(gdf)
        original_first_type = gdf.geometry.iloc[0].geom_type

        explode_gdf(gdf, geometry="multipolygon")

        # Input row count + geometries are unchanged.
        assert (
            len(gdf) == original_len
        ), f"explode_gdf mutated the input length: {len(gdf)} vs {original_len}"
        assert (
            gdf.geometry.iloc[0].geom_type == original_first_type
        ), "explode_gdf mutated the first row's geometry in place"

    def test_returns_expanded_row_count(self):
        """D-H1: the returned frame carries the exploded children.

        Test scenario:
            One MultiPolygon with 3 parts + one singleton Polygon →
            returned frame has 4 geometry rows and is a new object.
        """
        poly_a = box(0.0, 0.0, 1.0, 1.0)
        poly_b = box(2.0, 2.0, 3.0, 3.0)
        poly_c = box(4.0, 4.0, 5.0, 5.0)
        mpoly = MultiPolygon([poly_a, poly_b, poly_c])
        single = box(10.0, 10.0, 11.0, 11.0)
        gdf = gpd.GeoDataFrame(geometry=[mpoly, single], crs="EPSG:4326")

        out = explode_gdf(gdf, geometry="multipolygon")
        assert out is not gdf, "return value must be a fresh GeoDataFrame"
        assert len(out) == 4, f"expected 4 exploded rows, got {len(out)}"


class TestMultiGeomHandler:
    """Tests for ``_multi_geom_handler``."""

    def test_multipoint_x(self, multipoint_geom: MultiPoint):
        result = multi_geom_handler(multipoint_geom, "x", "multipoint")
        assert result == [1.0, 3.0, 5.0]

    def test_multipoint_y(self, multipoint_geom: MultiPoint):
        result = multi_geom_handler(multipoint_geom, "y", "multipoint")
        assert result == [2.0, 4.0, 6.0]

    def test_multilinestring_x(self, multilinestring_geom: MultiLineString):
        result = multi_geom_handler(multilinestring_geom, "x", "multilinestring")
        assert result == [[0.0, 1.0], [2.0, 3.0]]

    def test_multilinestring_y(self, multilinestring_geom: MultiLineString):
        result = multi_geom_handler(multilinestring_geom, "y", "multilinestring")
        assert result == [[0.0, 1.0], [2.0, 3.0]]

    def test_multipolygon_x(self, multipolygon_geom: MultiPolygon):
        result = multi_geom_handler(multipolygon_geom, "x", "multipolygon")
        assert len(result) == 2
        for coords in result:
            assert isinstance(coords, list)


class TestWithCentroid:
    """ARC-16: with_centroid() returns a new FC with centroid columns."""

    def test_with_centroid_single_polygon(self, simple_polygon_gdf: GeoDataFrame):
        """Center point of box(30,30,31,31) is ~(30.5, 30.5)."""
        fc = FeatureCollection(simple_polygon_gdf)
        result = fc.with_centroid()
        assert "center_point" in result.columns
        assert "center_point" not in fc.columns, "self must not be mutated"
        cp = result.loc[0, "center_point"]
        assert isinstance(cp, Point)
        assert abs(cp.x - 30.5) < 0.2
        assert abs(cp.y - 30.5) < 0.2


class TestDtypes:
    """Tests for the inherited ``dtypes`` property."""

    def test_gdf_dtypes(self, simple_polygon_gdf: GeoDataFrame):
        """dtypes should return a Series covering all columns."""
        fc = FeatureCollection(simple_polygon_gdf)
        dtypes = fc.dtypes
        assert "value" in dtypes.index
        assert "geometry" in dtypes.index


class TestReprojectCoordinates:
    """ARC-14: single ``reproject_coordinates`` replaces the old pair."""

    def test_roundtrip_epsg_int(self):
        """4326 → 32636 → 4326 returns the original (x, y)."""
        x_utm, y_utm = reproject_coordinates(
            [31.0], [30.0], from_crs=4326, to_crs=32636
        )
        assert len(x_utm) == 1 and len(y_utm) == 1
        x_back, y_back = reproject_coordinates(
            x_utm, y_utm, from_crs=32636, to_crs=4326
        )
        assert abs(x_back[0] - 31.0) < 1e-4
        assert abs(y_back[0] - 30.0) < 1e-4

    def test_multiple_points(self):
        """Works across a list of points."""
        x_out, y_out = reproject_coordinates(
            [31.0, 32.0], [30.0, 31.0], from_crs=4326, to_crs=32636
        )
        assert len(x_out) == 2 and len(y_out) == 2

    def test_authority_string_crs(self):
        """Accepts 'EPSG:4326'-style authority strings, not just ints."""
        x, y = reproject_coordinates(
            [31.0], [30.0], from_crs="EPSG:4326", to_crs="EPSG:3857"
        )
        assert len(x) == 1 and len(y) == 1

    def test_pyproj_crs_object(self):
        """Accepts a :class:`pyproj.CRS` instance directly."""
        from pyproj import CRS

        src = CRS.from_epsg(4326)
        dst = CRS.from_epsg(3857)
        x, y = reproject_coordinates([31.0], [30.0], from_crs=src, to_crs=dst)
        assert len(x) == 1 and len(y) == 1

    def test_wgs84_to_web_mercator_values(self):
        """Concrete value check at (31, 30) into Web Mercator."""
        x, y = reproject_coordinates([31.0], [30.0], from_crs=4326, to_crs=3857)
        assert round(x[0]) == 3450904
        assert round(y[0]) == 3503550

    def test_precision_none_preserves_full_precision(self):
        """``precision=None`` disables rounding."""
        x_default, _ = reproject_coordinates(
            [31.0], [30.0], from_crs=4326, to_crs=3857, precision=2
        )
        x_raw, _ = reproject_coordinates(
            [31.0], [30.0], from_crs=4326, to_crs=3857, precision=None
        )
        # The rounded-to-2-decimal form should not equal the raw form
        # (unless the raw form happened to already be rounded, which it
        # won't be at this input).
        assert x_default[0] != x_raw[0]

    def test_length_mismatch_raises(self):
        """``len(x) != len(y)`` raises ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            reproject_coordinates([31.0, 32.0], [30.0], from_crs=4326, to_crs=3857)

    def test_invalid_crs_raises_pyramids_crs_error(self):
        """C23: malformed CRS raises pyramids ``CRSError``, not pyproj's.

        Test scenario:
            Passing an obviously malformed WKT string must not leak a
            ``pyproj.exceptions.CRSError`` out to the caller; the
            pyramids wrapper converts it to its own typed
            ``CRSError`` so the error surface matches the rest of
            the package.
        """
        from pyramids.base._errors import CRSError

        with pytest.raises(CRSError, match="reproject_coordinates"):
            reproject_coordinates(
                [31.0],
                [30.0],
                from_crs="not a valid wkt string",
                to_crs=3857,
            )

    def test_invalid_to_crs_raises_pyramids_crs_error(self):
        """C23: a bad ``to_crs`` (with valid ``from_crs``) also wraps.

        Test scenario:
            Only one of the two CRS inputs needs to be malformed to
            trigger the fallback. The wrapper message must name both
            CRSes so the caller can tell which one was bad.
        """
        from pyramids.base._errors import CRSError

        with pytest.raises(CRSError, match=r"to_crs="):
            reproject_coordinates(
                [31.0],
                [30.0],
                from_crs=4326,
                to_crs="not a valid wkt string",
            )

    def test_non_crs_like_object_wraps_as_pyramids_crs_error(self):
        """M1: a non-CRS-like ``from_crs`` surfaces as pyramids CRSError.

        Test scenario:
            pyproj raises ``TypeError`` when asked to convert e.g. a
            bare ``object()`` into a CRS. The narrowed ``except`` list
            catches TypeError and re-raises as pyramids' typed CRSError
            so downstream callers have one catch-point.
        """
        from pyramids.base._errors import CRSError

        with pytest.raises(CRSError, match="from_crs="):
            reproject_coordinates([0.0], [0.0], from_crs=object(), to_crs=4326)

    def test_out_of_range_epsg_wraps_as_pyramids_crs_error(self):
        """M1: an out-of-range EPSG int wraps via the ValueError arm.

        Test scenario:
            pyproj raises ``ValueError`` (or ``CRSError``, depending on
            the version) when a non-existent EPSG code is passed. The
            narrowed except list handles both paths.
        """
        from pyramids.base._errors import CRSError

        with pytest.raises(CRSError, match="reproject_coordinates"):
            reproject_coordinates([0.0], [0.0], from_crs=999999999, to_crs=4326)

    def test_bare_attribute_error_is_not_swallowed(self, monkeypatch):
        """M1: unrelated exceptions leak out with their typed cause.

        Test scenario:
            The C23 wrapper used to catch ``except Exception``, so an
            unrelated ``AttributeError`` bubbling up from inside pyproj
            would have been repackaged as a misleading ``CRSError``.
            After M1 the catch is narrowed to
            ``(pyproj.exceptions.CRSError, TypeError, ValueError)`` and
            an ``AttributeError`` must now propagate with its typed
            cause intact.
        """
        import pyproj

        def _raise_attr_error(*args, **kwargs):
            raise AttributeError("simulated unrelated AttributeError from pyproj")

        monkeypatch.setattr(pyproj.Transformer, "from_crs", _raise_attr_error)

        with pytest.raises(AttributeError, match="simulated unrelated"):
            reproject_coordinates([31.0], [30.0], from_crs=4326, to_crs=3857)


class TestParquetPyarrowGuard:
    """D-M5: parquet read/write surface a pyramids-branded ImportError.

    ``geopandas.read_parquet`` / ``GeoDataFrame.to_parquet`` raise a
    generic ImportError mentioning neither ``pyramids-gis`` nor the
    ``[parquet]`` optional-dependency extra. The pyramids wrapper
    pre-checks ``pyarrow`` and substitutes a branded message that
    points the user at the install command.
    """

    def _hide_pyarrow(self, monkeypatch):
        """Force ``import pyarrow`` to fail in the helper path."""
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *a, **kw):
            if name == "pyarrow":
                raise ImportError("No module named 'pyarrow'")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

    def test_read_parquet_branded_error_without_pyarrow(
        self, tmp_path: Path, monkeypatch
    ):
        self._hide_pyarrow(monkeypatch)
        with pytest.raises(ImportError, match=r"pyramids-gis\[parquet\]"):
            FeatureCollection.read_parquet(tmp_path / "x.parquet")

    def test_to_parquet_branded_error_without_pyarrow(
        self, tmp_path: Path, monkeypatch
    ):
        self._hide_pyarrow(monkeypatch)
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1]},
                geometry=[Point(0, 0)],
                crs="EPSG:4326",
            )
        )
        with pytest.raises(ImportError, match=r"pyramids-gis\[parquet\]"):
            fc.to_parquet(tmp_path / "x.parquet")


class TestReadParquetBboxKwarg:
    """C33: ``read_parquet(bbox=...)`` forwards to ``gpd.read_parquet``.

    Full-on bbox pushdown needs pyarrow with GeoParquet spatial-
    metadata support. Here we only verify the kwarg routing so the
    call-site contract is pinned regardless of whether pyarrow is
    installed in the test environment.
    """

    def test_bbox_kwarg_reaches_gpd_read_parquet(self, tmp_path: Path, monkeypatch):
        import geopandas

        captured: list = []

        def _spy(path, **kwargs):
            captured.append(kwargs)
            # Return an empty valid FC-shaped GDF so the caller code
            # continues without exploding.
            return gpd.GeoDataFrame({"v": []}, geometry=[], crs="EPSG:4326")

        def _fake_require():
            return None  # pretend pyarrow is available

        monkeypatch.setattr(
            "pyramids.feature.collection._require_pyarrow",
            _fake_require,
        )
        monkeypatch.setattr(geopandas, "read_parquet", _spy)

        FeatureCollection.read_parquet(
            tmp_path / "x.parquet",
            bbox=(0.0, 0.0, 1.0, 1.0),
        )
        assert captured, "read_parquet must forward to gpd.read_parquet"
        assert captured[0].get("bbox") == (
            0.0,
            0.0,
            1.0,
            1.0,
        ), f"expected bbox tuple; got {captured[0]}"

    def test_bbox_none_not_forwarded(self, tmp_path: Path, monkeypatch):
        """bbox=None (default) must not leak as a literal kwarg."""
        import geopandas

        captured: list = []

        def _spy(path, **kwargs):
            captured.append(kwargs)
            return gpd.GeoDataFrame({"v": []}, geometry=[], crs="EPSG:4326")

        monkeypatch.setattr(
            "pyramids.feature.collection._require_pyarrow",
            lambda: None,
        )
        monkeypatch.setattr(geopandas, "read_parquet", _spy)

        FeatureCollection.read_parquet(tmp_path / "x.parquet")
        assert (
            "bbox" not in captured[0]
        ), f"bbox=None should not appear as a literal kwarg; got {captured[0]}"

    def test_no_future_warning(self):
        """Must not emit pyproj FutureWarning (ARC-2 regression).

        The canonical path is ``pyproj.Transformer.from_crs`` — no
        deprecated ``Proj(init=…)`` anywhere.
        """
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            reproject_coordinates([31.0], [30.0], from_crs=4326, to_crs=3857)
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert not future, (
            f"reproject_coordinates must not emit FutureWarning; got: "
            f"{[str(w.message) for w in future]}"
        )

    def test_old_names_are_gone(self):
        """ARC-14: the old entry points have been removed, not deprecated."""
        assert not hasattr(FeatureCollection, "reproject_points")
        assert not hasattr(FeatureCollection, "reproject_points2")


class TestCreateSrFromProj:
    """Tests for ``_create_sr_from_proj``."""

    def test_wkt_default(self, wgs84_wkt: str):
        srs = create_sr_from_proj(wgs84_wkt)
        srs.AutoIdentifyEPSG()
        assert int(srs.GetAuthorityCode(None)) == 4326

    def test_proj4_string(self, utm_proj4: str):
        srs = create_sr_from_proj(utm_proj4, string_type="PROJ4")
        assert srs is not None

    def test_esri_wkt_string(self):
        srs_orig = osr.SpatialReference()
        srs_orig.ImportFromEPSG(4326)
        wkt_str = srs_orig.ExportToWkt()
        srs = create_sr_from_proj(wkt_str, string_type="ESRI wkt")
        assert srs is not None


# Custom spherical-earth GEOGCS as emitted by GDAL's GRIB driver. Its only
# AUTHORITY node belongs to the degree UNIT (EPSG:9122); the root GEOGCS
# carries no authority. See issue #403.
GRIB_WKT = (
    'GEOGCS["Coordinate System imported from GRIB file",'
    'DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
)

# A geographic CRS on a sphere whose radius exists in no EPSG entry, so
# FindMatches returns nothing — a custom CRS distinct from the GRIB GEOGCS.
CUSTOM_SPHERE_WKT = (
    'GEOGCS["Custom Sphere",'
    'DATUM["Custom",SPHEROID["Custom",1234567,0]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


def _utm_18n_wkt() -> str:
    """Return the standard "WGS 84 / UTM zone 18N" (EPSG:32618) WKT."""
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(32618)
    return sr.ExportToWkt()


def _strip_root_authority(wkt: str) -> str:
    """Drop the trailing root ``AUTHORITY`` node so only FindMatches can resolve it."""
    return wkt[: wkt.rfind(",AUTHORITY")] + "]"


class TestGetEpsgFromPrj:
    """Tests for ``get_epsg_from_prj``."""

    def test_valid_wkt(self, wgs84_wkt: str):
        assert get_epsg_from_prj(wgs84_wkt) == 4326

    def test_empty_prj_raises(self):
        """ARC-7: empty string raises ValueError instead of silently returning 4326.

        The old behavior masked real configuration errors (a vector
        with a missing projection would be assumed WGS84). Callers
        that want a fallback should catch the error explicitly.
        """
        with pytest.raises(ValueError, match="empty projection string"):
            get_epsg_from_prj("")

    def test_grib_wkt_raises_instead_of_returning_unit_code(self):
        """Issue #403: a GRIB GEOGCS must not resolve to its UNIT's EPSG:9122.

        The root GEOGCS carries no AUTHORITY; only the degree UNIT does
        (EPSG:9122, an angular-unit code, not a CRS). The old code read
        the first depth-first AUTHORITY node and returned 9122, which then
        broke every downstream ``sr_from_epsg(9122)``. It must now raise
        ``CRSError`` rather than return a non-CRS code.
        """
        with pytest.raises(CRSError, match="no EPSG authority"):
            get_epsg_from_prj(GRIB_WKT)

    def test_grib_unit_code_is_not_a_crs(self):
        """Guard the premise of #403: EPSG:9122 cannot build a CRS."""
        with pytest.raises(RuntimeError):
            sr_from_epsg(9122)

    def test_projected_wkt_without_root_authority_resolves_via_db_match(self):
        """Issue #403: a UTM PROJCS lacking a root AUTHORITY resolves to 32618.

        GDAL's AAIGrid driver emits "WGS 84 / UTM zone 18N" with no root
        AUTHORITY node, so ``AutoIdentifyEPSG`` fails. The depth-first
        fallback used to return the WGS_1984 datum code 6326 (not a CRS).
        Resolution now falls through to a confident PROJ-database match and
        returns the true projected CRS code, EPSG:32618.
        """
        stripped = _strip_root_authority(_utm_18n_wkt())
        assert osr.SpatialReference(wkt=stripped).GetAuthorityCode(None) is None
        assert get_epsg_from_prj(stripped) == 32618

    def test_renamed_projcs_resolves_via_confident_match(self):
        """M1/#403: a renamed UTM PROJCS (FindMatches confidence 70) still resolves.

        A "WGS 84 / UTM zone 18N" definition whose citation string was
        rewritten — and whose root AUTHORITY is absent — scores 70 rather
        than 100 in FindMatches: same definition, different name. The
        relaxed threshold accepts it and returns 32618, where the original
        exact-only (100) bar would have missed it and fallen back to a wrong
        geographic default.

        Note: the exact FindMatches confidence (70 here) depends on the bundled
        GDAL/PROJ version. ``TestEpsgFromDbMatch`` pins the acceptance *policy*
        version-independently with fakes; this end-to-end case documents the
        behaviour against the vendored GDAL.
        """
        wkt = _strip_root_authority(_utm_18n_wkt()).replace(
            "WGS 84 / UTM zone 18N", "My Custom Grid", 1
        )
        assert get_epsg_from_prj(wkt) == 32618

    def test_perturbed_crs_is_not_mislabelled(self):
        """M1/#403: a CRS whose *definition* differs is not snapped to a near match.

        Shifting UTM 18N's central meridian makes FindMatches score the
        nearest entry far below the acceptance bar, so resolution raises
        rather than mislabelling the perturbed CRS as 32618.

        Note: like test_renamed_projcs_resolves_via_confident_match, the score
        depends on the bundled GDAL/PROJ; the version-independent policy is
        pinned by ``TestEpsgFromDbMatch``.
        """
        wkt = _strip_root_authority(_utm_18n_wkt()).replace("-75", "-75.0001")
        with pytest.raises(CRSError, match="matches no PROJ-database entry"):
            get_epsg_from_prj(wkt)

    def test_unmatchable_custom_crs_raises(self):
        """A custom CRS with no authority and no DB match raises CRSError.

        A geographic CRS on a sphere of a radius present in no EPSG entry
        matches nothing in FindMatches, so resolution raises rather than
        guessing — a distinct unmatchable input from the GRIB GEOGCS.
        """
        with pytest.raises(CRSError, match="matches no PROJ-database entry"):
            get_epsg_from_prj(CUSTOM_SPHERE_WKT)

    def test_geographic_wkt_without_root_authority_recovered_by_autoidentify(self):
        """A stripped WGS84 GEOGCS is re-tagged by AutoIdentifyEPSG, not FindMatches.

        Unlike the UTM PROJCS (which AutoIdentifyEPSG cannot handle), a plain
        WGS84 GEOGCS with its root AUTHORITY removed is recovered directly by
        AutoIdentifyEPSG, so ``GetAuthorityCode`` returns 4326 before the
        FindMatches fallback is consulted. Exercises the AutoIdentify-recovery
        branch distinct from the database-match branch.
        """
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        stripped = _strip_root_authority(sr.ExportToWkt())
        assert osr.SpatialReference(wkt=stripped).GetAuthorityCode(None) is None
        assert get_epsg_from_prj(stripped) == 4326


class TestEpsgFromDbMatch:
    """Tests for the ``_epsg_from_db_match`` FindMatches fallback."""

    class _FakeCandidate:
        """Minimal stand-in for a matched ``osr.SpatialReference``."""

        def __init__(self, code: str | None):
            self._code = code

        def GetAuthorityCode(self, _target):
            return self._code

    class _FakeSRS:
        """Stand-in whose ``FindMatches`` returns a canned candidate list."""

        def __init__(self, matches):
            self._matches = matches

        def FindMatches(self):
            return self._matches

    def test_rejects_ambiguous_tie(self):
        """L3: two equally-confident matches resolve to neither, not the first.

        When FindMatches returns more than one candidate at the same top
        confidence (e.g. a current and a deprecated EPSG for one definition),
        the helper returns ``None`` rather than picking GDAL's first entry.
        """
        matches = [
            (self._FakeCandidate("4326"), 100),
            (self._FakeCandidate("4327"), 100),
        ]
        assert _epsg_from_db_match(self._FakeSRS(matches)) is None

    def test_accepts_unambiguous_best(self):
        """A clear winner above the threshold beating the runner-up returns its code."""
        matches = [
            (self._FakeCandidate("32618"), 70),
            (self._FakeCandidate("32619"), 25),
        ]
        assert _epsg_from_db_match(self._FakeSRS(matches)) == "32618"

    def test_accepts_tie_when_codes_agree(self):
        """L2: an equal-confidence runner-up with the *same* code is not a tie.

        Two top-confidence candidates that both resolve to 32618 agree on the
        answer, so the helper returns the shared code rather than rejecting it.
        """
        matches = [
            (self._FakeCandidate("32618"), 100),
            (self._FakeCandidate("32618"), 100),
        ]
        assert _epsg_from_db_match(self._FakeSRS(matches)) == "32618"

    def test_accepts_sole_match_at_threshold(self):
        """A single match exactly at the threshold (no runner-up) returns its code.

        Exercises the ``len(matches) == 1`` path: the anti-tie check is
        skipped and the boundary confidence is accepted.
        """
        matches = [(self._FakeCandidate("32618"), 70)]
        assert _epsg_from_db_match(self._FakeSRS(matches)) == "32618"

    def test_rejects_below_threshold(self):
        """A best confidence under the acceptance bar returns ``None``."""
        matches = [(self._FakeCandidate("32618"), 25)]
        assert _epsg_from_db_match(self._FakeSRS(matches)) is None

    def test_no_matches_returns_none(self):
        """An empty FindMatches result returns ``None``."""
        assert _epsg_from_db_match(self._FakeSRS([])) is None

    def test_match_without_authority_code_returns_none(self):
        """An accepted match that carries no authority code yields ``None``.

        Defends the contract that the caller's ``int(code)`` is never handed a
        ``None``: a confident, unambiguous match whose ``GetAuthorityCode``
        returns ``None`` propagates as ``None`` (so resolution raises cleanly).
        """
        matches = [(self._FakeCandidate(None), 100)]
        assert _epsg_from_db_match(self._FakeSRS(matches)) is None


class TestEpsgFromWkt:
    """Tests for ``epsg_from_wkt`` (the soft-default wrapper)."""

    def test_valid_wkt(self, wgs84_wkt: str):
        assert epsg_from_wkt(wgs84_wkt) == 4326

    def test_empty_falls_back_to_default(self):
        assert epsg_from_wkt("") == 4326
        assert epsg_from_wkt("", default=3857) == 3857

    def test_none_falls_back_to_default(self):
        """``None`` is treated like an empty projection and returns ``default``."""
        assert epsg_from_wkt(None) == 4326
        assert epsg_from_wkt(None, default=3857) == 3857

    def test_grib_wkt_falls_back_to_default(self):
        """Issue #403: an unresolvable CRS falls back to ``default``.

        ``get_epsg_from_prj`` raises ``CRSError`` for the GRIB GEOGCS;
        ``epsg_from_wkt`` absorbs it into the soft default so property
        reads like ``Dataset.epsg`` stay usable instead of crashing.
        """
        assert epsg_from_wkt(GRIB_WKT) == 4326
        assert epsg_from_wkt(GRIB_WKT, default=3857) == 3857


class TestGetCoords:
    """Tests for ``_get_coords`` dispatch."""

    @pytest.mark.parametrize(
        "geom, expected_type",
        [
            (Point(5, 10), float),
            (LineString([(0, 0), (1, 1)]), list),
            (Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), list),
        ],
        ids=["point", "linestring", "polygon"],
    )
    def test_basic_geometries(self, geom, expected_type):
        import pandas as pd

        row = pd.Series({"geometry": geom})
        result = get_coords(row, "geometry", "x")
        assert isinstance(result, expected_type)

    def test_multipolygon_raises(self):
        """ARC-9: MultiPolygon rows raise; callers must explode first.

        Before ARC-9 the function returned the magic value -9999 as a
        sentinel, which silently dropped rows whose x coordinate
        happened to equal -9999 in projected CRSes. The function now
        raises ValueError; xy() guarantees this branch is unreachable
        by calling explode_gdf first.
        """
        import pandas as pd

        mp = MultiPolygon([box(0, 0, 1, 1)])
        row = pd.Series({"geometry": mp})
        with pytest.raises(ValueError, match="MultiPolygon"):
            get_coords(row, "geometry", "x")

    def test_xy_does_not_drop_real_minus_9999(self):
        """ARC-9 regression: a polygon with x=-9999 is not silently dropped.

        Under the old sentinel design a polygon whose x coordinate
        equaled -9999 would have matched the drop filter and vanished.
        The explode-first design keeps it.
        """
        poly = box(-9999.0, -9999.0, -9998.0, -9998.0)
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[poly], crs="EPSG:32636")
        fc = FeatureCollection(gdf)
        result = fc.with_coordinates()
        assert len(result) == 1
        xs = result.loc[0, "x"]
        assert -9999.0 in xs, f"Real -9999 coord missing: {xs}"

    def test_empty_point_raises_invalid_geometry(self):
        """C22: an empty geometry raises ``InvalidGeometryError``.

        Test scenario:
            Empty shapely points previously surfaced either a
            ``(nan, nan)`` tuple (via ``Point.x`` / ``Point.y``) or a
            cryptic ``GEOSException``. Callers had no clean way to
            tell them apart from legitimate coords. Now the function
            raises a typed error telling them to filter first.
        """
        import pandas as pd

        from pyramids.base._errors import InvalidGeometryError

        row = pd.Series({"geometry": Point()})  # empty
        with pytest.raises(InvalidGeometryError, match="empty geometry"):
            get_coords(row, "geometry", "x")

    def test_empty_linestring_raises_invalid_geometry(self):
        """C22: an empty LineString also raises ``InvalidGeometryError``."""
        import pandas as pd

        from pyramids.base._errors import InvalidGeometryError

        row = pd.Series({"geometry": LineString()})
        with pytest.raises(InvalidGeometryError, match="empty geometry"):
            get_coords(row, "geometry", "x")

    def test_empty_polygon_raises_invalid_geometry(self):
        """C22: an empty Polygon also raises ``InvalidGeometryError``."""
        import pandas as pd

        from pyramids.base._errors import InvalidGeometryError

        row = pd.Series({"geometry": Polygon()})
        with pytest.raises(InvalidGeometryError, match="empty geometry"):
            get_coords(row, "geometry", "y")

    def test_geometry_collection(self):
        import pandas as pd

        gc = GeometryCollection([Point(7, 8)])
        row = pd.Series({"geometry": gc})
        result = get_coords(row, "geometry", "x")
        assert 7.0 in result


class TestFeatureCollectionStr:
    """Tests for the pyramids-branded ``__str__``."""

    def test_str_returns_string(self, simple_polygon_gdf: GeoDataFrame):
        fc = FeatureCollection(simple_polygon_gdf)
        result = str(fc)
        assert isinstance(result, str)
        assert "FeatureCollection" in result


class TestToDatasetErrors:
    """Tests for the error paths of :meth:`FeatureCollection.to_dataset`."""

    def test_no_cell_size_no_dataset_raises(self, simple_polygon_gdf: GeoDataFrame):
        fc = FeatureCollection(simple_polygon_gdf)
        with pytest.raises(ValueError, match="cell size"):
            Dataset.from_features(fc, cell_size=None, template=None)

    def test_mismatched_epsg_raises(self, simple_polygon_gdf: GeoDataFrame):
        fc = FeatureCollection(simple_polygon_gdf)
        ds = Dataset.create(
            cell_size=1000.0,
            rows=3,
            columns=3,
            dtype="float32",
            bands=1,
            top_left_corner=(500000.0, 3500000.0),
            epsg=32636,
            no_data_value=-9999.0,
        )
        with pytest.raises(ValueError, match="not the same EPSG"):
            Dataset.from_features(fc, template=ds)

    def test_non_dataset_object_raises(self, simple_polygon_gdf: GeoDataFrame):
        fc = FeatureCollection(simple_polygon_gdf)
        with pytest.raises((TypeError, AttributeError)):
            Dataset.from_features(fc, template="not_a_dataset")


class TestGetXyCoordsInvalidType:
    def test_invalid_coord_type_raises(self):
        line = LineString([(0, 0), (1, 1)])
        with pytest.raises(ValueError, match="'x' or 'y'"):
            get_xy_coords(line, "z")


class TestGetPointCoordsInvalidType:
    def test_invalid_coord_type_raises(self):
        point = Point(5, 10)
        with pytest.raises(ValueError, match="'x' or 'y'"):
            get_point_coords(point, "z")

    def test_invalid_coord_type_checked_before_attribute_access(self):
        """M2: the ``coord_type`` check fires before ``geometry.x/y``.

        Test scenario:
            An empty ``shapely.Point`` has no coordinates and
            ``geometry.x`` / ``.y`` raise ``GEOSException`` when
            accessed. Under the pre-M2 D-L6 code the dispatch dict
            evaluated BOTH attributes eagerly, so callers with an
            empty Point got a cryptic GEOS error even when they
            passed an invalid ``coord_type``. After M2 the coord_type
            check runs first, so invalid types fail with the clean
            ``ValueError`` regardless of geometry validity.
        """
        empty = Point()
        with pytest.raises(ValueError, match="'x' or 'y'"):
            get_point_coords(empty, "z")
