"""Integration tests for FeatureCollection (GeoDataFrame-subclass design).

These tests focus on behavior that survives the ARC-1a / ARC-1b refactor:
GeoDataFrame-only construction, the pyramids-specific properties
(``epsg``, ``total_bounds``, ``top_left_corner``, ``column``, ``dtypes``),
I/O (``read_file``, ``to_file``), geometry factories, rasterization
(``to_dataset``), xy / center_point, concate.

Tests that exercised the old OGR-accepting public surface
(``create_ds``, ``_copy_driver_to_memory``, ``_ds_to_gdf``,
``_gdf_to_ds``, the ``.feature`` property, ``layers_count``,
``layer_names``, ``file_name``) were removed — those surfaces no
longer exist.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from shapely.geometry.polygon import Polygon

from pyramids.dataset import Dataset
from pyramids.feature import (
    FeatureCollection,
    create_points,
    create_polygon,
    multi_geom_handler,
    point_collection,
    polygon_wkt,
)

pytestmark = pytest.mark.core


class TestAttributes:
    def test_total_bound_gdf(self, gdf: GeoDataFrame, gdf_bound: List):
        feature = FeatureCollection(gdf)
        assert all(np.isclose(feature.total_bounds, gdf_bound, rtol=0.0001))

    def test_top_left_corner_gdf(self, gdf: GeoDataFrame, gdf_bound: List):
        feature = FeatureCollection(gdf)
        assert feature.top_left_corner == [gdf_bound[0], gdf_bound[-1]]

    def test_columns_gdf(self, coello_gauges_gdf: GeoDataFrame):
        feature = FeatureCollection(coello_gauges_gdf)
        assert feature.column == ["id", "x", "y", "geometry"]


class TestReadFile:
    def test_read_returns_featurecollection(self, test_vector_path: str):
        fc = FeatureCollection.read_file(test_vector_path)
        # FeatureCollection IS a GeoDataFrame after ARC-1a, so this must hold.
        assert isinstance(fc, FeatureCollection)
        assert isinstance(fc, GeoDataFrame)


class TestToFile:
    def test_save_gdf(self, gdf: GeoDataFrame, test_save_vector_path: Path):
        vector = FeatureCollection(gdf)
        vector.to_file(test_save_vector_path)
        assert test_save_vector_path.exists()
        test_save_vector_path.unlink()


class TestCreatePolygon:
    """ARC-15: create_polygon is unconditional-Polygon; polygon_wkt gives WKT."""

    def test_create_polygon_returns_polygon(self, coordinates: List[Tuple[int, int]]):
        result = create_polygon(coordinates)
        assert isinstance(result, Polygon)

    def test_polygon_wkt(
        self, coordinates: List[Tuple[int, int]], coordinates_wkt: str
    ):
        """``polygon_wkt`` returns the WKT string (no deprecation)."""
        wkt = polygon_wkt(coordinates)
        assert isinstance(wkt, str)
        assert wkt == coordinates_wkt

    def test_create_polygon_wkt_kwarg_is_gone(self, coordinates: List[Tuple[int, int]]):
        """D-H2: the ARC-15 ``wkt=`` kwarg is deleted outright.

        Test scenario:
            Callers who wrote ``create_polygon(coords, wkt=True)`` must
            migrate to :meth:`polygon_wkt` — the polymorphic kwarg no
            longer exists, so the call surfaces a ``TypeError`` naming
            the unknown kwarg.
        """
        import pytest as _pt

        with _pt.raises(TypeError, match="wkt"):
            create_polygon(coordinates, wkt=True)

    def test_create_polygon_too_few_vertices_raises(self):
        """C21: fewer than 3 vertices raises ``InvalidGeometryError``."""
        import pytest as _pt

        from pyramids.base._errors import InvalidGeometryError

        with _pt.raises(InvalidGeometryError, match="at least 3 vertices"):
            create_polygon([(0, 0), (1, 1)])

    def test_create_polygon_zero_vertices_raises(self):
        """C21: empty input raises ``InvalidGeometryError``."""
        import pytest as _pt

        from pyramids.base._errors import InvalidGeometryError

        with _pt.raises(InvalidGeometryError, match="at least 3 vertices"):
            create_polygon([])

    def test_create_polygon_exactly_three_vertices_ok(self):
        """C21 boundary: exactly 3 vertices is accepted.

        Test scenario:
            The threshold is "at least 3". A coord list of length 3
            is the minimum-valid input and must construct a Polygon
            successfully.
        """
        triangle = create_polygon([(0, 0), (1, 0), (0, 1)])
        assert isinstance(triangle, Polygon)
        assert not triangle.is_empty


class TestCreatePoint:
    """ARC-15: create_points is the list form; point_collection is the FC form."""

    def test_create_points_returns_list(self, coordinates: List[Tuple[int, int]]):
        pts = create_points(coordinates)
        assert isinstance(pts, list)
        assert len(pts) == len(coordinates)

    def test_point_collection_returns_fc(self, coordinates: List[Tuple[int, int]]):
        fc = FeatureCollection(point_collection(coordinates, crs=4326))
        assert isinstance(fc, FeatureCollection)
        assert len(fc["geometry"]) == len(coordinates)
        assert fc.epsg == 4326

    def test_create_point_method_is_gone(self, coordinates: List[Tuple[int, int]]):
        """D-H2: the polymorphic ``create_point`` dispatcher is deleted.

        Test scenario:
            ARC-15 split this into :meth:`create_points` (list) and
            :meth:`point_collection` (FC). The polymorphic
            ``create_point`` pass-through is gone outright; the
            attribute simply doesn't exist on the class anymore.
        """
        assert not hasattr(FeatureCollection, "create_point"), (
            "FeatureCollection.create_point must be deleted (D-H2). "
            "Use create_points() or point_collection() instead."
        )


class TestToDataset:
    """Test rasterization via :meth:`FeatureCollection.to_dataset`."""

    class TestSingleColumnVector:
        def test_single_band_dataset_parameter(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            dataset = Dataset.read_file(raster_1band_coello_path)
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = Dataset.from_features(vector, template=dataset)
            assert src.epsg == polygon_corner_coello_gdf.crs.to_epsg()

            xmin, _, _, ymax, _, _ = dataset.geotransform
            assert src.geotransform[0] == xmin
            assert src.geotransform[3] == ymax
            assert src.cell_size == 4000
            arr = src.read_array()
            values = arr[~np.isclose(arr, src.no_data_value[0])]
            assert values.shape[0] == 16
            assert values.mean() == vector["fid"].values[0]

    class TestMultiColumnVector:
        def test_single_band_dataset_parameter_one_column(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            dataset = Dataset.read_file(raster_1band_coello_path)
            polygon_corner_coello_gdf["column_2"] = 2
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = Dataset.from_features(
                vector, template=dataset, column_name="column_2"
            )
            assert src.epsg == polygon_corner_coello_gdf.crs.to_epsg()

            arr = src.read_array()
            values = arr[~np.isclose(arr, src.no_data_value[0])]
            assert values.shape[0] == 16
            assert values.mean() == 2

        def test_single_band_dataset_parameter_multiple_columns(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            dataset = Dataset.read_file(raster_1band_coello_path)
            polygon_corner_coello_gdf["column_2"] = 2
            polygon_corner_coello_gdf["column_3"] = 3
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = Dataset.from_features(
                vector, template=dataset, column_name=["column_2", "column_3"]
            )
            arr = src.read_array()
            assert arr.shape == (2, 13, 14)
            band_1_values = arr[0, ~np.isclose(arr[0, :, :], src.no_data_value[0])]
            band_2_values = arr[1, ~np.isclose(arr[1, :, :], src.no_data_value[1])]
            assert band_1_values.mean() == 2
            assert band_2_values.mean() == 3

        def test_single_band_dataset_parameter_none_column_name(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            dataset = Dataset.read_file(raster_1band_coello_path)
            polygon_corner_coello_gdf["column_2"] = 2
            polygon_corner_coello_gdf["column_3"] = 3

            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = Dataset.from_features(vector, template=dataset, column_name=None)

            arr = src.read_array()
            assert arr.shape == (3, 13, 14)

    class TestCellSize:
        def test_none_column_name(self, polygon_corner_coello_gdf: GeoDataFrame):
            polygon_corner_coello_gdf["column_2"] = 2
            polygon_corner_coello_gdf["column_3"] = 3
            required_cell_size = 8000

            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = Dataset.from_features(
                vector, cell_size=required_cell_size, column_name=None
            )
            assert src.cell_size == required_cell_size
            arr = src.read_array()
            assert arr.shape == (3, 2, 2)

        def test_one_column(self, polygon_corner_coello_gdf: GeoDataFrame):
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = Dataset.from_features(vector, cell_size=200)
            assert src.cell_size == 200
            arr = src.read_array()
            values = arr[np.isclose(arr[:, :], 1.0)]
            assert values.shape[0] == 6400

        def test_multi_polygon(self, polygons_coello_gdf: GeoDataFrame):
            required_cell_size = 4000
            vector = FeatureCollection(polygons_coello_gdf)
            src = Dataset.from_features(
                vector, cell_size=required_cell_size, column_name=None
            )
            arr = src.read_array()
            assert arr.shape == (3, 13, 13)


class TestMultiGeomHandler:
    def test_multi_points_with_multiple_point(
        self, multi_point_geom, point_coords: list
    ):
        res = multi_geom_handler(multi_point_geom, "x", "MultiPoint")
        assert all(np.isclose(res, [point_coords[0], point_coords[0]], rtol=0.00001))

    def test_multi_points_with_one_point(
        self, multi_point_one_point_geom, point_coords: list
    ):
        res = multi_geom_handler(multi_point_one_point_geom, "x", "MultiPoint")
        assert np.isclose(res[0], point_coords[0], rtol=0.00001)

    def test_multi_polygons(self, multi_polygon_geom, multi_polygon_coords_x: list):
        res = multi_geom_handler(multi_polygon_geom, "x", "MultiPolygon")
        assert res == multi_polygon_coords_x

    def test_multi_linestring(self, multi_line_geom, multi_linestring_coords_x: list):
        res = multi_geom_handler(multi_line_geom, "x", "MULTILINESTRING")
        assert res == multi_linestring_coords_x


class TestWithCoordinates:
    """ARC-16: with_coordinates() — non-mutating, returns a new FC."""

    def test_points(self, points_gdf: GeoDataFrame, points_gdf_x, points_gdf_y):
        feature = FeatureCollection(points_gdf)
        result = feature.with_coordinates()
        assert result is not feature, "must return a new object, not self"
        assert "x" not in feature.columns, "self must not be mutated"
        assert all(np.isclose(result["y"].values, points_gdf_y, rtol=0.000001))
        assert all(np.isclose(result["x"].values, points_gdf_x, rtol=0.000001))

    def test_multi_points(
        self, multi_points_gdf: GeoDataFrame, multi_points_gdf_x, multi_points_gdf_y
    ):
        feature = FeatureCollection(multi_points_gdf)
        result = feature.with_coordinates()
        assert all(
            np.isclose(result.loc[0, "y"], multi_points_gdf_y[0][0], rtol=0.000001)
        )
        assert all(
            np.isclose(result.loc[0, "x"], multi_points_gdf_x[0][0], rtol=0.000001)
        )

    def test_multi_points_2(self, multi_points_gdf_2: GeoDataFrame, point_coords_2):
        feature = FeatureCollection(multi_points_gdf_2)
        result = feature.with_coordinates()
        assert result.loc[0, "x"] == [point_coords_2[0][0], point_coords_2[1][0]]
        assert result.loc[0, "y"] == [point_coords_2[0][1], point_coords_2[1][1]]

    def test_polygons(
        self, polygons_gdf: GeoDataFrame, polygon_gdf_x: list, polygon_gdf_y: list
    ):
        feature = FeatureCollection(polygons_gdf)
        result = feature.with_coordinates()
        assert isinstance(result.loc[0, "y"], list)
        assert all(np.isclose(result.loc[0, "y"], polygon_gdf_y, rtol=0.000001))
        assert all(np.isclose(result.loc[0, "x"], polygon_gdf_x, rtol=0.000001))

    def test_multi_polygons(
        self,
        multi_polygon_gdf: GeoDataFrame,
        multi_polygon_gdf_coords_x: list,
    ):
        feature = FeatureCollection(multi_polygon_gdf)
        result = feature.with_coordinates()
        assert isinstance(result.loc[0, "x"], list)
        assert all(
            np.isclose(result.loc[0, "x"], multi_polygon_gdf_coords_x, rtol=0.000001)
        )

    def test_geometry_collection(
        self,
        geometry_collection_gdf: GeoDataFrame,
        multi_polygon_gdf_coords_x: list,
    ):
        feature = FeatureCollection(geometry_collection_gdf)
        result = feature.with_coordinates()
        assert result.loc[0, "x"] == pytest.approx(100.0)
        assert result.loc[1, "x"] == [101.0, 102.0]
        assert result.loc[2, "x"] == [
            460717.3717217822,
            456004.5874004898,
            456929.2331169145,
            459285.1699671757,
            462651.74958306097,
            460717.3717217822,
        ]

    def test_returns_featurecollection(self, points_gdf: GeoDataFrame):
        """Subclass identity preserved on the returned object."""
        feature = FeatureCollection(points_gdf)
        assert isinstance(feature.with_coordinates(), FeatureCollection)


class TestConcat:
    """ARC-16: concat() mirrors pd.concat — returns a new FC, no inplace."""

    def test_returns_new_fc(self, geometry_collection_gdf: GeoDataFrame):
        feature = FeatureCollection(geometry_collection_gdf)
        result = feature.concat(geometry_collection_gdf)
        assert isinstance(result, FeatureCollection)
        assert result is not feature, "must not return self"
        assert len(result) == 2
        assert result.loc[0, "geometry"].geom_type == "GeometryCollection"

    def test_does_not_mutate_self(self, geometry_collection_gdf: GeoDataFrame):
        feature = FeatureCollection(geometry_collection_gdf)
        before = len(feature)
        _ = feature.concat(geometry_collection_gdf)
        assert len(feature) == before, "self must not be mutated"

    def test_pd_concat_idiom_also_works(self, geometry_collection_gdf: GeoDataFrame):
        """pd.concat([fc, other]) returns a FeatureCollection via _constructor."""
        import pandas as pd

        feature = FeatureCollection(geometry_collection_gdf)
        result = pd.concat([feature, geometry_collection_gdf])
        assert isinstance(result, FeatureCollection)
        assert len(result) == 2

    def test_concat_raises_on_crs_mismatch(self):
        """C32: concatenating two FCs in different CRSes raises CRSError.

        Test scenario:
            Silent CRS adoption would corrupt the ``other`` rows'
            coordinates. Force the caller to reproject first.
        """
        import pytest as _pt
        from shapely.geometry import Point as _Pt

        from pyramids.base._errors import CRSError

        fc_a = FeatureCollection(
            GeoDataFrame({"v": [1]}, geometry=[_Pt(0, 0)], crs="EPSG:4326")
        )
        fc_b = GeoDataFrame({"v": [2]}, geometry=[_Pt(1, 1)], crs="EPSG:3857")

        with _pt.raises(CRSError, match="CRS mismatch"):
            fc_a.concat(fc_b)

    def test_concat_permits_none_crs_on_one_side(self):
        """C32 negative: an unset CRS on either side is allowed.

        Test scenario:
            A freshly-constructed empty FC (crs=None) should still be
            able to absorb a CRS-carrying frame. The result adopts
            whichever side has a CRS.
        """
        from shapely.geometry import Point as _Pt

        fc_a = FeatureCollection(GeoDataFrame({"v": [1]}, geometry=[_Pt(0, 0)]))
        fc_b = GeoDataFrame({"v": [2]}, geometry=[_Pt(1, 1)], crs="EPSG:4326")

        result = fc_a.concat(fc_b)
        assert result.epsg == 4326
        assert len(result) == 2


def test_with_centroid(polygons_gdf: GeoDataFrame):
    feature = FeatureCollection(polygons_gdf)
    result = feature.with_centroid()
    assert result is not feature, "must return a new object"
    assert "center_point" in result.columns
    assert "center_point" not in feature.columns, "self must not be mutated"


class TestWithCentroidDegenerateGeometries:
    """C18: with_centroid handles empty / zero-area geometries safely.

    Feeding a degenerate geometry used to produce a ``(NaN, NaN)``
    Point silently — downstream reprojection / distance ops would
    then crash on the invalid coordinates. Now the method (1) emits a
    ``UserWarning`` naming the offending row indices and (2)
    substitutes an empty ``shapely.Point`` so the column invariant
    ("non-NaN point or empty point") holds.
    """

    def test_nan_coord_point_emits_warning(self):
        """Test scenario: one valid point + one NaN-coord point → one warning.

        ``Point(float('nan'), float('nan'))`` is not empty — it carries
        NaN coordinates which propagate through the np.mean call and
        produce a NaN average. C18 detects that and warns.
        """
        import warnings

        from shapely.geometry import Point as _Pt

        gdf = GeoDataFrame(
            {"v": [1, 2]},
            geometry=[_Pt(3, 4), _Pt(float("nan"), float("nan"))],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            fc.with_centroid()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert any("NaN centroids" in str(w.message) for w in user_warnings), (
            f"expected NaN-centroids warning; got {user_warnings}"
        )

    def test_nan_coord_row_substituted_with_empty_point(self):
        """Test scenario: the NaN-coord row's center_point is empty."""
        import warnings

        from shapely.geometry import Point as _Pt

        gdf = GeoDataFrame(
            {"v": [1, 2]},
            geometry=[_Pt(3, 4), _Pt(float("nan"), float("nan"))],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = fc.with_centroid()

        centers = result["center_point"].tolist()
        assert not centers[0].is_empty
        assert centers[1].is_empty

    def test_no_warning_for_all_valid_geometries(self):
        """Test scenario: every row has a valid geometry → no warning."""
        import warnings

        from shapely.geometry import Point as _Pt

        gdf = GeoDataFrame(
            {"v": [1, 2]},
            geometry=[_Pt(0, 0), _Pt(1, 1)],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            fc.with_centroid()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not any("NaN centroids" in str(w.message) for w in user_warnings), (
            f"should not warn for valid geometries; got {user_warnings}"
        )

    def test_warning_is_geometry_warning_category(self):
        """L6: the NaN-centroid warning uses ``GeometryWarning``.

        Test scenario:
            A caller must be able to suppress just pyramids geometry
            warnings with ``warnings.filterwarnings("ignore",
            category=GeometryWarning)`` — silencing the plain
            ``UserWarning`` would also silence unrelated geopandas /
            shapely warnings.
        """
        import warnings as _w

        from shapely.geometry import Point as _Pt

        from pyramids.base._errors import GeometryWarning

        nan = float("nan")
        gdf = GeoDataFrame(
            {"v": [1, 2]},
            geometry=[_Pt(0, 0), _Pt(nan, nan)],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always", GeometryWarning)
            fc.with_centroid()
        geo_warnings = [w for w in caught if issubclass(w.category, GeometryWarning)]
        assert geo_warnings, (
            f"expected a GeometryWarning; got {[w.category for w in caught]}"
        )

    def test_warning_can_be_filtered_by_category(self):
        """L6: ``filterwarnings("ignore", category=GeometryWarning)`` works.

        Test scenario:
            Install the filter, call ``with_centroid`` on a
            degenerate FC, and assert no warnings were captured. A
            plain-UserWarning-only warning would have needed a
            message-match filter to silence.
        """
        import warnings as _w

        from shapely.geometry import Point as _Pt

        from pyramids.base._errors import GeometryWarning

        nan = float("nan")
        gdf = GeoDataFrame(
            {"v": [1, 2]},
            geometry=[_Pt(0, 0), _Pt(nan, nan)],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            _w.filterwarnings("ignore", category=GeometryWarning)
            fc.with_centroid()
        geo_warnings = [w for w in caught if issubclass(w.category, GeometryWarning)]
        assert not geo_warnings, (
            f"GeometryWarning should have been filtered out; got {geo_warnings}"
        )

    def test_warning_lists_all_bad_row_indices(self):
        """C18: multiple NaN rows are all named in the warning message.

        Test scenario:
            When two rows produce NaN centroids, the warning message
            lists both indices so callers can filter them in one pass
            without re-running the detection.
        """
        import warnings

        from shapely.geometry import Point as _Pt

        nan = float("nan")
        gdf = GeoDataFrame(
            {"v": [1, 2, 3, 4]},
            geometry=[
                _Pt(0, 0),
                _Pt(nan, nan),
                _Pt(5, 5),
                _Pt(nan, nan),
            ],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            fc.with_centroid()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        bodies = " ".join(str(w.message) for w in user_warnings)
        assert "2 row(s)" in bodies or "[1, 3]" in bodies, (
            f"warning should summarise the two bad rows; got {bodies}"
        )


def test_old_names_are_gone():
    """ARC-16: xy / center_point / concatenate / concate removed outright."""
    assert not hasattr(FeatureCollection, "xy")
    assert not hasattr(FeatureCollection, "center_point")
    assert not hasattr(FeatureCollection, "concatenate")
    assert not hasattr(FeatureCollection, "concate")


def test_no_inplace_kwarg_on_public_methods():
    """ARC-16 regression: no pyramids-specific method exposes inplace=.

    Scans the class for any public method whose signature contains an
    ``inplace`` parameter; none of our pyramids-added methods should.
    (Inherited pandas/geopandas methods that still carry ``inplace``
    are not our concern — filter them out by only checking attributes
    declared on FeatureCollection itself.)
    """
    import inspect

    own = FeatureCollection.__dict__
    offenders = []
    for name, obj in own.items():
        if name.startswith("_"):
            continue
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        if "inplace" in sig.parameters:
            offenders.append(name)
    assert not offenders, (
        f"pyramids methods must not have inplace= (ARC-16): {offenders}"
    )
