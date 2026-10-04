"""Tests for `FeatureCollection`'s geodesic measurements.

These are the geodesic counterparts of the planar `distance` / `length` / `area`
that `FeatureCollection` inherits from `GeoDataFrame`. The planar ones measure in
the CRS's own units, so on a geographic CRS they report degrees; every test here
pins the ground answer, and several contrast it with the inherited one.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from pyramids.base._errors import CRSError, InvalidGeometryError
from pyramids.feature import FeatureCollection

# One degree at the equator and at 60 N, in kilometres, on WGS 84.
DEGREE_AT_EQUATOR_KM = 111.31949079327357
DEGREE_AT_60N_KM = 55.79947039326038


def _fc(geometries: list, crs: str | int = "EPSG:4326") -> FeatureCollection:
    """Build a FeatureCollection from a list of geometries."""
    return FeatureCollection(gpd.GeoDataFrame(geometry=geometries, crs=crs))


class TestGeodesicLength:
    """Ground length of each geometry, against the inherited planar `length`."""

    def test_equatorial_degree_line(self):
        """A one-degree line at the equator is ~111.3 km."""
        fc = _fc([LineString([(0, 0), (1, 0)])])
        assert fc.geodesic_length(unit="km").iloc[0] == pytest.approx(
            DEGREE_AT_EQUATOR_KM
        )

    def test_differs_from_the_inherited_planar_length(self):
        """The inherited `length` reports 1.0 degree for the same line.

        This is the defect the method exists to fix, so it is pinned rather than
        described: the planar answer is a coordinate difference, not a distance.
        """
        fc = _fc([LineString([(0, 0), (1, 0)])])
        with pytest.warns(UserWarning, match="geographic CRS"):
            planar = fc.length.iloc[0]
        assert planar == pytest.approx(1.0)
        assert fc.geodesic_length().iloc[0] == pytest.approx(
            DEGREE_AT_EQUATOR_KM * 1000
        )

    def test_shrinks_towards_the_pole(self):
        """The same degree of longitude is half as long at 60 N."""
        fc = _fc([LineString([(0, 60), (1, 60)])])
        assert fc.geodesic_length(unit="km").iloc[0] == pytest.approx(DEGREE_AT_60N_KM)

    def test_polygon_reports_its_perimeter(self):
        """A polygon's length is its perimeter, as `pyproj.Geod` defines it."""
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert _fc([square]).geodesic_length(unit="km").iloc[0] == pytest.approx(
            443.770917248302
        )

    def test_point_has_no_length(self):
        """A point reports zero rather than raising."""
        assert _fc([Point(5.0, 45.0)]).geodesic_length().iloc[0] == pytest.approx(0.0)

    def test_projected_collection_is_reprojected_before_measuring(self):
        """A projected collection gives the same ground length as a geographic one.

        The geometries are reprojected to the datum's geographic CRS first, so the
        answer is a ground length either way -- within the round-trip precision of
        the projection.
        """
        geographic = _fc([LineString([(0, 0), (1, 0)])])
        projected = FeatureCollection(geographic.to_crs(3857))
        assert projected.geodesic_length(unit="km").iloc[0] == pytest.approx(
            DEGREE_AT_EQUATOR_KM, rel=1e-6
        )

    def test_index_is_preserved(self):
        """The result is indexed like the collection, not positionally."""
        frame = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 0)]), LineString([(0, 60), (1, 60)])],
            crs="EPSG:4326",
            index=["a", "b"],
        )
        result = FeatureCollection(frame).geodesic_length(unit="km")
        assert list(result.index) == ["a", "b"]
        assert result.loc["b"] == pytest.approx(DEGREE_AT_60N_KM)

    def test_empty_collection_returns_empty_series(self):
        """An empty collection measures to an empty result, not an error."""
        assert len(_fc([]).geodesic_length()) == 0

    def test_unknown_unit_raises(self):
        """An unrecognised length unit is refused."""
        fc = _fc([LineString([(0, 0), (1, 0)])])
        with pytest.raises(ValueError, match="unknown length unit"):
            fc.geodesic_length(unit="furlong")

    def test_no_crs_raises_crs_error(self):
        """Without a CRS nothing says what the coordinates measure."""
        fc = FeatureCollection(gpd.GeoDataFrame(geometry=[Point(0, 0)]))
        with pytest.raises(CRSError, match="has no CRS"):
            fc.geodesic_length()


class TestGeodesicArea:
    """Ground area of each geometry, against the inherited planar `area`."""

    def test_equatorial_degree_square(self):
        """A one-degree square at the equator covers ~12 309 km2."""
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert _fc([square]).geodesic_area(unit="km2").iloc[0] == pytest.approx(
            12308.778361469452
        )

    def test_differs_from_the_inherited_planar_area(self):
        """The inherited `area` reports 1.0 square degree for the same square."""
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        fc = _fc([square])
        with pytest.warns(UserWarning, match="geographic CRS"):
            planar = fc.area.iloc[0]
        assert planar == pytest.approx(1.0)
        assert fc.geodesic_area(unit="km2").iloc[0] == pytest.approx(12308.778361469452)

    def test_same_square_covers_less_ground_at_high_latitude(self):
        """A degree square at 60 N covers about half the ground of one at the equator.

        Two cells of one lat/lon grid report the identical planar area while
        covering very different ground; this is the number that distinguishes them.
        """
        at_equator = _fc([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
        at_60 = _fc([Polygon([(0, 60), (1, 60), (1, 61), (0, 61)])])
        equator_km2 = at_equator.geodesic_area(unit="km2").iloc[0]
        high_km2 = at_60.geodesic_area(unit="km2").iloc[0]
        assert high_km2 == pytest.approx(6122.943163071411)
        assert high_km2 < equator_km2 / 2

    def test_ring_orientation_does_not_change_the_size(self):
        """A clockwise ring reports the same area as a counter-clockwise one.

        `pyproj.Geod` signs the area by winding; the magnitude is what every caller
        means by area.
        """
        ccw = _fc([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
        cw = _fc([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
        assert cw.geodesic_area().iloc[0] == pytest.approx(ccw.geodesic_area().iloc[0])
        assert cw.geodesic_area().iloc[0] > 0.0

    @pytest.mark.parametrize(
        ("unit", "expected"),
        [
            ("m2", 12308778361.469452),
            ("km2", 12308.778361469452),
            ("ha", 1230877.8361469451),
        ],
    )
    def test_unit_conversion(self, unit: str, expected: float):
        """Each area unit is the square-metre answer divided by its size."""
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert _fc([square]).geodesic_area(unit=unit).iloc[0] == pytest.approx(expected)

    def test_line_encloses_nothing(self):
        """A line has no area."""
        assert _fc([LineString([(0, 0), (1, 0)])]).geodesic_area().iloc[0] == (
            pytest.approx(0.0)
        )

    def test_unknown_unit_raises(self):
        """An unrecognised area unit is refused."""
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        with pytest.raises(ValueError, match="unknown area unit"):
            _fc([square]).geodesic_area(unit="acres")

    def test_agrees_with_the_raster_cell_area(self):
        """A degree square's area matches what `Dataset.cell_area` reports for one.

        The two use different algorithms on purpose -- a geodesic polygon here, a
        closed-form zone integral there -- so agreeing to four significant figures
        is a cross-check on both. They do not agree exactly, and `cell_area`'s
        docstring explains why: a geodesic polygon bows its parallel edges.
        """
        import numpy as np

        from pyramids.dataset import Dataset, GeoReference

        geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
        raster = Dataset.from_array(np.ones((1, 1), "float32"), geo_ref=geo_ref)
        raster_km2 = float(raster.cell_area(unit="km2")[0, 0])
        vector_km2 = _fc([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]).geodesic_area(
            unit="km2"
        )
        assert vector_km2.iloc[0] == pytest.approx(raster_km2, rel=1e-3)


class TestGeodesicDistance:
    """Ground distance from each point to a target, against planar `distance`."""

    def test_distance_to_a_single_point(self):
        """A single shapely point is measured against every feature."""
        fc = _fc([Point(0, 0), Point(0, 60)])
        result = fc.geodesic_distance(Point(1, 0), unit="km")
        assert result.iloc[0] == pytest.approx(DEGREE_AT_EQUATOR_KM)
        assert result.iloc[1] == pytest.approx(6655.0, rel=1e-3)

    def test_differs_from_the_inherited_planar_distance(self):
        """The inherited `distance` reports 1.0 degree for the equatorial pair."""
        fc = _fc([Point(0, 0)])
        with pytest.warns(UserWarning, match="geographic CRS"):
            planar = fc.distance(Point(1, 0)).iloc[0]
        assert planar == pytest.approx(1.0)
        assert fc.geodesic_distance(Point(1, 0), unit="km").iloc[0] == pytest.approx(
            DEGREE_AT_EQUATOR_KM
        )

    def test_elementwise_against_a_geoseries(self):
        """A collection of points is compared elementwise, in order."""
        fc = _fc([Point(0, 0), Point(0, 60)])
        targets = gpd.GeoSeries([Point(1, 0), Point(1, 60)], crs="EPSG:4326")
        result = fc.geodesic_distance(targets, unit="km")
        assert result.iloc[0] == pytest.approx(DEGREE_AT_EQUATOR_KM)
        assert result.iloc[1] == pytest.approx(DEGREE_AT_60N_KM)

    def test_elementwise_against_a_feature_collection(self):
        """Another FeatureCollection works as a target too."""
        fc = _fc([Point(0, 0)])
        result = fc.geodesic_distance(_fc([Point(1, 0)]), unit="km")
        assert result.iloc[0] == pytest.approx(DEGREE_AT_EQUATOR_KM)

    def test_identical_points_are_zero(self):
        """A point is no distance from itself."""
        fc = _fc([Point(12.5, 41.9)])
        assert fc.geodesic_distance(Point(12.5, 41.9)).iloc[0] == pytest.approx(0.0)

    def test_target_in_another_crs_is_reprojected(self):
        """A target carrying a different CRS is brought into this one first."""
        fc = _fc([Point(0, 0)])
        target = gpd.GeoSeries([Point(1, 0)], crs="EPSG:4326").to_crs(3857)
        result = fc.geodesic_distance(target, unit="km")
        assert result.iloc[0] == pytest.approx(DEGREE_AT_EQUATOR_KM, rel=1e-6)

    def test_length_mismatch_raises(self):
        """Elementwise comparison needs matching lengths."""
        fc = _fc([Point(0, 0), Point(0, 60)])
        targets = gpd.GeoSeries([Point(1, 0)], crs="EPSG:4326")
        with pytest.raises(ValueError, match="compared elementwise"):
            fc.geodesic_distance(targets)

    def test_non_point_source_raises(self):
        """A line has no single point to measure from."""
        fc = _fc([LineString([(0, 0), (1, 0)])])
        with pytest.raises(InvalidGeometryError, match="only point geometries"):
            fc.geodesic_distance(Point(0, 0))

    def test_non_point_target_raises(self):
        """Nor does a line target."""
        fc = _fc([Point(0, 0)])
        with pytest.raises(InvalidGeometryError, match="only point geometries"):
            fc.geodesic_distance(LineString([(0, 0), (1, 0)]))

    def test_error_names_the_alternative(self):
        """The refusal points at `geodesic_length`, so it is actionable."""
        fc = _fc([LineString([(0, 0), (1, 0)])])
        with pytest.raises(InvalidGeometryError, match="geodesic_length"):
            fc.geodesic_distance(Point(0, 0))

    def test_index_is_preserved(self):
        """The result is indexed like the collection."""
        frame = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(0, 60)], crs="EPSG:4326", index=[7, 9]
        )
        result = FeatureCollection(frame).geodesic_distance(Point(1, 0), unit="km")
        assert list(result.index) == [7, 9]
        assert result.loc[7] == pytest.approx(DEGREE_AT_EQUATOR_KM)

    def test_unknown_unit_raises(self):
        """An unrecognised length unit is refused."""
        fc = _fc([Point(0, 0)])
        with pytest.raises(ValueError, match="unknown length unit"):
            fc.geodesic_distance(Point(1, 0), unit="furlong")

    def test_no_crs_raises_crs_error(self):
        """Without a CRS nothing says what the coordinates measure."""
        fc = FeatureCollection(gpd.GeoDataFrame(geometry=[Point(0, 0)]))
        with pytest.raises(CRSError, match="has no CRS"):
            fc.geodesic_distance(Point(1, 0))
