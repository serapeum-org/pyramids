"""Tests for `FeatureCollection`'s geodesic measurements.

These are the geodesic counterparts of the planar `distance` / `length` / `area`
that `FeatureCollection` inherits from `GeoDataFrame`. The planar ones measure in
the CRS's own units, so on a geographic CRS they report degrees; every test here
pins the ground answer, and several contrast it with the inherited one.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

import pyramids.feature.collection as collection_module
from pyramids.base._errors import CRSError, InvalidGeometryError
from pyramids.dataset import Dataset, GeoReference
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
        fc = _fc([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
        with pytest.raises(ValueError, match="unknown area unit"):
            fc.geodesic_area(unit="acres")

    def test_agrees_with_the_raster_cell_area(self):
        """A degree square's area matches what `Dataset.cell_area` reports for one.

        The two use different algorithms on purpose -- a geodesic polygon here, a
        closed-form zone integral there -- so agreeing to four significant figures
        is a cross-check on both. They do not agree exactly, and `cell_area`'s
        docstring explains why: a geodesic polygon bows its parallel edges.
        """
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
        target = Point(1, 0)
        with pytest.warns(UserWarning, match="geographic CRS"):
            planar_series = fc.distance(target)
        assert planar_series.iloc[0] == pytest.approx(1.0)
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
        origin = Point(0, 0)
        with pytest.raises(InvalidGeometryError, match="only point geometries"):
            fc.geodesic_distance(origin)

    def test_non_point_target_raises(self):
        """Nor does a line target."""
        fc = _fc([Point(0, 0)])
        line = LineString([(0, 0), (1, 0)])
        with pytest.raises(InvalidGeometryError, match="only point geometries"):
            fc.geodesic_distance(line)

    def test_error_names_the_alternative(self):
        """The refusal points at `geodesic_length`, so it is actionable."""
        fc = _fc([LineString([(0, 0), (1, 0)])])
        origin = Point(0, 0)
        with pytest.raises(InvalidGeometryError, match="geodesic_length"):
            fc.geodesic_distance(origin)

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
        target = Point(1, 0)
        with pytest.raises(ValueError, match="unknown length unit"):
            fc.geodesic_distance(target, unit="furlong")

    def test_no_crs_raises_crs_error(self):
        """Without a CRS nothing says what the coordinates measure."""
        fc = FeatureCollection(gpd.GeoDataFrame(geometry=[Point(0, 0)]))
        target = Point(1, 0)
        with pytest.raises(CRSError, match="has no CRS"):
            fc.geodesic_distance(target)


class TestGradUnitGeodeticCRS:
    """A grad-axis geodetic counterpart must not be read as degrees (M1 sibling).

    `_geodesic_geometries` reprojects a projected collection into its datum's
    geographic CRS, which for the legacy French Lambert zones is `NTF (Paris)`
    with **grad** axes. Handing those to `pyproj.Geod` as degrees inflates every
    measurement, exactly as it did in `ground_distance_in_crs`.
    """

    @staticmethod
    def _reference_km(line: LineString, code: int) -> float:
        """Ground length of `line` routed explicitly through EPSG:4326."""
        native = gpd.GeoSeries([line], crs=code)
        in_degrees = native.to_crs(4326).iloc[0]
        geod = gpd.GeoSeries([line], crs=code).crs.get_geod()
        return geod.geometry_length(in_degrees) / 1000.0

    def test_geodesic_length_on_a_grad_datum_collection(self):
        """A Lambert-zone collection measures the same ground length as via WGS 84.

        The ~5e-5 tolerance absorbs the NTF-to-WGS84 datum shift the reference
        route introduces and this path avoids; the bug it pins was 11-45%.
        """
        line = LineString([(600000.0, 2200000.0), (700000.0, 2200000.0)])
        fc = _fc([line], crs=27562)
        assert fc.geodesic_length(unit="km").iloc[0] == pytest.approx(
            self._reference_km(line, 27562), rel=1e-3
        )

    def test_geodesic_area_on_a_grad_datum_collection(self):
        """Same for area, which goes through the same reprojection."""
        square = Polygon(
            [
                (600000.0, 2200000.0),
                (700000.0, 2200000.0),
                (700000.0, 2300000.0),
                (600000.0, 2300000.0),
            ]
        )
        fc = _fc([square], crs=27562)
        reference = gpd.GeoSeries([square], crs=27562).to_crs(4326).iloc[0]
        geod = gpd.GeoSeries([square], crs=27562).crs.get_geod()
        expected_km2 = abs(geod.geometry_area_perimeter(reference)[0]) / 1e6
        assert fc.geodesic_area(unit="km2").iloc[0] == pytest.approx(
            expected_km2, rel=1e-3
        )

    def test_geodesic_distance_on_a_grad_datum_collection(self):
        """And for distance, whose targets take the same reprojection path."""
        fc = _fc([Point(600000.0, 2200000.0)], crs=27562)
        target = gpd.GeoSeries([Point(700000.0, 2200000.0)], crs=27562)
        reference = gpd.GeoSeries(
            [Point(600000.0, 2200000.0), Point(700000.0, 2200000.0)], crs=27562
        ).to_crs(4326)
        geod = gpd.GeoSeries([Point(0, 0)], crs=27562).crs.get_geod()
        _, _, expected_m = geod.inv(
            reference.iloc[0].x,
            reference.iloc[0].y,
            reference.iloc[1].x,
            reference.iloc[1].y,
        )
        assert fc.geodesic_distance(target).iloc[0] == pytest.approx(
            expected_m, rel=1e-3
        )


class TestDegreeFactorComparison:
    """The degrees-per-unit factor is compared with a tolerance, not `==`.

    `to_degrees` is a quotient, so a degree CRS whose stored
    `unit_conversion_factor` differs from `math.pi / 180` in its last bit would
    fail an exact `!= 1.0` test and have every vertex pushed through `scale`.
    No CRS in the EPSG database does that today, so the factor is injected.

    The assertion is on the *branch*, not the number: rescaling by `1 + 2e-16`
    is invisible at float precision, so comparing measurements cannot tell the
    two implementations apart. Recording whether `GeoSeries.scale` is called
    can, and does -- these tests fail against an exact `!=` comparison.
    """

    @staticmethod
    def _spy_on_scale(monkeypatch) -> list:
        """Record every `GeoSeries.scale` call, forwarding to the real one."""
        calls: list = []
        real_scale = gpd.GeoSeries.scale

        def _recording(self, *args, **kwargs):
            calls.append(kwargs.get("xfact"))
            return real_scale(self, *args, **kwargs)

        monkeypatch.setattr(gpd.GeoSeries, "scale", _recording)
        return calls

    @staticmethod
    def _force_factor(monkeypatch, factor: float) -> None:
        """Make `_geodetic_frame` report `factor` degrees per unit."""
        real_frame = collection_module._geodetic_frame

        def _forced(crs):
            geodetic, _ = real_frame(crs)
            return geodetic, factor

        monkeypatch.setattr(collection_module, "_geodetic_frame", _forced)

    def test_a_factor_one_ulp_from_one_does_not_rescale(self, monkeypatch):
        """`1 + 2e-16` is degrees, so no rescale happens at all."""
        fc = _fc([LineString([(0, 0), (1, 0)])])
        calls = self._spy_on_scale(monkeypatch)
        self._force_factor(monkeypatch, 1.0 + 2e-16)
        fc.geodesic_length(unit="km")
        assert calls == []

    def test_the_grad_factor_still_rescales(self, monkeypatch):
        """The control: 0.9 is far outside the tolerance and must rescale."""
        fc = _fc([LineString([(0, 0), (1, 0)])])
        calls = self._spy_on_scale(monkeypatch)
        self._force_factor(monkeypatch, 0.9)
        measured = fc.geodesic_length(unit="km").iloc[0]
        assert calls == [0.9]
        assert measured == pytest.approx(DEGREE_AT_EQUATOR_KM * 0.9, rel=1e-3)


class TestTargetFrameReconciliation:
    """Every target is brought into the collection's frame first (R2 M1, M2).

    `Geod.inv` pairs two coordinates; they are only comparable if they are
    expressed in the same frame. Both halves of this went wrong independently: a
    bare shapely geometry was taken to be already in degrees, and a target
    carrying its own CRS was converted to *its* datum's geographic counterpart
    rather than the collection's.
    """

    EQUATOR_KM = DEGREE_AT_EQUATOR_KM

    def test_bare_geometry_is_read_in_the_collections_crs(self):
        """A bare point on a projected collection is in that collection's CRS.

        It used to be read as degrees: an easting of 111319.49 became a longitude,
        which `Geod` wrapped to 79 degrees and reported as 8848.87 km -- a 79x
        error with no warning.
        """
        geographic = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)
        projected = FeatureCollection(geographic.to_crs(3857))
        target = gpd.GeoSeries([Point(1, 0)], crs=4326).to_crs(3857).iloc[0]
        assert projected.geodesic_distance(target, unit="km").iloc[0] == (
            pytest.approx(self.EQUATOR_KM, rel=1e-9)
        )

    def test_the_three_spellings_of_one_target_agree(self):
        """A bare geometry, a CRS-less GeoSeries and a CRS-bearing one must match.

        They used to disagree: the first took a branch that skipped conversion
        entirely while the other two reprojected.
        """
        geographic = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)
        projected = FeatureCollection(geographic.to_crs(3857))
        point = gpd.GeoSeries([Point(1, 0)], crs=4326).to_crs(3857).iloc[0]
        bare = projected.geodesic_distance(point, unit="km").iloc[0]
        crsless = projected.geodesic_distance(gpd.GeoSeries([point]), unit="km").iloc[0]
        tagged = projected.geodesic_distance(
            gpd.GeoSeries([point], crs=3857), unit="km"
        ).iloc[0]
        assert bare == pytest.approx(crsless, rel=1e-12)
        assert bare == pytest.approx(tagged, rel=1e-12)

    def test_cross_crs_target_keeps_the_prime_meridian(self):
        """One physical point is zero away from itself across two CRSes.

        EPSG:27562's geodetic counterpart is NTF (Paris), which measures longitude
        from Paris. Converting the target to its own frame instead of the
        collection's dropped that offset and reported 112.47 km for a zero
        separation.
        """
        fc = _fc([Point(600000.0, 2200000.0)], crs=27562)
        same_point_elsewhere = gpd.GeoSeries(
            [Point(600000.0, 2200000.0)], crs=27562
        ).to_crs(4326)
        assert fc.geodesic_distance(same_point_elsewhere).iloc[0] == pytest.approx(
            0.0, abs=1.0
        )

    def test_cross_crs_target_keeps_the_datum_shift(self):
        """A NAD27 target is not the same ground point as the WGS 84 one.

        Same nominal degrees, different datum, so the separation is tens of
        metres -- not the 0.0 a dropped datum shift reports. The point is in
        Nebraska on purpose: NAD27 is a North American datum, and PROJ has no
        transformation for it over Europe, so a European coordinate shifts by
        exactly nothing and would pass this test with the bug present.
        """
        fc = _fc([Point(-100.0, 40.0)])
        nad27 = gpd.GeoSeries([Point(-100.0, 40.0)], crs=4267)
        separation = fc.geodesic_distance(nad27).iloc[0]
        assert separation == pytest.approx(34.589, rel=1e-3)

    def test_same_crs_target_is_still_zero(self):
        """The control: the same point in the same CRS stays zero."""
        fc = _fc([Point(600000.0, 2200000.0)], crs=27562)
        same = gpd.GeoSeries([Point(600000.0, 2200000.0)], crs=27562)
        assert fc.geodesic_distance(same).iloc[0] == pytest.approx(0.0, abs=1e-6)


class TestMissingGeometries:
    """A missing or empty geometry is named, not misdiagnosed (R2 S3)."""

    def test_missing_geometry_names_the_row(self):
        """`geom_type` is NaN for a missing geometry, so it used to slip through.

        The refusal then came from `_as_degrees` and read "lon1 must be finite
        degrees ... a projected coordinate is a common cause" -- the wrong cause,
        naming an internal argument the caller never supplied.
        """
        frame = gpd.GeoDataFrame(
            geometry=[Point(0, 0), None], crs="EPSG:4326", index=["a", "b"]
        )
        fc = FeatureCollection(frame)
        target = Point(1, 0)
        with pytest.raises(InvalidGeometryError, match="missing or empty geometry"):
            fc.geodesic_distance(target)

    def test_the_message_names_the_offending_index(self):
        """The row label is in the message, so it is actionable."""
        frame = gpd.GeoDataFrame(
            geometry=[Point(0, 0), None], crs="EPSG:4326", index=["a", "b"]
        )
        fc = FeatureCollection(frame)
        target = Point(1, 0)
        with pytest.raises(InvalidGeometryError, match=r"\['b'\]"):
            fc.geodesic_distance(target)

    def test_empty_geometry_is_refused_too(self):
        """An empty point has no coordinates either."""
        fc = _fc([Point(0, 0), Point()])
        target = Point(1, 0)
        with pytest.raises(InvalidGeometryError, match="missing or empty geometry"):
            fc.geodesic_distance(target)


class TestIndexAlignment:
    """Targets align on the index, like the inherited `distance` (R2 S6)."""

    def test_differently_ordered_index_is_aligned_not_zipped(self):
        """A reordered target pairs by label, not by position.

        Positional pairing silently matched Point(0,0) with Point(1,60) and gave
        [6654.64, 6654.64] km for what should be [111.32, 55.80].
        """
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                geometry=[Point(0, 0), Point(0, 60)], crs="EPSG:4326", index=[0, 1]
            )
        )
        targets = gpd.GeoSeries(
            [Point(1, 60), Point(1, 0)], crs="EPSG:4326", index=[1, 0]
        )
        result = fc.geodesic_distance(targets, unit="km")
        assert result.loc[0] == pytest.approx(DEGREE_AT_EQUATOR_KM)
        assert result.loc[1] == pytest.approx(DEGREE_AT_60N_KM)

    def test_mismatched_index_raises(self):
        """An index that does not cover the collection's rows cannot be paired."""
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                geometry=[Point(0, 0), Point(0, 60)], crs="EPSG:4326", index=["a", "b"]
            )
        )
        targets = gpd.GeoSeries(
            [Point(1, 0), Point(1, 60)], crs="EPSG:4326", index=["x", "y"]
        )
        with pytest.raises(ValueError, match="index does not match"):
            fc.geodesic_distance(targets)

    def test_a_plain_list_target_stays_positional(self):
        """A bare list carries no meaningful index, so it is paired in order."""
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                geometry=[Point(0, 0), Point(0, 60)], crs="EPSG:4326", index=["a", "b"]
            )
        )
        result = fc.geodesic_distance([Point(1, 0), Point(1, 60)], unit="km")
        assert result.loc["a"] == pytest.approx(DEGREE_AT_EQUATOR_KM)
        assert result.loc["b"] == pytest.approx(DEGREE_AT_60N_KM)
