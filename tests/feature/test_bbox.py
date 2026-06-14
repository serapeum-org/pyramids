"""Unit tests for pyramids.feature.bbox geometry helpers."""

from __future__ import annotations

import pytest
from shapely.geometry import MultiPolygon, Polygon

from pyramids.feature.bbox import (
    _crosses_antimeridian,
    _ring_crosses_antimeridian,
    _unwrap_polygon,
    normalise_longitude,
    split_antimeridian,
    split_polygon_antimeridian,
    to_shapely,
    transform,
)

pytestmark = pytest.mark.core


def _crossing_ring() -> Polygon:
    """Build a rectangular polygon straddling the antimeridian near Fiji.

    Returns:
        A Polygon with vertices at lon 175 and -175, lat -22..-12.
    """
    return Polygon([(175, -22), (-175, -22), (-175, -12), (175, -12)])


class TestSplitAntimeridian:
    """Tests for split_antimeridian."""

    def test_no_crossing_returns_single(self):
        """A bbox with west <= east is returned unchanged in a 1-list.

        Test scenario:
            `(-10, -5, 10, 5)` does not cross, so one bbox is returned.
        """
        result = split_antimeridian((-10.0, -5.0, 10.0, 5.0))
        assert result == [(-10.0, -5.0, 10.0, 5.0)], f"Unexpected split: {result}"

    def test_crossing_splits_in_two(self):
        """A bbox with west > east splits at the 180 deg meridian.

        Test scenario:
            `(175, -22, -175, -12)` yields an eastern and western half.
        """
        result = split_antimeridian((175.0, -22.0, -175.0, -12.0))
        assert result == [
            (175.0, -22.0, 180.0, -12.0),
            (-180.0, -22.0, -175.0, -12.0),
        ], f"Unexpected split: {result}"

    def test_equal_west_east_not_split(self):
        """A degenerate bbox with west == east is treated as non-crossing.

        Test scenario:
            `(10, 0, 10, 1)` keeps west <= east, so it is not split.
        """
        result = split_antimeridian((10.0, 0.0, 10.0, 1.0))
        assert len(result) == 1, f"west == east should not split, got {result}"

    def test_each_half_has_west_le_east(self):
        """Both halves of a crossing split satisfy west <= east.

        Test scenario:
            Every returned bbox is well-formed for downstream queries.
        """
        for west, _, east, _ in split_antimeridian((170.0, 0.0, -170.0, 10.0)):
            assert west <= east, f"Half not well-formed: west={west}, east={east}"


class TestNormaliseLongitude:
    """Tests for normalise_longitude."""

    def test_to_signed_from_360(self):
        """A 0..360 bbox is rewritten into signed longitudes.

        Test scenario:
            `350` becomes `-10` under the `-180..180` convention.
        """
        result = normalise_longitude((350.0, -5.0, 10.0, 5.0), "-180..180")
        assert result == (-10.0, -5.0, 10.0, 5.0), f"Unexpected result: {result}"

    def test_to_360_from_signed(self):
        """A signed bbox is rewritten into the 0..360 convention.

        Test scenario:
            `-10` becomes `350` under the `0..360` convention.
        """
        result = normalise_longitude((-10.0, -5.0, 10.0, 5.0), "0..360")
        assert result == (350.0, -5.0, 10.0, 5.0), f"Unexpected result: {result}"

    def test_latitude_untouched(self):
        """Latitude values are passed through unchanged.

        Test scenario:
            South/north stay put while longitudes are rewritten.
        """
        _, south, _, north = normalise_longitude((350.0, -7.5, 5.0, 12.5))
        assert (south, north) == (-7.5, 12.5), "Latitudes must not change"

    def test_invalid_convention_raises(self):
        """An unsupported convention name raises ValueError.

        Test scenario:
            `"0..180"` is not a recognised convention.
        """
        with pytest.raises(ValueError, match="convention must be one of"):
            normalise_longitude((0.0, 0.0, 1.0, 1.0), "0..180")

    def test_default_convention_is_signed(self):
        """The default convention is `-180..180`.

        Test scenario:
            Calling without a convention normalises into signed longitudes.
        """
        result = normalise_longitude((200.0, 0.0, 210.0, 1.0))
        assert result[0] == -160.0, f"Default should be -180..180, got {result}"


class TestTransform:
    """Tests for transform."""

    def test_identity_same_crs(self):
        """Transforming within the same CRS is an identity (modulo noise).

        Test scenario:
            4326 -> 4326 returns the original bbox to one decimal.
        """
        result = [round(v, 1) for v in transform((-10.0, -5.0, 10.0, 5.0), 4326, 4326)]
        assert result == [-10.0, -5.0, 10.0, 5.0], f"Identity failed: {result}"

    def test_to_web_mercator(self):
        """Reproject WGS84 degrees to Web Mercator metres.

        Test scenario:
            Longitude 10 deg maps to ~1,113,195 m easting.
        """
        _, _, east, _ = transform((0.0, 0.0, 10.0, 10.0), 4326, 3857)
        assert round(east) == 1113195, f"Unexpected mercator easting: {east}"

    def test_authority_strings_accepted(self):
        """CRS arguments may be `EPSG:XXXX` strings.

        Test scenario:
            Both CRSes given as authority strings still reproject.
        """
        result = transform((0.0, 0.0, 1.0, 1.0), "EPSG:4326", "EPSG:3857")
        assert result[2] > 0, f"Expected positive easting, got {result}"

    def test_geographic_destination_clamps_latitude(self):
        """Reprojecting into a geographic CRS keeps latitude within [-90, 90].

        Test scenario:
            A Web Mercator extent back to 4326 yields valid latitudes.
        """
        _, south, _, north = transform((-2.0e7, -2.0e7, 2.0e7, 2.0e7), 3857, 4326)
        assert (
            -90.0 <= south <= north <= 90.0
        ), f"Latitude not clamped: {south}..{north}"

    def test_densify_pts_parameter(self):
        """The densify_pts argument is accepted and affects the result.

        Test scenario:
            A coarse densification still returns a 4-tuple bbox.
        """
        result = transform((-100.0, 20.0, -90.0, 50.0), 4326, 3857, densify_pts=2)
        assert len(result) == 4, f"Expected a 4-tuple, got {result}"

    def test_src_crs_accepts_crs_object(self):
        """The source CRS is normalised, so a pyproj.CRS object is accepted.

        Test scenario:
            Passing `CRS.from_epsg(4326)` as `src_crs` reprojects identically
            to passing the EPSG int.
        """
        from pyproj import CRS

        via_obj = transform((0.0, 0.0, 1.0, 1.0), CRS.from_epsg(4326), 3857)
        via_int = transform((0.0, 0.0, 1.0, 1.0), 4326, 3857)
        assert via_obj == pytest.approx(
            via_int
        ), f"CRS-object src mismatch: {via_obj} vs {via_int}"

    def test_src_crs_accepts_wkt(self):
        """A WKT string is accepted for the source CRS.

        Test scenario:
            A 4326 WKT source transformed to 4326 is an identity to one decimal.
        """
        from pyproj import CRS

        wkt = CRS.from_epsg(4326).to_wkt()
        result = [round(v, 1) for v in transform((-10.0, -5.0, 10.0, 5.0), wkt, 4326)]
        assert result == [-10.0, -5.0, 10.0, 5.0], f"WKT src failed: {result}"


class TestToShapely:
    """Tests for to_shapely."""

    def test_bounds_round_trip(self):
        """The polygon's bounds equal the input bbox.

        Test scenario:
            `(-10, -5, 10, 5)` becomes a box with matching bounds.
        """
        assert to_shapely((-10.0, -5.0, 10.0, 5.0)).bounds == (-10.0, -5.0, 10.0, 5.0)

    def test_area(self):
        """The polygon area equals width times height.

        Test scenario:
            A 4x3 bbox has area 12.
        """
        assert (
            to_shapely((0.0, 0.0, 4.0, 3.0)).area == 12.0
        ), "Area should be width*height"


class TestRingCrossesAntimeridian:
    """Tests for _ring_crosses_antimeridian."""

    def test_detects_crossing(self):
        """A longitude step over 180 deg is detected as a crossing.

        Test scenario:
            175 -> -175 is a 350 deg step, so the ring crosses.
        """
        coords = [(175.0, 0.0), (-175.0, 0.0), (-175.0, 1.0)]
        assert _ring_crosses_antimeridian(coords) is True, "Crossing not detected"

    def test_no_crossing(self):
        """Small longitude steps are not flagged as a crossing.

        Test scenario:
            A modest 0..10 box has no step over 180 deg.
        """
        coords = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        assert _ring_crosses_antimeridian(coords) is False, "False crossing detected"


class TestCrossesAntimeridian:
    """Tests for _crosses_antimeridian."""

    def test_exterior_crossing(self):
        """A crossing exterior ring is detected.

        Test scenario:
            The Fiji box crosses on its exterior.
        """
        assert (
            _crosses_antimeridian(_crossing_ring()) is True
        ), "Exterior crossing missed"

    def test_no_crossing(self):
        """A polygon entirely within -180..180 does not cross.

        Test scenario:
            A 0..10 box is not flagged.
        """
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        assert _crosses_antimeridian(poly) is False, "False crossing detected"

    def test_hole_crossing_detected(self):
        """A crossing detected on an interior ring (hole) returns True.

        Test scenario:
            A non-crossing exterior with a crossing hole still reports True.
        """
        exterior = [(-179, -30), (179, -30), (179, 30), (-179, 30)]
        hole = [(178, -5), (-178, -5), (-178, 5), (178, 5)]
        poly = Polygon(exterior, [hole])
        assert _crosses_antimeridian(poly) is True, "Hole crossing not detected"


class TestUnwrapPolygon:
    """Tests for _unwrap_polygon."""

    def test_longitudes_mapped_to_0_360(self):
        """All longitudes are mapped into [0, 360) and made contiguous.

        Test scenario:
            The Fiji box's lons become {175, 185}, spanning <= 180 deg.
        """
        unwrapped = _unwrap_polygon(_crossing_ring())
        lons = [x for x, _ in unwrapped.exterior.coords]
        assert all(0.0 <= x < 360.0 for x in lons), f"Longitudes not in 0..360: {lons}"
        assert max(lons) - min(lons) <= 180.0, "Unwrapped polygon should be contiguous"

    def test_holes_preserved(self):
        """Interior rings survive the unwrap.

        Test scenario:
            A polygon with one hole still has one interior ring afterwards.
        """
        exterior = [(170, -10), (-170, -10), (-170, 10), (170, 10)]
        hole = [(175, -2), (-175, -2), (-175, 2), (175, 2)]
        unwrapped = _unwrap_polygon(Polygon(exterior, [hole]))
        assert len(unwrapped.interiors) == 1, "Hole was dropped during unwrap"


class TestSplitPolygonAntimeridian:
    """Tests for split_polygon_antimeridian."""

    def test_crossing_polygon_splits(self):
        """A crossing polygon becomes a 2-part MultiPolygon.

        Test scenario:
            The Fiji box splits into east and west parts.
        """
        result = split_polygon_antimeridian(_crossing_ring())
        assert (
            result.geom_type == "MultiPolygon"
        ), f"Expected MultiPolygon, got {result.geom_type}"
        assert len(result.geoms) == 2, f"Expected 2 parts, got {len(result.geoms)}"

    def test_parts_on_opposite_sides(self):
        """The two split parts sit on opposite sides of the 180 deg meridian.

        Test scenario:
            Centroids are near +177.5 and -177.5.
        """
        parts = split_polygon_antimeridian(_crossing_ring()).geoms
        sides = sorted(round(p.centroid.x, 1) for p in parts)
        assert sides == [-177.5, 177.5], f"Parts not on opposite sides: {sides}"

    def test_area_preserved(self):
        """Splitting preserves total area (10 deg x 10 deg = 100).

        Test scenario:
            The Fiji box spans 10 deg lon by 10 deg lat.
        """
        result = split_polygon_antimeridian(_crossing_ring())
        assert round(result.area, 6) == 100.0, f"Area not preserved: {result.area}"

    def test_non_crossing_unchanged(self):
        """A non-crossing polygon is returned unchanged.

        Test scenario:
            A 0..10 box keeps its bounds and is not wrapped.
        """
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = split_polygon_antimeridian(poly)
        assert result.bounds == (
            0.0,
            0.0,
            10.0,
            10.0,
        ), f"Unexpected bounds: {result.bounds}"

    def test_polygon_with_hole_crossing(self):
        """A crossing polygon with a hole still produces a valid split.

        Test scenario:
            Exterior and hole both cross; the result remains valid.
        """
        exterior = [(170, -10), (-170, -10), (-170, 10), (170, 10)]
        hole = [(175, -2), (-175, -2), (-175, 2), (175, 2)]
        result = split_polygon_antimeridian(Polygon(exterior, [hole]))
        assert result.is_valid, "Split polygon with hole should be valid"
        assert result.geom_type == "MultiPolygon", "Crossing polygon should split"

    def test_multipolygon_input(self):
        """A MultiPolygon input is handled per-part.

        Test scenario:
            A MultiPolygon containing one crossing part is split.
        """
        multi = MultiPolygon([_crossing_ring()])
        result = split_polygon_antimeridian(multi)
        assert result.is_valid, "Result should be valid"
        assert round(result.area, 6) == 100.0, f"Area not preserved: {result.area}"

    def test_invalid_type_raises(self):
        """A non-polygon geometry raises TypeError.

        Test scenario:
            Passing a string is rejected.
        """
        with pytest.raises(TypeError, match="expects a Polygon or MultiPolygon"):
            split_polygon_antimeridian("not a polygon")
