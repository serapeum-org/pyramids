"""Tests for the geodesy helpers in :mod:`pyramids.base.geodesy`."""

from __future__ import annotations

import numpy as np
import pytest

import pyramids.base.geodesy as geodesy_module
from pyramids.base._errors import CRSError
from pyramids.base.geodesy import (
    _length_scale,
    geodesic_distance,
    ground_distance_in_crs,
)

# One degree of longitude on the WGS 84 ellipsoid, at the equator and at 60 N.
# The second is the number that makes this module necessary: the same degree is
# half the ground distance, so no single metres-per-degree constant can exist.
DEGREE_AT_EQUATOR_M = 111319.49079327357
DEGREE_AT_60N_M = 55799.47039326038


class TestLengthScale:
    """The unit table behind both public functions."""

    @pytest.mark.parametrize(
        ("unit", "expected"),
        [("m", 1.0), ("km", 1000.0), ("mi", 1609.344), ("nmi", 1852.0)],
    )
    def test_known_units(self, unit: str, expected: float):
        """Each supported unit reports its length in metres."""
        assert _length_scale(unit) == expected

    @pytest.mark.parametrize("unit", ["KM", " km ", "  M", "NMI"])
    def test_case_and_whitespace_insensitive(self, unit: str):
        """Case and surrounding whitespace do not change the unit."""
        assert _length_scale(unit) == _length_scale(unit.strip().lower())

    @pytest.mark.parametrize("unit", ["m2", "feet", "", "metres"])
    def test_unknown_unit_raises(self, unit: str):
        """An unrecognised unit is refused, and the message lists the valid ones."""
        with pytest.raises(ValueError, match="unknown length unit"):
            _length_scale(unit)

    @pytest.mark.parametrize("unit", [None, 2, ["km"]])
    def test_non_string_unit_raises_value_error(self, unit):
        """A non-string is refused as a `ValueError`, not an `AttributeError`.

        `strip()` is attempted before the lookup, so a non-string fails there;
        the helper converts that into the `ValueError` it documents.
        """
        with pytest.raises(ValueError, match="unknown length unit"):
            _length_scale(unit)


class TestGeodesicDistance:
    """The inverse geodesic problem on a CRS's own ellipsoid."""

    def test_degree_at_equator(self):
        """A degree of longitude at the equator is ~111.3 km."""
        assert geodesic_distance(0.0, 0.0, 1.0, 0.0) == pytest.approx(
            DEGREE_AT_EQUATOR_M
        )

    def test_degree_shrinks_towards_the_pole(self):
        """The same degree at 60 N is about half as long."""
        at_60 = geodesic_distance(0.0, 60.0, 1.0, 60.0)
        assert at_60 == pytest.approx(DEGREE_AT_60N_M)
        assert at_60 < DEGREE_AT_EQUATOR_M / 1.9

    def test_identical_points_are_zero(self):
        """A point is no distance from itself."""
        assert geodesic_distance(12.5, 41.9, 12.5, 41.9) == pytest.approx(0.0)

    def test_symmetric(self):
        """Distance does not depend on which point is named first."""
        forward = geodesic_distance(-74.0, 40.7, 2.35, 48.86)
        backward = geodesic_distance(2.35, 48.86, -74.0, 40.7)
        assert forward == pytest.approx(backward)

    def test_scalar_input_returns_float(self):
        """Scalars in, scalar out -- not a 0-d array."""
        result = geodesic_distance(0.0, 0.0, 1.0, 0.0)
        assert isinstance(result, float)

    def test_array_input_returns_array(self):
        """Arrays in, array out, elementwise over the inputs."""
        result = geodesic_distance([0.0, 0.0], [0.0, 60.0], [1.0, 1.0], [0.0, 60.0])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(DEGREE_AT_EQUATOR_M)
        assert result[1] == pytest.approx(DEGREE_AT_60N_M)

    @pytest.mark.parametrize(
        ("unit", "divisor"),
        [("m", 1.0), ("km", 1000.0), ("mi", 1609.344), ("nmi", 1852.0)],
    )
    def test_unit_conversion(self, unit: str, divisor: float):
        """Every unit is the metre answer divided by its length in metres."""
        converted = geodesic_distance(0.0, 0.0, 1.0, 0.0, unit=unit)
        assert converted == pytest.approx(DEGREE_AT_EQUATOR_M / divisor)

    def test_unknown_unit_raises(self):
        """An unrecognised unit is refused before any geodesy happens."""
        with pytest.raises(ValueError, match="unknown length unit"):
            geodesic_distance(0.0, 0.0, 1.0, 0.0, unit="furlong")

    def test_projected_crs_contributes_its_datum_ellipsoid(self):
        """A projected CRS is accepted and measures on its own datum.

        UTM 36N is WGS 84, so it must agree exactly with EPSG:4326 -- the inputs
        are geographic degrees either way, and only the ellipsoid is taken from
        the CRS.
        """
        projected = geodesic_distance(0.0, 0.0, 1.0, 0.0, crs=32636)
        assert projected == pytest.approx(DEGREE_AT_EQUATOR_M)

    def test_ellipsoid_comes_from_the_crs_not_a_wgs84_default(self):
        """A CRS on a sphere gives a different answer, proving the datum is read.

        EPSG:4047 is the GRS 1980 authalic *sphere* (`f=0`), so a degree there is
        ~124 m shorter than on the WGS 84 ellipsoid. If the implementation
        defaulted to WGS 84 this would be indistinguishable.
        """
        on_sphere = geodesic_distance(0.0, 0.0, 1.0, 0.0, crs=4047)
        assert on_sphere == pytest.approx(111195.04881760638)
        assert on_sphere != pytest.approx(DEGREE_AT_EQUATOR_M)

    def test_unresolvable_crs_raises_crs_error(self):
        """A CRS that cannot be parsed surfaces as pyramids' `CRSError`."""
        with pytest.raises(CRSError):
            geodesic_distance(0.0, 0.0, 1.0, 0.0, crs="not a crs at all")

    def test_crs_without_ellipsoid_raises_crs_error(self, monkeypatch):
        """A datum naming no ellipsoid is refused rather than assumed.

        No EPSG code in the database reaches this branch, so the resolved CRS is
        stubbed: the guard exists for engineering/local CRSes whose datum carries
        no figure of the earth.
        """

        class _NoEllipsoid:
            name = "Stubbed CRS"

            @staticmethod
            def get_geod():
                return None

        monkeypatch.setattr(
            geodesy_module, "crs_from_user_input", lambda _crs: _NoEllipsoid()
        )
        with pytest.raises(CRSError, match="declares no ellipsoid"):
            geodesic_distance(0.0, 0.0, 1.0, 0.0, crs=4326)


class TestGroundDistanceInCRS:
    """A ground distance expressed in a CRS's own units, at a place."""

    def test_degrees_at_the_equator(self):
        """100 km at the equator is ~0.898 degrees of longitude."""
        span = ground_distance_in_crs(100_000.0, crs=4326, at=(0.0, 0.0))
        assert span == pytest.approx(0.8983152841195214)

    def test_degrees_grow_towards_the_pole(self):
        """The same 100 km needs about twice the degrees at 60 N.

        This is the behaviour the `at` argument exists for: a single
        degrees-per-metre factor would be wrong everywhere but one latitude.
        """
        at_equator = ground_distance_in_crs(100_000.0, crs=4326, at=(0.0, 0.0))
        at_60 = ground_distance_in_crs(100_000.0, crs=4326, at=(0.0, 60.0))
        assert at_60 == pytest.approx(1.7917177623766591)
        assert at_60 / at_equator == pytest.approx(2.0, rel=0.01)

    def test_web_mercator_metres_are_stretched_by_latitude(self):
        """Web Mercator's unit is a metre only at the equator.

        At 60 N its scale factor is `1 / cos(60) = 2`, so 100 km of ground spans
        about 200 000 Web-Mercator metres -- the trap this function exists to
        avoid for anyone sizing a scale bar on a web map.
        """
        y_at_60n = 8399737.89
        span = ground_distance_in_crs(100_000.0, crs=3857, at=(0.0, y_at_60n))
        assert span == pytest.approx(199466.86870209937)
        assert span / 100_000.0 == pytest.approx(2.0, rel=0.01)

    def test_projected_crs_is_near_but_not_exactly_true_scale(self):
        """UTM is nearly true-scale, so 100 km of ground is ~100 000 units.

        Not *exactly*: the zone's scale factor distorts by a few parts in ten
        thousand, which is precisely why this is measured rather than assumed.
        """
        span = ground_distance_in_crs(100_000.0, crs=32636, at=(500_000.0, 3_300_000.0))
        assert span == pytest.approx(100_000.0, rel=1e-3)
        assert span != pytest.approx(100_000.0, rel=1e-6)

    def test_round_trips_against_geodesic_distance(self):
        """Walking the returned span back out measures the original distance.

        Approximate rather than exact: a geodesic heading east drifts slightly in
        latitude, and this check holds the latitude fixed on the way back.
        """
        span = ground_distance_in_crs(100_000.0, crs=4326, at=(10.0, 45.0))
        measured = geodesic_distance(10.0, 45.0, 10.0 + span, 45.0)
        assert measured == pytest.approx(100_000.0, rel=1e-3)

    def test_azimuth_changes_the_answer(self):
        """North and east are different questions away from the equator.

        A degree of latitude is nearly constant, so measuring north at 60 N needs
        about half the degrees that measuring east does.
        """
        east = ground_distance_in_crs(100_000.0, crs=4326, at=(0.0, 60.0), azimuth=90.0)
        north = ground_distance_in_crs(100_000.0, crs=4326, at=(0.0, 60.0), azimuth=0.0)
        assert north == pytest.approx(0.8975059986338607)
        assert north < east

    @pytest.mark.parametrize("distance", [0.0, -1.0, -100_000.0])
    def test_non_positive_distance_raises(self, distance: float):
        """A zero or negative ground distance is refused."""
        with pytest.raises(ValueError, match="finite and positive"):
            ground_distance_in_crs(distance, crs=4326, at=(0.0, 0.0))

    @pytest.mark.parametrize("distance", [float("nan"), float("inf")])
    def test_non_finite_distance_raises(self, distance: float):
        """A non-finite ground distance is refused."""
        with pytest.raises(ValueError, match="finite and positive"):
            ground_distance_in_crs(distance, crs=4326, at=(0.0, 0.0))

    @pytest.mark.parametrize("at", [(0.0,), (0.0, 0.0, 0.0), ()])
    def test_at_must_be_a_pair(self, at: tuple):
        """`at` is an `(x, y)` pair; anything else is refused."""
        with pytest.raises(ValueError, match=r"at must be an \(x, y\) pair"):
            ground_distance_in_crs(100_000.0, crs=4326, at=at)

    def test_unresolvable_crs_raises_crs_error(self):
        """A CRS that cannot be parsed surfaces as pyramids' `CRSError`."""
        with pytest.raises(CRSError):
            ground_distance_in_crs(100_000.0, crs="not a crs at all", at=(0.0, 0.0))

    def test_point_outside_the_crs_domain_raises_value_error(self):
        """An un-invertible endpoint is refused instead of returning `inf`.

        An orthographic projection only shows one hemisphere, so a point near the
        limb whose geodesic walks over the edge has no image: PROJ hands back a
        non-finite coordinate rather than raising, and the guard turns that into a
        `ValueError` naming the point. Web Mercator is deliberately *not* used
        here -- PROJ extrapolates it past its nominal latitude limit rather than
        failing, so it never reaches this branch.
        """
        ortho = "+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        with pytest.raises(ValueError, match="outside the usable domain"):
            ground_distance_in_crs(500_000.0, crs=ortho, at=(6_370_000.0, 0.0))

    def test_near_the_limb_still_measures(self):
        """Just inside the same projection's limb the answer is still defined.

        Guards the test above against passing for the wrong reason: the refusal
        must come from leaving the hemisphere, not from orthographic input being
        rejected outright.
        """
        ortho = "+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        span = ground_distance_in_crs(500_000.0, crs=ortho, at=(6_000_000.0, 0.0))
        assert span == pytest.approx(151000.48218178842)
