"""`transform` is the package's one bbox reprojection.

It moved from `feature.bbox` down to `base` so the coverage readers could use
it without importing `feature` (and geopandas). These tests cover the function
itself; `test_base_does_not_import_feature.py` covers the move.

Two behaviours are easy to lose in a relocation and are pinned here: the edges
are *densified* before reprojection, so a curved CRS boundary is not crudely
axis-aligned, and latitudes are clamped when the destination is geographic, so
float overshoot at the poles cannot produce an out-of-range coordinate.
"""

from __future__ import annotations

import pytest

from pyramids.base._bbox import transform

pytestmark = pytest.mark.core


class TestTransform:
    """Reprojection, densification, and the geographic clamp."""

    def test_a_same_crs_transform_is_the_identity(self):
        """Nothing moves when the two CRSes are the same.

        Test scenario:
            The pipeline still runs -- pyproj builds a transformer either way --
            so the box has to come back where it started, modulo float noise.
        """
        result = transform((-10.0, -5.0, 10.0, 5.0), 4326, 4326)

        assert [round(v, 6) for v in result] == [-10.0, -5.0, 10.0, 5.0]

    def test_it_reprojects_degrees_to_metres(self):
        """The nominal case, against a known Web Mercator easting.

        Test scenario:
            10 degrees of longitude at the equator is 1,113,195 m in EPSG:3857.
        """
        result = transform((0.0, 0.0, 10.0, 10.0), 4326, 3857)

        assert round(result[2]) == 1113195

    @pytest.mark.parametrize(
        "src_crs",
        [4326, "EPSG:4326", "epsg:4326"],
        ids=["int", "authority", "lowercase-authority"],
    )
    def test_either_crs_may_be_given_in_any_accepted_form(self, src_crs):
        """Whatever `pyproj.CRS.from_user_input` takes, this takes.

        Args:
            src_crs: The source CRS in one of its accepted spellings.

        Test scenario:
            Callers pass EPSG ints, authority strings and WKT
            interchangeably; all must resolve to the same transform.
        """
        result = transform((0.0, 0.0, 10.0, 10.0), src_crs, 3857)

        assert round(result[2]) == 1113195

    def test_the_edges_are_densified_before_reprojecting(self):
        """A curved boundary is not approximated by its corners alone.

        Test scenario:
            Reprojecting WGS84 to EPSG:3035 bends the parallels. With one
            densification point the southern edge is placed by the corners; with
            the default 21 it follows the curve, and the two answers differ by
            tens of kilometres. Losing the densification in the move would have
            been invisible without this.
        """
        coarse = transform((-20.0, 30.0, 20.0, 60.0), 4326, 3035, densify_pts=1)
        fine = transform((-20.0, 30.0, 20.0, 60.0), 4326, 3035, densify_pts=21)

        assert coarse[1] != fine[1]
        assert abs(coarse[1] - fine[1]) > 10_000

    def test_latitudes_are_clamped_for_a_geographic_destination(self):
        """Float overshoot at the pole must not escape as latitude > 90.

        Test scenario:
            A polar-stereographic box covering the North Pole reprojects to
            lon/lat with a northern edge that overshoots 90 by float noise.
            The result has to be exactly 90, because a consumer treating it as
            a latitude would otherwise be handed an impossible coordinate.
        """
        result = transform(
            (-3_000_000.0, -3_000_000.0, 3_000_000.0, 3_000_000.0), 3413, 4326
        )

        assert result[3] == 90.0
        assert -90.0 <= result[1] <= 90.0

    def test_a_projected_destination_is_not_clamped(self):
        """The clamp is latitude-specific and must not touch metres.

        Test scenario:
            Clamping a projected northing to 90 would collapse the box. The
            guard is conditioned on the destination being geographic, which
            this asserts by exceeding 90 comfortably.
        """
        result = transform((0.0, 0.0, 10.0, 10.0), 4326, 3857)

        assert result[3] > 90.0

    def test_it_round_trips_through_a_projected_crs(self):
        """There and back returns the original box within a metre-scale error.

        Test scenario:
            Densified reprojection is not exactly invertible, but a round trip
            through Web Mercator has to land back within a small tolerance --
            enough to catch an axis swap or a dropped term.
        """
        original = (-10.0, 35.0, 10.0, 55.0)

        there = transform(original, 4326, 3857)
        back = transform(there, 3857, 4326)

        assert back == pytest.approx(original, abs=1e-6)
