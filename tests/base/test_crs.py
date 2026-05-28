"""Tests for CRS-resolution helpers in :mod:`pyramids.base.crs`."""

from __future__ import annotations

import pytest
from osgeo import osr
from pyproj import CRS

from pyramids.base._errors import CRSError
from pyramids.base.crs import epsg_from_user_input, sr_from_user_input

pytestmark = pytest.mark.core


class TestEpsgFromUserInput:
    """Tests for :func:`pyramids.base.crs.epsg_from_user_input`."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (4326, 4326),
            (3857, 3857),
            ("EPSG:3857", 3857),
            ("3857", 3857),
            ("epsg:32636", 32636),
        ],
    )
    def test_resolves_int_and_string_forms(self, value, expected):
        """epsg_from_user_input maps int and string CRS forms to the EPSG code.

        Args:
            value: A CRS given as an int or string.
            expected: The EPSG code it should resolve to.

        Test scenario:
            Integers pass through; authority and bare-numeric strings resolve to codes.
        """
        assert epsg_from_user_input(value) == expected, f"{value!r} -> {expected}"

    def test_resolves_pyproj_crs(self):
        """epsg_from_user_input resolves a pyproj.CRS to its EPSG code.

        Test scenario:
            A CRS.from_epsg(32636) object resolves back to 32636.
        """
        assert (
            epsg_from_user_input(CRS.from_epsg(32636)) == 32636
        ), "CRS object mismatch"

    def test_resolves_wkt_string(self):
        """epsg_from_user_input resolves a WKT string to its EPSG code.

        Test scenario:
            The WKT exported from EPSG:3857 resolves back to 3857.
        """
        wkt = CRS.from_epsg(3857).to_wkt()
        assert epsg_from_user_input(wkt) == 3857, "WKT did not resolve to 3857"

    def test_bool_rejected(self):
        """epsg_from_user_input rejects a bool (an int subclass) as a CRS.

        Test scenario:
            True is not a meaningful CRS and raises CRSError.
        """
        with pytest.raises(CRSError, match="not a valid CRS"):
            epsg_from_user_input(True)

    def test_uninterpretable_input_raises(self):
        """epsg_from_user_input rejects a string that is not a CRS.

        Test scenario:
            A nonsense string raises CRSError mentioning it could not be interpreted.
        """
        with pytest.raises(CRSError, match="could not interpret"):
            epsg_from_user_input("not-a-crs")

    def test_crs_without_epsg_raises(self):
        """epsg_from_user_input rejects a valid CRS that has no EPSG code.

        Test scenario:
            A custom PROJ4 Lambert azimuthal equal-area definition parses but maps to no
            EPSG code, so a CRSError mentioning the missing code is raised.
        """
        proj4 = "+proj=laea +lat_0=52 +lon_0=10 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
        with pytest.raises(CRSError, match="no corresponding EPSG code"):
            epsg_from_user_input(proj4)

    def test_crserror_is_valueerror(self):
        """CRSError raised by the resolver is also a ValueError.

        Test scenario:
            Callers catching ValueError (the historical contract) still catch the new
            CRSError.
        """
        with pytest.raises(ValueError):
            epsg_from_user_input("not-a-crs")


class TestSrFromUserInput:
    """Tests for :func:`pyramids.base.crs.sr_from_user_input`."""

    def test_epsg_int_resolves_to_authority(self):
        """sr_from_user_input(int) returns an SRS that exposes the EPSG authority.

        Test scenario:
            An EPSG int round-trips so the SRS's root AUTHORITY code is the same int.
        """
        sr = sr_from_user_input(3857)
        assert sr.GetAuthorityName(None) == "EPSG", "authority should be EPSG"
        assert sr.GetAuthorityCode(None) == "3857", "code should round-trip"

    def test_esri_authority_string_resolves(self):
        """sr_from_user_input resolves the ESRI authority for non-EPSG world projections.

        Test scenario:
            ESRI:54030 (Robinson) and ESRI:54009 (Mollweide) parse and return projected
            SRSes whose name identifies the projection.
        """
        sr_robinson = sr_from_user_input("ESRI:54030")
        sr_mollweide = sr_from_user_input("ESRI:54009")
        assert sr_robinson.IsProjected() == 1, "Robinson must be projected"
        assert sr_mollweide.IsProjected() == 1, "Mollweide must be projected"
        assert "Robinson" in sr_robinson.GetName(), "name should contain 'Robinson'"
        assert "Mollweide" in sr_mollweide.GetName(), "name should contain 'Mollweide'"

    def test_proj4_orthographic_resolves(self):
        """sr_from_user_input resolves a proj4 string with no EPSG / ESRI authority.

        Test scenario:
            An orthographic proj4 with custom centre lat/lon parses into a projected SRS.
            The projection has no authority entry, so GetAuthorityCode is None — exactly
            the case the EPSG-only path used to reject.
        """
        proj4 = "+proj=ortho +lat_0=39 +lon_0=-9 +datum=WGS84 +units=m +no_defs"
        sr = sr_from_user_input(proj4)
        assert sr.IsProjected() == 1, "ortho must be projected"
        assert sr.GetAuthorityCode(None) is None, (
            "ortho carries no authority code by design"
        )

    def test_pyproj_crs_round_trips(self):
        """sr_from_user_input accepts a pyproj.CRS instance.

        Test scenario:
            A CRS.from_epsg(32636) instance produces an SRS whose AUTHORITY code matches.
        """
        sr = sr_from_user_input(CRS.from_epsg(32636))
        assert sr.GetAuthorityCode(None) == "32636", "code should round-trip"

    def test_traditional_axis_order_set(self):
        """sr_from_user_input uses the traditional GIS axis order.

        Test scenario:
            Lon/lat-first ordering matches the rest of the pyramids stack (geotransform,
            reproject_coordinates), so transforms compose without axis surprises.
        """
        sr = sr_from_user_input(4326)
        assert sr.GetAxisMappingStrategy() == osr.OAMS_TRADITIONAL_GIS_ORDER, (
            "axis order should be traditional GIS (x=lon, y=lat)"
        )

    def test_bool_rejected(self):
        """sr_from_user_input rejects a bool (an int subclass).

        Test scenario:
            True is not a meaningful CRS and raises CRSError, matching the
            epsg_from_user_input guard so users get a consistent error.
        """
        with pytest.raises(CRSError, match="not a valid CRS"):
            sr_from_user_input(True)

    def test_uninterpretable_input_raises(self):
        """sr_from_user_input rejects a string that is not a CRS.

        Test scenario:
            A nonsense string raises CRSError mentioning it could not be interpreted.
        """
        with pytest.raises(CRSError, match="could not interpret"):
            sr_from_user_input("not-a-crs")
