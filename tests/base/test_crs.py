"""Tests for CRS-resolution helpers in :mod:`pyramids.base.crs`."""

from __future__ import annotations

import pytest
from pyproj import CRS

from pyramids.base._errors import CRSError
from pyramids.base.crs import epsg_from_user_input

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
