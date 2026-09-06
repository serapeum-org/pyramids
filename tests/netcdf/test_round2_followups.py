"""Four narrow contracts the second review round found stated but not held.

Each is a small thing said in a docstring, a comment or a commit message that
the code did not do. They are grouped here because they share that shape, not
a subsystem: a bbox identity check that could not fire, an xarray export that
put coordinates among the data variables, a time-unit predicate that treated
undecoded bytes as "not a time axis", and a reopen documented as class-
preserving that never was.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import osr

from pyramids.netcdf import NetCDF
from pyramids.netcdf.netcdf import Container, Variable
from pyramids.netcdf.utils import decode_cf_time, is_cf_time_units

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"
BOUNDED = DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"
CURVILINEAR = DATA / "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc"
RECTILINEAR = DATA / "cf__5v__1d4-4d1__y-asc.nc"


class TestTheBboxIdentityCheckIgnoresAxisOrder:
    """`IsSame` is mapping-sensitive, so the operands are normalised first."""

    @staticmethod
    def _authority_order(epsg: int) -> osr.SpatialReference:
        """An SRS carrying authority axis order, as `GetSpatialRef()` returns.

        Args:
            epsg: The code to build.

        Returns:
            osr.SpatialReference: Left on GDAL's default mapping strategy,
                deliberately un-normalised.
        """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        return srs

    def test_a_geographic_crs_matches_itself_across_axis_orders(self):
        """The regression: EPSG:4326 compared unequal to EPSG:4326.

        Test scenario:
            The source comes out of `sr_from_user_input` stamped traditional,
            while a destination read off a store keeps authority order.
            `IsSame` answers False across that difference for a geographic
            CRS, so the identity shortcut never fired and the caller's exact
            box came back as a densified round-trip.
        """
        bbox = (-78.0, 38.0, -75.0, 40.0)

        out = NetCDF._reproject_bbox_envelope(
            bbox, 4326, self._authority_order(4326), 25
        )

        assert out == bbox, "identity path did not fire, so the box was resampled"

    def test_a_projected_crs_matches_itself_too(self):
        """Not only the axis-swapped case; the shortcut must be general.

        Test scenario:
            A projected CRS has the same axis order either way, so this one
            already worked -- it is here so a future narrowing of the check
            cannot pass by fixing only the geographic case.
        """
        bbox = (500000.0, 4600000.0, 510000.0, 4610000.0)

        out = NetCDF._reproject_bbox_envelope(
            bbox, 32636, self._authority_order(32636), 25
        )

        assert out == bbox

    def test_a_genuinely_different_crs_still_reprojects(self):
        """Widening the identity check must not swallow a real difference.

        Test scenario:
            EPSG:4326 into EPSG:32636 has to leave degrees behind and come
            back as metres; an over-eager shortcut would hand back the
            degrees unchanged.
        """
        bbox = (32.0, 30.0, 33.0, 31.0)

        min_x, min_y, max_x, max_y = NetCDF._reproject_bbox_envelope(
            bbox, 4326, self._authority_order(32636), 25
        )

        assert min_x > 1e5, "min_x came back in degrees, not projected metres"
        assert max_y > 1e6, "max_y came back in degrees, not projected metres"
        assert max_x > min_x, "the x axis came back inverted"
        assert max_y > min_y, "the y axis came back inverted"


class TestToXarrayPutsCfNonDataArraysInCoords:
    """Reading every readable array must not relabel them all as data."""

    def test_bounds_arrays_are_coordinates(self):
        """`lat_bnds` belongs under `lat`, not beside `tos`.

        Test scenario:
            Exporting the bounds as data variables makes a `to_netcdf()`
            round-trip write them back as ordinary variables, losing the CF
            relationship the file declared.
        """
        exported = NetCDF.read_file(str(BOUNDED)).to_xarray()

        assert set(exported.data_vars) == {"tos"}
        assert {"lat_bnds", "lon_bnds", "time_bnds"} <= set(exported.coords)

    def test_curvilinear_auxiliary_coordinates_are_coordinates(self):
        """The 2-D `lat_rho` / `lon_rho` pair are coordinates by CF role.

        Test scenario:
            The ROMS fixture names them in its `coordinates` attribute, which
            is exactly what makes them auxiliary coordinates rather than
            fields.
        """
        exported = NetCDF.read_file(str(CURVILINEAR)).to_xarray()

        assert set(exported.data_vars) == {"salt", "zeta"}
        assert {"lat_rho", "lon_rho"} <= set(exported.coords)

    def test_nothing_readable_is_lost_in_the_move(self):
        """Promotion, not omission -- the export still carries every array.

        Test scenario:
            The reason the loop reads the readable superset is that an aux
            array used to vanish. Moving those arrays into `coords` must not
            reintroduce the loss it fixed.
        """
        nc = NetCDF.read_file(str(BOUNDED))

        exported = nc.to_xarray()

        present = set(exported.data_vars) | set(exported.coords)
        assert set(nc._readable_variable_names()) <= present

    def test_a_store_with_no_cf_classification_is_left_alone(self):
        """A non-CF file must not be second-guessed.

        Test scenario:
            The promotion reads the CF classification. Where there is none it
            has no opinion, so every readable array stays a data variable and
            the export is exactly what it was.
        """
        nc = NetCDF.read_file(str(RECTILINEAR))

        exported = nc.to_xarray()

        assert set(nc.variable_names) <= set(exported.data_vars)


class TestBytesUnitsAreDecodedNotIgnored:
    """A `units` that arrived undecoded still names the axis it names."""

    def test_the_predicate_decodes_bytes(self):
        """Making the predicate total must not make it wrong.

        Test scenario:
            `bytes` are neither a `str` nor a malformed unit -- they are the
            same unit, undecoded. Answering False put them back in the hole
            the predicate was written to fill.
        """
        assert is_cf_time_units(b"days since 1970-01-01") is True
        assert is_cf_time_units(bytearray(b"hours since 2000-01-01")) is True

    def test_bytes_that_are_not_a_time_unit_are_still_refused(self):
        """Decoding is not the same as accepting.

        Test scenario:
            A decoded spatial unit has to fail the shape test like its string
            spelling does, or the predicate would answer True for anything
            that happened to arrive as bytes.
        """
        assert is_cf_time_units(b"degrees_north") is False

    def test_undecodable_bytes_are_refused_rather_than_raised(self):
        """A malformed file must still not crash the reader.

        Test scenario:
            The predicate is total by design -- the CF compliance checker
            reports on bad files rather than dying on them -- so bytes that
            are not valid UTF-8 answer False.
        """
        assert is_cf_time_units(b"\xff\xfe not utf-8") is False

    def test_a_bytes_time_axis_actually_decodes(self):
        """The consumer, not just the predicate.

        Test scenario:
            `decode_cf_time` gates on the predicate, so a bytes `units` used
            to hand back raw numbers with nothing raised. It must now produce
            dates.
        """
        decoded = decode_cf_time(np.array([0.0, 1.0]), b"days since 1970-01-01")

        assert list(decoded) != [0.0, 1.0]
        assert "1970-01-01" in str(decoded[0])


class TestPersistToReopensAsAContainer:
    """The reopen goes through `type(self)`; it is not class-preserving."""

    def test_a_variable_persists_to_a_container(self, tmp_path):
        """What the docstring used to promise, and what actually happens.

        Test scenario:
            `NetCDF.read_file` ends in `return Container(...)` and ignores
            `cls`, so routing the lookup through `type(self)` does not make a
            `Variable` come back a `Variable`. The docstring said it did.
        """
        variable = NetCDF.read_file(str(RECTILINEAR)).get_variable("temperature")
        assert isinstance(variable, Variable)

        reopened = variable._persist_to(tmp_path / "out.nc")

        assert isinstance(reopened, Container)
        assert not isinstance(reopened, Variable)

    def test_no_path_hands_back_the_receiver_untouched(self):
        """`path=None` is the one case that does preserve the class.

        Test scenario:
            It returns `self`, so a `Variable` stays a `Variable` -- which is
            why the guarantee looked true from the call sites that pass
            `None`.
        """
        variable = NetCDF.read_file(str(RECTILINEAR)).get_variable("temperature")

        assert variable._persist_to(None) is variable
