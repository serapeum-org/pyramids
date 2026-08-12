"""Tests for CRS-resolution helpers in :mod:`pyramids.base.crs`."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from osgeo import gdal, osr
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError as PyprojCRSError

import pyramids.base.crs as crs_module
from pyramids.base._errors import CRSError
from pyramids.base._raster_meta import RasterMeta
from pyramids.base.crs import (
    _integer_code,
    _pyproj_can_resolve_epsg,
    clear_crs_caches,
    crs_from_user_input,
    crs_spec,
    epsg_from_user_input,
    epsg_from_wkt,
    get_epsg_from_prj,
    reproject_coordinates,
    sr_from_epsg,
    sr_from_user_input,
)
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestGetEpsgFromPrjNonNumericAuthority:
    """A non-numeric root authority code (OGC:CRS84) must not crash the resolver."""

    @staticmethod
    def _crs84_wkt() -> str:
        srs = osr.SpatialReference()
        srs.SetFromUserInput("OGC:CRS84")  # lon/lat WGS 84; code is "CRS84"
        return srs.ExportToWkt()

    def test_crs84_raises_crserror_not_int_valueerror(self):
        """CRS84 (code 'CRS84') raises the documented CRSError, never int('CRS84').

        CRS84 has no EPSG code of its own, so the strict resolver treats it as an
        unresolvable custom CRS — but via the clear CRSError path, not a raw
        ``invalid literal for int()`` crash from the non-numeric authority code.
        """
        wkt = self._crs84_wkt()
        with pytest.raises(CRSError, match="could not resolve an EPSG"):
            get_epsg_from_prj(wkt)

    def test_epsg_from_wkt_absorbs_crs84_to_4326(self):
        """The soft ``epsg_from_wkt`` path yields 4326 for a CRS84 raster.

        This is the path ``Dataset.epsg`` uses, so a WMS/WMTS layer reported as
        OGC:CRS84 reads back as EPSG:4326 instead of crashing.
        """
        assert epsg_from_wkt(self._crs84_wkt()) == 4326


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
        assert epsg_from_user_input(CRS.from_epsg(32636)) == 32636, (
            "CRS object mismatch"
        )

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

    def test_wkt_string_round_trips_to_epsg(self):
        """sr_from_user_input accepts a raw WKT string and preserves the authority.

        Test scenario:
            Round-trip via `CRS.from_epsg(3857).to_wkt()` → `sr_from_user_input(wkt)`
            must produce an SRS whose root AUTHORITY resolves back to EPSG:3857. WKT
            is one of the documented input forms but was not exercised explicitly —
            this locks down that pyproj's WKT parsing path round-trips cleanly.
        """
        wkt = CRS.from_epsg(3857).to_wkt()
        sr = sr_from_user_input(wkt)
        assert sr.GetAuthorityCode(None) == "3857", (
            f"WKT round-trip should preserve EPSG code, got {sr.GetAuthorityCode(None)!r}"
        )

    @pytest.mark.parametrize(
        "value",
        [None, "", []],
        ids=["none", "empty-string", "list"],
    )
    def test_uninterpretable_values_rejected(self, value):
        """sr_from_user_input rejects values that pyproj cannot interpret as a CRS.

        Args:
            value: Input that cannot be turned into a CRS (None / "" / list).

        Test scenario:
            Each of these inputs makes `pyproj.CRS.from_user_input` raise; the wrapper
            converts the raise into a pyramids `CRSError`. Tested as a parametrized
            sweep so future additions (`{}`, custom objects, etc.) extend the matrix
            in one place.
        """
        with pytest.raises(CRSError, match="could not interpret"):
            sr_from_user_input(value)

    def test_lambert_azimuthal_equal_area_proj4_resolves(self):
        """sr_from_user_input accepts a Lambert azimuthal equal-area proj4 (no EPSG code).

        Test scenario:
            LAEA over central Europe is one of the proj4 definitions that
            `epsg_from_user_input` explicitly rejects (no EPSG code). The SRS-based
            helper must accept it and produce a projected reference — this is the
            class of input #418 was filed for, beyond just orthographic / Robinson.
        """
        proj4 = "+proj=laea +lat_0=52 +lon_0=10 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
        sr = sr_from_user_input(proj4)
        assert sr.IsProjected() == 1, "LAEA must be projected"
        assert sr.GetAuthorityCode(None) is None, (
            "LAEA proj4 carries no authority code by design"
        )

    def test_returns_distinct_instances_for_repeated_calls(self):
        """sr_from_user_input returns a fresh SRS each call, never a shared singleton.

        Test scenario:
            Two calls with the same input must return distinct `osr.SpatialReference`
            instances so mutating one (e.g. via `SetAxisMappingStrategy`) can't leak
            into another caller's reference. Identity check (`is not`) plus equality
            of the WKT bytes verifies "different objects, same content".
        """
        sr1 = sr_from_user_input(4326)
        sr2 = sr_from_user_input(4326)
        assert sr1 is not sr2, "repeated calls must return distinct SRS instances"
        assert sr1.ExportToWkt() == sr2.ExportToWkt(), (
            "distinct instances should still encode the same CRS"
        )


class TestReprojectCoordinatesRoundingParity:
    """`reproject_coordinates` must round exactly as the per-point loop did.

    The vectorization (ARC-55) replaced a per-point `transformer.transform`
    loop with one array call. That is value-preserving only if the rounding
    stays on the built-in `round`: `np.round` scales by `10**precision`,
    rounds and divides back, so it disagrees with correctly-rounded decimal
    on values that are not exactly representable.
    """

    @staticmethod
    def _per_point_reference(xs, ys, from_crs, to_crs, precision):
        """Re-implement the pre-vectorization body as the parity oracle."""
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        out_x, out_y = [], []
        for x_value, y_value in zip(xs, ys):
            new_x, new_y = transformer.transform(x_value, y_value)
            if precision is not None:
                new_x = round(new_x, precision)
                new_y = round(new_y, precision)
            out_x.append(new_x)
            out_y.append(new_y)
        return out_x, out_y

    def test_matches_the_per_point_loop_on_a_random_sweep(self):
        """A 2000-point sweep at the default precision reproduces the old output.

        Test scenario:
            Draw lon/lat across the full valid domain, reproject to Web
            Mercator at `precision=6`, and compare element-by-element against
            the per-point reference. Exact equality is required — a single
            last-digit drift here is a silent change to every reprojected
            geometry vertex.
        """
        rng = np.random.default_rng(20260725)
        xs = rng.uniform(-179.0, 179.0, 2000).tolist()
        ys = rng.uniform(-85.0, 85.0, 2000).tolist()
        got_x, got_y = reproject_coordinates(xs, ys, from_crs=4326, to_crs=3857)
        want_x, want_y = self._per_point_reference(xs, ys, 4326, 3857, 6)
        assert got_x == want_x, "x output must match the per-point loop exactly"
        assert got_y == want_y, "y output must match the per-point loop exactly"

    # Values on which `round` and `np.round` disagree at the given precision, so
    # each case genuinely discriminates between the two rounding rules instead of
    # asserting parity with itself. Fed through an identity transform, which
    # leaves them untouched for the rounding step. No case exists for
    # `precision=0`: on integral rounding the two rules agree.
    @pytest.mark.parametrize(
        ("precision", "values"),
        [
            (1, [0.45, 1.05, 1.6500000000000001]),
            (2, [0.015, 0.025, 0.065, 2.675]),
            (3, [0.0025, 0.0055, 0.0075]),
            (6, [3.5e-06, 4.5e-06, 1.25e-05]),
        ],
    )
    def test_matches_the_per_point_loop_across_precisions(self, precision, values):
        """Every precision keeps per-point parity on known divergent values."""
        non_divergent = [
            value
            for value in values
            if round(value, precision) == float(np.round(value, precision))
        ]
        assert non_divergent == [], (
            f"precondition: every case must separate the two rounding rules; "
            f"these do not: {non_divergent}"
        )
        got_x, got_y = reproject_coordinates(
            values, values, from_crs=4326, to_crs=4326, precision=precision
        )
        want_x, want_y = self._per_point_reference(
            values, values, 4326, 4326, precision
        )
        assert (got_x, got_y) == (want_x, want_y), (
            f"precision={precision} must match the per-point loop"
        )

    def test_uses_builtin_round_not_numpy_round(self):
        """The half-way case that separates the two rounding rules.

        Test scenario:
            `round(x, 2)` and `np.round(x, 2)` disagree on 2.675 (2.67 vs
            2.68). Feeding a null transform a coordinate that lands on such a
            value pins the built-in behaviour, so a future refactor cannot
            swap in `np.round` unnoticed.
        """
        assert round(2.675, 2) != float(np.round(2.675, 2)), (
            "precondition: the two rounding rules must actually differ here"
        )
        got_x, _ = reproject_coordinates(
            [2.675], [8.475], from_crs=4326, to_crs=4326, precision=2
        )
        assert got_x[0] == round(2.675, 2), (
            f"expected built-in rounding {round(2.675, 2)}, got {got_x[0]}"
        )


# Codes observed to live in GDAL's vendored PROJ database but not in the one pyproj
# bundles. The list is a *search space*, not an assertion: whichever entry still shows
# the skew on the installed pair is the one the tests use, and if pyproj catches up on
# all of them the suite skips rather than fails. EPSG:10857 (SIRGAS 2000 / Brazil
# Albers, the Brazil Data Cube grid) is the code that produced issue #943.
_SKEW_CANDIDATE_CODES = (10857, 10634, 10688, 10723, 11043)

# A *deprecated* code both databases carry, but read differently: GDAL's
# ImportFromEPSG silently substitutes the non-deprecated replacement (EPSG:4087),
# pyproj returns the code as asked. Pins the ordering choice in `crs_from_user_input`.
_DEPRECATED_CODE = 32663
_DEPRECATED_REPLACEMENT = 4087


@pytest.fixture(scope="module")
def skew_code() -> int:
    """An EPSG code GDAL's PROJ database resolves but pyproj's cannot.

    Skips the whole class when the installed GDAL / pyproj pair happens to agree on
    every candidate — the defect is a property of the *pair*, so there is nothing to
    regress against once the databases line up.
    """
    found = None
    for code in _SKEW_CANDIDATE_CODES:
        srs = osr.SpatialReference()
        try:
            srs.ImportFromEPSG(code)
        except RuntimeError:
            continue
        try:
            CRS.from_epsg(code)
        except PyprojCRSError:
            found = code
            break
    if found is None:
        pytest.skip(
            "installed GDAL and pyproj agree on every candidate code; the "
            "PROJ-database skew of issue #943 cannot be reproduced here"
        )
    return found


class TestProjDatabaseSkew:
    """CRSes GDAL's PROJ database knows and pyproj's does not (issue #943).

    pyramids *resolves* a raster's CRS with GDAL (`FindMatches`, `ImportFromEPSG`) but
    consumes it with pyproj, and the two ship different PROJ databases. A code GDAL
    hands out but pyproj cannot look up used to crash every downstream read.
    """

    def test_premise_pyproj_alone_cannot_build_the_code(self, skew_code):
        """The defect's precondition, asserted rather than assumed.

        Test scenario:
            Bare `pyproj.CRS.from_epsg` fails on the code while GDAL builds it
            happily. Without this the rest of the class could pass vacuously.
        """
        with pytest.raises(PyprojCRSError):
            CRS.from_epsg(skew_code)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(skew_code)
        assert srs.GetName(), "GDAL must resolve the code for the skew to exist"

    def test_crs_from_user_input_heals_int_code(self, skew_code):
        """An EPSG int only GDAL knows still yields a usable pyproj CRS.

        Test scenario:
            `crs_from_user_input(<code>)` falls back to GDAL's database and returns
            the same CRS GDAL names, instead of raising "crs not found".
        """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(skew_code)
        got = crs_from_user_input(skew_code)
        assert got.name == srs.GetName(), (
            f"expected the CRS GDAL names ({srs.GetName()!r}), got {got.name!r}"
        )

    @pytest.mark.parametrize("spelling", ["EPSG:{}", "{}"])
    def test_crs_from_user_input_heals_string_spellings(self, skew_code, spelling):
        """Both string spellings are healed, not just the bare int.

        Test scenario:
            STAC `proj:code` carries the `"EPSG:<code>"` authority form, and pyproj
            also accepts a bare numeric string — GDAL accepts only the prefixed one,
            so the rescue must normalise before delegating.
        """
        got = crs_from_user_input(spelling.format(skew_code))
        assert got.name == crs_from_user_input(skew_code).name, (
            f"{spelling.format(skew_code)!r} must resolve to the same CRS as the int"
        )

    def test_sr_from_user_input_builds_the_code(self, skew_code):
        """`sr_from_user_input` no longer raises on a GDAL-only code.

        Test scenario:
            This is the exact call that raised `CRSError: could not interpret
            10857 as a CRS` in the original report.
        """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(skew_code)
        assert sr_from_user_input(skew_code).GetName() == srs.GetName()

    def test_epsg_from_user_input_recovers_the_code(self, skew_code):
        """The EPSG integer survives the round trip through the rescue path.

        Test scenario:
            A CRS rebuilt from WKT has no pyproj catalogue entry, so `to_epsg()`
            alone returns None; the helper must ask GDAL and recover the code.
        """
        assert epsg_from_user_input(f"EPSG:{skew_code}") == skew_code

    def test_reproject_coordinates_accepts_the_code(self, skew_code):
        """Coordinates reproject out of a CRS only GDAL's database names.

        Test scenario:
            `reproject_coordinates` builds its transformer through the healing
            helper, so a GDAL-only source CRS transforms rather than raising.
        """
        xs, ys = reproject_coordinates(
            [5_000_000.0], [10_000_000.0], from_crs=skew_code, to_crs=4326
        )
        assert len(xs) == len(ys) == 1
        assert all(np.isfinite([xs[0], ys[0]])), (
            f"expected finite lon/lat, got ({xs[0]}, {ys[0]})"
        )

    def test_deprecated_code_keeps_pyprojs_reading(self):
        """The rescue must not fire when pyproj already has an answer.

        Test scenario:
            GDAL's `ImportFromEPSG(32663)` silently substitutes the non-deprecated
            EPSG:4087. Since pyproj resolves 32663 itself, the GDAL path must never
            run — otherwise the fix would quietly change the meaning of every
            deprecated code that works today.
        """
        # Assert the precondition through the same entry point the code under test
        # uses (`SetFromUserInput`), not `ImportFromEPSG` -- otherwise the guard can
        # hold while the path that matters behaves differently.
        srs = osr.SpatialReference()
        srs.SetFromUserInput(f"EPSG:{_DEPRECATED_CODE}")
        assert srs.GetAuthorityCode(None) == str(_DEPRECATED_REPLACEMENT), (
            "precondition: GDAL must actually substitute the replacement code"
        )
        assert crs_from_user_input(_DEPRECATED_CODE).to_epsg() == _DEPRECATED_CODE, (
            "pyproj resolved the code, so its reading must be kept unchanged"
        )

    def test_genuinely_bad_input_still_raises(self):
        """Healing must not turn nonsense into a CRS.

        Test scenario:
            A string that names no CRS, and a numeric code neither database
            carries, both still raise the documented `CRSError`.
        """
        with pytest.raises(CRSError, match="could not interpret"):
            crs_from_user_input("not-a-crs")
        with pytest.raises(CRSError, match="could not interpret"):
            crs_from_user_input(999_999)

    def test_bool_still_rejected(self):
        """`True` is an int but not a CRS.

        Test scenario:
            The bool guard must sit ahead of the rescue, so `True` never becomes
            "EPSG:1".
        """
        with pytest.raises(CRSError, match="not a valid CRS"):
            crs_from_user_input(True)

    def test_non_epsg_authority_not_passed_off_as_epsg(self):
        """An ESRI code must not be reported as an EPSG code.

        Test scenario:
            Robinson (`ESRI:54030`) has no EPSG code. The GDAL fallback reads its
            authority, so it must check the authority *name* and refuse rather
            than return 54030.
        """
        with pytest.raises(CRSError, match="no corresponding EPSG code"):
            epsg_from_user_input("ESRI:54030")

    def test_read_and_describe_a_raster_in_such_a_crs(self, skew_code):
        """The end-to-end failure from the report: reading and describing the raster.

        Test scenario:
            An in-memory raster carrying the GDAL-only CRS is opened and cropped to
            a lon/lat bbox. This is the `Dataset.crop(...)` call that raised
            `CRSError` against the Brazil Data Cube COGs.
        """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(skew_code)
        area = srs.GetAreaOfUse()
        if area is None:
            pytest.skip(
                f"EPSG:{skew_code} declares no area of use to place a raster in"
            )
        # Place the raster on the projection's own origin (its false easting /
        # northing), which is the one point guaranteed to be well inside the
        # projection's valid domain. Deriving it from the area of use instead can
        # land on a latitude the projection rejects, and GDAL's cutline transform
        # then fails for reasons that have nothing to do with the CRS lookup.
        # Anchor on the projection's origin, which for a projected CRS is its false
        # easting/northing and for a geographic one is 0/0 -- both well inside the
        # valid domain. Requiring a *non-zero* false easting silently skipped this
        # whole test for every geographic candidate, so it would have stopped running
        # entirely the day pyproj picked up EPSG:10857.
        centre_x = srs.GetProjParm("false_easting") or 0.0
        centre_y = srs.GetProjParm("false_northing") or 0.0
        projected = bool(srs.IsProjected())
        size = 64
        pixel = 1000.0 if projected else 0.01
        # `uuid` rather than the code alone: a fixed /vsimem path collides when the
        # suite runs distributed (`pytest -n`), where two workers share the process's
        # virtual filesystem namespace.
        path = f"/vsimem/proj_db_skew_{skew_code}_{uuid.uuid4().hex}.tif"
        raster = gdal.GetDriverByName("GTiff").Create(
            path, size, size, 1, gdal.GDT_Float32
        )
        try:
            raster.SetProjection(srs.ExportToWkt())
            raster.SetGeoTransform(
                [
                    centre_x - (size / 2) * pixel,
                    pixel,
                    0.0,
                    centre_y + (size / 2) * pixel,
                    0.0,
                    -pixel,
                ]
            )
            raster.GetRasterBand(1).WriteArray(
                np.arange(size * size, dtype="float32").reshape(size, size)
            )
            raster.FlushCache()
            raster = None

            dataset = Dataset.read_file(path)
            assert dataset.epsg == skew_code, (
                f"the raster should report EPSG:{skew_code}, got {dataset.epsg}"
            )
            # Derive the crop window from the raster's own extent rather than naming
            # fixed coordinates: a bbox that misses the raster still "succeeds", so a
            # fixed one would pass just as well if crop silently returned nothing.
            min_x, min_y, max_x, max_y = dataset.bounds.total_bounds
            (west, east), (south, north) = reproject_coordinates(
                [min_x, max_x],
                [min_y, max_y],
                from_crs=skew_code,
                to_crs=4326,
                precision=None,
            )
            assert all(np.isfinite([west, south, east, north])), (
                "the raster's own extent should reproject to finite lon/lat"
            )
            # Every one of these raised the issue-#943 `CRSError` before the fix,
            # because each hands the raster's EPSG code to pyproj (directly, or
            # through geopandas). They are the read/describe surface a caller needs
            # before anything else works.
            assert dataset.footprint().crs is not None, (
                "footprint() should return a georeferenced GeoDataFrame"
            )
            polygons = dataset.to_polygons(band=0)
            assert len(polygons) > 0, "to_polygons() should return polygons"
            assert polygons.crs is not None, "to_polygons() output should carry a CRS"
            assert RasterMeta.from_dataset(dataset).crs is not None, (
                "RasterMeta should describe the raster's CRS"
            )
            # And the operation from the original report. A rescued CRS reports
            # `to_epsg() is None`, which used to strip it out of the staged cutline
            # and fail the warp ("Cutline transformation failed", issue #964); the
            # cutline CRS is now stated explicitly, so this works too.
            min_x, min_y, max_x, max_y = dataset.bounds.total_bounds
            inset_x, inset_y = (max_x - min_x) / 4, (max_y - min_y) / 4
            cropped = dataset.crop(
                bbox=[
                    min_x + inset_x,
                    min_y + inset_y,
                    max_x - inset_x,
                    max_y - inset_y,
                ],
                epsg=skew_code,
                touch=True,
            )
            # shape is (bands, rows, cols): assert the *spatial* axes, since shape[0]
            # is the band count and would be 1 whatever crop returned.
            _, rows, cols = cropped.shape
            subset = f"a strict, non-empty subset of {size}x{size}"
            assert 0 < rows < size, f"rows should be {subset}, got {rows}"
            assert 0 < cols < size, f"cols should be {subset}, got {cols}"
        finally:
            raster = None
            # Only after every handle above is dropped: unlinking a /vsimem path that
            # still has an open dataset leaves GDAL holding a freed file.
            dataset = None
            gdal.Unlink(path)


class TestPyprojCanResolveEpsg:
    """Tests for the cached probe behind `crs_spec`'s code-vs-WKT choice."""

    def test_true_for_a_code_pyproj_carries(self):
        """A code in pyproj's own database is resolvable.

        Test scenario:
            EPSG:4326 is in every PROJ database, so the probe must say yes.
        """
        assert _pyproj_can_resolve_epsg(4326) is True

    def test_false_for_a_code_pyproj_lacks(self, skew_code):
        """A code only GDAL carries is not resolvable by pyproj.

        Test scenario:
            This is the condition that makes `crs_spec` prefer the WKT.
        """
        assert _pyproj_can_resolve_epsg(skew_code) is False

    def test_false_for_a_nonexistent_code(self):
        """A code no database carries is not resolvable.

        Test scenario:
            The probe must answer rather than propagate pyproj's exception.
        """
        assert _pyproj_can_resolve_epsg(999_999) is False


class TestCrsSpecResolvability:
    """`crs_spec` must return a specification downstream libraries can consume."""

    def test_prefers_a_resolvable_code_over_the_wkt(self):
        """An ordinary code is still preferred, so nothing changes for normal CRSes.

        Test scenario:
            EPSG:4326 resolves everywhere, so the int wins over the WKT.
        """
        wkt = sr_from_epsg(4326).ExportToWkt()
        assert crs_spec(4326, wkt) == 4326

    def test_falls_back_to_the_wkt_for_an_unresolvable_code(self, skew_code):
        """A code pyproj cannot look up yields the WKT instead.

        Test scenario:
            Returning the code would hand every consumer a specification that
            raises; the WKT describes the same CRS and parses everywhere.
        """
        wkt = sr_from_epsg(skew_code).ExportToWkt()
        assert crs_spec(skew_code, wkt) == wkt

    def test_returns_the_code_when_there_is_no_wkt_to_fall_back_to(self, skew_code):
        """With no WKT available the code is returned anyway.

        Test scenario:
            Half a specification beats none: the caller can still route it through
            `crs_from_user_input`, which heals it.
        """
        assert crs_spec(skew_code, "") == skew_code
        assert crs_spec(skew_code, None) == skew_code

    def test_no_crs_at_all_is_none(self):
        """Absence is still reported as `None`, not an empty string.

        Test scenario:
            The pre-existing contract must survive the resolvability change.
        """
        assert crs_spec(None, "") is None


class TestIntegerCode:
    """Tests for the NumPy-aware integer-code helper."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (4326, 4326),
            (np.int64(4326), 4326),
            (np.int32(4326), 4326),
            (True, None),
            (False, None),
            ("4326", None),
            (4326.0, None),
            (None, None),
        ],
    )
    def test_recognises_integers_but_not_bools_or_text(self, value, expected):
        """Integral values become plain ints; bools and non-integers do not.

        Args:
            value: The candidate specification.
            expected: The code it should reduce to, or None.

        Test scenario:
            NumPy scalars must count (they arrive from arrays and raster metadata),
            while `True` must never become EPSG:1.
        """
        assert _integer_code(value) == expected

    def test_returns_a_plain_python_int(self):
        """A NumPy scalar is converted, not passed through.

        Test scenario:
            Downstream string formatting (`f"EPSG:{code}"`) must not embed a NumPy
            repr, so the helper returns a built-in `int`.
        """
        assert type(_integer_code(np.int64(4326))) is int


class TestRescueDefensiveBranches:
    """The rescue path's failure branches must degrade, never escape."""

    def test_returns_none_when_pyproj_rejects_gdals_wkt(self, monkeypatch, skew_code):
        """A WKT that GDAL exports but pyproj refuses yields `None`, not a crash.

        Test scenario:
            Forcing `CRS.from_wkt` to raise simulates an export pyproj cannot read;
            the caller then reports the original pyproj failure instead.
        """

        def _reject(*_args, **_kwargs):
            raise PyprojCRSError("simulated rejection")

        monkeypatch.setattr(crs_module.CRS, "from_wkt", staticmethod(_reject))
        # The rescue memoises on the normalised text, so a value cached by an earlier
        # test would answer before the patched call is ever reached.
        clear_crs_caches()
        assert crs_module._pyproj_crs_via_gdal(skew_code) is None
        clear_crs_caches()

    def test_epsg_via_gdal_survives_an_authority_lookup_failure(self, monkeypatch):
        """A `RuntimeError` reading the authority yields `None`.

        Test scenario:
            GDAL's accessors raise under `UseExceptions`; the helper must answer
            `None` rather than propagate.
        """

        def _boom(self, _target):
            raise RuntimeError("simulated GDAL failure")

        monkeypatch.setattr(osr.SpatialReference, "GetAuthorityName", _boom)
        # `_epsg_via_gdal` memoises on the normalised text, so a value cached by an
        # earlier call would answer before the patched accessor is reached.
        clear_crs_caches()
        assert crs_module._epsg_via_gdal("EPSG:4326") is None
        clear_crs_caches()

    def test_epsg_matches_definition_is_false_for_an_unbuildable_code(self):
        """A code that cannot be built cannot be shown to match.

        Test scenario:
            `_epsg_matches_definition` must answer False rather than let
            `sr_from_epsg`'s failure escape.
        """
        srs = sr_from_epsg(4326)
        assert crs_module._epsg_matches_definition(999_999, srs) is False

    def test_gdal_parse_failure_returns_none(self):
        """Text GDAL cannot read yields `None` from the shared parse primitive.

        Test scenario:
            Covers the non-zero-return / raise path of `SetFromUserInput`.
        """
        assert crs_module._gdal_srs_from_text("definitely-not-a-crs") is None
        assert crs_module._gdal_input_text(object()) is None

    def test_non_zero_return_from_gdal_is_treated_as_failure(self, monkeypatch):
        """A non-zero `SetFromUserInput` return yields `None`, not a half-built SRS.

        Test scenario:
            pyramids installs `gdal.UseExceptions()`, so GDAL normally raises rather
            than returning a code — this pins the safety net for a caller that has
            disabled exceptions, where ignoring the return would hand back an empty
            spatial reference that silently claims to be a CRS.
        """
        monkeypatch.setattr(
            osr.SpatialReference, "SetFromUserInput", lambda self, *_a, **_k: 1
        )
        clear_crs_caches()
        assert crs_module._gdal_srs_from_text("EPSG:4326") is None
        clear_crs_caches()


class TestReprojectCoordinatesCrsErrors:
    """`reproject_coordinates` wraps CRS-parsing failures in pyramids' CRSError."""

    def test_unparseable_source_crs_raises_crs_error(self):
        """A source CRS that names nothing raises `CRSError` naming both CRSes.

        Test scenario:
            The wrapper exists so callers need not import pyproj to catch a bad-CRS
            failure; the message must identify which pair failed.
        """
        with pytest.raises(CRSError, match="reproject_coordinates failed to parse CRS"):
            reproject_coordinates([1.0], [1.0], from_crs="not-a-crs", to_crs=4326)

    def test_unparseable_target_crs_raises_crs_error(self):
        """The same applies to the target CRS.

        Test scenario:
            Both ends go through the healing helper, so both must surface the same
            wrapped error rather than a raw pyproj exception.
        """
        with pytest.raises(CRSError, match="reproject_coordinates failed to parse CRS"):
            reproject_coordinates([1.0], [1.0], from_crs=4326, to_crs="not-a-crs")

    def test_a_code_only_gdal_knows_still_transforms(self, skew_code):
        """A GDAL-only code builds a transformer instead of raising.

        Test scenario:
            This is the `reproject_coordinates` half of issue #943.
        """
        xs, ys = reproject_coordinates(
            [5_000_000.0], [10_000_000.0], from_crs=skew_code, to_crs=4326
        )
        assert np.isfinite([xs[0], ys[0]]).all(), (
            f"expected finite lon/lat, got ({xs[0]}, {ys[0]})"
        )


class TestRescueRefusesSubstitutions:
    """The rescue resolves the code that was asked for, or nothing (issues L1/L2)."""

    def test_an_esri_code_is_not_accepted_for_an_epsg_request(self):
        """`54030` as an EPSG request must not resolve to ESRI:54030.

        Test scenario:
            `SetFromUserInput("EPSG:54030")` happily returns Robinson under the ESRI
            authority. Accepting it would answer a request for one CRS with another,
            under a different authority, silently.
        """
        with pytest.raises(CRSError):
            crs_from_user_input(54030)

    def test_a_deprecated_code_is_not_replaced_by_its_successor(self):
        """The rescue refuses GDAL's non-deprecated replacement.

        Test scenario:
            `SetFromUserInput("EPSG:32663")` resolves to EPSG:4087. The pyproj-first
            ordering hides this while pyproj carries the code, so the guard is what
            actually prevents it on the rescue path.
        """
        assert crs_module._pyproj_crs_via_gdal(32663) is None
        assert crs_module._epsg_via_gdal("EPSG:32663") is None

    def test_the_requested_code_is_still_resolved(self, skew_code):
        """A code GDAL resolves *as itself* is still accepted.

        Test scenario:
            The guard must reject substitutions without rejecting the whole point of
            the rescue.
        """
        rescued = crs_module._pyproj_crs_via_gdal(skew_code)
        assert rescued is not None, f"EPSG:{skew_code} should still be rescued"

    def test_a_wkt_request_names_no_code_and_is_not_guarded(self):
        """A WKT names no code, so there is nothing to substitute.

        Test scenario:
            The guard keys on a *code* request; a definition-only input must pass
            through it untouched.
        """
        wkt = sr_from_epsg(4326).ExportToWkt()
        assert crs_module._pyproj_crs_via_gdal(wkt) is not None


class TestResolutionIsCached:
    """The rescue's cost is paid once per specification, not per call."""

    def test_repeated_resolution_returns_the_identical_object(self, skew_code):
        """A second call is served from the cache.

        Test scenario:
            The expensive step is pyproj's *failed* lookup, which happens before the
            GDAL rescue is reached — so the cache has to sit at the entry point.
            Identity is the observable proof it does.
        """
        first = crs_from_user_input(skew_code)
        second = crs_from_user_input(skew_code)
        assert first is second, "a repeated resolution should be served from the cache"

    def test_unhashable_input_is_still_rejected_cleanly(self):
        """An unhashable value cannot key the cache and must not crash on that.

        Test scenario:
            A list is not a CRS; it must raise `CRSError` rather than the `TypeError`
            a cache lookup would produce.
        """
        with pytest.raises(CRSError):
            crs_from_user_input([1, 2])

    def test_unhashable_input_reaches_epsg_resolution_uncached(self):
        """An unhashable value cannot key the cache and is still rejected cleanly.

        Test scenario:
            `epsg_from_user_input` mirrors `crs_from_user_input`'s cache dispatch, so
            it needs the same escape hatch: a list must raise `CRSError` rather than
            the `TypeError` a cache lookup would produce.
        """
        with pytest.raises(CRSError):
            epsg_from_user_input([1, 2])
