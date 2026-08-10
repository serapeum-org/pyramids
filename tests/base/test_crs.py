"""Tests for CRS-resolution helpers in :mod:`pyramids.base.crs`."""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr
from pyproj import CRS, Transformer

from pyramids.base._errors import CRSError
from pyramids.base.crs import (
    crs_from_user_input,
    epsg_from_user_input,
    epsg_from_wkt,
    get_epsg_from_prj,
    reproject_coordinates,
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
        except Exception:
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
        with pytest.raises(Exception):
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
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(_DEPRECATED_CODE)
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

    def test_crop_on_a_raster_in_such_a_crs(self, skew_code):
        """The end-to-end failure from the report: reading and cropping the raster.

        Test scenario:
            An in-memory raster carrying the GDAL-only CRS is opened and cropped to
            a lon/lat bbox. This is the `Dataset.crop(...)` call that raised
            `CRSError` against the Brazil Data Cube COGs.
        """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(skew_code)
        path = f"/vsimem/proj_db_skew_{skew_code}.tif"
        raster = gdal.GetDriverByName("GTiff").Create(path, 64, 64, 1, gdal.GDT_Float32)
        try:
            raster.SetProjection(srs.ExportToWkt())
            raster.SetGeoTransform(
                [5_000_000.0, 1000.0, 0.0, 10_000_000.0, 0.0, -1000.0]
            )
            raster.GetRasterBand(1).WriteArray(
                np.arange(64 * 64, dtype="float32").reshape(64, 64)
            )
            raster.FlushCache()
            raster = None

            dataset = Dataset.read_file(path)
            assert dataset.epsg == skew_code, (
                f"the raster should report EPSG:{skew_code}, got {dataset.epsg}"
            )
            cropped = dataset.crop(
                bbox=[-46.8, -23.7, -46.3, -23.2], epsg=4326, touch=True
            )
            assert cropped.shape[0] >= 1, "crop must return a raster, not raise"
        finally:
            raster = None
            gdal.Unlink(path)
