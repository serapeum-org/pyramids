"""Linux-only integration test: read a real GOES granule from NOAA S3 and
verify the geostationary geotransform fix (issue #449) end to end.

The synthetic tests in ``test_geostationary_geotransform.py`` prove the
radians-to-metres mechanism. This test closes the remaining gap by reading an
actual GOES-16 (``lon_0=-75``) and GOES-18 (``lon_0=-137``) ABI granule and
asserting the cube comes back with a geostationary CRS, a metre geotransform,
and a non-degenerate ``to_crs(4326)``.

Gating:

* **Linux only** — GDAL's netCDF driver needs Linux ``userfaultfd`` to open a
  ``.nc`` over any ``/vsi*`` path (the same constraint as
  ``test_netcdf_archive.py``), so it is skipped on Windows / macOS.
* **Opt-in** — it hits the public NOAA Open Data buckets (no credentials, via
  ``AWS_NO_SIGN_REQUEST``). To keep the default suite offline and fast it only
  runs when ``PYRAMIDS_RUN_GOES_GRANULE_TEST=1`` is set.
* **Network-tolerant** — if the bucket/prefix is unreachable or empty it skips
  rather than fails.

Run it with::

    PYRAMIDS_RUN_GOES_GRANULE_TEST=1 pixi run -e dev pytest \
        tests/netcdf/test_goes_real_granule.py -v
"""
from __future__ import annotations

import fnmatch
import os
import sys

import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.slow

_OPT_IN = os.environ.get("PYRAMIDS_RUN_GOES_GRANULE_TEST") == "1"

# A historical full-disc Cloud & Moisture Imagery (CMIPF) hour NOAA retains:
# day-of-year 180 (~Jun 28) 2024, 12Z, ABI band 13 (clean longwave IR).
_PREFIX = "ABI-L2-CMIPF/2024/180/12/"
_MEMBER_GLOB = "OR_ABI-L2-CMIPF-M6C13_{sat}_*.nc"
_S3_CONFIG = {"AWS_NO_SIGN_REQUEST": "YES", "AWS_REGION": "us-east-1"}


def _first_granule(bucket: str, sat: str) -> str | None:
    """Return a ``/vsis3/`` path to the first matching CMIPF granule, or None.

    Returns None on any listing failure (offline, throttled, empty prefix) so
    the caller can skip cleanly.
    """
    result: str | None = None
    with gdal.config_options(_S3_CONFIG):
        try:
            entries = gdal.ReadDir(f"/vsis3/{bucket}/{_PREFIX}")
        except RuntimeError:
            entries = None
        if entries:
            pattern = _MEMBER_GLOB.format(sat=sat)
            members = sorted(e for e in entries if fnmatch.fnmatch(e, pattern))
            if members:
                result = f"/vsis3/{bucket}/{_PREFIX}{members[0]}"
    return result


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="GDAL netCDF driver needs Linux userfaultfd for /vsis3/ paths",
)
@pytest.mark.skipif(
    not _OPT_IN,
    reason="set PYRAMIDS_RUN_GOES_GRANULE_TEST=1 to run the NOAA S3 granule test",
)
@pytest.mark.parametrize(
    "bucket, sat, expected_lon_0",
    [
        ("noaa-goes16", "G16", -75.0),
        ("noaa-goes18", "G18", -137.0),
    ],
)
class TestRealGoesGranule:
    """Read an actual GOES granule and verify the geostationary fix."""

    def test_geostationary_geotransform_and_reproject(
        self, bucket: str, sat: str, expected_lon_0: float
    ):
        """The granule reads with a metre geotransform and reprojects cleanly.

        Args:
            bucket: NOAA Open Data bucket (``noaa-goes16`` / ``noaa-goes18``).
            sat: ABI satellite token in the filename (``G16`` / ``G18``).
            expected_lon_0: Sub-satellite longitude for that platform.

        Test scenario:
            Open the CMIPF ``CMI`` variable straight from S3, assert it is read
            as geostationary with a metre-scaled geotransform centred on the
            expected sub-satellite longitude, and that ``to_crs(4326)`` yields a
            non-degenerate extent (the user-facing symptom from issue #449).
        """
        path = _first_granule(bucket, sat)
        if path is None:
            pytest.skip(f"no CMIPF granule reachable at s3://{bucket}/{_PREFIX}")

        with gdal.config_options(_S3_CONFIG):
            cube = NetCDF.read_file(path).get_variable("CMI")

            assert cube._is_geostationary(), "granule CRS not read as geostationary"

            gt = cube.geotransform
            assert abs(gt[1]) > 1000, f"geotransform still in radians: {gt}"

            srs = cube.raster.GetSpatialRef()
            lon_0 = srs.GetProjParm("central_meridian", 999.0)
            assert lon_0 == pytest.approx(expected_lon_0, abs=1.0), (
                f"sub-satellite longitude {lon_0} != expected {expected_lon_0}"
            )

            warped = cube.to_crs(4326)
            minx, miny, maxx, maxy = warped.bbox
            assert maxx - minx > 1.0, f"degenerate reprojected width: {warped.bbox}"
            assert maxy - miny > 1.0, f"degenerate reprojected height: {warped.bbox}"
