"""Tests for CF grid mapping reading (CF-4).

Tests grid_mapping_to_srs() round-trip: write CRS via
srs_to_grid_mapping, reconstruct via grid_mapping_to_srs,
verify the SRS matches.
"""

import pytest
from osgeo import osr

from pyramids.netcdf.cf import grid_mapping_to_srs, srs_to_grid_mapping

pytestmark = pytest.mark.core


class TestGridMappingToSrs:
    """Tests for cf.grid_mapping_to_srs."""

    def _round_trip(self, epsg):
        """Write SRS -> CF params -> reconstruct SRS -> check."""
        srs_orig = osr.SpatialReference()
        srs_orig.ImportFromEPSG(epsg)
        gm_name, params = srs_to_grid_mapping(srs_orig)
        # Remove crs_wkt to test the parameter-based reconstruction
        params_no_wkt = {k: v for k, v in params.items() if k != "crs_wkt"}
        srs_rebuilt = grid_mapping_to_srs(gm_name, params_no_wkt)
        return srs_orig, srs_rebuilt, gm_name

    def test_crs_wkt_fast_path(self):
        """grid_mapping_to_srs uses crs_wkt when available.

        Test scenario:
            If crs_wkt is in params, it should be used directly
            without parsing individual parameters.
        """
        srs_orig = osr.SpatialReference()
        srs_orig.ImportFromEPSG(32637)
        _, params = srs_to_grid_mapping(srs_orig)
        srs_rebuilt = grid_mapping_to_srs("transverse_mercator", params)
        assert srs_rebuilt.IsSame(srs_orig), "SRS should match when crs_wkt is provided"

    def test_geographic_4326_round_trip(self):
        """EPSG:4326 round-trip through CF parameters.

        Test scenario:
            latitude_longitude with WGS84 ellipsoid should produce
            a geographic SRS.
        """
        srs_orig, srs_rebuilt, gm_name = self._round_trip(4326)
        assert (
            gm_name == "latitude_longitude"
        ), f"Expected latitude_longitude, got {gm_name}"
        assert srs_rebuilt.IsGeographic() == 1, "Rebuilt SRS should be geographic"

    def test_utm_32637_round_trip(self):
        """EPSG:32637 round-trip through CF parameters.

        Test scenario:
            Transverse Mercator with zone 37N parameters should
            reconstruct correctly.
        """
        srs_orig, srs_rebuilt, gm_name = self._round_trip(32637)
        assert (
            gm_name == "transverse_mercator"
        ), f"Expected transverse_mercator, got {gm_name}"
        assert srs_rebuilt.IsProjected() == 1, "Rebuilt SRS should be projected"
        cm_orig = srs_orig.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN, 0.0)
        cm_rebuilt = srs_rebuilt.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN, 0.0)
        assert (
            abs(cm_orig - cm_rebuilt) < 0.01
        ), f"Central meridian mismatch: {cm_orig} vs {cm_rebuilt}"

    def test_unsupported_raises_valueerror(self):
        """Unknown grid_mapping_name without crs_wkt raises ValueError.

        Test scenario:
            A made-up projection name should raise ValueError.
        """
        with pytest.raises(ValueError, match="Unsupported"):
            grid_mapping_to_srs("rotated_latitude_longitude", {})

    def test_earth_radius_sphere(self):
        """earth_radius param creates a spherical CRS.

        Test scenario:
            When earth_radius is specified instead of semi_major_axis,
            the SRS should be spherical.
        """
        srs = grid_mapping_to_srs(
            "latitude_longitude",
            {
                "earth_radius": 6371000.0,
            },
        )
        assert srs.IsGeographic() == 1, "Should be geographic"
        assert (
            abs(srs.GetSemiMajor() - 6371000.0) < 1.0
        ), f"Expected semi_major ~6371000, got {srs.GetSemiMajor()}"
