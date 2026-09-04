"""The two SRS builders do not agree on axis order, and must not be swapped.

`sr_from_user_input` stamps `OAMS_TRADITIONAL_GIS_ORDER`; `sr_from_epsg` leaves
GDAL's authority-compliant default, which for a geographic CRS is latitude
first. Its docstring claimed the two matched, which is what made collapsing them
look harmless -- swapping one for the other transposes coordinates on any
geographic CRS.
"""

from __future__ import annotations

import pytest
from osgeo import gdal, osr

from pyramids.base._coverage import resolve_native_srs
from pyramids.base.crs import sr_from_epsg, sr_from_user_input

pytestmark = pytest.mark.core

TRADITIONAL = osr.OAMS_TRADITIONAL_GIS_ORDER


class TestAxisOrderContract:
    """Each builder's axis order is a contract, pinned here."""

    def test_sr_from_user_input_is_traditional(self):
        """Longitude first, matching the geotransform."""
        assert sr_from_user_input(4326).GetAxisMappingStrategy() == TRADITIONAL

    def test_sr_from_epsg_is_not_traditional(self):
        """Authority-compliant: latitude first for a geographic CRS.

        This is the asymmetry the docstring used to deny.
        """
        assert sr_from_epsg(4326).GetAxisMappingStrategy() != TRADITIONAL

    @pytest.mark.parametrize("crs", [4326, "EPSG:4326", "EPSG:3857", 3857])
    def test_user_input_is_traditional_for_every_spelling(self, crs):
        """The stamp does not depend on how the CRS was written."""
        assert sr_from_user_input(crs).GetAxisMappingStrategy() == TRADITIONAL


@pytest.fixture
def dataset_without_a_crs():
    """A GDAL dataset carrying no spatial reference, forcing the shim."""
    handle = gdal.GetDriverByName("MEM").Create("", 4, 4, 1, gdal.GDT_Float32)
    handle.SetGeoTransform((0.0, 1.0, 0.0, 4.0, 0.0, -1.0))
    return handle


class TestResolveNativeSrs:
    """`resolve_native_srs` now agrees with its own sibling branch."""

    def test_a_geographic_coverage_crs_is_traditional(self, dataset_without_a_crs):
        """It used to build the SRS raw, leaving authority order in place.

        `native_projwin` builds its transformer with `always_xy=True`, and the
        other branch of this same function already produced traditional order,
        so the raw build was the odd one out.
        """
        result = resolve_native_srs(dataset_without_a_crs, "EPSG:4326")

        assert result.GetAxisMappingStrategy() == TRADITIONAL

    def test_an_uninterpretable_crs_still_raises_value_error(
        self, dataset_without_a_crs
    ):
        """The wrapper's contract is unchanged, including for a CRSError."""
        with pytest.raises(ValueError, match="could not be interpreted"):
            resolve_native_srs(dataset_without_a_crs, "not a crs at all")
