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


@pytest.fixture
def dataset_with_an_authority_ordered_crs():
    """A GDAL dataset whose own SRS is latitude-first, as a driver hands it back.

    Returns:
        gdal.Dataset: A 4x4 MEM raster carrying EPSG:4326 in the mapping GDAL
            attaches by default, which is the input the clone branch has to
            correct.
    """
    handle = gdal.GetDriverByName("MEM").Create("", 4, 4, 1, gdal.GDT_Float32)
    handle.SetGeoTransform((0.0, 1.0, 0.0, 4.0, 0.0, -1.0))
    handle.SetSpatialRef(sr_from_epsg(4326))
    return handle


class TestResolveNativeSrs:
    """`resolve_native_srs` now agrees with its own sibling branch."""

    def test_a_geographic_coverage_crs_is_traditional(self, dataset_without_a_crs):
        """It used to build the SRS raw, leaving authority order in place.

        Test scenario:
            The `coverage_crs` branch built its SRS with a bare
            `SetFromUserInput`, which leaves GDAL's authority-compliant
            default -- latitude first for a geographic CRS -- while the clone
            branch produced traditional order, so the two disagreed. What
            consumes the difference is `SetSpatialRef` on the result raster,
            which carries the SRS *object*, and with it the mapping, onto what
            the caller gets back. It is deliberately not `native_projwin`: that
            sees the SRS only as `ExportToWkt()`, and WKT does not encode the
            axis mapping at all, so the earlier reading of this test (that
            `native_projwin`'s `always_xy` transformer was the reason) named a
            mechanism the stamp is invisible to.
        """
        result = resolve_native_srs(dataset_without_a_crs, "EPSG:4326")

        assert result.GetAxisMappingStrategy() == TRADITIONAL, (
            "the coverage_crs branch handed back an authority-ordered SRS"
        )

    def test_the_clone_branch_is_traditional_too(
        self, dataset_with_an_authority_ordered_crs
    ):
        """The branch that reads the dataset's own SRS is the other half of the fix.

        Args:
            dataset_with_an_authority_ordered_crs: A raster whose SRS is
                latitude-first, which is what a driver attaches.

        Test scenario:
            Agreement between the branches is only meaningful if the input to
            this one actually differs, so the dataset's own strategy is
            asserted first: it is authority-compliant, and passing that through
            would make a WCS/WMS result raster declare latitude-first or
            longitude-first depending only on whether the service advertised a
            CRS PROJ could resolve.
        """
        raw = dataset_with_an_authority_ordered_crs.GetSpatialRef()
        assert raw.GetAxisMappingStrategy() != TRADITIONAL, (
            "the fixture no longer reproduces the driver's default mapping"
        )

        result = resolve_native_srs(dataset_with_an_authority_ordered_crs, None)

        assert result.GetAxisMappingStrategy() == TRADITIONAL, (
            "the clone branch passed the driver's axis order through"
        )

    def test_the_stamp_lands_on_a_clone_not_on_the_dataset(
        self, dataset_with_an_authority_ordered_crs
    ):
        """Correcting the axis order must not rewrite the caller's dataset.

        Args:
            dataset_with_an_authority_ordered_crs: A raster whose SRS is
                latitude-first.

        Test scenario:
            The stamp is applied to `srs.Clone()`. Applying it to the object
            `GetSpatialRef()` returned would change how every *other* reader of
            that same dataset interprets its coordinates, as a side effect of
            asking a coverage helper what its CRS is.
        """
        resolve_native_srs(dataset_with_an_authority_ordered_crs, None)

        after = dataset_with_an_authority_ordered_crs.GetSpatialRef()
        assert after.GetAxisMappingStrategy() != TRADITIONAL, (
            "resolving the native SRS mutated the dataset's own axis mapping"
        )

    def test_the_coverage_crs_branch_keeps_its_authority_code(
        self, dataset_without_a_crs
    ):
        """Routing through pyproj shortens the WKT; the code it is read by survives.

        Args:
            dataset_without_a_crs: A raster forcing the `coverage_crs` branch.

        Test scenario:
            `sr_from_user_input` round-trips the CRS through pyproj, which
            drops the nested `AUTHORITY` nodes a bare `SetFromUserInput` left
            on the datum, spheroid, prime meridian and unit -- so a result
            raster's `.GetProjection()` string is shorter than it used to be
            for any read whose CRS came from an explicit `coverage_crs`. What
            must not change is the *root* authority node, because that is what
            `GetAuthorityCode(None)` reads and therefore what `Dataset.epsg`
            reports.
        """
        result = resolve_native_srs(dataset_without_a_crs, "EPSG:4326")

        assert result.GetAuthorityCode(None) == "4326", (
            "the pyproj round-trip lost the root authority node, so Dataset.epsg "
            f"would no longer resolve; got {result.GetAuthorityCode(None)!r}"
        )

    def test_an_uninterpretable_crs_still_raises_value_error(
        self, dataset_without_a_crs
    ):
        """The wrapper's contract is unchanged, including for a CRSError."""
        with pytest.raises(ValueError, match="could not be interpreted"):
            resolve_native_srs(dataset_without_a_crs, "not a crs at all")
