"""Tests for how pyramids reports a raster that has no CRS (ARC-26).

The old behaviour rewrote an absent projection to EPSG:4326, which conflated
three cases. These tests pin the distinction that replaced it: no evidence means
no CRS, a CF grid's degrees axes still mean WGS 84, and a projected grid is
never relabelled on the strength of auxiliary lat/lon arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import CRSError
from pyramids.base.crs import (
    cf_geographic_wkt,
    crs_spec,
    epsg_of_crs,
    require_crs_spec,
    sr_from_epsg,
)
from pyramids.dataset import Dataset
from pyramids.dataset.ops._geobox_zarr import geobox_crs

pytestmark = pytest.mark.core

_CF_GEOGRAPHIC = "tests/data/netcdf/cf__7v__1d3-2d3-3d1__y-asc.nc"
_GEOSTATIONARY = "tests/data/netcdf/cf__9v__1d7-2d2__geos__y-desc.nc"


@pytest.fixture(scope="function")
def crs_less_raster(tmp_path) -> str:
    """Create a GeoTIFF with a geotransform but no projection.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        str: Path to the created raster.
    """
    path = str(tmp_path / "nocrs.tif")
    dataset = gdal.GetDriverByName("GTiff").Create(path, 4, 4, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    dataset.GetRasterBand(1).WriteArray(np.ones((4, 4), dtype="uint8"))
    dataset = None
    return path


class TestEpsgOfCrs:
    """Tests for `epsg_of_crs`."""

    def test_empty_projection_is_absent_not_wgs84(self):
        """An empty projection reports no CRS.

        Test scenario:
            This is the whole point of ARC-26 — the old helper substituted 4326
            here, so an ungeoreferenced raster claimed to be WGS 84.
        """
        assert epsg_of_crs("") is None, "an empty projection must not resolve to a CRS"
        assert epsg_of_crs(None) is None, (
            "a missing projection must not resolve to a CRS"
        )

    def test_real_projection_resolves(self):
        """A real projection still resolves to its EPSG code."""
        wkt = sr_from_epsg(3857).ExportToWkt()
        assert epsg_of_crs(wkt) == 3857, "a valid CRS must still resolve"


class TestCrsSpec:
    """Tests for `crs_spec` and `require_crs_spec`."""

    def test_prefers_epsg_then_wkt_then_none(self):
        """The spec falls back in order and ends at `None`.

        Test scenario:
            Replaces the `dataset.epsg or dataset.crs` idiom, which evaluated to
            the empty string once `epsg` could be `None` — a value every CRS
            constructor rejects with an opaque "Invalid projection".
        """
        assert crs_spec(4326, "WKT") == 4326, "an EPSG code should win"
        assert crs_spec(None, "WKT") == "WKT", "the WKT is the fallback"
        assert crs_spec(None, "") is None, "no CRS must be reported as None, not ''"

    def test_require_names_the_operation(self):
        """`require_crs_spec` refuses with a message naming the operation."""
        with pytest.raises(CRSError, match="reproject"):
            require_crs_spec(None, "", "reproject")

    def test_require_passes_a_present_crs_through(self):
        """A present CRS is returned unchanged."""
        assert require_crs_spec(3857, "", "reproject") == 3857, (
            "a real CRS must pass through"
        )


class TestCfGeographicInference:
    """Tests for the CF degrees-axes convention."""

    def test_degrees_axes_mean_wgs84(self):
        """Degrees on both axes identify a geographic grid.

        Test scenario:
            CF leaves the datum implicit for a lat/lon grid with no
            `grid_mapping`; the whole ecosystem reads those as WGS 84.
        """
        assert cf_geographic_wkt({"degrees_east", "degrees_north"}), (
            "CF degrees axes are geographic"
        )

    @pytest.mark.parametrize(
        "units", [{"degree_east", "degree_north"}, {"degreeE", "degreeN"}]
    )
    def test_cf_singular_spellings_are_accepted(self, units: set[str]):
        """CF permits singular and abbreviated axis spellings.

        Args:
            units: Axis unit spellings to accept.

        Test scenario:
            The ROMS staggered sample uses `degree_east`, so matching only the
            plural form read it as having no CRS at all.
        """
        lowered = {u.lower() for u in units}
        assert cf_geographic_wkt(lowered), f"{units} should be recognised as geographic"

    def test_one_axis_alone_is_not_evidence(self):
        """A longitude axis without a latitude axis proves nothing."""
        assert cf_geographic_wkt({"degrees_east"}) == "", (
            "one axis alone is not a geographic grid"
        )

    def test_projected_axis_units_veto_the_inference(self):
        """A metre axis means the grid is projected, whatever else it ships.

        Test scenario:
            A projected CF file often carries 2-D auxiliary lat/lon arrays in
            degrees. Reading those as the CRS would label a 30-metre grid as
            WGS 84.
        """
        result = cf_geographic_wkt({"degrees_east", "degrees_north"}, {"m"})
        assert result == "", "a metre axis must veto the geographic inference"

    def test_metres_on_a_data_variable_do_not_veto(self):
        """Only axis units veto — a data variable may legitimately be in metres.

        Test scenario:
            A ROMS bathymetry or sea-surface height is in metres on an otherwise
            geographic grid, so the veto reads dimension axes only.
        """
        result = cf_geographic_wkt({"degrees_east", "degrees_north", "meter"}, set())
        assert result, "metres on a data variable must not veto a geographic grid"


class TestDatasetReportsAbsentCrs:
    """Tests for `Dataset.epsg` on a raster with no CRS."""

    def test_crs_less_raster_reports_none(self, crs_less_raster: str):
        """An ungeoreferenced raster reports no CRS rather than WGS 84."""
        dataset = Dataset.read_file(crs_less_raster)
        assert dataset.epsg is None, f"expected no CRS, got EPSG:{dataset.epsg}"

    def test_create_from_array_propagates_absence(self):
        """Building from a falsy `epsg` yields an ungeoreferenced raster.

        Test scenario:
            A result rebuilt from an ungeoreferenced source must not acquire a
            projection its input never had.
        """
        dataset = Dataset.create_from_array(
            np.ones((4, 4), dtype="float32"),
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=None,
        )
        assert dataset.epsg is None, f"expected no CRS, got EPSG:{dataset.epsg}"


class TestGeoboxCrs:
    """Tests for `geobox_crs`."""

    def test_wkt_is_authoritative(self):
        """The recorded WKT wins over the EPSG code."""
        assert geobox_crs({"crs_wkt": "GEOGCS[]", "epsg": 4326}) == "GEOGCS[]", (
            "crs_wkt is authoritative"
        )

    def test_epsg_used_when_no_wkt(self):
        """A bare EPSG code is used when no WKT was recorded."""
        assert geobox_crs({"crs_wkt": "", "epsg": 3857}) == 3857, (
            "the EPSG code is the fallback"
        )

    def test_zero_sentinel_is_absent_not_wgs84(self):
        """The `epsg: 0` sentinel means no CRS, not WGS 84.

        Test scenario:
            The readers previously spelled this `geobox["epsg"] or 4326`, which
            resurrected the removed default on every round trip.
        """
        assert geobox_crs({"crs_wkt": "", "epsg": 0}) is None, (
            "epsg 0 means no authority code"
        )


class TestOperationsRefuseWithoutACrs:
    """Tests that CRS-requiring operations refuse rather than fabricate."""

    def test_to_crs_refuses(self, crs_less_raster: str):
        """`to_crs` refuses instead of stamping the target code.

        Test scenario:
            Warping from an unknown frame produced a raster carrying `to_epsg`
            as an unearned claim, with the geotransform untouched.
        """
        dataset = Dataset.read_file(crs_less_raster)
        with pytest.raises(CRSError, match="reproject"):
            dataset.to_crs(3857)

    def test_align_refuses_when_both_sides_lack_a_crs(self, crs_less_raster: str):
        """Two CRS-less rasters do not slip past the guard by comparing equal.

        Test scenario:
            Both report `epsg is None`, so an equality check on `epsg` alone
            skipped the guard and the result was stamped with the reference's
            projection anyway.
        """
        first = Dataset.read_file(crs_less_raster)
        second = Dataset.read_file(crs_less_raster)
        with pytest.raises(CRSError, match="align"):
            first.align(second)


class TestReadsStillWorkWithoutACrs:
    """Tests that reads which need no transform keep working."""

    def test_read_part_reads_in_the_rasters_own_coordinates(self, crs_less_raster: str):
        """A bbox read needs no CRS when no reprojection is involved.

        Test scenario:
            `bbox_crs` defaults to 4326 and never equals `None`, so guarding the
            transform unconditionally broke a plain windowed read that worked
            before. With no CRS the bbox is read in the raster's own space —
            here the fixture's own extent, x 0..4 and y -4..0.
        """
        dataset = Dataset.read_file(crs_less_raster)
        assert dataset.read_part((1, -3, 3, -1)) is not None, (
            "a windowed read must not require a CRS"
        )

    def test_point_reads_in_the_rasters_own_coordinates(self, crs_less_raster: str):
        """Point sampling likewise needs no CRS when no transform is involved."""
        dataset = Dataset.read_file(crs_less_raster)
        assert dataset.point(2, -2) is not None, "point sampling must not require a CRS"


class TestNetCDFCrsResolution:
    """Tests for how a NetCDF resolves its CRS (ARC-26)."""

    def test_cf_geographic_variable_resolves_to_wgs84(self):
        """A CF variable with degrees axes and no grid_mapping reads as 4326."""
        from pyramids.netcdf import NetCDF

        container = NetCDF.read_file(_CF_GEOGRAPHIC)
        variable = container.get_variable("tos")
        assert variable.epsg == 4326, f"CF degrees axes should read as WGS 84, got {variable.epsg}"

    def test_container_borrows_its_variables_crs(self):
        """A root container has no projection of its own and borrows one.

        Test scenario:
            The georeference lives on the variables, so `GetProjection()` is
            empty for the container even when every variable is georeferenced.
        """
        from pyramids.netcdf import NetCDF

        container = NetCDF.read_file(_CF_GEOGRAPHIC)
        assert container.raster.GetProjection() == "", "the container should carry no projection itself"
        assert container.epsg == 4326, f"the container should borrow its variables' CRS, got {container.epsg}"

    def test_geostationary_is_not_relabelled(self):
        """A geostationary grid keeps reporting no EPSG code.

        Test scenario:
            Its auxiliary arrays are in degrees, so an inference that ignored
            axis units would mislabel the scan-angle grid as WGS 84 (#706).
        """
        from pyramids.netcdf import NetCDF

        variable = NetCDF.read_file(_GEOSTATIONARY).get_variable("CMI")
        assert variable.epsg is None, f"a geostationary grid has no EPSG code, got {variable.epsg}"
        assert variable.crs, "but it does carry a CRS as WKT"

    def test_resolved_epsg_is_memoised(self):
        """Repeated `.epsg` reads do not re-run the variable scan.

        Test scenario:
            Resolution opens every MDArray, and `.epsg` sits on the hot path for
            spatial ops, so the answer has to stick — including when it is None.
        """
        from pyramids.netcdf import NetCDF

        variable = NetCDF.read_file(_CF_GEOGRAPHIC).get_variable("tos")
        first = variable.epsg
        assert variable._epsg_resolved is True, "the first read should mark the resolution as done"
        assert variable.epsg == first, "the memoised answer must match the resolved one"
