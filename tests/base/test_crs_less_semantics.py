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
    crs_equal,
    crs_spec,
    epsg_of_crs,
    require_crs_spec,
    sr_from_epsg,
)
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.ops._geobox_zarr import geobox_crs
from pyramids.netcdf import NetCDF

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
        container = NetCDF.read_file(_CF_GEOGRAPHIC)
        variable = container.get_variable("tos")
        assert variable.epsg == 4326, (
            f"CF degrees axes should read as WGS 84, got {variable.epsg}"
        )

    def test_container_borrows_its_variables_crs(self):
        """A root container has no projection of its own and borrows one.

        Test scenario:
            The georeference lives on the variables, so `GetProjection()` is
            empty for the container even when every variable is georeferenced.
        """
        container = NetCDF.read_file(_CF_GEOGRAPHIC)
        assert container.raster.GetProjection() == "", (
            "the container should carry no projection itself"
        )
        assert container.epsg == 4326, (
            f"the container should borrow its variables' CRS, got {container.epsg}"
        )

    def test_geostationary_is_not_relabelled(self):
        """A geostationary grid keeps reporting no EPSG code.

        Test scenario:
            Its auxiliary arrays are in degrees, so an inference that ignored
            axis units would mislabel the scan-angle grid as WGS 84 (#706).
        """
        variable = NetCDF.read_file(_GEOSTATIONARY).get_variable("CMI")
        assert variable.epsg is None, (
            f"a geostationary grid has no EPSG code, got {variable.epsg}"
        )
        assert variable.crs, "but it does carry a CRS as WKT"

    def test_resolved_epsg_is_memoised(self):
        """Repeated `.epsg` reads do not re-run the variable scan.

        Test scenario:
            Resolution opens every MDArray, and `.epsg` sits on the hot path for
            spatial ops, so the answer has to stick — including when it is None.
        """
        variable = NetCDF.read_file(_CF_GEOGRAPHIC).get_variable("tos")
        first = variable.epsg
        assert variable._epsg_resolved is True, (
            "the first read should mark the resolution as done"
        )
        assert variable.epsg == first, "the memoised answer must match the resolved one"


class TestDefaultCoordinateReads:
    """Tests for the default `bbox_crs` / `point_crs`, on a *georeferenced* raster.

    Every pre-existing call site passes the CRS explicitly, so a regression in
    the default path was invisible to the suite.
    """

    @pytest.fixture(scope="function")
    def wgs84_raster(self, tmp_path) -> str:
        """Create a small GeoTIFF that does carry EPSG:4326.

        Args:
            tmp_path: pytest temporary directory.

        Returns:
            str: Path to the created raster.
        """
        path = str(tmp_path / "wgs84.tif")
        dataset = gdal.GetDriverByName("GTiff").Create(path, 8, 8, 1, gdal.GDT_Byte)
        dataset.SetGeoTransform([0, 1, 0, 8, 0, -1])
        dataset.SetProjection(sr_from_epsg(4326).ExportToWkt())
        dataset.GetRasterBand(1).WriteArray(np.ones((8, 8), dtype="uint8"))
        dataset = None
        return path

    def test_point_without_a_crs_argument(self, wgs84_raster: str):
        """`point(x, y)` works on a georeferenced raster with no `point_crs`.

        Test scenario:
            Defaulting `point_crs` to None must mean "already in the raster's
            coordinates", not "transform from None" — which raised on every
            georeferenced raster.
        """
        dataset = Dataset.read_file(wgs84_raster)
        assert dataset.point(2, 2) is not None, (
            "point() must work without an explicit CRS"
        )

    def test_read_part_without_a_crs_argument(self, wgs84_raster: str):
        """`read_part(bbox)` works on a georeferenced raster with no `bbox_crs`."""
        dataset = Dataset.read_file(wgs84_raster)
        assert dataset.read_part((1, 1, 4, 4)) is not None, (
            "read_part() must work without an explicit CRS"
        )

    def test_explicit_matching_crs_still_works(self, wgs84_raster: str):
        """Passing the raster's own CRS explicitly is still a no-op transform."""
        dataset = Dataset.read_file(wgs84_raster)
        assert dataset.point(2, 2, point_crs=4326) is not None, (
            "an explicit matching CRS must still work"
        )


def _tagged_raster(path: str, metadata: dict, geotransform: list) -> Dataset:
    """Build a projection-less GeoTIFF carrying CF-style `<var>#units` metadata.

    Args:
        path: Output path for the raster.
        metadata: GDAL metadata items, e.g. `{"lon#units": "degrees_east"}`.
        geotransform: Six-element GDAL geotransform.

    Returns:
        Dataset: The raster, read back through pyramids.
    """
    raster = gdal.GetDriverByName("GTiff").Create(path, 4, 4, 1, gdal.GDT_Float32)
    raster.SetGeoTransform(geotransform)
    raster.SetMetadata(metadata)
    raster = None
    return Dataset.read_file(path)


_UTM_GEOTRANSFORM = [400000.0, 30.0, 0.0, 5000000.0, 0.0, -30.0]
_GEOGRAPHIC_GEOTRANSFORM = [-10.0, 0.5, 0.0, 55.0, 0.0, -0.5]


class TestCfInferenceEvidence:
    """Tests for which metadata may and may not imply a geographic CRS.

    Each case is a raster the old default read as EPSG:4326; the inference must
    be narrower than "some variable somewhere is in degrees".
    """

    def test_a_data_variable_in_degrees_is_not_evidence(self, tmp_path):
        """Degrees on a *data* variable do not make a UTM grid geographic.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            A wind direction in `degrees_east` and a solar angle in `degreeN`
            on a metre-scale grid - neither describes the horizontal frame.
        """
        dataset = _tagged_raster(
            str(tmp_path / "wind.tif"),
            {"wind_from_direction#units": "degrees_east", "sun_elev#units": "degreeN"},
            _UTM_GEOTRANSFORM,
        )
        assert dataset.epsg is None, "a data variable's units must not define the CRS"

    def test_metre_axes_outside_the_name_list_still_veto(self, tmp_path):
        """A projected grid is not relabelled because its axes are named oddly.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            Metre axes named `xdim2` / `ydim2` (on no conventional-name list)
            beside auxiliary `lon` / `lat` in degrees.
        """
        dataset = _tagged_raster(
            str(tmp_path / "xdim2.tif"),
            {
                "xdim2#units": "m",
                "ydim2#units": "m",
                "lon#units": "degrees_east",
                "lat#units": "degrees_north",
            },
            _UTM_GEOTRANSFORM,
        )
        assert dataset.epsg is None, "a metre-scale grid must not be read as WGS 84"

    def test_a_real_cf_geographic_grid_is_still_wgs84(self, tmp_path):
        """The inference the change is *for* still fires.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            Degrees axes on a degree-scale grid, with no `grid_mapping`.
        """
        dataset = _tagged_raster(
            str(tmp_path / "geo.tif"),
            {"lon#units": "degrees_east", "lat#units": "degrees_north"},
            _GEOGRAPHIC_GEOTRANSFORM,
        )
        assert dataset.epsg == 4326, "a CF degrees grid must still read as WGS 84"

    def test_curvilinear_auxiliary_coordinates_are_evidence(self, tmp_path):
        """2-D lat/lon named in a `coordinates` attribute count as coordinates.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            A NEMO-style grid whose lat/lon are auxiliary coordinates - they
            declare no `axis`, so only the `coordinates` reference identifies
            them.
        """
        dataset = _tagged_raster(
            str(tmp_path / "nemo.tif"),
            {
                "sst#coordinates": "nav_lon nav_lat",
                "nav_lon#units": "degrees_east",
                "nav_lat#units": "degrees_north",
            },
            _GEOGRAPHIC_GEOTRANSFORM,
        )
        assert dataset.epsg == 4326, "auxiliary lat/lon coordinates are evidence"

    @pytest.mark.parametrize(
        "vertical_name", ["deptht", "olevel", "nav_lev", "zlev", "sigma"]
    )
    def test_a_declared_vertical_axis_in_metres_does_not_veto(
        self, tmp_path, vertical_name: str
    ):
        """`axis: Z` in metres says nothing about the horizontal CRS.

        Args:
            tmp_path: pytest temporary directory.
            vertical_name: Model-specific vertical-axis name under test.

        Test scenario:
            Vertical-axis names that no allow-list covers, each declaring
            `axis = "Z"` with metre units, on a degrees grid.
        """
        dataset = _tagged_raster(
            str(tmp_path / f"{vertical_name}.tif"),
            {
                "lon#units": "degrees_east",
                "lon#axis": "X",
                "lat#units": "degrees_north",
                "lat#axis": "Y",
                f"{vertical_name}#units": "m",
                f"{vertical_name}#axis": "Z",
            },
            _GEOGRAPHIC_GEOTRANSFORM,
        )
        assert dataset.epsg == 4326, f"{vertical_name} must not strip the CRS"

    def test_a_declared_horizontal_axis_in_metres_does_veto(self, tmp_path):
        """`axis: X` in metres is exactly the counter-evidence that must veto.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            Projected axes declaring their role, beside auxiliary degrees.
        """
        dataset = _tagged_raster(
            str(tmp_path / "projected.tif"),
            {
                "x#units": "m",
                "x#axis": "X",
                "y#units": "m",
                "y#axis": "Y",
                "lon#units": "degrees_east",
                "lat#units": "degrees_north",
            },
            _GEOGRAPHIC_GEOTRANSFORM,
        )
        assert dataset.epsg is None, "metre axes must veto the geographic reading"


class TestCrsEquality:
    """Tests for `crs_equal`, the comparison `align` uses to decide on a warp."""

    def test_two_spellings_of_one_crs_are_equal(self):
        """WKT1 and WKT2 of the same CRS must not look like different systems.

        Test scenario:
            `crs_spec` falls back to raw WKT for a CRS with no EPSG authority,
            so a raw `!=` there warps data through an identity transform.
        """
        spatial_ref = sr_from_epsg(32636)
        wkt1 = spatial_ref.ExportToWkt()
        wkt2 = spatial_ref.ExportToWkt(["FORMAT=WKT2"])
        assert wkt1 != wkt2, "fixture must use two genuinely different spellings"
        assert crs_equal(wkt1, wkt2), "one CRS in two spellings must compare equal"

    def test_different_systems_are_not_equal(self):
        """Distinct CRSes stay distinct."""
        assert not crs_equal(4326, 32636), "different systems must not compare equal"

    def test_absence_matches_only_absence(self):
        """Two CRS-less rasters match; one CRS-less and one not do not."""
        assert crs_equal(None, None), "two absent CRSes describe the same (nothing)"
        assert not crs_equal(None, 4326), "absent and present must not match"

    def test_align_does_not_warp_between_two_spellings(self, tmp_path, monkeypatch):
        """`align` skips the warp when both grids are really the same CRS.

        Args:
            tmp_path: pytest temporary directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            Two co-registered rasters whose WKT differs only in spelling. A raw
            `!=` sends this through a full `gdal.Warp`.
        """
        spatial_ref = sr_from_epsg(32636)
        paths = []
        spellings = [
            spatial_ref.ExportToWkt(),
            spatial_ref.ExportToWkt(["FORMAT=WKT2"]),
        ]
        for index, wkt in enumerate(spellings):
            path = str(tmp_path / f"grid{index}.tif")
            raster = gdal.GetDriverByName("GTiff").Create(path, 4, 4, 1, gdal.GDT_Byte)
            raster.SetGeoTransform(_UTM_GEOTRANSFORM)
            raster.SetProjection(wkt)
            raster.GetRasterBand(1).WriteArray(np.ones((4, 4), dtype="uint8"))
            raster = None
            paths.append(path)

        source = Dataset.read_file(paths[0])
        reference = Dataset.read_file(paths[1])
        calls = []
        original = Dataset.to_crs

        def _spy(self, *args, **kwargs):
            """Record the call, then delegate."""
            calls.append(args)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Dataset, "to_crs", _spy)
        aligned = source.align(reference)
        assert not calls, "align warped between two spellings of one CRS"
        assert aligned is not None, "align must still return a dataset"


class TestNetCDFGlobalAttributeProvenance:
    """Tests for which NetCDF global attributes may define the dataset's CRS."""

    @pytest.mark.xarray
    def test_our_own_writer_round_trips(self, tmp_path):
        """A cube written by `to_netcdf` reads its CRS back.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            The geobox writer emits `epsg` / `crs_wkt` beside `GeoTransform`;
            that pair is what the reader adopts.
        """
        paths = []
        for index in range(2):
            path = str(tmp_path / f"t{index}.tif")
            Dataset.create_from_array(
                np.arange(20, dtype="int16").reshape(4, 5),
                top_left_corner=(400000, 5000000),
                cell_size=30,
                epsg=32636,
                path=path,
            ).close()
            paths.append(path)
        out = str(tmp_path / "cube.nc")
        DatasetCollection.from_files(paths).to_netcdf(out)
        assert NetCDF.read_file(out).epsg == 32636, "our own file must round-trip"

    def test_a_stray_foreign_epsg_attribute_is_not_adopted(self, tmp_path):
        """A root `epsg` attribute with no geobox companion defines no CRS.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            A third-party store tagging itself `epsg = 32636` without the
            `GeoTransform` the geobox writer always emits alongside.
        """
        path = str(tmp_path / "foreign.nc")
        raster = gdal.GetDriverByName("netCDF").Create(path, 4, 4, 1, gdal.GDT_Float32)
        raster.SetMetadataItem("epsg", "32636")
        raster = None
        assert NetCDF.read_file(path).epsg is None, "a stray attribute is not a CRS"
