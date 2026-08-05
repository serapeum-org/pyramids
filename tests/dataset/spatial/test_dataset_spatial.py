"""Integration tests for Dataset spatial operations: resample, reproject, align,
crop, cluster, overlay, and mask."""

import gc
import weakref
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, osr
from pyproj import CRS as PyprojCRS
from shapely.geometry import Polygon, box

from pyramids.dataset import Dataset
from pyramids.dataset.engines.spatial import Spatial
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


class TestResample:
    def test_single_band(
        self,
        src: gdal.Dataset,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        resample_raster_result_dims: tuple,
    ):
        src = Dataset(src)
        dst = src.resample(
            resample_raster_cell_size,
            method=resample_raster_resample_technique,
        )

        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == resample_raster_result_dims
        assert (
            dst.raster.GetGeoTransform()[1] == resample_raster_cell_size
            and dst.raster.GetGeoTransform()[-1] == -1 * resample_raster_cell_size
        )
        assert np.isclose(
            dst.raster.GetRasterBand(1).GetNoDataValue(),
            src.raster.GetRasterBand(1).GetNoDataValue(),
            rtol=0.00001,
        )
        assert dst.raster.GetProjection() == src.raster.GetProjection()

    def test_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        resampled_multi_band_dims: tuple,
        sentinel_resample_arr: np.ndarray,
    ):
        cell_size = 0.00015
        src = Dataset(sentinel_raster)
        dst = src.resample(
            cell_size,
            method=resample_raster_resample_technique,
        )

        dst_arr = dst.raster.ReadAsArray()
        assert dst.rows == resampled_multi_band_dims[0]
        assert dst.columns == resampled_multi_band_dims[1]
        assert dst.raster.GetGeoTransform()[1] == pytest.approx(
            cell_size
        ) and dst.raster.GetGeoTransform()[-1] == pytest.approx(-1 * cell_size)

        # GDAL bilinear output drifts across minor versions (e.g. 3.12 -> 3.13
        # shifts up to ~5% of pixels by 100+ counts; max diff observed ~986 on
        # uint16 reflectance). max() == min() avoids uint16 underflow on the
        # subtraction. Compare the mean drift rather than per-pixel: a real
        # resampler regression (wrong kernel, swapped axis, off-by-one
        # geotransform) shifts the whole array and pushes mean past 50; a
        # GDAL kernel tweak only shifts a small fraction and keeps mean ~18.
        diff = np.maximum(sentinel_resample_arr, dst_arr) - np.minimum(
            sentinel_resample_arr, dst_arr
        )
        assert diff.mean() < 50
        assert dst.raster.GetProjection() == src.raster.GetProjection()


class TestReproject:
    def test_option_maintain_alignment_single_band(
        self,
        src: gdal.Dataset,
        project_raster_to_epsg: int,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        src_shape: tuple,
    ):
        src = Dataset(src)
        dst = src.to_crs(to_epsg=project_raster_to_epsg, maintain_alignment=True)

        proj = dst.raster.GetProjection()
        sr = osr.SpatialReference(wkt=proj)
        epsg = int(sr.GetAttrValue("AUTHORITY", 1))
        assert epsg == project_raster_to_epsg
        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == src_shape

    def test_option_maintain_alignment_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
    ):
        epsg = 32637
        src = Dataset(sentinel_raster)
        dst = src.to_crs(to_epsg=epsg, maintain_alignment=True)
        assert dst.band_count == src.band_count
        assert dst.epsg == epsg

    def test_option_donot_maintain_alignment(
        self,
        src: gdal.Dataset,
        project_raster_to_epsg: int,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        src_shape: tuple,
    ):
        src = Dataset(src)
        dst = src.to_crs(to_epsg=project_raster_to_epsg, maintain_alignment=False)

        proj = dst.crs
        sr = osr.SpatialReference(wkt=proj)
        epsg = int(sr.GetAttrValue("AUTHORITY", 1))
        assert epsg == project_raster_to_epsg
        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == src_shape

    def test_option_do_not_maintain_alignment_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
    ):
        epsg = 32637
        src = Dataset(sentinel_raster)
        dst = src.to_crs(to_epsg=epsg, maintain_alignment=False)
        assert dst.band_count == src.band_count
        assert dst.epsg == epsg

    def test_robinson_esri_authority_string(
        self,
        src: gdal.Dataset,
    ):
        """to_crs accepts an ESRI authority string for a non-EPSG CRS (#418).

        Test scenario:
            Robinson (ESRI:54030) has no EPSG code. It used to raise
            ``CRSError: ... pass an EPSG integer``; the new code path
            warps directly against the spatial reference, so the result
            is a projected raster whose root CRS name identifies Robinson.
        """
        src_ds = Dataset(src)
        dst = src_ds.to_crs(to_epsg="ESRI:54030")
        dst_sr = osr.SpatialReference(wkt=dst.crs)
        assert dst_sr.IsProjected() == 1
        assert "Robinson" in dst_sr.GetName(), (
            f"expected Robinson in dst CRS name, got {dst_sr.GetName()!r}"
        )

    def test_mollweide_esri_authority_string_maintain_alignment(
        self,
        src: gdal.Dataset,
        src_shape: tuple,
    ):
        """maintain_alignment=True works for the non-EPSG ESRI:54009 Mollweide CRS (#418).

        Test scenario:
            The alignment-preserving path used to call ``sr_from_epsg`` directly
            on the target, which can't resolve ESRI codes. The SRS-based rewrite
            keeps the same row/column count and reaches the Mollweide projection.
        """
        src_ds = Dataset(src)
        dst = src_ds.to_crs(to_epsg="ESRI:54009", maintain_alignment=True)
        dst_sr = osr.SpatialReference(wkt=dst.crs)
        assert dst_sr.IsProjected() == 1
        assert "Mollweide" in dst_sr.GetName(), (
            f"expected Mollweide in dst CRS name, got {dst_sr.GetName()!r}"
        )
        dst_shape = dst.raster.ReadAsArray().shape
        # Mollweide projects so differently from the source CRS that the
        # corner-sampled cell-step calculation can drift by a single
        # column/row vs. src_shape. The point of maintain_alignment=True
        # is "reuse the source cell step", not pixel-exact shape — a
        # tolerance of 1 covers the rounding noise without masking a
        # regression that would shift the whole footprint.
        assert abs(dst_shape[0] - src_shape[0]) <= 1
        assert abs(dst_shape[1] - src_shape[1]) <= 1

    def test_orthographic_proj4(
        self,
        src: gdal.Dataset,
    ):
        """to_crs accepts a proj4 orthographic string with no authority code (#418).

        Test scenario:
            A bespoke orthographic centred at (lat 39, lon -9) — Lisbon — has neither
            an EPSG nor an ESRI code. The result is a projected raster with positive
            row/column counts; bands carry the source's nodata so off-disc cells are
            indistinguishable from masked.
        """
        proj4 = "+proj=ortho +lat_0=39 +lon_0=-9 +datum=WGS84 +units=m +no_defs"
        src_ds = Dataset(src)
        dst = src_ds.to_crs(to_epsg=proj4)
        dst_sr = osr.SpatialReference(wkt=dst.crs)
        assert dst_sr.IsProjected() == 1
        assert dst.rows > 0 and dst.columns > 0
        assert dst.no_data_value[0] == src_ds.no_data_value[0]

    def test_pyproj_crs_object(
        self,
        src: gdal.Dataset,
    ):
        """to_crs accepts a pyproj.CRS instance for back-compat with the EPSG path (#418).

        Test scenario:
            Passing CRS.from_epsg(3857) yields the same dst.epsg as passing 3857
            directly — the new SRS-based front door must not break the EPSG fast path.
        """
        src_ds = Dataset(src)
        dst = src_ds.to_crs(to_epsg=PyprojCRS.from_epsg(3857))
        assert dst.epsg == 3857

    def test_orthographic_off_disc_cells_are_nodata(self):
        """Orthographic warp fills off-disc cells with the source nodata (#418 DoD).

        Test scenario:
            Project a globe-spanning 1° 4326 raster (covering -180..180, -90..90)
            into a polar-orthographic centred at the North Pole. The visible disc
            in the projected output is the northern hemisphere; the corner pixels
            sit beyond the projection's domain and must come back as nodata, not
            as data values. This exercises the real off-disc masking gdal.Warp
            performs — preserving nodata metadata alone (already covered) is not
            enough to verify the DoD.
        """
        arr = np.ones((180, 360), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(-180.0, 90.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        dst = ds.to_crs(
            to_epsg="+proj=ortho +lat_0=90 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        )
        out = dst.read_array()
        nodata = dst.no_data_value[0]
        corners = np.array([out[0, 0], out[0, -1], out[-1, 0], out[-1, -1]])
        assert np.allclose(corners, nodata), (
            f"off-disc corner pixels should equal nodata={nodata}, got {corners}"
        )

    def test_to_crs_non_epsg_with_bilinear_resampling(self):
        """to_crs runs the bilinear resampling path against a non-EPSG target (#418).

        Test scenario:
            All existing non-EPSG tests use the default nearest-neighbour interpolation;
            this one exercises the bilinear branch through the INTERPOLATION_METHODS
            lookup to confirm that the SRS-based front door doesn't break method
            dispatch. Asserts the output has a Mollweide projection and a non-empty
            array (bilinear must produce finite values on the visible domain).
        """
        arr = np.linspace(0.0, 1.0, num=20 * 20, dtype=np.float32).reshape(20, 20)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 10.0),
            cell_size=0.5,
            epsg=4326,
            no_data_value=-9999.0,
        )
        dst = ds.to_crs(to_epsg="ESRI:54009", method="bilinear")
        dst_sr = osr.SpatialReference(wkt=dst.crs)
        assert "Mollweide" in dst_sr.GetName(), (
            f"expected Mollweide in dst CRS name, got {dst_sr.GetName()!r}"
        )
        out = dst.read_array()
        finite = out[out != dst.no_data_value[0]]
        assert finite.size > 0, "bilinear warp should produce at least one finite cell"

    def test_to_crs_non_epsg_idempotent_shape(self):
        """Reprojecting twice to the same non-EPSG CRS yields the same shape (#418).

        Test scenario:
            Project a 4326 raster into Robinson (ESRI:54030), then project the result
            into Robinson again. The second projection is effectively an identity in
            the projected CRS; output shape and bbox must be stable. Catches the
            "non-EPSG warp returns a slightly different raster each call" class of
            regression that would surface if the SRS were rebuilt with a drifting
            authority/WKT each call.
        """
        arr = np.ones((180, 360), dtype=np.float32)
        src_ds = Dataset.create_from_array(
            arr,
            top_left_corner=(-180.0, 90.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        first = src_ds.to_crs(to_epsg="ESRI:54030")
        second = first.to_crs(to_epsg="ESRI:54030")
        assert (first.rows, first.columns) == (second.rows, second.columns), (
            f"shapes drift: first={first.rows}x{first.columns}, "
            f"second={second.rows}x{second.columns}"
        )
        for a, b in zip(first.geotransform, second.geotransform):
            assert abs(a - b) < 1e-6, (
                f"geotransform drift between projections: {first.geotransform} vs "
                f"{second.geotransform}"
            )

    def test_to_crs_multiband_to_esri_string(self):
        """Multi-band rasters reproject to an ESRI authority string (#418).

        Test scenario:
            All existing ESRI:54030 tests use a single-band fixture; this one
            confirms that the band-count is preserved when warping a 3-band raster
            into Robinson. A regression here would mean the new dst_srs_arg branch
            interacts badly with gdal.Warp's multi-band handling.
        """
        arr = np.stack([np.full((10, 10), v, dtype=np.float32) for v in (1, 2, 3)])
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 10.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        dst = ds.to_crs(to_epsg="ESRI:54030")
        assert dst.band_count == ds.band_count, (
            f"band count drift: src={ds.band_count}, dst={dst.band_count}"
        )

    def test_to_crs_same_epsg_maintain_alignment_is_identity(self):
        """to_crs(source_epsg, maintain_alignment=True) returns a bit-identical raster (M1).

        Test scenario:
            With the M1 fix (semantic IsSame() identity check after axis-order
            normalisation), passing the source's own EPSG into the alignment-
            preserving path must hit the shortcut and emit a Dataset whose
            geotransform and shape match the source exactly. A regression here
            would mean the shortcut never fires and every identity reprojection
            silently round-trips through reproject_coordinates.
        """
        arr = np.ones((5, 5), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(10.0, 50.0),
            cell_size=0.5,
            epsg=4326,
            no_data_value=-9999.0,
        )
        dst = ds.to_crs(to_epsg=4326, maintain_alignment=True)
        assert dst.geotransform == ds.geotransform
        assert dst.rows == ds.rows
        assert dst.columns == ds.columns

    def test_to_crs_accepts_wkt_string(self):
        """to_crs accepts a raw WKT string end-to-end (#418).

        Test scenario:
            `sr_from_user_input` is independently tested with WKT input, but the full
            `Spatial.to_crs(to_epsg=wkt)` path was not exercised — only EPSG ints /
            authority strings / proj4 / pyproj.CRS. Exporting EPSG:32636 to WKT and
            passing the bytes back into `to_crs` must reproject to the same UTM zone,
            confirming that the SRS-based front door survives a full WKT round-trip.
        """
        arr = np.ones((5, 5), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(30.0, 30.0),
            cell_size=0.1,
            epsg=4326,
            no_data_value=-9999.0,
        )
        wkt = PyprojCRS.from_epsg(32636).to_wkt()
        dst = ds.to_crs(to_epsg=wkt)
        assert dst.epsg == 32636, f"WKT input should resolve to 32636, got {dst.epsg}"

    def test_to_crs_projected_identity_maintain_alignment(self):
        """Identity shortcut fires for a projected source (UTM → same UTM).

        Test scenario:
            The existing identity test uses 4326 → 4326 — a geographic CRS where
            axis-order normalisation matters. UTM (32636) is projected with
            unambiguous easting/northing axis order regardless of mapping strategy,
            which exercises a structurally different IsSame() match. Asserts the
            geotransform and shape are preserved, confirming the IsSame shortcut
            also works for projected CRSes (not only for the geographic edge case
            that motivated the axis normalisation).
        """
        arr = np.ones((5, 5), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            geo=(500_000.0, 100.0, 0.0, 5_500_000.0, 0.0, -100.0),
            epsg=32636,
            no_data_value=-9999.0,
        )
        dst = ds.to_crs(to_epsg=32636, maintain_alignment=True)
        assert dst.geotransform == ds.geotransform, (
            f"projected identity shortcut should preserve geotransform; "
            f"got {dst.geotransform} vs source {ds.geotransform}"
        )
        assert (dst.rows, dst.columns) == (ds.rows, ds.columns), (
            f"projected identity shortcut should preserve shape; "
            f"got {(dst.rows, dst.columns)} vs source {(ds.rows, ds.columns)}"
        )

    def test_to_crs_lng_gt_180_maintain_alignment_shifts_west(self):
        """maintain_alignment=True shifts a >180 longitude origin into the western hemisphere.

        Test scenario:
            A geographic source with top_left_corner=(200.0, 50.0) sits in the
            0..360 longitude convention. Routing through
            _reproject_with_ReprojectImage (maintain_alignment=True) reaches the
            west-edge branch that subtracts 360 from the origin before corner
            reprojection, so the output extent on Web Mercator lands in the
            *western* hemisphere (negative metres). A regression in that branch
            would either crash, produce an extent off the visible Web Mercator
            range (~+22e6 m), or silently fall through with an unshifted origin.
            Pre-#418 this branch was dead code (legacy `src_epsg == "4326"`
            int-vs-str guard); the new IsGeographic() guard makes it reachable
            but previously had no test coverage.
        """
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(200.0, 50.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        dst = ds.to_crs(to_epsg=3857, maintain_alignment=True)
        assert dst.epsg == 3857
        assert dst.geotransform[0] < 0, (
            "west-edge shift should land the origin in the western hemisphere "
            f"(negative metres on Web Mercator); got geotransform[0]={dst.geotransform[0]}"
        )


class TestAlign:
    def test_align_single_band(
        self,
        src: gdal.Dataset,
        src_shape: tuple,
        # src_no_data_value: float,
        src_geotransform: tuple,
        soil_raster: gdal.Dataset,
    ):
        mask_obj = Dataset(src)
        dataset = Dataset(soil_raster)
        dataset_aligned = dataset.align(mask_obj)
        assert dataset_aligned.raster.ReadAsArray().shape == src_shape
        nodataval = dataset_aligned.raster.GetRasterBand(1).GetNoDataValue()
        src_no_data_value = dataset.no_data_value[0]
        assert np.isclose(nodataval, src_no_data_value, rtol=0.000001)
        geotransform = dataset_aligned.raster.GetGeoTransform()
        assert src_geotransform == geotransform

    def test_align_multi_band(
        self,
        resampled_multiband: gdal.Dataset,
        sentinel_raster: gdal.Dataset,
        resampled_multi_band_dims: tuple,
        src_geotransform: tuple,
    ):
        alignment_src = Dataset(resampled_multiband)
        dataset = Dataset(sentinel_raster)
        dataset_aligned = dataset.align(alignment_src)
        assert dataset_aligned.rows == resampled_multi_band_dims[0]
        assert dataset_aligned.columns == resampled_multi_band_dims[1]
        assert dataset.top_left_corner == dataset_aligned.top_left_corner


class TestCrop:
    def test_crop_single_band_dataset_with_single_band_mask(
        self,
        src: gdal.Dataset,
        aligned_raster,
        src_arr: np.ndarray,
        src_no_data_value: float,
    ):
        mask_obj = Dataset(src)
        aligned_raster: Dataset = Dataset(aligned_raster)
        cropped: Dataset = aligned_raster.spatial._crop_aligned(mask_obj)
        dst_arr_cropped = cropped.raster.ReadAsArray()
        # check that all the places of the nodatavalue are the same in both arrays
        src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
        dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
        assert (dst_arr_cropped == src_arr).all()

    def test_crop_multi_band_dataset_with_single_band_mask(
        self,
        sentinel_raster: gdal.Dataset,
        sentinel_crop,
        sentinel_crop_arr_without_no_data_value: np.ndarray,
    ):
        mask_obj = Dataset(sentinel_crop)
        aligned_raster = Dataset(sentinel_raster)

        cropped: Dataset = aligned_raster.spatial._crop_aligned(mask_obj)
        dst_arr_cropped = cropped.raster.ReadAsArray()
        # filter the no_data_value out of the array
        arr = dst_arr_cropped[
            ~np.isclose(dst_arr_cropped, cropped.no_data_value[0], rtol=0.001)
        ]
        assert np.array_equal(sentinel_crop_arr_without_no_data_value, arr)

    def test_crop_multi_band_dataset_with_multi_band_mask(self):
        # the dataset has 4 bands
        arr = np.random.default_rng(0).random((4, 6, 5))
        geotransform = (0, 0.05, 0, 0, 0, -0.05)
        dataset = Dataset.create_from_array(arr, geo=geotransform, epsg=4326)
        # the mask has 3 bands
        arr_mask = np.random.default_rng(0).random((3, 2, 2))
        geotransform = (0.1, 0.05, 0.0, -0.1, 0.0, -0.05)
        mask = Dataset.create_from_array(arr_mask, geo=geotransform, epsg=4326)
        cropped_dataset = dataset.crop(mask=mask)

        assert cropped_dataset.shape == (4, 2, 2)
        np.testing.assert_array_equal(arr[:, 2:4, 2:4], cropped_dataset.read_array())

    def test_crop_dataset_with_array(
        self,
        aligned_raster,
        src_arr: np.ndarray,
        src_no_data_value: float,
    ):
        aligned_raster = Dataset(aligned_raster)
        cropped = aligned_raster.spatial._crop_aligned(
            src_arr, mask_noval=src_no_data_value
        )
        dst_arr_cropped = cropped.raster.ReadAsArray()
        # check that all the places of the nodatavalue are the same in both arrays
        src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
        dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
        assert (dst_arr_cropped == src_arr).all()

    def test_crop_un_aligned(
        self,
        soil_raster: gdal.Dataset,
        aligned_raster: gdal.Dataset,
    ):
        # the soil raster has epsg=2116 and
        # Geotransform = (830606.744300001, 30.0, 0.0, 1011325.7178760837, 0.0, -30.0)
        # the aligned_raster has an epsg = 32618 and
        # Geotransform = (432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0)
        mask_obj = Dataset(soil_raster)
        aligned_raster = Dataset(aligned_raster)
        aligned_raster.spatial._crop_with_raster(mask_obj)


class TestCropWithPolygon:
    def test_inplace(
        self,
        rhine_raster: gdal.Dataset,
        polygon_mask: GeoDataFrame,
        crop_by_wrap_touch_true_result: gdal.Dataset,
    ):
        """
        Check that the inplace option is working
        """
        dataset = Dataset(rhine_raster)
        cells = dataset.count_domain_cells()
        dataset = dataset.crop(polygon_mask, touch=True)
        new_cells = dataset.count_domain_cells()
        assert cells != new_cells

    def test_by_warp_touch_true_single_band(
        self,
        rhine_raster: gdal.Dataset,
        polygon_mask: GeoDataFrame,
        crop_by_wrap_touch_true_result: gdal.Dataset,
    ):
        """
        when the touch option is True in the function, the cells that touches the mask polygon but does not lie
        entirely inside the mask will be included

        Check the number of the cropped cells and the no_data_value
        """
        src_obj = Dataset(rhine_raster)
        cropped_raster = src_obj.spatial._crop_with_polygon_warp(
            polygon_mask, touch=True
        )

        validation_dataset = Dataset(crop_by_wrap_touch_true_result)
        assert (
            validation_dataset.count_domain_cells()
            == cropped_raster.count_domain_cells()
        )
        assert isinstance(cropped_raster.raster, gdal.Dataset)
        assert cropped_raster.no_data_value[0] == src_obj.no_data_value[0]

    def test_by_warp_touch_true_multi_band(self):
        """Test that the function works with multi-band raster."""
        arr = np.random.default_rng(0).random((4, 6, 5))
        geotransform = (0, 0.05, 0, 0, 0, -0.05)
        dataset = Dataset.create_from_array(arr, geo=geotransform, epsg=4326)
        mask = gpd.GeoDataFrame(
            geometry=[Polygon([(0.1, -0.1), (0.1, -0.2), (0.2, -0.2), (0.2, -0.1)])],
            crs=4326,
        )
        cropped_dataset = dataset.crop(mask=mask, touch=True)
        arr = cropped_dataset.read_array()
        assert arr.shape == (4, 2, 2)

    def test_by_warp_touch_false(
        self,
        rhine_raster: gdal.Dataset,
        polygon_mask: GeoDataFrame,
        crop_by_wrap_touch_false_result: gdal.Dataset,
    ):
        """
        when the touch option is False in the function, only the cells that lie entirely inside the mask will be
        included

        Check the number of the cropped cells and the no_data_value
        """
        src_obj = Dataset(rhine_raster)
        cropped_raster = src_obj.spatial._crop_with_polygon_warp(
            polygon_mask, touch=False
        )

        validation_dataset = Dataset(crop_by_wrap_touch_false_result)
        assert (
            validation_dataset.count_domain_cells()
            == cropped_raster.count_domain_cells()
        )
        assert isinstance(cropped_raster.raster, gdal.Dataset)
        assert cropped_raster.no_data_value[0] == src_obj.no_data_value[0]

    def test_by_warp_touch_multi_band(
        self,
        era5_image: gdal.Dataset,
        era5_mask: GeoDataFrame,
    ):
        """
        when the touch option is False in the function, only the cells that lie
        entirely inside the mask will be included

        Check the number of the cropped cells and the no_data_value
        """
        src_obj = Dataset(era5_image)

        cropped_raster = src_obj.spatial._crop_with_polygon_warp(era5_mask, touch=True)
        assert isinstance(cropped_raster.raster, gdal.Dataset)
        assert cropped_raster.no_data_value[0] == src_obj.no_data_value[0]
        assert cropped_raster.band_count == src_obj.band_count
        assert cropped_raster.shape == (9, 1, 2)
        arr = cropped_raster.read_array()
        vals = np.array(
            [
                [[2.70369720e02, 2.70399017e02]],
                [[2.69744751e02, 2.69651001e02]],
                [[2.73901245e02, 2.73889526e02]],
                [[2.74255188e02, 2.74235657e02]],
                [[2.75303284e02, 2.75260315e02]],
                [[3.67523193e-01, 3.67843628e-01]],
                [[3.72436523e-01, 3.73031616e-01]],
                [[3.85742188e-01, 3.90228271e-01]],
                [[1.88440349e-03, 1.81000944e-03]],
            ]
        )
        assert np.isclose(arr, vals, rtol=0.00001).all()

    def test_with_irregular_polygon(
        self,
        raster_1band_coello_gdal_dataset: Dataset,
        rasterized_mask_values: np.ndarray,
        coello_irregular_polygon_gdf: GeoDataFrame,
    ):
        """the input mask vector is given as geodataframe.

        Parameters
        ----------
        rasterized_mask_values: array for comparison
        """
        dataset = Dataset(raster_1band_coello_gdal_dataset)
        # test with irregular mask polygon
        cropped = dataset.spatial._crop_with_polygon_warp(
            coello_irregular_polygon_gdf, touch=False
        )

        assert isinstance(cropped, Dataset)
        arr = cropped.raster.ReadAsArray()
        values = arr[~np.isclose(arr, dataset.no_data_value[0], rtol=0.0001)]
        assert np.array_equal(values, rasterized_mask_values), (
            "the extracted values in the dataframe do not equal the real values in the array"
        )


class TestCluster2:
    """Test converting raster to polygon."""

    def test_single_band(
        self,
        test_image: gdal.Dataset,
    ):
        dataset = Dataset(test_image)
        gdf = dataset.to_polygons()
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == 4
        assert all(gdf.columns == ["GPP", "geometry"])
        assert all(gdf.geometry.geom_type == "Polygon")

    def test_multi_band_all_bands(
        self,
        sentinel_raster: gdal.Dataset,
    ):
        dataset = Dataset(sentinel_raster)
        gdf = dataset.to_polygons()
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == 1767
        assert all(
            elem in gdf.columns for elem in [dataset.band_names[0]] + ["geometry"]
        )
        assert all(gdf.geometry.geom_type == "Polygon")


class TestOverlay:
    def test_single_band(self, rhine_raster: gdal.Dataset, germany_classes: Path):
        src_obj = Dataset(rhine_raster)
        classes_src = Dataset.read_file(germany_classes)
        class_dict = src_obj.overlay(classes_src)
        arr = classes_src.read_array()
        class_values = np.unique(arr)
        assert len(class_dict.keys()) == len(class_values) - 1
        extracted_classes = list(class_dict.keys())
        real_classes = class_values.tolist()[:-1]
        assert all(i in real_classes for i in extracted_classes)

    def test_multi_band(
        self, sentinel_raster: gdal.Dataset, sentinel_classes: gdal.Dataset
    ):
        dataset = Dataset(sentinel_raster)
        classes_src = Dataset(sentinel_classes)
        class_dict = dataset.overlay(classes_src, band=1)
        arr = classes_src.read_array()
        class_values = np.unique(arr)
        assert len(class_dict.keys()) == len(class_values)
        extracted_classes = list(class_dict.keys())
        real_classes = class_values.tolist()
        assert all(i in real_classes for i in extracted_classes)


class TestMAsk:
    def test_get_mask(self, src: gdal.Dataset):
        dataset = Dataset(src)
        values = dataset.read_array()
        no_data_value = dataset.no_data_value[0]
        values[~np.isclose(values, no_data_value)] = 255
        values[np.isclose(values, no_data_value)] = 0
        arr = dataset.get_mask(band=0)
        np.testing.assert_equal(values, arr)
        vals = np.unique(arr)
        assert np.array_equal(vals, [0, 255])


class TestToCrsWarpSourceLifetime:
    """A `to_crs` VRT reads through to its source, so it must keep it alive."""

    @staticmethod
    def _memory_dataset() -> Dataset:
        """An 8x8 in-memory raster in EPSG:4326.

        Returns:
            Dataset: In-memory single-band dataset.
        """
        array = np.arange(64, dtype="float32").reshape(8, 8)
        raster = gdal.GetDriverByName("MEM").Create("", 8, 8, 1, gdal.GDT_Float32)
        raster.SetGeoTransform((0.0, 1.0, 0.0, 8.0, 0.0, -1.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        raster.SetProjection(srs.ExportToWkt())
        band = raster.GetRasterBand(1)
        band.WriteArray(array)
        band.SetNoDataValue(-9999.0)
        return Dataset(raster)

    def test_result_pins_its_warp_source(self) -> None:
        """`to_crs` records the source it warps from.

        Test scenario:
            The reprojected result is a warped VRT, not materialised pixels, so
            it dereferences the source on every read. `to_crs` pinned nothing at
            all, and the other three warp sites pinned the engine's
            `weakref.proxy` back-reference — which keeps nothing alive. The pin
            must be a strong reference to the source GDAL raster.
        """
        source = self._memory_dataset()
        reprojected = source.to_crs(3857)
        pinned = getattr(reprojected, "_warp_source", None)
        assert pinned is source.raster, (
            f"to_crs must pin the source GDAL raster, got {pinned!r}"
        )
        assert not isinstance(pinned, weakref.ProxyTypes), (
            "the pin must be a strong reference; a weakref.proxy keeps nothing alive"
        )

    def test_result_readable_after_the_source_goes_out_of_scope(self) -> None:
        """The result stays readable once the caller drops its own reference.

        Test scenario:
            Chained reprojection (`ds.to_crs(a).to_crs(b)`) is the shape that
            exposed this: the intermediate is unreferenced by the caller, so
            without the pin its backing store could be collected while the
            outer VRT still points at it.
        """

        reprojected = self._memory_dataset().to_crs(3857)
        gc.collect()
        values = np.asarray(reprojected.read_array())
        assert values.shape == (8, 8), f"expected an 8x8 read, got {values.shape}"
        assert np.isfinite(values).any(), "the reprojected read returned no finite data"

    def test_chained_reprojection_round_trips(self) -> None:
        """Two chained reprojections still produce a readable dataset."""

        chained = self._memory_dataset().to_crs(3857).to_crs(4326)
        gc.collect()
        values = np.asarray(chained.read_array())
        assert values.size > 0, "chained to_crs produced an empty read"
        assert np.isfinite(values).any(), "chained to_crs returned no finite data"


class TestCutlineBorderTrim:
    """ARC-20: the no-data frame left by a cutline warp is trimmed for any sentinel."""

    @staticmethod
    def _bordered(no_data: float) -> Dataset:
        """A 5x5 raster whose outer ring is entirely no-data.

        Args:
            no_data: The sentinel to write into the border and register on the
                band.

        Returns:
            Dataset: the bordered raster.
        """
        array = np.full((5, 5), no_data, dtype="float32")
        array[1:4, 1:4] = np.arange(9, dtype="float32").reshape(3, 3)
        return Dataset.create_from_array(
            array,
            top_left_corner=(0.0, 5.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=no_data,
        )

    def test_a_nan_border_is_trimmed(self):
        """A NaN sentinel is detected and its frame removed.

        Test scenario:
            The trim tested `array == value_to_remove`. NaN never equals itself,
            so on a NaN-sentinel raster nothing matched, no rows or columns were
            dropped, and the crop came back the original size with the no-data
            border still attached.
        """
        trimmed = Spatial._correct_wrap_cutline_error(self._bordered(np.nan))
        assert (trimmed.rows, trimmed.columns) == (3, 3), (
            f"the all-NaN frame must be trimmed to 3x3, got "
            f"{trimmed.rows}x{trimmed.columns}"
        )
        np.testing.assert_array_equal(
            np.asarray(trimmed.read_array()),
            np.arange(9, dtype="float32").reshape(3, 3),
        )

    def test_a_numeric_border_is_still_trimmed(self):
        """The ordinary sentinel path is unchanged.

        Test scenario:
            `is_no_data` replaced a plain `==`, so the numeric case has to keep
            behaving exactly as before.
        """
        trimmed = Spatial._correct_wrap_cutline_error(self._bordered(-9999.0))
        assert (trimmed.rows, trimmed.columns) == (3, 3), (
            f"the all-no-data frame must be trimmed to 3x3, got "
            f"{trimmed.rows}x{trimmed.columns}"
        )

    def test_the_trimmed_origin_moves_to_the_first_valid_cell(self):
        """Trimming shifts the geotransform, it does not just reshape the array.

        Test scenario:
            The border is one cell wide on a cell size of 1, so the top-left
            corner moves one cell right and one cell down.
        """
        trimmed = Spatial._correct_wrap_cutline_error(self._bordered(np.nan))
        assert (trimmed.geotransform[0], trimmed.geotransform[3]) == (1.0, 4.0), (
            f"expected the origin at (1.0, 4.0), got {trimmed.geotransform[:4]}"
        )


class TestCropBoundsTheReadToTheCrop:
    """#854: a touch=True polygon crop reads its output window, not the whole source."""

    @staticmethod
    def _large_source(no_data: float = -9999.0) -> Dataset:
        """A 400x400 raster over x[0,4] y[0,4] with a no-data value set."""
        array = np.arange(400 * 400, dtype="float32").reshape(400, 400)
        return Dataset.create_from_array(
            array,
            top_left_corner=(0.0, 4.0),
            cell_size=0.01,
            epsg=4326,
            no_data_value=no_data,
        )

    @staticmethod
    def _crs_less_source(no_data: float = -9999.0) -> Dataset:
        """The `_large_source` grid built via a raw MEM raster with no projection set."""
        array = np.arange(400 * 400, dtype="float32").reshape(400, 400)
        mem = gdal.GetDriverByName("MEM").Create("", 400, 400, 1, gdal.GDT_Float32)
        mem.SetGeoTransform((0.0, 0.01, 0.0, 4.0, 0.0, -0.01))
        band = mem.GetRasterBand(1)
        band.WriteArray(array)
        band.SetNoDataValue(no_data)
        return Dataset(mem)

    @staticmethod
    def _grid_aligned_mask() -> gpd.GeoDataFrame:
        """A 50x50-cell box whose edges sit on the source's cell boundaries."""
        return gpd.GeoDataFrame(geometry=[box(1.0, 3.0, 1.5, 3.5)], crs=4326)

    def test_touch_true_reads_the_crop_window_not_the_source(self, monkeypatch):
        """Peak read tracks the output, not the source — the #854 regression.

        Test scenario:
            A 50x50 window is cropped from a 400x400 source. Before the fix the
            trim read the full 160000-cell source back; now it reads ~the output.
        """
        dataset = self._large_source()
        peak = {"cells": 0}
        original = Dataset.read_array

        def spy(self, *args, **kwargs):
            arr = original(self, *args, **kwargs)
            peak["cells"] = max(peak["cells"], getattr(arr, "size", 0))
            return arr

        monkeypatch.setattr(Dataset, "read_array", spy)
        out = dataset.crop(mask=self._grid_aligned_mask(), touch=True)
        output_cells = out.shape[-1] * out.shape[-2]
        assert peak["cells"] <= 4 * output_cells, (
            f"the crop must read ~its output ({output_cells} cells), not the "
            f"400x400 source; read {peak['cells']}"
        )
        assert peak["cells"] < 160000, "the full source array must not be read"

    def test_bounded_touch_crop_equals_the_unbounded_reference(self):
        """touch=True (windowed warp + NumPy trim) matches touch=False (cropToCutline).

        Test scenario:
            The two paths share no code — touch=False crops to the cutline inside
            GDAL, touch=True windows the warp then trims in NumPy — so on a
            grid-aligned box, where "touch" and "inside" coincide, an identical
            result is strong evidence the window dropped or added no pixel.
        """
        dataset = self._large_source()
        mask = self._grid_aligned_mask()
        windowed = dataset.crop(mask=mask, touch=True)
        reference = dataset.crop(mask=mask, touch=False)
        assert windowed.shape == reference.shape, (
            f"shapes differ: {windowed.shape} vs {reference.shape}"
        )
        np.testing.assert_array_equal(
            np.asarray(windowed.read_array()), np.asarray(reference.read_array())
        )
        # The two paths reach the origin through different float arithmetic, so it
        # can differ by a ULP (~1e-16 deg); every practical tolerance is far coarser.
        np.testing.assert_allclose(
            windowed.geotransform, reference.geotransform, atol=1e-9
        )

    def test_window_covers_the_cutline_with_a_one_cell_margin(self):
        """The window contains the polygon envelope plus a cell of touch margin.

        Test scenario:
            The margin is what lets a grazing cell survive the trim, so the
            window must reach one cell beyond the envelope on every side and stay
            snapped to the source grid.
        """
        dataset = self._large_source()
        west, south, east, north = Spatial._cutline_window_bounds(
            dataset, self._grid_aligned_mask()
        )
        assert west <= 1.0 - 0.01 + 1e-9, f"the window must clear the west edge: {west}"
        assert east >= 1.5 + 0.01 - 1e-9, f"the window must clear the east edge: {east}"
        assert south <= 3.0 - 0.01 + 1e-9, f"the window must clear the south edge: {south}"
        assert north >= 3.5 + 0.01 - 1e-9, f"the window must clear the north edge: {north}"
        assert abs((west / 0.01) - round(west / 0.01)) < 1e-6, (
            "the window edges must stay snapped to the source grid"
        )

    def test_window_is_none_for_a_rotated_grid(self):
        """A sheared geotransform disables the optimisation and falls back safely.

        Test scenario:
            Rotated pixels are not axis-aligned, so an axis-aligned outputBounds
            cannot map onto them; the helper returns None and the caller warps
            the full source as before.
        """
        rotated = Dataset.create_from_array(
            np.zeros((10, 10), dtype="float32"),
            geo=(0.0, 1.0, 0.5, 10.0, 0.5, -1.0),
            epsg=4326,
        )
        assert (
            Spatial._cutline_window_bounds(rotated, self._grid_aligned_mask()) is None
        ), "a rotated grid must fall back to the full-source warp"

    def test_window_is_none_when_the_cutline_has_no_crs(self):
        """Without a cutline CRS the envelope cannot be projected; fall back."""
        dataset = self._large_source()
        crs_less = gpd.GeoDataFrame(geometry=[box(1.0, 3.0, 1.5, 3.5)], crs=None)
        assert Spatial._cutline_window_bounds(dataset, crs_less) is None, (
            "a CRS-less cutline must fall back rather than guess a projection"
        )

    def test_window_is_none_when_the_source_has_no_crs(self):
        """A source with no projection cannot host the reprojected envelope; fall back."""
        crs_less_source = self._crs_less_source()
        window = Spatial._cutline_window_bounds(
            crs_less_source, self._grid_aligned_mask()
        )
        assert window is None, (
            "a CRS-less source must fall back rather than assume the cutline's CRS"
        )

    def test_the_window_is_lossless_on_a_non_grid_aligned_polygon(self, monkeypatch):
        """A same-CRS polygon with mid-cell vertices crops identically windowed or full.

        Test scenario:
            Every other equivalence test uses a grid-aligned box, so the floor/ceil
            snapping and the one-cell touch margin are never exercised on an edge that
            falls between cell boundaries. This polygon's vertices sit mid-cell; the
            windowed touch=True crop must byte-match the same crop with the window forced
            off, proving the arithmetic loses no grazed cell.
        """
        mask = gpd.GeoDataFrame(
            geometry=[Polygon([(1.013, 3.027), (1.487, 2.964), (1.402, 2.511), (0.978, 2.603)])],
            crs=4326,
        )
        windowed = self._large_source().crop(mask=mask, touch=True)
        monkeypatch.setattr(
            Spatial, "_cutline_window_bounds", staticmethod(lambda src, feature: None)
        )
        full = self._large_source().crop(mask=mask, touch=True)
        assert windowed.shape == full.shape, (
            f"the window truncated a non-aligned polygon: {windowed.shape} vs {full.shape}"
        )
        np.testing.assert_array_equal(
            np.asarray(windowed.read_array()), np.asarray(full.read_array())
        )

    def test_a_feature_collection_gives_the_same_window_as_a_geodataframe(self):
        """The helper answers identically for the FeatureCollection production passes it.

        Test scenario:
            The unit tests exercise the helper with a raw GeoDataFrame, but crop() always
            hands it a FeatureCollection; confirm the two forms of the same mask yield the
            same window.
        """
        gdf = self._grid_aligned_mask()
        from_gdf = Spatial._cutline_window_bounds(self._large_source(), gdf)
        from_fc = Spatial._cutline_window_bounds(self._large_source(), FeatureCollection(gdf))
        assert from_gdf == from_fc, (
            f"FeatureCollection and GeoDataFrame must agree: {from_fc} vs {from_gdf}"
        )
        assert from_gdf is not None, "precondition: a same-CRS mask yields a window"

    def test_window_is_none_for_a_cutline_in_a_different_crs(self):
        """A cutline not already in the source CRS is not eligible for the window.

        Test scenario:
            geopandas reprojects by moving vertices while GDAL densifies the cutline,
            so a curving reprojection could under-cover GDAL's masked region; the helper
            declines and the crop falls back to the correct full-source warp.
        """
        dataset = self._large_source()
        reprojected = self._grid_aligned_mask().to_crs(32631)
        assert Spatial._cutline_window_bounds(dataset, reprojected) is None, (
            "a reprojected cutline must fall back rather than risk a truncated crop"
        )

    def test_window_is_none_for_a_south_up_grid(self):
        """A non-north-up geotransform is declined, not mis-snapped.

        Test scenario:
            The pixel math assumes dx>0 and dy<0; a south-up (dy>0) grid would invert
            the row snapping, so the helper must fall back rather than compute a wrong
            window.
        """
        south_up = Dataset.create_from_array(
            np.zeros((10, 10), dtype="float32"),
            geo=(0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            epsg=4326,
        )
        window = Spatial._cutline_window_bounds(south_up, self._grid_aligned_mask())
        assert window is None, "a south-up grid must fall back to the full-source warp"

    def test_a_different_crs_cutline_crops_through_the_full_source_fallback(self):
        """crop() completes for a cutline outside the source CRS, via the full warp.

        Test scenario:
            The helper declines a different-CRS cutline (it cannot bound a curving
            reprojection safely), so crop() takes the full-source path and must still
            return the region's real pixels, untruncated. The structural guarantee — the
            window is never built for a different CRS — is pinned by
            `test_window_is_none_for_a_cutline_in_a_different_crs`; here we confirm the
            fallback is wired and produces a sound crop end to end.
        """
        mask = self._grid_aligned_mask().to_crs(32631)
        cropped = self._large_source().crop(mask=mask, touch=True)
        assert 45 <= cropped.shape[-2] <= 55, (
            f"the cross-CRS crop's rows must cover the ~50-cell region: {cropped.shape}"
        )
        assert 45 <= cropped.shape[-1] <= 55, (
            f"the cross-CRS crop's cols must cover the ~50-cell region: {cropped.shape}"
        )
        assert np.isfinite(np.asarray(cropped.read_array())).any(), (
            "the fallback crop must contain real data"
        )

    def test_window_is_none_for_a_non_finite_envelope(self):
        """An empty cutline yields NaN bounds, so the helper falls back."""
        dataset = self._large_source()
        empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=4326))
        assert Spatial._cutline_window_bounds(dataset, empty) is None, (
            "a non-finite (NaN) envelope must fall back to the full-source warp"
        )

    def test_window_is_none_when_the_cutline_is_outside_the_source(self):
        """A cutline beyond the source extent clips to an empty window; fall back."""
        dataset = self._large_source()
        outside = gpd.GeoDataFrame(geometry=[box(10.0, 10.0, 11.0, 11.0)], crs=4326)
        assert Spatial._cutline_window_bounds(dataset, outside) is None, (
            "a cutline off the source must clip to empty and fall back"
        )

    def test_crop_falls_back_to_the_full_source_warp_when_the_window_is_none(self):
        """A CRS-less source (window=None) crops to the exact same pixels as the windowed path."""
        crs_less_mask = gpd.GeoDataFrame(geometry=[box(1.0, 3.0, 1.5, 3.5)], crs=None)
        reference = self._large_source().crop(mask=self._grid_aligned_mask(), touch=True)
        crs_less_source = self._crs_less_source()
        assert Spatial._cutline_window_bounds(crs_less_source, crs_less_mask) is None, (
            "a CRS-less source must trigger the full-source fallback"
        )
        fallback = crs_less_source.crop(mask=crs_less_mask, touch=True)
        assert fallback.shape == reference.shape, (
            f"fallback shape {fallback.shape} != reference {reference.shape}"
        )
        np.testing.assert_array_equal(
            np.asarray(fallback.read_array()), np.asarray(reference.read_array())
        )

    def test_bounded_touch_crop_is_grid_origin_independent(self):
        """The windowed crop matches the reference on a 0..360-longitude grid too."""
        array = np.arange(200 * 200, dtype="float32").reshape(200, 200)
        source = Dataset.create_from_array(
            array,
            top_left_corner=(200.0, 40.0),
            cell_size=0.1,
            epsg=4326,
            no_data_value=-9999.0,
        )
        mask = gpd.GeoDataFrame(geometry=[box(210.0, 22.0, 213.0, 25.0)], crs=4326)
        windowed = source.crop(mask=mask, touch=True)
        reference = source.crop(mask=mask, touch=False)
        assert windowed.shape == reference.shape, (
            f"shapes differ on a 0..360 grid: {windowed.shape} vs {reference.shape}"
        )
        np.testing.assert_array_equal(
            np.asarray(windowed.read_array()), np.asarray(reference.read_array())
        )

    def test_crop_with_polygon_warp_rejects_a_non_feature(self):
        """A mask that is neither GeoDataFrame nor FeatureCollection raises TypeError."""
        dataset = self._large_source()
        with pytest.raises(TypeError, match="FeatureCollection or GeoDataFrame"):
            dataset.spatial._crop_with_polygon_warp("not a feature", touch=True)
