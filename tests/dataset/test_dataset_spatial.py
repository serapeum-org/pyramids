"""Integration tests for Dataset spatial operations: resample, reproject, align,
crop, cluster, overlay, and mask."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, osr
from pyproj import CRS as PyprojCRS
from shapely.geometry import Polygon

from pyramids.dataset import Dataset
from pyramids.dataset.engines import Vectorize

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
        assert (
            dst.raster.GetGeoTransform()[1] == pytest.approx(cell_size)
            and dst.raster.GetGeoTransform()[-1] == pytest.approx(-1 * cell_size)
        )

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
        assert (
            "Robinson" in dst_sr.GetName()
        ), f"expected Robinson in dst CRS name, got {dst_sr.GetName()!r}"

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
        assert (
            "Mollweide" in dst_sr.GetName()
        ), f"expected Mollweide in dst CRS name, got {dst_sr.GetName()!r}"
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
        assert np.allclose(
            corners, nodata
        ), f"off-disc corner pixels should equal nodata={nodata}, got {corners}"

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
        assert (
            "Mollweide" in dst_sr.GetName()
        ), f"expected Mollweide in dst CRS name, got {dst_sr.GetName()!r}"
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
        assert (
            dst.band_count == ds.band_count
        ), f"band count drift: src={ds.band_count}, dst={dst.band_count}"

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
        crop_save_to: str,
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
        assert np.array_equal(
            values, rasterized_mask_values
        ), "the extracted values in the dataframe do not equal the real values in the array"


class TestCluster2:
    """Test converting raster to polygon."""

    def test_single_band(
        self,
        test_image: gdal.Dataset,
    ):
        dataset = Dataset(test_image)
        gdf = dataset.cluster2()
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == 4
        assert all(gdf.columns == ["GPP", "geometry"])
        assert all(gdf.geometry.geom_type == "Polygon")

    def test_multi_band_all_bands(
        self,
        sentinel_raster: gdal.Dataset,
    ):
        dataset = Dataset(sentinel_raster)
        gdf = dataset.cluster2()
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


class TestClustering:

    def test_generated_data(self):
        arr = np.random.default_rng(42).integers(1, 5, size=(3, 3))
        top_left_corner = (0, 0)
        cell_size = 0.05
        dataset = Dataset.create_from_array(
            arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
        )

        lower_value = 2
        upper_value = 4
        cluster_array, count, position, values = dataset.cluster(
            lower_value, upper_value
        )
        assert isinstance(cluster_array, np.ndarray)
        assert isinstance(count, int)
        assert isinstance(position, list)
        assert isinstance(values, list)

    def test_cluster(self, rhine_dem: gdal.Dataset, clusters: np.ndarray):
        dataset = Dataset(rhine_dem)
        lower_value = 0.1
        upper_value = 20
        cluster_array, count, position, values = dataset.cluster(
            lower_value, upper_value
        )
        assert count == 155
        assert np.array_equal(cluster_array, clusters)
        assert len(position) == 2364
        assert len(values) == 2364


def test_nearest_neigbors():
    arr = np.random.default_rng(0).random((5, 5))
    top_left_corner = (0, 0)
    cell_size = 0.05
    dataset = Dataset.create_from_array(
        arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
    )
    req_rows = [1, 3]
    req_cols = [2, 4]
    no_data_value = dataset.no_data_value[0]
    new_array = Vectorize._nearest_neighbour(arr, no_data_value, req_rows, req_cols)
    assert new_array.shape == arr.shape
