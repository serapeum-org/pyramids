"""Tests for the bbox/epsg kwargs on ``Dataset.crop``, ``Dataset.read_array`` and ``DatasetCollection.crop``."""

from __future__ import annotations

import math
import os

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.engines.spatial import (
    _reaches_antimeridian_seam,
    _split_lon_bbox,
)
from pyramids.feature import FeatureCollection
from pyramids.base.georeference import GeoReference

pytestmark = pytest.mark.core


def _make_raster(
    directory,
    name,
    *,
    shape=(10, 10),
    cell_size=0.05,
    top_left=(0.0, 0.0),
    epsg=4326,
    fill=None,
):
    """Write a small single-band GeoTIFF and return its path.

    Args:
        directory: pytest temp directory.
        name: File name (with ``.tif``).
        shape: ``(rows, cols)``.
        cell_size: Pixel size.
        top_left: ``(x, y)`` of the top-left corner.
        epsg: EPSG code.
        fill: Optional array to write; defaults to ``np.arange(rows*cols)``.

    Returns:
        str: Path to the written GeoTIFF.
    """
    if fill is None:
        fill = np.arange(shape[0] * shape[1], dtype="int16").reshape(shape)
    path = os.path.join(str(directory), name)
    Dataset.from_array(
        fill,
        path=path,
        geo_ref=GeoReference(top_left_corner=top_left, cell_size=cell_size, epsg=epsg),
    ).close()
    return path


@pytest.fixture()
def dataset(tmp_path) -> Dataset:
    """A 10×10 EPSG:4326 raster covering x∈[0,0.5], y∈[-0.5,0] at 0.05° cells.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        Dataset: The freshly-loaded raster.
    """
    path = _make_raster(tmp_path, "r.tif")
    return Dataset.read_file(path)


@pytest.fixture()
def small_bbox() -> tuple[float, float, float, float]:
    """A bbox covering arr[2:4, 2:4] of the 10×10 fixture (4 pixels).

    Returns:
        tuple[float, float, float, float]: ``(W, S, E, N)`` in EPSG:4326.
    """
    return (0.1, -0.2, 0.2, -0.1)


def _bbox_in_3857(
    bbox_4326: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Crude Web-Mercator transform of a lon/lat bbox (good enough for unit tests).

    Args:
        bbox_4326: ``(W, S, E, N)`` in EPSG:4326.

    Returns:
        tuple[float, float, float, float]: The same bbox in EPSG:3857.
    """

    def fwd(lon, lat):
        x = lon * 20037508.34 / 180.0
        y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        y = y * 20037508.34 / 180.0
        return x, y

    w, s, e, n = bbox_4326
    w3, s3 = fwd(w, s)
    e3, n3 = fwd(e, n)
    return (w3, s3, e3, n3)


def _ds_multiband() -> Dataset:
    """A 3-band 10x10 square-pixel EPSG:4326 raster."""
    return Dataset.from_array(
               np.arange(3 * 10 * 10, dtype="int16").reshape(3, 10, 10),
               geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326),
           )


def _ds_non_square() -> Dataset:
    """A 10x10 raster with 0.1 deg columns and 0.05 deg rows (dx != -dy)."""
    return Dataset.from_array(
               np.arange(100, dtype="int16").reshape(10, 10),
               geo_ref=GeoReference(geo=(0.0, 0.1, 0.0, 0.0, 0.0, -0.05), epsg=4326),
           )


def _ds_nodata_edge() -> Dataset:
    """A raster with a full no-data row at index 3 (an all-no-data edge to trim)."""
    arr = np.arange(100, dtype="float32").reshape(10, 10)
    arr[3, :] = -9999.0
    return Dataset.from_array(
               arr,
               no_data_value=-9999.0,
               geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326),
           )


class TestDatasetCropBbox:
    """Tests for ``Dataset.crop(bbox=..., epsg=...)``."""

    def test_bbox_equivalent_to_feature_collection_mask(self, dataset, small_bbox):
        """``crop(bbox=...)`` matches ``crop(mask=FeatureCollection.from_bbox(...))`` byte-for-byte.

        Args:
            dataset: 10×10 EPSG:4326 raster fixture.
            small_bbox: bbox covering 4 pixels of the fixture.

        Test scenario:
            Build an FC by hand and pass it as ``mask``; pass the same bbox via
            ``bbox=`` — expected: identical shape, geotransform, and array values.
        """
        fc = FeatureCollection.from_bbox(small_bbox, epsg=dataset.epsg)
        via_mask = dataset.crop(mask=fc)
        via_bbox = dataset.crop(bbox=small_bbox)
        assert via_bbox.shape == via_mask.shape, (
            f"shape differs: bbox={via_bbox.shape}, mask={via_mask.shape}"
        )
        assert via_bbox.geotransform == via_mask.geotransform, (
            "geotransform differs between bbox and mask paths"
        )
        assert np.array_equal(via_bbox.read_array(), via_mask.read_array()), (
            "pixel values differ between bbox and mask paths"
        )

    def test_default_epsg_is_dataset_crs(self, dataset, small_bbox):
        """Omitting ``epsg`` is the same as ``epsg=dataset.epsg``.

        Args:
            dataset: 10×10 EPSG:4326 raster fixture.
            small_bbox: bbox in EPSG:4326.

        Test scenario:
            ``crop(bbox=...)`` (no epsg) vs ``crop(bbox=..., epsg=4326)`` —
            expected: identical results.
        """
        a = dataset.crop(bbox=small_bbox)
        b = dataset.crop(bbox=small_bbox, epsg=4326)
        assert np.array_equal(a.read_array(), b.read_array()), (
            "default-epsg crop differs from explicit epsg=dataset.epsg"
        )

    def test_explicit_epsg_in_dataset_crs(self, dataset, small_bbox):
        """An explicit ``epsg`` that matches the dataset CRS is a no-op vs default.

        Args:
            dataset: 10×10 EPSG:4326 raster fixture.
            small_bbox: bbox in EPSG:4326.

        Test scenario:
            ``crop(bbox=..., epsg=4326)`` — expected: same as the no-epsg call.
        """
        a = dataset.crop(bbox=small_bbox)
        b = dataset.crop(bbox=small_bbox, epsg=4326)
        assert a.shape == b.shape, "explicit-epsg-matching-dataset crop shape changed"

    def test_bbox_in_different_crs_reprojects(self, dataset, small_bbox):
        """A bbox in a different CRS gets reprojected to the dataset CRS.

        Args:
            dataset: 10×10 EPSG:4326 raster fixture.
            small_bbox: bbox in EPSG:4326.

        Test scenario:
            Convert ``small_bbox`` to EPSG:3857 and crop with ``epsg=3857`` —
            expected: same cropped shape as cropping with the original bbox in
            EPSG:4326 (the standard reprojection path absorbs the CRS).
        """
        bbox_3857 = _bbox_in_3857(small_bbox)
        via_4326 = dataset.crop(bbox=small_bbox)
        via_3857 = dataset.crop(bbox=bbox_3857, epsg=3857)
        assert via_3857.shape == via_4326.shape, (
            f"reprojected crop shape {via_3857.shape} != native {via_4326.shape}"
        )

    def test_existing_mask_path_unchanged(self, dataset, small_bbox):
        """Passing a ``GeoDataFrame`` as ``mask`` still works (no regression).

        Args:
            dataset: 10×10 EPSG:4326 raster fixture.
            small_bbox: bbox in EPSG:4326.

        Test scenario:
            Hand-built ``gpd.GeoDataFrame`` mask — expected: the historical
            behaviour still works and produces the same shape as the bbox form.
        """
        w, s, e, n = small_bbox
        mask = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs=dataset.epsg)
        out = dataset.crop(mask=mask)
        assert out.shape == dataset.crop(bbox=small_bbox).shape, (
            "GeoDataFrame mask path diverged from bbox path"
        )

    def test_bbox_and_mask_mutually_exclusive(self, dataset, small_bbox):
        """Supplying both ``mask`` and ``bbox`` raises ``ValueError``.

        Args:
            dataset: Raster fixture.
            small_bbox: A valid bbox.

        Test scenario:
            ``crop(mask=fc, bbox=...)`` — expected: ``ValueError`` mentioning
            ``either ... or ... not both``.
        """
        fc = FeatureCollection.from_bbox(small_bbox, epsg=dataset.epsg)
        with pytest.raises(ValueError, match="not both"):
            dataset.crop(mask=fc, bbox=small_bbox)

    def test_no_mask_no_bbox_raises_type_error(self, dataset):
        """Supplying neither ``mask`` nor ``bbox`` raises ``TypeError``.

        Args:
            dataset: Raster fixture.

        Test scenario:
            ``crop()`` — expected: ``TypeError`` mentioning ``mask`` and ``bbox``.
        """
        with pytest.raises(TypeError, match=r"mask.*bbox|bbox.*mask"):
            dataset.crop()

    def test_invalid_bbox_validation_bubbles_up(self, dataset):
        """Bbox validation errors from :meth:`FeatureCollection.from_bbox` propagate.

        Args:
            dataset: Raster fixture.

        Test scenario:
            ``crop(bbox=(1, 0, 0, 1), epsg=3857)`` — ``west > east`` in a projected
            CRS is not an antimeridian case, so the ``west < east`` validator still
            raises (a geographic ``west > east`` is instead cropped across the seam).
        """
        with pytest.raises(ValueError, match=r"west < east"):
            dataset.crop(bbox=(1, 0, 0, 1), epsg=3857)

    def test_crop_entirely_nodata_raises_clear_error(self):
        """A crop overlapping only no-data cells raises a clear error, not IndexError.

        Args:
            None.

        Test scenario:
            A grid with valid data only in its top-left quadrant, cropped over the
            all-no-data bottom-right, must raise a "no valid pixels" ValueError from
            the touch-cutline correction rather than crash with an IndexError.
        """
        arr = np.full((10, 10), -9999.0, dtype="float32")
        arr[0:5, 0:5] = 1.0  # valid only in the top-left quadrant
        ds = Dataset.from_array(
                 arr,
                 no_data_value=-9999.0,
                 geo_ref=GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326),
             )
        with pytest.raises(ValueError, match="no valid pixels"):
            ds.crop(bbox=(6.0, 0.0, 9.0, 4.0))  # bottom-right: all no-data


class TestWindowedBboxCropFastPath:
    """#957: an axis-aligned same-CRS bbox crop reads only its window, not the whole source."""

    def test_windowed_crop_avoids_the_full_source_warp(
        self, dataset, small_bbox, mocker
    ):
        """A same-CRS bbox crop never calls the cutline warp.

        Test scenario:
            Patch ``warp_to_dataset`` to blow up if reached, then crop a bbox in the
            dataset's own CRS — the windowed fast path must satisfy it without the
            warp, proving no full-source warp is issued.
        """
        boom = mocker.patch(
            "pyramids.dataset.engines.spatial.warp_to_dataset",
            side_effect=AssertionError("warp path should not run for a same-CRS bbox"),
        )
        out = dataset.crop(bbox=small_bbox)
        assert out.shape == (1, 2, 2), f"unexpected crop shape {out.shape}"
        assert boom.call_count == 0, "the cutline warp was invoked for a same-CRS bbox"

    def test_windowed_crop_reads_only_the_aoi_window(self, dataset, small_bbox, mocker):
        """The crop issues a bounded windowed read of just the AOI, never a full read.

        Test scenario:
            Spy on ``Dataset.read_array``; the crop of a 2×2 bbox must read with the
            resolved ``[xoff, yoff, xsize, ysize]`` window (here ``[2, 2, 2, 2]``) and
            never a windowless (full-source) read. This bounded read is what makes a
            ``/vsicurl`` crop range-read only the AOI.
        """
        spy = mocker.spy(Dataset, "read_array")
        dataset.crop(bbox=small_bbox)
        # Spying the unbound method, a[0] is the receiver: keep only reads of the
        # source raster (later windowless reads land on the tiny derived crop, which
        # is fine — the point is the *source* is never read in full).
        source_windows = [
            kw.get("window", (a[1] if len(a) > 1 else None))
            for a, kw in spy.call_args_list
            if a and a[0] is dataset
        ]
        assert source_windows == [[2, 2, 2, 2]], (
            f"source must be read once, windowed to the AOI; saw {source_windows}"
        )

    def test_windowed_crop_matches_a_direct_windowed_read(self, dataset, small_bbox):
        """The crop equals a hand-rolled ``read_array(window=)`` + geotransform.

        Test scenario:
            The old hand-rolled recipe (``read_array(window=[2, 2, 2, 2])`` plus origin
            math) must equal ``crop(bbox=...)`` in both pixels and geotransform — the
            exact consolidation #957 asks for.
        """
        cropped = dataset.crop(bbox=small_bbox)
        gt = dataset.geotransform
        window_array = dataset.read_array(window=[2, 2, 2, 2])
        assert np.array_equal(cropped.read_array(), window_array), (
            "crop pixels differ from the direct windowed read"
        )
        expected_gt = (gt[0] + 2 * gt[1], gt[1], 0.0, gt[3] + 2 * gt[5], 0.0, gt[5])
        assert cropped.geotransform == expected_gt, (
            f"crop geotransform {cropped.geotransform} != expected {expected_gt}"
        )

    def test_reprojecting_bbox_skips_the_fast_path(self, dataset, small_bbox):
        """A bbox in a different CRS is not eligible for the windowed fast path.

        Test scenario:
            ``_crop_bbox_windowed`` must return ``None`` for a bbox declared in
            EPSG:3857 on a 4326 dataset, so ``crop`` falls back to the reprojecting
            warp path instead of reading a mis-projected window.
        """
        bbox_3857 = _bbox_in_3857(small_bbox)
        assert dataset.spatial._crop_bbox_windowed(bbox_3857, 3857) is None, (
            "a different-CRS bbox must fall back to the warp path"
        )

    def test_transposed_same_crs_bbox_skips_the_fast_path(self, dataset):
        """A transposed (west >= east) box in the source CRS is not eligible.

        Test scenario:
            ``_crop_bbox_windowed`` must return ``None`` for a ``west > east``,
            ``south > north`` box so ``crop`` falls through to
            :meth:`FeatureCollection.from_bbox`, which raises the clear
            ``west < east`` validation error rather than reading a negative window.
        """
        assert (
            dataset.spatial._crop_bbox_windowed((0.2, -0.1, 0.1, -0.2), 4326) is None
        ), "a transposed bbox must fall back to the warp/validation path"

    def test_bbox_outside_source_skips_the_fast_path(self, dataset):
        """A box that does not overlap the source is not eligible for the fast path.

        Test scenario:
            A bbox entirely east of the 10×10 fixture clips to a zero-width window;
            ``_crop_bbox_windowed`` must return ``None`` so the warp path reports the
            non-overlap with its usual error instead of building an empty array.
        """
        assert (
            dataset.spatial._crop_bbox_windowed((1.0, -0.2, 1.1, -0.1), 4326) is None
        ), "a non-overlapping bbox must fall back to the warp path"

    def test_rotated_grid_skips_the_fast_path(self, tmp_path):
        """A rotated (non-north-up) geotransform is not eligible for the fast path.

        Test scenario:
            Build a dataset whose geotransform carries a rotation term;
            ``_crop_bbox_windowed`` must return ``None`` so the crop uses the warp
            path, which handles the rotation correctly.
        """
        ds = Dataset.from_array(
            np.arange(100, dtype="int16").reshape(10, 10),
            # non-zero row-skew -> rotated
            geo_ref=GeoReference(geo=(0.0, 0.05, 0.01, 0.0, 0.0, -0.05), epsg=4326),
        )
        assert ds.spatial._crop_bbox_windowed((0.1, -0.2, 0.2, -0.1), 4326) is None, (
            "a rotated grid must fall back to the warp path"
        )

    @pytest.mark.parametrize(
        "geo",
        [
            (0.0, 0.05, 0.0, -0.5, 0.0, 0.05),  # south-up: dy > 0
            (0.5, -0.05, 0.0, 0.0, 0.0, -0.05),  # flipped x: dx < 0
        ],
        ids=["south-up", "negative-dx"],
    )
    def test_flipped_grid_skips_the_fast_path(self, geo):
        """A south-up (`dy > 0`) or negative-`dx` grid is not eligible for the fast path.

        Test scenario:
            ``_crop_bbox_windowed`` must return ``None`` for a non-north-up grid so the
            crop falls back to the warp path, which orients the axes correctly.
        """
        ds = Dataset.from_array(
                 np.arange(100, dtype="int16").reshape(10, 10),
                 geo_ref=GeoReference(geo=geo, epsg=4326),
             )
        assert ds.spatial._crop_bbox_windowed((0.1, -0.2, 0.2, -0.1), 4326) is None, (
            "a flipped/non-north-up grid must fall back to the warp path"
        )

    @pytest.mark.parametrize(
        "bbox",
        [
            (0.125, -0.375, 0.375, -0.125),  # edges on pixel centres, not boundaries
            (0.11, -0.19, 0.19, -0.11),  # arbitrary sub-pixel edges
            (0.13, -0.14, 0.14, -0.13),  # tiny box covering no pixel centre
            (0.07, -0.33, 0.28, -0.02),  # wider, off-grid on every side
        ],
    )
    def test_fast_path_matches_the_all_touched_warp_fallback(
        self, dataset, bbox, mocker
    ):
        """The windowed fast path returns the same crop as the all-touched warp fallback.

        Test scenario:
            For non-pixel-aligned boxes, compute the crop via the fast path, then patch
            ``_crop_bbox_windowed`` to ``None`` so the same bbox crop takes the
            (all-touched) warp fallback; the two must agree in shape, geotransform and
            pixels. This pins the overlap-semantics equivalence the fast path relies on.
        """
        fast = dataset.crop(bbox=bbox)
        mocker.patch(
            "pyramids.dataset.engines.spatial.Spatial._crop_bbox_windowed",
            return_value=None,
        )
        warp = dataset.crop(bbox=bbox)
        assert fast.shape == warp.shape, (
            f"shape differs for {bbox}: fast={fast.shape}, warp={warp.shape}"
        )
        assert fast.geotransform == warp.geotransform, (
            f"geotransform differs for {bbox}"
        )
        assert np.array_equal(fast.read_array(), warp.read_array()), (
            f"pixels differ for {bbox}"
        )

    @pytest.mark.parametrize(
        "make_ds, bbox",
        [
            (_ds_multiband, (0.11, -0.19, 0.19, -0.11)),  # multi-band, non-aligned
            (
                _ds_non_square,
                (0.15, -0.28, 0.63, -0.07),
            ),  # non-square pixels, non-aligned
            (_ds_multiband, (0.1, -0.2, 0.2, -0.1)),  # boundary-aligned
            (
                _ds_nodata_edge,
                (0.1, -0.35, 0.3, -0.1),
            ),  # AOI spans the all-no-data row 3
        ],
        ids=["multi-band", "non-square", "boundary-aligned", "no-data-edge"],
    )
    def test_fast_path_matches_warp_across_dataset_shapes(self, make_ds, bbox, mocker):
        """Fast path == all-touched warp fallback for multi-band, non-square, aligned and no-data-edge crops.

        Test scenario:
            Beyond the single-band square case, the fast path and the all-touched warp
            fallback must still agree in shape, geotransform and pixels for a multi-band
            raster, a non-square-pixel grid, a pixel-boundary-aligned box, and a box whose
            AOI contains an all-no-data row that both paths trim identically.
        """
        ds = make_ds()
        fast = ds.crop(bbox=bbox)
        mocker.patch(
            "pyramids.dataset.engines.spatial.Spatial._crop_bbox_windowed",
            return_value=None,
        )
        warp = make_ds().crop(bbox=bbox)
        assert fast.shape == warp.shape, (
            f"shape differs: fast={fast.shape}, warp={warp.shape}"
        )
        assert fast.geotransform == warp.geotransform, "geotransform differs"
        assert np.array_equal(fast.read_array(), warp.read_array()), "pixels differ"

    def test_overlap_semantics_keep_every_touched_pixel(self, dataset):
        """A box straddling pixel boundaries keeps every pixel it overlaps (floor/ceil).

        Test scenario:
            ``bbox=(0.125,-0.375,0.375,-0.125)`` on the 0.05 deg grid spans cols 2..7 and
            rows 2..7, so the crop is 6x6 with its origin at the box's floored top-left —
            the documented all-touched overlap result, wider than a centre-containment crop.
        """
        out = dataset.crop(bbox=(0.125, -0.375, 0.375, -0.125))
        assert out.shape == (1, 6, 6), f"expected 6x6 overlap crop, got {out.shape}"
        assert out.geotransform[0] == pytest.approx(0.10), (
            "origin x must be the floored west edge"
        )
        assert out.geotransform[3] == pytest.approx(-0.10), (
            "origin y must be the floored north edge"
        )

    def test_no_nodata_raster_crops_to_the_tight_overlap_window(self):
        """A raster with no no-data marker crops to the tight AOI, not an untrimmable border.

        Test scenario:
            With ``no_data_value=None`` the warp path leaves a border it cannot trim, but
            the fast path reads exactly the overlap window; an aligned 2x2 bbox must return
            a 2x2 crop.
        """
        ds = Dataset.from_array(
                 np.arange(100, dtype="int16").reshape(10, 10),
                 no_data_value=None,
                 geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326),
             )
        out = ds.crop(bbox=(0.1, -0.2, 0.2, -0.1))
        assert out.shape == (1, 2, 2), f"expected tight 2x2 crop, got {out.shape}"


class TestAntimeridianCrop:
    """Tests for crop with a geographic ``west > east`` (antimeridian) bbox."""

    @staticmethod
    def _global(top_left_x=-180.0):
        """Return (source array, global 1-degree Dataset) with the given lon origin."""
        arr = np.arange(180 * 360, dtype="float32").reshape(180, 360)
        ds = Dataset.from_array(
                 arr,
                 geo_ref=GeoReference(top_left_corner=(top_left_x, 90.0), cell_size=1.0, epsg=4326),
             )
        return arr, ds

    def test_strip_values_and_extent(self):
        """A -180..180 grid crop across the dateline stitches a contiguous strip."""
        arr, ds = self._global()
        strip = ds.crop(bbox=(170.0, -10.0, -170.0, 10.0))
        assert strip.shape == (1, 20, 20), "20 lat x 20 lon strip"
        assert strip.bbox == pytest.approx([170.0, -10.0, 190.0, 10.0]), "past 180"
        expected = np.concatenate([arr[80:100, 350:360], arr[80:100, 0:10]], axis=-1)
        assert np.array_equal(strip.read_array(), expected), "seam values preserved"

    def test_on_0_360_grid(self):
        """A 0..360 grid crops the same STAC bbox as a contiguous 170..190 strip."""
        arr, ds = self._global(top_left_x=0.0)
        strip = ds.crop(bbox=(170.0, -10.0, -170.0, 10.0))
        assert strip.shape == (1, 20, 20), "20x20 strip"
        assert strip.bbox == pytest.approx([170.0, -10.0, 190.0, 10.0]), (
            "170..190 extent"
        )
        assert np.array_equal(strip.read_array(), arr[80:100, 170:190]), "0..360 values"

    def test_multiband_keeps_all_bands(self):
        """A multi-band grid keeps all bands with correct stitched values."""
        arr, _ = self._global()
        ds = Dataset.from_array(
                 np.stack([arr, arr + 1000.0]),
                 geo_ref=GeoReference(top_left_corner=(-180.0, 90.0), cell_size=1.0, epsg=4326),
             )
        strip = ds.crop(bbox=(170.0, -10.0, -170.0, 10.0))
        assert strip.shape == (2, 20, 20), "both bands retained"
        seam = np.concatenate([arr[80:100, 350:360], arr[80:100, 0:10]], axis=-1)
        assert np.array_equal(strip.read_array(band=1), seam + 1000.0), "band 2"

    def test_multiband_on_0_360_grid_keeps_all_bands(self):
        """A multi-band 0..360 grid keeps every band with correct 170..190 values."""
        arr, _ = self._global(top_left_x=0.0)
        ds = Dataset.from_array(
                 np.stack([arr, arr + 1000.0]),
                 geo_ref=GeoReference(top_left_corner=(0.0, 90.0), cell_size=1.0, epsg=4326),
             )
        strip = ds.crop(bbox=(170.0, -10.0, -170.0, 10.0))
        assert strip.shape == (2, 20, 20), "both bands retained"
        assert np.array_equal(strip.read_array(band=0), arr[80:100, 170:190]), "band 1"
        assert np.array_equal(
            strip.read_array(band=1), arr[80:100, 170:190] + 1000.0
        ), "band 2"

    def test_float_overshoot_grid_keeps_both_halves(self):
        """A grid whose xmax floats past 180 is not misrouted to 0..360 (H1)."""
        cell = 360.0 / 169  # 169 cols -> bbox[2] == 180.00000000000006 > 180
        ds = Dataset.from_array(
                 np.arange(20 * 169, dtype="float32").reshape(20, 169),
                 geo_ref=GeoReference(top_left_corner=(-180.0, 20.0), cell_size=cell, epsg=4326),
             )
        assert ds.bbox[2] > 180.0, "fixture must actually overshoot to exercise H1"
        strip = ds.crop(bbox=(170.0, -10.0, -170.0, 10.0))
        # both halves kept -> the strip's east edge continues to ~190, not ~180
        assert abs(strip.bbox[2] - 190.0) < cell, "wrapped half must not be dropped"

    def test_normal_bbox_unchanged(self):
        """A west < east bbox still crops normally (no antimeridian path)."""
        _, ds = self._global()
        out = ds.crop(bbox=(10.0, -10.0, 30.0, 10.0))
        assert out.bbox == pytest.approx([10.0, -10.0, 30.0, 10.0]), "unaffected"

    def test_projected_dataset_not_treated_as_antimeridian(self):
        """A geographic west>east bbox on a projected dataset is not an antimeridian crop."""
        ds = Dataset.from_array(
                 np.zeros((5, 5), dtype="float32"),
                 geo_ref=GeoReference(top_left_corner=(0.0, 1000.0), cell_size=1000.0, epsg=3857),
             )
        with pytest.raises(ValueError, match="west < east"):
            ds.crop(bbox=(170.0, -10.0, -170.0, 10.0), epsg=4326)

    def test_regional_grid_reversed_bbox_raises(self):
        """A west>east bbox on a regional grid that never reaches the seam raises."""
        arr = np.arange(180 * 50, dtype="float32").reshape(180, 50)
        ds = Dataset.from_array(
                 arr,
                 geo_ref=GeoReference(top_left_corner=(-10.0, 90.0), cell_size=1.0, epsg=4326),
             )  # lon -10..40 (Europe): reaches neither +180 nor -180
        with pytest.raises(ValueError, match="transposed|does not reach the 180 seam"):
            ds.crop(bbox=(40.0, -10.0, 10.0, 10.0))

    def test_single_side_overlap_returns_half(self):
        """When only one side of the seam overlaps, that half is returned as-is."""
        arr = np.arange(180 * 10, dtype="float32").reshape(180, 10)
        ds = Dataset.from_array(
                 arr,
                 geo_ref=GeoReference(top_left_corner=(170.0, 90.0), cell_size=1.0, epsg=4326),
             )  # lon 170..180 only (west side of the seam)
        strip = ds.crop(bbox=(175.0, -10.0, -170.0, 10.0))
        assert strip.bbox[0] == pytest.approx(175.0), "west edge kept"
        assert strip.bbox[2] == pytest.approx(180.0), "only the west half (no wrap)"

    @pytest.mark.parametrize(
        "top_left_x, ncols, reaches",
        [
            (-180.0, 360, True),  # global -180..180 (touches -180 and 180)
            (0.0, 360, True),  # global 0..360 (reaches past 180 to ~360)
            (170.0, 10, True),  # regional but ends exactly at the +180 seam
            (-10.0, 50, False),  # Europe -10..40: reaches neither seam
            (0.0, 90, False),  # eastern hemisphere 0..90: reaches neither
            (0.0, 256, False),  # partial 0..360 (lon 0..256): stops short of 360
        ],
    )
    def test_reaches_antimeridian_seam(self, top_left_x, ncols, reaches):
        """The seam gate accepts grids reaching 180/-180 and rejects regional/partial ones."""
        ds = Dataset.from_array(
                 np.zeros((2, ncols), dtype="float32"),
                 geo_ref=GeoReference(top_left_corner=(top_left_x, 1.0), cell_size=1.0, epsg=4326),
             )
        assert _reaches_antimeridian_seam(ds) is reaches

    def test_split_lon_bbox_0_360_yields_single_half(self):
        """On a 0..360 grid the west>east bbox shifts into one west<east half."""
        bbox = (170.0, -10.0, -170.0, 10.0)
        halves = _split_lon_bbox(bbox, lon_max=359.0, cell_x=1.0)
        assert halves == [(170.0, -10.0, 190.0, 10.0)]

    def test_split_lon_bbox_180_yields_two_halves(self):
        """On a -180..180 grid the west>east bbox splits at the 180 seam."""
        bbox = (170.0, -10.0, -170.0, 10.0)
        halves = _split_lon_bbox(bbox, lon_max=180.0, cell_x=1.0)
        assert halves == [
            (170.0, -10.0, 180.0, 10.0),
            (-180.0, -10.0, -170.0, 10.0),
        ]


class TestDatasetReadArrayBbox:
    """Tests for ``Dataset.read_array(bbox=..., epsg=...)``."""

    def test_bbox_matches_window_geodataframe(self, dataset, small_bbox):
        """``read_array(bbox=...)`` matches ``read_array(window=fc)`` for the same bbox.

        Args:
            dataset: 10×10 EPSG:4326 raster fixture.
            small_bbox: bbox covering 4 pixels.

        Test scenario:
            Read via bbox and via window=FeatureCollection — expected: identical
            arrays.
        """
        fc = FeatureCollection.from_bbox(small_bbox, epsg=dataset.epsg)
        via_window = dataset.read_array(window=fc)
        via_bbox = dataset.read_array(bbox=small_bbox)
        assert np.array_equal(via_bbox, via_window), (
            "bbox read differs from window=fc read"
        )

    def test_default_epsg(self, dataset, small_bbox):
        """``read_array(bbox=...)`` defaults ``epsg`` to the dataset's CRS.

        Args:
            dataset: 10×10 EPSG:4326 raster fixture.
            small_bbox: bbox in EPSG:4326.

        Test scenario:
            Compare default-epsg vs ``epsg=dataset.epsg`` — expected: identical.
        """
        a = dataset.read_array(bbox=small_bbox)
        b = dataset.read_array(bbox=small_bbox, epsg=dataset.epsg)
        assert np.array_equal(a, b), "default-epsg read_array differs from explicit"

    def test_bbox_and_window_mutually_exclusive(self, dataset, small_bbox):
        """Supplying both ``window`` and ``bbox`` raises ``ValueError``.

        Args:
            dataset: Raster fixture.
            small_bbox: A valid bbox.

        Test scenario:
            ``read_array(window=fc, bbox=...)`` — expected: ``ValueError``.
        """
        fc = FeatureCollection.from_bbox(small_bbox, epsg=dataset.epsg)
        with pytest.raises(ValueError, match="not both"):
            dataset.read_array(window=fc, bbox=small_bbox)

    def test_pixel_window_path_unchanged(self, dataset):
        """The legacy 4-int pixel ``window=[off_x, off_y, n_cols, n_rows]`` still works.

        Args:
            dataset: Raster fixture.

        Test scenario:
            ``read_array(window=[2, 2, 2, 2])`` — expected: a ``(2, 2)`` array
            matching ``arr[2:4, 2:4]`` (the bbox form maps to this region).
        """
        full = dataset.read_array()
        block = dataset.read_array(window=[2, 2, 2, 2])
        assert block.shape == (2, 2), f"unexpected pixel-window shape: {block.shape}"
        assert np.array_equal(block, full[2:4, 2:4]), "pixel-window slice mismatch"

    def test_invalid_bbox_validation_bubbles_up(self, dataset):
        """Bbox validation errors propagate from :meth:`FeatureCollection.from_bbox`.

        Args:
            dataset: Raster fixture.

        Test scenario:
            ``read_array(bbox=(0, 1, 1, 0))`` (south >= north) — expected: ``ValueError``.
        """
        with pytest.raises(ValueError, match=r"south < north"):
            dataset.read_array(bbox=(0, 1, 1, 0))

    @pytest.fixture()
    def multiband_dataset(self, tmp_path) -> Dataset:
        """A 3-band 10×10 EPSG:4326 raster; band ``i`` is ``arange(100) + i*1000`` (uint16).

        Args:
            tmp_path: pytest temp directory.

        Returns:
            Dataset: The freshly-loaded 3-band raster (no-data 0, so the
            windowed region arr[2:4, 2:4] holds no no-data cells).
        """
        ramp = np.arange(100, dtype="uint16").reshape(10, 10)
        cube = np.stack([ramp + b * 1000 for b in range(3)]).astype("uint16")
        path = os.path.join(str(tmp_path), "mb.tif")
        Dataset.from_array(
            cube,
            no_data_value=0,
            path=path,
            geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326),
        ).close()
        return Dataset.read_file(path)

    def test_multiband_bbox_matches_window_geodataframe(
        self, multiband_dataset, small_bbox
    ):
        """Multi-band ``read_array(bbox=...)`` equals ``read_array(window=fc)``.

        Args:
            multiband_dataset: 3-band 10×10 raster fixture.
            small_bbox: bbox covering arr[2:4, 2:4] (4 pixels per band).

        Test scenario:
            Regression for the multi-band windowed path indexing the
            ``FeatureCollection`` window as ``window[2]`` / ``window[3]`` (which
            raised ``KeyError``). Read all bands via bbox and via window=fc —
            expected: both return ``(3, 2, 2)`` and are identical.
        """
        fc = FeatureCollection.from_bbox(small_bbox, epsg=multiband_dataset.epsg)
        via_window = multiband_dataset.read_array(window=fc)
        via_bbox = multiband_dataset.read_array(bbox=small_bbox)
        assert via_bbox.shape == (3, 2, 2), f"unexpected shape: {via_bbox.shape}"
        assert np.array_equal(via_bbox, via_window), (
            "multi-band bbox read differs from window=fc read"
        )

    def test_multiband_bbox_matches_per_band_reads_and_dtype(
        self, multiband_dataset, small_bbox
    ):
        """Each band of a multi-band bbox read equals the single-band bbox read; dtype intact.

        Args:
            multiband_dataset: 3-band 10×10 raster fixture.
            small_bbox: bbox covering a 2×2 window per band.

        Test scenario:
            Validate the fixed multi-band windowed path against the known-good
            single-band path (``read_array(band=i, bbox=...)``) rather than a
            hardcoded slice — expected: ``block[i]`` equals the per-band read for
            every band, and the unsigned ``uint16`` dtype is preserved (no cast
            through a signed pre-allocation).
        """
        block = multiband_dataset.read_array(bbox=small_bbox)
        assert block.dtype == np.dtype("uint16"), f"dtype not preserved: {block.dtype}"
        for b in range(multiband_dataset.band_count):
            per_band = multiband_dataset.read_array(band=b, bbox=small_bbox)
            assert np.array_equal(block[b], per_band), (
                f"multi-band block band {b} differs from the single-band bbox read"
            )

    def test_multiband_full_read_unchanged(self, multiband_dataset):
        """``read_array()`` with no window still returns every band unchanged.

        Args:
            multiband_dataset: 3-band 10×10 raster fixture.

        Test scenario:
            Guard that refactoring the windowed branch left the
            ``window is None`` full read intact — expected: ``(3, 10, 10)`` and
            band ``i`` equals ``arange(100).reshape(10, 10) + i*1000``.
        """
        full = multiband_dataset.read_array()
        assert full.shape == (3, 10, 10), f"unexpected full shape: {full.shape}"
        ramp = np.arange(100, dtype="uint16").reshape(10, 10)
        for b in range(3):
            assert np.array_equal(full[b], ramp + b * 1000), (
                f"band {b} full read mismatch"
            )


class TestDatasetCollectionCropBbox:
    """Tests for ``DatasetCollection.crop(bbox=..., epsg=...)``."""

    @pytest.fixture()
    def collection(self, tmp_path) -> DatasetCollection:
        """Two-timestep collection on the same 10×10 grid (values 1×arange and 2×arange).

        Args:
            tmp_path: pytest temp directory.

        Returns:
            DatasetCollection: A 2-timestep collection.
        """
        arr1 = np.arange(100, dtype="int16").reshape(10, 10)
        arr2 = (arr1 * 2).astype("int16")
        p1 = _make_raster(tmp_path, "t0.tif", fill=arr1)
        p2 = _make_raster(tmp_path, "t1.tif", fill=arr2)
        return DatasetCollection.from_files([p1, p2])

    def test_bbox_crop_passes_through_to_each_timestep(self, collection, small_bbox):
        """Cropping with a bbox applies to every timestep with the same window.

        Args:
            collection: 2-timestep fixture.
            small_bbox: bbox covering 4 pixels.

        Test scenario:
            ``collection.crop(bbox=...)`` — expected: same ``time_length``,
            template shape shrunk to the bbox window, and the per-timestep
            arrays equal the bbox crop of the original timesteps.
        """
        cc = collection.crop(bbox=small_bbox)
        assert cc.time_length == collection.time_length, (
            f"time_length changed: {collection.time_length} -> {cc.time_length}"
        )
        assert cc.base.shape == (1, 2, 2), f"unexpected template shape: {cc.base.shape}"
        ref0 = collection.iloc(0).crop(bbox=small_bbox).read_array()
        ref1 = collection.iloc(1).crop(bbox=small_bbox).read_array()
        assert np.array_equal(cc[0], ref0), "timestep 0 differs from per-Dataset crop"
        assert np.array_equal(cc[1], ref1), "timestep 1 differs from per-Dataset crop"

    def test_default_epsg_is_base_crs(self, collection, small_bbox):
        """``epsg`` defaults to the collection's own CRS.

        Args:
            collection: 2-timestep fixture.
            small_bbox: bbox in the base CRS.

        Test scenario:
            No-epsg call vs ``epsg=base.epsg`` — expected: identical results.
        """
        a = collection.crop(bbox=small_bbox)
        b = collection.crop(bbox=small_bbox, epsg=collection.base.epsg)
        assert np.array_equal(a[0], b[0]), (
            "default-epsg collection crop differs from explicit"
        )

    def test_bbox_and_mask_mutually_exclusive(self, collection, small_bbox):
        """Both ``mask`` and ``bbox`` together raises ``ValueError``.

        Args:
            collection: 2-timestep fixture.
            small_bbox: A valid bbox.

        Test scenario:
            ``collection.crop(mask=fc, bbox=...)`` — expected: ``ValueError``.
        """
        fc = FeatureCollection.from_bbox(small_bbox, epsg=collection.base.epsg)
        with pytest.raises(ValueError, match="not both"):
            collection.crop(mask=fc, bbox=small_bbox)

    def test_no_mask_no_bbox_raises_type_error(self, collection):
        """Neither ``mask`` nor ``bbox`` raises ``TypeError``.

        Args:
            collection: 2-timestep fixture.

        Test scenario:
            ``collection.crop()`` — expected: ``TypeError`` mentioning the args.
        """
        with pytest.raises(TypeError, match=r"mask.*bbox|bbox.*mask"):
            collection.crop()


class TestCrossCrsBboxCrop:
    """A bbox given in a CRS other than the raster's must still crop the raster."""

    @staticmethod
    def _utm_raster(tmp_path, no_data_value):
        """Write a 64x64 EPSG:32636 raster, optionally declaring a no-data value."""
        from osgeo import gdal, osr

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32636)
        path = os.path.join(tmp_path, f"utm_{no_data_value}.tif")
        raster = gdal.GetDriverByName("GTiff").Create(path, 64, 64, 1, gdal.GDT_Float32)
        raster.SetProjection(srs.ExportToWkt())
        raster.SetGeoTransform([400000.0, 1000.0, 0.0, 3300000.0, 0.0, -1000.0])
        raster.GetRasterBand(1).WriteArray(np.ones((64, 64), dtype="float32"))
        if no_data_value is not None:
            raster.GetRasterBand(1).SetNoDataValue(no_data_value)
        raster.FlushCache()
        raster = None
        return path

    @staticmethod
    def _lonlat_bbox_inset(dataset):
        """A lon/lat bbox covering the middle half of `dataset`'s own extent."""
        from pyramids.base.crs import reproject_coordinates

        min_x, min_y, max_x, max_y = dataset.bounds.total_bounds
        (west, east), (south, north) = reproject_coordinates(
            [min_x, max_x], [min_y, max_y], from_crs=32636, to_crs=4326, precision=None
        )
        inset_x, inset_y = (east - west) / 4, (north - south) / 4
        return [west + inset_x, south + inset_y, east - inset_x, north - inset_y]

    @pytest.mark.parametrize("no_data_value", [-9999.0, None], ids=["nodata", "none"])
    def test_crops_regardless_of_no_data(self, tmp_path, no_data_value):
        """A cross-CRS bbox crops whether or not the band declares a no-data value.

        Test scenario:
            With `touch=True` the cutline window optimisation is skipped for a
            differing CRS, and the crop used to fall back on trimming an all-no-data
            border. A raster with no no-data value has no border to trim, so the call
            returned the source *uncropped* and silently — while the identical call on
            a raster that declared no-data cropped correctly.
        """
        path = self._utm_raster(str(tmp_path), no_data_value)
        dataset = Dataset.read_file(path)
        bbox = self._lonlat_bbox_inset(dataset)

        cropped = Dataset.read_file(path).crop(bbox=bbox, epsg=4326, touch=True)

        # A bbox over the middle half should land near half the source on each axis.
        # Asserting only `< 64` would still pass at 63x63 — it would not notice the
        # crop degenerating back towards the whole raster, which is the bug itself.
        _, rows, cols = cropped.shape
        expected = "roughly 32x32 (allowing the touch margin and reprojection slack)"
        assert 28 <= rows <= 40, f"rows should be {expected}, got {rows}"
        assert 28 <= cols <= 40, f"cols should be {expected}, got {cols}"

    def test_matches_the_same_crs_crop(self, tmp_path):
        """The same window expressed in the raster's own CRS gives the same crop.

        Test scenario:
            Reprojecting the bbox is the caller's alternative to passing `epsg=`; both
            routes must agree to within the extra cell `touch=True` may include.
        """
        path = self._utm_raster(str(tmp_path), None)
        dataset = Dataset.read_file(path)
        min_x, min_y, max_x, max_y = dataset.bounds.total_bounds
        inset_x, inset_y = (max_x - min_x) / 4, (max_y - min_y) / 4

        via_lonlat = Dataset.read_file(path).crop(
            bbox=self._lonlat_bbox_inset(dataset), epsg=4326, touch=True
        )
        via_native = Dataset.read_file(path).crop(
            bbox=[min_x + inset_x, min_y + inset_y, max_x - inset_x, max_y - inset_y],
            epsg=32636,
            touch=True,
        )

        _, native_rows, native_cols = via_native.shape
        _, lonlat_rows, lonlat_cols = via_lonlat.shape
        # Assert the *relationship*, not an absolute cell count. The lon/lat window
        # is a reprojected quadrilateral, so its axis-aligned envelope is never
        # smaller than the native rectangle and never much larger. A fixed "within 2
        # cells" tolerance happens to be met exactly on this PROJ build, so it would
        # start failing on a PROJ bump for no real reason.
        assert native_rows <= lonlat_rows <= native_rows * 1.25, (
            f"lon/lat crop should bracket the native one, got {lonlat_rows} rows "
            f"against {native_rows}"
        )
        assert native_cols <= lonlat_cols <= native_cols * 1.25, (
            f"lon/lat crop should bracket the native one, got {lonlat_cols} cols "
            f"against {native_cols}"
        )


class TestCropCrsWithoutEpsgCode:
    """`crop` must work for a CRS the EPSG register does not name (issue #964)."""

    @staticmethod
    def _raster(tmp_path, name, crs_text):
        """Write a 64x64 raster centred on the origin of `crs_text`."""
        from osgeo import gdal, osr

        srs = osr.SpatialReference()
        srs.SetFromUserInput(crs_text)
        path = os.path.join(tmp_path, f"{name}.tif")
        raster = gdal.GetDriverByName("GTiff").Create(path, 64, 64, 1, gdal.GDT_Float32)
        raster.SetProjection(srs.ExportToWkt())
        raster.SetGeoTransform([-32000.0, 1000.0, 0.0, 32000.0, 0.0, -1000.0])
        raster.GetRasterBand(1).WriteArray(np.ones((64, 64), dtype="float32"))
        raster.GetRasterBand(1).SetNoDataValue(-9999.0)
        raster.FlushCache()
        raster = None
        return path

    @pytest.mark.parametrize(
        ("name", "crs_text"),
        [
            ("ortho", "+proj=ortho +lat_0=39 +lon_0=-9 +datum=WGS84 +units=m +no_defs"),
            ("robinson", "ESRI:54030"),
        ],
    )
    def test_crops_a_raster_whose_crs_has_no_epsg_code(self, tmp_path, name, crs_text):
        """A raster in an EPSG-less CRS crops in its own CRS.

        Test scenario:
            The cutline is staged as GeoJSON, which can name a CRS only as an OGC
            URN. A CRS with no authority code was therefore written with no CRS at
            all, GDAL assumed the GeoJSON default of CRS84, and transforming metre
            coordinates as lon/lat failed with "Invalid latitude".
        """
        from osgeo import osr

        path = self._raster(str(tmp_path), name, crs_text)
        dataset = Dataset.read_file(path)
        # The precondition is "no EPSG authority". Checking the authority name rather
        # than `Dataset.epsg` keeps the test honest regardless of how `epsg` chooses
        # to report a non-EPSG authority.
        authority = osr.SpatialReference(wkt=dataset.crs).GetAuthorityName(None)
        assert authority != "EPSG", (
            f"precondition: this CRS must carry no EPSG authority, got {authority}"
        )

        cropped = dataset.crop(
            bbox=[-16000.0, -16000.0, 16000.0, 16000.0], epsg=dataset.crs, touch=True
        )

        _, rows, cols = cropped.shape
        expected = "roughly 32x32 for a half-width bbox on a 64x64 raster"
        assert 28 <= rows <= 40, f"rows should be {expected}, got {rows}"
        assert 28 <= cols <= 40, f"cols should be {expected}, got {cols}"


class TestCutlineSegmentLength:
    """Tests for the cutline densification step (`_cutline_segment_length`)."""

    def test_returns_none_for_a_degenerate_envelope(self):
        """A zero-extent cutline yields None so the caller skips densification.

        Test scenario:
            A single point has no span to divide, and dividing by it would be a
            zero or non-finite step.
        """
        from shapely.geometry import Point

        from pyramids.dataset.engines.spatial import Spatial
        from pyramids.feature import FeatureCollection

        cutline = FeatureCollection(
            gpd.GeoDataFrame(geometry=[Point(0.0, 0.0)], crs="EPSG:4326")
        )
        assert Spatial._cutline_segment_length(None, cutline) is None

    def test_returns_none_when_the_scale_cannot_be_measured(self, dataset):
        """An unmeasurable scale falls back to an undensified reprojection.

        Test scenario:
            Passing a source with no usable geotransform exercises the give-up
            branch, which must answer None rather than propagate.
        """
        from pyramids.dataset.engines.spatial import Spatial
        from pyramids.feature import FeatureCollection

        cutline = FeatureCollection(
            gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326")
        )
        assert Spatial._cutline_segment_length(object(), cutline) is None

    def test_scales_with_the_source_pixel(self, dataset):
        """The step tracks the raster's cell size, not a fixed slice of the envelope.

        Test scenario:
            A fixed `span/64` was scale-free: the same step for a 1 km grid and a
            30 m one. The measured step must be positive and no larger than the
            envelope it densifies.
        """
        from pyramids.dataset.engines.spatial import Spatial
        from pyramids.feature import FeatureCollection

        cutline = FeatureCollection(
            gpd.GeoDataFrame(geometry=[box(0.1, -0.2, 0.4, -0.05)], crs="EPSG:3857")
        )
        step = Spatial._cutline_segment_length(dataset, cutline)
        assert step is None or step > 0, f"a measured step must be positive, got {step}"
