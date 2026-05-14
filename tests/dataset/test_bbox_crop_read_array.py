"""Tests for the bbox/epsg kwargs on ``Dataset.crop``, ``Dataset.read_array`` and ``DatasetCollection.crop``."""

from __future__ import annotations

import math
import os
from typing import Tuple

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.feature import FeatureCollection

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
    Dataset.create_from_array(
        fill,
        top_left_corner=top_left,
        cell_size=cell_size,
        epsg=epsg,
        path=path,
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
def small_bbox() -> Tuple[float, float, float, float]:
    """A bbox covering arr[2:4, 2:4] of the 10×10 fixture (4 pixels).

    Returns:
        tuple[float, float, float, float]: ``(W, S, E, N)`` in EPSG:4326.
    """
    return (0.1, -0.2, 0.2, -0.1)


def _bbox_in_3857(
    bbox_4326: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
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
        assert (
            via_bbox.shape == via_mask.shape
        ), f"shape differs: bbox={via_bbox.shape}, mask={via_mask.shape}"
        assert (
            via_bbox.geotransform == via_mask.geotransform
        ), "geotransform differs between bbox and mask paths"
        assert np.array_equal(
            via_bbox.read_array(), via_mask.read_array()
        ), "pixel values differ between bbox and mask paths"

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
        assert np.array_equal(
            a.read_array(), b.read_array()
        ), "default-epsg crop differs from explicit epsg=dataset.epsg"

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
        assert (
            via_3857.shape == via_4326.shape
        ), f"reprojected crop shape {via_3857.shape} != native {via_4326.shape}"

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
        assert (
            out.shape == dataset.crop(bbox=small_bbox).shape
        ), "GeoDataFrame mask path diverged from bbox path"

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
            ``crop(bbox=(1, 0, 0, 1))`` (west >= east) — expected: ``ValueError``
            from the underlying validator.
        """
        with pytest.raises(ValueError, match=r"west < east"):
            dataset.crop(bbox=(1, 0, 0, 1))


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
        assert np.array_equal(
            via_bbox, via_window
        ), "bbox read differs from window=fc read"

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
        assert (
            cc.time_length == collection.time_length
        ), f"time_length changed: {collection.time_length} -> {cc.time_length}"
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
        assert np.array_equal(
            a[0], b[0]
        ), "default-epsg collection crop differs from explicit"

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
