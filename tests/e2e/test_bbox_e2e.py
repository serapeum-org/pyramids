"""End-to-end workflow tests for the bbox convenience APIs (PY-5).

Simulates the consumer pattern these helpers exist for: a caller has a plain
``(west, south, east, north)`` bbox (e.g. from a GEE region request, or a
hand-picked AOI) and wants to clip a raster — or every timestep of a
collection — without first hand-building a :class:`FeatureCollection`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core

CELL_SIZE = 0.05


def _write_raster(directory, name, values, *, epsg=4326):
    """Write a 10×10 GeoTIFF and return its path.

    Args:
        directory: pytest temp directory.
        name: File name (with ``.tif``).
        values: ``(10, 10)`` int16 fill values.
        epsg: EPSG code.

    Returns:
        str: Path to the written GeoTIFF.
    """
    path = os.path.join(str(directory), name)
    Dataset.create_from_array(
        np.asarray(values, dtype="int16"),
        top_left_corner=(0.0, 0.0),
        cell_size=CELL_SIZE,
        epsg=epsg,
        path=path,
    ).close()
    return path


class TestBboxClipPersistE2E:
    """E2E: caller has a bbox -> clip a raster -> save -> reload."""

    def test_bbox_clip_to_disk_round_trip(self, tmp_path):
        """A bbox-cropped raster persists to disk and reloads with the same values.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            10×10 raster of ``arange`` -> ``crop(bbox=(0.1, -0.2, 0.2, -0.1))``
            -> ``to_file`` -> ``read_file`` -> expected: a 2×2 raster whose
            values equal ``arr[2:4, 2:4]``.
        """
        arr = np.arange(100, dtype="int16").reshape(10, 10)
        src = Dataset.read_file(_write_raster(tmp_path, "src.tif", arr))
        cropped = src.crop(bbox=(0.1, -0.2, 0.2, -0.1))

        out = tmp_path / "clip.tif"
        cropped.to_file(str(out))
        assert out.exists(), "cropped raster was not written"

        reloaded = Dataset.read_file(str(out))
        assert reloaded.shape == (1, 2, 2), f"unexpected shape: {reloaded.shape}"
        assert np.array_equal(reloaded.read_array(), arr[2:4, 2:4]), (
            "round-tripped pixels do not match the source slice"
        )

    def test_read_array_bbox_matches_geodataframe_window(self, tmp_path):
        """``read_array(bbox=...)`` matches the historical ``window=FeatureCollection`` form.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Read the same area both ways — expected: byte-for-byte equal
            arrays. (The bbox path is sugar over the existing
            ``window=GeoDataFrame`` path; this is the documented invariant.
            Bbox-to-pixel edge handling is delegated to
            ``_convert_polygon_to_window``, so the two paths necessarily
            agree.)
        """
        arr = np.arange(100, dtype="int16").reshape(10, 10)
        src = Dataset.read_file(_write_raster(tmp_path, "src.tif", arr))
        bbox = (0.1, -0.2, 0.2, -0.1)
        via_bbox = src.read_array(bbox=bbox)
        via_window = src.read_array(
            window=FeatureCollection.from_bbox(bbox, epsg=src.epsg)
        )
        assert np.array_equal(via_bbox, via_window), (
            "bbox read_array diverged from the equivalent window=fc form"
        )

    def test_bbox_in_foreign_crs_reprojects(self, tmp_path):
        """A WGS84 bbox crops a Web-Mercator raster after reprojection.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Write a raster in EPSG:3857; pass a bbox in EPSG:4326 with
            ``epsg=4326`` — expected: cropped raster's bounds, reprojected back
            to EPSG:4326, contain the input bbox (the reprojection path absorbs
            the CRS difference).
        """
        arr = np.arange(100, dtype="int16").reshape(10, 10)
        src_path = os.path.join(str(tmp_path), "merc.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=10000.0,
            epsg=3857,
            path=src_path,
        ).close()
        src = Dataset.read_file(src_path)

        cropped = src.crop(bbox=(0.05, -0.4, 0.4, -0.05), epsg=4326)
        assert cropped.shape[0] >= 1 and cropped.shape[1] >= 1, (
            f"reprojected crop yielded an empty raster: {cropped.shape}"
        )
        assert cropped.epsg == 3857, (
            "cropping must keep the dataset's CRS — only the bbox is reprojected in"
        )


class TestBboxCollectionE2E:
    """E2E: caller has a bbox -> clip every timestep of a DatasetCollection."""

    def test_collection_bbox_clip_per_timestep(self, tmp_path):
        """A bbox crop on a collection produces the same per-timestep window as a Dataset crop.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Build two co-registered timesteps (``arr`` and ``arr*2``), crop the
            collection with the bbox, and compare each per-timestep array to
            ``Dataset.crop(bbox=...)`` of the same source.
        """
        arr = np.arange(100, dtype="int16").reshape(10, 10)
        p0 = _write_raster(tmp_path, "t0.tif", arr)
        p1 = _write_raster(tmp_path, "t1.tif", (arr * 2).astype("int16"))
        collection = DatasetCollection.from_files([p0, p1])

        cropped = collection.crop(bbox=(0.1, -0.2, 0.2, -0.1))
        assert cropped.time_length == 2, (
            f"expected 2 timesteps, got {cropped.time_length}"
        )
        assert cropped.base.shape == (
            1,
            2,
            2,
        ), f"unexpected template shape: {cropped.base.shape}"
        expected0 = Dataset.read_file(p0).crop(bbox=(0.1, -0.2, 0.2, -0.1)).read_array()
        expected1 = Dataset.read_file(p1).crop(bbox=(0.1, -0.2, 0.2, -0.1)).read_array()
        assert np.array_equal(cropped[0], expected0), "timestep 0 mismatch"
        assert np.array_equal(cropped[1], expected1), "timestep 1 mismatch"


class TestBboxPrimitiveAlsoUsableStandaloneE2E:
    """E2E: callers may build a FeatureCollection from a bbox and reuse it."""

    def test_from_bbox_then_crop_with_mask(self, tmp_path):
        """A FC built via ``from_bbox`` is a drop-in mask for ``crop``.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Build an FC via :meth:`FeatureCollection.from_bbox` once, pass it
            as ``mask`` to two crops — expected: same output both times, and
            equal to ``crop(bbox=...)`` directly.
        """
        arr = np.arange(100, dtype="int16").reshape(10, 10)
        src = Dataset.read_file(_write_raster(tmp_path, "r.tif", arr))
        fc = FeatureCollection.from_bbox((0.1, -0.2, 0.2, -0.1), epsg=src.epsg)

        a = src.crop(mask=fc)
        b = src.crop(mask=fc)
        c = src.crop(bbox=(0.1, -0.2, 0.2, -0.1))
        assert np.array_equal(a.read_array(), b.read_array()), (
            "mask reuse changed output"
        )
        assert np.array_equal(a.read_array(), c.read_array()), (
            "mask path differs from bbox path"
        )
