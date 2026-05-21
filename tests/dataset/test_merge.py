"""Tests for :func:`pyramids.dataset.merge.merge_rasters` ``method=`` (PB-7).

Covers the overlap-resolution rule added to ``merge_rasters``: the z-order
``first`` / ``last`` paths and the ``min`` / ``max`` / ``sum`` reduction paths
(via ``_merge_reduce``), no-coverage fill, the ``n`` source-nodata knob,
multi-band reduction, and the guard / error branches. Source rasters are written
to ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.dataset.merge import merge_rasters

pytestmark = pytest.mark.core


def _write(path, arr, top_left, *, cell_size=1.0, nodata=-9999.0):
    """Write ``arr`` to ``path`` as a GeoTIFF and return the path string."""
    ds = Dataset.create_from_array(
        arr, top_left_corner=top_left, cell_size=cell_size, epsg=4326, no_data_value=nodata
    )
    ds.to_file(str(path))
    return str(path)


@pytest.fixture(scope="function")
def overlapping_pair(tmp_path):
    """Two 4x4 rasters overlapping in a 2-column strip on a shared 6x4 grid.

    Raster A (value 10) sits at columns 0..3; raster B (value 20) at columns
    2..5. Their union is 6 wide × 4 tall, with columns 2..3 overlapping.

    Returns:
        tuple[str, str]: (path_a, path_b).
    """
    a = np.full((4, 4), 10.0, dtype="float32")
    b = np.full((4, 4), 20.0, dtype="float32")
    pa = _write(tmp_path / "a.tif", a, (0, 4))
    pb = _write(tmp_path / "b.tif", b, (2, 4))
    return pa, pb


class TestMergeMethod:
    """Tests for the ``method=`` overlap rule of ``merge_rasters``."""

    @pytest.mark.parametrize(
        "method, expected_overlap",
        [("last", 20.0), ("first", 10.0), ("min", 10.0), ("max", 20.0), ("sum", 30.0)],
    )
    def test_overlap_resolution(self, overlapping_pair, tmp_path, method, expected_overlap):
        """Each method resolves the overlap strip to the expected value.

        Args:
            method: The merge method under test.
            expected_overlap: The value the overlapping columns should hold.

        Test scenario:
            Columns 2..3 are covered by both A(10) and B(20); the non-overlap
            columns keep their single source's value for every method.
        """
        pa, pb = overlapping_pair
        out = tmp_path / f"out_{method}.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, method=method)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.shape == (4, 6), f"Expected union shape (4, 6), got {arr.shape}"
        assert arr[0, 2] == expected_overlap and arr[0, 3] == expected_overlap, (
            f"{method} overlap should be {expected_overlap}, got {arr[0, 2]} / {arr[0, 3]}"
        )
        assert arr[0, 0] == 10.0, f"A-only column changed: {arr[0, 0]}"
        assert arr[0, 5] == 20.0, f"B-only column changed: {arr[0, 5]}"

    def test_default_method_is_last(self, overlapping_pair, tmp_path):
        """Omitting method defaults to last-wins (backward compatible).

        Test scenario:
            No method argument yields the same overlap as method='last'.
        """
        pa, pb = overlapping_pair
        out = tmp_path / "default.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == 20.0, f"Default should be last-wins (20), got {arr[0, 2]}"

    def test_reduce_fills_uncovered_with_nodata(self, tmp_path):
        """Reduction methods write nodata where no source covers a pixel.

        Test scenario:
            A occupies the top-left 2x2, B the bottom-right 2x2 of a 4x4 union;
            the off-diagonal quadrants are covered by neither and become nodata
            even for 'sum' (which would otherwise yield 0).
        """
        a = np.full((2, 2), 5.0, dtype="float32")
        b = np.full((2, 2), 7.0, dtype="float32")
        pa = _write(tmp_path / "tl.tif", a, (0, 4))
        pb = _write(tmp_path / "br.tif", b, (2, 2))
        out = tmp_path / "gappy.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, method="sum")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 0] == 5.0, f"Top-left should be A=5, got {arr[0, 0]}"
        assert arr[3, 3] == 7.0, f"Bottom-right should be B=7, got {arr[3, 3]}"
        assert arr[0, 3] == -1.0, f"Uncovered top-right should be nodata -1, got {arr[0, 3]}"
        assert arr[3, 0] == -1.0, f"Uncovered bottom-left should be nodata -1, got {arr[3, 0]}"

    def test_reduce_multiband(self, tmp_path):
        """Reduction operates per band on multi-band sources.

        Test scenario:
            Two 2-band rasters fully overlapping: max picks the larger value in
            each band independently.
        """
        a = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 8.0)]).astype("float32")
        b = np.stack([np.full((3, 3), 4.0), np.full((3, 3), 2.0)]).astype("float32")
        pa = _write(tmp_path / "ma.tif", a, (0, 3))
        pb = _write(tmp_path / "mb.tif", b, (0, 3))
        out = tmp_path / "mmax.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, method="max")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 1, 1] == 4.0, f"Band 0 max should be 4, got {arr[0, 1, 1]}"
        assert arr[1, 1, 1] == 8.0, f"Band 1 max should be 8, got {arr[1, 1, 1]}"

    def test_n_ignores_source_value_in_reduction(self, tmp_path):
        """The n knob makes a source pixel value count as no-data in reduction.

        Test scenario:
            A is all 10; B is all 20 but with n=20 ignored, so min over the
            overlap is 10 (B's 20 is excluded), not 10-vs-20.
        """
        a = np.full((4, 4), 10.0, dtype="float32")
        b = np.full((4, 4), 20.0, dtype="float32")
        pa = _write(tmp_path / "na.tif", a, (0, 4), nodata=-9999.0)
        pb = _write(tmp_path / "nb.tif", b, (2, 4), nodata=-9999.0)
        out = tmp_path / "n_min.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, n=20, method="min")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == 10.0, f"Overlap min ignoring 20 should be 10, got {arr[0, 2]}"
        assert arr[0, 5] == -1.0, f"B-only column was all-ignored -> nodata, got {arr[0, 5]}"

    def test_invalid_method_raises(self, overlapping_pair, tmp_path):
        """An unknown method raises ValueError.

        Test scenario:
            'mean' is not a supported merge method.
        """
        pa, pb = overlapping_pair
        with pytest.raises(ValueError, match="method must be one of"):
            merge_rasters([pa, pb], tmp_path / "x.tif", method="mean")

    def test_failed_vrt_zorder_raises(self, overlapping_pair, tmp_path, monkeypatch):
        """A None from BuildVRT on the z-order path raises RuntimeError.

        Test scenario:
            Monkeypatching gdal.BuildVRT to return None triggers the defensive
            guard in the last/first path.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = overlapping_pair
        monkeypatch.setattr(merge_mod.gdal, "BuildVRT", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="gdal.BuildVRT returned None"):
            merge_rasters([pa, pb], tmp_path / "x.tif", method="last")

    def test_failed_vrt_reduce_raises(self, overlapping_pair, tmp_path, monkeypatch):
        """A None from BuildVRT on the reduce path raises RuntimeError.

        Test scenario:
            Monkeypatching gdal.BuildVRT to return None triggers the defensive
            guard inside _merge_reduce.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = overlapping_pair
        monkeypatch.setattr(merge_mod.gdal, "BuildVRT", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="gdal.BuildVRT returned None"):
            merge_rasters([pa, pb], tmp_path / "x.tif", method="sum")


class TestDatasetCollectionMergeMethod:
    """Tests that DatasetCollection.merge threads method through."""

    def test_collection_merge_method(self, overlapping_pair, tmp_path):
        """DatasetCollection.merge forwards method= to merge_rasters.

        Test scenario:
            A file-backed collection of the two overlapping rasters merged with
            method='sum' yields 30 in the overlap.
        """
        from pyramids.dataset.collection import DatasetCollection

        pa, pb = overlapping_pair
        collection = DatasetCollection.read_multiple_files(
            [pa, pb], with_order=False, date=False
        )
        out = tmp_path / "coll_sum.tif"
        collection.merge(out, no_data_value=-9999.0, method="sum")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == 30.0, f"Collection sum overlap should be 30, got {arr[0, 2]}"
