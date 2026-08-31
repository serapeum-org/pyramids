"""Integration tests for Dataset I/O: save, translate, tiling, distributed read, histogram, to_xyz."""

from pathlib import Path
from types import GeneratorType

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal
from pandas import DataFrame

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestSave:
    def test_save_rasters(
        self,
        src: gdal.Dataset,
        save_raster_path: Path,
    ):
        if save_raster_path.exists():
            save_raster_path.unlink()
        src = Dataset(src)
        src.to_file(save_raster_path)
        assert Path(src.file_name) == save_raster_path
        assert save_raster_path.exists()
        src = None
        save_raster_path.unlink()

    def test_save_ascii(
        self,
        src: gdal.Dataset,
        ascii_file_save_to: Path,
    ):
        if ascii_file_save_to.exists():
            ascii_file_save_to.unlink()

        src = Dataset(src)
        src.to_file(ascii_file_save_to)
        assert ascii_file_save_to.exists()
        ascii_file_save_to.unlink()


class TestToFileReopen:
    """Tests for the ``reopen`` flag of ``Dataset.to_file``.

    ``reopen=True`` (default) reopens the written file and swaps it into the
    source in place; ``reopen=False`` writes without that mutation. The
    non-mutating variant is what ``DatasetCollection.to_file`` relies on to
    stream borrowed per-timestep handles without disturbing its cached datasets.
    """

    @staticmethod
    def _mem(arr: np.ndarray) -> Dataset:
        """Build a path-less in-memory Dataset wrapping ``arr``.

        Args:
            arr: 2D or 3D (bands, rows, cols) array to wrap.

        Returns:
            Dataset: A MEM-backed dataset (``file_name == ""``).
        """
        rows = arr.shape[-2]
        return Dataset.from_array(
            arr,
            geo_ref=GeoReference(top_left_corner=(0, rows), cell_size=1.0, epsg=4326),
        )

    def test_reopen_false_leaves_mem_source_unmutated(self, tmp_path: Path):
        """reopen=False must not swap a MEM source's handle to the written file.

        Test scenario:
            A path-less MEM dataset (``file_name == ""``) written with
            ``reopen=False`` stays path-less afterwards, while the output file is
            created and holds the exact pixels.
        """
        arr = np.arange(20, dtype="float32").reshape(4, 5)
        source = self._mem(arr)
        assert source.file_name == "", "precondition: a MEM dataset is path-less"

        out = tmp_path / "unmutated.tif"
        source.to_file(out, reopen=False)

        assert source.file_name == "", (
            f"reopen=False must not repoint the source; got {source.file_name!r}"
        )
        assert out.exists(), f"output file was not written: {out}"
        reloaded = Dataset.read_file(str(out))
        np.testing.assert_allclose(
            reloaded.read_array(),
            arr,
            err_msg="written pixels differ from the source array",
        )

    def test_reopen_true_swaps_mem_source_in_place(self, tmp_path: Path):
        """reopen=True (the default) repoints the source at the freshly written file.

        Test scenario:
            After ``to_file`` with the default flag, ``source.file_name`` equals
            the output path — the documented in-place swap.
        """
        source = self._mem(np.arange(20, dtype="float32").reshape(4, 5))
        out = tmp_path / "swapped.tif"
        source.to_file(out)
        assert Path(source.file_name) == out, (
            f"default reopen=True should swap file_name to {out}, "
            f"got {source.file_name!r}"
        )

    def test_reopen_false_leaves_on_disk_source_file_name(self, tmp_path: Path):
        """reopen=False on an on-disk source keeps its original file_name.

        Test scenario:
            A Dataset opened from ``in.tif`` and written to a different path
            ``out.tif`` with ``reopen=False`` still reports ``in.tif`` as its
            ``file_name`` (the source is not repointed at the new output).
        """
        in_path = tmp_path / "in.tif"
        arr = np.arange(12, dtype="float32").reshape(3, 4)
        self._mem(arr).to_file(in_path)  # anchor an on-disk source (reopen=True)

        source = Dataset.read_file(str(in_path))
        assert Path(source.file_name) == in_path, "precondition: source is on-disk"

        out_path = tmp_path / "out.tif"
        source.to_file(out_path, reopen=False)

        assert Path(source.file_name) == in_path, (
            "reopen=False must leave an on-disk source pointing at its original "
            f"file; got {source.file_name!r}"
        )
        np.testing.assert_allclose(
            Dataset.read_file(str(out_path)).read_array(),
            arr,
            err_msg="output content differs from the source",
        )

    def test_reopen_false_preserves_deflate_and_roundtrips(self, tmp_path: Path):
        """reopen=False still finalises a readable compressed GeoTIFF.

        Test scenario:
            The write path flushes and closes the CreateCopy handle before
            returning even when ``reopen=False``, so the DEFLATE-compressed output
            is complete on disk (regression #570) and a second handle reads back
            the real pixels rather than all-nodata.
        """
        arr = np.zeros((64, 64), dtype="float32")
        arr[10:20, 10:20] = 5.0
        out = tmp_path / "deflate.tif"

        self._mem(arr).to_file(out, reopen=False)

        info = gdal.Info(str(out))
        assert "COMPRESSION=DEFLATE" in info, "expected a DEFLATE-compressed GeoTIFF"
        np.testing.assert_allclose(
            Dataset.read_file(str(out)).read_array(),
            arr,
            err_msg="reopen=False output did not round-trip through a fresh handle",
        )

    def test_reopen_false_multiband_roundtrip(self, tmp_path: Path):
        """reopen=False writes every band of a multiband source faithfully.

        Test scenario:
            A 3-band MEM source written with ``reopen=False`` reloads to an
            identical ``(3, rows, cols)`` array.
        """
        arr = np.arange(3 * 4 * 5, dtype="float32").reshape(3, 4, 5)
        out = tmp_path / "multiband.tif"

        self._mem(arr).to_file(out, reopen=False)

        reloaded = Dataset.read_file(str(out)).read_array()
        assert reloaded.shape == (3, 4, 5), f"unexpected shape {reloaded.shape}"
        np.testing.assert_allclose(
            reloaded, arr, err_msg="multiband content differs after reopen=False"
        )

    def test_reopen_false_ignored_for_ascii(self, tmp_path: Path):
        """reopen has no effect for the ASCII driver (which never reopens).

        Test scenario:
            ``to_file(..., driver="ascii", reopen=False)`` writes the file without
            error — the ASCII branch does not go through the reopen/swap path, so
            the flag is a documented no-op there.
        """
        out = tmp_path / "grid.asc"
        source = self._mem(np.ones((3, 3), dtype="float32"))

        source.to_file(out, driver="ascii", reopen=False)

        assert out.exists(), f"ASCII output was not written: {out}"

    def test_reopen_false_forwarded_through_compute_false(self, tmp_path: Path):
        """The compute=False (dask.delayed) path forwards reopen=False.

        Test scenario:
            A deferred write of an on-disk source with ``reopen=False`` writes the
            output on ``.compute()`` and leaves the source's ``file_name``
            unchanged — the delayed branch threads the flag through to the
            synchronous writer.
        """
        pytest.importorskip("dask")
        in_path = tmp_path / "src.tif"
        arr = np.arange(12, dtype="float32").reshape(3, 4)
        self._mem(arr).to_file(in_path)
        source = Dataset.read_file(str(in_path))

        out = tmp_path / "deferred.tif"
        delayed = source.to_file(out, reopen=False, compute=False)
        assert delayed is not None, "compute=False should return a Delayed"
        delayed.compute()

        assert out.exists(), f"deferred output was not written: {out}"
        assert Path(source.file_name) == in_path, (
            "reopen=False through compute=False must not repoint the source"
        )


class TestNCtoGeoTIFF:
    def test_convert_0_360_to_180_180_longitude_new_dataset(self, noah: gdal.Dataset):
        dataset = Dataset(noah)
        new_dataset = dataset.wrap_longitude()
        lon = new_dataset.lon
        assert lon.max() < 180
        assert new_dataset.top_left_corner == (-180, 90)

    def test_convert_0_360_to_180_180_longitude_inplace(self, noah: gdal.Dataset):
        dataset = Dataset(noah)
        dataset = dataset.wrap_longitude()
        lon = dataset.lon
        assert lon.max() < 180
        assert dataset.top_left_corner == (-180, 90)


class TestTiling:
    def test_window(self, raster_1band_coello_path):
        dataset = Dataset.read_file(raster_1band_coello_path)
        tiles_details = dataset.io._tile_offsets(size=6)
        assert isinstance(tiles_details, GeneratorType)
        tiles_details_l = list(tiles_details)
        assert tiles_details_l == [
            (0, 0, 6, 6),
            (6, 0, 6, 6),
            (12, 0, 2, 6),
            (0, 6, 6, 6),
            (6, 6, 6, 6),
            (12, 6, 2, 6),
            (0, 12, 6, 1),
            (6, 12, 6, 1),
            (12, 12, 2, 1),
        ]


class TestDistributedRead:  # unittest.TestCase
    def test_get_block_arrangement_default(self, src: Dataset):
        dataset = Dataset(src)
        dataset.block_size = [[5, 5]]
        df = dataset.get_block_arrangement()

        # Check if the DataFrame is correct
        expected_df = pd.DataFrame(
            [
                {"x_offset": 0, "y_offset": 0, "window_xsize": 5, "window_ysize": 5},
                {"x_offset": 5, "y_offset": 0, "window_xsize": 5, "window_ysize": 5},
                {"x_offset": 10, "y_offset": 0, "window_xsize": 4, "window_ysize": 5},
                {"x_offset": 0, "y_offset": 5, "window_xsize": 5, "window_ysize": 5},
                {"x_offset": 5, "y_offset": 5, "window_xsize": 5, "window_ysize": 5},
                {"x_offset": 10, "y_offset": 5, "window_xsize": 4, "window_ysize": 5},
                {"x_offset": 0, "y_offset": 10, "window_xsize": 5, "window_ysize": 3},
                {"x_offset": 5, "y_offset": 10, "window_xsize": 5, "window_ysize": 3},
                {"x_offset": 10, "y_offset": 10, "window_xsize": 4, "window_ysize": 3},
                # Add more rows as needed to fully test all cases
            ],
            columns=["x_offset", "y_offset", "window_xsize", "window_ysize"],
        )

        pd.testing.assert_frame_equal(df, expected_df)


class TestHistogram:
    def test_get_histogram(self, src: gdal.Dataset):
        dataset = Dataset(src)
        hist, ranges = dataset.get_histogram(band=0)
        assert len(ranges) == 6
        assert hist == [75, 6, 0, 4, 2, 1]


def test_to_xyz():
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    top_left_corner = (0, 0)
    cell_size = 0.05
    dataset = Dataset.from_array(
        arr,
        geo_ref=GeoReference(
            top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
        ),
    )
    # test with default parameters
    df = dataset.to_xyz()
    check_df = DataFrame(
        {
            "lon": [0.025, 0.075, 0.025, 0.075],
            "lat": [-0.025, -0.025, -0.075, -0.075],
            "Band_1": [1, 2, 3, 4],
            "Band_2": [5, 6, 7, 8],
        }
    )

    pd.testing.assert_frame_equal(df, check_df)
    # test with one bands as integer
    df = dataset.to_xyz(bands=0)
    pd.testing.assert_frame_equal(df, check_df.loc[:, ["lon", "lat", "Band_1"]])

    # test with one band as integer
    df = dataset.to_xyz(bands=[1])
    pd.testing.assert_frame_equal(df, check_df.loc[:, ["lon", "lat", "Band_2"]])
    with pytest.raises(ValueError):
        dataset.to_xyz(bands="1")


class TestTranslate:
    def test_scale(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(1, 10, size=(5, 5)).astype(np.float32)
        top_left_corner = (0, 0)
        cell_size = 0.05
        dataset = Dataset.from_array(
            arr,
            geo_ref=GeoReference(
                top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
            ),
        )
        dataset.scale = [0.1]
        unscaled_dataset = dataset.translate(unscale=True)
        unscaled_arr = unscaled_dataset.read_array()
        np.testing.assert_almost_equal(unscaled_arr, arr * 0.1)
