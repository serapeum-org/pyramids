"""Unit tests for DatasetCollection methods that lack coverage.

Targets untested / low-coverage code paths in
``pyramids.dataset.collection``, including:
- ``create_cube`` classmethod
- ``merge`` static method (via a temp-file round-trip)
- ``apply`` method with ufunc
- ``overlay`` with classes
- ``__iter__``, ``head``, ``tail``, ``first``, ``last``
- ``to_file`` with string path and list of paths
- ``values`` deleter
- ``__str__`` / ``__repr__``
- ``shape`` property
- Error paths for ``__getitem__``, ``__setitem__``, ``open_multi_dataset``
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset, DatasetCollection


def _make_mem_dataset(
    rows: int = 5,
    cols: int = 6,
    epsg: int = 4326,
    no_data: float = -9999.0,
    fill_value: float = 1.0,
) -> Dataset:
    """Create a minimal in-memory Dataset filled with ``fill_value``."""
    src = Dataset.create(
        cell_size=1.0,
        rows=rows,
        columns=cols,
        dtype="float32",
        bands=1,
        top_left_corner=(0.0, float(rows)),
        epsg=epsg,
        no_data_value=no_data,
    )
    arr = np.full((rows, cols), fill_value, dtype=np.float32)
    src.raster.GetRasterBand(1).WriteArray(arr)
    return src


@pytest.fixture()
def base_dataset() -> Dataset:
    """A small 5x6 in-memory Dataset."""
    return _make_mem_dataset()


@pytest.fixture()
def cube_with_values(base_dataset: Dataset) -> DatasetCollection:
    """A DatasetCollection with 3 time steps and pre-set values."""
    md = DatasetCollection.create_cube(base_dataset, dataset_length=3)
    values = np.arange(3 * 5 * 6, dtype=np.float64).reshape(3, 5, 6)
    md.values = values
    return md


class TestCreateCube:
    """Tests for the ``create_cube`` classmethod."""

    def test_returns_dataset_collection(self, base_dataset: Dataset):
        """create_cube should return a DatasetCollection instance."""
        md = DatasetCollection.create_cube(base_dataset, dataset_length=4)
        assert isinstance(
            md, DatasetCollection
        ), f"Expected DatasetCollection, got {type(md)}"

    def test_time_length_matches(self, base_dataset: Dataset):
        """The time_length should match the given dataset_length."""
        md = DatasetCollection.create_cube(base_dataset, dataset_length=7)
        assert md.time_length == 7, f"Expected time_length=7, got {md.time_length}"

    def test_base_is_same_dataset(self, base_dataset: Dataset):
        """The base property should reference the provided Dataset."""
        md = DatasetCollection.create_cube(base_dataset, dataset_length=1)
        assert md.base is base_dataset, "base should be the original Dataset"

    def test_files_is_none(self, base_dataset: Dataset):
        """create_cube does not set files so it should be None."""
        md = DatasetCollection.create_cube(base_dataset, dataset_length=2)
        assert md.files is None, "files should be None for create_cube"


class TestStringRepresentation:
    """Tests for __str__ and __repr__."""

    def test_str_contains_epsg(self, base_dataset: Dataset):
        """String representation should mention the EPSG code."""
        md = DatasetCollection(base_dataset, time_length=2, files=["a.tif", "b.tif"])
        text = str(md)
        assert "EPSG" in text, "__str__ should contain 'EPSG'"

    def test_repr_contains_dimension(self, base_dataset: Dataset):
        """Repr should contain dimension info."""
        md = DatasetCollection(base_dataset, time_length=2, files=["a.tif", "b.tif"])
        text = repr(md)
        assert "Dimension" in text, "__repr__ should contain 'Dimension'"

    def test_str_works_without_files(self, base_dataset: Dataset):
        """H1 regression: __str__ must not TypeError when files=None.

        Pre-fix `len(self.files)` raised on collections built without
        a `files=` argument (legacy in-memory constructions, anything
        produced by `crop(inplace=False)` / `apply()`). Post-fix the
        source line falls back to `time_length` and labels the cube
        as in-memory.
        """
        md = DatasetCollection(base_dataset, time_length=2)
        assert md.files is None
        text = str(md)
        assert (
            "in-memory" in text
        ), f"in-memory cube should label itself as such; got: {text}"
        assert "Time length: 2" in text

    def test_repr_works_without_files(self, base_dataset: Dataset):
        """H1 regression: same as __str__ but for __repr__."""
        md = DatasetCollection(base_dataset, time_length=2)
        text = repr(md)
        assert "in-memory" in text
        assert "Time length: 2" in text


class TestShapeProperties:
    """Tests for shape, rows, columns."""

    def test_shape(self, cube_with_values: DatasetCollection):
        """shape should be (time_length, rows, columns)."""
        expected = (3, 5, 6)
        assert (
            cube_with_values.shape == expected
        ), f"Expected shape {expected}, got {cube_with_values.shape}"

    def test_rows(self, cube_with_values: DatasetCollection):
        """rows should match the base dataset."""
        assert (
            cube_with_values.rows == 5
        ), f"Expected rows=5, got {cube_with_values.rows}"

    def test_columns(self, cube_with_values: DatasetCollection):
        """columns should match the base dataset."""
        assert (
            cube_with_values.columns == 6
        ), f"Expected columns=6, got {cube_with_values.columns}"


class TestIterationMethods:
    """Tests for __iter__, head, tail, first, last."""

    def test_iter_count(self, cube_with_values: DatasetCollection):
        """Iterating should yield time_length 2D arrays."""
        items = list(cube_with_values)
        assert len(items) == 3, f"Expected 3 items, got {len(items)}"
        for item in items:
            assert item.shape == (
                5,
                6,
            ), f"Each iterated slice should be (5,6), got {item.shape}"

    def test_head_default(self, cube_with_values: DatasetCollection):
        """head() with default n=5 should clamp to available time steps."""
        result = cube_with_values.head()
        assert result.shape[0] == 3, "head(5) on a cube with 3 steps should return 3"

    def test_head_custom(self, cube_with_values: DatasetCollection):
        """head(2) should return the first 2 time steps."""
        result = cube_with_values.head(n=2)
        assert result.shape == (2, 5, 6), f"Expected (2,5,6), got {result.shape}"

    def test_tail_default(self, cube_with_values: DatasetCollection):
        """tail() with default n=-5 should clamp to available time steps."""
        result = cube_with_values.tail()
        assert result.shape[0] == 3, "tail(-5) on a cube with 3 steps should return 3"

    def test_tail_custom(self, cube_with_values: DatasetCollection):
        """tail(-1) should return the last time step only."""
        result = cube_with_values.tail(n=-1)
        assert result.shape == (1, 5, 6), f"Expected (1,5,6), got {result.shape}"

    def test_first(self, cube_with_values: DatasetCollection):
        """first() should return the first time slice (2D array)."""
        result = cube_with_values.first()
        assert result.shape == (5, 6), f"Expected (5,6), got {result.shape}"
        expected_first = np.arange(3 * 5 * 6, dtype=np.float64).reshape(3, 5, 6)[0]
        np.testing.assert_array_equal(result, expected_first)

    def test_last(self, cube_with_values: DatasetCollection):
        """last() should return the final time slice (2D array)."""
        result = cube_with_values.last()
        assert result.shape == (5, 6), f"Expected (5,6), got {result.shape}"
        expected_last = np.arange(3 * 5 * 6, dtype=np.float64).reshape(3, 5, 6)[-1]
        np.testing.assert_array_equal(result, expected_last)


class TestItemAccess:
    """Tests for __getitem__, __setitem__, __len__."""

    def test_len(self, cube_with_values: DatasetCollection):
        """len() should return the number of time steps."""
        assert (
            len(cube_with_values) == 3
        ), f"Expected len=3, got {len(cube_with_values)}"

    def test_getitem(self, cube_with_values: DatasetCollection):
        """Indexing should return a 2D slice."""
        result = cube_with_values[1]
        assert result.shape == (5, 6), f"Expected (5,6), got {result.shape}"

    def test_setitem(self, cube_with_values: DatasetCollection):
        """Setting a slice should update the values."""
        new_arr = np.ones((5, 6), dtype=np.float64) * 999
        cube_with_values[0] = new_arr
        np.testing.assert_array_equal(
            cube_with_values[0], new_arr, err_msg="__setitem__ did not update the array"
        )


class TestValuesSetter:
    """Tests for the values setter dimension check."""

    def test_correct_dimensions(self, cube_with_values: DatasetCollection):
        """Setting values with the same shape should succeed."""
        new_arr = np.zeros((3, 5, 6), dtype=np.float64)
        cube_with_values.values = new_arr
        np.testing.assert_array_equal(
            cube_with_values.values,
            new_arr,
            err_msg="Values setter should accept same-shape array",
        )

    def test_wrong_dimensions_raises(self, cube_with_values: DatasetCollection):
        """Setting values with a different shape should raise ValueError."""
        wrong_arr = np.zeros((2, 5, 6), dtype=np.float64)
        with pytest.raises(ValueError, match="differs from the dimension"):
            cube_with_values.values = wrong_arr


class TestApply:
    """Tests for ``apply`` method with ufunc."""

    def test_apply_numpy_ufunc(self, base_dataset: Dataset):
        """apply with np.abs should process all non-nodata cells.

        After the L-3 refactor ``apply`` is out-of-place — returns a
        new ``DatasetCollection`` instead of mutating ``self``. The
        assertion runs against the returned collection's ``values``.
        """
        md = DatasetCollection.create_cube(base_dataset, dataset_length=2)
        values = np.full((2, 5, 6), -5.0)
        values[:, 0, -1] = -9999.0  # set nodata in one cell
        md.values = values
        result = md.apply(np.abs)
        # Non-nodata cells should now be positive in the returned collection.
        non_nodata = result.values[:, :, :-1]
        assert np.all(
            non_nodata >= 0
        ), "All non-nodata values should be positive after np.abs"

    def test_apply_custom_ufunc(self, base_dataset: Dataset):
        """apply with a custom function via np.frompyfunc.

        After the L-3 refactor ``apply`` is out-of-place — see
        :meth:`test_apply_numpy_ufunc`.
        """
        md = DatasetCollection.create_cube(base_dataset, dataset_length=2)
        values = np.full((2, 5, 6), 10.0)
        values[:, 0, 0] = -9999.0
        md.values = values
        double_fn = np.frompyfunc(lambda x: x * 2, 1, 1)
        result = md.apply(double_fn)
        assert (
            result.values[0, 1, 0] == pytest.approx(20.0)
        ), f"Expected 20.0 after doubling, got {result.values[0, 1, 0]}"

    def test_apply_non_callable_raises(self, cube_with_values: DatasetCollection):
        """apply with a non-callable argument should raise TypeError."""
        with pytest.raises(TypeError, match="should be a function"):
            cube_with_values.apply("not_a_function")


class TestToFile:
    """Tests for ``to_file`` with string path and list of paths."""

    def test_to_file_with_directory_path(self, cube_with_values: DatasetCollection):
        """to_file with a directory path should create one file per time step."""
        tmp_dir = Path(tempfile.mkdtemp())
        out_dir = tmp_dir / "output_rasters"
        try:
            cube_with_values.to_file(out_dir)
            files = list(out_dir.iterdir())
            assert len(files) == 3, f"Expected 3 files, got {len(files)}"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_to_file_with_list_of_paths(self, cube_with_values: DatasetCollection):
        """to_file with a list of paths should write to each path."""
        tmp_dir = Path(tempfile.mkdtemp())
        sub_dir = tmp_dir / "sub"
        sub_dir.mkdir(exist_ok=True)
        paths = [sub_dir / f"raster_{i}.tif" for i in range(3)]
        try:
            cube_with_values.to_file(paths)
            for p in paths:
                assert p.exists(), f"Expected file at {p}"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_to_file_wrong_list_length_raises(
        self, cube_with_values: DatasetCollection
    ):
        """to_file with a list whose length != time_length should raise ValueError."""
        with pytest.raises(ValueError, match="does not equal"):
            cube_with_values.to_file(["a.tif", "b.tif"])


class TestIloc:
    """Tests for iloc method."""

    def test_returns_dataset(self, cube_with_values: DatasetCollection):
        """iloc should return a Dataset."""
        ds = cube_with_values.iloc(0)
        assert isinstance(ds, Dataset), f"Expected Dataset, got {type(ds)}"

    def test_iloc_array_matches_values(self, cube_with_values: DatasetCollection):
        """The array from iloc should match the corresponding slice."""
        ds = cube_with_values.iloc(1)
        arr = ds.read_array()
        expected = cube_with_values.values[1, :, :]
        np.testing.assert_array_almost_equal(
            arr, expected, decimal=4, err_msg="iloc array should match values slice"
        )


import datetime as dt
import re

from pyramids.base._errors import DatasetNotFoundError

pytestmark = pytest.mark.core


class TestReadMultipleFilesErrors:
    """Tests for error paths in ``read_multiple_files``."""

    def test_invalid_path_type_raises_type_error(self):
        """Passing a non-string/non-list path should raise TypeError."""
        with pytest.raises(TypeError, match="string/Path/list type"):
            DatasetCollection.read_multiple_files(12345)

    def test_nonexistent_path_raises_file_not_found(self, tmp_path):
        """Passing a non-existent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            DatasetCollection.read_multiple_files(tmp_path / "nonexistent_dir")

    def test_empty_directory_raises_file_not_found(self, tmp_path):
        """A directory with no .tif files should raise FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir(exist_ok=True)
        with pytest.raises(FileNotFoundError, match="empty"):
            DatasetCollection.read_multiple_files(empty_dir)

    def test_read_from_list_of_files(self, tmp_path):
        """Reading from a pre-built list of file paths should work."""
        # create two small GeoTIFF files
        src = _make_mem_dataset(rows=3, cols=3)
        paths = []
        for i in range(2):
            p = str(tmp_path / f"raster_{i}.tif")
            src.to_file(p)
            paths.append(p)
        md = DatasetCollection.read_multiple_files(paths)
        assert md.time_length == 2, f"Expected time_length=2, got {md.time_length}"
        assert md.files == paths, "files should match the input list"

    def test_with_order_date_mismatch_raises(self, tmp_path):
        """Filenames that don't match regex should raise ValueError."""
        src = _make_mem_dataset(rows=3, cols=3)
        paths = []
        for name in ["no_date_a.tif", "no_date_b.tif"]:
            p = str(tmp_path / name)
            src.to_file(p)
            paths.append(p)
        with pytest.raises(ValueError, match="does not match"):
            DatasetCollection.read_multiple_files(
                paths,
                with_order=True,
                regex_string=r"\d{4}\.\d{2}\.\d{2}",
            )

    def test_with_order_missing_fmt_raises(self, tmp_path):
        """Setting with_order=True and date=True without file_name_data_fmt raises."""
        src = _make_mem_dataset(rows=3, cols=3)
        dir_path = tmp_path / "ordered_rasters"
        dir_path.mkdir(exist_ok=True)
        for name in ["2020.01.01.tif", "2020.01.02.tif"]:
            src.to_file(dir_path / name)
        with pytest.raises(ValueError, match="file_name_data_fmt"):
            DatasetCollection.read_multiple_files(
                dir_path,
                with_order=True,
                date=True,
                regex_string=r"\d{4}\.\d{2}\.\d{2}",
                file_name_data_fmt=None,
            )

    def test_with_order_numeric_no_date(self, tmp_path):
        """Reading with_order=True and date=False sorts by numeric match."""
        src = _make_mem_dataset(rows=3, cols=3)
        dir_path = tmp_path / "numeric_rasters"
        dir_path.mkdir(exist_ok=True)
        for name in ["3_raster.tif", "1_raster.tif", "2_raster.tif"]:
            src.to_file(dir_path / name)
        md = DatasetCollection.read_multiple_files(
            dir_path,
            with_order=True,
            date=False,
            regex_string=r"\d+",
        )
        assert md.time_length == 3, f"Expected time_length=3, got {md.time_length}"

    def test_with_order_numeric_start_end_filter(self, tmp_path):
        """Numeric ordering with start/end should filter files."""
        src = _make_mem_dataset(rows=3, cols=3)
        dir_path = tmp_path / "filter_rasters"
        dir_path.mkdir(exist_ok=True)
        for name in [
            "1_raster.tif",
            "2_raster.tif",
            "3_raster.tif",
            "4_raster.tif",
        ]:
            src.to_file(dir_path / name)
        md = DatasetCollection.read_multiple_files(
            dir_path,
            with_order=True,
            date=False,
            regex_string=r"\d+",
            start=2,
            end=3,
        )
        assert (
            md.time_length == 2
        ), f"Expected time_length=2 after filtering, got {md.time_length}"


# Tests for "without values raises" deleted: after the L-3 refactor
# the collection's per-timestep handles open lazily on first
# access, so iloc/__getitem__/__setitem__ no longer need an upfront
# `open_multi_dataset()` call. The error path those tests asserted
# no longer exists.


class TestAlignErrors:
    """Tests for align method error path."""

    def test_non_dataset_alignment_src_raises(
        self, cube_with_values: DatasetCollection
    ):
        """Passing a non-Dataset as alignment_src should raise TypeError."""
        with pytest.raises(TypeError, match="Dataset object"):
            cube_with_values.align("not_a_dataset")
