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

import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import AlignmentError, OptionalPackageDoesNotExist
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.collection import _target_epsg
from tests.dataset.collection._helpers import make_int16_collection


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
        assert isinstance(md, DatasetCollection), (
            f"Expected DatasetCollection, got {type(md)}"
        )

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
        assert "in-memory" in text, (
            f"in-memory cube should label itself as such; got: {text}"
        )
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
        assert cube_with_values.shape == expected, (
            f"Expected shape {expected}, got {cube_with_values.shape}"
        )

    def test_rows(self, cube_with_values: DatasetCollection):
        """rows should match the base dataset."""
        assert cube_with_values.rows == 5, (
            f"Expected rows=5, got {cube_with_values.rows}"
        )

    def test_columns(self, cube_with_values: DatasetCollection):
        """columns should match the base dataset."""
        assert cube_with_values.columns == 6, (
            f"Expected columns=6, got {cube_with_values.columns}"
        )


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
        assert len(cube_with_values) == 3, (
            f"Expected len=3, got {len(cube_with_values)}"
        )

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
        assert np.all(non_nodata >= 0), (
            "All non-nodata values should be positive after np.abs"
        )

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
        assert result.values[0, 1, 0] == pytest.approx(20.0), (
            f"Expected 20.0 after doubling, got {result.values[0, 1, 0]}"
        )

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
        assert md.time_length == 2, (
            f"Expected time_length=2 after filtering, got {md.time_length}"
        )


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


def _write_geotiff(
    path,
    arr: np.ndarray,
    *,
    top_left: tuple = (0.0, 4.0),
    cell_size: float = 1.0,
    epsg: int = 4326,
    no_data: float = -9999.0,
) -> str:
    """Write ``arr`` to a GeoTIFF at ``path`` and return the path as a string."""
    Dataset.create_from_array(
        arr,
        top_left_corner=top_left,
        cell_size=cell_size,
        epsg=epsg,
        no_data_value=no_data,
        path=str(path),
    ).close()
    return str(path)


class TestMemDatasetFromArray:
    """Tests for ``_mem_dataset_from_array`` (ARC-70)."""

    @pytest.fixture()
    def collection(self) -> DatasetCollection:
        """A single-timestep in-memory collection whose base is float32."""
        base = Dataset.create_from_array(
            np.ones((3, 4), dtype="float32"),
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        return DatasetCollection(base, time_length=1)

    def test_default_source_uses_base_georef(self, collection: DatasetCollection):
        """With no ``source`` the result inherits the base's geotransform + EPSG.

        Test scenario:
            Wrap an array without ``source`` — expected: geotransform equals the
            base template's and the EPSG is the base's 4326.
        """
        arr = np.arange(12, dtype="float64").reshape(3, 4)
        result = collection._mem_dataset_from_array(arr)
        assert result.geotransform == pytest.approx(collection.base.geotransform), (
            f"geotransform not copied from base: {result.geotransform}"
        )
        assert result.epsg == 4326, f"expected epsg 4326, got {result.epsg}"

    def test_preserves_input_array_dtype(self, collection: DatasetCollection):
        """A float64 input is not down-cast through the float32 base dtype.

        Test scenario:
            Wrap a float64 array over a float32 base — expected: the result reads
            back as float64 (the base dtype does not silently round it).
        """
        arr = np.arange(12, dtype="float64").reshape(3, 4)
        result = collection._mem_dataset_from_array(arr)
        assert result.read_array().dtype == np.float64, (
            f"input dtype not preserved: {result.read_array().dtype}"
        )

    def test_values_round_trip(self, collection: DatasetCollection):
        """The wrapped array reads back element-for-element.

        Test scenario:
            Wrap a non-trivial float64 array — expected: ``read_array`` returns
            the same values.
        """
        arr = np.arange(12, dtype="float64").reshape(3, 4) * 1.5
        result = collection._mem_dataset_from_array(arr)
        np.testing.assert_array_equal(
            result.read_array(), arr, err_msg="values not preserved by wrap"
        )

    def test_explicit_source_overrides_base_georef(self, collection: DatasetCollection):
        """Passing ``source=`` copies that dataset's georef, not the base's.

        Test scenario:
            Wrap an array with an explicit ``source`` on a different grid —
            expected: geotransform matches the source, not the collection base.
        """
        other = Dataset.create_from_array(
            np.ones((3, 4), dtype="float32"),
            top_left_corner=(100.0, 50.0),
            cell_size=2.0,
            epsg=4326,
            no_data_value=-1.0,
        )
        arr = np.zeros((3, 4), dtype="float32")
        result = collection._mem_dataset_from_array(arr, source=other)
        assert result.geotransform == pytest.approx(other.geotransform), (
            f"source georef not used: {result.geotransform}"
        )
        assert result.geotransform != pytest.approx(collection.base.geotransform), (
            "result should not carry the base georef when source is given"
        )


class TestRequireFiles:
    """Tests for the ``_require_files`` file-backed guard (ARC-70)."""

    def test_returns_files_list_for_file_backed(self, tmp_path):
        """A file-backed collection returns its own ``files`` list unchanged.

        Test scenario:
            Guard a file-backed collection — expected: the live ``files`` list is
            returned (same object), not a copy.
        """
        col, paths = make_int16_collection(tmp_path, count=2)
        result = col._require_files("to_zarr")
        assert result == paths, f"expected {paths}, got {result}"
        assert result is col.files, "should return the live files list, not a copy"

    def test_none_files_raises_naming_method(self, base_dataset: Dataset):
        """An in-memory collection (files=None) raises RuntimeError naming the method.

        Test scenario:
            Guard an in-memory collection — expected: RuntimeError whose message
            names the offending method and mentions ``file-backed``.
        """
        col = DatasetCollection(base_dataset, time_length=2)
        with pytest.raises(RuntimeError, match="to_kerchunk") as exc:
            col._require_files("to_kerchunk")
        assert "file-backed" in str(exc.value), f"unexpected message: {exc.value}"

    def test_empty_files_list_raises(self, base_dataset: Dataset):
        """An empty files list is guarded the same as None (the len == 0 branch).

        Test scenario:
            Guard a collection built with ``files=[]`` — expected: RuntimeError
            naming the method.
        """
        col = DatasetCollection(base_dataset, time_length=0, files=[])
        with pytest.raises(RuntimeError, match="to_zarr"):
            col._require_files("to_zarr")


class TestTailHeadRegression:
    """Tests for the ARC-46 head/tail fix and empty-safe ``_stack_band0``."""

    @pytest.fixture()
    def expected(self) -> np.ndarray:
        """The (3, 5, 6) cube backing the ``cube_with_values`` fixture."""
        return np.arange(3 * 5 * 6, dtype=np.float64).reshape(3, 5, 6)

    def test_tail_positive_n_returns_last_n(
        self, cube_with_values: DatasetCollection, expected: np.ndarray
    ):
        """tail(2) returns the LAST 2 timesteps (ARC-46: no longer skips the first n).

        Test scenario:
            ``tail(2)`` on a 3-step cube — expected: shape (2, 5, 6) equal to the
            last two source slices, not the tail-after-skipping-2 single slice.
        """
        result = cube_with_values.tail(2)
        assert result.shape == (2, 5, 6), f"expected (2,5,6), got {result.shape}"
        np.testing.assert_array_equal(
            result, expected[1:], err_msg="tail(2) is not the last 2 timesteps"
        )

    def test_tail_both_signs_equal(self, cube_with_values: DatasetCollection):
        """tail(3) == tail(-3): the sign of ``n`` is ignored.

        Test scenario:
            Compare positive and negative ``n`` — expected: identical arrays.
        """
        np.testing.assert_array_equal(
            cube_with_values.tail(3),
            cube_with_values.tail(-3),
            err_msg="tail(n) and tail(-n) disagree",
        )

    def test_tail_clamps_to_time_length(
        self, cube_with_values: DatasetCollection, expected: np.ndarray
    ):
        """tail(99) clamps to all available timesteps.

        Test scenario:
            ``abs(n)`` larger than ``time_length`` — expected: the full cube.
        """
        result = cube_with_values.tail(99)
        assert result.shape == (3, 5, 6), f"expected full cube, got {result.shape}"
        np.testing.assert_array_equal(result, expected)

    def test_tail_zero_is_empty(self, cube_with_values: DatasetCollection):
        """tail(0) returns an empty (0, rows, cols) cube, not a stack error.

        Test scenario:
            ``n == 0`` — expected: a (0, 5, 6) array via the empty-safe path.
        """
        result = cube_with_values.tail(0)
        assert result.shape == (0, 5, 6), f"expected (0,5,6), got {result.shape}"

    def test_head_zero_is_empty(self, cube_with_values: DatasetCollection):
        """head(0) returns an empty (0, rows, cols) cube, not a stack error.

        Test scenario:
            ``n == 0`` — expected: a (0, 5, 6) array via the empty-safe path.
        """
        result = cube_with_values.head(0)
        assert result.shape == (0, 5, 6), f"expected (0,5,6), got {result.shape}"

    def test_stack_band0_empty_selection(self, cube_with_values: DatasetCollection):
        """_stack_band0([]) returns a (0, rows, cols) array (empty-safe).

        Test scenario:
            Stack an empty selection — expected: a (0, 5, 6) array instead of the
            ``np.stack`` "need at least one array" error.
        """
        result = cube_with_values._stack_band0([])
        assert result.shape == (0, 5, 6), f"expected (0,5,6), got {result.shape}"

    def test_stack_band0_non_empty(
        self, cube_with_values: DatasetCollection, expected: np.ndarray
    ):
        """_stack_band0 over all datasets reproduces the full cube.

        Test scenario:
            Stack every timestep's band 0 — expected: the (3, 5, 6) source cube.
        """
        result = cube_with_values._stack_band0(cube_with_values.datasets)
        assert result.shape == (3, 5, 6), f"expected (3,5,6), got {result.shape}"
        np.testing.assert_array_equal(result, expected)


class TestFromFilesValidate:
    """Tests for ``from_files(validate=...)`` + ``_validate_headers`` (ARC-75b)."""

    def test_validate_true_matching_files_succeeds(self, tmp_path):
        """Homogeneous files pass validation and build the collection.

        Test scenario:
            Two same-shape / same-dtype int16 files with ``validate=True`` —
            expected: a 2-timestep collection, no error.
        """
        t0 = _write_geotiff(
            tmp_path / "t0.tif", np.arange(20, dtype="int16").reshape(4, 5)
        )
        t1 = _write_geotiff(
            tmp_path / "t1.tif", np.arange(20, dtype="int16").reshape(4, 5) + 5
        )
        col = DatasetCollection.from_files([t0, t1], validate=True)
        assert col.time_length == 2, f"expected 2 timesteps, got {col.time_length}"

    def test_validate_true_shape_mismatch_raises(self, tmp_path):
        """A file with a different (rows, cols) raises AlignmentError naming it.

        Test scenario:
            Template (4, 5) + a (3, 5) file with ``validate=True`` — expected:
            AlignmentError whose message names the offending path.
        """
        t0 = _write_geotiff(tmp_path / "t0.tif", np.zeros((4, 5), dtype="int16"))
        bad = _write_geotiff(tmp_path / "bad.tif", np.zeros((3, 5), dtype="int16"))
        with pytest.raises(AlignmentError, match="bad.tif") as exc:
            DatasetCollection.from_files([t0, bad], validate=True)
        assert "does not match" in str(exc.value), f"unexpected message: {exc.value}"

    def test_validate_true_dtype_mismatch_raises(self, tmp_path):
        """A file with a different dtype raises AlignmentError naming it.

        Test scenario:
            Template int16 + a float64 file (same shape) with ``validate=True`` —
            expected: AlignmentError naming the offending path.
        """
        t0 = _write_geotiff(tmp_path / "t0.tif", np.zeros((4, 5), dtype="int16"))
        bad = _write_geotiff(tmp_path / "bad.tif", np.zeros((4, 5), dtype="float64"))
        with pytest.raises(AlignmentError, match="bad.tif"):
            DatasetCollection.from_files([t0, bad], validate=True)

    def test_validate_false_does_not_open_other_files(self, tmp_path, monkeypatch):
        """validate=False (default) opens only the first file, never the rest.

        Test scenario:
            Spy on ``Dataset.read_file`` — expected: only the template path is
            opened; the second file is left untouched.
        """
        t0 = _write_geotiff(tmp_path / "t0.tif", np.zeros((4, 5), dtype="int16"))
        t1 = _write_geotiff(tmp_path / "t1.tif", np.zeros((4, 5), dtype="int16"))
        opened: list[str] = []
        orig = Dataset.read_file

        def spy(path, *args, **kwargs):
            opened.append(str(path))
            return orig(path, *args, **kwargs)

        monkeypatch.setattr(Dataset, "read_file", staticmethod(spy))
        DatasetCollection.from_files([t0, t1], validate=False)
        assert t0 in opened, f"template file was not opened: {opened}"
        assert t1 not in opened, f"validate=False opened a non-template file: {opened}"

    def test_validate_true_opens_every_file(self, tmp_path, monkeypatch):
        """validate=True opens every file's header (the opt-in cost).

        Test scenario:
            Spy on ``Dataset.read_file`` with ``validate=True`` — expected: the
            second file is opened too.
        """
        t0 = _write_geotiff(tmp_path / "t0.tif", np.zeros((4, 5), dtype="int16"))
        t1 = _write_geotiff(tmp_path / "t1.tif", np.zeros((4, 5), dtype="int16"))
        opened: list[str] = []
        orig = Dataset.read_file

        def spy(path, *args, **kwargs):
            opened.append(str(path))
            return orig(path, *args, **kwargs)

        monkeypatch.setattr(Dataset, "read_file", staticmethod(spy))
        DatasetCollection.from_files([t0, t1], validate=True)
        assert t1 in opened, f"validate=True did not open the second file: {opened}"


class TestDatasetAtLazyHandles:
    """Tests for ``_dataset_at`` + the per-index ``_handle_cache`` (ARC-44)."""

    def test_first_opens_only_one_file(self, tmp_path):
        """``first()`` on a file-backed cube opens one file, not all N.

        Test scenario:
            A 3-file collection, then ``first()`` — expected: the bulk
            ``_datasets`` list is still ``None`` and exactly one handle is
            cached (only index 0 was opened).
        """
        col, _ = make_int16_collection(tmp_path, count=3)
        result = col.first()
        assert col._datasets is None, "first() must not materialise the bulk list"
        assert len(col._handle_cache) == 1, (
            f"first() should open one file; cached {len(col._handle_cache)}"
        )
        assert set(col._handle_cache) == {0}, (
            f"expected index 0, got {set(col._handle_cache)}"
        )
        np.testing.assert_array_equal(
            result,
            np.arange(20, dtype="int16").reshape(4, 5),
            err_msg="first() returned the wrong timestep array",
        )

    def test_first_then_last_caches_two(self, tmp_path):
        """``first()`` then ``last()`` caches exactly two handles.

        Test scenario:
            A 3-file collection — expected: after ``first()`` + ``last()`` the
            handle cache holds indices ``{0, 2}`` and the bulk list stays lazy.
        """
        col, _ = make_int16_collection(tmp_path, count=3)
        col.first()
        col.last()
        assert col._datasets is None, (
            "point accessors must not materialise the bulk list"
        )
        assert len(col._handle_cache) == 2, (
            f"first()+last() should cache 2 handles; got {len(col._handle_cache)}"
        )
        assert set(col._handle_cache) == {0, 2}, (
            f"expected indices {{0, 2}}, got {set(col._handle_cache)}"
        )

    def test_negative_index_normalises(self, tmp_path):
        """A negative index reads from the end and caches the normalised slot.

        Test scenario:
            ``_dataset_at(-1)`` on a 3-file cube — expected: the same handle as
            index 2, cached under key 2 (negative normalised via ``range``).
        """
        col, _ = make_int16_collection(tmp_path, count=3)
        last = col._dataset_at(-1)
        assert set(col._handle_cache) == {2}, (
            f"negative index not normalised: {set(col._handle_cache)}"
        )
        assert col._dataset_at(2) is last, (
            "negative and positive index disagree on the handle"
        )

    def test_handle_reused_within_instance(self, tmp_path):
        """Repeated access to the same index returns the cached handle.

        Test scenario:
            Two ``_dataset_at(0)`` calls — expected: the identical ``Dataset``
            object, proving the cache slot is reused, not reopened.
        """
        col, _ = make_int16_collection(tmp_path, count=2)
        first = col._dataset_at(0)
        again = col._dataset_at(0)
        assert first is again, "the same index should return one cached handle"
        assert len(col._handle_cache) == 1, "a re-read must not add a cache slot"

    def test_getitem_int_uses_handle_cache(self, tmp_path):
        """``collection[i]`` (int) reads one file through the handle cache.

        Test scenario:
            ``col[1]`` on a 3-file cube — expected: a 2D array equal to the
            second timestep, with only index 1 cached and the bulk list lazy.
        """
        col, _ = make_int16_collection(tmp_path, count=3)
        arr = col[1]
        assert arr.shape == (4, 5), f"expected a 2D (4,5) slice, got {arr.shape}"
        assert col._datasets is None, (
            "__getitem__[int] must not materialise the bulk list"
        )
        assert set(col._handle_cache) == {1}, (
            f"expected index 1 cached, got {set(col._handle_cache)}"
        )
        np.testing.assert_array_equal(
            arr, np.arange(20, dtype="int16").reshape(4, 5) + 100
        )

    def test_defers_to_bulk_once_materialised(self, tmp_path):
        """Once ``.datasets`` is built, ``_dataset_at`` returns from that list.

        Test scenario:
            Access ``.datasets`` to materialise the bulk cache — expected:
            ``_dataset_at(i)`` returns the same object as ``datasets[i]`` and no
            per-index handle cache is populated.
        """
        col, _ = make_int16_collection(tmp_path, count=3)
        bulk = col.datasets
        assert col._dataset_at(1) is bulk[1], (
            "should defer to the materialised bulk list"
        )
        assert col._handle_cache == {}, (
            "the per-index cache must stay empty after bulk build"
        )

    def test_legacy_in_memory_returns_base(self, base_dataset: Dataset):
        """A legacy ``files=None`` collection returns the base at every index.

        Test scenario:
            ``DatasetCollection(base, time_length=3)`` with no files/datasets —
            expected: every ``_dataset_at`` returns the shared base template.
        """
        col = DatasetCollection(base_dataset, time_length=3)
        assert col._dataset_at(0) is base_dataset, "legacy cube should return the base"
        assert col._dataset_at(2) is base_dataset, "legacy cube should return the base"
        assert col._handle_cache == {}, "legacy path must not open files"

    def test_getstate_drops_handle_cache_on_pickle(self, tmp_path):
        """Pickling drops the live-handle ``_handle_cache`` (ARC-44 + pickle).

        Test scenario:
            Populate the cache via ``first()`` then pickle round-trip —
            expected: no ``gdal.Dataset`` in the payload, and the unpickled
            instance starts with an empty cache and a lazy bulk list.
        """
        col, _ = make_int16_collection(tmp_path, count=2)
        col.first()
        assert len(col._handle_cache) == 1, "precondition: one cached handle"
        payload = pickle.dumps(col)
        assert b"gdal.Dataset" not in payload, (
            "a live gdal handle leaked into the pickle"
        )
        restored = pickle.loads(payload)
        assert restored._handle_cache == {}, "unpickled cache should be empty"
        assert restored._datasets is None, "unpickled bulk list should be lazy"
        assert restored.time_length == 2, "time_length should survive the round-trip"


class TestTargetEpsg:
    """Tests for the module-level ``_target_epsg`` helper (ARC-54)."""

    def test_int_passthrough(self):
        """An integer EPSG is returned unchanged.

        Test scenario:
            ``_target_epsg(3857)`` — expected: ``3857`` without a pyproj lookup.
        """
        assert _target_epsg(3857) == 3857, "an int EPSG should pass through unchanged"

    def test_authority_string_resolves(self):
        """An ``EPSG:`` authority string resolves to its integer code.

        Test scenario:
            ``_target_epsg("EPSG:4326")`` — expected: ``4326``.
        """
        assert _target_epsg("EPSG:4326") == 4326, (
            "authority string should resolve to 4326"
        )

    def test_non_epsg_crs_returns_none(self):
        """A CRS with no EPSG code (proj4 Robinson) returns ``None``.

        Test scenario:
            A proj4 string with no registered EPSG — expected: ``None`` so
            ``to_crs`` takes the direct per-timestep fallback.
        """
        proj4 = "+proj=laea +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        assert _target_epsg(proj4) is None, "a no-EPSG CRS should return None"


class TestToCrsEager:
    """Eager ``to_crs`` routing through the plan-once + fallback paths (ARC-54)."""

    def test_epsg_target_returns_reprojected_collection(self, tmp_path):
        """``to_crs(epsg)`` (non-inplace) returns a new reprojected collection.

        Test scenario:
            A 4326 file-backed cube reprojected to 3857 — expected: a distinct
            ``DatasetCollection`` at EPSG 3857 with the same ``time_length``.
        """
        col, _ = make_int16_collection(tmp_path, count=2)
        out = col.to_crs(3857)
        assert isinstance(out, DatasetCollection), (
            f"expected a collection, got {type(out)}"
        )
        assert out is not col, "non-inplace to_crs should return a new collection"
        assert out.base.epsg == 3857, f"expected EPSG 3857, got {out.base.epsg}"
        assert out.time_length == 2, (
            f"time_length should be preserved, got {out.time_length}"
        )

    def test_non_epsg_target_reprojects_eagerly(self, tmp_path):
        """A no-EPSG target still reprojects eagerly via the direct fallback.

        Test scenario:
            ``to_crs`` to a proj4 LAEA CRS (no EPSG code) — expected: a
            ``DatasetCollection`` with the same ``time_length``; ``_target_epsg``
            returned ``None`` so the ``Reprojector`` plan-once path was skipped.
        """
        proj4 = "+proj=laea +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        assert _target_epsg(proj4) is None, (
            "precondition: the target must have no EPSG code"
        )
        col, _ = make_int16_collection(tmp_path, count=2)
        out = col.to_crs(proj4)
        assert isinstance(out, DatasetCollection), (
            f"expected a collection, got {type(out)}"
        )
        assert out.time_length == 2, (
            f"time_length should be preserved, got {out.time_length}"
        )

    def test_compute_false_without_dask_raises(self, tmp_path, monkeypatch):
        """``compute=False`` without dask raises ``OptionalPackageDoesNotExist``.

        Test scenario:
            Simulate a missing dask import — expected: the ``[lazy]`` extra error
            from ``_apply_operator``'s deferred branch, naming the extra.
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dask" or name.startswith("dask."):
                raise ImportError("no dask")
            return real_import(name, *args, **kwargs)

        col, _ = make_int16_collection(tmp_path, count=2)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(OptionalPackageDoesNotExist, match="lazy"):
            col.to_crs(3857, compute=False)
