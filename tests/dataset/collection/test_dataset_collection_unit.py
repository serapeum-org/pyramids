"""Unit tests for DatasetCollection methods that lack coverage.

Targets untested / low-coverage code paths in
``pyramids.dataset.collection``, including:
- ``create`` classmethod
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

import datetime as dt
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import AlignmentError, OptionalPackageDoesNotExist
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.collection import _target_epsg
from tests.dataset.collection._helpers import make_int16_collection

pytestmark = pytest.mark.core


def _make_mem_dataset(
    rows: int = 5,
    cols: int = 6,
    epsg: int = 4326,
    no_data: float = -9999.0,
    fill_value: float = 1.0,
) -> Dataset:
    """Create a minimal in-memory Dataset filled with ``fill_value``."""
    src = Dataset.create(
        rows=rows,
        columns=cols,
        dtype="float32",
        bands=1,
        no_data_value=no_data,
        geo_ref=GeoReference(
            cell_size=1.0, top_left_corner=(0.0, float(rows)), epsg=epsg
        ),
    )
    arr = np.full((rows, cols), fill_value, dtype=np.float32)
    src.raster.GetRasterBand(1).WriteArray(arr)
    return src


def _make_multiband_collection(
    tmp_path, count: int = 2, bands: int = 3, rows: int = 4, cols: int = 5
) -> tuple[DatasetCollection, np.ndarray]:
    """Build a file-backed collection whose timesteps carry several bands.

    Args:
        tmp_path: pytest temp directory.
        count: Number of timesteps to materialise.
        bands: Bands per timestep.
        rows: Raster height.
        cols: Raster width.

    Returns:
        tuple[DatasetCollection, np.ndarray]: the collection plus the
        ``(time, bands, rows, cols)`` source cube it was written from, so a
        ``band=None`` stack can be compared against the values that went in.
    """
    source = np.arange(count * bands * rows * cols, dtype="float32").reshape(
        count, bands, rows, cols
    )
    paths = []
    for i in range(count):
        path = str(tmp_path / f"multiband_t{i}.tif")
        Dataset.from_array(
            source[i],
            no_data_value=-9999.0,
            path=path,
            geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
        ).close()
        paths.append(path)
    return DatasetCollection.from_files(paths), source


@pytest.fixture()
def base_dataset() -> Dataset:
    """A small 5x6 in-memory Dataset."""
    return _make_mem_dataset()


@pytest.fixture()
def cube_with_values(base_dataset: Dataset) -> DatasetCollection:
    """A DatasetCollection with 3 time steps and pre-set values."""
    md = DatasetCollection.from_dataset(base_dataset, time_length=3)
    values = np.arange(3 * 5 * 6, dtype=np.float64).reshape(3, 5, 6)
    md.values = values
    return md


class TestCreateCube:
    """Tests for the ``create`` classmethod."""

    def test_returns_dataset_collection(self, base_dataset: Dataset):
        """create should return a DatasetCollection instance."""
        md = DatasetCollection.from_dataset(base_dataset, time_length=4)
        assert isinstance(md, DatasetCollection), (
            f"Expected DatasetCollection, got {type(md)}"
        )

    def test_time_length_matches(self, base_dataset: Dataset):
        """The time_length should match the given time_length."""
        md = DatasetCollection.from_dataset(base_dataset, time_length=7)
        assert md.time_length == 7, f"Expected time_length=7, got {md.time_length}"

    def test_base_is_same_dataset(self, base_dataset: Dataset):
        """The base property should reference the provided Dataset."""
        md = DatasetCollection.from_dataset(base_dataset, time_length=1)
        assert md.base is base_dataset, "base should be the original Dataset"

    def test_files_is_none(self, base_dataset: Dataset):
        """create does not set files so it should be None."""
        md = DatasetCollection.from_dataset(base_dataset, time_length=2)
        assert md.files is None, "files should be None for create"


class TestStringRepresentation:
    """Tests for __str__ and __repr__."""

    def test_str_contains_epsg(self, base_dataset: Dataset):
        """String representation should mention the EPSG code."""
        md = DatasetCollection(base_dataset, time_length=2, files=["a.tif", "b.tif"])
        text = str(md)
        assert "EPSG" in text, "__str__ should contain 'EPSG'"

    def test_repr_is_concise_single_line(self, base_dataset: Dataset):
        """__repr__ is a one-line ``ClassName(...)`` carrying the key fields."""
        md = DatasetCollection(base_dataset, time_length=2, files=["a.tif", "b.tif"])
        text = repr(md)
        assert "\n" not in text, f"__repr__ must be single-line; got: {text!r}"
        assert text.startswith("DatasetCollection("), text
        assert "time_length=2" in text
        assert "files=2" in text
        assert "dims=" in text
        assert "epsg=" in text

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
        assert "time_length=2" in text

    def test_repr_does_not_raise_after_close(self, base_dataset: Dataset):
        """__repr__ stays usable (never raises) even after the handles are closed.

        Reading the geo-attributes off a closed base would raise; the defensive
        ``_summary`` guard degrades those fields to ``?`` instead of blowing up
        the representation.
        """
        md = DatasetCollection(base_dataset, time_length=2)
        md.close()
        text = repr(md)
        assert text.startswith("DatasetCollection("), text
        assert "time_length=2" in text


class TestCloseAndContextManager:
    """Tests for close() and the context-manager protocol."""

    def test_close_releases_base_and_is_idempotent(self, base_dataset: Dataset):
        """close() releases the base handle and a second call is a no-op."""
        md = DatasetCollection(base_dataset, time_length=2)
        assert base_dataset._raster is not None
        md.close()
        assert base_dataset._raster is None, "base handle should be released"
        md.close()  # idempotent — must not raise

    def test_close_clears_caches(self, cube_with_values: DatasetCollection):
        """close() drops the per-timestep handle caches."""
        cube_with_values.close()
        assert cube_with_values._datasets is None
        assert cube_with_values._handle_cache == {}

    def test_context_manager_closes_on_exit(self, base_dataset: Dataset):
        """The with-block releases the collection's handles on exit."""
        with DatasetCollection(base_dataset, time_length=2) as md:
            assert md._base._raster is not None
        assert md._base._raster is None, "handles should be released on exit"

    def test_close_drops_zarr_store(self, base_dataset: Dataset):
        """close() drops a from_zarr collection's resolved store reference."""
        md = DatasetCollection(base_dataset, time_length=2, zarr_store=object())
        assert md._zarr_store is not None
        md.close()
        assert md._zarr_store is None, "zarr store reference should be dropped"


class TestIntegerIndexing:
    """Tests for NumPy-integer keys and shape validation on __getitem__/__setitem__."""

    def test_getitem_accepts_numpy_integer(self, cube_with_values: DatasetCollection):
        """A NumPy integer key reads the same slice as a Python int."""
        expected = np.arange(3 * 5 * 6, dtype=np.float64).reshape(3, 5, 6)
        np.testing.assert_allclose(cube_with_values[np.int64(1)], expected[1])

    def test_setitem_accepts_numpy_integer(self, cube_with_values: DatasetCollection):
        """A NumPy integer key assigns a slice (previously raised TypeError)."""
        new = np.full((5, 6), 42.0)
        cube_with_values[np.int64(0)] = new
        np.testing.assert_allclose(cube_with_values[0], new)

    def test_setitem_rejects_shape_mismatch(self, cube_with_values: DatasetCollection):
        """A wrong-sized array is rejected instead of silently misaligning the cube."""
        with pytest.raises(ValueError, match="does not match the collection"):
            cube_with_values[0] = np.zeros((3, 3))

    def test_setitem_rejects_non_integer_key(self, cube_with_values: DatasetCollection):
        """A non-integer key still raises TypeError."""
        with pytest.raises(TypeError, match="only accepts an integer"):
            cube_with_values["x"] = np.zeros((5, 6))

    def test_iter_yields_per_timestep_arrays(self, cube_with_values: DatasetCollection):
        """Iteration yields each timestep's band-0 array in order."""
        expected = np.arange(3 * 5 * 6, dtype=np.float64).reshape(3, 5, 6)
        seen = list(cube_with_values)
        assert len(seen) == 3
        for i, arr in enumerate(seen):
            np.testing.assert_allclose(arr, expected[i])


class TestFromFilesGlob:
    """The glob filter of from_files when given a folder."""

    @staticmethod
    def _write_tif(path: Path, fill: float = 1.0) -> None:
        """Write a small single-band GeoTIFF to ``path``."""
        _make_mem_dataset(fill_value=fill).to_file(str(path))

    def test_glob_default_reads_tif_and_skips_sidecars(self, tmp_path: Path):
        """The default ``*.tif`` glob reads rasters and skips non-.tif sidecars."""
        self._write_tif(tmp_path / "r0.tif")
        self._write_tif(tmp_path / "r1.tif")
        (tmp_path / "notes.txt").write_text("sidecar")
        (tmp_path / "r0.tif.aux.xml").write_text("<PAMDataset/>")
        cube = DatasetCollection.from_files(str(tmp_path))
        assert cube.time_length == 2, "only the two .tif rasters should be read"

    def test_glob_custom_pattern_selects_subset(self, tmp_path: Path):
        """A custom glob narrows the selection to matching names."""
        self._write_tif(tmp_path / "keep_a.tif")
        self._write_tif(tmp_path / "keep_b.tif")
        self._write_tif(tmp_path / "drop_c.tif")
        cube = DatasetCollection.from_files(str(tmp_path), glob="keep_*.tif")
        assert cube.time_length == 2, "glob should select only the keep_* rasters"

    def test_glob_no_match_raises(self, tmp_path: Path):
        """A glob matching no file raises FileNotFoundError."""
        self._write_tif(tmp_path / "r.tif")
        with pytest.raises(FileNotFoundError):
            DatasetCollection.from_files(str(tmp_path), glob="*.nc")

    def test_missing_folder_raises(self, tmp_path: Path):
        """A non-existent folder raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DatasetCollection.from_files(tmp_path / "nope")


class TestFromFilesDateOrdering:
    """date_format ordering / subsetting on from_files (folder or list)."""

    @staticmethod
    def _write_days(folder: Path, days) -> None:
        """Write one dated GeoTIFF per day (fill == day) into ``folder``."""
        for day in days:
            p = folder / f"r_1979.01.{day:02d}.tif"
            _make_mem_dataset(fill_value=float(day)).to_file(str(p))

    def test_date_format_sorts_and_sets_time_axis(self, tmp_path: Path):
        """date_format sorts by the file-name date and makes it the time axis."""
        self._write_days(tmp_path, (3, 1, 2))
        cube = DatasetCollection.from_files(tmp_path, date_format="%Y.%m.%d")
        assert cube.time == [dt.datetime(1979, 1, d) for d in (1, 2, 3)]
        np.testing.assert_allclose(cube[0], 1.0)  # earliest date has fill 1.0

    def test_date_format_on_an_explicit_list(self, tmp_path: Path):
        """date_format works on a list too, not just a folder."""
        self._write_days(tmp_path, (2, 1))
        files = [str(p) for p in tmp_path.glob("*.tif")]
        cube = DatasetCollection.from_files(files, date_format="%Y.%m.%d")
        assert cube.time == [dt.datetime(1979, 1, 1), dt.datetime(1979, 1, 2)]

    def test_start_end_subsets(self, tmp_path: Path):
        """start/end keep the inclusive date range."""
        self._write_days(tmp_path, (1, 2, 3, 4))
        cube = DatasetCollection.from_files(
            tmp_path,
            date_format="%Y.%m.%d",
            start=dt.datetime(1979, 1, 2),
            end=dt.datetime(1979, 1, 3),
        )
        assert cube.time == [dt.datetime(1979, 1, 2), dt.datetime(1979, 1, 3)]

    def test_custom_date_regex(self, tmp_path: Path):
        """date_regex locates a non-default date pattern in the names."""
        for day in (2, 1):
            _make_mem_dataset().to_file(str(tmp_path / f"evap_1979_{day}_1.tif"))
        cube = DatasetCollection.from_files(
            tmp_path, date_regex=r"\d{4}_\d{1,2}_\d{1,2}", date_format="%Y_%m_%d"
        )
        assert cube.time == [dt.datetime(1979, 1, 1), dt.datetime(1979, 2, 1)]

    def test_start_end_without_date_format_raises(self, tmp_path: Path):
        """start/end without date_format raises ValueError."""
        self._write_days(tmp_path, (1,))
        start = dt.datetime(1979, 1, 1)
        with pytest.raises(ValueError, match="needs date_format"):
            DatasetCollection.from_files(tmp_path, start=start)

    def test_date_not_found_raises(self, tmp_path: Path):
        """A file name with no matching date raises ValueError."""
        _make_mem_dataset().to_file(str(tmp_path / "no_date_here.tif"))
        with pytest.raises(ValueError, match="matched no date"):
            DatasetCollection.from_files(tmp_path, date_format="%Y.%m.%d")

    def test_empty_list_raises(self):
        """An empty list raises ValueError."""
        with pytest.raises(ValueError, match="at least one path"):
            DatasetCollection.from_files([])

    def test_start_end_empty_range_raises(self, tmp_path: Path):
        """A date range that excludes every file raises FileNotFoundError."""
        self._write_days(tmp_path, (1, 2))
        start = dt.datetime(1999, 1, 1)
        end = dt.datetime(1999, 12, 31)
        with pytest.raises(FileNotFoundError, match="within the given start/end"):
            DatasetCollection.from_files(
                tmp_path, date_format="%Y.%m.%d", start=start, end=end
            )

    def test_start_only_bound(self, tmp_path: Path):
        """Passing only ``start`` keeps the files on/after that date."""
        self._write_days(tmp_path, (1, 2, 3))
        cube = DatasetCollection.from_files(
            tmp_path, date_format="%Y.%m.%d", start=dt.datetime(1979, 1, 2)
        )
        assert cube.time == [dt.datetime(1979, 1, 2), dt.datetime(1979, 1, 3)]

    def test_end_only_bound(self, tmp_path: Path):
        """Passing only ``end`` keeps the files on/before that date."""
        self._write_days(tmp_path, (1, 2, 3))
        cube = DatasetCollection.from_files(
            tmp_path, date_format="%Y.%m.%d", end=dt.datetime(1979, 1, 2)
        )
        assert cube.time == [dt.datetime(1979, 1, 1), dt.datetime(1979, 1, 2)]


class TestFromFilesConstruction:
    """Construction paths of from_files: single file, validate, explicit meta."""

    def test_single_file_list(self, tmp_path: Path):
        """A one-element list builds a one-timestep collection."""
        p = tmp_path / "only.tif"
        _make_mem_dataset().to_file(str(p))
        cube = DatasetCollection.from_files([str(p)])
        assert cube.time_length == 1
        assert cube.files == [str(p)]

    def test_single_file_path_string(self, tmp_path: Path):
        """A single file-path string is read as a one-timestep collection (not globbed)."""
        p = tmp_path / "only.tif"
        _make_mem_dataset().to_file(str(p))
        cube = DatasetCollection.from_files(str(p))
        assert cube.time_length == 1
        assert cube.files == [str(p)]

    def test_validate_passes_for_homogeneous_files(self, tmp_path: Path):
        """validate=True succeeds when every file matches the template."""
        for name in ("a.tif", "b.tif"):
            _make_mem_dataset(rows=5, cols=6).to_file(str(tmp_path / name))
        cube = DatasetCollection.from_files(tmp_path, validate=True)
        assert cube.time_length == 2

    def test_validate_detects_shape_mismatch(self, tmp_path: Path):
        """validate=True raises AlignmentError when a file's shape differs."""
        _make_mem_dataset(rows=5, cols=6).to_file(str(tmp_path / "a.tif"))
        _make_mem_dataset(rows=4, cols=4).to_file(str(tmp_path / "b.tif"))
        with pytest.raises(AlignmentError):
            DatasetCollection.from_files(tmp_path, validate=True)


class TestReadMultipleFilesDeprecated:
    """The read_multiple_files shim warns but still works."""

    def test_emits_deprecation_warning_and_reads(self, rasters_folder_path: str):
        """It emits DeprecationWarning and returns a working collection."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            cube = DatasetCollection.read_multiple_files(rasters_folder_path)
        assert cube.time_length == 6

    def test_numeric_ordering_empty_range_raises(self, tmp_path: Path):
        """The legacy numeric mode with an out-of-range start/end raises."""
        for name in ("1_r.tif", "2_r.tif"):
            _make_mem_dataset().to_file(str(tmp_path / name))
        with pytest.raises(FileNotFoundError, match="within the given start/end"):
            DatasetCollection.read_multiple_files(
                tmp_path,
                with_order=True,
                date=False,
                regex_string=r"\d+",
                start=90,
                end=99,
            )

    def test_start_end_without_date_format_raises(self, tmp_path: Path):
        """start/end with no parseable date format raises (the old contract)."""
        for name in ("a.tif", "b.tif"):
            _make_mem_dataset().to_file(str(tmp_path / name))
        with pytest.raises(ValueError, match="needs a date format"):
            DatasetCollection.read_multiple_files(
                tmp_path, start="1979-01-02", end="1979-01-05"
            )

    def test_numeric_mode_sets_integer_time_axis(self, tmp_path: Path):
        """date=False leaves the integer order keys on .time (not None)."""
        for name in ("2_r.tif", "1_r.tif"):
            _make_mem_dataset().to_file(str(tmp_path / name))
        with pytest.warns(DeprecationWarning):
            cube = DatasetCollection.read_multiple_files(
                tmp_path, with_order=True, date=False, regex_string=r"\d+"
            )
        assert cube.time == [1, 2], f"expected integer axis [1, 2], got {cube.time}"

    def test_date_without_order_preserves_input_order(self, tmp_path: Path):
        """with_order=False + a date format labels time WITHOUT reordering the list."""
        paths = []
        for day in (3, 1, 2):  # deliberately unsorted
            p = tmp_path / f"r_1979.01.{day:02d}.tif"
            _make_mem_dataset().to_file(str(p))
            paths.append(str(p))
        with pytest.warns(DeprecationWarning):
            cube = DatasetCollection.read_multiple_files(
                paths, with_order=False, date=True, file_name_data_fmt="%Y.%m.%d"
            )
        assert cube.time == [
            dt.datetime(1979, 1, 3),
            dt.datetime(1979, 1, 1),
            dt.datetime(1979, 1, 2),
        ], f"input order should be preserved; got {cube.time}"


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
        md = DatasetCollection.from_dataset(base_dataset, time_length=2)
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
        md = DatasetCollection.from_dataset(base_dataset, time_length=2)
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

    def test_to_file_writes_correct_per_timestep_data(
        self, cube_with_values: DatasetCollection
    ):
        """Each written file must hold its own timestep's pixels, not the base template.

        Guards the streaming rewrite: to_file now hands each ``iloc(i)`` handle
        straight to ``Dataset.to_file`` (GDAL CreateCopy) instead of the old
        read_array + in-memory copy. The per-slice values must survive intact.
        """
        expected = np.arange(3 * 5 * 6, dtype=np.float64).reshape(3, 5, 6)
        tmp_dir = Path(tempfile.mkdtemp())
        out_dir = tmp_dir / "stack"
        try:
            cube_with_values.to_file(out_dir)
            for i in range(3):
                written = Dataset.read_file(str(out_dir / f"{i}.tif"))
                np.testing.assert_allclose(written.read_array(), expected[i])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_to_file_preserves_collection_data(
        self, cube_with_values: DatasetCollection
    ):
        """Saving must not mutate the collection's in-memory per-timestep data.

        ``cube_with_values`` is ``datasets=``-backed (``_files is None``), so the
        old trailing ``self._datasets = None`` collapsed every slice to the base
        template on the next read. A save is a pure side-effect-free export now:
        the collection still yields its original per-slice values afterwards.
        """
        expected = np.arange(3 * 5 * 6, dtype=np.float64).reshape(3, 5, 6)
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            cube_with_values.to_file(tmp_dir / "stack")
            for i in range(3):
                np.testing.assert_allclose(cube_with_values[i], expected[i])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_to_file_does_not_mutate_source_handles(
        self, cube_with_values: DatasetCollection
    ):
        """Per-timestep handles stay path-less MEM datasets after a save.

        ``cube_with_values`` is ``datasets=``-backed by in-memory MEM datasets
        (``file_name == ""``). ``to_file`` streams them with ``reopen=False``, so
        none of them are repointed at the written files.
        """
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            cube_with_values.to_file(tmp_dir / "stack")
            for i in range(3):
                assert cube_with_values.iloc(i).file_name == "", (
                    f"timestep {i} handle was repointed to a file after save"
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_to_file_accepts_path_object_directory(
        self, cube_with_values: DatasetCollection
    ):
        """A ``pathlib.Path`` directory is accepted (not only ``str``).

        Test scenario:
            Passing a ``Path`` directory writes ``0.tif``..``2.tif`` just like a
            string path — exercising the ``isinstance(path, (str, Path))`` branch.
        """
        tmp_dir = Path(tempfile.mkdtemp())
        out_dir = tmp_dir / "as_path"
        try:
            cube_with_values.to_file(out_dir)
            assert sorted(p.name for p in out_dir.iterdir()) == [
                "0.tif",
                "1.tif",
                "2.tif",
            ]
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_to_file_list_paths_creates_missing_parent(
        self, cube_with_values: DatasetCollection
    ):
        """A list of paths whose parent dir does not exist yet is created.

        Test scenario:
            Exercises the ``else`` branch that ``mkdir(parents=True)`` the parent
            of the first path when it is missing.
        """
        tmp_dir = Path(tempfile.mkdtemp())
        missing_parent = tmp_dir / "not_yet" / "here"
        paths = [missing_parent / f"t{i}.tif" for i in range(3)]
        try:
            assert not missing_parent.exists(), "precondition: parent is absent"
            cube_with_values.to_file(paths)
            for p in paths:
                assert p.exists(), f"expected file at {p}"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_to_file_single_timestep(self, base_dataset: Dataset):
        """A one-timestep collection writes exactly one file with the base pixels.

        Test scenario:
            ``create(base, 1)`` written to a directory yields a single
            ``0.tif`` whose pixels equal the base template.
        """
        cube = DatasetCollection.from_dataset(base_dataset, time_length=1)
        tmp_dir = Path(tempfile.mkdtemp())
        out_dir = tmp_dir / "single"
        try:
            cube.to_file(out_dir)
            files = sorted(p.name for p in out_dir.iterdir())
            assert files == ["0.tif"], f"expected a single 0.tif, got {files}"
            written = Dataset.read_file(str(out_dir / "0.tif")).read_array()
            np.testing.assert_allclose(written, base_dataset.read_array())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_to_file_list_infers_driver_from_path_extension(
        self, base_dataset: Dataset
    ):
        """An explicit path list honors each path's extension, not the default driver.

        ``to_file`` passes no ``driver`` to the per-timestep write, so a list of
        ``.asc`` paths writes ASCII grids even though the collection's default
        driver is ``geotiff`` — the pre-streaming extension-inference contract.
        Guards against silently writing GeoTIFF bytes into ``.asc``-named files.
        """
        cube = DatasetCollection.from_dataset(base_dataset, time_length=2)
        tmp_dir = Path(tempfile.mkdtemp())
        paths = [tmp_dir / f"grid_{i}.asc" for i in range(2)]
        try:
            cube.to_file(paths)
            for p in paths:
                assert p.exists(), f"expected an output at {p}"
                magic = p.read_bytes()[:2]
                assert magic not in (b"II", b"MM"), (
                    f"{p} was written as a TIFF, not ASCII (driver override leaked)"
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_to_file_preserves_color_table(self):
        """A palette (color table) on a slice survives the streaming write.

        The old ``read_array()`` + ``_mem_dataset_from_array()`` round-trip
        flattened output through ``from_array``, dropping color tables;
        the ``CreateCopy`` stream preserves them. Guards that fidelity benefit
        against a regression back to a flattening write path (this test fails on
        the pre-rewrite path).
        """
        src = Dataset.create(
            rows=3,
            columns=3,
            dtype="uint8",
            bands=1,
            no_data_value=0,
            geo_ref=GeoReference(cell_size=1.0, top_left_corner=(0.0, 3.0), epsg=4326),
        )
        band = src.raster.GetRasterBand(1)
        ct = gdal.ColorTable()
        ct.SetColorEntry(1, (255, 0, 0, 255))
        ct.SetColorEntry(2, (0, 255, 0, 255))
        band.SetRasterColorTable(ct)
        band.WriteArray(np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]], dtype=np.uint8))
        cube = DatasetCollection.from_dataset(src, time_length=1)
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            cube.to_file(tmp_dir / "paletted")
            reloaded = Dataset.read_file(str(tmp_dir / "paletted" / "0.tif"))
            rct = reloaded.raster.GetRasterBand(1).GetRasterColorTable()
            assert rct is not None, "color table was dropped by to_file"
            assert rct.GetColorEntry(1) == (255, 0, 0, 255), (
                f"palette entry not preserved: {rct.GetColorEntry(1)}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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
        """Filenames that don't match the regex should raise ValueError."""
        src = _make_mem_dataset(rows=3, cols=3)
        paths = []
        for name in ["no_date_a.tif", "no_date_b.tif"]:
            p = str(tmp_path / name)
            src.to_file(p)
            paths.append(p)
        with pytest.raises(ValueError, match="matched no date"):
            DatasetCollection.read_multiple_files(
                paths,
                with_order=True,
                regex_string=r"\d{4}\.\d{2}\.\d{2}",
                file_name_data_fmt="%Y.%m.%d",
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
    Dataset.from_array(
        arr,
        no_data_value=no_data,
        path=str(path),
        geo_ref=GeoReference(top_left_corner=top_left, cell_size=cell_size, epsg=epsg),
    ).close()
    return str(path)


class TestMemDatasetFromArray:
    """Tests for ``_mem_dataset_from_array`` (ARC-70)."""

    @pytest.fixture()
    def collection(self) -> DatasetCollection:
        """A single-timestep in-memory collection whose base is float32."""
        base = Dataset.from_array(
            np.ones((3, 4), dtype="float32"),
            no_data_value=-9999.0,
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
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
        other = Dataset.from_array(
            np.ones((3, 4), dtype="float32"),
            no_data_value=-1.0,
            geo_ref=GeoReference(
                top_left_corner=(100.0, 50.0), cell_size=2.0, epsg=4326
            ),
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
    """Tests for the ARC-46 head/tail fix and empty-safe ``_stack_timesteps``."""

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

    def test_empty_selection_matches_cube_dtype(self, tmp_path):
        """head(0)/tail(0) carry the cube's dtype, not NumPy's default float64 (N1).

        Test scenario:
            An int16-backed collection — expected: ``head(0)`` and ``tail(0)`` are
            int16 (matching a non-empty selection), not the float64 that an
            untyped ``np.empty`` would give.
        """
        collection, _ = make_int16_collection(tmp_path)
        assert collection.head(0).dtype == np.int16, (
            f"head(0) dtype {collection.head(0).dtype} != int16"
        )
        assert collection.tail(0).dtype == np.int16, (
            f"tail(0) dtype {collection.tail(0).dtype} != int16"
        )
        assert collection.head(1).dtype == np.int16, (
            "non-empty head should be int16 too"
        )

    def test_stack_timesteps_empty_selection(self, cube_with_values: DatasetCollection):
        """_stack_timesteps([]) returns a (0, rows, cols) array (empty-safe).

        Test scenario:
            Stack an empty selection — expected: a (0, 5, 6) array instead of the
            ``np.stack`` "need at least one array" error.
        """
        result = cube_with_values._stack_timesteps([])
        assert result.shape == (0, 5, 6), f"expected (0,5,6), got {result.shape}"

    def test_stack_timesteps_non_empty(
        self, cube_with_values: DatasetCollection, expected: np.ndarray
    ):
        """_stack_timesteps over all datasets reproduces the full cube.

        Test scenario:
            Stack every timestep's band 0 — expected: the (3, 5, 6) source cube.
        """
        result = cube_with_values._stack_timesteps(cube_with_values.datasets)
        every_band = cube_with_values._stack_timesteps(
            cube_with_values.datasets, band=None
        )

        assert every_band.shape[0] == len(cube_with_values.datasets), (
            "band=None must keep time on the leading axis, not collapse it -- "
            "it is what the RGB time-lapse asks for, and the only caller that "
            "passes anything but band 0"
        )
        assert result.shape == (3, 5, 6), f"expected (3,5,6), got {result.shape}"
        np.testing.assert_array_equal(result, expected)

    def test_stack_timesteps_band_none_keeps_the_band_axis(self, tmp_path):
        """band=None stacks every band into ``(time, bands, rows, cols)``.

        Test scenario:
            The RGB time-lapse is the only caller that passes ``band=None``, and
            it needs a 4-D cube whose leading axis is time and whose second axis
            is the colour bands. If the helper collapsed the bands into time (or
            kept reading band 0 regardless), cleopatra would composite frames out
            of timesteps and the animation would render as a single smeared
            image.
        """
        collection, source = _make_multiband_collection(tmp_path, count=2, bands=3)

        result = collection._stack_timesteps(collection.datasets, band=None)

        assert result.shape == (2, 3, 4, 5), (
            f"expected (time=2, bands=3, 4, 5), got {result.shape}"
        )
        np.testing.assert_array_equal(
            result, source, err_msg="band=None did not reproduce the per-band source"
        )

    def test_band_none_rank_is_unreachable_for_a_single_band_collection(
        self, cube_with_values: DatasetCollection
    ):
        """A one-band collection squeezes under band=None, and the RGB path refuses it.

        Test scenario:
            ``Dataset.read_array(band=None)`` drops the band axis for a
            single-band raster, so ``band=None`` yields ``(time, rows, cols)``
            there rather than the ``(time, bands, rows, cols)`` the RGB
            time-lapse needs. That rank ambiguity is only harmless while
            ``_validate_rgb_animation`` keeps rejecting a collection with too few
            bands — if that guard were relaxed, cleopatra would be handed a 3-D
            array where it expects 4-D and composite time as colour.
        """
        stacked = cube_with_values._stack_timesteps(
            cube_with_values.datasets, band=None
        )

        np.testing.assert_array_equal(
            stacked,
            cube_with_values._stack_timesteps(cube_with_values.datasets, band=0),
            err_msg="a single-band collection should squeeze to the band-0 stack",
        )
        with pytest.raises(ValueError, match="needs at least 3 bands"):
            cube_with_values.plot(rgb_options={"rgb": [0, 1, 2]})

    def test_the_empty_band_none_cube_keeps_the_band_axis(self, tmp_path):
        """The empty result has the rank a populated one of the same request has.

        Args:
            tmp_path: pytest temp directory for the multi-band GeoTIFFs.

        Test scenario:
            The empty guard returned `(0, rows, cols)` whatever `band` was, so
            `band=None` on a multi-band collection gave a 3-D empty cube where a
            populated one is 4-D. The RGB time-lapse branches on exactly that
            rank, so an empty collection came back as "RGB animate requires a
            4-D ... got 3-D. Pass rgb only with a multi-band temporal stack" --
            telling the caller their bands are wrong when what is wrong is that
            there are no timesteps at all.
        """
        collection, _ = _make_multiband_collection(tmp_path, count=2, bands=3)

        empty = collection._stack_timesteps([], band=None)
        populated = collection._stack_timesteps(collection.datasets, band=None)

        assert empty.shape == (0, 3, 4, 5), (
            f"expected (time=0, bands=3, 4, 5), got {empty.shape}"
        )
        assert empty.shape[1:] == populated.shape[1:], (
            "the empty cube must differ from the populated one only in time: "
            f"{empty.shape} vs {populated.shape}"
        )

    def test_the_empty_band_zero_cube_stays_three_dimensional(self, tmp_path):
        """The extra axis belongs to `band=None`, not to a multi-band collection.

        Args:
            tmp_path: pytest temp directory for the multi-band GeoTIFFs.

        Test scenario:
            `values`, `head` and `tail` all read band 0 of the same multi-band
            collection and are documented as `(time, rows, cols)`. Giving the
            empty guard a band axis whenever the collection has bands would
            change the shape those three return on an empty selection.
        """
        collection, _ = _make_multiband_collection(tmp_path, count=2, bands=3)

        empty = collection._stack_timesteps([], band=0)

        assert empty.shape == (0, 4, 5), f"expected (0, 4, 5), got {empty.shape}"

    def test_a_single_band_collection_squeezes_under_band_none_when_empty_too(
        self, cube_with_values: DatasetCollection
    ):
        """The guard's band-count test has to match what `read_array` does.

        Args:
            cube_with_values: A 3-timestep single-band 5x6 collection.

        Test scenario:
            `Dataset.read_array(band=None)` drops the band axis for a
            single-band raster, so a populated `band=None` stack is 3-D there.
            An empty one that added a length-1 band axis would be the same
            rank mismatch in the other direction.
        """
        empty = cube_with_values._stack_timesteps([], band=None)
        populated = cube_with_values._stack_timesteps(
            cube_with_values.datasets, band=None
        )

        assert empty.shape == (0, 5, 6), f"expected (0, 5, 6), got {empty.shape}"
        assert empty.shape[1:] == populated.shape[1:], (
            "the empty cube must differ from the populated one only in time: "
            f"{empty.shape} vs {populated.shape}"
        )

    def test_stack_timesteps_empty_carries_the_collection_dtype(self, tmp_path):
        """The empty guard types the array from the collection, not float64 (N1).

        Test scenario:
            ``np.empty(shape)`` with no dtype yields float64. An int16 collection
            whose ``head(0)`` came back float64 would change dtype under an empty
            selection, so a caller concatenating an empty slice onto a real one
            would silently upcast the whole stack.
        """
        collection, _ = make_int16_collection(tmp_path)

        result = collection._stack_timesteps([])

        assert result.dtype == np.int16, (
            f"empty stack should carry the collection dtype int16, got {result.dtype}"
        )
        assert result.shape == (0, collection.rows, collection.columns), (
            f"expected (0, rows, cols), got {result.shape}"
        )

    @pytest.mark.parametrize(
        ("accessor", "expected_count"),
        [("values", 3), ("head", 2), ("tail", 2)],
        ids=["values", "head", "tail"],
    )
    def test_accessors_route_through_stack_timesteps(
        self,
        cube_with_values: DatasetCollection,
        monkeypatch,
        accessor: str,
        expected_count: int,
    ):
        """``values``/``head``/``tail`` all materialise through the one helper.

        Args:
            cube_with_values: A three-timestep collection.
            monkeypatch: pytest fixture, used to spy on the helper.
            accessor: The public accessor under test.
            expected_count: How many timesteps that accessor should stack.

        Test scenario:
            The helper is where the empty guard and the dtype choice live. An
            accessor that went back to a bare ``np.stack`` would keep working on
            a populated collection and only break on an empty selection, with
            ``np.stack``'s "need at least one array" naming neither the
            collection nor the timestep.
        """
        calls: list[tuple[int, int | None]] = []
        original = DatasetCollection._stack_timesteps

        def _spy(self, datasets, band=0):
            calls.append((len(datasets), band))
            return original(self, datasets, band=band)

        monkeypatch.setattr(DatasetCollection, "_stack_timesteps", _spy)
        if accessor == "values":
            result = cube_with_values.values
        else:
            result = getattr(cube_with_values, accessor)(2)

        assert len(calls) == 1, f"{accessor} should stack once, got {len(calls)} calls"
        assert calls[0] == (expected_count, 0), (
            f"{accessor} stacked {calls[0]}, expected ({expected_count}, band 0)"
        )
        assert result.shape == (expected_count, 5, 6), (
            f"{accessor} returned {result.shape}, expected ({expected_count}, 5, 6)"
        )

    def test_plot_stacks_the_requested_band_through_the_helper(
        self, cube_with_values: DatasetCollection, monkeypatch
    ):
        """``plot`` builds its animation cube with the shared helper, band first.

        Args:
            cube_with_values: A three-timestep collection.
            monkeypatch: pytest fixture, used to spy on the helper and to stand in
                for the cleopatra dispatch.

        Test scenario:
            ``plot`` used to inline its own ``np.stack``, which is the call that
            raised an unattributable error on an empty collection. The render
            seam is stubbed so this stays an offline unit test; what is pinned is
            that the array handed to cleopatra is the helper's ``(time, rows,
            cols)`` cube for the requested band.
        """
        calls: list[tuple[int, int | None]] = []
        original = DatasetCollection._stack_timesteps
        requests: list[Any] = []

        def _spy(self, datasets, band=0):
            calls.append((len(datasets), band))
            return original(self, datasets, band=band)

        def _fake_render(request, **kwargs):
            requests.append(request)
            return "glyph"

        monkeypatch.setattr(DatasetCollection, "_stack_timesteps", _spy)
        monkeypatch.setattr("pyramids.dataset.collection.render_array", _fake_render)
        result = cube_with_values.plot(band=0)

        assert result == "glyph", (
            f"plot should return the render result, got {result!r}"
        )
        assert calls == [(3, 0)], f"plot stacked {calls}, expected one (3, band 0) call"
        assert requests[0].arr.shape == (3, 5, 6), (
            f"cleopatra received {requests[0].arr.shape}, expected (3, 5, 6)"
        )

    def test_rgb_plot_stacks_every_band_through_the_helper(self, tmp_path, monkeypatch):
        """The RGB time-lapse asks the helper for every band, not just band 0.

        Args:
            tmp_path: pytest temp directory for the three-band fixture.
            monkeypatch: pytest fixture, used to spy on the helper and to stand in
                for the cleopatra dispatch.

        Test scenario:
            This is the second inline ``np.stack`` the helper replaced, and the
            only one that passes ``band=None``. If it lost the argument the
            true-colour composite would be fed one band repeated three times and
            every frame would render greyscale.
        """
        collection, source = _make_multiband_collection(tmp_path, count=2, bands=3)
        calls: list[tuple[int, int | None]] = []
        original = DatasetCollection._stack_timesteps
        requests: list[Any] = []

        def _spy(self, datasets, band=0):
            calls.append((len(datasets), band))
            return original(self, datasets, band=band)

        def _fake_render(request, **kwargs):
            requests.append(request)
            return "glyph"

        monkeypatch.setattr(DatasetCollection, "_stack_timesteps", _spy)
        monkeypatch.setattr("pyramids.dataset.collection.render_array", _fake_render)
        result = collection.plot(rgb_options={"rgb": [0, 1, 2]})

        assert result == "glyph", (
            f"plot should return the render result, got {result!r}"
        )
        assert calls == [(2, None)], (
            f"the RGB path stacked {calls}, expected one (2, band=None) call"
        )
        np.testing.assert_array_equal(
            requests[0].arr,
            source,
            err_msg="the RGB animation cube is not the per-band source stack",
        )


class TestPlottingAnEmptyCollection:
    """A render has no frames to draw, and the refusal has to say so."""

    def test_plot_refuses_an_empty_collection_by_name(
        self, base_dataset: Dataset, monkeypatch
    ):
        """The empty cube is a read's answer, not a render's.

        Args:
            base_dataset: A single 5x6 raster to build the collection from.
            monkeypatch: pytest fixture, standing in for the cleopatra dispatch
                so the refusal is what fails the call, not a missing extra.

        Test scenario:
            `_stack_timesteps` returns an empty `(0, rows, cols)` cube for an
            empty collection, which is right for `values` / `head` / `tail` and
            useless for an animation: the zero-length stack reached cleopatra
            and came back as numpy's "zero-size array to reduction operation
            minimum which has no identity", which names neither the collection
            nor the reason.
        """
        collection = DatasetCollection.from_dataset(base_dataset, time_length=0)
        monkeypatch.setattr(
            "pyramids.dataset.collection.render_array", lambda request, **kw: "glyph"
        )

        with pytest.raises(ValueError, match="empty collection"):
            collection.plot()

    def test_the_rgb_path_refuses_it_too(self, base_dataset: Dataset, monkeypatch):
        """Both render paths stack timesteps, so both need the guard.

        Args:
            base_dataset: A single 5x6 raster to build the collection from.
            monkeypatch: pytest fixture, standing in for the cleopatra dispatch.

        Test scenario:
            The RGB time-lapse builds its own `(time, bands, rows, cols)` stack;
            an empty collection makes that 4-D and empty rather than 3-D, so it
            fails somewhere else again unless the refusal comes first.
        """
        collection = DatasetCollection.from_dataset(base_dataset, time_length=0)
        monkeypatch.setattr(
            "pyramids.dataset.collection.render_array", lambda request, **kw: "glyph"
        )

        with pytest.raises(ValueError, match="empty collection"):
            collection.plot(rgb_options={"rgb": [0, 1, 2]})

    def test_the_read_accessors_still_return_an_empty_cube(self, base_dataset: Dataset):
        """Refusing the render must not turn a legitimate read into an error.

        Test scenario:
            `values` on an empty collection is `(0, rows, cols)` of the
            collection's dtype -- the answer the empty-safe stack helper was
            written for -- and the plot guard sits in `plot`, not in it.

        Args:
            base_dataset: A single 5x6 raster to build the collection from.
        """
        collection = DatasetCollection.from_dataset(base_dataset, time_length=0)

        assert collection.values.shape == (0, 5, 6)


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

    def test_crs_equal_treats_same_epsg_encodings_as_equal(self):
        """_crs_equal treats same-EPSG CRS as equal across encodings (N2).

        Test scenario:
            An EPSG:4326 CRS vs an equivalent proj4 longlat-WGS84 CRS (which
            ``pyproj``'s strict ``==`` reports unequal) — expected: ``_crs_equal``
            returns True via the shared EPSG code, so validation does not reject a
            co-registered file on a cosmetic encoding difference; a genuinely
            different system (EPSG:3857) still returns False.
        """
        from pyproj import CRS

        from pyramids.dataset.collection import _crs_equal

        epsg = CRS.from_epsg(4326)
        proj4 = CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")
        assert epsg != proj4, "precondition: pyproj's == is strict for this pair"
        assert _crs_equal(epsg, proj4), "same-EPSG encodings should compare equal"
        assert not _crs_equal(epsg, CRS.from_epsg(3857)), "different CRS must differ"

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
        assert "must share" in str(exc.value), f"unexpected message: {exc.value}"

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

    def test_validate_true_shifted_extent_raises(self, tmp_path):
        """A same-shape raster with a shifted extent raises on the geotransform (M1)."""
        t0 = _write_geotiff(tmp_path / "t0.tif", np.zeros((4, 5), dtype="int16"))
        bad = _write_geotiff(
            tmp_path / "bad.tif", np.zeros((4, 5), dtype="int16"), top_left=(100.0, 4.0)
        )
        with pytest.raises(AlignmentError, match="geotransform") as exc:
            DatasetCollection.from_files([t0, bad], validate=True)
        assert "bad.tif" in str(exc.value), f"path not named: {exc.value}"

    def test_validate_true_crs_mismatch_raises(self, tmp_path):
        """A same-shape/geotransform raster in a different CRS raises on CRS (M1)."""
        t0 = _write_geotiff(tmp_path / "t0.tif", np.zeros((4, 5), dtype="int16"))
        bad = _write_geotiff(
            tmp_path / "bad.tif", np.zeros((4, 5), dtype="int16"), epsg=3857
        )
        with pytest.raises(AlignmentError, match="CRS"):
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
