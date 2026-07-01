"""Unit tests for Dataset read/write/create/translate/copy/file and array/window I/O."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal

from pyramids.base._errors import FailedToSaveError, OutOfBoundsError, ReadOnlyError
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestTranslate:
    """Tests for the translate method."""

    def test_translate_returns_dataset(self, single_band_dataset):
        """translate() should return a new Dataset object."""
        result = single_band_dataset.translate()
        assert isinstance(result, Dataset), "translate() should return a Dataset"

    def test_translate_unscale(self):
        """translate(unscale=True) should apply scale and offset."""
        arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        ds.scale = [0.1]
        ds.offset = [100.0]
        unscaled = ds.translate(unscale=True)
        result_arr = unscaled.read_array()
        expected = arr * 0.1 + 100.0
        np.testing.assert_allclose(
            result_arr,
            expected,
            atol=0.01,
            err_msg="Unscaled values are incorrect",
        )


class TestWriteArray:
    """Tests for the write_array method."""

    def test_write_array_no_offset(self):
        """write_array with default offset should overwrite from (0, 0)."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        patch = np.array([[99.0, 99.0], [99.0, 99.0]], dtype=np.float32)
        ds.write_array(patch, top_left_corner=[0, 0])
        result = ds.read_array()
        assert result[0, 0] == pytest.approx(99.0), "Top-left cell should be 99 after write"
        assert result[0, 1] == pytest.approx(99.0), "Cell (0,1) should be 99 after write"

    def test_write_array_with_offset(self):
        """write_array with offset should write at the given position."""
        arr = np.zeros((4, 4), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        patch = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
        ds.write_array(patch, top_left_corner=[1, 1])
        result = ds.read_array()
        assert result[1, 1] == pytest.approx(7.0), "Offset write failed at (1,1)"
        assert result[2, 2] == pytest.approx(10.0), "Offset write failed at (2,2)"
        assert result[0, 0] == pytest.approx(0.0), "Cell outside patch should be unchanged"

    def test_write_array_multi_band(self):
        """write_array writes a multi-band patch at an offset across every band.

        Ports the multi-band scenario from the legacy duplicate (which used the
        on-disk ``empty-to-fill-multi-band.tif`` fixture). A (2, 2, 2) patch
        written at row/col offset (3, 5) must round-trip exactly on both bands.
        """
        base = np.zeros((2, 8, 8), dtype=np.float64)
        ds = Dataset.create_from_array(
            base,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
        )
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        xoff = 5
        yoff = 3
        ds.write_array(arr, top_left_corner=[yoff, xoff])
        retrieved = ds._raster.ReadAsArray(xoff, yoff, 2, 2)
        np.testing.assert_array_equal(arr, retrieved)


class TestCreateFromArray:
    """Tests for ``Dataset.create_from_array`` (the public form)."""

    def test_single_band(self):
        """Create a single-band dataset from a 2D array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        geo = (0.0, 0.5, 0.0, 10.0, 0.0, -0.5)
        result = Dataset.create_from_array(arr, geo=geo, epsg=4326)
        assert isinstance(result, Dataset), "Should return a Dataset"
        read_arr = result.read_array()
        np.testing.assert_array_equal(read_arr, arr, err_msg="Array values mismatch")

    def test_multi_band(self):
        """Create a multi-band dataset from a 3D array."""
        arr = np.ones((3, 4, 5), dtype=np.float64)
        geo = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
        result = Dataset.create_from_array(arr, geo=geo, epsg=4326)
        assert result.band_count == 3, "Expected 3 bands"


class TestToXyz:
    """Tests for the to_xyz method."""

    def test_to_xyz_returns_dataframe(self):
        """to_xyz without path should return a DataFrame."""
        arr = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            dtype=np.int32,
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        df = ds.to_xyz()
        assert isinstance(df, pd.DataFrame), "to_xyz should return DataFrame"
        assert "lon" in df.columns, "DataFrame should have 'lon' column"
        assert "lat" in df.columns, "DataFrame should have 'lat' column"
        assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

    def test_to_xyz_specific_bands(self):
        """to_xyz with specific bands should only include those bands."""
        arr = np.ones((3, 4, 4), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df = ds.to_xyz(bands=[0])
        band_cols = [c for c in df.columns if c not in ("lon", "lat")]
        assert len(band_cols) == 1, f"Expected 1 band column, got {len(band_cols)}"

    def test_to_xyz_int_band(self):
        """to_xyz with a single integer band should work."""
        arr = np.ones((2, 3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df = ds.to_xyz(bands=1)
        band_cols = [c for c in df.columns if c not in ("lon", "lat")]
        assert len(band_cols) == 1, "Should have exactly 1 band column"

    def test_to_xyz_invalid_bands_raises(self):
        """to_xyz with invalid bands type should raise ValueError."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        with pytest.raises(ValueError, match="integer or a list"):
            ds.to_xyz(bands="invalid")


class TestCreateFromArray:
    """Tests for create_from_array edge cases."""

    def test_missing_geo_and_top_left_raises(self):
        """create_from_array without geo or top_left_corner should raise."""
        arr = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="top_left_corner"):
            Dataset.create_from_array(arr, epsg=4326)

    def test_3d_array_creates_multi_band(self):
        """A 3D array should create a multi-band dataset."""
        arr = np.ones((4, 5, 6), dtype=np.float64)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
        )
        assert (
            ds.band_count == 4
        ), f"Expected 4 bands from 3D array, got {ds.band_count}"
        assert ds.rows == 5, f"Expected 5 rows, got {ds.rows}"
        assert ds.columns == 6, f"Expected 6 columns, got {ds.columns}"


class TestReadArray:
    """Tests for read_array method edge cases."""

    def test_read_array_invalid_band_raises(self, single_band_dataset):
        """Reading a non-existent band should raise ValueError."""
        with pytest.raises(ValueError, match="band index"):
            single_band_dataset.read_array(band=5)

    def test_read_array_window(self, single_band_dataset):
        """Reading with a window should return the correct subarray."""
        arr = single_band_dataset.read_array(band=0, window=[0, 0, 2, 2])
        assert arr.shape == (2, 2), f"Expected (2,2) window, got {arr.shape}"
        np.testing.assert_array_equal(
            arr,
            np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32),
            err_msg="Window values are wrong",
        )

    def test_read_array_multi_band_no_band_arg(self, multi_band_dataset):
        """Reading multi-band without band arg should return 3D array."""
        arr = multi_band_dataset.read_array()
        assert arr.ndim == 3, "Multi-band read should return 3D"
        assert arr.shape[0] == 3, "First dimension should be band count"

    def test_read_array_negative_band_raises(self, single_band_dataset):
        """B-12: negative band index now rejected with a clear error.

        Pre-fix the validator only checked the upper bound; a
        negative index slipped through to `_iloc` which raised
        `IndexError`. After centralising the band-bounds check the
        eager path raises a `ValueError` mentioning the valid range.
        """
        with pytest.raises(ValueError, match="band index should be between"):
            single_band_dataset.read_array(band=-1)

    @pytest.mark.lazy
    def test_read_array_negative_band_lazy_raises(self, single_band_dataset):
        """Lazy path: negative band index rejected with same error."""
        with pytest.raises(ValueError, match="band index should be between"):
            single_band_dataset.read_array(band=-1, chunks="auto")


class TestCopy:
    """Tests for the copy method."""

    def test_copy_in_memory(self, single_band_dataset):
        """copy() without path should produce an in-memory copy."""
        copied = single_band_dataset.copy()
        assert isinstance(copied, Dataset), "copy() should return Dataset"
        assert copied.access == "write", "Copied dataset should have write access"
        np.testing.assert_array_equal(
            copied.read_array(),
            single_band_dataset.read_array(),
            err_msg="Copied array differs from original",
        )
        assert id(copied) != id(
            single_band_dataset
        ), "Copy should be a different object"

    def test_copy_to_disk(self, single_band_dataset, tmp_path):
        """copy(path=...) should create a file on disk."""
        path = tmp_path / "test_copy.tif"
        copied = single_band_dataset.copy(path=path)
        assert path.exists(), "File should exist on disk"
        assert isinstance(copied, Dataset), "Should return Dataset"
        copied.close()


class TestToFile:
    """Tests for the to_file method."""

    def test_to_file_geotiff(self, tmp_path):
        """to_file should save to a .tif file."""
        arr = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
        )
        path = tmp_path / "output.tif"
        ds.to_file(path)
        assert path.exists(), "File should exist after to_file"
        reopened = Dataset.read_file(path)
        np.testing.assert_array_almost_equal(
            reopened.read_array(),
            arr,
            err_msg="File data differs from original",
        )

    def test_to_file_netcdf_roundtrip(self, tmp_path):
        """to_file('.nc') must write a NetCDF that can be reopened.

        Regression: the writer applied ``COMPRESS=DEFLATE`` to every driver,
        which forces the netCDF driver into the NC4C format. Some GDAL builds
        cannot read NC4C back, so ``to_file("out.nc")`` produced a file that
        no reader (not even GDAL) could open. The fix scopes the GeoTIFF
        creation options to the GeoTIFF driver only.
        """
        from pyramids.netcdf import NetCDF

        arr = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
        )
        path = tmp_path / "output.nc"
        ds.to_file(path)
        assert path.exists(), "File should exist after to_file"
        # The whole point of the regression: the written file is readable again.
        reopened = NetCDF.read_file(str(path))
        assert reopened.variable_names, "NetCDF should expose at least one variable"
        values = reopened.get_variable(reopened.variable_names[0]).read_array()
        np.testing.assert_array_almost_equal(
            np.asarray(values).squeeze(),
            arr,
            err_msg="NetCDF round-trip data differs from original",
        )

    def test_to_file_geotiff_keeps_deflate(self, tmp_path):
        """GeoTIFF output must stay DEFLATE-compressed by default.

        Locks the other half of the per-driver creation-options fix: scoping
        the options to GTiff must not drop the GeoTIFF default compression.
        """
        from osgeo import gdal

        arr = np.zeros((20, 20), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
        )
        path = tmp_path / "compressed.tif"
        ds.to_file(path)
        info = gdal.Info(str(path))
        assert "COMPRESSION=DEFLATE" in info, "GeoTIFF lost its default DEFLATE"

    def test_to_file_wrong_type_raises(self, single_band_dataset):
        """to_file with a non-string path should raise TypeError."""
        with pytest.raises(TypeError, match="string"):
            single_band_dataset.to_file(123)


class TestCreateDataset:
    """Tests for _create_dataset static method edge cases."""

    def test_create_in_memory(self):
        """_create_dataset without path creates an in-memory dataset."""
        ds = Dataset._create_dataset(5, 3, 1, gdal.GDT_Float32)
        assert ds is not None, "In-memory dataset should not be None"
        assert ds.RasterXSize == 5, "Columns should be 5"
        assert ds.RasterYSize == 3, "Rows should be 3"

    def test_create_on_disk(self, tmp_path):
        """_create_dataset with a .tif path creates a file on disk."""
        path = tmp_path / "test_create.tif"
        ds = Dataset._create_dataset(
            4, 4, 1, gdal.GDT_Float32, driver="GTiff", path=path
        )
        assert ds is not None, "Disk dataset should not be None"
        ds.FlushCache()
        ds = None
        assert path.exists(), "File should exist on disk"

    def test_create_non_string_path_raises(self):
        """_create_dataset with a non-string path should raise TypeError."""
        with pytest.raises(TypeError, match="string"):
            Dataset._create_dataset(4, 4, 1, gdal.GDT_Float32, driver="GTiff", path=123)

    def test_create_wrong_extension_raises(self, tmp_path):
        """_create_dataset with a non-.tif path for GTiff should raise."""
        path = str(tmp_path / "wrong.xyz")
        with pytest.raises(TypeError, match=".tif"):
            Dataset._create_dataset(
                4, 4, 1, gdal.GDT_Float32, driver="GTiff", path=path
            )


class TestReadBlockError:
    """Tests for _read_block error handling."""

    def test_read_block_out_of_bounds_raises(self, single_band_dataset):
        """Reading a block with a window outside raster bounds raises."""
        with pytest.raises(OutOfBoundsError):
            single_band_dataset._read_block(band=0, window=[0, 0, 100, 100])


class TestAddBand:
    """Tests for the add_band method."""

    def test_add_band_not_inplace(self, single_band_dataset):
        """add_band(inplace=False) should return a new Dataset with extra band."""
        new_arr = np.ones((3, 3), dtype=np.float32) * 42
        result = single_band_dataset.add_band(new_arr, inplace=False)
        assert result is not None, "add_band not inplace should return a Dataset"
        assert result.band_count == 2, f"Expected 2 bands, got {result.band_count}"

    def test_add_band_inplace(self, single_band_dataset):
        """add_band(inplace=True) should modify the dataset in place."""
        new_arr = np.ones((3, 3), dtype=np.float32) * 99
        result = single_band_dataset.add_band(new_arr, inplace=True)
        assert result is None, "add_band inplace should return None"
        assert (
            single_band_dataset.band_count == 2
        ), "Band count should increase after inplace add"

    def test_add_band_with_unit(self, single_band_dataset):
        """add_band with unit should set the unit on the new band."""
        new_arr = np.ones((3, 3), dtype=np.float32)
        result = single_band_dataset.add_band(new_arr, unit="meters", inplace=False)
        last_band = result._iloc(result.band_count - 1)
        assert (
            last_band.GetUnitType() == "meters"
        ), "Unit should be 'meters' on added band"


class TestTranslateWithPath:
    """Tests for translate with output path."""

    def test_translate_to_path(self, single_band_dataset, tmp_path):
        """translate with a path should save to a GTiff file."""
        path = tmp_path / "translated.tif"
        result = single_band_dataset.translate(path=path)
        assert isinstance(result, Dataset), "translate with path should return Dataset"
        assert path.exists(), "Translated file should exist on disk"


class TestWriteArrayErrors:
    """Tests for write_array error handling."""

    def test_write_array_default_top_left(self):
        """write_array with top_left_corner=None should default to (0,0)."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
        )
        new_data = np.full((3, 3), 77.0, dtype=np.float32)
        ds.write_array(new_data, top_left_corner=[0, 0])
        result = ds.read_array()
        assert np.all(
            result == pytest.approx(77.0)
        ), "All cells should be 77 after writing with None top_left"


class TestToFileOptions:
    """Tests for to_file with various creation options."""

    def test_to_file_with_tile_length_raises(self, single_band_dataset, tmp_path):
        """to_file with invalid tile_length exercises the RuntimeError handler."""
        path = str(tmp_path / "tiled.tif")
        with pytest.raises(FailedToSaveError):
            single_band_dataset.to_file(path, tile_length=256)

    def test_to_file_with_creation_options(self, single_band_dataset, tmp_path):
        """to_file with creation_options should pass them to GDAL."""
        path = tmp_path / "opts.tif"
        single_band_dataset.to_file(path, creation_options=["BIGTIFF=YES"])
        assert path.exists(), "Output file with creation options should exist"

    def test_to_file_runtime_error_raises(self, single_band_dataset):
        """to_file to an invalid path should raise FailedToSaveError."""
        with pytest.raises(FailedToSaveError):
            single_band_dataset.to_file("/nonexistent/path/to/file.tif")


class TestOverviews:
    """Tests for overview-related methods."""

    def test_overview_count_initially_zero(self, single_band_dataset):
        """overview_count should be [0] for a dataset without overviews."""
        counts = single_band_dataset.overview_count
        assert counts == [0], f"Expected [0] overview count, got {counts}"

    def test_create_overviews(self, tmp_path):
        """create_overviews should build overviews on a disk-based dataset."""
        arr = np.ones((64, 64), dtype=np.float32)
        path = str(tmp_path / "overview_test.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        ds.create_overviews(
            resampling_method="nearest",
            overview_levels=[2, 4],
        )
        counts = ds.overview_count
        assert counts[0] >= 2, f"Expected at least 2 overviews, got {counts[0]}"

    def test_create_overviews_invalid_levels_raises(self, tmp_path):
        """create_overviews with invalid levels should raise ValueError."""
        arr = np.ones((32, 32), dtype=np.float32)
        path = str(tmp_path / "ov_invalid.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        with pytest.raises(ValueError, match="power-of-two"):
            ds.create_overviews(overview_levels=[3, 5])

    def test_create_overviews_invalid_levels_type_raises(self, tmp_path):
        """create_overviews with non-list levels should raise TypeError."""
        arr = np.ones((32, 32), dtype=np.float32)
        path = str(tmp_path / "ov_type.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        with pytest.raises(TypeError, match="list"):
            ds.create_overviews(overview_levels=4)

    def test_create_overviews_invalid_method_raises(self, tmp_path):
        """create_overviews with invalid method raises ValueError."""
        arr = np.ones((32, 32), dtype=np.float32)
        path = str(tmp_path / "ov_method.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        with pytest.raises(ValueError, match="resampling_method"):
            ds.create_overviews(resampling_method="INVALID_METHOD")

    def test_get_overview_no_overviews_raises(self, single_band_dataset):
        """get_overview should raise if no overviews exist."""
        with pytest.raises(ValueError, match="no overviews"):
            single_band_dataset.get_overview(band=0, overview_index=0)

    def test_get_overview_and_read(self, tmp_path):
        """get_overview and read_overview_array should work after creation."""
        arr = np.arange(0, 64 * 64, dtype=np.float32).reshape(64, 64)
        path = str(tmp_path / "ov_read.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        ds.create_overviews(overview_levels=[2])
        ovr = ds.get_overview(band=0, overview_index=0)
        assert ovr is not None, "Overview should not be None"
        ovr_arr = ds.read_overview_array(band=0, overview_index=0)
        assert ovr_arr.ndim == 2, "Overview array should be 2D"
        assert (
            ovr_arr.shape[0] == 32
        ), f"Expected 32 rows for 2x overview, got {ovr_arr.shape[0]}"

    def test_get_overview_index_too_large_raises(self, tmp_path):
        """get_overview with too large index should raise ValueError."""
        arr = np.ones((64, 64), dtype=np.float32)
        path = str(tmp_path / "ov_idx.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        ds.create_overviews(overview_levels=[2])
        with pytest.raises(ValueError, match="less than"):
            ds.get_overview(band=0, overview_index=99)

    def test_recreate_overviews(self, tmp_path):
        """recreate_overviews should refresh overview data."""
        arr = np.ones((64, 64), dtype=np.float32)
        path = str(tmp_path / "ov_recreate.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        ds.create_overviews(overview_levels=[2])
        ds.recreate_overviews(resampling_method="nearest")
        counts = ds.overview_count
        assert counts[0] >= 1, "Overview count should be >= 1 after recreate"

    def test_recreate_overviews_invalid_method_raises(self, tmp_path):
        """recreate_overviews with invalid method raises ValueError."""
        arr = np.ones((64, 64), dtype=np.float32)
        path = str(tmp_path / "ov_bad_method.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        ds.create_overviews(overview_levels=[2])
        with pytest.raises(ValueError, match="resampling_method"):
            ds.recreate_overviews(resampling_method="BAD")

    def test_read_overview_array_multi_band(self, tmp_path):
        """read_overview_array with band=None on multi-band reads all bands."""
        arr = np.ones((3, 64, 64), dtype=np.float32)
        path = str(tmp_path / "ov_multi.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        ds.create_overviews(overview_levels=[2])
        ovr_arr = ds.read_overview_array(band=None, overview_index=0)
        assert ovr_arr.ndim == 3, "Multi-band overview should be 3D"
        assert ovr_arr.shape[0] == 3, "First dimension should be 3 bands"

    def test_read_overview_array_band_out_of_range(self, tmp_path):
        """read_overview_array with out-of-range band should raise."""
        arr = np.ones((64, 64), dtype=np.float32)
        path = str(tmp_path / "ov_oob.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        ds.create_overviews(overview_levels=[2])
        with pytest.raises(ValueError, match="band index"):
            ds.read_overview_array(band=99, overview_index=0)


class TestToXyzPath:
    """Tests for to_xyz with file path output."""

    def test_to_xyz_to_file(self, single_band_dataset, tmp_path):
        """to_xyz with a path should write to file and return None."""
        path = tmp_path / "output.xyz"
        result = single_band_dataset.to_xyz(path=path)
        assert result is None, "to_xyz with path should return None"
        assert path.exists(), "XYZ output file should exist"

    def test_to_xyz_all_bands_default(self):
        """to_xyz with bands=None should include all bands."""
        arr = np.ones((2, 3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df = ds.to_xyz()
        band_cols = [c for c in df.columns if c not in ("lon", "lat")]
        assert len(band_cols) == 2, f"Expected 2 band columns, got {len(band_cols)}"


class TestReadOverviewArrayBranches:
    """Tests for read_overview_array branching paths."""

    def test_read_overview_no_band_single_band(self, tmp_path):
        """read_overview_array band=None on single-band."""
        arr = np.ones((64, 64), dtype=np.float32)
        path = str(tmp_path / "ov_single.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        ds.create_overviews(overview_levels=[2])
        ovr_arr = ds.read_overview_array(band=None)
        assert ovr_arr.ndim == 2, "Single-band overview should be 2D when band=None"

    def test_read_overview_no_band_no_overview_raises(self, tmp_path):
        """read_overview_array band=None with no overview raises."""
        arr = np.ones((3, 64, 64), dtype=np.float32)
        path = str(tmp_path / "ov_none.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        with pytest.raises(ValueError, match="overviews"):
            ds.read_overview_array(band=None)

    def test_read_overview_band_no_overview_raises(self, tmp_path):
        """read_overview_array with band having no overviews raises."""
        arr = np.ones((64, 64), dtype=np.float32)
        path = str(tmp_path / "ov_noov.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds = Dataset.read_file(path, read_only=False)
        with pytest.raises(ValueError, match="overviews"):
            ds.read_overview_array(band=0)


class TestToXyzNoBands:
    """Tests for to_xyz with bands=None default."""

    def test_to_xyz_none_bands_single_band(self):
        """to_xyz with bands=None on single-band dataset."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df = ds.to_xyz(bands=None)
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert "lon" in df.columns, "Should have lon column"
        assert "lat" in df.columns, "Should have lat column"


class TestRecreateOverviewsReadOnly:
    """Tests for recreate_overviews on read-only dataset."""

    def test_recreate_overviews_read_only_raises(self, tmp_path):
        """recreate_overviews on read-only raises ReadOnlyError."""
        arr = np.ones((64, 64), dtype=np.float32)
        path = str(tmp_path / "ov_ro.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ds_rw = Dataset.read_file(path, read_only=False)
        ds_rw.create_overviews(overview_levels=[2])
        ds_rw.close()
        ds_ro = Dataset.read_file(path, read_only=True)
        with pytest.raises(ReadOnlyError):
            ds_ro.recreate_overviews()


class TestToXyzEdgeCases:
    """Tests for to_xyz edge cases."""

    def test_to_xyz_to_file_returns_none(self, single_band_dataset, tmp_path):
        """to_xyz with path outputs to file and returns None."""
        path = tmp_path / "xyz_out.xyz"
        result = single_band_dataset.to_xyz(path=path)
        assert result is None, "to_xyz with path returns None"
        assert path.exists(), "XYZ output file should exist on disk"


class TestWindow:
    """Tests for _window generator method."""

    def test_window_yields_tuples(self):
        """_window should yield (xoff, yoff, xsize, ysize) tuples."""
        arr = np.ones((10, 10), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        windows = list(ds.io._tile_offsets(size=5))
        assert len(windows) > 0, "Should yield at least 1 window"
        for w in windows:
            assert len(w) == 4, "Each window is (xoff, yoff, w, h)"

    def test_window_covers_raster(self):
        """_window should cover the entire raster."""
        arr = np.ones((7, 7), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        windows = list(ds.io._tile_offsets(size=3))
        assert len(windows) >= 4, f"Should yield at least 4 windows for 7x7 with size 3"


class TestGetTile:
    """Tests for get_tile generator method."""

    def test_get_tile_yields_arrays(self):
        """get_tile should yield numpy arrays."""
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        tiles = list(ds.get_tile(size=5))
        assert len(tiles) > 0, "Should yield at least 1 tile"
        for t in tiles:
            assert isinstance(t, np.ndarray), "Each tile should be a numpy array"


class TestWriteArrayException:
    """Tests for write_array exception path."""

    def test_write_array_wrong_shape_raises(self):
        """write_array with incompatible shape raises an exception."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
        )
        bad_arr = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(Exception):
            ds.write_array(bad_arr, top_left_corner=[0, 0])


class TestReadBlockReRaise:
    """Tests for _read_block re-raising non-OutOfBoundsError."""

    def test_read_block_generic_error(self):
        """_read_block re-raises errors that are not out-of-bounds."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
        )
        mock_band = MagicMock()
        mock_band.ReadAsArray.side_effect = RuntimeError("some read error")
        with patch.object(ds, "_iloc", return_value=mock_band):
            with pytest.raises(RuntimeError, match="some read"):
                ds._read_block(band=0, window=[0, 0, 2, 2])


class TestToFileBlockSize:
    """Tests for to_file with block_size configured."""

    def test_to_file_with_block_size(self, tmp_path):
        """to_file should include block size options when set."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        ds._block_size = [(256, 256)]
        path = tmp_path / "block.tif"
        ds.to_file(path)
        assert path.exists(), "File should exist after saving with block_size"


class TestMapBlocks:
    """Tests for the map_blocks method (block-by-block processing)."""

    def test_map_blocks_doubles_values(self):
        """map_blocks should apply the function to every tile.

        Test scenario:
            Apply x*2 to a 6x6 raster with tile_size=3. The result
            should have all values doubled.
        """
        arr = np.arange(1, 37, dtype=np.float32).reshape(6, 6)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
        )
        result = ds.map_blocks(lambda tile: tile * 2, tile_size=3)
        expected = arr * 2
        np.testing.assert_array_almost_equal(
            result.read_array(), expected, err_msg="map_blocks should double all values"
        )

    def test_map_blocks_preserves_spatial_metadata(self, single_band_dataset):
        """map_blocks result should have same geotransform, CRS, no_data_value.

        Test scenario:
            The output dataset should inherit all spatial metadata from
            the source.
        """
        result = single_band_dataset.map_blocks(lambda tile: tile, tile_size=2)
        assert (
            result.geotransform == single_band_dataset.geotransform
        ), "Geotransform should be preserved"
        assert result.epsg == single_band_dataset.epsg, "EPSG should be preserved"
        assert (
            result.no_data_value == single_band_dataset.no_data_value
        ), "No-data value should be preserved"

    def test_map_blocks_identity_matches_read_array(self):
        """map_blocks with identity function should produce the same array.

        Test scenario:
            Applying an identity function tile-by-tile should yield the
            exact same array as read_array().
        """
        arr = np.random.default_rng(42).random((10, 10)).astype(np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
        )
        result = ds.map_blocks(lambda tile: tile, tile_size=4)
        np.testing.assert_array_equal(
            result.read_array(),
            ds.read_array(),
            err_msg="Identity map_blocks should reproduce the original",
        )

    def test_map_blocks_single_band(self):
        """map_blocks with band parameter should process only that band.

        Test scenario:
            On a 2-band dataset, map_blocks(band=1) should produce a
            1-band output with only band 1's values transformed.
        """
        arr = np.ones((2, 4, 4), dtype=np.float32)
        arr[0, :, :] = 10
        arr[1, :, :] = 20
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
        )
        result = ds.map_blocks(lambda tile: tile + 5, tile_size=2, band=1)
        assert result.band_count == 1, f"Expected 1 band, got {result.band_count}"
        result_arr = result.read_array()
        assert np.allclose(
            result_arr, 25.0
        ), f"Expected all 25.0 (20+5), got {result_arr}"

    def test_map_blocks_non_square_raster(self):
        """map_blocks should handle non-square rasters with partial edge tiles.

        Test scenario:
            A 7x5 raster with tile_size=3 will have partial tiles at the
            edges. All tiles should be processed correctly.
        """
        arr = np.ones((7, 5), dtype=np.float32) * 3.0
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
        )
        result = ds.map_blocks(lambda tile: tile + 1, tile_size=3)
        assert np.allclose(
            result.read_array(), 4.0
        ), "All cells should be 4.0 (3+1), including edge tiles"

    def test_map_blocks_tile_size_larger_than_raster(self):
        """map_blocks should work when tile_size exceeds the raster dimensions.

        Test scenario:
            A single tile covers the entire raster — should behave like
            a normal array operation.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
        )
        result = ds.map_blocks(lambda tile: tile**2, tile_size=1000)
        expected = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            result.read_array(),
            expected,
            err_msg="Single-tile map_blocks should square values",
        )

    def test_map_blocks_multi_band_all_bands(self):
        """map_blocks without band parameter should process all bands.

        Test scenario:
            A 3-band dataset processed with map_blocks(band=None) should
            produce a 3-band output with all bands transformed.
        """
        arr = np.ones((3, 4, 4), dtype=np.float32)
        arr[0] = 1
        arr[1] = 2
        arr[2] = 3
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
        )
        result = ds.map_blocks(lambda tile: tile * 10, tile_size=2)
        assert result.band_count == 3, f"Expected 3 bands, got {result.band_count}"
        assert np.allclose(result.read_array(band=0), 10.0), "Band 0 should be 10"
        assert np.allclose(result.read_array(band=1), 20.0), "Band 1 should be 20"
        assert np.allclose(result.read_array(band=2), 30.0), "Band 2 should be 30"
