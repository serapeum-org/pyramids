"""Unit tests for Dataset no-data value handling and gap filling."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyramids.base._errors import NoDataValueError, ReadOnlyError
from pyramids.dataset import Dataset
from pyramids.dataset.abstract_dataset import DEFAULT_NO_DATA_VALUE

pytestmark = pytest.mark.core


class TestSetNoDataValueErrors:
    """Tests for _set_no_data_value error handling."""

    def test_set_nodata_read_only_raises(self, tmp_path):
        """_set_no_data_value on a read-only dataset should raise ReadOnlyError."""
        arr = np.ones((3, 3), dtype=np.float32)
        path = str(tmp_path / "readonly.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ro_ds = Dataset.read_file(path, read_only=True)
        with pytest.raises(ReadOnlyError):
            ro_ds.bands._set_no_data_value(-1234.0)


class TestChangeNoDataValueAttr:
    """Tests for _change_no_data_value_attr method."""

    def test_change_nodata_attr_updates_internal(self, single_band_dataset):
        """_change_no_data_value_attr should update the internal list."""
        single_band_dataset.bands._change_no_data_value_attr(0, -1111.0)
        assert (
            single_band_dataset.no_data_value[0] == -1111.0
        ), "no_data_value attribute not updated"

    def test_no_data_value_setter_with_list(self, multi_band_dataset):
        """Setting no_data_value with a list should update all bands."""
        multi_band_dataset.no_data_value = [-1.0, -2.0, -3.0]
        assert multi_band_dataset.no_data_value == (
            -1.0,
            -2.0,
            -3.0,
        ), "no_data_value list setter failed"

    def test_no_data_value_setter_with_scalar(self, single_band_dataset):
        """Setting no_data_value with a scalar should update band 0."""
        single_band_dataset.no_data_value = -5555.0
        assert (
            single_band_dataset.no_data_value[0] == -5555.0
        ), "no_data_value scalar setter failed"

    def test_no_data_value_getter_returns_tuple(self, single_band_dataset):
        """B-17: getter returns immutable tuple, not list.

        The read-only contract is now expressed at the type level —
        mutating the returned object can never propagate to internal
        state. Use the setter to change values.
        """
        ndv = single_band_dataset.no_data_value
        assert isinstance(ndv, tuple), f"Expected tuple, got {type(ndv)}"

    def test_no_data_value_setter_scalar_broadcasts_to_all_bands(
        self, multi_band_dataset
    ):
        """B-17: scalar setter now writes every band, not just band 0.

        Pre-fix only band 0 was updated; bands 1..N kept stale
        values. Post-fix the scalar is broadcast to all bands.
        """
        multi_band_dataset.no_data_value = -7777.0
        assert multi_band_dataset.no_data_value == (
            -7777.0,
            -7777.0,
            -7777.0,
        )

    def test_no_data_value_setter_length_mismatch_raises(self, multi_band_dataset):
        """B-17: list/tuple shorter than band_count is rejected up-front.

        Pre-fix a too-short sequence silently set only the first N
        bands; the remaining bands kept their old values with no
        warning. Post-fix the setter raises ValueError early.
        """
        with pytest.raises(ValueError, match="does not match band_count"):
            multi_band_dataset.no_data_value = [-1.0]

    def test_no_data_value_setter_accepts_tuple(self, multi_band_dataset):
        """B-17: setter accepts tuple inputs alongside lists."""
        multi_band_dataset.no_data_value = (-1.0, -2.0, -3.0)
        assert multi_band_dataset.no_data_value == (-1.0, -2.0, -3.0)

    def test_no_data_value_setter_accepts_1d_ndarray(self, multi_band_dataset):
        """M4: 1-D ndarray with len == band_count is treated as per-band."""
        multi_band_dataset.no_data_value = np.array(
            [-1.0, -2.0, -3.0], dtype=np.float64
        )
        assert multi_band_dataset.no_data_value == (-1.0, -2.0, -3.0)

    def test_no_data_value_setter_accepts_0d_ndarray(self, multi_band_dataset):
        """M4: 0-D ndarray is treated as a scalar broadcast to every band."""
        multi_band_dataset.no_data_value = np.array(-7.0, dtype=np.float64)
        assert multi_band_dataset.no_data_value == (-7.0, -7.0, -7.0)

    def test_no_data_value_setter_rejects_2d_ndarray(self, multi_band_dataset):
        """M4: 2-D ndarray inputs are rejected with a clear error."""
        with pytest.raises(ValueError, match="ndarray must be 0-D"):
            multi_band_dataset.no_data_value = np.zeros((2, 3), dtype=np.float64)

    def test_no_data_value_setter_ndarray_length_mismatch_raises(
        self, multi_band_dataset
    ):
        """M4: 1-D ndarray with wrong length goes through the same check."""
        with pytest.raises(ValueError, match="does not match band_count"):
            multi_band_dataset.no_data_value = np.array([-1.0], dtype=np.float64)


class TestFillGaps:
    """Tests for the fill_gaps method."""

    def test_fill_gaps_basic(self):
        """fill_gaps should fill src nodata cells where the mask has valid data."""
        nd = -9999.0
        # mask has valid cells everywhere
        mask_arr = np.ones((3, 3), dtype=np.float32) * 5.0
        mask_ds = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        # src has one cell as nodata that the mask says is valid
        src_arr = np.ones((3, 3), dtype=np.float32) * 10.0
        src_arr[1, 1] = nd
        src_ds = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        result = src_ds.fill_gaps(mask_ds, src_arr.copy())
        # The gap cell should now be filled (not nodata)
        assert not np.isclose(
            result[1, 1], nd, rtol=0.001
        ), "The gap cell (1,1) should have been filled"


class TestCheckNoDataValue:
    """Tests for _check_no_data_value method."""

    def test_check_nodata_nan_float(self):
        """NaN no-data for float dtype should be preserved as NaN."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=None,
        )
        ndv = ds.no_data_value[0]
        # For float types, None maps to NaN (or stays None)
        assert ndv is None or np.isnan(
            ndv
        ), f"Expected None or NaN for float no_data with None input, got {ndv}"

    def test_check_nodata_overflow(self):
        """No-data value that overflows the dtype should fall back to a valid sentinel."""
        arr = np.ones((3, 3), dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-3.4028230607370965e38,
        )
        ndv = ds.no_data_value[0]
        assert ndv is not None, "overflowing no_data_value should fall back, not stay None"


class TestFill:
    """Tests for the fill method."""

    def test_fill_value(self, single_band_dataset):
        """fill should replace all domain cells with the given value."""
        filled = single_band_dataset.fill(42)
        arr = filled.read_array()
        assert np.all(arr == 42), "All cells should be 42 after fill"

    def test_fill_inplace(self, single_band_dataset):
        """fill(inplace=True) should modify the dataset in place."""
        result = single_band_dataset.fill(99, inplace=True)
        assert result is single_band_dataset, "inplace fill should return self"
        arr = single_band_dataset.read_array()
        assert np.all(arr == 99), "All cells should be 99 after inplace fill"


class TestSetNoDataValueEdge:
    """Tests for _set_no_data_value edge-case error handling."""

    def test_set_nodata_type_conversion(self):
        """_set_no_data_value with a value needing float64 conversion."""
        arr = np.ones((3, 3), dtype=np.float64)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        ds.bands._set_no_data_value([-1234.0])
        assert ds.no_data_value[0] == -1234.0, "No data value should be updated"


class TestSetNoDataValueBackend:
    """Tests for _set_no_data_value_backend error handling."""

    def test_backend_read_only_raises(self, tmp_path):
        """_set_no_data_value_backend on read-only dataset raises ReadOnlyError."""
        arr = np.ones((3, 3), dtype=np.float32)
        path = str(tmp_path / "ro_backend.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        ro_ds = Dataset.read_file(path, read_only=True)
        with pytest.raises(ReadOnlyError):
            ro_ds.bands._set_no_data_value_backend(0, -1234.0)


class TestChangeNoDataValueNan:
    """Tests for change_no_data_value with NaN old values."""

    def test_change_nodata_nan_old_value(self):
        """change_no_data_value with None old_value uses np.isnan path."""
        arr = np.array(
            [[np.nan, 2.0], [3.0, np.nan]],
            dtype=np.float32,
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=np.nan,
        )
        new_ds = ds.change_no_data_value(-9999.0, old_value=None)
        result = new_ds.read_array()
        assert np.isclose(result[0, 0], -9999.0), "NaN cells should now be -9999"
        assert np.isclose(result[0, 1], 2.0), "Valid cells should remain unchanged"

    def test_change_nodata_list_new_value(self):
        """change_no_data_value with new_value as list (branch 3016)."""
        arr = np.array([[-9999.0, 2.0], [3.0, -9999.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        new_ds = ds.change_no_data_value([-1.0], old_value=-9999.0)
        result = new_ds.read_array()
        assert np.isclose(
            result[0, 0], -1.0
        ), "Old nodata cells should be replaced with list"

    def test_change_nodata_list_old_value(self):
        """change_no_data_value with old_value as list."""
        arr = np.array([[-9999.0, 2.0], [3.0, -9999.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        new_ds = ds.change_no_data_value(-1.0, old_value=[-9999.0])
        result = new_ds.read_array()
        assert np.isclose(
            result[0, 0], -1.0
        ), "Old nodata cells should be replaced with list old"


class TestFillNanNodata:
    """Tests for fill method with NaN no_data_value."""

    def test_fill_with_nan_nodata(self):
        """fill should work when no_data_value is None/NaN."""
        arr = np.array([[np.nan, 2.0], [3.0, np.nan]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=None,
        )
        filled = ds.fill(42)
        result = filled.read_array()
        non_nan = result[~np.isnan(result)]
        assert np.all(non_nan == 42), "Non-NaN cells should be 42 after fill"

    def test_fill_non_nan_nodata(self):
        """fill should replace all non-nodata cells when nodata is set."""
        nd = -9999.0
        arr = np.array([[nd, 2.0], [3.0, nd]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        filled = ds.fill(10)
        result = filled.read_array()
        assert np.isclose(result[0, 1], 10.0), "Valid cell should be set to 10"


class TestFillNoneNodata:
    """Tests for fill when no_data_value is None."""

    def test_fill_none_nodata_value(self):
        """fill should handle no_data_value=None by treating NaN as nodata."""
        arr = np.array([[np.nan, 2.0], [3.0, np.nan]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        # Manually set the internal nodata to None to exercise the
        # fallback branch that infers the sentinel from NaN cells
        ds._no_data_value = [None]
        filled = ds.fill(42)
        result = filled.read_array()
        # Non-NaN cells should be set to 42
        assert np.isclose(
            result[0, 1], 42.0
        ), "Valid cell should be 42 after fill with None nodata"


class TestSetNoDataValueRecovery:
    """Tests for _set_no_data_value error recovery branches."""

    def test_set_nodata_with_incompatible_dtype(self):
        """_set_no_data_value with value needing float64 conversion."""
        arr = np.ones((3, 3), dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds.bands._set_no_data_value([-9999])
        assert ds.no_data_value[0] is not None, "No data value should be set"


class TestCreateNoDataNone:
    """Tests for create with no_data_value=None."""

    def test_create_without_nodata(self):
        """create without no_data_value should not set nodata."""
        ds = Dataset.create(
            rows=3,
            columns=3,
            cell_size=0.05,
            dtype="float32",
            bands=1,
            top_left_corner=(0.0, 0.0),
            epsg=4326,
        )
        assert ds is not None, "Dataset should be created"
        assert ds.rows == 3, "Should have 3 rows"


class TestSetNoDataValueMocked:
    """Tests for _set_no_data_value error paths using mocks."""

    def test_set_nodata_double_conversion_via_mock(self):
        """_set_no_data_value retries with float64 on type error."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        err_msg = "in method 'Band_SetNoDataValue', argument 2 of type 'double'"
        call_count = [0]
        original = ds.bands._set_no_data_value_backend

        def side_effect(band, val):
            """Raise on first call, succeed on retry."""
            call_count[0] += 1
            if call_count[0] == 1:
                raise TypeError(err_msg)
            original(band, val)

        with patch.object(
            ds.bands,
            "_set_no_data_value_backend",
            side_effect=side_effect,
        ):
            ds.bands._set_no_data_value([-1234.0])
        assert call_count[0] >= 2, "Should have retried after TypeError"

    def test_set_nodata_fallback_to_default_via_mock(self):
        """_set_no_data_value falls back to DEFAULT_NO_DATA_VALUE."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        call_count = [0]
        original = ds.bands._set_no_data_value_backend

        def side_effect(band, val):
            """Raise on first call, succeed on retry."""
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("some unknown error")
            original(band, val)

        with patch.object(
            ds.bands,
            "_set_no_data_value_backend",
            side_effect=side_effect,
        ):
            ds.bands._set_no_data_value([-1234.0])
        assert call_count[0] >= 2, "Should have retried with default value"
        assert ds.no_data_value[0] == pytest.approx(DEFAULT_NO_DATA_VALUE), (
            "an unknown backend error should fall back to DEFAULT_NO_DATA_VALUE, "
            "not leave the requested -1234.0 in place"
        )


class TestSetNoDataValueBackendMocked:
    """Tests for _set_no_data_value_backend error paths using mocks."""

    def test_backend_type_conversion_error(self):
        """_set_no_data_value_backend retries with float64."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        err_msg = " argument 2 of type 'double'"
        original_get_band = ds.raster.GetRasterBand
        call_count = [0]

        def mock_get_band(band_num):
            """Return a mock band that fails Fill on first call."""
            real_band = original_get_band(band_num)
            original_fill = real_band.Fill
            wrapper_count = call_count

            def mock_fill(val):
                wrapper_count[0] += 1
                if wrapper_count[0] == 1:
                    raise RuntimeError(err_msg)
                return original_fill(val)

            real_band.Fill = mock_fill
            return real_band

        with patch.object(ds.raster, "GetRasterBand", mock_get_band):
            ds.bands._set_no_data_value_backend(0, -1234.0)

        assert ds.no_data_value[0] == pytest.approx(-1234.0), (
            "after the Fill retry the band nodata should hold the requested value"
        )

    def test_backend_generic_error_raises(self):
        """_set_no_data_value_backend raises ValueError on unknown error."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        original_get_band = ds.raster.GetRasterBand

        def mock_get_band(band_num):
            """Return a mock band that always fails Fill."""
            real_band = original_get_band(band_num)

            def mock_fill(val):
                raise RuntimeError("some strange error")

            real_band.Fill = mock_fill
            return real_band

        with patch.object(ds.raster, "GetRasterBand", mock_get_band):
            with pytest.raises(ValueError, match="Failed to fill"):
                ds.bands._set_no_data_value_backend(0, -1234.0)


class TestChangeNoDataValueTypeError:
    """Tests for change_no_data_value TypeError path via mock."""

    def test_change_nodata_type_error_raises(self):
        """change_no_data_value catches TypeError and raises NoDataValueError."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        original_read = ds.read_array

        def mock_read(band=None):
            """Return array that raises TypeError on assignment."""
            result = original_read(band=band)
            mock_arr = MagicMock(wraps=result)

            def raise_type_error(key, value):
                raise TypeError("incompatible type")

            mock_arr.__setitem__ = raise_type_error
            mock_arr.__getitem__ = result.__getitem__
            return mock_arr

        with patch.object(ds, "read_array", mock_read):
            with pytest.raises(NoDataValueError):
                ds.change_no_data_value(-1.0, old_value=-9999.0)


class TestChangeNoDataAttrConversion:
    """Tests for _change_no_data_value_attr type conversion."""

    def test_change_nodata_attr_type_conversion(self):
        """_change_no_data_value_attr converts to float64 on error."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        call_count = [0]
        original_get_band = ds.raster.GetRasterBand

        def mock_get_band(band_num):
            """Return band with mocked SetNoDataValue."""
            real_band = original_get_band(band_num)
            original_set = real_band.SetNoDataValue

            def mock_set(val):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError(
                        "in method 'Band_SetNoDataValue', "
                        "argument 2 of type 'double'"
                    )
                return original_set(val)

            real_band.SetNoDataValue = mock_set
            return real_band

        with patch.object(ds.raster, "GetRasterBand", mock_get_band):
            ds.bands._change_no_data_value_attr(0, -1234.0)
        assert (
            ds.no_data_value[0] == -1234.0
        ), "nodata should be updated after type conversion"

    def test_change_nodata_attr_read_only_error(self):
        """_change_no_data_value_attr raises ReadOnlyError on write fail."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        original_get_band = ds.raster.GetRasterBand
        err_msg = "Attempt to write to read only dataset in GDALRasterBand::Fill()."

        def mock_get_band(band_num):
            """Return band that raises on SetNoDataValue."""
            real_band = original_get_band(band_num)
            real_band.SetNoDataValue = MagicMock(side_effect=RuntimeError(err_msg))
            return real_band

        with patch.object(ds.raster, "GetRasterBand", mock_get_band):
            with pytest.raises(ReadOnlyError):
                ds.bands._change_no_data_value_attr(0, -1234.0)


class TestFillGapsLessNodata:
    """Tests for fill_gaps where mask has more valid cells."""

    def test_fill_gaps_mask_more_valid(self):
        """fill_gaps when mask has more valid cells than src."""
        nd = -9999.0
        mask_arr = np.ones((3, 3), dtype=np.float32) * 5.0
        mask_ds = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        src_arr = np.ones((3, 3), dtype=np.float32) * 10.0
        src_arr[0, 0] = nd
        src_arr[1, 1] = nd
        src_ds = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        result = src_ds.fill_gaps(mask_ds, src_arr.copy())
        # The mask has more valid cells than src, so both src gaps are filled
        # from their valid neighbours (all 10.0) and no nodata should remain.
        assert result[0, 0] == pytest.approx(10.0)
        assert result[1, 1] == pytest.approx(10.0)
        assert not np.any(result == nd), "every src gap should have been filled"

    def test_fill_gaps_equal_valid(self):
        """fill_gaps when mask and src have same valid cells."""
        nd = -9999.0
        mask_arr = np.ones((3, 3), dtype=np.float32) * 5.0
        mask_arr[1, 1] = nd
        mask_ds = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        src_arr = np.ones((3, 3), dtype=np.float32) * 10.0
        src_arr[1, 1] = nd
        src_ds = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        result = src_ds.fill_gaps(mask_ds, src_arr.copy())
        # Equal valid-cell counts, so no interpolation happens and the src gap
        # at (1, 1) is left untouched.
        assert result[1, 1] == pytest.approx(nd), "equal valid counts should skip filling"
        assert result[0, 0] == pytest.approx(10.0)
