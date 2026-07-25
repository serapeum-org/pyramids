"""Integration tests for Dataset no-data value handling and gap filling."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import NoDataValueError, ReadOnlyError
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestNoDataValue:
    def test_set_no_data_value_error_read_only(
        self,
        src_set_no_data_value: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src_set_no_data_value)
        # The fill-based path writes pixels; on a read-only handle it must raise
        # deterministically via the access flag, not depend on GDAL's error text (N3).
        with pytest.raises(ReadOnlyError):
            src.bands._set_no_data_value(-99999.0)

    def test_no_data_value_setter_rejects_read_only(
        self,
        src_set_no_data_value: gdal.Dataset,
    ):
        """The no_data_value setter raises ReadOnlyError on a read-only on-disk dataset.

        Consistent with the other metadata setters: mutating the marker on a read-only
        on-disk handle would change only the in-memory attribute without persisting, so
        it is rejected. Reopen with read_only=False (or edit an in-memory copy) instead.
        """
        src = Dataset(src_set_no_data_value)
        assert src.access == "read_only", "fixture must be read-only for this test"
        with pytest.raises(ReadOnlyError, match="read-only"):
            src.no_data_value = -123.0

    def test_set_no_data_value(
        self,
        src_update: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src_update)
        src.bands._set_no_data_value(5.0)
        # check if the no_data_value in the Dataset object is set
        assert src.raster.GetRasterBand(1).GetNoDataValue() == 5
        # check if the no_data_value of the Dataset object is set5
        assert src.no_data_value[0] == 5

    def test_change_no_data_value(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src)
        arr = src.read_array()
        old_value = arr[0, 0]
        new_val = -6666
        src = src.change_no_data_value(new_val, old_value)
        # check if the no_data_value in the Datacube object is set
        assert src.raster.GetRasterBand(1).GetNoDataValue() == new_val
        # check if the no_data_value of the Dataset object is set
        assert src.no_data_value[0] == new_val
        # check that the no_data_value type has changed to float like the band dtype
        assert isinstance(src.no_data_value[0], float)
        # check if the new_val for the no_data_value is set in the bands
        arr = src.read_array(0)
        val = arr[0, 0]
        assert val == new_val

    def test_change_no_data_value_setter(
        self,
        chang_no_data_dataset: gdal.Dataset,
        src_no_data_value: float,
    ):
        """
        check setting the gdal attribute only but not the value of the nodata cells
        """
        # copy() yields a writable in-memory dataset; the metadata setter is guarded
        # against a read-only on-disk handle (the fixture opens GA_ReadOnly).
        dataset = Dataset(chang_no_data_dataset).copy()
        new_val = -6666
        dataset.no_data_value = new_val
        # check if the no_data_value in the Dataset object is set
        assert dataset.raster.GetRasterBand(1).GetNoDataValue() == new_val
        # check if the no_data_value of the Dataset object is set
        assert dataset.no_data_value == (new_val,)

    def test_change_no_data_error_different_data_type(
        self, int_none_nodatavalue_attr_0_stored: gdal.Dataset
    ):
        # try to store None in the array (int)
        dataset = Dataset(int_none_nodatavalue_attr_0_stored)
        with pytest.raises(NoDataValueError):
            dataset.change_no_data_value(None, 0)

    def test_change_no_data_nan_into_int_band_raises(self):
        """A NaN no-data into an integer band is a dtype mismatch -> NoDataValueError.

        The NaN->int cast raises ``ValueError`` ("cannot convert float NaN to
        integer") rather than ``TypeError``/``FloatingPointError``; the guard must
        still surface it as the package-level ``NoDataValueError`` and not leak a raw
        numpy error to the caller.
        """
        dataset = Dataset.create_from_array(
            np.arange(9, dtype="int32").reshape(3, 3),
            geo=(0, 1, 0, 3, 0, -1),
            epsg=4326,
            no_data_value=0,
        )
        with pytest.raises(NoDataValueError):
            dataset.change_no_data_value(np.nan, 0)

    @pytest.mark.parametrize(
        "dtype, expected",
        [
            ("uint8", 255),
            ("uint16", 65535),
            ("uint32", 4294967295),
            ("int8", -128),
        ],
    )
    def test_create_from_array_default_nodata_fits_small_dtype(self, dtype, expected):
        """``create_from_array`` with the default no-data falls back when -9999 overflows.

        Regression: the default ``no_data_value`` (-9999) does not fit unsigned
        integer bands (or ``int8``); the conversion raised ``OverflowError`` and
        the handler re-raised by re-casting the same value. It now falls back to
        a dtype-valid sentinel (the dtype max for unsigned ints, the dtype min
        for too-small signed ints) and warns instead of crashing.

        Args:
            dtype: A small integer dtype that cannot represent -9999.
            expected: The dtype-valid no-data the fallback should pick.

        Test scenario:
            Build a single-band raster of ``dtype`` without passing
            ``no_data_value`` — expected: construction succeeds, warns about the
            out-of-range value, and stores ``expected`` as the no-data value.
        """
        arr = np.full((4, 4), 5, dtype=dtype)
        with pytest.warns(UserWarning, match="out of range"):
            dataset = Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            )
        assert dataset.no_data_value[0] == expected, (
            f"{dtype}: expected fallback no-data {expected}, "
            f"got {dataset.no_data_value[0]}"
        )

    @pytest.mark.parametrize("dtype", ["int16", "int32", "float32", "float64"])
    def test_create_from_array_default_nodata_fits_large_dtype(self, dtype):
        """Dtypes that can hold -9999 keep the default no-data and do not warn.

        Args:
            dtype: An integer/float dtype wide enough for -9999.

        Test scenario:
            Build a raster without ``no_data_value`` — expected: no warning and
            the no-data value is the default -9999.
        """
        arr = np.full((4, 4), 5, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dataset = Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            )
        assert dataset.no_data_value[0] == -9999, (
            f"{dtype}: expected default no-data -9999, got {dataset.no_data_value[0]}"
        )


class TestFillRaster:
    def test_memory_raster(
        self, src: gdal.Dataset, fill_raster_path: Path, fill_raster_value: int
    ):
        src = Dataset(src)
        dst = src.fill(fill_raster_value)
        arr = dst.raster.ReadAsArray()
        no_data_val = dst.raster.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, no_data_val, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals[0] == fill_raster_value

        # test inplace
        src.fill(fill_raster_value, inplace=True)
        arr = src.raster.ReadAsArray()
        vals = arr[~np.isclose(arr, no_data_val, rtol=0.00000000000001)]
        assert vals[0] == fill_raster_value

    def test_disk_raster(
        self, src: gdal.Dataset, fill_raster_path: Path, fill_raster_value: int
    ):
        if fill_raster_path.exists():
            fill_raster_path.unlink()
        src = Dataset(src)
        src.fill(fill_raster_value, path=fill_raster_path)
        "now the resulted raster is saved to disk"
        dst = gdal.Open(str(fill_raster_path))
        arr = dst.ReadAsArray()
        no_data_val = dst.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, no_data_val, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals[0] == fill_raster_value
