"""The GDAL to numpy dtype map has one reader, not three.

`Dataset.dtype` and `Dataset.numpy_dtype` each scanned `DTYPE_CONVERSION_DF`
with their own boolean mask, and `RasterMeta.from_dataset` carried a third
fallback that derived the dtype from the GDAL band directly. All three answer
the same question, and the frame scan costs a pandas mask per band on the
`read_array` hot path.

They now read `_GDAL_TO_NUMPY` through `gdal_to_numpy_dtype` (the name) and
`gdal_to_numpy_type` (the type).
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._raster_meta import RasterMeta
from pyramids.base._utils import gdal_to_numpy_dtype, gdal_to_numpy_type
from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.core

DTYPES = ["uint8", "int16", "uint16", "int32", "uint32", "float32", "float64"]


class TestTheTwoHelpersAgree:
    """The name and the type describe the same numpy dtype."""

    @pytest.mark.parametrize("name", DTYPES)
    def test_name_and_type_agree(self, name: str):
        """`gdal_to_numpy_dtype` names what `gdal_to_numpy_type` returns."""
        code = gdal.GetDataTypeByName(
            {
                "uint8": "Byte",
                "int16": "Int16",
                "uint16": "UInt16",
                "int32": "Int32",
                "uint32": "UInt32",
                "float32": "Float32",
                "float64": "Float64",
            }[name]
        )

        assert np.dtype(gdal_to_numpy_type(code)).name == gdal_to_numpy_dtype(code)

    def test_a_placeholder_code_raises(self):
        """`GDT_Unknown` has no numpy counterpart and says so."""
        with pytest.raises(ValueError, match="unsupported GDAL data type"):
            gdal_to_numpy_type(gdal.GDT_Unknown)

    def test_the_other_placeholder_raises_too(self):
        """`GDT_TypeCount` is the map's upper bound, not a type.

        Test scenario:
            GDAL's enum ends with a count sentinel. It sits one past the last
            real code, so a map keyed by range rather than by membership would
            hand back the last type's numpy counterpart instead of raising.
        """
        with pytest.raises(ValueError, match="unsupported GDAL data type"):
            gdal_to_numpy_type(gdal.GDT_TypeCount)

    @pytest.mark.parametrize("code", [-1, 9999], ids=["negative", "far-out-of-range"])
    def test_a_code_outside_the_enum_raises_and_lists_what_is_supported(self, code):
        """The error has to be actionable, not just a KeyError.

        Args:
            code: A GDAL data-type code that does not exist.

        Test scenario:
            An out-of-range code names the offending value and enumerates the
            supported codes, so the caller can see what to pass instead.
        """
        with pytest.raises(ValueError) as exc_info:
            gdal_to_numpy_type(code)

        message = str(exc_info.value)
        assert str(code) in message, f"the offending code should appear: {message}"
        assert "Supported types are" in message, f"no supported list in: {message}"


class TestDatasetDtypeProperties:
    """The two properties round-trip whatever the raster was built with."""

    @pytest.mark.parametrize("name", DTYPES)
    def test_dtype_reports_what_was_written(self, name: str):
        """`dtype` names the band's actual type."""
        dataset = Dataset.from_array(
            np.ones((4, 4), dtype=name),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        )

        assert dataset.dtype[0] == name

    @pytest.mark.parametrize("name", DTYPES)
    def test_numpy_dtype_matches_dtype(self, name: str):
        """The type form and the name form stay in step."""
        dataset = Dataset.from_array(
            np.ones((4, 4), dtype=name),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        )

        assert np.dtype(dataset.numpy_dtype[0]).name == dataset.dtype[0]


class TestRasterMetaDtype:
    """`RasterMeta` reports the same dtype as the dataset it came from."""

    @pytest.mark.parametrize("name", DTYPES)
    def test_meta_matches_the_dataset(self, name: str):
        """The third fallback is gone; there is one answer."""
        dataset = Dataset.from_array(
            np.ones((4, 4), dtype=name),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        )

        assert RasterMeta.from_dataset(dataset).dtype == dataset.dtype[0]
