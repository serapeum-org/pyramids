"""What `Dataset.dtype` reports, and why nothing may branch on that string.

`dtype` used to read the driver catalog's own `name` column; it now reads the
numpy type's name, so a few strings changed -- `byte` became `uint8`, and the
complex codes collapsed onto numpy's two widths. The new strings are the
correct ones (`uint8` is what the STAC `raster:bands.data_type` field wants),
but the surface is public, so it is pinned here rather than left implicit.

One consumer was branching on it. `_coerce_band_no_data` decided "is this band
unsigned" with `dtype[i].startswith("u")`, which `"byte"` failed -- so the one
unsigned type most rasters actually use took the signed pass-through branch and
a `None` no-data was left unset instead of becoming the dtype max. It now asks
the dtype, like `_fallback_no_data` beside it already did.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._utils import gdal_to_numpy_dtype
from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.core

GEO = GeoReference(top_left_corner=(0.0, 5.0), cell_size=1.0, epsg=4326)


def _raster(dtype) -> Dataset:
    """A 4x5 single-band raster of the given numpy dtype."""
    return Dataset.from_array(np.ones((4, 5), dtype=dtype), geo_ref=GEO)


class TestTheReportedDtypeStrings:
    """The public surface, pinned per type."""

    @pytest.mark.parametrize(
        ("numpy_dtype", "reported"),
        [
            (np.uint8, "uint8"),
            (np.uint16, "uint16"),
            (np.int16, "int16"),
            (np.int32, "int32"),
            (np.uint32, "uint32"),
            (np.float32, "float32"),
            (np.float64, "float64"),
        ],
    )
    def test_the_dtype_is_the_numpy_name(self, numpy_dtype, reported: str):
        """`byte` is gone; a Byte raster reports `uint8`.

        Args:
            numpy_dtype: The dtype the raster is built with.
            reported: What `Dataset.dtype` must call it.

        Test scenario:
            This string reaches users directly and reaches the STAC writer's
            `raster:bands[].data_type`, where `uint8` is the value the spec
            asks for and `byte` was not.
        """
        assert _raster(numpy_dtype).dtype == [reported]

    def test_it_agrees_with_the_shared_gdal_to_numpy_map(self):
        """One map behind both, so the property cannot drift from it.

        Test scenario:
            `Dataset.dtype` and `gdal_to_numpy_dtype` are the same lookup seen
            from two places; a raster's reported dtype has to match what the
            helper says its GDAL code means.
        """
        dataset = _raster(np.uint8)

        code = dataset.raster.GetRasterBand(1).DataType

        assert dataset.dtype[0] == gdal_to_numpy_dtype(code)
        assert code == gdal.GDT_Byte


class TestTheUnsignedNoDataDefault:
    """`None` on an unsigned band means "use the dtype max"."""

    @pytest.mark.parametrize(
        ("numpy_dtype", "expected"),
        [
            (np.uint8, 255),
            (np.uint16, 65535),
            (np.uint32, 4294967295),
        ],
    )
    def test_an_unsigned_band_gets_its_dtype_max(self, numpy_dtype, expected):
        """Byte is the case the string test missed.

        Args:
            numpy_dtype: An unsigned band dtype.
            expected: The sentinel that dtype's maximum gives.

        Test scenario:
            `None` cannot be stored in an unsigned band, so the intent has
            always been to substitute the dtype's maximum. Deciding that from
            the display string meant `uint8` -- reported as `byte` -- fell
            through to the signed branch and kept `None`.
        """
        dataset = _raster(numpy_dtype)

        assert dataset.bands._coerce_band_no_data(0, None) == expected

    @pytest.mark.parametrize(
        "numpy_dtype", [np.int16, np.int32, np.float32, np.float64]
    )
    def test_a_signed_or_float_band_passes_none_through(self, numpy_dtype):
        """These can represent the absence, so nothing is substituted.

        Args:
            numpy_dtype: A signed or floating band dtype.

        Test scenario:
            Only the unsigned branch substitutes. Widening the test to "is it
            an integer" would start stamping a sentinel on signed bands too.
        """
        dataset = _raster(numpy_dtype)

        assert dataset.bands._coerce_band_no_data(0, None) is None

    def test_it_agrees_with_the_overflow_fallback(self):
        """Two places answer "is this unsigned"; they must not disagree.

        Test scenario:
            `_fallback_no_data` picks a sentinel when the requested value
            overflows and has always asked the dtype. If the two disagreed, a
            Byte band would get 255 down one path and `None` down the other.
        """
        # The dataset is bound to a name: the engine reaches its parent through
        # a weakref proxy, so a temporary would be collected mid-test.
        dataset = _raster(np.uint8)
        bands = dataset.bands

        assert bands._coerce_band_no_data(0, None) == bands._fallback_no_data(0)
