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

Building the substituted sentinel as a numpy scalar rather than a Python `int`
is what makes the two paths agree on type as well as value -- and it is visible
on `Dataset.no_data_value`, which is why the last class here pins what a caller
now reads back.
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
            The type is asserted too: both answers flow into
            `Dataset.no_data_value` and out to GDAL, and `255 ==
            np.uint8(255)` is True, so an `==` alone cannot see one path
            returning a Python `int` and the other a numpy scalar.
        """
        # The dataset is bound to a name: the engine reaches its parent through
        # a weakref proxy, so a temporary would be collected mid-test.
        dataset = _raster(np.uint8)
        bands = dataset.bands

        coerced = bands._coerce_band_no_data(0, None)
        fallback = bands._fallback_no_data(0)

        assert coerced == fallback
        assert type(coerced) is type(fallback), (
            f"same value, different types: {type(coerced)} vs {type(fallback)}"
        )
        assert np.dtype(type(coerced)) == np.uint8


class TestTheSentinelOnThePublicProperty:
    """The type change reaches `Dataset.no_data_value`, so it is pinned here.

    `_coerce_band_no_data` builds the substituted maximum as a numpy scalar of
    the band dtype, for type parity with `_fallback_no_data`. GDAL's
    `SetNoDataValue` takes a C double and refuses a numpy `uint8`, so the value
    is retried as `float64` and that is what the property reports: a uint8 band
    that read back `(255,)` -- a Python `int` -- now reads back
    `(np.float64(255.0),)`. Same number, different type, and nothing warns.
    """

    @staticmethod
    def _unsigned_with_substituted_sentinel(dtype: str) -> Dataset:
        """A raster whose no-data was substituted from the dtype maximum.

        Args:
            dtype: An unsigned band dtype name.

        Returns:
            Dataset: A 4x4 raster whose requested `NaN` sentinel was replaced.
        """
        return Dataset.create(
            rows=4,
            columns=4,
            bands=1,
            dtype=dtype,
            no_data_value=np.nan,
            geo_ref=GEO,
        )

    @pytest.mark.parametrize(("dtype", "expected"), [("uint8", 255), ("uint16", 65535)])
    def test_the_value_is_still_the_dtype_maximum(self, dtype: str, expected: int):
        """Only the type changed; the number a caller compares against did not.

        Args:
            dtype: An unsigned band dtype name.
            expected: That dtype's maximum.

        Test scenario:
            Anything comparing with `==` is unaffected, which is why the change
            is silent and why it is worth pinning rather than trusting.
        """
        (sentinel,) = self._unsigned_with_substituted_sentinel(dtype).no_data_value

        assert sentinel == expected

    def test_it_is_a_numpy_scalar_rather_than_a_builtin(self):
        """The part a downstream can trip over.

        Test scenario:
            `json.dumps`, `%d` formatting and `is`-comparisons all behave
            differently for a numpy scalar, and arithmetic on a numpy *integer*
            scalar wraps at the dtype bound instead of promoting. Pinned so a
            future change back to a builtin is a deliberate one.
        """
        (sentinel,) = self._unsigned_with_substituted_sentinel("uint8").no_data_value

        assert isinstance(sentinel, np.generic), (
            f"expected a numpy scalar, got {type(sentinel)}"
        )
        assert not isinstance(sentinel, int), (
            f"expected the builtin int to be gone, got {type(sentinel)}"
        )


class TestHalfPrecisionRasters:
    """GDAL 3.13 added `Float16` / `CFloat16`, and the bundled build makes them.

    The conversion table stopped at `Int8`, so a raster in either type read and
    computed fine but `dataset.dtype` -- and therefore `print(dataset)` --
    raised. Centralising the map on one lookup is what made adding the two rows
    a one-line change.
    """

    @pytest.mark.parametrize(
        ("code_name", "expected"),
        [("GDT_Float16", "float16"), ("GDT_CFloat16", "complex64")],
    )
    def test_the_half_precision_codes_convert(self, code_name: str, expected: str):
        """Both new codes resolve through the shared map.

        Args:
            code_name: The `gdal.GDT_*` attribute to look up.
            expected: The numpy dtype name it must convert to.
        """
        code = getattr(gdal, code_name, None)
        if code is None:
            pytest.skip("this GDAL predates the half-precision types")

        assert gdal_to_numpy_dtype(code) == expected

    def test_a_float16_raster_can_be_printed(self, tmp_path):
        """The reachable symptom: `__str__` reads `dtype`.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            The raster reads and computes either way; what failed was asking it
            what type it is, which `print()` does.
        """
        if not hasattr(gdal, "GDT_Float16"):
            pytest.skip("this GDAL predates the half-precision types")
        path = tmp_path / "half.tif"
        raster = gdal.GetDriverByName("GTiff").Create(
            str(path), 4, 4, 1, gdal.GDT_Float16
        )
        raster.SetGeoTransform((0.0, 1.0, 0.0, 4.0, 0.0, -1.0))
        raster = None

        dataset = Dataset.read_file(str(path))

        assert dataset.dtype == ["float16"]
        assert str(dataset)


class TestChangeNoDataValueOnAByteRaster:
    """`None` on a Byte band became the dtype maximum rather than a refusal.

    Deciding "is this band unsigned" from the dtype string missed `byte`, so a
    Byte raster took the signed branch and `change_no_data_value(None)` raised.
    Every other unsigned width already substituted its maximum, so this makes
    Byte consistent -- but it is a behaviour change at a public entry point and
    is pinned here rather than left to be discovered.
    """

    def test_it_now_stores_the_dtype_maximum(self):
        """Consistent with `uint16` / `uint32`, which always did this.

        Test scenario:
            `None` cannot be stored in an unsigned band. Substituting the
            maximum is what the other widths do; Byte used to refuse instead.
        """
        dataset = _raster(np.uint8)

        dataset.change_no_data_value(None)

        assert list(dataset.no_data_value) == [255]

    @pytest.mark.parametrize(
        ("numpy_dtype", "expected"), [(np.uint16, 65535), (np.uint32, 4294967295)]
    )
    def test_the_wider_unsigned_types_are_unchanged(self, numpy_dtype, expected):
        """Byte was brought into line with these, not the reverse.

        Args:
            numpy_dtype: An unsigned band dtype.
            expected: The maximum it substitutes.
        """
        dataset = _raster(numpy_dtype)

        dataset.change_no_data_value(None)

        assert list(dataset.no_data_value) == [expected]
