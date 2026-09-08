"""What `Dataset.dtype` reports, and why nothing may branch on that string.

`dtype` used to read the driver catalog's own `name` column; it now reads the
numpy type's name, so a few strings changed -- `byte` became `uint8`, and the
complex codes collapsed onto numpy's two widths. The new strings are the
correct ones (`uint8` is what the STAC `raster:bands.data_type` field wants),
but the surface is public, so it is pinned here rather than left implicit.

One consumer used to branch on it. `_coerce_band_no_data` decided "is this band
unsigned" with `dtype[i].startswith("u")`, which `"byte"` failed, and answered
`None` with the dtype maximum for the ones that passed. Neither half survives:
the string test was replaced by an honest dtype test, and then the substitution
itself was removed. A `None` no-data now passes through on every dtype, because
it is the caller saying this band declares no sentinel, and 65535 is as real a
value in a `uint16` DEM as 255 is white in 8-bit imagery.

The rest of the module pins what a caller reads back from that: an unstorable
sentinel stays unstorable, `change_no_data_value` refuses rather than inventing
one, and `_fallback_no_data` -- which repairs a sentinel the caller *asked* for
-- keeps substituting, so the two are pinned as deliberately disagreeing.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._domain import DEFAULT_NO_DATA_VALUE
from pyramids.base._errors import NoDataValueError
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


class TestTheUnsetSentinelPassesThrough:
    """`None` means "this band declares no sentinel", on every dtype."""

    @pytest.mark.parametrize(
        "numpy_dtype",
        [np.uint8, np.uint16, np.uint32, np.int16, np.int32, np.float32, np.float64],
    )
    def test_no_dtype_fabricates_a_sentinel(self, numpy_dtype):
        """The one rule, checked across the width of the type table.

        Args:
            numpy_dtype: A band dtype, unsigned / signed / floating.

        Test scenario:
            The unsigned types wider than a byte used to answer `None` with
            their own maximum. That put every genuinely-65535 cell of a
            `uint16` raster out of domain, which is the same defect Byte was
            excluded for. Parametrised across all three families so a future
            "just the unsigned ones" special case fails here.
        """
        dataset = _raster(numpy_dtype)

        assert dataset.bands._coerce_band_no_data(0, None) is None

    @pytest.mark.parametrize(
        ("numpy_dtype", "fallback"),
        [(np.uint8, 255), (np.uint16, 65535), (np.uint32, 4294967295)],
    )
    def test_it_deliberately_disagrees_with_the_overflow_fallback(
        self, numpy_dtype, fallback
    ):
        """Two paths, two questions; pinning that they answer differently.

        Args:
            numpy_dtype: An unsigned band dtype.
            fallback: The sentinel the overflow repair picks for it.

        Test scenario:
            `_fallback_no_data` fires when the caller asked for a sentinel the
            band cannot hold, so substituting a storable one honours the
            request. `_coerce_band_no_data(None)` fires when they asked for
            none, where substituting answers a question nobody posed. The
            disagreement used to hold for Byte alone; it now holds for every
            unsigned width, and pinning it stops a tidy-up from making the two
            agree again.
        """
        dataset = _raster(numpy_dtype)
        bands = dataset.bands

        assert bands._coerce_band_no_data(0, None) is None
        assert bands._fallback_no_data(0) == fallback

    def test_a_storable_sentinel_is_still_cast_to_the_band_dtype(self):
        """Removing the substitution did not remove the coercion.

        Test scenario:
            The pass-through is only for `None` / `NaN`. A real value still
            arrives as a numpy scalar of the band's own dtype, which is what
            keeps the comparison against the band's values exact.
        """
        dataset = _raster(np.uint16)

        coerced = dataset.bands._coerce_band_no_data(0, 4000)

        assert coerced == 4000
        assert np.dtype(type(coerced)) == np.uint16


class TestTheSentinelOnThePublicProperty:
    """What a caller reads back from a band that cannot store its sentinel.

    `Dataset.create(dtype="uint16", no_data_value=np.nan)` used to report
    `(65535,)`: `NaN` is unstorable there, so the maximum stood in. It now
    reports `(nan,)`, the same as a Byte or a signed band always did. The
    consequence a downstream actually feels is the domain -- with 65535
    declared, a band holding 65535 had those cells masked; with nothing
    storable declared, they are data.
    """

    @staticmethod
    def _unsigned_asking_for_nan(dtype: str) -> Dataset:
        """A raster whose requested `NaN` sentinel the band cannot store.

        Args:
            dtype: An unsigned band dtype name.

        Returns:
            Dataset: A 4x4 raster created with `no_data_value=np.nan`.
        """
        return Dataset.create(
            rows=4,
            columns=4,
            bands=1,
            dtype=dtype,
            no_data_value=np.nan,
            geo_ref=GEO,
        )

    @pytest.mark.parametrize("dtype", ["uint16", "uint32"])
    def test_the_unstorable_sentinel_is_reported_as_it_was_asked_for(self, dtype: str):
        """No maximum is invented in its place.

        Args:
            dtype: An unsigned band dtype name wider than a byte.

        Test scenario:
            Reporting the dtype maximum was a silent answer to a question the
            caller did not ask, and it is the value their data is likeliest to
            hold at saturation.
        """
        (sentinel,) = self._unsigned_asking_for_nan(dtype).no_data_value

        assert np.isnan(sentinel), f"expected NaN to pass through, got {sentinel!r}"

    def test_the_dtype_maximum_stays_in_the_domain(self):
        """The defect the substitution caused, stated as a cell count.

        Test scenario:
            A `uint16` band holding 65535 in half its cells has all of them as
            data. Under the substitution those cells matched the fabricated
            sentinel and `count_domain_cells` reported half as many, so a mean
            or a histogram silently skipped the saturated end of the raster.
        """
        values = np.full((4, 4), 65535, dtype=np.uint16)
        values[:2] = 1
        dataset = Dataset.from_array(values, geo_ref=GEO, no_data_value=np.nan)

        assert dataset.count_domain_cells() == 16


class TestChangeNoDataValueToAnUnstorableSentinel:
    """Every integer band refuses; none of them invents a sentinel.

    Byte and the signed types already refused. The wider unsigned types
    succeeded and left their maximum behind -- 255 is white in 8-bit imagery
    and 65535 is the saturated end of a `uint16` product, so the argument
    against fabricating one is the same, only less often noticed. `align` then
    hands the raster to `gdal.ReprojectImage`, which rewrites the real maxima
    one lower to keep them distinguishable from the sentinel.
    """

    @pytest.mark.parametrize(
        "numpy_dtype",
        [np.uint8, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64],
    )
    def test_an_integer_band_refuses_an_unstorable_sentinel(self, numpy_dtype):
        """The rule, across the integer types.

        Args:
            numpy_dtype: An integer band dtype.

        Test scenario:
            `None` resolves to NaN, which no integer band can hold. Refusing
            says so; answering with the maximum invents a sentinel the caller
            never asked for, at the value their data most likely uses. The
            64-bit widths are here because they are the ones whose extremes do
            not survive a `float64` round trip, so they are likeliest to
            diverge -- and `docs/migration.md` names `uint64` in its table.
        """
        dataset = _raster(numpy_dtype)

        with pytest.raises(NoDataValueError):
            dataset.change_no_data_value(None)

    def test_a_float_band_still_accepts_it(self):
        """The refusal is about storability, not about `None`.

        Test scenario:
            A floating band can represent the default sentinel, so nothing is
            refused there. Pinning it keeps the refusal from widening into "no
            band may resolve a `None`".
        """
        dataset = _raster(np.float32)

        dataset.change_no_data_value(None)

        assert dataset.no_data_value[0] == DEFAULT_NO_DATA_VALUE


class TestTheOverflowFallbackIsUntouched:
    """The repair path keeps substituting, and must not be swept up.

    It runs when the caller *asked* for a sentinel that overflows the band --
    the default `-9999` on a `uint8` raster is the common case -- where picking
    a storable value is what they wanted. Removing the other substitution left
    this one deliberately in place, so it is pinned separately.
    """

    @pytest.mark.parametrize(
        ("numpy_dtype", "expected"),
        [(np.uint8, 255), (np.uint16, 65535), (np.int8, -128)],
    )
    def test_it_substitutes_a_storable_sentinel(self, numpy_dtype, expected):
        """Args: numpy_dtype: A band dtype too narrow for `-9999`.

        Args:
            numpy_dtype: A band dtype the default sentinel overflows.
            expected: The storable sentinel picked in its place.

        Test scenario:
            Unsigned bands take their maximum, signed ones too narrow for the
            default take their minimum. Unchanged by this branch.
        """
        dataset = _raster(numpy_dtype)

        assert dataset.bands._fallback_no_data(0) == expected

    def test_a_signed_band_that_fits_the_default_keeps_it(self):
        """Test scenario: nothing is substituted when nothing overflows."""
        dataset = _raster(np.int16)

        assert dataset.bands._fallback_no_data(0) == DEFAULT_NO_DATA_VALUE


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
        # Asserted rather than skipped. Every supported build has these -- the
        # conda pin is `gdal >=3.13.3` and the oldest vendored wheel carries
        # 3.12.4, both past RFC 100 (GDAL 3.11) -- so a missing constant means
        # the environment is not one this package supports, which a silent skip
        # would hide behind a green run.
        code = getattr(gdal, code_name, None)

        assert code is not None, (
            f"this GDAL has no {code_name}; the supported floor is 3.11 (RFC 100)"
        )
        assert gdal_to_numpy_dtype(code) == expected, (
            f"{code_name} did not convert to {expected}"
        )

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
