"""A reported dtype must name one GDAL type, not four.

`Dataset.dtype` moved from the catalog's `name` column to the numpy name. That
fixed something real -- `byte` is not a numpy name, and `uint8` is what STAC's
`raster:bands[].data_type` wants -- but it flattened a divergence the `name`
column existed to hold.

numpy has no complex type whose components are integers or half-floats, so
`CInt16`, `CInt32`, `CFloat16` and `CFloat32` all map onto `complex64`.
Reporting that name loses which of them a band is, and feeding it back through
`numpy_to_gdal_dtype` -- which accepts a string -- resolves to whichever row
comes first, so a `CFloat32` raster round-tripped to `CInt16` where it used to
raise.
"""

from __future__ import annotations

import pytest
from osgeo import gdal

from pyramids.base._utils import (
    _AMBIGUOUS_NUMPY_NAMES,
    gdal_dtype_name,
    numpy_to_gdal_dtype,
)

pytestmark = pytest.mark.core


class TestAmbiguityIsDerivedFromTheCatalog:
    """Listing the ambiguous names by hand is what lets a new row slip in."""

    def test_complex64_is_ambiguous(self):
        """Test scenario: four GDAL types share it, so it identifies none of them."""
        assert "complex64" in _AMBIGUOUS_NUMPY_NAMES

    @pytest.mark.parametrize("name", ["uint8", "int16", "float32", "complex128"])
    def test_the_unshared_names_are_not(self, name: str):
        """Args: name: A numpy name exactly one GDAL type maps onto.

        Test scenario:
            Over-broad ambiguity would push ordinary types back to the
            catalog's spelling and undo the `byte` -> `uint8` change this
            branch made deliberately.
        """
        assert name not in _AMBIGUOUS_NUMPY_NAMES


class TestTheReportedName:
    """numpy's where it is faithful, the catalog's where it is not."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (gdal.GDT_Byte, "uint8"),
            (gdal.GDT_Int16, "int16"),
            (gdal.GDT_UInt16, "uint16"),
            (gdal.GDT_Float32, "float32"),
            (gdal.GDT_Float64, "float64"),
            (gdal.GDT_CFloat64, "complex128"),
        ],
    )
    def test_an_unambiguous_type_reports_its_numpy_name(self, code: int, expected: str):
        """Args: code: A GDAL type. expected: Its numpy name.

        Test scenario:
            `uint8` is the case this branch changed on purpose, and the STAC
            field depends on it. It must not regress to `byte`.
        """
        assert gdal_dtype_name(code) == expected

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (gdal.GDT_CInt16, "complex-int16"),
            (gdal.GDT_CInt32, "complex-int32"),
            (gdal.GDT_CFloat32, "complex-float32"),
        ],
    )
    def test_an_ambiguous_type_keeps_the_catalog_name(self, code: int, expected: str):
        """The regression: three types answered `complex64` alike.

        Args:
            code: A GDAL complex type numpy cannot name uniquely.
            expected: The catalog spelling that identifies it.

        Test scenario:
            `Dataset.dtype` could not tell a `CInt16` raster from a `CFloat32`
            one, and neither could the STAC `data_type` it feeds.
        """
        assert gdal_dtype_name(code) == expected

    def test_the_four_complex_types_stay_distinguishable(self):
        """The property, rather than four separate equalities.

        Test scenario:
            Whatever the spellings are, they must differ -- that is the whole
            point. A future change that renamed them would still have to keep
            them apart.
        """
        names = {
            gdal_dtype_name(code)
            for code in (gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32)
        }

        assert len(names) == 3, f"types collapsed onto {names}"

    def test_an_unknown_code_is_refused(self):
        """Test scenario: a code outside the catalog has no name to report."""
        with pytest.raises(ValueError, match="conversion catalog"):
            gdal_dtype_name(9999)


class TestTheRoundTrip:
    """Feeding the reported name back must not resolve to a different type."""

    @pytest.mark.parametrize(
        "code",
        [gdal.GDT_Byte, gdal.GDT_Int16, gdal.GDT_Float32, gdal.GDT_CFloat64],
    )
    def test_an_unambiguous_name_round_trips(self, code: int):
        """Args: code: A GDAL type whose numpy name is its own.

        Test scenario:
            Handing `Dataset.dtype` back to `numpy_to_gdal_dtype` is the
            obvious thing to do with it, and for these it must give the type
            you started from.
        """
        assert numpy_to_gdal_dtype(gdal_dtype_name(code)) == code

    @pytest.mark.parametrize(
        "code", [gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32]
    )
    def test_an_ambiguous_type_refuses_rather_than_resolving_wrongly(self, code: int):
        """The regression's second half, and the worse one.

        Args:
            code: A GDAL complex type numpy cannot name uniquely.

        Test scenario:
            Reporting `complex64` made the round trip *succeed* and return
            `CInt16` for a `CFloat32` raster -- a wrong answer where the older
            spelling raised. Refusing is the honest outcome: the name does not
            identify a numpy type, so there is nothing to convert.
        """
        with pytest.raises(TypeError):
            numpy_to_gdal_dtype(gdal_dtype_name(code))
