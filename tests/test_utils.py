import numpy as np
import pytest
from osgeo import gdal, ogr

from pyramids.base._errors import DriverNotExistError
from pyramids.base._utils import (
    Catalog,
    color_name_to_gdal_constant,
    gdal_constant_to_color_name,
    gdal_to_numpy_dtype,
    gdal_to_ogr_dtype,
    numpy_to_gdal_dtype,
    ogr_ds_to_gdal_dataset,
    ogr_to_numpy_dtype,
)

pytestmark = pytest.mark.core


def test_numpy_to_gdal_dtype(arr: np.ndarray):
    # test with array input
    gdal_type = numpy_to_gdal_dtype(arr)
    assert gdal_type is gdal.GDT_Float32
    # test with  a dtye input
    gdal_type = numpy_to_gdal_dtype(arr.dtype)
    assert gdal_type is gdal.GDT_Float32
    # test with  a dtye input
    gdal_type = numpy_to_gdal_dtype("float32")
    assert gdal_type is gdal.GDT_Float32


def test_gdal_to_numpy_dtype():
    assert gdal_to_numpy_dtype(6) == "float32"
    assert gdal_to_numpy_dtype(7) == "float64"
    assert gdal_to_numpy_dtype(2) == "uint16"
    try:
        gdal_to_numpy_dtype(20)
    except ValueError:
        pass


def test_gdal_to_ogr_dtype(test_image: gdal.Dataset, src: gdal.Dataset):
    assert gdal_to_ogr_dtype(test_image) == 0
    assert gdal_to_ogr_dtype(src) == 2


def test_ogr_to_numpy_dtype():
    assert ogr_to_numpy_dtype(0) == np.int32
    try:
        ogr_to_numpy_dtype(1)
    except ValueError:
        pass


class TestCatalog:
    def test_create_instance(self):
        catalog = Catalog()
        assert hasattr(catalog, "catalog")

    def test_get_driver(self):
        catalog = Catalog()
        driver = catalog.get_driver("memory")
        assert isinstance(driver, dict)

    def test_get_driver_by_ext(self):
        catalog = Catalog()
        driver = catalog.get_driver_by_extension("nc")
        assert driver.get("GDAL Name") == "netCDF"
        try:
            catalog.get_driver_by_extension("mm")
        except DriverNotExistError:
            pass

    def test_get_gdal_name(self):
        catalog = Catalog()
        name = catalog.get_gdal_name("memory")
        assert name == "MEM"

    def test_exists(self):
        catalog = Catalog()
        assert catalog.exists("memory")
        assert not catalog.exists("MEM")

    def test_get_extension(self):
        catalog = Catalog()
        ext = catalog.get_extension("geotiff")
        assert ext == "tif"

    def test_get_driver_name(self):
        catalog = Catalog()
        name = catalog.get_driver_name("AAIGrid")
        assert name == "ascii"

    def test_cog_entry_is_creation_capable(self):
        """Task 10 — gdal_drivers.yaml COG entry was corrected."""
        catalog = Catalog()
        # YAML parses `yes` as boolean True.
        assert catalog.get_driver("cog")["Creation"] is True

    def test_cog_entry_supports_georef(self):
        catalog = Catalog()
        assert catalog.get_driver("cog")["Geo-referencing"] is True

    def test_cog_entry_has_no_extension(self):
        """COG must not claim .tif; GTiff owns that extension."""
        catalog = Catalog()
        assert catalog.get_extension("cog") is None

    def test_tif_extension_still_resolves_to_geotiff(self):
        """Regression guardrail for the COG-vs-GTiff disambiguation rule."""
        catalog = Catalog()
        assert catalog.get_driver_name_by_extension("tif") == "geotiff"


def test_ogr_ds_togdal_dataset(data_source: ogr.DataSource):
    gdal_ds = ogr_ds_to_gdal_dataset(data_source)
    assert isinstance(gdal_ds, gdal.Dataset)


def test_color_name_to_gdal_constant():
    assert color_name_to_gdal_constant("red") == 3
    assert color_name_to_gdal_constant("green") == 4
    assert color_name_to_gdal_constant("blue") == 5
    with pytest.raises(ValueError):
        color_name_to_gdal_constant("fff")


def test_gdal_constant_to_color_name():
    assert gdal_constant_to_color_name(3) == "red"
    assert gdal_constant_to_color_name(4) == "green"
    assert gdal_constant_to_color_name(5) == "blue"
    with pytest.raises(ValueError):
        gdal_constant_to_color_name(17)


class TestNumpyToGdalDtypeInvalidInput:
    """Tests for numpy_to_gdal_dtype with invalid input types."""

    def test_invalid_input_raises_value_error(self):
        """Passing a non-array, non-dtype, non-string raises ValueError."""
        with pytest.raises(
            ValueError,
            match="not a numpy array",
        ):
            numpy_to_gdal_dtype(12345)

    def test_invalid_list_raises_value_error(self):
        """Passing a list instead of an array raises ValueError."""
        with pytest.raises(ValueError, match="not a numpy array"):
            numpy_to_gdal_dtype([1, 2, 3])


class TestOgrToNumpyDtypeCoverage:
    """Tests for ogr_to_numpy_dtype covering codes 12 and generic matched branch."""

    def test_code_12_returns_int64(self):
        """OGR code 12 (OFTInteger64) should map to np.int64."""
        result = ogr_to_numpy_dtype(12)
        assert result == np.int64, f"Expected np.int64 for OGR code 12, got {result}"

    def test_code_2_returns_float64(self):
        """OGR code 2 (OFTReal) should map to np.float64."""
        result = ogr_to_numpy_dtype(2)
        assert result == np.float64, f"Expected np.float64 for OGR code 2, got {result}"

    def test_unsupported_code_raises_value_error(self):
        """An OGR code with no matching numpy dtype should raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            ogr_to_numpy_dtype(99)


class TestImportCleopatra:
    """Tests for import_cleopatra utility function."""

    def test_import_cleopatra_raises_when_missing(self, monkeypatch):
        """If cleopatra import fails, OptionalPackageDoesNotExist is raised."""
        import builtins

        from pyramids.base._errors import OptionalPackageDoesNotExist

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            """Block cleopatra from being imported."""
            if name == "cleopatra":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from pyramids.base._utils import import_cleopatra

        with pytest.raises(OptionalPackageDoesNotExist):
            import_cleopatra("cleopatra is required")
