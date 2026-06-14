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


class TestRequireCleopatra:
    """Tests for the D-5 ``require_cleopatra`` consolidation helper."""

    def test_default_message_returns_none_when_installed(self):
        """``require_cleopatra()`` returns nothing when cleopatra is available.

        Test scenario:
            The shared D-5 guard is a thin wrapper around
            `import_cleopatra` with a default message pointing at the
            `[viz]` extra. When cleopatra is installed (as in the test
            env) the call returns `None` silently. Auto-skips when the
            `[viz]` extra is not installed (bare-wheel CI job, etc.).
        """
        pytest.importorskip("cleopatra", reason="cleopatra not installed (viz extra)")
        from pyramids.base._utils import require_cleopatra

        result = require_cleopatra()
        assert (
            result is None
        ), f"require_cleopatra must return None on success; got {result!r}"

    def test_custom_message_returns_none_when_installed(self):
        """An explicit ``msg`` does not change the success path.

        Test scenario:
            A caller-supplied override message only matters on the
            failure path. With cleopatra installed, `require_cleopatra`
            must still return `None` regardless of the message
            argument. Auto-skips when the `[viz]` extra is not
            installed.
        """
        pytest.importorskip("cleopatra", reason="cleopatra not installed (viz extra)")
        from pyramids.base._utils import require_cleopatra

        result = require_cleopatra("custom override")
        assert (
            result is None
        ), f"require_cleopatra must return None on success; got {result!r}"

    def test_default_message_mentions_viz_extra_when_missing(self, monkeypatch):
        """Default error message points at the ``[viz]`` install extra.

        Test scenario:
            Block the ``cleopatra`` import so ``require_cleopatra()``
            raises ``OptionalPackageDoesNotExist``. The default message
            from ``_DEFAULT_CLEOPATRA_MSG`` mentions the ``[viz]`` extra
            so the user knows how to install it.
        """
        import builtins

        from pyramids.base._errors import OptionalPackageDoesNotExist

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cleopatra":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from pyramids.base._utils import require_cleopatra

        with pytest.raises(OptionalPackageDoesNotExist) as exc:
            require_cleopatra()
        msg = str(exc.value)
        assert (
            "viz" in msg.lower() or "cleopatra" in msg.lower()
        ), f"Default error must mention viz/cleopatra; got {msg!r}"

    def test_custom_message_passed_through_on_failure(self, monkeypatch):
        """A custom ``msg`` overrides the default error text.

        Test scenario:
            Patch ``cleopatra`` import to fail and call
            ``require_cleopatra(msg="custom-from-test")``. The raised
            exception must contain exactly that string so each caller
            can supply a domain-specific install hint.
        """
        import builtins

        from pyramids.base._errors import OptionalPackageDoesNotExist

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cleopatra":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from pyramids.base._utils import require_cleopatra

        with pytest.raises(OptionalPackageDoesNotExist) as exc:
            require_cleopatra(msg="custom-from-test")
        assert "custom-from-test" in str(
            exc.value
        ), f"Custom msg must appear in the exception; got {exc.value!r}"

    def test_none_msg_uses_default(self, monkeypatch):
        """``require_cleopatra(msg=None)`` is identical to no argument.

        Test scenario:
            ``msg=None`` is the documented default; ``require_cleopatra``
            falls back to ``_DEFAULT_CLEOPATRA_MSG``. Confirm explicit
            ``None`` does not raise a TypeError and falls through to the
            default message on failure.
        """
        import builtins

        from pyramids.base._errors import OptionalPackageDoesNotExist

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cleopatra":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from pyramids.base._utils import _DEFAULT_CLEOPATRA_MSG, require_cleopatra

        with pytest.raises(OptionalPackageDoesNotExist) as exc:
            require_cleopatra(msg=None)
        assert (
            str(exc.value) == _DEFAULT_CLEOPATRA_MSG
        ), f"Expected default message; got {exc.value!r}"
