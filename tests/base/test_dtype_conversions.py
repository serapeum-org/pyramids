import numpy as np
import pytest
from osgeo import gdal, gdalconst, ogr

from pyramids.base._errors import DriverNotExistError, OptionalPackageDoesNotExist
from pyramids.base._utils import (
    DTYPE_CONVERSION_DF,
    GDAL_DTYPE,
    NUMPY_DTYPE,
    OGR_DTYPE,
    Catalog,
    _GDAL_TO_NUMPY,
    _GDAL_TO_OGR,
    _NUMPY_TO_GDAL,
    color_name_to_gdal_constant,
    gdal_constant_to_color_name,
    gdal_to_numpy_dtype,
    gdal_to_ogr_dtype,
    numpy_to_gdal_dtype,
    ogr_ds_to_gdal_dataset,
    ogr_to_numpy_dtype,
    require_optional,
)

pytestmark = pytest.mark.core


def test_numpy_to_gdal_dtype(arr: np.ndarray):
    # test with array input
    gdal_type = numpy_to_gdal_dtype(arr)
    assert gdal_type == gdal.GDT_Float32
    # test with  a dtye input
    gdal_type = numpy_to_gdal_dtype(arr.dtype)
    assert gdal_type == gdal.GDT_Float32
    # test with  a dtye input
    gdal_type = numpy_to_gdal_dtype("float32")
    assert gdal_type == gdal.GDT_Float32


def test_gdal_to_numpy_dtype():
    assert gdal_to_numpy_dtype(6) == "float32"
    assert gdal_to_numpy_dtype(7) == "float64"
    assert gdal_to_numpy_dtype(2) == "uint16"
    with pytest.raises(ValueError):
        gdal_to_numpy_dtype(20)


def test_gdal_to_ogr_dtype(test_image: gdal.Dataset, src: gdal.Dataset):
    assert gdal_to_ogr_dtype(test_image) == 0
    assert gdal_to_ogr_dtype(src) == 2


def test_ogr_to_numpy_dtype():
    assert ogr_to_numpy_dtype(0) == np.int32
    with pytest.raises(ValueError):
        ogr_to_numpy_dtype(1)


class TestCatalog:
    def test_create_instance(self):
        catalog = Catalog()
        assert hasattr(catalog, "drivers")

    def test_get_driver(self):
        catalog = Catalog()
        driver = catalog.get_driver("memory")
        assert isinstance(driver, dict)

    def test_get_driver_by_ext(self):
        catalog = Catalog()
        driver = catalog.get_driver_by_extension("nc")
        assert driver.get("GDAL Name") == "netCDF"
        with pytest.raises(DriverNotExistError):
            catalog.get_driver_by_extension("mm")

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
        assert result is None, (
            f"require_cleopatra must return None on success; got {result!r}"
        )

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
        assert result is None, (
            f"require_cleopatra must return None on success; got {result!r}"
        )

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
        assert "viz" in msg.lower() or "cleopatra" in msg.lower(), (
            f"Default error must mention viz/cleopatra; got {msg!r}"
        )

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
        assert "custom-from-test" in str(exc.value), (
            f"Custom msg must appear in the exception; got {exc.value!r}"
        )

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
        assert str(exc.value) == _DEFAULT_CLEOPATRA_MSG, (
            f"Expected default message; got {exc.value!r}"
        )


class TestDtypeLookupTables:
    """ARC-62: the precomputed dicts must reproduce the old DataFrame masks."""

    def test_complex64_keeps_the_first_matching_gdal_code(self):
        """`np.complex64` maps to GDT_CInt16, the first of its three rows.

        Test scenario:
            `NUMPY_DTYPE` lists `np.complex64` against GDT_CInt16,
            GDT_CInt32 and GDT_CFloat32. The old lookup took `.values[0]`,
            so the first row won. A plain dict comprehension would have
            kept the *last* and silently returned GDT_CFloat32 — this
            pins the first-wins invariant the rewrite depends on.
        """
        assert numpy_to_gdal_dtype(np.dtype(np.complex64)) == gdalconst.GDT_CInt16, (
            "first-wins must select GDT_CInt16 for complex64"
        )
        assert numpy_to_gdal_dtype("complex64") == gdalconst.GDT_CInt16, (
            "the string spelling must resolve to the same first-wins entry"
        )

    def test_every_lookup_matches_the_source_dataframe(self):
        """Each dict agrees with a fresh `.values[0]` scan of the table.

        Test scenario:
            The dicts exist purely as a fast path for
            `DTYPE_CONVERSION_DF`, which is still shipped and still used
            for the error messages. Re-deriving each mapping from the
            DataFrame is the direct check that the two never drift.
            `_NUMPY_TO_GDAL` is included specifically because it is the
            one table with duplicate keys, so it is the only one where
            first-wins is more than a formality.
        """
        for column, table in (("numpy", _GDAL_TO_NUMPY), ("ogr", _GDAL_TO_OGR)):
            for gdal_code, expected in table.items():
                matched = DTYPE_CONVERSION_DF.loc[
                    DTYPE_CONVERSION_DF["gdal"] == gdal_code, column
                ]
                assert matched.values[0] == expected, (
                    f"{column} lookup for GDAL code {gdal_code} drifted from the "
                    f"table: dict has {expected}, DataFrame has {matched.values[0]}"
                )
        for np_dtype, expected in _NUMPY_TO_GDAL.items():
            matched = DTYPE_CONVERSION_DF.loc[
                DTYPE_CONVERSION_DF["numpy"] == np_dtype, "gdal"
            ]
            assert matched.values[0] == expected, (
                f"numpy->gdal lookup for {np_dtype} drifted from the table: dict "
                f"has {expected}, DataFrame has {matched.values[0]}"
            )

    def test_only_the_none_bearing_rows_are_dropped(self):
        """`_first_wins` omits exactly the rows with no counterpart.

        Test scenario:
            The dicts are built by dropping `None`-bearing pairs, which
            is what makes the "unsupported dtype" guards fire. Checking
            only the entries the dicts *contain* would never catch a
            row being dropped that should have been kept, so assert the
            key sets against the table directly.
        """
        # Compare against the source lists, not the DataFrame: pandas coerces the
        # mixed int/None `ogr` column to float64, turning every None into NaN, so
        # an `is not None` test there would be vacuously true.
        expected_gdal_to_numpy = {
            gdal_code
            for gdal_code, np_dtype in zip(GDAL_DTYPE, NUMPY_DTYPE)
            if np_dtype is not None
        }
        assert set(_GDAL_TO_NUMPY) == expected_gdal_to_numpy, (
            "GDAL->numpy keys must be exactly the rows carrying a numpy dtype"
        )
        expected_gdal_to_ogr = {
            gdal_code
            for gdal_code, ogr_type in zip(GDAL_DTYPE, OGR_DTYPE)
            if ogr_type is not None
        }
        assert set(_GDAL_TO_OGR) == expected_gdal_to_ogr, (
            "GDAL->OGR keys must be exactly the rows carrying an OGR type"
        )

    def test_unknown_gdal_type_raises_value_error(self):
        """`GDT_Unknown` has no numpy row, so it is reported as unsupported.

        Test scenario:
            The old code indexed an all-`None` match and died with
            `AttributeError: 'NoneType' object has no attribute
            '__name__'`. The dict drops `None`-valued rows so the
            existing "not supported" guard fires instead.
        """
        with pytest.raises(ValueError, match="not supported"):
            gdal_to_numpy_dtype(gdalconst.GDT_Unknown)

    def test_unmapped_numpy_dtype_raises_value_error(self):
        """A numpy dtype with no GDAL counterpart raises, not `IndexError`."""
        with pytest.raises(ValueError, match="not supported"):
            numpy_to_gdal_dtype(np.dtype("float16"))

    def test_complex_band_has_no_ogr_equivalent(self):
        """A complex-typed band raises instead of `int(None)`-ing.

        Test scenario:
            `gdal_to_ogr_dtype` previously did `int(...values[0])` on an
            OGR column whose complex rows are `None`, producing
            `TypeError`. It now reports the missing mapping explicitly.
        """
        src = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_CFloat32)
        with pytest.raises(ValueError, match="no OGR equivalent"):
            gdal_to_ogr_dtype(src)


class TestRequireOptional:
    """ARC-67: the single guard behind the ten `import_*` helpers."""

    def test_returns_none_for_guard_only_callers(self):
        """The guard form returns nothing when the module imports."""
        assert require_optional("numpy", "unused") is None

    def test_returns_the_submodule_for_a_dotted_name(self):
        """A dotted name yields the submodule, not its top-level package.

        Test scenario:
            `__import__("a.b")` returns `a`, so the helper reads
            `sys.modules[name]` back instead. `import_basemap` passes
            `cleopatra.tiles`, and only this path distinguishes the two.
        """
        module = require_optional("numpy.linalg", "unused", return_module=True)
        assert module.__name__ == "numpy.linalg", (
            f"expected the submodule, got {module.__name__}"
        )

    def test_missing_module_raises_the_supplied_hint(self):
        """The install hint is raised verbatim, chained to the ImportError."""
        with pytest.raises(OptionalPackageDoesNotExist, match="install the extra"):
            require_optional("pyramids_not_a_real_module", "install the extra")

    def test_missing_module_chains_the_original_importerror(self):
        """The raised error keeps `__cause__` so the real reason survives."""
        try:
            require_optional("pyramids_not_a_real_module", "install the extra")
        except OptionalPackageDoesNotExist as exc:
            assert isinstance(exc.__cause__, ImportError), (
                f"expected a chained ImportError, got {exc.__cause__!r}"
            )
