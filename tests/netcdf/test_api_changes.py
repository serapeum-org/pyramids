"""Regression tests for the Wave A/B public-API changes (API-3/4/6/8/9/10/11).

These pin the deprecation-alias, validation, and immutability contracts so a future change
cannot silently revert them. They are unit-level — the ``subset() -> NetCDF`` return-type
change (API-2) is already covered by ``test_consistent_return_types`` and the static-plot
``x_dim``/``y_dim`` wiring (API-5) by the plot tests.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from pyramids.netcdf import ColorOpts, ColourOpts, LabeledArray, NetCDF
from pyramids.netcdf._kerchunk_facade import _normalize_backend
from pyramids.netcdf.labeled import _LabeledArray, _is_zarr_store
from pyramids.netcdf.models import CFInfo


@pytest.fixture(scope="module")
def small_nc():
    """A tiny single-variable in-memory NetCDF container.

    Returns:
        NetCDF: A 2x4x4 container with one ``t`` variable (a time axis of length 2).
    """
    arr = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    return NetCDF.create_from_array(
        arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326, variable_name="t"
    )


class TestColorOptsAlias:
    """API-4: ColorOpts canonical, ColourOpts deprecated alias."""

    def test_color_opts_is_canonical(self):
        """``ColorOpts`` constructs without any warning and stores its fields.

        Test scenario:
            ``ColorOpts(cmap=...)`` is the canonical class — no DeprecationWarning.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            opts = ColorOpts(cmap="viridis", robust=True)
        assert opts.cmap == "viridis", f"cmap not stored: {opts.cmap}"
        assert opts.robust is True, "robust flag not stored"

    def test_colour_opts_warns_and_is_subclass(self):
        """``ColourOpts`` warns on construction and remains a ``ColorOpts`` subclass.

        Test scenario:
            Instantiating the British-spelling alias emits a DeprecationWarning, yields a
            working ``ColorOpts`` instance (isinstance holds), and keeps its fields.
        """
        with pytest.warns(DeprecationWarning, match="ColourOpts is deprecated"):
            opts = ColourOpts(cmap="magma")
        assert isinstance(opts, ColorOpts), "ColourOpts must subclass ColorOpts"
        assert opts.cmap == "magma", f"cmap not stored on alias: {opts.cmap}"

    def test_colour_opts_compares_equal_to_colour_opts_by_value(self):
        """``ColourOpts`` keeps value-equality (both directions) with an equal ``ColorOpts`` (M3).

        Test scenario:
            Before the alias became a subclass it *was* ``ColorOpts``, so equal fields
            compared equal. The dataclass ``__eq__`` enforces an exact class match, which would
            silently break ``ColourOpts(cmap="x") == ColorOpts(cmap="x")``. The alias overrides
            ``__eq__``/``__hash__`` to compare by field value, so equality holds in both
            directions, hashes match, and they dedupe in a set.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colour = ColourOpts(cmap="viridis", robust=True)
        color = ColorOpts(cmap="viridis", robust=True)
        assert colour == color, "ColourOpts should equal an identical ColorOpts"
        assert color == colour, "equality must be symmetric"
        assert hash(colour) == hash(color), "equal options must hash equally"
        assert len({colour, color}) == 1, "equal options must dedupe in a set"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            other = ColourOpts(cmap="magma")
        assert colour != other, "different fields must not compare equal"


class TestVariableNamesDeprecation:
    """API-3: variable_names property canonical, get_variable_names() deprecated."""

    def test_property_does_not_warn(self, small_nc):
        """The ``variable_names`` property returns the names without warning.

        Test scenario:
            Reading the property under ``simplefilter('error')`` must not raise.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            names = small_nc.variable_names
        assert "t" in names, f"expected 't' in {names}"

    def test_method_warns(self, small_nc):
        """``get_variable_names()`` warns but returns the same names.

        Test scenario:
            The deprecated method emits a DeprecationWarning and matches the property.
        """
        with pytest.warns(DeprecationWarning, match="get_variable_names"):
            names = small_nc.get_variable_names()
        assert names == small_nc.variable_names, "deprecated method must match property"


class TestDimensionModelRename:
    """API-6: ClassicDim* canonical names, old names deprecated aliases."""

    def test_new_names_import_clean(self):
        """The renamed classes import without warning and are usable.

        Test scenario:
            ``ClassicDimensionInfo`` / ``ClassicDimMetadata`` import under
            ``simplefilter('error')`` and the dim class constructs.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from pyramids.netcdf.dimensions import (
                ClassicDimensionInfo,
                ClassicDimMetadata,
            )
        dim = ClassicDimensionInfo(name="time", size=2, values=[0, 1])
        assert dim.name == "time", f"name not stored: {dim.name}"
        assert ClassicDimMetadata is not None, "ClassicDimMetadata should import"

    @pytest.mark.parametrize("old_name", ["DimMetaData", "MetaData"])
    def test_old_names_warn(self, old_name):
        """Accessing a deprecated dimension-model name warns and resolves correctly.

        Args:
            old_name: The deprecated attribute name on the dimensions module.

        Test scenario:
            ``getattr(dimensions, old_name)`` emits a DeprecationWarning and returns the
            renamed class.
        """
        import pyramids.netcdf.dimensions as dims

        with pytest.warns(DeprecationWarning, match=old_name):
            obj = getattr(dims, old_name)
        assert obj is not None, f"{old_name} alias should resolve to a class"

    def test_unknown_name_raises(self):
        """An unknown module attribute still raises ``AttributeError`` (PEP 562).

        Test scenario:
            ``dimensions.NoSuchThing`` raises AttributeError, not a warning.
        """
        import pyramids.netcdf.dimensions as dims

        with pytest.raises(AttributeError, match="NoSuchThing"):
            _ = dims.NoSuchThing


class TestNormalizeBackend:
    """API-10: kerchunk backend canonical 'legacy', 'kerchunk' deprecated alias."""

    @pytest.mark.parametrize("backend", ["native", "legacy"])
    def test_canonical_backends_pass_through(self, backend):
        """``native`` and ``legacy`` pass through unchanged and without warning.

        Args:
            backend: A canonical backend value.

        Test scenario:
            ``_normalize_backend`` returns the value verbatim and emits no warning.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _normalize_backend(backend) == backend, "canonical value changed"

    def test_kerchunk_alias_warns_and_maps_to_legacy(self):
        """``'kerchunk'`` warns and normalises to ``'legacy'``.

        Test scenario:
            The deprecated alias maps to legacy with a DeprecationWarning.
        """
        with pytest.warns(DeprecationWarning, match="backend='kerchunk'"):
            assert _normalize_backend("kerchunk") == "legacy", "alias must map to legacy"

    def test_unknown_backend_raises(self):
        """An unrecognised backend raises ``ValueError``.

        Test scenario:
            ``_normalize_backend('bogus')`` raises with a message naming the valid values.
        """
        with pytest.raises(ValueError, match="must be 'native' or 'legacy'"):
            _normalize_backend("bogus")


class TestIsZarrStoreEngineValidation:
    """API-11: LabeledDataset engine validation — raise on unknown."""

    @pytest.mark.parametrize(
        "engine, expected",
        [("zarr", True), ("netcdf", False), ("netcdf4", False), ("h5netcdf", False)],
    )
    def test_valid_engines(self, engine, expected):
        """Recognised engines classify correctly without raising.

        Args:
            engine: A recognised engine name.
            expected: Whether it selects the Zarr path.

        Test scenario:
            ``zarr`` -> True; the NetCDF/HDF5 family -> False.
        """
        path = "store.zarr" if expected else "store.nc"
        assert _is_zarr_store(path, engine) is expected, f"{engine} misclassified"

    def test_unknown_engine_raises(self):
        """A typo'd engine raises ``ValueError`` instead of silently opening as NetCDF.

        Test scenario:
            ``engine='netcfd'`` (typo) raises with a message listing valid engines.
        """
        with pytest.raises(ValueError, match="engine must be one of"):
            _is_zarr_store("store.nc", "netcfd")


class TestCFInfoFrozen:
    """API-8: CFInfo is an immutable (frozen) dataclass."""

    def test_mutation_raises(self):
        """Reassigning a CFInfo field raises ``FrozenInstanceError``.

        Test scenario:
            A constructed CFInfo rejects attribute assignment.
        """
        info = CFInfo(
            cf_version="1.8",
            conventions={"CF": "1.8"},
            classifications={},
            grid_mappings={},
            bounds_map={},
            data_variable_names=[],
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.cf_version = "1.9"


class TestDimensionModelConsolidation:
    """API-7: DimensionInfo is canonical, with a from_classic_metadata bridge."""

    def test_from_classic_metadata_maps_fields(self):
        """``DimensionInfo.from_classic_metadata`` carries name / size / attrs over.

        Test scenario:
            A ``ClassicDimensionInfo`` converts to the canonical ``DimensionInfo`` with a
            ``/<name>`` full name, ``None`` MDIM type/direction/indexing-variable, and the
            parsed attrs preserved.
        """
        from pyramids.netcdf.dimensions import ClassicDimensionInfo
        from pyramids.netcdf.models import DimensionInfo

        classic = ClassicDimensionInfo(
            name="time", size=2, values=[0, 31], attrs={"axis": "T"}
        )
        dim = DimensionInfo.from_classic_metadata(classic)
        assert (dim.name, dim.size, dim.full_name) == ("time", 2, "/time"), (
            f"unexpected canonical fields: {dim}"
        )
        assert dim.type is None and dim.indexing_variable is None, "MDIM fields should be None"
        assert dim.attrs["axis"] == "T", "classic attrs must carry over"

    def test_from_classic_metadata_coerces_missing_size_to_int_zero(self):
        """A classic dimension with no size yields ``size == 0`` (int), not None (review L5).

        Test scenario:
            ``ClassicDimensionInfo.size`` is legitimately ``None`` when the classic parser
            finds no ``DEF`` size, but ``DimensionInfo.size`` is typed ``int``. The bridge
            must coerce the missing size to ``0`` so the canonical model never violates its
            own annotation.
        """
        from pyramids.netcdf.dimensions import ClassicDimensionInfo
        from pyramids.netcdf.models import DimensionInfo

        classic = ClassicDimensionInfo(name="time", size=None, values=[], attrs={})
        dim = DimensionInfo.from_classic_metadata(classic)
        assert dim.size == 0, f"missing size should coerce to 0, got {dim.size!r}"
        assert isinstance(dim.size, int), f"size must be int, got {type(dim.size).__name__}"

    def test_to_dimension_info_round_trips_through_bridge(self):
        """``ClassicDimensionInfo.to_dimension_info`` yields the canonical model.

        Test scenario:
            The inverse convenience produces a ``DimensionInfo`` equal to calling the
            factory directly.
        """
        from pyramids.netcdf.dimensions import ClassicDimensionInfo
        from pyramids.netcdf.models import DimensionInfo

        classic = ClassicDimensionInfo(name="lat", size=4, attrs={"units": "degrees_north"})
        dim = classic.to_dimension_info()
        assert isinstance(dim, DimensionInfo), "bridge must return the canonical model"
        assert dim.name == "lat" and dim.size == 4, f"fields not carried: {dim}"
        assert dim.attrs["units"] == "degrees_north", "attrs must carry over"


class TestLabeledArrayPublic:
    """API-9: LabeledArray is public; _LabeledArray is a back-compat alias."""

    def test_public_name_and_alias_identity(self):
        """``LabeledArray`` is exported and ``_LabeledArray`` is the same class.

        Test scenario:
            The public and underscore names refer to one class, and instances expose
            ``values`` / ``dims`` / ``shape``.
        """
        assert _LabeledArray is LabeledArray, "underscore alias must be the same class"
        arr = LabeledArray(values=np.zeros(3), dims=("x",), shape=(3,))
        assert arr.dims == ("x",), f"dims not stored: {arr.dims}"
        assert arr.shape == (3,), f"shape not stored: {arr.shape}"
