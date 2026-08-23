"""`register_dataset_accessor` extension hook (#1034)."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from pyramids.dataset import Dataset, register_dataset_accessor
from pyramids.dataset.dataset import _ACCESSOR_REGISTRY

pytestmark = pytest.mark.core


@pytest.fixture
def plain() -> Dataset:
    """A small in-memory raster."""
    return Dataset.create_from_array(
        np.zeros((2, 3), dtype="float32"),
        top_left_corner=(0, 0),
        cell_size=1.0,
        epsg=4326,
    )


@pytest.fixture
def cleanup_accessors():
    """Remove any accessors registered by a test so class state does not leak."""
    before = set(_ACCESSOR_REGISTRY)
    yield
    for name in set(_ACCESSOR_REGISTRY) - before:
        _ACCESSOR_REGISTRY.pop(name, None)
        if name in vars(Dataset):
            delattr(Dataset, name)


class TestRegisterDatasetAccessor:
    """Registration, laziness, caching, and collisions."""

    def test_register_and_use(self, plain, cleanup_accessors):
        """A registered accessor is reachable and reads through to the Dataset."""

        @register_dataset_accessor("summary1")
        class Summary:
            def __init__(self, ds):
                self._ds = ds

            def n(self):
                return self._ds.band_count

        assert plain.summary1.n() == 1

    def test_lazy_build(self, plain, cleanup_accessors):
        """The accessor is built on first access, not at registration/open time."""
        built = []

        @register_dataset_accessor("summary2")
        class Summary:
            def __init__(self, ds):
                built.append(1)
                self._ds = ds

        assert built == [], "must not be built before first access"
        _ = plain.summary2
        assert built == [1], "built exactly once on first access"

    def test_cached_identity(self, plain, cleanup_accessors):
        """Repeated access returns the same cached instance."""

        @register_dataset_accessor("summary3")
        class Summary:
            def __init__(self, ds):
                self._ds = ds

        assert plain.summary3 is plain.summary3, "must be cached per Dataset"

    def test_state_persists(self, plain, cleanup_accessors):
        """State set on the accessor persists across accesses."""

        @register_dataset_accessor("summary4")
        class Summary:
            def __init__(self, ds):
                self._ds = ds

        plain.summary4.tag = 42
        assert plain.summary4.tag == 42

    def test_collision_engine_name(self, cleanup_accessors):
        """Registering an engine name raises ValueError."""
        with pytest.raises(ValueError, match="shadows"):

            @register_dataset_accessor("spatial")
            class Bad:
                def __init__(self, ds):
                    self._ds = ds

    def test_collision_netcdf_engine_name(self, cleanup_accessors):
        """A NetCDF engine name is reserved once pyramids.netcdf is imported."""
        import pyramids.netcdf  # noqa: F401  (populates the reserved set)

        with pytest.raises(ValueError, match="shadows"):

            @register_dataset_accessor("interop")
            class Bad:
                def __init__(self, ds):
                    self._ds = ds

    def test_collision_method_name(self, cleanup_accessors):
        """Registering an existing Dataset method/property raises ValueError."""
        with pytest.raises(ValueError, match="shadows"):

            @register_dataset_accessor("crop")
            class Bad:
                def __init__(self, ds):
                    self._ds = ds

    def test_reregister_warns_and_overwrites(self, cleanup_accessors):
        """Re-registering a name warns and installs the newer class."""

        @register_dataset_accessor("summary5")
        class First:
            def __init__(self, ds):
                self._ds = ds

        with pytest.warns(UserWarning, match="overriding"):

            @register_dataset_accessor("summary5")
            class Second:
                def __init__(self, ds):
                    self._ds = ds

        assert _ACCESSOR_REGISTRY["summary5"] is Second

    def test_obj_none_introspection(self, cleanup_accessors):
        """Class access returns the accessor class (so hasattr is safe)."""

        @register_dataset_accessor("summary6")
        class Summary:
            def __init__(self, ds):
                self._ds = ds

        assert Dataset.summary6 is Summary
        assert hasattr(Dataset, "summary6")


class TestAccessorLifecycle:
    """Invalidation, NetCDF inheritance, and cycle-free lifetime."""

    def test_invalidated_on_update_inplace(self, plain, cleanup_accessors):
        """An in-place op that swaps the raster drops the cached accessor."""
        built = []

        @register_dataset_accessor("summary7")
        class Summary:
            def __init__(self, ds):
                built.append(1)
                self._ds = ds

        _ = plain.summary7
        assert "summary7" in plain.__dict__ and built == [1]
        plain.epsg = 3857  # triggers _update_inplace
        assert "summary7" not in plain.__dict__, "cache dropped by _update_inplace"
        built.clear()
        _ = plain.summary7
        assert built == [1], "rebuilt fresh after the swap"
        assert plain.epsg == 3857, "the in-place op still applied"

    def test_netcdf_inherits_accessor(self, cleanup_accessors):
        """An accessor registered on Dataset is reachable on a NetCDF instance."""
        from pyramids.netcdf import NetCDF

        @register_dataset_accessor("summary8")
        class Summary:
            def __init__(self, ds):
                self._ds = ds

            def n(self):
                return self._ds.band_count

        nc = NetCDF.read_file(
            "tests/data/netcdf/none__4v__1d1-2d2-3d1__curv.nc"
        ).get_variable("Tair")
        assert nc.summary8.n() == nc.band_count

    def test_cycle_free_lifetime(self, cleanup_accessors):
        """The weak back-reference keeps the Dataset collectable after use."""

        @register_dataset_accessor("summary9")
        class Summary:
            def __init__(self, ds):
                self._ds = ds

        ds = Dataset.create_from_array(
            np.zeros((2, 2)), top_left_corner=(0, 0), cell_size=1.0, epsg=4326
        )
        _ = ds.summary9
        ref = weakref.ref(ds)
        del ds
        gc.collect()
        assert ref() is None, "the accessor's weak proxy must not leak the Dataset"

    def test_invalidated_on_update_inplace_netcdf(self, cleanup_accessors):
        """The NetCDF `_update_inplace` override also drops the cached accessor."""
        from pyramids.netcdf import NetCDF

        built = []

        @register_dataset_accessor("summary10")
        class Summary:
            def __init__(self, ds):
                built.append(1)
                self._ds = ds

        var = NetCDF.read_file(
            "tests/data/netcdf/none__4v__1d1-2d2-3d1__curv.nc"
        ).get_variable("Tair")
        _ = var.summary10
        assert "summary10" in var.__dict__ and built == [1]
        var._update_inplace(var.raster)  # exercises the NetCDF override branch
        assert "summary10" not in var.__dict__, "NetCDF override must drop the cache"
        built.clear()
        _ = var.summary10
        assert built == [1], "rebuilt fresh after the NetCDF swap"

    def test_pickle_rebuilds_accessor(self, tmp_path, cleanup_accessors):
        """A cached accessor is not pickled and rebuilds lazily after unpickle."""
        import pickle

        @register_dataset_accessor("summary11")
        class Summary:
            def __init__(self, ds):
                self._ds = ds

            def n(self):
                return self._ds.band_count

        path = tmp_path / "p.tif"
        Dataset.create_from_array(
            np.zeros((2, 2), dtype="float32"),
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
        ).to_file(str(path))
        ds = Dataset.read_file(str(path))
        _ = ds.summary11
        restored = pickle.loads(pickle.dumps(ds))
        assert "summary11" not in restored.__dict__, "accessor must not be pickled"
        assert restored.summary11.n() == 1, "rebuilds lazily after unpickle"
