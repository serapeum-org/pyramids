"""DASK-40b: LazyFeatureCollection class."""

from __future__ import annotations

import pickle

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask_geopandas
from pyramids.base.protocols import LazySpatialObject, SpatialObject, is_lazy
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.parquet_lazy

try:
    import_dask_geopandas("dask-geopandas not installed")
    import dask_geopandas
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK_GP = False
else:
    HAS_DASK_GP = True
requires_dask_geopandas = pytest.mark.skipif(
    not HAS_DASK_GP, reason="dask-geopandas not installed"
)


@pytest.fixture
def small_gdf():
    return gpd.GeoDataFrame(
        {"id": list(range(10)), "class": ["a"] * 5 + ["b"] * 5},
        geometry=[Point(i, i) for i in range(10)],
        crs="EPSG:4326",
    )


@pytest.fixture
def lfc(small_gdf):
    dg = pytest.importorskip("dask_geopandas")
    from pyramids.feature import LazyFeatureCollection

    ddf = dg.from_geopandas(small_gdf, npartitions=2)
    return LazyFeatureCollection.from_dask_gdf(ddf)


@requires_dask_geopandas
class TestLazyFeatureCollection:
    def test_is_dask_gdf_subclass(self, lfc):
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        assert isinstance(lfc, dg.GeoDataFrame)
        assert isinstance(lfc, LazyFeatureCollection)

    def test_is_not_feature_collection(self, lfc):
        assert not isinstance(lfc, FeatureCollection)

    def test_satisfies_lazy_spatial_object_protocol(self, lfc):
        """ARC-V4: lazy FCs implement :class:`LazySpatialObject`, not
        :class:`SpatialObject` — the eager-looking ``top_left_corner``
        property was intentionally removed because it silently triggers
        an O(partitions) dask reduction. Consumers that accept either
        backend type-check against the union of both protocols.
        """
        assert isinstance(lfc, LazySpatialObject)
        assert not isinstance(lfc, SpatialObject), (
            "LazyFC must NOT satisfy the eager SpatialObject protocol "
            "(would hide a lazy reduction behind a property access)"
        )

    def test_is_lazy_protocol_helper(self, lfc):
        assert is_lazy(lfc) is True

    def test_compute_returns_eager_feature_collection(self, lfc, small_gdf):
        eager = lfc.compute()
        assert isinstance(eager, FeatureCollection)
        assert len(eager) == 10
        assert eager.crs == small_gdf.crs

    def test_persist_returns_lazy_fc(self, lfc):
        from pyramids.feature import LazyFeatureCollection

        persisted = lfc.persist()
        assert isinstance(persisted, LazyFeatureCollection)

    def test_to_file_raises_not_implemented(self, lfc, tmp_path):
        with pytest.raises(NotImplementedError, match=r"compute\(\)\.to_file"):
            lfc.to_file(tmp_path / "out.geojson")

    def test_plot_raises_not_implemented(self, lfc):
        with pytest.raises(NotImplementedError, match=r"compute\(\)\.plot"):
            lfc.plot()

    def test_epsg_property(self, lfc):
        assert lfc.epsg == 4326

    def test_compute_total_bounds_replaces_top_left_corner(self, lfc):
        """ARC-V4: top_left_corner is gone; compute_total_bounds is the path.

        Test scenario:
            The previous eager-looking ``top_left_corner`` property on
            LazyFC silently ran an ``O(partitions)`` dask reduction.
            After ARC-V4 the property is removed; the explicit
            ``compute_total_bounds()`` helper is the supported way to
            materialise bounds. Pin both the removal and the new
            helper's 4-element return.
        """
        import numpy as np

        bounds = lfc.compute_total_bounds()
        assert isinstance(bounds, np.ndarray)
        assert bounds.shape == (4,)
        # Fixture builds Points at (i, i) for i in 0..9 → bbox [0, 0, 9, 9].
        assert bounds.tolist() == [0.0, 0.0, 9.0, 9.0]

    def test_inherited_ops_preserve_pyramids_subclass(self, lfc):
        """Inherited dask-geopandas ops must return LazyFeatureCollection.

        Test scenario:
            dask-geopandas / dask-expr do not honour pandas' classic
            ``_constructor`` hook. Without the ``__getattribute__``
            rebrand, methods like ``to_crs`` / ``clip`` / ``copy`` /
            ``drop_duplicates`` drop the :class:`LazyFeatureCollection`
            subclass and return a plain
            :class:`dask_geopandas.GeoDataFrame`, silently removing
            ``compute_total_bounds`` / ``epsg`` / ``is_lazy_fc`` from
            the result. Pin every common frame-returning inherited op.
        """
        from pyramids.feature import LazyFeatureCollection, is_lazy_fc

        assert isinstance(lfc.to_crs(3857), LazyFeatureCollection)
        assert isinstance(lfc.copy(), LazyFeatureCollection)
        assert isinstance(lfc.drop_duplicates(), LazyFeatureCollection)
        assert isinstance(
            lfc.repartition(npartitions=1),
            LazyFeatureCollection,
        )
        # pyramids-specific helpers survive the rebrand.
        reproj = lfc.to_crs(3857)
        assert is_lazy_fc(reproj)
        assert reproj.epsg == 3857
        bounds = reproj.compute_total_bounds()
        assert bounds.shape == (4,)

    def test_head_returns_eager_feature_collection(self, lfc):
        """``head(N)`` auto-computes to an eager geopandas frame.

        Test scenario:
            :meth:`dask_geopandas.GeoDataFrame.head` materialises the
            first N rows eagerly. The rebrand hook recognises the
            geopandas :class:`GeoDataFrame` return type and swaps it to
            :class:`FeatureCollection` so the pyramids type invariant
            holds on both sides of the lazy/eager boundary.
        """
        result = lfc.head(3)
        assert isinstance(result, FeatureCollection)
        assert len(result) == 3

    def test_repr_is_pyramids_branded(self, lfc):
        text = repr(lfc)
        assert text.startswith("LazyFeatureCollection(")
        assert "npartitions=2" in text

    def test_pickle_roundtrip(self, lfc, small_gdf):
        from pyramids.feature import LazyFeatureCollection

        restored = pickle.loads(pickle.dumps(lfc))
        assert isinstance(restored, LazyFeatureCollection)
        assert len(restored.compute()) == len(small_gdf)

    def test_read_file_classmethod_delegates_to_eager_reader(self, small_gdf, tmp_path):
        from pyramids.feature import LazyFeatureCollection

        p = tmp_path / "pts.geojson"
        small_gdf.to_file(p, driver="GeoJSON")
        lfc_direct = LazyFeatureCollection.read_file(str(p))
        assert isinstance(lfc_direct, LazyFeatureCollection)
        assert len(lfc_direct.compute()) == 10

    def test_from_dask_gdf_public_name_is_canonical(self, small_gdf):
        """ARC-V1: public ``from_dask_gdf`` is the supported constructor."""
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        ddf = dg.from_geopandas(small_gdf, npartitions=2)
        lfc_public = LazyFeatureCollection.from_dask_gdf(ddf)
        lfc_alias = LazyFeatureCollection._from_dask_gdf(ddf)
        assert isinstance(lfc_public, LazyFeatureCollection)
        assert isinstance(lfc_alias, LazyFeatureCollection)
        assert len(lfc_public.compute()) == len(lfc_alias.compute()) == 10

    def test_from_dask_gdf_preserves_state(self, small_gdf):
        """ARC-V2: class-swap drops no instance state.

        Test scenario:
            ``from_dask_gdf`` uses ``result.__class__ = cls``. The
            invariant that enables this safely is "LazyFeatureCollection
            adds no extra instance state beyond dask_geopandas.GeoDataFrame."
            If a future commit adds ``__slots__`` or an ``__init__``,
            this test catches it by comparing non-dunder ``vars()`` keys
            before and after the swap.
        """
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        ddf = dg.from_geopandas(small_gdf, npartitions=2)
        pre = {k for k in vars(ddf).keys() if not k.startswith("__")}
        lfc = LazyFeatureCollection.from_dask_gdf(ddf)
        post = {k for k in vars(lfc).keys() if not k.startswith("__")}
        assert (
            pre == post
        ), f"class-swap leaked state: added={post - pre}, dropped={pre - post}"

    def test_no_extra_slots(self):
        """ARC-V2: pin that LazyFeatureCollection declares no ``__slots__``.

        Test scenario:
            Slotted attributes on the subclass would be silently
            dropped by the class-swap constructor (the source
            dask-geopandas frame wouldn't have them set). Assert at
            the class level that no ``__slots__`` is declared.
        """
        from pyramids.feature import LazyFeatureCollection

        assert "__slots__" not in LazyFeatureCollection.__dict__, (
            "LazyFeatureCollection must not declare __slots__ while the "
            "class-swap constructor (from_dask_gdf) is in use — slotted "
            "attributes would be silently dropped by the swap."
        )
