"""Tests for the :class:`pyramids.base.protocols.SpatialObject` protocol.

both :class:`pyramids.dataset.Dataset` and
:class:`pyramids.feature.FeatureCollection` implement the shared
`SpatialObject` protocol, so callers can write generic code that
accepts either a raster or a vector.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask_geopandas
from pyramids.base.protocols import LazySpatialObject, SpatialObject
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

try:
    import_dask_geopandas("dask-geopandas not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK_GP = False
else:
    HAS_DASK_GP = True

pytestmark = pytest.mark.core


@pytest.fixture
def fc() -> FeatureCollection:
    """A simple FeatureCollection with one polygon in EPSG:32636."""
    gdf = gpd.GeoDataFrame(
        {"v": [1]},
        geometry=[box(500000.0, 3400000.0, 510000.0, 3410000.0)],
        crs="EPSG:32636",
    )
    return FeatureCollection(gdf)


@pytest.fixture
def ds() -> Dataset:
    """A small in-memory Dataset in EPSG:32636."""
    src = Dataset.create(
        cell_size=1000.0,
        rows=10,
        columns=10,
        dtype="float32",
        bands=1,
        top_left_corner=(500000.0, 3410000.0),
        epsg=32636,
        no_data_value=-9999.0,
    )
    src.raster.GetRasterBand(1).WriteArray(np.ones((10, 10), dtype=np.float32))
    return src


class TestSpatialObjectProtocol:
    """Both Dataset and FeatureCollection satisfy SpatialObject."""

    def test_feature_collection_is_spatial_object(self, fc: FeatureCollection):
        """`isinstance(fc, SpatialObject)` is True at runtime."""
        assert isinstance(fc, SpatialObject)

    def test_dataset_is_spatial_object(self, ds: Dataset):
        """`isinstance(ds, SpatialObject)` is True at runtime."""
        assert isinstance(ds, SpatialObject)

    def test_both_expose_epsg(self, fc: FeatureCollection, ds: Dataset):
        """Both types expose `epsg` as the same kind of value."""
        assert fc.epsg == 32636
        assert ds.epsg == 32636

    def test_both_expose_total_bounds(self, fc: FeatureCollection, ds: Dataset):
        """Both types expose `total_bounds` as [minx, miny, maxx, maxy]."""
        fc_bounds = np.asarray(fc.total_bounds)
        ds_bounds = np.asarray(ds.total_bounds)
        assert fc_bounds.shape == (4,)
        assert ds_bounds.shape == (4,)
        # FC bounds match the box we built.
        assert np.allclose(fc_bounds, [500000.0, 3400000.0, 510000.0, 3410000.0])
        # Dataset bbox reflects its geotransform (10 cols × 10 rows × 1000m).
        assert np.allclose(ds_bounds, [500000.0, 3400000.0, 510000.0, 3410000.0])

    def test_both_expose_top_left_corner(self, fc: FeatureCollection, ds: Dataset):
        """Both types expose `top_left_corner` as [minx, maxy]."""
        fc_tl = list(fc.top_left_corner)
        ds_tl = list(ds.top_left_corner)
        assert fc_tl == [500000.0, 3410000.0]
        assert ds_tl == [500000.0, 3410000.0]

    def test_generic_consumer_accepts_both(self, fc: FeatureCollection, ds: Dataset):
        """A function typed against SpatialObject accepts both classes.

        This is the real value of the protocol — users can write
        cross-type utilities without importing both concrete classes.
        """

        def epsg_of(obj: SpatialObject) -> int | None:
            return obj.epsg

        assert epsg_of(fc) == 32636
        assert epsg_of(ds) == 32636


class TestSpatialObjectNegative:
    """Objects missing required attributes must NOT be SpatialObjects."""

    def test_plain_string_is_not_spatial_object(self):
        assert not isinstance("not spatial", SpatialObject)

    def test_dict_is_not_spatial_object(self):
        assert not isinstance({"epsg": 4326}, SpatialObject)

    def test_plain_geodataframe_is_spatial_object_only_when_attrs_present(self):
        """A plain GeoDataFrame does not have `epsg` / `top_left_corner`.

        So it does NOT satisfy the protocol even though it has
        `total_bounds` / `to_file` / `read_file` / `plot`.
        """
        plain = gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        assert not isinstance(plain, SpatialObject)


@pytest.mark.skipif(not HAS_DASK_GP, reason="dask-geopandas not installed")
class TestLazySpatialObject:
    """:class:`LazyFeatureCollection` satisfies :class:`LazySpatialObject`.

    The eager :class:`SpatialObject` protocol is NOT satisfied by
    LazyFeatureCollection (intentionally — it would require
    `top_left_corner` to silently run an `O(partitions)` dask
    reduction). Lazy objects satisfy the separate
    :class:`LazySpatialObject` protocol, which exposes `npartitions`
    / `compute` / `persist` and a lazy `total_bounds`. Consumers
    that want to accept both types should type against
    `SpatialObject | LazySpatialObject` and branch via
    :func:`pyramids.feature.is_lazy_fc`.
    """

    def test_lazy_feature_collection_satisfies_lazy_protocol(self):
        """`isinstance(lfc, LazySpatialObject)` is True at runtime."""
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        gdf = gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        ddf = dg.from_geopandas(gdf, npartitions=1)
        lfc = LazyFeatureCollection.from_dask_gdf(ddf)
        assert isinstance(lfc, LazySpatialObject)

    def test_lazy_feature_collection_does_not_satisfy_eager_protocol(self):
        """LazyFC no longer fakes eager `top_left_corner`.

        Test scenario:
            After the split the lazy class drops
            `top_left_corner` entirely (rather than silently
            calling `.compute()` inside it). As a result,
            `isinstance(lfc, SpatialObject)` returns False —
            intentionally, honestly.
        """
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        gdf = gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        ddf = dg.from_geopandas(gdf, npartitions=1)
        lfc = LazyFeatureCollection.from_dask_gdf(ddf)
        assert not isinstance(lfc, SpatialObject)

    def test_generic_consumer_accepts_lazy_via_union(self):
        """A function typed against `SpatialObject | LazySpatialObject` accepts both."""
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        gdf = gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[box(500000.0, 3400000.0, 510000.0, 3410000.0)],
            crs="EPSG:32636",
        )
        ddf = dg.from_geopandas(gdf, npartitions=1)
        lfc = LazyFeatureCollection.from_dask_gdf(ddf)

        def epsg_of(obj: SpatialObject | LazySpatialObject) -> int | None:
            return obj.epsg

        assert epsg_of(lfc) == 32636


@pytest.mark.skipif(not HAS_DASK_GP, reason="dask-geopandas not installed")
class TestComputeTotalBounds:
    """the explicit `compute_total_bounds()` helper forces reduction."""

    def test_returns_four_element_array(self):
        """The helper materialises the lazy Scalar into a concrete array."""
        import dask_geopandas as dg
        import numpy as np

        from pyramids.feature import LazyFeatureCollection

        gdf = gpd.GeoDataFrame(
            {"v": [1, 2]},
            geometry=[box(0, 0, 1, 1), box(2, 3, 4, 5)],
            crs="EPSG:4326",
        )
        ddf = dg.from_geopandas(gdf, npartitions=1)
        lfc = LazyFeatureCollection.from_dask_gdf(ddf)
        bounds = lfc.compute_total_bounds()
        assert isinstance(bounds, np.ndarray)
        assert bounds.shape == (4,)
        assert bounds.tolist() == [0.0, 0.0, 4.0, 5.0]

    def test_top_left_corner_property_is_gone(self):
        """eager-only `top_left_corner` no longer exists on LazyFC.

        Test scenario:
            The previous implementation had a `top_left_corner`
            property that silently called `.compute()`. After the
            split it is removed outright; `hasattr` reports
            False, and explicit access raises `AttributeError`.
        """
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        gdf = gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        ddf = dg.from_geopandas(gdf, npartitions=1)
        lfc = LazyFeatureCollection.from_dask_gdf(ddf)
        assert not hasattr(lfc, "top_left_corner")
        with pytest.raises(AttributeError, match="top_left_corner"):
            lfc.top_left_corner  # noqa: B018
