"""ARC-31: FeatureCollection is pickle-safe end-to-end.

Distributed frameworks (``dask.distributed``, ``multiprocessing``,
``concurrent.futures.ProcessPoolExecutor``, ``ray``) pickle objects
across processes. Under ARC-1a the public API of
``FeatureCollection`` is a pure ``geopandas.GeoDataFrame`` subclass
— no ``ogr.DataSource`` / ``gdal.Dataset`` is ever held as instance
state. ARC-31 locks that invariant down with tests so a future
accidental ``self._ds = ogr.Open(...)`` regression is caught
immediately.

Covered:

* Plain ``pickle`` round-trip preserves type, rows, columns, CRS.
* ``cloudpickle`` round-trip (dask.distributed uses it preferentially).
* The invariant survives operations that internally build a
  short-lived OGR DataSource — the scratch object lives inside
  :func:`pyramids.feature._ogr.as_datasource`'s ``with`` block and
  is released before the call returns; ``self`` never holds it.
* The ARC-6 epsg cache (attributes listed in ``_metadata``) is
  preserved across pickle — pandas subclass contract.
"""

from __future__ import annotations

import importlib.util
import pickle

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, box

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture
def fc() -> FeatureCollection:
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"id": [1, 2, 3], "name": ["a", "b", "c"]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326",
        )
    )


@pytest.fixture
def polygon_fc() -> FeatureCollection:
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"class_id": [7]},
            geometry=[box(0, 0, 5, 5)],
            crs="EPSG:32636",
        )
    )


class TestPicklePlain:
    """Standard :mod:`pickle` round-trip preserves identity and contents."""

    def test_round_trip_preserves_type(self, fc: FeatureCollection):
        restored = pickle.loads(pickle.dumps(fc))
        assert isinstance(restored, FeatureCollection)
        assert type(restored) is FeatureCollection

    def test_round_trip_preserves_rows(self, fc: FeatureCollection):
        restored = pickle.loads(pickle.dumps(fc))
        assert len(restored) == len(fc)
        assert list(restored["id"]) == list(fc["id"])
        assert list(restored["name"]) == list(fc["name"])

    def test_round_trip_preserves_crs(self, fc: FeatureCollection):
        restored = pickle.loads(pickle.dumps(fc))
        assert restored.epsg == 4326

    def test_round_trip_preserves_geometries(self, fc: FeatureCollection):
        restored = pickle.loads(pickle.dumps(fc))
        for g_orig, g_new in zip(fc.geometry, restored.geometry):
            assert g_orig.equals(g_new)

    def test_highest_protocol(self, fc: FeatureCollection):
        """Works with the current highest pickle protocol."""
        data = pickle.dumps(fc, protocol=pickle.HIGHEST_PROTOCOL)
        restored = pickle.loads(data)
        assert len(restored) == len(fc)


class TestMetadataDedup:
    """C3: ``_metadata`` is de-duplicated against geopandas upstream additions.

    ARC-31 fixed the pickle bug by splatting ``GeoDataFrame._metadata``
    first; if a future geopandas release adds one of our own names
    (``_epsg_cache_crs`` / ``_epsg_cache_value``) to the parent list,
    the pyramids subclass used to end up with a duplicate. Python
    allows duplicate list entries, but pandas' ``_metadata`` processing
    may not be idempotent. ``dict.fromkeys`` de-dupes while preserving
    insertion order.
    """

    def test_metadata_has_no_duplicates(self):
        assert len(FeatureCollection._metadata) == len(set(FeatureCollection._metadata))

    def test_metadata_preserves_parent_ordering(self):
        from geopandas import GeoDataFrame as _GDF

        # Parent entries come first, pyramids additions last.
        parent = list(_GDF._metadata)
        for idx, name in enumerate(parent):
            assert FeatureCollection._metadata[idx] == name

    def test_metadata_contains_pyramids_caches(self):
        assert "_epsg_cache_crs" in FeatureCollection._metadata
        assert "_epsg_cache_value" in FeatureCollection._metadata

    def test_metadata_stays_deduped_in_subclass(self):
        """C3: a further subclass does not reintroduce duplicates.

        Test scenario:
            A user may subclass FeatureCollection and splat its own
            ``_metadata`` list. Even if the child re-adds a name already
            present in the parent, the final list must still resolve to
            a unique set. The class attribute pattern relies on Python's
            MRO, so this mirrors geopandas' contract.
        """

        class ChildFC(FeatureCollection):
            _metadata = list(
                dict.fromkeys(
                    [
                        *FeatureCollection._metadata,
                        "_epsg_cache_crs",  # intentionally re-listed
                    ]
                )
            )

        assert len(ChildFC._metadata) == len(set(ChildFC._metadata))

    def test_metadata_survives_copy(self, fc: FeatureCollection):
        """C3: ``.copy()`` preserves the de-duped ``_metadata`` list."""
        duplicate = fc.copy()
        assert list(type(duplicate)._metadata) == list(FeatureCollection._metadata)
        assert len(type(duplicate)._metadata) == len(set(type(duplicate)._metadata))


@pytest.mark.skipif(
    importlib.util.find_spec("cloudpickle") is None,
    reason="cloudpickle not installed",
)
class TestCloudpickle:
    """``cloudpickle`` (dask.distributed) round-trip works."""

    def test_round_trip(self, fc: FeatureCollection):
        import cloudpickle

        restored = cloudpickle.loads(cloudpickle.dumps(fc))
        assert isinstance(restored, FeatureCollection)
        assert len(restored) == len(fc)
        assert restored.epsg == 4326


class TestNoOgrStateLeak:
    """No OGR/GDAL handle is ever held as instance state.

    The invariant ARC-1a established is that the only state on a
    FeatureCollection is the GeoDataFrame data plus the ARC-6 cache
    attributes in ``_metadata``. Methods that need OGR (``rasterize``,
    etc.) open a DataSource via ``pyramids.feature._ogr.as_datasource``
    inside a ``with`` block and drop it before returning. These
    tests pickle the FC AFTER such a call and prove ``self`` was not
    contaminated.
    """

    def test_pickle_after_rasterize(self, polygon_fc: FeatureCollection):
        """Dataset.from_features opens an internal DataSource; FC still pickles."""
        # Side-effect-only call that exercises _ogr.as_datasource
        # under the hood.
        _ = Dataset.from_features(polygon_fc, cell_size=1.0, column_name="class_id")

        # polygon_fc must still be pickle-safe after the above.
        restored = pickle.loads(pickle.dumps(polygon_fc))
        assert isinstance(restored, FeatureCollection)
        assert len(restored) == 1
        assert restored.epsg == 32636

    def test_no_gdal_or_ogr_in_instance_dict(self, polygon_fc: FeatureCollection):
        """Reflectively verify no OGR/GDAL handle lives on ``self``."""
        from osgeo import gdal, ogr

        _ = Dataset.from_features(polygon_fc, cell_size=1.0, column_name="class_id")
        for name, value in polygon_fc.__dict__.items():
            assert not isinstance(value, (ogr.DataSource, gdal.Dataset)), (
                f"instance attribute {name!r} holds an OGR/GDAL "
                f"handle ({type(value).__name__}); that breaks pickling "
                f"(ARC-31 regression)."
            )


class TestMetadataSurvives:
    """The ARC-6 epsg cache survives pickle (listed in ``_metadata``)."""

    def test_epsg_cache_preserved(self, fc: FeatureCollection):
        # Prime the cache.
        assert fc.epsg == 4326
        assert fc._epsg_cache_value == 4326
        # Round-trip.
        restored = pickle.loads(pickle.dumps(fc))
        # Cache attributes are in _metadata so pandas preserves them.
        assert hasattr(restored, "_epsg_cache_value")
        assert restored._epsg_cache_value == 4326


class TestNestedStructures:
    """Pickled inside a list/dict still round-trips."""

    def test_list_of_fcs(self, fc: FeatureCollection, polygon_fc: FeatureCollection):
        payload = [fc, polygon_fc]
        restored = pickle.loads(pickle.dumps(payload))
        assert len(restored) == 2
        assert all(isinstance(r, FeatureCollection) for r in restored)
        assert restored[0].epsg == 4326
        assert restored[1].epsg == 32636
