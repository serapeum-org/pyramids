"""`crop` and `apply` gain the deferred mode `to_crs` and `align` already had.

All four are per-timestep operations driven by the same `_apply_operator`, but
only two exposed `compute=False`. The other two hard-coded eager execution, so a
many-raster crop could not be folded into one dask graph the way a reproject
could.

The time axis is also carried onto the result now -- guarded on length, so a
stale axis is dropped rather than stamped onto a collection it no longer
describes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.dataset.collection import DatasetCollection

pytestmark = pytest.mark.core


def build_dataset(value: float) -> Dataset:
    """A 4x4 raster filled with `value`."""
    return Dataset.from_array(
        np.full((4, 4), value, dtype="float32"),
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    )


@pytest.fixture
def collection() -> DatasetCollection:
    """A three-timestep collection of identical small rasters."""
    datasets = [build_dataset(i) for i in range(3)]
    return DatasetCollection(datasets[0], time_length=len(datasets), datasets=datasets)


class TestApplyDeferred:
    """`apply` honours `compute=False`."""

    def test_eager_by_default(self, collection: DatasetCollection):
        """The default stays eager and returns a collection."""
        result = collection.apply(lambda a: a + 1)

        assert isinstance(result, DatasetCollection)
        assert len(result.datasets) == 3

    def test_deferred_returns_something_computable(self, collection: DatasetCollection):
        """`compute=False` defers, and computing it yields the collection."""
        dask = pytest.importorskip("dask")

        deferred = collection.apply(lambda a: a + 1, compute=False)

        assert not isinstance(deferred, DatasetCollection)
        computed = dask.compute(deferred, scheduler="synchronous")[0]
        assert isinstance(computed, DatasetCollection)
        assert len(computed.datasets) == 3

    def test_deferred_rejects_inplace(self, collection: DatasetCollection):
        """The `compute=False` + `inplace=True` combination stays refused."""
        pytest.importorskip("dask")

        with pytest.raises(ValueError, match="cannot be combined with inplace"):
            collection.apply(lambda a: a + 1, inplace=True, compute=False)


class TestCropDeferred:
    """`crop` honours `compute=False`."""

    def test_deferred_rejects_inplace(self, collection: DatasetCollection):
        """The same guard applies to crop."""
        pytest.importorskip("dask")

        with pytest.raises(ValueError, match="cannot be combined with inplace"):
            collection.crop(
                bbox=[0.0, 0.0, 2.0, 2.0], epsg=4326, inplace=True, compute=False
            )


class TestTimeAxisPropagation:
    """The time axis survives a per-timestep operation, when it still fits."""

    def test_a_matching_time_axis_is_carried(self, collection: DatasetCollection):
        """A time axis of the right length reaches the result."""
        collection.time = ["a", "b", "c"]

        result = collection.apply(lambda a: a + 1)

        assert result.time == ["a", "b", "c"]

    def test_a_stale_time_axis_is_dropped(self, collection: DatasetCollection):
        """An axis that no longer matches the handle count is not carried.

        `_time` is assigned directly here because the public setter validates
        length and refuses this state. The guard exists for the state arising
        another way -- an operation that replaces the handles without revisiting
        `_time` -- where stamping the stale axis onto a new collection would
        propagate the mismatch and make it harder to trace back.
        """
        collection._time = ["a", "b"]

        assert collection._derived_kwargs(3) == {"time": None}

    def test_no_time_axis_stays_absent(self, collection: DatasetCollection):
        """A collection without a time axis produces one without."""
        result = collection.apply(lambda a: a + 1)

        assert result.time is None
