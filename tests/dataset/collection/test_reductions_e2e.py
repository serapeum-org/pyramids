"""End-to-end tests for DatasetCollection reductions + pickle (seams).

 are each covered by per-task suites. This file exercises
the seams where one task's output feeds another:

1. Pickle + cross-process compute of :meth:`DatasetCollection.mean`
   on a spawn subprocess (the canonical dask.distributed shape).
2. :meth:`DatasetCollection.groupby` results validated against
   per-group numpy means computed directly.
3. :meth:`DatasetCollection.from_files` metadata survives pickle +
   unpickle; the reconstructed collection still computes correctly.
4. Lazy `.data` → `.min()` preserves RasterMeta-derived shape
   and dtype (reductions drop the time axis, nothing else).
"""

from __future__ import annotations

import multiprocessing
import pickle

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask
from pyramids.dataset import Dataset, DatasetCollection

try:
    import_dask("dask not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK = False
else:
    HAS_DASK = True

pytestmark = pytest.mark.lazy


requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


@pytest.fixture
def four_ramp_files(tmp_path):
    """Four timesteps filled with values 1..4 on a 3×4 grid."""
    paths = []
    for i in range(4):
        arr = np.full((3, 4), float(i + 1), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / f"f{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


def _worker_mean(payload: bytes) -> float:
    """Worker: unpickle a DatasetCollection + return mean scalar."""
    collection = pickle.loads(payload)
    return float(np.asarray(collection.mean()).mean())


@requires_dask
class TestCollectionReductionsE2E:
    """Cross-task integration for DatasetCollection lazy path."""

    def test_mean_via_spawn_subprocess(self, four_ramp_files):
        """mean() pickles + computes on a fresh worker process."""
        collection = DatasetCollection.from_files(four_ramp_files)
        payload = pickle.dumps(collection)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            scalar = pool.apply(_worker_mean, (payload,))
        assert scalar == pytest.approx(2.5)

    def test_groupby_results_match_manual(self, four_ramp_files):
        """Per-group means equal direct numpy computations."""
        collection = DatasetCollection.from_files(four_ramp_files)
        grouped = collection.groupby([0, 1, 0, 1]).mean()
        assert np.allclose(grouped[0], np.mean([1.0, 3.0]))
        assert np.allclose(grouped[1], np.mean([2.0, 4.0]))

    def test_pickle_collection_preserves_meta_and_computes(self, four_ramp_files):
        """Cross-pickle DatasetCollection still reduces + carries meta."""
        original = DatasetCollection.from_files(four_ramp_files)
        revived = pickle.loads(pickle.dumps(original))
        assert revived.time_length == original.time_length
        assert revived.meta.rows == original.meta.rows
        assert revived.meta.epsg == original.meta.epsg
        assert np.allclose(revived.mean(), original.mean())

    def test_min_reduction_preserves_meta_shape(self, four_ramp_files):
        """min() over time collapses axis 0 only, keeps (bands, rows, cols)."""
        collection = DatasetCollection.from_files(four_ramp_files)
        result = collection.min()
        assert result.shape == (1, 3, 4)
        assert result.dtype == np.dtype(collection.meta.dtype)
        assert np.allclose(result, 1.0)
