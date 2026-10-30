"""Benchmarks for DatasetCollection (temporal stacks): reductions, apply, COG export, mosaic.

A collection is a time-series of co-registered rasters backed by a dask cube, so the headline
costs are the across-time reduction (``mean``), the per-timestep map (``apply``), the per-slice
COG export (``to_cog_stack``) and the mosaic (``merge``). The template is deliberately smaller
than the raster-I/O fixtures (512x512 x 12 steps) so a full sweep of these heavier,
multi-timestep operations still finishes in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection, GeoReference

TIME_LENGTH = 12


@pytest.fixture(scope="session")
def collection_template(geo_ref: GeoReference) -> Dataset:
    """A 512x512 float32 template raster that supplies the collection's geobox."""
    array = np.random.default_rng(1).random((512, 512), dtype=np.float32)
    return Dataset.from_array(array, geo_ref=geo_ref, no_data_value=-9999.0)


@pytest.fixture(scope="session")
def collection(collection_template: Dataset) -> DatasetCollection:
    """An in-memory 12-timestep collection sharing the template's geobox."""
    return DatasetCollection.from_dataset(collection_template, TIME_LENGTH)


@pytest.mark.benchmark(group="collection")
def test_from_dataset(benchmark, collection_template: Dataset):
    """Scaffold a 12-timestep in-memory collection from a template raster."""
    benchmark(lambda: DatasetCollection.from_dataset(collection_template, TIME_LENGTH))


@pytest.mark.benchmark(group="collection")
def test_mean_reduction(benchmark, collection: DatasetCollection):
    """Element-wise mean across the time axis (dask reduction, materialized)."""
    result = benchmark(lambda: collection.mean(skipna=True))
    assert result.shape[-2:] == (512, 512)


@pytest.mark.benchmark(group="collection")
def test_apply(benchmark, collection: DatasetCollection):
    """Map a per-pixel ufunc over every timestep (out-of-place, computed)."""
    benchmark(lambda: collection.apply(np.sqrt, compute=True))


@pytest.mark.benchmark(group="collection")
def test_to_cog_stack(benchmark, collection: DatasetCollection, tmp_path):
    """Export each timestep of the collection as an individual COG."""
    out = tmp_path / "cog_stack"
    benchmark(lambda: collection.to_cog_stack(str(out), overwrite=True))


@pytest.mark.benchmark(group="collection")
def test_merge(benchmark, collection: DatasetCollection, tmp_path):
    """Mosaic the collection's co-registered slices into a single raster."""
    dst = tmp_path / "merged.tif"
    benchmark(lambda: collection.merge(str(dst)))
