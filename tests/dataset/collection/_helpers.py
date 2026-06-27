"""Shared builders for the DatasetCollection test modules.

Kept in a non-``test_`` module so pytest does not collect it, and so the
``to_netcdf`` happy-path suite (xarray-marked) and the missing-xarray branch
(core) can share one collection builder instead of duplicating it.
"""

from __future__ import annotations

import os

import numpy as np

from pyramids.dataset import Dataset, DatasetCollection


def make_int16_collection(tmp_path, count: int = 2, no_data_value: int = -9999):
    """Build a small int16 file-backed collection.

    Args:
        tmp_path: pytest temp directory.
        count: Number of timesteps to materialise.
        no_data_value: Value stamped as nodata on each timestep.

    Returns:
        tuple[DatasetCollection, list[str]]: the collection plus its backing
        paths, so tests can introspect ``_files``.
    """
    paths = []
    for i in range(count):
        arr = np.arange(20, dtype="int16").reshape(4, 5) + 100 * i
        p = os.path.join(str(tmp_path), f"t{i}.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0, 0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=no_data_value,
            path=p,
        ).close()
        paths.append(p)
    return DatasetCollection.from_files(paths), paths
