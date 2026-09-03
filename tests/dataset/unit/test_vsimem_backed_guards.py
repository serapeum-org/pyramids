"""A `/vsimem`-backed dataset must not look on-disk to the guards.

`Dataset.from_bytes(data, name="x")` stages the bytes in `/vsimem` and records
the requested name as a cosmetic `_file_name`. Two predicates asked only whether
that name was empty or started with `/vsimem/`:

* `__reduce__` produced a pickle recipe that reopened a path which does not
  exist, so the failure surfaced on unpickle, in another process;
* the COG `_on_disk_path` accepted a MEM container whose `file_name` is a bare
  driver placeholder such as `"netcdf"`, and returned it as a validatable path.

The pickling case is a real defect and these tests fail against the previous
behaviour. The COG case is hardening: `is_cog` already answered `False`, because
the validation downstream rejected the non-existent path, so the predicate was
returning something it should never have returned but no caller could see it.
The tests below pin the contract either way.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.core


@pytest.fixture
def geotiff_bytes(tmp_path: Path) -> bytes:
    """The bytes of a small on-disk GeoTIFF."""
    source = tmp_path / "src.tif"
    Dataset.from_array(
        np.arange(16, dtype="float32").reshape(4, 4),
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    ).to_file(str(source))
    return source.read_bytes()


class TestPicklingAVsimemBackedDataset:
    """The recipe is refused up front, not on unpickle."""

    def test_from_bytes_with_a_name_refuses_to_pickle(self, geotiff_bytes: bytes):
        """A cosmetic `name=` does not make an in-memory dataset picklable."""
        dataset = Dataset.from_bytes(geotiff_bytes, name="x")

        with pytest.raises(TypeError):
            pickle.dumps(dataset)

    def test_from_bytes_without_a_name_still_refuses(self, geotiff_bytes: bytes):
        """The pre-existing path test keeps working."""
        with pytest.raises(TypeError):
            pickle.dumps(Dataset.from_bytes(geotiff_bytes))

    def test_an_on_disk_dataset_still_pickles(self, tmp_path: Path):
        """The guard did not widen to real files."""
        source = tmp_path / "real.tif"
        Dataset.from_array(
            np.ones((4, 4), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        ).to_file(str(source))

        assert pickle.dumps(Dataset.read_file(str(source)))


class TestCogPredicateOnMemoryDatasets:
    """`is_cog` answers False for anything with no real backing file."""

    def test_a_vsimem_backed_dataset_is_not_a_cog(self, geotiff_bytes: bytes):
        """The `name=` placeholder no longer reads as an on-disk path."""
        dataset = Dataset.from_bytes(geotiff_bytes, name="x")

        assert dataset.is_cog is False

    def test_a_mem_dataset_is_not_a_cog(self):
        """A plain in-memory raster has nothing to validate."""
        dataset = Dataset.from_array(
            np.ones((4, 4), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        )

        assert dataset.is_cog is False

    def test_a_closed_dataset_does_not_raise(self, tmp_path: Path):
        """`driver_type` requires an open handle, so it is asked last.

        Pinned because reordering these tests would turn a `False` into a
        `RuntimeError` out of a property.
        """
        source = tmp_path / "real.tif"
        Dataset.from_array(
            np.ones((4, 4), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        ).to_file(str(source))
        dataset = Dataset.read_file(str(source))
        dataset.close()

        assert dataset.is_cog in (True, False)
