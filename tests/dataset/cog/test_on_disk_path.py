"""Tests for the shared COG._on_disk_path predicate (ARC-5).

is_cog, validate_cog, and info must all agree on what counts as a validatable
on-disk backing file, via one shared predicate.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset
from tests.dataset.cog.conftest import COG_GEOTRANSFORM

pytestmark = pytest.mark.core


@pytest.fixture
def mem_dataset() -> Dataset:
    """An in-memory Dataset with no on-disk backing file.

    Returns:
        Dataset: A MEM-backed dataset.
    """
    arr = np.ones((16, 16), dtype="float32")
    return Dataset.create_from_array(arr, geo=COG_GEOTRANSFORM, epsg=4326)


class TestOnDiskPath:
    """Tests for the shared on-disk-path predicate and its consumers."""

    def test_mem_dataset_has_no_path(self, mem_dataset):
        """A MEM dataset resolves to no on-disk path.

        Args:
            mem_dataset: Fixture MEM Dataset.

        Test scenario:
            _on_disk_path returns None for an unsaved/in-memory dataset.
        """
        assert mem_dataset.cog._on_disk_path() is None, "MEM dataset must have no path"

    def test_saved_dataset_has_path(self, mem_dataset, tmp_path):
        """A dataset written to a COG resolves to its on-disk path.

        Args:
            mem_dataset: Fixture MEM Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            After to_cog the reopened dataset reports a real path.
        """
        out = mem_dataset.to_cog(tmp_path / "x.tif")
        reopened = Dataset.read_file(str(out))
        assert reopened.cog._on_disk_path() == reopened.file_name, "path mismatch"

    def test_consumers_agree_on_mem(self, mem_dataset):
        """is_cog/validate_cog/info agree a MEM dataset is not validatable.

        Args:
            mem_dataset: Fixture MEM Dataset.

        Test scenario:
            is_cog is False; validate_cog and info both raise FileNotFoundError.
        """
        assert mem_dataset.is_cog is False, "MEM dataset is_cog must be False"
        with pytest.raises(FileNotFoundError):
            mem_dataset.validate_cog()
        with pytest.raises(FileNotFoundError):
            mem_dataset.cog_info()
