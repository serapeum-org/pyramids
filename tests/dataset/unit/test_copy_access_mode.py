"""An in-memory copy is writable, whatever the source was.

`Dataset.copy(path=None)` used to hand back a dataset carrying the source's own
access mode, so copying a read-only raster produced a read-only copy -- an
in-memory object, owned entirely by the caller, that refused to be written to.
The copy exists precisely so the caller has something they can change.

This is a public behaviour change and not a deduplication, so it is pinned here
for both source modes rather than left to be discovered.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.core

GEO = GeoReference(top_left_corner=(0.0, 5.0), cell_size=1.0, epsg=4326)


@pytest.fixture
def source_path(tmp_path):
    """A small GeoTIFF on disk, so it can be reopened in either mode."""
    path = tmp_path / "source.tif"
    Dataset.from_array(np.ones((4, 5), dtype=np.float32), geo_ref=GEO, path=str(path))
    return str(path)


class TestTheInMemoryCopyIsWritable:
    """The mode the copy carries, per source mode."""

    @pytest.mark.parametrize("read_only", [True, False], ids=["read-only", "writable"])
    def test_the_copy_is_writable_from_either_source_mode(self, source_path, read_only):
        """A read-only source no longer yields a read-only copy.

        Args:
            source_path: A GeoTIFF fixture on disk.
            read_only: How the source is opened.

        Test scenario:
            The copy is a fresh in-memory raster the caller owns outright, so
            its mode is a property of the copy rather than inherited from
            whatever the source happened to be opened as.
        """
        source = Dataset.read_file(source_path, read_only=read_only)

        assert source.copy().access == "write"

    def test_the_copy_actually_accepts_a_write(self, source_path):
        """The mode string is only worth pinning if it is honoured.

        Test scenario:
            Copying a read-only raster and then writing to the copy used to
            raise `ReadOnlyError`, which is the concrete symptom the mode
            change fixes.
        """
        source = Dataset.read_file(source_path, read_only=True)

        copy = source.copy()
        copy.write_array(np.full((4, 5), 7.0, dtype=np.float32))

        assert copy.read_array()[0, 0] == pytest.approx(7.0)

    def test_the_source_is_left_read_only(self, source_path):
        """Copying must not relax the original's protection.

        Test scenario:
            The change is about the copy. A source opened read-only stays
            read-only, so the defensive-snapshot use of `copy()` still gets a
            source nobody can write through.
        """
        source = Dataset.read_file(source_path, read_only=True)

        _ = source.copy()

        assert source.access == "read_only"
