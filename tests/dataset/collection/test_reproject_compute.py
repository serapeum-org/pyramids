"""Tests for the deferred ``compute=False`` reproject / align path (ARC-54).

``DatasetCollection.to_crs`` and ``align`` plan the operation once (a single
``Reprojector`` / ``Aligner`` reused across every timestep) and, with
``compute=False``, defer the whole thing into one :class:`dask.delayed.Delayed`
that builds the reprojected / aligned collection when computed. These tests
cover the deferred branch: the returned ``Delayed``, its computed result, the
``inplace`` conflict guard, and the no-EPSG fallback under ``compute=False``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection
from tests._marks import requires_dask

pytestmark = pytest.mark.lazy


@pytest.fixture
def align_ref() -> Dataset:
    """A 2x3 EPSG:4326 template used as the alignment reference (differs in size)."""
    return Dataset.create_from_array(
        np.zeros((2, 3), dtype=np.float32),
        top_left_corner=(0.0, 4.0),
        cell_size=2.0,
        epsg=4326,
    )


class TestToCrsCompute:
    """Deferred ``to_crs(compute=False)`` over the plan-once EPSG path."""

    @requires_dask
    def test_returns_delayed(self, three_files):
        """``compute=False`` returns a ``dask.delayed.Delayed``, not a collection.

        Test scenario:
            ``to_crs(3857, compute=False)`` — expected: a ``Delayed`` object
            (the whole reproject deferred into one graph).
        """
        from dask.delayed import Delayed

        collection = DatasetCollection.from_files(three_files)
        deferred = collection.to_crs(3857, compute=False)
        assert isinstance(deferred, Delayed), (
            f"expected a Delayed, got {type(deferred)}"
        )

    @requires_dask
    def test_delayed_compute_reprojects(self, three_files):
        """Computing the ``Delayed`` yields a reprojected collection.

        Test scenario:
            ``to_crs(3857, compute=False).compute()`` — expected: a
            ``DatasetCollection`` at EPSG 3857 with the source ``time_length``.
        """
        collection = DatasetCollection.from_files(three_files)
        result = collection.to_crs(3857, compute=False).compute()
        assert isinstance(result, DatasetCollection), (
            f"expected a collection, got {type(result)}"
        )
        assert result.base.epsg == 3857, f"expected EPSG 3857, got {result.base.epsg}"
        assert result.time_length == 3, (
            f"time_length should be preserved, got {result.time_length}"
        )

    @requires_dask
    def test_compute_false_with_inplace_raises(self, three_files):
        """``compute=False`` combined with ``inplace=True`` raises ``ValueError``.

        Test scenario:
            Both flags together — expected: a ``ValueError`` (a deferred graph
            cannot mutate the collection in place).
        """
        collection = DatasetCollection.from_files(three_files)
        with pytest.raises(
            ValueError, match="compute=False cannot be combined with inplace"
        ):
            collection.to_crs(3857, inplace=True, compute=False)

    @requires_dask
    def test_non_epsg_target_deferred(self, three_files):
        """A no-EPSG target under ``compute=False`` defers the direct fallback.

        Test scenario:
            ``to_crs(<proj4 LAEA>, compute=False).compute()`` — expected: a
            ``DatasetCollection`` with the source ``time_length`` (the per-step
            ``dask.delayed(ds.to_crs)`` fallback ran, not the ``Reprojector``).
        """
        proj4 = "+proj=laea +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        collection = DatasetCollection.from_files(three_files)
        result = collection.to_crs(proj4, compute=False).compute()
        assert isinstance(result, DatasetCollection), (
            f"expected a collection, got {type(result)}"
        )
        assert result.time_length == 3, (
            f"time_length should be preserved, got {result.time_length}"
        )


class TestAlignCompute:
    """Deferred ``align(compute=False)`` over the plan-once ``Aligner`` path."""

    @requires_dask
    def test_returns_delayed(self, three_files, align_ref):
        """``align(ref, compute=False)`` returns a ``dask.delayed.Delayed``.

        Test scenario:
            ``align(ref, compute=False)`` — expected: a ``Delayed`` object.
        """
        from dask.delayed import Delayed

        collection = DatasetCollection.from_files(three_files)
        deferred = collection.align(align_ref, compute=False)
        assert isinstance(deferred, Delayed), (
            f"expected a Delayed, got {type(deferred)}"
        )

    @requires_dask
    def test_delayed_compute_aligns(self, three_files, align_ref):
        """Computing the ``Delayed`` yields a collection aligned to the reference.

        Test scenario:
            ``align(ref, compute=False).compute()`` — expected: a
            ``DatasetCollection`` whose grid matches ``ref`` (rows/cols) with the
            source ``time_length``.
        """
        collection = DatasetCollection.from_files(three_files)
        result = collection.align(align_ref, compute=False).compute()
        assert isinstance(result, DatasetCollection), (
            f"expected a collection, got {type(result)}"
        )
        assert result.base.rows == align_ref.rows, (
            f"aligned rows {result.base.rows} != reference {align_ref.rows}"
        )
        assert result.base.columns == align_ref.columns, (
            f"aligned columns {result.base.columns} != reference {align_ref.columns}"
        )
        assert result.time_length == 3, (
            f"time_length should be preserved, got {result.time_length}"
        )

    @requires_dask
    def test_compute_false_with_inplace_raises(self, three_files, align_ref):
        """``align(compute=False, inplace=True)`` raises ``ValueError``.

        Test scenario:
            Both flags together — expected: a ``ValueError`` from the shared
            ``_apply_operator`` guard.
        """
        collection = DatasetCollection.from_files(three_files)
        with pytest.raises(
            ValueError, match="compute=False cannot be combined with inplace"
        ):
            collection.align(align_ref, inplace=True, compute=False)
