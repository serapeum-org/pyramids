"""Tests for :class:`pyramids.dataset.ops.reproject.Reprojector`.

DASK-9: plan-once, apply-many reprojection operator + :class:`Aligner`
subclass that targets a reference :class:`Dataset` geobox. Supports
``compute=False`` for deferred (dask.delayed) application.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.dataset.ops.reproject import (
    Aligner,
    Reprojector,
    ReprojectPlan,
)

pytestmark = pytest.mark.core

try:
    from dask.delayed import Delayed

    HAS_DASK = True
except ImportError:  # pragma: no cover
    Delayed = None
    HAS_DASK = False


requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


@pytest.fixture
def wgs84_dataset(tmp_path):
    arr = np.zeros((4, 4), dtype=np.float32)
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=(0.0, 4.0),
        cell_size=1.0,
        epsg=4326,
    )
    return ds


@pytest.fixture
def wgs84_dataset_fine(tmp_path):
    arr = np.zeros((8, 8), dtype=np.float32)
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=(0.0, 4.0),
        cell_size=0.5,
        epsg=4326,
    )
    return ds


class TestReprojectPlanPicklable:
    """``ReprojectPlan`` is a frozen dataclass; must round-trip pickle."""

    def test_pickle_roundtrip(self):
        plan = ReprojectPlan(target_epsg=3857, method="cubic")
        restored = pickle.loads(pickle.dumps(plan))
        assert restored.target_epsg == 3857
        assert restored.method == "cubic"
        assert restored == plan

    def test_frozen(self):
        plan = ReprojectPlan(target_epsg=3857)
        with pytest.raises(Exception):
            plan.target_epsg = 4326  # type: ignore


class TestReprojectorEager:
    """``Reprojector`` applied with ``compute=True`` yields a new Dataset."""

    def test_default_compute_true(self, wgs84_dataset):
        op = Reprojector(target_epsg=3857)
        out = op(wgs84_dataset)
        assert isinstance(out, Dataset)
        assert out.epsg == 3857

    def test_explicit_compute_true(self, wgs84_dataset):
        op = Reprojector(target_epsg=3857)
        out = op(wgs84_dataset, compute=True)
        assert out.epsg == 3857

    def test_reuse_across_multiple_sources(self, wgs84_dataset, wgs84_dataset_fine):
        op = Reprojector(target_epsg=3857)
        a = op(wgs84_dataset)
        b = op(wgs84_dataset_fine)
        assert a.epsg == 3857
        assert b.epsg == 3857

    def test_plan_property_exposes_spec(self):
        op = Reprojector(target_epsg=3857, method="cubic", maintain_alignment=True)
        assert op.plan.target_epsg == 3857
        assert op.plan.method == "cubic"
        assert op.plan.maintain_alignment is True


class TestReprojectorPickle:
    """Reprojector instances themselves must be picklable (plan is frozen)."""

    def test_pickle_roundtrip(self):
        op = Reprojector(target_epsg=3857, method="cubic")
        restored = pickle.loads(pickle.dumps(op))
        assert restored.plan == op.plan


class TestReprojectorLazy:
    """``compute=False`` returns :class:`dask.delayed.Delayed`."""

    @requires_dask
    def test_returns_delayed(self, wgs84_dataset):
        op = Reprojector(target_epsg=3857)
        result = op(wgs84_dataset, compute=False)
        assert isinstance(result, Delayed)

    @requires_dask
    def test_delayed_compute_equals_eager(self, wgs84_dataset):
        op = Reprojector(target_epsg=3857)
        eager = op(wgs84_dataset)
        lazy = op(wgs84_dataset, compute=False).compute()
        assert lazy.epsg == eager.epsg
        assert lazy.rows == eager.rows


class TestAlignerEager:
    """``Aligner`` matches the reference geobox."""

    def test_aligns_to_reference_shape(self, wgs84_dataset, wgs84_dataset_fine):
        aligner = Aligner(wgs84_dataset)
        out = aligner(wgs84_dataset_fine)
        assert out.rows == wgs84_dataset.rows
        assert out.columns == wgs84_dataset.columns

    def test_plan_epsg_matches_reference(self, wgs84_dataset):
        aligner = Aligner(wgs84_dataset)
        assert aligner.plan.target_epsg == wgs84_dataset.epsg


class TestAlignerLazy:
    """``Aligner(..., compute=False)`` returns :class:`Delayed`."""

    @requires_dask
    def test_returns_delayed(self, wgs84_dataset, wgs84_dataset_fine):
        aligner = Aligner(wgs84_dataset)
        result = aligner(wgs84_dataset_fine, compute=False)
        assert isinstance(result, Delayed)

    @requires_dask
    def test_delayed_compute_matches_eager(self, wgs84_dataset, wgs84_dataset_fine):
        aligner = Aligner(wgs84_dataset)
        eager = aligner(wgs84_dataset_fine)
        lazy = aligner(wgs84_dataset_fine, compute=False).compute()
        assert lazy.rows == eager.rows


class TestImportErrorWithoutDask:
    """Lazy path without dask raises actionable ``ImportError``."""

    def test_reprojector_lazy_raises(self, wgs84_dataset, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dask":
                raise ImportError("no dask")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        op = Reprojector(target_epsg=3857)
        with pytest.raises(ImportError, match="pyramids-gis\\[lazy\\]"):
            op(wgs84_dataset, compute=False)
