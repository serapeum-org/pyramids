"""Tests for :meth:`pyramids.dataset.Dataset.map_blocks` lazy dispatch.

when `chunks=` is given, route through
:func:`dask.array.map_blocks` and return a :class:`dask.array.Array`.
The default path (`chunks=None`) preserves today's eager tile-by-tile
behavior unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask
from pyramids.dataset import Dataset

try:
    import_dask("dask not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK = False
else:
    HAS_DASK = True

pytestmark = pytest.mark.lazy


requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


@pytest.fixture
def square_dataset(tmp_path) -> Dataset:
    """Create a 10×10 single-band float32 Dataset anchored on disk."""
    path = str(tmp_path / "grid.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, 10, 10, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 1.0, 0.0, 10.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.arange(100, dtype=np.float32).reshape(10, 10))
    ds.FlushCache()
    ds = None
    return Dataset.read_file(path)


class TestEagerPathUnchanged:
    """`chunks=None` keeps the existing behavior."""

    def test_default_returns_dataset(self, square_dataset):
        result = square_dataset.map_blocks(lambda a: a * 2, tile_size=5)
        assert isinstance(result, Dataset)

    def test_default_applies_function_eagerly(self, square_dataset):
        result = square_dataset.map_blocks(lambda a: a + 100, tile_size=5)
        arr = result.read_array()
        expected = square_dataset.read_array() + 100
        np.testing.assert_array_equal(arr, expected)


class TestLazyDispatch:
    """`chunks=<spec>` returns a dask.array.Array."""

    @requires_dask
    def test_chunks_auto_returns_dask(self, square_dataset):
        result = square_dataset.map_blocks(
            lambda a: a * 2,
            chunks="auto",
            band=0,
        )
        assert hasattr(result, "dask")

    @requires_dask
    def test_lazy_compute_matches_eager(self, square_dataset):
        lazy = square_dataset.map_blocks(
            lambda a: a * 2,
            chunks=(5, 5),
            band=0,
        )
        eager = square_dataset.map_blocks(
            lambda a: a * 2,
            tile_size=5,
            band=0,
        )
        np.testing.assert_array_equal(lazy.compute(), eager.read_array())

    @requires_dask
    def test_lazy_dtype_kwarg_propagates(self, square_dataset):
        result = square_dataset.map_blocks(
            lambda a: a.astype(np.int32),
            chunks=(5, 5),
            band=0,
            dtype=np.int32,
        )
        assert result.dtype == np.int32

    @requires_dask
    def test_lazy_chained_map_blocks(self, square_dataset):
        step1 = square_dataset.map_blocks(lambda a: a + 1, chunks=(5, 5), band=0)
        step2 = step1.map_blocks(lambda a: a * 2)  # dask.array.map_blocks
        result = step2.compute()
        expected = (
            (square_dataset.read_array()[0] + 1) * 2
            if (square_dataset.band_count > 1)
            else (square_dataset.read_array() + 1) * 2
        )
        np.testing.assert_array_equal(result, expected)

    @requires_dask
    def test_drop_axis_kwarg_forwarded(self, square_dataset):
        """`drop_axis=` is forwarded to :func:`dask.array.map_blocks`.

        Test scenario:
            A block function that reduces along an axis (`sum`) must
            produce a lazy array of lower rank when the user passes
            `drop_axis=0`. This covers the `kwargs["drop_axis"]=...`
            assignment in the lazy branch of :meth:`map_blocks`.
        """
        result = square_dataset.map_blocks(
            lambda a: a.sum(axis=0),
            chunks=(10, 10),
            band=0,
            drop_axis=0,
            dtype=np.float32,
        )
        assert hasattr(
            result, "dask"
        ), "Expected a dask.array.Array when chunks= is provided"
        assert (
            result.ndim == 1
        ), f"drop_axis=0 must reduce rank by 1; got ndim={result.ndim}"
        np.testing.assert_array_equal(
            result.compute(),
            square_dataset.read_array().sum(axis=0),
        )

    @requires_dask
    def test_new_axis_kwarg_forwarded(self, square_dataset):
        """`new_axis=` is forwarded to :func:`dask.array.map_blocks`.

        Test scenario:
            A block function that adds a leading axis (`np.expand_dims`)
            returns an array of higher rank. The lazy-path wiring must
            forward `new_axis=0` so the dask graph knows about the
            extra dimension. This covers the `kwargs["new_axis"]=...`
            assignment.
        """
        result = square_dataset.map_blocks(
            lambda a: np.expand_dims(a, axis=0),
            chunks=(10, 10),
            band=0,
            new_axis=0,
            dtype=np.float32,
        )
        assert hasattr(
            result, "dask"
        ), "Expected a dask.array.Array when chunks= is provided"
        assert (
            result.ndim == 3
        ), f"new_axis=0 must increase rank by 1; got ndim={result.ndim}"


class TestImportErrorWithoutDask:
    """`chunks=` without dask raises actionable `ImportError`."""

    def test_raises_without_dask(self, square_dataset, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("dask"):
                raise ImportError("no dask")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[lazy\\]"):
            square_dataset.map_blocks(
                lambda a: a,
                chunks="auto",
                band=0,
            )
