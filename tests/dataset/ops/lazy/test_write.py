"""Tests for :meth:`pyramids.dataset.Dataset.to_file` ``compute=False``.

DASK-7: defer the write via ``dask.delayed`` when ``compute=False``;
add :meth:`Dataset.to_raster` alias for rioxarray API parity.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask
from pyramids.dataset import Dataset

pytestmark = pytest.mark.lazy

try:
    import_dask("dask not installed")
    import dask
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK = False
else:
    HAS_DASK = True
requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


@pytest.fixture
def tiny_dataset(tmp_path) -> Dataset:
    """Create a tiny Dataset anchored on disk for pickle/delayed tests."""
    path = str(tmp_path / "src.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, 4, 3, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.arange(12, dtype=np.float32).reshape(3, 4))
    ds.FlushCache()
    ds = None
    return Dataset.read_file(path)


class TestComputeTrueUnchanged:
    """``compute=True`` preserves today's synchronous behavior."""

    def test_compute_true_writes_file(self, tiny_dataset, tmp_path):
        out = tmp_path / "out.tif"
        result = tiny_dataset.to_file(str(out))
        assert result is None
        assert out.exists()

    def test_default_compute_is_true(self, tiny_dataset, tmp_path):
        out = tmp_path / "default.tif"
        result = tiny_dataset.to_file(str(out))
        assert result is None
        assert out.exists()


class TestComputeFalseReturnsDelayed:
    """``compute=False`` returns a :class:`dask.delayed.Delayed`."""

    @requires_dask
    def test_returns_delayed(self, tiny_dataset, tmp_path):
        out = tmp_path / "delayed.tif"
        delayed = tiny_dataset.to_file(str(out), compute=False)
        from dask.delayed import Delayed

        assert isinstance(delayed, Delayed)
        assert not out.exists()

    @requires_dask
    def test_delayed_compute_writes_file(self, tiny_dataset, tmp_path):
        out = tmp_path / "after_compute.tif"
        delayed = tiny_dataset.to_file(str(out), compute=False)
        delayed.compute()
        assert out.exists()

    @requires_dask
    def test_delayed_output_readable(self, tiny_dataset, tmp_path):
        out = tmp_path / "readable.tif"
        tiny_dataset.to_file(str(out), compute=False).compute()
        re_read = Dataset.read_file(str(out))
        arr = re_read.read_array()
        np.testing.assert_array_equal(arr, tiny_dataset.read_array())


class TestToRasterAlias:
    """``to_raster`` is a thin forward of ``to_file``."""

    def test_to_raster_writes_file(self, tiny_dataset, tmp_path):
        out = tmp_path / "via_alias.tif"
        tiny_dataset.to_raster(str(out))
        assert out.exists()

    @requires_dask
    def test_to_raster_compute_false(self, tiny_dataset, tmp_path):
        out = tmp_path / "via_alias_delayed.tif"
        delayed = tiny_dataset.to_raster(str(out), compute=False)
        from dask.delayed import Delayed

        assert isinstance(delayed, Delayed)
        delayed.compute()
        assert out.exists()


class TestEarlyPicklingError:
    """L2: MEM-driver Dataset raises PicklingError at schedule time, not compute."""

    @requires_dask
    def test_mem_dataset_raises_at_schedule(self, tmp_path):
        import pickle

        import numpy as np

        arr = np.zeros((3, 4), dtype=np.float32)
        mem_ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        with pytest.raises(pickle.PicklingError, match="on-disk"):
            mem_ds.to_file(str(tmp_path / "nope.tif"), compute=False)


class TestImportError:
    """``compute=False`` without dask raises actionable ``ImportError``."""

    def test_raises_without_dask(self, tiny_dataset, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dask":
                raise ImportError("no dask available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[lazy\\]"):
            tiny_dataset.to_file(str(tmp_path / "nope.tif"), compute=False)
