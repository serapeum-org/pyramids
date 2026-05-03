"""Tests for :mod:`pyramids.dataset.ops._focal` (DASK-26)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

try:
    import_dask("dask not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK = False
else:
    HAS_DASK = True
requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


@pytest.fixture
def constant_raster(tmp_path):
    """5×5 raster of constant value — focal_mean must equal that value."""
    arr = np.full((5, 5), 7.0, dtype=np.float32)
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=(0.0, 5.0),
        cell_size=1.0,
        epsg=4326,
    )
    path = str(tmp_path / "const.tif")
    ds.to_file(path)
    return Dataset.read_file(path)


@pytest.fixture
def ramp_raster(tmp_path):
    """5×5 ramp along the x-axis so slope is non-zero."""
    arr = np.tile(np.arange(5, dtype=np.float32), (5, 1))
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=(0.0, 5.0),
        cell_size=1.0,
        epsg=4326,
    )
    path = str(tmp_path / "ramp.tif")
    ds.to_file(path)
    return Dataset.read_file(path)


class TestFocalMean:
    def test_constant_yields_constant(self, constant_raster):
        out = constant_raster.focal_mean(radius=1)
        assert np.allclose(out, 7.0)

    def test_returns_same_shape(self, constant_raster):
        out = constant_raster.focal_mean(radius=1)
        assert out.shape == (5, 5)

    @requires_dask
    def test_lazy_matches_eager(self, constant_raster):
        eager = constant_raster.focal_mean(radius=1)
        lazy = constant_raster.focal_mean(radius=1, chunks="auto").compute()
        assert np.allclose(eager, lazy)


class TestFocalStd:
    def test_constant_is_zero(self, constant_raster):
        out = constant_raster.focal_std(radius=1)
        assert np.allclose(out, 0.0, atol=1e-6)

    def test_ramp_has_nonzero(self, ramp_raster):
        out = ramp_raster.focal_std(radius=1)
        assert float(out.mean()) > 0.0


class TestFocalApply:
    def test_identity_func(self, constant_raster):
        out = constant_raster.focal_apply(lambda w: w[4], radius=1)
        assert np.allclose(out, 7.0)


class TestSlope:
    def test_constant_has_zero_slope(self, constant_raster):
        out = constant_raster.slope()
        assert np.allclose(out, 0.0, atol=1e-6)

    def test_ramp_has_positive_slope(self, ramp_raster):
        out = ramp_raster.slope()
        inside = out[1:-1, 1:-1]
        assert float(inside.mean()) > 0.0

    @requires_dask
    def test_lazy_slope_equals_eager_interior(self, ramp_raster):
        """Interior cells agree; edge cells can differ due to boundary.

        ``map_overlap`` uses the reflect-halo inside each chunk,
        whereas ``np.gradient`` uses forward/backward differences at
        the outermost rows/cols. Compare the interior only.
        """
        eager = ramp_raster.slope()
        lazy = ramp_raster.slope(chunks="auto").compute()
        assert np.allclose(eager[1:-1, 1:-1], lazy[1:-1, 1:-1])


class TestAspect:
    def test_returns_degrees(self, ramp_raster):
        out = ramp_raster.aspect()
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 360.0


class TestHillshade:
    def test_values_in_byte_range(self, ramp_raster):
        out = ramp_raster.hillshade()
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    @requires_dask
    def test_lazy_matches_eager_interior(self, ramp_raster):
        """Same boundary caveat as slope — compare interior cells."""
        eager = ramp_raster.hillshade()
        lazy = ramp_raster.hillshade(chunks="auto").compute()
        assert np.allclose(eager[1:-1, 1:-1], lazy[1:-1, 1:-1])


class TestImportError:
    def test_chunks_without_dask_raises(self, constant_raster, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("dask"):
                raise ImportError("no dask")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[lazy\\]"):
            constant_raster.focal_mean(radius=1, chunks="auto")
