"""Tests for :mod:`pyramids.dataset.ops._focal` (DASK-26)."""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


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

    @pytest.mark.lazy
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

    @pytest.mark.lazy
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

    @pytest.mark.lazy
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


class TestFocalNoDataHandling:
    """Focal kernels must not let a no-data sentinel leak into real cells."""

    @staticmethod
    def _write(array: np.ndarray, path, no_data_value: float = -9999.0) -> Dataset:
        """Write `array` to a GeoTIFF carrying `no_data_value` and open it.

        Args:
            array: Pixel values.
            path: Destination path.
            no_data_value: Sentinel to register on the band.

        Returns:
            Dataset: The dataset read back from disk.
        """
        rows, cols = array.shape
        raster = gdal.GetDriverByName("GTiff").Create(
            str(path), cols, rows, 1, gdal.GDT_Float32
        )
        raster.SetGeoTransform((0.0, 1.0, 0.0, float(rows), 0.0, -1.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        raster.SetProjection(srs.ExportToWkt())
        band = raster.GetRasterBand(1)
        band.WriteArray(array.astype("float32"))
        band.SetNoDataValue(no_data_value)
        raster.FlushCache()
        raster = None
        return Dataset.read_file(str(path))

    @pytest.fixture(scope="function")
    def planar_dem(self) -> np.ndarray:
        """A 3x3 DEM with a constant gradient, so every slope is equal.

        Returns:
            numpy.ndarray: The elevation grid.
        """
        return np.array(
            [[10.0, 11.0, 12.0], [10.5, 11.5, 12.5], [11.0, 12.0, 13.0]],
            dtype="float32",
        )

    def test_sentinel_does_not_contaminate_neighbours(self, planar_dem, tmp_path):
        """A `-9999` cell must not change the slope of cells away from it.

        Test scenario:
            On a constant-gradient DEM every cell has the same slope. Punching
            a `-9999` sentinel into the centre previously fed that value to
            `np.gradient`, driving all four orthogonal neighbours to a
            saturated 90 degrees while looking like plausible terrain. The four
            corners, which the sentinel's window does not reach, must keep
            their original value.
        """
        clean = self._write(planar_dem, tmp_path / "clean.tif")
        dirty_array = planar_dem.copy()
        dirty_array[1, 1] = -9999.0
        dirty = self._write(dirty_array, tmp_path / "dirty.tif")

        clean_slope = np.asarray(clean.slope())
        dirty_slope = np.asarray(dirty.slope())

        for corner in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            assert clean_slope[corner] == pytest.approx(dirty_slope[corner]), (
                f"corner {corner} changed from {clean_slope[corner]} to "
                f"{dirty_slope[corner]} because of a distant no-data cell"
            )

    def test_sentinel_cell_and_its_window_become_no_data(self, planar_dem, tmp_path):
        """The no-data cell and the cells it invalidates carry the sentinel.

        Test scenario:
            A cell with no elevation has no slope, and neither do the cells
            whose window overlaps it. `np.gradient` uses a centred difference,
            so it happily computes a value *at* the no-data cell from its
            neighbours — blanking the input alone is not enough, the cell must
            be masked explicitly.
        """
        dirty_array = planar_dem.copy()
        dirty_array[1, 1] = -9999.0
        dataset = self._write(dirty_array, tmp_path / "dirty.tif")
        slope = np.asarray(dataset.slope())

        assert slope[1, 1] == -9999.0, (
            f"the no-data cell must stay no-data, got {slope[1, 1]}"
        )
        for neighbour in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            assert slope[neighbour] == -9999.0, (
                f"neighbour {neighbour} of a no-data cell must be no-data, got "
                f"{slope[neighbour]}"
            )

    def test_raster_without_the_sentinel_is_untouched(self, tmp_path):
        """A band where no cell is no-data keeps a fully finite result.

        Test scenario:
            Guards the opposite direction: the masking must not start blanking
            cells on rasters that never contained the sentinel.
        """
        array = np.arange(25, dtype="float32").reshape(5, 5)
        dataset = self._write(array, tmp_path / "plain.tif")
        slope = np.asarray(dataset.slope())
        assert np.isfinite(slope).all(), (
            f"a raster with no no-data cells must be all finite, got {slope}"
        )

    def test_focal_mean_respects_no_data(self, planar_dem, tmp_path):
        """The box filter masks no-data too, not just the DEM derivatives.

        Test scenario:
            `focal_mean` averages a window, so a `-9999` inside it drags the
            mean down by roughly the sentinel divided by the window size. All
            six kernels share one dispatcher, so covering one filter alongside
            `slope` shows the guard is applied there rather than per kernel.
        """
        dirty_array = planar_dem.copy()
        dirty_array[1, 1] = -9999.0
        dataset = self._write(dirty_array, tmp_path / "dirty.tif")
        smoothed = np.asarray(dataset.focal_mean(radius=1))
        assert smoothed[1, 1] == -9999.0, (
            f"the no-data cell must stay no-data, got {smoothed[1, 1]}"
        )
        finite = smoothed[smoothed != -9999.0]
        assert (finite > 0).all(), (
            f"no surviving cell may be dragged negative by the sentinel: {finite}"
        )
