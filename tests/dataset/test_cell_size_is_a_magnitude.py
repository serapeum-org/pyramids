"""`cell_size` means one thing, and that thing is never negative.

Five places derived the scalar cell size from a geotransform's x term, and they
disagreed on whether to take its magnitude. `RasterMeta` and the netCDF flip
paths did; the base `AbstractDataset` constructor and two netCDF re-georeference
paths did not. On the ordinary north-up grid the x term is positive and nothing
showed, but on a west-flipped one -- which the netCDF reader itself constructs,
stamping a negative pixel width -- `Dataset.cell_size` came back negative while
`RasterMeta.cell_size` for the same raster came back positive.

A negative size is wrong for every consumer: an ASCII grid's `cellsize` header,
the divisor in a slope or aspect gradient, the value `GeoReference` re-expands
into a transform, the comparison that decides two rasters share a grid, and
`grid_size`, which refuses a non-positive resolution outright.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._grid import grid_size
from pyramids.base._raster_meta import RasterMeta
from pyramids.dataset import Dataset
from pyramids.dataset.transform import GeoTransform

pytestmark = pytest.mark.core

NORTH_UP = (0.0, 30.0, 0.0, 180.0, 0.0, -30.0)
WEST_FLIPPED = (240.0, -30.0, 0.0, 180.0, 0.0, -30.0)
SOUTH_UP = (0.0, 30.0, 0.0, 0.0, 0.0, 30.0)


def _raster(geotransform) -> Dataset:
    """A small in-memory raster carrying `geotransform`."""
    ds = gdal.GetDriverByName("MEM").Create("", 8, 6, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.ones((6, 8), dtype=np.float32))
    return Dataset(ds)


class TestGeoTransformCellSize:
    """The one derivation the rest now read."""

    @pytest.mark.parametrize(
        "geotransform",
        [NORTH_UP, WEST_FLIPPED, SOUTH_UP],
        ids=["north-up", "west-flipped", "south-up"],
    )
    def test_it_is_the_magnitude_whatever_the_grid_direction(self, geotransform):
        """Direction lives in the transform; a size is a size.

        Args:
            geotransform: The GDAL 6-tuple under test.

        Test scenario:
            All three grids have 30-unit cells and differ only in which way
            their axes run. Each must report the same cell size.
        """
        assert GeoTransform(*geotransform).cell_size == 30.0

    def test_it_is_never_negative(self):
        """The property the flipped case violated.

        Test scenario:
            A negative pixel width is legitimate -- it means the columns run
            west -- but it is not a size, and every consumer of this scalar
            treats it as one.
        """
        assert GeoTransform(*WEST_FLIPPED).pixel_width < 0
        assert GeoTransform(*WEST_FLIPPED).cell_size > 0


class TestEveryReaderAgrees:
    """The five derivations, pinned against each other."""

    @pytest.mark.parametrize(
        "geotransform",
        [NORTH_UP, WEST_FLIPPED, SOUTH_UP],
        ids=["north-up", "west-flipped", "south-up"],
    )
    def test_the_dataset_and_its_metadata_snapshot_match(self, geotransform):
        """The regression: these two disagreed on a west-flipped grid.

        Args:
            geotransform: The GDAL 6-tuple written to the raster.

        Test scenario:
            `RasterMeta.from_dataset(ds).cell_size` is documented as the
            absolute pixel size and always was. `ds.cell_size` was the raw
            signed term, so the two returned -30 and 30 for one raster.
        """
        dataset = _raster(geotransform)

        assert dataset.cell_size == RasterMeta.from_dataset(dataset).cell_size

    @pytest.mark.parametrize(
        "geotransform",
        [NORTH_UP, WEST_FLIPPED, SOUTH_UP],
        ids=["north-up", "west-flipped", "south-up"],
    )
    def test_the_dataset_agrees_with_the_transform_it_carries(self, geotransform):
        """The cached scalar must not drift from its own geotransform.

        Args:
            geotransform: The GDAL 6-tuple written to the raster.

        Test scenario:
            `_cell_size` is a cache filled at construction. It has to hold what
            deriving it from `ds.geotransform` would give, or the two answers a
            caller can reach differ.
        """
        dataset = _raster(geotransform)

        assert dataset.cell_size == GeoTransform(*dataset.geotransform).cell_size

    def test_the_ordinary_grid_is_unchanged(self):
        """The common case must not have moved while fixing the rare one.

        Test scenario:
            Nearly every raster is north-up with a positive x term, where the
            old code was already right. Its answer is pinned exactly.
        """
        assert _raster(NORTH_UP).cell_size == 30.0


class TestWhyTheSignMattered:
    """The consumers a negative cell size actually broke."""

    def test_a_flipped_raster_can_now_size_a_grid(self):
        """`grid_size` refuses a non-positive resolution.

        Test scenario:
            Feeding a flipped raster's cell size into the shared sizing helper
            used to raise "resolution must be strictly positive" -- a failure
            with nothing in its message about the raster being mirrored.
        """
        cell = _raster(WEST_FLIPPED).cell_size

        assert grid_size(240.0, 180.0, (cell, cell), max_px=None) == (8, 6)

    def test_two_rasters_of_one_resolution_compare_equal(self):
        """Alignment checks compare these scalars directly.

        Test scenario:
            A mirrored raster and a north-up one with 30-unit cells have the
            same resolution. While one reported -30, the alignment comparison
            in `Spatial` treated them as differently-sized grids.
        """
        assert _raster(WEST_FLIPPED).cell_size == _raster(NORTH_UP).cell_size
