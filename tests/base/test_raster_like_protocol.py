"""Tests for the :class:`pyramids.base.protocols.RasterLike` protocol.

Both :class:`pyramids.dataset.Dataset` and its
:class:`pyramids.netcdf.NetCDF` subclass implement the raster-specific
`RasterLike` structural type, so callers can write generic raster code
that accepts either without importing the concrete classes.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.base.protocols import RasterLike, SpatialObject
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

GEO = (0.0, 0.01, 0, 1.0, 0, -0.01)


@pytest.fixture
def ds() -> Dataset:
    """A small in-memory single-band Dataset in EPSG:32636."""
    src = Dataset.create(
        cell_size=1000.0,
        rows=10,
        columns=10,
        dtype="float32",
        bands=1,
        top_left_corner=(500000.0, 3410000.0),
        epsg=32636,
        no_data_value=-9999.0,
    )
    src.raster.GetRasterBand(1).WriteArray(np.ones((10, 10), dtype=np.float32))
    return src


@pytest.fixture
def nc() -> NetCDF:
    """A small in-memory NetCDF container built from a 2-D array."""
    return NetCDF.create_from_array(
        arr=np.ones((5, 8), dtype=np.float64), geo=GEO, variable_name="v"
    )


class TestRasterLikeProtocol:
    """Dataset and NetCDF satisfy RasterLike; it refines SpatialObject."""

    def test_dataset_is_raster_like(self, ds: Dataset):
        """`isinstance(ds, RasterLike)` is True at runtime."""
        assert isinstance(ds, RasterLike)

    def test_netcdf_is_raster_like(self, nc: NetCDF):
        """`isinstance(nc, RasterLike)` is True at runtime."""
        assert isinstance(nc, RasterLike)

    def test_raster_like_is_a_spatial_object(self, ds: Dataset):
        """A RasterLike is also a SpatialObject (the protocol extends it)."""
        assert isinstance(ds, SpatialObject)

    def test_exposes_raster_surface(self, ds: Dataset):
        """The raster-specific members are present and have sane values."""
        assert ds.rows == 10 and ds.columns == 10
        assert ds.band_count == 1
        assert ds.cell_size == pytest.approx(1000.0)
        assert len(ds.geotransform) == 6
        assert ds.raster is not None

    def test_generic_consumer_accepts_both(self, ds: Dataset, nc: NetCDF):
        """A function typed against RasterLike accepts both Dataset and NetCDF.

        This is the value of the protocol — raster-generic utilities without
        importing the concrete Dataset / NetCDF classes.
        """

        def cell_count(raster: RasterLike) -> int:
            return raster.rows * raster.columns

        assert cell_count(ds) == 100
        # The NetCDF container's rows/columns reflect its underlying multidim
        # handle (not a variable's dims); assert only that the RasterLike-typed
        # function accepts and runs on it.
        assert isinstance(cell_count(nc), int)


class TestRasterLikeNegative:
    """Objects missing the raster surface must NOT be RasterLike."""

    def test_plain_string_is_not_raster_like(self):
        """A string lacks every raster attribute."""
        assert not isinstance("not a raster", RasterLike)

    def test_dict_is_not_raster_like(self):
        """A dict with a stray `epsg` key still lacks the raster surface."""
        assert not isinstance({"epsg": 4326}, RasterLike)

    def test_bare_gdal_dataset_is_not_raster_like(self, ds: Dataset):
        """A raw `gdal.Dataset` handle has no `epsg`/`read_array`, so it is not RasterLike."""
        assert not isinstance(ds.raster, RasterLike)

    def test_ndarray_is_not_raster_like(self):
        """A numpy array is not a raster object."""
        assert not isinstance(np.zeros((4, 4)), RasterLike)

    def test_vector_spatial_object_is_not_raster_like(self):
        """A FeatureCollection is a SpatialObject but lacks the raster surface, so not RasterLike.

        This locks RasterLike to the raster-specific contract: refining
        SpatialObject must not accidentally widen back to accepting vectors.
        """
        fc = FeatureCollection(
            gpd.GeoDataFrame({"v": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        )
        assert isinstance(fc, SpatialObject), "fixture must be a SpatialObject"
        assert not isinstance(fc, RasterLike), "a vector SpatialObject must not be RasterLike"
