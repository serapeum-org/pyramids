"""The :class:`SubDataset` value object — one nested raster inside a container.

A *container* raster has no pixels of its own; its payload is a set of named
sub-rasters that GDAL calls **subdatasets**. Containers are how GDAL models
NetCDF, HDF4/HDF5, Zarr, GRIB, WMS/WMTS, and the Sentinel-1 (`SAFE`) and
Sentinel-2 (`SENTINEL2`) products. `SubDataset` is a lightweight, picklable
description of one such nested raster — its openable connection string, GDAL's
own blurb, and its position in the container's list — with :meth:`SubDataset.open`
to materialise it.

The description string is driver-specific (a Sentinel-2 entry looks nothing like
a NetCDF one), so this value object deliberately exposes it **raw** and parses
nothing; a format-specific layer parses it if it needs to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osgeo import gdal

    from pyramids.dataset.dataset import Dataset


@dataclass(frozen=True)
class SubDataset:
    """One nested raster inside a container, as GDAL reports it.

    Attributes:
        name: The fully-qualified GDAL connection string for the subdataset (for
            example ``'SENTINEL2_L2A:/path/MTD_MSIL2A.xml:60m:EPSG_32632'`` or
            ``'NETCDF:"file.nc":temperature'``). It is openable on its own.
        description: GDAL's human-readable description of the subdataset. Its
            format is driver-specific — do not parse it generically.
        index: The 0-based position of this subdataset in the container's list.

    Examples:
        - A subdataset carries the openable name, GDAL's description, and its index:
            ```python
            >>> from pyramids.dataset._subdataset import SubDataset
            >>> sd = SubDataset(name='NETCDF:"f.nc":temp', description='[2x3] temp', index=0)
            >>> sd.name
            'NETCDF:"f.nc":temp'
            >>> sd.index
            0

            ```
    """

    name: str
    description: str
    index: int

    def open(self) -> Dataset:
        """Open this subdataset as a base :class:`~pyramids.dataset.dataset.Dataset`.

        Returns:
            Dataset: The opened subdataset. This always returns a **base**
            ``Dataset``; to preserve the parent container's concrete class (e.g.
            keep a ``NetCDF`` a ``NetCDF``), call
            :meth:`~pyramids.dataset.dataset.Dataset.open_subdataset` on the parent
            instead.
        """
        # Local import: breaks the _subdataset -> dataset -> abstract_dataset ->
        # _subdataset cycle (abstract_dataset imports SubDataset at module load).
        from pyramids.dataset.dataset import Dataset

        # warn_on_container=False: opening a subdataset by its value object is a
        # deliberate drill-in, so a nested-container warning here would be noise.
        return Dataset.read_file(self.name, warn_on_container=False)


def subdatasets_of(raster: gdal.Dataset) -> list[SubDataset]:
    """Build the :class:`SubDataset` list for a raw GDAL dataset handle.

    The single place the ``SUBDATASETS`` domain is turned into
    :class:`SubDataset` value objects. Both
    :attr:`~pyramids.dataset.abstract_dataset.RasterBase.subdatasets` and the
    WMTS layer-hint helper in :mod:`pyramids.dataset._wms` build on it, so the
    enumeration lives in exactly one spot.

    Args:
        raster: An open ``gdal.Dataset``. A plain raster (no subdatasets) yields an
            empty list.

    Returns:
        list[SubDataset]: One :class:`SubDataset` per nested raster, in GDAL's
        order; ``[]`` when the handle has none.

    Examples:
        - Enumerate a NetCDF container's four subdatasets from a raw GDAL handle:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset._subdataset import subdatasets_of
            >>> handle = gdal.Open("tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc")
            >>> subs = subdatasets_of(handle)
            >>> len(subs)
            4
            >>> subs[0].index, subs[0].name.endswith(":Band1")
            (0, True)

            ```
        - A plain single-band raster has no subdatasets:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset._subdataset import subdatasets_of
            >>> subdatasets_of(gdal.Open("tests/data/geotiff/coello-without-color-table.tif"))
            []

            ```
    """
    return [
        SubDataset(name, description, i)
        for i, (name, description) in enumerate(raster.GetSubDatasets())
    ]
