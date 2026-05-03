"""Picklable raster-metadata dataclass shared by lazy collection paths.

frozen dataclass wrapping the geo + dtype + nodata info that
:class:`~pyramids.dataset.DatasetCollection` needs to know about each
timestep, without holding a live :class:`gdal.Dataset` handle. The
DatasetCollection lazy path reads per-file data through a
:class:`~pyramids.base._file_manager.CachingFileManager` and only
needs the metadata at construction time.

We deliberately avoid :class:`odc.geo.GeoBox` + :class:`affine.Affine`
here — the geotransform is stored as a plain tuple, and :class:`pyproj.CRS`
comes free via geopandas. No new optional dependency is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from pyproj import CRS

from pyramids.base._utils import gdal_to_numpy_dtype

if TYPE_CHECKING:
    from pyramids.dataset import Dataset


GeoTransform = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class RasterMeta:
    """Immutable picklable snapshot of a raster's geobox + dtype info.

    Used by :class:`~pyramids.dataset.DatasetCollection` to cache
    per-file metadata without storing a live :class:`gdal.Dataset`
    handle. All fields are primitives or pyproj / affine objects so
    the whole dataclass pickles cleanly — safe for
    :mod:`dask.distributed` round-trips.

    Attributes:
        rows: Number of rows.
        columns: Number of columns.
        band_count: Number of raster bands.
        dtype: numpy dtype string (`"float32"`, `"int16"`,...).
        transform: GDAL-style geotransform tuple
            `(top_left_x, pixel_w, row_skew, top_left_y, col_skew,
            pixel_h)`. Stored as a plain tuple so the dataclass
            pickles cleanly without an `affine` dependency.
        crs: :class:`pyproj.CRS` for the dataset. Pickles via its WKT.
        nodata: Per-band nodata tuple. `None` entries mean the
            band has no nodata sentinel.
        block_size: Per-band `(block_width, block_height)` tuple
            captured at construction — reused as the default dask
            chunk shape in lazy read paths.
        band_names: Optional per-band names.

    Examples:
        - Construct manually and inspect the geobox:
            ```python
            >>> from pyproj import CRS
            >>> from pyramids.base._raster_meta import RasterMeta
            >>> meta = RasterMeta(
            ... rows=10, columns=12, band_count=1, dtype="float32",
            ... transform=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0),
            ... crs=CRS.from_epsg(4326),
            ...)
            >>> meta.epsg
            4326
            >>> meta.shape
            (1, 10, 12)
            >>> meta.cell_size
            1.0

            ```
    """

    rows: int
    columns: int
    band_count: int
    dtype: str
    transform: GeoTransform
    crs: CRS
    nodata: tuple[float | None, ...] = field(default_factory=tuple)
    block_size: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    band_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape `(band_count, rows, columns)`."""
        return (self.band_count, self.rows, self.columns)

    @property
    def epsg(self) -> int | None:
        """EPSG code if the CRS has one; else `None`."""
        try:
            return self.crs.to_epsg()
        except Exception:  # pragma: no cover - defensive
            return None

    @property
    def cell_size(self) -> float:
        """Absolute x-direction pixel size."""
        return abs(self.transform[1])

    @property
    def geotransform(self) -> GeoTransform:
        """GDAL-style geotransform tuple (alias of :attr:`transform`)."""
        return self.transform

    @classmethod
    def from_dataset(cls, ds: Dataset) -> RasterMeta:
        """Snapshot metadata from a live :class:`Dataset`.

        Args:
            ds: The source :class:`~pyramids.dataset.Dataset`.

        Returns:
            RasterMeta: Immutable copy of `ds`'s geobox + dtype + nodata.
        """
        transform = tuple(float(v) for v in ds.geotransform)
        crs = CRS.from_epsg(int(ds.epsg)) if ds.epsg else CRS.from_wkt(ds.crs)
        nodata_raw = tuple(ds.no_data_value) if ds.no_data_value else ()
        nodata = tuple(None if v is None else float(v) for v in nodata_raw)
        block_size = tuple(tuple(bs) for bs in ds._block_size)
        band_names = tuple(ds.band_names or ())
        first_dtype = ds.numpy_dtype[0] if ds.numpy_dtype else None
        if first_dtype is None:
            # derive from GDAL band dtype rather than hardcoding
            # float64 — otherwise a Dataset with an int16 band would
            # get a bogus float64 dtype metadata on the RasterMeta and
            # downstream dask graphs produce wrong-dtype arrays.
            band_type = ds.raster.GetRasterBand(1).DataType
            dtype = str(gdal_to_numpy_dtype(band_type))
        else:
            dtype = str(np.dtype(first_dtype))
        return cls(
            rows=int(ds.rows),
            columns=int(ds.columns),
            band_count=int(ds.band_count),
            dtype=dtype,
            transform=transform,
            crs=crs,
            nodata=nodata,
            block_size=block_size,
            band_names=band_names,
        )


__all__ = ["RasterMeta"]
