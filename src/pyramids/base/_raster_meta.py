"""Picklable raster-metadata dataclass shared by lazy collection paths.

frozen dataclass wrapping the geo + dtype + nodata info that
:class:`~pyramids.dataset.DatasetCollection` needs to know about each
timestep, without holding a live :class:`gdal.Dataset` handle. The
DatasetCollection lazy path reads per-file data through a
:class:`~pyramids.base._file_manager.CachingFileManager` and only
needs the metadata at construction time.

The geotransform is stored as a plain 6-tuple and the CRS as a
:class:`pyproj.CRS`, so no extra geometry/affine dependency is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from pyproj import CRS

from pyramids.base._utils import gdal_to_numpy_dtype
from pyramids.base.crs import crs_from_user_input, crs_spec

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
        crs: :class:`pyproj.CRS` for the dataset, or `None` when it has no
            CRS at all (an ASCII grid, say). Pickles via its WKT.
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
            ... )
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
    crs: CRS | None
    nodata: tuple[float | None, ...] = field(default_factory=tuple)
    block_size: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    band_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape `(band_count, rows, columns)`."""
        return (self.band_count, self.rows, self.columns)

    @property
    def epsg(self) -> int | None:
        """EPSG code if there is a CRS and it has one; else `None`.

        `crs` is `None` for a genuinely ungeoreferenced raster (an ASCII grid,
        say), which is a normal state rather than a defensive edge case — hence
        the explicit check rather than relying on the handler below.
        """
        code = None
        if self.crs is not None:
            try:
                code = self.crs.to_epsg()
            except (AttributeError, ValueError):
                code = None
        return code

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
        transform = cast(
            tuple[float, float, float, float, float, float],
            tuple(float(v) for v in ds.geotransform),
        )
        # `ds.crs` is empty for a genuinely ungeoreferenced raster (an ASCII
        # grid, say). `CRS.from_wkt("")` raises, so report no CRS instead of
        # inventing one (ARC-26); consumers already handle `crs is None`.
        # `crs_spec` picks the code or the WKT, whichever the CRS libraries can
        # actually resolve, and `crs_from_user_input` heals a code that only GDAL's
        # PROJ database carries -- otherwise a raster in such a CRS cannot be
        # described at all (issue #943).
        spec = crs_spec(ds.epsg, ds.crs)
        crs = None if spec is None else crs_from_user_input(spec)
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
