"""GeoZarr / CF geobox encoding shared by the zarr read/write paths.

pyramids writes raster geo-referencing into a Zarr store following the CF /
GeoZarr convention so the result is auto-georeferenced by rioxarray, odc-geo and
:func:`xarray.open_zarr` without pyramids in the loop:

* a scalar ``spatial_ref`` grid-mapping array holding ``crs_wkt`` + a
  space-delimited ``GeoTransform`` (and the EPSG code, or ``0`` when the CRS has
  no authority code);
* 1-D ``x`` / ``y`` coordinate arrays at pixel centres;
* a ``grid_mapping="spatial_ref"`` attribute on each data array;
* ``_ARRAY_DIMENSIONS`` on every array — the Zarr-v2 convention xarray uses to
  recover dimension names.

:func:`read_geobox` reads that layout back and also accepts the legacy pyramids
flat-attribute layout (CRS / GeoTransform stored directly on the ``data`` array),
emitting a :class:`DeprecationWarning` so stores written before this convention
keep opening.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

ZARR_SCHEMA_VERSION = "2"
GRID_MAPPING_VAR = "spatial_ref"

_LEGACY_WARNING = (
    "This Zarr store uses the legacy pyramids geobox layout (geo-referencing "
    "stored as flat attributes on the 'data' array). Re-save it with to_zarr to "
    "adopt the GeoZarr 'spatial_ref' convention; legacy reading will be removed "
    "in a future release."
)


def pixel_centre_coords(
    geotransform: tuple[float, ...], rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-D ``x`` / ``y`` pixel-centre coordinates for a north-up geotransform.

    Args:
        geotransform: GDAL 6-tuple ``(x0, dx, 0, y0, 0, dy)`` (``dy`` negative for
            a north-up raster).
        rows: Number of rows (the length of ``y``).
        cols: Number of columns (the length of ``x``).

    Returns:
        A ``(x, y)`` tuple of 1-D float64 arrays giving the centre coordinate of
        each column and row.
    """
    gt = geotransform
    x = gt[0] + (np.arange(cols, dtype="float64") + 0.5) * gt[1]
    y = gt[3] + (np.arange(rows, dtype="float64") + 0.5) * gt[5]
    return x, y


def write_geobox(
    group: Any,
    *,
    data_name: str,
    epsg: int,
    geotransform: tuple[float, ...],
    crs_wkt: str,
    rows: int,
    cols: int,
    dims: list[str],
) -> None:
    """Write the GeoZarr geobox (``spatial_ref`` + ``x`` / ``y`` coords) into ``group``.

    Adds the ``spatial_ref`` grid-mapping scalar and the 1-D ``x`` / ``y``
    coordinate arrays, tags every array (including ``data_name``) with
    ``_ARRAY_DIMENSIONS`` and points the data array at the grid mapping via its
    ``grid_mapping`` attribute. The caller is responsible for the data array
    itself and for consolidating metadata afterwards.

    Args:
        group: An open, writable :class:`zarr.hierarchy.Group`.
        data_name: Name of the already-written data array in ``group``.
        epsg: EPSG code, or ``0`` when the CRS has no authority code.
        geotransform: GDAL 6-tuple.
        crs_wkt: CRS as WKT (the authoritative CRS; preferred over ``epsg`` on read).
        rows: Raster row count.
        cols: Raster column count.
        dims: Dimension names of the data array, e.g. ``["band", "y", "x"]``.
    """
    x, y = pixel_centre_coords(geotransform, rows, cols)
    zx = group.create_dataset("x", data=x, overwrite=True)
    zx.attrs["_ARRAY_DIMENSIONS"] = ["x"]
    zy = group.create_dataset("y", data=y, overwrite=True)
    zy.attrs["_ARRAY_DIMENSIONS"] = ["y"]

    sr = group.create_dataset(
        GRID_MAPPING_VAR, data=np.array(int(epsg or 0), dtype="int64"), overwrite=True
    )
    sr.attrs.update(
        {
            "_ARRAY_DIMENSIONS": [],
            "crs_wkt": crs_wkt,
            "spatial_ref": crs_wkt,
            "GeoTransform": " ".join(str(v) for v in geotransform),
            "epsg": int(epsg or 0),
        }
    )

    data = group[data_name]
    data.attrs["_ARRAY_DIMENSIONS"] = list(dims)
    data.attrs["grid_mapping"] = GRID_MAPPING_VAR


def read_geobox(group: Any, *, data_name: str = "data") -> dict[str, Any]:
    """Recover ``{crs_wkt, geotransform, epsg, legacy}`` from a Zarr ``group``.

    Prefers the GeoZarr ``spatial_ref`` grid-mapping array when present; otherwise
    falls back to the legacy flat attributes on the data array and warns.

    Args:
        group: An open :class:`zarr.hierarchy.Group`.
        data_name: Name of the data array (used for the legacy fallback).

    Returns:
        Dict with ``crs_wkt`` (str | None), ``geotransform`` (tuple[float, ...]),
        ``epsg`` (int; ``0`` when unknown) and ``legacy`` (bool).

    Raises:
        KeyError: When no ``GeoTransform`` can be found in either layout.
    """
    if GRID_MAPPING_VAR in group:
        attrs = dict(group[GRID_MAPPING_VAR].attrs)
        legacy = False
    else:
        attrs = dict(group[data_name].attrs)
        legacy = True
        warnings.warn(_LEGACY_WARNING, DeprecationWarning, stacklevel=2)

    crs_wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
    geotransform = tuple(float(v) for v in attrs["GeoTransform"].split())
    epsg = int(attrs.get("epsg") or 0)
    return {
        "crs_wkt": crs_wkt,
        "geotransform": geotransform,
        "epsg": epsg,
        "legacy": legacy,
    }
