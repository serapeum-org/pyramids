"""GeoZarr / CF geobox encoding shared by the zarr read/write paths.

pyramids writes raster geo-referencing into a Zarr store following the CF /
GeoZarr convention so the result is auto-georeferenced by rioxarray, odc-geo and
:func:`xarray.open_zarr` without pyramids in the loop:

* a scalar ``spatial_ref`` grid-mapping array holding ``crs_wkt`` + a
  space-delimited ``GeoTransform`` (and the EPSG code, or ``0`` when the CRS has
  no authority code);
* 1-D ``x`` / ``y`` coordinate arrays at pixel centres;
* a ``grid_mapping="spatial_ref"`` attribute on each data array;
* ``_ARRAY_DIMENSIONS`` attr on every array (the Zarr-v2 convention xarray uses
  to recover dimension names) **plus** the Zarr-v3-native ``dimension_names``
  field on the array metadata, so consumers on either version see the dims.

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

    Examples:
        - Centres sit half a pixel in from the origin; ``y`` descends for a
          north-up transform:
            ```python
            >>> x, y = pixel_centre_coords((0.0, 1.0, 0.0, 4.0, 0.0, -1.0), rows=4, cols=5)
            >>> x.tolist()
            [0.5, 1.5, 2.5, 3.5, 4.5]
            >>> y.tolist()
            [3.5, 2.5, 1.5, 0.5]

            ```
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

    def _put(name: str, values: np.ndarray, var_dims: list[str]):
        values = np.asarray(values)
        arr = group.create_array(
            name,
            shape=values.shape,
            dtype=values.dtype,
            dimension_names=tuple(var_dims),
            overwrite=True,
        )
        arr[...] = values
        # Keep the Zarr-v2 `_ARRAY_DIMENSIONS` attr alongside v3 `dimension_names`
        # so readers on either convention recover the dim names.
        arr.attrs["_ARRAY_DIMENSIONS"] = list(var_dims)
        return arr

    x, y = pixel_centre_coords(geotransform, rows, cols)
    _put("x", x, ["x"])
    _put("y", y, ["y"])
    sr = _put(GRID_MAPPING_VAR, np.array(int(epsg or 0), dtype="int64"), [])
    sr.attrs.update(
        {
            "crs_wkt": crs_wkt,
            "spatial_ref": crs_wkt,
            "GeoTransform": " ".join(str(v) for v in geotransform),
            "epsg": int(epsg or 0),
        }
    )

    data = group[data_name]
    data.attrs["_ARRAY_DIMENSIONS"] = list(dims)
    data.attrs["grid_mapping"] = GRID_MAPPING_VAR


def finalize_zarr_metadata(
    resolved_store: Any,
    *,
    root_attrs: dict[str, Any],
    data_attrs: dict[str, Any],
    epsg: int,
    geotransform: tuple[float, ...],
    crs_wkt: str,
    rows: int,
    cols: int,
    dims: list[str],
) -> None:
    """Write root + data attrs and the GeoZarr geobox, then consolidate metadata.

    Shared finalize step for both the Dataset and DatasetCollection writers: the
    callers supply their own ``root_attrs`` / ``data_attrs`` payloads and the
    data-array ``dims``; the open / attr-set / :func:`write_geobox` /
    ``consolidate_metadata`` boilerplate lives here once.

    Args:
        resolved_store: A zarr-compatible mapping / store, opened in append mode.
        root_attrs: Attributes to set on the root group.
        data_attrs: Attributes to set on the ``data`` array.
        epsg: EPSG code (``0`` when the CRS has no authority code).
        geotransform: GDAL 6-tuple.
        crs_wkt: CRS as WKT.
        rows: Raster row count.
        cols: Raster column count.
        dims: Dimension names of the ``data`` array.
    """
    import zarr

    root = zarr.open_group(resolved_store, mode="a")
    root.attrs.update(root_attrs)
    root["data"].attrs.update(data_attrs)
    write_geobox(
        root,
        data_name="data",
        epsg=epsg,
        geotransform=geotransform,
        crs_wkt=crs_wkt,
        rows=rows,
        cols=cols,
        dims=dims,
    )
    # Consolidated metadata is an explicit speedup for many-array group listing
    # (used by every from_zarr call to read attrs/shape without scanning chunks).
    # zarr v3 emits a ZarrUserWarning that this isn't yet in the spec; suppress
    # it locally so it doesn't leak into every user's stderr.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Consolidated metadata is currently not part"
        )
        zarr.consolidate_metadata(resolved_store)


_NON_DATA_ARRAYS = {
    "x",
    "y",
    "lon",
    "lat",
    "longitude",
    "latitude",
    "time",
    "band",
    GRID_MAPPING_VAR,
    "crs",
}


def normalize_compressors(compressor: Any) -> dict[str, Any]:
    """Map a user ``compressor=`` argument to zarr-v3 ``create_array`` kwargs.

    ``"auto"`` keeps zarr's default codec (no kwarg); ``None`` writes an
    uncompressed array; a single codec or an iterable of codecs becomes the v3
    ``compressors=`` list (v3 expects an iterable of codecs).
    """
    if compressor == "auto":
        return {}
    if compressor is None:
        return {"compressors": None}
    if isinstance(compressor, (list, tuple)):
        return {"compressors": list(compressor)}
    return {"compressors": [compressor]}


def detect_data_var(group: Any) -> str:
    """Pick the primary data array name in a (possibly foreign) GeoZarr group.

    Prefers ``"data"`` (pyramids' own name); otherwise an array carrying a
    ``grid_mapping`` attribute; otherwise the highest-dimension array that is
    not an obvious coordinate (``x``/``y``/``spatial_ref``/…).

    Raises:
        KeyError: When no candidate data array can be found.
    """
    if "data" in group:
        return "data"
    arrays = list(group.array_keys())
    for name in arrays:
        if "grid_mapping" in dict(group[name].attrs):
            return name
    candidates = [n for n in arrays if n not in _NON_DATA_ARRAYS]
    if not candidates:
        raise KeyError(f"no data array found in zarr group; arrays={arrays}")
    return max(candidates, key=lambda n: group[n].ndim)


def _transform_from_xy(group: Any) -> tuple[float, ...]:
    """Derive a GDAL geotransform from 1-D ``x`` / ``y`` pixel-centre coords."""
    if "x" not in group or "y" not in group:
        raise KeyError(
            "cannot determine GeoTransform: no GeoTransform attr and no x/y coords"
        )
    x = np.asarray(group["x"][:])
    y = np.asarray(group["y"][:])
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dy = float(y[1] - y[0]) if y.size > 1 else -1.0
    return (float(x[0]) - dx / 2.0, dx, 0.0, float(y[0]) - dy / 2.0, 0.0, dy)


def read_geobox(group: Any, *, data_name: str | None = None) -> dict[str, Any]:
    """Recover ``{crs_wkt, geotransform, epsg, legacy}`` from a Zarr ``group``.

    Tolerant of foreign GeoZarr stores (rioxarray / odc-geo / GDAL), not just
    pyramids-written ones (FR-8): the data array is auto-detected when
    ``data_name`` is ``None``; the CRS comes from the grid-mapping variable named
    by the data array's ``grid_mapping`` attribute (default ``spatial_ref``),
    reading ``crs_wkt`` / ``spatial_ref`` / ``proj:wkt2`` and ``epsg`` /
    ``proj:epsg``; the transform is taken from ``GeoTransform`` or, failing that,
    derived from the 1-D ``x`` / ``y`` coordinates. Stores with the geo-referencing
    as flat attributes on the data array (legacy pyramids layout) still read, with
    a :class:`DeprecationWarning`.

    Args:
        group: An open :class:`zarr.hierarchy.Group`.
        data_name: Name of the data array; auto-detected when ``None``.

    Returns:
        Dict with ``crs_wkt`` (str | None), ``geotransform`` (tuple[float, ...]),
        ``epsg`` (int; ``0`` when unknown) and ``legacy`` (bool).

    Raises:
        KeyError: When no ``GeoTransform`` and no ``x``/``y`` coords are present.
    """
    if data_name is None:
        data_name = detect_data_var(group)
    # Warn (but don't fail) when the store advertises a schema version we don't
    # know about — keeps forward-compat soft instead of brittle.
    schema_version = group.attrs.get("pyramids_zarr_version")
    if schema_version is not None and str(schema_version) not in {
        "1",
        ZARR_SCHEMA_VERSION,
    }:
        warnings.warn(
            f"unknown pyramids_zarr_version {schema_version!r}; attempting to "
            f"read with the v{ZARR_SCHEMA_VERSION} schema",
            stacklevel=2,
        )
    data_attrs = dict(group[data_name].attrs)
    gm_name = data_attrs.get("grid_mapping", GRID_MAPPING_VAR)

    legacy = False
    if gm_name in group:
        attrs = dict(group[gm_name].attrs)
    elif GRID_MAPPING_VAR in group:
        attrs = dict(group[GRID_MAPPING_VAR].attrs)
    else:
        attrs = data_attrs
        legacy = True
        warnings.warn(_LEGACY_WARNING, DeprecationWarning, stacklevel=2)

    crs_wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref") or attrs.get("proj:wkt2")
    epsg = int(attrs.get("epsg") or attrs.get("proj:epsg") or 0)
    gt_str = attrs.get("GeoTransform")
    if gt_str:
        geotransform = tuple(float(v) for v in gt_str.split())
    else:
        geotransform = _transform_from_xy(group)
    return {
        "crs_wkt": crs_wkt,
        "geotransform": geotransform,
        "epsg": epsg,
        "legacy": legacy,
    }
