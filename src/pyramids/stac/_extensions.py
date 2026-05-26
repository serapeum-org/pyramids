"""Read STAC ``proj`` / ``raster`` / ``eo`` extension metadata (PB-1).

This module builds a cube/grid skeleton — CRS, geotransform, shape, per-band
nodata / scale / offset, band names — directly from the STAC Item JSON, without
opening any asset: pure dict reads (via :func:`pyramids.stac._item.asset_field`,
no pystac
dependency) that yield a grid/band-metadata dict downstream code can use to build
a VRT (PB-5), a multi-asset cube (PB-2), or a grid match (PC-2) **without** a
header open.

Scope note: these are *readers* only. They deliberately do **not** stamp the
metadata onto a :class:`~pyramids.dataset.Dataset` returned by
:func:`pyramids.stac.load_asset`, because that reader opens assets read-only
(remote ``/vsicurl`` COGs cannot be opened for write), and mutating a read-only
GDAL handle (``SetProjection`` / ``SetNoDataValue`` / ``SetScale``) raises under
``gdal.UseExceptions()``. Writable consumers (VRT/stack builders) apply the
metadata themselves from the dict this module returns.
"""

from __future__ import annotations

from typing import Any

from pyramids.stac._item import asset_field, get_asset, item_properties

_NODATA_STRINGS = {
    "nan": float("nan"),
    "+nan": float("nan"),
    "-nan": float("nan"),
    "inf": float("inf"),
    "+inf": float("inf"),
    "infinity": float("inf"),
    "-inf": float("-inf"),
    "-infinity": float("-inf"),
}


def parse_number(value: Any, default: Any = None) -> Any:
    """Coerce a STAC numeric field to a float, honouring nan/inf strings.

    The ``raster`` extension allows non-finite nodata values to be encoded as
    the strings ``"nan"`` / ``"inf"`` / ``"-inf"``.

    Args:
        value: The raw field value (number, numeric string, nan/inf string,
            or ``None``).
        default: Returned when `value` is ``None`` or cannot be parsed.

    Returns:
        A float for numeric / nan-inf inputs, otherwise `default`.

    Examples:
        - A plain number passes through as a float:
            ```python
            >>> from pyramids.stac._extensions import parse_number
            >>> parse_number(-9999)
            -9999.0

            ```
        - The string ``"-inf"`` becomes negative infinity:
            ```python
            >>> parse_number("-inf")
            -inf

            ```
        - An unparseable value falls back to the default:
            ```python
            >>> parse_number("n/a", default=0.0)
            0.0

            ```
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _NODATA_STRINGS:
            return _NODATA_STRINGS[token]
        try:
            return float(value)
        except ValueError:
            return default
    return default


def affine_to_geotransform(transform: Any) -> tuple[float, ...]:
    """Convert a STAC ``proj:transform`` affine to a GDAL geotransform.

    ``proj:transform`` is the affine ordering ``[a, b, c, d, e, f]``
    (mapping ``(col, row)`` to ``(x, y)``: ``x = a*col + b*row + c``,
    ``y = d*col + e*row + f``). GDAL's geotransform is the reordering
    ``(c, a, b, f, d, e)`` — i.e. ``(x_origin, x_res, x_rot, y_origin, y_rot,
    y_res)``. A 9-element affine (with a trailing ``[0, 0, 1]`` row) is
    accepted; only the first six coefficients are used.

    Args:
        transform: A 6- or 9-element ``proj:transform`` sequence.

    Returns:
        The 6-tuple GDAL geotransform.

    Raises:
        ValueError: When `transform` has fewer than six coefficients.

    Examples:
        - A north-up 30 m grid reorders to the GDAL geotransform:
            ```python
            >>> from pyramids.stac._extensions import affine_to_geotransform
            >>> affine_to_geotransform([30.0, 0.0, 224985.0, 0.0, -30.0, 6790215.0])
            (224985.0, 30.0, 0.0, 6790215.0, 0.0, -30.0)

            ```
        - The trailing ``[0, 0, 1]`` row of a 9-element affine is ignored:
            ```python
            >>> affine_to_geotransform([10.0, 0.0, 100.0, 0.0, -10.0, 200.0, 0.0, 0.0, 1.0])
            (100.0, 10.0, 0.0, 200.0, 0.0, -10.0)

            ```
    """
    coeffs = list(transform)
    if len(coeffs) < 6:
        raise ValueError(
            f"proj:transform must have at least 6 coefficients, got {len(coeffs)}: "
            f"{coeffs!r}"
        )
    a, b, c, d, e, f = (float(x) for x in coeffs[:6])
    return (c, a, b, f, d, e)


def geotransform_to_affine(geotransform: Any) -> list[float]:
    """Convert a GDAL geotransform to a STAC ``proj:transform`` affine.

    The inverse of :func:`affine_to_geotransform`. GDAL's geotransform is
    ``(c, a, b, f, d, e)`` — ``(x_origin, x_res, x_rot, y_origin, y_rot,
    y_res)`` — and ``proj:transform`` is the affine ordering
    ``[a, b, c, d, e, f]``.

    Args:
        geotransform: A 6-element GDAL geotransform.

    Returns:
        The 6-element ``proj:transform`` affine.

    Raises:
        ValueError: When `geotransform` has fewer than six coefficients.

    Examples:
        - A north-up 30 m grid maps back to the affine order:
            ```python
            >>> from pyramids.stac._extensions import geotransform_to_affine
            >>> geotransform_to_affine((224985.0, 30.0, 0.0, 6790215.0, 0.0, -30.0))
            [30.0, 0.0, 224985.0, 0.0, -30.0, 6790215.0]

            ```
    """
    gt = list(geotransform)
    if len(gt) < 6:
        raise ValueError(
            f"geotransform must have at least 6 coefficients, got {len(gt)}: {gt!r}"
        )
    c, a, b, f, d, e = (float(x) for x in gt[:6])
    return [a, b, c, d, e, f]


def read_extension_metadata(item: Any, asset_key: str | None = None) -> dict[str, Any]:
    """Read ``proj`` / ``raster`` / ``eo`` extension fields for a STAC asset.

    Item-level fields (under ``properties``) are read first and an asset-level
    value of the same key overrides them, matching the STAC convention that an
    asset narrows item-level metadata. No asset file is opened.

    Args:
        item: A STAC Item (pystac object or raw dict). When `asset_key` is
            ``None`` the `item` is treated as the asset itself.
        asset_key: The asset key whose metadata to read, or ``None`` to read
            a bare asset.

    Returns:
        A dict with keys:

        * ``epsg`` — ``proj:epsg`` (int) or ``None``.
        * ``crs`` — ``proj:code`` (e.g. ``"EPSG:32633"``) when present, else
          derived from ``proj:epsg``, else ``None``.
        * ``transform`` — raw ``proj:transform`` list or ``None``.
        * ``geotransform`` — the GDAL geotransform derived from ``transform``,
          or ``None`` when no transform is present.
        * ``shape`` — ``proj:shape`` ``[rows, cols]`` or ``None``.
        * ``raster_bands`` — ``raster:bands`` list or ``None``.
        * ``eo_bands`` — ``eo:bands`` list or ``None``.
        * ``band_names`` — names derived from ``eo:bands`` (``name`` or
          ``common_name``) when every band has one, else ``None``.

    Raises:
        StacAssetError: When `asset_key` is given but absent from the item.

    Examples:
        - Read a Sentinel-2-style asset's projection metadata from raw JSON:
            ```python
            >>> from pyramids.stac._extensions import read_extension_metadata
            >>> item = {
            ...     "properties": {"proj:epsg": 32633},
            ...     "assets": {"B04": {
            ...         "href": "s3://b/B04.tif",
            ...         "proj:shape": [10980, 10980],
            ...         "proj:transform": [10.0, 0.0, 600000.0, 0.0, -10.0, 5300040.0],
            ...         "raster:bands": [{"nodata": 0, "scale": 0.0001}],
            ...         "eo:bands": [{"name": "B04", "common_name": "red"}],
            ...     }},
            ... }
            >>> meta = read_extension_metadata(item, "B04")
            >>> meta["crs"]
            'EPSG:32633'
            >>> meta["geotransform"]
            (600000.0, 10.0, 0.0, 5300040.0, 0.0, -10.0)
            >>> meta["band_names"]
            ['B04']

            ```
        - An asset-level ``proj:epsg`` overrides the item-level value:
            ```python
            >>> item = {
            ...     "properties": {"proj:epsg": 4326},
            ...     "assets": {"dem": {"href": "x.tif", "proj:epsg": 3857}},
            ... }
            >>> read_extension_metadata(item, "dem")["epsg"]
            3857

            ```
        - A bare asset with no extension fields yields all-empty metadata:
            ```python
            >>> meta = read_extension_metadata({"href": "x.tif"})
            >>> (meta["crs"], meta["geotransform"], meta["raster_bands"])
            (None, None, None)

            ```
    """
    props = item_properties(item)
    asset = get_asset(item, asset_key) if asset_key is not None else item

    def pick(key: str, default: Any = None) -> Any:
        return asset_field(asset, key, props.get(key, default))

    epsg = pick("proj:epsg")
    code = pick("proj:code")
    if code is None and epsg is not None:
        code = f"EPSG:{epsg}"

    transform = pick("proj:transform")
    geotransform = affine_to_geotransform(transform) if transform else None

    eo_bands = asset_field(asset, "eo:bands")
    band_names: list[str] | None = None
    if eo_bands:
        names = [b.get("name") or b.get("common_name") for b in eo_bands]
        if all(names):
            band_names = list(names)

    return {
        "epsg": epsg,
        "crs": code,
        "transform": transform,
        "geotransform": geotransform,
        "shape": pick("proj:shape"),
        "raster_bands": asset_field(asset, "raster:bands"),
        "eo_bands": eo_bands,
        "band_names": band_names,
    }


__all__ = [
    "affine_to_geotransform",
    "geotransform_to_affine",
    "parse_number",
    "read_extension_metadata",
]
