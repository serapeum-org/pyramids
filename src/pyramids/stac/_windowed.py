"""Item-level windowed reads: pull a bbox / point / geometry window off an asset.

A caller who wants one window out of one asset of one STAC Item otherwise writes
the same four lines every time — resolve the asset, sign it, open it, crop it.
These helpers are that composition and nothing more: each resolves the asset
through :func:`pyramids.stac.load_asset` (so signing, the `/vsicurl` fast-read
preset and the `alternate-assets` preference all behave identically to a full
read) and then delegates to the overview-decimated reads the `COG` engine
already owns:

| helper                  | delegates to                                  |
|-------------------------|-----------------------------------------------|
| :func:`read_item_part`  | :meth:`pyramids.dataset.Dataset.read_part`    |
| :func:`read_item_preview` | :meth:`pyramids.dataset.Dataset.preview`    |
| :func:`read_item_point` | :meth:`pyramids.dataset.Dataset.point`        |
| :func:`read_item_feature` | :meth:`pyramids.dataset.Dataset.read_part`  |

No windowing arithmetic lives here. The decimation, the pixel snapping, the
partial-overlap padding and the CRS transform are the engine's, so an item-level
read and a `Dataset`-level one answer the same array — and a COG over
`/vsicurl/` still fetches only the byte ranges its window needs, because the
window is pushed into GDAL's read rather than applied to an array afterwards.

**Scope.** These are `numpy`-returning reads of a single asset. Deliberately
absent, and not planned here: XYZ/TMS tile serving, PNG/JPEG rendering,
colormaps, and cross-asset band-math expressions. That stack is a dynamic tile
server (rio-tiler's territory), not pyramids' mission. The one XYZ primitive
pyramids does own — :meth:`pyramids.dataset.Dataset.read_tile`, which is a
`read_part` over Web-Mercator tile bounds — stays at the `Dataset` level, where
it is a raster read rather than a serving layer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

import numpy as np

from pyramids.base._errors import UnsupportedAssetError
from pyramids.dataset import Dataset
from pyramids.stac._loader import load_asset

# STAC states item geometries and bboxes in WGS84 lon/lat, so an item-level
# window defaults to EPSG:4326 -- unlike the `Dataset`-level reads, which default
# to the raster's own coordinates because they have no item to inherit a CRS
# convention from. Pass `None` to opt back into the raster's coordinates.
STAC_WINDOW_CRS = 4326
"""Default CRS of a window passed to the item-level reads (STAC's lon/lat)."""


def _open_windowed(
    item_or_asset: Any,
    asset_key: str | None,
    signer: Any,
    alternate: str | Sequence[str] | None,
) -> Dataset:
    """Open an asset for a windowed read, rejecting readers that cannot do one.

    Args:
        item_or_asset: A STAC Item (pystac object or raw dict) or an Asset.
        asset_key: Asset name when passing an Item; `None` for an Asset.
        signer: Optional signer, forwarded to :func:`load_asset`.
        alternate: `alternate-assets` preference, forwarded to
            :func:`load_asset`.

    Returns:
        The opened :class:`~pyramids.dataset.Dataset`.

    Raises:
        UnsupportedAssetError: The asset opened as something other than a
            `Dataset` — today only a 4-D Zarr cube, which
            :func:`load_asset` answers as a
            :class:`~pyramids.dataset.DatasetCollection` and which carries no
            single raster to window.
    """
    opened = load_asset(item_or_asset, asset_key, signer=signer, alternate=alternate)
    if not isinstance(opened, Dataset):
        raise UnsupportedAssetError(
            f"asset {asset_key!r} opened as {type(opened).__name__}, which has no "
            "windowed read; open it with load_asset and slice the collection instead."
        )
    return opened


def _coordinate_pairs(node: Any) -> Iterator[tuple[float, float]]:
    """Yield `(x, y)` from a nested GeoJSON `coordinates` structure.

    Args:
        node: A GeoJSON position, or any nesting of positions (line, ring,
            polygon, multi-polygon).

    Yields:
        Each position's first two ordinates; a 3-D position's elevation is
        dropped, since the window is horizontal.
    """
    if isinstance(node, (list, tuple)):
        if len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]):
            yield float(node[0]), float(node[1])
        else:
            for child in node:
                yield from _coordinate_pairs(child)


def _geometry_pairs(geometry: Any) -> Iterator[tuple[float, float]]:
    """Yield every position in a GeoJSON geometry, Feature or collection.

    Args:
        geometry: A GeoJSON geometry, `Feature`, `FeatureCollection` or
            `GeometryCollection` mapping, or a bare `coordinates` nesting.

    Yields:
        Every `(x, y)` position found anywhere in the structure.
    """
    if isinstance(geometry, Mapping):
        if "coordinates" in geometry:
            yield from _coordinate_pairs(geometry["coordinates"])
        inner = geometry.get("geometry")
        if isinstance(inner, Mapping):
            yield from _geometry_pairs(inner)
        for member in geometry.get("geometries") or ():
            yield from _geometry_pairs(member)
        for feature in geometry.get("features") or ():
            yield from _geometry_pairs(feature)
    else:
        yield from _coordinate_pairs(geometry)


def geometry_bounds(geometry: Any) -> tuple[float, float, float, float]:
    """Return a geometry's `(min_x, min_y, max_x, max_y)` envelope.

    Accepts anything a STAC caller already has in hand: a shapely geometry (read
    through its `bounds`), a GeoJSON geometry / `Feature` /
    `FeatureCollection` / `GeometryCollection` mapping, or a bare `coordinates`
    nesting.

    Args:
        geometry: The geometry to bound.

    Returns:
        The horizontal envelope as `(min_x, min_y, max_x, max_y)`, in whatever
        CRS the geometry's coordinates are stated in.

    Raises:
        ValueError: The geometry carries no coordinates to bound.

    Examples:
        - A GeoJSON polygon reduces to its envelope:
            ```python
            >>> from pyramids.stac._windowed import geometry_bounds
            >>> polygon = {
            ...     "type": "Polygon",
            ...     "coordinates": [[[1.0, 2.0], [3.0, 2.0], [3.0, 5.0], [1.0, 2.0]]],
            ... }
            >>> geometry_bounds(polygon)
            (1.0, 2.0, 3.0, 5.0)

            ```
        - A `Feature` wrapper is unwrapped, and 3-D positions lose elevation:
            ```python
            >>> feature = {
            ...     "type": "Feature",
            ...     "properties": {},
            ...     "geometry": {"type": "Point", "coordinates": [7.0, 8.0, 100.0]},
            ... }
            >>> geometry_bounds(feature)
            (7.0, 8.0, 7.0, 8.0)

            ```
        - An empty geometry is an error rather than a silent whole-raster read:
            ```python
            >>> geometry_bounds({"type": "Polygon", "coordinates": []})
            Traceback (most recent call last):
                ...
            ValueError: geometry carries no coordinates to build a window from

            ```
    """
    bounds = getattr(geometry, "bounds", None)
    if bounds is not None:
        envelope = (
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
            float(bounds[3]),
        )
    else:
        pairs = list(_geometry_pairs(geometry))
        if not pairs:
            raise ValueError("geometry carries no coordinates to build a window from")
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        envelope = (min(xs), min(ys), max(xs), max(ys))
    return envelope


def read_item_part(
    item_or_asset: Any,
    asset_key: str | None,
    bbox: Sequence[float],
    *,
    bbox_crs: int | str | None = STAC_WINDOW_CRS,
    signer: Any = None,
    alternate: str | Sequence[str] | None = None,
    **read_options: Any,
) -> np.typing.NDArray | tuple[np.typing.NDArray, tuple[float, ...]]:
    """Read a bbox window out of one asset of a STAC Item.

    Opens the asset through :func:`~pyramids.stac.load_asset` and delegates to
    :meth:`~pyramids.dataset.Dataset.read_part`, so the read is decimated from
    the nearest overview and a COG over `/vsicurl/` fetches only the window's
    byte ranges.

    Args:
        item_or_asset: A STAC Item (pystac object or raw dict) or an Asset.
        asset_key: Asset name when passing an Item; `None` for an Asset.
        bbox: `(min_x, min_y, max_x, max_y)` window, stated in `bbox_crs`.
        bbox_crs: CRS of `bbox`. Defaults to :data:`STAC_WINDOW_CRS`
            (EPSG:4326), the CRS STAC states item geometry in; pass `None` to
            give the bbox in the asset's own coordinates instead.
        signer: Optional signer, forwarded to :func:`~pyramids.stac.load_asset`.
        alternate: `alternate-assets` key (or keys in preference order) to
            prefer over the canonical href.
        **read_options: Forwarded verbatim to
            :meth:`~pyramids.dataset.Dataset.read_part` — `dst_width`,
            `dst_height`, `resampling`, `band`, `return_transform`.

    Returns:
        numpy.ndarray: `(rows, cols)` for a single band or
        `(bands, rows, cols)` for all of them; an `(array, geotransform)` tuple
        when `return_transform=True` is forwarded.

    Raises:
        StacAssetError: The asset is missing or has no href.
        UnsupportedAssetError: The asset's type matches no reader, or opened as
            something with no windowed read (a 4-D Zarr cube).
        CRSError: A `bbox_crs` was given but the asset has no CRS to transform
            into.
        OutOfBoundsError: The window does not intersect the asset at all.

    Examples:
        - Pull a lon/lat window out of a catalog's red band:
            ```python
            >>> from pyramids.stac import read_item_part  # doctest: +SKIP
            >>> window = read_item_part(  # doctest: +SKIP
            ...     item, "B04", (12.4, 41.8, 12.6, 42.0), dst_width=256, dst_height=256
            ... )
            >>> window.shape  # doctest: +SKIP
            (256, 256)

            ```
    """
    dataset = _open_windowed(item_or_asset, asset_key, signer, alternate)
    return dataset.read_part(
        (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        bbox_crs=bbox_crs,
        **read_options,
    )


def read_item_preview(
    item_or_asset: Any,
    asset_key: str | None = None,
    *,
    max_size: int = 1024,
    signer: Any = None,
    alternate: str | Sequence[str] | None = None,
    **read_options: Any,
) -> np.typing.NDArray:
    """Read a whole-asset thumbnail for one asset of a STAC Item.

    Opens the asset through :func:`~pyramids.stac.load_asset` and delegates to
    :meth:`~pyramids.dataset.Dataset.preview`, which pulls from a coarse
    overview where one exists — so previewing a large COG costs a few ranged
    requests rather than the whole file.

    Args:
        item_or_asset: A STAC Item (pystac object or raw dict) or an Asset.
        asset_key: Asset name when passing an Item; `None` for an Asset.
        max_size: Maximum pixels on the longer edge. Defaults to 1024.
        signer: Optional signer, forwarded to :func:`~pyramids.stac.load_asset`.
        alternate: `alternate-assets` key (or keys in preference order) to
            prefer over the canonical href.
        **read_options: Forwarded verbatim to
            :meth:`~pyramids.dataset.Dataset.preview` — `resampling`, `band`.

    Returns:
        numpy.ndarray: The downsampled array, `(rows, cols)` or
        `(bands, rows, cols)`.

    Raises:
        StacAssetError: The asset is missing or has no href.
        UnsupportedAssetError: The asset's type matches no reader, or opened as
            something with no windowed read (a 4-D Zarr cube).

    Examples:
        - Thumbnail an item's visual asset at 128 px:
            ```python
            >>> from pyramids.stac import read_item_preview  # doctest: +SKIP
            >>> thumb = read_item_preview(item, "visual", max_size=128)  # doctest: +SKIP

            ```
    """
    dataset = _open_windowed(item_or_asset, asset_key, signer, alternate)
    # The `Dataset` facades forward through `*args, **kwargs` and so erase to
    # `Any`; cast back to the engine's declared return rather than widening this
    # signature.
    return cast("np.typing.NDArray", dataset.preview(max_size=max_size, **read_options))


def read_item_point(
    item_or_asset: Any,
    asset_key: str | None,
    point: Sequence[float],
    *,
    point_crs: int | str | None = STAC_WINDOW_CRS,
    signer: Any = None,
    alternate: str | Sequence[str] | None = None,
    **read_options: Any,
) -> np.typing.NDArray:
    """Sample one asset of a STAC Item at a single coordinate.

    Opens the asset through :func:`~pyramids.stac.load_asset` and delegates to
    :meth:`~pyramids.dataset.Dataset.point`, which reads the single pixel
    covering the coordinate.

    Args:
        item_or_asset: A STAC Item (pystac object or raw dict) or an Asset.
        asset_key: Asset name when passing an Item; `None` for an Asset.
        point: `(x, y)` — lon/lat by default — to sample.
        point_crs: CRS of `point`. Defaults to :data:`STAC_WINDOW_CRS`
            (EPSG:4326); pass `None` to give the coordinate in the asset's own
            coordinates instead.
        signer: Optional signer, forwarded to :func:`~pyramids.stac.load_asset`.
        alternate: `alternate-assets` key (or keys in preference order) to
            prefer over the canonical href.
        **read_options: Forwarded verbatim to
            :meth:`~pyramids.dataset.Dataset.point` — `band`.

    Returns:
        numpy.ndarray: A 0-d array for a single band, or a `(bands,)` array
        when no `band` is given.

    Raises:
        StacAssetError: The asset is missing or has no href.
        UnsupportedAssetError: The asset's type matches no reader, or opened as
            something with no windowed read (a 4-D Zarr cube).
        CRSError: A `point_crs` was given but the asset has no CRS to transform
            into.
        OutOfBoundsError: The coordinate falls outside the asset's extent.

    Examples:
        - Sample every band of an item's asset at one lon/lat:
            ```python
            >>> from pyramids.stac import read_item_point  # doctest: +SKIP
            >>> read_item_point(item, "B04", (12.5, 41.9))  # doctest: +SKIP
            array([1234.])

            ```
    """
    dataset = _open_windowed(item_or_asset, asset_key, signer, alternate)
    return cast(
        "np.typing.NDArray",
        dataset.point(
            float(point[0]), float(point[1]), point_crs=point_crs, **read_options
        ),
    )


def read_item_feature(
    item_or_asset: Any,
    asset_key: str | None,
    geometry: Any,
    *,
    geometry_crs: int | str | None = STAC_WINDOW_CRS,
    signer: Any = None,
    alternate: str | Sequence[str] | None = None,
    **read_options: Any,
) -> np.typing.NDArray | tuple[np.typing.NDArray, tuple[float, ...]]:
    """Read the window covering a geometry out of one asset of a STAC Item.

    The geometry is reduced to its envelope (:func:`geometry_bounds`) and read
    through :func:`read_item_part`. The result is therefore the geometry's
    **bounding window**, not a masked cut-out: cells inside the window but
    outside the geometry keep their values. Masking is a separate operation —
    rasterize the geometry with
    :meth:`pyramids.feature.FeatureCollection.to_dataset`, or clip the opened
    asset with :meth:`pyramids.dataset.Dataset.clip`, when the shape itself must
    be honoured.

    Args:
        item_or_asset: A STAC Item (pystac object or raw dict) or an Asset.
        asset_key: Asset name when passing an Item; `None` for an Asset.
        geometry: A shapely geometry, or a GeoJSON geometry / `Feature` /
            `FeatureCollection` / `GeometryCollection` mapping.
        geometry_crs: CRS of the geometry's coordinates. Defaults to
            :data:`STAC_WINDOW_CRS` (EPSG:4326), which is what GeoJSON states;
            pass `None` for a geometry already in the asset's coordinates.
        signer: Optional signer, forwarded to :func:`~pyramids.stac.load_asset`.
        alternate: `alternate-assets` key (or keys in preference order) to
            prefer over the canonical href.
        **read_options: Forwarded verbatim to
            :meth:`~pyramids.dataset.Dataset.read_part` — `dst_width`,
            `dst_height`, `resampling`, `band`, `return_transform`.

    Returns:
        numpy.ndarray: The geometry's bounding window, shaped as
        :func:`read_item_part` returns it.

    Raises:
        ValueError: The geometry carries no coordinates.
        StacAssetError: The asset is missing or has no href.
        UnsupportedAssetError: The asset's type matches no reader, or opened as
            something with no windowed read (a 4-D Zarr cube).
        OutOfBoundsError: The envelope does not intersect the asset at all.

    Examples:
        - Read the window covering an AOI polygon:
            ```python
            >>> from pyramids.stac import read_item_feature  # doctest: +SKIP
            >>> aoi = {"type": "Polygon", "coordinates": [[[12.4, 41.8], [12.6, 41.8],
            ...                                            [12.6, 42.0], [12.4, 41.8]]]}
            >>> window = read_item_feature(item, "B04", aoi)  # doctest: +SKIP

            ```
    """
    return read_item_part(
        item_or_asset,
        asset_key,
        geometry_bounds(geometry),
        bbox_crs=geometry_crs,
        signer=signer,
        alternate=alternate,
        **read_options,
    )


__all__ = [
    "STAC_WINDOW_CRS",
    "geometry_bounds",
    "read_item_feature",
    "read_item_part",
    "read_item_point",
    "read_item_preview",
]
