"""STAC ItemCollection → :class:`DatasetCollection`.

Given a sequence of STAC Items — :class:`pystac.Item`
objects, raw JSON dicts, or anything else with `.assets` and
`.bbox` semantics — extract the chosen asset's `href` from each
item and delegate to :meth:`DatasetCollection.from_files`. Full
odc-stac-style features (geobox-tiled graph, auto-geobox derivation,
`fuse_func`, `errors_as_nodata`) are deliberately out of scope —
those users are better served by the odc-stac or stackstac packages
directly.

The implementation is fully duck-typed. pyramids does **not** import
or depend on pystac; the STAC Item / Asset contract is interpreted
via :func:`getattr` + dict lookup. Users typically build Items via
:mod:`pystac-client` (which carries pystac transitively) or from
raw JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pyramids.dataset.collection import DatasetCollection


def _iter_items(items: Any) -> list[Any]:
    """Normalise `items` to a list of STAC Items.

    Accepts a :class:`pystac.ItemCollection`, a list, or any iterable
    yielding STAC items.
    """
    if hasattr(items, "__iter__"):
        return list(items)
    raise TypeError(
        f"items must be iterable (ItemCollection or list), got {type(items).__name__}"
    )


def _resolve_asset_href(item: Any, asset_key: str) -> str:
    """Return the href of a named asset on a STAC Item.

    Supports both :class:`pystac.Asset` (`.href` attribute) and
    raw-dict STAC assets (`{"href": "..."}`) so callers can pass
    either a :class:`pystac.Item` or a plain JSON dict.

    Args:
        item: Any object with an `assets` dict mapping asset keys
            to objects / dicts bearing an `href`.
        asset_key: Asset name (`"B04"`, `"visual"`,...).

    Returns:
        str: The asset's href.

    Raises:
        KeyError: When `asset_key` is not present on the item, or
            when the asset exists but has no `href`.
    """
    assets = getattr(item, "assets", None)
    if assets is None and isinstance(item, dict):
        assets = item.get("assets")
    if assets is None or asset_key not in assets:
        available = list(assets or [])
        item_id = getattr(item, "id", None)
        if item_id is None and isinstance(item, dict):
            item_id = item.get("id", "?")
        raise KeyError(
            f"asset {asset_key!r} not found on STAC item "
            f"{item_id}; available: {available}"
        )
    asset = assets[asset_key]
    href = getattr(asset, "href", None)
    if href is None and isinstance(asset, dict):
        href = asset.get("href")
    if href is None:
        item_id = getattr(item, "id", None)
        if item_id is None and isinstance(item, dict):
            item_id = item.get("id", "?")
        raise KeyError(f"asset {asset_key!r} on STAC item {item_id} has no 'href'")
    return str(href)


def _item_intersects_bbox(
    item: Any,
    bbox: tuple[float, float, float, float],
) -> bool:
    """Return True if `item.bbox` overlaps `bbox` (lon/lat box).

    Reads `item.bbox` as either an attribute (pystac.Item) or a
    dict key (raw JSON). Items without a bbox are treated as
    intersecting (permissive default — the caller opted in to the
    bbox filter, not the item).
    """
    item_bbox = getattr(item, "bbox", None)
    if item_bbox is None and isinstance(item, dict):
        item_bbox = item.get("bbox")
    if item_bbox is None:
        result = True
    else:
        minx, miny, maxx, maxy = bbox
        i_minx, i_miny, i_maxx, i_maxy = item_bbox
        result = not (i_maxx < minx or i_minx > maxx or i_maxy < miny or i_miny > maxy)
    return result


def from_stac(
    items: Any,
    asset: str,
    *,
    patch_url: Callable[[str], str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    max_items: int | None = None,
) -> DatasetCollection:
    """Build a :class:`DatasetCollection` from a STAC ItemCollection.

    Extracts one named asset's href from each item, optionally runs
    `patch_url` on each href (typical use: sign a Planetary Computer
    URL), and forwards to :meth:`DatasetCollection.from_files`.

    The item interface is fully duck-typed. Any of these shapes work:

    * :class:`pystac.Item` objects (`item.assets["B04"].href`).
    * Raw STAC JSON dicts (`item["assets"]["B04"]["href"]`).
    * Any object exposing a dict-like `.assets` attribute whose
      values bear a `.href` attribute or `"href"` key.

    pyramids does not import pystac; users who construct Items via
    :mod:`pystac_client` / :mod:`pystac` pick that dependency up
    through those libraries directly.

    Args:
        items: Iterable of STAC Items (see duck-typed shapes above).
        asset: Asset key (e.g. `"B04"`, `"visual"`) whose
            `href` on each item becomes a timestep in the
            resulting collection.
        patch_url: Optional callable applied to each href — use for
            signing requester-pays URLs
            (`planetary_computer.sign`, etc.).
        bbox: Optional `(minx, miny, maxx, maxy)` lon/lat filter;
            items whose `bbox` doesn't intersect are dropped
            before hrefs are resolved.
        max_items: Optional cap on the number of items consumed
            (after bbox filtering).

    Returns:
        DatasetCollection: A file-backed collection whose
        `time_length` equals `len(items)` and whose per-timestep
        backing file is the resolved asset URL.

    Raises:
        KeyError: When any item is missing the requested asset.
        ValueError: When `items` yields zero items after filtering.

    Examples:
        - Build a DatasetCollection from raw STAC JSON dicts (no
          pystac required):
            ```python
            >>> raw_items = [  # doctest: +SKIP
            ...     {"assets": {"B04": {"href": "s3://.../scene1_B04.tif"}}},
            ...     {"assets": {"B04": {"href": "s3://.../scene2_B04.tif"}}},
            ... ]
            >>> from pyramids.dataset._stac import from_stac  # doctest: +SKIP
            >>> collection = from_stac(raw_items, asset="B04")  # doctest: +SKIP
            >>> collection.time_length  # doctest: +SKIP
            2

            ```
    """
    item_list = _iter_items(items)
    if bbox is not None:
        item_list = [i for i in item_list if _item_intersects_bbox(i, bbox)]
    if max_items is not None:
        item_list = item_list[:max_items]
    hrefs = []
    for item in item_list:
        href = _resolve_asset_href(item, asset)
        if patch_url is not None:
            href = patch_url(href)
        hrefs.append(href)

    from pyramids.dataset.collection import DatasetCollection

    return DatasetCollection.from_files(hrefs)


__all__ = ["from_stac"]
