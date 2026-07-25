"""Duck-typed accessors for STAC Items and Assets (no pystac dependency).

Both :mod:`pyramids.stac._loader` (asset → reader dispatch) and
:mod:`pyramids.dataset._stac` (ItemCollection → :class:`DatasetCollection`)
need to read the same fields off a STAC Item / Asset that may be a
:class:`pystac.Item` / :class:`pystac.Asset` object **or** a raw STAC JSON
dict. Before this module those two readers each carried their own
``getattr``-then-dict-fallback resolver, which duplicated the contract and
risked drifting apart.

Centralising the access here gives one duck-typed contract:

* :class:`pystac.Item` / :class:`pystac.Asset` objects (attribute access:
  ``item.assets``, ``asset.href``, ``asset.media_type``, ``asset.extra_fields``).
* Raw STAC JSON dicts (``item["assets"]``, ``asset["href"]``, ``asset["type"]``,
  and extension fields as top-level keys on the asset dict).

pyramids does **not** import or depend on pystac. The extension-field accessor
(:func:`asset_field`) is the extension point for reading ``proj`` / ``raster`` /
``eo`` metadata (PB-1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pyramids.base._errors import StacAssetError


def item_id(item: Any) -> Any:
    """Return a STAC Item's id for error messages, or `"?"` when absent.

    Args:
        item: A STAC Item (pystac object or raw dict).

    Returns:
        The item id (`item.id` or `item["id"]`), or `"?"` when neither is set.

    Examples:
        - A raw dict item exposes its id:
            ```python
            >>> from pyramids.stac._item import item_id
            >>> item_id({"id": "scene-1", "assets": {}})
            'scene-1'

            ```
        - A missing id falls back to a placeholder:
            ```python
            >>> item_id({"assets": {}})
            '?'

            ```
    """
    iid = getattr(item, "id", None)
    if iid is None and isinstance(item, dict):
        iid = item.get("id", "?")
    return iid if iid is not None else "?"


def item_properties(item: Any) -> Mapping[str, Any]:
    """Return a STAC Item's `properties` mapping (empty when absent).

    Args:
        item: A STAC Item (pystac object or raw dict).

    Returns:
        The item's `properties` mapping, or an empty dict when not present.

    Examples:
        - Properties carry item-level extension fields (e.g. `proj:epsg`):
            ```python
            >>> from pyramids.stac._item import item_properties
            >>> item_properties({"properties": {"proj:epsg": 32633}})["proj:epsg"]
            32633

            ```
        - A property-less item yields an empty mapping:
            ```python
            >>> item_properties({"id": "x"})
            {}

            ```
    """
    props = getattr(item, "properties", None)
    if props is None and isinstance(item, dict):
        props = item.get("properties")
    return props or {}


def item_bbox(item: Any) -> Sequence[float] | None:
    """Return a STAC Item's `bbox`, or `None` when absent.

    A STAC bbox is WGS84 by spec and is either 2D (`[west, south, east, north]`)
    or 3D (`[west, south, min_elev, east, north, max_elev]`). This accessor only
    locates it on either item shape; interpreting the length is the caller's job
    (see :func:`pyramids.dataset._stac._horizontal_bounds`).

    Args:
        item: A STAC Item (pystac object or raw dict).

    Returns:
        The bbox sequence, or `None` when the item carries none.

    Examples:
        - A raw dict item exposes its bbox:
            ```python
            >>> from pyramids.stac._item import item_bbox
            >>> item_bbox({"bbox": [1.0, 2.0, 3.0, 4.0]})
            [1.0, 2.0, 3.0, 4.0]

            ```
        - A bbox-less item returns None:
            ```python
            >>> item_bbox({"id": "x"}) is None
            True

            ```
        - A 3D bbox comes back with its elevation members intact:
            ```python
            >>> item_bbox({"bbox": [1.0, 2.0, 0.0, 3.0, 4.0, 100.0]})[2]
            0.0

            ```

    See Also:
        - :func:`item_properties`: the sibling accessor for item-level
          extension fields.
        - :func:`pyramids.dataset._stac._horizontal_bounds`: reduces either
          bbox length to `(west, south, east, north)`.
    """
    bbox = getattr(item, "bbox", None)
    if bbox is None and isinstance(item, dict):
        bbox = item.get("bbox")
    return bbox


def get_assets(item: Any) -> Mapping[str, Any] | None:
    """Return a STAC Item's `assets` mapping, or `None` when absent.

    Args:
        item: A STAC Item (pystac object or raw dict).

    Returns:
        The assets mapping (asset key → asset object/dict), or `None`.

    Examples:
        - A raw item exposes its assets mapping:
            ```python
            >>> from pyramids.stac._item import get_assets
            >>> sorted(get_assets({"assets": {"B04": {"href": "b04.tif"}}}))
            ['B04']

            ```
        - An item without assets returns None:
            ```python
            >>> get_assets({"id": "x"}) is None
            True

            ```
    """
    assets = getattr(item, "assets", None)
    if assets is None and isinstance(item, dict):
        assets = item.get("assets")
    return assets


def get_asset(item: Any, asset_key: str) -> Any:
    """Return the named asset object/dict on a STAC Item.

    Args:
        item: A STAC Item (pystac object or raw dict).
        asset_key: The asset key to look up (e.g. `"B04"`, `"visual"`).

    Returns:
        The asset (a `pystac.Asset` or a raw asset dict).

    Raises:
        StacAssetError: The asset key is absent from the item (subclasses
            :class:`KeyError`, so `except KeyError:` still catches it).

    Examples:
        - Resolve a named asset dict:
            ```python
            >>> from pyramids.stac._item import get_asset
            >>> get_asset({"assets": {"B04": {"href": "b04.tif"}}}, "B04")["href"]
            'b04.tif'

            ```
        - A missing asset raises an error that lists what is available:
            ```python
            >>> get_asset({"assets": {"B04": {"href": "x"}}}, "B99")
            Traceback (most recent call last):
                ...
            pyramids.base._errors.StacAssetError: "asset 'B99' not found on STAC item ?; available: ['B04']"

            ```
    """
    assets = get_assets(item)
    if assets is None or asset_key not in assets:
        raise StacAssetError(
            f"asset {asset_key!r} not found on STAC item {item_id(item)}; "
            f"available: {list(assets or [])}"
        )
    return assets[asset_key]


def asset_href(asset: Any, *, item: Any = None, asset_key: str | None = None) -> str:
    """Return an asset's `href` as a string.

    Args:
        asset: A STAC Asset (pystac object with `.href`, or a dict with
            `"href"`).
        item: The owning Item, used only to enrich the error message.
        asset_key: The asset key, used only to enrich the error message.

    Returns:
        The asset href as a string.

    Raises:
        StacAssetError: The asset has no href (subclasses :class:`KeyError`).

    Examples:
        - Read the href from a dict asset:
            ```python
            >>> from pyramids.stac._item import asset_href
            >>> asset_href({"href": "s3://b/scene.tif", "type": "image/tiff"})
            's3://b/scene.tif'

            ```
        - A href-less asset raises a descriptive error:
            ```python
            >>> asset_href({"type": "image/tiff"}, asset_key="B04")
            Traceback (most recent call last):
                ...
            pyramids.base._errors.StacAssetError: "asset 'B04' on STAC item ? has no 'href'"

            ```
    """
    href = getattr(asset, "href", None)
    if href is None and isinstance(asset, dict):
        href = asset.get("href")
    if href is None:
        raise StacAssetError(
            f"asset {asset_key!r} on STAC item {item_id(item)} has no 'href'"
        )
    return str(href)


def asset_media_type(asset: Any) -> str | None:
    """Return an asset's media type, or `None` when absent.

    Reads `asset.media_type` (pystac) or `asset["type"]` (raw STAC JSON).

    Args:
        asset: A STAC Asset (pystac object or raw dict).

    Returns:
        The media-type string, or `None`.

    Examples:
        - A dict asset stores the media type under `"type"`:
            ```python
            >>> from pyramids.stac._item import asset_media_type
            >>> asset_media_type({"href": "x.tif", "type": "image/tiff"})
            'image/tiff'

            ```
        - A typeless asset returns None:
            ```python
            >>> asset_media_type({"href": "x.tif"}) is None
            True

            ```
    """
    media_type = getattr(asset, "media_type", None)
    if media_type is None and isinstance(asset, dict):
        media_type = asset.get("type")
    return media_type


def asset_field(asset: Any, key: str, default: Any = None) -> Any:
    """Return an extension field from an asset (e.g. `proj:transform`).

    Extension fields live as top-level keys on a raw asset dict, and in
    `asset.extra_fields` on a `pystac.Asset`. This is the building block for
    reading `proj` / `raster` / `eo` metadata (PB-1).

    Args:
        asset: A STAC Asset (pystac object or raw dict).
        key: The extension field key (e.g. `"proj:transform"`, `"raster:bands"`).
        default: Value returned when the field is absent.

    Returns:
        The field value, or `default`.

    Examples:
        - Read a projection field from a dict asset:
            ```python
            >>> from pyramids.stac._item import asset_field
            >>> asset_field({"href": "x.tif", "proj:epsg": 32633}, "proj:epsg")
            32633

            ```
        - Absent fields fall back to the default:
            ```python
            >>> asset_field({"href": "x.tif"}, "proj:epsg", default="missing")
            'missing'

            ```
    """
    if isinstance(asset, dict):
        return asset.get(key, default)
    extra = getattr(asset, "extra_fields", None)
    if isinstance(extra, dict) and key in extra:
        return extra[key]
    return default


__all__ = [
    "asset_field",
    "asset_href",
    "asset_media_type",
    "get_asset",
    "get_assets",
    "item_bbox",
    "item_id",
    "item_properties",
]
