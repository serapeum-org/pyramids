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

from collections.abc import Sequence
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
    either a :class:`pystac.Item` or a plain JSON dict. Delegates to the
    shared duck-typed accessors in :mod:`pyramids.stac._item` so this loader
    and :func:`pyramids.stac._loader._resolve_asset` interpret the contract
    identically.

    Args:
        item: Any object with an `assets` dict mapping asset keys
            to objects / dicts bearing an `href`.
        asset_key: Asset name (`"B04"`, `"visual"`,...).

    Returns:
        str: The asset's href.

    Raises:
        StacAssetError: When `asset_key` is not present on the item, or when
            the asset exists but has no `href` (subclasses :class:`KeyError`).
    """
    # Imported lazily to break the pyramids.dataset -> pyramids.stac ->
    # pyramids.dataset import cycle: dataset/__init__ loads collection -> _stac
    # before Dataset is bound, and pyramids.stac.__init__ imports _loader, which
    # imports pyramids.dataset.Dataset. Top-level importing pyramids.stac here
    # would therefore fail mid-init. (Same carve-out as the DatasetCollection
    # import in from_stac below.)
    from pyramids.stac._item import asset_href, get_asset

    asset = get_asset(item, asset_key)
    return asset_href(asset, item=item, asset_key=asset_key)


def _horizontal_bounds(b: Sequence[float]) -> tuple[float, float, float, float]:
    """Extract `(west, south, east, north)` from a 2D or 3D bbox.

    A GeoJSON / STAC bbox (RFC 7946 §5) is `[west, south, east, north]`
    in 2D and `[west, south, min_elev, east, north, max_elev]` in 3D.
    The horizontal members are the first two values and the two values
    starting at the midpoint, so this works for both lengths.

    Args:
        b: A bbox sequence of length 4 (2D) or 6 (3D).

    Returns:
        The `(west, south, east, north)` horizontal extent as floats.

    Raises:
        ValueError: When `b` has neither 4 nor 6 elements.
    """
    n = len(b)
    if n not in (4, 6):
        raise ValueError(
            f"bbox must have 4 (2D) or 6 (3D) elements, got {n}: {list(b)!r}"
        )
    half = n // 2
    return float(b[0]), float(b[1]), float(b[half]), float(b[half + 1])


def _item_intersects_bbox(
    item: Any,
    bbox: Sequence[float],
) -> bool:
    """Return True if `item.bbox` overlaps `bbox` (lon/lat box).

    Reads `item.bbox` as either an attribute (pystac.Item) or a
    dict key (raw JSON). Both the query `bbox` and the item bbox may be
    2D (4-element) or 3D (6-element) — only the horizontal extent is
    compared (see :func:`_horizontal_bounds`). Items without a bbox are
    treated as intersecting (permissive default — the caller opted in to
    the bbox filter, not the item).
    """
    item_bbox = getattr(item, "bbox", None)
    if item_bbox is None and isinstance(item, dict):
        item_bbox = item.get("bbox")
    if item_bbox is None:
        result = True
    else:
        minx, miny, maxx, maxy = _horizontal_bounds(bbox)
        i_minx, i_miny, i_maxx, i_maxy = _horizontal_bounds(item_bbox)
        result = not (i_maxx < minx or i_minx > maxx or i_maxy < miny or i_miny > maxy)
    return result


def from_stac(
    items: Any,
    asset: str,
    *,
    patch_url: Callable[[str], str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    max_items: int | None = None,
    signer: Any = None,
) -> DatasetCollection:
    """Build a :class:`DatasetCollection` from a STAC ItemCollection.

    .. note::
        This is the private implementation. The **public API** is the
        :meth:`DatasetCollection.from_stac` classmethod — call
        ``DatasetCollection.from_stac(items, asset, ...)`` rather than importing
        this function. It lives in a separate module only to break the
        ``dataset`` ↔ ``stac`` ↔ ``collection`` import cycle, and is not
        re-exported from :mod:`pyramids.dataset`.

    Extracts one named asset's href from each item, optionally runs
    `patch_url` and then a `signer` on each href, and forwards to
    :meth:`DatasetCollection.from_files`.

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
        patch_url: Optional callable applied to each href (runs before
            `signer`) — a low-level hook for ad-hoc URL rewriting.
        bbox: Optional `(minx, miny, maxx, maxy)` lon/lat filter;
            items whose `bbox` doesn't intersect are dropped
            before hrefs are resolved.
        max_items: Optional cap on the number of items consumed
            (after bbox filtering).
        signer: Optional signer exposing `sign_href(str) -> str` and
            `gdal_env() -> dict[str, str]` (e.g. a
            :class:`pyramids.stac.signers.Signer`). When given, **both**
            hooks are applied — exactly as :func:`pyramids.stac.load_asset`
            does: every resolved href is rewritten through
            `signer.sign_href` (e.g. grafting a SAS token), and
            `signer.gdal_env()` is captured onto the returned collection so
            every (eager and lazy) read of the backing files installs those
            credentials (`AWS_REQUEST_PAYER`, an `Authorization` header, …).
            This makes both URL-signing signers and env-credentialed signers
            (Requester-Pays, bearer) work through `from_stac`. `None`
            (default) leaves hrefs untouched and captures no config.

    Returns:
        DatasetCollection: A file-backed collection whose
        `time_length` equals `len(items)` and whose per-timestep
        backing file is the resolved asset URL.

    Raises:
        StacAssetError: When any item is missing the requested asset
            (subclasses `KeyError`).
        ValueError: When `items` yields zero items after filtering.

    Examples:
        - Build a DatasetCollection from raw STAC JSON dicts (no
          pystac required) via the public classmethod:
            ```python
            >>> raw_items = [  # doctest: +SKIP
            ...     {"assets": {"B04": {"href": "s3://.../scene1_B04.tif"}}},
            ...     {"assets": {"B04": {"href": "s3://.../scene2_B04.tif"}}},
            ... ]
            >>> from pyramids.dataset import DatasetCollection  # doctest: +SKIP
            >>> collection = DatasetCollection.from_stac(raw_items, asset="B04")  # doctest: +SKIP
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
        if signer is not None:
            href = signer.sign_href(href)
        hrefs.append(href)

    gdal_env = signer.gdal_env() if signer is not None else None

    from pyramids.dataset.collection import DatasetCollection

    return DatasetCollection.from_files(hrefs, gdal_env=gdal_env)


__all__ = ["from_stac"]
