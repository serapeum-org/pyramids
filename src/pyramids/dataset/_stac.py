"""STAC ItemCollection → :class:`DatasetCollection`.

Given a sequence of STAC Items — :class:`pystac.Item`
objects, raw JSON dicts, or anything else with `.assets` and
`.bbox` semantics — extract the chosen asset's `href` from each
item and delegate to :meth:`DatasetCollection.from_files`. Advanced
features (geobox-tiled graph, auto-geobox derivation, `fuse_func`,
`errors_as_nodata`) are deliberately out of scope.

The implementation is fully duck-typed. pyramids does **not** import
or depend on pystac; the STAC Item / Asset contract is interpreted
via :func:`getattr` + dict lookup. Users typically build Items via
:mod:`pystac-client` (which carries pystac transitively) or from
raw JSON.
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime as _datetime_cls
from datetime import timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable

from osgeo import osr

from pyramids.base._artifacts import artifact_dir
from pyramids.base._errors import StacAssetError

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


def _validate_lonlat_bbox(bbox: Sequence[float]) -> None:
    """Validate that `bbox` is a lon/lat (WGS84) box (L1).

    STAC item bboxes are WGS84 by spec, and :meth:`from_stac` compares the
    query box against them directly, so the query box must also be lon/lat.
    A projected box (e.g. UTM metres like ``600000``) silently matches nothing;
    rejecting it up front turns that into a clear error.

    Args:
        bbox: A 2D (4-element) or 3D (6-element) bbox.

    Raises:
        ValueError: When any horizontal coordinate falls outside
            ``[-180, 180]`` (longitude) / ``[-90, 90]`` (latitude).
    """
    west, south, east, north = _horizontal_bounds(bbox)
    lon_ok = -180.0 <= west <= 180.0 and -180.0 <= east <= 180.0
    lat_ok = -90.0 <= south <= 90.0 and -90.0 <= north <= 90.0
    if not (lon_ok and lat_ok):
        raise ValueError(
            "bbox must be lon/lat (WGS84) within longitude [-180, 180] and "
            f"latitude [-90, 90], got {list(bbox)!r}. STAC item bboxes are "
            "WGS84; reproject a projected box before filtering."
        )


def _lon_segments(west: float, east: float) -> list[tuple[float, float]]:
    """Split a longitude interval into non-wrapping segments (L3).

    A box with ``west > east`` crosses the antimeridian and covers
    ``[west, 180] ∪ [-180, east]``; otherwise it is a single ``[west, east]``.
    """
    if west <= east:
        return [(west, east)]
    return [(west, 180.0), (-180.0, east)]


def _lon_overlaps(a_west: float, a_east: float, b_west: float, b_east: float) -> bool:
    """Return True if two longitude intervals overlap, antimeridian-aware (L3)."""
    return any(
        not (a_e < b_w or a_w > b_e)
        for a_w, a_e in _lon_segments(a_west, a_east)
        for b_w, b_e in _lon_segments(b_west, b_east)
    )


def _item_intersects_bbox(
    item: Any,
    bbox: Sequence[float],
) -> bool:
    """Return True if `item.bbox` overlaps `bbox` (lon/lat box).

    Reads `item.bbox` as either an attribute (pystac.Item) or a
    dict key (raw JSON). Both the query `bbox` and the item bbox may be
    2D (4-element) or 3D (6-element) — only the horizontal extent is
    compared (see :func:`_horizontal_bounds`). Longitude overlap is
    antimeridian-aware (a box with ``west > east`` is treated as wrapping the
    dateline). Items without a bbox are treated as intersecting (permissive
    default — the caller opted in to the bbox filter, not the item).
    """
    item_bbox = getattr(item, "bbox", None)
    if item_bbox is None and isinstance(item, dict):
        item_bbox = item.get("bbox")
    if item_bbox is None:
        result = True
    else:
        q_west, q_south, q_east, q_north = _horizontal_bounds(bbox)
        i_west, i_south, i_east, i_north = _horizontal_bounds(item_bbox)
        lat_overlap = not (i_north < q_south or i_south > q_north)
        result = lat_overlap and _lon_overlaps(q_west, q_east, i_west, i_east)
    return result


def from_stac(
    items: Any,
    asset: str | Sequence[str],
    *,
    patch_url: Callable[[str], str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    max_items: int | None = None,
    signer: Any = None,
    align: bool = True,
    skip_missing: bool = False,
    groupby: str | None = None,
    like: Any = None,
    crs: int | str | None = None,
    resolution: float | None = None,
    bounds: Sequence[float] | None = None,
    anchor: str = "edge",
) -> DatasetCollection:
    """Build a :class:`DatasetCollection` from a STAC ItemCollection.

    .. note::
        This is the private implementation. The **public API** is the
        :meth:`DatasetCollection.from_stac` classmethod — call
        ``DatasetCollection.from_stac(items, asset, ...)`` rather than importing
        this function. It lives in a separate module only to break the
        ``dataset`` ↔ ``stac`` ↔ ``collection`` import cycle, and is not
        re-exported from :mod:`pyramids.dataset`.

    Two modes, selected by the type of `asset`:

    * **Single asset** (`asset` is a `str`): extract that asset's href from
      each item, run `patch_url` then `signer` on it, and forward the hrefs
      to :meth:`DatasetCollection.from_files` — a lazy, file-backed time
      stack (one band-set per timestep, read on demand via `/vsicurl`).
    * **Multi-asset** (`asset` is a sequence of keys, e.g.
      `["red", "green", "blue", "nir"]`): for each item, stack the named
      assets band-wise into one multi-band raster (band order = `asset`
      order, band names = the asset keys), then time-stack those per-item
      rasters. This is the "assets → band axis" model. Mixed-resolution
      assets are resampled onto the **first** asset's grid when `align=True`
      (the default).

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
        asset: Either a single asset key (`str`, e.g. `"B04"`, `"visual"`) for
            a single-asset time stack, or a sequence of keys (e.g.
            `["B04", "B03", "B02"]`) to stack those assets band-wise into one
            multi-band raster per timestep (band order = sequence order).
        patch_url: Optional callable applied to each href (runs before
            `signer`) — a low-level hook for ad-hoc URL rewriting.
        bbox: Optional `(minx, miny, maxx, maxy)` lon/lat filter;
            items whose `bbox` doesn't intersect are dropped
            before hrefs are resolved. This is a **client-side
            post-filter** over the already-materialised `items`; to bound
            the query at the STAC API itself use
            :func:`pyramids.stac.search` (M3).
        max_items: Optional cap on the number of items consumed (after
            bbox filtering). Also a **client-side** cap over `items`, not
            an API paging limit — see :func:`pyramids.stac.search`.
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
        align: Multi-asset only. When `True` (default), assets at differing
            native resolutions are resampled onto the first requested asset's
            grid (nearest, via :meth:`Dataset.from_band_files`). When `False`,
            a grid/CRS mismatch among an item's assets raises
            :class:`~pyramids.base._errors.AlignmentError`. Ignored in
            single-asset mode.
        skip_missing: When `True`, items missing any requested asset are
            dropped instead of raising. When `False` (default), a missing
            asset raises :class:`~pyramids.base._errors.StacAssetError`.
        groupby: How items map to timesteps. `None` (default) keeps one
            timestep per item.

            `"solar_day"` produces one timestep per acquisition date for
            **tiled optical Earth-observation** catalogs (Sentinel-2, Landsat,
            HLS, MODIS), where one overpass of an AOI is delivered as many
            granules/tiles. Each item's solar day is its UTC timestamp shifted
            by `centroid_longitude / 15` hours (≈ local solar time; see
            :func:`_solar_day`), reduced to a calendar date — the shift keeps a
            single overpass on one date instead of splitting it across
            UTC midnight. Items sharing a solar day are mosaicked with
            `merge_rasters(method="first")` (first-valid pixel wins on overlap;
            see :func:`_from_stac_solar_day`). `time_length` is the number of
            distinct solar days, in chronological order. Single-asset only.
        like: Optional target grid as an existing
            :class:`~pyramids.dataset.Dataset`; every timestep of the built
            cube is reprojected/resampled onto its CRS + grid (via
            :meth:`DatasetCollection.align`), guaranteeing pixel co-registration.
            Mutually exclusive with `crs`/`resolution`/`bounds`.
        crs: Target CRS (EPSG int or CRS string) for an explicit target grid.
            Must be given together with `resolution` and `bounds`.
        resolution: Target pixel size (CRS units) for an explicit target grid.
        bounds: Target `(minx, miny, maxx, maxy)` extent (in `crs`) for an
            explicit target grid.
        anchor: Grid-snap rule for the explicit `crs`/`resolution`/`bounds`
            grid. `"edge"` (default) snaps pixel edges to multiples of
            `resolution` (so independently-built grids co-register).

    Returns:
        DatasetCollection: A file-backed collection whose `time_length`
        equals the number of items kept. Single-asset mode backs each
        timestep directly with the resolved asset URL (lazy); multi-asset
        mode backs each timestep with a per-item multi-band raster
        materialised under a shared process-level temp root that is removed at
        interpreter exit (see :mod:`pyramids.base._artifacts`).

    Raises:
        StacAssetError: When an item is missing a requested asset and
            `skip_missing` is `False` (subclasses `KeyError`).
        AlignmentError: Multi-asset with `align=False` and an item's assets
            do not share a grid/CRS.
        ValueError: When no items remain after filtering / skipping.

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
        _validate_lonlat_bbox(bbox)
        item_list = [i for i in item_list if _item_intersects_bbox(i, bbox)]
    if max_items is not None:
        item_list = item_list[:max_items]

    gdal_env = signer.gdal_env() if signer is not None else None

    def _sign(href: str) -> str:
        if patch_url is not None:
            href = patch_url(href)
        if signer is not None:
            href = signer.sign_href(href)
        return href

    # Imported lazily to break the pyramids.dataset -> pyramids.stac ->
    # pyramids.dataset import cycle (see _resolve_asset_href above).
    from pyramids.dataset.collection import DatasetCollection

    target_grid = _resolve_target_grid(like, crs, resolution, bounds, anchor)

    if groupby is not None:
        if groupby != "solar_day":
            raise ValueError(f"groupby must be None or 'solar_day', got {groupby!r}.")
        if not isinstance(asset, str):
            raise ValueError(
                "groupby='solar_day' supports a single asset (str), not a "
                "multi-asset sequence."
            )
        collection = _from_stac_solar_day(
            item_list, asset, patch_url, signer, DatasetCollection
        )
    elif isinstance(asset, str):
        hrefs = [_sign(_resolve_asset_href(item, asset)) for item in item_list]
        collection = DatasetCollection.from_files(hrefs, gdal_env=gdal_env)
    else:
        collection = _from_stac_multi_asset(
            item_list, list(asset), _sign, gdal_env, align, skip_missing, DatasetCollection
        )

    if target_grid is not None:
        collection = collection.align(target_grid)
    return collection


def _resolve_target_grid(
    like: Any,
    crs: int | str | None,
    resolution: float | None,
    bounds: Sequence[float] | None,
    anchor: str,
) -> Any:
    """Resolve the PC-2 grid-match arguments to a template Dataset (or None).

    Args:
        like: An existing :class:`~pyramids.dataset.Dataset` to match, or
            `None`.
        crs: Target CRS (with `resolution` + `bounds`) for an explicit grid.
        resolution: Target pixel size.
        bounds: Target `(minx, miny, maxx, maxy)` extent.
        anchor: Grid-snap rule (`"edge"` supported).

    Returns:
        The `like` Dataset, a freshly built template Dataset for an explicit
        grid, or `None` when no grid-match was requested.

    Raises:
        ValueError: `like` is combined with `crs`/`resolution`/`bounds`; the
            explicit-grid trio is given only partially; or `anchor` is
            unsupported.
    """
    explicit = (crs, resolution, bounds)
    if like is not None:
        if any(v is not None for v in explicit):
            raise ValueError(
                "like= is mutually exclusive with crs/resolution/bounds."
            )
        return like
    if all(v is None for v in explicit):
        return None
    if any(v is None for v in explicit):
        raise ValueError(
            "crs, resolution, and bounds must all be given together (or use "
            "like=)."
        )
    if anchor != "edge":
        raise ValueError(f"anchor must be 'edge', got {anchor!r}.")

    import math

    import numpy as np

    from pyramids.dataset.dataset import Dataset

    minx, miny, maxx, maxy = (float(v) for v in bounds)
    minx = math.floor(minx / resolution) * resolution
    miny = math.floor(miny / resolution) * resolution
    maxx = math.ceil(maxx / resolution) * resolution
    maxy = math.ceil(maxy / resolution) * resolution
    cols = max(int(round((maxx - minx) / resolution)), 1)
    rows = max(int(round((maxy - miny) / resolution)), 1)
    # Guard against an absurd request (e.g. a degrees/metres resolution mix-up)
    # so a typo raises a clear, actionable error instead of allocating a
    # multi-GB template and OOM-ing. The template carries only the target CRS +
    # geotransform + shape for align; its pixels are never read.
    n_pixels = rows * cols
    if n_pixels > _MAX_TEMPLATE_PIXELS:
        raise ValueError(
            f"target grid is {rows} x {cols} = {n_pixels:,} pixels, exceeding "
            f"the {_MAX_TEMPLATE_PIXELS:,}-pixel limit for an in-memory alignment "
            "template. Use a coarser resolution, a smaller bounds, or pass "
            "like=<Dataset> to match an existing grid."
        )
    return Dataset.create_from_array(
        np.zeros((rows, cols), dtype="float32"),
        top_left_corner=(minx, maxy),
        cell_size=resolution,
        epsg=crs,
    )


def _item_datetime(item: Any) -> _datetime_cls:
    """Return a STAC Item's datetime as a tz-aware :class:`datetime`.

    Reads `item.datetime` (pystac) or `properties["datetime"]` (raw JSON).

    Raises:
        ValueError: The item carries no datetime.
    """
    when = getattr(item, "datetime", None)
    if when is None:
        props = getattr(item, "properties", None)
        if props is None and isinstance(item, dict):
            props = item.get("properties")
        when = (props or {}).get("datetime")
    if when is None:
        raise ValueError(
            f"item {_item_id(item)} has no datetime; required for "
            "groupby='solar_day'."
        )
    if isinstance(when, str):
        when = _datetime_cls.fromisoformat(when.replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


def _item_centroid_lon(item: Any) -> float:
    """Return the longitude of an item's bbox centroid (0.0 when no bbox)."""
    bbox = getattr(item, "bbox", None)
    if bbox is None and isinstance(item, dict):
        bbox = item.get("bbox")
    if not bbox:
        return 0.0
    west, _s, east, _n = _horizontal_bounds(bbox)
    if west <= east:
        return (west + east) / 2.0
    # Antimeridian-crossing box (L3): average across the dateline by shifting
    # the eastern edge +360, then normalise the midpoint back to [-180, 180].
    mid = (west + east + 360.0) / 2.0
    return mid - 360.0 if mid > 180.0 else mid


def _solar_day(item: Any) -> str:
    """Return an item's solar-day label (ISO date).

    The UTC datetime is shifted by the centroid longitude (15°/hour) so a
    single overpass is not split across the UTC-midnight boundary, then reduced
    to its calendar date.
    """
    shifted = _item_datetime(item) + timedelta(
        hours=_item_centroid_lon(item) / 15.0
    )
    return shifted.date().isoformat()


def _item_id(item: Any) -> Any:
    """Best-effort item id for error messages."""
    iid = getattr(item, "id", None)
    if iid is None and isinstance(item, dict):
        iid = item.get("id", "?")
    return iid if iid is not None else "?"


def _from_stac_solar_day(
    item_list: list[Any],
    asset: str,
    patch_url: Callable[[str], str] | None,
    signer: Any,
    collection_cls: Any,
) -> DatasetCollection:
    """Mosaic same-solar-day items of one asset into one timestep each.

    Items are grouped by :func:`_solar_day`; each group's asset hrefs are
    mosaicked with ``merge_rasters(method="first")`` (the signer is applied
    there, so hrefs are not pre-signed here — only `patch_url` is). The per-day
    mosaics, in chronological order, back the returned collection.

    Args:
        item_list: The (filtered) STAC items.
        asset: The single asset key to mosaic.
        patch_url: Optional href rewriter applied before the merge's signer.
        signer: Optional signer (applied by `merge_rasters`).
        collection_cls: The :class:`DatasetCollection` class (cycle-free).

    Returns:
        DatasetCollection: One timestep per distinct solar day.

    Raises:
        ValueError: No items remain to group.
    """
    from pyramids.dataset.merge import merge_rasters

    if not item_list:
        raise ValueError("from_stac(groupby='solar_day') received no items.")

    groups: dict[str, list[str]] = defaultdict(list)
    for item in item_list:
        href = _resolve_asset_href(item, asset)
        if patch_url is not None:
            href = patch_url(href)
        groups[_solar_day(item)].append(href)

    out_dir = artifact_dir()
    per_day_paths: list[str] = []
    for day in sorted(groups):
        out_path = os.path.join(out_dir, f"{day}.tif")
        merge_rasters(groups[day], out_path, method="first", signer=signer)
        per_day_paths.append(out_path)

    return collection_cls.from_files(per_day_paths)


def _from_stac_multi_asset(
    item_list: list[Any],
    asset_keys: list[str],
    sign: Callable[[str], str],
    gdal_env: dict[str, str] | None,
    align: bool,
    skip_missing: bool,
    collection_cls: Any,
) -> DatasetCollection:
    """Stack multiple assets per item into a band axis, then time-stack them.

    For each item, the named assets are resolved, signed, and stacked
    band-wise into one multi-band GeoTIFF (band names = `asset_keys`) under a
    temporary directory; those per-item rasters then back the collection. See
    :func:`from_stac` for the parameter contract.

    Args:
        item_list: The (already filtered/capped) STAC items.
        asset_keys: Asset keys to stack, in band order.
        sign: The combined patch_url + signer.sign_href href rewriter.
        gdal_env: Signer GDAL config installed around the per-asset opens.
        align: Resample mismatched assets onto the first asset's grid.
        skip_missing: Drop items missing any requested asset instead of raising.
        collection_cls: The :class:`DatasetCollection` class (passed in to keep
            this helper import-cycle-free).

    Returns:
        DatasetCollection: One multi-band timestep per kept item.

    Raises:
        StacAssetError: An item lacks a requested asset and `skip_missing`
            is `False`.
        ValueError: No items remain after skipping.
    """
    # Lazy imports: cycle-break (Dataset) + reuse the shared env helper.
    from pyramids.base.remote import cloud_config_from_env
    from pyramids.dataset.dataset import Dataset

    out_dir = artifact_dir()
    per_item_paths: list[str] = []
    for idx, item in enumerate(item_list):
        try:
            hrefs = [sign(_resolve_asset_href(item, key)) for key in asset_keys]
        except StacAssetError:
            if skip_missing:
                continue
            raise
        out_path = os.path.join(out_dir, f"stac_item_{idx}.tif")
        with cloud_config_from_env(gdal_env):
            Dataset.from_band_files(
                hrefs, band_names=asset_keys, align=align, path=out_path
            )
        per_item_paths.append(out_path)

    if not per_item_paths:
        raise ValueError(
            "from_stac produced no items (all were missing a requested asset "
            "or filtered out)."
        )
    return collection_cls.from_files(per_item_paths)


DEFAULT_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Safety ceiling for an in-memory grid-match template. The template is a
# float32 raster (DatasetCollection.align adopts the template's dtype for its
# resampled output, so it must stay compatible with the source data — a smaller
# dtype would corrupt floats), i.e. ~1 GiB at this limit. Large enough for a
# full Sentinel-2 tile grid (~10980²) or a sizeable mosaic, small enough to turn
# a degrees/metres resolution mix-up into a clear error instead of an OOM.
_MAX_TEMPLATE_PIXELS = 250_000_000


def _utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing `(lon, lat)`.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.

    Returns:
        `326NN` (northern hemisphere) or `327NN` (southern) for UTM zone `NN`.

    Examples:
        - A point in the Italian Alps falls in UTM 32N:
            ```python
            >>> from pyramids.dataset._stac import _utm_epsg
            >>> _utm_epsg(11.0, 46.0)
            32632

            ```
        - A southern-hemisphere point uses the 327xx band:
            ```python
            >>> _utm_epsg(-58.0, -34.0)
            32721

            ```
    """
    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def _point_aoi_bbox(
    lat: float,
    lon: float,
    edge_size: int,
    resolution: float,
    units: str,
) -> tuple[int, tuple[float, float, float, float]]:
    """Compute the local-UTM EPSG and the 4326 search bbox for a point cube.

    The center `(lat, lon)` is reprojected to its local UTM, snapped to the
    `resolution` grid, and expanded to a square AOI of `edge_size` pixels
    (`units="px"`) or metres (`units="m"`); the UTM square is reprojected back
    to EPSG:4326 for the STAC search.

    Args:
        lat: Center latitude (degrees).
        lon: Center longitude (degrees).
        edge_size: Cube side length, in pixels (`units="px"`) or metres
            (`units="m"`).
        resolution: Pixel size in metres.
        units: `"px"` or `"m"`.

    Returns:
        A `(utm_epsg, bbox_4326)` tuple, with `bbox_4326` = `(w, s, e, n)`.

    Raises:
        ValueError: When `units` is not `"px"` or `"m"`.
    """
    if units not in ("px", "m"):
        raise ValueError(f"units must be 'px' or 'm', got {units!r}.")
    from pyproj import Transformer

    utm_epsg = _utm_epsg(lon, lat)
    to_utm = Transformer.from_crs(4326, utm_epsg, always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    cx = round(cx / resolution) * resolution
    cy = round(cy / resolution) * resolution
    half = (edge_size / 2.0) * resolution if units == "px" else edge_size / 2.0
    utm_bbox = (cx - half, cy - half, cx + half, cy + half)

    to_wgs = Transformer.from_crs(utm_epsg, 4326, always_xy=True)
    minx, miny, maxx, maxy = utm_bbox
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    lons, lats = [], []
    for x, y in corners:
        clon, clat = to_wgs.transform(x, y)
        lons.append(clon)
        lats.append(clat)
    return utm_epsg, (min(lons), min(lats), max(lons), max(lats))


def from_point(
    lat: float,
    lon: float,
    *,
    collection: str,
    bands: str | Sequence[str],
    start_date: str,
    end_date: str,
    edge_size: int,
    resolution: float,
    units: str = "px",
    stac: str = DEFAULT_STAC_URL,
    query: Any = None,
    signer: Any = None,
    align: bool = True,
) -> DatasetCollection:
    """Build a point-centred STAC cube (cubo-style convenience constructor).

    Composes :func:`pyramids.stac.search` (find items) and :func:`from_stac`
    (build the cube) around a point + edge-size + resolution. The center
    `(lat, lon)` is reprojected to its local UTM zone, snapped to the
    `resolution` grid, and expanded to a square AOI of `edge_size` pixels (or
    metres); that AOI (reprojected to EPSG:4326) drives the STAC search.

    .. note::
        The returned cube is on the matched assets' native grid clipped to the
        AOI — it is **not yet** resampled to an exact `edge_size`×`edge_size`
        local-UTM grid. Exact target-grid resampling arrives with the
        `geobox=`/`like=` grid match (PC-2). For now `from_point` is the
        convenience AOI + search + stack wrapper.

    Args:
        lat: Center latitude in degrees (EPSG:4326).
        lon: Center longitude in degrees (EPSG:4326).
        collection: STAC collection id to search.
        bands: A single asset key or a sequence (multi-asset band axis; see
            :func:`from_stac`).
        start_date: Search start (RFC 3339 / `YYYY-MM-DD`).
        end_date: Search end (RFC 3339 / `YYYY-MM-DD`).
        edge_size: Cube side length, in pixels (`units="px"`) or metres
            (`units="m"`).
        resolution: Pixel size in metres.
        units: `"px"` (default) or `"m"`.
        stac: STAC API root URL. Defaults to the Microsoft Planetary Computer
            (which needs a :class:`pyramids.stac.signers.PlanetaryComputerSigner`).
        query: Optional STAC `query` extension dict (e.g.
            `{"eo:cloud_cover": {"lt": 10}}`).
        signer: Optional signer, forwarded to both the search and the reads.
        align: Multi-asset resolution policy, forwarded to :func:`from_stac`.

    Returns:
        DatasetCollection: A time-stacked cube over the point AOI.

    Raises:
        ValueError: When `units` is invalid, or the search yields no items.
        OptionalPackageDoesNotExist: When `pystac-client` (the `[stac]` extra)
            is not installed.

    Examples:
        - Build a 64×64 px, 10 m Sentinel-2 cube around a point (network +
          a PC signer required):
            ```python
            >>> from pyramids.dataset import DatasetCollection  # doctest: +SKIP
            >>> from pyramids.stac import PlanetaryComputerSigner  # doctest: +SKIP
            >>> cube = DatasetCollection.from_point(  # doctest: +SKIP
            ...     lat=46.0, lon=11.0, collection="sentinel-2-l2a",
            ...     bands=["B04", "B03", "B02"],
            ...     start_date="2021-06-01", end_date="2021-06-10",
            ...     edge_size=64, resolution=10,
            ...     query={"eo:cloud_cover": {"lt": 10}},
            ...     signer=PlanetaryComputerSigner(),
            ... )

            ```
    """
    _utm_epsg_code, bbox_4326 = _point_aoi_bbox(lat, lon, edge_size, resolution, units)

    from pyramids.stac.search import search

    items = search(
        stac,
        collection,
        bbox=bbox_4326,
        datetime=f"{start_date}/{end_date}",
        query=query,
        signer=signer,
    )
    return from_stac(items, bands, signer=signer, align=align)


def _bbox_ring(bbox: Sequence[float]) -> dict[str, Any]:
    """Return a closed GeoJSON Polygon ring for `[minx, miny, maxx, maxy]`."""
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]
        ],
    }


def _footprint_4326(
    native_bbox: Sequence[float], epsg: int | None, precision: int
) -> tuple[dict[str, Any], list[float]]:
    """Reproject a native-CRS bbox ring to EPSG:4326 (geometry + bbox).

    Args:
        native_bbox: `[minx, miny, maxx, maxy]` in the dataset's CRS.
        epsg: The dataset's EPSG code, or a falsy value when it has no CRS.
        precision: Decimal places to round the reprojected coordinates to.

    Returns:
        A `(geometry, bbox)` tuple: a GeoJSON Polygon and a 4-element
        `[w, s, e, n]` bbox, both in EPSG:4326. A CRS-less dataset yields the
        world extent and emits a warning.
    """
    minx, miny, maxx, maxy = native_bbox
    if not epsg:
        warnings.warn(
            "Dataset has no CRS; setting the STAC geometry/bbox to the world "
            "extent (-180, -90, 180, 90).",
            stacklevel=3,
        )
        world = [-180.0, -90.0, 180.0, 90.0]
        return _bbox_ring(world), world

    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    if int(epsg) != 4326:
        src = osr.SpatialReference()
        src.ImportFromEPSG(int(epsg))
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4326)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(src, dst)
        corners = [
            (round(x, precision), round(y, precision))
            for x, y, *_ in transform.TransformPoints(corners)
        ]
    else:
        corners = [(round(x, precision), round(y, precision)) for x, y in corners]

    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    geometry = {"type": "Polygon", "coordinates": [[list(c) for c in corners]]}
    return geometry, [min(lons), min(lats), max(lons), max(lats)]


def to_stac_item(
    dataset: Any,
    item_id: str,
    *,
    asset_href: str,
    datetime: Any = None,
    start_datetime: Any = None,
    end_datetime: Any = None,
    asset_key: str = "data",
    asset_media_type: str | None = None,
    asset_roles: Sequence[str] = ("data",),
    with_proj: bool = True,
    with_raster: bool = True,
    precision: int = 6,
) -> dict[str, Any]:
    """Describe a pyramids :class:`~pyramids.dataset.Dataset` as a STAC Item dict.

    The inverse of :func:`from_stac`: emit a STAC-JSON Item (GeoJSON Feature)
    from a dataset's own metadata, with the `proj` and `raster` extensions
    populated. The footprint is the dataset's bounding rectangle reprojected to
    EPSG:4326 (the default footprint mode). pystac is **not** required —
    a plain dict is returned, ready to serialise or feed back into
    :func:`from_stac`.

    Args:
        dataset: A :class:`~pyramids.dataset.Dataset` (read via its public
            geo-properties: `epsg`, `geotransform`, `bbox`, `rows`, `columns`,
            `band_count`, `no_data_value`, `dtype`).
        item_id: The STAC Item id.
        asset_href: The href to record for the single data asset.
        datetime: The item datetime — a `datetime.datetime` (serialised via
            `isoformat()`) or an RFC 3339 string. When `None` **and** a
            `start_datetime`/`end_datetime` range is given, the `datetime`
            property is null and the range is written (the only STAC-valid way
            to have a null `datetime`). When `None` with no range, it defaults
            to the current UTC time so the Item is always valid.
        start_datetime: Optional range start (datetime or RFC 3339 string),
            written to `properties.start_datetime`.
        end_datetime: Optional range end, written to `properties.end_datetime`.
        asset_key: Key for the data asset (default `"data"`).
        asset_media_type: Optional media type for the asset (e.g.
            `"image/tiff; application=geotiff; profile=cloud-optimized"`).
        asset_roles: Roles for the asset (default `("data",)`).
        with_proj: Populate the `proj` extension (epsg/code/shape/transform/bbox)
            from the dataset grid.
        with_raster: Populate `raster:bands` (per-band `data_type` + `nodata`)
            on the asset.
        precision: Decimal places for the reprojected footprint coordinates.

    Returns:
        A STAC Item as a dict (a GeoJSON Feature with `properties`, `assets`,
        `bbox`, `geometry`, and `stac_extensions`).

    Examples:
        - Round-trip a dataset to a STAC Item dict (via the Dataset method):
            ```python
            >>> import numpy as np  # doctest: +SKIP
            >>> from pyramids.dataset import Dataset  # doctest: +SKIP
            >>> ds = Dataset.create_from_array(  # doctest: +SKIP
            ...     np.ones((4, 4), "float32"), top_left_corner=(0.0, 4.0),
            ...     cell_size=1.0, epsg=4326,
            ... )
            >>> item = ds.to_stac_item("scene-1", asset_href="s3://b/scene.tif")  # doctest: +SKIP
            >>> item["properties"]["proj:code"]  # doctest: +SKIP
            'EPSG:4326'

            ```
    """
    # Lazy import: pyramids.stac.* pulls _loader -> pyramids.dataset, which would
    # cycle if imported at module load (see _resolve_asset_href above).
    from pyramids.stac._extensions import geotransform_to_affine

    epsg = dataset.epsg
    native_bbox = list(dataset.bbox)
    geometry, bbox_4326 = _footprint_4326(native_bbox, epsg, precision)

    def _iso(value: Any) -> Any:
        return value.isoformat() if hasattr(value, "isoformat") else value

    # A null `datetime` is only STAC-valid alongside a start/end range. When the
    # caller gives neither, default to "now" so the Item is always valid
    # instead of silently emitting a null-datetime Feature.
    if datetime is None and not (start_datetime and end_datetime):
        datetime = _datetime_cls.now(timezone.utc)
    properties: dict[str, Any] = {"datetime": _iso(datetime)}
    if start_datetime is not None:
        properties["start_datetime"] = _iso(start_datetime)
    if end_datetime is not None:
        properties["end_datetime"] = _iso(end_datetime)
    stac_extensions: list[str] = []

    if with_proj and epsg:
        properties["proj:epsg"] = epsg
        properties["proj:code"] = f"EPSG:{epsg}"
        properties["proj:shape"] = [dataset.rows, dataset.columns]
        properties["proj:transform"] = geotransform_to_affine(dataset.geotransform)
        properties["proj:bbox"] = native_bbox
        stac_extensions.append(
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json"
        )

    asset: dict[str, Any] = {"href": asset_href, "roles": list(asset_roles)}
    if asset_media_type is not None:
        asset["type"] = asset_media_type

    if with_raster:
        nodata = dataset.no_data_value
        dtypes = dataset.dtype
        bands = []
        for i in range(dataset.band_count):
            band: dict[str, Any] = {"data_type": dtypes[i]}
            nd = nodata[i] if i < len(nodata) else None
            if nd is not None:
                band["nodata"] = nd
            bands.append(band)
        asset["raster:bands"] = bands
        stac_extensions.append(
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json"
        )

    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "geometry": geometry,
        "bbox": bbox_4326,
        "properties": properties,
        "assets": {asset_key: asset},
        "links": [],
        "stac_extensions": stac_extensions,
    }


__all__ = ["from_point", "from_stac", "to_stac_item"]
