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

import math
import os
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, timedelta
from datetime import datetime as _datetime_cls
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import shapely
from osgeo import osr
from pyproj import Transformer
from shapely.geometry import MultiPolygon, box, mapping, shape

from pyramids.base._artifacts import artifact_dir
from pyramids.base._errors import StacAssetError
from pyramids.base.crs import sr_from_user_input
from pyramids.base.georeference import GeoReference
from pyramids.dataset.grid import Grid
from pyramids.utm import utm_epsg

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

    Reads `item.bbox` through the shared duck-typed accessor
    (:func:`pyramids.stac._item.item_bbox`), so pystac Items and raw JSON dicts
    are handled identically here and in the STAC readers. Both the query `bbox`
    and the item bbox may be 2D (4-element) or 3D (6-element) — only the
    horizontal extent is compared (see :func:`_horizontal_bounds`). Longitude
    overlap is antimeridian-aware (a box with ``west > east`` is treated as
    wrapping the dateline). Items without a bbox are treated as intersecting
    (permissive default — the caller opted in to the bbox filter, not the item).
    """
    # Imported lazily to break the pyramids.dataset -> pyramids.stac ->
    # pyramids.dataset import cycle (see _resolve_asset_href above).
    from pyramids.stac._item import item_bbox

    box = item_bbox(item)
    if box is None:
        result = True
    else:
        q_west, q_south, q_east, q_north = _horizontal_bounds(bbox)
        i_west, i_south, i_east, i_north = _horizontal_bounds(box)
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
    grid: Grid | None = None,
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
        grid: Optional :class:`~pyramids.dataset.Grid` describing the target
            output grid; every timestep of the built cube is reprojected /
            resampled onto it (via :meth:`DatasetCollection.align`), guaranteeing
            pixel co-registration. `None` (default) or an empty `Grid()` keeps
            each timestep's native grid. Use `Grid(like=<Dataset>)` to match an
            existing grid, or `Grid(crs=..., resolution=..., bounds=...)` for an
            explicit one.

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

    target_grid = _resolve_target_grid(grid)

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
            item_list,
            list(asset),
            _sign,
            gdal_env,
            align,
            skip_missing,
            DatasetCollection,
        )

    if target_grid is not None:
        # align()'s default inplace=False (used here) always returns a new
        # collection; only inplace=True returns None.
        collection = cast("DatasetCollection", collection.align(target_grid))
    return collection


def _resolve_target_grid(grid: Grid | None) -> Any:
    """Resolve a :class:`~pyramids.dataset.Grid` to a template Dataset (or None).

    The mode invariants (``like`` xor the ``crs``/``resolution``/``bounds`` trio,
    the trio being all-or-nothing, and the ``anchor`` value) are validated by
    :meth:`Grid.__post_init__`, so this only has to build the template.

    Args:
        grid: A :class:`~pyramids.dataset.Grid`, or `None`.

    Returns:
        The `grid.like` Dataset, a freshly built template Dataset for an explicit
        grid, or `None` when no grid was requested (``None`` or an empty
        ``Grid()``).
    """
    if grid is None or grid.is_empty:
        return None
    if grid.like is not None:
        return grid.like
    # The trio is complete here (guaranteed by Grid.__post_init__).
    crs, resolution, bounds = grid.crs, grid.resolution, grid.bounds
    assert crs is not None and resolution is not None and bounds is not None

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
    return Dataset.from_array(
        np.zeros((rows, cols), dtype="float32"),
        geo_ref=GeoReference(
            top_left_corner=(minx, maxy), cell_size=resolution, epsg=crs
        ),
    )


def _item_datetime(item: Any) -> _datetime_cls:
    """Return a STAC Item's datetime as a tz-aware :class:`datetime`.

    Reads `item.datetime` (pystac) or `properties["datetime"]` (raw JSON, via
    the shared :func:`pyramids.stac._item.item_properties` accessor). An RFC
    3339 string is parsed; a naive datetime is assumed to be UTC.

    Args:
        item: A STAC Item (pystac object or raw dict).

    Returns:
        The item's acquisition time as a timezone-aware :class:`datetime`.

    Raises:
        ValueError: The item carries neither `.datetime` nor
            `properties["datetime"]` — the error names the item via the shared
            :func:`pyramids.stac._item.item_id` accessor.
    """
    # Imported lazily to break the pyramids.dataset -> pyramids.stac ->
    # pyramids.dataset import cycle (see _resolve_asset_href above).
    from pyramids.stac._item import item_id, item_properties

    when = getattr(item, "datetime", None)
    if when is None:
        when = item_properties(item).get("datetime")
    if when is None:
        raise ValueError(
            f"item {item_id(item)} has no datetime; required for groupby='solar_day'."
        )
    if isinstance(when, str):
        when = _datetime_cls.fromisoformat(when.replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when


def _item_centroid_lon(item: Any) -> float:
    """Return the longitude of an item's bbox centroid (0.0 when no bbox).

    The bbox is read through the shared
    :func:`pyramids.stac._item.item_bbox` accessor, so pystac Items and raw
    JSON dicts behave identically. A box crossing the antimeridian
    (``west > east``) is averaged across the dateline rather than through the
    prime meridian.

    Args:
        item: A STAC Item (pystac object or raw dict).

    Returns:
        The centroid longitude in degrees, normalised to ``[-180, 180]``, or
        `0.0` when the item carries no bbox (no solar-day shift is applied).
    """
    # Imported lazily to break the pyramids.dataset -> pyramids.stac ->
    # pyramids.dataset import cycle (see _resolve_asset_href above).
    from pyramids.stac._item import item_bbox

    box = item_bbox(item)
    if not box:
        centroid = 0.0
    else:
        west, _s, east, _n = _horizontal_bounds(box)
        if west <= east:
            centroid = (west + east) / 2.0
        else:
            # Antimeridian-crossing box (L3): average across the dateline by
            # shifting the eastern edge +360, then normalise the midpoint back
            # to [-180, 180].
            mid = (west + east + 360.0) / 2.0
            centroid = mid - 360.0 if mid > 180.0 else mid
    return centroid


def _solar_day(item: Any) -> str:
    """Return an item's solar-day label (ISO date).

    The UTC datetime is shifted by the centroid longitude (15°/hour) so a
    single overpass is not split across the UTC-midnight boundary, then reduced
    to its calendar date.
    """
    shifted = _item_datetime(item) + timedelta(hours=_item_centroid_lon(item) / 15.0)
    return shifted.date().isoformat()


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

    # collection_cls is always the real DatasetCollection class (passed by every
    # caller); typed Any here only to dodge the import cycle noted above.
    return cast("DatasetCollection", collection_cls.from_files(per_day_paths))


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
        with cloud_config_from_env(gdal_env, path=hrefs):
            Dataset.from_band_files(
                hrefs, band_names=asset_keys, align=align, path=out_path
            )
        per_item_paths.append(out_path)

    if not per_item_paths:
        raise ValueError(
            "from_stac produced no items (all were missing a requested asset "
            "or filtered out)."
        )
    # collection_cls is always the real DatasetCollection class (passed by every
    # caller); typed Any here only to dodge the import cycle noted above.
    return cast("DatasetCollection", collection_cls.from_files(per_item_paths))


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

    Thin private wrapper delegating to the public :func:`pyramids.utm.utm_epsg`, so
    the STAC point-cube path and the public helper stay in lock-step.

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
    return utm_epsg(lon, lat)


def _point_aoi_bbox(
    lat: float,
    lon: float,
    edge_size: int,
    resolution: float,
    units: str,
) -> tuple[int, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Compute the local-UTM EPSG, the UTM AOI, and the 4326 search bbox.

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
        A `(utm_epsg, utm_bbox, bbox_4326)` tuple: the local UTM EPSG code, the
        resolution-snapped AOI square `(minx, miny, maxx, maxy)` in that UTM CRS
        (the exact target grid), and the same square reprojected to EPSG:4326
        `(w, s, e, n)` for the STAC search.

    Raises:
        ValueError: When `units` is not `"px"` or `"m"`.
    """
    if units not in ("px", "m"):
        raise ValueError(f"units must be 'px' or 'm', got {units!r}.")
    epsg = _utm_epsg(lon, lat)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    cx = round(cx / resolution) * resolution
    cy = round(cy / resolution) * resolution
    half = (edge_size / 2.0) * resolution if units == "px" else edge_size / 2.0
    utm_bbox = (cx - half, cy - half, cx + half, cy + half)

    to_wgs = Transformer.from_crs(epsg, 4326, always_xy=True)
    minx, miny, maxx, maxy = utm_bbox
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    lons, lats = [], []
    for x, y in corners:
        clon, clat = to_wgs.transform(x, y)
        lons.append(clon)
        lats.append(clat)
    return epsg, utm_bbox, (min(lons), min(lats), max(lons), max(lats))


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

    The returned cube is resampled onto the exact `edge_size`×`edge_size`
    local-UTM target grid the AOI defines: `from_point` builds a
    :class:`~pyramids.dataset.Grid` (`crs` = the local UTM zone, `resolution`,
    `bounds` = the snapped UTM square) and forwards it to :func:`from_stac`, so
    every timestep is co-registered on that grid regardless of the assets'
    native CRS.

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
            (which needs an ``earthlens.stac.PlanetaryComputerSigner``).
        query: Optional STAC `query` extension dict (e.g.
            `{"eo:cloud_cover": {"lt": 10}}`).
        signer: Optional signer, forwarded to both the search and the reads.
        align: Multi-asset resolution policy, forwarded to :func:`from_stac`.

    Returns:
        DatasetCollection: A time-stacked cube over the point AOI, resampled
        onto the exact `edge_size`×`edge_size` local-UTM grid.

    Raises:
        ValueError: When `units` is invalid, or the search yields no items.
        OptionalPackageDoesNotExist: When `pystac-client` (the `[stac]` extra)
            is not installed.

    Examples:
        - Build a 64×64 px, 10 m Sentinel-2 cube around a point (network +
          a PC signer required):
            ```python
            >>> from pyramids.dataset import DatasetCollection  # doctest: +SKIP
            >>> from earthlens.stac import PlanetaryComputerSigner  # doctest: +SKIP
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
    utm_epsg, utm_bbox, bbox_4326 = _point_aoi_bbox(
        lat, lon, edge_size, resolution, units
    )

    from pyramids.stac.search import search

    items = search(
        stac,
        collection,
        bbox=bbox_4326,
        datetime=f"{start_date}/{end_date}",
        query=query,
        signer=signer,
    )
    # Resample every timestep onto the exact edge_size x edge_size local-UTM
    # target grid the AOI defines (PC-2), so the point cube is co-registered.
    grid = Grid(crs=utm_epsg, resolution=resolution, bounds=utm_bbox)
    return from_stac(items, bands, signer=signer, align=align, grid=grid)


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


def _transform_to_4326(
    points: Sequence[tuple[float, float]], epsg: int, precision: int
) -> list[tuple[float, float]]:
    """Reproject `(x, y)` pairs from `epsg` into lon/lat, rounded to `precision`.

    The single reprojection path for everything this module emits — the bbox
    ring and the valid-data footprint both go through it, so their axis order
    can never diverge.

    Args:
        points: The native-CRS coordinate pairs to reproject.
        epsg: The source EPSG code.
        precision: Decimal places to round the lon/lat output to.

    Returns:
        The rounded `(lon, lat)` pairs, in the input order.
    """
    if int(epsg) != 4326:
        # `sr_from_user_input` already stamps traditional axis order, which is
        # the whole reason both operands were being built by hand here.
        src = sr_from_user_input(int(epsg))
        dst = sr_from_user_input(4326)
        transform = osr.CoordinateTransformation(src, dst)
        out = [
            (round(x, precision), round(y, precision))
            for x, y, *_ in transform.TransformPoints([tuple(p) for p in points])
        ]
    else:
        out = [(round(x, precision), round(y, precision)) for x, y in points]
    return out


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
            "Cannot reproject the footprint to EPSG:4326 (the dataset has no "
            "EPSG code); setting the STAC geometry/bbox to the world extent "
            "(-180, -90, 180, 90).",
            stacklevel=3,
        )
        world = [-180.0, -90.0, 180.0, 90.0]
        return _bbox_ring(world), world

    ring = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    corners = _transform_to_4326(ring, int(epsg), precision)

    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    geometry = {"type": "Polygon", "coordinates": [[list(c) for c in corners]]}
    return geometry, [min(lons), min(lats), max(lons), max(lats)]


def _round_coords(value: Any, precision: int) -> Any:
    """Convert a GeoJSON coordinate tree to rounded, plain Python lists.

    Args:
        value: A coordinate tree (nested sequences) or a single number.
        precision: Decimal places to round every coordinate to.

    Returns:
        The same tree with `list` containers and rounded `float` leaves.
    """
    if isinstance(value, list | tuple):
        rounded: Any = [_round_coords(item, precision) for item in value]
    else:
        rounded = round(float(value), precision)
    return rounded


def _geojson_geometry(geom: Any, precision: int) -> dict[str, Any]:
    """Serialise a shapely geometry as a GeoJSON dict with rounded list coordinates.

    Args:
        geom: The shapely geometry to serialise (already in EPSG:4326).
        precision: Decimal places to round every coordinate to.

    Returns:
        A GeoJSON geometry dict (`type` + `coordinates`).
    """
    mapped = mapping(geom)
    return {
        "type": mapped["type"],
        "coordinates": _round_coords(mapped["coordinates"], precision),
    }


def _reproject_geometry_4326(geom: Any, epsg: int, precision: int) -> Any:
    """Reproject an arbitrary shapely geometry into EPSG:4326.

    Args:
        geom: The shapely geometry, in the CRS `epsg` describes.
        epsg: The source EPSG code.
        precision: Decimal places to round the lon/lat output to.

    Returns:
        The reprojected shapely geometry.
    """

    def _project(coords: np.ndarray) -> np.ndarray:
        pairs = _transform_to_4326(
            [(float(x), float(y)) for x, y in coords], epsg, precision
        )
        return np.asarray(pairs, dtype="float64").reshape(len(pairs), 2)

    return shapely.transform(geom, _project)


def _shift_lon(geom: Any, offset: float) -> Any:
    """Return `geom` with every longitude moved by `offset` degrees.

    Args:
        geom: A shapely geometry in lon/lat.
        offset: The longitude shift in degrees (e.g. `-360.0`).

    Returns:
        The shifted shapely geometry.
    """
    return shapely.transform(
        geom, lambda c: np.column_stack([c[:, 0] + offset, c[:, 1]])
    )


def _unwrap_lon(geom: Any) -> Any:
    """Lift negative longitudes into a continuous `[0, 360)` frame.

    Args:
        geom: A shapely geometry whose longitudes all sit in `[-180, 180]` but
            which wraps the antimeridian.

    Returns:
        The geometry with its eastern-hemisphere (negative) longitudes raised
        by 360 degrees, so the ring is continuous across the seam.
    """
    return shapely.transform(
        geom,
        lambda c: np.column_stack(
            [np.where(c[:, 0] < 0.0, c[:, 0] + 360.0, c[:, 0]), c[:, 1]]
        ),
    )


def _polygon_parts(geom: Any) -> list[Any]:
    """Return the non-empty polygonal parts of a possibly mixed geometry.

    Args:
        geom: Any shapely geometry (a clip can yield a collection).

    Returns:
        The `Polygon` parts, dropping empties and lower-dimension leftovers.
    """
    geoms = list(getattr(geom, "geoms", [geom]))
    return [g for g in geoms if g.geom_type == "Polygon" and not g.is_empty]


def _split_at_seam(
    geom: Any, precision: int
) -> tuple[dict[str, Any], list[float]] | None:
    """Cut a continuous-longitude geometry at the +180 meridian.

    Args:
        geom: A shapely geometry whose longitudes run continuously across the
            seam (so some of them exceed +180 or fall below -180).
        precision: Decimal places to round the emitted coordinates to.

    Returns:
        A `(geometry, bbox)` tuple whose geometry is a `MultiPolygon` and whose
        bbox follows the RFC 7946 antimeridian convention (`west > east`), or
        `None` when the geometry turns out to sit wholly on one side of the
        seam (nothing to split).
    """
    coords = shapely.get_coordinates(geom)
    miny, maxy = float(coords[:, 1].min()), float(coords[:, 1].max())
    minx, maxx = float(coords[:, 0].min()), float(coords[:, 0].max())
    pad = 1.0
    western = geom.intersection(box(minx - pad, miny - pad, 180.0, maxy + pad))
    eastern = geom.intersection(box(180.0, miny - pad, maxx + pad, maxy + pad))
    west_parts = _polygon_parts(western)
    east_parts = _polygon_parts(_shift_lon(eastern, -360.0))
    if not west_parts or not east_parts:
        result = None
    else:
        merged = MultiPolygon(west_parts + east_parts)
        west = min(shapely.get_coordinates(p)[:, 0].min() for p in west_parts)
        east = max(shapely.get_coordinates(p)[:, 0].max() for p in east_parts)
        bbox = [
            round(float(west), precision),
            round(miny, precision),
            round(float(east), precision),
            round(maxy, precision),
        ]
        result = (_geojson_geometry(merged, precision), bbox)
    return result


def _split_antimeridian(
    geometry: dict[str, Any], bbox: list[float], precision: int
) -> tuple[dict[str, Any], list[float]]:
    """Split an emitted EPSG:4326 footprint that crosses the antimeridian.

    GeoJSON cannot carry a ring that wraps +/-180, so a crossing footprint is
    cut at the seam into a two-part `MultiPolygon` and its bbox is emitted with
    `west > east` (RFC 7946 section 5.2) rather than collapsed to the whole
    globe. A footprint that does not cross is returned untouched, so the
    default (`footprint="bbox"`) output is unchanged.

    Crossing is detected from the emitted coordinates: longitudes outside
    `[-180, 180]` (a grid that simply runs past the seam), or a longitude span
    wider than 180 degrees (a reprojected scene whose corners landed on both
    sides of it). A geometry that reaches both -180 and +180 is treated as a
    global extent and left alone.

    Args:
        geometry: The emitted GeoJSON geometry in EPSG:4326.
        bbox: Its `[w, s, e, n]` bbox.
        precision: Decimal places to round re-emitted coordinates to.

    Returns:
        A `(geometry, bbox)` tuple — the inputs unchanged when there is no
        crossing, otherwise a `MultiPolygon` and a `west > east` bbox.
    """
    geom = shape(geometry)
    lons = shapely.get_coordinates(geom)[:, 0]
    west, east = float(lons.min()), float(lons.max())
    global_extent = west <= -180.0 and east >= 180.0
    crossing = not global_extent and (
        east > 180.0 or west < -180.0 or east - west > 180.0
    )
    result = (geometry, bbox)
    if crossing:
        if west >= -180.0 and east <= 180.0:
            geom = _unwrap_lon(geom)
        split = _split_at_seam(geom, precision)
        if split is not None:
            result = split
        elif east > 180.0 or west < -180.0:
            # Wholly beyond the seam (e.g. a grid at lon 181..185): wrap it back
            # into [-180, 180] as a plain polygon rather than leaving it invalid.
            shifted = _shift_lon(geom, -360.0 if west >= 180.0 else 360.0)
            coords = shapely.get_coordinates(shifted)
            result = (
                _geojson_geometry(shifted, precision),
                [
                    round(float(coords[:, 0].min()), precision),
                    round(float(coords[:, 1].min()), precision),
                    round(float(coords[:, 0].max()), precision),
                    round(float(coords[:, 1].max()), precision),
                ],
            )
    return result


def _data_footprint_4326(
    dataset: Any,
    epsg: int,
    precision: int,
    *,
    band: int,
    max_samples: int | None,
    simplify_tolerance: float | None,
    densify: float | None,
) -> tuple[dict[str, Any], list[float]]:
    """Build the valid-pixel footprint of a dataset in EPSG:4326.

    Uses :meth:`pyramids.dataset.Dataset.footprint` (the polygonised non-nodata
    mask, in the dataset CRS), densifies it in native units, reprojects it
    through the same path as the bbox ring, then optionally simplifies it in
    degrees. Multi-part coverage stays a `MultiPolygon` — it is never merged to
    a convex hull.

    Args:
        dataset: The dataset to footprint.
        epsg: The dataset's EPSG code.
        precision: Decimal places to round the reprojected coordinates to.
        band: Zero-based band index to footprint.
        max_samples: Approximate pixel budget for the mask read — accuracy
            traded for speed — or `None` for an exact read.
        simplify_tolerance: Douglas-Peucker tolerance **in degrees**, applied
            after reprojection, or `None` to keep every vertex.
        densify: Maximum segment length **in native CRS units**, applied before
            reprojection so long edges bend with the projection, or `None`/`0`
            to leave the edges alone.

    Returns:
        A `(geometry, bbox)` tuple in EPSG:4326. Falls back to the bbox ring
        (with a warning) when the band holds no valid pixels.
    """
    gdf = dataset.footprint(band=band, max_samples=max_samples)
    if gdf is None:
        warnings.warn(
            "footprint='data' found no valid pixels; falling back to the bbox "
            "footprint.",
            stacklevel=3,
        )
        result = _footprint_4326(list(dataset.bbox), epsg, precision)
    else:
        geom = gdf.geometry.union_all()
        if densify:
            geom = shapely.segmentize(geom, max_segment_length=densify)
        geom = _reproject_geometry_4326(geom, int(epsg), precision)
        if simplify_tolerance is not None:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        bounds = [round(float(v), precision) for v in geom.bounds]
        result = (_geojson_geometry(geom, precision), bounds)
    return result


def _to_iso(value: Any) -> Any:
    """Serialise a datetime-like value via ``isoformat()``; pass strings/None through."""
    return value.isoformat() if hasattr(value, "isoformat") else value


def _proj_fields(
    dataset: Any, epsg: Any, native_bbox: list, transform_fn: Any
) -> dict[str, Any]:
    """Build the ``proj`` extension property fields from the dataset grid."""
    return {
        "proj:epsg": epsg,
        "proj:code": f"EPSG:{epsg}",
        "proj:shape": [dataset.rows, dataset.columns],
        "proj:transform": transform_fn(dataset.geotransform),
        "proj:bbox": native_bbox,
    }


def _encode_nodata(nd: Any) -> Any:
    """Encode a nodata value the way the raster extension expects.

    A non-finite sentinel cannot be written as JSON, so the schema spells it as
    the strings `"nan"`, `"inf"` or `"-inf"`. A finite sentinel is passed
    through **as it came in**, so existing output is unchanged.

    Args:
        nd: The band's nodata value.

    Returns:
        The sentinel itself when finite, or its string spelling when not.
    """
    try:
        value = float(nd)
    except (TypeError, ValueError):
        encoded: Any = nd
    else:
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "inf" if value > 0 else "-inf"
        else:
            encoded = nd
    return encoded


def _band_statistics(
    dataset: Any, index: int, approx_ok: bool
) -> dict[str, float] | None:
    """Build a raster-extension `statistics` object for one band.

    Args:
        dataset: The dataset to read statistics from.
        index: Zero-based band index.
        approx_ok: Let GDAL answer from overviews / a subsample.

    Returns:
        `{minimum, maximum, mean, stddev}` in physical units, or `None` when
        GDAL cannot compute them (a band with no valid pixels), in which case a
        warning is emitted.
    """
    try:
        row = dataset.stats(band=index, approx_ok=approx_ok).iloc[0]
    except RuntimeError:
        warnings.warn(
            f"band {index} has no valid pixels; omitting raster statistics",
            stacklevel=3,
        )
        stats = None
    else:
        stats = {
            "minimum": float(row["min"]),
            "maximum": float(row["max"]),
            "mean": float(row["mean"]),
            "stddev": float(row["std"]),
        }
    return stats


def _band_histogram(dataset: Any, index: int, bins: int) -> dict[str, Any] | None:
    """Build a raster-extension `histogram` object for one band.

    Args:
        dataset: The dataset to read the histogram from.
        index: Zero-based band index.
        bins: Number of buckets to ask for.

    Returns:
        `{count, min, max, buckets}` with `count` the number of buckets (the
        schema's meaning), or `None` when GDAL cannot bucket the band — a
        constant or all-nodata band has no range to split — in which case a
        warning is emitted.
    """
    try:
        counts, edges = dataset.get_histogram(band=index, bins=bins)
    except RuntimeError:
        warnings.warn(
            f"band {index} has no value range; omitting the raster histogram",
            stacklevel=3,
        )
        histogram = None
    else:
        bounds = [float(v) for edge in edges for v in edge]
        histogram = {
            "count": len(counts),
            "min": min(bounds),
            "max": max(bounds),
            "buckets": [int(c) for c in counts],
        }
    return histogram


def _raster_bands(
    dataset: Any,
    *,
    with_stats: bool = False,
    with_histogram: bool = False,
    histogram_bins: int = 10,
    stats_approx_ok: bool = True,
) -> list[dict[str, Any]]:
    """Build the ``raster:bands`` list for an asset.

    Always emits per-band `data_type` and (when set) `nodata`, plus `scale` /
    `offset` for a band whose CF packing is non-identity. `statistics` and
    `histogram` are opt-in, since both ask GDAL to look at pixels.

    Args:
        dataset: The dataset whose bands are described.
        with_stats: Add a `statistics` object per band.
        with_histogram: Add a `histogram` object per band.
        histogram_bins: Number of histogram buckets when `with_histogram`.
        stats_approx_ok: Let GDAL answer statistics approximately.

    Returns:
        One dict per band, in band order.
    """
    nodata = dataset.no_data_value
    dtypes = dataset.dtype
    scales = dataset.scale
    offsets = dataset.offset
    bands: list[dict[str, Any]] = []
    for i in range(dataset.band_count):
        band: dict[str, Any] = {"data_type": dtypes[i]}
        nd = nodata[i] if i < len(nodata) else None
        if nd is not None:
            band["nodata"] = _encode_nodata(nd)
        if i < len(scales) and scales[i] not in (None, 1.0):
            band["scale"] = float(scales[i])
        if i < len(offsets) and offsets[i] not in (None, 0.0):
            band["offset"] = float(offsets[i])
        if with_stats:
            stats = _band_statistics(dataset, i, stats_approx_ok)
            if stats is not None:
                band["statistics"] = stats
        if with_histogram:
            histogram = _band_histogram(dataset, i, histogram_bins)
            if histogram is not None:
                band["histogram"] = histogram
        bands.append(band)
    return bands


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
    with_stats: bool = False,
    with_histogram: bool = False,
    histogram_bins: int = 10,
    with_eo: bool = False,
    stats_approx_ok: bool = True,
    footprint: str = "bbox",
    footprint_band: int = 0,
    footprint_max_samples: int | None = None,
    simplify_tolerance: float | None = None,
    densify: float | None = None,
    precision: int = 6,
) -> dict[str, Any]:
    """Describe a pyramids :class:`~pyramids.dataset.Dataset` as a STAC Item dict.

    The inverse of :func:`from_stac`: emit a STAC-JSON Item (GeoJSON Feature)
    from a dataset's own metadata, with the `proj` and `raster` extensions
    populated. The footprint is the dataset's bounding rectangle reprojected to
    EPSG:4326 (the default footprint mode), or — with `footprint="data"` — the
    polygonised extent of its valid (non-nodata) pixels. Either way a footprint
    that crosses the antimeridian is split into a `MultiPolygon` and gets a
    `west > east` bbox. pystac is **not** required — a plain dict is returned,
    ready to serialise or feed back into :func:`from_stac`.

    The band-metadata and footprint keywords are all opt-in: with none of them
    passed the emitted Item is exactly what earlier versions produced.

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
        with_raster: Populate `raster:bands` (per-band `data_type` + `nodata`,
            plus `scale`/`offset` for a CF-packed band) on the asset.
        with_stats: Add a per-band `statistics` object (`minimum`, `maximum`,
            `mean`, `stddev`, in physical units) to `raster:bands`. A band with
            no valid pixels is skipped with a warning.
        with_histogram: Add a per-band `histogram` object (`count`, `min`,
            `max`, `buckets`) to `raster:bands`. A band with no value range
            (constant or all-nodata) is skipped with a warning.
        histogram_bins: Number of histogram buckets when `with_histogram`.
        with_eo: Add `eo:bands` (band names) to the asset and the `eo` schema
            to `stac_extensions`.
        stats_approx_ok: Let GDAL answer `with_stats` from overviews or a
            subsample (fast); pass `False` for exact figures.
        footprint: `"bbox"` (default) for the dataset's bounding rectangle, or
            `"data"` for the polygonised extent of its valid pixels. A
            CRS-less dataset always uses the bbox path.
        footprint_band: Zero-based band to footprint when `footprint="data"`.
        footprint_max_samples: Approximate pixel budget for the valid-pixel
            mask — accuracy traded for speed — or `None` for an exact read.
        simplify_tolerance: Douglas-Peucker tolerance **in degrees**, applied
            to the data footprint after reprojection, or `None` to keep every
            vertex.
        densify: Maximum segment length **in native CRS units**, applied to the
            data footprint before reprojection so long edges follow the
            projection's curvature; `None`/`0` leaves the edges alone.
        precision: Decimal places for the reprojected footprint coordinates.

    Returns:
        A STAC Item as a dict (a GeoJSON Feature with `properties`, `assets`,
        `bbox`, `geometry`, and `stac_extensions`).

    Raises:
        ValueError: `footprint` is neither `"bbox"` nor `"data"`.

    Examples:
        - Round-trip a dataset to a STAC Item dict (via the Dataset method):
            ```python
            >>> import numpy as np  # doctest: +SKIP
            >>> from pyramids.dataset import Dataset, GeoReference  # doctest: +SKIP
            >>> ds = Dataset.from_array(  # doctest: +SKIP
            ...     np.ones((4, 4), "float32"),
            ...     geo_ref=GeoReference(
            ...         top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326
            ...     ),
            ... )
            >>> item = ds.to_stac_item("scene-1", asset_href="s3://b/scene.tif")  # doctest: +SKIP
            >>> item["properties"]["proj:code"]  # doctest: +SKIP
            'EPSG:4326'

            ```
    """
    # Lazy import: pyramids.stac.* pulls _loader -> pyramids.dataset, which would
    # cycle if imported at module load (see _resolve_asset_href above).
    from pyramids.stac._extensions import geotransform_to_affine

    # `dataset.epsg` is None for a real CRS with no EPSG authority code (e.g.
    # geostationary) and reports None for a raster with no
    # projection at all. Either way a falsy `epsg` here (empty `dataset.crs`, or a
    # WKT-only CRS whose `epsg` is None) makes the world-bbox branch fire and the
    # proj:epsg field be omitted; the WKT stays available on `dataset.crs`.
    if footprint not in ("bbox", "data"):
        raise ValueError(f"footprint must be 'bbox' or 'data', got {footprint!r}.")

    epsg = dataset.epsg if dataset.crs else None
    native_bbox = list(dataset.bbox)
    if footprint == "data" and epsg:
        geometry, bbox_4326 = _data_footprint_4326(
            dataset,
            int(epsg),
            precision,
            band=footprint_band,
            max_samples=footprint_max_samples,
            simplify_tolerance=simplify_tolerance,
            densify=densify,
        )
    else:
        # "bbox" mode, or a CRS-less dataset (which falls through to the
        # world-extent branch inside `_footprint_4326`).
        geometry, bbox_4326 = _footprint_4326(native_bbox, epsg, precision)
    geometry, bbox_4326 = _split_antimeridian(geometry, bbox_4326, precision)

    # A null `datetime` is only STAC-valid alongside a start/end range. When the
    # caller gives neither, default to "now" so the Item is always valid
    # instead of silently emitting a null-datetime Feature.
    if datetime is None and not (start_datetime and end_datetime):
        datetime = _datetime_cls.now(UTC)
    properties: dict[str, Any] = {"datetime": _to_iso(datetime)}
    if start_datetime is not None:
        properties["start_datetime"] = _to_iso(start_datetime)
    if end_datetime is not None:
        properties["end_datetime"] = _to_iso(end_datetime)
    stac_extensions: list[str] = []

    if with_proj and epsg:
        properties.update(
            _proj_fields(dataset, epsg, native_bbox, geotransform_to_affine)
        )
        stac_extensions.append(
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json"
        )

    asset: dict[str, Any] = {"href": asset_href, "roles": list(asset_roles)}
    if asset_media_type is not None:
        asset["type"] = asset_media_type

    if with_raster:
        asset["raster:bands"] = _raster_bands(
            dataset,
            with_stats=with_stats,
            with_histogram=with_histogram,
            histogram_bins=histogram_bins,
            stats_approx_ok=stats_approx_ok,
        )
        stac_extensions.append(
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json"
        )

    if with_eo:
        asset["eo:bands"] = [{"name": name} for name in dataset.band_names]
        stac_extensions.append(
            "https://stac-extensions.github.io/eo/v1.1.0/schema.json"
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
