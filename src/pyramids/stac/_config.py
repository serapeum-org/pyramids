"""Caller-supplied asset metadata, band aliases, and the read-side materialiser.

Two STAC read features meet here, because both end in the same place — a
writable copy of the opened asset:

* **`rescale`** (STAC-04) turns stored counts into physical units using the
  asset's `raster:bands` `scale` / `offset`.
* **`cfg`** (STAC-07) supplies per-asset metadata a thin catalog omits
  (`data_type`, `nodata`, `unit`) and maps **band aliases** (`"rededge"` ->
  `"B05"`) to real asset keys, mirroring odc-stac's `stac_cfg`.

Why they share code: :func:`pyramids.stac.load_asset` opens an asset
**read-only** (a remote `/vsicurl` COG cannot be opened for write), and the
`Dataset.scale` / `Dataset.offset` / `Dataset.no_data_value` /
`Dataset.band_units` setters all raise
:class:`~pyramids.base._errors.ReadOnlyError` on a read-only on-disk handle. So
neither feature can stamp anything onto the handle it was given; both have to
build a new, writable in-memory dataset instead. :func:`materialise` is that one
rebuild, and :class:`AssetOverrides` is the single description of what to apply.

**The config schema** is a plain nested dict, keyed by collection id:

```python
cfg = {
    "sentinel-2-l2a": {
        "assets": {
            "*": {"data_type": "uint16", "nodata": 0, "unit": "1"},
            "SCL": {"data_type": "uint8", "nodata": 0},
        },
        "aliases": {"red": "B04", "rededge": "B05"},
        "warnings": "ignore",
    },
}
```

A `"*"` collection key supplies defaults for every collection (and is the only
section that applies to a bare asset dict, which carries no collection id).
Precedence, lowest first: `cfg["*"]["assets"]["*"]`, `cfg["*"]["assets"][key]`,
`cfg[collection]["assets"]["*"]`, `cfg[collection]["assets"][key]`.

There is deliberately **no `scale` / `offset`** in `cfg` — packing comes from the
item's own `raster:bands` via `rescale=True`, which keeps the schema
odc-compatible.

**Overrides fill gaps; they do not replace what the asset already declares.** A
configured `nodata` is applied only to bands the opened raster reports no
no-data value for, and a configured `unit` only to bands carrying no unit
label; a skipped override warns with :class:`AssetMetadataWarning` unless the
collection sets `warnings: "ignore"`. `data_type` is the exception: a raster
always has a dtype, so a configured one is applied whenever it differs from the
native dtype (an explicit cast of the stored counts).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyramids.base._utils import _is_identity_packing, apply_unpack
from pyramids.base.georeference import GeoReference
from pyramids.dataset.dataset import Dataset
from pyramids.stac._extensions import parse_number, read_extension_metadata

_WILDCARD = "*"

OVERRIDE_KEYS = ("data_type", "nodata", "unit")
"""The per-asset keys :func:`resolve_asset_metadata` results are read for.

Mirrors odc-stac's `ConversionConfig` minus `scale` / `offset`, which belong to
the item's `raster:bands` (see `rescale`) rather than to the caller's config.
"""


class AssetMetadataWarning(UserWarning):
    """A configured override was not applied because the asset already declares it.

    Emitted as a warning rather than raised because the config is a gap-filler:
    an item that turns out to carry the metadata is the good case, and the read
    should still succeed. Silence it per collection with `warnings: "ignore"`.
    """


def item_collection_id(item: Any) -> str | None:
    """Return a STAC Item's collection id, or `None` when it declares none.

    Duck-typed like the rest of the STAC readers: `item.collection_id`
    (pystac) then `item["collection"]` (raw STAC JSON). A bare asset carries
    neither and answers `None`, which is why `cfg` supports a `"*"` collection
    key.

    Args:
        item: A STAC Item (pystac object or raw dict), or an Asset.

    Returns:
        The collection id as a string, or `None`.

    Examples:
        - Raw STAC JSON carries it as the `collection` member:
            ```python
            >>> from pyramids.stac._config import item_collection_id
            >>> item_collection_id({"collection": "sentinel-2-l2a", "assets": {}})
            'sentinel-2-l2a'

            ```
        - A bare asset dict declares none:
            ```python
            >>> item_collection_id({"href": "a.tif"}) is None
            True

            ```
    """
    value = getattr(item, "collection_id", None)
    if value is None and isinstance(item, Mapping):
        value = item.get("collection")
    return None if value is None else str(value)


def _sections(cfg: Any, collection_id: str | None) -> list[Mapping[str, Any]]:
    """Return the `cfg` sections that apply, least specific first.

    Args:
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The item's collection id, or `None`.

    Returns:
        The `"*"` section followed by the collection's own section, skipping
        either when absent or not a mapping.
    """
    keys: list[str] = [_WILDCARD]
    if collection_id is not None and collection_id != _WILDCARD:
        keys.append(collection_id)
    sections: list[Mapping[str, Any]] = []
    if isinstance(cfg, Mapping):
        for key in keys:
            section = cfg.get(key)
            if isinstance(section, Mapping):
                sections.append(section)
    return sections


def resolve_asset_metadata(
    cfg: Any, collection_id: str | None, asset_key: str | None
) -> dict[str, Any]:
    """Merge the per-collection `'*'` asset defaults with a per-asset override.

    Precedence (highest last): the `"*"` collection's `"*"` asset defaults, its
    named-asset entry, the collection's own `"*"` defaults, the collection's
    named-asset entry. The result **fills gaps** in the STAC metadata — it is
    :func:`materialise` that decides whether a value is actually missing.

    Args:
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The item's collection id, or `None` to read the `"*"`
            section only.
        asset_key: The asset key being read, or `None` for a bare asset (only
            the `"*"` asset defaults then apply).

    Returns:
        The merged mapping, empty when nothing is configured. Keys outside
        :data:`OVERRIDE_KEYS` are kept verbatim but never applied.

    Examples:
        - A named asset's entry wins over the `'*'` defaults:
            ```python
            >>> from pyramids.stac._config import resolve_asset_metadata
            >>> cfg = {"c": {"assets": {"*": {"nodata": 0, "data_type": "uint16"},
            ...                         "SCL": {"data_type": "uint8"}}}}
            >>> resolve_asset_metadata(cfg, "c", "SCL") == {
            ...     "nodata": 0, "data_type": "uint8"
            ... }
            True
            >>> resolve_asset_metadata(cfg, "c", "B02") == {
            ...     "nodata": 0, "data_type": "uint16"
            ... }
            True

            ```
        - An unconfigured collection resolves to nothing:
            ```python
            >>> resolve_asset_metadata(cfg, "other", "x")
            {}
            >>> resolve_asset_metadata(None, "c", "SCL")
            {}

            ```
    """
    merged: dict[str, Any] = {}
    for section in _sections(cfg, collection_id):
        assets = section.get("assets")
        if not isinstance(assets, Mapping):
            continue
        for key in (_WILDCARD, asset_key):
            entry = assets.get(key) if key is not None else None
            if isinstance(entry, Mapping):
                merged.update(entry)
    return merged


def resolve_alias(cfg: Any, collection_id: str | None, name: str) -> str:
    """Map a band alias to its real asset key, or return `name` unchanged.

    Reads `cfg[collection]["aliases"]`, falling back to `cfg["*"]["aliases"]`,
    so a caller can ask for `"rededge"` and read the asset the catalog calls
    `"B05"`. Must run **before** href resolution, or the real asset key is
    never looked up.

    Args:
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The item's collection id, or `None`.
        name: The asset key or alias the caller asked for.

    Returns:
        The aliased asset key when one is configured, else `name`.

    Examples:
        - A configured alias resolves; anything else passes through:
            ```python
            >>> from pyramids.stac._config import resolve_alias
            >>> cfg = {"c": {"aliases": {"red": "B04"}}}
            >>> resolve_alias(cfg, "c", "red")
            'B04'
            >>> resolve_alias(cfg, "c", "green")
            'green'
            >>> resolve_alias(None, "c", "red")
            'red'

            ```
    """
    resolved = name
    for section in _sections(cfg, collection_id):
        aliases = section.get("aliases")
        if isinstance(aliases, Mapping) and name in aliases:
            resolved = str(aliases[name])
    return resolved


def warnings_ignored(cfg: Any, collection_id: str | None) -> bool:
    """Return whether this collection asks for metadata warnings to be silenced.

    Args:
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The item's collection id, or `None`.

    Returns:
        `True` when the collection (or the `"*"` section) sets
        `warnings: "ignore"`.

    Examples:
        - Only the configured collection is silenced:
            ```python
            >>> from pyramids.stac._config import warnings_ignored
            >>> cfg = {"c": {"warnings": "ignore"}}
            >>> warnings_ignored(cfg, "c"), warnings_ignored(cfg, "d")
            (True, False)

            ```
    """
    return any(
        section.get("warnings") == "ignore" for section in _sections(cfg, collection_id)
    )


def band_packing(raster_bands: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read per-band `scale` / `offset` out of a STAC `raster:bands` list.

    Values go through :func:`pyramids.stac._extensions.parse_number`, so the
    `raster` extension's `"nan"` / `"inf"` spellings and numeric strings are
    honoured and an unparseable entry falls back to the identity.

    Args:
        raster_bands: The verbatim `raster:bands` list, or `None`.

    Returns:
        A `(scales, offsets)` pair of equal-length tuples — empty when
        `raster_bands` is missing or empty.

    Examples:
        - A one-band packed asset:
            ```python
            >>> from pyramids.stac._config import band_packing
            >>> band_packing([{"scale": 0.01, "offset": 5}])
            ((0.01,), (5.0,))

            ```
        - A band declaring neither is the identity:
            ```python
            >>> band_packing([{"nodata": 0}])
            ((1.0,), (0.0,))
            >>> band_packing(None)
            ((), ())

            ```
    """
    scales: list[float] = []
    offsets: list[float] = []
    for band in raster_bands or ():
        entry = band if isinstance(band, Mapping) else {}
        scales.append(float(parse_number(entry.get("scale"), 1.0)))
        offsets.append(float(parse_number(entry.get("offset"), 0.0)))
    return tuple(scales), tuple(offsets)


@dataclass(frozen=True)
class AssetOverrides:
    """What a STAC read should apply to an asset after opening it.

    The single currency between `rescale` (which fills `scales` / `offsets`
    from the item's `raster:bands`) and `cfg` (which fills the rest), so
    :func:`materialise` has one thing to apply and one place to decide whether
    a rebuild is needed at all.

    Attributes:
        scales: Per-band multiplicative packing factors, in band order.
        offsets: Per-band additive packing offsets, in band order.
        no_data_value: A configured no-data value, or `None`.
        data_type: A configured numpy dtype name, or `None`.
        unit: A configured unit label, or `None`.
        quiet: Suppress :class:`AssetMetadataWarning` (the collection's
            `warnings: "ignore"`).
    """

    scales: tuple[float, ...] = ()
    offsets: tuple[float, ...] = ()
    no_data_value: float | None = None
    data_type: str | None = None
    unit: str | None = None
    quiet: bool = False

    def packing(self, band: int) -> tuple[float, float]:
        """Return the `(scale, offset)` pair for one band index.

        A band beyond the declared `raster:bands` gets the identity, so a
        catalog that documents only the first band of a multi-band asset leaves
        the rest untouched instead of borrowing band 0's factor.

        Args:
            band: Zero-based band index.

        Returns:
            The `(scale, offset)` pair to apply to that band.
        """
        scale = self.scales[band] if band < len(self.scales) else 1.0
        offset = self.offsets[band] if band < len(self.offsets) else 0.0
        return scale, offset

    @property
    def rescales(self) -> bool:
        """Whether any band declares a non-identity `scale` / `offset`."""
        # Not a zip over the two tuples: a declaration carrying only one of the
        # pair leaves the other empty, and zip would truncate it away.
        return any(
            not _is_identity_packing(*self.packing(index))
            for index in range(max(len(self.scales), len(self.offsets)))
        )

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to apply, so the opened handle can be kept."""
        return not (
            self.rescales
            or self.no_data_value is not None
            or self.data_type is not None
            or self.unit is not None
        )


def _configured_dtype(configured: Mapping[str, Any]) -> str | None:
    """Validate and return a configured `data_type`, or `None`.

    Args:
        configured: The merged per-asset config.

    Returns:
        The dtype name, or `None` when none is configured.

    Raises:
        ValueError: `data_type` is not a name numpy understands.
    """
    name = configured.get("data_type")
    if name is not None:
        try:
            np.dtype(name)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cfg data_type {name!r} is not a numpy dtype name; use e.g. "
                "'uint8', 'int16', 'uint16', 'float32'."
            ) from exc
    return None if name is None else str(name)


def resolve_overrides(
    item_or_asset: Any,
    asset_key: str | None = None,
    *,
    rescale: bool = False,
    cfg: Any = None,
    collection_id: str | None = None,
) -> AssetOverrides:
    """Collect everything a read should apply to one asset.

    Reads the item's `raster:bands` packing (only when `rescale` is asked for)
    and the caller's `cfg` entry for the asset. No file is opened: this is pure
    metadata, and :func:`materialise` is what touches pixels.

    Args:
        item_or_asset: A STAC Item (pystac object or raw dict), or an Asset.
        asset_key: Asset name when passing an Item; `None` for an Asset. Must
            already be alias-resolved (see :func:`resolve_alias`).
        rescale: Read the asset's `raster:bands` `scale` / `offset`.
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The collection the item belongs to; `None` reads only
            the `"*"` section of `cfg`.

    Returns:
        The resolved :class:`AssetOverrides` (`is_empty` when nothing applies).

    Raises:
        ValueError: `cfg` configures a `data_type` numpy does not know.
        StacAssetError: `asset_key` is given but absent from the item.

    Examples:
        - `rescale` picks the packing out of `raster:bands`:
            ```python
            >>> from pyramids.stac._config import resolve_overrides
            >>> asset = {"href": "a.tif", "raster:bands": [{"scale": 0.01}]}
            >>> overrides = resolve_overrides(asset, rescale=True)
            >>> overrides.packing(0), overrides.rescales
            ((0.01, 0.0), True)

            ```
        - Without `rescale`, and without `cfg`, there is nothing to apply:
            ```python
            >>> resolve_overrides(asset).is_empty
            True

            ```
    """
    scales: tuple[float, ...] = ()
    offsets: tuple[float, ...] = ()
    if rescale:
        metadata = read_extension_metadata(item_or_asset, asset_key)
        scales, offsets = band_packing(metadata.get("raster_bands"))
    configured = resolve_asset_metadata(cfg, collection_id, asset_key)
    unit = configured.get("unit")
    return AssetOverrides(
        scales=scales,
        offsets=offsets,
        no_data_value=parse_number(configured.get("nodata")),
        data_type=_configured_dtype(configured),
        unit=None if unit is None else str(unit),
        quiet=warnings_ignored(cfg, collection_id),
    )


def _skipped(name: str, value: Any, declared: Any, quiet: bool) -> None:
    """Warn that a configured override was already declared by the asset.

    Args:
        name: The override key (`"nodata"` / `"unit"`).
        value: The configured value that was not applied.
        declared: What the opened raster declares instead.
        quiet: Say nothing (the collection's `warnings: "ignore"`).
    """
    if not quiet:
        warnings.warn(
            f"cfg {name}={value!r} was not applied: the asset already declares "
            f"{declared!r}. Overrides fill gaps in the STAC metadata rather than "
            "replacing a value the asset carries.",
            AssetMetadataWarning,
            stacklevel=3,
        )


def _nodata_targets(dataset: Any, overrides: AssetOverrides) -> list[Any]:
    """Resolve the no-data value each band should end up declaring.

    The configured value fills only the bands that report none; a band that
    already declares one keeps it (and the skip is reported once).

    Args:
        dataset: The opened raster.
        overrides: The resolved overrides.

    Returns:
        One no-data value per band (possibly `None`), in band order.
    """
    native = list(dataset.no_data_value)
    configured = overrides.no_data_value
    resolved = list(native)
    if configured is not None:
        present = [value for value in native if value is not None]
        if present:
            _skipped("nodata", configured, present[0], overrides.quiet)
        resolved = [configured if value is None else value for value in native]
    return resolved


def _unit_targets(dataset: Any, overrides: AssetOverrides) -> list[str] | None:
    """Resolve the unit label each band should end up carrying, or `None`.

    Args:
        dataset: The opened raster.
        overrides: The resolved overrides.

    Returns:
        One unit label per band when a configured unit fills at least one gap,
        else `None` (nothing to stamp).
    """
    targets: list[str] | None = None
    if overrides.unit is not None:
        native = [str(unit or "") for unit in dataset.band_units]
        labelled = [unit for unit in native if unit]
        if labelled:
            _skipped("unit", overrides.unit, labelled[0], overrides.quiet)
        resolved = [unit or overrides.unit for unit in native]
        targets = None if resolved == native else resolved
    return targets


def _band_first(dataset: Any) -> Any:
    """Read a raster's stored counts as a band-first masked array.

    Reads with `unpack=False` so the counts are raw (any packing the *file*
    declares is deliberately left for the caller to decide about) and
    `masked=True` so the declared no-data sentinels are masked **before** any
    arithmetic touches them.

    Args:
        dataset: The opened raster.

    Returns:
        A `(bands, rows, columns)` masked array of stored values.
    """
    stored = np.ma.asanyarray(dataset.read_array(unpack=False, masked=True))
    # np.ma.atleast_3d would append the new axis, not prepend it.
    return stored[np.newaxis, :, :] if stored.ndim == 2 else stored


def _collapse(values: list[Any]) -> Any:
    """Return a single value when every band agrees, else the per-band list.

    `Dataset.from_array` takes either, and a scalar keeps the common
    single-band case readable in tracebacks and tests.

    Args:
        values: One value per band.

    Returns:
        The shared value, or `values` unchanged.
    """
    return values[0] if len(set(map(repr, values))) == 1 else values


def materialise(
    dataset: Any, overrides: AssetOverrides, *, path: str | None = None
) -> Any:
    """Return a writable raster with `overrides` applied, or `dataset` untouched.

    The one rebuild shared by `rescale` (STAC-04) and `cfg` (STAC-07), because
    both are blocked by the same thing: the asset is open read-only, so nothing
    can be stamped onto it. The order is **mask -> scale -> fill**, matching how
    pyramids unpacks CF-packed netCDF data:

    1. the stored counts are read with the declared no-data masked
       (`unpack=False, masked=True`), so a sentinel is never scaled into a
       plausible-looking physical value;
    2. a configured `data_type` casts those counts;
    3. each band's `scale` / `offset` is applied to the valid pixels only,
       promoting the result to `float32`;
    4. the mask is filled — with `NaN` when rescaling (physical no-data has no
       meaningful stored sentinel), else with the band's resolved no-data value.

    The result is built by :meth:`Dataset.from_array`, so it declares **identity
    packing** (`scale == 1`, `offset == 0`): a later `read_array(unpack=True)`
    cannot apply the STAC scale a second time.

    Args:
        dataset: The opened raster (read-only is fine; it is never mutated).
        overrides: What to apply, from :func:`resolve_overrides`.
        path: Optional destination for the rebuilt raster; `None` keeps it in
            memory.

    Returns:
        A new :class:`~pyramids.dataset.Dataset` when something was applied, or
        `dataset` itself when nothing was — so a caller can test identity
        (`result is dataset`) to find out whether it still holds the lazy
        handle.

    Examples:
        - A packed asset's counts become physical values, no-data preserved:
            ```python
            >>> import numpy as np
            >>> from pyramids.base.georeference import GeoReference
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.stac._config import AssetOverrides, materialise
            >>> packed = Dataset.from_array(
            ...     np.array([[0, 100], [200, 300]], dtype="int16"),
            ...     no_data_value=0,
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0),
            ... )
            >>> physical = materialise(packed, AssetOverrides(scales=(0.01,)))
            >>> physical.read_array().tolist()
            [[nan, 1.0], [2.0, 3.0]]
            >>> physical.scale, physical.offset
            ([1.0], [0])

            ```
        - Nothing configured means nothing is rebuilt:
            ```python
            >>> materialise(packed, AssetOverrides()) is packed
            True

            ```
    """
    nodata = _nodata_targets(dataset, overrides)
    units = _unit_targets(dataset, overrides)
    native_dtype = list(dataset.dtype)[0] if dataset.band_count else None
    data_type = overrides.data_type if overrides.data_type != native_dtype else None
    changes_nodata = nodata != list(dataset.no_data_value)
    result = dataset
    if overrides.rescales or data_type is not None or changes_nodata or units:
        stored = _band_first(dataset)
        if data_type is not None:
            stored = stored.astype(data_type)
        planes: list[np.ndarray] = []
        declared: list[Any] = []
        for index in range(stored.shape[0]):
            plane = stored[index]
            fill: Any = nodata[index] if index < len(nodata) else None
            if overrides.rescales:
                scale, offset = overrides.packing(index)
                plane = np.ma.asanyarray(apply_unpack(plane, scale, offset)).astype(
                    "float32"
                )
                fill = np.nan
            declared.append(fill)
            planes.append(np.ma.filled(plane, 0 if fill is None else fill))
        result = Dataset.from_array(
            np.stack(planes),
            no_data_value=_collapse(declared),
            # Reconstructed from the geotransform rather than a corner + cell
            # size, so a rotated or anisotropic grid survives the rebuild.
            geo_ref=GeoReference(geo=tuple(dataset.geotransform), epsg=dataset.epsg),
            path=path,
        )
        if units:
            result.band_units = units
    return result


__all__ = [
    "OVERRIDE_KEYS",
    "AssetMetadataWarning",
    "AssetOverrides",
    "band_packing",
    "item_collection_id",
    "materialise",
    "resolve_alias",
    "resolve_asset_metadata",
    "resolve_overrides",
    "warnings_ignored",
]
