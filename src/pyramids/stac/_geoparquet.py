"""Serialize STAC Items to/from GeoParquet (PD-3).

stac-geoparquet stores a STAC ItemCollection as one columnar GeoParquet file
(geometry as WKB, WGS84) for bulk transfer + fast spatial filtering, avoiding
thousands of per-item JSON requests. pyramids already has geopandas (core) and a
`FeatureCollection` (a `GeoDataFrame` subclass) with GeoParquet I/O, plus the
`[parquet]` extra (pyarrow) — so the round-trip needs **no new dependency**.

Two on-disk layouts live here, sharing one pair of entry points:

- The **JSON-blob** layout (the default, `spec=False`): each row carries the item
  geometry (so the file is a valid, spatially-filterable GeoParquet) plus the
  full STAC Item as a `stac_item` JSON column, so :func:`from_geoparquet`
  reconstructs the exact item dicts — ready to feed
  :meth:`pyramids.dataset.DatasetCollection.from_stac`. Lossless, but only
  pyramids knows how to read the blob back.
- The **spec** layout (`spec=True`, or :func:`to_geoparquet_spec` /
  :func:`from_geoparquet_spec`): the STAC-GeoParquet 1.1 shape — `properties`
  flattened into top-level typed columns, WKB geometry in OGC:CRS84, a `bbox`
  struct, and `assets` / `links` as Arrow structs. That is what lets DuckDB,
  pyarrow and the rest of the STAC tooling push spatial *and* attribute
  predicates down into the file. Written with **pyarrow only** — the
  `stac-geoparquet` package is deliberately not a dependency. It preserves
  every value, but the round trip is not an identity on the item dict: see
  the normalisations listed on :func:`from_geoparquet_spec`.

Requires the `[parquet]` extra (pyarrow) for the Parquet read/write itself.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyproj
import shapely
import shapely.geometry

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import extra_hint, require_optional

__all__ = [
    "from_geoparquet",
    "from_geoparquet_spec",
    "to_geoparquet",
    "to_geoparquet_spec",
]

_ITEM_COLUMN = "stac_item"

_GEOPARQUET_VERSION = "1.1.0"
_SPEC_VERSION = "1.1.0"
_GEOMETRY_COLUMN = "geometry"
_BBOX_COLUMN = "bbox"

_GEO_METADATA_KEY = b"geo"
_STAC_METADATA_KEY = b"stac-geoparquet"
_PYRAMIDS_METADATA_KEY = b"pyramids:stac-geoparquet"

_LEADING_COLUMNS = (
    "type",
    "stac_version",
    "stac_extensions",
    "id",
    _GEOMETRY_COLUMN,
    _BBOX_COLUMN,
    "collection",
)
_TRAILING_COLUMNS = ("links", "assets")
_ROOT_COLUMNS = _LEADING_COLUMNS + _TRAILING_COLUMNS
_STRING_COLUMNS = ("type", "stac_version", "id", "collection")

_DATETIME_PROPERTIES = (
    "datetime",
    "start_datetime",
    "end_datetime",
    "created",
    "updated",
    "published",
    "expires",
    "unpublished",
)

_PARQUET_HINT = extra_hint(
    "Spec-compliant STAC-GeoParquet I/O requires the pyarrow dependency.",
    "parquet",
)

# The CRS GeoParquet assumes when a geometry column omits `crs`. It is written
# out explicitly (as PROJJSON) because an explicit `"crs": null` means the
# opposite — "no CRS assigned to this column" — and readers honour that.
_CRS84 = "OGC:CRS84"


def _no_items_error(caller: str) -> ValueError:
    """Build the empty-input error, naming the function the caller invoked.

    Args:
        caller: The public function name to quote in the message.

    Returns:
        ValueError: The error to raise (not raised here, so the traceback
            starts at the caller).
    """
    return ValueError(f"{caller} received no items.")


def _item_to_dict(item: Any) -> dict[str, Any]:
    """Return a STAC Item as a plain dict (calls `.to_dict()` when available)."""
    if hasattr(item, "to_dict"):
        return cast(dict[str, Any], item.to_dict())
    if isinstance(item, dict):
        return item
    raise TypeError(
        f"STAC item must be a dict or expose to_dict(), got {type(item).__name__}."
    )


def _checked_bbox(item: dict[str, Any]) -> list[Any] | None:
    """Return an item's `bbox` as a list, refusing a malformed one by name.

    Args:
        item: The normalised item dict.

    Returns:
        list | None: The bbox values, or `None` when the item carries no bbox
            (or an empty one).

    Raises:
        ValueError: When the bbox holds a number of values STAC does not
            define — anything other than 4 (2D) or 6 (3D).
    """
    raw = item.get(_BBOX_COLUMN)
    checked: list[Any] | None = None
    if raw:
        values = list(raw)
        if len(values) not in (4, 6):
            raise ValueError(
                f"STAC item {item.get('id')!r} carries a {len(values)}-element "
                f"bbox {values}; a STAC bbox holds 4 values (2D) or 6 (3D)."
            )
        checked = values
    return checked


def _item_geometry(item: dict[str, Any]) -> Any:
    """Build a shapely geometry from an item's `geometry` or `bbox`.

    Args:
        item: The normalised item dict.

    Returns:
        A shapely geometry, or `None` when the item carries neither a
        `geometry` nor a `bbox`.

    Raises:
        ValueError: When the item carries no `geometry` and its `bbox` — the
            fallback this then has to read — is neither 4- nor 6-valued (raised
            by :func:`_checked_bbox`). An item that carries a geometry is not
            held to that, so the JSON-blob layout stays as tolerant of a
            malformed bbox as it was.
    """
    geometry = None
    geom = item.get("geometry")
    if geom:
        geometry = shapely.geometry.shape(geom)
    else:
        bbox = _checked_bbox(item)
        if bbox:
            half = len(bbox) // 2
            geometry = shapely.geometry.box(
                bbox[0], bbox[1], bbox[half], bbox[half + 1]
            )
    return geometry


def _require_parquet() -> tuple[Any, Any]:
    """Import `pyarrow` and `pyarrow.parquet`, or raise the `[parquet]` hint.

    Returns:
        tuple: `(pyarrow, pyarrow.parquet)` module objects.

    Raises:
        OptionalPackageDoesNotExist: When pyarrow is not installed.
    """
    arrow = require_optional("pyarrow", _PARQUET_HINT, return_module=True)
    parquet = require_optional("pyarrow.parquet", _PARQUET_HINT, return_module=True)
    return arrow, parquet


def _parse_rfc3339(value: Any) -> datetime | None:
    """Coerce an RFC 3339 string (or datetime) to an aware UTC `datetime`.

    Args:
        value: An RFC 3339 timestamp string, a `datetime`, or `None`.

    Returns:
        datetime | None: The timezone-aware datetime, or `None` when `value` is
            `None` / an empty string.

    Raises:
        ValueError: When `value` is a string that is not RFC 3339.
        TypeError: When `value` is neither a string, a `datetime`, nor `None`.
    """
    moment: datetime | None = None
    if isinstance(value, datetime):
        moment = value
    elif value is None:
        moment = None
    elif isinstance(value, str):
        text = value.strip()
        if text:
            if text.endswith(("Z", "z")):
                text = f"{text[:-1]}+00:00"
            moment = datetime.fromisoformat(text)
    else:
        raise TypeError(f"not a timestamp: {type(value).__name__}")
    if moment is not None and moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment


def _format_rfc3339(value: datetime) -> str:
    """Render an aware `datetime` as the RFC 3339 spelling STAC uses.

    Args:
        value: The datetime to render (naive values are read as UTC).

    Returns:
        str: `YYYY-MM-DDTHH:MM:SSZ`, with fractional seconds only when the
            value carries microseconds.
    """
    moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    moment = moment.astimezone(UTC)
    pattern = "%Y-%m-%dT%H:%M:%S.%f" if moment.microsecond else "%Y-%m-%dT%H:%M:%S"
    return f"{moment.strftime(pattern)}Z"


def _from_arrow(value: Any) -> Any:
    """Turn a value read back out of Arrow into plain STAC JSON.

    Arrow structs are *rectangular*: unifying `{"a": 1}` and `{"b": 2}` yields
    one struct type with both fields, so every row reports the field it never
    had as `None`. Those nulls are dropped here, which restores the original
    sparse dicts; timestamps are rendered back to their RFC 3339 strings.

    The cost is that a member whose value was *written* as `None` is dropped
    too — the column cannot tell "null" from "not set" — so an explicitly null
    member does not survive the round trip.

    Args:
        value: A value from `pyarrow.Table.to_pylist()` — a scalar, a list, a
            dict, or a `datetime`.

    Returns:
        The same value with `None` members removed from every nested mapping
        and every `datetime` replaced by its RFC 3339 string.
    """
    if isinstance(value, Mapping):
        plain: Any = {
            key: _from_arrow(item) for key, item in value.items() if item is not None
        }
    elif isinstance(value, list):
        plain = [_from_arrow(item) for item in value]
    elif isinstance(value, datetime):
        plain = _format_rfc3339(value)
    else:
        plain = value
    return plain


def _numeric_kind(value: Any) -> str | None:
    """Name the numeric kind of a value, or `None` when it is not a number.

    Args:
        value: Any STAC property value.

    Returns:
        str | None: `"bool"`, `"int"`, `"float"`, or `None`. `bool` is reported
            before `int` because every `bool` is an `int` in Python.
    """
    kind = None
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    elif isinstance(value, float):
        kind = "float"
    return kind


def _mixes_numeric_types(values: list[Any]) -> bool:
    """Report whether a column mixes `bool` / `int` / `float` at any depth.

    Arrow *unifies* those rather than refusing them — `[1, 2.5]` infers
    `double`, so the int comes back as `1.0` — which is the one widening the
    JSON fallback cannot detect by catching an exception. Struct fields and
    list items are checked too, so the same widening one level inside `assets`
    or `links` is caught.

    Args:
        values: One value per item, `None` where the item lacked the key.

    Returns:
        bool: `True` when two different numeric kinds share a position.
    """
    kinds = {kind for kind in (_numeric_kind(value) for value in values) if kind}
    mixed = len(kinds) > 1
    if not mixed:
        fields: dict[Any, list[Any]] = {}
        items: list[Any] = []
        for value in values:
            if isinstance(value, Mapping):
                for key, member in value.items():
                    fields.setdefault(key, []).append(member)
            elif isinstance(value, list):
                items.extend(value)
        groups = list(fields.values())
        if items:
            groups.append(items)
        mixed = any(_mixes_numeric_types(group) for group in groups)
    return mixed


def _has_empty_struct(arrow: Any, type_: Any) -> bool:
    """Report whether an Arrow type holds a zero-field struct anywhere.

    `arrow.array([{}])` infers `struct<>`, `arrow.array([{"red": {}}])` infers
    `struct<red: struct<>>` and `arrow.array([[{}]])` infers
    `list<item: struct<>>`. All three are valid Arrow and none can be written
    to Parquet ("Cannot write struct type … with no child field"), and the
    failure surfaces at write time rather than at array construction — so it
    has to be detected from the type instead of caught.

    Args:
        arrow: The `pyarrow` module.
        type_: A `pyarrow.DataType`.

    Returns:
        bool: `True` when the type is, contains, or lists a struct with no
            fields.
    """
    found = False
    if arrow.types.is_struct(type_):
        found = type_.num_fields == 0 or any(
            _has_empty_struct(arrow, type_.field(index).type)
            for index in range(type_.num_fields)
        )
    elif hasattr(type_, "value_type"):
        # Every list-like and dictionary type reaches its member type this way.
        found = _has_empty_struct(arrow, type_.value_type)
    return found


def _spec_column(arrow: Any, key: str, values: list[Any]) -> tuple[Any, bool]:
    """Build one Arrow array from a column of Python values.

    Arrow infers a type from the values, which gives genuinely typed, queryable
    columns for the common case. Three kinds of column cannot be stored that
    way and fall back to a JSON-encoded string column instead, which the reader
    decodes again from the name recorded by the caller:

    - Values Arrow refuses to unify into one type (mixed scalar types, structs
      whose shared field has two different types, an integer too large for any
      Arrow integer).
    - Values Arrow unifies *too eagerly*: a column mixing `bool` / `int` /
      `float` would silently widen (`1` → `1.0`), so it is refused up front by
      :func:`_mixes_numeric_types`.
    - Values whose inferred type contains a zero-field struct (an empty dict,
      e.g. the `"assets": {}` every asset-less `pystac.Item` emits), which
      Parquet cannot encode — see :func:`_has_empty_struct`.

    No data is lost on that path: the values are JSON-encoded as they are, and
    a value JSON cannot represent raises rather than being stringified.

    Args:
        arrow: The `pyarrow` module.
        key: The column name, quoted in the error when the values can be
            stored neither as Arrow nor as JSON.
        values: One value per item, `None` where the item lacked the key.

    Returns:
        tuple: `(array, is_json)` — the Arrow array and whether its strings are
            JSON-encoded payloads rather than real values.

    Raises:
        TypeError: When the values share no Arrow type and are not
            JSON-serialisable either.
    """
    is_json = False
    array = None
    if all(value is None for value in values):
        array = arrow.array(values, type=arrow.string())
    elif not _mixes_numeric_types(values):
        try:
            inferred = arrow.array(values)
        except (
            arrow.ArrowInvalid,
            arrow.ArrowTypeError,
            arrow.ArrowNotImplementedError,
            OverflowError,
        ):
            inferred = None
        if inferred is not None and not _has_empty_struct(arrow, inferred.type):
            array = inferred
    if array is None:
        array = arrow.array(_json_encoded(key, values), type=arrow.string())
        is_json = True
    return array, is_json


def _json_encoded(key: str, values: list[Any]) -> list[str | None]:
    """JSON-encode a column's values verbatim, or say why it cannot be done.

    Args:
        key: The column name, quoted in the error message.
        values: One value per item, `None` where the item lacked the key.

    Returns:
        list: One JSON document per value, `None` passed through.

    Raises:
        TypeError: When a value is not JSON-serialisable. No `default=` hook is
            used on purpose: coercing such a value to its string form would
            lose exactly the data the fallback exists to preserve.
    """
    encoded: list[str | None] = []
    try:
        encoded = [None if value is None else json.dumps(value) for value in values]
    except TypeError as error:
        raise TypeError(
            f"STAC-GeoParquet cannot store the {key!r} column: its values share "
            f"no Arrow type, and JSON cannot encode them either ({error})."
        ) from error
    return encoded


def _spec_typed_column(arrow: Any, key: str, values: list[Any]) -> tuple[Any, bool]:
    """Build one column, pinning the type for the keys the spec types for us.

    `id`, `type`, `stac_version` and `collection` are strings and
    `stac_extensions` a list of strings in every STAC Item, so those columns are
    built with that type rather than inferred — otherwise a file whose items all
    carry `"stac_extensions": []` would land a `list<null>` column, which is
    valid Arrow but useless to a reader expecting the spec schema. Anything the
    declared type rejects, and every other key, goes through
    :func:`_spec_column`.

    Only Arrow's own conversion errors are caught when the declared type is
    tried; a plain `TypeError` is a bug in this module (a wrong-arity call, a
    `None` where the module is expected) and propagates rather than being
    turned into a silent JSON column.

    Args:
        arrow: The `pyarrow` module.
        key: The column name.
        values: One value per item, `None` where the item lacked the key.

    Returns:
        tuple: `(array, is_json)` — the Arrow array and whether its strings are
            JSON-encoded payloads rather than real values.

    Raises:
        TypeError: When the values can be stored neither as Arrow nor as JSON
            (raised by :func:`_spec_column`).
    """
    declared = None
    if key in _STRING_COLUMNS:
        declared = arrow.string()
    elif key == "stac_extensions":
        declared = arrow.list_(arrow.string())
    array = None
    is_json = False
    if declared is not None:
        try:
            array = arrow.array(values, type=declared)
        except (
            arrow.ArrowInvalid,
            arrow.ArrowTypeError,
            arrow.ArrowNotImplementedError,
            OverflowError,
        ):
            array = None
    if array is None:
        array, is_json = _spec_column(arrow, key, values)
    return array, is_json


def _timestamp_column(arrow: Any, values: list[Any]) -> Any:
    """Build a `timestamp[us, UTC]` array, or `None` when the values are not times.

    An all-`None` column still becomes a timestamp array: the schema of a
    time-named spec column must not depend on whether this particular file
    happens to carry a value for it.

    Sub-microsecond precision is lost — the column stores microseconds, so
    `...05.123456789Z` is stored (and read back) as `...05.123456Z`.

    Args:
        arrow: The `pyarrow` module.
        values: One value per item, `None` where the item lacked the key.

    Returns:
        A `pyarrow.Array` of UTC timestamps, or `None` when at least one value
        is not an RFC 3339 timestamp (the caller then falls back to
        :func:`_spec_column`).
    """
    column = None
    try:
        parsed = [_parse_rfc3339(value) for value in values]
    except (ValueError, TypeError):
        parsed = None
    if parsed is not None:
        column = arrow.array(parsed, type=arrow.timestamp("us", tz="UTC"))
    return column


def _bbox_column(arrow: Any, item_dicts: list[dict[str, Any]]) -> Any:
    """Build the STAC-GeoParquet `bbox` struct column.

    Args:
        arrow: The `pyarrow` module.
        item_dicts: The normalised item dicts.

    Returns:
        A `pyarrow.Array` of `struct<xmin, ymin, xmax, ymax>` — extended with
        `zmin` / `zmax` as soon as *any* item carries a 3D bbox, so one 2D bbox
        in the collection does not demote the others (the 2D rows then hold a
        null `zmin` / `zmax`). Items without a `bbox` get a null struct.

    Raises:
        ValueError: When an item's bbox is neither 4- nor 6-valued (raised by
            :func:`_checked_bbox`).
    """
    boxes = [_checked_bbox(item) for item in item_dicts]
    three_d = any(len(box) == 6 for box in boxes if box)
    names = (
        ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")
        if three_d
        else ("xmin", "ymin", "xmax", "ymax")
    )
    rows: list[dict[str, float | None] | None] = []
    for box in boxes:
        if box is None:
            rows.append(None)
            continue
        half = len(box) // 2
        flat: list[Any] = [box[0], box[1], box[half], box[half + 1]]
        if three_d:
            zmin = box[2] if len(box) == 6 else None
            zmax = box[5] if len(box) == 6 else None
            flat = [box[0], box[1], zmin, box[half], box[half + 1], zmax]
        rows.append(
            dict(
                zip(names, (None if value is None else float(value) for value in flat))
            )
        )
    struct = arrow.struct([arrow.field(name, arrow.float64()) for name in names])
    return arrow.array(rows, type=struct)


def _geo_metadata(geometries: list[Any], with_covering: bool) -> dict[str, Any]:
    """Compose the GeoParquet `geo` file-metadata block for the geometry column.

    The `crs` member carries the OGC:CRS84 PROJJSON document. GeoParquet 1.1
    reads an *omitted* `crs` as CRS84 but an explicit `null` as "no CRS
    assigned to this column", so writing the document is what makes a reader
    (geopandas, DuckDB) hand back a georeferenced frame.

    Args:
        geometries: The per-item shapely geometries (`None` entries allowed).
        with_covering: Whether to advertise the `bbox` struct column as the
            geometry column's bounding-box covering. Only true when every row
            has a bbox — a covering that skips a row would silently drop it
            from a spatial-predicate pushdown.

    Returns:
        dict: The GeoParquet 1.1 metadata document, ready to be JSON-encoded.
    """
    present = [geom for geom in geometries if geom is not None]
    column: dict[str, Any] = {
        "encoding": "WKB",
        "geometry_types": sorted({geom.geom_type for geom in present}),
        "crs": pyproj.CRS.from_user_input(_CRS84).to_json_dict(),
    }
    if present:
        bounds = shapely.total_bounds(present)
        column["bbox"] = [float(value) for value in bounds]
    if with_covering:
        column["covering"] = {
            "bbox": {
                "xmin": [_BBOX_COLUMN, "xmin"],
                "ymin": [_BBOX_COLUMN, "ymin"],
                "xmax": [_BBOX_COLUMN, "xmax"],
                "ymax": [_BBOX_COLUMN, "ymax"],
            }
        }
    return {
        "version": _GEOPARQUET_VERSION,
        "primary_column": _GEOMETRY_COLUMN,
        "columns": {_GEOMETRY_COLUMN: column},
    }


def _property_keys(item_dicts: list[dict[str, Any]]) -> list[str]:
    """List every `properties` key across the items, in a stable column order.

    `datetime` (and its sibling time fields) lead, so the columns most queries
    filter on sit at a fixed position; the rest follow alphabetically.

    Args:
        item_dicts: The normalised item dicts.

    Returns:
        list: The ordered property keys. `datetime` is always present, even
            when no item carries it, so the schema is stable.
    """
    seen: list[str] = []
    for item in item_dicts:
        for key in item.get("properties") or {}:
            if key not in seen:
                seen.append(key)
    ordered = [key for key in _DATETIME_PROPERTIES if key in seen]
    ordered += sorted(key for key in seen if key not in _DATETIME_PROPERTIES)
    if "datetime" not in ordered:
        ordered.insert(0, "datetime")
    return ordered


def _root_extra_keys(item_dicts: list[dict[str, Any]]) -> list[str]:
    """List item-root keys that are neither spec columns nor `properties`.

    Args:
        item_dicts: The normalised item dicts.

    Returns:
        list: The sorted extra root keys (each becomes its own column).
    """
    seen: set[str] = set()
    for item in item_dicts:
        seen.update(
            key for key in item if key not in _ROOT_COLUMNS and key != "properties"
        )
    return sorted(seen)


def to_geoparquet(items: Any, path: str | Path, *, spec: bool = False) -> None:
    """Write a sequence of STAC Items to a GeoParquet file.

    With `spec=False` (the default) each item becomes a row carrying its
    geometry (a valid, spatially-filterable GeoParquet geometry in EPSG:4326)
    and the full item as a JSON column. With `spec=True` the whole write is
    delegated to :func:`to_geoparquet_spec`, which produces the columnar
    STAC-GeoParquet 1.1 layout instead — different columns, different
    metadata, and the extra guards listed under `Raises`.

    Args:
        items: Iterable of STAC Items (`pystac.Item` objects with `to_dict()`,
            or raw STAC-JSON dicts — e.g. from
            :meth:`pyramids.dataset.Dataset.to_stac_item`).
        path: Destination `.parquet` path.
        spec: Write the interoperable STAC-GeoParquet 1.1 layout (flattened
            property columns, WKB geometry in OGC:CRS84, `bbox` struct,
            `assets` / `links` structs) by delegating to
            :func:`to_geoparquet_spec`, instead of the pyramids JSON-blob
            layout. Defaults to `False`, which keeps the historical behaviour
            byte for byte.

    Raises:
        ValueError: When `items` is empty. With `spec=True`, also when a
            `properties` key collides with a reserved spec column name, or
            when an item's `bbox` is neither 4- nor 6-valued.
        TypeError: When an item is neither a dict nor exposes `to_dict()`;
            with `spec=True`, also when a column's values can be stored
            neither as Arrow nor as JSON.
        OptionalPackageDoesNotExist: When pyarrow (the `[parquet]` extra) is not
            installed (raised by `FeatureCollection.to_parquet`, or directly
            when `spec=True`).

    Examples:
        - Round-trip a couple of item dicts through GeoParquet:
            ```python
            >>> import tempfile, os
            >>> from pyramids.stac import to_geoparquet, from_geoparquet  # doctest: +SKIP
            >>> items = [{"id": "a", "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
            ...           "properties": {"datetime": "2023-01-01T00:00:00Z"}, "assets": {}}]
            >>> path = os.path.join(tempfile.mkdtemp(), "items.parquet")  # doctest: +SKIP
            >>> to_geoparquet(items, path)  # doctest: +SKIP
            >>> from_geoparquet(path)[0]["id"]  # doctest: +SKIP
            'a'

            ```

    See Also:
        to_geoparquet_spec: The spec layout written directly, without the flag.
    """
    if spec:
        to_geoparquet_spec(items, path)
        return

    from pyramids.feature import FeatureCollection

    rows = []
    geometries = []
    for item in items:
        as_dict = _item_to_dict(item)
        geometries.append(_item_geometry(as_dict))
        rows.append({"id": as_dict.get("id"), _ITEM_COLUMN: json.dumps(as_dict)})

    if not rows:
        raise _no_items_error("to_geoparquet")

    fc = FeatureCollection(rows, geometry=geometries, crs="EPSG:4326")
    fc.to_parquet(str(path))


def from_geoparquet(path: str | Path, *, spec: bool = False) -> list[dict[str, Any]]:
    """Read STAC Items back from a GeoParquet written by :func:`to_geoparquet`.

    Both layouts are readable through this one entry point. With `spec=False`
    (the default) the file's metadata is probed first: a file carrying the
    `stac-geoparquet` key goes to the spec reader, so a spec file reads
    correctly even when the flag was not passed, and everything else takes the
    unchanged JSON-blob path. Passing `spec=True` selects the spec reader
    outright.

    Args:
        path: Path to a `.parquet` file produced by :func:`to_geoparquet` (either
            layout).
        spec: Read the file as the STAC-GeoParquet 1.1 layout by delegating to
            :func:`from_geoparquet_spec`. Defaults to `False` (JSON-blob, with
            the auto-detect fallback described above).

    Returns:
        The list of STAC Item dicts (ready for
        :meth:`pyramids.dataset.DatasetCollection.from_stac`).

    Raises:
        OptionalPackageDoesNotExist: When pyarrow (the `[parquet]` extra) is not
            installed (raised by `FeatureCollection.read_parquet`).
        KeyError: When the file lacks the `stac_item` JSON column and carries no
            STAC-GeoParquet file metadata (so it was written by neither layout).
        ValueError: When `spec=True` is passed for a file that is not in the
            spec layout (raised by :func:`from_geoparquet_spec`).

    See Also:
        from_geoparquet_spec: The spec layout read directly, without the flag.
    """
    if spec or _is_spec_geoparquet(path):
        return from_geoparquet_spec(path)

    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_parquet(str(path))
    return [json.loads(blob) for blob in fc[_ITEM_COLUMN]]


def _is_spec_geoparquet(path: str | Path) -> bool:
    """Report whether a Parquet file carries the STAC-GeoParquet file metadata.

    Probing is best-effort: when pyarrow is missing, or the path is not a
    readable Parquet file, this answers `False` and the JSON-blob reader runs
    and reports the real problem itself. That keeps the historical `spec=False`
    error messages exactly as they were, which is why the handler covers
    `pyarrow.ArrowException` as a whole rather than a list of subclasses —
    `ArrowNotImplementedError` (via `NotImplementedError`),
    `ArrowCapacityError`, `ArrowMemoryError`, `ArrowKeyError`,
    `ArrowIndexError`, `ArrowSerializationError`, `ArrowCancelled` and
    `ArrowTypeError` derive from neither `OSError` nor `ValueError` and would
    otherwise escape a default `from_geoparquet(path)` call. `OSError` is kept
    for the plain cases pyarrow does not raise itself (a missing file) and
    `ValueError` for `ArrowInvalid`, which pre-dates `ArrowException` in the
    MRO.

    A failure that is *not* one of those — a bug in this function — still
    propagates.

    Args:
        path: Path to a `.parquet` file.

    Returns:
        bool: `True` when the file's key-value metadata holds the
            `stac-geoparquet` key written by :func:`to_geoparquet_spec` (and by
            the `stac-geoparquet` package).
    """
    detected = False
    modules: tuple[Any, Any] | None = None
    try:
        modules = _require_parquet()
    except OptionalPackageDoesNotExist:
        modules = None
    if modules is not None:
        arrow, parquet = modules
        try:
            metadata = parquet.read_metadata(str(path)).metadata or {}
            detected = _STAC_METADATA_KEY in metadata
        except (OSError, ValueError, arrow.ArrowException):
            detected = False
    return detected


def to_geoparquet_spec(items: Any, path: str | Path) -> None:
    """Write STAC Items as a spec-compliant STAC-GeoParquet 1.1 file.

    This is the interoperable, columnar layout — the one DuckDB, pyarrow and the
    wider STAC tooling can filter on without decoding a JSON blob per row. It is
    produced with **pyarrow only**; the `stac-geoparquet` package is not a
    dependency of pyramids.

    On-disk columns, in order:

    - `type`, `stac_version`, `stac_extensions`, `id` — the item envelope.
    - `geometry` — WKB, OGC:CRS84 (written as PROJJSON in the GeoParquet `geo`
      file metadata, not left to the reader's default).
    - `bbox` — `struct<xmin, ymin, xmax, ymax>`, widened to
      `struct<xmin, ymin, zmin, xmax, ymax, zmax>` as soon as any item carries
      a 3D bbox. It is registered as the geometry column's `covering` only when
      *every* row has a bbox: a covering that does not span a row would make a
      spatial-predicate pushdown skip it.
    - `collection` — always emitted, null for items without one.
    - `datetime` — `timestamp[us, UTC]`, always emitted (an all-null column
      included), unless a value is not an RFC 3339 timestamp at all.
    - One column per remaining `properties` key, alphabetically, Arrow-typed.
    - One column per extra item-root key, alphabetically.
    - `links`, `assets` — Arrow structs (`assets` keyed by asset name) when at
      least one item carries them; a null string column when no item does,
      since a struct has no fields to infer then.

    A column Arrow cannot type — or would type only by widening an `int` to a
    `float`, or by inferring a zero-field struct from an empty dict such as the
    `"assets": {}` of an asset-less item — is JSON-encoded instead, which keeps
    the values intact at the cost of that column's queryability.

    Three file-metadata keys are written: `geo` (GeoParquet 1.1),
    `stac-geoparquet` (the spec version) and `pyramids:stac-geoparquet` (which
    columns are JSON-encoded and which belong at the item root — read back by
    :func:`from_geoparquet_spec`, and ignorable by every other reader).

    The file is written through a sibling temporary file and moved into place,
    so a failed write leaves neither a truncated new file nor a damaged
    previous one.

    Args:
        items: Iterable of STAC Items (`pystac.Item` objects with `to_dict()`,
            or raw STAC-JSON dicts — e.g. from
            :meth:`pyramids.dataset.Dataset.to_stac_item`).
        path: Destination `.parquet` path.

    Raises:
        ValueError: When `items` is empty, when a `properties` key collides
            with one of the reserved spec column names (`id`, `geometry`,
            `bbox`, `assets`, …) — the item could not be reconstructed — or
            when an item's `bbox` holds neither 4 nor 6 values.
        TypeError: When an item is neither a dict nor exposes `to_dict()`, or
            when a column's values can be stored neither as Arrow nor as JSON.
        OptionalPackageDoesNotExist: When pyarrow (the `[parquet]` extra) is not
            installed.

    Examples:
        - Write the columnar layout and read it with plain pyarrow:
            ```python
            >>> import os, tempfile  # doctest: +SKIP
            >>> import pyarrow.parquet as pq  # doctest: +SKIP
            >>> from pyramids.stac import to_geoparquet_spec  # doctest: +SKIP
            >>> items = [{"id": "a", "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
            ...           "bbox": [1.0, 2.0, 1.0, 2.0], "assets": {},
            ...           "properties": {"datetime": "2023-01-01T00:00:00Z"}}]
            >>> path = os.path.join(tempfile.mkdtemp(), "spec.parquet")  # doctest: +SKIP
            >>> to_geoparquet_spec(items, path)  # doctest: +SKIP
            >>> "datetime" in pq.read_table(path).column_names  # doctest: +SKIP
            True

            ```

    See Also:
        to_geoparquet: The pyramids JSON-blob layout (lossless, pyramids-only).
        from_geoparquet_spec: The matching reader.
    """
    arrow, parquet = _require_parquet()

    item_dicts = [_item_to_dict(item) for item in items]
    if not item_dicts:
        raise _no_items_error("to_geoparquet_spec")

    property_keys = _property_keys(item_dicts)
    root_extras = _root_extra_keys(item_dicts)
    clashes = sorted(set(property_keys) & set(_ROOT_COLUMNS))
    clashes += sorted(set(root_extras) & set(property_keys))
    if clashes:
        raise ValueError(
            "STAC-GeoParquet reserves the column names "
            f"{', '.join(sorted(set(clashes)))} — an item flattens a property "
            "onto one of them, which cannot be read back. Rename the property "
            "or write the JSON-blob layout with spec=False."
        )

    geometries = [_item_geometry(item) for item in item_dicts]
    names: list[str] = []
    arrays: list[Any] = []
    json_columns: list[str] = []

    for key in ("type", "stac_version", "stac_extensions", "id"):
        values = [item.get(key) for item in item_dicts]
        array, is_json = _spec_typed_column(arrow, key, values)
        names.append(key)
        arrays.append(array)
        if is_json:
            json_columns.append(key)

    names.append(_GEOMETRY_COLUMN)
    arrays.append(
        arrow.array(
            [None if geom is None else shapely.to_wkb(geom) for geom in geometries],
            type=arrow.binary(),
        )
    )

    bbox_array = _bbox_column(arrow, item_dicts)
    names.append(_BBOX_COLUMN)
    arrays.append(bbox_array)

    for key in ("collection", *property_keys, *root_extras, *_TRAILING_COLUMNS):
        if key in property_keys:
            values = [(item.get("properties") or {}).get(key) for item in item_dicts]
        else:
            values = [item.get(key) for item in item_dicts]
        array = None
        if key in _DATETIME_PROPERTIES:
            array = _timestamp_column(arrow, values)
        if array is None:
            array, is_json = _spec_typed_column(arrow, key, values)
            if is_json:
                json_columns.append(key)
        names.append(key)
        arrays.append(array)

    table = arrow.Table.from_arrays(arrays, names=names)
    metadata = {
        _GEO_METADATA_KEY: json.dumps(
            _geo_metadata(geometries, bbox_array.null_count == 0)
        ).encode(),
        _STAC_METADATA_KEY: json.dumps({"version": _SPEC_VERSION}).encode(),
        _PYRAMIDS_METADATA_KEY: json.dumps(
            {
                "version": 1,
                "json_columns": json_columns,
                "root_columns": root_extras,
            }
        ).encode(),
    }
    _write_table_atomically(parquet, table.replace_schema_metadata(metadata), path)


def _write_table_atomically(parquet: Any, table: Any, path: str | Path) -> None:
    """Write an Arrow table to `path` so a failure never leaves a broken file.

    The Parquet writer creates its target before it can know whether the whole
    table encodes, so a failure half-way through used to leave a 0-byte file —
    which then defeated :func:`_is_spec_geoparquet` and made the next read
    report "Parquet file size is 0 bytes" from the JSON-blob reader. Writing a
    sibling temporary file and moving it into place keeps the target either
    absent or complete, and leaves a previous version intact on failure.

    Args:
        parquet: The `pyarrow.parquet` module.
        table: The table to write.
        path: The destination path.
    """
    target = Path(path)
    handle, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(handle)
    moved = False
    try:
        parquet.write_table(table, temporary)
        os.replace(temporary, target)
        moved = True
    finally:
        if not moved:
            Path(temporary).unlink(missing_ok=True)


def from_geoparquet_spec(path: str | Path) -> list[dict[str, Any]]:
    """Read STAC Items back from a spec-compliant STAC-GeoParquet file.

    The inverse of :func:`to_geoparquet_spec`, and tolerant of files written by
    other STAC-GeoParquet writers: the reserved spec columns are recognised by
    name, every other column is read back as a `properties` entry, timestamp
    columns are rendered to their RFC 3339 strings and the WKB geometry is
    decoded to GeoJSON. Null cells become absent keys, which is what restores
    the original sparse items out of Arrow's rectangular structs.

    Four normalisations are inherent to the columnar layout, and are the same
    ones the `stac-geoparquet` package applies — so the round trip preserves
    every *value*, but is **not** an identity on the item dict:

    - Timestamps come back in the `Z` spelling (`...T00:00:00Z`), whatever
      offset spelling they were written with, because the column stores a real
      instant rather than the original text.
    - Timestamps are stored with microsecond resolution, so a nanosecond
      timestamp loses its last three digits (`...05.123456789Z` →
      `...05.123456Z`).
    - A key whose value was explicitly `null` comes back absent, since the
      column cannot tell "null" from "not set". That applies to a `null`
      `geometry` and to a key nested inside `assets` just as much as to a
      top-level property.
    - Every item gains a `properties` dict, empty when the row held no property
      values, even if the written item carried no `properties` key at all.

    Args:
        path: Path to a `.parquet` file in the STAC-GeoParquet 1.1 layout.

    Returns:
        The list of STAC Item dicts (ready for
        :meth:`pyramids.dataset.DatasetCollection.from_stac`). Every item
        carries a `properties` dict, empty when the row held no property values.

    Raises:
        OptionalPackageDoesNotExist: When pyarrow (the `[parquet]` extra) is not
            installed.
        ValueError: When the file is not in the spec layout — the pyramids
            JSON-blob layout (read it with :func:`from_geoparquet` instead), or
            a Parquet file carrying neither the `stac-geoparquet` metadata nor
            any reserved spec column.

    See Also:
        to_geoparquet_spec: The matching writer.
        from_geoparquet: Reads either layout, auto-detecting the spec one.
    """
    _, parquet = _require_parquet()

    table = parquet.read_table(str(path))
    metadata = table.schema.metadata or {}
    _require_spec_layout(path, metadata, table.column_names)
    hints: dict[str, Any] = {}
    raw_hints = metadata.get(_PYRAMIDS_METADATA_KEY)
    if raw_hints:
        hints = json.loads(raw_hints)
    json_columns = set(hints.get("json_columns") or ())
    root_columns = set(hints.get("root_columns") or ()) | set(_ROOT_COLUMNS)

    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    items: list[dict[str, Any]] = []
    for row in range(table.num_rows):
        item: dict[str, Any] = {}
        properties: dict[str, Any] = {}
        for name, values in columns.items():
            value = values[row]
            if value is None:
                continue
            if name == _GEOMETRY_COLUMN:
                item[_GEOMETRY_COLUMN] = json.loads(
                    shapely.to_geojson(shapely.from_wkb(value))
                )
            elif name == _BBOX_COLUMN:
                item[_BBOX_COLUMN] = _bbox_from_struct(value)
            else:
                decoded = (
                    json.loads(value)
                    if name in json_columns and isinstance(value, str)
                    else value
                )
                target = item if name in root_columns else properties
                target[name] = _from_arrow(decoded)
        item["properties"] = properties
        items.append(item)
    return items


def _require_spec_layout(
    path: str | Path, metadata: Mapping[bytes, bytes], column_names: list[str]
) -> None:
    """Refuse a file the spec reader would otherwise decode into nonsense.

    Without this check a JSON-blob file read with `spec=True` comes back as
    `{"id": ..., "properties": {"stac_item": "<raw json string>"}}` and no
    exception — garbage that looks like a result. A file written by another
    STAC-GeoParquet writer is still accepted: it is recognised by its reserved
    spec columns, not by pyramids' own metadata block.

    Args:
        path: The file being read, quoted in the message.
        metadata: The table's key-value file metadata.
        column_names: The table's column names.

    Raises:
        ValueError: When the file carries the JSON-blob layout's `stac_item`
            column, or neither the `stac-geoparquet` metadata key nor any
            reserved spec column.
    """
    if _STAC_METADATA_KEY not in metadata:
        if _ITEM_COLUMN in column_names:
            raise ValueError(
                f"{path} is in the pyramids JSON-blob layout (it carries the "
                f"{_ITEM_COLUMN!r} column), not the STAC-GeoParquet spec "
                "layout. Read it with from_geoparquet(path) instead."
            )
        if not set(_ROOT_COLUMNS) & set(column_names):
            raise ValueError(
                f"{path} is not a STAC-GeoParquet file: it carries neither the "
                "'stac-geoparquet' file metadata nor any of the reserved spec "
                f"columns ({', '.join(_ROOT_COLUMNS)})."
            )


def _bbox_from_struct(value: Mapping[str, Any]) -> list[float]:
    """Flatten a STAC-GeoParquet `bbox` struct back to the STAC bbox list.

    Args:
        value: The struct read out of the `bbox` column, keyed `xmin` / `ymin` /
            `xmax` / `ymax` and optionally `zmin` / `zmax`.

    Returns:
        list: `[xmin, ymin, xmax, ymax]`, or `[xmin, ymin, zmin, xmax, ymax,
            zmax]` for a 3D struct, in the order STAC mandates.
    """
    order: tuple[str, ...] = ("xmin", "ymin", "xmax", "ymax")
    if value.get("zmin") is not None:
        order = ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")
    return [value[name] for name in order]
