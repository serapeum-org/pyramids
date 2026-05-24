"""Serialize STAC Items to/from GeoParquet (PD-3).

stac-geoparquet stores a STAC ItemCollection as one columnar GeoParquet file
(geometry as WKB, WGS84) for bulk transfer + fast spatial filtering, avoiding
thousands of per-item JSON requests. pyramids already has geopandas (core) and a
`FeatureCollection` (a `GeoDataFrame` subclass) with GeoParquet I/O, plus the
`[parquet]` extra (pyarrow) — so the round-trip needs **no new dependency**.

This is a lossless pyramids variant: each row carries the item geometry (so the
file is a valid, spatially-filterable GeoParquet) plus the full STAC Item as a
JSON column, so :func:`from_geoparquet` reconstructs the exact item dicts —
ready to feed :meth:`pyramids.dataset.DatasetCollection.from_stac`.

Requires the `[parquet]` extra (pyarrow) for the Parquet read/write itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import shapely.geometry

_ITEM_COLUMN = "stac_item"


def _item_to_dict(item: Any) -> dict[str, Any]:
    """Return a STAC Item as a plain dict (calls `.to_dict()` when available)."""
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, dict):
        return item
    raise TypeError(
        f"STAC item must be a dict or expose to_dict(), got {type(item).__name__}."
    )


def _item_geometry(item: dict[str, Any]) -> Any:
    """Build a shapely geometry from an item's `geometry` or `bbox`."""
    geom = item.get("geometry")
    if geom:
        return shapely.geometry.shape(geom)
    bbox = item.get("bbox")
    if bbox:
        b = list(bbox)
        half = len(b) // 2
        return shapely.geometry.box(b[0], b[1], b[half], b[half + 1])
    return None


def to_geoparquet(items: Any, path: str | Path) -> None:
    """Write a sequence of STAC Items to a GeoParquet file.

    Each item becomes a row carrying its geometry (a valid, spatially-filterable
    GeoParquet geometry in EPSG:4326) and the full item as a JSON column.

    Args:
        items: Iterable of STAC Items (`pystac.Item` objects with `to_dict()`,
            or raw STAC-JSON dicts — e.g. from
            :meth:`pyramids.dataset.Dataset.to_stac_item`).
        path: Destination `.parquet` path.

    Raises:
        ValueError: When `items` is empty.
        OptionalPackageDoesNotExist: When pyarrow (the `[parquet]` extra) is not
            installed (raised by `FeatureCollection.to_parquet`).

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
    """
    from pyramids.feature import FeatureCollection

    rows = []
    geometries = []
    for item in items:
        as_dict = _item_to_dict(item)
        geometries.append(_item_geometry(as_dict))
        rows.append({"id": as_dict.get("id"), _ITEM_COLUMN: json.dumps(as_dict)})

    if not rows:
        raise ValueError("to_geoparquet received no items.")

    fc = FeatureCollection(rows, geometry=geometries, crs="EPSG:4326")
    fc.to_parquet(str(path))


def from_geoparquet(path: str | Path) -> list[dict[str, Any]]:
    """Read STAC Items back from a GeoParquet written by :func:`to_geoparquet`.

    Args:
        path: Path to a `.parquet` file produced by :func:`to_geoparquet`.

    Returns:
        The list of STAC Item dicts (ready for
        :meth:`pyramids.dataset.DatasetCollection.from_stac`).

    Raises:
        OptionalPackageDoesNotExist: When pyarrow (the `[parquet]` extra) is not
            installed (raised by `FeatureCollection.read_parquet`).
        KeyError: When the file lacks the `stac_item` JSON column (not written
            by :func:`to_geoparquet`).
    """
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_parquet(str(path))
    return [json.loads(blob) for blob in fc[_ITEM_COLUMN]]
