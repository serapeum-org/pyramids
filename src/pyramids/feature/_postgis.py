"""PostGIS read/write for :class:`~pyramids.feature.FeatureCollection`.

Implementation behind :meth:`pyramids.feature.FeatureCollection.from_postgis` and
:meth:`~pyramids.feature.FeatureCollection.to_postgis`. The transport is GDAL's
native **OGR PostgreSQL driver** (the ``PG:`` connection string) — no SQLAlchemy /
psycopg dependency; the features decode through the same OGR / pyogrio reader that
backs :class:`FeatureCollection`, exactly like :mod:`pyramids.feature._wfs`.

The OGR PostgreSQL driver ships in the conda-forge ``libgdal-pg`` package; if it is
absent from the GDAL build, :func:`_require_pg_driver` raises a clear
:class:`~pyramids.base._errors.PostGISError` pointing at the fix. Provider /
deployment specifics (cloud-SQL auth, pooling) are the caller's responsibility via
the connection string (see ``docs/SCOPE.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import geopandas as gpd
from osgeo import ogr

from pyramids.base._errors import PostGISError

if TYPE_CHECKING:
    from pyramids.feature.collection import FeatureCollection

_PG_DRIVER = "PostgreSQL"

# pyogrio/fiona write modes keyed by the user-facing if_exists value.
_IF_EXISTS_MODES = {"fail", "replace", "append"}


def _require_pg_driver() -> None:
    """Raise if the GDAL OGR PostgreSQL driver is not available in this build."""
    if ogr.GetDriverByName(_PG_DRIVER) is None:
        raise PostGISError(
            "the GDAL OGR PostgreSQL driver is not available in this build; install "
            "the conda-forge 'libgdal-pg' package (e.g. `pixi add libgdal-pg`)."
        )


def _pg_connection(connection: str) -> str:
    """Normalise a connection to a GDAL ``PG:`` datasource string.

    Accepts an already-prefixed ``PG:host=… dbname=…`` string (used as-is) or a
    bare ``host=… dbname=…`` keyword string (prefixed with ``PG:``).
    """
    return connection if connection.startswith("PG:") else f"PG:{connection}"


def _qualified_table(table: str, schema: str | None) -> str:
    """Return ``schema.table`` when a schema is given, else ``table``."""
    return f"{schema}.{table}" if schema else table


def _read_kwargs(
    bbox: tuple[float, float, float, float] | None,
    where: str | None,
    columns: list[str] | None,
    max_features: int | None,
) -> dict[str, Any]:
    """Assemble the pyogrio / GDAL read filters (bbox, where, columns, count)."""
    kwargs: dict[str, Any] = {}
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(f"bbox must be (minx, miny, maxx, maxy), got {bbox!r}")
        minx, miny, maxx, maxy = (float(v) for v in bbox)
        if minx >= maxx or miny >= maxy:
            raise ValueError(f"bbox must have minx < maxx and miny < maxy, got {bbox!r}")
        kwargs["bbox"] = (minx, miny, maxx, maxy)
    if where is not None:
        kwargs["where"] = where
    if columns is not None:
        kwargs["columns"] = columns
    if max_features is not None:
        if max_features < 0:
            raise ValueError(f"max_features must be >= 0 or None, got {max_features}")
        kwargs["rows"] = max_features
    return kwargs


def from_postgis(
    featurecollection_cls: type["FeatureCollection"],
    connection: str,
    *,
    table: str | None = None,
    sql: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    where: str | None = None,
    columns: list[str] | None = None,
    max_features: int | None = None,
) -> "FeatureCollection":
    """Read a PostGIS table or query into a :class:`FeatureCollection`.

    The public API is the :meth:`pyramids.feature.FeatureCollection.from_postgis`
    classmethod, which forwards here. See it for the full parameter documentation.

    Raises:
        ValueError: neither / both of ``table`` and ``sql`` were given, ``bbox`` is
            malformed, or ``max_features`` is negative.
        PostGISError: the PostgreSQL driver is unavailable or the read failed.
    """
    if (table is None) == (sql is None):
        raise ValueError("from_postgis: provide exactly one of `table` or `sql`")
    _require_pg_driver()
    conn = _pg_connection(connection)
    read_kwargs = _read_kwargs(bbox, where, columns, max_features)
    try:
        if sql is not None:
            # pyogrio treats `columns` and `sql` as mutually exclusive — select
            # the columns inside the query instead.
            read_kwargs.pop("columns", None)
            gdf = gpd.read_file(conn, sql=sql, **read_kwargs)
        else:
            gdf = gpd.read_file(conn, layer=table, **read_kwargs)
    except Exception as exc:  # noqa: BLE001 — normalise any read failure to PostGISError
        target = sql if sql is not None else table
        raise PostGISError(f"PostGIS read failed for {target!r}: {exc}") from exc
    return featurecollection_cls(gdf)


def to_postgis(
    fc: "FeatureCollection",
    connection: str,
    *,
    table: str,
    if_exists: str = "fail",
    schema: str | None = None,
    geometry_column: str = "geom",
    srid: int | None = None,
) -> None:
    """Write a :class:`FeatureCollection` to a PostGIS table.

    The public API is the :meth:`pyramids.feature.FeatureCollection.to_postgis`
    method, which forwards here. See it for the full parameter documentation.

    Raises:
        ValueError: ``if_exists`` is not one of ``"fail"`` / ``"replace"`` /
            ``"append"``.
        PostGISError: the PostgreSQL driver is unavailable or the write failed.
    """
    if if_exists not in _IF_EXISTS_MODES:
        raise ValueError(
            f"if_exists must be one of {sorted(_IF_EXISTS_MODES)}, got {if_exists!r}"
        )
    _require_pg_driver()
    conn = _pg_connection(connection)
    layer = _qualified_table(table, schema)
    # FeatureCollection.to_file takes (driver, layer, mode, **creation_options); PG
    # layer creation options (GEOMETRY_NAME, SRID) are passed as individual kwargs.
    creation_options: dict[str, str] = {"GEOMETRY_NAME": geometry_column}
    if srid is not None:
        creation_options["SRID"] = str(int(srid))
    mode = "a" if if_exists == "append" else "w"
    try:
        fc.to_file(conn, driver=_PG_DRIVER, layer=layer, mode=mode, **creation_options)
    except Exception as exc:  # noqa: BLE001 — normalise any write failure to PostGISError
        raise PostGISError(f"PostGIS write failed for {layer!r}: {exc}") from exc
