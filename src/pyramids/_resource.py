"""Decompress-aware resource reader.

A single entry point, :func:`read_resource`, that takes a local file path
(optionally plus a format hint), sniffs the format, transparently handles
``.gz`` / ``.zip`` / ``.tar`` containers, and returns the appropriate pyramids
type:

* raster  (``.tif`` / ``.tiff`` / ``.cog`` / ``.nc`` / ``.nc4`` / ``.vrt``)
  → :class:`pyramids.dataset.Dataset`
* vector  (``.gpkg`` / ``.shp`` / ``.geojson`` / ``.gdb`` / ``.kml`` / …)
  → :class:`pyramids.feature.FeatureCollection`
* tabular (``.csv`` / ``.tsv`` / ``.xlsx`` / ``.xls`` / ``.parquet``)
  → :class:`pandas.DataFrame`

This is a thin sniff-and-dispatch shim over the existing per-family readers
(:meth:`Dataset.read_file`, :meth:`FeatureCollection.read_file`) plus a
:mod:`pandas` branch for tabular formats. Decompression is delegated to GDAL's
virtual filesystem via :mod:`pyramids._io` — no new decompression code lives
here.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Union

import pandas as pd

from pyramids import _io
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

ResourceKind = Literal["raster", "vector", "tabular"]

# Compression extensions that wrap a single inner file — peel one layer to read
# the inner suffix (e.g. ``kontur.gpkg.gz`` → ``.gpkg``).
_GZIP_SUFFIXES = {".gz"}
# Container extensions whose members must be inspected to know the family —
# a bare ``.zip`` name does not say whether it holds a raster, vector, or table.
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz"}

_RASTER_SUFFIXES = {".tif", ".tiff", ".cog", ".nc", ".nc4", ".vrt"}
_VECTOR_SUFFIXES = {
    ".gpkg",
    ".shp",
    ".geojson",
    ".json",
    ".gdb",
    ".kml",
    ".fgb",
    ".gml",
}
_TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".pq"}

# CKAN / source format labels → family. Keys are lower-cased and stripped of any
# leading dot; used as a tiebreaker when the suffix alone is ambiguous (notably
# a bare ``.zip``).
_FMT_TO_KIND: dict[str, ResourceKind] = {
    "geotiff": "raster",
    "gtiff": "raster",
    "tif": "raster",
    "tiff": "raster",
    "cog": "raster",
    "netcdf": "raster",
    "nc": "raster",
    "vrt": "raster",
    "geopackage": "vector",
    "gpkg": "vector",
    "shapefile": "vector",
    "shp": "vector",
    "geojson": "vector",
    "json": "vector",
    "kml": "vector",
    "gdb": "vector",
    "fgb": "vector",
    "flatgeobuf": "vector",
    "gml": "vector",
    "csv": "tabular",
    "tsv": "tabular",
    "txt": "tabular",
    "excel": "tabular",
    "xlsx": "tabular",
    "xls": "tabular",
    "parquet": "tabular",
}


def _strip_compression(name: str) -> str:
    """Peel one ``.gz`` layer so the inner suffix can be sniffed.

    ``"kontur.gpkg.gz"`` → ``"kontur.gpkg"``; everything else is returned
    unchanged (``.zip`` / ``.tar`` are containers, handled separately).
    """
    p = Path(name)
    inner = p.stem if p.suffix.lower() in _GZIP_SUFFIXES else name
    return inner


def sniff_kind(path: Union[str, Path], fmt: str | None = None) -> ResourceKind:
    """Determine the resource family from a path (+ optional format hint).

    Peels one ``.gz`` compression layer to read the inner suffix, maps that
    suffix to ``"raster"`` / ``"vector"`` / ``"tabular"``, and falls back to the
    ``fmt`` label when the suffix is unknown or ambiguous (e.g. a bare ``.zip``,
    whose contents are not implied by the name).

    This is a pure, name-based helper — it performs no I/O. For an ambiguous
    container with no usable ``fmt``, :func:`read_resource` peeks inside the
    archive instead.

    Args:
        path: Path to the (possibly compressed) resource.
        fmt: Optional source format label (e.g. a CKAN ``"Geopackage"`` /
            ``"GeoTIFF"`` / ``"CSV"`` string) used as a tiebreaker.

    Returns:
        ResourceKind: ``"raster"``, ``"vector"``, or ``"tabular"``.

    Raises:
        ValueError: When the family cannot be determined from the suffix or
            ``fmt``.

    Examples:
        - A GeoTIFF sniffs as raster::

            >>> from pyramids._resource import sniff_kind
            >>> sniff_kind("worldpop.tif")
            'raster'

        - A gzipped GeoPackage peels one layer to the inner ``.gpkg``::

            >>> sniff_kind("kontur.gpkg.gz")
            'vector'

        - A bare ``.zip`` leans on the format label::

            >>> sniff_kind("meta_hrsl.zip", fmt="GeoTIFF")
            'raster'
    """
    inner = Path(_strip_compression(str(path)))
    ext = inner.suffix.lower()
    kind: ResourceKind | None = None
    if ext in _RASTER_SUFFIXES:
        kind = "raster"
    elif ext in _VECTOR_SUFFIXES:
        kind = "vector"
    elif ext in _TABULAR_SUFFIXES:
        kind = "tabular"
    if kind is None and fmt is not None:
        kind = _FMT_TO_KIND.get(fmt.strip().lower().lstrip("."))
    if kind is None:
        raise ValueError(
            f"could not determine the resource kind from {str(path)!r}"
            f"{'' if fmt is None else f' (fmt={fmt!r})'}; pass an explicit "
            "kind='raster'|'vector'|'tabular' or a recognised fmt label."
        )
    return kind


def _sniff_from_archive(path: Path) -> ResourceKind | None:
    """Peek inside a ``.zip`` / ``.tar`` and infer the family from its members.

    Best-effort: returns ``None`` (rather than raising) when the archive cannot
    be listed or holds no member with a recognised suffix, so the caller can
    surface a single, clear error.
    """
    kind: ResourceKind | None = None
    if path.suffix.lower() in _ARCHIVE_SUFFIXES:
        try:
            members = _io.archive_members(_io.archive_dir_vsi(path, "auto"))
        except (FileFormatNotSupportedError, FileNotFoundError, RuntimeError):
            members = []
        for member in members:
            ext = Path(member).suffix.lower()
            if ext in _RASTER_SUFFIXES:
                kind = "raster"
                break
            if ext in _VECTOR_SUFFIXES:
                kind = "vector"
                break
            if ext in _TABULAR_SUFFIXES:
                kind = "tabular"
                break
    return kind


def _determine_kind(path: Path, fmt: str | None) -> ResourceKind:
    """Resolve the family from name + ``fmt``, peeking into archives as needed."""
    try:
        kind = sniff_kind(path, fmt=fmt)
    except ValueError:
        kind = _sniff_from_archive(path)
        if kind is None:
            raise
    return kind


def _warn_if_multilayer(path: Path) -> None:
    """Warn when a vector source exposes more than one layer / vector member.

    Covers both multi-layer single files (GeoPackage, GDB, KML — via
    :meth:`FeatureCollection.list_layers`) and archives bundling several vector
    members (e.g. a HOTOSM ``.zip`` of Shapefiles — via the archive member list).
    Never raises: the warning is advisory and must not block the read.
    """
    names: list[str] = []
    if path.suffix.lower() in _ARCHIVE_SUFFIXES:
        try:
            members = _io.archive_members(_io.archive_dir_vsi(path, "auto"))
            names = [m for m in members if Path(m).suffix.lower() in _VECTOR_SUFFIXES]
        except (FileFormatNotSupportedError, FileNotFoundError, RuntimeError):
            names = []
    else:
        try:
            names = FeatureCollection.list_layers(str(path))
        except Exception:  # noqa: BLE001 — advisory check, any failure → no warning
            names = []
    if len(names) > 1:
        warnings.warn(
            f"{str(path)!r} contains {len(names)} layers {names!r}; reading the "
            f"first ({names[0]!r}). Pass layer=<name|index> to choose a specific "
            "one, or call FeatureCollection.list_layers(path) to enumerate them.",
            stacklevel=3,
        )


def _read_vector(
    path: Path, layer: Union[str, int, None]
) -> FeatureCollection:
    """Read a vector resource, applying the multi-layer policy.

    When ``layer`` is not specified and the source exposes more than one layer,
    warn and read the first (driver-default) layer — never silently drop data.
    """
    if layer is None:
        _warn_if_multilayer(path)
    return FeatureCollection.read_file(str(path), layer=layer)


def _read_tabular(path: Path) -> pd.DataFrame:
    """Read a tabular resource into a :class:`pandas.DataFrame`.

    ``pandas`` infers ``.gz`` / ``.zip`` compression for CSV/TSV from the
    suffix. ``.xlsx`` needs ``openpyxl`` and ``.parquet`` needs ``pyarrow`` —
    when missing, the underlying :class:`ImportError` is re-raised with install
    guidance.
    """
    source = str(path)
    ext = Path(_strip_compression(path.name)).suffix.lower()
    result: pd.DataFrame
    if ext in {".csv", ".txt"}:
        result = pd.read_csv(source)
    elif ext == ".tsv":
        result = pd.read_csv(source, sep="\t")
    elif ext in {".xlsx", ".xls"}:
        try:
            result = pd.read_excel(source)
        except ImportError as exc:  # openpyxl / xlrd not installed
            raise ImportError(
                f"reading {ext} files needs an Excel engine (e.g. 'openpyxl'); "
                "install it into the environment to read this resource."
            ) from exc
    elif ext in {".parquet", ".pq"}:
        try:
            result = pd.read_parquet(source)
        except ImportError as exc:  # pyarrow / fastparquet not installed
            raise ImportError(
                "reading .parquet files needs a parquet engine (e.g. 'pyarrow'); "
                "install it into the environment to read this resource."
            ) from exc
    else:
        raise ValueError(
            f"unsupported tabular suffix {ext!r} for {source!r}; expected one of "
            f"{sorted(_TABULAR_SUFFIXES)}."
        )
    return result


def read_resource(
    path: Union[str, Path],
    fmt: str | None = None,
    *,
    kind: ResourceKind | None = None,
    layer: Union[str, int, None] = None,
) -> Union[Dataset, FeatureCollection, pd.DataFrame]:
    """Read a downloaded resource into the appropriate pyramids type.

    Sniffs the format (suffix + ``fmt``, peeking inside ``.zip`` / ``.tar`` when
    needed), transparently handles ``.gz`` / ``.zip`` / ``.tar`` containers via
    GDAL's virtual filesystem, and dispatches:

    * raster  → :class:`pyramids.dataset.Dataset`
    * vector  → :class:`pyramids.feature.FeatureCollection`
    * tabular → :class:`pandas.DataFrame`

    Args:
        path: Local path to the (possibly compressed) resource.
        fmt: Optional source format label (e.g. a CKAN ``"Geopackage"`` /
            ``"GeoTIFF"`` / ``"CSV"`` string) used as a tiebreaker when the
            suffix is ambiguous — notably a bare ``.zip`` whose contents are not
            implied by the name.
        kind: Optional explicit family override (``"raster"`` / ``"vector"`` /
            ``"tabular"``) when both suffix and ``fmt`` are unreliable.
        layer: For multi-layer vector containers (a ``.gpkg`` / ``.gdb`` with
            several layers, or a ``.zip`` of Shapefiles), select by name or
            index. ``None`` reads the first / default layer and warns when more
            exist (data is never silently dropped).

    Returns:
        Dataset | FeatureCollection | pandas.DataFrame: The resource read into
        its pyramids / pandas type.

    Raises:
        ValueError: When the format cannot be determined or is unsupported.

    Examples:
        - Read a WorldPop GeoTIFF as a raster ``Dataset``::

            >>> from pyramids._resource import read_resource
            >>> ds = read_resource("worldpop.tif")  # doctest: +SKIP

        - Read a gzipped Kontur GeoPackage as a ``FeatureCollection``::

            >>> fc = read_resource("kontur.gpkg.gz")  # doctest: +SKIP

        - Read a zipped GeoTIFF, disambiguated by the CKAN format label::

            >>> ds = read_resource("meta_hrsl.zip", fmt="GeoTIFF")  # doctest: +SKIP

    See Also:
        - :func:`sniff_kind`: the name-based family classifier used here.
        - :meth:`pyramids.dataset.Dataset.read_file`: the raster reader.
        - :meth:`pyramids.feature.FeatureCollection.read_file`: the vector reader.
    """
    path = Path(path)
    resolved_kind = kind if kind is not None else _determine_kind(path, fmt)
    if resolved_kind == "raster":
        result: Union[Dataset, FeatureCollection, pd.DataFrame] = Dataset.read_file(
            str(path)
        )
    elif resolved_kind == "vector":
        result = _read_vector(path, layer)
    elif resolved_kind == "tabular":
        result = _read_tabular(path)
    else:
        raise ValueError(
            f"unsupported resource kind {resolved_kind!r}; expected one of "
            "'raster', 'vector', 'tabular'."
        )
    return result
