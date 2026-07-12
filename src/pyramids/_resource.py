"""Decompress-aware resource reader.

A single entry point, :func:`read_resource`, that takes a local file path
(optionally plus a format hint), sniffs the format, transparently handles
``.gz`` / ``.zip`` / ``.tar`` / ``.tar.gz`` containers, and returns the
appropriate pyramids type:

* raster  (``.tif`` / ``.tiff`` / ``.cog`` / ``.nc`` / ``.nc4`` / ``.vrt``)
  → :class:`pyramids.dataset.Dataset`
* vector  (``.gpkg`` / ``.shp`` / ``.geojson`` / ``.gdb`` / ``.kml`` / …)
  → :class:`pyramids.feature.FeatureCollection`
* tabular (``.csv`` / ``.tsv`` / ``.xlsx`` / ``.xls`` / ``.parquet``)
  → :class:`pandas.DataFrame`

This is a thin sniff-and-dispatch shim over the existing per-family readers
(:meth:`Dataset.read_file`, :meth:`FeatureCollection.read_file`) plus a
:mod:`pandas` branch for tabular formats. Decompression is delegated to GDAL's
virtual filesystem via :mod:`pyramids._io` (for raster/vector) and to
:mod:`pandas`' own compression inference (for tabular) — no new decompression
code lives here.

Note on ``.json``: a ``.json`` suffix is assumed to be GeoJSON and routed to the
vector reader. For a non-spatial JSON table, pass ``kind="tabular"`` (or a
tabular ``fmt``) explicitly.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import pandas as pd

from pyramids import _io
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

ResourceKind = Literal["raster", "vector", "tabular"]

# Single-file compression layer — peel it to read the inner suffix
# (e.g. ``kontur.gpkg.gz`` → ``.gpkg``).
_GZIP_SUFFIXES = {".gz"}
# Container extensions whose members must be inspected to know the family and to
# target the right member. ``.tar.gz`` is matched separately (its ``Path.suffix``
# is ``.gz``) via :func:`_is_archive`.
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz"}
# ``.tar.gz`` needs its own compound suffix — ``Path.suffix`` only sees ``.gz`` —
# so name it once and reuse it across the archive checks (S1192).
_TAR_GZ_SUFFIX = ".tar.gz"

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
_TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".pq"}

# kind → the suffix set used to pick the matching member inside an archive.
_KIND_SUFFIXES: dict[ResourceKind, set[str]] = {
    "raster": _RASTER_SUFFIXES,
    "vector": _VECTOR_SUFFIXES,
    "tabular": _TABULAR_SUFFIXES,
}

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
    "excel": "tabular",
    "xlsx": "tabular",
    "xls": "tabular",
    "parquet": "tabular",
}


def _is_archive(path: Path) -> bool:
    """True when ``path`` is a multi-member container (``.zip``/``.tar``/…).

    ``.tar.gz`` is included even though its :attr:`~pathlib.Path.suffix` is
    ``.gz`` — GDAL's ``/vsitar/`` handler decompresses it inline.
    """
    name = path.name.lower()
    return name.endswith(_TAR_GZ_SUFFIX) or Path(name).suffix in _ARCHIVE_SUFFIXES


def _strip_compression(name: str) -> str:
    """Peel one compression / container layer so the inner suffix can be sniffed.

    ``"kontur.gpkg.gz"`` → ``"kontur.gpkg"``, ``"rivers.shp.zip"`` →
    ``"rivers.shp"``, ``"noah.tif.tar.gz"`` → ``"noah.tif"``. A bare container
    with no inner name (``"download.zip"``) peels to its stem (``"download"``),
    which carries no usable suffix — the caller then leans on ``fmt`` or peeks
    inside the archive.
    """
    low = name.lower()
    if low.endswith(_TAR_GZ_SUFFIX):
        inner = name[: -len(_TAR_GZ_SUFFIX)]
    else:
        p = Path(name)
        inner = (
            p.stem if p.suffix.lower() in _GZIP_SUFFIXES | _ARCHIVE_SUFFIXES else name
        )
    return inner


def sniff_kind(path: str | Path, fmt: str | None = None) -> ResourceKind:
    """Determine the resource family from a path (+ optional format hint).

    Peels one compression / container layer to read the inner suffix, maps that
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

        - A zipped Shapefile whose name carries the inner suffix::

            >>> sniff_kind("rivers.shp.zip")
            'vector'

        - A bare ``.zip`` leans on the format label::

            >>> sniff_kind("meta_hrsl.zip", fmt="GeoTIFF")
            'raster'
    """
    inner = Path(_strip_compression(str(Path(path).name)))
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


def _archive_members_for_kind(path: Path, kind: ResourceKind) -> list[str]:
    """List archive members whose suffix matches ``kind`` (best-effort).

    Returns ``[]`` (rather than raising) when the archive cannot be listed, so
    callers can fall back to the whole-archive path.
    """
    try:
        members = _io.archive_members(_io.archive_dir_vsi(path, "auto"))
    except (FileFormatNotSupportedError, FileNotFoundError, RuntimeError):
        members = []
    suffixes = _KIND_SUFFIXES[kind]
    return [m for m in members if Path(m).suffix.lower() in suffixes]


def _sniff_from_archive(path: Path) -> ResourceKind | None:
    """Peek inside a container and infer the family from the first known member.

    Best-effort: returns ``None`` when the archive cannot be listed or holds no
    member with a recognised suffix, so the caller can surface a single, clear
    error.
    """
    kind: ResourceKind | None = None
    if _is_archive(path):
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


def _read_raster(path: Path) -> Dataset:
    """Read a raster, targeting the matching member when ``path`` is an archive.

    For a ``.zip`` / ``.tar`` / ``.tar.gz`` the raster member is selected
    explicitly (``<archive>/<member>``) rather than relying on the archive
    handler's first-member default — which, for a tar, would otherwise open the
    container directory and fail.
    """
    source = str(path)
    if _is_archive(path):
        members = _archive_members_for_kind(path, "raster")
        if members:
            source = f"{path}/{members[0]}"
    return Dataset.read_file(source)


def _warn_if_multilayer(path: Path) -> None:
    """Warn when a single vector file exposes more than one layer (GPKG/GDB/KML).

    Never raises: the warning is advisory and must not block the read. Archive
    containers are handled separately by :func:`_select_vector_member`.
    """
    try:
        names = FeatureCollection.list_layers(str(path))
    except Exception:  # noqa: BLE001 — advisory check; any failure → no warning
        names = []
    if len(names) > 1:
        warnings.warn(
            f"{str(path)!r} contains {len(names)} layers {names!r}; reading the "
            f"first ({names[0]!r}). Pass layer=<name|index> to choose a specific "
            "one, or call FeatureCollection.list_layers(path) to enumerate them.",
            stacklevel=3,
        )


def _select_vector_member(
    members: list[str], layer: str | int | None, path: Path
) -> tuple[str, str | int | None]:
    """Pick the archive member to read and the layer to forward to the reader.

    A vector archive may hold several vector files (e.g. a HOTOSM ``.zip`` of
    Shapefiles). ``layer`` selects one by member stem (str) or index (int);
    ``None`` reads the first and warns when more exist. When ``layer`` names a
    member it is consumed here (the member *is* the selection), so ``None`` is
    forwarded as the internal layer.
    """
    stems = {Path(m).stem: m for m in members}
    member = members[0]
    passthrough = layer
    if isinstance(layer, str) and layer in stems:
        member = stems[layer]
        passthrough = None
    elif isinstance(layer, int) and not isinstance(layer, bool):
        if not 0 <= layer < len(members):
            raise IndexError(
                f"layer index {layer} is out of range for the {len(members)} "
                f"vector member(s) in {str(path)!r}: {members!r}"
            )
        member = members[layer]
        passthrough = None
    elif layer is None and len(members) > 1:
        warnings.warn(
            f"{str(path)!r} contains {len(members)} vector members {members!r}; "
            f"reading the first ({members[0]!r}). Pass layer=<name|index> to "
            "choose a specific one.",
            stacklevel=3,
        )
    return member, passthrough


def _read_vector(path: Path, layer: str | int | None) -> FeatureCollection:
    """Read a vector resource, applying the multi-layer / multi-member policy.

    Plain multi-layer files (GPKG/GDB) default to the first layer and warn;
    ``layer=`` selects. Archive containers resolve to a specific vector member
    (``<archive>/<member>``) so a zipped Shapefile reads its ``.shp`` rather than
    an alphabetically-first sidecar.
    """
    source = str(path)
    passthrough_layer = layer
    if _is_archive(path):
        members = _archive_members_for_kind(path, "vector")
        if members:
            member, passthrough_layer = _select_vector_member(members, layer, path)
            source = f"{path}/{member}"
    elif layer is None:
        _warn_if_multilayer(path)
    return FeatureCollection.read_file(source, layer=passthrough_layer)


def _read_tabular(path: Path) -> pd.DataFrame:
    """Read a tabular resource into a :class:`pandas.DataFrame`.

    ``pandas`` infers ``.gz`` / ``.zip`` / ``.tar`` compression from the suffix,
    so the path is passed through verbatim. ``.xlsx`` needs ``openpyxl`` and
    ``.parquet`` needs ``pyarrow`` — when missing, the underlying
    :class:`ImportError` is re-raised with install guidance.
    """
    source = str(path)
    ext = Path(_strip_compression(path.name)).suffix.lower()
    result: pd.DataFrame
    if ext == ".csv":
        result = pd.read_csv(source)
    elif ext == ".tsv":
        result = pd.read_csv(source, sep="\t")
    elif ext in {".xlsx", ".xls"}:
        try:
            result = pd.read_excel(source)
        except ImportError as exc:  # openpyxl (.xlsx) / xlrd (.xls) not installed
            raise ImportError(
                f"reading {ext} files needs an Excel engine (openpyxl for .xlsx, "
                "xlrd for legacy .xls); install it into the environment to read "
                "this resource."
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
    path: str | Path,
    fmt: str | None = None,
    *,
    kind: ResourceKind | None = None,
    layer: str | int | None = None,
) -> Dataset | FeatureCollection | pd.DataFrame:
    """Read a downloaded resource into the appropriate pyramids type.

    Sniffs the format (suffix + ``fmt``, peeking inside ``.zip`` / ``.tar`` when
    needed), transparently handles ``.gz`` / ``.zip`` / ``.tar`` / ``.tar.gz``
    containers, and dispatches:

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
            exist (data is never silently dropped). Ignored — with a warning —
            for raster and tabular resources.

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
    if layer is not None and resolved_kind != "vector":
        warnings.warn(
            f"layer={layer!r} is ignored for {resolved_kind} resources; it only "
            "applies to vector data.",
            stacklevel=2,
        )
    if resolved_kind == "raster":
        result: Dataset | FeatureCollection | pd.DataFrame = _read_raster(path)
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
