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
# Named once and reused across the three tables below, which all key on it
# (S1192).
_PARQUET_SUFFIX = ".parquet"
_TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", _PARQUET_SUFFIX, ".pq"}

# One detection core for both public readers (`read_resource` here and
# `pyramids.io.load_resource`, which adapts these helpers). Previously each
# module carried its own suffix table and its own sniffing rules, so the two
# drifted apart; keep new formats in these tables only.
#
# Deliberately NOT a superset of the two originals. Excel is absent: `.xlsx` is
# a zip container, so its magic resolves to `"zip"` and an `"excel"` token would
# be unreachable by sniffing anyway, while adding `.xls` would turn a resource
# that used to come back as raw bytes into a hard ImportError (no Excel engine
# is a declared dependency). Excel is still reachable deliberately — via
# `read_resource`'s `_TABULAR_SUFFIXES` family lookup, or by passing an explicit
# `fmt=`/`kind=` — just never by sniffing.
_EXT_TO_FORMAT: dict[str, str] = {
    ".shp": "shp",
    ".gpkg": "gpkg",
    ".geojson": "geojson",
    ".json": "geojson",
    ".csv": "csv",
    ".tsv": "tsv",
    _PARQUET_SUFFIX: "parquet",
    ".pq": "parquet",
    ".nc": "nc",
    ".nc4": "nc",
    ".cdf": "nc",
    ".tif": "tif",
    ".tiff": "tif",
    ".grib": "grib",
    ".grib2": "grib",
    ".grb": "grib",
    ".grb2": "grib",
    ".zip": "zip",
}

# Extension -> the pandas reader `read_tabular` should use. Separate from
# `_EXT_TO_FORMAT` because it covers the Excel suffixes that deliberately stay
# out of the sniffing vocabulary above.
_EXT_TO_TABULAR_READER: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".xlsx": "excel",
    ".xls": "excel",
    _PARQUET_SUFFIX: "parquet",
    ".pq": "parquet",
}

# Format label -> the pandas reader to use. Keys are normalised (lower-cased,
# leading dot stripped) so the CKAN spellings that portals actually publish —
# "CSV", "Excel", ".xlsx" — all resolve. Both public readers funnel their
# caller-supplied format through this, so neither can accept a vocabulary the
# other rejects.
_FMT_TO_TABULAR_READER: dict[str, str] = {
    "csv": "csv",
    "tsv": "tsv",
    "excel": "excel",
    "xlsx": "excel",
    "xls": "excel",
    "parquet": "parquet",
    "pq": "parquet",
}


def normalise_format(value: str | None) -> str | None:
    """Normalise a format label for lookup, or `None` when there is none.

    Portals publish the same format under many spellings — `"CSV"`, `"csv"`,
    `".csv"`, `"GeoTIFF"`. Case and a leading dot carry no meaning, so strip
    them once here rather than at each comparison. A blank label is treated as
    absent, so `fmt=""` behaves like `fmt=None` instead of silently skipping an
    override the caller believed they had set.

    Args:
        value: A format label, or `None`.

    Returns:
        str | None: The lower-cased, de-dotted label, or `None` when `value` is
        `None` or blank.

    Examples:
        - Spelling variants collapse to one token:
            ```python
            >>> from pyramids._resource import normalise_format
            >>> [normalise_format(v) for v in ("CSV", "csv", ".Csv", "  csv ")]
            ['csv', 'csv', 'csv', 'csv']

            ```
        - A missing or blank label is reported as absent:
            ```python
            >>> from pyramids._resource import normalise_format
            >>> normalise_format(None) is None and normalise_format("  ") is None
            True

            ```
    """
    result: str | None = None
    if value is not None:
        cleaned = value.strip().lower().lstrip(".")
        result = cleaned or None
    return result


# Leading magic bytes -> format token. `SQLite format 3` is listed first so it
# is tested before any shorter prefix that could overlap. TIFF needs an exact
# 4-byte window (both byte orders) rather than a prefix test, so it is handled
# separately below.
_MAGIC_TO_FORMAT: tuple[tuple[bytes, str], ...] = (
    (b"SQLite format 3", "gpkg"),
    (b"PK\x03\x04", "zip"),
    (b"\x89HDF", "nc"),
    (b"PAR1", "parquet"),
)
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")
# Classic-netCDF format version byte following the `CDF` prefix: 1 (classic),
# 2 (64-bit offset), 5 (CDF-5). Checked so a text file starting "CDF" is not
# claimed as netCDF -- see `sniff_magic`.
_CDF_VERSIONS = (b"\x01", b"\x02", b"\x05")
# GRIB edition byte at offset 7: 1 or 2. Same reasoning as `_CDF_VERSIONS`.
_GRIB_EDITIONS = (b"\x01", b"\x02")

# Format token -> resource family, for callers that work in `ResourceKind`.
_FORMAT_TO_KIND: dict[str, ResourceKind] = {
    "shp": "vector",
    "gpkg": "vector",
    "geojson": "vector",
    "csv": "tabular",
    "tsv": "tabular",
    "parquet": "tabular",
    "nc": "raster",
    "tif": "raster",
    "grib": "raster",
}


def sniff_magic(path: str | Path) -> str | None:
    """Classify a file by its leading magic bytes.

    Reads the first 16 bytes only. Unlike :func:`sniff_kind` this performs I/O,
    but it is authoritative where the name lies — a portal resource declared
    ``CSV`` that is really a ``.shp.zip``, for instance.

    Args:
        path: Path to a local file.

    Returns:
        str | None: A format token (see :data:`_EXT_TO_FORMAT` for the
        vocabulary), or `None` when the file is unreadable or unrecognised.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(16)
    except OSError:
        head = b""

    result: str | None = None
    if head[:4] in _TIFF_MAGIC:
        result = "tif"
    elif head[:3] == b"CDF" and head[3:4] in _CDF_VERSIONS:
        # Classic netCDF is `CDF` + a one-byte version. Testing the prefix alone
        # would claim any text file whose first line begins "CDF..." — a CSV with
        # a column called CDF, say — as netCDF.
        result = "nc"
    elif head[:4] == b"GRIB" and head[7:8] in _GRIB_EDITIONS:
        # GRIB is `GRIB` + 3 reserved/length bytes + a one-byte edition number.
        # Same reasoning: the four-letter prefix on its own is a plausible CSV
        # header.
        result = "grib"
    else:
        for signature, fmt in _MAGIC_TO_FORMAT:
            if head.startswith(signature):
                result = fmt
                break
    return result


def sniff_format(path: str | Path) -> str:
    """Classify a file's format from its magic bytes, then its extension.

    Magic-byte detection takes precedence over the extension, so a mis-named
    resource is still classified correctly. No `python-magic` dependency.

    Args:
        path: Path to a local file.

    Returns:
        str: One of `"shp"`, `"gpkg"`, `"geojson"`, `"csv"`, `"tsv"`,
        `"parquet"`, `"nc"`, `"tif"`, `"grib"`, `"zip"`, or `"unknown"` when
        neither the magic bytes nor the extension identify the file. Excel is
        deliberately absent — see the note on :data:`_EXT_TO_FORMAT`.

    Examples:
        - A GeoTIFF is detected from its `II*\\0` / `MM\\0*` magic bytes:
            ```python
            >>> from pyramids.io import sniff_format
            >>> sniff_format("tests/data/geotiff/era5_land_monthly_averaged.tif")
            'tif'

            ```
        - A NetCDF file is detected (HDF5 or classic-CDF magic):
            ```python
            >>> sniff_format("tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc")
            'nc'

            ```
        - An unknown / missing file is reported as `"unknown"`:
            ```python
            >>> sniff_format("does-not-exist.bin")
            'unknown'

            ```

    See Also:
        sniff_magic: The magic-byte half, without the extension fallback.
        sniff_kind: The pure name-based family classifier (no I/O).
    """
    result = sniff_magic(path)
    if result is None:
        result = _EXT_TO_FORMAT.get(Path(path).suffix.lower(), "unknown")
    return result


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
    """Resolve the family from name + ``fmt``, then bytes, then archive members.

    Order is cheapest-first: the pure name/``fmt`` lookup, then the magic-byte
    sniff (one 16-byte read, which also identifies an extension-less download),
    then peeking inside a container. Magic bytes are consulted only when the name
    is inconclusive, so a correctly-named resource keeps its existing answer.
    """
    try:
        kind = sniff_kind(path, fmt=fmt)
    except ValueError:
        sniffed = _FORMAT_TO_KIND.get(sniff_format(path))
        if sniffed is None:
            sniffed = _sniff_from_archive(path)
        if sniffed is None:
            raise
        kind = sniffed
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


def read_tabular(path: str | Path, fmt: str | None = None) -> pd.DataFrame:
    """Read a tabular resource into a :class:`pandas.DataFrame`.

    ``pandas`` infers ``.gz`` / ``.zip`` / ``.tar`` compression from the suffix,
    so the path is passed through verbatim. ``.xlsx`` needs ``openpyxl`` and
    ``.parquet`` needs ``pyarrow`` — when missing, the underlying
    :class:`ImportError` is re-raised with install guidance.

    Args:
        path: Path to the tabular resource.
        fmt: Optional reader reader (`"csv"`, `"tsv"`, `"excel"`, `"parquet"`)
            that **overrides** the suffix. Required for a resource whose name
            carries no usable extension — a portal download named by id, say —
            where the caller knows the declared format. When omitted the suffix
            decides, falling back to the magic bytes.

    Returns:
        pandas.DataFrame: The parsed table.

    Raises:
        ValueError: Neither `fmt`, the suffix, nor the magic bytes identify a
            supported tabular format.
        ImportError: The chosen reader needs an optional engine that is not
            installed (`openpyxl`/`xlrd` for Excel, `pyarrow` for Parquet).
    """
    # Accept a plain string too: this is public-named and reached from both
    # readers, so it must not depend on the caller having wrapped the path.
    path = Path(path)
    source = str(path)
    ext = Path(_strip_compression(path.name)).suffix.lower()
    label = normalise_format(fmt)
    if label is not None:
        # An explicit label wins over the name — but only if it names a reader.
        # Falling through to the suffix here would silently ignore the override
        # and then blame the file's extension for the failure.
        reader = _FMT_TO_TABULAR_READER.get(label)
        if reader is None:
            raise ValueError(
                f"unsupported tabular format {fmt!r} for {source!r}; expected one "
                f"of {sorted(set(_FMT_TO_TABULAR_READER))}."
            )
    else:
        # The name wins over the bytes. The magic-byte tail is what lets an
        # extension-less Parquet download resolve instead of dead-ending on an
        # empty suffix; it is mapped through the same table so a non-tabular
        # reader (`"zip"`, `"tif"`, `"unknown"`) cannot leak through as a reader.
        reader = _EXT_TO_TABULAR_READER.get(ext) or _FMT_TO_TABULAR_READER.get(
            sniff_format(path)
        )
    result: pd.DataFrame
    if reader == "csv":
        result = pd.read_csv(source)
    elif reader == "tsv":
        result = pd.read_csv(source, sep="\t")
    elif reader == "excel":
        try:
            result = pd.read_excel(source)
        except ImportError as exc:  # openpyxl (.xlsx) / xlrd (.xls) not installed
            raise ImportError(
                f"reading {ext or 'excel'} files needs an Excel engine (openpyxl "
                "for .xlsx, xlrd for legacy .xls); install it into the environment "
                "to read this resource."
            ) from exc
    elif reader == "parquet":
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
            f"{sorted(_TABULAR_SUFFIXES)}, or pass an explicit fmt= when the name "
            "carries no usable extension."
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
        # Forward the declared label when it names a tabular reader, so an
        # extension-less download resolves here exactly as it does through
        # `load_resource`. A non-tabular label (a raster/vector `fmt` paired with
        # an explicit `kind="tabular"`) is dropped rather than rejected, leaving
        # the suffix and magic bytes to decide as before.
        result = read_tabular(
            path, fmt=_FMT_TO_TABULAR_READER.get(normalise_format(fmt) or "")
        )
    else:
        raise ValueError(
            f"unsupported resource kind {resolved_kind!r}; expected one of "
            "'raster', 'vector', 'tabular'."
        )
    return result
