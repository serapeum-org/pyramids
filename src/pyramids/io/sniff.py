"""Detect a downloaded resource's real format and open it with the right reader.

CKAN portals (HDX is one) serve heterogeneous resources whose declared ``format``
field lies often enough that you must sniff the actual bytes: a resource declared
`CSV` may really be a `.shp.zip`. :func:`sniff_format` classifies a file by its
magic bytes first, then its extension; :func:`load_resource` dispatches to the
matching pyramids/pandas reader and returns the most natural object:

- vector (`.shp` / `.gpkg` / GeoJSON) -> :class:`~pyramids.feature.FeatureCollection`
- tabular CSV -> :class:`pandas.DataFrame`
- raster GeoTIFF / GRIB -> :class:`~pyramids.dataset.Dataset`
- NetCDF -> :class:`~pyramids.netcdf.NetCDF`
- Parquet -> :class:`~pyramids.feature.FeatureCollection` when it carries a GeoParquet
  `geo` key, otherwise a :class:`pandas.DataFrame`
- ZIP -> the single contained data file (re-dispatched), or the extraction dir
- anything unrecognised -> raw :class:`bytes`

Detection uses magic bytes + extension only — no `python-magic` dependency. The
CKAN/HDX API client itself stays in the consumer; this module is the generic
format-detection + dispatch primitive.

Detection itself lives in :mod:`pyramids._resource`, which both public readers
share, so add a new *format* to `_resource._EXT_TO_FORMAT` / `_MAGIC_TO_FORMAT`.
The tables that remain here are *dispatch policy*, not detection: which tokens
this module can honour, and which archive members count as the one to load.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from pyramids._resource import (
    normalise_format,
    read_tabular,
    sniff_format,
    sniff_magic,
)
from pyramids.base._artifacts import artifact_dir
from pyramids.base._utils import import_pyarrow
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF

__all__ = ["load_resource", "sniff_format", "sniff_magic"]

_VECTOR_FORMATS = frozenset({"shp", "gpkg", "geojson"})
_TABULAR_FORMATS = frozenset({"csv", "tsv"})
# Every token this dispatcher understands, so an `expected_format=` it cannot
# honour is rejected instead of silently returning raw bytes. Derived from the
# branches below plus the shared detection vocabulary.
_KNOWN_FORMATS = (
    _VECTOR_FORMATS
    | _TABULAR_FORMATS
    | frozenset({"parquet", "nc", "tif", "grib", "zip", "unknown"})
)
# Data extensions a ZIP may wrap and that `_load_zip` re-dispatches to a reader,
# in two tiers. Both are listed rather than derived from `_EXT_TO_FORMAT`,
# because this is a dispatch policy rather than a detection vocabulary.
#
# The tiers exist because "exactly one primary member" is ambiguous when an
# archive ships data plus a tabular sidecar. A zip of `grid.tif` + `meta.tsv`
# must resolve to the raster, but a zip holding only `table.tsv` should still
# load that table. So geospatial members are considered first, and the tabular
# tier is consulted only when the archive contains no geospatial member at all.
_PRIMARY_EXTS = frozenset(
    {
        ".shp",
        ".gpkg",
        ".geojson",
        ".json",
        ".csv",
        ".parquet",
        ".pq",
        ".tif",
        ".tiff",
        ".nc",
        ".nc4",
        ".cdf",
        ".grib",
        ".grib2",
        ".grb",
        ".grb2",
    }
)
# Tabular formats that count only when nothing in `_PRIMARY_EXTS` is present.
# `.csv` stays in the primary tier for backwards compatibility — it was there
# before the tiering and archives relying on it must keep resolving.
_SECONDARY_EXTS = frozenset({".tsv", ".xlsx", ".xls"})
_PARQUET_EXTRA_HINT = (
    "Reading Parquet requires the optional 'pyarrow' dependency. Install with one of:\n"
    "  - PyPI:        pip install 'pyramids-gis[parquet]'\n"
    "  - conda-forge: conda install -c conda-forge pyramids-parquet"
)


def _load_parquet(path: Path) -> FeatureCollection | pd.DataFrame:
    """Open a Parquet file as a FeatureCollection (GeoParquet) or DataFrame.

    Args:
        path: Path to a `.parquet` file.

    Returns:
        A :class:`~pyramids.feature.FeatureCollection` when the file carries a
        GeoParquet `geo` metadata key, otherwise a :class:`pandas.DataFrame`.

    Raises:
        OptionalPackageDoesNotExist: `pyarrow` (the `[parquet]` extra) is
            not installed.
    """
    import_pyarrow(_PARQUET_EXTRA_HINT)
    import pyarrow.parquet as pq  # [parquet] extra, guarded above

    metadata = pq.read_metadata(str(path)).metadata or {}
    if b"geo" in metadata:
        return FeatureCollection.read_parquet(str(path))
    return pd.read_parquet(path)


def _load_zip(path: Path, extract_to: Path | None) -> Any:
    """Extract a ZIP and load its single primary data member, or return the dir.

    Args:
        path: Path to a `.zip` file.
        extract_to: Directory to extract into; when `None` a directory under the
            process-scoped artefact root is used (reclaimed at interpreter exit).

    Returns:
        The loaded resource (re-dispatched through :func:`load_resource`) when a
        single primary data file is found, otherwise the :class:`~pathlib.Path`
        of the extraction directory.
    """
    # `artifact_dir()` rather than a bare `tempfile.mkdtemp()`: the extracted
    # files must outlive this call (GDAL/geopandas read them lazily, so a
    # `TemporaryDirectory` would delete them mid-read), but an untracked mkdtemp
    # is never reclaimed — every load_resource(<zip>) leaked one. The shared
    # artefact root is swept by an atexit hook.
    dest = Path(extract_to) if extract_to is not None else Path(artifact_dir())
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if not n.endswith("/")]
        archive.extractall(dest)

    primaries = [m for m in members if Path(m).suffix.lower() in _PRIMARY_EXTS]
    if not primaries:
        # No geospatial/CSV member: fall back to the tabular tier so an archive
        # that holds only a spreadsheet still resolves to a frame rather than to
        # the extraction directory.
        primaries = [m for m in members if Path(m).suffix.lower() in _SECONDARY_EXTS]
    shapefiles = [m for m in primaries if m.lower().endswith(".shp")]

    target: str | None = None
    if shapefiles:
        target = shapefiles[0]
    elif len(primaries) == 1:
        target = primaries[0]

    return load_resource(dest / target) if target is not None else dest


def load_resource(
    path: str | Path,
    *,
    expected_format: str | None = None,
    extract_to: str | Path | None = None,
) -> Any:
    """Read a downloaded resource by its detected (or declared) format.

    Detection priority: an explicit `expected_format` override, otherwise
    :func:`sniff_format`. ZIP files are inspected for a single primary data
    member (a `.shp` set, or one `.gpkg` / `.tif` / `.nc` / `.csv` /
    `.parquet` / GeoJSON) and the dispatch re-runs on it; ZIPs without a clear
    primary are extracted and the directory path is returned.

    Args:
        path: Path to the downloaded resource.
        expected_format: Optional format override (one of the
            :func:`sniff_format` tokens); skips sniffing when given. Use it when
            the name carries no usable extension — a portal download named by id
            — since the override is forwarded to the reader rather than
            re-derived from the suffix.
        extract_to: Directory to extract a ZIP into. When `None` a directory
            under the process-scoped artefact root is used, which is swept at
            interpreter exit rather than left behind.

    Returns:
        The most natural object for the format: a
        :class:`~pyramids.feature.FeatureCollection` for vector data, a
        :class:`pandas.DataFrame` for CSV / non-geo Parquet, a
        :class:`~pyramids.dataset.Dataset` for GeoTIFF / GRIB, a
        :class:`~pyramids.netcdf.NetCDF` for NetCDF, the contained resource (or
        directory :class:`~pathlib.Path`) for a ZIP, or raw :class:`bytes` for
        an unrecognised format.

    Raises:
        OptionalPackageDoesNotExist: A Parquet resource is read without the
            `[parquet]` extra installed.
        ValueError: A tabular resource whose format cannot be determined from
            `expected_format`, its suffix, or its magic bytes.

    Examples:
        - Load a GeoTIFF resource as a Dataset:
            ```python
            >>> from pyramids.io import load_resource
            >>> ds = load_resource("tests/data/geotiff/era5_land_monthly_averaged.tif")
            >>> ds.band_count
            9

            ```
        - Force a format with `expected_format` (skips sniffing):
            ```python
            >>> nc = load_resource(
            ...     "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc",
            ...     expected_format="nc",
            ... )
            >>> "precipitation" in " ".join(nc.variable_names).lower() or len(nc.variable_names) > 0
            True

            ```
    """
    p = Path(path)
    # Normalise the caller's label: portals publish "CSV" / "GeoJSON" / ".csv",
    # and an unnormalised value used to miss every branch below and fall through
    # to the raw-bytes default — a silent wrong return type on the documented
    # extension-less path. An unrecognised label is rejected outright rather than
    # degraded to bytes.
    declared = normalise_format(expected_format)
    if declared is not None and declared not in _KNOWN_FORMATS:
        raise ValueError(
            f"unknown expected_format {expected_format!r}; expected one of "
            f"{sorted(_KNOWN_FORMATS)} (case-insensitive), or omit it to sniff."
        )
    fmt = declared or sniff_format(p)

    if fmt in _VECTOR_FORMATS:
        result: Any = FeatureCollection.read_file(str(p))
    elif fmt in _TABULAR_FORMATS:
        # Shared tabular reader, so `.tsv` works here too rather than falling
        # through to raw bytes as it did while this module carried its own
        # dispatch table. `fmt` is forwarded so an explicit `expected_format=`
        # still wins on a resource whose name has no usable extension.
        result = read_tabular(p, fmt=fmt)
    elif fmt == "parquet":
        result = _load_parquet(p)
    elif fmt == "nc":
        result = NetCDF.read_file(str(p))
    elif fmt in ("tif", "grib"):
        result = Dataset.read_file(str(p))
    elif fmt == "zip":
        result = _load_zip(p, Path(extract_to) if extract_to is not None else None)
    else:
        result = p.read_bytes()
    return result
