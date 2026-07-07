"""Detect a downloaded resource's real format and open it with the right reader.

CKAN portals (HDX is one) serve heterogeneous resources whose declared ``format``
field lies often enough that you must sniff the actual bytes: a resource declared
`CSV` may really be a `.shp.zip`. :func:`sniff_format` classifies a file by its
magic bytes first, then its extension; :func:`load_resource` dispatches to the
matching pyramids/pandas reader and returns the most natural object:

- vector (`.shp` / `.gpkg` / GeoJSON) -> :class:`~pyramids.feature.FeatureCollection`
- tabular CSV -> :class:`pandas.DataFrame`
- raster GeoTIFF / GRIB -> :class:`~pyramids.dataset.Dataset`
- NetCDF -> :class:`~pyramids.netcdf.NetCDF` (no xarray)
- Parquet -> :class:`~pyramids.feature.FeatureCollection` when it carries a GeoParquet
  `geo` key, otherwise a :class:`pandas.DataFrame`
- ZIP -> the single contained data file (re-dispatched), or the extraction dir
- anything unrecognised -> raw :class:`bytes`

Detection uses magic bytes + extension only — no `python-magic` dependency. The
CKAN/HDX API client itself stays in the consumer; this module is the generic
format-detection + dispatch primitive.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from pyramids.base._utils import import_pyarrow
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF

_EXT_FORMAT: dict[str, str] = {
    ".shp": "shp",
    ".gpkg": "gpkg",
    ".geojson": "geojson",
    ".json": "geojson",
    ".csv": "csv",
    ".parquet": "parquet",
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

_VECTOR_FORMATS = frozenset({"shp", "gpkg", "geojson"})
# Data extensions a ZIP may wrap and that `_load_zip` re-dispatches to a reader.
# Kept in sync with the data formats in `_EXT_FORMAT` (everything except the
# ``.zip`` archive container itself).
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
_PARQUET_EXTRA_HINT = (
    "Reading Parquet requires the optional 'pyarrow' dependency. Install with one of:\n"
    "  - PyPI:        pip install 'pyramids-gis[parquet]'\n"
    "  - conda-forge: conda install -c conda-forge pyramids-parquet"
)


def sniff_format(path: str | Path) -> str:
    """Classify a file's format from its magic bytes, then its extension.

    Magic-byte detection (ZIP, TIFF, HDF5/NetCDF, Parquet, GeoPackage/SQLite)
    takes precedence over the extension, so a mis-named resource is still
    classified correctly. No `python-magic` dependency.

    Args:
        path: Path to a local file.

    Returns:
        A normalised format string: one of `"shp"`, `"gpkg"`,
        `"geojson"`, `"csv"`, `"parquet"`, `"nc"`, `"tif"`,
        `"grib"`, `"zip"`, or `"unknown"`.

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
    """
    p = Path(path)
    head = b""
    try:
        with open(p, "rb") as handle:
            head = handle.read(16)
    except OSError:
        head = b""

    result = "unknown"
    if head.startswith(b"PK\x03\x04"):
        result = "zip"
    elif head[:4] in (b"II*\x00", b"MM\x00*"):
        result = "tif"
    elif head.startswith(b"\x89HDF"):
        result = "nc"
    elif head[:3] == b"CDF":
        result = "nc"
    elif head.startswith(b"GRIB"):
        result = "grib"
    elif head.startswith(b"PAR1"):
        result = "parquet"
    elif head.startswith(b"SQLite format 3"):
        result = "gpkg"
    elif p.suffix.lower() in _EXT_FORMAT:
        result = _EXT_FORMAT[p.suffix.lower()]
    return result


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
        extract_to: Directory to extract into; a temp dir is used when `None`.

    Returns:
        The loaded resource (re-dispatched through :func:`load_resource`) when a
        single primary data file is found, otherwise the :class:`~pathlib.Path`
        of the extraction directory.
    """
    dest = Path(extract_to) if extract_to is not None else Path(tempfile.mkdtemp())
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if not n.endswith("/")]
        archive.extractall(dest)

    primaries = [m for m in members if Path(m).suffix.lower() in _PRIMARY_EXTS]
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
            :func:`sniff_format` strings); skips sniffing when given.
        extract_to: Directory to extract a ZIP into; a temp dir is used when
            `None`.

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
    fmt = expected_format or sniff_format(p)

    if fmt in _VECTOR_FORMATS:
        result: Any = FeatureCollection.read_file(str(p))
    elif fmt == "csv":
        result = pd.read_csv(p)
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
