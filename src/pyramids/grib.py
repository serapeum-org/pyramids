"""Read GRIB1/GRIB2 files (local or cloud) as pyramids datasets via GDAL.

GRIB is decoded through GDAL's native `GRIB` driver — no extra Python decoding
dependency. The driver ships in the conda-forge `libgdal-grib` plugin (a core
pyramids dependency, alongside `libgdal-netcdf` / `libgdal-hdf4`);
:func:`open_grib` raises a clear error if a GDAL build lacks it.

Each GRIB *message* becomes a band of the returned
:class:`~pyramids.dataset.Dataset`, carrying the per-message GRIB metadata
(element, level, reference / valid time, forecast horizon). Use
:func:`grib_band_metadata` to introspect those bands and pick the band(s) you
want. Cloud URIs (`s3://` / `gs://` / `https://` / `/vsi*`) are resolved
through the same path as :meth:`pyramids.dataset.Dataset.read_file`.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from osgeo import gdal

from pyramids.base._errors import DriverNotExistError
from pyramids.dataset.dataset import Dataset

_GRIB_DRIVER = "GRIB"

_GRIB_FIELDS: dict[str, str] = {
    "element": "GRIB_ELEMENT",
    "short_name": "GRIB_SHORT_NAME",
    "comment": "GRIB_COMMENT",
    "unit": "GRIB_UNIT",
    "discipline": "GRIB_DISCIPLINE",
}


def _require_grib_driver() -> None:
    """Raise :class:`DriverNotExistError` when the GDAL GRIB driver is absent.

    Raises:
        DriverNotExistError: The active GDAL build has no `GRIB` driver.
    """
    if gdal.GetDriverByName(_GRIB_DRIVER) is None:
        raise DriverNotExistError(
            "GDAL build does not include the GRIB driver. Install the "
            "conda-forge `libgdal-grib` plugin (e.g. `pixi install -e dev`)."
        )


def _parse_grib_seconds(value: str | None) -> datetime | None:
    """Parse a GRIB `"<unix_seconds> sec UTC"` timestamp to an aware datetime.

    Args:
        value: A GRIB metadata value such as `"1700000000 sec UTC"`, or
            `None`.

    Returns:
        A timezone-aware UTC :class:`~datetime.datetime`, or `None` when the
        value is missing or not parseable.
    """
    result: datetime | None = None
    if value:
        token = value.split()[0]
        try:
            result = datetime.fromtimestamp(int(token), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            result = None
    return result


def _parse_leading_int(value: str | None) -> int | None:
    """Parse the leading integer of a GRIB metadata value (e.g. `"10800 s"`).

    Args:
        value: A GRIB metadata value whose first token is an integer, or
            `None`.

    Returns:
        The leading integer, or `None` when missing / not parseable.
    """
    result: int | None = None
    if value:
        token = value.split()[0]
        try:
            result = int(token)
        except ValueError:
            result = None
    return result


def open_grib(path: str | Path, *, vsi: str | None = None) -> Dataset:
    """Open a GRIB1/GRIB2 file as a pyramids :class:`~pyramids.dataset.Dataset`.

    Decodes via GDAL's native `GRIB` driver; every GRIB message is a band.
    Local paths, `/vsi*` paths, and `http(s)://` / `s3://` / `gs://` /
    `az://` URIs are all accepted (resolved by
    :meth:`~pyramids.dataset.Dataset.read_file`).

    Args:
        path: Path or URI to a GRIB1/GRIB2 file.
        vsi: Optional explicit archive kind forwarded to
            :meth:`~pyramids.dataset.Dataset.read_file` (e.g. for a GRIB inside
            a `.zip`); `None` opens `path` directly.

    Returns:
        A :class:`~pyramids.dataset.Dataset` with one band per GRIB message.
        Use :func:`grib_band_metadata` to identify the bands.

    Raises:
        DriverNotExistError: The GDAL build lacks the GRIB driver (install the
            `libgdal-grib` plugin).

    Examples:
        - Open a local GRIB2 file and inspect its bands (requires libgdal-grib):
            ```python
            >>> from pyramids.grib import open_grib, grib_band_metadata  # doctest: +SKIP
            >>> ds = open_grib("gfs.t00z.pgrb2.0p25.f000.grib2")  # doctest: +SKIP
            >>> ds.band_count  # doctest: +SKIP
            5
            >>> grib_band_metadata(ds)[0]["element"]  # doctest: +SKIP
            'TMP'

            ```
        - Open a GRIB file from cloud storage:
            ```python
            >>> ds = open_grib(  # doctest: +SKIP
            ...     "s3://noaa-gfs-bdp-pds/gfs.20260518/00/atmos/"
            ...     "gfs.t00z.pgrb2.0p25.f000"
            ... )

            ```
    """
    _require_grib_driver()
    return Dataset.read_file(path, vsi=vsi)


def grib_band_metadata(dataset: Dataset) -> list[dict[str, Any]]:
    """Return per-band GRIB metadata for a dataset opened from a GRIB file.

    One dict per band exposes the common GRIB message fields, with reference /
    valid times decoded to timezone-aware UTC datetimes and the forecast
    horizon as an integer number of seconds. Missing fields are `None`.

    Args:
        dataset: A :class:`~pyramids.dataset.Dataset` opened from a GRIB file
            (e.g. via :func:`open_grib`).

    Returns:
        A list (one entry per band, in band order) of dicts with keys:
        `band` (1-based index), `element`, `short_name`, `comment`,
        `unit`, `discipline`, `ref_time` / `valid_time`
        (:class:`~datetime.datetime` or `None`), and `forecast_seconds`
        (:class:`int` or `None`).

    Examples:
        - Find the band index carrying 2-metre temperature (requires libgdal-grib):
            ```python
            >>> from pyramids.grib import open_grib, grib_band_metadata  # doctest: +SKIP
            >>> ds = open_grib("gfs.t00z.pgrb2.0p25.f000.grib2")  # doctest: +SKIP
            >>> meta = grib_band_metadata(ds)  # doctest: +SKIP
            >>> [m["band"] for m in meta if m["element"] == "TMP"]  # doctest: +SKIP
            [1]

            ```
        - Read the valid time of the first band:
            ```python
            >>> meta = grib_band_metadata(open_grib("f000.grib2"))  # doctest: +SKIP
            >>> meta[0]["valid_time"].year  # doctest: +SKIP
            2026

            ```
    """
    raster = dataset.raster
    metadata: list[dict[str, Any]] = []
    for index in range(1, raster.RasterCount + 1):
        band_md = raster.GetRasterBand(index).GetMetadata()
        entry: dict[str, Any] = {"band": index}
        for key, grib_key in _GRIB_FIELDS.items():
            entry[key] = band_md.get(grib_key)
        entry["ref_time"] = _parse_grib_seconds(band_md.get("GRIB_REF_TIME"))
        entry["valid_time"] = _parse_grib_seconds(band_md.get("GRIB_VALID_TIME"))
        entry["forecast_seconds"] = _parse_leading_int(
            band_md.get("GRIB_FORECAST_SECONDS")
        )
        metadata.append(entry)
    return metadata


def _select_grib_band(
    metadata: list[dict[str, Any]], variable: str | int | None
) -> int:
    """Resolve the 0-based band index for `variable`.

    Args:
        metadata: Per-band metadata from :func:`grib_band_metadata`.
        variable: How to pick the message. An `int` is a 1-based band number
            (the unambiguous escape hatch for files that repeat an element
            across levels/horizons). A non-empty `str` matches the GRIB element
            (case-insensitive, e.g. `"TMP"`); when several messages share the
            element the first is used and a warning is emitted. `None` selects
            the sole band of a single-message file.

    Returns:
        The 0-based band index to keep.

    Raises:
        ValueError: `variable` is not a `str`, `int`, or `None` (a `bool` is
            rejected too); `variable` is `None` but the file holds more than one
            message; an `int` band number is out of range; `variable` is an
            empty string; or no message carries the requested element.
    """
    band_count = len(metadata)
    # `variable`, `band` and the user's band number are all 1-based; the returned
    # index (and `to_cog(indexes=)`) is 0-based, hence the `- 1` conversions below.
    if isinstance(variable, bool) or not isinstance(variable, (str, int, type(None))):
        raise ValueError(
            f"variable must be a band number, element name, or None; got {variable!r}."
        )
    if isinstance(variable, int):
        if not 1 <= variable <= band_count:
            raise ValueError(
                f"band number {variable} out of range 1..{band_count}."
            )
        selected = variable - 1
    elif variable is None:
        if band_count > 1:
            elements = [m.get("element") for m in metadata]
            raise ValueError(
                f"GRIB file has {band_count} messages; "
                f"pass variable= to pick one of {elements}."
            )
        selected = 0
    else:
        wanted = variable.strip().upper()
        if not wanted:
            raise ValueError(
                "variable must be a non-empty element name or band number."
            )
        matches = [
            m for m in metadata if (m.get("element") or "").strip().upper() == wanted
        ]
        if not matches:
            elements = sorted({m.get("element") for m in metadata if m.get("element")})
            raise ValueError(
                f"No GRIB message with element {variable!r}; "
                f"available elements: {elements}."
            )
        if len(matches) > 1:
            warnings.warn(
                f"{len(matches)} GRIB messages match element {variable!r}; "
                f"using the first (band {matches[0]['band']}).",
                stacklevel=3,
            )
        selected = matches[0]["band"] - 1
    return selected


def grib_to_cog(
    grib_path: str | Path,
    *,
    output: str | Path,
    variable: str | int | None = None,
    target_crs: int | str | None = None,
    cog_profile: str = "deflate",
    vsi: str | None = None,
) -> Path:
    """Convert a single GRIB message to a Cloud-Optimized GeoTIFF in one call.

    Chains :func:`open_grib` → select the `variable` band → write with
    :meth:`~pyramids.dataset.Dataset.to_cog`. Everything is in-repo (GDAL's GRIB
    driver + pyramids' COG writer) — no `rasterio` / `rio-cogeo` / `cfgrib`. The
    written COG carries the GRIB band's CRS and its `GRIB_*` band metadata, and
    passes the COG validator (:func:`pyramids.dataset.cog.cog_info` `.is_cog`).

    Args:
        grib_path: Path or URI to a GRIB1/GRIB2 file (local, `/vsi*`, or a cloud
            `s3://` / `gs://` / `https://` URI — resolved like
            :func:`open_grib`).
        output: Destination COG path. Its parent directory must already exist.
        variable: Which GRIB message to convert. A `str` matches the GRIB element
            (case-insensitive, e.g. `"TMP"`); when several messages share the
            element the first is used and a warning is emitted. Pass an `int`
            1-based band number to select a message unambiguously (e.g. one
            element repeated across pressure levels). `None` is allowed only when
            the file holds a single message.
        target_crs: Optional CRS to reproject to before the COG is written — an
            EPSG `int` or a WKT `str` (forwarded to `to_cog(target_srs=...)`).
            `None` keeps the GRIB's native CRS.
        cog_profile: Named COG compression preset forwarded to
            :meth:`~pyramids.dataset.Dataset.to_cog` (`profile=`), e.g.
            `"deflate"`, `"zstd"`, `"lzw"`.
        vsi: Optional explicit archive kind forwarded to :func:`open_grib` (e.g.
            for a GRIB inside a `.zip`).

    Returns:
        The `output` path as a :class:`~pathlib.Path`.

    Raises:
        DriverNotExistError: The GDAL build lacks the GRIB driver.
        FileNotFoundError: `grib_path` does not exist (raised by :func:`open_grib`).
        ValueError: `variable` cannot be resolved (`None` on a multi-message file,
            an out-of-range band number, an empty string, or an unknown element),
            or `cog_profile` is not a recognised COG profile (raised by `to_cog`).

    Examples:
        - Convert the 2-metre temperature message to a COG (requires libgdal-grib):
            ```python
            >>> from pyramids.grib import grib_to_cog  # doctest: +SKIP
            >>> from pyramids.dataset.cog import cog_info  # doctest: +SKIP
            >>> out = grib_to_cog(  # doctest: +SKIP
            ...     "tmp2m.grib2", variable="TMP", output="tmp2m_cog.tif"
            ... )
            >>> cog_info(out).is_cog  # doctest: +SKIP
            True

            ```
    """
    with open_grib(grib_path, vsi=vsi) as dataset:
        band_index = _select_grib_band(grib_band_metadata(dataset), variable)
        result = dataset.to_cog(
            output,
            indexes=[band_index],
            profile=cog_profile,
            target_srs=target_crs,
        )
    return result
