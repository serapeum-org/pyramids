"""Read GRIB1/GRIB2 files (local or cloud) as pyramids datasets via GDAL.

GRIB is decoded through GDAL's native ``GRIB`` driver — no ``cfgrib`` / ``eccodes``
/ ``xarray`` dependency. The driver ships in the conda-forge ``libgdal-grib``
plugin (a core pyramids dependency, alongside ``libgdal-netcdf`` /
``libgdal-hdf4``); :func:`open_grib` raises a clear error if a GDAL build lacks
it.

Each GRIB *message* becomes a band of the returned
:class:`~pyramids.dataset.Dataset`, carrying the per-message GRIB metadata
(element, level, reference / valid time, forecast horizon). Use
:func:`grib_band_metadata` to introspect those bands — the GDAL-native
equivalent of inspecting cfgrib's ``filter_by_keys`` — and pick the band(s) you
want. Cloud URIs (``s3://`` / ``gs://`` / ``https://`` / ``/vsi*``) are resolved
through the same path as :meth:`pyramids.dataset.Dataset.read_file`.
"""

from __future__ import annotations

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
        DriverNotExistError: The active GDAL build has no ``GRIB`` driver.
    """
    if gdal.GetDriverByName(_GRIB_DRIVER) is None:
        raise DriverNotExistError(
            "GDAL build does not include the GRIB driver. Install the "
            "conda-forge `libgdal-grib` plugin (e.g. `pixi install -e dev`)."
        )


def _parse_grib_seconds(value: str | None) -> datetime | None:
    """Parse a GRIB ``"<unix_seconds> sec UTC"`` timestamp to an aware datetime.

    Args:
        value: A GRIB metadata value such as ``"1700000000 sec UTC"``, or
            ``None``.

    Returns:
        A timezone-aware UTC :class:`~datetime.datetime`, or ``None`` when the
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
    """Parse the leading integer of a GRIB metadata value (e.g. ``"10800 s"``).

    Args:
        value: A GRIB metadata value whose first token is an integer, or
            ``None``.

    Returns:
        The leading integer, or ``None`` when missing / not parseable.
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

    Decodes via GDAL's native ``GRIB`` driver; every GRIB message is a band.
    Local paths, ``/vsi*`` paths, and ``http(s)://`` / ``s3://`` / ``gs://`` /
    ``az://`` URIs are all accepted (resolved by
    :meth:`~pyramids.dataset.Dataset.read_file`). No ``cfgrib`` / ``xarray``.

    Args:
        path: Path or URI to a GRIB1/GRIB2 file.
        vsi: Optional explicit archive kind forwarded to
            :meth:`~pyramids.dataset.Dataset.read_file` (e.g. for a GRIB inside
            a ``.zip``); ``None`` opens ``path`` directly.

    Returns:
        A :class:`~pyramids.dataset.Dataset` with one band per GRIB message.
        Use :func:`grib_band_metadata` to identify the bands.

    Raises:
        DriverNotExistError: The GDAL build lacks the GRIB driver (install the
            ``libgdal-grib`` plugin).

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
    horizon as an integer number of seconds. Missing fields are ``None``.

    Args:
        dataset: A :class:`~pyramids.dataset.Dataset` opened from a GRIB file
            (e.g. via :func:`open_grib`).

    Returns:
        A list (one entry per band, in band order) of dicts with keys:
        ``band`` (1-based index), ``element``, ``short_name``, ``comment``,
        ``unit``, ``discipline``, ``ref_time`` / ``valid_time``
        (:class:`~datetime.datetime` or ``None``), and ``forecast_seconds``
        (:class:`int` or ``None``).

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
