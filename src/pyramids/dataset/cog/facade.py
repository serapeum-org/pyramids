"""High-level Cloud Optimized GeoTIFF writer facade.

:func:`write_cog` is the single-call entry point that accepts a pyramids
:class:`~pyramids.dataset.Dataset`, a raw :class:`osgeo.gdal.Dataset`, a
NumPy array (with ``crs`` + ``transform``), an :class:`xarray.DataArray`,
or a path to an existing raster; applies pyramids' house-style COG
defaults (DEFLATE, dtype-aware predictor, 512px tiles, average overviews,
``BIGTIFF=IF_SAFER``); and — by default — round-trips the written file
through :func:`pyramids.dataset.cog.validate.validate` before returning.

The function reuses the existing COG machinery rather than re-deriving it:
normalisation builds a :class:`~pyramids.dataset.Dataset` and delegates the
actual write to :meth:`pyramids.dataset.engines.cog.COG.to_cog`, so the
categorical-resampling guardrail and option validation already wired into
that path apply here too.

This module deliberately imports :class:`~pyramids.dataset.Dataset` inside
the function body (not at module top level) to break the
``pyramids.dataset.dataset`` -> ``pyramids.dataset.engines.cog`` ->
``pyramids.dataset.cog`` import cycle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from osgeo import gdal
from pyproj import CRS

from pyramids.base._errors import ReadOnlyError
from pyramids.base._utils import resolve_cog_predictor
from pyramids.dataset.cog.options import CreationOptions
from pyramids.dataset.cog.validate import ValidationReport
from pyramids.dataset.cog.validate import validate as _validate_file

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyramids.dataset.dataset import Dataset

logger = logging.getLogger(__name__)


PYRAMIDS_COG_DEFAULTS: dict[str, Any] = {
    "COMPRESS": "DEFLATE",
    "BLOCKSIZE": 512,
    "BIGTIFF": "IF_SAFER",
    "NUM_THREADS": "ALL_CPUS",
    "STATISTICS": "YES",
}
"""Pyramids house-style COG creation options (the *static* subset).

These mirror the kwarg defaults of
:meth:`pyramids.dataset.engines.cog.COG.to_cog`, which is the single owner of
COG write policy (ARC-1). ``PREDICTOR`` and ``OVERVIEW_RESAMPLING`` are
intentionally absent: both are resolved per-dtype inside ``to_cog`` (integer →
``PREDICTOR=2`` / ``mode`` overviews; float → ``PREDICTOR=3`` / ``average``
overviews) unless the caller overrides them. Kept here for back-compat and as
documentation; :func:`write_cog` no longer applies it directly — it delegates
to ``to_cog``.
"""


# Back-compat alias: the dtype→predictor rule now lives in
# ``pyramids.base._utils`` so the engine and facade share one definition (ARC-2).
_resolve_predictor = resolve_cog_predictor


def _coerce_epsg(crs: Any) -> int:
    """Coerce a user CRS into an integer EPSG code.

    Args:
        crs: An EPSG integer, an ``"EPSG:XXXX"`` string, a WKT/PROJ
            string, or any value :meth:`pyproj.CRS.from_user_input`
            understands.

    Returns:
        The integer EPSG code.

    Raises:
        ValueError: When ``crs`` has no EPSG representation.

    Examples:
        - An integer is returned unchanged:
            ```python
            >>> _coerce_epsg(4326)
            4326

            ```
        - Authority strings are resolved to their code:
            ```python
            >>> _coerce_epsg("EPSG:3857")
            3857

            ```
    """
    if isinstance(crs, int):
        return crs
    epsg = CRS.from_user_input(crs).to_epsg()
    if epsg is None:
        raise ValueError(
            f"Could not resolve an EPSG code from crs={crs!r}; pass an "
            "integer EPSG code explicitly."
        )
    return epsg


def _array_to_dataset(
    arr: np.ndarray,
    crs: Any | None,
    transform: tuple[float, ...] | None,
    nodata: float | int | None,
) -> Dataset:
    """Build an in-memory :class:`Dataset` from a raw NumPy array.

    Args:
        arr: 2-D ``(rows, cols)`` or 3-D ``(bands, rows, cols)`` array.
        crs: Required CRS (anything :func:`_coerce_epsg` accepts).
        transform: Required 6-tuple GDAL geotransform.
        nodata: Optional NoData scalar.

    Returns:
        A new in-memory :class:`Dataset`.

    Raises:
        ValueError: When ``crs`` or ``transform`` is missing.
    """
    # Function-scope import: avoids the dataset <-> engines.cog <-> cog cycle.
    from pyramids.dataset.dataset import Dataset

    if crs is None or transform is None:
        raise ValueError(
            "Writing a COG from a NumPy array requires both `crs` and "
            "`transform` (a 6-tuple GDAL geotransform)."
        )
    kwargs: dict[str, Any] = {"geo": tuple(transform), "epsg": _coerce_epsg(crs)}
    if nodata is not None:
        kwargs["no_data_value"] = nodata
    return Dataset.create_from_array(arr, **kwargs)


def _dataarray_to_dataset(
    da: Any,
    crs: Any | None,
    nodata: float | int | None,
) -> Dataset:
    """Build an in-memory :class:`Dataset` from an :class:`xarray.DataArray`.

    The geotransform is derived from the spatial coordinates (cell size from
    the first coordinate step, top-left from the first cell edge). The CRS is
    taken from the explicit ``crs`` argument, then from ``da.attrs['crs']``,
    then from a ``.rio`` accessor's ``crs`` if one is present on the object,
    otherwise an error is raised. The accessor is only consulted
    opportunistically — no extra dependency is required.

    Args:
        da: A 2-D or 3-D :class:`xarray.DataArray` with longitude/latitude
            (or x/y) coordinates.
        crs: Optional CRS override; wins over any embedded CRS.
        nodata: Optional NoData scalar.

    Returns:
        A new in-memory :class:`Dataset`.

    Raises:
        ValueError: When spatial coordinates or a CRS cannot be determined.
    """
    x_name = next((c for c in ("x", "longitude", "lon") if c in da.coords), None)
    y_name = next((c for c in ("y", "latitude", "lat") if c in da.coords), None)
    if x_name is None or y_name is None:
        raise ValueError(
            "Could not find longitude/latitude (or x/y) coordinates on the "
            "DataArray; build a Dataset explicitly and pass that instead."
        )

    x = np.asarray(da[x_name].values, dtype="float64")
    y = np.asarray(da[y_name].values, dtype="float64")
    if x.size < 2 or y.size < 2:
        raise ValueError("DataArray spatial coordinates need at least 2 cells.")
    cell_x = float(x[1] - x[0])
    cell_y = float(y[1] - y[0])
    transform = (
        float(x[0]) - cell_x / 2.0,
        cell_x,
        0.0,
        float(y[0]) - cell_y / 2.0,
        0.0,
        cell_y,
    )

    resolved_crs = crs if crs is not None else da.attrs.get("crs")
    if resolved_crs is None:
        rio = getattr(da, "rio", None)
        resolved_crs = getattr(rio, "crs", None) if rio is not None else None
    if resolved_crs is None:
        raise ValueError(
            "Could not determine a CRS for the DataArray; pass `crs=` "
            "explicitly (e.g. crs=4326)."
        )

    return _array_to_dataset(np.asarray(da.values), resolved_crs, transform, nodata)


def _normalize_to_dataset(
    data: Any,
    crs: Any | None,
    transform: tuple[float, ...] | None,
    nodata: float | int | None,
) -> Dataset:
    """Coerce any supported input into a pyramids :class:`Dataset`.

    Args:
        data: A :class:`Dataset`, :class:`osgeo.gdal.Dataset`,
            :class:`numpy.ndarray`, :class:`xarray.DataArray`, or a path
            to an existing raster.
        crs: Required only for the NumPy-array form.
        transform: Required only for the NumPy-array form.
        nodata: Optional NoData scalar; applied to array-built datasets and
            set on pre-built datasets when provided. For a read-only pre-built
            input (a path, or a read-only ``gdal.Dataset`` / ``Dataset``) the value
            is set on an in-memory copy so the source is never mutated and
            ``write_cog(<path>, ..., nodata=...)`` keeps working.

    Returns:
        A :class:`Dataset` ready to be written as a COG.

    Raises:
        ValueError: When required ``crs``/``transform`` are missing for an
            array input.
        TypeError: When ``data`` is an unsupported type.
    """
    # Function-scope import: avoids the dataset <-> engines.cog <-> cog cycle.
    from pyramids.dataset.dataset import Dataset

    ds: Dataset
    if isinstance(data, Dataset):
        ds = data
    elif isinstance(data, gdal.Dataset):
        ds = Dataset(data)
    elif isinstance(data, (str, Path)):
        ds = Dataset.read_file(data)
    elif isinstance(data, np.ndarray):
        return _array_to_dataset(data, crs, transform, nodata)
    elif type(data).__name__ == "DataArray":
        return _dataarray_to_dataset(data, crs, nodata)
    else:
        raise TypeError(
            "write_cog accepts a Dataset, gdal.Dataset, numpy.ndarray, "
            f"xarray.DataArray, or a path; got {type(data).__name__}."
        )

    if nodata is not None:
        # write_cog CreateCopies ds into the COG, so the nodata marker must be set on
        # ds first. A path / gdal.Dataset / Dataset input may be opened read-only
        # (read_file / Dataset() default to read_only), which the metadata-setter guard
        # now rejects to prevent a silent PAM spill — so apply nodata on an in-memory
        # copy in that case rather than mutating (or writing PAM next to) the source.
        try:
            ds.no_data_value = [nodata] * ds.band_count
        except ReadOnlyError:
            ds = ds.copy()
            ds.no_data_value = [nodata] * ds.band_count
    return ds


def write_cog(
    data: Any,
    output: str | Path,
    *,
    crs: Any | None = None,
    transform: tuple[float, ...] | None = None,
    nodata: float | int | None = None,
    options: CreationOptions | None = None,
    validate: bool = True,
    strict: bool = False,
) -> tuple[Path, ValidationReport | None]:
    """Write raster data to disk as a Cloud Optimized GeoTIFF.

    Thin convenience facade over
    :meth:`pyramids.dataset.engines.cog.COG.to_cog`. It accepts a wider
    range of inputs (NumPy array, ``xarray.DataArray``, ``gdal.Dataset``,
    path, or :class:`~pyramids.dataset.Dataset`), normalises them into a
    :class:`~pyramids.dataset.Dataset`, then **delegates the entire write
    to ``to_cog``** — which owns all COG policy: the house defaults
    (DEFLATE, 512px tiles, ``BIGTIFF=IF_SAFER``, ``NUM_THREADS=ALL_CPUS``,
    embedded statistics), the dtype-aware predictor (``2`` for integer,
    ``3`` for float), the dtype-aware default overview resampling
    (``mode`` for categorical, ``average`` for continuous), and the
    ``STATISTICS`` retry. Because policy lives in one place, ``write_cog``
    and a direct ``ds.to_cog(...)`` produce identical output for identical
    input. Any caller-supplied ``options`` are forwarded as ``extra`` and
    override the defaults. By default the result is round-tripped through
    :func:`pyramids.dataset.cog.validate.validate`.

    Args:
        data: Source raster. Accepted forms:

            - :class:`~pyramids.dataset.Dataset` — used directly.
            - :class:`osgeo.gdal.Dataset` — wrapped in a :class:`Dataset`.
            - :class:`numpy.ndarray` — requires ``crs`` and ``transform``.
            - :class:`xarray.DataArray` — geotransform is derived from the
              spatial coordinates; CRS from ``crs`` / DataArray metadata.
            - :class:`str` / :class:`~pathlib.Path` — an existing raster.
        output: Destination path. The parent directory must exist.
        crs: CRS for the NumPy-array form (EPSG int, ``"EPSG:XXXX"``, WKT,
            or PROJ string). Also used as a fallback for DataArrays.
        transform: 6-tuple GDAL geotransform; required for the array form.
        nodata: NoData scalar. For an array input it is passed to array
            construction; for a pre-built input it is applied before the write —
            on an in-memory copy when the source is read-only, so the source is
            never mutated.
        options: Caller overrides merged on top of
            :data:`PYRAMIDS_COG_DEFAULTS`. Keys are GDAL COG driver
            options (validated downstream). When ``PREDICTOR`` is absent it
            is auto-resolved from the raster dtype.
        validate: When ``True`` (default), validate the written file and
            raise :class:`RuntimeError` if it is not a valid COG.
        strict: Promote validation warnings to errors.

    Returns:
        A ``(output_path, report)`` tuple. ``report`` is a
        :class:`~pyramids.dataset.cog.validate.ValidationReport` when
        ``validate`` is ``True``, otherwise ``None``.

    Raises:
        ValueError: Required ``crs``/``transform`` missing for an array.
        TypeError: ``data`` is an unsupported type.
        RuntimeError: ``validate`` is ``True`` and the file failed COG
            validation.

    Examples:
        - Write a COG from a NumPy array (predictor auto-resolves to 3 for
          float):
            ```python
            >>> import numpy as np  # doctest: +SKIP
            >>> arr = np.random.rand(256, 256).astype("float32")  # doctest: +SKIP
            >>> path, report = write_cog(  # doctest: +SKIP
            ...     arr, "out.tif", crs=4326,
            ...     transform=(0.0, 0.01, 0.0, 10.0, 0.0, -0.01),
            ... )
            >>> report.is_valid  # doctest: +SKIP
            True

            ```
        - Re-encode an existing raster with overrides and skip validation:
            ```python
            >>> path, report = write_cog(  # doctest: +SKIP
            ...     "plain.tif", "scene_cog.tif",
            ...     options={"COMPRESS": "ZSTD", "LEVEL": 18},
            ...     validate=False,
            ... )
            >>> report is None  # doctest: +SKIP
            True

            ```
    """
    ds = _normalize_to_dataset(data, crs, transform, nodata)

    # Single write policy lives in COG.to_cog (ARC-1): house defaults, the
    # dtype-aware predictor (ARC-2), the category-safe default overview
    # resampling (ARC-3), and the STATISTICS retry (ARC-4) are all applied
    # there. write_cog only normalises the input and forwards the caller's
    # overrides as `extra`, so write_cog and a direct ds.to_cog(...) produce
    # identical output for identical input.
    output_path = ds.to_cog(output, extra=options or None)

    report: ValidationReport | None = None
    if validate:
        report = _validate_file(output_path, strict=strict)
        if not report.is_valid:
            raise RuntimeError(
                f"write_cog produced an invalid COG at {output_path}: {report.errors}"
            )
    return output_path, report
