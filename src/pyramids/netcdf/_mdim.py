"""Shared low-level GDAL multidimensional (MDIM) helpers.

This module collects the small MDIM primitives that were previously duplicated across
``netcdf.py``, ``_lazy.py``, ``labeled.py`` and the ``ugrid`` subpackage: opening a root
group, opening an MDArray behind the ``RuntimeError``/``None`` guard, reducing a per-band
NoData list to a scalar, and probing whether a stored array runs south-to-north (so its Y
axis needs flipping to a north-up raster).

Keeping these in one place means a behavioural fix — e.g. a new GDAL guard — lands once
instead of in a dozen near-identical copies. The helpers are intentionally thin and
side-effect free; the heavier, workaround-coupled read paths (strided windowed reads and
the GDAL 3.13 ``AsClassicDataset`` materialisation) deliberately stay in ``netcdf.py``.
"""

from __future__ import annotations

from typing import Any, cast

from osgeo import gdal


def root_group(ds: gdal.Dataset, *, required: bool = False) -> gdal.Group | None:
    """Return the MDIM root group of ``ds``.

    Args:
        ds: An open GDAL dataset.
        required: When ``True``, raise :class:`ValueError` if the dataset exposes no
            root group (i.e. it is not an MDIM NetCDF/HDF5/Zarr input) instead of
            returning ``None``.

    Returns:
        The :class:`osgeo.gdal.Group` root group, or ``None`` when the dataset has no
        multidimensional model and ``required`` is ``False``.

    Examples:
        - Fetch the root group of an opened multidim store:
            ```python
            >>> from osgeo import gdal  # doctest: +SKIP
            >>> ds = gdal.OpenEx("data.nc", gdal.OF_MULTIDIM_RASTER)  # doctest: +SKIP
            >>> rg = root_group(ds)  # doctest: +SKIP
            >>> sorted(rg.GetMDArrayNames())  # doctest: +SKIP
            ['lat', 'lon', 'tas']

            ```
        - A classic (non-MDIM) dataset yields ``None`` unless ``required``:
            ```python
            >>> from osgeo import gdal  # doctest: +SKIP
            >>> ds = gdal.Open("plain.tif")  # doctest: +SKIP
            >>> root_group(ds) is None  # doctest: +SKIP
            True

            ```
    """
    rg = ds.GetRootGroup()
    if rg is None and required:
        raise ValueError(
            "Dataset has no root group; multidimensional reads require MDIM "
            "(NetCDF/HDF5/Zarr) inputs."
        )
    return rg


def open_mdarray(rg: gdal.Group, name: str) -> gdal.MDArray | None:
    """Open an MDArray by name, returning ``None`` instead of raising.

    GDAL raises :class:`RuntimeError` (or returns ``None``) for a missing array; this
    folds both outcomes into a single ``None`` so callers can guard with a plain
    ``if md_arr is None``.

    Args:
        rg: The store's multidimensional root group.
        name: The MDArray (variable) name.

    Returns:
        The opened :class:`osgeo.gdal.MDArray`, or ``None`` when it cannot be opened.

    Examples:
        - Open a variable that exists:
            ```python
            >>> arr = open_mdarray(rg, "tas")  # doctest: +SKIP
            >>> arr.GetDimensionCount()  # doctest: +SKIP
            3

            ```
        - A missing variable returns ``None`` instead of raising:
            ```python
            >>> open_mdarray(rg, "does_not_exist") is None  # doctest: +SKIP
            True

            ```
    """
    try:
        return rg.OpenMDArray(name)
    except RuntimeError:
        return None


def scalar_no_data(no_data_value: Any) -> Any:
    """Reduce a per-band NoData list/tuple to its first (scalar) value.

    Scalars pass through unchanged; an empty list/tuple is returned as-is.

    Args:
        no_data_value: A scalar NoData value or a per-band sequence of them.

    Returns:
        The scalar NoData value (the first element of a non-empty sequence).

    Examples:
        - A per-band list collapses to its first value:
            ```python
            >>> scalar_no_data([-9999.0, -9999.0, -9999.0])
            -9999.0

            ```
        - A scalar passes straight through:
            ```python
            >>> scalar_no_data(0)
            0

            ```
        - An empty sequence is returned unchanged:
            ```python
            >>> scalar_no_data([])
            []

            ```
    """
    if isinstance(no_data_value, (list, tuple)) and no_data_value:
        return no_data_value[0]
    return no_data_value


def needs_y_flip(rg: gdal.Group, md_arr: gdal.MDArray) -> bool:
    """Report whether an MDArray's Y dimension is stored south-to-north.

    Uses ``AsClassicDataset`` to inspect the derived geotransform's Y pixel size: a
    positive value means the rows ascend south-to-north and must be flipped to produce
    a north-up raster.

    Args:
        rg: The root group (passed through to ``AsClassicDataset`` and kept alive to
            prevent SWIG garbage collection of the view).
        md_arr: The MDArray to probe.

    Returns:
        ``True`` when the array has 2+ dimensions and is stored south-to-north;
        ``False`` for 1-D arrays or already north-up data.

    Examples:
        - A south-to-north 2-D grid needs flipping to a north-up raster:
            ```python
            >>> needs_y_flip(rg, rg.OpenMDArray("tas"))  # doctest: +SKIP
            True

            ```
        - A 1-D coordinate array never needs flipping:
            ```python
            >>> needs_y_flip(rg, rg.OpenMDArray("lat"))  # doctest: +SKIP
            False

            ```
    """
    dims = md_arr.GetDimensions()
    if len(dims) < 2:
        return False
    try:
        src = md_arr.AsClassicDataset(len(dims) - 1, len(dims) - 2, rg)
        return cast("bool", src.GetGeoTransform()[5] > 0)
    except Exception:  # nosec B110 - driver/orientation probe is best-effort
        return False
