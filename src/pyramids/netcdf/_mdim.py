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

from typing import Any

import numpy as np
from osgeo import gdal

#: PROJECTION attribute of the CF geostationary (GOES / Himawari / MTG) fixed-grid CRS.
GEOSTATIONARY_PROJECTION = "Geostationary_Satellite"


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


def dataset_is_geostationary(dataset: gdal.Dataset) -> bool:
    """Report whether ``dataset``'s CRS is the CF geostationary projection.

    Args:
        dataset: Any open GDAL dataset, including an ``AsClassicDataset`` view.

    Returns:
        ``True`` for a geostationary (GOES / Himawari / MTG) fixed-grid CRS.

    Examples:
        - A GOES granule's classic view reports the fixed-grid projection:
            ```python
            >>> dataset_is_geostationary(gdal.Open('NETCDF:"goes.nc":CMI'))  # doctest: +SKIP
            True

            ```
    """
    srs = dataset.GetSpatialRef() if dataset is not None else None
    return bool(
        srs is not None and srs.GetAttrValue("PROJECTION") == GEOSTATIONARY_PROJECTION
    )


def scaled_axis_ascends(dims: list, index: int) -> bool | None:
    """Report whether an axis' **scaled** coordinate increases, or ``None`` if it cannot be read.

    Mirrors GDAL's own classic netCDF driver rule (``frmts/netcdf/netcdfdataset.cpp``), which
    applies the coordinate variable's ``scale_factor``/``add_offset`` before deciding::

        yMinMax[i] = add_offset + yMinMax[i] * scale_factor
        bBottomUp  = (yMinMax[0] <= yMinMax[1])

    There ``yMinMax`` holds the axis' **first and last** values, so the rule is "the axis
    ascends". This reads them the same way, with two deliberate departures: a constant axis
    (``first == last``, which GDAL's ``<=`` would call ascending) is reported as unknown, and so
    is a non-finite endpoint — in both cases the geotransform sign is the better signal.

    An ``AsClassicDataset`` geotransform must **not** be used for this. GDAL derives it from the
    *raw* indexing-variable values (``GDALMDArray::GuessGeoTransform`` -> ``IsRegularlySpaced``),
    never from ``GetUnscaled()``. A coordinate packed with a **negative** ``scale_factor`` — the
    radian scan angles of a geostationary granule — therefore ascends in raw storage while the
    physical axis descends, so the sign is positive for an array that is already north-up.
    Trusting it mirrors the raster about its own geotransform (#705).

    Args:
        dims: The MDArray's dimensions, as returned by ``GetDimensions()``.
        index: Index of the axis' dimension within ``dims``.

    Returns:
        ``True``/``False`` when the scaled coordinate strictly increases/decreases; ``None``
        when the dimension exposes no usable 1-D coordinate (curvilinear grids, mesh files, bare
        index dimensions), when either endpoint is non-finite, or when the axis is constant / of
        size 1. The caller then falls back to the geotransform sign.

    Examples:
        - A geostationary ``y`` packed with a negative ``scale_factor`` descends physically, even
          though its raw values ascend:
            ```python
            >>> scaled_axis_ascends(cmi.GetDimensions(), 0)  # doctest: +SKIP
            False

            ```
        - A dimension with no coordinate variable cannot say:
            ```python
            >>> scaled_axis_ascends(mesh.GetDimensions(), 0)  # doctest: +SKIP

            ```
    """
    values = None
    result = None
    values, inverted = _coordinate_values(_indexing_variable(dims, index))
    if values is not None and values.ndim == 1 and values.size >= 2:
        first, last = float(values[0]), float(values[-1])
        # A NaN endpoint (a fill value in the coordinate) compares unequal to everything and
        # less-than nothing, so a naive `first < last` would silently call the axis descending
        # and mirror the raster. Report "unknown" and let the caller's fallback decide.
        if np.isfinite(first) and np.isfinite(last) and first != last:
            result = (first < last) != inverted
    return result


def _indexing_variable(dims: list, index: int) -> gdal.MDArray | None:
    """The dimension's 1-D coordinate variable, or ``None`` when it has none GDAL can read."""
    try:
        indexing_var = dims[index].GetIndexingVariable()
    except (RuntimeError, AttributeError):
        indexing_var = None
    if indexing_var is not None and indexing_var.GetDimensionCount() != 1:
        indexing_var = None
    return indexing_var


def _coordinate_values(indexing_var: gdal.MDArray | None) -> tuple[Any, bool]:
    """Read a coordinate's scaled values, or its raw values plus a "reverse me" flag.

    ``GetUnscaled()`` is a zero-copy view applying ``scale_factor``/``add_offset``; it is a no-op
    (scale 1, offset 0) for an unpacked coordinate. When GDAL declines to build it, never silently
    compare the RAW values: a negative ``scale_factor`` -- exactly what a geostationary granule packs
    its scan angle with -- reverses the physical direction, which is the whole of #705. Read raw and
    report that its order is inverted.

    Returns:
        ``(values, inverted)``, where ``values`` is ``None`` when nothing could be read.
    """
    values = None
    inverted = False
    if indexing_var is not None:
        try:
            unscaled = indexing_var.GetUnscaled()
        except (RuntimeError, AttributeError):
            unscaled = None
        try:
            if unscaled is not None:
                values = np.asarray(unscaled.ReadAsArray())
            else:
                values = np.asarray(indexing_var.ReadAsArray())
                scale = indexing_var.GetScale()
                inverted = scale is not None and scale < 0
        except (RuntimeError, AttributeError, TypeError):
            values = None
    return values, inverted


def y_axis_is_bottom_up(dims: list, y_index: int, classic_view: gdal.Dataset) -> bool:
    """Report whether the Y axis runs south-to-north and must be reversed to north-up order.

    Falls back to the geotransform sign only when the Y dimension exposes no usable 1-D
    coordinate — there the raw geotransform is all GDAL has. That fallback is *unsafe for a
    geostationary axis*: GDAL builds the geotransform from the raw scan angles, which ascend
    under the negative ``scale_factor`` a granule packs them with, so the sign says "flip" for an
    array that is already north-up (#705). Since such a cube goes on to adopt the classic
    driver's north-up metre geotransform, never flip it on that signal alone.

    Args:
        dims: The MDArray's dimensions.
        y_index: Index of the Y dimension within ``dims``.
        classic_view: The ``AsClassicDataset`` view, used only for the fallback.

    Returns:
        ``True`` when the array must be reversed along Y.

    Examples:
        - A south-to-north latitude must be reversed:
            ```python
            >>> y_axis_is_bottom_up(tas.GetDimensions(), 0, view)  # doctest: +SKIP
            True

            ```
    """
    ascends = scaled_axis_ascends(dims, y_index)
    if ascends is None:
        geostationary = dataset_is_geostationary(classic_view)
        result = classic_view.GetGeoTransform()[5] > 0 and not geostationary
    else:
        result = ascends
    return result


def x_axis_is_right_to_left(dims: list, x_index: int, classic_view: gdal.Dataset) -> bool:
    """Report whether the X axis runs east-to-west and must be reversed to west-to-east order.

    The exact mirror of :func:`y_axis_is_bottom_up`, applied to columns. GDAL's classic netCDF
    driver never flips X — it emits a negative ``gt[1]`` instead — but a negative pixel width
    leaks through pyramids' ``abs()``-based cell size and bbox arithmetic, so normalize the array
    to ``col 0 = west`` the same way rows are normalized to ``row 0 = north``.

    Args:
        dims: The MDArray's dimensions.
        x_index: Index of the X dimension within ``dims``.
        classic_view: The ``AsClassicDataset`` view, used only for the fallback.

    Returns:
        ``True`` when the array must be reversed along X.

    Examples:
        - A west-to-east longitude is already in raster order:
            ```python
            >>> x_axis_is_right_to_left(tas.GetDimensions(), 1, view)  # doctest: +SKIP
            False

            ```
    """
    ascends = scaled_axis_ascends(dims, x_index)
    if ascends is None:
        geostationary = dataset_is_geostationary(classic_view)
        result = classic_view.GetGeoTransform()[1] < 0 and not geostationary
    else:
        result = not ascends
    return result


def axis_flips(rg: gdal.Group, md_arr: gdal.MDArray) -> tuple[bool, bool]:
    """Return ``(needs_y_flip, needs_x_flip)`` for the array's **trailing** raster plane.

    Both flips are decided together — one ``AsClassicDataset`` build, one read of each coordinate —
    because every caller needs the pair. The plane is always the last two dimensions; unlike the
    eager ``get_variable`` path (which resolves an explicit / CF-detected ``x_dim``/``y_dim`` via
    ``_resolve_spatial_dims``), the lazy and ``_read_variable`` paths carry no plane override, so
    they normalize the trailing two axes. For every variable whose spatial plane *is* trailing —
    which is every on-disk fixture and the ordinary 2-D/3-D/4-D case — this matches the eager read.

    Args:
        rg: The root group, kept alive to prevent SWIG garbage collection of the view.
        md_arr: The MDArray to probe.

    Returns:
        ``(needs_y_flip, needs_x_flip)``; ``(False, False)`` for a 1-D array or when the orientation
        cannot be probed.
    """
    dims = md_arr.GetDimensions()
    result = (False, False)
    if len(dims) >= 2:
        try:
            view = md_arr.AsClassicDataset(len(dims) - 1, len(dims) - 2, rg)
            result = (
                y_axis_is_bottom_up(dims, len(dims) - 2, view),
                x_axis_is_right_to_left(dims, len(dims) - 1, view),
            )
        except Exception:  # nosec B110 - driver/orientation probe is best-effort
            result = (False, False)
    return result


def needs_y_flip(rg: gdal.Group, md_arr: gdal.MDArray) -> bool:
    """Report whether an MDArray's Y dimension is stored south-to-north.

    Decides from the coordinate variable with its ``scale_factor``/``add_offset`` applied (see
    :func:`scaled_axis_ascends`), the same rule the eager ``get_variable`` read uses, so the two
    read paths cannot disagree about which way up the data is.

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
    return axis_flips(rg, md_arr)[0]


def needs_x_flip(rg: gdal.Group, md_arr: gdal.MDArray) -> bool:
    """Report whether an MDArray's X dimension is stored east-to-west.

    The column-wise mirror of :func:`needs_y_flip`. A descending longitude is legal CF but no
    known producer writes one; without this, a lazy read would return the columns mirrored with
    respect to the eager read of the same variable.

    Args:
        rg: The root group (kept alive to prevent SWIG garbage collection of the view).
        md_arr: The MDArray to probe.

    Returns:
        ``True`` when the array has 2+ dimensions and is stored east-to-west.

    Examples:
        - A west-to-east grid is already in raster order:
            ```python
            >>> needs_x_flip(rg, rg.OpenMDArray("tas"))  # doctest: +SKIP
            False

            ```
    """
    return axis_flips(rg, md_arr)[1]
