"""Weighted statistics, the one operation here that can reduce the spatial axes.

`reduce`, `coarsen`, `rolling` and the rest work along one band dimension and leave the grid
alone, so they share `_along_dim`'s loop. A weighted mean over latitude and longitude instead
collapses the grid itself — the answer has no cells to sit in — so it runs here, rebuilding the
result on a grid of one cell spanning the source extent. That keeps it a `NetCDF`: it still
carries its band dimensions and their stamps, and `sel`, `isel`, `reduce` and `to_file` all
still work on it.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pyramids.base.crs import crs_from_user_input, crs_spec, require_crs_spec
from pyramids.netcdf.engines._along_dim import (
    _Applied,
    _carry_auxiliaries,
    _gaps_as_nan,
    _read_no_data,
    _reduces_as_a_variable,
    _variable_from_applied,
)

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF

_WEIGHTED_HOWS = ("mean", "sum", "sum_of_weights", "std", "var")
"""The statistics `weighted` computes. A weighted quantile is not among them: it needs the
values sorted per cell against a running weight, which is a different algorithm from these
sums, so `reduce(how="quantile")` remains the unweighted answer."""

_ROW_ALIAS = "y"
"""The name `from_array` gives the row axis, accepted for any store's row dimension."""

_COLUMN_ALIAS = "x"
"""The name `from_array` gives the column axis, accepted for any store's column dimension."""


def _weighted_result(
    nc: NetCDF, weights: Any, dims: Any, *, how: str, skipna: bool
) -> NetCDF:
    """Weight `nc` over `dims` and rebuild the result.

    Args:
        nc: The container or variable `weighted` was called on.
        weights: `"area"`, an array broadcastable to the weighted axes, or a `NetCDF` on the
            same grid.
        dims: The dimensions to weight over; `None` asks for both spatial axes.
        how: One of `_WEIGHTED_HOWS`.
        skipna: Whether gaps are skipped.

    Returns:
        NetCDF: A container for a container, a variable for a variable.

    Raises:
        ValueError: As `_weighted_axes` and `_weights_for` raise, or the container has no data
            variables.
    """
    if _reduces_as_a_variable(nc):
        result = _weighted_variable(nc, nc, weights, dims, how=how, skipna=skipna)
    else:
        result = _weighted_container(nc, weights, dims, how=how, skipna=skipna)
    return result


def _weighted_variable(
    nc: NetCDF, var: NetCDF, weights: Any, dims: Any, *, how: str, skipna: bool
) -> NetCDF:
    """Weight one variable and hand back a variable.

    Args:
        nc: The object `weighted` was called on, which owns the array helpers.
        var: The variable to weight.
        weights: The weights, as `weighted` takes them.
        dims: The dimensions to weight over.
        how: The statistic.
        skipna: Whether gaps are skipped.

    Returns:
        NetCDF: The weighted variable, on a grid reduced where a spatial axis went.

    Raises:
        ValueError: As `_weighted_axes` and `_weights_for` raise.
    """
    applied, geotransform = _weighted_applied(
        nc, var, weights, dims, how=how, skipna=skipna
    )
    return _variable_from_applied(var, applied, geotransform)


def _weighted_container(
    nc: NetCDF, weights: Any, dims: Any, *, how: str, skipna: bool
) -> NetCDF:
    """Weight every gridded variable of a container.

    Every gridded variable takes part when a spatial axis is weighted, since every one has the
    grid. When a band dimension is weighted, a gridded variable without it is carried over
    unchanged, as `reduce` carries it.

    Args:
        nc: The container.
        weights: The weights, as `weighted` takes them.
        dims: The dimensions to weight over.
        how: The statistic.
        skipna: Whether gaps are skipped.

    Returns:
        NetCDF: The weighted container.

    Raises:
        ValueError: The container has no data variables, or as `_weighted_axes` and
            `_weights_for` raise.

    Warns:
        UserWarning: An auxiliary variable spans a weighted band dimension and is dropped.
    """
    if not nc.variable_names:
        raise ValueError("Cannot weight an empty container (no data variables).")
    rg = nc._working_group()
    spatial_vars = nc._spatial_variable_names(rg)
    aux_vars = nc._carryable_aux_names(rg, spatial_vars)
    result = None
    removed: list[str] = []
    time_attrs: dict[str, tuple[str, str]] = {}
    for var_name in spatial_vars:
        var = nc._require_raster_variable(var_name)
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)
        geotransform = var.geotransform
        names = _weighted_names(var, dims)
        if all(name in band_names or name not in var._band_dim_names for name in names):
            applied, geotransform = _weighted_applied(
                nc, var, weights, dims, how=how, skipna=skipna
            )
            arr, band_names, values_map, ndv = applied
            removed = [name for name in names if name in var._band_dim_names]
        else:
            arr = nc._materialize_variable_array(var)
        result = nc._stack_reduced_variable(
            result,
            var_name,
            arr,
            geotransform,
            crs_spec(var.epsg, var.crs),
            ndv,
            band_names,
            values_map,
        )
        time_attrs.update(
            {
                name: attrs
                for name, attrs in var._resolved_band_dim_time_attrs().items()
                if name in band_names
            }
        )
    cast("NetCDF", result)._band_dim_time_attrs = time_attrs
    _carry_auxiliaries(nc, cast("NetCDF", result), rg, aux_vars, removed, "weighted")
    return cast("NetCDF", result)


def _weighted_applied(
    nc: NetCDF, var: NetCDF, weights: Any, dims: Any, *, how: str, skipna: bool
) -> tuple[_Applied, tuple]:
    """The weighted values of one variable, and the geotransform they sit on.

    Args:
        nc: The object `weighted` was called on.
        var: The variable.
        weights: The weights, as `weighted` takes them.
        dims: The dimensions to weight over.
        how: The statistic.
        skipna: Whether gaps are skipped.

    Returns:
        tuple: The `_Applied` result and its geotransform.

    Raises:
        ValueError: As `_weighted_axes` and `_weights_for` raise.
    """
    band_names = list(var._band_dim_names)
    values_map = dict(var._band_dim_values_map)
    ndv = _read_no_data(var)
    axes, names = _weighted_axes(var, dims)
    arr = nc._materialize_variable_array(var, lazy=True)
    spread = _weights_for(var, weights, arr.shape, axes)
    values = _weighted_statistic(arr, spread, axes, how, ndv, skipna)
    # The reduced spatial axes stay as one cell, since the result is still a raster; a reduced
    # band dimension goes, as a collapsing `reduce` drops it.
    band_axes = tuple(axis for axis in axes if axis < len(band_names))
    if band_axes:
        values = np.squeeze(values, axis=band_axes)
    kept = [name for name in band_names if name not in names]
    layout = {name: values_map.get(name) for name in kept}
    rows = len(band_names) in axes
    columns = len(band_names) + 1 in axes
    return (
        _Applied(np.asarray(values), kept, layout, np.nan),
        _weighted_geotransform(var, rows, columns),
    )


def _spatial_names(var: NetCDF) -> tuple[str, str]:
    """The names of the variable's row and column axes.

    A store variable reports them as its last two dimensions (`latitude` / `longitude`,
    `y` / `x`, ...). A variable built in memory, or derived by an operator, may report none, and
    answers the `y` / `x` a rebuild gives it.

    Args:
        var: The variable.

    Returns:
        tuple[str, str]: The row axis' name and the column axis' name.
    """
    declared = list(var._md_array_dims)
    return (
        (declared[-2], declared[-1])
        if len(declared) >= 2
        else (_ROW_ALIAS, _COLUMN_ALIAS)
    )


def _weighted_names(var: NetCDF, dims: Any) -> tuple[str, ...]:
    """The dimension names `weighted` was asked for, as a tuple.

    Args:
        var: The variable, whose spatial axes name the default.
        dims: `None` for both spatial axes, one name, or a sequence of names.

    Returns:
        tuple[str, ...]: The requested names.
    """
    if dims is None:
        names: tuple[str, ...] = _spatial_names(var)
    elif isinstance(dims, str):
        names = (dims,)
    else:
        names = tuple(dims)
    return names


def _weighted_axes(var: NetCDF, dims: Any) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """The axes of the unflattened array `weighted` reduces, and the names that chose them.

    The array is laid out `(*band_dim_sizes, rows, columns)`, so a band dimension is its own
    position in `_band_dim_names`, the row axis follows them and the column axis follows that.

    Args:
        var: The variable.
        dims: `None` for both spatial axes, one name, or a sequence of names.

    Returns:
        tuple: The axes, ascending, and the names as given.

    Raises:
        ValueError: `dims` is empty, names a dimension the variable does not have, names one
            twice, or mixes a band dimension with a spatial axis — which would leave a
            container's variables on two different grids.
    """
    band_names = list(var._band_dim_names)
    row_name, column_name = _spatial_names(var)
    names = _weighted_names(var, dims)
    if not names:
        raise ValueError("weighted() needs at least one dimension to weight over.")
    axes: list[int] = []
    spatial = 0
    for name in names:
        if name in band_names:
            axis = band_names.index(name)
        elif name in (row_name, _ROW_ALIAS):
            axis = len(band_names)
            spatial += 1
        elif name in (column_name, _COLUMN_ALIAS):
            axis = len(band_names) + 1
            spatial += 1
        else:
            available = [*band_names, row_name, column_name]
            raise ValueError(
                f"weighted() cannot weight over {name!r}: this variable has {available}."
            )
        if axis in axes:
            raise ValueError(f"weighted() was given {name!r} twice.")
        axes.append(axis)
    if spatial and spatial != len(names):
        raise ValueError(
            "weighted() takes either spatial axes or band dimensions, not both: weighting a "
            "band dimension keeps the grid, while weighting a spatial axis reduces it, and a "
            "container cannot hold variables on two grids."
        )
    return tuple(sorted(axes)), names


def _weights_for(var: NetCDF, weights: Any, shape: tuple, axes: tuple[int, ...]) -> Any:
    """The weights as an array broadcastable to the variable's own shape.

    Args:
        var: The variable being weighted.
        weights: `"area"` for cos-latitude weights, an array broadcastable to the weighted
            axes' shape, or a `NetCDF` on the same grid.
        shape: The unflattened shape of the variable, `(*band_dim_sizes, rows, columns)`.
        axes: The axes being weighted.

    Returns:
        The weights, shaped to broadcast against `shape`.

    Raises:
        ValueError: `"area"` on a grid that is not geographic; an unknown name; a `NetCDF` on
            another grid; weights holding a gap; or weights that do not broadcast onto the
            weighted axes.
    """
    if isinstance(weights, str):
        if weights != "area":
            raise ValueError(
                f"weighted() knows the weighting 'area' (cos-latitude); got {weights!r}. "
                f"Pass an array or a NetCDF for anything else."
            )
        values = _area_weights(var)
    elif hasattr(weights, "read_array"):
        # A container's own raster is a placeholder, not its variables' grid, so compare and read
        # through its first gridded variable.
        if not _reduces_as_a_variable(weights) and getattr(
            weights, "variable_names", None
        ):
            weights = weights._require_raster_variable(weights.variable_names[0])
        if not var.spatial.same_grid(weights):
            raise ValueError(
                "weighted() needs weights on the same grid as the variable; the raster passed "
                "has a different grid."
            )
        values = np.asarray(weights.read_array(), dtype="float64")
        if values.ndim > 2:
            values = values[0]
    else:
        values = np.asarray(weights, dtype="float64")
    if np.isnan(values).any():
        raise ValueError(
            "weighted() weights cannot contain missing values; replace them with zero to "
            "leave those cells out."
        )
    weighted_shape = tuple(shape[axis] for axis in axes)
    try:
        spread = np.asarray(np.broadcast_to(values, weighted_shape)).reshape(
            _placed_shape(shape, axes, weighted_shape)
        )
    except ValueError:
        # Weights describing every cell — the grid, say, while only its rows are weighted —
        # broadcast against the variable itself instead of against the weighted axes alone.
        try:
            spread = np.broadcast_to(values, shape)
        except ValueError:
            raise ValueError(
                f"weighted() cannot broadcast weights of shape {values.shape} onto the "
                f"weighted axes {weighted_shape}, nor onto the variable's {tuple(shape)}."
            ) from None
    return spread


def _placed_shape(
    shape: tuple, axes: tuple[int, ...], weighted_shape: tuple[int, ...]
) -> list[int]:
    """The weighted axes' lengths at their own positions, and `1` everywhere else.

    Args:
        shape: The variable's unflattened shape.
        axes: The weighted axes.
        weighted_shape: Their lengths.

    Returns:
        list[int]: The shape the weights take so they broadcast against the variable.
    """
    placed = [1] * len(shape)
    for axis, length in zip(axes, weighted_shape):
        placed[axis] = length
    return placed


def _area_weights(var: NetCDF) -> np.ndarray:
    """`cos(latitude)` per row of a geographic grid, shaped `(rows, 1)`.

    Args:
        var: The variable, whose geotransform gives each row's latitude.

    Returns:
        numpy.ndarray: One weight per row.

    Raises:
        ValueError: The variable has no CRS, or its CRS is not geographic, where a latitude is
            not what the rows measure.
    """
    crs = crs_from_user_input(
        require_crs_spec(var.epsg, var.crs, "compute area weights")
    )
    if not crs.is_geographic:
        raise ValueError(
            "weights='area' is cos(latitude), which needs a geographic CRS; this raster is in "
            f"{crs.name!r}. Pass an array of weights, or warp it to a geographic CRS first."
        )
    geo = var.geotransform
    latitudes = np.asarray(
        [geo[3] + (row + 0.5) * geo[5] for row in range(var.rows)], dtype="float64"
    )
    return np.cos(np.deg2rad(latitudes)).reshape(var.rows, 1)


def _weighted_statistic(
    arr: Any, spread: Any, axes: tuple[int, ...], how: str, ndv: Any, skipna: bool
) -> Any:
    """The weighted statistic of `arr` over `axes`, the reduced axes kept as length 1.

    A gap leaves both sums, so a weighted mean is the mean of the cells there are. A slice with
    no valid cell, or whose weights total zero, has no statistic and comes back NaN — the
    `sum_of_weights` of a slice with valid cells is still its total, zero included.

    Args:
        arr: The unflattened values, numpy or dask.
        spread: The weights, shaped to broadcast against `arr`.
        axes: The axes to reduce.
        how: One of `_WEIGHTED_HOWS`.
        ndv: The sentinel as it appears in `arr`, or `None`.
        skipna: Whether gaps are skipped.

    Returns:
        The statistic, float64.
    """
    data = _gaps_as_nan(arr, ndv) if skipna else arr.astype("float64")
    valid = ~np.isnan(data)
    weights = np.broadcast_to(spread, data.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        total = np.sum(np.where(valid, weights, 0.0), axis=axes, keepdims=True)
        anything = np.any(valid, axis=axes, keepdims=True)
        weighted_sum = np.sum(
            np.where(valid, weights * np.where(valid, data, 0.0), 0.0),
            axis=axes,
            keepdims=True,
        )
        usable = anything & (total != 0)
        safe = np.where(total == 0, 1.0, total)
        mean = weighted_sum / safe
        if how == "sum_of_weights":
            values = np.where(anything, total, np.nan)
        elif how == "sum":
            values = np.where(usable, weighted_sum, np.nan)
        elif how == "mean":
            values = np.where(usable, mean, np.nan)
        else:
            deviation = np.sum(
                np.where(valid, weights * np.where(valid, data - mean, 0.0) ** 2, 0.0),
                axis=axes,
                keepdims=True,
            )
            variance = deviation / safe
            values = np.where(
                usable, variance if how == "var" else np.sqrt(variance), np.nan
            )
        result = np.asarray(values)
    return result


def _weighted_geotransform(var: NetCDF, rows: bool, columns: bool) -> tuple:
    """The geotransform of a result whose spatial axes were reduced.

    A reduced axis becomes one cell spanning the source's whole extent along it, so the result
    covers exactly the bounding box the source covered.

    Args:
        var: The source variable.
        rows: Whether the row axis was reduced.
        columns: Whether the column axis was reduced.

    Returns:
        tuple: The geotransform.
    """
    geo = list(var.geotransform)
    if columns:
        geo[1] = geo[1] * var.columns
    if rows:
        geo[5] = geo[5] * var.rows
    return tuple(geo)
