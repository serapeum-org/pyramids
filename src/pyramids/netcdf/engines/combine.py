"""Joining cubes: `concat` along a dimension, `merge` by variable.

The two ways two cubes become one. `concat` puts them end to end along a dimension they
both have — two halves of a time series making the whole — and `merge` puts their variables
side by side on the grid they share.

Both refuse a grid mismatch rather than resampling: `align` is the explicit step for that,
and doing it implicitly inside a join would move data without saying so.

**A name to keep apart.** `DatasetCollection.merge(dst, ...)` already means a *spatial
mosaic written to `dst`* — several rasters covering neighbouring ground becoming one. This
`NetCDF.merge` is the other operation entirely: one grid, several variables. They live on
different classes and take different arguments, and `NetCDF.merge`'s docstring names the
other one so the two are not mistaken for each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pyramids.base._errors import AlignmentError
from pyramids.base.crs import crs_spec
from pyramids.netcdf.engines._along_dim import (
    _gaps_as_nan,
    _read_no_data,
    _reduces_as_a_variable,
)

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF

_COMPAT_MODES = ("no_conflicts", "override")
"""What `merge` does with a variable both cubes carry."""


def concat(objs: Any, dim: str) -> NetCDF:
    """Join cubes end to end along one of their dimensions.

    Args:
        objs: The cubes, in the order they are joined. Containers or variables, at least
            one, all on the same grid and all carrying `dim`.
        dim: The dimension to join along.

    Returns:
        NetCDF: One cube whose `dim` is as long as the inputs' put together, holding each
        input's coordinates for it in order.

    Raises:
        ValueError: `objs` is empty; the cubes carry different variables; a variable does
            not have `dim`; or their band dimensions do not otherwise line up.
        AlignmentError: The cubes are not on the same grid.
    """
    cubes = _checked(objs, "concat")
    first = cubes[0]
    names = _shared_variables(cubes, "concat")
    result = None
    for name in names:
        parts = [_variable_of(cube, name) for cube in cubes]
        band_names = list(parts[0]._band_dim_names)
        if dim not in band_names:
            raise ValueError(
                f"concat() joins along a non-spatial dimension, and {name!r} has "
                f"{band_names or 'none'}; {dim!r} is not among them."
            )
        _check_other_dimensions(parts, dim, name)
        axis = band_names.index(dim)
        values, sentinel = _joined_values(cubes, parts, axis)
        values_map = dict(parts[0]._band_dim_values_map)
        values_map[dim] = _joined_coordinates(parts, dim)
        result = first._stack_reduced_variable(
            result,
            name,
            values,
            parts[0].geotransform,
            crs_spec(parts[0].epsg, parts[0].crs),
            sentinel,
            band_names,
            values_map,
        )
    return cast("NetCDF", result)


def _joined_values(
    cubes: list[NetCDF], parts: list[NetCDF], axis: int
) -> tuple[np.ndarray, Any]:
    """The cubes' values end to end, with every cube's gaps still marked as gaps.

    Each cube writes its **own** sentinel into its gap cells, and the joined variable can
    declare only one. Taking the first cube's and concatenating the rest as stored turns
    every other cube's gaps into ordinary numbers — a `-1.0` that was "missing" becomes a
    measurement, and every later mean or fill consumes it. So when the cubes disagree, each
    is masked with its own sentinel first and the chosen one is written back.

    Cubes that already agree are concatenated as they are, which keeps an integer band
    integer: the conversion is only paid when it buys something.

    Args:
        cubes: The cubes being joined, for their array helpers.
        parts: The matching variable of each cube.
        axis: The axis to join along.

    Returns:
        tuple: The joined values, and the no-data value the result should declare.
    """
    arrays = [
        np.asarray(cube._materialize_variable_array(part))
        for cube, part in zip(cubes, parts)
    ]
    sentinels = [_read_no_data(part) for part in parts]
    first = sentinels[0]
    if all(_same_sentinel(one, first) for one in sentinels):
        return np.concatenate(arrays, axis=axis), first
    target = first if first is not None else np.nan
    normalised = [
        np.where(np.isnan(_gaps_as_nan(array, sentinel)), target, array)
        for array, sentinel in zip(arrays, sentinels)
    ]
    return np.concatenate(normalised, axis=axis), target


def _same_sentinel(one: Any, other: Any) -> bool:
    """Whether two declared no-data values mean the same thing.

    Args:
        one: A cube's sentinel, or `None`.
        other: Another cube's sentinel, or `None`.

    Returns:
        bool: `True` when both are absent, both are NaN, or both are the same number.
    """
    if one is None or other is None:
        return one is None and other is None
    if np.isnan(one) and np.isnan(other):
        return True
    return bool(one == other)


def merge(objs: Any, *, compat: str = "no_conflicts") -> NetCDF:
    """Put the variables of several cubes side by side on the grid they share.

    Args:
        objs: The cubes, containers or variables, at least one and all on the same grid.
        compat: What to do with a variable more than one cube carries. `"no_conflicts"`
            (default) accepts it only when the cubes agree on its values and refuses
            otherwise; `"override"` takes the first cube's copy without comparing.

    Returns:
        NetCDF: One container holding the union of the variables, in the order the cubes
        were given.

    Raises:
        ValueError: `objs` is empty, `compat` is unknown, or two cubes carry the same
            variable with different values under `"no_conflicts"`.
        AlignmentError: The cubes are not on the same grid.
    """
    if compat not in _COMPAT_MODES:
        raise ValueError(
            f"merge() takes compat={' or '.join(repr(one) for one in _COMPAT_MODES)}, "
            f"got {compat!r}."
        )
    cubes = _checked(objs, "merge")
    result = None
    taken: dict[str, NetCDF] = {}
    for cube in cubes:
        for name in _variable_names(cube):
            part = _variable_of(cube, name)
            if name in taken:
                _check_no_conflict(taken[name], part, name, compat)
                continue
            taken[name] = part
            result = cubes[0]._stack_reduced_variable(
                result,
                name,
                np.asarray(cube._materialize_variable_array(part)),
                part.geotransform,
                crs_spec(part.epsg, part.crs),
                _read_no_data(part),
                list(part._band_dim_names),
                dict(part._band_dim_values_map),
            )
    return cast("NetCDF", result)


def _checked(objs: Any, caller: str) -> list[NetCDF]:
    """The cubes as a list, once there is at least one and they share a grid.

    Args:
        objs: The cubes as the caller gave them.
        caller: The member named in the refusals.

    Returns:
        list[NetCDF]: The cubes.

    Raises:
        ValueError: There are none.
        AlignmentError: They are not all on one grid.
    """
    cubes = list(objs)
    if not cubes:
        raise ValueError(f"{caller}() needs at least one cube to join, got none.")
    first = cubes[0]
    for cube in cubes[1:]:
        if not _grids_match(first, cube):
            raise AlignmentError(
                f"{caller}() needs every cube on the same grid. The first is "
                f"{first.rows}x{first.columns} at {first.geotransform} in EPSG:{first.epsg}, "
                f"and another is {cube.rows}x{cube.columns} at {cube.geotransform} in "
                f"EPSG:{cube.epsg}. Align them first (`other = other.align(first)`)."
            )
    return cubes


def _grids_match(first: NetCDF, other: NetCDF) -> bool:
    """Whether two cubes describe the same cells.

    Compared through the variables rather than the containers: a container's own raster is
    a placeholder, so `same_grid` on two containers compares two placeholders.

    Args:
        first: The cube the others are checked against.
        other: The cube being checked.

    Returns:
        bool: `True` when the grids agree.
    """
    left = _first_gridded(first)
    right = _first_gridded(other)
    return bool(left.spatial.same_grid(right))


def _first_gridded(cube: NetCDF) -> NetCDF:
    """The cube itself when it is a variable, or its first gridded variable.

    Args:
        cube: The container or variable.

    Returns:
        NetCDF: Something carrying a real grid.
    """
    return (
        cube
        if _reduces_as_a_variable(cube)
        else _variable_of(cube, _variable_names(cube)[0])
    )


def _variable_names(cube: NetCDF) -> list[str]:
    """The gridded variables of a container, or the one name a variable answers to.

    Args:
        cube: The container or variable.

    Returns:
        list[str]: The names.
    """
    if _reduces_as_a_variable(cube):
        return [cube._source_var_name or "variable"]
    return list(cube._spatial_variable_names(cube._working_group()))


def _variable_of(cube: NetCDF, name: str) -> NetCDF:
    """One variable of a cube, or the cube itself when it is one.

    Args:
        cube: The container or variable.
        name: The variable's name.

    Returns:
        NetCDF: The variable.
    """
    return cube if _reduces_as_a_variable(cube) else cube._require_raster_variable(name)


def _shared_variables(cubes: list[NetCDF], caller: str) -> list[str]:
    """The variables every cube carries, in the first cube's order.

    Args:
        cubes: The cubes.
        caller: The member named in the refusal.

    Returns:
        list[str]: The names.

    Raises:
        ValueError: The cubes do not carry the same variables.
    """
    names = _variable_names(cubes[0])
    for cube in cubes[1:]:
        theirs = _variable_names(cube)
        if set(theirs) != set(names):
            raise ValueError(
                f"{caller}() joins cubes that carry the same variables: one has {names} "
                f"and another has {theirs}. Use merge() to put different variables side "
                f"by side."
            )
    return names


def _check_other_dimensions(parts: list[NetCDF], dim: str, name: str) -> None:
    """Refuse cubes whose other band dimensions do not line up.

    Args:
        parts: One variable per cube.
        dim: The dimension being joined along, which may differ in length.
        name: The variable's name, for the refusal.

    Raises:
        ValueError: The other band dimensions differ in name or in length.
    """

    def layout(part: NetCDF) -> list[tuple[str, int]]:
        sizes = list(part._band_dim_sizes)
        return [
            (other, sizes[index])
            for index, other in enumerate(part._band_dim_names)
            if other != dim
        ]

    first = layout(parts[0])
    for part in parts[1:]:
        if layout(part) != first:
            raise ValueError(
                f"concat() needs {name!r} to agree on every dimension but {dim!r}: one is "
                f"{first} and another is {layout(part)}."
            )


def _joined_coordinates(parts: list[NetCDF], dim: str) -> list | None:
    """The joined dimension's coordinates, or `None` when any cube has none for it.

    Args:
        parts: One variable per cube.
        dim: The dimension being joined along.

    Returns:
        list | None: The coordinates end to end, or `None` when one cube cannot contribute
        its own — a half-labelled axis would say the stamps describe cells they do not.
    """
    joined: list = []
    for part in parts:
        stamps = part._band_dim_values_map.get(dim)
        if stamps is None:
            return None
        joined.extend(list(stamps))
    return joined


def _check_no_conflict(kept: NetCDF, candidate: NetCDF, name: str, compat: str) -> None:
    """Refuse a variable two cubes carry with different values.

    Args:
        kept: The copy already taken.
        candidate: The copy another cube offers.
        name: The variable's name.
        compat: The mode; `"override"` skips the comparison.

    Raises:
        ValueError: The two disagree and `compat` is `"no_conflicts"`.
    """
    if compat == "override":
        return
    if not kept.equals(candidate):
        raise ValueError(
            f"merge() found {name!r} in more than one cube with different values. Pass "
            f"compat='override' to take the first copy, or drop the variable from one of "
            f"them."
        )
