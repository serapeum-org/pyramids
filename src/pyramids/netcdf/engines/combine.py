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

    Every cube's gaps stay gaps: the joined variable declares the first cube's sentinel
    and the others are rewritten to it. The one case that cannot come out right is a
    later cube holding the first cube's sentinel as a real measurement — that cell is
    read as missing afterwards, so change one of the sentinels before joining.

    Args:
        objs: The cubes, in the order they are joined. Containers or variables, at least
            one, all on the same grid and all carrying `dim`.
        dim: The dimension to join along.

    Returns:
        NetCDF: One cube whose `dim` is as long as the inputs' put together, holding each
        input's coordinates for it in order — or `None` for that dimension's coordinates
        when any cube contributes none, since a half-labelled axis would misdescribe its
        own cells.

    Raises:
        ValueError: `objs` is empty; the cubes carry different variables; a variable does
            not have `dim`; or their band dimensions do not otherwise line up.
        AlignmentError: The cubes are not on the same grid.
    """
    cubes = _checked(objs, "concat")
    first = cubes[0]
    names = _shared_variables(cubes, "concat")
    result = None
    time_attrs: dict = {}
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
        time_attrs.update(_carried_time_attrs(parts, band_names))
    cast("NetCDF", result)._band_dim_time_attrs = time_attrs
    return cast("NetCDF", result)


def _carried_time_attrs(parts: list[NetCDF], band_names: list[str]) -> dict:
    """The CF `(units, calendar)` the joined variable should carry for its dimensions.

    A rebuilt container has no store to read them from, so a join carries the parts'
    the way every other rebuild in the package does. Without them the joined axis is
    bare numbers again and a date `sel`, a frequency `reduce` or a decoding `to_xarray`
    all lose the calendar.

    Only a dimension every part agrees about is carried. A part that declares nothing
    counts as a disagreement: applying the other part's calendar to its bare stamps is
    how `6.0` comes to read as 6 hours since 2020 when it meant 6 days since 1990, and a
    confidently wrong calendar is worse than the undecodable axis this carry exists to
    fix. That is `_label_combined`'s policy for a dimension the operands disagree on.

    Args:
        parts: One variable per cube.
        band_names: The result's band dimensions.

    Returns:
        dict: The units per dimension every part declares the same, and no others.
    """
    declared: dict[str, list] = {name: [] for name in band_names}
    for part in parts:
        resolved = part._resolved_band_dim_time_attrs()
        for name in band_names:
            declared[name].append(resolved.get(name))
    return {
        name: attrs[0]
        for name, attrs in declared.items()
        if attrs[0] is not None and all(one == attrs[0] for one in attrs)
    }


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

    One cell this cannot get right, because only one sentinel can be declared: a later
    cube that *holds the first cube's sentinel as a measurement* keeps that number, and
    the joined cube then reads it as missing. Joining cubes whose sentinels collide with
    each other's data needs one of them changed first
    (`cube.change_no_data_value(...)`).

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
            (default) is xarray's rule: the copies are combined cell by cell, each filling
            the gaps of the other, and only a cell both of them hold a value in — a
            different value — is a conflict. `"override"` takes the first cube's copy as
            it stands, gaps and all, without reading any other.

    Returns:
        NetCDF: One container holding the union of the variables, in the order the cubes
        were given.

    Raises:
        ValueError: `objs` is empty, `compat` is unknown, or two cubes disagree about a
            cell they both judged under `"no_conflicts"`.
        AlignmentError: The cubes are not on the same grid.
    """
    if compat not in _COMPAT_MODES:
        raise ValueError(
            f"merge() takes compat={' or '.join(repr(one) for one in _COMPAT_MODES)}, "
            f"got {compat!r}."
        )
    cubes = _checked(objs, "merge")
    copies: dict[str, list[tuple[NetCDF, NetCDF]]] = {}
    order: list[str] = []
    for cube in cubes:
        for name in _variable_names(cube):
            if name not in copies:
                copies[name] = []
                order.append(name)
            copies[name].append((cube, _variable_of(cube, name)))
    result = None
    time_attrs: dict = {}
    for name in order:
        part = copies[name][0][1]
        result = cubes[0]._stack_reduced_variable(
            result,
            name,
            _merged_values(copies[name], name, compat),
            part.geotransform,
            crs_spec(part.epsg, part.crs),
            _read_no_data(part),
            list(part._band_dim_names),
            dict(part._band_dim_values_map),
        )
        carried = _carried_time_attrs([part], list(part._band_dim_names))
        for dim, attrs in carried.items():
            # Two variables sharing a dimension and declaring it differently leave it
            # undecodable rather than stamped with whichever was processed last.
            time_attrs[dim] = attrs if time_attrs.get(dim, attrs) == attrs else None
    cast("NetCDF", result)._band_dim_time_attrs = {
        dim: attrs for dim, attrs in time_attrs.items() if attrs is not None
    }
    return cast("NetCDF", result)


def _checked(objs: Any, caller: str) -> list[NetCDF]:
    """The cubes as a list, once there is at least one and they share a grid.

    Args:
        objs: The cubes as the caller gave them.
        caller: The member named in the refusals.

    Returns:
        list[NetCDF]: The cubes.

    Raises:
        ValueError: There are none, or one of them is not a cube.
        AlignmentError: They are not all on one grid.
    """
    cubes = list(objs)
    if not cubes:
        raise ValueError(f"{caller}() needs at least one cube to join, got none.")
    # Duck-typed on what the join reads rather than on the class, because importing
    # `NetCDF` here would close the cycle `netcdf -> combine -> netcdf`. Without the
    # check the first grid comparison reaches for `_band_dim_names` on whatever it was
    # handed and the caller gets an `AttributeError` from inside the join.
    strangers = [one for one in cubes if not hasattr(one, "_band_dim_names")]
    if strangers:
        raise ValueError(
            f"{caller}() joins NetCDF cubes — containers or variables — and was given "
            f"{type(strangers[0]).__name__}. Read the file with NetCDF.read_file(), or "
            f"take a variable with get_variable()."
        )
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

    def coordinates(part: NetCDF, name: str) -> tuple:
        """The dimension's stamps as a comparable tuple, empty when it carries none.

        Args:
            part: The variable.
            name: The dimension's name.

        Returns:
            tuple: The stamps as floats, or `()`.
        """
        stamps = part._band_dim_values_map.get(name)
        return () if stamps is None else tuple(float(one) for one in stamps)

    def layout(part: NetCDF) -> list[tuple[str, int, tuple]]:
        """The variable's dimensions other than `dim`, as comparable `(name, size, stamps)`.

        Args:
            part: The variable.

        Returns:
            list[tuple[str, int, tuple]]: One entry per other dimension, in array order.
        """
        sizes = list(part._band_dim_sizes)
        return [
            (other, sizes[index], coordinates(part, other))
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


def _merged_values(
    copies: list[tuple[NetCDF, NetCDF]], name: str, compat: str
) -> np.typing.NDArray:
    """The cells of a variable, taken from every cube that carries it.

    `"override"` takes the first copy and reads no other. `"no_conflicts"` is xarray's
    rule and not strict equality: a cell only conflicts when **both** copies hold a value
    there and the two differ, so complementary gaps fill each other in.

    Args:
        copies: The `(cube, variable)` pairs carrying this name, in the order given.
        name: The variable's name, for the refusal.
        compat: The mode.

    Returns:
        numpy.ndarray: The cells to write.

    Raises:
        ValueError: Two copies hold different values in a cell both of them judged, or
            their band dimensions do not line up.
    """
    cube, part = copies[0]
    values = np.asarray(cube._materialize_variable_array(part))
    if compat != "override":
        for other_cube, other_part in copies[1:]:
            values = _filled_from(values, part, other_cube, other_part, name)
    return values


def _filled_from(
    values: np.typing.NDArray,
    part: NetCDF,
    cube: NetCDF,
    other: NetCDF,
    name: str,
) -> np.typing.NDArray:
    """`values` with the gaps another copy of the same variable can fill.

    The conversion to NaN and back is only paid when a cell is actually taken, so two
    gapless copies — or two whose gaps line up — leave an integer band integer.

    Args:
        values: The cells taken so far.
        part: The variable they came from, for its sentinel and band dimensions.
        cube: The cube offering another copy, for its array helper.
        other: That copy.
        name: The variable's name, for the refusals.

    Returns:
        numpy.ndarray: The cells, with what the other copy could add.

    Raises:
        ValueError: The layouts differ, a band dimension is stamped differently, or a
            cell both copies judged disagrees.
    """
    theirs = np.asarray(cube._materialize_variable_array(other))
    if theirs.shape != values.shape or tuple(other._band_dim_names) != tuple(
        part._band_dim_names
    ):
        raise ValueError(
            f"merge() found {name!r} in more than one cube with different layouts: one is "
            f"{values.shape} over {tuple(part._band_dim_names)} and another is "
            f"{theirs.shape} over {tuple(other._band_dim_names)}."
        )
    # `merge` joins no dimension, so every one of them has to line up before the cells
    # can. Without this the copies are fused and stamped with the first's coordinates:
    # two measurements a hundred hours apart become one step. `concat` is the member for
    # putting them end to end.
    for dim in part._band_dim_names:
        mine_stamps = _stamps(part, dim)
        their_stamps = _stamps(other, dim)
        if mine_stamps != their_stamps:
            raise ValueError(
                f"merge() found {name!r} in more than one cube with different "
                f"coordinates for {dim!r}: one is {list(mine_stamps)} and another is "
                f"{list(their_stamps)}. Use concat() to put them end to end along "
                f"{dim!r}."
            )
    sentinel = _read_no_data(part)
    mine = _gaps_as_nan(values, sentinel)
    yours = _gaps_as_nan(theirs, _read_no_data(other))
    my_gaps = np.isnan(mine)
    if np.any(~my_gaps & ~np.isnan(yours) & (mine != yours)):
        raise ValueError(
            f"merge() found {name!r} in more than one cube with different values. Pass "
            f"compat='override' to take the first copy, or drop the variable from one of "
            f"them."
        )
    takeable = my_gaps & ~np.isnan(yours)
    if takeable.any():
        filled = np.where(my_gaps, yours, mine)
        if sentinel is not None and not np.isnan(sentinel):
            filled = _narrowed(
                np.where(np.isnan(filled), sentinel, filled), values.dtype
            )
    else:
        filled = values
    return np.asarray(filled)


def _stamps(part: NetCDF, dim: str) -> tuple:
    """A dimension's coordinates as a comparable tuple, empty when it carries none.

    Args:
        part: The variable.
        dim: The dimension's name.

    Returns:
        tuple: The stamps as floats, or `()`.
    """
    values = part._band_dim_values_map.get(dim)
    return () if values is None else tuple(float(one) for one in values)


def _narrowed(filled: np.typing.NDArray, dtype: np.dtype) -> np.typing.NDArray:
    """`filled` back in the band's own type, unless an integer band would lose by it.

    The masking to NaN and back is a float round trip, and casting it straight back to
    the band's type is right for a float band — that is ordinary precision. On an integer
    band it is not: a borrowed `2.7` would be stored as `2`, and `70000.0` would wrap to
    `4464`, both silently. So an integer band takes its type back only when every value
    is a whole number inside its range, and otherwise the result widens — the same
    judgement a masked raster's dtype makes.

    Args:
        filled: The combined values, as floats.
        dtype: The first copy's band type.

    Returns:
        numpy.ndarray: `filled` in `dtype`, or as it stands when that would lose a value.
    """
    values = filled
    if not np.issubdtype(dtype, np.integer):
        values = filled.astype(dtype, copy=False)
    else:
        limits = np.iinfo(dtype)
        whole = bool(np.isfinite(filled).all()) and bool(
            np.all(filled == np.trunc(filled))
        )
        if (
            whole
            and limits.min <= float(filled.min())
            and float(filled.max()) <= limits.max
        ):
            values = filled.astype(dtype, copy=False)
    return values
