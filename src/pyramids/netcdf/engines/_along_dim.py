"""The loop every operation along a non-spatial dimension runs through, and those operations.

`reduce` and `coarsen` transform each variable's values along one of its band dimensions and
rebuild the result. What differs between them is the per-variable step; what they share is
everything around it — the receiver's checks, which variables of a container take part, how
the auxiliary variables are carried or dropped, how a coordinate-less dimension and the CF time
units a result carries survive the rebuild. That shared part is `_apply_to_variable` and
`_apply_to_container`, and each member hands it an `_AlongDim` describing its own step.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

from pyramids.base.crs import crs_spec
from pyramids.netcdf._mdim import scalar_no_data

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF


class _Applied(NamedTuple):
    """What an operation made of one variable: the values and the band layout describing them.

    Attributes:
        values: The unflattened result, `(*band_dim_sizes, rows, cols)`, in numpy.
        band_names: The result's band dimensions, outermost first.
        values_map: Each band dimension's coordinates, or `None` for one without.
        no_data: The no-data value the result declares, or `None` for none.
    """

    values: np.ndarray
    band_names: list[str]
    values_map: dict[str, Any]
    no_data: Any


class _AlongDim(ABC):
    """One operation `_apply_to_variable` and `_apply_to_container` run along a band dimension.

    Attributes:
        caller: The member the user called, named in refusals and warnings.
        verb: How the refusal of an empty container names the operation, as in
            `"Cannot <verb> an empty container (no data variables)."`.
        keeps_length: Whether the dimension keeps its length. A container's auxiliary variable
            spanning a dimension that keeps its length is carried over unchanged; one spanning a
            dimension that changes length is dropped with a warning.
    """

    caller: str = ""
    verb: str = ""
    keeps_length: bool = False

    def start(self) -> None:
        """Work out what waits for the receiver to pass its own checks. Nothing, by default."""

    @abstractmethod
    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Run the operation on one variable that has `dim`.

        Args:
            nc: The object the member was called on, which owns the array helpers.
            var: The variable; `dim` is one of its band dimensions.
            dim: The dimension the operation runs along.

        Returns:
            _Applied: The result's values and band layout.
        """


@dataclass
class _Reduction(_AlongDim):
    """`reduce` and `coarsen`: reduce each group of steps along a dimension, or collapse it.

    Attributes:
        how: The reduction.
        groups: Resolves the groups of positions, or `None` to collapse — called by `start`,
            once the receiver has passed its checks, so a bad dimension is reported before a
            grouping is worked out.
        skipna: Whether gaps are skipped.
        q: The quantile, for `how="quantile"`.
        caller: `"reduce"` or `"coarsen"`.
        resize: The length to cut or pad the dimension to first, for `coarsen`; `None` leaves
            it alone.
        window_mean_coords: Label each group with the mean of its members' coordinates
            (`coarsen`) instead of its first member's (`reduce`).
    """

    how: str
    groups: Callable[[], list | None]
    skipna: bool
    q: float | None
    caller: str = "reduce"
    resize: int | None = None
    window_mean_coords: bool = False
    _positions: list | None = field(default=None, init=False, repr=False)

    @property
    def verb(self) -> str:  # type: ignore[override]
        """The member's own name: `"Cannot reduce ..."`, `"Cannot coarsen ..."`."""
        return self.caller

    def start(self) -> None:
        """Resolve the groups."""
        self._positions = self.groups()

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Reduce one variable along `dim` through `_reduced_array`.

        Args:
            nc: The object the member was called on.
            var: The variable.
            dim: The dimension to reduce.

        Returns:
            _Applied: The reduced values and band layout.
        """
        return _Applied(
            *_reduced_array(
                nc,
                var,
                dim,
                self.how,
                group_positions=self._positions,
                skipna=self.skipna,
                q=self.q,
                resize=self.resize,
                window_mean_coords=self.window_mean_coords,
            )
        )


def _reduced_array(
    nc: NetCDF,
    var: NetCDF,
    dim: str,
    how: str,
    *,
    group_positions: list | None,
    skipna: bool,
    q: float | None,
    resize: int | None = None,
    window_mean_coords: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, Any], Any]:
    """Reduce one raster variable along `dim`, the step a container and a variable share.

    A file-backed variable that still reads as its store (`NetCDF._reads_as_its_store`) is read
    as a chunked dask array when dask is installed, and the `np.*` / `np.nan*` reducers dispatch
    to dask on it, so the reduction stays lazy until `np.asarray` computes the reduced result
    (ARC-47); `_reduce_variable_array` needs no dask-specific code. Anything else — an in-memory
    variable, a cut or other derived variable, or any variable without dask — is read eagerly.

    Args:
        nc: The object `reduce` was called on, which owns the reduce helpers.
        var: The variable to reduce; `dim` must be one of its band dimensions.
        dim: The dimension to reduce.
        how: The reduction.
        group_positions: The resolved groups, or `None` to collapse.
        skipna: Whether gaps are skipped.
        q: The quantile, for `how="quantile"`.
        resize: The length to cut or pad `dim` to before grouping, for `coarsen`;
            `None` leaves it alone.
        window_mean_coords: Label each group with the mean of its members' coordinates
            (`coarsen`) instead of its first member's (`reduce`).

    Returns:
        tuple: The reduced numpy array, its band dimension names, its coordinate map, and
        the no-data value the reduced band declares — `None` for a count, `255` for a flag,
        otherwise the variable's sentinel in the units the reduction read (`_read_no_data`).

    Raises:
        ValueError: `group_positions` does not cover `dim`, after any `resize`, exactly.
    """
    # Local import breaks the netcdf.py <-> engines.selection import cycle.
    from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _FLAG_NO_DATA

    band_names = list(var._band_dim_names)
    values_map = dict(var._band_dim_values_map)
    ndv = _read_no_data(var)
    axis = band_names.index(dim)
    coords = values_map.get(dim)
    arr = nc._materialize_variable_array(var, lazy=True)
    size = arr.shape[axis]
    if resize is not None and resize != size:
        arr = _resize_axis(arr, axis, resize)
    arr, band_names, values_map = nc._reduce_variable_array(
        arr,
        axis,
        dim,
        band_names,
        values_map,
        how,
        skipna,
        ndv,
        None,
        group_positions,
        q,
    )
    if window_mean_coords and group_positions is not None:
        values_map[dim] = _window_coordinates(coords, group_positions, size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arr = np.asarray(arr)
    result_ndv = ndv
    if how == "count":
        result_ndv = None
    elif how in _COUNTING_REDUCERS:
        result_ndv = _FLAG_NO_DATA
    return arr, band_names, values_map, result_ndv


def _apply_to_variable(nc: NetCDF, dim: str, op: _AlongDim) -> NetCDF:
    """Run `op` along `dim` of a single variable and hand back a variable.

    The result carries `nc`'s units for the band dimensions it keeps (`_band_dim_time_attrs`),
    since the container it is rebuilt from has no store to read them from.

    Args:
        nc: The variable.
        dim: The dimension the operation runs along.
        op: The operation.

    Returns:
        NetCDF: The result, named after `nc` or `"variable"` when `nc` has no name of its own.

    Raises:
        ValueError: `nc` has no band dimensions, or `dim` is not one of them — both checked
            before `op` starts — or `op` refuses the variable (for `reduce`, a frequency with no
            decodable time coordinate, or labels that do not cover `dim`).
    """
    _assert_band_dimension(nc, dim, caller=op.caller)
    op.start()
    arr, band_names, values_map, ndv = op.apply(nc, nc, dim)
    name = nc._source_var_name or "variable"
    container = nc._stack_reduced_variable(
        None,
        name,
        arr,
        nc.geotransform,
        crs_spec(nc.epsg, nc.crs),
        ndv,
        band_names,
        values_map,
    )
    variable = cast("NetCDF", container.get_variable(name))
    # `from_array` numbers a dimension it is given no coordinates for, so a dimension this
    # variable held unlabelled (an operator result's dropped stamps) would come back stamped
    # 0..n-1 and `sel` would match positions as if they were stamps. Put the gap back.
    unlabelled = [
        dim_name for dim_name in band_names if values_map.get(dim_name) is None
    ]
    for dim_name in unlabelled:
        variable._band_dim_values_map[dim_name] = None
    if unlabelled:
        variable._band_dim_name, variable._band_dim_values = (
            variable._derive_primary_band_view(
                variable._band_dim_names,
                variable._band_dim_values_map,
                variable._band_dim_sizes,
                variable._band_count,
            )
        )
    # The rebuilt container has no store to read time units from, so carry the operand's for
    # the dimensions that kept their stamps.
    variable._band_dim_time_attrs = {
        dim_name: attrs
        for dim_name, attrs in nc._resolved_band_dim_time_attrs().items()
        if dim_name in variable._band_dim_names and dim_name not in unlabelled
    }
    return variable


def _apply_to_container(nc: NetCDF, dim: str, op: _AlongDim) -> NetCDF:
    """Run `op` along `dim` of every gridded variable of a container that has it.

    Gridded variables without `dim` are carried over, as are auxiliary variables that do not
    span it. An auxiliary variable that spans `dim` is carried over too when `op` keeps the
    dimension's length, and dropped with a warning when it does not. The result container
    carries the source variables' units for the band dimensions they keep
    (`_band_dim_time_attrs`), which `get_variable` finds through the result and copies onto the
    variable it takes from it.

    Args:
        nc: The container.
        dim: The dimension the operation runs along.
        op: The operation.

    Returns:
        NetCDF: The result container.

    Raises:
        ValueError: The container has no data variables (checked before `op` starts), no
            gridded variable has `dim`, or `op` refuses a variable.

    Warns:
        UserWarning: An auxiliary variable spans `dim`, whose length `op` changes, and is
            dropped, or one that is kept cannot be carried over. Both messages name
            `op.caller`.
    """
    names = nc.variable_names
    if not names:
        raise ValueError(f"Cannot {op.verb} an empty container (no data variables).")

    op.start()

    # Reduce only the gridded variables; non-spatial auxiliaries (no y/x axes)
    # can't go through the raster reduce path, so they are carried through
    # unchanged below — the same split crop / to_crs use (#513). Resolve the root
    # group once and reuse it for the spanning-aux probe further down.
    rg = nc._working_group()
    spatial_vars = nc._spatial_variable_names(rg)
    aux_vars = nc._carryable_aux_names(rg, spatial_vars)

    result = None
    found = False
    time_attrs: dict[str, tuple[str, str]] = {}
    for var_name in spatial_vars:
        var = nc._require_raster_variable(var_name)
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)

        if dim in band_names:
            found = True
            arr, band_names, values_map, ndv = op.apply(nc, var, dim)
        else:
            arr = nc._materialize_variable_array(var)

        result = nc._stack_reduced_variable(
            result,
            var_name,
            arr,
            var.geotransform,
            crs_spec(var.epsg, var.crs),
            ndv,
            band_names,
            values_map,
        )
        # The rebuilt container has no store to read time units from; carry the source
        # variables' so a variable taken from it still decodes its stamps.
        time_attrs.update(
            {
                name: attrs
                for name, attrs in var._resolved_band_dim_time_attrs().items()
                if name in band_names
            }
        )

    if not found:
        raise ValueError(
            f"Dimension {dim!r} is not a non-spatial dimension of any "
            f"variable in this container."
        )
    # Auxiliary variables that span the reduced dimension cannot be carried
    # verbatim — they would keep the full-length axis while the gridded
    # variables collapse it, leaving an inconsistent dimension length. Drop
    # those with a warning; carry the rest unchanged.
    carry_aux: list[str] = []
    spanning_aux: list[str] = []
    for name in aux_vars:
        var_dims = nc._variable_dim_names(rg, name)
        spans = dim in var_dims and not op.keeps_length
        (spanning_aux if spans else carry_aux).append(name)
    if spanning_aux:
        warnings.warn(
            f"{op.caller}() dropped auxiliary variable(s) {spanning_aux} that span "
            f"the reduced dimension {dim!r}; carrying them unchanged would "
            f"leave an inconsistent {dim!r} length in the result.",
            # stacklevel=4: the user calls NetCDF.reduce (or NetCDF.coarsen), which
            # forwards through the one-line façade to the Selection method, which calls
            # this helper, so the user's call site is four frames up.
            stacklevel=4,
        )
    cast("NetCDF", result)._band_dim_time_attrs = time_attrs
    nc._carry_aux_variables(cast("NetCDF", result), carry_aux, op.caller)
    return cast("NetCDF", result)


def _reduces_as_a_variable(nc: NetCDF) -> bool:
    """Whether `reduce` / `coarsen` treat `nc` as one variable rather than a container.

    A `Variable` is one. So is anything that carries band dimensions: an operator result takes
    its left operand's class, so a classic-mode NetCDF on the left of a labelled variable gives
    a `Container`-class raster holding the right operand's layout. A root container, opened
    multidimensional or classic, has no band dimensions of its own, so this never sends one
    down the variable path.

    Args:
        nc: The object `reduce` or `coarsen` was called on.

    Returns:
        bool: `True` for a `Variable` or anything carrying band dimensions.
    """
    # Local import breaks the netcdf.py <-> engines.selection import cycle.
    from pyramids.netcdf.netcdf import Variable

    return isinstance(nc, Variable) or bool(nc._band_dim_names)


def _assert_band_dimension(nc: NetCDF, dim_name: str, *, caller: str) -> None:
    """Refuse a name that is not one of this variable's band dimensions.

    Shared by `sel`, `isel`, `reduce` and `coarsen` so they report an unknown dimension
    identically — the plan for `isel` asks for exactly the `ValueError` `sel` already
    raises, and the only way to keep that true is to raise it in one place.

    Args:
        nc: The variable subset being selected from or reduced.
        dim_name: The dimension the caller named.
        caller: `"sel"`, `"isel"`, `"reduce"` or `"coarsen"`, named in the message about a
            variable with no band dimensions.

    Raises:
        ValueError: The variable tracks no band dimensions, or `dim_name` is not one.
    """
    if not nc._band_dim_names:
        raise ValueError(
            f"{caller}() requires a variable with at least one non-spatial "
            f"dimension. This variable has no band dimensions tracked."
        )
    if dim_name not in nc._band_dim_names:
        raise ValueError(
            f"Dimension {dim_name!r} does not match any band dimension "
            f"of this variable {list(nc._band_dim_names)!r}."
        )


def _read_no_data(var: NetCDF) -> Any:
    """The no-data value as it appears in the values a reduction reads.

    The reduce path reads a variable unpacked, so a CF-packed variable's fill cells hold
    `_FillValue * scale_factor + add_offset`, never the stored `_FillValue` itself. Masking
    against the stored value would count every fill cell as data. The sentinel is unpacked the
    same way the read unpacks the data (`Analysis._physical_no_data`), so the two compare equal.
    An unpacked variable's sentinel is returned unchanged.

    Args:
        var: The variable being reduced or carried over.

    Returns:
        Any: The sentinel in read units, or `None` when the variable declares none.
    """
    ndv = scalar_no_data(var.no_data_value)
    return None if ndv is None else var.analysis._physical_no_data(0)


def _resize_axis(arr: Any, axis: int, size: int) -> Any:
    """Cut `axis` down to `size` steps, or pad it out to `size` with NaN gaps.

    Padding casts to float64 first, so an integer band can hold the NaN. Under `skipna`
    every reducer skips the padding; without it a statistic over a padded window is NaN,
    `count` still leaves the padding out, and `all` / `any` read it as true. A `size` equal
    to the current length takes the padding path too and returns a float64 copy. Both paths
    stay lazy on a dask array.

    Args:
        arr: The unflattened array, numpy or dask.
        axis: The axis to resize.
        size: The length it should have.

    Returns:
        The resized array.
    """
    current = arr.shape[axis]
    if size < current:
        index: list[slice] = [slice(None)] * arr.ndim
        index[axis] = slice(0, size)
        result = arr[tuple(index)]
    else:
        padding_shape = list(arr.shape)
        padding_shape[axis] = size - current
        result = np.concatenate(
            [arr.astype("float64"), np.full(padding_shape, np.nan)], axis=axis
        )
    return result


def _window_coordinates(
    coords: list | None, positions: list[np.ndarray], size: int
) -> list | None:
    """Label each window with the mean of its real members' coordinates.

    Args:
        coords: The dimension's coordinate values, or `None`.
        positions: The positions each window covers, padding included.
        size: The dimension's real length; positions at or past it are padding.

    Returns:
        list | None: One float per window when every coordinate is a number (a boolean does
        not count as one), each window's first coordinate when some are not, and `None`
        when there are none.
    """
    labels = None
    if coords is not None:
        numeric = all(
            isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            for value in coords
        )
        if numeric:
            labels = [
                float(np.mean([coords[int(i)] for i in members if i < size]))
                for members in positions
            ]
        else:
            labels = [coords[int(members[0])] for members in positions]
    return labels
