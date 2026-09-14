"""Mechanical "same behaviour as xarray" assertions for the NetCDF surface.

`pyramids` ships `to_xarray()`, so for any operation implemented on both sides the acceptance
test for parity can be written once::

    assert_parity(nc.read_array(variable="t"), to_xr(nc, "t"))

Five systematic differences make the naive comparison fail, and each is a thing to *assert*
rather than paper over. The whole point of this module is that they are normalised in one
place, visibly, instead of being re-derived (and re-got-wrong) per test:

1. **Y orientation.** pyramids presents every raster north-up; `to_xarray()` reports the
   coordinate in *storage* order. On a y-ascending file the two run opposite ways. The flip is
   decided by the stored latitude coordinate itself — ascending means storage order is the
   reverse of raster order — and never by the geotransform, which one repo fixture
   (`cf__5v__1d4-4d1__geog__y-desc.nc`) does not carry meaningfully.
2. **No-data vs NaN.** pyramids declares a sentinel; xarray has only NaN. Both sides are masked
   to NaN at the sentinel and the two gap masks are asserted equal, so a disagreement about
   *where* the gaps are cannot hide behind a value comparison that skips them.
3. **dtype.** A reduction returns float64 whatever went in, so values are compared with
   `assert_allclose` and the dtype contract is asserted separately, by name, when the caller
   states one.
4. **CF packing.** `read_array()` unpacks by default; `to_xarray()` hands over the *stored*
   values. The xarray side is unpacked here with the variable's own `scale_factor` /
   `add_offset` so both sides are in physical units.
5. **Band order.** GDAL flattens every non-spatial dimension row-major into one bands axis;
   xarray keeps them separate. The pyramids array is reshaped back through
   :func:`~pyramids.netcdf._mdim.unflatten_band_axes` before comparing.

The sentinel mask is taken from the **stored** values on both sides, never from the unpacked
ones. On a packed variable `read_array()` scales the fill value along with the data, so the
declared sentinel no longer occurs in the result: on `cf__20v__1d3-3d17__y-desc.nc` the 63072
cells holding `-32767` come back as `0.0748…`, indistinguishable from a real measurement.
Masking after unpacking would therefore find no gaps at all and quietly compare fill against
fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pyramids.netcdf._mdim import unflatten_band_axes
from pyramids.netcdf.netcdf import NetCDF

#: Dimension names pyramids' raster view treats as the y axis, in the order they are looked for.
Y_DIMENSION_NAMES = ("lat", "latitude", "y", "yc")


@dataclass(frozen=True)
class ParityView:
    """One side of a parity comparison, normalised onto pyramids' raster orientation.

    Attributes:
        values: Physical values as `float64`, north-up along the y axis, with every gap as
            `NaN`. Shaped `(*band_dims, rows, cols)`.
        gaps: Boolean mask of the no-data cells, shaped like `values`.
        dims: The dimension names of `values`, spatial axes last.
        coords: The dimension coordinates, y normalised to raster (north-up) order.
        dtype: The dtype the *source* held, before the float64 promotion this view applies.
    """

    values: np.ndarray
    gaps: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    dtype: np.dtype


def y_dimension(nc: NetCDF) -> str | None:
    """The name of a container's y dimension, or ``None`` when it declares none.

    Args:
        nc: The container to inspect.

    Returns:
        The first of :data:`Y_DIMENSION_NAMES` the container declares, or `None`.

    Examples:
        - A CF fixture names it `lat`:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import y_dimension
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> y_dimension(nc)
          'lat'

          ```
    """
    sizes = nc.dimension_sizes
    return next((name for name in Y_DIMENSION_NAMES if name in sizes), None)


def stored_y_ascends(nc: NetCDF) -> bool:
    """Whether the y coordinate is stored south-to-north, so storage order reverses the raster.

    Args:
        nc: The container to inspect.

    Returns:
        `True` when the stored y coordinate increases with its index, which means
        `to_xarray()`'s array runs opposite to `read_array()`'s north-up rows. `False` for a
        descending axis, and for a container with no y dimension or fewer than two y cells,
        where there is nothing to flip.

    Examples:
        - The y-ascending fixture is stored south-to-north:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import stored_y_ascends
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> stored_y_ascends(nc)
          True

          ```
        - The y-descending one is already in raster order:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import stored_y_ascends
          >>> nc = NetCDF.read_file("tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc")
          >>> stored_y_ascends(nc)
          False

          ```
    """
    name = y_dimension(nc)
    ascends = False
    if name is not None:
        values = np.asarray(nc.get_dimension_values(name))
        if values.size > 1:
            ascends = bool(values[0] < values[-1])
    return ascends


def _packing(variable: Any) -> tuple[float, float]:
    """The variable's CF `scale_factor` / `add_offset`, defaulted to the identity."""
    scale = variable.scale[0]
    offset = variable.offset[0]
    return (
        1.0 if scale is None else float(scale),
        0.0 if offset is None else float(offset),
    )


def _gap_mask(stored: np.ndarray, sentinel: Any) -> np.ndarray:
    """Where the stored values hold the declared sentinel.

    Args:
        stored: The values as the file holds them, before any unpacking.
        sentinel: The declared no-data value, or `None`.

    Returns:
        A boolean array shaped like `stored`; all `False` when nothing is declared.
    """
    if sentinel is None:
        mask = np.zeros(stored.shape, dtype=bool)
    elif isinstance(sentinel, float) and np.isnan(sentinel):
        mask = np.isnan(stored)
    else:
        mask = stored == sentinel
    return mask


def to_xr(nc: NetCDF, variable: str) -> ParityView:
    """The xarray view of ``variable``, normalised onto pyramids' raster orientation.

    Applies normalisations 1, 2 and 4: the y axis is flipped to north-up when the file stores it
    ascending, the stored values are unpacked into physical units, and the sentinel cells become
    `NaN`.

    Args:
        nc: The container to export.
        variable: The variable to take from the exported cube.

    Returns:
        ParityView: The xarray side, ready to compare.

    Raises:
        KeyError: `variable` is not in the exported cube.

    Examples:
        - The view is north-up and physical, whatever the file stores:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import to_xr
          >>> nc = NetCDF.read_file("tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc")
          >>> view = to_xr(nc, "rhum")
          >>> view.dims
          ('time', 'level', 'lat', 'lon')
          >>> float(view.values.max().round(3))
          100.0

          ```
    """
    exported = nc.to_xarray()
    if variable not in exported:
        raise KeyError(
            f"{variable!r} is not in the exported cube; it holds {sorted(exported)}"
        )
    array = exported[variable]
    handle = nc.get_variable(variable)
    scale, offset = _packing(handle)

    stored = np.asarray(array.values)
    gaps = _gap_mask(stored, handle.no_data_value[0])
    values = np.where(gaps, np.nan, stored.astype("float64") * scale + offset)

    name = y_dimension(nc)
    coords = {
        key: np.asarray(exported.coords[key].values)
        for key in array.dims
        if key in exported.coords
    }
    if name in array.dims and stored_y_ascends(nc):
        axis = list(array.dims).index(name)
        values = np.flip(values, axis)
        gaps = np.flip(gaps, axis)
        if name in coords:
            coords[name] = coords[name][::-1]
    return ParityView(values, gaps, tuple(array.dims), coords, stored.dtype)


def from_pyramids(nc: NetCDF, variable: str, values: Any = None) -> ParityView:
    """The pyramids view of ``variable``, reshaped onto the dimensions xarray keeps.

    Applies normalisations 2 and 5: the flat GDAL bands axis is rebuilt into the variable's own
    band dimensions, and the sentinel cells become `NaN` — taken from the *stored* values,
    because unpacking scales the fill value along with the data.

    Args:
        nc: The container to read.
        variable: The variable to read.
        values: An already-computed result to normalise instead of reading, in raster order
            and physical units. Defaults to `read_array(variable=variable)`.

    Returns:
        ParityView: The pyramids side, ready to compare.

    Examples:
        - Reading a two-band-dimension variable rebuilds both axes:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import from_pyramids
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> from_pyramids(nc, "temperature").values.shape
          (4, 3, 5, 6)

          ```
    """
    handle = nc.get_variable(variable)
    scale, offset = _packing(handle)
    stored = np.asarray(nc.read_array(variable=variable, unpack=False))
    gaps = _gap_mask(stored, handle.no_data_value[0])

    physical = (
        np.asarray(nc.read_array(variable=variable), dtype="float64")
        if values is None
        else np.asarray(values, dtype="float64")
    )
    physical = np.where(gaps, np.nan, physical)

    # `_band_dim_names` is the variable's own non-spatial axes in storage order, which is what
    # GDAL flattened; the variable's `dimension_names` is not usable here because a subset
    # renames its y dimension (`lat` -> `subset_lat_4_-1_5`).
    band_dims = tuple(handle._band_dim_names)
    sizes = tuple(handle._band_dim_sizes)
    if len(band_dims) > 1:
        physical = unflatten_band_axes(physical, band_dims, sizes)
        gaps = unflatten_band_axes(gaps, band_dims, sizes)

    y_name = y_dimension(nc)
    spatial = [
        name
        for name in (nc.dimension_names or [])
        if name not in band_dims and name != y_name
    ]
    dims = (*band_dims, *(n for n in (y_name,) if n), *spatial)

    coords: dict[str, np.ndarray] = {}
    for name in dims:
        stored_coord = np.asarray(nc.get_dimension_values(name))
        if name == y_name and stored_y_ascends(nc):
            stored_coord = stored_coord[::-1]
        coords[name] = stored_coord
    return ParityView(physical, gaps, dims, coords, stored.dtype)


def assert_parity(
    pyr: ParityView,
    xr_obj: ParityView,
    *,
    rtol: float = 1e-9,
    dtype: Any = None,
) -> None:
    """Assert that a pyramids result and the xarray view agree.

    Checks the four things a parity claim actually rests on, in the order that fails most
    informatively: the shape, then *where* the gaps are, then the values, then the dtype
    contract when the caller states one.

    Args:
        pyr: The pyramids side, from :func:`from_pyramids`.
        xr_obj: The xarray side, from :func:`to_xr`.
        rtol: Relative tolerance for the value comparison.
        dtype: The dtype the pyramids side must hold, when the operation contracts for one.
            `None` skips the check — a reduction promotes to float64 whatever went in, and
            the promotion is the contract rather than a violation of it.

    Raises:
        AssertionError: Any of the four checks fails.

    Examples:
        - A no-op read matches its own export on every fixture:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import assert_parity, from_pyramids, to_xr
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> assert_parity(from_pyramids(nc, "temperature"), to_xr(nc, "temperature"))

          ```
    """
    assert pyr.values.shape == xr_obj.values.shape, (
        f"shape differs: pyramids {pyr.values.shape} vs xarray {xr_obj.values.shape}"
        f" (dims {pyr.dims} vs {xr_obj.dims})"
    )
    assert np.array_equal(pyr.gaps, xr_obj.gaps), (
        f"the gap masks differ: pyramids marks {int(pyr.gaps.sum())} cells, "
        f"xarray {int(xr_obj.gaps.sum())}, disagreeing on "
        f"{int(np.logical_xor(pyr.gaps, xr_obj.gaps).sum())}"
    )
    np.testing.assert_allclose(
        pyr.values,
        xr_obj.values,
        rtol=rtol,
        equal_nan=True,
        err_msg="the values differ after orientation, packing and gap normalisation",
    )
    if dtype is not None:
        assert np.dtype(pyr.dtype) == np.dtype(dtype), (
            f"dtype contract broken: expected {np.dtype(dtype)}, got {np.dtype(pyr.dtype)}"
        )
