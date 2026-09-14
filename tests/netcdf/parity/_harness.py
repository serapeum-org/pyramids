"""Mechanical "same behaviour as xarray" assertions for the NetCDF surface.

`pyramids` ships `to_xarray()`, so for any operation implemented on both sides the acceptance
test for parity can be written once:

```python
assert_parity(from_pyramids(nc, "t"), to_xr(nc, "t"))
```

Five systematic differences make the naive comparison fail, and each is a thing to *assert*
rather than paper over. The whole point of this module is that they are normalised in one
place, visibly, instead of being re-derived (and re-got-wrong) per test:

1. **Y orientation.** pyramids presents every raster north-up; `to_xarray()` reports the
   coordinate in *storage* order. On a y-ascending file the two run opposite ways. The flip is
   decided by the stored latitude coordinate itself — ascending means storage order is the
   reverse of raster order — and never by the container's geotransform, which one repo fixture
   (`cf__5v__1d4-4d1__geog__y-desc.nc`) reports as the placeholder `(0.0, 1.0, 0, 0.0, 0, -1.0)`
   while its latitudes actually run 65.0 down to 63.25.
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
   `unflatten_band_axes()` before comparing.

The gaps come from `read_array(masked=True)` on the pyramids side and from the stored values
on the xarray side. Neither side compares an unpacked value against the declared sentinel:
`no_data_value` is a **stored** value, and on a packed variable an unpacked read carries the
fill scaled along with the data — on `cf__20v__1d3-3d17__y-desc.nc` the 63072 cells holding
`-32767` read back as `0.0864`. That is documented behaviour with two supported routes around
it (`unpack=False` or `masked=True`), not a trap; the harness takes the masked one because it
is a single read and cannot drift from whatever the mask rules become.
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

    Both sides of a comparison are built as a `ParityView` — `to_xr` makes the xarray one and
    `from_pyramids` the pyramids one — so `assert_parity` never has to know which repo an array
    came from. The view is frozen: normalise once, then compare, and use `dataclasses.replace`
    to derive a variant rather than mutating one in place.

    Attributes:
        values: Physical values as `float64`, north-up along the y axis, with every gap as
            `NaN`. Shaped `(*band_dims, rows, cols)`. Always `float64`, whatever the file
            stores, because the comparison is `assert_allclose` on physical units.
        gaps: Boolean mask of the no-data cells, shaped like `values`.
        dims: The dimension names of `values`, spatial axes last.
        coords: The dimension coordinates, y normalised to raster (north-up) order.
        dtype: The dtype the *source* held, before the float64 promotion this view applies.
            This is what a dtype contract is asserted against, not `values.dtype`.

    Examples:
        - A packed int16 file keeps its stored dtype on the view while the values are float64:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import from_pyramids
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__20v__1d3-3d17__y-desc.nc")
          >>> view = from_pyramids(nc, "tcw")
          >>> view.dtype
          dtype('int16')
          >>> view.values.dtype
          dtype('float64')
          >>> int(view.gaps.sum())
          63072

          ```
        - The dims name the axes of `values`, and `coords` carries one entry per dim:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import from_pyramids
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> view = from_pyramids(nc, "temperature")
          >>> view.dims
          ('time', 'pressure_level', 'lat', 'lon')
          >>> view.values.shape
          (4, 3, 5, 6)
          >>> view.coords["pressure_level"]
          array([1000.,  850.,  500.])

          ```
        - A view is immutable, so a derived one is made with `dataclasses.replace`:

          ```python
          >>> import dataclasses
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import from_pyramids
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> view = from_pyramids(nc, "temperature")
          >>> first_step = dataclasses.replace(view, values=view.values[:1], gaps=view.gaps[:1])
          >>> first_step.values.shape
          (1, 3, 5, 6)
          >>> first_step.dtype
          dtype('float64')

          ```

    See Also:
        to_xr: Builds the xarray side of a comparison.
        from_pyramids: Builds the pyramids side of a comparison.
        assert_parity: Compares two views.
    """

    values: np.ndarray
    gaps: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    dtype: np.dtype


def y_dimension(nc: NetCDF) -> str | None:
    """The name of a container's y dimension, or `None` when it declares none.

    The lookup is by name only, in the fixed order of `Y_DIMENSION_NAMES`; a container whose
    vertical axis is called something else reads as having no y dimension, which the harness
    treats as "nothing to flip" rather than as an error.

    Args:
        nc: The container to inspect.

    Returns:
        The first of `Y_DIMENSION_NAMES` the container declares, or `None`.

    Examples:
        - A CF fixture names it `lat`:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import y_dimension
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> y_dimension(nc)
          'lat'

          ```
        - A projected fixture names it plainly `y`, and the same lookup finds it:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import y_dimension
          >>> nc = NetCDF.read_file("tests/data/netcdf/coards__4v__1d2-2d2__scaleoffset__y-asc.nc")
          >>> y_dimension(nc)
          'y'
          >>> sorted(nc.dimension_sizes)
          ['x', 'y']

          ```
        - A one-dimensional store has no horizontal axis at all:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import y_dimension
          >>> nc = NetCDF.read_file("tests/data/netcdf/none__1v__1d1.nc")
          >>> print(y_dimension(nc))
          None

          ```

    See Also:
        stored_y_ascends: Uses this name to decide whether storage order reverses the raster.
    """
    sizes = nc.dimension_sizes
    return next((name for name in Y_DIMENSION_NAMES if name in sizes), None)


def stored_y_ascends(nc: NetCDF) -> bool:
    """Whether the y coordinate is stored south-to-north, so storage order reverses the raster.

    The answer comes from the stored coordinate values, not from the geotransform: only the
    first and last values are compared, so a coordinate that is monotonic (as a dimension
    coordinate is) settles it in one read.

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
          >>> import numpy as np
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import stored_y_ascends
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> np.asarray(nc.get_dimension_values("lat"))
          array([40., 41., 42., 43., 44.])
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
        - A container with no y dimension reports `False`, because there is nothing to flip:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import stored_y_ascends, y_dimension
          >>> nc = NetCDF.read_file("tests/data/netcdf/none__1v__1d1.nc")
          >>> print(y_dimension(nc))
          None
          >>> stored_y_ascends(nc)
          False

          ```

    See Also:
        y_dimension: Finds the dimension this reads.
    """
    name = y_dimension(nc)
    ascends = False
    if name is not None:
        values = np.asarray(nc.get_dimension_values(name))
        if values.size > 1:
            ascends = bool(values[0] < values[-1])
    return ascends


def _packing(variable: Any) -> tuple[float, float]:
    """The variable's CF `scale_factor` / `add_offset`, defaulted to the identity.

    Both attributes are read from the variable's first band — pyramids reports one entry per
    band and CF packing is a variable-level attribute, so the bands all carry the same pair —
    and an absent attribute becomes the identity, so an unpacked variable can go through the
    same arithmetic as a packed one.

    Args:
        variable: The variable handle, from `NetCDF.get_variable()`, whose `scale` and `offset`
            sequences are read at index 0.

    Returns:
        The `(scale, offset)` pair as Python floats, `(1.0, 0.0)` when the variable declares
        neither, so that `stored * scale + offset` is physical units either way.

    Examples:
        - A packed float32 variable reports the pair the file declares:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import _packing
          >>> nc = NetCDF.read_file("tests/data/netcdf/coards__4v__1d2-2d2__scaleoffset__y-asc.nc")
          >>> _packing(nc.get_variable("z"))
          (0.01, 1.5)
          >>> _packing(nc.get_variable("q"))
          (0.1, 2.5)

          ```
        - An unpacked variable defaults to the identity, so the same arithmetic is a no-op:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import _packing
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> _packing(nc.get_variable("temperature"))
          (1.0, 0.0)

          ```
        - The pair turns a stored int16 count into its physical value:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import _packing
          >>> nc = NetCDF.read_file("tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc")
          >>> scale, offset = _packing(nc.get_variable("rhum"))
          >>> round(scale, 4), round(offset, 2)
          (0.01, 302.66)
          >>> round(-30266 * scale + offset, 2)
          0.0

          ```
    """
    scale = variable.scale[0]
    offset = variable.offset[0]
    return (
        1.0 if scale is None else float(scale),
        0.0 if offset is None else float(offset),
    )


def _gap_mask(stored: np.ndarray, sentinel: Any) -> np.ndarray:
    """Where the stored values hold the declared sentinel.

    The comparison is against the *stored* values on purpose: `no_data_value` is a stored
    number, so on a packed variable an unpacked array no longer holds it anywhere. A NaN
    sentinel cannot be found by equality and is matched with `np.isnan` instead.

    Args:
        stored: The values as the file holds them, before any unpacking.
        sentinel: The declared no-data value, or `None`.

    Returns:
        A boolean array shaped like `stored`; all `False` when nothing is declared.

    Examples:
        - A numeric sentinel is matched by equality:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import _gap_mask
          >>> _gap_mask(np.array([[1, -9999], [3, 4]]), -9999)
          array([[False,  True],
                 [False, False]])

          ```
        - A NaN sentinel is matched by `np.isnan`, since `nan == nan` is never true:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import _gap_mask
          >>> _gap_mask(np.array([1.0, np.nan]), float("nan"))
          array([False,  True])

          ```
        - A variable that declares no fill has no gaps:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import _gap_mask
          >>> _gap_mask(np.array([1.0, 2.0, 3.0]), None)
          array([False, False, False])

          ```
        - On a real packed file the mask is taken from the unpacked-nothing read:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import _gap_mask
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__20v__1d3-3d17__y-desc.nc")
          >>> stored = nc.read_array(variable="tcw", unpack=False)
          >>> nc.get_variable("tcw").no_data_value[0]
          -32767.0
          >>> int(_gap_mask(stored, -32767.0).sum())
          63072

          ```
    """
    if sentinel is None:
        mask = np.zeros(stored.shape, dtype=bool)
    elif isinstance(sentinel, float) and np.isnan(sentinel):
        mask = np.isnan(stored)
    else:
        mask = stored == sentinel
    return mask


def to_xr(nc: NetCDF, variable: str) -> ParityView:
    """The xarray view of `variable`, normalised onto pyramids' raster orientation.

    Applies normalisations 1, 2 and 4: the y axis is flipped to north-up when the file stores it
    ascending, the stored values are unpacked into physical units, and the sentinel cells become
    `NaN`.

    Args:
        nc: The container to export.
        variable: The variable to take from the exported cube.

    Returns:
        ParityView: The xarray side, ready to compare. Its `dims` are the exported array's own,
        its `dtype` the stored one, and its `values` always `float64`.

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
        - On a y-ascending file the coordinate comes back reversed, in raster order:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import to_xr
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> np.asarray(nc.get_dimension_values("lat"))
          array([40., 41., 42., 43., 44.])
          >>> to_xr(nc, "temperature").coords["lat"]
          array([44., 43., 42., 41., 40.])

          ```
        - A name the export does not carry is reported with the names it does:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import to_xr
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> to_xr(nc, "humidity")
          Traceback (most recent call last):
              ...
          KeyError: "'humidity' is not in the exported cube; it holds ['temperature']"

          ```

    See Also:
        from_pyramids: Builds the other side of the comparison.
        assert_parity: Compares the two.
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
    """The pyramids view of `variable`, reshaped onto the dimensions xarray keeps.

    Applies normalisations 2 and 5: the flat GDAL bands axis is rebuilt into the variable's own
    band dimensions, and the sentinel cells become `NaN` — taken from the *stored* values,
    because unpacking scales the fill value along with the data.

    No y flip happens here: `read_array()` is already north-up. The gaps always come from a
    `read_array(masked=True)` of the file, `values` or no `values`, so a passed-in result is
    masked by where the *source* has gaps rather than by anything the caller computed.

    Args:
        nc: The container to read.
        variable: The variable to read.
        values: An already-computed result to normalise instead of reading, in raster order
            and physical units. Defaults to `read_array(variable=variable)`.

    Returns:
        ParityView: The pyramids side, ready to compare. `values` is `float64` with the gaps as
        `NaN`; `dtype` reports the dtype the file stores.

    Raises:
        ValueError: `variable` is not a variable of this container.

    Examples:
        - Reading a two-band-dimension variable rebuilds both axes:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import from_pyramids
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> view = from_pyramids(nc, "temperature")
          >>> view.values.shape
          (4, 3, 5, 6)
          >>> view.dims
          ('time', 'pressure_level', 'lat', 'lon')

          ```
        - A packed file keeps its stored dtype on the view and its gaps as `NaN`:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import from_pyramids
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__20v__1d3-3d17__y-desc.nc")
          >>> view = from_pyramids(nc, "tcw")
          >>> view.dtype
          dtype('int16')
          >>> int(view.gaps.sum())
          63072
          >>> bool(np.isnan(view.values[view.gaps]).all())
          True

          ```
        - A precomputed result is normalised instead of read, and the fill an unpacked read
          scaled along with the data is still recognised as a gap:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import from_pyramids
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__20v__1d3-3d17__y-desc.nc")
          >>> unpacked = nc.read_array(variable="tcw")
          >>> view = from_pyramids(nc, "tcw", values=unpacked)
          >>> float(round(unpacked[view.gaps][0], 4))
          0.0864
          >>> bool(np.isnan(view.values[view.gaps]).all())
          True

          ```
        - An unknown variable is refused by the container, which lists what it holds:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import from_pyramids
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> from_pyramids(nc, "humidity")
          Traceback (most recent call last):
              ...
          ValueError: humidity is not a valid variable name in ['temperature']

          ```

    See Also:
        to_xr: Builds the other side of the comparison.
        assert_parity: Compares the two.
    """
    handle = nc.get_variable(variable)
    # `masked=True` is the supported way to read physical values and keep the gaps
    # identifiable: it unpacks and masks in one read, so the mask cannot drift from the
    # packing the way a hand-rolled comparison against `no_data_value` would.
    masked = nc.read_array(variable=variable, masked=True)
    gaps = np.ma.getmaskarray(np.ma.asarray(masked))

    physical = (
        np.ma.filled(np.ma.asarray(masked).astype("float64"), np.nan)
        if values is None
        else np.where(gaps, np.nan, np.asarray(values, dtype="float64"))
    )

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
    return ParityView(physical, gaps, dims, coords, np.dtype(handle.dtype[0]))


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
    contract when the caller states one. Each failure names both sides, so the message says
    which normalisation did not hold rather than only that two arrays differ.

    Args:
        pyr: The pyramids side, from `from_pyramids`.
        xr_obj: The xarray side, from `to_xr`.
        rtol: Relative tolerance for the value comparison. NaN matches NaN, so the gaps are
            compared as positions by the mask check and skipped here.
        dtype: The dtype the pyramids side must hold, when the operation contracts for one.
            Checked against `ParityView.dtype` — the dtype the file stores — not against the
            float64 `values`. `None` skips the check: a reduction promotes to float64 whatever
            went in, and the promotion is the contract rather than a violation of it.

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
        - A packed variable holds its stored dtype through the comparison, and `dtype=None`
          asks for no dtype claim at all:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import assert_parity, from_pyramids, to_xr
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__20v__1d3-3d17__y-desc.nc")
          >>> pyr, xr_view = from_pyramids(nc, "tcw"), to_xr(nc, "tcw")
          >>> assert_parity(pyr, xr_view, dtype="int16")
          >>> assert_parity(pyr, xr_view)

          ```
        - Stating the wrong dtype fails last, after the values have already agreed:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import assert_parity, from_pyramids, to_xr
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__20v__1d3-3d17__y-desc.nc")
          >>> try:
          ...     assert_parity(from_pyramids(nc, "tcw"), to_xr(nc, "tcw"), dtype="float32")
          ... except AssertionError as error:
          ...     print(str(error).splitlines()[0])
          dtype contract broken: expected float32, got int16

          ```
        - A shape disagreement is reported first, with the dims of both sides:

          ```python
          >>> import dataclasses
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import assert_parity, from_pyramids, to_xr
          >>> nc = NetCDF.read_file("tests/data/netcdf/coards__4v__1d2-2d2__scaleoffset__y-asc.nc")
          >>> pyr = from_pyramids(nc, "z")
          >>> top_row = dataclasses.replace(pyr, values=pyr.values[:1], gaps=pyr.gaps[:1])
          >>> try:
          ...     assert_parity(top_row, to_xr(nc, "z"))
          ... except AssertionError as error:
          ...     print(str(error).splitlines()[0])
          shape differs: pyramids (1, 21) vs xarray (21, 21) (dims ('y', 'x') vs ('y', 'x'))

          ```

    See Also:
        from_pyramids: Builds the `pyr` side.
        to_xr: Builds the `xr_obj` side.
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
