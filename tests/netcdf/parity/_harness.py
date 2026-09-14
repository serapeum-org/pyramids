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
   decided by the stored y coordinate — ascending means storage order is the reverse of raster
   order — and never by the geotransform. Not because the geotransform is unreliable, but
   because it describes the view rather than the file: pyramids *always* presents north-up, so
   its y step is always negative and it cannot distinguish a file stored south-to-north from
   one stored north-to-south. The stored coordinate is the only thing that carries that. On
   several containers here the geotransform is a bare placeholder as well, but that is a
   separate defect and not the reason.

   Where the coordinate cannot carry it either — it is absent, a synthesised row index, or not
   monotonic — the harness raises rather than defaults. See `flip_needed`; the stores
   that trip each case are catalogued in `_catalogue.UNSUPPORTED`.
2. **No-data vs NaN.** pyramids declares a sentinel; xarray has only NaN. Both sides are masked
   to NaN at the sentinel and the two gap masks are asserted equal, so a disagreement about
   *where* the gaps are cannot hide behind a value comparison that skips them.
3. **dtype.** A reduction returns float64 whatever went in, so values are compared with
   `assert_allclose` and the dtype contract is asserted separately, by name, when the caller
   states one.
4. **CF packing.** `read_array()` unpacks by default; `to_xarray()` hands over the *stored*
   values. The xarray side is unpacked here with the variable's own `scale_factor` /
   `add_offset` so both sides are in physical units.
5. **Band order.** GDAL flattens every non-spatial dimension row-major into one bands axis —
   and squeezes it away entirely when it is length 1 — while xarray keeps them separate. The
   pyramids array is reshaped back through `_with_band_axes` before comparing.

The gaps come from `read_array(masked=True)` on the pyramids side and from the stored values
on the xarray side. Neither side compares an unpacked value against the declared sentinel:
`no_data_value` is a **stored** value, and on a packed variable an unpacked read carries the
fill scaled along with the data — on `cf__20v__1d3-3d17__y-desc.nc` the 63072 cells holding
`-32767` read back as `0.0864`. That is documented behaviour with two supported routes around
it (`unpack=False` or `masked=True`), not a trap; the harness takes the masked one because it
is a single read and cannot drift from whatever the mask rules become.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyramids.netcdf.netcdf import NetCDF

#: `get_variable` renames the axis it subsets to `subset_<name>_<start>_<step>_<count>`.
_SUBSET_RENAME = re.compile(r"^subset_(.+)_-?\d+_-?\d+_-?\d+$")

#: Dimension names pyramids' raster view treats as the y axis, in the order they are looked for.
Y_DIMENSION_NAMES = ("lat", "latitude", "y")


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
        coords: The dimension coordinates as each side presents them, y normalised to raster
            (north-up) order.
        decoded_coords: The same coordinates as instants, for the axes that carry CF time
            units. `to_xarray()` decodes a time axis and `get_dimension_values` does not, so
            the two sides present it differently; this is what makes them comparable, and the
            two decodings are independent, which makes the comparison a real cross-check.
        dtype: The dtype a `dtype=` contract is checked against: the **result's** own dtype
            when the view carries an operation's output, and the stored dtype otherwise. A
            reduction promotes to float64 whatever the file holds, and that promotion is the
            contract worth asserting.
        source_dtype: The dtype the file stores, kept alongside `dtype` so an operation's
            result and the dtype it came from are both available. `None` on a view that was
            never given one, which is every `to_xr` view: there `dtype` already is the stored
            dtype. The `dtype=` contract is asserted against `dtype`, never against this or
            against the float64 `values.dtype`.

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
    decoded_coords: dict[str, np.ndarray] = field(default_factory=dict)
    source_dtype: np.dtype | None = None


class ParityUnsupported(RuntimeError):
    """The harness cannot state a parity relationship for this store.

    Raised rather than answered with a guess. Every alternative to raising here is a silent
    wrong answer that a downstream parity test would report as agreement — the failure mode
    this whole module exists to prevent.

    Examples:
        - A store whose y axis has no coordinate variable is refused, and the message names
          the axis and the reason:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import ParityUnsupported, from_pyramids
          >>> nc = NetCDF.read_file("tests/data/netcdf/none__4v__1d1-2d2-3d1__curv.nc")
          >>> try:
          ...     from_pyramids(nc, "Tair")
          ... except ParityUnsupported as error:
          ...     print(str(error)[:39])
          the 'y' axis has no coordinate variable

          ```
        - The same refusal covers a result the harness cannot describe, not only a store it
          cannot orient:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import ParityUnsupported, from_pyramids
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> try:
          ...     from_pyramids(nc, "temperature", values=np.zeros((2, 3, 5, 6)))
          ... except ParityUnsupported as error:
          ...     print(str(error)[:44])
          the result for 'temperature' is (2, 3, 5, 6)

          ```

    See Also:
        flip_needed: Raises this for a y axis that cannot decide the orientation.
        from_pyramids: Raises it for a result the harness cannot label or describe.
    """


def _variable_dims(nc: NetCDF, handle: Any) -> tuple[str, ...]:
    """The dimension names of `handle`'s array, spatial axes last.

    Taken from the **variable**, not the container. A container declares every dimension any of
    its variables uses, so deriving the spatial axes as "whatever the container declares that is
    not a band dimension" yields names the array does not have the moment its variables differ:
    `cf__12v__1d4-2d5-3d2-4d1__y-asc.nc::area` is 2-D and would collect five names.

    A variable subset renames its y dimension (`lat` becomes `subset_lat_4_-1_5`), so that one
    slot is resolved back through the container's own name.

    Args:
        nc: The container the variable came from.
        handle: The variable subset.

    Returns:
        The dimension names, `(*band_dims, y, x)` for a variable with a raster plane.

    Raises:
        ParityUnsupported: The variable reports no dimension names to work from.

    Examples:
        - A four-dimensional variable reports its own axes, band dimensions first:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import _variable_dims
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> _variable_dims(nc, nc.get_variable("temperature"))
          ('time', 'pressure_level', 'lat', 'lon')

          ```
        - A 2-D variable reports two names even though its container declares five, which is
          what deriving the axes from the container would have got wrong:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import _variable_dims
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__12v__1d4-2d5-3d2-4d1__y-asc.nc")
          >>> sorted(nc.dimension_sizes)
          ['bnds', 'lat', 'lon', 'plev', 'time']
          >>> _variable_dims(nc, nc.get_variable("area"))
          ('lat', 'lon')

          ```
        - The renamed y axis of a variable subset is resolved back to the container's
          spelling:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import _variable_dims
          >>> path = "tests/data/netcdf/coards__4v__1d2-2d2__scaleoffset__y-asc.nc"
          >>> nc = NetCDF.read_file(path)
          >>> nc.get_variable("z").dimension_names
          ['subset_y_20_-1_21', 'x']
          >>> _variable_dims(nc, nc.get_variable("z"))
          ('y', 'x')

          ```

    See Also:
        _unrenamed: Resolves one renamed name.
    """
    declared = handle.dimension_names
    if not declared:
        raise ParityUnsupported(
            "the variable reports no dimension names, so its axes cannot be labelled"
        )
    known = set(nc.dimension_sizes)
    return tuple(_unrenamed(name, known) for name in declared)


def _unrenamed(name: str, known: set[str]) -> str:
    """The container's spelling of a dimension a variable subset renamed.

    `get_variable` reports its y axis as `subset_<name>_<start>_<step>_<count>` —
    `subset_y_20_-1_21` on the scale/offset store, `subset_lines_479_-1_480` on
    `none__5v__1d2-2d2-3d1__curv.nc` — because the subset carries the window it was cut
    with. Both sides have
    to agree on one spelling, and the container's is the one
    `to_xarray` uses. Resolved by pattern rather than by looking only at the y slot, so a store
    whose y is spelled something the harness does not recognise is still labelled correctly.

    Args:
        name: The dimension name as the variable reports it.
        known: The dimension names the container declares.

    Returns:
        The container's name when `name` is a recognisable rename of one, else `name` itself.

    Examples:
        - A renamed axis resolves back:

          ```python
          >>> from tests.netcdf.parity._harness import _unrenamed
          >>> _unrenamed("subset_lat_4_-1_5", {"lat", "lon"})
          'lat'

          ```
        - A name the container does not declare is left alone:

          ```python
          >>> from tests.netcdf.parity._harness import _unrenamed
          >>> _unrenamed("subset_zz_1_2_3", {"lat", "lon"})
          'subset_zz_1_2_3'

          ```
    """
    resolved = name
    match = _SUBSET_RENAME.match(name)
    if match and match.group(1) in known:
        resolved = match.group(1)
    return resolved


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

    The answer comes from the stored coordinate values, not from the geotransform, which
    describes pyramids' north-up view rather than the file's storage order. Only the first and
    last values are compared, so a coordinate that is monotonic (as a dimension coordinate is)
    settles it in one read.

    Kept for the container-level question the tests ask; `flip_needed` is what the
    normalisation uses, because it decides from the variable's own y axis and refuses the cases
    this one cannot answer.

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


def _is_index_axis(values: np.ndarray) -> bool:
    """Whether a coordinate is the bare index GDAL synthesises for an axis that has none.

    `cf__9v__1d7-2d2__geos__y-desc.nc` declares a `y` of `int16` `[0, 1, ... 499]`: not a
    position on the earth, just the row number. It ascends, so a rule that reads "ascending
    means the storage order is reversed" flips a raster that is already north-up — verified,
    the two sides agree only when the flip is undone.

    Args:
        values: The stored coordinate.

    Returns:
        `True` when the values are exactly `0, 1, 2, …` in an integer dtype.

    Examples:
        - A synthesised index is recognised:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import _is_index_axis
          >>> _is_index_axis(np.arange(5, dtype="int16"))
          True

          ```
        - Real latitudes are not, even when they happen to be whole numbers:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import _is_index_axis
          >>> _is_index_axis(np.array([40.0, 41.0, 42.0]))
          False

          ```
    """
    return bool(
        np.issubdtype(values.dtype, np.integer)
        and values.size
        and np.array_equal(values, np.arange(values.size))
    )


def flip_needed(nc: NetCDF, y_name: str | None) -> bool:
    """Whether the xarray side has to be flipped to reach pyramids' north-up raster order.

    Decided from the y coordinate, and only when that coordinate can carry the decision.
    Everything else raises: a silent `False` here is a wrong reference array that every
    downstream parity test would report as agreement.

    Args:
        nc: The container to read the coordinate from.
        y_name: The variable's y dimension, already resolved to the container's spelling, or
            `None` for a variable with no raster plane.

    Returns:
        `True` when the file stores its rows south-to-north, so storage order reverses the
        raster. `False` when it is already north-up, or when there is no y axis to flip.

    Raises:
        ParityUnsupported: The y axis exists but cannot decide — it has no coordinate
            variable, its coordinate is a synthesised index, or it is not monotonic.

    Examples:
        - A real ascending axis is flipped:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import flip_needed
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> flip_needed(nc, "lat")
          True

          ```
        - A store whose y is a bare row index is refused rather than guessed at:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import ParityUnsupported, flip_needed
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__9v__1d7-2d2__geos__y-desc.nc")
          >>> try:
          ...     flip_needed(nc, "y")
          ... except ParityUnsupported as error:
          ...     print(str(error)[:39])
          the 'y' axis is a synthesised row index

          ```
    """
    needed = False
    if y_name is not None:
        raw = nc.get_dimension_values(y_name)
        if raw is None:
            raise ParityUnsupported(
                f"the {y_name!r} axis has no coordinate variable, so which way the file "
                "stores its rows cannot be read; this store is out of scope for parity"
            )
        values = np.asarray(raw)
        if values.size > 1:
            if not np.issubdtype(values.dtype, np.number):
                # `np.diff` has no loop for a string dtype, so the monotonic check below
                # raised `UFuncTypeError` — neither a refusal a caller can catch nor an
                # answer. Two variables in the repo reach this with a `<U19` coordinate.
                raise ParityUnsupported(
                    f"the {y_name!r} axis has a {values.dtype} coordinate, which cannot say "
                    "which way the rows are stored"
                )
            if _is_index_axis(values):
                raise ParityUnsupported(
                    f"the {y_name!r} axis is a synthesised row index, not a position, so it "
                    "cannot say which way the rows are stored"
                )
            ascending = bool(values[0] < values[-1])
            ordered = (
                np.all(np.diff(values) > 0)
                if ascending
                else np.all(np.diff(values) < 0)
            )
            if not ordered:
                raise ParityUnsupported(
                    f"the {y_name!r} coordinate is not monotonic, so the storage order is not "
                    "a simple reversal of the raster order"
                )
            needed = ascending
    return needed


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
    elif _is_nan(sentinel):
        # `isinstance(sentinel, float)` misses a `np.float32` NaN, which `== nan` then reports
        # as "no gaps anywhere" — a silently unmasked raster rather than an error.
        mask = np.isnan(stored)
    else:
        mask = stored == sentinel
    return mask


def _is_nan(sentinel: Any) -> bool:
    """Whether a declared sentinel is NaN, for any float type numpy or Python can hold.

    Args:
        sentinel: The declared no-data value.

    Returns:
        `True` when it is a floating NaN, `False` for every other value and dtype.

    Examples:
        - A numpy float32 NaN counts, where an `isinstance(..., float)` test would miss it:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import _is_nan
          >>> _is_nan(np.float32("nan")), _is_nan(float("nan"))
          (True, True)

          ```
        - An ordinary sentinel does not:

          ```python
          >>> from tests.netcdf.parity._harness import _is_nan
          >>> _is_nan(-9999.0), _is_nan(-32767)
          (False, False)

          ```
    """
    as_array = np.asarray(sentinel)
    return bool(np.issubdtype(as_array.dtype, np.floating) and np.isnan(as_array))


def _decoded_time(nc: NetCDF, name: str) -> np.ndarray | None:
    """The instants of a CF time dimension, decoded by pyramids' own reader.

    `get_time_variable` answers `None` for an axis that is not a time axis, which is the
    discriminator used here. It is a different code path from the one `to_xarray` decodes
    with, so comparing the two is a genuine cross-check rather than a tautology.

    Args:
        nc: The container to read.
        name: The dimension to decode.

    Returns:
        The instants as `datetime64[ns]`, or `None` when the axis is not a decodable time axis.

    Examples:
        - A CF time axis comes back as instants:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import _decoded_time
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> [str(when) for when in _decoded_time(nc, "time")[:2]]
          ['2024-01-01T00:00:00.000000000', '2024-01-01T06:00:00.000000000']

          ```
        - A pre-1582 origin decodes here even though `to_xarray` exports it as offsets, which
          is why the two sides are compared only where both produced instants:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import _decoded_time
          >>> nc = NetCDF.read_file("tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc")
          >>> str(_decoded_time(nc, "time")[0])
          '2003-01-01T00:00:00.000000000'

          ```
        - A non-temporal axis answers `None`, which is the discriminator:

          ```python
          >>> from pyramids.netcdf.netcdf import NetCDF
          >>> from tests.netcdf.parity._harness import _decoded_time
          >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
          >>> print(_decoded_time(nc, "lat"))
          None

          ```

    See Also:
        to_xr: Compares these instants against `to_xarray()`'s own decoding.
    """
    decoded: np.ndarray | None = None
    try:
        # Microseconds, not whole seconds: truncating here and then demanding exact equality
        # made any sub-second axis fail spuriously — the stored offsets agree, only the
        # harness's own rounding differed.
        stamps = nc.get_time_variable(name, "%Y-%m-%d %H:%M:%S.%f")
    except (ValueError, TypeError, KeyError):
        stamps = None
    if stamps:
        decoded = np.asarray(stamps, dtype="datetime64[us]").astype("datetime64[ns]")
    return decoded


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
        ParityUnsupported: The variable reports no dimension names, or its y axis cannot say
            which way the file stores its rows. Decided unconditionally, so a store `to_xr`
            answers for is one `from_pyramids` answers for too.

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
    # Both exports are needed and neither is avoidable: the decoded one carries the values
    # and the instants, the undecoded one the stored coordinates the two sides compare on.
    scale, offset = _packing(handle)

    stored = np.asarray(array.values)
    gaps = _gap_mask(stored, handle.no_data_value[0])
    values = np.where(gaps, np.nan, stored.astype("float64") * scale + offset)

    dims = _variable_dims(nc, handle)
    name = dims[-2] if len(dims) >= 2 else None
    # The stored coordinates come from a second export with the decoding turned off, so the
    # two sides can always be compared like with like: `get_dimension_values` reports stored
    # offsets, and `to_xarray()` decodes a CF time axis to `datetime64[ns]`. The decoded form
    # is kept alongside and compared only where *both* sides produced one -- they do not
    # always, and the asymmetry is deliberate rather than a defect: `to_xarray` declines an
    # axis it cannot write back (a pre-1582 origin decodes to `cftime` objects GDAL has no
    # band type for) while `get_time_variable` decodes it happily, so
    # `coards__5v__1d4-4d1__y-desc.nc` has instants on the pyramids side and none on the
    # xarray side.
    as_stored = nc.to_xarray(decode_times=False)
    coords = {
        key: np.asarray(as_stored.coords[key].values)
        for key in array.dims
        if key in as_stored.coords
    }
    decoded_coords = {
        key: np.asarray(exported.coords[key].values).astype("datetime64[ns]")
        for key in array.dims
        if key in exported.coords
        and np.issubdtype(np.asarray(exported.coords[key].values).dtype, np.datetime64)
    }
    # `flip_needed` is called unconditionally, not only when the name is among the export's
    # dims: it is what refuses an undecidable store, and skipping it here let `to_xr` return a
    # view for a store `from_pyramids` refuses — two sides disagreeing about whether the
    # comparison is even possible.
    flip = flip_needed(nc, name)
    if name in array.dims and flip:
        axis = list(array.dims).index(name)
        values = np.flip(values, axis)
        gaps = np.flip(gaps, axis)
        if name in coords:
            coords[name] = coords[name][::-1]
        if name in decoded_coords:
            decoded_coords[name] = decoded_coords[name][::-1]
    return ParityView(
        values, gaps, tuple(array.dims), coords, stored.dtype, decoded_coords
    )


def _with_band_axes(array: np.ndarray, sizes: tuple[int, ...]) -> np.ndarray:
    """Rebuild the variable's band axes on a `(bands, rows, cols)` read.

    GDAL flattens every non-spatial dimension into one bands axis, and `read_array` **squeezes**
    it away entirely when it is length 1 — so a variable with a single size-1 band dimension
    (`cf__12v__1d4-2d5-3d2-4d1__y-asc.nc::pr`) reads back 2-D where `to_xarray` reports
    `('time', 'lat', 'lon')`. That is not an exotic shape: `isel` and a single-label `sel`
    produce exactly it, so T3 and T4 would have had nothing to compare.

    Args:
        array: The read array, flattened or squeezed.
        sizes: The variable's non-spatial dimension sizes, in storage order.

    Returns:
        `array` shaped `(*sizes, rows, cols)`, or unchanged when it already is.

    Examples:
        - A squeezed single-band read regains its axis:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import _with_band_axes
          >>> _with_band_axes(np.zeros((4, 5)), (1,)).shape
          (1, 4, 5)

          ```
        - A flattened two-band-dimension read regains both:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import _with_band_axes
          >>> _with_band_axes(np.zeros((12, 4, 5)), (4, 3)).shape
          (4, 3, 4, 5)

          ```
    """
    rebuilt = array
    if sizes:
        target = (*sizes, *array.shape[-2:])
        if array.shape != target and array.size == int(np.prod(target)):
            rebuilt = array.reshape(target)
    return rebuilt


def _result_view(
    values: Any, gaps: Any, source_gaps: np.ndarray, variable: str
) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    """Normalise an operation's result into the `(values, gaps, dtype)` a view carries.

    The extension point every task after T0 goes through. A result may keep the source's
    shape and mask (`ds * 2`), change the mask (`fillna`, `where`, `interpolate_na`), or
    change the shape outright (`reduce`, `isel`, `coarsen`) — so the source mask is reused
    only when it still fits, and otherwise the caller has to say what the gaps are.

    Args:
        values: The operation's result, in raster order and physical units. Either the flat
            `(bands, rows, cols)` shape `read_array` returns or the rebuilt band axes.
        gaps: The result's no-data mask, or `None` to reuse the source's when it fits.
        source_gaps: The mask the stored array carries.
        variable: The variable's name, for the error message.

    Returns:
        The values as float64 with the gaps as `NaN`, the mask, and the result's own dtype.

    Raises:
        ParityUnsupported: The result's shape does not match the source's and no `gaps=` was
            given, or the supplied mask does not match the result.

    Examples:
        - A same-shape result reuses the source mask, and its gaps become `NaN`:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import _result_view
          >>> source_gaps = np.array([[False, True], [False, False]])
          >>> values, mask, dtype = _result_view(
          ...     np.array([[1.0, 5.0], [3.0, 4.0]]), None, source_gaps, "v"
          ... )
          >>> values
          array([[ 1., nan],
                 [ 3.,  4.]])
          >>> int(mask.sum()), dtype
          (1, dtype('float64'))

          ```
        - A shape-changing result with no `gaps=` is refused, because the source's mask no
          longer describes it:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import ParityUnsupported, _result_view
          >>> source_gaps = np.zeros((2, 2), dtype=bool)
          >>> try:
          ...     _result_view(np.zeros((3, 2)), None, source_gaps, "v")
          ... except ParityUnsupported as error:
          ...     print(str(error)[:52])
          the result for 'v' is (3, 2) where the source is (2,

          ```
        - A `gaps=` mask that does not describe the result is refused too:

          ```python
          >>> import numpy as np
          >>> from tests.netcdf.parity._harness import ParityUnsupported, _result_view
          >>> source_gaps = np.zeros((2, 2), dtype=bool)
          >>> try:
          ...     _result_view(
          ...         np.zeros((2, 2)), np.zeros((3, 2), dtype=bool), source_gaps, "v"
          ...     )
          ... except ParityUnsupported as error:
          ...     print(str(error)[:49])
          the `gaps=` mask for 'v' is (3, 2) but the result

          ```

    See Also:
        from_pyramids: The only caller, which passes the source mask it read.
    """
    array = np.asarray(values)
    flat = (int(np.prod(source_gaps.shape[:-2])), *source_gaps.shape[-2:])
    if array.shape == flat and array.shape != source_gaps.shape:
        # A caller holding a plain `read_array` result has the flat `(bands, rows, cols)`
        # shape GDAL returns, not the rebuilt band axes. *Only* that shape is reinterpreted:
        # matching on "same size, same trailing two axes" also accepted a result whose band
        # axes were transposed — `(3, 4, …)` where the source is `(4, 3, …)` — and silently
        # reshaped it into agreement, routing around the dimension-name check that exists to
        # catch exactly that.
        array = array.reshape(source_gaps.shape)
    if gaps is not None:
        mask = np.asarray(gaps, dtype=bool)
        if mask.shape != array.shape:
            raise ParityUnsupported(
                f"the `gaps=` mask for {variable!r} is {mask.shape} but the result is "
                f"{array.shape}; they have to describe the same array"
            )
    elif array.shape == source_gaps.shape:
        mask = source_gaps
    else:
        raise ParityUnsupported(
            f"the result for {variable!r} is {array.shape} where the source is "
            f"{source_gaps.shape}, so the source's no-data mask does not describe it; pass "
            "`gaps=` (and `dims_override=` for the new axes) for a shape-changing operation"
        )
    return np.where(mask, np.nan, array.astype("float64")), mask, array.dtype


def from_pyramids(
    nc: NetCDF,
    variable: str,
    values: Any = None,
    *,
    gaps: Any = None,
    dims_override: Sequence[str] | None = None,
    coords_override: dict[str, Any] | None = None,
) -> ParityView:
    """The pyramids view of `variable`, reshaped onto the dimensions xarray keeps.

    Applies normalisations 2 and 5: the flat GDAL bands axis is rebuilt into the variable's own
    band dimensions, and the sentinel cells become `NaN` — taken from the *stored* values,
    because unpacking scales the fill value along with the data.

    No y flip happens here: `read_array()` is already north-up. With no `values`, the gaps
    come from a `read_array(masked=True)` of the file. With one,
    they come from `gaps=` when given and from the source only when the source's mask still
    describes the result — an operation that changes which cells are gaps (`fillna`, `where`,
    `interpolate_na`) supplies its own, and one that changes the shape must.

    Args:
        nc: The container to read.
        variable: The variable to read.
        values: An already-computed result to normalise instead of reading, in raster order
            and physical units. Defaults to `read_array(variable=variable)`. Either the flat
            `(bands, rows, cols)` shape `read_array` returns or the rebuilt band axes.
        gaps: The result's no-data mask. Defaults to the source's, which is reused only when
            it still describes the result — an operation that changes which cells are gaps
            (`fillna`, `where`, `interpolate_na`) has to supply its own.
        dims_override: The result's dimension names, for an operation that changes the axes.
            Defaults to the variable's own.
        coords_override: The result's coordinates, keyed by dimension name, for an operation
            that subsets or replaces them. Any axis not named here takes the container's,
            which is refused when its length no longer matches the result.

    Returns:
        ParityView: The pyramids side, ready to compare. `values` is `float64` with the gaps as
        `NaN`; `dtype` reports the dtype the file stores.

    Raises:
        ValueError: `variable` is not a variable of this container.
        ParityUnsupported: The result cannot be labelled or described — its shape does not
            match the source's and no `gaps=` was given, the supplied mask does not describe
            it, the dimension names do not match its rank, or a coordinate's length does not
            match its axis. Also when the y axis cannot say which way the file stores its
            rows.

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
    source_gaps = np.ma.getmaskarray(np.ma.asarray(masked))
    source_values = np.ma.filled(np.ma.asarray(masked).astype("float64"), np.nan)

    # `_band_dim_names` is the variable's own non-spatial axes in storage order, which is what
    # GDAL flattened.
    band_dims = tuple(handle._band_dim_names)
    sizes = tuple(handle._band_dim_sizes)
    source_values = _with_band_axes(source_values, sizes)
    source_gaps = _with_band_axes(source_gaps, sizes)

    source_dtype = np.dtype(handle.dtype[0])
    if values is None:
        physical, result_gaps = source_values, source_gaps
        result_dtype = source_dtype
    else:
        physical, result_gaps, result_dtype = _result_view(
            values, gaps, source_gaps, variable
        )

    dims = tuple(dims_override) if dims_override else _variable_dims(nc, handle)
    if len(dims) != physical.ndim:
        raise ParityUnsupported(
            f"{variable!r} resolves to {len(dims)} dimension name(s) {dims} for a "
            f"{physical.ndim}-D array; the harness cannot label its axes. Pass "
            "`dims_override=` when the operation changes the axes."
        )

    # The y axis is the variable's own, found by name in whatever `dims` ends up being.
    # Taking `dims[-2]` positionally was right for a plain read and wrong for every
    # shape-changing one: a zonal mean over `lon` leaves `(time, pressure_level, lat)`, whose
    # `dims[-2]` is `pressure_level`, so the flip was decided from the pressure axis and the
    # real latitudes were left in storage order.
    source_dims = _variable_dims(nc, handle)
    y_name = source_dims[-2] if len(source_dims) >= 2 else None
    flip = flip_needed(nc, y_name)
    supplied = dict(coords_override or {})
    coords: dict[str, np.ndarray] = {}
    decoded_coords: dict[str, np.ndarray] = {}
    for axis, name in enumerate(dims):
        if name in supplied:
            coords[name] = np.asarray(supplied[name])
            continue
        raw_coord = nc.get_dimension_values(name)
        if raw_coord is None:
            # An axis the operation introduced, or one the store declares without a
            # coordinate variable. There is nothing to compare, so it is left out rather
            # than recorded as `array(None)`.
            continue
        stored_coord = np.asarray(raw_coord)
        instants = _decoded_time(nc, name)
        if name == y_name and flip:
            stored_coord = stored_coord[::-1]
            instants = None if instants is None else instants[::-1]
        if stored_coord.shape[:1] != physical.shape[axis : axis + 1]:
            raise ParityUnsupported(
                f"the {name!r} coordinate has {stored_coord.shape[0]} value(s) but the "
                f"result's matching axis is {physical.shape[axis]} long; a subsetting "
                "operation has to pass `coords_override=` with the coordinates it kept"
            )
        coords[name] = stored_coord
        if instants is not None:
            decoded_coords[name] = instants
    return ParityView(
        physical, result_gaps, dims, coords, result_dtype, decoded_coords, source_dtype
    )


def assert_parity(
    pyr: ParityView,
    xr_obj: ParityView,
    *,
    rtol: float = 1e-9,
    atol: float = 0.0,
    dtype: Any = None,
    coords: bool = True,
) -> None:
    """Assert that a pyramids result and the xarray view agree.

    Checks everything a parity claim rests on, in the order that fails most informatively: the
    dimension names, the shape, the coordinates, then *where* the gaps are, then the values,
    then the dtype contract when the caller states one. Each failure names both sides, so the
    message says which normalisation did not hold rather than only that two arrays differ.

    The names and coordinates are checked because the values alone cannot see a mislabelled or
    transposed result: `coards__4v__1d2-2d2__scaleoffset__y-asc.nc::z` is 21x21 and symmetric
    under transpose, so a swapped axis order compares equal cell for cell.

    Args:
        pyr: The pyramids side, from `from_pyramids`.
        xr_obj: The xarray side, from `to_xr`.
        rtol: Relative tolerance for the value and coordinate comparisons. NaN matches NaN, so
            the gaps are compared as positions by the mask check and skipped here.
        atol: Absolute tolerance, for a result whose values pass through zero — a relative
            tolerance alone can never be met there.
        dtype: The dtype the pyramids side must hold, when the operation contracts for one.
            Checked against `ParityView.dtype`, which is the **result's** dtype when the view
            carries one and the stored dtype otherwise — never against the float64 `values`.
            `None` skips the check, which is right for a plain read: the stored dtype is the
            file's business, not the operation's.
        coords: Whether to compare the coordinates. `False` for an operation that deliberately
            changes them, which must then assert them itself.

    Raises:
        AssertionError: Any of the six checks fails.

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
    assert pyr.dims == xr_obj.dims, (
        f"the dimension names differ: pyramids {pyr.dims} vs xarray {xr_obj.dims}"
    )
    assert pyr.values.shape == xr_obj.values.shape, (
        f"shape differs: pyramids {pyr.values.shape} vs xarray {xr_obj.values.shape}"
        f" (dims {pyr.dims} vs {xr_obj.dims})"
    )
    if coords:
        assert set(pyr.coords) == set(xr_obj.coords), (
            f"the coordinate names differ: pyramids {sorted(pyr.coords)} vs xarray "
            f"{sorted(xr_obj.coords)}"
        )
        for name in pyr.coords:
            # A time axis is compared as instants: `to_xarray` decodes it and
            # `get_dimension_values` does not, so the stored offsets and the datetimes are the
            # same axis in two representations. Both decodings are pyramids' own, by different
            # paths, so agreeing is a real check.
            np.testing.assert_allclose(
                np.asarray(pyr.coords[name], dtype="float64"),
                np.asarray(xr_obj.coords[name], dtype="float64"),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                err_msg=f"the {name!r} coordinate differs between the two sides",
            )
            # Where both sides decoded the axis, the instants have to agree too. The two
            # decodings come from different code paths, so this is a real cross-check.
            if name in pyr.decoded_coords and name in xr_obj.decoded_coords:
                assert np.array_equal(
                    pyr.decoded_coords[name], xr_obj.decoded_coords[name]
                ), (
                    f"the {name!r} instants differ: pyramids "
                    f"{pyr.decoded_coords[name][:3]} vs xarray "
                    f"{xr_obj.decoded_coords[name][:3]}"
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
        atol=atol,
        equal_nan=True,
        err_msg="the values differ after orientation, packing and gap normalisation",
    )
    if dtype is not None:
        assert np.dtype(pyr.dtype) == np.dtype(dtype), (
            f"dtype contract broken: expected {np.dtype(dtype)}, got {np.dtype(pyr.dtype)}"
        )
