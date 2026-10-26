"""UGRID connectivity array wrapper.

Provides the Connectivity class that handles UGRID connectivity
arrays with start_index normalization and fill_value masking
for mixed-element meshes.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from osgeo import gdal

from pyramids.netcdf.utils import _read_attributes


@dataclass
class Connectivity:
    """Wrapper for UGRID connectivity arrays.

    Handles start_index normalization (always 0-indexed internally)
    and fill_value masking for mixed-element meshes. Connectivity
    arrays map between mesh elements (e.g., face-to-node, edge-to-node).

    Attributes:
        data: Index array, always 0-indexed internally. Shape is
            (n_elements, max_nodes_per_element).
        fill_value: Value used to pad rows for elements with fewer
            nodes than max_nodes_per_element. Default: -1.
        cf_role: UGRID cf_role string (e.g., "face_node_connectivity").
        original_start_index: Original start_index from the file (0 or 1).
    """

    data: np.ndarray
    fill_value: int
    cf_role: str
    original_start_index: int

    @classmethod
    def from_gdal_array(cls, md_arr: gdal.MDArray, cf_role: str) -> Connectivity:
        """Read connectivity from a GDAL MDArray, normalizing start_index.

        Reads the raw index array, subtracts start_index if it is 1,
        and replaces the original fill value with -1 for internal use.

        Args:
            md_arr: GDAL MDArray containing the connectivity data.
            cf_role: The cf_role attribute value for this connectivity.

        Returns:
            Connectivity instance with 0-indexed data.
        """
        attrs = _read_attributes(md_arr)
        start_index = int(cast("int", attrs.get("start_index", 0)))
        raw_fill = attrs.get("_FillValue")
        if raw_fill is not None:
            raw_fill = int(cast("int", raw_fill))
        else:
            raw_fill = -999

        raw_data = md_arr.ReadAsArray()
        if raw_data is None:
            raise ValueError(f"Cannot read connectivity array for cf_role='{cf_role}'.")
        # `astype` already returns a fresh, writable array (mutated below), so the extra `.copy()`
        # only duplicated the connectivity array for no benefit (#982).
        data = raw_data.astype(np.intp)

        mask = data == raw_fill

        if start_index != 0:
            data[~mask] -= start_index

        data[mask] = -1

        result = cls(
            data=data,
            fill_value=-1,
            cf_role=cf_role,
            original_start_index=start_index,
        )
        return result

    @property
    def n_elements(self) -> int:
        """Number of elements (rows in the connectivity array)."""
        result = self.data.shape[0]
        return cast("int", result)

    @property
    def max_nodes_per_element(self) -> int:
        """Maximum number of nodes per element (columns)."""
        result = self.data.shape[1] if self.data.ndim > 1 else 1
        return result

    def get_element(self, idx: int) -> np.typing.NDArray:
        """Return valid node indices for a single element.

        Excludes fill values, returning only the actual node indices
        that define this element.

        Args:
            idx: Element index (row in the connectivity array).

        Returns:
            1D array of valid node indices for the element.
        """
        row = self.data[idx]
        if self.data.ndim == 1:
            result = np.atleast_1d(row)
        else:
            result = row[row != self.fill_value]
        return cast("np.typing.NDArray", result)

    def nodes_per_element(self) -> np.typing.NDArray:
        """Return the number of valid nodes per element.

        Returns:
            1D integer array of length n_elements with the count
            of valid (non-fill) nodes for each element.
        """
        if self.data.ndim == 1:
            result = np.ones(self.n_elements, dtype=np.intp)
        else:
            result = np.sum(self.data != self.fill_value, axis=1).astype(np.intp)
        return result

    def is_triangular(self) -> bool:
        """Check if all elements have exactly 3 nodes.

        Returns:
            True if every element has exactly 3 valid nodes.
        """
        counts = self.nodes_per_element()
        result = bool(np.all(counts == 3))
        return result

    def as_masked(self) -> np.ma.MaskedArray:
        """Return a masked array with fill values masked.

        Returns:
            MaskedArray where positions with fill_value are masked.
        """
        result = np.ma.MaskedArray(self.data, mask=(self.data == self.fill_value))
        return result


#: Sentinel distinguishing "no ``fill_value`` attribute" (a bare index array) from
#: ``fill_value=None`` (a table declaring no sentinels). The two need different answers.
_MISSING = object()


def _normalised_connectivity(
    connectivity: Connectivity,
    index_count: int,
    label: str,
    addresses: str = "node",
    topology: str = "mesh",
) -> Connectivity:
    """Return a table whose padding is marked with ``-1``, or refuse an unreadable one.

    Every member that walks a connectivity table decides "is this row padding?" by comparing
    against the stored fill value -- a number pyramids does not control, because it is
    whatever the file or the driver declared. This is the single gate that turns that into
    something the rest of the package can rely on. Afterwards the table is a signed
    :data:`numpy.intp` array in which padding, and nothing else, is negative.

    The rules, and why each is needed:

    - **The result is a signed index dtype.** An unsigned table cannot hold ``-1`` at all, so
      a ``uint8`` table padded with ``255`` -- or one declaring ``fill_value=-1``, which no
      ``uint8`` value can equal -- would otherwise reach every consumer with its padding
      indistinguishable from node 255. xugrid gives the same rule its full reason:
      "Internally we always use a fill value of -1. This ensures we can always index with the
      fill value as well, since any array will have at least size 1."
    - **A non-integral fill is refused** rather than narrowed. Truncating ``999999.5`` to
      ``999999`` compares against a value absent from the table, so the padding survives
      while ``fill_value`` claims ``-1``.
    - **Padding and every other negative become** ``-1``. A stray negative addresses an
      element from the wrong end exactly as silently as the conventional fill does, which is
      why :attr:`Network1d._real_edges` already excludes all of them.
    - **The surviving indices are checked** against ``index_count`` when it is known, so a
      table that disagrees with its coordinate arrays is reported here rather than
      half-rewritten into something that merely looks smaller.

    Two situations are refused rather than guessed at, both because the alternative is a
    silent wrong answer:

    - The fill **is a legal index and occurs** in the table. A padded slot and a real element
      are then indistinguishable; on a 4-node mesh with ``fill_value=2`` the rows ``[1, 2]``
      and ``[2, 3]`` look exactly like padding, and dropping them is how an edge table
      produced a wrong topological order with no exception.
    - The fill **is non-negative and occurs but the range is unknown** (``index_count`` is
      ``0``, which happens because ``edge_node_connectivity`` is optional in UGRID while
      ``face_edge_connectivity`` is not). Deriving a bound from the table's own largest index
      was tried and is worse than refusing: it is too small whenever the mesh has more
      elements than this table references, and the rewrite then discards a legal index
      silently.

    Args:
        connectivity: The connectivity table, or a bare index array, that the caller supplied.
        index_count: The number of elements this table's indices address, or ``0`` when no
            companion table gives it.
        label: The attribute name, used in errors and as the ``cf_role`` of a table built
            from a bare array.
        addresses: What the indices point at -- ``"node"``, ``"edge"`` or ``"face"``.
            Defaults to ``"node"``.
        topology: What the table belongs to, ``"mesh"`` or ``"network"``, so a
            :class:`Network1d` refusal does not call itself a mesh. Defaults to ``"mesh"``.

    Returns:
        Connectivity: The same object only when it is already exactly what the package
            expects -- a :class:`Connectivity` whose ``fill_value`` is ``-1``, whose dtype is
            a signed integer, and which holds no negative outside its padding. Otherwise a
            **new** table with ``fill_value=-1`` and a :data:`numpy.intp` ``data`` array; a
            bare array is wrapped into one, so every member that reads the table works and
            :class:`Mesh2d` accepts the same input :class:`Network1d` does. The caller's
            object and array are never modified, so a caller holding its own reference still
            sees the fill it declared -- read the table back off the mesh.

    Raises:
        TypeError: ``fill_value`` is neither an integer nor ``None`` (a float, a string,
            ``nan``, a bool).
        ValueError: The fill is a legal index for this table and occurs in it; or it is
            non-negative and occurs while the range is unknown; or a surviving index falls
            outside ``[0, index_count)``.
    """
    declared: Any = getattr(connectivity, "fill_value", _MISSING)
    if declared is _MISSING:
        # A bare index array. `Network1d` has always accepted one, and `topology_subset`
        # still defends against it, but every member that reads the table needs a
        # `Connectivity` -- and `Mesh2d` rejected it outright from inside `__init__`. Wrapping
        # it here makes the two constructors agree and the object usable.
        declared, wrapped = -1, True
    else:
        wrapped = False
    if declared is None:
        # `Connectivity(fill_value=None)` declares a table with no sentinels at all.
        return connectivity
    if isinstance(declared, (bool, np.bool_)):
        # `bool` subclasses `int` and `np.bool_` has `__index__`, so either would be read as
        # element 0 or 1. `Network1d._validated_outlet` rejects them for the same reason.
        raise TypeError(
            f"{label}.fill_value must be an integer or None, got {declared!r} "
            f"({type(declared).__name__}). A boolean is not an index, even though it "
            f"behaves like one."
        )
    try:
        fill = operator.index(declared)
    except TypeError as error:
        raise TypeError(
            f"{label}.fill_value must be an integer or None, got {declared!r} "
            f"({type(declared).__name__}). A non-integral fill cannot be compared against "
            f"the index table."
        ) from error

    original = np.asarray(getattr(connectivity, "data", connectivity))
    signed = np.issubdtype(original.dtype, np.signedinteger)
    if not wrapped and fill == -1 and signed:
        # The path every in-repo reader takes, and `Mesh2d` takes it up to five times per
        # construction. A signed table whose fill is already -1 needs the array walked once,
        # to rule out a stray negative, rather than twice for comparisons whose answers are
        # then thrown away.
        if not bool((original < -1).any()):
            return connectivity

    is_padding = original == fill
    occurs = bool(is_padding.any())
    if fill >= 0 and occurs:
        if not index_count:
            raise ValueError(
                f"{label}.fill_value is {fill} and occurs in the table, but nothing says "
                f"how many {addresses}s this {topology} has, so it cannot be told from a "
                f"real {addresses} index. Supply the companion table that defines the "
                f"{addresses} count, or use a negative fill value (-1 is what the built-in "
                f"readers write)."
            )
        if fill < index_count:
            raise ValueError(
                f"{label}.fill_value is {fill}, which is also a valid {addresses} index "
                f"in this {topology} (the table's indices run over [0, {index_count})), and "
                f"it occurs in the table -- so a padded slot cannot be told from a real "
                f"{addresses}. Use a negative fill value (-1 is what the built-in readers "
                f"write) or any value of at least {index_count}."
            )

    negative = original < 0
    # A signed dtype throughout: `-1` is unrepresentable in an unsigned one, so preserving
    # the caller's dtype would raise or wrap the sentinel onto a real index.
    rewritten = original.astype(np.intp, copy=True)
    rewritten[is_padding | negative] = -1
    if index_count:
        beyond = rewritten >= index_count
        if bool(beyond.any()):
            # `argwhere` gives one position per axis, so this works for a 1-D table as well
            # as the usual 2-D one.
            position = tuple(int(value) for value in np.argwhere(beyond)[0])
            raise ValueError(
                f"{label} names {addresses} {int(rewritten[position])} at {position}, "
                f"which is outside [0, {index_count}). The table and the {topology}'s own "
                f"{addresses} count disagree."
            )
    result = Connectivity(
        data=rewritten,
        fill_value=-1,
        cf_role=getattr(connectivity, "cf_role", label),
        original_start_index=getattr(connectivity, "original_start_index", 0),
    )
    return result
