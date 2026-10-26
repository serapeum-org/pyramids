"""UGRID connectivity array wrapper.

Provides the Connectivity class that handles UGRID connectivity
arrays with start_index normalization and fill_value masking
for mixed-element meshes.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import cast

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
    something the rest of the package can rely on: afterwards, padding is ``-1`` and nothing
    else is negative.

    What it does, and why each step is needed:

    - **The result is always a signed index dtype** (:data:`numpy.intp`), matching what
      :meth:`Connectivity.from_gdal_array` already produces and what every consumer assumes.
      Preserving an unsigned dtype cannot work: ``-1`` has no representation in it, so
      writing the sentinel either raises or wraps to the fill's own value. xugrid gives the
      same rule its full reason -- "Internally we always use a fill value of -1. This ensures
      we can always index with the fill value as well, since any array will have at least
      size 1." -- i.e. ``-1`` is chosen because it is a *usable* index, not merely a reserved
      one.
    - **A non-integral fill is refused** rather than narrowed. Truncating ``999999.5`` to
      ``999999`` compares against a value that is not in the table, so the padding survives
      while ``fill_value`` claims ``-1`` -- the exact ambiguity this function removes,
      recreated silently and with the audit trail erased.
    - **The refusal keys on whether the fill occurs**, not merely on its value. The UGRID
      examples declare ``_FillValue = 999999`` unconditionally, padded or not, so a mesh with
      a million nodes would otherwise fail to load over a sentinel it never uses.
    - **Padding and every other negative become** ``-1``. A stray negative addresses a node
      from the wrong end exactly as silently as the conventional fill does, which is why
      :attr:`Network1d._real_edges` already excludes all of them rather than just the fill.

    An ambiguous table is refused: when the fill is a legal index *and* present, a padded row
    and a real element are indistinguishable and there is no correct reading to normalise to.
    On a 4-node mesh with ``fill_value=2``, the rows ``[1, 2]`` and ``[2, 3]`` are real
    elements that look exactly like padding -- dropping them silently is how an edge table
    produced a wrong topological order with no exception.

    ``index_count`` is the size of the range **this** table indexes, which is not always the
    node count: a ``face_edge_connectivity`` indexes edges and a ``face_face_connectivity``
    indexes faces. It is ``0`` when the companion table is absent (``edge_node_connectivity``
    is optional in UGRID while ``face_edge_connectivity`` is present), and the range is then
    taken from the table's own largest non-fill index rather than treated as unlimited --
    otherwise an unvalidatable fill would still be rewritten.

    Args:
        connectivity: The connectivity table the caller supplied.
        index_count: The number of elements this table's indices address, or ``0`` when that
            is not known from a companion table.
        label: The attribute name, used in the error so the caller knows which table.
        addresses: What the indices point at -- ``"node"``, ``"edge"`` or ``"face"`` -- so the
            refusal names the right thing. Defaults to ``"node"``.
        topology: What the table belongs to, ``"mesh"`` or ``"network"``, so a
            :class:`Network1d` refusal does not call itself a mesh. Defaults to ``"mesh"``.

    Returns:
        Connectivity: The same object when there is nothing to do -- no ``fill_value``
            attribute at all (a bare array), a ``None`` fill, or a ``-1`` fill with no other
            negative to rewrite, which is what every in-repo reader produces.

            Otherwise a **new** table with ``fill_value=-1`` and a :data:`numpy.intp`
            ``data`` array. The caller's object and its array are left exactly as they were,
            never updated in place, so a caller holding its own reference still sees the fill
            it declared while the mesh holds the normalised copy. Read the table back off the
            mesh rather than from the object you passed in.

    Raises:
        TypeError: ``fill_value`` is neither an integer nor ``None`` (a float, a string,
            ``nan``).
        ValueError: The fill value is a legal index for this table *and* occurs in it, so
            padding and data cannot be told apart.
    """
    declared = getattr(connectivity, "fill_value", None)
    if declared is None:
        # Two real inputs have no fill to normalise, and both are reached by the suite:
        # `Network1d(edge_node_connectivity=np.empty((0, 2)))` passes a bare ndarray, and a
        # `Connectivity(fill_value=None)` declares "this table has no sentinels". The
        # duck-typing is deliberate rather than sloppy -- the annotation understates what
        # callers actually pass -- so neither can be narrowed away.
        return connectivity
    if isinstance(declared, bool):
        # `bool` subclasses `int`, so `operator.index(True)` is 1 and the fill would be read
        # as node 1. `Network1d._validated_outlet` rejects it for the same reason.
        raise TypeError(
            f"{label}.fill_value must be an integer or None, got {declared!r} (bool). A "
            f"boolean is not an index, even though it behaves like one."
        )
    try:
        fill = operator.index(declared)
    except TypeError as error:
        raise TypeError(
            f"{label}.fill_value must be an integer or None, got {declared!r} "
            f"({type(declared).__name__}). A non-integral fill cannot be compared against "
            f"the index table."
        ) from error

    original = np.asarray(connectivity.data)
    is_padding = original == fill
    negative = original < 0
    if fill >= 0 and bool(is_padding.any()):
        # The range this table's indices may legally take. When the companion table is
        # absent (`index_count == 0`) the largest real index in this table is the best
        # available bound -- better than rewriting against a fill nothing could validate.
        limit = index_count
        if not limit:
            real = original[~is_padding]
            limit = int(real.max()) + 1 if real.size else 0
        if fill < limit:
            raise ValueError(
                f"{label}.fill_value is {fill}, which is also a valid {addresses} index "
                f"in this {topology} (the table's indices run over [0, {limit})), and it "
                f"occurs in the table -- so a padded slot cannot be told from a real "
                f"{addresses}. Use a negative fill value (-1 is what the built-in readers "
                f"write) or any value of at least {limit}."
            )

    if fill == -1 and not bool((negative & ~is_padding).any()):
        result = connectivity
    else:
        # A signed dtype throughout: `-1` is unrepresentable in an unsigned one, so
        # preserving the caller's dtype would raise or wrap the sentinel onto a real index.
        rewritten = original.astype(np.intp, copy=True)
        rewritten[is_padding | negative] = -1
        result = Connectivity(
            data=rewritten,
            fill_value=-1,
            cf_role=connectivity.cf_role,
            original_start_index=connectivity.original_start_index,
        )
    return result
