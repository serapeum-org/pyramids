"""UGRID connectivity array wrapper.

Provides the Connectivity class that handles UGRID connectivity
arrays with start_index normalization and fill_value masking
for mixed-element meshes.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        start_index = int(attrs.get("start_index", 0))
        raw_fill = attrs.get("_FillValue")
        if raw_fill is not None:
            raw_fill = int(raw_fill)
        else:
            raw_fill = -999

        raw_data = md_arr.ReadAsArray()
        if raw_data is None:
            raise ValueError(f"Cannot read connectivity array for cf_role='{cf_role}'.")
        data = raw_data.copy().astype(np.intp)

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
        return result

    @property
    def max_nodes_per_element(self) -> int:
        """Maximum number of nodes per element (columns)."""
        result = self.data.shape[1] if self.data.ndim > 1 else 1
        return result

    def get_element(self, idx: int) -> np.ndarray:
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
        return result

    def nodes_per_element(self) -> np.ndarray:
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
