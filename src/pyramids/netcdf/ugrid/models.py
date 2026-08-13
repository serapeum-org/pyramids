"""Data models for UGRID unstructured mesh metadata and variables.

This module defines the core data structures used throughout the
UGRID subpackage: topology metadata, mesh variables (data on the
mesh), and dataset-level metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from osgeo import gdal

from pyramids.netcdf._mdim import open_mdarray

#: Default UGRID mesh-topology variable name used when a file/in-memory mesh has no
#: explicit name (the de-facto convention written by Deltares D-Flow FM and others).
DEFAULT_MESH_NAME = "mesh2d"


@dataclass(frozen=True)
class MeshTopologyInfo:
    """Parsed UGRID topology metadata from a NetCDF file.

    Represents the structure of a single mesh topology variable,
    including references to coordinate variables, connectivity
    arrays, and data variables defined on the mesh.

    Attributes:
        mesh_name: Name of the topology variable (e.g., "mesh2d").
        topology_dimension: Mesh dimensionality (1=network, 2=surface, 3=volume).
        node_x_var: Name of the node x-coordinate variable.
        node_y_var: Name of the node y-coordinate variable.
        face_node_var: Name of the face-node connectivity variable.
        edge_node_var: Name of the edge-node connectivity variable.
        face_edge_var: Name of the face-edge connectivity variable.
        face_face_var: Name of the face-face connectivity variable.
        edge_face_var: Name of the edge-face connectivity variable.
        boundary_node_var: Name of the boundary-node connectivity variable.
        face_x_var: Name of the face center x-coordinate variable.
        face_y_var: Name of the face center y-coordinate variable.
        edge_x_var: Name of the edge center x-coordinate variable.
        edge_y_var: Name of the edge center y-coordinate variable.
        data_variables: Mapping of variable name to mesh location
            (e.g., {"water_level": "face"}).
        crs_wkt: Well-Known Text representation of the CRS, if available.
    """

    mesh_name: str
    topology_dimension: int
    node_x_var: str
    node_y_var: str
    face_node_var: str | None = None
    edge_node_var: str | None = None
    face_edge_var: str | None = None
    face_face_var: str | None = None
    edge_face_var: str | None = None
    boundary_node_var: str | None = None
    face_x_var: str | None = None
    face_y_var: str | None = None
    edge_x_var: str | None = None
    edge_y_var: str | None = None
    data_variables: dict[str, str] = field(default_factory=dict)
    crs_wkt: str | None = None


def _read_time_slab(
    path: str, var_name: str, start: int, stop: int | None
) -> np.typing.NDArray | None:
    """Read only a time slab of ``var_name`` straight from ``path`` (axis-0 time).

    Reads ``[start]`` (when ``stop`` is ``None``) or ``[start:stop]`` along the leading axis via a
    windowed MDArray read, so a single-step / short-range selection never materialises the whole
    ``(n_time, n_elements)`` array (issue #982). The leading axis is dropped for a single step, so the
    result matches ``data[index]``.
    """
    ds = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR)
    if ds is None:
        raise ValueError(f"GDAL cannot re-open {path!r} for a windowed variable read.")
    try:
        rg = ds.GetRootGroup()
        md = open_mdarray(rg, var_name) if rg is not None else None
        if md is None:
            raise ValueError(f"Variable {var_name!r} is no longer present in {path!r}.")
        sizes = [d.GetSize() for d in md.GetDimensions()]
        start_idx = [0] * len(sizes)
        start_idx[0] = start
        count = list(sizes)
        count[0] = 1 if stop is None else stop - start
        # `ReadAsArray` returns a fresh, numpy-owned array, so the slab stays valid after the
        # dataset is closed in `finally` (closing the per-selection reopen deterministically keeps
        # Windows from holding a read handle on the file — review N3).
        slab = md.ReadAsArray(array_start_idx=start_idx, count=count)
    finally:
        ds = None
    if slab is None:
        result = None
    else:
        result = slab[0] if stop is None else slab
    return result


@dataclass
class MeshVariable:
    """Data variable defined on a mesh location.

    Wraps a numpy array of values associated with mesh elements
    (nodes, faces, or edges). Supports lazy loading via a loader
    callable that defers reading until data is first accessed.

    Attributes:
        name: Variable name in the NetCDF file.
        location: Mesh location ("node", "face", or "edge").
        mesh_name: Name of the associated mesh topology variable.
        shape: Shape of the data array.
        attributes: Dictionary of NetCDF variable attributes.
        nodata: No-data / fill value for masked elements.
        units: Physical units string (e.g., "m", "m/s").
        standard_name: CF standard name (e.g., "sea_surface_height").
        _data: Eagerly loaded data array, or None if using lazy loading.
        _loader: Callable that returns the data array on first access.
    """

    name: str
    location: str
    mesh_name: str
    shape: tuple[int, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    nodata: float | None = None
    units: str | None = None
    standard_name: str | None = None
    dimensions: tuple[str, ...] = field(default_factory=tuple)
    _data: np.ndarray | None = field(default=None, repr=False)
    _loader: Callable[[], np.ndarray] | None = field(default=None, repr=False)
    _dtype: np.dtype | None = field(default=None, repr=False)
    # File this variable was read from, threaded so a single-step / short-range time selection can
    # read only that slab from disk instead of the whole `(n_time, n_elements)` array (issue #982).
    _source_path: str | None = field(default=None, repr=False)

    @property
    def data(self) -> np.typing.NDArray | None:
        """Return the data array, triggering lazy load if needed."""
        if self._data is None and self._loader is not None:
            self._data = self._loader()
        return self._data

    @property
    def has_data_source(self) -> bool:
        """True when the variable can produce an array — eager data or a lazy loader.

        Metadata-only, so callers can decide whether to read a variable's values without forcing
        the load a bare ``.data`` access would trigger.
        """
        return self._data is not None or self._loader is not None

    @property
    def n_elements(self) -> int:
        """Number of mesh elements (last dimension of shape)."""
        result = self.shape[-1] if self.shape else 0
        return result

    @property
    def time_index(self) -> int | None:
        """Axis of the real time dimension, or `None` when the variable is not temporal.

        Identifies the time axis by name from `dimensions` (a dimension named `time`/`t`, or
        `time…`/`…_time`/`…_time_…` per the CF/UGRID convention) rather than assuming any extra axis
        is time. The word-boundary match avoids substring false-positives like `runtime` / `lifetime`
        / `daytime`, and a non-temporal multi-dimensional variable such as `(n_layers, n_face)`
        reports `None` (ARC-19). When `dimensions` is unknown — e.g. an array-built variable that
        carries no dimension names — it falls back to the historical heuristic (leading axis of a
        >1-D array).

        Returns:
            The 0-based index of the time axis, or `None` if the variable has no time dimension.
        """
        if self.dimensions:
            for axis, dim in enumerate(self.dimensions):
                lowered = dim.lower()
                if (
                    lowered in ("t", "time")
                    or lowered.startswith("time")
                    or lowered.endswith("_time")
                    or "_time_" in lowered
                ):
                    return axis
            return None
        return 0 if len(self.shape) > 1 else None

    @property
    def has_time(self) -> bool:
        """True when the variable has a real time dimension.

        Unlike a bare `ndim > 1` test, this is driven by `time_index`, so a non-temporal
        `(n_layers, n_face)` variable is not mis-flagged as temporal (ARC-19).
        """
        return self.time_index is not None

    @property
    def n_time_steps(self) -> int:
        """Number of time steps. Returns 0 if the variable has no time dimension."""
        idx = self.time_index
        result = self.shape[idx] if idx is not None and self.shape else 0
        return result

    @property
    def dtype(self) -> np.dtype:
        """Data type of the variable.

        Returns the explicitly set dtype if available, falls back to
        the loaded data's dtype, and defaults to float64.
        """
        if self._dtype is not None:
            result = self._dtype
        elif self._data is not None:
            result = self._data.dtype
        else:
            result = np.dtype("float64")
        return result

    def load_array(self) -> np.typing.NDArray | None:
        """Return the full array **without** memoising it on this variable.

        ``.data`` caches the loaded array on the instance; ``load_array`` reads it for a one-shot
        consumer (e.g. writing every variable to a file) without retaining it, so a bulk operation
        over many variables need not hold them all resident at once (issue #982). Returns an already
        loaded array when present.
        """
        if self._data is not None:
            result = self._data
        elif self._loader is not None:
            result = self._loader()
        else:
            result = None
        return result

    def _can_window_time(self, start: int, stop: int | None) -> bool:
        """True when a time slab can be read straight from disk instead of loading everything.

        Requires an on-disk source, an axis-0 time dimension (what the slab reader indexes), and
        non-negative in-range indices; otherwise the caller falls back to a full load then slice.
        A reversed **or empty** range (``stop <= start``) is excluded so the in-memory slice — which
        yields a consistent empty array — handles it, rather than passing a zero/negative ``count``
        to GDAL, which raises under ``gdal.UseExceptions()`` (reviews L2, R2-M1).
        """
        n_steps = self.n_time_steps
        return (
            self._source_path is not None
            and self.time_index == 0
            and start >= 0
            and start < n_steps
            and (stop is None or (start < stop <= n_steps))
        )

    def _read_time_window(
        self, start: int, stop: int | None
    ) -> np.typing.NDArray | None:
        """Return the ``[start]`` step (``stop is None``) or ``[start:stop]`` range of the time axis.

        Reads only the requested slab from disk when possible (:meth:`_can_window_time`); otherwise
        slices an already loaded array, or falls back to a full load then slice.
        """
        if self._data is not None:
            result = self._data[start] if stop is None else self._data[start:stop]
        elif self._can_window_time(start, stop):
            result = _read_time_slab(
                cast("str", self._source_path), self.name, start, stop
            )
        else:
            data = self.data
            if data is None:
                result = None
            else:
                result = data[start] if stop is None else data[start:stop]
        return result

    def sel_time(self, index: int) -> np.typing.NDArray:
        """Select a single time step, reading only that slab from disk when possible.

        Args:
            index: Time step index.

        Returns:
            1D array of values at the given time step.

        Raises:
            IndexError: If index is out of range.
            ValueError: If the variable has no time dimension, or has no
                loaded data array.
        """
        if not self.has_time:
            raise ValueError(f"Variable '{self.name}' has no time dimension.")
        slab = self._read_time_window(index, None)
        if slab is None:
            raise ValueError(f"Variable '{self.name}' has no loaded data.")
        return cast("np.typing.NDArray", slab)

    def sel_time_range(self, start: int, stop: int) -> MeshVariable:
        """Select a time range, reading only that slab from disk when possible.

        Args:
            start: Start time index (inclusive).
            stop: Stop time index (exclusive).

        Returns:
            New MeshVariable with the selected time range.

        Raises:
            ValueError: If the variable has no time dimension, or has no
                loaded data array.
        """
        if not self.has_time:
            raise ValueError(f"Variable '{self.name}' has no time dimension.")
        slab = self._read_time_window(start, stop)
        if slab is None:
            raise ValueError(f"Variable '{self.name}' has no loaded data.")
        return self.with_data(slab)

    def with_data(self, data: np.ndarray | None) -> MeshVariable:
        """Return a copy of this variable carrying ``data``, keeping all other metadata.

        The new variable is eager (``_data`` set, no loader); its ``shape`` is taken from
        ``data`` when provided, else the original ``shape`` is retained. Used wherever a
        derived dataset (time selection, spatial clip, …) replaces a variable's array while
        preserving its name / location / mesh / attributes.

        Args:
            data: The replacement data array, or ``None`` to keep the original shape with
                no eager data.

        Returns:
            MeshVariable: The copy carrying ``data``.
        """
        return MeshVariable(
            name=self.name,
            location=self.location,
            mesh_name=self.mesh_name,
            shape=data.shape if data is not None else self.shape,
            attributes=self.attributes,
            nodata=self.nodata,
            units=self.units,
            standard_name=self.standard_name,
            dimensions=self.dimensions,
            _data=data,
        )


@dataclass(frozen=True)
class UgridMetadata:
    """Full metadata summary for a UGRID dataset.

    Aggregates topology information, data variable inventory,
    global attributes, and mesh element counts for display
    and inspection purposes.

    Attributes:
        mesh_topologies: List of parsed mesh topologies in the file.
        data_variables: Mapping of variable name to location.
        global_attributes: File-level NetCDF attributes.
        conventions: Conventions string (e.g., "CF-1.8 UGRID-1.0").
        n_nodes: Total number of mesh nodes.
        n_faces: Total number of mesh faces.
        n_edges: Total number of mesh edges.
    """

    mesh_topologies: tuple[MeshTopologyInfo, ...] = ()
    data_variables: dict[str, str] = field(default_factory=dict)
    global_attributes: dict[str, Any] = field(default_factory=dict)
    conventions: str | None = None
    n_nodes: int = 0
    n_faces: int = 0
    n_edges: int = 0
