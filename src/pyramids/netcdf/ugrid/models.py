"""Data models for UGRID unstructured mesh metadata and variables.

This module defines the core data structures used throughout the
UGRID subpackage: topology metadata, mesh variables (data on the
mesh), and dataset-level metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
    _data: np.ndarray | None = field(default=None, repr=False)
    _loader: Callable[[], np.ndarray] | None = field(default=None, repr=False)
    _dtype: np.dtype | None = field(default=None, repr=False)

    @property
    def data(self) -> np.ndarray | None:
        """Return the data array, triggering lazy load if needed."""
        if self._data is None and self._loader is not None:
            self._data = self._loader()
        return self._data

    @property
    def n_elements(self) -> int:
        """Number of mesh elements (last dimension of shape)."""
        result = self.shape[-1] if self.shape else 0
        return result

    @property
    def has_time(self) -> bool:
        """True if the data has a time dimension (2D or higher)."""
        result = len(self.shape) > 1
        return result

    @property
    def n_time_steps(self) -> int:
        """Number of time steps. Returns 0 if no time dimension."""
        result = self.shape[0] if self.has_time else 0
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

    def sel_time(self, index: int) -> np.ndarray:
        """Select a single time step.

        Args:
            index: Time step index.

        Returns:
            1D array of values at the given time step.

        Raises:
            IndexError: If index is out of range.
            ValueError: If the variable has no time dimension.
        """
        if not self.has_time:
            raise ValueError(f"Variable '{self.name}' has no time dimension.")
        result = self.data[index]
        return result

    def sel_time_range(self, start: int, stop: int) -> MeshVariable:
        """Select a time range, returning a new MeshVariable.

        Args:
            start: Start time index (inclusive).
            stop: Stop time index (exclusive).

        Returns:
            New MeshVariable with the selected time range.

        Raises:
            ValueError: If the variable has no time dimension.
        """
        if not self.has_time:
            raise ValueError(f"Variable '{self.name}' has no time dimension.")
        sliced_data = self.data[start:stop]
        result = MeshVariable(
            name=self.name,
            location=self.location,
            mesh_name=self.mesh_name,
            shape=sliced_data.shape,
            attributes=self.attributes,
            nodata=self.nodata,
            units=self.units,
            standard_name=self.standard_name,
            _data=sliced_data,
        )
        return result


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
