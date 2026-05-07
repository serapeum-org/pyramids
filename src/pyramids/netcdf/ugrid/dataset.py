"""UgridDataset — top-level container for UGRID NetCDF mesh data.

Combines mesh topology, data variables, metadata, and provides
the user-facing API for reading, inspecting, and operating on
unstructured mesh data.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from osgeo import gdal
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point

from pyramids.base.crs import sr_from_epsg
from pyramids.basemap.basemap import add_basemap
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.netcdf.cf import write_global_attributes
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.interpolation import mesh_to_grid
from pyramids.netcdf.ugrid.io import (
    parse_ugrid_topology,
    write_ugrid_data_variable,
    write_ugrid_topology,
)
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable, UgridMetadata
from pyramids.netcdf.utils import _read_attributes


class UgridDataset:
    """Container for UGRID NetCDF mesh data.

    Combines mesh topology, data variables, and global attributes
    into a single object with GIS-aware operations. Does NOT inherit
    from Dataset or RasterBase — the raster paradigm does not
    apply to unstructured meshes.

    Attributes:
        _mesh: Mesh2d topology instance.
        _data_variables: Mapping of variable name to MeshVariable.
        _global_attributes: File-level NetCDF attributes.
        _topology_info: Parsed UGRID topology metadata.
        _crs_wkt: CRS in WKT format.
        _file_name: Source file path, if read from disk.
    """

    def __init__(
        self,
        mesh: Mesh2d,
        data_variables: dict[str, MeshVariable],
        global_attributes: dict[str, Any],
        topology_info: MeshTopologyInfo | None = None,
        crs_wkt: str | None = None,
        file_name: str | None = None,
    ):
        self._mesh = mesh
        self._data_variables = data_variables
        self._global_attributes = global_attributes
        self._topology_info = topology_info
        self._crs_wkt = crs_wkt
        self._file_name = file_name
        self._cached_crs: Any = None

    @classmethod
    def read_file(cls, path: str | Path) -> UgridDataset:
        """Open a UGRID NetCDF file.

        Automatically detects mesh topology, separates data variables
        from topology/coordinate variables, and builds the mesh.

        Args:
            path: Path to the .nc file.

        Returns:
            UgridDataset instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If no UGRID topology is found in the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ds = gdal.OpenEx(
            str(path),
            gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR,
        )
        if ds is None:
            raise ValueError(f"GDAL cannot open file: {path}")

        rg = ds.GetRootGroup()
        if rg is None:
            raise ValueError(f"Cannot get root group from: {path}")

        topologies = parse_ugrid_topology(rg)
        if not topologies:
            raise ValueError(f"No UGRID mesh topology found in: {path}")

        topo_info = topologies[0]
        mesh = Mesh2d.from_gdal_group(rg, topo_info)

        data_variables = _read_data_variables(rg, topo_info)

        global_attrs = _read_global_attributes(rg)

        ds = None

        result = cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes=global_attrs,
            topology_info=topo_info,
            crs_wkt=topo_info.crs_wkt,
            file_name=str(path),
        )
        return result

    @property
    def mesh(self) -> Mesh2d:
        """The mesh topology."""
        return self._mesh

    @property
    def mesh_name(self) -> str:
        """Name of the mesh topology variable."""
        result = self._topology_info.mesh_name if self._topology_info else "mesh2d"
        return result

    @property
    def data_variable_names(self) -> list[str]:
        """Names of all data variables."""
        result = list(self._data_variables.keys())
        return result

    @property
    def crs(self) -> CRS | None:
        """CRS as a pyproj.CRS object, or None. Cached after first access."""
        if self._cached_crs is None and self._crs_wkt is not None:
            try:
                self._cached_crs = CRS.from_wkt(self._crs_wkt)
            except Exception:
                pass
        return self._cached_crs

    @property
    def epsg(self) -> int | None:
        """EPSG code of the CRS, or None."""
        crs = self.crs
        result = crs.to_epsg() if crs is not None else None
        return result

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Mesh bounding box as (xmin, ymin, xmax, ymax)."""
        return self._mesh.bounds

    @property
    def global_attributes(self) -> dict[str, Any]:
        """File-level NetCDF attributes."""
        return self._global_attributes

    @property
    def n_node(self) -> int:
        """Number of mesh nodes."""
        return self._mesh.n_node

    @property
    def n_face(self) -> int:
        """Number of mesh faces."""
        return self._mesh.n_face

    @property
    def n_edge(self) -> int:
        """Number of mesh edges."""
        return self._mesh.n_edge

    def get_data(self, variable_name: str) -> MeshVariable:
        """Get a data variable by name.

        Args:
            variable_name: Name of the data variable.

        Returns:
            MeshVariable instance.

        Raises:
            KeyError: If the variable name is not found.
        """
        if variable_name not in self._data_variables:
            raise KeyError(
                f"Variable '{variable_name}' not found. "
                f"Available: {self.data_variable_names}"
            )
        result = self._data_variables[variable_name]
        return result

    def __getitem__(self, key: str) -> MeshVariable:
        """Get a data variable by name using bracket notation."""
        return self.get_data(key)

    @property
    def metadata(self) -> UgridMetadata:
        """Full metadata summary for this dataset."""
        topo_tuple = (self._topology_info,) if self._topology_info else ()
        data_vars = {name: var.location for name, var in self._data_variables.items()}
        conventions = self._global_attributes.get("Conventions")
        result = UgridMetadata(
            mesh_topologies=topo_tuple,
            data_variables=data_vars,
            global_attributes=self._global_attributes,
            conventions=conventions,
            n_nodes=self.n_node,
            n_faces=self.n_face,
            n_edges=self.n_edge,
        )
        return result

    def to_dataset(
        self,
        variable_name: str,
        cell_size: float,
        method: str = "nearest",
        bounds: tuple[float, float, float, float] | None = None,
        epsg: int | None = None,
        nodata: float = -9999.0,
    ) -> Dataset:
        """Convert a mesh variable to a regular-grid Dataset.

        Interpolates mesh data onto a regular grid and returns a
        standard pyramids Dataset. This is the bridge between
        unstructured (UGRID) and structured (raster) worlds.

        Args:
            variable_name: Name of the data variable to rasterize.
            cell_size: Target grid cell size in coordinate units.
            method: Interpolation method ("nearest" or "linear").
            bounds: Target (xmin, ymin, xmax, ymax). Defaults to mesh bounds.
            epsg: Target EPSG code. Defaults to mesh CRS.
            nodata: No-data value for the output raster.

        Returns:
            pyramids Dataset with the interpolated data.
        """
        var = self.get_data(variable_name)
        data = var.data
        if data is None:
            raise ValueError(f"Variable '{variable_name}' has no data loaded.")
        if var.has_time:
            data = data[0]

        grid_array, geotransform = mesh_to_grid(
            mesh=self._mesh,
            data=data,
            location=var.location,
            cell_size=cell_size,
            method=method,
            bounds=bounds,
            nodata=nodata,
        )

        target_epsg = epsg or self.epsg or 4326
        result = Dataset.create_from_array(
            grid_array,
            geo=geotransform,
            epsg=target_epsg,
            no_data_value=nodata,
        )
        return result

    def clip(self, mask: Any, touch: bool = True) -> UgridDataset:
        """Clip the mesh to a polygon mask.

        Selects faces that intersect (touch=True) or are fully
        contained within (touch=False) the mask polygon.

        Args:
            mask: Polygon mask (GeoDataFrame, FeatureCollection,
                or Shapely geometry).
            touch: If True, include faces touching the boundary.

        Returns:
            New UgridDataset with clipped mesh and data.
        """
        from pyramids.netcdf.ugrid.spatial import clip_mesh

        result = clip_mesh(self, mask, touch=touch)
        return result

    def subset_by_bounds(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> UgridDataset:
        """Subset mesh to faces within a bounding box.

        Args:
            xmin: Minimum x-coordinate.
            ymin: Minimum y-coordinate.
            xmax: Maximum x-coordinate.
            ymax: Maximum y-coordinate.

        Returns:
            New UgridDataset with subset mesh and data.
        """
        from pyramids.netcdf.ugrid.spatial import subset_by_bounds

        result = subset_by_bounds(self, xmin, ymin, xmax, ymax)
        return result

    def to_crs(self, to_epsg: int) -> UgridDataset:
        """Reproject all node coordinates to a new CRS.

        Uses pyproj.Transformer to reproject node coordinates.
        Face/edge center coordinates are recomputed after reprojection.
        Data values are preserved — only coordinates change.

        Args:
            to_epsg: Target EPSG code.

        Returns:
            New UgridDataset with reprojected coordinates.
        """
        source_epsg = self.epsg
        if source_epsg is None:
            raise ValueError(
                "Cannot reproject: source CRS is unknown. "
                "Set CRS before calling to_crs()."
            )

        transformer = Transformer.from_crs(
            f"EPSG:{source_epsg}",
            f"EPSG:{to_epsg}",
            always_xy=True,
        )
        new_node_x, new_node_y = transformer.transform(
            self._mesh.node_x,
            self._mesh.node_y,
        )

        new_face_x = None
        new_face_y = None
        if self._mesh._face_x is not None and self._mesh._face_y is not None:
            new_face_x, new_face_y = transformer.transform(
                self._mesh._face_x,
                self._mesh._face_y,
            )

        new_edge_x = None
        new_edge_y = None
        if self._mesh._edge_x is not None and self._mesh._edge_y is not None:
            new_edge_x, new_edge_y = transformer.transform(
                self._mesh._edge_x,
                self._mesh._edge_y,
            )

        new_mesh = Mesh2d(
            node_x=new_node_x,
            node_y=new_node_y,
            face_node_connectivity=self._mesh.face_node_connectivity,
            edge_node_connectivity=self._mesh.edge_node_connectivity,
            face_edge_connectivity=self._mesh.face_edge_connectivity,
            face_face_connectivity=self._mesh.face_face_connectivity,
            edge_face_connectivity=self._mesh.edge_face_connectivity,
            face_x=new_face_x,
            face_y=new_face_y,
            edge_x=new_edge_x,
            edge_y=new_edge_y,
        )

        srs = sr_from_epsg(to_epsg)
        new_crs_wkt = srs.ExportToWkt()

        new_topo_info = None
        if self._topology_info is not None:
            new_topo_info = replace(self._topology_info, crs_wkt=new_crs_wkt)

        result = UgridDataset(
            mesh=new_mesh,
            data_variables=self._data_variables,
            global_attributes=self._global_attributes,
            topology_info=new_topo_info,
            crs_wkt=new_crs_wkt,
        )
        return result

    @property
    def time_values(self) -> list | None:
        """Parsed time coordinate values from the first temporal variable.

        Returns None if no variables have a time dimension.
        """
        result = None
        for var in self._data_variables.values():
            if var.has_time:
                time_attr = var.attributes.get("time_values")
                if time_attr is not None:
                    result = list(time_attr)
                else:
                    result = list(range(var.n_time_steps))
                break
        return result

    def sel_time(self, index: int) -> UgridDataset:
        """Select a single time step from all temporal variables.

        Non-temporal variables are kept unchanged.

        Args:
            index: Time step index.

        Returns:
            New UgridDataset with single time step data.
        """
        new_data_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if var.has_time:
                sliced_data = var.sel_time(index)
                new_data_vars[name] = MeshVariable(
                    name=var.name,
                    location=var.location,
                    mesh_name=var.mesh_name,
                    shape=sliced_data.shape,
                    attributes=var.attributes,
                    nodata=var.nodata,
                    units=var.units,
                    standard_name=var.standard_name,
                    _data=sliced_data,
                )
            else:
                new_data_vars[name] = var

        result = UgridDataset(
            mesh=self._mesh,
            data_variables=new_data_vars,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
        )
        return result

    def sel_time_range(self, start: int, stop: int) -> UgridDataset:
        """Select a time range from all temporal variables.

        Args:
            start: Start index (inclusive).
            stop: Stop index (exclusive).

        Returns:
            New UgridDataset with the selected time range.
        """
        new_data_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if var.has_time:
                new_data_vars[name] = var.sel_time_range(start, stop)
            else:
                new_data_vars[name] = var

        result = UgridDataset(
            mesh=self._mesh,
            data_variables=new_data_vars,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
        )
        return result

    def to_file(self, path: str | Path) -> None:
        """Write to a UGRID-compliant NetCDF file.

        Creates a NetCDF file with topology variable, node coordinates,
        connectivity arrays, face/edge centers, data variables, and
        global attributes following the UGRID convention.

        Args:
            path: Output file path.
        """
        path = Path(path)
        drv = gdal.GetDriverByName("netCDF")
        ds = drv.CreateMultiDimensional(str(path))
        rg = ds.GetRootGroup()

        mesh_name = self.mesh_name
        dims = write_ugrid_topology(rg, self._mesh, mesh_name, self._crs_wkt)

        for var in self._data_variables.values():
            if var.has_time and "time" not in dims:
                time_dim = rg.CreateDimension("time", None, None, var.n_time_steps)
                dims["time"] = time_dim
            write_ugrid_data_variable(rg, var, mesh_name, dims)

        global_attrs = dict(self._global_attributes)
        if "Conventions" not in global_attrs:
            global_attrs["Conventions"] = "CF-1.8 UGRID-1.0"
        write_global_attributes(rg, global_attrs)

        ds = None

    def to_geodataframe(
        self,
        variable_name: str | None = None,
        location: str = "face",
    ) -> gpd.GeoDataFrame:
        """Convert mesh to a GeoDataFrame.

        For faces: each row is a Polygon with data columns.
        For nodes: each row is a Point.
        For edges: each row is a LineString.

        Args:
            variable_name: Optional data variable to include as a column.
            location: Mesh location ("face", "node", or "edge").

        Returns:
            geopandas GeoDataFrame.
        """
        geometries = []
        if location == "face":
            from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex

            spatial_idx = MeshSpatialIndex(self._mesh)
            geometries = spatial_idx.face_polygons
        elif location == "node":
            for i in range(self.n_node):
                geometries.append(Point(self._mesh.node_x[i], self._mesh.node_y[i]))
        elif location == "edge":
            if self._mesh.edge_node_connectivity is None:
                raise ValueError("Edge connectivity not available.")
            enc = self._mesh.edge_node_connectivity
            for i in range(enc.n_elements):
                nodes = enc.get_element(i)
                coords = [(self._mesh.node_x[n], self._mesh.node_y[n]) for n in nodes]
                geometries.append(LineString(coords))
        else:
            raise ValueError(f"Unknown location: {location}")

        data_dict: dict[str, Any] = {}
        if variable_name is not None:
            var = self.get_data(variable_name)
            if var.location == location:
                var_data = var.data
                if var_data is not None and var.has_time:
                    var_data = var_data[0]
                data_dict[variable_name] = var_data

        gdf = gpd.GeoDataFrame(data_dict, geometry=geometries)
        if self.crs is not None:
            gdf = gdf.set_crs(self.crs)

        result = gdf
        return result

    def to_feature_collection(
        self,
        variable_name: str | None = None,
        location: str = "face",
    ) -> FeatureCollection:
        """Convert mesh to a pyramids FeatureCollection.

        Args:
            variable_name: Optional data variable to include.
            location: Mesh location ("face", "node", or "edge").

        Returns:
            pyramids FeatureCollection.
        """
        gdf = self.to_geodataframe(variable_name, location)
        result = FeatureCollection(gdf)
        return result

    @classmethod
    def create_from_arrays(
        cls,
        node_x: np.ndarray,
        node_y: np.ndarray,
        face_node_connectivity: np.ndarray,
        data: dict[str, np.ndarray] | None = None,
        data_locations: dict[str, str] | None = None,
        epsg: int = 4326,
        mesh_name: str = "mesh2d",
    ) -> UgridDataset:
        """Create a UgridDataset programmatically from arrays.

        Args:
            node_x: Node x-coordinates.
            node_y: Node y-coordinates.
            face_node_connectivity: (n_faces, max_nodes) array of node
                indices. Use -1 as fill value for mixed meshes.
            data: Optional dict mapping variable name to data array.
            data_locations: Optional dict mapping variable name to
                location ("face", "node", "edge"). Defaults to "face".
            epsg: EPSG code for the CRS.
            mesh_name: Name for the topology variable.

        Returns:
            UgridDataset instance.
        """
        fnc = Connectivity(
            data=np.asarray(face_node_connectivity, dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.asarray(node_x, dtype=np.float64),
            node_y=np.asarray(node_y, dtype=np.float64),
            face_node_connectivity=fnc,
        )

        data_variables: dict[str, MeshVariable] = {}
        topo_data_vars: dict[str, str] = {}
        if data is not None:
            if data_locations is None:
                data_locations = {}
            for name, arr in data.items():
                loc = data_locations.get(name, "face")
                topo_data_vars[name] = loc
                data_variables[name] = MeshVariable(
                    name=name,
                    location=loc,
                    mesh_name=mesh_name,
                    shape=arr.shape,
                    _data=arr,
                )

        srs = sr_from_epsg(epsg)
        crs_wkt = srs.ExportToWkt()

        topo_info = MeshTopologyInfo(
            mesh_name=mesh_name,
            topology_dimension=2,
            node_x_var=f"{mesh_name}_node_x",
            node_y_var=f"{mesh_name}_node_y",
            face_node_var=f"{mesh_name}_face_nodes",
            data_variables=topo_data_vars,
            crs_wkt=crs_wkt,
        )

        result = cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes={"Conventions": "CF-1.8 UGRID-1.0"},
            topology_info=topo_info,
            crs_wkt=crs_wkt,
        )
        return result

    def plot(
        self,
        variable_name: str,
        ax: Any = None,
        cmap: str = "viridis",
        title: str | None = None,
        basemap: bool | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot a mesh data variable.

        Args:
            variable_name: Name of the data variable to plot.
            ax: matplotlib Axes. Created if None.
            cmap: Colormap name.
            title: Plot title. Defaults to variable name.
            basemap: If True, add an OpenStreetMap basemap. If a string,
                use it as the tile provider name (e.g. "CartoDB.Positron").
                Default is None (no basemap). Requires the [viz] extra.
            **kwargs: Additional arguments passed to plot_mesh_data.

        Returns:
            cleopatra.mesh_glyph.MeshGlyph instance with the plot
                rendered. Use the returned object to access Figure/Axes
                or call additional MeshGlyph methods.
        """
        from pyramids.netcdf.ugrid.plot import plot_mesh_data

        var = self.get_data(variable_name)
        data = var.data
        if var.has_time:
            data = data[0]
        if title is None:
            title = variable_name
        result = plot_mesh_data(
            self._mesh,
            data,
            location=var.location,
            ax=ax,
            cmap=cmap,
            title=title,
            **kwargs,
        )

        if basemap:
            if self.epsg is None:
                raise ValueError(
                    "UgridDataset must have a CRS (epsg) to " "use basemap."
                )
            source = basemap if isinstance(basemap, str) else None
            ax = result.ax if hasattr(result, "ax") else result
            add_basemap(ax, crs=self.epsg, source=source)

        return result

    def plot_outline(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot mesh wireframe.

        Args:
            ax: matplotlib Axes. Created if None.
            **kwargs: Additional arguments passed to plot_mesh_outline.

        Returns:
            cleopatra.mesh_glyph.MeshGlyph instance with the wireframe
                rendered.
        """
        from pyramids.netcdf.ugrid.plot import plot_mesh_outline

        result = plot_mesh_outline(self._mesh, ax=ax, **kwargs)
        return result

    def __str__(self) -> str:
        """Human-readable summary of the dataset."""
        lines = [
            f"UgridDataset: {self._file_name or '(in-memory)'}",
            f"  Mesh: {self.mesh_name}",
            f"  Nodes: {self.n_node}, Faces: {self.n_face}, Edges: {self.n_edge}",
            f"  Bounds: {self.bounds}",
            f"  CRS: {self.epsg or 'unknown'}",
            f"  Data variables ({len(self._data_variables)}):",
        ]
        for name, var in self._data_variables.items():
            lines.append(f"    {name}: location={var.location}, shape={var.shape}")
        result = "\n".join(lines)
        return result

    def __repr__(self) -> str:
        """Repr string for the dataset."""
        result = (
            f"UgridDataset(mesh='{self.mesh_name}', "
            f"n_node={self.n_node}, n_face={self.n_face}, n_edge={self.n_edge}, "
            f"variables={self.data_variable_names})"
        )
        return result


def _read_data_variables(
    rg: gdal.Group,
    topo_info: MeshTopologyInfo,
) -> dict[str, MeshVariable]:
    """Read all data variables from a GDAL root group.

    Creates MeshVariable instances with eagerly loaded data for each
    variable that references the mesh topology.

    Args:
        rg: GDAL root group.
        topo_info: Parsed topology info with data variable names and locations.

    Returns:
        Dictionary mapping variable name to MeshVariable.
    """
    variables: dict[str, MeshVariable] = {}

    for var_name, location in topo_info.data_variables.items():
        md_arr = rg.OpenMDArray(var_name)
        if md_arr is None:
            continue
        attrs = _read_attributes(md_arr)
        dims = md_arr.GetDimensions()
        shape = tuple(d.GetSize() for d in dims) if dims else ()

        nodata = attrs.get("_FillValue")
        if nodata is not None:
            nodata = float(nodata)
        units = attrs.get("units")
        standard_name = attrs.get("standard_name")

        data = md_arr.ReadAsArray()
        if data is not None:
            data = data.copy()

        variables[var_name] = MeshVariable(
            name=var_name,
            location=location,
            mesh_name=topo_info.mesh_name,
            shape=shape,
            attributes=attrs,
            nodata=nodata,
            units=units,
            standard_name=standard_name,
            _data=data,
        )

    return variables


def _read_global_attributes(rg: gdal.Group) -> dict[str, Any]:
    """Read global attributes from a GDAL root group.

    Args:
        rg: GDAL root group.

    Returns:
        Dictionary of global attribute name-value pairs.
    """
    result = _read_attributes(rg)
    return result
