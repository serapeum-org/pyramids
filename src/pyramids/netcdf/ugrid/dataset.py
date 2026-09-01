"""UgridDataset — top-level container for UGRID NetCDF mesh data.

Combines mesh topology, data variables, metadata, and provides
the user-facing API for reading, inspecting, and operating on
unstructured mesh data.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import numpy as np
import shapely

if TYPE_CHECKING:
    from cleopatra.glyphs.gridded.array_glyph import PointOverlay
    from cleopatra.styling.colorbar import ColorBar
    from cleopatra.styling.params import Contour, DataStyle
    from cleopatra.styling.scaling import ColorScaling
from osgeo import gdal
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box

from pyramids.base.crs import crs_from_user_input, sr_from_epsg
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset._plot_helpers import mesh_render as _mesh_render
from pyramids.dataset._plot_helpers import nonnull_group_kwargs as _nonnull_group_kwargs
from pyramids.feature import FeatureCollection
from pyramids.netcdf._mdim import open_mdarray
from pyramids.netcdf.cf import write_global_attributes
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.interpolation import mesh_to_grid
from pyramids.netcdf.ugrid.io import (
    parse_ugrid_topology,
    write_ugrid_data_variable,
    write_ugrid_topology,
)
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import (
    DEFAULT_MESH_NAME,
    MeshTopologyInfo,
    MeshVariable,
    UgridMetadata,
)
from pyramids.netcdf.ugrid.spatial import (
    MeshSpatialIndex,
    clip_mesh,
    subset_by_bounds,
)
from pyramids.netcdf.utils import (
    _dtype_to_str,
    _read_attributes,
    read_cf_attributes,
)


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

        # Resolve to an absolute path before threading it into the lazy variable loaders:
        # data reads are deferred to first `.data` access (PERF-3), which re-opens the file.
        # A relative path would break that deferred open if the process changed directory in
        # the meantime; the old eager read was immune because it read while still in `read_file`.
        data_variables = _read_data_variables(rg, topo_info, str(path.resolve()))

        global_attrs = _read_attributes(rg)

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
        result = (
            self._topology_info.mesh_name if self._topology_info else DEFAULT_MESH_NAME
        )
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
            except Exception:  # nosec B110 - best-effort CRS parse; falls back to None
                pass
        return cast("CRS | None", self._cached_crs)

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
        result = Dataset.from_array(
            grid_array,
            no_data_value=nodata,
            geo_ref=GeoReference(
                geo=cast(
                    "tuple[float, float, float, float, float, float]", geotransform
                ),
                epsg=target_epsg,
            ),
        )
        return result

    def crop(
        self,
        mask: Any = None,
        touch: bool = True,
        *,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        epsg: int | None = None,
    ) -> UgridDataset:
        """Crop the mesh to a polygon mask or a bbox — the unstructured-mesh analogue of crop.

        The mesh equivalent of :meth:`pyramids.dataset.Dataset.crop` / :meth:`NetCDF.crop`. Rather
        than warping a raster, it selects the **faces** that fall inside the region (renumbering the
        node connectivity for the resulting sub-mesh) and keeps the data on the surviving elements.
        Delegates to :meth:`clip` for a polygon and :meth:`subset_by_bounds` for a bbox; this method
        exists so the spatial-subset call is named ``crop`` across the raster and mesh classes alike.

        Args:
            mask (Any):
                Polygon mask — a shapely geometry, ``GeoDataFrame``, or ``FeatureCollection``.
                Mutually exclusive with ``bbox``.
            touch (bool):
                Applies only to a polygon ``mask``: if ``True`` (default), keep faces that touch the
                mask boundary; if ``False``, keep only faces fully inside it. Ignored for ``bbox``,
                which always selects faces by its axis-aligned envelope (``subset_by_bounds``).
                Defaults to True.
            bbox (tuple or list of 4 floats, keyword-only):
                ``(west, south, east, north)`` in the mesh CRS, or in ``epsg`` when supplied. Accepts
                a tuple or a list. Selects faces by the axis-aligned envelope; not affected by
                ``touch``. Mutually exclusive with ``mask``.
            epsg (int, keyword-only):
                CRS of ``bbox``. When it differs from the mesh CRS the box is reprojected to the mesh
                CRS and subset by its envelope; when it equals the mesh CRS it is a no-op. Defaults to
                the mesh CRS.

        Returns:
            UgridDataset: A new sub-mesh — faces inside the region, connectivity renumbered, and data
                variables subset to the surviving elements.

        Raises:
            ValueError: If both ``mask`` and ``bbox`` are supplied, if ``bbox`` is not a 4-tuple, or
                if ``epsg`` is given for a ``bbox`` but the mesh has no CRS to reproject into.
            TypeError: If neither ``mask`` nor ``bbox`` is supplied.

        Examples:
            - Crop a mesh to a polygon (faces intersecting it survive):
                ```python
                >>> from shapely.geometry import Polygon
                >>> from pyramids.netcdf import UgridDataset
                >>> ug = UgridDataset.read_file("mesh.nc")                        # doctest: +SKIP
                >>> sub = ug.crop(Polygon([(-1, -1), (0, -1), (0, 1), (-1, 1)]))  # doctest: +SKIP
                >>> sub.n_face <= ug.n_face                                       # doctest: +SKIP
                True

                ```
            - Crop to a bounding box in the mesh's own CRS:
                ```python
                >>> sub = ug.crop(bbox=(-1.0, -1.0, 0.0, 1.0))                    # doctest: +SKIP

                ```

        See Also:
            clip: Polygon-mask subsetting that ``crop`` delegates to when ``mask`` is given.
            subset_by_bounds: Bounding-box subsetting that ``crop`` delegates to when ``bbox`` is
                given.
            pyramids.dataset.Dataset.crop: The raster equivalent on a gridded dataset.
        """
        if bbox is not None:
            if mask is not None:
                raise ValueError("crop accepts either `mask` or `bbox`, not both")
            if len(bbox) != 4:
                raise ValueError(
                    "bbox must be a 4-tuple of (west, south, east, north), "
                    f"got {len(bbox)} value(s)"
                )
            west, south, east, north = bbox
            if epsg is not None and (self.epsg is None or int(epsg) != int(self.epsg)):
                if self.epsg is None:
                    raise ValueError(
                        f"cannot reproject a bbox given in EPSG:{int(epsg)} into a mesh that has no "
                        "CRS; drop epsg to treat the bbox as native coordinates"
                    )
                # Reproject the bbox to the mesh CRS and subset by its envelope, so the bbox path
                # selects faces with the same rule (subset_by_bounds) regardless of the source CRS.
                west, south, east, north = (
                    gpd.GeoSeries([box(west, south, east, north)], crs=epsg)
                    .to_crs(self.epsg)
                    .total_bounds
                )
            result = self.subset_by_bounds(west, south, east, north)
        elif mask is not None:
            result = self.clip(mask, touch=touch)
        else:
            raise TypeError(
                "crop requires a `mask` (polygon) or a `bbox` (west, south, east, north) tuple"
            )
        return result

    def _wrap_subset(
        self, mesh: Mesh2d, data_variables: dict[str, MeshVariable]
    ) -> UgridDataset:
        """Wrap a ``(mesh, data_variables)`` pair from the spatial subsetters into a dataset.

        The spatial subsetting helpers (:func:`clip_mesh` / :func:`subset_by_bounds`) return
        the rebuilt mesh and data variables rather than a dataset (STR-3 — keeps
        ``ugrid.spatial`` independent of this module). This carries the source dataset's
        global attributes / topology info / CRS onto the subset.

        Args:
            mesh: The subset mesh.
            data_variables: The sliced data variables.

        Returns:
            UgridDataset: The wrapped subset.
        """
        return UgridDataset(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=None,
        )

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
        mesh, data_variables = clip_mesh(self, mask, touch=touch)
        return self._wrap_subset(mesh, data_variables)

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
        mesh, data_variables = subset_by_bounds(self, xmin, ymin, xmax, ymax)
        return self._wrap_subset(mesh, data_variables)

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

        # Through `crs_from_user_input` so a mesh in a CRS whose code only GDAL's
        # PROJ database carries still reprojects (issue #943).
        transformer = Transformer.from_crs(
            crs_from_user_input(f"EPSG:{source_epsg}"),
            crs_from_user_input(f"EPSG:{to_epsg}"),
            always_xy=True,
        )
        new_node_x, new_node_y = transformer.transform(
            self._mesh.node_x,
            self._mesh.node_y,
        )

        new_face_x = None
        new_face_y = None
        if self._mesh.has_face_coords:
            new_face_x, new_face_y = transformer.transform(
                self._mesh.face_x,
                self._mesh.face_y,
            )

        new_edge_x = None
        new_edge_y = None
        if self._mesh.has_edge_coords:
            new_edge_x, new_edge_y = transformer.transform(
                self._mesh.edge_x,
                self._mesh.edge_y,
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
                new_data_vars[name] = var.with_data(var.sel_time(index))
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
            # `load_array` reads each variable without memoising it on the shared dataset, so the
            # write streams one variable at a time instead of holding the whole cube resident (#982).
            write_ugrid_data_variable(
                rg, var.with_data(var.load_array()), mesh_name, dims
            )

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
        geometries = self._build_geometries(location)

        data_dict: dict[str, Any] = {}
        if variable_name is not None:
            var = self.get_data(variable_name)
            if var.location == location:
                # For a temporal variable only the first step is tabulated; `sel_time(0)` reads just
                # that slab instead of loading every step to slice `[0]` (#982). `has_data_source`
                # avoids a temporal-specific `sel_time` error for a variable with no readable data
                # (checked without forcing a load — review L3).
                if var.has_time:
                    var_data = var.sel_time(0) if var.has_data_source else None
                else:
                    var_data = var.data
                # A variable with no readable data becomes a length-correct null column rather than
                # raising — pandas rejects a scalar `None` column ("must pass an index") (review L3).
                if var_data is None:
                    var_data = np.full(len(geometries), np.nan)
                data_dict[variable_name] = var_data

        gdf = gpd.GeoDataFrame(data_dict, geometry=geometries)
        if self.crs is not None:
            gdf = gdf.set_crs(self.crs)

        result = gdf
        return result

    def _build_geometries(self, location: str) -> list:
        """Build the geometry list for a mesh location.

        Args:
            location: Mesh location ("face", "node", or "edge").

        Returns:
            List of shapely geometries — Polygons for "face", Points for
            "node", LineStrings for "edge".

        Raises:
            ValueError: If `location` is unknown, or edge connectivity is
                unavailable for an edge conversion.
        """
        if location == "face":
            geometries = MeshSpatialIndex(self._mesh).face_polygons
        elif location == "node":
            # Vectorized point construction — meshes routinely have 1e5-1e7 nodes, so a per-node
            # Python `Point(...)` loop is a hot spot (ARC-59). `list(...)` keeps the return type
            # consistent with the face/edge branches (review N2).
            geometries = list(shapely.points(self._mesh.node_x, self._mesh.node_y))
        elif location == "edge":
            if self._mesh.edge_node_connectivity is None:
                raise ValueError("Edge connectivity not available.")
            geometries = self._edge_linestrings(self._mesh.edge_node_connectivity)
        else:
            raise ValueError(f"Unknown location: {location}")
        return geometries

    def _edge_linestrings(self, enc: Connectivity) -> Any:
        """Build one LineString per edge, vectorized for standard 2-node edges (ARC-59)."""
        node_idx = np.asarray(enc.data)
        # A `None` fill means no sentinels are present, so the fast path is valid without an
        # elementwise `node_idx != None` compare (which NumPy deprecates) (review N3).
        no_fill = enc.fill_value is None or bool(np.all(node_idx != enc.fill_value))
        if node_idx.ndim == 2 and node_idx.shape[1] == 2 and no_fill:
            xs = self._mesh.node_x[node_idx]
            ys = self._mesh.node_y[node_idx]
            return list(shapely.linestrings(np.stack([xs, ys], axis=-1)))
        # Rare ragged / filled edge connectivity: fall back to a per-edge build.
        return [
            LineString(
                [
                    (self._mesh.node_x[n], self._mesh.node_y[n])
                    for n in enc.get_element(i)
                ]
            )
            for i in range(enc.n_elements)
        ]

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
    def from_arrays(
        cls,
        node_x: np.ndarray,
        node_y: np.ndarray,
        face_node_connectivity: np.ndarray,
        data: dict[str, np.ndarray] | None = None,
        data_locations: dict[str, str] | None = None,
        epsg: int = 4326,
        mesh_name: str = DEFAULT_MESH_NAME,
    ) -> UgridDataset:
        """Create a UgridDataset programmatically from arrays.

        Plural because an unstructured mesh is not one array: the topology
        needs node coordinates *and* a face-node connectivity table before any
        data can be attached. It therefore takes a flat `epsg` rather than the
        :class:`~pyramids.base.georeference.GeoReference` the gridded
        constructors take — a mesh carries its own coordinates, so there is no
        affine transform to describe.

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

        Examples:
            - Build the smallest possible mesh — two triangles — and inspect
              its topology:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import UgridDataset
                >>> mesh = UgridDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ... )
                >>> (mesh.n_node, mesh.n_face)
                (4, 2)
                >>> mesh.bounds
                (0.0, 0.0, 1.0, 1.0)

                ```
            - Attach a per-face variable and read it back:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import UgridDataset
                >>> mesh = UgridDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"depth": np.array([1.5, 2.5])},
                ... )
                >>> mesh.data_variable_names
                ['depth']
                >>> mesh["depth"].location
                'face'
                >>> mesh["depth"].data.tolist()
                [1.5, 2.5]

                ```
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
        colorbar: bool | ColorBar | None = None,
        points: np.ndarray | PointOverlay | None = None,
        kind: str = "auto",
        color: ColorScaling | None = None,
        contour: Contour | None = None,
        data_style: DataStyle | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot a mesh data variable.

        N-6 — this facade now goes through the same module-level
        helper as the raster path. The mesh-specific dispatch lives in
        :func:`pyramids.dataset._plot_helpers.mesh_render`; both
        ``Dataset.plot``/``NetCDF.plot`` and ``UgridDataset.plot`` share
        the "resolve data, hand to a single helper" contract so the
        two formats no longer maintain independent plotting code paths.

        Args:
            variable_name: Name of the data variable to plot.
            ax: matplotlib Axes. Created if None.
            cmap: Colormap name.
            title: Plot title. Defaults to variable name.
            basemap: If True, add an OpenStreetMap basemap. If a string,
                use it as the tile provider name (e.g. "CartoDB.Positron").
                Default is None (no basemap). Requires the [viz] extra.
            colorbar (bool or ColorBar, optional): Colour-bar spec, part of the
                shared plot signature. A ``pyramids.plot.ColorBar(label=…, …)``
                configures the bar; ``False`` hides it and ``None`` (default) uses
                cleopatra's default (a bar is drawn). Only forwarded when set.
            points (np.ndarray or PointOverlay, optional): Accepted for signature
                symmetry with the raster plot family, but a **no-op here** — a mesh
                has no point-overlay layer (the mesh geometry is the data). Ignored.
            kind (str, optional): Accepted for signature symmetry with the raster
                plot family, but a **no-op here** — ``kind`` selects a raster
                renderer (``imshow``/``pcolormesh``); the mesh always renders via
                ``tripcolor``/``tricontour``. Ignored.
            color (ColorScaling, optional): Colour-scale spec
                ``pyramids.plot.ColorScaling`` (linear / power / sym-log / boundary /
                midpoint norm), e.g. ``ColorScaling.power(gamma=0.7)``. Default ``None``.
            contour (Contour, optional): Contour-line spec
                ``pyramids.plot.Contour(levels=…, label_kw=…)``. Default ``None``.
            data_style (DataStyle, optional): Data-style / relief spec
                ``pyramids.plot.DataStyle(style=…, hillshade=…)``. (A mesh has no
                cell-value overlay, so there is no ``cells`` param here.) Default ``None``.
            **kwargs: Additional arguments passed to mesh_render
                (forwarded to plot_mesh_data). Notably ``colorbar``
                (``bool``, default ``True``): pass ``colorbar=False`` to
                suppress the per-mesh colorbar when you want to attach a
                custom or shared one to ``glyph.ax``. Also ``style`` (name of
                a cleopatra ``DATA_STYLES`` preset, e.g. ``"flow_accumulation"``)
                and ``hillshade`` (``True`` or a params dict) to colour / relief-
                shade the mesh; both require cleopatra >= 0.24 (``hillshade``
                needs node-centered data). Distinct from
                :meth:`pyramids.dataset.Dataset.hillshade`, which *returns* a
                shaded-relief array.

        Returns:
            cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph instance with the plot
                rendered. Use the returned object to access the matplotlib
                handles and the mappable:

                - ``glyph.fig`` / ``glyph.ax`` — Figure and Axes.
                - ``glyph.im`` — the mesh mappable (the
                  ``tripcolor``/``tricontour(f)`` artist) set by ``plot()``;
                  use it for a custom colorbar or ``glyph.im.set_clim(...)``.
                  It is ``None`` after :meth:`plot_outline` (an outline
                  carries no scalar mapping).
                - ``glyph.apply_style(style)`` (cleopatra >= 0.25) — re-apply a
                  ``DATA_STYLES`` preset by name in place, without re-plotting.

        Raises:
            ValueError: If the selected variable has no loaded data, or if
                `basemap` is requested while the dataset has no CRS (`epsg`).
        """
        var = self.get_data(variable_name)
        data = var.data
        if data is None:
            raise ValueError(f"Variable {variable_name!r} has no loaded data to plot.")
        if var.has_time:
            data = data[0]
        if title is None:
            title = variable_name
        if basemap and self.epsg is None:
            raise ValueError("UgridDataset must have a CRS (epsg) to use basemap.")
        # ``points`` / ``kind`` are part of the shared raster-family plot signature
        # but have no meaning for a mesh (no point overlay; the renderer is fixed to
        # tripcolor/tricontour), so they are accepted and ignored. ``colorbar`` and the
        # typed render groups (``color`` / ``contour`` / ``data_style``) map onto the mesh
        # backend and are forwarded only when set (so cleopatra's backend defaults are
        # preserved otherwise).
        if colorbar is not None:
            kwargs["colorbar"] = colorbar
        kwargs.update(
            _nonnull_group_kwargs(color=color, contour=contour, data_style=data_style)
        )
        result = _mesh_render(
            mesh=self._mesh,
            data=data,
            location=var.location,
            ax=ax,
            cmap=cmap,
            title=title,
            basemap=basemap,
            basemap_epsg=self.epsg,
            **kwargs,
        )
        return result

    def plot_outline(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot mesh wireframe.

        Args:
            ax: matplotlib Axes. Created if None.
            **kwargs: Additional arguments passed to plot_mesh_outline.

        Returns:
            cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph instance with the wireframe
                rendered. ``glyph.fig`` / ``glyph.ax`` are the matplotlib
                handles; ``glyph.im`` is ``None`` (an outline carries no
                scalar mapping, so no mappable is produced).
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


def _make_variable_loader(path: str, var_name: str):
    """Build a zero-arg loader that reads one variable's array on first access.

    The store opened in :meth:`UgridDataset.read_file` is closed before any
    :class:`MeshVariable` data is touched, so a lazy loader cannot capture the live
    root group — it re-opens ``path`` and reads ``var_name`` on demand instead. This
    keeps ``read_file`` metadata-only: variables the caller never touches are never read.

    Args:
        path: File path to re-open for the read.
        var_name: Name of the MDArray to read.

    Returns:
        Callable[[], np.ndarray | None]: A loader returning the variable's array (or
        ``None`` when it has no readable values).
    """

    def _load() -> np.typing.NDArray | None:
        ds = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR)
        if ds is None:
            raise ValueError(f"GDAL cannot re-open {path!r} for a lazy variable read.")
        rg = ds.GetRootGroup()
        md = open_mdarray(rg, var_name) if rg is not None else None
        if md is None:
            raise ValueError(
                f"Variable {var_name!r} is no longer present in {path!r} on lazy read."
            )
        # `ReadAsArray()` already returns a fresh, numpy-owned array, so an extra `.copy()` only
        # duplicates the largest arrays for no benefit (#982). The typed local coerces GDAL's
        # untyped `Any` return to the declared type (no-any-return).
        data: np.typing.NDArray | None = md.ReadAsArray()
        return data

    return _load


def _read_data_variables(
    rg: gdal.Group,
    topo_info: MeshTopologyInfo,
    path: str,
) -> dict[str, MeshVariable]:
    """Read every mesh data variable's metadata, deferring the array read.

    Creates a :class:`MeshVariable` per variable that references the mesh topology.
    Only metadata (attributes, shape, dtype, nodata, units, standard name) is read
    eagerly; the array itself loads lazily on first ``.data`` access via a re-opening
    loader, so ``read_file`` does not pull every variable into memory.

    Args:
        rg: GDAL root group (used for metadata only).
        topo_info: Parsed topology info with data variable names and locations.
        path: File path threaded into each variable's lazy loader.

    Returns:
        Dictionary mapping variable name to MeshVariable.
    """
    variables: dict[str, MeshVariable] = {}

    for var_name, location in topo_info.data_variables.items():
        md_arr = open_mdarray(rg, var_name)
        if md_arr is None:
            continue
        attrs = read_cf_attributes(md_arr)
        dims = md_arr.GetDimensions()
        shape = tuple(d.GetSize() for d in dims) if dims else ()
        dim_names = tuple(d.GetName() for d in dims) if dims else ()

        nodata = attrs.get("_FillValue")
        if nodata is not None:
            nodata = float(cast("float", nodata))
        units = cast("str | None", attrs.get("units"))
        standard_name = cast("str | None", attrs.get("standard_name"))
        try:
            dtype = np.dtype(_dtype_to_str(md_arr.GetDataType()))
        except (RuntimeError, TypeError):
            dtype = None

        variables[var_name] = MeshVariable(
            name=var_name,
            location=location,
            mesh_name=topo_info.mesh_name,
            shape=shape,
            attributes=attrs,
            nodata=nodata,
            units=units,
            standard_name=standard_name,
            dimensions=dim_names,
            _loader=_make_variable_loader(path, var_name),
            _dtype=dtype,
            _source_path=path,
        )

    return variables
