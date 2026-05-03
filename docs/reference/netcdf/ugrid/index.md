# UGRID Subpackage

The `ugrid` subpackage provides full support for UGRID (Unstructured
Grid) NetCDF files used in hydrological and coastal modeling (D-Flow FM,
SCHISM, FVCOM, ADCIRC).

## Architecture

`UgridDataset` is a standalone class that does **not** inherit from
`Dataset` or `RasterBase`. Unstructured meshes have no
geotransform, rows, or columns. Instead, data is indexed by topology
elements (nodes, faces, edges).

```
UgridDataset (top-level container)
    |-- mesh: Mesh2d (topology)
    |     |-- node_x, node_y: coordinates
    |     |-- face_node_connectivity: Connectivity
    |     |-- edge_node_connectivity: Connectivity
    |     +-- face_centroids, face_areas, triangulation (lazy)
    |
    |-- data_variables: dict[str, MeshVariable]
    |-- to_dataset()  -----> Dataset (mesh-to-raster bridge)
    |-- to_geodataframe() -> GeoDataFrame (mesh-to-vector bridge)
    |-- clip() / subset_by_bounds() (spatial operations)
    +-- to_file() (UGRID-compliant NetCDF write)
```

## Quick Start

```python
from pyramids.netcdf.ugrid import UgridDataset

# Read a UGRID file
ds = UgridDataset.read_file("model_output.nc")
print(ds)

# Access mesh topology
print(f"Nodes: {ds.n_node}, Faces: {ds.n_face}, Edges: {ds.n_edge}")
print(f"Bounds: {ds.bounds}")

# Access data
water_level = ds["water_level"]
print(f"Location: {water_level.location}, Shape: {water_level.shape}")

# Convert to regular grid (raster)
raster = ds.to_dataset("water_level", cell_size=100.0)
raster.to_file("water_level.tif")

# Convert to GeoDataFrame
gdf = ds.to_geodataframe("water_level", location="face")

# Clip to a polygon
clipped = ds.clip(polygon_mask)

# Reproject
reprojected = ds.to_crs(4326)
```

## Module Reference

| Module | Description |
|--------|-------------|
| [dataset](dataset.md) | `UgridDataset` main entry point |
| [mesh](mesh.md) | `Mesh2d` topology class |
| [connectivity](connectivity.md) | `Connectivity` array wrapper |
| [spatial](spatial.md) | Spatial indexing, clipping, subsetting |
| [interpolation](interpolation.md) | Mesh-to-grid interpolation |
| [io](io.md) | Topology detection and NetCDF I/O |
| [models](models.md) | Data model classes |
| [plot](plot.md) | Mesh visualization |
