# Spatial Operations

Spatial indexing (KD-tree, STRtree), point-in-face queries, mesh
clipping by polygon, and bounding box subsetting.

`MeshSpatialIndex` wraps a `Mesh2d` and builds its KD-trees / STRtree lazily on first query. The
module-level `clip_mesh` / `subset_by_bounds` functions use it to return a new `(Mesh2d, data)` pair —
they back `UgridDataset.clip` and `UgridDataset.subset_by_bounds`:

```mermaid
classDiagram
    class MeshSpatialIndex {
      +node_tree
      +face_tree
      +face_strtree
      +face_polygons
      +locate_nearest_node(x, y, k)
      +locate_nearest_face(x, y, k)
      +locate_nodes_in_bounds(...)
      +locate_faces_in_bounds(...)
      +locate_faces(x, y)
    }
    class Mesh2d
    MeshSpatialIndex o-- Mesh2d : mesh
    note for MeshSpatialIndex "node_tree / face_tree / face_strtree are lazy"
```

::: pyramids.netcdf.ugrid.MeshSpatialIndex
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids.netcdf.ugrid.spatial.clip_mesh
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

::: pyramids.netcdf.ugrid.spatial.subset_by_bounds
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
