# Mesh2d

2D unstructured mesh topology class. Holds node coordinates,
face/edge connectivity arrays, and lazy-computed geometric
properties (centroids, areas, triangulation).

`Mesh2d` is pure NumPy plus up to five `Connectivity` arrays — it holds no GDAL handles. Geometric
properties (`face_centroids`, `face_areas`, `fan_triangles`) are computed on first access and cached:

```mermaid
classDiagram
    class Mesh2d {
      +node_x / node_y
      +face_node_connectivity : Connectivity
      +edge_node_connectivity : Connectivity
      +face_edge_connectivity : Connectivity
      +face_face_connectivity : Connectivity
      +edge_face_connectivity : Connectivity
      +n_node / n_face / n_edge
      +bounds
      +face_centroids
      +face_areas
      +fan_triangles
      +get_face_polygon(i)
      +build_edge_connectivity()
      +from_gdal_group(rg, topo)$
    }
    class Connectivity {
      data · fill_value · cf_role
      +n_elements
      +get_element(i)
      +is_triangular()
      +as_masked()
    }
    Mesh2d *-- Connectivity
    note for Mesh2d "face_centroids / face_areas / fan_triangles are lazy + cached"
```

::: pyramids.netcdf.ugrid.Mesh2d
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
