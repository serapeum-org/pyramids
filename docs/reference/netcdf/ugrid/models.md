# UGRID Data Models

Dataclasses for UGRID topology metadata, mesh variables, and
dataset-level metadata summaries.

`UgridMetadata` is the dataset-level summary: it aggregates one `MeshTopologyInfo` per mesh and records
the variable **inventory** as a `data_variables: dict[str, str]` map (variable name → mesh location). The
`MeshVariable` records themselves are held by [`UgridDataset`](dataset.md), not by `UgridMetadata`. A
`MeshVariable` carries a lazy loader so its `data` is only materialised on access, and offers
time-slicing helpers:

```mermaid
classDiagram
    class UgridMetadata {
      data_variables : dict[str,str]
      global_attributes
      conventions
      n_nodes / n_faces / n_edges
    }
    class MeshTopologyInfo {
      mesh_name
      topology_dimension
      node_x_var / node_y_var
      face_node_var / edge_node_var
      crs_wkt
    }
    class MeshVariable {
      name · location · mesh_name
      shape · nodata · units
      +data
      +n_time_steps
      +sel_time(i)
      +sel_time_range(a, b)
      +with_data(arr)
    }
    UgridMetadata o-- MeshTopologyInfo : mesh_topologies
    note for UgridMetadata "data_variables is a name → location map, not MeshVariable records"
    note for MeshVariable "held by UgridDataset; data is loaded lazily on first access"
```

::: pyramids.netcdf.ugrid.MeshTopologyInfo
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids.netcdf.ugrid.MeshVariable
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids.netcdf.ugrid.UgridMetadata
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
