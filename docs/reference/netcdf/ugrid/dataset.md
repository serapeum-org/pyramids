# UgridDataset

Top-level container for UGRID NetCDF mesh data. Provides the
user-facing API for reading, writing, inspecting, and operating
on unstructured mesh data.

`UgridDataset` is a thin, GIS-aware facade: its public methods delegate the heavy lifting to the
sibling modules of the subpackage (spatial ops, interpolation, I/O, plotting):

```mermaid
flowchart LR
    U["UgridDataset"]
    U -->|"clip · subset_by_bounds"| SP["spatial.clip_mesh<br/>spatial.subset_by_bounds"]
    U -->|"to_dataset"| IN["interpolation.mesh_to_grid"]
    U -->|"read_file · to_file"| IO["io.parse_ugrid_topology<br/>io.write_ugrid_topology"]
    U -->|"plot · plot_outline"| PL["plot.plot_mesh_data<br/>plot.plot_mesh_outline"]
    U -->|"metadata"| MD["UgridMetadata"]
```

See the [subpackage overview](index.md) for how `UgridDataset` composes `Mesh2d`, `Connectivity`, and
`MeshVariable`.

::: pyramids.netcdf.ugrid.UgridDataset
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
