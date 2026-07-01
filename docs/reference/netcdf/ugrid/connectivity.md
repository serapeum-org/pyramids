# Connectivity

Wrapper for UGRID connectivity arrays that handles `start_index`
normalization (always 0-indexed internally) and `_FillValue`
masking for mixed-element meshes.

`from_gdal_array` reads the raw array (which may be 1-based per UGRID) and normalizes it to a 0-indexed,
fill-masked internal form; the accessors then read individual elements or a fully masked view:

```mermaid
flowchart LR
    A[("GDAL MDArray<br/>start_index 0/1<br/>_FillValue")]
    C["Connectivity<br/>0-indexed, fill-masked"]
    A -->|from_gdal_array| C
    C -->|"get_element(i)"| E["node indices of element i"]
    C -->|as_masked| M["masked array<br/>(ragged / mixed-element)"]
    C -->|is_triangular| T{"3 nodes each?"}
```

::: pyramids.netcdf.ugrid.Connectivity
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
