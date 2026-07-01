# NetCDF Data Models

Immutable dataclasses for NetCDF metadata: variables, dimensions,
groups, CF info, and the top-level `NetCDFMetadata` container.

`NetCDFMetadata` is the aggregate the [metadata extractor](metadata.md) builds. It composes the
per-element info records below — one `VariableInfo` per array, one `DimensionInfo` per dimension, a
`GroupInfo` tree, a single `StructuralInfo` (driver-level) and a single `CFInfo` (CF cross-reference):

```mermaid
classDiagram
    class NetCDFMetadata {
      driver
      root_group
      global_attributes
      +get_dimension(name)
    }
    class GroupInfo {
      name
      full_name
      attributes
      children
      variables
    }
    class VariableInfo {
      name
      dtype
      shape
      dimensions
      unit
      nodata
      scale / offset
      srs_wkt / srs_projjson
      coordinate_variables
    }
    class DimensionInfo {
      name
      size
      type
      direction
      indexing_variable
    }
    class StructuralInfo {
      driver_name
      driver_metadata
    }
    class CFInfo {
      cf_version
      conventions
      classifications
      grid_mappings
      bounds_map
    }
    NetCDFMetadata o-- GroupInfo : groups
    NetCDFMetadata o-- VariableInfo : variables
    NetCDFMetadata o-- DimensionInfo : dimensions
    NetCDFMetadata o-- StructuralInfo : structural
    NetCDFMetadata o-- CFInfo : cf
    GroupInfo o-- GroupInfo : children
```

::: pyramids.netcdf.models.NetCDFMetadata
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids.netcdf.models.VariableInfo
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids.netcdf.models.DimensionInfo
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids.netcdf.models.GroupInfo
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids.netcdf.models.CFInfo
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids.netcdf.models.StructuralInfo
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
