# Dimension Parsing

Dimension metadata extraction, time coordinate parsing, and
variable-dimension relationship handling for NetCDF files.

This module parses the classic-mode `NETCDF_DIM_*` metadata GDAL exposes on non-multidimensional
opens. `parse_gdal_netcdf_dimensions` builds a `DimensionsIndex` of `ClassicDimensionInfo` records;
`ClassicDimMetadata` pairs that index with the per-dimension attribute dicts, and each
`ClassicDimensionInfo` can convert to the MDim-mode [`DimensionInfo`](models.md) model:

```mermaid
classDiagram
    class ClassicDimMetadata {
      +names
      +get_attrs(name)
      +get_dimension(name)
      +iter_dimensions()
      +from_metadata()$
    }
    class DimensionsIndex {
      +names
      +to_dict()
      +to_metadata()
      +from_metadata()$
    }
    class ClassicDimensionInfo {
      name
      size
      values
      def_fields
      attrs
      +to_dimension_info()
    }
    class DimensionInfo
    ClassicDimMetadata *-- DimensionsIndex
    DimensionsIndex o-- ClassicDimensionInfo
    ClassicDimensionInfo ..> DimensionInfo : to_dimension_info()
```

::: pyramids.netcdf.dimensions
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
