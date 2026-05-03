# How it works

This overview explains the system boundaries and data flow of the pyramids package.

## System Context (C4: Context)

```mermaid
flowchart LR
  user(User) -->|Provides GIS data paths & commands| pyramids{{pyramids package}}
  ext1[(Raster files\nGeoTIFF/ASC/NetCDF)] --> pyramids
  ext2[(Vector files\nShapefile/GeoJSON/GeoPackage)] --> pyramids
  pyramids --> out1[(Processed rasters\nGeoTIFF/ASC)]
  pyramids --> out2[(Processed vectors\nGeoJSON/GPKG)]
```

## Runtime Containers (C4: Containers)

```mermaid
flowchart TB
  subgraph Process
    A[Dataset]:::c --> B[_io]:::c
    C[DatasetCollection]:::c --> B
    D[FeatureCollection]:::c --> B
    A --> E[_utils]
    C --> E
    D --> E
  end
  classDef c fill:#eef,stroke:#88f
```

## Components (C4: Components)

```mermaid
flowchart LR
  io[_io: read_file, to_ascii, path parsing]\nutils[_utils: geometry/index helpers]
  ds[dataset.Dataset]\nabs[dataset.abstract_dataset.RasterBase]
  dc[dataset.DatasetCollection]\nfc[feature.FeatureCollection]

  abs --> ds
  ds --> io
  dc --> ds
  fc --> io
  ds --> utils
  fc --> utils
```

## Data Flow

1. Input paths are parsed; archives (.zip/.gz/.tar) are handled in `_io`.
2. Raster inputs are loaded into `Dataset`; vector inputs into `FeatureCollection`.
3. `DatasetCollection` orchestrates collections of datasets for temporal/spatial ops.
4. Results are exported via `to_file` (GeoTIFF/ASCII/GeoJSON, etc.).

See the diagrams page for UML and sequence flows.
