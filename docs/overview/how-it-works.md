# How it works

This overview explains the system boundaries and data flow of the pyramids package.

## System Context (C4: Context)

```mermaid
flowchart LR
  user(User) -->|Provides GIS data paths & commands| pyramids{{pyramids package}}
  ext1[(Raster files\nGeoTIFF/ASC/NetCDF)] --> pyramids
  ext2[(Vector files\nShapefile/GeoJSON/GeoPackage)] --> pyramids
  ext3[(UGRID NetCDF\nunstructured meshes)] --> pyramids
  ext4[(Cloud / archive\ns3:// · gs:// · az:// · zip · gzip · tar)] --> pyramids
  pyramids --> out1[(Processed rasters\nGeoTIFF · COG · ASC)]
  pyramids --> out2[(Processed vectors\nGeoJSON · GPKG)]
  pyramids --> out3[(Lazy stacks\nZarr · kerchunk JSON)]
```

## Runtime Containers (C4: Containers)

```mermaid
flowchart TB
  subgraph Process["pyramids process"]
    subgraph Base["pyramids.base"]
      CRS[crs]:::b
      FM[_file_manager]:::b
      DOM[_domain]:::b
      META[_raster_meta]:::b
      REMOTE[remote]:::b
      IO[_io]:::b
    end

    subgraph Raster["pyramids.dataset"]
      DS[Dataset]:::c
      DC[DatasetCollection]:::c
      Engines[engines.IO · Spatial · Bands · Analysis · Cell · Vectorize · COG]:::e
    end

    subgraph NC["pyramids.netcdf"]
      NCD[NetCDF]:::c
      UG[UgridDataset]:::c
    end

    subgraph Vec["pyramids.feature"]
      FC[FeatureCollection]:::c
    end

    DS --> Base
    DC --> Base
    NCD --> Base
    UG --> Base
    FC --> Base
    DS --> Engines
    NCD --> DS
    DC --> DS
  end
  classDef c fill:#eef,stroke:#88f
  classDef b fill:#efe,stroke:#8a8
  classDef e fill:#fee,stroke:#c88
```

## Components (C4: Components)

```mermaid
flowchart LR
  subgraph PB["pyramids.base"]
    crs[crs: sr_from_epsg · sr_from_wkt · reproject_coordinates]
    fm[_file_manager: CachingFileManager · FILE_CACHE LRU]
    dom[_domain: is_no_data · inside_domain]
    meta[_raster_meta: RasterMeta]
  end
  subgraph PD["pyramids.dataset"]
    abs[abstract_dataset.RasterBase]
    ds[Dataset]
    dc[DatasetCollection]
    rops[ops: _focal · _zarr · _zonal · vectorize · io · reproject]
    eng[engines: IO · Spatial · Bands · Analysis · Cell · Vectorize · COG]
    merge[merge.merge_rasters]
    redop[_reduce_ops.resolve_dask_op]
  end
  subgraph PN["pyramids.netcdf"]
    nc[NetCDF]
    cf[cf]
    lazy[_lazy._apply_unpack]
    ugds[ugrid.UgridDataset]
  end
  subgraph PF["pyramids.feature"]
    fc[FeatureCollection]
    coords[geometry: Coords · GeometryCoords · create_polygon · create_point]
  end
  pyio[_io: zip · gzip · tar · /vsi-rewrite]

  abs --> ds
  ds --> nc
  ds --> eng
  ds --> rops
  dc --> ds
  dc --> redop
  dc --> merge
  nc --> lazy
  nc --> cf
  ugds --> ds
  fc --> coords
  ds --> crs
  fc --> crs
  ds --> fm
  dc --> fm
  ds --> dom
  dc --> dom
  dc --> meta
  pyio --> ds
  pyio --> nc
  pyio --> fc
```

## Data Flow

1. Input paths are normalised — archives (`.zip`/`.gz`/`.tar`) and remote
   URLs (`s3://`, `gs://`, `az://`, `http(s)://`) are rewritten to GDAL
   virtual filesystem paths in `pyramids._io` /
   `pyramids.base.remote`.
2. Raster inputs become `Dataset` (GeoTIFF, ASC) / `NetCDF` (NetCDF, HDF5) /
   `UgridDataset` (UGRID NetCDF) instances; vector inputs become
   `FeatureCollection`. Each concrete class composes engines for its
   public-API families (`ds.spatial`, `ds.io`, …).
3. `DatasetCollection` wraps `N` co-registered `Dataset` instances into a
   lazy `(T, B, R, C)` dask cube. Per-timestep ops (`crop`, `to_crs`,
   `align`, `apply`) loop the per-step gdal handles; time-axis reductions
   (`mean / sum / std / groupby`) run through `_reduce_ops.resolve_dask_op`
   with optional `flox` acceleration.
4. Results are exported via `to_file` (GeoTIFF / ASCII), `to_cog` (COG),
   `to_zarr` (chunked + metadata), `to_kerchunk` (NetCDF/HDF5 sidecar),
   `merge` (mosaic), or vectorized into `FeatureCollection`.

See the diagrams page for UML and sequence flows.
