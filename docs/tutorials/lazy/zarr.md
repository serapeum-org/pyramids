# What is Zarr (and why pyramids writes it)

## Zarr in one line

**Zarr is a storage format for large, chunked, compressed N-dimensional arrays — designed so each chunk is an
independently addressable object and all metadata is plain JSON.** That makes it fast to read partially, cheap
to write in parallel, and native to cloud object storage (S3 / GCS / Azure).

pyramids reads and writes Zarr v3 stores for both single rasters (`Dataset`) and time-stacked cubes
(`DatasetCollection`). This page explains the concept; for the full argument-by-argument contract see the
[Zarr reference](../../reference/zarr.md).

## The idea: arrays split into chunks

A Zarr array is not one big file. It is:

- a small JSON metadata document describing the array's shape, chunk shape, dtype, fill value, and codecs; plus
- one object per **chunk** — a fixed-size block of the array (e.g. a `256 × 256` tile), each compressed
  independently.

```text
my_array.zarr/
├── zarr.json            # shape=(2,1024,1024), chunks=(1,256,256), dtype=float32, codec=blosc
├── c/0/0/0              # chunk (band 0, rows 0–255,   cols 0–255)
├── c/0/0/1              # chunk (band 0, rows 0–255,   cols 256–511)
└── ...                  # one object per chunk
```

Because chunks are separate objects, a reader that wants one corner of a huge array fetches only the few chunks
it overlaps — not the whole file. And a writer can produce many chunks **at once**, in parallel, with no
central index to serialize through.

## What it is used for

- **Arrays bigger than RAM** — read and process a window or a few bands without loading everything.
- **Parallel / distributed I/O** — every Dask chunk maps to one Zarr chunk, so writes fan out across cores or a
  cluster. In pyramids, Zarr is the *only* raster output path that writes in true parallel.
- **Cloud-native access** — a Zarr store on S3 is just a prefix of objects; clients do simple `GET`s for the
  chunks they need, no special server.
- **Incremental datasets** — append new timesteps to a cube, or overwrite a single time slice (a "region"
  write), without rewriting the whole store.
- **Fast previews** — multiscale pyramid levels let a viewer grab a decimated overview instead of full
  resolution.

## How Zarr relates to HDF5/NetCDF and to kerchunk

- **vs HDF5 / NetCDF:** same goal (chunked N-D arrays) but HDF5 packs everything into one file with an internal
  index, which is awkward over object storage. Zarr spreads chunks across many objects with JSON metadata —
  built for the cloud.
- **vs [kerchunk](kerchunk.md):** kerchunk does *not* move data — it writes a tiny manifest that makes an
  existing NetCDF/HDF5 file **look like** a Zarr store by pointing at byte ranges inside it. `to_zarr` is the
  opposite: it writes a **real, new** Zarr store with the data copied and re-chunked. Use kerchunk to virtualize
  archives in place; use `to_zarr` when you want an independent, re-chunked, cloud-optimized copy.

## pyramids examples

> All Zarr I/O needs the `[lazy]` extra (`zarr>=3`, `dask`, `fsspec`).

### Single raster — `Dataset.to_zarr` / `from_zarr`

```python
import numpy as np
from pyramids.dataset import Dataset

arr = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
ds = Dataset.create_from_array(arr, top_left_corner=(0.0, 8.0), cell_size=1.0, epsg=4326)
ds.band_names = ["red", "nir"]

# Write — parallel chunk writes, GeoZarr layout (CRS + GeoTransform preserved)
ds.to_zarr("scene.zarr")

# Read back — values, CRS, geotransform, nodata, and band names all round-trip
rt = Dataset.from_zarr("scene.zarr")
print(rt.epsg)          # 4326
print(rt.band_names)    # ['red', 'nir']
```

The store follows the CF / **GeoZarr** convention, so it also opens georeferenced in other standards-aware
readers (GDAL's Zarr driver, `xarray.open_zarr`, rioxarray) without pyramids in the loop.

### Time-stacked cube — `DatasetCollection.to_zarr` / `from_zarr`

```python
from pathlib import Path
from pyramids.dataset import DatasetCollection

cube = DatasetCollection.from_files(sorted(Path("/data/series").glob("*.tif")))   # 4-D (T, B, R, C)
cube.to_zarr("series.zarr")

# Reopen lazily — data is read straight from the store via dask, not into memory
reopened = DatasetCollection.from_zarr("series.zarr")
print(reopened.time_length)
```

### Incremental writes — append a timestep / overwrite a slice

```python
# initial write
cube_jan.to_zarr("series.zarr")

# append later timesteps along the time axis (no full rewrite)
cube_feb.to_zarr("series.zarr", mode="a", append_dim="time")

# overwrite one existing time slice (e.g. fix a bad month) — an xarray-style region write
cube_fixed.to_zarr("series.zarr", mode="a", region={"time": slice(1, 2)})
```

### Multiscale pyramid levels (fast previews)

```python
# write full resolution plus decimated overviews: data_2 / data_4 / data_8
ds.to_zarr("scene.zarr", overview_factors=[2, 4, 8])

# read a coarse level back — its cell size scales by the factor
preview = Dataset.from_zarr("scene.zarr", level=4)
print(preview.cell_size == ds.cell_size * 4)   # True
```

### Cloud store + codec control

```python
import zarr

# write to S3 with explicit fsspec options and a zstd Blosc codec
ds.to_zarr(
    "s3://my-bucket/scene.zarr",
    compressor=zarr.codecs.BloscCodec(cname="zstd"),
    storage_options={"anon": False},
)

# read it back the same way
remote = Dataset.from_zarr("s3://my-bucket/scene.zarr", storage_options={"anon": True})
```

## Summary

- Zarr stores a large array as **many independently compressed chunk objects + JSON metadata**, which makes
  partial reads, parallel writes, and cloud access natural.
- pyramids writes/reads Zarr v3 in GeoZarr layout: `Dataset.to_zarr` / `from_zarr` for single rasters,
  `DatasetCollection.to_zarr` / `from_zarr` for cubes, with append/region writes and multiscale overviews.
- Reach for `to_zarr` when you want a real re-chunked cloud-optimized copy; reach for
  [kerchunk](kerchunk.md) when you want to virtualize existing files without copying.

## See also

- [Zarr reference](../../reference/zarr.md) — full API, on-disk layout, foreign GeoZarr stores, STAC, legacy
  stores.
- [Kerchunk reference manifests](kerchunk.md) — virtualize NetCDF/HDF5 as Zarr without copying.
- Example notebooks: *Zarr basics*, *Zarr cubes — append & region*, *Multiscale pyramids*, *Foreign GeoZarr
  stores* (under Examples → Zarr).
