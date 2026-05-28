# Zarr

pyramids reads and writes **Zarr v3** stores for both single rasters
(`Dataset`) and time-stacked cubes (`DatasetCollection`). On disk, the layout
follows the CF / GeoZarr convention so a pyramids store opens georeferenced in
any standards-aware reader (rioxarray, odc-geo, `xarray.open_zarr`, GDAL's
Zarr driver) without needing pyramids in the loop. Conversely, the reader is
tolerant: it opens GeoZarr stores written by other tools too.

> **Requires the `[lazy]` extra** (`zarr>=3`, `dask`, `fsspec`).
> `[netcdf-lazy]` adds `kerchunk>=0.2.10` for NetCDF→Zarr reference manifests.

## Quick start

```python
import numpy as np
from pyramids.dataset import Dataset

arr = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
ds = Dataset.create_from_array(arr, top_left_corner=(0.0, 8.0),
                               cell_size=1.0, epsg=4326)
ds.band_names = ["red", "nir"]
ds.to_file("source.tif")

# Write — parallel chunk writes, GeoZarr layout, default Blosc codec
Dataset.read_file("source.tif").to_zarr("out.zarr")

# Read — round-trips values, CRS, GeoTransform, nodata, band names
rt = Dataset.from_zarr("out.zarr")
assert rt.epsg == 4326
assert rt.band_names == ["red", "nir"]
```

## On-disk layout (CF / GeoZarr)

Every pyramids-written store carries:

| Array          | Shape          | Role                                              |
|----------------|----------------|---------------------------------------------------|
| `data`         | `(B, R, C)` or `(T, B, R, C)` for cubes | the raster array            |
| `x`            | `(C,)`         | pixel-centre x coordinates                        |
| `y`            | `(R,)`         | pixel-centre y coordinates                        |
| `spatial_ref`  | `()` (scalar)  | grid-mapping variable carrying `crs_wkt` + `GeoTransform` |

…plus the standard attrs:

- `data.attrs["_ARRAY_DIMENSIONS"]` and `dimension_names` field (xarray v2 + v3
  compat) — `["band","y","x"]` for `Dataset`, `["time","band","y","x"]` for cubes.
- `data.attrs["grid_mapping"] = "spatial_ref"` (CF link to the CRS variable).
- `data.attrs` also keeps the pyramids round-trip metadata: `no_data_value`,
  `band_names`, `dtype`, `epsg`, `GeoTransform` (redundant with `spatial_ref` for
  legacy readers).
- Root attrs: `pyramids_zarr_version = "2"`; cubes also carry `time_length` and
  `pyramids_file_list`; multiscale stores carry the OGC `multiscales` block
  (see below).

`Dataset.from_zarr` recovers the geobox from `spatial_ref` (preferring its WKT,
falling back to EPSG, then to deriving the transform from `x`/`y` if those are
present but `GeoTransform` is absent — for foreign stores).

## Single-raster API: `Dataset`

### Write — `Dataset.to_zarr`

```python
ds.to_zarr(
    "out.zarr",
    *,
    compute=True,           # False -> dask.delayed
    mode="w",               # "w" overwrite; "a" requires append_dim/region on a cube
    chunks=None,            # tuple (B, R, C) or "auto"; controls dask chunking
    storage_options=None,   # fsspec options for s3:// / gs:// / ...
    compressor="auto",      # "auto" = default; None = uncompressed;
                            # a v3 codec (or list) — e.g. zarr.codecs.BloscCodec(cname="zstd")
    overview_factors=None,  # e.g. [2, 4, 8] to also write decimated pyramid levels
    overview_resampling="average",
)
```

### Read — `Dataset.from_zarr`

```python
ds = Dataset.from_zarr(
    "out.zarr",
    *,
    chunks=None,            # tuple or "auto" -> dask-parallel chunked read
    storage_options=None,
    level=1,                # pyramid factor; 1 = full res, 2/4/... = overview
    data_name=None,         # override the array name for foreign GeoZarr stores
)
```

## Cube API: `DatasetCollection`

### Write — `DatasetCollection.to_zarr`

```python
col = DatasetCollection.from_files(["t0.tif", "t1.tif", "t2.tif"])

col.to_zarr(
    "cube.zarr",
    *,
    compute=True,
    mode="w",
    storage_options=None,
    compressor="auto",
    append_dim=None,        # "time" to append new timesteps to an existing cube
    region=None,            # {"time": slice(a, b), ...} to overwrite a slice
)
```

### Read — `DatasetCollection.from_zarr`

```python
col = DatasetCollection.from_zarr("cube.zarr")
col.time_length      # int
col.data             # lazy dask.array (T, B, R, C) read straight from the store
col.mean()           # time reduction without materialising the whole cube
```

## Incremental cube writes (`append_dim` / `region`)

```python
# 1) initial write
DatasetCollection.from_files(jan_files).to_zarr("cube.zarr")

# 2) append later timesteps along time
DatasetCollection.from_files(feb_files).to_zarr(
    "cube.zarr", mode="a", append_dim="time",
)

# 3) overwrite a specific time slice (e.g. fix a bad month)
DatasetCollection.from_files(fixed_march).to_zarr(
    "cube.zarr", mode="a", region={"time": slice(2, 3)},
)
```

`mode="a"` without `append_dim` or `region` raises — there is no silent "open
without overwriting" mode.

## Codec / compression control

`compressor=` is forwarded to zarr v3's `create_array(compressors=...)`. A
single codec is wrapped in a list automatically; `None` means **uncompressed**;
`"auto"` keeps zarr's default codec.

```python
from zarr.codecs import BloscCodec
ds.to_zarr("zstd.zarr", compressor=BloscCodec(cname="zstd", clevel=5))
ds.to_zarr("raw.zarr",  compressor=None)
```

## Multiscale pyramid levels

```python
ds.to_zarr("ms.zarr", overview_factors=[2, 4, 8])
# adds data_2 / data_4 / data_8 plus an OGC/OME-Zarr `multiscales` attribute

# read a level back (cell size scales by the factor)
preview = Dataset.from_zarr("ms.zarr", level=4)
# preview.cell_size == ds.cell_size * 4
```

The `multiscales` attribute follows the OGC / OME-Zarr v0.4 layout
(list of multiscale defs with `axes` + `datasets[].coordinateTransformations`
of type `scale`) so GDAL's Zarr v3 driver also exposes the levels as overviews.

## Foreign GeoZarr stores

The reader auto-detects the primary data array (preferring `"data"`, then any
array with a `grid_mapping` attr, then the highest-dim non-coordinate array),
follows the CF `grid_mapping` link to the CRS variable (defaulting to
`"spatial_ref"`), and falls back to deriving the transform from `x` / `y`
coordinates when `GeoTransform` is absent — so stores written by rioxarray,
odc-geo, or GDAL's Zarr driver open without pyramids in the loop.

For ambiguous foreign stores you can pin the array explicitly:

```python
Dataset.from_zarr("some-foreign.zarr", data_name="elevation")
```

## STAC

`pyramids.stac.load_asset` routes Zarr STAC assets through the shared reader:
a 4-D cube becomes a lazy `DatasetCollection`, anything lower-dimensional a
`Dataset` — the foreign-GeoZarr tolerance above means non-pyramids stores load
the same way.

## Cloud stores

Local paths and URLs are passed straight to zarr v3, which resolves
`file://`, `s3://`, `gs://`, `az://`, … via fsspec. For URL stores that need
credentials, pass them through `storage_options=`:

```python
ds = Dataset.from_zarr(
    "s3://my-bucket/elev.zarr",
    storage_options={"anon": True},
)
```

`storage_options` is plumbed to a v3 `zarr.storage.FsspecStore.from_url(...)`
so the options reach fsspec correctly.

## Legacy stores

Stores written before this layout (geo-referencing as flat attributes on the
`data` array, no `spatial_ref` coordinate) still open. The reader emits a
`DeprecationWarning` once and recovers everything that's there. Re-save with
`to_zarr` to migrate the on-disk layout to the GeoZarr convention; legacy
reading will be removed in a future release.

## See also

- [`DatasetCollection`](dataset_collection.md) — the cube API the zarr writer/reader build on.
- The architecture review in `planning/zarr/zarr-architecture-review.md` for the design rationale.
