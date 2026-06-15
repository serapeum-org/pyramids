# Kerchunk reference manifests

## What kerchunk is, in one line

**Kerchunk lets you read an existing archival file (NetCDF / HDF5 / GRIB / TIFF) *as if it were a cloud-native
Zarr store* — without rewriting or converting the data.** It does this by producing a small JSON "reference
manifest" that maps Zarr chunk keys to byte ranges inside the original file(s).

pyramids emits these manifests from `NetCDF` and `DatasetCollection`; this page explains what they are, why
they are useful, and how to generate them with the pyramids API.

## The problem it solves

Formats like NetCDF4 / HDF5 are excellent archival formats but awkward for cloud or lazy access:

- To read one variable, a client traditionally has to open the whole file and walk HDF5's internal B-tree
  layout.
- Over object storage (S3 / GCS / Azure) or plain HTTP, that means many small range requests driven by HDF5
  internals — slow and chatty.

[Zarr](https://zarr.dev/), by contrast, is designed for the cloud: each chunk is addressable independently
(`var/0.0.1`), metadata is plain JSON, and a client can fetch exactly the chunks it needs with simple `GET`
range requests.

Kerchunk bridges the two. It scans the HDF5 file **once**, records where every chunk physically lives
(`[file_url, byte_offset, length]`), and writes that out as a Zarr-shaped JSON document. Any Zarr-aware
consumer then opens the *manifest* and reads chunks straight out of the original file by byte range — **no
data is copied and no conversion happens.**

## What a manifest looks like

A kerchunk v1 manifest is a JSON document. The interesting part is `refs`, mapping Zarr store keys to either
metadata strings or `[url, offset, length]` byte-range pointers:

```json
{
  "version": 1,
  "refs": {
    ".zgroup": "{\"zarr_format\":2}",
    "values/.zarray": "{...shape, chunks, dtype, compressor...}",
    "values/.zattrs": "{\"_ARRAY_DIMENSIONS\":[\"bands\",\"y\",\"x\"]}",
    "values/0.0.0": ["data.nc", 14068, 2184]
  }
}
```

That last line says: *the chunk at grid position `0.0.0` of `values` lives in `data.nc`, at byte offset
`14068`, and is `2184` bytes long.* Small chunks are inlined directly (base64) instead of referenced, so a
fully self-contained manifest is possible for tiny arrays.

## Generating a manifest with pyramids

### Single file — `NetCDF.to_kerchunk`

```python
from pyramids.netcdf import NetCDF

nc = NetCDF.read_file("noah_20240101.nc")
manifest = nc.to_kerchunk("noah_20240101.kerchunk.json")

# The manifest is also returned as a dict for inspection
print(manifest["version"])                 # 1
print(sorted(manifest["refs"])[:4])         # ['.zattrs', '.zgroup', 'values/.zarray', 'values/.zattrs']
print(manifest["refs"]["values/0.0.0"])     # ['noah_20240101.nc', 14068, 2184]
```

Useful keyword arguments:

- `inline_threshold` (default `500`): chunks smaller than this many bytes are embedded directly in the
  manifest (base64) rather than referenced by offset. Raise it to make a more self-contained manifest; lower
  it to keep the JSON tiny.
- `vlen_encode` (default `"embed"`): how variable-length (VLEN) strings are handled.

### Many files into one virtual cube — `NetCDF.combine_kerchunk`

The headline use case: turn a directory of per-timestep NetCDFs into a **single** lazy, cloud-readable dataset
described by one tiny sidecar JSON.

```python
from pathlib import Path
from pyramids.netcdf import NetCDF

sources = sorted(str(p) for p in Path("/data/noah").glob("noah_*.nc"))

manifest = NetCDF.combine_kerchunk(
    sources,
    "noah_2024.kerchunk.json",
    concat_dims=("time",),         # stack every variable that carries `time`
    identical_dims=("lat", "lon"), # shared across files; taken from the first
)
```

Every variable whose dimensions include the concat dimension is stacked along it (its shape grows to the sum
across files); variables without it (e.g. the `lat` / `lon` coordinates) are carried through unchanged.

### A collection — `DatasetCollection.to_kerchunk`

If you already manage a folder of co-registered files as a `DatasetCollection`, it can emit the combined
manifest directly:

```python
from pyramids.dataset import DatasetCollection

collection = DatasetCollection.from_files(sorted(Path("/data/noah").glob("noah_*.nc")))
manifest = collection.to_kerchunk("noah_2024.kerchunk.json", concat_dim="time")
```

## How the manifest is built (and why it is zarr-v3-safe)

There are two distinct roles in the kerchunk workflow, and it helps to keep them separate:

1. **Generating** the manifest — scanning the source files and recording chunk byte ranges. This is what
   pyramids does.
2. **Consuming** the manifest — opening it lazily and reading chunks. This is done by Zarr-aware clients.

pyramids generates manifests **natively**: it walks the HDF5 container with `h5py` and writes the Zarr-v2
metadata and chunk references directly, without ever instantiating a live Zarr group. This is deliberate —
the older approach of driving kerchunk's HDF5 translator through Zarr's machinery could deadlock under
Zarr v3 during *generation* (see issue #530). The native builder side-steps that entirely, so
`to_kerchunk` / `combine_kerchunk` are safe to call anywhere, including inside long-lived processes and on CI.

For files that use HDF5 features the native builder does not cover (for example VLEN/compound dtypes), it
automatically falls back to the kerchunk translator and emits a warning.

## Inspecting a manifest

Because the functions return the manifest as a plain dict, you can inspect it with no extra dependencies:

```python
from pyramids.netcdf import NetCDF

manifest = NetCDF.read_file("noah_20240101.nc").to_kerchunk("noah.json")
refs = manifest["refs"]

# Which variables are described?
variables = sorted(key[: -len("/.zarray")] for key in refs if key.endswith("/.zarray"))
print(variables)                                  # ['bands', 'values', 'x', 'y', ...]

# Where does a given chunk live?
print(refs["values/0.0.0"])                       # ['noah_20240101.nc', 14068, 2184]

# Read a variable's Zarr metadata (it is a JSON string)
import json
print(json.loads(refs["values/.zarray"])["shape"])   # [3, 13, 14]
print(json.loads(refs["values/.zattrs"]))             # {'_ARRAY_DIMENSIONS': ['bands', 'y', 'x'], ...}
```

## Consuming a manifest

The manifest is a *Zarr reference store*, so it is opened by any Zarr-aware reader pointed at it through an
fsspec "reference" filesystem. The canonical consumer in the scientific-Python stack is xarray
(`xr.open_dataset(manifest, engine="kerchunk")`), but the manifest is just data — anything that speaks the
Zarr v2 reference spec can read it. pyramids' role is to **produce** correct manifests; reading the underlying
arrays directly (rather than through a manifest) is what `NetCDF.read_file` / `NetCDF.read_array` already do.

## Why the name "kerchunk"?

It is a coined name with no acronym — it evokes the *ker-chunk* of slotting data into chunks. The broader
ecosystem has since standardized the idea as **"Zarr virtual datasets"** (the Zarr reference specification),
with newer front-ends such as VirtualiZarr, but the manifest format pyramids emits is the original kerchunk
v1 document.

## Summary

- Kerchunk is a **virtualization layer**: it makes legacy scientific files behave like cloud-native Zarr via a
  **byte-range index**, with no data movement.
- pyramids generates manifests with `NetCDF.to_kerchunk` (single file),
  `NetCDF.combine_kerchunk` (many files → one cube), and `DatasetCollection.to_kerchunk`.
- Generation is native and Zarr-v3-safe; the returned dict can be inspected directly.
- A few-kilobyte JSON can stand in for a directory of multi-gigabyte NetCDFs, opened lazily by any Zarr-aware
  consumer.
