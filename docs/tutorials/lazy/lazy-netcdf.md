# Lazy NetCDF — `NetCDF`

NetCDFs are typically the heaviest single files in a GIS pipeline —
reanalysis, climate projections, ocean models. Pyramids exposes four
native lazy entry points covering single-file reads, multi-file
stacks, and zero-copy Zarr-backed cube indexing — **no xarray in the
lazy read path**, so installs stay slim and pyramids stays a peer of
xarray rather than a consumer:

| Entry point                              | Use when                                        |
|------------------------------------------|-------------------------------------------------|
| `NetCDF.read_array(chunks=…)`            | One file, one variable, partial reads           |
| `NetCDF.open_mfdataset(paths, variable)` | Many files, one variable, stacked on time axis  |
| `NetCDF.to_kerchunk(path)`               | Emit a JSON index so downstream reads skip HDF5 |
| `NetCDF.combine_kerchunk(paths, ...)`    | Combine per-file manifests into one cube index  |

See [lazy-compute.md](lazy-compute.md) for schedulers and `configure`
basics.

Installing the extras:

```bash
pip install 'pyramids-gis[lazy]'          # read_array + open_mfdataset
pip install 'pyramids-gis[netcdf-lazy]'   # + kerchunk manifests + h5py
```

## Reading one variable lazily

`NetCDF.read_array` preserves the legacy eager path on
`chunks=None`. Any other value flips to a dask-backed
`dask.array.Array` whose chunks are materialised on demand through
GDAL's MDArray reader:

```python
from pyramids.netcdf import NetCDF

nc = NetCDF.read_file("era5_t2m_2023.nc")

temp_lazy = nc.read_array("t2m", chunks={"time": 24, "lat": 256, "lon": 256})
temp_lazy.shape        # (8760, 721, 1440)
temp_lazy.chunks       # chunked tuples per axis
```

Chunk spec accepts the same forms as `Dataset.read_array`: `None`
(eager), `"auto"`, `-1` (single chunk), an int, a tuple, or a dict
keyed by axis index or axis name.

### Native chunks = best default

Every variable has a preferred on-disk chunk size exposed via
`VariableInfo.block_size`. The default `chunks="auto"` respects it,
so if the file was written with `chunksizes=(1, 256, 256)`, that's
what pyramids picks. Overriding only pays off when your access
pattern is diagonal to the on-disk layout (for example, reading a
time-series for one pixel over a time-chunked file).

### `unpack=True` applies scale/offset lazily

NetCDF variables often store packed ints with `scale_factor` /
`add_offset` CF attributes. `unpack=True` applies the transformation
in the dask graph — no materialisation:

```python
temp_unpacked = nc.read_array("t2m", unpack=True, chunks={"time": 24})
# temp_unpacked.dtype == float32, still lazy
```

### Locks

Same mechanism as `Dataset`: `lock=None` picks the right
`SerializableLock` or `dask.distributed.Lock` automatically.
`lock=False` opts into lock-free reads when you know each worker has
its own handle.

## Many files, one variable — `open_mfdataset`

Unlike `xarray.open_mfdataset`, the pyramids helper is deliberately
narrow: one variable at a time, no "by_coords" inference, no combine
strategies. For the common hydrology / meteorology case — 365 daily
`noah_YYYYMMDD.nc` files, stack `precipitation`, reduce on time —
that narrowness is the whole point.

```python
from pyramids.netcdf import NetCDF

stack = NetCDF.open_mfdataset(
    "/data/noah_*.nc",           # glob string
    variable="precipitation",
    chunks={"time": 1, "lat": 256, "lon": 256},
    parallel=True,               # fan out file opens over workers
)

stack.shape          # (n_files, H, W)
stack.mean(axis=0).compute()   # climatological mean
```

Three arguments worth knowing:

- `parallel=True` wraps each file's metadata read in `dask.delayed`
  so opening 500 files on a distributed cluster fans out across
  workers instead of blocking sequentially. The probe of the first
  file stays eager so the stack shape + dtype are known at
  graph-construction time.
- `chunks=` forwards to every per-file `read_array` call. Pick
  carefully: a chunk spec that doesn't align with per-file layout
  forces re-chunking on every read.
- `preprocess=` applies a user callable to each `NetCDF` subset
  before its array is extracted. Useful for unpacking, cropping, or
  dropping auxiliary variables.

```python
def only_europe(nc):
    return nc.crop(bbox=(-10.0, 35.0, 30.0, 65.0))

stack = NetCDF.open_mfdataset(
    "/data/era5/*.nc",
    variable="t2m",
    preprocess=only_europe,
    parallel=True,
)
```

`paths` accepts a glob string, an explicit path, or a list of paths.
Glob inputs are expanded and sorted alphabetically so the stack
order is deterministic.

## Zero-copy cube indexing — kerchunk

A kerchunk manifest is a JSON document containing byte-range pointers
into each source file; no pixel data is moved. Downstream consumers
open the resulting archive as a lazy Zarr-backed xarray with
**zero rewrite** — the speedup is the avoided decode cost.

Emit a manifest for a single file:

```python
from pyramids.netcdf import NetCDF

nc = NetCDF.read_file("era5_2023.nc")
manifest = nc.to_kerchunk(
    "era5_2023.kerchunk.json",
    inline_threshold=500,     # embed chunks smaller than 500 bytes
    vlen_encode="embed",      # embed VLEN strings for compatibility
)
```

Combine many manifests into one cube index:

```python
from pathlib import Path

srcs = sorted(Path("/data/era5").glob("era5_*.nc"))
manifest = NetCDF.combine_kerchunk(
    srcs,
    "era5_cube.json",
    concat_dims=("time",),
    identical_dims=("lat", "lon"),
)
```

Consume the manifest with fsspec + xarray:

```python
import fsspec
import xarray as xr

mapper = fsspec.get_mapper(
    "reference://",
    fo="era5_cube.json",
    remote_protocol="s3",
    remote_options={"anon": True},
)
ds = xr.open_zarr(mapper, consolidated=False)   # lazy, zero-copy
```

Kerchunk is behind `[netcdf-lazy]`. `to_kerchunk` and
`combine_kerchunk` raise a branded `ImportError` naming the extra
when it's absent.

### When to use kerchunk vs `open_mfdataset`

| You want…                                 | Reach for               |
|-------------------------------------------|-------------------------|
| One-off parallel read-and-reduce          | `open_mfdataset`        |
| A *reusable* index other tools can open   | `combine_kerchunk`      |
| Cloud data read through xarray/fsspec     | `combine_kerchunk`      |
| Only pyramids consumers                   | `open_mfdataset`        |

## Handing a pyramids lazy array to xarray

Pyramids deliberately does **not** register as an
``engine=`` for `xr.open_dataset`. It is a peer of xarray, not a
backend beneath it. If your pipeline ends in xarray anyway, use
`NetCDF.to_xarray()` for an eager conversion, or wrap pyramids' lazy
array in `xr.DataArray` yourself:

```python
import xarray as xr

from pyramids.netcdf import NetCDF

nc = NetCDF.read_file("era5_2023.nc")
arr = nc.read_array("t2m", chunks={"time": 24, "lat": 256, "lon": 256})
# arr is a dask.array.Array read via pyramids' CachingFileManager.

da = xr.DataArray(arr, dims=("time", "lat", "lon"), name="t2m")
```

You keep pyramids' `CachingFileManager` + `SerializableLock`
semantics (handle pooling, distributed-safe pickling) on the read
side, and xarray's ergonomics on the downstream side — no backend
adapter layer in between.

Install: `[netcdf-lazy]` pulls `kerchunk` + `h5py`; add the
`[xarray]` extra on top if you want `xr.DataArray` / `.to_xarray()` /
`.from_xarray()`.

## A worked pipeline — ERA5 on AWS

```python
from pyramids import configure
from pyramids.netcdf import NetCDF

configure(cloud_defaults=True, aws={"aws_unsigned": True})

nc = NetCDF.read_file(
    "https://era5-pds.s3.amazonaws.com/2020/01/data/air_temperature_at_2_metres.nc",
)

# 1. Check metadata cheaply.
nc.variable_names

# 2. Read one month, 6 hours at a time, 256x256 spatial chunks.
temp = nc.read_array(
    "air_temperature_at_2_metres",
    chunks={"time": 6, "lat": 256, "lon": 256},
)

# 3. Monthly mean on time axis — builds dask graph.
monthly = temp.mean(axis=0)

# 4. Materialise — single parallel compute.
monthly_np = monthly.compute()
```

The full notebook at
`docs/examples/netcdf/dask-lazy-netcdf.ipynb` adds plotting, writes
the output to a local Zarr store, and demonstrates the equivalent
kerchunk-based path.

## Minimal-install behaviour

Calls that need dask raise an `ImportError` naming the `[lazy]`
extra:

```text
ImportError: dask is required for lazy NetCDF reads; install pyramids-gis[lazy]
```

Calls that need kerchunk name `[netcdf-lazy]`. `NetCDF.to_xarray()` /
`.from_xarray()` name `[xarray]`. The eager
`NetCDF.read_array(chunks=None)` path is always available and does
not touch dask / xarray / kerchunk.

## UGRID unstructured grids

`UgridDataset` (FVCOM, SCHISM, ADCIRC, Delft3D) is not yet wired to
pyramids' lazy stack — it currently uses eager reads throughout.
Lazy UGRID is tracked as a follow-on. The
`docs/examples/netcdf/ugrid/dask-lazy-ugrid.ipynb` notebook shows how
to layer dask on top of the eager `UgridDataset` manually using
`dask.delayed` for FVCOM NECOFS OPeNDAP endpoints; that path works
today but isn't yet a first-class pyramids API.
