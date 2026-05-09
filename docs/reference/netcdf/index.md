# NetCDF Class

The `NetCDF` class extends `Dataset` for structured (regular grid)
NetCDF files. It wraps GDAL's Multidimensional API to provide
variable access, time dimension handling, and CF-compliant metadata.

## Lazy / Dask reads

Every NetCDF entry point has a lazy variant that keeps memory bounded
on multi-GB reanalysis and climate-projection files:

| Entry point                              | Purpose                                        |
|------------------------------------------|------------------------------------------------|
| `NetCDF.read_array(chunks=…)`            | One file, one variable, partial reads          |
| `NetCDF.open_mfdataset(paths, variable)` | Many files → single stacked dask array         |
| `NetCDF.to_kerchunk(path)`               | Emit a JSON index so downstream reads are free |
| `NetCDF.combine_kerchunk(paths, …)`      | Combine per-file manifests into one cube index |
| `NetCDF.to_xarray()` / `.from_xarray()`  | Round-trip interop with `xarray.Dataset`       |

```python
from pyramids.netcdf import NetCDF

nc = NetCDF.read_file("era5.nc")
t2m = nc.read_array(
    "t2m", chunks={"time": 24, "lat": 256, "lon": 256},
)
t2m.mean(axis=0).compute()        # monthly mean, parallel
```

See [Lazy NetCDF](../../tutorials/lazy/lazy-netcdf.md) for chunk-size rules,
CF scale/offset unpacking, and kerchunk manifest emission.

Install: `pip install 'pyramids-gis[lazy]'` for the core path,
`[netcdf-lazy]` for kerchunk, `[xarray]` for the `to_xarray` /
`from_xarray` round-trip helpers.

::: pyramids.netcdf.NetCDF
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
