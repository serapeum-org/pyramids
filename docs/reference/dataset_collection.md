# DataCube Class

![Datacube diagram](./../_images/pyramids-multi-dataset.svg)

## Lazy time-series cubes

`DatasetCollection` stacks per-file reads into a 4-D
`(T, B, R, C)` lazy `dask.array.Array` exposed as `collection.data`.
Workers never serialise a live `gdal.Dataset` handle — each timestep
opens on demand through pyramids' `CachingFileManager`.

```python
from pyramids.dataset import DatasetCollection

cube = DatasetCollection.from_files(sorted_tifs)
cube.data                   # dask.array.Array, (T, B, R, C)
cube.mean(skipna=True)      # numpy array (B, R, C)
cube.groupby(months).mean() # dict {month: ndarray}
cube.to_zarr("out.zarr")    # parallel write
cube.to_kerchunk("idx.json")  # NetCDF/HDF5 only
```

See [Lazy collections](../tutorials/lazy/lazy-collection.md) for construction
(`from_files`, `from_stac`, `read_multiple_files`), reductions,
`groupby` via flox, and the two serialisation formats (Zarr +
kerchunk).

Install: `pip install 'pyramids-gis[lazy,stac,netcdf-lazy]'` for the
full surface; the core reductions require only `[lazy]`.

::: pyramids.dataset.DatasetCollection
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
