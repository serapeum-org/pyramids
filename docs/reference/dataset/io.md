# I/O Operations

Array reading/writing, file serialization, tiling, and overview operations.

## Lazy reads — `chunks=…`

`Dataset.read_array(chunks=…)` opts in to a lazy `dask.array.Array`
rather than the default eager `numpy.ndarray`. The same switch powers
every per-pixel op (`focal_*`, `slope`, `aspect`, `hillshade`,
`focal_apply`). `chunks=None` (the default) preserves the legacy
numpy path and does not import dask.

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("big.tif")
lazy = ds.read_array(chunks=(1, 1024, 1024))   # dask.array.Array
lazy.mean(axis=(1, 2)).compute()
```

See [Lazy rasters](../../tutorials/lazy/lazy-raster.md) for chunk-size rules,
locks, `Dataset.to_zarr` / `from_zarr`, and parallel Zarr writes.

Install: `pip install 'pyramids-gis[lazy]'`.

::: pyramids.dataset._collaborators.IO
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
