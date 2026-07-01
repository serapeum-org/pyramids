# Merge & stack

Free functions for combining multiple rasters into one.

```mermaid
flowchart LR
    M["<b>pyramids.dataset.merge</b>"]
    M --> MR["<b>merge_rasters</b><br/>mosaic overlapping rasters<br/>into one (reproject on the fly)"]
    M --> SB["<b>stack_bands</b><br/>stack single-band rasters<br/>into one multi-band Dataset"]
```

- `merge_rasters` — mosaic several (overlapping or adjacent) rasters into a single raster covering
  their union.
- `stack_bands` — stack several single-band rasters into one multi-band raster.

See the [Mosaic & merge notebook](../../examples/operations/mosaic-merge.ipynb) for runnable
examples.

::: pyramids.dataset.merge.merge_rasters
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

::: pyramids.dataset.merge.stack_bands
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
