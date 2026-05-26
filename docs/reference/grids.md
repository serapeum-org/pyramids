# Exotic Model-Grid Adapters

`pyramids.grids` turns model grids that are **not** row-major rasters — ORCA curvilinear
ocean grids, octahedral reduced-Gaussian grids, and HEALPix sphere pixelizations — into a
regular-grid [`Dataset`][pyramids.dataset.Dataset].

Rather than shipping a new regridder, each adapter reshapes its grid into one of the two
regridding bridges pyramids already has, then reuses it:

- a **mesh** bridge —
  [`mesh_to_grid`][pyramids.netcdf.ugrid.interpolation.mesh_to_grid] (reached through
  [`UgridDataset.to_dataset`][pyramids.netcdf.ugrid.dataset.UgridDataset.to_dataset]); and
- a **scattered-point** bridge —
  [`grid_points`][pyramids.dataset.ops.interpolate.grid_points].

| Adapter | Input grid | Bridge it reuses |
| --- | --- | --- |
| [`from_orca`](#pyramids.grids.from_orca) | curvilinear `(ny, nx)` lon/lat + data | UGRID quad mesh → `mesh_to_grid` |
| [`from_octahedral`](#pyramids.grids.from_octahedral) | ragged per-point lat/lon + values | scattered points → `grid_points` |
| [`from_healpix`](#pyramids.grids.from_healpix) | per-pixel HEALPix values (`nside`) | pixel centres → `grid_points` |

Every adapter returns a single-band `Dataset` (array + geotransform + CRS) that renders with
no further GIS code on the caller side.

!!! note "No `healpy` dependency"
    `from_healpix` needs only the HEALPix pixel→centre mapping (`pix2ang`), which is a
    closed-form from the HEALPix paper. pyramids implements it in plain NumPy for both the
    **RING** and **NESTED** pixel orderings, so no `healpy` (or other HEALPix C library) is
    required.

See the [exotic grids example notebook](../examples/grids/grids.ipynb) for runnable
end-to-end usage of all three adapters.

## Functions

::: pyramids.grids.from_orca
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

::: pyramids.grids.from_octahedral
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

::: pyramids.grids.from_healpix
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
