# Georeferencing (GCPs & RPCs)

```mermaid
flowchart LR
    GE(("Georef<br/>ds.georef"))
    GE --> GC["<b>ground-control points</b><br/>gcps · gcp_count · gcp_projection<br/>has_gcps · set_gcps"]
    GE --> RP["<b>rational-polynomial coeffs</b><br/>rpcs · has_rpcs · set_rpcs"]
    GE --> WA["<b>warp to a map grid</b><br/>orthorectify · georeference"]
```

Read, attach, and warp-from **ground-control points** and **rational-polynomial coefficients** to
georeference raw / un-orthorectified imagery — the one case where a raster is not described by a simple
affine geotransform. Accessed as `ds.georef`, with same-named facades on `Dataset` (`ds.gcps`,
`ds.set_gcps`, `ds.georeference`, `ds.rpcs`, `ds.set_rpcs`, `ds.orthorectify`).

Everything routes through GDAL — pyramids does not implement any sensor model itself.

::: pyramids.dataset.engines.Georef
    options:
        show_root_heading: true
        show_source: true

::: pyramids.dataset._gcp.GroundControlPoint
    options:
        show_root_heading: true
        show_source: true
