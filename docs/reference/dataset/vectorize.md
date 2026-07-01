# Vectorization & Clustering

Raster-to-vector conversion, clustering, and translate.

```mermaid
flowchart LR
    VE(("Vectorize<br/>ds.vectorize"))
    VE --> TV["<b>raster → vector</b><br/>to_feature_collection · contour"]
    VE --> CL["<b>cluster</b><br/>cluster · cluster2"]
    VE --> TR["<b>translate</b><br/>translate"]
```

::: pyramids.dataset.engines.Vectorize
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
