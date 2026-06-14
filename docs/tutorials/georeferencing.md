# Georeferencing raw imagery (GCPs & RPCs)

Most rasters pyramids handles are already georeferenced by an **affine geotransform** — a clean grid
aligned to a CRS. Raw imagery is different: a scanned map, a drone frame, or an un-orthorectified
satellite scene instead carries either **ground-control points** (pixel↔map tie points) or **rational
polynomial coefficients** (a vendor sensor model). This page shows how to turn both into a normal
georeferenced raster.

All of it lives on `ds.georef` (with flat facades on `Dataset`) and routes through GDAL.

## Ground-control points (rubber-sheeting)

A `GroundControlPoint` says *"pixel `(col, row)` is at map coordinate `(x, y)`"*. Attach a handful with
`set_gcps`, then `georeference` fits a polynomial (or thin-plate spline) through them and warps the
pixels onto a regular grid.

```python
import numpy as np
from pyramids.dataset import Dataset
from pyramids.dataset import GroundControlPoint

# a raw 8x8 image with no useful geotransform
raw = Dataset.create_from_array(
    np.arange(64, dtype="float32").reshape(8, 8),
    top_left_corner=(0.0, 8.0),
    cell_size=1.0,
)

# four corner tie points in EPSG:4326
raw.set_gcps(
    [
        GroundControlPoint(row=0, col=0, x=10.0, y=50.0),
        GroundControlPoint(row=0, col=8, x=11.0, y=50.0),
        GroundControlPoint(row=8, col=0, x=10.0, y=49.0),
        GroundControlPoint(row=8, col=8, x=11.0, y=49.0),
    ],
    projection=4326,
)

print(raw.gcp_count, raw.has_gcps)          # 4 True

georeferenced = raw.georeference()          # warp from the GCPs (polynomial order 1)
print(georeferenced.epsg)                   # 4326
print(georeferenced.bbox)                   # ~ [10, 49, 11, 50]
```

- Use `transform="tps"` for a thin-plate spline (good for many, irregularly-spaced points).
- Use `order=2` / `order=3` for higher-degree polynomials (needs ≥6 / ≥10 points).
- Pass `to_epsg=` to reproject into another CRS in the same warp, or `lazy=True` for a VRT-backed view
  that warps only the window you read.

From the command line:

```bash
pyramids georeference raw.tif out.tif \
    --gcp 0 0 10 50 --gcp 8 0 11 50 --gcp 0 8 10 49 --gcp 8 8 11 49 \
    --gcp-crs 4326
```

## Rational polynomial coefficients (orthorectification)

High-resolution satellite scenes ship ~90 RPC coefficients describing the sensor geometry. Read them
with `ds.rpcs`, attach them with `set_rpcs`, and remove terrain distortion with `orthorectify` —
ideally against a DEM:

```python
ds = Dataset.read_file("scene.tif")     # a scene whose RPCs GDAL read on open
if ds.has_rpcs:
    ortho = ds.orthorectify(dem="dem.tif")          # DEM-based orthorectification
    # or, with no DEM, a constant elevation:
    ortho = ds.orthorectify(rpc_height=120.0)
```

From the command line:

```bash
pyramids orthorectify scene.tif ortho.tif --dem dem.tif
pyramids orthorectify scene.tif ortho.tif --rpc-height 120
```

## Scope

GCPs and RPCs are *generic* GDAL georeferencing primitives, so they fit pyramids' remit. pyramids
exposes GDAL's GCP/RPC transformers — it does not implement sensor-specific photogrammetry. See the
[Georeferencing reference](../reference/dataset/georef.md) for the full API.
