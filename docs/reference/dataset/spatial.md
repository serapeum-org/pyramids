# Spatial Operations

Crop, align, reproject, resample, CRS handling, and coordinate conversion.

## Crop with a polygon, raster, or bbox tuple

`Dataset.crop(mask)` accepts a `FeatureCollection` / `GeoDataFrame`
polygon mask or another `Dataset` as a raster mask. For the common
"clip to a geographic bounding box" case, pass the keyword-only
`bbox=(W, S, E, N)` (and `epsg=` if the bbox isn't in the dataset's
own CRS) — pyramids builds the one-row `FeatureCollection` for you and
routes through the same polygon path. The same `bbox=` / `epsg=` pair
is accepted by `DatasetCollection.crop` (built once and reused across
timesteps) and by `Dataset.read_array` (for a windowed read).

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("dem.tif")

# bbox in the dataset's own CRS
ds.crop(bbox=(6.8, 50.3, 7.2, 50.6))

# bbox in WGS84 against a Web-Mercator raster
ds.crop(bbox=(6.8, 50.3, 7.2, 50.6), epsg=4326)
```

`mask=` and `bbox=` are mutually exclusive. If you need the underlying
one-row `FeatureCollection` for other ops, build it with
`FeatureCollection.from_bbox((W, S, E, N), epsg=…)`.

## Reproject — eager `to_crs(...)` vs lazy `warped_view(...)`

`Dataset.to_crs(to_epsg)` **materialises** a reprojected raster: it warps every
pixel into the target CRS and returns a new `Dataset`. Use it when you will
consume the whole reprojected result.

`Dataset.warped_view(crs)` returns a **lazy** reprojected view — an in-memory
warped VRT where nothing is resampled until a window is read, and a windowed
read warps only that window. Prefer it for tile serving, partial reads, and
chained virtual pipelines. The view pins its source alive.

| | `to_crs` | `warped_view` |
|--|----------|---------------|
| When pixels warp | immediately (whole raster) | lazily, per window read |
| Returns | a fully materialised `Dataset` | a VRT-backed view `Dataset` |
| Best for | consuming the whole result | tile serving / partial reads |

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("dem.tif")               # e.g. EPSG:4326
webmerc = ds.to_crs(3857)                       # eager: all pixels warped now
view = ds.warped_view(3857)                     # lazy: warps only what you read
tile = view.read_array(bbox=(...), epsg=3857)   # this window is warped on demand
```

Both accept a `method=` resampling name; `warped_view` also takes `cell_size=`
and `bbox=` to fix the output grid/extent up front.

::: pyramids.dataset.engines.Spatial
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
