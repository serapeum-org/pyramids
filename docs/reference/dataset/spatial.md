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

::: pyramids.dataset.engines.Spatial
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
