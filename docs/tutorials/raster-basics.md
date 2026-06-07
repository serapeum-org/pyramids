# Tutorial: Raster basics

This tutorial demonstrates the core single-raster workflow: opening a raster, inspecting its
georeferencing, reading values into a NumPy array, and saving a copy.

Prerequisites: install pyramids (see [Installation](../installation.md)) and have a sample raster.
The snippets below use a GeoTIFF shipped with the repository's test data.

## Open a raster and inspect it

```python
from pyramids.dataset import Dataset

src = "tests/data/geotiff/dem.tif"  # adjust as needed

ds = Dataset.read_file(src)
print("Size (cols x rows):", ds.columns, ds.rows)
print("Bands:", ds.band_count)
print("EPSG:", ds.epsg)
print("Cell size:", ds.cell_size)
print("Bounding box [xmin, ymin, xmax, ymax]:", ds.bbox)
print("No-data value:", ds.no_data_value)
```

`Dataset.read_file` opens the file through GDAL without reading the pixels; the metadata above is
available immediately.

## Read pixel values

```python
# Read all bands into a NumPy array of shape (bands, rows, cols)
arr = ds.read_array()
print(arr.shape, arr.dtype)

# Read a single band (zero-based) into a 2-D array
band0 = ds.read_array(band=0)
print(band0.shape)
```

## Save a copy

```python
out = "tutorial_dem_copy.tif"
ds.to_file(out)
print("Saved:", out)
```

Expected outcome: an output GeoTIFF is written that opens in any GIS tool.

## Next steps

- [Dataset basics notebook](../examples/dataset/dataset.ipynb) — end-to-end, runnable.
- [Spatial operations](../examples/dataset/spatial-operation-methods.ipynb) — crop, reproject, align.
- [Cloud Optimized GeoTIFFs](cog.md) — write and validate COGs.
- [Dataset API reference](../reference/dataset/index.md) — every method, with signatures.
