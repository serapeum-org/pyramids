import geopandas as gpd

from pyramids.dataset import Dataset

# %%
# dataset = Dataset.read_file("tests/bug/02_RFCF.tif")
dataset = Dataset.read_file("tests/bug/01_TT.tif")
# dataset = Dataset.read_file("tests/bug/smaller_raster.tif")
# dataset.no_data_value
# dataset.plot(ticks_spacing=0.5)
gdf = gpd.read_file("tests/bug/coello-basin-extended.geojson")
cropped = dataset.crop(gdf, touch=False)

# cropped.plot(ticks_spacing=0.5)
arr = cropped.read_array()

# %%
cropped_3 = dataset.crop(gdf, touch=True)
cropped_3.plot(ticks_spacing=0.5)
arr = cropped_3.read_array()
stats = cropped_3.stats()
cropped_3.bbox
cropped_3.bounds.to_file("tests/bug/bounds.geojson")

# cropped.to_file("tests/bug/touch_true.tif")
# print(dataset.count_domain_cells())
# print(cropped.count_domain_cells())
