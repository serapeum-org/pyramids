import matplotlib

matplotlib.use("TkAgg")
from pyramids.dataset import Dataset

path = r"tests/data/geotiff/noah-precipitation-1979.tif"
# %%
dataset = Dataset.read_file(path)
fig, ax = dataset.plot(
    band=0,
    figsize=(10, 5),
    title="NOAA daily Precipitation 1979-01-01",
    cbar_label="Raindall mm/day",
    vmax=30,
    cbar_length=0.85,
)
dataset.lon
# %%
new_dataset = dataset.wrap_longitude()
new_dataset.plot(
    band=0,
    figsize=(10, 5),
    title="NOAA daily Precipitation 1979-01-01",
    cbar_label="Raindall mm/day",
    vmax=30,
    cbar_length=0.85,
)
# %% save the new raster as a geotiff
new_dataset.to_file("examples/data/geotiff/noaa-daily-precipitation-converted.tif")
