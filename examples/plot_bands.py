import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from pyramids.dataset import Dataset, DatasetCollection

path = "tests/data/raster-folder/1_MSWEP_1979.01.02.tif"
# %%
dataset = Dataset.read_file(path)
fig, ax = dataset.plot(band=0)

# %%
rasters_folder_path = "tests/data/raster-folder"
cube = DatasetCollection.read_multiple_files(rasters_folder_path, with_order=False)
cube.open_multi_dataset()
# dataset = cube.iloc(0)
# fig, ax= dataset.plot()
cube.plot()
# %% plot rgb
dataset = Dataset.read_file(
    "tests/data/geotiff/S2A_MSIL2A_20200215T082021_N0214_R121_T36SXA_20200215T110825_image_0_0.tif"
)
fig, ax = dataset.plot()
