import geopandas as gpd
from osgeo import gdal

from pyramids.convert import Convert

# %%
"""
RasterToPolygon takes a gdal Datacube object and group neighboring cells with the same value into one
polygon, the resulted vector will be saved to disk as a geojson file
"""
src_path = gdal.Open("examples/data/convert_data/test_image.tif")
# if you give the second parameter a path, the function will save the resulted polygon to disk
polygonized_raster_path = "examples/data/convert_data/polygonized.geojson"
# Convert.rasterToPolygon(src_path, polygonized_raster_path, driver="GeoJSON")

"""
return the result polygon
"""
gdf = Convert.raster_to_polygon(src_path)
# %%
"""
test convert polygon into raster

- The raster cell values will be taken from the column name given in the vector_filed in the vector file.
- all the new raster geotransform data will be copied from the given raster.
- raster and vector should have the same projection

"""
## case 1 first two input parameters are paths for files on disk

input_vector_path = "examples/data/convert_data/mask.geojson"
src_raster_path = "examples/data/convert_data/raster_to_df.tif"
# Path for output raster. if given the resulted raster will be saved to disk.
output_raster = "examples/data/convert_data/rasterized_mask.tif"
Convert.polygonToRaster(input_vector_path, src_raster_path, output_raster)
src = gdal.Open(output_raster)
src.RasterXSize
src.RasterYSize

## case 2 the input vector is a geodataframe object
gdf = gpd.read_file(input_vector_path)
print(gdf)
Convert.polygonToRaster(gdf, src_raster_path, output_raster)

## case 3 there is no given path to save the output raster to disk to it will be returned as an output.
src = Convert.polygonToRaster(gdf, src_raster_path)
print(type(src))
# %%
"""
Dataset To DataFrame

The function do the following
- Flatten the array in each band in the raster then mask the values if a vector
file is given otherwise it will flatten all values.

- Put the values for each band in a column in a dataframe under the name of the raster band, but if no meta
    data in the raster band exists, an index number will be used [1, 2, 3, ...]
- The values in the dataframe will be ordered rows by rows from top to bottom
- The function has a add_geometry parameter with two possible values ["point", "polygon"], which you can
    specify the type of shapely geometry you want to create from each cell,
        - If point is chosen, the created point will be at the center of each cell
        - If a polygon is chosen, a square polygon will be created that covers the entire cell.
-
src : [str/gdal Datacube]
    Path to raster file.
vector : Optional[GeoDataFrame/str]
    GeoDataFrame for the vector file path to vector file. If given, it will be used to clip the raster
add_geometry: [str]
    "Polygon", or "Point" if you want to add a polygon geometry of the cells as  column in dataframe.
    Default is None.
tile: [bool]
    True to use tiles in extracting the values from the raster. Default is False.
tile_size: [int]
    tile size. Default is 1500.
"""
gdf = Convert.rasterToGeoDataFrame(src_raster_path, add_geometry="Point")
"""
the resulted geodataframe will have the band value under the name of the band (if the raster file has a metadata,
if not, the bands will be indexed from 1 to the number of bands)
"""
print(gdf.columns)
# gdf.to_file("examples/data/convert_data/raster_to_polygon.geojson")
"""
examples/data/convert_data/raster_to_polygon.png
"""

## case 2 mask the raster
gdf = gpd.read_file(input_vector_path)
df = Convert.rasterToGeoDataFrame(src_raster_path, gdf)
# print(df)
# Band_1  fid
# 0        1    1
# 1        2    1
# 2        3    1
# 3        4    1
# 4       15    1
# 5       16    1
# 6       17    1
# 7       18    1
# 8       29    1
# 9       30    1
# 10      31    1
# 11      32    1
# 12      43    1
# 13      44    1
# 14      45    1
# 15      46    1

# %%
