"""Vectorize engine.

Owns the Vectorize family of operations on a Dataset. Accessed as
``ds.vectorize``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.vectorize.<method>(...)`` are equivalent.
"""

from __future__ import annotations

import collections
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from hpc.indexing import get_pixels
from osgeo import gdal, ogr
from pandas import DataFrame

from pyramids.base._utils import gdal_to_ogr_dtype
from pyramids.base.crs import sr_from_wkt
from pyramids.feature import FeatureCollection
from pyramids.feature import _ogr as _feature_ogr

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset

from pyramids.dataset.engines._base import _Engine, logger


class Vectorize(_Engine["Dataset"]):
    """Mixin providing vectorization, clustering, and translate methods for Dataset."""

    def _band_to_polygon(self, band: int, col_name: str) -> GeoDataFrame:
        gdal_band = self._ds.raster.GetRasterBand(band + 1)
        srs = sr_from_wkt(self._ds.crs)

        # Build the OGR DataSource directly â€” FeatureCollection.create_ds
        # was deleted because it exposed ogr.DataSource on the public API.
        # Here the DataSource is purely local scratch space for gdal.Polygonize.
        dst_ds = ogr.GetDriverByName("MEM").CreateDataSource("memData")
        if dst_ds is None:
            raise RuntimeError("Failed to create in-memory OGR DataSource")
        dst_layer = dst_ds.CreateLayer(col_name, srs=srs)
        dtype = gdal_to_ogr_dtype(self._ds.raster)
        new_field = ogr.FieldDefn(col_name, dtype)
        dst_layer.CreateField(new_field)
        gdal.Polygonize(gdal_band, gdal_band, dst_layer, 0, [], callback=None)

        return _feature_ogr.datasource_to_gdf(dst_ds)

    def contour(
        self,
        *,
        interval: float | None = None,
        fixed_levels: list[float] | None = None,
        base: float = 0.0,
        band: int = 0,
        attribute: str = "elev",
        polygonize: bool = False,
    ) -> FeatureCollection:
        """Trace iso-value contours from a raster band into a vector layer.

        The GDAL-native equivalent of ``gdal_contour``: it walks one band of the
        raster and emits either contour *lines* at each level (the default) or
        filled contour *polygons* between successive levels (``polygonize=True``).
        Levels are set either by a regular ``interval`` (anchored at ``base``) or
        by an explicit ``fixed_levels`` list — supply exactly one of the two.

        Args:
            interval (float | None):
                Spacing between contour levels, anchored at ``base`` (e.g.
                ``interval=10`` with ``base=0`` yields levels …, -10, 0, 10, …).
                Mutually exclusive with ``fixed_levels``.
            fixed_levels (list[float] | None):
                Explicit list of levels to contour. Mutually exclusive with
                ``interval``.
            base (float):
                The level the regular ``interval`` is anchored to. Ignored when
                ``fixed_levels`` is given. Defaults to ``0.0``.
            band (int):
                Zero-based index of the band to contour. Defaults to ``0``.
            attribute (str):
                Name of the elevation attribute written to each feature. In
                ``polygonize`` mode two columns ``<attribute>_min`` and
                ``<attribute>_max`` are written instead. Defaults to ``"elev"``.
            polygonize (bool):
                When ``True`` emit filled polygons between levels rather than
                contour lines. Defaults to ``False``.

        Returns:
            FeatureCollection:
                A :class:`~pyramids.feature.FeatureCollection` of contour
                ``LineString`` features (or ``Polygon`` features when
                ``polygonize=True``), carrying the source CRS and the elevation
                attribute(s). An empty collection is returned when no contours
                are produced.

        Raises:
            ValueError: Neither or both of ``interval`` / ``fixed_levels`` were
                given, ``interval`` is non-positive, ``fixed_levels`` is empty,
                or ``band`` is out of range.

        Examples:
            - Contour a west-to-east elevation ramp at a regular interval and
              inspect the levels produced:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.tile(np.arange(10, dtype="float32"), (10, 1))
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 10), cell_size=1.0, epsg=4326
                ... )
                >>> contours = ds.contour(interval=2.0)
                >>> sorted(contours["elev"].tolist())
                [2.0, 4.0, 6.0, 8.0]
                >>> contours.geometry.geom_type.unique().tolist()
                ['LineString']

                ```
            - Request explicit levels and read back how many lines were traced:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.tile(np.arange(10, dtype="float32"), (10, 1))
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 10), cell_size=1.0, epsg=4326
                ... )
                >>> contours = ds.contour(fixed_levels=[3.0, 5.0, 7.0])
                >>> len(contours)
                3

                ```
        """
        if (interval is None) == (fixed_levels is None):
            raise ValueError(
                "provide exactly one of interval or fixed_levels (got "
                f"interval={interval!r}, fixed_levels={fixed_levels!r})."
            )
        if interval is not None and interval <= 0:
            raise ValueError(f"interval must be positive, got {interval!r}.")
        if fixed_levels is not None and len(fixed_levels) == 0:
            raise ValueError("fixed_levels must contain at least one level.")
        if band < 0 or band >= self._ds.band_count:
            raise ValueError(
                f"band {band} is out of range for a {self._ds.band_count}-band dataset."
            )

        gdal_band = self._ds.raster.GetRasterBand(band + 1)
        srs = sr_from_wkt(self._ds.crs)
        geom_type = ogr.wkbPolygon if polygonize else ogr.wkbLineString

        dst_ds = ogr.GetDriverByName("MEM").CreateDataSource("contourData")
        if dst_ds is None:
            raise RuntimeError("Failed to create in-memory OGR DataSource")
        dst_layer = dst_ds.CreateLayer("contour", srs=srs, geom_type=geom_type)
        dst_layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

        options = ["ID_FIELD=0"]
        if polygonize:
            dst_layer.CreateField(ogr.FieldDefn(f"{attribute}_min", ogr.OFTReal))
            dst_layer.CreateField(ogr.FieldDefn(f"{attribute}_max", ogr.OFTReal))
            options += ["POLYGONIZE=YES", "ELEV_FIELD_MIN=1", "ELEV_FIELD_MAX=2"]
        else:
            dst_layer.CreateField(ogr.FieldDefn(attribute, ogr.OFTReal))
            options.append("ELEV_FIELD=1")

        if interval is not None:
            options += [f"LEVEL_INTERVAL={interval}", f"LEVEL_BASE={base}"]
        else:
            # The exactly-one-of guard above proves fixed_levels is not None here.
            assert fixed_levels is not None
            options.append("FIXED_LEVELS=" + ",".join(str(lvl) for lvl in fixed_levels))

        no_data_value = gdal_band.GetNoDataValue()
        if no_data_value is not None:
            options.append(f"NODATA={no_data_value}")

        gdal.ContourGenerateEx(gdal_band, dst_layer, options=options)

        return FeatureCollection(_feature_ogr.datasource_to_gdf(dst_ds))

    def to_feature_collection(
        self,
        mask: GeoDataFrame | None = None,
        add_geometry: str | None = None,
        tile: bool = False,
        tile_size: int = 256,
        touch: bool = True,
    ) -> DataFrame | GeoDataFrame:
        """Convert a dataset to a vector.

        The function does the following:
            - Flatten the array in each band in the raster then mask the values if a mask is given
                otherwise it will flatten all values.
            - Put the values for each band in a column in a dataframe under the name of the raster band,
                but if no meta-data in the raster band exists, an index number will be used [1, 2, 3, ...]
            - The function has an add_geometry parameter with two possible values ["point", "polygon"], which you can
                specify the type of shapely geometry you want to create from each cell,

                - If point is chosen, the created point will be at the center of each cell
                - If a polygon is chosen, a square polygon will be created that covers the entire cell.

        Args:
            mask (GeoDataFrame, optional):
                GeoDataFrame to clip the raster. If given, the raster will be cropped to the mask extent.
            add_geometry (str):
                "Polygon" or "Point" if you want to add a polygon geometry of the cells as column in dataframe.
                Default is None.
            tile (bool):
                True to use tiles in extracting the values from the raster. Default is False.
            tile_size (int):
                Tile size. Default is 1500.
            touch (bool):
                Include the cells that touch the polygon not only those that lie entirely inside the polygon mask.
                Default is True.

        Returns:
            DataFrame | GeoDataFrame:
                The resulting frame will have the band value under the name of the band (if the raster file has
                metadata; if not, the bands will be indexed from 1 to the number of bands).

        Examples:
            - Create a dataset from array with 2 bands and 3*3 array each:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.rand(2, 3, 3)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )
              >>> print(dataset.read_array(band=0)) # doctest: +SKIP
              [[0.88625832 0.81804328 0.99372706]
               [0.85333054 0.35448201 0.78079262]
               [0.43887136 0.68166208 0.53170966]]
              >>> print(dataset.read_array(band=1)) # doctest: +SKIP
              [[0.07051872 0.67650833 0.17625027]
               [0.41258071 0.38327938 0.18783139]
               [0.83741314 0.70446373 0.64913575]]

              ```

            - Convert the dataset to dataframe by calling the `to_feature_collection` method:

              ```python
              >>> df = dataset.to_feature_collection()
              >>> list(df.columns)
              ['Band_1', 'Band_2']
              >>> len(df)  # one row per cell of the 3x3 grid
              9
              >>> print(df) # doctest: +SKIP
                   Band_1    Band_2
              0  0.886258  0.070519
              1  0.818043  0.676508
              2  0.993727  0.176250
              3  0.853331  0.412581
              4  0.354482  0.383279
              5  0.780793  0.187831
              6  0.438871  0.837413
              7  0.681662  0.704464
              8  0.531710  0.649136

              ```

            - Convert the dataset into geodataframe with either a polygon or a point geometry that represents each cell.
                To specify the geometry type use the parameter `add_geometry`:

                  ```python
                  >>> gdf = dataset.to_feature_collection(add_geometry="point")
                  >>> print(gdf) # doctest: +SKIP
                       Band_1    Band_2                  geometry
                  0  0.886258  0.070519  POINT (0.02500 -0.02500)
                  1  0.818043  0.676508  POINT (0.07500 -0.02500)
                  2  0.993727  0.176250  POINT (0.12500 -0.02500)
                  3  0.853331  0.412581  POINT (0.02500 -0.07500)
                  4  0.354482  0.383279  POINT (0.07500 -0.07500)
                  5  0.780793  0.187831  POINT (0.12500 -0.07500)
                  6  0.438871  0.837413  POINT (0.02500 -0.12500)
                  7  0.681662  0.704464  POINT (0.07500 -0.12500)
                  8  0.531710  0.649136  POINT (0.12500 -0.12500)
                  >>> gdf = dataset.to_feature_collection(add_geometry="polygon")
                  >>> print(gdf) # doctest: +SKIP
                       Band_1    Band_2                                           geometry
                  0  0.886258  0.070519  POLYGON ((0.00000 0.00000, 0.05000 0.00000, 0....
                  1  0.818043  0.676508  POLYGON ((0.05000 0.00000, 0.10000 0.00000, 0....
                  2  0.993727  0.176250  POLYGON ((0.10000 0.00000, 0.15000 0.00000, 0....
                  3  0.853331  0.412581  POLYGON ((0.00000 -0.05000, 0.05000 -0.05000, ...
                  4  0.354482  0.383279  POLYGON ((0.05000 -0.05000, 0.10000 -0.05000, ...
                  5  0.780793  0.187831  POLYGON ((0.10000 -0.05000, 0.15000 -0.05000, ...
                  6  0.438871  0.837413  POLYGON ((0.00000 -0.10000, 0.05000 -0.10000, ...
                  7  0.681662  0.704464  POLYGON ((0.05000 -0.10000, 0.10000 -0.10000, ...
                  8  0.531710  0.649136  POLYGON ((0.10000 -0.10000, 0.15000 -0.10000, ...

                  ```

            - Use a mask to crop part of the dataset, and then convert the cropped part to a dataframe/geodataframe:

              - Create a mask that covers only the cell in the middle of the dataset.

                  ```python
                  >>> import geopandas as gpd
                  >>> from shapely.geometry import Polygon
                  >>> poly = gpd.GeoDataFrame(
                  ...             geometry=[Polygon([(0.05, -0.05), (0.05, -0.1), (0.1, -0.1), (0.1, -0.05)])], crs=4326
                  ... )
                  >>> df = dataset.to_feature_collection(mask=poly)
                  >>> print(df) # doctest: +SKIP
                       Band_1    Band_2
                  0  0.354482  0.383279

                  ```

            - If you have a big dataset, and you want to convert it to dataframe in tiles (do not read the whole dataset
                at once but in tiles), you can use the `tile` and the `tile_size` parameters. The values will be the
                same as above; the difference is reading in chunks:

                  ```python
                  >>> gdf = dataset.to_feature_collection(tile=True, tile_size=1)
                  >>> print(gdf) # doctest: +SKIP
                       Band_1    Band_2
                  0  0.886258  0.070519
                  1  0.818043  0.676508
                  2  0.993727  0.176250
                  3  0.853331  0.412581
                  4  0.354482  0.383279
                  5  0.780793  0.187831
                  6  0.438871  0.837413
                  7  0.681662  0.704464
                  8  0.531710  0.649136

                  ```

        """
        band_names = self._ds.band_names

        if mask is not None:
            src_ds = self._ds.crop(mask=mask, touch=touch)
        else:
            src_ds = self._ds

        # Both branches must read `src_ds` -- the cropped dataset when a mask was
        # given. Reading `self` here silently discarded the mask, so a tiled call
        # returned values for the whole raster while the geometry attached below
        # came from the cropped extent.
        if tile:
            df = src_ds.vectorize._extract_values_tiled(band_names, tile_size)
        else:
            df = src_ds.vectorize._extract_values_full(band_names)

        df.drop(columns=["burn_value", "geometry"], errors="ignore", inplace=True)

        if add_geometry:
            df = self._attach_geometry(src_ds, df, add_geometry)

        return df

    def _extract_values_tiled(self, band_names: list, tile_size: int) -> pd.DataFrame:
        """Extract raster band values into a DataFrame using tiles.

        Args:
            band_names (list): Band names for the DataFrame columns.
            tile_size (int): Tile size in pixels.

        Returns:
            pd.DataFrame: Concatenated DataFrame from all tiles.
        """
        no_data_value = self._ds.no_data_value[0]
        df_list = []
        for arr in self._ds.get_tile(tile_size):
            idx = (1, 2) if arr.ndim > 2 else (0, 1)
            mask_arr = np.ones((arr.shape[idx[0]], arr.shape[idx[1]]))
            pixels = get_pixels(arr, mask_arr).transpose()
            df = pd.DataFrame(pixels, columns=band_names)
            if no_data_value is not None:
                df.replace(no_data_value, np.nan, inplace=True)
            df.dropna(axis=0, inplace=True, ignore_index=True)
            if not df.empty:
                df_list.append(df)

        if not df_list:
            return pd.DataFrame(columns=band_names)

        return pd.concat(df_list, ignore_index=True)

    def _extract_values_full(self, band_names: list) -> pd.DataFrame:
        """Extract all raster band values into a DataFrame (no tiling).

        Args:
            band_names (list): Band names for the DataFrame columns.

        Returns:
            pd.DataFrame: DataFrame with one column per band, no-data rows removed.
        """
        arr = self._ds.read_array()

        if self._ds.band_count == 1:
            pixels = arr.flatten()
        else:
            pixels = (
                arr.flatten()
                .reshape(self._ds.band_count, self._ds.columns * self._ds.rows)
                .transpose()
            )
        df = pd.DataFrame(pixels, columns=band_names)
        if self._ds.no_data_value[0] is not None:
            df.replace(self._ds.no_data_value[0], np.nan, inplace=True)
        df.dropna(axis=0, inplace=True, ignore_index=True)
        return df

    @staticmethod
    def _attach_geometry(src, df: pd.DataFrame, geometry_type: str) -> gpd.GeoDataFrame:
        """Attach point or polygon geometry to a DataFrame.

        Args:
            src: The dataset to derive cell geometries from.
            df (pd.DataFrame): DataFrame with band values.
            geometry_type (str): "point" or "polygon".

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with geometry column.
        """
        if geometry_type.lower() == "point":
            coords = src.get_cell_points(domain_only=True)
        else:
            coords = src.get_cell_polygons(domain_only=True)

        gdf = gpd.GeoDataFrame(df.loc[:], geometry=coords["geometry"].to_list())
        gdf = gdf.set_crs(coords.crs.to_epsg())
        return gdf

    def translate(self, path: str | Path | None = None, **kwargs) -> Dataset:
        """Convert, subset, resample, or rescale the raster via ``gdal.Translate``.

        The translate function can be used to
        - Convert Between Formats: Convert a raster from one format to another (e.g., from GeoTIFF to JPEG).
        - Subset: Extract a subregion of a raster.
        - Resample: Change the resolution of a raster.
        - Reproject: Change the coordinate reference system of a raster.
        - Scale Values: Scale pixel values to a new range.
        - Change Data Type: Convert the data type of the raster.
        - Apply Compression: Apply compression to the output raster.
        - Apply No-Data Values: Define no-data values for the output raster.

        Args:
            path (str | Path | None):
                Path to save the output. If None (default), the output is kept in memory.
            **kwargs:
                Options forwarded to :func:`gdal.TranslateOptions`:
                unscale:
                    unscale values with scale and offset metadata.
                scaleParams:
                    list of scale parameters, each of the form [src_min,src_max] or [src_min,src_max,dst_min,dst_max]
                outputType:
                    output type (gdalconst.GDT_Byte, etc...)
                exponents:
                    list of exponentiation parameters
                bandList:
                    array of band numbers (index start at 1)
                maskBand:
                    mask band to generate or not ("none", "auto", "mask", 1, ...)
                creationOptions:
                    list or dict of creation options
                srcWin:
                    subwindow in pixels to extract: [left_x, top_y, width, height]
                projWin:
                    subwindow in projected coordinates to extract: [ulx, uly, lrx, lry]
                projWinSRS:
                    SRS in which projWin is expressed
                outputBounds:
                    assigned output bounds: [ulx, uly, lrx, lry]
                outputGeotransform:
                    assigned geotransform matrix (array of 6 values) (mutually exclusive with outputBounds)
                metadataOptions:
                    list or dict of metadata options
                outputSRS:
                    assigned output SRS
                noData:
                    nodata value (or "none" to unset it)
                rgbExpand:
                    Color palette expansion mode: "gray", "rgb", "rgba"
                xmp:
                    whether to copy XMP metadata
                resampleAlg:
                    resampling mode
                overviewLevel:
                    To specify which overview level of source files must be used
                domainMetadataOptions:
                    list or dict of domain-specific metadata options

        Returns:
            Dataset:
                The translated dataset (in memory when ``path`` is None).

        Examples:
        Scale & offset:
            - the translate function can be used to get rid of the scale and offset that are used to manipulate the
            dataset, to get the real values of the dataset.

            Scale:
                - First we will create a dataset from a float32 array with values between 1 and 10, and then we will
                    assign a scale of 0.1 to the dataset.

                    >>> import numpy as np
                    >>> from pyramids.dataset import Dataset
                    >>> arr = np.random.randint(1, 10, size=(5, 5)).astype(np.float32)
                    >>> print(arr) # doctest: +SKIP
                    [[5. 5. 3. 4. 2.]
                     [2. 5. 5. 8. 5.]
                     [7. 5. 6. 1. 2.]
                     [6. 8. 1. 5. 8.]
                     [2. 5. 2. 2. 9.]]
                    >>> top_left_corner = (0, 0)
                    >>> cell_size = 0.05
                    >>> dataset = Dataset.create_from_array(
                    ...     arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326
                    ... )
                    >>> print(dataset)  # doctest: +SKIP
                    <BLANKLINE>
                                Top Left Corner: (0.0, 0.0)
                                Cell size: 0.05
                                Dimension: 5 * 5
                                EPSG: 4326
                                Number of Bands: 1
                                Band names: ['Band_1']
                                Band colors: {0: 'undefined'}
                                Band units: ['']
                                Scale: [1.0]
                                Offset: [0]
                                Mask: -9999.0
                                Data type: float32
                                File: ...
                    <BLANKLINE>
                    >>> dataset.scale = [0.1]

                - now lets unscale the dataset values.

                    >>> unscaled_dataset = dataset.translate(unscale=True)
                    >>> print(unscaled_dataset) # doctest: +SKIP
                    <BLANKLINE>
                                Top Left Corner: (0.0, 0.0)
                                Cell size: 0.05
                                Dimension: 5 * 5
                                EPSG: 4326
                                Number of Bands: 1
                                Band names: ['Band_1']
                                Band colors: {0: 'undefined'}
                                Band units: ['']
                                Scale: [1.0]
                                Offset: [0]
                                Mask: -9999.0
                                Data type: float32
                                File:
                    <BLANKLINE>
                    >>> print(unscaled_dataset.read_array()) # doctest: +SKIP
                    [[0.5 0.5 0.3 0.4 0.2]
                     [0.2 0.5 0.5 0.8 0.5]
                     [0.7 0.5 0.6 0.1 0.2]
                     [0.6 0.8 0.1 0.5 0.8]
                     [0.2 0.5 0.2 0.2 0.9]]

            offset:
                - You can also unshift the values of the dataset if the dataset has an offset. To remove the offset
                    from all values in the dataset, you can read the values using the `read_array` and then add the
                    offset value to the array. we will create a dataset from the same array we created above (values
                    are between 1, and 10) with an offset of 100.

                    >>> dataset = Dataset.create_from_array(
                    ...     arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326
                    ... )
                    >>> print(dataset)  # doctest: +SKIP
                    <BLANKLINE>
                                Top Left Corner: (0.0, 0.0)
                                Cell size: 0.05
                                Dimension: 5 * 5
                                EPSG: 4326
                                Number of Bands: 1
                                Band names: ['Band_1']
                                Band colors: {0: 'undefined'}
                                Band units: ['']
                                Scale: [1.0]
                                Offset: [0]
                                Mask: -9999.0
                                Data type: float32
                                File: ...
                    <BLANKLINE>

                - set the offset to 100.

                    >>> dataset.offset = [100]

                - check if the offset has been set.

                    >>> print(dataset.offset)
                    [100.0]

                - now lets unscale the dataset values.

                    >>> unscaled_dataset = dataset.translate(unscale=True)
                    >>> print(unscaled_dataset.read_array()) # doctest: +SKIP
                    [[105. 105. 103. 104. 102.]
                     [102. 105. 105. 108. 105.]
                     [107. 105. 106. 101. 102.]
                     [106. 108. 101. 105. 108.]
                     [102. 105. 102. 102. 109.]]

                - as you see, all the values have been shifted by 100. now if you check the offset of the dataset

                    >>> print(unscaled_dataset.offset)
                    [0]

            Offset and Scale together:
                - we can unscale and get rid of the offset at the same time.

                    >>> dataset = Dataset.create_from_array(
                    ...     arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326
                    ... )

                - set the offset to 100, and a scale of 0.1.

                    >>> dataset.offset = [100]
                    >>> dataset.scale = [0.1]

                - check if the offset has been set.

                    >>> print(dataset.offset)
                    [100.0]
                    >>> print(dataset.scale)
                    [0.1]

                - now lets unscale the dataset values.

                    >>> unscaled_dataset = dataset.translate(unscale=True)
                    >>> print(unscaled_dataset.read_array()) # doctest: +SKIP
                    [[100.5 100.5 100.3 100.4 100.2]
                     [100.2 100.5 100.5 100.8 100.5]
                     [100.7 100.5 100.6 100.1 100.2]
                     [100.6 100.8 100.1 100.5 100.8]
                     [100.2 100.5 100.2 100.2 100.9]]

                - Now you can see that the values were multiplied first by the scale; then the offset value was added.
                    `value * scale + offset`

                    >>> print(unscaled_dataset.offset)
                    [0]
                    >>> print(unscaled_dataset.scale)
                    [1.0]

        Scale between two values:
            - you can scale the values of the dataset between two values, for example, you can scale the values
                between two values 0 and 1.

                >>> dataset = Dataset.create_from_array(
                ...     arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326
                ... )
                >>> print(dataset.stats()) # doctest: +SKIP
                        min  max  mean      std
                Band_1  1.0  9.0   4.0  2.19089
                >>> scaled_dataset = dataset.translate(scaleParams=[[1, 9, 0, 255]], outputType=gdal.GDT_Byte)
                >>> print(scaled_dataset.read_array()) # doctest: +SKIP
                [[128 128  64  96  32]
                 [ 32 128 128 223 128]
                 [191 128 159   0  32]
                 [159 223   0 128 223]
                 [ 32 128  32  32 255]]


        """
        if path is None:
            driver = "MEM"
            path = ""
        else:
            driver = "GTiff"

        options = gdal.TranslateOptions(format=driver, **kwargs)
        dst = gdal.Translate(str(path), self._ds.raster, options=options)
        result = self._ds.__class__(dst, access="write")
        return result

    @staticmethod
    def _nearest_neighbour(
        array: np.ndarray, no_data_value: float | int, rows: list, cols: list
    ) -> np.typing.NDArray:
        """Fill specified cells with the value of the nearest neighbor.

            - The _nearest_neighbour method fills the cells with the given indices in rows and cols with the value
                of the nearest neighbor.
            - The raster grid is square, so the 4 perpendicular directions are of the same proximity; the function
                gives priority to the right, left, bottom, and then top, and similarly for 45-degree directions:
                right-bottom, left-bottom, left-top, right-top.

        Args:
            array (np.ndarray):
                Array to fill some of its cells with the nearest value.
            no_data_value (float | int):
                Value stored in cells that are out of the domain.
            rows (list[int]):
                Row indices of the cells you want to fill with the nearest neighbor.
            cols (list[int]):
                Column indices of the cells you want to fill with the nearest neighbor.

        Returns:
            np.ndarray:
                Cells of given indices filled with the value of the nearest neighbor.

        Examples:
            - Fill a single no-data hole from its right-hand neighbour (the first
              direction the algorithm tries):

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset.engines.vectorize import Vectorize
              >>> arr = np.array([[1.0, 2.0, 3.0], [4.0, -9999.0, 6.0], [7.0, 8.0, 9.0]])
              >>> filled = Vectorize._nearest_neighbour(arr, -9999.0, [1], [1])
              >>> float(filled[1, 1])
              6.0

              ```
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(
                "src should be read using gdal (gdal dataset please read it using gdal library) "
            )
        if not isinstance(rows, list):
            raise TypeError("The `rows` input has to be of type list")
        if not isinstance(cols, list):
            raise TypeError("The `cols` input has to be of type list")

        no_rows = np.shape(array)[0]
        no_cols = np.shape(array)[1]

        for i in range(len(rows)):
            # give the cell the value of the cell that is at the right
            if cols[i] + 1 < no_cols:
                if array[rows[i], cols[i] + 1] != no_data_value:
                    array[rows[i], cols[i]] = array[rows[i], cols[i] + 1]

            elif array[rows[i], cols[i] - 1] != no_data_value and cols[i] - 1 > 0:
                # give the cell the value of the cell that is at the left
                array[rows[i], cols[i]] = array[rows[i], cols[i] - 1]

            elif array[rows[i] - 1, cols[i]] != no_data_value and rows[i] - 1 > 0:
                # give the cell the value of the cell that is at the bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i]]

            elif array[rows[i] + 1, cols[i]] != no_data_value and rows[i] + 1 < no_rows:
                # give the cell the value of the cell that is at the Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i]]

            elif (
                array[rows[i] - 1, cols[i] + 1] != no_data_value
                and rows[i] - 1 > 0
                and cols[i] + 1 <= no_cols
            ):
                # give the cell the value of the cell that is at the right bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i] + 1]

            elif (
                array[rows[i] - 1, cols[i] - 1] != no_data_value
                and rows[i] - 1 > 0
                and cols[i] - 1 > 0
            ):
                # give the cell the value of the cell that is at the left bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i] - 1]

            elif (
                array[rows[i] + 1, cols[i] - 1] != no_data_value
                and rows[i] + 1 <= no_rows
                and cols[i] - 1 > 0
            ):
                # give the cell the value of the cell that is at the left Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i] - 1]

            elif (
                array[rows[i] + 1, cols[i] + 1] != no_data_value
                and rows[i] + 1 <= no_rows
                and cols[i] + 1 <= no_cols
            ):
                # give the cell the value of the cell that is at the right Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i] + 1]
            else:
                logger.warning("the cell is isolated (No surrounding cells exist)")
        return array

    @staticmethod
    def _group_neighbours(
        array, i, j, lower_bound, upper_bound, position, values, count, cluster
    ) -> None:
        """Group neighboring cells with the same values using iterative BFS.

        Uses a queue-based breadth-first search instead of recursion to avoid
        hitting Python's recursion limit on large connected regions.

        Note: The starting cell (i, j) is enqueued but not marked. When a
        discovered neighbor later checks its own neighbors, it will find (i, j)
        still unmarked and add it to position/values. Therefore the starting
        cell appears in the output whenever it has at least one in-bound
        neighbor. The caller (cluster) handles truly isolated cells separately.
        """
        rows, cols = array.shape
        # 8-connected neighbour offsets, in the row-major order of the original
        # (-1, 0, 1) x (-1, 0, 1) double loop (minus the (0, 0) self-cell).
        offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        queue: collections.deque[tuple[int, int]] = collections.deque()
        queue.append((i, j))

        while queue:
            ci, cj = queue.popleft()
            for di, dj in offsets:
                ni, nj = ci + di, cj + dj
                if (
                    0 <= ni < rows
                    and 0 <= nj < cols
                    and cluster[ni, nj] == 0
                    and lower_bound <= array[ni, nj] <= upper_bound
                ):
                    cluster[ni, nj] = count
                    position.append([ni, nj])
                    values.append(array[ni, nj])
                    queue.append((ni, nj))

    def cluster(
        self, lower_bound: Any, upper_bound: Any
    ) -> tuple[np.typing.NDArray, int, list, list]:
        """Group all the connected values between two bounds.

        Args:
            lower_bound (Number):
                Lower bound of the cluster.
            upper_bound (Number):
                Upper bound of the cluster.

        Returns:
            tuple[np.ndarray, int, list, list]:
                - cluster (np.ndarray):
                    Array with integers representing the cluster number per cell.
                - count (int):
                    Number of clusters in the array.
                - position (list[list[int, int]]):
                    List of [row, col] indices for the position of each value.
                - values (list[Number]):
                    Values stored in each cell in the cluster.

        Examples:
            - First, we will create a dataset with 10 rows and 10 columns.

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> np.random.seed(10)
              >>> arr = np.random.randint(1, 5, size=(5, 5))
              >>> print(arr) # doctest: +SKIP
              [[2 3 3 2 3]
               [3 4 1 1 1]
               [1 3 3 2 2]
               [4 1 1 3 2]
               [2 4 2 3 2]]
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )
              >>> dataset.plot(
              ...     color_scale="boundary-norm", bounds=[1, 1.9, 4.1, 5], display_cell_value=True, num_size=12,
              ...     background_color_threshold=5
              ... )  # doctest: +SKIP

              ```
              ![cluster](./../../_images/dataset/cluster.png)

            - Now let's cluster the values in the dataset that are between 2 and 4.

              ```python
              >>> lower_value = 2
              >>> upper_value = 4
              >>> cluster_array, count, position, values = dataset.cluster(lower_value, upper_value)

              ```
            - The first returned output is a binary array with 1 indicating that the cell value is inside the
                cluster, and 0 is outside.

              ```python
              >>> print(cluster_array)  # doctest: +SKIP
              [[1. 1. 1. 1. 1.]
               [1. 1. 0. 0. 0.]
               [0. 1. 1. 1. 1.]
               [1. 0. 0. 1. 1.]
               [1. 1. 1. 1. 1.]]

              ```
            - The second returned value is the number of connected clusters.

              ```python
              >>> print(count) # doctest: +SKIP
              2

              ```
            - The third returned value is the indices of the cells that belong to the cluster.

              ```python
              >>> print(position) # doctest: +SKIP
              [[1, 0], [2, 1], [2, 2], [3, 3], [4, 3], [4, 4], [3, 4], [2, 4], [2, 3], [4, 2], [4, 1], [3, 0], [4, 0], [1, 1], [0, 2], [0, 3], [0, 4], [0, 1], [0, 0]]

              ```
            - The fourth returned value is a list of the values that are in the cluster (extracted from these cells).

              ```python
              >>> print(values) # doctest: +SKIP
              [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 4, 4, 2, 4, 3, 2, 3, 3, 2]

              ```

        """
        data = self._ds.read_array()
        position: list[list[int]] = []
        values: list[Any] = []
        count = 1
        cluster = np.zeros(shape=(data.shape[0], data.shape[1]))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if lower_bound <= data[i, j] <= upper_bound and cluster[i, j] == 0:
                    self._group_neighbours(
                        data,
                        i,
                        j,
                        lower_bound,
                        upper_bound,
                        position,
                        values,
                        count,
                        cluster,
                    )
                    if cluster[i, j] == 0:
                        position.append([i, j])
                        values.append(data[i, j])
                        cluster[i, j] = count
                    count += 1

        return cluster, count, position, values

    def to_polygons(
        self,
        band: int | list[int] | None = None,
    ) -> GeoDataFrame:
        """Polygonize a raster band — merge connected equal-valued cells into polygons.

        - Creates vector polygons for all connected regions of pixels in the raster sharing a common
            pixel value (group neighboring cells with the same value into one polygon).

        Args:
            band (int | List[int] | None):
                Band index 0, 1, 2, 3, ...

        Returns:
            GeoDataFrame:
                GeodataFrame containing polygon geomtries for all connected regions.

        Examples:
            - First, we will create a 10*10 dataset full of random integer between 1, and 5.

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> np.random.seed(200)
              >>> arr = np.random.randint(1, 5, size=(10, 10))
              >>> print(arr)  # doctest: +SKIP
              [[3 2 1 1 3 4 1 4 2 3]
               [4 2 2 4 3 3 1 2 4 4]
               [4 2 4 2 3 4 2 1 4 3]
               [3 2 1 4 3 3 4 1 1 4]
               [1 2 4 2 2 1 3 2 3 1]
               [1 4 4 4 1 1 4 2 1 1]
               [1 3 2 3 3 4 1 3 1 3]
               [4 1 3 3 3 4 1 4 1 1]
               [2 1 3 3 4 2 2 1 3 4]
               [2 3 2 2 4 2 1 3 2 2]]
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )

              ```

            - Now, let's cluster the connected equal cells into polygons.
              ```python
              >>> gdf = dataset.to_polygons()
              >>> print(gdf)  # doctest: +SKIP
                  Band_1                                           geometry
              0        3  POLYGON ((0 0, 0 -0.05, 0.05 -0.05, 0.05 0, 0 0))
              1        1  POLYGON ((0.1 0, 0.1 -0.05, 0.2 -0.05, 0.2 0, ...
              2        4  POLYGON ((0.25 0, 0.25 -0.05, 0.3 -0.05, 0.3 0...
              3        4  POLYGON ((0.35 0, 0.35 -0.05, 0.4 -0.05, 0.4 0...
              4        2  POLYGON ((0.4 0, 0.4 -0.05, 0.45 -0.05, 0.45 0...
              5        3  POLYGON ((0.45 0, 0.45 -0.05, 0.5 -0.05, 0.5 0...
              6        1  POLYGON ((0.3 0, 0.3 -0.1, 0.35 -0.1, 0.35 0, ...
              7        4  POLYGON ((0.15 -0.05, 0.15 -0.1, 0.2 -0.1, 0.2...
              8        2  POLYGON ((0.35 -0.05, 0.35 -0.1, 0.4 -0.1, 0.4...
              9        4  POLYGON ((0 -0.05, 0 -0.15, 0.05 -0.15, 0.05 -...
              10       4  POLYGON ((0.4 -0.05, 0.4 -0.15, 0.45 -0.15, 0....
              11       4  POLYGON ((0.1 -0.1, 0.1 -0.15, 0.15 -0.15, 0.1...

              ```

        """
        if band is None:
            band = 0

        names = self._ds.band_names

        def _band_name(index: int) -> str:
            # band_names can be empty (e.g. a mask dataset built internally by footprint); fall back.
            return names[index] if index < len(names) else f"Band_{index + 1}"

        if isinstance(band, int):
            gdf = self._band_to_polygon(band, _band_name(band))
        else:
            gdfs = []
            for b in band:
                gdfs.append(self._band_to_polygon(b, _band_name(b)))
            gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

        return gdf

    def cluster2(self, band: int | list[int] | None = None) -> GeoDataFrame:
        """Deprecated alias for :meth:`to_polygons`.

        .. deprecated::
            ``cluster2`` was renamed to :meth:`to_polygons` for clarity. This alias forwards to
            :meth:`to_polygons` and will be removed in a future release.
        """
        warnings.warn(
            "cluster2 is deprecated and will be removed in a future release; use to_polygons instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_polygons(band=band)
