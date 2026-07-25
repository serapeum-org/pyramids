"""Bands engine.

Owns the Bands family of operations on a Dataset. Accessed as
``ds.bands``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.bands.<method>(...)`` are equivalent.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal
from pandas import DataFrame

from pyramids.base._domain import is_no_data
from pyramids.base._errors import NoDataValueError, ReadOnlyError
from pyramids.base._utils import (
    color_name_to_gdal_constant,
    gdal_constant_to_color_name,
    gdal_to_numpy_dtype,
    numpy_to_gdal_dtype,
    require_cleopatra,
)
from pyramids.dataset.abstract_dataset import DEFAULT_NO_DATA_VALUE
from pyramids.feature import create_polygon

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset

from pyramids.dataset.engines._base import _Engine

# Substring GDAL raises when a write is attempted on a read-only band; matched in
# several no-data setters below to re-raise a friendly ReadOnlyError. Named once
# to avoid duplicating the literal (S1192).
_GDAL_READ_ONLY_FILL_ERROR = (
    "Attempt to write to read only dataset in GDALRasterBand::Fill()."
)


class Bands(_Engine["Dataset"]):
    """Mixin providing band metadata, attribute table, and color table operations."""

    def _iloc(self, i: int) -> gdal.Band:
        """Access a GDAL Band by 0-based index.

        The returned band object is only valid while the parent dataset
        is open. Do not store the band reference — use it immediately
        and discard it.

        Args:
            i (int):
                Band index (0-based).

        Returns:
            gdal.Band:
                Gdal Band.

        Raises:
            IndexError: If the index is negative or out of bounds.
            RuntimeError: If the dataset has been closed.
        """
        # RuntimeError (via _require_open) is intentional here: the dataset is
        # *closed*, not read-only, so ReadOnlyError would be misleading.
        self._ds._require_open()
        if i < 0:
            raise IndexError("negative index not supported")

        if i > self._ds.band_count - 1:
            raise IndexError(
                f"index {i} is out of bounds for axis 0 with size {self._ds.band_count}"
            )
        band = self._ds.raster.GetRasterBand(i + 1)
        return band

    def get_attribute_table(self, band: int = 0) -> DataFrame:
        """Get the attribute table for a given band.

            - Get the attribute table of a band.

        Args:
            band (int):
                Band index, the index starts from 1.

        Returns:
            DataFrame:
                DataFrame with the attribute table.

        Examples:
            - Read a dataset and fetch its attribute table:

              ```python
              >>> from pyramids.dataset import Dataset
              >>> import pandas as pd
              >>> dataset = Dataset.create(
              ...     cell_size=0.05, rows=5, columns=5, dtype="float32", bands=1,
              ...     top_left_corner=(0, 0), epsg=4326, no_data_value=-9999,
              ... )
              >>> dataset.set_attribute_table(
              ...     pd.DataFrame({"Category": ["Low", "High"], "Description": ["dry", "wet"]})
              ... )
              >>> df = dataset.get_attribute_table()
              >>> df["Category"].tolist()
              ['Low', 'High']

              ```
        """
        band_obj = self._iloc(band)
        rat = band_obj.GetDefaultRAT()
        if rat is None:
            df = None
        else:
            df = self._attribute_table_to_df(rat)

        return df

    def set_attribute_table(self, df: DataFrame, band: int | None = None) -> None:
        """Set the attribute table for a band.

        The attribute table can be used to associate tabular data with the values of a raster band.
        This is particularly useful for categorical raster data, such as land cover classifications,
        where each pixel value corresponds to a category that has additional attributes (e.g., class
        name, color description).

        Notes:
            - The attribute table is stored in an xml file by the name of the raster file with the
              extension of .aux.xml.
            - Setting an attribute table to a band will overwrite the existing attribute table if it
              exists.
            - Setting an attribute table to a band does not need the dataset to be opened in a write
              mode.

        Args:
            df (DataFrame):
                DataFrame with the attribute table.
            band (int):
                Band index.

        Examples:
            - First create a dataset:

              ```python
              >>> from pyramids.dataset import Dataset
              >>> import pandas as pd
              >>> dataset = Dataset.create(
              ... cell_size=0.05, rows=10, columns=10, dtype="float32", bands=1,
              ... top_left_corner=(0, 0), epsg=4326, no_data_value=-9999
              ... )

              ```

            - Create a DataFrame with the attribute table:

              ```python
              >>> data = {
              ...     "Value": [1, 2, 3],
              ...     "ClassName": ["Forest", "Water", "Urban"],
              ...     "Color": ["#008000", "#0000FF", "#808080"],
              ... }
              >>> df = pd.DataFrame(data)

              ```

            - Set the attribute table to the dataset:

              ```python
              >>> dataset.set_attribute_table(df, band=0)

              ```

            - Then the attribute table can be retrieved using the `get_attribute_table` method.
            - The content of the attribute table will be stored in an xml file by the name of the
              raster file with the extension of .aux.xml. The content of the file will be like the
              following:

              ```xml

                  <PAMDataset>
                    <PAMRasterBand band="1">
                      <GDALRasterAttributeTable tableType="thematic">
                        <FieldDefn index="0">
                          <Name>Precipitation Range (mm)</Name>
                          <Type>2</Type>
                          <Usage>0</Usage>
                        </FieldDefn>
                        <FieldDefn index="1">
                          <Name>Category</Name>
                          <Type>2</Type>
                          <Usage>0</Usage>
                        </FieldDefn>
                        <FieldDefn index="2">
                          <Name>Description</Name>
                          <Type>2</Type>
                          <Usage>0</Usage>
                        </FieldDefn>
                        <Row index="0">
                          <F>0-50</F>
                          <F>Low</F>
                          <F>Very low precipitation</F>
                        </Row>
                        <Row index="1">
                          <F>51-100</F>
                          <F>Moderate</F>
                          <F>Moderate precipitation</F>
                        </Row>
                        <Row index="2">
                          <F>101-200</F>
                          <F>High</F>
                          <F>High precipitation</F>
                        </Row>
                        <Row index="3">
                          <F>201-500</F>
                          <F>Very High</F>
                          <F>Very high precipitation</F>
                        </Row>
                        <Row index="4">
                          <F>&gt;500</F>
                          <F>Extreme</F>
                          <F>Extreme precipitation</F>
                        </Row>
                      </GDALRasterAttributeTable>
                    </PAMRasterBand>
                  </PAMDataset>

              ```
        """
        rat = self._df_to_attribute_table(df)
        band = band if band is not None else 0
        band_obj = self._iloc(band)
        band_obj.SetDefaultRAT(rat)

    @staticmethod
    def _df_to_attribute_table(df: DataFrame) -> gdal.RasterAttributeTable:
        """df_to_attribute_table.

            Convert a DataFrame to a GDAL RasterAttributeTable.

        Args:
            df (DataFrame):
                DataFrame with columns to be converted to RAT columns.

        Returns:
            gdal.RasterAttributeTable:
                The resulting RasterAttributeTable.
        """
        # Create a new RasterAttributeTable
        rat = gdal.RasterAttributeTable()

        # Create columns in the RAT based on the DataFrame columns
        for column in df.columns:
            dtype = df[column].dtype
            if pd.api.types.is_integer_dtype(dtype):
                rat.CreateColumn(column, gdal.GFT_Integer, gdal.GFU_Generic)
            elif pd.api.types.is_float_dtype(dtype):
                rat.CreateColumn(column, gdal.GFT_Real, gdal.GFU_Generic)
            else:  # Assume string for any other type
                rat.CreateColumn(column, gdal.GFT_String, gdal.GFU_Generic)

        # Populate the RAT with the DataFrame data
        for row_index in range(len(df)):
            for col_index, column in enumerate(df.columns):
                dtype = df[column].dtype
                value = df.iloc[row_index, col_index]
                if pd.api.types.is_integer_dtype(dtype):
                    rat.SetValueAsInt(row_index, col_index, int(value))
                elif pd.api.types.is_float_dtype(dtype):
                    rat.SetValueAsDouble(row_index, col_index, float(value))
                else:  # Assume string for any other type
                    rat.SetValueAsString(row_index, col_index, str(value))

        return rat

    @staticmethod
    def _attribute_table_to_df(rat: gdal.RasterAttributeTable) -> DataFrame:
        """attribute_table_to_df.

        Convert a GDAL RasterAttributeTable to a pandas DataFrame.

        Args:
            rat (gdal.RasterAttributeTable):
                The RasterAttributeTable to convert.

        Returns:
            pd.DataFrame: The resulting DataFrame.
        """
        columns: list[tuple[str, int]] = []
        data: dict[str, list[Any]] = {}

        # Get the column names and create empty lists for data
        for col_index in range(rat.GetColumnCount()):
            col_name = rat.GetNameOfCol(col_index)
            col_type = rat.GetTypeOfCol(col_index)
            columns.append((col_name, col_type))
            data[col_name] = []

        # Get the row count
        row_count = rat.GetRowCount()

        # Populate the data dictionary with RAT values
        for row_index in range(row_count):
            for col_index, (col_name, col_type) in enumerate(columns):
                if col_type == gdal.GFT_Integer:
                    value = rat.GetValueAsInt(row_index, col_index)
                elif col_type == gdal.GFT_Real:
                    value = rat.GetValueAsDouble(row_index, col_index)
                else:  # gdal.GFT_String
                    value = rat.GetValueAsString(row_index, col_index)
                data[col_name].append(value)

        # Create the DataFrame
        df = pd.DataFrame(data)
        return df

    def add_band(
        self,
        array: np.ndarray,
        unit: Any | None = None,
        attribute_table: DataFrame | None = None,
        inplace: bool = False,
    ) -> None | Dataset:
        """Add a new band to the dataset.

        Args:
            array (np.ndarray):
                2D array to add as a new band.
            unit (Any, optional):
                Unit of the values in the new band.
            attribute_table (DataFrame, optional):
                Attribute table provides a way to associate tabular data with the values of a
                raster band. This is particularly useful for categorical raster data, such as land
                cover classifications, where each pixel value corresponds to a category that has
                additional attributes (e.g., class name, color, description).
                Default is None.
            inplace (bool, optional):
                If True the new band will be added to the current dataset, if False the new band
                will be added to a new dataset. Default is False.

        Returns:
            None

        Examples:
            - First create a dataset:

              ```python
              >>> from pyramids.dataset import Dataset
              >>> dataset = Dataset.create(
              ... cell_size=0.05, rows=10, columns=10, dtype="float32", bands=1,
              ... top_left_corner=(0, 0), epsg=4326, no_data_value=-9999
              ... )
              >>> print(dataset)  # doctest: +NORMALIZE_WHITESPACE
              <BLANKLINE>
                          Top Left Corner: (0.0, 0.0)
                          Cell size: 0.05
                          Dimension: 10 * 10
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

              ```

            - Create a 2D array to add as a new band:

              ```python
              >>> import numpy as np
              >>> array = np.random.rand(10, 10)

              ```

            - Add the new band to the dataset inplace:

              ```python
              >>> dataset.add_band(array, unit="m", attribute_table=None, inplace=True)
              >>> print(dataset)  # doctest: +NORMALIZE_WHITESPACE
              <BLANKLINE>
                          Top Left Corner: (0.0, 0.0)
                          Cell size: 0.05
                          Dimension: 10 * 10
                          EPSG: 4326
                          Number of Bands: 2
                          Band names: ['Band_1', 'Band_2']
                          Band colors: {0: 'undefined', 1: 'undefined'}
                          Band units: ['', 'm']
                          Scale: [1.0, 1.0]
                          Offset: [0, 0]
                          Mask: -9999.0
                          Data type: float32
                          File:

              ```

            - The new band will be added to the dataset inplace.
            - You can also add an attribute table to the band when you add a new band to the
              dataset.

              ```python
              >>> import pandas as pd
              >>> data = {
              ...     "Value": [1, 2, 3],
              ...     "ClassName": ["Forest", "Water", "Urban"],
              ...     "Color": ["#008000", "#0000FF", "#808080"],
              ... }
              >>> df = pd.DataFrame(data)
              >>> dataset.add_band(array, unit="m", attribute_table=df, inplace=True)

              ```

        See Also:
            Dataset.create_from_array: create a new dataset from an array.
            Dataset.create: create a new dataset with an empty band.
            Dataset.dataset_like: create a new dataset from another dataset.
            Dataset.get_attribute_table: get the attribute table for a specific band.
            Dataset.set_attribute_table: Set the attribute table for a specific band.
        """
        # check the dimensions of the new array
        if array.ndim != 2:
            raise ValueError("The array must be 2D.")
        if array.shape[0] != self._ds.rows or array.shape[1] != self._ds.columns:
            raise ValueError(
                f"The array must have the same dimensions as the raster."
                f"{self._ds.rows} {self._ds.columns}"
            )
        # check if the dataset is opened in a write mode
        if inplace:
            if self._ds.access == "read_only":
                raise ValueError("The dataset is not opened in a write mode.")
            else:
                src = self._ds._raster
        else:
            src = gdal.GetDriverByName("MEM").CreateCopy("", self._ds._raster)

        dtype = numpy_to_gdal_dtype(array.dtype)
        num_bands = src.RasterCount
        src.AddBand(dtype, [])
        band = src.GetRasterBand(num_bands + 1)

        if unit is not None:
            band.SetUnitType(unit)

        if attribute_table is not None:
            # Attach the RAT to the raster band
            rat = Bands._df_to_attribute_table(attribute_table)
            band.SetDefaultRAT(rat)

        band.WriteArray(array)

        if inplace:
            self._ds._update_inplace(src, self._ds.access)
            return None
        else:
            return self._ds.__class__(src, self._ds.access)

    def _get_band_names(self) -> list[str]:
        """Get band names from band metadata if exists otherwise will return index [1,2, ...].

        Returns:
            list[str]:
                List of band names.
        """
        names = []
        for i in range(1, self._ds.band_count + 1):
            band = self._ds.raster.GetRasterBand(i)

            if band.GetDescription():
                # Use the band description.
                names.append(band.GetDescription())
            else:
                # Check for metadata.
                band_name = f"Band_{band.GetBand()}"
                metadata = band.GetDataset().GetMetadata_Dict()

                # If in metadata, return the metadata entry, else Band_N.
                if band_name in metadata and metadata[band_name]:
                    names.append(metadata[band_name])
                else:
                    names.append(band_name)

        return names

    def _set_band_names(self, name_list: list) -> None:
        """Set band names from a given list of names.

        Returns:
            list[str]:
                List of band names.
        """
        for i in range(self._ds.band_count):
            # first set the band name in the gdal dataset object
            band = self._ds.raster.GetRasterBand(i + 1)
            band.SetDescription(name_list[i])
            # second, change the band names in the _band_names property.
            self._ds._band_names[i] = name_list[i]

    @property
    def band_color(self) -> dict[int, str]:
        """Band colors."""
        color_dict = {}
        for i in range(self._ds.band_count):
            band_color = self._iloc(i).GetColorInterpretation()
            band_color = band_color if band_color is not None else 0
            color_dict[i] = gdal_constant_to_color_name(band_color)
        return color_dict

    @band_color.setter
    def band_color(self, values: dict[int, str]):
        """Assign color interpretation to dataset bands.

        Args:
            values (Dict[int, str]):
                Dictionary with band index as key and color name as value.
                e.g. {1: 'Red', 2: 'Green', 3: 'Blue'}. Possible values are
                ['undefined', 'gray_index', 'palette_index', 'red', 'green', 'blue',
                'alpha', 'hue', 'saturation', 'lightness', 'cyan', 'magenta', 'yellow',
                'black', 'YCbCr_YBand', 'YCbCr_CbBand', 'YCbCr_CrBand']

        Examples:
            - Create `Dataset` consisting of 1 band, 10 rows, 10 columns, at lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> import pandas as pd
              >>> arr = np.random.randint(1, 3, size=(10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )

              ```

            - Assign a color interpretation to the dataset band (i.e., gray, red, green, or
              blue) using a dictionary with the band index as the key and the color
              interpretation as the value:

              ```python
              >>> dataset.band_color = {0: 'gray_index'}

              ```

            - Assign RGB color interpretation to dataset bands:

              ```python
              >>> arr = np.random.randint(1, 3, size=(3, 10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )
              >>> dataset.band_color = {0: 'red', 1: 'green', 2: 'blue'}

              ```
        """
        for key, val in values.items():
            if key > self._ds.band_count:
                raise ValueError(
                    f"band index should be between 0 and {self._ds.band_count}"
                )
            gdal_const = color_name_to_gdal_constant(val)
            self._iloc(key).SetColorInterpretation(gdal_const)

    def get_band_by_color(self, color_name: str) -> int | None:
        """Get the band associated with a given color.

        Args:
            color_name (str):
                One of ['undefined', 'gray_index', 'palette_index', 'red', 'green',
                'blue', 'alpha', 'hue', 'saturation', 'lightness', 'cyan', 'magenta',
                'yellow', 'black', 'YCbCr_YBand', 'YCbCr_CbBand', 'YCbCr_CrBand'].

        Returns:
            int:
                Band index.

        Examples:
            - Create `Dataset` consisting of 3 bands and assign RGB colors:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.randint(1, 3, size=(3, 10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )
              >>> dataset.band_color = {0: 'red', 1: 'green', 2: 'blue'}

              ```

            - Now use `get_band_by_color` to know which band is the red band, for example:

              ```python
              >>> band = dataset.get_band_by_color('red')
              >>> print(band)
              0

              ```
        """
        colors = list(self.band_color.values())
        if color_name not in colors:
            band = None
        else:
            band = colors.index(color_name)
        return band

    @property
    def color_table(self) -> DataFrame:
        """Color table.

        Returns:
            DataFrame:
                A DataFrame with columns: band, values, color.

        Examples:
            - Create `Dataset` consisting of 4 bands, 10 rows, 10 columns, at lon/lat
              (0, 0):

              ```python
              >>> import numpy as np
              >>> import pandas as pd
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.randint(1, 3, size=(2, 10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )

              ```

            - Set color table for band 1:

              ```python
              >>> color_table = pd.DataFrame({
              ...     "band": [1, 1, 1, 2, 2, 2],
              ...     "values": [1, 2, 3, 1, 2, 3],
              ...     "color": ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C",
              ...         "#D6C19C"]
              ... })
              >>> dataset.color_table = color_table
              >>> print(dataset.color_table)
                band values  red green blue alpha
              0    1      0    0     0    0     0
              1    1      1  112   153   89   255
              2    1      2  242   238  162   255
              3    1      3  242   206  133   255
              4    2      0    0     0    0     0
              5    2      1  194   140  124   255
              6    2      2  214   193  156   255
              7    2      3  214   193  156   255

              ```

            - Define opacity per color by adding an 'alpha' column (0 transparent to 255
              opaque). If 'alpha' is missing, it will be assumed fully opaque (255):

              ```python
              >>> color_table = pd.DataFrame({
              ...     "band": [1, 1, 1, 2, 2, 2],
              ...     "values": [1, 2, 3, 1, 2, 3],
              ...     "color": ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C",
              ...         "#D6C19C"],
              ...     "alpha": [255, 128, 0, 255, 128, 0]
              ... })
              >>> dataset.color_table = color_table
              >>> print(dataset.color_table)
                band values  red green blue alpha
              0    1      0    0     0    0     0
              1    1      1  112   153   89   255
              2    1      2  242   238  162   128
              3    1      3  242   206  133     0
              4    2      0    0     0    0     0
              5    2      1  194   140  124   255
              6    2      2  214   193  156   128
              7    2      3  214   193  156     0

              ```
        """
        return self._get_color_table()

    @color_table.setter
    def color_table(self, df: DataFrame):
        """Get color table.

        Args:
            df (DataFrame):
                DataFrame with columns: band, values, color. Example layout:
                    ```python
                    band  values    color  alpha
                    0    1       1  #709959    255
                    1    1       2  #F2EEA2    255
                    2    1       3  #F2CE85    138
                    3    2       1  #C28C7C    100
                    4    2       2  #D6C19C    100
                    5    2       3  #D6C19C    100

                    ```

        Examples:
            - Create `Dataset` consisting of 4 bands, 10 rows, 10 columns, at lon/lat
              (0, 0):

              ```python
              >>> import numpy as np
              >>> import pandas as pd
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.randint(1, 3, size=(2, 10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )

              ```

            - Set color table for band 1:

              ```python
              >>> color_table = pd.DataFrame({
              ...     "band": [1, 1, 1, 2, 2, 2],
              ...     "values": [1, 2, 3, 1, 2, 3],
              ...     "color": ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C",
              ...         "#D6C19C"]
              ... })
              >>> dataset.color_table = color_table
              >>> print(dataset.color_table)
                band values  red green blue alpha
              0    1      0    0     0    0     0
              1    1      1  112   153   89   255
              2    1      2  242   238  162   255
              3    1      3  242   206  133   255
              4    2      0    0     0    0     0
              5    2      1  194   140  124   255
              6    2      2  214   193  156   255
              7    2      3  214   193  156   255

              ```

            - You can also define the opacity of each color by adding a value between 0
              (fully transparent) and 255 (fully opaque) to the DataFrame for each color.
              If the 'alpha' column is not present, it will be assumed to be fully opaque
              (255):

              ```python
              >>> color_table = pd.DataFrame({
              ...     "band": [1, 1, 1, 2, 2, 2],
              ...     "values": [1, 2, 3, 1, 2, 3],
              ...     "color": ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C",
              ...         "#D6C19C"],
              ...     "alpha": [255, 128, 0, 255, 128, 0]
              ... })
              >>> dataset.color_table = color_table
              >>> print(dataset.color_table)
                band values  red green blue alpha
              0    1      0    0     0    0     0
              1    1      1  112   153   89   255
              2    1      2  242   238  162   128
              3    1      3  242   206  133     0
              4    2      0    0     0    0     0
              5    2      1  194   140  124   255
              6    2      2  214   193  156   128
              7    2      3  214   193  156     0

              ```
        """
        if not isinstance(df, DataFrame):
            raise TypeError(f"df should be a DataFrame not {type(df)}")

        if not {"band", "values", "color"}.issubset(df.columns):
            raise ValueError(  # noqa
                "df should have the following columns: band, values, color"
            )

        self._set_color_table(df, overwrite=True)

    def _set_color_table(self, color_df: DataFrame, overwrite: bool = False) -> None:
        """_set_color_table.

        Args:
            color_df (DataFrame):
                DataFrame with columns: band, values, color. Example:
                ```python
                band  values    color
                0    1       1  #709959
                1    1       2  #F2EEA2
                2    1       3  #F2CE85
                3    2       1  #C28C7C
                4    2       2  #D6C19C
                5    2       3  #D6C19C

                ```
            overwrite (bool):
                True to overwrite the existing color table. Default is False.
        """
        require_cleopatra()
        from cleopatra.colors import Colors

        color = Colors(color_df["color"].tolist())
        color_rgb = color.to_rgb(normalized=False)
        color_df = color_df.copy(deep=True)
        color_df.loc[:, ["red", "green", "blue"]] = color_rgb

        if "alpha" not in color_df.columns:
            color_df.loc[:, "alpha"] = 255

        for band_idx, df_band in color_df.groupby("band"):
            band = self._ds.raster.GetRasterBand(band_idx)

            if overwrite:
                color_table = gdal.ColorTable()
            else:
                color_table = band.GetColorTable()

            for i, row in df_band.iterrows():
                color_table.SetColorEntry(
                    row["values"],
                    (row["red"], row["green"], row["blue"], row["alpha"]),
                )

            band.SetColorTable(color_table)
            # band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    def _get_color_table(self, band: int | None = None) -> DataFrame:
        """Get color table.

        Args:
            band (int, optional):
                Band index. Default is None.

        Returns:
            pandas.DataFrame:
                A DataFrame with columns ["band", "values", "red", "green", "blue",
                "alpha"] describing the color table.
        """
        df = pd.DataFrame(columns=["band", "values", "red", "green", "blue", "alpha"])
        band_iter: Iterable[int] = (
            range(self._ds.band_count) if band is None else [band]
        )
        row = 0
        for band in band_iter:
            color_table = self._ds.raster.GetRasterBand(band + 1).GetRasterColorTable()
            if color_table is None:
                continue  # band has no colour palette (the common case) — skip it
            for i in range(color_table.GetCount()):
                df.loc[row, ["red", "green", "blue", "alpha"]] = (
                    color_table.GetColorEntry(i)
                )
                df.loc[row, ["band", "values"]] = band + 1, i
                row += 1

        return df

    def _coerce_band_no_data(self, i: int, val: Any) -> Any:
        """Coerce one band's no-data value to its dtype.

        Non-``None``/``NaN`` values cast to the band's numpy dtype (may raise
        ``OverflowError`` when out of range); ``None``/``NaN`` on an unsigned band
        uses the dtype max, and passes through unchanged otherwise.
        """
        # if not None or np.nan
        if val is not None and not np.isnan(val):
            # cast to the band dtype (raises OverflowError when out of range)
            result = self._ds.numpy_dtype[i](val)
        elif self._ds.dtype[i].startswith("u"):
            # None/np.nan on an Unsigned integer band would misbehave; use the
            # dtype max bound as the no_data_value.
            result = np.iinfo(self._ds.dtype[i]).max
        else:
            # None/np.nan on any non-unsigned dtype: pass through unchanged.
            result = val
        return result

    def _fallback_no_data(self, i: int) -> Any:
        """Pick a dtype-valid no-data sentinel when the requested value overflows.

        The dtype max for unsigned ints (matching the None/NaN branch), the dtype
        min for signed ints too small to hold the default, and the default for
        floats (which can always represent it).
        """
        np_dtype = np.dtype(self._ds.numpy_dtype[i])
        # np.issubdtype narrows at runtime but isn't recognised by the numpy
        # stubs, so cast to the exact integer family each branch just proved.
        if np.issubdtype(np_dtype, np.unsignedinteger):
            unsigned_dtype = cast("np.dtype[np.unsignedinteger]", np_dtype)
            fallback = np_dtype.type(np.iinfo(unsigned_dtype).max)
        elif np.issubdtype(np_dtype, np.integer):
            signed_dtype = cast("np.dtype[np.signedinteger]", np_dtype)
            info = np.iinfo(signed_dtype)
            if info.min <= DEFAULT_NO_DATA_VALUE <= info.max:
                fallback = np_dtype.type(DEFAULT_NO_DATA_VALUE)
            else:
                fallback = np_dtype.type(info.min)
        else:
            fallback = np_dtype.type(DEFAULT_NO_DATA_VALUE)
        return fallback

    def _check_no_data_value(self, no_data_value: list | tuple) -> list:
        """Validate the no_data_value against each band's dtype.

        Always returns a **fresh list** — the input sequence is
        copied at the top of the function and never mutated, so
        `Dataset.no_data_value` (which returns a tuple) and
        caller-owned lists alike are safe to pass without aliasing
        surprises.

        Args:
            no_data_value:
                Per-band no-data value(s) to validate. Accepts
                both `list` and `tuple`.

        Returns:
            list: A new list with each entry coerced to the
            corresponding band's numpy dtype (or the dtype's max
            for unsigned integer bands when the input is `None` /
            `NaN`).
        """
        no_data_value = list(no_data_value)
        # convert the no_data_value based on the dtype of each raster band.
        for i in range(len(self._ds.gdal_dtype)):
            try:
                no_data_value[i] = self._coerce_band_no_data(i, no_data_value[i])
            except OverflowError:
                # The requested no_data_value does not fit the band dtype (e.g.
                # the default -9999 on a uint16 band, or -3.4e38 on int64); fall
                # back to a dtype-valid sentinel instead of re-casting.
                fallback = self._fallback_no_data(i)
                warnings.warn(
                    f"The no_data_value {no_data_value[i]!r} is out of range for band "
                    f"dtype {np.dtype(self._ds.numpy_dtype[i])}; falling back to {fallback}."
                )
                no_data_value[i] = fallback
        return no_data_value

    def _set_no_data_value(
        self, no_data_value: Any | list = DEFAULT_NO_DATA_VALUE
    ) -> None:
        """setNoDataValue.
            - Set the no data value in all raster bands.
            - Fill the whole raster with the no_data_value.
            - used only when creating an empty driver.
            now the no_data_value is converted to the dtype of the raster bands and updated in the
            dataset attribute, gdal no_data_value attribute, used to fill the raster band.
            from here you have to use the no_data_value stored in the no_data_value attribute as it is updated.
        Args:
            no_data_value (numeric):
                No data value to fill the masked part of the array.
        """
        if not isinstance(no_data_value, (list, tuple)):
            no_data_value = [no_data_value] * self._ds.band_count
        no_data_value = self._check_no_data_value(no_data_value)
        # Read-only is detected from GDAL's Fill() error below, not from
        # self._ds.access: wrapping a gdal.Open(path, GA_Update) handle with the
        # default Dataset(...) constructor reports access == "read_only" while the
        # handle is genuinely writable, so the access flag would block legitimate
        # writes. GDAL's error reflects the handle's true write mode.
        for band in range(self._ds.band_count):
            try:
                # now the no_data_value is converted to the dtype of the raster bands and updated in the
                # dataset attribute, gdal no_data_value attribute, used to fill the raster band.
                # from here you have to use the no_data_value stored in the no_data_value attribute as it is updated.
                self._set_no_data_value_backend(band, no_data_value[band])
            except Exception as e:
                if str(e).__contains__(_GDAL_READ_ONLY_FILL_ERROR):
                    raise ReadOnlyError(
                        "The Dataset is open with a read only, please read the raster using update access mode"
                    )
                elif str(e).__contains__(
                    "in method 'Band_SetNoDataValue', argument 2 of type 'double'"
                ):
                    self._set_no_data_value_backend(
                        band, np.float64(no_data_value[band])
                    )
                else:
                    self._set_no_data_value_backend(band, DEFAULT_NO_DATA_VALUE)
                    self._ds.logger.warning(
                        "the type of the given no_data_value differs from the dtype of the raster"
                        f"no_data_value now is set to {DEFAULT_NO_DATA_VALUE} in the raster"
                    )

    def _calculate_bbox(self) -> list:
        """Calculate bounding box from the geotransform's separate X/Y pixel sizes.

        Uses ``geotransform[1]``/``geotransform[5]`` rather than a single ``cell_size`` so non-square
        grids (e.g. 2° longitude, 1° latitude) are not stretched.
        """
        gt = self._ds.geotransform
        x_min, y_max = gt[0], gt[3]
        x_max = x_min + self._ds.columns * gt[1]
        y_min = y_max + self._ds.rows * gt[5]
        return [x_min, y_min, x_max, y_max]

    def _calculate_bounds(self) -> GeoDataFrame:
        """Get the bbox as a geodataframe with a polygon geometry."""
        x_min, y_min, x_max, y_max = self._calculate_bbox()
        coords = [(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max)]
        poly = create_polygon(coords)
        gdf = gpd.GeoDataFrame(geometry=[poly])
        gdf.set_crs(self._ds.epsg or self._ds.crs, inplace=True)
        return gdf

    def _set_no_data_value_backend(self, band: int, no_data_value: Any) -> None:
        """
            - band starts from 0 to the number of bands-1.
        Args:
            band:
                Band index, starts from 0.
            no_data_value:
                Numerical value.
        """
        # check if the dtype of the no_data_value complies with the dtype of the raster itself.
        self._change_no_data_value_attr(band, no_data_value)
        # initialize the band with the nodata value instead of 0
        # the no_data_value may have changed inside the _change_no_data_value_attr method to float64, so redefine it.
        no_data_value = self._ds.no_data_value[band]
        try:
            self._ds.raster.GetRasterBand(band + 1).Fill(no_data_value)
        except Exception as e:
            if str(e).__contains__(" argument 2 of type 'double'"):
                self._ds.raster.GetRasterBand(band + 1).Fill(np.float64(no_data_value))
            elif str(e).__contains__(_GDAL_READ_ONLY_FILL_ERROR) or str(e).__contains__(
                "attempt to write to dataset opened in read-only mode."
            ):
                raise ReadOnlyError(
                    "The Dataset is open with a read only, please read the raster using update access mode"
                )
            else:
                raise ValueError(
                    f"Failed to fill the band {band} with value: {no_data_value}, because of {e}"
                )
        # update the no_data_value in the Dataset object
        self._ds._no_data_value[band] = no_data_value

    def _change_no_data_value_attr(self, band: int, no_data_value) -> None:
        """Change the no_data_value attribute.
            - Change only the no_data_value attribute in the gdal Dataset object.
            - Change the no_data_value in the Dataset object for the given band index.
            - The corresponding value in the array will not be changed.
        Args:
            band (int):
                Band index, starts from 0.
            no_data_value (Any):
                No data value.
        """
        try:
            self._ds.raster.GetRasterBand(band + 1).SetNoDataValue(no_data_value)
        except Exception as e:
            if str(e).__contains__(_GDAL_READ_ONLY_FILL_ERROR):
                raise ReadOnlyError(
                    "The Dataset is open with a read only, please read the raster using update "
                    "access mode"
                )
            # TypeError
            elif e.args == (
                "in method 'Band_SetNoDataValue', argument 2 of type 'double'",
            ):
                no_data_value = np.float64(no_data_value)
                self._ds.raster.GetRasterBand(band + 1).SetNoDataValue(no_data_value)
        self._ds._no_data_value[band] = no_data_value

    def change_no_data_value(
        self, new_value: Any, old_value: Any | None = None, inplace: bool = False
    ) -> Dataset | None:
        """Change No Data Value.
            - Set the no data value in all raster bands.
            - Fill the whole raster with the no_data_value.
            - Change the no_data_value in the array in all bands.
        Args:
            new_value (numeric):
                No data value to set in the raster bands. Only a `list` is read per band, and it must
                have `band_count` entries; any other value takes the scalar path and is applied to every
                band, so it must be a numeric scalar. A `tuple` of per-band values is not supported.
            old_value (numeric):
                Old no data value that is already in the raster bands. Follows the same per-band `list`
                convention as `new_value`.
            inplace (bool):
                If True, the original dataset will be modified. If False, a new dataset will be created.
                Default is False.

        Returns:
            Dataset | None:
                A new Dataset with the updated no-data value, or ``None``
                when ``inplace=True`` -- see :meth:`Analysis.apply` for why.

        Raises:
            NoDataValueError:
                If `new_value` cannot be stored in a band's dtype — e.g. `None` or `NaN`
                given for an integer band — the dtype mismatch is reported instead of
                leaking a raw numpy `TypeError`/`ValueError`. Also raised when
                `new_value` or `old_value` is given as a list whose length does not
                match `band_count`.

        Warning:
            The `change_no_data_value` method creates a new dataset in memory in order to change the `no_data_value` in the raster bands.
        Examples:
            - Create a Dataset (4 bands, 10 rows, 10 columns) at lon/lat (0, 0):
              ```python
              >>> from pyramids.dataset import Dataset
              >>> dataset = Dataset.create(
              ...     cell_size=0.05, rows=3, columns=3, bands=1, top_left_corner=(0, 0),dtype="float32",
              ...     epsg=4326, no_data_value=-9
              ... )
              >>> arr = dataset.read_array()
              >>> print(arr)
              [[-9. -9. -9.]
               [-9. -9. -9.]
               [-9. -9. -9.]]
              >>> print(dataset.no_data_value) # doctest: +SKIP
              [-9.0]

              ```
            - The dataset is full of the no_data_value. Now change it using `change_no_data_value`:
              ```python
              >>> new_dataset = dataset.change_no_data_value(-10, -9)
              >>> arr = new_dataset.read_array()
              >>> print(arr)
              [[-10. -10. -10.]
               [-10. -10. -10.]
               [-10. -10. -10.]]
              >>> print(new_dataset.no_data_value) # doctest: +SKIP
              [-10.0]

              ```
        """
        if not isinstance(new_value, list):
            new_value = [new_value] * self._ds.band_count
        if len(new_value) != self._ds.band_count:
            raise NoDataValueError(
                f"new_value must be a scalar or a list of length band_count "
                f"({self._ds.band_count}); got a list of length {len(new_value)}."
            )
        if old_value is not None and not isinstance(old_value, list):
            old_value = [old_value] * self._ds.band_count
        if old_value is not None and len(old_value) != self._ds.band_count:
            raise NoDataValueError(
                f"old_value must be a scalar or a list of length band_count "
                f"({self._ds.band_count}); got a list of length {len(old_value)}."
            )
        dst = gdal.GetDriverByName("MEM").CreateCopy("", self._ds.raster, 0)
        # create a new dataset
        new_dataset = self._ds.__class__(dst, "write")
        # the new_value could change inside the _set_no_data_value method before it is used to set the no_data_value
        # attribute in the gdal object/pyramids object and to fill the band.
        new_dataset._set_no_data_value(new_value)
        # now we have to use the no_data_value value in the no_data_value attribute in the Dataset object as it is
        # updated.
        new_value = new_dataset.no_data_value
        for band in range(self._ds.band_count):
            arr = self._ds.read_array(band)
            # old_value is normalized to a per-band list above (matching
            # new_value); index it per-band here too instead of comparing
            # against the whole list.
            band_old_value = old_value[band] if old_value is not None else None
            try:
                with np.errstate(invalid="raise"):
                    arr[is_no_data(arr, band_old_value)] = new_value[band]
            # A dtype mismatch surfaces differently across numpy paths: a None value
            # is not subscriptable (TypeError), a NaN cast into an integer band raises
            # ValueError ("cannot convert float NaN to integer"), and an invalid
            # floating-point cast trips errstate (FloatingPointError). Map all of them
            # to the package-level NoDataValueError.
            except (TypeError, ValueError, FloatingPointError):
                raise NoDataValueError(
                    f"The dtype of the given no_data_value: {new_value[band]} differs from the dtype of the "
                    f"band: {gdal_to_numpy_dtype(self._ds.gdal_dtype[band])}"
                )
            new_dataset.raster.GetRasterBand(band + 1).WriteArray(arr)

        if inplace:
            self._ds._update_inplace(new_dataset.raster)
            return None
        return new_dataset
