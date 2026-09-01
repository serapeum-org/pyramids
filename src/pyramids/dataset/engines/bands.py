"""Bands engine.

Owns the Bands family of operations on a Dataset. Accessed as
``ds.bands``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.bands.<method>(...)`` are equivalent.
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

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

from pyramids.base.crs import crs_spec
from pyramids.dataset._driver import MEMORY_DRIVER, resolve_output_driver
from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.engines._validate import validate_band_index

# Substring GDAL raises when a write is attempted on a read-only band; matched in
# several no-data setters below to re-raise a friendly ReadOnlyError. Named once
# to avoid duplicating the literal (S1192). GDAL prefixes it with the file, band
# and refusing call -- "ro.tif, band 1: GDALRasterBand::Fill(): attempt to write
# to dataset opened in read-only mode." -- so the test has to be a substring one.
_GDAL_READ_ONLY_MESSAGE = "attempt to write to dataset opened in read-only mode."


def _is_read_only_error(error: BaseException) -> bool:
    """Whether a GDAL exception means "this dataset is open read-only".

    GDAL signals it with a plain `RuntimeError` carrying the file, band and
    refusing call ahead of the reason, so the classification is a substring
    match. Kept in one place: it was repeated at three no-data setters, each
    testing a different subset of the wordings, which meant two of them never
    recognised a refusal at all and surfaced a raw `RuntimeError` instead of
    `ReadOnlyError`. Consolidating them is a deliberate behaviour change --
    those two paths now raise `ReadOnlyError` where they previously did not.

    Note that not every write to a read-only dataset reaches here.
    `SetNoDataValue` on a read-only GTiff succeeds on GDAL 3.13 by writing to
    the PAM sidecar instead of refusing, so its caller's guard is what stops it,
    not this classifier.

    Args:
        error: The exception raised by GDAL.

    Returns:
        bool: `True` when the message matches GDAL's read-only refusal.
    """
    return _GDAL_READ_ONLY_MESSAGE in str(error)


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
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> import pandas as pd
              >>> dataset = Dataset.create(
              ...     rows=5,
              ...     columns=5,
              ...     dtype="float32",
              ...     bands=1,
              ...     no_data_value=-9999,
              ...     geo_ref=GeoReference(cell_size=0.05, top_left_corner=(0, 0), epsg=4326),
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
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> import pandas as pd
              >>> dataset = Dataset.create(
              ...     rows=10,
              ...     columns=10,
              ...     dtype="float32",
              ...     bands=1,
              ...     no_data_value=-9999,
              ...     geo_ref=GeoReference(cell_size=0.05, top_left_corner=(0, 0), epsg=4326),
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
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> dataset = Dataset.create(
              ...     rows=10,
              ...     columns=10,
              ...     dtype="float32",
              ...     bands=1,
              ...     no_data_value=-9999,
              ...     geo_ref=GeoReference(cell_size=0.05, top_left_corner=(0, 0), epsg=4326),
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
            Dataset.from_array: create a new dataset from an array.
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

    def _resolve_band_selectors(self, bands: Any) -> list[int]:
        """Resolve band selectors to validated 1-based GDAL band indices.

        Each selector is a **1-based** band index or a band name (matched against
        :attr:`~pyramids.dataset.Dataset.band_names`, first match wins). Order is
        preserved and duplicates are allowed. Everything is validated *before* any
        GDAL call, so an invalid selector raises a clear error rather than a raw
        GDAL ``RuntimeError`` (out-of-range) or a silent all-bands read (empty).

        Args:
            bands: A list or tuple of selectors — 1-based ``int`` indices and/or
                band-name ``str``s.

        Returns:
            The 1-based GDAL band indices, in the requested order.

        Raises:
            TypeError: ``bands`` is not a list/tuple, or a selector is a ``bool``
                or an unsupported type.
            ValueError: ``bands`` is empty, an index is out of range, or a name is
                not among the band names.
        """
        if not isinstance(bands, (list, tuple)):
            raise TypeError(
                "select expects a list or tuple of band indices or names, got "
                f"{type(bands).__name__}"
            )
        if len(bands) == 0:
            raise ValueError("select requires at least one band; got an empty list.")
        count = self._ds.band_count
        if count == 0:
            raise ValueError(
                "this dataset has no bands to select; if it is a NetCDF container, "
                "extract a variable first (e.g. with get_variable)."
            )
        names = self._ds.band_names
        indices = [self._resolve_one_selector(sel, count, names) for sel in bands]
        return indices

    def _resolve_one_selector(self, selector: Any, count: int, names: list[str]) -> int:
        """Resolve a single band selector to a validated 1-based GDAL index.

        Args:
            selector: A 1-based ``int`` index (numpy ints accepted) or a band-name
                ``str`` matched against ``names``.
            count: The dataset's band count, for range validation.
            names: The dataset's band names, for name resolution.

        Returns:
            The 1-based GDAL band index for ``selector``.

        Raises:
            TypeError: ``selector`` is a ``bool`` or an unsupported type.
            ValueError: an index is out of range, or a name is not among ``names``.
        """
        if isinstance(selector, bool):
            raise TypeError(
                f"band selector must be an int or str, not bool: {selector!r}"
            )
        if isinstance(selector, numbers.Integral):
            # numbers.Integral admits numpy ints too; normalize to a plain int.
            one_based = int(selector)
            if not 1 <= one_based <= count:
                raise ValueError(
                    f"band index {one_based} is out of range for a {count}-band "
                    f"dataset (valid 1..{count})."
                )
        elif isinstance(selector, str):
            if selector not in names:
                raise ValueError(f"{selector!r} is not a band name; available: {names}")
            one_based = names.index(selector) + 1
        else:
            raise TypeError(
                "band selector must be an int index or str name, got "
                f"{type(selector).__name__}"
            )
        return one_based

    def select(
        self, bands: list[int | str] | tuple[int | str, ...], *, lazy: bool = False
    ) -> Dataset:
        """Return a new Dataset with a subset of bands, in the requested order.

        Copies the chosen bands (via GDAL ``Translate`` with a ``bandList``) into a
        fresh raster, carrying each band's per-band state — name/description,
        no-data value, scale, offset, unit, colour table and interpretation, band
        metadata, and the raster attribute table (category names). Band selectors
        are **1-based** (matching the product band numbering, e.g. ``select([4, 8])``
        for Sentinel-2 B04/B08) — note this differs from the **0-based** ``band``
        argument of :meth:`~pyramids.dataset.engines.IO.read_array`. Duplicates are
        allowed (e.g. to expand a single band into an RGB triplet).

        The result is a **base** :class:`~pyramids.dataset.Dataset` even when called
        on a subclass: a band subset is an ordinary classic-mode raster, so — as
        with :meth:`~pyramids.dataset.Dataset.open_subdataset` — a ``NetCDF``
        variable subset is returned as a plain raster rather than a re-parsed
        multidimensional view.

        Args:
            bands: The bands to keep, as a list/tuple of 1-based ``int`` indices
                and/or band-name ``str``s. Order is preserved; duplicates allowed.
            lazy: When ``True``, return a VRT-backed view that references the source
                (deferred, memory-light); when ``False`` (default), materialize an
                in-memory copy.

        Returns:
            Dataset: A new base ``Dataset`` holding the selected bands in order.

        Raises:
            TypeError: ``bands`` is not a list/tuple, or a selector is a ``bool`` or
                an unsupported type.
            ValueError: ``bands`` is empty, an index is out of range, or a name is
                not among the band names.

        Examples:
            - Select and reorder two bands of a three-band raster:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.arange(3 * 4).reshape(3, 2, 2).astype("float32")
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> subset = ds.bands.select([3, 1])
                >>> subset.band_count
                2

                ```
        """
        # Local import breaks the engines.bands -> dataset import cycle; select
        # returns a base Dataset (a band subset is a classic raster, mirroring
        # open_subdataset), so it cannot use `self._ds.__class__`.
        from pyramids.dataset.dataset import Dataset

        indices = self._resolve_band_selectors(bands)
        options = gdal.TranslateOptions(
            bandList=indices, format="VRT" if lazy else "MEM"
        )
        out = gdal.Translate("", self._ds.raster, options=options)
        if not lazy:
            # The MEM path drops the unit type; re-apply it per selected band (a
            # harmless no-op on the VRT path, which carries units natively).
            src_units = self._ds.band_units
            for position, one_based in enumerate(indices):
                out.GetRasterBand(position + 1).SetUnitType(
                    src_units[one_based - 1] or ""
                )
        result = Dataset(out)
        return result

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
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> import numpy as np
              >>> import pandas as pd
              >>> arr = np.random.randint(1, 3, size=(10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
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
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )
              >>> dataset.band_color = {0: 'red', 1: 'green', 2: 'blue'}

              ```
        """
        for key, val in values.items():
            # `>= band_count`, via the shared check: the old `key > band_count`
            # let index 1 through on a 1-band dataset (indices are 0-based) and
            # never rejected a negative one, so `_iloc` failed further in with a
            # GDAL error instead of a clear ValueError here.
            validate_band_index(key, self._ds.band_count)
            gdal_const = color_name_to_gdal_constant(val)
            self._iloc(key).SetColorInterpretation(gdal_const)

    @property
    def metadata(self) -> list[dict[str, str]]:
        """Per-band metadata (default domain), one mapping per band, in band order.

        This is the per-band sibling of the dataset-level :attr:`Dataset.meta_data`.
        It is where GDAL stores what a band physically *is* — for a Sentinel-2 band,
        the centre wavelength, bandwidth, and solar irradiance; for Sentinel-1, the
        swath and polarization; and so on for any format's per-band tags. Only the
        default metadata domain is exposed (``IMAGE_STRUCTURE`` and other domains are
        out of scope until domain support lands).

        Returns:
            list[dict[str, str]]: One mapping per band, indexed like
            :attr:`Dataset.band_names` (0-based, band order). A band that carries no
            metadata yields an empty ``dict`` (never ``None``), so callers can index
            without guarding.

        Examples:
            - A two-band raster whose bands carry metadata reports it per band:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.zeros((2, 4, 4), dtype="int16")
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> dataset.band_meta_data = [{"WAVELENGTH": "443"}, {"WAVELENGTH": "490"}]
              >>> dataset.bands.metadata[0]["WAVELENGTH"]
              '443'
              >>> dataset.bands.metadata[1]["WAVELENGTH"]
              '490'

              ```

            - A band with no metadata yields an empty mapping, not ``None``:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> arr = np.zeros((1, 4, 4), dtype="int16")
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> dataset.bands.metadata
              [{}]

              ```
        """
        return self.get_metadata()

    @metadata.setter
    def metadata(self, value: list[dict[str, str]]) -> None:
        """Replace each band's default-domain metadata, one mapping per band.

        The assignment **replaces** each band's metadata with the mapping given for
        it (``SetMetadata``), rather than merging into what is already there — a
        whole-list assignment reads as "set each band's metadata to exactly this",
        consistent with the other per-band list setters (``band_units``, ``scale``,
        ``offset``). Assigning an empty mapping to a band therefore clears it. To edit
        a single key without disturbing the rest, read the mapping, update it, and
        assign the list back. (The dataset-level :attr:`Dataset.meta_data` setter
        merges instead, because it is a single dict, not a per-band list.)

        Args:
            value: One mapping per band, in band order; its length must equal the
                band count.

        Raises:
            ReadOnlyError: The dataset is a read-only on-disk file (setting band
                metadata would spill a PAM ``.aux.xml`` sidecar); reopen with
                ``read_only=False`` or edit an in-memory copy instead.
            ValueError: ``value`` does not carry exactly one mapping per band.
        """
        self.set_metadata(value)

    @overload
    def get_metadata(
        self, band: None = ..., domain: str = ...
    ) -> list[dict[str, str]]: ...

    @overload
    def get_metadata(self, band: int, domain: str = ...) -> dict[str, str]: ...

    def get_metadata(
        self, band: int | None = None, domain: str = ""
    ) -> list[dict[str, str]] | dict[str, str]:
        """Read per-band metadata from a metadata domain.

        The domain-aware form behind :attr:`metadata`. The default domain (``""``)
        holds format-specific band semantics; other GDAL domains expose other tags,
        e.g. ``"IMAGE_STRUCTURE"`` carries ``NBITS`` / ``COMPRESSION`` / ``PIXELTYPE``.

        Args:
            band: A 0-based band index to read just that band, or ``None`` (default)
                to read every band in band order.
            domain: The GDAL metadata domain to read; ``""`` (default) is the default
                domain.

        Returns:
            list[dict[str, str]] | dict[str, str]: One mapping per band when ``band``
            is ``None``; a single band's mapping when ``band`` is given. A band with
            no metadata in the domain yields an empty ``dict`` (never ``None``).

        Raises:
            IndexError: ``band`` is out of range.

        Examples:
            - Read the ``IMAGE_STRUCTURE`` domain of a single band:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> dataset = Dataset.from_array(
              ...     np.zeros((4, 4), dtype="int16"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> dataset.bands.get_metadata(band=0, domain="IMAGE_STRUCTURE")
              {}

              ```
        """
        if band is None:
            return [
                cast(dict[str, str], self._iloc(i).GetMetadata(domain))
                for i in range(self._ds.band_count)
            ]
        return cast(dict[str, str], self._iloc(band).GetMetadata(domain))

    def set_metadata(
        self,
        value: list[dict[str, str]] | dict[str, str],
        band: int | None = None,
        domain: str = "",
    ) -> None:
        """Replace per-band metadata in a metadata domain.

        The domain-aware form behind the :attr:`metadata` setter. Replaces (does not
        merge) the metadata of the target band(s) in ``domain``; assigning an empty
        mapping clears it. Use :meth:`set_metadata_item` to change one key in place.

        Args:
            value: One mapping per band (a list, when ``band`` is ``None``) or a
                single band's mapping (a dict, when ``band`` is given).
            band: A 0-based band index to set just that band, or ``None`` (default) to
                set every band from a per-band list.
            domain: The GDAL metadata domain to write; ``""`` (default) is the default
                domain.

        Raises:
            ReadOnlyError: The dataset is a read-only on-disk file.
            ValueError: ``band`` is ``None`` but ``value`` does not carry exactly one
                mapping per band.
            IndexError: ``band`` is out of range.
        """
        self._ds._require_writable("set band metadata")
        if band is None:
            if len(value) != self._ds.band_count:
                raise ValueError(
                    f"band_meta_data needs one mapping per band: expected "
                    f"{self._ds.band_count}, got {len(value)}."
                )
            for i, band_md in enumerate(cast(list[dict[str, str]], value)):
                self._iloc(i).SetMetadata(band_md, domain)
        else:
            self._iloc(band).SetMetadata(cast(dict[str, str], value), domain)

    def set_metadata_item(
        self, key: str, value: str, band: int = 0, domain: str = ""
    ) -> None:
        """Set a single band-metadata key in place, leaving the rest untouched.

        The merge counterpart to :meth:`set_metadata` (which replaces a band's whole
        mapping): this updates or adds one ``key`` on one band via GDAL's
        ``SetMetadataItem``.

        Args:
            key: The metadata key to set.
            value: The value to store.
            band: The 0-based band index. Defaults to the first band.
            domain: The GDAL metadata domain to write; ``""`` (default) is the default
                domain.

        Raises:
            ReadOnlyError: The dataset is a read-only on-disk file.
            IndexError: ``band`` is out of range.
        """
        self._ds._require_writable("set band metadata")
        self._iloc(band).SetMetadataItem(key, value, domain)

    @property
    def band_units(self) -> list[str]:
        """Per-band unit labels, one per band, in band order."""
        return self._ds._band_units

    @band_units.setter
    def band_units(self, value: list[str]) -> None:
        """Relabel each band's unit.

        This only relabels; it does not convert the stored values — see
        :meth:`Dataset.convert_units` for a value-transforming conversion.

        Raises:
            ReadOnlyError: The dataset is a read-only on-disk file.
        """
        self._ds._require_writable("set band units")
        self._ds._band_units = value
        for i, val in enumerate(value):
            self._iloc(i).SetUnitType(val)

    @property
    def scale(self) -> list[float]:
        """Per-band scale factors (pixel value -> real-world value).

        Returns:
            list[float]: One scale per band; ``1.0`` for a band with no scale set.
        """
        scale_list = []
        for i in range(self._ds.band_count):
            band_scale = self._iloc(i).GetScale()
            scale_list.append(band_scale if band_scale is not None else 1.0)
        return scale_list

    @scale.setter
    def scale(self, value: list[float]) -> None:
        """Set each band's scale factor.

        Raises:
            ReadOnlyError: The dataset is a read-only on-disk file.
        """
        self._ds._require_writable("set the band scale")
        for i, val in enumerate(value):
            self._iloc(i).SetScale(val)

    @property
    def offset(self) -> list[float]:
        """Per-band offsets (pixel value -> real-world value).

        Returns:
            list[float]: One offset per band; ``0`` for a band with no offset set.
        """
        offset_list = []
        for i in range(self._ds.band_count):
            band_offset = self._iloc(i).GetOffset()
            offset_list.append(band_offset if band_offset is not None else 0)
        return offset_list

    @offset.setter
    def offset(self, value: list[float]) -> None:
        """Set each band's offset.

        Raises:
            ReadOnlyError: The dataset is a read-only on-disk file.
        """
        self._ds._require_writable("set the band offset")
        for i, val in enumerate(value):
            self._iloc(i).SetOffset(val)

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
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1, 3, size=(3, 10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
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
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1, 3, size=(2, 10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
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
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1, 3, size=(2, 10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
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

    def set_color_ramp(
        self,
        band: int = 1,
        *,
        start_value: int,
        end_value: int,
        start_color: str | None = None,
        end_color: str | None = None,
        colormap: str | None = None,
    ) -> None:
        """Attach a colour table interpolated across a value range.

        Fills every integer value in `[start_value, end_value]` with a colour, so a
        continuous raster does not need every stop enumerated by hand the way the plain
        `color_table` setter demands. Exactly one of two modes must be given:

        - a **two-colour linear ramp** between `start_color` and `end_color`, built with
          `gdal.ColorTable.CreateColorRamp`;
        - a named matplotlib **`colormap`** (e.g. `"viridis"`), sampled evenly across the
          range.

        The generated entries flow through the same `_set_color_table` path the enumerated
        setter uses, so the attached palette matches it in form. Needs the `[viz]` extra
        (the `colormap` mode also uses its matplotlib).

        Args:
            band (int):
                1-based band to colour. Defaults to 1.
            start_value (int):
                First value in the ramp; must be `>= 0` (GDAL colour indices are
                non-negative). Keyword-only.
            end_value (int):
                Last value in the ramp; must exceed `start_value` (keyword-only). One
                colour entry is written per integer in `[start_value, end_value]`, so a
                very wide range (e.g. every UInt16 value, 0..65535) builds a
                correspondingly large table — keep the range to the classes you actually
                need. The GDAL palette is dense from index 0, so the table always spans
                `0..end_value`: a narrow ramp at a high `start_value` still allocates every
                lower index, and values below `start_value` are left transparent
                `(0, 0, 0, 0)`. Cost therefore scales with `end_value`, not the range width.
            start_color (str, optional):
                Hex colour at `start_value`. Give together with `end_color`, and not with
                `colormap`.
            end_color (str, optional):
                Hex colour at `end_value`. Give together with `start_color`.
            colormap (str, optional):
                Named matplotlib colormap sampled across the range. Give instead of the
                `start_color` / `end_color` pair.

        Raises:
            TypeError:
                `start_value` or `end_value` is not an integer.
            ValueError:
                `band` is outside `1..band_count`; `start_value` is negative; `end_value`
                is not greater than `start_value`; only one of `start_color` / `end_color`
                is given (or one is blank); the mode is ambiguous (neither a colour pair
                nor a colormap, or both); or `colormap` is not a known matplotlib name.

        Examples:
            - A two-colour ramp green -> tan across values 1..5 fills the three
              intermediate stops for you:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1, 6, size=(10, 10))
              >>> ds = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> ds.set_color_ramp(
              ...     band=1, start_value=1, end_value=5,
              ...     start_color="#709959", end_color="#F2CE85",
              ... )
              >>> print(ds.color_table)
                band values  red green blue alpha
              0    1      0    0     0    0     0
              1    1      1  112   153   89   255
              2    1      2  144   166  100   255
              3    1      3  177   179  111   255
              4    1      4  209   192  122   255
              5    1      5  242   206  133   255

              ```

            - A named matplotlib colormap is sampled evenly across the range instead of a
              two-colour pair; value 1 is viridis' dark-purple start:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.randint(1, 6, size=(10, 10))
              >>> ds = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> ds.set_color_ramp(band=1, start_value=1, end_value=5, colormap="viridis")
              >>> row = ds.color_table.set_index("values").loc[1, ["red", "green", "blue"]]
              >>> [int(component) for component in row]
              [68, 1, 84]

              ```

        See Also:
            color_table: The lower-level setter that takes an explicit per-value colour
                table; `set_color_ramp` generates that table from a ramp and forwards it
                through the same path.
        """
        start_value, end_value = self._validate_color_ramp_args(
            band, start_value, end_value, start_color, end_color, colormap
        )

        require_cleopatra()
        from cleopatra.styling.colors import Colors

        if colormap:
            ramp = self._ramp_from_colormap(colormap, start_value, end_value)
        else:
            # ``_validate_color_ramp_args`` guarantees the colour pair is present when
            # ``colormap`` is unset (exactly one of the pair / colormap is given), but mypy
            # can't follow that cross-method narrowing — now that the mypy override reads
            # cleopatra's inline types, ``Colors`` needs concrete ``str`` colours.
            assert start_color is not None
            assert end_color is not None
            start_rgb, end_rgb = Colors([start_color, end_color]).to_rgb(
                normalized=False
            )
            ramp = gdal.ColorTable()
            ramp.CreateColorRamp(
                start_value, (*start_rgb, 255), end_value, (*end_rgb, 255)
            )

        rows = []
        for value in range(start_value, end_value + 1):
            entry = ramp.GetColorEntry(value)
            rows.append(
                {
                    "band": band,
                    "values": value,
                    "color": "#{:02x}{:02x}{:02x}".format(*entry[:3]),
                    "alpha": entry[3],
                }
            )
        self._set_color_table(DataFrame(rows), overwrite=True)

    def _validate_color_ramp_args(
        self,
        band: int,
        start_value: int,
        end_value: int,
        start_color: str | None,
        end_color: str | None,
        colormap: str | None,
    ) -> tuple[int, int]:
        """Validate `set_color_ramp` inputs and return the coerced integer range.

        Raises the same `TypeError` / `ValueError` documented on `set_color_ramp`.
        Extracted to keep the public method's cognitive complexity low.

        Args:
            band (int):
                1-based band index to colour.
            start_value (int):
                First value in the ramp.
            end_value (int):
                Last value in the ramp.
            start_color (str, optional):
                Hex colour at `start_value`, paired with `end_color`.
            end_color (str, optional):
                Hex colour at `end_value`, paired with `start_color`.
            colormap (str, optional):
                Named matplotlib colormap, given instead of the colour pair.

        Returns:
            tuple[int, int]:
                The coerced `(start_value, end_value)` integers.
        """
        if not 1 <= band <= self._ds.band_count:
            raise ValueError(
                f"band {band} is out of range for a {self._ds.band_count}-band "
                "dataset (bands are 1-based)"
            )
        for name, value in (("start_value", start_value), ("end_value", end_value)):
            # bool is an int subclass but not a meaningful colour index; reject it
            # (and any non-numeric type) with the documented TypeError rather than a
            # cryptic downstream one. float('inf').is_integer() is False, so a
            # non-finite float lands here too instead of raising OverflowError.
            if isinstance(value, bool) or not isinstance(
                value, (int, np.integer, float, np.floating)
            ):
                raise TypeError(
                    f"{name} must be an integer, not {type(value).__name__}"
                )
            if not float(value).is_integer():
                raise TypeError(f"{name} must be a whole number, got {value!r}")
        start_value, end_value = int(start_value), int(end_value)
        if start_value < 0:
            raise ValueError(
                f"start_value ({start_value}) must be >= 0: GDAL colour indices are "
                "non-negative"
            )
        if end_value <= start_value:
            raise ValueError(
                f"end_value ({end_value}) must be greater than start_value "
                f"({start_value})"
            )
        # Treat an empty/blank string as "not given", so the mode guards reject it with a
        # clear message instead of letting it reach cleopatra/matplotlib as a cryptic one.
        has_pair = bool(start_color) and bool(end_color)
        if bool(start_color) != bool(end_color):
            raise ValueError("start_color and end_color must both be given")
        if has_pair == bool(colormap):
            raise ValueError(
                "provide exactly one of a (start_color, end_color) pair or a colormap="
            )
        return start_value, end_value

    @staticmethod
    def _ramp_from_colormap(
        colormap: str, start_value: int, end_value: int
    ) -> gdal.ColorTable:
        """Sample a named matplotlib colormap evenly across `[start_value, end_value]`.

        Args:
            colormap (str):
                Named matplotlib colormap (e.g. `"viridis"`).
            start_value (int):
                First value in the ramp.
            end_value (int):
                Last value in the ramp.

        Returns:
            gdal.ColorTable:
                A colour table with one opaque entry per value in the range.
        """
        # matplotlib is a hard cleopatra dependency, so it is importable once
        # require_cleopatra() has passed (the caller enforces that); imported here for
        # the same reason _set_color_table imports cleopatra lazily (optional viz extra).
        from matplotlib import colormaps

        if colormap not in colormaps:
            raise ValueError(
                f"unknown colormap {colormap!r}; pass a name from matplotlib's "
                "registry (e.g. 'viridis', 'terrain')"
            )
        cmap = colormaps[colormap]
        span = end_value - start_value
        ramp = gdal.ColorTable()
        for offset in range(span + 1):
            red, green, blue, _ = cmap(offset / span)
            ramp.SetColorEntry(
                start_value + offset,
                (round(red * 255), round(green * 255), round(blue * 255), 255),
            )
        return ramp

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
        from cleopatra.styling.colors import Colors

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
                if _is_read_only_error(e):
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
        gdf.set_crs(crs_spec(self._ds.epsg, self._ds.crs), inplace=True)
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
            elif _is_read_only_error(e):
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
            if _is_read_only_error(e):
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

    def _normalize_no_data_arg(self, value: Any, name: str) -> list:
        """Normalize a scalar or per-band no-data value to a list of length band_count.

        Args:
            value: A scalar (broadcast to every band) or a per-band list.
            name: The argument name, used in the error message.

        Returns:
            list: A per-band list of length `band_count`.

        Raises:
            NoDataValueError: `value` is a list whose length is not `band_count`.
        """
        if not isinstance(value, list):
            return [value] * self._ds.band_count
        if len(value) != self._ds.band_count:
            raise NoDataValueError(
                f"{name} must be a scalar or a list of length band_count "
                f"({self._ds.band_count}); got a list of length {len(value)}."
            )
        return value

    def _swap_all_bands(
        self, new_dataset: Dataset, new_value: Any, old_value: list | None
    ) -> None:
        """Swap every band's old no-data cells to the new value, tile by tile.

        Args:
            new_dataset: The cloned destination written in place.
            new_value: Per-band new no-data values (already dtype-coerced).
            old_value: Per-band old no-data values, or `None` to match NaN.
        """
        for band in range(self._ds.band_count):
            band_old_value = old_value[band] if old_value is not None else None
            self._swap_no_data_tiled(new_dataset, band, band_old_value, new_value[band])

    @staticmethod
    def _discard_partial_output(new_dataset: Dataset, target: str) -> None:
        """Release the handle and delete a partially-written disk output (best-effort).

        Closes the wrapper (needed with the caller dropping its own `dst` reference,
        or GDAL keeps the file locked on Windows) and unlinks the partial file plus
        its sidecar, swallowing `OSError` so a residual lock never masks the original
        exception -- the file just lingers.

        Args:
            new_dataset: The destination wrapper to close.
            target: The output file path whose partial file/sidecar to remove.
        """
        new_dataset.close()
        for leftover in (target, f"{target}.aux.xml"):
            try:
                Path(leftover).unlink()
            except OSError:
                pass

    def change_no_data_value(
        self,
        new_value: Any,
        old_value: Any | None = None,
        inplace: bool = False,
        *,
        path: str | Path | None = None,
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
            path (str | Path | None):
                Destination for a disk-backed result. Its extension alone selects the
                driver (`.tif` -> GTiff, `.nc` -> netCDF, …), so this is no longer
                GeoTIFF-only. When given, the raster is cloned to that file and the
                no-data swap is streamed tile by tile, so the whole raster is never held
                in RAM — genuinely out-of-core for a block-based format such as GTiff.
                `None` (default) keeps the result in memory.

                The driver must be one the raster can be *built* with, not merely
                copied to: the swap is written into the clone afterwards, and a
                write-by-copy-only driver hands back a read-only handle. PNG and
                JPEG are therefore refused up front with
                :class:`FileFormatNotSupportedError`, naming the extension and the
                driver — rather than reaching GDAL and failing with a
                :class:`ReadOnlyError` that says nothing about the format. Use
                `.tif`, `.nc`, or another updatable format.

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
            DriverNotExistError:
                `path` has no extension, or one the driver catalog does not know.
            FileFormatNotSupportedError:
                `path`'s extension maps to a write-by-copy-only driver (PNG, JPEG).
                Such a driver returns a read-only dataset from `CreateCopy`, so the
                swapped values could not be written back into the clone.

        Warning:
            With `path=None` the method clones the raster in memory to change the
            `no_data_value`; pass `path` for a disk-backed, out-of-core result.
        Examples:
            - Create a Dataset (4 bands, 10 rows, 10 columns) at lon/lat (0, 0):
              ```python
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> dataset = Dataset.create(
              ...     rows=3,
              ...     columns=3,
              ...     bands=1,
              ...     dtype="float32",
              ...     no_data_value=-9,
              ...     geo_ref=GeoReference(cell_size=0.05, top_left_corner=(0, 0), epsg=4326),
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
        new_value = self._normalize_no_data_arg(new_value, "new_value")
        if old_value is not None:
            old_value = self._normalize_no_data_arg(old_value, "old_value")
        # Clone the full header + pixels with GDAL's block-based CreateCopy so
        # every band's colour table, description, scale/offset, RAT and metadata
        # survive exactly (no explicit per-property copy to drift out of sync).
        # A `path` makes the clone a disk-backed GeoTIFF, so with the tiled swap
        # below the whole raster is never held in RAM (out-of-core); `None` keeps
        # it in memory. The old<->new no-data swap is then streamed one tile at a
        # time, so a full band is never materialised as a NumPy array (#969).
        # From the extension, not hardcoded: a `.nc` path silently produced a
        # mislabelled GTiff. NOT `for_copy`, despite the CreateCopy below: this
        # method streams the no-data swap into the clone afterwards, and a
        # copy-only driver hands back a read-only handle -- `.png` passed the
        # resolver and then died with a ReadOnlyError naming nothing about the
        # format. The strict gate refuses it up front, with a message that does.
        driver = resolve_output_driver(path) if path else MEMORY_DRIVER
        target = str(path) if path is not None else ""
        dst = gdal.GetDriverByName(driver).CreateCopy(target, self._ds.raster, 0)
        new_dataset = self._ds.__class__(dst, "write")
        try:
            # _set_no_data_value may coerce new_value to each band's dtype; read it
            # back from the object so the swap below uses the stored values.
            new_dataset._set_no_data_value(new_value)
            new_value = new_dataset.no_data_value
            self._swap_all_bands(new_dataset, new_value, old_value)
        except Exception:
            # A mid-stream failure must not leave a half-written GeoTIFF behind:
            # drop the local handle and let the helper release + delete the partial
            # file before re-raising (best-effort, so it never masks the error).
            if path is not None:
                dst = None
                self._discard_partial_output(new_dataset, target)
            raise
        # Flush the block cache so a disk-backed GeoTIFF has the swapped pixels on
        # disk before it is reopened (a no-op for the in-memory driver).
        new_dataset.raster.FlushCache()

        if inplace:
            self._ds._update_inplace(new_dataset.raster)
            return None
        return new_dataset

    def _swap_no_data_tiled(
        self,
        new_dataset: Dataset,
        band: int,
        band_old_value: Any,
        new_band_value: Any,
    ) -> None:
        """Replace a band's old-no-data cells with the new value, one tile at a time.

        The caller's `_set_no_data_value` fills the destination band with the new
        no-data value, so every tile is read from the *source*, has its old
        no-data cells swapped to the new value, and is written back at its offset
        -- reconstructing the band exactly as the previous whole-band read/swap/
        write did, but never holding more than one tile as a NumPy array. The swap
        is attempted on every tile so an invalid `new_band_value` dtype raises just
        as the whole-band assignment did, even for a band with no matching cells.

        Args:
            new_dataset: The destination Dataset written in place.
            band: Zero-based index of the band to swap.
            band_old_value: The old no-data value to locate (``None`` matches NaN).
            new_band_value: The new no-data value written into the matched cells.

        Raises:
            NoDataValueError: `new_band_value` cannot be stored in the band dtype.
        """
        dst_band = new_dataset.raster.GetRasterBand(band + 1)
        for xoff, yoff, xsize, ysize in self._ds.io._tile_offsets():
            tile = self._ds.read_array(band=band, window=[xoff, yoff, xsize, ysize])
            mask = is_no_data(tile, band_old_value)
            try:
                with np.errstate(invalid="raise"):
                    tile[mask] = new_band_value
            # A dtype mismatch surfaces differently across numpy paths: a None value
            # is not subscriptable (TypeError), a NaN cast into an integer band raises
            # ValueError ("cannot convert float NaN to integer"), and an invalid
            # floating-point cast trips errstate (FloatingPointError). Map all of them
            # to the package-level NoDataValueError.
            except (TypeError, ValueError, FloatingPointError):
                raise NoDataValueError(
                    f"The dtype of the given no_data_value: {new_band_value} differs from the dtype of the "
                    f"band: {gdal_to_numpy_dtype(self._ds.gdal_dtype[band])}"
                )
            dst_band.WriteArray(tile, xoff, yoff)
