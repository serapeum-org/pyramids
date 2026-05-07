"""Utility module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from osgeo import gdal, gdalconst, ogr  # gdal_array,
from osgeo.gdal import Dataset
from pandas import DataFrame

from pyramids import __path__
from pyramids.base._errors import DriverNotExistError, OptionalPackageDoesNotExist

DTYPE_NAMES = [
    None,
    "byte",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "float32",
    "float64",
    "complex-int16",
    "complex-int32",
    "complex-float32",
    "complex-float64",
    "uint64",
    "int64",
    "int8",
    "count",
]

GDAL_DTYPE = [
    gdalconst.GDT_Unknown,
    gdalconst.GDT_Byte,
    gdalconst.GDT_UInt16,
    gdalconst.GDT_Int16,
    gdalconst.GDT_UInt32,
    gdalconst.GDT_Int32,
    gdalconst.GDT_Float32,
    gdalconst.GDT_Float64,
    gdalconst.GDT_CInt16,
    gdalconst.GDT_CInt32,
    gdalconst.GDT_CFloat32,
    gdalconst.GDT_CFloat64,
    gdalconst.GDT_UInt64,
    gdalconst.GDT_Int64,
    gdalconst.GDT_Int8,
    gdalconst.GDT_TypeCount,
]

GDAL_DTYPE_CODE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

OGR_DTYPE = [
    None,
    ogr.OFTInteger,
    ogr.OFTInteger,
    ogr.OFTInteger,
    ogr.OFTInteger64,
    ogr.OFTInteger64,
    ogr.OFTReal,
    ogr.OFTReal,
    None,
    None,
    None,
    None,
    ogr.OFTInteger64,
    ogr.OFTInteger64,
    ogr.OFTInteger,
    None,
]

NUMPY_DTYPE = [
    None,
    np.uint8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.float32,
    np.float64,
    np.complex64,
    np.complex64,
    np.complex64,
    np.complex128,
    np.uint64,
    np.int64,
    np.int8,
    None,
]

DTYPE_CONVERSION_DF = DataFrame(
    columns=["id", "name", "numpy", "gdal", "ogr"],
    data=list(zip(GDAL_DTYPE_CODE, DTYPE_NAMES, NUMPY_DTYPE, GDAL_DTYPE, OGR_DTYPE)),
)

COLOR_INTERPRETATIONS = [
    gdal.GCI_Undefined,  # 0
    gdal.GCI_GrayIndex,  # 1
    gdal.GCI_PaletteIndex,  # 2
    gdal.GCI_RedBand,  # 3
    gdal.GCI_GreenBand,  # 4
    gdal.GCI_BlueBand,  # 5
    gdal.GCI_AlphaBand,  # 6
    gdal.GCI_HueBand,  # 7
    gdal.GCI_SaturationBand,  # 8
    gdal.GCI_LightnessBand,  # 9
    gdal.GCI_CyanBand,  # 10
    gdal.GCI_MagentaBand,  # 11
    gdal.GCI_YellowBand,  # 12
    gdal.GCI_BlackBand,  # 13
    gdal.GCI_YCbCr_YBand,  # 14
    gdal.GCI_YCbCr_CbBand,  # 15
    gdal.GCI_YCbCr_CrBand,  # 16
]

COLOR_NAMES = [
    "undefined",
    "gray_index",
    "palette_index",
    "red",
    "green",
    "blue",
    "alpha",
    "hue",
    "saturation",
    "lightness",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "YCbCr_YBand",
    "YCbCr_CbBand",
    "YCbCr_CrBand",
]

COLOR_TABLE = DataFrame(
    columns=["id", "gdal_constant", "name"],
    data=list(zip(range(len(COLOR_NAMES)), COLOR_INTERPRETATIONS, COLOR_NAMES)),
)
INTERPOLATION_METHODS = {
    "nearest neighbor": gdal.GRA_NearestNeighbour,
    "cubic": gdal.GRA_Cubic,
    "bilinear": gdal.GRA_Bilinear,
}


def color_name_to_gdal_constant(color_name: str) -> int:
    """Convert color name to GDAL constant.

    Args:
        color_name (str): Color name.

    Returns:
        int: GDAL constant corresponding to the color name.
    """
    if color_name not in COLOR_NAMES:
        raise ValueError(
            f"{color_name} is not a valid color name, possible names are: {COLOR_NAMES}"
        )

    gdal_constant = int(
        COLOR_TABLE.loc[COLOR_TABLE["name"] == color_name, "gdal_constant"].values[0]
    )
    return gdal_constant


def gdal_constant_to_color_name(gdal_constant: int) -> str:
    """Convert GDAL constant to color name.

    Args:
        gdal_constant (int): GDAL constant.

    Returns:
        str: Color name corresponding to the GDAL constant.
    """
    if gdal_constant not in COLOR_INTERPRETATIONS:
        raise ValueError(
            f"{gdal_constant} is not a valid gdal constant, possible constants are: {COLOR_INTERPRETATIONS}"
        )
    color_name = COLOR_TABLE.loc[
        COLOR_TABLE["gdal_constant"] == gdal_constant, "name"
    ].values[0]
    return str(color_name)


def numpy_to_gdal_dtype(arr: np.ndarray | np.dtype | str) -> int:
    """Map function between numpy and GDAL data types.

    Args:
        arr (np.ndarray | np.dtype | str): Numpy array or numpy data type.

    Returns:
        int: GDAL data type code.
    """
    if isinstance(arr, np.ndarray):
        np_dtype = arr.dtype
    elif isinstance(arr, np.dtype):
        np_dtype = arr
    elif isinstance(arr, str):
        np_dtype = np.dtype(arr)
    else:
        raise ValueError(
            "The given input is not a numpy array or a numpy data type, please provide a valid input"
        )
    # integer as gdal does not accept the dtype if it is int64
    gdal_type = int(
        DTYPE_CONVERSION_DF.loc[
            DTYPE_CONVERSION_DF["numpy"] == np_dtype, "gdal"
        ].values[0]
    )
    return gdal_type


def ogr_to_numpy_dtype(dtype_code: int):
    """Convert OGR dtype into numpy dtype.

    Args:
        dtype_code (int): OGR data type code
            - ogr.OFTInteger: 0
            - ogr.OFTIntegerList: 1
            - ogr.OFTReal: 2
            - ogr.OFTRealList: 3
            - ogr.OFTString: 4
            - ogr.OFTStringList: 5
            - ogr.OFTWideString: 6
            - ogr.OFTWideStringList: 7
            - ogr.OFTBinary: 8
            - ogr.OFTDate: 9
            - ogr.OFTTime: 10
            - ogr.OFTDateTime: 11
            - ogr.OFTInteger64: 12
            - ogr.OFTInteger64List: 13

    Returns:
        numpy.dtype: Numpy data type corresponding to the OGR code.
    """
    # since there are more than one numpy dtype for the ogr.OFTInteger (0), and the ogr.OFTInteger64 (12),
    # we will return int32 for 0 and int64 for 12.
    result_dtype: type
    if dtype_code == 0:
        result_dtype = np.int32
    elif dtype_code == 12:
        result_dtype = np.int64
    elif dtype_code == 2:
        result_dtype = np.float64
    else:
        matched = DTYPE_CONVERSION_DF.loc[
            DTYPE_CONVERSION_DF["ogr"] == dtype_code, "numpy"
        ]

        if len(matched) == 0:
            raise ValueError(
                f"The given OGR data type is not supported: {dtype_code}, available types are: "
                f"{DTYPE_CONVERSION_DF['ogr'].unique().tolist()}"
            )
        else:
            result_dtype = matched.values[0]

    return result_dtype


def gdal_to_numpy_dtype(dtype: int) -> str:
    """Convert GDAL dtype into numpy dtype.

    Args:
        dtype (int): GDAL data type code.

    Returns:
        str: Name of the corresponding numpy dtype.
    """
    matched_dtypes = DTYPE_CONVERSION_DF.loc[
        DTYPE_CONVERSION_DF["gdal"] == dtype, "numpy"
    ]
    if len(matched_dtypes) == 0:
        raise ValueError(
            f"The given GDAL data type is not supported: {dtype}, available types are: "
            f"{DTYPE_CONVERSION_DF['gdal'].unique().tolist()}"
        )
    result_name = str(matched_dtypes.values[0].__name__)
    return result_name


def gdal_to_ogr_dtype(src: Dataset, band: int = 1):
    """Get the corresponding OGR data type for a given GDAL band.

    Args:
        src (gdal.Dataset): GDAL dataset.
        band (int): Band index (1-based). Default is 1.

    Returns:
        int: OGR data type code corresponding to the band GDAL dtype.
    """
    raster_band = src.GetRasterBand(band)
    gdal_dtype = raster_band.DataType
    return int(
        DTYPE_CONVERSION_DF.loc[
            DTYPE_CONVERSION_DF["gdal"] == gdal_dtype, "ogr"
        ].values[0]
    )


class Catalog:
    """Data Catalog."""

    def __init__(self, raster_driver=True):
        """Initialize the catalog."""
        if raster_driver:
            path = "gdal_drivers.yaml"
        else:
            path = "ogr_drivers.yaml"
        self.catalog = self._get_gdal_catalog(path)

    @staticmethod
    def _get_gdal_catalog(path: str):
        catalog_path = Path(__path__[0]) / f"base/data/{path}"
        with open(catalog_path) as stream:
            gdal_catalog = yaml.safe_load(stream)

        return gdal_catalog

    def get_driver(self, driver: str):
        """Get Driver data from the catalog."""
        return self.catalog.get(driver)

    def get_gdal_name(self, driver: str):
        """Get GDAL name."""
        driver_data = self.get_driver(driver)
        return driver_data.get("GDAL Name")

    def get_driver_name_by_extension(self, extension: str):
        """Get driver by extension.

        Args:
            extension (str): Extension of the file.

        Returns:
            str: Driver name.
        """
        try:
            key = next(
                key
                for key, value in self.catalog.items()
                if value.get("extension") is not None
                and value.get("extension") == extension
            )
        except StopIteration:
            raise DriverNotExistError(
                f"The given extension: {extension} is not associated with any driver in the "
                "driver catalog, if this driver is supported by gdal please open and issue to "
                "asking for youe extension to be added to the catalog"
                "https://github.com/serapeum-org/pyramids/issues/new?assignees=&labels=&template=feature_request.md&title=add%20extension"
            )

        return key

    def get_driver_by_extension(self, extension):
        """Get driver by extension.

        Args:
            extension (str): Extension of the file.

        Returns:
            dict: Driver dictionary.
        """
        driver_name = self.get_driver_name_by_extension(extension)
        return self.get_driver(driver_name)

    def exists(self, driver: str):
        """Check if the driver exist in the catalog."""
        return driver in self.catalog.keys()

    def get_extension(self, driver: str):
        """Get driver extension."""
        driver_data = self.get_driver(driver)
        return driver_data.get("extension")

    def get_driver_name(self, gdal_name) -> str | None:
        """Get driver name."""
        result_key = None
        for key, value in self.catalog.items():
            name = value.get("GDAL Name")
            if gdal_name == name:
                result_key = str(key)
                break
        return result_key


def import_cleopatra(message: str):
    """Import cleopatra."""
    try:
        import cleopatra  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_flox(message: str):
    """Import flox."""
    try:
        import flox  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_zarr(message: str):
    """Import zarr."""
    try:
        import zarr  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_dask_geopandas(message: str):
    """Import dask_geopandas."""
    try:
        import dask_geopandas  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_pyarrow(message: str):
    """Import pyarrow."""
    try:
        import pyarrow  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_dask(message: str):
    """Import dask."""
    try:
        import dask  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_cftime(message: str):
    """Import cftime."""
    try:
        import cftime  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_xarray(message: str):
    """Import xarray."""
    try:
        import xarray  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_kerchunk(message: str):
    """Import kerchunk."""
    try:
        import kerchunk  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_basemap(message: str):
    """Import basemap dependencies (mercantile, xyzservices, Pillow)."""
    try:
        import mercantile  # noqa
        import xyzservices  # noqa
        from PIL import Image  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def ogr_ds_to_gdal_dataset(ogr_ds: ogr.DataSource) -> gdal.Dataset:
    """Convert ogr.DataSource object to a gdal.Dataset.

    Args:
        ogr_ds (ogr.DataSource): OGR data source object.

    Returns:
        gdal.Dataset: An in-memory GDAL dataset converted from the OGR source.
    """
    gdal_ds = gdal.GetDriverByName("Memory").Create("", 0, 0, 0, gdal.GDT_Unknown)

    for i in range(ogr_ds.GetLayerCount()):
        layer = ogr_ds.GetLayerByIndex(i)
        gdal_layer = gdal_ds.CreateLayer(
            layer.GetName(), layer.GetSpatialRef(), layer.GetLayerDefn().GetGeomType()
        )
        for field in layer.schema:
            gdal_layer.CreateField(ogr.FieldDefn(field.name, field.type))
        for feature in layer:
            gdal_feature = ogr.Feature(feature.GetDefnRef())
            gdal_feature.SetGeometry(feature.GetGeometryRef())
            for field in layer.schema:
                field_value = feature.GetField(field.name)
                gdal_feature.SetField(field.name, field_value)
            gdal_layer.CreateFeature(gdal_feature)

    return gdal_ds
