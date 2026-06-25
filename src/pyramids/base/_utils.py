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
# Human-readable name for GDAL's "no colour interpretation set" sentinel
# (`gdal.GCI_Undefined` -> `COLOR_NAMES[0]`). Use this constant instead
# of the bare string literal so a name change can't silently break the
# "is this band an RGB channel?" checks (e.g. `Dataset._resolve_plot_band`).
UNDEFINED_COLOR_INTERP = COLOR_NAMES[0]

COLOR_TABLE = DataFrame(
    columns=["id", "gdal_constant", "name"],
    data=list(zip(range(len(COLOR_NAMES)), COLOR_INTERPRETATIONS, COLOR_NAMES)),
)
# Resampling-method name -> GDAL warp/translate constant. Covers every
# ``gdal.GRA_*`` algorithm of the supported GDAL floor; the snake_case names
# match rasterio's ``Resampling`` enum so users migrating from rasterio can
# keep their method strings. ``"nearest neighbor"`` is the historical pyramids
# name and stays as an alias of ``"nearest"``. Constants introduced by newer
# GDAL versions are guarded with ``hasattr`` so importing pyramids never fails
# on an older GDAL.
INTERPOLATION_METHODS = {
    "nearest neighbor": gdal.GRA_NearestNeighbour,
    "nearest": gdal.GRA_NearestNeighbour,
    "bilinear": gdal.GRA_Bilinear,
    "cubic": gdal.GRA_Cubic,
    "cubic_spline": gdal.GRA_CubicSpline,
    "lanczos": gdal.GRA_Lanczos,
    "average": gdal.GRA_Average,
    "mode": gdal.GRA_Mode,
    "max": gdal.GRA_Max,
    "min": gdal.GRA_Min,
    "med": gdal.GRA_Med,
    "q1": gdal.GRA_Q1,
    "q3": gdal.GRA_Q3,
    **({"sum": gdal.GRA_Sum} if hasattr(gdal, "GRA_Sum") else {}),
    **({"rms": gdal.GRA_RMS} if hasattr(gdal, "GRA_RMS") else {}),
}

# Methods that exist only on newer GDAL; used to give a version-aware error when
# they are requested on a build that lacks them, instead of a generic "does not
# exist" that lists a method set differing from the documented one.
_VERSION_GATED_METHODS = {
    "sum": ("GRA_Sum", "3.1"),
    "rms": ("GRA_RMS", "3.3"),
}


def resolve_resampling(method: str) -> int:
    """Resolve a resampling-method name to its GDAL ``GRA_*`` constant.

    Normalises case and surrounding whitespace, so ``"Lanczos"`` and
    ``" average "`` are accepted. The valid names are the keys of
    :data:`INTERPOLATION_METHODS` (rasterio-style snake_case plus the
    historical ``"nearest neighbor"`` alias).

    Args:
        method: Resampling method name, case-insensitive (e.g. ``"nearest"``,
            ``"bilinear"``, ``"cubic"``, ``"average"``, ``"lanczos"``).

    Returns:
        int: The matching ``gdal.GRA_*`` constant.

    Raises:
        TypeError: ``method`` is not a string.
        ValueError: ``method`` does not name a supported algorithm; the
            message lists the valid names.

    Examples:
        - Names are case- and whitespace-insensitive:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._utils import resolve_resampling
            >>> resolve_resampling(" Bilinear ") == gdal.GRA_Bilinear
            True

            ```
        - The historical pyramids name still resolves:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._utils import resolve_resampling
            >>> resolve_resampling("nearest neighbor") == gdal.GRA_NearestNeighbour
            True

            ```
        - Unknown names are rejected with the valid set in the message:
            ```python
            >>> from pyramids.base._utils import resolve_resampling
            >>> try:
            ...     resolve_resampling("sinc")
            ... except ValueError as exc:
            ...     print("does not exist" in str(exc))
            True

            ```
    """
    if not isinstance(method, str):
        raise TypeError(
            f"resampling method must be a string, got {type(method).__name__}."
        )
    key = method.lower().strip()
    if key not in INTERPOLATION_METHODS:
        if key in _VERSION_GATED_METHODS:
            attr, min_ver = _VERSION_GATED_METHODS[key]
            raise ValueError(
                f"resampling method {method!r} requires GDAL >= {min_ver} "
                f"(gdal.{attr} is unavailable in the installed GDAL "
                f"{gdal.__version__})."
            )
        raise ValueError(
            f"The given interpolation method: {method!r} does not exist, "
            f"existing methods are {sorted(INTERPOLATION_METHODS)}"
        )
    return INTERPOLATION_METHODS[key]


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


INTEGER_GDAL_DTYPES: frozenset[int] = frozenset(
    {
        gdalconst.GDT_Byte,
        gdalconst.GDT_Int8,
        gdalconst.GDT_UInt16,
        gdalconst.GDT_Int16,
        gdalconst.GDT_UInt32,
        gdalconst.GDT_Int32,
        gdalconst.GDT_UInt64,
        gdalconst.GDT_Int64,
    }
)
"""GDAL data-type codes treated as integer (categorical-capable).

Single source of truth for "is this raster integer?" decisions across the COG
write path — predictor selection (:func:`resolve_cog_predictor`), the default
overview-resampling policy (:func:`default_cog_overview_resampling`), and the
categorical-resampling guardrail in :mod:`pyramids.dataset.engines.cog`.
"""


def is_integer_gdal_dtype(gdal_dtype: int) -> bool:
    """Return ``True`` when ``gdal_dtype`` is one of the integer GDAL types.

    Args:
        gdal_dtype (int): A GDAL data-type code (e.g. ``gdal.GDT_Int16``).

    Returns:
        bool: ``True`` for integer types, ``False`` for floating-point/complex.
    """
    return gdal_dtype in INTEGER_GDAL_DTYPES


def resolve_cog_predictor(gdal_dtype: int) -> int:
    """Pick the DEFLATE/ZSTD predictor that suits a GDAL data type.

    Mirrors GDAL/libtiff semantics: ``PREDICTOR=2`` (horizontal differencing)
    for integer rasters, ``PREDICTOR=3`` (floating-point predictor) for float
    rasters. The numeric form is verified to be accepted by the GDAL COG driver
    and round-trips to the same ``IMAGE_STRUCTURE`` PREDICTOR token across the
    supported GDAL matrix (ARC-8), so no int→string normalisation is needed;
    the string aliases ``STANDARD`` / ``FLOATING_POINT`` are equally accepted
    when a caller passes them explicitly.

    Args:
        gdal_dtype (int): A GDAL data-type code (e.g. ``gdal.GDT_Float32``).

    Returns:
        int: ``2`` for integer types, ``3`` for floating-point types.
    """
    return 2 if gdal_dtype in INTEGER_GDAL_DTYPES else 3


def default_cog_overview_resampling(gdal_dtype: int, has_color_table: bool) -> str:
    """Pick a category-safe default overview resampler for a raster.

    Categorical rasters (integer dtype or a colour table) must not be averaged
    when building overviews — averaging invents values that never existed
    (e.g. a land-cover class of ``3.5``). Continuous rasters benefit from
    ``average``.

    Args:
        gdal_dtype (int): A GDAL data-type code for the first band.
        has_color_table (bool): Whether the first band carries a colour table.

    Returns:
        str: ``"mode"`` for categorical sources, ``"average"`` otherwise.
    """
    categorical = has_color_table or gdal_dtype in INTEGER_GDAL_DTYPES
    return "mode" if categorical else "average"


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


_DEFAULT_CLEOPATRA_MSG = (
    "The current operation uses the cleopatra package. Install with one of:\n"
    "  - PyPI:        pip install 'pyramids-gis[viz]'\n"
    "  - conda-forge: conda install -c conda-forge pyramids-viz\n"
    "  - or directly: https://github.com/serapeum-org/cleopatra"
)


def import_cleopatra(message: str):
    """Import cleopatra."""
    try:
        import cleopatra  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def require_cleopatra(msg: str | None = None) -> None:
    """Single guard for the optional cleopatra dependency.

    Consolidates the scattered `import_cleopatra(<bespoke message>)`
    calls that used to live next to every plot / colour helper. The
    default error message points at the `[viz]` install extra; callers
    that want a domain-specific hint pass `msg` and override it. The
    helper returns `None` when cleopatra is importable and raises
    :class:`OptionalPackageDoesNotExist` otherwise — no side effects
    beyond the import check.

    Args:
        msg: Override for the default error message. `None` uses the
            shared default that points at the `[viz]`` install extra.

    Raises:
        OptionalPackageDoesNotExist: When cleopatra is not importable.

    Examples:
        - When cleopatra is installed the call returns silently and
          the rest of the plotting facade proceeds. The doctest
          monkeypatches the import to a stub so the example works
          even on environments without the ``[viz]`` extra:

            ```python
            >>> import sys, types
            >>> from pyramids.base._utils import require_cleopatra
            >>> stub = types.ModuleType("cleopatra")
            >>> sys.modules.setdefault("cleopatra", stub) is not None
            True
            >>> require_cleopatra() is None
            True

            ```

        - When cleopatra is missing the helper raises
          :class:`OptionalPackageDoesNotExist`. The doctest hides the
          existing import then forces the failure path with a custom
          message and restores the cached module afterwards:

            ```python
            >>> import sys
            >>> from pyramids.base._utils import require_cleopatra
            >>> from pyramids.base._errors import OptionalPackageDoesNotExist
            >>> saved = sys.modules.pop("cleopatra", None)
            >>> sys.modules["cleopatra"] = None  # block the import
            >>> try:
            ...     require_cleopatra("custom hint")
            ... except OptionalPackageDoesNotExist as exc:
            ...     print(str(exc))
            ... finally:
            ...     sys.modules.pop("cleopatra", None)
            ...     if saved is not None:
            ...         sys.modules["cleopatra"] = saved
            custom hint

            ```
    """
    effective = msg if msg is not None else _DEFAULT_CLEOPATRA_MSG
    import_cleopatra(effective)


def import_flox(message: str):
    """Import flox."""
    try:
        import flox  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def lazy_extra_hint(prefix: str) -> str:
    """Compose an install hint for the optional ``[lazy]`` extra.

    The shared PyPI / conda-forge install commands for the ``[lazy]`` extra
    (dask / zarr / fsspec / flox) are defined once here so the zarr / dask /
    flox call sites don't each copy them; only the lead sentence varies.

    Args:
        prefix: The domain-specific lead sentence, ending in a period (e.g.
            ``"Zarr IO requires the optional 'zarr' dependency."``). It is
            placed verbatim at the start of the returned message.

    Returns:
        A single string: ``prefix`` followed by ``"Install with one of:"`` and
        two indented bullet lines giving the PyPI and conda-forge commands.

    Examples:
        - The lead sentence is preserved and a header line follows:
            ```python
            >>> lazy_extra_hint("Op needs the optional 'dask' dependency.").splitlines()[0]
            "Op needs the optional 'dask' dependency. Install with one of:"

            ```
        - Both install commands are present in the body:
            ```python
            >>> hint = lazy_extra_hint("X requires the optional 'zarr' dependency.")
            >>> "pip install 'pyramids-gis[lazy]'" in hint
            True
            >>> "conda install -c conda-forge pyramids-lazy" in hint
            True

            ```
    """
    return (
        f"{prefix} Install with one of:\n"
        "  - PyPI:        pip install 'pyramids-gis[lazy]'\n"
        "  - conda-forge: conda install -c conda-forge pyramids-lazy"
    )


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


def import_pystac_client(message: str):
    """Import pystac_client."""
    try:
        import pystac_client  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_stac_asset(message: str):
    """Import stac_asset (ships via the optional [stac] extra)."""
    try:
        import stac_asset  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_dask(message: str):
    """Import dask."""
    try:
        import dask  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_kerchunk(message: str):
    """Import kerchunk."""
    try:
        import kerchunk  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_h5py(message: str):
    """Import and return :mod:`h5py`.

    Unlike the guard-only helpers above, this returns the imported module because
    callers need the live ``h5py`` handle to walk an HDF5 container.

    Args:
        message: The install hint raised when h5py is missing (compose it with
            :func:`lazy_extra_hint`).

    Returns:
        The imported ``h5py`` module.

    Raises:
        OptionalPackageDoesNotExist: When h5py is not installed.
    """
    try:
        import h5py
    except ImportError:
        raise OptionalPackageDoesNotExist(message)
    return h5py


def import_basemap(message: str):
    """Import the web-tile basemap backend (``cleopatra.tiles``, the ``[tiles]`` extra)."""
    try:
        from cleopatra import tiles  # noqa
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
