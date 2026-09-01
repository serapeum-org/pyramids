"""Utility module."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any, cast

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


def _first_wins(keys: Sequence[Any], values: Sequence[Any]) -> dict[Any, Any]:
    """Zip `keys` onto `values`, keeping the first pair for a duplicated key.

    Mirrors the ``.values[0]`` semantics of the DataFrame masks these lookup
    tables replace: ``NUMPY_DTYPE`` maps ``np.complex64`` onto three different
    GDAL codes, and the earliest row wins. Pairs with a ``None`` on either side
    are dropped so the "unsupported dtype" guards still raise instead of
    returning a bogus ``None``.

    Args:
        keys (Sequence[Any]): Lookup keys, in table order.
        values (Sequence[Any]): Values positionally paired with `keys`.

    Returns:
        dict[Any, Any]: The first-wins mapping, with ``None``-bearing pairs
        removed.
    """
    mapping: dict[Any, Any] = {}
    for key, value in zip(keys, values):
        if key is None or value is None:
            continue
        if key not in mapping:
            mapping[key] = value
    return mapping


# Precomputed dtype lookups. The conversion helpers below used to run a
# full-column pandas boolean mask over `DTYPE_CONVERSION_DF` on every call --
# roughly 250 us for what is a fixed 16-row table lookup, paid per band and per
# timestep. These dicts are built once at import; `DTYPE_CONVERSION_DF` is kept
# because it is part of the module's public surface and still supplies the
# "available types" listings in the error messages.
_NUMPY_TO_GDAL: dict[Any, Any] = _first_wins(
    [None if dtype is None else np.dtype(dtype) for dtype in NUMPY_DTYPE], GDAL_DTYPE
)
_GDAL_TO_NUMPY: dict[Any, Any] = _first_wins(GDAL_DTYPE, NUMPY_DTYPE)
_GDAL_TO_OGR: dict[Any, Any] = _first_wins(GDAL_DTYPE, OGR_DTYPE)
# No OGR->numpy table: `OGR_DTYPE` holds only OFTInteger (0), OFTReal (2) and
# OFTInteger64 (12), and `ogr_to_numpy_dtype` answers all three directly, so
# such a dict could never be read.

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

# The GDAL colour interpretations that mark a band as an RGB channel
# (`red` / `green` / `blue` -> `COLOR_NAMES[3:6]`). Only these signal that a
# multi-band raster is RGB imagery. Every other interpretation -- `undefined`,
# `palette_index`, `gray_index`, alpha, CMYK, HSL, YCbCr -- is single-channel
# or paletted and must not trigger the RGB plot heuristic (see
# `Dataset._resolve_plot_band`, issue #910). Kept as an allowlist so a new
# non-RGB interpretation can never silently re-enable the false-RGB bug.
RGB_CHANNEL_INTERPS = frozenset(COLOR_NAMES[3:6])

COLOR_TABLE = DataFrame(
    columns=["id", "gdal_constant", "name"],
    data=list(zip(range(len(COLOR_NAMES)), COLOR_INTERPRETATIONS, COLOR_NAMES)),
)
# The historical pyramids default resampling method. Shared as the default
# argument value across the reproject / warp / align / overview APIs and reused
# as the canonical alias key below, so the literal is defined exactly once
# (S1192).
DEFAULT_RESAMPLING = "nearest neighbor"

# Resampling-method name -> GDAL warp/translate constant. Covers every
# ``gdal.GRA_*`` algorithm of the supported GDAL floor; the snake_case names
# match the common ``Resampling`` enum names so migrating users can
# keep their method strings. ``"nearest neighbor"`` is the historical pyramids
# name and stays as an alias of ``"nearest"``. Constants introduced by newer
# GDAL versions are guarded with ``hasattr`` so importing pyramids never fails
# on an older GDAL.
INTERPOLATION_METHODS = {
    DEFAULT_RESAMPLING: gdal.GRA_NearestNeighbour,
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
    :data:`INTERPOLATION_METHODS` (snake_case plus the
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
    return cast(int, INTERPOLATION_METHODS[key])


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


def resolve_cog_predictor(gdal_dtype: int, nbits: int | None = None) -> int:
    """Pick the DEFLATE/ZSTD predictor that suits a GDAL data type and bit width.

    Mirrors GDAL/libtiff semantics: ``PREDICTOR=2`` (horizontal differencing)
    for integer rasters, ``PREDICTOR=3`` (floating-point predictor) for float
    rasters. The numeric form is verified to be accepted by the GDAL COG driver
    and round-trips to the same ``IMAGE_STRUCTURE`` PREDICTOR token across the
    supported GDAL matrix (ARC-8), so no int→string normalisation is needed;
    the string aliases ``STANDARD`` / ``FLOATING_POINT`` are equally accepted
    when a caller passes them explicitly.

    ``PREDICTOR=2`` is only accepted by libtiff for 8/16/32/64-bit samples. A
    band carrying a sub-byte-aligned ``NBITS`` (e.g. ``12`` from a Sentinel-2
    JP2 source) would make the COG driver reject the write, so a non-standard
    ``nbits`` falls back to ``1`` (no predictor). Pass the *effective* output
    width — after any promotion to the next supported width — so a promoted
    ``12 -> 16`` still benefits from the predictor. ``None`` (the default) means
    "the natural width of the dtype", which is always a supported width.

    Args:
        gdal_dtype (int): A GDAL data-type code (e.g. ``gdal.GDT_Float32``).
        nbits (int | None): The effective sample width in bits, or ``None`` for
            the dtype's natural width. ``PREDICTOR=2`` is only chosen for
            integer rasters whose width is ``None`` or one of 8/16/32/64.

    Returns:
        int: ``3`` for floating-point types; for integer types, ``2`` when the
        width is predictor-safe (``None``/8/16/32/64) else ``1`` (no predictor).

    Examples:
        - An integer raster at its natural width uses the horizontal predictor:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._utils import resolve_cog_predictor
            >>> resolve_cog_predictor(gdal.GDT_UInt16)
            2

            ```
        - A float raster uses the floating-point predictor:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._utils import resolve_cog_predictor
            >>> resolve_cog_predictor(gdal.GDT_Float32)
            3

            ```
        - A sub-byte-aligned integer width falls back to no predictor:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._utils import resolve_cog_predictor
            >>> resolve_cog_predictor(gdal.GDT_UInt16, nbits=12)
            1
            >>> resolve_cog_predictor(gdal.GDT_UInt16, nbits=16)
            2

            ```
    """
    if gdal_dtype in INTEGER_GDAL_DTYPES:
        return 2 if nbits in (None, 8, 16, 32, 64) else 1
    return 3


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

    Raises:
        ValueError: If the input is not array/dtype/str-like, or if the numpy
            dtype has no GDAL counterpart.
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
    matched = _NUMPY_TO_GDAL.get(np_dtype)
    if matched is None:
        raise ValueError(
            f"The given numpy data type is not supported: {np_dtype}, available types are: "
            f"{DTYPE_CONVERSION_DF['numpy'].dropna().tolist()}"
        )
    return int(matched)


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

    Raises:
        ValueError: If `dtype_code` is not one of the three OGR types the
            conversion table covers.
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
        # `OGR_DTYPE` contains only these three codes, so anything else is
        # unsupported by definition -- there is no table left to consult.
        raise ValueError(
            f"The given OGR data type is not supported: {dtype_code}, available types are: "
            f"{DTYPE_CONVERSION_DF['ogr'].unique().tolist()}"
        )

    return result_dtype


def gdal_to_numpy_dtype(dtype: int) -> str:
    """Convert GDAL dtype into numpy dtype.

    Args:
        dtype (int): GDAL data type code.

    Returns:
        str: Name of the corresponding numpy dtype.

    Raises:
        ValueError: If `dtype` has no numpy counterpart — including the
            placeholder codes ``GDT_Unknown`` and ``GDT_TypeCount``, whose table
            rows carry no numpy type.
    """
    matched = _GDAL_TO_NUMPY.get(dtype)
    if matched is None:
        raise ValueError(
            f"The given GDAL data type is not supported: {dtype}, available types are: "
            f"{DTYPE_CONVERSION_DF['gdal'].unique().tolist()}"
        )
    result_name = str(matched.__name__)
    return result_name


def gdal_to_ogr_dtype(src: Dataset, band: int = 1):
    """Get the corresponding OGR data type for a given GDAL band.

    Args:
        src (gdal.Dataset): GDAL dataset.
        band (int): Band index (1-based). Default is 1.

    Returns:
        int: OGR data type code corresponding to the band GDAL dtype.

    Raises:
        ValueError: If the band's GDAL dtype has no OGR counterpart (the
            complex types and the ``GDT_Unknown`` / ``GDT_TypeCount``
            placeholders).
    """
    raster_band = src.GetRasterBand(band)
    gdal_dtype = raster_band.DataType
    matched = _GDAL_TO_OGR.get(gdal_dtype)
    if matched is None:
        raise ValueError(
            f"The given GDAL data type has no OGR equivalent: {gdal_dtype}, available types are: "
            f"{DTYPE_CONVERSION_DF['gdal'].unique().tolist()}"
        )
    return int(matched)


@cache
def _build_catalog(raster_driver: bool) -> Catalog:
    """Build (once) the catalog for one driver family.

    Positional-only in practice: `get_catalog` always calls it the same way, so
    the cache cannot be split by call spelling.

    Args:
        raster_driver: `True` for the GDAL raster catalog, `False` for OGR.

    Returns:
        Catalog: The single instance for that family.
    """
    return Catalog(raster_driver=raster_driver)


def get_catalog(raster_driver: bool = True) -> Catalog:
    """Return the process-wide driver catalog, building it on first use.

    `Catalog.__init__` opens and parses a YAML file, which costs ~17 ms. Three
    modules need one, and building three meant parsing the raster table twice.
    Cached rather than assigned at import so nothing pays the cost until a
    driver is actually looked up.

    The result is shared, and `Catalog` exposes no mutators -- but it holds the
    plain dict `yaml.safe_load` returned, so a caller that reaches into
    `.drivers` and edits it changes what every other consumer sees. Treat it as
    read-only.

    The two families are cached separately (`maxsize=2`), so a raster lookup
    never forces a re-parse of the vector table or the reverse.

    Args:
        raster_driver: `True` (default) for the GDAL raster catalog, `False`
            for the OGR vector one.

    Returns:
        Catalog: The shared instance for that driver family.

    Examples:
        - Look a raster extension up through the shared catalog:
            ```python
            >>> from pyramids.base._utils import get_catalog
            >>> catalog = get_catalog()
            >>> catalog.get_driver_name_by_extension("tif")
            'geotiff'
            >>> catalog.get_gdal_name("geotiff")
            'GTiff'

            ```
        - Every caller gets the same object, so the YAML is parsed once:
            ```python
            >>> from pyramids.base._utils import get_catalog
            >>> get_catalog() is get_catalog()
            True

            ```
        - The vector family is a separate catalog with its own entries:
            ```python
            >>> from pyramids.base._utils import get_catalog
            >>> vector = get_catalog(raster_driver=False)
            >>> vector.get_driver_name_by_extension("geojson")
            'geojson'
            >>> vector.get_gdal_name("esri shapefile")
            'ESRI Shapefile'
            >>> vector is get_catalog()
            False

            ```

    See Also:
        - :class:`Catalog`: The catalog itself; construct one directly only when
          an unshared, mutable copy is genuinely needed.
    """
    # Delegated rather than cached here: `lru_cache` keys on the *call form*,
    # so `get_catalog()`, `get_catalog(True)` and `get_catalog(raster_driver=True)`
    # were three separate entries for one catalog -- and with the old maxsize=2
    # a third spelling evicted the first, silently re-parsing the YAML and
    # returning a different object. The "process-wide, shared" guarantee the
    # docstring makes only holds if the key is normalised.
    return _build_catalog(bool(raster_driver))


class Catalog:
    """Data Catalog."""

    def __init__(self, raster_driver=True):
        """Initialize the catalog."""
        if raster_driver:
            path = "gdal_drivers.yaml"
        else:
            path = "ogr_drivers.yaml"
        self.drivers = self._get_gdal_catalog(path)

    @staticmethod
    def _get_gdal_catalog(path: str):
        catalog_path = Path(__path__[0]) / f"base/data/{path}"
        with open(catalog_path) as stream:
            gdal_catalog = yaml.safe_load(stream)

        return gdal_catalog

    def get_driver(self, driver: str):
        """Get Driver data from the catalog.

        Args:
            driver (str): The catalog key (e.g. `"geotiff"`).

        Returns:
            dict | None: The driver's entry, or `None` when the key is unknown.

        Warning:
            The returned dict is the catalog's own, and the catalog is shared
            process-wide by :func:`get_catalog`. Mutating it changes what every
            other consumer sees. Treat it as read-only.
        """
        return self.drivers.get(driver)

    def get_gdal_name(self, driver: str):
        """Get GDAL name."""
        driver_data = self.get_driver(driver)
        return driver_data.get("GDAL Name")

    def get_driver_name_by_extension(self, extension: str):
        """Get driver by extension.

        Matches the driver's canonical `extension` or any of its `aliases` —
        further spellings of the same format (`tiff` for GTiff, `jpg` for
        JPEG). Without them, sibling spellings of one format resolved
        differently: `.tif` worked while `.tiff` raised `DriverNotExistError`,
        and `.jpeg` reported "cannot create" while `.jpg` reported "unknown
        format".

        An entry is matched on its `aliases` even when it declares no canonical
        `extension`, so an alias-only row is reachable.

        Args:
            extension (str): Extension of the file, without the leading dot and already
                lower-cased (`"tif"`, not `".TIF"`). Matched against each entry's canonical
                `extension` first, then against its `aliases` list. Must be a non-empty
                string: an empty value (or `None`) is rejected rather than matched, because
                the catalog holds rows whose `extension` is null and a falsy argument would
                otherwise resolve to whichever of them comes first.

        Returns:
            str: The catalog key for the driver (e.g. `"geotiff"`), not the GDAL short name —
                pass it to :meth:`get_driver` or :meth:`get_gdal_name` to go further.

        Raises:
            DriverNotExistError: `extension` is empty or `None`, or no driver claims it.

        Examples:
            - The canonical extension resolves to its catalog key:
                ```python
                >>> from pyramids.base._utils import Catalog
                >>> catalog = Catalog(raster_driver=True)
                >>> catalog.get_driver_name_by_extension("tif")
                'geotiff'
                >>> catalog.get_gdal_name(catalog.get_driver_name_by_extension("tif"))
                'GTiff'

                ```
            - An alias resolves to the same driver as the canonical spelling:
                ```python
                >>> from pyramids.base._utils import Catalog
                >>> catalog = Catalog(raster_driver=True)
                >>> catalog.get_driver_name_by_extension("tiff")
                'geotiff'
                >>> catalog.get_driver_name_by_extension("jpg") == catalog.get_driver_name_by_extension("jpeg")
                True

                ```
            - An extension no entry claims is refused:
                ```python
                >>> from pyramids.base._errors import DriverNotExistError
                >>> from pyramids.base._utils import Catalog
                >>> catalog = Catalog(raster_driver=True)
                >>> try:
                ...     catalog.get_driver_name_by_extension("xyzzy")
                ... except DriverNotExistError as error:
                ...     print(str(error).split(" is not")[0])
                The given extension: xyzzy

                ```
            - An empty extension is refused instead of matching a null-extension entry:
                ```python
                >>> from pyramids.base._errors import DriverNotExistError
                >>> from pyramids.base._utils import Catalog
                >>> catalog = Catalog(raster_driver=True)
                >>> try:
                ...     catalog.get_driver_name_by_extension("")
                ... except DriverNotExistError as error:
                ...     print(str(error).split(";")[0])
                An empty extension is not associated with any driver

                ```

        See Also:
            - :meth:`get_driver_by_extension`: The same lookup, returning the driver entry.
            - :meth:`get_extension`: The inverse — the canonical extension for a catalog key.
        """
        # Guard the *argument*, not the row. Dropping the old per-row
        # `extension is not None` check is what lets a row carrying only
        # `aliases` be found at all, but it would also let a `None` argument
        # match the null-extension `memory` row by `None == None` -- a lookup
        # for nothing resolving to the in-memory driver.
        if not extension:
            raise DriverNotExistError(
                "An empty extension is not associated with any driver; pass the "
                "file extension to look up."
            )
        try:
            key = next(
                key
                for key, value in self.drivers.items()
                # No `extension is not None` guard on the row: it skipped the
                # entry before the alias test was reached, so a row carrying
                # only `aliases` was unreachable by any of them.
                if value.get("extension") == extension
                or extension in (value.get("aliases") or ())
            )
        except StopIteration:
            raise DriverNotExistError(
                f"The given extension: {extension} is not associated with any driver in the "
                "driver catalog, if this driver is supported by gdal please open an issue "
                "asking for your extension to be added to the catalog: "
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
        return driver in self.drivers.keys()

    def get_extension(self, driver: str):
        """Get the driver's canonical file extension.

        Only the canonical spelling is returned — the one to build a filename with. The
        further spellings a driver also answers to live in its `aliases` list and are
        reachable through :meth:`get_driver_name_by_extension`, not here: `"geotiff"` returns
        `"tif"` even though `"tiff"` resolves back to it.

        Args:
            driver (str): Catalog key for the driver (e.g. `"geotiff"`), as returned by
                :meth:`get_driver_name_by_extension` or :meth:`get_driver_name`.

        Returns:
            str | None: The extension without a leading dot, or `None` for an entry that has
                no file extension at all — the in-memory `"memory"` driver, or a driver whose
                row simply omits the key.

        Raises:
            AttributeError: `driver` is not a key in the catalog, so there is no entry to read.

        Examples:
            - The canonical spelling, not the alias:
                ```python
                >>> from pyramids.base._utils import Catalog
                >>> catalog = Catalog(raster_driver=True)
                >>> catalog.get_extension("geotiff")
                'tif'
                >>> f"out.{catalog.get_extension('netcdf')}"
                'out.nc'

                ```
            - The in-memory driver writes nothing, so it has no extension:
                ```python
                >>> from pyramids.base._utils import Catalog
                >>> catalog = Catalog(raster_driver=True)
                >>> print(catalog.get_extension("memory"))
                None

                ```

        See Also:
            - :meth:`get_driver_name_by_extension`: The inverse lookup, which also matches
              aliases.
        """
        driver_data = self.get_driver(driver)
        return driver_data.get("extension")

    def get_driver_name(self, gdal_name) -> str | None:
        """Get the catalog key for a GDAL short name.

        The inverse of :meth:`get_gdal_name`: it walks the catalog looking for the entry whose
        `GDAL Name` matches, and answers with that entry's key. The key is what the rest of the
        catalog API takes, so this is the bridge from a name GDAL handed back (e.g. from
        `dataset.GetDriver().ShortName`) into the catalog's own vocabulary.

        Args:
            gdal_name: GDAL driver short name to look up, matched exactly and
                case-sensitively (`"GTiff"`, not `"gtiff"`).

        Returns:
            str | None: The catalog key (e.g. `"geotiff"`), or `None` when no entry carries
                that GDAL name.

        Examples:
            - Round-trip a GDAL name through the catalog:
                ```python
                >>> from pyramids.base._utils import Catalog
                >>> catalog = Catalog(raster_driver=True)
                >>> catalog.get_driver_name("GTiff")
                'geotiff'
                >>> catalog.get_gdal_name(catalog.get_driver_name("GTiff"))
                'GTiff'

                ```
            - An unknown or mis-cased name yields `None` rather than raising:
                ```python
                >>> from pyramids.base._utils import Catalog
                >>> catalog = Catalog(raster_driver=True)
                >>> print(catalog.get_driver_name("NotADriver"))
                None
                >>> print(catalog.get_driver_name("gtiff"))
                None

                ```

        See Also:
            - :meth:`get_gdal_name`: The forward direction, key -> GDAL short name.
        """
        result_key = None
        for key, value in self.drivers.items():
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


def require_optional(module_name: str, message: str, *, return_module: bool = False):
    """Import an optional dependency, or raise the extra's install hint.

    One implementation behind every `import_<package>` guard in this module.
    Each of those had its own copy of the same `try: import X except
    ImportError: raise OptionalPackageDoesNotExist(message)` body; they are now
    one-line delegations to this helper, keeping their names so call sites (and
    the tests that monkeypatch them per module) are unaffected.

    The import goes through the builtin `__import__` rather than
    `importlib.import_module`, so it stays interceptable by call sites and tests
    that patch `builtins.__import__` — which several guard tests do to simulate
    a missing package.

    One nuance for dotted names: `import_basemap` previously spelled
    `from cleopatra import tiles`, which calls
    `__import__("cleopatra", ..., fromlist=("tiles",))`. Here it becomes
    `__import__("cleopatra.basemap.tiles")`, so the parent package resolves internally
    rather than through `builtins.__import__`. A patch keyed on
    `name == "cleopatra"` no longer intercepts it; key on the dotted name
    instead. No in-tree test depends on this.

    Args:
        module_name: Dotted module path to import, e.g. ``"zarr"`` or
            ``"cleopatra.basemap.tiles"``.
        message: The install hint raised when the import fails. Compose it with
            :func:`lazy_extra_hint` for the ``[lazy]`` extra.
        return_module: When `True` the imported module object is returned so the
            caller can use it without a bare inline import of its own. The
            guard-only callers leave it `False` and get `None`.

    Returns:
        The imported module when `return_module` is `True`, otherwise `None`.

    Raises:
        OptionalPackageDoesNotExist: When the module cannot be imported.

    Examples:
        - An installed dependency resolves, and the guard-only form returns
          nothing:
            ```python
            >>> from pyramids.base._utils import require_optional
            >>> require_optional("numpy", "install the [x] extra") is None
            True

            ```
        - Ask for the module object back when the caller needs to use it. A
          dotted name resolves to the submodule itself, not its top-level
          package:
            ```python
            >>> from pyramids.base._utils import require_optional
            >>> mod = require_optional(
            ...     "numpy.linalg", "install the [x] extra", return_module=True
            ... )
            >>> mod.__name__
            'numpy.linalg'

            ```
        - A missing package raises the supplied hint verbatim:
            ```python
            >>> from pyramids.base._utils import require_optional
            >>> from pyramids.base._errors import OptionalPackageDoesNotExist
            >>> try:
            ...     require_optional("not_a_real_package", "install the [x] extra")
            ... except OptionalPackageDoesNotExist as exc:
            ...     print(exc)
            install the [x] extra

            ```
    """
    try:
        __import__(module_name)
    except ImportError as exc:
        raise OptionalPackageDoesNotExist(message) from exc
    module = None
    if return_module:
        # `__import__` returns the top-level package for a dotted name, so read the
        # requested module back out of sys.modules instead of using its return value.
        # A module that removed itself from sys.modules during import would give a
        # bare KeyError here; surface the branded error instead.
        try:
            module = sys.modules[module_name]
        except KeyError as exc:
            raise OptionalPackageDoesNotExist(message) from exc
    return module


def import_cleopatra(message: str):
    """Import cleopatra."""
    return require_optional("cleopatra", message)


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
            >>> _saved = sys.modules.get("cleopatra")
            >>> sys.modules["cleopatra"] = types.ModuleType("cleopatra")  # pretend installed
            >>> require_cleopatra() is None
            True
            >>> if _saved is None:  # restore sys.modules so later doctests still see the real cleopatra
            ...     del sys.modules["cleopatra"]
            ... else:
            ...     sys.modules["cleopatra"] = _saved

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


def lazy_extra_hint(prefix: str) -> str:
    """Compose an install hint for the optional ``[lazy]`` extra.

    The shared PyPI / conda-forge install commands for the ``[lazy]`` extra
    (dask / zarr / fsspec) are defined once here so the zarr / dask call
    sites don't each copy them; only the lead sentence varies.

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
    return require_optional("zarr", message)


def import_dask_geopandas(message: str):
    """Import dask_geopandas."""
    return require_optional("dask_geopandas", message)


def import_pyarrow(message: str):
    """Import pyarrow."""
    return require_optional("pyarrow", message)


def import_pystac_client(message: str):
    """Import pystac_client."""
    return require_optional("pystac_client", message)


def import_stac_asset(message: str):
    """Import stac_asset (ships via the optional [stac] extra)."""
    return require_optional("stac_asset", message)


def import_dask(message: str):
    """Import and return the ``dask`` module, or raise the ``[lazy]`` extra hint.

    Returned so callers can use ``dask`` without a bare inline ``import`` of
    their own (dask is optional, so it cannot be a top-level import). Callers
    that only need the guard may ignore the return value.
    """
    return require_optional("dask", message, return_module=True)


def import_kerchunk(message: str):
    """Import kerchunk."""
    return require_optional("kerchunk", message)


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
    return require_optional("h5py", message, return_module=True)


def import_basemap(message: str):
    """Import the web-tile basemap backend (``cleopatra.basemap.tiles``, the ``[tiles]`` extra)."""
    return require_optional("cleopatra.basemap.tiles", message)


def ogr_ds_to_gdal_dataset(ogr_ds: ogr.DataSource) -> gdal.Dataset:
    """Convert ogr.DataSource object to a gdal.Dataset.

    Args:
        ogr_ds (ogr.DataSource): OGR data source object.

    Returns:
        gdal.Dataset: An in-memory GDAL dataset converted from the OGR source.
    """
    gdal_ds = gdal.GetDriverByName("MEM").Create("", 0, 0, 0, gdal.GDT_Unknown)

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


def apply_unpack(
    arr: Any,
    scale: float | np.ndarray | None,
    offset: float | np.ndarray | None,
) -> Any:
    """Apply a scale/offset unpacking transform to a lazy or eager array.

    Computes ``arr * scale + offset`` as `float64`, the single shared primitive
    behind both the NetCDF CF `scale_factor`/`add_offset` path and the raster
    :meth:`~pyramids.dataset.engines.IO.read_array` ``scaled=True`` path. When
    both ``scale`` and ``offset`` are `None` the array is returned unchanged (no
    float promotion), so an unset band is a genuine no-op. ``scale``/``offset``
    may be scalars or a broadcastable `numpy` array (e.g. a per-band
    ``(bands, 1, 1)`` factor); a `dask` array input keeps the arithmetic lazy.

    Args:
        arr: The raw array (dask or numpy, possibly a masked array).
        scale: Multiplicative factor, or `None` to skip scaling.
        offset: Additive offset, or `None` to skip offsetting.

    Returns:
        The (possibly transformed) array, cast to `float64` when a
        transformation was applied.

    Examples:
        - A band with neither scale nor offset is returned unchanged:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._utils import apply_unpack
            >>> apply_unpack(np.array([0, 1, 2]), None, None)
            array([0, 1, 2])

            ```
        - Scale and offset are applied as float64:
            ```python
            >>> apply_unpack(np.array([0, 1, 2]), 0.1, 5.0)
            array([5. , 5.1, 5.2])

            ```
    """
    if scale is None and offset is None:
        result = arr
    else:
        result = arr.astype(np.float64)
        if scale is not None:
            result = result * scale
        if offset is not None:
            result = result + offset
    return result
