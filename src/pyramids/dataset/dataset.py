"""
Dataset module.

raster contains python functions to handle raster data align them together based on a source raster, perform any
algebraic operation on cell's values.
"""

from __future__ import annotations

import logging
import warnings
import weakref
from collections.abc import Callable, Sequence
from dataclasses import replace
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack, cast

import geopandas as gpd
import numpy as np
from osgeo import gdal

from pyramids import _io
from pyramids.base._errors import AlignmentError, ContainerRasterWarning, CRSError
from pyramids.base._utils import (
    DTYPE_CONVERSION_DF,
    RGB_CHANNEL_INTERPS,
    numpy_to_gdal_dtype,
)
from pyramids.base.crs import (
    PROJECTED_AXIS_UNITS,
    VERTICAL_AXIS_NAMES,
    cf_geographic_wkt,
    crs_spec,
    epsg_of_crs,
    sr_from_epsg,
    sr_from_user_input,
    within_lonlat_range,
)
from pyramids.base.georeference import GeoReference
from pyramids.base.remote import cloud_config_from_env, redact_credentials
from pyramids.dataset._driver import (
    MEMORY_DRIVER,
    copy_yields_writable,
    resolve_output_driver,
)
from pyramids.dataset._ogc_coverages import from_ogc_coverages as _from_ogc_coverages
from pyramids.dataset._plot_helpers import nonnull_group_kwargs
from pyramids.dataset._wcs import from_wcs as _from_wcs
from pyramids.dataset._wms import from_wms as _from_wms
from pyramids.dataset._wms import from_wmts as _from_wmts
from pyramids.dataset.abstract_dataset import (
    DEFAULT_NO_DATA_VALUE,
    RasterBase,
)
from pyramids.dataset.engines import (
    COG,
    IO,
    Analysis,
    Bands,
    Cell,
    Georef,
    Spatial,
    Vectorize,
)
from pyramids.dataset.ops._focal import (
    aspect,
    focal_apply,
    focal_mean,
    focal_std,
    hillshade,
    slope,
)
from pyramids.dataset.ops._zarr import (
    read_dataset_from_zarr,
    write_dataset_to_zarr,
)
from pyramids.dataset.ops._zonal import zonal_stats as _zonal_stats
from pyramids.dataset.ops.interpolate import grid_points
from pyramids.dataset.ops.units import convert_array
from pyramids.dataset.ops.vectorize import rasterize_features
from pyramids.feature import FeatureCollection, create_polygon

# tuple of collaborator attribute names. Used by
# `Dataset.__init__` to wire the eight collaborators and by
# `_update_inplace` to re-bind their `_ds` back-references after
# `__dict__.update` (see audit §3.3).
_COLLABORATOR_ATTRS = (
    "io",
    "spatial",
    "bands",
    "analysis",
    "cell",
    "vectorize",
    "cog",
    "georef",
)

# Registry of third-party accessors registered via `register_dataset_accessor`
# (name -> accessor class), plus the set of names a registration must not shadow.
# `_RESERVED_ACCESSOR_NAMES` is seeded from the built-in engines. Engines are
# *instance* attributes (set in `__init__`), so they are invisible to
# `hasattr(Dataset, name)` — the reserved set is how a name clash with them is caught.
# The NetCDF-specific engine names are seeded here too so they are reserved even
# before `pyramids.netcdf` is imported (that module's circular-import carveout means
# importing `pyramids.dataset` alone does not pull it in); `pyramids.netcdf` re-adds
# them on import (idempotent), which also covers any names added there in future.
_ACCESSOR_REGISTRY: dict[str, type] = {}
_RESERVED_ACCESSOR_NAMES: set[str] = set(_COLLABORATOR_ATTRS)
_RESERVED_ACCESSOR_NAMES.update({"interop", "varops", "selection"})


class _CachedAccessor:
    """Descriptor that lazily builds and caches a registered Dataset accessor.

    Mirrors the pandas/xarray ``CachedAccessor``: a non-data descriptor (``__get__``
    only) so that once the accessor is cached in the instance ``__dict__`` every
    later lookup is a plain dict hit that never re-enters the descriptor. The
    accessor receives a :func:`weakref.proxy` of the owning Dataset (matching the
    engine layer's weak back-reference discipline) so the Dataset ↔ accessor graph
    stays acyclic and GDAL handles are released promptly.
    """

    def __init__(self, name: str, accessor_cls: type) -> None:
        self._name = name
        self._accessor_cls = accessor_cls

    def __get__(self, obj: Any, cls: type | None = None) -> Any:
        if obj is None:
            # Class access (e.g. `hasattr(Dataset, name)`) yields the accessor class.
            result = self._accessor_cls
        else:
            result = self._accessor_cls(weakref.proxy(obj))
            # Cache in the instance dict; being a non-data descriptor, the cached
            # value shadows this descriptor on every subsequent access.
            object.__setattr__(obj, self._name, result)
        return result


def _invalidate_cached_accessors(ds: Any) -> None:
    """Drop any cached registered accessors so they rebuild against the live raster.

    Called from ``_update_inplace`` (on both `Dataset` and `NetCDF`) after the
    ``__dict__.update`` swap: the merge would otherwise keep an accessor built
    against the *old* raster (the same stale-cache hazard handled for
    ``_cf_crs_cache``).
    """
    for name in _ACCESSOR_REGISTRY:
        ds.__dict__.pop(name, None)


def _accessor_name_conflict(name: str) -> bool:
    """Whether ``name`` would shadow an existing Dataset (or subclass) attribute.

    Rejects a name that is a reserved engine name (engines are *instance*
    attributes set in ``__init__``, so they are invisible to ``hasattr``), or that
    already exists as a class-level attribute/method/property on `Dataset` **or any
    of its imported subclasses**. The subclass walk is what catches names defined
    only on `NetCDF` / `Variable` / `Container` (for example ``variable_names`` or
    ``get_variable``) once ``pyramids.netcdf`` has been imported — checking only
    `Dataset` would let such a name register and then shadow-split across the class
    hierarchy.
    """
    conflict = name in _RESERVED_ACCESSOR_NAMES
    if not conflict:
        # Walk Dataset's subclass DAG (Dataset -> NetCDF -> Variable/Container, plus any
        # third-party subclass). `__subclasses__()` only yields more-derived classes, so
        # the walk always terminates without needing a visited set.
        classes = [Dataset]
        while classes:
            cls = classes.pop()
            if hasattr(cls, name):
                conflict = True
                break
            classes.extend(cls.__subclasses__())
    return conflict


def register_dataset_accessor(name: str) -> Callable[[type], type]:
    """Register a custom accessor on `Dataset` (and, by inheritance, `NetCDF`).

    An xarray/pandas-style extension hook: a third-party package attaches a custom
    namespace to every `Dataset` without subclassing or editing the built-in engine
    set. The decorated class is instantiated lazily on first access as
    ``accessor_cls(ds)`` — where ``ds`` is a transparent weak proxy of the Dataset —
    and cached on the instance, so repeated access returns the same object. Because
    registration is on the base `Dataset`, the accessor is available on `NetCDF`
    and its `Variable`/`Container` subclasses too.

    Your ``__init__(self, ds)`` receives a weak proxy: read through it (for example
    ``self._ds.read_array()``); do not persist a strong reference to it.

    Args:
        name: The attribute name to expose on `Dataset`. It must not shadow an
            existing attribute, method, property, or engine name.

    Returns:
        A class decorator that registers ``accessor_cls`` and returns it unchanged.

    Raises:
        ValueError: ``name`` shadows an existing Dataset attribute, method, or
            engine (a name already registered is instead overwritten with a
            `UserWarning`, matching pandas/xarray).

    Examples:
        - Register a custom accessor and use it on any Dataset:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference, register_dataset_accessor
            >>> @register_dataset_accessor("summary")
            ... class Summary:
            ...     def __init__(self, ds):
            ...         self._ds = ds
            ...     def describe(self):
            ...         return f"{self._ds.band_count}-band EPSG:{self._ds.epsg}"
            >>> ds = Dataset.from_array(
            ...     np.zeros((2, 3)),
            ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
            ... )
            >>> ds.summary.describe()
            '1-band EPSG:4326'
            >>> ds.summary is ds.summary  # built once, then cached per Dataset
            True
            >>> from pyramids.dataset.dataset import _ACCESSOR_REGISTRY
            >>> delattr(Dataset, "summary"); _ = _ACCESSOR_REGISTRY.pop("summary")

            ```
    """

    def decorator(accessor_cls: type) -> type:
        if name in _ACCESSOR_REGISTRY:
            warnings.warn(
                f"registration of accessor {accessor_cls!r} under name {name!r} is "
                f"overriding a preexisting accessor with the same name.",
                UserWarning,
                stacklevel=2,
            )
        elif _accessor_name_conflict(name):
            raise ValueError(
                f"cannot register accessor {name!r}: it shadows an existing Dataset "
                f"(or NetCDF) attribute, method, or engine."
            )
        setattr(Dataset, name, _CachedAccessor(name, accessor_cls))
        _ACCESSOR_REGISTRY[name] = accessor_cls
        return accessor_cls

    return decorator


# Sentinel for `Dataset.from_band_files(no_data_value=...)` so the helper can
# tell "caller didn't pass one — inherit from the source rasters" apart from
# "caller explicitly passed `None`" (which means "stamp no no-data sentinel").
_INHERIT_NO_DATA = object()

# Default CRS for the ``bbox`` of the web-service readers (from_wcs / from_wms /
# from_wmts): lon/lat WGS 84.
_DEFAULT_CRS = "EPSG:4326"

# Default GTiff creation options for out-of-core allocation
# (`create_empty` / `empty_like`). TILED keeps windowed writes block-aligned
# so a `write_array(window=)` does not amplify into a full-row rewrite;
# SPARSE_OK lets never-written blocks cost no disk (they read back as the
# band no-data value, not 0); BIGTIFF is mandatory past the 4 GB
# classic-TIFF ceiling (a 2.7 B-cell float32 raster is ~10 GB) and must be
# set at creation, not switched mid-write. DEFLATE matches the COG-writer
# convention. Callers can override the whole list (e.g. to align
# BLOCKXSIZE/BLOCKYSIZE to a different tile size).
OUT_OF_CORE_CREATION_OPTIONS = [
    "TILED=YES",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
    "SPARSE_OK=TRUE",
    "BIGTIFF=YES",
    "COMPRESS=DEFLATE",
]


class NoDataSentinelWarning(UserWarning):
    """Warns that a disk-backed sparse raster was allocated with no no-data sentinel.

    Emitted by :meth:`Dataset.create_empty` / :meth:`Dataset.empty_like` when the
    target is a sparse GTiff and ``no_data_value`` resolves to ``None``: with no
    sentinel, never-written blocks read back as ``0`` rather than no-data. Callers
    who intentionally build sentinel-free rasters can silence it precisely with
    ``warnings.filterwarnings("ignore", category=NoDataSentinelWarning)`` instead of
    muting every :class:`UserWarning`.
    """


def _derive_band_names(paths: list[str]) -> list[str]:
    """Derive band names from a list of single-band raster paths.

    For the common one-file-per-band layouts:

    * Earth Engine downloads — ``<assetSlug>.<bandName>.tif`` → ``<bandName>``
      (the part after the last dot in the file stem).
    * Landsat / Sentinel per-band files — ``..._SR_B4.TIF`` (no extra dots) →
      the whole stem ``..._SR_B4``.

    Duplicate names get a ``_<n>`` suffix so the result always has one unique
    name per input path.

    Args:
        paths: Resolved raster paths (already VSI-normalised).

    Returns:
        list[str]: One band name per input path, in order, all unique.
    """
    raw = []
    for path in paths:
        stem = Path(path).stem
        token = stem.rsplit(".", 1)[-1] if "." in stem else stem
        raw.append(token or stem)
    seen: dict[str, int] = {}
    names: list[str] = []
    for name in raw:
        if name in seen:
            seen[name] += 1
            names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            names.append(name)
    return names


def _same_grid(a: Dataset, b: Dataset) -> bool:
    """Return True if datasets ``a`` and ``b`` share CRS, size, and geotransform.

    Geotransform components are compared with a small relative tolerance so
    that byte-for-byte-identical grids (the normal case for per-band files of
    one scene) compare equal even after the round-trip through GDAL's
    floating-point geotransform.

    Args:
        a: Reference dataset.
        b: Dataset to compare against ``a``.

    Returns:
        bool: ``True`` iff both rasters occupy the same pixel grid in the
        same CRS.
    """
    return (
        a.epsg == b.epsg
        and a.rows == b.rows
        and a.columns == b.columns
        and bool(
            np.allclose(
                np.asarray(a.geotransform), np.asarray(b.geotransform), rtol=1e-7
            )
        )
    )


def _remap_nodata_to(arr: np.ndarray, src_nd: Any, dst_nd: Any) -> np.typing.NDArray:
    """Replace ``src_nd`` cells in ``arr`` with ``dst_nd`` when the two differ.

    Used by :meth:`Dataset.from_band_files` (``align=True`` branch) so that
    each per-source aligned band's fringe — filled by GDAL with the source's
    own no-data sentinel — matches the resolved output no-data after the
    "first-source-wins" reconciliation. The ``np.nan == np.nan -> False``
    quirk is handled so float-NaN sentinels are treated as equal.

    Args:
        arr: The aligned-band array to remap in place semantically (a new
            array is returned; ``arr`` is not mutated).
        src_nd: No-data sentinel currently in ``arr`` (this band's source).
        dst_nd: No-data sentinel the output band will declare.

    Returns:
        np.ndarray: ``arr`` unchanged when ``src_nd == dst_nd`` (incl. the
        both-NaN case) or when either is ``None`` (no sentinel to remap);
        otherwise a copy with the source sentinel rewritten to ``dst_nd``.
    """
    if src_nd is None or dst_nd is None:
        return arr
    src_is_nan = isinstance(src_nd, float) and np.isnan(src_nd)
    dst_is_nan = isinstance(dst_nd, float) and np.isnan(dst_nd)
    if src_is_nan and dst_is_nan:
        return arr
    if not src_is_nan and not dst_is_nan and src_nd == dst_nd:
        return arr
    try:
        dst_typed = np.asarray(dst_nd, dtype=arr.dtype).item()
    except (ValueError, OverflowError):
        # Sentinel doesn't fit the array dtype; leave the array alone (the
        # UserWarning from from_band_files already flagged the disagreement).
        return arr
    mask = np.isnan(arr) if src_is_nan else (arr == src_nd)
    return np.where(mask, dst_typed, arr)


if TYPE_CHECKING:
    import dask.array as da
    from cleopatra.basemap.geo import Basemap
    from cleopatra.glyphs.gridded.array_glyph import PlotKwargs, PointOverlay
    from cleopatra.styling.colorbar import ColorBar
    from cleopatra.styling.params import CellValues, Contour, DataStyle
    from cleopatra.styling.scaling import ColorScaling
    from geopandas import GeoDataFrame
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# Conventional coordinate-variable names, used as a fallback when a file
# declares no `<var>#axis` attributes. Units on these count as *axis* units,
# the only ones allowed to veto the CF geographic inference — except the
# vertical names above, which describe depth rather than the horizontal CRS.
# The X-ish and Y-ish halves of `_AXIS_VARIABLE_NAMES`, used to require that a
# veto comes from a PAIR of axes. Together they partition that list exactly, so
# the pairing rule introduces no new vocabulary.
_X_AXIS_NAMES = frozenset(
    {
        "x",
        "xc",
        "xdim",
        "x_dim",
        "lon",
        "longitude",
        "long",
        "rlon",
        "east",
        "easting",
        "nav_lon",
    }
)
_Y_AXIS_NAMES = frozenset(
    {
        "y",
        "yc",
        "ydim",
        "y_dim",
        "lat",
        "latitude",
        "rlat",
        "north",
        "northing",
        "nav_lat",
    }
)

_AXIS_VARIABLE_NAMES = frozenset(
    {
        "x",
        "y",
        "xc",
        "yc",
        "xdim",
        "ydim",
        "x_dim",
        "y_dim",
        "lon",
        "lat",
        "longitude",
        "latitude",
        "long",
        "rlon",
        "rlat",
        "east",
        "north",
        "easting",
        "northing",
        "nav_lon",
        "nav_lat",
    }
)


# Identity transform: one unit per pixel, origin at (0, 0), north-up. Used when a
# caller allocates a raster without saying where it sits in space.
_IDENTITY_GEO = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)


def _resolves_to_gtiff(path: str | Path | None) -> bool:
    """Whether a destination path selects the GTiff driver.

    The sparse / tiled / BigTIFF creation options and the unwritten-block
    no-data guarantee are GTiff-specific. Now that the driver comes from the
    extension rather than an explicit argument, "a path was given" no longer
    implies GTiff — a `.nc` destination really is netCDF — so the two have to
    be asked separately.

    Args:
        path: The destination, or `None` for an in-memory raster.

    Returns:
        bool: `True` only for a path whose extension resolves to GTiff.
    """
    return path is not None and resolve_output_driver(path) == "GTiff"


def _crs_wkt_from_epsg(epsg: str | int | None) -> str:
    """WKT for an EPSG code, or an empty string when the CRS is deliberately unset.

    Args:
        epsg: EPSG code, any CRS string GDAL accepts, or `None` to leave the
            raster without a CRS.

    Returns:
        str: The CRS as WKT, or `""` when `epsg` is falsy.
    """
    if not epsg:
        # A source can be genuinely ungeoreferenced. Propagate that rather than
        # stamping WGS 84 on it (ARC-26); SetProjection("") leaves it unset.
        wkt = ""
    else:
        try:
            # Keep the exact `sr_from_epsg` path for an EPSG int/numeric string;
            # anything else (e.g. a geostationary WKT with no EPSG) goes through
            # `sr_from_user_input` so it survives a rebuild instead of dying on
            # `int(None)` (#706).
            wkt = str(sr_from_epsg(int(epsg)).ExportToWkt())
        except (TypeError, ValueError):
            wkt = str(sr_from_user_input(epsg).ExportToWkt())
    return wkt


class Dataset(RasterBase):
    """Single-band or multi-band raster dataset (GeoTIFF, etc.).

    Wraps a GDAL dataset with spatial operations (crop, reproject, align,
    mosaic), band-level I/O, and no-data handling. For NetCDF files use
    the :class:`~pyramids.netcdf.NetCDF` subclass; for temporal stacks of
    rasters use :class:`~pyramids.dataset.DatasetCollection`.

    The eight public-API families are exposed as collaborator instances
    (`ds.io`, `ds.spatial`, `ds.bands`, `ds.analysis`,
    `ds.cell`, `ds.vectorize`, `ds.cog`, `ds.georef`) and via thin facade
    methods on the Dataset itself, so `ds.crop(mask)` and
    `ds.spatial.crop(mask)` are equivalent. Each collaborator holds a
    weakref proxy back to the Dataset; the proxy keeps GDAL handle
    release deterministic on Windows.
    """

    # Instance attributes assigned outside ``__init__`` — lazily by the io
    # engine (``_backend``), by warp/georef operations (``_warp_source``),
    # by the bytes/VSI round-trip (``_vsimem_path``), or on the base
    # ``Dataset`` that ``NetCDF`` produces for a flattened band axis (the
    # ``_band_dim_*`` / ``_variable_attrs`` group, also initialised in
    # ``NetCDF.__init__``). Declared here so the checker knows the surface;
    # the runtime values are set where each is produced.
    _backend: str
    # Strong reference to the *source GDAL raster* a warped VRT reads through, kept
    # so it outlives the VRT. It must not be the pyramids `Dataset`: engines reach
    # their parent through a `weakref.proxy`, so pinning that would keep nothing
    # alive. `NetCDF.warped_view` carries this across when it re-wraps a view.
    _warp_source: gdal.Dataset | None
    _vsimem_path: str
    _band_dim_name: str | None
    _band_dim_values: list[Any] | None
    _band_dim_names: tuple[str, ...]
    _band_dim_values_map: dict[str, list[Any] | None]
    _band_dim_sizes: tuple[int, ...]
    _variable_attrs: dict[str, Any]

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        *,
        gdal_env: dict[str, str] | None = None,
        open_options: tuple[str, ...] | list[str] | None = None,
    ):
        """Wrap an open ``gdal.Dataset`` as a :class:`Dataset`.

        A thin override of :meth:`RasterBase.__init__` that attaches a logger and
        forwards every argument unchanged; see the base for the full contract.
        Prefer :meth:`read_file` over constructing directly.

        Args:
            src: An open :class:`osgeo.gdal.Dataset` to wrap.
            access: The mode ``src`` was opened with — ``"read_only"`` (default)
                or ``"write"``.
            gdal_env: GDAL config captured for reopen paths; ``None`` captures
                nothing.
            open_options: GDAL open options captured for reopen paths; ``None``
                captures nothing (#1025).
        """
        self.logger = logging.getLogger(__name__)
        super().__init__(
            src, access=access, gdal_env=gdal_env, open_options=open_options
        )

        self._no_data_value = [
            src.GetRasterBand(i).GetNoDataValue() for i in range(1, self.band_count + 1)
        ]
        self._band_names = self._get_band_names()
        self._band_units = [
            src.GetRasterBand(i).GetUnitType() for i in range(1, self.band_count + 1)
        ]

        # Each collaborator owns the bodies of one public-API family
        # (io, spatial, bands, analysis, cell, vectorize, cog) and
        # holds a `weakref.proxy(self)` back-reference. Dataset
        # exposes facade methods that delegate to the collaborator,
        # so both `ds.crop(mask)` and `ds.spatial.crop(mask)` are
        # equivalent.
        self.io = IO(self)
        self.spatial = Spatial(self)
        self.bands = Bands(self)
        self.analysis = Analysis(self)
        self.cell = Cell(self)
        self.vectorize = Vectorize(self)
        self.cog = COG(self)
        self.georef = Georef(self)

    def _update_inplace(self, src: gdal.Dataset, access: str | None = None) -> None:
        """Swap internal state from a new GDAL dataset.

        Creates a fresh instance of `type(self)` and copies its
        internal state into `self`. Using `type(self)` rather
        than the literal `Dataset` is what keeps a NetCDF instance
        a NetCDF after any in-place op (set_crs, change_no_data_value,
        apply(inplace=True), to_file). Subclasses that carry extra
        state across the swap (e.g. NetCDF's variable-subset
        attributes) override this method.

        after `__dict__.update`, the collaborators on
        `self` came from `new.__dict__` and point at the temporary
        `new` instance, not at `self`. Re-bind every collaborator's
        `_ds` to `self` so subsequent `self.spatial.crop(...)`
        calls reach back into `self`, not the discarded `new`.

        Why ``collab._ds = self_proxy`` works despite the slot:
            ``_Engine`` declares ``__slots__ = ("_ds",)`` (see
            :mod:`pyramids.dataset.engines._base`). Slots prevent
            adding *new* attributes to an instance, not reassigning
            existing ones, so direct rebinding of the single
            declared slot stays legal. The proxy is freshly built
            from ``self`` (not pulled from ``new``) so the engines
            point at the live Dataset after the swap.
        """
        new = type(self)(src, access=access or self._access)
        self.__dict__.update(new.__dict__)
        # `update` merges, so a lazily-set cache the fresh instance never
        # populated would survive the swap and describe the OLD raster. Drop the
        # inferred-CF-CRS memo explicitly.
        self.__dict__.pop("_cf_crs_cache", None)
        # Re-bind via `weakref.proxy` so the back-reference stays
        # weak after the dict swap (matches `_Engine.__init__`).
        # Direct slot reassignment is allowed because `_Engine`
        # declares `_ds` as its only slot — see method docstring.
        self_proxy = weakref.proxy(self)
        for attr in _COLLABORATOR_ATTRS:
            collab = self.__dict__.get(attr)
            if collab is not None:
                collab._ds = self_proxy
        # Drop cached registered accessors so they rebuild against the new raster
        # (the `__dict__.update` above would otherwise keep a stale one).
        _invalidate_cached_accessors(self)

    def focal_mean(
        self, radius: int = 1, *, chunks=None, band: int = 0
    ) -> np.ndarray | da.Array:
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.focal_mean`."""
        return focal_mean(self, radius=radius, chunks=chunks, band=band)

    def focal_std(
        self, radius: int = 1, *, chunks=None, band: int = 0
    ) -> np.ndarray | da.Array:
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.focal_std`."""
        return focal_std(self, radius=radius, chunks=chunks, band=band)

    def focal_apply(
        self, func, radius: int = 1, *, chunks=None, band: int = 0
    ) -> np.ndarray | da.Array:
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.focal_apply`."""
        return focal_apply(self, func, radius=radius, chunks=chunks, band=band)

    def slope(
        self, *, chunks=None, band: int = 0, units: str = "degrees"
    ) -> np.ndarray | da.Array:
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.slope`."""
        return slope(self, chunks=chunks, band=band, units=units)

    def aspect(self, *, chunks=None, band: int = 0) -> np.ndarray | da.Array:
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.aspect`."""
        return aspect(self, chunks=chunks, band=band)

    def hillshade(
        self,
        *,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        chunks=None,
        band: int = 0,
    ) -> np.ndarray | da.Array:
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.hillshade`."""
        return hillshade(
            self,
            azimuth=azimuth,
            altitude=altitude,
            chunks=chunks,
            band=band,
        )

    def get_cell_coords(self, *args, **kwargs):
        """Facade — delegates to :meth:`Cell.get_cell_coords <pyramids.dataset.engines.Cell.get_cell_coords>`."""
        return self.cell.get_cell_coords(*args, **kwargs)

    def get_cell_polygons(self, *args, **kwargs):
        """Facade — delegates to :meth:`Cell.get_cell_polygons <pyramids.dataset.engines.Cell.get_cell_polygons>`."""
        return self.cell.get_cell_polygons(*args, **kwargs)

    def get_cell_points(self, *args, **kwargs):
        """Facade — delegates to :meth:`Cell.get_cell_points <pyramids.dataset.engines.Cell.get_cell_points>`."""
        return self.cell.get_cell_points(*args, **kwargs)

    def map_to_array_coordinates(self, *args, **kwargs):
        """Facade — delegates to :meth:`Cell.map_to_array_coordinates <pyramids.dataset.engines.Cell.map_to_array_coordinates>`."""
        return self.cell.map_to_array_coordinates(*args, **kwargs)

    def array_to_map_coordinates(self, *args, **kwargs):
        """Facade — delegates to :meth:`Cell.array_to_map_coordinates <pyramids.dataset.engines.Cell.array_to_map_coordinates>`."""
        return self.cell.array_to_map_coordinates(*args, **kwargs)

    def to_cog(self, *args, **kwargs):
        """Facade — delegates to :meth:`COG.to_cog <pyramids.dataset.engines.COG.to_cog>`."""
        return self.cog.to_cog(*args, **kwargs)

    @property
    def is_cog(self) -> bool:
        """Facade — delegates to :attr:`COG.is_cog <pyramids.dataset.engines.COG.is_cog>`."""
        return self.cog.is_cog

    def validate_cog(self, *args, **kwargs):
        """Facade — delegates to :meth:`COG.validate_cog <pyramids.dataset.engines.COG.validate_cog>`."""
        return self.cog.validate_cog(*args, **kwargs)

    def cog_info(self, *args, **kwargs):
        """Facade — delegates to :meth:`COG.info <pyramids.dataset.engines.COG.info>`."""
        return self.cog.info(*args, **kwargs)

    def to_cog_bytes(self, *args, **kwargs):
        """Facade — delegates to :meth:`COG.to_cog_bytes <pyramids.dataset.engines.COG.to_cog_bytes>`."""
        return self.cog.to_cog_bytes(*args, **kwargs)

    def read_part(self, *args, **kwargs):
        """Facade — delegates to :meth:`COG.read_part <pyramids.dataset.engines.COG.read_part>`."""
        return self.cog.read_part(*args, **kwargs)

    def preview(self, *args, **kwargs):
        """Facade — delegates to :meth:`COG.preview <pyramids.dataset.engines.COG.preview>`."""
        return self.cog.preview(*args, **kwargs)

    def point(self, *args, **kwargs):
        """Facade — delegates to :meth:`COG.point <pyramids.dataset.engines.COG.point>`."""
        return self.cog.point(*args, **kwargs)

    def read_tile(self, *args, **kwargs):
        """Facade — delegates to :meth:`COG.read_tile <pyramids.dataset.engines.COG.read_tile>`."""
        return self.cog.read_tile(*args, **kwargs)

    def to_feature_collection(self, *args, **kwargs):
        """Facade — delegates to :meth:`Vectorize.to_feature_collection <pyramids.dataset.engines.Vectorize.to_feature_collection>`."""
        return self.vectorize.to_feature_collection(*args, **kwargs)

    def contour(self, *args, **kwargs):
        """Facade — delegates to :meth:`Vectorize.contour <pyramids.dataset.engines.Vectorize.contour>`."""
        return self.vectorize.contour(*args, **kwargs)

    def translate(self, *args, **kwargs):
        """Facade — delegates to :meth:`Vectorize.translate <pyramids.dataset.engines.Vectorize.translate>`."""
        return self.vectorize.translate(*args, **kwargs)

    def cluster(self, *args, **kwargs):
        """Facade — delegates to :meth:`Vectorize.cluster <pyramids.dataset.engines.Vectorize.cluster>`."""
        return self.vectorize.cluster(*args, **kwargs)

    def to_polygons(self, *args, **kwargs):
        """Facade — delegates to :meth:`Vectorize.to_polygons <pyramids.dataset.engines.Vectorize.to_polygons>`."""
        return self.vectorize.to_polygons(*args, **kwargs)

    def cluster2(self, *args, **kwargs):
        """Deprecated alias for :meth:`to_polygons` — delegates to
        :meth:`Vectorize.cluster2 <pyramids.dataset.engines.Vectorize.cluster2>`."""
        return self.vectorize.cluster2(*args, **kwargs)

    def stats(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.stats <pyramids.dataset.engines.Analysis.stats>`."""
        return self.analysis.stats(*args, **kwargs)

    def count_domain_cells(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.count_domain_cells <pyramids.dataset.engines.Analysis.count_domain_cells>`."""
        return self.analysis.count_domain_cells(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.apply <pyramids.dataset.engines.Analysis.apply>`.

        The collaborator returns `None` for `inplace=True` so the facade
        can substitute the actual `self` (preserving identity); the proxy
        used by the collaborator's back-reference would otherwise fail
        `result is ds` checks.
        """
        result = self.analysis.apply(*args, **kwargs)
        return self if result is None else result

    def fill(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.fill <pyramids.dataset.engines.Analysis.fill>`.

        The collaborator returns `None` for `inplace=True`; see
        :meth:`apply` for the rationale.
        """
        result = self.analysis.fill(*args, **kwargs)
        return self if result is None else result

    def extract(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.extract <pyramids.dataset.engines.Analysis.extract>`."""
        return self.analysis.extract(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.sample <pyramids.dataset.engines.Analysis.sample>`."""
        return self.analysis.sample(*args, **kwargs)

    def sieve(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.sieve <pyramids.dataset.engines.Analysis.sieve>`."""
        return self.analysis.sieve(*args, **kwargs)

    def proximity(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.proximity <pyramids.dataset.engines.Analysis.proximity>`."""
        return self.analysis.proximity(*args, **kwargs)

    def overlay(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.overlay <pyramids.dataset.engines.Analysis.overlay>`."""
        return self.analysis.overlay(*args, **kwargs)

    def get_mask(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.get_mask <pyramids.dataset.engines.Analysis.get_mask>`."""
        return self.analysis.get_mask(*args, **kwargs)

    def mask_flags(self, *args, **kwargs):
        """Facade — :meth:`Analysis.mask_flags <pyramids.dataset.engines.Analysis.mask_flags>`."""
        return self.analysis.mask_flags(*args, **kwargs)

    def read_masks(self, *args, **kwargs):
        """Facade — :meth:`Analysis.read_masks <pyramids.dataset.engines.Analysis.read_masks>`."""
        return self.analysis.read_masks(*args, **kwargs)

    def create_mask_band(self, *args, **kwargs):
        """Facade — :meth:`Analysis.create_mask_band <pyramids.dataset.engines.Analysis.create_mask_band>`."""
        return self.analysis.create_mask_band(*args, **kwargs)

    def footprint(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.footprint <pyramids.dataset.engines.Analysis.footprint>`."""
        return self.analysis.footprint(*args, **kwargs)

    def get_histogram(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.get_histogram <pyramids.dataset.engines.Analysis.get_histogram>`."""
        return self.analysis.get_histogram(*args, **kwargs)

    def plot_histogram(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.plot_histogram <pyramids.dataset.engines.Analysis.plot_histogram>`."""
        return self.analysis.plot_histogram(*args, **kwargs)

    def to_image(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.to_image <pyramids.dataset.engines.Analysis.to_image>`."""
        return self.analysis.to_image(*args, **kwargs)

    def plot_vector_field(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.plot_vector_field <pyramids.dataset.engines.Analysis.plot_vector_field>`."""
        return self.analysis.plot_vector_field(*args, **kwargs)

    def _resolve_plot_band(
        self, band: int | None, rgb: list[int] | None
    ) -> tuple[int, list[int] | None]:
        """Resolve which band index (and effective ``rgb`` list) to render for :meth:`plot`.

        Applies the GeoTIFF / Sentinel-imagery band-resolution policy that used to live
        inside :meth:`Analysis.plot`. The rules, in order, are:

        1. If ``band`` is explicitly provided, it is returned as-is (and ``rgb`` passes
           through untouched).
        2. If the dataset has fewer than 3 bands, return ``(0, rgb)``.
        3. If the dataset has 3+ bands but **no** band is tagged as an RGB channel
           (``red``/``green``/``blue``), return ``(0, rgb)``. This is the D-1 fix:
           ``band_count >= 3`` alone is not a sufficient signal that the data is an RGB
           image — multi-band scalar cubes (e.g. time series stacked into one GeoTIFF)
           also have ``band_count >= 3`` and must not be misinterpreted as RGB. Only the
           three RGB-channel interpretations count; ``palette_index``, ``gray_index`` and
           the other single-channel tags are rendered as single bands, not RGB (see #910).
        4. Otherwise, treat the dataset as RGB imagery. If ``rgb`` was supplied, its
           first entry is the red band. If it was not supplied, resolve red/green/blue
           via :meth:`get_band_by_color`; fall back to ``[2, 1, 0]`` (the default
           Sentinel-2 band order) only when one or more colour channels can't be
           identified.

        Args:
            band: User-supplied band index, or ``None`` to trigger the heuristic.
            rgb: User-supplied ``[r, g, b]`` band index list, or ``None``.

        Returns:
            tuple[int, list[int] | None]: The resolved single-band index and the
                effective ``rgb`` list to forward to :meth:`Analysis.plot`. The ``rgb``
                element is ``None`` when no RGB rendering should happen.

        Examples:
            - Explicit ``band`` is always returned untouched (rule 1):

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(4, 8, 8).astype(np.float32)
              >>> ds = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.1, epsg=4326),
              ... )
              >>> ds._resolve_plot_band(band=2, rgb=None)
              (2, None)

              ```

            - Single-band raster falls back to band ``0`` (rule 2):

              ```python
              >>> single = np.random.rand(6, 6).astype(np.float32)
              >>> ds_1band = Dataset.from_array(
              ...     single,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.1, epsg=4326),
              ... )
              >>> ds_1band._resolve_plot_band(band=None, rgb=None)
              (0, None)

              ```

            - Multi-band dataset with no ``ColorInterpretation`` defaults to band ``0``
              (rule 3, the D-1 fix). ``Dataset.from_array`` produces a multi-band
              MEM raster whose bands all report ``undefined`` colour interpretation —
              asserted explicitly here so this doctest fails loudly if that ever changes:

              ```python
              >>> list(ds.band_color.values())
              ['undefined', 'undefined', 'undefined', 'undefined']
              >>> ds._resolve_plot_band(band=None, rgb=None)
              (0, None)

              ```

            - Explicit ``rgb`` passes through alongside an explicit ``band``:

              ```python
              >>> ds._resolve_plot_band(band=1, rgb=[2, 1, 0])
              (1, [2, 1, 0])

              ```

            - A ``palette_index`` band is not an RGB channel, so it does not trigger
              the RGB branch — the raster resolves to the paletted band so its GDAL
              colour table renders (#913; band ``0`` here). A fresh dataset is built
              here so the example is independent of the tags set above:

              ```python
              >>> paletted = np.random.rand(3, 8, 8).astype(np.float32)
              >>> ds_pal = Dataset.from_array(
              ...     paletted,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.1, epsg=4326),
              ... )
              >>> ds_pal.band_color = {0: 'palette_index'}
              >>> ds_pal._resolve_plot_band(band=None, rgb=None)
              (0, None)

              ```
        """
        if band is not None:
            # Coerce to a plain ``int`` here too (the RGB branch already
            # does) so the return type matches the ``tuple[int, ...]``
            # docstring even when the caller passed e.g. a ``numpy.int64``.
            resolved_band = int(band)
            resolved_rgb = rgb
        elif self.band_count < 3:
            resolved_band = 0
            resolved_rgb = rgb
        else:
            resolved_band, resolved_rgb = self._resolve_multiband_plot(rgb)
        return resolved_band, resolved_rgb

    def _resolve_multiband_plot(
        self, rgb: list[int] | None
    ) -> tuple[int, list[int] | None]:
        """Resolve ``(band, rgb)`` for a raster with ``band_count >= 3``.

        Only a true RGB channel (``red``/``green``/``blue``) marks the raster as
        RGB imagery; ``undefined``, ``palette_index``, ``gray_index`` and the
        other single-channel/paletted interpretations do not -- otherwise a
        multi-band raster carrying any of them is mis-resolved as a false-colour
        RGB composite (see #910). A non-RGB raster renders a single band:
        ``rgb[0]`` when an explicit ``rgb`` is supplied, else the first
        ``palette_index`` band so its GDAL colour table renders (#913), else
        band 0. The downstream ``exclude_value`` nodata mask keys off the same
        band the render uses.
        """
        band_colors = list(self.band_color.values())
        has_rgb_interp = any(c in RGB_CHANNEL_INTERPS for c in band_colors)
        if not has_rgb_interp:
            resolved_rgb = rgb
            if rgb is not None:
                resolved_band = int(rgb[0])
            else:
                # Prefer a paletted band so its colour table renders (#913),
                # otherwise fall back to band 0.
                palette_bands = [
                    i for i, c in enumerate(band_colors) if c == "palette_index"
                ]
                resolved_band = palette_bands[0] if palette_bands else 0
        else:
            resolved_rgb = rgb if rgb is not None else self._infer_rgb_band_order()
            resolved_band = int(resolved_rgb[0])
        return resolved_band, resolved_rgb

    def _infer_rgb_band_order(self) -> list[int]:
        """Infer the ``[r, g, b]`` band-index order from the bands' colour tags.

        Resolves red/green/blue via :meth:`get_band_by_color`; falls back to the
        Sentinel-2 default ``[2, 1, 0]`` (emitting a :class:`DeprecationWarning`)
        when any channel cannot be identified from the tags.
        """
        candidate: list[int | None] = [
            self.get_band_by_color("red"),
            self.get_band_by_color("green"),
            self.get_band_by_color("blue"),
        ]
        if None in candidate:
            warnings.warn(
                "The implicit Sentinel-2 RGB band order [2, 1, 0] used "
                "when colour-interpretation is absent is deprecated and "
                "will be removed: it is a remote-sensing sensor "
                "assumption, not a generic raster default. Pass an "
                "explicit rgb=[...] (e.g. via rgb_options) instead.",
                DeprecationWarning,
                stacklevel=5,
            )
            resolved_rgb = [2, 1, 0]
        else:
            # None NOT in candidate here, so every element is a
            # plain int -- mypy does not narrow list contents from
            # an `in` check.
            resolved_rgb = [int(v) for v in cast("list[int]", candidate)]
        return resolved_rgb

    # The override is deliberate: it narrows the base's open **kwargs to cleopatra's
    # typed PlotKwargs. The fig/ax pair itself matches the RasterBase contract.
    def plot(  # type: ignore[override]
        self,
        band: int | None = None,
        exclude_value: Any | None = None,
        overview: bool | None = False,
        overview_index: int | None = 0,
        basemap: bool | str | dict[str, Any] | Basemap | None = None,
        colorbar: bool | ColorBar | None = None,
        points: np.ndarray | PointOverlay | None = None,
        kind: str = "auto",
        title: str | None = None,
        color: ColorScaling | None = None,
        contour: Contour | None = None,
        cells: CellValues | None = None,
        data_style: DataStyle | None = None,
        rgb_options: dict | None = None,
        *,
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs: Unpack[PlotKwargs],
    ):
        """Plot the values/overviews of a band.

        Facade for :meth:`Analysis.plot <pyramids.dataset.engines.Analysis.plot>`. Resolves
        the band index via :meth:`_resolve_plot_band` (GeoTIFF/Sentinel semantics) and then
        forwards the call to the generic rendering engine.

        When ``band`` is ``None`` and the dataset looks like an RGB image — i.e. it has
        at least 3 bands **and** at least one band is tagged as an RGB channel
        (``red``/``green``/``blue``) — the red band is auto-selected (either from
        ``rgb[0]`` or by resolving the colour tags). A ``palette_index``, ``gray_index``
        or other non-RGB interpretation does **not** count as RGB imagery. Otherwise the
        facade defaults to band ``0``. See :meth:`Analysis.plot` for the full kwargs
        surface.

        The four satellite-imagery options (``rgb``, ``surface_reflectance``, ``cutoff``,
        ``percentile``) are passed through the single ``rgb_options=`` dict.

        Args:
            band (int, optional):
                Band index to render. When ``None``, the index is resolved by
                :meth:`_resolve_plot_band`.
            exclude_value (Any, optional):
                Pixel value to mask out before plotting. Default is ``None``.
            overview (bool, optional):
                If ``True``, plot the overview pyramid level instead of the full-resolution
                array. Default is ``False``.
            overview_index (int, optional):
                Index of the overview level to plot when ``overview=True``. Default is ``0``.
            basemap (bool, str, or Basemap, optional):
                Reference layer, dispatched by type. ``True`` or a tile-provider string
                (e.g. ``"CartoDB.Positron"``) overlays a pyramids web-tile basemap. A
                ``pyramids.plot.Basemap(relief=..., features=...)``
                draws a shaded-relief / coastline layer instead. Passing a ``dict`` here is
                a deprecated alias for ``Basemap`` (emits a ``DeprecationWarning``). Default
                is ``None``. Requires the ``[viz]`` extra.
            colorbar (bool or ColorBar, optional):
                Colour-bar spec. A ``pyramids.plot.ColorBar(label=…, length=…,
                orientation=…, …)`` draws a configured bar. The loose ``cbar_*`` /
                ``ticks_spacing`` kwargs it replaces were removed — passing one now raises a
                :class:`ValueError` pointing here. ``False`` hides the bar, ``None`` uses
                cleopatra's default. Default is ``None``.
            points (np.ndarray or PointOverlay, optional):
                Point overlay. A 3-column array ``(value, row, col)`` draws unstyled
                points; pass a ``pyramids.plot.PointOverlay(points, color=…, size=…, …)``
                to style them. Default is ``None``.
            kind (str, optional):
                Renderer to use. ``"auto"`` (default) picks per data; otherwise one of
                ``"imshow"`` / ``"pcolormesh"`` / ``"contour"`` / ``"contourf"``.
            title (str, optional):
                Axes title. Default is ``None`` (cleopatra's default title).
            color (ColorScaling, optional):
                Colour-scale spec ``pyramids.plot.ColorScaling`` (linear / power / sym-log /
                boundary / midpoint norm), e.g. ``ColorScaling.power(gamma=0.7)`` or
                ``ColorScaling.boundary(bounds=[0, 0.5, 1])``. Default ``None``.
            contour (Contour, optional):
                Contour-line spec ``pyramids.plot.Contour(levels=…, labels=…, label_kw=…)``.
                Default ``None``.
            cells (CellValues, optional):
                Per-cell value annotation ``pyramids.plot.CellValues(show=…, size=…,
                background_threshold=…)``. Default ``None``.
            data_style (DataStyle, optional):
                Data-style / relief spec ``pyramids.plot.DataStyle(style=…, hillshade=…)``.
                Default ``None``.
            rgb_options (dict, optional):
                Grouped Sentinel-imagery options for a true-colour composite. Accepted
                keys: ``"rgb"`` (3- or 4-element band-index list ``[r, g, b(, a)]``, only
                honoured when the dataset has >= 3 bands and a colour interpretation),
                ``"surface_reflectance"`` (reflectance scale factor, e.g. ``10000`` for
                Sentinel-2), ``"cutoff"`` (per-band clip values), ``"percentile"``
                (percentile stretch). Default is ``None``.
            fig (matplotlib.figure.Figure, optional):
                Draw into this figure instead of creating one. Pass it alongside ``ax``;
                supplying ``fig`` on its own currently raises inside cleopatra
                (serapeum-org/cleopatra#326). Default is ``None``.
            ax (matplotlib.axes.Axes, optional):
                Draw into these axes instead of creating them. This is what lets several
                rasters share one figure — a ``plt.subplots`` grid of side-by-side panels —
                while every panel keeps the georeferenced extent and nodata masking that
                ``plot`` applies. An axes already carries its figure, so ``ax`` on its own
                is sufficient. A shared colour range across panels is then applied through
                the returned glyph, whose colour bar tracks its mappable
                (``glyph.cbar.mappable is glyph.im``), e.g. ``glyph.im.set_clim(0, vmax)``.
                Default is ``None``.

                ```python
                >>> import matplotlib.pyplot as plt  # doctest: +SKIP
                >>> fig, axes = plt.subplots(1, 3)  # doctest: +SKIP
                >>> panels = [ds.plot(fig=fig, ax=a) for a in axes]  # doctest: +SKIP

                ```
            **kwargs:
                Additional keyword arguments forwarded verbatim to
                :meth:`Analysis.plot`. See that method for the full kwargs surface
                (figure size, color scale, color bar, basemap, etc.). Notably
                ``add_colorbar`` (``bool``, default ``True``) is a cleopatra
                pass-through: set ``add_colorbar=False`` to suppress the
                auto-generated colorbar (the returned glyph's ``cbar`` is then
                ``None``).

        Returns:
            ArrayGlyph: A cleopatra ``ArrayGlyph`` wrapping the rendered figure.
                Use it to drop down to raw matplotlib:

                - ``glyph.fig`` / ``glyph.ax`` — the :class:`matplotlib.figure.Figure`
                  and :class:`matplotlib.axes.Axes`.
                - ``glyph.im`` — the colour-mapped mappable (populated for every
                  ``kind=``: imshow/pcolormesh/contour/contourf). Use it to tweak
                  colour limits after the fact, e.g. ``glyph.im.set_clim(0, 100)``.
                - ``glyph.cbar`` — the auto-created :class:`matplotlib.colorbar.Colorbar`,
                  or ``None`` when ``add_colorbar=False`` (or for RGB renders).

                ```python
                >>> glyph = dataset.plot(band=0, kind="pcolormesh")  # doctest: +SKIP
                >>> glyph.im.set_clim(0, 100)  # doctest: +SKIP
                >>> _ = glyph.cbar.set_label("elevation [m]")  # doctest: +SKIP

                ```

        Examples:
            - Render the first band of a single-band MEM raster. Tagged ``+SKIP`` because
              the call requires the optional ``[viz]`` extra (cleopatra + matplotlib):

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(8, 8).astype(np.float32)
              >>> ds = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.1, epsg=4326),
              ... )
              >>> cleo = ds.plot()  # doctest: +SKIP
              >>> cleo.fig          # doctest: +SKIP
              <Figure size 800x800 with 2 Axes>

              ```

            - Override the resolved band index. The facade forwards ``band=1`` straight
              to the engine without consulting the heuristic:

              ```python
              >>> cleo = ds.plot(band=1)  # doctest: +SKIP

              ```

            - Render a multi-band raster as a true-colour composite via the
              recommended ``rgb_options=`` group:

              ```python
              >>> arr3 = np.random.rand(3, 8, 8).astype(np.float32)
              >>> rgb_ds = Dataset.from_array(
              ...     arr3,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.1, epsg=4326),
              ... )
              >>> cleo = rgb_ds.plot(  # doctest: +SKIP
              ...     rgb_options={"rgb": [0, 1, 2], "surface_reflectance": 255},
              ... )

              ```
        """
        rgb, surface_reflectance, cutoff, percentile = self._unpack_rgb_options(
            rgb_options
        )
        resolved_band, resolved_rgb = self._resolve_plot_band(band, rgb)
        # Spread the explicitly-set cleopatra render groups as their own ``**`` (not merged
        # into the typed ``**kwargs``, whose PlotKwargs TypedDict has no group keys); the
        # unset ones are dropped so they do not override cleopatra's backend default for
        # that group.
        group_kwargs = nonnull_group_kwargs(
            color=color, contour=contour, cells=cells, data_style=data_style
        )
        return self.analysis.plot(
            band=resolved_band,
            exclude_value=exclude_value,
            rgb=resolved_rgb,
            surface_reflectance=surface_reflectance,
            cutoff=cutoff,
            overview=overview,
            overview_index=overview_index,
            percentile=percentile,
            basemap=basemap,
            colorbar=colorbar,
            points=points,
            kind=kind,
            title=title,
            fig=fig,
            ax=ax,
            **group_kwargs,
            **kwargs,
        )

    @staticmethod
    def _unpack_rgb_options(
        rgb_options: dict | None,
    ) -> tuple[list[int] | None, int | None, list | None, int | None]:
        """Unpack the ``rgb_options`` group into the four Sentinel-imagery values.

        Args:
            rgb_options: Grouped Sentinel-imagery options, or ``None``. Accepted keys:
                ``"rgb"``, ``"surface_reflectance"``, ``"cutoff"``, ``"percentile"``.

        Returns:
            tuple: ``(rgb, surface_reflectance, cutoff, percentile)`` — each the value from
                ``rgb_options`` or ``None`` when absent.

        Raises:
            ValueError: If ``rgb_options`` contains a key outside the accepted set.

        Examples:
            - Unpack the grouped form (``None`` yields an all-``None`` tuple):

                ```python
                >>> from pyramids.dataset import Dataset
                >>> Dataset._unpack_rgb_options(
                ...     {"rgb": [2, 1, 0], "surface_reflectance": 10000}
                ... )
                ([2, 1, 0], 10000, None, None)
                >>> Dataset._unpack_rgb_options(None)
                (None, None, None, None)

                ```

            - An unknown key raises :class:`ValueError`:

                ```python
                >>> Dataset._unpack_rgb_options(  # doctest: +IGNORE_EXCEPTION_DETAIL
                ...     {"unknown": 1}
                ... )
                Traceback (most recent call last):
                    ...
                ValueError: Unknown keys in `rgb_options`: ['unknown']...

                ```
        """
        accepted = ("rgb", "surface_reflectance", "cutoff", "percentile")
        opts = rgb_options or {}
        unknown = set(opts) - set(accepted)
        if unknown:
            raise ValueError(
                f"Unknown keys in `rgb_options`: {sorted(unknown)}. "
                f"Accepted: {sorted(accepted)}."
            )
        return (
            opts.get("rgb"),
            opts.get("surface_reflectance"),
            opts.get("cutoff"),
            opts.get("percentile"),
        )

    def crop(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.crop <pyramids.dataset.engines.Spatial.crop>`."""
        return self.spatial.crop(*args, **kwargs)

    def to_crs(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.to_crs <pyramids.dataset.engines.Spatial.to_crs>`."""
        return self.spatial.to_crs(*args, **kwargs)

    def set_gcps(self, *args, **kwargs):
        """Facade — delegates to :meth:`Georef.set_gcps <pyramids.dataset.engines.Georef.set_gcps>`."""
        return self.georef.set_gcps(*args, **kwargs)

    def georeference(self, *args, **kwargs):
        """Facade — :meth:`Georef.georeference <pyramids.dataset.engines.Georef.georeference>`."""
        return self.georef.georeference(*args, **kwargs)

    @property
    def gcps(self):
        """Facade — :attr:`Georef.gcps <pyramids.dataset.engines.Georef.gcps>`."""
        return self.georef.gcps

    @property
    def gcp_count(self):
        """Facade — :attr:`Georef.gcp_count <pyramids.dataset.engines.Georef.gcp_count>`."""
        return self.georef.gcp_count

    @property
    def gcp_projection(self):
        """Facade — :attr:`Georef.gcp_projection <pyramids.dataset.engines.Georef.gcp_projection>`."""
        return self.georef.gcp_projection

    @property
    def has_gcps(self):
        """Facade — :attr:`Georef.has_gcps <pyramids.dataset.engines.Georef.has_gcps>`."""
        return self.georef.has_gcps

    @property
    def rpcs(self):
        """Facade — :attr:`Georef.rpcs <pyramids.dataset.engines.Georef.rpcs>`."""
        return self.georef.rpcs

    @property
    def has_rpcs(self):
        """Facade — :attr:`Georef.has_rpcs <pyramids.dataset.engines.Georef.has_rpcs>`."""
        return self.georef.has_rpcs

    def set_rpcs(self, *args, **kwargs):
        """Facade — :meth:`Georef.set_rpcs <pyramids.dataset.engines.Georef.set_rpcs>`."""
        return self.georef.set_rpcs(*args, **kwargs)

    def orthorectify(self, *args, **kwargs):
        """Facade — :meth:`Georef.orthorectify <pyramids.dataset.engines.Georef.orthorectify>`."""
        return self.georef.orthorectify(*args, **kwargs)

    @property
    def geolocation(self):
        """Facade — :attr:`Georef.geolocation <pyramids.dataset.engines.Georef.geolocation>`."""
        return self.georef.geolocation

    @property
    def has_geolocation(self):
        """Facade — :attr:`Georef.has_geolocation <pyramids.dataset.engines.Georef.has_geolocation>`."""
        return self.georef.has_geolocation

    def geolocate(self, *args, **kwargs):
        """Facade — :meth:`Georef.geolocate <pyramids.dataset.engines.Georef.geolocate>`."""
        return self.georef.geolocate(*args, **kwargs)

    def _geolocation_source(self) -> Dataset:
        """The Dataset whose GDAL handle carries the ``GEOLOCATION`` domain.

        A base raster carries its geolocation arrays on its own handle, so the
        default is ``self``. ``NetCDF`` overrides this to reopen the classic GDAL
        handle, on which the domain is exposed (the multidimensional view drops it).
        """
        return self

    def warped_view(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.warped_view <pyramids.dataset.engines.Spatial.warped_view>`."""
        return self.spatial.warped_view(*args, **kwargs)

    def set_crs(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.set_crs <pyramids.dataset.engines.Spatial.set_crs>`."""
        return self.spatial.set_crs(*args, **kwargs)

    def wrap_longitude(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.wrap_longitude <pyramids.dataset.engines.Spatial.wrap_longitude>`."""
        return self.spatial.wrap_longitude(*args, **kwargs)

    def resample(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.resample <pyramids.dataset.engines.Spatial.resample>`."""
        return self.spatial.resample(*args, **kwargs)

    def align(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.align <pyramids.dataset.engines.Spatial.align>`."""
        return self.spatial.align(*args, **kwargs)

    def fill_gaps(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.fill_gaps <pyramids.dataset.engines.Spatial.fill_gaps>`."""
        return self.spatial.fill_gaps(*args, **kwargs)

    def read_array(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.read_array <pyramids.dataset.engines.IO.read_array>`."""
        return self.io.read_array(*args, **kwargs)

    def _materialize_md_view(self) -> None:
        """Make the backing raster window-readable. No-op for an ordinary raster.

        Hook overridden by :class:`pyramids.netcdf.NetCDF`, whose variable subsets are backed by a
        GDAL multidimensional ``AsClassicDataset`` view that GDAL >= 3.13 cannot read with a partial
        window (it raises ``arrayStartIdx[...] >= <dim>``). The override replaces that view with a
        materialised in-memory raster. A plain :class:`Dataset` is already window-readable, so this
        does nothing.
        """
        return None

    def read_windows(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.read_windows <pyramids.dataset.engines.IO.read_windows>`."""
        return self.io.read_windows(*args, **kwargs)

    def write_array(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.write_array <pyramids.dataset.engines.IO.write_array>`."""
        return self.io.write_array(*args, **kwargs)

    def to_file(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.to_file <pyramids.dataset.engines.IO.to_file>`."""
        return self.io.to_file(*args, **kwargs)

    def to_bytes(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.to_bytes <pyramids.dataset.engines.IO.to_bytes>`."""
        return self.io.to_bytes(*args, **kwargs)

    def to_raster(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.to_raster <pyramids.dataset.engines.IO.to_raster>`."""
        return self.io.to_raster(*args, **kwargs)

    def get_block_arrangement(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.get_block_arrangement <pyramids.dataset.engines.IO.get_block_arrangement>`."""
        return self.io.get_block_arrangement(*args, **kwargs)

    def get_tile(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.get_tile <pyramids.dataset.engines.IO.get_tile>`."""
        return self.io.get_tile(*args, **kwargs)

    def map_blocks(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.map_blocks <pyramids.dataset.engines.IO.map_blocks>`."""
        return self.io.map_blocks(*args, **kwargs)

    def to_xyz(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.to_xyz <pyramids.dataset.engines.IO.to_xyz>`."""
        return self.io.to_xyz(*args, **kwargs)

    def to_terrain_rgb(self, *args, **kwargs):
        """Facade — delegates to
        :meth:`IO.to_terrain_rgb <pyramids.dataset.engines.IO.to_terrain_rgb>`."""
        return self.io.to_terrain_rgb(*args, **kwargs)

    @property
    def overview_count(self):
        """Facade — delegates to :attr:`IO.overview_count <pyramids.dataset.engines.IO.overview_count>`."""
        return self.io.overview_count

    def create_overviews(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.create_overviews <pyramids.dataset.engines.IO.create_overviews>`."""
        return self.io.create_overviews(*args, **kwargs)

    def recreate_overviews(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.recreate_overviews <pyramids.dataset.engines.IO.recreate_overviews>`."""
        return self.io.recreate_overviews(*args, **kwargs)

    def get_overview(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.get_overview <pyramids.dataset.engines.IO.get_overview>`."""
        return self.io.get_overview(*args, **kwargs)

    def get_overview_dataset(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.get_overview_dataset <pyramids.dataset.engines.IO.get_overview_dataset>`."""
        return self.io.get_overview_dataset(*args, **kwargs)

    def read_overview_array(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.read_overview_array <pyramids.dataset.engines.IO.read_overview_array>`."""
        return self.io.read_overview_array(*args, **kwargs)

    def _read_block(self, *args, **kwargs):
        """Facade — concrete override of the abstract :meth:`RasterBase._read_block`."""
        return self.io._read_block(*args, **kwargs)

    def get_attribute_table(self, *args, **kwargs):
        """Facade — delegates to :meth:`Bands.get_attribute_table <pyramids.dataset.engines.Bands.get_attribute_table>`."""
        return self.bands.get_attribute_table(*args, **kwargs)

    def set_attribute_table(self, *args, **kwargs):
        """Facade — delegates to :meth:`Bands.set_attribute_table <pyramids.dataset.engines.Bands.set_attribute_table>`."""
        return self.bands.set_attribute_table(*args, **kwargs)

    def add_band(self, *args, **kwargs):
        """Facade — delegates to :meth:`Bands.add_band <pyramids.dataset.engines.Bands.add_band>`."""
        return self.bands.add_band(*args, **kwargs)

    def get_band_by_color(self, *args, **kwargs):
        """Facade — delegates to :meth:`Bands.get_band_by_color <pyramids.dataset.engines.Bands.get_band_by_color>`."""
        return self.bands.get_band_by_color(*args, **kwargs)

    def select_bands(self, *args, **kwargs):
        """Facade — delegates to :meth:`Bands.select <pyramids.dataset.engines.Bands.select>`."""
        return self.bands.select(*args, **kwargs)

    def change_no_data_value(self, *args, **kwargs):
        """Facade — concrete override of the abstract :meth:`RasterBase.change_no_data_value`.

        The collaborator returns `None` for the `inplace=True` path; the
        facade substitutes `self` for identity preservation, matching
        :meth:`apply` and :meth:`fill`.
        """
        result = self.bands.change_no_data_value(*args, **kwargs)
        return self if result is None else result

    @property
    def band_color(self):
        """Facade — delegates to :attr:`Bands.band_color <pyramids.dataset.engines.Bands.band_color>`."""
        return self.bands.band_color

    @band_color.setter
    def band_color(self, values):
        """Facade setter.

        Raises:
            ReadOnlyError: The dataset is opened read-only on-disk (a bare
                `SetColorInterpretation` would otherwise silently spill a PAM sidecar).
        """
        self._require_writable("set band colors")
        self.bands.band_color = values

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
        """Facade — delegates to :meth:`Bands.set_color_ramp <pyramids.dataset.engines.Bands.set_color_ramp>`.

        Raises:
            ReadOnlyError: The dataset is opened read-only on-disk (writing the palette
                would otherwise silently spill a PAM sidecar).
        """
        self._require_writable("set a color ramp")
        return self.bands.set_color_ramp(
            band,
            start_value=start_value,
            end_value=end_value,
            start_color=start_color,
            end_color=end_color,
            colormap=colormap,
        )

    @property
    def color_table(self):
        """Facade — delegates to :attr:`Bands.color_table <pyramids.dataset.engines.Bands.color_table>`."""
        return self.bands.color_table

    @color_table.setter
    def color_table(self, df):
        """Facade setter.

        Raises:
            ReadOnlyError: The dataset is opened read-only on-disk (a bare
                `SetColorTable` would otherwise silently spill a PAM sidecar).
        """
        self._require_writable("set the color table")
        self.bands.color_table = df

    def _check_no_data_value(self, *args, **kwargs):
        """Facade — concrete override of the abstract :meth:`RasterBase._check_no_data_value`."""
        return self.bands._check_no_data_value(*args, **kwargs)

    def _set_no_data_value(self, *args, **kwargs):
        """Facade — concrete override of the abstract :meth:`RasterBase._set_no_data_value`."""
        return self.bands._set_no_data_value(*args, **kwargs)

    def _calculate_bbox(self) -> list:
        """Concrete override of :meth:`RasterBase._calculate_bbox`.

        Direct on Dataset (not via the Bands collaborator) because the
        `bbox` / `bounds` properties are reachable before the
        collaborator is wired during `Dataset.__init__`.
        """
        # Derive the extent from the geotransform's separate X/Y pixel sizes (gt[1], gt[5]) rather
        # than a single cell_size, so non-square grids (e.g. 2° lon, 1° lat) are not stretched.
        gt = self.geotransform
        x_min, y_max = gt[0], gt[3]
        x_max = x_min + self.columns * gt[1]
        y_min = y_max + self.rows * gt[5]
        return [x_min, y_min, x_max, y_max]

    def _calculate_bounds(self):
        """Concrete override of :meth:`RasterBase._calculate_bounds`."""
        x_min, y_min, x_max, y_max = self._calculate_bbox()
        coords = [(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max)]
        poly = create_polygon(coords)
        gdf = gpd.GeoDataFrame(geometry=[poly])
        gdf.set_crs(crs_spec(self.epsg, self.crs), inplace=True)
        return gdf

    def _get_band_names(self) -> list[str]:
        """Concrete override of :meth:`RasterBase._get_band_names`.

        Defined directly on Dataset (not via the bands collaborator)
        because `Dataset.__init__` calls `self._get_band_names()`
        before the `Bands` collaborator is wired up. Mirrors
        :meth:`Bands._get_band_names`.
        """
        names: list[str] = []
        for i in range(1, self.band_count + 1):
            band = self.raster.GetRasterBand(i)
            if band.GetDescription():
                names.append(band.GetDescription())
            else:
                band_name = f"Band_{band.GetBand()}"
                metadata = band.GetDataset().GetMetadata_Dict()
                if band_name in metadata and metadata[band_name]:
                    names.append(metadata[band_name])
                else:
                    names.append(band_name)
        return names

    def _get_crs(self) -> str:
        """Concrete override of :meth:`RasterBase._get_crs`.

        Defined directly on Dataset rather than as a facade because
        `RasterBase.__init__` calls `_get_epsg()` (which calls
        `_get_crs()`) before `Dataset.__init__` has a chance to wire
        up the Spatial collaborator. The Spatial collaborator's
        `Spatial._get_crs` returns the projection as GDAL reports it; this
        override adds the CF inference on top, so the two bodies differ.
        """
        crs = str(self.raster.GetProjection())
        cached: str | None = getattr(self, "_cf_crs_cache", None)
        if not crs and cached is not None:
            # Memoised: the scan below walks the whole metadata dict, and `.crs`
            # is read on every spatial operation. Keyed to the current raster —
            # `_update_inplace` drops it.
            crs = cached
        elif not crs:
            crs = self._infer_cf_crs()
            self._cf_crs_cache = crs
        return crs

    def _infer_cf_crs(self) -> str:
        """WGS 84 WKT when CF metadata says this is a lat/lon grid, else ``""``.

        A CF NetCDF opened through the *classic* driver reports no projection but
        exposes its coordinate metadata as ``<var>#units`` / ``<var>#axis``.
        Degrees east/north on a coordinate there mean a geographic grid by CF
        convention, so read it as WGS 84 rather than as ungeoreferenced (ARC-26).
        Any other unprojected raster still reports no CRS.

        The multidim equivalent is :meth:`pyramids.netcdf.NetCDF._cf_geographic_crs`,
        which reads the same evidence off the GDAL group API instead of off
        flattened metadata keys.

        Returns:
            str: WGS 84 WKT when the evidence says geographic, otherwise ``""``.
        """
        metadata = self.raster.GetMetadata() or {}
        evidence_names, axis_names, vertical_names = self._classify_cf_variables(
            metadata
        )
        units = self._units_of(metadata, evidence_names, set())
        axis_units = self._paired_axis_units(metadata, axis_names, vertical_names)
        crs = cf_geographic_wkt(units, axis_units)
        # Last check, on the geometry rather than the metadata: a grid whose own
        # coordinates fall outside the lon/lat range is not lat/lon, no matter
        # which variable supplied the degrees.
        if crs and not within_lonlat_range(self._own_extent()):
            crs = ""
        return crs

    @staticmethod
    def _classify_cf_variables(
        metadata: dict,
    ) -> tuple[set[str], set[str], set[str]]:
        """Split CF variable names into evidence, veto and vertical sets.

        Three distinct roles, easy to conflate:

        * **Evidence** — variables whose degrees units may imply a geographic
          grid. Only plausible coordinates qualify: one that declares
          `axis: X|Y`, one named in a `coordinates` attribute (how CF identifies
          a curvilinear grid's 2-D lat/lon, which declare no `axis`), or one
          carrying a conventional coordinate name. A wind direction in
          `degrees_east` is a data variable and is not evidence.
        * **Veto** — variables whose projected units mean the grid is not
          lat/lon. The declared axes are *unioned* with the name list rather than
          replacing it: a file may declare `axis` on some variables (the
          near-universal `time#axis = "T"`) while leaving its projected x/y
          undeclared, and keying on the declarations alone then misses the veto
          entirely. Only `axis: X|Y` counts as a declared horizontal axis — a
          declared `T` is not one.
        * **Vertical** — declared `axis: Z`, plus the conventional names. A depth
          or height in metres says nothing about the horizontal frame, so it must
          never veto.

        Args:
            metadata: GDAL metadata dict, with flattened `<var>#attr` keys.

        Returns:
            tuple[set[str], set[str], set[str]]: `(evidence, veto, vertical)`
            variable names, lower-cased.
        """
        coordinate_refs: set[str] = set()
        declared_horizontal: set[str] = set()
        declared_vertical: set[str] = set()
        for key, value in metadata.items():
            if not isinstance(value, str):
                continue
            lowered = key.lower()
            if lowered.endswith("#coordinates"):
                coordinate_refs.update(name.lower() for name in value.split())
            elif lowered.endswith("#axis"):
                name = key.rsplit("#", 1)[0].rsplit("/", 1)[-1].lower()
                role = value.strip().upper()
                if role == "Z":
                    declared_vertical.add(name)
                elif role in ("X", "Y"):
                    declared_horizontal.add(name)
        evidence = declared_horizontal | coordinate_refs | _AXIS_VARIABLE_NAMES
        veto = declared_horizontal | _AXIS_VARIABLE_NAMES
        return evidence, veto, set(VERTICAL_AXIS_NAMES) | declared_vertical

    @staticmethod
    def _paired_axis_units(
        metadata: dict, include: set[str], exclude: set[str]
    ) -> set[str]:
        """Projected units, but only when both an X and a Y axis carry them.

        A real projected grid always has *both* horizontal axes in projected
        units. One variable named `x` in metres beside `lon` / `lat` in degrees
        is a data variable — a ROMS bathymetry, a sea-surface height — and must
        not strip a geographic grid's CRS; `east` **and** `north`, or `rlon`
        **and** `rlat`, is a grid and must. Requiring the pair is what lets the
        name-based veto stay strict without destroying the CRS of a file that
        merely contains a similarly-named data variable.

        Args:
            metadata: GDAL metadata dict, with flattened `<var>#attr` keys.
            include: Candidate axis names.
            exclude: Names to skip (the vertical set).

        Returns:
            set[str]: The projected units when an X/Y pair carries them,
            otherwise an empty set.
        """
        per_axis: dict[str, str] = {}
        for key, value in metadata.items():
            if not isinstance(value, str) or not key.lower().endswith("#units"):
                continue
            name = key.rsplit("#", 1)[0].rsplit("/", 1)[-1].lower()
            if name in include and name not in exclude:
                per_axis[name] = value.strip().lower()
        projected = {
            name: unit
            for name, unit in per_axis.items()
            if unit in PROJECTED_AXIS_UNITS
        }
        has_x = any(name in _X_AXIS_NAMES for name in projected)
        has_y = any(name in _Y_AXIS_NAMES for name in projected)
        return set(projected.values()) if has_x and has_y else set()

    @staticmethod
    def _units_of(metadata: dict, include: set[str], exclude: set[str]) -> set[str]:
        """Lower-cased ``#units`` values of the named variables.

        Args:
            metadata: GDAL metadata dict, with flattened ``<var>#attr`` keys.
            include: Variable names to collect units from.
            exclude: Variable names to skip even when they are in `include`.

        Returns:
            set[str]: The matching unit strings.
        """
        collected = set()
        for key, value in metadata.items():
            if not isinstance(value, str) or not key.lower().endswith("#units"):
                continue
            name = key.rsplit("#", 1)[0].rsplit("/", 1)[-1].lower()
            if name in include and name not in exclude:
                collected.add(value.strip().lower())
        return collected

    def _own_extent(self) -> tuple[float, float, float, float] | None:
        """Corner-to-corner extent in the raster's own coordinates, or ``None``.

        Read straight off the GDAL handle rather than through :attr:`bounds`,
        because this runs from :meth:`_get_crs` during ``RasterBase.__init__``,
        before the Spatial collaborator exists.

        Returns:
            ``(min_x, min_y, max_x, max_y)``, or ``None`` when the raster carries
            no real geotransform (GDAL's identity default) and its extent
            therefore says nothing.
        """
        result: tuple[float, float, float, float] | None = None
        try:
            geotransform = self.raster.GetGeoTransform()
            columns, rows = self.raster.RasterXSize, self.raster.RasterYSize
        except (RuntimeError, AttributeError):
            geotransform = None
        # GDAL hands back the identity transform for a raster that has none; its
        # "extent" is then pixel indices, which must not veto anything.
        if geotransform and tuple(geotransform) != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            # All four corners: with a rotated geotransform (non-zero gt[2] /
            # gt[4]) the two diagonal corners do not bound the other two, and
            # under a symmetric rotation they coincide -- collapsing the extent
            # to a point, which silently disables the range check.
            corners = [(0, 0), (columns, 0), (0, rows), (columns, rows)]
            xs = [
                geotransform[0] + col * geotransform[1] + row * geotransform[2]
                for col, row in corners
            ]
            ys = [
                geotransform[3] + col * geotransform[4] + row * geotransform[5]
                for col, row in corners
            ]
            result = (min(xs), min(ys), max(xs), max(ys))
        return result

    def _get_epsg(self) -> int | None:
        """Concrete override of :meth:`RasterBase._get_epsg`.

        Defined directly on Dataset for the same reason as
        :meth:`_get_crs`.

        Returns `None` for a raster with no CRS, honouring the documented
        `int | None` contract of :attr:`epsg` — a missing georeference must not
        be reported as WGS 84.
        """
        return epsg_of_crs(self._get_crs())

    def zonal_stats(
        self,
        fc,
        *,
        stats=("mean",),
        method: str = "rasterize",
        band: int = 0,
    ):
        """Compute zonal statistics of this dataset over a polygon FeatureCollection.

        Thin forwarder to
        :func:`pyramids.dataset.ops._zonal.zonal_stats`; see that
        function for the full argument contract.

        Args:
            fc: A :class:`pyramids.feature.FeatureCollection` of
                polygons sharing this dataset's CRS.
            stats: Sequence of stat names (`"mean"`, `"sum"`,
                `"min"`, `"max"`, `"std"`, `"var"`,
                `"count"`).
            method: `"rasterize"` is the only supported value today;
                an area-weighted `"fractional"` method is planned.
            band: Zero-based band index.

        Returns:
            pandas.DataFrame: Indexed by `fc.index`; one column per stat.
        """
        return _zonal_stats(self, fc, stats=stats, method=method, band=band)

    def to_zarr(
        self,
        store,
        *,
        compute: bool = True,
        mode: str = "w",
        chunks=None,
        storage_options: dict | None = None,
        compressor="auto",
        overview_factors: list | None = None,
        overview_resampling: str = "average",
    ):
        """Serialise this Dataset to a Zarr store (parallel writes per chunk).

        Thin forwarder to
        :func:`pyramids.dataset.ops._zarr.write_dataset_to_zarr`; see
        that function for the full argument contract. Zarr is the
        only raster output format where pyramids can write in true
        parallel — each dask chunk becomes an independent Zarr chunk
        file. Requires the `[lazy]` optional extra.

        Args:
            store: Target store (path / fsspec URL / zarr.Store).
            compute: `True` writes immediately; `False` returns a
                :class:`dask.delayed.Delayed`.
            mode: Zarr open mode, usually `"w"` or `"a"`.
            chunks: Chunk spec forwarded to :meth:`read_array`.
                `None` defaults to `"auto"` via the zarr helper.
            storage_options: fsspec options for cloud stores.
            compressor: Zarr codec(s) for the `data` array. `"auto"` (default)
                keeps zarr's default codec; pass a zarr-v3 codec or list of them
                (e.g. `zarr.codecs.BloscCodec(cname="zstd")`) to override, or
                `None` for an uncompressed array.
            overview_factors: Optional downsample factors (e.g. `[2, 4, 8]`) to
                also write decimated multiscale pyramid levels as `data_<factor>`
                arrays plus a `multiscales` attribute. Requires `compute=True`.
                Read a level back with `Dataset.from_zarr(store, level=factor)`.
            overview_resampling: GDAL resampling for the pyramid levels
                (`"average"` default, `"nearest"`, `"bilinear"`, ...).

        Raises:
            OverviewTargetError: `overview_factors` was given and this dataset cannot
                hold overviews — a plain VRT whose description is not a path: an empty
                one, a blank one, or inline VRT XML. The levels are built through
                `create_overviews`, which refuses that shape, so the target is checked
                pre-flight and no store is written at all. The check runs *before* the
                `compute` one, so a call that is wrong in both ways reports this rather
                than the `ValueError` below — passing `compute=True` would still leave
                the dataset refused. Save it with `to_file(path)` and write the Zarr
                from the saved raster.
            ValueError: `overview_factors` was given with `compute=False`; the pyramid
                levels are written eagerly.
        """
        resolved_chunks = chunks if chunks is not None else "auto"
        return write_dataset_to_zarr(
            self,
            store,
            compute=compute,
            mode=mode,
            chunks=resolved_chunks,
            storage_options=storage_options,
            compressor=compressor,
            overview_factors=overview_factors,
            overview_resampling=overview_resampling,
        )

    @classmethod
    def from_zarr(
        cls,
        store,
        *,
        chunks=None,
        storage_options: dict | None = None,
        level: int = 1,
        data_name: str | None = None,
    ) -> Dataset:
        """Load a pyramids-written Zarr store into a new :class:`Dataset`.

        Thin forwarder to
        :func:`pyramids.dataset.ops._zarr.read_dataset_from_zarr`.

        Args:
            store: Input store (path / fsspec URL / zarr.Store).
            chunks: If non-None, the loaded Dataset is flagged as
                dask-backed so downstream `read_array` calls return
                lazy arrays.
            storage_options: fsspec options for cloud stores.
            level: Pyramid downsample factor to read (`1` = full resolution).
                Pass a factor written via `to_zarr(overview_factors=...)` to read
                that decimated overview level.
            data_name: Explicit name of the data array. ``None`` (default)
                auto-detects; pass an explicit name to read a specific variable
                from a foreign GeoZarr store whose auto-detect picks the wrong
                array.
        """
        return read_dataset_from_zarr(
            store,
            chunks=chunks,
            storage_options=storage_options,
            level=level,
            data_name=data_name,
        )

    def __str__(self) -> str:
        """Human-readable multi-line summary, or a `<Dataset: closed>` sentinel.

        `repr()` / `str()` run in debuggers, logging, and pytest introspection, so a
        closed dataset returns a sentinel rather than raising (a raising `__repr__`
        would mask the surrounding error). Reads that must fail loudly use
        `_require_open` instead.
        """
        message = "<Dataset: closed>"
        if self._raster is not None:
            message = f"""
            Top Left Corner: {self.top_left_corner}
            Cell size: {self.cell_size}
            Dimension: {self.rows} * {self.columns}
            EPSG: {self.epsg}
            Number of Bands: {self.band_count}
            Band names: {self.band_names}
            Band colors: {self.band_color}
            Band units: {self.band_units}
            Scale: {self.scale}
            Offset: {self.offset}
            Mask: {self.no_data_value[0]}
            Data type: {self.dtype[0]}
            File: {self.file_name}
        """
        return message

    def __repr__(self) -> str:
        """GDAL info string, or a `<Dataset: closed>` sentinel on a closed dataset.

        The info string's ``Files:`` section lists every source a VRT
        references, and for a mosaic built by
        :func:`pyramids.stac.build_vrt_from_stac` with a bearer signer those
        paths carry the live token — so the text goes through
        :func:`~pyramids.base.remote.redact_credentials` first. ``repr`` is
        called far more often than deliberately: pytest prints it for every
        operand of a failing assertion, ``logging.error("%r", ds)`` is idiomatic,
        and a notebook auto-displays it.
        """
        info = "<Dataset: closed>"
        if self._raster is not None:
            info = redact_credentials(str(gdal.Info(self.raster)))
        return info

    @property
    def access(self) -> str:
        """
        Access mode.

        Returns:
            str:
                The access mode of the dataset (read_only/write).
        """
        return str(super().access)

    @property
    def raster(self) -> gdal.Dataset:
        """Base GDAL Dataset (read-only)."""
        return super().raster

    @property
    def rows(self) -> int:
        """Number of rows in the raster array."""
        return int(self._rows)

    @property
    def columns(self) -> int:
        """Number of columns in the raster array."""
        return int(self._columns)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape (bands, rows, columns)."""
        return self.band_count, self.rows, self.columns

    @property
    def epsg(self) -> int | None:
        """EPSG number, or ``None``.

        ``None`` means either the raster has **no CRS at all** — pyramids does
        not assume WGS 84 for an unprojected grid (ARC-26) — or its CRS carries
        no EPSG authority code (a geostationary fixed-grid projection, say).
        Read :attr:`crs` to tell the two apart: it is empty in the first case and
        a WKT string in the second.
        """
        return self._epsg

    @epsg.setter
    def epsg(self, value: int):
        """EPSG number.

        Raises:
            ReadOnlyError: The dataset is opened read-only.
        """
        self._require_writable("set the EPSG code")
        sr = sr_from_epsg(value)
        self.raster.SetProjection(sr.ExportToWkt())
        self._update_inplace(self._raster)

    @property
    def crs(self) -> str:
        """Coordinate reference system.

        Returns:
            str:
                the coordinate reference system of the dataset.

        See Also:
            Dataset.set_crs : Set the Coordinate Reference System (CRS).
            Dataset.to_crs : Reproject the dataset to any projection.
            Dataset.epsg : epsg number of the dataset coordinate reference system.
        """
        return self._get_crs()

    @crs.setter
    def crs(self, value: str):
        """Coordinate reference system.

        Args:
            value (str):
                WellKnownText (WKT) string.

        Raises:
            ReadOnlyError: The dataset is opened read-only on-disk (setting the CRS
                would otherwise silently spill a PAM sidecar).

        See Also:
            - Dataset.set_crs: Set the Coordinate Reference System (CRS).
            - Dataset.to_crs: Reproject the dataset to any projection.
            - Dataset.epsg: EPSG number of the dataset coordinate reference system.
        """
        self.set_crs(value)

    @property
    def cell_size(self) -> float:
        """Cell size."""
        return float(self._cell_size)

    @property
    def band_count(self) -> int:
        """Number of bands in the raster."""
        return int(self._band_count)

    @property
    def band_names(self) -> list[str]:
        """Band names."""
        return self._get_band_names()

    @band_names.setter
    def band_names(self, name_list: list):
        """Band names setter.

        Raises:
            ReadOnlyError: The dataset is opened read-only on-disk (a bare
                `SetDescription` would otherwise silently spill a PAM sidecar).
        """
        self._require_writable("set band names")
        self.bands._set_band_names(name_list)

    @property
    def band_units(self) -> list[str]:
        """Facade — delegates to :attr:`Bands.band_units <pyramids.dataset.engines.Bands.band_units>`."""
        return self.bands.band_units

    @band_units.setter
    def band_units(self, value: list[str]):
        """Facade setter.

        Raises:
            ReadOnlyError: The dataset is opened read-only on-disk.
        """
        self.bands.band_units = value

    def convert_units(self, target: str, band: int | None = None) -> Dataset:
        """Convert band values to ``target`` units, returning a new Dataset.

        Unlike the :attr:`band_units` setter — which only relabels bands — this
        actually transforms the stored values using a small affine conversion table
        (see :func:`pyramids.dataset.ops.units.convert_array`) and records the new
        unit on the result. No-data cells are preserved unchanged. The output is a
        new in-memory ``float64`` Dataset; the source is left untouched.

        Args:
            target: Target unit label (e.g. ``"celsius"``, ``"hPa"``, ``"knots"``).
            band: Zero-based band index to convert. ``None`` (default) converts every
                band; bands already in ``target`` units are passed through unchanged.

        Returns:
            A new :class:`Dataset` with converted values and updated
            :attr:`band_units`.

        .. deprecated::
            Physical value-unit conversion (Kelvin/Celsius, m/s/knots, Pa/hPa,
            m/mm) is atmospheric/geophysical domain logic, not a generic GIS
            raster primitive, and will be **removed** from pyramids. Keep the
            unit *metadata* on :attr:`band_units` and perform the value
            conversion in the downstream science-domain consumer. Calling this
            method emits a :class:`DeprecationWarning`.

        Raises:
            ValueError: ``band`` is out of range, a converted band has no source unit
                set, or the ``(source, target)`` pair is unsupported.

        Examples:
            - Convert a Kelvin raster to Celsius and read the new values:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.array([[273.15, 283.15], [293.15, 303.15]]),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> ds.band_units = ["K"]
                >>> converted = ds.convert_units("celsius")
                >>> converted.read_array().tolist()
                [[0.0, 10.0], [20.0, 30.0]]
                >>> converted.band_units
                ['celsius']

                ```
            - An unsupported target raises a clear error:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.from_array(
                ...     np.array([[273.15]]),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> ds.band_units = ["K"]
                >>> try:
                ...     ds.convert_units("furlongs")
                ... except ValueError as exc:
                ...     print("No unit conversion" in str(exc))
                True

                ```
        """
        warnings.warn(
            "Dataset.convert_units is deprecated and will be removed: physical "
            "value-unit conversion (K/celsius, m s-1/knots, Pa/hPa, m/mm) is "
            "domain logic, not a GIS primitive. Keep unit metadata on band_units "
            "and convert values in the downstream science-domain consumer.",
            DeprecationWarning,
            stacklevel=2,
        )
        if band is not None and not 0 <= band < self.band_count:
            raise ValueError(
                f"band {band} is out of range for a {self.band_count}-band dataset."
            )

        band_indices = range(self.band_count) if band is None else [band]
        source_units = list(self.band_units)
        new_units = list(self.band_units)

        full = self.read_array()
        single_band = self.band_count == 1
        stack = full[np.newaxis, ...] if single_band else full
        # astype(copy=True by default) already returns a fresh writable array;
        # the trailing .copy() was a redundant second full-cube copy.
        out = stack.astype("float64")
        no_data = self.no_data_value

        for index in band_indices:
            layer = out[index]
            nodata_value = no_data[index]
            mask = layer == nodata_value if nodata_value is not None else None
            converted = convert_array(layer, source_units[index], target)
            if mask is not None:
                converted[mask] = nodata_value
            out[index] = converted
            new_units[index] = target

        result_array = out[0] if single_band else out
        # `Dataset.from_array`, not `self.from_array`: this method is inherited
        # by NetCDF / Container / Variable, whose override returns a *bandless*
        # Container -- so the `band_units` assignment below died with
        # `IndexError: index 0 is out of bounds for axis 0 with size 0`, three
        # frames from the cause. A unit conversion yields a plain raster in
        # every case, so the base constructor is the right one to name.
        result = Dataset.from_array(
            result_array,
            no_data_value=list(no_data),
            geo_ref=GeoReference(
                geo=self.geotransform, epsg=crs_spec(self.epsg, self.crs)
            ),
        )
        result.band_units = new_units
        return result

    @property
    def no_data_value(self) -> tuple:
        """Per-band nodata markers as an immutable tuple.

        Returns a `tuple` (not a `list`) to make the read-only
        contract explicit — assign through the setter to change
        values; mutating the returned object never propagates to
        the underlying state.
        """
        return tuple(self._no_data_value)

    @no_data_value.setter
    def no_data_value(self, value: list | tuple | np.ndarray | Number):
        """Set the no_data_value marker on every band.

        Args:
            value: Either a scalar (broadcast to all bands) or a
                sequence (`list`, `tuple`, or 1-D :class:`numpy.ndarray`)
                with `len == band_count` providing one value per band.
                A 0-D ndarray is treated as a scalar.

        Raises:
            ReadOnlyError: The dataset is opened read-only on-disk (a bare
                `SetNoDataValue` would otherwise silently mutate only the
                in-memory attribute, persisting nothing).
            ValueError: When `value` is a sequence whose length
                differs from `band_count`, or a multi-dimensional
                ndarray (only 0-D scalars and 1-D sequences are
                accepted).

        Notes:
            - The setter does not change the values of the cells to the new no_data_value, it only changes the
            `no_data_value` attribute.
            - Use this method to change the `no_data_value` attribute to match the value that is stored in the cells.
            - To change the values of the cells, to the new no_data_value, use the `change_no_data_value` method.

        See Also:
            - Dataset.change_no_data_value: Change the No Data Value.
        """
        self._require_writable("set the no-data value")
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = value.item()
            elif value.ndim == 1:
                value = value.tolist()
            else:
                raise ValueError(
                    f"no_data_value ndarray must be 0-D (scalar) or 1-D "
                    f"(per-band sequence); got ndim={value.ndim}"
                )
        if isinstance(value, (list, tuple)):
            if len(value) != self.band_count:
                raise ValueError(
                    f"no_data_value sequence length {len(value)} does "
                    f"not match band_count {self.band_count}"
                )
            for i, val in enumerate(value):
                self.bands._change_no_data_value_attr(i, val)
        else:
            for i in range(self.band_count):
                self.bands._change_no_data_value_attr(i, value)

    @property
    def meta_data(self):
        """Meta-data."""
        return super().meta_data

    @meta_data.setter
    def meta_data(self, value: dict[str, str]):
        """Meta-data.

        Raises:
            ReadOnlyError: The dataset is opened read-only.
        """
        self._require_writable("set metadata")
        for key, val in value.items():
            self._raster.SetMetadataItem(key, val)
        # The CF geographic inference reads this metadata (axis units), so the
        # memoised answer is stale once it changes -- and `_epsg` was memoised
        # from it during __init__, so re-derive that too. Dropping only the WKT
        # cache left `.crs` reporting WGS 84 while `.epsg` reported None, a
        # combination documented to mean "a CRS with no EPSG authority".
        self.__dict__.pop("_cf_crs_cache", None)
        self._epsg = self._get_epsg()

    def set_meta_data(
        self, value: dict[str, str] | list[str], domain: str = ""
    ) -> None:
        """Replace a *named* GDAL metadata domain.

        Writes ``value`` into ``domain`` with ``SetMetadata`` — a **replace**, not a
        merge (the replace semantics match the band-level
        :meth:`Bands.set_metadata <pyramids.dataset.engines.Bands.set_metadata>`;
        unlike it, this method **refuses the default domain** — see below). Assigning
        ``{}`` (or ``[]``) empties the domain's keys, though the domain name itself may
        still be listed by :attr:`meta_data_domains`.

        Most domains take a ``KEY=VALUE`` mapping; an ``xml:*`` domain instead takes a
        single-element ``list[str]`` of one XML document, mirroring what
        :meth:`get_meta_data` returns for it (passing a ``dict`` to an ``xml:*`` domain
        is a mistake — GDAL flattens it to ``["KEY=VALUE"]``).

        The **default** domain (``""``) is deliberately rejected: it holds
        GDAL/CF-managed keys — ``AREA_OR_POINT`` and the CF axis metadata that drives
        CRS inference — and a whole-domain replace would silently drop them (and the
        CRS/EPSG caches would then re-derive from the corrupted state). Use the
        :attr:`meta_data` setter for the default domain; it merges per key and
        refreshes those caches.

        Args:
            value: The metadata to write into ``domain`` — a ``dict[str, str]``
                mapping for a ``KEY=VALUE`` domain, or a single-element ``list[str]``
                for an ``xml:*`` domain.
            domain: The named GDAL metadata domain to write (for example
                ``"IMAGE_STRUCTURE"``, ``"RPC"``, or a custom domain). The empty
                default domain is not accepted.

        Raises:
            ValueError: ``domain`` is the empty default domain — use the
                :attr:`meta_data` setter instead.
            ReadOnlyError: The dataset is a read-only on-disk file.
        """
        if not domain:
            raise ValueError(
                "set_meta_data writes named domains only; use the `meta_data` setter "
                "for the default domain (it merges per key and refreshes CRS caches)."
            )
        self._require_writable("set metadata")
        self._raster.SetMetadata(value, domain)

    def open_subdataset(self, key: int | str) -> Dataset:
        """Open one of this container's subdatasets, carrying its open context.

        Resolves ``key`` against :attr:`subdatasets` and reopens the chosen nested
        raster with this dataset's access mode, GDAL environment, and open options.

        The result is a **base** :class:`Dataset`: a subdataset connection string is
        a classic-mode raster reference, so it is opened as an ordinary raster
        (unlike :meth:`SubDataset.open`, this carries the parent's access mode, GDAL
        env, and open options). The parent's open options are reapplied verbatim to
        the child open. If the parent is open in update mode the child is opened in
        update mode too; not every driver supports updating a subdataset connection
        string, so a write-mode open can fail for some containers. For a ``NetCDF``
        container, use
        :meth:`~pyramids.netcdf.netcdf.NetCDF.get_variable` / ``NetCDF.variables``
        instead when you want the multidimensional, ``NetCDF``-preserving view of a
        variable — those handle the multidim open a raw subdataset string cannot.

        Args:
            key: An index into :attr:`subdatasets` (0-based; negative indices count
                from the end, per Python list semantics), or a subdataset's full
                ``name`` (its GDAL connection string).

        Returns:
            Dataset: The opened subdataset as a base ``Dataset``.

        Raises:
            TypeError: ``key`` is neither an ``int`` index nor a ``str`` name.
            IndexError: ``key`` is an out-of-range index.
            ValueError: ``key`` is a name that is not among this container's
                subdatasets.
        """
        subs = self.subdatasets
        if isinstance(key, bool):
            raise TypeError(
                f"key must be an int index or a str name, not bool: {key!r}"
            )
        if isinstance(key, int):
            name = subs[key].name  # negative indices follow Python list semantics
        elif isinstance(key, str):
            if key not in {sub.name for sub in subs}:
                # Connection strings can embed credentials (signed URLs, SAS tokens);
                # redact before echoing them in the error.
                available = [redact_credentials(sub.name) for sub in subs]
                raise ValueError(
                    f"{redact_credentials(key)!r} is not a subdataset of this "
                    f"dataset; available: {available}"
                )
            name = key
        else:
            raise TypeError(
                f"key must be an int index or a str name, got {type(key).__name__}"
            )
        # Open the classic-mode subdataset string as a base Dataset. read_file both
        # installs the captured GDAL env around the open (so remote credentials apply)
        # and re-attaches it to the result, so no separate context/attach is needed.
        # warn_on_container=False: the caller deliberately drilled into a subdataset, so
        # a container warning here (if the target is itself a nested container) is noise.
        return Dataset.read_file(
            name,
            read_only=self.access == "read_only",
            gdal_env=self._gdal_env or None,
            open_options=list(self._open_options) or None,
            warn_on_container=False,
        )

    @property
    def band_meta_data(self) -> list[dict[str, str]]:
        """Per-band metadata, one mapping per band, in band order.

        The per-band sibling of :attr:`meta_data`. Facade — delegates to
        :attr:`Bands.metadata <pyramids.dataset.engines.Bands.metadata>`; see it for
        the empty-band and default-domain conventions.

        Returns:
            list[dict[str, str]]: One mapping per band (0-based, band order); an empty
            ``dict`` for a band with no metadata.
        """
        return self.bands.metadata

    @band_meta_data.setter
    def band_meta_data(self, value: list[dict[str, str]]) -> None:
        """Replace each band's metadata (one mapping per band).

        Facade setter — delegates to
        :attr:`Bands.metadata <pyramids.dataset.engines.Bands.metadata>`, which
        replaces (does not merge) each band's default-domain metadata.

        Raises:
            ReadOnlyError: The dataset is a read-only on-disk file.
            ValueError: ``value`` does not carry exactly one mapping per band.
        """
        self.bands.metadata = value

    @property
    def file_name(self) -> str:
        """File name."""
        return super().file_name

    @property
    def driver_type(self):
        """Driver Type."""
        return super().driver_type

    @property
    def scale(self) -> list[float]:
        """Facade — delegates to :attr:`Bands.scale <pyramids.dataset.engines.Bands.scale>`.

        The scale converts the pixel values to the real-world values.
        """
        return self.bands.scale

    @scale.setter
    def scale(self, value: list[float]):
        """Facade setter.

        Raises:
            ReadOnlyError: The dataset is opened read-only on-disk.
        """
        self.bands.scale = value

    @property
    def offset(self):
        """Facade — delegates to :attr:`Bands.offset <pyramids.dataset.engines.Bands.offset>`.

        The offset converts the pixel values to the real-world values.
        """
        return self.bands.offset

    @offset.setter
    def offset(self, value: list[float]):
        """Facade setter.

        Raises:
            ReadOnlyError: The dataset is opened read-only on-disk.
        """
        self.bands.offset = value

    @property
    def top_left_corner(self):
        """Top left corner coordinates.

        See Also:
            - Dataset.geotransform: Dataset geotransform.
        """
        return super().top_left_corner

    @property
    def bounds(self) -> GeoDataFrame:
        """Bounds - the bbox as a geodataframe with a polygon geometry.

        See Also:
            - Dataset.bbox: Dataset bounding box.
        """
        return self._calculate_bounds()

    @property
    def bbox(self) -> list:
        """Bound box [xmin, ymin, xmax, ymax].

        See Also:
            - Dataset.bounds: Dataset bounding polygon.
        """
        return self._calculate_bbox()

    def to_stac_item(
        self,
        item_id: str,
        *,
        asset_href: str,
        datetime=None,
        start_datetime=None,
        end_datetime=None,
        asset_key: str = "data",
        asset_media_type: str | None = None,
        with_proj: bool = True,
        with_raster: bool = True,
        precision: int = 6,
    ) -> dict:
        """Describe this raster as a STAC Item dict (proj + raster extensions).

        Thin forwarder to :func:`pyramids.dataset._stac.to_stac_item` — the
        inverse of :meth:`DatasetCollection.from_stac`. Returns a plain
        STAC-JSON dict (pystac not required); the footprint is this dataset's
        bounding rectangle reprojected to EPSG:4326.

        Args:
            item_id: The STAC Item id.
            asset_href: Href to record for the single data asset.
            datetime: Item datetime (`datetime.datetime` or RFC 3339 string).
                `None` with no range defaults to the current UTC time; `None`
                with `start_datetime`/`end_datetime` writes a null `datetime`
                plus the range (the STAC-valid null-datetime form).
            start_datetime: Optional range start, written to
                `properties.start_datetime`.
            end_datetime: Optional range end, written to
                `properties.end_datetime`.
            asset_key: Key for the data asset (default `"data"`).
            asset_media_type: Optional media type for the asset.
            with_proj: Populate the `proj` extension from the grid.
            with_raster: Populate `raster:bands` (data_type + nodata).
            precision: Decimal places for the reprojected footprint.

        Returns:
            dict: The STAC Item (a GeoJSON Feature).
        """
        # Imported here to avoid the dataset <-> stac import cycle at load time.
        from pyramids.dataset._stac import to_stac_item

        return to_stac_item(
            self,
            item_id,
            asset_href=asset_href,
            datetime=datetime,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            asset_key=asset_key,
            asset_media_type=asset_media_type,
            with_proj=with_proj,
            with_raster=with_raster,
            precision=precision,
        )

    @property
    def total_bounds(self) -> np.typing.NDArray:
        """Bounding box `[minx, miny, maxx, maxy]` as a NumPy array.

        introduced this property so that `Dataset` and
        :class:`pyramids.feature.FeatureCollection` expose the same
        shape (`GeoDataFrame.total_bounds` is the geopandas name
        for exactly this array), letting both classes satisfy the
        :class:`pyramids.base.protocols.SpatialObject` protocol.
        """
        return np.asarray(self._calculate_bbox())

    @property
    def lon(self) -> np.typing.NDArray:
        """Longitude / x cell-centre coordinates.

        Uses the geotransform's pixel width (``geotransform[1]``) so the axis is
        correct even when cells are not square (pixel width != pixel height). Reads the
        cached ``_geotransform`` (like :attr:`top_left_corner`) rather than the
        ``geotransform`` property, so subclasses that derive ``geotransform`` from
        ``lon``/``lat`` (e.g. :class:`~pyramids.netcdf.NetCDF`) do not recurse.

        Examples:
            - Read the column-centre longitudes of a small raster:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.zeros((2, 3)),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.5, epsg=4326),
                ... )
                >>> ds.lon.tolist()
                [0.25, 0.75, 1.25]

                ```

        See Also:
            - Dataset.x: Dataset x coordinates.
            - Dataset.lat: Dataset latitude.
        """
        pixel_width = self._geotransform[1]
        x_coords = self.get_x_lon_dimension_array(
            self.top_left_corner[0], pixel_width, self.columns
        )
        return x_coords

    @property
    def lat(self) -> np.typing.NDArray:
        """Latitude / y cell-centre coordinates.

        Uses the geotransform's pixel height (``abs(geotransform[5])``) rather than
        :attr:`cell_size` (which only tracks pixel width), so the axis is correct for
        non-square cells. Reads the cached ``_geotransform`` (like
        :attr:`top_left_corner`) rather than the ``geotransform`` property, so
        subclasses that derive ``geotransform`` from ``lon``/``lat`` (e.g.
        :class:`~pyramids.netcdf.NetCDF`) do not recurse.

        Examples:
            - Row-centre latitudes decrease from north to south:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.zeros((2, 3)),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.5, epsg=4326),
                ... )
                >>> ds.lat.tolist()
                [-0.25, -0.75]

                ```
            - With non-square cells the latitude axis uses the pixel height, not the
              pixel width:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.from_array(
                ...     np.zeros((2, 3)),
                ...     geo_ref=GeoReference(geo=(10.0, 2.0, 0.0, 50.0, 0.0, -1.0), epsg=4326),
                ... )
                >>> ds.lat.tolist()
                [49.5, 48.5]

                ```

        See Also:
            - Dataset.x: Dataset x coordinates.
            - Dataset.y: Dataset y coordinates.
            - Dataset.lon: Dataset longitude.
        """
        pixel_height = abs(self._geotransform[5])
        y_coords = self.get_y_lat_dimension_array(
            self.top_left_corner[1], pixel_height, self.rows
        )
        return y_coords

    @property
    def x(self) -> np.typing.NDArray:
        """X cell-centre coordinates (alias of :attr:`lon`).

        Examples:
            - x mirrors lon for the same raster:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.zeros((2, 3)),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.5, epsg=4326),
                ... )
                >>> ds.x.tolist()
                [0.25, 0.75, 1.25]

                ```

        See Also:
            - Dataset.lon: the longitude axis this property aliases.
            - Dataset.y: Dataset y coordinates.
        """
        return self.lon

    @property
    def y(self) -> np.typing.NDArray:
        """Y cell-centre coordinates (alias of :attr:`lat`).

        Examples:
            - y mirrors lat for the same raster:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.zeros((2, 3)),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.5, epsg=4326),
                ... )
                >>> ds.y.tolist()
                [-0.25, -0.75]

                ```

        See Also:
            - Dataset.lat: the latitude axis this property aliases.
            - Dataset.x: Dataset x coordinates.
        """
        return self.lat

    @property
    def gdal_dtype(self):
        """Data Type."""
        return [
            self.raster.GetRasterBand(i).DataType for i in range(1, self.band_count + 1)
        ]

    @property
    def numpy_dtype(self) -> list[type]:
        """List of the numpy data Type of each band, the data type is a numpy function."""
        return [
            DTYPE_CONVERSION_DF.loc[DTYPE_CONVERSION_DF["gdal"] == i, "numpy"].values[0]
            for i in self.gdal_dtype
        ]

    @property
    def dtype(self) -> list[str]:
        """List of the data Type of each band as strings."""
        return [
            DTYPE_CONVERSION_DF.loc[DTYPE_CONVERSION_DF["gdal"] == i, "name"].values[0]
            for i in self.gdal_dtype
        ]

    @classmethod
    def read_file(
        cls,
        path: str | Path,
        read_only=True,
        file_i: int = 0,
        *,
        vsi: str | None = None,
        gdal_env: dict[str, str] | None = None,
        open_options: dict[str, str] | list[str] | tuple[str, ...] | None = None,
        warn_on_container: bool = True,
    ) -> Dataset:
        """Open a raster from a path, URL, or archive member.

        Plain local paths, ``/vsi*`` paths, and URL schemes
        (``http(s)://``, ``s3://``, ``gs://``, ``az://``, ``abfs://`` / ``abfss://``,
        ``file://``) are all accepted — URLs are transparently rewritten to
        GDAL's virtual filesystem (GDAL fetches via HTTP range requests for
        ``http(s)``). Compressed archives are detected from the extension; pass
        ``vsi=`` to be explicit about it (e.g. an archive with an unusual
        extension, or to open a specific member by index).

        Args:
            path (str | Path):
                Path or URL of the file to open.
            read_only (bool):
                File mode; set to ``False`` to open in update mode.
            file_i (int):
                Which member to open when ``path`` is (or is forced to be) a
                multi-file archive. Default ``0``.
            vsi (str | None):
                Treat ``path`` as an archive of this kind and open member
                ``file_i`` from inside it: ``"zip"``, ``"tar"`` (also
                ``"tar.gz"`` / ``"tgz"``), ``"gzip"`` (also ``"gz"``), or
                ``"auto"`` (infer from the extension). Default ``None`` —
                ``path`` is opened directly / extension-sniffed as before.
                Works for archives reachable locally or over the network
                (``/vsizip//vsicurl/…`` is built automatically) **provided the
                file name carries a recognised archive extension** — GDAL's
                archive handlers key off the extension, so an extension-less
                download URL must first be fetched and saved with a ``.zip``
                name (or written to ``/vsimem/<name>.zip`` via
                :func:`osgeo.gdal.FileFromMemBuffer`).
            gdal_env (dict[str, str] | None):
                Optional GDAL config (cloud credentials, HTTP knobs) installed
                for this open **and captured on the returned dataset**, so it is
                re-installed around its reads. Needed by the read paths that
                open the file again instead of reusing this handle:
                ``threadsafe=True`` per-thread handles, lazy ``chunks=`` reads
                inside dask tasks, and unpickling on a worker.
                :func:`pyramids.stac.load_asset` passes a signer's
                ``gdal_env()`` here. It does **not** reach a VRT's source opens
                — GDAL ignores the thread-local config there, so
                :func:`pyramids.stac.build_vrt_from_stac` puts those credentials
                in the source path instead. Default ``None`` — no extra config,
                nothing captured.
            open_options:
                GDAL open options as a mapping
                (``{"GEOREF_SOURCES": "INTERNAL"}``) or GDAL's native
                ``["KEY=VALUE"]`` list. Forwarded to the driver and captured on
                the returned :class:`Dataset`, so the paths that reopen the file
                (``threadsafe=True`` handles, lazy ``chunks=`` reads, unpickle on
                a worker) reopen with the same options. Default ``None``.
            warn_on_container:
                When the path opens to a *container* — a raster with no bands of
                its own whose payload is a set of nested subdatasets (NetCDF/HDF/
                Zarr, GRIB, WMS/WMTS, a Sentinel product) — emit a
                :class:`~pyramids.errors.ContainerRasterWarning` naming the
                subdatasets, instead of silently returning a 0-band dataset. Use
                :attr:`subdatasets` to list them and :meth:`open_subdataset` to open
                one. Set ``False`` to open a container quietly (callers that open
                containers on purpose). Default ``True``.

        Returns:
            Dataset:
                Opened dataset instance.

        See Also:
            - :meth:`read_array`: read the values stored in a dataset band.
            - :meth:`from_bytes`: open a raster held in memory.
            - :attr:`gdal_env`: the config captured by ``gdal_env=``.
            - :meth:`pyramids.dataset.DatasetCollection.from_archive`: open
              *every* member of an archive as a temporal stack.
        """
        # Normalize once here so the value captured on the instance below is the
        # KEY=VALUE list form (a raw dict would lose its values when the base
        # __init__ tuple-ifies it). _io.read_file re-normalizes idempotently for
        # its own direct callers — the double pass is intentional and harmless.
        options = _io.normalize_open_options(open_options)
        with cloud_config_from_env(gdal_env, path=str(path)):
            src = _io.read_file(
                path,
                read_only=read_only,
                file_i=file_i,
                vsi=vsi,
                open_options=options,
            )
        dataset = cls(
            src,
            access="read_only" if read_only else "write",
            gdal_env=gdal_env,
            open_options=options,
        )
        if warn_on_container and not dataset.band_count:
            subdatasets = dataset.subdatasets
            if subdatasets:
                count = len(subdatasets)
                # Cap the preview so a many-variable container (tens of NetCDF/HDF
                # variables) does not produce an unbounded warning string.
                shown = [redact_credentials(sub.name) for sub in subdatasets[:10]]
                if count > len(shown):
                    shown.append(f"… and {count - 10} more")
                warnings.warn(
                    f"{redact_credentials(str(path))!r} is a container raster with no "
                    f"bands of its own; it has {count} subdataset(s). Use "
                    f".subdatasets to list them and .open_subdataset(<index or name>) "
                    f"to open one. Available: {shown}",
                    ContainerRasterWarning,
                    stacklevel=2,
                )
        return dataset

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        suffix: str = ".tif",
        name: str | None = None,
        read_only: bool = True,
    ) -> Dataset:
        """Open a raster held in memory as a byte string.

        Writes ``data`` to a temporary GDAL ``/vsimem/`` path and opens
        it — no on-disk temp file needed. Useful for HTTP response
        bodies (``requests.get(url).content``), object-store
        ``get_object`` payloads, database blobs, and test fixtures.

        This is **not** a URL helper. Reading from a URL is already
        supported by :meth:`read_file`, which rewrites ``http(s)://``,
        ``s3://``, ``gs://``, ``az://``, ``abfs://`` / ``abfss://`` and ``file://``
        to GDAL ``/vsi*`` paths. Use ``from_bytes`` only when you
        already hold the bytes.

        The ``/vsimem/`` entry is removed automatically when the
        returned :class:`Dataset` is garbage-collected
        (:func:`weakref.finalize`); :meth:`close` does not need to be
        called for cleanup. Note that an in-memory dataset is **not
        picklable** — :meth:`__reduce__` raises ``TypeError`` for
        ``/vsimem/`` paths; call :meth:`to_file` first to anchor it to
        disk before sending it to another process.

        Args:
            data: Raw bytes of a raster (GeoTIFF, ASCII grid, ...). For
                NetCDF bytes use :meth:`pyramids.netcdf.NetCDF.from_bytes`.
            suffix: Extension hint for GDAL's driver detection. Needed
                only for headerless formats (e.g. ESRI ASCII grid:
                ``suffix=".asc"``); GDAL sniffs anything with a magic
                header regardless. Defaults to ``".tif"``.
            name: Optional label recorded as the dataset's
                :attr:`file_name` (cosmetic only — it is still an
                in-memory dataset). Defaults to ``None``.
            read_only: Open the dataset read-only. Defaults to ``True``.

        Returns:
            Dataset: The opened in-memory dataset.

        Raises:
            TypeError: ``data`` is not a bytes-like object.
            ValueError: GDAL could not open the bytes (corrupt /
                truncated payload, or a headerless format without a
                ``suffix`` hint).

        Examples:
            - Open the bytes of a downloaded GeoTIFF and inspect it (the
              bytes here come from a file, but they could just as well be
              ``requests.get(url).content``):
                ```python
                >>> from pathlib import Path
                >>> from pyramids.dataset import Dataset
                >>> data = Path("tests/data/acc4000.tif").read_bytes()
                >>> ds = Dataset.from_bytes(data, name="downloaded-scene")
                >>> ds.band_count
                1
                >>> ds.shape
                (1, 13, 14)
                >>> ds.epsg
                32618
                >>> ds.file_name
                'downloaded-scene'
                >>> ds.close()

                ```
            - The bytes path yields the same data as opening the file directly:
                ```python
                >>> from pathlib import Path
                >>> from pyramids.dataset import Dataset
                >>> data = Path("tests/data/acc4000.tif").read_bytes()
                >>> from_bytes = Dataset.from_bytes(data)
                >>> from_file = Dataset.read_file("tests/data/acc4000.tif")
                >>> from_bytes.shape == from_file.shape
                True
                >>> from_bytes.epsg == from_file.epsg
                True

                ```
            - An in-memory dataset cannot be pickled — anchor it to disk first:
                ```python
                >>> import pickle
                >>> from pathlib import Path
                >>> from pyramids.dataset import Dataset
                >>> data = Path("tests/data/acc4000.tif").read_bytes()
                >>> try:
                ...     pickle.dumps(Dataset.from_bytes(data))
                ... except TypeError as exc:
                ...     print("to_file" in str(exc))
                True

                ```

        See Also:
            - :meth:`read_file`: open a raster from a path or URL.
            - :meth:`to_file`: write an in-memory dataset to disk.
            - :meth:`pyramids.netcdf.NetCDF.from_bytes`: the NetCDF variant.
        """
        src, vsi_path = _io.bytes_to_gdal(data, suffix=suffix, read_only=read_only)
        try:
            obj = cls(src, access="read_only" if read_only else "write")
        except Exception as e:
            src = None
            _io.silent_unlink(vsi_path)
            raise ValueError(
                "could not open the supplied bytes as a raster dataset "
                f"(the data may be corrupt or truncated): {e}"
            ) from e
        obj._vsimem_path = vsi_path
        weakref.finalize(obj, _io.silent_unlink, vsi_path)
        if name is not None:
            obj._file_name = str(name)
        return obj

    @classmethod
    def from_wcs(
        cls,
        endpoint: str,
        *,
        coverage: str,
        bbox: tuple[float, float, float, float],
        crs: str = _DEFAULT_CRS,
        output_crs: str | None = None,
        resolution: float | tuple[float, float] | None = None,
        version: str | None = None,
        coverage_crs: str | None = None,
        wcs_format: str | None = None,
        output: str | Path | None = None,
        resample: str = "nearest",
        auth: tuple[str, str] | None = None,
        timeout: float = 60.0,
        extra_params: dict[str, str] | None = None,
        direct: bool = False,
        subset_axes: tuple[str, str] | None = None,
    ) -> Dataset:
        """Read a coverage subset from an OGC Web Coverage Service (WCS).

        Fetches a windowed subset of a coverage from a WCS server and returns it
        as a :class:`Dataset`. The transport is GDAL's native WCS driver, so the
        WCS ``1.0.0`` vs ``2.0.x`` dialect fork — ``bbox`` + ``resx/resy`` versus
        named-axis ``subsets`` + ``scaling`` — is handled inside GDAL; the caller
        always supplies a single lon/lat ``bbox`` (plus optional ``resolution``
        and ``output_crs``).

        Two things GDAL does **not** do for every server, which this method adds:

        * **CRS shim.** Some servers advertise a coverage CRS under an authority
          code absent from the local PROJ database (notably ISRIC SoilGrids'
          ``EPSG:152160``, a custom Interrupted Goode Homolosine). GDAL then opens
          the coverage without a spatial reference and cannot place the request
          window. Pass ``coverage_crs`` with the coverage's real CRS and it is
          attached client-side.
        * **bbox reprojection.** ``bbox`` is given in ``crs`` (lon/lat by
          default) and transformed into the coverage's native CRS with ``pyproj``
          before the request, so subsetting lands on the correct pixels even when
          the server only honours its native CRS.

        For a **``GetCoverage``-only endpoint** — a "WCS shim" that returns
        ``502``/``400`` for ``GetCapabilities``/``DescribeCoverage`` but serves
        ``GetCoverage`` (e.g. Copernicus EDO/GDO) — pass ``direct=True``. That skips
        both discovery steps and issues a KVP ``GetCoverage`` built straight from
        ``coverage`` / ``crs`` / ``bbox`` / ``wcs_format`` / ``extra_params``, so the
        caller owns correctness (no capabilities check). For WCS ``2.0.x`` the
        ``SUBSET`` axis labels default to ``("Long", "Lat")`` for a geographic
        ``crs`` — override with ``subset_axes`` if the server names its axes
        differently.

        A non-conformant shim may also reject the spec KVP spellings themselves: the
        Copernicus EDO/GDO MapServer ``500``s on the uppercase ``COVERAGEID`` key and
        on ``SUBSETTINGCRS=`` (it wants a lowercase ``coverageID`` and the WCS-1.x
        ``CRS=``). In direct mode ``extra_params`` can override a built-in KVP by key,
        so pass ``extra_params={"coverageID": <id>, "CRS": <crs>}`` to hand such a
        server its exact spelling — the override replaces the built-in rather than
        duplicating it.

        Args:
            endpoint: The WCS service URL, including any server-specific query
                prefix (e.g. ``"https://maps.isric.org/mapserv?map=/map/nitrogen.map"``).
                Catalog / coverage-name routing belongs in the calling layer, not
                here.
            coverage: The coverage identifier as advertised by
                ``GetCapabilities`` (e.g. ``"nitrogen_0-5cm_mean"``). A value the
                server does not advertise raises :class:`ValueError`.
            bbox: ``(minx, miny, maxx, maxy)`` in ``crs`` order (lon/lat for the
                default ``"EPSG:4326"``).
            crs: CRS of ``bbox``. Defaults to ``"EPSG:4326"``.
            output_crs: Optional CRS to reproject the result into (any form
                :meth:`to_crs` accepts). ``None`` (default) keeps the coverage's
                native CRS.
            resolution: Output pixel size in the units of ``output_crs`` (or the
                native CRS when ``output_crs`` is ``None``). A scalar gives square
                pixels; an ``(x_res, y_res)`` pair gives non-square pixels.
                ``None`` (default) keeps the coverage's native resolution.
            version: Force a WCS protocol version (``"1.0.0"``, ``"2.0.1"``, …).
                ``None`` (default) lets GDAL negotiate from the server's
                capabilities. Note that some MapServer builds silently downgrade a
                requested ``2.0.x`` to ``1.0.0``.
            coverage_crs: The coverage's CRS, used only when the server's
                advertised CRS does not resolve in PROJ (see the CRS-shim note).
                Any proj4 / WKT / authority string ``pyproj`` understands.
            wcs_format: Optional GDAL ``PreferredFormat`` for the ``GetCoverage``
                response (e.g. ``"GEOTIFF_INT16"``). ``None`` lets GDAL pick from
                the coverage's advertised formats.
            output: Optional path to also write the result to as a GeoTIFF. The
                method still returns the :class:`Dataset`.
            resample: Resampling method for the ``output_crs`` / ``resolution``
                warp. Defaults to ``"nearest"``.
            auth: Optional ``(username, password)`` for Basic-authed services.
            timeout: HTTP timeout in seconds for the metadata / coverage
                requests. Defaults to ``60.0``.
            extra_params: Optional extra ``GetCoverage`` query parameters folded
                into the request (a workaround hook for server quirks). In direct
                mode a key that matches a built-in KVP (case-insensitively, with the
                cross-version pairs ``CRS``/``SUBSETTINGCRS`` and
                ``COVERAGE``/``COVERAGEID`` each treated as one) *overrides* it with
                the given spelling and value — e.g. ``{"coverageID": "spaST"}`` sends
                a lowercase key, ``{"CRS": "EPSG:4326"}`` sends the WCS-1.x CRS token
                instead of ``SUBSETTINGCRS``. Non-matching keys are appended in caller
                order (e.g. a ``TIME`` axis). The fixed protocol keys ``SERVICE`` /
                ``VERSION`` / ``REQUEST`` / ``SUBSET`` cannot be overridden and raise
                :class:`ValueError`; because ``SUBSET`` is locked, an additional
                WCS-2.0 ``SUBSET`` axis (e.g. a temporal subset) cannot be added in
                direct mode — use discovery mode for that. Two keys targeting the
                same built-in parameter (e.g. both ``CRS`` and ``SUBSETTINGCRS``) also
                raise.
            direct: When ``True``, skip ``GetCapabilities``/``DescribeCoverage`` and
                issue a KVP ``GetCoverage`` directly — for shim servers that only
                implement ``GetCoverage``. Defaults to ``False`` (full handshake).
            subset_axes: Direct mode, WCS ``2.0.x`` only — the ``(x, y)`` ``SUBSET``
                axis labels. ``None`` (default) derives them from ``crs``
                (``("Long", "Lat")`` for geographic, ``("X", "Y")`` otherwise). These
                defaults are a best-effort guess — direct mode skips the
                ``DescribeCoverage`` that would reveal the coverage's real (case-
                sensitive) axis labels — so MapServer-family shims often need
                ``subset_axes=("x", "y")`` or the server's exact axis names.

        Returns:
            Dataset: The fetched coverage subset.

        Raises:
            ValueError: ``bbox`` is malformed, ``coverage`` is not advertised
                (discovery mode), ``coverage_crs`` cannot be interpreted, the
                requested window exceeds the pixel ceiling
                (:data:`~pyramids.base._coverage.MAX_PX`; a native-resolution read
                over a wide ``bbox`` — pass a coarser ``resolution`` or a smaller
                ``bbox`` to bound it), or (direct mode) the WCS version is
                unsupported, ``1.0.0`` lacks a ``resolution``, or an ``extra_params``
                key targets a locked protocol parameter.
            pyramids.errors.WCSError: The server could not be reached or returned
                an error / a non-raster (``<ows:ExceptionReport>``) body.

        Examples:
            Read a Netherlands subset of SoilGrids nitrogen (its native CRS needs
            the ``coverage_crs`` shim):

            ```python
            >>> ds = Dataset.from_wcs(  # doctest: +SKIP
            ...     "https://maps.isric.org/mapserv?map=/map/nitrogen.map",
            ...     coverage="nitrogen_0-5cm_mean",
            ...     bbox=(5.0, 51.0, 6.0, 52.0),
            ...     coverage_crs="+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs",
            ... )

            ```

            Direct mode for a ``GetCoverage``-only endpoint (Copernicus EDO/GDO),
            whose ``GetCapabilities``/``DescribeCoverage`` return ``502``/``400``.
            EDO also rejects the spec KVP spellings, so override the coverage key and
            CRS token via ``extra_params`` to send the lowercase ``coverageID`` and
            the WCS-1.x ``CRS=`` it accepts:

            ```python
            >>> ds = Dataset.from_wcs(  # doctest: +SKIP
            ...     "https://drought.emergency.copernicus.eu/api/wcs?map=DO_WCS",
            ...     coverage="spaST",
            ...     bbox=(10.0, 45.0, 15.0, 48.0),
            ...     crs="EPSG:4326",
            ...     version="2.0.0",
            ...     wcs_format="GEOTIFF",
            ...     direct=True,
            ...     extra_params={
            ...         "coverageID": "spaST",
            ...         "CRS": "EPSG:4326",
            ...         "TIME": "2023-06-01",
            ...         "SELECTED_TIMESCALE": "01",
            ...     },
            ... )

            ```

        See Also:
            - :meth:`read_file`: open a raster from a path or URL.
            - :meth:`from_bytes`: open a raster already held in memory.
        """
        return _from_wcs(
            cls,
            endpoint,
            coverage=coverage,
            bbox=bbox,
            crs=crs,
            output_crs=output_crs,
            resolution=resolution,
            version=version,
            coverage_crs=coverage_crs,
            wcs_format=wcs_format,
            output=output,
            resample=resample,
            auth=auth,
            timeout=timeout,
            extra_params=extra_params,
            direct=direct,
            subset_axes=subset_axes,
        )

    @classmethod
    def from_wms(
        cls,
        endpoint: str,
        *,
        layers: str | list[str] | tuple[str, ...],
        bbox: tuple[float, float, float, float],
        crs: str = _DEFAULT_CRS,
        size: tuple[int, int] | None = None,
        resolution: float | tuple[float, float] | None = None,
        image_format: str = "image/png",
        version: str = "1.3.0",
        bands: int = 3,
        output_crs: str | None = None,
        output: str | Path | None = None,
        resample: str = "nearest",
        auth: tuple[str, str] | None = None,
        timeout: float = 60.0,
    ) -> Dataset:
        """Render a WMS ``GetMap`` window into a :class:`Dataset`.

        Fetches a server-rendered map image for ``bbox`` from an OGC Web Map
        Service via GDAL's native WMS driver, and returns it as a georeferenced
        raster. Because WMS renders in the requested ``crs``, the ``bbox`` is the
        request window directly — no client-side reprojection is needed.

        The result is **rendered imagery** (RGB / RGBA pixels), not data values: a
        WMS styles the data server-side. Use :meth:`from_wcs` /
        :meth:`from_ogc_coverages` when you need the underlying coverage values.

        Args:
            endpoint: The WMS base URL, ending with ``?`` or ``&`` so GDAL can
                append the ``GetMap`` query (e.g.
                ``"https://ows.terrestris.de/osm/service?"``). Layer catalogs and
                auth routing belong in the calling layer, not here.
            layers: One layer name, or several to composite, as advertised by the
                service ``GetCapabilities`` (joined with commas for the request).
            bbox: ``(minx, miny, maxx, maxy)`` in ``crs`` order (lon/lat for the
                default ``"EPSG:4326"``).
            crs: CRS of ``bbox`` and of the rendered request. Defaults to
                ``"EPSG:4326"`` (GDAL handles the WMS 1.3.0 lat/lon axis order).
            size: Output image size ``(width, height)`` in pixels. Mutually
                exclusive with ``resolution``; exactly one is required.
            resolution: Output pixel size in ``crs`` units — a scalar (square) or
                ``(x_res, y_res)`` pair — divided into the bbox extent to size the
                image. Mutually exclusive with ``size``.
            image_format: WMS ``FORMAT`` MIME type. Defaults to ``"image/png"``.
            version: WMS protocol version. Defaults to ``"1.3.0"``.
            bands: Number of bands to request (``3`` RGB, ``4`` RGBA). Defaults to
                ``3``.
            output_crs: Optional CRS to reproject the result into (any form
                :meth:`to_crs` accepts). ``None`` keeps ``crs``.
            output: Optional path to also write the result to as a GeoTIFF.
            resample: Resampling method for the ``output_crs`` warp. Defaults to
                ``"nearest"``.
            auth: Optional ``(username, password)`` for Basic-authed services.
            timeout: HTTP timeout in seconds. Defaults to ``60.0``.

        Returns:
            Dataset: The rendered map window.

        Raises:
            ValueError: ``bbox`` is malformed, ``layers`` is empty, or ``size`` /
                ``resolution`` was not given exactly once.
            pyramids.errors.WMSError: The server could not be reached or returned a
                non-raster body.

        Examples:
            Render a small OSM window as a 512-px-wide PNG raster:

            ```python
            >>> ds = Dataset.from_wms(  # doctest: +SKIP
            ...     "https://ows.terrestris.de/osm/service?",
            ...     layers="OSM-WMS",
            ...     bbox=(5.0, 51.0, 6.0, 52.0),
            ...     size=(512, 512),
            ... )

            ```

        See Also:
            - :meth:`from_wmts`: the tiled (WMTS) sibling.
            - :meth:`from_wcs`: read coverage *data values* instead of imagery.
        """
        return _from_wms(
            cls,
            endpoint,
            layers=layers,
            bbox=bbox,
            crs=crs,
            size=size,
            resolution=resolution,
            image_format=image_format,
            version=version,
            bands=bands,
            output_crs=output_crs,
            output=output,
            resample=resample,
            auth=auth,
            timeout=timeout,
        )

    @classmethod
    def from_wmts(
        cls,
        endpoint: str,
        *,
        layer: str,
        bbox: tuple[float, float, float, float],
        crs: str = _DEFAULT_CRS,
        tile_matrix_set: str | None = None,
        resolution: float | tuple[float, float] | None = None,
        layer_crs: str | None = None,
        output_crs: str | None = None,
        output: str | Path | None = None,
        resample: str = "nearest",
        auth: tuple[str, str] | None = None,
        timeout: float = 60.0,
    ) -> Dataset:
        """Crop a WMTS tile-pyramid layer to ``bbox`` into a :class:`Dataset`.

        Opens a Web Map Tile Service layer as a full georeferenced tile pyramid
        via GDAL's native WMTS driver, then crops ``bbox`` out of it (reprojecting
        the bbox into the layer's native CRS with ``pyproj``, mirroring
        :meth:`from_wcs`). The result is **rendered imagery** (RGB / RGBA), not data
        values.

        Args:
            endpoint: The WMTS ``GetCapabilities`` URL (e.g.
                ``"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml"``).
            layer: The layer identifier as advertised by the capabilities document.
                A value the service does not advertise raises :class:`ValueError`
                (with the available layers listed).
            bbox: ``(minx, miny, maxx, maxy)`` in ``crs`` order.
            crs: CRS of ``bbox``. Defaults to ``"EPSG:4326"``.
            tile_matrix_set: Optional tile-matrix-set id to pin. ``None`` lets GDAL
                pick the layer's default.
            resolution: Output pixel size in the layer's native CRS units — GDAL
                reads from the matching overview level. ``None`` (default) uses the
                finest level, which can be **very large** for a wide bbox; pass
                ``resolution`` to coarsen a large area.
            layer_crs: The layer's CRS, used only when the WMTS layer opens without
                a resolvable spatial reference (any proj4 / WKT / authority string).
            output_crs: Optional CRS to reproject the result into. ``None`` keeps
                the layer's native CRS.
            output: Optional path to also write the result to as a GeoTIFF.
            resample: Resampling method for the crop / warp. Defaults to
                ``"nearest"``.
            auth: Optional ``(username, password)`` for Basic-authed services.
            timeout: HTTP timeout in seconds. Defaults to ``60.0``.

        Returns:
            Dataset: The cropped WMTS window.

        Raises:
            ValueError: ``bbox`` is malformed, ``layer`` is not advertised,
                ``layer_crs`` cannot be interpreted, or the requested window exceeds
                the pixel ceiling (:data:`~pyramids.base._coverage.MAX_PX`; a
                finest-level read over a wide ``bbox`` — pass a coarser ``resolution``
                or a smaller ``bbox`` to bound it).
            pyramids.errors.WMSError: The server could not be reached or the tile
                read failed.

        Examples:
            Crop a NASA GIBS true-colour window (coarsened to ~0.01° pixels):

            ```python
            >>> ds = Dataset.from_wmts(  # doctest: +SKIP
            ...     "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml",
            ...     layer="MODIS_Terra_CorrectedReflectance_TrueColor",
            ...     bbox=(5.0, 51.0, 6.0, 52.0),
            ...     resolution=0.01,
            ... )

            ```

        See Also:
            - :meth:`from_wms`: the untiled (WMS ``GetMap``) sibling.
            - :meth:`from_wcs`: read coverage *data values* instead of imagery.
        """
        return _from_wmts(
            cls,
            endpoint,
            layer=layer,
            bbox=bbox,
            crs=crs,
            tile_matrix_set=tile_matrix_set,
            resolution=resolution,
            layer_crs=layer_crs,
            output_crs=output_crs,
            output=output,
            resample=resample,
            auth=auth,
            timeout=timeout,
        )

    @classmethod
    def from_ogc_coverages(
        cls,
        endpoint: str,
        *,
        coverage: str,
        bbox: tuple[float, float, float, float],
        output_crs: str | None = None,
        resolution: float | tuple[float, float] | None = None,
        coverage_crs: str | None = None,
        output: str | Path | None = None,
        resample: str = "nearest",
        auth: tuple[str, str] | None = None,
        timeout: float = 60.0,
    ) -> Dataset:
        """Read a coverage subset from an **OGC API – Coverages** service.

        Fetches a windowed subset of a coverage from an OGC API – Coverages
        service and returns it as a :class:`Dataset`. OGC API – Coverages is the
        modern REST/JSON successor to WCS: a landing page links to
        ``/collections`` and each coverage exposes ``/collections/{id}/coverage``
        with format negotiation. The transport is GDAL's native ``OGCAPI`` driver,
        so discovery, GeoTIFF negotiation and the windowed read happen inside GDAL;
        the caller supplies a single lon/lat ``bbox`` (plus optional ``resolution``
        and ``output_crs``). The driver exposes the coverage as an unbounded virtual
        raster, so the ``bbox`` is applied at read time as a native-CRS ``projWin``
        window (not passed through as a service-side ``bbox`` subset). This is the
        OGC-API-era sibling of :meth:`from_wcs`.

        A ``bbox`` is **required**. The driver exposes the coverage as an unbounded
        virtual raster, so a windowless read is impossible; pyramids projects the
        lon/lat ``bbox`` into the coverage's native CRS and reads it with an
        explicit output-size cap so the fetch always stays bounded.

        The ``coverage`` is validated against a (cached) ``/collections`` document
        so an unadvertised coverage fails fast with a clear :class:`ValueError`
        rather than an opaque driver error.

        Args:
            endpoint: The OGC API landing-page / base URL (e.g.
                ``"https://maps.gnosis.earth/ogcapi"``). Catalog / coverage-name
                routing belongs in the calling layer, not here.
            coverage: The coverage identifier as advertised by ``/collections``
                (e.g. ``"SRTM_ViewFinderPanorama"``). A value the service does not
                advertise raises :class:`ValueError`.
            bbox: **Required** ``(minx, miny, maxx, maxy)`` spatial subset in
                **lon/lat (CRS84)**. It is projected into the coverage's native CRS
                and read as a bounded, size-capped window; an unbounded full read is
                not supported (the virtual raster spans the whole coverage).
            output_crs: Optional CRS to reproject the result into (any form
                :meth:`to_crs` accepts). ``None`` (default) keeps the coverage's
                native CRS.
            resolution: Approximate pixel size of the read window, in the units of
                the coverage's **native CRS** (CRS84 degrees by default). A scalar
                gives square pixels; an ``(x_res, y_res)`` pair gives non-square
                pixels; every axis must be strictly positive (:class:`ValueError`
                otherwise). The window size is ``round(span / resolution)`` per
                axis, so the realised cell size equals ``resolution`` exactly only
                when ``span / resolution`` is integral and is otherwise the nearest
                whole-pixel fit. ``None`` (default) caps the longer side of the
                window at 1024 px (preserving the bbox aspect ratio). A window
                larger than 25000 px on either side is rejected with
                :class:`ValueError`. When ``output_crs`` is set, ``resolution``
                sizes the native-CRS read; the reprojected output's pixel size is
                then chosen by the warp.
            coverage_crs: The coverage's CRS, used only when the service's
                advertised CRS does not resolve in PROJ so GDAL opens the coverage
                with no spatial reference. Any proj4 / WKT / authority string
                ``pyproj`` understands. ``None`` (default) relies on the CRS the
                service advertises. Mirrors :meth:`from_wcs`.
            output: Optional path to also write the result to as a GeoTIFF. The
                method still returns the :class:`Dataset`.
            resample: Resampling method for the ``output_crs`` reprojection.
                Defaults to ``"nearest"``.
            auth: Optional ``(username, password)`` for Basic-authed services.
            timeout: HTTP timeout in seconds for the metadata / coverage requests
                (whole seconds; a value below 1 is clamped to 1). Defaults to
                ``60.0``.

        Returns:
            Dataset: The fetched coverage subset.

        Raises:
            ValueError: ``bbox`` is malformed, ``coverage`` is not advertised, or
                ``coverage_crs`` cannot be interpreted.
            pyramids.errors.OGCAPIError: The service could not be reached or
                returned an error / a non-raster body.

        Examples:
            Read a small bbox subset of a public coverage (network call — skipped
            in doctests):

            ```python
            >>> ds = Dataset.from_ogc_coverages(  # doctest: +SKIP
            ...     "https://maps.gnosis.earth/ogcapi",
            ...     coverage="SRTM_ViewFinderPanorama",
            ...     bbox=(5.0, 51.0, 6.0, 52.0),
            ... )

            ```

        See Also:
            - :meth:`from_wcs`: the classic WCS sibling.
            - :meth:`pyramids.feature.FeatureCollection.from_ogc_features`: the OGC
              API – Features (vector) sibling.
            - :meth:`read_file`: open a raster from a path or URL.
        """
        return _from_ogc_coverages(
            cls,
            endpoint,
            coverage=coverage,
            bbox=bbox,
            output_crs=output_crs,
            resolution=resolution,
            coverage_crs=coverage_crs,
            output=output,
            resample=resample,
            auth=auth,
            timeout=timeout,
        )

    def copy(self, path: str | Path | None = None) -> Dataset:
        """Deep copy.

        Args:
            path (str, optional):
                Destination for the copy. `None` (default) copies into memory
                with the `MEM` driver. Otherwise the extension alone selects
                the output format (`.tif` -> GTiff, `.nc` -> netCDF,
                `.png` -> PNG, …), so `copy` doubles as a format conversion
                and is not GeoTIFF-only. The copy is made with `CreateCopy`,
                so a write-by-copy-only format such as PNG or JPEG is accepted
                here even though the `Create`-based constructors
                (`from_array`, `create_empty`) refuse it.

        Returns:
            Dataset: An independent copy. Access mode of the returned
            Dataset:

            * `path is None` (in-memory copy) → access mode of the
              source is preserved. A `copy()` of a read-only source
              stays read-only at the pyramids level (the underlying
              MEM driver is always writable; pyramids enforces the
              flag itself).
            * `path is not None` and the format supports `Create`
              (GTiff, netCDF, HFA, …) → `"write"`, because the caller
              has just made a new file they presumably want to
              populate.
            * `path is not None` and the format is write-by-copy only
              (`.png`, `.jpg` / `.jpeg`, `.jp2` / `.j2k`, `.asc`) →
              `"read_only"`. `CreateCopy` hands back a read-only
              dataset for those, so claiming otherwise would let a
              write fail inside GDAL instead of raising
              :class:`~pyramids.errors.ReadOnlyError` here.

        Raises:
            DriverNotExistError: `path` has no extension, or one the driver
                catalog does not know.
            FileFormatNotSupportedError: `path` names a format that writes a
                reference rather than a self-contained raster (`.vrt`), which
                would produce a file GDAL cannot reopen.

        Examples:
            - Copy into memory and edit the copy without touching the source:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> src = Dataset.from_array(
                ...     np.zeros((3, 4), dtype="int16"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> clone = src.copy()
                >>> clone.write_array(np.full((3, 4), 7, dtype="int16"))
                >>> int(clone.read_array().max()), int(src.read_array().max())
                (7, 0)

                ```
            - The destination extension picks the format, so a copy can convert:
                ```python
                >>> import os, tempfile
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> src = Dataset.from_array(
                ...     np.arange(12, dtype="uint8").reshape(3, 4),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ...     no_data_value=None,
                ... )
                >>> out = os.path.join(tempfile.mkdtemp(), "converted.png")
                >>> png = src.copy(out)
                >>> png.raster.GetDriver().ShortName
                'PNG'
                >>> png.close()

                ```
        """
        if path is None:
            path = ""
            driver = MEMORY_DRIVER
            new_access = self._access
        else:
            # From the extension, not hardcoded: `copy(path="x.nc")` produced a
            # GTiff carrying a netCDF name. `for_copy` because this writes
            # with CreateCopy, which copy-only formats support.
            driver = resolve_output_driver(path, for_copy=True)
            # A copy-only driver returns a read-only handle, so claiming
            # "write" here meant `write_array` leaked a raw GDAL error past
            # the package's own ReadOnlyError guard.
            new_access = "write" if copy_yields_writable(driver) else "read_only"

        src = gdal.GetDriverByName(driver).CreateCopy(str(path), self._raster)
        return Dataset(src, access=new_access)

    def close(self) -> None:
        """Close the dataset.

        Safe to call multiple times — subsequent calls after the first are no-ops.

        Also releases the per-thread file manager created by
        ``read_array(threadsafe=True)``: the calling thread's handle is
        closed eagerly and the manager reference is dropped, so handles
        held by other (finished) threads are released with it. Without
        this, lingering read-only handles would keep the file locked on
        Windows after ``close()``.
        """
        if self._raster is not None:
            self._raster.FlushCache()
            self._raster = None
        manager = getattr(self, "_thread_manager", None)
        if manager is not None:
            manager.close()
            self._thread_manager = None

    @staticmethod
    def _create_dataset(
        cols: int,
        rows: int,
        bands: int,
        dtype: int,
        path: str | Path | None = None,
        options: list[str] | None = None,
    ) -> gdal.Dataset:
        """Create a GDAL driver.

            creates a driver and save it to disk and in memory if the path is not given.

        Args:
            cols (int):
                Number of columns.
            rows (int):
                Number of rows.
            bands (int):
                Number of bands.
            dtype:
                GDAL data type.
            path (str):
                Destination, which alone selects the driver: `None` builds in
                memory (MEM), otherwise the extension decides via
                :func:`~pyramids.dataset._driver.resolve_output_driver`.
            options (list[str] | None):
                GDAL creation options for the disk driver (e.g.
                ``["TILED=YES", "SPARSE_OK=TRUE", "BIGTIFF=YES"]``). When
                `None` (default), GTiff falls back to ``["COMPRESS=LZW"]`` —
                the historical behaviour — and every other disk driver gets an
                empty list. Must not be given without a `path`: the MEM driver
                takes no creation options, so they would be silently dropped.

        Returns:
            gdal.Dataset: The freshly allocated GDAL dataset — in memory when
            `path` is `None`, otherwise created on disk by the driver the
            extension selected.

        Raises:
            ValueError: `options` is given without a `path`.
            DriverNotExistError: `path` has no extension, or one the driver
                catalog does not know.
            FileFormatNotSupportedError: `path`'s extension maps to a
                write-by-copy-only format, which has no working `Create`.
        """
        if path is None and options is not None:
            raise ValueError(
                "creation options need a path to write to; pass path='out.tif' for a "
                "disk-backed raster, or drop the options for an in-memory one (the "
                "MEM driver takes none, so they would be silently dropped)."
            )
        driver = resolve_output_driver(path)
        if path is None:
            dataset = gdal.GetDriverByName(driver).Create("", cols, rows, bands, dtype)
        else:
            if options is not None:
                # Callers that need tiled / sparse / BigTIFF output (e.g.
                # create_empty) pass their own options.
                creation_options = options
            elif driver == "GTiff":
                # LZW is lossless and compresses well, at the cost of more
                # computation. GTiff-specific, so other drivers get nothing
                # rather than an option they would reject.
                creation_options = ["COMPRESS=LZW"]
            else:
                creation_options = []
            dataset = gdal.GetDriverByName(driver).Create(
                str(path), cols, rows, bands, dtype, creation_options
            )
        return dataset

    @classmethod
    def _build_dataset(
        cls,
        cols: int,
        rows: int,
        bands: int,
        dtype: int,
        geo: tuple,
        crs: str,
        no_data_value: Any | None = DEFAULT_NO_DATA_VALUE,
        path: str | Path | None = None,
        access: str = "write",
        array: np.ndarray | None = None,
        options: list[str] | None = None,
    ) -> Dataset:
        """Build a Dataset: allocate, set geo/CRS, optionally fill no-data, optionally write.

        Single canonical factory for raster construction. Consolidates the
        ``_create_dataset + SetGeoTransform + SetProjection + wrap +
        _set_no_data_value (+ WriteArray)` pattern that `create``,
        `from_array`, `dataset_like`, and the per-op factories
        across `Spatial` / `Analysis` all need.

        Args:
            cols: Number of columns.
            rows: Number of rows.
            bands: Number of bands.
            dtype: GDAL data type code.
            geo: Geotransform tuple
                `(top_left_x, pixel_w, row_skew, top_left_y, col_skew,
                pixel_h)`.
            crs: Projection as WKT string.
            no_data_value: No-data value. Scalar (broadcast to all bands)
                or list (one per band). Pass `None` to skip the
                `_set_no_data_value` call so bands have no no-data
                sentinel — the same behaviour the public `create`
                factory exposes.
            path: Destination, which alone selects the driver. `None`
                (default) keeps the dataset in memory (MEM); otherwise the
                extension decides.
            access: Access mode for the Dataset wrapper. Default `"write"`.
                Note: MEM driver datasets can be written to regardless
                of access mode since the access flag is enforced at the
                pyramids level, not by GDAL.
            array: Optional numpy array to write into the bands after
                construction. When the array is 2-D it goes to band 1;
                when 3-D, `array[i, :, :]` goes to band `i+1`. The
                caller is responsible for matching `array.shape` to
                `bands x rows x cols` (or `rows x cols` for a
                single-band array). Default `None` (allocate but
                don't write).
            options: GDAL creation options forwarded to
                :meth:`_create_dataset` for disk drivers (e.g. the
                tiled / sparse / BigTIFF set used by :meth:`create_empty`).
                `None` (default) keeps the historical ``["COMPRESS=LZW"]``
                for GTiff and is ignored by the MEM driver.

        Returns:
            Dataset: A fully configured Dataset object.
        """
        dst = cls._create_dataset(cols, rows, bands, dtype, path=path, options=options)
        dst.SetGeoTransform(geo)
        dst.SetProjection(crs)
        dst_obj = cls(dst, access=access)
        if no_data_value is not None:
            dst_obj._set_no_data_value(no_data_value=no_data_value)
        if array is not None:
            if array.ndim == 2:
                dst_obj.raster.GetRasterBand(1).WriteArray(array)
            else:
                for i in range(bands):
                    dst_obj.raster.GetRasterBand(i + 1).WriteArray(array[i, :, :])
            dst_obj._raster.FlushCache()
        return dst_obj

    @classmethod
    def create(
        cls,
        rows: int,
        columns: int,
        dtype: str,
        bands: int,
        *,
        geo_ref: GeoReference,
        no_data_value: Any | None = None,
        path: str | Path | None = None,
    ) -> Dataset:
        """Create a new dataset, optionally filled with the no_data_value.

        With a `no_data_value` the sentinel is stamped on every band and the
        array is filled with it. The parameter defaults to `None`, which stamps
        no sentinel and performs no fill -- the bands read back as whatever the
        driver allocates (0 for GTiff). Pass one explicitly to get a filled
        raster, as :meth:`create_empty` documents for the same opt-out.

        Args:
            rows (int):
                Number of rows.
            columns (int):
                Number of columns.
            dtype (str):
                Data type.
            bands (int):
                Number of bands to create in the output raster. Required.
            geo_ref (GeoReference):
                How the array maps to space — an affine ``geo`` transform, or a
                ``top_left_corner`` + ``cell_size``, plus the ``epsg``. Required;
                a raster has to be placed somewhere. Note the CRS is not:
                ``epsg`` defaults to 4326, so a reference that omits it stamps
                WGS 84 rather than refusing. This method previously took a
                **required** ``epsg`` and raised ``TypeError`` when it was
                missing — pass ``epsg`` explicitly, or ``epsg=None`` for a
                deliberately CRS-less raster.
            no_data_value (float|None):
                No data value.
            path (str, optional):
                Destination, which alone decides the driver. `None` (default)
                builds the raster in memory; otherwise the extension selects the
                format (``.tif`` -> GTiff, ``.nc`` -> netCDF, …).

        Returns:
            Dataset: A new dataset

        Raises:
            ValueError: `geo_ref` carries neither a ``geo`` nor a complete
                ``top_left_corner`` + ``cell_size`` pair.
            DriverNotExistError: `path` has no extension, or one the driver
                catalog does not know.
            FileFormatNotSupportedError: `path`'s extension maps to a
                write-by-copy-only format such as PNG, which cannot be built
                with ``Create``.

        Examples:
            - Create a filled in-memory raster and read a cell back:
                ```python
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.create(
                ...     3, 4, "float32", 1,
                ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0)),
                ...     no_data_value=-9999.0,
                ... )
                >>> (ds.rows, ds.columns, ds.band_count)
                (3, 4, 1)
                >>> float(ds.read_array()[0, 0])
                -9999.0

                ```
            - Place it with a corner and a cell size instead of a transform:
                ```python
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.create(
                ...     2, 2, "int16", 1,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(10.0, 50.0), cell_size=0.5, epsg=4326
                ...     ),
                ... )
                >>> tuple(ds.geotransform)
                (10.0, 0.5, 0.0, 50.0, 0.0, -0.5)
                >>> ds.epsg
                4326

                ```

        See Also:
            - :meth:`create_empty`: Allocate the header only, without filling
              every cell — the out-of-core sibling of this method.
            - :meth:`from_array`: Build a raster around an array you already
              have.
        """
        gdal_dtype = numpy_to_gdal_dtype(dtype)
        crs_wkt = _crs_wkt_from_epsg(geo_ref.epsg)
        geotransform = geo_ref.resolve_geotransform()
        return cls._build_dataset(
            columns,
            rows,
            bands,
            gdal_dtype,
            geotransform,
            crs_wkt,
            no_data_value,
            path=path,
        )

    @classmethod
    def create_empty(
        cls,
        rows: int,
        cols: int,
        *,
        bands: int = 1,
        dtype: str = "float32",
        geo_ref: GeoReference | None = None,
        no_data_value: Any = DEFAULT_NO_DATA_VALUE,
        path: str | Path | None = None,
        options: list[str] | None = None,
    ) -> Dataset:
        """Allocate an empty, header-only raster without materialising a full array.

        Out-of-core algorithms allocate the output once and scatter result
        windows into it with
        ``write_array(array, window=Window(col_off, row_off, cols, rows))``
        (see :class:`~pyramids.dataset.window.Window`).
        With a ``.tif`` path the file is **tiled, sparse,
        and BigTIFF** (see :data:`OUT_OF_CORE_CREATION_OPTIONS`), so a
        50 000 x 50 000 float32 raster is created in O(1) RAM, never-written
        blocks cost no disk, and writes past the 4 GB classic-TIFF ceiling
        succeed. A never-written cell reads back as ``no_data_value`` (not 0) —
        on GTiff because SPARSE_OK + the band no-data sentinel returns no-data
        for unwritten blocks, and on MEM because the band is filled with the
        no-data value at allocation — so downstream code must treat unwritten
        tiles as no-data.

        Args:
            rows: Number of rows of the output raster.
            cols: Number of columns of the output raster.
            bands: Number of bands. Default 1.
            dtype: NumPy dtype name for the bands (e.g. ``"float32"``,
                ``"int16"``). Default ``"float32"``.
            geo_ref: How the raster maps to space — an affine ``geo``
                transform, or a ``top_left_corner`` + ``cell_size``, plus the
                ``epsg``. Unlike the other constructors this one is optional:
                a header-only allocation often does not care where it sits, so
                `None` (default) — or a reference carrying *no* transform at
                all, such as ``GeoReference(epsg=3857)`` — keeps the identity
                transform ``(0.0, 1.0, 0.0, 0.0, 0.0, -1.0)``, a unit-pixel
                grid with the origin at ``(0, 0)``. A **partially** specified
                reference is not covered by that convenience: a
                ``top_left_corner`` without a ``cell_size`` (or the reverse)
                raises, exactly as it does in :meth:`from_array` and
                :meth:`create`, rather than silently discarding the half that
                was supplied.
            no_data_value: No-data sentinel stamped on every band at
                creation. Default :data:`DEFAULT_NO_DATA_VALUE`. Keep it set
                so sparse unwritten blocks read back as no-data rather than 0.
                Passing ``None`` skips the band fill and stamps no sentinel,
                which opts out of that guarantee — a sparse GTiff's unwritten
                blocks then read back as **0**, not no-data. A path that
                resolves to GTiff emits a :class:`NoDataSentinelWarning`;
                the in-RAM ``"MEM"`` driver is dense, and the other disk
                drivers are not sparse, so neither warns.
            path: Destination, which alone decides the driver. `None`
                (default) builds an in-memory ``"MEM"`` raster; otherwise the
                extension selects the format (``.tif`` -> GTiff). The sparse /
                tiled / BigTIFF defaults below apply only when the path
                resolves to GTiff.
            options: GDAL creation options. `None` (default) uses
                :data:`OUT_OF_CORE_CREATION_OPTIONS` for GTiff. Override to
                align ``BLOCKXSIZE`` / ``BLOCKYSIZE`` to your tile size or to
                change compression. Forwarded to whichever disk driver the
                extension selects — only the *default* set is GTiff-specific;
                passing `options` without a `path` raises rather than silently
                dropping them.

        Returns:
            Dataset: An empty raster whose bands read back as `no_data_value`
            before any write. On GTiff this is sparse — SPARSE_OK keeps
            never-written blocks unallocated and GDAL returns the no-data
            sentinel for them; on MEM every band is filled with `no_data_value`
            at allocation, so unwritten MEM cells read back as no-data too.

        Raises:
            ValueError: ``options`` is given without a ``path`` — creation
                options apply only to a disk driver, so accepting them for an
                in-memory raster would silently drop them; or `geo_ref` is
                partially specified (one half of the
                ``top_left_corner`` / ``cell_size`` pair).
            DriverNotExistError: ``path`` has an extension the driver catalog
                does not know.
            FileFormatNotSupportedError: ``path``'s extension maps to a
                write-by-copy-only format, which cannot be allocated with
                ``Create``.

        Examples:
            - Allocate an in-memory empty raster and read its no-data metadata:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_empty(
                ...     4, 5, dtype="float32", no_data_value=-9999.0
                ... )
                >>> (ds.rows, ds.columns, ds.band_count)
                (4, 5, 1)
                >>> float(ds.no_data_value[0])
                -9999.0

                ```
            - Allocate, then scatter a window into it and read it back:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from pyramids.dataset import Window
                >>> ds = Dataset.create_empty(4, 4, dtype="float32")
                >>> block = np.arange(4, dtype="float32").reshape(2, 2)
                >>> ds.write_array(block, window=Window(1, 1, 2, 2))
                >>> ds.read_array(window=[1, 1, 2, 2]).tolist()
                [[0.0, 1.0], [2.0, 3.0]]

                ```

        See Also:
            - :meth:`empty_like`: Allocate an empty raster shaped like an
              existing template instead of from explicit dimensions.
            - :meth:`create`: Allocate a raster and eagerly fill every cell
              with the no-data value (no sparse / BigTIFF defaults).
            - :meth:`write_array`: Scatter a window into the allocated raster
              (``window=(row_off, col_off, n_rows, n_cols)``).
        """
        # The old "GTiff without a path" guard is gone: that combination is now
        # unrepresentable, because the driver comes from the path.
        # Creation options apply only to a disk driver (path given); the MEM
        # driver takes none. Reject explicit options that would be dropped
        # rather than silently ignoring them.
        if options is not None and path is None:
            raise ValueError(
                "create_empty received `options` but no `path`: GDAL creation "
                "options apply only to a disk driver. Pass a `path`, or drop "
                "`options` for the in-memory MEM raster."
            )
        # Only a sparse GTiff can read back 0 instead of no-data for a block
        # that was never written, so the warning is specific to that target.
        # The MEM driver (path is None) is a dense in-RAM buffer, and the other
        # disk drivers are not sparse either — warning about unwritten sparse
        # blocks on a netCDF would state a reason that does not apply to it.
        if no_data_value is None and _resolves_to_gtiff(path):
            warnings.warn(
                "create_empty(no_data_value=None) on a GTiff target stamps no "
                "no-data sentinel, so unwritten sparse blocks read back as 0, not "
                "no-data. Pass a no_data_value to keep the 'unwritten == no-data' "
                "guarantee.",
                NoDataSentinelWarning,
                stacklevel=2,
            )
        gdal_dtype = numpy_to_gdal_dtype(dtype)
        # `create_empty` allocates a header; where it sits in space is often
        # irrelevant to the caller. A geo_ref that carries no transform at all
        # (e.g. `GeoReference(epsg=3857)`) therefore keeps the identity one
        # rather than raising, which is what the flat `epsg=`-only form did.
        geo_ref = geo_ref if geo_ref is not None else GeoReference()
        # Substitute the identity transform only when the reference carries no
        # georeferencing at all. A *half*-filled pair (a corner with no cell
        # size, or the reverse) is a mistake, not a request for the origin:
        # falling back here would silently discard the half the caller did
        # supply, and place the raster at (0, 0) with 1-unit pixels. Leaving it
        # to `resolve_geotransform()` makes it raise, which is what the same
        # value object already does in `from_array` and `create`.
        if (
            geo_ref.geo is None
            and geo_ref.top_left_corner is None
            and geo_ref.cell_size is None
        ):
            geo_ref = replace(geo_ref, geo=_IDENTITY_GEO)
        crs_wkt = _crs_wkt_from_epsg(geo_ref.epsg)
        geo = geo_ref.resolve_geotransform()
        # The tiled / sparse / BigTIFF defaults are GTiff-specific, so apply them
        # only when the path actually resolves to GTiff.
        if options is None and _resolves_to_gtiff(path):
            options = list(OUT_OF_CORE_CREATION_OPTIONS)
        return cls._build_dataset(
            cols,
            rows,
            bands,
            gdal_dtype,
            geo,
            crs_wkt,
            no_data_value,
            path=path,
            options=options,
            array=None,
        )

    @classmethod
    def empty_like(
        cls,
        template: Dataset,
        *,
        dtype: str | None = None,
        bands: int | None = None,
        no_data_value: Any = _INHERIT_NO_DATA,
        path: str | Path | None = None,
        options: list[str] | None = None,
    ) -> Dataset:
        """Allocate an empty raster aligned to a template's geo / epsg / shape / nodata.

        The header-only sibling of :meth:`dataset_like` — same spatial
        footprint as `template` (geotransform, CRS, rows, columns, no-data),
        but **no array is written**, so it can allocate an out-of-core output
        the size of an input DEM without materialising it. The driver comes
        from `path`: MEM when it is `None`, otherwise whatever the extension
        selects. A `.tif` destination additionally gets the tiled / sparse /
        BigTIFF defaults in :data:`OUT_OF_CORE_CREATION_OPTIONS`.

        Args:
            template: Source raster whose geotransform, CRS, shape, and
                no-data value the output copies.
            dtype: NumPy dtype name for the output bands. `None` (default)
                reuses the template's dtype.
            bands: Number of output bands. `None` (default) reuses the
                template's band count.
            no_data_value: No-data sentinel for the output. Default inherits
                from the template: when the band count is unchanged and every
                template band has a sentinel, the **per-band** no-data values
                are preserved; otherwise (a `bands` override, or a template
                band with no sentinel) the template's first-band value is used.
                Pass an explicit scalar or per-band list to override. If this
                resolves to ``None`` (passed explicitly, or inherited from a
                template with no no-data set), no sentinel is stamped and a
                sparse GTiff's unwritten blocks read back as **0**, not no-data;
                a path that resolves to GTiff emits a
                :class:`NoDataSentinelWarning` (the in-RAM MEM result and the
                other, non-sparse disk drivers do not warn).
            path: Destination, which alone decides the driver. `None`
                (default) keeps the raster in memory (MEM); otherwise the
                extension selects the format (`.tif` -> GTiff, `.nc` ->
                netCDF, ...).
            options: GDAL creation options forwarded to the disk driver.
                `None` (default) uses :data:`OUT_OF_CORE_CREATION_OPTIONS`,
                but only when the path resolves to GTiff — those options are
                GTiff-specific, so any other disk driver gets none. Passing
                `options` without a `path` raises rather than silently
                dropping them.

        Returns:
            Dataset: An empty raster matching the template's footprint.

        Raises:
            ValueError: `options` is given without a `path` (creation
                options apply only to a disk driver).
            DriverNotExistError: `path` has no extension, or one the driver
                catalog does not know.
            FileFormatNotSupportedError: `path`'s extension maps to a
                write-by-copy-only format, which cannot be allocated with
                `Create`.

        Examples:
            - Allocate an empty raster shaped like an existing one, with a
              different dtype:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> template = Dataset.from_array(
                ...     np.ones((3, 4, 5), dtype="float32"),
                ...     no_data_value=-9999.0,
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 10.0), cell_size=0.5, epsg=4326),
                ... )
                >>> out = Dataset.empty_like(template, dtype="int16")
                >>> (out.rows, out.columns, out.band_count, out.epsg)
                (4, 5, 3, 4326)
                >>> out.geotransform == template.geotransform
                True

                ```
            - Reduce the band count and inherit the template's no-data value,
              then confirm the empty output reads back as no-data:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> template = Dataset.from_array(
                ...     np.ones((3, 4, 4), dtype="float32"),
                ...     no_data_value=-9999.0,
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326),
                ... )
                >>> out = Dataset.empty_like(template, bands=1)
                >>> out.band_count
                1
                >>> float(out.no_data_value[0])
                -9999.0

                ```

        See Also:
            - :meth:`create_empty`: Allocate an empty raster from explicit
              dimensions / CRS instead of copying a template.
            - :meth:`dataset_like`: The array-writing sibling — copies the
              template footprint *and* writes a supplied array.
            - :meth:`write_array`: Scatter a window into the allocated raster.
        """
        if options is not None and path is None:
            raise ValueError(
                "empty_like received `options` but no `path`: GDAL creation options "
                "apply only to a disk driver. Pass a `path`, or drop "
                "`options` for the in-memory MEM raster."
            )
        gdal_dtype = (
            template.gdal_dtype[0] if dtype is None else numpy_to_gdal_dtype(dtype)
        )
        n_bands = template.band_count if bands is None else bands
        if no_data_value is not _INHERIT_NO_DATA:
            nodata = no_data_value
        else:
            template_nd = template.no_data_value
            # Preserve the template's per-band sentinels when the band count is
            # unchanged and every band actually has one; otherwise (band-count
            # override, or a band with no sentinel) fall back to band 0's value.
            if bands is None and all(v is not None for v in template_nd):
                nodata = list(template_nd)
            else:
                nodata = template_nd[0]
        # Warn only for a GTiff target, the only one whose unwritten sparse
        # blocks read back as 0 when no sentinel is stamped. An in-RAM MEM
        # result (no path) is dense, and the other disk drivers are not sparse,
        # so the stated reason would not apply to them.
        if nodata is None and _resolves_to_gtiff(path):
            warnings.warn(
                "empty_like produced a GTiff raster with no no-data sentinel "
                "(no_data_value resolved to None, explicitly or inherited from a "
                "template with no no-data), so unwritten sparse blocks read back as "
                "0, not no-data. Pass no_data_value to keep the 'unwritten == "
                "no-data' guarantee.",
                NoDataSentinelWarning,
                stacklevel=2,
            )
        # The tiled / sparse / BigTIFF defaults are GTiff-specific, so apply them
        # only when the path actually resolves to GTiff — matching `create_empty`.
        # Hardcoding "GTiff for any path" was harmless while the driver was
        # passed down explicitly; now that it comes from the extension, a `.nc`
        # destination really is netCDF and must not be handed GTiff options.
        if options is None and _resolves_to_gtiff(path):
            options = list(OUT_OF_CORE_CREATION_OPTIONS)
        return cls._build_dataset(
            template.columns,
            template.rows,
            n_bands,
            gdal_dtype,
            template.geotransform,
            template.crs,
            nodata,
            path=path,
            options=options,
            array=None,
        )

    @classmethod
    def from_features(
        cls,
        features: FeatureCollection,
        *,
        cell_size: Any | None = None,
        template: Dataset | None = None,
        snap_to_template: bool = False,
        column_name: str | list[str] | None = None,
    ) -> Dataset:
        """Rasterize a :class:`FeatureCollection` into a new :class:`Dataset`.

        Burns the values from `column_name` (or every attribute
        column if `None`) into a single-band or multi-band raster.
        When a `template` Dataset is given, the output adopts its
        geotransform, cell size, row/column count, and no-data value —
        the vector is burned onto the template's fixed grid, so features
        outside it are clipped. With `snap_to_template=True` the template
        supplies only the cell size and grid alignment while the extent is
        cropped to the features (snapped onto the template's grid lines),
        giving a small raster co-registered with the template. Otherwise
        `cell_size` controls the resolution and the extent is derived from
        :attr:`FeatureCollection.total_bounds`.

        Args:
            features (FeatureCollection):
                The vector to rasterize.
            cell_size (int | float | None):
                Cell size for the new raster. Required unless
                `template` is given.
            template (Dataset | None):
                Optional template raster. When supplied, the output
                inherits its geotransform and no-data value. Features
                that fall entirely outside the template extent, or an
                empty FeatureCollection, produce an all-nodata raster
                and emit a `UserWarning` (#46); use `cell_size` instead
                to size the output to the features.
            snap_to_template (bool):
                When `True` (requires `template`), keep the template's
                cell size and grid alignment but size the output to the
                features' bounds snapped outward onto the template's grid
                lines — a small raster that still co-registers with the
                template pixel-for-pixel (#46). Requires a square,
                axis-aligned template and features with valid (non-NaN)
                geometry bounds. The output is sized to the features, so a
                fine-celled template with far-apart features can allocate a
                large raster.
            column_name (str | list[str] | None):
                Attribute column(s) to burn as band values. `None`
                burns every non-geometry column as a separate band.
                Mixed-dtype column lists are promoted to the smallest
                numpy dtype that holds every selected column without
                lossy cast (numpy result-type rules).

        Returns:
            Dataset: The burned raster.

        Raises:
            ValueError: `cell_size` missing or non-positive,
                `column_name` empty or referencing missing columns,
                `snap_to_template` set without a `template`, or (in snap
                mode) a rotated / non-square template or features with no
                valid (non-NaN) geometry bounds.
            TypeError: `template` is not a Dataset, or
                `column_name` is not `str` / `list` / `None`.
            CRSError: `features.epsg` is `None`, or
                `template.epsg!= features.epsg`.

        Examples:
            - `cell_size` sizes the output to the feature bounds:

              ```python
              >>> import geopandas as gpd
              >>> from shapely.geometry import box
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> from pyramids.feature import FeatureCollection
              >>> gdf = gpd.GeoDataFrame(
              ...     {"class_id": [7]},
              ...     geometry=[box(0.0, 0.0, 3.0, 3.0)],
              ...     crs="EPSG:4326",
              ... )
              >>> raster = Dataset.from_features(
              ...     FeatureCollection(gdf), cell_size=1.0, column_name="class_id"
              ... )
              >>> (raster.rows, raster.columns)
              (3, 3)
              >>> int(raster.read_array().max())
              7

              ```

            - A `template` burns onto its fixed grid, so the output adopts the template's
              shape (features outside it would warn and yield all-nodata — see #46):

              ```python
              >>> import numpy as np
              >>> template = Dataset.from_array(
              ...     np.zeros((5, 5), dtype="int32"),
              ...     geo_ref=GeoReference(top_left_corner=(0.0, 5.0), cell_size=1.0, epsg=4326),
              ... )
              >>> inside = FeatureCollection(
              ...     gpd.GeoDataFrame(
              ...         {"class_id": [7]},
              ...         geometry=[box(1.0, 1.0, 4.0, 4.0)],
              ...         crs="EPSG:4326",
              ...     )
              ... )
              >>> burned = Dataset.from_features(
              ...     inside, template=template, column_name="class_id"
              ... )
              >>> (burned.rows, burned.columns)
              (5, 5)

              ```

            - `snap_to_template=True` keeps the template's grid but crops to the features,
              so the output is small yet co-registered (its origin is on the template's
              grid lines):

              ```python
              >>> snapped = Dataset.from_features(
              ...     inside,
              ...     template=template,
              ...     snap_to_template=True,
              ...     column_name="class_id",
              ... )
              >>> (snapped.rows, snapped.columns)
              (3, 3)
              >>> snapped.top_left_corner
              (1.0, 4.0)

              ```
        """
        return rasterize_features(
            features,
            cls,
            cell_size=cell_size,
            template=template,
            snap_to_template=snap_to_template,
            column_name=column_name,
        )

    @classmethod
    def from_points(
        cls,
        points: FeatureCollection,
        value_column: str,
        *,
        algorithm: str = "invdist:power=2.0:smoothing=0.0",
        cell_size: float | None = None,
        width: int | None = None,
        height: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        epsg: Any | None = None,
    ) -> Dataset:
        """Interpolate scattered point samples onto a regular grid (``gdal.Grid``).

        The GDAL-native equivalent of ``gdal_grid`` — turns an irregular point
        layer (gauge readings, soundings, station observations) into a
        continuous single-band raster. The output extent defaults to the points'
        bounding box and the resolution is set by ``cell_size`` (or an explicit
        ``width``/``height``).

        Args:
            points (FeatureCollection):
                A point :class:`FeatureCollection` carrying ``value_column``.
            value_column (str):
                Numeric attribute column to interpolate (the Z field).
            algorithm (str):
                A ``gdal.Grid`` algorithm string. Defaults to inverse-distance
                weighting (``"invdist:power=2.0:smoothing=0.0"``). Other options
                include ``"invdistnn"``, ``"nearest"``, ``"linear"``, and
                ``"average"``.
            cell_size (float | None):
                Output pixel size in the points' CRS units. Required unless both
                ``width`` and ``height`` are given.
            width (int | None):
                Output width in pixels. Overrides ``cell_size`` on the x axis.
            height (int | None):
                Output height in pixels. Overrides ``cell_size`` on the y axis.
            bbox (tuple[float, float, float, float] | None):
                ``(minx, miny, maxx, maxy)`` output extent. Defaults to the
                points' total bounds.
            epsg (int | None):
                Output EPSG code. Defaults to the points' CRS.

        Returns:
            Dataset: A single-band raster of the interpolated surface.

        Raises:
            ValueError: ``value_column`` missing, output bounds degenerate, or
                neither ``cell_size`` nor ``width``+``height`` provided.
            FailedToSaveError: ``gdal.Grid`` produced no dataset.

        Examples:
            - Inverse-distance interpolate four corner readings onto a 1-degree
              grid and read back the surface shape:
                ```python
                >>> from shapely.geometry import Point
                >>> from geopandas import GeoDataFrame
                >>> from pyramids.feature import FeatureCollection
                >>> from pyramids.dataset import Dataset
                >>> gdf = GeoDataFrame(
                ...     {"rain": [10.0, 20.0, 30.0, 40.0]},
                ...     geometry=[Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)],
                ...     crs="EPSG:4326",
                ... )
                >>> ds = Dataset.from_points(FeatureCollection(gdf), "rain", cell_size=1.0)
                >>> (ds.rows, ds.columns, ds.band_count)
                (10, 10, 1)

                ```
            - Use nearest-neighbour with an explicit output size:
                ```python
                >>> from shapely.geometry import Point
                >>> from geopandas import GeoDataFrame
                >>> from pyramids.feature import FeatureCollection
                >>> from pyramids.dataset import Dataset
                >>> gdf = GeoDataFrame(
                ...     {"z": [1.0, 2.0, 3.0, 4.0]},
                ...     geometry=[Point(0, 0), Point(5, 0), Point(0, 5), Point(5, 5)],
                ...     crs="EPSG:4326",
                ... )
                >>> ds = Dataset.from_points(
                ...     FeatureCollection(gdf), "z", algorithm="nearest", width=5, height=5
                ... )
                >>> ds.columns
                5

                ```
        """
        return grid_points(
            points,
            value_column,
            cls,
            algorithm=algorithm,
            cell_size=cell_size,
            width=width,
            height=height,
            bbox=bbox,
            epsg=epsg,
        )

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        *,
        geo_ref: GeoReference,
        no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
        path: str | Path | None = None,
    ) -> Dataset:
        """Create a new dataset from an array.

        Args:
            arr (np.ndarray):
                Numpy array.
            geo_ref (GeoReference):
                How the array maps to space — an affine ``geo`` transform, or a
                ``top_left_corner`` + ``cell_size``, plus the ``epsg``. Required;
                a raster has to be placed somewhere. An ``epsg`` of `None` (or
                `0`) creates an ungeoreferenced raster that reports no CRS,
                rather than one silently stamped WGS 84.
            no_data_value (Any, optional):
                No data value to mask the cells out of the domain. The default is -9999.
            path (str, optional):
                Destination. `None` (default) builds the raster in memory;
                otherwise the extension selects the driver (``.tif`` -> GTiff,
                ``.nc`` -> netCDF, …). A ``.nc`` destination here writes a
                *classic* single-variable netCDF through the plain GDAL raster
                API; for a multi-variable, CF-attributed store use
                :meth:`pyramids.netcdf.NetCDF.from_array`, which goes through
                the multidimensional path.

        Returns:
            Dataset:
                Dataset object will be returned.

        Raises:
            ValueError: `geo_ref` carries neither a ``geo`` nor a complete
                ``top_left_corner`` + ``cell_size`` pair.
            DriverNotExistError: `path` has no extension, or one the driver
                catalog does not know.
            FileFormatNotSupportedError: `path`'s extension maps to a
                write-by-copy-only format such as PNG, which cannot be built
                with ``Create``.

        Examples:
            - Wrap a 2-D array, then read it back:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.arange(6, dtype="float32").reshape(2, 3)
                >>> ds = Dataset.from_array(
                ...     arr, geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0))
                ... )
                >>> ds.read_array().tolist()
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
                >>> (ds.rows, ds.columns, ds.band_count)
                (2, 3, 1)

                ```
            - A leading axis becomes bands:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> stack = np.ones((3, 2, 2), dtype="int16")
                >>> ds = Dataset.from_array(
                ...     stack,
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0),
                ... )
                >>> ds.band_count
                3

                ```
            - `epsg=None` builds an ungeoreferenced raster rather than
              silently claiming WGS 84:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.zeros((2, 2), dtype="float32"),
                ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=None),
                ... )
                >>> ds.crs
                ''

                ```

        See Also:
            - :meth:`dataset_like`: Reuse another dataset's georeferencing
              instead of stating it.
            - :meth:`create`: Allocate a raster filled with the no-data value
              when there is no array yet.
        """
        geo = geo_ref.resolve_geotransform()
        epsg = geo_ref.epsg

        if arr.ndim == 2:
            bands = 1
            rows = int(arr.shape[0])
            cols = int(arr.shape[1])
        else:
            bands = arr.shape[0]
            rows = int(arr.shape[1])
            cols = int(arr.shape[2])

        # The shared helper owns the CRS rules — the `sr_from_epsg` path for an
        # EPSG int/numeric string, the `sr_from_user_input` fallback that carries
        # a no-EPSG CRS such as geostationary through as WKT (#706), and the
        # empty-string result that leaves a deliberately ungeoreferenced raster
        # unprojected rather than stamping a default (ARC-26).
        crs_wkt = _crs_wkt_from_epsg(epsg)

        return cls._build_dataset(
            cols,
            rows,
            bands,
            numpy_to_gdal_dtype(arr),
            geo,
            crs_wkt,
            no_data_value,
            path=path,
            array=arr,
        )

    @classmethod
    def dataset_like(
        cls,
        src: Dataset,
        array: np.ndarray,
        path: str | Path | None = None,
    ) -> Dataset:
        """Create a new dataset like another dataset.

        dataset_like method creates a Dataset from an array like another source dataset. The new dataset
        will have the same `projection`, `coordinates` or the `top left corner` of the original dataset,
        `cell size`, `no_data_velue`, and number of `rows` and `columns`.
        the array and the source dataset should have the same number of columns and rows

        Args:
            src (Dataset):
                source raster to get the spatial information
            array (ndarray):
                data to store in the new dataset.
            path (str, optional):
                path to save the new dataset, if not given, the method will return in-memory dataset.

        Returns:
            Dataset:
                if the `path` is given, the method will save the new raster to the given path, else the
                method will return an in-memory dataset.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array should be of type numpy array")

        bands = 1 if array.ndim == 2 else array.shape[0]
        return cls._build_dataset(
            src.columns,
            src.rows,
            bands,
            numpy_to_gdal_dtype(array),
            src.geotransform,
            src.crs,
            src.no_data_value[0],
            path=path,
            array=array,
        )

    @classmethod
    def from_band_files(
        cls,
        files: Sequence[str | Path],
        *,
        band_names: list[str] | None = None,
        align: bool = False,
        no_data_value: Any = _INHERIT_NO_DATA,
        path: str | Path | None = None,
    ) -> Dataset:
        """Stack N single-band rasters into one multi-band :class:`Dataset`.

        Each input file becomes one band, in order, with its name preserved.
        This is the natural target for an Earth Engine default download
        (``<assetSlug>.<bandName>.tif`` — one file per band), a Landsat
        Collection-2 scene (per-band ``.TIF``), or a Sentinel-2 SAFE
        (per-band JP2s).

        By default all inputs must already share the same grid and CRS;
        pass ``align=True`` to resample mismatched rasters onto the first
        file's grid (nearest-neighbour, via :meth:`align`). When the inputs
        have different numpy dtypes the output dtype is the smallest type
        that holds every input without a lossy cast.

        Args:
            files: Paths (or URLs / ``/vsi*`` strings) of the single-band
                rasters to stack. Order is preserved as band order.
            band_names: Explicit band names, one per file. When ``None``
                (default) names are derived from the file names
                (``<slug>.<band>.tif`` → ``<band>``; dotless stems are kept
                whole; duplicates get a ``_<n>`` suffix).
            align: When ``False`` (default), a grid/CRS mismatch among the
                inputs raises :class:`AlignmentError`. When ``True``, every
                input is resampled onto ``files[0]``'s grid first.
            no_data_value: No-data value stamped on the output bands. When
                omitted, it is inherited from the source rasters (a warning
                is issued if they disagree, and the first file's value
                wins; if no source declares one, the output has none). Pass
                an explicit value (including ``None`` for "no no-data
                sentinel") to override.
            path: Output path, whose extension selects the driver as it does
                for every other factory (``.tif`` -> GTiff, ``.nc`` ->
                netCDF, …). When ``None`` (default) the result is an
                in-memory dataset.

                Write-by-copy-only formats (`.png`, `.jp2`) are refused. One
                of the two internal write paths could produce them --
                aligned, same-dtype inputs go through a VRT and
                `CreateCopy` -- but which path runs depends on `align` and
                on whether the sources share a dtype, neither of which says
                anything about the destination format. Accepting `.png` only
                sometimes would make the destination's legality depend on an
                unrelated argument, so both paths answer alike.

        Returns:
            Dataset: A multi-band dataset with ``band_count == len(files)``
            and ``band_names`` set.

        Raises:
            ValueError: ``files`` is empty, ``band_names`` length does not
                match ``files``, or an input has more than one band.
            AlignmentError: ``align=False`` and the inputs do not share a
                grid/CRS.
            CRSError: An input raster has no CRS.
            DriverNotExistError: ``path`` has no extension, or one the driver
                catalog does not know.
            FileFormatNotSupportedError: ``path``'s extension maps to a
                write-by-copy-only format, whichever write path the inputs
                take.

        Examples:
            - Stack three per-band GeoTIFFs into one 3-band dataset; band
              names come from the file names:
                ```python
                >>> import numpy as np
                >>> import tempfile, os
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> d = tempfile.mkdtemp()
                >>> paths = []
                >>> triples = [("scene.B2.tif", 2), ("scene.B3.tif", 3), ("scene.B4.tif", 4)]
                >>> for name, val in triples:
                ...     p = os.path.join(d, name)
                ...     _ = Dataset.from_array(
                ...         np.full((4, 5), val, dtype="int16"),
                ...         geo_ref=GeoReference(
                ...             top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...         ),
                ...         path=p,
                ...     ).close()
                ...     paths.append(p)
                >>> ds = Dataset.from_band_files(paths)
                >>> ds.band_count
                3
                >>> ds.band_names
                ['B2', 'B3', 'B4']
                >>> [int(ds.read_array(band=i).flat[0]) for i in range(3)]
                [2, 3, 4]

                ```
            - Override the band names explicitly:
                ```python
                >>> ds = Dataset.from_band_files(paths, band_names=["blue", "green", "red"])
                >>> ds.band_names
                ['blue', 'green', 'red']

                ```
            - Mismatched grids are rejected unless ``align=True``:
                ```python
                >>> odd = os.path.join(d, "odd.tif")
                >>> _ = Dataset.from_array(
                ...     np.zeros((8, 9), dtype="int16"),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=0.5, epsg=4326
                ...     ),
                ...     path=odd,
                ... ).close()
                >>> try:
                ...     Dataset.from_band_files([paths[0], odd])
                ... except AlignmentError as exc:
                ...     print("align=True" in str(exc))
                True
                >>> aligned = Dataset.from_band_files([paths[0], odd], align=True)
                >>> aligned.band_count
                2
                >>> (aligned.rows, aligned.columns) == (
                ...     Dataset.read_file(paths[0]).rows,
                ...     Dataset.read_file(paths[0]).columns,
                ... )
                True

                ```

        See Also:
            - :meth:`align`: resample one dataset onto another's grid.
            - :meth:`from_array`: build a dataset from a numpy array.
            - :meth:`pyramids.dataset.DatasetCollection.from_files`: stack
              rasters along *time* instead of along *bands*.
        """
        resolved_paths = [str(_io._parse_path(str(p))) for p in files]
        if not resolved_paths:
            raise ValueError("from_band_files requires at least one file")

        datasets = [cls.read_file(p) for p in resolved_paths]
        for p, ds in zip(resolved_paths, datasets):
            if ds.band_count != 1:
                raise ValueError(
                    f"{p!r} has {ds.band_count} bands; from_band_files expects exactly "
                    "one band per file"
                )
            if not ds.crs:
                raise CRSError(f"{p!r} has no CRS; cannot stack rasters without a CRS")

        template = datasets[0]

        if band_names is not None:
            out_names = list(band_names)
            if len(out_names) != len(resolved_paths):
                raise ValueError(
                    f"band_names has {len(out_names)} entries but {len(resolved_paths)} "
                    "files were given"
                )
        else:
            out_names = _derive_band_names(resolved_paths)

        if no_data_value is _INHERIT_NO_DATA:
            source_nd = [ds.no_data_value[0] for ds in datasets]
            present = [v for v in source_nd if v is not None]
            if not present:
                resolved_nd: Any | None = None
            else:
                resolved_nd = source_nd[0] if source_nd[0] is not None else present[0]
                # NaN != NaN, so plain set() over-reports disagreement for
                # float-NaN sentinels (the GeoTIFF default for float rasters).
                # Normalise NaN to a single key so we only warn when distinct
                # *real* values are present.
                distinct = {
                    "__nan__" if isinstance(v, float) and np.isnan(v) else v
                    for v in present
                }
                if len(distinct) > 1:
                    warnings.warn(
                        f"source rasters disagree on no-data value ({sorted(set(present))}); "
                        f"using {resolved_nd!r}",
                        stacklevel=2,
                    )
        else:
            resolved_nd = no_data_value

        if not align:
            for p, ds in zip(resolved_paths[1:], datasets[1:]):
                if not _same_grid(template, ds):
                    raise AlignmentError(
                        f"{p!r} does not share the grid/CRS of {resolved_paths[0]!r}; "
                        "pass align=True to resample mismatched rasters onto the first "
                        "file's grid"
                    )

        # gdal.BuildVRT(separate=True) does not promote dtypes (it truncates the
        # wider bands) — take that low-memory band-by-band path only when the
        # grids already match and every input shares one dtype. Otherwise read
        # the (possibly resampled) band arrays and let numpy pick the common dtype.
        uniform_dtype = len({ds.gdal_dtype[0] for ds in datasets}) == 1

        if align or not uniform_dtype:
            # Resolve the common output dtype up front so the output can be
            # allocated once and written band-by-band, instead of reading every
            # band, np.stacking them into a second full-cube copy, and writing the
            # lot — peak drops from ~O(N·grid) to one band + the output (ARC-50).
            target_np_dtype = np.result_type(*(ds.numpy_dtype[0] for ds in datasets))
            grid_template = None
            if align:
                # Resample every input onto the first file's grid in the promoted
                # dtype. Dataset.align adopts the alignment source's dtype, so cast
                # the template first to avoid truncating wider inputs (e.g. a float
                # band onto an int template).
                # `Dataset.from_array`, not `cls.from_array`: same reason as
                # `convert_units`. On a NetCDF subclass the override returns a
                # bandless Container, and this template is then read band-wise.
                grid_template = Dataset.from_array(
                    template.read_array(band=0).astype(target_np_dtype, copy=False),
                    # epsg is None only for a no-EPSG CRS reported as such (a NetCDF
                    # geostationary grid); from_array raises CRSError on None, so
                    # fall back to the WKT. No-op for a plain Dataset (#706).
                    geo_ref=GeoReference(
                        geo=template.geotransform,
                        epsg=crs_spec(template.epsg, template.crs),
                    ),
                    no_data_value=resolved_nd,
                )
            obj = cls._build_dataset(
                template.columns,
                template.rows,
                len(resolved_paths),
                numpy_to_gdal_dtype(target_np_dtype),
                template.geotransform,
                template.crs,
                resolved_nd,
                path=path,
                array=None,
            )
            for band_i, ds_i in enumerate(datasets):
                if align and not _same_grid(template, ds_i):
                    arr = ds_i.align(grid_template).read_array(band=0)
                else:
                    # Same grid (or the non-align mixed-dtype path): just cast to
                    # the promoted dtype, which is lossless.
                    arr = ds_i.read_array(band=0).astype(target_np_dtype, copy=False)
                if align:
                    # Dataset.align fills the warp fringe with the SOURCE's sentinel;
                    # when sources disagree on nodata (first-wins resolved_nd + a
                    # UserWarning) remap so the array matches the band's declared
                    # nodata. A same-grid source skips the warp and is lossless.
                    arr = _remap_nodata_to(arr, ds_i.no_data_value[0], resolved_nd)
                obj.raster.GetRasterBand(band_i + 1).WriteArray(arr)
                del arr
            obj._raster.FlushCache()
        else:
            vrt = gdal.BuildVRT("", resolved_paths, separate=True)
            if (
                vrt is None
            ):  # pragma: no cover - BuildVRT returns None only on bad input
                raise AlignmentError(
                    f"gdal.BuildVRT could not stack {resolved_paths!r}"
                )
            if path is not None:
                # Resolve the driver from the extension like every other
                # factory. Hardcoding GTiff here is what forced the `.tif`-only
                # guard above: without it a `.nc` destination produced a GTiff
                # carrying a netCDF name, a file whose extension lies about its
                # contents. LZW is GTiff-specific, so it is applied only there.
                # Deliberately NOT `for_copy`, though this branch does use
                # `CreateCopy`. Which branch runs depends on `align` and on
                # whether the sources share a dtype -- things the caller cannot
                # easily predict -- so accepting `.png` here and refusing it on
                # the `Create` branch would make the destination's legality
                # depend on an unrelated argument. That is the same defect this
                # branch removed from `merge_rasters`; one gate for both paths.
                driver = resolve_output_driver(path)
                options = ["COMPRESS=LZW"] if driver == "GTiff" else []
                dst = gdal.GetDriverByName(driver).CreateCopy(
                    str(path), vrt, strict=1, options=options
                )
            else:
                dst = gdal.GetDriverByName(MEMORY_DRIVER).CreateCopy("", vrt, strict=1)
            vrt = None
            # BuildVRT(separate=True) carries each source band's no-data through;
            # honour an explicit override (including ``None`` = drop it).
            for i in range(dst.RasterCount):
                band = dst.GetRasterBand(i + 1)
                if resolved_nd is None:
                    band.DeleteNoDataValue()
                else:
                    band.SetNoDataValue(float(resolved_nd))
            obj = cls(dst, access="write")

        obj.band_names = out_names
        obj._raster.FlushCache()
        return obj

    @classmethod
    def from_archive(
        cls,
        url_or_path: str | Path,
        *,
        kind: str = "auto",
        member_glob: str = "*",
        band_names: list[str] | None = None,
        align: bool = False,
        no_data_value: Any = _INHERIT_NO_DATA,
        path: str | Path | None = None,
    ) -> Dataset:
        """Open every raster in an archive and merge them into one multi-band Dataset.

        Lists the archive's members (locally or over the network — a remote ZIP
        is read via the chained ``/vsizip//vsicurl/…`` path) and hands them to
        :meth:`from_band_files`. For "one Dataset per member" (a temporal stack)
        use :meth:`pyramids.dataset.DatasetCollection.from_archive` instead.

        GDAL driver ``open_options`` are **not** threaded through this
        band-stacking entry point; if a member needs a driver option, open it
        directly with :meth:`read_file` (which accepts ``open_options=``).

        The archive's file name must carry a recognised extension (``.zip`` /
        ``.tar`` / ``.tar.gz`` / ``.gz``) — GDAL's archive handlers key off the
        extension. An extension-less download URL (e.g. an Earth Engine
        ``getDownloadURL`` ending in ``:getPixels``) must first be fetched and
        saved with a ``.zip`` name (or written to ``/vsimem/<name>.zip`` via
        :func:`osgeo.gdal.FileFromMemBuffer`) before calling this.

        Args:
            url_or_path: Path or URL of the archive (``.zip`` / ``.tar`` /
                ``.tar.gz`` / ``.gz``).
            kind: Archive kind — ``"zip"``, ``"tar"`` (also ``"tar.gz"`` /
                ``"tgz"``), ``"gzip"`` (also ``"gz"``), or ``"auto"`` (default,
                infer from the extension).
            member_glob: :mod:`fnmatch` pattern selecting which members to stack.
                Default ``"*"`` (all top-level members, sorted by name). Pass e.g.
                ``"*.tif"`` for an archive that also ships sidecar files.
            band_names: Explicit per-band names; ``None`` derives them from the
                member names (see :meth:`from_band_files`).
            align: When ``True``, resample mismatched members onto the first
                member's grid instead of raising :class:`AlignmentError`.
            no_data_value: No-data value for the output bands; omitted means
                "inherit from the members".
            path: Output ``.tif`` path; ``None`` keeps the result in memory.

        Returns:
            Dataset: A multi-band dataset, one band per matching archive member.

        Raises:
            FileFormatNotSupportedError: ``kind="auto"`` and the extension is
                not recognised, or the archive could not be listed.
            FileNotFoundError: No member matched ``member_glob``.
            ValueError / AlignmentError / CRSError: As for :meth:`from_band_files`.

        Examples:
            - Stack the raster members of a local ZIP into one multi-band dataset
              (band names come from the member names):
                ```python
                >>> import os, tempfile, zipfile
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> d = tempfile.mkdtemp()
                >>> members = []
                >>> pairs = [("scene.B2.tif", 2), ("scene.B3.tif", 3)]
                >>> for name, val in pairs:
                ...     p = os.path.join(d, name)
                ...     _ = Dataset.from_array(
                ...         np.full((4, 5), val, dtype="int16"),
                ...         geo_ref=GeoReference(
                ...             top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...         ),
                ...         path=p,
                ...     ).close()
                ...     members.append(p)
                >>> zip_path = os.path.join(d, "download.zip")
                >>> with zipfile.ZipFile(zip_path, "w") as zf:
                ...     for m in members:
                ...         zf.write(m, arcname=os.path.basename(m))
                >>> ds = Dataset.from_archive(zip_path, member_glob="*.tif")
                >>> ds.band_count
                2
                >>> ds.band_names
                ['B2', 'B3']
                >>> [int(ds.read_array(band=i).flat[0]) for i in range(2)]
                [2, 3]

                ```

        See Also:
            - :meth:`from_band_files`: stack a known list of single-band rasters.
            - :meth:`pyramids.dataset.DatasetCollection.from_archive`: open each
              member as a separate timestep instead of merging them into bands.
        """
        dir_vsi = _io._archive_dir_vsi(url_or_path, kind)
        members = _io._archive_members(dir_vsi, member_glob)
        member_paths = [f"{dir_vsi}/{m}" for m in members]
        return cls.from_band_files(
            member_paths,
            band_names=band_names,
            align=align,
            no_data_value=no_data_value,
            path=path,
        )
