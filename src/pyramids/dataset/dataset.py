"""
Dataset module.

raster contains python functions to handle raster data align them together based on a source raster, perform any
algebraic operation on cell's values.
"""

from __future__ import annotations

import logging
import warnings
import weakref
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
from osgeo import gdal

from pyramids import _io
from pyramids.base._errors import AlignmentError, CRSError
from pyramids.base._utils import (
    DTYPE_CONVERSION_DF,
    UNDEFINED_COLOR_INTERP,
    numpy_to_gdal_dtype,
)
from pyramids.base.crs import epsg_from_wkt, sr_from_epsg
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
# `Dataset.__init__` to wire the seven collaborators and by
# `_update_inplace` to re-bind their `_ds` back-references after
# `__dict__.update` (see audit §3.3).
_COLLABORATOR_ATTRS = ("io", "spatial", "bands", "analysis", "cell", "vectorize", "cog")

# Sentinel for `Dataset.from_band_files(no_data_value=...)` so the helper can
# tell "caller didn't pass one — inherit from the source rasters" apart from
# "caller explicitly passed `None`" (which means "stamp no no-data sentinel").
_INHERIT_NO_DATA = object()

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


def _remap_nodata_to(arr: np.ndarray, src_nd: Any, dst_nd: Any) -> np.ndarray:
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
    from geopandas import GeoDataFrame


class Dataset(RasterBase):
    """Single-band or multi-band raster dataset (GeoTIFF, etc.).

    Wraps a GDAL dataset with spatial operations (crop, reproject, align,
    mosaic), band-level I/O, and no-data handling. For NetCDF files use
    the :class:`~pyramids.netcdf.NetCDF` subclass; for temporal stacks of
    rasters use :class:`~pyramids.dataset.DatasetCollection`.

    The seven public-API families are exposed as collaborator instances
    (`ds.io`, `ds.spatial`, `ds.bands`, `ds.analysis`,
    `ds.cell`, `ds.vectorize`, `ds.cog`) and via thin facade
    methods on the Dataset itself, so `ds.crop(mask)` and
    `ds.spatial.crop(mask)` are equivalent. Each collaborator holds a
    weakref proxy back to the Dataset; the proxy keeps GDAL handle
    release deterministic on Windows.
    """

    def __init__(self, src: gdal.Dataset, access: str = "read_only"):
        """__init__."""
        self.logger = logging.getLogger(__name__)
        super().__init__(src, access=access)

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
        # Re-bind via `weakref.proxy` so the back-reference stays
        # weak after the dict swap (matches `_Engine.__init__`).
        # Direct slot reassignment is allowed because `_Engine`
        # declares `_ds` as its only slot — see method docstring.
        self_proxy = weakref.proxy(self)
        for attr in _COLLABORATOR_ATTRS:
            collab = self.__dict__.get(attr)
            if collab is not None:
                collab._ds = self_proxy

    def focal_mean(self, radius: int = 1, *, chunks=None, band: int = 0):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.focal_mean`."""
        return focal_mean(self, radius=radius, chunks=chunks, band=band)

    def focal_std(self, radius: int = 1, *, chunks=None, band: int = 0):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.focal_std`."""
        return focal_std(self, radius=radius, chunks=chunks, band=band)

    def focal_apply(self, func, radius: int = 1, *, chunks=None, band: int = 0):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.focal_apply`."""
        return focal_apply(self, func, radius=radius, chunks=chunks, band=band)

    def slope(self, *, chunks=None, band: int = 0, units: str = "degrees"):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.slope`."""
        return slope(self, chunks=chunks, band=band, units=units)

    def aspect(self, *, chunks=None, band: int = 0):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.aspect`."""
        return aspect(self, chunks=chunks, band=band)

    def hillshade(
        self,
        *,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        chunks=None,
        band: int = 0,
    ):
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

    def cluster2(self, *args, **kwargs):
        """Facade — delegates to :meth:`Vectorize.cluster2 <pyramids.dataset.engines.Vectorize.cluster2>`."""
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
        3. If the dataset has 3+ bands but **no** band has a GDAL ``ColorInterpretation``
           set (i.e. every band reports ``undefined``), return ``(0, rgb)``. This is the
           D-1 fix: ``band_count >= 3`` alone is not a sufficient signal that the data
           is an RGB image — multi-band scalar cubes (e.g. time series stacked into one
           GeoTIFF) also have ``band_count >= 3`` and must not be misinterpreted as RGB.
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
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.rand(4, 8, 8).astype(np.float32)
              >>> ds = Dataset.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=0.1, epsg=4326,
              ... )
              >>> ds._resolve_plot_band(band=2, rgb=None)
              (2, None)

              ```

            - Single-band raster falls back to band ``0`` (rule 2):

              ```python
              >>> single = np.random.rand(6, 6).astype(np.float32)
              >>> ds_1band = Dataset.create_from_array(
              ...     single, top_left_corner=(0, 0), cell_size=0.1, epsg=4326,
              ... )
              >>> ds_1band._resolve_plot_band(band=None, rgb=None)
              (0, None)

              ```

            - Multi-band dataset with no ``ColorInterpretation`` defaults to band ``0``
              (rule 3, the D-1 fix). ``Dataset.create_from_array`` produces a multi-band
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
            band_colors = list(self.band_color.values())
            has_color_interp = any(c != UNDEFINED_COLOR_INTERP for c in band_colors)
            if not has_color_interp:
                resolved_band = 0
                resolved_rgb = rgb
            else:
                if rgb is None:
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
                            stacklevel=3,
                        )
                        resolved_rgb = [2, 1, 0]
                    else:
                        resolved_rgb = [int(v) for v in candidate]
                else:
                    resolved_rgb = rgb
                resolved_band = int(resolved_rgb[0])
        return resolved_band, resolved_rgb

    def plot(
        self,
        band: int | None = None,
        exclude_value: Any | None = None,
        rgb: list[int] | None = None,
        surface_reflectance: int | None = None,
        cutoff: list | None = None,
        overview: bool | None = False,
        overview_index: int | None = 0,
        percentile: int | None = None,
        basemap: bool | str | None = None,
        rgb_options: dict | None = None,
        **kwargs: Any,
    ):
        """Plot the values/overviews of a band.

        Facade for :meth:`Analysis.plot <pyramids.dataset.engines.Analysis.plot>`. Resolves
        the band index via :meth:`_resolve_plot_band` (GeoTIFF/Sentinel semantics) and then
        forwards the call to the generic rendering engine.

        When ``band`` is ``None`` and the dataset looks like an RGB image — i.e. it has
        at least 3 bands **and** at least one band has a GDAL ``ColorInterpretation`` set —
        the red band is auto-selected (either from ``rgb[0]`` or by resolving the colour
        tags). Otherwise the facade defaults to band ``0``. See
        :meth:`Analysis.plot` for the full kwargs surface.

        The four satellite-imagery kwargs ``rgb``, ``surface_reflectance``, ``cutoff``,
        and ``percentile`` may be grouped under a single ``rgb_options=`` dict
        (recommended) or passed loose at the top level (deprecated; emits
        :class:`DeprecationWarning`). When both forms are mixed, the values inside
        ``rgb_options`` win.

        Args:
            band (int, optional):
                Band index to render. When ``None``, the index is resolved by
                :meth:`_resolve_plot_band`.
            exclude_value (Any, optional):
                Pixel value to mask out before plotting. Default is ``None``.
            rgb (list[int], optional):
                **Deprecated**; pass via ``rgb_options={"rgb": [...]}`` instead.
                Three- or four-element list of band indices ``[r, g, b]`` (optionally
                ``[r, g, b, a]``) to render the dataset as a true-colour composite.
                Only honoured when the dataset has at least 3 bands and at least one
                band reports a colour interpretation. Default is ``None``.
            surface_reflectance (int, optional):
                **Deprecated**; pass via ``rgb_options={"surface_reflectance": ...}``.
                Surface-reflectance scale factor used to normalise satellite reflectance
                bands (typically ``10000`` for Sentinel-2). Default is ``None``.
            cutoff (list, optional):
                **Deprecated**; pass via ``rgb_options={"cutoff": ...}``.
                Per-band clip values used when rendering RGB composites. Default is
                ``None``.
            overview (bool, optional):
                If ``True``, plot the overview pyramid level instead of the full-resolution
                array. Default is ``False``.
            overview_index (int, optional):
                Index of the overview level to plot when ``overview=True``. Default is ``0``.
            percentile (int, optional):
                **Deprecated**; pass via ``rgb_options={"percentile": ...}``.
                Percentile used when computing colour-scale limits. Default is ``None``.
            basemap (bool or str, optional):
                If ``True``, overlay an OpenStreetMap basemap. If a string, use it as the
                contextily/xyzservices tile-provider name (e.g. ``"CartoDB.Positron"``).
                Default is ``None``. Requires the ``[viz]`` extra.
            rgb_options (dict, optional):
                Grouped Sentinel-imagery kwargs. Accepted keys:
                ``"rgb"``, ``"surface_reflectance"``, ``"cutoff"``,
                ``"percentile"``. Recommended over the loose top-level
                kwargs (which emit :class:`DeprecationWarning`). Default
                is ``None``.
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
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.rand(8, 8).astype(np.float32)
              >>> ds = Dataset.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=0.1, epsg=4326,
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
              >>> rgb_ds = Dataset.create_from_array(
              ...     arr3, top_left_corner=(0, 0), cell_size=0.1, epsg=4326,
              ... )
              >>> cleo = rgb_ds.plot(  # doctest: +SKIP
              ...     rgb_options={"rgb": [0, 1, 2], "surface_reflectance": 255},
              ... )

              ```

            - The deprecated loose-kwarg form still works but emits a
              :class:`DeprecationWarning`. New code should prefer the
              grouped ``rgb_options=`` form shown above:

              ```python
              >>> cleo = rgb_ds.plot(  # doctest: +SKIP
              ...     rgb=[0, 1, 2], surface_reflectance=255,
              ... )
              DeprecationWarning: Passing `rgb=`, `surface_reflectance=`...

              ```
        """
        rgb, surface_reflectance, cutoff, percentile = self._merge_rgb_options(
            rgb_options=rgb_options,
            rgb=rgb,
            surface_reflectance=surface_reflectance,
            cutoff=cutoff,
            percentile=percentile,
        )
        resolved_band, resolved_rgb = self._resolve_plot_band(band, rgb)
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
            **kwargs,
        )

    @staticmethod
    def _merge_rgb_options(
        *,
        rgb_options: dict | None,
        rgb: list[int] | None,
        surface_reflectance: int | None,
        cutoff: list | None,
        percentile: int | None,
    ) -> tuple[list[int] | None, int | None, list | None, int | None]:
        """Merge `rgb_options=` with the deprecated loose top-level kwargs.

        Returns the resolved four-tuple ``(rgb, surface_reflectance,
        cutoff, percentile)`` passed forward to the renderer. Values in
        ``rgb_options`` win over the loose kwargs on collision; using
        any of the four loose kwargs emits a :class:`DeprecationWarning`
        recommending the grouped form.

        Args:
            rgb_options: Recommended grouped form (``{"rgb": ..., ...}``).
                Accepted keys: ``"rgb"``, ``"surface_reflectance"``,
                ``"cutoff"``, ``"percentile"``. ``None`` means
                no grouped options were supplied.
            rgb: Deprecated top-level kwarg.
            surface_reflectance: Deprecated top-level kwarg.
            cutoff: Deprecated top-level kwarg.
            percentile: Deprecated top-level kwarg.

        Returns:
            tuple: ``(rgb, surface_reflectance, cutoff, percentile)``
                resolved values, with the grouped form taking precedence.

        Raises:
            ValueError: If ``rgb_options`` contains a key outside the
                accepted set ``{"rgb", "surface_reflectance", "cutoff",
                "percentile"}``.

        Examples:
            - Pass everything through the grouped form (recommended).
              No warnings are emitted and the returned four-tuple
              mirrors the inputs in order:

                ```python
                >>> import warnings
                >>> from pyramids.dataset import Dataset
                >>> with warnings.catch_warnings(record=True) as caught:
                ...     warnings.simplefilter("always")
                ...     result = Dataset._merge_rgb_options(
                ...         rgb_options={
                ...             "rgb": [2, 1, 0],
                ...             "surface_reflectance": 10000,
                ...         },
                ...         rgb=None,
                ...         surface_reflectance=None,
                ...         cutoff=None,
                ...         percentile=None,
                ...     )
                >>> result
                ([2, 1, 0], 10000, None, None)
                >>> [w.category.__name__ for w in caught]
                []

                ```

            - Passing a value via the loose top-level kwarg path emits
              a :class:`DeprecationWarning` that names the kwarg(s)
              used and points to the grouped replacement:

                ```python
                >>> import warnings
                >>> from pyramids.dataset import Dataset
                >>> with warnings.catch_warnings(record=True) as caught:
                ...     warnings.simplefilter("always")
                ...     result = Dataset._merge_rgb_options(
                ...         rgb_options=None,
                ...         rgb=[2, 1, 0],
                ...         surface_reflectance=None,
                ...         cutoff=None,
                ...         percentile=None,
                ...     )
                >>> result
                ([2, 1, 0], None, None, None)
                >>> caught[0].category.__name__
                'DeprecationWarning'

                ```

            - When both forms collide, ``rgb_options`` wins. A
              :class:`DeprecationWarning` is still emitted for the loose
              kwarg:

                ```python
                >>> import warnings
                >>> from pyramids.dataset import Dataset
                >>> with warnings.catch_warnings(record=True) as caught:
                ...     warnings.simplefilter("always")
                ...     result = Dataset._merge_rgb_options(
                ...         rgb_options={"rgb": [0, 1, 2]},
                ...         rgb=[3, 4, 5],
                ...         surface_reflectance=None,
                ...         cutoff=None,
                ...         percentile=None,
                ...     )
                >>> result
                ([0, 1, 2], None, None, None)

                ```

            - An unknown key in ``rgb_options`` raises
              :class:`ValueError`:

                ```python
                >>> from pyramids.dataset import Dataset
                >>> Dataset._merge_rgb_options(  # doctest: +IGNORE_EXCEPTION_DETAIL
                ...     rgb_options={"unknown": 1},
                ...     rgb=None,
                ...     surface_reflectance=None,
                ...     cutoff=None,
                ...     percentile=None,
                ... )
                Traceback (most recent call last):
                    ...
                ValueError: Unknown keys in `rgb_options`: ['unknown']...

                ```
        """
        loose_pairs = {
            "rgb": rgb,
            "surface_reflectance": surface_reflectance,
            "cutoff": cutoff,
            "percentile": percentile,
        }
        opts = rgb_options or {}
        unknown = set(opts) - set(loose_pairs)
        if unknown:
            raise ValueError(
                f"Unknown keys in `rgb_options`: {sorted(unknown)}. "
                f"Accepted: {sorted(loose_pairs)}."
            )
        used_loose = [name for name, value in loose_pairs.items() if value is not None]
        # Split the deprecated loose kwargs into those overridden by a
        # matching `rgb_options` key (a collision — the loose value is
        # discarded) and those used on their own. They get distinct
        # warning text so the message isn't misleading: a user who *did*
        # use the grouped form shouldn't be told "group them" again.
        overridden = [name for name in used_loose if name in opts]
        pure_loose = [name for name in used_loose if name not in opts]
        if pure_loose:
            warnings.warn(
                "Passing "
                + ", ".join(f"`{name}=`" for name in pure_loose)
                + " as top-level kwargs to `Dataset.plot` is deprecated. "
                "Group them under `rgb_options={...}` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        if overridden:
            warnings.warn(
                "Both the loose "
                + ", ".join(f"`{name}=`" for name in overridden)
                + " kwarg(s) and `rgb_options` were given for the same key(s); "
                "`rgb_options` wins — drop the loose form.",
                DeprecationWarning,
                stacklevel=3,
            )
        rgb = opts.get("rgb", rgb)
        surface_reflectance = opts.get("surface_reflectance", surface_reflectance)
        cutoff = opts.get("cutoff", cutoff)
        percentile = opts.get("percentile", percentile)
        return rgb, surface_reflectance, cutoff, percentile

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
        self.bands.band_color = values

    @property
    def color_table(self):
        """Facade — delegates to :attr:`Bands.color_table <pyramids.dataset.engines.Bands.color_table>`."""
        return self.bands.color_table

    @color_table.setter
    def color_table(self, df):
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
        x_min, y_max = self.top_left_corner
        y_min = y_max - self.rows * self.cell_size
        x_max = x_min + self.columns * self.cell_size
        return [x_min, y_min, x_max, y_max]

    def _calculate_bounds(self):
        """Concrete override of :meth:`RasterBase._calculate_bounds`."""
        x_min, y_min, x_max, y_max = self._calculate_bbox()
        coords = [(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max)]
        poly = create_polygon(coords)
        gdf = gpd.GeoDataFrame(geometry=[poly])
        gdf.set_crs(epsg=self.epsg, inplace=True)
        return gdf

    def _get_band_names(self):
        """Concrete override of :meth:`RasterBase._get_band_names`.

        Defined directly on Dataset (not via the bands collaborator)
        because `Dataset.__init__` calls `self._get_band_names()`
        before the `Bands` collaborator is wired up. Mirrors
        :meth:`Bands._get_band_names`.
        """
        names = []
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
        `_get_crs` body is the same one-liner.
        """
        return str(self.raster.GetProjection())

    def _get_epsg(self) -> int:
        """Concrete override of :meth:`RasterBase._get_epsg`.

        Defined directly on Dataset for the same reason as
        :meth:`_get_crs`.
        """
        return epsg_from_wkt(self._get_crs())

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
        """__str__."""
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
        """__repr__."""
        return str(gdal.Info(self.raster))

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
    def geotransform(self) -> tuple[float, float, float, float, float, float]:
        """WKT projection.

        (top left corner X/lon coordinate, cell_size, 0, top left corner y/lat coordinate, 0, -cell_size).

        See Also:
            - Dataset.top_left_corner: Coordinate of the top left corner of the dataset.
            - Dataset.epsg: EPSG number of the dataset coordinate reference system.
        """
        gt: tuple[float, float, float, float, float, float] = self._geotransform
        return gt

    @property
    def epsg(self) -> int:
        """EPSG number."""
        return self._epsg

    @epsg.setter
    def epsg(self, value: int):
        """EPSG number."""
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
        """Band names."""
        self.bands._set_band_names(name_list)

    @property
    def band_units(self) -> list[str]:
        """Band units."""
        return self._band_units

    @band_units.setter
    def band_units(self, value: list[str]):
        """Band units setter."""
        self._band_units = value
        for i, val in enumerate(value):
            self._iloc(i).SetUnitType(val)

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
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.array([[273.15, 283.15], [293.15, 303.15]]),
                ...     top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
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
                >>> ds = Dataset.create_from_array(
                ...     np.array([[273.15]]), top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
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
        out = stack.astype("float64").copy()
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
        result = self.create_from_array(
            result_array,
            geo=self.geotransform,
            epsg=self.epsg,
            no_data_value=list(no_data),
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
        """Meta-data."""
        for key, val in value.items():
            self._raster.SetMetadataItem(key, val)

    @property
    def block_size(self) -> list[tuple[int, int]]:
        """Block Size.

        The block size is the size of the block that the raster is divided into, the block size is used to
        read and write the raster data in blocks.

        See Also:
            - Dataset.get_block_arrangement: Get block arrangement to read the dataset in chunks.
            - Dataset.get_tile: Get tiles.
            - Dataset.read_array: Read the data stored in the dataset bands.
        """
        return self._block_size

    @block_size.setter
    def block_size(self, value: list[tuple[int, int]]):
        """Block Size.

        Args:
            value (List[Tuple[int, int]]):
                block size for each band in the raster(512, 512).
        """
        if len(value[0]) != 2:
            raise ValueError("block size should be a tuple of 2 integers")

        self._block_size = value

    @property
    def file_name(self):
        """File name."""
        return super().file_name

    @property
    def driver_type(self):
        """Driver Type."""
        return super().driver_type

    @property
    def scale(self) -> list[float]:
        """Scale.

        The value of the scale is used to convert the pixel values to the real-world values.
        """
        scale_list = []
        for i in range(self.band_count):
            band_scale = self._iloc(i).GetScale()
            scale_list.append(band_scale if band_scale is not None else 1.0)
        return scale_list

    @scale.setter
    def scale(self, value: list[float]):
        """Scale."""
        for i, val in enumerate(value):
            self._iloc(i).SetScale(val)

    @property
    def offset(self):
        """Offset.

        The value of the offset is used to convert the pixel values to the real-world values.
        """
        offset_list = []
        for i in range(self.band_count):
            band_offset = self._iloc(i).GetOffset()
            offset_list.append(band_offset if band_offset is not None else 0)
        return offset_list

    @offset.setter
    def offset(self, value: list[float]):
        """Offset."""
        for i, val in enumerate(value):
            self._iloc(i).SetOffset(val)

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
    def total_bounds(self) -> np.ndarray:
        """Bounding box `[minx, miny, maxx, maxy]` as a NumPy array.

        introduced this property so that `Dataset` and
        :class:`pyramids.feature.FeatureCollection` expose the same
        shape (`GeoDataFrame.total_bounds` is the geopandas name
        for exactly this array), letting both classes satisfy the
        :class:`pyramids.base.protocols.SpatialObject` protocol.
        """
        return np.asarray(self._calculate_bbox())

    @property
    def lon(self) -> np.ndarray:
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
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.zeros((2, 3)), top_left_corner=(0.0, 0.0), cell_size=0.5, epsg=4326,
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
    def lat(self) -> np.ndarray:
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
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.zeros((2, 3)), top_left_corner=(0.0, 0.0), cell_size=0.5, epsg=4326,
                ... )
                >>> ds.lat.tolist()
                [-0.25, -0.75]

                ```
            - With non-square cells the latitude axis uses the pixel height, not the
              pixel width:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.zeros((2, 3)), geo=(10.0, 2.0, 0.0, 50.0, 0.0, -1.0), epsg=4326,
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
    def x(self) -> np.ndarray:
        """X cell-centre coordinates (alias of :attr:`lon`).

        Examples:
            - x mirrors lon for the same raster:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.zeros((2, 3)), top_left_corner=(0.0, 0.0), cell_size=0.5, epsg=4326,
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
    def y(self) -> np.ndarray:
        """Y cell-centre coordinates (alias of :attr:`lat`).

        Examples:
            - y mirrors lat for the same raster:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.zeros((2, 3)), top_left_corner=(0.0, 0.0), cell_size=0.5, epsg=4326,
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
    ) -> Dataset:
        """Open a raster from a path, URL, or archive member.

        Plain local paths, ``/vsi*`` paths, and URL schemes
        (``http(s)://``, ``s3://``, ``gs://``, ``az://`` / ``abfs://``,
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

        Returns:
            Dataset:
                Opened dataset instance.

        See Also:
            - :meth:`read_array`: read the values stored in a dataset band.
            - :meth:`from_bytes`: open a raster held in memory.
            - :meth:`pyramids.dataset.DatasetCollection.from_archive`: open
              *every* member of an archive as a temporal stack.
        """
        src = _io.read_file(path, read_only=read_only, file_i=file_i, vsi=vsi)
        return cls(src, access="read_only" if read_only else "write")

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
        ``s3://``, ``gs://``, ``az://`` / ``abfs://`` and ``file://``
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

    def copy(self, path: str | Path | None = None) -> Dataset:
        """Deep copy.

        Args:
            path (str, optional):
                Destination path to save the copied dataset. If None
                is passed, the copied dataset is created in memory.

        Returns:
            Dataset: An independent copy. Access mode of the returned
            Dataset:

            * `path is None` (in-memory copy) → access mode of the
              source is preserved. A `copy()` of a read-only source
              stays read-only at the pyramids level (the underlying
              MEM driver is always writable; pyramids enforces the
              flag itself).
            * `path is not None` (on-disk copy) → `"write"`,
              because the caller has just created a new file they
              presumably want to populate.
        """
        if path is None:
            path = ""
            driver = "MEM"
            new_access = self._access
        else:
            driver = "GTiff"
            new_access = "write"

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
        driver: str = "MEM",
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
            driver (str):
                Driver type ["GTiff", "MEM"].
            path (str):
                Path to save the GTiff driver.
            options (list[str] | None):
                GDAL creation options for the disk driver (e.g.
                ``["TILED=YES", "SPARSE_OK=TRUE", "BIGTIFF=YES"]``). When
                `None` (default), GTiff falls back to ``["COMPRESS=LZW"]`` —
                the historical behaviour. Ignored for the MEM driver, which
                takes no creation options.

        Returns:
            gdal driver
        """
        if path:
            driver = "GTiff" if driver == "MEM" else driver
            if not isinstance(path, (str, Path)):
                raise TypeError(
                    f"The path input should be string or Path type, given: {type(path)}"
                )
            path = Path(path)
            if driver == "GTiff" and path.suffix != ".tif":
                raise TypeError(
                    "The path to save the created raster should end with .tif"
                )
            # LZW is a lossless compression method achieve the highest compression but with a lot
            # of computations. Callers that need tiled / sparse / BigTIFF
            # output (e.g. create_empty) pass their own options.
            creation_options = ["COMPRESS=LZW"] if options is None else options
            src = gdal.GetDriverByName(driver).Create(
                str(path), cols, rows, bands, dtype, creation_options
            )
        else:
            # for memory drivers
            driver = "MEM"
            src = gdal.GetDriverByName(driver).Create("", cols, rows, bands, dtype)
        return src

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
        driver: str = "MEM",
        path: str | Path | None = None,
        access: str = "write",
        array: np.ndarray | None = None,
        options: list[str] | None = None,
    ) -> Dataset:
        """Build a Dataset: allocate, set geo/CRS, optionally fill no-data, optionally write.

        Single canonical factory for raster construction. Consolidates the
        ``_create_dataset + SetGeoTransform + SetProjection + wrap +
        _set_no_data_value (+ WriteArray)` pattern that `create``,
        `create_from_array`, `dataset_like`, and the per-op factories
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
            driver: GDAL driver type. Default `"MEM"`.
            path: Path for disk-based drivers. `None` keeps the
                dataset in memory.
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
        dst = cls._create_dataset(
            cols, rows, bands, dtype, driver=driver, path=path, options=options
        )
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
        cell_size: int | float,
        rows: int,
        columns: int,
        dtype: str,
        bands: int,
        top_left_corner: tuple,
        epsg: int,
        no_data_value: Any | None = None,
        path: str | Path | None = None,
    ) -> Dataset:
        """Create a new dataset and fill it with the no_data_value.

        The new dataset will have an array filled with the no_data_value.

        Args:
            cell_size (int|float):
                Cell size.
            rows (int):
                Number of rows.
            columns (int):
                Number of columns.
            dtype (str):
                Data type.
            bands (int|None):
                Number of bands to create in the output raster.
            top_left_corner (Tuple):
                Coordinates of the top left corner point.
            epsg (int):
                EPSG number to identify the projection of the coordinates in the created raster.
            no_data_value (float|None):
                No data value.
            path (str, optional):
                Path on disk; if None, the dataset is created in memory. Default is None.

        Returns:
            Dataset: A new dataset
        """
        gdal_dtype = numpy_to_gdal_dtype(dtype)
        crs_wkt = sr_from_epsg(epsg).ExportToWkt()
        geotransform = (
            top_left_corner[0],
            cell_size,
            0,
            top_left_corner[1],
            0,
            -1 * cell_size,
        )
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
        geo: tuple[float, float, float, float, float, float] | None = None,
        epsg: int = 4326,
        no_data_value: Any = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "GTiff",
        path: str | Path | None = None,
        options: list[str] | None = None,
    ) -> Dataset:
        """Allocate an empty, header-only raster without materialising a full array.

        Out-of-core algorithms allocate the output once and scatter result
        windows into it with
        ``write_array(array, window=Window(col_off, row_off, cols, rows))``
        (see :class:`~pyramids.dataset.window.Window`).
        For the default ``driver_type="GTiff"`` the file is **tiled, sparse,
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
            geo: Geotransform
                ``(top_left_x, pixel_w, row_skew, top_left_y, col_skew,
                pixel_h)``. Default ``(0.0, 1.0, 0.0, 0.0, 0.0, -1.0)`` — a
                unit-pixel grid with the origin at ``(0, 0)``.
            epsg: EPSG code for the projection. Default 4326.
            no_data_value: No-data sentinel stamped on every band at
                creation. Default :data:`DEFAULT_NO_DATA_VALUE`. Keep it set
                so sparse unwritten blocks read back as no-data rather than 0.
                Passing ``None`` skips the band fill and stamps no sentinel,
                which opts out of that guarantee — a sparse GTiff's unwritten
                blocks then read back as **0**, not no-data. On the disk/GTiff
                path this emits a :class:`NoDataSentinelWarning`; the in-RAM
                ``"MEM"`` driver is dense and does not warn.
            driver_type: GDAL driver. ``"GTiff"`` (default) writes a
                disk-backed file and requires `path`; ``"MEM"`` keeps the
                raster in RAM and requires `path` to be `None`. Note that any
                non-`None` `path` produces a GTiff regardless of `driver_type`
                — the underlying allocator promotes ``"MEM"`` + `path` to
                GTiff.
            path: Output path (``.tif``) for a disk-backed raster. Pass a path
                for the GTiff driver; leave as `None` for an in-memory
                ``"MEM"`` raster.
            options: GDAL creation options. `None` (default) uses
                :data:`OUT_OF_CORE_CREATION_OPTIONS` for GTiff. Override to
                align ``BLOCKXSIZE`` / ``BLOCKYSIZE`` to your tile size or to
                change compression. Applies only to the disk/GTiff driver;
                passing `options` without a `path` raises rather than silently
                dropping them.

        Returns:
            Dataset: An empty raster whose bands read back as `no_data_value`
            before any write. On GTiff this is sparse — SPARSE_OK keeps
            never-written blocks unallocated and GDAL returns the no-data
            sentinel for them; on MEM every band is filled with `no_data_value`
            at allocation, so unwritten MEM cells read back as no-data too.

        Raises:
            ValueError: ``options`` is given without a ``path`` (creation
                options apply only to the disk/GTiff driver); or
                ``driver_type="GTiff"`` (the default) is requested
                without a `path`. Pass a `path`, or use ``driver_type="MEM"``
                for an in-memory raster.

        Examples:
            - Allocate an in-memory empty raster and read its no-data metadata:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_empty(
                ...     4, 5, dtype="float32", no_data_value=-9999.0, driver_type="MEM"
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
                >>> ds = Dataset.create_empty(4, 4, dtype="float32", driver_type="MEM")
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
        if driver_type == "GTiff" and path is None:
            raise ValueError(
                "create_empty(driver_type='GTiff') needs a path to write the raster "
                "to; pass path='out.tif' for a disk-backed raster, or "
                "driver_type='MEM' for an in-memory one. (Without a path the GTiff "
                "tiled/sparse/BigTIFF options would be silently dropped.)"
            )
        # Creation options apply only to the disk/GTiff driver (path given); the
        # MEM driver takes none. Reject explicit options that would be dropped
        # rather than silently ignoring them.
        if options is not None and path is None:
            raise ValueError(
                "create_empty received `options` but no `path`: GDAL creation "
                "options apply only to the disk/GTiff driver. Pass a `path`, or drop "
                "`options` for the in-memory MEM raster."
            )
        # Only the disk/GTiff path is sparse, where a missing sentinel makes
        # never-written blocks read back as 0 instead of no-data. The MEM driver
        # (path is None) is a dense in-RAM buffer where a sentinel-free raster is
        # an ordinary, unsurprising choice — don't warn there. (A non-None path
        # always yields a GTiff, including the MEM+path promotion.)
        if no_data_value is None and path is not None:
            warnings.warn(
                "create_empty(no_data_value=None) on a disk/GTiff target stamps no "
                "no-data sentinel, so unwritten sparse blocks read back as 0, not "
                "no-data. Pass a no_data_value to keep the 'unwritten == no-data' "
                "guarantee.",
                NoDataSentinelWarning,
                stacklevel=2,
            )
        gdal_dtype = numpy_to_gdal_dtype(dtype)
        crs_wkt = sr_from_epsg(epsg).ExportToWkt()
        if geo is None:
            geo = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
        if options is None and driver_type == "GTiff":
            options = list(OUT_OF_CORE_CREATION_OPTIONS)
        return cls._build_dataset(
            cols,
            rows,
            bands,
            gdal_dtype,
            geo,
            crs_wkt,
            no_data_value,
            driver=driver_type,
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
        the size of an input DEM without materialising it. Backed by GTiff
        when `path` is given (tiled / sparse / BigTIFF via
        :data:`OUT_OF_CORE_CREATION_OPTIONS`), otherwise MEM.

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
                on the disk/GTiff path (``path`` given) this emits a
                :class:`NoDataSentinelWarning` (the in-RAM MEM result does not
                warn).
            path: Output path (``.tif``) for a disk-backed raster. `None`
                (default) keeps the raster in memory (MEM driver).
            options: GDAL creation options for the GTiff driver. `None`
                (default) uses :data:`OUT_OF_CORE_CREATION_OPTIONS`. Applies
                only to the disk/GTiff driver; passing `options` without a
                `path` raises rather than silently dropping them.

        Returns:
            Dataset: An empty raster matching the template's footprint.

        Raises:
            ValueError: ``options`` is given without a ``path`` (creation
                options apply only to the disk/GTiff driver).

        Examples:
            - Allocate an empty raster shaped like an existing one, with a
              different dtype:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> template = Dataset.create_from_array(
                ...     np.ones((3, 4, 5), dtype="float32"),
                ...     top_left_corner=(0.0, 10.0), cell_size=0.5, epsg=4326,
                ...     no_data_value=-9999.0,
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
                >>> from pyramids.dataset import Dataset
                >>> template = Dataset.create_from_array(
                ...     np.ones((3, 4, 4), dtype="float32"),
                ...     top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326,
                ...     no_data_value=-9999.0,
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
                "apply only to the disk/GTiff driver. Pass a `path`, or drop "
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
        # Warn only for the disk/GTiff target (path given), where a missing
        # sentinel makes unwritten sparse blocks read back as 0. An in-RAM MEM
        # result (no path) is dense and a sentinel-free raster is unsurprising.
        if nodata is None and path is not None:
            warnings.warn(
                "empty_like produced a disk/GTiff raster with no no-data sentinel "
                "(no_data_value resolved to None, explicitly or inherited from a "
                "template with no no-data), so unwritten sparse blocks read back as "
                "0, not no-data. Pass no_data_value to keep the 'unwritten == "
                "no-data' guarantee.",
                NoDataSentinelWarning,
                stacklevel=2,
            )
        driver_type = "GTiff" if path is not None else "MEM"
        if options is None and driver_type == "GTiff":
            options = list(OUT_OF_CORE_CREATION_OPTIONS)
        return cls._build_dataset(
            template.columns,
            template.rows,
            n_bands,
            gdal_dtype,
            template.geotransform,
            template.crs,
            nodata,
            driver=driver_type,
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
        column_name: str | list[str] | None = None,
    ) -> Dataset:
        """Rasterize a :class:`FeatureCollection` into a new :class:`Dataset`.

        Burns the values from `column_name` (or every attribute
        column if `None`) into a single-band or multi-band raster.
        When a `template` Dataset is given, the output adopts its
        geotransform, cell size, row/column count, and no-data value.
        Otherwise `cell_size` controls the resolution and the extent
        is derived from :attr:`FeatureCollection.total_bounds`.

        Args:
            features (FeatureCollection):
                The vector to rasterize.
            cell_size (int | float | None):
                Cell size for the new raster. Required unless
                `template` is given.
            template (Dataset | None):
                Optional template raster. When supplied, the output
                inherits its geotransform and no-data value.
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
                `column_name` empty or referencing missing columns.
            TypeError: `template` is not a Dataset, or
                `column_name` is not `str` / `list` / `None`.
            CRSError: `features.epsg` is `None`, or
                `template.epsg!= features.epsg`.
        """
        return rasterize_features(
            features,
            cls,
            cell_size=cell_size,
            template=template,
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
    def create_from_array(  # type: ignore[override]
        cls,
        arr: np.ndarray,
        top_left_corner: tuple[float, float] | None = None,
        cell_size: int | float | None = None,
        geo: tuple[float, float, float, float, float, float] | None = None,
        epsg: str | int = 4326,
        no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "MEM",
        path: str | Path | None = None,
    ) -> Dataset:
        """Create a new dataset from an array.

        Args:
            arr (np.ndarray):
                Numpy array.
            top_left_corner (Tuple[float, float], optional):
                The coordinates of the top left corner of the dataset.
            cell_size (int|float, optional):
                Cell size in the same units of the coordinate reference system defined by the `epsg`
                parameter.
            geo (Tuple[float, float, float, float, float, float], optional):
                Geotransform tuple (minimum lon/x, pixel-size, rotation, maximum lat/y, rotation,
                pixel-size).
            epsg (int):
                Integer reference number to the projection (https://epsg.io/).
            no_data_value (Any, optional):
                No data value to mask the cells out of the domain. The default is -9999.
            driver_type (str, optional):
                Driver type ["GTiff", "MEM", "netcdf"]. Default is "MEM".
            path (str, optional):
                Path to save the driver.

        Returns:
            Dataset:
                Dataset object will be returned.
        """
        if geo is None:
            if top_left_corner is None or cell_size is None:
                raise ValueError(
                    "Either top_left_corner and cell_size or geo should be provided."
                )
            geo = (
                top_left_corner[0],
                cell_size,
                0,
                top_left_corner[1],
                0,
                -1 * cell_size,
            )

        if arr.ndim == 2:
            bands = 1
            rows = int(arr.shape[0])
            cols = int(arr.shape[1])
        else:
            bands = arr.shape[0]
            rows = int(arr.shape[1])
            cols = int(arr.shape[2])

        return cls._build_dataset(
            cols,
            rows,
            bands,
            numpy_to_gdal_dtype(arr),
            geo,
            sr_from_epsg(int(epsg)).ExportToWkt(),
            no_data_value,
            driver=driver_type,
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
        files: list[str | Path],
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
            path: Output ``.tif`` path. When ``None`` (default) the result
                is an in-memory dataset.

        Returns:
            Dataset: A multi-band dataset with ``band_count == len(files)``
            and ``band_names`` set.

        Raises:
            ValueError: ``files`` is empty, ``band_names`` length does not
                match ``files``, an input has more than one band, or ``path``
                does not end in ``.tif``.
            AlignmentError: ``align=False`` and the inputs do not share a
                grid/CRS.
            CRSError: An input raster has no CRS.

        Examples:
            - Stack three per-band GeoTIFFs into one 3-band dataset; band
              names come from the file names:
                ```python
                >>> import numpy as np
                >>> import tempfile, os
                >>> from pyramids.dataset import Dataset
                >>> d = tempfile.mkdtemp()
                >>> paths = []
                >>> for name, val in [("scene.B2.tif", 2), ("scene.B3.tif", 3), ("scene.B4.tif", 4)]:
                ...     p = os.path.join(d, name)
                ...     _ = Dataset.create_from_array(
                ...         np.full((4, 5), val, dtype="int16"),
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326, path=p,
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
                >>> _ = Dataset.create_from_array(
                ...     np.zeros((8, 9), dtype="int16"),
                ...     top_left_corner=(0, 0), cell_size=0.5, epsg=4326, path=odd,
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
            - :meth:`create_from_array`: build a dataset from a numpy array.
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

        if path is not None and not str(path).lower().endswith(".tif"):
            # TypeError to match ``_create_dataset`` (used by every other
            # factory: ``create_from_array``, ``dataset_like`` etc.) — keeping
            # one convention across the public surface.
            raise TypeError("the path to save the stacked raster should end with .tif")

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
            if align:
                # Resample every input onto the first file's grid in the
                # promoted dtype. Dataset.align adopts the alignment source's
                # dtype, so cast the template first to avoid truncating wider
                # inputs (e.g. a float band onto an int template).
                target_np_dtype = np.result_type(
                    *(ds.numpy_dtype[0] for ds in datasets)
                )
                grid_template = cls.create_from_array(
                    template.read_array(band=0).astype(target_np_dtype, copy=False),
                    geo=template.geotransform,
                    epsg=template.epsg,
                    no_data_value=resolved_nd,
                )
                # Dataset.align uses the source's no_data_value to fill the warp
                # destination, so the aligned fringe carries the SOURCE's sentinel.
                # When sources disagree on nodata (resolved_nd is the first one
                # by "first-wins" policy + a UserWarning), bands whose source's
                # sentinel != resolved_nd would still have that sentinel in the
                # fringe, which would no longer match the output band's declared
                # nodata. Remap so what's in the array matches what's declared.
                # Sources that already match the template grid skip the full
                # gdal.Warp round-trip and just astype, which is lossless.
                band_arrays = []
                for ds_i in datasets:
                    if _same_grid(template, ds_i):
                        arr = ds_i.read_array(band=0).astype(
                            target_np_dtype, copy=False
                        )
                    else:
                        arr = ds_i.align(grid_template).read_array(band=0)
                    band_arrays.append(
                        _remap_nodata_to(arr, ds_i.no_data_value[0], resolved_nd)
                    )
            else:
                band_arrays = [ds.read_array(band=0) for ds in datasets]
            stacked = np.stack(band_arrays, axis=0)
            obj = cls._build_dataset(
                template.columns,
                template.rows,
                len(resolved_paths),
                numpy_to_gdal_dtype(stacked),
                template.geotransform,
                template.crs,
                resolved_nd,
                path=path,
                array=stacked,
            )
        else:
            vrt = gdal.BuildVRT("", resolved_paths, separate=True)
            if (
                vrt is None
            ):  # pragma: no cover - BuildVRT returns None only on bad input
                raise AlignmentError(
                    f"gdal.BuildVRT could not stack {resolved_paths!r}"
                )
            if path is not None:
                dst = gdal.GetDriverByName("GTiff").CreateCopy(
                    str(path), vrt, strict=1, options=["COMPRESS=LZW"]
                )
            else:
                dst = gdal.GetDriverByName("MEM").CreateCopy("", vrt, strict=1)
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
                >>> from pyramids.dataset import Dataset
                >>> d = tempfile.mkdtemp()
                >>> members = []
                >>> for name, val in [("scene.B2.tif", 2), ("scene.B3.tif", 3)]:
                ...     p = os.path.join(d, name)
                ...     _ = Dataset.create_from_array(
                ...         np.full((4, 5), val, dtype="int16"),
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326, path=p,
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
