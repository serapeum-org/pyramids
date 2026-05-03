"""
Dataset module.

raster contains python functions to handle raster data align them together based on a source raster, perform any
algebraic operation on cell's values.
"""

from __future__ import annotations

import logging
import weakref
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
from osgeo import gdal

from pyramids import _io
from pyramids.base._utils import (
    DTYPE_CONVERSION_DF,
    numpy_to_gdal_dtype,
)
from pyramids.base.crs import epsg_from_wkt, sr_from_epsg
from pyramids.dataset.abstract_dataset import (
    DEFAULT_NO_DATA_VALUE,
    RasterBase,
)
from pyramids.dataset._collaborators import (
    COG,
    IO,
    Analysis,
    Bands,
    Cell,
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
from pyramids.dataset.ops.vectorize import rasterize_features
from pyramids.feature import FeatureCollection

# tuple of collaborator attribute names. Used by
# `Dataset.__init__` to wire the seven collaborators and by
# `_update_inplace` to re-bind their `_ds` back-references after
# `__dict__.update` (see audit §3.3).
_COLLABORATOR_ATTRS = ("io", "spatial", "bands", "analysis", "cell", "vectorize", "cog")

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
        """
        new = type(self)(src, access=access or self._access)
        self.__dict__.update(new.__dict__)
        # Re-bind via `weakref.proxy` so the back-reference stays
        # weak after the dict swap (matches `_Collaborator.__init__`).
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
        """Facade — delegates to :meth:`Cell.get_cell_coords <pyramids.dataset._collaborators.Cell.get_cell_coords>`."""
        return self.cell.get_cell_coords(*args, **kwargs)

    def get_cell_polygons(self, *args, **kwargs):
        """Facade — delegates to :meth:`Cell.get_cell_polygons <pyramids.dataset._collaborators.Cell.get_cell_polygons>`."""
        return self.cell.get_cell_polygons(*args, **kwargs)

    def get_cell_points(self, *args, **kwargs):
        """Facade — delegates to :meth:`Cell.get_cell_points <pyramids.dataset._collaborators.Cell.get_cell_points>`."""
        return self.cell.get_cell_points(*args, **kwargs)

    def map_to_array_coordinates(self, *args, **kwargs):
        """Facade — delegates to :meth:`Cell.map_to_array_coordinates <pyramids.dataset._collaborators.Cell.map_to_array_coordinates>`."""
        return self.cell.map_to_array_coordinates(*args, **kwargs)

    def array_to_map_coordinates(self, *args, **kwargs):
        """Facade — delegates to :meth:`Cell.array_to_map_coordinates <pyramids.dataset._collaborators.Cell.array_to_map_coordinates>`."""
        return self.cell.array_to_map_coordinates(*args, **kwargs)

    def to_cog(self, *args, **kwargs):
        """Facade — delegates to :meth:`COG.to_cog <pyramids.dataset._collaborators.COG.to_cog>`."""
        return self.cog.to_cog(*args, **kwargs)

    @property
    def is_cog(self) -> bool:
        """Facade — delegates to :attr:`COG.is_cog <pyramids.dataset._collaborators.COG.is_cog>`."""
        return self.cog.is_cog

    def validate_cog(self, *args, **kwargs):
        """Facade — delegates to :meth:`COG.validate_cog <pyramids.dataset._collaborators.COG.validate_cog>`."""
        return self.cog.validate_cog(*args, **kwargs)

    def to_feature_collection(self, *args, **kwargs):
        """Facade — delegates to :meth:`Vectorize.to_feature_collection <pyramids.dataset._collaborators.Vectorize.to_feature_collection>`."""
        return self.vectorize.to_feature_collection(*args, **kwargs)

    def translate(self, *args, **kwargs):
        """Facade — delegates to :meth:`Vectorize.translate <pyramids.dataset._collaborators.Vectorize.translate>`."""
        return self.vectorize.translate(*args, **kwargs)

    def cluster(self, *args, **kwargs):
        """Facade — delegates to :meth:`Vectorize.cluster <pyramids.dataset._collaborators.Vectorize.cluster>`."""
        return self.vectorize.cluster(*args, **kwargs)

    def cluster2(self, *args, **kwargs):
        """Facade — delegates to :meth:`Vectorize.cluster2 <pyramids.dataset._collaborators.Vectorize.cluster2>`."""
        return self.vectorize.cluster2(*args, **kwargs)

    def stats(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.stats <pyramids.dataset._collaborators.Analysis.stats>`."""
        return self.analysis.stats(*args, **kwargs)

    def count_domain_cells(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.count_domain_cells <pyramids.dataset._collaborators.Analysis.count_domain_cells>`."""
        return self.analysis.count_domain_cells(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.apply <pyramids.dataset._collaborators.Analysis.apply>`.

        The collaborator returns `None` for `inplace=True` so the facade
        can substitute the actual `self` (preserving identity); the proxy
        used by the collaborator's back-reference would otherwise fail
        `result is ds` checks.
        """
        result = self.analysis.apply(*args, **kwargs)
        return self if result is None else result

    def fill(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.fill <pyramids.dataset._collaborators.Analysis.fill>`.

        The collaborator returns `None` for `inplace=True`; see
        :meth:`apply` for the rationale.
        """
        result = self.analysis.fill(*args, **kwargs)
        return self if result is None else result

    def extract(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.extract <pyramids.dataset._collaborators.Analysis.extract>`."""
        return self.analysis.extract(*args, **kwargs)

    def overlay(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.overlay <pyramids.dataset._collaborators.Analysis.overlay>`."""
        return self.analysis.overlay(*args, **kwargs)

    def get_mask(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.get_mask <pyramids.dataset._collaborators.Analysis.get_mask>`."""
        return self.analysis.get_mask(*args, **kwargs)

    def footprint(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.footprint <pyramids.dataset._collaborators.Analysis.footprint>`."""
        return self.analysis.footprint(*args, **kwargs)

    def get_histogram(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.get_histogram <pyramids.dataset._collaborators.Analysis.get_histogram>`."""
        return self.analysis.get_histogram(*args, **kwargs)

    def plot(self, *args, **kwargs):
        """Facade — delegates to :meth:`Analysis.plot <pyramids.dataset._collaborators.Analysis.plot>`."""
        return self.analysis.plot(*args, **kwargs)

    def crop(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.crop <pyramids.dataset._collaborators.Spatial.crop>`."""
        return self.spatial.crop(*args, **kwargs)

    def to_crs(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.to_crs <pyramids.dataset._collaborators.Spatial.to_crs>`."""
        return self.spatial.to_crs(*args, **kwargs)

    def set_crs(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.set_crs <pyramids.dataset._collaborators.Spatial.set_crs>`."""
        return self.spatial.set_crs(*args, **kwargs)

    def convert_longitude(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.convert_longitude <pyramids.dataset._collaborators.Spatial.convert_longitude>`."""
        return self.spatial.convert_longitude(*args, **kwargs)

    def resample(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.resample <pyramids.dataset._collaborators.Spatial.resample>`."""
        return self.spatial.resample(*args, **kwargs)

    def align(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.align <pyramids.dataset._collaborators.Spatial.align>`."""
        return self.spatial.align(*args, **kwargs)

    def fill_gaps(self, *args, **kwargs):
        """Facade — delegates to :meth:`Spatial.fill_gaps <pyramids.dataset._collaborators.Spatial.fill_gaps>`."""
        return self.spatial.fill_gaps(*args, **kwargs)

    def read_array(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.read_array <pyramids.dataset._collaborators.IO.read_array>`."""
        return self.io.read_array(*args, **kwargs)

    def write_array(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.write_array <pyramids.dataset._collaborators.IO.write_array>`."""
        return self.io.write_array(*args, **kwargs)

    def to_file(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.to_file <pyramids.dataset._collaborators.IO.to_file>`."""
        return self.io.to_file(*args, **kwargs)

    def to_raster(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.to_raster <pyramids.dataset._collaborators.IO.to_raster>`."""
        return self.io.to_raster(*args, **kwargs)

    def get_block_arrangement(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.get_block_arrangement <pyramids.dataset._collaborators.IO.get_block_arrangement>`."""
        return self.io.get_block_arrangement(*args, **kwargs)

    def get_tile(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.get_tile <pyramids.dataset._collaborators.IO.get_tile>`."""
        return self.io.get_tile(*args, **kwargs)

    def map_blocks(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.map_blocks <pyramids.dataset._collaborators.IO.map_blocks>`."""
        return self.io.map_blocks(*args, **kwargs)

    def to_xyz(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.to_xyz <pyramids.dataset._collaborators.IO.to_xyz>`."""
        return self.io.to_xyz(*args, **kwargs)

    @property
    def overview_count(self):
        """Facade — delegates to :attr:`IO.overview_count <pyramids.dataset._collaborators.IO.overview_count>`."""
        return self.io.overview_count

    def create_overviews(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.create_overviews <pyramids.dataset._collaborators.IO.create_overviews>`."""
        return self.io.create_overviews(*args, **kwargs)

    def recreate_overviews(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.recreate_overviews <pyramids.dataset._collaborators.IO.recreate_overviews>`."""
        return self.io.recreate_overviews(*args, **kwargs)

    def get_overview(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.get_overview <pyramids.dataset._collaborators.IO.get_overview>`."""
        return self.io.get_overview(*args, **kwargs)

    def read_overview_array(self, *args, **kwargs):
        """Facade — delegates to :meth:`IO.read_overview_array <pyramids.dataset._collaborators.IO.read_overview_array>`."""
        return self.io.read_overview_array(*args, **kwargs)

    def _read_block(self, *args, **kwargs):
        """Facade — concrete override of the abstract :meth:`RasterBase._read_block`."""
        return self.io._read_block(*args, **kwargs)

    def get_attribute_table(self, *args, **kwargs):
        """Facade — delegates to :meth:`Bands.get_attribute_table <pyramids.dataset._collaborators.Bands.get_attribute_table>`."""
        return self.bands.get_attribute_table(*args, **kwargs)

    def set_attribute_table(self, *args, **kwargs):
        """Facade — delegates to :meth:`Bands.set_attribute_table <pyramids.dataset._collaborators.Bands.set_attribute_table>`."""
        return self.bands.set_attribute_table(*args, **kwargs)

    def add_band(self, *args, **kwargs):
        """Facade — delegates to :meth:`Bands.add_band <pyramids.dataset._collaborators.Bands.add_band>`."""
        return self.bands.add_band(*args, **kwargs)

    def get_band_by_color(self, *args, **kwargs):
        """Facade — delegates to :meth:`Bands.get_band_by_color <pyramids.dataset._collaborators.Bands.get_band_by_color>`."""
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
        """Facade — delegates to :attr:`Bands.band_color <pyramids.dataset._collaborators.Bands.band_color>`."""
        return self.bands.band_color

    @band_color.setter
    def band_color(self, values):
        self.bands.band_color = values

    @property
    def color_table(self):
        """Facade — delegates to :attr:`Bands.color_table <pyramids.dataset._collaborators.Bands.color_table>`."""
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
        poly = FeatureCollection.create_polygon(coords)
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
        """
        resolved_chunks = chunks if chunks is not None else "auto"
        return write_dataset_to_zarr(
            self,
            store,
            compute=compute,
            mode=mode,
            chunks=resolved_chunks,
            storage_options=storage_options,
        )

    @classmethod
    def from_zarr(
        cls,
        store,
        *,
        chunks=None,
        storage_options: dict | None = None,
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
        """
        return read_dataset_from_zarr(
            store,
            chunks=chunks,
            storage_options=storage_options,
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
            Mask: {self._no_data_value[0]}
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

    @property
    def no_data_value(self) -> list:
        """No data value that marks the cells out of the domain."""
        return list(self._no_data_value)

    @no_data_value.setter
    def no_data_value(self, value: list | Number):
        """no_data_value.

        No data value that marks the cells out of the domain

        Notes:
            - The setter does not change the values of the cells to the new no_data_value, it only changes the
            `no_data_value` attribute.
            - Use this method to change the `no_data_value` attribute to match the value that is stored in the cells.
            - To change the values of the cells, to the new no_data_value, use the `change_no_data_value` method.

        See Also:
            - Dataset.change_no_data_value: Change the No Data Value.
        """
        if isinstance(value, list):
            for i, val in enumerate(value):
                self.bands._change_no_data_value_attr(i, val)
        else:
            self.bands._change_no_data_value_attr(0, value)

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
        """Longitude coordinates.

        See Also:
            - Dataset.x: Dataset x coordinates.
            - Dataset.lat: Dataset latitude.
        """
        x_coords = self.get_x_lon_dimension_array(
            self.top_left_corner[0], self.cell_size, self.columns
        )
        return x_coords

    @property
    def lat(self) -> np.ndarray:
        """Latitude-coordinate.

        See Also:
            - Dataset.x: Dataset x coordinates.
            - Dataset.y: Dataset y coordinates.
            - Dataset.lon: Dataset longitude.
        """
        y_coords = self.get_y_lat_dimension_array(
            self.top_left_corner[1], self.cell_size, self.rows
        )
        return y_coords

    @property
    def x(self) -> np.ndarray:
        """X-coordinate/Longitude.

        See Also:
            - Dataset.lat: Dataset latitude.
            - Dataset.y: Dataset y coordinates.
            - Dataset.lon: Dataset longitude.
        """
        # X_coordinate = upper-left corner x + index * cell size + cell-size/2
        return self.lon

    @property
    def y(self) -> np.ndarray:
        """Y-coordinate/Latitude.

        See Also:
            - Dataset.x: Dataset y coordinates.
            - Dataset.lat: Dataset latitude.
            - Dataset.lon: Dataset longitude.
        """
        # Y_coordinate = upper-left corner y - index * cell size - cell-size/2
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
    ) -> Dataset:
        """read_file.

        Args:
            path (str):
                Path of file to open.
            read_only (bool):
                File mode, set to False, to open in "update" mode.
            file_i (int):
                Index to the file inside the compressed file you want to read, if the compressed file
                has only one file. Default is 0.

        Returns:
            Dataset:
                Opened dataset instance.

        See Also:
            - Dataset.read_array: Read the values stored in a dataset band.
        """
        src = _io.read_file(path, read_only=read_only, file_i=file_i)
        return cls(src, access="read_only" if read_only else "write")

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
        """
        if self._raster is not None:
            self._raster.FlushCache()
            self._raster = None

    @staticmethod
    def _create_dataset(
        cols: int,
        rows: int,
        bands: int,
        dtype: int,
        driver: str = "MEM",
        path: str | Path | None = None,
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
            # of computations.
            src = gdal.GetDriverByName(driver).Create(
                str(path), cols, rows, bands, dtype, ["COMPRESS=LZW"]
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

        Returns:
            Dataset: A fully configured Dataset object.
        """
        dst = cls._create_dataset(cols, rows, bands, dtype, driver=driver, path=path)
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
