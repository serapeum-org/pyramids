"""
RasterBase.

State-holding base class that :class:`pyramids.dataset.Dataset` (and any
future Dataset variant — LazyDataset, COGDataset, …) inherits. Owns the
`gdal.Dataset` handle, geotransform, EPSG, dtype, and the abstract
contract that subclasses must implement. The L-2 collaborator pattern
(see :mod:`pyramids.dataset.engines`) attaches op families
(`ds.io`, `ds.spatial`, etc.) to instances of subclasses; this base
class provides the state they read through their weakref proxies.

The module file is still named `abstract_dataset.py` for backwards
compatibility with the module path; the class itself was renamed from
`AbstractDataset` to `RasterBase`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, cast

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal

from pyramids.base._utils import (
    Catalog,
)
from pyramids.base.crs import epsg_from_wkt, sr_from_epsg
from pyramids.base.protocols import ArrayLike, FloatArray
from pyramids.dataset.transform import GeoTransform
from pyramids.dataset.window import Window

if TYPE_CHECKING:
    from pyramids.base._file_manager import ThreadLocalFileManager
from pyramids.feature import FeatureCollection

DEFAULT_NO_DATA_VALUE = -9999
CATALOG = Catalog()
OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# Overview-build resampling names (gdal.Dataset.BuildOverviews family). This is a
# different GDAL name-space from the warp algorithms in
# pyramids.base._utils.INTERPOLATION_METHODS (gdal.GRA_*) used by to_crs/resample.
RESAMPLING_METHODS = [
    "NEAREST",
    "CUBIC",
    "AVERAGE",
    "GAUSS",
    "CUBICSPLINE",
    "LANCZOS",
    "MODE",
    "AVERAGE_MAGPHASE",
    "RMS",
    "BILINEAR",
]


def _reconstruct_dataset(cls: type[RasterBase], path: str, access: str) -> RasterBase:
    """Re-open a dataset from its pickle recipe tuple.

    Called by :meth:`RasterBase.__reduce__` on unpickle. Routes
    through the target class's `read_file` classmethod so subclass
    behavior (NetCDF mode flags, COG mixins) is preserved — subclasses
    that need to carry extra state (for example
    :class:`~pyramids.netcdf.NetCDF`) override `__reduce__` directly.

    Args:
        cls: The concrete :class:`RasterBase` subclass to
            reconstruct (`Dataset`, `NetCDF`, etc.).
        path: The on-disk path or VSI URL to re-open.
        access: Access mode string; `"read_only"` opens read-only,
            any other value opens for update.

    Returns:
        RasterBase: A freshly opened instance of `cls`.
    """
    read_only = access == "read_only"
    return cls.read_file(path, read_only=read_only)


class RasterBase(ABC):
    """RasterBase."""

    default_no_data_value = DEFAULT_NO_DATA_VALUE

    def __init__(self, src: gdal.Dataset, access: str = "read_only"):
        """__init__."""
        if not isinstance(src, gdal.Dataset):
            raise TypeError(  # pragma: no cover
                "src should be read using gdal (gdal dataset please read it using gdal"
                f" library) given {type(src)}"
            )
        self._access = access
        self._raster = src
        # Per-thread file manager for read_array(threadsafe=True); created
        # lazily by the IO engine and released by close().
        self._thread_manager: ThreadLocalFileManager | None = None
        self._geotransform: tuple[float, float, float, float, float, float] = (
            src.GetGeoTransform()
        )
        self._cell_size = self._geotransform[1]
        self._file_name: str = src.GetDescription()
        # the epsg property returns the value of the _epsg attribute, so if the projection changes in any function, the
        # function should also change the value of the _epsg attribute.
        self._epsg = self._get_epsg()
        # array and dimensions
        self._rows = src.RasterYSize
        self._columns = src.RasterXSize
        self._band_count = src.RasterCount
        self._block_size = [
            src.GetRasterBand(i).GetBlockSize() for i in range(1, self._band_count + 1)
        ]

    def __reduce__(self):
        """Return a recipe tuple that re-opens the dataset on unpickle.

        Serialising a live `gdal.Dataset` pointer is not possible
        (native C++ handle, no copy semantics). Instead we emit the
        minimal recipe `(class, file_name, access)` and reconstruct
        on unpickle by calling `cls.read_file(path, read_only=...)`.

        The GDAL handle is therefore opened **on the receiving process
        / thread**, which is the invariant dask.distributed needs.

        Raises:
            TypeError: The dataset has no on-disk path (empty
                `_file_name` or a `/vsimem/` path). In-memory
                datasets are not reconstructible from the recipe;
                call :meth:`to_file` first to anchor them to disk.
        """
        path = self._file_name
        if not path or path.startswith("/vsimem/"):
            raise TypeError(
                f"{type(self).__name__} has no on-disk path "
                f"(file_name={path!r}); pickling an in-memory "
                "dataset is not supported. Call .to_file(path) "
                "first to anchor it to disk."
            )
        return (_reconstruct_dataset, (type(self), path, self._access))

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the dataset.

        If close() raises an error, it is suppressed when an original
        exception is already propagating (to avoid masking it). If no
        original exception exists, the close error propagates normally.
        """
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False

    @abstractmethod
    def close(self) -> None:
        """Close the dataset and release the underlying handle."""
        pass

    @abstractmethod
    def align(self, *args, **kwargs):
        """Align this dataset to another dataset's grid."""
        pass

    @abstractmethod
    def __str__(self):
        """__str__."""
        pass

    @abstractmethod
    def __repr__(self):
        """__repr__."""
        pass

    @property
    def access(self):
        """Access mode (read_only/write)."""
        return self._access

    @property
    def raster(self) -> gdal.Dataset:
        """The base GDAL Dataset (read-only)."""
        return self._raster

    @property
    @abstractmethod
    def rows(self) -> int:
        """Number of rows in the raster array."""
        pass

    @property
    @abstractmethod
    def columns(self) -> int:
        """Number of columns in the raster array."""
        pass

    @property
    @abstractmethod
    def shape(self):
        """Shape (bands, rows, columns)."""
        pass

    @property
    @abstractmethod
    def band_count(self) -> int:
        """Number of bands in the raster."""
        pass

    @property
    @abstractmethod
    def band_names(self) -> list[str]:
        """Band names."""
        pass

    @property
    @abstractmethod
    def bbox(self) -> list:
        """Bound box [xmin, ymin, xmax, ymax]."""
        pass

    @property
    def geotransform(self) -> tuple[float, float, float, float, float, float]:
        """WKT projection.(x, cell_size, 0, y, 0, -cell_size)."""
        return self._geotransform

    @property
    def transform(self) -> GeoTransform:
        """The geotransform as an affine-style :class:`GeoTransform` object.

        Unlike the bare :attr:`geotransform` tuple, the returned object has
        named fields and algebra: ``transform * (col, row)`` maps pixel to map
        space, ``transform.inverse * (x, y)`` maps back.

        Examples:
            - Map the top-left pixel corner and invert back:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.ones((4, 4)), top_left_corner=(0, 4), cell_size=1.0, epsg=4326,
                ... )
                >>> ds.transform * (0, 0)
                (0.0, 4.0)
                >>> ds.transform.inverse * (2.0, 3.0)
                (2.0, 1.0)

                ```
        """
        return GeoTransform(*self._geotransform)

    def xy(
        self,
        rows: Number | list[Number] | np.ndarray,
        cols: Number | list[Number] | np.ndarray,
        *,
        center: bool = True,
    ) -> tuple[Any, Any]:
        """Return the map coordinates ``(x, y)`` of array cells.

        The rasterio-style companion of :meth:`rowcol`. Computed from the
        exact affine :attr:`transform`, so non-square pixels and rotated
        grids are handled; scalar input returns scalars, sequence input
        returns lists. (For point-table workflows use the cell engine's
        :meth:`array_to_map_coordinates`, which is also affine-exact.)

        Args:
            rows: Row index (or indices) of the cell(s).
            cols: Column index (or indices) of the cell(s).
            center: ``True`` (default) returns the cell centre, ``False`` the
                top-left cell corner.

        Returns:
            tuple: ``(x, y)`` scalars for scalar input, ``(xs, ys)`` lists for
                sequence input.

        Examples:
            - The centre of the top-left cell of a unit grid at (0, 4):
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.ones((4, 4)), top_left_corner=(0, 4), cell_size=1.0, epsg=4326,
                ... )
                >>> ds.xy(0, 0)
                (0.5, 3.5)
                >>> ds.xy(0, 0, center=False)
                (0.0, 4.0)

                ```
            - Vectorised input returns coordinate lists:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.ones((4, 4)), top_left_corner=(0, 4), cell_size=1.0, epsg=4326,
                ... )
                >>> xs, ys = ds.xy([0, 1], [0, 1])
                >>> xs
                [0.5, 1.5]
                >>> ys
                [3.5, 2.5]

                ```

        See Also:
            rowcol: The inverse, mapping map coordinates to cell indices.
            transform: The affine-style geotransform object.
        """
        # np.ndim == 0 treats Python scalars, NumPy scalars, and 0-d arrays
        # alike; np.isscalar misses 0-d arrays (np.isscalar(np.array(5)) is False).
        # cast: the numpy stub's np.ndim overload set doesn't recognise
        # numbers.Number, though it accepts any Number instance at runtime.
        scalar = np.ndim(cast(Any, rows)) == 0 and np.ndim(cast(Any, cols)) == 0
        rows_arr = np.atleast_1d(np.asarray(rows, dtype=float))
        cols_arr = np.atleast_1d(np.asarray(cols, dtype=float))
        shift = 0.5 if center else 0.0
        gt = self.transform
        xs_arr = (
            gt.x_origin
            + (cols_arr + shift) * gt.pixel_width
            + (rows_arr + shift) * gt.row_rotation
        )
        ys_arr = (
            gt.y_origin
            + (cols_arr + shift) * gt.column_rotation
            + (rows_arr + shift) * gt.pixel_height
        )
        xs = [float(value) for value in xs_arr]
        ys = [float(value) for value in ys_arr]
        result = (xs[0], ys[0]) if scalar else (xs, ys)
        return result

    def rowcol(
        self,
        x: Number | list[Number] | np.ndarray,
        y: Number | list[Number] | np.ndarray,
    ) -> tuple[Any, Any]:
        """Return the array indices ``(row, col)`` of map coordinates.

        The rasterio-style companion of :meth:`xy`. Computed from the exact
        inverse affine :attr:`transform`, so non-square pixels and rotated
        grids are handled; scalar input returns scalar ints, sequence input
        returns index arrays. (For point-table workflows use the cell
        engine's :meth:`map_to_array_coordinates`, which handles non-square
        pixels via the per-axis coordinate arrays but matches the nearest
        cell rather than inverting the affine, so it does not resolve
        rotated grids.)

        Args:
            x: X (longitude/easting) coordinate(s).
            y: Y (latitude/northing) coordinate(s).

        Returns:
            tuple: ``(row, col)`` ints for scalar input, ``(rows, cols)``
                lists of ints for sequence input (symmetric with :meth:`xy`).

        Examples:
            - The cell containing a point on a unit grid at (0, 4):
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.ones((4, 4)), top_left_corner=(0, 4), cell_size=1.0, epsg=4326,
                ... )
                >>> ds.rowcol(0.5, 3.5)
                (0, 0)
                >>> ds.rowcol(2.5, 1.5)
                (2, 2)

                ```
            - xy/rowcol round-trip through cell centres:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.ones((4, 4)), top_left_corner=(0, 4), cell_size=1.0, epsg=4326,
                ... )
                >>> ds.rowcol(*ds.xy(3, 1))
                (3, 1)

                ```

        See Also:
            xy: The inverse, mapping cell indices to map coordinates.
            transform: The affine-style geotransform object.
        """
        # np.ndim == 0 treats Python scalars, NumPy scalars, and 0-d arrays
        # alike; np.isscalar misses 0-d arrays (np.isscalar(np.array(5)) is False).
        # cast: the numpy stub's np.ndim overload set doesn't recognise
        # numbers.Number, though it accepts any Number instance at runtime.
        scalar = np.ndim(cast(Any, x)) == 0 and np.ndim(cast(Any, y)) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        inv = self.transform.inverse
        cols_f = inv.x_origin + x_arr * inv.pixel_width + y_arr * inv.row_rotation
        rows_f = inv.y_origin + x_arr * inv.column_rotation + y_arr * inv.pixel_height
        rows_idx = np.floor(rows_f).astype(int)
        cols_idx = np.floor(cols_f).astype(int)
        result: tuple[Any, Any]
        if scalar:
            result = (int(rows_idx[0]), int(cols_idx[0]))
        else:
            # Return Python lists for sequence input, matching xy() (and rasterio)
            # so the two companions have a symmetric container contract.
            result = (
                [int(value) for value in rows_idx],
                [int(value) for value in cols_idx],
            )
        return result

    @property
    def top_left_corner(self):
        """Top left corner coordinates."""
        xmin, _, _, ymax, _, _ = self._geotransform
        return xmin, ymax

    @property
    @abstractmethod
    def epsg(self) -> int | None:
        """EPSG number, or ``None`` for a CRS with no EPSG code (e.g. geostationary)."""
        pass

    @property
    @abstractmethod
    def crs(self) -> str:
        """Coordinate reference system."""
        pass

    @crs.setter
    @abstractmethod
    def crs(self, value: str):
        """Coordinate reference system."""
        pass

    @property
    @abstractmethod
    def cell_size(self) -> float:
        """Cell size."""
        pass

    @property
    @abstractmethod
    def no_data_value(self):
        """No data value that marks the cells out of the domain."""
        pass

    @no_data_value.setter
    @abstractmethod
    def no_data_value(self, value: list | Number):
        """no_data_value.

        No data value that marks the cells out of the domain

        Notes:
            - the setter does not change the values of the cells to the new no_data_value, it only changes the
              `no_data_value` attribute.
            - use this method to change the `no_data_value` attribute to match the value that is stored in the cells.
            - to change the values of the cells to the new no_data_value, use the `change_no_data_value` method.
        """
        pass

    @property
    def meta_data(self):
        """Meta data."""
        return self._raster.GetMetadata()

    @staticmethod
    def get_x_lon_dimension_array(pivot_x, cell_size, columns) -> FloatArray:
        """Build a 1-D array of x/longitude cell-centre coordinates.

        Args:
            pivot_x: X coordinate of the upper-left corner of
                the raster (left edge of the first pixel).
            cell_size: Pixel width in map units.
            columns: Number of columns in the raster.

        Returns:
            np.ndarray: 1-D array of length *columns* with the
                centre x coordinate of each column.
        """
        x_coords = np.array(
            [pivot_x + i * cell_size + cell_size / 2 for i in range(columns)]
        )
        return x_coords

    @staticmethod
    def get_y_lat_dimension_array(pivot_y, cell_size, rows) -> FloatArray:
        """Build a 1-D array of y/latitude cell-centre coordinates.

        Coordinates decrease from north to south (top to bottom).

        Args:
            pivot_y: Y coordinate of the upper-left corner of
                the raster (top edge of the first pixel).
            cell_size: Pixel height in map units (positive).
            rows: Number of rows in the raster.

        Returns:
            np.ndarray: 1-D array of length *rows* with the
                centre y coordinate of each row.
        """
        y_coords = np.array(
            [pivot_y - i * cell_size - cell_size / 2 for i in range(rows)]
        )
        return y_coords

    def _iloc(self, i: int) -> gdal.Band:
        """Access a GDAL Band by 0-based index.

        Hosted on `RasterBase` so every collaborator can resolve
        `self._ds._iloc(i)` without depending on `BandMetadata` being
        in the MRO. The duplicate body on `BandMetadata` is kept during
        Stage 1 of the L-2 migration (both bodies are identical) and is
        removed in Stage 2 PR2.7 when the bands collaborator lands.

        The returned band object is only valid while the parent dataset
        is open. Do not store the band reference — use it immediately
        and discard it.

        Args:
            i: Band index (0-based).

        Returns:
            gdal.Band: GDAL band object.

        Raises:
            IndexError: If the index is negative or out of bounds.
            RuntimeError: If the dataset has been closed.
        """
        if self._raster is None:
            raise RuntimeError(
                "Cannot access band on a closed dataset. "
                "The dataset has been closed via close() or a context manager."
            )
        if i < 0:
            raise IndexError("negative index not supported")
        if i > self.band_count - 1:
            raise IndexError(
                f"index {i} is out of bounds for axis 0 with size {self.band_count}"
            )
        return self.raster.GetRasterBand(i + 1)

    @property
    def block_size(self) -> list[tuple[int, int]]:
        """Block Size.

        The block size is the size of the block that the raster is divided into, the block size is used to read and
        write the raster data in blocks.

        Examples:
            - Get the block size of a dataset and print it:

              ```python
              >>> dataset = Dataset.read_file("tests/data/geotiff/era5_land_monthly_averaged.tif")
              >>> size = dataset.block_size
              >>> print(size)
              [(128, 128)]

              ```
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

    def block_windows(
        self, band: int = 0, *, window: Window | None = None
    ) -> Generator[Window, None, None]:
        """Yield a :class:`Window` for every native block of ``band``.

        Walks the raster in its on-disk block layout (tiles for tiled
        formats, full-width strips otherwise), clipping edge blocks to the
        raster extent. With ``window`` given, only blocks intersecting it
        are yielded, clipped to the window — useful for streaming a region.

        Yields windows only (no pixel reads), so it suits write pipelines
        and planners; use :meth:`iter_blocks` to also read each block.

        Args:
            band: Band index whose block layout drives the walk. Default 0.
            window: Optional region of interest; only intersecting blocks
                are yielded, clipped to it.

        Yields:
            Window: The next block window, row-major.

        Examples:
            - The block windows of a 5x5 in-memory raster tile it exactly
              once (MEM rasters expose full-width strip blocks):
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.ones((5, 5)), top_left_corner=(0, 5), cell_size=1.0, epsg=4326,
                ... )
                >>> windows = list(ds.block_windows())
                >>> sum(w.cols * w.rows for w in windows) == ds.rows * ds.columns
                True

                ```
            - Restrict the walk to a region of interest:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from pyramids.dataset.window import Window
                >>> ds = Dataset.create_from_array(
                ...     np.ones((6, 6)), top_left_corner=(0, 6), cell_size=1.0, epsg=4326,
                ... )
                >>> roi = Window(col_off=1, row_off=1, cols=3, rows=3)
                >>> all(w.intersection(roi) == w for w in ds.block_windows(window=roi))
                True

                ```

        See Also:
            iter_blocks: The reading variant, yielding ``(Window, ndarray)``.
        """
        block_x, block_y = self.block_size[band]
        if window is None:
            row_start, row_stop = 0, self.rows
            col_start, col_stop = 0, self.columns
        else:
            # Walk only the blocks that can intersect the ROI: start at the
            # block-aligned floor of the window and stop at its far edge, instead
            # of building and discarding every block of the whole raster.
            row_start = max(0, (window.row_off // block_y) * block_y)
            col_start = max(0, (window.col_off // block_x) * block_x)
            row_stop = min(self.rows, window.row_off + window.rows)
            col_stop = min(self.columns, window.col_off + window.cols)
        for row in range(row_start, row_stop, block_y):
            for col in range(col_start, col_stop, block_x):
                block = Window(
                    col_off=col,
                    row_off=row,
                    cols=min(block_x, self.columns - col),
                    rows=min(block_y, self.rows - row),
                )
                if window is not None:
                    intersected = block.intersection(window)
                    if intersected is None:
                        continue
                    block = intersected
                yield block

    def iter_blocks(
        self, band: int = 0, *, window: Window | None = None
    ) -> Generator[tuple[Window, np.typing.NDArray], None, None]:
        """Yield ``(Window, ndarray)`` for every native block of ``band``.

        The streaming read companion of :meth:`block_windows`: each yielded
        array is the block's pixels, so arbitrarily large rasters can be
        processed block-by-block in constant memory.

        This iterator is **serial**: it reads each block through the shared
        handle and does not accept ``threadsafe`` / ``chunks``. To read blocks
        in parallel, iterate :meth:`block_windows` and call
        ``read_array(window=..., threadsafe=True)`` per window from worker
        threads instead.

        Args:
            band: Band index to read. Default 0.
            window: Optional region of interest; only intersecting blocks
                are yielded, clipped to it.

        Yields:
            tuple[Window, np.ndarray]: The block window and its pixel values
                (shape ``window.shape``), row-major.

        Examples:
            - Stream a raster and rebuild it block-by-block:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> src_arr = np.arange(25, dtype="float32").reshape(5, 5)
                >>> ds = Dataset.create_from_array(
                ...     src_arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326,
                ... )
                >>> rebuilt = np.zeros_like(src_arr)
                >>> for w, block in ds.iter_blocks():
                ...     rebuilt[w.row_off : w.row_off + w.rows, w.col_off : w.col_off + w.cols] = block
                >>> bool((rebuilt == src_arr).all())
                True

                ```

        See Also:
            block_windows: The windows-only variant (no pixel reads).
        """
        for block in self.block_windows(band, window=window):
            # This iterator never passes chunks=, so read_array always returns a
            # plain ndarray here (the dask.Array arm of ArrayLike is unreachable).
            array = cast(np.typing.NDArray, self.read_array(band=band, window=block))
            yield block, array

    @property
    def file_name(self) -> str:
        """File name."""
        return self._file_name

    @property
    def driver_type(self):
        """Driver Type."""
        drv = self.raster.GetDriver()
        driver_type = drv.GetDescription() if drv is not None else None
        return CATALOG.get_driver_name(driver_type)

    @classmethod
    @abstractmethod
    def read_file(cls, path: str | Path, read_only=True) -> RasterBase:
        """Read file.

        Args:
            path (str):
                Path of file to open.
            read_only (bool):
                File mode, set as False, to open in "update" mode.

        Returns:
            Dataset: The opened dataset instance.
        """
        pass

    @abstractmethod
    def read_array(
        self,
        band: int | None = None,
        window: Window | GeoDataFrame | list[int] | None = None,
    ) -> ArrayLike:
        """Read Array.

            - read the values stored in a given band.

        Data Chuncks/blocks
            When a raster dataset is stored on disk, it might not be stored as one continuous chunk of data. Instead,
            it can be divided into smaller rectangular blocks or tiles. These blocks can be individually accessed,
            which is particularly useful for large datasets:
                Efficiency: Reading or writing small blocks requires less memory than dealing with the entire dataset
                    at once. This is especially beneficial when only a small portion of the data needs to be processed.
                Performance: For certain file formats and operations, working with optimal block sizes can significantly
                    improve performance. For example, if the block size matches the reading or processing window,
                        Pyramids can minimize disk access and data transfer.

        Args:
            band (int, optional):
                the band you want to get its data, If None the data of all bands will be read. Default is None
            window (List[int], optional):
                window to specify a block of data to read from the dataset. the window should be a list of 4 integers [offset_x, offset_y, window_columns, window_rows]. Default is None.

        Returns:
            np.ndarray:
                array with all the values in the raster.

        Examples:
            - Read a 5x5 window from the dataset and print its shape:

              ```python
              >>> dataset = Dataset.read_file("tests/data/geotiff/era5_land_monthly_averaged.tif")
              >>> arr = dataset.read_array(window=[0, 0, 5, 5])
              >>> print(arr.shape)
              (5, 5)

              ```
        """
        pass

    @abstractmethod
    def _read_block(
        self, band: int, window: list[int] | GeoDataFrame | None = None
    ) -> np.typing.NDArray:
        """Read block of data from the dataset.

        Args:
            band (int):
                Band index.
            window (List[int]):
                window to specify a block of data to read from the dataset. the window should be a list of 4 integers
                [offset_x, offset_y, window_columns, window_rows]. Default is None.

        Returns:
            np.ndarray:
                array with the values of the block. The shape of the array is (window[2], window[3]),
                and the location of the block in the raster is (window[0], window[1]).
        """
        pass

    @abstractmethod
    def plot(
        self,
        band: int | None = None,
        exclude_value: Any | None = None,
        rgb: list[int] | None = None,
        surface_reflectance: int | None = None,
        cutoff: list | None = None,
        overview: bool = False,
        overview_index: int = 0,
        **kwargs,
    ):
        """Plot.

            - plot the values/overviews of a given band.

        Args:
            band (int, optional):
                The band you want to get its data. Default is 0.
            exclude_value (Any, optional):
                Value to exclude from the plot. Default is None.
            rgb (List[int], optional):
                RGB band indices. Default is [3, 2, 1].
            surface_reflectance (int | None, optional):
                Surface reflectance value used to normalise satellite reflectance bands
                (typically ``10000`` for Sentinel-2). Default is ``None`` — concrete
                subclasses are responsible for picking a meaningful default when relevant.
            cutoff (List, optional):
                Clip the range of pixel values for each band (take only the pixel values from 0 to the value of
                the cutoff and scale them back to between 0 and 1). Default is None.
            overview (bool, optional):
                True if you want to plot the overview. Default is False.
            overview_index (int, optional):
                Index of the overview. Default is 0.
            **kwargs: Additional plotting options.
                points (array):
                    3 column array with the first column as the value you want to display for the point, the second
                    is the rows index of the point in the array, and the third column as the column index in the array.
                    The second and third columns tell the location of the point in the array.
                point_color (str):
                    Point color.
                point_size (Any):
                    Size of the point.
                pid_color (str):
                    The color of the annotation of the point. Default is blue.
                pid_size (Any):
                    Size of the point annotation.
                figsize (tuple, optional):
                    Figure size. The default is (8, 8).
                title (str, optional):
                    Title of the plot. The default is 'Total Discharge'.
                title_size (int, optional):
                    Title size. The default is 15.
                orientation (str, optional):
                    Orientation of the color bar horizontal/vertical. The default is 'vertical'.
                rotation (number, optional):
                    Rotation of the color bar label. The default is -90.
                cbar_length (float, optional):
                    Ratio to control the height of the color bar. The default is 0.75.
                ticks_spacing (int, optional):
                    Spacing in the color bar ticks. The default is 2.
                cbar_label_size (int, optional):
                    Size of the color bar label. The default is 12.
                cbar_label (str, optional):
                    Label of the color bar. The default is 'Discharge m3/s'.
                color_scale (str, optional):
                    Color-scale mode. One of "linear", "power", "sym-lognorm", "boundary-norm", "midpoint"
                    (case-insensitive), or a ``cleopatra.styles.ColorScale`` member. Integer codes are no
                    longer accepted. The default is "linear".
                gamma (float, optional):
                    Exponent for the "power" color scale. The default is 1./2.
                line_threshold (float, optional):
                    ``linthresh`` for the "sym-lognorm" color scale. The default is 0.0001.
                line_scale (float, optional):
                    ``linscale`` for the "sym-lognorm" color scale. The default is 0.001.
                bounds (List, optional):
                    A list of numbers used as discrete bounds for the "boundary-norm" color scale. Default is None.
                midpoint (float, optional):
                    Midpoint value for the "midpoint" color scale. The default is 0.
                cmap (str, optional):
                    Color style. The default is 'coolwarm_r'.
                display_cell_value (bool, optional):
                    True if you want to display the values of the cells as text.
                num_size (int, optional):
                    Size of the numbers plotted on top of each cell. The default is 8.
                background_color_threshold (float | int, optional):
                    Threshold value. If the value of the cell is greater, the plotted numbers will be black; if smaller,
                     the plotted number will be white. If None, maxvalue/2 will be considered. The default is None.

        Returns:
            Tuple[Axes, Any]:
                The axes of the matplotlib figure and the figure object.
        """
        pass

    @classmethod
    @abstractmethod
    def create_from_array(
        cls,
        arr: np.ndarray,
        geo: tuple[float, float, float, float, float, float],
        bands_values: list | None = None,
        epsg: str | int = 4326,
        no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "MEM",
        path: str | None = None,
        variable_name: str | None = None,
    ):
        """Create dataset from array.

            - Create_from_array method creates a `Dataset` from a given array and geotransform data.

        Args:
            arr (np.ndarray):
                Numpy array.
            geo (Tuple[float, float, float, float, float, float]):
                Geotransform tuple [minimum lon/x, pixel-size, rotation, maximum lat/y, rotation, pixel-size].
            bands_values (List, optional):
                Name of the bands to be used in the netcdf file. Default is None.
            epsg (int | str, optional):
                Integer reference number to the new projection (https://epsg.io/) (default 3857 the reference no of WGS84 web mercator). Default is 4326.
            no_data_value (Any, optional):
                No data value to mask the cells out of the domain. The default is -9999.
            driver_type (str, optional):
                Driver type ["GTiff", "MEM", "netcdf"]. Default is "MEM".
            path (str, optional):
                Path to save the driver.
            variable_name (str, optional):
                Name of the variable in the netcdf file. Default is None.

        Returns:
            RasterBase:
                Dataset object.
        """
        pass

    @abstractmethod
    def _get_crs(self) -> str:
        """Get coordinate reference system."""
        pass

    def set_crs(self, crs: str | None = None, epsg: int | None = None):
        """Set Coordinates reference system.

            Set the Coordinate Reference System (CRS) of a

        Args:
            crs (str): Optional if epsg is specified. WKT string.
                ```python
                i.e. 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"],
                AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",
                0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],
                AUTHORITY["EPSG","4326"]]'
                ```
            epsg (int):
                Optional if crs is specified. EPSG code specifying the projection.
        """
        # first change the projection of the gdal dataset object
        # second change the epsg attribute of the Dataset object
        if self.driver_type == "ascii":
            raise TypeError(
                "Setting CRS for ASCII file is not possible, you can save the files to a geotiff and then reset the crs"
            )
        else:
            if crs is not None:
                self.raster.SetProjection(crs)
                # ARC-7: get_epsg_from_prj raises on empty input;
                # epsg_from_wkt absorbs the historical 4326 fallback so
                # datasets with a missing projection still get tagged.
                self._epsg = epsg_from_wkt(crs)
            elif epsg is not None:
                sr = sr_from_epsg(epsg)
                self.raster.SetProjection(sr.ExportToWkt())
                self._epsg = epsg
            else:
                raise ValueError("Either crs or epsg must be provided.")

    @abstractmethod
    def to_crs(
        self,
        to_epsg: int,
        method: str = "nearest neighbor",
        maintain_alignment: bool = False,
    ) -> RasterBase:
        """To EPSG.

        to_epsg reprojects a raster to any projection
        (default the WGS84 web mercator projection, without resampling)

        Args:
            to_epsg (int):
                Reference number to the new projection (https://epsg.io/) (default 3857 the reference no of WGS84 web mercator).
            method (str):
                Resampling method, case-insensitive. Default is "nearest neighbor". Allowed values: "nearest"
                (alias "nearest neighbor"), "bilinear", "cubic", "cubic_spline", "lanczos", "average",
                "mode", "max", "min", "med", "q1", "q3", "sum", and "rms" (the GDAL warp algorithms;
                "sum"/"rms" need GDAL >= 3.1/3.3). See https://gisgeography.com/raster-resampling/.
            maintain_alignment (bool):
                True to maintain the number of rows and columns of the raster the same after reprojection. Default is False.
            inplace (bool):
                True to make changes inplace. Default is False.

        Returns:
            Dataset | None:
                Dataset object, if inplace is True, the method returns None.

        Examples:
            - Reproject a dataset to EPSG:3857:

              ```python
              >>> from pyramids.dataset import Dataset
              >>> dataset = Dataset.read_file("path/raster_name.tif")
              >>> reprojected_dataset = dataset.to_crs(to_epsg=3857)

              ```
        """
        pass

    @abstractmethod
    def _get_epsg(self) -> int | None:
        """Get EPSG.

            This function reads the projection of a GEOGCS file or tiff file

        Returns:
            int: EPSG number.
        """
        pass

    @abstractmethod
    def _check_no_data_value(self, no_data_value: list):
        """Validate The no_data_value with the dtype of the object.

        Args:
            no_data_value:
                    The no-data value to validate.

        Returns:
            Any:
                Convert the no_data_value to comply with the dtype.
        """
        # convert the no_data_value based on the dtype of each raster band.
        pass

    @abstractmethod
    def _set_no_data_value(self, no_data_value: Any | list = DEFAULT_NO_DATA_VALUE):
        """Set the NoDataValue.

            - Set the no data value in all raster bands.
            - Fill the whole raster with the no_data_value.
            - used only when creating an empty driver.

            now the no_data_value is converted to the dtype of the raster bands and updated in the
            dataset attribute, gdal nodatavalue attribute, used to fill the raster band.
            from here you have to use the no_data_value stored in the no_data_value attribute as it is updated.

        Args:
            no_data_value (numeric):
                no data value to fill the masked part of the array.
        """
        pass

    @abstractmethod
    def change_no_data_value(
        self, new_value: Any, old_value: Any | None = None, inplace: bool = False
    ):
        """Change No Data Value.

            - Set the no data value in all raster bands.
            - Fill the whole raster with the no_data_value.
            - Change the no_data_value in the array in all bands.

        Args:
            new_value (numeric):
                No data value to set in the raster bands.
            old_value (numeric, optional):
                Old no data value that is already in the raster bands.
            inplace (bool):
                If True, the original dataset will be modified. If False, a new dataset will be created.
                Default is False.
        """
        pass

    @abstractmethod
    def to_file(self, path: str | Path, band: int = 0) -> None:
        """Save dataset to disk.

            to_file a raster to a path, the type of the driver (georiff/netcdf/ascii) will be implied from the
            extension at the end of the given path.

        Args:
            path (str):
                A path including the name of the dataset with the extension at the end (i.e. "data/cropped.tif").
            band (int):
                Band index, needed only in case of ascii drivers. Default is 0.

        Examples:
            - Save a dataset to a new GeoTIFF file:

              ```python
              >>> dataset = Dataset.read_file("path/to/file/***.tif")
              >>> dataset.to_file("save_raster_test.tif")

              ```

        Notes:
            The object will still refer to the dataset before saving. If you want to use the new saved dataset you have to read the file again.
        """
        pass

    @abstractmethod
    def crop(
        self,
        mask: GeoDataFrame | FeatureCollection,
        touch: bool = True,
    ) -> RasterBase:
        """Crop.

            Crop/Clip the Dataset object using a polygon/raster.

        Args:
            mask (GeoDataFrame | FeatureCollection):
                GeoDataFrame with a polygon geometry, or a Dataset object.
            touch (bool):
                Include the cells that touch the polygon not only those that lie entirely inside the polygon mask. Default is True.
            inplace (bool):
                True to make the changes in place.

        Returns:
            RasterBase: Dataset Object.
        """
        pass

    @abstractmethod
    def extract(
        self,
        exclude_value: Any | None = None,
        mask: FeatureCollection | GeoDataFrame | None = None,
    ) -> np.typing.NDArray:
        """Extract.

            - Extract method gets all the values in a raster, and excludes the values in the exclude_value parameter.
            - If the mask parameter is given, the raster will be clipped to the extent of the given mask and the
            values within the mask are extracted.

        Args:
            exclude_value (Numeric, optional):
                Values you want to exclude from extracted values.
            feature (FeatureCollection | GeoDataFrame, optional):
                Vector file contains geometries you want to extract the values at their location. Default is None.
        """
        # Optimize: make the read_array return only the array for inside the mask feature, and not to read the whole
        #  raster
        pass

    @abstractmethod
    def overlay(
        self,
        classes_map,
        band: int = 0,
        exclude_value: float | int | None = None,
    ) -> dict[float, list[float]]:
        """Overlay.

            overlay extracts all the values in raster file if you have two maps one with classes, and the other map
            contains any type of values, and you want to know the values in each class.

        Args:
            classes_map (RasterBase):
                Dataset object for the raster that has classes you want to overlay with the raster.
            band (int):
                If the raster is multi-band raster choose the band you want to overlay with the classes map. Default is 0.
            exclude_value (float | int, optional):
                Values you want to exclude from extracted values.

        Returns:
            Dict[List[float], List[float]]:
                Dictionary with a list of values in the basemap as keys and for each key a list of all the intersected values in the maps from the path.
        """
        pass

    @abstractmethod
    def create_overviews(
        self, resampling_method: str = "nearest", overview_levels: list | None = None
    ):
        """Create overviews for the dataset.

        Args:
            resampling_method (str, optional):
                The resampling method used to create the overviews, by default "nearest".
                Possible values are:
                    "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE",
                    "AVERAGE_MAGPHASE", "RMS", "BILINEAR".
            overview_levels (list, optional):
                The overview levels, restricted to the typical power-of-two reduction factors. Default [2, 4, 8, 16, 32].

        Returns:
            Tuple[str, list]:
                Information about whether overviews are internal or external, and the overview_count list per band.
                - External (.ovr file):
                    If the dataset is read with `read_only=True` then the overviews' file will be
                    created in the same directory of the dataset, with the same name of the dataset and .ovr extension.
                - Internal:
                    If the dataset is read with `read_only=False` then the overviews will be created internally in the
                    dataset, and the dataset needs to be saved/flushed to save the new changes to disk.
        """
        pass

    @abstractmethod
    def recreate_overviews(self, resampling_method: str = "nearest"):
        """Recreate overviews for the dataset.

        Args:
            resampling_method (str, optional):
                The resampling method used to create the overviews, by default "nearest".
                Possible values are:
                    "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE", "AVERAGE_MAGPHASE",
                    "RMS", "BILINEAR".

        Raises:
            ValueError:
                resampling_method should be one of {"NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS",
                "MODE", "AVERAGE_MAGPHASE", "RMS", "BILINEAR"}.
            ReadOnlyError:
                If the overviews are internal and the Dataset is opened with a read only.
                Please read the dataset using read_only=False
        """
        pass

    @abstractmethod
    def get_overview(self, band: int = 0, overview_index: int = 0) -> gdal.Band:
        """Get an overview of a band.

        Args:
            band (int, optional):
                The band index, by default 0.
            overview_index (int):
                Index of the overview. Default is 0.

        Returns:
            gdal.Band:
                GDAL band object.
        """
        pass

    @abstractmethod
    def read_overview_array(
        self, band: int | None = None, overview_index: int = 0
    ) -> np.typing.NDArray:
        """Read an overview array.

            - read the values stored in a given band.

        Args:
            band (int, optional):
                The band you want to get its data; if None, the data of all bands will be read. Default is None.
            overview_index (int):
                Index of the overview. Default is 0.

        Returns:
            np.ndarray:
                Array with all the values in the raster.
        """
        pass
