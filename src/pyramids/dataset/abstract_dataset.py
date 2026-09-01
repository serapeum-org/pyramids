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

import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal

from pyramids.base._errors import ReadOnlyError
from pyramids.base._utils import (
    DEFAULT_RESAMPLING,
    get_catalog,
)
from pyramids.base.crs import epsg_of_crs, sr_from_epsg
from pyramids.base.georeference import GeoReference
from pyramids.base.protocols import ArrayLike, FloatArray
from pyramids.base.remote import cloud_config_from_env
from pyramids.dataset._subdataset import SubDataset, subdatasets_of
from pyramids.dataset.transform import GeoTransform
from pyramids.dataset.window import Window

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pyramids.base._file_manager import ThreadLocalFileManager

from pyramids.feature import FeatureCollection

DEFAULT_NO_DATA_VALUE = -9999
CATALOG = get_catalog()
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


_EngineMethod = TypeVar("_EngineMethod", bound=Callable[..., Any])


def under_gdal_env(method: _EngineMethod) -> _EngineMethod:
    """Run an engine entry point under the dataset's captured cloud config.

    Internal: an engine decorator, not public API. It is unprefixed only because
    the engine modules import it across package boundaries; it is meaningless on
    anything that does not expose `self._ds`.

    The engines reach their dataset through `self._ds`, so one decorator covers
    an entry point without it remembering to open a `with` block.

    Apply it at the **entry point**, not at the primitives it calls: nesting is
    harmless but not free — each install constructs a `CloudConfig` and makes
    `gdal.config_options` save and restore every key, which measured at ~4x on a
    small windowed read when the decorator sat on both `read_array` and the
    `_read_block` it calls.

    Never apply it to a generator. Spanning the `yield`s would keep the config
    installed while the *consumer's* loop body runs, and `gdal.config_options`
    restores a snapshot on exit — so two interleaved generators leaving in
    non-LIFO order silently strip each other's config. A generator installs the
    config per item instead.

    The signature is preserved (a bound `TypeVar`), so a decorated public method
    keeps its argument checking rather than degrading to `(*args, **kwargs) ->
    Any`.

    Args:
        method: An engine method whose `self` exposes `_ds`.

    Returns:
        The method wrapped so `RasterBase._cloud_config` is installed around it.

    Raises:
        TypeError: `method` is a generator function — see above.
    """
    if inspect.isgeneratorfunction(method):
        raise TypeError(
            f"under_gdal_env cannot wrap the generator {method.__qualname__!r}: the "
            "config would stay installed across its yields, leaking into the "
            "caller's scope and corrupting a concurrently-open generator's config. "
            "Install it per item inside the generator instead."
        )

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._ds._cloud_config():
            return method(self, *args, **kwargs)

    return cast(_EngineMethod, wrapper)


def _reconstruct_dataset(
    cls: type[RasterBase],
    path: str,
    access: str,
    gdal_env: dict[str, str] | None = None,
    open_options: tuple[str, ...] | None = None,
) -> RasterBase:
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
        access: Access mode string carried by the pickle recipe. **Ignored** for the
            open mode — the dataset is always reopened read-only (see below). Kept in
            the signature so existing recipe tuples still unpickle.
        gdal_env: The captured cloud config (a signer's `gdal_env()`), installed
            around the re-open and re-attached to the instance so the worker's
            reads authenticate as the originating process's did. Defaults to
            `None` so a three-element recipe from an older pickle still loads.
        open_options: The GDAL open options captured on the originating dataset,
            as a hashable tuple, forwarded to `read_file` so the worker reopens
            with the same driver behaviour. Defaults to `None` so a four-element
            recipe from an older pickle (with no options) still loads (#1025).

    Returns:
        RasterBase: A freshly opened instance of `cls`, opened read-only.
    """
    # Always reopen read-only, regardless of the pickled access mode: every
    # distributed consumer reads the reconstructed handle (the lazy read fan-out
    # ships its own read-only file manager; the delayed to_file task only reads the
    # source for a CreateCopy), and reopening N update-mode handles on one file
    # across workers risks lock contention / a corrupt write. A caller that truly
    # needs a writable handle should reopen it explicitly on the worker.
    #
    # The env is applied here rather than forwarded to read_file so every
    # subclass benefits without widening its own signature.
    with cloud_config_from_env(gdal_env, path=path):
        # warn_on_container=False: unpickling an already-opened container reopens a
        # handle the caller previously held, so re-emitting the container warning here
        # would be spurious. Only the base Dataset reaches this reconstruct (NetCDF has
        # its own _reconstruct_netcdf), so read_file always accepts the keyword.
        dataset = cls.read_file(
            path,
            read_only=True,
            open_options=list(open_options) if open_options else None,
            warn_on_container=False,
        )
    dataset.attach_gdal_env(gdal_env)
    return dataset


class RasterBase(ABC):
    """RasterBase."""

    default_no_data_value = DEFAULT_NO_DATA_VALUE

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        *,
        gdal_env: dict[str, str] | None = None,
        open_options: tuple[str, ...] | list[str] | None = None,
    ):
        """Wrap an already-open ``gdal.Dataset`` and snapshot its geo-properties.

        Args:
            src: An open :class:`osgeo.gdal.Dataset` to wrap. Callers normally go
                through :meth:`read_file` rather than constructing directly.
            access: The mode ``src`` was opened with — ``"read_only"`` (default)
                or ``"write"``. Recorded so mutating operations can refuse a
                read-only handle.
            gdal_env: GDAL config (cloud credentials, HTTP knobs) this dataset was
                opened with, captured so it is re-installed around every read that
                reopens the file (``threadsafe=True`` handles, lazy ``chunks=``
                reads, unpickle on a worker). Stored as a plain dict so it
                survives pickling; ``None`` (default) captures nothing.
            open_options: GDAL open options this dataset was opened with, as a
                mapping-derived or native ``["KEY=VALUE"]`` sequence. Stored as a
                tuple so it stays hashable for the file-manager cache key and
                stable across pickling, and reapplied on the same reopen paths as
                ``gdal_env``. ``None`` (default) captures nothing (#1025).

        Raises:
            TypeError: ``src`` is not an :class:`osgeo.gdal.Dataset`.
        """
        if not isinstance(src, gdal.Dataset):
            raise TypeError(  # pragma: no cover
                "src should be read using gdal (gdal dataset please read it using gdal"
                f" library) given {type(src)}"
            )
        self._access = access
        self._raster = src
        # Cloud credentials/config this dataset was opened with (a STAC signer's
        # gdal_env, say). Re-installed around the reads that do not go through
        # the handle opened above: `threadsafe=True` opens one handle per
        # thread, a lazy `chunks=` read opens inside the dask task, and
        # unpickling on a worker re-opens from the path. Applied structurally by
        # the `@under_gdal_env` decorator on the engine read primitives. A plain
        # dict so it survives pickling; empty (the common case) costs a
        # `nullcontext`. It does *not* reach a VRT's source opens — GDAL ignores
        # the thread-local config there, so those credentials ride the source
        # path instead (see `pyramids.stac._vrt`).
        self._gdal_env: dict[str, str] = dict(gdal_env) if gdal_env else {}
        # GDAL open options this dataset was opened with (a driver knob like
        # `L1B_MODE=DATASTRIP`). Stored as a tuple so it stays hashable for
        # the file-manager cache key and stable across pickling, and carried
        # onto every reopen path the way `_gdal_env` is (threadsafe handles,
        # lazy `chunks=` reads, unpickle) so a worker reopens with the same
        # driver behaviour (#1025).
        self._open_options: tuple[str, ...] = tuple(open_options or ())
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

    @property
    def open_options(self) -> list[str]:
        """GDAL open options captured at read time, as a `KEY=VALUE` list.

        Empty when the dataset was opened without any (the common case).
        Reapplied on every path that reopens the file rather than reusing the
        live handle, so driver behaviour survives a worker reopen (#1025).
        """
        return list(self._open_options)

    @property
    def gdal_env(self) -> dict[str, str]:
        """GDAL config re-installed around every read of this dataset.

        Populated when the dataset was opened with cloud credentials — for
        example :func:`pyramids.stac.load_asset` with a Requester-Pays or
        bearer-token signer. Empty for an ordinary local or anonymous open.

        Note:
            This is a **property**, while the same-named
            :meth:`pyramids.stac.signers.Signer.gdal_env` is a *method* — the
            two meet in one workflow (`load_asset(signer=…)` then `ds.gdal_env`),
            so `ds.gdal_env()` is an easy slip that fails with
            `TypeError: 'dict' object is not callable`.

        Returns:
            A copy of the captured config, so mutating it does not affect the
            dataset.

        Examples:
            - An ordinary open captures nothing:
                ```python
                >>> from pyramids.dataset import Dataset
                >>> Dataset.read_file("tests/data/acc4000.tif").gdal_env
                {}

                ```
            - An open that carried credentials keeps them for later reads:
                ```python
                >>> ds = Dataset.read_file(
                ...     "tests/data/acc4000.tif",
                ...     gdal_env={"AWS_REQUEST_PAYER": "requester"},
                ... )
                >>> ds.gdal_env["AWS_REQUEST_PAYER"]
                'requester'

                ```
        """
        return dict(self._gdal_env)

    def _cloud_config(self) -> Any:
        """Return a context manager installing :attr:`gdal_env` for a read.

        A :class:`contextlib.nullcontext` when nothing was captured, so the
        overwhelmingly common unsigned case pays nothing per read.
        """
        return cloud_config_from_env(self._gdal_env, path=self._file_name)

    def attach_gdal_env(self, gdal_env: dict[str, str] | None) -> None:
        """Capture `gdal_env` on an already-open dataset.

        The supported way for a reader that cannot take `gdal_env=` at open time
        — :func:`pyramids.grib.open_grib`, :meth:`pyramids.netcdf.NetCDF.read_file`
        — to hand its caller's credentials to the object it just built, instead
        of the caller reaching in and setting the private attribute.

        Args:
            gdal_env: The GDAL config to capture. `None` or empty clears it.

        Examples:
            - Attach credentials to a dataset opened without them:
                ```python
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.read_file("tests/data/acc4000.tif")
                >>> ds.attach_gdal_env({"AWS_REQUEST_PAYER": "requester"})
                >>> ds.gdal_env["AWS_REQUEST_PAYER"]
                'requester'

                ```
            - Clearing it leaves the dataset reading with no extra config:
                ```python
                >>> ds = Dataset.read_file("tests/data/acc4000.tif")
                >>> ds.attach_gdal_env({"AWS_REQUEST_PAYER": "requester"})
                >>> ds.attach_gdal_env(None)
                >>> ds.gdal_env
                {}

                ```
        """
        self._gdal_env = dict(gdal_env) if gdal_env else {}

    def __reduce__(self):
        """Return a recipe tuple that re-opens the dataset on unpickle.

        Serialising a live `gdal.Dataset` pointer is not possible
        (native C++ handle, no copy semantics). Instead we emit the
        recipe `(class, file_name, access, gdal_env, open_options)` and
        reconstruct on unpickle by calling `cls.read_file(path, ...)`
        under the captured GDAL config, so a signed remote dataset
        re-opens on the worker with its credentials and driver options.

        The GDAL handle is therefore opened **on the receiving process
        / thread**, which is the invariant dask.distributed needs.

        Recipes written before these fields existed still unpickle here (the
        parameter defaults for `gdal_env` and `open_options`), but a recipe
        written by *this* version needs a reader that accepts five arguments — so
        a mixed-version cluster has to upgrade the workers, not only the client.

        Security:
            The recipe carries :attr:`gdal_env` **and** :attr:`open_options`
            verbatim, so pickling a dataset opened with credentials — whether in
            the config or as a driver open option that some drivers accept a
            secret through — serialises them. `dask.distributed` spills graphs to
            disk and quotes task keys in error reports — treat such a pickle as a
            secret, and prefer a short-lived token.

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
        return (
            _reconstruct_dataset,
            (
                type(self),
                path,
                self._access,
                dict(self._gdal_env),
                tuple(self._open_options),
            ),
        )

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

    def _require_writable(self, action: str) -> None:
        """Guard a metadata write that would spill a PAM sidecar to a read-only file.

        The metadata setters (crs, epsg, scale, offset, band_units, meta_data) call
        GDAL `Set*` with no access check, so on a read-only **on-disk** file GDAL
        silently mutates the in-memory object and spills a PAM `.aux.xml` sidecar next
        to it. Call this at the top of every metadata setter so they fail loudly
        instead. (A pixel-write's GDAL-error shim cannot catch a pure metadata `Set*`
        — GDAL just writes PAM — so an explicit check is the only reliable guard, and
        the SWIG binding exposes no handle-level `GetAccess`.)

        The condition is "read-only access **and** backed by a real on-disk file":
        an in-memory dataset (`copy`, `from_array()` with no path, a `/vsimem/`
        raster, a `MEM`-driver container) reports `access == "read_only"` by
        constructor default yet is a genuinely writable in-RAM handle that cannot spill
        a sidecar, so it is *not* blocked — that is the established
        edit-in-memory-then-save workflow. Only a real on-disk file opened read-only
        can spill PAM, and only that is rejected. In-memory is detected by the `MEM`
        driver or a `/vsimem/` / empty path (a `MEM` container may still carry a
        placeholder `file_name` like `'netcdf'`, so the driver is the reliable signal).

        Args:
            action: Short phrase completing "...read_only=False to <action>." — used
                in the error message (e.g. "set the CRS").

        Raises:
            ReadOnlyError: The dataset is a read-only on-disk file.
            RuntimeError: The dataset has been closed.
        """
        self._require_open()
        file_name = self._file_name
        in_memory = (
            not file_name
            or file_name.startswith("/vsimem/")
            or self.driver_type == "memory"
        )
        if self._access == "read_only" and not in_memory:
            raise ReadOnlyError(
                "The Dataset is opened read-only. Please read the dataset using "
                f"read_only=False to {action}."
            )

    def _require_open(self) -> None:
        """Guard a state read against a closed dataset.

        After `close()` (or context-manager exit) `self._raster` is `None`; a
        state-reading member that dereferences it then raises a bare
        `AttributeError` (or, for `__repr__`, silently returns `'None'`). Call this
        first so every such member raises the same clear error instead.

        Scope: this guard is wired into the members ARC-43 flagged — the ones that
        misbehaved most visibly (`meta_data`, `driver_type`, `_iloc`), plus
        `__str__`/`__repr__` (which return the `<Dataset: closed>` sentinel). It is
        deliberately not applied to every GDAL-dereferencing reader (`crs`, `bbox`,
        `scale`, `offset`, ...); those still raise a bare `AttributeError` on a closed
        handle as before — broadening the guard there is future work, not a regression.

        Raises:
            RuntimeError: The dataset has been closed.
        """
        if self._raster is None:
            raise RuntimeError(
                "Cannot use a closed dataset. "
                "The dataset has been closed via close() or a context manager."
            )

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
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.ones((4, 4)),
                ...     geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
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
        rows: np.typing.ArrayLike,
        cols: np.typing.ArrayLike,
        *,
        center: bool = True,
    ) -> tuple[Any, Any]:
        """Return the map coordinates ``(x, y)`` of array cells.

        The array-style companion of :meth:`rowcol`. Computed from the
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
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.ones((4, 4)),
                ...     geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
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
                >>> ds = Dataset.from_array(
                ...     np.ones((4, 4)),
                ...     geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
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
        scalar = np.ndim(rows) == 0 and np.ndim(cols) == 0
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
        x: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
    ) -> tuple[Any, Any]:
        """Return the array indices ``(row, col)`` of map coordinates.

        The array-style companion of :meth:`xy`. Computed from the exact
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
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.ones((4, 4)),
                ...     geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
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
                >>> ds = Dataset.from_array(
                ...     np.ones((4, 4)),
                ...     geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
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
        scalar = np.ndim(x) == 0 and np.ndim(y) == 0
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
            # Return Python lists for sequence input, matching xy()
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
        self._require_open()
        return self._raster.GetMetadata()

    @property
    def subdatasets(self) -> list[SubDataset]:
        """The container's subdatasets (nested rasters), in GDAL's order.

        A normal raster returns ``[]``. A *container* — a NetCDF/HDF/Zarr store, a
        Sentinel-1/-2 product, a WMS endpoint — returns one entry per nested
        raster. Open one with
        :meth:`~pyramids.dataset.dataset.Dataset.open_subdataset` (keeps this
        class) or :meth:`SubDataset.open` (a base ``Dataset``).

        Returns:
            list[SubDataset]: One :class:`~pyramids.dataset._subdataset.SubDataset`
            per nested raster, in the order GDAL lists them; ``[]`` for a raster
            that has none.

        Examples:
            - A plain raster has no subdatasets:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.zeros((2, 2)),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> ds.subdatasets
                []

                ```
        """
        self._require_open()
        return subdatasets_of(self._raster)

    @property
    def meta_data_domains(self) -> list[str]:
        """The GDAL metadata domains this dataset actually exposes.

        Returns:
            list[str]: Domain names — ``""`` is the default domain, and named
            domains such as ``"IMAGE_STRUCTURE"``, ``"SUBDATASETS"``, ``"RPC"`` or
            an ``xml:*`` product-XML domain appear when present. ``[]`` when the
            handle exposes none (GDAL's ``None`` is normalised away).
        """
        self._require_open()
        return cast("list[str]", self._raster.GetMetadataDomainList() or [])

    def get_meta_data(self, domain: str = "") -> dict[str, str] | list[str]:
        """Read this dataset's metadata from a specific GDAL domain.

        The domain-aware companion to :attr:`meta_data`. On a base :class:`Dataset`,
        ``get_meta_data("")`` returns the same mapping as :attr:`meta_data`; on a
        ``NetCDF`` the two differ — there :attr:`meta_data` is a processed
        ``NetCDFMetadata`` while this returns the raw default-domain dict. Other
        domains expose GDAL's named metadata, e.g. ``"IMAGE_STRUCTURE"``
        (``COMPRESSION`` / ``INTERLEAVE`` / ``NBITS``) or ``"RPC"``. Use
        :attr:`meta_data_domains` to see which domains exist.

        Args:
            domain: The GDAL metadata domain to read; ``""`` (default) is the
                default domain.

        Returns:
            dict[str, str] | list[str]: The domain's metadata. Most domains return a
            ``KEY=VALUE`` mapping (``{}`` when the domain is absent); an ``xml:*``
            domain returns a list of one XML string, mirroring GDAL's ``GetMetadata``.

        Examples:
            - The default domain matches :attr:`meta_data` on a base ``Dataset``:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.zeros((2, 2)),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> ds.get_meta_data() == ds.meta_data
                True

                ```
        """
        self._require_open()
        return cast("dict[str, str] | list[str]", self._raster.GetMetadata(domain))

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

        Hosted on `RasterBase` so any collaborator can resolve
        `self._ds._iloc(i)` through its back-reference to the dataset,
        without `RasterBase` having to know about the collaborators. The
        `Bands` collaborator keeps an identical `_iloc` of its own for its
        internal calls (`engines/bands.py`); the two bodies are equivalent.

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
        self._require_open()
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
              >>> from pyramids.dataset import Dataset
              >>> dataset = Dataset.read_file("examples/data/acc4000.tif")
              >>> size = dataset.block_size
              >>> print(size)
              [[128, 128]]

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
    ) -> Generator[Window]:
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
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.ones((5, 5)),
                ...     geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
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
                >>> ds = Dataset.from_array(
                ...     np.ones((6, 6)),
                ...     geo_ref=GeoReference(top_left_corner=(0, 6), cell_size=1.0, epsg=4326),
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
    ) -> Generator[tuple[Window, np.typing.NDArray]]:
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
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> src_arr = np.arange(25, dtype="float32").reshape(5, 5)
                >>> ds = Dataset.from_array(
                ...     src_arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
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
        self._require_open()
        drv = self.raster.GetDriver()
        driver_type = drv.GetDescription() if drv is not None else None
        return CATALOG.get_driver_name(driver_type)

    @classmethod
    @abstractmethod
    def read_file(
        cls,
        path: str | Path,
        read_only=True,
        file_i: int = 0,
        *,
        open_options: dict[str, str] | list[str] | tuple[str, ...] | None = None,
        warn_on_container: bool = True,
    ) -> RasterBase:
        """Read file.

        Args:
            path (str):
                Path of file to open.
            read_only (bool):
                File mode, set as False, to open in "update" mode.
            file_i (int):
                Which member to open when ``path`` is a multi-file archive.
                Default ``0``.
            open_options (dict | list | tuple | None):
                GDAL open options as a mapping or ``["KEY=VALUE"]`` sequence,
                forwarded to the driver and captured on the returned instance so
                the reopen paths reapply them. Default ``None`` — no options.
            warn_on_container (bool):
                Whether to warn when the path opens to a subdataset container (a
                0-band raster whose payload is nested subdatasets) rather than
                returning it silently. Part of the contract so the pickle
                reconstruct can suppress it; concrete readers may honour or ignore
                it. Default ``True``.

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
              >>> from pyramids.dataset import Dataset
              >>> dataset = Dataset.read_file("examples/data/acc4000.tif")
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
        overview: bool = False,
        overview_index: int = 0,
        rgb_options: dict | None = None,
        *,
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs,
    ):
        """Plot.

            - plot the values/overviews of a given band.

        Args:
            band (int, optional):
                The band you want to get its data. Default is 0.
            exclude_value (Any, optional):
                Value to exclude from the plot. Default is None.
            overview (bool, optional):
                True if you want to plot the overview. Default is False.
            overview_index (int, optional):
                Index of the overview. Default is 0.
            rgb_options (dict, optional):
                Grouped Sentinel-imagery options for a true-colour composite. Accepted
                keys: ``"rgb"`` (band indices), ``"surface_reflectance"``, ``"cutoff"``,
                ``"percentile"``. Default is ``None``.
            fig (matplotlib.figure.Figure, optional):
                Draw into this figure instead of creating one. Pass it alongside ``ax``;
                supplying ``fig`` on its own currently raises inside cleopatra
                (serapeum-org/cleopatra#326). Default is ``None``.
            ax (matplotlib.axes.Axes, optional):
                Draw into these axes instead of creating them, which is what lets several
                rasters share one figure. An axes already carries its figure, so ``ax`` on
                its own is sufficient. Default is ``None``.
            **kwargs: Additional plotting options.
                points (array | PointOverlay):
                    Point overlay. A 3 column array with the first column as the value you want to display for
                    the point, the second is the rows index of the point in the array, and the third column as
                    the column index in the array. The second and third columns tell the location of the point
                    in the array. To style the points, pass a
                    ``pyramids.plot.PointOverlay(points, color=..., size=..., label_color=...,
                    label_size=...)`` instead of a bare array; the loose ``point_*`` / ``pid_*``
                    styling kwargs were removed — set the styling on the ``PointOverlay``.
                figsize (tuple, optional):
                    Figure size. The default is (8, 8).
                title (str, optional):
                    Title of the plot. The default is 'Total Discharge'.
                title_size (int, optional):
                    Title size. The default is 15.
                colorbar (bool | ColorBar, optional):
                    Colour-bar spec ``pyramids.plot.ColorBar(label=..., length=..., orientation=...,
                    label_size=..., label_rotation=..., label_location=..., ticks_spacing=...)``.
                    The loose ``cbar_*`` / ``ticks_spacing`` kwargs it replaces were removed —
                    passing one now raises a ``ValueError``. ``False`` hides the bar, ``None``
                    uses the default.
                cmap (str, optional):
                    Color style. The default is 'coolwarm_r'.

                Colour-scale, contour, cell-value and data-style options moved onto the typed
                render groups (all re-exported from ``pyramids.plot``): pass
                ``color=ColorScaling(...)`` (linear / power / sym-log / boundary / midpoint
                norm), ``contour=Contour(levels=..., ...)``, ``cells=CellValues(show=...,
                size=..., background_threshold=...)`` and ``data_style=DataStyle(style=...,
                hillshade=...)``. The loose forms they replace (``color_scale`` / ``gamma`` /
                ``line_threshold`` / ``line_scale`` / ``bounds`` / ``midpoint`` / ``levels`` /
                ``display_cell_value`` / ``num_size`` / ``background_color_threshold`` /
                ``style`` / ``hillshade``) were removed and now raise a ``ValueError``.

        Returns:
            ArrayGlyph:
                A cleopatra ``ArrayGlyph`` wrapping the rendered figure. Use
                ``glyph.fig`` / ``glyph.ax`` to drop down to raw matplotlib.
        """
        pass

    @classmethod
    @abstractmethod
    def from_array(
        cls,
        arr: np.ndarray,
        *,
        geo_ref: GeoReference,
        no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
        path: str | Path | None = None,
    ):
        """Create a dataset from an array.

        These four parameters are the contract every concrete raster class
        honours, so a caller holding the declared base type can build either a
        :class:`~pyramids.dataset.Dataset` or a
        :class:`~pyramids.netcdf.NetCDF` with the same call. A subclass may add
        keyword-only parameters of its own (``NetCDF`` adds ``variable_name``,
        ``dims``, ``encoding`` and ``attrs``), but it must keep accepting these
        and must declare them — overriding with a bare ``*args, **kwargs`` hides
        an incompatible signature from static checkers, which is how the two
        constructors previously came to accept different keyword sets.

        Args:
            arr (np.ndarray):
                The array to wrap. A 2-D array is a single band; the leading
                axis of a 3-D array indexes bands.
            geo_ref (GeoReference):
                How the array maps to space — an affine ``geo`` transform, or a
                ``top_left_corner`` + ``cell_size``, plus the ``epsg``. Required
                and keyword-only; a raster has to be placed somewhere.
            no_data_value (Any, optional):
                No data value to mask the cells out of the domain. The default
                is -9999.
            path (str | Path, optional):
                Destination, which alone decides the driver. `None` (default)
                builds the raster in memory; otherwise the extension selects the
                format (``.tif`` -> GTiff, ``.nc`` -> netCDF, …).

        Returns:
            RasterBase:
                The newly created dataset. Concrete classes narrow this — see
                :meth:`pyramids.dataset.Dataset.from_array` and
                :meth:`pyramids.netcdf.NetCDF.from_array`.
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

        Raises:
            TypeError: The dataset is backed by an ASCII driver, which cannot store a CRS.
            ValueError: Neither `crs` nor `epsg` was provided. Validated before the
                read-only guard, so an invalid call reports this regardless of access mode.
            ReadOnlyError: The dataset is opened read-only.
        """
        # ASCII cannot store a CRS in any mode, so that TypeError takes precedence
        # over the read-only guard below.
        if self.driver_type == "ascii":
            raise TypeError(
                "Setting CRS for ASCII file is not possible, you can save the files to a geotiff and then reset the crs"
            )
        # Validate the arguments before the read-only guard so an invalid call
        # (neither crs nor epsg) reports the actionable ValueError regardless of
        # access mode, rather than a ReadOnlyError that hides the real mistake.
        if crs is None and epsg is None:
            raise ValueError("Either crs or epsg must be provided.")
        self._require_writable("set the CRS")
        # first change the projection of the gdal dataset object
        # second change the epsg attribute of the Dataset object
        if crs is not None:
            self.raster.SetProjection(crs)
            # An empty projection means "no CRS", which propagates as None
            # rather than being tagged WGS 84 (ARC-26).
            self._epsg = epsg_of_crs(crs)
        else:
            # crs is None here, so epsg is not None (the neither-None check above
            # rejects both being None); cast narrows it for the type checker without
            # a redundant always-true runtime condition.
            sr = sr_from_epsg(cast(int, epsg))
            self.raster.SetProjection(sr.ExportToWkt())
            self._epsg = epsg

    @abstractmethod
    def to_crs(
        self,
        to_epsg: int,
        method: str = DEFAULT_RESAMPLING,
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
              >>> dataset = Dataset.read_file("examples/data/acc4000.tif")
              >>> reprojected_dataset = dataset.to_crs(to_epsg=3857)
              >>> reprojected_dataset.epsg
              3857

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
              >>> import os, tempfile
              >>> from pyramids.dataset import Dataset
              >>> dataset = Dataset.read_file("examples/data/acc4000.tif")
              >>> out_path = os.path.join(tempfile.mkdtemp(), "saved.tif")
              >>> dataset.to_file(out_path)
              >>> os.path.exists(out_path)
              True

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
    ) -> None:
        """Create overviews for the dataset.

        Args:
            resampling_method (str, optional):
                The resampling method used to create the overviews, by default "nearest".
                Possible values are:
                    "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE",
                    "AVERAGE_MAGPHASE", "RMS", "BILINEAR".
            overview_levels (list, optional):
                The overview levels, as reduction factors restricted to the powers of two from 2 to 2048.
                Defaults to the full set.

        Returns:
            None:
                The levels are built on the dataset itself; read the count per band from `overview_count`.

        Raises:
            TypeError:
                `overview_levels` is not a list.
            ValueError:
                `overview_levels` holds a factor outside the supported set, or `resampling_method` is not one of
                the allowed values.
            OverviewTargetError:
                Checked *before* the arguments, since no argument value can make this dataset work, so a call
                that is wrong in both ways reports this rather than the `TypeError` / `ValueError` above.
                The dataset is a plain VRT whose description is not a path — an empty one, a blank one, or inline
                VRT XML. A plain VRT owns no pixel storage, so its overviews can only go to an external sidecar,
                and there is nothing to name one after; save it with `to_file(path)` and build the levels on the
                saved raster. A *warped* VRT is exempt: it holds its overviews in RAM. Subclasses `ValueError`.
            RuntimeError:
                GDAL failed to build the levels.

        Notes:
            - External (.ovr file): if the dataset is read with `read_only=True` then the overviews' file is
              created in the same directory as the dataset, with the same name and an `.ovr` extension.
            - Internal: for a format that supports internal overviews, reading with `read_only=False` puts them
              inside the dataset, which then needs to be saved/flushed to persist them to disk. A *plain* VRT has
              no internal storage, so its levels go to an external sidecar in either access mode; a warped VRT
              holds them in RAM and writes no sidecar at all.
            - On a **warped** VRT, `resampling_method` has no effect: GDAL resamples those levels with the
              warper's own algorithm, so `"average"` and `"nearest"` produce identical pixels. Build the levels
              on a saved raster instead if the method matters.
        """
        pass

    @abstractmethod
    def recreate_overviews(self, resampling_method: str = "nearest") -> None:
        """Recreate overviews for the dataset.

        Regenerates the existing overviews in place; it never builds new ones — call
        `create_overviews` for that. A band with no overviews has nothing to regenerate,
        so it is reported through a warning rather than skipped silently.

        A band's levels are rebuilt in a single GDAL pass, which cascades: each deeper
        level is decimated from the level above rather than from the full-resolution
        band, matching what `create_overviews` does. A level >= 1 therefore holds what a
        per-level rebuild wrote only where the resampling survives being applied twice --
        `nearest` always, and `average`/`rms` on a floating-point band with no no-data.
        See `docs/migration.md`.

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
            OverviewTargetError:
                The dataset cannot hold regenerated levels, for either of two reasons. The first is checked
                *before* the arguments, since no argument value can make such a dataset work, so a call that is
                wrong in both ways reports this rather than the `ValueError` above.
                It is a plain VRT whose
                description is not a path — an empty one, a blank one, or inline VRT XML — so there is nothing to
                name an external sidecar after; save it with `to_file(path)` and build the levels there with
                `create_overviews()` (`to_file` does not carry overviews, so there is nothing to regenerate).
                Or the levels a band exposes are owned by a VRT, which computes them on read instead of storing
                them — a *warped* VRT's bands, or the levels a plain VRT inherits from the source it wraps — so
                no access mode makes them writable; give the handle levels of its own with `create_overviews()`.
                Unlike `create_overviews`, a warped VRT is **not** exempt here. Also raised when the levels are
                *stored* yet GDAL refuses them on a dataset already open for writing: they are reached through a
                source GDAL opens read-only, so there is no reopen left to advise — regenerate on the raster that
                owns them. Subclasses `ValueError`.
            ReadOnlyError:
                If GDAL refuses to rewrite the overviews the call targets, those levels are *stored*, and this
                handle is open read-only — internal overviews inside a read-only dataset, or an external .ovr
                that a later handle reopened read-only. Please read the dataset using read_only=False. That is
                the one shape where reopening is worth trying, not a promise it will succeed: a VRT serving an
                explicit `<Overview>` owns a real .ovr that GDAL opens read-only whatever the parent's mode, and
                reopening turns this into the `OverviewTargetError` above. A level a VRT *computes* is separated
                out the same way, since GDAL reports it with the same error number. Two spellings refuse with a
                different number instead (`CPLE_AppDefined`): a VRT carrying its own `<OverviewList>`, in either
                access mode, and a writable handle whose .ovr is itself a VRT. Both surface as GDAL's own
                `RuntimeError`, which already names the cause.
            RuntimeError:
                Any other GDAL regeneration failure, so a disk-full, corrupt-overview or transport failure
                is not relabelled as an access-mode error. GDAL's own error is re-raised carrying a note
                that names the band it stopped on — a band's levels regenerate in one call, so no level is
                named; a failing status that raised nothing is turned into one.

        Note:
            Bands are regenerated in order and the exceptions above are raised on the first band that fails, so
            earlier bands — and, within the failing band, earlier levels — may already have been rewritten. The
            dataset is not rolled back.

        Warns:
            UserWarning:
                No band has overviews, so there is nothing to regenerate; or only some
                bands have them, and the empty ones were skipped. Also when the dataset
                has no bands at all. None of these fire on a *plain* pathless VRT — that raises
                `OverviewTargetError` first, since "call create_overviews() to build them" is advice it would
                also refuse. A warped VRT is exempt from that guard, so an empty one still warns; its refusal
                comes only from the regeneration attempt itself.
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
    def get_overview_dataset(self, band: int | None = None, overview_index: int = 0):
        """Get an overview level as a standalone Dataset.

        The `gdal.Band` from `get_overview` carries no geotransform and no CRS; this
        returns the same pixels as a read-only `Dataset` view whose cell size is scaled
        by the decimation factor. The caller owns the returned handle.

        Args:
            band (int | None, optional):
                The band to take; None (the default) keeps every band.
            overview_index (int):
                Index of the overview. Default is 0.

        Returns:
            Dataset:
                The overview level, carrying the parent's CRS, no-data value and band
                metadata.
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
