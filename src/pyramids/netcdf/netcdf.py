"""
netcdf module.

netcdf contains python functions to handle netcdf data. gdal class: https://gdal.org/api/index.html#python-api.
"""

from __future__ import annotations

import os
import tempfile
import weakref
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal

from pyramids import _io
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import numpy_to_gdal_dtype
from pyramids.base.crs import sr_from_epsg
from pyramids.base.protocols import ArrayLike
from pyramids.dataset import DEFAULT_NO_DATA_VALUE, Dataset
from pyramids.netcdf._kerchunk import combine_kerchunk, to_kerchunk
from pyramids.netcdf._lazy import _apply_unpack, build_lazy_array
from pyramids.netcdf._mfdataset import open_mfdataset
from pyramids.netcdf.cf import (
    build_coordinate_attrs,
    srs_to_grid_mapping,
    write_attributes_to_md_array,
    write_global_attributes,
)
from pyramids.netcdf.dimensions import DimMetaData
from pyramids.netcdf.metadata import get_metadata
from pyramids.netcdf.models import NetCDFMetadata
from pyramids.netcdf.utils import create_time_conversion_func


class _LazyVariableDict(dict):
    """Dict that loads NetCDF variables on first access per key.

    Avoids the cost of calling `get_variable()` (which does
    `AsClassicDataset` + Y-flip) for every variable upfront.
    Only the variables actually accessed are loaded.

    Note:
        This class is **not thread-safe**. Concurrent access from
        multiple threads may cause `get_variable()` to be called
        more than once for the same key. Use external locking if
        thread-safety is required.
    """

    def __init__(self, nc: NetCDF) -> None:
        super().__init__()
        self._nc = nc
        self._names: list[str] = nc.get_variable_names()

    def __getitem__(self, key: str) -> NetCDF:
        if not dict.__contains__(self, key) and key in self._names:
            dict.__setitem__(self, key, self._nc.get_variable(key))
        return dict.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> NetCDF | Any:
        if key in self._names:
            return self[key]
        return default

    def __contains__(self, key: object) -> bool:
        return key in self._names

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self):
        return iter(self._names)

    def keys(self) -> list[str]:
        return self._names

    def values(self) -> list[NetCDF]:
        return [self[k] for k in self._names]

    def items(self) -> list[tuple[str, NetCDF]]:
        return [(k, self[k]) for k in self._names]


def _reconstruct_netcdf(
    path: str,
    access: str,
    is_md_array: bool,
    is_subset: bool,
    source_var_name: str | None,
) -> NetCDF:
    """Re-open a :class:`NetCDF` from its pickle recipe tuple.

    Called by :meth:`NetCDF.__reduce__` on unpickle. Carries four
    bits of extra state beyond the base :class:`RasterBase`
    recipe so the reconstructed instance retains identity:

    * `is_md_array` — was the file opened via
      :data:`gdal.OF_MULTIDIM_RASTER` (MDIM mode) or classic mode?
    * `is_subset` — is this instance a container or a single variable?
    * `source_var_name` — when `is_subset` is True, the variable
      path to re-traverse via :meth:`NetCDF.get_variable`.

    Args:
        path: On-disk path or VSI URL to re-open.
        access: `"read_only"` opens read-only; any other value opens
            for update.
        is_md_array: Whether to pass `open_as_multi_dimensional=True`
            to :meth:`NetCDF.read_file`.
        is_subset: If True and `source_var_name` is not None, the
            rebuilt container is then drilled into via
            :meth:`NetCDF.get_variable` before return.
        source_var_name: Variable path for the subset drill-down.

    Returns:
        NetCDF: Container or variable-subset instance.
    """
    read_only = access == "read_only"
    container = NetCDF.read_file(
        path,
        read_only=read_only,
        open_as_multi_dimensional=is_md_array,
    )
    if is_subset and source_var_name is not None:
        result = container.get_variable(source_var_name)
    else:
        result = container
    return result


class NetCDF(Dataset):
    """NetCDF.

    NetCDF class is a recursive data structure or self-referential object.
    The NetCDF class contains methods to deal with NetCDF files.

    NetCDF Creation guidelines:
        https://acdguide.github.io/Governance/create/create-basics.html
    """

    def __reduce__(self):  # type: ignore[override]
        """Emit the extended recipe tuple carrying NetCDF mode flags.

        Overrides :meth:`RasterBase.__reduce__` to include
        `_is_md_array`, `_is_subset`, and `_source_var_name`,
        which are required to reconstruct a container vs a
        variable-subset with matching identity.

        For variable-subset instances the `_file_name` attribute
        reflects the subset's GDAL description, which is typically
        empty or driver-specific. We therefore fall back to the
        parent container's `_file_name` when reconstructing a
        subset.

        Raises:
            TypeError: The NetCDF has no on-disk path (empty
                `_file_name` or a `/vsimem/` path). Pickling an
                in-memory NetCDF is not supported.
        """
        path = self._file_name
        if (not path) and self._is_subset:
            parent = getattr(self, "_parent_nc", None)
            if parent is not None:
                path = parent._file_name
        if not path or path.startswith("/vsimem/"):
            raise TypeError(
                f"NetCDF has no on-disk path (file_name={self._file_name!r}); "
                "pickling an in-memory NetCDF is not supported. Call "
                ".to_file(path) first to anchor it to disk."
            )
        return (
            _reconstruct_netcdf,
            (
                path,
                self._access,
                bool(self._is_md_array),
                bool(self._is_subset),
                self._source_var_name,
            ),
        )

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        open_as_multi_dimensional: bool = True,
    ):
        """Initialize a NetCDF dataset wrapper.

        Args:
            src: A GDAL dataset handle (either classic or multidimensional).
            access: Access mode, either `"read_only"` or `"write"`.
                Defaults to `"read_only"`.
            open_as_multi_dimensional: If True the dataset was opened with
                `gdal.OF_MULTIDIM_RASTER` and supports groups, MDArrays,
                and dimensions. If False it was opened in classic raster
                mode (subdatasets, bands). Defaults to True.
        """
        super().__init__(src, access=access)
        # set the is_subset to false before retrieving the variables
        if open_as_multi_dimensional:
            self._is_md_array = True
            self._is_subset = False
        else:
            self._is_md_array = False
            self._is_subset = False
        # Caches (invalidated by _replace_raster, add_variable, remove_variable)
        self._cached_variables: dict[str, NetCDF] | None = None
        self._cached_meta_data: NetCDFMetadata | None = None
        # Origin-tracking attributes set by get_variable (RT-4)
        self._parent_nc: NetCDF | None = None
        self._source_var_name: str | None = None
        self._gdal_md_arr_ref: Any = None
        self._gdal_rg_ref: Any = None
        self._md_array_dims: list[str] = []
        self._band_dim_name: str | None = None
        self._band_dim_values: list[Any] | None = None
        self._variable_attrs: dict[str, Any] = {}
        self._scale: float | None = None
        self._offset: float | None = None

    def _update_inplace(  # type: ignore[override]
        self, src: gdal.Dataset, access: str | None = None
    ) -> None:
        """Swap internal state, preserving NetCDF-specific attributes.

        The base `Dataset._update_inplace` rebuilds via
        `type(self)(src, access)` and overwrites `self.__dict__`.
        For a NetCDF that runs `NetCDF.__init__` with a default
        `open_as_multi_dimensional=True`, which would reset
        `_is_md_array` to True and clear every variable-subset
        attribute. This override snapshots the subset state, runs the
        base swap with the current MDIM mode, then restores the
        snapshot — so a variable subset stays a subset across
        `set_crs`, `apply(inplace=True)`, `change_no_data_value`,
        and the `epsg` setter.
        """
        preserved = {
            "_is_md_array": self._is_md_array,
            "_is_subset": self._is_subset,
            "_parent_nc": self._parent_nc,
            "_source_var_name": self._source_var_name,
            "_gdal_md_arr_ref": self._gdal_md_arr_ref,
            "_gdal_rg_ref": self._gdal_rg_ref,
            "_md_array_dims": self._md_array_dims,
            "_band_dim_name": self._band_dim_name,
            "_band_dim_values": self._band_dim_values,
            "_variable_attrs": self._variable_attrs,
            "_scale": self._scale,
            "_offset": self._offset,
        }
        new = NetCDF(
            src,
            access=access or self._access,
            open_as_multi_dimensional=self._is_md_array,
        )
        self.__dict__.update(new.__dict__)
        self.__dict__.update(preserved)
        # collaborators in `new.__dict__` point at
        # `new` via `weakref.proxy`; re-bind to a proxy of `self`
        # so callers using `self.spatial.crop(...)` after this update
        # reach the surviving instance instead of the discarded `new`.
        self_proxy = weakref.proxy(self)
        for attr in ("io", "spatial", "bands", "analysis", "cell", "vectorize", "cog"):
            collab = self.__dict__.get(attr)
            if collab is not None:
                collab._ds = self_proxy

    def __str__(self):
        """Return a human-readable summary of the NetCDF dataset."""
        message = f"""
            Cell size: {self.cell_size}
            Dimension: {self.rows} * {self.columns}
            EPSG: {self.epsg}
            projection: {self.crs}
            Variables: {self.variable_names}
            Metadata: {self.meta_data}
            File: {self.file_name}
        """
        return message

    def __repr__(self):
        """__repr__."""
        return super().__repr__()

    @property
    def top_left_corner(self):
        """Top left corner coordinates."""
        xmin, _, _, ymax, _, _ = self._geotransform
        return xmin, ymax

    @property
    def lon(self) -> np.ndarray:
        """Longitude / x-coordinate values as a 1D array.

        Looks for a variable named `"lon"` first, then `"x"`.

        Returns:
            np.ndarray or None: Flattened coordinate array, or None if
            neither `lon` nor `x` exists in the dataset.
        """
        lon = self._read_variable("lon")
        if lon is None:
            lon = self._read_variable("x")

        result: np.ndarray
        if lon is not None:
            result = lon.reshape(lon.size)
        else:
            result = super().lon
        return result

    @property
    def lat(self) -> np.ndarray:
        """Latitude / y-coordinate values as a 1D array.

        Looks for a variable named `"lat"` first, then `"y"`.

        Returns:
            np.ndarray or None: Flattened coordinate array, or None if
            neither `lat` nor `y` exists in the dataset.
        """
        lat = self._read_variable("lat")
        if lat is None:
            lat = self._read_variable("y")

        result: np.ndarray
        if lat is not None:
            result = lat.reshape(lat.size)
        else:
            result = super().lat
        return result

    @property
    def x(self) -> np.ndarray:
        """x-coordinate/longitude."""
        # X_coordinate = upper-left corner x + index * cell size + cell-size/2
        return self.lon

    @property
    def y(self) -> np.ndarray:
        """y-coordinate/latitude."""
        # Y_coordinate = upper-left corner y - index * cell size - cell-size/2
        return self.lat

    @property
    def geotransform(self):
        """Geotransform.

        Computes from lon/lat coordinate arrays if available.
        Falls back to the parent GDAL GetGeoTransform() otherwise.
        """
        if self.lon is not None and self.lat is not None:
            return (
                self.lon[0] - self.cell_size / 2,
                self.cell_size,
                0,
                self.lat[0] + self.cell_size / 2,
                0,
                -self.cell_size,
            )
        return self._geotransform

    @property
    def variable_names(self) -> list[str]:
        """Names of data variables (excluding dimension coordinate arrays).

        Returns:
            list[str]: Variable names. For MDIM mode these come from
            `GetMDArrayNames()` minus dimension names; for classic mode
            from `GetSubDatasets()`.
        """
        return self.get_variable_names()

    @property
    def variables(self) -> dict[str, NetCDF]:
        """All data variables as a lazy dict of `{name: NetCDF}` subsets.

        Variables are loaded on first access per key, not all at once.
        Cached after loading; invalidated by `add_variable` /
        `remove_variable` / `set_variable`.

        Returns:
            dict[str, NetCDF]: Mapping from variable name to its subset.
        """
        if self._cached_variables is None:
            self._cached_variables = _LazyVariableDict(self)
        return self._cached_variables

    @property
    def no_data_value(self):
        """No data value that marks the cells out of the domain."""
        return self._no_data_value

    @no_data_value.setter
    def no_data_value(self, value: list | Number):
        """Set the no-data value that marks cells outside the domain.

        The setter only changes the `no_data_value` attribute; it does
        **not** modify the underlying cell values. Use this to align the
        attribute with whatever sentinel is already stored in the cells.
        To actually rewrite cell values, use `change_no_data_value`.

        Args:
            value: New no-data value. A single number applied to all
                bands, or a list with one value per band.
        """
        if isinstance(value, list):
            for i, val in enumerate(value):
                self.bands._change_no_data_value_attr(i, val)
        else:
            self.bands._change_no_data_value_attr(0, value)

    @property
    def file_name(self):
        """File path, with the `NETCDF:"path":var` prefix stripped if present.

        Returns:
            str: Clean file path without the NETCDF prefix.
        """
        if self._file_name.startswith("NETCDF"):
            name = self._file_name.split(":")[1][1:-1]
        else:
            name = self._file_name
        return name

    @property
    def time_stamp(self):
        """Time coordinate values parsed from the CF-compliant `time` variable.

        Returns:
            list[str] | None: Formatted time strings, or None if no time
                dimension with a `units` attribute is found.
        """
        return self.get_time_variable()

    def _check_not_container(self, operation: str):
        """Raise ValueError if this is a root MDIM container (not a variable subset)."""
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            raise ValueError(
                f"Spatial operations are not supported on the NetCDF container. "
                f"Use nc.get_variable('var_name').{operation}(...) instead."
            )

    def plot(self, band=None, **kwargs):
        """Plot a band of the dataset.

        Blocked on root MDIM containers — extract a variable first.

        Raises:
            ValueError: If called on a root MDIM container.
        """
        self._check_not_container("plot")
        return super().plot(band=band, **kwargs)

    def read_array(
        self,
        variable: str | None = None,
        band: int | None = None,
        window: list[int] | None = None,
        unpack: bool = False,
        *,
        chunks: Any = None,
        lock: Any = None,
    ) -> ArrayLike:
        """Read array from the dataset (eager by default, lazy with `chunks`).

        Args:
            variable: When this instance is a root MDIM container,
                the variable name to read. When the instance is
                already a variable subset (`nc.get_variable("x")`)
                this argument must be `None` — the variable is
                already pinned.
            band: Band index to read, or None for all bands. Only
                honored on the eager path (`chunks=None`).
            window: Spatial window to read. Only honored on the
                eager path.
            unpack: If True and the variable has CF `scale_factor`
                and/or `add_offset`, apply the transformation
                `real = raw * scale + offset`. Defaults to False.
                Applied lazily via :mod:`dask.array` arithmetic when
                `chunks` is given — the compute graph stays lazy
                until the caller materializes it.
            chunks: Chunking spec for a lazy return. `None` (the
                default) returns an eager :class:`numpy.ndarray` and
                preserves the legacy behavior. Any of `int`,
                `tuple`, `dict`, or the string `"auto"` switches
                to a :class:`dask.array.Array` backed by MDArray
                chunk reads. Defaults chunked at the variable's
                native `GetBlockSize` (see
                :attr:`pyramids.netcdf.models.VariableInfo.block_size`);
                a conservative `(1,..., rows, cols)` fallback is
                used when the driver doesn't advertise one.
            lock: Lock passed to the underlying
                :class:`pyramids.base._file_manager.CachingFileManager`.
                `None` → :func:`pyramids.base._locks.default_lock`
                (a :class:`SerializableLock`, or a
                `dask.distributed.Lock` when a client is active).
                `False` → :class:`pyramids.base._locks.DummyLock`.
                Only meaningful when `chunks` is not `None`.

        Returns:
            np.ndarray or dask.array.Array: The array data, eager
            (numpy) by default or lazy (dask) when `chunks` is
            supplied. The lazy array computes chunk-by-chunk through
            `md_arr.ReadAsArray(array_start_idx=starts, count=counts)`.

        Raises:
            ValueError: If called on a root MDIM container without a
                `variable` argument, or when a subset is called
                with a conflicting `variable` name.
            ImportError: If `chunks` is given but `dask` is not
                installed. Install the `[lazy]` extra.
        """
        is_container = (
            self._is_md_array and not self._is_subset and self.band_count == 0
        )
        if is_container:
            if variable is None:
                self._check_not_container("read_array")
            subset = self.get_variable(variable)
            return subset.read_array(
                band=band,
                window=window,
                unpack=unpack,
                chunks=chunks,
                lock=lock,
            )
        if variable is not None and variable != self._source_var_name:
            raise ValueError(
                f"This NetCDF instance is already pinned to variable "
                f"{self._source_var_name!r}; cannot re-read as "
                f"{variable!r}. Call read_array on the parent container "
                "instead."
            )
        if chunks is None:
            result = super().read_array(band=band, window=window)
            if unpack:
                scale = getattr(self, "_scale", None)
                offset = getattr(self, "_offset", None)
                if scale is not None or offset is not None:
                    result = result.astype(np.float64)
                    if scale is not None:
                        result = result * scale
                    if offset is not None:
                        result = result + offset
        else:
            parent = self._parent_nc if self._parent_nc is not None else self
            path = parent._file_name
            if path.startswith("NETCDF"):
                path = path.split(":")[1][1:-1]
            var_name = self._source_var_name
            if var_name is None:
                raise ValueError(
                    "Lazy read requires a variable name; pass "
                    "`variable=` on the container or call read_array "
                    "on a subset from `get_variable()`."
                )
            result = build_lazy_array(
                path=path,
                variable_name=var_name,
                chunks=chunks,
                lock=lock,
            )
            if unpack:
                result = _apply_unpack(
                    result,
                    getattr(self, "_scale", None),
                    getattr(self, "_offset", None),
                )
        return result

    def _preserve_netcdf_metadata(self, result: Dataset) -> NetCDF:
        """Wrap a Dataset result as a NetCDF, preserving variable-subset metadata.

        When spatial operations (crop, to_crs, resample) are called on a
        NetCDF variable subset, the parent `Dataset` mixin returns a
        plain `Dataset`. This helper re-wraps the result as a `NetCDF`
        and copies over the variable-specific attributes so that methods
        like `sel()`, `read_array(unpack=True)`, and further spatial
        operations continue to work with consistent return types.

        Args:
            result: The `Dataset` (or `NetCDF`) returned by a parent
                spatial operation.

        Returns:
            NetCDF: The same data wrapped as a `NetCDF` with all
                variable-subset metadata preserved.
        """
        if isinstance(result, NetCDF):
            wrapped = result
        else:
            wrapped = NetCDF(
                result._raster,
                access=result._access,
                open_as_multi_dimensional=False,
            )
        wrapped._is_md_array = self._is_md_array
        wrapped._is_subset = self._is_subset
        wrapped._band_dim_name = self._band_dim_name
        if (
            self._band_dim_values is not None
            and wrapped._band_count > 0
            and len(self._band_dim_values) != wrapped._band_count
        ):
            wrapped._band_dim_values = None
        else:
            wrapped._band_dim_values = self._band_dim_values
        wrapped._variable_attrs = self._variable_attrs
        wrapped._scale = self._scale
        wrapped._offset = self._offset
        wrapped._parent_nc = self._parent_nc
        wrapped._source_var_name = self._source_var_name
        wrapped._gdal_md_arr_ref = None
        wrapped._gdal_rg_ref = None
        return wrapped

    def crop(self, mask: Any, touch: bool = True) -> NetCDF:
        """Crop the dataset using a polygon or raster mask.

        On a **root MDIM container** this crops every variable and
        returns a new in-memory NetCDF container with the cropped
        results. On a **variable subset** it delegates to the
        parent `Dataset.crop()` and wraps the result as `NetCDF`
        to preserve variable metadata (`_band_dim_name`,
        `_band_dim_values`, `sel()`, etc.).

        Args:
            mask: GeoDataFrame with polygon geometry, or a Dataset
                to use as a spatial mask.
            touch: If True, include cells that touch the mask
                boundary. Defaults to True.

        Returns:
            NetCDF: Cropped container or variable subset.
        """
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            result = self._apply_to_all_variables(
                "crop",
                {"mask": mask, "touch": touch},
            )
        else:
            result = super().crop(mask=mask, touch=touch)
            result = self._preserve_netcdf_metadata(result)
        return result

    def _apply_to_all_variables(self, operation, op_kwargs):
        """Apply an operation to every variable in the container.

        Args:
            operation: Name of the Dataset method to call (e.g. "crop").
            op_kwargs: Keyword arguments to pass to the method.

        Returns:
            NetCDF: New container with the operation applied to all variables.

        Raises:
            ValueError: If the container has no data variables.
        """
        if not self.variable_names:
            raise ValueError(
                "Cannot apply operation to an empty container (no data variables)."
            )

        result = None
        for var_name in self.variable_names:
            var = self.get_variable(var_name)
            var_result = getattr(var, operation)(**op_kwargs)
            # to_crs returns a VRT — materialize before the source goes
            # out of scope. read_array also squeezes singleton-band 3-D
            # variables to 2-D, so re-expand when the variable carried a
            # band/time/level dim originally.
            var_arr = var_result.read_array()
            if var_arr.ndim == 2 and var._band_dim_name is not None:
                var_arr = np.expand_dims(var_arr, axis=0)
            var_ndv = var_result.no_data_value
            var_ndv_scalar = (
                var_ndv[0] if isinstance(var_ndv, list) and var_ndv else var_ndv
            )

            if result is None:
                # First variable: build the container.
                result = NetCDF.create_from_array(
                    arr=var_arr,
                    geo=var_result.geotransform,
                    epsg=var_result.epsg,
                    no_data_value=var_ndv_scalar,
                    variable_name=var_name,
                    extra_dim_name=var._band_dim_name or "time",
                    extra_dim_values=var._band_dim_values,
                )
            else:
                # Subsequent variables: drop into the existing container.
                ds = Dataset.create_from_array(
                    var_arr,
                    geo=var_result.geotransform,
                    epsg=var_result.epsg,
                    no_data_value=var_ndv_scalar,
                )
                ds._band_dim_name = var._band_dim_name
                ds._band_dim_values = var._band_dim_values
                result.set_variable(var_name, ds)

        return result

    def to_crs(
        self,
        to_epsg: int,
        method: str = "nearest neighbor",
        maintain_alignment: bool = False,
    ) -> NetCDF:
        """Reproject the dataset to a different CRS.

        On a **root MDIM container** this reprojects every variable
        and returns a new container. On a **variable subset** it
        delegates to `Dataset.to_crs()` and wraps the result as
        `NetCDF` to preserve variable metadata.

        Args:
            to_epsg: Target EPSG code (e.g., 4326, 32637).
            method: Resampling method. Defaults to `"nearest neighbor"`.
            maintain_alignment: If True, keep the same number of rows
                and columns. Defaults to False.

        Returns:
            NetCDF: Reprojected container or variable subset.
        """
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            result = self._apply_to_all_variables(
                "to_crs",
                {
                    "to_epsg": to_epsg,
                    "method": method,
                    "maintain_alignment": maintain_alignment,
                },
            )
        else:
            result = super().to_crs(
                to_epsg=to_epsg,
                method=method,
                maintain_alignment=maintain_alignment,
            )
            result = self._preserve_netcdf_metadata(result)
        return result

    def resample(
        self,
        cell_size: float,
        method: str = "nearest neighbor",
    ) -> NetCDF:
        """Resample the dataset to a different cell size.

        On a **root MDIM container** this resamples every variable
        and returns a new container. On a **variable subset** it
        delegates to `Dataset.resample()` and wraps the result as
        `NetCDF` to preserve variable metadata.

        Args:
            cell_size: New cell size.
            method: Resampling method. Defaults to `"nearest neighbor"`.

        Returns:
            NetCDF: Resampled container or variable subset.
        """
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            result = self._apply_to_all_variables(
                "resample",
                {"cell_size": cell_size, "method": method},
            )
        else:
            result = super().resample(
                cell_size=cell_size,
                method=method,
            )
            result = self._preserve_netcdf_metadata(result)
        return result

    def sel(self, **kwargs: Any) -> NetCDF:
        """Select a subset of bands by coordinate values.

        Extracts bands whose coordinate values match the given
        criteria. Works on variable subsets that have
        `_band_dim_name` and `_band_dim_values` set by
        `get_variable()`.

        The result is always a `NetCDF` instance with the same
        variable metadata preserved, so that `sel()` can be
        chained and NetCDF-specific methods like
        `read_array(unpack=True)` remain available.

        Args:
            **kwargs: One keyword argument where the key is the
                dimension name and the value is one of:

                - A single number: select one band by exact value.
                - A list of numbers: select multiple bands.
                - A `slice(start, stop)`: select bands where
                  `start <= coord <= stop`.

        Returns:
            NetCDF: A new NetCDF variable subset with only the
                selected bands and all variable metadata preserved.

        Raises:
            ValueError: If the dimension name doesn't match
                `_band_dim_name`, or no matching bands are found.

        Examples:
            Select a single time step::

                var.sel(time=6)

            Select multiple time steps::

                var.sel(time=[0, 12, 24])

            Select a range::

                var.sel(time=slice(6, 18))
        """
        if len(kwargs) != 1:
            raise ValueError("sel() requires exactly one keyword argument.")

        dim_name, selector = next(iter(kwargs.items()))

        if self._band_dim_name is None:
            raise ValueError(
                "sel() requires a variable with a non-spatial dimension. "
                "This variable has no band dimension tracked."
            )
        if dim_name != self._band_dim_name:
            raise ValueError(
                f"Dimension '{dim_name}' does not match the band "
                f"dimension '{self._band_dim_name}'."
            )
        if self._band_dim_values is None:
            raise ValueError(
                "No coordinate values available for dimension " f"'{dim_name}'."
            )

        coords = self._band_dim_values

        if isinstance(selector, slice):
            start = selector.start if selector.start is not None else coords[0]
            stop = selector.stop if selector.stop is not None else coords[-1]
            band_indices = [i for i, v in enumerate(coords) if start <= v <= stop]
        elif isinstance(selector, list):
            coord_set = set(selector)
            band_indices = [i for i, v in enumerate(coords) if v in coord_set]
        else:
            band_indices = [i for i, v in enumerate(coords) if v == selector]

        if not band_indices:
            raise ValueError(
                f"No bands match {dim_name}={selector}. " f"Available values: {coords}"
            )

        selected_coords = [coords[i] for i in band_indices]

        # Read only the selected bands instead of loading the full array.
        # Each band index maps to a 1-based GDAL band in the classic
        # dataset view created by get_variable().
        #
        # Trade-off: band-by-band reads avoid loading the entire variable
        # into memory, which matters for large variables with few selected
        # bands. However, when *most* bands are selected the per-band
        # GDAL overhead may be slower than a single full read followed by
        # NumPy slicing. In practice the difference is small because GDAL
        # MEM driver reads are cheap; revisit if profiling shows a
        # bottleneck for large on-disk NetCDFs.
        band_arrays = [self.read_array(band=i) for i in band_indices]
        if len(band_arrays) == 1:
            selected = band_arrays[0]
        else:
            selected = np.stack(band_arrays, axis=0)

        ndv = self.no_data_value
        ndv_scalar = ndv[0] if isinstance(ndv, list) and ndv else ndv
        ds_result = Dataset.create_from_array(
            selected,
            geo=self.geotransform,
            epsg=self.epsg,
            no_data_value=ndv_scalar,
        )
        result = self._preserve_netcdf_metadata(ds_result)
        result._band_dim_values = selected_coords

        return result

    @classmethod
    def read_file(  # type: ignore[override]
        cls,
        path: str | Path,
        read_only: bool = True,
        open_as_multi_dimensional: bool = True,
    ) -> NetCDF:
        """Open a NetCDF file from disk.

        Args:
            path: Path to the `.nc` file.
            read_only: If True, open in read-only mode. Set to False for
                write access. Defaults to True.
            open_as_multi_dimensional: If True, open with
                `gdal.OF_MULTIDIM_RASTER` to access the full group /
                dimension / variable hierarchy. If False, open in classic
                raster mode where each variable is a subdataset.
                Defaults to True.

        Returns:
            NetCDF: The opened dataset.
        """
        src = _io.read_file(path, read_only, open_as_multi_dimensional)
        if read_only:
            read_only = "read_only"
        else:
            read_only = "write"
        return cls(
            src, access=read_only, open_as_multi_dimensional=open_as_multi_dimensional
        )

    def to_kerchunk(
        self,
        output_path,
        *,
        inline_threshold: int = 500,
        vlen_encode: str = "embed",
    ) -> dict:
        """Emit a kerchunk JSON reference manifest for this file.

        Thin forwarder to :func:`pyramids.netcdf._kerchunk.to_kerchunk`
        using `self._file_name` as the source path. Requires the
        `[netcdf-lazy]` optional extra.

        Args:
            output_path: Path where the manifest JSON is written.
            inline_threshold: Chunks smaller than this many bytes are
                embedded directly. Default 500.
            vlen_encode: VLEN string handling mode. Default `"embed"`.

        Returns:
            dict: The manifest dict that was written.
        """
        return to_kerchunk(
            self._file_name,
            output_path,
            inline_threshold=inline_threshold,
            vlen_encode=vlen_encode,
        )

    @classmethod
    def combine_kerchunk(
        cls,
        paths,
        output_path,
        *,
        concat_dims=("time",),
        identical_dims=("lat", "lon"),
        inline_threshold: int = 500,
    ) -> dict:
        """Emit a combined kerchunk manifest spanning many NetCDFs.

        Thin forwarder to
        :func:`pyramids.netcdf._kerchunk.combine_kerchunk`. Requires
        the `[netcdf-lazy]` optional extra.

        Args:
            paths: Sequence of NetCDF paths to combine.
            output_path: Path where the combined manifest is written.
            concat_dims: Dimension name(s) along which to concatenate.
                Default `("time",)`.
            identical_dims: Dimensions expected to match across all
                files. Default `("lat", "lon")`.
            inline_threshold: Chunks smaller than this inline bytes are
                embedded. Default 500.

        Returns:
            dict: The combined manifest.
        """
        return combine_kerchunk(
            paths,
            output_path,
            concat_dims=concat_dims,
            identical_dims=identical_dims,
            inline_threshold=inline_threshold,
        )

    @classmethod
    def open_mfdataset(
        cls,
        paths,
        variable: str,
        *,
        chunks=None,
        parallel: bool = False,
        preprocess=None,
    ):
        """Open many NetCDFs and stack `variable` into one lazy dask array.

        Thin forwarder to
        :func:`pyramids.netcdf._mfdataset.open_mfdataset`; see that
        function for the full argument contract. Requires the
        `[lazy]` optional extra.

        Args:
            paths: Glob string, explicit path, or sequence of paths.
            variable: Name of the variable to extract from each file.
            chunks: Chunk spec forwarded to
                :meth:`NetCDF.read_array`.
            parallel: Fan out per-file opens through `dask.delayed`.
            preprocess: Optional callable applied to each
                :class:`NetCDF` before extraction.

        Returns:
            dask.array.Array: Stack of shape `(n_files, *var_shape)`.
        """
        return open_mfdataset(
            paths,
            variable,
            chunks=chunks,
            parallel=parallel,
            preprocess=preprocess,
        )

    @property
    def meta_data(self) -> NetCDFMetadata:
        """Structured metadata for this NetCDF.

        Uses the GDAL Multidimensional API (groups, arrays, dimensions) when
        the file was opened with `open_as_multi_dimensional=True`. Falls
        back to the classic `NETCDF_DIM_*` parser (`dimensions.py`) when
        opened in classic mode (no root group available).

        Cached on first access. Invalidated by add_variable/remove_variable.

        Returns:
            NetCDFMetadata
        """
        if self._cached_meta_data is None:
            open_options = {
                "Open Mode": "SHARED" if self.is_subset else "MULTIDIM_RASTER"
            }
            self._cached_meta_data = get_metadata(self._raster, open_options)
        return self._cached_meta_data

    @meta_data.setter
    def meta_data(self, value: dict[str, str] | NetCDFMetadata) -> None:
        """Set metadata on this NetCDF dataset."""
        if isinstance(value, dict):
            for key, val in value.items():
                self._raster.SetMetadataItem(key, val)
        else:
            self._cached_meta_data = value

    def get_all_metadata(self, open_options: dict | None = None) -> NetCDFMetadata:
        """Get full MDIM metadata (uncached).

        Unlike `meta_data` (which is cached), this always re-traverses
        the GDAL multidimensional structure.

        Args:
            open_options: Driver-specific open options forwarded to
                `get_metadata()`. Defaults to None.

        Returns:
            NetCDFMetadata
        """
        result = get_metadata(self._raster, open_options)
        return result

    def get_time_variable(
        self, var_name: str = "time", time_format: str = "%Y-%m-%d"
    ) -> list[str] | None:
        """Parse the time coordinate variable into formatted date strings.

        Reads the `units` attribute (e.g., `"days since 1979-01-01"`)
        from the dimension metadata and converts raw numeric values to
        human-readable date strings.

        Args:
            var_name: Name of the time dimension / variable.
                Defaults to `"time"`.
            time_format: strftime format for the output strings.
                Defaults to `"%Y-%m-%d"`.

        Returns:
            list[str] or None: Formatted time strings, or None if the
            time dimension is not found or lacks a `units` attribute.
        """
        time_stamp = None
        time_dim = self.meta_data.get_dimension(var_name)
        if time_dim is not None:
            units = time_dim.attrs.get("units")
            if units is not None:
                calendar = time_dim.attrs.get("calendar", "standard")
                time_vals = self._read_variable(var_name)
                if time_vals is not None:
                    func = create_time_conversion_func(
                        units, time_format, calendar=calendar
                    )
                    time_stamp = list(map(func, time_vals.reshape(-1)))
        return time_stamp

    def _get_dimension_names(self) -> list[str] | None:
        rg = self._raster.GetRootGroup()
        if rg is not None:
            dims = rg.GetDimensions()
            dims_names: list[str] | None = [dim.GetName() for dim in dims]
        else:
            dims_names = None
        return dims_names

    @property
    def dimension_names(self) -> list[str] | None:
        """Names of all dimensions in the root group (e.g., `["x", "y", "time"]`).

        Returns:
            list[str] or None: Dimension names, or None if no root group
            is available (classic mode).
        """
        return self._get_dimension_names()

    def _get_dimension(self, name: str) -> gdal.Dimension:
        dim_names = self.dimension_names
        if dim_names is not None and name in dim_names:
            rg = self._raster.GetRootGroup()
            dims = rg.GetDimensions()
            dim = dims[dim_names.index(name)]
        else:
            dim = None
        return dim

    def _needs_y_flip(self, rg, md_arr) -> bool:
        """Check if an MDArray's Y dimension goes south-to-north.

        Uses AsClassicDataset to check the geotransform Y pixel size.
        Returns True if the data needs flipping (positive Y pixel size).
        Returns False for 1-D arrays or when orientation is already correct.

        Args:
            rg: The root group (kept alive to prevent SWIG GC).
            md_arr: The MDArray to check.
        """
        result = False
        dims = md_arr.GetDimensions()
        if len(dims) >= 2:
            try:
                src = md_arr.AsClassicDataset(len(dims) - 1, len(dims) - 2, rg)
                result = src.GetGeoTransform()[5] > 0
            except Exception:
                pass
        return result

    def _read_variable(
        self,
        var: str,
        window: list[tuple[int, int]] | None = None,
    ) -> np.ndarray | None:
        """Read a variable's data as a numpy array, optionally windowed.

        Uses the MDIM root group when available (avoids opening a new GDAL
        handle). Falls back to the classic `NETCDF:file:var` path.

        For arrays with 2+ dimensions, the Y axis is flipped if the data
        is stored south-to-north (matching the flip in `get_variable`).

        Args:
            var: Variable name in the dataset.
            window: Per-dimension window as a list of `(start, count)`
                tuples, one per dimension of the target variable. For
                example, `[(0, 1), (100, 256), (200, 256)]` reads
                time[0:1], y[100:356], x[200:456]. When `None` the
                full variable is read. Only supported in MDIM mode;
                ignored in classic mode.

        Returns:
            np.ndarray or None: The variable data, or None if the
                variable is not found.
        """
        result = None
        rg = self._raster.GetRootGroup()
        if rg is not None:
            try:
                md_arr = rg.OpenMDArray(var)
                if md_arr is not None:
                    if window is not None:
                        starts = [w[0] for w in window]
                        counts = [w[1] for w in window]
                        result = md_arr.ReadAsArray(
                            array_start_idx=starts,
                            count=counts,
                        )
                    else:
                        result = md_arr.ReadAsArray()
                    # Flip Y axis if south-to-north (same as get_variable)
                    if result is not None and result.ndim >= 2:
                        if window is None and self._needs_y_flip(rg, md_arr):
                            y_axis = result.ndim - 2
                            result = np.flip(result, axis=y_axis)
            except Exception:
                pass  # nosec B110
            # Fall back to dimension indexing variable
            if result is None:
                dim = self._get_dimension(var)
                if dim is not None:
                    iv = dim.GetIndexingVariable()
                    if iv is not None:
                        if window is not None and len(window) == 1:
                            starts = [window[0][0]]
                            counts = [window[0][1]]
                            result = iv.ReadAsArray(
                                array_start_idx=starts,
                                count=counts,
                            )
                        else:
                            result = iv.ReadAsArray()
        else:
            # Classic mode: open via subdataset string
            try:
                ds = gdal.Open(f"NETCDF:{self.file_name}:{var}")
                if ds is not None:
                    result = ds.ReadAsArray()
                ds = None
            except (RuntimeError, AttributeError):
                pass
        return result

    @property
    def group_names(self) -> list[str]:
        """Names of sub-groups in the root group.

        Returns:
            list[str]: Sub-group names (e.g. `["forecast", "analysis"]`).
            Empty list if no sub-groups exist or the dataset is in
            classic mode.
        """
        rg = self._raster.GetRootGroup()
        result = []
        if rg is not None:
            try:
                names = rg.GetGroupNames()
                if names:
                    result = list(names)
            except Exception:
                pass
        return result

    def get_group(self, group_name: str) -> NetCDF:
        """Open a sub-group as a NetCDF container.

        The returned object wraps the sub-group's GDAL dataset and
        exposes the sub-group's variables and dimensions via the
        same API as the root container.

        Args:
            group_name: Name of the sub-group. Supports nested paths
                separated by `/` (e.g. `"forecast/surface"`).

        Returns:
            NetCDF: A container backed by the sub-group.

        Raises:
            ValueError: If the group doesn't exist or the dataset
                has no root group.
        """
        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError("get_group requires a multidimensional container.")

        # Navigate nested paths: "forecast/surface" → open each level
        group = rg
        parts = group_name.split("/")
        for part in parts:
            try:
                group = group.OpenGroup(part)
            except Exception:
                group = None
            if group is None:
                raise ValueError(
                    f"Group '{group_name}' not found. "
                    f"Available groups: {self.group_names}"
                )

        # Create a multidimensional dataset from the sub-group.
        # GDAL doesn't have a direct "group → dataset" conversion,
        # so we build a MEM MDIM dataset and copy the group's
        # arrays and dimensions into it.
        dst = gdal.GetDriverByName("MEM").CreateMultiDimensional("group")
        dst_rg = dst.GetRootGroup()

        # Copy dimensions from the sub-group
        dim_map = {}
        for gdal_dim in group.GetDimensions() or []:
            dim_name = gdal_dim.GetName()
            new_dim = dst_rg.CreateDimension(
                dim_name, gdal_dim.GetType(), None, gdal_dim.GetSize()
            )
            iv = gdal_dim.GetIndexingVariable()
            if iv is not None:
                coord_arr = dst_rg.CreateMDArray(
                    dim_name,
                    [new_dim],
                    gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(iv.ReadAsArray())),
                )
                coord_arr.Write(iv.ReadAsArray())
                new_dim.SetIndexingVariable(coord_arr)
            dim_map[dim_name] = new_dim

        # Copy arrays from the sub-group
        for arr_name in group.GetMDArrayNames() or []:
            md_arr = group.OpenMDArray(arr_name)
            if md_arr is None:
                continue
            arr_dims = md_arr.GetDimensions()
            # Map source dims to destination dims (by name)
            new_dims = []
            for d in arr_dims:
                d_name = d.GetName()
                if d_name in dim_map:
                    new_dims.append(dim_map[d_name])
                else:
                    # Dimension from parent group — create locally
                    new_d = dst_rg.CreateDimension(
                        d_name, d.GetType(), None, d.GetSize()
                    )
                    dim_map[d_name] = new_d
                    new_dims.append(new_d)
            arr_data = md_arr.ReadAsArray()
            arr_dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr_data))
            new_arr = dst_rg.CreateMDArray(arr_name, new_dims, arr_dtype)
            new_arr.Write(arr_data)
            ndv = md_arr.GetNoDataValue()
            if ndv is not None:
                new_arr.SetNoDataValueDouble(ndv)
            srs = md_arr.GetSpatialRef()
            if srs is not None:
                new_arr.SetSpatialRef(srs)

        result = NetCDF(dst)
        return result

    def get_variable_names(self) -> list[str]:
        """Return names of data variables, excluding dimension coordinates.

        Uses CF classification when metadata is cached (fast path).
        Otherwise queries `GetMDArrayNames()` and filters out dimension
        arrays and 0-dimensional scalar variables (grid_mapping etc.).
        In classic mode, parses subdataset metadata.

        Returns:
            list[str]: Variable names (e.g., `["temperature", "precipitation"]`).
        """
        if self._cached_meta_data is not None and self._cached_meta_data.cf is not None:
            variable_names = list(self._cached_meta_data.cf.data_variable_names)
        else:
            rg = self._raster.GetRootGroup()
            if rg is not None:
                all_names = rg.GetMDArrayNames()
                dim_names = {dim.GetName() for dim in rg.GetDimensions()}
                filtered = []
                for var in all_names:
                    if var in dim_names:
                        continue
                    md_arr = rg.OpenMDArray(var)
                    if md_arr is not None and len(md_arr.GetDimensions()) == 0:
                        continue
                    filtered.append(var)
                variable_names = filtered
            else:
                variable_names = [
                    var[1].split(" ")[1] for var in self._raster.GetSubDatasets()
                ]

        return variable_names

    def _read_md_array(self, variable_name: str):
        """Convert an MDArray to a classic GDAL dataset via AsClassicDataset.

        The last two dimensions become X (columns) and Y (rows); all
        remaining dimensions are flattened into bands.

        If the Y dimension is stored south-to-north (positive Y pixel
        size), it is reversed via `MDArray.GetView()` **before** the
        conversion. This is a lazy, zero-copy operation — GDAL handles
        the reversed indexing internally without reading the whole array.

        Returns a tuple `(classic_dataset, md_array, root_group)` so
        callers can keep the GDAL objects alive. `AsClassicDataset`
        returns a **view** whose C++ backing depends on the MDArray and
        root group; if the Python SWIG wrappers for those are garbage-
        collected the view becomes a dangling pointer (segfault on
        Windows).
        """
        rg = self._raster.GetRootGroup()
        md_arr = rg.OpenMDArray(variable_name)
        dtype = md_arr.GetDataType()
        dims = md_arr.GetDimensions()

        if len(dims) == 1:
            if dtype.GetClass() == gdal.GEDTC_STRING:
                return md_arr, md_arr, rg
            src = md_arr.AsClassicDataset(0, 1, rg)
            return src, md_arr, rg

        iXDim = len(dims) - 1
        iYDim = len(dims) - 2

        # First pass: check if Y orientation needs flipping.
        src = md_arr.AsClassicDataset(iXDim, iYDim, rg)

        if src.GetGeoTransform()[5] > 0:
            # Positive Y pixel size = south-to-north (NetCDF convention).
            # Use GetView to reverse the Y dimension — this is lazy and
            # zero-copy; GDAL handles reversed indexing internally.
            slices = ",".join("::-1" if i == iYDim else ":" for i in range(len(dims)))
            md_arr = md_arr.GetView(f"[{slices}]")
            src = md_arr.AsClassicDataset(iXDim, iYDim, rg)

        return src, md_arr, rg

    def get_variable(self, variable_name: str) -> NetCDF:
        """Extract a single variable as a classic-raster NetCDF object.

        The returned object carries origin metadata so that modified data
        can be written back via `set_variable()`.

        Supports group-qualified names: `"forecast/temperature"` first
        navigates to the `forecast` sub-group, then extracts
        `temperature` from it.

        Args:
            variable_name: Name of the variable to extract. Use `/`
                to separate group path from variable name.

        Returns:
            NetCDF: A subset backed by a classic dataset where
                non-spatial dimensions are mapped to bands.

        Raises:
            ValueError: If `variable_name` is not present in the dataset.
        """
        # Handle group-qualified names: "forecast/temperature"
        if "/" in variable_name:
            parts = variable_name.rsplit("/", 1)
            group_nc = self.get_group(parts[0])
            cube = group_nc.get_variable(parts[1])
            return cube  # single return below handles non-group path

        if variable_name not in self.variable_names:
            raise ValueError(
                f"{variable_name} is not a valid variable name in {self.variable_names}"
            )

        prefix = self.driver_type.upper()
        rg = self._raster.GetRootGroup()
        md_arr_ref = None
        rg_ref = None

        if prefix == "MEMORY" or rg is not None:
            src, md_arr_ref, rg_ref = self._read_md_array(variable_name)
            if isinstance(src, gdal.Dataset):
                cube = NetCDF(src)
                cube._is_md_array = True
                # _read_md_array uses GetView to flip the data lazily,
                # and GDAL usually corrects the geotransform. But when
                # the Y dimension has no indexing variable (e.g. WRF
                # "south_north"), the geotransform may still be wrong.
                # Fix it on the wrapper object (no data copy).
                gt = cube._geotransform
                if gt[5] > 0:
                    cube._geotransform = (
                        gt[0],
                        gt[1],
                        gt[2],
                        gt[3] + gt[5] * cube._rows,
                        gt[4],
                        -gt[5],
                    )
                    cube._cell_size = abs(gt[1])
            else:
                cube = src
            # Keep GDAL SWIG references alive — AsClassicDataset returns a
            # view whose C++ backing is owned by the MDArray/root group.
            # Without these the view becomes a dangling pointer on Windows.
            cube._gdal_md_arr_ref = md_arr_ref
            cube._gdal_rg_ref = rg_ref
        else:
            src = gdal.Open(f"{prefix}:{self.file_name}:{variable_name}")
            if src is None:
                raise ValueError(
                    f"Could not open variable '{variable_name}' via "
                    f"'{prefix}:{self.file_name}:{variable_name}'"
                )
            cube = NetCDF(src)
            cube._is_md_array = False

        cube._is_subset = True

        # --- RT-4: Track variable origin for round-trip ---
        cube._parent_nc = self
        cube._source_var_name = variable_name

        md_arr = md_arr_ref if rg is not None else None
        if rg is not None:
            if md_arr is not None:
                dims = md_arr.GetDimensions()
                cube._md_array_dims = [d.GetName() for d in dims]

                # Identify which dimension became bands (all except X/Y)
                if len(dims) > 2:
                    spatial_indices = {len(dims) - 1, len(dims) - 2}
                    band_dims = [
                        d for i, d in enumerate(dims) if i not in spatial_indices
                    ]
                    if len(band_dims) == 1:
                        cube._band_dim_name = band_dims[0].GetName()
                        iv = band_dims[0].GetIndexingVariable()
                        try:
                            cube._band_dim_values = (
                                iv.ReadAsArray().tolist() if iv is not None else None
                            )
                        except RuntimeError:
                            # String-typed indexing variables (e.g. WRF
                            # "Times") can't be read via ReadAsArray in
                            # GDAL SWIG bindings — fall back to indices.
                            cube._band_dim_values = list(range(band_dims[0].GetSize()))
                    else:
                        cube._band_dim_name = None
                        cube._band_dim_values = None
                else:
                    cube._band_dim_name = None
                    cube._band_dim_values = None

                # Copy variable attributes
                cube._variable_attrs = {}
                try:
                    for attr in md_arr.GetAttributes():
                        cube._variable_attrs[attr.GetName()] = attr.Read()
                except Exception:
                    pass  # nosec B110

                # Scale/offset for CF packed data
                try:
                    cube._scale = md_arr.GetScale()
                    cube._offset = md_arr.GetOffset()
                except Exception:
                    cube._scale = None
                    cube._offset = None
            else:
                cube._md_array_dims = []
                cube._band_dim_name = None
                cube._band_dim_values = None
                cube._variable_attrs = {}
                cube._scale = None
                cube._offset = None
        else:
            cube._md_array_dims = []
            cube._band_dim_name = None
            cube._band_dim_values = None
            cube._variable_attrs = {}
            cube._scale = None
            cube._offset = None

        return cube

    def _replace_raster(self, new_raster: gdal.Dataset):
        """Replace the internal GDAL dataset, closing the old one if different.

        Re-derives all base-class state (geotransform, CRS, band info, etc.)
        without resetting NetCDF-specific flags (_is_md_array, _is_subset).
        """
        old = self._raster
        if old is not None and old is not new_raster:
            old.FlushCache()
        # RasterBase state
        self._raster = new_raster
        self._geotransform = new_raster.GetGeoTransform()
        self._cell_size = self._geotransform[1]
        self._file_name = new_raster.GetDescription()
        self._epsg = self._get_epsg()
        self._rows = new_raster.RasterYSize
        self._columns = new_raster.RasterXSize
        self._band_count = new_raster.RasterCount
        self._block_size = [
            new_raster.GetRasterBand(i).GetBlockSize()
            for i in range(1, self._band_count + 1)
        ]
        # Dataset state
        self._no_data_value = [
            new_raster.GetRasterBand(i).GetNoDataValue()
            for i in range(1, self._band_count + 1)
        ]
        self._band_names = self._get_band_names()
        self._band_units = [
            new_raster.GetRasterBand(i).GetUnitType()
            for i in range(1, self._band_count + 1)
        ]
        # Invalidate caches
        self._cached_variables = None
        self._cached_meta_data = None

    def _invalidate_caches(self):
        """Invalidate cached variables and metadata."""
        self._cached_variables = None
        self._cached_meta_data = None

    @property
    def is_subset(self) -> bool:
        """Whether this object represents a single-variable subset.

        Returns:
            bool: True if the dataset is a variable subset extracted
                via `get_variable()`.
        """
        return self._is_subset

    @property
    def is_md_array(self):
        """Whether this dataset was opened in multidimensional mode.

        Returns:
            bool: True if the dataset was opened via
                `gdal.OF_MULTIDIM_RASTER` and supports groups,
                MDArrays, and dimensions.
        """
        return self._is_md_array

    def to_file(  # type: ignore[override]
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> None:
        """Save the dataset to disk.

        For `.nc` / `.nc4` files the full multidimensional structure
        (groups, dimensions, variables, attributes) is preserved via
        `CreateCopy` with the netCDF driver. For other extensions
        (e.g. `.tif`), the parent `Dataset.to_file` is used — but only
        on variable subsets, not on root MDIM containers.

        Args:
            path: Destination file path. The extension determines the
                output driver (`.nc` -> netCDF, `.tif` -> GeoTIFF, etc.).
            **kwargs: Forwarded to `Dataset.to_file` for non-NetCDF
                extensions (e.g. `tile_length`, `creation_options`).

        Raises:
            RuntimeError: If the netCDF `CreateCopy` call fails.
            ValueError: If a root MDIM container is saved to a non-NC
                extension (use `.nc` or extract a variable first).
        """
        path = Path(path)
        extension = path.suffix[1:].lower()
        if extension in ("nc", "nc4"):
            dst = gdal.GetDriverByName("netCDF").CreateCopy(str(path), self._raster, 0)
            if dst is None:
                raise RuntimeError(f"Failed to save NetCDF to {path}")
            dst.FlushCache()
            dst = None
        else:
            if self._is_md_array and not self._is_subset:
                raise ValueError(
                    "Cannot save a multidimensional NetCDF container as "
                    f"'{extension}'. Use .nc extension or extract a "
                    "variable first with .get_variable()."
                )
            super().to_file(path, **kwargs)

    def copy(self, path: str | Path | None = None) -> NetCDF:
        """Create a deep copy of this NetCDF dataset.

        Args:
            path: Destination file path. If None, the copy is created
                in memory using the MEM driver. Defaults to None.

        Returns:
            NetCDF: A new NetCDF object with copied data.

        Raises:
            RuntimeError: If `CreateCopy` fails.
        """
        if path is None:
            path = ""
            driver = "MEM"
        else:
            driver = "netCDF"

        src = gdal.GetDriverByName(driver).CreateCopy(str(path), self._raster)
        if src is None:
            raise RuntimeError(f"Failed to copy NetCDF dataset to '{path}'")
        return NetCDF(src, access="write")

    @staticmethod
    def _create_dimension(
        group: gdal.Group,
        dim_name: str,
        dtype,
        values: np.ndarray,
        dim_type=None,
        set_indexing: bool = True,
        is_geographic: bool = True,
    ) -> gdal.Dimension:
        """Create a dimension with its coordinate array and CF attributes.

        Args:
            group: GDAL root group.
            dim_name: Dimension name.
            dtype: GDAL ExtendedDataType.
            values: Coordinate values.
            dim_type: GDAL dimension type constant.
            set_indexing: If True, call SetIndexingVariable (works
                on MEM driver). If False, skip it (required for
                netCDF driver which doesn't support it).
            is_geographic: If True, coordinate units are degrees.
                If False, units are metres. Defaults to True.

        Returns:
            gdal.Dimension
        """
        dim = group.CreateDimension(dim_name, dim_type, None, values.shape[0])
        coord_arr = group.CreateMDArray(dim_name, [dim], dtype)
        coord_arr.Write(values)
        if set_indexing:
            dim.SetIndexingVariable(coord_arr)
        cf_attrs = build_coordinate_attrs(dim_name, is_geographic)
        if cf_attrs:
            write_attributes_to_md_array(coord_arr, cf_attrs)
        return dim

    @staticmethod
    def create_main_dimension(
        group: gdal.Group, dim_name: str, dtype: int, values: np.ndarray
    ) -> gdal.Dimension:
        """Create a NetCDF dimension with an indexing variable.

        The dimension type is inferred from `dim_name`:
        `y`/`lat`/`latitude` -> horizontal Y,
        `x`/`lon`/`longitude` -> horizontal X,
        `bands`/`time` -> temporal.

        The dimension is registered in the group together with a
        matching MDArray that stores the coordinate values.

        Args:
            group: Root group (or sub-group) of the multidimensional
                dataset.
            dim_name: Name of the dimension to create.
            dtype: GDAL `ExtendedDataType` for the indexing variable.
            values: Coordinate values for the dimension.

        Returns:
            gdal.Dimension: The newly created dimension.
        """
        if dim_name in ["y", "lat", "latitude"]:
            dim_type = gdal.DIM_TYPE_HORIZONTAL_Y
        elif dim_name in ["x", "lon", "longitude"]:
            dim_type = gdal.DIM_TYPE_HORIZONTAL_X
        elif dim_name in ["bands", "time"]:
            dim_type = gdal.DIM_TYPE_TEMPORAL
        else:
            dim_type = None
        dim = group.CreateDimension(dim_name, dim_type, None, values.shape[0])
        x_values = group.CreateMDArray(dim_name, [dim], dtype)
        x_values.Write(values)
        dim.SetIndexingVariable(x_values)
        return dim

    @classmethod
    def create_from_array(  # type: ignore[override]
        cls,
        arr: np.ndarray,
        geo: tuple[float, float, float, float, float, float] | None = None,
        epsg: str | int = 4326,
        no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
        path: str | Path | None = None,
        variable_name: str | None = None,
        extra_dim_name: str = "time",
        extra_dim_values: list | None = None,
        top_left_corner: tuple[float, float] | None = None,
        cell_size: int | float | None = None,
        chunk_sizes: tuple | list | None = None,
        compression: str | None = None,
        compression_level: int | None = None,
        title: str | None = None,
        institution: str | None = None,
        source: str | None = None,
        history: str | None = None,
    ) -> NetCDF:
        """Create a NetCDF dataset from a NumPy array and geotransform.

        For 3-D arrays the first axis is treated as a non-spatial
        dimension (time, level, depth, etc.) whose name and coordinate
        values are controlled by `extra_dim_name` and
        `extra_dim_values`.

        The driver is inferred from `path`: if `path` is `None`
        the dataset is created in memory (MEM driver); if a path is
        provided the netCDF driver writes to disk.

        Args:
            arr: 2-D `(rows, cols)` or 3-D
                `(extra_dim, rows, cols)` NumPy array.
            geo: Geotransform tuple `(x_min, pixel_size, rotation,
                y_max, rotation, pixel_size)`.
            epsg: EPSG code for the spatial reference.
                Defaults to 4326.
            no_data_value: Sentinel value for cells outside the
                domain. Defaults to DEFAULT_NO_DATA_VALUE.
            path: Output file path. If `None`, the dataset is
                created in memory. Defaults to None.
            variable_name: Name of the data variable in the NetCDF
                file. Defaults to `"data"`.
            extra_dim_name: Name of the non-spatial dimension for 3-D
                arrays (e.g. `"time"`, `"level"`, `"depth"`).
                Ignored for 2-D arrays. Defaults to `"time"`.
            extra_dim_values: Coordinate values for the non-spatial
                dimension. Must have length `arr.shape[0]` for 3-D
                arrays. Defaults to `[0, 1, 2,..., N-1]`.
            top_left_corner: `(x, y)` of the top-left corner. Used
                with `cell_size` to build `geo` when `geo` is
                not provided. Defaults to None.
            cell_size: Pixel size. Used with `top_left_corner` to
                build `geo`. Defaults to None.
            chunk_sizes: Chunk sizes for the data variable as a tuple
                matching the array dimensions (e.g. `(1, 256, 256)`
                for 3-D). Only effective when writing to disk.
                Defaults to None (GDAL default chunking).
            compression: Compression algorithm name (`"DEFLATE"`,
                `"ZSTD"`, etc.). Only effective when writing to
                disk. Defaults to None (no compression).
            compression_level: Compression level (e.g. 1-9 for
                DEFLATE). Defaults to None (GDAL default).
            title: CF global attribute `title`. Short
                description of the dataset. Defaults to None.
            institution: CF global attribute `institution`.
                Where the data was produced. Defaults to None.
            source: CF global attribute `source`. How the
                data was produced. Defaults to None.
            history: CF global attribute `history`. Audit
                trail of processing steps. Defaults to None.

        Returns:
            NetCDF: The newly created NetCDF dataset.
        """
        if geo is None and top_left_corner is not None and cell_size is not None:
            geo = (
                top_left_corner[0],
                cell_size,
                0,
                top_left_corner[1],
                0,
                -cell_size,
            )
        if geo is None:
            raise ValueError(
                "Either 'geo' or both 'top_left_corner' and "
                "'cell_size' must be provided."
            )

        if arr.ndim == 2:
            rows = int(arr.shape[0])
            cols = int(arr.shape[1])
        else:
            rows = int(arr.shape[1])
            cols = int(arr.shape[2])

        if extra_dim_values is None and arr.ndim == 3:
            extra_dim_values = list(range(arr.shape[0]))

        if arr.ndim == 3:
            DimMetaData(
                name=extra_dim_name,
                size=arr.shape[0],
                values=extra_dim_values,
            )

        if variable_name is None:
            variable_name = "data"

        dst_ds = cls._create_netcdf_from_array(
            arr,
            variable_name,
            cols,
            rows,
            extra_dim_name,
            extra_dim_values,
            geo,
            epsg,
            no_data_value,
            path=path,
            chunk_sizes=chunk_sizes,
            compression=compression,
            compression_level=compression_level,
            title=title,
            institution=institution,
            source=source,
            history=history,
        )
        result = cls(dst_ds)

        return result

    @staticmethod
    def _create_netcdf_from_array(
        arr: np.ndarray,
        variable_name: str,
        cols: int,
        rows: int,
        extra_dim_name: str = "time",
        extra_dim_values: list | None = None,
        geo: tuple[float, float, float, float, float, float] | None = None,
        epsg: str | int | None = None,
        no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
        path: str | Path | None = None,
        chunk_sizes: tuple | list | None = None,
        compression: str | None = None,
        compression_level: int | None = None,
        title: str | None = None,
        institution: str | None = None,
        source: str | None = None,
        history: str | None = None,
    ) -> gdal.Dataset:
        """Build a multidimensional GDAL dataset from an array.

        The driver is inferred from `path`: `None` -> MEM (in-memory),
        otherwise the netCDF driver writes to disk.

        Args:
            arr: 2-D `(rows, cols)` or 3-D
                `(extra_dim, rows, cols)` NumPy array.
            variable_name: Name of the data variable.
            cols: Number of columns.
            rows: Number of rows.
            extra_dim_name: Name of the non-spatial dimension
                (e.g. `"time"`, `"level"`). Defaults to `"time"`.
            extra_dim_values: Coordinate values for the non-spatial
                dimension. Defaults to None.
            geo: Geotransform tuple. Defaults to None.
            epsg: EPSG code. Defaults to None.
            no_data_value: No-data sentinel. Defaults to
                DEFAULT_NO_DATA_VALUE.
            path: Output file path. If None, created in memory.
                Defaults to None.
            chunk_sizes: Chunk sizes for the variable. Defaults to
                None.
            compression: Compression algorithm. Defaults to None.
            compression_level: Compression level. Defaults to None.
            title: CF global attribute `title`. Defaults to None.
            institution: CF global attribute `institution`.
                Defaults to None.
            source: CF global attribute `source`.
                Defaults to None.
            history: CF global attribute `history`.
                Defaults to None.

        Returns:
            gdal.Dataset: The created multidimensional GDAL dataset.
        """
        if variable_name is None:
            raise ValueError("Variable_name cannot be None")
        if geo is None:
            raise ValueError("geo cannot be None")

        dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        x_dim_values = NetCDF.get_x_lon_dimension_array(geo[0], geo[1], cols)
        y_dim_values = NetCDF.get_y_lat_dimension_array(geo[3], geo[1], rows)

        if path is not None:
            driver_type = "netCDF"
        else:
            driver_type = "MEM"
            path = "netcdf"

        src = gdal.GetDriverByName(driver_type).CreateMultiDimensional(str(path))
        rg = src.GetRootGroup()

        # Set CF global attributes on root group
        cf_global = {"Conventions": "CF-1.8"}
        if title is not None:
            cf_global["title"] = title
        if institution is not None:
            cf_global["institution"] = institution
        if source is not None:
            cf_global["source"] = source
        if history is not None:
            cf_global["history"] = history
        write_global_attributes(rg, cf_global)

        # Build creation options for chunking and compression
        create_options = []
        if chunk_sizes is not None:
            create_options.append(f"BLOCKSIZE={','.join(str(s) for s in chunk_sizes)}")
        if compression is not None:
            create_options.append(f"COMPRESS={compression}")
        if compression_level is not None:
            create_options.append(f"ZLEVEL={compression_level}")

        # netCDF driver doesn't support SetIndexingVariable — create
        # dimension arrays manually without linking them.
        use_set_indexing = driver_type == "MEM"

        # Determine if CRS is geographic (lon/lat) or projected (m)
        is_geographic = True
        if epsg is not None:
            srs_check = sr_from_epsg(int(epsg))
            is_geographic = srs_check.IsGeographic() == 1

        dim_x = NetCDF._create_dimension(
            rg,
            "x",
            dtype,
            np.array(x_dim_values),
            gdal.DIM_TYPE_HORIZONTAL_X,
            use_set_indexing,
            is_geographic=is_geographic,
        )
        dim_y = NetCDF._create_dimension(
            rg,
            "y",
            dtype,
            np.array(y_dim_values),
            gdal.DIM_TYPE_HORIZONTAL_Y,
            use_set_indexing,
            is_geographic=is_geographic,
        )

        if arr.ndim == 3:
            extra_dim = NetCDF._create_dimension(
                rg,
                extra_dim_name,
                dtype,
                np.array(extra_dim_values),
                gdal.DIM_TYPE_TEMPORAL,
                use_set_indexing,
            )
            md_arr = rg.CreateMDArray(
                variable_name,
                [extra_dim, dim_y, dim_x],
                dtype,
                create_options if create_options else [],
            )
        else:
            md_arr = rg.CreateMDArray(
                variable_name,
                [dim_y, dim_x],
                dtype,
                create_options if create_options else [],
            )

        # Set metadata BEFORE writing data — netCDF driver requires
        # nodata to be set before the first Write call.
        md_arr.SetNoDataValueDouble(no_data_value)
        if epsg is None:
            raise ValueError("epsg cannot be None")
        srse = sr_from_epsg(int(epsg))
        md_arr.SetSpatialRef(srse)
        md_arr.Write(arr)

        # Create CF grid_mapping variable (MEM driver only — the netCDF
        # driver creates its own via SetSpatialRef above). Use
        # "spatial_ref" as the variable name to avoid collision with
        # GDAL's automatic "crs" during CreateCopy to netCDF.
        if driver_type == "MEM":
            gm_name, gm_params = srs_to_grid_mapping(srse)
            gm_dtype = gdal.ExtendedDataType.Create(gdal.GDT_Int32)
            gm_var_name = "spatial_ref"
            crs_arr = rg.CreateMDArray(gm_var_name, [], gm_dtype)
            crs_arr.Write(np.array(0, dtype=np.int32))
            gm_params["grid_mapping_name"] = gm_name
            write_attributes_to_md_array(crs_arr, gm_params)
            write_attributes_to_md_array(md_arr, {"grid_mapping": gm_var_name})

        return src

    @staticmethod
    def _add_md_array_to_group(dst_group, var_name, src_mdarray):
        """Copy an MDArray from one group to another, preserving data and metadata."""
        src_dims = src_mdarray.GetDimensions()
        arr = src_mdarray.ReadAsArray()
        dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        new_md_array = dst_group.CreateMDArray(var_name, src_dims, dtype)
        new_md_array.Write(arr)
        ndv = src_mdarray.GetNoDataValue()
        if ndv is not None:
            try:
                new_md_array.SetNoDataValueDouble(ndv)
            except Exception:
                pass

        new_md_array.SetSpatialRef(src_mdarray.GetSpatialRef())

    @staticmethod
    def _get_or_create_dimension(
        rg: gdal.Group, dim_name: str, values: np.ndarray, dtype, dim_type=None
    ) -> gdal.Dimension:
        """Reuse an existing dimension or create a new one.

        If a dimension with `dim_name` already exists in the root group
        and has the same size as `values`, it is returned directly.
        On size mismatch, a new dimension with a `_{size}` suffix is
        created to avoid conflicts.

        Args:
            rg: The root group of the multidimensional dataset.
            dim_name: Name of the dimension (e.g., `"x"`, `"time"`).
            values: Coordinate values for this dimension.
            dtype: GDAL `ExtendedDataType` for the indexing variable.
            dim_type: GDAL dimension type constant (e.g.,
                `gdal.DIM_TYPE_HORIZONTAL_X`). Defaults to None.

        Returns:
            gdal.Dimension: The reused or newly created dimension.
        """
        for existing_dim in rg.GetDimensions() or []:
            if existing_dim.GetName() == dim_name:
                if existing_dim.GetSize() == len(values):
                    return existing_dim
                # Size mismatch — need a new dimension with a unique name
                dim_name = f"{dim_name}_{len(values)}"
                break

        return NetCDF.create_main_dimension(rg, dim_name, dtype, values)

    @property
    def global_attributes(self) -> dict[str, Any]:
        """Global attributes from the root group.

        Returns a live dict read from the GDAL root group each time.
        For MDIM mode, reads from the root group's attributes.
        For classic mode, reads from GDAL's `GetMetadata()`.

        Returns:
            dict[str, Any]: Key-value mapping of global attributes.
        """
        rg = self._raster.GetRootGroup()
        result = {}
        if rg is not None:
            try:
                for attr in rg.GetAttributes():
                    result[attr.GetName()] = attr.Read()
            except Exception:
                pass
        else:
            result = dict(self._raster.GetMetadata())
        return result

    def set_global_attribute(self, name: str, value: Any):
        """Set a global attribute on the root group.

        Creates or updates a single attribute on the root group.

        Args:
            name: Attribute name (e.g. `"history"`,
                `"Conventions"`).
            value: Attribute value. Supports str, int, float.

        Raises:
            ValueError: If the dataset has no root group
                (not opened in MDIM mode).
        """
        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError(
                "set_global_attribute requires a multidimensional "
                "container. Open the file with "
                "open_as_multi_dimensional=True."
            )
        # Delete existing attribute if present (GDAL raises on duplicate)
        try:
            rg.DeleteAttribute(name)
        except Exception:
            pass
        if isinstance(value, str):
            attr = rg.CreateAttribute(name, [], gdal.ExtendedDataType.CreateString())
        elif isinstance(value, float):
            attr = rg.CreateAttribute(
                name, [], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
            )
        elif isinstance(value, int):
            attr = rg.CreateAttribute(
                name, [], gdal.ExtendedDataType.Create(gdal.GDT_Int32)
            )
        else:
            attr = rg.CreateAttribute(name, [], gdal.ExtendedDataType.CreateString())
            value = str(value)
        attr.Write(value)
        self._invalidate_caches()

    def delete_global_attribute(self, name: str):
        """Delete a global attribute from the root group.

        If the attribute does not exist, the call is silently ignored.

        Args:
            name: Attribute name to delete.

        Raises:
            ValueError: If the dataset has no root group.
        """
        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError(
                "delete_global_attribute requires a multidimensional " "container."
            )
        try:
            rg.DeleteAttribute(name)
        except Exception:
            pass  # attribute may not exist — silently ignored
        self._invalidate_caches()

    def set_variable(
        self,
        variable_name: str,
        dataset: Dataset,
        band_dim_name: str | None = None,
        band_dim_values: list | None = None,
        attrs: dict | None = None,
    ):
        """Write a classic Dataset back as an MDArray variable in this container.

        This is the reverse of `get_variable()`. After performing GIS
        operations (crop, reproject, etc.) on a variable subset, use this
        method to store the result back into the NetCDF container.

        Args:
            variable_name: Name for the variable in this container. If a
                variable with this name already exists it is replaced.
            dataset: A classic raster dataset, typically the result of a
                GIS operation on a variable obtained via `get_variable()`.
            band_dim_name: Name of the dimension that maps to bands
                (e.g. `"time"`, `"bands"`). Auto-detected from the
                dataset's `_band_dim_name` attribute when available.
                Defaults to None.
            band_dim_values: Coordinate values for the band dimension.
                Auto-detected from `_band_dim_values` when available.
                Defaults to None.
            attrs: Variable attributes to set (e.g. `{"units": "K"}`).
                Auto-detected from `_variable_attrs` when available.
                Defaults to None.

        Raises:
            ValueError: If called on a dataset without a root group
                (not opened in multidimensional mode).
        """
        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError(
                "set_variable requires a multidimensional container. "
                "Open the file with open_as_multi_dimensional=True."
            )

        # Auto-detect from tracked origin metadata (RT-4)
        if band_dim_name is None and hasattr(dataset, "_band_dim_name"):
            band_dim_name = dataset._band_dim_name
        if band_dim_values is None and hasattr(dataset, "_band_dim_values"):
            band_dim_values = dataset._band_dim_values
        if attrs is None and hasattr(dataset, "_variable_attrs"):
            attrs = dataset._variable_attrs

        # Delete existing variable if present
        if variable_name in self.variable_names:
            rg.DeleteMDArray(variable_name)

        # Read data from the classic dataset
        arr = dataset.read_array()
        gt: tuple[float, float, float, float, float, float] = dataset.geotransform
        data_dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        # Coordinate dimensions must always be float64 to avoid truncation
        # when the data array is integer (e.g., classified rasters).
        coord_dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

        # Build spatial dimensions from the geotransform
        x_values = np.array(
            NetCDF.get_x_lon_dimension_array(gt[0], gt[1], dataset.columns)
        )
        y_values = np.array(
            NetCDF.get_y_lat_dimension_array(gt[3], abs(gt[5]), dataset.rows)
        )
        dim_x = self._get_or_create_dimension(
            rg, "x", x_values, coord_dtype, gdal.DIM_TYPE_HORIZONTAL_X
        )
        dim_y = self._get_or_create_dimension(
            rg, "y", y_values, coord_dtype, gdal.DIM_TYPE_HORIZONTAL_Y
        )

        # Build band dimension if the data is 3D
        if arr.ndim == 3:
            if band_dim_name is None:
                band_dim_name = "bands"
            if band_dim_values is None:
                band_dim_values = list(range(arr.shape[0]))
            dim_band = self._get_or_create_dimension(
                rg,
                band_dim_name,
                np.array(band_dim_values, dtype=np.float64),
                coord_dtype,
                gdal.DIM_TYPE_TEMPORAL,
            )
            md_arr = rg.CreateMDArray(
                variable_name, [dim_band, dim_y, dim_x], data_dtype
            )
        else:
            md_arr = rg.CreateMDArray(variable_name, [dim_y, dim_x], data_dtype)

        # Write array data
        md_arr.Write(arr)

        # Set spatial reference (RT-7: attribute copying)
        if dataset.epsg:
            srs = sr_from_epsg(dataset.epsg)
            md_arr.SetSpatialRef(srs)

        # Set no-data value
        if dataset.no_data_value and dataset.no_data_value[0] is not None:
            try:
                md_arr.SetNoDataValueDouble(float(dataset.no_data_value[0]))
            except Exception:
                pass  # nosec B110

        # Set variable attributes (RT-7)
        if attrs:
            write_attributes_to_md_array(md_arr, attrs)

        self._invalidate_caches()

    def crop_variable(
        self, variable_name: str, mask: Any, touch: bool = True
    ) -> NetCDF:
        """Crop a single variable and store the result back.

        Convenience method that combines `get_variable` → `crop`
        → `set_variable` in one call.

        Args:
            variable_name: Name of the variable to crop.
            mask: GeoDataFrame with polygon geometry, or a Dataset
                to use as a spatial mask.
            touch: If True, include cells touching the mask boundary.
                Defaults to True.

        Returns:
            NetCDF: This container (modified in-place).
        """
        var = self.get_variable(variable_name)
        cropped = var.crop(mask, touch=touch)
        self.set_variable(variable_name, cropped)
        return self

    def reproject_variable(
        self, variable_name: str, to_epsg: int, method: str = "nearest neighbor"
    ) -> NetCDF:
        """Reproject a single variable and store the result back.

        Convenience method that combines `get_variable` → `to_crs`
        → `set_variable` in one call.

        Args:
            variable_name: Name of the variable to reproject.
            to_epsg: Target EPSG code (e.g. 4326, 32637).
            method: Resampling method. Defaults to
                `"nearest neighbor"`.

        Returns:
            NetCDF: This container (modified in-place).
        """
        var = self.get_variable(variable_name)
        reprojected = var.to_crs(to_epsg, method=method)
        # to_crs returns a VRT-backed dataset — materialize it into
        # a MEM dataset so the data survives after the VRT source
        # (the variable subset) is garbage collected.
        arr = reprojected.read_array()
        no_data_value = reprojected.no_data_value
        ndv_scalar = (
            no_data_value[0]
            if isinstance(no_data_value, list) and no_data_value
            else no_data_value
        )
        materialized = Dataset.create_from_array(
            arr,
            geo=reprojected.geotransform,
            epsg=reprojected.epsg,
            no_data_value=ndv_scalar,
        )
        materialized._band_dim_name = var._band_dim_name
        materialized._band_dim_values = var._band_dim_values
        materialized._variable_attrs = var._variable_attrs
        self.set_variable(variable_name, materialized)
        return self

    def resample_variable(
        self,
        variable_name: str,
        cell_size: int | float,
        method: str = "nearest neighbor",
    ) -> NetCDF:
        """Resample a single variable and store the result back.

        Convenience method that combines `get_variable` → `resample`
        → `set_variable` in one call.

        Args:
            variable_name: Name of the variable to resample.
            cell_size: New cell size.
            method: Resampling method. Defaults to
                `"nearest neighbor"`.

        Returns:
            NetCDF: This container (modified in-place).
        """
        var = self.get_variable(variable_name)
        resampled = var.resample(cell_size, method=method)
        self.set_variable(variable_name, resampled)
        return self

    def add_variable(self, dataset: Dataset | NetCDF, variable_name: str | None = None):
        """Copy MDArray variables from another NetCDF into this container.

        Args:
            dataset: Source NetCDF dataset whose variables will be copied.
                Must have a root group (opened in MDIM mode).
            variable_name: Specific variable name(s) to copy. If None, all
                variables from the source are copied. If a variable with
                the same name already exists, it is renamed with a
                `"-new"` suffix.
        """
        src_rg = self._raster.GetRootGroup()
        var_rg = dataset._raster.GetRootGroup()
        names_to_copy: list[str]
        if variable_name is not None:
            names_to_copy = [variable_name]
        elif isinstance(dataset, NetCDF):
            names_to_copy = dataset.variable_names
        else:
            names_to_copy = []

        for var in names_to_copy:
            md_arr = var_rg.OpenMDArray(var)
            # If the variable name already exists in the destination dataset,
            # use a suffixed name to avoid overwriting the original.
            target_name = f"{var}-new" if var in self.variable_names else var
            self._add_md_array_to_group(src_rg, target_name, md_arr)
        self._invalidate_caches()

    def remove_variable(self, variable_name: str):
        """Delete a variable from this container.

        If the dataset is backed by a file on disk, a MEM copy is made first
        so that the on-disk file is not modified. The internal raster
        reference is replaced with the modified copy.

        Args:
            variable_name: Name of the variable to remove.
        """
        if self.driver_type == "memory":
            dst = self._raster
        else:
            dst = gdal.GetDriverByName("MEM").CreateCopy("", self._raster, 0)

        rg = dst.GetRootGroup()
        rg.DeleteMDArray(variable_name)

        self._replace_raster(dst)

    def rename_variable(self, old_name: str, new_name: str):
        """Rename a variable in this container.

        Internally extracts the variable data and metadata, creates
        a new variable with the new name, and removes the old one.

        Args:
            old_name: Current name of the variable.
            new_name: Desired new name.

        Raises:
            ValueError: If `old_name` doesn't exist or `new_name`
                already exists.
        """
        if old_name not in self.variable_names:
            raise ValueError(
                f"Variable '{old_name}' not found. " f"Available: {self.variable_names}"
            )
        if new_name in self.variable_names:
            raise ValueError(f"Variable '{new_name}' already exists.")

        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError("rename_variable requires a multidimensional container.")

        md_arr = rg.OpenMDArray(old_name)
        self._add_md_array_to_group(rg, new_name, md_arr)
        rg.DeleteMDArray(old_name)
        self._invalidate_caches()

    def to_xarray(self) -> Any:
        """Convert this NetCDF container to an `xarray.Dataset`.

        Builds an in-memory `xarray.Dataset` that mirrors the
        variables, coordinates, dimensions, and global attributes of
        this pyramids NetCDF container.

        The entire conversion goes through GDAL's Multidimensional
        API — the same reader the rest of pyramids' NetCDF code uses.
        No xarray engine plugin (`netcdf4`, `h5netcdf`,
        `scipy.io.netcdf`) is involved, so the `[xarray]` extra
        does not need to pull a NetCDF backend: pyramids is the
        backend. The returned `xr.Dataset` holds already-
        materialised numpy arrays; for lazy reads use
        :meth:`read_array(chunks=...)` and wrap the result in
        :class:`xarray.DataArray` yourself.

        Requires the optional `xarray` package. Install it with::

            pip install 'pyramids-gis[xarray]'

        Returns:
            xarray.Dataset: An xarray Dataset with the same
            variables, coordinates, and global attributes.

        Raises:
            pyramids.base._errors.OptionalPackageDoesNotExist:
                If `xarray` is not installed.
            ValueError: If the underlying GDAL handle is not a
                multidimensional container (open the file with
                `open_as_multi_dimensional=True`).

        Examples:
            Convert a pyramids NetCDF to xarray::

                nc = NetCDF.read_file("temperature.nc")
                ds = nc.to_xarray()
                print(ds)
        """
        try:
            import xarray as xr
        except ImportError:
            raise OptionalPackageDoesNotExist(
                "xarray is required for to_xarray(). "
                "Install it with: pip install 'pyramids-gis[xarray]'"
            )

        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError(
                "to_xarray requires a multidimensional container. "
                "Open the file with open_as_multi_dimensional=True."
            )

        coords: dict[str, Any] = {}
        dims = rg.GetDimensions() or []
        for d in dims:
            dim_name = d.GetName()
            iv = d.GetIndexingVariable()
            if iv is None:
                continue
            coord_attrs: dict[str, Any] = {}
            try:
                for attr in iv.GetAttributes():
                    coord_attrs[attr.GetName()] = attr.Read()
            except Exception:
                pass
            unit = iv.GetUnit()
            if unit and "units" not in coord_attrs:
                coord_attrs["units"] = unit
            coords[dim_name] = ([dim_name], iv.ReadAsArray(), coord_attrs)

        data_vars: dict[str, Any] = {}
        for var_name in self.variable_names:
            md_arr = rg.OpenMDArray(var_name)
            if md_arr is None:
                continue
            arr_dims = md_arr.GetDimensions() or []
            arr_dim_names = [ad.GetName() for ad in arr_dims]
            arr_data = md_arr.ReadAsArray()
            var_attrs: dict[str, Any] = {}
            try:
                for attr in md_arr.GetAttributes():
                    var_attrs[attr.GetName()] = attr.Read()
            except Exception:
                pass
            # GDAL's netCDF driver normalises the CF `units` attribute
            # to MDArray.GetUnit() / SetUnit() rather than a regular
            # attribute. Merge it back into var_attrs for a clean
            # round-trip through xr.Dataset.
            unit = md_arr.GetUnit()
            if unit and "units" not in var_attrs:
                var_attrs["units"] = unit
            data_vars[var_name] = (arr_dim_names, arr_data, var_attrs)

        result = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=self.global_attributes,
        )
        return result

    @classmethod
    def from_xarray(
        cls,
        dataset: Any,
        path: str | Path | None = None,
    ) -> NetCDF:
        """Create a pyramids NetCDF from an `xarray.Dataset`.

        Extracts dimensions, coordinates, data variables, and
        attributes from the `xarray.Dataset` and writes them to a
        NetCDF file through pyramids' own GDAL Multidimensional
        writer. No xarray engine plugin (`netcdf4`, `h5netcdf`)
        is invoked — pyramids is the writer, so the `[xarray]`
        extra does not need to pull a NetCDF backend.

        Usage::

            ds = xr.open_dataset("input.nc")
            #... xarray processing...
            nc = NetCDF.from_xarray(ds)
            var = nc.get_variable("temperature")
            cropped = var.crop(mask)

        Requires the optional `xarray` package.

        Args:
            dataset: An `xarray.Dataset` instance.
            path: File path where the NetCDF will be written. If
                `None`, a temp `.nc` is created and cleaned up
                when the returned object is garbage-collected.

        Returns:
            NetCDF: A pyramids NetCDF container backed by the data
            from the xarray Dataset.

        Raises:
            pyramids.base._errors.OptionalPackageDoesNotExist:
                If `xarray` is not installed.
            TypeError: If *dataset* is not an `xarray.Dataset`.
        """
        try:
            import xarray as xr
        except ImportError:
            raise OptionalPackageDoesNotExist(
                "xarray is required for from_xarray(). "
                "Install it with: pip install 'pyramids-gis[xarray]'"
            )

        if not isinstance(dataset, xr.Dataset):
            raise TypeError(f"Expected xarray.Dataset, got {type(dataset).__name__}")

        cleanup_temp = False
        if path is not None:
            path = str(path)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
            path = tmp.name
            tmp.close()
            cleanup_temp = True

        mem_src = cls._build_multidim_from_xarray(dataset)
        dst = gdal.GetDriverByName("netCDF").CreateCopy(path, mem_src, 0)
        if dst is None:
            raise RuntimeError(f"Failed to write NetCDF to {path}")
        dst.FlushCache()
        dst = None
        mem_src = None

        result = cls.read_file(path, read_only=True)
        if cleanup_temp:
            result._xarray_temp_path = path
            weakref.finalize(result, os.unlink, path)
        return result

    @staticmethod
    def _build_multidim_from_xarray(dataset: Any) -> gdal.Dataset:
        """Build an in-memory GDAL multidim container from an xarray Dataset.

        Creates dimensions from `dataset.sizes`, writes each
        coordinate as a 1-D indexing MDArray, writes each data
        variable as an N-D MDArray whose dimensions are resolved by
        name. Variable and global attributes are copied via pyramids'
        own `write_attributes_to_md_array` / `write_global_attributes`
        helpers so every type the CF layer already handles (str, int,
        float, bool, list) round-trips without going through xarray's
        NetCDF writer.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("from_xarray")
        root = src.GetRootGroup()

        gdal_dims: dict[str, gdal.Dimension] = {}
        for dim_name, dim_size in dataset.sizes.items():
            gdal_dims[dim_name] = root.CreateDimension(
                dim_name,
                "",
                "",
                int(dim_size),
            )

        def _apply_attrs(md_arr: gdal.MDArray, attrs: dict[str, Any]) -> None:
            """Write xarray var attrs, routing `units` through SetUnit.

            GDAL's netCDF writer moves the CF `units` attribute onto
            the MDArray's own unit slot; if we also write it as a regular
            attribute it's dropped on the next CreateCopy. Split it out
            so the round trip is lossless.
            """
            if not attrs:
                return
            remaining = dict(attrs)
            unit = remaining.pop("units", None)
            if unit is not None:
                md_arr.SetUnit(str(unit))
            if remaining:
                write_attributes_to_md_array(md_arr, remaining)

        for coord_name, coord in dataset.coords.items():
            if coord_name not in gdal_dims:
                continue
            values = np.asarray(coord.values)
            ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
            md_arr = root.CreateMDArray(
                coord_name,
                [gdal_dims[coord_name]],
                ext,
            )
            md_arr.Write(np.ascontiguousarray(values))
            _apply_attrs(md_arr, dict(coord.attrs))

        for var_name, var in dataset.data_vars.items():
            values = np.asarray(var.values)
            ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
            md_arr = root.CreateMDArray(
                var_name,
                [gdal_dims[d] for d in var.dims],
                ext,
            )
            md_arr.Write(np.ascontiguousarray(values))
            _apply_attrs(md_arr, dict(var.attrs))

        if dataset.attrs:
            write_global_attributes(root, dict(dataset.attrs))

        return src
