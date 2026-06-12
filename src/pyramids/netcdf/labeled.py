"""Non-gridded, label-indexed NetCDF / Zarr reading (GDAL multidimensional).

:class:`~pyramids.dataset.Dataset` and :class:`~pyramids.netcdf.NetCDF` model
raster data on a `(y, x)` grid. Many scientific stores are instead indexed by
a 1-D **label** dimension — a `feature_id` / `station` / `node` axis with
no spatial grid (for example a river-network `feature_id x time` streamflow
table, or a station time series). Reading those as if they were rasters is wrong.

:class:`LabeledDataset` reads such a store through GDAL's **multidimensional**
API (`gdal.OpenEx(..., OF_MULTIDIM_RASTER)` — the same engine pyramids uses for
gridded cloud reads), exposing the store's dimensions, coordinates, and variables
without forcing a raster interpretation, and without xarray / dask. Reading is
lazy: opening a store reads only metadata; a data array is materialised only when
a slice is realised (`__getitem__` / `to_dataframe` / a write). Selections are
index bookkeeping applied at read time, so `select` / `select_time` /
`select_bbox` read only the requested labels x timesteps.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

import cftime
import numpy as np
import pandas as pd
from osgeo import gdal

from pyramids.base._utils import import_pyarrow
from pyramids.base.remote import CloudConfig, _to_vsi

# Soft guard: realising a store this large into a DataFrame loads it all into
# memory. Above this many bytes the write methods warn the caller to slice first
# (the estimate is dtype x selected size — no data is read to compute it).
_LARGE_REALISE_BYTES = 512 * 1024 * 1024

_PARQUET_INSTALL_HINT = (
    "Writing Parquet needs the optional 'pyarrow' dependency. Install it with "
    "one of:\n"
    "  - PyPI:  pip install 'pyramids-gis[parquet]'\n"
    "  - conda: conda install -c conda-forge pyarrow"
)

# Remote object-store / http URL schemes (the part before "://"). A store opened
# from one of these is read through GDAL's /vsi* virtual filesystem.
_REMOTE_SCHEMES = ("s3", "gs", "gcs", "az", "abfs", "http", "https")


def _get_attr(obj: Any, name: str) -> Any:
    """Return MDArray/group attribute ``name`` or ``None`` if it does not exist.

    ``GetAttribute`` raises under ``gdal.UseExceptions()`` rather than returning
    ``None``, so the lookup is normalised here.
    """
    try:
        return obj.GetAttribute(name)
    except RuntimeError:
        return None


def _is_remote_url(source: str) -> bool:
    """True for a remote object-store / http URL (not a local filesystem path)."""
    scheme = source.split("://", 1)[0].lower() if "://" in source else ""
    return scheme in _REMOTE_SCHEMES


def _is_zarr_store(path: str | Path, engine: str | None) -> bool:
    """Decide whether `path` should be opened as Zarr.

    Honours an explicit `engine` first (`"zarr"` -> Zarr, any other engine
    -> not Zarr), then falls back to a `.zarr` suffix check.
    """
    if engine is not None:
        return engine == "zarr"
    return str(path).rstrip("/\\").endswith(".zarr")


class LabeledDataset:
    """A non-gridded, label-indexed NetCDF/Zarr store, read via GDAL multidim.

    Wraps a GDAL multidimensional root group whose primary axis is a 1-D label
    dimension (`feature_id` / `station` / `node`) rather than a `(y, x)`
    grid. Use it to inspect a store's structure, select by label, slice in time,
    subset by bbox, and write out tabular output (DataFrame / Parquet / CSV).

    Coordinates are detected the CF way: a dimension's indexing variable (a 1-D
    array named like its dimension) plus any variable named in a data variable's
    `coordinates` attribute (e.g. `latitude` / `longitude` / `gage_id`
    over the label dimension). Everything else is a data variable.
    """

    def __init__(
        self,
        ds: gdal.Dataset,
        group: gdal.Group,
        *,
        coord_names: list[str],
        var_names: list[str],
        dim_order: list[str],
        full_sizes: dict[str, int],
        index: dict[str, np.ndarray] | None = None,
        scalar_dims: frozenset[str] = frozenset(),
    ) -> None:
        """Low-level constructor. Most callers use :meth:`read_file`.

        Args:
            ds: The open GDAL multidim `gdal.Dataset` (kept alive by the view).
            group: The store's root :class:`osgeo.gdal.Group`.
            coord_names: Coordinate variable names.
            var_names: Data variable names (the visible subset).
            dim_order: Dimension names in the store's declared order.
            full_sizes: Each dimension's full length in the file.
            index: Per-dimension kept integer indices (into the full array);
                a missing dimension means "keep all".
            scalar_dims: Dimensions reduced to a single label (dropped from the
                reported dims/sizes).
        """
        self._ds = ds
        self._group = group
        self._coord_names = coord_names
        self._var_names = var_names
        self._dim_order = dim_order
        self._full_sizes = full_sizes
        self._index = dict(index or {})
        self._scalar_dims = scalar_dims

    @classmethod
    def read_file(
        cls,
        path: str | Path,
        *,
        variables: Iterable[str] | None = None,
        group: str | None = None,
        engine: str | None = None,
        anon: bool = False,
    ) -> LabeledDataset:
        """Open a label-indexed NetCDF or Zarr store lazily via GDAL multidim.

        The store kind is inferred from `path` (`.zarr` -> Zarr, otherwise
        NetCDF/HDF5) unless `engine` is given. Only metadata is read on open;
        data is materialised lazily on the first slice. Remote `s3://` /
        `gs://` / `az://` URLs are read through GDAL's `/vsi*` filesystem —
        pass `anon=True` for an unsigned public bucket.

        Args:
            path: Local path or a remote URL (`s3://`, `gs://`, ...).
            variables: Restrict to these data variables. `None` keeps all.
            group: Sub-group to open. `None` opens the root group.
            engine: Force the store kind — `"zarr"` for Zarr, any other value
                for NetCDF/HDF5. `None` infers from the path suffix.
            anon: Open the remote store anonymously (unsigned;
                `AWS_NO_SIGN_REQUEST` for S3 and the equivalent elsewhere).

        Returns:
            LabeledDataset: The opened store.

        Raises:
            ValueError: GDAL could not open the store as a multidimensional
                dataset.

        Examples:
            - Open the NWM retrospective Zarr anonymously::

                >>> store = LabeledDataset.read_file(  # doctest: +SKIP
                ...     "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
                ...     anon=True,
                ... )
        """
        source = str(path)
        gdal_path = _to_vsi(source) if _is_remote_url(source) else source
        if _is_zarr_store(source, engine):
            gdal_path = f'ZARR:"{gdal_path}"' if not gdal_path.startswith("ZARR:") else gdal_path
        try:
            with CloudConfig(aws_no_sign_request=anon):
                ds = gdal.OpenEx(gdal_path, gdal.OF_MULTIDIM_RASTER)
        except RuntimeError as exc:
            # gdal.UseExceptions() raises (rather than returning None) for a
            # missing / unrecognised store; normalise to a clear ValueError.
            raise ValueError(
                f"GDAL could not open {source!r} as a multidimensional store: {exc}"
            ) from exc
        if ds is None:
            raise ValueError(
                f"GDAL could not open {source!r} as a multidimensional store."
            )
        root = ds.GetRootGroup()
        grp = root.OpenGroup(group) if group else root
        if grp is None:
            raise ValueError(f"group {group!r} not found in {source!r}.")
        return cls._from_group(ds, grp, variables)

    @classmethod
    def _from_group(
        cls,
        ds: gdal.Dataset,
        grp: gdal.Group,
        variables: Iterable[str] | None,
    ) -> LabeledDataset:
        """Classify a group's arrays into coordinates and data variables."""
        array_names = list(grp.GetMDArrayNames())
        # Real dimensions are those some array actually spans; this drops phantom
        # group dimensions (e.g. a `string8` char-length axis a string array
        # carries internally but does not expose on its own GetDimensions()).
        used_dims: set[str] = set()
        for name in array_names:
            used_dims.update(d.GetName() for d in grp.OpenMDArray(name).GetDimensions())
        dim_order = [d.GetName() for d in grp.GetDimensions() if d.GetName() in used_dims]
        full_sizes = {
            d.GetName(): int(d.GetSize())
            for d in grp.GetDimensions()
            if d.GetName() in used_dims
        }
        # Dimension coordinates: an array named like a dimension's indexing var.
        dim_coords = set()
        for dim in grp.GetDimensions():
            iv = dim.GetIndexingVariable()
            if iv is not None:
                dim_coords.add(iv.GetName())
        # Auxiliary coordinates: anything listed in a variable's `coordinates`
        # attribute (CF). Data variables: the rest.
        aux_coords: set[str] = set()
        data_candidates = [n for n in array_names if n not in dim_coords]
        for name in data_candidates:
            attr = _get_attr(grp.OpenMDArray(name), "coordinates")
            if attr is not None:
                aux_coords.update(attr.ReadAsString().split())
        coord_names = [n for n in array_names if n in dim_coords or n in aux_coords]
        var_names = [n for n in array_names if n not in coord_names]
        if variables is not None:
            wanted = list(variables)
            var_names = [n for n in var_names if n in wanted]
        return cls(
            ds,
            grp,
            coord_names=coord_names,
            var_names=var_names,
            dim_order=dim_order,
            full_sizes=full_sizes,
        )

    def _replace(
        self, index: dict[str, np.ndarray], scalar_dims: frozenset[str]
    ) -> LabeledDataset:
        """Return a new view over the same store with a different selection."""
        return type(self)(
            self._ds,
            self._group,
            coord_names=self._coord_names,
            var_names=self._var_names,
            dim_order=self._dim_order,
            full_sizes=self._full_sizes,
            index=index,
            scalar_dims=scalar_dims,
        )

    def _selected_size(self, dim: str) -> int:
        """Current selected length of `dim` (full when no selection applied)."""
        idx = self._index.get(dim)
        return self._full_sizes[dim] if idx is None else int(idx.size)

    def _array_dims(self, name: str) -> list[str]:
        """Dimension names of array `name`, in storage order."""
        return [d.GetName() for d in self._group.OpenMDArray(name).GetDimensions()]

    @property
    def sizes(self) -> dict[str, int]:
        """Mapping of dimension name to selected length (scalar dims excluded)."""
        return {
            d: self._selected_size(d)
            for d in self._dim_order
            if d not in self._scalar_dims
        }

    @property
    def dimensions(self) -> list[str]:
        """Dimension names (in store order), excluding scalar-selected ones."""
        return [d for d in self._dim_order if d not in self._scalar_dims]

    @property
    def coordinates(self) -> list[str]:
        """Coordinate variable names (e.g. `feature_id`, `time`, `latitude`)."""
        return list(self._coord_names)

    @property
    def variables(self) -> list[str]:
        """Data-variable names (excludes coordinates)."""
        return list(self._var_names)

    def _read(self, name: str) -> tuple[np.ndarray, tuple[str, ...]]:
        """Read array `name` with the current selection applied.

        Returns:
            tuple: `(values, dims)` — the materialised numpy array (object
            dtype for string arrays) and its remaining dimension names (after
            scalar dims are squeezed out).
        """
        arr = self._group.OpenMDArray(name)
        dim_names = self._array_dims(name)
        is_string = arr.GetDataType().GetClass() == gdal.GEDTC_STRING
        if is_string:
            values = np.asarray(arr.Read(), dtype=object)
            for axis, dim in enumerate(dim_names):
                idx = self._index.get(dim)
                if idx is not None:
                    values = np.take(values, np.asarray(idx), axis=axis)
        else:
            starts, counts, local = [], [], []
            for dim in dim_names:
                idx = self._index.get(dim)
                if idx is None:
                    starts.append(0)
                    counts.append(self._full_sizes[dim])
                    local.append(None)
                else:
                    lo, hi = int(idx.min()), int(idx.max())
                    starts.append(lo)
                    counts.append(hi - lo + 1)
                    local.append(np.asarray(idx) - lo)
            values = np.asarray(arr.ReadAsArray(array_start_idx=starts, count=counts))
            for axis, loc in enumerate(local):
                if loc is not None:
                    values = np.take(values, loc, axis=axis)
        keep_axes = tuple(i for i, d in enumerate(dim_names) if d not in self._scalar_dims)
        scalar_axes = tuple(i for i, d in enumerate(dim_names) if d in self._scalar_dims)
        if scalar_axes:
            values = np.squeeze(values, axis=scalar_axes)
        out_dims = tuple(dim_names[i] for i in keep_axes)
        return values, out_dims

    def _coord_full(self, name: str) -> np.ndarray:
        """Read a coordinate's full (unselected) values."""
        arr = self._group.OpenMDArray(name)
        if arr.GetDataType().GetClass() == gdal.GEDTC_STRING:
            return np.asarray(arr.Read(), dtype=object)
        return np.asarray(arr.ReadAsArray())

    def _coord_current(self, name: str, dim: str) -> np.ndarray:
        """Read coordinate `name` over `dim` at the current selection."""
        full = self._coord_full(name)
        idx = self._index.get(dim)
        return full if idx is None else full[np.asarray(idx)]

    def _require_coord(self, name: str) -> None:
        """Raise `KeyError` if `name` is not a coordinate of the store."""
        if name not in self._coord_names:
            raise KeyError(
                f"{name!r} is not a coordinate of this store; available: "
                f"{self.coordinates}"
            )

    def select(self, **labels: Any) -> LabeledDataset:
        """Select by label-dimension coordinate value(s).

        Each keyword is a coordinate (typically a dimension coordinate such as
        `feature_id` / `station`) mapped to a value or list of values to
        keep, in the requested order. A scalar drops that dimension. Missing
        labels are reported rather than silently dropped.

        Args:
            **labels: `dim=value` or `dim=[values]` selectors. A scalar
                drops the dimension; a list/tuple/array keeps the matching
                entries in the requested order. An empty sequence is an error.

        Returns:
            LabeledDataset: A new view with only the selected labels.

        Raises:
            KeyError: `dim` is not a coordinate, or any value is absent.
            ValueError: A selection list is empty.

        Examples:
            - Keep three reaches by `feature_id`::

                >>> sub = store.select(feature_id=[101, 202, 303])  # doctest: +SKIP
        """
        index = dict(self._index)
        scalar = set(self._scalar_dims)
        for dim, values in labels.items():
            self._require_coord(dim)
            current = self._coord_current(dim, dim)
            is_scalar = not isinstance(values, (list, tuple, np.ndarray))
            requested = [values] if is_scalar else list(values)
            if not requested:
                raise ValueError(
                    f"empty selection list for {dim!r}; pass at least one value"
                )
            found = np.isin(np.asarray(requested), current)
            if not found.all():
                missing = [v for v, ok in zip(requested, found) if not ok]
                raise KeyError(f"{dim!r} values not found: {missing}")
            # Positions within the current selection, in REQUEST order.
            positions = [int(np.flatnonzero(current == v)[0]) for v in requested]
            base = self._index.get(dim)
            base = np.arange(self._full_sizes[dim]) if base is None else np.asarray(base)
            index[dim] = base[positions]
            if is_scalar:
                scalar.add(dim)
        return self._replace(index, frozenset(scalar))

    def select_by_coord(self, coord: str, values: Iterable[Any]) -> LabeledDataset:
        """Select along a non-dimension 1-D coordinate (e.g. a `gage_id` join).

        `coord` is a 1-D coordinate over the label dimension; the store is
        masked to the entries whose coordinate value is in `values`, preserving
        the store's original order.

        Args:
            coord: Name of the 1-D coordinate to filter on.
            values: Coordinate values to keep.

        Returns:
            LabeledDataset: A new view with only the matching entries.

        Raises:
            KeyError: `coord` is absent or not 1-D, or any value is missing.
            ValueError: `values` is empty.

        Examples:
            - Select reaches by their USGS gauge id::

                >>> sub = store.select_by_coord("gage_id", ["01010000"])  # doctest: +SKIP
        """
        self._require_coord(coord)
        dims = self._array_dims(coord)
        if len(dims) != 1:
            raise KeyError(f"{coord!r} must be 1-D to select on; it is {len(dims)}-D")
        dim = dims[0]
        current = self._coord_current(coord, dim)
        requested = list(values)
        if not requested:
            raise ValueError(
                f"empty selection list for {coord!r}; pass at least one value"
            )
        found = np.isin(np.asarray(requested), current)
        if not found.all():
            missing = [v for v, ok in zip(requested, found) if not ok]
            raise KeyError(f"{coord!r} values not found: {missing}")
        keep_positions = np.flatnonzero(np.isin(current, np.asarray(requested)))
        base = self._index.get(dim)
        base = np.arange(self._full_sizes[dim]) if base is None else np.asarray(base)
        index = dict(self._index)
        index[dim] = base[keep_positions]
        return self._replace(index, self._scalar_dims)

    def _time_to_num(self, value: Any, unit: str, calendar: str) -> float:
        """Convert a date string / datetime to the time axis's numeric scale."""
        ts = pd.Timestamp(value)
        dt = cftime.datetime(
            ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second,
            calendar=calendar,
        )
        return float(cftime.date2num(dt, unit, calendar))

    def select_time(
        self,
        start: Any = None,
        end: Any = None,
        *,
        time_dim: str = "time",
    ) -> LabeledDataset:
        """Slice the time axis to a `[start, end]` window (inclusive).

        Composes with :meth:`select` / :meth:`select_by_coord`. Bounds are
        compared in the time variable's own numeric scale (`units` + CF
        `calendar`), so non-standard calendars (`360_day` / `noleap`) work.

        Args:
            start: Lower bound (inclusive). A string (`"2010-06-01"`),
                `datetime`, or `None` to start at the first step.
            end: Upper bound (inclusive). Same forms; `None` runs to the last.
            time_dim: Name of the time dimension. Defaults to `"time"`.

        Returns:
            LabeledDataset: A new view restricted to the in-range timesteps.

        Raises:
            KeyError: `time_dim` is not a coordinate of the store.
            ValueError: The window selects no timesteps.

        Examples:
            - Restrict to a summer window::

                >>> sub = store.select_time("2010-06-01", "2010-08-31")  # doctest: +SKIP
        """
        self._require_coord(time_dim)
        arr = self._group.OpenMDArray(time_dim)
        unit = arr.GetUnit()
        cal_attr = _get_attr(arr, "calendar")
        calendar = cal_attr.ReadAsString() if cal_attr is not None else "standard"
        current = self._coord_current(time_dim, time_dim).astype("float64")
        mask = np.ones(current.shape, dtype=bool)
        if start is not None:
            mask &= current >= self._time_to_num(start, unit, calendar)
        if end is not None:
            mask &= current <= self._time_to_num(end, unit, calendar)
        if not mask.any():
            full = self._coord_full(time_dim)
            raise ValueError(
                f"no timesteps in window [{start}, {end}]; the store spans "
                f"[{full.min()}, {full.max()}] in '{unit}'"
            )
        base = self._index.get(time_dim)
        base = np.arange(self._full_sizes[time_dim]) if base is None else np.asarray(base)
        index = dict(self._index)
        index[time_dim] = base[np.flatnonzero(mask)]
        return self._replace(index, self._scalar_dims)

    def select_bbox(
        self,
        bbox: tuple[float, float, float, float],
        *,
        lon: str = "longitude",
        lat: str = "latitude",
    ) -> LabeledDataset:
        """Subset to labels whose 1-D lon/lat coords fall inside a bbox.

        For a label-indexed store, `latitude` / `longitude` are 1-D
        coordinates over the label dimension (one position per label); this masks
        the label dimension to the entries inside the bounding box. (Gridded
        raster crops belong on :class:`~pyramids.dataset.Dataset`, not here.)

        Args:
            bbox: `(min_lon, min_lat, max_lon, max_lat)` — i.e.
                `(west, south, east, north)`, inclusive.
            lon: Name of the 1-D longitude coordinate. Defaults to `"longitude"`.
            lat: Name of the 1-D latitude coordinate. Defaults to `"latitude"`.

        Returns:
            LabeledDataset: A new view with only the labels inside the bbox.

        Raises:
            KeyError: `lon`/`lat` is missing, not 1-D, or over different dims.
            ValueError: No label falls inside the bbox.

        Examples:
            - Keep reaches inside a small box::

                >>> sub = store.select_bbox((-77.0, 40.0, -75.0, 42.0))  # doctest: +SKIP
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        for name in (lon, lat):
            self._require_coord(name)
        lon_dims = self._array_dims(lon)
        lat_dims = self._array_dims(lat)
        if len(lon_dims) != 1 or len(lat_dims) != 1:
            raise KeyError(
                f"{lon!r} and {lat!r} must be 1-D coordinates over the label "
                "dimension to bbox-subset a label-indexed store"
            )
        if lon_dims != lat_dims:
            raise KeyError(
                f"{lon!r} (dim {lon_dims[0]!r}) and {lat!r} (dim {lat_dims[0]!r}) "
                "must be over the same dimension"
            )
        dim = lon_dims[0]
        lon_vals = self._coord_current(lon, dim).astype("float64")
        lat_vals = self._coord_current(lat, dim).astype("float64")
        mask = (
            (lon_vals >= min_lon)
            & (lon_vals <= max_lon)
            & (lat_vals >= min_lat)
            & (lat_vals <= max_lat)
        )
        if not mask.any():
            raise ValueError(
                f"no labels inside bbox {bbox}; the store spans longitude "
                f"[{lon_vals.min()}, {lon_vals.max()}], latitude "
                f"[{lat_vals.min()}, {lat_vals.max()}]"
            )
        base = self._index.get(dim)
        base = np.arange(self._full_sizes[dim]) if base is None else np.asarray(base)
        index = dict(self._index)
        index[dim] = base[np.flatnonzero(mask)]
        return self._replace(index, self._scalar_dims)

    def _estimated_nbytes(self) -> int:
        """Estimate the bytes a full materialisation would read (no read done)."""
        total = 0
        for name in self._var_names:
            arr = self._group.OpenMDArray(name)
            itemsize = arr.GetDataType().GetSize() or 4
            cells = 1
            for dim in self._array_dims(name):
                cells *= self._selected_size(dim)
            total += cells * itemsize
        return total

    def to_dataframe(self) -> pd.DataFrame:
        """Return the store as a tidy :class:`pandas.DataFrame`.

        Every dimension/coordinate becomes a column and each data variable a
        column, one row per `(label, time, ...)` cell. Realises the selected
        data, so slice with :meth:`select` / :meth:`select_time` /
        :meth:`select_bbox` first for large stores.

        Returns:
            pandas.DataFrame: The tidy table.

        Examples:
            - Tidy a small streamflow subset::

                >>> df = store.select(feature_id=[101]).to_dataframe()  # doctest: +SKIP
        """
        if self._estimated_nbytes() > _LARGE_REALISE_BYTES:
            warnings.warn(
                f"realising ~{self._estimated_nbytes() / 1e9:.1f} GB into a "
                "DataFrame; slice with select / select_time / select_bbox first "
                "to avoid loading the whole store into memory.",
                stacklevel=2,
            )
        out_dims = self.dimensions
        if out_dims:
            shape = tuple(self._selected_size(d) for d in out_dims)
            grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
            columns: dict[str, np.ndarray] = {}
            for coord in self._coord_names:
                values, dims = self._read(coord)
                columns[coord] = self._broadcast_to_rows(values, dims, out_dims, grids)
            for var in self._var_names:
                values, dims = self._read(var)
                columns[var] = self._broadcast_to_rows(values, dims, out_dims, grids)
            return pd.DataFrame({k: np.asarray(v).ravel() for k, v in columns.items()})
        # Fully scalar selection -> a single row.
        row = {}
        for name in self._coord_names + self._var_names:
            values, _ = self._read(name)
            row[name] = [np.asarray(values).reshape(-1)[0]]
        return pd.DataFrame(row)

    @staticmethod
    def _broadcast_to_rows(
        values: np.ndarray,
        dims: tuple[str, ...],
        out_dims: list[str],
        grids: list[np.ndarray],
    ) -> np.ndarray:
        """Broadcast an array over `dims` onto the full `out_dims` row grid."""
        if not dims:
            return np.full(grids[0].shape, np.asarray(values).reshape(-1)[0])
        axis_for = {d: i for i, d in enumerate(out_dims)}
        gathered = values
        # index `values` along the out-dims it actually spans, via the meshgrid.
        idx: list[Any] = []
        for d in dims:
            idx.append(grids[axis_for[d]])
        return gathered[tuple(idx)]

    def to_parquet(self, path: str | Path, **kwargs: Any) -> Path:
        """Write the store to a Parquet file (tidy table).

        Args:
            path: Output `.parquet` path.
            **kwargs: Forwarded to :meth:`pandas.DataFrame.to_parquet`.

        Returns:
            Path: The written file path.

        Raises:
            OptionalPackageDoesNotExist: When pyarrow is not installed.
        """
        import_pyarrow(_PARQUET_INSTALL_HINT)
        path = Path(path)
        self.to_dataframe().to_parquet(str(path), index=False, **kwargs)
        return path

    def to_csv(self, path: str | Path, **kwargs: Any) -> Path:
        """Write the store to a CSV file (tidy table).

        Args:
            path: Output `.csv` path.
            **kwargs: Forwarded to :meth:`pandas.DataFrame.to_csv`.

        Returns:
            Path: The written file path.
        """
        path = Path(path)
        self.to_dataframe().to_csv(str(path), index=False, **kwargs)
        return path

    def __getitem__(self, key: str) -> _LabeledArray:
        """Return a variable or coordinate as a small `(values, dims, shape)` view."""
        if key not in self._coord_names and key not in self._var_names:
            raise KeyError(f"{key!r} is not in this store")
        values, dims = self._read(key)
        values = np.asarray(values)
        return _LabeledArray(values=values, dims=dims, shape=values.shape)

    def __contains__(self, key: str) -> bool:
        """True when `key` is a variable or coordinate in the store."""
        return key in self._coord_names or key in self._var_names

    def __repr__(self) -> str:
        """Compact, structure-only representation (no data read)."""
        return (
            f"<LabeledDataset sizes={self.sizes} "
            f"coords={self.coordinates} variables={self.variables}>"
        )


class _LabeledArray:
    """A materialised variable/coordinate slice: `values` plus `dims`/`shape`."""

    __slots__ = ("values", "dims", "shape")

    def __init__(self, values: np.ndarray, dims: tuple[str, ...], shape: tuple[int, ...]):
        self.values = values
        self.dims = dims
        self.shape = shape

    def __repr__(self) -> str:
        return f"_LabeledArray(dims={self.dims}, shape={self.shape})"
