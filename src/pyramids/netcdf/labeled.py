"""Non-gridded, label-indexed NetCDF / Zarr reading.

:class:`~pyramids.dataset.Dataset` and :class:`~pyramids.netcdf.NetCDF` model
raster data on a ``(y, x)`` grid. Many scientific stores are instead indexed by
a 1-D **label** dimension — a ``feature_id`` / ``station`` / ``node`` axis with
no spatial grid (for example a river-network ``feature_id x time`` streamflow
table, or a station time series). GDAL reads those as if they were rasters,
which is wrong.

:class:`LabeledDataset` reads such a store into a thin, lazily-backed wrapper
over an :class:`xarray.Dataset`, exposing the store's dimensions, coordinates,
and variables without forcing a raster interpretation.

Backed by xarray (the optional ``[xarray]`` extra). Reading is lazy: opening a
store reads only metadata; data arrays are not materialised until a slice is
realised. Pass ``chunks={}`` (or any xarray ``chunks`` spec) to back the arrays
with dask for chunked, out-of-core reads of very large stores (needs the
``[lazy]`` extra).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pyramids.base._utils import import_dask, import_pyarrow, import_xarray

# Soft guard: writing a store this large into a DataFrame realises it all into
# memory. Above this many bytes, the write methods warn the caller to slice
# first (xarray computes ``nbytes`` from dtype x size — no data is read).
_LARGE_REALISE_BYTES = 512 * 1024 * 1024

_XARRAY_INSTALL_HINT = (
    "Reading a label-indexed NetCDF/Zarr store needs the optional 'xarray' "
    "dependency. Install it with one of:\n"
    "  - PyPI:  pip install 'pyramids-gis[xarray]'\n"
    "  - conda: conda install -c conda-forge xarray"
)
_PARQUET_INSTALL_HINT = (
    "Writing Parquet needs the optional 'pyarrow' dependency. Install it with "
    "one of:\n"
    "  - PyPI:  pip install 'pyramids-gis[parquet]'\n"
    "  - conda: conda install -c conda-forge pyarrow"
)
_DASK_INSTALL_HINT = (
    "Chunked (dask-backed) reads need the optional 'dask' dependency. Install "
    "it with one of:\n"
    "  - PyPI:  pip install 'pyramids-gis[lazy]'\n"
    "  - conda: conda install -c conda-forge dask"
)

# Remote object-store / http URL schemes (the part before "://"). A store opened
# from one of these is assumed large, so reads default to dask chunks (never load
# the whole thing).
_REMOTE_SCHEMES = ("s3", "gs", "gcs", "az", "abfs", "http", "https")
# Sentinel so an explicit ``chunks=None`` is distinguishable from "not passed".
_UNSET = object()


def _is_remote_url(source: str) -> bool:
    """True for a remote object-store / http URL (not a local filesystem path)."""
    scheme = source.split("://", 1)[0].lower() if "://" in source else ""
    return scheme in _REMOTE_SCHEMES


def _is_zarr_store(path: str | Path, engine: str | None) -> bool:
    """Decide whether ``path`` should be opened as Zarr.

    Honours an explicit ``engine`` first (``"zarr"`` -> Zarr, any other engine
    -> not Zarr), then falls back to a ``.zarr`` suffix check.
    """
    if engine is not None:
        return engine == "zarr"
    return str(path).rstrip("/\\").endswith(".zarr")


class LabeledDataset:
    """A non-gridded, label-indexed NetCDF/Zarr store.

    Wraps an :class:`xarray.Dataset` whose primary axis is a 1-D label
    dimension (``feature_id`` / ``station`` / ``node``) rather than a ``(y, x)``
    grid. Use it to inspect a store's structure, select by label, slice in
    time, subset by bbox, and write out tabular output (DataFrame / Parquet /
    CSV).

    Args:
        dataset: The backing :class:`xarray.Dataset`.

    Attributes:
        dataset: The wrapped :class:`xarray.Dataset` (escape hatch for advanced
            xarray operations).
    """

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset

    @classmethod
    def read_file(
        cls,
        path: str | Path,
        *,
        variables: Iterable[str] | None = None,
        group: str | None = None,
        chunks: Any = _UNSET,
        engine: str | None = None,
        anon: bool = False,
        storage_options: dict[str, Any] | None = None,
        consolidated: bool | None = None,
    ) -> LabeledDataset:
        """Open a label-indexed NetCDF or Zarr store lazily.

        The store kind is inferred from ``path`` (``.zarr`` -> Zarr, otherwise
        NetCDF/HDF5) unless ``engine`` is given. Only metadata is read on open;
        data is materialised lazily. Remote ``s3://`` / ``gs://`` URLs are
        resolved via fsspec — pass ``anon=True`` for an unsigned public bucket
        (needs the ``[lazy]`` extra, which ships ``s3fs``).

        Args:
            path: Local path or a fsspec URL (``s3://``, ``gs://``, ...).
            variables: Restrict to these data variables. ``None`` keeps all.
            group: NetCDF/Zarr group to open. ``None`` opens the root group.
            chunks: xarray ``chunks`` spec. Defaults to ``{}`` (dask-backed,
                chunked) for a **remote** URL — so slicing reads only the
                touched chunks and a huge store is never loaded whole — and to
                ``None`` (no dask) for a **local** path. Pass an explicit value
                to override: ``None`` (numpy-backed; a later full-variable read
                loads it all), ``{}`` / ``"auto"`` / a dict (dask, needs the
                ``[lazy]`` extra).
            engine: Force an xarray engine (e.g. ``"zarr"``, ``"h5netcdf"``,
                ``"netcdf4"``). ``None`` lets xarray infer it.
            anon: Open the remote store anonymously (unsigned). Shorthand for
                ``storage_options={"anon": True}``; an explicit
                ``storage_options["anon"]`` wins.
            storage_options: fsspec options for remote stores. Merged with
                ``anon``.
            consolidated: Zarr only — whether to read consolidated metadata.
                ``None`` lets xarray decide.

        Returns:
            LabeledDataset: The opened store.

        Raises:
            OptionalPackageDoesNotExist: When xarray is not installed.

        Examples:
            - Open the NWM retrospective Zarr anonymously, lazily::

                >>> store = LabeledDataset.read_file(  # doctest: +SKIP
                ...     "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
                ...     anon=True,
                ...     chunks={},
                ... )
        """
        xr = import_xarray(_XARRAY_INSTALL_HINT)
        source = str(path)
        if chunks is _UNSET:
            chunks = {} if _is_remote_url(source) else None
        if chunks is not None:
            # dask backs any non-None chunks spec; fail early with a clear hint.
            import_dask(_DASK_INSTALL_HINT)
        if anon:
            storage_options = {"anon": True, **(storage_options or {})}
        if _is_zarr_store(source, engine):
            dataset = xr.open_zarr(
                source,
                chunks=chunks,
                group=group,
                consolidated=consolidated,
                storage_options=storage_options,
            )
        else:
            open_kwargs: dict[str, Any] = {"chunks": chunks, "group": group}
            if engine is not None:
                open_kwargs["engine"] = engine
            if storage_options is not None:
                open_kwargs["storage_options"] = storage_options
            dataset = xr.open_dataset(source, **open_kwargs)
        if variables is not None:
            dataset = dataset[list(variables)]
        return cls(dataset)

    @property
    def dataset(self) -> Any:
        """The backing :class:`xarray.Dataset`."""
        return self._dataset

    @property
    def sizes(self) -> dict[str, int]:
        """Mapping of dimension name to length (e.g. ``{"time": 4, "feature_id": 3}``)."""
        return dict(self._dataset.sizes)

    @property
    def dimensions(self) -> list[str]:
        """Dimension names, in the store's order."""
        return list(self._dataset.sizes)

    @property
    def coordinates(self) -> list[str]:
        """Coordinate variable names (e.g. ``feature_id``, ``time``, ``latitude``)."""
        return list(self._dataset.coords)

    @property
    def variables(self) -> list[str]:
        """Data-variable names (excludes coordinates)."""
        return list(self._dataset.data_vars)

    def select(self, **labels: Any) -> LabeledDataset:
        """Select by label-dimension coordinate value(s).

        Each keyword is a coordinate (typically a dimension coordinate such as
        ``feature_id`` / ``station``) mapped to a value or list of values to
        keep. A 1-D non-dimension coordinate also works; for an explicit,
        order-preserving join on one use :meth:`select_by_coord`. Missing labels
        are reported rather than silently dropped.

        Args:
            **labels: ``dim=value`` or ``dim=[values]`` selectors. A scalar
                drops that dimension; a list keeps the matching entries in the
                requested order.

        Returns:
            LabeledDataset: A new store with only the selected labels.

        Raises:
            KeyError: When ``dim`` is not a coordinate, or any requested value
                is absent (the missing values are named).
            ValueError: When a selection list is empty.

        Examples:
            - Keep three reaches by ``feature_id``::

                >>> sub = store.select(feature_id=[101, 202, 303])  # doctest: +SKIP
        """
        result = self._dataset
        for dim, values in labels.items():
            # value-based selection needs a coordinate; a bare dimension with no
            # coordinate variable cannot be selected by label.
            if dim not in self._dataset.coords:
                raise KeyError(
                    f"{dim!r} is not a coordinate of this store; available: "
                    f"{self.coordinates}"
                )
            # a scalar selects-and-drops the dimension; any sequence keeps it.
            # Normalise sequences to a list so a tuple is treated as a list of
            # labels, not a single (MultiIndex-style) label.
            is_scalar = not isinstance(values, (list, tuple, np.ndarray))
            requested = [values] if is_scalar else list(values)
            if not requested:
                raise ValueError(
                    f"empty selection list for {dim!r}; pass at least one value"
                )
            # Membership test against the numpy coordinate (vectorised, no
            # full-coordinate Python set) so a million-label store stays cheap.
            coord_values = np.asarray(self._dataset[dim].values)
            found = np.isin(np.asarray(requested), coord_values)
            if not found.all():
                missing = [v for v, ok in zip(requested, found) if not ok]
                raise KeyError(f"{dim!r} values not found: {missing}")
            result = result.sel({dim: values if is_scalar else requested})
        return type(self)(result)

    def select_by_coord(self, coord: str, values: Iterable[Any]) -> LabeledDataset:
        """Select along a non-dimension 1-D coordinate (e.g. a ``gage_id`` join).

        ``coord`` is a 1-D coordinate over the label dimension (for example a
        ``gage_id`` array of shape ``(feature_id,)``); the store is masked to the
        entries whose coordinate value is in ``values``.

        Args:
            coord: Name of the 1-D coordinate to filter on.
            values: Coordinate values to keep.

        Returns:
            LabeledDataset: A new store with only the matching entries, in the
            store's original order.

        Raises:
            KeyError: When ``coord`` is absent or not 1-D, or any requested value
                is not present (the missing values are named).
            ValueError: When ``values`` is empty.

        Examples:
            - Select reaches by their USGS gauge id::

                >>> sub = store.select_by_coord("gage_id", ["01010000"])  # doctest: +SKIP
        """
        if coord not in self._dataset.coords:
            raise KeyError(
                f"{coord!r} is not a coordinate of this store; available: "
                f"{self.coordinates}"
            )
        coord_arr = self._dataset[coord]
        if coord_arr.ndim != 1:
            raise KeyError(f"{coord!r} must be 1-D to select on; it is {coord_arr.ndim}-D")
        dim = coord_arr.dims[0]
        coord_values = np.asarray(coord_arr.values)
        requested = list(values)
        if not requested:
            raise ValueError(
                f"empty selection list for {coord!r}; pass at least one value"
            )
        # Vectorised membership test (no full-coordinate Python set).
        found = np.isin(np.asarray(requested), coord_values)
        if not found.all():
            missing = [v for v, ok in zip(requested, found) if not ok]
            raise KeyError(f"{coord!r} values not found: {missing}")
        keep = np.flatnonzero(np.isin(coord_values, requested))
        return type(self)(self._dataset.isel({dim: keep}))

    def select_time(
        self,
        start: Any = None,
        end: Any = None,
        *,
        time_dim: str = "time",
    ) -> LabeledDataset:
        """Slice the time axis to a ``[start, end]`` window (inclusive).

        Composes with :meth:`select` / :meth:`select_by_coord` so only the
        requested labels x timesteps are read. ``cftime`` axes (non-standard
        calendars such as ``360_day`` / ``noleap``) are handled by xarray.

        Args:
            start: Lower bound (inclusive). A string (``"2010-06-01"``),
                ``datetime``/``cftime``, or ``None`` to start at the first step.
            end: Upper bound (inclusive). Same forms; ``None`` runs to the last
                step.
            time_dim: Name of the time dimension. Defaults to ``"time"``.

        Returns:
            LabeledDataset: A new store restricted to the in-range timesteps.

        Raises:
            KeyError: When ``time_dim`` is not a coordinate of the store.
            ValueError: When the window selects no timesteps (e.g. it lies
                entirely outside the store's time span) — reported, never a
                silent empty result.

        Examples:
            - Restrict to a summer window::

                >>> sub = store.select_time("2010-06-01", "2010-08-31")  # doctest: +SKIP
        """
        if time_dim not in self._dataset.coords:
            raise KeyError(
                f"{time_dim!r} is not a coordinate of this store; available: "
                f"{self.coordinates}"
            )
        result = self._dataset.sel({time_dim: slice(start, end)})
        if int(result.sizes.get(time_dim, 0)) == 0:
            times = self._dataset[time_dim].values
            raise ValueError(
                f"no timesteps in window [{start}, {end}]; the store spans "
                f"[{times.min()}, {times.max()}]"
            )
        return type(self)(result)

    def select_bbox(
        self,
        bbox: tuple[float, float, float, float],
        *,
        lon: str = "longitude",
        lat: str = "latitude",
    ) -> LabeledDataset:
        """Subset to labels whose 1-D lon/lat coords fall inside a bbox.

        For a label-indexed store, ``latitude`` / ``longitude`` are 1-D
        coordinates over the label dimension (one position per ``feature_id`` /
        ``station``); this masks the label dimension to the entries inside the
        bounding box. (Gridded raster crops belong on
        :class:`~pyramids.dataset.Dataset`, not here.)

        Args:
            bbox: ``(min_lon, min_lat, max_lon, max_lat)`` — i.e.
                ``(west, south, east, north)`` — in the store's coordinate units
                (degrees). Bounds are inclusive.
            lon: Name of the 1-D longitude coordinate. Defaults to
                ``"longitude"``.
            lat: Name of the 1-D latitude coordinate. Defaults to ``"latitude"``.

        Returns:
            LabeledDataset: A new store with only the labels inside the bbox.

        Raises:
            KeyError: When ``lon`` / ``lat`` is missing, not 1-D, or the two are
                not over the same dimension.
            ValueError: When no label falls inside the bbox (the coordinate
                ranges are reported) — never a silent empty result.

        Examples:
            - Keep reaches inside a small box::

                >>> sub = store.select_bbox((-77.0, 40.0, -75.0, 42.0))  # doctest: +SKIP
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        for name in (lon, lat):
            if name not in self._dataset.coords:
                raise KeyError(
                    f"{name!r} is not a coordinate of this store; available: "
                    f"{self.coordinates}"
                )
        lon_arr = self._dataset[lon]
        lat_arr = self._dataset[lat]
        if lon_arr.ndim != 1 or lat_arr.ndim != 1:
            raise KeyError(
                f"{lon!r} and {lat!r} must be 1-D coordinates over the label "
                "dimension to bbox-subset a label-indexed store"
            )
        if lon_arr.dims != lat_arr.dims:
            raise KeyError(
                f"{lon!r} (dims {lon_arr.dims}) and {lat!r} (dims {lat_arr.dims}) "
                "must be over the same dimension"
            )
        dim = lon_arr.dims[0]
        lon_vals = np.asarray(lon_arr.values)
        lat_vals = np.asarray(lat_arr.values)
        mask = (
            (lon_vals >= min_lon)
            & (lon_vals <= max_lon)
            & (lat_vals >= min_lat)
            & (lat_vals <= max_lat)
        )
        keep = np.flatnonzero(mask)
        if keep.size == 0:
            raise ValueError(
                f"no labels inside bbox {bbox}; the store spans longitude "
                f"[{lon_vals.min()}, {lon_vals.max()}], latitude "
                f"[{lat_vals.min()}, {lat_vals.max()}]"
            )
        return type(self)(self._dataset.isel({dim: keep}))

    def to_dataframe(self) -> pd.DataFrame:
        """Return the store as a tidy :class:`pandas.DataFrame`.

        Every dimension/coordinate becomes a column and each data variable a
        column, one row per ``(label, time, ...)`` cell — the canonical shape
        for a ``feature_id x time`` table. Realises the (possibly lazy) data, so
        slice with :meth:`select` / :meth:`select_time` / :meth:`select_bbox`
        first for large stores.

        Returns:
            pandas.DataFrame: The tidy table with the index reset to columns.

        Examples:
            - Tidy a small streamflow subset::

                >>> df = store.select(feature_id=[101]).to_dataframe()  # doctest: +SKIP
        """
        nbytes = int(getattr(self._dataset, "nbytes", 0) or 0)
        if nbytes > _LARGE_REALISE_BYTES:
            warnings.warn(
                f"realising ~{nbytes / 1e9:.1f} GB into a DataFrame; slice with "
                "select / select_time / select_bbox first to avoid loading the "
                "whole store into memory.",
                stacklevel=2,
            )
        return self._dataset.to_dataframe().reset_index()

    def to_parquet(self, path: str | Path, **kwargs: Any) -> Path:
        """Write the store to a Parquet file (tidy table).

        Args:
            path: Output ``.parquet`` path.
            **kwargs: Forwarded to :meth:`pandas.DataFrame.to_parquet`.

        Returns:
            Path: The written file path.

        Raises:
            OptionalPackageDoesNotExist: When pyarrow is not installed.

        Examples:
            - Write a subset to Parquet::

                >>> store.to_parquet("out.parquet")  # doctest: +SKIP
        """
        import_pyarrow(_PARQUET_INSTALL_HINT)
        path = Path(path)
        self.to_dataframe().to_parquet(str(path), index=False, **kwargs)
        return path

    def to_csv(self, path: str | Path, **kwargs: Any) -> Path:
        """Write the store to a CSV file (tidy table).

        Args:
            path: Output ``.csv`` path.
            **kwargs: Forwarded to :meth:`pandas.DataFrame.to_csv`.

        Returns:
            Path: The written file path.

        Examples:
            - Write a subset to CSV::

                >>> store.to_csv("out.csv")  # doctest: +SKIP
        """
        path = Path(path)
        self.to_dataframe().to_csv(str(path), index=False, **kwargs)
        return path

    def __getitem__(self, key: str) -> Any:
        """Return a variable or coordinate as an :class:`xarray.DataArray`."""
        return self._dataset[key]

    def __contains__(self, key: str) -> bool:
        """True when ``key`` is a variable or coordinate in the store."""
        return key in self._dataset

    def __repr__(self) -> str:
        """Compact, structure-only representation (no data read)."""
        return (
            f"<LabeledDataset sizes={self.sizes} "
            f"coords={self.coordinates} variables={self.variables}>"
        )
