"""Non-gridded, label-indexed NetCDF / Zarr reading (PY-G / P-A).

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

from pathlib import Path
from typing import Any, Iterable

from pyramids.base._utils import import_xarray

_XARRAY_INSTALL_HINT = (
    "Reading a label-indexed NetCDF/Zarr store needs the optional 'xarray' "
    "dependency. Install it with one of:\n"
    "  - PyPI:  pip install 'pyramids-gis[xarray]'\n"
    "  - conda: conda install -c conda-forge xarray"
)


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
    grid. Use it to inspect a store's structure and (in later tasks) select by
    label, slice in time, subset by bbox, and write out tabular/raster output.

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
        chunks: Any = None,
        engine: str | None = None,
        storage_options: dict[str, Any] | None = None,
        consolidated: bool | None = None,
    ) -> LabeledDataset:
        """Open a label-indexed NetCDF or Zarr store lazily.

        The store kind is inferred from ``path`` (``.zarr`` -> Zarr, otherwise
        NetCDF/HDF5) unless ``engine`` is given. Only metadata is read on open;
        data is materialised lazily.

        Args:
            path: Local path (or, with ``storage_options``, a fsspec URL) of the
                store.
            variables: Restrict to these data variables. ``None`` keeps all.
            group: NetCDF/Zarr group to open. ``None`` opens the root group.
            chunks: xarray ``chunks`` spec. ``None`` (default) reads without
                dask (lazy on access); ``{}`` / ``"auto"`` / a dict backs the
                arrays with dask for chunked reads (needs the ``[lazy]`` extra).
            engine: Force an xarray engine (e.g. ``"zarr"``, ``"h5netcdf"``,
                ``"netcdf4"``). ``None`` lets xarray infer it.
            storage_options: fsspec options for remote stores (e.g.
                ``{"anon": True}`` for an unsigned public bucket).
            consolidated: Zarr only — whether to read consolidated metadata.
                ``None`` lets xarray decide.

        Returns:
            LabeledDataset: The opened store.

        Raises:
            OptionalPackageDoesNotExist: When xarray is not installed.
        """
        xr = import_xarray(_XARRAY_INSTALL_HINT)
        source = str(path)
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
