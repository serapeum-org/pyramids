"""``CubeNetCDFWriter`` — streams a datacube into a single multidim NetCDF.

The transient-engine counterpart of :meth:`DatasetCollection.to_netcdf` (compare
:class:`pyramids.netcdf._plot.NetCDFPlot`): a ``CubeNetCDFWriter(collection)`` is
built per write, owns the derived cube-write state (band count / names / dtype)
so the three phases share it as instance attributes instead of threading it, and
runs the phases — time-axis resolution (delegated to
:class:`pyramids.dataset._cube_time.TimeAxis`), schema assembly, and the
one-timestep-at-a-time streaming write.

It lives under ``pyramids.netcdf`` (next to the streaming writer it drives) rather
than in ``pyramids.dataset`` so that ``pyramids.dataset.collection`` imports it
lazily, honouring the ``pyramids.netcdf`` → ``pyramids.dataset.Dataset``
circular-import carveout (see :meth:`DatasetCollection.to_netcdf`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pyramids.base._errors import AlignmentError
from pyramids.dataset._cube_time import TimeAxis
from pyramids.netcdf.engines.interop import open_streaming_multidim_netcdf

if TYPE_CHECKING:
    from pyramids.base._raster_meta import RasterMeta
    from pyramids.dataset.collection import DatasetCollection


class CubeNetCDFWriter:
    """Streams a :class:`DatasetCollection`'s ``(T, B, Y, X)`` cube to a NetCDF.

    Instantiate with the source collection and call :meth:`write`. The whole cube
    is never resident: each timestep is read, cast, and written as a slab, so peak
    memory is a single timestep plus the coordinate axes (ARC-46).

    Attributes:
        band_count: Number of bands (the collection template's band count).
        names: Per-band variable names.
        var_dtype: Dtype every variable is written at (each timestep is cast to it).

    Examples:
        - Drive a write from a collection (the engine behind
          :meth:`DatasetCollection.to_netcdf`; needs a real collection so it is not
          run here):

            ```python
            >>> CubeNetCDFWriter(collection).write(  # doctest: +SKIP
            ...     "cube.nc", time_dim="time", var_per_band=True
            ... )

            ```
    """

    _meta: RasterMeta
    band_count: int
    names: list[str]
    var_dtype: np.dtype

    def __init__(self, collection: DatasetCollection) -> None:
        """Capture the source collection.

        Args:
            collection: The collection whose ``(T, B, Y, X)`` cube is written.
        """
        self._collection = collection

    def write(
        self,
        path: str | Path,
        *,
        time_dim: str = "time",
        time_coords: Sequence[Any] | None = None,
        var_per_band: bool = True,
    ) -> None:
        """Write the collection's cube to a single multidim NetCDF at ``path``.

        Args:
            path: Output ``.nc`` path.
            time_dim: Name of the time dimension.
            time_coords: Explicit time-axis values, or ``None`` to auto-resolve
                (the collection's own time axis, else a positional index).
            var_per_band: One variable per band (``True``), else a single 4-D
                ``data`` variable with a ``band`` coordinate.

        Raises:
            ValueError: When the collection is empty (``time_length == 0``) or
                ``len(time_coords) != time_length``.
            AlignmentError: When a timestep's shape or band count differs from the
                collection template.
            RuntimeError: When the GDAL multidim NetCDF writer fails to write the
                file.
        """
        collection = self._collection
        if collection.time_length == 0:
            raise ValueError(
                "to_netcdf: cannot write an empty collection (time_length == 0)."
            )
        # Resolve (and length-validate) the time axis before deriving the band
        # state, matching the original to_netcdf order so the same error wins when
        # both the axis and the template are in play.
        axis = TimeAxis.resolve(time_coords, collection.time_length, collection.time)
        meta = collection._meta
        self._meta = meta
        self.band_count = int(meta.shape[0])
        self.names = (
            list(meta.band_names)
            if meta.band_names
            else [f"band_{i + 1}" for i in range(self.band_count)]
        )
        self.var_dtype = np.dtype(meta.dtype)

        dims, coords, var_specs, root_attrs = self._build_schema(
            axis, time_dim=time_dim, var_per_band=var_per_band
        )
        with open_streaming_multidim_netcdf(
            path, dims, coords, var_specs, root_attrs, crs_wkt=root_attrs.get("crs_wkt")
        ) as writer:
            self._stream(writer, dims=dims, var_per_band=var_per_band)

    def _build_schema(
        self,
        axis: TimeAxis,
        *,
        time_dim: str,
        var_per_band: bool,
    ) -> tuple[
        dict[str, int],
        dict[str, tuple[np.ndarray, dict[str, Any]]],
        dict[str, tuple[tuple[str, ...], np.dtype | str, dict[str, Any]]],
        dict[str, Any],
    ]:
        """Assemble the ``(dims, coords, var_specs, root_attrs)`` for the writer.

        Reads the collection template (:attr:`_meta`) and base grid for the ``y`` /
        ``x`` coordinate axes, the geobox root attributes, and the typed ``nodata``
        attribute. One variable per band, or a single 4-D ``data`` variable with a
        ``band`` coordinate.

        Args:
            axis: The resolved time axis (values + CF attributes).
            time_dim: Name of the time dimension.
            var_per_band: Emit one variable per band, else a single 4-D ``data``.

        Returns:
            tuple: ``(dims, coords, var_specs, root_attrs)`` ready for
            :func:`open_streaming_multidim_netcdf`.
        """
        meta = self._meta
        band_count = self.band_count
        names = self.names
        var_dtype = self.var_dtype
        nodata = (meta.nodata or (None,))[0]
        y_coord = np.asarray(self._collection._base.y)
        x_coord = np.asarray(self._collection._base.x)

        # The CF ``_FillValue`` is declared via ``SetNoDataValueDouble`` at MDArray
        # creation in ``open_streaming_multidim_netcdf`` (netCDF rejects a fill value
        # once data exists, so it must be set before any slab is streamed) — that is
        # what CF readers mask on. This ``nodata`` attribute is kept in addition — on
        # the root group (matches ``to_zarr``) and on every data variable — so
        # pyramids' own reader recovers the no-data value on round-trip.
        var_attrs: dict[str, Any] = {}
        typed_nodata = None
        if nodata is not None:
            typed_nodata = np.asarray(nodata, dtype=var_dtype).item()
            var_attrs["nodata"] = typed_nodata

        dims: dict[str, int] = {time_dim: int(axis.values.shape[0])}
        coords: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
            time_dim: (axis.values, axis.attrs),
        }
        # Each data variable is created at full shape but written one timestep slab
        # at a time, so the whole (T, B, Y, X) cube is never resident (ARC-46).
        var_specs: dict[str, tuple[tuple[str, ...], np.dtype | str, dict[str, Any]]]
        if var_per_band:
            var_specs = {
                names[i]: ((time_dim, "y", "x"), var_dtype, dict(var_attrs))
                for i in range(band_count)
            }
        else:
            # GDAL's multidim NetCDF writer can't write a string coord, so the band
            # axis carries an integer index and the human names ride along on the
            # root group as a ``band_names`` attribute; a reader recovers them there.
            dims["band"] = band_count
            coords["band"] = (np.arange(band_count), {})
            var_specs = {
                "data": ((time_dim, "band", "y", "x"), var_dtype, dict(var_attrs)),
            }
        dims["y"] = int(y_coord.shape[0])
        dims["x"] = int(x_coord.shape[0])
        coords["y"] = (y_coord, {})
        coords["x"] = (x_coord, {})

        root_attrs: dict[str, Any] = {"Conventions": "CF-1.8"}
        try:
            crs_wkt = meta.crs.to_wkt() if meta.crs is not None else None
        except AttributeError:
            crs_wkt = None
        if crs_wkt:
            root_attrs["crs_wkt"] = crs_wkt
        if meta.epsg is not None:
            root_attrs["epsg"] = int(meta.epsg)
        root_attrs["GeoTransform"] = " ".join(str(v) for v in meta.geotransform)
        if not var_per_band:
            root_attrs["band_names"] = ",".join(names)
        if typed_nodata is not None:
            root_attrs["nodata"] = typed_nodata
        return dims, coords, var_specs, root_attrs

    def _stream(
        self,
        writer: Any,
        *,
        dims: dict[str, int],
        var_per_band: bool,
    ) -> None:
        """Stream each timestep into ``writer`` one slab at a time.

        Reads one dataset at a time (peak memory = a single timestep, not the whole
        cube), normalises each to ``(band, rows, cols)``, and writes it as a slab —
        per band when ``var_per_band`` else as one 4-D ``data`` variable.

        Args:
            writer: The streaming writer yielded by
                :func:`open_streaming_multidim_netcdf`.
            dims: The resolved dimension-length map (supplies ``y`` / ``x``).
            var_per_band: Write one variable per band, else a single 4-D ``data``.

        Raises:
            AlignmentError: When a timestep's ``(band, rows, cols)`` shape differs
                from the collection template ``(band_count, y, x)``.
        """
        collection = self._collection
        band_count = self.band_count
        names = self.names
        var_dtype = self.var_dtype
        expected = (band_count, dims["y"], dims["x"])
        for t, ds in enumerate(collection.datasets):
            block = np.asarray(ds.read_array()).astype(var_dtype, copy=False)
            if block.ndim == 2:
                block = block[np.newaxis, :, :]
            if block.shape != expected:
                where = (
                    collection.files[t]
                    if collection.files and t < len(collection.files)
                    else f"timestep {t}"
                )
                raise AlignmentError(
                    f"to_netcdf: {where} has shape {block.shape}, but the "
                    f"collection template is {expected} (band, rows, cols); "
                    f"every timestep must share the base grid and band count."
                )
            if var_per_band:
                for i in range(band_count):
                    writer.write_slab(names[i], t, block[i])
            else:
                writer.write_slab("data", t, block)
