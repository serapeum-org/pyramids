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

from pyramids.base._domain import is_stored_no_data
from pyramids.base._errors import AlignmentError
from pyramids.base._utils import _is_identity_packing, apply_unpack
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

    CF packing decides the storage (see :meth:`_decide_storage`). When one recipe
    describes every slab a variable holds, the stored counts are written with that
    recipe as `scale_factor` / `add_offset` -- the store is copied. When none does,
    the cube is materialised: physical `float64` values, no recipe, and `NaN` for
    the gaps.

    Attributes:
        band_count: Number of bands (the collection template's band count).
        names: Per-band variable names.
        var_dtype: Dtype every variable is written at (each timestep is cast to it):
            the collection's stored dtype, or `float64` for a materialised cube.

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
        """Write the collection's cube to a single multidim NetCDF at `path`.

        The storage is decided before the schema is built (see `_decide_storage`):
        stored counts under one CF recipe when a single recipe fits every slab, else a
        materialised physical `float64` cube with `NaN` gaps.

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
        self._decide_storage(var_per_band)

        dims, coords, var_specs, root_attrs = self._build_schema(
            axis, time_dim=time_dim, var_per_band=var_per_band
        )
        with open_streaming_multidim_netcdf(
            path, dims, coords, var_specs, root_attrs, crs_wkt=root_attrs.get("crs_wkt")
        ) as writer:
            self._stream(writer, dims=dims, var_per_band=var_per_band)

    def _decide_storage(self, var_per_band: bool) -> None:
        """Choose between writing the counts with their recipe and materialising them.

        The compact form -- stored counts plus `scale_factor` / `add_offset` -- is only
        honest when one recipe describes every slab a variable will hold. That fails in
        two reachable ways: the timesteps are packed differently (each is its own file,
        and nothing makes them agree), or, for the single 4-D `data` variable, the bands
        are. Writing counts under a recipe that does not describe them -- or under none
        -- reads back wrong: bands packed at 0.01 / 0.1 came back 100.0 / 100.0, and a
        second timestep at 0.1 read as if it were 0.01.

        When one recipe does fit, it is taken from the timesteps, not the template, so a
        template packed unlike its timesteps cannot mislabel them. When none fits, the
        cube is materialised: physical `float64`, no recipe, and `NaN` for the gaps, since
        a single declared sentinel cannot name gaps whose physical value differs from one
        timestep to the next.

        Each timestep's recipe is resolved per band through `_effective_packing`, with an
        identity or unusable pair counted as no packing, so an unpacked collection agrees
        with itself and is written as stored.

        The decision is recorded on the writer for `_build_schema` and `_stream`:
        `_materialise` (whether to write physical values), `_recipe` (the per-band
        `(scale, offset)` pairs to declare, or `None` when materialising), and, for a
        materialised cube that holds any packed band, `var_dtype` widened to `float64`.

        Args:
            var_per_band: One variable per band, which can carry a recipe per band;
                otherwise a single 4-D variable, which can carry only one.
        """
        recipes = [
            tuple(
                (None, None) if _is_identity_packing(*pair) else pair
                for pair in (
                    dataset._effective_packing(index)
                    for index in range(self.band_count)
                )
            )
            for dataset in self._collection.datasets
        ]
        agree_across_time = all(recipe == recipes[0] for recipe in recipes)
        agree_across_bands = len(set(recipes[0])) == 1
        self._materialise = not agree_across_time or (
            not var_per_band and not agree_across_bands
        )
        self._recipe = None if self._materialise else list(recipes[0])
        if self._materialise and any(
            pair != (None, None) for recipe in recipes for pair in recipe
        ):
            self.var_dtype = np.dtype("float64")

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
        """Assemble the `(dims, coords, var_specs, root_attrs)` for the writer.

        Reads the collection template (:attr:`_meta`) and base grid for the `y` /
        `x` coordinate axes, the geobox root attributes, and the typed `nodata`
        attribute. One variable per band, or a single 4-D `data` variable with a
        `band` coordinate.

        The storage `_decide_storage` chose shapes the variables. Stored counts get the
        template's declared no-data and the recipe taken from the timesteps as
        `scale_factor` / `add_offset` -- per variable when `var_per_band`, or on the 4-D
        variable when every band shares it. A materialised cube gets no recipe and
        declares `NaN` as its no-data, which is what its gaps hold.

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
        materialise = getattr(self, "_materialise", False)
        # A materialised cube holds physical values with its gaps as NaN (see
        # `_decide_storage`), so NaN is the fill it declares.
        nodata = np.nan if materialise else (meta.nodata or (None,))[0]
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

        # Unless it is materialised, the cube is written at the collection's *stored*
        # dtype, so the slabs are streamed with `unpack=False` and the packing recipe has
        # to travel with them as CF attributes (GDAL keeps `scale_factor` / `add_offset`
        # in the MDArray's own slots, and lifts these two into them on the next read).
        # Writing physical values instead would cast `14.35` back into an `int16` cube as
        # `14`, and writing counts without the recipe would leave them meaning nothing.
        # A materialised cube holds physical values, so it declares no recipe at all.
        base = self._collection._base
        recipe = getattr(self, "_recipe", None)
        if materialise:
            packing = [(None, None)] * band_count
        elif recipe is not None:
            packing = recipe
        else:
            packing = [base._effective_packing(index) for index in range(band_count)]

        def _packing_attrs(index: int) -> dict[str, Any]:
            """The CF packing attributes for one band, empty when it declares none.

            Per band, because a collection may carry a different factor on each and
            stamping band 1's recipe onto all of them mislabels the rest. `var_per_band`
            can honour that; the single 4-D `data` variable cannot, so it takes the
            packing only when every band agrees.

            Args:
                index: Zero-based band index.

            Returns:
                dict: `scale_factor` / `add_offset`, or empty for an unpacked band.
            """
            scale, offset = packing[index]
            if _is_identity_packing(scale, offset):
                return {}
            attrs: dict[str, Any] = {}
            if scale is not None:
                attrs["scale_factor"] = scale
            if offset is not None:
                attrs["add_offset"] = offset
            return attrs

        shared_packing = _packing_attrs(0) if len(set(packing)) == 1 else {}

        dims: dict[str, int] = {time_dim: int(axis.values.shape[0])}
        coords: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
            time_dim: (axis.values, axis.attrs),
        }
        # Each data variable is created at full shape but written one timestep slab
        # at a time, so the whole (T, B, Y, X) cube is never resident (ARC-46).
        var_specs: dict[str, tuple[tuple[str, ...], np.dtype | str, dict[str, Any]]]
        if var_per_band:
            var_specs = {
                names[i]: (
                    (time_dim, "y", "x"),
                    var_dtype,
                    {**var_attrs, **_packing_attrs(i)},
                )
                for i in range(band_count)
            }
        else:
            # GDAL's multidim NetCDF writer can't write a string coord, so the band
            # axis carries an integer index and the human names ride along on the
            # root group as a ``band_names`` attribute; a reader recovers them there.
            dims["band"] = band_count
            coords["band"] = (np.arange(band_count), {})
            var_specs = {
                "data": (
                    (time_dim, "band", "y", "x"),
                    var_dtype,
                    {**var_attrs, **shared_packing},
                ),
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

    @staticmethod
    def _physical_block(dataset: Any) -> np.ndarray:
        """One timestep in physical units, with its own recipe and its gaps as NaN.

        Used when no single recipe describes the whole cube (see `_decide_storage`). Each
        band is masked against its own stored sentinel -- where the sentinel lives --
        then unpacked with that timestep's own recipe (`_effective_packing`), so
        timesteps packed differently all land in the same physical units. A band that
        declares no sentinel has no gaps; an unpacked band keeps its values.

        Args:
            dataset: The timestep's `Dataset`.

        Returns:
            np.ndarray: `(bands, rows, cols)` in `float64`, NaN wherever the source had
                no data.

        Examples:
            - Each band is unpacked with its own recipe and its gap becomes `NaN`:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from pyramids.netcdf._cube_netcdf_writer import CubeNetCDFWriter
                >>> step = Dataset.from_array(
                ...     np.array([[[100, -9999]], [[100, 200]]], dtype="int16"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ...     no_data_value=-9999,
                ... )
                >>> step.scale = [0.5, 0.25]
                >>> CubeNetCDFWriter._physical_block(step).tolist()
                [[[50.0, nan]], [[25.0, 50.0]]]

                ```
        """
        raw = np.asarray(dataset.read_array(unpack=False))
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]
        declared = dataset.no_data_value
        out = np.empty(raw.shape, dtype=np.float64)
        for index in range(raw.shape[0]):
            values = np.asarray(
                apply_unpack(raw[index], *dataset._effective_packing(index)),
                dtype=np.float64,
            )
            values[is_stored_no_data(raw[index], declared[index])] = np.nan
            out[index] = values
        return out

    def _stream(
        self,
        writer: Any,
        *,
        dims: dict[str, int],
        var_per_band: bool,
    ) -> None:
        """Stream each timestep into `writer` one slab at a time.

        Reads one dataset at a time (peak memory = a single timestep, not the whole
        cube), normalises each to `(band, rows, cols)`, and writes it as a slab —
        per band when `var_per_band` else as one 4-D `data` variable. A timestep is
        read as its stored counts (`unpack=False`) for a compact cube, or through
        `_physical_block` for a materialised one, and cast to `var_dtype` either way.

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
        materialise = getattr(self, "_materialise", False)
        for t, ds in enumerate(collection.datasets):
            raw = (
                self._physical_block(ds)
                if materialise
                else np.asarray(ds.read_array(unpack=False))
            )
            block = raw.astype(var_dtype, copy=False)
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
