"""COG engine.

Owns the COG family of operations on a Dataset. Accessed as
``ds.cog``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.cog.<method>(...)`` are equivalent.
"""

from __future__ import annotations

import math
import uuid
import warnings
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from osgeo import gdal
from pyproj import Transformer

from pyramids._io import read_vsi_bytes, silent_unlink
from pyramids.base._errors import FailedToSaveError, OutOfBoundsError
from pyramids.base._utils import (
    default_cog_overview_resampling,
    is_integer_gdal_dtype,
    numpy_to_gdal_dtype,
    resolve_cog_predictor,
)
from pyramids.dataset.cog import (
    COGInfo,
    ValidationReport,
    cog_info,
    merge_options,
    profile_options,
    translate_to_cog,
    validate,
    validate_blocksize,
    validate_profile,
)
from pyramids.dataset.cog.validate import _resolve_read_config, config_context
from pyramids.dataset.engines._base import _Engine

if TYPE_CHECKING:
    from pyramids.dataset.dataset import (  # noqa: F401  (forward ref in _Engine["Dataset"])
        Dataset,
    )

_AVERAGING_RESAMPLERS: frozenset[str] = frozenset(
    {"average", "bilinear", "cubic", "cubicspline", "lanczos"}
)


_RESAMPLING_ALG: dict[str, int] = {
    "nearest": gdal.GRIORA_NearestNeighbour,
    "nearest neighbor": gdal.GRIORA_NearestNeighbour,
    "bilinear": gdal.GRIORA_Bilinear,
    "cubic": gdal.GRIORA_Cubic,
    "cubicspline": gdal.GRIORA_CubicSpline,
    "cubic_spline": gdal.GRIORA_CubicSpline,
    "lanczos": gdal.GRIORA_Lanczos,
    "average": gdal.GRIORA_Average,
    "mode": gdal.GRIORA_Mode,
    **({"gauss": gdal.GRIORA_Gauss} if hasattr(gdal, "GRIORA_Gauss") else {}),
    **({"rms": gdal.GRIORA_RMS} if hasattr(gdal, "GRIORA_RMS") else {}),
}
"""Map a resampling name to its GDAL ``GRIORA_*`` decimated-read algorithm.

The names mirror :data:`pyramids.base._utils.INTERPOLATION_METHODS` where the
two algorithm families overlap (``cubic_spline`` is accepted alongside the
historical ``cubicspline``); ``gauss`` / ``rms`` are guarded for older GDAL.
"""


def _resolve_read_resampling(resampling: str) -> int:
    """Resolve a decimated-read resampling name to its ``GRIORA_*`` constant.

    Normalises case and surrounding whitespace before the lookup, mirroring
    :func:`pyramids.base._utils.resolve_resampling` for the warp family.

    Args:
        resampling: Method name, case-insensitive (a key of
            :data:`_RESAMPLING_ALG`).

    Returns:
        int: The matching ``gdal.GRIORA_*`` constant.

    Raises:
        TypeError: ``resampling`` is not a string.
        ValueError: ``resampling`` does not name a registered algorithm; the
            message lists the valid names.
    """
    if not isinstance(resampling, str):
        raise TypeError(
            f"resampling method must be a string, got {type(resampling).__name__}."
        )
    key = resampling.lower().strip()
    if key not in _RESAMPLING_ALG:
        raise ValueError(
            f"unknown resampling {resampling!r}; choose from {sorted(_RESAMPLING_ALG)}"
        )
    return _RESAMPLING_ALG[key]


_PALETTE_GDAL_DTYPES: frozenset[int] = frozenset({gdal.GDT_Byte, gdal.GDT_UInt16})
"""GDAL dtypes for which a colour table (palette) is meaningful."""


_WEB_MERCATOR_HALF_EXTENT: float = 20037508.342789244
"""Half the Web-Mercator (EPSG:3857) world extent in metres."""


def _xyz_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return the EPSG:3857 bounds of an XYZ/slippy-map tile.

    Args:
        z: Zoom level (>= 0).
        x: Tile column index in ``[0, 2**z)``.
        y: Tile row index in ``[0, 2**z)`` (origin at the top-left / north-west).

    Returns:
        ``(west, south, east, north)`` in EPSG:3857 metres.

    Examples:
        - The single tile at zoom 0 spans the whole Web-Mercator world:
            ```python
            >>> w, s, e, n = _xyz_bounds_3857(0, 0, 0)
            >>> round(w), round(s), round(e), round(n)
            (-20037508, -20037508, 20037508, 20037508)

            ```
        - Tile (0, 0) at zoom 1 is the north-west quadrant:
            ```python
            >>> w, s, e, n = _xyz_bounds_3857(1, 0, 0)
            >>> round(w), round(n)
            (-20037508, 20037508)
            >>> round(e), round(s)
            (0, 0)

            ```
    """
    r = _WEB_MERCATOR_HALF_EXTENT
    n_tiles = 2**z
    span = (2 * r) / n_tiles
    west = -r + x * span
    east = -r + (x + 1) * span
    north = r - y * span
    south = r - (y + 1) * span
    return west, south, east, north


@lru_cache(maxsize=64)
def _cached_transformer(src_crs: Any, dst_crs: Any) -> Transformer:
    """Return a cached `pyproj.Transformer` for a CRS pair.

    Building a transformer parses both CRS definitions and resolves a
    transformation pipeline, which is far more expensive than using it. The COG
    read paths rebuilt one per tile and per point, so a `read_tile` loop or a
    batch of `point` lookups paid that cost on every call.

    Args:
        src_crs: Source CRS -- anything `Transformer.from_crs` accepts, as long
            as it is hashable (an EPSG int or a WKT string).
        dst_crs: Destination CRS, same forms.

    Returns:
        Transformer: A shared, reusable transformer. `pyproj` transformers are
        safe to reuse; only construction is costly.
    """
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


class COG(_Engine["Dataset"]):
    """Cloud Optimized GeoTIFF read/write/validate operations for `Dataset`.

    Owns the real implementations of `to_cog`, `is_cog` (property),
    and `validate_cog`. `Dataset` exposes a same-named facade for each
    so `ds.to_cog(...)` and `ds.cog.to_cog(...)` are equivalent.

    `to_cog` is the **single owner of COG write policy**: it applies the
    house defaults, resolves the dtype-aware predictor and overview
    resampling, and runs the `STATISTICS` retry. The
    :func:`pyramids.dataset.cog.write_cog` facade is a thin delegator that
    only normalises its input and forwards overrides here, so both entry
    points produce identical output for identical input. The
    categorical-raster resampling guardrail
    (`_warn_if_categorical_with_averaging`) lives here too.
    """

    def to_cog(
        self,
        path: str | Path,
        *,
        profile: str | None = None,
        compress: str | None = None,
        level: int | None = None,
        quality: int | None = None,
        blocksize: int = 512,
        predictor: str | int | None = None,
        bigtiff: str = "IF_SAFER",
        num_threads: int | str = "ALL_CPUS",
        overview_resampling: str | None = None,
        overview_count: int | None = None,
        overview_compress: str | None = None,
        tiling_scheme: str | None = None,
        zoom_level: int | None = None,
        zoom_level_strategy: str = "auto",
        aligned_levels: int | None = None,
        resampling: str = "nearest",
        add_mask: bool = False,
        sparse_ok: bool = False,
        target_srs: int | str | None = None,
        statistics: bool = True,
        indexes: list[int] | None = None,
        out_dtype: str | None = None,
        nodata: float | int | None = None,
        band_tags: dict[int, dict[str, Any]] | None = None,
        colormap: dict[int, tuple[int, int, int, int]] | None = None,
        metadata: dict[str, Any] | None = None,
        config: dict[str, str] | None = None,
        extra: Mapping[str, Any] | list[str] | None = None,
    ) -> Path:
        """Save the dataset as a Cloud Optimized GeoTIFF.

        Args:
            path: Destination path. Parent directory must exist.
            profile: Named compression preset (case-insensitive) — one of
                `deflate`, `zstd`, `lzw`, `packbits`, `jpeg`, `webp`,
                `lerc`, `lerc_deflate`, `lerc_zstd`, `raw`. Seeds the
                compression options; explicit `compress`/`level`/`quality`
                and `extra` override it. `jpeg`/`webp` enforce dtype/band
                constraints (Byte; 1-3 / 3-4 bands).
            compress: Compression method. `DEFLATE`, `LZW`, and
                `NONE` are guaranteed by every GDAL build. `JPEG`
                is almost always available. `ZSTD`, `WEBP`,
                `LERC`, `LERC_DEFLATE`, and `LERC_ZSTD` require
                the GDAL build to have been compiled with the
                corresponding library (libzstd / libwebp / LERC); on
                a GDAL build lacking them, the COG driver will raise
                at write time. To probe what your GDAL supports:

                ```python
                from osgeo import gdal
                meta = gdal.GetDriverByName("GTiff").GetMetadataItem(
                    "DMD_CREATIONOPTIONLIST"
                )
                print("ZSTD" in (meta or ""))
                ```
            level: Compression level (e.g., 1-12 for DEFLATE, 1-22 ZSTD).
            quality: Lossy-compression quality 1-100 (JPEG/WEBP).
            blocksize: Internal tile size; power of 2 in [64, 4096].
            predictor: `"YES"`/`"STANDARD"`/`"FLOATING_POINT"` or 1/2/3.
                Defaults to `None`, which auto-resolves per the source
                dtype: `2` (horizontal differencing) for integer rasters,
                `3` (floating-point predictor) for float rasters. Pass an
                explicit value to override.
            bigtiff: `"IF_SAFER"` (default), `"YES"`, `"NO"`,
                `"IF_NEEDED"`.
            num_threads: Worker threads; `"ALL_CPUS"` or an int.
            overview_resampling: `nearest`, `average`, `bilinear`,
                `cubic`, `cubicspline`, `lanczos`, `mode`,
                `rms`, `gauss`. Defaults to `None`, which auto-resolves
                per the source dtype: `mode` for categorical sources
                (integer dtype or a colour table) and `average` for
                continuous (float) sources. The categorical guardrail
                warns only when *you* explicitly pass an averaging method
                on categorical data — never for this auto-resolved default.
            overview_count: Number of overview levels (default: auto).
            overview_compress: Compression for overview IFDs.
            tiling_scheme: e.g., `"GoogleMapsCompatible"` for a
                web-optimized COG (EPSG:3857).
            zoom_level: Advanced tiling-scheme knob: pin the maximum zoom level.
            zoom_level_strategy: Advanced tiling-scheme knob: `auto` (default),
                `lower`, or `upper` zoom-level selection.
            aligned_levels: Advanced tiling-scheme knob: number of overview
                levels aligned to the tiling scheme.
            resampling: Warp resampling when `tiling_scheme` or
                `target_srs` reprojects.
            add_mask: Add an alpha band for transparency.
            sparse_ok: Allow sparse (unfilled) tiles.
            target_srs: Reproject before write. Int for EPSG or a WKT
                / PROJ string.
            statistics: Compute and embed band statistics.
            indexes: 0-based band indices to keep, in order (e.g. `[3, 2, 1]`
                to select and reorder bands). `None` keeps all bands. When
                set, the source is pre-processed through an in-memory
                `gdal.Translate` before the COG write.
            out_dtype: Output NumPy dtype name to cast to (e.g. `"uint8"`,
                `"int16"`). `None` keeps the source dtype. The dtype-aware
                predictor is resolved from the *post-cast* dtype.
            nodata: NoData value to set on the output. `None` keeps the
                source NoData.
            band_tags: Per-band metadata to stamp onto the output, keyed by
                0-based band index, e.g. `{0: {"name": "NDVI"}}`. Useful when
                the source is a bare array/DataArray that carries no band
                descriptions.
            colormap: Palette to attach to band 1, mapping pixel value to an
                `(R, G, B, A)` tuple, e.g. `{0: (0, 0, 0, 255), 1: (255, 0, 0, 255)}`.
            metadata: Dataset-level metadata items to stamp onto the output.
            config: GDAL config options (e.g. `{"GDAL_NUM_THREADS": "4"}`)
                applied via `gdal.config_options` for the duration of the
                write. `None` (default) applies no extra config.
            extra: Additional GDAL creation options as a mapping or
                legacy `['KEY=VALUE',...]` list. Overrides
                conflicting kwargs.

        Returns:
            Path: The resolved destination path.

        Raises:
            ValueError: Invalid blocksize or unknown option key.
            FileNotFoundError: Parent directory does not exist.
            FailedToSaveError: GDAL CreateCopy failed.
            DriverNotExistError: GDAL build lacks the COG driver.

        Warnings:
            UserWarning: When the source looks categorical (integer
                dtype or has a color table) and `overview_resampling`
                is an averaging method.

        Note:
            Setting `tiling_scheme` (e.g., `GoogleMapsCompatible`)
            implies a specific SRS — `target_srs` is ignored in that
            case. A `UserWarning` is emitted if both are provided.

        Note:
            **Larger-than-RAM / parallel writes.** The GDAL COG driver does the
            two-pass overview layout internally and *streams* from the source
            dataset, so a raster bigger than RAM can be COG-encoded as long as
            the source is **on-disk** (or a `/vsi*` file) rather than a fully
            in-RAM array — anchor a MEM dataset with `to_file(path)` first if
            needed. There is no truly dask-parallel COG writer yet:
            `to_file(compute=False)` returns a `dask.delayed` that wraps the
            *synchronous* GDAL write (GeoTIFF writes are serialised by GDAL's
            own file lock), so it defers *scheduling*, not memory or per-tile
            parallelism. For parallel cloud writes use a Zarr-backed output.

        Examples:
            - Write a compressed COG from an in-memory Dataset:
                ```python
                >>> import numpy as np  # doctest: +SKIP
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> arr = np.random.rand(256, 256).astype("float32")  # doctest: +SKIP
                >>> ds = Dataset.create_from_array(  # doctest: +SKIP
                ...     arr, top_left_corner=(0, 0), cell_size=0.001, epsg=4326,
                ... )
                >>> out = ds.to_cog("out.tif", compress="ZSTD")  # doctest: +SKIP
                >>> out.name  # doctest: +SKIP
                'out.tif'

                ```
            - Produce a web-optimized COG for a tile server:
                ```python
                >>> web = ds.to_cog("web.tif", tiling_scheme="GoogleMapsCompatible")  # doctest: +SKIP
                >>> reopened = Dataset.read_file(web)  # doctest: +SKIP
                >>> reopened.epsg  # doctest: +SKIP
                3857

                ```
            - Forward additional GDAL options through `extra`:
                ```python
                >>> _ = ds.to_cog(  # doctest: +SKIP
                ...     "precise.tif",
                ...     compress="LERC",
                ...     extra={"MAX_Z_ERROR": 0.001},
                ... )

                ```
        """
        validate_blocksize(blocksize)
        if tiling_scheme is not None and target_srs is not None:
            warnings.warn(
                "Both tiling_scheme and target_srs provided; "
                "tiling_scheme wins and target_srs is ignored.",
                UserWarning,
                stacklevel=2,
            )
            target_srs = None

        # Build the effective source (PB-4): when band-subsetting, casting the
        # dtype, or (re)setting NoData, pre-process through an in-memory
        # gdal.Translate so the predictor/overview policy below — and the COG
        # write itself — see the *output* bands, not the original source.
        source_ds, source_band0 = self._effective_source(
            indexes, out_dtype, nodata, band_tags, colormap, metadata
        )

        # Resolve a named profile (PB-5): it seeds the compression options;
        # explicit kwargs and `extra` override it. jpeg/webp enforce dtype/band
        # constraints against the *effective* source.
        eff_compress, eff_level, eff_quality, profile_extra = self._resolve_compression(
            profile, compress, level, quality, source_ds, source_band0
        )

        # Single house policy lives here (ARC-1): `to_cog` resolves the
        # dtype-dependent defaults so a direct `ds.to_cog(...)` and the
        # `write_cog(...)` facade — which now just delegates here — produce
        # identical output for identical input.
        if predictor is None:
            # Per-dtype predictor (ARC-2): 2 for integer, 3 for float. GeoTIFF
            # bands share a dtype, so band 0 decides for the whole file. Pass an
            # explicit `predictor=` to override for an (atypical) mixed source.
            predictor = resolve_cog_predictor(source_band0.DataType)
        caller_chose_resampling = overview_resampling is not None
        if overview_resampling is None:
            # Category-safe default (ARC-3): `mode` for integer/colour-table
            # sources, `average` for continuous. Chosen so the default never
            # corrupts categorical rasters and never trips the guardrail below.
            overview_resampling = default_cog_overview_resampling(
                source_band0.DataType, source_band0.GetColorTable() is not None
            )
        if caller_chose_resampling:
            # Only warn when the *caller* explicitly asked for an averaging
            # resampler on categorical data — never for a default we picked.
            self._warn_if_categorical_with_averaging(
                overview_resampling, band=source_band0
            )

        num_threads_str = (
            num_threads if isinstance(num_threads, str) else str(num_threads)
        )
        defaults: dict[str, Any] = {
            "COMPRESS": eff_compress,
            "LEVEL": eff_level,
            "QUALITY": eff_quality,
            **profile_extra,
            "BLOCKSIZE": blocksize,
            "PREDICTOR": predictor,
            "BIGTIFF": bigtiff,
            "NUM_THREADS": num_threads_str,
            "OVERVIEW_RESAMPLING": overview_resampling,
            "OVERVIEW_COUNT": overview_count,
            "OVERVIEW_COMPRESS": overview_compress,
            "TILING_SCHEME": tiling_scheme,
            "ZOOM_LEVEL": zoom_level,
            "ZOOM_LEVEL_STRATEGY": zoom_level_strategy,
            "ALIGNED_LEVELS": aligned_levels,
            "WARP_RESAMPLING": (resampling if (tiling_scheme or target_srs) else None),
            "ADD_ALPHA": True if add_mask else None,
            "SPARSE_OK": True if sparse_ok else None,
            "STATISTICS": "YES" if statistics else None,
        }
        if target_srs is not None:
            defaults["TARGET_SRS"] = (
                f"EPSG:{target_srs}" if isinstance(target_srs, int) else target_srs
            )

        options = merge_options(defaults, extra)
        with config_context(config):
            self._translate_with_statistics_retry(path, options, src=source_ds)
        return Path(path)

    @staticmethod
    def _resolve_compression(
        profile: str | None,
        compress: str | None,
        level: int | None,
        quality: int | None,
        source_ds: gdal.Dataset,
        source_band0: Any,
    ) -> tuple[str, int | None, int | None, dict[str, Any]]:
        """Resolve effective compression options, seeding from a named profile.

        A named `profile` supplies default COMPRESS/LEVEL/QUALITY plus any extra
        profile options; explicit `compress`/`level`/`quality` kwargs override the
        seeded defaults. `jpeg`/`webp` profiles validate dtype/band constraints
        against the effective source before seeding.

        Args:
            profile: Named compression preset, or None.
            compress: Explicit compression method override, or None.
            level: Explicit compression level override, or None.
            quality: Explicit lossy quality override, or None.
            source_ds: Effective source dataset (for band-count validation).
            source_band0: Band 0 of the effective source (for dtype validation).

        Returns:
            `(eff_compress, eff_level, eff_quality, profile_extra)` where
            `profile_extra` holds profile options other than COMPRESS/LEVEL/QUALITY.
        """
        profile_opts: dict[str, Any] = {}
        if profile is not None:
            validate_profile(
                profile,
                gdal.GetDataTypeName(source_band0.DataType),
                source_ds.RasterCount,
            )
            profile_opts = profile_options(profile)
        eff_compress = (
            compress
            if compress is not None
            else profile_opts.get("COMPRESS", "DEFLATE")
        )
        eff_level = level if level is not None else profile_opts.get("LEVEL")
        eff_quality = quality if quality is not None else profile_opts.get("QUALITY")
        profile_extra = {
            k: v
            for k, v in profile_opts.items()
            if k not in ("COMPRESS", "LEVEL", "QUALITY")
        }
        return eff_compress, eff_level, eff_quality, profile_extra

    def _effective_source(
        self,
        indexes: list[int] | None,
        out_dtype: str | None,
        nodata: float | int | None,
        band_tags: dict[int, dict[str, Any]] | None = None,
        colormap: dict[int, tuple[int, int, int, int]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[gdal.Dataset, Any]:
        """Return the source dataset (optionally pre-processed) and its band 0.

        When band-subsetting / dtype-casting / setting NoData (PB-4) the backing
        raster is run through an in-memory ``gdal.Translate``; when stamping
        band tags / colourmap / metadata (PC-2) it is copied to a MEM dataset
        first so the user's open dataset is **never mutated**. With neither, the
        backing raster is returned unchanged.

        Args:
            indexes: 0-based band indices to keep/reorder, or ``None``.
            out_dtype: Output NumPy dtype name to cast to, or ``None``.
            nodata: NoData value to set, or ``None``.
            band_tags: Per-band metadata keyed by 0-based band index, or ``None``.
            colormap: Palette for band 1 (value -> RGBA), or ``None``.
            metadata: Dataset-level metadata, or ``None``.

        Returns:
            A ``(dataset, band1)`` tuple where ``band1`` is GDAL band 1 of the
            returned dataset (used for predictor/resampling resolution).
        """
        # The COG write (and the gdal.Translate pre-process below) do tiled reads of the source; a
        # NetCDF multidim view can't be window-read by GDAL >= 3.13, so materialise it first (no-op
        # for an ordinary raster).
        self._ds._materialize_md_view()
        needs_translate = (
            indexes is not None or out_dtype is not None or nodata is not None
        )
        needs_stamp = bool(band_tags or colormap or metadata)
        if not needs_translate and not needs_stamp:
            ds = self._ds._raster
            return ds, ds.GetRasterBand(1)

        if needs_translate:
            translate_kwargs: dict[str, Any] = {}
            if indexes is not None:
                # pyramids band indices are 0-based; GDAL bandList is 1-based.
                translate_kwargs["bandList"] = [i + 1 for i in indexes]
            if out_dtype is not None:
                translate_kwargs["outputType"] = numpy_to_gdal_dtype(out_dtype)
            if nodata is not None:
                translate_kwargs["noData"] = nodata
            mem = gdal.Translate("", self._ds._raster, format="MEM", **translate_kwargs)
        else:
            # Stamp-only: copy so the user's dataset is not mutated.
            mem = gdal.GetDriverByName("MEM").CreateCopy("", self._ds._raster)
        if mem is None:
            raise FailedToSaveError(
                "failed to build the pre-processed COG source "
                f"(indexes={indexes}, out_dtype={out_dtype}, nodata={nodata})"
            )
        if needs_stamp:
            self._stamp_metadata(mem, band_tags, colormap, metadata)
        return mem, mem.GetRasterBand(1)

    @staticmethod
    def _stamp_metadata(
        ds: gdal.Dataset,
        band_tags: dict[int, dict[str, Any]] | None,
        colormap: dict[int, tuple[int, int, int, int]] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Stamp band tags / colourmap / dataset metadata onto a MEM dataset.

        Args:
            ds: The (copied) dataset to mutate.
            band_tags: Per-band metadata keyed by 0-based band index.
            colormap: Palette for band 1 (value -> RGBA tuple). GeoTIFF only
                supports a colour table on a single-band `Byte` / `UInt16`
                raster; a `ValueError` is raised up-front for other dtypes
                (GDAL would otherwise fail deep in `CreateCopy`).
            metadata: Dataset-level metadata items.

        Raises:
            ValueError: When `colormap` is applied to a band whose dtype is
                not `Byte`/`UInt16`.
        """
        if metadata:
            ds.SetMetadata({str(k): str(v) for k, v in metadata.items()})
        if colormap:
            band = ds.GetRasterBand(1)
            if band.DataType not in _PALETTE_GDAL_DTYPES:
                raise ValueError(
                    f"colormap is only supported on Byte/UInt16 rasters; got "
                    f"{gdal.GetDataTypeName(band.DataType)}. Cast first with "
                    f"to_cog(..., out_dtype='uint8'), or drop the colormap."
                )
            color_table = gdal.ColorTable()
            for value, rgba in colormap.items():
                color_table.SetColorEntry(int(value), tuple(rgba))
            band.SetColorTable(color_table)
            band.SetColorInterpretation(gdal.GCI_PaletteIndex)
        if band_tags:
            for index, tags in band_tags.items():
                # 0-based index -> GDAL 1-based band number.
                ds.GetRasterBand(index + 1).SetMetadata(
                    {str(k): str(v) for k, v in tags.items()}
                )

    def to_cog_bytes(self, **kwargs: Any) -> bytes:
        """Encode the dataset as a COG and return the file contents as bytes.

        Writes the COG to an in-memory GDAL ``/vsimem/`` file (no temp file on
        disk), reads the bytes back, and unlinks the virtual file. Useful for
        uploading a COG directly to an object store (S3 / GCS / Azure) without
        touching the local filesystem.

        Args:
            **kwargs: Forwarded verbatim to :meth:`to_cog` (e.g. ``compress``,
                ``blocksize``, ``predictor``, ``extra``). The same house
                defaults and dtype-aware resolution apply.

        Returns:
            bytes: The complete COG file contents.

        Raises:
            FailedToSaveError: GDAL failed to encode the COG.

        Examples:
            - Encode an in-memory Dataset to COG bytes and upload them:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("scene.tif")  # doctest: +SKIP
                >>> blob = ds.to_cog_bytes(compress="ZSTD")  # doctest: +SKIP
                >>> len(blob) > 0  # doctest: +SKIP
                True
                >>> blob[:2] in (b"II", b"MM")  # TIFF byte-order marker  # doctest: +SKIP
                True

                ```
        """
        vsi_path = f"/vsimem/{uuid.uuid4().hex}.tif"
        try:
            self.to_cog(vsi_path, **kwargs)
            try:
                data = read_vsi_bytes(vsi_path)
            except FileNotFoundError as exc:
                raise FailedToSaveError(
                    f"could not reopen in-memory COG at {vsi_path}"
                ) from exc
        finally:
            # silent_unlink: when to_cog fails before creating the file, a
            # plain gdal.Unlink raises under gdal.UseExceptions() and masks
            # the original exception. Sweep the PAM sidecar too.
            silent_unlink(vsi_path)
            silent_unlink(f"{vsi_path}.aux.xml")
        return data

    def _translate_with_statistics_retry(
        self,
        path: str | Path,
        options: dict[str, Any],
        src: gdal.Dataset | None = None,
    ) -> None:
        """Write the COG, retrying once without STATISTICS on the known failure.

        Some GDAL builds abort the ``STATISTICS=YES`` sampling pass on float
        on-disk sources with "no valid pixels found in sampling". The COG
        itself is fine without embedded statistics, so on that specific error
        we retry once with ``STATISTICS`` dropped. Lives here (ARC-4) rather
        than in the :func:`write_cog` facade so a direct ``ds.to_cog(...)`` is
        equally robust.

        Args:
            path: Destination file path.
            options: Fully-merged COG creation options.
            src: Source :class:`gdal.Dataset` to encode. Defaults to the
                backing raster; a pre-processed in-memory dataset is passed
                when band-subsetting / casting / setting NoData (PB-4).
        """
        if src is None:
            # COG CreateCopy does tiled reads of the source; a NetCDF multidim view can't be
            # window-read by GDAL >= 3.13, so materialise it first (no-op for an ordinary raster).
            self._ds._materialize_md_view()
        source = self._ds._raster if src is None else src

        def _run(opts: dict[str, Any]) -> None:
            dst: gdal.Dataset | None = None
            try:
                dst = translate_to_cog(source, path, opts)
                dst.FlushCache()
            finally:
                dst = None

        try:
            _run(options)
        except (RuntimeError, FailedToSaveError) as exc:
            # translate_to_cog wraps CreateCopy RuntimeErrors into
            # FailedToSaveError; a deferred STATISTICS failure at FlushCache
            # time surfaces as a raw RuntimeError — catch both.
            statistics_on = str(options.get("STATISTICS", "")).upper() in (
                "YES",
                "TRUE",
            )
            if statistics_on and "valid pixels" in str(exc).lower():
                retry = {k: v for k, v in options.items() if k != "STATISTICS"}
                _run(retry)
            else:
                raise

    @property
    def is_cog(self) -> bool:
        """`True` iff the backing file on disk is a valid COG.

        `False` for MEM datasets, `/vsimem/` paths, and unsaved
        datasets (empty :attr:`file_name`).

        Examples:
            - Check the backing file of a newly-opened COG:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("scene.tif")  # doctest: +SKIP
                >>> ds.is_cog  # doctest: +SKIP
                True

                ```
            - Plain GeoTIFFs and MEM datasets return False:
                ```python
                >>> plain = Dataset.read_file("plain.tif")  # doctest: +SKIP
                >>> plain.is_cog  # doctest: +SKIP
                False

                ```
            - Use in a conditional pipeline:
                ```python
                >>> if not ds.is_cog:  # doctest: +SKIP
                ...     ds.to_cog("fixed.tif")

                ```
        """
        result: bool
        fn = self._on_disk_path()
        if fn is None:
            result = False
        else:
            result = self._is_cog_cheap(fn)
        return result

    @staticmethod
    def _is_cog_cheap(path: str) -> bool:
        """Fast, metadata-only heuristic for "is this file a COG?" (ARC-7).

        Avoids the full COG validator on every `is_cog` access (which reads the
        whole IFD/offset table — costly over `/vsicurl`). Checks: GTiff driver,
        no external `.ovr` sidecar, internally tiled (square blocks or a single
        tile), and internal overviews present when the image is larger than one
        tile. This can FALSE-POSITIVE on a tiled GeoTIFF that is not laid out in
        strict COG order — use :meth:`validate_cog` for the authoritative check.

        Args:
            path: On-disk or remote `/vsi*` path.

        Returns:
            bool: `True` when the file looks like a COG by the cheap heuristic.
        """
        cfg = _resolve_read_config(path, None)
        with config_context(cfg):
            try:
                ds = gdal.Open(path)
            except RuntimeError:
                return False
            if ds is None:
                return False
            try:
                if ds.GetDriver().ShortName != "GTiff":
                    return False
                files = ds.GetFileList() or []
                if any(str(f).lower().endswith(".ovr") for f in files):
                    return False
                band = ds.GetRasterBand(1)
                block_x, block_y = band.GetBlockSize()
                width, height = ds.RasterXSize, ds.RasterYSize
                single_tile = block_x >= width and block_y >= height
                tiled = block_x == block_y or single_tile
                if not tiled:
                    return False
                needs_overviews = max(width, height) > max(block_x, block_y)
                if needs_overviews and band.GetOverviewCount() == 0:
                    return False
                return True
            finally:
                ds = None

    def validate_cog(
        self, strict: bool = False, config: dict[str, str] | None = None
    ) -> ValidationReport:
        """Validate the backing file as a COG.

        Args:
            strict: If `True`, warnings are treated as errors.
            config: GDAL config options for the read; defaults to the remote
                read tuning for `/vsicurl` paths (see
                :func:`pyramids.dataset.cog.validate.validate`).

        Returns:
            ValidationReport with errors, warnings, and structural details.

        Raises:
            FileNotFoundError: Dataset has no on-disk backing file
                (MEM-only or `/vsimem/`).

        Examples:
            - Validate and branch on the result:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("scene.tif")  # doctest: +SKIP
                >>> report = ds.validate_cog()  # doctest: +SKIP
                >>> bool(report)  # doctest: +SKIP
                True

                ```
            - Strict mode promotes warnings to errors:
                ```python
                >>> strict = ds.validate_cog(strict=True)  # doctest: +SKIP
                >>> if not strict:  # doctest: +SKIP
                ...     for err in strict.errors: print(err)

                ```
            - Inspect structural details from the report:
                ```python
                >>> report.details.get("blocksize")  # doctest: +SKIP
                [512, 512]

                ```
        """
        fn = self._on_disk_path()
        if fn is None:
            raise FileNotFoundError(
                "Dataset has no on-disk backing file to validate "
                "(is this a MEM or /vsimem/ dataset?)"
            )
        return validate(fn, strict=strict, config=config)

    def info(self, config: dict[str, str] | None = None) -> COGInfo:
        """Return structured COG metadata for the backing file.

        Reads only headers/metadata (no pixels) and reports compression,
        predictor, blocksize, dtype, CRS/bounds/resolution, the overview
        pyramid, per-band tags, and colour-table presence. See
        :class:`pyramids.dataset.cog.inspect.COGInfo`.

        Args:
            config: GDAL config options for the read; defaults to the remote
                read tuning for `/vsicurl` paths.

        Returns:
            COGInfo: The structured metadata for the on-disk file.

        Raises:
            FileNotFoundError: Dataset has no on-disk backing file
                (MEM-only or `/vsimem/`).

        Examples:
            - Inspect a COG's compression and overview pyramid:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("scene_cog.tif")  # doctest: +SKIP
                >>> info = ds.cog_info()  # doctest: +SKIP
                >>> info.compression  # doctest: +SKIP
                'DEFLATE'
                >>> [o.decimation for o in info.overviews]  # doctest: +SKIP
                [2, 4, 8]

                ```
            - Read the tile size and band count:
                ```python
                >>> info.blocksize  # doctest: +SKIP
                (512, 512)
                >>> info.band_count  # doctest: +SKIP
                1

                ```
        """
        fn = self._on_disk_path()
        if fn is None:
            raise FileNotFoundError(
                "Dataset has no on-disk backing file to inspect "
                "(is this a MEM or /vsimem/ dataset?)"
            )
        return cog_info(fn, config=config)

    def _on_disk_path(self) -> str | None:
        """Return the validatable on-disk path of the backing raster, or None.

        A single predicate shared by :attr:`is_cog`, :meth:`validate_cog`, and
        :meth:`info` (ARC-5) so the definition of "has a real backing file to
        validate/inspect" cannot drift between them.

        Returns:
            str | None: The file path when the dataset is backed by a real
            on-disk (or remote `/vsi*`, but not in-memory `/vsimem/`) file;
            `None` for MEM datasets, `/vsimem/` paths, and unsaved datasets.
        """
        fn = self._ds.file_name
        if not fn or fn.startswith("/vsimem/"):
            return None
        return fn

    def read_part(
        self,
        bbox: tuple[float, float, float, float],
        *,
        dst_width: int | None = None,
        dst_height: int | None = None,
        bbox_crs: int = 4326,
        resampling: str = "bilinear",
        band: int | None = None,
    ) -> np.typing.NDArray:
        """Read a geographic window, decimated from the nearest overview.

        Requesting a `dst_width`/`dst_height` smaller than the source window
        makes GDAL serve the data from the nearest overview level, so for a COG
        over `/vsicurl/` only the relevant byte ranges are fetched — the
        cloud-native partial-read pattern.

        Args:
            bbox: `(min_x, min_y, max_x, max_y)` window in `bbox_crs`.
            dst_width: Output width in pixels. Defaults to the source window
                width (no decimation).
            dst_height: Output height in pixels. Defaults to the source window
                height.
            bbox_crs: EPSG code of `bbox`. Reprojected to the dataset CRS
                when different. Defaults to 4326 (WGS84 lon/lat).
            resampling: Resampling method, case-insensitive. One of `nearest`,
                `bilinear`, `cubic`, `cubicspline` (alias `cubic_spline`),
                `lanczos`, `average`, `mode`, plus `gauss` and `rms` when the
                GDAL build provides them.
            band: 0-based band index. `None` reads all bands.

        Returns:
            numpy.ndarray: `(rows, cols)` for a single band, or
            `(bands, rows, cols)` for all bands; always sized
            `dst_height x dst_width` (the requested output size). Pixel values
            only — no transform, bounds, or CRS is attached.

        Raises:
            TypeError: `resampling` is not a string.
            ValueError: Unknown `resampling`.
            OutOfBoundsError: The window does not intersect the raster at all.

        Note:
            A window that only **partially** overlaps the raster is **not**
            stretched to fill the output: the intersection is read and placed
            at its correct offset inside a `dst_height x dst_width` buffer
            whose out-of-raster remainder is filled with NoData (the band's
            NoData value, else NaN for float / `0` for integer — see
            :meth:`_nodata_fill`). A fully-inside window is returned without
            padding. This keeps the result aligned to the requested window,
            which matters for edge tiles served by :meth:`read_tile`.

        Examples:
            - Read a 256x256 decimated thumbnail of a bbox:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("scene_cog.tif")  # doctest: +SKIP
                >>> arr = ds.read_part(  # doctest: +SKIP
                ...     (12.4, 41.8, 12.6, 42.0), dst_width=256, dst_height=256,
                ... )
                >>> arr.shape[-2:]  # doctest: +SKIP
                (256, 256)

                ```
        """
        alg = _resolve_read_resampling(resampling)
        # This serves a decimated window from the source; a NetCDF multidim view can't be window-read
        # by GDAL >= 3.13, so materialise it first (no-op for an ordinary raster).
        self._ds._materialize_md_view()
        ds = self._ds._raster
        min_x, min_y, max_x, max_y = self._reproject_bbox(bbox, bbox_crs)
        inv = gdal.InvGeoTransform(ds.GetGeoTransform())
        px_tl, py_tl = gdal.ApplyGeoTransform(inv, min_x, max_y)
        px_br, py_br = gdal.ApplyGeoTransform(inv, max_x, min_y)

        # The full requested window, in source pixel coordinates (may extend
        # beyond the raster on any side).
        req_xoff = int(math.floor(min(px_tl, px_br)))
        req_yoff = int(math.floor(min(py_tl, py_br)))
        req_xsize = int(math.ceil(max(px_tl, px_br))) - req_xoff
        req_ysize = int(math.ceil(max(py_tl, py_br))) - req_yoff
        if req_xsize <= 0 or req_ysize <= 0:
            raise OutOfBoundsError(
                f"bbox {bbox} (crs {bbox_crs}) has zero pixel extent"
            )

        # Intersection of the requested window with the raster.
        ix0 = max(0, req_xoff)
        iy0 = max(0, req_yoff)
        ix1 = min(ds.RasterXSize, req_xoff + req_xsize)
        iy1 = min(ds.RasterYSize, req_yoff + req_ysize)
        if ix1 - ix0 <= 0 or iy1 - iy0 <= 0:
            raise OutOfBoundsError(
                f"bbox {bbox} (crs {bbox_crs}) does not intersect the raster"
            )

        out_w = dst_width if dst_width is not None else req_xsize
        out_h = dst_height if dst_height is not None else req_ysize
        source = ds if band is None else ds.GetRasterBand(band + 1)

        fully_inside = (
            ix0 == req_xoff
            and iy0 == req_yoff
            and ix1 == req_xoff + req_xsize
            and iy1 == req_yoff + req_ysize
        )
        if fully_inside:
            return np.asarray(
                source.ReadAsArray(
                    ix0,
                    iy0,
                    ix1 - ix0,
                    iy1 - iy0,
                    buf_xsize=out_w,
                    buf_ysize=out_h,
                    resample_alg=alg,
                )
            )

        # Partial overlap: read only the intersection, then place it at its
        # correct offset inside a full-size output buffer padded with NoData,
        # so the returned array stays aligned to the requested window.
        scale_x = out_w / req_xsize
        scale_y = out_h / req_ysize
        ox0 = max(0, min(out_w, int(round((ix0 - req_xoff) * scale_x))))
        oy0 = max(0, min(out_h, int(round((iy0 - req_yoff) * scale_y))))
        ox1 = max(ox0 + 1, min(out_w, int(round((ix1 - req_xoff) * scale_x))))
        oy1 = max(oy0 + 1, min(out_h, int(round((iy1 - req_yoff) * scale_y))))
        sub = np.asarray(
            source.ReadAsArray(
                ix0,
                iy0,
                ix1 - ix0,
                iy1 - iy0,
                buf_xsize=ox1 - ox0,
                buf_ysize=oy1 - oy0,
                resample_alg=alg,
            )
        )
        fill = self._nodata_fill(ds.GetRasterBand(1))
        if sub.ndim == 3:
            out = np.full((sub.shape[0], out_h, out_w), fill, dtype=sub.dtype)
            out[:, oy0:oy1, ox0:ox1] = sub
        else:
            out = np.full((out_h, out_w), fill, dtype=sub.dtype)
            out[oy0:oy1, ox0:ox1] = sub
        return out

    @staticmethod
    def _nodata_fill(band: Any) -> float:
        """Pick a fill value for padding partial reads.

        Args:
            band: The GDAL band whose NoData value (if any) to use.

        Returns:
            float: The band's NoData value, else NaN for floating-point bands
            and ``0`` for integer bands.
        """
        nodata = band.GetNoDataValue()
        if nodata is not None:
            return cast(float, nodata)
        return 0 if is_integer_gdal_dtype(band.DataType) else float("nan")

    def preview(
        self,
        *,
        max_size: int = 1024,
        resampling: str = "bilinear",
        band: int | None = None,
    ) -> np.typing.NDArray:
        """Read a whole-image thumbnail downsampled to `max_size` on the long edge.

        Pulls from a coarse overview when one exists, so previewing a huge COG
        is cheap.

        Args:
            max_size: Maximum pixels on the longer edge. Defaults to 1024.
            resampling: Resampling method (see :meth:`read_part`).
            band: 0-based band index. `None` reads all bands.

        Returns:
            numpy.ndarray: The downsampled array, `(rows, cols)` or
            `(bands, rows, cols)`. Pixel values only — no transform, bounds,
            or CRS is attached to the returned array.

        Raises:
            TypeError: `resampling` is not a string.
            ValueError: Unknown `resampling`.

        Examples:
            - Build a 128px thumbnail of a single band:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("scene_cog.tif")  # doctest: +SKIP
                >>> thumb = ds.preview(max_size=128, band=0)  # doctest: +SKIP
                >>> max(thumb.shape)  # doctest: +SKIP
                128

                ```
        """
        alg = _resolve_read_resampling(resampling)
        width, height = self._ds.columns, self._ds.rows
        scale = max(width, height) / max_size
        if scale <= 1:
            out_w, out_h = width, height
        else:
            out_w, out_h = max(1, round(width / scale)), max(1, round(height / scale))
        ds = self._ds._raster
        source = ds if band is None else ds.GetRasterBand(band + 1)
        return np.asarray(
            source.ReadAsArray(buf_xsize=out_w, buf_ysize=out_h, resample_alg=alg)
        )

    def point(
        self,
        x: float,
        y: float,
        *,
        point_crs: int = 4326,
        band: int | None = None,
    ) -> np.typing.NDArray:
        """Sample band value(s) at a single coordinate.

        Args:
            x: X / longitude / easting in `point_crs`.
            y: Y / latitude / northing in `point_crs`.
            point_crs: EPSG code of `(x, y)`. Reprojected to the dataset CRS
                when different. Defaults to 4326.
            band: 0-based band index. `None` samples all bands.

        Returns:
            numpy.ndarray: A scalar 0-d array for a single band, or a
            `(bands,)` array when `band` is `None`. Pixel values only — no
            coordinate metadata is attached.

        Raises:
            OutOfBoundsError: The point falls outside the raster extent.

        Examples:
            - Sample all bands at a lon/lat coordinate:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("scene_cog.tif")  # doctest: +SKIP
                >>> ds.point(12.5, 41.9)  # doctest: +SKIP
                array([1234.], dtype=float32)

                ```
        """
        col, row = self._world_to_pixel(x, y, point_crs)
        if not (0 <= col < self._ds.columns and 0 <= row < self._ds.rows):
            raise OutOfBoundsError(
                f"point ({x}, {y}) in crs {point_crs} is outside the raster extent"
            )
        ds = self._ds._raster
        source = ds if band is None else ds.GetRasterBand(band + 1)
        arr = np.asarray(source.ReadAsArray(col, row, 1, 1))
        return arr.reshape(-1) if band is None else arr.reshape(())

    def read_tile(
        self,
        z: int,
        x: int,
        y: int,
        *,
        tilesize: int = 256,
        resampling: str = "bilinear",
        band: int | None = None,
    ) -> np.typing.NDArray:
        """Read a Web-Mercator XYZ/slippy-map tile.

        Computes the EPSG:3857 bounds of tile `(z, x, y)` from the closed-form
        Web-Mercator formula and delegates to :meth:`read_part` at `tilesize`
        resolution — no extra tiling dependency needed.

        Args:
            z: Zoom level.
            x: Tile column index.
            y: Tile row index (origin top-left / north-west).
            tilesize: Output tile size in pixels (square). Defaults to 256.
            resampling: Resampling method (see :meth:`read_part`).
            band: 0-based band index. `None` reads all bands.

        Returns:
            numpy.ndarray: A `(tilesize, tilesize)` or
            `(bands, tilesize, tilesize)` array. Pixel values only — the tile's
            georeferencing is defined by its `(z, x, y)`, not attached to the
            array; edge tiles are NoData-padded (see :meth:`read_part`).

        Raises:
            OutOfBoundsError: The tile does not intersect the raster.

        Examples:
            - Read the zoom-0 world tile of a global COG:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("global_cog.tif")  # doctest: +SKIP
                >>> tile = ds.read_tile(0, 0, 0)  # doctest: +SKIP
                >>> tile.shape[-2:]  # doctest: +SKIP
                (256, 256)

                ```
        """
        bounds = _xyz_bounds_3857(z, x, y)
        return self.read_part(
            bounds,
            dst_width=tilesize,
            dst_height=tilesize,
            bbox_crs=3857,
            resampling=resampling,
            band=band,
        )

    def _reproject_bbox(
        self, bbox: tuple[float, float, float, float], bbox_crs: int
    ) -> tuple[float, float, float, float]:
        """Reproject a bbox into the dataset CRS, returning its envelope.

        Args:
            bbox: `(min_x, min_y, max_x, max_y)` in `bbox_crs`.
            bbox_crs: EPSG code of `bbox`.

        Returns:
            `(min_x, min_y, max_x, max_y)` in the dataset CRS. When
            `bbox_crs` already matches the dataset EPSG the bbox is
            returned unchanged.
        """
        min_x, min_y, max_x, max_y = bbox
        if self._ds.epsg == bbox_crs:
            return min_x, min_y, max_x, max_y
        transformer = _cached_transformer(bbox_crs, self._ds.epsg or self._ds.crs)
        corners = [
            transformer.transform(min_x, min_y),
            transformer.transform(min_x, max_y),
            transformer.transform(max_x, min_y),
            transformer.transform(max_x, max_y),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def _world_to_pixel(self, x: float, y: float, point_crs: int) -> tuple[int, int]:
        """Convert a world coordinate to integer `(col, row)` pixel indices.

        Args:
            x: X / longitude in `point_crs`.
            y: Y / latitude in `point_crs`.
            point_crs: EPSG code of `(x, y)`.

        Returns:
            `(col, row)` integer pixel indices (floored).
        """
        if self._ds.epsg != point_crs:
            transformer = _cached_transformer(point_crs, self._ds.epsg or self._ds.crs)
            x, y = transformer.transform(x, y)
        inv = gdal.InvGeoTransform(self._ds._raster.GetGeoTransform())
        col, row = gdal.ApplyGeoTransform(inv, x, y)
        return int(math.floor(col)), int(math.floor(row))

    def _warn_if_categorical_with_averaging(
        self, overview_resampling: str, band: Any | None = None
    ) -> None:
        """Emit a `UserWarning` if an averaging resampler is used on categorical data.

        Args:
            overview_resampling: The resampling method requested by the
                caller. Case-insensitive. Only averaging-family methods
                (`average`, `bilinear`, `cubic`, `cubicspline`,
                `lanczos`) trigger the check.
            band: GDAL band whose dtype/colour-table decides "categorical".
                Defaults to band 1 of the backing raster; a pre-processed
                (cast/subset) band is passed when those options are used so
                the check reflects the *output* dtype (PB-4).

        Warns:
            UserWarning: When `overview_resampling` is an averaging
                method and the source has a color table OR integer
                dtype — both strong signals of categorical data.

        Note:
            Silent when `overview_resampling` is `nearest` or
            `mode` (both category-safe) or when the source is
            floating-point and has no color table (continuous data).

        Examples:
            - Integer dataset + averaging method emits a warning:
                ```python
                >>> import warnings  # doctest: +SKIP
                >>> with warnings.catch_warnings(record=True) as caught:  # doctest: +SKIP
                ...     warnings.simplefilter("always")
                ...     byte_ds.cog._warn_if_categorical_with_averaging("average")
                ...     [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
                ['overview_resampling=\\'average\\' averages pixel values, ...']

                ```
            - Nearest resampling is always silent:
                ```python
                >>> with warnings.catch_warnings(record=True) as caught:  # doctest: +SKIP
                ...     warnings.simplefilter("always")
                ...     byte_ds.cog._warn_if_categorical_with_averaging("nearest")
                ...     len(caught)
                0

                ```
        """
        if overview_resampling.lower() not in _AVERAGING_RESAMPLERS:
            return
        first_band = band if band is not None else self._ds._raster.GetRasterBand(1)
        has_color_table = first_band.GetColorTable() is not None
        is_integer = is_integer_gdal_dtype(first_band.DataType)
        if has_color_table or is_integer:
            warnings.warn(
                f"overview_resampling={overview_resampling!r} averages pixel "
                "values, which corrupts categorical rasters (land cover, IDs). "
                "Use overview_resampling='nearest' or 'mode' instead.",
                UserWarning,
                stacklevel=3,
            )
