"""COG engine.

Owns the COG family of operations on a Dataset. Accessed as
``ds.cog``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.cog.<method>(...)`` are equivalent.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

from osgeo import gdal

from pyramids.dataset.cog import (
    ValidationReport,
    merge_options,
    translate_to_cog,
    validate,
    validate_blocksize,
)
from pyramids.dataset.engines._base import _Engine

_AVERAGING_RESAMPLERS: frozenset[str] = frozenset(
    {"average", "bilinear", "cubic", "cubicspline", "lanczos"}
)


_INTEGER_DTYPES: frozenset[int] = frozenset(
    {
        gdal.GDT_Byte,
        gdal.GDT_UInt16,
        gdal.GDT_Int16,
        gdal.GDT_UInt32,
        gdal.GDT_Int32,
        gdal.GDT_UInt64,
        gdal.GDT_Int64,
        gdal.GDT_Int8,
    }
)


class COG(_Engine):
    """Cloud Optimized GeoTIFF read/write/validate operations for `Dataset`.

    Owns the real implementations of `to_cog`, `is_cog` (property),
    and `validate_cog` after L-2 PR 2.2. `Dataset` exposes a
    same-named facade for each so `ds.to_cog(...)` and
    `ds.cog.to_cog(...)` are equivalent. The categorical-raster
    resampling guardrail (`_warn_if_categorical_with_averaging`)
    lives here too.
    """

    def to_cog(
        self,
        path: str | Path,
        *,
        compress: str = "DEFLATE",
        level: int | None = None,
        quality: int | None = None,
        blocksize: int = 512,
        predictor: str | int | None = None,
        bigtiff: str = "IF_SAFER",
        num_threads: int | str = "ALL_CPUS",
        overview_resampling: str = "nearest",
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
        extra: Mapping[str, Any] | list[str] | None = None,
    ) -> Path:
        """Save the dataset as a Cloud Optimized GeoTIFF.

        Args:
            path: Destination path. Parent directory must exist.
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
            bigtiff: `"IF_SAFER"` (default), `"YES"`, `"NO"`,
                `"IF_NEEDED"`.
            num_threads: Worker threads; `"ALL_CPUS"` or an int.
            overview_resampling: `nearest`, `average`, `bilinear`,
                `cubic`, `cubicspline`, `lanczos`, `mode`,
                `rms`, `gauss`.
            overview_count: Number of overview levels (default: auto).
            overview_compress: Compression for overview IFDs.
            tiling_scheme: e.g., `"GoogleMapsCompatible"` for a
                web-optimized COG (EPSG:3857).
            zoom_level, zoom_level_strategy, aligned_levels: Advanced
                tiling-scheme knobs.
            resampling: Warp resampling when `tiling_scheme` or
                `target_srs` reprojects.
            add_mask: Add an alpha band for transparency.
            sparse_ok: Allow sparse (unfilled) tiles.
            target_srs: Reproject before write. Int for EPSG or a WKT
                / PROJ string.
            statistics: Compute and embed band statistics.
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
        self._warn_if_categorical_with_averaging(overview_resampling)
        if tiling_scheme is not None and target_srs is not None:
            warnings.warn(
                "Both tiling_scheme and target_srs provided; "
                "tiling_scheme wins and target_srs is ignored.",
                UserWarning,
                stacklevel=2,
            )
            target_srs = None

        num_threads_str = (
            num_threads if isinstance(num_threads, str) else str(num_threads)
        )
        defaults: dict[str, Any] = {
            "COMPRESS": compress,
            "LEVEL": level,
            "QUALITY": quality,
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

        dst: gdal.Dataset | None = None
        try:
            dst = translate_to_cog(self._ds._raster, path, options)
            dst.FlushCache()
        finally:
            dst = None

        return Path(path)

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
        fn = self._ds.file_name
        if not fn or fn.startswith("/vsimem/"):
            result = False
        else:
            try:
                result = validate(fn).is_valid
            except FileNotFoundError:
                result = False
        return result

    def validate_cog(self, strict: bool = False) -> ValidationReport:
        """Validate the backing file as a COG.

        Args:
            strict: If `True`, warnings are treated as errors.

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
        fn = self._ds.file_name
        if not fn or fn.startswith("/vsimem/"):
            raise FileNotFoundError(
                "Dataset has no on-disk backing file to validate "
                "(is this a MEM or /vsimem/ dataset?)"
            )
        return validate(fn, strict=strict)

    def _warn_if_categorical_with_averaging(self, overview_resampling: str) -> None:
        """Emit a `UserWarning` if an averaging resampler is used on categorical data.

        Args:
            overview_resampling: The resampling method requested by the
                caller. Case-insensitive. Only averaging-family methods
                (`average`, `bilinear`, `cubic`, `cubicspline`,
                `lanczos`) trigger the check.

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
        first_band = self._ds._raster.GetRasterBand(1)
        has_color_table = first_band.GetColorTable() is not None
        is_integer = first_band.DataType in _INTEGER_DTYPES
        if has_color_table or is_integer:
            warnings.warn(
                f"overview_resampling={overview_resampling!r} averages pixel "
                "values, which corrupts categorical rasters (land cover, IDs). "
                "Use overview_resampling='nearest' or 'mode' instead.",
                UserWarning,
                stacklevel=3,
            )
