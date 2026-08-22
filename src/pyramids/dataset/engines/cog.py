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
from pyramids.base._utils import is_integer_gdal_dtype
from pyramids.base.crs import crs_equal, crs_from_user_input, require_crs_spec
from pyramids.dataset.abstract_dataset import under_gdal_env
from pyramids.dataset.cog import (
    BandSelection,
    COGInfo,
    Compression,
    Layout,
    Overviews,
    Tags,
    Tiling,
    ValidationReport,
    cog_info,
    merge_options,
    translate_to_cog,
    validate,
    validate_profile,
)
from pyramids.dataset.cog.options import _reconcile_predictor_with_nbits
from pyramids.dataset.cog.validate import _resolve_read_config, config_context
from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.engines._validate import world_to_pixel

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
    # Both sides go through `crs_from_user_input` so a code that lives in GDAL's PROJ
    # database but not pyproj's still builds a transformer (issue #943).
    return Transformer.from_crs(
        crs_from_user_input(src_crs), crs_from_user_input(dst_crs), always_xy=True
    )


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
        compression: str | Compression | None = None,
        overviews: Overviews | None = None,
        tiling: Tiling | None = None,
        bands: BandSelection | None = None,
        tags: Tags | None = None,
        layout: Layout | None = None,
        config: dict[str, str] | None = None,
        extra: Mapping[str, Any] | list[str] | None = None,
    ) -> Path:
        """Save the dataset as a Cloud Optimized GeoTIFF.

        The write options are organised into grouped, validated dataclasses
        (:class:`~pyramids.dataset.cog.Compression`,
        :class:`~pyramids.dataset.cog.Overviews`,
        :class:`~pyramids.dataset.cog.Tiling`,
        :class:`~pyramids.dataset.cog.BandSelection`,
        :class:`~pyramids.dataset.cog.Tags`,
        :class:`~pyramids.dataset.cog.Layout`) — see each class for its fields.
        The whole set is accessible under the ``cog`` namespace, e.g.
        ``from pyramids.dataset import cog`` then ``cog.Compression(...)``.

        Args:
            path: Destination path. Parent directory must exist.
            compression: How pixel bytes are compressed. Either a named profile
                string (`deflate`, `zstd`, `lzw`, `packbits`, `jpeg`, `webp`,
                `lerc`, `lerc_deflate`, `lerc_zstd`, `raw`) or a
                :class:`~pyramids.dataset.cog.Compression`. `None` uses the
                house `DEFLATE` default. The `jpeg`/`webp` **profile strings**
                enforce dtype/band constraints (Byte; 1-3 / 3-4 bands) up front;
                a direct `Compression(compress="JPEG")` is passed to GDAL
                unchecked. The predictor auto-resolves per source dtype (`2`
                integer, `3` float) unless set on `Compression`.
            overviews: The internal overview pyramid, as an
                :class:`~pyramids.dataset.cog.Overviews`. `None` builds the
                default pyramid with dtype-aware resampling (`mode` for
                categorical sources, `average` for continuous).
            tiling: Reprojection / web-tiling layout, as a
                :class:`~pyramids.dataset.cog.Tiling` (`target_srs`, `scheme`
                e.g. `"GoogleMapsCompatible"`, warp `resampling`, zoom knobs).
                `None` writes in the source CRS with no tiling scheme.
            bands: Band selection / dtype cast / NoData applied before the
                write, as a :class:`~pyramids.dataset.cog.BandSelection`. Any of
                its fields pre-processes the source through an in-memory
                `gdal.Translate` so the predictor/overview policy sees the
                output bands.
            tags: Metadata and colour table to stamp onto the output, as a
                :class:`~pyramids.dataset.cog.Tags` (`band_tags`, `colormap`,
                `metadata`).
            layout: Physical layout / driver behaviour, as a
                :class:`~pyramids.dataset.cog.Layout` (`blocksize`, `bigtiff`,
                `num_threads`, `add_mask`, `sparse_ok`, `statistics`). `None`
                uses the house defaults.
            config: GDAL config options (e.g. `{"GDAL_NUM_THREADS": "4"}`)
                applied via `gdal.config_options` for the duration of the
                write. `None` (default) applies no extra config.
            extra: Additional GDAL creation options as a mapping or
                legacy `['KEY=VALUE',...]` list. Overrides conflicting
                group fields.

        Returns:
            Path: The resolved destination path.

        Raises:
            ValueError: Invalid blocksize or unknown option key.
            FileNotFoundError: Parent directory does not exist.
            FailedToSaveError: GDAL CreateCopy failed.
            DriverNotExistError: GDAL build lacks the COG driver.

        Warnings:
            UserWarning: When the source looks categorical (integer dtype or a
                color table) and `overviews.resampling` is an averaging method;
                and when both `tiling.scheme` and `tiling.target_srs` are set
                (`scheme` wins, `target_srs` is ignored).

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
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> out = ds.to_cog("out.tif", compression="zstd")  # doctest: +SKIP
                >>> out.name  # doctest: +SKIP
                'out.tif'

                ```
            - Produce a web-optimized COG for a tile server:
                ```python
                >>> from pyramids.dataset import cog  # doctest: +SKIP
                >>> web = ds.to_cog(  # doctest: +SKIP
                ...     "web.tif", tiling=cog.Tiling(scheme="GoogleMapsCompatible"),
                ... )

                ```
            - Select bands and forward additional GDAL options through `extra`:
                ```python
                >>> from pyramids.dataset import cog  # doctest: +SKIP
                >>> _ = ds.to_cog(  # doctest: +SKIP
                ...     "precise.tif",
                ...     compression="lerc",
                ...     bands=cog.BandSelection(indexes=[2, 1, 0]),
                ...     extra={"MAX_Z_ERROR": 0.001},
                ... )

                ```
        """
        # The jpeg/webp dtype/band check is a named-profile convenience (it mirrors
        # the profile presets), so it fires only when the method was selected via a
        # profile string — a direct `Compression(compress="JPEG")` goes straight to
        # GDAL, matching the pre-refactor flat `compress="JPEG"` path (which GDAL
        # accepts for e.g. 4-band Byte).
        compression_from_profile = isinstance(compression, str)
        compression = Compression.coerce(compression)
        if compression is None:
            compression = Compression()
        overviews = overviews or Overviews()
        tiling = tiling or Tiling()
        bands = bands or BandSelection()
        tags = tags or Tags()
        layout = layout or Layout()

        # Build the effective source (PB-4): band-subset/cast/NoData routes through
        # `BandSelection._translate` and tag/colourmap/metadata through
        # `Tags._stamp`, so the predictor/overview policy below — and the COG write
        # itself — see the *output* bands, and the user's dataset is never mutated.
        source_ds, source_band0 = self._effective_source(bands, tags)

        # Single house policy (ARC-1): each option group serializes its own fields
        # — `Compression`/`Overviews` resolve the dtype-aware predictor and overview
        # resampling from the source band — so a direct `ds.to_cog(...)` and the
        # `write_cog(...)` facade produce identical output for identical input.
        # Assembled before the checks below so `Tiling._to_options`'
        # scheme-vs-target_srs conflict warning keeps firing ahead of them (and is
        # not swallowed by a `validate_profile` raise), matching pre-refactor order.
        defaults = {
            **compression._to_options(source_band0),
            **overviews._to_options(source_band0),
            **tiling._to_options(),
            **layout._to_options(),
        }

        # The jpeg/webp dtype/band constraints (PB-5) mirror the named-profile
        # presets, so they are enforced against the *effective* source only for the
        # profile-string path; a direct `Compression(compress="JPEG")` goes straight
        # to GDAL, matching the pre-refactor flat `compress="JPEG"` behaviour.
        if compression_from_profile and compression.compress is not None:
            validate_profile(
                compression.compress.lower(),
                gdal.GetDataTypeName(source_band0.DataType),
                source_ds.RasterCount,
            )

        # Guardrail (ARC-3): warn only when the *caller* explicitly asked for an
        # averaging resampler on categorical data — never for a default resolved
        # inside `Overviews._to_options`.
        if overviews.resampling is not None:
            self._warn_if_categorical_with_averaging(
                overviews.resampling, band=source_band0
            )

        options = merge_options(defaults, extra)
        # If a caller forced a sub-byte-aligned NBITS through `extra`, the
        # promoted-width predictor computed above would make GDAL reject the
        # write — drop the predictor so the caller's explicit width is honoured.
        _reconcile_predictor_with_nbits(options)
        with config_context(config):
            self._translate_with_statistics_retry(path, options, src=source_ds)
        return Path(path)

    def _effective_source(
        self, bands: BandSelection, tags: Tags
    ) -> tuple[gdal.Dataset, Any]:
        """Return the source dataset (optionally pre-processed) and its band 0.

        Orchestrates the two per-group transforms: a band subset / dtype cast /
        NoData override (PB-4) runs the source through
        :meth:`~pyramids.dataset.cog.BandSelection._translate`; band tags /
        colourmap / metadata (PC-2) are applied by
        :meth:`~pyramids.dataset.cog.Tags._stamp` onto a MEM copy so the user's
        open dataset is **never mutated**. With neither, the backing raster is
        returned unchanged.

        Args:
            bands: The band-selection group deciding the ``gdal.Translate``.
            tags: The tag/colourmap/metadata group to stamp.

        Returns:
            A ``(dataset, band1)`` tuple where ``band1`` is GDAL band 1 of the
            returned dataset (used for predictor/resampling resolution).

        Raises:
            FailedToSaveError: When the stamp-only MEM copy fails.
        """
        # The COG write (and the gdal.Translate pre-process) do tiled reads of the source; a
        # NetCDF multidim view can't be window-read by GDAL >= 3.13, so materialise it first (no-op
        # for an ordinary raster).
        self._ds._materialize_md_view()
        needs_translate = bands._needs_translate()
        needs_stamp = tags._has_any()
        if not needs_translate and not needs_stamp:
            ds = self._ds._raster
            return ds, ds.GetRasterBand(1)

        if needs_translate:
            mem = bands._translate(self._ds._raster)
        else:
            # Stamp-only: copy so the user's dataset is not mutated.
            mem = gdal.GetDriverByName("MEM").CreateCopy("", self._ds._raster)
            if mem is None:
                raise FailedToSaveError(
                    "failed to copy the source dataset for COG metadata stamping"
                )
        if needs_stamp:
            tags._stamp(mem)
        return mem, mem.GetRasterBand(1)

    def to_cog_bytes(self, **kwargs: Any) -> bytes:
        """Encode the dataset as a COG and return the file contents as bytes.

        Writes the COG to an in-memory GDAL ``/vsimem/`` file (no temp file on
        disk), reads the bytes back, and unlinks the virtual file. Useful for
        uploading a COG directly to an object store (S3 / GCS / Azure) without
        touching the local filesystem.

        Args:
            **kwargs: Forwarded verbatim to :meth:`to_cog` (e.g.
                ``compression``, ``layout``, ``bands``, ``extra``). The same
                house defaults and dtype-aware resolution apply.

        Returns:
            bytes: The complete COG file contents.

        Raises:
            FailedToSaveError: GDAL failed to encode the COG.

        Examples:
            - Encode an in-memory Dataset to COG bytes and upload them:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("scene.tif")  # doctest: +SKIP
                >>> blob = ds.to_cog_bytes(compression="zstd")  # doctest: +SKIP
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
    @under_gdal_env
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

    @under_gdal_env
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

    @under_gdal_env
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

    @under_gdal_env
    def read_part(
        self,
        bbox: tuple[float, float, float, float],
        *,
        dst_width: int | None = None,
        dst_height: int | None = None,
        bbox_crs: int | str | None = None,
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
            bbox_crs: CRS of `bbox`, reprojected to the dataset CRS when it
                differs. Defaults to `None`, meaning the bbox is already in the
                raster's own coordinates, so nothing is transformed.
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
            CRSError: An explicit `bbox_crs` was given but the raster has no CRS
                to transform into. Omit it to read in the raster's own
                coordinates (ARC-26).
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
        geotransform = ds.GetGeoTransform()
        px_tl, py_tl = world_to_pixel(geotransform, min_x, max_y)
        px_br, py_br = world_to_pixel(geotransform, max_x, min_y)

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

    @under_gdal_env
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

    @under_gdal_env
    def point(
        self,
        x: float,
        y: float,
        *,
        point_crs: int | str | None = None,
        band: int | None = None,
    ) -> np.typing.NDArray:
        """Sample band value(s) at a single coordinate.

        Args:
            x: X / longitude / easting in `point_crs`.
            y: Y / latitude / northing in `point_crs`.
            point_crs: CRS of `(x, y)`, reprojected to the dataset CRS when it
                differs. Defaults to `None`, meaning the coordinates are already
                in the raster's own CRS, so nothing is transformed.
            band: 0-based band index. `None` samples all bands.

        Returns:
            numpy.ndarray: A scalar 0-d array for a single band, or a
            `(bands,)` array when `band` is `None`. Pixel values only — no
            coordinate metadata is attached.

        Raises:
            CRSError: An explicit `point_crs` was given but the raster has no
                CRS to transform into. Omit it to read in the raster's own
                coordinates (ARC-26).
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

    @under_gdal_env
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
        self, bbox: tuple[float, float, float, float], bbox_crs: int | str | None
    ) -> tuple[float, float, float, float]:
        """Reproject a bbox into the dataset CRS, returning its envelope.

        Args:
            bbox: `(min_x, min_y, max_x, max_y)` in `bbox_crs`.
            bbox_crs: CRS of `bbox`. `None` (default) means the bbox is already
                in the raster's own coordinates.

        Returns:
            `(min_x, min_y, max_x, max_y)` in the dataset CRS. When
            `bbox_crs` already matches the dataset EPSG the bbox is
            returned unchanged.
        """
        min_x, min_y, max_x, max_y = bbox
        envelope = bbox
        # `None` means the caller named no CRS, so the bbox is already in the
        # raster's own coordinates and there is nothing to transform (ARC-26).
        # An *explicit* bbox_crs goes through `require_crs_spec`, so a raster
        # with no CRS reports the mismatch rather than silently ignoring the
        # argument.
        if bbox_crs is not None:
            target = require_crs_spec(
                self._ds.epsg, self._ds.crs, "read a bbox window in another CRS"
            )
            if not crs_equal(bbox_crs, target):
                transformer = _cached_transformer(bbox_crs, target)
                corners = [
                    transformer.transform(min_x, min_y),
                    transformer.transform(min_x, max_y),
                    transformer.transform(max_x, min_y),
                    transformer.transform(max_x, max_y),
                ]
                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                envelope = (min(xs), min(ys), max(xs), max(ys))
        return envelope

    def _world_to_pixel(
        self, x: float, y: float, point_crs: int | str | None
    ) -> tuple[int, int]:
        """Convert a world coordinate to integer `(col, row)` pixel indices.

        Args:
            x: X / longitude in `point_crs`.
            y: Y / latitude in `point_crs`.
            point_crs: CRS of `(x, y)`. `None` (the default) means the point is
                already in the raster's own coordinates.

        Returns:
            `(col, row)` integer pixel indices (floored).
        """
        # Mirrors the bbox path: `None` means the point is already in the
        # raster's own coordinates, so there is nothing to transform. Without
        # this early-out the default builds a transformer FROM None and fails on
        # every georeferenced raster. Past it the caller HAS named a CRS, so the
        # raster must have one to transform into -- `require_crs_spec` rather
        # than a silent skip, which ignored the argument that was passed.
        if point_crs is not None:
            target = require_crs_spec(
                self._ds.epsg, self._ds.crs, "sample a point given in another CRS"
            )
            if not crs_equal(point_crs, target):
                transformer = _cached_transformer(point_crs, target)
                x, y = transformer.transform(x, y)
        col, row = world_to_pixel(self._ds._raster.GetGeoTransform(), x, y)
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
