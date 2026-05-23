"""Free-function entry points for merging a list of rasters.

* :func:`merge_rasters` — mosaic rasters that tile the *same area* into one
  raster (spatial merge).
* :func:`stack_bands` — stack N single-band rasters that cover the *same
  grid* into one multi-band raster (band-wise merge). Thin alias for
  :meth:`pyramids.dataset.Dataset.from_band_files`.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal, osr

from pyramids.base._utils import INTERPOLATION_METHODS
from pyramids.base.remote import CloudConfig
from pyramids.dataset.dataset import _INHERIT_NO_DATA, Dataset

_VRT_METHODS = ("first", "last")
_REDUCE_METHODS = ("min", "max", "sum")
_MERGE_METHODS = _VRT_METHODS + _REDUCE_METHODS


def _cloud_config(signer: Any):
    """Return a context manager installing ``signer``'s GDAL config, or a no-op.

    Mirrors how :func:`pyramids.stac.load_asset` applies a signer: the signer's
    ``gdal_env()`` mapping is fed into :class:`~pyramids.base.remote.CloudConfig`
    so authenticated cloud reads work for the duration of the ``with`` block.

    Args:
        signer (Any): A signer exposing ``gdal_env() -> dict[str, str]`` (e.g. a
            :class:`pyramids.stac.signers.Signer`), or ``None``.

    Returns:
        A context manager: a :class:`contextlib.nullcontext` when ``signer`` is
        ``None`` (no GDAL config installed, behaviour unchanged), otherwise a
        :class:`~pyramids.base.remote.CloudConfig` seeded with
        ``signer.gdal_env()``.
    """
    return nullcontext() if signer is None else CloudConfig(extra=signer.gdal_env())


def merge_rasters(
    src: list[str | Path],
    dst: str | Path,
    no_data_value: float | int | str = "0",
    init: float | int | str = "nan",
    n: float | int | str = "nan",
    method: str = "last",
    dst_crs: int | str | None = None,
    resampling: str = "nearest neighbor",
    signer: Any = None,
) -> None:
    """Merge a group of rasters into one raster, resolving overlaps by ``method``.

    The overlap-resolution rule mirrors ``rasterio.merge(method=…)``:

    * ``"last"`` (default) / ``"first"`` — z-order compositing: the last (or
      first) source covering a pixel wins. Implemented cheaply with
      :func:`gdal.BuildVRT` + :func:`gdal.Translate`.
    * ``"min"`` / ``"max"`` / ``"sum"`` — per-pixel reduction across every
      source overlapping that pixel, ignoring no-data. Each source is aligned
      onto the union grid and the bands are stacked and reduced with NaN-aware
      numpy.

    Args:
        src (list[str | Path]):
            List of paths to all input rasters.
        dst (str | Path):
            Path to the output raster.
        no_data_value (float | int | str):
            Stamped on the output bands as the nodata marker. For the reduction
            methods it also fills pixels with no source coverage.
        init (float | int | str):
            Reported value for pixels with no source coverage in the VRT (z-order
            methods only). Maps to :func:`gdal.BuildVRTOptions` ``VRTNodata``.
        n (float | int | str):
            Source pixels matching this value are ignored — both when building
            the VRT mosaic (z-order) and when reducing (the value is treated as
            source no-data).
        method (str):
            Overlap-resolution rule: one of ``"first"``, ``"last"`` (default),
            ``"min"``, ``"max"``, ``"sum"``.
        dst_crs (int | str | None):
            Target CRS for the mosaic, as an EPSG code (``32632``) or any
            GDAL-parseable CRS string (``"EPSG:32632"``, a WKT, a PROJ string).
            Each source whose CRS differs from the target is reprojected onto it
            (via :func:`gdal.Warp`) **before** compositing, so tiles in different
            CRSs — e.g. a Sentinel-2 AOI straddling two UTM zones — mosaic
            correctly. ``None`` (default) keeps the previous behaviour: sources
            are assumed to share a CRS and are composited as-is, *except* when
            they are found to disagree, in which case they are reprojected onto
            the first source's CRS. Reprojection always happens before the
            ``BuildVRT``/``Warp`` compositing step because that step has no
            reprojection capability and assumes a single shared grid.
        resampling (str):
            Resampling method used when a source is reprojected to ``dst_crs``
            (or to the common CRS on auto-detect). One of
            :data:`pyramids.base._utils.INTERPOLATION_METHODS`:
            ``"nearest neighbor"`` (default), ``"bilinear"``, or ``"cubic"``.
            Prefer ``"bilinear"``/``"cubic"`` for continuous data (reflectance,
            DEM) to avoid the blockiness nearest introduces across reprojection.
            Ignored when no source is reprojected.
        signer (Any):
            Optional signer exposing ``sign_href(str) -> str`` and
            ``gdal_env() -> dict[str, str]`` (e.g. a
            :class:`pyramids.stac.signers.Signer`). When given, **both** hooks
            are applied — exactly as :func:`pyramids.stac.load_asset` does:
            ``signer.sign_href`` rewrites every source path first (e.g. grafting
            a SAS token onto a blob URL), then ``signer.gdal_env()`` is installed
            via :class:`~pyramids.base.remote.CloudConfig` for the duration of
            the merge. This means URL-signing signers (Planetary Computer SAS,
            whose credential rides the href and whose ``gdal_env()`` is empty)
            and env-based signers (Requester-Pays, bearer) both authenticate
            without wrapping the call in a ``with CloudConfig(...)`` block.
            ``None`` (default) leaves source hrefs untouched and installs no
            extra config.

    Returns:
        None

    Note:
        The z-order methods (``"first"``/``"last"``) preserve each source's data
        type via ``BuildVRT`` + ``Translate``. The reduction methods
        (``"min"``/``"max"``/``"sum"``) align every source onto the union grid
        with ``gdal.Warp`` (nearest resampling — exact for already-aligned tiles)
        and write a single-precision-safe **Float64** output regardless of the
        source dtype, so they may differ in dtype from a z-order merge of the
        same integer inputs.

    Raises:
        ValueError: ``method``/``resampling`` is not a supported value,
            ``dst_crs`` cannot be parsed as a CRS, or a source carries no CRS.
        RuntimeError: GDAL failed to open a source, reproject it, or build the
            source mosaic.

    Examples:
        - Mosaic two tiles, keeping the larger value wherever they overlap:
            ```python
            >>> from pyramids.dataset.merge import merge_rasters
            >>> merge_rasters(  # doctest: +SKIP
            ...     ["tile_a.tif", "tile_b.tif"],
            ...     "mosaic_max.tif",
            ...     no_data_value=-9999.0,
            ...     method="max",
            ... )

            ```
        - Default last-wins compositing (unchanged from the previous behaviour):
            ```python
            >>> merge_rasters(["tile_a.tif", "tile_b.tif"], "mosaic.tif")  # doctest: +SKIP

            ```
        - Mosaic tiles from two UTM zones into a single CRS:
            ```python
            >>> merge_rasters(  # doctest: +SKIP
            ...     ["utm32_tile.tif", "utm33_tile.tif"],
            ...     "mosaic_utm32.tif",
            ...     dst_crs=32632,
            ... )

            ```
        - Mosaic Requester-Pays S3 tiles by passing a signer (no ``with`` block):
            ```python
            >>> from pyramids.stac import AWSRequesterPaysSigner  # doctest: +SKIP
            >>> merge_rasters(  # doctest: +SKIP
            ...     ["s3://bucket/a.tif", "s3://bucket/b.tif"],
            ...     "mosaic.tif",
            ...     signer=AWSRequesterPaysSigner(region="us-west-2"),
            ... )

            ```
    """
    if method not in _MERGE_METHODS:
        raise ValueError(
            f"method must be one of {list(_MERGE_METHODS)}, got {method!r}."
        )

    # SMELL: `init` and `n` default to the string `"nan"`, which
    # round-trips through GDAL as float NaN. For integer-typed
    # rasters (e.g. UInt16) GDAL emits a warning per band:
    # `Band data type of <T> cannot represent the specified NoData
    # value of nan`. The defaults are kept for backwards-compat
    # with the previous gdal_merge.main-based signature; callers
    # that hit integer rasters should pass an explicit numeric
    # value instead of relying on the default.
    src_paths = [str(p) for p in src]
    if signer is not None:
        # Apply the signer's href rewrite to every source (e.g. graft a SAS
        # token onto a blob URL) so URL-signing signers authenticate. A no-op
        # for signers that authenticate via gdal_env() only — the base
        # sign_href returns the href unchanged. This mirrors load_asset, which
        # applies BOTH signer hooks (sign_href + gdal_env); applying only the
        # env half here would silently read URL-signed sources unauthenticated.
        src_paths = [signer.sign_href(p) for p in src_paths]

    # All GDAL reads/writes run under the signer's cloud config (a no-op when
    # signer is None) so authenticated remote sources open with the right
    # credentials for the whole merge.
    with _cloud_config(signer):
        # Put every source on one CRS before compositing. The BuildVRT/Warp
        # mosaic below cannot reproject — it stitches pixel grids assuming a
        # shared CRS — so mismatched sources must be warped first or they would
        # mis-align silently. `_keepalive` holds the in-memory warped VRTs so
        # GDAL does not free them while the mosaic is built.
        sources, _keepalive = _prepare_sources(src_paths, dst_crs, resampling)

        if method in _REDUCE_METHODS:
            _merge_reduce(sources, str(dst), method, no_data_value, n)
            return

        # z-order: "last" keeps natural order (last source wins); "first"
        # reverses so the original first source is placed last in the VRT and
        # therefore wins.
        ordered = list(reversed(sources)) if method == "first" else sources
        vrt_opts = gdal.BuildVRTOptions(
            srcNodata=str(n),
            VRTNodata=str(init),
        )
        vrt_ds = gdal.BuildVRT("", ordered, options=vrt_opts)
        if vrt_ds is None:
            raise RuntimeError(
                f"gdal.BuildVRT returned None for sources {src_paths!r}; "
                "check that all paths are readable rasters with consistent "
                "band counts and CRS."
            )
        translate_opts = gdal.TranslateOptions(
            creationOptions=["COMPRESS=LZW"],
            noData=str(no_data_value),
        )
        out_ds = gdal.Translate(str(dst), vrt_ds, options=translate_opts)
        out_ds.FlushCache()
        out_ds = None
        vrt_ds = None


def _as_srs(crs: int | str) -> osr.SpatialReference:
    """Build an :class:`osr.SpatialReference` from an EPSG code or a CRS string.

    Args:
        crs: An EPSG code as an ``int`` (e.g. ``32632``), or any CRS string
            GDAL can parse via ``SetFromUserInput`` — ``"EPSG:32632"``, a WKT
            string, or a PROJ string.

    Returns:
        osr.SpatialReference: The parsed spatial reference.

    Raises:
        ValueError: ``crs`` could not be parsed as a CRS.
    """
    srs = osr.SpatialReference()
    try:
        # GDAL may signal a parse failure either by a non-zero return code or,
        # when exceptions are enabled (pyramids enables them at import), by
        # raising RuntimeError. Treat both as a bad CRS.
        if isinstance(crs, int):
            failed = srs.ImportFromEPSG(crs) != 0
        else:
            failed = srs.SetFromUserInput(str(crs)) != 0
    except RuntimeError as exc:
        raise ValueError(f"Could not parse dst_crs={crs!r} as a CRS.") from exc
    if failed:
        raise ValueError(f"Could not parse dst_crs={crs!r} as a CRS.")
    return srs


def _prepare_sources(
    src_paths: list[str],
    dst_crs: int | str | None,
    resampling: str = "nearest neighbor",
) -> tuple[list, list]:
    """Put every source on a single CRS before the mosaic is composited.

    The compositing step (:func:`gdal.BuildVRT` / :func:`gdal.Warp`) does not
    reproject — it assumes all sources share a CRS. This helper warps any source
    whose CRS differs from the target onto an in-memory warped VRT first, so
    tiles from different CRSs (e.g. neighbouring UTM zones) mosaic correctly.

    Each source is opened exactly once; the open handle is reused both to read
    the source CRS and (when no reproject is needed) as the compositor input, so
    no path is opened twice — relevant under Requester-Pays where each open is
    billable.

    Args:
        src_paths: Source raster paths.
        dst_crs: Target CRS as an EPSG code or CRS string. ``None`` keeps the
            previous behaviour — sources are returned untouched when they all
            share a CRS, and only reprojected (onto the first source's CRS) when
            they are found to disagree.
        resampling: Resampling method used when reprojecting a mismatched-CRS
            source — one of :data:`pyramids.base._utils.INTERPOLATION_METHODS`
            (``"nearest neighbor"`` (default), ``"bilinear"``, ``"cubic"``).
            Unused when no source is reprojected.

    Returns:
        tuple[list, list]: ``(sources, keepalive)``. ``sources`` is the
            per-source input to feed the compositor — open
            :class:`gdal.Dataset` handles (warped VRTs for reprojected sources,
            plain opens otherwise). ``keepalive`` holds the same datasets so the
            caller keeps them referenced (and prevents GDAL from freeing them)
            until the mosaic is built.

    Raises:
        ValueError: ``dst_crs`` (or ``resampling``) could not be parsed, or a
            source carries no CRS.
        RuntimeError: A source could not be opened, or a reprojecting
            :func:`gdal.Warp` failed.
    """
    if resampling not in INTERPOLATION_METHODS:
        raise ValueError(
            f"resampling must be one of {sorted(INTERPOLATION_METHODS)}, "
            f"got {resampling!r}."
        )

    # Open each source once; read its CRS from that same handle.
    opened: list = []
    source_srs: list[osr.SpatialReference] = []
    for path in src_paths:
        dataset = gdal.Open(path)
        if dataset is None:
            raise RuntimeError(f"gdal.Open returned None for source {path!r}.")
        wkt = dataset.GetProjection()
        if not wkt:
            raise ValueError(
                f"source {path!r} has no CRS; every source must carry a CRS to "
                "be merged/reprojected."
            )
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        opened.append(dataset)
        source_srs.append(srs)

    target_srs = _as_srs(dst_crs) if dst_crs is not None else None
    if target_srs is None:
        # Auto-detect: no reproject when every source already shares a CRS.
        disagree = any(not source_srs[0].IsSame(other) for other in source_srs[1:])
        if not disagree:
            return opened, opened
        target_srs = source_srs[0]

    # At least one source needs reprojecting (or dst_crs forces a target). Feed
    # the compositor open datasets uniformly — gdal.BuildVRT rejects a mix of
    # path strings and dataset objects.
    target_wkt = target_srs.ExportToWkt()
    resample_alg = INTERPOLATION_METHODS[resampling]
    sources: list = []
    for dataset, srs in zip(opened, source_srs):
        if srs.IsSame(target_srs):
            sources.append(dataset)
            continue
        warped = gdal.Warp(
            "",
            dataset,
            options=gdal.WarpOptions(
                format="VRT", dstSRS=target_wkt, resampleAlg=resample_alg
            ),
        )
        if warped is None:
            raise RuntimeError(
                f"gdal.Warp returned None reprojecting a source to the target CRS."
            )
        sources.append(warped)
    return sources, sources


def _merge_reduce(
    src_paths: list,
    dst: str,
    method: str,
    no_data_value: float | int | str,
    n: float | int | str,
) -> None:
    """Merge sources by reducing overlapping pixels with min/max/sum.

    Each source is warped onto the union grid (from a scratch
    :func:`gdal.BuildVRT`), stacked, and reduced with a NaN-aware numpy reducer.
    Pixels with no source coverage are written as ``no_data_value``.

    Args:
        src_paths: Source rasters as path strings or already-open
            :class:`gdal.Dataset` objects (e.g. reprojected warped VRTs from
            :func:`_prepare_sources`).
        dst: Output raster path.
        method: One of ``"min"``, ``"max"``, ``"sum"``.
        no_data_value: Output no-data value and no-coverage fill.
        n: Source pixel value to treat as no-data (``"nan"`` means none).

    Raises:
        RuntimeError: GDAL failed to build the union mosaic for the sources.
    """
    template = gdal.BuildVRT("", src_paths)
    if template is None:
        raise RuntimeError(
            f"gdal.BuildVRT returned None for sources {src_paths!r}; "
            "check that all paths are readable rasters with consistent "
            "band counts and CRS."
        )
    geotransform = template.GetGeoTransform()
    projection = template.GetProjection()
    x_size, y_size = template.RasterXSize, template.RasterYSize
    band_count = template.RasterCount
    template = None

    output_bounds = [
        geotransform[0],
        geotransform[3] + geotransform[5] * y_size,
        geotransform[0] + geotransform[1] * x_size,
        geotransform[3],
    ]
    src_nodata = None if str(n).lower() == "nan" else float(n)

    layers = []
    for path in src_paths:
        warp_opts = gdal.WarpOptions(
            format="MEM",
            outputBounds=output_bounds,
            width=x_size,
            height=y_size,
            srcNodata=src_nodata,
            dstNodata=float("nan"),
        )
        warped = gdal.Warp("", path, options=warp_opts)
        array = warped.ReadAsArray().astype("float64")
        if array.ndim == 2:
            array = array[np.newaxis, ...]
        layers.append(array)
        warped = None

    cube = np.stack(layers)
    coverage = np.count_nonzero(~np.isnan(cube), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if method == "min":
            reduced = np.nanmin(cube, axis=0)
        elif method == "max":
            reduced = np.nanmax(cube, axis=0)
        else:
            reduced = np.nansum(cube, axis=0)

    fill = float(no_data_value)
    reduced = np.where(coverage == 0, fill, reduced)

    out_ds = gdal.GetDriverByName("GTiff").Create(
        dst, x_size, y_size, band_count, gdal.GDT_Float64, options=["COMPRESS=LZW"]
    )
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    for band_index in range(band_count):
        out_band = out_ds.GetRasterBand(band_index + 1)
        out_band.WriteArray(reduced[band_index])
        out_band.SetNoDataValue(fill)
    out_ds.FlushCache()
    out_ds = None


def stack_bands(
    files: list[str | Path],
    *,
    band_names: list[str] | None = None,
    align: bool = False,
    no_data_value: Any = _INHERIT_NO_DATA,
    path: str | Path | None = None,
    signer: Any = None,
) -> Dataset:
    """Stack N single-band rasters into one multi-band :class:`Dataset`.

    Free-function alias for :meth:`pyramids.dataset.Dataset.from_band_files`
    — see that method for the full contract, edge cases, and examples.

    Args:
        files: Single-band raster paths/URLs to stack (order = band order).
        band_names: Explicit per-band names; ``None`` derives them from the
            file names.
        align: When ``True``, resample mismatched inputs onto ``files[0]``'s
            grid instead of raising :class:`~pyramids.base._errors.AlignmentError`.
        no_data_value: No-data value for the output bands; omitted means
            "inherit from the source rasters".
        path: Output ``.tif`` path; ``None`` keeps the result in memory.
        signer: Optional signer exposing ``sign_href(str) -> str`` and
            ``gdal_env() -> dict[str, str]`` (e.g. a
            :class:`pyramids.stac.signers.Signer`). When given, **both** hooks
            are applied (as in :func:`pyramids.stac.load_asset`): every input
            href is rewritten through ``signer.sign_href`` first, then
            ``signer.gdal_env()`` is installed via
            :class:`~pyramids.base.remote.CloudConfig` for the duration of the
            stack, so authenticated cloud inputs (URL-signed or env-credentialed)
            read with the right credentials. ``None`` (default) leaves behaviour
            unchanged.

    Returns:
        Dataset: A multi-band dataset, one band per input file.
    """
    if signer is not None:
        files = [signer.sign_href(str(f)) for f in files]
    with _cloud_config(signer):
        result = Dataset.from_band_files(
            files,
            band_names=band_names,
            align=align,
            no_data_value=no_data_value,
            path=path,
        )
    return result
