"""Structured Cloud Optimized GeoTIFF inspection.

Provides :func:`cog_info` — a GDAL-only, metadata-only inspection of a raster
that answers "what compression / predictor / blocksize / overview pyramid does
this COG use?" without reading any pixels. It depends only on the GDAL Python
bindings pyramids already uses.

The result is a frozen :class:`COGInfo` dataclass carrying the band/geo profile
plus a per-level :class:`OverviewLevel` list. Validity is delegated to
:func:`pyramids.dataset.cog.validate.validate` so :attr:`COGInfo.is_cog` agrees
with :meth:`pyramids.dataset.engines.cog.COG.validate_cog`.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osgeo import gdal

from pyramids.dataset.cog.validate import _resolve_read_config, validate


@dataclass(frozen=True)
class OverviewLevel:
    """One level of a COG's internal overview pyramid.

    Attributes:
        index: Zero-based overview index (0 is the coarsest-to-finest order
            GDAL reports, i.e. index 0 is the first/largest overview).
        width: Overview width in pixels.
        height: Overview height in pixels.
        blocksize: ``(block_x, block_y)`` tile size of this overview.
        decimation: Integer shrink factor relative to full resolution,
            ``round(full_width / width)`` (e.g. ``2``, ``4``, ``8``).
    """

    index: int
    width: int
    height: int
    blocksize: tuple[int, int]
    decimation: int


@dataclass(frozen=True)
class COGInfo:
    """Structured metadata describing a (Cloud Optimized) GeoTIFF.

    Attributes:
        is_cog: ``True`` iff the file validates as a COG (delegated to
            :func:`pyramids.dataset.cog.validate.validate`).
        driver: GDAL driver short name (e.g. ``"GTiff"``).
        width: Full-resolution width in pixels.
        height: Full-resolution height in pixels.
        band_count: Number of raster bands.
        dtype: GDAL data-type name of band 1 (e.g. ``"Float32"``).
        crs_epsg: EPSG code of the CRS, or ``None`` when unresolved.
        bounds: ``(min_x, min_y, max_x, max_y)`` in the raster CRS.
        resolution: ``(pixel_width, pixel_height)`` (both positive).
        compression: ``IMAGE_STRUCTURE`` compression token, or ``None``.
        predictor: ``IMAGE_STRUCTURE`` predictor token, or ``None``.
        interleave: ``IMAGE_STRUCTURE`` interleave token, or ``None``.
        blocksize: ``(block_x, block_y)`` tile size of the full-res image.
        overviews: Per-level overview metadata, finest index first.
        band_tags: Per-band metadata dict keyed by 1-based band index.
        colormap: ``True`` when band 1 carries a colour table.
    """

    is_cog: bool
    driver: str
    width: int
    height: int
    band_count: int
    dtype: str
    crs_epsg: int | None
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    compression: str | None
    predictor: str | None
    interleave: str | None
    blocksize: tuple[int, int]
    overviews: list[OverviewLevel] = field(default_factory=list)
    band_tags: dict[int, dict[str, Any]] = field(default_factory=dict)
    colormap: bool = False

    @property
    def overview_count(self) -> int:
        """Number of overview levels present.

        Returns:
            int: ``len(self.overviews)``.
        """
        return len(self.overviews)


def cog_info(path: str | Path, config: dict[str, str] | None = None) -> COGInfo:
    """Inspect a raster and return its structured COG metadata.

    Reads only headers/metadata (no pixels), so it is cheap even for very large
    or remote (``/vsicurl/``) COGs. Validity is determined by the same validator
    that backs :meth:`pyramids.dataset.engines.cog.COG.validate_cog`.

    Args:
        path: Local path or ``/vsi*`` path to a raster GDAL can open.
        config: GDAL config options applied (via `gdal.config_options`) while
            opening. When `None` and `path` is remote, the
            :data:`~pyramids.dataset.cog.options.COG_READ_DEFAULTS` are applied.

    Returns:
        COGInfo: The structured metadata, including the overview pyramid.

    Raises:
        FileNotFoundError: When ``path`` cannot be opened by GDAL.

    Examples:
        - Inspect a COG and read its compression and overview pyramid:
            ```python
            >>> from pyramids.dataset.cog import cog_info  # doctest: +SKIP
            >>> info = cog_info("scene_cog.tif")  # doctest: +SKIP
            >>> info.compression  # doctest: +SKIP
            'DEFLATE'
            >>> [o.decimation for o in info.overviews]  # doctest: +SKIP
            [2, 4, 8]

            ```
        - A plain (non-COG) GeoTIFF reports ``is_cog=False`` with no overviews:
            ```python
            >>> info = cog_info("plain.tif")  # doctest: +SKIP
            >>> info.is_cog, info.overview_count  # doctest: +SKIP
            (False, 0)

            ```
    """
    p = str(path)
    cfg = _resolve_read_config(p, config)
    with gdal.config_options(cfg) if cfg else nullcontext():
        info = _cog_info_impl(p)
    return info


def _cog_info_impl(p: str) -> COGInfo:
    """Build the :class:`COGInfo` for ``p`` (config context already applied).

    Args:
        p: Local path or ``/vsi*`` path.

    Returns:
        COGInfo: The structured metadata.

    Raises:
        FileNotFoundError: When ``p`` cannot be opened by GDAL.
    """
    try:
        ds = gdal.Open(p)
    except RuntimeError as exc:
        # With gdal.UseExceptions() a missing/unopenable path raises rather
        # than returning None; surface it as FileNotFoundError for callers.
        raise FileNotFoundError(p) from exc
    if ds is None:
        raise FileNotFoundError(p)

    try:
        band0 = ds.GetRasterBand(1)
        struct = ds.GetMetadata("IMAGE_STRUCTURE")
        block_x, block_y = band0.GetBlockSize()
        width, height = ds.RasterXSize, ds.RasterYSize

        gt = ds.GetGeoTransform()
        min_x, max_y = gt[0], gt[3]
        max_x = min_x + gt[1] * width
        min_y = max_y + gt[5] * height

        srs = ds.GetSpatialRef()
        epsg: int | None = None
        if srs is not None:
            code = srs.GetAuthorityCode(None)
            epsg = int(code) if code is not None else None

        overviews: list[OverviewLevel] = []
        for i in range(band0.GetOverviewCount()):
            ovr = band0.GetOverview(i)
            obx, oby = ovr.GetBlockSize()
            decimation = round(width / ovr.XSize) if ovr.XSize else 0
            overviews.append(
                OverviewLevel(
                    index=i,
                    width=ovr.XSize,
                    height=ovr.YSize,
                    blocksize=(obx, oby),
                    decimation=decimation,
                )
            )

        band_tags = {
            i: dict(ds.GetRasterBand(i).GetMetadata())
            for i in range(1, ds.RasterCount + 1)
        }
        info = COGInfo(
            is_cog=validate(p).is_valid,
            driver=ds.GetDriver().ShortName,
            width=width,
            height=height,
            band_count=ds.RasterCount,
            dtype=gdal.GetDataTypeName(band0.DataType),
            crs_epsg=epsg,
            bounds=(min_x, min_y, max_x, max_y),
            resolution=(abs(gt[1]), abs(gt[5])),
            compression=struct.get("COMPRESSION"),
            predictor=struct.get("PREDICTOR"),
            interleave=struct.get("INTERLEAVE"),
            blocksize=(block_x, block_y),
            overviews=overviews,
            band_tags=band_tags,
            colormap=band0.GetColorTable() is not None,
        )
    finally:
        ds = None
    return info
