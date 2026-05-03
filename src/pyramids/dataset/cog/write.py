"""Write a gdal.Dataset to disk as a Cloud Optimized GeoTIFF.

Thin wrapper around `gdal.GetDriverByName("COG").CreateCopy` that
normalizes and validates options via :mod:`pyramids.dataset.cog.options`
before handing off. Intended to be called from
:meth:`pyramids.dataset._collaborators.COG.to_cog`; callers own the
dataset's lifecycle.
"""

from __future__ import annotations

import logging
from pathlib import Path

from osgeo import gdal

from pyramids.base._errors import DriverNotExistError, FailedToSaveError
from pyramids.dataset.cog.options import (
    CreationOptions,
    to_gdal_options,
    validate_option_keys,
)

logger = logging.getLogger(__name__)


def translate_to_cog(
    src: gdal.Dataset,
    path: str | Path,
    options: CreationOptions,
) -> gdal.Dataset:
    """Write a :class:`gdal.Dataset` to disk as a Cloud Optimized GeoTIFF.

    Runs `gdal.GetDriverByName("COG").CreateCopy` after normalizing
    and validating `options`. The source dataset is flushed before
    `CreateCopy`; the destination dataset is returned unflushed so
    the caller can arrange its lifecycle.

    Args:
        src: Source :class:`gdal.Dataset` (in-memory or on-disk). Will
            be flushed before :func:`gdal.Driver.CreateCopy`. Caller
            retains ownership.
        path: Destination file path. Parent directory must exist.
        options: Mapping of COG driver options. Keys are normalized to
            upper case and validated against
            :data:`~pyramids.dataset.cog.options.COG_DRIVER_OPTIONS`.

    Returns:
        gdal.Dataset: The newly written dataset. The caller is
        responsible for calling :meth:`FlushCache` and releasing the
        reference (`dst = None`).

    Raises:
        DriverNotExistError: The GDAL build lacks the COG driver.
        FileNotFoundError: The parent directory of `path` does not exist.
        ValueError: An option key is not in
            :data:`~pyramids.dataset.cog.options.COG_DRIVER_OPTIONS`.
        FailedToSaveError: :func:`gdal.Driver.CreateCopy` returned
            `None` or raised :class:`RuntimeError`.

    Examples:
        - Write a minimal COG from an in-memory source:
            ```python
            >>> from osgeo import gdal  # doctest: +SKIP
            >>> src = gdal.GetDriverByName("MEM").Create("", 256, 256, 1, gdal.GDT_Float32)  # doctest: +SKIP
            >>> dst = translate_to_cog(src, "out.tif", {"COMPRESS": "DEFLATE"})  # doctest: +SKIP
            >>> dst.FlushCache()  # doctest: +SKIP
            >>> dst = None  # doctest: +SKIP

            ```
        - Override the default blocksize and compression level:
            ```python
            >>> dst = translate_to_cog(  # doctest: +SKIP
            ...     src,
            ...     "out.tif",
            ...     {"COMPRESS": "ZSTD", "LEVEL": 18, "BLOCKSIZE": 256},
            ... )
            >>> dst.FlushCache()  # doctest: +SKIP

            ```
        - Unknown option keys are rejected up-front:
            ```python
            >>> translate_to_cog(src, "out.tif", {"NONSENSE": "x"})  # doctest: +SKIP
            Traceback (most recent call last):
                ...
            ValueError: Unknown COG driver option(s): ['NONSENSE']...

            ```
    """
    driver = gdal.GetDriverByName("COG")
    if driver is None:
        raise DriverNotExistError(
            "GDAL build does not include the COG driver. "
            "Install GDAL >= 3.4 via conda-forge (`pixi install -e dev`)."
        )

    path = Path(path)
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")

    validate_option_keys(options)
    gdal_opts = to_gdal_options(options)

    logger.debug("translate_to_cog: %s -> %s", src.GetDescription(), path)
    src.FlushCache()

    dst: gdal.Dataset | None
    try:
        dst = driver.CreateCopy(str(path), src, 0, options=gdal_opts)
    except RuntimeError as exc:
        raise FailedToSaveError(
            f"GDAL COG CreateCopy failed for {path}: {exc}"
        ) from exc

    if dst is None:
        raise FailedToSaveError(f"GDAL COG CreateCopy returned None for {path}")
    return dst
