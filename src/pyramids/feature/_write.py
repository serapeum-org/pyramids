"""I/O writer engine for :class:`~pyramids.feature.FeatureCollection` (ARC-36).

Module-level implementations of the collection's writers (vector file, GeoParquet,
PMTiles / MVT vector tiles), kept out of the god-class as free functions that take
the collection as their first argument. The `FeatureCollection` write methods are
thin facades over these functions.

The underlying geopandas writers are real methods (not accessors), so they are
called unbound as ``gpd.GeoDataFrame.to_file(fc, ...)`` -- equivalent to the old
in-class ``super().to_file(...)`` while bypassing the FeatureCollection facade. The
``list_layers`` cache invalidation (ARC-42) stays in the ``to_file`` facade;
:func:`to_vector_tiles` therefore calls ``fc.to_file`` (the facade) so it still fires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd

from pyramids.base._utils import get_catalog, import_pyarrow

_CATALOG = get_catalog(raster_driver=False)


def to_file(
    fc: Any,
    path: str | Path,
    driver: str = "geojson",
    *,
    layer: str | None = None,
    mode: str = "w",
    **creation_options: Any,
) -> None:
    """Write `fc` to a vector file via the pyogrio engine (raw write; the facade clears the cache)."""
    if mode not in ("w", "a"):
        raise ValueError(f"mode must be 'w' (write) or 'a' (append); got {mode!r}.")
    try:
        resolved = _CATALOG.get_gdal_name(driver) or driver
    except AttributeError:
        resolved = driver
    # Pin the engine to pyogrio to match read_file / iter_features; callers can override
    # via engine="fiona" in creation_options.
    passthrough: dict[str, Any] = {
        "driver": resolved,
        "mode": mode,
        "engine": "pyogrio",
    }
    if layer is not None:
        passthrough["layer"] = layer
    passthrough.update(creation_options)
    gpd.GeoDataFrame.to_file(fc, path, **passthrough)


def to_parquet(
    fc: Any,
    path: str | Path,
    *,
    compression: str = "snappy",
    index: bool | None = None,
    **kwargs: Any,
) -> None:
    """Write `fc` to GeoParquet, raising a pyramids-branded ImportError if pyarrow is absent."""
    import_pyarrow(
        "GeoParquet support requires the optional 'pyarrow' dependency. Install with one of:\n"
        "  - PyPI:        pip install 'pyramids-gis[parquet]'\n"
        "  - conda-forge: conda install -c conda-forge pyramids-parquet"
    )
    gpd.GeoDataFrame.to_parquet(
        fc, path, compression=compression, index=index, **kwargs
    )


def to_vector_tiles(
    fc: Any,
    path: str | Path,
    driver: str,
    *,
    min_zoom: int,
    max_zoom: int | None,
    layer_name: str | None,
    **creation_options: Any,
) -> Path:
    """Write `fc` as a tiled-vector pyramid (PMTiles / MVT) via ``fc.to_file``, returning the path."""
    options = dict(creation_options)
    options["MINZOOM"] = min_zoom
    if max_zoom is not None:
        options["MAXZOOM"] = max_zoom
    # Call the facade (not the raw writer) so the list_layers cache invalidation runs.
    fc.to_file(path, driver=driver, layer=layer_name, **options)
    return Path(path)


def to_pmtiles(
    fc: Any,
    path: str | Path,
    *,
    min_zoom: int = 0,
    max_zoom: int | None = None,
    layer_name: str | None = None,
    **creation_options: Any,
) -> Path:
    """Write `fc` as a single ``.pmtiles`` archive."""
    return to_vector_tiles(
        fc,
        path,
        "PMTiles",
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        layer_name=layer_name,
        **creation_options,
    )


def to_mvt(
    fc: Any,
    path: str | Path,
    *,
    min_zoom: int = 0,
    max_zoom: int | None = None,
    layer_name: str | None = None,
    **creation_options: Any,
) -> Path:
    """Write `fc` as an MVT ``{z}/{x}/{y}.pbf`` tile tree."""
    return to_vector_tiles(
        fc,
        path,
        "MVT",
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        layer_name=layer_name,
        **creation_options,
    )
