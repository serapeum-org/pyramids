"""Natural Earth vector map features as pyramids :class:`FeatureCollection` objects.

``pyramids.basemap`` otherwise provides only XYZ *tile imagery* (via
:func:`pyramids.basemap.add_basemap`). Coastlines, country borders, land, ocean,
rivers and lakes are *vector* data, not tiles. This module fetches the requested
Natural Earth layer (https://www.naturalearthdata.com) on demand, caches the
downloaded archive under a local cache directory, and reads it into a
:class:`~pyramids.feature.FeatureCollection` through GDAL's ``/vsizip/`` virtual
filesystem.

Downloads use the Python standard library (:mod:`urllib.request`) — no new
third-party dependency. The cache directory defaults to ``~/.pyramids/naturalearth``
and can be overridden with the ``PYRAMIDS_CACHE_DIR`` environment variable.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from pyramids.feature import FeatureCollection

# layer name -> (Natural Earth category, dataset suffix). The dataset stem is
# ``ne_{resolution}_{suffix}`` and the category selects the CDN sub-path.
_LAYERS: dict[str, tuple[str, str]] = {
    "coastline": ("physical", "coastline"),
    "land": ("physical", "land"),
    "ocean": ("physical", "ocean"),
    "rivers": ("physical", "rivers_lake_centerlines"),
    "lakes": ("physical", "lakes"),
    "borders": ("cultural", "admin_0_boundary_lines_land"),
}

_RESOLUTIONS = ("110m", "50m", "10m")

_BASE_URL = "https://naciscdn.org/naturalearth"


def available_layers() -> list[str]:
    """List the Natural Earth layer names :func:`natural_earth` accepts.

    Returns:
        Sorted list of supported ``layer`` values.

    Examples:
        - Check which layers are available:
            ```python
            >>> from pyramids.basemap.features import available_layers
            >>> "coastline" in available_layers()
            True
            >>> "borders" in available_layers()
            True

            ```
    """
    return sorted(_LAYERS.keys())


def available_resolutions() -> list[str]:
    """List the Natural Earth resolutions :func:`natural_earth` accepts.

    Returns:
        The supported ``resolution`` values, coarsest first.

    Examples:
        - Resolutions follow Natural Earth's scale naming:
            ```python
            >>> from pyramids.basemap.features import available_resolutions
            >>> available_resolutions()
            ['110m', '50m', '10m']

            ```
    """
    return list(_RESOLUTIONS)


def _dataset_stem(layer: str, resolution: str) -> str:
    """Return the Natural Earth dataset stem, e.g. ``ne_110m_coastline``."""
    _, suffix = _LAYERS[layer]
    return f"ne_{resolution}_{suffix}"


def _download_url(layer: str, resolution: str) -> str:
    """Build the Natural Earth CDN download URL for a layer/resolution."""
    category, _ = _LAYERS[layer]
    stem = _dataset_stem(layer, resolution)
    return f"{_BASE_URL}/{resolution}/{category}/{stem}.zip"


def _cache_dir() -> Path:
    """Return the directory used to cache downloaded Natural Earth archives.

    Honors the ``PYRAMIDS_CACHE_DIR`` environment variable; otherwise defaults to
    ``~/.pyramids/naturalearth``. The directory is created if missing.
    """
    override = os.environ.get("PYRAMIDS_CACHE_DIR")
    root = Path(override) if override else Path.home() / ".pyramids"
    cache = root / "naturalearth"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` atomically via a temporary file.

    Raises:
        ValueError: ``url`` is not an HTTP(S) URL.
        OSError: The download failed (network error, HTTP error, etc.). The original
            error is chained so the cause is preserved.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"refusing to download from non-HTTP(S) URL: {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": "pyramids-gis"})
    try:
        # scheme restricted to http(s) by the guard above
        with urllib.request.urlopen(request) as response:  # noqa: S310  # nosec B310
            fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=str(destination.parent))
            os.close(fd)
            tmp_path = Path(tmp_name)
            with open(tmp_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
    except OSError as exc:
        raise OSError(
            f"failed to download Natural Earth data from {url!r}: {exc}. Check the "
            "network connection, or download the archive manually into the cache "
            f"directory ({destination.parent})."
        ) from exc
    tmp_path.replace(destination)


def _ensure_cached(layer: str, resolution: str) -> Path:
    """Return the local cached zip for a layer/resolution, downloading if absent."""
    stem = _dataset_stem(layer, resolution)
    archive = _cache_dir() / f"{stem}.zip"
    if not archive.exists():
        _download(_download_url(layer, resolution), archive)
    return archive


def natural_earth(
    layer: str = "coastline",
    resolution: str = "110m",
) -> FeatureCollection:
    """Load a Natural Earth vector layer as a pyramids :class:`FeatureCollection`.

    The layer is downloaded from the Natural Earth CDN on first use and cached
    locally (see :func:`_cache_dir`); subsequent calls read straight from the cache.

    Args:
        layer: One of :func:`available_layers` — ``"coastline"``, ``"borders"``,
            ``"land"``, ``"ocean"``, ``"rivers"``, or ``"lakes"``.
        resolution: One of :func:`available_resolutions` — ``"110m"`` (coarsest),
            ``"50m"``, or ``"10m"`` (finest).

    Returns:
        A :class:`~pyramids.feature.FeatureCollection` of the layer geometry, ready to
        draw (e.g. via cleopatra) without any GIS code in the caller.

    Raises:
        ValueError: ``layer`` or ``resolution`` is not supported.
        OSError: The layer is not cached and the download failed.

    Examples:
        - Unknown layers are rejected with the list of valid names:
            ```python
            >>> from pyramids.basemap.features import natural_earth
            >>> try:
            ...     natural_earth("countries")
            ... except ValueError as exc:
            ...     print("unknown Natural Earth layer" in str(exc))
            True

            ```
        - Fetch coastlines (downloads on first use, then reads from cache):
            ```python
            >>> from pyramids.basemap.features import natural_earth
            >>> fc = natural_earth("coastline", resolution="110m")  # doctest: +SKIP
            >>> len(fc)  # doctest: +SKIP
            134

            ```
    """
    if layer not in _LAYERS:
        raise ValueError(
            f"unknown Natural Earth layer {layer!r}; choose from {available_layers()}."
        )
    if resolution not in _RESOLUTIONS:
        raise ValueError(
            f"unknown Natural Earth resolution {resolution!r}; choose from "
            f"{list(_RESOLUTIONS)}."
        )

    archive = _ensure_cached(layer, resolution)
    stem = _dataset_stem(layer, resolution)
    result = FeatureCollection.read_file(f"{archive}/{stem}.shp")
    return result
