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
from typing import TYPE_CHECKING

from pyramids._io import archive_dir_vsi, archive_members
from pyramids.base._errors import FileFormatNotSupportedError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyramids.dataset import Dataset
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


def _cache_dir(subdir: str = "naturalearth") -> Path:
    """Return the directory used to cache downloaded basemap data.

    Honors the ``PYRAMIDS_CACHE_DIR`` environment variable; otherwise defaults to
    ``~/.pyramids/{subdir}``. The directory is created if missing.

    Args:
        subdir: Cache sub-directory under the pyramids cache root. Defaults to
            ``"naturalearth"`` (the vector layers); :func:`relief` uses ``"relief"``.
    """
    override = os.environ.get("PYRAMIDS_CACHE_DIR")
    root = Path(override) if override else Path.home() / ".pyramids"
    cache = root / subdir
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
    fd, tmp_name = tempfile.mkstemp(suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        # scheme restricted to http(s) by the guard above
        with urllib.request.urlopen(request) as response:  # noqa: S310  # nosec B310
            with open(tmp_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
        tmp_path.replace(destination)
    except OSError as exc:
        # Don't leave a partial/empty temp archive behind on a failed download.
        tmp_path.unlink(missing_ok=True)
        raise OSError(
            f"failed to download basemap data from {url!r}: {exc}. Check the "
            "network connection, or download the file manually into the cache "
            f"directory ({destination.parent})."
        ) from exc


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
        FileNotFoundError: The cached archive contains no ``.shp`` member.

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

    # Local import breaks the import cycle pyramids.feature.collection ->
    # pyramids.basemap (add_basemap) -> pyramids.basemap.features -> pyramids.feature.
    from pyramids.feature import FeatureCollection

    archive = _ensure_cached(layer, resolution)
    # Pick the shapefile from the archive by listing its members rather than assuming
    # a fixed name, so the read is robust to Natural Earth archive-layout differences.
    # Prefer the conventional ``ne_{res}_{name}.shp`` stem when present.
    shp_members = archive_members(archive_dir_vsi(archive, "zip"), "*.shp")
    preferred = f"{_dataset_stem(layer, resolution)}.shp"
    member = preferred if preferred in shp_members else shp_members[0]
    result = FeatureCollection.read_file(f"{archive}/{member}")
    return result


# Pre-baked, downsampled cross-blended hypsometric tint (3-band RGB, EPSG:4326),
# derived from the public-domain Natural Earth ``HYP_50M_SR_W`` raster and hosted on
# the pyramids release assets. The native Natural Earth raster is ~97 MB; these are
# small downsamples (sub-MB / low-MB) meant only as a stock_img-style backdrop.
# Keyed by ``resolution`` -> release-asset filename.
_RELIEF_PRODUCTS: dict[str, str] = {
    "low": "ne_hypso_rgb_720x360.tif",  # ~0.5 degree (720 x 360)
    "medium": "ne_hypso_rgb_1440x720.tif",  # ~0.25 degree (1440 x 720)
}

_RELIEF_BASE_URL = (
    "https://github.com/serapeum-org/pyramids/releases/download/basemap-data-v1"
)


def available_relief_resolutions() -> list[str]:
    """List the resolutions :func:`relief` accepts, coarsest first.

    Returns:
        The supported ``resolution`` values for :func:`relief`.

    Examples:
        - The relief backdrop ships in two downsampled sizes:
            ```python
            >>> from pyramids.basemap.features import available_relief_resolutions
            >>> available_relief_resolutions()
            ['low', 'medium']

            ```
    """
    # Derive from the product table so the two can't drift (insertion order is
    # already coarsest-first: low, medium).
    return list(_RELIEF_PRODUCTS)


def _relief_url(resolution: str) -> str:
    """Build the release-asset download URL for a relief ``resolution``."""
    return f"{_RELIEF_BASE_URL}/{_RELIEF_PRODUCTS[resolution]}"


def relief(resolution: str = "low") -> Dataset:
    """Return a low-res global cross-blended hypsometric relief raster.

    A small RGB hypsometric-tint-with-shaded-relief image (3-band, EPSG:4326) — the
    offline analogue of cartopy's ``GeoAxes.stock_img()``. The image is downsampled
    from public-domain Natural Earth data, downloaded from the pyramids release assets
    on first use and cached locally (see :func:`_cache_dir`), so it works offline
    afterward — mirroring :func:`natural_earth`.

    Unlike the XYZ tile basemaps (Web-Mercator only), this returns a reprojectable
    EPSG:4326 raster, so it can back maps in other projections. Drawing/placement
    (z-order, extent, globe transforms) is left to the visualization layer.

    Args:
        resolution: One of :func:`available_relief_resolutions` — ``"low"`` (~0.5
            degree, the default) or ``"medium"`` (~0.25 degree).

    Returns:
        A :class:`~pyramids.dataset.Dataset` (3-band RGB, EPSG:4326), ready to draw or
        to reproject with :meth:`~pyramids.dataset.Dataset.to_crs`.

    Raises:
        ValueError: ``resolution`` is not supported.
        OSError: The raster is not cached and the download failed, or a cached file
            could not be read (it is removed so the next call re-fetches).

    Examples:
        - Unknown resolutions are rejected with the list of valid values:
            ```python
            >>> from pyramids.basemap.features import relief
            >>> try:
            ...     relief("high")
            ... except ValueError as exc:
            ...     print("unknown relief resolution" in str(exc))
            True

            ```
        - Fetch the backdrop (downloads on first use, then reads from cache):
            ```python
            >>> from pyramids.basemap.features import relief
            >>> ds = relief("low")  # doctest: +SKIP
            >>> ds.band_count, ds.epsg  # doctest: +SKIP
            (3, 4326)

            ```
    """
    if resolution not in _RELIEF_PRODUCTS:
        raise ValueError(
            f"unknown relief resolution {resolution!r}; choose from "
            f"{available_relief_resolutions()}."
        )

    # Local import breaks the import cycle pyramids.dataset -> pyramids.basemap.
    from pyramids.dataset import Dataset

    cached = _cache_dir("relief") / _RELIEF_PRODUCTS[resolution]
    if not cached.exists():
        _download(_relief_url(resolution), cached)
    try:
        result = Dataset.read_file(str(cached))
    except (RuntimeError, ValueError, OSError, FileFormatNotSupportedError) as exc:
        # These are the failures that mean the cached bytes are not a readable raster
        # — a truncated download or an HTML error page served with HTTP 200. GDAL
        # surfaces an unopenable file as RuntimeError; read_file wraps some cases as
        # FileFormatNotSupportedError / FileNotFoundError. Drop the file so the next
        # call re-fetches, instead of failing forever on a poisoned cache. Anything
        # else (e.g. a TypeError from a programming bug) is not a corrupt-cache signal
        # and is left to propagate unchanged.
        cached.unlink(missing_ok=True)
        raise OSError(
            f"cached relief raster {cached} could not be read ({exc}); removed it — "
            "retry to re-download."
        ) from exc
    return result
