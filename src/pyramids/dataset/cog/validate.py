"""COG validation wrapping `osgeo_utils` sample validator.

Provides :func:`validate` — a thin wrapper over
`osgeo_utils.samples.validate_cloud_optimized_geotiff.validate` (GDAL
ships it as a "sample"; the signature has drifted between GDAL 3.4 /
3.6 / 3.8 / 3.12, so we defensively probe the return shape). If the
import fails entirely, a minimal in-house fallback checks that the file
is tiled and has overviews.

Returns a :class:`ValidationReport` — a frozen dataclass usable as a
:class:`bool` (`is_valid`) with `errors`, `warnings`, and
`details` fields for richer reporting.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from osgeo import gdal

from pyramids.dataset.cog.options import COG_READ_DEFAULTS


@contextmanager
def config_context(config: dict[str, str] | None) -> Iterator[None]:
    """Apply GDAL config options for the duration of the block.

    Prefers :func:`gdal.config_options` (GDAL >= 3.3) and falls back to manually
    setting and restoring each option on older builds, so callers never need to
    probe for the context manager themselves.

    Args:
        config: Mapping of GDAL config option names to values, or ``None`` /
            empty to apply nothing.

    Yields:
        None: control returns to the ``with`` body with the options applied.
    """
    if not config:
        yield
        return
    options_cm = getattr(gdal, "config_options", None)
    if options_cm is not None:
        with options_cm(config):
            yield
        return
    previous = {key: gdal.GetConfigOption(key, None) for key in config}
    try:
        for key, value in config.items():
            gdal.SetConfigOption(key, value)
        yield
    finally:
        for key, old in previous.items():
            gdal.SetConfigOption(key, old)


_REMOTE_PREFIXES: tuple[str, ...] = (
    "/vsicurl",
    "/vsis3",
    "/vsigs",
    "/vsiaz",
    "/vsioss",
    "/vsiswift",
    "http://",
    "https://",
)


def _is_remote(path: str) -> bool:
    """Return ``True`` for a remote / network-backed path.

    Args:
        path: A local path or ``/vsi*`` path.

    Returns:
        bool: ``True`` for ``/vsicurl``/cloud-VSI/HTTP(S) paths.
    """
    return path.startswith(_REMOTE_PREFIXES) or "://" in path


def _resolve_read_config(
    path: str, config: dict[str, str] | None
) -> dict[str, str] | None:
    """Pick the GDAL config to apply for a read.

    Args:
        path: The path being read.
        config: An explicit config, or ``None``.

    Returns:
        ``config`` when given; otherwise
        :data:`~pyramids.dataset.cog.options.COG_READ_DEFAULTS` for remote
        paths, else ``None``.
    """
    if config is not None:
        return config
    return dict(COG_READ_DEFAULTS) if _is_remote(path) else None


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating whether a file is a Cloud Optimized GeoTIFF.

    Attributes:
        is_valid: `True` iff :attr:`errors` is empty (and, under
            `strict=True`, no warnings either).
        errors: Error messages (empty when valid).
        warnings: Non-fatal warnings (e.g., "no overviews").
        details: Structural metadata from the validator — typically
            `ifd_offsets`, `data_offsets`, and, in the fallback
            path, `blocksize` and `overview_count`.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Truthy iff the file validates as a COG.

        Examples:
            - A valid report is truthy:
                ```python
                >>> bool(ValidationReport(is_valid=True))
                True

                ```
            - An invalid report (with errors) is falsy:
                ```python
                >>> bool(ValidationReport(is_valid=False, errors=["bad"]))
                False

                ```
            - The report is usable directly in conditionals:
                ```python
                >>> report = ValidationReport(is_valid=True, details={"blocksize": [512, 512]})
                >>> "OK" if report else "bad"
                'OK'
                >>> report.details["blocksize"]
                [512, 512]

                ```
        """
        return self.is_valid


def _raise_if_missing(path: str) -> None:
    """Raise :class:`FileNotFoundError` if `path` does not resolve.

    Uses :func:`Path.exists` for local paths and :func:`gdal.VSIStatL`
    for `/vsi*` paths — both locale-independent. This replaces the
    previous substring matching against GDAL's error message, which
    was brittle across GDAL versions and non-English locales.

    Args:
        path: Local path or `/vsi*` path to probe.

    Raises:
        FileNotFoundError: When the path cannot be resolved.

    Examples:
        - An existing local file returns `None` silently:
            ```python
            >>> import os, tempfile, pathlib
            >>> fd, name = tempfile.mkstemp(suffix=".txt")
            >>> os.close(fd)
            >>> p = pathlib.Path(name)
            >>> _ = p.write_text("hi")
            >>> _raise_if_missing(str(p)) is None
            True
            >>> p.unlink()

            ```
        - A missing local file raises `FileNotFoundError`:
            ```python
            >>> try:
            ...     _raise_if_missing("definitely-does-not-exist-12345.tif")
            ... except FileNotFoundError as exc:
            ...     print("missing:", exc)
            missing: definitely-does-not-exist-12345.tif

            ```
        - `/vsi*` paths delegate to `gdal.VSIStatL`:
            ```python
            >>> try:
            ...     _raise_if_missing("/vsimem/unreachable_doctest_xyz.tif")
            ... except FileNotFoundError as exc:
            ...     print("missing:", exc)
            missing: /vsimem/unreachable_doctest_xyz.tif

            ```
    """
    if path.startswith("/vsi"):
        stat = gdal.VSIStatL(path)
        if stat is None:
            raise FileNotFoundError(path)
    elif not Path(path).exists():
        raise FileNotFoundError(path)


def _osgeo_validate(
    path: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Invoke the osgeo_utils sample validator; return `(errors, warnings, details)`.

    The sample validator's signature has drifted across GDAL versions.
    We probe defensively: GDAL 3.6+ returns
    `(warnings, errors, details)` while older builds may return just
    `(warnings, errors)`.

    Args:
        path: File path or `/vsi*` path.

    Returns:
        Tuple of `(errors, warnings, details)` — errors listed first
        to match this module's public convention.

    Raises:
        ImportError: The `osgeo_utils` sample module is unavailable.
        FileNotFoundError: The underlying file cannot be opened
            (raised via `ValidateCloudOptimizedGeoTIFFException`).
    """
    from osgeo_utils.samples import validate_cloud_optimized_geotiff as v

    # Structural pre-check before invoking the validator — avoids
    # depending on GDAL's error-message phrasing (which varies by
    # version and locale) to detect "file not found".
    _raise_if_missing(path)

    try:
        result = v.validate(path, full_check=True)
    except v.ValidateCloudOptimizedGeoTIFFException as exc:
        # If a ValidateCloudOptimizedGeoTIFFException escapes despite
        # the pre-check, it's not about a missing file — surface it
        # as a validation error rather than letting it propagate.
        return [str(exc)], [], {}
    except RuntimeError as exc:
        # Same rationale as above for RuntimeErrors from gdal.Open
        # inside the sample validator (locale-independent fallback).
        return [str(exc)], [], {}

    errors: list[str]
    warnings: list[str]
    details: dict[str, Any]
    if len(result) == 3:
        warnings, errors, details = result
    else:  # pragma: no cover — defensive; older GDAL
        warnings, errors = result
        details = {}
    return list(errors), list(warnings), dict(details)


def _fallback_validate(
    path: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Minimal in-house validator used when the sample module is unavailable.

    Checks: file opens; image is tiled (block dimensions smaller than
    full extent); at least one overview present. Does NOT check the
    IFD-before-data layout; recommends upgrading GDAL if used.

    Heuristic limitations:
        The "is stripped" check compares the block shape reported by
        :func:`GetBlockSize` — stripped TIFFs typically return
        `(width, small_N)` (e.g. `(512, 4)`) while tiled files
        return `(tile, tile)`. The rule used is `by!= bx and
        by * 4 < bx`, which:

        - Correctly flags standard stripped layouts (`(W, 1)`,
          `(W, 4)`, `(W, 8)`).
        - Correctly passes square-tiled COGs (`(256, 256)`,
          `(512, 512)`).
        - Can FALSE-NEGATIVE on pathological cases such as
          near-square strips (`by == bx`) — extremely rare in
          practice.
        - Can FALSE-POSITIVE on legitimately non-square TIFF tiles
          (e.g. `(512, 128)` used for tall elongated rasters) —
          also rare; the GTiff driver requires square tiles for COG.

        The authoritative check is the TIFF `TILEWIDTH` /
        `STRIPBYTECOUNTS` tag, but reading it requires either
        :mod:`tifffile` or a direct `libtiff` binding. We accept
        the heuristic because this fallback runs only when
        :mod:`osgeo_utils.samples.validate_cloud_optimized_geotiff`
        is unavailable — which, in practice, is never on GDAL >= 3.4.

    Args:
        path: File path or `/vsi*` path.

    Returns:
        `(errors, warnings, details)` — same convention as
        :func:`_osgeo_validate`.
    """
    # Locale-independent missing-file pre-check (ARC-6): each validator path
    # owns this so validate() needs no redundant outer call.
    _raise_if_missing(path)
    errors: list[str] = []
    warnings: list[str] = ["using fallback validator; osgeo_utils sample unavailable"]
    details: dict[str, Any] = {}
    ds = gdal.Open(path)
    if ds is None:
        errors.append(f"cannot open {path}")
    else:
        band = ds.GetRasterBand(1)
        bx, by = band.GetBlockSize()
        details["blocksize"] = [bx, by]
        details["overview_count"] = band.GetOverviewCount()
        # See the "Heuristic limitations" note in the docstring.
        is_stripped = by != bx and by * 4 < bx
        if is_stripped:
            errors.append("not tiled (stripped layout)")
        if band.GetOverviewCount() == 0:
            warnings.append("no overviews present")
        ds = None
    return errors, warnings, details


def _probe_overview_layout(path: str) -> dict[str, Any]:
    """Cheaply read the tile size and overview pyramid layout of a raster.

    Metadata-only (no pixel reads). Returns an empty dict when the file cannot
    be opened — callers merge the result into a richer ``details`` dict.

    Args:
        path: Local path or ``/vsi*`` path.

    Returns:
        A dict with ``blocksize`` (``[bx, by]``), ``overview_count`` (int), and
        ``overviews`` (a list of per-level dicts with ``index``, ``width``,
        ``height``, ``blocksize``, ``decimation``). Empty when unreadable.
    """
    out: dict[str, Any] = {}
    try:
        ds = gdal.Open(path)
    except RuntimeError:
        return out
    if ds is None:
        return out
    try:
        band = ds.GetRasterBand(1)
        block_x, block_y = band.GetBlockSize()
        width = ds.RasterXSize
        overviews = []
        for i in range(band.GetOverviewCount()):
            ovr = band.GetOverview(i)
            obx, oby = ovr.GetBlockSize()
            overviews.append(
                {
                    "index": i,
                    "width": ovr.XSize,
                    "height": ovr.YSize,
                    "blocksize": [obx, oby],
                    "decimation": round(width / ovr.XSize) if ovr.XSize else 0,
                }
            )
        out["blocksize"] = [block_x, block_y]
        out["overview_count"] = len(overviews)
        out["overviews"] = overviews
    finally:
        ds = None
    return out


def validate(
    path: str | Path,
    strict: bool = False,
    config: dict[str, str] | None = None,
) -> ValidationReport:
    """Validate that the file at `path` is a valid Cloud Optimized GeoTIFF.

    Delegates to `osgeo_utils.samples.validate_cloud_optimized_geotiff`
    when available (GDAL ≥ 3.4). Falls back to a minimal in-house check
    (tiled + overviews) when the import fails.

    Args:
        path: Local path or `/vsi*` VSI path.
        strict: If `True`, warnings are promoted to errors.
        config: GDAL config options applied (via `gdal.config_options`) for
            the duration of the validation. When `None` and `path` is a
            remote/`/vsicurl` path, the
            :data:`~pyramids.dataset.cog.options.COG_READ_DEFAULTS` are
            applied so remote reads avoid a directory-listing round-trip.

    Returns:
        ValidationReport: Includes `is_valid`, error/warning lists,
        and a `details` dict with structural metadata. Usable as a
        boolean (`bool(report) == report.is_valid`).

    Raises:
        FileNotFoundError: When a *local* `path` does not exist. VSI
            paths are passed through to GDAL, which reports the error
            through the normal validator surface.

    Examples:
        - Validate a local COG and inspect the report:
            ```python
            >>> from pyramids.dataset.cog import validate  # doctest: +SKIP
            >>> report = validate("scene.tif")  # doctest: +SKIP
            >>> bool(report)  # doctest: +SKIP
            True
            >>> report.details.get("blocksize")  # doctest: +SKIP
            [512, 512]

            ```
        - Strict mode promotes warnings (e.g. "no overviews") to errors:
            ```python
            >>> strict = validate("scene.tif", strict=True)  # doctest: +SKIP
            >>> if not strict:  # doctest: +SKIP
            ...     for err in strict.errors: print(err)

            ```
        - Validate a cloud-hosted COG via VSI path:
            ```python
            >>> validate("/vsis3/public-bucket/scene.tif").is_valid  # doctest: +SKIP
            True

            ```
    """
    p = str(path)
    cfg = _resolve_read_config(p, config)
    with config_context(cfg):
        # The missing-file pre-check lives in each validator path
        # (_osgeo_validate / _fallback_validate), so no redundant outer call
        # here (ARC-6).
        try:
            errors, warnings, details = _osgeo_validate(p)
        except ImportError:  # pragma: no cover — osgeo_utils is a hard dep of GDAL
            errors, warnings, details = _fallback_validate(p)
        # Enrich with the per-overview layout (PC-4) so a report is
        # self-sufficient for debugging "why is this slow / not a COG" without
        # a second cog_info() call. Existing validator keys win on conflict.
        details = {**_probe_overview_layout(p), **dict(details)}

    if strict:
        errors = list(errors) + list(warnings)
        warnings = []

    return ValidationReport(
        is_valid=not errors,
        errors=list(errors),
        warnings=list(warnings),
        details=dict(details),
    )
