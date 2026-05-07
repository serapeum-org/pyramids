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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osgeo import gdal


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


def validate(path: str | Path, strict: bool = False) -> ValidationReport:
    """Validate that the file at `path` is a valid Cloud Optimized GeoTIFF.

    Delegates to `osgeo_utils.samples.validate_cloud_optimized_geotiff`
    when available (GDAL ≥ 3.4). Falls back to a minimal in-house check
    (tiled + overviews) when the import fails.

    Args:
        path: Local path or `/vsi*` VSI path.
        strict: If `True`, warnings are promoted to errors.

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
    _raise_if_missing(p)

    try:
        errors, warnings, details = _osgeo_validate(p)
    except ImportError:  # pragma: no cover — osgeo_utils is a hard dep of GDAL
        errors, warnings, details = _fallback_validate(p)

    if strict:
        errors = list(errors) + list(warnings)
        warnings = []

    return ValidationReport(
        is_valid=not errors,
        errors=list(errors),
        warnings=list(warnings),
        details=dict(details),
    )
