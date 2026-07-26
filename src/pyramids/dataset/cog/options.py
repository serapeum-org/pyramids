"""COG creation-option types, serialization, and validation.

Provides the :data:`CreationOptions` alias (a `Mapping[str, Any]`), the named
:data:`PROFILES`, and pure-Python helpers used by
:mod:`pyramids.dataset.cog.write` and :class:`pyramids.dataset.engines.cog.COG`:

- :func:`to_gdal_options` — serialize a mapping into GDAL's `['KEY=VALUE',...]` list form.
- :func:`merge_options` — merge defaults with user-supplied extras (dict or legacy `list[str]`).
- :func:`validate_blocksize` — enforce the COG driver's power-of-2-in-[64, 4096] constraint.
- :func:`validate_option_keys` — gate unknown keys against :data:`COG_DRIVER_OPTIONS`.
- :func:`profile_options` / :func:`validate_profile` — named compression presets.

The module has no GDAL dependency — all helpers operate on plain Python
values. GDAL is invoked only at the write call site.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CreationOptions = Mapping[str, Any]
"""Alias for a mapping of GDAL creation-option names to Python values.

Keys are the GDAL option names (case-insensitive; normalized to upper case
during serialization). Values are scalars that stringify cleanly; booleans
are translated to `"YES"`/`"NO"`; `None` entries are dropped.
"""


COG_DRIVER_OPTIONS: frozenset[str] = frozenset(
    {
        "COMPRESS",
        "LEVEL",
        "QUALITY",
        "NUM_THREADS",
        "BLOCKSIZE",
        "BIGTIFF",
        "RESAMPLING",
        "OVERVIEW_RESAMPLING",
        "OVERVIEW_COUNT",
        "OVERVIEW_COMPRESS",
        "OVERVIEW_QUALITY",
        "WARP_RESAMPLING",
        "OVERVIEW_PREDICTOR",
        "PREDICTOR",
        "NBITS",
        "TARGET_SRS",
        "RES",
        "EXTENT",
        "ALIGNED_LEVELS",
        "ADD_ALPHA",
        "TILING_SCHEME",
        "ZOOM_LEVEL",
        "ZOOM_LEVEL_STRATEGY",
        "MAX_Z_ERROR",
        "STATISTICS",
        "GEOTIFF_VERSION",
        "SPARSE_OK",
        "COPY_SRC_MDD",
        "SRC_MDD",
    }
)
"""Whitelist of GDAL COG driver option keys (uppercased)."""


_VALID_BLOCKSIZES: frozenset[int] = frozenset({64, 128, 256, 512, 1024, 2048, 4096})


COG_READ_DEFAULTS: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "0.5",
    "VSI_CACHE": "TRUE",
}
"""GDAL config options that make remote ``/vsicurl/`` COG reads efficient.

Without ``GDAL_DISABLE_READDIR_ON_OPEN`` a remote open issues a directory
listing — often the single biggest latency hit. ``GDAL_HTTP_MULTIRANGE`` lets
GDAL issue the scattered tile ranges a COG read produces as one multi-range
request (``GDAL_HTTP_MERGE_CONSECUTIVE_RANGES`` then coalesces the adjacent
ones), and the retry pair rides out a transient 5xx from object storage instead
of failing the whole read. That budget is deliberately larger than the OGC
discovery one (:data:`pyramids.base._ogc_api.GDAL_HTTP_MAX_RETRY`): a COG read
issues many range requests, so one flaky range should not lose the whole read,
whereas a discovery pre-check is a single request in front of work that has not
started yet. Applied by
:func:`pyramids.dataset.cog.validate.validate` and
:func:`pyramids.dataset.cog.inspect.cog_info` for remote paths when the caller
passes no explicit ``config``. Pure strings (no GDAL dependency here)."""


PROFILES: dict[str, dict[str, Any]] = {
    "deflate": {"COMPRESS": "DEFLATE", "LEVEL": 9},
    "zstd": {"COMPRESS": "ZSTD", "LEVEL": 9},
    "lzw": {"COMPRESS": "LZW"},
    "packbits": {"COMPRESS": "PACKBITS"},
    "jpeg": {"COMPRESS": "JPEG", "QUALITY": 85},
    "webp": {"COMPRESS": "WEBP", "QUALITY": 85},
    "lerc": {"COMPRESS": "LERC", "MAX_Z_ERROR": 0.0},
    "lerc_deflate": {"COMPRESS": "LERC_DEFLATE", "MAX_Z_ERROR": 0.0},
    "lerc_zstd": {"COMPRESS": "LERC_ZSTD", "MAX_Z_ERROR": 0.0},
    "raw": {"COMPRESS": "NONE"},
}
"""Named COG creation profiles (compression presets).

Each profile seeds only the compression-related options; the predictor and
overview resampling are still resolved per-dtype by
:meth:`pyramids.dataset.engines.cog.COG.to_cog`, and explicit kwargs / ``extra``
override the profile. Limited to keys in :data:`COG_DRIVER_OPTIONS`.
"""


_PROFILE_DTYPE_CONSTRAINTS: dict[str, tuple[frozenset[str], tuple[int, int]]] = {
    # profile -> (allowed GDAL dtype names, (min_bands, max_bands))
    "jpeg": (frozenset({"Byte"}), (1, 3)),
    "webp": (frozenset({"Byte"}), (3, 4)),
}
"""Per-profile dtype/band constraints enforced by :func:`validate_profile`."""


def profile_options(name: str) -> dict[str, Any]:
    """Return a copy of the named profile's creation options.

    Args:
        name: Profile name (case-insensitive), e.g. ``"deflate"``, ``"zstd"``,
            ``"jpeg"``.

    Returns:
        A new dict of the profile's options.

    Raises:
        ValueError: When ``name`` is not a known profile.

    Examples:
        - Look up the zstd preset:
            ```python
            >>> profile_options("zstd")
            {'COMPRESS': 'ZSTD', 'LEVEL': 9}

            ```
        - Names are case-insensitive:
            ```python
            >>> profile_options("LZW")
            {'COMPRESS': 'LZW'}

            ```
        - Unknown names are rejected:
            ```python
            >>> profile_options("bogus")  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ...
            ValueError: unknown COG profile 'bogus'...

            ```
    """
    key = name.lower()
    if key not in PROFILES:
        raise ValueError(
            f"unknown COG profile {name!r}; choose from {sorted(PROFILES)}"
        )
    return dict(PROFILES[key])


def validate_profile(name: str, dtype_name: str, band_count: int) -> None:
    """Raise :class:`ValueError` if a source violates a profile's constraints.

    Some profiles only accept specific dtypes / band counts (JPEG: ``Byte``
    with 1-3 bands; WEBP: ``Byte`` with 3-4 bands). Other profiles are
    unconstrained and pass silently.

    Args:
        name: Profile name (case-insensitive).
        dtype_name: GDAL dtype name of the source (e.g. ``"Byte"``,
            ``"Float32"``).
        band_count: Number of bands in the source.

    Raises:
        ValueError: When the source dtype or band count is incompatible.

    Examples:
        - An unconstrained profile always passes:
            ```python
            >>> validate_profile("deflate", "Float32", 4)

            ```
        - JPEG requires Byte with <= 3 bands:
            ```python
            >>> validate_profile("jpeg", "Float32", 1)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ...
            ValueError: jpeg profile requires dtype in...

            ```
    """
    key = name.lower()
    if key not in _PROFILE_DTYPE_CONSTRAINTS:
        return
    allowed_dtypes, (min_bands, max_bands) = _PROFILE_DTYPE_CONSTRAINTS[key]
    if dtype_name not in allowed_dtypes:
        raise ValueError(
            f"{key} profile requires dtype in {sorted(allowed_dtypes)}; "
            f"got {dtype_name}. Use a different profile (e.g. 'deflate')."
        )
    if not (min_bands <= band_count <= max_bands):
        raise ValueError(
            f"{key} profile requires {min_bands}-{max_bands} bands; got {band_count}."
        )


def _stringify(value: Any) -> str:
    """Convert a Python value to the string form GDAL expects.

    Booleans become `"YES"`/`"NO"` (GDAL's convention); everything else
    falls back to :class:`str`.

    Args:
        value: Any Python value.

    Returns:
        The GDAL-style string form.

    Examples:
        - Booleans translate to GDAL's YES/NO convention:
            ```python
            >>> _stringify(True)
            'YES'
            >>> _stringify(False)
            'NO'

            ```
        - Non-bool scalars defer to str():
            ```python
            >>> _stringify(512)
            '512'
            >>> _stringify(3.14)
            '3.14'

            ```
        - Strings are passed through unchanged:
            ```python
            >>> _stringify("DEFLATE")
            'DEFLATE'

            ```
    """
    result: str
    if isinstance(value, bool):
        result = "YES" if value else "NO"
    else:
        result = str(value)
    return result


def to_gdal_options(opts: CreationOptions | None) -> list[str]:
    """Serialize a mapping into GDAL's `['KEY=VALUE',...]` list form.

    Keys are uppercased; values are stringified via :func:`_stringify`
    (booleans become `"YES"`/`"NO"`). `None` values are skipped so
    callers can pass optional kwargs through unchanged.

    Args:
        opts: Mapping of option names to values, or `None`.

    Returns:
        List of `"KEY=VALUE"` strings. Empty list when `opts` is `None`.

    Examples:
        - Serialize a compression config:
            ```python
            >>> to_gdal_options({"COMPRESS": "DEFLATE", "LEVEL": 9})
            ['COMPRESS=DEFLATE', 'LEVEL=9']

            ```
        - Booleans become GDAL's YES/NO convention:
            ```python
            >>> to_gdal_options({"STATISTICS": True, "SPARSE_OK": False})
            ['STATISTICS=YES', 'SPARSE_OK=NO']

            ```
        - None values are dropped so optional kwargs flow through unchanged:
            ```python
            >>> to_gdal_options({"COMPRESS": "LZW", "LEVEL": None})
            ['COMPRESS=LZW']
            >>> to_gdal_options(None)
            []

            ```
    """
    result: list[str]
    if opts is None:
        result = []
    else:
        result = [
            f"{str(k).upper()}={_stringify(v)}"
            for k, v in opts.items()
            if v is not None
        ]
    return result


def _parse_list_extra(items: list[str]) -> dict[str, Any]:
    """Parse `['KEY=VALUE',...]` legacy list form back to a dict.

    Args:
        items: List of `"KEY=VALUE"` strings.

    Returns:
        Dict with uppercased keys and string values (split on first `=`).

    Raises:
        ValueError: If any item lacks an `=`.

    Examples:
        - Parse a multi-entry list:
            ```python
            >>> _parse_list_extra(["COMPRESS=DEFLATE", "LEVEL=9"])
            {'COMPRESS': 'DEFLATE', 'LEVEL': '9'}

            ```
        - Keys are uppercased; values are preserved verbatim:
            ```python
            >>> _parse_list_extra(["compress=lzw"])
            {'COMPRESS': 'lzw'}

            ```
        - Empty input yields an empty dict:
            ```python
            >>> _parse_list_extra([])
            {}

            ```
    """
    parsed: dict[str, Any] = {}
    for entry in items:
        if "=" not in entry:
            raise ValueError(f"creation_options entry missing '=': {entry!r}")
        k, _, v = entry.partition("=")
        parsed[str(k).upper()] = v
    return parsed


def merge_options(
    defaults: CreationOptions,
    extra: CreationOptions | list[str] | None,
) -> dict[str, Any]:
    """Merge default options with user-supplied extras; extras win.

    Accepts `extra` as either a mapping `{'KEY': value}` or the legacy
    list form `['KEY=VALUE',...]` used by
    :meth:`pyramids.dataset.Dataset.to_file`. All keys in the returned
    dict are uppercased; `None` values from either source are dropped.

    Args:
        defaults: Baseline options (typically derived from kwargs in
            :meth:`pyramids.dataset.engines.COG.to_cog`).
        extra: User-provided overrides as a mapping, `list[str]`, or
            `None`.

    Returns:
        New :class:`dict` with all keys uppercased and `None` values
        removed; `extra` entries override `defaults` on conflict.

    Raises:
        ValueError: When a legacy list-form entry lacks `=`.

    Examples:
        - Dict extras override defaults on conflict:
            ```python
            >>> merge_options({"COMPRESS": "DEFLATE"}, {"COMPRESS": "ZSTD"})
            {'COMPRESS': 'ZSTD'}

            ```
        - Legacy list-of-string form is also accepted for back-compat:
            ```python
            >>> merge_options({"COMPRESS": "DEFLATE"}, ["LEVEL=9"])
            {'COMPRESS': 'DEFLATE', 'LEVEL': '9'}

            ```
        - None extras returns a copy of the defaults:
            ```python
            >>> merge_options({"COMPRESS": "DEFLATE"}, None)
            {'COMPRESS': 'DEFLATE'}

            ```
    """
    merged: dict[str, Any] = {
        str(k).upper(): v for k, v in defaults.items() if v is not None
    }
    # `extra is None` -> nothing to merge, keep the defaults.
    if isinstance(extra, list):
        merged.update(_parse_list_extra(extra))
    elif extra is not None:
        merged.update({str(k).upper(): v for k, v in extra.items() if v is not None})
    return merged


def validate_blocksize(value: int) -> None:
    """Raise :class:`ValueError` if `value` is not a valid COG tile size.

    The GDAL COG driver requires `BLOCKSIZE` to be a power of 2 in
    the closed range [64, 4096].

    Args:
        value: Proposed blocksize.

    Raises:
        ValueError: If `value` is outside the allowed set.

    Examples:
        - Valid power-of-2 blocksizes return silently:
            ```python
            >>> validate_blocksize(512)
            >>> validate_blocksize(256)

            ```
        - Non-power-of-2 is rejected:
            ```python
            >>> validate_blocksize(500) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ...
            ValueError: blocksize must be a power of 2 in [64, 4096]; got 500...

            ```
        - Out-of-range values are rejected:
            ```python
            >>> validate_blocksize(32) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ...
            ValueError: blocksize must be a power of 2 in [64, 4096]; got 32...

            ```
    """
    if value not in _VALID_BLOCKSIZES:
        raise ValueError(
            f"blocksize must be a power of 2 in [64, 4096]; got {value}. "
            f"Valid values: {sorted(_VALID_BLOCKSIZES)}"
        )


def validate_option_keys(opts: CreationOptions) -> None:
    """Raise :class:`ValueError` for any key not in :data:`COG_DRIVER_OPTIONS`.

    Keys are compared case-insensitively.

    Args:
        opts: Mapping of option names to values.

    Raises:
        ValueError: If any key is not a recognized COG driver option.

    Examples:
        - Known keys return silently:
            ```python
            >>> validate_option_keys({"COMPRESS": "DEFLATE"})
            >>> validate_option_keys({"BLOCKSIZE": 512, "BIGTIFF": "IF_SAFER"})

            ```
        - Unknown keys raise ValueError naming the offender:
            ```python
            >>> validate_option_keys({"NONSENSE": "x"}) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ...
            ValueError: Unknown COG driver option(s): ['NONSENSE']...

            ```
        - Empty mapping is accepted:
            ```python
            >>> validate_option_keys({})

            ```
    """
    unknown = {str(k).upper() for k in opts.keys()} - COG_DRIVER_OPTIONS
    if unknown:
        raise ValueError(
            f"Unknown COG driver option(s): {sorted(unknown)}. "
            f"Valid options: {sorted(COG_DRIVER_OPTIONS)}"
        )
