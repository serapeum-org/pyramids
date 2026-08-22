"""COG creation-option types, serialization, and validation.

Provides the :data:`CreationOptions` alias (a `Mapping[str, Any]`), the named
:data:`PROFILES`, and helpers used by :mod:`pyramids.dataset.cog.write` and
:class:`pyramids.dataset.engines.cog.COG`:

- :func:`to_gdal_options` — serialize a mapping into GDAL's `['KEY=VALUE',...]` list form.
- :func:`merge_options` — merge defaults with user-supplied extras (dict or legacy `list[str]`).
- :func:`validate_blocksize` — enforce the COG driver's power-of-2-in-[64, 4096] constraint.
- :func:`validate_option_keys` — gate unknown keys against :data:`COG_DRIVER_OPTIONS`.
- :func:`profile_options` / :func:`validate_profile` — named compression presets.

Each grouped option dataclass owns the logic that turns *its own* fields into
the effective GDAL options via a ``_to_options`` method (``Layout``/``Tiling``
serialize from their fields alone; ``Compression``/``Overviews`` also take the
source band so the dtype-aware predictor / overview-resampling default can be
resolved). ``BandSelection`` and ``Tags`` instead *transform* the source raster
— ``_translate`` runs the in-memory band-subset/cast/NoData ``gdal.Translate``
and ``_stamp`` applies the colour table / metadata — so the write call site is a
thin orchestrator over these methods rather than the home of the mapping logic.

These ``_``-prefixed methods are private by convention, but they are the COG
engine's internal contract: :class:`pyramids.dataset.engines.cog.COG` is their
only intended caller (a sibling module in this subpackage), and they are *not*
part of the public API. Rename or re-signature them in lockstep with that engine.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from osgeo import gdal

from pyramids.base._errors import FailedToSaveError
from pyramids.base._utils import (
    default_cog_overview_resampling,
    numpy_to_gdal_dtype,
    resolve_cog_predictor,
)

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


_PREDICTOR_COMPRESSORS: frozenset[str] = frozenset({"DEFLATE", "LZW", "LZMA", "ZSTD"})
"""Compression methods for which the GDAL ``PREDICTOR`` option is meaningful.

The COG driver ignores ``PREDICTOR`` (and emits a ``RuntimeWarning``) for every
other method — LERC, NONE, JPEG, WEBP, PACKBITS — so :meth:`Compression._to_options`
omits it there.
"""


_PREDICTOR_SAFE_NBITS: tuple[int, ...] = (8, 16, 32, 64)
"""Sample widths (bits) libtiff accepts for ``PREDICTOR=2``.

A band whose ``NBITS`` is not one of these (e.g. a 12-bit Sentinel-2 JP2 source)
makes the COG driver reject ``PREDICTOR=2`` and, if the narrow width is inherited
onto the output, silently clips values above its domain. :func:`_promote_nbits`
widens such a source to its dtype's natural width (always one of these), and
:func:`_reconcile_predictor_with_nbits` drops the predictor if a caller
nonetheless forces a narrow width.
"""


_UNSIGNED_INT_GDAL_DTYPES: frozenset[int] = frozenset(
    {gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_UInt32, gdal.GDT_UInt64}
)
"""GDAL unsigned-integer dtypes whose ``NBITS`` promotion is meaningful.

Only unsigned integers carry the sub-byte-``NBITS`` predictor/clip problem, and
their natural container width (8/16/32/64) is the only ``NBITS`` GDAL accepts for
the dtype. Float ``NBITS`` (half / 24-bit float) is a distinct, legitimate feature
that must not be rewritten, and GDAL ignores ``NBITS`` on signed integers — so
:func:`_promote_nbits` promotes only for these dtypes.
"""


def _read_source_nbits(band: Any) -> int | None:
    """Return a band's declared ``NBITS`` (bits per sample), or ``None``.

    Reads the ``NBITS`` item from the band's ``IMAGE_STRUCTURE`` metadata domain,
    where GDAL drivers (e.g. ``SENTINEL2``) report a sub-byte-aligned sample
    width propagated from the source.

    Args:
        band: A GDAL raster band.

    Returns:
        The declared width in bits, or ``None`` when the band carries no
        ``NBITS`` or a non-integer value (its samples then use the dtype's
        natural width). A malformed value is ignored rather than raised, mirroring
        the defensive parse in :func:`_reconcile_predictor_with_nbits`.
    """
    raw = band.GetMetadataItem("NBITS", "IMAGE_STRUCTURE")
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _promote_nbits(nbits: int | None, gdal_dtype: int) -> int | None:
    """Promote a sub-natural integer ``NBITS`` to the dtype's natural width.

    The only ``NBITS`` GDAL accepts for an unsigned-integer dtype is its natural
    container width (``Byte -> 8``, ``UInt16 -> 16``, ``UInt32 -> 32``,
    ``UInt64 -> 64``) — always one of :data:`_PREDICTOR_SAFE_NBITS`, so promoting a
    narrower/odd width to it is both predictor-safe and clip-safe. A width below
    the natural one is promoted; a width already at/above it, ``None``, or on a
    non-unsigned-integer dtype returns ``None`` (no override).

    Promotion widens the output to the natural width, trading a little storage (a
    1-bit mask becomes 8-bit; an in-domain 12-bit raster becomes 16-bit) for
    clip-safety on *derived* data whose values have grown past the declared narrow
    domain — the #1023 case. Float dtypes are deliberately left alone: a float
    ``NBITS`` (half / 24-bit float) is a distinct, legitimate encoding that
    widening would corrupt. GDAL ignores ``NBITS`` on signed integers, so those
    are left alone too (the predictor still drops to ``1`` for a narrow width).

    Args:
        nbits: The source band's declared width, or ``None``.
        gdal_dtype: The source band's GDAL data-type code (e.g. ``gdal.GDT_UInt16``).

    Returns:
        The natural width to write as ``NBITS``, or ``None`` when no promotion is
        needed or applicable.

    Examples:
        - A 12-bit UInt16 source promotes to its natural 16:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset.cog.options import _promote_nbits
            >>> _promote_nbits(12, gdal.GDT_UInt16)
            16

            ```
        - A 12-bit UInt32 source promotes to its natural 32 (not 16):
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset.cog.options import _promote_nbits
            >>> _promote_nbits(12, gdal.GDT_UInt32)
            32

            ```
        - An already-natural width, ``None``, or a float dtype is left alone:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset.cog.options import _promote_nbits
            >>> _promote_nbits(16, gdal.GDT_UInt16) is None
            True
            >>> _promote_nbits(None, gdal.GDT_UInt16) is None
            True
            >>> _promote_nbits(12, gdal.GDT_Float32) is None
            True

            ```
    """
    if nbits is None or gdal_dtype not in _UNSIGNED_INT_GDAL_DTYPES:
        return None
    natural = gdal.GetDataTypeSize(gdal_dtype)
    return natural if nbits < natural else None


def _reconcile_predictor_with_nbits(options: dict[str, Any]) -> None:
    """Drop ``PREDICTOR`` when the final ``NBITS`` cannot honour it.

    ``PREDICTOR=2``/``3`` require an 8/16/32/64-bit sample. When a caller forces
    a sub-byte-aligned ``NBITS`` through ``extra`` (overriding the promoted
    default), keeping the predictor would make GDAL reject the write. Mutates
    ``options`` in place, removing ``PREDICTOR`` so the caller's explicit narrow
    width is honoured. An already-disabled predictor (``1`` or ``"NO"``,
    case-insensitive) is left untouched.

    Args:
        options: The merged GDAL creation-option dict (post :func:`merge_options`).

    Examples:
        - A caller-forced narrow width drops the predictor:
            ```python
            >>> from pyramids.dataset.cog.options import _reconcile_predictor_with_nbits
            >>> opts = {"NBITS": 12, "PREDICTOR": 2}
            >>> _reconcile_predictor_with_nbits(opts)
            >>> "PREDICTOR" in opts
            False

            ```
        - A predictor-safe width leaves the predictor in place:
            ```python
            >>> from pyramids.dataset.cog.options import _reconcile_predictor_with_nbits
            >>> opts = {"NBITS": 16, "PREDICTOR": 2}
            >>> _reconcile_predictor_with_nbits(opts)
            >>> opts["PREDICTOR"]
            2

            ```
    """
    nbits = options.get("NBITS")
    if nbits is None:
        return
    try:
        nbits_value = int(nbits)
    except (TypeError, ValueError):
        return
    predictor = options.get("PREDICTOR")
    if (
        nbits_value not in _PREDICTOR_SAFE_NBITS
        and predictor is not None
        and str(predictor).upper() not in ("1", "NO")
    ):
        options.pop("PREDICTOR", None)


_PALETTE_GDAL_DTYPES: frozenset[int] = frozenset({gdal.GDT_Byte, gdal.GDT_UInt16})
"""GDAL dtypes for which a colour table (palette) is meaningful."""


COG_READ_DEFAULTS: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "0.5",
    "VSI_CACHE": "TRUE",
}
"""GDAL config options that make remote ``/vsicurl/`` COG reads efficient.

Without ``GDAL_DISABLE_READDIR_ON_OPEN`` a remote open issues a directory
listing — often the single biggest latency hit, and the reason this preset
exists. ``GDAL_HTTP_MULTIRANGE`` lets GDAL issue the scattered tile ranges a COG
read produces as one multi-range request (``GDAL_HTTP_MERGE_CONSECUTIVE_RANGES``
then coalesces the adjacent ones), and the retry pair rides out a transient 5xx
from object storage instead of failing the whole read. That budget is
deliberately larger than the OGC discovery one
(:data:`pyramids.base._ogc_api.GDAL_HTTP_MAX_RETRY`): a COG read issues many
range requests, so one flaky range should not lose the whole read, whereas a
discovery pre-check is a single request in front of work that has not started
yet.

Deliberately **not** included: ``CPL_VSIL_CURL_ALLOWED_EXTENSIONS``. Restricting
`/vsicurl/` to ``.tif,.tiff`` makes GDAL refuse any URL whose path does not end
in one — extensionless object keys, presigned S3 links carrying a query string,
Planetary Computer and STAC asset hrefs — so a preset meant to speed reads up
instead made them impossible. The readdir-skip above already delivers the
latency win on its own. Applied by
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


@dataclass(frozen=True)
class Compression:
    """How pixel bytes are compressed when writing a COG.

    Grouped option object for :meth:`pyramids.dataset.engines.cog.COG.to_cog`'s
    ``compression=`` argument. The string form of that argument is a named
    :data:`PROFILES` entry (e.g. ``"zstd"``) and is expanded into this type by
    :meth:`coerce`, so ``to_cog(path, compression="zstd")`` and
    ``to_cog(path, compression=Compression(compress="ZSTD", level=9))`` are
    equivalent.

    Attributes:
        compress: Compression method (e.g. ``"DEFLATE"``, ``"ZSTD"``,
            ``"LZW"``, ``"JPEG"``, ``"WEBP"``, ``"LERC"``, ``"NONE"``).
            ``None`` lets ``to_cog`` apply its ``DEFLATE`` default.
        level: Compression level (e.g. 1-12 for DEFLATE, 1-22 for ZSTD).
        quality: Lossy quality 1-100 (JPEG/WEBP).
        predictor: ``"YES"``/``"NO"``/``"STANDARD"``/``"FLOATING_POINT"`` or
            ``1``/``2``/``3`` (the COG driver's ``PREDICTOR`` tokens; ``"NO"``
            /``1`` disables it). ``None`` auto-resolves per source dtype (``2``
            integer, ``3`` float).
        max_z_error: Maximum per-pixel error for the LERC family.

    Examples:
        - Build an explicit compression policy and read its fields:
            ```python
            >>> from pyramids.dataset.cog import Compression
            >>> comp = Compression(compress="ZSTD", level=18)
            >>> comp.compress
            'ZSTD'
            >>> comp.level
            18

            ```
        - An out-of-range quality is rejected at construction:
            ```python
            >>> Compression(quality=200)
            Traceback (most recent call last):
                ...
            ValueError: quality must be in 1..100; got 200.

            ```
    """

    compress: str | None = None
    level: int | None = None
    quality: int | None = None
    predictor: str | int | None = None
    max_z_error: float | None = None

    def __post_init__(self) -> None:
        """Validate quality and predictor ranges."""
        if self.quality is not None and not 1 <= self.quality <= 100:
            raise ValueError(f"quality must be in 1..100; got {self.quality}.")
        if self.predictor is not None and self.predictor not in {
            1,
            2,
            3,
            "1",
            "2",
            "3",
            "YES",
            "NO",
            "STANDARD",
            "FLOATING_POINT",
        }:
            raise ValueError(
                f"predictor must be one of 1/2/3 (int or str) / 'YES'/'NO'/"
                f"'STANDARD'/'FLOATING_POINT'; got {self.predictor!r}."
            )

    @classmethod
    def coerce(cls, value: str | Compression | None) -> Compression | None:
        """Coerce a profile-name string or ``Compression`` into a ``Compression``.

        Args:
            value: A named :data:`PROFILES` string (e.g. ``"zstd"``), an existing
                :class:`Compression`, or ``None``.

        Returns:
            The coerced :class:`Compression`, or ``None`` when ``value`` is
            ``None`` (``to_cog`` then applies its ``DEFLATE`` default).

        Raises:
            ValueError: When ``value`` is a string that is not a known profile.

        Examples:
            - A profile name expands to its preset:
                ```python
                >>> Compression.coerce("zstd")
                Compression(compress='ZSTD', level=9, quality=None, predictor=None, max_z_error=None)

                ```
            - ``None`` passes through:
                ```python
                >>> Compression.coerce(None) is None
                True

                ```
        """
        if value is None or isinstance(value, Compression):
            return value
        opts = profile_options(value)
        return cls(
            compress=opts.get("COMPRESS"),
            level=opts.get("LEVEL"),
            quality=opts.get("QUALITY"),
            max_z_error=opts.get("MAX_Z_ERROR"),
        )

    def _to_options(self, source_band: Any) -> dict[str, Any]:
        """Map the compression fields to their GDAL COG creation options.

        The compression method falls back to the house ``DEFLATE`` default; the
        predictor is resolved per the source dtype when unset (``2`` for
        integer, ``3`` for float) and is emitted only for methods that honour it
        (:data:`_PREDICTOR_COMPRESSORS`) — GDAL ignores and warns on ``PREDICTOR``
        for LERC/NONE/JPEG/WEBP. ``max_z_error`` (LERC family) rides along as
        ``MAX_Z_ERROR``. ``None`` values are kept and dropped later by
        :func:`merge_options` / :func:`to_gdal_options`.

        An **unsigned-integer** source band whose ``NBITS`` is below its natural
        width (e.g. ``12`` from a Sentinel-2 JP2 ``UInt16`` source) is promoted to
        that natural width via :func:`_promote_nbits` and the width is emitted as
        ``NBITS`` — otherwise the inherited narrow width both trips ``PREDICTOR=2``
        and silently clips values above its domain. The predictor is resolved
        against the *promoted* width, so a promoted ``12 -> 16`` still uses
        ``PREDICTOR=2``. Float and signed-integer dtypes are left alone (a float
        ``NBITS`` is a distinct encoding; GDAL ignores ``NBITS`` on signed ints).
        A caller can still override ``NBITS`` through ``extra``; the write site then
        reconciles the predictor via :func:`_reconcile_predictor_with_nbits`.

        Args:
            source_band: Band 0 of the effective source, whose ``DataType`` drives
                the predictor default and whose ``NBITS`` drives the width.

        Returns:
            The compression slice of the GDAL creation-option dict
            (``COMPRESS``/``LEVEL``/``QUALITY``/``MAX_Z_ERROR``/``PREDICTOR``, plus
            ``NBITS`` when a sub-natural unsigned-integer source width is promoted).
        """
        compress = self.compress if self.compress is not None else "DEFLATE"
        source_nbits = _read_source_nbits(source_band)
        promoted_nbits = _promote_nbits(source_nbits, source_band.DataType)
        # The width the output will actually use: the promoted natural width, else
        # the source's own (already natural, a float/signed width we leave alone, or
        # None for the dtype default).
        effective_nbits = promoted_nbits if promoted_nbits is not None else source_nbits
        predictor = self.predictor
        if predictor is None:
            predictor = resolve_cog_predictor(source_band.DataType, effective_nbits)
        options: dict[str, Any] = {
            "COMPRESS": compress,
            "LEVEL": self.level,
            "QUALITY": self.quality,
            "MAX_Z_ERROR": self.max_z_error,
            "PREDICTOR": (
                predictor if compress.upper() in _PREDICTOR_COMPRESSORS else None
            ),
        }
        if promoted_nbits is not None:
            options["NBITS"] = promoted_nbits
        return options


@dataclass(frozen=True)
class Overviews:
    """The internal overview pyramid written into the COG.

    Grouped option object for ``to_cog``'s ``overviews=`` argument.

    Attributes:
        resampling: Overview resampling method (``nearest``, ``average``,
            ``bilinear``, ``cubic``, ``cubicspline``, ``lanczos``, ``mode``,
            ``rms``, ``gauss``). ``None`` auto-resolves per source dtype
            (``mode`` for categorical, ``average`` for continuous).
        count: Number of overview levels. ``None`` lets GDAL decide.
        compress: Compression for the overview IFDs. ``None`` inherits the
            full-resolution compression.

    Examples:
        - Pin an averaging pyramid with four levels:
            ```python
            >>> from pyramids.dataset.cog import Overviews
            >>> ov = Overviews(resampling="average", count=4)
            >>> ov.resampling
            'average'
            >>> ov.count
            4

            ```
        - A negative overview count is rejected:
            ```python
            >>> Overviews(count=-1)
            Traceback (most recent call last):
                ...
            ValueError: overview count must be >= 0; got -1.

            ```
    """

    resampling: str | None = None
    count: int | None = None
    compress: str | None = None

    def __post_init__(self) -> None:
        """Validate the overview count."""
        if self.count is not None and self.count < 0:
            raise ValueError(f"overview count must be >= 0; got {self.count}.")

    def _to_options(self, source_band: Any) -> dict[str, Any]:
        """Map the overview fields to their GDAL COG creation options.

        When ``resampling`` is unset it resolves to a category-safe default from
        the source dtype / colour table (``mode`` for categorical, ``average``
        for continuous) so the default never corrupts a categorical raster. The
        caller (:meth:`pyramids.dataset.engines.cog.COG.to_cog`) still owns the
        guardrail warning for an *explicit* averaging choice on categorical data.

        Args:
            source_band: Band 0 of the effective source, whose ``DataType`` and
                colour table drive the resampling default.

        Returns:
            The overview slice of the GDAL creation-option dict
            (``OVERVIEW_RESAMPLING``/``OVERVIEW_COUNT``/``OVERVIEW_COMPRESS``).
        """
        resampling = self.resampling
        if resampling is None:
            resampling = default_cog_overview_resampling(
                source_band.DataType, source_band.GetColorTable() is not None
            )
        return {
            "OVERVIEW_RESAMPLING": resampling,
            "OVERVIEW_COUNT": self.count,
            "OVERVIEW_COMPRESS": self.compress,
        }


@dataclass(frozen=True)
class Tiling:
    """Reprojection and web-tiling layout of the output COG.

    Grouped option object for ``to_cog``'s ``tiling=`` argument. ``scheme`` and
    ``target_srs`` are mutually exclusive: when both are set, ``scheme`` wins and
    ``target_srs`` is ignored (``to_cog`` emits a ``UserWarning``).

    Attributes:
        target_srs: Reproject before write. An EPSG integer, or a WKT / PROJ
            string.
        resampling: Warp resampling used when reprojecting (``target_srs``) or
            tiling (``scheme``). Ignored when neither is set.
        scheme: A tiling scheme such as ``"GoogleMapsCompatible"`` for a
            web-optimized COG (EPSG:3857).
        zoom_level: Pin the maximum zoom level (advanced tiling-scheme knob).
        zoom_level_strategy: ``auto`` (default), ``lower``, or ``upper``.
        aligned_levels: Number of overview levels aligned to the tiling scheme.

    Examples:
        - Request a web-optimized tiling scheme:
            ```python
            >>> from pyramids.dataset.cog import Tiling
            >>> til = Tiling(scheme="GoogleMapsCompatible")
            >>> til.scheme
            'GoogleMapsCompatible'
            >>> til.zoom_level_strategy
            'auto'

            ```
        - An unknown zoom-level strategy is rejected:
            ```python
            >>> Tiling(zoom_level_strategy="sideways")
            Traceback (most recent call last):
                ...
            ValueError: zoom_level_strategy must be 'auto'/'lower'/'upper'; got 'sideways'.

            ```
    """

    target_srs: int | str | None = None
    resampling: str = "nearest"
    scheme: str | None = None
    zoom_level: int | None = None
    zoom_level_strategy: str = "auto"
    aligned_levels: int | None = None

    def __post_init__(self) -> None:
        """Validate the zoom-level strategy."""
        if self.zoom_level_strategy not in {"auto", "lower", "upper"}:
            raise ValueError(
                f"zoom_level_strategy must be 'auto'/'lower'/'upper'; "
                f"got {self.zoom_level_strategy!r}."
            )

    def _to_options(self) -> dict[str, Any]:
        """Map the tiling / reprojection fields to their GDAL COG options.

        ``scheme`` and ``target_srs`` are mutually exclusive: when both are set,
        ``scheme`` wins, ``target_srs`` is dropped, and a ``UserWarning`` is
        emitted. ``WARP_RESAMPLING`` is only emitted when actually reprojecting
        (a scheme or a surviving ``target_srs``); an integer ``target_srs``
        becomes ``EPSG:<n>``, a string is forwarded verbatim.

        Returns:
            The tiling slice of the GDAL creation-option dict.

        Examples:
            - A tiling scheme drives the scheme keys and warp resampling:
                ```python
                >>> from pyramids.dataset.cog import Tiling
                >>> Tiling(scheme="GoogleMapsCompatible")._to_options()["TILING_SCHEME"]
                'GoogleMapsCompatible'

                ```
            - An integer target SRS is formatted as an EPSG string:
                ```python
                >>> Tiling(target_srs=3857)._to_options()["TARGET_SRS"]
                'EPSG:3857'

                ```
            - With neither scheme nor target SRS, warp resampling is dropped:
                ```python
                >>> Tiling()._to_options()["WARP_RESAMPLING"] is None
                True

                ```
        """
        eff_target_srs = self.target_srs
        if self.scheme is not None and eff_target_srs is not None:
            warnings.warn(
                "Both tiling.scheme and tiling.target_srs provided; "
                "scheme wins and target_srs is ignored.",
                UserWarning,
                stacklevel=3,
            )
            eff_target_srs = None
        reprojecting = bool(self.scheme or eff_target_srs)
        options: dict[str, Any] = {
            "TILING_SCHEME": self.scheme,
            "ZOOM_LEVEL": self.zoom_level,
            "ZOOM_LEVEL_STRATEGY": self.zoom_level_strategy,
            "ALIGNED_LEVELS": self.aligned_levels,
            "WARP_RESAMPLING": self.resampling if reprojecting else None,
        }
        if eff_target_srs is not None:
            options["TARGET_SRS"] = (
                f"EPSG:{eff_target_srs}"
                if isinstance(eff_target_srs, int)
                else eff_target_srs
            )
        return options


@dataclass(frozen=True)
class BandSelection:
    """Band selection and pixel-dtype transform applied before the COG write.

    Grouped option object for ``to_cog``'s ``bands=`` argument. Any of these
    routes the source through an in-memory ``gdal.Translate`` so the predictor /
    overview policy and the write itself see the *output* bands.

    Attributes:
        indexes: 0-based band indices to keep, in order (e.g. ``[3, 2, 1]`` to
            select and reorder). ``None`` keeps all bands.
        out_dtype: Output NumPy dtype name to cast to (e.g. ``"uint8"``).
            ``None`` keeps the source dtype.
        nodata: NoData value to set on the output. ``None`` keeps the source
            NoData.

    Note:
        ``indexes`` is a plain list, so ``frozen=True`` only blocks rebinding the
        field, not mutating the list, and an instance is not hashable once
        ``indexes`` is populated. Treat these as option carriers, not value keys.

    Examples:
        - Select and reorder three bands, casting the output:
            ```python
            >>> from pyramids.dataset.cog import BandSelection
            >>> sel = BandSelection(indexes=[2, 1, 0], out_dtype="uint8")
            >>> sel.indexes
            [2, 1, 0]
            >>> sel.out_dtype
            'uint8'

            ```
        - A negative (non-0-based) index is rejected:
            ```python
            >>> BandSelection(indexes=[0, -1])
            Traceback (most recent call last):
                ...
            ValueError: band indexes must be >= 0 (0-based); got [0, -1].

            ```
    """

    indexes: list[int] | None = None
    out_dtype: str | None = None
    nodata: float | int | None = None

    def __post_init__(self) -> None:
        """Validate band indices."""
        if self.indexes is not None and any(i < 0 for i in self.indexes):
            raise ValueError(
                f"band indexes must be >= 0 (0-based); got {self.indexes}."
            )

    def _needs_translate(self) -> bool:
        """Return ``True`` when any field requires an in-memory pre-process.

        A band subset, a dtype cast, or a NoData override all route the source
        through :meth:`_translate`; with none of them set the backing raster is
        used unchanged.

        Returns:
            ``True`` if ``indexes``, ``out_dtype``, or ``nodata`` is set.

        Examples:
            - A bare selection needs no pre-process:
                ```python
                >>> from pyramids.dataset.cog import BandSelection
                >>> BandSelection()._needs_translate()
                False

                ```
            - Any populated field flips it on (here a dtype cast):
                ```python
                >>> from pyramids.dataset.cog import BandSelection
                >>> BandSelection(out_dtype="uint8")._needs_translate()
                True

                ```
        """
        return (
            self.indexes is not None
            or self.out_dtype is not None
            or self.nodata is not None
        )

    def _translate(self, source: gdal.Dataset) -> gdal.Dataset:
        """Run the in-memory ``gdal.Translate`` for the selected fields.

        Applies band selection/reordering (0-based indices mapped to GDAL's
        1-based ``bandList``), the output dtype cast, and the NoData override,
        producing a MEM dataset so the predictor / overview policy and the COG
        write see the *output* bands rather than the original source.

        Args:
            source: The backing :class:`gdal.Dataset` to pre-process.

        Returns:
            A new in-memory :class:`gdal.Dataset`.

        Raises:
            FailedToSaveError: When ``gdal.Translate`` returns no dataset.
        """
        translate_kwargs: dict[str, Any] = {}
        if self.indexes is not None:
            # pyramids band indices are 0-based; GDAL bandList is 1-based.
            translate_kwargs["bandList"] = [i + 1 for i in self.indexes]
        if self.out_dtype is not None:
            translate_kwargs["outputType"] = numpy_to_gdal_dtype(self.out_dtype)
        if self.nodata is not None:
            translate_kwargs["noData"] = self.nodata
        mem = gdal.Translate("", source, format="MEM", **translate_kwargs)
        if mem is None:
            raise FailedToSaveError(
                "failed to build the pre-processed COG source "
                f"(indexes={self.indexes}, out_dtype={self.out_dtype}, "
                f"nodata={self.nodata})"
            )
        return mem


@dataclass(frozen=True)
class Tags:
    """Metadata and colour table stamped onto the output COG.

    Grouped option object for ``to_cog``'s ``tags=`` argument.

    Attributes:
        band_tags: Per-band metadata keyed by 0-based band index, e.g.
            ``{0: {"name": "NDVI"}}``.
        colormap: Palette for band 1, mapping pixel value to an ``(R, G, B, A)``
            tuple. GeoTIFF only supports a colour table on a single-band
            ``Byte`` / ``UInt16`` raster.
        metadata: Dataset-level metadata items.

    Note:
        The fields are plain dicts, so ``frozen=True`` only blocks rebinding
        them, not mutating their contents, and an instance is not hashable once such a
        field is populated. Treat these as option carriers, not value keys.

    Examples:
        - Stamp a band description and read it back:
            ```python
            >>> from pyramids.dataset.cog import Tags
            >>> tags = Tags(band_tags={0: {"name": "NDVI"}}, metadata={"source": "s2"})
            >>> tags.band_tags[0]["name"]
            'NDVI'
            >>> tags.metadata["source"]
            's2'

            ```
        - A bare instance carries nothing:
            ```python
            >>> Tags().colormap is None
            True

            ```
    """

    band_tags: dict[int, dict[str, Any]] | None = None
    colormap: dict[int, tuple[int, int, int, int]] | None = None
    metadata: dict[str, Any] | None = None

    def _has_any(self) -> bool:
        """Return ``True`` when any tag / colourmap / metadata is set to stamp.

        Returns:
            ``True`` if ``band_tags``, ``colormap``, or ``metadata`` is non-empty.

        Examples:
            - A bare instance carries nothing to stamp:
                ```python
                >>> from pyramids.dataset.cog import Tags
                >>> Tags()._has_any()
                False

                ```
            - Any populated field makes it true (here dataset metadata):
                ```python
                >>> from pyramids.dataset.cog import Tags
                >>> Tags(metadata={"source": "s2"})._has_any()
                True

                ```
        """
        return bool(self.band_tags or self.colormap or self.metadata)

    def _stamp(self, ds: gdal.Dataset) -> None:
        """Stamp band tags / colourmap / dataset metadata onto a dataset.

        Mutates ``ds`` in place — the caller passes a MEM copy so the user's open
        dataset is never touched. Dataset metadata and per-band tags are written
        as strings; the colourmap builds a palette on band 1 and flips its colour
        interpretation to ``PaletteIndex``.

        Args:
            ds: The (copied) :class:`gdal.Dataset` to mutate.

        Raises:
            ValueError: When ``colormap`` targets a band whose dtype is not
                ``Byte`` / ``UInt16`` (GeoTIFF only supports a colour table
                there; GDAL would otherwise fail deep in ``CreateCopy``).
        """
        if self.metadata:
            ds.SetMetadata({str(k): str(v) for k, v in self.metadata.items()})
        if self.colormap:
            band = ds.GetRasterBand(1)
            if band.DataType not in _PALETTE_GDAL_DTYPES:
                raise ValueError(
                    f"colormap is only supported on Byte/UInt16 rasters; got "
                    f"{gdal.GetDataTypeName(band.DataType)}. Cast first with "
                    f"to_cog(..., bands=cog.BandSelection(out_dtype='uint8')), "
                    f"or drop the colormap."
                )
            color_table = gdal.ColorTable()
            for value, rgba in self.colormap.items():
                color_table.SetColorEntry(int(value), tuple(rgba))
            band.SetColorTable(color_table)
            band.SetColorInterpretation(gdal.GCI_PaletteIndex)
        if self.band_tags:
            for index, tags in self.band_tags.items():
                # 0-based index -> GDAL 1-based band number.
                ds.GetRasterBand(index + 1).SetMetadata(
                    {str(k): str(v) for k, v in tags.items()}
                )


@dataclass(frozen=True)
class Layout:
    """Physical layout and driver behaviour of the output COG.

    Grouped option object for ``to_cog``'s ``layout=`` argument.

    Attributes:
        blocksize: Internal tile size; a power of 2 in [64, 4096].
        bigtiff: ``"IF_SAFER"`` (default), ``"YES"``, ``"NO"``, ``"IF_NEEDED"``.
        num_threads: Worker threads; ``"ALL_CPUS"`` or an int.
        add_mask: Add an alpha band for transparency.
        sparse_ok: Allow sparse (unfilled) tiles.
        statistics: Compute and embed band statistics.

    Examples:
        - Override the tile size and read the defaults it keeps:
            ```python
            >>> from pyramids.dataset.cog import Layout
            >>> lay = Layout(blocksize=256)
            >>> lay.blocksize
            256
            >>> lay.bigtiff
            'IF_SAFER'

            ```
        - A non-power-of-2 blocksize is rejected at construction:
            ```python
            >>> Layout(blocksize=500)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            ValueError: blocksize must be a power of 2 in [64, 4096]; got 500...

            ```
    """

    blocksize: int = 512
    bigtiff: str = "IF_SAFER"
    num_threads: int | str = "ALL_CPUS"
    add_mask: bool = False
    sparse_ok: bool = False
    statistics: bool = True

    def __post_init__(self) -> None:
        """Validate blocksize and bigtiff."""
        validate_blocksize(self.blocksize)
        if self.bigtiff not in {"IF_SAFER", "YES", "NO", "IF_NEEDED"}:
            raise ValueError(
                f"bigtiff must be 'IF_SAFER'/'YES'/'NO'/'IF_NEEDED'; "
                f"got {self.bigtiff!r}."
            )

    def _to_options(self) -> dict[str, Any]:
        """Map the physical-layout fields to their GDAL COG creation options.

        ``num_threads`` is stringified (``"ALL_CPUS"`` or an int becomes its
        decimal string); the boolean toggles map to ``YES``/dropped — an unset
        toggle is left as ``None`` so :func:`merge_options` /
        :func:`to_gdal_options` omit it rather than forcing ``NO``.

        Returns:
            The layout slice of the GDAL creation-option dict.

        Examples:
            - Defaults keep the house tile size and enable statistics:
                ```python
                >>> from pyramids.dataset.cog import Layout
                >>> opts = Layout()._to_options()
                >>> opts["BLOCKSIZE"], opts["STATISTICS"]
                (512, 'YES')

                ```
            - An integer thread count is stringified; unset toggles drop out:
                ```python
                >>> opts = Layout(num_threads=4)._to_options()
                >>> opts["NUM_THREADS"], opts["ADD_ALPHA"], opts["SPARSE_OK"]
                ('4', None, None)

                ```
        """
        num_threads = (
            self.num_threads
            if isinstance(self.num_threads, str)
            else str(self.num_threads)
        )
        return {
            "BLOCKSIZE": self.blocksize,
            "BIGTIFF": self.bigtiff,
            "NUM_THREADS": num_threads,
            "ADD_ALPHA": True if self.add_mask else None,
            "SPARSE_OK": True if self.sparse_ok else None,
            "STATISTICS": "YES" if self.statistics else None,
        }


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
