"""Download STAC assets to local files via stac-asset (PC-3).

pyramids reads STAC assets lazily through GDAL `/vsicurl/`; some workflows want
local copies instead (offline processing, repeated reads, archival). This module
wraps `stac_asset`'s synchronous download behind the optional `[stac]` extra,
returning the local paths so they can feed
:meth:`pyramids.dataset.DatasetCollection.from_files`.

Three entry points mirror the three `stac_asset.blocking` downloaders — one
Item (:func:`download_item`), a whole ItemCollection
(:func:`download_item_collection`, e.g. the result of
:func:`pyramids.stac.search`), or a Collection (:func:`download_collection`).
All three are blocking: the async `stac_asset` coroutines cannot run inside a
live event loop, and pyramids exposes no async API.

`stac-asset` pulls heavy async dependencies (`aiohttp`, `aiobotocore`), so it is
**not** a core dependency — it ships via the `[stac]` extra (alongside
`pystac-client`). Install with one of:

- PyPI: ``pip install 'pyramids-gis[stac]'``
- conda-forge: ``conda install -c conda-forge pyramids-stac``
  (stac-asset is not on conda-forge; install it alone with ``pip install stac-asset``)

The per-protocol client (HTTP / S3 / Planetary Computer / Earthdata) is selected
by `stac_asset` from each asset href.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyramids.base._utils import extra_hint, import_stac_asset

# The caveat is this guard's alone: `[stac]` carries stac-asset on PyPI but the
# conda-forge package cannot, so the composed block needs a line after it.
_STAC_ASSET_INSTALL_HINT = (
    extra_hint("download_item requires the optional 'stac-asset' dependency.", "stac")
    + "\n"
    + "                 (stac-asset is not on conda-forge; install it alone: pip install stac-asset)"
)


def _coerce_strategy(module: Any, value: Any, enum_name: str) -> Any:
    """Resolve a strategy spelled as a string to its `stac_asset` enum member.

    The enums (`FileNameStrategy`, `ErrorStrategy`) live in `stac_asset`, which
    callers without the `[stac]` extra cannot import — so the wrappers also take
    the member *name* (case-insensitive, e.g. ``"key"``, ``"keep"``). Anything
    that is not a string is handed to `stac_asset` untouched.

    Args:
        module: The imported `stac_asset` module (the enums are read off it).
        value: An enum member, or its name as a string.
        enum_name: Attribute name of the enum class on `stac_asset`.

    Returns:
        The enum member when `value` was a string, otherwise `value` unchanged.

    Raises:
        ValueError: `value` is a string that names no member of the enum.
    """
    coerced = value
    if isinstance(value, str):
        enum_class = getattr(module, enum_name)
        try:
            coerced = enum_class[value.strip().upper()]
        except KeyError as exc:
            allowed = ", ".join(repr(member.name.lower()) for member in enum_class)
            raise ValueError(f"Unknown {enum_name} {value!r}; expected one of {allowed}.") from exc
    return coerced


def _config_kwargs(module: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Translate the wrappers' download options into `stac_asset.Config` kwargs.

    Only the three long-standing fields are always passed; every widened option
    is added solely when the caller set it, so a default call builds exactly the
    `Config` this module has always built.

    Args:
        module: The imported `stac_asset` module (used to resolve the enums).
        options: The public keyword options, already collected per function.

    Returns:
        The keyword arguments for `stac_asset.Config`.

    Raises:
        ValueError: A strategy was given as a string naming no enum member.
    """
    kwargs: dict[str, Any] = {
        "include": list(options["include"]) if options["include"] else [],
        "exclude": list(options["exclude"]) if options["exclude"] else [],
        "s3_requester_pays": options["s3_requester_pays"],
    }
    if options["alternate_assets"]:
        kwargs["alternate_assets"] = list(options["alternate_assets"])
    if options["file_name_strategy"] is not None:
        kwargs["file_name_strategy"] = _coerce_strategy(module, options["file_name_strategy"], "FileNameStrategy")
    if options["error_strategy"] is not None:
        kwargs["error_strategy"] = _coerce_strategy(module, options["error_strategy"], "ErrorStrategy")
    if options["fail_fast"] is not None:
        kwargs["fail_fast"] = bool(options["fail_fast"])
    if options["warn"] is not None:
        kwargs["warn"] = bool(options["warn"])
    return kwargs


def _download(function_name: str, target: Any, directory: str | Path, **options: Any) -> Any:
    """Guard the optional dependency, build the `Config`, and run a downloader.

    Shared body of the three public wrappers: they differ only in which
    `stac_asset.blocking` function they name and what they accept as `target`.

    Args:
        function_name: Name of the `stac_asset.blocking` downloader to call.
        target: The pystac object handed to that downloader.
        directory: Destination directory (stringified for `stac_asset`).
        **options: The public keyword options, including `max_concurrent`
            (a downloader argument, not a `Config` field).

    Returns:
        Whatever the `stac_asset` downloader returns — the same object type as
        `target`, with asset hrefs rewritten to the local paths.

    Raises:
        OptionalPackageDoesNotExist: When `stac-asset` is not installed.
        ValueError: A strategy was given as a string naming no enum member.
    """
    import_stac_asset(_STAC_ASSET_INSTALL_HINT)
    import stac_asset.blocking

    max_concurrent = options.pop("max_concurrent", None)
    config = stac_asset.Config(**_config_kwargs(stac_asset, options))
    extra: dict[str, Any] = {} if max_concurrent is None else {"max_concurrent_downloads": int(max_concurrent)}
    downloader = getattr(stac_asset.blocking, function_name)
    return downloader(target, str(directory), config=config, **extra)


def download_item(
    item: Any,
    directory: str | Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    alternate_assets: list[str] | None = None,
    s3_requester_pays: bool = False,
    file_name_strategy: Any | None = None,
    error_strategy: Any | None = None,
    fail_fast: bool | None = None,
    warn: bool | None = None,
    max_concurrent: int | None = None,
) -> Any:
    """Download a STAC Item's assets to a local directory.

    A thin, synchronous wrapper over ``stac_asset.blocking.download_item`` (the
    async `download_item` cannot run inside a live event loop). The per-protocol
    client is chosen by `stac_asset` from each asset href.

    Args:
        item: A `pystac.Item` (stac-asset operates on pystac objects).
        directory: Destination directory for the downloaded assets.
        include: Optional asset keys to include (others skipped).
        exclude: Optional asset keys to exclude.
        alternate_assets: Keys of the `alternate-assets` extension to prefer
            over each asset's own href, in order (e.g. ``["s3"]`` to pull from a
            bucket instead of the HTTPS mirror). Unset by default, which keeps
            the primary href.
        s3_requester_pays: Opt into Requester-Pays for `s3://` assets.
        file_name_strategy: How downloaded files are named — a
            `stac_asset.FileNameStrategy` member or its name as a string
            (``"file_name"`` keeps the href's file name, ``"key"`` names each
            file after its asset key). Defaults to stac-asset's `FILE_NAME`.
        error_strategy: What happens to the already-downloaded files when a
            download fails — a `stac_asset.ErrorStrategy` member or its name
            (``"delete"`` or ``"keep"``). Defaults to stac-asset's `DELETE`.
        fail_fast: Raise on the first asset error instead of gathering them all.
        warn: Emit a warning per failed asset instead of raising.
        max_concurrent: Cap on concurrent asset downloads (stac-asset's
            `max_concurrent_downloads`, a downloader argument rather than a
            config field). Defaults to stac-asset's own limit.

    Returns:
        The downloaded `pystac.Item` (with asset hrefs rewritten to the local
        paths), as returned by `stac_asset`.

    Raises:
        OptionalPackageDoesNotExist: When `stac-asset` is not installed.
        ValueError: A strategy was given as a string naming no enum member.

    Examples:
        - Download an item's assets, then build a collection from the locals
          (requires the `[stac]` extra + network):
            ```python
            >>> from pyramids.stac import download_item  # doctest: +SKIP
            >>> local = download_item(item, "scenes/")  # doctest: +SKIP
            >>> hrefs = [a.href for a in local.assets.values()]  # doctest: +SKIP

            ```
    """
    return _download(
        "download_item",
        item,
        directory,
        include=include,
        exclude=exclude,
        alternate_assets=alternate_assets,
        s3_requester_pays=s3_requester_pays,
        file_name_strategy=file_name_strategy,
        error_strategy=error_strategy,
        fail_fast=fail_fast,
        warn=warn,
        max_concurrent=max_concurrent,
    )


def download_item_collection(
    item_collection: Any,
    directory: str | Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    alternate_assets: list[str] | None = None,
    s3_requester_pays: bool = False,
    file_name_strategy: Any | None = None,
    error_strategy: Any | None = None,
    fail_fast: bool | None = None,
    warn: bool | None = None,
    max_concurrent: int | None = None,
) -> Any:
    """Download every Item's assets from an ItemCollection to a local directory.

    The many-item companion of :func:`download_item`, wrapping
    ``stac_asset.blocking.download_item_collection``. Hand it the result of
    :func:`pyramids.stac.search` to take a whole search result offline; the
    returned ItemCollection can then feed
    :meth:`pyramids.dataset.DatasetCollection.from_files`.

    Args:
        item_collection: A `pystac.ItemCollection`.
        directory: Destination directory for the downloaded assets.
        include: Optional asset keys to include (others skipped).
        exclude: Optional asset keys to exclude.
        alternate_assets: Keys of the `alternate-assets` extension to prefer
            over each asset's own href, in order.
        s3_requester_pays: Opt into Requester-Pays for `s3://` assets.
        file_name_strategy: How downloaded files are named — a
            `stac_asset.FileNameStrategy` member or its name as a string
            (``"file_name"`` or ``"key"``).
        error_strategy: What happens to the already-downloaded files when a
            download fails — a `stac_asset.ErrorStrategy` member or its name
            (``"delete"`` or ``"keep"``).
        fail_fast: Raise on the first asset error instead of gathering them all.
        warn: Emit a warning per failed asset instead of raising.
        max_concurrent: Cap on concurrent asset downloads (stac-asset's
            `max_concurrent_downloads`). Defaults to stac-asset's own limit.

    Returns:
        The downloaded `pystac.ItemCollection`, every asset href rewritten to
        its local path, as returned by `stac_asset`.

    Raises:
        OptionalPackageDoesNotExist: When `stac-asset` is not installed.
        ValueError: A strategy was given as a string naming no enum member.

    Examples:
        - Take a whole search result offline (requires the `[stac]` extra +
          network):
            ```python
            >>> from pyramids.stac import download_item_collection  # doctest: +SKIP
            >>> local = download_item_collection(  # doctest: +SKIP
            ...     items, "scenes/", include=["B04"], max_concurrent=8
            ... )

            ```
    """
    return _download(
        "download_item_collection",
        item_collection,
        directory,
        include=include,
        exclude=exclude,
        alternate_assets=alternate_assets,
        s3_requester_pays=s3_requester_pays,
        file_name_strategy=file_name_strategy,
        error_strategy=error_strategy,
        fail_fast=fail_fast,
        warn=warn,
        max_concurrent=max_concurrent,
    )


def download_collection(
    collection: Any,
    directory: str | Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    alternate_assets: list[str] | None = None,
    s3_requester_pays: bool = False,
    file_name_strategy: Any | None = None,
    error_strategy: Any | None = None,
    fail_fast: bool | None = None,
    warn: bool | None = None,
    max_concurrent: int | None = None,
) -> Any:
    """Download a STAC Collection's own assets to a local directory.

    Wraps ``stac_asset.blocking.download_collection``, which fetches the assets
    hanging off the Collection itself (overviews, thumbnails, licences — not the
    assets of its Items; for those, search the Collection and call
    :func:`download_item_collection`).

    Args:
        collection: A `pystac.Collection`.
        directory: Destination directory for the downloaded assets.
        include: Optional asset keys to include (others skipped).
        exclude: Optional asset keys to exclude.
        alternate_assets: Keys of the `alternate-assets` extension to prefer
            over each asset's own href, in order.
        s3_requester_pays: Opt into Requester-Pays for `s3://` assets.
        file_name_strategy: How downloaded files are named — a
            `stac_asset.FileNameStrategy` member or its name as a string
            (``"file_name"`` or ``"key"``).
        error_strategy: What happens to the already-downloaded files when a
            download fails — a `stac_asset.ErrorStrategy` member or its name
            (``"delete"`` or ``"keep"``).
        fail_fast: Raise on the first asset error instead of gathering them all.
        warn: Emit a warning per failed asset instead of raising.
        max_concurrent: Cap on concurrent asset downloads (stac-asset's
            `max_concurrent_downloads`). Defaults to stac-asset's own limit.

    Returns:
        The downloaded `pystac.Collection`, its asset hrefs rewritten to the
        local paths, as returned by `stac_asset`.

    Raises:
        OptionalPackageDoesNotExist: When `stac-asset` is not installed.
        ValueError: A strategy was given as a string naming no enum member.

    Examples:
        - Fetch a collection's own assets (requires the `[stac]` extra +
          network):
            ```python
            >>> from pyramids.stac import download_collection  # doctest: +SKIP
            >>> local = download_collection(collection, "catalog/")  # doctest: +SKIP

            ```
    """
    return _download(
        "download_collection",
        collection,
        directory,
        include=include,
        exclude=exclude,
        alternate_assets=alternate_assets,
        s3_requester_pays=s3_requester_pays,
        file_name_strategy=file_name_strategy,
        error_strategy=error_strategy,
        fail_fast=fail_fast,
        warn=warn,
        max_concurrent=max_concurrent,
    )
