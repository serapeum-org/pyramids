"""Download STAC assets to local files via stac-asset (PC-3).

pyramids reads STAC assets lazily through GDAL `/vsicurl/`; some workflows want
local copies instead (offline processing, repeated reads, archival). This module
wraps `stac_asset`'s synchronous download behind the optional `[stac]` extra,
returning the local paths so they can feed
:meth:`pyramids.dataset.DatasetCollection.from_files`.

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

from pyramids.base._utils import import_stac_asset

_STAC_ASSET_INSTALL_HINT = (
    "download_item requires the optional 'stac-asset' dependency. Install with one of:\n"
    "  - PyPI:        pip install 'pyramids-gis[stac]'\n"
    "  - conda-forge: conda install -c conda-forge pyramids-stac\n"
    "                 (stac-asset is not on conda-forge; install it alone: pip install stac-asset)"
)


def download_item(
    item: Any,
    directory: str | Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    s3_requester_pays: bool = False,
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
        s3_requester_pays: Opt into Requester-Pays for `s3://` assets.

    Returns:
        The downloaded `pystac.Item` (with asset hrefs rewritten to the local
        paths), as returned by `stac_asset`.

    Raises:
        OptionalPackageDoesNotExist: When `stac-asset` is not installed.

    Examples:
        - Download an item's assets, then build a collection from the locals
          (requires the `[stac]` extra + network):
            ```python
            >>> from pyramids.stac import download_item  # doctest: +SKIP
            >>> local = download_item(item, "scenes/")  # doctest: +SKIP
            >>> hrefs = [a.href for a in local.assets.values()]  # doctest: +SKIP

            ```
    """
    import_stac_asset(_STAC_ASSET_INSTALL_HINT)
    import stac_asset.blocking
    from stac_asset import Config

    config = Config(
        include=list(include) if include else [],
        exclude=list(exclude) if exclude else [],
        s3_requester_pays=s3_requester_pays,
    )
    return stac_asset.blocking.download_item(item, str(directory), config=config)
