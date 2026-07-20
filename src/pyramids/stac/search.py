"""Typed STAC item-search helper (PB-3).

A thin, typed wrapper over ``pystac_client.Client.search`` that makes the common
AOI / time / cloud query one call and returns an ``ItemCollection`` ready for
:meth:`pyramids.dataset.DatasetCollection.from_stac`. It:

* opens a client from a URL (or accepts an already-open ``Client``);
* **gates** a CQL2 ``filter`` on the endpoint advertising the ``FILTER``
  conformance class, raising a clear error instead of pystac-client's opaque one;
* accepts a shapely geometry **or** a GeoJSON dict for ``intersects``;
* **bounds the query at the API** — ``bbox`` / ``datetime`` / ``max_items`` /
  ``limit`` are forwarded to ``client.search`` so the server (and paging) does
  the work (M3). This contrasts with :func:`pyramids.dataset._stac.from_stac`,
  whose own ``bbox`` / ``max_items`` are *client-side post-filters* over an
  already-materialised item list.

`pystac-client` is an optional dependency. Install with one of:

- PyPI: ``pip install 'pyramids-gis[stac]'``
- conda-forge: ``conda install -c conda-forge pyramids-stac``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pyramids.base._utils import import_pystac_client
from pyramids.stac.client import open_client

_STAC_INSTALL_HINT = (
    "search requires the optional 'pystac-client' dependency. Install with one of:\n"
    "  - PyPI:        pip install 'pyramids-gis[stac]'\n"
    "  - conda-forge: conda install -c conda-forge pyramids-stac"
)


def search(
    client_or_url: Any,
    collections: str | Sequence[str],
    *,
    bbox: Sequence[float] | None = None,
    intersects: Any = None,
    datetime: Any = None,
    query: Any = None,
    filter: Any = None,
    sortby: Any = None,
    max_items: int | None = None,
    limit: int | None = None,
    signer: Any = None,
) -> Any:
    """Run a STAC item search and return the matched ``ItemCollection``.

    Args:
        client_or_url: An open ``pystac_client.Client``, or a STAC API root URL
            (opened via :func:`pyramids.stac.open_client`, wiring `signer`).
        collections: Collection id, or a sequence of collection ids, to search.
        bbox: Optional `(minx, miny, maxx, maxy)` lon/lat box, forwarded to the
            API. Mutually exclusive with `intersects` (STAC API rule).
        intersects: Optional AOI geometry — a shapely geometry (anything with a
            ``__geo_interface__``) or a GeoJSON-geometry dict. A shapely geometry
            is converted to GeoJSON before the request.
        datetime: Optional RFC 3339 datetime or interval string
            (e.g. ``"2023-06/2023-08"``), forwarded to the API.
        query: Optional ``query`` extension dict (e.g.
            ``{"eo:cloud_cover": {"lt": 20}}``).
        filter: Optional CQL2 filter (cql2-json dict or cql2-text string). When
            given, the endpoint must advertise the ``FILTER`` conformance class.
        sortby: Optional sort specification forwarded to the API.
        max_items: Optional cap on the total number of items returned (bounds
            paging at the API).
        limit: Optional page size forwarded to the API.
        signer: Optional signer used only when `client_or_url` is a URL (to open
            the client). Ignored when an open client is passed.

    Returns:
        The ``pystac.ItemCollection`` of matched items, ready to hand to
        ``DatasetCollection.from_stac``.

    Raises:
        OptionalPackageDoesNotExist: When `pystac-client` is not installed.
        ValueError: When both `bbox` and `intersects` are given (mutually
            exclusive per the STAC API spec), or when a `filter` is given but
            the endpoint does not advertise the CQL2 ``FILTER`` conformance
            class.

    Examples:
        - Search a collection over an AOI and time window, then build a cube
          (requires the `[stac]` extra and network access):
            ```python
            >>> from pyramids.stac import search  # doctest: +SKIP
            >>> from pyramids.dataset import DatasetCollection  # doctest: +SKIP
            >>> items = search(  # doctest: +SKIP
            ...     "https://earth-search.aws.element84.com/v1",
            ...     "sentinel-2-l2a",
            ...     bbox=(11.0, 46.0, 11.2, 46.2),
            ...     datetime="2023-06/2023-08",
            ...     query={"eo:cloud_cover": {"lt": 20}},
            ...     max_items=10,
            ... )
            >>> cube = DatasetCollection.from_stac(items, asset=["red", "green", "blue"])  # doctest: +SKIP

            ```
    """
    if bbox is not None and intersects is not None:
        raise ValueError(
            "bbox and intersects are mutually exclusive (STAC API spec); pass only one."
        )

    import_pystac_client(_STAC_INSTALL_HINT)
    from pystac_client import ConformanceClasses

    client = (
        open_client(client_or_url, signer=signer)
        if isinstance(client_or_url, str)
        else client_or_url
    )

    if filter is not None and not client.conforms_to(ConformanceClasses.FILTER):
        raise ValueError(
            "the STAC endpoint does not advertise the CQL2 FILTER conformance "
            "class, so a `filter` cannot be used against it."
        )

    if intersects is not None and hasattr(intersects, "__geo_interface__"):
        intersects = intersects.__geo_interface__

    return client.search(
        collections=collections,
        bbox=bbox,
        intersects=intersects,
        datetime=datetime,
        query=query,
        filter=filter,
        sortby=sortby,
        max_items=max_items,
        limit=limit,
    ).item_collection()
