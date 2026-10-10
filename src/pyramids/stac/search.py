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

Two entry points share one query builder:

* :func:`search` — eager: returns the matched ``ItemCollection``.
* :func:`item_search` — lazy: returns the ``ItemSearch`` itself, so the caller
  can ask for the total hit count (``.matched()``), stream items (``.items()``)
  or walk result pages (``.pages()``) without materialising everything first.

That split decides who owns the HTTP session when a URL is passed rather than an
open ``Client``. :func:`search` materialises everything, so the client it opened
is closed before it returns (see :mod:`pyramids.stac._clients`). The lazy
``ItemSearch`` keeps issuing requests through its client, so :func:`item_search`
leaves the one it opened open — and releases it only when the query is rejected
and nothing is returned. Pass an open ``Client`` when that lifetime matters.

`pystac-client` is an optional dependency. Install with one of:

- PyPI: ``pip install 'pyramids-gis[stac]'``
- conda-forge: ``conda install -c conda-forge pyramids-stac``
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

from pyramids.base._utils import extra_hint, import_pystac_client
from pyramids.stac._clients import close_client, resolved_client
from pyramids.stac.client import open_client

_STAC_INSTALL_HINT = extra_hint(
    "search requires the optional 'pystac-client' dependency.",
    "stac",
)


def _reject_both_aois(bbox: Any, intersects: Any) -> None:
    """Refuse the `bbox` + `intersects` combination the STAC API spec forbids.

    Checked by both entry points before either touches a client, so the
    rejection costs no HTTP session — :func:`search` resolves its client only
    once the query is known to be well formed.

    Args:
        bbox: The caller's `bbox`, or `None`.
        intersects: The caller's `intersects`, or `None`.

    Raises:
        ValueError: Both were given.
    """
    if bbox is not None and intersects is not None:
        raise ValueError(
            "bbox and intersects are mutually exclusive (STAC API spec); pass only one."
        )


def item_search(
    client_or_url: Any,
    collections: str | Sequence[str],
    *,
    ids: str | Sequence[str] | None = None,
    bbox: Sequence[float] | None = None,
    intersects: Any = None,
    datetime: Any = None,
    query: Any = None,
    filter: Any = None,
    sortby: Any = None,
    fields: Any = None,
    max_items: int | None = None,
    limit: int | None = None,
    signer: Any = None,
) -> Any:
    """Build a STAC item search and return the lazy ``ItemSearch`` object.

    Same query as :func:`search`, but nothing is fetched: the returned
    ``pystac_client.ItemSearch`` is the handle to the *unexecuted* search, so the
    caller can ask it for the total hit count or page through results instead of
    materialising every item at once:

    * ``.matched()`` — total number of hits, **or ``None``** when the server does
      not advertise a count (it is an optional STAC API field; never assume it).
    * ``.items()`` — stream items one by one, paging transparently.
    * ``.pages()`` — iterate whole result pages (one request per page).
    * ``.item_collection()`` — materialise everything, exactly what
      :func:`search` returns.

    Args:
        client_or_url: An open ``pystac_client.Client``, or a STAC API root URL
            (opened via :func:`pyramids.stac.open_client`, wiring `signer`).
        collections: Collection id, or a sequence of collection ids, to search.
        ids: Optional item id, or sequence of item ids, to fetch directly.
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
        fields: Optional ``fields`` extension selection — an include/exclude dict
            (``{"include": ["id"], "exclude": ["geometry"]}``) or a list of field
            names. Trimming required fields can yield Items that no longer
            validate, so prefer reading such a response as dicts.
        max_items: Optional cap on the total number of items returned (bounds
            paging at the API).
        limit: Optional page size forwarded to the API.
        signer: Optional signer used only when `client_or_url` is a URL (to open
            the client). Ignored when an open client is passed. A client opened
            here stays open, because the returned ``ItemSearch`` reads through
            it; it is closed only if the query is rejected.

    Returns:
        The ``pystac_client.ItemSearch`` for the query, not yet executed.

    Raises:
        OptionalPackageDoesNotExist: When `pystac-client` is not installed.
        ValueError: When both `bbox` and `intersects` are given (mutually
            exclusive per the STAC API spec), or when a `filter` is given but
            the endpoint does not advertise the CQL2 ``FILTER`` conformance
            class.

    Examples:
        - Count the hits before deciding to download anything (requires the
          `[stac]` extra and network access):
            ```python
            >>> from pyramids.stac import item_search  # doctest: +SKIP
            >>> hits = item_search(  # doctest: +SKIP
            ...     "https://earth-search.aws.element84.com/v1",
            ...     "sentinel-2-l2a",
            ...     bbox=(11.0, 46.0, 11.2, 46.2),
            ...     datetime="2023-06/2023-08",
            ... )
            >>> hits.matched()  # None when the server reports no count  # doctest: +SKIP
            128
            >>> first_page = next(hits.pages())  # doctest: +SKIP

            ```
    """
    _reject_both_aois(bbox, intersects)

    import_pystac_client(_STAC_INSTALL_HINT)
    from pystac_client import ConformanceClasses

    opened = isinstance(client_or_url, str)
    client = open_client(client_or_url, signer=signer) if opened else client_or_url
    # The returned `ItemSearch` keeps reading through this client, so it cannot
    # be closed here -- it is released only when the query is rejected and
    # nothing is handed back. `pop_all` is what hands ownership over.
    with contextlib.ExitStack() as stack:
        if opened:
            stack.callback(close_client, client)

        if filter is not None and not client.conforms_to(ConformanceClasses.FILTER):
            raise ValueError(
                "the STAC endpoint does not advertise the CQL2 FILTER conformance "
                "class, so a `filter` cannot be used against it."
            )

        if intersects is not None and hasattr(intersects, "__geo_interface__"):
            intersects = intersects.__geo_interface__

        search_object = client.search(
            collections=collections,
            ids=ids,
            bbox=bbox,
            intersects=intersects,
            datetime=datetime,
            query=query,
            filter=filter,
            sortby=sortby,
            fields=fields,
            max_items=max_items,
            limit=limit,
        )
        stack.pop_all()
    return search_object


def search(
    client_or_url: Any,
    collections: str | Sequence[str],
    *,
    ids: str | Sequence[str] | None = None,
    bbox: Sequence[float] | None = None,
    intersects: Any = None,
    datetime: Any = None,
    query: Any = None,
    filter: Any = None,
    sortby: Any = None,
    fields: Any = None,
    max_items: int | None = None,
    limit: int | None = None,
    signer: Any = None,
) -> Any:
    """Run a STAC item search and return the matched ``ItemCollection``.

    The eager counterpart of :func:`item_search`: it builds the very same query
    and immediately materialises it. Reach for :func:`item_search` instead when
    the hit count (``.matched()``) or page-by-page iteration is wanted — this
    function always returns a materialised ``ItemCollection``.

    Args:
        client_or_url: An open ``pystac_client.Client``, or a STAC API root URL
            (opened via :func:`pyramids.stac.open_client`, wiring `signer`).
        collections: Collection id, or a sequence of collection ids, to search.
        ids: Optional item id, or sequence of item ids, to fetch directly.
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
        fields: Optional ``fields`` extension selection — an include/exclude dict
            (``{"include": ["id"], "exclude": ["geometry"]}``) or a list of field
            names. Trimming required fields can yield Items that no longer
            validate, so prefer reading such a response as dicts.
        max_items: Optional cap on the total number of items returned (bounds
            paging at the API).
        limit: Optional page size forwarded to the API.
        signer: Optional signer used only when `client_or_url` is a URL (to open
            the client). Ignored when an open client is passed; a client opened
            from a URL here is closed again once the items are materialised.

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
        - Fetch two known items and keep only a couple of fields:
            ```python
            >>> items = search(  # doctest: +SKIP
            ...     "https://earth-search.aws.element84.com/v1",
            ...     "sentinel-2-l2a",
            ...     ids=["S2B_32TPS_20230601_0_L2A", "S2A_32TPS_20230606_0_L2A"],
            ...     fields={"include": ["id", "properties.datetime"]},
            ... )

            ```
    """
    _reject_both_aois(bbox, intersects)
    # Guarded here as well as in `item_search`: this function resolves the
    # client itself (to own its lifetime), so the missing-dependency error must
    # still come before any attempt to open one.
    import_pystac_client(_STAC_INSTALL_HINT)
    with resolved_client(client_or_url, signer=signer, opener=open_client) as client:
        items = item_search(
            client,
            collections,
            ids=ids,
            bbox=bbox,
            intersects=intersects,
            datetime=datetime,
            query=query,
            filter=filter,
            sortby=sortby,
            fields=fields,
            max_items=max_items,
            limit=limit,
        ).item_collection()
    return items
