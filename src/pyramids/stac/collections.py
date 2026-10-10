"""Discover what a STAC API offers: its collections and its queryable fields.

:mod:`pyramids.stac.search` answers "which *items* match this query"; this module
answers the two questions that come before it:

* **What is here?** — :func:`list_collections` enumerates every collection the
  endpoint advertises, and :func:`search_collections` narrows that list at the
  API (free text, bounding box, time, CQL2) when the endpoint supports it.
* **What can I filter on?** — :func:`get_queryables` returns the JSON Schema of
  the properties a CQL2 ``filter`` may reference, for the whole endpoint or for a
  named set of collections.

Each helper takes the same first argument as :func:`pyramids.stac.search` — an
already-open ``pystac_client.Client`` or a STAC API root URL, in which case the
client is opened through :func:`pyramids.stac.open_client` so a `signer` is wired
into both of its hooks.

The server-side extensions these wrap are optional, so each helper checks the
relevant conformance class first and raises a clear :class:`ValueError` naming
the missing class, instead of letting pystac-client fail opaquely deeper down.

`pystac-client` is an optional dependency. Install with one of:

- PyPI: ``pip install 'pyramids-gis[stac]'``
- conda-forge: ``conda install -c conda-forge pyramids-stac``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pyramids.base._utils import extra_hint, import_pystac_client
from pyramids.stac.client import open_client

_STAC_INSTALL_HINT = extra_hint(
    "collection discovery requires the optional 'pystac-client' dependency.",
    "stac",
)


def list_collections(client_or_url: Any, *, signer: Any = None) -> list[dict[str, Any]]:
    """List every collection the endpoint advertises, as plain dicts.

    Walks ``Client.get_collections()``, which reads the ``/collections`` endpoint
    on a STAC API and falls back to the catalog's child links on a static
    catalog — so this works even against an endpoint advertising no extension at
    all, and needs no conformance gate.

    Args:
        client_or_url: An open ``pystac_client.Client``, or a STAC API root URL
            (opened via :func:`pyramids.stac.open_client`, wiring `signer`).
        signer: Optional signer used only when `client_or_url` is a URL (to open
            the client). Ignored when an open client is passed.

    Returns:
        One dict per collection, each the collection's STAC JSON (``id``,
        ``extent``, ``summaries``, ...). Empty when the endpoint exposes none.

    Raises:
        OptionalPackageDoesNotExist: When `pystac-client` is not installed.

    Examples:
        - List the collection ids of a public catalog (requires the `[stac]`
          extra and network access):
            ```python
            >>> from pyramids.stac import list_collections  # doctest: +SKIP
            >>> cols = list_collections("https://earth-search.aws.element84.com/v1")  # doctest: +SKIP
            >>> sorted(c["id"] for c in cols)[:2]  # doctest: +SKIP
            ['cop-dem-glo-30', 'cop-dem-glo-90']

            ```
    """
    import_pystac_client(_STAC_INSTALL_HINT)

    client = (
        open_client(client_or_url, signer=signer)
        if isinstance(client_or_url, str)
        else client_or_url
    )
    return [collection.to_dict() for collection in client.get_collections()]


def get_queryables(
    client_or_url: Any,
    collections: str | Sequence[str] | None = None,
    *,
    signer: Any = None,
) -> dict[str, Any]:
    """Return the JSON Schema of the fields a CQL2 ``filter`` may reference.

    With no `collections`, the endpoint-wide queryables are returned; with them,
    the per-collection schemas are merged (a property defined differently by two
    collections is dropped from the merge, per the STAC API Filter extension).

    Args:
        client_or_url: An open ``pystac_client.Client``, or a STAC API root URL
            (opened via :func:`pyramids.stac.open_client`, wiring `signer`).
        collections: Optional collection id, or sequence of ids, whose
            queryables are merged. ``None`` asks the endpoint for its global set.
        signer: Optional signer used only when `client_or_url` is a URL (to open
            the client). Ignored when an open client is passed.

    Returns:
        The queryables JSON Schema — a dict whose ``"properties"`` maps each
        queryable field name to its schema — suitable for ``jsonschema.validate``.

    Raises:
        OptionalPackageDoesNotExist: When `pystac-client` is not installed.
        ValueError: When the endpoint does not advertise the CQL2 ``FILTER``
            conformance class, so it exposes no queryables to read.

    Examples:
        - Discover what a collection can be filtered on (requires the `[stac]`
          extra and network access):
            ```python
            >>> from pyramids.stac import get_queryables  # doctest: +SKIP
            >>> schema = get_queryables(  # doctest: +SKIP
            ...     "https://earth-search.aws.element84.com/v1", "sentinel-2-l2a"
            ... )
            >>> "eo:cloud_cover" in schema["properties"]  # doctest: +SKIP
            True

            ```
    """
    import_pystac_client(_STAC_INSTALL_HINT)
    from pystac_client import ConformanceClasses

    client = (
        open_client(client_or_url, signer=signer)
        if isinstance(client_or_url, str)
        else client_or_url
    )

    if not client.conforms_to(ConformanceClasses.FILTER):
        raise ValueError(
            "the STAC endpoint does not advertise the CQL2 FILTER conformance "
            "class, so it publishes no queryables."
        )

    queryables: dict[str, Any]
    if collections is None:
        queryables = client.get_queryables()
    else:
        names = [collections] if isinstance(collections, str) else list(collections)
        queryables = client.get_merged_queryables(names)
    return queryables


def search_collections(
    client_or_url: Any,
    *,
    q: str | None = None,
    bbox: Sequence[float] | None = None,
    datetime: Any = None,
    query: Any = None,
    filter: Any = None,
    sortby: Any = None,
    fields: Any = None,
    max_collections: int | None = None,
    limit: int | None = None,
    signer: Any = None,
) -> list[dict[str, Any]]:
    """Search the endpoint's collections at the API, returning them as dicts.

    The collection-level counterpart of :func:`pyramids.stac.search`: the
    narrowing happens server-side, so an endpoint publishing hundreds of
    collections never has to be enumerated client-side. Requires the
    ``COLLECTION_SEARCH`` conformance class; free-text `q` additionally requires
    ``COLLECTION_SEARCH_FREE_TEXT``. Use :func:`list_collections` when the
    endpoint supports neither.

    Args:
        client_or_url: An open ``pystac_client.Client``, or a STAC API root URL
            (opened via :func:`pyramids.stac.open_client`, wiring `signer`).
        q: Optional free-text query matched against collection text fields
            (title, description, keywords).
        bbox: Optional `(minx, miny, maxx, maxy)` lon/lat box; keeps collections
            whose spatial extent intersects it.
        datetime: Optional RFC 3339 datetime or interval string; keeps
            collections whose temporal extent intersects it.
        query: Optional ``query`` extension dict.
        filter: Optional CQL2 filter (cql2-json dict or cql2-text string).
        sortby: Optional sort specification forwarded to the API.
        fields: Optional ``fields`` extension selection (include/exclude dict or
            list of field names). Trimming required fields yields partial
            collection dicts.
        max_collections: Optional cap on the total number of collections
            returned (bounds paging at the API).
        limit: Optional page size forwarded to the API.
        signer: Optional signer used only when `client_or_url` is a URL (to open
            the client). Ignored when an open client is passed.

    Returns:
        One dict per matching collection, in the order the API returned them.

    Raises:
        OptionalPackageDoesNotExist: When `pystac-client` is not installed.
        ValueError: When the endpoint does not advertise the
            ``COLLECTION_SEARCH`` conformance class, or when `q` is given and it
            does not advertise ``COLLECTION_SEARCH_FREE_TEXT``.

    Examples:
        - Find the elevation collections of a catalog (requires the `[stac]`
          extra and network access):
            ```python
            >>> from pyramids.stac import search_collections  # doctest: +SKIP
            >>> hits = search_collections(  # doctest: +SKIP
            ...     "https://planetarycomputer.microsoft.com/api/stac/v1",
            ...     q="elevation",
            ...     max_collections=5,
            ... )
            >>> [c["id"] for c in hits]  # doctest: +SKIP
            ['cop-dem-glo-30', 'cop-dem-glo-90']

            ```
    """
    import_pystac_client(_STAC_INSTALL_HINT)
    from pystac_client import ConformanceClasses

    client = (
        open_client(client_or_url, signer=signer)
        if isinstance(client_or_url, str)
        else client_or_url
    )

    if not client.conforms_to(ConformanceClasses.COLLECTION_SEARCH):
        raise ValueError(
            "the STAC endpoint does not advertise the COLLECTION_SEARCH "
            "conformance class, so its collections cannot be searched at the "
            "API; use list_collections() and filter the result instead."
        )

    if q is not None and not client.conforms_to(
        ConformanceClasses.COLLECTION_SEARCH_FREE_TEXT
    ):
        raise ValueError(
            "the STAC endpoint does not advertise the COLLECTION_SEARCH_FREE_TEXT "
            "conformance class, so the free-text `q` argument cannot be used "
            "against it."
        )

    result = client.collection_search(
        q=q,
        bbox=bbox,
        datetime=datetime,
        query=query,
        filter=filter,
        sortby=sortby,
        fields=fields,
        max_collections=max_collections,
        limit=limit,
    )
    return list(result.collections_as_dicts())
