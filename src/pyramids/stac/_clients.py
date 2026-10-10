"""Own the lifetime of a `pystac_client.Client` opened from a URL.

Every public STAC entry point that takes a *client or URL* argument
(:mod:`pyramids.stac.search`, :mod:`pyramids.stac.collections`) resolves it the
same way: an open ``Client`` is used as given, a URL is opened through
:func:`pyramids.stac.open_client`. Only the second case mints a
``requests.Session``, and only the function that minted it can release it — the
caller never sees that client.

So resolution and ownership are one decision, made here once:

* :func:`resolved_client` yields the client and, on the way out, closes it
  **only when it opened it**. A client handed in is left alone, because the
  caller is still using it (the usual pattern is one client, several queries).
* :func:`close_client` is the release itself. ``pystac_client.Client`` publishes
  no ``close()``, so the session is reached through its ``StacApiIO``; a future
  version growing a public ``close()`` is preferred automatically.

A lazy result is the one case this cannot cover: an ``ItemSearch`` keeps issuing
requests through the client it was built from, so the client has to outlive the
call that created it. :func:`pyramids.stac.item_search` therefore closes its
client only when the query is rejected and nothing is returned.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from pyramids.stac.client import open_client


def close_client(client: Any) -> None:
    """Release the HTTP session a `pystac_client.Client` holds.

    ``Client.open`` builds a ``requests.Session`` with mounted retry adapters
    and keeps it on the client's ``StacApiIO``. Dropping the client without
    closing it leaves the pooled connections to the garbage collector, which
    reports them as :class:`ResourceWarning` at an unrelated point in the
    program.

    The public ``close()`` is tried first so this keeps working if
    pystac-client grows one; the ``StacApiIO`` session is today's actual holder.
    A client exposing neither is left alone rather than poked at.

    Args:
        client: The client to release (duck-typed, like the rest of the STAC
            readers — tests pass a stand-in shaped like the real one).

    Examples:
        - The session of an already-released client is closed exactly once:
            ```python
            >>> from pyramids.stac._clients import close_client
            >>> class _Session:
            ...     def __init__(self):
            ...         self.closes = 0
            ...     def close(self):
            ...         self.closes += 1
            >>> class _IO:
            ...     def __init__(self):
            ...         self.session = _Session()
            >>> class _Client:
            ...     def __init__(self):
            ...         self._stac_io = _IO()
            >>> client = _Client()
            >>> close_client(client)
            >>> client._stac_io.session.closes
            1

            ```
        - A client with nothing to close is accepted silently:
            ```python
            >>> close_client(object())

            ```
    """
    session = getattr(getattr(client, "_stac_io", None), "session", None)
    closer = getattr(client, "close", None)
    if not callable(closer):
        closer = getattr(session, "close", None)
    if callable(closer):
        closer()


@contextlib.contextmanager
def resolved_client(
    client_or_url: Any,
    *,
    signer: Any = None,
    opener: Any = open_client,
) -> Iterator[Any]:
    """Yield the client for a *client or URL* argument, closing what it opened.

    Args:
        client_or_url: An open ``pystac_client.Client``, or a STAC API root URL
            to open through `opener`.
        signer: Optional signer, used only when a URL is opened.
        opener: The function a URL is opened with. Each caller passes its own
            module-level :func:`pyramids.stac.open_client` reference, so the
            indirection a test installs on that module is still honoured.

    Yields:
        The resolved client. When this context manager opened it, its session is
        closed on exit — including when the body raises. A client passed in is
        never closed.

    Examples:
        - A caller's client survives the block:
            ```python
            >>> from pyramids.stac._clients import resolved_client
            >>> class _Client:
            ...     def __init__(self):
            ...         self.closed = False
            ...     def close(self):
            ...         self.closed = True
            >>> client = _Client()
            >>> with resolved_client(client) as resolved:
            ...     resolved is client
            True
            >>> client.closed
            False

            ```
    """
    opened = isinstance(client_or_url, str)
    client = opener(client_or_url, signer=signer) if opened else client_or_url
    try:
        yield client
    finally:
        if opened:
            close_client(client)
