"""Unit tests for pyramids.stac.collections (collection discovery, STAC-13).

The helpers wrap ``pystac_client.Client``'s collection and queryable endpoints.
Tests drive a fake client (no network) and cover dict conversion, the merged vs
endpoint-wide queryables split, the COLLECTION_SEARCH / FREE_TEXT / FILTER
conformance gates, the URL-opens-a-client path, and the missing-pystac-client
guard.
"""

from __future__ import annotations

import collections.abc
import sys
import types

import pytest

import pyramids.stac as stac_package
from pyramids.base._errors import OptionalPackageDoesNotExist

try:  # pragma: no cover - exercised by whether the [stac] extra is installed
    from pystac_client.stac_api_io import StacApiIO
except ImportError:  # pragma: no cover - same
    StacApiIO = None

from pyramids.stac.collections import (
    get_queryables,
    list_collections,
    search_collections,
)

pytestmark = pytest.mark.core

_COLLECTIONS_MOD = sys.modules["pyramids.stac.collections"]

_FILTER = "filter"
_COLLECTION_SEARCH = "collection-search"
_FREE_TEXT = "collection-search#free-text"


class _FakeCollection:
    """Stand-in for a pystac.Collection carrying only its id."""

    def __init__(self, identifier):
        self.identifier = identifier

    def to_dict(self):
        """Return the collection's STAC JSON."""
        return {"id": self.identifier, "type": "Collection"}


class _FakeCollectionSearch:
    """Stand-in for pystac_client.CollectionSearch recording its kwargs."""

    def __init__(self, kwargs, identifiers):
        self.kwargs = kwargs
        self.identifiers = identifiers

    def collections_as_dicts(self):
        """Yield one dict per matching collection."""
        for identifier in self.identifiers:
            yield {"id": identifier, "type": "Collection"}


class _FakeSession:
    """Stand-in for the ``requests.Session`` a Client's ``StacApiIO`` holds.

    Modelled on the real attribute chain (``client._stac_io.session``) rather
    than on a ``Client.close()`` that pystac-client does not define, so a fake
    that is easier to close than the real thing cannot make the tests pass.
    """

    def __init__(self):
        self.closed = False

    def close(self):
        """Record that the session was released."""
        self.closed = True


class _FakeClient:
    """Stand-in for pystac_client.Client with a declared conformance set."""

    def __init__(self, conforms=(), identifiers=("alpha", "beta")):
        self._conforms = set(conforms)
        self._identifiers = list(identifiers)
        self.collection_search_kwargs = None
        self.merged_queryables_for = None
        self.session = _FakeSession()
        self._stac_io = types.SimpleNamespace(session=self.session)

    def conforms_to(self, conformance_class):
        """Report whether the (fake) endpoint advertises the class."""
        return conformance_class in self._conforms

    def get_collections(self):
        """Yield the endpoint's collections."""
        for identifier in self._identifiers:
            yield _FakeCollection(identifier)

    def get_queryables(self):
        """Return the endpoint-wide queryable schema."""
        return {"properties": {"datetime": {"type": "string"}}}

    def get_merged_queryables(self, collections):
        """Record the requested ids and return their merged queryable schema."""
        self.merged_queryables_for = collections
        return {"properties": {"eo:cloud_cover": {"type": "number"}}}

    def collection_search(self, **kwargs):
        """Record kwargs and return a fake search over the known collections."""
        self.collection_search_kwargs = kwargs
        return _FakeCollectionSearch(kwargs, self._identifiers)


@pytest.fixture(autouse=True)
def _stub_pystac_client(request, monkeypatch):
    """Stub the pystac-client import + ConformanceClasses so no extra is needed.

    Each helper calls ``import_pystac_client`` and some then do ``from
    pystac_client import ConformanceClasses``. We satisfy the guard and inject a
    tiny module exposing the three classes the helpers gate on, so the tests run
    without the extra installed.

    Tests marked ``stac`` deliberately want the real package (to pin the
    conformance-class names the stub would otherwise paper over), so the stub
    steps aside for them.
    """
    if request.node.get_closest_marker("stac") is None:
        monkeypatch.setattr(
            _COLLECTIONS_MOD, "import_pystac_client", lambda *a, **k: None
        )
        fake_mod = types.ModuleType("pystac_client")
        fake_mod.ConformanceClasses = types.SimpleNamespace(
            FILTER=_FILTER,
            COLLECTION_SEARCH=_COLLECTION_SEARCH,
            COLLECTION_SEARCH_FREE_TEXT=_FREE_TEXT,
        )
        monkeypatch.setitem(sys.modules, "pystac_client", fake_mod)


class TestListCollections:
    """Tests for list_collections()."""

    def test_returns_collection_dicts(self):
        """Every advertised collection comes back as its STAC JSON dict.

        Test scenario:
            A client yielding two collections produces two dicts, in order.
        """
        result = list_collections(_FakeClient())
        assert [c["id"] for c in result] == ["alpha", "beta"], (
            f"should return every collection in order, got {result}"
        )

    def test_empty_endpoint_returns_empty_list(self):
        """An endpoint with no collections degrades to an empty list.

        Test scenario:
            No conformance class is advertised and nothing is yielded; the
            helper returns [] rather than raising.
        """
        result = list_collections(_FakeClient(identifiers=()))
        assert result == [], f"an empty catalog should yield [], got {result}"

    def test_url_opens_client(self, monkeypatch):
        """A URL argument is opened into a client via open_client.

        Test scenario:
            list_collections(url, signer=...) calls open_client with the signer
            and lists the resulting client's collections.
        """
        opened = {}
        client = _FakeClient()

        def fake_open_client(url, *, signer=None):
            opened["url"] = url
            opened["signer"] = signer
            return client

        monkeypatch.setattr(_COLLECTIONS_MOD, "open_client", fake_open_client)
        result = list_collections("https://example.com/v1", signer="SIGNER")
        assert opened == {"url": "https://example.com/v1", "signer": "SIGNER"}, (
            f"URL/signer not forwarded to open_client: {opened}"
        )
        assert len(result) == 2, (
            f"should list the opened client's collections: {result}"
        )


class TestGetQueryables:
    """Tests for get_queryables()."""

    def test_endpoint_wide_schema(self):
        """With no collections the endpoint-wide queryables are returned.

        Test scenario:
            get_queryables() is used, not get_merged_queryables().
        """
        client = _FakeClient(conforms=[_FILTER])
        result = get_queryables(client)
        assert "datetime" in result["properties"], (
            f"should return the endpoint-wide schema, got {result}"
        )
        assert client.merged_queryables_for is None, (
            "the merged endpoint should not be called without collections"
        )

    def test_merged_schema_for_collections(self):
        """Named collections are merged through get_merged_queryables().

        Test scenario:
            A sequence of ids is forwarded as a list and the merged schema
            returned.
        """
        client = _FakeClient(conforms=[_FILTER])
        result = get_queryables(client, ["a", "b"])
        assert client.merged_queryables_for == ["a", "b"], (
            f"collections not forwarded: {client.merged_queryables_for}"
        )
        assert "eo:cloud_cover" in result["properties"], (
            f"should return the merged schema, got {result}"
        )

    def test_single_collection_string_is_wrapped(self):
        """A bare collection id is wrapped in a list before the request.

        Test scenario:
            get_queryables(client, "a") must not iterate the string's letters.
        """
        client = _FakeClient(conforms=[_FILTER])
        get_queryables(client, "a")
        assert client.merged_queryables_for == ["a"], (
            f"a string id should be wrapped, got {client.merged_queryables_for}"
        )

    def test_empty_collections_asks_for_the_global_set(self):
        """An empty sequence means the same as `None`, not an API error (N4).

        Test scenario:
            `[]` is not `None`, so it used to take the per-collection branch and
            surface pystac-client's "cannot get_merged_queryables from empty
            Iterable". It names no collection, so the documented `None`
            behaviour is the only sensible reading.
        """
        client = _FakeClient(conforms=[_FILTER])
        result = get_queryables(client, [])
        assert "datetime" in result["properties"], (
            f"an empty sequence should read the endpoint-wide schema, got {result}"
        )
        assert client.merged_queryables_for is None, (
            "the merged endpoint should not be called for an empty sequence"
        )

    def test_requires_filter_conformance(self):
        """A non-conforming endpoint raises a clear error, not an opaque one.

        Test scenario:
            conforms_to(FILTER) is False -> ValueError naming the class.
        """
        with pytest.raises(ValueError, match="FILTER conformance"):
            get_queryables(_FakeClient())


class TestSearchCollections:
    """Tests for search_collections()."""

    def test_forwards_query_kwargs(self):
        """Every typed kwarg reaches client.collection_search unchanged.

        Test scenario:
            q/bbox/datetime/max_collections/limit are forwarded verbatim and
            the matches come back as dicts.
        """
        client = _FakeClient(conforms=[_COLLECTION_SEARCH, _FREE_TEXT])
        result = search_collections(
            client,
            q="elevation",
            bbox=(11.0, 46.0, 11.2, 46.2),
            datetime="2023-06/2023-08",
            max_collections=5,
            limit=2,
        )
        kwargs = client.collection_search_kwargs
        assert kwargs["q"] == "elevation", f"q not forwarded: {kwargs}"
        assert kwargs["bbox"] == (11.0, 46.0, 11.2, 46.2), (
            f"bbox not forwarded: {kwargs}"
        )
        assert kwargs["datetime"] == "2023-06/2023-08", (
            f"datetime not forwarded: {kwargs}"
        )
        assert kwargs["max_collections"] == 5 and kwargs["limit"] == 2, (
            f"paging not forwarded: {kwargs}"
        )
        assert [c["id"] for c in result] == ["alpha", "beta"], (
            f"should return collection dicts, got {result}"
        )

    def test_requires_collection_search_conformance(self):
        """A non-conforming endpoint raises before any request.

        Test scenario:
            conforms_to(COLLECTION_SEARCH) is False -> ValueError pointing at
            list_collections() as the fallback.
        """
        client = _FakeClient()
        with pytest.raises(ValueError, match="COLLECTION_SEARCH conformance"):
            search_collections(client, q="elevation")
        assert client.collection_search_kwargs is None, (
            "the gate must fire before client.collection_search is called"
        )

    def test_free_text_requires_its_own_conformance(self):
        """`q` against an endpoint without FREE_TEXT raises a clear error.

        Test scenario:
            COLLECTION_SEARCH is advertised but COLLECTION_SEARCH_FREE_TEXT is
            not, so only the free-text argument is rejected.
        """
        client = _FakeClient(conforms=[_COLLECTION_SEARCH])
        with pytest.raises(ValueError, match="COLLECTION_SEARCH_FREE_TEXT"):
            search_collections(client, q="elevation")

    def test_non_free_text_search_without_free_text_conformance(self):
        """A spatial-only search needs COLLECTION_SEARCH alone.

        Test scenario:
            No `q` is given, so the FREE_TEXT gate is not consulted and the
            search runs.
        """
        client = _FakeClient(conforms=[_COLLECTION_SEARCH])
        result = search_collections(client, bbox=(0, 0, 1, 1))
        assert [c["id"] for c in result] == ["alpha", "beta"], (
            f"a bbox-only search should run, got {result}"
        )

    def test_url_opens_client(self, monkeypatch):
        """A URL argument is opened into a client via open_client.

        Test scenario:
            search_collections(url, signer=...) opens the client with the signer
            before searching.
        """
        opened = {}
        client = _FakeClient(conforms=[_COLLECTION_SEARCH])

        def fake_open_client(url, *, signer=None):
            opened["url"] = url
            opened["signer"] = signer
            return client

        monkeypatch.setattr(_COLLECTIONS_MOD, "open_client", fake_open_client)
        search_collections("https://example.com/v1", signer="SIGNER")
        assert opened == {"url": "https://example.com/v1", "signer": "SIGNER"}, (
            f"URL/signer not forwarded to open_client: {opened}"
        )


@pytest.mark.stac
class TestRealConformanceClasses:
    """The gates name conformance classes that exist in pystac-client itself.

    These run against the real ``pystac_client.ConformanceClasses`` (the stub is
    waived for the ``stac`` marker), so a renamed or misspelled class surfaces
    here instead of silently never gating.
    """

    def test_filter_gate_uses_the_real_class(self):
        """get_queryables() reads the real FILTER member, both ways.

        Test scenario:
            A client advertising the real FILTER enum member passes the gate; a
            client advertising nothing is rejected.
        """
        from pystac_client import ConformanceClasses

        passing = get_queryables(_FakeClient(conforms=[ConformanceClasses.FILTER]))
        assert "datetime" in passing["properties"], (
            f"the real FILTER member should pass the gate, got {passing}"
        )
        with pytest.raises(ValueError, match="FILTER conformance"):
            get_queryables(_FakeClient())

    def test_collection_search_gates_use_the_real_classes(self):
        """search_collections() reads the real COLLECTION_SEARCH members.

        Test scenario:
            COLLECTION_SEARCH alone allows a bbox search but not a free-text
            one; adding COLLECTION_SEARCH_FREE_TEXT allows `q` too.
        """
        from pystac_client import ConformanceClasses

        spatial_only = _FakeClient(conforms=[ConformanceClasses.COLLECTION_SEARCH])
        result = search_collections(spatial_only, bbox=(0, 0, 1, 1))
        assert [c["id"] for c in result] == ["alpha", "beta"], (
            f"the real COLLECTION_SEARCH member should pass the gate, got {result}"
        )
        with pytest.raises(ValueError, match="COLLECTION_SEARCH_FREE_TEXT"):
            search_collections(spatial_only, q="elevation")

        free_text = _FakeClient(
            conforms=[
                ConformanceClasses.COLLECTION_SEARCH,
                ConformanceClasses.COLLECTION_SEARCH_FREE_TEXT,
            ]
        )
        assert search_collections(free_text, q="elevation"), (
            "the real FREE_TEXT member should allow a free-text search"
        )


class TestClientLifetime:
    """A client opened from a URL is released again; a caller's is left alone.

    Test scenario:
        ``Client.open`` builds a ``requests.Session`` and keeps it alive on the
        returned object. Every helper here accepts a URL, so every one of them
        can mint a session the caller never sees and therefore cannot close —
        which is the leak. Ownership is the rule: close what this module opened,
        never what it was handed.
    """

    @pytest.fixture
    def opened(self, monkeypatch):
        """Make `open_client` hand back one recording client, and return it.

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Returns:
            _FakeClient: The client every URL call in the test will receive.
        """
        client = _FakeClient(conforms=[_FILTER, _COLLECTION_SEARCH, _FREE_TEXT])
        monkeypatch.setattr(
            _COLLECTIONS_MOD, "open_client", lambda url, *, signer=None: client
        )
        return client

    def test_list_collections_closes_the_client_it_opened(self, opened):
        """A URL call to list_collections releases its session."""
        list_collections("https://example.com/v1")
        assert opened.session.closed, "list_collections leaked its client session"

    def test_get_queryables_closes_the_client_it_opened(self, opened):
        """A URL call to get_queryables releases its session."""
        get_queryables("https://example.com/v1")
        assert opened.session.closed, "get_queryables leaked its client session"

    def test_search_collections_closes_the_client_it_opened(self, opened):
        """A URL call to search_collections releases its session."""
        search_collections("https://example.com/v1", bbox=(0, 0, 1, 1))
        assert opened.session.closed, "search_collections leaked its client session"

    def test_a_rejected_conformance_gate_still_closes_the_client(self, monkeypatch):
        """The session is released even when the gate refuses the request.

        Test scenario:
            The error path is the one a retry loop hits repeatedly, so a leak
            there accumulates fastest.
        """
        client = _FakeClient()
        monkeypatch.setattr(
            _COLLECTIONS_MOD, "open_client", lambda url, *, signer=None: client
        )
        with pytest.raises(ValueError, match="FILTER conformance"):
            get_queryables("https://example.com/v1")
        assert client.session.closed, "the rejected call leaked its client session"

    @pytest.mark.parametrize(
        "call",
        [
            list_collections,
            lambda client: get_queryables(client),
            lambda client: search_collections(client, bbox=(0, 0, 1, 1)),
        ],
        ids=["list_collections", "get_queryables", "search_collections"],
    )
    def test_a_caller_supplied_client_is_left_open(self, call):
        """A client passed in is not closed — the caller still owns it.

        Test scenario:
            Closing it would break the common pattern of opening one client and
            running several queries through it.
        """
        client = _FakeClient(conforms=[_FILTER, _COLLECTION_SEARCH, _FREE_TEXT])
        call(client)
        assert not client.session.closed, "a caller's client must not be closed"

    @pytest.mark.stac
    def test_the_real_client_exposes_the_session_that_gets_closed(self):
        """The attribute chain the close walks exists in pystac-client itself.

        Test scenario:
            ``Client`` publishes no ``close()``, so the session is reached
            through its ``StacApiIO``. Pinning that against the installed
            package means an upstream rename surfaces here rather than as a
            silent no-op release.
        """
        assert StacApiIO is not None, "the `stac` marker should gate this test"
        stac_io = StacApiIO()
        try:
            session = getattr(stac_io, "session", None)
            assert session is not None, "StacApiIO no longer exposes `session`"
            assert callable(getattr(session, "close", None)), (
                f"the session is no longer closable: {type(session).__name__}"
            )
        finally:
            stac_io.session.close()


class TestModuleNameDoesNotShadowTheStdlib:
    """`pyramids.stac.collections` is named after the STAC noun, safely (N3).

    Test scenario:
        The module shares its name with the stdlib `collections`, which is a
        footgun only if the resolution is ambiguous. Under absolute imports it
        is not: the sibling is reachable only as `pyramids.stac.collections`,
        so the module's own `from collections.abc import Sequence` reads the
        stdlib and no other importer's `collections` is affected. Renaming a
        module whose names are already exported would cost more than it buys,
        so the claim is pinned here instead.
    """

    def test_the_module_resolves_the_stdlib_collections(self):
        """`Sequence` in the module's namespace is `collections.abc.Sequence`."""
        assert _COLLECTIONS_MOD.Sequence is collections.abc.Sequence, (
            f"the sibling module shadowed the stdlib: {_COLLECTIONS_MOD.Sequence}"
        )

    def test_importing_it_leaves_the_stdlib_module_in_place(self):
        """`sys.modules['collections']` is still the stdlib module."""
        assert sys.modules["collections"] is collections, (
            "importing pyramids.stac.collections displaced the stdlib module"
        )
        assert _COLLECTIONS_MOD is not collections, (
            "the sibling and the stdlib module are the same object"
        )
        assert _COLLECTIONS_MOD.__name__ == "pyramids.stac.collections", (
            f"unexpected module name: {_COLLECTIONS_MOD.__name__}"
        )

    def test_the_public_names_are_reachable_without_the_module_path(self):
        """The package re-exports them, so callers need not spell the module.

        Test scenario:
            This is what makes the name a documentation question rather than an
            API one — `from pyramids.stac import list_collections` is the
            supported spelling.
        """
        for name in ("list_collections", "get_queryables", "search_collections"):
            assert name in stac_package.__all__, f"{name} is not re-exported"
            assert getattr(stac_package, name) is getattr(_COLLECTIONS_MOD, name), (
                f"{name} is re-exported as a different object"
            )


class TestCollectionsMissingDependency:
    """The missing-pystac-client guard fires before any client access."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda: list_collections("https://example.com/v1"),
            lambda: get_queryables("https://example.com/v1"),
            lambda: search_collections("https://example.com/v1"),
        ],
        ids=["list_collections", "get_queryables", "search_collections"],
    )
    def test_raises_optional_package_error(self, call, monkeypatch):
        """Each helper raises OptionalPackageDoesNotExist without the extra.

        Args:
            call: A zero-argument invocation of one helper.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            import_pystac_client raises -> the error points at the [stac] extra
            and no client is ever opened.
        """

        def _raise(*_a, **_k):
            raise OptionalPackageDoesNotExist(
                "collection discovery requires 'pystac-client'"
            )

        monkeypatch.setattr(_COLLECTIONS_MOD, "import_pystac_client", _raise)
        with pytest.raises(OptionalPackageDoesNotExist, match="pystac-client"):
            call()
