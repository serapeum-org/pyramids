"""Unit tests for pyramids.stac.collections (collection discovery, STAC-13).

The helpers wrap ``pystac_client.Client``'s collection and queryable endpoints.
Tests drive a fake client (no network) and cover dict conversion, the merged vs
endpoint-wide queryables split, the COLLECTION_SEARCH / FREE_TEXT / FILTER
conformance gates, the URL-opens-a-client path, and the missing-pystac-client
guard.
"""

from __future__ import annotations

import sys
import types

import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
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


class _FakeClient:
    """Stand-in for pystac_client.Client with a declared conformance set."""

    def __init__(self, conforms=(), identifiers=("alpha", "beta")):
        self._conforms = set(conforms)
        self._identifiers = list(identifiers)
        self.collection_search_kwargs = None
        self.merged_queryables_for = None

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
