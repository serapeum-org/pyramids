"""Unit tests for pyramids.stac.search (typed STAC item-search helper, PB-3/M3).

The helper forwards a typed query to ``pystac_client.Client.search`` and returns
the matched ``ItemCollection``. Tests drive a fake client (no network) and cover
kwarg forwarding, the CQL2 conformance gate, shapely->GeoJSON conversion, the
URL-opens-a-client path, and the missing-pystac-client guard.

``item_search`` is the lazy sibling returning the un-executed ``ItemSearch``; its
tests check that the two share one query builder and that ``search`` keeps
returning a materialised ``ItemCollection``.
"""

from __future__ import annotations

import sys

import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.stac.search import item_search, search

pytestmark = pytest.mark.core

# The package exports a `search` function that shadows the `search` submodule
# attribute on `pyramids.stac`, so attribute-based lookups resolve to the
# function. Fetch the real module object from sys.modules to monkeypatch its
# globals (import_pystac_client / open_client).
_SEARCH_MOD = sys.modules["pyramids.stac.search"]


class _FakeSearch:
    """Stand-in for pystac_client.ItemSearch."""

    def __init__(self, kwargs):
        self.kwargs = kwargs

    def item_collection(self):
        """Return a sentinel standing in for the matched ItemCollection."""
        return ("ITEMS", self.kwargs)

    def matched(self):
        """Return a stand-in total hit count."""
        return 7

    def items(self):
        """Yield stand-in items one by one."""
        yield "ITEM-0"
        yield "ITEM-1"


class _FakeClient:
    """Stand-in for pystac_client.Client recording the search kwargs."""

    def __init__(self, conforms=True):
        self._conforms = conforms
        self.search_kwargs = None

    def conforms_to(self, _conformance_class):
        """Report whether the (fake) endpoint advertises the class."""
        return self._conforms

    def search(self, **kwargs):
        """Record kwargs and return a fake search whose item_collection echoes them."""
        self.search_kwargs = kwargs
        return _FakeSearch(kwargs)


@pytest.fixture(autouse=True)
def _stub_pystac_client(monkeypatch):
    """Stub the pystac-client import + ConformanceClasses so no extra is needed.

    ``search`` calls ``import_pystac_client`` then ``from pystac_client import
    ConformanceClasses``. We satisfy the guard and inject a tiny module exposing
    a ``ConformanceClasses.FILTER`` attribute so the test runs without the extra.
    """
    import types

    monkeypatch.setattr(_SEARCH_MOD, "import_pystac_client", lambda *a, **k: None)
    fake_mod = types.ModuleType("pystac_client")
    fake_mod.ConformanceClasses = types.SimpleNamespace(FILTER="filter")
    monkeypatch.setitem(sys.modules, "pystac_client", fake_mod)


class TestSearch:
    """Tests for the search() helper."""

    def test_forwards_query_kwargs(self):
        """All typed kwargs reach client.search unchanged.

        Test scenario:
            collections/bbox/datetime/query/max_items/limit are forwarded
            verbatim and the matched ItemCollection is returned.
        """
        client = _FakeClient()
        result, kwargs = search(
            client,
            "sentinel-2-l2a",
            bbox=(11.0, 46.0, 11.2, 46.2),
            datetime="2023-06/2023-08",
            query={"eo:cloud_cover": {"lt": 20}},
            max_items=10,
            limit=5,
        )
        assert result == "ITEMS", (
            f"should return the item_collection sentinel, got {result}"
        )
        assert kwargs["collections"] == "sentinel-2-l2a", (
            f"collections not forwarded: {kwargs}"
        )
        assert kwargs["bbox"] == (
            11.0,
            46.0,
            11.2,
            46.2,
        ), f"bbox not forwarded: {kwargs}"
        assert kwargs["datetime"] == "2023-06/2023-08", (
            f"datetime not forwarded: {kwargs}"
        )
        assert kwargs["max_items"] == 10 and kwargs["limit"] == 5, (
            f"paging not forwarded: {kwargs}"
        )

    def test_bbox_and_intersects_mutually_exclusive(self):
        """L2: passing both bbox and intersects raises before any client call.

        Test scenario:
            The STAC API forbids both; the helper rejects it early with a clear
            message (no client needed).
        """
        client = _FakeClient()
        with pytest.raises(ValueError, match="mutually exclusive"):
            search(
                client,
                "c",
                bbox=(0, 0, 1, 1),
                intersects={"type": "Point", "coordinates": [0, 0]},
            )

    def test_filter_requires_conformance(self):
        """A CQL2 filter against a non-conforming endpoint raises ValueError.

        Test scenario:
            conforms_to(FILTER) is False -> clear error before any search.
        """
        client = _FakeClient(conforms=False)
        with pytest.raises(ValueError, match="FILTER conformance"):
            search(client, "c", filter={"op": "=", "args": [{"property": "x"}, 1]})

    def test_filter_passes_when_conforming(self):
        """A filter is forwarded when the endpoint advertises FILTER.

        Test scenario:
            conforms_to True -> the filter reaches client.search.
        """
        client = _FakeClient(conforms=True)
        flt = {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 20]}
        _, kwargs = search(client, "c", filter=flt)
        assert kwargs["filter"] == flt, f"filter not forwarded: {kwargs}"

    def test_intersects_shapely_converted_to_geojson(self):
        """A shapely-like geometry is converted to a GeoJSON dict.

        Test scenario:
            An object exposing __geo_interface__ is replaced by that mapping
            before the request.
        """

        class _Geom:
            __geo_interface__ = {"type": "Point", "coordinates": [11.0, 46.0]}

        client = _FakeClient()
        _, kwargs = search(client, "c", intersects=_Geom())
        assert kwargs["intersects"] == {
            "type": "Point",
            "coordinates": [11.0, 46.0],
        }, f"shapely geom should convert to GeoJSON, got {kwargs['intersects']}"

    def test_intersects_geojson_dict_passthrough(self):
        """A GeoJSON dict is forwarded unchanged.

        Test scenario:
            A plain dict (no __geo_interface__) reaches client.search as-is.
        """
        geom = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
        client = _FakeClient()
        _, kwargs = search(client, "c", intersects=geom)
        assert kwargs["intersects"] == geom, (
            f"GeoJSON dict should pass through: {kwargs['intersects']}"
        )

    def test_url_opens_client(self, monkeypatch):
        """A URL argument is opened into a client via open_client.

        Test scenario:
            search(url, ...) calls open_client(url, signer=...) and searches the
            resulting client.
        """
        opened = {}
        client = _FakeClient()

        def fake_open_client(url, *, signer=None):
            opened["url"] = url
            opened["signer"] = signer
            return client

        monkeypatch.setattr(_SEARCH_MOD, "open_client", fake_open_client)
        result, _ = search("https://example.com/v1", "c", signer="SIGNER")
        assert opened["url"] == "https://example.com/v1", f"URL not opened: {opened}"
        assert opened["signer"] == "SIGNER", (
            f"signer not forwarded to open_client: {opened}"
        )
        assert result == "ITEMS", "should return the searched client's items"

    def test_forwards_ids(self):
        """STAC-14: `ids` reaches client.search unchanged.

        Test scenario:
            A sequence of item ids is forwarded verbatim so the API fetches
            those items directly.
        """
        client = _FakeClient()
        _, kwargs = search(client, "c", ids=["a", "b"])
        assert kwargs["ids"] == ["a", "b"], f"ids not forwarded: {kwargs}"

    def test_forwards_fields(self):
        """STAC-14: an include/exclude `fields` selection reaches client.search.

        Test scenario:
            The fields-extension dict is forwarded verbatim.
        """
        client = _FakeClient()
        selection = {"include": ["id"], "exclude": ["geometry"]}
        _, kwargs = search(client, "c", fields=selection)
        assert kwargs["fields"] == selection, f"fields not forwarded: {kwargs}"

    def test_ids_and_fields_default_to_none(self):
        """Omitting the new kwargs forwards None, leaving the query unchanged.

        Test scenario:
            A call written before STAC-14 produces the same request as before,
            with `ids`/`fields` explicitly unset.
        """
        client = _FakeClient()
        search(client, "c")
        assert client.search_kwargs["ids"] is None, (
            f"ids should default to None: {client.search_kwargs}"
        )
        assert client.search_kwargs["fields"] is None, (
            f"fields should default to None: {client.search_kwargs}"
        )


class TestItemSearch:
    """Tests for item_search(), the lazy counterpart exposing paging."""

    def test_returns_the_search_object(self):
        """item_search returns the ItemSearch itself, not an ItemCollection.

        Test scenario:
            The returned object exposes matched()/items()/pages-style access
            rather than the materialised item collection.
        """
        client = _FakeClient()
        result = item_search(client, "c")
        assert result.matched() == 7, (
            f"should expose the server hit count, got {result.matched()}"
        )
        assert list(result.items()) == ["ITEM-0", "ITEM-1"], (
            "should expose the item iterator"
        )

    def test_search_materialises_what_item_search_defers(self):
        """search() is item_search() plus item_collection(), same query.

        Test scenario:
            Both entry points build identical kwargs; only search() calls
            item_collection() on the result (backward-compatible default).
        """
        lazy_client = _FakeClient()
        eager_client = _FakeClient()
        deferred = item_search(lazy_client, "c", bbox=(0, 0, 1, 1), limit=3)
        eager, _ = search(eager_client, "c", bbox=(0, 0, 1, 1), limit=3)
        assert lazy_client.search_kwargs == eager_client.search_kwargs, (
            "both entry points should build the same query"
        )
        assert eager == "ITEMS", f"search should materialise the items, got {eager}"
        assert deferred.item_collection() == ("ITEMS", deferred.kwargs), (
            "the deferred search should still materialise on demand"
        )

    def test_shares_the_guards_with_search(self):
        """item_search keeps the bbox/intersects and FILTER guards.

        Test scenario:
            Both guards fire from the lazy entry point too, before any request.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):
            item_search(
                _FakeClient(),
                "c",
                bbox=(0, 0, 1, 1),
                intersects={"type": "Point", "coordinates": [0, 0]},
            )
        with pytest.raises(ValueError, match="FILTER conformance"):
            item_search(_FakeClient(conforms=False), "c", filter={"op": "isNull"})


class TestSearchMissingDependency:
    """The missing-pystac-client guard fires before any client access."""

    def test_raises_optional_package_error(self, monkeypatch):
        """search raises OptionalPackageDoesNotExist when the extra is absent.

        Test scenario:
            import_pystac_client raises -> the error points at the [stac] extra.
        """

        def _raise(*_a, **_k):
            raise OptionalPackageDoesNotExist("search requires 'pystac-client'")

        monkeypatch.setattr(_SEARCH_MOD, "import_pystac_client", _raise)
        with pytest.raises(OptionalPackageDoesNotExist, match="pystac-client"):
            search("https://example.com/v1", "c")
