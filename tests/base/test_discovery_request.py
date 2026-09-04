"""Tests for `discovery_request`, the one builder behind every pyramids HTTP discovery fetch.

Three call sites used to assemble their own header dict and their own preemptive
Basic credentials: the OGC API `/collections` pre-check, the ArcGIS
VectorTileServer metadata/tile fetch, and the remote-GeoJSON staging download.
Folding them into one builder is only safe while the *differences* between them
survive — each declares a different `User-Agent`, and a service that filters on
the agent string must keep seeing what it saw before.

These tests pin both halves: the builder's own contract (User-Agent default and
override, JSON negotiation, preemptive Basic auth, target URL) and the three
callers still sending exactly what they sent before the dedup.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from typing import Any

import pytest

from pyramids.base._ogc_api import (
    DISCOVERY_HEADERS,
    USER_AGENT,
    discovery_request,
    get_collections,
)
from pyramids.feature import _read
from pyramids.feature.collection import FeatureCollection

pytestmark = pytest.mark.core

_OGC_USER_AGENT = "pyramids-gis OGC API client"
_VTS_USER_AGENT = "pyramids-gis VectorTileServer client"

# A one-feature GeoJSON body, small enough to inline, valid enough for geopandas
# to read back off disk after the staging download.
_GEOJSON = json.dumps(
    {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "a"},
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
            }
        ],
    }
).encode()


def _basic(user: str, password: str) -> str:
    """Return the `Authorization` header value for HTTP Basic `user`/`password`."""
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def _decode_basic(header: str) -> tuple[str, str]:
    """Decode a `Basic <base64>` header back into its `(user, password)` pair.

    Args:
        header: The full `Authorization` header value.

    Returns:
        tuple[str, str]: The credentials the header carries.
    """
    scheme, _, token = header.partition(" ")
    assert scheme == "Basic", f"expected a Basic challenge, got {scheme!r}"
    user, _, password = base64.b64decode(token).decode().partition(":")
    return user, password


class TestDiscoveryRequest:
    """The builder's own contract: agent, content negotiation, auth, target."""

    def test_default_user_agent_is_the_plain_library_name(self):
        """With no `user_agent`, the request declares the bare library name.

        Test scenario:
            The remote-GeoJSON stage takes this default. If the fallback ever
            became one of the protocol-specific agents (the OGC API or
            VectorTileServer string), a plain GeoJSON download would start
            announcing itself as a protocol client it is not speaking.
        """
        request = discovery_request("https://host/data.geojson", None)

        assert request.get_header("User-agent") == USER_AGENT, (
            f"default agent should be {USER_AGENT!r}, got "
            f"{request.get_header('User-agent')!r}"
        )
        assert USER_AGENT == "pyramids-gis", (
            f"the plain library name is part of the wire contract, got {USER_AGENT!r}"
        )

    @pytest.mark.parametrize(
        "user_agent",
        [_OGC_USER_AGENT, _VTS_USER_AGENT, "pyramids-gis"],
        ids=["ogc-api", "vectortileserver", "plain"],
    )
    def test_user_agent_override_replaces_the_default(self, user_agent: str):
        """An explicit `user_agent` is sent verbatim, not merged with the default.

        Args:
            user_agent: The client string a caller declares.

        Test scenario:
            The three callers deliberately declare three different clients. If
            the override were ignored (or appended to the default), every
            request would collapse onto one agent string and an ArcGIS
            deployment filtering on the VectorTileServer agent would start
            refusing tile reads.
        """
        request = discovery_request("https://host/x", None, user_agent=user_agent)

        assert request.get_header("User-agent") == user_agent, (
            f"expected {user_agent!r}, got {request.get_header('User-agent')!r}"
        )

    def test_accept_json_default_negotiates_json(self):
        """`accept_json` defaults to True and sets `Accept: application/json`.

        Test scenario:
            The discovery documents are content-negotiated: a service that
            defaults to HTML answers the `/collections` pre-check with a page
            instead of a document unless JSON is asked for. Losing the default
            would turn a healthy endpoint into a "non-JSON body" error.
        """
        request = discovery_request("https://host/collections", None)

        assert request.get_header("Accept") == "application/json", (
            f"expected the JSON Accept header, got {request.get_header('Accept')!r}"
        )
        assert request.get_header("Accept") == DISCOVERY_HEADERS["Accept"], (
            "the builder must reuse the shared Accept constant, not its own literal"
        )

    def test_accept_json_false_sends_no_accept_header(self):
        """`accept_json=False` sets no `Accept` header at all.

        Test scenario:
            A vector tile is protobuf, not JSON. Sending `Accept:
            application/json` on a `.pbf` fetch invites a strict server to
            answer 406, so the flag must remove the header rather than swap it
            for a wildcard.
        """
        request = discovery_request("https://host/1/2/3.pbf", None, accept_json=False)

        assert request.get_header("Accept") is None, (
            f"a non-JSON fetch must send no Accept, got {request.get_header('Accept')!r}"
        )

    def test_auth_is_sent_preemptively_and_round_trips(self):
        """`auth=(user, pass)` puts a decodable `Authorization: Basic` on the request.

        Test scenario:
            A service that answers 403 without a 401 challenge never gets
            credentials out of a reactive `HTTPBasicAuthHandler`. The header has
            to be on the first request, and the base64 payload has to decode
            back to the exact pair the caller passed — a swapped or re-encoded
            pair would authenticate as nobody.
        """
        request = discovery_request("https://host/x", ("ada", "s3cret:with:colons"))

        header = request.get_header("Authorization")
        assert header is not None, "credentials must be on the first request"
        assert header == _basic("ada", "s3cret:with:colons"), (
            f"unexpected Basic payload: {header!r}"
        )
        user, password = _decode_basic(header)
        assert (user, password) == ("ada", "s3cret:with:colons"), (
            f"credentials did not round-trip: {(user, password)!r}"
        )

    def test_no_auth_sends_no_authorization_header(self):
        """`auth=None` leaves the request unauthenticated.

        Test scenario:
            An open endpoint must not receive an empty or `None:None` Basic
            header — some services reject a malformed challenge outright rather
            than ignoring it.
        """
        request = discovery_request("https://host/x", None)

        assert request.get_header("Authorization") is None, (
            f"unauthenticated request carries {request.get_header('Authorization')!r}"
        )

    def test_request_targets_the_given_url(self):
        """The built request points at the URL it was handed, unmodified.

        Test scenario:
            The callers pre-compose their URLs (query-string auth tokens,
            `f=json`, tile coordinates). A builder that normalised or re-joined
            the URL would silently drop an ArcGIS `?token=…` and turn every
            secured read into a 403.
        """
        url = "https://host/ogc/collections?api_key=secret&f=json"
        request = discovery_request(url, None)

        assert isinstance(request, urllib.request.Request), (
            f"expected a urllib Request, got {type(request)!r}"
        )
        assert request.full_url == url, f"expected {url!r}, got {request.full_url!r}"
        assert request.get_method() == "GET", (
            f"a discovery fetch must be a GET, got {request.get_method()!r}"
        )


class TestDiscoveryRequestCallers:
    """The three call sites still send what they sent before the dedup."""

    def test_get_collections_declares_the_ogc_api_client(self, monkeypatch):
        """The `/collections` pre-check keeps its OGC API User-Agent and JSON Accept.

        Args:
            monkeypatch: pytest fixture, used to stub the network seam.

        Test scenario:
            The pre-check guards a much larger GDAL read; it is the request a
            service sees first. Losing the protocol-specific agent would change
            what an OGC API deployment logs and, where it filters, whether the
            collection list is served at all.
        """
        captured: list[Any] = []

        def _fake_get(request, timeout, **kwargs):
            captured.append((request, timeout))
            return b'{"collections": [{"id": "lakes"}]}'

        monkeypatch.setattr("pyramids.base._ogc_api.http_get_with_retry", _fake_get)
        get_collections.cache_clear()
        try:
            names = get_collections("https://host/ogc", None, 30.0)
        finally:
            get_collections.cache_clear()

        assert names == frozenset({"lakes"}), f"unexpected collections: {names}"
        assert len(captured) == 1, f"expected one fetch, got {len(captured)}"
        request, _timeout = captured[0]
        assert request.get_header("User-agent") == _OGC_USER_AGENT, (
            f"expected the OGC API agent, got {request.get_header('User-agent')!r}"
        )
        assert request.get_header("Accept") == "application/json", (
            "the discovery document must still be negotiated as JSON"
        )
        assert request.full_url.endswith("/collections?f=json"), (
            f"unexpected discovery URL: {request.full_url!r}"
        )

    def test_get_collections_sends_preemptive_basic_auth(self, monkeypatch):
        """Credentials given to `get_collections` reach the request itself.

        Args:
            monkeypatch: pytest fixture, used to stub the network seam.

        Test scenario:
            A secured OGC API endpoint that answers 403 rather than 401 would
            never see the credentials if they went into a reactive handler, so
            the pre-check would fail before the authenticated GDAL read even
            started.
        """
        captured: list[Any] = []

        def _fake_get(request, timeout, **kwargs):
            captured.append(request)
            return b'{"collections": []}'

        monkeypatch.setattr("pyramids.base._ogc_api.http_get_with_retry", _fake_get)
        get_collections.cache_clear()
        try:
            get_collections("https://host/ogc", ("ada", "s3cret"), 30.0)
        finally:
            get_collections.cache_clear()

        header = captured[0].get_header("Authorization")
        assert _decode_basic(header) == ("ada", "s3cret"), (
            f"credentials did not reach the pre-check: {header!r}"
        )

    def test_vts_request_keeps_the_vectortileserver_user_agent(self):
        """Both VectorTileServer fetches still declare the ArcGIS-facing agent.

        Test scenario:
            Some ArcGIS deployments filter on the User-Agent. Routing the fetch
            through the shared builder must not let it fall back to the plain
            library name, or the metadata and tile reads start being refused by
            exactly the deployments the string was added for.
        """
        meta = _read._vts_request(
            "https://host/VectorTileServer", None, accept_json=True
        )
        tile = _read._vts_request(
            "https://host/tile/1/2/3.pbf", None, accept_json=False
        )

        assert meta.get_header("User-agent") == _VTS_USER_AGENT, (
            f"metadata agent changed: {meta.get_header('User-agent')!r}"
        )
        assert tile.get_header("User-agent") == _VTS_USER_AGENT, (
            f"tile agent changed: {tile.get_header('User-agent')!r}"
        )
        assert meta.get_header("Accept") == "application/json", (
            "the metadata fetch still negotiates JSON"
        )
        assert tile.get_header("Accept") is None, (
            "a protobuf tile fetch must not ask for JSON"
        )

    def test_remote_geojson_stage_takes_the_default_agent(self, monkeypatch):
        """The staging download sends the plain agent, no Accept and no credentials.

        Args:
            monkeypatch: pytest fixture, used to stub `urllib.request.urlopen`.

        Test scenario:
            This stage exists to keep GDAL out of the remote read (issue #1008),
            so its request is a bare https GET. If the shared builder started
            attaching a JSON `Accept` or a protocol agent here, a plain GeoJSON
            host could answer differently than the one the workaround was
            written against.
        """
        captured: list[Any] = []

        def _fake_urlopen(request, timeout=None):
            captured.append(request)
            return io.BytesIO(_GEOJSON)

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        result = _read._read_remote_geojson_staged(
            FeatureCollection, "https://host/data.geojson", {}
        )

        assert isinstance(result, FeatureCollection), (
            f"expected a FeatureCollection, got {type(result)!r}"
        )
        assert len(captured) == 1, f"expected one download, got {len(captured)}"
        request = captured[0]
        assert request.get_header("User-agent") == USER_AGENT, (
            f"the stage should take the default agent, got "
            f"{request.get_header('User-agent')!r}"
        )
        assert request.get_header("Accept") is None, (
            "the staged download asks for no particular content type"
        )
        assert request.get_header("Authorization") is None, (
            "the staged download sends no credentials"
        )
        assert request.full_url == "https://host/data.geojson", (
            f"unexpected download URL: {request.full_url!r}"
        )

    def test_the_three_callers_declare_three_distinct_agents(self, monkeypatch):
        """The three User-Agents stay three, not one.

        Args:
            monkeypatch: pytest fixture, used to stub the two network seams.

        Test scenario:
            This is the whole risk of folding three request builders into one:
            the agent strings are a real, deliberate difference and not copies
            that drifted. A future "simplification" that unified them would pass
            every other test in this file while changing what three different
            services see.
        """
        captured: list[Any] = []

        def _fake_get(request, timeout, **kwargs):
            captured.append(request)
            return b'{"collections": []}'

        def _fake_urlopen(request, timeout=None):
            captured.append(request)
            return io.BytesIO(_GEOJSON)

        monkeypatch.setattr("pyramids.base._ogc_api.http_get_with_retry", _fake_get)
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        get_collections.cache_clear()
        try:
            get_collections("https://host/ogc", None, 30.0)
        finally:
            get_collections.cache_clear()
        captured.append(_read._vts_request("https://host/x", None, accept_json=True))
        _read._read_remote_geojson_staged(
            FeatureCollection, "https://host/data.geojson", {}
        )

        agents = [request.get_header("User-agent") for request in captured]
        assert agents == [_OGC_USER_AGENT, _VTS_USER_AGENT, USER_AGENT], (
            f"the three callers no longer declare three distinct clients: {agents}"
        )
        assert len(set(agents)) == 3, (
            f"agent strings collapsed onto each other: {agents}"
        )
