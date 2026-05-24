"""Unit tests for pyramids.stac.signers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyramids.stac.signers import (
    AnonymousSigner,
    AWSRequesterPaysSigner,
    BearerTokenSigner,
    PlanetaryComputerSigner,
    Signer,
    _BaseSigner,
)

pytestmark = pytest.mark.core


class TestBaseSigner:
    """Tests for the _BaseSigner no-op defaults."""

    def test_sign_request_is_noop(self):
        """sign_request returns None (leave the request unchanged).

        Test scenario:
            The base signer does not touch outgoing requests.
        """
        assert (
            _BaseSigner().sign_request(object()) is None
        ), "Base sign_request must be a no-op"

    def test_sign_item_is_noop(self):
        """sign_item returns None (leave the item unchanged).

        Test scenario:
            The base signer does not mutate returned items.
        """
        assert (
            _BaseSigner().sign_item(object()) is None
        ), "Base sign_item must be a no-op"

    def test_sign_href_passthrough(self):
        """sign_href returns the href unchanged.

        Test scenario:
            The base signer rewrites no hrefs.
        """
        href = "https://example.com/a.tif"
        assert _BaseSigner().sign_href(href) == href, "Base sign_href must pass through"

    def test_gdal_env_empty(self):
        """gdal_env returns an empty mapping.

        Test scenario:
            The base signer contributes no GDAL config.
        """
        assert _BaseSigner().gdal_env() == {}, "Base gdal_env must be empty"


class TestAnonymousSigner:
    """Tests for AnonymousSigner."""

    def test_name(self):
        """The signer reports the `anonymous` name.

        Test scenario:
            `name` identifies the signer in logs/config.
        """
        assert AnonymousSigner().name == "anonymous", "Unexpected signer name"

    def test_all_boundaries_are_noops(self):
        """Every boundary is a no-op for the anonymous signer.

        Test scenario:
            request/item/href/env are all pass-through / empty.
        """
        signer = AnonymousSigner()
        assert signer.sign_request(object()) is None, "request should be untouched"
        assert signer.sign_item(object()) is None, "item should be untouched"
        assert signer.sign_href("x") == "x", "href should pass through"
        assert signer.gdal_env() == {}, "gdal_env should be empty"

    def test_satisfies_signer_protocol(self):
        """AnonymousSigner is recognised as a Signer.

        Test scenario:
            Structural runtime_checkable Protocol membership holds.
        """
        assert isinstance(
            AnonymousSigner(), Signer
        ), "Should satisfy the Signer protocol"


class TestAWSRequesterPaysSigner:
    """Tests for AWSRequesterPaysSigner."""

    def test_name(self):
        """The signer reports the `aws-requester-pays` name.

        Test scenario:
            `name` identifies the signer.
        """
        assert AWSRequesterPaysSigner().name == "aws-requester-pays", "Unexpected name"

    def test_region_default_none(self):
        """region defaults to None when not provided.

        Test scenario:
            Constructing without a region stores None.
        """
        assert AWSRequesterPaysSigner().region is None, "Default region should be None"

    def test_region_stored(self):
        """A provided region is stored.

        Test scenario:
            `region='us-west-2'` is retained for caller use.
        """
        assert (
            AWSRequesterPaysSigner(region="us-west-2").region == "us-west-2"
        ), "Region not stored"

    def test_gdal_env_requester_pays(self):
        """gdal_env opts into Requester-Pays and trims redundant calls.

        Test scenario:
            The env sets AWS_REQUEST_PAYER plus the cloud-read knobs.
        """
        env = AWSRequesterPaysSigner().gdal_env()
        assert env["AWS_REQUEST_PAYER"] == "requester", "Must opt into requester-pays"
        assert (
            env["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
        ), "Should disable readdir"
        assert env["CPL_VSIL_CURL_USE_HEAD"] == "NO", "Should disable HEAD"

    def test_request_and_href_are_noops(self):
        """The signer rewrites no requests or hrefs (read-side only).

        Test scenario:
            Only gdal_env matters for Requester-Pays.
        """
        signer = AWSRequesterPaysSigner()
        assert signer.sign_request(object()) is None, "request should be untouched"
        assert signer.sign_href("x") == "x", "href should pass through"

    def test_satisfies_signer_protocol(self):
        """AWSRequesterPaysSigner is recognised as a Signer.

        Test scenario:
            Structural Protocol membership holds.
        """
        assert isinstance(
            AWSRequesterPaysSigner(), Signer
        ), "Should satisfy the Signer protocol"


class TestBearerTokenSigner:
    """Tests for BearerTokenSigner."""

    def test_name(self):
        """The signer reports the `bearer` name.

        Test scenario:
            `name` identifies the signer.
        """
        assert BearerTokenSigner("t").name == "bearer", "Unexpected name"

    def test_sign_request_sets_header_static_token(self):
        """A static token is injected as an Authorization header.

        Test scenario:
            sign_request mutates and returns the request.
        """
        signer = BearerTokenSigner("abc123")
        request = SimpleNamespace(headers={})
        returned = signer.sign_request(request)
        assert request.headers["Authorization"] == "Bearer abc123", "Header not set"
        assert returned is request, "sign_request should return the same request object"

    def test_gdal_env_static_token(self):
        """gdal_env carries the bearer header for asset reads.

        Test scenario:
            GDAL_HTTP_HEADERS holds the Authorization header.
        """
        env = BearerTokenSigner("abc123").gdal_env()
        assert (
            env["GDAL_HTTP_HEADERS"] == "Authorization: Bearer abc123"
        ), "Bad GDAL header"

    def test_callable_token_resolved_each_use(self):
        """A callable token is resolved on every use.

        Test scenario:
            Mutating the source value changes the resolved token.
        """
        box = {"v": "first"}
        signer = BearerTokenSigner(lambda: box["v"])
        first = signer.gdal_env()["GDAL_HTTP_HEADERS"]
        box["v"] = "second"
        second = signer.gdal_env()["GDAL_HTTP_HEADERS"]
        assert (
            first == "Authorization: Bearer first"
        ), f"Unexpected first token: {first}"
        assert (
            second == "Authorization: Bearer second"
        ), f"Callable not re-resolved: {second}"

    def test_callable_token_in_request(self):
        """A callable token is used when signing a request.

        Test scenario:
            sign_request resolves the callable for the header.
        """
        request = SimpleNamespace(headers={})
        BearerTokenSigner(lambda: "fresh").sign_request(request)
        assert (
            request.headers["Authorization"] == "Bearer fresh"
        ), "Callable token not used"

    def test_item_and_href_are_noops(self):
        """Bearer signing does not rewrite items or hrefs.

        Test scenario:
            Only request/env carry the token.
        """
        signer = BearerTokenSigner("t")
        assert signer.sign_item(object()) is None, "item should be untouched"
        assert signer.sign_href("x") == "x", "href should pass through"

    def test_callable_returning_none_raises(self):
        """A callable token resolving to None raises instead of 'Bearer None'.

        Test scenario:
            A refresher that returns `None` is rejected by `gdal_env`.
        """
        signer = BearerTokenSigner(lambda: None)
        with pytest.raises(ValueError, match="non-empty string"):
            signer.gdal_env()

    def test_empty_token_raises(self):
        """An empty-string token is rejected (not sent as `Bearer`).

        Test scenario:
            `sign_request` with an empty token raises ValueError.
        """
        request = SimpleNamespace(headers={})
        with pytest.raises(ValueError, match="non-empty string"):
            BearerTokenSigner("").sign_request(request)

    def test_non_string_token_raises(self):
        """A non-string token (e.g. an int) is rejected.

        Test scenario:
            A callable returning a non-str raises ValueError.
        """
        signer = BearerTokenSigner(lambda: 12345)
        with pytest.raises(ValueError, match="non-empty string"):
            signer.gdal_env()

    def test_satisfies_signer_protocol(self):
        """BearerTokenSigner is recognised as a Signer.

        Test scenario:
            Structural Protocol membership holds.
        """
        assert isinstance(
            BearerTokenSigner("t"), Signer
        ), "Should satisfy the Signer protocol"


class TestPlanetaryComputerSigner:
    """Tests for the native PC SAS signer (PB-4); token fetch is stubbed."""

    @pytest.fixture
    def signer(self, monkeypatch):
        """A PC signer whose token fetch is stubbed (no network).

        Args:
            monkeypatch: pytest fixture.

        Returns:
            PlanetaryComputerSigner: records fetch calls in ``signer.fetches``;
            each fetch returns a token derived from the container with a
            far-future expiry.
        """
        s = PlanetaryComputerSigner()
        s.fetches = []

        def fake_fetch(account, container):
            s.fetches.append((account, container))
            return (f"sig=tok-{container}", 9_999_999_999.0)

        monkeypatch.setattr(s, "_fetch_token", fake_fetch)
        return s

    def test_is_a_signer(self, signer):
        """The PC signer satisfies the Signer protocol and is named.

        Test scenario:
            Structural protocol membership + the advertised name.
        """
        assert isinstance(signer, Signer), "PC signer should satisfy the Signer protocol"
        assert signer.name == "planetary-computer", f"unexpected name: {signer.name}"

    def test_gdal_env_is_empty(self, signer):
        """The credential rides the URL, so gdal_env() is empty.

        Test scenario:
            No GDAL config is needed for a SAS-signed /vsicurl href.
        """
        assert signer.gdal_env() == {}, f"PC signer gdal_env should be empty, got {signer.gdal_env()}"

    def test_signs_blob_href(self, signer):
        """A PC blob href gets the SAS token appended.

        Test scenario:
            account/container are parsed and ?<token> is appended.
        """
        out = signer.sign_href("https://sent2.blob.core.windows.net/sentinel/B04.tif")
        assert out == "https://sent2.blob.core.windows.net/sentinel/B04.tif?sig=tok-sentinel", out
        assert signer.fetches == [("sent2", "sentinel")], f"unexpected fetch: {signer.fetches}"

    def test_appends_with_ampersand_when_query_present(self, signer):
        """An existing query string gets the token appended with '&'.

        Test scenario:
            A blob href that already has a (non-SAS) query keeps it.
        """
        out = signer.sign_href("https://a.blob.core.windows.net/c/b.tif?foo=1")
        assert out == "https://a.blob.core.windows.net/c/b.tif?foo=1&sig=tok-c", out

    def test_non_blob_href_passthrough(self, signer):
        """A non-Azure href is returned unchanged and triggers no fetch.

        Test scenario:
            An s3:// href is not a PC blob.
        """
        href = "s3://bucket/scene.tif"
        assert signer.sign_href(href) == href, "non-blob href must pass through"
        assert signer.fetches == [], "no token should be fetched for a non-blob href"

    def test_public_bucket_not_signed(self, signer):
        """The public ai4edatasetspublicassets bucket is never signed.

        Test scenario:
            Public assets need no SAS token.
        """
        href = "https://ai4edatasetspublicassets.blob.core.windows.net/c/b.tif"
        assert signer.sign_href(href) == href, "public bucket must not be signed"
        assert signer.fetches == [], "public bucket should not trigger a fetch"

    def test_already_signed_href_untouched(self, signer):
        """An href already carrying SAS params is left as-is.

        Test scenario:
            Presence of se/sig means it is already signed.
        """
        href = "https://a.blob.core.windows.net/c/b.tif?se=2034&sig=abc"
        assert signer.sign_href(href) == href, "already-signed href must be untouched"
        assert signer.fetches == [], "already-signed href should not trigger a fetch"

    def test_token_cached_across_calls(self, signer):
        """A token is fetched once per (account, container) and reused.

        Test scenario:
            Two hrefs in the same container fetch only once.
        """
        signer.sign_href("https://a.blob.core.windows.net/c/one.tif")
        signer.sign_href("https://a.blob.core.windows.net/c/two.tif")
        assert signer.fetches == [("a", "c")], f"token should be cached, got {signer.fetches}"

    def test_sign_item_rewrites_dict_assets(self, signer):
        """sign_item rewrites every blob asset href on a raw-dict item in place.

        Test scenario:
            A dict item's blob asset is signed; a non-blob asset is untouched.
        """
        item = {
            "assets": {
                "B04": {"href": "https://a.blob.core.windows.net/c/B04.tif"},
                "thumb": {"href": "https://example.com/t.png"},
            }
        }
        assert signer.sign_item(item) is None, "sign_item must return None (modifier contract)"
        assert item["assets"]["B04"]["href"].endswith("?sig=tok-c"), item["assets"]["B04"]["href"]
        assert item["assets"]["thumb"]["href"] == "https://example.com/t.png", "non-blob untouched"

    def test_sign_item_handles_attribute_assets(self, signer):
        """sign_item rewrites .href on pystac-like asset objects.

        Test scenario:
            An item exposing .assets with objects bearing .href is signed.
        """
        asset = SimpleNamespace(href="https://a.blob.core.windows.net/c/B03.tif")
        item = SimpleNamespace(assets={"B03": asset})
        signer.sign_item(item)
        assert asset.href.endswith("?sig=tok-c"), f"attribute asset not signed: {asset.href}"

    def test_parse_expiry_handles_z_suffix_and_missing(self):
        """_parse_expiry parses RFC3339 'Z' and returns 0.0 for bad input.

        Test scenario:
            A 'Z'-suffixed timestamp parses to a positive epoch; None -> 0.0.
        """
        ts = PlanetaryComputerSigner._parse_expiry("2034-01-01T00:00:00Z")
        assert ts > 2_000_000_000.0, f"expected a far-future epoch, got {ts}"
        assert PlanetaryComputerSigner._parse_expiry(None) == 0.0, "missing expiry should be 0.0"


class TestSignerProtocol:
    """Tests for the Signer protocol membership rules."""

    def test_plain_object_is_not_a_signer(self):
        """An object lacking the signer methods is not a Signer.

        Test scenario:
            A bare object fails the structural check.
        """
        assert not isinstance(object(), Signer), "Bare object must not satisfy Signer"

    def test_duck_typed_object_is_a_signer(self):
        """Any object with the four methods + name satisfies the protocol.

        Test scenario:
            A SimpleNamespace exposing the surface is accepted.
        """
        duck = SimpleNamespace(
            name="duck",
            sign_request=lambda request: None,
            sign_item=lambda item: None,
            sign_href=lambda href: href,
            gdal_env=lambda: {},
        )
        assert isinstance(duck, Signer), "Duck-typed signer should satisfy the protocol"
