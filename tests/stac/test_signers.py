"""Unit tests for pyramids.stac.signers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyramids.stac.signers import (
    AnonymousSigner,
    AWSRequesterPaysSigner,
    BearerTokenSigner,
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
        """The signer reports the ``anonymous`` name.

        Test scenario:
            ``name`` identifies the signer in logs/config.
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
        """The signer reports the ``aws-requester-pays`` name.

        Test scenario:
            ``name`` identifies the signer.
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
            ``region='us-west-2'`` is retained for caller use.
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
        """The signer reports the ``bearer`` name.

        Test scenario:
            ``name`` identifies the signer.
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

    def test_satisfies_signer_protocol(self):
        """BearerTokenSigner is recognised as a Signer.

        Test scenario:
            Structural Protocol membership holds.
        """
        assert isinstance(
            BearerTokenSigner("t"), Signer
        ), "Should satisfy the Signer protocol"


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
