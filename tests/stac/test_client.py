"""Unit tests for pyramids.stac.client.open_client."""

from __future__ import annotations

import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.stac.client import open_client
from pyramids.stac.signers import BearerTokenSigner


@pytest.mark.stac
class TestOpenClient:
    """Tests for open_client (require pystac-client; Client.open is mocked)."""

    def test_wires_signer_into_both_hooks(self, mocker):
        """The signer's methods are passed as both pystac-client hooks.

        Args:
            mocker: pytest-mock fixture.

        Test scenario:
            `modifier` is `signer.sign_item` and `request_modifier` is
            `signer.sign_request`.
        """
        import pystac_client

        mock_open = mocker.patch.object(
            pystac_client.Client, "open", return_value="CLIENT"
        )
        signer = BearerTokenSigner("tok")
        result = open_client("https://example.com/v1", signer=signer)
        assert result == "CLIENT", "open_client should return the Client.open result"
        kwargs = mock_open.call_args.kwargs
        assert kwargs["modifier"] == signer.sign_item, "modifier should be sign_item"
        assert (
            kwargs["request_modifier"] == signer.sign_request
        ), "request_modifier should be sign_request"

    def test_defaults_to_anonymous_signer(self, mocker):
        """With no signer, an AnonymousSigner is wired in.

        Args:
            mocker: pytest-mock fixture.

        Test scenario:
            The modifier is bound to a signer named `anonymous`.
        """
        import pystac_client

        mock_open = mocker.patch.object(pystac_client.Client, "open", return_value="C")
        open_client("https://example.com/v1")
        modifier = mock_open.call_args.kwargs["modifier"]
        assert (
            modifier.__self__.name == "anonymous"
        ), "Default signer should be anonymous"

    def test_forwards_url_headers_timeout(self, mocker):
        """url, headers, and timeout are forwarded to Client.open.

        Args:
            mocker: pytest-mock fixture.

        Test scenario:
            The wrapper passes its arguments through unchanged.
        """
        import pystac_client

        mock_open = mocker.patch.object(pystac_client.Client, "open", return_value="C")
        open_client("https://example.com/v1", headers={"X-Api": "y"}, timeout=10)
        args, kwargs = mock_open.call_args
        assert args[0] == "https://example.com/v1", "URL should be forwarded"
        assert kwargs["headers"] == {"X-Api": "y"}, "headers should be forwarded"
        assert kwargs["timeout"] == 10, "timeout should be forwarded"


class TestOpenClientMissingDependency:
    """Tests for the missing-pystac-client guard (no extra required)."""

    pytestmark = pytest.mark.core

    def test_raises_optional_package_error(self, mocker):
        """open_client raises when pystac-client is not importable.

        Args:
            mocker: pytest-mock fixture.

        Test scenario:
            The import guard fires before any network/Client access and the
            error message points at the [stac] extra.
        """
        mocker.patch(
            "pyramids.stac.client.import_pystac_client",
            side_effect=OptionalPackageDoesNotExist(
                "open_client requires the optional 'pystac-client' dependency."
            ),
        )
        with pytest.raises(OptionalPackageDoesNotExist, match="pystac-client"):
            open_client("https://example.com/v1")
