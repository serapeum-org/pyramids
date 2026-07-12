"""Tests for the PY-10 AWS Requester-Pays helpers on `pyramids.base.remote`.

Covers the new `CloudConfig.aws_request_payer` field + mutual-exclusion guard,
the `requester_pays_kwargs` / `s3fs_requester_pays_kwargs` helpers, and the
`RequesterPays` context manager (cost warning + GDAL knobs + set/restore).
"""

from __future__ import annotations

import warnings

import pytest
from osgeo import gdal

from pyramids.base.remote import (
    CloudConfig,
    RequesterPays,
    requester_pays_kwargs,
    s3fs_requester_pays_kwargs,
)

pytestmark = pytest.mark.core

_ACK_ENV = "PYRAMIDS_REQUESTER_PAYS_ACK"


class TestCloudConfigRequesterPays:
    """Tests for the aws_request_payer field and its guard."""

    def test_field_maps_to_gdal_option(self):
        """aws_request_payer=True maps to AWS_REQUEST_PAYER=requester.

        Test scenario:
            The new field surfaces in as_gdal_config.
        """
        cfg = CloudConfig(aws_request_payer=True).as_gdal_config()
        assert cfg["AWS_REQUEST_PAYER"] == "requester", f"Unexpected: {cfg}"

    def test_default_off_omits_key(self):
        """The option is absent unless requested.

        Test scenario:
            A default CloudConfig emits no AWS_REQUEST_PAYER key.
        """
        assert "AWS_REQUEST_PAYER" not in CloudConfig().as_gdal_config()

    def test_mutually_exclusive_with_anonymous(self):
        """Anonymous + Requester-Pays together raise ValueError.

        Test scenario:
            AWS rejects anonymous Requester-Pays, so the pair is refused.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):
            CloudConfig(aws_no_sign_request=True, aws_request_payer=True)


class TestRequesterPaysKwargs:
    """Tests for requester_pays_kwargs."""

    def test_boto3_kwargs(self):
        """Returns the per-call boto3 Requester-Pays kwarg.

        Test scenario:
            The mapping splats into `client.get_object(...)`.
        """
        assert requester_pays_kwargs() == {"RequestPayer": "requester"}


class TestS3fsRequesterPaysKwargs:
    """Tests for s3fs_requester_pays_kwargs."""

    def test_defaults_opt_in_and_forbid_anon(self):
        """Default kwargs enable Requester-Pays and disable anonymous access.

        Test scenario:
            No region given → just requester_pays + anon flags.
        """
        assert s3fs_requester_pays_kwargs() == {"requester_pays": True, "anon": False}

    def test_region_adds_client_kwargs(self):
        """A region pins the boto3 client region.

        Test scenario:
            region='us-west-2' adds client_kwargs while keeping the flags.
        """
        out = s3fs_requester_pays_kwargs(region="us-west-2")
        assert out["client_kwargs"] == {"region_name": "us-west-2"}
        assert out["requester_pays"] is True and out["anon"] is False


class TestRequesterPaysContextManager:
    """Tests for the RequesterPays context manager."""

    @pytest.fixture(autouse=True)
    def _clean_ack_env(self, monkeypatch):
        """Ensure the ack env var is unset for each test.

        Args:
            monkeypatch: pytest monkeypatch fixture.
        """
        monkeypatch.delenv(_ACK_ENV, raising=False)

    def test_warns_by_default(self):
        """Entering without acknowledgement emits a cost UserWarning.

        Test scenario:
            No ack_charges and no env var → UserWarning about billing.
        """
        with pytest.warns(UserWarning, match="Requester-Pays"):
            with RequesterPays(region="us-west-2"):
                # entering the context is what emits the billing warning
                pass

    def test_ack_charges_silences_warning(self):
        """ack_charges=True suppresses the cost warning.

        Test scenario:
            warnings-as-errors does not trip when ack_charges is set.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with RequesterPays(region="us-west-2", ack_charges=True):
                # entering must not raise: ack_charges suppresses the warning
                pass

    def test_env_ack_silences_warning(self, monkeypatch):
        """PYRAMIDS_REQUESTER_PAYS_ACK=1 suppresses the cost warning.

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            The env var acknowledges charges process-wide.
        """
        monkeypatch.setenv(_ACK_ENV, "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with RequesterPays(region="us-west-2"):
                # entering must not raise: the env var suppresses the warning
                pass

    def test_yields_config_with_payer_and_knobs(self):
        """The yielded config carries Requester-Pays and the cloud knobs.

        Test scenario:
            AWS_REQUEST_PAYER plus the four GDAL HTTP/VSI knobs and region.
        """
        with RequesterPays(region="us-west-2", ack_charges=True) as cfg:
            gdal_cfg = cfg.as_gdal_config()
        assert gdal_cfg["AWS_REQUEST_PAYER"] == "requester"
        assert gdal_cfg["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
        assert gdal_cfg["CPL_VSIL_CURL_USE_HEAD"] == "NO"
        assert gdal_cfg["GDAL_HTTP_MULTIPLEX"] == "YES"
        assert gdal_cfg["GDAL_HTTP_VERSION"] == "2"
        assert gdal_cfg["AWS_REGION"] == "us-west-2"

    def test_forwards_credentials(self):
        """Explicit AWS credentials are forwarded to the config.

        Test scenario:
            Key/secret/session token reach as_gdal_config.
        """
        with RequesterPays(
            aws_access_key_id="AK",
            aws_secret_access_key="SK",
            aws_session_token="TT",
            ack_charges=True,
        ) as cfg:
            gdal_cfg = cfg.as_gdal_config()
        assert gdal_cfg["AWS_ACCESS_KEY_ID"] == "AK"
        assert gdal_cfg["AWS_SECRET_ACCESS_KEY"] == "SK"
        assert gdal_cfg["AWS_SESSION_TOKEN"] == "TT"

    def test_sets_and_restores_gdal_option(self):
        """The GDAL option is set inside the block and restored on exit.

        Test scenario:
            AWS_REQUEST_PAYER is 'requester' inside and back to its prior
            value afterwards (thread-local GDAL config).
        """
        before = gdal.GetConfigOption("AWS_REQUEST_PAYER")
        with RequesterPays(ack_charges=True):
            inside = gdal.GetConfigOption("AWS_REQUEST_PAYER")
        after = gdal.GetConfigOption("AWS_REQUEST_PAYER")
        assert inside == "requester", "option not set inside the block"
        assert after == before, "option not restored on exit"
