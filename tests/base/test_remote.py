"""Unit tests for pyramids.base.remote."""

from __future__ import annotations

import os
import warnings
from contextlib import nullcontext

import numpy as np  # noqa: E402
import pytest
from osgeo import gdal

from pyramids import _io
from pyramids.base.remote import (
    _BUCKET_URL_SCHEMES,
    _CLOUD_VSI_PREFIXES,
    _NETWORK_VSI_PREFIXES,
    _OBJECT_STORE_VSI_PREFIXES,
    _VSI_PREFIXES,
    URL_SCHEMES,
    CloudConfig,
    _chain_archive_vsi,
    _to_vsi,
    is_network_backed,
    is_remote,
    redact_credentials,
    signer_cloud_config,
)
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


class _EnvSigner:
    """Minimal signer stand-in exposing only ``gdal_env()``."""

    def __init__(self, env):
        self._env = dict(env)

    def gdal_env(self):
        """Return the advertised GDAL config mapping."""
        return dict(self._env)


class TestSignerCloudConfig:
    """Tests for the shared ``signer_cloud_config`` helper (M4)."""

    def test_none_returns_nullcontext(self):
        """A ``None`` signer yields a no-op nullcontext.

        Test scenario:
            ``signer_cloud_config(None)`` installs no GDAL config.
        """
        assert isinstance(signer_cloud_config(None), nullcontext), (
            "expected a no-op context"
        )

    def test_signer_returns_seeded_cloudconfig(self):
        """A signer yields a CloudConfig carrying its ``gdal_env()``.

        Test scenario:
            The returned CloudConfig's config equals the signer's env.
        """
        ctx = signer_cloud_config(_EnvSigner({"AWS_REQUEST_PAYER": "requester"}))
        assert isinstance(ctx, CloudConfig), f"expected CloudConfig, got {type(ctx)}"
        assert ctx.as_gdal_config() == {"AWS_REQUEST_PAYER": "requester"}, (
            f"unexpected config: {ctx.as_gdal_config()}"
        )

    def test_config_active_within_block(self):
        """The signer's config is live inside the block and torn down after.

        Test scenario:
            ``GDAL_HTTP_TIMEOUT`` is set within the block and unset afterwards.
        """
        assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") is None, "precondition: unset"
        with signer_cloud_config(_EnvSigner({"GDAL_HTTP_TIMEOUT": "42"})):
            assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") == "42", (
                "config not active in block"
            )
        assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") is None, (
            "config not restored after block"
        )


class TestToVsi:
    def test_s3(self):
        assert _to_vsi("s3://bucket/key.tif") == "/vsis3/bucket/key.tif"

    def test_s3_nested_path(self):
        assert _to_vsi("s3://bucket/a/b/c/key.tif") == "/vsis3/bucket/a/b/c/key.tif"

    def test_gs(self):
        assert _to_vsi("gs://bucket/key.tif") == "/vsigs/bucket/key.tif"

    def test_az(self):
        assert _to_vsi("az://container/blob.tif") == "/vsiaz/container/blob.tif"

    def test_abfs_maps_to_vsiadls(self):
        """`abfs://` is the Gen2 scheme, so it routes to the Gen2 handler.

        It mapped to `/vsiaz/` (Blob) until #918. `abfs` is the Azure Blob
        *File System* driver, which is Data Lake Gen2 everywhere else in the
        Azure and Hadoop ecosystems; a flat Blob account is reached with
        `az://`.
        """
        assert _to_vsi("abfs://container/blob.tif") == "/vsiadls/container/blob.tif"

    def test_abfss_maps_to_vsiadls(self):
        """The TLS spelling of the Gen2 scheme resolves the same way."""
        assert _to_vsi("abfss://container/blob.tif") == "/vsiadls/container/blob.tif"

    def test_there_is_no_adls_scheme(self):
        """`adls://` is not a real scheme, so it is not accepted.

        The registered Azure Data Lake name is `adl://` (Gen1, which GDAL does
        not handle). Inventing `adls://` would collide with it visually and fail
        in the fsspec-backed readers, which resolve the scheme themselves.
        """
        assert "adls" not in URL_SCHEMES, "adls:// is not an ecosystem scheme"
        assert _to_vsi("adls://container/blob.tif") == "adls://container/blob.tif"

    @pytest.mark.parametrize("configured", [None, "prodlake", "otheracct"])
    def test_gen2_account_authority_rewrites_deterministically(self, configured):
        """The rewrite does not depend on ambient credentials.

        Args:
            configured: The `AZURE_STORAGE_ACCOUNT` in force, or `None`.

        Test scenario:
            `_to_vsi` runs on essentially every open, so the same URL must
            produce the same path whether or not credentials are configured and
            whether or not a `CloudConfig` block is active.
        """
        with gdal.config_option("AZURE_STORAGE_ACCOUNT", configured):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resolved = _to_vsi(
                    "abfss://raw@prodlake.dfs.core.windows.net/2026/x.tif"
                )
        assert resolved == "/vsiadls/raw/2026/x.tif", resolved

    def test_a_conflicting_account_warns_rather_than_raising(self):
        """A URL account that disagrees with the configured one is surfaced.

        Test scenario:
            GDAL takes the account from configuration, so the read would go
            somewhere other than the URL names. That is worth flagging — but not
            by raising from a path-rewriting helper on the open path.
        """
        with gdal.config_option("AZURE_STORAGE_ACCOUNT", "otheracct"):
            with pytest.warns(UserWarning, match="otheracct"):
                _to_vsi("abfss://raw@prodlake.dfs.core.windows.net/x.tif")

    def test_a_matching_account_is_silent(self):
        """No warning when the URL and the configuration agree."""
        with gdal.config_option("AZURE_STORAGE_ACCOUNT", "prodlake"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert (
                    _to_vsi("abfss://raw@prodlake.dfs.core.windows.net/x.tif")
                    == "/vsiadls/raw/x.tif"
                )

    def test_the_account_may_come_from_the_connection_string(self):
        """GDAL accepts the account embedded in the connection string.

        Test scenario:
            Reading only `AZURE_STORAGE_ACCOUNT` warned about a conflict on a
            setup that is perfectly valid for `/vsiadls/`.
        """
        connection = "DefaultEndpointsProtocol=https;AccountName=prodlake;AccountKey=k"
        with gdal.config_option("AZURE_STORAGE_CONNECTION_STRING", connection):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert (
                    _to_vsi("abfss://raw@prodlake.dfs.core.windows.net/x.tif")
                    == "/vsiadls/raw/x.tif"
                )

    def test_bare_gen2_authority_is_still_a_plain_container(self):
        """Without an `@`, the authority is the filesystem name as before."""
        assert _to_vsi("abfs://fs/2026/x.tif") == "/vsiadls/fs/2026/x.tif"

    def test_https_simple(self):
        assert _to_vsi("https://foo.com/x.tif") == "/vsicurl/https://foo.com/x.tif"

    def test_https_with_query(self):
        url = "https://foo.com/x.tif?sig=abc&exp=123"
        assert _to_vsi(url) == f"/vsicurl/{url}"

    def test_http_plain(self):
        assert _to_vsi("http://foo.com/x.tif") == "/vsicurl/http://foo.com/x.tif"

    def test_file_uri_posix(self):
        assert _to_vsi("file:///srv/data/x.tif") == "/srv/data/x.tif"

    def test_file_uri_windows(self):
        assert _to_vsi("file:///C:/data/x.tif") == "C:/data/x.tif"

    def test_already_vsi_s3_passthrough(self):
        assert _to_vsi("/vsis3/bucket/key.tif") == "/vsis3/bucket/key.tif"

    def test_already_vsi_curl_passthrough(self):
        assert _to_vsi("/vsicurl/https://x/y.tif") == "/vsicurl/https://x/y.tif"

    def test_already_vsi_mem_passthrough(self):
        assert _to_vsi("/vsimem/temp.tif") == "/vsimem/temp.tif"

    def test_already_vsi_zip_passthrough(self):
        assert _to_vsi("/vsizip/a.zip/b.tif") == "/vsizip/a.zip/b.tif"

    def test_local_posix_unchanged(self):
        assert _to_vsi("/home/user/data.tif") == "/home/user/data.tif"

    def test_local_windows_drive_unchanged(self):
        assert _to_vsi("C:/data/x.tif") == "C:/data/x.tif"

    def test_relative_path_unchanged(self):
        assert _to_vsi("data/x.tif") == "data/x.tif"

    def test_dods_maps_to_netcdf_dap(self):
        assert (
            _to_vsi("dods://test.opendap.org/opendap/data/nc/coads.nc")
            == 'NETCDF:"https://test.opendap.org/opendap/data/nc/coads.nc"'
        )

    def test_dods_preserves_query(self):
        assert _to_vsi("dods://h/path?a=b") == 'NETCDF:"https://h/path?a=b"'

    def test_dods_without_double_slash(self):
        # urlparse still classifies "dods:host/path" as scheme dods; must not crash.
        assert _to_vsi("dods:host/path") == 'NETCDF:"https://host/path"'

    def test_dods_uppercase_scheme(self):
        # scheme match is case-insensitive; the slice uses the lower-cased length.
        assert (
            _to_vsi("DODS://test.opendap.org/data.nc")
            == 'NETCDF:"https://test.opendap.org/data.nc"'
        )

    def test_dods_routed_through_parse_path(self):
        # The read path (read_file -> _parse_path -> _to_vsi) must yield the NETCDF: form.
        assert _io._parse_path("dods://h/x.nc") == 'NETCDF:"https://h/x.nc"'


class TestIsRemote:
    @pytest.mark.parametrize(
        "path",
        [
            "s3://bucket/key.tif",
            "gs://bucket/key.tif",
            "az://container/blob.tif",
            "abfs://container/blob.tif",
            "https://foo.com/x.tif",
            "http://foo.com/x.tif",
            "/vsis3/bucket/key.tif",
            "/vsicurl/https://foo/x.tif",
            "/vsimem/x.tif",
            "/vsizip/a.zip/b.tif",
            "dods://test.opendap.org/opendap/data/nc/coads.nc",
        ],
    )
    def test_true_cases(self, path):
        assert is_remote(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/data.tif",
            "C:/data/x.tif",
            "data/x.tif",
            "relative/path.tif",
            "./x.tif",
        ],
    )
    def test_false_cases(self, path):
        assert is_remote(path) is False


class TestIsNetworkBacked:
    """`is_network_backed` narrows `is_remote` to the sources that cross the network."""

    @pytest.mark.parametrize(
        "path, network",
        [
            ("/vsis3/bucket/key.tif", True),
            ("/vsigs/bucket/key.tif", True),
            ("/vsiaz/container/blob.tif", True),
            ("/vsicurl/https://foo/x.tif", True),
            ("/vsicurl?empty_dir=yes&url=https%3A%2F%2Fh%2Fa.tif", True),
            ("/vsicurl_streaming/https://foo/x.tif", True),
            ("/vsiswift/container/x.tif", True),
            ("s3://bucket/key.tif", True),
            ("abfs://container/blob.tif", True),
            ("dods://test.opendap.org/opendap/data/nc/coads.nc", True),
            ("/vsimem/x.tif", False),
            ("/vsizip/a.zip/b.tif", False),
            ("/vsigzip/a.gz", False),
            ("/vsitar/a.tar/b.tif", False),
        ],
    )
    def test_only_the_network_handlers_are_network_backed(self, path, network):
        """Every path here is remote; only the network-fetching ones are credentialed.

        Test scenario:
            The local virtual filesystems (`/vsimem/`, `/vsizip/`, `/vsigzip/`,
            `/vsitar/`) never authenticate, so a caller reasoning about credentials must
            not treat them like a cloud source — expected: `is_remote` is `True` for the
            whole table while `is_network_backed` splits it.
        """
        assert is_remote(path) is True, f"{path} should be remote"
        assert is_network_backed(path) is network, (
            f"is_network_backed({path!r}) should be {network}"
        )


class TestCloudConfigAsGdalConfig:
    def test_empty_default(self):
        assert CloudConfig().as_gdal_config() == {}

    def test_aws_full(self):
        cfg = CloudConfig(
            aws_access_key_id="AK",
            aws_secret_access_key="SEC",
            aws_session_token="TOK",
            aws_region="us-east-1",
        ).as_gdal_config()
        assert cfg == {
            "AWS_ACCESS_KEY_ID": "AK",
            "AWS_SECRET_ACCESS_KEY": "SEC",
            "AWS_SESSION_TOKEN": "TOK",
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
        }

    def test_skips_none_fields(self):
        cfg = CloudConfig(aws_region="eu-west-1").as_gdal_config()
        assert cfg == {"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "eu-west-1"}

    def test_no_sign_request_true(self):
        assert CloudConfig(aws_no_sign_request=True).as_gdal_config() == {
            "AWS_NO_SIGN_REQUEST": "YES"
        }

    def test_no_sign_request_false_absent(self):
        assert (
            "AWS_NO_SIGN_REQUEST"
            not in CloudConfig(aws_no_sign_request=False).as_gdal_config()
        )

    def test_gs_fields(self):
        cfg = CloudConfig(
            gs_access_key_id="GA",
            gs_secret_access_key="GS",
        ).as_gdal_config()
        assert cfg == {"GS_ACCESS_KEY_ID": "GA", "GS_SECRET_ACCESS_KEY": "GS"}

    def test_azure_fields(self):
        cfg = CloudConfig(
            azure_storage_account="acct",
            azure_storage_access_key="key",
            azure_storage_sas_token="sas",
        ).as_gdal_config()
        assert cfg == {
            "AZURE_STORAGE_ACCOUNT": "acct",
            "AZURE_STORAGE_ACCESS_KEY": "key",
            "AZURE_STORAGE_SAS_TOKEN": "sas",
        }

    def test_extra_passthrough(self):
        cfg = CloudConfig(
            extra={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
        ).as_gdal_config()
        assert cfg == {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}

    def test_extra_with_aws(self):
        cfg = CloudConfig(
            aws_region="us-east-1",
            extra={"VSI_CACHE": "TRUE"},
        ).as_gdal_config()
        assert cfg == {
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
            "VSI_CACHE": "TRUE",
        }

    def test_region_drives_both_keys_identically(self):
        """aws_region sets AWS_REGION and AWS_DEFAULT_REGION to the same value.

        Test scenario:
            A single aws_region field must emit both GDAL keys so an inherited
            AWS_DEFAULT_REGION env value cannot override an explicit region.
        """
        cfg = CloudConfig(aws_region="ap-south-1").as_gdal_config()
        assert cfg["AWS_REGION"] == "ap-south-1", f"AWS_REGION wrong: {cfg}"
        assert cfg["AWS_DEFAULT_REGION"] == "ap-south-1", (
            f"AWS_DEFAULT_REGION wrong: {cfg}"
        )

    def test_no_default_region_without_region(self):
        """AWS_DEFAULT_REGION is absent when aws_region is not provided.

        Test scenario:
            None-valued aws_region must drop both region keys (no empty leak).
        """
        cfg = CloudConfig(aws_no_sign_request=True).as_gdal_config()
        assert "AWS_REGION" not in cfg, f"unexpected AWS_REGION: {cfg}"
        assert "AWS_DEFAULT_REGION" not in cfg, f"unexpected AWS_DEFAULT_REGION: {cfg}"

    def test_extra_overrides_default_region(self):
        """An explicit extra AWS_DEFAULT_REGION wins over the aws_region-derived one.

        Test scenario:
            The extra escape hatch can decouple the two keys when a caller needs
            a different default region.
        """
        cfg = CloudConfig(
            aws_region="us-east-1",
            extra={"AWS_DEFAULT_REGION": "us-west-2"},
        ).as_gdal_config()
        assert cfg["AWS_REGION"] == "us-east-1", f"AWS_REGION wrong: {cfg}"
        assert cfg["AWS_DEFAULT_REGION"] == "us-west-2", f"extra did not win: {cfg}"


class TestCloudConfigContextManager:
    def test_enter_exit_no_options(self):
        with CloudConfig():
            pass  # no-op, should not raise

    def test_sets_options_inside_block(self):
        sentinel_key = "AWS_REGION"
        # Ensure it's not set ambiently
        gdal.SetConfigOption(sentinel_key, None)

        with CloudConfig(aws_region="us-east-2"):
            assert gdal.GetConfigOption(sentinel_key) == "us-east-2"

    def test_restores_previous_value_on_exit(self):
        key = "AWS_REGION"
        gdal.SetConfigOption(key, "us-west-2")
        try:
            with CloudConfig(aws_region="us-east-1"):
                assert gdal.GetConfigOption(key) == "us-east-1"
            assert gdal.GetConfigOption(key) == "us-west-2"
        finally:
            gdal.SetConfigOption(key, None)

    def test_restores_on_exception(self):
        key = "AWS_REGION"
        gdal.SetConfigOption(key, "before")
        cfg = CloudConfig(aws_region="during")
        # The in-context "during" value is covered by
        # test_restores_previous_value_on_exit; this test only asserts restoration
        # when the body raises, so the pytest.raises block holds a single raiser.
        try:
            with pytest.raises(RuntimeError):
                with cfg:
                    raise RuntimeError("boom")
            assert gdal.GetConfigOption(key) == "before"
        finally:
            gdal.SetConfigOption(key, None)

    def test_no_sign_request_applied(self):
        gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", None)
        with CloudConfig(aws_no_sign_request=True):
            assert gdal.GetConfigOption("AWS_NO_SIGN_REQUEST") == "YES"
        assert gdal.GetConfigOption("AWS_NO_SIGN_REQUEST") is None

    def test_extra_applied(self):
        gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", None)
        with CloudConfig(extra={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}):
            assert gdal.GetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN") == "EMPTY_DIR"
        assert gdal.GetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN") is None


# ---------------------------------------------------------------------------
# End-to-end cloud I/O via a local HTTP server (Task 12)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHttpCogRead:
    """Read a COG served over HTTP through the /vsicurl/ rewrite."""

    def test_read_cog_over_http(self, http_server):
        from pyramids.dataset import Dataset

        url = f"{http_server}/valid.tif"
        ds = Dataset.read_file(url)
        assert ds.rows == 256
        assert ds.columns == 256
        assert ds.epsg == 4326

    def test_read_cog_over_http_is_cog_true(self, http_server):
        from pyramids.dataset import Dataset

        url = f"{http_server}/valid.tif"
        ds = Dataset.read_file(url)
        # is_cog works over VSI paths too
        assert ds.is_cog is True

    def test_read_cog_over_http_array_matches(self, http_server):
        from pyramids.dataset import Dataset

        url = f"{http_server}/valid.tif"
        ds = Dataset.read_file(url)
        arr = ds.read_array()
        expected = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        assert np.array_equal(arr, expected)

    def test_read_plain_gtiff_over_http_may_require_range(self, http_server):
        """Plain (stripped) GTiff needs byte-range requests.

        Python's stdlib `http.server` does not support HTTP Range,
        so GDAL will fail for files that cannot be read sequentially.
        We only assert that *if* it raises, the URL was rewritten to
        /vsicurl/ first — i.e., the pipeline is correct even if the
        fixture HTTP server can't serve it.
        """
        from pyramids.dataset import Dataset

        url = f"{http_server}/plain.tif"
        try:
            ds = Dataset.read_file(url)
            assert ds.rows == 256
        except RuntimeError as exc:
            # Expected: stdlib HTTP server lacks Range support.
            assert "Range" in str(exc) or "range" in str(exc)


class TestS3UrlRewriteNoNetwork:
    """Verify string-level rewriting for s3:// without hitting the network."""

    def test_s3_rewrite_attempt_reaches_gdal(self):
        from pyramids.dataset import Dataset

        # We only care that pyramids doesn't raise ValueError about
        # unknown schemes, and that the rewrite reached /vsis3/.
        # GDAL raises RuntimeError or OSError for missing S3 resources.
        with pytest.raises((RuntimeError, OSError)):
            Dataset.read_file("s3://nonexistent-bucket-xyz-1234/nope.tif")

    def test_pipeline_uses_vsis3_prefix(self):
        from pyramids._io import _parse_path

        assert _parse_path("s3://b/k.tif") == "/vsis3/b/k.tif"


class TestToVsiArchiveChaining:
    """Tests for `_to_vsi`'s archive-chaining behavior.

    Named after the public entry point (`_to_vsi`) rather than the
    internal helper (`_chain_archive_vsi`) so the test class name
    remains stable if the helper is later renamed or inlined.

    Covers the pre-existing gap where
    `https://host/archive.tar/inner.tif` -> `/vsicurl/...` lost
    access to the inner file because GDAL needs `/vsitar//vsicurl/...`.
    """

    def test_tar_inside_https(self):
        """HTTPS URL pointing into .tar archive gets /vsitar/ prefix."""
        url = "https://example.com/archive.tar/inner.tif"
        result = _to_vsi(url)
        assert result == "/vsitar//vsicurl/https://example.com/archive.tar/inner.tif", (
            f"Expected chained /vsitar/ + /vsicurl/, got: {result}"
        )

    def test_zip_inside_s3(self):
        """S3 URL pointing into .zip archive gets /vsizip/ prefix."""
        url = "s3://bucket/archive.zip/inner.tif"
        result = _to_vsi(url)
        assert result == "/vsizip//vsis3/bucket/archive.zip/inner.tif", (
            f"Expected chained /vsizip/ + /vsis3/, got: {result}"
        )

    def test_gz_inside_https(self):
        """HTTPS URL pointing into .gz file gets /vsigzip/ prefix."""
        url = "https://example.com/data.gz/inner.asc"
        result = _to_vsi(url)
        assert result == "/vsigzip//vsicurl/https://example.com/data.gz/inner.asc", (
            f"Expected chained /vsigzip/ + /vsicurl/, got: {result}"
        )

    def test_tar_gz_inside_https(self):
        """HTTPS URL pointing into .tar.gz archive routes via /vsitar/."""
        url = "https://example.com/archive.tar.gz/inner.tif"
        result = _to_vsi(url)
        assert "/vsitar/" in result, (
            f".tar.gz must route through /vsitar/, got: {result}"
        )

    def test_tgz_inside_gs(self):
        """GCS URL pointing into .tgz archive routes via /vsitar/."""
        url = "gs://bucket/data.tgz/inner.tif"
        result = _to_vsi(url)
        assert result.startswith("/vsitar//vsigs/"), (
            f".tgz must chain /vsitar/ + /vsigs/, got: {result}"
        )

    def test_plain_tif_over_https_no_chain(self):
        """URL without an archive segment is not chained."""
        url = "https://example.com/scene.tif"
        result = _to_vsi(url)
        assert result == "/vsicurl/https://example.com/scene.tif", (
            f"Non-archive URL must not chain; got: {result}"
        )

    def test_archive_named_in_url_but_not_traversed(self):
        """URL ending at archive name (no trailing /) is not chained.

        Test scenario:
            If the user points at the archive file itself rather than
            a member inside it, GDAL can download and inspect the
            archive — no chained VSI needed.
        """
        url = "https://example.com/archive.tar"
        result = _to_vsi(url)
        assert result == "/vsicurl/https://example.com/archive.tar", (
            f"Archive-name-only URL must not chain; got: {result}"
        )

    def test_local_zip_path_unchanged_by_chain(self):
        """Local .zip/foo.tif is left for pyramids._io._parse_path to handle."""
        p = "/local/path/archive.zip/inner.tif"
        result = _to_vsi(p)
        assert result == p, (
            f"Local archive paths must be left to _parse_path, got: {result}"
        )


class TestCloudConfigCtxAttribute:
    """CloudConfig._ctx is a typed field and is cleared on exit."""

    def test_ctx_is_none_before_enter(self):
        """Fresh CloudConfig has _ctx as None, not an undefined attribute."""
        cfg = CloudConfig(aws_region="us-east-1")
        assert cfg._ctx is None, (
            f"Fresh CloudConfig must have _ctx is None, got: {cfg._ctx!r}"
        )

    def test_ctx_is_cleared_after_exit(self):
        """After __exit__, _ctx returns to None (no lingering reference)."""
        cfg = CloudConfig(aws_region="us-east-1")
        with cfg:
            assert cfg._ctx is not None, "_ctx must be set inside the with block"
        assert cfg._ctx is None, f"_ctx must be cleared after exit, got: {cfg._ctx!r}"

    def test_ctx_not_in_repr(self):
        """_ctx is declared repr=False so it does not leak into repr()."""
        cfg = CloudConfig(aws_region="us-east-1")
        assert "_ctx" not in repr(cfg), (
            f"_ctx must be excluded from repr; got: {repr(cfg)}"
        )

    def test_ctx_not_in_equality_comparison(self):
        """_ctx is compare=False so __eq__ still works across with-blocks."""
        a = CloudConfig(aws_region="us-east-1")
        b = CloudConfig(aws_region="us-east-1")
        with a:
            assert a == b, (
                f"CloudConfigs with equal public fields must compare equal "
                f"regardless of _ctx state; got a={a!r}, b={b!r}"
            )


class TestToVsiArchiveChainingEdgeCases:
    """M1 (2nd review): boundary-anchored archive marker detection.

    These scenarios used to FALSE-POSITIVE under the substring-only
    implementation: query-string injection, hostname ending in an
    archive extension, and nested archives. The boundary-anchored
    regex + path-component extraction in `_extract_archive_search_region`
    must correctly handle them.
    """

    def test_query_string_with_dot_tar_not_chained(self):
        """Presigned URL containing `.tar/` in the query must not chain.

        Test scenario:
            A presigned URL may embed `archive.tar` in the query
            value (e.g. as part of a signed key). The file itself is
            a plain GeoTIFF; prepending /vsitar/ would break the read.
        """
        url = "https://foo.com/x.tif?key=archive.tar/inner&sig=abc"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f"Query-string .tar/ must not trigger archive chaining; got: {result}"
        )

    def test_query_string_with_dot_zip_not_chained(self):
        """Same protection for .zip inside a query string."""
        url = "https://foo.com/scene.tif?asset=pkg.zip/inner.tif"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f"Query-string .zip/ must not trigger archive chaining; got: {result}"
        )

    def test_query_string_with_dot_gz_not_chained(self):
        """Same protection for .gz inside a query string."""
        url = "https://foo.com/scene.tif?backup=data.gz/inner"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f"Query-string .gz/ must not trigger archive chaining; got: {result}"
        )

    def test_hostname_ending_in_gz_not_chained(self):
        """Hostname whose label ends in .gz must not trigger archive chaining.

        Test scenario:
            A hostname like `data.gz.example.com` or `weird.gz` is
            legitimate and unrelated to gzip archives. The URL's path
            component is the only authoritative source for archive
            markers.
        """
        url = "https://weird.gz/file.tif"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f"Hostname ending in .gz must not trigger archive chaining; got: {result}"
        )

    def test_hostname_with_dot_tar_not_chained(self):
        """Hostname containing .tar (e.g. tar.example.com) is not an archive."""
        url = "https://tar.example.com/file.tif"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f"Hostname containing .tar must not trigger archive chaining; got: {result}"
        )

    def test_nested_outer_archive_wins(self):
        """Nested archive path only applies the outermost archive prefix.

        Test scenario:
            `outer.zip/inner.tar/file.tif` — only the OUTER.zip
            marker is honored; GDAL's chained-VSI syntax doesn't
            compose through arbitrary nesting, so silently applying
            /vsitar//vsizip/... would produce un-openable paths in
            most real cases. Documented single-layer limitation.
        """
        url = "https://foo.com/outer.zip/inner.tar/file.tif"
        result = _to_vsi(url)
        expected = "/vsizip//vsicurl/https://foo.com/outer.zip/inner.tar/file.tif"
        assert result == expected, (
            f"Nested archive: only outermost (.zip) should chain; got: {result}"
        )

    def test_path_with_dot_tar_in_directory_name_chained(self):
        """A legitimate `archive.tar/` segment in the path IS chained.

        Test scenario:
            Regression guard that the boundary-anchored regex still
            catches the common case: a path segment named
            `something.tar/` points INTO an archive and must be
            chained.
        """
        url = "https://foo.com/path/archive.tar/inner.tif"
        result = _to_vsi(url)
        assert result.startswith("/vsitar//vsicurl/"), (
            f"Legitimate archive segment must chain; got: {result}"
        )

    def test_s3_key_with_dot_zip_segment_chained(self):
        """S3 key that traverses a .zip segment is correctly chained."""
        url = "s3://bucket/folder/archive.zip/inner.tif"
        result = _to_vsi(url)
        assert result == "/vsizip//vsis3/bucket/folder/archive.zip/inner.tif", (
            f"S3 archive segment must chain; got: {result}"
        )

    def test_tar_gz_prefers_vsitar_over_vsigzip(self):
        """.tar.gz/ must match before .gz/ in the regex alternation."""
        url = "https://foo.com/archive.tar.gz/inner.tif"
        result = _to_vsi(url)
        assert result.startswith("/vsitar//vsicurl/"), (
            f".tar.gz/ must route through /vsitar/, got: {result}"
        )
        assert not result.startswith("/vsigzip/"), (
            f".tar.gz/ must NOT route through /vsigzip/, got: {result}"
        )

    def test_uppercase_archive_extension_matched(self):
        """Case-insensitive matching — `.ZIP/` is treated like `.zip/`."""
        url = "https://foo.com/ARCHIVE.ZIP/INNER.TIF"
        result = _to_vsi(url)
        assert result.startswith("/vsizip//vsicurl/"), (
            f"Uppercase .ZIP/ must still chain; got: {result}"
        )

    def test_non_archive_extension_not_chained(self):
        """Non-archive extensions at path boundaries are not chained."""
        url = "https://foo.com/container.tif/inner.tif"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f".tif/ is not an archive; must not chain; got: {result}"
        )


@pytest.mark.slow
@pytest.mark.live
class TestLiveOpenDAP:
    """Read a real OPeNDAP/THREDDS dataset over dods:// via GDAL's netCDF DAP support."""

    URL = os.environ.get(
        "PYRAMIDS_OPENDAP_URL",
        "dods://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/surface/air.sig995.2012.nc",
    )
    VAR = os.environ.get("PYRAMIDS_OPENDAP_VAR", "air")

    def test_read_opendap_schema(self):
        """A dods:// URL opens as a NetCDF and its variable schema is read without a full download."""
        nc = NetCDF.read_file(self.URL)
        assert self.VAR in list(nc.variables), (
            f"{self.VAR!r} not in {list(nc.variables)[:8]}"
        )
        variable = nc.get_variable(self.VAR)
        assert variable.shape[0] >= 1


class TestRedactCredentials:
    """Credential values must not survive into a log line or message."""

    def test_embedded_header_value_is_blanked(self):
        """An embedded bearer header is replaced, the rest stays readable."""
        text = "Can't open /vsicurl?header.Authorization=Bearer%20SECRET&url=x"
        out = redact_credentials(text)
        assert "SECRET" not in out, f"token survived: {out}"
        assert "header.Authorization=<redacted>" in out, out

    def test_sas_query_parameter_is_blanked(self):
        """A SAS-style `sig=` parameter is caught as well."""
        out = redact_credentials("https://h/a.tif?sv=2021&sig=SECRET")
        assert out == "https://h/a.tif?sv=2021&sig=<redacted>", out

    def test_ordinary_message_is_untouched(self):
        """A message with no credential is returned verbatim."""
        text = "Can't open /vsicurl/https://h/a.tif. Skipping it"
        assert redact_credentials(text) == text, "an innocent message was altered"

    def test_multiple_headers_all_blanked(self):
        """Every credential option in one string is redacted."""
        text = "/vsicurl?header.Authorization=Bearer%20A&header.X-Api-Key=B&url=x"
        out = redact_credentials(text)
        assert "Bearer%20A" not in out and "=B&" not in out, f"a value survived: {out}"

    def test_name_merely_ending_in_a_credential_word_is_kept(self):
        """A name that only ends with a credential word is not one.

        The previous `\b`-anchored pattern matched the tail of `my_token=` and
        `bucket-key=`, redacting values that were never secrets.
        """
        text = "https://h/a.tif?my_token=abc&bucket-key=def"
        assert redact_credentials(text) == text, (
            f"a non-credential was redacted: {text}"
        )

    def test_credential_after_an_ampersand_is_redacted(self):
        """A credential option later in the query is still caught."""
        out = redact_credentials("https://h/a.tif?sv=2021&sig=SECRET&sp=r")
        assert "SECRET" not in out, f"token survived: {out}"
        assert "sv=2021" in out and "sp=r" in out, f"neighbours mangled: {out}"

    def test_is_remote_accepts_the_query_form(self):
        """The `/vsicurl?` form is classified as remote like `/vsicurl/` is."""
        assert is_remote("/vsicurl?empty_dir=yes&url=https%3A%2F%2Fh%2Fa.tif"), (
            "the query form must be recognised as remote"
        )


class TestHandlerTableInvariant:
    """Tests for the relationship between the four handler tables (#918).

    The tables answer different questions and are not copies of one list, so
    the thing worth pinning is how they relate: `cloud` subset of `network`
    subset of `vsi`, and no purely local handler in the network set.
    """

    def test_network_is_a_subset_of_vsi(self):
        """Every network handler is recognised as a VSI path at all."""
        missing = set(_NETWORK_VSI_PREFIXES) - set(_VSI_PREFIXES)
        assert not missing, f"network handlers absent from _VSI_PREFIXES: {missing}"

    def test_cloud_is_a_subset_of_network(self):
        """Archive chaining is only offered for handlers that touch the network."""
        missing = set(_CLOUD_VSI_PREFIXES) - set(_NETWORK_VSI_PREFIXES)
        assert not missing, f"chaining handlers that are not network-backed: {missing}"

    @pytest.mark.parametrize(
        "prefix", ["/vsimem/", "/vsizip/", "/vsitar/", "/vsigzip/"]
    )
    def test_local_handlers_are_not_network_backed(self, prefix: str):
        """An in-memory or archive handler reads no network.

        Args:
            prefix: A purely local VSI prefix.
        """
        assert prefix in _VSI_PREFIXES, f"{prefix} should be a known VSI prefix"
        assert prefix not in _NETWORK_VSI_PREFIXES, f"{prefix} is not network-backed"

    def test_every_url_scheme_is_actually_rewritten(self):
        """No scheme in the map may fall through `_to_vsi` unchanged.

        The real risk the derivation guards against: a scheme added to
        `URL_SCHEMES` whose prefix no branch of `_to_vsi` handles would be
        returned as-is and handed to GDAL raw. Asserting on the rewrite is what
        catches that; asserting that the derived set matches its own derivation
        cannot fail.
        """
        for scheme in URL_SCHEMES:
            rewritten = _to_vsi(f"{scheme}://container/key.tif")
            assert rewritten != f"{scheme}://container/key.tif", (
                f"{scheme}:// is in URL_SCHEMES but _to_vsi leaves it unchanged"
            )

    @pytest.mark.parametrize(
        "prefix",
        [
            "/vsis3_streaming/",
            "/vsigs_streaming/",
            "/vsiaz_streaming/",
            "/vsioss_streaming/",
            "/vsiswift_streaming/",
        ],
    )
    def test_streaming_handlers_are_network_backed(self, prefix: str):
        """GDAL's `_streaming` twins read the same services over the network.

        Args:
            prefix: A streaming VSI prefix.

        Test scenario:
            Only `/vsicurl_streaming/` was listed, so the five object-store
            streaming handlers classified as local files.
        """
        path = f"{prefix}bucket/key.tif"
        assert is_remote(path), f"{prefix} must be remote"
        assert is_network_backed(path), f"{prefix} must be network-backed"

    def test_streaming_handlers_do_not_chain_archives(self):
        """Their remainder is an option list, so an archive marker is untrustworthy."""
        path = "/vsis3_streaming/b/a.zip/x.tif"
        assert _to_vsi(path) == path, "streaming handlers must not chain"


class TestAdlsHandler:
    """Tests for the ADLS Gen2 handler recognised in #918."""

    ADLS_PATH = "/vsiadls/container/x.tif"

    def test_is_remote(self):
        """A Gen2 path is remote, where it used to read as a local file."""
        assert is_remote(self.ADLS_PATH), "an ADLS path must be recognised as remote"

    def test_is_network_backed(self):
        """It is network-backed too, so credential reasoning applies."""
        assert is_network_backed(self.ADLS_PATH), "an ADLS path is network-backed"

    def test_chains_an_archive(self):
        """A zipped raster on Gen2 is rewritten the way the S3 equivalent is."""
        chained = _to_vsi("/vsiadls/c/a.zip/x.tif")
        assert chained == "/vsizip//vsiadls/c/a.zip/x.tif", chained


class TestNetworkHandlersChainArchives:
    """Tests that every network handler may chain an archive (#918).

    `_CLOUD_VSI_PREFIXES` used to be s3/gs/az/curl only, so a zipped raster on
    Alibaba OSS, OpenStack Swift or HDFS was never rewritten.

    These assert the path *rewrite* only. `/vsihdfs/` and `/vsiwebhdfs/` need a
    GDAL built with HDFS support and are not registered in every build (the
    project's own is one that lacks them), so nothing here opens a dataset.
    """

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/vsioss/c/a.zip/x.tif", "/vsizip//vsioss/c/a.zip/x.tif"),
            ("/vsiswift/c/a.zip/x.tif", "/vsizip//vsiswift/c/a.zip/x.tif"),
            (
                "/vsihdfs/hdfs://nn:8020/d/a.zip/x.tif",
                "/vsizip//vsihdfs/hdfs://nn:8020/d/a.zip/x.tif",
            ),
            (
                "/vsiwebhdfs/http://h:50070/webhdfs/v1/a.tar/x.tif",
                "/vsitar//vsiwebhdfs/http://h:50070/webhdfs/v1/a.tar/x.tif",
            ),
        ],
    )
    def test_chains(self, path: str, expected: str):
        """Each network handler gets the archive prefix it needs.

        Args:
            path: A VSI path pointing inside an archive.
            expected: The chained form GDAL needs to read the inner file.
        """
        assert _to_vsi(path) == expected, (
            "chaining must be reachable through the public rewrite"
        )

    @pytest.mark.parametrize(
        "path",
        [
            "/vsihdfs/hdfs://data.gz/x.tif",
            "/vsiwebhdfs/http://data.zip/webhdfs/v1/x.tif",
        ],
    )
    def test_a_hostname_that_looks_like_an_archive_does_not_chain(self, path: str):
        """The URL-embedding handlers strip the hostname before searching.

        Args:
            path: A path whose *hostname* ends in an archive extension.

        Test scenario:
            `/vsihdfs/` and `/vsiwebhdfs/` embed a full URL after the prefix, so
            scanning the raw remainder would let a host named `data.gz` trigger
            chaining — the same trap `/vsicurl/` already guarded against.
        """
        assert _to_vsi(path) == path, "a hostname must not trigger chaining"
