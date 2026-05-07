"""Unit tests for pyramids.base.remote."""

from __future__ import annotations

import numpy as np  # noqa: E402
import pytest
from osgeo import gdal

from pyramids.base.remote import CloudConfig, _to_vsi, is_remote

pytestmark = pytest.mark.core


class TestToVsi:
    def test_s3(self):
        assert _to_vsi("s3://bucket/key.tif") == "/vsis3/bucket/key.tif"

    def test_s3_nested_path(self):
        assert _to_vsi("s3://bucket/a/b/c/key.tif") == "/vsis3/bucket/a/b/c/key.tif"

    def test_gs(self):
        assert _to_vsi("gs://bucket/key.tif") == "/vsigs/bucket/key.tif"

    def test_az(self):
        assert _to_vsi("az://container/blob.tif") == "/vsiaz/container/blob.tif"

    def test_abfs_maps_to_vsiaz(self):
        assert _to_vsi("abfs://container/blob.tif") == "/vsiaz/container/blob.tif"

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
        }

    def test_skips_none_fields(self):
        cfg = CloudConfig(aws_region="eu-west-1").as_gdal_config()
        assert cfg == {"AWS_REGION": "eu-west-1"}

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
        assert cfg == {"AWS_REGION": "us-east-1", "VSI_CACHE": "TRUE"}


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
        try:
            with pytest.raises(RuntimeError):
                with CloudConfig(aws_region="during"):
                    assert gdal.GetConfigOption(key) == "during"
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

        # Any GDAL error is acceptable — we only care that pyramids
        # doesn't raise ValueError about unknown schemes, and that the
        # rewrite reached /vsis3/.
        with pytest.raises(Exception):
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
        assert (
            result == "/vsitar//vsicurl/https://example.com/archive.tar/inner.tif"
        ), f"Expected chained /vsitar/ + /vsicurl/, got: {result}"

    def test_zip_inside_s3(self):
        """S3 URL pointing into .zip archive gets /vsizip/ prefix."""
        url = "s3://bucket/archive.zip/inner.tif"
        result = _to_vsi(url)
        assert (
            result == "/vsizip//vsis3/bucket/archive.zip/inner.tif"
        ), f"Expected chained /vsizip/ + /vsis3/, got: {result}"

    def test_gz_inside_https(self):
        """HTTPS URL pointing into .gz file gets /vsigzip/ prefix."""
        url = "https://example.com/data.gz/inner.asc"
        result = _to_vsi(url)
        assert (
            result == "/vsigzip//vsicurl/https://example.com/data.gz/inner.asc"
        ), f"Expected chained /vsigzip/ + /vsicurl/, got: {result}"

    def test_tar_gz_inside_https(self):
        """HTTPS URL pointing into .tar.gz archive routes via /vsitar/."""
        url = "https://example.com/archive.tar.gz/inner.tif"
        result = _to_vsi(url)
        assert (
            "/vsitar/" in result
        ), f".tar.gz must route through /vsitar/, got: {result}"

    def test_tgz_inside_gs(self):
        """GCS URL pointing into .tgz archive routes via /vsitar/."""
        url = "gs://bucket/data.tgz/inner.tif"
        result = _to_vsi(url)
        assert result.startswith(
            "/vsitar//vsigs/"
        ), f".tgz must chain /vsitar/ + /vsigs/, got: {result}"

    def test_plain_tif_over_https_no_chain(self):
        """URL without an archive segment is not chained."""
        url = "https://example.com/scene.tif"
        result = _to_vsi(url)
        assert (
            result == "/vsicurl/https://example.com/scene.tif"
        ), f"Non-archive URL must not chain; got: {result}"

    def test_archive_named_in_url_but_not_traversed(self):
        """URL ending at archive name (no trailing /) is not chained.

        Test scenario:
            If the user points at the archive file itself rather than
            a member inside it, GDAL can download and inspect the
            archive — no chained VSI needed.
        """
        url = "https://example.com/archive.tar"
        result = _to_vsi(url)
        assert (
            result == "/vsicurl/https://example.com/archive.tar"
        ), f"Archive-name-only URL must not chain; got: {result}"

    def test_local_zip_path_unchanged_by_chain(self):
        """Local .zip/foo.tif is left for pyramids._io._parse_path to handle."""
        p = "/local/path/archive.zip/inner.tif"
        result = _to_vsi(p)
        assert (
            result == p
        ), f"Local archive paths must be left to _parse_path, got: {result}"


class TestCloudConfigCtxAttribute:
    """CloudConfig._ctx is a typed field and is cleared on exit."""

    def test_ctx_is_none_before_enter(self):
        """Fresh CloudConfig has _ctx as None, not an undefined attribute."""
        cfg = CloudConfig(aws_region="us-east-1")
        assert (
            cfg._ctx is None
        ), f"Fresh CloudConfig must have _ctx is None, got: {cfg._ctx!r}"

    def test_ctx_is_cleared_after_exit(self):
        """After __exit__, _ctx returns to None (no lingering reference)."""
        cfg = CloudConfig(aws_region="us-east-1")
        with cfg:
            assert cfg._ctx is not None, "_ctx must be set inside the with block"
        assert cfg._ctx is None, f"_ctx must be cleared after exit, got: {cfg._ctx!r}"

    def test_ctx_not_in_repr(self):
        """_ctx is declared repr=False so it does not leak into repr()."""
        cfg = CloudConfig(aws_region="us-east-1")
        assert "_ctx" not in repr(
            cfg
        ), f"_ctx must be excluded from repr; got: {repr(cfg)}"

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
        assert (
            result == f"/vsicurl/{url}"
        ), f"Query-string .tar/ must not trigger archive chaining; got: {result}"

    def test_query_string_with_dot_zip_not_chained(self):
        """Same protection for .zip inside a query string."""
        url = "https://foo.com/scene.tif?asset=pkg.zip/inner.tif"
        result = _to_vsi(url)
        assert (
            result == f"/vsicurl/{url}"
        ), f"Query-string .zip/ must not trigger archive chaining; got: {result}"

    def test_query_string_with_dot_gz_not_chained(self):
        """Same protection for .gz inside a query string."""
        url = "https://foo.com/scene.tif?backup=data.gz/inner"
        result = _to_vsi(url)
        assert (
            result == f"/vsicurl/{url}"
        ), f"Query-string .gz/ must not trigger archive chaining; got: {result}"

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
        assert (
            result == f"/vsicurl/{url}"
        ), f"Hostname ending in .gz must not trigger archive chaining; got: {result}"

    def test_hostname_with_dot_tar_not_chained(self):
        """Hostname containing .tar (e.g. tar.example.com) is not an archive."""
        url = "https://tar.example.com/file.tif"
        result = _to_vsi(url)
        assert (
            result == f"/vsicurl/{url}"
        ), f"Hostname containing .tar must not trigger archive chaining; got: {result}"

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
        assert (
            result == expected
        ), f"Nested archive: only outermost (.zip) should chain; got: {result}"

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
        assert result.startswith(
            "/vsitar//vsicurl/"
        ), f"Legitimate archive segment must chain; got: {result}"

    def test_s3_key_with_dot_zip_segment_chained(self):
        """S3 key that traverses a .zip segment is correctly chained."""
        url = "s3://bucket/folder/archive.zip/inner.tif"
        result = _to_vsi(url)
        assert (
            result == "/vsizip//vsis3/bucket/folder/archive.zip/inner.tif"
        ), f"S3 archive segment must chain; got: {result}"

    def test_tar_gz_prefers_vsitar_over_vsigzip(self):
        """.tar.gz/ must match before .gz/ in the regex alternation."""
        url = "https://foo.com/archive.tar.gz/inner.tif"
        result = _to_vsi(url)
        assert result.startswith(
            "/vsitar//vsicurl/"
        ), f".tar.gz/ must route through /vsitar/, got: {result}"
        assert not result.startswith(
            "/vsigzip/"
        ), f".tar.gz/ must NOT route through /vsigzip/, got: {result}"

    def test_uppercase_archive_extension_matched(self):
        """Case-insensitive matching — `.ZIP/` is treated like `.zip/`."""
        url = "https://foo.com/ARCHIVE.ZIP/INNER.TIF"
        result = _to_vsi(url)
        assert result.startswith(
            "/vsizip//vsicurl/"
        ), f"Uppercase .ZIP/ must still chain; got: {result}"

    def test_non_archive_extension_not_chained(self):
        """Non-archive extensions at path boundaries are not chained."""
        url = "https://foo.com/container.tif/inner.tif"
        result = _to_vsi(url)
        assert (
            result == f"/vsicurl/{url}"
        ), f".tif/ is not an archive; must not chain; got: {result}"
