"""Tests that LabeledDataset rewrites the same URL schemes as Dataset (#918).

``labeled.py`` kept its own scheme tuple, so a handler added to ``URL_SCHEMES`` was
understood by ``Dataset`` and not here — an ADLS Gen2 store would have been opened as
if it were a path on disk. The list is now derived from the rewriter's own table.
"""

from __future__ import annotations

import pytest

from pyramids.base.remote import _DODS_SCHEME, URL_SCHEMES, _to_vsi
from pyramids.netcdf.labeled import _is_remote_url

pytestmark = pytest.mark.core


class TestSchemeRewriteRecognition:
    """Tests for `_is_remote_url` against the shared scheme table."""

    @pytest.mark.parametrize("scheme", sorted(URL_SCHEMES))
    def test_every_url_scheme_needs_rewriting(self, scheme: str):
        """A scheme the rewriter understands must reach the rewriter here too.

        Args:
            scheme: A scheme from the shared `URL_SCHEMES` table.
        """
        assert _is_remote_url(f"{scheme}://container/store.zarr"), (
            f"{scheme}:// is rewritten by _to_vsi, so it must not be passed to GDAL raw"
        )

    @pytest.mark.parametrize("scheme", ["abfs", "abfss"])
    def test_gen2_schemes_are_recognised(self, scheme: str):
        """The Gen2 schemes specifically, which #918 added.

        Args:
            scheme: An ADLS Gen2 URL scheme.
        """
        assert _is_remote_url(f"{scheme}://container/store.zarr"), (
            f"{scheme}:// names ADLS Gen2 and must not be read as a local path"
        )

    def test_dods_is_recognised(self):
        """`dods://` lives outside `URL_SCHEMES` but still needs rewriting.

        Test scenario:
            It rewrites to a `NETCDF:` connection string rather than a `/vsi*`
            prefix, so deriving from `URL_SCHEMES` alone would miss it and hand
            GDAL a raw `dods://` URI.
        """
        assert _is_remote_url("dods://test.opendap.org/data.nc"), (
            "dods:// must be rewritten to a NETCDF: connection string"
        )
        assert _DODS_SCHEME == "dods", "the constant this relies on"

    def test_file_uris_need_rewriting_even_though_they_are_local(self):
        """`file://` names a local path, but GDAL cannot open the URI form.

        Test scenario:
            "Is it remote?" and "does it need rewriting?" are different
            questions. Treating `file://` as local skipped the rewrite and
            handed GDAL a URI it rejects.
        """
        assert _is_remote_url("file:///data/store.zarr"), (
            "file:// must be stripped to a plain path before GDAL sees it"
        )

    @pytest.mark.parametrize("source", ["/tmp/store.zarr", "C:/data/store.zarr"])
    def test_plain_paths_need_no_rewriting(self, source: str):
        """A path with no scheme is passed through untouched.

        Args:
            source: A local filesystem path.
        """
        assert not _is_remote_url(source), f"{source} needs no rewrite"

    def test_the_gcs_alias_is_functional(self):
        """`gcs://` is claimed here because it is now actually rewritten.

        Test scenario:
            It used to be claimed on the false premise that GDAL accepts it, so
            the raw URI reached GDAL and failed. It is a real, fsspec-registered
            alias for Google Cloud Storage, so the fix was to map it to
            `/vsigs/` rather than to stop claiming it.
        """
        assert _is_remote_url("gcs://bucket/store.zarr"), "gcs:// is handled"
        assert _to_vsi("gcs://bucket/store.zarr") == "/vsigs/bucket/store.zarr"

    @pytest.mark.parametrize("scheme", ["ftp", "sftp", "mailto"])
    def test_a_genuinely_unhandled_scheme_is_not_claimed(self, scheme: str):
        """A scheme no handler covers is left for the caller to fail on.

        Args:
            scheme: A URL scheme pyramids does not rewrite.
        """
        assert not _is_remote_url(f"{scheme}://host/store.zarr"), (
            f"{scheme}:// is not rewritten, so it must not be claimed"
        )
