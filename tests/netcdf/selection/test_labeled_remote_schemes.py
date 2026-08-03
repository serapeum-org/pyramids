"""Tests that LabeledDataset recognises the same remote URL schemes as Dataset (#918).

``labeled.py`` kept its own scheme tuple, so a handler added to ``URL_SCHEMES`` was
remote for ``Dataset`` and local here — an ADLS Gen2 store would have been opened as
if it were a path on disk. The list is now derived from the one source.
"""

from __future__ import annotations

import pytest

from pyramids.base.remote import URL_SCHEMES
from pyramids.netcdf.labeled import _is_remote_url

pytestmark = pytest.mark.core


class TestRemoteSchemeRecognition:
    """Tests for `_is_remote_url` against the shared scheme table."""

    @pytest.mark.parametrize(
        "scheme", sorted(set(URL_SCHEMES) - {"file"})
    )
    def test_every_url_scheme_is_remote(self, scheme: str):
        """A scheme the rewriter understands is remote to the labeled reader too.

        Args:
            scheme: A scheme from the shared `URL_SCHEMES` table.
        """
        assert _is_remote_url(f"{scheme}://container/store.zarr"), (
            f"{scheme}:// is rewritten to a /vsi* path, so it must read as remote"
        )

    @pytest.mark.parametrize("scheme", ["adls", "abfss", "abfs"])
    def test_gen2_schemes_are_remote(self, scheme: str):
        """The Gen2 schemes specifically, which #918 added.

        Args:
            scheme: An ADLS Gen2 URL scheme.
        """
        assert _is_remote_url(f"{scheme}://container/store.zarr"), (
            f"{scheme}:// names ADLS Gen2 and must not read as a local path"
        )

    @pytest.mark.parametrize(
        "source", ["/tmp/store.zarr", "C:/data/store.zarr", "file:///tmp/store.zarr"]
    )
    def test_local_paths_are_not_remote(self, source: str):
        """A local path — including `file://` — is not remote.

        Args:
            source: A local filesystem path or URI.
        """
        assert not _is_remote_url(source), f"{source} is local"

    def test_the_gcs_alias_is_kept(self):
        """`gcs://` is not in `URL_SCHEMES` but GDAL accepts it, so it stays."""
        assert _is_remote_url("gcs://bucket/store.zarr"), "gcs:// must stay recognised"
