"""Unit tests for pyramids.base._artifacts (shared scratch artefact root, M1)."""

from __future__ import annotations

import os

import pytest

from osgeo import gdal

from pyramids.base import _artifacts
from pyramids.base._artifacts import artifact_dir, cleanup, register_vsimem

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def _reset_artifacts():
    """Clean any artefact state before and after each test for isolation."""
    cleanup()
    yield
    cleanup()


class TestArtifactDir:
    """Tests for artifact_dir."""

    def test_returns_existing_unique_dirs(self):
        """Each call returns a new, existing directory.

        Test scenario:
            Two calls yield distinct existing directories.
        """
        a, b = artifact_dir(), artifact_dir()
        assert os.path.isdir(a) and os.path.isdir(b), "both dirs should exist"
        assert a != b, f"dirs should be unique, got {a} == {b}"

    def test_dirs_share_one_root(self):
        """All artefact dirs live under a single shared root.

        Test scenario:
            Two calls share the same parent directory (bounds proliferation).
        """
        a, b = artifact_dir(), artifact_dir()
        assert os.path.dirname(a) == os.path.dirname(b), (
            f"dirs should share one root: {os.path.dirname(a)} vs {os.path.dirname(b)}"
        )

    def test_root_reused_after_first_create(self):
        """The shared root is created once and reused.

        Test scenario:
            The module-level _ROOT equals the parent of an artefact dir.
        """
        d = artifact_dir()
        assert _artifacts._ROOT == os.path.dirname(d), "root should be the dirs' parent"


class TestCleanup:
    """Tests for cleanup (also the atexit hook)."""

    def test_removes_root_and_contents(self):
        """cleanup removes the shared root and everything under it.

        Test scenario:
            A file written under an artefact dir is gone after cleanup.
        """
        d = artifact_dir()
        marker = os.path.join(d, "f.txt")
        with open(marker, "w") as fh:
            fh.write("x")
        root = _artifacts._ROOT
        cleanup()
        assert not os.path.exists(marker), "artefact file should be removed"
        assert not os.path.isdir(root), f"root should be removed: {root}"
        assert _artifacts._ROOT is None, "root handle should reset after cleanup"

    def test_unlinks_registered_vsimem(self):
        """cleanup unlinks tracked /vsimem paths.

        Test scenario:
            A /vsimem file is registered, then removed by cleanup.
        """
        path = "/vsimem/pyramids_artifacts_test.bin"
        gdal.FileFromMemBuffer(path, b"hello")
        assert gdal.VSIStatL(path) is not None, "precondition: /vsimem file exists"
        register_vsimem(path)
        cleanup()
        assert gdal.VSIStatL(path) is None, "/vsimem path should be unlinked after cleanup"

    def test_cleanup_is_idempotent(self):
        """Calling cleanup twice (or with no root) does not raise.

        Test scenario:
            A second cleanup on an already-clean state is a no-op.
        """
        artifact_dir()
        cleanup()
        cleanup()  # must not raise
