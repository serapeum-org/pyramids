"""Tests for anonymous S3 bucket-region resolution on LabeledDataset reads (issue #535).

GDAL's ``/vsis3`` skips region auto-resolution under ``AWS_NO_SIGN_REQUEST``, so an
anonymous read of a bucket outside ``us-east-1`` fails with an unfollowed
``PermanentRedirect``. ``resolve_s3_region`` recovers the region from the
``x-amz-bucket-region`` header and ``LabeledDataset.read_file`` pins it before the open.

These tests are fully offline — the HTTP probe and the GDAL open are mocked.
"""

from __future__ import annotations

import urllib.error

import pytest
from osgeo import gdal

from pyramids.base import remote as remote_mod
from pyramids.base.remote import resolve_s3_region
from pyramids.netcdf import LabeledDataset
from pyramids.netcdf import labeled as labeled_mod

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def _clear_region_cache():
    """Clear the per-process region cache around each test so probes are observable."""
    remote_mod._S3_REGION_CACHE.clear()
    yield
    remote_mod._S3_REGION_CACHE.clear()


class _FakeResponse:
    """Minimal context-manager response exposing a ``headers`` mapping."""

    def __init__(self, headers: dict):
        self.headers = headers

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _FakeOpener:
    """Stand-in for ``urllib.request.build_opener`` result with a scripted ``open``."""

    def __init__(self, *, response=None, error=None):
        self._response = response
        self._error = error

    def open(self, request, timeout=None):
        """Return the scripted response or raise the scripted error."""
        if self._error is not None:
            raise self._error
        return self._response


class TestResolveS3Region:
    """Tests for the anonymous bucket-region HEAD probe."""

    def test_reads_region_header_on_success(self, monkeypatch):
        """A 200 response's ``x-amz-bucket-region`` header is returned.

        Test scenario:
            The opener yields a response carrying the region header.
        """
        opener = _FakeOpener(
            response=_FakeResponse({"x-amz-bucket-region": "eu-central-1"})
        )
        monkeypatch.setattr(
            remote_mod.urllib.request, "build_opener", lambda *a, **k: opener
        )
        assert resolve_s3_region("some-bucket") == "eu-central-1"

    def test_reads_region_header_from_http_error(self, monkeypatch):
        """The region is recovered from a 301/403 ``HTTPError``'s headers.

        Test scenario:
            S3 returns the region header even on the PermanentRedirect (301).
        """
        err = urllib.error.HTTPError(
            url="https://b.s3.amazonaws.com",
            code=301,
            msg="Moved Permanently",
            hdrs={"x-amz-bucket-region": "ap-southeast-2"},
            fp=None,
        )
        opener = _FakeOpener(error=err)
        monkeypatch.setattr(
            remote_mod.urllib.request, "build_opener", lambda *a, **k: opener
        )
        assert resolve_s3_region("b") == "ap-southeast-2"

    def test_offline_returns_none(self, monkeypatch):
        """A network failure resolves to ``None`` (caller falls back to GDAL).

        Test scenario:
            The opener raises ``URLError`` (offline / blocked).
        """
        opener = _FakeOpener(error=urllib.error.URLError("offline"))
        monkeypatch.setattr(
            remote_mod.urllib.request, "build_opener", lambda *a, **k: opener
        )
        assert resolve_s3_region("b") is None

    def test_result_is_cached(self, monkeypatch):
        """The probe runs once per bucket; the cached value is reused.

        Test scenario:
            A second call does not re-open; flipping the opener has no effect.
        """
        calls = {"n": 0}

        def build(*a, **k):
            calls["n"] += 1
            return _FakeOpener(
                response=_FakeResponse({"x-amz-bucket-region": "us-west-2"})
            )

        monkeypatch.setattr(remote_mod.urllib.request, "build_opener", build)
        first = resolve_s3_region("cached-bucket")
        second = resolve_s3_region("cached-bucket")
        assert (first, second) == ("us-west-2", "us-west-2")
        assert calls["n"] == 1


class TestReadFileRegionWiring:
    """Tests that ``read_file`` pins the right ``AWS_REGION`` for the GDAL open."""

    @staticmethod
    def _capture_open(monkeypatch) -> dict:
        """Patch ``gdal.OpenEx`` to record the active config then short-circuit.

        Args:
            monkeypatch: pytest fixture.

        Returns:
            A dict populated with ``region`` / ``nosign`` read inside the open.
        """
        captured: dict = {}

        def fake_openex(path, flags):
            captured["region"] = gdal.GetConfigOption("AWS_REGION")
            captured["nosign"] = gdal.GetConfigOption("AWS_NO_SIGN_REQUEST")
            raise RuntimeError("stop after capturing config")

        monkeypatch.setattr(labeled_mod.gdal, "OpenEx", fake_openex)
        return captured

    def test_anonymous_s3_auto_resolves_region(self, monkeypatch):
        """An anonymous S3 read pins the auto-resolved region for the open.

        Test scenario:
            ``resolve_s3_region`` yields eu-central-1; the open sees AWS_REGION set.
        """
        monkeypatch.setattr(
            labeled_mod, "resolve_s3_region", lambda bucket: "eu-central-1"
        )
        captured = self._capture_open(monkeypatch)
        with pytest.raises(ValueError):
            LabeledDataset.read_file(
                "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
                anon=True,
            )
        assert captured["region"] == "eu-central-1"
        assert captured["nosign"] == "YES"

    def test_explicit_region_overrides_and_skips_probe(self, monkeypatch):
        """An explicit ``region`` is used verbatim and the probe is not called.

        Test scenario:
            ``resolve_s3_region`` would raise if called; the open sees us-west-2.
        """

        def _boom(bucket):
            raise AssertionError("resolve_s3_region must not be called")

        monkeypatch.setattr(labeled_mod, "resolve_s3_region", _boom)
        captured = self._capture_open(monkeypatch)
        with pytest.raises(ValueError):
            LabeledDataset.read_file(
                "s3://bucket/store.zarr", anon=True, region="us-west-2"
            )
        assert captured["region"] == "us-west-2"

    def test_non_anonymous_s3_does_not_probe(self, monkeypatch):
        """A signed S3 read leaves region resolution to GDAL (no probe).

        Test scenario:
            ``resolve_s3_region`` would raise if called; no AWS_REGION is pinned.
        """

        def _boom(bucket):
            raise AssertionError("resolve_s3_region must not be called for signed reads")

        monkeypatch.setattr(labeled_mod, "resolve_s3_region", _boom)
        captured = self._capture_open(monkeypatch)
        with pytest.raises(ValueError):
            LabeledDataset.read_file("s3://bucket/store.zarr", anon=False)
        assert captured["region"] in (None, "")

    def test_local_path_does_not_probe(self, monkeypatch, tmp_path):
        """A local path never triggers a region probe.

        Test scenario:
            ``resolve_s3_region`` would raise if called for a filesystem path.
        """

        def _boom(bucket):
            raise AssertionError("resolve_s3_region must not be called for local paths")

        monkeypatch.setattr(labeled_mod, "resolve_s3_region", _boom)
        captured = self._capture_open(monkeypatch)
        missing = tmp_path / "store.zarr"
        with pytest.raises(ValueError):
            LabeledDataset.read_file(str(missing), anon=True)
        assert captured["region"] in (None, "")
