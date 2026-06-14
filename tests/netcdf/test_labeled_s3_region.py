"""Tests for anonymous S3 bucket-region resolution on LabeledDataset reads (issue #535).

GDAL's ``/vsis3`` skips region auto-resolution under ``AWS_NO_SIGN_REQUEST``, so an
anonymous read of a bucket outside ``us-east-1`` fails with an unfollowed
``PermanentRedirect``. ``resolve_s3_region`` recovers the region from the
``x-amz-bucket-region`` header and ``LabeledDataset.read_file`` pins it before the open.

These tests are fully offline — the HTTP probe and the GDAL open are mocked.
"""

from __future__ import annotations

import email.message
import io
import urllib.error
import urllib.request
import urllib.response

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

    def test_request_construction_error_returns_none(self, monkeypatch):
        """A ``ValueError`` building the request resolves to ``None`` (never raises).

        Test scenario:
            A malformed bucket string makes ``Request(...)`` raise ``ValueError``;
            the guard inside the helper swallows it so the caller's never-raises
            contract holds and the open falls back to GDAL's behaviour.
        """

        def _raise(*args, **kwargs):
            raise ValueError("unknown url type")

        monkeypatch.setattr(remote_mod.urllib.request, "Request", _raise)
        assert resolve_s3_region("bad bucket") is None

    def test_returns_none_when_region_header_absent(self, monkeypatch):
        """A 200 response without the region header resolves to ``None``.

        Test scenario:
            The bucket answers but omits ``x-amz-bucket-region`` (e.g. a
            non-S3 endpoint); the probe yields ``None`` rather than crashing.
        """
        opener = _FakeOpener(response=_FakeResponse({}))
        monkeypatch.setattr(
            remote_mod.urllib.request, "build_opener", lambda *a, **k: opener
        )
        assert resolve_s3_region("b") is None

    def test_http_error_without_headers_returns_none(self, monkeypatch):
        """An ``HTTPError`` carrying no headers resolves to ``None``.

        Test scenario:
            The error has ``headers is None``; the ``exc.headers else None``
            guard returns ``None`` instead of raising ``AttributeError``.
        """
        err = urllib.error.HTTPError(
            url="https://b.s3.amazonaws.com", code=403, msg="Forbidden", hdrs=None, fp=None
        )
        opener = _FakeOpener(error=err)
        monkeypatch.setattr(
            remote_mod.urllib.request, "build_opener", lambda *a, **k: opener
        )
        assert resolve_s3_region("b") is None

    def test_no_redirect_handler_is_a_noop_redirect_handler(self):
        """``_NoRedirectHandler`` is a redirect handler whose redirect is a no-op.

        Test scenario:
            It subclasses urllib's redirect handler, and invoking
            ``redirect_request`` does not raise (it suppresses the redirect so
            urllib raises ``HTTPError`` on a 3xx instead of following it). The
            end-to-end header read through this handler is covered by
            ``test_real_301_flow_reads_region_header``.
        """
        handler = remote_mod._NoRedirectHandler()
        assert isinstance(
            handler, urllib.request.HTTPRedirectHandler
        ), "must subclass urllib's redirect handler"
        handler.redirect_request("request", "fp", 301, "Moved", {}, "https://elsewhere")

    def test_real_301_flow_reads_region_header(self, monkeypatch):
        """A real 301 driven through ``_NoRedirectHandler`` yields the region.

        Test scenario:
            A fake HTTPS transport returns a 301 carrying ``x-amz-bucket-region``.
            The real ``build_opener(_NoRedirectHandler)`` chain must suppress the
            redirect, raise ``HTTPError``, and let ``resolve_s3_region`` read the
            header off it — exercising the handler end-to-end, not just a mocked
            ``HTTPError``.
        """
        headers = email.message.Message()
        headers["x-amz-bucket-region"] = "eu-central-1"

        class _Fake301HTTPSHandler(urllib.request.HTTPSHandler):
            """Fake transport that answers every request with a 301 + region header."""

            def https_open(self, req):
                """Return a 301 response carrying the region header."""
                response = urllib.response.addinfourl(
                    io.BytesIO(b""), headers, req.full_url, code=301
                )
                # urllib's HTTPErrorProcessor reads response.msg when turning a
                # non-2xx response into an HTTPError; addinfourl has no .msg, so
                # set it here (a future urllib change here would surface as an
                # AttributeError in this test).
                response.msg = "Moved Permanently"
                return response

        real_build_opener = urllib.request.build_opener

        def build_with_fake_transport(*handlers):
            return real_build_opener(*handlers, _Fake301HTTPSHandler())

        monkeypatch.setattr(
            remote_mod.urllib.request, "build_opener", build_with_fake_transport
        )
        assert resolve_s3_region("regional-bucket") == "eu-central-1"

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

    def test_anonymous_s3_unresolved_region_falls_back(self, monkeypatch):
        """When the region cannot be resolved, the open proceeds with none pinned.

        Test scenario:
            ``resolve_s3_region`` returns ``None`` (offline); ``AWS_REGION`` is
            left unset so GDAL's own behaviour is preserved (no regression).
        """
        monkeypatch.setattr(labeled_mod, "resolve_s3_region", lambda bucket: None)
        captured = self._capture_open(monkeypatch)
        with pytest.raises(ValueError):
            LabeledDataset.read_file("s3://bucket/store.zarr", anon=True)
        assert captured["region"] in (None, ""), f"region should be unset, got {captured['region']!r}"

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
