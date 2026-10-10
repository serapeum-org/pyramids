"""Unit tests for the opt-in asset reachability / content-type check (STAC-17).

Nothing here touches the network: every probe goes through an injected opener, or
through a monkeypatched :func:`urllib.request.urlopen`.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest

from pyramids.base._errors import StacError
from pyramids.stac import _loader
from pyramids.stac._loader import (
    AssetVerificationWarning,
    _media_types_agree,
    load_asset,
    resolved_href,
    verify_asset,
)

pytestmark = pytest.mark.core

_GEOTIFF = str(
    Path(__file__).parents[1] / "data" / "geotiff" / "era5_land_monthly_averaged.tif"
)
_COG_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"
_HREF = "https://host/a.tif"


class _Response:
    """A minimal urllib-style response: headers, an empty body, a context."""

    def __init__(self, content_type: str | None = None):
        self.headers = {} if content_type is None else {"Content-Type": content_type}

    def read(self) -> bytes:
        """Return an empty body (a HEAD has none)."""
        return b""

    def close(self) -> None:
        """Match the file-like contract; nothing to release."""

    def __enter__(self):
        """Return self so `with opener.open(...)` works."""
        return self

    def __exit__(self, *exc_info) -> bool:
        """Never swallow an exception."""
        return False


class _Opener:
    """Opener that records every request and replays a scripted outcome."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[urllib.request.Request] = []

    def open(self, target, timeout=None):
        """Record `target` and return (or raise) the next scripted outcome."""
        self.calls.append(target)
        outcome = self._outcomes[min(len(self.calls), len(self._outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def methods(self) -> list[str]:
        """Return the HTTP method of each recorded request."""
        return [call.get_method() for call in self.calls]


def _http_error(code: int, **headers: str) -> urllib.error.HTTPError:
    """Build an `HTTPError` with a readable body.

    Args:
        code: The status code to report.
        **headers: Response headers to carry (e.g. `Retry_After="0"`, whose
            underscore is normalised to the header's hyphen).

    Returns:
        urllib.error.HTTPError: Ready to be raised by a fake opener.
    """
    wire = {name.replace("_", "-"): value for name, value in headers.items()}
    return urllib.error.HTTPError(_HREF, code, "nope", wire, None)


def _verification_warnings(recwarn) -> list:
    """Return only the verification warnings a recorder captured.

    The recorder also picks up unrelated `ResourceWarning`s raised when the
    interpreter collects an unread fake `HTTPError` from an earlier test.

    Args:
        recwarn: pytest's warning recorder.

    Returns:
        list: The recorded :class:`AssetVerificationWarning` entries.
    """
    return [w for w in recwarn if issubclass(w.category, AssetVerificationWarning)]


def _fake_read_file(calls: list):
    """Return a `Dataset.read_file` stand-in that records its calls.

    Args:
        calls: List the stand-in appends each href to.

    Returns:
        Callable: A drop-in for `Dataset.read_file` returning a sentinel.
    """

    def read_file(path, *args, **kwargs):
        """Record the opened path and return a sentinel instead of a Dataset."""
        calls.append(path)
        return "opened"

    return read_file


class TestMediaTypesAgree:
    """The content-type comparison is lenient but still catches a real clash."""

    @pytest.mark.parametrize(
        ("declared", "served"),
        [
            ("image/tiff", "image/tiff"),
            (_COG_TYPE, "image/tiff"),
            ("image/tiff", "image/tiff; application=geotiff"),
            ("image/tiff", "IMAGE/TIFF"),
            ("image/tiff", "application/octet-stream"),
            ("image/tiff", None),
            (None, "text/html"),
        ],
    )
    def test_consistent_pairs_agree(self, declared, served):
        """Parameters, case and non-informative types are not contradictions."""
        assert _media_types_agree(declared, served), f"{declared!r} vs {served!r}"

    @pytest.mark.parametrize(
        ("declared", "served"),
        [
            ("image/tiff", "text/html"),
            (_COG_TYPE, "application/json"),
            ("application/x-netcdf", "image/tiff"),
        ],
    )
    def test_contradicting_pairs_disagree(self, declared, served):
        """A genuinely different served type is reported."""
        assert not _media_types_agree(declared, served), f"{declared!r} vs {served!r}"


class TestVerifyAsset:
    """`verify_asset` HEADs an HTTP href and reports what it finds."""

    def test_matching_type_is_quiet(self, recwarn):
        """A consistent content type warns about nothing and is returned."""
        opener = _Opener(_Response("image/tiff"))
        served = verify_asset(_HREF, _COG_TYPE, opener=opener)
        assert served == "image/tiff", f"served type not returned: {served!r}"
        assert opener.methods == ["HEAD"], f"not a HEAD: {opener.methods}"
        emitted = _verification_warnings(recwarn)
        assert emitted == [], f"unexpected warning: {[w.message for w in emitted]}"

    def test_mismatch_warns(self):
        """A contradicting content type raises an AssetVerificationWarning."""
        opener = _Opener(_Response("text/html"))
        with pytest.warns(AssetVerificationWarning, match="text/html"):
            verify_asset(_HREF, "image/tiff", opener=opener)

    def test_mismatch_is_strict_on_request(self):
        """`strict=True` turns the same finding into a StacError."""
        opener = _Opener(_Response("text/html"))
        with pytest.raises(StacError, match="declared 'image/tiff'"):
            verify_asset(_HREF, "image/tiff", strict=True, opener=opener)

    def test_message_redacts_the_query_string(self):
        """A signed href never reaches the message with its token intact."""
        opener = _Opener(_Response("text/html"))
        with pytest.warns(AssetVerificationWarning) as caught:
            verify_asset(f"{_HREF}?sig=SECRET", "image/tiff", opener=opener)
        message = str(caught[0].message)
        assert "SECRET" not in message, f"token leaked: {message}"
        assert "<redacted>" in message, f"href not redacted: {message}"

    def test_missing_declared_type_checks_reachability_only(self, recwarn):
        """With no declared type only reachability is verified."""
        opener = _Opener(_Response("text/html"))
        served = verify_asset(_HREF, None, opener=opener)
        assert served == "text/html", f"served type not returned: {served!r}"
        emitted = _verification_warnings(recwarn)
        assert emitted == [], f"unexpected warning: {[w.message for w in emitted]}"

    def test_error_status_warns_as_unreachable(self):
        """A 404 is reported as unreachable rather than as a type clash."""
        opener = _Opener(_http_error(404))
        with pytest.warns(AssetVerificationWarning, match="unreachable: HTTP 404"):
            verify_asset(_HREF, "image/tiff", opener=opener)

    def test_error_status_is_strict_on_request(self):
        """An unreachable asset raises under `strict=True`."""
        opener = _Opener(_http_error(404))
        with pytest.raises(StacError, match="unreachable: HTTP 404"):
            verify_asset(_HREF, "image/tiff", strict=True, opener=opener)

    def test_transport_failure_warns(self):
        """A connection-level failure is reported, not propagated."""
        opener = _Opener(urllib.error.URLError("no route to host"))
        with pytest.warns(AssetVerificationWarning, match="unreachable"):
            verify_asset(_HREF, "image/tiff", opener=opener)

    def test_head_refusal_falls_back_to_a_ranged_get(self, recwarn):
        """A host that rejects HEAD is probed with a 1-byte ranged GET."""
        opener = _Opener(_http_error(405), _Response("image/tiff"))
        served = verify_asset(_HREF, "image/tiff", opener=opener)
        assert served == "image/tiff", f"fallback lost the type: {served!r}"
        assert opener.methods == ["HEAD", "GET"], f"wrong probes: {opener.methods}"
        assert opener.calls[1].get_header("Range") == "bytes=0-0", "not a ranged GET"
        emitted = _verification_warnings(recwarn)
        assert emitted == [], f"unexpected warning: {[w.message for w in emitted]}"

    def test_retryable_status_is_not_mistaken_for_a_head_refusal(self):
        """A retryable status exhausts the shared retry budget, then reports.

        `Retry-After: 0` keeps the three attempts instant, and the probe must
        stay a HEAD throughout: 429 is a rate limit, not a refusal of the
        method, so it must not trigger the ranged-GET fallback.
        """
        rate_limited = [_http_error(429, Retry_After="0") for _ in range(3)]
        opener = _Opener(*rate_limited)
        with pytest.warns(AssetVerificationWarning, match="unreachable: HTTP 429"):
            verify_asset(_HREF, "image/tiff", opener=opener)
        assert opener.methods == ["HEAD"] * 3, f"wrong probes: {opener.methods}"

    @pytest.mark.parametrize("href", ["s3://bucket/a.tif", _GEOTIFF, "gs://b/a.tif"])
    def test_non_http_hrefs_are_skipped(self, href, recwarn):
        """Only HTTP(S) hrefs are probed; everything else is left to GDAL."""
        opener = _Opener(_Response("text/html"))
        assert verify_asset(href, "image/tiff", opener=opener) is None, "not skipped"
        assert opener.calls == [], f"probed a non-HTTP href: {opener.calls}"
        emitted = _verification_warnings(recwarn)
        assert emitted == [], f"unexpected warning: {[w.message for w in emitted]}"


class TestVerifyThroughTheLoader:
    """`verify=` is wired into `load_asset` / `resolved_href`, and is off by default."""

    @pytest.fixture
    def opened(self, monkeypatch):
        """Stub `Dataset.read_file` and return the list of opened hrefs.

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Returns:
            list: Appended to on every stubbed open.
        """
        calls: list[str] = []
        monkeypatch.setattr(_loader.Dataset, "read_file", _fake_read_file(calls))
        return calls

    @pytest.fixture
    def requests(self, monkeypatch):
        """Replace `urllib.request.urlopen` with a recorder and return its log.

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Returns:
            list: Appended to on every HTTP request that is attempted.
        """
        calls: list[urllib.request.Request] = []

        def urlopen(target, timeout=None):
            """Record the request and answer with a text/html response."""
            calls.append(target)
            return _Response("text/html")

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        return calls

    def test_verify_false_issues_no_request(self, opened, requests, recwarn):
        """The default read pays for no HEAD at all."""
        asset = {"href": _HREF, "type": _COG_TYPE}
        assert load_asset(asset) == "opened", "stubbed open not reached"
        assert requests == [], f"an unrequested HTTP call was made: {requests}"
        emitted = _verification_warnings(recwarn)
        assert emitted == [], f"unexpected warning: {[w.message for w in emitted]}"

    def test_verify_true_warns_and_still_opens(self, opened, requests):
        """A verified mismatch warns but does not block the read."""
        asset = {"href": _HREF, "type": _COG_TYPE}
        with pytest.warns(AssetVerificationWarning, match="text/html"):
            load_asset(asset, verify=True)
        assert len(requests) == 1, f"expected one probe, got {len(requests)}"
        assert opened == [_HREF], f"asset was not opened after the warning: {opened}"

    def test_verify_strict_raises_before_opening(self, opened, requests):
        """`verify_strict=True` fails before GDAL is asked for anything."""
        asset = {"href": _HREF, "type": _COG_TYPE}
        with pytest.raises(StacError, match="declared"):
            load_asset(asset, verify=True, verify_strict=True)
        assert opened == [], f"opened the asset despite a strict failure: {opened}"

    def test_verify_probes_the_signed_alternate_href(self, opened, requests):
        """The probe targets the href that will actually be opened."""
        asset = {
            "href": "https://host/canonical.tif",
            "type": _COG_TYPE,
            "alternate": {"mirror": {"href": "https://mirror/a.tif"}},
        }

        class _Signer:
            """Signer appending a token, to prove the probe sees the final href."""

            def sign_href(self, href):
                """Append a fake token."""
                return f"{href}?sig=tok"

            def gdal_env(self):
                """Return no extra GDAL config."""
                return {}

        with pytest.warns(AssetVerificationWarning):
            load_asset(asset, signer=_Signer(), alternate="mirror", verify=True)
        probed = requests[0].full_url
        assert probed == "https://mirror/a.tif?sig=tok", f"probed {probed}"

    def test_resolved_href_verifies_too(self, requests):
        """`resolved_href(verify=True)` runs the same check without opening."""
        asset = {"href": _HREF, "type": _COG_TYPE}
        with pytest.warns(AssetVerificationWarning, match="text/html"):
            href = resolved_href(asset, verify=True)
        assert href == _HREF, f"href changed by verification: {href}"

    def test_local_asset_is_opened_without_a_probe(self, requests, recwarn):
        """A local asset verifies as a no-op and reads normally."""
        asset = {"href": _GEOTIFF, "type": _COG_TYPE}
        dataset = load_asset(asset, verify=True)
        assert dataset.band_count >= 1, "local asset did not open"
        assert requests == [], f"probed a local path: {requests}"
        emitted = _verification_warnings(recwarn)
        assert emitted == [], f"unexpected warning: {[w.message for w in emitted]}"
