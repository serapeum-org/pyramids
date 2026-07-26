"""Tests for the bounded retry around the OGC discovery fetches (ARC-73).

The `/collections` and WCS `GetCapabilities` requests are single small fetches
in front of a much larger read, so a dropped connection or a transient 502
failing them outright wastes the whole call. `http_get_with_retry` retries those
— and only those — with exponential backoff.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

from pyramids.base._ogc_api import (
    HTTP_RETRY_ATTEMPTS,
    RETRYABLE_STATUS,
    http_get_with_retry,
)

pytestmark = pytest.mark.core


class _Opener:
    """Opener double replaying a scripted sequence of results per call."""

    def __init__(self, results):
        """Store the scripted results.

        Args:
            results: One entry per expected call — an exception instance to
                raise, or a bytes payload to return.
        """
        self.results = list(results)
        self.calls = []
        self.timeouts = []

    def open(self, target, timeout=None):
        """Return (or raise) the next scripted result.

        Args:
            target: The URL or Request handed to the opener.
            timeout: The per-attempt timeout.

        Returns:
            A file-like object over the scripted payload.

        Raises:
            BaseException: When the scripted entry is an exception.
        """
        self.calls.append(target)
        self.timeouts.append(timeout)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return io.BytesIO(result)


def _http_error(code):
    """Build an :class:`urllib.error.HTTPError` with `code`."""
    return urllib.error.HTTPError("https://h/x", code, "boom", {}, None)


class TestHttpGetWithRetry:
    """Tests for http_get_with_retry."""

    def test_first_attempt_success_returns_body(self):
        """A healthy endpoint is fetched once.

        Test scenario:
            No retry budget is spent when the first attempt succeeds.
        """
        opener = _Opener([b"payload"])
        result = http_get_with_retry("https://h/x", 5, opener=opener)
        assert result == b"payload", f"unexpected body: {result!r}"
        assert len(opener.calls) == 1, f"expected one call, got {opener.calls}"

    @pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
    def test_retryable_status_is_retried(self, status):
        """Rate limiting and transient server faults are retried.

        Args:
            status: The retryable HTTP status under test.

        Test scenario:
            One failure then a success -> two calls and the recovered body.
        """
        opener = _Opener([_http_error(status), b"ok"])
        result = http_get_with_retry(
            "https://h/x", 5, opener=opener, sleep=lambda _s: None
        )
        assert result == b"ok", f"body not recovered after a {status}: {result!r}"
        assert len(opener.calls) == 2, (
            f"expected a retry, got {len(opener.calls)} call(s)"
        )

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
    def test_client_error_is_not_retried(self, status):
        """A client error is raised on the first attempt.

        Args:
            status: The non-retryable HTTP status under test.

        Test scenario:
            Retrying a bad endpoint or bad credentials only multiplies latency.
        """
        opener = _Opener([_http_error(status), b"never reached"])
        with pytest.raises(urllib.error.HTTPError) as exc:
            http_get_with_retry("https://h/x", 5, opener=opener, sleep=lambda _s: None)
        assert exc.value.code == status, f"wrong error surfaced: {exc.value}"
        assert len(opener.calls) == 1, f"a {status} must not be retried"

    def test_transport_error_is_retried(self):
        """A dropped connection is retried.

        Test scenario:
            `URLError` (an OSError) on the first attempt, success on the second.
        """
        opener = _Opener([urllib.error.URLError("connection reset"), b"ok"])
        result = http_get_with_retry(
            "https://h/x", 5, opener=opener, sleep=lambda _s: None
        )
        assert result == b"ok", f"body not recovered: {result!r}"
        assert len(opener.calls) == 2, "a transport error should be retried"

    def test_exhausted_attempts_reraise_the_last_error(self):
        """The failure is re-raised unchanged once the budget is spent.

        Test scenario:
            Callers wrap it in their own error type, so it must not be masked.
        """
        opener = _Opener([_http_error(503)] * HTTP_RETRY_ATTEMPTS)
        with pytest.raises(urllib.error.HTTPError) as exc:
            http_get_with_retry("https://h/x", 5, opener=opener, sleep=lambda _s: None)
        assert exc.value.code == 503, f"wrong error surfaced: {exc.value}"
        assert len(opener.calls) == HTTP_RETRY_ATTEMPTS, (
            f"expected {HTTP_RETRY_ATTEMPTS} attempts, got {len(opener.calls)}"
        )

    def test_backoff_grows_exponentially(self):
        """Each retry waits twice as long as the previous one.

        Test scenario:
            Two failures with delay=0.5 -> waits of 0.5 s and 1.0 s.
        """
        waits = []
        opener = _Opener([_http_error(502), _http_error(502), b"ok"])
        http_get_with_retry(
            "https://h/x", 5, opener=opener, delay=0.5, sleep=waits.append
        )
        assert waits == [0.5, 1.0], f"unexpected backoff schedule: {waits}"

    def test_attempts_one_disables_retrying(self):
        """`attempts=1` makes the helper a plain single fetch.

        Test scenario:
            The first failure is raised immediately.
        """
        opener = _Opener([_http_error(503), b"never reached"])
        with pytest.raises(urllib.error.HTTPError):
            http_get_with_retry("https://h/x", 5, opener=opener, attempts=1)
        assert len(opener.calls) == 1, "attempts=1 should not retry"

    def test_timeout_is_forwarded_per_attempt(self):
        """Every attempt carries the caller's timeout.

        Test scenario:
            A retry must not silently drop the timeout.
        """
        opener = _Opener([_http_error(503), b"ok"])
        http_get_with_retry("https://h/x", 7.5, opener=opener, sleep=lambda _s: None)
        assert opener.timeouts == [7.5, 7.5], (
            f"timeout not forwarded: {opener.timeouts}"
        )

    def test_error_body_is_not_consumed(self):
        """The helper never reads an HTTPError's body.

        Test scenario:
            An `HTTPError` body can be read only once and the caller needs it
            for its message, so the retry loop must only inspect the status.
        """
        error = urllib.error.HTTPError(
            "https://h/x", 404, "missing", {}, io.BytesIO(b'{"detail": "gone"}')
        )
        opener = _Opener([error])
        with pytest.raises(urllib.error.HTTPError) as exc:
            http_get_with_retry("https://h/x", 5, opener=opener)
        assert exc.value.read() == b'{"detail": "gone"}', "the error body was consumed"


class TestRetryEdgeCases:
    """Boundary behaviour of the retry loop (review L2 / L3)."""

    @pytest.mark.parametrize("attempts", [0, -1])
    def test_non_positive_attempts_rejected(self, attempts):
        """A zero or negative budget is a programming error, not an empty body.

        Args:
            attempts: The invalid budget under test.

        Test scenario:
            Falling out of the loop would hand the caller `None`, and the
            failure would surface far from its cause.
        """
        opener = _Opener([b"never"])
        with pytest.raises(ValueError, match="attempts must be >= 1"):
            http_get_with_retry("https://h/x", 5, opener=opener, attempts=attempts)

    def test_retry_after_header_is_honoured(self):
        """A 429's `Retry-After` overrides the computed backoff.

        Test scenario:
            The server states the correct delay; guessing can retry straight
            back into the same rate limit.
        """
        waits = []
        error = urllib.error.HTTPError(
            "https://h/x", 429, "slow down", {"Retry-After": "4"}, None
        )
        opener = _Opener([error, b"ok"])
        http_get_with_retry(
            "https://h/x", 5, opener=opener, delay=0.5, sleep=waits.append
        )
        assert waits == [4.0], f"Retry-After not honoured: {waits}"

    @pytest.mark.parametrize("value", ["not-a-number", "-1", "3600"])
    def test_unusable_retry_after_falls_back_to_backoff(self, value):
        """A malformed, negative or absurd delay falls back to the schedule.

        Args:
            value: The unusable header value under test.

        Test scenario:
            An HTTP-date, a negative number, or an hour-long stall must not be
            taken at face value by a discovery pre-check.
        """
        waits = []
        error = urllib.error.HTTPError(
            "https://h/x", 503, "busy", {"Retry-After": value}, None
        )
        opener = _Opener([error, b"ok"])
        http_get_with_retry(
            "https://h/x", 5, opener=opener, delay=0.5, sleep=waits.append
        )
        assert waits == [0.5], f"expected the computed backoff, got {waits}"

    def test_gdal_retry_budget_matches_the_urllib_one(self):
        """GDAL counts retries, this helper counts attempts — same total.

        Test scenario:
            Emitting HTTP_RETRY_ATTEMPTS verbatim gave the driver read one more
            attempt than the pre-check.
        """
        from pyramids.base._ogc_api import GDAL_HTTP_MAX_RETRY

        assert GDAL_HTTP_MAX_RETRY == HTTP_RETRY_ATTEMPTS - 1, (
            f"budget mismatch: GDAL {GDAL_HTTP_MAX_RETRY} vs {HTTP_RETRY_ATTEMPTS}"
        )
