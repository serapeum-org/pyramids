"""Regression tests for issue #1008 — remote GeoJSON is staged, not streamed.

``FeatureCollection.read_file`` on a *redirecting* remote GeoJSON over GDAL's
``/vsicurl/`` can segfault the interpreter in a build whose bundled libcurl/OpenSSL
differs from the interpreter's (the manylinux wheel, whose vendored OpenSSL 3 clashes
with CPython's). The fix fetches the bytes with :mod:`urllib` (Python's own TLS, which
follows the redirect) and hands GDAL a plain local file, so GDAL never does the remote
read.

These tests cover the three helpers added for the fix — :func:`_strip_vsicurl`,
:func:`_is_remote_geojson`, :func:`_read_remote_geojson_staged` — and the routing branch
they drive in :func:`pyramids.feature._read.read_file`. Every offline test is
deterministic (mocked download) and runs in the normal matrix; the single ``live`` test
reads the real geoBoundaries source end to end.
"""

import io
import urllib.error
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.feature import _read
from pyramids.feature.collection import FeatureCollection

# geoBoundaries KEN ADM1, pinned SHA — a github.com/.../raw/<sha>/… URL that 302-redirects
# to raw.githubusercontent.com; this is the exact source from issue #1008.
_GEOBOUNDARIES_KEN_ADM1 = (
    "/vsicurl/https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
    "releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson"
)

_POINT_GEOJSON = (
    b'{"type":"FeatureCollection","features":[{"type":"Feature",'
    b'"properties":{"n":1},"geometry":{"type":"Point","coordinates":[0,0]}}]}'
)


def _fake_gdf() -> gpd.GeoDataFrame:
    """Return a one-row GeoDataFrame standing in for a parsed GeoJSON.

    Returns:
        geopandas.GeoDataFrame: A single WGS84 point feature with an ``n`` column.
    """
    return gpd.GeoDataFrame({"n": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")


class TestStripVsicurl:
    """Tests for :func:`pyramids.feature._read._strip_vsicurl`."""

    def test_strips_leading_vsicurl_prefix(self):
        """Strip the ``/vsicurl/`` wrapper to recover the bare URL.

        Test scenario:
            A ``/vsicurl/https://…`` path returns the ``https://…`` URL behind it.
        """
        result = _read._strip_vsicurl("/vsicurl/https://host/x.geojson")
        assert result == "https://host/x.geojson", f"prefix not stripped: {result}"

    def test_bare_url_is_unchanged(self):
        """Leave a URL that carries no ``/vsicurl/`` prefix untouched.

        Test scenario:
            A bare ``https://…`` URL is returned verbatim.
        """
        result = _read._strip_vsicurl("https://host/x.geojson")
        assert result == "https://host/x.geojson", f"bare URL altered: {result}"

    def test_strips_streaming_prefix(self):
        """Strip a leading `/vsicurl_streaming/` wrapper as well as `/vsicurl/`.

        Test scenario:
            A `/vsicurl_streaming/https://…` path returns the bare `https://…` URL.
        """
        result = _read._strip_vsicurl("/vsicurl_streaming/https://host/x.geojson")
        assert result == "https://host/x.geojson", f"streaming prefix not stripped: {result}"

    def test_only_leading_prefix_is_stripped(self):
        """Strip only a *leading* ``/vsicurl/``, never one nested mid-path.

        Test scenario:
            A chained ``/vsizip//vsicurl/…`` path does not start with the prefix, so it
            is returned unchanged.
        """
        chained = "/vsizip//vsicurl/https://host/x.zip"
        assert _read._strip_vsicurl(chained) == chained, "nested prefix wrongly stripped"


class TestIsRemoteGeojson:
    """Tests for :func:`pyramids.feature._read._is_remote_geojson`."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/vsicurl/https://host/x.geojson", True),
            ("/vsicurl_streaming/https://host/x.geojson", True),
            ("https://host/x.geojson", True),
            ("https://host/data.geojson?token=abc", True),
            ("https://host/x.json", False),
            ("https://host/x.geojson/", True),
            ("HTTPS://host/x.geojson", True),
            ("https://host/X.GEOJSON", True),
            ("http://host/x.geojson", False),
            ("/vsicurl/https://host/x.gpkg", False),
            ("https://host/x.tif", False),
            ("https://host/x.json.gz", False),
            ("/data/local/x.geojson", False),
            ("s3://bucket/x.geojson", False),
            ("gs://bucket/x.geojson", False),
            ("https://host/data.zip/inner.geojson", False),
            ("https://host/data.tar.gz/inner.geojson", False),
            ("https://my.zip/data.geojson", True),
            ("https://host.gz/data.geojson", True),
        ],
    )
    def test_detection_matrix(self, path: str, expected: bool):
        """Only an ``https://`` GeoJSON (bare or ``/vsicurl/``-wrapped) is detected.

        Args:
            path: The candidate path or URL.
            expected: Whether it should be routed through local staging.

        Test scenario:
            ``https://`` GeoJSON (any case, trailing slash, or query string) is ``True``;
            plain ``http://``, non-GeoJSON extensions, object-store schemes, and local
            paths are ``False``.
        """
        assert _read._is_remote_geojson(path) is expected, f"misclassified {path!r}"

    def test_accepts_path_object(self):
        """Accept a :class:`pathlib.Path` and classify a local path as ``False``.

        Test scenario:
            A local ``Path`` (no ``https://`` scheme) is not staged.
        """
        assert _read._is_remote_geojson(Path("/data/local/x.geojson")) is False


class TestReadRemoteGeojsonStaged:
    """Tests for :func:`pyramids.feature._read._read_remote_geojson_staged`."""

    def test_downloads_and_reads_local_copy(self, monkeypatch: pytest.MonkeyPatch):
        """Download the bytes and read them from a local file, never ``/vsicurl/``.

        Test scenario:
            The mocked download yields a one-point GeoJSON; the resulting collection has
            one feature and ``gpd.read_file`` is handed a local ``.geojson`` path.
        """
        seen: dict[str, str] = {}
        real_read_file = _read.gpd.read_file

        def _spy_read_file(target, **kwargs):
            seen["path"] = str(target)
            return real_read_file(target, **kwargs)

        monkeypatch.setattr(
            _read.urllib.request, "urlopen", lambda request, **_: io.BytesIO(_POINT_GEOJSON)
        )
        monkeypatch.setattr(_read.gpd, "read_file", _spy_read_file)

        fc = _read._read_remote_geojson_staged(FeatureCollection, _GEOBOUNDARIES_KEN_ADM1, {})

        assert isinstance(fc, FeatureCollection), f"wrong return type: {type(fc)}"
        assert len(fc) == 1, f"expected one feature, got {len(fc)}"
        assert "/vsicurl/" not in seen["path"], f"read a local copy, not /vsicurl/: {seen['path']}"
        assert seen["path"].endswith(".geojson"), f"staged file is not .geojson: {seen['path']}"

    def test_request_strips_vsicurl_and_sets_user_agent(self, monkeypatch: pytest.MonkeyPatch):
        """Fetch the bare URL (``/vsicurl/`` removed) with an explicit User-Agent.

        Test scenario:
            The :class:`urllib.request.Request` handed to ``urlopen`` targets the
            redirect-following ``https://`` URL and carries a ``pyramids-gis`` UA header.
        """
        captured: dict[str, object] = {}

        def _fake_urlopen(request, **_):
            captured["url"] = request.full_url
            captured["ua"] = request.get_header("User-agent")
            return io.BytesIO(_POINT_GEOJSON)

        monkeypatch.setattr(_read.urllib.request, "urlopen", _fake_urlopen)

        _read._read_remote_geojson_staged(FeatureCollection, _GEOBOUNDARIES_KEN_ADM1, {})

        expected_url = _GEOBOUNDARIES_KEN_ADM1[len("/vsicurl/") :]
        assert captured["url"] == expected_url, f"unexpected fetch URL: {captured['url']}"
        assert captured["ua"] == "pyramids-gis", f"missing/incorrect UA header: {captured['ua']}"

    def test_passthrough_kwargs_reach_the_reader(self, monkeypatch: pytest.MonkeyPatch):
        """Forward filter kwargs to ``gpd.read_file`` on the staged local file.

        Test scenario:
            ``columns`` and ``where`` passed through ``read_file`` reach the local read
            unchanged, so pushdown filters still apply after staging.
        """
        captured: dict[str, object] = {}

        def _spy_read_file(target, **kwargs):
            captured["kwargs"] = kwargs
            return _fake_gdf()

        monkeypatch.setattr(
            _read.urllib.request, "urlopen", lambda request, **_: io.BytesIO(_POINT_GEOJSON)
        )
        monkeypatch.setattr(_read.gpd, "read_file", _spy_read_file)

        passthrough = {"columns": ["n"], "where": "n = 1"}
        _read._read_remote_geojson_staged(FeatureCollection, _GEOBOUNDARIES_KEN_ADM1, passthrough)

        assert captured["kwargs"] == passthrough, f"kwargs not forwarded: {captured['kwargs']}"

    def test_download_error_propagates(self, monkeypatch: pytest.MonkeyPatch):
        """Let a download failure surface as the original ``URLError``.

        Test scenario:
            When ``urlopen`` raises :class:`urllib.error.URLError`, the staging helper
            does not swallow it — the caller sees the network error.
        """

        def _boom(request, **_):
            raise urllib.error.URLError("boom")

        monkeypatch.setattr(_read.urllib.request, "urlopen", _boom)

        with pytest.raises(urllib.error.URLError, match="boom"):
            _read._read_remote_geojson_staged(FeatureCollection, _GEOBOUNDARIES_KEN_ADM1, {})

    def test_rejects_non_https_url(self):
        """Refuse a non-https URL as a self-guard on the https invariant.

        Test scenario:
            Reaching the helper directly with a plain `http://` URL raises `ValueError`
            rather than silently fetching over an unintended scheme.
        """
        with pytest.raises(ValueError, match="https://"):
            _read._read_remote_geojson_staged(FeatureCollection, "http://host/x.geojson", {})

    def test_download_uses_an_explicit_timeout(self, monkeypatch: pytest.MonkeyPatch):
        """Pass an explicit `timeout` so a stalled TLS server cannot hang the read forever.

        Test scenario:
            `urlopen` receives `timeout=_REMOTE_READ_TIMEOUT` rather than the unbounded
            process default.
        """
        captured: dict[str, object] = {}

        def _fake_urlopen(request, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return io.BytesIO(_POINT_GEOJSON)

        monkeypatch.setattr(_read.urllib.request, "urlopen", _fake_urlopen)

        _read._read_remote_geojson_staged(FeatureCollection, _GEOBOUNDARIES_KEN_ADM1, {})

        assert captured["timeout"] == _read._REMOTE_READ_TIMEOUT, (
            f"expected an explicit timeout, got {captured['timeout']}"
        )


class TestReadFileRemoteGeojsonRouting:
    """Tests for the remote-GeoJSON routing branch in :func:`pyramids.feature._read.read_file`."""

    def test_pandas_remote_geojson_is_staged(self, monkeypatch: pytest.MonkeyPatch):
        """Route a remote GeoJSON to staging on the default (pandas) backend.

        Test scenario:
            With ``backend='pandas'`` and an ``https://`` GeoJSON URL, ``read_file``
            delegates to ``_read_remote_geojson_staged`` and returns its result.
        """
        sentinel = object()
        monkeypatch.setattr(
            _read, "_read_remote_geojson_staged", lambda cls, path, passthrough: sentinel
        )

        result = _read.read_file(FeatureCollection, _GEOBOUNDARIES_KEN_ADM1)

        assert result is sentinel, "remote GeoJSON was not routed to staging"

    def test_non_geojson_remote_is_not_staged(self, monkeypatch: pytest.MonkeyPatch):
        """Leave a non-GeoJSON remote read on the ``/vsicurl/`` path.

        Test scenario:
            A remote ``.gpkg`` URL never enters staging; it flows through the normal
            ``_read_file_healing_crs`` reader (mocked here to avoid I/O).
        """

        def _must_not_stage(*_args, **_kwargs):
            raise AssertionError("non-GeoJSON remote should not be staged")

        monkeypatch.setattr(_read, "_read_remote_geojson_staged", _must_not_stage)
        monkeypatch.setattr(_read, "_read_file_healing_crs", lambda resolved, passthrough: _fake_gdf())

        result = _read.read_file(FeatureCollection, "https://host/x.gpkg")

        assert isinstance(result, FeatureCollection), f"wrong return type: {type(result)}"

    def test_dask_backend_bypasses_staging(self, monkeypatch: pytest.MonkeyPatch):
        """Never stage on the dask backend, even for a remote GeoJSON.

        Test scenario:
            With ``backend='dask'`` and a GeoJSON URL, ``read_file`` skips staging and
            delegates to ``read_file_dask``.
        """
        sentinel = object()

        def _must_not_stage(*_args, **_kwargs):
            raise AssertionError("dask backend should not stage")

        monkeypatch.setattr(_read, "_read_remote_geojson_staged", _must_not_stage)
        monkeypatch.setattr(_read, "read_file_dask", lambda *a, **k: sentinel)

        result = _read.read_file(FeatureCollection, _GEOBOUNDARIES_KEN_ADM1, backend="dask")

        assert result is sentinel, "dask backend did not bypass staging"

    def test_invalid_backend_raises(self):
        """Reject an unknown backend with a clear ``ValueError``.

        Test scenario:
            A non-GeoJSON local path with ``backend='threads'`` raises ``ValueError``
            naming the accepted backends.
        """
        with pytest.raises(ValueError, match="backend must be 'pandas' or 'dask'"):
            _read.read_file(FeatureCollection, "local/x.shp", backend="threads")

    @pytest.mark.live
    def test_read_geoboundaries_ken_adm1_end_to_end(self):
        """Read the real #1008 geoBoundaries GeoJSON end to end without crashing.

        Test scenario:
            The redirecting geoBoundaries KEN ADM1 URL reads as non-empty polygons over
            the staging path (real network).
        """
        fc = FeatureCollection.read_file(_GEOBOUNDARIES_KEN_ADM1)
        assert len(fc) > 0, "geoBoundaries KEN ADM1 should read as non-empty polygons"


class TestGdalHttpOptionsGuard:
    """Tests for the GDAL-HTTP-options guard that keeps authed reads on `/vsicurl/` (#1008 M1)."""

    def test_active_when_an_option_is_set(self, monkeypatch: pytest.MonkeyPatch):
        """`_gdal_http_options_active` is True when any GDAL HTTP option is set.

        Test scenario:
            A set `GDAL_HTTP_HEADERS` (e.g. a bearer token) makes the guard report active.
        """
        monkeypatch.setattr(
            _read.gdal,
            "GetConfigOption",
            lambda key: "Bearer x" if key == "GDAL_HTTP_HEADERS" else None,
        )
        assert _read._gdal_http_options_active() is True, "set option should read as active"

    def test_active_for_bearer_only_config(self, monkeypatch: pytest.MonkeyPatch):
        """A bearer-only config (`GDAL_HTTP_BEARER`) is detected so the token still reaches GDAL.

        Test scenario:
            A token-gated read authenticated purely with `GDAL_HTTP_BEARER` is not staged.
        """
        monkeypatch.setattr(
            _read.gdal,
            "GetConfigOption",
            lambda key: "tok" if key == "GDAL_HTTP_BEARER" else None,
        )
        assert _read._gdal_http_options_active() is True, "bearer token should read as active"

    def test_inactive_when_no_option_is_set(self, monkeypatch: pytest.MonkeyPatch):
        """`_gdal_http_options_active` is False when no GDAL HTTP option is set.

        Test scenario:
            With every option unset, the guard reports inactive and staging may proceed.
        """
        monkeypatch.setattr(_read.gdal, "GetConfigOption", lambda key: None)
        assert _read._gdal_http_options_active() is False, "unset options should read inactive"

    def test_read_file_skips_staging_when_option_active(self, monkeypatch: pytest.MonkeyPatch):
        """A set GDAL HTTP option keeps the remote GeoJSON on the `/vsicurl/` reader.

        Test scenario:
            When the guard is active, `read_file` does not stage — the caller's GDAL auth
            still reaches GDAL — and the read flows through the normal reader.
        """

        def _must_not_stage(*_args, **_kwargs):
            raise AssertionError("staging must be skipped when a GDAL HTTP option is set")

        monkeypatch.setattr(_read, "_gdal_http_options_active", lambda: True)
        monkeypatch.setattr(_read, "_read_remote_geojson_staged", _must_not_stage)
        monkeypatch.setattr(_read, "_read_file_healing_crs", lambda resolved, passthrough: _fake_gdf())

        result = _read.read_file(FeatureCollection, _GEOBOUNDARIES_KEN_ADM1)

        assert isinstance(result, FeatureCollection), f"wrong return type: {type(result)}"
