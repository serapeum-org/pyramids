"""Regression tests for issue #1008 — remote GeoJSON is staged, not streamed.

``FeatureCollection.read_file`` on a *redirecting* remote GeoJSON over GDAL's
``/vsicurl/`` can segfault the interpreter in a build whose bundled libcurl/OpenSSL
differs from the interpreter's (the manylinux wheel, whose vendored OpenSSL 3 clashes
with CPython's). The fix fetches the bytes with :mod:`urllib` (Python's own TLS, which
follows the redirect) and hands GDAL a plain local file, so GDAL never does the remote
read.

The offline tests below assert that routing decision deterministically (they run in the
normal matrix); the ``live`` test reads the real geoBoundaries source end to end.
"""

import io

import pytest

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


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/vsicurl/https://host/x.geojson", True),
        ("https://host/x.geojson", True),
        ("http://host/x.geojson", True),
        ("https://host/x.json", True),
        ("https://host/data.geojson?token=abc", True),
        ("/vsicurl/https://host/x.gpkg", False),
        ("https://host/x.tif", False),
        ("/data/local/x.geojson", False),
        ("s3://bucket/x.geojson", False),
    ],
)
def test_is_remote_geojson_detection(path: str, expected: bool):
    """Only ``http(s)://`` GeoJSON (bare or ``/vsicurl/``-wrapped) is staged."""
    assert _read._is_remote_geojson(path) is expected


def test_remote_geojson_is_staged_not_streamed(monkeypatch: pytest.MonkeyPatch):
    """A remote GeoJSON is downloaded and read from a local file, never over ``/vsicurl/``."""
    seen: dict[str, str] = {}
    real_read_file = _read.gpd.read_file

    def _spy_read_file(target, **kwargs):
        seen["path"] = str(target)
        return real_read_file(target, **kwargs)

    monkeypatch.setattr(
        _read.urllib.request, "urlopen", lambda request, **_: io.BytesIO(_POINT_GEOJSON)
    )
    monkeypatch.setattr(_read.gpd, "read_file", _spy_read_file)

    fc = FeatureCollection.read_file(_GEOBOUNDARIES_KEN_ADM1)

    assert len(fc) == 1
    assert "/vsicurl/" not in seen["path"], f"read a local copy, not /vsicurl/: {seen['path']}"
    assert seen["path"].endswith(".geojson")


@pytest.mark.live
def test_read_geoboundaries_ken_adm1_end_to_end():
    """The #1008 geoBoundaries KEN ADM1 GeoJSON reads without crashing (real network)."""
    fc = FeatureCollection.read_file(_GEOBOUNDARIES_KEN_ADM1)
    assert len(fc) > 0, "geoBoundaries KEN ADM1 should read as non-empty polygons"
