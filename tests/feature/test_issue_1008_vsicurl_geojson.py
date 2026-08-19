"""Reproduction / regression test for issue #1008.

``FeatureCollection.read_file`` segfaults the interpreter (exit 139) on Linux when
reading a redirecting remote GeoJSON over ``/vsicurl/``. The geoBoundaries KEN ADM1
GeoJSON is served from a ``github.com/<org>/<repo>/raw/<sha>/…`` URL that
302-redirects to ``raw.githubusercontent.com``; GDAL's GeoJSON driver following that
redirect over ``/vsicurl/`` faults natively. The identical read succeeds on Windows.

This is the exact repro from the issue, added as a plain test so the CI matrix
(Windows / Linux / macOS) shows the per-OS behaviour: Windows and macOS should read
the polygons, Linux is expected to crash until #1008 is fixed.
"""

from pyramids.feature.collection import FeatureCollection

# geoBoundaries KEN ADM1, pinned SHA (resolved from the gbOpen API on 2026-08-19).
_GEOBOUNDARIES_KEN_ADM1 = (
    "/vsicurl/https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
    "releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson"
)


def test_read_redirecting_remote_geojson_over_vsicurl():
    """Read the geoBoundaries KEN ADM1 GeoJSON over ``/vsicurl/`` (issue #1008)."""
    fc = FeatureCollection.read_file(_GEOBOUNDARIES_KEN_ADM1)
    assert len(fc) > 0, "geoBoundaries KEN ADM1 should read as non-empty polygons"
