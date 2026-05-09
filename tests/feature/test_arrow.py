"""Tests for :meth:`FeatureCollection.open_arrow`.

DASK-27: thin wrapper over :func:`pyogrio.raw.open_arrow` that
surfaces a :class:`pyarrow.RecordBatchReader` so callers can iterate
vector features in batches without materializing the entire table.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_pyarrow
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.parquet

try:
    import pyogrio  # noqa: F401
    from pyogrio.raw import open_arrow  # noqa: F401

    HAS_PYOGRIO = True
except ImportError:  # pragma: no cover
    HAS_PYOGRIO = False


try:
    import_pyarrow("pyarrow not installed")
    import pyarrow
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_PYARROW = False
else:
    HAS_PYARROW = True
requires_arrow = pytest.mark.skipif(
    not (HAS_PYOGRIO and HAS_PYARROW),
    reason="pyogrio + pyarrow both required for Arrow reads",
)


@pytest.fixture
def small_gpkg(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"id": list(range(10)), "name": list("abcdefghij")},
        geometry=[Point(i, i) for i in range(10)],
        crs="EPSG:4326",
    )
    path = tmp_path / "pts.gpkg"
    gdf.to_file(path, driver="GPKG")
    return str(path)


class TestOpenArrow:
    """pyogrio.raw.open_arrow returns a context manager yielding (meta, stream).

    Newer pyogrio versions made ``open_arrow`` a contextmanager; the
    stream lifetime is scoped to the ``with`` block. The concrete
    stream type depends on pyarrow availability — with pyarrow it
    becomes a real ``pyarrow.RecordBatchReader``; without it the
    stream is a minimal placeholder that supports the C-Stream ABI
    only.
    """

    @requires_arrow
    def test_stream_via_context_manager(self, small_gpkg):
        """Calling as a context manager yields a (meta, stream) pair."""
        with FeatureCollection.open_arrow(small_gpkg) as ctx:
            assert isinstance(ctx, tuple)
            assert len(ctx) == 2

    @requires_arrow
    def test_row_count_matches_source(self, small_gpkg):
        """Reading via pyarrow.RecordBatchReader.read_all yields 10 rows."""
        import pyarrow as pa

        with FeatureCollection.open_arrow(small_gpkg) as (_, stream):
            reader = pa.RecordBatchReader.from_stream(stream)
            table = reader.read_all()
        assert table.num_rows == 10


class TestImportError:
    def test_raises_without_pyogrio(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pyogrio.raw":
                raise ImportError("no pyogrio")
            if name == "pyogrio":
                raise ImportError("no pyogrio")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        dummy = str(tmp_path / "ignored.gpkg")
        with pytest.raises(ImportError, match="pyogrio"):
            FeatureCollection.open_arrow(dummy)
