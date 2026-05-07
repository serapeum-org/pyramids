"""ARC-V8: ``LazyFeatureCollection.to_parquet`` pyramids-shaped contract.

The parent :class:`dask_geopandas.GeoDataFrame` has its own
``to_parquet`` that returns ``None`` or a :class:`dask.delayed.Delayed`
depending on ``compute=``. The pyramids wrapper on
:class:`LazyFeatureCollection` normalises this to the same contract as
:meth:`FeatureCollection.to_parquet` and :meth:`LazyFeatureCollection.to_file`
— writers always write, they never defer.

Round-trip: lazy write → re-read as lazy → compute → len matches the
original.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask_geopandas, import_pyarrow

pytestmark = pytest.mark.parquet_lazy

try:
    import_dask_geopandas("dask-geopandas not installed")
    import dask_geopandas
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK_GP = False
else:
    HAS_DASK_GP = True
try:
    import_pyarrow("pyarrow not installed")
    import pyarrow
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_PYARROW = False
else:
    HAS_PYARROW = True
requires_dask_geopandas = pytest.mark.skipif(
    not HAS_DASK_GP, reason="dask-geopandas not installed"
)
requires_pyarrow = pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")


@pytest.fixture
def small_gdf():
    """10 points in EPSG:4326."""
    return gpd.GeoDataFrame(
        {"id": list(range(10)), "class": ["a"] * 5 + ["b"] * 5},
        geometry=[Point(i, i) for i in range(10)],
        crs="EPSG:4326",
    )


@pytest.fixture
def lfc(small_gdf):
    """A 2-partition LazyFeatureCollection built from small_gdf."""
    dg = pytest.importorskip("dask_geopandas")
    from pyramids.feature import LazyFeatureCollection

    ddf = dg.from_geopandas(small_gdf, npartitions=2)
    return LazyFeatureCollection.from_dask_gdf(ddf)


@requires_pyarrow
@requires_dask_geopandas
class TestToParquetContract:
    """The pyramids wrapper forces ``compute=True`` and returns None."""

    def test_returns_none(self, lfc, tmp_path):
        """``to_parquet`` returns ``None`` — matches the eager-side convention."""
        out = tmp_path / "lazy.parquet"
        result = lfc.to_parquet(str(out))
        assert result is None

    def test_writes_partitioned_directory(self, lfc, tmp_path):
        """Dask writes a directory of ``part.N.parquet`` files, not one file."""
        out = tmp_path / "parts"
        lfc.to_parquet(str(out))
        assert out.is_dir()
        parts = list(out.glob("*.parquet"))
        # 2 partitions → 2 part files.
        assert len(parts) == 2

    def test_compute_false_raises(self, lfc, tmp_path):
        """``compute=False`` is rejected (violates the to_* contract)."""
        out = tmp_path / "lazy.parquet"
        with pytest.raises(ValueError, match="compute=False"):
            lfc.to_parquet(str(out), compute=False)


@requires_pyarrow
@requires_dask_geopandas
class TestRoundTrip:
    """Lazy → Parquet → lazy read back → compute → equal to original."""

    def test_round_trip_preserves_rows(self, lfc, tmp_path, small_gdf):
        """Row count survives the write+read+compute loop."""
        from pyramids.feature import FeatureCollection, LazyFeatureCollection

        out = tmp_path / "round.parquet"
        lfc.to_parquet(str(out))
        reread = FeatureCollection.read_parquet(str(out), backend="dask")
        assert isinstance(reread, LazyFeatureCollection)
        eager = reread.compute()
        assert isinstance(eager, FeatureCollection)
        assert len(eager) == len(small_gdf)

    def test_round_trip_preserves_crs(self, lfc, tmp_path, small_gdf):
        """CRS survives the Parquet round-trip via standard GeoParquet metadata."""
        from pyramids.feature import FeatureCollection

        out = tmp_path / "round_crs.parquet"
        lfc.to_parquet(str(out))
        reread = FeatureCollection.read_parquet(str(out), backend="dask")
        eager = reread.compute()
        assert eager.crs == small_gdf.crs
