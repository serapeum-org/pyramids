"""Tests for :meth:`FeatureCollection.read_parquet(backend="dask")`.

DASK-23 + DASK-23F: extend the GeoParquet reader with split_row_groups /
filters / blocksize kwargs, routed to :func:`dask_geopandas.read_parquet`
when ``backend="dask"``. The lazy return is wrapped in a
:class:`~pyramids.feature.LazyFeatureCollection` so the dask branch stays
inside the pyramids type system.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask_geopandas, import_pyarrow
from pyramids.feature import FeatureCollection

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
def small_parquet(tmp_path):
    """GeoParquet fixture — pyarrow is asserted at the class level via
    the ``requires_pyarrow`` mark on every class that uses this fixture,
    so no per-fixture skip is needed (N4)."""
    gdf = gpd.GeoDataFrame(
        {"id": list(range(10)), "class": ["water"] * 5 + ["land"] * 5},
        geometry=[Point(i, i) for i in range(10)],
        crs="EPSG:4326",
    )
    p = tmp_path / "pts.parquet"
    gdf.to_parquet(p)
    return str(p)


# N4: pyarrow-dependent classes carry the skip as a class-level mark so
# pytest emits a single "[N] skipped: pyarrow not installed" entry per
# class rather than one SKIPPED per test method with no shared reason.
@requires_pyarrow
class TestDefaultBackend:
    def test_returns_feature_collection(self, small_parquet):
        fc = FeatureCollection.read_parquet(small_parquet)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 10


@requires_pyarrow
@requires_dask_geopandas
class TestDaskBackend:
    def test_returns_lazy_feature_collection(self, small_parquet):
        """DASK-23F: lazy read returns a LazyFeatureCollection."""
        from pyramids.feature import LazyFeatureCollection

        lfc = FeatureCollection.read_parquet(small_parquet, backend="dask")
        assert isinstance(lfc, LazyFeatureCollection)
        assert lfc.npartitions >= 1

    def test_filters_pushed_down(self, small_parquet):
        lfc = FeatureCollection.read_parquet(
            small_parquet,
            backend="dask",
            filters=[("class", "=", "water")],
        )
        eager = lfc.compute()
        assert isinstance(eager, FeatureCollection)
        assert set(eager["class"].unique()) == {"water"}

    def test_columns_projection(self, small_parquet):
        lfc = FeatureCollection.read_parquet(
            small_parquet,
            backend="dask",
            columns=["id", "geometry"],
        )
        eager = lfc.compute()
        assert list(eager.columns) == ["id", "geometry"]

    def test_spatial_partitions_attribute_survives_wrapping(self, small_parquet):
        """DASK-23F: the spatial_partitions attribute survives the LazyFC wrap.

        Small fixtures without spatial metadata set ``spatial_partitions`` to
        ``None``; the test asserts the attribute is reachable, not its value.
        """
        lfc = FeatureCollection.read_parquet(small_parquet, backend="dask")
        assert hasattr(lfc, "spatial_partitions")


class TestValidation:
    """Validation tests work without pyarrow — path is never read."""

    def test_unknown_backend_raises(self, tmp_path):
        dummy = str(tmp_path / "ignored.parquet")
        with pytest.raises(ValueError, match="backend"):
            FeatureCollection.read_parquet(dummy, backend="bogus")

    def test_import_error_without_dask_geopandas(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dask_geopandas":
                raise ImportError("no dask-geopandas")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        dummy = str(tmp_path / "ignored.parquet")
        with pytest.raises(ImportError, match="pyramids-gis\\[parquet-lazy\\]"):
            FeatureCollection.read_parquet(dummy, backend="dask")

    @requires_dask_geopandas
    def test_import_error_when_pyarrow_missing_on_dask_branch(
        self, tmp_path, monkeypatch
    ):
        """M3: dask_geopandas installed but pyarrow isn't → branded hint.

        Test scenario:
            dask_geopandas is available so the first gate passes, then
            ``_require_pyarrow()`` runs and detects a missing pyarrow.
            The error message must name ``pyramids-gis[parquet]`` so
            users see a pyramids-branded install hint (not the
            upstream pyarrow / pyogrio message).
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pyarrow":
                raise ImportError("no pyarrow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Drop any already-cached pyarrow module so the fake_import
        # fires on the next ``import pyarrow`` inside _require_pyarrow.
        monkeypatch.delitem(__import__("sys").modules, "pyarrow", raising=False)
        dummy = str(tmp_path / "ignored.parquet")
        with pytest.raises(ImportError, match=r"pyramids-gis\[parquet\]"):
            FeatureCollection.read_parquet(dummy, backend="dask")
