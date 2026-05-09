"""ARC-V3: public dispatch helpers on :mod:`pyramids.feature`.

Pin the contract of :func:`has_lazy_backend` and :func:`is_lazy_fc`:

* ``has_lazy_backend()`` returns ``True`` when ``dask-geopandas`` was
  importable at :mod:`pyramids.feature` import time, else ``False``.
* ``is_lazy_fc(obj)`` returns ``True`` iff ``obj`` is a
  :class:`LazyFeatureCollection` instance. Safe to call on minimal
  installs — returns ``False`` rather than raising ``ImportError``.

Both helpers replace the ``try/except ImportError`` dance the L3
sentinel pattern otherwise forces on library authors.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask_geopandas
from pyramids.feature import (
    FeatureCollection,
    has_lazy_backend,
    is_lazy_fc,
)

pytestmark = pytest.mark.parquet_lazy

try:
    import_dask_geopandas("dask-geopandas not installed")
    import dask_geopandas
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK_GP = False
else:
    HAS_DASK_GP = True
requires_dask_geopandas = pytest.mark.skipif(
    not HAS_DASK_GP, reason="dask-geopandas not installed"
)


@pytest.fixture
def eager_fc() -> FeatureCollection:
    """Simple eager FeatureCollection for negative-case tests."""
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[Point(0, 0)],
            crs="EPSG:4326",
        )
    )


class TestHasLazyBackend:
    """Feature-detection helper."""

    def test_returns_bool(self):
        """The return type is always bool, not truthy-ish."""
        result = has_lazy_backend()
        assert isinstance(result, bool)

    def test_matches_actual_install(self):
        """The flag reflects whether dask-geopandas imported successfully."""
        assert has_lazy_backend() is HAS_DASK_GP


class TestIsLazyFcNegative:
    """Objects that are NOT a LazyFeatureCollection return False."""

    def test_eager_fc_is_not_lazy(self, eager_fc):
        """An eager FeatureCollection is not a LazyFC."""
        assert is_lazy_fc(eager_fc) is False

    def test_string_is_not_lazy(self):
        """Plain objects return False, not raise."""
        assert is_lazy_fc("not a frame") is False

    def test_none_is_not_lazy(self):
        """``None`` returns False cleanly."""
        assert is_lazy_fc(None) is False

    def test_dict_is_not_lazy(self):
        """Dicts return False (no LazyFC duck-typing leakage)."""
        assert is_lazy_fc({"npartitions": 4}) is False

    def test_plain_geodataframe_is_not_lazy(self):
        """An eager geopandas.GeoDataFrame is not a LazyFC."""
        gdf = gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[Point(0, 0)],
            crs="EPSG:4326",
        )
        assert is_lazy_fc(gdf) is False


@requires_dask_geopandas
class TestIsLazyFcPositive:
    """A constructed :class:`LazyFeatureCollection` returns True."""

    def test_from_dask_gdf_is_lazy(self):
        """A LazyFC built via ``from_dask_gdf`` is recognised."""
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        gdf = gpd.GeoDataFrame(
            {"v": [1, 2]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )
        ddf = dg.from_geopandas(gdf, npartitions=1)
        lfc = LazyFeatureCollection.from_dask_gdf(ddf)
        assert is_lazy_fc(lfc) is True

    def test_compute_result_is_not_lazy(self):
        """``.compute()`` produces an eager FC — no longer a LazyFC."""
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        gdf = gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[Point(0, 0)],
            crs="EPSG:4326",
        )
        ddf = dg.from_geopandas(gdf, npartitions=1)
        lfc = LazyFeatureCollection.from_dask_gdf(ddf)
        eager = lfc.compute()
        assert is_lazy_fc(eager) is False


class TestMinimalInstallSafety:
    """Both helpers are safe to call when dask-geopandas is absent."""

    def test_is_lazy_fc_does_not_raise_without_dg(self, monkeypatch):
        """Simulated minimal install → ``is_lazy_fc`` returns False cleanly.

        Test scenario:
            With ``_HAS_DASK_GEOPANDAS`` patched to False (the
            minimal-install case), ``is_lazy_fc(anything)`` returns
            False without trying to import the lazy class — the short
            circuit at the top of the function guarantees it.
        """
        import pyramids.feature as pf

        monkeypatch.setattr(pf, "_HAS_DASK_GEOPANDAS", False)
        # Any object we pass, including things that might look lazy,
        # must return False without raising.
        assert pf.is_lazy_fc(object()) is False
        assert pf.is_lazy_fc(None) is False

    def test_has_lazy_backend_reflects_flag(self, monkeypatch):
        """``has_lazy_backend()`` tracks ``_HAS_DASK_GEOPANDAS`` exactly."""
        import pyramids.feature as pf

        monkeypatch.setattr(pf, "_HAS_DASK_GEOPANDAS", False)
        assert pf.has_lazy_backend() is False

        monkeypatch.setattr(pf, "_HAS_DASK_GEOPANDAS", True)
        assert pf.has_lazy_backend() is True
