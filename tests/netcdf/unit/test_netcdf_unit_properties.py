"""Unit tests for NetCDF core properties, caches, exception propagation, and spatial delegates."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import box

from tests.netcdf.conftest import make_2d_nc
from tests.netcdf.unit._netcdf_unit_helpers import _make_3d_nc

pytestmark = pytest.mark.core


class TestGeotransformFallback:
    """Tests for geotransform property when lon/lat are unavailable."""

    def test_geotransform_falls_back_when_lon_lat_none(self):
        """Verify geotransform returns _geotransform when lon/lat are None.

        Covers the branch returning self._geotransform when
        lon and lat are not available from _read_variable.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        # Patch lon and lat to return None to force fallback
        with (
            patch.object(
                type(var), "lon", new_callable=PropertyMock, return_value=None
            ),
            patch.object(
                type(var), "lat", new_callable=PropertyMock, return_value=None
            ),
        ):
            gt = var.geotransform
            assert gt is not None, "geotransform should not be None"
            assert (
                gt == var._geotransform
            ), f"Expected _geotransform fallback {var._geotransform}, got {gt}"


class TestInvalidateCaches:
    """`_invalidate_caches` clears every per-instance cache (review L7)."""

    def test_invalidate_clears_geostationary_gt_cache(self):
        """`_invalidate_caches` empties the geostationary geotransform cache.

        Test scenario:
            The per-variable `_geostationary_gt_cache` is derived from the backing
            geometry, so it must not survive a raster swap / in-place update.
            Seed it, invalidate, and assert it (and the metadata cache) are cleared.
        """
        nc = make_2d_nc()
        nc._geostationary_gt_cache["elevation"] = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
        nc._cached_meta_data = object()

        nc._invalidate_caches()

        assert (
            nc._geostationary_gt_cache == {}
        ), f"geostationary GT cache must be cleared, got {nc._geostationary_gt_cache}"
        assert nc._cached_meta_data is None, "metadata cache must be cleared"


class TestNarrowedExceptionPropagation:
    """CON-9 narrowed `except RuntimeError` blocks let unexpected exceptions propagate (gap G9)."""

    def _nc_with_fake_root_group(self, group_names_side_effect):
        """A NetCDF whose root group's `GetGroupNames` raises the given exception.

        Args:
            group_names_side_effect: Exception instance to raise from `GetGroupNames`.

        Returns:
            NetCDF: An in-memory dataset with its `_raster` swapped for a mock whose
            root group raises on `GetGroupNames`.
        """
        nc = make_2d_nc()
        fake_rg = MagicMock()
        fake_rg.GetGroupNames.side_effect = group_names_side_effect
        fake_raster = MagicMock()
        fake_raster.GetRootGroup.return_value = fake_rg
        nc._raster = fake_raster
        return nc

    def test_group_names_propagates_unexpected_exception(self):
        """An exception type outside the narrowed `except` propagates out of `group_names` (G9).

        Test scenario:
            CON-9 narrowed `group_names` to `except RuntimeError`. A `KeyError` from
            `GetGroupNames` is *unexpected* and must now propagate (the whole point of the
            narrowing) rather than being silently swallowed as it was under the old broad
            `except Exception`.
        """
        nc = self._nc_with_fake_root_group(KeyError("boom"))
        with pytest.raises(KeyError, match="boom"):
            _ = nc.group_names

    def test_group_names_degrades_on_runtime_error(self):
        """An expected `RuntimeError` still degrades to an empty list (G9).

        Test scenario:
            The narrowed `except RuntimeError` keeps the graceful-degrade contract: a GDAL
            `RuntimeError` from `GetGroupNames` yields `[]`, not a propagated error.
        """
        nc = self._nc_with_fake_root_group(RuntimeError("gdal driver error"))
        assert (
            nc.group_names == []
        ), "RuntimeError should degrade to an empty group list"


class TestSpatialOperationDelegates:
    """Tests for crop() and to_crs() delegation to parent class."""

    def test_crop_delegates_to_super(self):
        """Verify crop() passes through to Dataset.crop for subsets.

        Covers super().crop() call after _check_not_container.
        """
        nc = _make_3d_nc(rows=20, cols=24, bands=2)
        var = nc.get_variable("temperature")
        mask = gpd.GeoDataFrame(
            geometry=[box(1.0, 1.0, 5.0, 5.0)],
            crs="EPSG:4326",
        )
        result = var.crop(mask, touch=True)
        assert result is not None, "crop should return a new Dataset"
        assert (
            result.rows <= var.rows
        ), f"Cropped rows {result.rows} should be <= original {var.rows}"

    def test_to_crs_delegates_to_super(self):
        """Verify to_crs() passes through to Dataset.to_crs for subsets.

        Covers super().to_crs() call after _check_not_container.
        """
        nc = _make_3d_nc(rows=10, cols=12, bands=1, epsg=4326)
        var = nc.get_variable("temperature")
        result = var.to_crs(to_epsg=32637)
        assert result is not None, "to_crs should return a reprojected Dataset"
