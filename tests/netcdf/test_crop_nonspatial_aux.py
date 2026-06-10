"""Regression tests for issue #513.

``NetCDF.crop`` (and the shared ``_apply_to_all_variables`` fan-out used by
``to_crs``) used to crash on a multi-variable root container that carried a
non-spatial auxiliary variable — e.g. an ERA5 cube with ``t2m(valid_time,
latitude, longitude)`` plus a 1-D ``expver(valid_time)`` / ``number(valid_time)``
that has no ``y`` / ``x`` axes. The fan-out now crops only the gridded variables
(detected per variable) and **carries the non-spatial auxiliary variables through
unchanged** into the result, the way ``rioxarray.clip`` leaves them alone.

These tests build the cube through ``NetCDF.from_xarray`` (pyramids' own writer),
so the whole module is skipped when xarray is unavailable. ERA5's real ``expver``
is a string variable, which ``from_xarray`` does not write; the defect is
dtype-agnostic (it is about a 1-D non-spatial variable in the fan-out), so a
numeric ``number`` aux reproduces it faithfully.
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.netcdf.netcdf import NetCDF

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.core


def _era5_like_cube(*, with_spatial=True, with_aux=True):
    """Build an ERA5-shaped multidimensional cube via ``NetCDF.from_xarray``.

    Args:
        with_spatial: Include the gridded ``t2m(valid_time, latitude, longitude)``.
        with_aux: Include the non-spatial 1-D ``number(valid_time)`` auxiliary.

    Returns:
        NetCDF: The opened multi-variable container.
    """
    n_t, n_lat, n_lon = 4, 5, 5
    lat = np.arange(5.0, 0.0, -1.0)
    lon = np.arange(0.0, 5.0, 1.0)
    data_vars = {}
    if with_spatial:
        data_vars["t2m"] = (
            ("valid_time", "latitude", "longitude"),
            np.ones((n_t, n_lat, n_lon), "float32"),
        )
    if with_aux:
        data_vars["number"] = (("valid_time",), np.array([0, 0, 1, 1], dtype="int32"))
    coords = {"valid_time": np.arange(n_t), "latitude": lat, "longitude": lon}
    ds = xr.Dataset(data_vars, coords=coords)
    ds.latitude.attrs.update(units="degrees_north", standard_name="latitude")
    ds.longitude.attrs.update(units="degrees_east", standard_name="longitude")
    return NetCDF.from_xarray(ds)


def _raw_values(cube, var_name):
    """Read a variable's raw MDArray values straight from the root group."""
    return np.asarray(cube._raster.GetRootGroup().OpenMDArray(var_name).ReadAsArray())


_MASK = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 3.0, 3.0)], crs="EPSG:4326")


class TestCropNonSpatialAux:
    """``crop`` tolerates and preserves a non-spatial aux variable (#513)."""

    def test_crop_succeeds_and_keeps_both_variables(self):
        """Cropping an ERA5-shaped cube keeps the gridded and the aux variable.

        Test scenario:
            ``t2m`` + 1-D ``number`` container, crop to a box -> the fan-out no
            longer crashes; the result contains both ``t2m`` (with its 4
            valid_time bands intact) and the carried-through ``number``. Actual
            mask clipping is covered by ``TestWholeContainerCrop`` — here the
            synthetic ``from_xarray`` cube carries no CRS, so we only assert the
            #513 regression.
        """
        cube = _era5_like_cube()
        cropped = cube.crop(mask=_MASK, touch=True)
        assert "t2m" in cropped.variable_names, "spatial t2m should survive the crop"
        assert "number" in cropped.variable_names, "non-spatial aux should be carried"
        t2m = cropped.get_variable("t2m")
        assert t2m.band_count == 4, f"4 valid_time bands should survive, got {t2m.band_count}"

    def test_crop_carries_aux_values_unchanged(self):
        """The carried non-spatial variable keeps its raw values.

        Test scenario:
            ``number == [0, 0, 1, 1]`` before and after the crop.
        """
        cube = _era5_like_cube()
        cropped = cube.crop(mask=_MASK, touch=True)
        assert list(_raw_values(cropped, "number")) == [0, 0, 1, 1], (
            "carried aux values should be unchanged"
        )

    def test_crop_does_not_warn(self):
        """A cube with a carried aux variable crops without any skip/carry warning.

        Test scenario:
            crop the ERA5-shaped cube -> no ``non-spatial`` / ``carry`` warning.
        """
        cube = _era5_like_cube()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            cube.crop(mask=_MASK, touch=True)
        noise = [r for r in records if "non-spatial" in str(r.message) or "carry" in str(r.message)]
        assert not noise, f"unexpected skip/carry warning: {[str(r.message) for r in noise]}"

    def test_all_nonspatial_container_raises(self):
        """A container with no gridded variable raises a clear error.

        Test scenario:
            Only 1-D variables -> ``crop`` raises ``ValueError`` naming the lack
            of a spatial variable, rather than crashing in the fan-out.
        """
        cube = _era5_like_cube(with_spatial=False, with_aux=True)
        with pytest.raises(ValueError, match="at least one spatial"):
            cube.crop(mask=_MASK, touch=True)

    def test_to_crs_also_carries_nonspatial_aux(self):
        """The same fan-out fix covers ``to_crs`` (reprojection).

        Test scenario:
            Reprojecting an ERA5-shaped cube reprojects ``t2m`` and carries
            ``number`` through.
        """
        cube = _era5_like_cube()
        reprojected = cube.to_crs(3857)
        assert "t2m" in reprojected.variable_names, "spatial t2m should be reprojected"
        assert "number" in reprojected.variable_names, "non-spatial aux should be carried"

    def test_reduce_also_carries_nonspatial_aux(self):
        """``reduce`` (its own fan-out loop) tolerates the aux variable too.

        Test scenario:
            Reducing the time dimension of an ERA5-shaped cube reduces ``t2m``
            and carries the non-spatial ``number`` through instead of crashing
            in ``get_variable`` (the same #513 defect class).
        """
        cube = _era5_like_cube()
        reduced = cube.reduce("valid_time", "mean")
        assert "t2m" in reduced.variable_names, "spatial t2m should be reduced"
        assert "number" in reduced.variable_names, "non-spatial aux should be carried"
        assert reduced.get_variable("t2m").band_count == 1, "valid_time collapsed to 1"
