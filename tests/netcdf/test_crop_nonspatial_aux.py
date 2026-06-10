"""Regression tests for issue #513.

``NetCDF.crop`` (and the shared ``_apply_to_all_variables`` fan-out used by
``to_crs``) used to crash on a multi-variable root container that carried a
non-spatial auxiliary variable — e.g. an ERA5 cube with ``t2m(valid_time,
latitude, longitude)`` plus a 1-D ``expver(valid_time)`` / ``number(valid_time)``
that has no ``y`` / ``x`` axes. The fan-out now skips the non-spatial variables
(with a warning) and crops only the gridded ones.

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


_MASK = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 3.0, 3.0)], crs="EPSG:4326")


class TestCropNonSpatialAux:
    """``crop`` tolerates a non-spatial aux variable instead of crashing (#513)."""

    def test_crop_succeeds_and_keeps_spatial_variable(self):
        """Cropping an ERA5-shaped cube returns the gridded variable.

        Test scenario:
            ``t2m`` + 1-D ``number`` container, crop to a box -> the fan-out no
            longer crashes; the result contains ``t2m`` (with its 4 valid_time
            bands intact) and drops the non-spatial ``number``. The actual mask
            clipping is exercised by ``TestWholeContainerCrop`` — here the
            synthetic ``from_xarray`` cube carries no CRS, so we only assert the
            #513 regression (the fan-out tolerates the aux variable).
        """
        cube = _era5_like_cube()
        with pytest.warns(UserWarning, match="non-spatial"):
            cropped = cube.crop(mask=_MASK, touch=True)
        assert "t2m" in cropped.variable_names, "spatial t2m should survive the crop"
        assert "number" not in cropped.variable_names, "non-spatial aux should be dropped"
        t2m = cropped.get_variable("t2m")
        assert t2m.band_count == 4, f"4 valid_time bands should survive, got {t2m.band_count}"

    def test_crop_warns_naming_the_skipped_variable(self):
        """The skip warning names the non-spatial variable.

        Test scenario:
            The emitted ``UserWarning`` mentions ``number`` so the drop is
            transparent to the caller.
        """
        cube = _era5_like_cube()
        with pytest.warns(UserWarning, match="number"):
            cube.crop(mask=_MASK, touch=True)

    def test_crop_clean_cube_does_not_warn(self):
        """A cube with no aux variable crops without a skip warning.

        Test scenario:
            ``t2m``-only container -> no ``non-spatial`` warning is emitted.
        """
        cube = _era5_like_cube(with_aux=False)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            cube.crop(mask=_MASK, touch=True)
        skip_warnings = [r for r in records if "non-spatial" in str(r.message)]
        assert not skip_warnings, "a clean cube should not emit a skip warning"

    def test_all_nonspatial_container_raises(self):
        """A container with no gridded variable raises a clear error.

        Test scenario:
            Only 1-D variables -> ``crop`` raises ``ValueError`` naming the lack
            of a spatial variable, rather than crashing in the fan-out.
        """
        cube = _era5_like_cube(with_spatial=False, with_aux=True)
        with pytest.raises(ValueError, match="at least one spatial"):
            cube.crop(mask=_MASK, touch=True)

    def test_to_crs_also_skips_nonspatial_aux(self):
        """The same fan-out fix covers ``to_crs`` (reprojection).

        Test scenario:
            Reprojecting an ERA5-shaped cube skips ``number`` (with a warning)
            and reprojects ``t2m``.
        """
        cube = _era5_like_cube()
        with pytest.warns(UserWarning, match="non-spatial"):
            reprojected = cube.to_crs(3857)
        assert "t2m" in reprojected.variable_names, "spatial t2m should be reprojected"
        assert "number" not in reprojected.variable_names, "non-spatial aux dropped"
