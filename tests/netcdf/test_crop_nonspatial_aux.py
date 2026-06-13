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
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.netcdf.netcdf import NetCDF

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.core


def _era5_like_cube(*, with_spatial=True, with_aux=True, with_second_spatial=False):
    """Build an ERA5-shaped multidimensional cube via ``NetCDF.from_xarray``.

    Args:
        with_spatial: Include the gridded ``t2m(valid_time, latitude, longitude)``.
        with_aux: Include the non-spatial 1-D ``number(valid_time)`` auxiliary.
        with_second_spatial: Also include a second gridded ``tp`` variable.

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
    if with_second_spatial:
        data_vars["tp"] = (
            ("valid_time", "latitude", "longitude"),
            np.full((n_t, n_lat, n_lon), 2.0, "float32"),
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
        assert (
            t2m.band_count == 4
        ), f"4 valid_time bands should survive, got {t2m.band_count}"

    def test_crop_carries_aux_values_unchanged(self):
        """The carried non-spatial variable keeps its raw values.

        Test scenario:
            ``number == [0, 0, 1, 1]`` before and after the crop.
        """
        cube = _era5_like_cube()
        cropped = cube.crop(mask=_MASK, touch=True)
        assert list(_raw_values(cropped, "number")) == [
            0,
            0,
            1,
            1,
        ], "carried aux values should be unchanged"

    def test_crop_does_not_warn(self):
        """A cube with a carried aux variable crops without any skip/carry warning.

        Test scenario:
            crop the ERA5-shaped cube -> no ``non-spatial`` / ``carry`` warning.
        """
        cube = _era5_like_cube()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            cube.crop(mask=_MASK, touch=True)
        noise = [
            r
            for r in records
            if "non-spatial" in str(r.message) or "carry" in str(r.message)
        ]
        assert (
            not noise
        ), f"unexpected skip/carry warning: {[str(r.message) for r in noise]}"

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
        assert (
            "number" in reprojected.variable_names
        ), "non-spatial aux should be carried"

    def test_reduce_drops_aux_spanning_the_reduced_dim(self):
        """reduce drops an aux variable that spans the reduced dim, with a warning (M5).

        Test scenario:
            ``number(valid_time)`` spans the reduced dimension; carrying it
            verbatim would leave an inconsistent ``valid_time`` length while
            ``t2m`` collapses it. reduce must drop the spanning aux and warn,
            not produce a malformed container or crash in ``get_variable``.
        """
        cube = _era5_like_cube()
        with pytest.warns(UserWarning, match="span the reduced dimension"):
            reduced = cube.reduce("valid_time", "mean")
        assert "t2m" in reduced.variable_names, "spatial t2m should be reduced"
        assert (
            "number" not in reduced.variable_names
        ), "an aux variable spanning the reduced dim must be dropped"
        assert reduced.get_variable("t2m").band_count == 1, "valid_time collapsed to 1"

    def test_unrecognised_grid_demoted_to_aux_warns(self):
        """A 2-D variable with non-standard axes warns when demoted to aux (M6).

        Test scenario:
            ``weird(alpha, beta)`` is a real grid but its axes carry no CF
            metadata and no known x/y names, so it is classified non-spatial and
            carried through untransformed. crop must warn that it is NOT being
            cropped/reprojected, rather than dropping it silently.
        """
        n_t = 4
        lat = np.arange(5.0, 0.0, -1.0)
        lon = np.arange(0.0, 5.0, 1.0)
        ds = xr.Dataset(
            {
                "t2m": (
                    ("valid_time", "latitude", "longitude"),
                    np.ones((n_t, 5, 5), "float32"),
                ),
                "weird": (("alpha", "beta"), np.ones((5, 5), "float32")),
            },
            coords={
                "valid_time": np.arange(n_t),
                "latitude": lat,
                "longitude": lon,
                "alpha": np.arange(5.0),
                "beta": np.arange(5.0),
            },
        )
        ds.latitude.attrs.update(units="degrees_north", standard_name="latitude")
        ds.longitude.attrs.update(units="degrees_east", standard_name="longitude")
        cube = NetCDF.from_xarray(ds)
        with pytest.warns(UserWarning, match="not recognised as spatial"):
            cube.crop(_MASK)

    def test_nonspatial_2d_aux_does_not_warn_demotion(self):
        """A legit non-spatial 2-D aux ``(valid_time, level)`` does not trip the warning (L5).

        Test scenario:
            A ``lut(valid_time, level)`` lookup table has two *recognised*
            non-spatial axes, so it is carried through without the alarming
            "not recognised as spatial" demotion warning (which is reserved for
            variables with >= 2 unrecognised axes, i.e. a likely unmapped grid).
        """
        n_t, n_lev = 4, 3
        lat = np.arange(5.0, 0.0, -1.0)
        lon = np.arange(0.0, 5.0, 1.0)
        ds = xr.Dataset(
            {
                "t2m": (
                    ("valid_time", "latitude", "longitude"),
                    np.ones((n_t, 5, 5), "float32"),
                ),
                "lut": (("valid_time", "level"), np.ones((n_t, n_lev), "float32")),
            },
            coords={
                "valid_time": np.arange(n_t),
                "latitude": lat,
                "longitude": lon,
                "level": np.arange(n_lev),
            },
        )
        ds.latitude.attrs.update(units="degrees_north", standard_name="latitude")
        ds.longitude.attrs.update(units="degrees_east", standard_name="longitude")
        cube = NetCDF.from_xarray(ds)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            cube.crop(_MASK)
        demotion = [r for r in records if "not recognised as spatial" in str(r.message)]
        assert not demotion, (
            f"non-spatial 2-D aux must not trip the demotion warning: "
            f"{[str(r.message) for r in demotion]}"
        )


class TestMultiSpatialPlusAux:
    """A container with two gridded variables + one aux crops all spatial, carries aux."""

    def test_crop_keeps_every_spatial_and_carries_aux(self):
        """Two spatial variables are both cropped; the aux is carried through.

        Test scenario:
            ``t2m`` + ``tp`` (both gridded) + 1-D ``number`` -> the fan-out builds
            a multi-variable result containing all three.
        """
        cube = _era5_like_cube(with_second_spatial=True)
        cropped = cube.crop(mask=_MASK, touch=True)
        for name in ("t2m", "tp", "number"):
            assert name in cropped.variable_names, f"{name} should be in the result"
        assert cropped.get_variable("tp").band_count == 4, "tp keeps its 4 bands"


def _dim(name):
    """A Mock GDAL dimension whose ``GetName()`` returns ``name``."""
    dim = Mock()
    dim.GetName.return_value = name
    return dim


def _attr(name, value):
    """A Mock GDAL attribute exposing ``GetName()`` / ``ReadAsString()``."""
    attr = Mock()
    attr.GetName.return_value = name
    attr.ReadAsString.return_value = value
    return attr


def _fake_rg(var_dims, coord_attrs=None):
    """A Mock root group: ``OpenMDArray(name)`` -> a variable MDArray or a coordinate.

    Args:
        var_dims: ``{var_name: [dim_name, ...]}`` for the variable(s) under test.
        coord_attrs: ``{dim_name: {attr: value}}`` driving CF detection; a name in
            neither map raises ``RuntimeError`` as GDAL does.
    """
    coord_attrs = coord_attrs or {}

    def open_mdarray(name):
        if name in var_dims:
            md = Mock()
            md.GetDimensions.return_value = [_dim(d) for d in var_dims[name]]
            return md
        if name in coord_attrs:
            coord = Mock()
            coord.GetAttributes.return_value = [
                _attr(k, v) for k, v in coord_attrs[name].items()
            ]
            return coord
        raise RuntimeError(f"Array {name} does not exist")

    rg = Mock()
    rg.OpenMDArray.side_effect = open_mdarray
    return rg


class TestVariableIsSpatial:
    """``_variable_is_spatial`` decides griddability from a variable's own dims."""

    def test_one_dim_is_not_spatial(self):
        """A 1-D variable (e.g. ``number(valid_time)``) is non-spatial.

        Test scenario:
            A single-dimension variable cannot form a raster -> ``False``.
        """
        rg = _fake_rg({"number": ["valid_time"]})
        assert NetCDF._variable_is_spatial(NetCDF, rg, "number") is False

    def test_two_non_spatial_dims_is_not_spatial(self):
        """A 2-D variable with no recognised spatial axes is non-spatial.

        Test scenario:
            ``time_bnds(valid_time, nbnds)`` — no CF attrs, no known names -> the
            fan-out must not treat bounds as a raster.
        """
        rg = _fake_rg({"time_bnds": ["valid_time", "nbnds"]})
        assert NetCDF._variable_is_spatial(NetCDF, rg, "time_bnds") is False

    def test_known_name_axes_is_spatial(self):
        """Well-known ``y`` / ``x`` dimension names mark a variable spatial.

        Test scenario:
            ``t2m(valid_time, y, x)`` -> ``True`` via name detection.
        """
        rg = _fake_rg({"t2m": ["valid_time", "y", "x"]})
        assert NetCDF._variable_is_spatial(NetCDF, rg, "t2m") is True

    def test_cf_attr_axes_is_spatial(self):
        """CF ``standard_name`` attributes mark a variable spatial.

        Test scenario:
            ``t2m(time, rows, cols)`` whose ``rows`` / ``cols`` coords carry
            ``latitude`` / ``longitude`` -> ``True`` via CF detection (names alone
            wouldn't match).
        """
        rg = _fake_rg(
            {"t2m": ["time", "rows", "cols"]},
            coord_attrs={
                "time": {"axis": "T"},
                "rows": {"standard_name": "latitude"},
                "cols": {"standard_name": "longitude"},
            },
        )
        assert NetCDF._variable_is_spatial(NetCDF, rg, "t2m") is True

    def test_missing_variable_is_not_spatial(self):
        """A variable whose ``OpenMDArray`` raises is treated as non-spatial.

        Test scenario:
            ``OpenMDArray`` raises ``RuntimeError`` -> ``False`` (no crash).
        """
        rg = _fake_rg({})
        assert NetCDF._variable_is_spatial(NetCDF, rg, "ghost") is False

    def test_none_mdarray_is_not_spatial(self):
        """A ``None`` MDArray is treated as non-spatial.

        Test scenario:
            ``OpenMDArray`` returns ``None`` -> ``False``.
        """
        rg = Mock()
        rg.OpenMDArray.side_effect = lambda name: None
        assert NetCDF._variable_is_spatial(NetCDF, rg, "v") is False


class TestSpatialVariableNames:
    """``_spatial_variable_names`` lists only the gridded variables."""

    def test_lists_only_gridded_variables(self):
        """The 1-D aux is excluded from the spatial variable list.

        Test scenario:
            ``t2m`` + ``number`` container -> ``["t2m"]``.
        """
        cube = _era5_like_cube()
        assert cube._spatial_variable_names() == ["t2m"]

    def test_variable_subset_has_no_root_group(self):
        """A classic-mode variable subset (no root group) yields an empty list.

        Test scenario:
            ``get_variable`` returns a classic dataset with ``GetRootGroup() is
            None`` -> ``[]``.
        """
        cube = _era5_like_cube()
        assert cube.get_variable("t2m")._spatial_variable_names() == []


class TestCarryAuxVariablesWarn:
    """``_carry_aux_variables`` warns (not raises) when a copy fails."""

    def test_warns_when_add_variable_fails(self):
        """A failed ``add_variable`` warns and does not abort the operation.

        Test scenario:
            ``result.add_variable`` raises ``RuntimeError`` -> a ``UserWarning``
            naming the variable, no exception propagated.
        """
        cube = _era5_like_cube()
        result = Mock()
        result.add_variable.side_effect = RuntimeError("boom")
        with pytest.warns(UserWarning, match="could not carry"):
            cube._carry_aux_variables(result, ["number"], "crop")

    def test_aggregates_multiple_failures_into_one_warning(self):
        """Several failed carries produce one warning naming all of them (M4).

        Test scenario:
            Two aux variables fail to copy; a single aggregated ``UserWarning``
            must name both rather than emitting one warning per variable, so the
            data loss is visible at a glance.
        """
        cube = _era5_like_cube()
        result = Mock()
        result.add_variable.side_effect = RuntimeError("boom")
        with pytest.warns(UserWarning, match="could not carry 2 non-spatial") as record:
            cube._carry_aux_variables(result, ["number", "expver"], "crop")
        assert len(record) == 1, f"must emit exactly one warning, got {len(record)}"
        message = str(record[0].message)
        assert "'number'" in message and "'expver'" in message, message
