"""Tests for NetCDF.rename_dims.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from osgeo import gdal

from pyramids.netcdf import ExtraDimensions, GeoReference
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import SEED

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0, 5.0, 0, -1.0)
CF_FIXTURE = "tests/data/netcdf/cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
GROUPS_FIXTURE = "tests/data/netcdf/none__35v__1d35__groups-nc4.nc"


def _string_aux_container():
    """A raw multidimensional store with a string auxiliary variable along `time`.

    `from_array` cannot carry a string auxiliary variable (the kind ERA5 ships as `expver`), and
    the store rebuild has a separate code path for string MDArrays, so the fixture is built
    directly against GDAL's multidim API.
    """
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
    rg = store.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    dim_time = rg.CreateDimension("time", gdal.DIM_TYPE_TEMPORAL, "", 2)
    dim_y = rg.CreateDimension("y", gdal.DIM_TYPE_HORIZONTAL_Y, "NORTH", 2)
    dim_x = rg.CreateDimension("x", gdal.DIM_TYPE_HORIZONTAL_X, "EAST", 2)
    for dim, values in (
        (dim_time, [0.0, 1.0]),
        (dim_y, [1.0, 0.0]),
        (dim_x, [0.0, 1.0]),
    ):
        coord = rg.CreateMDArray(dim.GetName(), [dim], f64)
        coord.Write(np.array(values))
        dim.SetIndexingVariable(coord)
    data = rg.CreateMDArray("t2m", [dim_time, dim_y, dim_x], f64)
    data.Write(np.arange(8.0).reshape(2, 2, 2))
    expver = rg.CreateMDArray(
        "expver", [dim_time], gdal.ExtendedDataType.CreateString()
    )
    expver.Write(["0001", "0005"])
    return NetCDF(store, open_as_multi_dimensional=True)


def _make_nc(var_name="temperature"):
    """A one-variable cube with a single band dimension `time` (length 3)."""
    arr = np.random.default_rng(SEED).random((3, 5, 8)).astype(np.float64)
    return NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=GEO),
        variable_name=var_name,
        dims=ExtraDimensions(name="time", values=[0, 6, 12]),
    )


def _make_2d_nc():
    """A one-variable cube with two band dimensions `time` (2) and `level` (3)."""
    arr = np.random.default_rng(SEED).random((2, 3, 5, 8)).astype(np.float64)
    return NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=GEO),
        variable_name="v",
        dims=ExtraDimensions(dims=[("time", [0, 6]), ("level", [1000, 850, 500])]),
    )


def _make_multi_nc():
    """A container with two variables that both span the band dimension `time`."""
    nc = _make_nc("temp")
    arr2 = np.random.default_rng(99).random((3, 5, 8)).astype(np.float64)
    second = NetCDF.from_array(
        arr=arr2,
        geo_ref=GeoReference(geo=GEO),
        variable_name="pressure",
        dims=ExtraDimensions(name="time", values=[0, 6, 12]),
    )
    nc.set_variable("pressure", second.get_variable("pressure"))
    return nc


class TestRenameDimsHappyPath:
    """A rename changes the label, never the cells."""

    def test_single_variable_renames_the_dimension(self):
        """`rename_dims(time='t')` relabels the band dimension on a single variable."""
        var = _make_nc().get_variable("temperature")
        out = var.rename_dims(time="t")
        assert out._band_dim_names == ("t",), f"got {out._band_dim_names}"

    def test_the_coordinates_follow_the_rename(self):
        """The renamed dimension keeps its coordinate stamps under the new key."""
        var = _make_nc().get_variable("temperature")
        out = var.rename_dims(time="t")
        assert out._band_dim_values_map["t"] == [0, 6, 12], (
            "coordinates must follow the new name"
        )
        assert "time" not in out._band_dim_values_map, "the old key must be gone"

    def test_the_cells_are_unchanged(self):
        """A rename moves no data — the array is identical before and after."""
        var = _make_nc().get_variable("temperature")
        before = np.asarray(var.read_array())
        out = var.rename_dims(time="t")
        assert_allclose(np.asarray(out.read_array()), before)

    def test_a_single_variable_rename_is_cell_free(self):
        """The single-variable path rewraps rather than rebuilds, so it is not marked in-memory."""
        var = _make_nc().get_variable("temperature")
        out = var.rename_dims(time="t")
        assert out._rebuilt_in_memory is False, (
            "a rename must keep the variable's lazy read"
        )

    def test_the_mapping_form_and_kwargs_form_agree(self):
        """`rename_dims({'time': 't'})` equals `rename_dims(time='t')`."""
        var = _make_nc().get_variable("temperature")
        assert (
            var.rename_dims({"time": "t"})._band_dim_names
            == var.rename_dims(time="t")._band_dim_names
        )

    def test_two_dimension_names_can_be_exchanged_in_one_call(self):
        """Exchanging two names in one call relabels them in place — a relabel, not a transpose.

        `rename_dims(time="level", level="time")` swaps the two *labels*; the axes keep their order
        (axis 0 stays axis 0), so the data is not transposed. `transpose` is the axis-reorder op.
        """
        var = _make_2d_nc().get_variable("v")
        out = var.rename_dims(time="level", level="time")
        assert out._band_dim_names == ("level", "time"), f"got {out._band_dim_names}"
        assert out._band_dim_sizes == (2, 3), (
            f"axes keep their order: {out._band_dim_sizes}"
        )

    def test_a_container_renames_every_variable_that_spans_the_dim(self):
        """On a container the renamed dimension changes on every variable that has it."""
        cont = _make_multi_nc()
        out = cont.rename_dims(time="t")
        assert out.get_variable("temp")._band_dim_names == ("t",), (
            "temp must be renamed"
        )
        assert out.get_variable("pressure")._band_dim_names == ("t",), (
            "pressure must be renamed"
        )

    def test_a_container_rename_preserves_cells(self):
        """A container rename carries each variable's cells over unchanged."""
        cont = _make_multi_nc()
        before = np.asarray(cont.get_variable("temp").read_array())
        out = cont.rename_dims(time="t")
        assert_allclose(np.asarray(out.get_variable("temp").read_array()), before)

    def test_a_container_rename_preserves_cf_time_attrs(self):
        """A container rename re-keys the renamed dim's CF time attributes (units/calendar) (H1).

        The single-variable path kept them, but the container path dropped them, because
        `_apply_per_variable` filtered the source's old-keyed attributes by the new names.
        """
        nc = NetCDF.read_file("tests/data/netcdf/cf__12v__1d4-2d5-3d2-4d1__y-asc.nc")
        source = nc.get_variable("pr")._resolved_band_dim_time_attrs().get("time")
        assert source is not None, (
            "precondition: 'pr' carries CF time attributes on 'time'"
        )
        out = nc.rename_dims(time="t")
        carried = out.get_variable("pr")._band_dim_time_attrs.get("t")
        assert carried == source, (
            f"renamed dim must keep its CF time attrs: {carried} != {source}"
        )

    def test_an_empty_rename_is_a_no_op_on_a_container(self):
        """`rename_dims()` with nothing to change returns an equivalent container (L2)."""
        cont = _make_multi_nc()
        out = cont.rename_dims()
        assert set(out.variable_names) == set(cont.variable_names), (
            "variables preserved"
        )
        assert out.get_variable("temp")._band_dim_names == ("time",), "dims unchanged"

    def test_an_identity_rename_on_a_variable_stays_lazy(self):
        """An identity rename on a single variable rewraps (stays lazy), not rebuilds (L2)."""
        var = _make_nc().get_variable("temperature")
        out = var.rename_dims(time="time")
        assert out._rebuilt_in_memory is False, (
            "an identity rename must keep the lazy read"
        )
        assert out._band_dim_names == ("time",), f"got {out._band_dim_names}"

    def test_a_string_auxiliary_variable_survives_a_rename(self):
        """A string auxiliary variable (e.g. ERA5's `expver`) is carried through, values intact.

        The store rebuild copies string MDArrays by a separate path from numeric ones; without
        coverage it was unverified that a rename kept them at all.
        """
        out = _string_aux_container().rename_dims(time="tt")
        assert "tt" in (out.dimension_names or []), (
            f"renamed dim missing: {out.dimension_names}"
        )
        assert "time" not in (out.dimension_names or []), "the old dim must be gone"
        carried = out._raster.GetRootGroup().OpenMDArray("expver")
        assert carried is not None, "the string auxiliary variable must survive"
        assert carried.Read() == ["0001", "0005"], (
            f"values must be intact, got {carried.Read()}"
        )

    def test_a_group_view_renames_its_own_dimension(self):
        """`get_group(...).rename_dims(...)` renames the dim the group's arrays span.

        A group view's arrays span a dimension declared in an ancestor group, which the working
        group does not list, so the rebuild has to resolve the axis from the array itself — it
        used to die with a bare `KeyError` on the dimension name.
        """
        group = NetCDF.read_file(GROUPS_FIXTURE).get_group(
            NetCDF.read_file(GROUPS_FIXTURE).group_names[0]
        )
        before = sorted(group.variable_names)
        out = group.rename_dims({"recNum": "rec"})
        assert "rec" in (out.dimension_names or []), f"got {out.dimension_names}"
        assert "recNum" not in (out.dimension_names or []), "the old dim must be gone"
        assert sorted(out.variable_names) == before, (
            "the group's inventory must be unchanged"
        )


class TestRenameDimsErrors:
    """Every refusal is a ValueError naming the offending input."""

    def test_an_unknown_dimension_is_refused(self):
        """Renaming a name that is not a band dimension raises."""
        var = _make_nc().get_variable("temperature")
        with pytest.raises(ValueError, match="not a band dimension"):
            var.rename_dims(nope="x")

    def test_a_spatial_target_is_refused(self):
        """Renaming a band dimension to a spatial axis name raises."""
        var = _make_nc().get_variable("temperature")
        with pytest.raises(ValueError, match="spatial axis name"):
            var.rename_dims(time="lat")

    def test_two_renames_to_the_same_name_are_refused(self):
        """Two band dimensions renamed to one name raises on the duplicate target."""
        var = _make_2d_nc().get_variable("v")
        with pytest.raises(ValueError, match="duplicate"):
            var.rename_dims(time="z", level="z")

    def test_a_target_that_already_exists_is_refused(self):
        """Renaming onto an existing, non-renamed band dimension raises."""
        var = _make_2d_nc().get_variable("v")
        with pytest.raises(ValueError, match="already names a band dimension"):
            var.rename_dims(time="level")


class TestRenameDimsContainerStoreSurgery:
    """A container rename is store-level surgery on a real CF file (M2, M3).

    The in-memory `from_array` cubes the other tests use carry no auxiliary/bounds variables
    and no orphan-prone store structure, so these cases run against the CF fixture — a file with
    `*_bnds` bounds variables, CF time units on `time`, and pressure units on `plev` — which is
    where the rebuild has to get the store right.
    """

    def test_a_container_rename_removes_the_old_dimension(self):
        """After `rename_dims(time='tt')` the old `time` dimension is gone, not orphaned (M2)."""
        nc = NetCDF.read_file(CF_FIXTURE)
        out = nc.rename_dims(time="tt")
        dims = out.dimension_names or []
        assert "time" not in dims, f"the old dimension must be gone, got {sorted(dims)}"
        assert "tt" in dims, f"the new dimension must be present, got {sorted(dims)}"

    def test_a_container_rename_keeps_the_variable_inventory(self):
        """The rebuild leaves the data-variable inventory unchanged — no bounds surface (M2)."""
        nc = NetCDF.read_file(CF_FIXTURE)
        before = sorted(nc.variable_names)
        out = nc.rename_dims(time="tt")
        assert sorted(out.variable_names) == before, (
            f"inventory changed: {sorted(out.variable_names)} != {before}"
        )

    def test_a_container_rename_carries_bounds_onto_the_new_dim(self):
        """The CF `time_bnds` bounds array follows `time` onto the renamed axis (M2).

        Read off the result's store rather than the enumeration, because a bounds variable is
        deliberately not a data variable; what matters is that it spans `tt`, not the orphaned
        `time`, so the coordinate-bounds relationship is kept.
        """
        nc = NetCDF.read_file(CF_FIXTURE)
        out = nc.rename_dims(time="tt")
        root = out._raster.GetRootGroup()
        assert "time_bnds" in (root.GetMDArrayNames() or []), (
            "bounds array must be kept"
        )
        bnds_dims = [d.GetName() for d in root.OpenMDArray("time_bnds").GetDimensions()]
        assert "tt" in bnds_dims, (
            f"time_bnds must span the renamed axis, got {bnds_dims}"
        )
        assert "time" not in bnds_dims, (
            f"time_bnds must not span the orphan, got {bnds_dims}"
        )

    def test_exchanging_two_dims_leaves_the_non_time_axis_untagged(self, tmp_path):
        """Exchanging `time`<->`plev` must not tag the pressure axis with CF time units (M3).

        The former-`time` axis keeps its `days since ...` units under its new name; the pressure
        axis (coords ~[100000, 92500, 85000]) must carry none, and this must survive a
        `to_file` + reload so a standards-compliant reader never decodes pressures as dates.
        """
        nc = NetCDF.read_file(CF_FIXTURE)
        exchanged = nc.rename_dims(time="plev", plev="time")
        path = tmp_path / "exchanged.nc"
        exchanged.to_file(str(path))
        reloaded = NetCDF.read_file(str(path)).get_variable("ua")
        attrs = reloaded._resolved_band_dim_time_attrs()
        values = reloaded._band_dim_values_map
        pressure_axis = next(
            name
            for name, vals in values.items()
            if vals is not None
            and len(vals)
            and abs(float(np.ravel(vals)[0]) - 100000.0) < 1.0
        )
        assert pressure_axis not in attrs, (
            f"the pressure axis {pressure_axis!r} must carry no time units, got {attrs}"
        )
        assert len(attrs) == 1, (
            f"only the former-time axis may stay time-tagged, got {attrs}"
        )


class TestRenameDimsDiskRoundTrip:
    """The rename survives a write/read cycle."""

    def test_renamed_dimension_survives_a_round_trip(self, tmp_path):
        """A container's renamed dimension is still renamed after saving and reloading.

        Written at the container level: ``to_file`` on a lone variable emits a generic multiband
        raster that carries no named dimension, so the round trip that proves a *rename* persists
        is the container one, where the store records the dimension name.
        """
        cont = _make_multi_nc()
        out = cont.rename_dims(time="t")
        path = tmp_path / "renamed.nc"
        out.to_file(str(path))
        reloaded = NetCDF.read_file(str(path)).get_variable("temp")
        assert "t" in reloaded._band_dim_names, (
            f"expected 't' in {reloaded._band_dim_names}"
        )
