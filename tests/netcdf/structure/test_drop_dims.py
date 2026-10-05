"""Tests for NetCDF.drop_dims (container-only; removes variables spanning a dimension).

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import ExtraDimensions, GeoReference
from pyramids.netcdf.netcdf import Container, NetCDF
from tests.netcdf.conftest import SEED

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0, 5.0, 0, -1.0)


def _time_var(name, length=3):
    """A one-variable cube with a `time` band dimension of the given length."""
    arr = np.random.default_rng(SEED).random((length, 5, 8)).astype(np.float64)
    return NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=GEO),
        variable_name=name,
        dims=ExtraDimensions(name="time", values=list(range(length))),
    )


def _level_var(name, length=2):
    """A one-variable cube with a `level` band dimension of the given length."""
    arr = np.random.default_rng(7).random((length, 5, 8)).astype(np.float64)
    return NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=GEO),
        variable_name=name,
        dims=ExtraDimensions(name="level", values=list(range(length))),
    )


def _mixed_container():
    """A container with `temp` along `time` and `geoid` along `level`."""
    cont = _time_var("temp")
    cont.set_variable("geoid", _level_var("geoid").get_variable("geoid"))
    return cont


def _packed_survivor_container():
    """A container with a CF-packed `int16` survivor along `level` and a float var along `time`.

    Built as a raw multidimensional store because `from_array` carries no packing: `drop_dims`
    must keep the survivor's stored dtype (`int16`), scale/offset and no-data sentinel exactly,
    which a `merge`-style rebuild (reading the physical, unpacked array) silently widened (M1).
    """
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
    rg = store.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    dim_time = rg.CreateDimension("time", gdal.DIM_TYPE_TEMPORAL, "", 3)
    dim_level = rg.CreateDimension("level", "", "", 2)
    dim_y = rg.CreateDimension("y", gdal.DIM_TYPE_HORIZONTAL_Y, "NORTH", 2)
    dim_x = rg.CreateDimension("x", gdal.DIM_TYPE_HORIZONTAL_X, "EAST", 2)
    for dim, values in (
        (dim_time, [0.0, 1.0, 2.0]),
        (dim_level, [850.0, 500.0]),
        (dim_y, [1.0, 0.0]),
        (dim_x, [0.0, 1.0]),
    ):
        coord = rg.CreateMDArray(dim.GetName(), [dim], f64)
        coord.Write(np.array(values))
        dim.SetIndexingVariable(coord)
    packed = rg.CreateMDArray(
        "packed",
        [dim_level, dim_y, dim_x],
        gdal.ExtendedDataType.Create(gdal.GDT_Int16),
    )
    packed.SetScale(0.01)
    packed.SetOffset(0.0)
    packed.SetNoDataValueDouble(-999)
    packed.Write(np.arange(2 * 2 * 2, dtype="int16").reshape(2, 2, 2))
    on_time = rg.CreateMDArray(
        "tempvar",
        [dim_time, dim_y, dim_x],
        gdal.ExtendedDataType.Create(gdal.GDT_Float32),
    )
    on_time.Write(np.zeros((3, 2, 2), dtype="float32"))
    return Container(store)


class TestDropDimsHappyPath:
    """Dropping a dimension removes the variables defined along it, and only those."""

    def test_the_variable_along_the_dropped_dim_is_gone(self):
        """`drop_dims('time')` removes the variable that spans `time`."""
        out = _mixed_container().drop_dims("time")
        assert "temp" not in out.variable_names, "temp spans time and must be dropped"

    def test_a_variable_on_another_dim_survives(self):
        """A variable spanning a different dimension is untouched."""
        out = _mixed_container().drop_dims("time")
        assert "geoid" in out.variable_names, "geoid spans level and must survive"

    def test_the_dropped_dimension_is_gone_from_dims(self):
        """The dimension itself is removed, not just the variables along it (M2).

        Rebuilding from the survivors drops the orphan dimension that in-place removal left
        declared in the store, matching the method name and xarray's `drop_dims`.
        """
        out = _mixed_container().drop_dims("time")
        assert "time" not in (out.dimension_names or []), (
            f"'time' orphaned in {out.dimension_names}"
        )
        assert "level" in (out.dimension_names or []), (
            "the surviving variable's dim must remain"
        )

    def test_drop_dims_does_not_mutate_the_receiver(self):
        """The original container keeps its variable; the drop happens on a copy."""
        cont = _mixed_container()
        cont.drop_dims("time")
        assert "temp" in cont.variable_names, "drop_dims must not mutate the receiver"

    def test_multiple_dims_can_be_dropped(self):
        """Dropping both dimensions empties the container of data variables."""
        out = _mixed_container().drop_dims(["time", "level"])
        assert out.variable_names == [], (
            f"expected no variables, got {out.variable_names}"
        )


class TestDropDimsPreservesPacking:
    """Dropping a dimension keeps each survivor's stored dtype, packing and no-data (M1)."""

    def test_a_packed_survivor_keeps_its_dtype_scale_and_no_data(self):
        """An `int16`/scale-0.01/no-data-(-999) survivor is unchanged after `drop_dims('time')`.

        The result's store is inspected directly: the survivor must still be stored as `int16`
        with the same scale and no-data sentinel, not widened to `float64`/scale-1.0 the way a
        rebuild that reads the physical array would leave it.
        """
        out = _packed_survivor_container().drop_dims("time")
        assert "packed" in out.variable_names, "the survivor must be kept"
        assert "time" not in (out.dimension_names or []), (
            "the dropped dimension must be gone"
        )
        survivor = out._raster.GetRootGroup().OpenMDArray("packed")
        dtype = gdal.GetDataTypeName(survivor.GetDataType().GetNumericDataType())
        assert dtype == "Int16", f"stored dtype must stay int16, got {dtype}"
        assert survivor.GetScale() == 0.01, (
            f"scale must be kept, got {survivor.GetScale()}"
        )
        assert survivor.GetNoDataValue() == -999, (
            f"no-data sentinel must be kept, got {survivor.GetNoDataValue()}"
        )


class TestDropDimsErrors:
    """Refusals: a single variable, an unknown dimension, a bad errors flag."""

    def test_a_single_variable_is_refused(self):
        """Called on a lone variable, drop_dims redirects to the container/remove_variable."""
        var = _time_var("temp").get_variable("temp")
        with pytest.raises(ValueError, match="removes whole variables"):
            var.drop_dims("time")

    def test_an_unknown_dimension_raises_by_default(self):
        """`errors='raise'` (default) refuses a dimension the container does not have."""
        with pytest.raises(ValueError, match="not a dimension of this container"):
            _mixed_container().drop_dims("season")

    def test_an_unknown_dimension_is_skipped_when_ignored(self):
        """`errors='ignore'` skips an unknown dimension and drops nothing for it."""
        out = _mixed_container().drop_dims("season", errors="ignore")
        assert set(out.variable_names) == {"temp", "geoid"}, "nothing should be dropped"

    def test_a_bad_errors_flag_raises(self):
        """An errors flag other than raise/ignore is refused."""
        with pytest.raises(ValueError, match="errors must be"):
            _mixed_container().drop_dims("time", errors="warn")
