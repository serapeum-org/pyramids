"""CF `units` reaches dimension metadata whichever driver opened the file (#1078).

GDAL moves a CF `units` attribute onto the MDArray / indexing-variable **unit slot** and drops
it from the attribute list. That is not driver-specific — netCDF and HDF5 both do it — so
`_read_attributes(indexing_variable)` never contains `units` on either.

What differed was the compensation. `MetadataBuilder._topup_dim_attrs_from_classic` refills
`units`/`calendar` from classic-mode metadata (`<dim>#<attr>` keys), which the netCDF driver
publishes and the HDF5 driver does not. So the time axis decoded locally and came back as `None`
over `/vsicurl` on Windows, where GDAL serves NetCDF-4 through the HDF5 driver.

The fix folds the unit slot back in at the source, so it no longer depends on which driver — or
on the classic top-up — supplied it.

The HDF5 driver is exercised directly, via `gdal.OpenEx(..., allowed_drivers=["HDF5"])`, which
selects it in-process without `GDAL_SKIP` and without a subprocess.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF
from pyramids.netcdf.metadata import MetadataBuilder
from pyramids.netcdf.models import DimensionInfo
from pyramids.netcdf.utils import _merge_unit, _read_attributes

pytestmark = pytest.mark.core

TIME_UNITS = "days since 1950-01-01"
TIME_VALUES = (27996.0, 27997.0, 27998.0)
EXPECTED_DATES = ["2026-08-26", "2026-08-27", "2026-08-28"]


@pytest.fixture
def cf_time_path(tmp_path) -> str:
    """A NetCDF-4 cube whose time axis carries CF `units` and `calendar`.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written file.
    """
    path = str(tmp_path / "cf_time.nc")
    ds = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path, [], ["FORMAT=NC4"])
    rg = ds.GetRootGroup()
    d_time = rg.CreateDimension("time", "", "", len(TIME_VALUES))
    d_lat = rg.CreateDimension("lat", "", "", 2)
    d_lon = rg.CreateDimension("lon", "", "", 2)
    time = rg.CreateMDArray(
        "time", [d_time], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    time.Write(np.array(TIME_VALUES))
    for key, value in (
        ("units", TIME_UNITS),
        ("calendar", "gregorian"),
        ("standard_name", "time"),
        ("axis", "T"),
    ):
        attr = time.CreateAttribute(key, [], gdal.ExtendedDataType.CreateString())
        attr.Write(value)
    lat = rg.CreateMDArray(
        "lat", [d_lat], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lat.Write(np.array([51.0, 51.25]))
    for key, value in (("units", "degrees_north"), ("standard_name", "latitude")):
        attr = lat.CreateAttribute(key, [], gdal.ExtendedDataType.CreateString())
        attr.Write(value)
    lon = rg.CreateMDArray(
        "lon", [d_lon], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    lon.Write(np.array([3.0, 3.25]))
    var = rg.CreateMDArray(
        "twl", [d_time, d_lat, d_lon], gdal.ExtendedDataType.Create(gdal.GDT_Float32)
    )
    var.Write(np.zeros((len(TIME_VALUES), 2, 2), "float32"))
    time = lat = lon = var = d_time = d_lat = d_lon = rg = None
    ds.Close()
    del ds
    gc.collect()
    return path


class TestUnitSlotIsFoldedIn:
    """The unit slot reaches dimension attrs without help from the classic top-up."""

    def test_gdal_drops_units_from_the_attribute_list(self, cf_time_path):
        """The premise: GDAL exposes `units` only through the unit slot.

        Args:
            cf_time_path: The written cube fixture.

        Test scenario:
            Pins why the fold-in is needed at all. If a future GDAL starts listing `units` as an
            ordinary attribute this fails, and the fold-in becomes redundant rather than silently
            load-bearing.
        """
        ds = gdal.OpenEx(cf_time_path, gdal.OF_MULTIDIM_RASTER)
        try:
            array = ds.GetRootGroup().OpenMDArray("time")
            assert "units" not in _read_attributes(array), (
                "GDAL now lists units as an attribute; the fold-in may be redundant"
            )
            assert array.GetUnit() == TIME_UNITS, (
                f"the unit slot must hold the CF units, got {array.GetUnit()!r}"
            )
        finally:
            ds = None

    def test_dimension_attrs_carry_units(self, cf_time_path):
        """`get_dimension("time").attrs` includes the CF units.

        Args:
            cf_time_path: The written cube fixture.

        Test scenario:
            This is what `get_time_variable` reads, and what `get_time_values`' docstring points
            callers at as the documented fallback.

            A contract test, not the regression guard: under the netCDF driver it also passes
            without the fix, because the classic top-up supplies `units` there. It pins the
            end state whichever mechanism provides it;
            `test_time_decodes_without_the_classic_topup` is the one that fails without the
            fold-in.
        """
        nc = NetCDF.read_file(cf_time_path)
        try:
            attrs = nc.meta_data.get_dimension("time").attrs
            assert attrs.get("units") == TIME_UNITS, (
                f"expected the CF units in dimension attrs, got {dict(attrs)}"
            )
        finally:
            nc.close()

    def test_units_survive_the_hdf5_driver(self, cf_time_path):
        """The real HDF5 driver yields a dimension carrying the CF units.

        Args:
            cf_time_path: The written cube fixture.

        Test scenario:
            This is the driver the bug is about — the one GDAL selects for a `/vsi` NetCDF-4 read
            on Windows — reached in-process via `allowed_drivers`. On `main` this dimension comes
            back without `units`, because the classic top-up the netCDF path relies on publishes
            no `<dim>#<attr>` keys under HDF5.
        """
        ds = gdal.OpenEx(
            cf_time_path, gdal.OF_MULTIDIM_RASTER, allowed_drivers=["HDF5"]
        )
        try:
            assert ds.GetDriver().ShortName == "HDF5", (
                f"expected the HDF5 driver, got {ds.GetDriver().ShortName}"
            )
            time_dim = next(
                d for d in ds.GetRootGroup().GetDimensions() if d.GetName() == "time"
            )
            attrs = DimensionInfo.from_gdal_dim(time_dim, "/").attrs
            assert attrs.get("units") == TIME_UNITS, (
                f"the HDF5 driver must still yield the CF units, got {dict(attrs)}"
            )
        finally:
            ds = None

    def test_time_decodes_without_the_classic_topup(self, cf_time_path, monkeypatch):
        """The time axis decodes end to end when classic metadata supplies nothing.

        Args:
            cf_time_path: The written cube fixture.
            monkeypatch: Used to empty the classic top-up.

        Test scenario:
            Covers the user-facing symptom — `get_time_variable()` returning `None` — which
            `NetCDF.read_file` cannot reach through the HDF5 driver directly, since it does not
            expose `allowed_drivers`. An empty classic top-up is the state that driver leaves
            behind, so this reproduces the same condition one layer up.
        """
        monkeypatch.setattr(
            MetadataBuilder, "_read_classic_metadata_for_topup", lambda self: {}
        )
        nc = NetCDF.read_file(cf_time_path)
        try:
            attrs = nc.meta_data.get_dimension("time").attrs
            assert attrs.get("units") == TIME_UNITS, (
                "units must not depend on the classic top-up, which the HDF5 driver leaves empty"
            )
            assert nc.get_time_variable() == EXPECTED_DATES, (
                f"expected decoded dates, got {nc.get_time_variable()}"
            )
        finally:
            nc.close()


class TestNonTemporalDimension:
    """Giving every dimension its units must not turn a wrong query into an exception."""

    def test_spatial_dimension_returns_none(self, cf_time_path):
        """`get_time_variable("lat")` returns None rather than raising.

        Args:
            cf_time_path: The written cube fixture.

        Test scenario:
            The documented contract is `None` when the dimension has no usable `units`. Before
            #1078 a spatial dimension had no `units` at all under the HDF5 driver, so it returned
            `None` by accident. Now it carries `degrees_north`, which reaches the CF time parse —
            so the parse must be guarded, or a wrong-dimension query becomes
            `ValueError: Unrecognized time units`.
        """
        nc = NetCDF.read_file(cf_time_path)
        try:
            assert nc.meta_data.get_dimension("lat").attrs.get("units") == (
                "degrees_north"
            ), (
                "the fixture's lat must carry non-temporal units for this to mean anything"
            )
            assert nc.get_time_variable("lat") is None, (
                "a non-temporal dimension must return None, not raise"
            )
        finally:
            nc.close()


class TestMergeUnit:
    """`_merge_unit` contract — it must never overwrite or invent a CF value."""

    def test_existing_units_wins(self):
        """An attribute the file really carries is not replaced by the unit slot.

        Test scenario:
            A file may declare both; the attribute is the authoritative CF value.
        """

        class _Obj:
            @staticmethod
            def GetUnit():
                return "from the slot"

        attrs = {"units": "from the attribute"}
        assert _merge_unit(attrs, _Obj())["units"] == "from the attribute"

    @pytest.mark.parametrize(
        "unit", ["", None, 42, object()], ids=["empty", "none", "int", "object"]
    )
    def test_non_string_unit_is_ignored(self, unit):
        """Only a real non-empty string becomes a CF attribute.

        Args:
            unit: The value the stand-in's `GetUnit` returns.

        Test scenario:
            `GetUnit`'s contract is `str`, and every value in a CF attrs dict is one. An object
            that merely *has* the method — a partially-built handle, a test stand-in — must not
            put a non-CF value into metadata.
        """

        class _Obj:
            @staticmethod
            def GetUnit():
                return unit

        assert "units" not in _merge_unit({}, _Obj())

    def test_failing_get_unit_is_survivable(self):
        """A raising `GetUnit` leaves the attributes untouched.

        Test scenario:
            Metadata reads elsewhere degrade rather than raise; a unit slot that errors must not
            take the whole dimension read down with it.
        """

        class _Obj:
            @staticmethod
            def GetUnit():
                raise RuntimeError("no unit")

        assert _merge_unit({"axis": "T"}, _Obj()) == {"axis": "T"}
