"""Tests for CF convention write support (CF-1).

Tests the new ``cf.py`` module functions and the CF global attribute
parameters added to ``NetCDF.create_from_array`` and
``_create_netcdf_from_array``.

Covers:
    - ``write_attributes_to_md_array``: str/int/float/list attributes
    - ``write_global_attributes``: group-level attribute writing
    - ``create_from_array`` default Conventions attribute
    - ``create_from_array`` with title/institution/source/history
    - Round-trip: create -> to_file -> read_file -> check global_attributes
"""

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf.cf import write_attributes_to_md_array, write_global_attributes
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

GEO = (30.0, 0.5, 0, 35.0, 0, -0.5)
SEED = 42


def _make_mem_md_array():
    """Create an in-memory MEM multidimensional dataset with one MDArray.

    Returns:
        Tuple of (gdal.Dataset, gdal.Group, gdal.MDArray). The dataset
        must be kept alive as long as the group/array are accessed.
    """
    ds = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
    rg = ds.GetRootGroup()
    dim = rg.CreateDimension("x", None, None, 5)
    dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    md_arr = rg.CreateMDArray("data", [dim], dtype)
    md_arr.Write(np.arange(5, dtype=np.float64))
    return ds, rg, md_arr


class TestWriteAttributesToMdArray:
    """Tests for cf.write_attributes_to_md_array."""

    def test_string_attribute(self):
        """Write a string attribute and read it back.

        Test scenario:
            A string attribute should be stored and retrievable via
            GDAL GetAttributes.
        """
        ds, _, md_arr = _make_mem_md_array()
        write_attributes_to_md_array(md_arr, {"units": "K"})
        attrs = {a.GetName(): a.Read() for a in md_arr.GetAttributes()}
        assert "units" in attrs, f"Expected 'units' in attrs, got {list(attrs.keys())}"
        assert attrs["units"] == "K", f"Expected 'K', got {attrs['units']!r}"

    def test_float_attribute(self):
        """Write a float attribute and read it back.

        Test scenario:
            A float attribute should be stored with float64 type.
        """
        ds, _, md_arr = _make_mem_md_array()
        write_attributes_to_md_array(md_arr, {"scale_factor": 0.01})
        attrs = {a.GetName(): a.Read() for a in md_arr.GetAttributes()}
        assert (
            "scale_factor" in attrs
        ), f"Expected 'scale_factor' in attrs, got {list(attrs.keys())}"
        assert (
            abs(attrs["scale_factor"] - 0.01) < 1e-12
        ), f"Expected 0.01, got {attrs['scale_factor']}"

    def test_int_attribute(self):
        """Write an integer attribute and read it back.

        Test scenario:
            An int attribute should be stored with Int32 type.
        """
        ds, _, md_arr = _make_mem_md_array()
        write_attributes_to_md_array(md_arr, {"start_index": 1})
        attrs = {a.GetName(): a.Read() for a in md_arr.GetAttributes()}
        assert (
            "start_index" in attrs
        ), f"Expected 'start_index' in attrs, got {list(attrs.keys())}"
        assert attrs["start_index"] == 1, f"Expected 1, got {attrs['start_index']}"

    def test_multiple_attributes(self):
        """Write multiple attributes of different types at once.

        Test scenario:
            A mixed dict of str, int, float should all be stored.
        """
        ds, _, md_arr = _make_mem_md_array()
        write_attributes_to_md_array(
            md_arr,
            {
                "standard_name": "air_temperature",
                "add_offset": 273.15,
                "flag_count": 3,
            },
        )
        attrs = {a.GetName(): a.Read() for a in md_arr.GetAttributes()}
        assert (
            attrs["standard_name"] == "air_temperature"
        ), f"Expected 'air_temperature', got {attrs['standard_name']!r}"
        assert (
            abs(attrs["add_offset"] - 273.15) < 1e-10
        ), f"Expected 273.15, got {attrs['add_offset']}"
        assert attrs["flag_count"] == 3, f"Expected 3, got {attrs['flag_count']}"

    def test_list_of_floats_attribute(self):
        """Write a list-of-floats attribute.

        Test scenario:
            A list of numeric values should be written as a
            multi-element float64 attribute.
        """
        ds, _, md_arr = _make_mem_md_array()
        write_attributes_to_md_array(md_arr, {"standard_parallel": [30.0, 60.0]})
        attrs = {a.GetName(): a.Read() for a in md_arr.GetAttributes()}
        assert (
            "standard_parallel" in attrs
        ), f"Expected 'standard_parallel' in attrs, got {list(attrs.keys())}"
        val = attrs["standard_parallel"]
        assert len(val) == 2, f"Expected 2 elements, got {len(val)}"

    def test_empty_dict_is_noop(self):
        """Writing an empty dict should not raise or add attributes.

        Test scenario:
            An empty attrs dict should be a no-op.
        """
        ds, _, md_arr = _make_mem_md_array()
        write_attributes_to_md_array(md_arr, {})
        attrs = list(md_arr.GetAttributes())
        assert len(attrs) == 0, f"Expected 0 attributes, got {len(attrs)}"

    def test_none_value_converted_to_string(self):
        """Writing a None value should convert it to string 'None'.

        Test scenario:
            Non-standard types fall through to the else branch and
            are converted via str().
        """
        ds, _, md_arr = _make_mem_md_array()
        write_attributes_to_md_array(md_arr, {"comment": None})
        attrs = {a.GetName(): a.Read() for a in md_arr.GetAttributes()}
        assert (
            "comment" in attrs
        ), f"Expected 'comment' in attrs, got {list(attrs.keys())}"
        assert attrs["comment"] == "None", f"Expected 'None', got {attrs['comment']!r}"


class TestWriteGlobalAttributes:
    """Tests for cf.write_global_attributes."""

    def test_string_global_attribute(self):
        """Write a string attribute to a root group.

        Test scenario:
            A string attribute on the root group should be readable
            via GetAttributes.
        """
        ds = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = ds.GetRootGroup()
        write_global_attributes(rg, {"Conventions": "CF-1.8"})
        attrs = {a.GetName(): a.Read() for a in rg.GetAttributes()}
        assert (
            attrs["Conventions"] == "CF-1.8"
        ), f"Expected 'CF-1.8', got {attrs.get('Conventions')!r}"

    def test_multiple_global_attributes(self):
        """Write multiple global attributes of mixed types.

        Test scenario:
            Multiple attributes (str, float, int) should all be
            stored on the root group.
        """
        ds = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = ds.GetRootGroup()
        write_global_attributes(
            rg,
            {
                "title": "Test dataset",
                "version": 2.0,
                "count": 10,
            },
        )
        attrs = {a.GetName(): a.Read() for a in rg.GetAttributes()}
        assert (
            attrs["title"] == "Test dataset"
        ), f"Expected 'Test dataset', got {attrs['title']!r}"
        assert (
            abs(attrs["version"] - 2.0) < 1e-10
        ), f"Expected 2.0, got {attrs['version']}"
        assert attrs["count"] == 10, f"Expected 10, got {attrs['count']}"

    def test_empty_dict_is_noop(self):
        """Writing an empty dict should be a no-op.

        Test scenario:
            No attributes should be added.
        """
        ds = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = ds.GetRootGroup()
        write_global_attributes(rg, {})
        attrs = list(rg.GetAttributes())
        assert len(attrs) == 0, f"Expected 0 attributes, got {len(attrs)}"


class TestCreateFromArrayCFGlobalAttributes:
    """Tests for CF global attributes in NetCDF.create_from_array."""

    def test_default_conventions_attribute(self):
        """create_from_array with no CF params sets Conventions=CF-1.8.

        Test scenario:
            The default behavior should always include the Conventions
            attribute on the root group.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO, variable_name="temp")
        ga = nc.global_attributes
        assert (
            "Conventions" in ga
        ), f"Expected 'Conventions' in global attributes, got {list(ga.keys())}"
        assert (
            ga["Conventions"] == "CF-1.8"
        ), f"Expected 'CF-1.8', got {ga['Conventions']!r}"

    def test_title_attribute(self):
        """create_from_array with title param stores it in global attrs.

        Test scenario:
            Passing title should add it to the global attributes.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
            title="My dataset",
        )
        ga = nc.global_attributes
        assert (
            ga.get("title") == "My dataset"
        ), f"Expected 'My dataset', got {ga.get('title')!r}"

    def test_institution_attribute(self):
        """create_from_array with institution param stores it.

        Test scenario:
            Passing institution should add it to global attributes.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
            institution="Deltares",
        )
        ga = nc.global_attributes
        assert (
            ga.get("institution") == "Deltares"
        ), f"Expected 'Deltares', got {ga.get('institution')!r}"

    def test_source_attribute(self):
        """create_from_array with source param stores it.

        Test scenario:
            Passing source should add it to global attributes.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
            source="Model v1.0",
        )
        ga = nc.global_attributes
        assert (
            ga.get("source") == "Model v1.0"
        ), f"Expected 'Model v1.0', got {ga.get('source')!r}"

    def test_history_attribute(self):
        """create_from_array with history param stores it.

        Test scenario:
            Passing history should add it to global attributes.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
            history="Created by test",
        )
        ga = nc.global_attributes
        assert (
            ga.get("history") == "Created by test"
        ), f"Expected 'Created by test', got {ga.get('history')!r}"

    def test_all_cf_global_attributes(self):
        """create_from_array with all CF params stores them all.

        Test scenario:
            Passing all four optional params should produce all
            four attributes plus Conventions.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
            title="Full test",
            institution="Test Lab",
            source="Unit test",
            history="Created in test",
        )
        ga = nc.global_attributes
        assert (
            ga["Conventions"] == "CF-1.8"
        ), f"Expected 'CF-1.8', got {ga.get('Conventions')!r}"
        assert (
            ga["title"] == "Full test"
        ), f"Expected 'Full test', got {ga.get('title')!r}"
        assert (
            ga["institution"] == "Test Lab"
        ), f"Expected 'Test Lab', got {ga.get('institution')!r}"
        assert (
            ga["source"] == "Unit test"
        ), f"Expected 'Unit test', got {ga.get('source')!r}"
        assert (
            ga["history"] == "Created in test"
        ), f"Expected 'Created in test', got {ga.get('history')!r}"

    def test_none_params_excluded(self):
        """create_from_array with None CF params omits them.

        Test scenario:
            When title/institution/source/history are None (default),
            only Conventions should be present.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO, variable_name="temp")
        ga = nc.global_attributes
        assert "Conventions" in ga, "Conventions must always be present"
        assert (
            "title" not in ga
        ), f"title should not be present, got {ga.get('title')!r}"
        assert (
            "institution" not in ga
        ), f"institution should not be present, got {ga.get('institution')!r}"
        assert (
            "source" not in ga
        ), f"source should not be present, got {ga.get('source')!r}"
        assert (
            "history" not in ga
        ), f"history should not be present, got {ga.get('history')!r}"

    def test_3d_array_has_conventions(self):
        """create_from_array with 3D array also gets Conventions.

        Test scenario:
            3D arrays (with extra dimension) should also get CF
            global attributes.
        """
        arr = np.random.RandomState(SEED).rand(3, 5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="precip",
            extra_dim_name="time",
            title="3D test",
        )
        ga = nc.global_attributes
        assert (
            ga["Conventions"] == "CF-1.8"
        ), f"Expected 'CF-1.8', got {ga.get('Conventions')!r}"
        assert ga["title"] == "3D test", f"Expected '3D test', got {ga.get('title')!r}"

    def test_backward_compatible_no_params(self):
        """Existing calls without CF params should still work.

        Test scenario:
            Calling create_from_array with only the original
            parameters should produce a valid NetCDF with data
            preserved.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="data",
        )
        var = nc.get_variable("data")
        result = var.read_array()
        np.testing.assert_allclose(
            result[0] if result.ndim == 3 else result,
            arr,
            atol=1e-10,
            err_msg="Data should be preserved in backward-compatible call",
        )


class TestCFGlobalAttributesRoundTrip:
    """Round-trip tests: create -> to_file -> read_file -> verify."""

    def test_round_trip_conventions(self, tmp_path):
        """Conventions attribute survives write-to-disk and reload.

        Test scenario:
            Create in-memory, write to .nc, read back, check
            Conventions is still CF-1.8.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
        )
        out_path = str(tmp_path / "test_conventions.nc")
        nc.to_file(out_path)
        nc2 = NetCDF.read_file(out_path)
        ga = nc2.global_attributes
        assert (
            "Conventions" in ga
        ), f"Conventions lost after round-trip, got {list(ga.keys())}"
        assert (
            ga["Conventions"] == "CF-1.8"
        ), f"Expected 'CF-1.8' after round-trip, got {ga['Conventions']!r}"

    def test_round_trip_all_cf_attrs(self, tmp_path):
        """All CF global attributes survive the round-trip.

        Test scenario:
            Create with all four CF params, write to disk,
            read back, verify all are preserved.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
            title="Round-trip test",
            institution="Test Institute",
            source="pytest",
            history="Created for testing",
        )
        out_path = str(tmp_path / "test_all_attrs.nc")
        nc.to_file(out_path)
        nc2 = NetCDF.read_file(out_path)
        ga = nc2.global_attributes
        assert (
            ga.get("title") == "Round-trip test"
        ), f"title lost after round-trip, got {ga.get('title')!r}"
        assert (
            ga.get("institution") == "Test Institute"
        ), f"institution lost after round-trip, got {ga.get('institution')!r}"
        assert (
            ga.get("source") == "pytest"
        ), f"source lost after round-trip, got {ga.get('source')!r}"
        assert (
            ga.get("history") == "Created for testing"
        ), f"history lost after round-trip, got {ga.get('history')!r}"

    def test_round_trip_with_disk_path(self, tmp_path):
        """create_from_array with path= writes CF attrs directly.

        Test scenario:
            Creating directly to disk (path= parameter) should also
            set CF global attributes. We verify by reading the global
            attributes from the in-memory handle returned by
            create_from_array (the netCDF driver writes to disk during
            creation, and the returned handle is backed by that file).
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        out_path = str(tmp_path / "direct_write.nc")
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
            path=out_path,
            title="Direct write",
        )
        ga = nc.global_attributes
        assert (
            ga.get("Conventions") == "CF-1.8"
        ), f"Conventions missing from direct write, got {ga.get('Conventions')!r}"
        assert (
            ga.get("title") == "Direct write"
        ), f"title missing from direct write, got {ga.get('title')!r}"
