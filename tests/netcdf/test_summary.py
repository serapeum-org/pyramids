"""Tests for ``NetCDF.summary`` and its parity with ``UgridDataset.summary``.

Both classes forward to the shared ``pyramids.base._summary.variable_summary``; these tests pin the
NetCDF wrapper (container + single variable, physical units, no-data) and prove that equivalent data
produces an identical frame on both classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._summary import variable_summary
from pyramids.netcdf import NetCDF
from pyramids.netcdf.ugrid import UgridDataset

DATA = Path(__file__).resolve().parents[1] / "data" / "netcdf"
CF_12V = DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
ONE_VAR = DATA / "none__1v__1d1.nc"


class TestNetCDFSummary:
    def test_container_has_one_row_per_variable(self):
        nc = NetCDF.read_file(str(CF_12V))
        df = nc.summary()
        assert list(df.index) == list(nc.data_vars)
        assert df.index.name == "variable"
        assert list(df.columns) == ["count", "min", "max", "mean", "std"]
        assert df["count"].dtype == np.int64

    def test_values_match_physical_read_array(self):
        nc = NetCDF.read_file(str(CF_12V))
        name = list(nc.data_vars)[0]
        df = nc.summary(variables=[name])
        phys = np.asarray(nc.data_vars[name].read_array(masked=True))
        phys = np.ma.asarray(phys).astype("float64").filled(np.nan).ravel()
        assert df.loc[name, "count"] == int(np.isfinite(phys).sum())
        assert df.loc[name, "mean"] == pytest.approx(np.nanmean(phys))
        assert df.loc[name, "max"] == pytest.approx(np.nanmax(phys))

    def test_single_variable_file(self):
        nc = NetCDF.read_file(str(ONE_VAR))
        df = nc.summary()
        assert len(df) == len(nc.data_vars)
        assert len(df) >= 1

    def test_unknown_variable_raises(self):
        nc = NetCDF.read_file(str(CF_12V))
        with pytest.raises(KeyError):
            nc.summary(variables=["definitely_absent"])

    def test_unknown_metric_raises(self):
        nc = NetCDF.read_file(str(CF_12V))
        with pytest.raises(ValueError, match="unknown summary metric"):
            nc.summary(metrics=("bogus",))

    def test_metric_selection(self):
        nc = NetCDF.read_file(str(CF_12V))
        df = nc.summary(metrics=("mean", "median"))
        assert list(df.columns) == ["mean", "median"]


class TestCrossClassParity:
    def test_same_values_same_frame_on_both_classes(self):
        # A mesh variable and a shared-helper oracle over the identical values must agree, and the
        # NetCDF wrapper over its own read must equal the helper on that read — so both classes
        # reduce to the one shared frame.
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": values.reshape(3, 2)},
            data_locations={"d": "face"},
        )
        mesh["d"].dimensions = ("time", "nMesh2d_face")
        oracle = variable_summary({"d": values})
        mesh_row = mesh.summary().loc["d"]
        for metric in oracle.columns:
            assert mesh_row[metric] == pytest.approx(oracle.loc["d", metric])

        nc = NetCDF.read_file(str(CF_12V))
        name = list(nc.data_vars)[0]
        phys = (
            np.ma.asarray(nc.data_vars[name].read_array(masked=True))
            .astype("float64")
            .filled(np.nan)
        )
        nc_oracle = variable_summary({name: phys})
        nc_row = nc.summary(variables=[name]).loc[name]
        for metric in nc_oracle.columns:
            if np.isnan(nc_oracle.loc[name, metric]):
                assert np.isnan(nc_row[metric])
            else:
                assert nc_row[metric] == pytest.approx(nc_oracle.loc[name, metric])
