"""Tests for which NetCDF dimensions may veto the CRS inference (ARC-26).

Separate from `test_crs_less_semantics.py` because writing the fixtures needs
the optional interop engine: that module is `core` and must import in an
extras-free wheel run, so it cannot carry a module-level optional-dependency
import.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.crs import sr_from_epsg
from pyramids.netcdf import NetCDF

xr = pytest.importorskip("xarray")

pytestmark = [pytest.mark.interop]


class TestNetCDFAxisClassification:
    """Tests for which NetCDF dimensions may veto the geographic inference.

    GDAL *guesses* a dimension is `VERTICAL` from metre units alone, so trusting
    that type discards exactly the metre axes the veto is built on (round-5 H1).
    """

    @staticmethod
    def _write(path, coords, dim_order, extra=None):
        """Write a NetCDF with the given coordinate attributes and dimension order."""
        sizes = {
            "x": 4,
            "y": 4,
            "lon": 4,
            "lat": 4,
            "deptht": 3,
            "olevel": 3,
            "nav_lev": 3,
            "lev": 3,
            "time": 2,
        }
        data_vars = {
            "v": (
                tuple(dim_order),
                np.ones(tuple(sizes[d] for d in dim_order), "float32"),
            )
        }
        if extra:
            data_vars.update(extra)
        xr.Dataset(
            data_vars,
            coords={
                name: (name, np.arange(sizes[name], dtype="f8") + 1.0, attrs)
                for name, attrs in coords.items()
            },
        ).to_netcdf(path)
        return str(path)

    METRES = {"units": "m"}
    EAST = {"units": "degrees_east"}
    NORTH = {"units": "degrees_north"}

    @staticmethod
    def _write_utm_with_aux_lonlat(path):
        """Write a UTM NetCDF whose only degrees are 2-D auxiliary coordinates.

        The auxiliary arrays are named through `coordinates`, which is what makes
        them evidence. Without them the classic reader answers "no CRS" from the
        metadata alone and the multidim rule under test is never reached.
        """
        xr.Dataset(
            {
                "sst": (
                    ("y", "x"),
                    np.ones((4, 4), "float32"),
                    {"coordinates": "lon2d lat2d"},
                ),
                "lon2d": (("y", "x"), np.full((4, 4), 5.0), {"units": "degrees_east"}),
                "lat2d": (
                    ("y", "x"),
                    np.full((4, 4), 45.0),
                    {"units": "degrees_north"},
                ),
            },
            coords={
                "x": (
                    "x",
                    np.array([400000.0, 400030.0, 400060.0, 400090.0]),
                    {"units": "m"},
                ),
                "y": (
                    "y",
                    np.array([5000090.0, 5000060.0, 5000030.0, 5000000.0]),
                    {"units": "m"},
                ),
            },
        ).to_netcdf(str(path))
        return str(path)

    def test_undeclared_metre_axes_still_veto(self, tmp_path):
        """A UTM grid whose x/y declare only `units = "m"` is not WGS 84.

        Test scenario:
            GDAL types both dimensions `VERTICAL` because their unit is metres,
            so trusting that type reads the grid as EPSG:4326 (round-5 H1).
        """
        path = self._write_utm_with_aux_lonlat(tmp_path / "utm.nc")
        assert NetCDF.read_file(path).epsg is None, (
            "a metre grid must not read as WGS 84"
        )

    @pytest.mark.parametrize("vertical", ["deptht", "olevel", "nav_lev"])
    def test_a_vertical_axis_in_metres_does_not_veto(self, tmp_path, vertical):
        """A depth axis in metres says nothing about the horizontal frame.

        Args:
            tmp_path: pytest temporary directory.
            vertical: Model-specific vertical-axis name under test.

        Test scenario:
            Names no allow-list covers, on a grid whose horizontal axes are
            degrees (round-4 M2).
        """
        path = self._write(
            tmp_path / f"{vertical}.nc",
            {"lon": self.EAST, "lat": self.NORTH, vertical: self.METRES},
            [vertical, "lat", "lon"],
        )
        assert NetCDF.read_file(path).epsg == 4326, f"{vertical} must not strip the CRS"

    def test_a_vertical_axis_between_the_horizontal_ones(self, tmp_path):
        """The vertical axis need not be the outermost dimension.

        Test scenario:
            The `(lat, lev, lon)` layout of a real repo fixture, which refutes
            any rule keyed on dimension position.
        """
        path = self._write(
            tmp_path / "interleaved.nc",
            {"lon": self.EAST, "lat": self.NORTH, "lev": self.METRES},
            ["lat", "lev", "lon"],
        )
        assert NetCDF.read_file(path).epsg == 4326, "an interleaved lev must not veto"

    def test_degrees_on_data_variables_are_not_evidence(self, tmp_path):
        """A wind direction in degrees does not make a UTM NetCDF geographic.

        Test scenario:
            The multidim reader took every array's unit as evidence, so a NetCDF
            and the byte-equivalent GeoTIFF disagreed (round-5 M2).
        """
        path = str(tmp_path / "winddir.nc")
        xr.Dataset(
            {
                "winddir": (
                    ("y", "x"),
                    np.ones((4, 4), "float32"),
                    {"units": "degrees_east"},
                ),
                "sunazi": (
                    ("y", "x"),
                    np.ones((4, 4), "float32"),
                    {"units": "degrees_north"},
                ),
            },
            coords={
                # No units on the coordinates, so nothing vetoes: the only
                # degrees in the file are on the data variables, and whether
                # they count as evidence is exactly what this pins.
                "x": ("x", np.array([400000.0, 400030.0, 400060.0, 400090.0])),
                "y": ("y", np.array([5000090.0, 5000060.0, 5000030.0, 5000000.0])),
            },
        ).to_netcdf(path)
        assert NetCDF.read_file(path).epsg is None, (
            "a data variable's units are not evidence"
        )


class TestGlobalAttributeProvenance:
    """Tests for which NetCDF root attributes may define the dataset's CRS.

    The geobox writer always emits `GeoTransform` beside `crs_wkt` / `epsg`, so
    that companion is what distinguishes our own file from a third-party store
    that happens to carry an attribute called `epsg` (round-5 M5).
    """

    @staticmethod
    def _write(path, attrs):
        """Write a UTM-gridded NetCDF carrying the given root attributes."""
        dataset = xr.Dataset(
            {"v": (("y", "x"), np.ones((4, 4), "float32"))},
            coords={
                "x": ("x", np.array([400000.0, 400030.0, 400060.0, 400090.0])),
                "y": ("y", np.array([5000090.0, 5000060.0, 5000030.0, 5000000.0])),
            },
        )
        dataset.attrs.update(attrs)
        dataset.to_netcdf(str(path))
        return str(path)

    def test_an_epsg_attribute_alone_is_not_adopted(self, tmp_path):
        """A root `epsg` with no `GeoTransform` companion defines no CRS.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            A third-party store tagging itself `epsg = 32636` — a processing
            note, a source-data tag — must not become the dataset's CRS.
        """
        path = self._write(tmp_path / "foreign.nc", {"epsg": 32636})
        assert NetCDF.read_file(path).epsg is None, "a stray attribute is not a CRS"

    def test_an_epsg_attribute_beside_a_geotransform_is_adopted(self, tmp_path):
        """The pair the geobox writer emits together is adopted.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            The positive counterpart, so the gate cannot pass by refusing
            everything.
        """
        path = self._write(
            tmp_path / "ours.nc",
            {"epsg": 32636, "GeoTransform": "399985 30 0 5000075 0 -30"},
        )
        assert NetCDF.read_file(path).epsg == 32636, "our own pair must be adopted"

    def test_a_crs_wkt_attribute_alone_is_not_adopted(self, tmp_path):
        """The same rule applies to `crs_wkt`.

        Args:
            tmp_path: pytest temporary directory.
        """
        path = self._write(
            tmp_path / "wkt_only.nc", {"crs_wkt": sr_from_epsg(32636).ExportToWkt()}
        )
        assert NetCDF.read_file(path).epsg is None, "a stray crs_wkt is not a CRS"
