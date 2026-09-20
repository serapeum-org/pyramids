"""`to_dataframe` without xarray — the index, the sentinel and the refusals.

`NetCDF.to_dataframe` builds its frame with pandas, which is a core dependency, so the
cases that do not need xarray to state an expectation belong in the extras-free suite. The
frame-for-frame comparison against `xarray.Dataset.to_dataframe` lives beside it in
`tests/netcdf/parity/test_to_dataframe_matches_xarray.py` and carries the `interop` marker.

What is pinned here is the part the parity test cannot reach: the `MultiIndex` a dimension
without coordinates falls back to, a sentinel that is already NaN, and the refusal a
container with no gridded variable answers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
STAMPS = [0.0, 6.0]
CELLS = np.arange(8.0).reshape(2, 2, 2)

# ROMS-shaped: `salt` and `zeta` are enumerated as data variables, and neither carries a
# recognised (y, x) pair, so the store has variables and no gridded one.
NO_GRIDDED_VARIABLE = (
    Path(__file__).parents[1]
    / "data"
    / "netcdf"
    / "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc"
)


def _container(
    values: np.ndarray = CELLS, no_data_value: float | None = None
) -> NetCDF:
    """A `(time, y, x)` container holding one variable.

    Args:
        values: The cells.
        no_data_value: The sentinel to declare.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        values,
        geo_ref=GEO,
        variable_name="t",
        no_data_value=no_data_value,
        dims=ExtraDimensions(name="time", values=STAMPS),
    )


class TestTheIndex:
    """The `MultiIndex` naming the dimensions, outermost first."""

    def test_the_spatial_levels_are_cell_centres(self):
        """`y` and `x` come off the geotransform, half a cell in from the corner.

        Test scenario:
            The origin is `(0.0, 2.0)` on a unit grid, so the two rows are centred at
            `1.5` and `0.5` and the two columns at `0.5` and `1.5` — north-up, whatever
            order the store keeps its rows in.
        """
        frame = _container().to_dataframe()
        assert list(dict.fromkeys(frame.index.get_level_values("y"))) == [1.5, 0.5]
        assert list(dict.fromkeys(frame.index.get_level_values("x"))) == [0.5, 1.5]

    def test_a_dimension_without_coordinates_is_indexed_by_position(self):
        """An axis carrying no stamps is labelled `0, 1, ...` rather than left out.

        Test scenario:
            Every level of the index has to be as long as its dimension, so a dimension
            with nothing to label it falls back to positions. The variable is held as one
            object because `get_variable` answers a fresh view each call.
        """
        variable = _container().get_variable("t")
        variable._band_dim_values_map["time"] = None
        frame = variable.to_dataframe()
        assert list(frame.index.names) == ["time", "y", "x"]
        assert list(dict.fromkeys(frame.index.get_level_values("time"))) == [0, 1]
        assert len(frame) == 8

    def test_the_rows_are_in_array_order(self):
        """Reading the column back gives the cells as the array lays them out."""
        assert _container().to_dataframe()["t"].tolist() == list(range(8))


class TestTheSentinel:
    """A declared no-data value becomes NaN, whatever it is."""

    def test_a_numeric_sentinel_becomes_nan(self):
        """The cell holding `-9999.0` reads as missing, not as a very cold day."""
        values = CELLS.copy()
        values[0, 0, 0] = -9999.0
        column = _container(values, no_data_value=-9999.0).to_dataframe()["t"]
        assert np.isnan(column.iloc[0])
        assert column.iloc[1] == pytest.approx(1.0)

    def test_a_nan_sentinel_needs_no_translating(self):
        """A store already spelling its gaps NaN is handed over unchanged.

        Test scenario:
            The translation compares each cell against the sentinel, and NaN equals
            nothing — including itself — so a NaN sentinel has to be recognised as
            already-done rather than compared against.
        """
        values = CELLS.copy()
        values[0, 0, 1] = np.nan
        frame = _container(values, no_data_value=np.nan).to_dataframe()
        assert np.isnan(frame["t"].iloc[1])
        assert frame["t"].iloc[0] == pytest.approx(0.0)
        assert len(frame) == 8
        assert len(frame.dropna(how="all")) == 7

    def test_dropna_removes_the_nan_sentinel_row(self):
        """The opt-in convenience reads a NaN sentinel as the gap it is."""
        values = CELLS.copy()
        values[0, 0, 1] = np.nan
        assert (
            len(_container(values, no_data_value=np.nan).to_dataframe(dropna=True)) == 7
        )


class TestRefusals:
    """What `to_dataframe` will not answer, and what it says instead."""

    def test_a_container_with_no_gridded_variable(self):
        """A store can hold variables and still have no cells to lay out on an index.

        Test scenario:
            On this ROMS-shaped store `salt` and `zeta` are enumerated but neither
            carries a recognised `(y, x)` pair, so there is no grid for the index to
            describe. The refusal has to say that rather than fail while reading a
            variable it never found.
        """
        container = NetCDF.read_file(str(NO_GRIDDED_VARIABLE))
        assert container.variable_names, "precondition: the store enumerates variables"
        assert not container._spatial_variable_names(), (
            "precondition: none of them is gridded"
        )
        with pytest.raises(ValueError, match="needs at least one gridded variable"):
            container.to_dataframe()

    def test_an_unknown_variable_names_the_ones_there_are(self):
        """The refusal lists the columns the caller could have asked for."""
        with pytest.raises(ValueError, match=r"gridded variables are \['t'\]"):
            _container().to_dataframe(variables="rain")


class TestAPackedVariable:
    """A CF-packed variable's fill cells reach the frame as NaN, not as a number."""

    @staticmethod
    def _packed(tmp_path) -> str:
        """A one-band NetCDF whose band carries `scale_factor`, `add_offset` and a fill.

        Args:
            tmp_path: pytest's temporary directory.

        Returns:
            str: The file's path.
        """
        path = str(tmp_path / "packed.nc")
        dataset = gdal.GetDriverByName("netCDF").Create(path, 2, 1, 1, gdal.GDT_Int16)
        dataset.SetGeoTransform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
        band = dataset.GetRasterBand(1)
        band.SetScale(0.1)
        band.SetOffset(5.0)
        band.SetNoDataValue(-9999)
        band.WriteArray(np.array([[10, -9999]], dtype="int16"))
        dataset = None
        return path

    def test_the_fill_cell_is_nan(self, tmp_path):
        """The gap is missing in the frame, as it is in every other reader of the cube.

        Test scenario:
            The values were unpacked to physical units while the sentinel was read straight
            off `no_data_value`, which is the *stored* `_FillValue`. The fill cell therefore
            held `-994.9` — scale and offset applied to `-9999` — and matched nothing, so it
            arrived in pandas as a measurement. `isnull()` on the same variable answered `1`,
            because the `Analysis` path unpacks the sentinel first.

        Args:
            tmp_path: pytest's temporary directory.
        """
        nc = NetCDF.read_file(self._packed(tmp_path))
        column = nc.to_dataframe().iloc[:, 0].tolist()
        assert column[0] == pytest.approx(6.0)
        assert np.isnan(column[1])

    def test_dropna_drops_the_fill_row(self, tmp_path):
        """Once the cell is NaN, `dropna` can see it.

        Args:
            tmp_path: pytest's temporary directory.
        """
        nc = NetCDF.read_file(self._packed(tmp_path))
        assert len(nc.to_dataframe()) == 2
        assert len(nc.to_dataframe(dropna=True)) == 1

    def test_the_frame_agrees_with_isnull(self, tmp_path):
        """The two readers of the same cube mark the same cells missing.

        Args:
            tmp_path: pytest's temporary directory.
        """
        nc = NetCDF.read_file(self._packed(tmp_path))
        variable = nc.get_variable(nc.variable_names[0])
        flags = np.asarray(variable.isnull().read_array()).ravel().tolist()
        missing = [int(np.isnan(value)) for value in nc.to_dataframe().iloc[:, 0]]
        assert missing == flags
