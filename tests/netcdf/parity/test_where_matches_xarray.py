"""`where` answers what `xarray.DataArray.where` answers, for each condition form.

The three forms — an array, a raster on the same grid, and a callable — all reduce to the
same mask, so each is compared against `xds.where(...)` on the same values. `drop=True` is
compared by shape and by the values that survive, since xarray trims coordinate labels and
pyramids trims the grid.

**One deliberate difference.** A condition raster carries its own no-data value, and a cell
holding it reads as **false** here. xarray has no equivalent — its condition is a plain
boolean array — so the comparison is made with a condition that has no gaps, and the gap
behaviour is pinned in `tests/dataset/analysis/test_where.py` instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, NetCDF
from pyramids.netcdf import GeoReference as NCGeoReference

pytestmark = pytest.mark.interop

xr = pytest.importorskip("xarray")

GEO_REF = GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326)
NDV = -9999.0
VALUES = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, NDV, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])


def _raster() -> Dataset:
    """The 3x4 raster the comparisons run on, one cell of it no-data.

    Returns:
        Dataset: The raster.
    """
    return Dataset.from_array(VALUES, geo_ref=GEO_REF, no_data_value=NDV)


def _exported() -> xr.DataArray:
    """The same values as xarray sees them, the gap as NaN.

    Returns:
        xr.DataArray: The array, dimensions `(y, x)`.
    """
    values = np.where(VALUES == NDV, np.nan, VALUES)
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]},
    )


def _read(raster: Dataset) -> np.ndarray:
    """The raster's values with its gaps as NaN, for comparison with xarray's.

    Args:
        raster: The raster to read.

    Returns:
        np.ndarray: The values.
    """
    values = np.asarray(raster.read_array(), dtype="float64")
    sentinel = raster.no_data_value[0]
    if sentinel is not None and not np.isnan(sentinel):
        values = np.where(values == sentinel, np.nan, values)
    return values


class TestWhereMatchesXarray:
    """Each condition form answers what xarray answers for the same mask."""

    @pytest.mark.parametrize("threshold", [2.0, 5.0, 9.0, 11.0])
    def test_an_array_condition(self, threshold):
        """`where(values > t)` matches `xds.where(xds > t)` for four thresholds.

        Args:
            threshold: The comparison threshold.
        """
        exported = _exported()
        np.testing.assert_allclose(
            _read(_raster().where(VALUES > threshold)),
            exported.where(exported > threshold).values,
            equal_nan=True,
        )

    @pytest.mark.parametrize("other", [-1.0, 0.0, 100.0])
    def test_the_other_value(self, other):
        """`where(cond, other)` writes `other` exactly where xarray writes it.

        Args:
            other: What an unselected cell holds.
        """
        exported = _exported()
        np.testing.assert_allclose(
            np.asarray(
                _raster().where(VALUES > 5, other).read_array(), dtype="float64"
            ),
            exported.where(exported > 5, other).values,
            equal_nan=True,
        )

    def test_a_callable_condition(self):
        """A callable resolves to the same mask as the array it returns."""
        exported = _exported()
        np.testing.assert_allclose(
            _read(_raster().where(lambda values: values > 5)),
            exported.where(exported > 5).values,
            equal_nan=True,
        )

    def test_a_raster_condition_without_gaps(self):
        """A flags raster with no no-data value selects exactly what the array selects."""
        exported = _exported()
        flags = Dataset.from_array(
            (VALUES > 5).astype("uint8"), geo_ref=GEO_REF, no_data_value=None
        )
        np.testing.assert_allclose(
            _read(_raster().where(flags)),
            exported.where(exported > 5).values,
            equal_nan=True,
        )

    def test_the_source_gap_stays_a_gap(self):
        """The cell that was already missing is missing in both answers."""
        exported = _exported()
        ours = _read(_raster().where(VALUES > 0))
        theirs = exported.where(exported > 0).values
        assert np.isnan(ours[1, 1])
        assert np.isnan(theirs[1, 1])


class TestDropMatchesXarray:
    """`drop=True` keeps the same block of cells xarray keeps."""

    @pytest.mark.parametrize("threshold", [5.0, 9.0, 10.0])
    def test_the_surviving_block(self, threshold):
        """The trimmed values equal xarray's trimmed values, for three thresholds.

        Args:
            threshold: The comparison threshold.
        """
        exported = _exported()
        theirs = exported.where(exported > threshold, drop=True)
        ours = _raster().where(VALUES > threshold, drop=True)
        assert ours.rows == theirs.shape[0], (ours.rows, theirs.shape)
        assert ours.columns == theirs.shape[1], (ours.columns, theirs.shape)
        np.testing.assert_allclose(_read(ours), theirs.values, equal_nan=True)

    def test_the_trimmed_grid_holds_the_same_coordinates(self):
        """The kept rows and columns are the ones xarray kept, read off the geotransform."""
        exported = _exported()
        theirs = exported.where(exported > 9, drop=True)
        ours = _raster().where(VALUES > 9, drop=True)
        geo = ours.geotransform
        centres_x = [geo[0] + (i + 0.5) * geo[1] for i in range(ours.columns)]
        centres_y = [geo[3] + (i + 0.5) * geo[5] for i in range(ours.rows)]
        np.testing.assert_allclose(centres_x, theirs.x.values)
        np.testing.assert_allclose(centres_y, theirs.y.values)


class TestTheReceivers:
    """`where` masks a raster, and a NetCDF variable is one; a container is not."""

    @staticmethod
    def _container():
        """A two-step container holding one gridded variable.

        Returns:
            NetCDF: The container.
        """
        return NetCDF.from_array(
            np.arange(24.0).reshape(2, 3, 4),
            geo_ref=NCGeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326),
            variable_name="v",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
        )

    def test_a_variable_keeps_its_band_dimensions(self):
        """The result still carries `time` and its stamps, so it still selects."""
        variable = self._container().get_variable("v")
        result = variable.where(variable > 10)
        assert tuple(result._band_dim_names) == ("time",)
        assert result._band_dim_values_map["time"] == [0.0, 6.0]
        assert np.asarray(result.sel(time=6.0).read_array()).shape == (3, 4)

    def test_a_variable_masks_the_right_cells(self):
        """Only the values above the threshold survive, across both steps."""
        variable = self._container().get_variable("v")
        read = np.asarray(variable.where(variable > 10).read_array(), dtype="float64")
        kept = read[read != NDV]
        np.testing.assert_allclose(kept, np.arange(11.0, 24.0))

    def test_a_container_says_to_call_it_on_a_variable(self):
        """A container has no grid of its own, and the refusal names the way through.

        Test scenario:
            The grid check compared the condition against the container's placeholder
            raster and refused with an alignment message, which describes nothing the
            caller did.
        """
        container = self._container()
        condition = container.get_variable("v") > 10
        with pytest.raises(ValueError, match="get_variable"):
            container.where(condition)
