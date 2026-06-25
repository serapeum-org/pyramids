"""Unit tests for the shared MDIM helpers in :mod:`pyramids.netcdf._mdim`.

The helpers are thin wrappers over the GDAL multidimensional API. Happy paths are
exercised against real in-memory MDIM datasets built with ``NetCDF.create_from_array``;
the orientation-probe branches of ``needs_y_flip`` (which depend on the derived
geotransform sign) are covered with lightweight mocks for determinism.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF
from pyramids.netcdf._mdim import (
    needs_y_flip,
    open_mdarray,
    root_group,
    scalar_no_data,
)


@pytest.fixture(scope="function")
def mdim_dataset():
    """A real in-memory MDIM ``gdal.Dataset`` carrying one 2-D variable ``v``.

    Returns:
        gdal.Dataset: An MDIM-capable dataset whose root group holds the ``v`` array
        (plus its ``x``/``y`` coordinate arrays), built north-up.
    """
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    nc = NetCDF.create_from_array(
        arr,
        top_left_corner=(0, 0),
        cell_size=1.0,
        epsg=4326,
        variable_name="v",
    )
    return nc._raster


@pytest.fixture(scope="function")
def classic_dataset():
    """A classic (non-MDIM) in-memory ``gdal.Dataset`` with no root group.

    Returns:
        gdal.Dataset: A 4x3 single-band MEM raster.
    """
    return gdal.GetDriverByName("MEM").Create("", 4, 3, 1, gdal.GDT_Float32)


class TestRootGroup:
    """Tests for ``root_group``."""

    def test_returns_group_for_mdim_dataset(self, mdim_dataset):
        """An MDIM dataset yields its root group exposing the stored arrays.

        Test scenario:
            A dataset built by ``create_from_array`` returns a non-None group whose
            array names include the ``v`` variable.
        """
        rg = root_group(mdim_dataset)
        assert rg is not None, "MDIM dataset should expose a root group"
        assert "v" in (rg.GetMDArrayNames() or []), "variable 'v' should be listed"

    def test_returns_none_for_classic_dataset(self, classic_dataset):
        """A classic raster (no MDIM model) returns ``None`` when not required.

        Test scenario:
            A plain MEM raster has no root group, so the helper returns None.
        """
        assert root_group(classic_dataset) is None, "classic raster has no root group"

    def test_required_raises_for_classic_dataset(self, classic_dataset):
        """``required=True`` raises ``ValueError`` when there is no root group.

        Test scenario:
            A classic raster with ``required=True`` raises a clear ValueError.
        """
        with pytest.raises(ValueError, match="no root group") as exc:
            root_group(classic_dataset, required=True)
        assert "MDIM" in str(exc.value), f"error should mention MDIM, got: {exc.value}"


class TestOpenMdarray:
    """Tests for ``open_mdarray``."""

    def test_opens_existing_array(self, mdim_dataset):
        """An existing variable name returns its MDArray.

        Test scenario:
            Opening ``v`` returns a 2-D MDArray.
        """
        rg = root_group(mdim_dataset)
        arr = open_mdarray(rg, "v")
        assert arr is not None, "existing variable should open"
        assert len(arr.GetDimensions()) == 2, "variable 'v' is 2-D"

    def test_missing_array_returns_none(self, mdim_dataset):
        """A missing variable name returns ``None`` instead of raising.

        Test scenario:
            Opening a non-existent name folds GDAL's RuntimeError into None.
        """
        rg = root_group(mdim_dataset)
        assert open_mdarray(rg, "does_not_exist") is None, "missing var should be None"


class TestScalarNoData:
    """Tests for ``scalar_no_data``."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            ([-9999.0, -9999.0, -9999.0], -9999.0),
            ((1, 2, 3), 1),
            (-1.5, -1.5),
            (0, 0),
            (None, None),
            ([], []),
            ((), ()),
        ],
    )
    def test_reduces_to_scalar(self, value, expected):
        """A per-band sequence collapses to its first value; scalars pass through.

        Args:
            value: The scalar or per-band NoData input.
            expected: The expected reduced result.

        Test scenario:
            Non-empty list/tuple -> first element; scalar/None/empty -> unchanged.
        """
        result = scalar_no_data(value)
        assert result == expected, f"expected {expected!r}, got {result!r}"

    def test_ndarray_passes_through_unchanged(self):
        """A numpy array is not a list/tuple, so it is returned unchanged.

        Test scenario:
            ``scalar_no_data`` only special-cases list/tuple; an ndarray is returned
            as-is (identity preserved).
        """
        arr = np.array([1.0, 2.0])
        result = scalar_no_data(arr)
        assert result is arr, "ndarray should be returned unchanged (identity)"


class TestNeedsYFlip:
    """Tests for ``needs_y_flip``."""

    def test_north_up_variable_does_not_need_flip(self, mdim_dataset):
        """A north-up 2-D variable reports no flip needed.

        Test scenario:
            ``create_from_array`` writes a north-up grid (negative Y pixel size), so the
            probe returns False.
        """
        rg = root_group(mdim_dataset)
        assert needs_y_flip(rg, open_mdarray(rg, "v")) is False, "north-up needs no flip"

    def test_one_dimensional_array_never_flips(self):
        """A 1-D array short-circuits to ``False`` without probing orientation.

        Test scenario:
            An MDArray reporting a single dimension returns False immediately.
        """
        md_arr = Mock()
        md_arr.GetDimensions.return_value = [Mock()]
        assert needs_y_flip(Mock(), md_arr) is False, "1-D array should not flip"

    def test_south_to_north_needs_flip(self):
        """A positive Y pixel size (south-to-north storage) reports a flip is needed.

        Test scenario:
            A 2-D array whose ``AsClassicDataset`` geotransform has a positive Y pixel
            size (index 5) returns True.
        """
        src = Mock()
        src.GetGeoTransform.return_value = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        md_arr = Mock()
        md_arr.GetDimensions.return_value = [Mock(), Mock()]
        md_arr.AsClassicDataset.return_value = src
        assert needs_y_flip(Mock(), md_arr) is True, "south-to-north should flip"

    def test_probe_failure_defaults_to_no_flip(self):
        """A failing orientation probe is treated as no-flip.

        Test scenario:
            When ``AsClassicDataset`` raises, the helper degrades to False.
        """
        md_arr = Mock()
        md_arr.GetDimensions.return_value = [Mock(), Mock()]
        md_arr.AsClassicDataset.side_effect = RuntimeError("cannot view")
        assert needs_y_flip(Mock(), md_arr) is False, "probe failure should not flip"
