"""A container-wide operation keeps each variable's packing.

`nc.get_variable("z").crop(mask)` and `nc.crop(mask).get_variable("z")` are the
same request spelled two ways, and they disagreed by the packing factor. The
fan-out rebuilds every variable through `from_array` and writes the array
`read_array()` hands it -- which is **raw**, `unpack=False` being the default --
but GDAL keeps `scale_factor` / `add_offset` in the MDArray's own scale and
offset slots rather than in its attribute dictionary, so the attribute carry
could not restore them. The rebuilt variable held packed counts with nothing
left to say they were packed, and `read_array(unpack=True)` returned them
unscaled: a hundredfold error on the suite's own `scale_factor=0.01` fixture,
silent.

Because the stored array stays raw, restoring the slots cannot double-apply --
which is the property that makes this safe, and is asserted here directly.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"
# Two variables, packed with *different* factors (0.01/1.5 and 0.1/2.5), so a
# carry that stamped one factor onto every variable would fail rather than pass.
PACKED = DATA / "coards__4v__1d2-2d2__scaleoffset__y-asc.nc"
UNPACKED = DATA / "cf__5v__1d4-4d1__y-asc.nc"


@pytest.fixture
def packed() -> NetCDF:
    """The packed COARDS fixture, opened fresh.

    Returns:
        NetCDF: A container whose `z` and `q` carry different packing.
    """
    return NetCDF.read_file(str(PACKED))


def _full_extent_mask(container: NetCDF, variable: str) -> gpd.GeoDataFrame:
    """A cutline covering a variable's whole extent.

    Cropping to the full extent keeps every cell, so the two spellings of the
    request are comparable value by value rather than only in shape.

    Args:
        container: The container to measure.
        variable: The variable whose bounds define the box.

    Returns:
        gpd.GeoDataFrame: A one-row frame holding the box.
    """
    variable_view = container.get_variable(variable)
    west, south, east, north = variable_view.bounds.total_bounds
    return gpd.GeoDataFrame(
        geometry=[box(west, south, east, north)], crs=variable_view.crs
    )


class TestTheFanOutCarriesThePacking:
    """The two spellings of one request must agree."""

    @pytest.mark.parametrize(
        ("name", "scale", "offset"),
        [("z", 0.01, 1.5), ("q", 0.1, 2.5)],
    )
    def test_the_slots_survive_a_container_crop(
        self, packed, name: str, scale: float, offset: float
    ):
        """The regression, on both variables and both fan-out paths.

        Args:
            packed: The packed fixture.
            name: The variable to check.
            scale: Its `scale_factor`.
            offset: Its `add_offset`.

        Test scenario:
            `z` is the container's first spatial variable and goes through the
            `from_array` branch; `q` is a later one and goes through
            `set_variable`. Only the first was stamped, so a single-variable
            fixture would have hidden half the defect. The factors differ, so
            a carry that reused one variable's packing for all of them fails
            here too.
        """
        mask = _full_extent_mask(packed, name)

        cropped = packed.crop(mask).get_variable(name)

        assert cropped._scale == scale, (
            f"{name}: expected scale {scale}, got {cropped._scale}"
        )
        assert cropped._offset == offset, (
            f"{name}: expected offset {offset}, got {cropped._offset}"
        )

    @pytest.mark.parametrize("name", ["z", "q"])
    def test_both_spellings_of_the_request_unpack_alike(self, packed, name: str):
        """The user-visible symptom: a hundredfold disagreement.

        Args:
            packed: The packed fixture.
            name: The variable to compare.

        Test scenario:
            Cropping the variable and cropping the container then taking the
            variable are the same request. Read with `unpack=True` they have
            to give the same physical values -- the per-variable path always
            did, and the container path returned raw counts.
        """
        mask = _full_extent_mask(packed, name)

        per_variable = packed.get_variable(name).crop(mask).read_array(unpack=True)
        via_container = packed.crop(mask).get_variable(name).read_array(unpack=True)

        assert np.allclose(np.asarray(per_variable), np.asarray(via_container)), (
            f"{name}: {np.asarray(per_variable).ravel()[:4]} != "
            f"{np.asarray(via_container).ravel()[:4]}"
        )

    @pytest.mark.parametrize("name", ["z", "q"])
    def test_the_stored_array_is_still_raw(self, packed, name: str):
        """What makes restoring the slots safe rather than a double-apply.

        Args:
            packed: The packed fixture.
            name: The variable to compare.

        Test scenario:
            The fan-out writes what `read_array()` returns, and that is raw
            because `unpack=False` is the default. If the rebuild ever started
            writing unpacked values, stamping the packing back on would scale
            them a second time -- so the rawness of the stored array is the
            precondition, and it is asserted rather than assumed.
        """
        mask = _full_extent_mask(packed, name)

        per_variable = packed.get_variable(name).crop(mask).read_array()
        via_container = packed.crop(mask).get_variable(name).read_array()

        assert np.array_equal(np.asarray(per_variable), np.asarray(via_container)), (
            f"{name}: the rebuilt array is not the raw one the source held"
        )

    def test_unpacking_actually_changes_the_values(self, packed):
        """The comparison above is only meaningful if packing does something.

        Test scenario:
            A fixture whose `scale_factor` happened to be 1.0 and `add_offset`
            0.0 would make every assertion here pass without testing
            anything. The raw and unpacked reads must differ.
        """
        mask = _full_extent_mask(packed, "z")

        cropped = packed.crop(mask).get_variable("z")
        raw = np.asarray(cropped.read_array())
        unpacked = np.asarray(cropped.read_array(unpack=True))

        assert not np.allclose(raw, unpacked), (
            "raw and unpacked reads agree, so this fixture proves nothing"
        )


class TestAnUnpackedVariableGainsNoPacking:
    """Carrying the slots must not invent them."""

    def test_a_container_without_packing_stays_without_it(self):
        """The stamp is conditional, not unconditional.

        Test scenario:
            Writing `SetScale(None)` -- or defaulting to 1.0 -- would put a
            packing declaration on every variable of every rebuilt container,
            which readers would then apply. A plain CF fixture must come
            through a crop with both slots still absent.
        """
        container = NetCDF.read_file(str(UNPACKED))
        name = container.variable_names[0]
        source = container.get_variable(name)
        assert source._scale is None, "fixture must be unpacked to prove anything"
        west, south, east, north = source.bounds.total_bounds
        mask = gpd.GeoDataFrame(
            geometry=[box(west, south, east, north)], crs=source.crs
        )

        cropped = container.crop(mask).get_variable(name)

        assert cropped._scale is None, f"invented a scale: {cropped._scale}"
        assert cropped._offset is None, f"invented an offset: {cropped._offset}"
