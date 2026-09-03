"""A container-wide operation must preserve each variable's own CF attributes.

`NetCDF.crop` / `to_crs` / `resample` fan out over every gridded variable and
rebuild the container from the results. That rebuild goes through `from_array`,
which takes CF *global* attributes only, so per-variable attributes such as
`long_name` and `standard_name` used to be dropped -- while the identical
single-variable call kept them. The container route and the single-variable
route disagreed about the same variable of the same file.
"""

from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"

# Fixtures whose first gridded variable actually carries CF attributes worth
# preserving; files with no such attributes cannot exercise the regression.
FIXTURES_WITH_ATTRS = [
    "cf__20v__1d3-3d17__y-desc.nc",
    "cf__5v__1d4-4d1__geog__y-desc.nc",
]


def variable_attrs(dataset) -> dict:
    """The per-variable CF attributes a NetCDF view carries."""
    return dict(getattr(dataset, "_variable_attrs", {}) or {})


@pytest.fixture(params=FIXTURES_WITH_ATTRS, ids=FIXTURES_WITH_ATTRS)
def container(request) -> NetCDF:
    """A real CF container whose first variable has CF attributes."""
    return NetCDF.read_file(str(DATA / request.param))


class TestFanOutPreservesVariableAttrs:
    """The container route keeps what the single-variable route keeps."""

    def test_crop_keeps_the_attributes_the_single_variable_route_keeps(
        self, container: NetCDF
    ):
        """A container-wide crop loses none of a variable's CF attributes."""
        name = container.variable_names[0]
        bounds = container.total_bounds
        bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

        single = container.get_variable(name).crop(bbox=bbox, epsg=4326)
        whole = container.crop(bbox=bbox, epsg=4326).get_variable(name)

        expected = variable_attrs(single)
        assert expected, "fixture no longer carries per-variable attributes"
        missing = set(expected) - set(variable_attrs(whole))
        assert not missing, f"container route dropped {sorted(missing)}"

    def test_the_carried_attribute_values_are_unchanged(self, container: NetCDF):
        """Carried attributes keep their values, not just their names."""
        name = container.variable_names[0]
        bounds = container.total_bounds
        bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

        single = variable_attrs(container.get_variable(name).crop(bbox=bbox, epsg=4326))
        whole = variable_attrs(
            container.crop(bbox=bbox, epsg=4326).get_variable(name)
        )

        for key, value in single.items():
            assert whole.get(key) == value
