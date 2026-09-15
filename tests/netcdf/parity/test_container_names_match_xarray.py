"""Iterating the container yields the same names `to_xarray` exports -- almost.

`sorted(nc) == sorted(nc.to_xarray().data_vars)` is the claim the mapping protocol is worth
having: a reader who learns `list(ds)` in xarray gets the same answer here. It holds on 23 of
the repo's 26 stores and fails on three, so it is asserted as a sweep over the stores where it
holds plus one named test per store where it does not.

The three are asserted rather than skipped because each is a different kind of disagreement,
and none of them is a defect on either side:

- the grouped store hits two limits of xarray's data model at once, and `to_xarray` warns
  about both: a `Dataset` is one flat namespace, so `group/var` is flattened to `group_var`,
  and it has one size per dimension name, so the variables whose `recNum` disagrees with the
  one already exported are skipped
- the GOES and UGRID stores are **classification** differences: `to_xarray` exports an array
  that CF classification leaves out of `variable_names`

Pinning them here means a change to either side shows up as a failing assertion naming the
store, rather than as a sweep that quietly covers one fewer file.
"""

from __future__ import annotations

import pytest

from tests.netcdf.parity._catalogue import DATA, open_fixture

pytestmark = pytest.mark.interop

#: The stores where the container's names and the export's disagree, each with its own test
#: below. Kept as a set so the sweep can exclude exactly these and nothing else.
DIVERGENT = {
    "none__35v__1d35__groups-nc4.nc",
    "cf__9v__1d7-2d2__geos__y-desc.nc",
    "ugrid__6v__1d5-2d1.nc",
}

#: Everything else in the fixture directory, globbed at **import** time. This is deliberate
#: and has a consequence worth knowing before you add a fixture: any `.nc` file dropped into
#: `tests/data/netcdf` for an unrelated feature joins this sweep automatically, and must
#: then either agree with `to_xarray` or be added to `DIVERGENT` with a test of its own. That
#: is the point -- a new store silently escaping the parity claim is the failure mode this
#: guards against -- but it does mean an unrelated fixture can turn this suite red.
AGREEING = tuple(
    sorted(path.name for path in DATA.glob("*.nc") if path.name not in DIVERGENT)
)


class TestTheNamesAgree:
    """The claim, on every store where it holds."""

    @pytest.mark.parametrize("name", AGREEING, ids=lambda name: name[:-3])
    def test_iterating_the_container_yields_what_the_export_holds(self, name: str):
        """`list(nc)` and `list(nc.to_xarray())` name the same variables.

        Args:
            name: The `.nc` file to compare.

        Test scenario:
            Both sides are sorted, because the claim is about the *set* of names: store
            order is pinned separately, and `to_xarray` is free to order its `data_vars`
            however it builds them.
        """
        nc = open_fixture(name)

        assert sorted(nc) == sorted(nc.to_xarray().data_vars)

    def test_the_sweep_covers_every_store_but_the_named_three(self):
        """The exclusion list cannot grow silently.

        Test scenario:
            A store added to `DIVERGENT` without a test of its own would shrink the sweep
            with nothing to show for it. The partition is asserted so that adding a fixture
            to the repo extends the sweep, and excluding one has to be deliberate.

            The partition is the whole assertion: a literal count of the excluded stores
            would be a second place to edit for the same change, and says nothing the
            set comparison does not already say.
        """
        every_store = {path.name for path in DATA.glob("*.nc")}

        assert DIVERGENT < every_store
        assert set(AGREEING) == every_store - DIVERGENT
        assert set(AGREEING).isdisjoint(DIVERGENT)


class TestWhereTheNamesDisagree:
    """One test per store, each naming what the disagreement is."""

    def test_a_grouped_store_keeps_its_paths_where_the_export_flattens_and_skips(self):
        """`group/var` against `group_var`, and 20 variables xarray has no room for.

        Test scenario:
            The container addresses a sub-group variable by its store path, which is what
            `get_variable` and `variables` both accept; `to_xarray` joins the path with an
            underscore, because a `Dataset` is one flat namespace and netCDF forbids `/` in
            a name.

            29 data variables become 9, and the shortfall is not the flattening: each
            flight group has its own `recNum`, and an xarray `Dataset` holds one size per
            dimension name, so the 20 whose `recNum` disagrees with the first exported one
            cannot go in the same object. Both facts are asserted through the warnings
            `to_xarray` raises rather than through the counts alone, because the counts
            would also match an export that dropped them silently -- and the warnings are
            what tell a reader to use `get_variable` for the rest.
        """
        nc = open_fixture("none__35v__1d35__groups-nc4.nc")

        with pytest.warns(UserWarning) as caught:
            cube = nc.to_xarray()

        exported = sorted(cube.data_vars)
        messages = [str(warning.message) for warning in caught]
        renamed = [text for text in messages if "renamed 8 variable" in text]
        skipped = [text for text in messages if "skipped 20 variable" in text]

        assert len(renamed) == 1
        assert len(skipped) == 1
        assert "get_variable" in renamed[0]

        grouped = [name for name in nc if "/" in name]
        assert len(grouped) == 28
        assert not any("/" in name for name in exported)

        flattened = {name.replace("/", "_") for name in grouped}
        assert set(exported) - {"UTC_time"} < flattened
        assert len(exported) == 9

    def test_the_geostationary_store_exports_a_variable_the_container_omits(self):
        """`DQF` is in the export and not in `variable_names`.

        Test scenario:
            `DQF` is the GOES data-quality flag array. It is shaped like the image and
            `to_xarray` exports it as a data variable; CF classification treats it as
            ancillary and leaves it out of the container's enumeration. `get_variable`
            still reads it, so the array is reachable -- it is the enumeration the two
            sides differ on.
        """
        nc = open_fixture("cf__9v__1d7-2d2__geos__y-desc.nc")
        exported = sorted(nc.to_xarray().data_vars)

        assert "DQF" in exported
        assert "DQF" not in nc
        # `get_variable` returns an object or raises, so `is not None` cannot fail.
        # The real claim is that the array is reachable *and* holds data.
        assert nc.get_variable("DQF").read_array().size > 0
        assert set(exported) - set(nc) == {"DQF"}

    def test_the_ugrid_store_exports_its_connectivity_array(self):
        """`face_node_connectivity` is in the export and not in `variable_names`.

        Test scenario:
            The connectivity array is mesh topology -- which node indices make up each
            face -- rather than a field sampled on the mesh, so the container does not
            enumerate it among the data variables. The export carries it, which is the
            same classification disagreement as the GOES store's, on a different kind of
            array. It is reachable through `get_variable` either way.
        """
        nc = open_fixture("ugrid__6v__1d5-2d1.nc")
        exported = sorted(nc.to_xarray().data_vars)

        assert "face_node_connectivity" in exported
        assert "face_node_connectivity" not in nc
        # `get_variable` returns an object or raises, so `is not None` cannot fail.
        # The real claim is that the array is reachable *and* holds data.
        assert nc.get_variable("face_node_connectivity").read_array().size > 0
        assert set(exported) - set(nc) == {"face_node_connectivity"}
