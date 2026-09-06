"""`variable_names` reports the store's declaration order, not a sorted one.

The property used to hand back the CF classification's own list, which is
sorted, so every multi-variable container was reported alphabetically whatever
order the file declared. It now filters the declared list instead, keeping the
store's order.

That is deliberate and it matters: `_fan_out_eager` templates a container-wide
result from the **first** spatial variable -- its geotransform, CRS, no-data and
extra dimensions -- so which name comes first decides what the output container
looks like, and the order propagates into `to_netcdf` and `to_xarray`. It is
also silent: the set of names is unchanged, only their order, so nothing raises
and nothing warns. Pinned here so the change cannot be undone by accident, and
documented for downstreams in `docs/migration.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"

# Two data variables declared `z` then `q`. Alphabetical order reverses them,
# which is what makes this file able to tell the two orders apart at all.
PACKED = DATA / "coards__4v__1d2-2d2__scaleoffset__y-asc.nc"

# Four, declared with `y_image_bounds` ahead of `x_image_bounds`. Three of the
# four are named `*_bounds` and are still data variables here, which looks like
# it contradicts `tests/netcdf/structure/test_variable_names_invariance.py`
# excluding `cf__7v`'s `lat_bnds`. It does not: no array in this store carries a
# CF `bounds` attribute pointing at them, while `cf__7v`'s `lat` declares
# `bounds = "lat_bnds"`. `TestTheClassificationFollowsDeclarations` in that file
# asserts both declarations, so the difference is pinned rather than implied.
GEOS = DATA / "cf__9v__1d7-2d2__geos__y-desc.nc"


class TestTheOrderIsTheStores:
    """What the file says, not what `sorted` says."""

    def test_a_two_variable_store_keeps_its_declaration_order(self):
        """The smallest case that can tell the two orders apart.

        Test scenario:
            `z` is declared first and sorts second. Reporting `['q', 'z']`
            would mean a container-wide `resample` or `to_crs` templates its
            output from `q` instead of `z`.
        """
        assert NetCDF.read_file(str(PACKED)).variable_names == ["z", "q"]

    def test_a_larger_store_keeps_it_too(self):
        """Not an artefact of two names happening to land that way.

        Test scenario:
            `y_image_bounds` precedes `x_image_bounds` in the file and follows
            it alphabetically, so the same reversal shows up in a store with
            more than one non-sorted pair.
        """
        names = NetCDF.read_file(str(GEOS)).variable_names

        assert names == ["CMI", "time_bounds", "y_image_bounds", "x_image_bounds"]

    def test_the_set_of_names_is_what_the_classification_allows(self):
        """Order is the only thing that changed.

        Test scenario:
            The declared list is *filtered* by the CF classification rather
            than replacing it, so a store variable the classification does not
            call a data variable is still left out.
        """
        nc = NetCDF.read_file(str(PACKED))

        assert set(nc.variable_names) == {"z", "q"}
