"""A refusal must list the names the check it failed actually accepts.

Four sites reject an unknown variable name, and every one of them gated on
:meth:`NetCDF._readable_variable_names` while printing :attr:`variable_names`
in the message. Those two lists deliberately differ -- the property enumerates
*data* variables, the gate asks what the store holds -- so a user who mistyped
`lat_rho` on a curvilinear store was told the file contains only `salt` and
`zeta`, and a user plotting a GOES granule was never told `DQF` was there.

The tests below assert the message *content*, not that a `ValueError` is
raised: the raise was never in doubt, the list inside it was. The rejected
name is deliberately one that shares no substring with the names being looked
for, so a message that merely echoes the input cannot pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF
from pyramids.netcdf._plot import CurvilinearCoordResolver, NetCDFPlot

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[2] / "data" / "netcdf"

# ROMS-shaped: `salt` / `zeta` are the data variables; `lat_rho` / `lon_rho` /
# `h` / `Cs_r` are readable auxiliaries the enumeration leaves out.
CURVILINEAR = DATA / "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc"

# GOES ABI: `DQF` is gridded and plottable but not enumerated, while
# `time_bounds` / `x_image_bounds` / `y_image_bounds` are enumerated and are
# not plottable (1-D, so `get_variable` hands back a raw MDArray).
GEOS = DATA / "cf__9v__1d7-2d2__geos__y-desc.nc"

# Shares no substring with any array name in either fixture.
ABSENT = "zzz_not_here"


class TestTheRefusalNamesWhatTheGateAccepts:
    """Every "Available:" list is the list its own check consults."""

    def test_get_variable_lists_every_name_it_would_have_accepted(self):
        """A name the gate accepts must be findable in the gate's refusal.

        Test scenario:
            `get_variable` gates on the readable superset, so `lat_rho` is
            accepted. Printing `variable_names` told the user the store held
            only `['salt', 'zeta']` -- so a typo in `lat_rho` was unfixable
            from the message, which is the whole point of listing anything.
        """
        dataset = NetCDF.read_file(str(CURVILINEAR))
        readable = dataset._readable_variable_names()
        assert "lat_rho" in readable, "fixture no longer carries the auxiliary"

        with pytest.raises(ValueError) as excinfo:
            dataset.get_variable(ABSENT)

        message = str(excinfo.value)
        missing = [name for name in readable if name not in message]
        assert not missing, (
            f"accepted names absent from the refusal {missing}: {message}"
        )

    def test_rename_variable_lists_every_name_it_would_have_accepted(self):
        """`rename_variable` accepts an auxiliary, so it must list one.

        Test scenario:
            The gate is the readable superset -- renaming `lat_rho` works --
            while the message listed the enumeration, so it advertised a
            narrower contract than the code honours.
        """
        dataset = NetCDF.read_file(str(CURVILINEAR))
        readable = dataset._readable_variable_names()

        with pytest.raises(ValueError) as excinfo:
            dataset.rename_variable(ABSENT, "whatever")

        message = str(excinfo.value)
        missing = [name for name in readable if name not in message]
        assert not missing, (
            f"accepted names absent from the refusal {missing}: {message}"
        )

    def test_the_coords_refusal_names_the_coordinate_fields_it_accepts(self):
        """`coords="lat_rho"` is exactly what this refusal should point at.

        Test scenario:
            A 2-D curvilinear coordinate field is the *only* kind of array
            `coords=` is ever given, and it is never a data variable -- so
            listing `variable_names` here could not name a single valid
            answer.
        """
        dataset = NetCDF.read_file(str(CURVILINEAR))
        readable = dataset._readable_variable_names()
        resolver = CurvilinearCoordResolver(dataset.get_variable("salt"))

        with pytest.raises(ValueError) as excinfo:
            resolver._coerce(ABSENT, "y")

        message = str(excinfo.value)
        missing = [name for name in readable if name not in message]
        assert not missing, (
            f"accepted names absent from the refusal {missing}: {message}"
        )

    def test_the_container_plot_refusal_lists_the_plottable_variables(self):
        """A plot refusal must list what can be plotted, no more and no less.

        Test scenario:
            On the GOES fixture the enumeration is wrong in both directions:
            it omits `DQF`, which is gridded and plots fine, and it includes
            `time_bounds`, which is 1-D and comes back from `get_variable` as
            a raw MDArray with no `plot`. The gridded set is the honest answer,
            so the refusal names it rather than either of the other two lists.
        """
        dataset = NetCDF.read_file(str(GEOS))
        plot = NetCDFPlot(dataset)

        with pytest.raises(ValueError) as excinfo:
            plot.run()

        message = str(excinfo.value)
        assert "DQF" in message, f"a plottable variable was not offered: {message}"
        assert "time_bounds" not in message, (
            f"a non-plottable variable was offered: {message}"
        )
