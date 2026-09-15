"""Parity tests for `isel()` and for a `tolerance`-bounded `sel()` against xarray.

Both members are new, and both make a claim about *xarray's* semantics rather than only about
pyramids': `isel` is documented as the positional selection xarray spells the same way, and
`tolerance` as raising the `KeyError` xarray raises. Neither claim can be checked inside
pyramids alone, so it is checked here, through the shared harness.

The harness is not optional decoration. pyramids reads north-up while `cf__5v__1d4-4d1__y-asc`
stores its rows south-up, so comparing the two libraries' raw arrays fails on all twelve
position pairs for a reason that has nothing to do with selection —
`TestIselMatchesXarray.test_the_orientation_normalisation_is_load_bearing` pins that.

Style: Google-style docstrings, <=120 char lines, no inline imports, descriptive assertions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.netcdf.parity._catalogue import open_fixture
from tests.netcdf.parity._harness import ParityView, assert_parity, from_pyramids, to_xr

pytestmark = pytest.mark.interop

FIXTURE = "cf__5v__1d4-4d1__y-asc.nc"
VARIABLE = "temperature"

NT, NL = 4, 3
TIME_VALUES = [0.0, 6.0, 12.0, 18.0]
LEVEL_VALUES = [1000.0, 850.0, 500.0]
SPATIAL_DIMS = ("lat", "lon")

#: 900 hPa is 50 from 850 and 100 from 1000, so 50 is the inclusive edge of the bound.
NEAREST_REQUEST = 900.0
NEAREST_MATCH = 850.0
NEAREST_LEVEL_INDEX = 1

POSITION_PAIRS = [(t, level) for t in range(NT) for level in range(NL)]
POSITION_IDS = [f"t{t}-l{level}" for t, level in POSITION_PAIRS]


@pytest.fixture(scope="module")
def container():
    """The synthetic 4-D store, opened once for the module."""
    return open_fixture(FIXTURE)


@pytest.fixture(scope="module")
def source(container):
    """The pyramids view of the whole variable, for the no-data mask a subset inherits.

    `isel` and `sel` choose bands; they never change which cells within a band are gaps. So the
    source's mask, sliced the same way as the values, is the mask that describes any of their
    results — which is what `from_pyramids(gaps=...)` needs for a shape-changing operation.
    """
    return from_pyramids(container, VARIABLE)


@pytest.fixture(scope="module")
def exported(container):
    """The xarray view of the whole variable, normalised onto pyramids' raster orientation."""
    return to_xr(container, VARIABLE)


def _xarray_side(exported: ParityView, **changes) -> ParityView:
    """Derive one side of a comparison from the whole-variable xarray view.

    The view is frozen, so a subset of it is made with `dataclasses.replace` — the idiom the
    harness documents. `decoded_coords` is dropped because every derived view here narrows or
    removes the time axis, and a full-length instants array alongside a narrowed coordinate
    would describe nothing.

    Args:
        exported: The whole-variable view from `to_xr`.
        **changes: Fields to replace — `values`, `gaps`, `dims`, `coords`.

    Returns:
        ParityView: The derived xarray side.
    """
    return dataclasses.replace(exported, decoded_coords={}, **changes)


class TestIselMatchesXarray:
    """`isel` has to select the band xarray's `isel` selects, for every position."""

    @pytest.mark.parametrize(
        ("time_index", "level_index"), POSITION_PAIRS, ids=POSITION_IDS
    )
    def test_every_position_pair_matches_the_export(
        self, container, source, exported, time_index, level_index
    ):
        """``isel(time=t, pressure_level=l)`` equals the export indexed at ``[t, l]``.

        Args:
            container: The opened store.
            source: The whole-variable pyramids view, for the no-data mask.
            exported: The whole-variable xarray view.
            time_index: Position along ``time``.
            level_index: Position along ``pressure_level``.

        Test scenario:
            All twelve pairs, because a single pair cannot distinguish "selects the right band"
            from "selects a band whose values happen to agree" on a cube this small. Indexing
            the normalised export is xarray's own `isel` composed with the harness' per-cell
            normalisations and its latitude flip, and `isel` over the two leading axes commutes
            with both — so the right-hand side really is `xr_ds.isel(time=t, pressure_level=l)`.
        """
        plane = np.asarray(
            container.get_variable(VARIABLE)
            .isel(time=time_index, pressure_level=level_index)
            .read_array()
        )

        pyramids_side = from_pyramids(
            container,
            VARIABLE,
            values=plane,
            gaps=source.gaps[time_index, level_index],
            dims_override=SPATIAL_DIMS,
        )
        xarray_side = _xarray_side(
            exported,
            values=exported.values[time_index, level_index],
            gaps=exported.gaps[time_index, level_index],
            dims=SPATIAL_DIMS,
            coords={name: exported.coords[name] for name in SPATIAL_DIMS},
        )

        assert_parity(pyramids_side, xarray_side)

    def test_the_orientation_normalisation_is_load_bearing(self, container):
        """The two libraries' raw planes differ, and differ only by the latitude flip.

        Test scenario:
            Without this, a reader could reasonably think the harness is ceremony around a
            comparison that would pass anyway. It would not: this store's rows ascend, pyramids
            reads north-up, and the raw arrays disagree cell for cell until one is reversed.
        """
        raw_export = np.asarray(
            container.to_xarray()[VARIABLE].isel(time=0, pressure_level=0).values
        )
        raw_read = np.asarray(
            container.get_variable(VARIABLE).isel(time=0, pressure_level=0).read_array()
        )

        assert not np.array_equal(raw_export, raw_read), (
            "if the raw arrays already agreed, the harness would be normalising nothing"
        )
        np.testing.assert_allclose(
            raw_export[::-1],
            raw_read,
            err_msg="the only difference must be the row order, not the values",
        )

    @pytest.mark.parametrize(
        ("time_index", "level_index"), POSITION_PAIRS, ids=POSITION_IDS
    )
    def test_a_position_names_the_same_coordinate_on_both_sides(
        self, container, time_index, level_index
    ):
        """Position ``i`` carries the same coordinate through both libraries' ``isel``.

        Args:
            container: The opened store.
            time_index: Position along ``time``.
            level_index: Position along ``pressure_level``.

        Test scenario:
            The values agreeing does not by itself prove the two libraries *label* the band the
            same way. This asks xarray's own `isel` what coordinate it landed on — off the
            undecoded export, so both sides speak in stored CF offsets — and compares it with
            what pyramids reports for the same position.
        """
        stored = container.to_xarray(decode_times=False)
        selected = stored[VARIABLE].isel(time=time_index, pressure_level=level_index)
        result = container.get_variable(VARIABLE).isel(
            time=time_index, pressure_level=level_index
        )

        assert float(selected["time"].values) == float(
            result.get_dimension_values("time")[0]
        ), (
            f"time position {time_index}: xarray says {float(selected['time'].values)}, "
            f"pyramids says {float(result.get_dimension_values('time')[0])}"
        )
        assert float(selected["pressure_level"].values) == float(
            result.get_dimension_values("pressure_level")[0]
        ), (
            f"level position {level_index}: xarray says "
            f"{float(selected['pressure_level'].values)}, pyramids says "
            f"{float(result.get_dimension_values('pressure_level')[0])}"
        )

    def test_a_list_selector_matches_the_export(self, container, source, exported):
        """``isel(time=[0, 2])`` keeps both steps, at every level, as the export does.

        Test scenario:
            A multi-position selector keeps a band axis rather than collapsing it, so the
            comparison covers the band ordering as well as the choice of bands — the part a
            single-plane result cannot see.
        """
        kept = [0, 2]
        selected = np.asarray(
            container.get_variable(VARIABLE).isel(time=kept).read_array()
        ).reshape(len(kept), NL, *exported.values.shape[-2:])

        pyramids_side = from_pyramids(
            container,
            VARIABLE,
            values=selected,
            gaps=source.gaps[kept],
            coords_override={"time": [TIME_VALUES[index] for index in kept]},
        )
        xarray_side = _xarray_side(
            exported,
            values=exported.values[kept],
            gaps=exported.gaps[kept],
            coords={
                **exported.coords,
                "time": np.asarray([TIME_VALUES[index] for index in kept]),
            },
        )

        assert_parity(pyramids_side, xarray_side)

    def test_a_slice_selector_matches_the_export(self, container, source, exported):
        """``isel(pressure_level=slice(1, 3))`` keeps the trailing two levels.

        Test scenario:
            The level axis is the *inner* of the two flattened band dims, so narrowing it
            strides through the flat band list rather than taking a contiguous run. That is the
            arithmetic a slice on the outer axis would never exercise, and it is where the band
            order and the declared `_band_dim_sizes` used to disagree — the case that sent six
            of eight planes to the wrong (time, level). Comparing against xarray is what says
            the fix agrees with another implementation and not merely with itself.
        """
        kept = [1, 2]
        selected = np.asarray(
            container.get_variable(VARIABLE)
            .isel(pressure_level=slice(1, 3))
            .read_array()
        ).reshape(NT, len(kept), *exported.values.shape[-2:])

        pyramids_side = from_pyramids(
            container,
            VARIABLE,
            values=selected,
            gaps=source.gaps[:, kept],
            coords_override={"pressure_level": [LEVEL_VALUES[index] for index in kept]},
        )
        xarray_side = _xarray_side(
            exported,
            values=exported.values[:, kept],
            gaps=exported.gaps[:, kept],
            coords={
                **exported.coords,
                "pressure_level": np.asarray([LEVEL_VALUES[index] for index in kept]),
            },
        )

        assert_parity(pyramids_side, xarray_side)


class TestSelToleranceMatchesXarray:
    """A bounded snap has to accept, refuse, and *fail the same way* xarray does."""

    @pytest.mark.parametrize("tolerance", [50, 60, 1000])
    def test_an_accepted_snap_matches_the_export(
        self, container, source, exported, tolerance
    ):
        """A snap inside the bound reads the level xarray's bounded snap reads.

        Args:
            container: The opened store.
            source: The whole-variable pyramids view, for the no-data mask.
            exported: The whole-variable xarray view.
            tolerance: A bound at or above the snap's distance of 50.

        Test scenario:
            The bound must not change *which* coordinate is chosen — only whether the choice
            stands. Comparing against the export at the 850 hPa index proves both libraries
            snapped to the same level, with the whole time axis left intact around it.
        """
        selected = np.asarray(
            container.get_variable(VARIABLE)
            .sel(pressure_level=NEAREST_REQUEST, method="nearest", tolerance=tolerance)
            .read_array()
        ).reshape(NT, 1, *exported.values.shape[-2:])

        pyramids_side = from_pyramids(
            container,
            VARIABLE,
            values=selected,
            gaps=source.gaps[:, NEAREST_LEVEL_INDEX : NEAREST_LEVEL_INDEX + 1],
            coords_override={"pressure_level": [NEAREST_MATCH]},
        )
        xarray_side = _xarray_side(
            exported,
            values=exported.values[:, NEAREST_LEVEL_INDEX : NEAREST_LEVEL_INDEX + 1],
            gaps=exported.gaps[:, NEAREST_LEVEL_INDEX : NEAREST_LEVEL_INDEX + 1],
            coords={**exported.coords, "pressure_level": np.asarray([NEAREST_MATCH])},
        )

        assert_parity(pyramids_side, xarray_side)

        chosen = float(
            container.to_xarray()[VARIABLE]
            .sel(pressure_level=NEAREST_REQUEST, method="nearest", tolerance=tolerance)
            .pressure_level.values
        )
        assert chosen == NEAREST_MATCH, (
            f"xarray must snap to {NEAREST_MATCH} as well, got {chosen}"
        )

    @pytest.mark.parametrize("tolerance", [0, 1, 49])
    def test_both_libraries_raise_key_error_outside_the_bound(
        self, container, tolerance
    ):
        """A bound below the distance refuses on both sides, and with the same exception type.

        Args:
            container: The opened store.
            tolerance: A bound below the snap's distance of 50.

        Test scenario:
            The docstring's claim is not merely "it raises" but "it raises `KeyError`, as it
            does in xarray". Asserting only pyramids would leave that half unchecked, so both
            calls are made and both are required to raise the same class — a caller's
            `except KeyError` has to work against either library.
        """
        with pytest.raises(KeyError):
            container.get_variable(VARIABLE).sel(
                pressure_level=NEAREST_REQUEST, method="nearest", tolerance=tolerance
            )

        with pytest.raises(KeyError):
            container.to_xarray()[VARIABLE].sel(
                pressure_level=NEAREST_REQUEST, method="nearest", tolerance=tolerance
            )

    def test_a_zero_bound_behaves_the_same_on_both_sides(self, container):
        """``tolerance=0`` accepts an exact coordinate and refuses a near one, in both.

        Test scenario:
            Zero is where an implementation is most likely to diverge — read as "no bound" it
            would accept everything, and read as "positive bound required" it would reject the
            exact request. Both outcomes are checked against xarray rather than assumed.
        """
        variable = container.get_variable(VARIABLE)
        exported_variable = container.to_xarray()[VARIABLE]

        pyramids_exact = variable.sel(
            pressure_level=NEAREST_MATCH, method="nearest", tolerance=0
        )
        xarray_exact = exported_variable.sel(
            pressure_level=NEAREST_MATCH, method="nearest", tolerance=0
        )
        assert float(pyramids_exact.get_dimension_values("pressure_level")[0]) == float(
            xarray_exact.pressure_level.values
        ), "an exact request under a zero bound must land on the same level in both"

        with pytest.raises(KeyError):
            variable.sel(pressure_level=850.5, method="nearest", tolerance=0)
        with pytest.raises(KeyError):
            exported_variable.sel(pressure_level=850.5, method="nearest", tolerance=0)

    def test_pyramids_refuses_a_bound_without_nearest_that_xarray_reads_as_a_miss(
        self, container
    ):
        """The one deliberate divergence: ``tolerance=`` alone is an argument error here.

        Test scenario:
            xarray accepts `tolerance=` without `method=`, then fails the lookup and reports a
            `KeyError` about the label — the bound was never applied. pyramids refuses the
            combination up front with a `ValueError` that says why. Recording the divergence
            here means it stays a decision rather than becoming a surprise.
        """
        with pytest.raises(ValueError, match="only meaningful with method='nearest'"):
            container.get_variable(VARIABLE).sel(
                pressure_level=NEAREST_REQUEST, tolerance=50
            )

        with pytest.raises(KeyError):
            container.to_xarray()[VARIABLE].sel(
                pressure_level=NEAREST_REQUEST, tolerance=50
            )
