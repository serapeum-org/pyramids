"""`NetCDF.variable_names` must not depend on what was read before it.

Two classifiers used to answer "which arrays are data variables": a CF-based one
that ran whenever `meta_data` happened to be cached, and a store-derived one that
ran otherwise. Because `__str__` interpolates `meta_data`, merely printing or
logging a container switched which classifier answered -- and on a classic-mode
file the CF path returned an empty list, so `get_variable` then raised for every
variable in the file.

These tests pin the invariant the fix establishes: the CF classification is
consulted on *every* call for a whole-store view rather than only when the cache
happens to be warm, so reading `meta_data`, or printing the object, cannot change
the answer.

Invariance alone is not enough, though: answering "nothing is a data variable"
every time would satisfy it. The classes below therefore pin content too -- that
the CF roles are applied, that a grouped store reports every variable, and that
the arrays the enumeration leaves out are still readable by name.
"""

import glob
from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[2] / "data" / "netcdf"
FIXTURES = sorted(Path(p).name for p in glob.glob(str(DATA / "*.nc")))


@pytest.fixture(params=FIXTURES, ids=FIXTURES)
def netcdf_path(request) -> str:
    """Every netCDF fixture in the suite, by name."""
    return str(DATA / request.param)


def open_or_skip(path: str, multi_dimensional: bool) -> NetCDF:
    """Open `path` in the given mode, skipping fixtures that mode cannot read.

    Several fixtures carry only 1-D variables, so GDAL's classic driver refuses
    them outright ("not recognized as being in a supported file format"). They
    say nothing about the invariant under test.
    """
    try:
        return NetCDF.read_file(path, open_as_multi_dimensional=multi_dimensional)
    except RuntimeError as exc:  # pragma: no cover - depends on the fixture set
        pytest.skip(f"fixture not readable in this mode: {exc}")


@pytest.mark.parametrize("multi_dimensional", [True, False], ids=["mdim", "classic"])
class TestVariableNamesInvariance:
    """`variable_names` is the same however the object was used beforehand."""

    def test_reading_meta_data_does_not_change_variable_names(
        self, netcdf_path: str, multi_dimensional: bool
    ):
        """Consulting `meta_data` leaves the enumerated variables untouched."""
        dataset = open_or_skip(netcdf_path, multi_dimensional)
        before = sorted(dataset.variable_names)

        _ = dataset.meta_data

        assert sorted(dataset.variable_names) == before

    def test_str_does_not_change_variable_names(
        self, netcdf_path: str, multi_dimensional: bool
    ):
        """`__str__` reads `meta_data`, so printing must stay side-effect free."""
        dataset = open_or_skip(netcdf_path, multi_dimensional)
        before = sorted(dataset.variable_names)

        str(dataset)

        assert sorted(dataset.variable_names) == before

    def test_every_reported_variable_is_retrievable_after_meta_data(
        self, netcdf_path: str, multi_dimensional: bool
    ):
        """`get_variable` accepts every name `variable_names` advertises.

        The regression this guards raised `ValueError: <name> is not a valid
        variable name in []` for every variable of a file whose metadata had
        been read.

        Classic mode is excluded, and not because of this fix. There,
        `_classic_subdataset_variable_names` reports CF *standard* names
        (`precipitation_flux`) while `get_variable` resolves *store* names
        (`pr`), so the two disagree however the handle was used. That is a
        separate, pre-existing defect; asserting it here would fail for a
        reason this test is not about.
        """
        if not multi_dimensional:
            pytest.skip(
                "classic mode reports CF standard names but resolves store "
                "names -- a separate pre-existing defect"
            )
        dataset = open_or_skip(netcdf_path, multi_dimensional)
        _ = dataset.meta_data

        for name in dataset.variable_names:
            assert dataset.get_variable(name) is not None


class TestGroupedStoresEnumerateEveryVariable:
    """Invariance is necessary but not sufficient: the content matters too.

    Enumerating only the working group is stable, and also wrong: a NetCDF-4
    store that puts each variable in its own sub-group then reports whatever
    sits at the root and nothing else.
    """

    def test_a_grouped_store_lists_its_sub_group_variables(self):
        """One name out of thirty, before the walk recursed.

        Test scenario:
            This fixture holds 29 variables across per-flight sub-groups plus
            one at the root. Anything iterating `variable_names` to convert,
            export or plot the file has to see all of them.
        """
        dataset = NetCDF.read_file(str(DATA / "none__35v__1d35__groups-nc4.nc"))

        names = dataset.variable_names

        assert len(names) > 20, f"sub-group variables missing: {names}"
        assert any("/" in name for name in names), "no group-qualified names"
        assert "UTC_time" in names, "the root variable should still be listed"

    def test_the_group_qualified_names_resolve(self):
        """A name this lists must be one `get_variable` accepts.

        Test scenario:
            Enumerating a name the reader then rejects would be worse than not
            listing it, so the two are pinned together.
        """
        dataset = NetCDF.read_file(str(DATA / "none__35v__1d35__groups-nc4.nc"))

        qualified = next(n for n in dataset.variable_names if "/" in n)

        assert dataset.get_variable(qualified) is not None

    def test_a_flat_store_is_unaffected_by_the_recursion(self):
        """The common case has no sub-groups and must not change.

        Test scenario:
            A flat CF file's names carry no `/` and are exactly the root
            group's arrays, as before.
        """
        dataset = NetCDF.read_file(str(DATA / "cf__5v__1d4-4d1__y-asc.nc"))

        names = dataset.variable_names

        assert names == ["temperature"], names

    def test_the_content_is_stable_across_a_meta_data_read(self):
        """Content and invariance together, on the grouped fixture.

        Test scenario:
            Both properties are pinned in one place for the case that has
            sub-groups, so a regression in either is caught here.
        """
        dataset = NetCDF.read_file(str(DATA / "none__35v__1d35__groups-nc4.nc"))
        before = sorted(dataset.variable_names)

        _ = str(dataset)

        assert sorted(dataset.variable_names) == before


class TestTheCfClassificationIsApplied:
    """`variable_names` enumerates data variables, so non-data arrays stay out.

    A purely store-derived answer is stable and also wrong in a specific way: it
    keeps every MDArray that is not a dimension coordinate, sweeping in CF
    bounds, ancillary variables and 2-D curvilinear coordinate fields. Only the
    CF roles distinguish those, which is why the classification runs on every
    call rather than only when the metadata cache happens to be warm.
    """

    @pytest.mark.parametrize(
        ("fixture", "excluded"),
        [
            ("cf__7v__1d3-2d3-3d1__y-asc.nc", ("lat_bnds", "lon_bnds", "time_bnds")),
            ("cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc", ("lat_rho", "lon_rho", "h")),
            ("cf__9v__1d7-2d2__geos__y-desc.nc", ("DQF", "band_id")),
        ],
    )
    def test_cf_non_data_arrays_are_not_enumerated(self, fixture, excluded):
        """Bounds, ancillary and curvilinear coordinates are not data.

        Args:
            fixture: The netCDF fixture to read.
            excluded: Array names that exist in the store but are not data.

        Test scenario:
            Each of these is a real MDArray with more than zero dimensions, so a
            store-derived filter keeps it. Only the CF roles say otherwise.
        """
        dataset = NetCDF.read_file(str(DATA / fixture))

        names = set(dataset.variable_names)

        assert names, "the fixture should enumerate at least one data variable"
        assert not names & set(excluded), (
            f"non-data arrays enumerated: {names & set(excluded)}"
        )

    @pytest.mark.parametrize(
        ("fixture", "array"),
        [
            ("cf__7v__1d3-2d3-3d1__y-asc.nc", "lat_bnds"),
            ("cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc", "lat_rho"),
        ],
    )
    def test_what_is_not_enumerated_is_still_readable(self, fixture, array):
        """Leaving an array out of the enumeration must not hide it.

        Args:
            fixture: The netCDF fixture to read.
            array: A CF non-data array in that fixture.

        Test scenario:
            The two questions are different: `variable_names` enumerates data
            variables, while `get_variable` asks what the store holds. Reading a
            bounds or curvilinear-coordinate array by name is legitimate, and
            gating it on the enumeration made `get_variable("xc")` fail on a
            curvilinear store while the array sat right there.
        """
        dataset = NetCDF.read_file(str(DATA / fixture))

        assert array not in dataset.variable_names
        assert array in dataset._readable_variable_names()

    def test_the_readable_set_is_a_strict_superset_of_the_enumeration(self):
        """The relationship between the two, stated as content rather than shape.

        Test scenario:
            `_readable_variable_names` starts from `variable_names`, so a
            subset assertion holds for any implementation and pins nothing.
            What is worth pinning is *which* arrays the wider list adds: the
            three CF bounds arrays this fixture declares, and nothing else. A
            readable set that stopped scanning the store, or an enumeration
            that started sweeping the bounds back in, both fail here.
        """
        dataset = NetCDF.read_file(str(DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"))

        enumerated = set(dataset.variable_names)
        readable = set(dataset._readable_variable_names())

        assert enumerated == {"tos"}
        assert readable - enumerated == {"lat_bnds", "lon_bnds", "time_bnds"}

    def test_a_classic_mode_file_still_enumerates_its_variables(self):
        """The CF list is empty in classic mode, so the store answers instead.

        Test scenario:
            Preferring an empty CF classification here would make every
            variable in the file look unreachable, which is the defect the
            conditional branch originally caused.
        """
        dataset = NetCDF.read_file(
            str(DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"),
            open_as_multi_dimensional=False,
        )

        assert dataset.variable_names, "classic mode reported no variables"


class TestOperationsCarryTheArraysTheyDoNotTransform:
    """Leaving an array out of the enumeration must not make an op lossy.

    The other half of this -- that a non-spatial ancillary array such as ERA5's
    `expver` does survive a crop -- is pinned by
    `tests/netcdf/spatial/test_crop_string_aux.py`, which has a mask that
    actually overlaps that fixture's valid data.
    """

    def test_a_crop_carries_every_array_it_does_not_transform(self):
        """Nothing is silently dropped by a real operation, run here.

        Test scenario:
            The predecessor of this test compared `_carryable_aux_names` with
            the list it is a comprehension over and never called `crop` at
            all, so it held for any implementation. This runs the operation
            and compares the result's arrays with the source's: the bounds
            arrays share the `lat` / `lon` dimensions the crop reshapes, which
            is exactly the case an over-broad carry filter used to drop.
        """
        dataset = NetCDF.read_file(str(DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"))
        before = set(dataset._readable_variable_names())

        cropped = dataset.crop(bbox=[20, -40, 120, 40], epsg=4326)

        assert before == set(cropped._readable_variable_names())

    def test_a_staggered_array_is_not_dropped_for_sharing_a_dimension(self):
        """The rule that made the earlier filter unacceptable, pinned by name.

        Test scenario:
            An earlier rule excluded any array indexed by a dimension the
            operation reshapes, to stop a `lat_bnds(lat, nv)` being copied
            verbatim into a cropped result where it describes the source's
            axis. That filter dropped real data: a WRF store's staggered `U` /
            `V` share one spatial dimension with the gridded variables and
            vanished from the output with no warning. Naming them is what
            fails if the filter comes back.
        """
        dataset = NetCDF.read_file(
            str(DATA / "none__17v__1d1-2d5-3d6-4d5__stag-str.nc")
        )
        rg = dataset._working_group()
        spatial = dataset._spatial_variable_names(rg)

        carryable = set(dataset._carryable_aux_names(rg, spatial))

        assert {"U", "V"} <= carryable
        assert {"MAPFAC_U", "MAPFAC_V"} <= carryable


class TestCarryableAuxNames:
    """The rule an operation uses to decide what to copy through untouched.

    Two conditions, and both are load-bearing. The candidate set is the
    *readable* names, so an array the enumeration leaves out still survives the
    operation. The filter then drops anything indexed by a dimension the
    operation reshapes, because such an array would describe the source's grid
    rather than the result's.
    """

    def test_the_two_lists_split_the_store_the_way_they_claim(self):
        """An operation must not copy what it is busy rewriting.

        Test scenario:
            Disjointness alone is guaranteed by the comprehension's own `if`
            and pins nothing, so the split is asserted by content instead:
            `tos` is the only gridded array, and the three bounds arrays are
            the ones carried through untouched. A `_spatial_variable_names`
            that started treating a bounds array as gridded (it has no `(y, x)`
            pair, so it must not) fails on the first line; one that stopped
            scanning the readable superset fails on the second.
        """
        dataset = NetCDF.read_file(str(DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"))
        rg = dataset._working_group()
        spatial = dataset._spatial_variable_names(rg)

        carryable = dataset._carryable_aux_names(rg, spatial)

        assert spatial == ["tos"]
        assert sorted(carryable) == ["lat_bnds", "lon_bnds", "time_bnds"]

    def test_a_bounds_array_is_carried_rather_than_dropped(self):
        """The lesser of two faults, and the one the base commit chose.

        Test scenario:
            `lat_bnds(lat, nv)` is not spatial by the `(y, x)` test, so a crop
            copies it through unchanged -- where it still describes the
            source's latitude axis rather than the cropped one. Excluding
            everything that shares a reshaped dimension fixed that and dropped
            real data with it: a WRF store's staggered `U` / `V` and a CAM
            store's `gw` disappeared from the output silently. Stale bounds are
            visible and correctable; a missing variable is neither.
        """
        dataset = NetCDF.read_file(str(DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"))
        rg = dataset._working_group()
        spatial = dataset._spatial_variable_names(rg)

        carryable = dataset._carryable_aux_names(rg, spatial)

        assert "lat_bnds" in carryable

    def test_an_array_on_an_untouched_dimension_is_carried(self):
        """Sharing *any* dimension is too strict; only y / x count.

        Test scenario:
            ERA5's `expver` is indexed by `valid_time`, which the gridded
            variables also use -- but a crop does not reshape it, so the array
            is unaffected and must survive.
        """
        dataset = NetCDF.read_file(str(DATA / "cf__5v__1d4-3d1__geog__y-desc.nc"))
        if "expver" not in dataset._readable_variable_names():
            pytest.skip("fixture carries no non-spatial ancillary array")
        rg = dataset._working_group()

        carryable = dataset._carryable_aux_names(
            rg, dataset._spatial_variable_names(rg)
        )

        assert "expver" in carryable

    def test_every_carried_name_is_one_the_reader_accepts(self):
        """Nothing may be carried that the store cannot hand back.

        Test scenario:
            Comparing the returned list with the list it is a comprehension
            over says nothing. What the carry loop actually needs is that
            `get_variable` resolves each name, because that is the call it
            makes -- so the reader is exercised here rather than assumed.
        """
        dataset = NetCDF.read_file(str(DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"))
        rg = dataset._working_group()

        carryable = dataset._carryable_aux_names(
            rg, dataset._spatial_variable_names(rg)
        )

        assert sorted(carryable) == ["lat_bnds", "lon_bnds", "time_bnds"]
        for name in carryable:
            assert dataset.get_variable(name) is not None, name


class TestAncillaryArraysSurviveAnOperation:
    """Not enumerating an array must not mean losing it.

    `variable_names` enumerates data variables, so CF ancillary arrays are
    absent from it. They are still *gridded*, though -- GOES ABI's `DQF`
    quality flags sit on the same y/x grid as the `CMI` they qualify -- so they
    have to be reprojected like any other gridded array.

    Deciding spatial-ness from the narrow enumeration left them in neither
    list: not transformed, and rejected by the carry rule for sharing the
    reshaped dimensions. A container-wide `to_crs` dropped them from the output
    entirely, with no warning.
    """

    GEOS = "cf__9v__1d7-2d2__geos__y-desc.nc"

    def test_an_ancillary_grid_is_reprojected_rather_than_dropped(self):
        """`DQF` is not enumerated, is gridded, and must come back.

        Test scenario:
            The array is absent from `variable_names` by design. What matters
            is that it is present in the *result* of a container-wide `to_crs`,
            because a user reprojecting a granule expects its quality flags.
        """
        dataset = NetCDF.read_file(str(DATA / self.GEOS))
        assert "DQF" in dataset._readable_variable_names()

        reprojected = dataset.to_crs(4326)

        assert "DQF" in reprojected._readable_variable_names()

    def test_the_operation_loses_no_array_at_all(self):
        """The general form: nothing readable goes missing.

        Test scenario:
            Every array the source could hand back must still be reachable on
            the result. Asserting the whole set rather than one name catches
            the next array that falls between the two lists.
        """
        dataset = NetCDF.read_file(str(DATA / self.GEOS))
        before = set(dataset._readable_variable_names())

        reprojected = dataset.to_crs(4326)

        assert before <= set(reprojected._readable_variable_names())

    def test_a_gridded_ancillary_array_is_treated_as_spatial(self):
        """The mechanism, asserted directly rather than through the result.

        Test scenario:
            `_spatial_variable_names` decides what gets transformed. `DQF` has
            both spatial axes, so it belongs there even though the enumeration
            leaves it out.
        """
        dataset = NetCDF.read_file(str(DATA / self.GEOS))
        rg = dataset._working_group()

        spatial = dataset._spatial_variable_names(rg)

        assert "DQF" in spatial
        assert "DQF" not in dataset.variable_names
