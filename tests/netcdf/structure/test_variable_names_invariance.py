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

    What the classification reads is what the *file declares*, never what an
    array is called. That is why the GEOS row below excludes only `DQF` and
    `band_id`, and not its three `*_bounds` arrays: nothing in that store
    declares `bounds`, so they are data there while `cf__7v`'s `lat_bnds` --
    named by `lat:bounds` -- is not. `TestTheClassificationFollowsDeclarations`
    pins that difference with the declarations behind it, so the two rows do not
    read as the classification contradicting itself.
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


class TestTheClassificationFollowsDeclarations:
    """Two fixtures, opposite answers about a `*_bounds` array, one rule.

    `tests/netcdf/test_variable_names_order.py` asserts the GEOS fixture
    enumerates `["CMI", "time_bounds", "y_image_bounds", "x_image_bounds"]`,
    while `TestTheCfClassificationIsApplied` above asserts `cf__7v` leaves
    `lat_bnds` / `lon_bnds` / `time_bnds` out. Read side by side those look like
    the classification contradicting itself. They are the same rule applied to
    two files that declare different things, and the declarations are asserted
    here so a reader does not have to take that on trust.

    It also records a real property of these fixtures: `cf__9v` is a shrunken
    GOES granule whose scalar `t` coordinate -- the variable a full granule
    gives `bounds = "time_bounds"` -- is not in the store at all. A future
    fixture refresh that restores `t` would move `time_bounds` out of
    `variable_names`, and this test says so before the order test fails
    mysteriously.
    """

    GEOS = "cf__9v__1d7-2d2__geos__y-desc.nc"
    CF_BOUNDS = "cf__7v__1d3-2d3-3d1__y-asc.nc"

    @staticmethod
    def _bounds_declarations(dataset: NetCDF) -> dict[str, str]:
        """Map each array that declares a CF `bounds` attribute to its value."""
        rg = dataset._working_group()
        declared = {}
        for name in rg.GetMDArrayNames():
            attrs = {
                a.GetName(): a.ReadAsString()
                for a in rg.OpenMDArray(name).GetAttributes()
            }
            if "bounds" in attrs:
                declared[name] = attrs["bounds"]
        return declared

    def test_a_declared_bounds_array_is_not_data(self):
        """`lat:bounds = "lat_bnds"` is the reason `lat_bnds` is excluded.

        Test scenario:
            The declaration is asserted alongside the outcome, so the test
            fails loudly if a fixture edit removes the attribute rather than
            silently passing for the wrong reason.
        """
        dataset = NetCDF.read_file(str(DATA / self.CF_BOUNDS))

        declared = self._bounds_declarations(dataset)

        assert declared == {
            "lat": "lat_bnds",
            "lon": "lon_bnds",
            "time": "time_bnds",
        }, f"the fixture no longer declares its bounds: {declared}"
        assert dataset.variable_names == ["tos"], (
            f"a declared bounds array was enumerated: {dataset.variable_names}"
        )

    def test_an_undeclared_bounds_shaped_array_is_data(self):
        """Nothing in the GEOS store says `time_bounds` bounds anything.

        Test scenario:
            The name ends in `_bounds` and the array is not data in any
            physical sense, but CF is a declaration language: with no
            `bounds` attribute pointing at it, the only defensible answer is
            to treat it as data rather than guess from the spelling. The
            arrays that *are* declared -- `DQF` via `ancillary_variables`,
            `band_id` / `band_wavelength` via `coordinates` -- are excluded on
            the same fixture, which is what makes this a rule and not a gap.
        """
        dataset = NetCDF.read_file(str(DATA / self.GEOS))

        declared = self._bounds_declarations(dataset)

        assert declared == {}, f"the fixture now declares bounds: {declared}"
        assert dataset.variable_names == [
            "CMI",
            "time_bounds",
            "y_image_bounds",
            "x_image_bounds",
        ], f"the undeclared bounds arrays changed status: {dataset.variable_names}"
        assert {"DQF", "band_id", "band_wavelength"} <= set(
            dataset._readable_variable_names()
        ) - set(dataset.variable_names), (
            "the declared non-data arrays should still be excluded"
        )

    def test_the_xarray_export_promotes_the_same_arrays_and_no_others(self):
        """The classification's other consumer, on the fixture nothing covered.

        Test scenario:
            `to_xarray` promotes CF non-data arrays out of `data_vars` into
            `coords`, and that promotion was only ever checked on `cf__7v`,
            where all three candidates are declared bounds. On GEOS the split
            runs the other way for three of the five, so this is where a
            promotion keyed on the *name* rather than the declaration would
            show up: `band_id` and `band_wavelength` are declared coordinates
            and must move, the three `*_bounds` are declared nothing and must
            stay.
        """
        dataset = NetCDF.read_file(str(DATA / self.GEOS))

        exported = dataset.to_xarray()

        assert {"band_id", "band_wavelength"} <= set(exported.coords), (
            f"declared coordinates were not promoted: {sorted(exported.coords)}"
        )
        assert {"time_bounds", "x_image_bounds", "y_image_bounds"} <= set(
            exported.data_vars
        ), f"undeclared arrays were promoted: {sorted(exported.data_vars)}"
        assert {"CMI", "DQF"} <= set(exported.data_vars), (
            f"a gridded array went missing: {sorted(exported.data_vars)}"
        )


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

    One condition, not two. The candidate set is the *readable* names, so an
    array the enumeration leaves out still survives the operation, and the only
    thing removed from it is what the operation is itself transforming. There
    is no second, dimension-based filter: an earlier one dropped anything
    indexed by a dimension the operation reshapes, and commit `0c6aa32b5`
    removed it because it took real data with it (see
    `test_a_bounds_array_is_carried_rather_than_dropped` below, and
    `TestOperationsCarryTheArraysTheyDoNotTransform` above, for the two halves
    of that trade).
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

    def test_the_carried_bounds_does_not_drag_the_source_grid_in_with_it(self):
        """The carry stops at the spatial axes, and this is why.

        Test scenario:
            An earlier version of this test asserted the opposite -- that the
            source `lat` axis is carried in beside its bounds array, so the
            stale 170 rows have something to be read against. That is a real
            benefit, and it cost far more than it was worth:
            :meth:`NetCDF._compute_geotransform` reads a `lon`/`lat` pair in
            preference to the stored transform, so a carried source axis made
            the *result* report the source grid. On `to_crs(3857)` the
            container answered a geotransform and a bbox in degrees while
            declaring metres.

            So the bounds array is still carried -- dropping it loses real data
            -- and its spatial axis is not. `lat_bnds` sits on a bare `lat`
            dimension, which netCDF allows, and the result's own `x`/`y`
            describe the grid it actually has.
        """
        dataset = NetCDF.read_file(str(DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"))

        cropped = dataset.crop(bbox=[20, -40, 120, 40], epsg=4326)
        rg = cropped._working_group()
        names = set(rg.GetMDArrayNames() or [])
        sizes = {
            name: [d.GetSize() for d in rg.OpenMDArray(name).GetDimensions()]
            for name in names
        }

        assert sizes["tos"][1] == 75, f"the crop should reshape tos: {sizes['tos']}"
        assert sizes["lat_bnds"] == [170, 2], (
            f"the bounds array is carried verbatim, on the source axis: {sizes}"
        )
        assert "lat" not in names, (
            "the source lat axis must not be carried: the container reads a "
            f"lat/lon pair as its own grid. arrays: {sorted(names)}"
        )

    def test_the_result_reports_its_own_grid_after_a_reprojection(self):
        """The consequence the carry had, stated where it can fail.

        Test scenario:
            `to_crs(3857)` returns a container declaring EPSG:3857. Its
            geotransform must be in metres. Carrying the source's `lat`/`lon`
            made it report the source's degrees -- silently, since the CRS said
            metres and only the numbers disagreed. A metre-scale pixel width is
            the cheapest thing to assert that a degrees answer cannot satisfy.
        """
        dataset = NetCDF.read_file(str(DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"))

        reprojected = dataset.to_crs(3857)
        pixel_width = abs(reprojected.geotransform[1])

        assert reprojected.epsg == 3857, f"the fixture changed: {reprojected.epsg}"
        assert pixel_width > 1000, (
            f"an EPSG:3857 container reported a pixel width of {pixel_width}, "
            "which is the source's degrees rather than metres"
        )

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
    list: not transformed, and -- under the dimension filter the carry rule
    applied at the time, since removed -- not carried either. A container-wide
    `to_crs` dropped them from the output entirely, with no warning.
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
