"""`isel`, `sel` and `squeeze` on a container, and the drop warning they share.

All three used to refuse a container: the variable route reads the receiver's own
`_band_dim_names`, and a container tracks none. They now take the dimension from the **store**
and cut through the shared along-dimension route, so they share one gate
(`_container_band_dimensions`, `_assert_container_dimension`) and one cut
(`_cut_container_along`) — which is what the parametrised tests here pin, so the three cannot
drift in what they refuse.

The auxiliary-drop warning gets its own class. It is the contract every length-changing
container operation shares, and the drop tests in `test_calendars.py` silence it so they can
assert values — silencing a warning in every test is how a warning stops being checked at all.
"""

import gc
import pathlib
import warnings

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.cf import write_attributes_to_md_array
from pyramids.netcdf.engines._along_dim import _TakeSteps
from pyramids.netcdf.netcdf import Container

NY, NX = 2, 3
GEO = (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)
UNITS = "days since 2001-01-01"


def _geo_ref() -> GeoReference:
    """The one grid every cube in this module shares.

    Returns:
        GeoReference: A 2x3 grid in EPSG:4326.
    """
    return GeoReference(geo=GEO, epsg=4326)


def _cube(stamps: list[float], *, name: str = "v") -> NetCDF:
    """A one-variable container on a `time` axis stamped `stamps`.

    Args:
        stamps: The time coordinates, one per step.
        name: The variable's name.

    Returns:
        NetCDF: The container.
    """
    planes = np.stack([np.full((NY, NX), float(index)) for index in range(len(stamps))])
    return NetCDF.from_array(
        planes,
        geo_ref=_geo_ref(),
        variable_name=name,
        dims=ExtraDimensions(name="time", values=list(stamps)),
    )


def _store_with_an_auxiliary_on_time(stamps: list[float] | None = None) -> gdal.Dataset:
    """A store whose `qc` auxiliary spans `time` while `t` is the gridded variable.

    `qc` is not gridded, so it is an *auxiliary*: an operation that changes `time`'s length
    cannot carry it and must say so. Built by hand because the sample files' auxiliaries sit on
    `bnds` rather than on the axis being cut.

    The default stamps 58/59/60 are 28 February, 29 February and 1 March on `all_leap`, so a
    move to `noleap` — which has no 29 February — drops exactly one step.

    Args:
        stamps: The time coordinates; the three above when `None`. A single stamp gives a
            one-step axis, which is what `squeeze` needs with the auxiliary still present.

    Returns:
        gdal.Dataset: The in-memory multidimensional store.
    """
    stamps_values = [58.0, 59.0, 60.0] if stamps is None else list(stamps)
    steps = len(stamps_values)
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
    root = store.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    lat = root.CreateDimension("lat", gdal.DIM_TYPE_HORIZONTAL_Y, "NORTH", NY)
    lon = root.CreateDimension("lon", gdal.DIM_TYPE_HORIZONTAL_X, "EAST", NX)
    time = root.CreateDimension("time", "", "", steps)
    for dim, values in ((lat, [1.5, 0.5]), (lon, [0.5, 1.5, 2.5])):
        coordinate = root.CreateMDArray(dim.GetName(), [dim], f64)
        coordinate.Write(np.array(values))
        dim.SetIndexingVariable(coordinate)
    axis = root.CreateMDArray("time", [time], f64)
    axis.Write(np.array(stamps_values))
    time.SetIndexingVariable(axis)
    write_attributes_to_md_array(axis, {"units": UNITS, "calendar": "all_leap"})
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(4326)
    gridded = root.CreateMDArray("t", [time, lat, lon], f64)
    gridded.Write(np.arange(float(steps) * NY * NX).reshape(steps, NY, NX))
    gridded.SetSpatialRef(reference)
    auxiliary = root.CreateMDArray("qc", [time], f64)
    auxiliary.Write(np.arange(1.0, steps + 1.0))
    return store


class TestTheAuxiliaryDropWarning:
    """An operation that shortens a container's axis cannot carry an auxiliary spanning it."""

    def test_convert_calendar_warns_and_names_what_it_dropped(self):
        """Dropping a date shortens the axis, so the auxiliary goes with a warning.

        Test scenario:
            A container whose `qc` spans `time` is moved from `all_leap` to `noleap`. The
            warning names the member, the auxiliary and the dimension, and `qc` is absent from
            the result.
        """
        container = Container(_store_with_an_auxiliary_on_time())
        assert "qc" in container.variable_names, "precondition: qc is present"

        with pytest.warns(UserWarning) as caught:
            converted = container.convert_calendar("noleap")

        message = str(caught[0].message)
        assert "convert_calendar()" in message, f"should name the member: {message}"
        assert "qc" in message, f"should name the auxiliary: {message}"
        assert "'time'" in message, f"should name the dimension: {message}"
        assert "qc" not in converted.variable_names, (
            f"the auxiliary should be gone, got {converted.variable_names}"
        )

    def test_the_gridded_variable_survives_the_drop(self):
        """Only the auxiliary is dropped; the data variable is cut, not discarded.

        Test scenario:
            The same conversion leaves `t` with two of its three steps.
        """
        container = Container(_store_with_an_auxiliary_on_time())

        with pytest.warns(UserWarning):
            converted = container.convert_calendar("noleap")

        sizes = converted.get_variable("t")._band_dim_sizes
        assert sizes == (2,), f"two steps should survive, got {sizes!r}"

    def test_a_conversion_that_drops_nothing_carries_the_auxiliary(self):
        """The other half of the rule: an unchanged length keeps the auxiliary.

        Test scenario:
            `align_on="year"` drops no step, so `qc` is carried over and nothing warns —
            asserted by promoting the warning to an error.
        """
        container = Container(_store_with_an_auxiliary_on_time())

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            converted = container.convert_calendar("noleap", align_on="year")

        assert "qc" in converted.variable_names, (
            f"an unchanged length should carry it, got {converted.variable_names}"
        )

    def test_container_squeeze_warns_for_the_same_reason(self):
        """`squeeze` collapses the axis, which is a length change too.

        Test scenario:
            A container whose `time` is one step long, with `qc` spanning it, warns on squeeze
            and names `squeeze()`. Built at one step rather than cut down to it, because the
            cut would already have dropped `qc` and left nothing for squeeze to drop.
        """
        container = Container(_store_with_an_auxiliary_on_time([58.0]))
        assert "qc" in container.variable_names, "precondition: qc is present"

        with pytest.warns(UserWarning) as caught:
            squeezed = container.squeeze("time")

        assert "squeeze()" in str(caught[0].message), (
            f"should name squeeze: {caught[0].message}"
        )
        assert squeezed.get_variable("t")._band_dim_names == (), (
            "the axis should be gone from the layout"
        )


class TestTheThreeShareOneGate:
    """What `isel`, `sel` and `squeeze` refuse on a container, refused the same way."""

    @pytest.mark.parametrize("member", ["isel", "sel", "squeeze"])
    def test_a_container_with_no_band_dimension_is_refused(self, member: str):
        """All three need a non-spatial axis, and say so in the same words.

        Args:
            member: The member under test.

        Test scenario:
            A plain 2-D container declares only `y` / `x`, so each member refuses it by name
            rather than failing further in.
        """
        flat = NetCDF.from_array(
            np.ones((NY, NX)), geo_ref=_geo_ref(), variable_name="v"
        )
        call = {
            "squeeze": lambda: flat.squeeze(),
            "isel": lambda: flat.isel(time=0),
            "sel": lambda: flat.sel(time=0.0),
        }[member]

        with pytest.raises(ValueError, match="needs a non-spatial dimension"):
            call()

    @pytest.mark.parametrize("member", ["isel", "sel", "squeeze"])
    def test_a_spatial_axis_is_refused_by_all_three(self, member: str):
        """The `(y, x)` plane is pinned by the geotransform on every route.

        Args:
            member: The member under test.

        Test scenario:
            Naming `y` is refused, and the message points at the operations that do change the
            grid.
        """
        container = _cube([0.0, 1.0, 2.0])
        call = {
            "squeeze": lambda: container.squeeze("y"),
            "isel": lambda: container.isel(y=0),
            "sel": lambda: container.sel(y=0.0),
        }[member]

        with pytest.raises(ValueError, match="spatial axis"):
            call()

    @pytest.mark.parametrize("member", ["isel", "sel"])
    def test_an_unknown_name_lists_the_band_dimensions(self, member: str):
        """The refusal has to be actionable, which the pre-change one was not.

        Args:
            member: The selector under test.

        Test scenario:
            A name that is no dimension at all is refused, and the message lists the ones that
            are rather than claiming the receiver tracks none.
        """
        container = _cube([0.0, 1.0, 2.0])
        call = {
            "isel": lambda: container.isel(nope=0),
            "sel": lambda: container.sel(nope=0.0),
        }[member]

        with pytest.raises(ValueError, match="not a band dimension of this container"):
            call()


class TestSeveralKeywordsCompose:
    """Each keyword narrows a different axis of the same band grid."""

    def _two_axis_container(self) -> NetCDF:
        """A `(time: 3, level: 2)` container.

        Returns:
            NetCDF: The container.
        """
        planes = np.arange(3.0 * 2 * NY * NX).reshape(3, 2, NY, NX)
        return NetCDF.from_array(
            planes,
            geo_ref=_geo_ref(),
            variable_name="v",
            dims=ExtraDimensions(
                dims=[("time", [0.0, 1.0, 2.0]), ("level", [10.0, 20.0])]
            ),
        )

    def test_isel_applies_both_keywords(self):
        """Neither keyword is lost and neither is applied twice.

        Test scenario:
            `(3, 2)` cut by `time=[0, 2], level=[1]` comes back `(2, 1)`.
        """
        sizes = (
            self._two_axis_container()
            .isel(time=[0, 2], level=[1])
            .get_variable("v")
            ._band_dim_sizes
        )

        assert sizes == (2, 1), f"both keywords should apply, got {sizes!r}"

    def test_sel_applies_both_keywords(self):
        """The label route composes the same way.

        Test scenario:
            The equivalent labels give the same `(2, 1)`.
        """
        sizes = (
            self._two_axis_container()
            .sel(time=[0.0, 2.0], level=20.0)
            .get_variable("v")
            ._band_dim_sizes
        )

        assert sizes == (2, 1), f"both keywords should apply, got {sizes!r}"

    def test_a_variable_without_the_dimension_is_carried_over(self):
        """A container's other variables must survive a cut that does not touch them.

        Test scenario:
            A second variable spanning only `level` is carried through a cut on `time`, with
            its own axis intact.
        """
        container = self._two_axis_container()
        donor = NetCDF.from_array(
            np.arange(2.0 * NY * NX).reshape(2, NY, NX),
            geo_ref=_geo_ref(),
            variable_name="u",
            dims=ExtraDimensions(name="level", values=[10.0, 20.0]),
        )
        container.set_variable("u", donor.get_variable("u"))

        cut = container.isel(time=[0])

        assert sorted(cut.variable_names) == ["u", "v"], (
            f"both variables should survive, got {cut.variable_names}"
        )
        assert cut.get_variable("u")._band_dim_sizes == (2,), (
            "the carried variable's own axis should be untouched"
        )


class TestTakeStepsOnAnUnlabelledAxis:
    """`_TakeSteps` keeps an unlabelled axis unlabelled, tested at the operation's level.

    Every public route arrives with coordinates — `from_array` numbers an axis given
    `values=None` — so a container cannot reach this branch. It still matters, because
    numbering the axis here would let `sel` match positions as if they were stamps, so it is
    exercised where it lives.
    """

    def test_an_axis_with_no_coordinates_stays_without_them(self):
        """A cut must not invent stamps for an axis that never had any.

        Test scenario:
            A variable whose `time` carries no coordinate values is cut by `_TakeSteps`; the
            result's dimension is still unlabelled rather than numbered `0..n-1`.
        """
        variable = _cube([0.0, 1.0, 2.0]).get_variable("v")
        variable._band_dim_values_map["time"] = None

        applied = _TakeSteps(kept=(0, 2)).apply(variable, variable, "time")

        assert applied.values_map["time"] is None, (
            f"an unlabelled axis should stay unlabelled, got {applied.values_map['time']!r}"
        )
        assert applied.values.shape[0] == 2, (
            f"two steps should survive, got {applied.values.shape!r}"
        )


def _grouped_store() -> gdal.Dataset:
    """A store whose arrays live in a sub-group, so the ROOT declares no dimensions.

    The shape that exposed review round 1's M4/M5: `nc.dimension_sizes` is `{}` on this root,
    because `inner` declares the dimensions, while `_apply_to_container` resolves them through
    `_working_group()` and finds them.

    Returns:
        gdal.Dataset: The in-memory multidimensional store.
    """
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
    inner = store.GetRootGroup().CreateGroup("inner")
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    lat = inner.CreateDimension("lat", gdal.DIM_TYPE_HORIZONTAL_Y, "NORTH", NY)
    lon = inner.CreateDimension("lon", gdal.DIM_TYPE_HORIZONTAL_X, "EAST", NX)
    time = inner.CreateDimension("time", "", "", 3)
    for dim, values in (
        (lat, [1.5, 0.5]),
        (lon, [0.5, 1.5, 2.5]),
        (time, [0.0, 1.0, 2.0]),
    ):
        coordinate = inner.CreateMDArray(dim.GetName(), [dim], f64)
        coordinate.Write(np.array(values))
        dim.SetIndexingVariable(coordinate)
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(4326)
    gridded = inner.CreateMDArray("t", [time, lat, lon], f64)
    gridded.Write(np.arange(3.0 * NY * NX).reshape(3, NY, NX))
    gridded.SetSpatialRef(reference)
    return store


class TestAHierarchicalStoresRoot:
    """The new routes must work where the pre-existing ones do (round 1, M4 and M5).

    On a grouped store's root `dimension_sizes` is empty, because the dimensions belong to the
    sub-group. The members added here first read that and so refused the store, claiming it
    "declares no dimensions" — untrue of the store, and inconsistent with `reduce` / `cumsum` /
    `diff` / `rolling`, which resolve the working group and work fine.
    """

    def test_the_root_really_declares_no_dimensions_of_its_own(self):
        """The precondition, so a later failure reads as a regression and not a bad fixture.

        Test scenario:
            `dimension_sizes` on the root is empty while the gridded variable has `time`.
        """
        root = Container(_grouped_store())

        assert root.dimension_sizes == {}, (
            f"the root should declare nothing of its own, got {root.dimension_sizes}"
        )

    @pytest.mark.parametrize(
        "member",
        [
            "isel",
            "sel",
            "squeeze",
            "cumulative",
            "differentiate",
            "integrate",
            "polyfit",
        ],
    )
    def test_every_new_member_works_on_the_root(self, member: str):
        """Each of them resolves the sub-group's dimensions rather than refusing.

        Args:
            member: The member under test.

        Test scenario:
            Called on the root of a grouped store, each answers instead of raising.
        """
        root = Container(_grouped_store())
        call = {
            "isel": lambda: root.isel(time=0),
            "sel": lambda: root.sel(time=1.0),
            "squeeze": lambda: root.squeeze(),
            "cumulative": lambda: root.cumulative("time").sum(),
            "differentiate": lambda: root.differentiate("time"),
            "integrate": lambda: root.integrate("time"),
            "polyfit": lambda: root.polyfit("time", 1),
        }[member]

        assert call() is not None, f"{member} should answer on a hierarchical root"

    def test_cumulative_accepts_exactly_what_cumsum_accepts(self):
        """The accessor must not be stricter than the member it forwards to.

        Two docstring claims rest on this — "the two spellings cannot drift" and "an accessor
        in hand is always one that will work" — and before the fix the accessor refused a
        dimension `cumsum` accepted.

        Test scenario:
            On a grouped root, `cumsum` and `cumulative(...).sum()` both answer and agree on the
            inventory they produce. The assertion stops at the inventory deliberately: how a
            rebuilt grouped store names and nests its variables is pre-existing behaviour and
            not what this finding was about.
        """
        root = Container(_grouped_store())

        direct = root.cumsum("time")
        through = root.cumulative("time").sum()

        assert direct.variable_names == through.variable_names, (
            f"the two spellings should answer the same inventory, got "
            f"{direct.variable_names} vs {through.variable_names}"
        )


def _store_without_a_time_coordinate() -> gdal.Dataset:
    """A store whose `time` dimension has **no** indexing variable at all.

    So no gridded variable carries stamps for it, which is the one case
    `_gridded_band_coordinates` falls through to `get_dimension_values` for.

    Returns:
        gdal.Dataset: The in-memory multidimensional store.
    """
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
    root = store.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    lat = root.CreateDimension("lat", gdal.DIM_TYPE_HORIZONTAL_Y, "NORTH", NY)
    lon = root.CreateDimension("lon", gdal.DIM_TYPE_HORIZONTAL_X, "EAST", NX)
    time = root.CreateDimension("time", "", "", 3)
    for dim, values in ((lat, [1.5, 0.5]), (lon, [0.5, 1.5, 2.5])):
        coordinate = root.CreateMDArray(dim.GetName(), [dim], f64)
        coordinate.Write(np.array(values))
        dim.SetIndexingVariable(coordinate)
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(4326)
    gridded = root.CreateMDArray("t", [time, lat, lon], f64)
    gridded.Write(np.arange(3.0 * NY * NX).reshape(3, NY, NX))
    gridded.SetSpatialRef(reference)
    return store


class TestAnAxisWithNoCoordinateArray:
    """The fallback in `_gridded_band_coordinates`, and what the members do with it."""

    def test_a_positional_cut_still_works_without_coordinates(self):
        """`isel` indexes positions, so it needs no stamps.

        Test scenario:
            A container whose `time` has no coordinate array is cut by position.
        """
        container = Container(_store_without_a_time_coordinate())

        cut = container.isel(time=[0, 2])

        assert cut.get_variable("t")._band_dim_sizes == (2,), (
            f"two steps should survive, got {cut.get_variable('t')._band_dim_sizes!r}"
        )

    def test_a_label_cut_is_refused_for_want_of_coordinates(self):
        """`sel` matches stamps, and there are none — the fallback finds nothing either.

        This is the path that reaches `_gridded_band_coordinates`' final
        `get_dimension_values` fallback: no gridded variable carries stamps for the axis.

        Test scenario:
            `sel(time=...)` is refused, pointing at `isel` instead.
        """
        container = Container(_store_without_a_time_coordinate())

        with pytest.raises(ValueError, match="isel"):
            container.sel(time=1.0)

    def test_a_numerical_member_is_refused_for_want_of_spacing(self):
        """A derivative has no meaning without coordinates to measure spacing from.

        Test scenario:
            `differentiate` is refused, asking for coordinates rather than for interpolation —
            the wording round 1's L4 corrected.
        """
        container = Container(_store_without_a_time_coordinate())

        with pytest.raises(ValueError, match="measure the spacing from"):
            container.differentiate("time")


class TestAPackedVariableKeepsItsUnits:
    """A container cut must answer physical units, not stored counts (round 2, C1).

    The round-1 perf fix swapped the container cut onto a raw GDAL band read, which applies no
    CF unpacking — so `container.isel(...)` answered the stored `int16` counts while
    `variable.isel(...)` answered kelvin. A shape-only assertion cannot see that, which is why
    these compare the two routes **numerically**. CF packing is the normal encoding for
    ERA5/CMIP-style stores, i.e. this module's audience.
    """

    PACKED = (
        pathlib.Path(__file__).parents[3]
        / "examples"
        / "data"
        / "netcdf"
        / "samples"
        / "cf__20v__1d3-3d17__y-desc.nc"
    )

    def test_the_fixture_really_is_packed(self):
        """The precondition, so a pass here cannot be vacuous.

        Test scenario:
            `p2t` declares a non-identity `(scale, offset)`.
        """
        variable = NetCDF.read_file(str(self.PACKED)).get_variable("p2t")

        scale, offset = variable._effective_packing(0)
        assert scale not in (None, 1.0) and offset not in (None, 0.0), (
            f"the fixture must be packed for this class to mean anything, got {scale}, {offset}"
        )

    def test_container_isel_agrees_with_variable_isel_numerically(self):
        """The two routes must answer the same numbers, not merely the same shape.

        Test scenario:
            The same positional cut through the container and through the variable match cell
            for cell. Before the fix the container answered the raw counts (6302.0 where the
            variable answered 273.9663 K).
        """
        through_variable = np.asarray(
            NetCDF.read_file(str(self.PACKED))
            .get_variable("p2t")
            .isel(time=[0, 2])
            .read_array()
        )
        through_container = np.asarray(
            NetCDF.read_file(str(self.PACKED))
            .isel(time=[0, 2])
            .get_variable("p2t")
            .read_array()
        )

        assert np.allclose(through_variable, through_container, equal_nan=True), (
            f"the two routes disagree: variable {through_variable[0, 0, :3]} vs container "
            f"{through_container[0, 0, :3]}"
        )

    def test_container_sel_agrees_too(self):
        """`sel` shares the same cut, so it shares the same exposure.

        Test scenario:
            A label cut through the container matches the equivalent positional cut through the
            variable.
        """
        stamps = NetCDF.read_file(str(self.PACKED)).get_dimension_values("time")
        through_variable = np.asarray(
            NetCDF.read_file(str(self.PACKED))
            .get_variable("p2t")
            .isel(time=[0])
            .read_array()
        )
        through_container = np.asarray(
            NetCDF.read_file(str(self.PACKED))
            .sel(time=float(stamps[0]))
            .get_variable("p2t")
            .read_array()
        )

        assert np.allclose(through_variable, through_container, equal_nan=True), (
            "sel on a container must answer physical units too"
        )

    def test_the_cut_result_is_in_the_same_units_as_its_declared_sentinel(self):
        """The raw read also paired counts with an *unpacked* sentinel.

        `_read_no_data` documents that it reads a variable unpacked, so a packed variable's
        fill cells hold `_FillValue * scale + offset`. Values in counts beside a sentinel in
        physical units means a packed variable's gaps stop being recognised.

        Test scenario:
            The cut's values span the same order of magnitude as the sentinel it declares —
            kelvin, not int16 counts.
        """
        cut = NetCDF.read_file(str(self.PACKED)).isel(time=[0]).get_variable("p2t")
        values = np.asarray(cut.read_array(), dtype="float64")
        finite = values[np.isfinite(values)]

        assert finite.size and 150.0 < float(finite.mean()) < 400.0, (
            f"a 2-metre temperature field should read in kelvin, got mean "
            f"{float(finite.mean()) if finite.size else 'nothing'}"
        )


class TestANoOpSqueezeKeepsAViewsIdentity:
    """`get_group(...).squeeze()` must stay in the group (round 2, H1).

    Round 1 replaced a `weakref.proxy` return with a fresh `Container` — and dropped the three
    fields `get_group` sets to identify a view, so the result was the **store root**: same type,
    wrong group, and a `get_variable` that fails on the root's group-qualified inventory. Round
    1's test used a flat container and could not see it.
    """

    def test_the_result_stays_inside_the_group(self):
        """The view's group path, inventory and parent pin all survive.

        Test scenario:
            A view on `inner` whose dimensions are all longer than one is squeezed; the result
            still reports the view's own single variable `t`, not the root's `inner/t`.
        """
        view = Container(_grouped_store()).get_group("inner")
        assert view.variable_names == ["t"], "precondition: the view sees its own name"

        result = view.squeeze()
        gc.collect()

        assert result._group_path == "inner", (
            f"the group path must survive, got {result._group_path!r}"
        )
        assert result.variable_names == ["t"], (
            f"the view's inventory must survive, got {result.variable_names}"
        )

    def test_the_result_is_still_usable(self):
        """The symptom a caller actually hits.

        Test scenario:
            `get_variable('t')` works on the result, where before it raised about the root's
            group-qualified inventory.
        """
        result = Container(_grouped_store()).get_group("inner").squeeze()
        gc.collect()

        assert result.get_variable("t")._band_dim_sizes == (3,), (
            "the squeezed view's variable must still be reachable by its own name"
        )
