"""The container behaves like a mapping, and answers to xarray's spellings.

`nc.variables` has been a mapping for a while; the container was not. `nc["t2m"]`,
`"t2m" in nc`, `len(nc)` and `list(nc)` all failed, which is the first thing a reader
arriving from xarray types. Every member added here delegates to `variables`, so what is
under test is that the delegation is *total* -- one enumeration, one refusal message, no
second list that can drift -- rather than that the mapping itself works, which
`test_variables_mapping_matches_accessor.py` already pins.

Two of those delegations are deliberate mismatches, and both are asserted rather than
assumed:

- `nc["nope"]` raises `KeyError` where `nc.get_variable("nope")` raises `ValueError`. The
  mapping protocol needs `KeyError` for `in`, `get` and `dict(nc)` to behave at all, so the
  two spellings of the same lookup cannot report the same miss the same way.
- `nc.dims` is a **mapping** of name to length, matching xarray, and is therefore *not*
  `dimension_names`, which is a list. That is the one alias in the set that does not alias
  the similarly-named member, so it is pinned from both directions.

The introspection trio is the other half: `dtypes`, `nbytes` and `info` all have to answer
without reading a pixel, or they are useless on the cubes that most need sizing. That is
asserted by making `read_array` raise.
"""

from __future__ import annotations

import io
from collections import abc
from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf import NetCDF
from pyramids.netcdf.labeled import LabeledArray
from pyramids.netcdf.netcdf import _variable_dtype, _variable_nbytes

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[2] / "data" / "netcdf"

#: Stores the sweep runs over, one per shape the new members have to survive: a plain 4-D CF
#: cube, a container declaring dimensions its variables do not all use, a 2-D packed store
#: whose variables report a renamed `subset_y_...` axis, a container holding only
#: `LabeledArray`s, a grouped store whose names carry a `group/var` path, and a curvilinear
#: store that reads more names than it enumerates.
SWEEP = (
    "cf__5v__1d4-4d1__y-asc.nc",
    "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc",
    "coards__4v__1d2-2d2__scaleoffset__y-asc.nc",
    "ugrid__6v__1d5-2d1.nc",
    "none__35v__1d35__groups-nc4.nc",
    "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc",
    "none__1v__1d1.nc",
)

CURVILINEAR = "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc"
MIXED_RANKS = "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
PLAIN = "cf__5v__1d4-4d1__y-asc.nc"
LABELLED_ONLY = "ugrid__6v__1d5-2d1.nc"


def open_store(name: str) -> NetCDF:
    """Open one of the fixture stores by file name.

    Args:
        name: The `.nc` file name under `tests/data/netcdf`.

    Returns:
        NetCDF: The opened container.
    """
    return NetCDF.read_file(str(DATA / name))


@pytest.fixture(params=SWEEP, ids=lambda name: name.split("__")[0] + "-" + name[:12])
def container(request: pytest.FixtureRequest) -> NetCDF:
    """Each store in `SWEEP`, opened fresh.

    Args:
        request: The pytest request carrying the parametrized file name.

    Returns:
        NetCDF: The opened container.
    """
    return open_store(request.param)


def forbid_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any raster read fail, so a test can assert nothing reads one.

    Args:
        monkeypatch: The patcher to install the exploding `read_array` through.
    """

    def explode(self: NetCDF, *args: object, **kwargs: object) -> None:
        raise AssertionError("read_array was called; this member must not read pixels")

    monkeypatch.setattr(NetCDF, "read_array", explode)


class TestTheContainerIsAMapping:
    """`__getitem__`, `in`, iteration and the `keys`/`values`/`items` trio."""

    def test_getitem_hands_back_what_the_variables_mapping_holds(
        self, container: NetCDF
    ):
        """Indexing is the mapping's lookup, not a second one.

        Args:
            container: One of the swept stores.

        Test scenario:
            Every name the container enumerates is fetched both ways. The two have to
            agree in type and in shape -- a separate lookup path would be free to resolve
            a grouped `group/var` name at the root, or to return the raw `LabeledArray`
            where the mapping returns a subset.
        """
        for name in container.variable_names:
            through_index = container[name]
            through_mapping = container.variables[name]

            assert type(through_index) is type(through_mapping)
            if isinstance(through_index, LabeledArray):
                assert np.array_equal(through_index.values, through_mapping.values)
            else:
                assert through_index.shape == through_mapping.shape

    def test_membership_follows_the_enumerated_names(self, container: NetCDF):
        """`in` answers for exactly the data variables.

        Args:
            container: One of the swept stores.

        Test scenario:
            Every enumerated name is a member and a name the file does not hold is not.
            The negative matters as much as the positive: a membership test that fell back
            to the accessor would answer `True` for a coordinate.
        """
        assert all(name in container for name in container.variable_names)
        assert "no-such-variable" not in container

    def test_a_readable_name_that_is_not_a_data_variable_is_not_a_member(self):
        """The asymmetry `variables` already has, seen from the container.

        Test scenario:
            `lat_rho` is the ROMS store's 2-D curvilinear latitude. `get_variable` reads
            it, `variable_names` does not list it, and so `"lat_rho" in nc` is `False`
            while `nc.get_variable("lat_rho")` returns an array. Indexing refuses it with
            the message the mapping already gives, which names the accessor that works --
            the whole point of delegating rather than raising a bare `KeyError`.
        """
        nc = open_store(CURVILINEAR)

        assert nc.get_variable("lat_rho") is not None
        assert "lat_rho" not in nc

        with pytest.raises(KeyError, match="is not a data variable") as excinfo:
            nc["lat_rho"]

        assert "get_variable('lat_rho')" in str(excinfo.value)

    def test_a_non_string_key_is_simply_absent(self, container: NetCDF):
        """`in` must not raise on a key that cannot name a variable.

        Args:
            container: One of the swept stores.

        Test scenario:
            `3 in nc` and `None in nc` are ordinary `False`, not a `TypeError`. Python
            calls `__contains__` with whatever it is given, and a mapping that raises
            there breaks `any(x in nc for x in mixed)`.
        """
        assert 3 not in container
        assert None not in container

    def test_iteration_yields_the_names_in_store_order(self, container: NetCDF):
        """`list(nc)` is `variable_names`, order included.

        Args:
            container: One of the swept stores.

        Test scenario:
            Order is asserted, not just membership: `variable_names` is documented as
            store order, and a set-based delegation would lose it silently.
        """
        assert list(container) == container.variable_names

    def test_len_counts_the_data_variables(self, container: NetCDF):
        """`len(nc)` matches what iteration yields.

        Args:
            container: One of the swept stores.

        Test scenario:
            On the curvilinear store the two could disagree in the direction that matters:
            the file holds six readable arrays and enumerates two.
        """
        assert len(container) == len(container.variable_names)

    def test_keys_values_and_items_agree_with_each_other(self, container: NetCDF):
        """The three views describe one set of variables.

        Args:
            container: One of the swept stores.

        Test scenario:
            `keys` is compared to the names, `items` to `keys` paired with `values`, and
            the lengths to `len(nc)`. `values` loads every variable, which is the only one
            of the three that reads.
        """
        assert container.keys() == container.variable_names
        assert len(container.values()) == len(container)
        assert [name for name, _ in container.items()] == container.keys()
        for (name, from_items), from_values in zip(
            container.items(), container.values(), strict=True
        ):
            assert type(from_items) is type(from_values) is type(container[name])

    def test_get_returns_the_default_rather_than_raising(self, container: NetCDF):
        """A miss through `get` is not an error.

        Args:
            container: One of the swept stores.

        Test scenario:
            The sentinel is returned for an absent name, `None` when no default is given,
            and a present name still resolves -- a `get` that swallowed every lookup would
            pass the first two assertions on its own.
        """
        sentinel = object()

        assert container.get("no-such-variable", sentinel) is sentinel
        assert container.get("no-such-variable") is None
        assert container.get(container.variable_names[0]) is not None

    def test_the_container_converts_to_a_plain_dict(self, container: NetCDF):
        """`dict(nc)` works, which needs `keys` and `__getitem__` to agree.

        Args:
            container: One of the swept stores.

        Test scenario:
            `dict()` builds from `keys()` and then indexes each one. A `keys()` reporting
            a name `__getitem__` refuses -- the grouped store's `group/var` paths are the
            candidate -- would raise here rather than return a short dict.
        """
        as_dict = dict(container)

        assert sorted(as_dict) == sorted(container.variable_names)

    def test_indexing_and_the_accessor_refuse_a_miss_differently(self):
        """The deliberate `KeyError` / `ValueError` split.

        Test scenario:
            Both spellings reject the same unknown name, and each raises what its own
            protocol requires. Pinned because it looks like an inconsistency and will be
            "fixed" otherwise: `in`, `get` and `dict(nc)` are all built on `KeyError`, so
            `__getitem__` cannot raise `ValueError`, and `get_variable` predates the
            mapping and cannot start raising `KeyError` without breaking callers.
        """
        nc = open_store(PLAIN)

        with pytest.raises(KeyError):
            nc["no-such-variable"]

        with pytest.raises(ValueError, match="is not a valid variable name"):
            nc.get_variable("no-such-variable")


class TestTheXarraySpellings:
    """`data_vars`, `dims`, `sizes`, `attrs` and `coords`."""

    def test_data_vars_is_a_mapping_that_iterates_as_the_names(self, container: NetCDF):
        """The xarray spelling indexes by name, and still iterates as a list of names.

        Args:
            container: One of the swept stores.

        Test scenario:
            The three things xarray callers do with `data_vars`: iterate it for names,
            take its length, and index it for a variable. The last is what a list could
            not do, and is the reason this alias is the mapping rather than
            `variable_names`.
        """
        assert list(container.data_vars) == container.variable_names
        assert len(container.data_vars) == len(container)
        for name in container.variable_names:
            assert container.data_vars[name] is container[name]

    def test_data_vars_is_not_the_variable_names_list(self):
        """The collision this alias was reshaped to avoid.

        Test scenario:
            `nc.data_vars["temperature"]` is a variable. Were `data_vars` the
            `variable_names` list, the same expression would be a `TypeError` — the exact
            failure `dims`' docstring argues against, applied to the alias a reader from
            xarray reaches for first.
        """
        nc = open_store(PLAIN)

        assert nc.data_vars["temperature"].band_count == 12
        assert nc.variable_names == ["temperature"]
        assert nc.data_vars != nc.variable_names

    def test_dims_maps_each_name_to_its_length(self, container: NetCDF):
        """`dims` is a mapping, as xarray's is.

        Args:
            container: One of the swept stores.

        Test scenario:
            Keys are the declared dimension names and each value is an `int`. The type
            check is the point: a `dims` aliased onto `dimension_names` would pass a
            key-set comparison against a list only by accident and fail here.
        """
        assert isinstance(container.dims, dict)
        assert sorted(container.dims) == sorted(container.dimension_names)
        assert all(isinstance(length, int) for length in container.dims.values())

    def test_dims_is_not_the_dimension_names_list(self):
        """The one alias that deliberately does not alias its namesake.

        Test scenario:
            `nc.dims["time"]` is a length. Were `dims` the `dimension_names` list, the same
            expression would be a `TypeError` at best and a silent index at worst, which is
            exactly the failure this naming choice was made to avoid.
        """
        nc = open_store(PLAIN)

        assert nc.dims["time"] == 4
        assert nc.dimension_names == ["time", "pressure_level", "lat", "lon"]
        assert nc.dims != nc.dimension_names

    def test_sizes_dims_and_dimension_sizes_are_one_mapping(self, container: NetCDF):
        """All three spellings answer the same thing.

        Args:
            container: One of the swept stores.
        """
        assert container.sizes == container.dims == container.dimension_sizes

    def test_attrs_is_the_global_attributes(self, container: NetCDF):
        """The alias reads the root group's attributes.

        Args:
            container: One of the swept stores.
        """
        assert container.attrs == container.global_attributes

    def test_coords_carries_each_dimensions_stored_values(self, container: NetCDF):
        """Every entry is what `get_dimension_values` returns for that name.

        Args:
            container: One of the swept stores.

        Test scenario:
            Keys are a subset of the declared dimensions and each array is compared to the
            per-name accessor. Equality against the accessor rather than against the file
            is what keeps the storage-order contract in one place -- `coords` must not
            quietly reorient a south-to-north axis that `get_dimension_values` leaves
            alone.
        """
        assert set(container.coords) <= set(container.dimension_names)
        for name, values in container.coords.items():
            assert np.array_equal(values, container.get_dimension_values(name))

    def test_coords_omits_a_dimension_with_no_coordinate_variable(self):
        """An unindexed dimension is absent, not mapped to `None`.

        Test scenario:
            `bnds` is the CMIP store's bounds axis; it is a declared dimension with no
            variable of its own. A `None` value would make `nc.coords[name].shape` fail
            for a caller iterating the mapping, so the key is left out -- which is also
            what xarray does.
        """
        nc = open_store(MIXED_RANKS)

        assert "bnds" in nc.dimension_sizes
        assert "bnds" not in nc.coords
        assert sorted(nc.coords) == ["lat", "lon", "plev", "time"]

    def test_coords_is_empty_when_no_dimension_is_indexed(self):
        """The unstructured case returns `{}` rather than failing.

        Test scenario:
            A UGRID store's dimensions (`n_node`, `n_face`, `n_max_face_nodes`) are counts,
            not axes, and none has a coordinate variable. Every one is dropped, so the
            mapping is empty.
        """
        nc = open_store(LABELLED_ONLY)

        assert nc.dimension_sizes
        assert nc.coords == {}


class TestCheapIntrospection:
    """`dtypes`, `nbytes` and `info` -- all of which must answer without reading."""

    def test_dtypes_has_one_entry_per_variable(self, container: NetCDF):
        """Each name maps to a dtype the store actually uses.

        Args:
            container: One of the swept stores.

        Test scenario:
            Keys match the enumerated names, and every value names a dtype numpy can
            construct. The latter catches the shape of the bug this member is most prone
            to: a raster subset reports `dtype` as one entry *per band*, and returning that
            list stringified would pass a key comparison and be useless.
        """
        assert sorted(container.dtypes) == sorted(container.variable_names)
        for name in container.variable_names:
            assert np.dtype(container.dtypes[name]) is not None

    def test_nbytes_is_the_sum_over_the_variables(self, container: NetCDF):
        """The total is each variable's cells times its item size.

        Args:
            container: One of the swept stores.

        Test scenario:
            Recomputed here from the shapes the variables report, which for a raster
            subset is `rows * columns * band_count` -- the whole cube, because GDAL
            flattens every non-spatial dimension into the bands axis.
        """
        expected = 0
        for name in container.variable_names:
            variable = container[name]
            if isinstance(variable, LabeledArray):
                expected += np.asarray(variable.values).nbytes
            else:
                cells = variable.rows * variable.columns * variable.band_count
                expected += cells * np.dtype(container.dtypes[name]).itemsize

        assert container.nbytes == expected

    def test_nbytes_is_right_for_a_known_cube(self):
        """A hand-checked number, so the sum is not merely self-consistent.

        Test scenario:
            `temperature` is float64 over `(time, pressure_level, lat, lon)` =
            `(4, 3, 5, 6)`, which is 360 cells of 8 bytes. Recomputing the total the same
            way the implementation does would agree with any wrong shape rule, so one
            store is pinned to an arithmetic answer.
        """
        nc = open_store(PLAIN)

        assert nc.nbytes == 4 * 3 * 5 * 6 * 8

    @pytest.mark.parametrize("member", ["dtypes", "nbytes"])
    def test_the_sizing_members_read_no_pixels(
        self, member: str, monkeypatch: pytest.MonkeyPatch
    ):
        """Sizing a cube must not load it.

        Args:
            member: The property to evaluate.
            monkeypatch: Used to make every raster read raise.

        Test scenario:
            `read_array` is replaced with one that raises, then the property is read. This
            is the whole reason `nbytes` is computed from shape and dtype: a cube far
            larger than memory has to be sizeable, and a `sum(v.read_array().nbytes)`
            implementation would pass every other test in this class.
        """
        nc = open_store(MIXED_RANKS)
        forbid_reads(monkeypatch)

        assert getattr(nc, member)

    def test_info_reads_no_pixels(self, monkeypatch: pytest.MonkeyPatch):
        """The summary is metadata only.

        Args:
            monkeypatch: Used to make every raster read raise.
        """
        nc = open_store(MIXED_RANKS)
        forbid_reads(monkeypatch)

        report = io.StringIO()
        nc.info(report)

        assert "pyramids.NetCDF {" in report.getvalue()

    def test_info_names_every_dimension_and_every_variable(self, container: NetCDF):
        """Nothing the container declares is missing from the summary.

        Args:
            container: One of the swept stores.
        """
        report = io.StringIO()
        container.info(report)
        text = report.getvalue()

        for name, size in container.dimension_sizes.items():
            assert f"{name} = {size} ;" in text
        for name in container.variable_names:
            assert f" {name}(" in text

    def test_info_reports_each_variables_own_axes(self):
        """A 2-D variable is not labelled with the container's five dimensions.

        Test scenario:
            The CMIP store declares `lat`, `lon`, `bnds`, `plev` and `time`, but `area` is
            `(lat, lon)` and `ua` is `(time, plev, lat, lon)`. The axes are read from the
            store rather than from the subset `get_variable` returns, because a subset
            renames its y dimension to the window it was cut with -- `subset_lat_127_-1_128`,
            a name this file does not have.
        """
        nc = open_store(MIXED_RANKS)

        report = io.StringIO()
        nc.info(report)
        text = report.getvalue()

        assert "float32 area(lat, lon) ;" in text
        assert "float32 ua(time, plev, lat, lon) ;" in text
        assert "subset_" not in text

    def test_info_writes_to_stdout_when_given_no_buffer(self, capsys):
        """The default destination matches xarray's.

        Args:
            capsys: Captures what the call prints.
        """
        nc = open_store(PLAIN)

        nc.info()

        assert "temperature" in capsys.readouterr().out


class TestTheMappingsAreSafeToHoldOnTo:
    """A returned mapping is the caller's; editing it must not reach the container."""

    @pytest.mark.parametrize("member", ["dims", "sizes", "attrs", "coords", "dtypes"])
    def test_mutating_a_returned_mapping_does_not_reach_the_container(
        self, member: str
    ):
        """Each property hands back a fresh mapping, not internal state.

        Args:
            member: The property to read twice.

        Test scenario:
            The mapping is read, corrupted, and read again. `dimension_sizes` and
            `global_attributes` both build a fresh dict per call today, so this holds -- but
            an alias is exactly where a future "cache it" change would land, and a cached
            mapping handed straight out would let `nc.dims["lat"] = 0` silently break every
            later reader.
        """
        nc = open_store(PLAIN)

        first = getattr(nc, member)
        first["injected"] = "corrupted"

        assert "injected" not in getattr(nc, member)

    def test_mutating_the_list_keys_returns_does_not_reach_the_container(self):
        """`keys()` hands back a copy, not the list the container runs on.

        Test scenario:
            The regression this pins: `_LazyVariableDict.keys()` used to return `_names`
            itself -- the very list `__iter__`, `__len__`, `__contains__`, `values` and
            `items` are all driven from -- so `nc.keys().append("X")` made `list(nc)` and
            `len(nc)` wrong and `dict(nc)` raise.

            Asserted against `list(nc)`, `len(nc)` and `items()`, **not** against
            `variable_names`: that property re-reads the store on every call, so it reports
            the right answer even while the container is corrupted, and a test written
            against it passes with the bug present.
        """
        nc = open_store(MIXED_RANKS)
        original = list(nc)

        names = nc.keys()
        names.reverse()
        names.append("INJECTED")

        assert list(nc) == original
        assert len(nc) == len(original)
        assert "INJECTED" not in nc
        assert [name for name, _ in nc.items()] == original

    def test_the_lazy_mapping_hands_out_a_copy_too(self):
        """The fix is at the source, so `nc.variables.keys()` is safe as well.

        Test scenario:
            The container delegates to the mapping, so fixing only `NetCDF.keys` would
            leave `nc.variables.keys()` handing out the live list to anyone who reached
            one level down.
        """
        nc = open_store(PLAIN)

        first = nc.variables.keys()
        first.append("INJECTED")

        assert nc.variables.keys() == ["temperature"]
        assert list(nc) == ["temperature"]


class TestAVariableSubsetReportsTheContainerContract:
    """What the new members answer on what `get_variable` hands back.

    A subset is a `NetCDF` too, so every member here is reachable on it. They all follow
    from the members they delegate to being *container* concepts, and the answers are
    surprising enough -- `nbytes` is `0` for a variable that plainly holds 2880 bytes --
    that they are pinned rather than left to be discovered.
    """

    def test_a_subset_enumerates_no_data_variables(self):
        """`variable_names` is empty on a subset, so the whole mapping follows.

        Test scenario:
            `len`, `list` and `dict` all agree with `variable_names`, which is `[]`. This
            is what makes every assertion below come out the way it does.
        """
        variable = open_store(PLAIN)["temperature"]

        assert variable.variable_names == []
        assert len(variable) == 0
        assert list(variable) == []
        assert dict(variable) == {}

    def test_a_subset_reports_no_dimension_sizes_but_keeps_its_names(self):
        """`dims` is `{}` while `dimension_names` has four entries.

        Test scenario:
            The one place the two members genuinely disagree. `dimension_sizes` needs a root
            group and a subset has none; `dimension_names` falls back to the names cached
            when the subset was built. Pinned so the `dims` docstring's warning stays true.
        """
        variable = open_store(PLAIN)["temperature"]

        assert variable.dims == {}
        assert variable.sizes == {}
        assert len(variable.dimension_names) == 4

    def test_a_subsets_coords_follow_its_names_rather_than_its_sizes(self):
        """`coords` is not empty even though `dims` is.

        Test scenario:
            This is why `coords` iterates `dimension_names` rather than `dims`: the subset
            can still read its non-spatial axes, and keying off `dims` would have thrown
            them away. It also means `set(coords) <= set(dims)` is *not* the invariant --
            `set(coords) <= set(dimension_names)` is.
        """
        variable = open_store(PLAIN)["temperature"]

        assert set(variable.coords) <= set(variable.dimension_names)
        assert not set(variable.coords) <= set(variable.dims)
        assert variable.coords["time"].tolist() == [0.0, 6.0, 12.0, 18.0]

    def test_a_subset_sizes_itself_as_zero(self):
        """`nbytes` counts data variables, and a subset enumerates none.

        Test scenario:
            `nc["temperature"].nbytes` is `0` while `nc.nbytes` is 2880 for the same data.
            Documented on the property, because the natural reading of `variable.nbytes` is
            "this variable's size" and that is not what it answers.
        """
        nc = open_store(PLAIN)

        assert nc.nbytes == 2880
        assert nc["temperature"].nbytes == 0
        assert nc["temperature"].dtypes == {}

    def test_a_subsets_info_prints_an_empty_shell(self):
        """The summary is well-formed but lists nothing.

        Test scenario:
            It must not raise -- `info` is reachable on a subset and a `KeyError` from the
            dtype lookup would be a poor way to find that out.
        """
        report = io.StringIO()
        open_store(PLAIN)["temperature"].info(report)
        text = report.getvalue()

        assert text.startswith("pyramids.NetCDF {")
        assert text.rstrip().endswith("}")


class TestTheSizingHelpers:
    """`_variable_dtype` and `_variable_nbytes`, on inputs no fixture produces."""

    def test_a_variable_reporting_no_bands_has_an_unknown_dtype(self):
        """The defensive arm of `_variable_dtype`.

        Test scenario:
            A raster subset reports `dtype` as one entry per band. No fixture in the repo
            has a zero-band variable, so the `"unknown"` answer is unreachable from a real
            store and is exercised through a stub instead -- it is the documented return,
            and a silent `IndexError` would be the alternative.
        """

        class NoBands:
            dtype: list[str] = []
            rows = 0
            columns = 0
            band_count = 0

        assert _variable_dtype(NoBands()) == "unknown"

    def test_a_variable_reporting_no_bands_sizes_as_zero(self):
        """The paired helper must survive the dtype the first one hands it.

        Test scenario:
            `_variable_nbytes` fed `"unknown"` to `np.dtype`, which raises
            `TypeError: data type 'unknown' not understood` — and numpy evaluates that
            operand whether or not the cell count is 0. So the one input `_variable_dtype`
            documents aborted `nc.nbytes` and `nc.info()` rather than contributing nothing.
            Covering only the dtype half left it invisible.
        """

        class NoBands:
            dtype: list[str] = []
            rows = 0
            columns = 0
            band_count = 0

        assert _variable_nbytes(NoBands()) == 0

    def test_a_labelled_array_is_sized_from_the_array_it_holds(self):
        """The `LabeledArray` arm of both helpers.

        Test scenario:
            A UGRID store holds only `LabeledArray`s, which have no `dtype` of their own and
            no raster plane, so both helpers have to read the array. Its size is already in
            memory, so nothing is loaded that was not there.
        """
        nc = open_store(LABELLED_ONLY)
        variable = nc["node_lon"]

        assert isinstance(variable, LabeledArray)
        assert _variable_dtype(variable) == str(np.asarray(variable.values).dtype)
        assert _variable_nbytes(variable) == np.asarray(variable.values).nbytes


class TestTheInfoSummaryIsWellFormed:
    """The shape of what `info` prints, beyond the names it contains."""

    def test_the_summary_has_every_section_in_order(self):
        """Opens, lists dimensions, lists variables, lists attributes, closes.

        Test scenario:
            The section order is the shape `ncdump -h` and xarray's `info` both use, which
            is the only reason to print this rather than a dict. Asserted by index so a
            reordering is caught, not just a missing line.
        """
        nc = open_store(PLAIN)

        report = io.StringIO()
        nc.info(report)
        lines = report.getvalue().splitlines()

        assert lines[0] == "pyramids.NetCDF {"
        assert lines.index("dimensions:") < lines.index("variables:")
        assert lines.index("variables:") < lines.index("// global attributes:")
        assert lines[-1] == "}"

    def test_a_container_with_no_global_attributes_still_closes(self):
        """The attributes section may be empty.

        Test scenario:
            The header is printed unconditionally, so a store with nothing to list produces
            a section with no entries rather than a missing brace.
        """
        nc = open_store(LABELLED_ONLY)
        nc.global_attributes.clear()

        report = io.StringIO()
        nc.info(report)
        lines = report.getvalue().splitlines()

        assert "// global attributes:" in lines
        assert lines[-1] == "}"


class TestAClassicContainer:
    """`open_as_multi_dimensional=False` — a documented mode with no multidim group."""

    CLASSIC = (
        "cf__20v__1d3-3d17__y-desc.nc",
        "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc",
        "cf__40v__1d28-2d9-3d3__nc4.nc",
    )

    @pytest.mark.parametrize("name", CLASSIC, ids=lambda name: name.split("__")[1])
    def test_the_introspection_trio_answers_rather_than_raising(self, name: str):
        """`dtypes`, `nbytes` and `info` all work without a root group.

        Args:
            name: A store to open in classic mode.

        Test scenario:
            Two separate failures used to live here. `info` called
            `_variable_dim_names(None, ...)`, which reaches `None.OpenMDArray` and raises
            `AttributeError` -- a bug, not a contract. And on two of these stores the
            classic subdataset enumeration reports a name GDAL then declines to open, so
            all three members propagated a raw GDAL `RuntimeError` from a legitimately
            opened container.
        """
        nc = NetCDF.read_file(str(DATA / name), open_as_multi_dimensional=False)
        assert nc._working_group() is None

        assert sorted(nc.dtypes) == sorted(set(nc.variable_names))
        assert nc.nbytes >= 0

        report = io.StringIO()
        nc.info(report)
        assert report.getvalue().rstrip().endswith("}")

    def test_a_name_that_will_not_open_reports_unknown_rather_than_raising(self):
        """An unopenable name is described, not fatal.

        Test scenario:
            `O3.COLUMN.PARTIAL_AVK` is enumerated by the classic subdataset list and
            refused by GDAL on the way in. It reports `"unknown"`, which is visible in
            `dtypes` and in the `info` summary -- deliberately not silent, because the
            enumeration reporting a name it cannot open is itself a defect worth seeing.
        """
        nc = NetCDF.read_file(
            str(DATA / "cf__40v__1d28-2d9-3d3__nc4.nc"), open_as_multi_dimensional=False
        )

        assert nc.dtypes["O3.COLUMN.PARTIAL_AVK"] == "unknown"

        report = io.StringIO()
        nc.info(report)
        assert "unknown O3.COLUMN.PARTIAL_AVK" in report.getvalue()

    def test_variables_are_printed_without_axes_when_there_is_no_group(self):
        """Classic mode has no per-variable dimension names to print.

        Test scenario:
            The axes come from the multidim group, which classic mode does not have. Each
            variable is printed with an empty axis list rather than a fabricated one, and
            the dimensions section is unaffected.
        """
        nc = NetCDF.read_file(
            str(DATA / "cf__20v__1d3-3d17__y-desc.nc"), open_as_multi_dimensional=False
        )

        report = io.StringIO()
        nc.info(report)
        text = report.getvalue()

        assert "int16 tcw() ;" in text
        assert "subset_" not in text


class TestWhatTheDundersChangedAboutProtocolDispatch:
    """Adding `__iter__` / `__len__` made a container duck-type as a sequence.

    Pinned, not fixed. The container previously had neither dunder, so NumPy treated it
    as an opaque object and every `isinstance(x, Iterable)` site in the tree answered
    `False` for it. Both changed as a side effect of the mapping protocol, and the
    decision was to document the new behaviour rather than add an `__array__` that
    refuses it. These tests exist so the choice is visible and cannot drift again
    unnoticed.
    """

    def test_a_container_now_coerces_to_an_array_of_its_names(self):
        """`np.asarray(nc)` builds names, where it used to box the container.

        Test scenario:
            NumPy sees `__len__` + `__iter__` and iterates rather than wrapping. Iterating
            a container yields names, so the result is a string array -- not data, and not
            a useful way to reach any. The assertion names the dtype kind so the intent is
            unmistakable: this is a list of labels.
        """
        nc = open_store(PLAIN)

        coerced = np.asarray(nc)

        assert coerced.dtype.kind == "U"
        assert coerced.tolist() == ["temperature"]

    @pytest.mark.parametrize("protocol", ["Iterable", "Sized", "Container"])
    def test_the_abc_checks_now_answer_true(self, protocol: str):
        """Three `collections.abc` checks flipped from `False`.

        Args:
            protocol: The `collections.abc` name to test against.

        Test scenario:
            A helper that gates on `isinstance(x, Iterable)` used to reject a container
            outright and now accepts it, receiving a list of name strings. That is the one
            practical consequence of the change worth knowing about.
        """
        nc = open_store(PLAIN)

        assert isinstance(nc, getattr(abc, protocol))

    def test_a_container_is_still_not_a_mapping_or_a_sequence(self):
        """The flips stop short of the two protocols that would imply more.

        Test scenario:
            `Mapping` needs `__getitem__`, `__len__`, `__iter__` *and* registration or the
            ABC's mixins; `Sequence` needs integer indexing. The container offers neither,
            so code branching on those still takes the same path it did before.
        """
        nc = open_store(PLAIN)

        assert not isinstance(nc, abc.Mapping)
        assert not isinstance(nc, abc.Sequence)

    def test_truthiness_still_refuses_despite_len(self):
        """`__bool__` wins over `__len__`, and still raises.

        Test scenario:
            `len(nc)` now works, which invites `if nc:` -- but `Dataset.__bool__` refuses
            class-wide because a raster's truth value is ambiguous, and it takes precedence
            over `__len__`. So the idiom the new dunder suggests raises, with a message
            about cells rather than about variables. `len(nc) == 0` is the spelling that
            answers.
        """
        nc = open_store(PLAIN)

        with pytest.raises(ValueError, match="truth value of a Dataset is ambiguous"):
            bool(nc)

        assert len(nc) == 1
