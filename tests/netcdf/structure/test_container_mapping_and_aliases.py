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

The introspection trio is the other half: `dtypes`, `nbytes` and `info` have to answer
without reading a pixel, or they are useless on the cubes that most need sizing. That is
asserted by counting `MDArray.ReadAsArray` -- the call that actually moves bytes -- and not
`NetCDF.read_array`, which none of the three calls and which therefore cannot detect
anything. Counting rather than forbidding also lets the one case where they *do* read be
pinned instead of denied: a variable with no raster plane is materialised to report its
type.
"""

from __future__ import annotations

import io
import re
import warnings
from collections import abc
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF
from pyramids.netcdf.labeled import LabeledArray
from pyramids.netcdf.netcdf import (
    _open_variable,
    _summarised,
    _variable_dtype,
    _variable_nbytes,
)

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


def count_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every array GDAL actually reads, and return the growing log.

    Patching `NetCDF.read_array` proves nothing here: `dtypes`, `nbytes` and `info` never
    call it. Their reads, where they happen, go through `get_variable` ->
    `_read_variable` -> `MDArray.ReadAsArray`, so that is what has to be watched -- the one
    call that actually moves bytes.

    Args:
        monkeypatch: The patcher to install the counting wrapper through.

    Returns:
        list[str]: One entry per read, appended as it happens.
    """
    reads: list[str] = []
    original = gdal.MDArray.ReadAsArray

    def record(self: gdal.MDArray, *args: object, **kwargs: object):
        reads.append(str(self.GetName()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(gdal.MDArray, "ReadAsArray", record)
    return reads


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

    def test_the_three_spellings_report_the_lengths_the_file_declares(self):
        """One hand-checked answer, so a wrong one is caught and not just a wrong wiring.

        Test scenario:
            The comparison above cannot fail while all three members `return
            dimension_sizes` -- it would only catch a future re-implementation. This pins
            what the store actually declares, read from the file with `ncdump -h`:
            `(time, pressure_level, lat, lon) = (4, 3, 5, 6)`.
        """
        nc = open_store(PLAIN)

        assert nc.sizes == {"time": 4, "pressure_level": 3, "lat": 5, "lon": 6}

    def test_attrs_is_the_global_attributes(self, container: NetCDF):
        """The alias reads the root group's attributes.

        Args:
            container: One of the swept stores.
        """
        assert container.attrs == container.global_attributes

    def test_attrs_reports_what_the_file_declares(self):
        """A hand-checked answer behind the alias comparison.

        Test scenario:
            The plain CF store declares exactly one global attribute. Pinned so the alias
            is shown to carry a real value, not merely to be wired to the same member.
        """
        nc = open_store(PLAIN)

        assert nc.attrs == {"Conventions": "CF-1.6"}

    def test_coords_reports_the_values_the_file_stores(self):
        """A hand-checked axis, read from the file rather than from the accessor.

        Test scenario:
            Comparing `coords[name]` to `get_dimension_values(name)` -- which is what
            `coords` is built from -- cannot catch a wrong answer, only a wrong wiring.
            These are the latitudes and levels the store declares, ascending and
            descending respectively, so an orientation applied here would show up.
        """
        nc = open_store(PLAIN)

        assert nc.coords["lat"].tolist() == [40.0, 41.0, 42.0, 43.0, 44.0]
        assert nc.coords["pressure_level"].tolist() == [1000.0, 850.0, 500.0]
        assert nc.coords["time"].tolist() == [0.0, 6.0, 12.0, 18.0]

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

    def test_dtypes_reports_the_types_the_file_declares(self):
        """Hand-checked types, including the one variable that is not floating point.

        Test scenario:
            `np.dtype(...)` succeeding proves only that the string names *a* type -- it
            passes for the meaningless `'object'` a string variable reports. The CMIP
            store mixes `float32` fields with an `int32` mask, so pinning it catches a
            member that reported the wrong type rather than no type.
        """
        nc = open_store(MIXED_RANKS)

        assert nc.dtypes == {
            "area": "float32",
            "msk_rgn": "int32",
            "pr": "float32",
            "tas": "float32",
            "ua": "float32",
        }

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
        assert sorted(container.dtypes) == sorted(set(container.variable_names))
        for name, kind in container.dtypes.items():
            if kind == "unknown":
                continue
            # Asserting the round trip, not `is not None`: `np.dtype()` returns a dtype or
            # raises, so an `is not None` check can only ever pass and says nothing.
            assert np.dtype(kind).name == kind, (
                f"{name} reports {kind!r}, which numpy normalises to "
                f"{np.dtype(kind).name!r} — the two should agree"
            )

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
    def test_the_sizing_members_read_no_data_variable(
        self, member: str, monkeypatch: pytest.MonkeyPatch
    ):
        """Sizing a cube of raster variables must not load one.

        Args:
            member: The property to evaluate.
            monkeypatch: Used to count the reads GDAL performs.

        Test scenario:
            `MDArray.ReadAsArray` is counted -- the call that actually moves bytes -- not
            `NetCDF.read_array`, which none of these members calls and which therefore
            cannot detect anything. This is the whole reason `nbytes` is computed from
            shape and dtype: a cube far larger than memory has to be sizeable, and a
            `sum(v.read_array().nbytes)` implementation would pass every other test here.

            What is asserted is precise, because "reads nothing" is not true and asserting
            it would only invite a weaker guard later: opening a variable resolves its
            **coordinate** axes, so `lat`, `lon`, `time` and `plev` are read. No *data*
            variable is, and every name read is a dimension -- which is what the docstrings
            mean by "reads no pixels", one small array per axis rather than one per cell.
        """
        nc = open_store(MIXED_RANKS)
        reads = count_reads(monkeypatch)

        assert getattr(nc, member)
        assert set(reads).isdisjoint(nc.variable_names)
        assert set(reads) <= set(nc.dimension_sizes)

    def test_info_reads_no_data_variable(self, monkeypatch: pytest.MonkeyPatch):
        """The summary is metadata plus coordinates, never a data plane.

        Args:
            monkeypatch: Used to count the reads GDAL performs.
        """
        nc = open_store(MIXED_RANKS)
        reads = count_reads(monkeypatch)

        report = io.StringIO()
        nc.info(report)

        assert "pyramids.NetCDF {" in report.getvalue()
        assert set(reads).isdisjoint(nc.variable_names)

    def test_sizing_a_store_of_labelled_arrays_does_read(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The caveat, asserted rather than assumed.

        Args:
            monkeypatch: Used to count the reads GDAL performs.

        Test scenario:
            A variable with no raster plane comes back as a `LabeledArray`, and building
            one materialises its array. So on the UGRID store `nbytes` reads every
            variable -- one `ReadAsArray` each -- which is exactly what the docstring used
            to deny. Pinned here so the claim and the behaviour cannot drift apart again:
            if the sizing is ever changed to work from the declared shape, this test is
            what says so.
        """
        nc = open_store(LABELLED_ONLY)
        reads = count_reads(monkeypatch)

        assert nc.nbytes == 192
        assert len(reads) == len(nc)

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

    @pytest.mark.parametrize("mapping", ["data_vars", "variables"])
    def test_the_variables_mapping_refuses_a_write(self, mapping: str):
        """It caches, so a write would be seen by `nc[name]` and by nothing else.

        Args:
            mapping: The spelling to write through.

        Test scenario:
            `_LazyVariableDict` subclasses `dict` to hold what it has loaded, and `_names`
            -- what `__iter__`, `__len__` and `__contains__` all answer from -- is a
            separate list. A write therefore landed in the cache alone, leaving the
            container disagreeing with itself: `nc["X"]` returned the injected value while
            `"X" in nc` was `False` and `list(nc)` never mentioned it. `data_vars` is the
            same object, so it had the same hole.
        """
        nc = open_store(PLAIN)

        with pytest.raises(TypeError, match="read-through view"):
            getattr(nc, mapping)["INJECTED"] = "not a variable"

        assert "INJECTED" not in nc
        assert list(nc) == ["temperature"]

    def test_the_variables_mapping_refuses_a_delete(self):
        """Deleting would only drop the cache entry, which reloads on the next lookup.

        Test scenario:
            The name would still be in `_names`, so `in`, `len` and iteration would be
            unchanged and the next `nc[name]` would read it again — a no-op dressed as a
            removal.
        """
        nc = open_store(PLAIN)

        with pytest.raises(TypeError, match="remove_variable"):
            del nc.variables["temperature"]

        assert "temperature" in nc

    def test_refusing_a_write_does_not_stop_the_mapping_caching(self):
        """The internal fill bypasses `__setitem__`, so laziness is unaffected.

        Test scenario:
            The cache is populated through `dict.__setitem__` directly, so overriding the
            bound method refuses callers without disabling the caching the mapping exists
            for. Asserted by identity: a second lookup is the same object, not a reload.
        """
        nc = open_store(PLAIN)

        assert nc["temperature"] is nc["temperature"]

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


class TestTheHelperThatOpensAVariable:
    """`_open_variable`, which decides which refusals introspection survives."""

    @pytest.mark.parametrize(
        "refusal", [RuntimeError, KeyError, ValueError], ids=lambda kind: kind.__name__
    )
    def test_a_name_the_container_refuses_is_reported_as_absent(
        self, refusal: type[Exception]
    ):
        """Every caught refusal comes back as `None`, whichever exception spells it.

        Args:
            refusal: The exception the container raises for the name.

        Test scenario:
            Only `RuntimeError` is reachable from a fixture -- the classic subdataset
            enumeration on `cf__40v__1d28-2d9-3d3__nc4.nc` reports an array GDAL then
            declines to open. The other two are the mapping's own spellings of a miss:
            `nc[name]` raises `KeyError`, and `get_variable` behind it raises `ValueError`,
            either of which reaches here whenever the enumeration and the mapping disagree
            about a name. Both sat in the caught tuple unexercised, so dropping either left
            this suite green while `dtypes`, `nbytes` and `info` began raising on a
            container they are documented to describe rather than stop at.
        """

        class Refuses:
            def __getitem__(self, name: str):
                raise refusal(name)

        assert _open_variable(Refuses(), "whatever") is None

    def test_an_error_that_is_not_a_refusal_still_propagates(self):
        """The catch is three named exceptions, not a blanket `except Exception`.

        Test scenario:
            The helper is here so a *defective enumeration* cannot take `info()` down, and
            catching everything would swallow the failures it is not there to absorb --
            a `TypeError` from a caller handing it something that is not a name, say.
            Pinned from the other side so a later widening to `except Exception` is a test
            failure rather than a silent loss of every error this file can raise.
        """

        class Breaks:
            def __getitem__(self, name: str):
                raise TypeError("not a name")

        with pytest.raises(TypeError, match="not a name"):
            _open_variable(Breaks(), "whatever")


class TestTheAttributeSummariser:
    """`_summarised`, on values whose `repr` is not already a single line."""

    def test_a_value_whose_repr_spans_lines_is_collapsed(self):
        """The replacement no string attribute reaches, and every fixture is a string.

        Test scenario:
            `repr` escapes a string's newline, so a value like `"first\\nsecond"` is one
            line before the helper touches it -- the fixture attributes that do carry raw
            newlines (the ROMS store's `NLM_LBC`) are strings, and all take the escaped
            arm. A value that is not a string is the other case: numpy renders a 2-D array
            one row per line, and those are real newlines *inside* the `repr`, which would
            split one attribute across several lines of the `ncdump -h` shape `info`
            imitates.
        """
        rendered = _summarised(np.arange(12).reshape(3, 4), limit=200)

        assert "\n" not in rendered
        assert rendered.startswith("array([[")
        assert "11" in rendered

    def test_a_value_whose_repr_carries_a_carriage_return_is_collapsed(self):
        """Stated as the invariant the three replacements exist for: one line, always.

        Test scenario:
            Both characters that end a line are removed, not just the one a text file on
            this platform uses -- an attribute written on another platform is as likely to
            arrive with the other. A bare CR survives `repr` only for a value that renders
            itself verbatim, which is why no store in the repo produces one.
        """

        class Verbatim:
            def __repr__(self) -> str:
                return "first\rsecond"

        assert _summarised(Verbatim()) == "first second"


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

    def test_a_container_with_no_global_attributes_still_closes(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The attributes section may be empty.

        Args:
            monkeypatch: Used to give the container no global attributes.

        Test scenario:
            The header is printed unconditionally, so a store with nothing to list produces
            a section with no entries rather than a missing brace -- asserted by checking
            that the header line is followed *directly* by the closing brace.

            `global_attributes` is monkeypatched rather than cleared. It reads the GDAL
            root group afresh on every call, so `nc.global_attributes.clear()` mutates a
            throwaway dict and the container still reports all 55 of the ugrid store's
            attributes -- the version of this test that did that never reached the case it
            names.
        """
        nc = open_store(LABELLED_ONLY)
        assert len(nc.global_attributes) > 0

        monkeypatch.setattr(type(nc), "global_attributes", property(lambda self: {}))

        report = io.StringIO()
        nc.info(report)
        lines = report.getvalue().splitlines()

        assert lines[-2] == "// global attributes:"
        assert lines[-1] == "}"

    def test_an_attribute_whose_value_spans_lines_stays_on_one_line(self):
        """One attribute is one printed line, however many lines its value holds.

        Test scenario:
            The ROMS store's `NLM_LBC` is a boundary-condition table -- raw newlines
            inside a single attribute value -- and printing it verbatim puts a dozen
            unlabelled lines in the middle of a section where every other line reads
            `:name = value ;`. Asserted as a count, one printed line per attribute the
            container reports, so any attribute that breaks the shape is caught rather
            than only this one; the table's own line is then checked to hold two of the
            rows the file stores separately, which is what "collapsed" means here.
        """
        nc = open_store(CURVILINEAR)
        assert "\n" in nc.attrs["NLM_LBC"]

        report = io.StringIO()
        nc.info(report)
        printed = [
            line for line in report.getvalue().splitlines() if line.startswith("\t:")
        ]

        assert len(printed) == len(nc.attrs)
        table = next(line for line in printed if line.startswith("\t:NLM_LBC"))
        assert "EDGE:" in table
        assert "zeta:" in table

    def test_an_over_long_attribute_is_truncated_rather_than_printed_whole(self):
        """A 1,800-character value is cut to the documented limit, and says that it was.

        Test scenario:
            The CMIP store's `history` is a full provenance log. Printed whole it is a
            dozen terminal lines for one attribute, burying the twenty-odd others in a
            summary meant to be read at a glance. The cut is asserted on the rendered
            value rather than on the line, so the 120-character limit plus its ellipsis is
            pinned exactly instead of through whatever the key happens to be called.
        """
        nc = open_store(MIXED_RANKS)
        assert len(nc.attrs["history"]) > 1000

        report = io.StringIO()
        nc.info(report)
        printed = next(
            line
            for line in report.getvalue().splitlines()
            if line.startswith("\t:history")
        )
        value = printed.split(" = ", 1)[1].removesuffix(" ;")

        # 120 characters, the ellipsis, and the closing quote the truncation restores.
        assert len(value) == 124
        assert value.endswith("...'")


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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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

    def test_the_dimension_mappings_are_empty_without_a_group(self):
        """`dims`, `sizes` and `coords` answer `{}` on a store full of variables.

        Test scenario:
            Classic mode has no multidim group to enumerate dimensions from, so
            `dimension_names` is `None` here -- not `[]`, which is what every other store
            in this suite produces. `coords` iterates that list, so its `or []` guard is
            all that stands between a documented open mode and
            `TypeError: 'NoneType' object is not iterable`, and no test reached it.

            The container still enumerates its 17 variables, so the empty mappings say "no
            group", not "no data" -- which is what makes the pairing worth asserting
            together, and why `info` prints an empty dimensions section directly above a
            full variables one.
        """
        nc = NetCDF.read_file(
            str(DATA / "cf__20v__1d3-3d17__y-desc.nc"), open_as_multi_dimensional=False
        )

        assert nc.dimension_names is None
        assert nc.dims == {}
        assert nc.sizes == {}
        assert nc.coords == {}
        assert len(nc) == 17

        report = io.StringIO()
        nc.info(report)
        lines = report.getvalue().splitlines()

        assert lines.index("variables:") == lines.index("dimensions:") + 1


class TestWhatTheDundersChangedAboutProtocolDispatch:
    """Adding `__iter__` / `__len__` made a container duck-type as a sequence.

    The container previously had neither dunder, so NumPy treated it as an opaque object
    and every `isinstance(x, Iterable)` site in the tree answered `False` for it. Both
    changed as a side effect of the mapping protocol.

    The array coercion is **refused** by `__array__` rather than documented, because on a
    variable it would have answered `nan` instead of raising. The `isinstance` flips are
    kept and pinned, since they follow from the protocol the task asked for.
    """

    @pytest.mark.parametrize("shape", ["container", "variable"])
    def test_coercing_to_an_array_is_refused(self, shape: str):
        """`np.asarray` raises rather than inventing an answer from the iteration.

        Args:
            shape: Whether to coerce the container or one of its variables.

        Test scenario:
            Without `__array__`, NumPy would see `__len__` + `__iter__` and convert by
            looping. Both answers would be wrong and one is dangerous: a container yields
            its *names*, so the result is a visibly-useless string array; a variable yields
            nothing at all, so NumPy would build an empty array. The refusal restores what
            NumPy did before the mapping protocol existed, which was to raise.
        """
        nc = open_store(PLAIN)
        target = nc if shape == "container" else nc["temperature"]

        with pytest.raises(TypeError, match="cannot be converted to an array"):
            np.asarray(target)

    def test_a_reduction_over_a_variable_raises_instead_of_answering_nan(self):
        """The reason the coercion is refused rather than documented.

        Test scenario:
            `np.mean` reaches for `np.asarray` first. Iterating a variable yields nothing,
            so without the refusal NumPy averaged an empty array and returned `nan` -- for
            a cube of 557,056 real values, in a shape that looks like a legitimate result
            and can travel a long way before anyone questions it. `origin/main` raised
            here, and so does this.
        """
        variable = open_store(MIXED_RANKS)["ua"]
        assert variable.band_count * variable.rows * variable.columns == 557056

        with pytest.raises(TypeError, match="cannot be converted to an array"):
            np.mean(variable)

        assert variable.read_array().shape == (17, 128, 256)

    def test_boxing_in_an_object_array_still_works(self):
        """The one coercion that is honoured, because it invents nothing.

        Test scenario:
            `dtype=object` stores the dataset as itself -- nothing is read and no value is
            fabricated -- and it worked before this branch. Refusing it too would have
            been a regression against `origin/main`, where `np.array([nc], dtype=object)`
            is `(1,)`, so `__array__` answers that request and refuses every other.
        """
        nc = open_store(PLAIN)

        boxed = np.array([nc], dtype=object)

        assert boxed.shape == (1,)
        assert boxed[0] is nc

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


class TestWhereTheAliasesDivergeFromXarray:
    """The gaps a reader who came for the xarray spelling should know about."""

    def test_a_dtype_is_a_string_not_a_numpy_dtype(self):
        """`nc.dtypes["t2m"] == np.float64` is `False` here and `True` in xarray.

        Test scenario:
            The values are the dtype's *name*. The comparison a reader would write from
            xarray habit therefore fails silently -- it is `False`, not an error -- so the
            working spelling is pinned beside it.
        """
        nc = open_store(PLAIN)

        assert nc.dtypes["temperature"] == "float64"
        assert nc.dtypes["temperature"] != np.float64
        assert np.dtype(nc.dtypes["temperature"]) == np.float64

    def test_a_coordinate_is_a_bare_array_not_a_data_array(self):
        """`nc.coords["lat"].values` is an `AttributeError`, where xarray has `.values`.

        Test scenario:
            xarray's `coords[name]` is a `DataArray` carrying its own `dims` and `attrs`.
            These are the values themselves, so the attribute access a reader would reach
            for has nothing behind it.
        """
        nc = open_store(PLAIN)

        assert isinstance(nc.coords["lat"], np.ndarray)
        with pytest.raises(AttributeError):
            nc.coords["lat"].values

    def test_writing_to_a_returned_mapping_is_discarded_rather_than_refused(self):
        """xarray's `dims` is `Frozen` and raises; this is a plain dict and forgets.

        Test scenario:
            The write succeeds against a throwaway, so nothing signals that it had no
            effect. Pinned as a known divergence rather than as a feature -- it is the one
            place where the xarray spelling is safer than this one.
        """
        nc = open_store(PLAIN)

        nc.dims["lat"] = 999

        assert nc.dims["lat"] == 5

    def test_a_classic_container_can_repeat_a_name_the_mappings_cannot(self):
        """`variable_names` is a list and may repeat; the mappings are keyed by name.

        Test scenario:
            The classic subdataset enumeration reports
            `mole_content_of_ozone_in_atmosphere_layer` more than once, so the store has 12
            names and 9 distinct ones. `len(nc)` counts the list and `len(nc.dtypes)` counts
            the keys, and they disagree -- which is why the sweep's invariant is stated
            against `set(variable_names)` rather than the list.
        """
        nc = NetCDF.read_file(
            str(DATA / "cf__40v__1d28-2d9-3d3__nc4.nc"), open_as_multi_dimensional=False
        )

        assert len(nc.variable_names) == 12
        assert len(set(nc.variable_names)) == 9
        assert len(nc) == 12
        assert len(nc.dtypes) == 9


class TestThePublicApiPageMatchesTheClass:
    """`docs/reference/netcdf/public-api.md` claims exact counts; hold it to them."""

    PAGE = Path(__file__).parents[3] / "docs" / "reference" / "netcdf" / "public-api.md"

    def test_the_declared_counts_match_reflection(self):
        """The page's opening sentence is checkable, so check it.

        Test scenario:
            The page states how many members `NetCDF` defines, broken down by kind. It
            said 65 before this branch and nothing noticed, because nothing compared it to
            the class. Counted here by reflection over the members `NetCDF` defines in its
            own body -- inherited ones are a separate figure the page quotes separately.
        """
        own = {
            name: value
            for name, value in vars(NetCDF).items()
            if not name.startswith("_")
        }
        counts = {"method": 0, "property": 0, "classmethod": 0, "staticmethod": 0}
        for value in own.values():
            if isinstance(value, property):
                counts["property"] += 1
            elif isinstance(value, classmethod):
                counts["classmethod"] += 1
            elif isinstance(value, staticmethod):
                counts["staticmethod"] += 1
            elif callable(value):
                counts["method"] += 1

        text = " ".join(self.PAGE.read_text(encoding="utf-8").split())
        remedy = (
            f"docs/reference/netcdf/public-api.md is out of date: NetCDF now defines "
            f"{len(own)} public members ({counts['method']} methods, "
            f"{counts['property']} properties, {counts['classmethod']} classmethods, "
            f"{counts['staticmethod']} staticmethod). Update the page's opening sentence "
            f"and add the new member to the table it belongs in."
        )

        # Matched on a word boundary: a bare `in` lets "1 staticmethod" pass against a page
        # reading "1 staticmethods", and the plural is exactly what drifts.
        for phrase in (
            f"{len(own)} in all",
            f"{counts['method']} methods",
            f"{counts['property']} properties",
            f"{counts['classmethod']} classmethods",
            f"{counts['staticmethod']} staticmethod",
        ):
            assert re.search(rf"{re.escape(phrase)}" + r"\b", text), remedy

    @pytest.mark.parametrize(
        "member",
        ["data_vars", "dims", "sizes", "attrs", "coords", "dtypes", "nbytes", "info()"],
    )
    def test_every_new_member_is_indexed(self, member: str):
        """A member absent from the index is one nobody can find.

        Args:
            member: The member the page must mention.
        """
        assert f"`{member}`" in self.PAGE.read_text(encoding="utf-8"), (
            f"{member} is missing from docs/reference/netcdf/public-api.md — add it to "
            f"the table it belongs in, so a reader scanning the index can find it."
        )


class TestNbytesSaysWhenItCouldNotSizeSomething:
    """A 90% shortfall must not look like an ordinary answer."""

    CLASSIC_SHORTFALL = "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"

    def test_a_classic_store_warns_and_names_what_it_skipped(self):
        """The three biggest cubes contribute nothing, and the warning says which.

        Test scenario:
            A classic container's subdataset list names a variable by its `standard_name`
            -- `precipitation_flux` for `pr`, `air_temperature` for `tas`,
            `eastward_wind` for `ua` -- and GDAL declines to open any of them. All three
            contribute 0 while the bounds arrays the multidimensional view excludes are
            counted instead, so the total is a plausible-looking number that is wrong by
            an order of magnitude. `assert nc.nbytes >= 0`, which is what this class
            asserted before, cannot catch that.
        """
        nc = NetCDF.read_file(
            str(DATA / self.CLASSIC_SHORTFALL), open_as_multi_dimensional=False
        )

        with pytest.warns(UserWarning, match="nbytes excludes 3 variable") as caught:
            classic = nc.nbytes

        # GDAL emits its own warnings while opening this store, so the one under test is
        # selected by content rather than by position.
        message = next(
            str(w.message) for w in caught if "nbytes excludes" in str(w.message)
        )
        assert "precipitation_flux" in message
        assert "air_temperature" in message
        assert "eastward_wind" in message
        assert "open_as_multi_dimensional=True" in message
        assert classic == 268304

    def test_the_multidimensional_view_sizes_the_same_file_properly(self):
        """The same store, opened the documented way, is ten times larger.

        Test scenario:
            Pinned beside the shortfall so the scale of it is in the suite rather than in
            a comment: 268,304 bytes against 2,752,512 for the same file. The
            multidimensional open resolves `pr`, `tas` and `ua` by their real names and
            warns about nothing.
        """
        nc = NetCDF.read_file(str(DATA / self.CLASSIC_SHORTFALL))

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            full = nc.nbytes

        assert full == 2752512
        assert full > 10 * 268304

    def test_a_store_that_opens_cleanly_does_not_warn(self):
        """The warning is about a real shortfall, not noise on every classic open.

        Test scenario:
            `cf__20v__1d3-3d17__y-desc.nc` opened classically enumerates 17 names and opens
            all of them, so its total is complete and nothing is raised. A warning here
            would train readers to ignore the one that matters.
        """
        nc = NetCDF.read_file(
            str(DATA / "cf__20v__1d3-3d17__y-desc.nc"), open_as_multi_dimensional=False
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            assert nc.nbytes > 0
