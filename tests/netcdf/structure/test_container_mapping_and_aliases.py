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
from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf import NetCDF
from pyramids.netcdf.labeled import LabeledArray

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

    def test_data_vars_is_the_variable_names(self, container: NetCDF):
        """The alias returns the canonical member's list.

        Args:
            container: One of the swept stores.
        """
        assert container.data_vars == container.variable_names

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
        assert set(container.coords) <= set(container.dimension_sizes)
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
