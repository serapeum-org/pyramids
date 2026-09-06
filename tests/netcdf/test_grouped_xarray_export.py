"""A grouped store's xarray export must key its variables by names netCDF accepts.

Enumerating a grouped netCDF-4 store names a sub-group array by its path
(`flight_a/CO`), and the export used that path verbatim as the `xr.Dataset` key.
netCDF reserves `/` as the group separator and rejects it in a variable name, so
the exported Dataset could not be written back: `NetCDF.from_xarray` raised
`NetCDF: Name contains illegal characters`, and `Dataset.to_netcdf` either
refused the name or silently turned it into a group, depending on which engine
xarray picked (round-4 N5).

The export now flattens the path, keeping the group as a prefix so two groups'
identically named leaves stay apart, and appending a numeric suffix for the rare
name that flattens onto one already in the dataset.

The file's second subject is where the export's warnings point. `stacklevel` was
a hard-coded `4`, counted along
`caller -> NetCDF.to_xarray -> Interop.to_xarray -> _data_vars_from_arrays`.
That is not the only chain: `nc.interop.to_xarray()` is one frame shorter, and a
counted constant blamed pytest's internals rather than the caller's own line
(round-4 N7).

The third is what a caller can do with a key once they have one. Flattening
makes the exported key a name `get_variable` refuses -- for eight of the nine
data variables of the fixture below -- and it is not injective, so the key
cannot be turned back into the store's name for the array. Every rewritten
variable therefore carries that name as its `pyramids_store_name` attribute, and
the warning that announces the rewrite covers all of them rather than only the
suffixed few (round-5 S1/N2).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF
from pyramids.netcdf.engines.interop import _promote_cf_non_data_arrays

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.interop

DATA = Path(__file__).parents[1] / "data" / "netcdf"
GROUPED = DATA / "none__35v__1d35__groups-nc4.nc"
FLAT = DATA / "cf__5v__1d4-4d1__y-asc.nc"

# Value written into each array of the synthetic store, so an export that mixed
# two arrays up is caught by reading the numbers back rather than the names.
STORE_VALUES = {
    "flight_a_CO": 1.0,
    "flight_a/CO": 2.0,
    "flight_b/CO": 3.0,
    "rec/Num": 4.0,
}


def write_grouped_store(path: Path) -> None:
    """Write a two-level netCDF-4 store whose group paths collide once flattened.

    The store is built to exercise every branch of the flattening in one file:

    - `flight_a/CO` and `flight_b/CO` are the same leaf in two groups, so they
      must not be reduced to a shared `CO`;
    - a root array is literally named `flight_a_CO`, which is what
      `flight_a/CO` flattens to;
    - the dimension coordinate is named `rec_Num`, which is what `rec/Num`
      flattens to -- coordinates and data variables share one namespace in an
      `xr.Dataset`, so that collides too.

    Args:
        path: Destination `.nc` path.
    """
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    dataset = gdal.GetDriverByName("netCDF").CreateMultiDimensional(str(path))
    root = dataset.GetRootGroup()
    dim = root.CreateDimension("rec_Num", "", "", 4)
    root.CreateMDArray("rec_Num", [dim], f64).Write(np.arange(4, dtype="float64"))
    for name, value in STORE_VALUES.items():
        group = root
        if "/" in name:
            group_name, name = name.split("/")
            group = root.CreateGroup(group_name)
        group.CreateMDArray(name, [dim], f64).Write(np.full(4, value))
    # Drop every child handle before Close(): the netCDF driver only flushes and
    # releases the file once they are gone, and a half-written file reopens as
    # "not recognized as being in a supported file format".
    del dim, root, group
    dataset.Close()


@pytest.fixture
def grouped_store(tmp_path) -> Path:
    """Path to a freshly written synthetic grouped store.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path: The `.nc` file, ready to read.
    """
    path = tmp_path / "flights.nc"
    write_grouped_store(path)
    return path


def export(path: Path, **kwargs):
    """Export a store to xarray with warnings silenced.

    Args:
        path: The `.nc` file to read.
        **kwargs: Forwarded to `NetCDF.to_xarray`.

    Returns:
        xarray.Dataset: The exported dataset.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exported = NetCDF.read_file(str(path)).to_xarray(**kwargs)
    return exported


class TestTheExportedNamesAreWritable:
    """The consequence: what comes out of `to_xarray` has to be writable again."""

    def test_no_exported_name_carries_a_group_separator(self):
        """A real grouped store must not hand out a name netCDF rejects.

        Test scenario:
            `none__35v__1d35__groups-nc4.nc` holds eight readable sub-group
            arrays. Keyed by their store path every one of them carries a `/`,
            which is the character netCDF reserves for the group separator and
            refuses in a name -- so the whole export was unwritable, not just
            an awkward corner of it.
        """
        exported = export(GROUPED)

        illegal = [n for n in exported.variables if "/" in str(n)]

        assert illegal == [], f"exported names netCDF cannot write: {illegal}"

    def test_the_grouped_export_writes_back_through_from_xarray(self, grouped_store):
        """`from_xarray(to_xarray(...))` raised on the name, before reading a byte.

        Args:
            grouped_store: The synthetic grouped `.nc` fixture.

        Test scenario:
            This is the round trip the export exists to support. It failed with
            `RuntimeError: NetCDF: Name contains illegal characters` -- raised
            by GDAL at `CreateMDArray`, naming neither the variable nor the
            reason, on a Dataset the user had just been handed as valid.
        """
        destination = grouped_store.parent / "round_trip.nc"

        written = NetCDF.from_xarray(export(grouped_store), path=str(destination))

        assert set(written._readable_variable_names()) == {
            "flight_a_CO",
            "flight_a_CO_2",
            "flight_b_CO",
            "rec_Num_2",
        }, "the written store does not hold one array per exported variable"

    def test_the_written_values_still_belong_to_their_own_variable(self, grouped_store):
        """Flattening four names onto three stems must not swap two arrays.

        Args:
            grouped_store: The synthetic grouped `.nc` fixture.

        Test scenario:
            Names alone cannot catch a renaming that pairs `flight_b/CO`'s key
            with `flight_a/CO`'s data. Each array holds a distinct constant, so
            reading the numbers back settles it.
        """
        destination = grouped_store.parent / "round_trip.nc"
        NetCDF.from_xarray(export(grouped_store), path=str(destination))

        written = NetCDF.read_file(str(destination))
        actual = {
            name: float(np.asarray(written._read_variable(name)).ravel()[0])
            for name in written._readable_variable_names()
        }

        assert actual == {
            "flight_a_CO": STORE_VALUES["flight_a_CO"],
            "flight_a_CO_2": STORE_VALUES["flight_a/CO"],
            "flight_b_CO": STORE_VALUES["flight_b/CO"],
            "rec_Num_2": STORE_VALUES["rec/Num"],
        }, "an array came back under another variable's name"


class TestTheGroupPathSurvivesTheFlattening:
    """Reducing a qualified name to its leaf would merge the groups; it must not."""

    def test_two_groups_identical_leaves_stay_apart(self, grouped_store):
        """`flight_a/CO` and `flight_b/CO` are different measurements.

        Args:
            grouped_store: The synthetic grouped `.nc` fixture.

        Test scenario:
            The other repair of this shape in the same release
            (`add_variable`'s `_free_target_name`) reduces a qualified name to
            its leaf, because it writes into an existing root group. Doing that
            here would key both flights' `CO` as `CO` and drop one of them --
            so the export keeps the group as a prefix instead.
        """
        exported = export(grouped_store)

        assert {"flight_a_CO_2", "flight_b_CO"} <= set(exported.data_vars), (
            f"the two groups' CO did not both survive: {sorted(exported.data_vars)}"
        )
        assert float(exported["flight_a_CO_2"].values[0]) == STORE_VALUES["flight_a/CO"]
        assert float(exported["flight_b_CO"].values[0]) == STORE_VALUES["flight_b/CO"]

    def test_a_collision_with_a_root_array_is_suffixed_not_overwritten(
        self, grouped_store
    ):
        """The root's own `flight_a_CO` keeps its name; the group's yields.

        Args:
            grouped_store: The synthetic grouped `.nc` fixture.

        Test scenario:
            Both names want the key `flight_a_CO`. Writing the second over the
            first would lose an array silently, which is the failure the whole
            finding is about. The enumeration lists a group's own arrays before
            it recurses, so the root array wins deterministically.
        """
        exported = export(grouped_store)

        assert (
            float(exported["flight_a_CO"].values[0]) == STORE_VALUES["flight_a_CO"]
        ), "the root array lost its own name to a sub-group array"
        assert "flight_a_CO_2" in exported.data_vars, (
            "the loser was dropped, not renamed"
        )

    def test_a_collision_with_a_dimension_coordinate_is_suffixed(self, grouped_store):
        """Coordinates and data variables share one namespace in an `xr.Dataset`.

        Args:
            grouped_store: The synthetic grouped `.nc` fixture.

        Test scenario:
            `rec/Num` flattens onto the dimension coordinate `rec_Num`.
            Handing xarray both would raise a merge error out of the
            constructor, so the coordinate names are reserved before the data
            variables are keyed.
        """
        exported = export(grouped_store)

        assert list(exported.coords) == ["rec_Num"], "the dimension coordinate moved"
        np.testing.assert_array_equal(
            exported["rec_Num"].values,
            np.arange(4, dtype="float64"),
            err_msg="the coordinate carries the sub-group array's values",
        )
        assert float(exported["rec_Num_2"].values[0]) == STORE_VALUES["rec/Num"]

    def test_every_rewritten_name_is_announced(self, grouped_store):
        """Not only the suffixed ones -- which was the minority that never happens.

        This test used to be `test_the_disambiguation_is_announced` and only
        required the two suffixed entries. That expectation was wrong: the
        warning fired on `suffixed`, so the plain `/`-to-`_` rewrite -- the case
        that actually occurs, eight of the nine exported data variables on the
        suite's own `GROUPED` fixture -- was applied in silence, even though
        `get_variable` refuses a flattened key exactly as flatly as a suffixed
        one. Requiring only a subset of the rewrites let that stand.

        Args:
            grouped_store: The synthetic grouped `.nc` fixture.

        Test scenario:
            Every one of the store's three sub-group arrays is exported under a
            key its own container will not take, so all three have to appear:
            the two suffixed (`flight_a/CO`, `rec/Num`) and the one that is only
            flattened (`flight_b/CO`). The store name is the one `get_variable`
            takes, so the message has to carry it -- and it is still one
            aggregated warning, not one per variable.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            NetCDF.read_file(str(grouped_store)).to_xarray()

        renamed = [str(w.message) for w in caught if "renamed" in str(w.message)]

        assert len(renamed) == 1, f"expected one aggregated warning, got {len(renamed)}"
        for rewrite in (
            "flight_a/CO -> flight_a_CO_2",
            "rec/Num -> rec_Num_2",
            "flight_b/CO -> flight_b_CO",
        ):
            assert rewrite in renamed[0], f"{rewrite!r} went unannounced: {renamed[0]}"

    def test_a_flat_store_is_keyed_exactly_as_before(self):
        """The flattening must be invisible to a store that has no groups.

        Test scenario:
            Every name in a flat store already flattens to itself, so no key
            may move and no warning may appear. This is the guard against the
            repair leaking into the common case.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exported = NetCDF.read_file(str(FLAT)).to_xarray()

        assert "temperature" in exported.data_vars, sorted(exported.data_vars)
        assert not [w for w in caught if "renamed" in str(w.message)], (
            "a flat store was told its names had been rewritten"
        )


class TestTheWarningNamesTheCallersOwnLine:
    """A warning that blames a pyramids source line is unreadable and unfilterable."""

    def test_the_facade_spelling_points_at_the_caller(self):
        """`nc.to_xarray()` is the chain the constant `4` was counted along.

        Test scenario:
            This is the value the hard-coded constant got right, kept so the
            computed replacement is pinned to the same answer rather than only
            to the case the constant got wrong.
        """
        dataset = NetCDF.read_file(str(GROUPED))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset.to_xarray()

        skipped = [w for w in caught if "skipped" in str(w.message)]

        assert skipped, "the grouped fixture no longer warns about its dimension clash"
        assert Path(skipped[0].filename).resolve() == Path(__file__).resolve(), (
            f"the warning blamed {skipped[0].filename}, not the line that called it"
        )

    def test_the_engine_spelling_points_at_the_caller(self):
        """`nc.interop.to_xarray()` is one frame shorter, and the constant missed it.

        Test scenario:
            The engine is a supported spelling of the same call -- the suite
            already round-trips through it. With the counted constant its
            warning was attributed four frames up from a three-frame chain,
            i.e. to pytest's own call machinery, so `-W` filters keyed on the
            user's module never matched and the printed location was fiction.
        """
        dataset = NetCDF.read_file(str(GROUPED))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset.interop.to_xarray()

        skipped = [w for w in caught if "skipped" in str(w.message)]

        assert skipped, "the grouped fixture no longer warns about its dimension clash"
        assert Path(skipped[0].filename).resolve() == Path(__file__).resolve(), (
            f"the warning blamed {skipped[0].filename}, not the line that called it"
        )


class TestTheCfPromotionSpeaksTheExportedNamespace:
    """The CF classification is keyed by store names; the dataset is not.

    `_promote_cf_non_data_arrays` moves CF bounds and auxiliary-coordinate
    arrays out of `data_vars` by looking their classification names up in the
    dataset. Flattening a group path means those two namespaces are no longer
    the same one, so the lookup goes through the export map. No fixture in the
    suite reaches the mismatch -- a sub-group array that classifies as a
    coordinate is filtered out of the enumeration before it can become a data
    variable -- so this pins the join directly rather than through a store.
    """

    @staticmethod
    def _bounds_export():
        """A one-variable export plus the container whose CF says it is a bounds array.

        Returns:
            tuple: The `xr.Dataset` keyed by the export name, and a stand-in
            container whose CF classification is keyed by the store name.
        """
        exported = xr.Dataset(
            data_vars={"flight_a_CO_bnds": (("recNum",), np.zeros(4))},
            coords={"recNum": (("recNum",), np.arange(4.0))},
        )
        container = SimpleNamespace(
            meta_data=SimpleNamespace(
                cf=SimpleNamespace(classifications={"flight_a/CO_bnds": "bounds"})
            )
        )
        return exported, container

    def test_a_renamed_array_is_still_promoted(self):
        """The store name is `flight_a/CO_bnds`; the dataset knows `flight_a_CO_bnds`.

        Test scenario:
            Without the map the classification name matches nothing in
            `data_vars`, so a CF bounds array that used to become a coordinate
            would silently stay an ordinary variable -- and `to_netcdf` would
            write the CF relationship away, which is the exact loss this
            promotion exists to prevent.
        """
        exported, container = self._bounds_export()

        promoted = _promote_cf_non_data_arrays(
            exported, container, {"flight_a/CO_bnds": "flight_a_CO_bnds"}
        )

        assert "flight_a_CO_bnds" in promoted.coords, (
            "the renamed bounds array was left in data_vars"
        )

    def test_the_map_is_what_does_it(self):
        """An empty map must leave the array alone, or the assertion above is free.

        Test scenario:
            If the promotion matched for some other reason the test above would
            pass with the translation deleted. Running the same call with no
            mapping shows the translation is what moves the array.
        """
        exported, container = self._bounds_export()

        promoted = _promote_cf_non_data_arrays(exported, container, {})

        assert "flight_a_CO_bnds" in promoted.data_vars, (
            "the array moved without the export map, so the map is not load-bearing"
        )


class TestAnExportedKeyLeadsBackToTheStoreName:
    """The export handed out keys the container refuses, and recorded no way back.

    `get_variable` takes the store's name (`flight_a/CO`); the exported Dataset
    shows the flattened key (`flight_a_CO_2`). The obvious next call after
    `to_xarray()` on a grouped store -- `get_variable(a_name_I_just_saw)` --
    therefore failed for nearly every variable, and the flattening cannot be
    inverted from the key: group `a_b` + array `c` and group `a` + array `b_c`
    both flatten to `a_b_c`, and `export_names` never left the call.

    The store name is now stamped on each rewritten variable as
    `pyramids_store_name`. That, rather than the widened warning, is the half a
    program can use: a warning is text, and it is gone by the time the caller
    holds the Dataset.
    """

    def test_the_keys_the_container_refuses_are_the_common_case(self):
        """The premise: this is not a corner, it is nearly every variable.

        Test scenario:
            On the suite's own grouped fixture eight of the nine exported data
            variables are keyed by a name `get_variable` rejects. Asserting the
            recovery below without first showing the keys really do fail would
            leave the recovery testing nothing.
        """
        container = NetCDF.read_file(str(GROUPED))
        exported = export(GROUPED)

        refused = [
            name for name in exported.data_vars if not _accepts(container, str(name))
        ]

        assert len(refused) == len(exported.data_vars) - 1, (
            f"expected all but one key to be refused, {len(refused)} were: {refused}"
        )

    def test_every_exported_key_can_be_taken_back_to_a_readable_name(self):
        """The property the fix owes the user, asserted over the whole export.

        Test scenario:
            For each exported data variable, `attrs['pyramids_store_name']` when
            present and the key itself otherwise must be a name `get_variable`
            accepts. That is the rule a caller can write once and apply to every
            variable, without knowing which of them were rewritten.
        """
        container = NetCDF.read_file(str(GROUPED))
        exported = export(GROUPED)

        unreachable = [
            name
            for name, variable in exported.data_vars.items()
            if not _accepts(
                container, str(variable.attrs.get("pyramids_store_name", name))
            )
        ]

        assert unreachable == [], (
            f"these exported keys lead nowhere the container will read: {unreachable}"
        )

    def test_the_recorded_name_distinguishes_two_arrays_one_key_cannot(
        self, grouped_store
    ):
        """The flattening is not injective, so the key alone is not an answer.

        Args:
            grouped_store: The synthetic grouped `.nc` fixture.

        Test scenario:
            `flight_a_CO` (the root array) and `flight_a_CO_2` (the group's)
            differ only by a suffix the store never used, and `flight_b_CO` is a
            flattened path that looks exactly like a root name. Reading the
            recorded store name off each shows which is which -- and the root
            array, whose key *is* its store name, is deliberately left unstamped
            so a flat export carries no pyramids-private attribute at all.
        """
        exported = export(grouped_store)

        recorded = {
            str(name): variable.attrs.get("pyramids_store_name")
            for name, variable in exported.data_vars.items()
        }

        assert recorded == {
            "flight_a_CO": None,
            "flight_a_CO_2": "flight_a/CO",
            "flight_b_CO": "flight_b/CO",
            "rec_Num_2": "rec/Num",
        }, f"the exported keys cannot be inverted: {recorded}"

    def test_a_flat_store_carries_no_provenance_attribute(self):
        """The note must not leak into the case that has nothing to record.

        Test scenario:
            Every key in a flat store already is the store name, so stamping it
            would put a pyramids-private attribute on every variable of every
            ordinary file -- and `from_xarray` would then write it out.
        """
        exported = export(FLAT)

        stamped = [
            str(name)
            for name, variable in exported.data_vars.items()
            if "pyramids_store_name" in variable.attrs
        ]

        assert stamped == [], f"a flat export was annotated for no reason: {stamped}"

    def test_the_note_is_not_written_into_the_file_it_describes(self, grouped_store):
        """It names the source store, which the written file is not.

        Args:
            grouped_store: The synthetic grouped `.nc` fixture.

        Test scenario:
            In the file `from_xarray` writes, the variable really is called what
            the key says -- it is flat, and it has no groups. Carrying
            `pyramids_store_name` across would leave that file asserting a group
            path it does not have, and a second export of it would repeat the
            claim. So the attribute is dropped on the way in, and re-exporting
            the written file produces no annotation.
        """
        destination = grouped_store.parent / "round_trip.nc"
        NetCDF.from_xarray(export(grouped_store), path=str(destination))

        re_exported = export(destination)

        assert all(
            "pyramids_store_name" not in variable.attrs
            for variable in re_exported.data_vars.values()
        ), "the source store's names were written into the flat file that replaced it"


def _accepts(container: NetCDF, name: str) -> bool:
    """Whether `get_variable` will read `name` from `container`.

    Args:
        container: The store the export came from.
        name: A candidate variable name.

    Returns:
        bool: True when `get_variable(name)` returns rather than raising.
    """
    accepted = True
    try:
        container.get_variable(name)
    except ValueError:
        accepted = False
    return accepted
