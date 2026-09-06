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

    def test_the_disambiguation_is_announced(self, grouped_store):
        """A numeric suffix is not derivable from the store name, so it is said aloud.

        Args:
            grouped_store: The synthetic grouped `.nc` fixture.

        Test scenario:
            `flight_a/CO -> flight_a_CO_2` cannot be guessed from either name
            on its own, and the store name is still the one `get_variable`
            takes. Renaming silently would leave a user unable to tell which
            of two `flight_a_CO*` columns came from where.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            NetCDF.read_file(str(grouped_store)).to_xarray()

        renamed = [str(w.message) for w in caught if "renamed" in str(w.message)]

        assert renamed, "the suffixed names were applied without a word"
        assert "flight_a/CO -> flight_a_CO_2" in renamed[0], renamed[0]
        assert "rec/Num -> rec_Num_2" in renamed[0], renamed[0]

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
