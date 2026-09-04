"""One list of the names a coordinate axis goes by.

The GeoTIFF path knew eleven spellings of the x axis -- `x`, `lon`, `rlon` from
a rotated-pole grid, `nav_lon` from NEMO, `easting` from a projected one. The
Zarr reader kept its own list of six.

That reader picks its data array by elimination: whatever is not a known
coordinate. A NEMO store's `nav_lon` / `nav_lat` are full 2-D fields, so with
them missing from its list they were offered as candidates alongside the real
variable -- and since the tie is broken on dimension count, `nav_lat` won.
Reading such a store returned a latitude field as the raster, silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._axes import AXIS_NAMES, X_AXIS_NAMES, Y_AXIS_NAMES
from pyramids.dataset.dataset import (
    _AXIS_VARIABLE_NAMES,
    _X_AXIS_NAMES,
    _Y_AXIS_NAMES,
)
from pyramids.dataset.ops._geobox_zarr import (
    _ALWAYS_COORDS,
    _NEVER_DATA_ARRAYS,
    _NON_DATA_ARRAYS,
    detect_data_var,
)

try:
    import zarr
except ImportError:  # pragma: no cover - exercised only without the extra
    zarr = None

# Imported inside a guard rather than through a module-level `importorskip`, so
# the file still collects without the `[lazy]` extra; the tests that need a real
# store carry `@pytest.mark.lazy` and are selected with `-m lazy`.
needs_zarr = pytest.mark.skipif(zarr is None, reason="requires the [lazy] extra")


class TestTheVocabularyIsOneList:
    """The three names in `dataset` are the shared ones, not copies."""

    @pytest.mark.core
    def test_the_two_halves_are_the_shared_ones(self):
        """A copy would drift the moment either list gained a spelling."""
        assert _X_AXIS_NAMES is X_AXIS_NAMES
        assert _Y_AXIS_NAMES is Y_AXIS_NAMES

    @pytest.mark.core
    def test_the_union_is_derived_rather_than_restated(self):
        """It used to be a third literal list that could disagree.

        Test scenario:
            The union was written out by hand next to the two halves it was
            meant to combine, with a comment asserting they partitioned it
            exactly. Deriving it makes that true by construction.
        """
        assert _AXIS_VARIABLE_NAMES == _X_AXIS_NAMES | _Y_AXIS_NAMES

    @pytest.mark.core
    def test_the_halves_do_not_overlap(self):
        """A name cannot be both axes, and a pairing rule relies on that.

        Test scenario:
            The x/y split exists so a rule can require a *pair* of axes to
            agree before it fires. A spelling landing in both halves would let
            one array satisfy that pair on its own.
        """
        assert not (X_AXIS_NAMES & Y_AXIS_NAMES)

    @pytest.mark.core
    @pytest.mark.parametrize(
        "name",
        ["nav_lon", "nav_lat", "rlon", "rlat", "easting", "northing", "xc", "yc"],
    )
    def test_the_less_common_spellings_are_all_there(self, name: str):
        """These are the ones the shorter list dropped.

        Args:
            name: A coordinate spelling from a real model convention.

        Test scenario:
            NEMO writes `nav_lon`/`nav_lat`, a rotated-pole grid writes
            `rlon`/`rlat`, a projected one `easting`/`northing`. Each has to be
            recognised as a coordinate by every reader, not just one.
        """
        assert name in AXIS_NAMES


class TestTheZarrReaderKnowsThemAll:
    """The reader whose short list caused the misidentification."""

    @pytest.mark.lazy
    def test_its_non_data_set_covers_the_shared_vocabulary(self):
        """Derived from the shared list, so it cannot fall behind again."""
        assert AXIS_NAMES <= _NON_DATA_ARRAYS

    @pytest.mark.lazy
    def test_it_still_excludes_the_non_spatial_arrays(self):
        """The reader's own additions must survive the derivation.

        Test scenario:
            `time`, `band` and the CRS arrays are not axes but are not data
            either. Composing the set from the axis vocabulary must add to
            them, not replace them.
        """
        assert {"time", "band", "crs"} <= _NON_DATA_ARRAYS

    @pytest.mark.lazy
    @needs_zarr
    def test_a_nemo_store_yields_its_variable_not_a_coordinate(self, tmp_path):
        """The regression, on the store shape that triggered it.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `nav_lon` / `nav_lat` are 2-D coordinate fields, the same shape as
            the variable. Unrecognised, they became candidates, and the
            dimension-count tie-break returned `nav_lat` as the raster data.
        """
        store = tmp_path / "nemo.zarr"
        group = zarr.open_group(str(store), mode="w")
        for name in ("nav_lat", "nav_lon", "sst"):
            array = group.create_array(name, shape=(6, 8), dtype="float32")
            array[:] = np.ones((6, 8), dtype=np.float32)

        assert detect_data_var(group) == "sst"

    @pytest.mark.lazy
    @needs_zarr
    def test_a_plain_x_y_store_is_unaffected(self, tmp_path):
        """The common case must not have moved.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            A store with 1-D `x` / `y` coordinates was always read correctly.
            Widening the vocabulary must not change which array it picks.
        """
        store = tmp_path / "plain.zarr"
        group = zarr.open_group(str(store), mode="w")
        group.create_array("x", shape=(8,), dtype="float64")[:] = np.arange(8.0)
        group.create_array("y", shape=(6,), dtype="float64")[:] = np.arange(6.0)
        array = group.create_array("temperature", shape=(6, 8), dtype="float32")
        array[:] = np.ones((6, 8), dtype=np.float32)

        assert detect_data_var(group) == "temperature"

    @pytest.mark.lazy
    @needs_zarr
    def test_a_store_of_only_coordinates_still_raises(self, tmp_path):
        """Widening the list must not turn "no data" into a wrong answer.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            With every array recognised as a coordinate there is no data array
            to return, which has to stay an explicit error rather than becoming
            an arbitrary pick.
        """
        store = tmp_path / "coords_only.zarr"
        group = zarr.open_group(str(store), mode="w")
        for name in ("nav_lat", "nav_lon"):
            array = group.create_array(name, shape=(6, 8), dtype="float32")
            array[:] = np.ones((6, 8), dtype=np.float32)

        with pytest.raises(KeyError, match="no data array"):
            detect_data_var(group)


class TestAnAxisNameCanStillBeAData_Array:
    """The vocabulary is a preference here, not a veto.

    `east`, `north`, `long` and `x_dim` are coordinate spellings *and* ordinary
    names for a data array -- an eastward wind or current component, say.
    Excluding them outright made a store whose only variable is called `east`
    raise "no data array found", which is worse than the problem being solved.
    """

    @pytest.mark.lazy
    @needs_zarr
    @pytest.mark.parametrize(
        ("names", "expected"),
        [
            (["east", "north", "x", "y"], "east"),
            (["northing", "x", "y"], "northing"),
            (["long", "x", "y"], "long"),
            (["x_dim", "x", "y"], "x_dim"),
        ],
    )
    def test_a_store_whose_variable_is_axis_named_is_still_readable(
        self, tmp_path, names, expected
    ):
        """Each of these raised `KeyError` when the vocabulary was a veto.

        Args:
            tmp_path: Fixture supplying a temporary directory.
            names: The arrays in the store.
            expected: The array that must be chosen as the data.

        Test scenario:
            With no non-coordinate-named array to choose, the reader falls back
            to the narrow list that was always excluded, so it picks the same
            array it did before the vocabulary was shared.
        """
        store = tmp_path / "axis_named.zarr"
        group = zarr.open_group(str(store), mode="w")
        for name in names:
            array = group.create_array(name, shape=(6, 8), dtype="float32")
            array[:] = np.ones((6, 8), dtype=np.float32)

        assert detect_data_var(group) == expected

    @pytest.mark.lazy
    @needs_zarr
    def test_a_real_variable_still_beats_an_axis_named_one(self, tmp_path):
        """The preference only applies when there is something to prefer.

        Test scenario:
            With both `east` and `sst` present, `sst` is not a coordinate
            spelling at all, so the first tier picks it and `east` is treated
            as the coordinate it probably is.
        """
        store = tmp_path / "mixed.zarr"
        group = zarr.open_group(str(store), mode="w")
        for name in ("east", "north", "sst"):
            array = group.create_array(name, shape=(6, 8), dtype="float32")
            array[:] = np.ones((6, 8), dtype=np.float32)

        assert detect_data_var(group) == "sst"

    @pytest.mark.core
    def test_the_three_tiers_are_nested(self):
        """The tiers are nested, so the fallback can only ever widen.

        Test scenario:
            Names that are only ever coordinates stay excluded in both tiers;
            the ambiguous ones sit between, excluded first and allowed second.
        """
        assert _ALWAYS_COORDS <= _NEVER_DATA_ARRAYS <= _NON_DATA_ARRAYS
