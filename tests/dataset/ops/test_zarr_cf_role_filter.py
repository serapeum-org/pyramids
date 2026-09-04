"""The Zarr reader learns what a CF store says about its own arrays.

`detect_data_var` picked by elimination on *names*: whatever is not a known
coordinate spelling, highest dimension first. A CF dataset written by
`xarray.Dataset.to_zarr` carries `lat_bnds` / `lon_bnds` alongside the data,
and those are 2-D too -- so they tied with the data variable on dimension
count and the alphabetical tie-break handed back `lat_bnds` as the raster.

Names are the last resort, not the first. A CF store says which of its arrays
are supporting cast: a coordinate names its `bounds`, a data variable names
its auxiliary `coordinates`, its `ancillary_variables`, its `cell_measures`
and its `grid_mapping`. Anything pointed at that way is not the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset.ops._geobox_zarr import _cf_referenced_names, detect_data_var

try:
    import zarr
except ImportError:  # pragma: no cover - exercised only without the extra
    zarr = None

# Guarded rather than a module-level `importorskip`, so the file still collects
# without the `[lazy]` extra; the tests that need a real store are marked.
needs_zarr = pytest.mark.skipif(zarr is None, reason="requires the [lazy] extra")


def _store(tmp_path, spec: dict[str, tuple[int, ...]], attrs=None):
    """Build a zarr group of zero-filled arrays with optional CF attributes.

    Args:
        tmp_path: The pytest temporary directory to write under.
        spec: `{array name: shape}` for every array in the group.
        attrs: Optional `{array name: {attr: value}}` to stamp afterwards.

    Returns:
        zarr.Group: The open group, ready for `detect_data_var`.
    """
    group = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    for name, shape in spec.items():
        array = group.create_array(name, shape=shape, dtype="float32")
        array[...] = np.zeros(shape, dtype=np.float32)
    for name, values in (attrs or {}).items():
        group[name].attrs.update(values)
    return group


class TestBoundsArraysLoseToTheDataVariable:
    """The reported failure: a CF store read its `lat_bnds` as the raster."""

    @pytest.mark.lazy
    @needs_zarr
    def test_a_cf_store_written_by_xarray_reads_its_data_variable(self, tmp_path):
        """The whole shape of the problem, in the store that produces it.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `tas(4, 6)` against `lat_bnds(4, 2)` and `lon_bnds(6, 2)`: all three
            are 2-D, so dimension count cannot separate them and `lat_bnds`
            sorts first. Every CF dataset `xarray` writes has this shape.
        """
        group = _store(
            tmp_path,
            {
                "tas": (4, 6),
                "lat_bnds": (4, 2),
                "lon_bnds": (6, 2),
                "x": (6,),
                "y": (4,),
            },
        )

        assert detect_data_var(group) == "tas"

    @pytest.mark.lazy
    @needs_zarr
    def test_bounds_declared_under_another_name_are_excluded_too(self, tmp_path):
        """The `_bnds` suffix is a convention; the attribute is the contract.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            A store may name its bounds array anything and point at it from
            the coordinate's `bounds` attribute. Excluding only the suffix
            spelling would read `edges` as the raster.
        """
        group = _store(
            tmp_path,
            {"tas": (4, 6), "edges": (4, 2), "lat": (4,)},
            {"lat": {"bounds": "edges"}},
        )

        assert detect_data_var(group) == "tas"

    @pytest.mark.lazy
    @needs_zarr
    def test_auxiliary_coordinates_lose_to_the_variable_that_names_them(self, tmp_path):
        """A curvilinear store, where the aux coords are full 2-D fields.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `lat_rho` / `lon_rho` are the same shape as `salt` and sort ahead
            of it. `salt` naming them in `coordinates` is what says they are
            coordinates -- their names alone do not.
        """
        group = _store(
            tmp_path,
            {"salt": (4, 6), "lat_rho": (4, 6), "lon_rho": (4, 6)},
            {"salt": {"coordinates": "lat_rho lon_rho"}},
        )

        assert detect_data_var(group) == "salt"

    @pytest.mark.lazy
    @needs_zarr
    def test_a_cell_measure_loses_to_the_variable_that_names_it(self, tmp_path):
        """`cell_measures` is `"measure: name"` pairs, not a plain name list.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `areacella` is a real 2-D field and sorts before `tas`. Reading
            the attribute as a flat list would take `area:` as a name and
            leave `areacella` a candidate.
        """
        group = _store(
            tmp_path,
            {"tas": (4, 6), "areacella": (4, 6)},
            {"tas": {"cell_measures": "area: areacella"}},
        )

        assert detect_data_var(group) == "tas"

    @pytest.mark.lazy
    @needs_zarr
    def test_an_ancillary_variable_loses_to_the_variable_that_names_it(self, tmp_path):
        """A quality flag is not the raster, whatever it sorts as.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            GOES-style `CMI` with its `DQF` status flag: both 2-D, and `DQF`
            sorts first.
        """
        group = _store(
            tmp_path,
            {"CMI": (4, 6), "DQF": (4, 6)},
            {"CMI": {"ancillary_variables": "DQF"}},
        )

        assert detect_data_var(group) == "CMI"


class TestTheFilterCannotEraseTheOnlyCandidate:
    """Excluding by reference must not exclude everything."""

    @pytest.mark.lazy
    @needs_zarr
    def test_an_array_does_not_demote_itself(self, tmp_path):
        """A malformed store that names itself still has a data array.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `tas` listing `tas` among its own `coordinates` is not valid CF,
            but a reader that took it literally would erase the only
            candidate the store has and raise on a perfectly readable array.
        """
        group = _store(tmp_path, {"tas": (4, 6)}, {"tas": {"coordinates": "tas"}})

        assert detect_data_var(group) == "tas"

    @pytest.mark.lazy
    @needs_zarr
    def test_a_store_of_only_bounds_raises_rather_than_picking_one(self, tmp_path):
        """No data array is an error, the same as a store of coordinates.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            Nothing here is a raster. Returning `lat_bnds` because it is what
            was left would be the original bug wearing a different hat.
        """
        group = _store(tmp_path, {"lat_bnds": (4, 2), "lon_bnds": (6, 2)})

        with pytest.raises(KeyError, match="no data array"):
            detect_data_var(group)

    @pytest.mark.lazy
    @needs_zarr
    def test_the_declared_grid_mapping_array_still_wins_outright(self, tmp_path):
        """Rule 2 runs before any of this, and must keep doing so.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            An array carrying `grid_mapping` is the store saying which one is
            georeferenced. That is an explicit declaration, so it outranks
            every inference -- including a name that would otherwise be
            filtered.
        """
        group = _store(
            tmp_path,
            {"east": (4, 6), "tas": (4, 6)},
            {"east": {"grid_mapping": "spatial_ref"}},
        )

        assert detect_data_var(group) == "east"


class TestWhatCountsAsReferenced:
    """The helper on its own, over the four attribute shapes."""

    @pytest.mark.lazy
    @needs_zarr
    def test_it_collects_every_reference_kind(self, tmp_path):
        """One pass, four attributes, plus the suffix convention.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            A store exercising all of them at once, so a regression in any
            single branch shows up here rather than only in whichever
            end-to-end case happens to use it.
        """
        group = _store(
            tmp_path,
            {
                "tas": (4, 6),
                "time_bnds": (3, 2),
                "edges": (4, 2),
                "lat": (4,),
                "aux": (4, 6),
                "flags": (4, 6),
                "areacella": (4, 6),
                "spatial_ref": (),
            },
            {
                "lat": {"bounds": "edges"},
                "tas": {
                    "coordinates": "aux",
                    "ancillary_variables": "flags",
                    "cell_measures": "area: areacella",
                    "grid_mapping": "spatial_ref",
                },
            },
        )

        referenced = _cf_referenced_names(group, list(group.array_keys()))

        assert referenced == {
            "time_bnds",
            "edges",
            "aux",
            "flags",
            "areacella",
            "spatial_ref",
        }

    @pytest.mark.lazy
    @needs_zarr
    def test_a_store_with_no_cf_attributes_references_nothing(self, tmp_path):
        """No opinion where the store expresses none.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            A bare store must come through the filter untouched, so the
            name-based rules decide exactly as they did before.
        """
        group = _store(tmp_path, {"sst": (6, 8), "nav_lat": (6, 8)})

        assert _cf_referenced_names(group, list(group.array_keys())) == set()
