"""Tests for the grouped read side of `from_stac`.

Covers STAC-06 (a flexible `groupby`: reserved names, property keys and
callables), the STAC-10 follow-up (`method=` on the grouped mosaic), STAC-16
(`fuse_func`, the in-place odc-style fuser) and STAC-09 (`errors_as_nodata`,
substituting a no-data plane for a present-but-unreadable asset).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from pyramids.base._errors import StacAssetError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, DatasetCollection, Grid
from pyramids.dataset._stac import (
    _group_key,
    _resolve_groupby,
    _sorted_group_keys,
)

pytestmark = pytest.mark.core


@pytest.fixture
def matching_grid():
    """A `Grid` matching the `three_local_items` rasters (3x3, cell 1, EPSG:4326)."""
    return Grid(crs=4326, resolution=1.0, bounds=(0.0, 0.0, 3.0, 3.0))


@pytest.fixture
def holed_items(tmp_path):
    """Two same-grid, same-group items where the first has a no-data hole.

    The first raster is ``1.0`` with its left column set to its declared no-data
    (``-9999``); the second is ``9.0`` everywhere. Both carry ``orbit=1``, so a
    property `groupby` puts them in one group and a fuser has a real hole to
    fill.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        list[dict]: Two STAC item dicts sharing one group key.
    """
    first = np.full((3, 3), 1.0, dtype="float32")
    first[:, 0] = -9999.0
    items = []
    for index, array in enumerate((first, np.full((3, 3), 9.0, dtype="float32"))):
        path = str(tmp_path / f"holed{index}.tif")
        Dataset.from_array(
            array,
            no_data_value=-9999.0,
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
        ).to_file(path)
        items.append(
            {
                "id": f"holed{index}",
                "bbox": [0.0, 0.0, 3.0, 3.0],
                "properties": {
                    "datetime": f"2023-07-0{index + 1}T00:00:00Z",
                    "orbit": 1,
                },
                "assets": {"data": {"href": path}},
            }
        )
    return items


@pytest.fixture
def offgrid_items(tmp_path):
    """Two same-group items whose rasters sit on grids half a cell apart.

    Both are 3x3 EPSG:4326 rasters at cell size 1 carrying ``orbit=1``, but the
    second starts at ``(0.5, 3.5)`` instead of ``(0.0, 3.0)``, so a fuser can
    only see them together once the second has been aligned onto the first.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        list[dict]: Two STAC item dicts sharing one group key.
    """
    corners = ((0.0, 3.0), (0.5, 3.5))
    items = []
    for index, (value, corner) in enumerate(zip((1.0, 9.0), corners)):
        path = str(tmp_path / f"offgrid{index}.tif")
        Dataset.from_array(
            np.full((3, 3), value, dtype="float32"),
            no_data_value=-9999.0,
            geo_ref=GeoReference(top_left_corner=corner, cell_size=1.0, epsg=4326),
        ).to_file(path)
        items.append(
            {
                "id": f"offgrid{index}",
                "bbox": [0.0, 0.0, 3.0, 3.0],
                "properties": {
                    "datetime": f"2023-08-0{index + 1}T00:00:00Z",
                    "orbit": 1,
                },
                "assets": {"data": {"href": path}},
            }
        )
    return items


def _break_asset(items, index, tmp_path, name="missing.tif"):
    """Return a copy of `items` whose item `index` points at a non-existent file."""
    broken = copy.deepcopy(items)
    broken[index]["assets"]["data"]["href"] = str(tmp_path / name)
    return broken


class TestResolveGroupby:
    """STAC-06: the single `groupby` -> key-function resolver."""

    def test_solar_day_name(self, three_local_items):
        """'solar_day' resolves to the solar-day label.

        Test scenario:
            The first item is a 2023-06-01 acquisition at lon ~1.5.
        """
        key_fn = _resolve_groupby("solar_day")
        assert key_fn(three_local_items[0]) == "2023-06-01", (
            f"expected the solar-day label, got {key_fn(three_local_items[0])!r}"
        )

    def test_id_name(self, three_local_items):
        """'id' resolves to the item id.

        Test scenario:
            The fixture's first item is id 'scene0'.
        """
        key_fn = _resolve_groupby("id")
        assert key_fn(three_local_items[0]) == "scene0", (
            f"expected the item id, got {key_fn(three_local_items[0])!r}"
        )

    def test_time_name(self, three_local_items):
        """'time' resolves to the item's ISO 8601 datetime.

        Test scenario:
            properties.datetime '2023-06-01T00:00:00Z' -> a UTC isoformat.
        """
        key_fn = _resolve_groupby("time")
        assert key_fn(three_local_items[0]) == "2023-06-01T00:00:00+00:00", (
            f"expected an isoformat datetime, got {key_fn(three_local_items[0])!r}"
        )

    def test_property_key(self, three_local_items):
        """Any other string is read from the item's properties.

        Test scenario:
            'orbit' -> the property value 1.
        """
        key_fn = _resolve_groupby("orbit")
        assert key_fn(three_local_items[0]) == 1, (
            f"expected the orbit property, got {key_fn(three_local_items[0])!r}"
        )

    def test_callable_passthrough(self, three_local_items):
        """A callable is used as the key function verbatim.

        Test scenario:
            A 1-arg lambda over properties is returned unchanged.
        """

        def by_orbit(item):
            return item["properties"]["orbit"]

        assert _resolve_groupby(by_orbit) is by_orbit, (
            "a callable groupby should be used as the key function itself"
        )

    def test_reserved_names_shadow_a_property(self, three_local_items):
        """The reserved 'time' wins over a property literally named 'time'.

        Test scenario:
            An item declaring properties['time'] = 'noon' still groups by its
            datetime, and the callable escape hatch reaches the property.
        """
        item = copy.deepcopy(three_local_items[0])
        item["properties"]["time"] = "noon"
        assert _resolve_groupby("time")(item) == "2023-06-01T00:00:00+00:00", (
            "the reserved 'time' must win over a same-named property"
        )
        escape = _resolve_groupby(lambda it: it["properties"]["time"])
        assert escape(item) == "noon", (
            "a callable must still reach a property shadowed by a reserved name"
        )

    def test_unsupported_type_raises(self):
        """A groupby that is neither a string nor a callable raises ValueError.

        Test scenario:
            groupby=3 is not a spec at all.
        """
        with pytest.raises(ValueError, match="groupby must be"):
            _resolve_groupby(3)

    def test_missing_property_raises(self, three_local_items):
        """A property key absent from the item raises, naming the item.

        Test scenario:
            'no_such' is on no item.
        """
        key_fn = _resolve_groupby("no_such")
        with pytest.raises(ValueError, match="absent on item"):
            key_fn(three_local_items[0])


class TestGroupKey:
    """STAC-06: key-function failures are reported against the offending item."""

    def test_raising_callable_names_the_item(self, three_local_items):
        """A callable that blows up is re-raised as a ValueError naming the item.

        Test scenario:
            A key function raising TypeError -> ValueError mentioning 'scene0'.
        """

        def boom(_item):
            raise TypeError("nope")

        with pytest.raises(ValueError, match="scene0"):
            _group_key(three_local_items[0], boom)

    def test_unhashable_key_raises(self, three_local_items):
        """A key that cannot be hashed is rejected with a clear message.

        Test scenario:
            A key function returning a list.
        """
        with pytest.raises(ValueError, match="not hashable"):
            _group_key(three_local_items[0], lambda _it: ["a", "b"])

    def test_sorted_keys_fall_back_to_str(self):
        """Mutually-unorderable keys sort by their string form instead of raising.

        Test scenario:
            {1, '1'} cannot be compared, so the order comes from str().
        """
        assert _sorted_group_keys({1, "1"}) == [1, "1"] or _sorted_group_keys(
            {1, "1"}
        ) == ["1", 1], (
            f"expected a str-ordered fallback, got {_sorted_group_keys({1, '1'})!r}"
        )


class TestFlexibleGroupby:
    """STAC-06: from_stac(groupby=...) collapses items into grouped timesteps."""

    def test_none_keeps_one_timestep_per_item(self, three_local_items):
        """groupby=None is unchanged: one timestep per item.

        Test scenario:
            3 items -> time_length 3.
        """
        coll = DatasetCollection.from_stac(three_local_items, asset="data")
        assert coll.time_length == 3, (
            f"expected 3 ungrouped timesteps, got {coll.time_length}"
        )

    def test_groupby_id_one_timestep_per_item(self, three_local_items):
        """groupby='id' gives one timestep per item id.

        Test scenario:
            3 distinct ids -> time_length 3.
        """
        coll = DatasetCollection.from_stac(
            three_local_items, asset="data", groupby="id"
        )
        assert coll.time_length == 3, f"expected 3 id groups, got {coll.time_length}"

    def test_groupby_time(self, three_local_items):
        """groupby='time' gives one timestep per distinct datetime.

        Test scenario:
            3 distinct datetimes -> time_length 3.
        """
        coll = DatasetCollection.from_stac(
            three_local_items, asset="data", groupby="time"
        )
        assert coll.time_length == 3, f"expected 3 time groups, got {coll.time_length}"

    def test_groupby_property_key(self, three_local_items):
        """groupby on a property key collapses the items sharing a value.

        Test scenario:
            orbit 1, 1, 2 -> 2 groups.
        """
        coll = DatasetCollection.from_stac(
            three_local_items, asset="data", groupby="orbit"
        )
        assert coll.time_length == 2, f"expected 2 orbit groups, got {coll.time_length}"

    def test_groupby_callable(self, three_local_items):
        """A 1-arg callable groups exactly like the equivalent property key.

        Test scenario:
            lambda over properties['orbit'] -> 2 groups.
        """
        coll = DatasetCollection.from_stac(
            three_local_items,
            asset="data",
            groupby=lambda item: item["properties"]["orbit"],
        )
        assert coll.time_length == 2, (
            f"expected 2 callable groups, got {coll.time_length}"
        )

    def test_groups_are_emitted_in_sorted_key_order(self, three_local_items):
        """Groups are stacked in sorted-key order, deterministically.

        Test scenario:
            orbit 1 (values 1 and 2, first-valid -> 1) precedes orbit 2 (3).
        """
        coll = DatasetCollection.from_stac(
            three_local_items, asset="data", groupby="orbit"
        )
        first = float(coll.datasets[0].read_array()[0, 0])
        second = float(coll.datasets[1].read_array()[0, 0])
        assert first == pytest.approx(1.0), (
            f"orbit 1 should come first and keep the first-valid 1.0, got {first}"
        )
        assert second == pytest.approx(3.0), (
            f"orbit 2 should come second with 3.0, got {second}"
        )

    def test_missing_property_raises(self, three_local_items):
        """A property key no item declares raises, naming the item.

        Test scenario:
            groupby='no_such'.
        """
        with pytest.raises(ValueError, match="absent on item"):
            DatasetCollection.from_stac(
                three_local_items, asset="data", groupby="no_such"
            )

    def test_multi_asset_rejected(self, three_local_items):
        """Grouping stays single-asset, with a clear message.

        Test scenario:
            asset=["data", "data"] with a groupby.
        """
        with pytest.raises(ValueError, match="single asset"):
            DatasetCollection.from_stac(
                three_local_items, asset=["data", "data"], groupby="orbit"
            )

    def test_no_items_raises(self):
        """An empty item list raises rather than building an empty cube.

        Test scenario:
            from_stac([], groupby='id').
        """
        with pytest.raises(ValueError, match="received no items"):
            DatasetCollection.from_stac([], asset="data", groupby="id")

    def test_skip_missing_drops_items_lacking_the_asset(self, three_local_items):
        """skip_missing drops a grouped item that lacks the asset key.

        Test scenario:
            The orbit-2 item loses its 'data' asset -> only the orbit-1 group.
        """
        items = copy.deepcopy(three_local_items)
        items[2]["assets"] = {}
        coll = DatasetCollection.from_stac(
            items, asset="data", groupby="orbit", skip_missing=True
        )
        assert coll.time_length == 1, (
            f"expected only the orbit-1 group to survive, got {coll.time_length}"
        )

    def test_skip_missing_with_no_surviving_item_raises(self, three_local_items):
        """Skipping every item leaves no group to build, which is an error.

        Test scenario:
            All three items lose their 'data' asset while skip_missing is on, so
            the grouping ends up empty rather than silently building nothing.
        """
        items = copy.deepcopy(three_local_items)
        for item in items:
            item["assets"] = {}
        with pytest.raises(ValueError, match="produced no groups"):
            DatasetCollection.from_stac(
                items, asset="data", groupby="orbit", skip_missing=True
            )

    def test_missing_asset_raises_by_default(self, three_local_items):
        """Without skip_missing a grouped item lacking the asset still raises.

        Test scenario:
            The orbit-2 item loses its 'data' asset.
        """
        items = copy.deepcopy(three_local_items)
        items[2]["assets"] = {}
        with pytest.raises(StacAssetError):
            DatasetCollection.from_stac(items, asset="data", groupby="orbit")


class TestGroupedMosaicMethod:
    """STAC-10 follow-up: method= reaches the grouped mosaic."""

    def test_default_is_first(self, three_local_items):
        """The default method is 'first', so today's behaviour is unchanged.

        Test scenario:
            The orbit-1 group of values 1 and 2 keeps 1.
        """
        coll = DatasetCollection.from_stac(
            three_local_items, asset="data", groupby="orbit"
        )
        assert float(coll.datasets[0].read_array()[0, 0]) == pytest.approx(1.0), (
            "the default grouped mosaic must still be first-valid"
        )

    def test_mean_reduces_the_group(self, three_local_items):
        """method='mean' averages the overlapping sources of a group.

        Test scenario:
            The orbit-1 group of values 1 and 2 -> 1.5.
        """
        coll = DatasetCollection.from_stac(
            three_local_items, asset="data", groupby="orbit", method="mean"
        )
        value = float(coll.datasets[0].read_array()[0, 0])
        assert value == pytest.approx(1.5), f"expected the mean 1.5, got {value}"

    def test_max_reduces_the_group(self, three_local_items):
        """method='max' takes the per-pixel maximum of a group.

        Test scenario:
            The orbit-1 group of values 1 and 2 -> 2.
        """
        coll = DatasetCollection.from_stac(
            three_local_items, asset="data", groupby="orbit", method="max"
        )
        value = float(coll.datasets[0].read_array()[0, 0])
        assert value == pytest.approx(2.0), f"expected the max 2.0, got {value}"

    def test_method_without_groupby_raises(self, three_local_items):
        """A method with no groupby raises instead of being silently ignored.

        Test scenario:
            method='mean' and groupby=None.
        """
        with pytest.raises(ValueError, match="only applies to a grouped mosaic"):
            DatasetCollection.from_stac(three_local_items, asset="data", method="mean")

    def test_unknown_method_raises(self, three_local_items):
        """An unsupported method is rejected by the merge it is forwarded to.

        Test scenario:
            method='nonsense'.
        """
        with pytest.raises(ValueError, match="method must be one of"):
            DatasetCollection.from_stac(
                three_local_items, asset="data", groupby="orbit", method="nonsense"
            )


class TestFuseFunc:
    """STAC-16: a caller-supplied in-place fuser replaces the grouped mosaic."""

    @staticmethod
    def _keep_first(dst, src):
        """Copy `src` only where `dst` is still no-data (NaN)."""
        empty = np.isnan(dst)
        dst[empty] = src[empty]

    def test_fuses_each_group(self, three_local_items):
        """A fuser runs per group and keeps one timestep per group.

        Test scenario:
            groupby='orbit' with a keep-first fuser -> 2 timesteps.
        """
        coll = DatasetCollection.from_stac(
            three_local_items,
            asset="data",
            groupby="orbit",
            fuse_func=self._keep_first,
        )
        assert coll.time_length == 2, (
            f"expected 2 fused orbit groups, got {coll.time_length}"
        )

    def test_fuser_fills_nodata_holes(self, holed_items):
        """The fuser sees no-data as NaN and can fill it from the next source.

        Test scenario:
            Item 1's left column is no-data; the keep-first fuser fills it with
            item 2's 9.0 while the rest stays item 1's 1.0.
        """
        coll = DatasetCollection.from_stac(
            holed_items, asset="data", groupby="orbit", fuse_func=self._keep_first
        )
        fused = coll.datasets[0].read_array()
        assert float(fused[0, 0]) == pytest.approx(9.0), (
            f"the hole should be filled from the second source, got {fused[0, 0]}"
        )
        assert float(fused[0, 1]) == pytest.approx(1.0), (
            f"the covered pixels should stay the first source's, got {fused[0, 1]}"
        )

    def test_fuser_sees_sources_in_item_order(self, holed_items):
        """The fuser is applied pairwise, in item order, within a group.

        Test scenario:
            A recording fuser over a 2-item group is called once, with the
            second item's values as `src`.
        """
        seen = []

        def record(dst, src):
            seen.append(float(np.nanmax(src)))
            self._keep_first(dst, src)

        DatasetCollection.from_stac(
            holed_items, asset="data", groupby="orbit", fuse_func=record
        )
        assert seen == [pytest.approx(9.0)], (
            f"expected one pairwise call with the second source, got {seen}"
        )

    def test_offgrid_source_is_aligned_onto_the_first(self, offgrid_items):
        """A group whose sources sit on different grids is aligned before fusing.

        Test scenario:
            The second item's grid is shifted by half a cell, so it must be
            aligned onto the first before the fuser ever sees it — the callback
            gets the reference shape and the aligned values reach the output.
        """
        seen = []

        def record(dst, src):
            seen.append(src.shape)
            np.copyto(dst, src, where=~np.isnan(src))

        coll = DatasetCollection.from_stac(
            offgrid_items, asset="data", groupby="orbit", fuse_func=record
        )
        assert seen == [(3, 3)], (
            f"the fuser must see the reference shape (3, 3), got {seen}"
        )
        fused = coll.datasets[0].read_array()
        assert fused.shape == (3, 3), f"the fused grid should be the first's: {fused}"
        assert float(fused[1, 1]) == pytest.approx(9.0), (
            f"the aligned source should have landed on the reference grid: {fused}"
        )

    def test_fuser_result_is_overridable(self, three_local_items):
        """A different fuser produces a different group value.

        Test scenario:
            A max fuser over the orbit-1 group (1 and 2) -> 2.
        """

        def fuse_max(dst, src):
            np.fmax(dst, src, out=dst)

        coll = DatasetCollection.from_stac(
            three_local_items, asset="data", groupby="orbit", fuse_func=fuse_max
        )
        value = float(coll.datasets[0].read_array()[0, 0])
        assert value == pytest.approx(2.0), f"expected the fused max 2.0, got {value}"

    def test_works_with_solar_day(self, three_local_items):
        """A fuser also applies to the solar-day grouping.

        Test scenario:
            3 distinct solar days -> 3 fused timesteps.
        """
        coll = DatasetCollection.from_stac(
            three_local_items,
            asset="data",
            groupby="solar_day",
            fuse_func=self._keep_first,
        )
        assert coll.time_length == 3, (
            f"expected 3 fused solar-day timesteps, got {coll.time_length}"
        )

    def test_without_groupby_raises(self, three_local_items):
        """A fuser with no groupby raises: there is no overlap to fuse.

        Test scenario:
            fuse_func and groupby=None.
        """
        with pytest.raises(ValueError, match="only applies to a grouped mosaic"):
            DatasetCollection.from_stac(
                three_local_items, asset="data", fuse_func=self._keep_first
            )

    def test_with_method_raises(self, three_local_items):
        """fuse_func and a non-default method are mutually exclusive.

        Test scenario:
            fuse_func together with method='mean'.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):
            DatasetCollection.from_stac(
                three_local_items,
                asset="data",
                groupby="orbit",
                method="mean",
                fuse_func=self._keep_first,
            )

    def test_non_callable_raises(self, three_local_items):
        """A fuse_func that is not callable is rejected up front.

        Test scenario:
            fuse_func='first'.
        """
        with pytest.raises(TypeError, match="fuse_func must be a callable"):
            DatasetCollection.from_stac(
                three_local_items, asset="data", groupby="orbit", fuse_func="first"
            )


class TestErrorsAsNodata:
    """STAC-09: a present-but-unreadable asset becomes a no-data plane."""

    def test_default_still_raises(self, three_local_items, tmp_path):
        """Without errors_as_nodata an unreadable asset still fails the read.

        Test scenario:
            The middle item points at a missing file; opening its timestep
            raises.
        """
        broken = _break_asset(three_local_items, 1, tmp_path)
        coll = DatasetCollection.from_stac(broken, asset="data")
        with pytest.raises(FileNotFoundError):
            _ = coll.datasets

    def test_default_still_raises_when_grouped(self, three_local_items, tmp_path):
        """Without errors_as_nodata a grouped build fails on an unreadable source.

        Test scenario:
            The middle item is missing and the mosaic cannot open it.
        """
        broken = _break_asset(three_local_items, 1, tmp_path)
        with pytest.raises(RuntimeError):
            DatasetCollection.from_stac(broken, asset="data", groupby="orbit")

    def test_fills_a_plane_in_single_asset_mode(self, three_local_items, tmp_path):
        """An unreadable timestep is kept as an all-NaN plane, not dropped.

        Test scenario:
            3 items, the middle one missing -> time_length 3 with a NaN plane.
        """
        broken = _break_asset(three_local_items, 1, tmp_path)
        with pytest.warns(RuntimeWarning, match="errors_as_nodata"):
            coll = DatasetCollection.from_stac(
                broken, asset="data", errors_as_nodata=True
            )
        assert coll.time_length == 3, (
            f"the unreadable timestep should be kept, got {coll.time_length}"
        )
        assert np.isnan(coll.datasets[1].read_array()).all(), (
            "the substituted timestep should be entirely no-data"
        )

    def test_readable_timesteps_keep_their_values(self, three_local_items, tmp_path):
        """Substituting one plane leaves the other timesteps untouched.

        Test scenario:
            Items 1 and 3 still read 1.0 and 3.0.
        """
        broken = _break_asset(three_local_items, 1, tmp_path)
        with pytest.warns(RuntimeWarning):
            coll = DatasetCollection.from_stac(
                broken, asset="data", errors_as_nodata=True
            )
        assert float(coll.datasets[0].read_array()[0, 0]) == pytest.approx(1.0), (
            "the first timestep should be unaffected"
        )
        assert float(coll.datasets[2].read_array()[0, 0]) == pytest.approx(3.0), (
            "the last timestep should be unaffected"
        )

    def test_warning_names_the_href(self, three_local_items, tmp_path):
        """The substitution warning names the asset that failed.

        Test scenario:
            The warning text carries the missing file's name.
        """
        broken = _break_asset(three_local_items, 1, tmp_path, name="gone.tif")
        with pytest.warns(RuntimeWarning, match="gone.tif"):
            DatasetCollection.from_stac(broken, asset="data", errors_as_nodata=True)

    def test_without_a_grid_and_no_readable_item_raises(
        self, three_local_items, tmp_path
    ):
        """With nothing readable and no grid= there is no grid to fill.

        Test scenario:
            Every item is unreadable and no grid is given -> ValueError.
        """
        broken = copy.deepcopy(three_local_items)
        for item in broken:
            item["assets"]["data"]["href"] = str(tmp_path / "gone.tif")
        with (
            pytest.warns(RuntimeWarning),
            pytest.raises(ValueError, match="no reference grid"),
        ):
            DatasetCollection.from_stac(broken, asset="data", errors_as_nodata=True)

    def test_grid_supplies_the_reference(
        self, three_local_items, tmp_path, matching_grid
    ):
        """A grid= target lets every timestep be substituted.

        Test scenario:
            All 3 items unreadable but grid= known -> 3 NaN planes.
        """
        broken = copy.deepcopy(three_local_items)
        for item in broken:
            item["assets"]["data"]["href"] = str(tmp_path / "gone.tif")
        with pytest.warns(RuntimeWarning):
            coll = DatasetCollection.from_stac(
                broken, asset="data", errors_as_nodata=True, grid=matching_grid
            )
        assert coll.time_length == 3, (
            f"expected 3 substituted timesteps, got {coll.time_length}"
        )
        assert np.isnan(coll.datasets[0].read_array()).all(), (
            "a grid-backed substitution should be entirely no-data"
        )

    def test_grouped_drops_an_unreadable_source(self, three_local_items, tmp_path):
        """An unreadable source leaves its group, which still mosaics.

        Test scenario:
            The orbit-1 group loses its second member -> the group reads 1.0.
        """
        broken = _break_asset(three_local_items, 1, tmp_path)
        with pytest.warns(RuntimeWarning):
            coll = DatasetCollection.from_stac(
                broken, asset="data", groupby="orbit", errors_as_nodata=True
            )
        assert coll.time_length == 2, (
            f"both orbit groups should survive, got {coll.time_length}"
        )
        assert float(coll.datasets[0].read_array()[0, 0]) == pytest.approx(1.0), (
            "the surviving orbit-1 source should back its group"
        )

    def test_grouped_empty_group_becomes_a_plane(self, three_local_items, tmp_path):
        """A group whose every source is unreadable becomes a no-data plane.

        Test scenario:
            The orbit-2 group's only item is missing; the orbit-1 group supplies
            the reference grid -> 2 timesteps, the second all no-data.
        """
        broken = _break_asset(three_local_items, 2, tmp_path)
        with pytest.warns(RuntimeWarning):
            coll = DatasetCollection.from_stac(
                broken, asset="data", groupby="orbit", errors_as_nodata=True
            )
        assert coll.time_length == 2, (
            f"the emptied group should still be a timestep, got {coll.time_length}"
        )
        assert np.isnan(coll.datasets[1].read_array()).all(), (
            "the emptied group's timestep should be entirely no-data"
        )

    def test_multi_asset_fills_a_plane(self, three_local_items, tmp_path):
        """A multi-asset item that cannot be stacked becomes a no-data plane.

        Test scenario:
            Two assets per item, the middle item's second asset missing -> a
            2-band NaN timestep.
        """
        items = copy.deepcopy(three_local_items)
        for item in items:
            item["assets"]["second"] = dict(item["assets"]["data"])
        items[1]["assets"]["second"]["href"] = str(tmp_path / "gone.tif")
        with pytest.warns(RuntimeWarning):
            coll = DatasetCollection.from_stac(
                items, asset=["data", "second"], errors_as_nodata=True
            )
        assert coll.time_length == 3, (
            f"the unreadable item should be kept, got {coll.time_length}"
        )
        assert coll.datasets[1].band_count == 2, (
            f"the plane should keep the band axis, got {coll.datasets[1].band_count}"
        )
        assert np.isnan(coll.datasets[1].read_array()).all(), (
            "the substituted multi-asset timestep should be entirely no-data"
        )

    def test_multi_asset_default_still_raises(self, three_local_items, tmp_path):
        """Without errors_as_nodata an unstackable multi-asset item still fails.

        Test scenario:
            Two assets per item with the middle item's second asset missing, and
            the no-data substitution switched off, so the band stack raises.
        """
        items = copy.deepcopy(three_local_items)
        for item in items:
            item["assets"]["second"] = dict(item["assets"]["data"])
        items[1]["assets"]["second"]["href"] = str(tmp_path / "gone.tif")
        with pytest.raises((OSError, RuntimeError)):
            DatasetCollection.from_stac(items, asset=["data", "second"])

    def test_does_not_swallow_a_missing_asset_key(self, three_local_items):
        """errors_as_nodata is not skip_missing: a missing key still raises.

        Test scenario:
            An item with no 'data' asset at all.
        """
        items = copy.deepcopy(three_local_items)
        items[1]["assets"] = {}
        with pytest.raises(StacAssetError):
            DatasetCollection.from_stac(items, asset="data", errors_as_nodata=True)
