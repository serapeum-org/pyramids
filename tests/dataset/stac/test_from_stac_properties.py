"""Tests for from_stac(properties=...) and from_stac(eo_band_names=...) (STAC-08)."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, DatasetCollection, Grid
from pyramids.dataset._stac import from_stac

pytestmark = pytest.mark.core


@pytest.fixture
def two_asset_item(tmp_path):
    """An item with two same-grid assets, one of which declares `eo:bands`.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        dict: A STAC item whose `red` asset names its band `"reflectance_red"`
        while `mask` declares no `eo:bands` at all.
    """
    paths = {}
    for name, value in (("red", 1.0), ("mask", 2.0)):
        path = str(tmp_path / f"{name}.tif")
        Dataset.from_array(
            np.full((2, 2), value, dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        ).to_file(path)
        paths[name] = path
    return {
        "type": "Feature",
        "id": "two-asset",
        "bbox": [0.0, 0.0, 2.0, 2.0],
        "properties": {"datetime": "2023-06-01T00:00:00Z", "eo:cloud_cover": 5},
        "assets": {
            "red": {
                "href": paths["red"],
                "eo:bands": [{"name": "reflectance_red", "common_name": "red"}],
            },
            "mask": {"href": paths["mask"]},
        },
        "stac_extensions": [],
    }


class TestPropertiesDefaultOff:
    """Tests that `properties` defaults to attaching nothing."""

    def test_default_attaches_nothing(self, three_local_items):
        """The default build leaves `time_attrs` unset.

        Test scenario:
            Three local items read with today's defaults.
        """
        collection = from_stac(three_local_items, asset="data")
        assert collection.time_attrs is None, (
            f"properties=False must attach nothing, got {collection.time_attrs!r}"
        )

    def test_an_empty_selection_attaches_nothing(self, three_local_items):
        """An explicitly empty key list is a no-op, not a list of empty dicts.

        Test scenario:
            `properties=[]` over three local items.
        """
        collection = from_stac(three_local_items, asset="data", properties=[])
        assert collection.time_attrs is None, (
            f"an empty selection must attach nothing, got {collection.time_attrs!r}"
        )

    def test_a_plain_collection_has_no_attrs(self, three_local_items):
        """A collection built outside `from_stac` also starts with `None`.

        Test scenario:
            `from_files` over the same rasters.
        """
        paths = [item["assets"]["data"]["href"] for item in three_local_items]
        collection = DatasetCollection.from_files(paths)
        assert collection.time_attrs is None, (
            f"from_files must not attach attributes, got {collection.time_attrs!r}"
        )

    def test_time_attrs_is_read_only(self, three_local_items):
        """`time_attrs` exposes the store without letting a caller replace it.

        Test scenario:
            Assigning to the property on a built collection.
        """
        collection = from_stac(three_local_items, asset="data", properties=True)
        with pytest.raises(AttributeError, match="time_attrs"):
            collection.time_attrs = [{}, {}, {}]


class TestTheAttributeStore:
    """Tests for the collection-side store the attributes live in."""

    def test_a_length_mismatch_is_refused(self, three_local_items):
        """The store enforces one dict per timestep.

        Test scenario:
            Attaching two dicts to a three-timestep collection.
        """
        collection = from_stac(three_local_items, asset="data")
        with pytest.raises(ValueError, match="time_attrs has length 2"):
            collection._attach_time_attrs([{"a": 1}, {"a": 2}])

    def test_the_store_can_be_cleared(self, three_local_items):
        """Attaching `None` drops the attributes again.

        Test scenario:
            A cube with attributes, then cleared.
        """
        collection = from_stac(three_local_items, asset="data", properties=True)
        collection._attach_time_attrs(None)
        assert collection.time_attrs is None, (
            f"the store must clear to None, got {collection.time_attrs!r}"
        )

    def test_the_store_copies_the_dicts(self, three_local_items):
        """The store holds its own dicts, not the caller's.

        Test scenario:
            Mutating a dict after attaching it.
        """
        collection = from_stac(three_local_items, asset="data")
        source = [{"a": 1}, {"a": 2}, {"a": 3}]
        collection._attach_time_attrs(source)
        source[0]["a"] = 99
        assert collection.time_attrs[0]["a"] == 1, (
            f"the store must not alias the caller's dicts, got "
            f"{collection.time_attrs!r}"
        )


class TestPropertiesAttached:
    """Tests for the per-timestep attributes attached to an ungrouped cube."""

    def test_selected_keys_are_attached(self, three_local_items):
        """A key list attaches exactly those keys, one dict per timestep.

        Test scenario:
            Three items with a varying `eo:cloud_cover`.
        """
        collection = from_stac(
            three_local_items, asset="data", properties=["eo:cloud_cover"]
        )
        attrs = collection.time_attrs
        assert len(attrs) == collection.time_length, (
            f"expected {collection.time_length} attribute dicts, got {len(attrs)}"
        )
        assert [a["eo:cloud_cover"] for a in attrs] == [0, 10, 20], (
            f"the cloud cover must follow item order, got {attrs!r}"
        )
        assert all(set(a) == {"eo:cloud_cover"} for a in attrs), (
            f"only the requested key may be attached, got {attrs!r}"
        )

    def test_true_attaches_every_property(self, three_local_items):
        """`properties=True` carries the whole `properties` mapping.

        Test scenario:
            Items declaring `datetime`, `orbit` and `eo:cloud_cover`.
        """
        collection = from_stac(three_local_items, asset="data", properties=True)
        attrs = collection.time_attrs
        assert set(attrs[0]) == {"datetime", "orbit", "eo:cloud_cover"}, (
            f"every property must be attached, got {sorted(attrs[0])}"
        )
        assert attrs[2]["datetime"] == "2023-06-03T00:00:00Z", (
            f"the third timestep's datetime is wrong: {attrs[2]!r}"
        )

    def test_a_single_string_selects_one_key(self, three_local_items):
        """A bare string is treated as a one-key selection.

        Test scenario:
            `properties="orbit"` over three items.
        """
        collection = from_stac(three_local_items, asset="data", properties="orbit")
        assert [a["orbit"] for a in collection.time_attrs] == [1, 1, 2], (
            f"the orbit must follow item order, got {collection.time_attrs!r}"
        )

    def test_an_absent_key_is_filled_with_none(self, three_local_items):
        """Every dict carries every requested key, `None` where undeclared.

        Test scenario:
            A key no item declares, requested alongside one they all do.
        """
        collection = from_stac(
            three_local_items, asset="data", properties=["orbit", "view:azimuth"]
        )
        attrs = collection.time_attrs
        assert all(a["view:azimuth"] is None for a in attrs), (
            f"an undeclared key must read back as None, got {attrs!r}"
        )
        assert all(set(a) == {"orbit", "view:azimuth"} for a in attrs), (
            f"the attribute table must stay rectangular, got {attrs!r}"
        )

    def test_the_attrs_are_plain_picklable_dicts(self, three_local_items):
        """The attached store survives a pickle round trip (Path B safety).

        Test scenario:
            A collection with attributes is pickled and restored.
        """
        collection = from_stac(three_local_items, asset="data", properties=True)
        restored = pickle.loads(pickle.dumps(collection))
        assert restored.time_attrs == collection.time_attrs, (
            f"the attributes must survive pickling, got {restored.time_attrs!r}"
        )

    def test_the_attrs_support_filtering_timesteps(self, three_local_items):
        """The attributes are enough to select timesteps by cloud cover.

        Test scenario:
            Keep the timesteps whose `eo:cloud_cover` is under 15.
        """
        collection = from_stac(
            three_local_items, asset="data", properties=["eo:cloud_cover"]
        )
        clear = [
            index
            for index, attrs in enumerate(collection.time_attrs)
            if attrs["eo:cloud_cover"] < 15
        ]
        assert clear == [0, 1], f"expected the first two timesteps, got {clear}"

    def test_attrs_survive_a_grid_aligned_build(self, three_local_items):
        """A `grid=` build, which re-wraps the cube, still carries the attrs.

        Test scenario:
            The same items aligned onto an explicit target grid.
        """
        template = Dataset.from_array(
            np.zeros((3, 3), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
        )
        collection = from_stac(
            three_local_items,
            asset="data",
            properties=["orbit"],
            grid=Grid(like=template),
        )
        assert len(collection.time_attrs) == collection.time_length, (
            f"a grid-aligned cube must keep one dict per timestep, got "
            f"{collection.time_attrs!r}"
        )

    def test_a_derived_collection_starts_without_attrs(self, three_local_items):
        """A per-timestep op produces a collection with no attributes.

        Test scenario:
            `to_crs` on a cube that carries attributes.
        """
        collection = from_stac(three_local_items, asset="data", properties=True)
        derived = collection.to_crs(3857)
        assert derived.time_attrs is None, (
            f"a derived collection must not inherit attributes, got "
            f"{derived.time_attrs!r}"
        )

    @pytest.mark.parametrize("bad", [3.5, object(), {"orbit": 1}])
    def test_an_unusable_spec_raises(self, three_local_items, bad):
        """A `properties` value that is not a bool / str / string sequence raises.

        Test scenario:
            A float, an arbitrary object, and a mapping.
        """
        with pytest.raises(TypeError, match="properties must"):
            from_stac(three_local_items, asset="data", properties=bad)

    def test_a_non_string_key_raises(self, three_local_items):
        """A sequence holding a non-string key is rejected.

        Test scenario:
            `properties=["orbit", 7]`.
        """
        with pytest.raises(TypeError, match="property-key strings"):
            from_stac(three_local_items, asset="data", properties=["orbit", 7])


class TestGroupedProperties:
    """Tests for the documented per-group attribute derivation."""

    def test_a_group_is_represented_by_its_first_item(self, three_local_items):
        """Each group's attributes come from its first item in item order.

        Test scenario:
            `groupby="orbit"` collapses three items (orbits 1, 1, 2) to two
            timesteps, so group 1's cloud cover must be the first item's.
        """
        collection = from_stac(
            three_local_items,
            asset="data",
            groupby="orbit",
            properties=["orbit", "eo:cloud_cover"],
        )
        attrs = collection.time_attrs
        assert len(attrs) == collection.time_length == 2, (
            f"expected 2 grouped timesteps and 2 dicts, got {collection.time_length} "
            f"and {attrs!r}"
        )
        assert [a["orbit"] for a in attrs] == [1, 2], (
            f"the groups must be emitted in sorted-key order, got {attrs!r}"
        )
        assert attrs[0]["eo:cloud_cover"] == 0, (
            "group 1 must be represented by its first item (cloud cover 0), got "
            f"{attrs[0]!r}"
        )

    def test_solar_day_grouping_also_attaches_one_dict_per_group(
        self, three_local_items
    ):
        """The reserved groupings honour the same length invariant.

        Test scenario:
            `groupby="solar_day"` over three items on three distinct dates.
        """
        collection = from_stac(
            three_local_items, asset="data", groupby="solar_day", properties=True
        )
        assert len(collection.time_attrs) == collection.time_length, (
            f"expected {collection.time_length} dicts, got {len(collection.time_attrs)}"
        )


class TestReducedGroupRepresentative:
    """M2: a non-`first` reduction makes the representative item misleading.

    `properties=` attaches the group's first item in item order. Those are the
    pixels `method="first"` emits, so the pairing is exact there — but every
    other reduction derives the timestep from the whole group, and the attached
    per-granule properties then describe a granule that did not produce most of
    the raster. The build stays well defined, so this warns.
    """

    def test_mean_with_properties_warns(self, three_local_items):
        """Combining a reducing method with properties= warns.

        Test scenario:
            `groupby="orbit"`, `method="mean"` and `properties=True`.
        """
        with pytest.warns(RuntimeWarning, match="method='mean' derives the timestep"):
            from_stac(
                three_local_items,
                asset="data",
                groupby="orbit",
                method="mean",
                properties=True,
            )

    def test_the_warning_is_accurate_about_the_mismatch(self, three_local_items):
        """The attached value really is the representative's, not the reduction's.

        Test scenario:
            Orbit 1 groups items with cloud cover 0 and 10 and pixel values 1.0
            and 2.0; `method="mean"` emits 1.5 while the attributes stay the
            first item's cloud cover 0 — which is what the warning is about.
        """
        with pytest.warns(RuntimeWarning, match="did not produce most of the emitted"):
            collection = from_stac(
                three_local_items,
                asset="data",
                groupby="orbit",
                method="mean",
                properties=["eo:cloud_cover"],
            )
        pixels = float(np.asarray(collection.iloc(0).read_array())[0, 0])
        assert pixels == pytest.approx(1.5), (
            f"method='mean' must average the group's 1.0 and 2.0, got {pixels}"
        )
        assert collection.time_attrs[0]["eo:cloud_cover"] == 0, (
            "the attached property is still the first item's, which is the mismatch "
            f"the warning names, got {collection.time_attrs[0]!r}"
        )

    def test_fuse_func_with_properties_warns(self, three_local_items):
        """A fuser is a whole-group reduction too, so it warns as well.

        Test scenario:
            `groupby="orbit"` with an in-place max fuser and `properties=True`.
        """

        def fuse(dst, src):
            np.copyto(dst, np.fmax(dst, src))

        with pytest.warns(RuntimeWarning, match="fuse_func derives the timestep"):
            from_stac(
                three_local_items,
                asset="data",
                groupby="orbit",
                fuse_func=fuse,
                properties=True,
            )

    def test_method_first_with_properties_is_quiet(self, three_local_items):
        """The default pairing is exact, so nothing warns.

        Test scenario:
            `groupby="orbit"` with the default method and `properties=True`.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from_stac(three_local_items, asset="data", groupby="orbit", properties=True)
        offenders = [
            str(w.message) for w in caught if "derives the timestep" in str(w.message)
        ]
        assert not offenders, (
            f"method='first' needs no representative warning, got {offenders}"
        )

    def test_a_reduction_without_properties_is_quiet(self, three_local_items):
        """Nothing is attached, so there is no mismatch to warn about.

        Test scenario:
            `method="mean"` with `properties` left at its default `False`.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from_stac(three_local_items, asset="data", groupby="orbit", method="mean")
        offenders = [
            str(w.message) for w in caught if "derives the timestep" in str(w.message)
        ]
        assert not offenders, (
            f"no properties means no representative warning, got {offenders}"
        )


class TestMultiAssetProperties:
    """Tests for the attributes attached by the multi-asset path."""

    def test_one_dict_per_kept_item(self, two_asset_item):
        """A multi-asset cube attaches one dict per emitted timestep.

        Test scenario:
            A single item with two assets stacked band-wise.
        """
        collection = from_stac(
            [two_asset_item], asset=["red", "mask"], properties=["eo:cloud_cover"]
        )
        assert collection.time_attrs == [{"eo:cloud_cover": 5}], (
            f"expected the item's cloud cover, got {collection.time_attrs!r}"
        )

    def test_a_skipped_item_is_absent_from_the_attrs(self, two_asset_item):
        """An item dropped by `skip_missing` contributes no attribute dict.

        Test scenario:
            Two items, the second lacking the `mask` asset, with
            `skip_missing=True`.
        """
        thin = {
            **two_asset_item,
            "id": "thin",
            "properties": {"datetime": "2023-06-02T00:00:00Z", "eo:cloud_cover": 99},
            "assets": {"red": two_asset_item["assets"]["red"]},
        }
        collection = from_stac(
            [two_asset_item, thin],
            asset=["red", "mask"],
            skip_missing=True,
            properties=["eo:cloud_cover"],
        )
        assert collection.time_attrs == [{"eo:cloud_cover": 5}], (
            f"the skipped item must not appear, got {collection.time_attrs!r}"
        )
        assert len(collection.time_attrs) == collection.time_length, (
            "the attribute list must match the timestep count, got "
            f"{len(collection.time_attrs)} vs {collection.time_length}"
        )


class TestEoBandNames:
    """Tests for naming multi-asset bands from `eo:bands`."""

    def test_default_keeps_the_asset_keys(self, two_asset_item):
        """Without the opt-in the band names stay the raw asset keys.

        Test scenario:
            A two-asset item whose first asset declares `eo:bands`.
        """
        collection = from_stac([two_asset_item], asset=["red", "mask"])
        names = collection.iloc(0).band_names
        assert names == ["red", "mask"], (
            f"the default must keep the asset keys, got {names}"
        )

    def test_the_opt_in_adopts_the_eo_band_name(self, two_asset_item):
        """`eo_band_names=True` renames a band that declares an `eo:bands` name.

        Test scenario:
            The `red` asset names its band `reflectance_red`; `mask` names none.
        """
        collection = from_stac(
            [two_asset_item], asset=["red", "mask"], eo_band_names=True
        )
        names = collection.iloc(0).band_names
        assert names == ["reflectance_red", "mask"], (
            f"expected the eo:bands name plus the asset-key fallback, got {names}"
        )

    def test_a_common_name_is_used_when_no_name_is_given(self, two_asset_item):
        """`common_name` stands in for a missing `eo:bands` `name`.

        Test scenario:
            The `red` asset declares only a `common_name`.
        """
        two_asset_item["assets"]["red"]["eo:bands"] = [{"common_name": "red_common"}]
        collection = from_stac(
            [two_asset_item], asset=["red", "mask"], eo_band_names=True
        )
        names = collection.iloc(0).band_names
        assert names == ["red_common", "mask"], (
            f"expected the common_name to be adopted, got {names}"
        )

    def test_a_multi_band_eo_declaration_keeps_the_asset_key(self, two_asset_item):
        """An `eo:bands` naming several bands cannot rename a one-band asset.

        Test scenario:
            The `red` asset declares two `eo:bands` entries.
        """
        two_asset_item["assets"]["red"]["eo:bands"] = [{"name": "a"}, {"name": "b"}]
        collection = from_stac(
            [two_asset_item], asset=["red", "mask"], eo_band_names=True
        )
        names = collection.iloc(0).band_names
        assert names == ["red", "mask"], (
            f"an ambiguous eo:bands must fall back to the asset key, got {names}"
        )

    def test_a_colliding_name_falls_back_to_the_asset_key(self, two_asset_item):
        """Two assets cannot be renamed to the same band name.

        Test scenario:
            Both assets declare the `eo:bands` name `"same"`.
        """
        two_asset_item["assets"]["red"]["eo:bands"] = [{"name": "same"}]
        two_asset_item["assets"]["mask"]["eo:bands"] = [{"name": "same"}]
        collection = from_stac(
            [two_asset_item], asset=["red", "mask"], eo_band_names=True
        )
        names = collection.iloc(0).band_names
        assert names == ["same", "mask"], (
            f"a colliding name must fall back to the asset key, got {names}"
        )

    def test_rejected_in_single_asset_mode(self, three_local_items):
        """The option is refused where there is no band axis to name.

        Test scenario:
            A single-asset build with `eo_band_names=True`.
        """
        with pytest.raises(ValueError, match="eo_band_names only applies"):
            from_stac(three_local_items, asset="data", eo_band_names=True)

    def test_rejected_in_grouped_mode(self, three_local_items):
        """A grouped build refuses the option too.

        Test scenario:
            `groupby="orbit"` with `eo_band_names=True`.
        """
        with pytest.raises(ValueError, match="eo_band_names only applies"):
            from_stac(
                three_local_items, asset="data", groupby="orbit", eo_band_names=True
            )
