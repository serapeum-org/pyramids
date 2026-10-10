"""The STAC features must be reachable from the public API, not only the private one.

Several tasks added parameters to the private `pyramids.dataset._stac.from_stac`
and to `Dataset.from_band_files`. A feature whose only caller is a private
function is not shipped, so these tests drive the public entry points.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.merge import stack_bands

pytestmark = pytest.mark.core


class TestFromStacPublicSurface:
    """`DatasetCollection.from_stac` forwards the parameters its tasks added."""

    def test_rescale_reaches_the_public_entry_point(self, packed_item):
        """`rescale=True` yields physical units through the public classmethod.

        Args:
            packed_item: Item declaring scale 0.01 over int16 counts.

        Test scenario:
            The forwarder originally omitted `rescale`, so the feature was only
            reachable via the private function — shipped but unusable.
        """
        cube = DatasetCollection.from_stac([packed_item], asset="data", rescale=True)
        values = cube.iloc(0).read_array()
        finite = values[np.isfinite(values)]
        assert finite.max() == pytest.approx(3.0), (
            f"expected physical max 3.0 from scale 0.01, got {finite.max()}"
        )

    def test_raw_counts_without_rescale(self, packed_item):
        """Omitting `rescale` still returns the stored counts.

        Args:
            packed_item: Item declaring scale 0.01 over int16 counts.

        Test scenario:
            The new parameter must not change the default, or every existing
            caller's values would shift by the scale factor.
        """
        cube = DatasetCollection.from_stac([packed_item], asset="data")
        values = cube.iloc(0).read_array()
        finite = values[np.isfinite(values)]
        assert finite.max() == pytest.approx(300.0), (
            f"expected stored max 300, got {finite.max()}"
        )

    def test_cfg_alias_reaches_the_public_entry_point(self, packed_item):
        """A `cfg` band alias resolves an asset by its alias name.

        Args:
            packed_item: Item whose only asset key is `data`.

        Test scenario:
            `cfg` was the other parameter the forwarder dropped; an alias is the
            cheapest proof it is threaded, since the asset key itself changes.
        """
        cfg = {"*": {"aliases": {"reflectance": "data"}}}
        cube = DatasetCollection.from_stac([packed_item], asset="reflectance", cfg=cfg)
        assert cube.time_length == 1, f"expected 1 timestep, got {cube.time_length}"

    def test_properties_attach_through_the_public_entry_point(self, three_local_items):
        """`properties=` attaches the selected item properties to the cube.

        Args:
            three_local_items: Three items carrying `eo:cloud_cover`.

        Test scenario:
            The attached table is what makes metadata-aware selection possible,
            so it has to survive the public call.
        """
        cube = DatasetCollection.from_stac(
            three_local_items, asset="data", properties=["eo:cloud_cover"]
        )
        covers = [attrs["eo:cloud_cover"] for attrs in cube.time_attrs]
        assert covers == [0, 10, 20], f"expected [0,10,20], got {covers}"


class TestTimeAttrsIsReallyReadOnly:
    """`DatasetCollection.time_attrs` hands back a snapshot, not internal state."""

    def test_mutating_an_entry_does_not_reach_the_collection(self, three_local_items):
        """Writing into a returned dict leaves the collection's own value alone.

        Args:
            three_local_items: Three items carrying `eo:cloud_cover`.

        Test scenario:
            The property is documented read-only, so a caller filtering a cube by
            provenance may edit the dicts it got back without silently rewriting
            the cube's metadata.
        """
        cube = DatasetCollection.from_stac(
            three_local_items, asset="data", properties=["eo:cloud_cover"]
        )
        cube.time_attrs[0]["eo:cloud_cover"] = 99
        assert cube.time_attrs[0]["eo:cloud_cover"] == 0, (
            "writing into a returned entry rewrote the collection's own value, got "
            f"{cube.time_attrs[0]['eo:cloud_cover']} instead of the attached 0"
        )

    def test_appending_to_the_list_cannot_desynchronise_time_length(
        self, three_local_items
    ):
        """Appending to the returned list does not change what the collection holds.

        Args:
            three_local_items: Three items carrying `eo:cloud_cover`.

        Test scenario:
            `len(time_attrs) == time_length` is the documented invariant, and an
            append on the live list broke it -- leaving a cube whose attribute
            table claims one more timestep than it has.
        """
        cube = DatasetCollection.from_stac(
            three_local_items, asset="data", properties=["eo:cloud_cover"]
        )
        cube.time_attrs.append({"eo:cloud_cover": 99})
        assert len(cube.time_attrs) == cube.time_length, (
            f"time_attrs grew to {len(cube.time_attrs)} against "
            f"{cube.time_length} timesteps, so the invariant is not enforced"
        )

    def test_a_nested_value_is_copied_too(self, three_local_items):
        """A nested list inside an entry is a copy, not the collection's own object.

        Args:
            three_local_items: Three items whose properties are extended with a
                nested value here.

        Test scenario:
            STAC properties routinely nest (`proj:transform`, `eo:bands`), so a
            per-entry shallow copy would still hand out the live inner objects --
            the same defect one level down.
        """
        for item in three_local_items:
            item["properties"]["proj:transform"] = [1.0, 0.0, 0.0]
        cube = DatasetCollection.from_stac(
            three_local_items, asset="data", properties=["proj:transform"]
        )
        cube.time_attrs[0]["proj:transform"][0] = 99.0
        assert cube.time_attrs[0]["proj:transform"][0] == pytest.approx(1.0), (
            "the nested list was shared with the collection, got "
            f"{cube.time_attrs[0]['proj:transform']}"
        )

    def test_editing_the_source_item_afterwards_does_not_reach_the_cube(
        self, three_local_items
    ):
        """The attached entries are copied in, not aliased to the Items' properties.

        Args:
            three_local_items: Three items whose properties are extended with a
                nested value here.

        Test scenario:
            The same defect on the way in: the dicts attached come straight off
            the Items' nested `properties`, so a shallow copy leaves the cube
            sharing the Items' inner lists, and editing an Item afterwards
            rewrites the cube's metadata.
        """
        for item in three_local_items:
            item["properties"]["proj:transform"] = [1.0, 0.0, 0.0]
        cube = DatasetCollection.from_stac(
            three_local_items, asset="data", properties=["proj:transform"]
        )
        three_local_items[0]["properties"]["proj:transform"][0] = 99.0
        assert cube.time_attrs[0]["proj:transform"][0] == pytest.approx(1.0), (
            "editing the source Item changed the attached entry, so the store "
            f"aliases the Item's own list: {cube.time_attrs[0]['proj:transform']}"
        )


class TestStackBandsResampling:
    """`stack_bands` forwards the resampling control it aliases."""

    @staticmethod
    def _mismatched_pair(tmp_path):
        """Write a fine 8x8 template and a coarse 4x4 gradient over the same extent.

        The second raster's cells are twice as wide, so `align=True` has to
        resample it onto the first's grid -- and its values form a gradient, so
        interpolating that resample gives different pixels from replicating it.

        Args:
            tmp_path: pytest temp directory.

        Returns:
            tuple[str, str]: (fine path, coarse path). The derived band names are
            ``"fine"`` and ``"coarse"``.
        """
        fine = tmp_path / "fine.tif"
        coarse = tmp_path / "coarse.tif"
        Dataset.from_array(
            np.zeros((8, 8), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=0.5, epsg=4326),
        ).to_file(str(fine))
        gradient = np.arange(16, dtype="float32").reshape(4, 4)
        Dataset.from_array(
            gradient,
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        ).to_file(str(coarse))
        return str(fine), str(coarse)

    def test_resampling_changes_the_aligned_band(self, tmp_path):
        """A named method reaches `from_band_files` and changes the stacked pixels.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            stack_bands enumerates its parameters explicitly, so an unforwarded
            `resampling` is silently ignored -- the stack still builds, on
            nearest neighbour. Stacking the same mismatched pair twice, once
            `nearest` and once `bilinear`, therefore pins the forwarding to an
            observable difference: identical arrays mean the kwarg was dropped.
            `bilinear` must also invent values the coarse source never held,
            which is what interpolating (rather than replicating) means.
        """
        fine, coarse = self._mismatched_pair(tmp_path)
        nearest = stack_bands(
            [fine, coarse], align=True, resampling="nearest"
        ).read_array(band=1)
        bilinear = stack_bands(
            [fine, coarse], align=True, resampling="bilinear"
        ).read_array(band=1)
        assert not np.allclose(nearest, bilinear), (
            "resampling='bilinear' produced the same pixels as 'nearest', so the "
            "kwarg never reached the alignment that honours it"
        )
        source_values = set(np.arange(16, dtype="float32").tolist())
        assert set(nearest.ravel().tolist()) <= source_values, (
            "nearest neighbour may only replicate the coarse source's own values, "
            f"got {sorted(set(nearest.ravel().tolist()) - source_values)} besides"
        )
        assert not set(bilinear.ravel().tolist()) <= source_values, (
            "bilinear must interpolate between the coarse source's values; every "
            "output value was one of the source's, which is nearest neighbour"
        )

    def test_the_alias_resamples_exactly_as_the_method_does(self, tmp_path):
        """The forwarded result is the method's result, not an approximation of it.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            `stack_bands` is documented as a free-function alias of
            `Dataset.from_band_files`, so the two must agree pixel for pixel on
            the same `resampling` -- the difference test above would still pass
            if the alias forwarded some *other* method.
        """
        fine, coarse = self._mismatched_pair(tmp_path)
        through_alias = stack_bands(
            [fine, coarse], align=True, resampling="bilinear"
        ).read_array(band=1)
        through_method = Dataset.from_band_files(
            [fine, coarse], align=True, resampling="bilinear"
        ).read_array(band=1)
        assert np.array_equal(through_alias, through_method), (
            "the alias and the method disagree on resampling='bilinear', so one of "
            "them is not using the method the caller named"
        )

    def test_a_per_band_mapping_reaches_the_method(self, tmp_path):
        """A ``{band name: method}`` mapping threads through, not just a plain string.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            The mapping form is the one that lets a categorical band stay nearest
            while a continuous one interpolates, and it takes a different branch
            of `_resolve_band_resampling` from the scalar form. Comparing against
            the method's own mapping result (never against another stack_bands
            call, which would degrade the same way) keeps it falsifiable.
        """
        fine, coarse = self._mismatched_pair(tmp_path)
        mapped = stack_bands(
            [fine, coarse], align=True, resampling={"coarse": "bilinear"}
        ).read_array(band=1)
        through_method = Dataset.from_band_files(
            [fine, coarse], align=True, resampling={"coarse": "bilinear"}
        ).read_array(band=1)
        default = stack_bands([fine, coarse], align=True).read_array(band=1)
        assert np.array_equal(mapped, through_method), (
            "the alias and the method disagree on the per-band mapping, so the "
            "mapping is not reaching from_band_files"
        )
        assert not np.allclose(mapped, default), (
            "naming the coarse band bilinear left the pixels at the nearest-"
            "neighbour default, so the mapping was dropped"
        )

    def test_resampling_without_align_is_refused(self, tmp_path):
        """Passing `resampling` with `align=False` raises rather than being ignored.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            `align=False` has nothing to resample, so the combination is refused
            up front rather than silently dropped. The match= is the point: a
            bare `pytest.raises((ValueError, TypeError))` also passes when the
            parameter does not exist at all, since an unexpected keyword is a
            TypeError.
        """
        paths = []
        for index in range(2):
            path = tmp_path / f"band{index}.tif"
            Dataset.from_array(
                np.full((4, 4), float(index), dtype="float32"),
                geo_ref=GeoReference(
                    top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326
                ),
            ).to_file(str(path))
            paths.append(str(path))

        with pytest.raises(ValueError, match="resampling only applies"):
            stack_bands(paths, align=False, resampling="bilinear")
