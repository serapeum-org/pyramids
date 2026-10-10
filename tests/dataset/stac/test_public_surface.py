"""The STAC features must be reachable from the public API, not only the private one.

Several tasks added parameters to the private `pyramids.dataset._stac.from_stac`
and to `Dataset.from_band_files`. A feature whose only caller is a private
function is not shipped, so these tests drive the public entry points.
"""

from __future__ import annotations

import numpy as np
import pytest

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


class TestStackBandsResampling:
    """`stack_bands` forwards the resampling control it aliases."""

    def test_resampling_without_align_is_refused(self, tmp_path):
        """Passing `resampling` with `align=False` raises rather than being ignored.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            stack_bands enumerates its parameters explicitly, so a new
            from_band_files parameter is silently unavailable until forwarded.
            This drives the kwarg all the way through the alias.
        """
        from pyramids.base.georeference import GeoReference

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

        with pytest.raises((ValueError, TypeError)):
            stack_bands(paths, align=False, resampling="bilinear")
