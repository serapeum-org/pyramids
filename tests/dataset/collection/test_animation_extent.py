"""A collection animates on world coordinates, like the raster it is made of.

`Analysis.plot` assembles the georeferencing half of its `RenderRequest` --
`extent`, and `coords` where a curvilinear grid needs them. Both
`RenderRequest`s in `DatasetCollection.plot` assembled everything *but* that,
so a three-timestep animation of the same rasters came back on pixel indices
(-0.5 to 3.5) where the single-raster call came back on world coordinates
(30.0 to 32.0).

`basemap=` was accepted on that path all the same, with no extent to place it
against.

The extent is the base raster's `bbox`, which every timestep shares -- a
collection is co-registered by construction, and `align` is what enforces it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.collection import DatasetCollection

pytestmark = pytest.mark.plot

TOP_LEFT = (30.0, 12.0)
CELL_SIZE = 0.5
EXPECTED_X = (30.0, 32.0)
EXPECTED_Y = (10.5, 12.0)


def _axes_of(glyph):
    """The matplotlib axes a cleopatra glyph rendered onto.

    Args:
        glyph: Whatever the plot call returned -- a glyph, or a
            `(figure, axes)` pair from the older facade.

    Returns:
        The axes object, so the caller can read its limits.
    """
    return glyph.ax if hasattr(glyph, "ax") else glyph[1]


@pytest.fixture
def rasters() -> list[Dataset]:
    """Three co-registered 3x4 rasters on a known WGS 84 grid.

    Built rather than read so the expected world extent is arithmetic the
    test states outright, not a property of a fixture file.

    Returns:
        list[Dataset]: One raster per timestep, differing only in values.
    """
    return [
        Dataset.from_array(
            np.arange(12, dtype="float32").reshape(3, 4) + step,
            geo_ref=GeoReference(
                top_left_corner=TOP_LEFT, cell_size=CELL_SIZE, epsg=4326
            ),
        )
        for step in range(3)
    ]


@pytest.fixture
def collection(rasters) -> DatasetCollection:
    """A three-timestep collection over those rasters.

    Args:
        rasters: The per-timestep fixture.

    Returns:
        DatasetCollection: The stack under test.
    """
    return DatasetCollection(rasters[0], time_length=len(rasters), datasets=rasters)


class TestTheAnimationIsGeoreferenced:
    """The extent reaches the renderer, so the axes are world coordinates."""

    def test_the_axes_span_the_rasters_own_extent(self, collection, rasters):
        """The regression: pixel indices where world coordinates belong.

        Args:
            collection: The stack under test.
            rasters: Its per-timestep rasters.

        Test scenario:
            A 4-column grid of 0.5-degree cells from 30.0 spans 30.0 to 32.0.
            Without the extent the renderer falls back to array indices and
            reports -0.5 to 3.5, which is off by two orders of magnitude and
            in the wrong units.
        """
        axes = _axes_of(collection.plot())

        assert tuple(round(float(v), 3) for v in axes.get_xlim()) == EXPECTED_X
        assert tuple(round(float(v), 3) for v in axes.get_ylim()) == EXPECTED_Y

    def test_it_matches_the_single_raster_call(self, collection, rasters):
        """The two spellings of the same picture must agree.

        Args:
            collection: The stack under test.
            rasters: Its per-timestep rasters.

        Test scenario:
            Plotting the base raster and animating the collection built from
            it describe the same ground. Asserting against `Dataset.plot`
            rather than against literals is what keeps the two paths tied
            together if the extent convention ever changes.
        """
        single = _axes_of(rasters[0].plot())

        animated = _axes_of(collection.plot())

        assert animated.get_xlim() == single.get_xlim()
        assert animated.get_ylim() == single.get_ylim()

    def test_the_extent_follows_the_base_raster(self, rasters):
        """It is read from the stack, not hard-coded to a default.

        Args:
            rasters: The per-timestep fixture, re-georeferenced here.

        Test scenario:
            A collection on a different grid has to animate on *that* grid. A
            fallback that returned a fixed or unit extent would pass the test
            above and fail this one.
        """
        moved = [
            Dataset.from_array(
                np.asarray(raster.read_array()),
                geo_ref=GeoReference(
                    top_left_corner=(-8.0, 50.0), cell_size=2.0, epsg=4326
                ),
            )
            for raster in rasters
        ]
        collection = DatasetCollection(moved[0], time_length=3, datasets=moved)

        axes = _axes_of(collection.plot())

        assert tuple(round(float(v), 3) for v in axes.get_xlim()) == (-8.0, 0.0)
        assert tuple(round(float(v), 3) for v in axes.get_ylim()) == (44.0, 50.0)
