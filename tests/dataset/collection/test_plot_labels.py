"""Animation frame-label routing for ``DatasetCollection.plot`` (issue #693).

``plot`` must default the animation frame labels to a plain index axis
(``range(time_length)``) and let a caller override them via an explicit
``animation_axis_values`` kwarg — without that kwarg colliding with the value the
method passes to :func:`render_array` internally (before the fix, passing it
raised ``TypeError: got multiple values for keyword argument``).

``render_array`` is patched so these assertions exercise only the pyramids-side
kwarg routing, with no cleopatra dependency and no real rendering.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection

pytestmark = pytest.mark.core


def _collection(count: int = 3, bands: int = 1) -> DatasetCollection:
    """Build a small in-memory collection of ``count`` identical timesteps."""
    arr = np.ones((bands, 4, 5), dtype="float32")
    src = Dataset.create_from_array(
        arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
    )
    return DatasetCollection.create_cube(src, count)


class TestPlotAnimationAxisValues:
    def test_defaults_to_index_axis(self):
        """With no override, frames are labelled by their index ``0..N-1``."""
        cube = _collection(3)
        with patch("pyramids.dataset.collection.render_array") as render:
            cube.plot(band=0)
        assert render.call_args.kwargs["animation_axis_values"] == [0, 1, 2]

    def test_explicit_axis_values_respected(self):
        """An explicit ``animation_axis_values`` wins over the index default.

        This also guards the kwarg collision: before the fix the same keyword was
        passed both by the caller (via ``**kwargs``) and hardcoded by ``plot``,
        which raised ``TypeError``.
        """
        cube = _collection(3)
        years = [2000, 2001, 2002]
        with patch("pyramids.dataset.collection.render_array") as render:
            cube.plot(band=0, animation_axis_values=years)
        assert render.call_args.kwargs["animation_axis_values"] == years

    def test_explicit_axis_values_respected_rgb(self):
        """The override also reaches the RGB (true-colour) animation branch."""
        cube = _collection(2, bands=3)
        labels = ["2000", "2001"]
        with patch("pyramids.dataset.collection.render_array") as render:
            cube.plot(
                rgb_options={"rgb": [0, 1, 2], "surface_reflectance": 255},
                animation_axis_values=labels,
            )
        assert render.call_args.kwargs["animation_axis_values"] == labels
