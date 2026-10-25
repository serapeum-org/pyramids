"""``plot_outline(basemap=...)`` support (issue #1354).

``plot(variable, basemap=True)`` drew a tile backdrop but ``plot_outline`` forwarded every
kwarg straight to the ``LineCollection``, so asking for a basemap under a wireframe raised
``LineCollection.set() got an unexpected keyword argument 'basemap'``. The outline is the first
visual in most mesh notebooks, which is exactly where a backdrop is wanted.

``add_basemap`` is patched throughout: these tests assert the **dispatch** (that the backdrop
is requested, with the right CRS and provider), not that tiles are fetched over the network.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from unittest.mock import patch  # noqa: E402

from pyramids.netcdf.ugrid import UgridDataset  # noqa: E402

pytestmark = pytest.mark.plot

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_FACES = np.array([[0, 1, 2], [0, 2, 3]])


def _mesh(epsg=32636):
    return UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"d": np.array([1.0, 2.0])},
        data_locations={"d": "face"},
        epsg=epsg,
    )


class TestBasemapIsAccepted:
    """The defect itself: `basemap=` must not reach the LineCollection."""

    def test_basemap_true_no_longer_raises(self):
        """Regression (#1354): this raised `unexpected keyword argument 'basemap'`."""
        with patch("pyramids.basemap.basemap.add_basemap") as add:
            glyph = _mesh().plot_outline(basemap=True)
        assert glyph is not None
        assert add.called, "the backdrop must actually be requested"

    def test_the_dataset_crs_is_used_for_the_tile_fetch(self):
        with patch("pyramids.basemap.basemap.add_basemap") as add:
            _mesh(epsg=32636).plot_outline(basemap=True)
        assert add.call_args.kwargs["crs"] == 32636

    def test_a_provider_string_is_forwarded_as_the_source(self):
        with patch("pyramids.basemap.basemap.add_basemap") as add:
            _mesh().plot_outline(basemap="CartoDB.Positron")
        assert add.call_args.kwargs["source"] == "CartoDB.Positron"

    def test_no_basemap_requests_no_backdrop(self):
        with patch("pyramids.basemap.basemap.add_basemap") as add:
            glyph = _mesh().plot_outline()
        assert not add.called
        assert glyph is not None


class TestStylingStillWorks:
    """The kwargs that always went to the line collection must keep going there."""

    def test_color_and_linewidth_are_still_honoured(self):
        glyph = _mesh().plot_outline(color="red", linewidth=2.0)
        assert glyph is not None

    def test_styling_and_basemap_together(self):
        with patch("pyramids.basemap.basemap.add_basemap") as add:
            glyph = _mesh().plot_outline(basemap=True, color="red", linewidth=2.0)
        assert glyph is not None
        assert add.called


class TestCrsGuard:
    """A basemap cannot be placed without a CRS, and that is the data path's rule too."""

    def test_a_mesh_without_a_crs_is_refused(self):
        """Mirrors `plot`: a CRS-less mesh raises rather than mis-placing the tiles."""
        no_crs = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={"d": np.array([1.0, 2.0])},
            data_locations={"d": "face"},
        )
        # `from_arrays` defaults to EPSG 4326, so drop the CRS to reach the guard - which is
        # the state a mesh read from a file with no grid mapping is in.
        no_crs._crs_wkt = None
        no_crs._cached_crs = None
        assert no_crs.epsg is None
        with pytest.raises(ValueError, match="CRS"):
            no_crs.plot_outline(basemap=True)

    def test_the_default_from_arrays_crs_behaves_like_the_data_path(self):
        """`from_arrays` supplies EPSG 4326, so neither entry point raises - they agree."""
        ds = _mesh(epsg=4326)
        with patch("pyramids.basemap.basemap.add_basemap") as add:
            ds.plot_outline(basemap=True)
            ds.plot("d", basemap=True)
        assert add.call_count == 2
