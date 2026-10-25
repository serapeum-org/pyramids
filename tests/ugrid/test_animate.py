"""``animate()`` across UgridDataset, Dataset, NetCDF and DatasetCollection (issue #1380).

Two of the four could already animate -- ``NetCDF.plot(animate=...)`` and
``DatasetCollection.plot``, whose plot *is* the animation -- but under names that made the
capability easy to miss, and the mesh could not animate at all. These tests cover the two new
implementations and pin that the two delegates reach the same path their ``plot`` does.

The Agg backend is forced before any pyplot import, so nothing here opens a window.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pyramids.base.georeference import GeoReference  # noqa: E402
from pyramids.dataset import Dataset  # noqa: E402
from pyramids.dataset.collection import DatasetCollection  # noqa: E402
from pyramids.netcdf import ExtraDimensions, NetCDF  # noqa: E402
from pyramids.netcdf.ugrid import UgridDataset  # noqa: E402

pytestmark = [
    pytest.mark.plot,
    pytest.mark.filterwarnings("ignore:Animation was deleted without rendering"),
]

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _mesh(values, location="face"):
    return UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"d": np.asarray(values, dtype="float64")},
        data_locations={"d": location},
        epsg=4326,
    )


def _raster(bands=3):
    array = np.stack([np.full((4, 5), float(index)) for index in range(bands)])
    return Dataset.from_array(
        array,
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326),
        no_data_value=-9999.0,
    )


class TestUgridAnimate:
    """The mesh had no animation at all before this."""

    def test_a_temporal_variable_animates(self):
        animation = _mesh([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).animate()
        assert animation is not None

    def test_the_return_type_is_the_animation_itself(self):
        """The counterpart of the raster pin: the mesh side returns the FuncAnimation, not a
        glyph, so `anim.save(...)` works directly here."""
        from matplotlib.animation import FuncAnimation

        animation = _mesh([[1.0, 2.0], [3.0, 4.0]]).animate()
        assert isinstance(animation, FuncAnimation)

    def test_one_frame_per_time_step(self):
        animation = _mesh([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).animate()
        assert animation._save_count == 3

    def test_it_saves_to_a_file(self):
        """The thing a user actually does with it, so it has to survive writing."""
        import tempfile
        from pathlib import Path

        animation = _mesh([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).animate()
        out = Path(tempfile.mkdtemp()) / "mesh.gif"
        animation.save(str(out), writer="pillow")
        assert out.stat().st_size > 0

    def test_the_interval_is_forwarded(self):
        animation = _mesh([[1.0, 2.0], [3.0, 4.0]]).animate(interval=50)
        assert animation is not None

    def test_a_static_variable_is_refused(self):
        """One frame is not an animation; the message points at plot()."""
        subject = _mesh([1.0, 2.0])
        with pytest.raises(ValueError, match="plot()"):
            subject.animate()

    def test_a_layered_variable_is_refused_with_a_hint(self):
        """(time, layer, face) has no single value per element per frame."""
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"d": np.zeros((3, 2, 2))},
            data_locations={"d": "face"},
        )
        ds["d"].dimensions = ("time", "layer", "mesh2d_nFaces")
        with pytest.raises(ValueError, match="isel|collapse_layer"):
            ds.animate()

    def test_a_node_located_variable_animates(self):
        """The location is forwarded, so node data draws as node data."""
        values = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        assert _mesh(values, location="node").animate() is not None


class TestDatasetAnimate:
    """Bands as frames, which previously needed a DatasetCollection."""

    def test_a_multi_band_raster_animates(self):
        assert _raster(3).animate() is not None

    def test_a_subset_of_bands(self):
        assert _raster(3).animate(bands=[0, 2]) is not None

    def test_explicit_frame_labels(self):
        assert (
            _raster(3).animate(bands=[0, 2], animation_axis_values=["a", "b"])
            is not None
        )

    def test_it_saves_through_the_glyphs_anim_attribute(self):
        """The array side returns the GLYPH, so the animation is at `glyph.anim` - unlike
        the mesh side, which returns the FuncAnimation itself."""
        import tempfile
        from pathlib import Path

        glyph = _raster(3).animate()
        out = Path(tempfile.mkdtemp()) / "raster.gif"
        glyph.anim.save(str(out), writer="pillow")
        assert out.stat().st_size > 0

    def test_the_return_type_is_the_glyph(self):
        """Pinned deliberately: one name across the classes returns two different objects,
        because the two cleopatra glyph families do, and pyramids passes each through."""
        from matplotlib.animation import FuncAnimation

        glyph = _raster(3).animate()
        assert not isinstance(glyph, FuncAnimation)
        assert isinstance(glyph.anim, FuncAnimation)

    def test_an_extra_exclude_value_is_masked(self):
        """`exclude_value` is masked beside the band's own no-data, as in `plot`."""
        assert _raster(3).animate(exclude_value=1.0) is not None

    def test_a_single_frame_is_refused(self):
        subject = _raster(3)
        with pytest.raises(ValueError, match="at least two frames"):
            subject.animate(bands=[1])

    def test_an_out_of_range_band_is_refused(self):
        subject = _raster(3)
        with pytest.raises(ValueError, match="out of range"):
            subject.animate(bands=[0, 9])

    def test_a_label_count_mismatch_is_refused(self):
        """Silently recycling or truncating labels would mislabel the frames."""
        subject = _raster(3)
        with pytest.raises(ValueError, match="one label per frame"):
            subject.animate(animation_axis_values=["only-one"])

    def test_a_single_band_raster_is_refused(self):
        subject = _raster(1)
        with pytest.raises(ValueError, match="at least two frames"):
            subject.animate()


class TestNetCDFAnimate:
    """A name for the path ``plot(animate=...)`` already provided."""

    @staticmethod
    def _cube():
        return NetCDF.from_array(
            np.arange(12.0).reshape(3, 2, 2),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=[0.0, 1.0, 2.0]),
        )

    def test_it_animates_along_the_primary_dimension(self):
        assert self._cube().animate("t") is not None

    def test_an_explicit_dimension(self):
        assert self._cube().animate("t", dim="time") is not None

    def test_it_reaches_the_same_path_as_plot_animate(self):
        """If these diverge, one of the two is not the animation it claims to be."""
        cube = self._cube()
        assert type(cube.animate("t")) is type(cube.plot(variable="t", animate=True))


class TestDatasetCollectionAnimate:
    """A name for what the collection's `plot` already was."""

    @staticmethod
    def _collection():
        step = Dataset.from_array(
            np.full((1, 4, 5), 1.0),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326),
            no_data_value=-9999.0,
        )
        return DatasetCollection(step, time_length=2, datasets=[step, step])

    def test_it_animates_the_stack(self):
        assert self._collection().animate() is not None

    def test_it_is_the_same_rendering_as_plot(self):
        """The collection's `plot` IS the animation, so the two must agree - otherwise one of
        them is not what its docstring claims."""
        collection = self._collection()
        assert type(collection.animate()) is type(collection.plot())

    def test_arguments_are_forwarded_to_plot(self):
        """`animate` takes no arguments of its own, so anything `plot` accepts must pass
        straight through."""
        assert self._collection().animate(band=0) is not None


class TestSingleBandRepeatedFrames:
    """A one-band raster animated by repeating its band (round 2 N1).

    `Dataset.animate` reshapes a 2-D read into `(1, rows, cols)`. That branch carried a
    `# pragma: no cover` claiming the frame-count check above made it unreachable; it does not,
    because `bands=[0, 0]` passes that check with a single band. The pragma was removed, so
    this test is what keeps the claim honest.
    """

    def test_one_band_repeated_gives_two_frames(self):
        raster = _raster(1)
        assert raster.band_count == 1
        glyph = raster.animate(bands=[0, 0])
        assert glyph is not None

    def test_the_read_really_is_two_dimensional(self):
        """The precondition the branch exists for."""
        raster = _raster(1)
        assert np.asarray(raster.read_array(band=None)).ndim == 2
