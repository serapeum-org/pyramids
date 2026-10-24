"""Terrain derivatives on a mesh DEM: slope / aspect / hillshade (issue #1374).

An unstructured bed-level mesh is a DEM, but the raster family reads a 3x3 pixel window that
has no mesh equivalent. These tests validate the per-face plane fit against planes whose slope
and aspect are known analytically, and pin the raster conventions the mesh versions mirror.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])


def _dem(node_z, location="node"):
    """A two-triangle mesh carrying ``node_z`` as the elevation."""
    return UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"z": np.asarray(node_z, dtype="float64")},
        data_locations={"z": location},
    )


class TestSlope:
    """Slope magnitude against analytically known planes."""

    def test_a_one_to_one_plane_is_45_degrees(self):
        """z = x rises one unit per unit east, so the surface is at 45 degrees."""
        assert _dem(_NODE_X).slope().tolist() == pytest.approx([45.0, 45.0])

    def test_radians(self):
        assert _dem(_NODE_X).slope("radians").tolist() == pytest.approx(
            [np.pi / 4, np.pi / 4]
        )

    def test_a_flat_mesh_has_zero_slope(self):
        assert _dem(np.zeros(4)).slope().tolist() == pytest.approx([0.0, 0.0])

    def test_a_steeper_plane(self):
        """z = 2x is arctan(2) ~= 63.43 degrees."""
        expected = np.degrees(np.arctan(2.0))
        assert _dem(2.0 * _NODE_X).slope().tolist() == pytest.approx(
            [expected, expected]
        )

    def test_invalid_units_are_refused(self):
        with pytest.raises(ValueError, match="degrees.*radians"):
            _dem(_NODE_X).slope("gradians")


class TestAspect:
    """Aspect names the direction the slope faces, clockwise from north."""

    def test_an_east_rising_plane_faces_west(self):
        """Mirrors the raster convention exactly: gradient east => faces west => 270."""
        assert _dem(_NODE_X).aspect().tolist() == pytest.approx([270.0, 270.0])

    def test_a_north_rising_plane_faces_south(self):
        """The sign mapping between map coordinates and the raster's row-wise derivative."""
        assert _dem(_NODE_Y).aspect().tolist() == pytest.approx([180.0, 180.0])

    def test_a_west_rising_plane_faces_east(self):
        assert _dem(-_NODE_X).aspect().tolist() == pytest.approx([90.0, 90.0])

    def test_the_range_is_zero_to_360(self):
        values = _dem(_NODE_X + 2.0 * _NODE_Y).aspect()
        assert np.all(values >= 0.0)
        assert np.all(values < 360.0)


class TestHillshade:
    """Shaded relief, matching the raster scaling."""

    def test_a_flat_mesh_saturates_at_the_illumination_level(self):
        """No gradient => pure sin(altitude) * 255, as the raster version answers."""
        expected = np.sin(np.deg2rad(45.0)) * 255.0
        assert _dem(np.zeros(4)).hillshade().tolist() == pytest.approx(
            [expected, expected]
        )

    def test_stays_within_zero_and_255(self):
        values = _dem(10.0 * _NODE_X).hillshade()
        assert np.all(values >= 0.0)
        assert np.all(values <= 255.0)

    def test_a_slope_lit_from_the_front_is_brighter_than_from_behind(self):
        """An east-facing surface is brighter lit from the east than from the west."""
        ds = _dem(-_NODE_X)  # rises west, faces east
        from_east = ds.hillshade(azimuth=90.0).mean()
        from_west = ds.hillshade(azimuth=270.0).mean()
        assert from_east > from_west


class TestInputLocations:
    """Where the elevation may live."""

    def test_a_face_located_elevation_is_averaged_onto_the_nodes(self):
        """Bed level usually lives on faces (`mesh2d_flowelem_bl`), so this must work."""
        ds = _dem(np.array([1.0, 2.0]), location="face")
        result = ds.slope()
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_an_edge_located_elevation_is_refused(self):
        """There is no defined route from edges to the nodes the plane fit needs."""
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={"z": np.zeros(5)},
            data_locations={"z": "edge"},
        )
        with pytest.raises(ValueError, match="node- or face-located"):
            ds.slope()

    def test_a_temporal_elevation_is_refused(self):
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={"z": np.zeros((3, 4))},
            data_locations={"z": "node"},
        )
        with pytest.raises(ValueError, match="one value per element"):
            ds.slope()


class TestDegenerateFaces:
    """A face with no plane through it yields NaN rather than raising."""

    def test_collinear_nodes_give_nan(self):
        """Three points on a line define no plane, so the gradient is undefined."""
        ds = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0]),  # all collinear
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"z": np.array([0.0, 1.0, 2.0])},
            data_locations={"z": "node"},
        )
        assert np.isnan(ds.slope()).all(), "a collinear face has no defined slope"
        assert np.isnan(ds.aspect()).all()

    def test_a_degenerate_face_does_not_poison_its_neighbours(self):
        """The NaN must stay local to the face that has no plane."""
        ds = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 3.0, 1.0]),
            node_y=np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [1, 2, 4]]),
            data={"z": np.array([0.0, 1.0, 2.0, 3.0, 5.0])},
            data_locations={"z": "node"},
        )
        result = ds.slope()
        assert np.isnan(result[0]), "the collinear face is undefined"
        assert np.isfinite(result[1]), "its neighbour is unaffected"


class TestKernelGuards:
    """The kernel's own validation, which the facade pre-empts from above."""

    def test_a_wrong_length_node_array_is_refused(self):
        """`face_plane_gradient` is the documented reuse surface, so its guard is its
        contract - the facade never passes a bad length, so only this reaches it."""
        from pyramids.netcdf.ugrid.terrain import face_plane_gradient

        mesh = _dem(_NODE_X).mesh
        with pytest.raises(ValueError, match="one value per node"):
            face_plane_gradient(mesh, np.array([1.0, 2.0]))

    def test_a_two_dimensional_node_array_is_refused(self):
        from pyramids.netcdf.ugrid.terrain import face_plane_gradient

        mesh = _dem(_NODE_X).mesh
        with pytest.raises(ValueError, match="one value per node"):
            face_plane_gradient(mesh, np.zeros((2, 4)))


_QUAD_X = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
_QUAD_Y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
_QUADS = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])


def _void_dem(values, location="node", nodata=-9999.0):
    """A two-quad DEM carrying a declared sentinel."""
    ds = UgridDataset.from_arrays(
        node_x=_QUAD_X,
        node_y=_QUAD_Y,
        face_node_connectivity=_QUADS,
        data={"z": np.asarray(values, dtype="float64")},
        data_locations={"z": location},
    )
    ds["z"].nodata = nodata
    return ds


class TestNodataIsNotTerrain:
    """A declared sentinel is a void, not an elevation (review H4).

    The raster kernels these formulas mirror document the opposite of what the mesh side did:
    "a centred difference straddling a void has no defined derivative". The mesh treated
    -9999 as a measurement and fabricated a vertical cliff out of it.
    """

    def test_a_sentinel_node_does_not_fabricate_a_cliff(self):
        """Was slope 89.99 degrees on an otherwise flat mesh."""
        result = _void_dem([0.0, 0.0, 0.0, 0.0, 0.0, -9999.0]).slope()
        assert result[0] == pytest.approx(0.0), "the untouched face is still flat"
        assert np.isnan(result[1]), "the face touching the void has no defined slope"

    def test_a_sentinel_node_does_not_fabricate_an_aspect(self):
        """Was a confident 45 degrees."""
        result = _void_dem([0.0, 0.0, 0.0, 0.0, 0.0, -9999.0]).aspect()
        assert np.isnan(result[1]), "no direction is defined across a void"

    def test_a_sentinel_node_does_not_fabricate_a_shade(self):
        """Was 0.0255 - a near-black face indistinguishable from a real shadow."""
        result = _void_dem([0.0, 0.0, 0.0, 0.0, 0.0, -9999.0]).hillshade()
        assert np.isnan(result[1]), "a void cannot be shaded"

    def test_a_face_located_sentinel_is_also_excluded(self):
        """Bed level usually lives on faces, and the sentinel was averaged onto every node of
        its face, spreading the contamination."""
        result = _void_dem([1.0, -9999.0], location="face").slope()
        assert np.isnan(result[1]), "the void's own face is undefined"

    def test_no_declared_nodata_leaves_every_face_defined(self):
        """Without a sentinel nothing is masked, so a plain DEM is unaffected."""
        ds = UgridDataset.from_arrays(
            node_x=_QUAD_X,
            node_y=_QUAD_Y,
            face_node_connectivity=_QUADS,
            data={"z": _QUAD_X.copy()},
            data_locations={"z": "node"},
        )
        assert np.all(np.isfinite(ds.slope())), "no sentinel declared, nothing masked"

    def test_nan_in_the_data_behaves_the_same_as_the_sentinel(self):
        """The two spellings of "missing" must agree."""
        sentinel = _void_dem([0.0, 0.0, 0.0, 0.0, 0.0, -9999.0]).slope()
        nan = _void_dem([0.0, 0.0, 0.0, 0.0, 0.0, np.nan], nodata=None).slope()
        assert np.isnan(sentinel[1]) and np.isnan(nan[1])
        assert sentinel[0] == pytest.approx(nan[0])
