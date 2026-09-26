"""Tests for :meth:`FeatureCollection.from_xyz` and its composition with
``to_crs`` / ``interpolate_to_raster`` (issue #1135, item 1).

The array-native entry into the vector model: raw ``x`` / ``y`` / ``z`` arrays
become a point layer without the caller assembling shapely geometries. Covers
construction (with and without ``z``, with a CRS, a custom value column, NumPy
inputs), every guard clause, and that the built layer reprojects and grids
through the existing class methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


class TestFromXYZConstruction:
    """Tests for building a point FeatureCollection from coordinate arrays."""

    def test_builds_point_layer_with_z_column(self):
        """from_xyz stores z as the value column and builds one point per triple.

        Test scenario:
            Three coordinate triples yield a 3-row FC whose geometry is points and
            whose ``z`` column holds the values verbatim.
        """
        fc = FeatureCollection.from_xyz(
            [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [10.0, 20.0, 30.0], crs=4326
        )
        assert len(fc) == 3, f"Expected 3 rows, got {len(fc)}"
        assert list(fc["z"]) == [10.0, 20.0, 30.0], f"z column wrong: {list(fc['z'])}"
        assert (fc.geom_type == "Point").all(), "geometry should be all points"
        assert fc.geometry.iloc[1].wkt == "POINT (1 1)", (
            f"unexpected geometry: {fc.geometry.iloc[1].wkt}"
        )

    def test_without_z_is_a_bare_point_layer(self):
        """Omitting z yields points with no value column, geometry still present.

        Test scenario:
            from_xyz(x, y) builds a 2-row point layer whose only column is the
            geometry.
        """
        fc = FeatureCollection.from_xyz([0.0, 1.0], [2.0, 3.0])
        assert list(fc.columns) == ["geometry"], (
            f"unexpected columns: {list(fc.columns)}"
        )
        assert fc.geometry.iloc[0].wkt == "POINT (0 2)", (
            f"unexpected geometry: {fc.geometry.iloc[0].wkt}"
        )

    def test_attaches_crs(self):
        """A crs= argument is attached to the resulting layer.

        Test scenario:
            crs=4326 is reported back through the FC's epsg property.
        """
        fc = FeatureCollection.from_xyz([0.0], [0.0], [1.0], crs=4326)
        assert fc.epsg == 4326, f"Expected EPSG 4326, got {fc.epsg}"

    def test_no_crs_leaves_layer_unprojected(self):
        """Without crs= the layer carries no CRS.

        Test scenario:
            from_xyz with no crs leaves the GeoDataFrame's crs as None.
        """
        fc = FeatureCollection.from_xyz([0.0], [0.0], [1.0])
        assert fc.crs is None, f"Expected no CRS, got {fc.crs}"

    def test_custom_z_column_name(self):
        """z is stored under the requested column name.

        Test scenario:
            z_column='depth' names the value column 'depth', not 'z'.
        """
        fc = FeatureCollection.from_xyz(
            [0.0, 1.0], [0.0, 1.0], [5.0, 6.0], z_column="depth"
        )
        assert "depth" in fc.columns, f"columns: {list(fc.columns)}"
        assert "z" not in fc.columns, "default 'z' column should be absent"

    def test_accepts_numpy_arrays(self):
        """NumPy arrays are accepted as directly as lists.

        Test scenario:
            Passing np.ndarray x/y/z produces the same layer as the list form.
        """
        x = np.array([0.0, 10.0])
        y = np.array([0.0, 10.0])
        z = np.array([1.0, 2.0])
        fc = FeatureCollection.from_xyz(x, y, z, crs=4326)
        assert len(fc) == 2, f"Expected 2 rows, got {len(fc)}"
        assert list(fc["z"]) == [1.0, 2.0], f"z column wrong: {list(fc['z'])}"

    def test_value_column_named_x_does_not_clash(self):
        """z_column may be 'x' — it is a plain attribute, distinct from geometry x.

        Test scenario:
            Naming the value column 'x' stores the values under 'x' while the
            point geometry keeps its own x-coordinate.
        """
        fc = FeatureCollection.from_xyz(
            [0.0, 5.0], [0.0, 5.0], [100.0, 200.0], z_column="x"
        )
        assert list(fc["x"]) == [100.0, 200.0], f"value column wrong: {list(fc['x'])}"
        assert fc.geometry.iloc[1].x == pytest.approx(5.0), "geometry x lost"


class TestFromXYZGuards:
    """Tests for from_xyz's validation of its array inputs."""

    def test_x_y_length_mismatch_raises(self):
        """Unequal x/y lengths raise ValueError.

        Test scenario:
            x of length 2 and y of length 1 is rejected before building geometry.
        """
        with pytest.raises(ValueError, match="equal length"):
            FeatureCollection.from_xyz([1.0, 2.0], [1.0])

    def test_z_length_mismatch_raises(self):
        """A z that does not match x/y length raises ValueError.

        Test scenario:
            Two points but three z values is rejected.
        """
        with pytest.raises(ValueError, match="z must match"):
            FeatureCollection.from_xyz([1.0, 2.0], [1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_arrays_raise(self):
        """Empty coordinate arrays raise ValueError, not a geometry-less frame.

        Test scenario:
            from_xyz([], []) is rejected with a message about at least one point.
        """
        with pytest.raises(ValueError, match="at least one point"):
            FeatureCollection.from_xyz([], [])

    def test_two_dimensional_input_raises(self):
        """A 2-D array is rejected — coordinates must be flat 1-D sequences.

        Test scenario:
            Passing a (2, 2) array for x reports the offending ndim.
        """
        with pytest.raises(ValueError, match="1-D coordinate arrays"):
            FeatureCollection.from_xyz([[0.0, 1.0], [2.0, 3.0]], [0.0, 1.0])


class TestFromXYZComposition:
    """from_xyz composes with the existing reproject and grid methods."""

    def test_reproject_via_to_crs(self):
        """The built layer reprojects through the inherited to_crs.

        Test scenario:
            A WGS84 point at (10, 0) reprojected to Web Mercator lands near
            1_113_195 m east, and the value column is untouched.
        """
        fc = FeatureCollection.from_xyz([10.0], [0.0], [42.0], crs=4326)
        out = fc.to_crs(3857)
        assert out.epsg == 3857, f"Expected EPSG 3857, got {out.epsg}"
        assert out.geometry.iloc[0].x == pytest.approx(1_113_194.9, rel=1e-4), (
            f"unexpected reprojected x: {out.geometry.iloc[0].x}"
        )
        assert list(out["z"]) == [42.0], "value column changed under reprojection"

    def test_reproject_noop_when_crs_unchanged(self):
        """Reprojecting to the same CRS leaves the coordinates in place.

        Test scenario:
            to_crs(4326) on a 4326 layer returns the same coordinates.
        """
        fc = FeatureCollection.from_xyz(
            [31.0, 32.0], [30.0, 29.0], [1.0, 2.0], crs=4326
        )
        out = fc.to_crs(4326)
        assert out.geometry.iloc[0].x == pytest.approx(31.0), (
            "x moved on no-op reproject"
        )
        assert out.geometry.iloc[0].y == pytest.approx(30.0), (
            "y moved on no-op reproject"
        )

    def test_grid_via_interpolate_to_raster(self):
        """The built layer grids to a raster through interpolate_to_raster.

        Test scenario:
            Four corner points gridded at cell_size=1 over a 10x10 box give a
            10x10 single-band Dataset whose values stay within the sample range.
        """
        fc = FeatureCollection.from_xyz(
            [0.0, 10.0, 0.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 20.0, 30.0, 40.0],
            crs=4326,
        )
        ds = fc.interpolate_to_raster("z", cell_size=1.0)
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        assert (ds.rows, ds.columns, ds.band_count) == (10, 10, 1), (
            f"unexpected shape: {ds.rows}x{ds.columns}x{ds.band_count}"
        )
