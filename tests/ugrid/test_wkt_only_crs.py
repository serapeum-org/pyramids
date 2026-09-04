"""A mesh whose CRS is WKT-only must still be usable.

`UgridDataset` describes its CRS with a WKT string copied from the file. A
projected CRS often has no EPSG code to resolve back to, so `epsg` is `None`
while the mesh plainly has a CRS. Two methods took `self.epsg` as the whole
answer: `to_dataset` fell back to EPSG:4326 and stamped degrees onto metre
coordinates, and `to_crs` refused to reproject at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = pytest.mark.core

# An orthographic CRS in metres. Deliberately one with no EPSG code, which is
# the whole point of the fixture.
ORTHO_WKT = (
    'PROJCS["ortho",'
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Orthographic"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",0],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
    'UNIT["metre",1]]'
)


@pytest.fixture
def wkt_only_projected_mesh(triangle_mesh) -> UgridDataset:
    """A mesh carrying a projected WKT and no EPSG code.

    Self-asserting: the fixture is worthless if the CRS ever resolves to an
    EPSG code, because the branch under test would stop being reached.
    """
    dataset = UgridDataset(
        mesh=triangle_mesh,
        data_variables={
            "h": MeshVariable(
                name="h",
                location="face",
                mesh_name="mesh2d",
                shape=(2,),
                _data=np.array([1.0, 2.0]),
            )
        },
        global_attributes={},
        crs_wkt=ORTHO_WKT,
    )
    assert dataset.epsg is None, "fixture must have no EPSG code"
    assert dataset.crs is not None, "fixture must have a parseable CRS"
    assert dataset.crs.is_projected, "fixture must be projected"
    return dataset


class TestCrsWktProperty:
    """`crs_wkt` exposes the WKT only when it parses."""

    def test_a_parseable_wkt_is_returned(self, wkt_only_projected_mesh: UgridDataset):
        """A mesh with a usable CRS hands back its WKT."""
        assert wkt_only_projected_mesh.crs_wkt == ORTHO_WKT

    def test_a_malformed_wkt_is_reported_as_absent(self, triangle_mesh):
        """A WKT that does not parse reads as no CRS, and never escapes raw.

        This is the condition that makes the rest of the change safe: `crs`
        deliberately swallows a parse failure and reports `None`. Returning
        `_crs_wkt` verbatim would hand the broken string to callers and move
        the parse out from under that handler.
        """
        dataset = UgridDataset(
            mesh=triangle_mesh,
            data_variables={},
            global_attributes={},
            crs_wkt="this is not WKT",
        )

        assert dataset.crs is None
        assert dataset.crs_wkt is None

    def test_a_mesh_without_a_crs_has_no_wkt(self, triangle_mesh):
        """No CRS at all means no WKT."""
        dataset = UgridDataset(
            mesh=triangle_mesh, data_variables={}, global_attributes={}
        )

        assert dataset.crs_wkt is None


class TestWktOnlyMeshIsUsable:
    """The two methods that previously took `epsg` as the whole answer."""

    def test_to_dataset_does_not_relabel_a_projected_mesh_as_wgs84(
        self, wkt_only_projected_mesh: UgridDataset
    ):
        """A projected mesh keeps its own CRS instead of defaulting to 4326.

        The old fallback stamped EPSG:4326 -- degrees -- onto coordinates in
        metres, producing a raster that claimed a CRS it did not have.
        """
        result = wkt_only_projected_mesh.to_dataset("h", cell_size=0.5)

        assert result.epsg != 4326, "projected mesh was relabelled as WGS84"

    def test_to_crs_reprojects_a_wkt_only_mesh(
        self, wkt_only_projected_mesh: UgridDataset
    ):
        """Reprojection works from a CRS described only by WKT.

        This used to raise "Cannot reproject: source CRS is unknown" while the
        WKT sat unused on the object.
        """
        result = wkt_only_projected_mesh.to_crs(4326)

        assert result.epsg == 4326

    def test_a_mesh_with_no_crs_still_refuses_to_reproject(self, triangle_mesh):
        """The error is kept for the case it was written for."""
        dataset = UgridDataset(
            mesh=triangle_mesh, data_variables={}, global_attributes={}
        )

        with pytest.raises(ValueError, match="source CRS is unknown"):
            dataset.to_crs(4326)

    def test_str_reports_the_crs_name_when_there_is_no_epsg_code(
        self, wkt_only_projected_mesh: UgridDataset
    ):
        """A mesh with a CRS but no code stops printing "unknown"."""
        assert "unknown" not in str(wkt_only_projected_mesh)


class TestAMeshWithNoCrsAtAllKeepsItsWgs84Default:
    """Teaching this line about WKT must not narrow the other case."""

    def test_it_still_rasterises_to_wgs84(self, triangle_mesh):
        """The default that predates the branch, on the mesh that relies on it.

        Test scenario:
            `triangle_mesh` has neither an EPSG code nor a WKT. Resolving the
            CRS through `crs_spec` alone yields `None` for it, which would
            hand back a georeference-free raster -- so
            `to_dataset(...).to_file("out.tif")` would write a GeoTIFF with no
            CRS where it has always written WGS 84.
        """
        dataset = UgridDataset(
            mesh=triangle_mesh,
            data_variables={
                "h": MeshVariable(
                    name="h",
                    location="face",
                    mesh_name="mesh2d",
                    shape=(2,),
                    _data=np.array([1.0, 2.0]),
                )
            },
            global_attributes={},
        )
        assert dataset.epsg is None and dataset.crs_wkt is None

        raster = dataset.to_dataset("h", cell_size=0.25)

        assert raster.epsg == 4326
        assert raster.crs, "the raster must carry a CRS, not an empty string"

    def test_an_explicit_epsg_still_wins(self, triangle_mesh):
        """The default is a fallback, not an override.

        Test scenario:
            A caller naming the CRS must get it. The `or 4326` sits behind the
            `epsg is not None` branch, so an argument of 3857 has to survive
            it.
        """
        dataset = UgridDataset(
            mesh=triangle_mesh,
            data_variables={
                "h": MeshVariable(
                    name="h",
                    location="face",
                    mesh_name="mesh2d",
                    shape=(2,),
                    _data=np.array([1.0, 2.0]),
                )
            },
            global_attributes={},
        )

        raster = dataset.to_dataset("h", cell_size=0.25, epsg=3857)

        assert raster.epsg == 3857

    def test_the_wkt_only_mesh_is_not_dragged_back_to_wgs84(
        self, wkt_only_projected_mesh: UgridDataset
    ):
        """The fix this restores a default behind must still hold.

        Test scenario:
            `crs_spec` returns the orthographic WKT for this mesh, which is
            truthy, so the `or 4326` never fires. Had it been written as a
            plain `self.epsg or 4326` the metre grid would be relabelled
            degrees again.
        """
        raster = wkt_only_projected_mesh.to_dataset("h", cell_size=1000.0)

        assert raster.epsg != 4326
        assert "Orthographic" in raster.crs
