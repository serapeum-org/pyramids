"""Tests for GCP/RPC georeferencing (the Georef engine, issue-driven GR-* tasks).

Fixtures are synthetic and offline: a small in-memory raster with four corner
ground-control points, generated via ``set_gcps`` rather than shipping a binary.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset
from pyramids.dataset._gcp import GroundControlPoint

pytestmark = pytest.mark.core


RPC_SAMPLE: dict[str, str] = {
    "HEIGHT_OFF": "100",
    "HEIGHT_SCALE": "50",
    "LAT_OFF": "49.5",
    "LAT_SCALE": "0.5",
    "LONG_OFF": "10.5",
    "LONG_SCALE": "0.5",
    "LINE_OFF": "4",
    "LINE_SCALE": "4",
    "SAMP_OFF": "4",
    "SAMP_SCALE": "4",
    "LINE_NUM_COEFF": " ".join(["0"] * 20),
    "LINE_DEN_COEFF": " ".join(["1"] + ["0"] * 19),
    "SAMP_NUM_COEFF": " ".join(["0"] * 20),
    "SAMP_DEN_COEFF": " ".join(["1"] + ["0"] * 19),
}


@pytest.fixture
def corner_gcps() -> list[GroundControlPoint]:
    """Four corner control points of an 8x8 raster in EPSG:4326.

    Returns:
        list[GroundControlPoint]: top-left, top-right, bottom-left, bottom-right.
    """
    return [
        GroundControlPoint(row=0, col=0, x=10.0, y=50.0, id="tl"),
        GroundControlPoint(row=0, col=8, x=11.0, y=50.0, id="tr"),
        GroundControlPoint(row=8, col=0, x=10.0, y=49.0, id="bl"),
        GroundControlPoint(row=8, col=8, x=11.0, y=49.0, id="br"),
    ]


@pytest.fixture
def writable_dataset() -> Dataset:
    """A writable in-memory 8x8 float32 dataset.

    Returns:
        Dataset: MEM-backed (always writable), no GCPs yet.
    """
    return Dataset.create_from_array(
        np.ones((8, 8), dtype="float32"), top_left_corner=(0.0, 8.0), cell_size=1.0
    )


class TestGroundControlPoint:
    """Tests for the GroundControlPoint value object."""

    def test_to_gdal_maps_pixel_and_map_coords(self):
        """`to_gdal` puts col/row on pixel/line and keeps the map coordinate.

        Test scenario:
            A point at (col=7, row=3) -> (x=1, y=2) becomes a gdal.GCP with the
            same pixel/line and X/Y.
        """
        g = GroundControlPoint(row=3.0, col=7.0, x=1.0, y=2.0).to_gdal()
        assert (g.GCPPixel, g.GCPLine, g.GCPX, g.GCPY) == (7.0, 3.0, 1.0, 2.0)

    def test_round_trip_through_gdal(self):
        """`from_gdal(to_gdal())` preserves all fields.

        Test scenario:
            A fully-populated point survives the GDAL round-trip.
        """
        original = GroundControlPoint(
            row=4.0, col=2.0, x=11.5, y=46.2, z=3.0, id="p1", info="note"
        )
        back = GroundControlPoint.from_gdal(original.to_gdal())
        assert back == original

    def test_empty_id_info_become_none(self):
        """Empty GDAL Id/Info come back as None, not empty strings.

        Test scenario:
            A point with no id/info round-trips to None id/info.
        """
        back = GroundControlPoint.from_gdal(
            GroundControlPoint(row=0, col=0, x=0.0, y=0.0).to_gdal()
        )
        assert back.id is None and back.info is None


class TestReadGCPs:
    """Tests for the GCP read properties (gcps / gcp_count / gcp_projection / has_gcps)."""

    def test_plain_raster_has_no_gcps(self, writable_dataset):
        """A raster with no GCPs reports zero / empty / None.

        Test scenario:
            Fresh dataset: gcp_count 0, gcps [], gcp_projection None, has_gcps False.
        """
        assert writable_dataset.gcp_count == 0
        assert writable_dataset.gcps == []
        assert writable_dataset.gcp_projection is None
        assert writable_dataset.has_gcps is False

    def test_reads_attached_gcps(self, writable_dataset, corner_gcps):
        """After set_gcps the read properties return the points and CRS.

        Test scenario:
            4 corner points attached -> count 4, has_gcps True, projection mentions
            4326, and the first point's pixel/map coords match the input.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        assert writable_dataset.gcp_count == 4
        assert writable_dataset.has_gcps is True
        assert "4326" in writable_dataset.gcp_projection
        first = writable_dataset.gcps[0]
        assert (first.col, first.row, first.x, first.y) == (0.0, 0.0, 10.0, 50.0)

    def test_round_trip_preserves_points(self, writable_dataset, corner_gcps):
        """The read-back points equal the value objects passed to set_gcps.

        Test scenario:
            set_gcps then gcps returns equal GroundControlPoint records.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        assert writable_dataset.gcps == corner_gcps


class TestReadRPC:
    """Tests for the RPC read properties (rpcs / has_rpcs)."""

    def test_plain_raster_has_no_rpcs(self, writable_dataset):
        """A raster without RPC metadata reports None / False.

        Test scenario:
            Fresh dataset: rpcs is None, has_rpcs is False.
        """
        assert writable_dataset.rpcs is None
        assert writable_dataset.has_rpcs is False

    def test_reads_rpc_metadata(self, writable_dataset):
        """rpcs returns the RPC domain set on the underlying raster.

        Test scenario:
            After SetMetadata(RPC_SAMPLE, "RPC"), rpcs["HEIGHT_OFF"] matches and
            has_rpcs is True.
        """
        writable_dataset.raster.SetMetadata(RPC_SAMPLE, "RPC")
        assert writable_dataset.has_rpcs is True
        assert writable_dataset.rpcs["HEIGHT_OFF"] == "100"


class TestSetRPC:
    """Tests for Georef.set_rpcs."""

    def test_round_trip(self, writable_dataset):
        """set_rpcs writes the RPC domain so rpcs reads it back.

        Test scenario:
            A complete RPC dict set then read returns equal values.
        """
        writable_dataset.set_rpcs(RPC_SAMPLE)
        assert writable_dataset.rpcs["HEIGHT_OFF"] == "100"
        assert writable_dataset.rpcs["LINE_DEN_COEFF"].split()[0] == "1"

    def test_stringifies_numeric_values(self, writable_dataset):
        """Numeric RPC values are stringified before writing.

        Test scenario:
            HEIGHT_OFF given as a float comes back as its string form.
        """
        rpc = dict(RPC_SAMPLE)
        rpc["HEIGHT_OFF"] = 123.5
        writable_dataset.set_rpcs(rpc)
        assert writable_dataset.rpcs["HEIGHT_OFF"] == "123.5"

    def test_missing_keys_raise(self, writable_dataset):
        """A dict missing required keys is rejected, listing them.

        Test scenario:
            Dropping HEIGHT_OFF raises ValueError naming the missing key.
        """
        rpc = dict(RPC_SAMPLE)
        del rpc["HEIGHT_OFF"]
        with pytest.raises(ValueError, match="HEIGHT_OFF"):
            writable_dataset.set_rpcs(rpc)

    def test_read_only_raises(self, tmp_path):
        """A read-only dataset rejects set_rpcs.

        Test scenario:
            read_only=True raises ReadOnlyError.
        """
        path = tmp_path / "plain.tif"
        Dataset.create_from_array(
            np.ones((4, 4), dtype="float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
        ).to_file(str(path))
        ds = Dataset.read_file(str(path), read_only=True)
        with pytest.raises(ReadOnlyError):
            ds.set_rpcs(RPC_SAMPLE)


class TestWarpFromGCPs:
    """Tests for Georef.georeference (warp from GCPs)."""

    def test_no_gcps_raises(self, writable_dataset):
        """georeference without GCPs is rejected.

        Test scenario:
            A dataset with no GCPs raises ValueError mentioning GCPs.
        """
        with pytest.raises(ValueError, match="no GCPs"):
            writable_dataset.georeference()

    def test_polynomial_warp_into_gcp_crs(self, writable_dataset, corner_gcps):
        """Default warp lands in the GCP CRS and brackets the GCP extent.

        Test scenario:
            4 corner GCPs (10-11E, 49-50N) -> epsg 4326 and bbox covers that box.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        out = writable_dataset.georeference()
        assert out.epsg == 4326
        xmin, ymin, xmax, ymax = out.bbox
        assert xmin <= 10.0 + 1e-6 and xmax >= 11.0 - 1e-6
        assert ymin <= 49.0 + 1e-6 and ymax >= 50.0 - 1e-6

    def test_tps_transform(self, writable_dataset, corner_gcps):
        """A thin-plate-spline warp also produces a 4326 raster.

        Test scenario:
            transform="tps" warps into the GCP CRS without error.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        out = writable_dataset.georeference(transform="tps")
        assert out.epsg == 4326

    def test_reproject_in_same_pass(self, writable_dataset, corner_gcps):
        """to_epsg reprojects the georeferenced result in one pass.

        Test scenario:
            georeference(to_epsg=3857) yields a Web-Mercator raster.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        out = writable_dataset.georeference(to_epsg=3857)
        assert out.epsg == 3857

    def test_lazy_equals_eager(self, writable_dataset, corner_gcps):
        """A lazy VRT view reads the same pixels as the eager result.

        Test scenario:
            lazy=True and lazy=False produce allclose arrays.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        eager = writable_dataset.georeference(lazy=False)
        lazy = writable_dataset.georeference(lazy=True)
        assert np.allclose(np.asarray(eager.read_array()), np.asarray(lazy.read_array()))

    def test_invalid_transform_raises(self, writable_dataset, corner_gcps):
        """An unsupported transform name is rejected.

        Test scenario:
            transform="bogus" raises ValueError.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        with pytest.raises(ValueError, match="polynomial.*tps|transform must"):
            writable_dataset.georeference(transform="bogus")

    def test_invalid_order_raises(self, writable_dataset, corner_gcps):
        """A polynomial order outside 1-3 is rejected.

        Test scenario:
            order=7 raises ValueError.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        with pytest.raises(ValueError, match="order must"):
            writable_dataset.georeference(order=7)


class TestSetGCPs:
    """Tests for Georef.set_gcps (and the Dataset facade)."""

    def test_attaches_gcps_and_projection(self, writable_dataset, corner_gcps):
        """set_gcps writes the points and an EPSG:4326 projection to the raster.

        Test scenario:
            After set_gcps the underlying GDAL dataset reports 4 GCPs and a
            4326 projection.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        raster = writable_dataset.raster
        assert raster.GetGCPCount() == 4
        assert "4326" in raster.GetGCPProjection()

    def test_empty_list_raises_value_error(self, writable_dataset):
        """An empty GCP list is rejected.

        Test scenario:
            set_gcps([], 4326) raises ValueError.
        """
        with pytest.raises(ValueError, match="at least one"):
            writable_dataset.set_gcps([], 4326)

    def test_read_only_raises(self, corner_gcps, tmp_path):
        """A read-only dataset rejects set_gcps.

        Test scenario:
            A dataset opened read_only=True raises ReadOnlyError.
        """
        path = tmp_path / "plain.tif"
        Dataset.create_from_array(
        np.ones((8, 8), dtype="float32"), top_left_corner=(0.0, 8.0), cell_size=1.0
    ).to_file(str(path))
        ds = Dataset.read_file(str(path), read_only=True)
        with pytest.raises(ReadOnlyError):
            ds.set_gcps(corner_gcps, 4326)
