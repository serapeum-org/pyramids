"""Unit tests for :mod:`pyramids.dataset.ops._geobox_zarr`.

Covers the GeoZarr geobox helpers with zarr-python only (no xarray): pixel-centre
coordinate maths, the ``write_geobox`` array/attribute layout, and ``read_geobox``
for both the GeoZarr ``spatial_ref`` branch and the legacy flat-attribute branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_zarr
from pyramids.dataset.ops._geobox_zarr import (
    GRID_MAPPING_VAR,
    detect_data_var,
    pixel_centre_coords,
    read_geobox,
    write_geobox,
)

pytestmark = pytest.mark.core

try:
    import_zarr("zarr not installed")
    import zarr
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_ZARR = False
else:
    HAS_ZARR = True
requires_zarr = pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")

_WKT_4326 = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)
_GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


class TestPixelCentreCoords:
    """Tests for pixel_centre_coords."""

    def test_centre_offsets(self):
        """Coordinates are at pixel centres, not edges.

        Test scenario:
            For origin (0, 4), dx=1, dy=-1, a 4×5 grid: x starts at 0.5 and steps
            by 1; y starts at 3.5 and steps by -1 (north-up).
        """
        x, y = pixel_centre_coords(_GT, rows=4, cols=5)
        np.testing.assert_allclose(x, [0.5, 1.5, 2.5, 3.5, 4.5], err_msg=f"x={x}")
        np.testing.assert_allclose(y, [3.5, 2.5, 1.5, 0.5], err_msg=f"y={y}")

    def test_lengths_match_rows_cols(self):
        """x has length cols and y has length rows.

        Test scenario:
            A 3-row, 7-col request yields len(x)==7 and len(y)==3.
        """
        x, y = pixel_centre_coords(_GT, rows=3, cols=7)
        assert (len(x), len(y)) == (7, 3), f"lengths wrong: x={len(x)}, y={len(y)}"

    def test_single_row_and_col(self):
        """A 1×1 grid yields one centre coordinate on each axis.

        Test scenario:
            rows=1, cols=1 returns x=[0.5] and y=[3.5] for the sample transform.
        """
        x, y = pixel_centre_coords(_GT, rows=1, cols=1)
        np.testing.assert_allclose(x, [0.5], err_msg=f"x={x}")
        np.testing.assert_allclose(y, [3.5], err_msg=f"y={y}")


@requires_zarr
class TestWriteGeobox:
    """Tests for write_geobox."""

    def _group_with_data(self, tmp_path, bands=2, rows=4, cols=5):
        store = str(tmp_path / "w.zarr")
        group = zarr.open_group(store, mode="w")
        group.create_array("data", data=np.zeros((bands, rows, cols), dtype="float32"))
        return group

    def test_creates_spatial_ref_and_xy(self, tmp_path):
        """write_geobox adds spatial_ref + x/y arrays alongside data.

        Test scenario:
            After writing, the group exposes ``spatial_ref``, ``x`` and ``y``
            arrays with x/y lengths matching cols/rows.
        """
        group = self._group_with_data(tmp_path, rows=4, cols=5)
        write_geobox(
            group,
            data_name="data",
            epsg=4326,
            geotransform=_GT,
            crs_wkt=_WKT_4326,
            rows=4,
            cols=5,
            dims=["band", "y", "x"],
        )
        keys = set(group.array_keys())
        assert {"data", GRID_MAPPING_VAR, "x", "y"} <= keys, f"keys={keys}"
        assert group["x"].shape == (5,), f"x shape {group['x'].shape}"
        assert group["y"].shape == (4,), f"y shape {group['y'].shape}"

    def test_sets_array_dimensions_and_grid_mapping(self, tmp_path):
        """Dims and grid_mapping attrs follow the GeoZarr convention.

        Test scenario:
            data carries ``_ARRAY_DIMENSIONS=['band','y','x']`` and
            ``grid_mapping='spatial_ref'``; x/y carry their 1-D dim names.
        """
        group = self._group_with_data(tmp_path, rows=4, cols=5)
        write_geobox(
            group,
            data_name="data",
            epsg=4326,
            geotransform=_GT,
            crs_wkt=_WKT_4326,
            rows=4,
            cols=5,
            dims=["band", "y", "x"],
        )
        assert group["data"].attrs["_ARRAY_DIMENSIONS"] == ["band", "y", "x"]
        assert group["data"].attrs["grid_mapping"] == GRID_MAPPING_VAR
        assert group["x"].attrs["_ARRAY_DIMENSIONS"] == ["x"]
        assert group["y"].attrs["_ARRAY_DIMENSIONS"] == ["y"]

    def test_spatial_ref_attrs_and_value(self, tmp_path):
        """spatial_ref stores WKT + GeoTransform + epsg; value is the epsg.

        Test scenario:
            The grid-mapping array holds ``crs_wkt``, a space-joined
            ``GeoTransform`` and ``epsg``; its scalar value equals the epsg.
        """
        group = self._group_with_data(tmp_path)
        write_geobox(
            group,
            data_name="data",
            epsg=4326,
            geotransform=_GT,
            crs_wkt=_WKT_4326,
            rows=4,
            cols=5,
            dims=["band", "y", "x"],
        )
        sr = group[GRID_MAPPING_VAR]
        assert sr.attrs["crs_wkt"] == _WKT_4326, "crs_wkt not stored"
        assert sr.attrs["GeoTransform"] == "0.0 1.0 0.0 4.0 0.0 -1.0"
        assert sr.attrs["epsg"] == 4326, f"epsg attr {sr.attrs['epsg']}"
        assert int(sr[()]) == 4326, f"spatial_ref value {int(sr[()])}"

    def test_epsg_zero_when_none(self, tmp_path):
        """A falsy epsg is stored as 0 (CRS with no authority code).

        Test scenario:
            Passing epsg=0 (or None-like) stores ``epsg=0`` and a scalar value
            of 0, while the WKT still carries the real CRS.
        """
        group = self._group_with_data(tmp_path)
        write_geobox(
            group,
            data_name="data",
            epsg=0,
            geotransform=_GT,
            crs_wkt=_WKT_4326,
            rows=4,
            cols=5,
            dims=["band", "y", "x"],
        )
        sr = group[GRID_MAPPING_VAR]
        assert sr.attrs["epsg"] == 0, f"epsg attr {sr.attrs['epsg']}"
        assert int(sr[()]) == 0, f"value {int(sr[()])}"


@requires_zarr
class TestReadGeobox:
    """Tests for read_geobox."""

    def _written_group(self, tmp_path):
        store = str(tmp_path / "r.zarr")
        group = zarr.open_group(store, mode="w")
        group.create_array("data", data=np.zeros((1, 4, 5), dtype="float32"))
        write_geobox(
            group,
            data_name="data",
            epsg=4326,
            geotransform=_GT,
            crs_wkt=_WKT_4326,
            rows=4,
            cols=5,
            dims=["band", "y", "x"],
        )
        return group

    def test_geozarr_branch(self, tmp_path):
        """Reads CRS/transform/epsg from the spatial_ref grid mapping.

        Test scenario:
            A group written by write_geobox reports legacy=False and recovers
            the WKT, the geotransform tuple and the epsg.
        """
        result = read_geobox(self._written_group(tmp_path))
        assert result["legacy"] is False, "should not be legacy"
        assert result["epsg"] == 4326, f"epsg {result['epsg']}"
        assert result["crs_wkt"] == _WKT_4326, "wkt not recovered"
        np.testing.assert_allclose(result["geotransform"], _GT)

    def test_legacy_branch_warns(self, tmp_path):
        """Legacy flat-attr store reads with a DeprecationWarning.

        Test scenario:
            A group whose ``data`` array carries the geobox as flat attrs (no
            spatial_ref array) is read with legacy=True and emits a
            DeprecationWarning, still recovering the transform + epsg.
        """
        store = str(tmp_path / "legacy.zarr")
        group = zarr.open_group(store, mode="w")
        data = group.create_array("data", data=np.zeros((1, 4, 5), dtype="float32"))
        data.attrs.update(
            {
                "spatial_ref": _WKT_4326,
                "GeoTransform": "0.0 1.0 0.0 4.0 0.0 -1.0",
                "epsg": 4326,
            }
        )
        with pytest.warns(DeprecationWarning, match="legacy pyramids geobox"):
            result = read_geobox(group)
        assert result["legacy"] is True, "should be legacy"
        assert result["epsg"] == 4326, f"epsg {result['epsg']}"
        np.testing.assert_allclose(result["geotransform"], _GT)

    def test_prefers_crs_wkt_key_over_spatial_ref(self, tmp_path):
        """``crs_wkt`` attr wins over the ``spatial_ref`` attr when both differ.

        Test scenario:
            The grid-mapping array carries both keys; read_geobox returns the
            ``crs_wkt`` value.
        """
        group = self._written_group(tmp_path)
        group[GRID_MAPPING_VAR].attrs["crs_wkt"] = "PREFERRED_WKT"
        group[GRID_MAPPING_VAR].attrs["spatial_ref"] = "OTHER_WKT"
        result = read_geobox(group)
        assert result["crs_wkt"] == "PREFERRED_WKT", f"got {result['crs_wkt']}"

    def _drop_geotransform(self, group):
        attrs = dict(group[GRID_MAPPING_VAR].attrs)
        attrs.pop("GeoTransform")
        group[GRID_MAPPING_VAR].attrs.clear()
        group[GRID_MAPPING_VAR].attrs.update(attrs)

    def test_missing_geotransform_derives_from_xy(self, tmp_path):
        """No GeoTransform attr → derive it from the x/y coords (FR-8).

        Test scenario:
            Removing the ``GeoTransform`` attr but keeping the ``x``/``y``
            pixel-centre coords makes read_geobox reconstruct the transform from
            them (matching the original) rather than raising.
        """
        group = self._written_group(tmp_path)
        self._drop_geotransform(group)
        result = read_geobox(group)
        np.testing.assert_allclose(result["geotransform"], _GT)

    def test_missing_geotransform_and_xy_raises(self, tmp_path):
        """No GeoTransform and no x/y coords → KeyError (FR-8).

        Test scenario:
            With both the ``GeoTransform`` attr and the ``x``/``y`` coordinate
            arrays gone, read_geobox can't determine the transform and raises.
        """
        group = self._written_group(tmp_path)
        self._drop_geotransform(group)
        del group["x"]
        del group["y"]
        with pytest.raises(KeyError):
            read_geobox(group)


@requires_zarr
class TestFinalizeZarrMetadata:
    """Tests for finalize_zarr_metadata (shared Dataset/collection finalizer)."""

    def test_writes_attrs_geobox_and_consolidates(self, tmp_path):
        """Root + data attrs, the geobox, and consolidated metadata are written.

        Test scenario:
            Given a group with a ``data`` array, finalize_zarr_metadata sets the
            supplied root/data attrs, writes the spatial_ref + x/y geobox, and
            consolidates — so ``.zmetadata`` exists and the arrays are present.
        """
        from pyramids.dataset.ops._geobox_zarr import finalize_zarr_metadata

        store = str(tmp_path / "fin.zarr")
        group = zarr.open_group(store, mode="w")
        group.create_array("data", data=np.zeros((1, 4, 5), dtype="float32"))
        finalize_zarr_metadata(
            store,
            root_attrs={"pyramids_zarr_version": "2", "time_length": 1},
            data_attrs={"dtype": "float32", "epsg": 4326},
            epsg=4326,
            geotransform=_GT,
            crs_wkt=_WKT_4326,
            rows=4,
            cols=5,
            dims=["band", "y", "x"],
        )
        reopened = zarr.open_group(store, mode="r")
        assert reopened.attrs["pyramids_zarr_version"] == "2", "root attr missing"
        assert reopened["data"].attrs["dtype"] == "float32", "data attr missing"
        assert {"spatial_ref", "x", "y"} <= set(reopened.array_keys()), "geobox missing"
        assert reopened["data"].attrs["grid_mapping"] == GRID_MAPPING_VAR


@requires_zarr
class TestForeignGeoZarr:
    """read_geobox / detect_data_var handle non-pyramids GeoZarr stores (FR-8)."""

    def _foreign_group(self, tmp_path, *, with_geotransform=False):
        store = str(tmp_path / "foreign.zarr")
        group = zarr.open_group(store, mode="w")
        elev = group.create_array(
            "elevation", data=np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        )
        elev.attrs.update(
            {"_ARRAY_DIMENSIONS": ["band", "y", "x"], "grid_mapping": "spatial_ref"}
        )
        sr = group.create_array(GRID_MAPPING_VAR, data=np.array(4326))
        sr_attrs = {"crs_wkt": _WKT_4326, "epsg": 4326}
        if with_geotransform:
            sr_attrs["GeoTransform"] = "0.0 1.0 0.0 3.0 0.0 -1.0"
        sr.attrs.update(sr_attrs)
        gx = group.create_array("x", data=np.array([0.5, 1.5, 2.5, 3.5]))
        gx.attrs["_ARRAY_DIMENSIONS"] = ["x"]
        gy = group.create_array("y", data=np.array([2.5, 1.5, 0.5]))
        gy.attrs["_ARRAY_DIMENSIONS"] = ["y"]
        return group

    def test_detect_data_var_by_grid_mapping(self, tmp_path):
        """detect_data_var picks the array carrying a grid_mapping attr.

        Test scenario:
            A store with no ``data`` array but an ``elevation`` array tagged
            ``grid_mapping`` resolves to ``"elevation"``.
        """
        group = self._foreign_group(tmp_path)
        assert detect_data_var(group) == "elevation", "wrong data var detected"

    def test_read_geobox_follows_grid_mapping(self, tmp_path):
        """CRS/epsg come from the grid_mapping-referenced var (FR-8).

        Test scenario:
            A foreign store with the CRS in a ``spatial_ref`` coord (referenced
            via the data var's ``grid_mapping``) and a stored GeoTransform reads
            without warning.
        """
        group = self._foreign_group(tmp_path, with_geotransform=True)
        result = read_geobox(group)
        assert result["legacy"] is False, "should not be legacy"
        assert result["epsg"] == 4326, f"epsg {result['epsg']}"
        np.testing.assert_allclose(
            result["geotransform"], (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
        )

    def test_read_geobox_derives_transform_from_xy(self, tmp_path):
        """A foreign store without GeoTransform derives it from x/y (FR-8).

        Test scenario:
            No ``GeoTransform`` attr; read_geobox reconstructs the transform from
            the pixel-centre x/y coords.
        """
        group = self._foreign_group(tmp_path, with_geotransform=False)
        result = read_geobox(group)
        np.testing.assert_allclose(
            result["geotransform"], (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
        )
