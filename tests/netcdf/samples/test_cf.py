"""CF-convention support: CFInfo detection plus the cf helper functions.

Detection is exercised across the sample files; the pure helpers are unit-tested directly.
"""

import numpy as np
import pytest
from osgeo import osr

from pyramids.netcdf import NetCDF
from pyramids.netcdf import cf

pytestmark = pytest.mark.core


def test_parse_conventions_variants():
    """``parse_conventions`` splits CF versions and recognizes COARDS / empty input."""
    assert cf.parse_conventions("CF-1.6") == {"CF": "1.6"}
    assert "COARDS" in cf.parse_conventions("COARDS")
    assert cf.parse_conventions(None) == {}


def test_cfinfo_matches_declared_convention(sample_name, sample, caps):
    """CFInfo reports a CF version + data variables for CF files; COARDS files declare COARDS."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata()
        if caps.get("convention") == "cf":
            assert meta.cf is not None and meta.cf.cf_version, f"{sample_name}: missing CF version"
            assert meta.cf.data_variable_names, f"{sample_name}: no data variables classified"
        elif caps.get("convention") == "coards":
            assert "COARDS" in (meta.cf.conventions if meta.cf else {}), f"{sample_name}: not COARDS"
    finally:
        nc.close()


def test_tos_bounds_and_classification(sample):
    """The tos file classifies its data/coordinate/bounds variables and maps bounds to their coordinates."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1__y-asc.nc"))
    try:
        cf_info = nc.get_all_metadata().cf
        assert cf_info.classifications["tos"] == "data"
        assert cf_info.classifications["lat"] == "coordinate"
        assert cf_info.classifications["lat_bnds"] == "bounds"
        assert cf_info.bounds_map["lat_bnds"] == "lat"
    finally:
        nc.close()


def test_grid_mapping_round_trip():
    """``srs_to_grid_mapping`` -> ``grid_mapping_to_srs`` reconstructs a geographic CRS."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    name, params = cf.srs_to_grid_mapping(srs)
    assert name == "latitude_longitude"
    restored = cf.grid_mapping_to_srs(name, params)
    assert isinstance(restored, osr.SpatialReference)
    assert restored.IsGeographic()


def test_detect_axis_for_coordinate_names():
    """``detect_axis`` recognizes latitude/longitude/time coordinate roles."""
    assert cf.detect_axis("lat", {"standard_name": "latitude"}, "degrees_north") == "Y"
    assert cf.detect_axis("lon", {"standard_name": "longitude"}, "degrees_east") == "X"
    assert cf.detect_axis("time", {}, "days since 2000-01-01") == "T"


def test_parse_cell_methods():
    """``parse_cell_methods`` parses a CF cell_methods string into per-dimension entries."""
    parsed = cf.parse_cell_methods("time: mean area: maximum")
    assert isinstance(parsed, list) and parsed
    assert {"dimensions": "time", "method": "mean"} in parsed


def test_apply_valid_range_mask():
    """Values outside ``[valid_min, valid_max]`` are replaced by the fill value (NaN by default)."""
    arr = np.array([1.0, 2.0, 3.0, 100.0])
    masked = cf.apply_valid_range_mask(arr, valid_min=0.0, valid_max=10.0)
    assert np.isnan(masked[3]) and not np.isnan(masked[0])


def test_decode_flags():
    """``decode_flags`` maps a packed value to its flag meanings."""
    meanings = cf.decode_flags(1, flag_values=[0, 1, 2], flag_meanings=["land", "water", "ice"])
    assert "water" in meanings


def test_validate_cf_returns_list(sample):
    """``validate_cf`` returns a list of messages (warnings/errors) for a CF file without raising."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1__y-asc.nc"))
    try:
        meta = nc.get_all_metadata()
        report = cf.validate_cf(meta.global_attributes, meta.variables, meta.dimensions)
        assert isinstance(report, list)
    finally:
        nc.close()
