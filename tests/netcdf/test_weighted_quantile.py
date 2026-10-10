"""NC2 — the area/weight quantile added to ``NetCDF.weighted``."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf import GeoReference, NetCDF

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)


def _single_band(values: np.ndarray) -> NetCDF:
    """A one-band raster of ``values`` (shape ``(rows, cols)``)."""
    rows, cols = values.shape
    return NetCDF.from_array(
        values.reshape(1, rows, cols).astype("float64"),
        geo_ref=GEO,
        variable_name="v",
        no_data_value=None,
    ).get_variable("v")


class TestWeightedQuantile:
    def test_equal_weights_match_hazen_median(self):
        var = _single_band(np.array([[1.0, 2.0], [3.0, 4.0]]))
        out = var.weighted(np.ones((2, 2)), how="quantile", q=0.5)
        got = float(np.asarray(out.read_array()).ravel()[0])
        assert got == pytest.approx(
            np.quantile([1.0, 2.0, 3.0, 4.0], 0.5, method="hazen")
        )

    def test_endpoints(self):
        var = _single_band(np.array([[1.0, 2.0], [3.0, 4.0]]))
        low = var.weighted(np.ones((2, 2)), how="quantile", q=0.0)
        high = var.weighted(np.ones((2, 2)), how="quantile", q=1.0)
        assert float(np.asarray(low.read_array()).ravel()[0]) == pytest.approx(1.0)
        assert float(np.asarray(high.read_array()).ravel()[0]) == pytest.approx(4.0)

    def test_quantile_requires_q(self):
        var = _single_band(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match="quantile"):
            var.weighted(np.ones((2, 2)), how="quantile")

    def test_q_out_of_range_rejected_early(self):
        var = _single_band(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match=r"q in \[0, 1\]"):
            var.weighted(np.ones((2, 2)), how="quantile", q=1.5)

    def test_bool_q_rejected(self):
        var = _single_band(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match=r"q in \[0, 1\]"):
            var.weighted(np.ones((2, 2)), how="quantile", q=True)

    def test_q_only_with_quantile(self):
        var = _single_band(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match="quantile"):
            var.weighted(np.ones((2, 2)), how="mean", q=0.5)
