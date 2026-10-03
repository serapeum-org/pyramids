"""`to_dask_dataframe` — lazy sibling of `to_dataframe` (#1234).

The lazy frame is tidy (dimension coordinates as columns, flat index) because a dask dataframe
has no `MultiIndex`; promoting those columns back to the index and sorting reproduces the eager
`to_dataframe()` exactly — values, dimension labels and gaps alike.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from tests._marks import requires_dask

pytestmark = pytest.mark.netcdf_lazy

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
FIX = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-4d1__y-asc.nc"
)
MULTIVAR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
)


def _cube() -> NetCDF:
    """A two-step, 2x2 single-variable cube with a time dimension."""
    return NetCDF.from_array(
        np.arange(8.0).reshape(2, 2, 2),
        geo_ref=GEO,
        variable_name="t",
        dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
    )


def _as_eager(ddf, eager):
    """Compute a tidy lazy frame back to `eager`'s shape (its MultiIndex, sorted)."""
    return ddf.compute().set_index(list(eager.index.names)).sort_index()


class TestToDaskDataframe:
    """A tidy dask frame whose computed, re-indexed form equals `to_dataframe`."""

    @requires_dask
    def test_returns_a_dask_dataframe(self):
        """The return type is a `dask.dataframe.DataFrame`, not an eager pandas frame."""
        import dask.dataframe as dd

        assert isinstance(_cube().to_dask_dataframe(), dd.DataFrame)

    @requires_dask
    def test_tidy_columns_are_the_dimensions_plus_the_variable(self):
        """The columns are the dimension coordinates beside the value column."""
        assert sorted(_cube().to_dask_dataframe().columns) == ["t", "time", "x", "y"]

    @requires_dask
    def test_compute_reindexed_equals_to_dataframe_in_memory(self):
        """On an in-memory cube the computed, re-indexed frame equals `to_dataframe()`."""
        nc = _cube()
        eager = nc.to_dataframe().sort_index()
        lazy = _as_eager(nc.to_dask_dataframe(), eager)
        assert_frame_equal(lazy, eager, check_dtype=False)

    @requires_dask
    def test_compute_reindexed_equals_to_dataframe_multidim_fixture(self):
        """On a multi-variable, multi-dimension store the two agree row for row."""
        nc = NetCDF.read_file(str(FIX))
        eager = nc.to_dataframe().sort_index()
        lazy = _as_eager(nc.to_dask_dataframe(), eager)
        assert_frame_equal(lazy[eager.columns], eager, check_dtype=False)

    @requires_dask
    def test_dropna_matches_to_dataframe(self):
        """`dropna=True` drops the same all-missing rows the eager frame drops."""
        arr = np.where(
            np.arange(8.0).reshape(2, 2, 2) == 3.0,
            -9999.0,
            np.arange(8.0).reshape(2, 2, 2),
        )
        nc = NetCDF.from_array(
            arr,
            geo_ref=GEO,
            variable_name="t",
            no_data_value=-9999.0,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
        )
        eager = nc.to_dataframe(dropna=True).sort_index()
        lazy = _as_eager(nc.to_dask_dataframe(dropna=True), eager)
        assert_frame_equal(lazy, eager, check_dtype=False)

    @requires_dask
    def test_empty_selection_is_refused(self):
        """An empty `variables` sequence is refused, exactly as `to_dataframe` refuses it."""
        with pytest.raises(ValueError):
            _cube().to_dask_dataframe(variables=[])

    @requires_dask
    def test_variables_with_mismatched_band_dimensions_are_refused(self):
        """Two variables that do not share band dimensions cannot line up on one index."""
        nc = NetCDF.read_file(str(MULTIVAR))
        with pytest.raises(ValueError, match="share their band"):
            nc.to_dask_dataframe(variables=["area", "pr"])

    @requires_dask
    def test_file_backed_read_masks_the_no_data_sentinel(self):
        """A file-backed variable with a non-NaN fill value reads lazily with its gaps as NaN."""
        nc = NetCDF.read_file(str(MULTIVAR))
        eager = nc.to_dataframe(variables=["pr"]).sort_index()
        lazy = _as_eager(nc.to_dask_dataframe(variables=["pr"]), eager)
        assert_frame_equal(lazy[eager.columns], eager, check_dtype=False)

    @requires_dask
    def test_dimension_columns_do_not_materialise_the_full_index(self, monkeypatch):
        """The dimension columns are built lazily from per-dim axes, never the full MultiIndex — M2."""
        from pyramids.netcdf.engines import interop

        def _boom(_var):
            raise AssertionError("to_dask_dataframe must not expand the full MultiIndex eagerly")

        monkeypatch.setattr(interop, "_frame_index", _boom)
        ddf = NetCDF.read_file(str(MULTIVAR)).to_dask_dataframe(variables=["pr"])
        assert "pr" in ddf.columns, "the value column must still be present"
