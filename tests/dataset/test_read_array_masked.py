"""Tests for `read_array(masked=True)` — MaskedArray reads honouring nodata and mask bands.

Covers the `IO.read_array` masked path (nodata comparison, NaN nodata, GDAL
mask bands, multi-band stacking, windowed reads), the unchanged default
behaviour, the dask guard, and the `NetCDF.read_array` threading.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset
from pyramids.dataset.window import Window
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def nodata_dataset() -> Dataset:
    """A 2x2 float32 dataset with one -9999 nodata cell at (0, 1).

    Returns:
        Dataset: Single-band in-memory dataset, nodata -9999.
    """
    arr = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 2), cell_size=1.0, epsg=4326, no_data_value=-9999.0
    )


@pytest.fixture(scope="function")
def mask_band_dataset(tmp_path) -> Dataset:
    """A GTiff with a PER_DATASET internal mask band masking cell (0, 1).

    The dataset has no nodata marker — the only invalidity signal is the
    GDAL mask band, exercising the flags-based branch.

    Returns:
        Dataset: Single-band dataset whose mask band zeroes one cell.
    """
    path = str(tmp_path / "masked.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, 2, 2, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 1, 0, 2, 0, -1))
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    ds.SetProjection(sr.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.array([[1, 2], [3, 4]], dtype="float32"))
    ds.CreateMaskBand(gdal.GMF_PER_DATASET)
    ds.GetRasterBand(1).GetMaskBand().WriteArray(
        np.array([[255, 0], [255, 255]], dtype="uint8")
    )
    ds.FlushCache()
    ds = None
    return Dataset.read_file(path)


class TestMaskedReads:
    """Tests for read_array(masked=True) on Dataset."""

    def test_nodata_cells_are_masked(self, nodata_dataset):
        """Cells equal to the nodata marker are masked; others are not.

        Test scenario:
            One -9999 cell -> mask count 1; `filled(0)` replaces it with 0
            while valid cells keep their values.
        """
        result = nodata_dataset.read_array(band=0, masked=True)
        assert isinstance(result, np.ma.MaskedArray), f"got {type(result).__name__}"
        assert result.mask.sum() == 1, f"expected 1 masked cell, got {result.mask.sum()}"
        assert result.mask[0, 1], "the -9999 cell must be the masked one"
        filled = result.filled(0)
        assert filled[0, 1] == pytest.approx(0.0), "filled() must replace the masked cell"
        assert filled[1, 1] == pytest.approx(4.0), "valid cells must survive filled()"

    def test_valid_pixel_near_large_sentinel_not_masked(self):
        """A valid float pixel close to a large sentinel is not masked (M1).

        Test scenario:
            With the default fuzzy ``is_no_data`` tolerance (rtol=0.001) a valid
            ``-9990`` pixel is within 0.1% of a ``-9999`` marker and would be
            wrongly masked. Only the exact ``-9999`` cell may be masked.
        """
        arr = np.array([[-9999.0, -9990.0], [-9000.0, 1.0]], dtype="float32")
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 2), cell_size=1.0, epsg=4326, no_data_value=-9999.0
        )
        result = ds.read_array(band=0, masked=True)
        assert result.mask[0, 0], "the exact -9999 cell must be masked"
        assert not result.mask[0, 1], "a valid -9990 pixel must not be masked"
        assert result.mask.sum() == 1, f"only one cell may be masked, got {result.mask.sum()}"

    def test_integer_band_uses_exact_nodata_equality(self):
        """Integer bands mask only the exact marker, never near values (M1).

        Test scenario:
            An int16 band with a ``-100`` marker: ``-100`` is masked but the
            adjacent ``-99`` (within 1% of the marker) is not.
        """
        arr = np.array([[-100, -99], [0, 5]], dtype="int16")
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 2), cell_size=1.0, epsg=4326, no_data_value=-100
        )
        result = ds.read_array(band=0, masked=True)
        assert result.mask[0, 0], "the exact -100 cell must be masked"
        assert not result.mask[0, 1], "a valid -99 pixel must not be masked"
        assert result.mask.sum() == 1, f"only one cell may be masked, got {result.mask.sum()}"

    def test_nan_nodata_masks_nan_cells(self):
        """A NaN nodata marker masks the NaN cells (NaN-aware comparison).

        Test scenario:
            `value == nan` is always False, so the implementation must use
            isnan for float NaN nodata.
        """
        arr = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype="float32")
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 2), cell_size=1.0, epsg=4326, no_data_value=np.nan
        )
        result = ds.read_array(band=0, masked=True)
        assert result.mask.sum() == 1, f"NaN cell not masked: {result.mask}"
        assert result.mask[0, 0], "the NaN cell must be the masked one"

    def test_multi_band_per_band_masks(self):
        """An all-bands read stacks a per-band mask.

        Test scenario:
            Band 0 has one nodata cell, band 1 none — the 3-D mask reflects
            each band independently.
        """
        band0 = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
        band1 = np.full((2, 2), 7.0, dtype="float32")
        ds = Dataset.create_from_array(
            np.stack([band0, band1]),
            top_left_corner=(0, 2), cell_size=1.0, epsg=4326, no_data_value=-9999.0,
        )
        result = ds.read_array(masked=True)
        assert result.shape == (2, 2, 2), f"unexpected shape {result.shape}"
        assert result.mask[0].sum() == 1, "band 0 must have one masked cell"
        assert result.mask[1].sum() == 0, "band 1 must have no masked cells"

    def test_gdal_mask_band_is_honoured(self, mask_band_dataset):
        """A PER_DATASET internal mask band masks cells without any nodata.

        Test scenario:
            The only invalidity signal is the mask band (flags branch); the
            zeroed mask cell is masked in the result.
        """
        result = mask_band_dataset.read_array(band=0, masked=True)
        assert result.mask.sum() == 1, f"mask band ignored: {result.mask}"
        assert result[0, 1] is np.ma.masked, "cell (0,1) must be masked"

    def test_windowed_read_masks_by_nodata(self, nodata_dataset):
        """A windowed masked read applies the nodata mask to the window.

        Test scenario:
            Window covering the top row contains the nodata cell; the mask
            aligns with the window shape.
        """
        result = nodata_dataset.read_array(band=0, window=[0, 0, 2, 1], masked=True)
        assert result.shape == (1, 2), f"unexpected window shape {result.shape}"
        assert result.mask.sum() == 1, "window must contain one masked cell"

    def test_windowed_read_honours_mask_band(self, mask_band_dataset):
        """A windowed masked read slices the GDAL mask band to the window.

        Test scenario:
            The mask band zeroes cell (0, 1); a window covering the top row
            must mask exactly that cell, proving the mask band is read with
            the same pixel offsets as the data.
        """
        result = mask_band_dataset.read_array(band=0, window=[0, 0, 2, 1], masked=True)
        assert result.shape == (1, 2), f"unexpected window shape {result.shape}"
        assert result.mask.sum() == 1, f"mask band ignored in window: {result.mask}"
        assert result.mask[0, 1], "the mask-band-zeroed cell must be masked"

    def test_window_object_masked_read_honours_mask_band(self, mask_band_dataset):
        """A ``Window`` object composes with ``masked=True`` on a mask-band raster.

        Test scenario:
            The same top-row window as the list form, but expressed as a
            ``Window`` object. Before normalization in ``_to_masked`` this
            raised ``TypeError: 'Window' object is not subscriptable`` because
            ``_band_mask`` slices the mask band with ``window[0..3]``.
        """
        result = mask_band_dataset.read_array(
            band=0, window=Window(0, 0, 2, 1), masked=True
        )
        assert result.shape == (1, 2), f"unexpected window shape {result.shape}"
        assert result.mask.sum() == 1, f"mask band ignored in Window read: {result.mask}"
        assert result.mask[0, 1], "the mask-band-zeroed cell must be masked"

    def test_bbox_masked_read(self):
        """A bbox-driven masked read masks nodata within the resolved window.

        Test scenario:
            On a 4x4 raster the bbox resolves to the central 2x2 block,
            which contains one -9999 cell; the geometry window resolves to
            pixel offsets and the mask aligns with the returned block.
        """
        arr = np.full((4, 4), 5.0, dtype="float32")
        arr[1, 1] = -9999.0
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 4), cell_size=1.0, epsg=4326, no_data_value=-9999.0
        )
        result = ds.read_array(band=0, bbox=(1.0, 1.0, 3.0, 3.0), masked=True)
        assert isinstance(result, np.ma.MaskedArray), f"got {type(result).__name__}"
        assert result.shape == (2, 2), f"unexpected bbox shape {result.shape}"
        assert result.mask.sum() == 1, f"bbox window must mask the -9999 cell: {result.mask}"
        masked_at = tuple(np.argwhere(result.mask)[0])
        assert result.data[masked_at] == pytest.approx(-9999.0), (
            "the mask must sit on the -9999 cell of the returned block"
        )

    def test_no_nodata_marker_leaves_nan_unmasked(self, tmp_path):
        """A band without a nodata marker masks nothing, even valid NaNs.

        Test scenario:
            No nodata, no mask band — a NaN cell is ordinary data and the
            mask must be all-False (is_no_data's None-as-NaN sentinel must
            not be applied to undeclared bands).
        """
        path = str(tmp_path / "no_nodata.tif")
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(path, 2, 2, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((0, 1, 0, 2, 0, -1))
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        ds.SetProjection(sr.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(
            np.array([[np.nan, 2.0], [3.0, 4.0]], dtype="float32")
        )
        ds.FlushCache()
        ds = None
        result = Dataset.read_file(path).read_array(band=0, masked=True)
        assert isinstance(result, np.ma.MaskedArray), f"got {type(result).__name__}"
        assert result.mask.sum() == 0, f"nothing must be masked: {result.mask}"

    def test_float_precision_nodata_is_masked(self):
        """A nodata value that is not exactly representable still masks.

        Test scenario:
            nodata 0.1 stored in a float32 band differs from the python
            double 0.1; the tolerance-based comparison (is_no_data) must
            still mask the cell where an exact == would miss it.
        """
        arr = np.array([[0.1, 2.0], [3.0, 4.0]], dtype="float32")
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 2), cell_size=1.0, epsg=4326, no_data_value=0.1
        )
        result = ds.read_array(band=0, masked=True)
        assert result.mask[0, 0], "float32-precision nodata cell must be masked"
        assert result.mask.sum() == 1, f"expected 1 masked cell, got {result.mask.sum()}"

    def test_default_returns_plain_ndarray(self, nodata_dataset):
        """masked=False (default) keeps the historical plain-ndarray contract.

        Test scenario:
            No MaskedArray unless explicitly requested.
        """
        result = nodata_dataset.read_array(band=0)
        assert type(result) is np.ndarray, f"default changed: {type(result).__name__}"

    def test_chunks_with_masked_raises(self, nodata_dataset):
        """masked=True with chunks= raises NotImplementedError.

        Test scenario:
            Lazy masked reads are explicitly unsupported in v1.
        """
        with pytest.raises(NotImplementedError, match="masked=True"):
            nodata_dataset.read_array(band=0, chunks=2, masked=True)


class TestNetCDFMaskedReads:
    """Tests for the masked= threading through NetCDF.read_array."""

    @pytest.fixture
    def nc_subset(self) -> NetCDF:
        """A single-variable NetCDF subset with one -9999 cell.

        Returns:
            NetCDF: The `t` variable subset of an in-memory container.
        """
        arr = np.array([[[1.0, -9999.0], [3.0, 4.0]]], dtype="float32")
        nc = NetCDF.create_from_array(
            arr, top_left_corner=(0, 2), cell_size=1.0, epsg=4326,
            variable_name="t", no_data_value=-9999.0,
        )
        return nc.get_variable("t")

    def test_subset_masked_read(self, nc_subset):
        """A variable subset honours masked=True through the super() path.

        Test scenario:
            The nodata cell is masked exactly as on a plain Dataset.
        """
        result = nc_subset.read_array(masked=True)
        assert isinstance(result, np.ma.MaskedArray), f"got {type(result).__name__}"
        assert result.mask.sum() == 1, f"expected 1 masked cell, got {result.mask.sum()}"

    def test_lazy_masked_raises(self, nc_subset):
        """The NetCDF lazy path rejects masked=True explicitly.

        Test scenario:
            chunks= + masked= raises before any dask graph is built.
        """
        with pytest.raises(NotImplementedError, match="masked=True"):
            nc_subset.read_array(chunks=2, masked=True)

    def test_unpack_preserves_mask(self, nc_subset):
        """CF unpack scaling preserves the mask built from raw values.

        Test scenario:
            With scale/offset attributes faked on the subset, masked +
            unpack returns a MaskedArray whose mask matches the raw
            no-data cells and whose valid values are scaled.
        """
        nc_subset._scale = 2.0
        nc_subset._offset = 1.0
        result = nc_subset.read_array(masked=True, unpack=True)
        assert isinstance(result, np.ma.MaskedArray), "unpack dropped the mask wrapper"
        assert result.mask.sum() == 1, f"mask lost through unpack: {result.mask}"
        assert result[0, 0] == pytest.approx(1.0 * 2.0 + 1.0), (
            f"valid cell not scaled: {result[0, 0]}"
        )
