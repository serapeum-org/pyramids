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
from pyramids.netcdf import GeoReference, NetCDF

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def nodata_dataset() -> Dataset:
    """A 2x2 float32 dataset with one -9999 nodata cell at (0, 1).

    Returns:
        Dataset: Single-band in-memory dataset, nodata -9999.
    """
    arr = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
    return Dataset.from_array(
        arr,
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
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
        assert result.mask.sum() == 1, (
            f"expected 1 masked cell, got {result.mask.sum()}"
        )
        assert result.mask[0, 1], "the -9999 cell must be the masked one"
        filled = result.filled(0)
        assert filled[0, 1] == pytest.approx(0.0), (
            "filled() must replace the masked cell"
        )
        assert filled[1, 1] == pytest.approx(4.0), "valid cells must survive filled()"

    def test_valid_pixel_near_large_sentinel_not_masked(self):
        """A valid float pixel close to a large sentinel is not masked (M1).

        Test scenario:
            With the default fuzzy ``is_no_data`` tolerance (rtol=0.001) a valid
            ``-9990`` pixel is within 0.1% of a ``-9999`` marker and would be
            wrongly masked. Only the exact ``-9999`` cell may be masked.
        """
        arr = np.array([[-9999.0, -9990.0], [-9000.0, 1.0]], dtype="float32")
        ds = Dataset.from_array(
            arr,
            no_data_value=-9999.0,
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
        )
        result = ds.read_array(band=0, masked=True)
        assert result.mask[0, 0], "the exact -9999 cell must be masked"
        assert not result.mask[0, 1], "a valid -9990 pixel must not be masked"
        assert result.mask.sum() == 1, (
            f"only one cell may be masked, got {result.mask.sum()}"
        )

    def test_integer_band_uses_exact_nodata_equality(self):
        """Integer bands mask only the exact marker, never near values (M1).

        Test scenario:
            An int16 band with a ``-100`` marker: ``-100`` is masked but the
            adjacent ``-99`` (within 1% of the marker) is not.
        """
        arr = np.array([[-100, -99], [0, 5]], dtype="int16")
        ds = Dataset.from_array(
            arr,
            no_data_value=-100,
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
        )
        result = ds.read_array(band=0, masked=True)
        assert result.mask[0, 0], "the exact -100 cell must be masked"
        assert not result.mask[0, 1], "a valid -99 pixel must not be masked"
        assert result.mask.sum() == 1, (
            f"only one cell may be masked, got {result.mask.sum()}"
        )

    def test_nan_nodata_masks_nan_cells(self):
        """A NaN nodata marker masks the NaN cells (NaN-aware comparison).

        Test scenario:
            `value == nan` is always False, so the implementation must use
            isnan for float NaN nodata.
        """
        arr = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype="float32")
        ds = Dataset.from_array(
            arr,
            no_data_value=np.nan,
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
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
        ds = Dataset.from_array(
            np.stack([band0, band1]),
            no_data_value=-9999.0,
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
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
        assert result.mask.sum() == 1, (
            f"mask band ignored in Window read: {result.mask}"
        )
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
        ds = Dataset.from_array(
            arr,
            no_data_value=-9999.0,
            geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
        )
        result = ds.read_array(band=0, bbox=(1.0, 1.0, 3.0, 3.0), masked=True)
        assert isinstance(result, np.ma.MaskedArray), f"got {type(result).__name__}"
        assert result.shape == (2, 2), f"unexpected bbox shape {result.shape}"
        assert result.mask.sum() == 1, (
            f"bbox window must mask the -9999 cell: {result.mask}"
        )
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
        ds = Dataset.from_array(
            arr,
            no_data_value=0.1,
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
        )
        result = ds.read_array(band=0, masked=True)
        assert result.mask[0, 0], "float32-precision nodata cell must be masked"
        assert result.mask.sum() == 1, (
            f"expected 1 masked cell, got {result.mask.sum()}"
        )

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


@pytest.fixture(scope="function")
def ramp8_float() -> Dataset:
    """An 8x8 float32 ramp (value == row*8 + col), nodata -9999 in two corners.

    Returns:
        Dataset: Single-band in-memory dataset large enough to decimate.
    """
    arr = np.arange(64, dtype="float32").reshape(8, 8)
    arr[0, 0] = -9999.0
    arr[7, 7] = -9999.0
    return Dataset.from_array(
        arr,
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    )


@pytest.fixture(scope="function")
def mask_band_8x8(tmp_path) -> Dataset:
    """An 8x8 GTiff whose PER_DATASET mask band zeroes the top two rows.

    The band carries no nodata marker, so the mask band is the only
    invalidity signal — exercising the decimated mask-band read.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        Dataset: Single-band dataset with a mask band over the top two rows.
    """
    path = str(tmp_path / "masked8.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, 8, 8, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 1, 0, 8, 0, -1))
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    ds.SetProjection(sr.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.arange(64, dtype="float32").reshape(8, 8))
    ds.CreateMaskBand(gdal.GMF_PER_DATASET)
    mask = np.full((8, 8), 255, dtype="uint8")
    mask[:2, :] = 0
    ds.GetRasterBand(1).GetMaskBand().WriteArray(mask)
    ds.FlushCache()
    ds = None
    return Dataset.read_file(path)


class TestMaskedDecimatedReads:
    """read_array(out_shape=..., masked=True) — masked decimated reads (#1156)."""

    def test_mask_equals_the_decimated_nodata_cells(self, ramp8_float):
        """A nearest decimated read masks exactly the decimated no-data cells.

        Test scenario:
            Decimating 8x8 -> 4x4 with nearest samples the raster's own pixels,
            so a sampled no-data cell stays the sentinel and the mask equals the
            decimated stored array compared to the marker. Restricted to nearest
            because a blending resampler absorbs this fixture's scattered no-data
            (covered by ``test_average_masks_only_a_fully_no_data_block``).
        """
        masked = ramp8_float.read_array(
            out_shape=(4, 4), masked=True, resampling="nearest"
        )
        stored = ramp8_float.read_array(
            out_shape=(4, 4), unpack=False, resampling="nearest"
        )
        assert isinstance(masked, np.ma.MaskedArray), f"got {type(masked).__name__}"
        assert masked.shape == (4, 4), f"unexpected shape {masked.shape}"
        assert masked.mask.sum() >= 1, "nearest must keep a sampled no-data cell masked"
        np.testing.assert_array_equal(
            masked.mask,
            stored == -9999.0,
            err_msg="mask must equal the decimated no-data cells",
        )

    @pytest.mark.parametrize("resampling", ["nearest", "average"])
    def test_average_masks_only_a_fully_no_data_block(self, resampling):
        """A blending resampler masks a fully-no-data block but absorbs a lone cell.

        Test scenario:
            GDAL drops no-data from an ``average``, so a 2x2 source block that is
            entirely no-data decimates to the sentinel (masked) while a lone
            no-data cell blends into a valid value (not masked). ``nearest`` is
            included as the contrast: it samples the raster's pixels, so it masks
            both the block cell and the sampled lone cell. The mask always equals
            the decimated read's sentinel cells.
        """
        arr = np.arange(64, dtype="float32").reshape(8, 8)
        arr[0:2, 0:2] = -9999.0
        arr[5, 5] = -9999.0
        ds = Dataset.from_array(
            arr,
            no_data_value=-9999.0,
            geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
        )
        masked = ds.read_array(out_shape=(4, 4), masked=True, resampling=resampling)
        stored = ds.read_array(out_shape=(4, 4), unpack=False, resampling=resampling)
        assert masked.mask[0, 0], "the fully-no-data block must be masked"
        np.testing.assert_array_equal(masked.mask, stored == -9999.0)
        if resampling == "average":
            assert masked.mask.sum() == 1, "average must absorb the lone no-data cell"

    def test_decimated_masked_read_over_a_sub_window(self):
        """A decimated masked read over a bbox sub-window masks that window's no-data.

        Test scenario:
            Decimating only the top-left quarter (which holds a fully-no-data 2x2
            block) to 2x2 forwards the resolved pixel window through to the mask
            band read, so the block cell is masked and the mask equals the
            decimated stored sentinel cells.
        """
        arr = np.arange(64, dtype="float32").reshape(8, 8)
        arr[0:2, 0:2] = -9999.0
        ds = Dataset.from_array(
            arr,
            no_data_value=-9999.0,
            geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
        )
        xmin, _, _, ymax = ds.bbox
        quarter = (xmin, ymax - 4.0, xmin + 4.0, ymax)
        masked = ds.read_array(
            bbox=quarter, out_shape=(2, 2), masked=True, resampling="nearest"
        )
        stored = ds.read_array(
            bbox=quarter, out_shape=(2, 2), unpack=False, resampling="nearest"
        )
        assert masked.shape == (2, 2), f"unexpected shape {masked.shape}"
        assert masked.mask.sum() >= 1, "the sub-window's no-data block must be masked"
        np.testing.assert_array_equal(masked.mask, stored == -9999.0)

    def test_integer_band_masks_by_exact_equality(self):
        """An integer band masks the decimated cells equal to the marker.

        Test scenario:
            An int16 band decimated with nearest keeps its -9999 marker; the
            mask is exact equality, with no fuzzy tolerance near the sentinel.
        """
        arr = np.arange(64, dtype="int16").reshape(8, 8)
        arr[0, 0] = -9999
        ds = Dataset.from_array(
            arr,
            no_data_value=-9999,
            geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
        )
        masked = ds.read_array(out_shape=(4, 4), masked=True, resampling="nearest")
        stored = ds.read_array(out_shape=(4, 4), unpack=False, resampling="nearest")
        np.testing.assert_array_equal(masked.mask, stored == -9999)

    def test_mask_band_is_decimated_to_the_output_shape(self, mask_band_8x8):
        """A GDAL mask band is decimated to the same buffer and lines up.

        Test scenario:
            The mask band zeroes the top two of eight rows and the band has no
            nodata marker; decimated 8x8 -> 4x4 (nearest) that is the top output
            row, so only it is masked.
        """
        masked = mask_band_8x8.read_array(
            out_shape=(4, 4), masked=True, resampling="nearest"
        )
        assert isinstance(masked, np.ma.MaskedArray), f"got {type(masked).__name__}"
        assert masked.mask[0].all(), "the decimated top row must be masked"
        assert not masked.mask[1:].any(), "rows below the mask band must be valid"

    def test_packed_band_masks_by_stored_value(self):
        """A packed band masks by the stored marker, then unpacks the data.

        Test scenario:
            An int16 band with scale 0.5 and a -9999 marker, decimated + masked,
            returns an unpacked float64 MaskedArray whose mask is the stored
            sentinel — the comparison runs before scale/offset apply.
        """
        arr = np.arange(64, dtype="int16").reshape(8, 8)
        arr[7, 7] = -9999
        ds = Dataset.from_array(
            arr,
            no_data_value=-9999,
            geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
        )
        ds.scale = [0.5]
        masked = ds.read_array(out_shape=(4, 4), masked=True, resampling="nearest")
        stored = ds.read_array(out_shape=(4, 4), unpack=False, resampling="nearest")
        assert masked.dtype == np.float64, f"packed read must unpack: {masked.dtype}"
        np.testing.assert_array_equal(masked.mask, stored == -9999)

    def test_all_bands_masked_stacks_a_per_band_mask(self):
        """An all-bands decimated masked read stacks a per-band mask.

        Test scenario:
            A two-band raster with a different no-data corner per band,
            decimated with band=None and masked, returns a 3-D MaskedArray whose
            mask equals the decimated stored values compared to the marker,
            band by band.
        """
        b0 = np.arange(64, dtype="float32").reshape(8, 8)
        b0[0, 0] = -9999.0
        b1 = np.arange(64, dtype="float32").reshape(8, 8) + 100.0
        b1[7, 7] = -9999.0
        ds = Dataset.from_array(
            np.stack([b0, b1]),
            no_data_value=-9999.0,
            geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
        )
        masked = ds.read_array(out_shape=(4, 4), masked=True, resampling="nearest")
        stored = ds.read_array(out_shape=(4, 4), unpack=False, resampling="nearest")
        assert masked.shape == (2, 4, 4), f"unexpected shape {masked.shape}"
        np.testing.assert_array_equal(masked.mask, stored == -9999.0)

    def test_without_masked_returns_plain_ndarray(self, ramp8_float):
        """out_shape without masked keeps the plain-ndarray contract.

        Test scenario:
            The decimated read is unchanged when masked is not requested.
        """
        result = ramp8_float.read_array(out_shape=(4, 4))
        assert type(result) is np.ndarray, f"masked leaked in: {type(result).__name__}"


class TestMaskedBoundlessReads:
    """read_array(window=..., boundless=True, masked=True) — masked padded reads (#1156)."""

    def test_masks_padding_and_invalid_pixels(self, ramp8_float):
        """Padding outside the raster and no-data inside it are both masked.

        Test scenario:
            A window running four columns off the right edge pads those columns
            (masked), and the in-raster -9999 corner cell is masked too, while a
            valid in-raster cell stays unmasked.
        """
        window = [6, 0, 6, 8]  # cols 6..11; raster has 8 cols -> 4 padding columns
        masked = ramp8_float.read_array(window=window, boundless=True, masked=True)
        assert isinstance(masked, np.ma.MaskedArray), f"got {type(masked).__name__}"
        assert masked.shape == (8, 6), f"unexpected shape {masked.shape}"
        assert masked.mask[:, 2:].all(), "the four out-of-raster columns must be masked"
        assert masked.mask[7, 1], "the in-raster no-data cell (7, 7) must be masked"
        assert not masked.mask[0, 0], "a valid in-raster cell must be unmasked"

    def test_mask_is_independent_of_fill_value(self, ramp8_float):
        """fill_value sets only the padding's data; the mask is the same either way.

        Test scenario:
            A boundless masked read with the default fill and with an explicit
            fill_value produce identical masks — fill_value chooses the padding's
            value, not what counts as padding — and the padding stays masked.
        """
        window = [6, 0, 6, 8]
        default_fill = ramp8_float.read_array(
            window=window, boundless=True, masked=True
        )
        explicit_fill = ramp8_float.read_array(
            window=window, boundless=True, masked=True, fill_value=123.0
        )
        np.testing.assert_array_equal(
            default_fill.mask,
            explicit_fill.mask,
            err_msg="the mask must not depend on fill_value",
        )
        assert explicit_fill.mask[:, 2:].all(), "the padding columns must stay masked"

    def test_window_fully_outside_is_all_masked(self, ramp8_float):
        """A window entirely off the raster masks every (all-padding) cell.

        Test scenario:
            No intersection with the raster -> the whole buffer is fill and
            every cell is masked.
        """
        masked = ramp8_float.read_array(
            window=[20, 20, 3, 3], boundless=True, masked=True
        )
        assert isinstance(masked, np.ma.MaskedArray), f"got {type(masked).__name__}"
        assert masked.mask.all(), "an all-padding window must be fully masked"

    def test_all_bands_masked_stacks_padding_and_per_band_invalid(self):
        """An all-bands boundless masked read stacks padding + per-band invalid.

        Test scenario:
            A two-band raster read through a window off the right edge with
            band=None returns a 3-D MaskedArray: the padding columns are masked
            in every band, band 0's own no-data cell is masked, and band 1 (no
            no-data cell in range) keeps its in-raster cells unmasked.
        """
        b0 = np.arange(64, dtype="float32").reshape(8, 8)
        b0[7, 7] = -9999.0
        b1 = np.arange(64, dtype="float32").reshape(8, 8) + 100.0
        ds = Dataset.from_array(
            np.stack([b0, b1]),
            no_data_value=-9999.0,
            geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
        )
        masked = ds.read_array(window=[6, 0, 6, 8], boundless=True, masked=True)
        assert masked.shape == (2, 8, 6), f"unexpected shape {masked.shape}"
        assert masked.mask[:, :, 2:].all(), (
            "padding columns must be masked in every band"
        )
        assert masked.mask[0, 7, 1], "band 0's in-raster no-data cell must be masked"
        assert not masked.mask[1, :, :2].any(), "band 1 has no in-raster no-data cell"

    def test_without_masked_returns_plain_ndarray(self, ramp8_float):
        """boundless without masked keeps the plain-ndarray contract.

        Test scenario:
            The padded read is unchanged when masked is not requested.
        """
        result = ramp8_float.read_array(window=[6, 0, 6, 8], boundless=True)
        assert type(result) is np.ndarray, f"masked leaked in: {type(result).__name__}"


class TestNetCDFMaskedReads:
    """Tests for the masked= threading through NetCDF.read_array."""

    @pytest.fixture
    def nc_subset(self) -> NetCDF:
        """A single-variable NetCDF subset with one -9999 cell.

        Returns:
            NetCDF: The `t` variable subset of an in-memory container.
        """
        arr = np.array([[[1.0, -9999.0], [3.0, 4.0]]], dtype="float32")
        nc = NetCDF.from_array(
            arr,
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
            variable_name="t",
            no_data_value=-9999.0,
        )
        return nc.get_variable("t")

    def test_subset_masked_read(self, nc_subset):
        """A variable subset honours masked=True through the super() path.

        Test scenario:
            The nodata cell is masked exactly as on a plain Dataset.
        """
        result = nc_subset.read_array(masked=True)
        assert isinstance(result, np.ma.MaskedArray), f"got {type(result).__name__}"
        assert result.mask.sum() == 1, (
            f"expected 1 masked cell, got {result.mask.sum()}"
        )

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
