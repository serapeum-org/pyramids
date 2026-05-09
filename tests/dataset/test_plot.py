from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal
from pandas import DataFrame

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.engines import Analysis, Bands

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config


class TestPlotDataSet:
    Config.set_matplotlib_backend("agg")

    @pytest.mark.plot
    def test_single_band(
        self,
        src: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        dataset = Dataset(src)
        array_glyph = dataset.plot(band=0)
        assert isinstance(array_glyph, ArrayGlyph)

    @pytest.mark.plot
    def test_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        dataset = Dataset(sentinel_raster)
        array_glyph = dataset.plot(rgb=[3, 2, 1])
        assert isinstance(array_glyph, ArrayGlyph)

    @pytest.mark.plot
    def test_multi_band_overviews(
        self,
        era5_image_internal_overviews_read_only_true: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        dataset = Dataset(era5_image_internal_overviews_read_only_true)
        array_glyph = dataset.plot(band=0, overview=True, overview_index=0)

        assert isinstance(array_glyph, ArrayGlyph)

    @pytest.mark.plot
    def test_basemap_true_calls_add_basemap(self, src: Dataset):
        """Test that basemap=True calls add_basemap with correct args."""
        dataset = Dataset(src)
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            dataset.plot(band=0, basemap=True)
            mock_add.assert_called_once()
            call_kwargs = mock_add.call_args[1]
            assert call_kwargs["crs"] == dataset.epsg

    @pytest.mark.plot
    def test_basemap_string_passes_source(self, src: Dataset):
        """Test that basemap='CartoDB.Positron' passes source."""
        dataset = Dataset(src)
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            dataset.plot(band=0, basemap="CartoDB.Positron")
            call_kwargs = mock_add.call_args[1]
            assert call_kwargs["source"] == "CartoDB.Positron"

    @pytest.mark.plot
    def test_basemap_false_skips(self, src: Dataset):
        """Test that basemap=False does not call add_basemap."""
        dataset = Dataset(src)
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            dataset.plot(band=0, basemap=False)
            mock_add.assert_not_called()


class TestPlotDatasetCollection:
    @pytest.mark.plot
    def test_geotiff(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        from cleopatra.array_glyph import ArrayGlyph

        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        cleo = cube.plot()
        assert isinstance(cleo, ArrayGlyph)


class TestColorTable:

    @pytest.mark.plot
    def test_generated_data(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(1, 3, size=(2, 5, 5))
        top_left_corner = (0, 0)
        cell_size = 0.05
        dataset = Dataset.create_from_array(
            arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
        )

        # without alpha
        color_table = pd.DataFrame(
            {
                "band": [1, 1, 1, 2, 2, 2],
                "values": [1, 2, 3, 1, 2, 3],
                "color": [
                    "#709959",
                    "#F2EEA2",
                    "#F2CE85",
                    "#C28C7C",
                    "#D6C19C",
                    "#D6C19C",
                ],
            }
        )
        dataset.color_table = color_table
        retrieved_color_table = dataset.color_table
        assert all(
            ["band", "values", "red", "green", "blue", "alpha"]
            == retrieved_color_table.columns
        )

    @pytest.mark.plot
    def test_get_color_table(self, src_with_color_table: Dataset):
        dataset = Dataset(src_with_color_table)
        df = dataset.bands._get_color_table()
        assert isinstance(df, DataFrame)
        assert all(df.columns == ["band", "values", "red", "green", "blue", "alpha"])
        assert all(df.band == 1)
        # test the color_table property
        df = dataset.color_table
        assert isinstance(df, DataFrame)
        assert all(df.columns == ["band", "values", "red", "green", "blue", "alpha"])
        assert all(df.band == 1)

    @pytest.mark.plot
    def test_set_color_table(self, src_without_color_table: Dataset):
        color_hex = ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C"]
        values = [1, 3, 5, 7, 9]
        df = pd.DataFrame(columns=["band", "values", "color"])
        df.loc[:, "values"] = values
        df.loc[:, "band"] = 1
        df.loc[:, "color"] = color_hex

        dataset = Dataset(src_without_color_table)
        dataset.bands._set_color_table(df, overwrite=True)

        color_table = dataset.raster.GetRasterBand(1).GetColorTable()
        assert color_table is not None, "the color table should not be None"
        assert color_table.GetCount() == 10, "the color table should have 5 colors"
        colors = [color_table.GetColorEntry(i) for i in range(color_table.GetCount())]
        assert colors == [
            (0, 0, 0, 0),
            (112, 153, 89, 255),
            (0, 0, 0, 0),
            (242, 238, 162, 255),
            (0, 0, 0, 0),
            (242, 206, 133, 255),
            (0, 0, 0, 0),
            (194, 140, 124, 255),
            (0, 0, 0, 0),
            (214, 193, 156, 255),
        ]
        # test the color_table property
        dataset.color_table = df


class TestColorRelief:
    color_hex = ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C"]
    values = [1, 3, 5, 7, 9]
    df = pd.DataFrame(columns=["values", "color"])
    df.loc[:, "values"] = values
    df.loc[:, "color"] = color_hex

    @pytest.mark.plot
    def test_process_color_table(self):

        color_table = Analysis._process_color_table(self.df)
        assert isinstance(color_table, DataFrame)
        assert all(color_table.columns == ["values", "red", "green", "blue", "alpha"])
