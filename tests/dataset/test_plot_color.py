"""Plot tests: colour tables, colour relief, and plot-band resolution."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal
from pandas import DataFrame

from pyramids.dataset import Dataset
from pyramids.dataset.engines import Analysis

pytestmark = pytest.mark.plot

_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config
Config.set_matplotlib_backend("agg")


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each plot test to bound memory.

    Plotting tests open figures via cleopatra/pyplot; without this teardown
    the suite accumulates them and matplotlib warns past 20 open figures.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


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


class TestResolvePlotBand:
    """Regression tests for the per-class band-resolution policy (PR-1 / D-0, D-1).

    These tests pin down where the RGB heuristic now lives (``Dataset.plot``) and
    that it only fires when there is genuine evidence the data is RGB imagery
    (i.e. at least one band carries a GDAL ``ColorInterpretation``). The generic
    ``Analysis.plot`` engine must no longer apply any band-resolution policy.

    See also:
        ``tests/dataset/test_plot_resolution.py::TestResolvePlotBandPolicy`` — the
        wider parametrised matrix over band counts / colour-interp configs / RGB
        overrides. This class holds the original PR-1 cases; the deliberate
        overlap exercises the same rules from a focused vs a tabular angle.
    """

    @pytest.mark.plot
    def test_multi_band_without_color_interpretation_defaults_to_band_zero(self):
        """D-1 regression: 3+ bands but no ``ColorInterpretation`` set must NOT
        be treated as RGB.

        A 4-band raster with ``GCI_Undefined`` on every band (e.g. a time stack
        masquerading as 4 GeoTIFF bands) must default to ``band=0``, not
        ``band=2`` (the legacy Sentinel-2 fallback head).
        """
        rng = np.random.default_rng(0)
        arr = rng.random((4, 10, 10)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        for i in range(dataset.band_count):
            assert dataset.bands._iloc(i).GetColorInterpretation() == gdal.GCI_Undefined

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=None, rgb=None)
        assert resolved_band == 0
        assert resolved_rgb is None

    @pytest.mark.plot
    def test_multi_band_with_color_interpretation_resolves_rgb(self):
        """Positive case: bands tagged ``red``/``green``/``blue`` are resolved
        to their declared indices and the red band is returned.
        """
        rng = np.random.default_rng(1)
        arr = rng.random((3, 10, 10)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.band_color = {0: "red", 1: "green", 2: "blue"}

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=None, rgb=None)
        assert resolved_band == 0
        assert resolved_rgb == [0, 1, 2]

    @pytest.mark.plot
    def test_explicit_band_passes_through(self):
        """When ``band`` is supplied, the heuristic must not override it."""
        rng = np.random.default_rng(2)
        arr = rng.random((3, 10, 10)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.band_color = {0: "red", 1: "green", 2: "blue"}

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=1, rgb=None)
        assert resolved_band == 1
        assert resolved_rgb is None

    @pytest.mark.plot
    def test_single_band_defaults_to_zero(self):
        """``band_count < 3`` must default to band 0 regardless of colour tags."""
        rng = np.random.default_rng(3)
        arr = rng.random((10, 10)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=None, rgb=None)
        assert resolved_band == 0
        assert resolved_rgb is None

    @pytest.mark.plot
    def test_multi_band_without_color_interpretation_plot_call(self):
        """End-to-end regression for D-1 via the full :meth:`Dataset.plot` facade.

        Intercept the analysis engine and verify it is called with ``band=0`` —
        not ``band=2`` — when the multi-band raster has no ColorInterpretation.
        """
        rng = np.random.default_rng(4)
        arr = rng.random((4, 10, 10)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        with patch.object(type(dataset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "sentinel"
            result = dataset.plot()

        assert result == "sentinel"
        assert mock_plot.call_count == 1
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs["band"] == 0
        assert call_kwargs["rgb"] is None
