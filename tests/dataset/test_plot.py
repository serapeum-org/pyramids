from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal
from pandas import DataFrame

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.engines import Analysis, Bands
from pyramids.netcdf.netcdf import NetCDF

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


class TestResolvePlotBand:
    """Regression tests for the per-class band-resolution policy (PR-1 / D-0, D-1).

    These tests pin down where the RGB heuristic now lives (``Dataset.plot``) and
    that it only fires when there is genuine evidence the data is RGB imagery
    (i.e. at least one band carries a GDAL ``ColorInterpretation``). The generic
    ``Analysis.plot`` engine must no longer apply any band-resolution policy.
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

        with patch.object(
            type(dataset.analysis), "plot", autospec=True
        ) as mock_plot:
            mock_plot.return_value = "sentinel"
            result = dataset.plot()

        assert result == "sentinel"
        assert mock_plot.call_count == 1
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs["band"] == 0
        assert call_kwargs["rgb"] is None


def _make_nc_subset_with_band_count(tmp_path, n_bands: int):
    """Build a NetCDF variable subset whose ``band_count`` is ``n_bands``.

    Uses the in-memory MEM driver path of :meth:`NetCDF.create_from_array` so
    each test gets a fresh fixture without disk churn.
    """
    rng = np.random.default_rng(42)
    arr = rng.random((n_bands, 5, 6)).astype("float32")
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(30.0, 0.5, 0, 35.0, 0, -0.5),
        variable_name="t2m",
        path=None,
        extra_dim_name="time",
        extra_dim_values=list(range(n_bands)),
    )
    return nc.get_variable("t2m"), arr


class TestNetCDFPlot:
    """Tests for the NetCDF override of ``plot`` (D-0 NetCDF surface fix).

    ``NetCDF.plot`` must:
    - Default to ``band=0`` even on multi-band variable subsets (no RGB heuristic).
    - Reject the satellite-imagery kwargs (``rgb``, ``surface_reflectance``,
      ``cutoff``) because NetCDF subsets are not Sentinel imagery.
    """

    @pytest.mark.plot
    def test_multi_band_subset_defaults_to_band_zero(self, tmp_path):
        """A 4-time-step variable subset must default to ``band=0``.

        The legacy RGB heuristic would have grabbed band 2 because
        ``band_count >= 3``. Verify the override bypasses ``super().plot()``
        entirely and forwards ``band=0`` to the engine.
        """
        nc_subset, _ = _make_nc_subset_with_band_count(tmp_path, n_bands=4)
        assert nc_subset.band_count == 4

        with patch.object(
            type(nc_subset.analysis), "plot", autospec=True
        ) as mock_plot:
            mock_plot.return_value = "sentinel"
            result = nc_subset.plot()

        assert result == "sentinel"
        assert mock_plot.call_count == 1
        assert mock_plot.call_args.kwargs["band"] == 0

    @pytest.mark.plot
    def test_rgb_kwarg_raises_type_error(self, tmp_path):
        """``rgb=`` is a Sentinel-imagery kwarg with no meaning on NetCDF."""
        nc_subset, _ = _make_nc_subset_with_band_count(tmp_path, n_bands=4)
        with pytest.raises(TypeError, match=r"rgb="):
            nc_subset.plot(rgb=[2, 1, 0])

    @pytest.mark.plot
    def test_surface_reflectance_kwarg_raises_type_error(self, tmp_path):
        """``surface_reflectance=`` is a Sentinel-imagery kwarg."""
        nc_subset, _ = _make_nc_subset_with_band_count(tmp_path, n_bands=4)
        with pytest.raises(TypeError, match=r"surface_reflectance="):
            nc_subset.plot(surface_reflectance=10000)

    @pytest.mark.plot
    def test_cutoff_kwarg_raises_type_error(self, tmp_path):
        """``cutoff=`` is a Sentinel-imagery kwarg."""
        nc_subset, _ = _make_nc_subset_with_band_count(tmp_path, n_bands=4)
        with pytest.raises(TypeError, match=r"cutoff="):
            nc_subset.plot(cutoff=[0.5])


class TestDatasetPlotFacade:
    """End-to-end facade tests for :meth:`Dataset.plot` after PR-1."""

    @pytest.mark.plot
    def test_returns_array_glyph(self):
        """:meth:`Dataset.plot` returns a cleopatra ``ArrayGlyph``.

        Test scenario:
            Calling the facade with a single-band dataset must return
            an instance of :class:`cleopatra.array_glyph.ArrayGlyph` so
            downstream callers can chain visual customisations.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        result = dataset.plot()
        assert isinstance(result, ArrayGlyph), (
            f"Expected ArrayGlyph, got {type(result).__name__}"
        )

    @pytest.mark.plot
    def test_two_consecutive_calls_return_independent_figures(self):
        """Two ``plot()`` calls on the same dataset must yield distinct figures.

        Test scenario:
            The facade must be a pure factory of ``ArrayGlyph``
            instances — no hidden state should leak between calls. Two
            consecutive invocations must return two distinct ``cleo.fig``
            references so downstream code can layer multiple plots.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        first = dataset.plot()
        second = dataset.plot()
        assert first is not second, "plot() must return a fresh ArrayGlyph each call"
        assert first.fig is not second.fig, (
            "Each call must own a distinct matplotlib Figure"
        )

    @pytest.mark.plot
    @pytest.mark.parametrize(
        "color_scale",
        [
            "linear",
            "power",
            "sym-lognorm",
        ],
    )
    def test_color_scale_string_aliases_work(self, color_scale):
        """M-2 docstring fix: cleopatra accepts ``color_scale`` string aliases.

        Args:
            color_scale: String alias for the colour-scale enum.

        Test scenario:
            The PR-1 docstring change documented the string aliases as
            valid ``color_scale`` values. Verify each alias is forwarded
            to cleopatra and yields an ``ArrayGlyph``.

        Notes:
            The integer codes (``1``-``5``) are also documented but are
            currently broken in the installed cleopatra release
            (``'int' object has no attribute 'lower'`` in
            ``cleopatra.glyph._update_color_norm``). When that
            upstream bug is fixed, extend the parametrize list to
            include the integer codes too.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        result = dataset.plot(color_scale=color_scale)
        assert isinstance(result, ArrayGlyph), (
            f"color_scale={color_scale!r} should return ArrayGlyph, "
            f"got {type(result).__name__}"
        )

    @pytest.mark.plot
    def test_facade_forwards_to_analysis_plot(self):
        """The facade must delegate to :meth:`Analysis.plot`, not duplicate logic.

        Test scenario:
            ``Dataset.plot`` is a thin facade. Mock the engine and
            verify a single call with the resolved positional and
            keyword args is made; the resolver should hand over
            ``band=0`` for a single-band raster.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        with patch.object(type(dataset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub-glyph"
            result = dataset.plot(figsize=(4, 4))

        assert result == "stub-glyph", f"Facade must return engine output, got {result!r}"
        assert mock_plot.call_count == 1
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs["band"] == 0, (
            f"Resolver should send band=0, got {call_kwargs.get('band')}"
        )
        assert call_kwargs["figsize"] == (4, 4), (
            f"Extra kwargs must propagate, got {call_kwargs.get('figsize')}"
        )


class TestDatasetPlotRgbOptions:
    """PR-4 / D-3 — `rgb_options=` group + DeprecationWarning on loose kwargs."""

    @pytest.mark.plot
    def test_rgb_options_group_works(self):
        """`rgb_options={"rgb": [...]}` works identically to the loose form."""
        rng = np.random.default_rng(11)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        with patch.object(type(dataset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub"
            dataset.plot(rgb_options={"rgb": [2, 1, 0]})
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs["rgb"] == [2, 1, 0]

    @pytest.mark.plot
    def test_rgb_options_carries_all_four_keys(self):
        """All four Sentinel kwargs flow through the `rgb_options=` group."""
        rng = np.random.default_rng(12)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        opts = {
            "rgb": [0, 1, 2],
            "surface_reflectance": 10000,
            "cutoff": [0.1, 0.9],
            "percentile": 2,
        }
        with patch.object(type(dataset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub"
            dataset.plot(rgb_options=opts)
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs["rgb"] == opts["rgb"]
        assert call_kwargs["surface_reflectance"] == opts["surface_reflectance"]
        assert call_kwargs["cutoff"] == opts["cutoff"]
        assert call_kwargs["percentile"] == opts["percentile"]

    @pytest.mark.plot
    def test_loose_rgb_emits_deprecation_warning(self):
        """Passing `rgb=` loose at the top level emits DeprecationWarning."""
        rng = np.random.default_rng(13)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        with patch.object(type(dataset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub"
            with pytest.warns(DeprecationWarning, match=r"rgb_options"):
                dataset.plot(rgb=[0, 1, 2])
        assert mock_plot.call_args.kwargs["rgb"] == [0, 1, 2]

    @pytest.mark.plot
    @pytest.mark.parametrize(
        "key,value",
        [
            ("surface_reflectance", 10000),
            ("cutoff", [0.1, 0.9]),
            ("percentile", 2),
        ],
    )
    def test_other_loose_kwargs_emit_deprecation_warning(self, key, value):
        """Each Sentinel-loose kwarg emits a DeprecationWarning individually.

        Args:
            key: One of ``"surface_reflectance"`` / ``"cutoff"`` /
                ``"percentile"``.
            value: Test value to pass in for that kwarg.
        """
        rng = np.random.default_rng(14)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        with patch.object(type(dataset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub"
            with pytest.warns(DeprecationWarning, match=r"rgb_options"):
                dataset.plot(**{key: value})
        assert mock_plot.call_args.kwargs[key] == value

    @pytest.mark.plot
    def test_rgb_options_unknown_key_raises(self):
        """Unknown keys in `rgb_options` are rejected with a clear error."""
        rng = np.random.default_rng(15)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        with pytest.raises(ValueError, match=r"Unknown keys"):
            dataset.plot(rgb_options={"bogus": True})

    @pytest.mark.plot
    def test_rgb_options_overrides_loose_kwarg(self):
        """When both forms are passed, `rgb_options` wins for the same key."""
        rng = np.random.default_rng(16)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        with patch.object(type(dataset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub"
            with pytest.warns(DeprecationWarning):
                dataset.plot(
                    rgb=[2, 1, 0],
                    rgb_options={"rgb": [0, 1, 2]},
                )
        assert mock_plot.call_args.kwargs["rgb"] == [0, 1, 2]


class TestAnalysisPlotEngine:
    """Tests for the post-PR-1 :meth:`Analysis.plot` engine contract.

    The engine is now band-agnostic: it requires a concrete ``band``
    integer and applies no resolution heuristic.
    """

    @pytest.mark.plot
    def test_explicit_band_renders_array_glyph(self):
        """Calling ``Analysis.plot(band=N)`` directly must work.

        Test scenario:
            Bypass the facade and hit the engine directly with an
            explicit band index — exercises the branch the PR-1 docs
            promise: the engine never resolves and is purely generic.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        result = dataset.analysis.plot(band=2)
        assert isinstance(result, ArrayGlyph), (
            f"Expected ArrayGlyph, got {type(result).__name__}"
        )

    @pytest.mark.plot
    def test_out_of_range_band_raises(self):
        """An out-of-range band must propagate the underlying error.

        Test scenario:
            ``read_array`` raises :class:`ValueError` (or :class:`IndexError`
            on the metadata path) when the requested band is past the
            last available band. The engine performs no resolution, so
            the error should surface to the caller unchanged.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        with pytest.raises((ValueError, IndexError)):
            dataset.analysis.plot(band=42)



