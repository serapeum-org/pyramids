import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal
from pandas import DataFrame

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset._plot_helpers import render_array
from pyramids.dataset.engines import Analysis, Bands
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each plot test to bound memory.

    Plotting tests open figures via cleopatra/pyplot; without this teardown
    the suite accumulates them and matplotlib warns past 20 open figures.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


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
    def test_constant_value_band_does_not_raise(self):
        """A flat / constant-value band plots without raising.

        Regression guard for the cleopatra 0.11.0 flat-data fix: a degenerate
        colour range (e.g. a single-class mask, a flat DEM tile, or a
        nodata-only window) used to raise ``ZeroDivisionError`` while deriving
        tick spacing. Such bands are routine in GIS rasters, so plotting one
        must succeed.
        """
        arr = np.ones((8, 8), dtype="float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        array_glyph = dataset.plot(band=0)
        assert isinstance(array_glyph, ArrayGlyph)

    @pytest.mark.plot
    def test_add_colorbar_toggle_controls_cbar(self, src: Dataset):
        """``add_colorbar`` flows through to the glyph's ``cbar``.

        Test scenario:
            The default render draws a colorbar (``glyph.cbar`` is set),
            and ``add_colorbar=False`` suppresses it (``glyph.cbar`` is
            ``None``). This pins the cleopatra pass-through documented on
            the plot facade.
        """
        dataset = Dataset(src)
        with_cbar = dataset.plot(band=0)
        assert with_cbar.cbar is not None, "default plot must draw a colorbar"
        without_cbar = dataset.plot(band=0, add_colorbar=False)
        assert (
            without_cbar.cbar is None
        ), "add_colorbar=False must suppress the colorbar"

    @pytest.mark.plot
    def test_glyph_exposes_mappable_im(self, src: Dataset):
        """The returned glyph exposes the colour-mapped artist as ``im``."""
        dataset = Dataset(src)
        glyph = dataset.plot(band=0)
        assert glyph.im is not None, "glyph.im (the mappable) must be populated"

    @pytest.mark.plot
    def test_plot_histogram_returns_fig_ax_hist(self, src: Dataset):
        """``plot_histogram`` renders and returns ``(fig, ax, hist)``."""
        dataset = Dataset(src)
        fig, ax, hist = dataset.plot_histogram(band=0, bins=10)
        assert fig is not None and ax is not None
        assert isinstance(hist, dict)

    @pytest.mark.plot
    def test_plot_histogram_excludes_invalid_samples(self):
        """No-data and ``exclude_value`` samples never reach the glyph.

        Test scenario:
            A band carrying a no-data pixel (``-9999``) and a repeated
            ``7.0`` is histogrammed with ``exclude_value=7.0``. Capturing
            the values handed to ``StatisticalGlyph`` proves both the
            no-data value and the explicit ``exclude_value`` are dropped,
            leaving only the genuine samples.
        """
        arr = np.array([[1.0, 2.0, -9999.0], [3.0, 7.0, 7.0]], dtype="float32")
        dataset = Dataset.create_from_array(
            arr,
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        captured: dict = {}

        class _FakeSG:
            @staticmethod
            def filter_kwargs(kw):
                return {}

            def __init__(self, values, ax=None, **kwargs):
                captured["values"] = np.asarray(values)

            def histogram(self, bins=15):
                return ("fig", "ax", {})

        with patch("cleopatra.statistical_glyph.StatisticalGlyph", new=_FakeSG):
            dataset.plot_histogram(band=0, bins=5, exclude_value=7.0)
        vals = sorted(captured["values"].tolist())
        assert vals == [
            1.0,
            2.0,
            3.0,
        ], f"nodata (-9999) and exclude_value (7.0) must be dropped; got {vals}"

    @pytest.mark.plot
    def test_invalid_color_scale_raises(self, src: Dataset):
        """An unsupported ``color_scale`` fails fast with a clear message.

        Test scenario:
            ``color_scale="bogus"`` is rejected before any rendering work,
            with a pyramids-side ``ValueError`` that names the offending
            value (and lists the valid options).
        """
        dataset = Dataset(src)
        with pytest.raises(ValueError, match=r"color_scale"):
            dataset.plot(band=0, color_scale="bogus")

    @pytest.mark.plot
    def test_valid_color_scale_any_case_passes(self, src: Dataset):
        """A valid ``color_scale`` passes regardless of case.

        Test scenario:
            ``ColorScale`` lookup is case-insensitive, so a mixed-case
            ``"Power"`` must validate and render without raising.
        """
        dataset = Dataset(src)
        glyph = dataset.plot(band=0, color_scale="Power")
        assert isinstance(glyph, ArrayGlyph)

    @pytest.mark.plot
    def test_to_image_returns_pil_image(self, src: Dataset):
        """``to_image`` exports a band as a colour-mapped PIL image.

        Test scenario:
            The returned object is a ``PIL.Image.Image`` sized to the
            band's (columns, rows), i.e. a real colour-mapped raster
            thumbnail rather than raw matplotlib state.
        """
        from PIL import Image

        dataset = Dataset(src)
        image = dataset.to_image(band=0, cmap="viridis")
        assert isinstance(image, Image.Image)
        assert image.size == (dataset.columns, dataset.rows)

    @pytest.mark.plot
    def test_to_image_constant_band_returns_image(self):
        """A flat / constant-value band still exports a (degenerate) image.

        Test scenario:
            A constant band has no dynamic range, so cleopatra's colormap
            normalisation is degenerate (and may warn); ``to_image`` must
            still return a ``PIL.Image.Image`` of the right size rather than
            raising. Behaviour is documented, not "fixed" in cleopatra.
        """
        import warnings

        from PIL import Image

        arr = np.full((4, 4), 5.0, dtype="float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = dataset.to_image(band=0)
        assert isinstance(image, Image.Image)
        assert image.size == (dataset.columns, dataset.rows)

    @pytest.mark.plot
    def test_to_image_all_nodata_raises(self):
        """A fully no-data band raises instead of rendering a blank image.

        Test scenario:
            Every pixel equals the no-data value, so there are no valid
            samples to colour-map; ``to_image`` must raise a targeted
            ``ValueError`` rather than feed an all-masked array to
            ``apply_colormap`` (whose normalisation is then degenerate).
        """
        arr = np.full((4, 4), -9999.0, dtype="float32")
        dataset = Dataset.create_from_array(
            arr,
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        with pytest.raises(ValueError, match="no valid"):
            dataset.to_image(band=0)

    @pytest.mark.plot
    def test_plot_histogram_all_nodata_raises(self):
        """A fully no-data band raises a clear error instead of feeding the
        glyph an empty array.

        Test scenario:
            Every pixel equals the no-data value, so after masking there are
            no samples; ``plot_histogram`` must raise a targeted ``ValueError``
            rather than passing an empty array to ``StatisticalGlyph``.
        """
        arr = np.full((4, 4), -9999.0, dtype="float32")
        dataset = Dataset.create_from_array(
            arr,
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        with pytest.raises(ValueError, match="no valid samples"):
            dataset.plot_histogram(band=0)

    @pytest.mark.plot
    def test_plot_histogram_integer_band_skips_nan_branch(self):
        """An integer-dtype band histograms without the float-NaN masking.

        Test scenario:
            ``np.isnan`` rejects integer arrays, so the masking guards on
            ``np.issubdtype(..., np.floating)``. An integer band must
            therefore histogram successfully (exercising the non-float
            branch) and still drop its no-data value.
        """
        arr = np.array([[1, 2, 0], [3, 4, 0]], dtype="int32")
        dataset = Dataset.create_from_array(
            arr,
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=0,
        )
        fig, ax, hist = dataset.plot_histogram(band=0, bins=4)
        assert fig is not None and ax is not None
        assert isinstance(hist, dict)

    @pytest.mark.plot
    def test_plot_histogram_draws_on_supplied_ax(self):
        """A caller-supplied ``ax`` is the one drawn on.

        Test scenario:
            Passing ``ax=`` binds the histogram to that axes; the returned
            axes must be the same object so callers can compose subplots.
        """
        import matplotlib.pyplot as plt

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
        )
        _fig, host_ax = plt.subplots()
        _f, ax, _h = dataset.plot_histogram(band=0, ax=host_ax)
        assert ax is host_ax, "histogram must draw on the supplied ax"

    @pytest.mark.plot
    def test_to_image_exclude_value_masks_extra_value(self):
        """``exclude_value`` is masked in addition to the no-data value.

        Test scenario:
            Passing ``exclude_value`` exercises the extra-mask branch in
            ``to_image``; the call must still return a correctly sized
            ``PIL.Image.Image``.
        """
        import warnings

        from PIL import Image

        arr = np.array([[1.0, 2.0, 7.0], [3.0, 4.0, 7.0]], dtype="float32")
        dataset = Dataset.create_from_array(
            arr,
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = dataset.to_image(band=0, exclude_value=7.0)
        assert isinstance(image, Image.Image)
        assert image.size == (dataset.columns, dataset.rows)

    @pytest.mark.plot
    def test_plot_vector_field_custom_bands(self):
        """Non-default ``u_band``/``v_band`` select the right components.

        Test scenario:
            On a 3-band stack, choosing bands 1 and 2 as (u, v) must render
            without error, confirming the band indices are honoured rather
            than hard-coded to 0/1.
        """
        rng = np.random.default_rng(11)
        stack = rng.standard_normal((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            stack, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
        )
        fig, ax, _ = dataset.plot_vector_field(u_band=1, v_band=2, kind="quiver")
        assert fig is not None and ax is not None

    @pytest.mark.plot
    def test_plot_vector_field_descending_x_is_flipped(self):
        """A descending-x geotransform is flipped to ascending for rendering.

        Test scenario:
            A raster whose x cell-centres decrease left-to-right (negative
            pixel width) exercises the ``x``-descending flip branch; the
            field must still render without error and the helper must not
            choke on the non-ascending coordinate axis.
        """
        rng = np.random.default_rng(3)
        uv = rng.standard_normal((2, 5, 5)).astype("float32")
        geo = (10.0, -1.0, 0.0, 0.0, 0.0, -1.0)
        dataset = Dataset.create_from_array(uv, geo=geo, epsg=4326)
        assert dataset.x[0] > dataset.x[-1], "x must be descending to hit the branch"
        fig, ax, _ = dataset.plot_vector_field(u_band=0, v_band=1, kind="streamplot")
        assert fig is not None and ax is not None

    @pytest.mark.plot
    def test_plot_vector_field_invalid_kind_raises(self):
        """An unsupported ``kind`` surfaces cleopatra's ``ValueError``.

        Test scenario:
            ``kind`` is forwarded to ``VectorGlyph.plot``, which only accepts
            ``quiver``/``barbs``/``streamplot``; an unknown kind must raise
            rather than silently fall back.
        """
        dataset = self._uv_dataset()
        with pytest.raises(ValueError):
            dataset.plot_vector_field(u_band=0, v_band=1, kind="bogus")

    @staticmethod
    def _uv_dataset():
        """Build a tiny 2-band (u, v) dataset for vector-field tests.

        The issue cites ``tests/data/flow_direction_array.npy`` as sample
        vector-field data, but that file does not exist in the repo, so a
        synthetic ``(2, rows, cols)`` u/v stack is used instead.
        """
        rng = np.random.default_rng(7)
        u = rng.standard_normal((6, 6)).astype("float32")
        v = rng.standard_normal((6, 6)).astype("float32")
        stack = np.stack([u, v])
        return Dataset.create_from_array(
            stack, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
        )

    @pytest.mark.plot
    @pytest.mark.parametrize("kind", ["quiver", "barbs", "streamplot"])
    def test_plot_vector_field_kinds(self, kind: str):
        """``plot_vector_field`` renders each VectorGlyph kind.

        Test scenario:
            A two-band (u, v) dataset renders as quiver / barbs /
            streamplot without error and returns the ``(fig, ax, im)``
            triple from ``VectorGlyph.plot``.
        """
        dataset = self._uv_dataset()
        fig, ax, _ = dataset.plot_vector_field(u_band=0, v_band=1, kind=kind)
        assert fig is not None and ax is not None

    @pytest.mark.plot
    def test_plot_vector_field_add_colorbar_false(self):
        """``add_colorbar=False`` suppresses the magnitude colorbar.

        Test scenario:
            Without a colorbar the figure owns a single Axes; the default
            (colorbar on) adds a second Axes for the colour bar.
        """
        dataset = self._uv_dataset()
        fig, _, _ = dataset.plot_vector_field(
            u_band=0, v_band=1, kind="quiver", add_colorbar=False
        )
        assert len(fig.axes) == 1, "add_colorbar=False must not add a colorbar axes"

    @pytest.mark.plot
    def test_plot_vector_field_band_out_of_range_raises(self):
        """A single-band dataset gives a clear error, not a GDAL/index crash.

        Test scenario:
            ``plot_vector_field`` defaults to ``v_band=1``; on a one-band
            raster that band does not exist, so it must raise a targeted
            ``ValueError`` naming the offending band rather than a low-level
            read failure.
        """
        arr = np.ones((1, 5, 5), dtype="float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
        )
        with pytest.raises(ValueError, match=r"v_band=1 is out of range"):
            dataset.plot_vector_field()

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

    @pytest.mark.plot
    def test_plot_no_nodata_value_does_not_crash(self, tmp_path):
        """`DatasetCollection.plot()` must render rasters that carry no no-data value.

        Regression for #480: rasters written without a no-data value (common for
        external exports, e.g. Google Earth Engine ``getDownloadURL``) report
        ``no_data_value == (None,)``. Previously the animate path forwarded
        ``[None]`` to cleopatra, which crashed in ``np.isclose(array, None)`` with
        ``TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'``.
        The ``None`` -> ``np.nan`` sanitisation (mirroring ``Analysis.plot``) masks
        nothing and renders every cell instead of raising.
        """
        rng = np.random.default_rng(0)
        files = []
        for i in range(3):
            arr = rng.random((1, 12, 12), dtype="float32")
            ds = Dataset.create_from_array(
                arr=arr, geo=(0, 0.1, 0, 2, 0, -0.1), epsg=4326, no_data_value=None
            )
            path = tmp_path / f"frame_{i}.tif"
            ds.to_file(str(path))
            files.append(str(path))

        cube = DatasetCollection.from_files(files)
        assert cube.base.no_data_value == (None,), "precondition: nodata is unset"
        glyph = cube.plot(band=0)
        assert isinstance(glyph, ArrayGlyph)

    @staticmethod
    def _rgb_cube(tmp_path, n_times=4, n_bands=3, dim=8):
        """Build a co-registered multi-band DatasetCollection for RGB tests."""
        rng = np.random.default_rng(0)
        files = []
        for t in range(n_times):
            arr = (rng.random((n_bands, dim, dim), dtype="float32") * 255).astype(
                "float32"
            )
            ds = Dataset.create_from_array(
                arr=arr, geo=(0, 1, 0, 0, 0, -1), epsg=4326
            )
            path = tmp_path / f"rgb_{t}.tif"
            ds.to_file(str(path))
            files.append(str(path))
        return DatasetCollection.from_files(files)

    @pytest.mark.plot
    def test_rgb_animate_keeps_every_frame(self, tmp_path):
        """`plot(rgb=...)` returns a true-colour animation, one frame per timestep.

        Regression for #538: passing ``rgb`` used to flow through ``**kwargs`` into
        ``render_array``'s ``rgb`` parameter with a single-band ``(time, rows, cols)``
        stack, so cleopatra read the time axis as the RGB channels — collapsing the
        frames into a single ``(rows, cols, 3)`` still. The dedicated ``rgb`` path now
        stacks every band per timestep and composites a ``(time, rows, cols, 3)`` stack,
        so all four frames survive.
        """
        cube = self._rgb_cube(tmp_path, n_times=4)
        glyph = cube.plot(rgb_options={"rgb": [0, 1, 2], "percentile": 2})
        assert isinstance(glyph, ArrayGlyph)
        assert glyph.arr.shape == (4, 8, 8, 3), "must keep all 4 true-colour frames"
        assert glyph.cbar is None, "true-colour frames carry no colorbar"

    @pytest.mark.plot
    def test_rgb_insufficient_bands_raises(self, tmp_path):
        """A misshapen `rgb=` raises instead of silently dropping frames (#538)."""
        cube = self._rgb_cube(tmp_path, n_times=3, n_bands=2)
        with pytest.raises(ValueError, match="needs at least 3 bands"):
            cube.plot(rgb_options={"rgb": [0, 1, 2]})

    @pytest.mark.plot
    def test_rgb_loose_kwarg_is_deprecated(self, tmp_path):
        """The loose top-level `rgb=` still works but warns, mirroring Dataset.plot."""
        cube = self._rgb_cube(tmp_path, n_times=3)
        with pytest.warns(DeprecationWarning, match="rgb_options"):
            glyph = cube.plot(rgb=[0, 1, 2], percentile=2)
        assert glyph.arr.shape == (3, 8, 8, 3)

    @pytest.mark.plot
    def test_single_band_path_unchanged(self, tmp_path):
        """Without `rgb`, plot() still yields a colormapped single-band animation."""
        cube = self._rgb_cube(tmp_path, n_times=3)
        glyph = cube.plot(band=0)
        assert isinstance(glyph, ArrayGlyph)
        assert glyph.arr.ndim == 3, "single-band animate stays (time, rows, cols)"

    @pytest.mark.plot
    def test_rgb_surface_reflectance_normalisation(self, tmp_path):
        """`rgb_options` may normalise via surface_reflectance instead of percentile.

        Exercises the non-percentile branch of cleopatra's ``prepare_array`` through
        the collection RGB path: the result is still a display-ready
        ``(time, rows, cols, 3)`` stack in ``[0, 1]``.
        """
        cube = self._rgb_cube(tmp_path, n_times=3)
        glyph = cube.plot(rgb_options={"rgb": [0, 1, 2], "surface_reflectance": 255})
        assert glyph.arr.shape == (3, 8, 8, 3), "RGB stack keeps every frame"
        assert float(glyph.arr.min()) >= 0.0 and float(glyph.arr.max()) <= 1.0, (
            "surface-reflectance frames must be normalised into [0, 1]"
        )

    @pytest.mark.plot
    def test_rgb_options_unknown_key_raises(self, tmp_path):
        """An unknown `rgb_options` key raises (delegated to `_merge_rgb_options`)."""
        cube = self._rgb_cube(tmp_path, n_times=2)
        with pytest.raises(ValueError, match=r"Unknown keys in `rgb_options`"):
            cube.plot(rgb_options={"bogus": 1})

    @pytest.mark.plot
    def test_rgb_options_wins_over_loose_kwarg(self, tmp_path):
        """On collision, `rgb_options` wins over the loose kwarg and still warns."""
        cube = self._rgb_cube(tmp_path, n_times=2, n_bands=4)
        with pytest.warns(DeprecationWarning, match="rgb_options` wins"):
            glyph = cube.plot(rgb=[1, 2, 3], rgb_options={"rgb": [0, 1, 2]})
        assert glyph.arr.shape == (2, 8, 8, 3), "grouped rgb composited every frame"


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

    See also:
        ``tests/dataset/test_plot_resolution.py::TestNetCDFPlotPolicy`` (the
        parametrised forbidden-kwarg / default-band matrix) and the full
        ``tests/netcdf/test_plot.py`` suite (the post-PR-2 xarray-aligned
        signature in depth — ``Selectors``/``ColourOpts``/``FacetSpec``,
        curvilinear coords, faceting, animate, lazy). This class holds the
        original PR-1/D-0 cases.
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

        with patch.object(type(nc_subset.analysis), "plot", autospec=True) as mock_plot:
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
        assert isinstance(
            result, ArrayGlyph
        ), f"Expected ArrayGlyph, got {type(result).__name__}"

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
        assert (
            first.fig is not second.fig
        ), "Each call must own a distinct matplotlib Figure"

    @pytest.mark.plot
    @pytest.mark.parametrize(
        "color_scale",
        [
            "linear",
            "power",
            "sym-lognorm",
            "boundary-norm",
            "Power",
        ],
    )
    def test_color_scale_string_aliases_work(self, color_scale):
        """``color_scale`` accepts the ``cleopatra.styles.ColorScale`` aliases.

        Args:
            color_scale: String alias for the colour-scale enum (lookup
                is case-insensitive on the cleopatra side).

        Test scenario:
            The plot docstrings document the string aliases as the valid
            ``color_scale`` values. Verify each alias (including a
            mixed-case spelling) is forwarded to cleopatra and yields an
            ``ArrayGlyph``.

        Notes:
            cleopatra's ``ColorScale`` StrEnum replaced the legacy
            integer codes (``1``-``5``); those are no longer accepted —
            see :meth:`test_color_scale_integer_codes_rejected`.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        plot_kwargs = {"color_scale": color_scale}
        if color_scale.lower() == "boundary-norm":
            plot_kwargs["bounds"] = [0.0, 0.25, 0.5, 0.75, 1.0]

        result = dataset.plot(**plot_kwargs)
        assert isinstance(result, ArrayGlyph), (
            f"color_scale={color_scale!r} should return ArrayGlyph, "
            f"got {type(result).__name__}"
        )

    @pytest.mark.plot
    @pytest.mark.parametrize("color_scale", [1, 2, 3, 4, 5, 0])
    def test_color_scale_integer_codes_rejected(self, color_scale):
        """Legacy integer ``color_scale`` codes raise a clear ``ValueError``.

        Args:
            color_scale: A legacy integer code that older releases
                accepted but cleopatra's ``ColorScale`` enum rejects.

        Test scenario:
            cleopatra now validates ``color_scale`` against the
            ``ColorScale`` StrEnum and raises ``ValueError`` for anything
            that is not one of the documented aliases (or a ``ColorScale``
            member). Confirm pyramids surfaces that error unchanged.
        """
        rng = np.random.default_rng(7)
        arr = rng.random((5, 5)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        with pytest.raises(ValueError, match="color_scale"):
            dataset.plot(color_scale=color_scale)

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

        assert (
            result == "stub-glyph"
        ), f"Facade must return engine output, got {result!r}"
        assert mock_plot.call_count == 1
        call_kwargs = mock_plot.call_args.kwargs
        assert (
            call_kwargs["band"] == 0
        ), f"Resolver should send band=0, got {call_kwargs.get('band')}"
        assert call_kwargs["figsize"] == (
            4,
            4,
        ), f"Extra kwargs must propagate, got {call_kwargs.get('figsize')}"


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

    def test_collision_warning_distinguishes_from_pure_loose(self):
        """A loose+grouped collision warns "rgb_options wins", not "group them" (M2).

        Test scenario:
            M2 fix — when both `rgb=` (loose) and `rgb_options={"rgb": ...}`
            are passed for the same key, the DeprecationWarning must say
            the loose value was overridden ("`rgb_options` wins — drop
            the loose form"), *not* the misleading "Group them under
            `rgb_options={...}` instead" message that implies the user
            forgot the grouped form. Conversely, a pure-loose call (no
            `rgb_options`) still gets the original "group them" wording.
        """
        rng = np.random.default_rng(17)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        with patch.object(type(dataset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub"
            with warnings.catch_warnings(record=True) as collide:
                warnings.simplefilter("always")
                dataset.plot(rgb=[2, 1, 0], rgb_options={"rgb": [0, 1, 2]})
            with warnings.catch_warnings(record=True) as pure:
                warnings.simplefilter("always")
                dataset.plot(rgb=[2, 1, 0])
        collide_msg = " ".join(str(w.message) for w in collide)
        pure_msg = " ".join(str(w.message) for w in pure)
        assert (
            "rgb_options` wins" in collide_msg and "drop the loose form" in collide_msg
        ), f"collision warning should say rgb_options wins; got: {collide_msg!r}"
        assert (
            "Group them under" not in collide_msg
        ), f"collision warning must not use the 'group them' wording; got: {collide_msg!r}"
        assert (
            "Group them under" in pure_msg
        ), f"pure-loose warning should keep the 'group them' wording; got: {pure_msg!r}"


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
        assert isinstance(
            result, ArrayGlyph
        ), f"Expected ArrayGlyph, got {type(result).__name__}"

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


class TestDatasetPlotRgbOptionsEdges:
    """PR-4 / D-3 edge cases for ``rgb_options=`` not in :class:`TestDatasetPlotRgbOptions`.

    Coverage targets the merge precedence between the grouped form and
    the loose-kwarg form, empty-dict handling, no-op semantics, and
    interactions with the deprecation warning.
    """

    @pytest.fixture(scope="function")
    def multiband_dataset(self):
        """Build an in-memory 3-band float32 dataset for RGB testing.

        Returns:
            Dataset: A small 3-band dataset with random values.
        """
        rng = np.random.default_rng(2025)
        arr = rng.random((3, 6, 6)).astype("float32")
        return Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

    @pytest.mark.plot
    def test_rgb_options_with_rgb_and_surface_reflectance(self, multiband_dataset):
        """Both `rgb` and `surface_reflectance` keys forward correctly.

        Test scenario:
            ``rgb_options={"rgb": [2, 1, 0], "surface_reflectance": 10000}``
            must populate both kwargs on the downstream Analysis.plot
            call without any DeprecationWarning (the grouped form is
            the recommended path).
        """
        with patch.object(
            type(multiband_dataset.analysis), "plot", autospec=True
        ) as mock_plot:
            mock_plot.return_value = "stub"
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                multiband_dataset.plot(
                    rgb_options={"rgb": [2, 1, 0], "surface_reflectance": 10000},
                )
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs["rgb"] == [
            2,
            1,
            0,
        ], f"rgb must be forwarded, got: {call_kwargs.get('rgb')}"
        assert call_kwargs["surface_reflectance"] == 10000, (
            f"surface_reflectance must be forwarded, "
            f"got: {call_kwargs.get('surface_reflectance')}"
        )
        deprecations = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecations, (
            f"Grouped form must not emit DeprecationWarning, got: "
            f"{[str(w.message) for w in deprecations]}"
        )

    @pytest.mark.plot
    def test_empty_rgb_options_dict_is_noop(self, multiband_dataset):
        """`rgb_options={}` (empty) leaves all kwargs at their defaults.

        Test scenario:
            An empty dict resolves to no overrides — the four
            Sentinel kwargs remain ``None`` (which the resolver
            propagates to the engine as ``rgb=None``, etc.). No
            DeprecationWarning is emitted because no loose kwargs
            were passed.
        """
        with patch.object(
            type(multiband_dataset.analysis), "plot", autospec=True
        ) as mock_plot:
            mock_plot.return_value = "stub"
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                multiband_dataset.plot(rgb_options={})
        call_kwargs = mock_plot.call_args.kwargs
        # No Sentinel kwargs were set; the resolver passes None through.
        assert (
            call_kwargs.get("rgb") is None
        ), f"Empty rgb_options should leave rgb=None, got: {call_kwargs.get('rgb')}"
        assert (
            call_kwargs.get("surface_reflectance") is None
        ), "Empty rgb_options should leave surface_reflectance=None"
        deprecations = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecations, "Empty rgb_options must not emit DeprecationWarning"

    @pytest.mark.plot
    def test_loose_rgb_with_empty_group_still_warns(self, multiband_dataset):
        """Loose `rgb=` with `rgb_options={}` still triggers the deprecation.

        Test scenario:
            The warning is gated on the loose kwargs being not-None
            (not on whether ``rgb_options`` is provided). An empty
            group doesn't suppress the warning when a loose kwarg is
            present. The loose ``rgb`` survives because the empty
            group has no entries to overwrite it.
        """
        with patch.object(
            type(multiband_dataset.analysis), "plot", autospec=True
        ) as mock_plot:
            mock_plot.return_value = "stub"
            with pytest.warns(DeprecationWarning, match=r"rgb_options"):
                multiband_dataset.plot(rgb=[0, 1, 2], rgb_options={})
        assert mock_plot.call_args.kwargs["rgb"] == [
            0,
            1,
            2,
        ], "Loose rgb must survive when rgb_options is empty"

    @pytest.mark.plot
    def test_only_loose_surface_reflectance_emits_warning_once(self, multiband_dataset):
        """A single loose `surface_reflectance=` triggers exactly one warning.

        Test scenario:
            The merge helper emits one warning regardless of how many
            loose kwargs were passed; with only one set, exactly one
            DeprecationWarning fires. Verifies the warning is not
            spammed per-kwarg.
        """
        with patch.object(
            type(multiband_dataset.analysis), "plot", autospec=True
        ) as mock_plot:
            mock_plot.return_value = "stub"
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                multiband_dataset.plot(surface_reflectance=10000)
        deprecations = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecations) == 1, (
            f"Exactly one DeprecationWarning expected, got {len(deprecations)}: "
            f"{[str(w.message) for w in deprecations]}"
        )
        assert "surface_reflectance" in str(
            deprecations[0].message
        ), f"Warning must name the loose kwarg, got: {deprecations[0].message}"

    @pytest.mark.plot
    def test_rgb_options_only_partial_override(self, multiband_dataset):
        """`rgb_options` overrides only its own keys; other loose kwargs survive.

        Test scenario:
            Pass loose ``percentile=2`` AND
            ``rgb_options={"rgb": [0, 1, 2]}``. The grouped form
            overrides only ``rgb`` (which wasn't set loose), and the
            loose ``percentile`` propagates unchanged. The
            DeprecationWarning still fires because at least one loose
            kwarg was used.
        """
        with patch.object(
            type(multiband_dataset.analysis), "plot", autospec=True
        ) as mock_plot:
            mock_plot.return_value = "stub"
            with pytest.warns(DeprecationWarning, match=r"percentile"):
                multiband_dataset.plot(
                    percentile=2,
                    rgb_options={"rgb": [0, 1, 2]},
                )
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs["rgb"] == [
            0,
            1,
            2,
        ], f"Grouped rgb must propagate, got: {call_kwargs.get('rgb')}"
        assert (
            call_kwargs["percentile"] == 2
        ), f"Loose percentile must survive, got: {call_kwargs.get('percentile')}"


class TestPlotPhase3CrossCutting:
    """Cross-cutting regressions for the PR-4 D-2 refactor.

    The shared ``render_array`` helper now backs ``Analysis.plot``,
    ``DatasetCollection.plot``, and the NetCDF facet path. These tests
    pin the public-contract invariants: types, return values, and
    behavioural equivalence vs. the engine directly.
    """

    @pytest.fixture(scope="function")
    def single_band_dataset(self):
        """Build a deterministic single-band dataset for cross-cutting tests.

        Returns:
            Dataset: A 1-band float32 dataset.
        """
        rng = np.random.default_rng(7777)
        arr = rng.random((6, 6)).astype("float32")
        return Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

    @pytest.mark.plot
    def test_dataset_plot_returns_array_glyph_post_refactor(self, single_band_dataset):
        """`Dataset.plot()` still returns an ArrayGlyph after D-2 collapse.

        Test scenario:
            The D-2 refactor routes ``Dataset.plot`` through the
            shared ``render_array`` helper. The public contract is
            unchanged: the call must return a cleopatra ArrayGlyph
            instance. Pre-refactor parity is required for downstream
            callers that chain visual customisations.
        """
        result = single_band_dataset.plot()
        assert isinstance(
            result, ArrayGlyph
        ), f"Dataset.plot() must return ArrayGlyph after D-2, got: {type(result).__name__}"

    @pytest.mark.plot
    def test_analysis_plot_returns_array_glyph_post_refactor(self, single_band_dataset):
        """`Analysis.plot(band=N)` still returns an ArrayGlyph after D-2.

        Test scenario:
            ``Analysis.plot`` is the engine; the D-2 collapse pushes
            cleo construction into ``render_array``. A direct engine
            call must still produce an ArrayGlyph (this is the
            existing public engine API).
        """
        result = single_band_dataset.analysis.plot(band=0)
        assert isinstance(result, ArrayGlyph), (
            f"Analysis.plot() must return ArrayGlyph after D-2, "
            f"got: {type(result).__name__}"
        )

    @pytest.mark.plot
    def test_dataset_collection_plot_returns_array_glyph_post_refactor(
        self, rasters_folder_path
    ):
        """`DatasetCollection.plot()` still returns an ArrayGlyph after D-2.

        Test scenario:
            ``DatasetCollection.plot`` was previously a direct
            cleopatra-constructor call; D-2 routes it through
            ``render_array`` with ``mode="animate"``. The return type
            (cleopatra ArrayGlyph) is preserved. Uses the same fixture
            chain as :class:`TestPlotDatasetCollection`.
        """
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        result = cube.plot()
        assert isinstance(result, ArrayGlyph), (
            f"DatasetCollection.plot() must return ArrayGlyph after D-2, "
            f"got: {type(result).__name__}"
        )

    @pytest.mark.plot
    def test_render_array_direct_call_matches_analysis_plot(self, single_band_dataset):
        """Calling `render_array(mode="plot")` directly produces the same array.

        Test scenario:
            The shared helper is module-private but the contract is
            stable: calling it with the same array + extent that
            ``Analysis.plot`` would compute internally yields an
            ArrayGlyph wrapping the same data. This guards against a
            regression where the engine and helper diverge on how the
            data array is reshaped.
        """
        arr = single_band_dataset.read_array(band=0)
        bbox = single_band_dataset.bbox
        helper_glyph = render_array(
            arr=arr,
            extent=bbox,
            exclude_value=[np.nan],
            mode="plot",
        )
        engine_glyph = single_band_dataset.analysis.plot(band=0)
        np.testing.assert_array_equal(
            helper_glyph.arr,
            engine_glyph.arr,
            err_msg="render_array(mode='plot') and Analysis.plot must produce "
            "identical .arr",
        )

    @pytest.mark.plot
    def test_render_array_invalid_mode_raises(self):
        """`render_array(mode="bogus")` raises ValueError naming the valid modes.

        Test scenario:
            The helper validates ``mode`` against
            ``("plot", "animate", "facet")``. An unknown value must
            raise a ValueError that lists the valid options so the
            caller can fix it.
        """
        arr = np.zeros((4, 4), dtype="float32")
        with pytest.raises(ValueError, match=r"Invalid mode") as exc_info:
            render_array(arr=arr, mode="bogus")
        msg = str(exc_info.value)
        assert "plot" in msg, f"Error must list 'plot', got: {msg}"
        assert "animate" in msg, f"Error must list 'animate', got: {msg}"
        assert "facet" in msg, f"Error must list 'facet', got: {msg}"

    @pytest.mark.plot
    def test_render_array_animate_requires_axis_values(self):
        """`render_array(mode="animate")` without `animation_axis_values` raises."""
        arr = np.zeros((3, 4, 4), dtype="float32")
        with pytest.raises(ValueError, match=r"animation_axis_values"):
            render_array(arr=arr, mode="animate")

    @pytest.mark.plot
    def test_render_array_facet_requires_facet_kwargs(self):
        """`render_array(mode="facet")` without `facet_kwargs` raises."""
        arr = np.zeros((3, 4, 4), dtype="float32")
        with pytest.raises(ValueError, match=r"facet_kwargs"):
            render_array(arr=arr, mode="facet")

    @pytest.mark.plot
    def test_render_array_rgb_animate_requires_4d(self):
        """RGB animate with a single-band 3-D stack raises rather than collapsing.

        Guards the #538 silent-frame-loss: a 3-D ``(time, rows, cols)`` array plus
        ``rgb`` would otherwise reach cleopatra as a single composited still.
        """
        arr = np.zeros((4, 8, 8), dtype="float32")
        with pytest.raises(ValueError, match=r"RGB animate requires a 4-D"):
            render_array(
                arr=arr,
                rgb=[0, 1, 2],
                mode="animate",
                animation_axis_values=[0, 1, 2, 3],
            )

    @pytest.mark.plot
    def test_render_array_rgb_animate_none_arr_raises(self):
        """RGB animate with `arr=None` hits the same 4-D guard and reports None-D."""
        with pytest.raises(ValueError, match=r"got None-D"):
            render_array(
                arr=None,
                rgb=[0, 1, 2],
                mode="animate",
                animation_axis_values=[0],
            )

    @pytest.mark.plot
    def test_render_array_basemap_requires_epsg(self):
        """`render_array(basemap=True, basemap_epsg=None)` raises with CRS hint.

        Test scenario:
            The basemap path requires a CRS to project the contextily
            tiles. Without one, the helper raises with the same
            "Dataset must have a CRS" message that the original
            ``Analysis.plot`` produced. We assert the error text
            matches so we don't drift away from the pre-refactor
            user-facing message.
        """
        rng = np.random.default_rng(321)
        arr = rng.random((4, 4)).astype("float32")
        with pytest.raises(ValueError, match=r"CRS"):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                basemap=True,
                basemap_epsg=None,
            )


class TestPlotPR6Cleanups:
    """Regression coverage for the PR-6 D-4 / D-5 cleanup."""

    def test_no_legacy_import_cleopatra_in_plot_modules(self):
        """D-5 — every plot-adjacent module uses ``require_cleopatra``.

        Test scenario:
            The legacy ``import_cleopatra(<verbose message>)`` call
            sites were consolidated into a single
            ``require_cleopatra()`` helper in
            :mod:`pyramids.base._utils`. A regression scan of the
            plot-adjacent modules must not turn up any remaining
            ``import_cleopatra(`` *calls* (the function definition
            stays around for downstream consumers). This is a
            grep-style test: the absence is the assertion.
        """
        from pathlib import Path

        repo = Path(__file__).parents[2]
        targets = [
            "src/pyramids/dataset/_plot_helpers.py",
            "src/pyramids/dataset/engines/analysis.py",
            "src/pyramids/dataset/engines/bands.py",
            "src/pyramids/netcdf/ugrid/plot.py",
        ]
        offenders = []
        for rel in targets:
            content = (repo / rel).read_text(encoding="utf-8")
            for line in content.splitlines():
                stripped = line.strip()
                if "import_cleopatra(" in stripped and not stripped.startswith("#"):
                    offenders.append(f"{rel}: {stripped}")
        assert not offenders, (
            f"Found legacy `import_cleopatra(` call sites: {offenders}. "
            "Use `require_cleopatra()` from `pyramids.base._utils`."
        )

    def test_require_cleopatra_callable_from_single_location(self):
        """D-5 — ``require_cleopatra`` lives in one place and is callable.

        Test scenario:
            The consolidated guard is exposed from
            ``pyramids.base._utils.require_cleopatra``. It should be
            importable and callable; the call succeeds silently when
            cleopatra is installed (which it is in the [viz] test env).
        """
        from pyramids.base._utils import require_cleopatra

        require_cleopatra()
        require_cleopatra("optional override message")

    def test_render_array_kwargs_not_double_forwarded(self):
        """D-4 — kwargs reach the constructor xor the render call.

        Test scenario:
            The render-call-only kwargs (``points``, ``point_color``,
            ``point_size``, ``pid_color``, ``pid_size``, ``kind``) must
            reach ``ArrayGlyph.plot``, not the constructor. Every
            other kwarg must land on the constructor exactly once.
            We patch both call sites and inspect the recorded kwargs.
        """
        from unittest.mock import patch as _patch

        from pyramids.dataset._plot_helpers import render_array as _rarr

        rng = np.random.default_rng(7)
        arr = rng.random((4, 4)).astype("float32")

        ctor_seen: dict = {}
        plot_seen: dict = {}

        class _FakeAxes:
            def __init__(self):
                self.aspect = "auto"

            def get_xlim(self):
                return (0.0, 1.0)

            def get_ylim(self):
                return (0.0, 1.0)

        class _FakeGlyph:
            @staticmethod
            def option_keys():
                return ArrayGlyph.option_keys()

            def __init__(self, array, **kwargs):
                ctor_seen.update(kwargs)
                self.arr = array
                self.ax = _FakeAxes()
                self.fig = None

            def plot(self, **kwargs):
                plot_seen.update(kwargs)
                return (None, self.ax)

        with _patch(
            "cleopatra.array_glyph.ArrayGlyph",
            new=_FakeGlyph,
        ):
            _rarr(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                cmap="viridis",
                vmin=0.0,
                kind="imshow",
                points=None,
            )

        assert (
            "cmap" in ctor_seen
        ), f"Ctor should own `cmap`; got ctor={ctor_seen}, plot={plot_seen}"
        assert (
            "cmap" not in plot_seen
        ), f"`cmap` must not double-forward; plot kwargs={plot_seen}"
        assert (
            "kind" in plot_seen
        ), f"`kind` should reach cleo.plot; got plot={plot_seen}"
        assert (
            "kind" not in ctor_seen
        ), f"`kind` should not be on the constructor; ctor={ctor_seen}"


class TestRenderArrayKwargRouting:
    """D-4 — fine-grained checks on which kwargs land where."""

    @staticmethod
    def _capture_calls():
        """Build a fake ``ArrayGlyph`` that records ctor/plot kwargs.

        Returns:
            tuple[type, dict, dict, dict, list]: A ``_FakeGlyph`` class
                wrapping ``__init__`` / ``plot`` / ``animate`` / ``facet``,
                plus dicts capturing each call's kwargs and an
                ``animate_args`` list capturing positional args.
        """
        ctor_seen: dict = {}
        plot_seen: dict = {}
        animate_seen: dict = {}
        facet_seen: dict = {}
        animate_args: list = []

        class _FakeAxes:
            def __init__(self):
                self.aspect = "auto"

            def get_xlim(self):
                return (0.0, 1.0)

            def get_ylim(self):
                return (0.0, 1.0)

        class _FakeGlyph:
            @staticmethod
            def option_keys():
                return ArrayGlyph.option_keys()

            def __init__(self, array, **kwargs):
                ctor_seen.clear()
                ctor_seen.update(kwargs)
                self.arr = array
                self.ax = _FakeAxes()
                self.fig = None

            def plot(self, **kwargs):
                plot_seen.clear()
                plot_seen.update(kwargs)
                return (None, self.ax)

            def animate(self, axis_values, **kwargs):
                animate_args.append(axis_values)
                animate_seen.clear()
                animate_seen.update(kwargs)
                return self

            def facet(self, **kwargs):
                facet_seen.clear()
                facet_seen.update(kwargs)
                return self

        return _FakeGlyph, ctor_seen, plot_seen, animate_seen, facet_seen, animate_args

    def test_constructor_owns_cmap_vmin_vmax_levels_cbar(self):
        """Style/scale kwargs all land on the constructor for plot mode.

        Test scenario:
            ``cmap``, ``vmin``, ``vmax``, ``levels``, ``cbar_kwargs``
            must reach ``ArrayGlyph.__init__`` so cleopatra's
            ``default_options`` is set in one place. None of them may
            also reach ``ArrayGlyph.plot``; otherwise the value would be
            overwritten twice (the PR-6 D-4 fix).
        """
        fake_cls, ctor, plot, _, _, _ = self._capture_calls()
        rng = np.random.default_rng(101)
        arr = rng.random((4, 4)).astype("float32")
        with patch("cleopatra.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                cmap="plasma",
                vmin=-1.0,
                vmax=2.0,
                levels=10,
                cbar_kwargs={"orientation": "horizontal"},
            )
        for key in ("cmap", "vmin", "vmax", "levels", "cbar_kwargs"):
            assert key in ctor, f"`{key}` must land on the constructor; ctor={ctor}"
            assert (
                key not in plot
            ), f"`{key}` must NOT also reach cleo.plot; plot={plot}"

    def test_render_call_only_kwargs_reach_plot(self):
        """``points``/``point_color``/``point_size``/``pid_color``/``pid_size``/``kind``.

        Test scenario:
            Every render-call-only kwarg in the D-4 list must reach
            ``ArrayGlyph.plot`` exclusively. The cleanup added the
            ``plot_call_only`` set in ``_plot_helpers.render_array``; a
            regression here would resurrect the double-forward bug.
        """
        fake_cls, ctor, plot, _, _, _ = self._capture_calls()
        rng = np.random.default_rng(202)
        arr = rng.random((4, 4)).astype("float32")
        render_only_kwargs = {
            "points": None,
            "point_color": "red",
            "point_size": 5,
            "pid_color": "blue",
            "pid_size": 7,
            "kind": "imshow",
        }
        with patch("cleopatra.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                **render_only_kwargs,
            )
        for key in render_only_kwargs:
            assert key in plot, f"`{key}` must reach cleo.plot; plot={plot}"
            assert (
                key not in ctor
            ), f"`{key}` must NOT also be on the constructor; ctor={ctor}"

    def test_animate_mode_merges_both_buckets_into_animate_call(self):
        """``mode='animate'`` — every kwarg flows into ``cleo.animate(...)``.

        Test scenario:
            cleopatra's ``ArrayGlyph.animate`` re-validates every kwarg
            against ``DEFAULT_OPTIONS``, so the D-4 documentation calls
            out the animate path as the exception: both render-call-only
            and constructor buckets merge into a single ``animate_kwargs``
            dict, and the constructor receives nothing.
        """
        fake_cls, ctor, _, animate, _, anim_args = self._capture_calls()
        rng = np.random.default_rng(303)
        arr = rng.random((4, 4)).astype("float32")
        with patch("cleopatra.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="animate",
                animation_axis_values=[0, 1, 2],
                cmap="viridis",
                kind="imshow",
                interval=50,
            )
        for key in ("cmap", "kind", "interval"):
            assert key in animate, (
                f"In animate mode, `{key}` must reach cleo.animate; "
                f"animate kwargs={animate}"
            )
            assert key not in ctor, (
                f"In animate mode, `{key}` must NOT be on the constructor; "
                f"ctor={ctor}"
            )
        assert anim_args == [
            [0, 1, 2]
        ], f"animation_axis_values must be positional; got {anim_args}"

    def test_facet_mode_routes_kind_to_facet_call(self):
        """``kind`` (render-call-only) reaches ``cleo.facet``, not the ctor.

        Test scenario:
            The facet branch in ``render_array`` calls
            ``cleo.facet(**facet_kwargs, **render_kwargs)``. ``kind`` is
            a render-call-only kwarg, so it must surface inside the
            facet call's kwargs while ``cmap`` (constructor bucket)
            lands on ``__init__``.
        """
        fake_cls, ctor, _, _, facet, _ = self._capture_calls()
        rng = np.random.default_rng(404)
        arr = rng.random((3, 4, 4)).astype("float32")
        with patch("cleopatra.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="facet",
                facet_kwargs={"col": "time", "col_coords": [0, 1, 2]},
                cmap="magma",
                kind="contourf",
            )
        assert "kind" in facet, f"`kind` should reach cleo.facet; facet kwargs={facet}"
        assert "cmap" in ctor, f"`cmap` should remain on the constructor; ctor={ctor}"
        assert (
            facet.get("col") == "time"
        ), f"facet_kwargs must reach cleo.facet via merge; got {facet}"

    def test_split_is_driven_by_option_keys(self):
        """The ctor/render split comes from ``ArrayGlyph.option_keys()``.

        Test scenario:
            A constructor option declared by cleopatra (``add_colorbar``)
            must route to ``__init__`` because it is in ``option_keys()``,
            so the split tracks cleopatra automatically instead of a
            hand-maintained list. ``kind`` is the documented exception: it
            *is* in ``option_keys()`` yet is force-routed to the render
            call (it is an explicit ``plot`` param read from the signature,
            not from ``default_options`` — routing it to the ctor would
            pin every render to ``kind="auto"``).
        """
        assert "add_colorbar" in ArrayGlyph.option_keys()
        assert "kind" in ArrayGlyph.option_keys()
        fake_cls, ctor, plot, _, _, _ = self._capture_calls()
        rng = np.random.default_rng(505)
        arr = rng.random((4, 4)).astype("float32")
        with patch("cleopatra.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                add_colorbar=False,
                kind="imshow",
            )
        assert (
            "add_colorbar" in ctor and "add_colorbar" not in plot
        ), f"`add_colorbar` is an option_keys() ctor option; ctor={ctor}"
        assert (
            "kind" in plot and "kind" not in ctor
        ), f"`kind` must be force-routed to the render call; plot={plot}"

    @pytest.mark.plot
    def test_kind_contourf_reaches_plot_not_clobbered(self):
        """Regression: ``kind="contourf"`` renders as contourf, not ``"auto"``.

        Test scenario:
            ``kind`` is in ``option_keys()`` yet ``ArrayGlyph.plot()``
            unconditionally rewrites ``default_options["kind"]`` with its own
            arg. The ``RENDER_ONLY_OVERRIDES`` set forces ``kind`` onto the
            render call so it is not clobbered back to ``"auto"``. Uses the
            real ``ArrayGlyph`` (not the fake) so the clobber path is actually
            exercised; the returned glyph must report ``"contourf"``.
        """
        rng = np.random.default_rng(909)
        arr = rng.random((5, 5)).astype("float32")
        glyph = render_array(
            arr=arr, extent=[0.0, 0.0, 1.0, 1.0], mode="plot", kind="contourf"
        )
        assert glyph.default_options["kind"] == "contourf", (
            "kind must reach ArrayGlyph.plot() and not be clobbered to 'auto'; "
            f"got {glyph.default_options.get('kind')!r}"
        )

    def test_invalid_kwarg_surfaces_cleopatra_valueerror(self):
        """An unknown kwarg is not swallowed — cleopatra raises ``ValueError``.

        Test scenario:
            A key absent from ``option_keys()`` (here ``bogus``) lands in
            ``render_kwargs`` and reaches the real ``ArrayGlyph.plot``,
            which rejects it. The routing must not silently drop unknown
            keys; it must defer to cleopatra's validation.
        """
        rng = np.random.default_rng(606)
        arr = rng.random((4, 4)).astype("float32")
        with pytest.raises(ValueError):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                bogus=1,
            )


class TestMeshRenderHelper:
    """Direct unit tests for the N-6 ``mesh_render`` sibling helper."""

    def test_mesh_render_basemap_without_epsg_raises(self):
        """``basemap=True`` + ``basemap_epsg=None`` raises ValueError.

        Test scenario:
            The mesh helper mirrors :func:`render_array`'s precondition:
            requesting a basemap without an EPSG must fail fast with a
            "CRS" hint before any rendering happens.
        """
        from pyramids.dataset._plot_helpers import mesh_render

        with pytest.raises(ValueError, match=r"CRS"):
            mesh_render(
                mesh=object(),
                data=np.array([1.0]),
                basemap=True,
                basemap_epsg=None,
            )

    def test_mesh_render_basemap_false_skips_add_basemap(self):
        """``basemap=False`` short-circuits before ``add_basemap`` is called.

        Test scenario:
            Patch ``plot_mesh_data`` to return a sentinel and patch the
            basemap module's ``add_basemap`` to record calls. With
            ``basemap=None`` the helper must return the sentinel without
            ever calling ``add_basemap``.
        """
        from pyramids.dataset._plot_helpers import mesh_render

        sentinel = object()
        with patch(
            "pyramids.netcdf.ugrid.plot.plot_mesh_data",
            return_value=sentinel,
        ):
            with patch(
                "pyramids.basemap.basemap.add_basemap",
            ) as mock_add:
                result = mesh_render(
                    mesh=object(),
                    data=np.array([1.0]),
                    location="face",
                )
        assert (
            result is sentinel
        ), f"mesh_render must return plot_mesh_data's result; got {result!r}"
        mock_add.assert_not_called()

    def test_mesh_render_forwards_kwargs_to_plot_mesh_data(self):
        """Kwargs (``cmap``, ``vmin``, ``vmax``, ``title``) flow through.

        Test scenario:
            ``mesh_render`` is a thin dispatcher — every kwarg except
            ``basemap``/``basemap_epsg`` is forwarded to
            ``plot_mesh_data``. Capture the call and verify each kwarg
            is preserved with the same value the caller supplied.
        """
        from pyramids.dataset._plot_helpers import mesh_render

        captured: dict = {}

        def _fake_plot(mesh, data, **kwargs):
            captured["mesh"] = mesh
            captured["data"] = data
            captured.update(kwargs)
            return "glyph"

        with patch(
            "pyramids.netcdf.ugrid.plot.plot_mesh_data",
            side_effect=_fake_plot,
        ):
            mesh = object()
            data = np.array([1.0, 2.0])
            mesh_render(
                mesh=mesh,
                data=data,
                location="node",
                cmap="plasma",
                vmin=0.0,
                vmax=10.0,
                title="t",
            )
        assert captured.get("location") == "node"
        assert captured.get("cmap") == "plasma"
        assert captured.get("vmin") == 0.0
        assert captured.get("vmax") == 10.0
        assert captured.get("title") == "t"

    def test_mesh_render_basemap_triggers_add_basemap_with_crs(self):
        """``basemap=True`` calls ``add_basemap`` with the supplied EPSG.

        Test scenario:
            With ``basemap=True`` and ``basemap_epsg=3857`` the helper
            must call ``pyramids.basemap.basemap.add_basemap`` once,
            forwarding the EPSG as the ``crs`` kwarg. The ax it picks up
            comes from the returned glyph's ``ax`` attribute, mirroring
            the raster path's behaviour.
        """
        from pyramids.dataset._plot_helpers import mesh_render

        fake_glyph = type("G", (), {"ax": object()})()
        with patch(
            "pyramids.netcdf.ugrid.plot.plot_mesh_data",
            return_value=fake_glyph,
        ):
            with patch(
                "pyramids.basemap.basemap.add_basemap",
            ) as mock_add:
                mesh_render(
                    mesh=object(),
                    data=np.array([1.0]),
                    basemap=True,
                    basemap_epsg=3857,
                )
        mock_add.assert_called_once()
        kwargs = mock_add.call_args.kwargs
        assert kwargs.get("crs") == 3857, f"`crs` must equal basemap_epsg; got {kwargs}"
        assert (
            kwargs.get("source") is None
        ), f"`source` should be None when basemap=True (no provider); got {kwargs}"

    def test_mesh_render_basemap_string_passes_source(self):
        """``basemap='CartoDB.Positron'`` forwards the string as ``source=``.

        Test scenario:
            The mesh helper mirrors the raster path: a basemap string is
            forwarded as the contextily provider name via the ``source``
            kwarg. Boolean ``True`` passes ``source=None`` (already
            covered above); the string variant is verified here.
        """
        from pyramids.dataset._plot_helpers import mesh_render

        fake_glyph = type("G", (), {"ax": object()})()
        with patch(
            "pyramids.netcdf.ugrid.plot.plot_mesh_data",
            return_value=fake_glyph,
        ):
            with patch(
                "pyramids.basemap.basemap.add_basemap",
            ) as mock_add:
                mesh_render(
                    mesh=object(),
                    data=np.array([1.0]),
                    basemap="CartoDB.Positron",
                    basemap_epsg=4326,
                )
        kwargs = mock_add.call_args.kwargs
        assert (
            kwargs.get("source") == "CartoDB.Positron"
        ), f"`source` must equal the basemap string; got {kwargs}"


class TestPR6CleanupGrepGuards:
    """Belt-and-braces D-5 regression scan complementing TestPlotPR6Cleanups."""

    def test_import_cleopatra_definition_still_exists(self):
        """The legacy ``import_cleopatra`` function stays importable.

        Test scenario:
            D-5 consolidated the call sites onto ``require_cleopatra``
            but kept the original ``import_cleopatra`` symbol around for
            downstream consumers. The PR-6 docstring explicitly promises
            back-compat — this test asserts the symbol is still
            importable and callable.
        """
        from pyramids.base._utils import import_cleopatra

        import_cleopatra("legacy back-compat call")

    def test_import_cleopatra_not_called_outside_definition(self):
        """Grep-style: ``import_cleopatra(`` only fires inside its own module.

        Test scenario:
            Sweep every ``.py`` file under ``src/pyramids`` and confirm
            no module outside ``base/_utils.py`` contains a call-style
            usage of ``import_cleopatra(...)``. The function definition
            and its internal forwarding call live in ``_utils.py``; any
            *other* module hit means a caller slipped through D-5.
        """
        import re
        from pathlib import Path

        repo = Path(__file__).parents[2]
        src = repo / "src" / "pyramids"
        pattern = re.compile(r"import_cleopatra\s*\(")
        offenders: list[str] = []
        for path in src.rglob("*.py"):
            if path.name == "_utils.py":
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, raw in enumerate(text.splitlines(), start=1):
                if not pattern.search(raw):
                    continue
                stripped = raw.strip()
                if stripped.startswith("#"):
                    continue
                rel = path.relative_to(repo).as_posix()
                offenders.append(f"{rel}:{lineno}: {stripped}")
        assert not offenders, (
            "D-5 — `import_cleopatra(` must only appear in "
            "`base/_utils.py`. Use `require_cleopatra()` elsewhere. "
            "Offenders:\n  " + "\n  ".join(offenders)
        )

    def test_require_cleopatra_used_by_plot_modules(self):
        """Plot-adjacent modules import ``require_cleopatra`` (positive check).

        Test scenario:
            The D-5 cleanup replaced the per-module ``import_cleopatra``
            calls with ``require_cleopatra``. Confirm each plot-adjacent
            module imports the new helper so a future regression that
            silently drops the guard surfaces here, not at runtime.
        """
        from pathlib import Path

        repo = Path(__file__).parents[2]
        targets = [
            "src/pyramids/dataset/_plot_helpers.py",
            "src/pyramids/dataset/engines/analysis.py",
            "src/pyramids/dataset/engines/bands.py",
            "src/pyramids/netcdf/ugrid/plot.py",
        ]
        missing: list[str] = []
        for rel in targets:
            text = (repo / rel).read_text(encoding="utf-8")
            if "require_cleopatra" not in text:
                missing.append(rel)
        assert (
            not missing
        ), f"Modules missing `require_cleopatra` import/usage: {missing}"
