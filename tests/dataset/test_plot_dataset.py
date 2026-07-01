"""Plot tests: Dataset.plot, DatasetCollection.plot, NetCDF-via-Dataset, and CRS stamps."""

import warnings
from unittest.mock import patch

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
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


class TestPlotDataSet:
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
    def test_rgba_four_channel_animation(self, tmp_path):
        """A four-index `rgb` composites an RGBA `(time, rows, cols, 4)` time-lapse.

        The alpha channel path is documented ("or four, with alpha"); this guards it
        against silent regressions in the per-frame compositing.
        """
        cube = self._rgb_cube(tmp_path, n_times=3, n_bands=4)
        glyph = cube.plot(rgb_options={"rgb": [0, 1, 2, 3], "percentile": 2})
        assert glyph.arr.shape == (3, 8, 8, 4), "RGBA stack keeps four channels"
        assert glyph.cbar is None, "true-colour frames carry no colorbar"

    @pytest.mark.plot
    @pytest.mark.parametrize("bad_rgb", [[0, 1], [0], []])
    def test_rgb_wrong_arity_raises(self, tmp_path, bad_rgb):
        """An `rgb` list that is not 3 or 4 indices raises a clear arity error.

        Args:
            bad_rgb: A malformed channel list (too few / empty) that must be rejected
                before reaching cleopatra (which would give a cryptic message, or
                ``max([])`` would raise on the empty list).
        """
        cube = self._rgb_cube(tmp_path, n_times=2, n_bands=4)
        with pytest.raises(ValueError, match=r"rgb must list 3 band indices"):
            cube.plot(rgb_options={"rgb": bad_rgb})

    @pytest.mark.plot
    def test_rgb_negative_index_raises(self, tmp_path):
        """A negative `rgb` index is rejected before band-count resolution.

        `max(rgb)` would otherwise underestimate the bands needed and let a
        wrap-around index through to a cryptic failure deep in cleopatra.
        """
        cube = self._rgb_cube(tmp_path, n_times=2, n_bands=4)
        with pytest.raises(ValueError, match=r"must be non-negative"):
            cube.plot(rgb_options={"rgb": [-1, 0, 1]})

    @pytest.mark.plot
    def test_rgb_with_exclude_value_warns(self, tmp_path):
        """Passing `exclude_value` alongside `rgb` warns that it is ignored."""
        cube = self._rgb_cube(tmp_path, n_times=2)
        with pytest.warns(UserWarning, match="exclude_value is ignored"):
            glyph = cube.plot(exclude_value=0, rgb_options={"rgb": [0, 1, 2]})
        assert glyph.arr.shape == (2, 8, 8, 3), "RGB stack still rendered"

    @pytest.mark.plot
    def test_rgb_options_wins_over_loose_kwarg(self, tmp_path):
        """On collision, `rgb_options` wins over the loose kwarg and still warns."""
        cube = self._rgb_cube(tmp_path, n_times=2, n_bands=4)
        with pytest.warns(DeprecationWarning, match="rgb_options` wins"):
            glyph = cube.plot(rgb=[1, 2, 3], rgb_options={"rgb": [0, 1, 2]})
        assert glyph.arr.shape == (2, 8, 8, 3), "grouped rgb composited every frame"


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


class TestPlotStampsGlyphCRS:
    """`Dataset.plot` stamps the dataset EPSG onto the returned glyph (issue #630)."""

    def test_dataset_plot_glyph_carries_epsg(self):
        """A plotted dataset's glyph exposes its CRS so reference layers need no crs=.

        Test scenario:
            Plot an EPSG:4326 single-band raster; the returned glyph's `crs` is 4326.
        """
        ds = Dataset.create_from_array(
            np.arange(12.0).reshape(3, 4), geo=(0.0, 1.0, 0, 3.0, 0, -1.0), epsg=4326
        )
        glyph = ds.plot()
        assert glyph.crs == 4326, f"expected glyph.crs == 4326, got {glyph.crs!r}"
