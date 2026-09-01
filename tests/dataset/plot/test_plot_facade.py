"""Plot tests: the Dataset.plot facade and RGB options."""

import inspect
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.engines import Analysis
from pyramids.netcdf import GeoReference, NetCDF

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.glyphs.gridded.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config
Config.set_matplotlib_backend("agg")

# Must follow the cleopatra importorskip above: matplotlib arrives via the [viz] extra, so
# importing it at the top of the file would error instead of skipping on a no-viz install.
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each plot test to bound memory.

    Plotting tests open figures via cleopatra/pyplot; without this teardown
    the suite accumulates them and matplotlib warns past 20 open figures.
    """
    yield
    plt.close("all")


class TestDatasetPlotFacade:
    """End-to-end facade tests for :meth:`Dataset.plot` after PR-1."""

    @pytest.mark.plot
    def test_returns_array_glyph(self):
        """:meth:`Dataset.plot` returns a cleopatra ``ArrayGlyph``.

        Test scenario:
            Calling the facade with a single-band dataset must return
            an instance of :class:`cleopatra.glyphs.gridded.array_glyph.ArrayGlyph` so
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
    @pytest.mark.parametrize("color_scale", [1, 2, 3, 4, 5, 0])
    def test_color_scale_integer_codes_rejected(self, color_scale):
        """Legacy integer ``color_scale`` codes raise a clear ``ValueError``.

        Args:
            color_scale: A legacy integer code that older releases accepted.

        Test scenario:
            cleopatra 0.30 moved the colour scale onto the ``color=ColorScaling`` group
            and rejects the loose ``color_scale`` *key* regardless of its value (an
            integer code, a string alias, or anything else), so pyramids surfaces that
            ``ValueError`` unchanged.
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

        assert result == "stub-glyph", (
            f"Facade must return engine output, got {result!r}"
        )
        assert mock_plot.call_count == 1
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs["band"] == 0, (
            f"Resolver should send band=0, got {call_kwargs.get('band')}"
        )
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
    def test_rgb_options_unknown_key_raises(self):
        """Unknown keys in `rgb_options` are rejected with a clear error."""
        rng = np.random.default_rng(15)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        with pytest.raises(ValueError, match=r"Unknown keys"):
            dataset.plot(rgb_options={"bogus": True})


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
        assert call_kwargs.get("rgb") is None, (
            f"Empty rgb_options should leave rgb=None, got: {call_kwargs.get('rgb')}"
        )
        assert call_kwargs.get("surface_reflectance") is None, (
            "Empty rgb_options should leave surface_reflectance=None"
        )
        deprecations = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecations, "Empty rgb_options must not emit DeprecationWarning"


class TestDatasetPlotFigAx:
    """`fig` / `ax` let several rasters share one figure (#1077)."""

    @pytest.fixture(scope="function")
    def dataset(self):
        """Build a small georeferenced single-band raster.

        Returns:
            Dataset: A 6x6 float32 dataset at EPSG:4326.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((6, 6)).astype("float32")
        return Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

    @pytest.mark.plot
    def test_draws_into_the_supplied_fig_and_ax(self, dataset):
        """Each panel of a subplots grid renders into the caller's figure and axes.

        Test scenario:
            Passing `fig` and `ax` must make the glyph adopt those exact objects and
            create no extra figure, which is what makes a side-by-side comparison grid
            possible.
        """
        fig, axes = plt.subplots(1, 3)
        opened_before = set(plt.get_fignums())

        for index, axis in enumerate(axes):
            glyph = dataset.plot(fig=fig, ax=axis, title=f"panel {index}")
            assert glyph.ax is axis, f"panel {index} must draw into the supplied axes"
            assert glyph.fig is fig, f"panel {index} must draw into the supplied figure"

        assert set(plt.get_fignums()) == opened_before, (
            "supplying fig/ax must not create additional figures"
        )

    @pytest.mark.plot
    def test_supplied_axes_keep_the_georeferenced_extent(self, dataset):
        """A panel drawn into caller axes still spans the raster's bbox, not pixel indices.

        Test scenario:
            The georeferenced extent is the reason to use `plot` over a raw `imshow`, so
            it must survive being drawn into externally-created axes.
        """
        fig, ax = plt.subplots()

        glyph = dataset.plot(fig=fig, ax=ax)

        xmin, _, xmax, _ = dataset.bbox
        assert glyph.ax.get_xlim() == pytest.approx((xmin, xmax)), (
            "the panel must span the raster bbox, not the 0..ncols pixel range"
        )

    @pytest.mark.plot
    def test_facade_forwards_fig_and_ax_to_the_engine(self, dataset):
        """The facade passes both through to `Analysis.plot` as named arguments."""
        fig, ax = plt.subplots()

        with patch.object(type(dataset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub-glyph"
            dataset.plot(fig=fig, ax=ax)

        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs["fig"] is fig, "fig must reach the engine"
        assert call_kwargs["ax"] is ax, "ax must reach the engine"

    @pytest.mark.plot
    def test_ax_alone_is_honoured(self, dataset):
        """Supplying only `ax` is sufficient — the panel adopts that axes and its figure.

        Test scenario:
            An axes already carries its figure, so `ax` on its own is enough to compose
            into a caller-owned layout. (The mirror case, `fig` without `ax`, currently
            raises inside cleopatra — see serapeum-org/cleopatra#326 — so it is not
            exercised here.)
        """
        fig, ax = plt.subplots()

        glyph = dataset.plot(ax=ax)

        assert glyph.ax is ax, "the panel must bind to the supplied axes"
        assert glyph.fig is fig, "and pick up that axes' own figure"

    @pytest.mark.plot
    @pytest.mark.xfail(
        strict=True,
        raises=AttributeError,
        reason=(
            "serapeum-org/cleopatra#326: cleopatra never derives an axes from a bare "
            "figure, so it writes the projection marker onto ax=None. Remove this pin "
            "once that lands and assert the success path instead."
        ),
    )
    def test_fig_alone_is_not_yet_supported(self, dataset):
        """`fig` without `ax` raises inside cleopatra until #326 lands.

        Test scenario:
            An axes carries its figure but not the reverse, so cleopatra leaves `ax` as
            `None` and then dereferences it. This pin is strict: when cleopatra starts
            adding the subplot itself the test XPASSes and fails, prompting the update.
        """
        fig, _ = plt.subplots()

        glyph = dataset.plot(fig=fig)

        assert glyph.fig is fig, "cleopatra should adopt the supplied figure"


class TestNetCDFPlotFigAx:
    """`NetCDF.plot` renders a variable into the caller's figure and axes (#1077).

    The signature contract below only proves the pair is declared; these tests prove
    the facade actually threads it down to the shared render call, so several
    variables really can share one `plt.subplots` grid.
    """

    @pytest.fixture(scope="function")
    def netcdf(self):
        """Build a small in-memory 2-D NetCDF container holding one variable.

        Returns:
            NetCDF: A 5x5 float32 container whose only variable is ``t2m``.
        """
        rng = np.random.default_rng(1077)
        arr = rng.random((5, 5)).astype("float32")
        return NetCDF.create_from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0, 5.0, 0, -1.0), epsg=4326),
            variable_name="t2m",
        )

    @pytest.mark.plot
    def test_draws_into_the_supplied_fig_and_ax(self, netcdf):
        """A variable renders into the caller's figure and axes without opening another.

        Test scenario:
            `fig` / `ax` used to be dropped on the NetCDF path, so the panel silently
            landed in a brand-new figure. The glyph must now adopt both objects and the
            open-figure set must be unchanged.
        """
        fig, ax = plt.subplots()
        opened_before = set(plt.get_fignums())

        glyph = netcdf.plot(variable="t2m", fig=fig, ax=ax)

        assert glyph.ax is ax, "the variable must draw into the supplied axes"
        assert glyph.fig is fig, "the variable must draw into the supplied figure"
        assert set(plt.get_fignums()) == opened_before, (
            "supplying fig/ax must not create additional figures"
        )

    @pytest.mark.plot
    def test_forwards_fig_and_ax_to_the_engine(self, netcdf):
        """The facade hands both through to `Analysis.plot` as named arguments."""
        fig, ax = plt.subplots()
        variable = netcdf.get_variable("t2m")

        with patch.object(type(variable.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub-glyph"
            netcdf.plot(variable="t2m", fig=fig, ax=ax)

        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs["fig"] is fig, "fig must reach the engine"
        assert call_kwargs["ax"] is ax, "ax must reach the engine"

    @pytest.mark.plot
    def test_ax_alone_is_honoured(self, netcdf):
        """Supplying only `ax` is enough — the panel picks up that axes' own figure.

        Test scenario:
            The docstring promises `ax` on its own suffices because an axes already
            carries its figure; this pins that promise on the NetCDF facade.
        """
        fig, ax = plt.subplots()

        glyph = netcdf.plot(variable="t2m", ax=ax)

        assert glyph.ax is ax, "the panel must bind to the supplied axes"
        assert glyph.fig is fig, "and pick up that axes' own figure"

    @pytest.mark.plot
    def test_supplied_axes_keep_the_georeferenced_extent(self, netcdf):
        """A panel drawn into caller axes still spans the variable's bbox.

        Test scenario:
            Drawing into external axes must not degrade the plot into a raw `imshow`
            over pixel indices — the georeferenced extent is the point of `plot`.
        """
        fig, ax = plt.subplots()

        glyph = netcdf.plot(variable="t2m", fig=fig, ax=ax)

        xmin, _, xmax, _ = netcdf.get_variable("t2m").bbox
        assert glyph.ax.get_xlim() == pytest.approx((xmin, xmax)), (
            "the panel must span the variable bbox, not the 0..ncols pixel range"
        )


class TestDatasetCollectionPlotFigAx:
    """`DatasetCollection.plot` animates into the caller's figure and axes (#1077)."""

    @pytest.fixture(scope="function")
    def collection(self):
        """Build a 3-timestep in-memory collection of identical single-band rasters.

        Returns:
            DatasetCollection: Three co-registered 4x5 float32 timesteps at EPSG:4326.
        """
        rng = np.random.default_rng(1077)
        arr = rng.random((1, 4, 5)).astype("float32")
        source = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        return DatasetCollection.from_dataset(source, 3)

    @pytest.mark.plot
    def test_threads_fig_and_ax_into_the_render_request(self, collection):
        """Both land on the `RenderRequest` the collection builds for `render_array`.

        Test scenario:
            The collection does not call the engine — it assembles a `RenderRequest`
            directly — so the forwarding contract is asserted on that request object.
        """
        fig, ax = plt.subplots()

        with patch("pyramids.dataset.collection.render_array") as mock_render:
            collection.plot(band=0, fig=fig, ax=ax)

        request = mock_render.call_args.args[0]
        assert request.fig is fig, "fig must be threaded into the RenderRequest"
        assert request.ax is ax, "ax must be threaded into the RenderRequest"

    @pytest.mark.plot
    def test_defaults_leave_the_request_pair_unset(self, collection):
        """Omitting both keeps the request's `fig` / `ax` at `None`.

        Test scenario:
            The defaults must stay `None` rather than a stray figure, so cleopatra
            keeps creating the animation's own figure when the caller supplies nothing.
        """
        with patch("pyramids.dataset.collection.render_array") as mock_render:
            collection.plot(band=0)

        request = mock_render.call_args.args[0]
        assert request.fig is None, f"expected fig=None, got {request.fig!r}"
        assert request.ax is None, f"expected ax=None, got {request.ax!r}"

    @pytest.mark.plot
    def test_animates_into_the_supplied_fig_and_ax(self, collection):
        """The real animate path binds the glyph to the caller's figure and axes.

        Test scenario:
            End-to-end through cleopatra: the time-lapse must be able to sit in one
            panel of a caller-owned layout instead of always owning a whole figure.
        """
        fig, ax = plt.subplots()
        opened_before = set(plt.get_fignums())

        glyph = collection.plot(band=0, fig=fig, ax=ax)

        assert glyph.ax is ax, "the animation must draw into the supplied axes"
        assert glyph.fig is fig, "the animation must draw into the supplied figure"
        assert set(plt.get_fignums()) == opened_before, (
            "supplying fig/ax must not create additional figures"
        )

    @pytest.mark.plot
    def test_ax_alone_is_honoured(self, collection):
        """Supplying only `ax` is enough on the collection facade too."""
        fig, ax = plt.subplots()

        glyph = collection.plot(band=0, ax=ax)

        assert glyph.ax is ax, "the animation must bind to the supplied axes"
        assert glyph.fig is fig, "and pick up that axes' own figure"

    @pytest.fixture(scope="function")
    def rgb_collection(self):
        """Build a 3-timestep collection whose rasters carry three bands.

        Returns:
            DatasetCollection: Three co-registered 3-band 4x5 float32 timesteps.
        """
        rng = np.random.default_rng(326)
        arr = rng.random((3, 4, 5)).astype("float32")
        source = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        return DatasetCollection.from_dataset(source, 3)

    @pytest.mark.plot
    def test_rgb_time_lapse_threads_fig_and_ax_too(self, rgb_collection):
        """The RGB branch builds its own request and must thread the pair as well.

        Test scenario:
            `plot` returns from two places — a true-colour time-lapse and the
            single-band one. Only the second honoured `fig` / `ax` at first, so an RGB
            cube silently opened its own figure. Both branches must behave alike.
        """
        fig, ax = plt.subplots()

        with patch("pyramids.dataset.collection.render_array") as mock_render:
            rgb_collection.plot(rgb_options={"rgb": [2, 1, 0]}, fig=fig, ax=ax)

        request = mock_render.call_args.args[0]
        assert request.fig is fig, "the RGB request must carry the supplied figure"
        assert request.ax is ax, "the RGB request must carry the supplied axes"


class TestPlotFigAxSignatureContract:
    """Every `plot` facade must expose the same keyword-only `fig` / `ax` pair (#1077)."""

    @pytest.mark.plot
    @pytest.mark.parametrize(
        "plot_callable",
        [
            pytest.param(Dataset.plot, id="Dataset.plot"),
            pytest.param(Analysis.plot, id="Analysis.plot"),
            pytest.param(NetCDF.plot, id="NetCDF.plot"),
            pytest.param(DatasetCollection.plot, id="DatasetCollection.plot"),
        ],
    )
    @pytest.mark.parametrize("name", ["fig", "ax"])
    def test_declared_as_a_keyword_only_parameter(self, plot_callable, name):
        """`fig` and `ax` are declared parameters, not keys fished out of `**kwargs`.

        Args:
            plot_callable: One of the four public `plot` entry points.
            name: The parameter being asserted, `fig` or `ax`.

        Test scenario:
            Before #1077 the pair was undeclared — either popped from `**kwargs` or
            dropped entirely — which hid them from `help()` and made mypy reject the
            call. Declaring them keyword-only keeps every facade's contract identical
            and stops a positional argument from ever binding to them.
        """
        parameters = inspect.signature(plot_callable).parameters

        assert name in parameters, (
            f"{plot_callable.__qualname__} must declare `{name}` explicitly, "
            f"got {list(parameters)}"
        )
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{plot_callable.__qualname__}'s `{name}` must be keyword-only, "
            f"got {parameters[name].kind}"
        )
        assert parameters[name].default is None, (
            f"{plot_callable.__qualname__}'s `{name}` must default to None, "
            f"got {parameters[name].default!r}"
        )
