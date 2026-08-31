"""Plot tests: the Dataset.plot facade and RGB options."""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.dataset.engines import Analysis
from pyramids.base.georeference import GeoReference

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.glyphs.gridded.array_glyph", reason="cleopatra not installed"
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
        dataset = Dataset.from_array(
                      arr,
                      geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
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
        dataset = Dataset.from_array(
                      arr,
                      geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
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
        dataset = Dataset.from_array(
                      arr,
                      geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
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
        dataset = Dataset.from_array(
                      arr,
                      geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
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
        dataset = Dataset.from_array(
                      arr,
                      geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
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
        dataset = Dataset.from_array(
                      arr,
                      geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
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
        dataset = Dataset.from_array(
                      arr,
                      geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
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
        return Dataset.from_array(
                   arr,
                   geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
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
