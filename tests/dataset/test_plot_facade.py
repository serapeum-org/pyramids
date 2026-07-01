"""Plot tests: the Dataset.plot facade and RGB options."""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.dataset.engines import Analysis

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
