"""Plot tests: Phase-3 cross-cutting behaviour and PR-6 cleanup guards."""

from unittest.mock import patch

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset._plot_helpers import render_array
from pyramids.dataset.engines import Analysis
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


class TestPlotPhase3CrossCutting:
    """Cross-cutting regressions for the PR-4 D-2 refactor.

    The shared ``render_array`` helper now backs ``Analysis.plot``,
    ``DatasetCollection.plot``, and the NetCDF facet path. These tests
    pin the public-contract invariants: types, return values, and
    behavioural equivalence vs. the engine directly.
    """

    @pytest.fixture(scope="function")
    def random_single_band_for_plot(self):
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
    def test_dataset_plot_returns_array_glyph_post_refactor(
        self, random_single_band_for_plot
    ):
        """`Dataset.plot()` still returns an ArrayGlyph after D-2 collapse.

        Test scenario:
            The D-2 refactor routes ``Dataset.plot`` through the
            shared ``render_array`` helper. The public contract is
            unchanged: the call must return a cleopatra ArrayGlyph
            instance. Pre-refactor parity is required for downstream
            callers that chain visual customisations.
        """
        result = random_single_band_for_plot.plot()
        assert isinstance(result, ArrayGlyph), (
            f"Dataset.plot() must return ArrayGlyph after D-2, got: {type(result).__name__}"
        )

    @pytest.mark.plot
    def test_analysis_plot_returns_array_glyph_post_refactor(
        self, random_single_band_for_plot
    ):
        """`Analysis.plot(band=N)` still returns an ArrayGlyph after D-2.

        Test scenario:
            ``Analysis.plot`` is the engine; the D-2 collapse pushes
            cleo construction into ``render_array``. A direct engine
            call must still produce an ArrayGlyph (this is the
            existing public engine API).
        """
        result = random_single_band_for_plot.analysis.plot(band=0)
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
    def test_render_array_direct_call_matches_analysis_plot(
        self, random_single_band_for_plot
    ):
        """Calling `render_array(mode="plot")` directly produces the same array.

        Test scenario:
            The shared helper is module-private but the contract is
            stable: calling it with the same array + extent that
            ``Analysis.plot`` would compute internally yields an
            ArrayGlyph wrapping the same data. This guards against a
            regression where the engine and helper diverge on how the
            data array is reshaped.
        """
        arr = random_single_band_for_plot.read_array(band=0)
        bbox = random_single_band_for_plot.bbox
        helper_glyph = render_array(
            arr=arr,
            extent=bbox,
            exclude_value=[np.nan],
            mode="plot",
        )
        engine_glyph = random_single_band_for_plot.analysis.plot(band=0)
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

        repo = Path(__file__).parents[3]
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

        assert callable(require_cleopatra), (
            "require_cleopatra must be importable and callable"
        )
        require_cleopatra()
        require_cleopatra("optional override message")

    def test_render_array_kwargs_not_double_forwarded(self):
        """D-4 — kwargs reach the constructor xor the render call.

        Test scenario:
            The render-call-only kwargs (anything outside
            ``ArrayGlyph.option_keys()``, such as ``points`` and ``kind``) must
            reach ``ArrayGlyph.plot``, not the constructor. Every
            other kwarg must land on the constructor exactly once.
            We patch both call sites and inspect the recorded kwargs.

            The loose point-styling names used below are superseded in cleopatra
            0.26 by ``PointOverlay``; they remain accepted upstream and serve here
            only as vehicles for the routing invariant.
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

        assert "cmap" in ctor_seen, (
            f"Ctor should own `cmap`; got ctor={ctor_seen}, plot={plot_seen}"
        )
        assert "cmap" not in plot_seen, (
            f"`cmap` must not double-forward; plot kwargs={plot_seen}"
        )
        assert "kind" in plot_seen, (
            f"`kind` should reach cleo.plot; got plot={plot_seen}"
        )
        assert "kind" not in ctor_seen, (
            f"`kind` should not be on the constructor; ctor={ctor_seen}"
        )


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

        assert callable(import_cleopatra), (
            "the legacy import_cleopatra symbol must stay importable"
        )
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

        repo = Path(__file__).parents[3]
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

        repo = Path(__file__).parents[3]
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
        assert not missing, (
            f"Modules missing `require_cleopatra` import/usage: {missing}"
        )
