"""The ``pyramids.plot_backends`` seam and the ``plot(backend=…)`` facade dispatch.

No renderer is installed in this suite — not cleopatra, not digitalearth. The seam is exercised
with in-memory entry points and a hand-rolled stand-in for ``digitalearth.quickmap``, reusing the
same discovery fixtures as the raster/vector driver tests, so a bare pyramids install proves the
whole path (discovery, the optional-import fallback, the branded not-installed error, and the
matplotlib-only guard) without pulling a plotting stack into the test environment.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from pyramids.base import _plot_backends
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._plot_backends import (
    clear_plot_backend_cache,
    forward_kwargs,
    is_colorbar_spec,
    plot_backends,
    reject_mpl_only_params,
    render_via_backend,
    resolve_plot_backend,
)
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from tests._driver_discovery_fixtures import make_publish_entry_points

pytestmark = pytest.mark.core

GEO = GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326)


class _Recorder:
    """A render-backend stand-in that records the one call it receives.

    Instances are callable with the registered-backend shape ``fn(obj, *, backend, **kwargs)`` and
    return a sentinel the tests assert identity on, so dispatch can be checked without a renderer.
    """

    def __init__(self) -> None:
        """Start with no recorded call."""
        self.obj: Any = None
        self.backend: str | None = None
        self.kwargs: dict[str, Any] = {}
        self.sentinel = object()

    def __call__(self, obj: Any, *, backend: str, **kwargs: Any) -> Any:
        """Record ``obj`` / ``backend`` / ``kwargs`` and return the sentinel."""
        self.obj = obj
        self.backend = backend
        self.kwargs = kwargs
        return self.sentinel


@pytest.fixture(autouse=True)
def _reset_cache() -> Any:
    """Clear the backend cache around every test so discovery never leaks between cases."""
    clear_plot_backend_cache()
    yield
    clear_plot_backend_cache()


@pytest.fixture
def publish_backends(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Publish in-memory ``pyramids.plot_backends`` entry points, then clean up the fakes."""
    publish, published = make_publish_entry_points(
        monkeypatch, _plot_backends, "_fake_plot_backend_"
    )
    yield publish
    for attribute in published:
        monkeypatch.delattr(_plot_backends, attribute, raising=False)


def _absent_digitalearth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the resolver see digitalearth as not installed."""
    monkeypatch.setattr(_plot_backends, "extra_missing", lambda name: True)


def _present_digitalearth(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    """Make the resolver import ``module`` as digitalearth."""
    monkeypatch.setattr(_plot_backends, "extra_missing", lambda name: False)
    monkeypatch.setattr(_plot_backends, "import_module", lambda name: module)


# --------------------------------------------------------------------------- discovery + cache


def test_bare_install_registers_no_backend() -> None:
    """A bare install's entry-point group is empty, so the registry is empty."""
    assert plot_backends() == {}


def test_entry_point_is_discovered(publish_backends: Any) -> None:
    """A published entry point is discovered and keyed by its entry-point name."""
    recorder = _Recorder()
    publish_backends(("web", recorder))
    assert plot_backends() == {"web": recorder}
    assert resolve_plot_backend("web") is recorder


def test_render_via_backend_forwards_obj_and_kwargs(publish_backends: Any) -> None:
    """``render_via_backend`` calls the resolved backend with ``(obj, backend=…, **forward)``."""
    recorder = _Recorder()
    publish_backends(("web", recorder))
    marker = object()
    result = render_via_backend(marker, "web", title="t", basemap=True)
    assert result is recorder.sentinel
    assert recorder.obj is marker
    assert recorder.backend == "web"
    assert recorder.kwargs == {"title": "t", "basemap": True}


def test_duplicate_name_keeps_first_and_warns(publish_backends: Any) -> None:
    """A second entry point with the same name is skipped with a ``RuntimeWarning``."""
    first, second = _Recorder(), _Recorder()
    publish_backends(("web", first), ("web", second))
    with pytest.warns(RuntimeWarning, match="registered twice"):
        registry = plot_backends()
    assert registry["web"] is first


def test_non_callable_entry_point_is_skipped(publish_backends: Any) -> None:
    """An entry point that loads a non-callable is skipped with a ``RuntimeWarning``."""
    publish_backends(("web", "not-callable"))
    with pytest.warns(RuntimeWarning, match="not callable"):
        assert plot_backends() == {}


# --------------------------------------------------------------------------- digitalearth fallback


def test_fallback_to_digitalearth_quickmap(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no entry point but digitalearth present, the resolver adapts ``quickmap``."""
    seen: dict[str, Any] = {}

    def quickmap(obj: Any, *, backend: str, **kwargs: Any) -> str:
        seen.update(obj=obj, backend=backend, kwargs=kwargs)
        return "rendered"

    _present_digitalearth(monkeypatch, types.SimpleNamespace(quickmap=quickmap))
    marker = object()
    result = render_via_backend(marker, "3d", z_exaggeration=2)
    assert result == "rendered"
    assert seen == {"obj": marker, "backend": "3d", "kwargs": {"z_exaggeration": 2}}


def test_entry_point_wins_over_digitalearth(
    publish_backends: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered entry point is preferred; digitalearth is never imported."""

    def _boom(name: str) -> Any:
        raise AssertionError(
            "digitalearth must not be imported when an entry point matches"
        )

    monkeypatch.setattr(_plot_backends, "extra_missing", lambda name: False)
    monkeypatch.setattr(_plot_backends, "import_module", _boom)
    recorder = _Recorder()
    publish_backends(("3d", recorder))
    assert resolve_plot_backend("3d") is recorder


def test_too_old_digitalearth_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A digitalearth without ``quickmap`` raises the branded upgrade hint."""
    _present_digitalearth(monkeypatch, types.SimpleNamespace())
    with pytest.raises(OptionalPackageDoesNotExist, match="too old"):
        resolve_plot_backend("3d")


# --------------------------------------------------------------------------- not-installed error


def test_unknown_backend_without_digitalearth(monkeypatch: pytest.MonkeyPatch) -> None:
    """No entry point and no digitalearth raises a hint naming the backend and the extra."""
    _absent_digitalearth(monkeypatch)
    with pytest.raises(OptionalPackageDoesNotExist) as excinfo:
        resolve_plot_backend("web")
    message = str(excinfo.value)
    assert "web" in message
    assert "digitalearth[web]" in message


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("interactive", "interactive"),
        ("3d", "3d"),
        ("web", "web"),
        ("mystery", "interactive"),
    ],
)
def test_install_hint_maps_backend_to_extra(
    name: str, extra: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The install hint maps each backend name to its digitalearth extra."""
    _absent_digitalearth(monkeypatch)
    with pytest.raises(OptionalPackageDoesNotExist, match=rf"digitalearth\[{extra}\]"):
        resolve_plot_backend(name)


# --------------------------------------------------------------------------- small helpers


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), (True, False), (False, False), (object(), True)],
)
def test_is_colorbar_spec(value: Any, expected: bool) -> None:
    """A ``ColorBar`` spec is anything that is neither ``None`` nor a ``bool``."""
    assert is_colorbar_spec(value) is expected


def test_forward_kwargs_drops_none() -> None:
    """``forward_kwargs`` keeps set values (including falsy non-``None``) and drops ``None``."""
    assert forward_kwargs(a=1, b=None, c=0, d="x", e=False) == {
        "a": 1,
        "c": 0,
        "d": "x",
        "e": False,
    }


def test_reject_mpl_only_params_raises_for_set_values() -> None:
    """A non-``None`` matplotlib-only param on a non-matplotlib backend is a ``ValueError``."""
    with pytest.raises(ValueError, match=r"\['color'\].*backend='web'"):
        reject_mpl_only_params("web", color=object(), contour=None)


def test_reject_mpl_only_params_noop_when_all_none() -> None:
    """All-``None`` candidates pass the guard silently."""
    assert reject_mpl_only_params("web", color=None, points=None, cells=None) is None


# --------------------------------------------------------------------------- facade dispatch


def _single_band_dataset() -> Dataset:
    """A tiny in-memory single-band raster for the facade dispatch tests."""
    return Dataset.from_array(
        np.arange(16, dtype=np.float32).reshape(4, 4), geo_ref=GEO
    )


def test_dataset_plot_routes_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Dataset.plot(backend=…)`` forwards the dataset and the agnostic kwargs to the backend."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    dataset = _single_band_dataset()
    result = dataset.plot(backend="web", title="DEM", basemap=True, cmap="viridis")
    assert result is recorder.sentinel
    assert recorder.obj is dataset
    assert recorder.backend == "web"
    assert recorder.kwargs["title"] == "DEM"
    assert recorder.kwargs["basemap"] is True
    assert recorder.kwargs["cmap"] == "viridis"


def test_dataset_plot_rejects_matplotlib_only_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleopatra-typed render group with a non-matplotlib backend raises before dispatch."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    dataset = _single_band_dataset()
    with pytest.raises(ValueError, match="only apply to backend='matplotlib'"):
        dataset.plot(backend="web", color=object())
    assert recorder.backend is None  # never dispatched


def test_dataset_animate_routes_with_animate_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Dataset.animate(backend=…)`` routes to the backend with ``animate=True``."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    dataset = _single_band_dataset()
    dataset.animate(backend="web", interval=120)
    assert recorder.backend == "web"
    assert recorder.kwargs["animate"] is True
    assert recorder.kwargs["interval"] == 120
