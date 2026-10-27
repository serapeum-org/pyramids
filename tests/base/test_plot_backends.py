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

import geopandas as gpd
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
from pyramids.dataset.collection import DatasetCollection
from pyramids.feature import FeatureCollection
from pyramids.netcdf import FacetSpec, NetCDF
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable
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
        registry = plot_backends()
    assert registry == {}


def test_load_skips_entry_point_that_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry point whose ``load()`` raises is skipped with a ``RuntimeWarning``, not propagated."""

    class _Boom:
        name = "boom"
        value = "your_pkg:boom"

        def load(self) -> Any:
            raise RuntimeError("kaboom")

    boom = _Boom()
    with pytest.warns(RuntimeWarning, match="could not be loaded"):
        loaded = _plot_backends._load(boom)
    assert loaded is None


def test_reentrant_discovery_returns_partial_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thread inside discovery sees the backends resolved so far, not a second discovery."""
    partial = {"web": _Recorder()}
    monkeypatch.setattr(_plot_backends._IN_PROGRESS, "resolved", partial, raising=False)
    result = plot_backends()
    assert result == partial
    assert (
        result is not partial
    )  # a copy, so a caller cannot mutate the in-progress mapping


def test_cache_double_check_prefers_value_set_during_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the cache is filled while discovery runs, the published value wins over the fresh scan."""
    existing = {"web": _Recorder()}

    def _fake_discover() -> dict[str, Any]:
        _plot_backends._CACHE = existing
        return {"other": _Recorder()}

    monkeypatch.setattr(_plot_backends, "_CACHE", None)
    monkeypatch.setattr(_plot_backends, "_discover", _fake_discover)
    assert plot_backends() == existing


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


def test_resolve_prefers_registered_even_when_falsy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered backend wins over the fallback even if the callable is falsy (``__bool__``)."""

    class _FalsyBackend:
        def __bool__(self) -> bool:
            return False

        def __call__(self, obj: Any, *, backend: str, **kwargs: Any) -> str:
            return "falsy-rendered"

    falsy = _FalsyBackend()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": falsy})
    monkeypatch.setattr(_plot_backends, "extra_missing", lambda name: True)
    assert resolve_plot_backend("web") is falsy


def test_too_old_digitalearth_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A digitalearth without ``quickmap`` raises the branded upgrade hint."""
    _present_digitalearth(monkeypatch, types.SimpleNamespace())
    with pytest.raises(OptionalPackageDoesNotExist, match="too old"):
        resolve_plot_backend("3d")


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        (True, False),
        (False, False),
        (np.True_, False),
        (np.False_, False),
        (object(), True),
    ],
)
def test_is_colorbar_spec(value: Any, expected: bool) -> None:
    """A ``ColorBar`` spec is anything that is neither ``None`` nor a Python/NumPy ``bool``."""
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


def test_dataset_plot_rejects_overview_on_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``overview`` is a matplotlib read directive, rejected rather than silently dropped."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    dataset = _single_band_dataset()
    with pytest.raises(ValueError, match="only apply to backend='matplotlib'"):
        dataset.plot(backend="web", overview=True, overview_index=2)
    assert recorder.backend is None


def test_dataset_plot_default_overview_is_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``overview=False`` / ``overview_index=0`` defaults do not trip the guard."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    dataset = _single_band_dataset()
    dataset.plot(backend="web", title="t")
    assert recorder.backend == "web"
    assert "overview" not in recorder.kwargs


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


def _netcdf_container() -> NetCDF:
    """A tiny in-memory single-variable NetCDF container for the facade dispatch tests."""
    return NetCDF.from_array(
        np.arange(16, dtype=np.float32).reshape(4, 4), geo_ref=GEO, variable_name="t2m"
    )


def test_netcdf_plot_routes_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``NetCDF.plot(backend=…)`` forwards the container, the variable and the agnostic kwargs."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    netcdf = _netcdf_container()
    netcdf.plot(variable="t2m", backend="web", title="t", basemap=True)
    assert recorder.obj is netcdf
    assert recorder.backend == "web"
    assert recorder.kwargs["variable"] == "t2m"
    assert recorder.kwargs["title"] == "t"
    assert recorder.kwargs["basemap"] is True


def test_netcdf_plot_rejects_facet_on_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A matplotlib-only ``facet`` with a non-matplotlib backend raises before dispatch."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    netcdf = _netcdf_container()
    facet = FacetSpec(col="time")
    with pytest.raises(ValueError, match="only apply to backend='matplotlib'"):
        netcdf.plot(variable="t2m", backend="web", facet=facet)
    assert recorder.backend is None  # never dispatched


def test_netcdf_plot_forwards_exclude_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """``exclude_value`` reaches the backend (parity with ``Dataset.plot``), not silently dropped."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    netcdf = _netcdf_container()
    netcdf.plot(variable="t2m", backend="web", exclude_value=-9999)
    assert recorder.kwargs["exclude_value"] == -9999


def test_netcdf_plot_rejects_chunks_on_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``chunks`` is a matplotlib read directive, rejected rather than silently dropped."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    netcdf = _netcdf_container()
    with pytest.raises(ValueError, match="only apply to backend='matplotlib'"):
        netcdf.plot(variable="t2m", backend="web", chunks={"time": 1})
    assert recorder.backend is None


def _feature_collection() -> FeatureCollection:
    """A tiny two-point in-memory FeatureCollection for the facade dispatch tests."""
    frame = gpd.GeoDataFrame(
        {"v": [1.0, 2.0]},
        geometry=gpd.points_from_xy([0.0, 1.0], [0.0, 1.0]),
        crs="EPSG:4326",
    )
    return FeatureCollection(frame)


def test_feature_plot_routes_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``FeatureCollection.plot(backend=…)`` forwards the collection and the column/title."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    features = _feature_collection()
    features.plot(column="v", backend="web", title="pts")
    assert recorder.obj is features
    assert recorder.backend == "web"
    assert recorder.kwargs["column"] == "v"
    assert recorder.kwargs["title"] == "pts"


def test_feature_plot_rejects_non_default_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-default ``engine`` with a non-matplotlib backend raises rather than being ignored."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    features = _feature_collection()
    with pytest.raises(ValueError, match="only apply to backend='matplotlib'"):
        features.plot(column="v", backend="web", engine="cleopatra")
    assert recorder.backend is None


def test_feature_plot_default_engine_is_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default ``engine='geopandas'`` does not trip the guard on a backend call."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    features = _feature_collection()
    features.plot(column="v", backend="web")
    assert recorder.backend == "web"


def _ugrid_dataset() -> UgridDataset:
    """A tiny in-memory 2-face triangular UgridDataset for the facade dispatch tests."""
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 0.5, 1.5]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )
    var = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=(2,),
        _data=np.array([10.0, 20.0]),
    )
    return UgridDataset(mesh=mesh, data_variables={"depth": var}, global_attributes={})


def test_ugrid_plot_drops_default_cmap_on_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``cmap='viridis'`` default is not forwarded, so the backend keeps its own default."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    mesh = _ugrid_dataset()
    mesh.plot("depth", backend="web")
    assert recorder.backend == "web"
    assert "cmap" not in recorder.kwargs


def test_ugrid_plot_forwards_explicit_cmap_on_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-set ``cmap`` is forwarded to the backend."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    mesh = _ugrid_dataset()
    mesh.plot("depth", backend="web", cmap="jet")
    assert recorder.kwargs["cmap"] == "jet"


def test_ugrid_plot_rejects_render_group_on_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleopatra render group on a non-matplotlib mesh backend raises before dispatch."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    mesh = _ugrid_dataset()
    with pytest.raises(ValueError, match="only apply to backend='matplotlib'"):
        mesh.plot("depth", backend="web", contour=object())
    assert recorder.backend is None


def test_ugrid_animate_routes_with_animate_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``UgridDataset.animate(backend=…)`` routes to the backend with ``animate=True``."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    mesh = _ugrid_dataset()
    mesh.animate("depth", backend="web", interval=150)
    assert recorder.backend == "web"
    assert recorder.kwargs["animate"] is True
    assert recorder.kwargs["interval"] == 150


def _single_dataset_collection() -> DatasetCollection:
    """A one-frame DatasetCollection whose single band can be dispatched to a backend."""
    return DatasetCollection(_single_band_dataset(), time_length=1)


def test_collection_plot_routes_with_animate_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DatasetCollection.plot(backend=…)`` routes with ``animate=True`` and forwards band/labels."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    collection = _single_dataset_collection()
    collection.plot(backend="web", band=1, animation_axis_values=["t0"])
    assert recorder.obj is collection
    assert recorder.backend == "web"
    assert recorder.kwargs["animate"] is True
    assert recorder.kwargs["band"] == 1
    assert recorder.kwargs["animation_axis_values"] == ["t0"]


def test_collection_plot_rejects_frame_label_on_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``frame_label`` (a cleopatra animation caption style) is rejected off matplotlib."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    collection = _single_dataset_collection()
    with pytest.raises(ValueError, match="only apply to backend='matplotlib'"):
        collection.plot(backend="web", frame_label=object())
    assert recorder.backend is None


def test_collection_animate_forwards_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DatasetCollection.animate`` forwards ``backend=`` through to ``plot``."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    collection = _single_dataset_collection()
    collection.animate(backend="web")
    assert recorder.backend == "web"
    assert recorder.kwargs["animate"] is True


def test_backend_forward_drops_non_none_signature_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A facade's non-``None`` default (band=0, interval=200) is dropped, not forced on the backend."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    _single_dataset_collection().plot(backend="web")
    assert "band" not in recorder.kwargs
    recorder.backend = None
    _single_band_dataset().animate(backend="web")
    assert "interval" not in recorder.kwargs
    recorder.backend = None
    _ugrid_dataset().animate("depth", backend="web")
    assert "interval" not in recorder.kwargs


def test_facade_forwards_bool_colorbar_but_rejects_colorbar_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bool ``colorbar`` is forwarded through a facade; a styled ``ColorBar`` instance is rejected."""
    recorder = _Recorder()
    monkeypatch.setattr(_plot_backends, "_CACHE", {"web": recorder})
    dataset = _single_band_dataset()
    dataset.plot(backend="web", colorbar=True)
    assert recorder.kwargs["colorbar"] is True
    recorder.backend = None
    with pytest.raises(ValueError, match="only apply to backend='matplotlib'"):
        dataset.plot(backend="web", colorbar=object())
    assert recorder.backend is None
