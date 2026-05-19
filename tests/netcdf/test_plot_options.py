"""Tests for the frozen option dataclasses backing :meth:`NetCDF.plot`.

These tests need neither cleopatra nor matplotlib — the dataclass
construction / immutability / re-export checks are pure Python and the
behavioural ``NetCDF.plot`` checks engine-mock ``Analysis.plot`` — so
the whole module runs in the main (``-m "not plot"``) suite. The
end-to-end ``FacetSpec`` -> ``FacetGrid`` and ``selectors=None``
render-parity cases live in :mod:`tests.netcdf.test_plot`, which is
``plot``-marked.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

import pyramids
from pyramids.netcdf import ColourOpts, FacetSpec, Selectors
from pyramids.netcdf.netcdf import NetCDF


def _make_3d_nc(n_times: int = 4, rows: int = 5, cols: int = 5) -> NetCDF:
    """Build a 3-D (time, lat, lon) NetCDF container in memory.

    Args:
        n_times: Number of time steps.
        rows: Number of latitude rows.
        cols: Number of longitude columns.

    Returns:
        NetCDF: Root MDIM container with a single variable ``t2m``.
    """
    rng = np.random.default_rng(0)
    arr = rng.random((n_times, rows, cols)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, float(rows), 0, -1.0),
        epsg=4326,
        variable_name="t2m",
        extra_dim_name="time",
        extra_dim_values=list(range(n_times)),
    )
    return nc


class TestPlotOptionDataclasses:
    """Default construction and immutability of the three option bags."""

    def test_selectors_defaults_are_all_none(self):
        """``Selectors()`` constructs with every field ``None``.

        Test scenario:
            A bare ``Selectors`` must be a no-op selector bag so
            ``NetCDF.plot`` can substitute it for a missing argument
            without pinning anything.
        """
        sel = Selectors()
        assert sel.time is None
        assert sel.level is None
        assert sel.member is None
        assert sel.sel is None
        assert sel.isel is None

    def test_colour_opts_defaults(self):
        """``ColourOpts()`` constructs with cleopatra-default colour state.

        Test scenario:
            Every colour control is ``None`` except ``robust`` (``False``)
            and ``add_colorbar`` (``True``), matching the xarray-aligned
            defaults the old loose signature exposed.
        """
        colour = ColourOpts()
        assert colour.cmap is None
        assert colour.vmin is None
        assert colour.vmax is None
        assert colour.robust is False
        assert colour.levels is None
        assert colour.norm is None
        assert colour.center is None
        assert colour.extend is None
        assert colour.add_colorbar is True
        assert colour.cbar_kwargs is None

    def test_facet_spec_defaults_are_all_none(self):
        """``FacetSpec()`` constructs with every field ``None``.

        Test scenario:
            A bare ``FacetSpec`` (or one with both ``col`` and ``row``
            unset) routes ``NetCDF.plot`` to the single-panel static
            path, so the defaults must be ``None``.
        """
        facet = FacetSpec()
        assert facet.col is None
        assert facet.row is None
        assert facet.col_wrap is None

    @pytest.mark.parametrize(
        "factory, field, value",
        [
            (lambda: Selectors(time=0), "time", 1),
            (lambda: Selectors(), "sel", {"time": 0}),
            (lambda: ColourOpts(cmap="viridis"), "cmap", "magma"),
            (lambda: ColourOpts(), "add_colorbar", False),
            (lambda: FacetSpec(col="time"), "col", "level"),
            (lambda: FacetSpec(), "col_wrap", 3),
        ],
    )
    def test_dataclasses_are_frozen(self, factory, field, value):
        """Assigning to any field of any option bag raises ``FrozenInstanceError``.

        Args:
            factory: Builds a representative instance of the bag.
            field: Name of the field to attempt to mutate.
            value: A new value for the mutation attempt.

        Test scenario:
            All three bags are ``@dataclass(frozen=True)`` so a caller
            cannot mutate the option bag after handing it to
            ``NetCDF.plot``.
        """
        instance = factory()
        with pytest.raises(FrozenInstanceError):
            setattr(instance, field, value)


class TestPlotOptionReExports:
    """The three dataclasses are re-exported from the netcdf subpackage."""

    def test_not_re_exported_at_top_level(self):
        """``from pyramids import Selectors`` is intentionally NOT supported.

        Test scenario:
            Earlier versions exposed ``Selectors`` / ``ColourOpts`` /
            ``FacetSpec`` directly on ``pyramids`` — an inconsistency
            because every other class (``Dataset``, ``NetCDF``,
            ``FeatureCollection``, …) required its full subpackage
            path. The plot dataclasses now follow the same convention
            and only live under ``pyramids.netcdf``.
        """
        assert not hasattr(pyramids, "Selectors")
        assert not hasattr(pyramids, "ColourOpts")
        assert not hasattr(pyramids, "FacetSpec")
        assert {"Selectors", "ColourOpts", "FacetSpec"}.isdisjoint(
            set(pyramids.__all__)
        )

    def test_netcdf_subpackage_re_export(self):
        """The names are also exported from ``pyramids.netcdf``.

        Test scenario:
            ``pyramids.netcdf`` already re-exports ``NetCDF`` and the
            metadata models; the option bags join them so a single
            ``from pyramids.netcdf import ...`` covers the plot API.
        """
        ncpkg = pyramids.netcdf

        assert ncpkg.Selectors is Selectors
        assert ncpkg.ColourOpts is ColourOpts
        assert ncpkg.FacetSpec is FacetSpec
        assert {"Selectors", "ColourOpts", "FacetSpec"} <= set(ncpkg.__all__)


class TestPlotConsumesOptionDataclasses:
    """``NetCDF.plot`` substitutes empty bags for ``None`` and unpacks fields."""

    def test_selectors_none_is_treated_as_empty(self):
        """``plot(..., selectors=None)`` behaves like the no-selector call.

        Test scenario:
            A missing ``selectors=`` must be normalised to
            ``Selectors()`` before the body reads its fields, so the
            engine sees the same ``band`` regardless of which form the
            caller used.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            var.plot(variable="t2m", selectors=None)
            none_band = mock_plot.call_args.kwargs["band"]
            mock_plot.reset_mock()
            var.plot(variable="t2m")
            default_band = mock_plot.call_args.kwargs["band"]
        assert none_band == default_band

    def test_colour_field_forwarded_to_engine(self):
        """``colour=ColourOpts(cmap="viridis")`` forwards ``cmap`` flattened.

        Test scenario:
            The dataclass unpacking happens inside ``NetCDF.plot``; the
            engine still receives the flat ``cmap=`` kwarg.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            var.plot(variable="t2m", colour=ColourOpts(cmap="viridis"))
        assert mock_plot.call_args.kwargs["cmap"] == "viridis"

    def test_colour_none_is_treated_as_empty(self):
        """``plot(..., colour=None)`` does not forward any colour kwarg.

        Test scenario:
            A missing ``colour=`` normalises to ``ColourOpts()`` whose
            fields are all ``None``/default, so no colour kwarg leaks to
            the engine (only the always-present ``rgb=None`` default).
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            var.plot(variable="t2m", colour=None)
        forwarded = mock_plot.call_args.kwargs
        assert "cmap" not in forwarded
        assert "vmin" not in forwarded
        assert "robust" not in forwarded

    def test_facet_none_takes_static_path(self):
        """``plot(..., facet=None)`` does not build a facet stack.

        Test scenario:
            A missing ``facet=`` normalises to ``FacetSpec()`` (all
            ``None``), so the static single-panel path runs and no
            ``_facet_stack`` is forwarded to the engine.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            var.plot(variable="t2m", facet=None)
        assert "_facet_stack" not in mock_plot.call_args.kwargs
