"""Unit tests for curvilinear coordinate resolution in ``NetCDFPlot``.

Drives ``_resolve_curvilinear_coords`` in isolation with a duck-typed fake NetCDF slice — no
cleopatra, no sample files — so these run in the main (``not plot``) suite. The end-to-end
render path is covered by ``tests/netcdf/plot/test_plot_coords.py`` and the real-sample
curvilinear crops by ``tests/netcdf/samples/test_curvilinear_crop.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from pyramids.netcdf._plot import NetCDFPlot

_SHAPE = (5, 6)  # (rows, cols)


class _FakeNC:
    """Minimal duck-typed NetCDF slice exposing only what ``_resolve_curvilinear_coords`` reads."""

    def __init__(self, arrays, shape=_SHAPE, variable_attrs=None, curvilinear_coords=None):
        """Store the name -> coord-array map plus the slice shape and optional CF / crop state."""
        self._arrays = arrays
        self.shape = shape
        self._parent_nc = None
        self._variable_attrs = variable_attrs if variable_attrs is not None else {}
        self._source_var_name = "v"
        if curvilinear_coords is not None:
            self._curvilinear_coords = curvilinear_coords

    @property
    def variable_names(self):
        """Return the declared variable names."""
        return list(self._arrays)

    def _read_variable(self, name):
        """Return the array for ``name`` (``None`` if unreadable)."""
        return self._arrays.get(name)


def _mimic_build_render_kwargs(engine, nc):
    """Stand in for ``NetCDFPlot._build_render_kwargs`` — the direct caller of the resolver."""
    return engine._resolve_curvilinear_coords(nc, coords=None)


def _mimic_run(engine, nc):
    """Stand in for ``NetCDFPlot.run`` — the frame the deprecation warning must be attributed to."""
    return _mimic_build_render_kwargs(engine, nc)


class TestConventionalNamesWarningFrame:
    """Characterization of the model-specific-names ``DeprecationWarning`` attribution frame."""

    def test_warning_attributed_two_frames_above_resolver(self):
        """The deprecation warning is attributed to the caller two real frames above the resolver.

        Test scenario:
            RASM-style ``xc``/``yc`` auto-detection fires the model-specific-names
            ``DeprecationWarning``. Its ``stacklevel`` must attribute it to ``_mimic_run``
            (mirroring the production ``run`` frame, two frames above
            ``_resolve_curvilinear_coords``), so a refactor that moves ``warn()`` across
            function boundaries without updating ``stacklevel`` is caught by this test.
        """
        fake = _FakeNC({"xc": np.full(_SHAPE, 200.0), "yc": np.full(_SHAPE, 45.0)})
        engine = NetCDFPlot(fake)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pair = _mimic_run(engine, fake)
        assert pair is not None, "xc/yc must resolve via the conventional-names source"
        depr = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning) and "model-specific" in str(w.message)
        ]
        assert depr, "the conventional-names source must emit a DeprecationWarning"
        rec = depr[0]
        assert rec.filename.endswith("test_curvilinear_coord_resolver.py"), (
            f"warning attributed to {rec.filename}, expected this test module (stacklevel drift)"
        )
        src_line = Path(rec.filename).read_text(encoding="utf-8").splitlines()[rec.lineno - 1]
        assert "_mimic_build_render_kwargs" in src_line, (
            f"stacklevel drift: warning attributed to {rec.filename}:{rec.lineno} -> "
            f"{src_line.strip()!r}; expected the `_mimic_run` frame two frames above the resolver"
        )
