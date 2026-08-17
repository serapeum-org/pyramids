"""Unit tests for curvilinear coordinate resolution in ``NetCDFPlot``.

Drives ``_resolve_curvilinear_coords`` in isolation with a duck-typed fake NetCDF slice — no
cleopatra, no sample files — so these run in the main (``not plot``) suite. The end-to-end
render path is covered by ``tests/netcdf/plot/test_plot_coords.py`` and the real-sample
curvilinear crops by ``tests/netcdf/samples/test_curvilinear_crop.py``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf._plot import CurvilinearCoordResolver, NetCDFPlot

_SHAPE = (5, 6)  # (rows, cols)


def _cols_1d():
    """A 1-D coordinate whose length equals the column count (matches the x axis)."""
    return np.arange(6.0)


def _rows_1d():
    """A 1-D coordinate whose length equals the row count (matches the y axis)."""
    return np.arange(5.0)


def _grid_2d(value=1.0):
    """A 2-D coordinate matching the slice shape ``(5, 6)``."""
    return np.full(_SHAPE, value)


def _bad_2d():
    """A 2-D coordinate whose shape does not match the slice ``(5, 6)``."""
    return np.full((3, 3), 0.0)


class _FakeNC:
    """Minimal duck-typed NetCDF slice exposing only what ``_resolve_curvilinear_coords`` reads."""

    def __init__(
        self, arrays, shape=_SHAPE, variable_attrs=None, curvilinear_coords=None
    ):
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
            if issubclass(w.category, DeprecationWarning)
            and "model-specific" in str(w.message)
        ]
        assert depr, "the conventional-names source must emit a DeprecationWarning"
        rec = depr[0]
        assert rec.filename.endswith("test_curvilinear_coord_resolver.py"), (
            f"warning attributed to {rec.filename}, expected this test module (stacklevel drift)"
        )
        src_line = (
            Path(rec.filename).read_text(encoding="utf-8").splitlines()[rec.lineno - 1]
        )
        assert "_mimic_build_render_kwargs" in src_line, (
            f"stacklevel drift: warning attributed to {rec.filename}:{rec.lineno} -> "
            f"{src_line.strip()!r}; expected the `_mimic_run` frame two frames above the resolver"
        )


class TestCurvilinearCoordResolver:
    """Unit tests for the four-source ``CurvilinearCoordResolver`` and its helpers."""

    def test_init_derives_context_from_slice(self):
        """The ctor stores nc and derives parent (self when no parent) and data_shape."""
        fake = _FakeNC({"lon": _cols_1d()})
        resolver = CurvilinearCoordResolver(fake)
        assert resolver.nc is fake, "nc must be stored"
        assert resolver.parent is fake, (
            "parent falls back to nc when _parent_nc is None"
        )
        assert resolver.data_shape == _SHAPE, (
            f"data_shape should be {_SHAPE}, got {resolver.data_shape}"
        )

    def test_init_uses_parent_nc_when_present(self):
        """When the slice has a parent container, coord variables are read off the parent."""
        parent = _FakeNC({"xc": _grid_2d(200.0), "yc": _grid_2d(45.0)})
        child = _FakeNC({})
        child._parent_nc = parent
        resolver = CurvilinearCoordResolver(child)
        assert resolver.parent is parent, "parent must be _parent_nc when it is set"

    def test_init_none_shape_yields_none_data_shape(self):
        """An empty ``nc.shape`` yields ``data_shape=None`` (the guard for sources 2-4)."""
        resolver = CurvilinearCoordResolver(_FakeNC({}, shape=()))
        assert resolver.data_shape is None, "empty shape must map to None data_shape"

    def test_resolve_explicit_wins_over_other_sources(self):
        """An explicit shape-valid ``coords=`` short-circuits the auto-detection sources.

        Test scenario:
            The slice also has stored coords and conventional xc/yc, but explicit coords
            win, so the returned pair is the explicit arrays and no DeprecationWarning fires.
        """
        lon, lat = _cols_1d(), _rows_1d()
        fake = _FakeNC(
            {"xc": _grid_2d(200.0), "yc": _grid_2d(45.0)},
            curvilinear_coords=(_grid_2d(1.0), _grid_2d(2.0)),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pair = CurvilinearCoordResolver(fake).resolve((lon, lat))
        assert pair is not None, "explicit coords must resolve"
        assert pair[0] is lon, "x must be the explicit lon array"
        assert pair[1] is lat, "y must be the explicit lat array"
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not deprecations, (
            "the conventional source must not be reached when explicit wins"
        )

    def test_resolve_none_data_shape_skips_fallback_sources(self):
        """With ``data_shape=None`` only source 1 runs; sources 2-4 are skipped.

        Test scenario:
            A slice with empty shape but conventional xc/yc present resolves to ``None`` and
            emits no DeprecationWarning, proving the conventional source was never reached.
        """
        fake = _FakeNC({"xc": _grid_2d(200.0), "yc": _grid_2d(45.0)}, shape=())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = CurvilinearCoordResolver(fake).resolve(None)
        assert result is None, "no source can resolve without a data shape"
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not deprecations, "sources 2-4 must be skipped when data_shape is None"

    def test_resolve_falls_through_to_conventional(self):
        """With no explicit/stored/CF coords, resolve reaches the conventional source."""
        xc, yc = _grid_2d(200.0), _grid_2d(45.0)
        fake = _FakeNC({"xc": xc, "yc": yc})
        with pytest.warns(DeprecationWarning, match="model-specific"):
            pair = CurvilinearCoordResolver(fake).resolve(None)
        assert pair is not None, "xc/yc must resolve via the conventional source"
        assert pair[0] is xc, "x must be xc"
        assert pair[1] is yc, "y must be yc"

    def test_from_explicit_none_returns_none(self):
        """A ``None`` explicit spec yields ``None`` without raising."""
        assert CurvilinearCoordResolver(_FakeNC({}))._from_explicit(None) is None, (
            "None coords must yield None"
        )

    @pytest.mark.parametrize("coords", [("only-one",), ("a", "b", "c"), "ab", 5])
    def test_from_explicit_non_length_2_raises(self, coords):
        """A spec that is not a length-2 tuple/list raises ``ValueError``.

        Args:
            coords: A malformed ``coords=`` value (wrong length or wrong type).
        """
        with pytest.raises(ValueError, match="length-2 sequence") as exc:
            CurvilinearCoordResolver(_FakeNC({}))._from_explicit(coords)
        assert "length-2 sequence" in str(exc.value), f"unexpected message: {exc.value}"

    def test_from_explicit_two_names_resolves(self):
        """Two coord-variable names are looked up and validated into a pair."""
        lon, lat = _cols_1d(), _rows_1d()
        resolver = CurvilinearCoordResolver(_FakeNC({"lon": lon, "lat": lat}))
        pair = resolver._from_explicit(("lon", "lat"))
        assert pair is not None, "named coords must resolve"
        assert pair[0] is lon, "x must be the looked-up lon array"
        assert pair[1] is lat, "y must be the looked-up lat array"

    def test_from_explicit_two_arrays_resolves(self):
        """Two array-likes are passed through and validated into a pair."""
        x2d, y2d = _grid_2d(1.0), _grid_2d(2.0)
        pair = CurvilinearCoordResolver(_FakeNC({}))._from_explicit((x2d, y2d))
        assert pair is not None, "array coords must resolve"
        assert pair[0] is x2d, "x array passes through unchanged"
        assert pair[1] is y2d, "y array passes through unchanged"

    def test_from_explicit_shape_mismatch_warns_and_returns_none(self, caplog):
        """A shape-mismatching explicit pair returns ``None`` and logs a warning."""
        with caplog.at_level(logging.WARNING, logger="pyramids.netcdf._plot"):
            result = CurvilinearCoordResolver(_FakeNC({}))._from_explicit(
                (_bad_2d(), _bad_2d())
            )
        assert result is None, "a shape-mismatching explicit pair must be rejected"
        warns = [r for r in caplog.records if r.name == "pyramids.netcdf._plot"]
        assert warns, "a mismatch warning must be logged on the _plot logger"
        assert "don't match" in warns[0].getMessage(), (
            f"unexpected message: {warns[0].getMessage()}"
        )

    def test_from_stored_present_and_matching(self):
        """Stored ``_curvilinear_coords`` that match the slice shape resolve to a pair."""
        x2d, y2d = _grid_2d(1.0), _grid_2d(2.0)
        resolver = CurvilinearCoordResolver(_FakeNC({}, curvilinear_coords=(x2d, y2d)))
        pair = resolver._from_stored()
        assert pair is not None, "stored matching coords must resolve"
        assert np.array_equal(pair[0], x2d), "x must be the stored array"
        assert np.array_equal(pair[1], y2d), "y must be the stored array"

    def test_from_stored_present_but_mismatching(self):
        """Stored coords that do not match the slice shape yield ``None``."""
        resolver = CurvilinearCoordResolver(
            _FakeNC({}, curvilinear_coords=(_bad_2d(), _bad_2d()))
        )
        assert resolver._from_stored() is None, (
            "shape-mismatching stored coords must be rejected"
        )

    def test_from_stored_absent(self):
        """No ``_curvilinear_coords`` attribute yields ``None``."""
        assert CurvilinearCoordResolver(_FakeNC({}))._from_stored() is None, (
            "absent stored coords must yield None"
        )

    def test_from_cf_resolves_via_cf_candidates(self):
        """A CF ``coordinates`` attribute resolves through CFCoordinateCandidates."""
        lon, lat = _cols_1d(), _rows_1d()
        fake = _FakeNC(
            {"lon": lon, "lat": lat}, variable_attrs={"coordinates": "lon lat"}
        )
        pair = CurvilinearCoordResolver(fake)._from_cf()
        assert pair is not None, "a CF coordinates attr must resolve"
        assert pair[0] is lon, "x must be the CF lon candidate"
        assert pair[1] is lat, "y must be the CF lat candidate"

    def test_from_cf_no_valid_pair_returns_none(self):
        """A CF attribute whose vars don't form a valid pair yields ``None``."""
        fake = _FakeNC({"bad": _bad_2d()}, variable_attrs={"coordinates": "bad"})
        assert CurvilinearCoordResolver(fake)._from_cf() is None, (
            "an unmatchable CF attr must yield None"
        )

    def test_from_conventional_matching_pair_warns(self):
        """A matching conventional pair resolves and emits the deprecation warning."""
        xc, yc = _grid_2d(200.0), _grid_2d(45.0)
        resolver = CurvilinearCoordResolver(_FakeNC({"xc": xc, "yc": yc}))
        with pytest.warns(DeprecationWarning, match="model-specific"):
            pair = resolver._from_conventional()
        assert pair is not None, "a matching conventional pair must resolve"
        assert pair[0] is xc, "x must be xc"
        assert pair[1] is yc, "y must be yc"

    def test_from_conventional_require_2d_gate_rejects_1d(self):
        """A generic ``require_2d`` pair (xc/yc) with 1-D arrays is rejected (no warning)."""
        fake = _FakeNC({"xc": _cols_1d(), "yc": _rows_1d()})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = CurvilinearCoordResolver(fake)._from_conventional()
        assert result is None, "1-D xc/yc must fail the require_2d gate"
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not deprecations, "a rejected pair must not emit the deprecation warning"

    def test_from_conventional_present_wrong_shape_logs_debug(self, caplog):
        """A present-but-wrong-shape conventional pair logs a debug skip and returns ``None``."""
        fake = _FakeNC({"XLONG": _bad_2d(), "XLAT": _bad_2d()})
        with caplog.at_level(logging.DEBUG, logger="pyramids.netcdf._plot"):
            result = CurvilinearCoordResolver(fake)._from_conventional()
        assert result is None, "a wrong-shape conventional pair must not resolve"
        debugs = [
            r
            for r in caplog.records
            if "don't match the data slice shape" in r.getMessage()
        ]
        assert debugs, (
            "a debug skip message must be logged for the present-but-wrong-shape pair"
        )

    def test_from_conventional_absent_names_returns_none(self):
        """No conventional names present yields ``None``."""
        assert (
            CurvilinearCoordResolver(
                _FakeNC({"other": _grid_2d()})
            )._from_conventional()
            is None
        ), "absent conventional names must yield None"

    def test_conventional_pair_arrays_unreadable_variable(self):
        """A conventional name that reads back ``None`` is treated as unusable."""
        fake = _FakeNC({"XLONG": _grid_2d(1.0), "XLAT": None})
        result = CurvilinearCoordResolver(fake)._conventional_pair_arrays(
            "XLONG", "XLAT", False
        )
        assert result is None, "an unreadable variable must yield None"

    @pytest.mark.parametrize(
        "x_arr, y_arr, expected_pair",
        [
            (_grid_2d(1.0), _grid_2d(2.0), True),
            (_bad_2d(), _bad_2d(), False),
            (_cols_1d(), _rows_1d(), True),
        ],
    )
    def test_validated(self, x_arr, y_arr, expected_pair):
        """``_validated`` returns the pair when shapes line up with the slice, else ``None``.

        Args:
            x_arr: Candidate x array.
            y_arr: Candidate y array.
            expected_pair: Whether the arrays should validate against ``(5, 6)``.
        """
        result = CurvilinearCoordResolver(_FakeNC({}))._validated(x_arr, y_arr)
        assert (result is not None) is expected_pair, (
            f"unexpected _validated result: {result}"
        )

    def test_validated_none_data_shape_returns_none(self):
        """With ``data_shape=None`` no pair can be validated."""
        resolver = CurvilinearCoordResolver(_FakeNC({}, shape=()))
        assert resolver._validated(_grid_2d(), _grid_2d()) is None, (
            "None data_shape must reject any pair"
        )

    def test_coerce_name_lookup_success(self):
        """A known variable name is resolved to its array."""
        lon = _cols_1d()
        resolver = CurvilinearCoordResolver(_FakeNC({"lon": lon}))
        assert resolver._coerce("lon", "x") is lon, (
            "a known name must resolve to its array"
        )

    def test_coerce_unknown_name_raises(self):
        """An unknown coord name raises ``ValueError``."""
        with pytest.raises(ValueError, match="is not a variable") as exc:
            CurvilinearCoordResolver(_FakeNC({"lon": _cols_1d()}))._coerce("nope", "x")
        assert "nope" in str(exc.value), f"error must echo the bad name: {exc.value}"

    def test_coerce_unreadable_name_raises(self):
        """A name present but reading back ``None`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="could not be read"):
            CurvilinearCoordResolver(_FakeNC({"lon": None}))._coerce("lon", "y")

    def test_coerce_array_like_passthrough(self):
        """An array-like spec is converted via ``numpy.asarray``."""
        result = CurvilinearCoordResolver(_FakeNC({}))._coerce(
            [[1.0, 2.0], [3.0, 4.0]], "x"
        )
        assert isinstance(result, np.ndarray), (
            f"expected ndarray, got {type(result).__name__}"
        )
        assert result.shape == (2, 2), f"expected (2, 2), got {result.shape}"

    def test_resolve_stored_wins_over_cf_and_conventional(self):
        """Stored crop coords (source 2) take precedence over CF (3) and conventional (4).

        Test scenario:
            The slice has stored coords AND a resolvable CF ``coordinates`` attr AND
            conventional xc/yc all present; the stored pair wins and no DeprecationWarning
            fires, proving sources 3-4 are not reached.
        """
        stored_x, stored_y = _grid_2d(1.0), _grid_2d(2.0)
        fake = _FakeNC(
            {
                "xc": _grid_2d(200.0),
                "yc": _grid_2d(45.0),
                "lon": _cols_1d(),
                "lat": _rows_1d(),
            },
            variable_attrs={"coordinates": "lon lat"},
            curvilinear_coords=(stored_x, stored_y),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pair = CurvilinearCoordResolver(fake).resolve(None)
        assert pair is not None, "stored coords must resolve"
        assert np.array_equal(pair[0], stored_x), (
            "x must be the stored array (source 2 wins)"
        )
        assert np.array_equal(pair[1], stored_y), (
            "y must be the stored array (source 2 wins)"
        )
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not deprecations, (
            "conventional source must not run when stored coords win"
        )

    def test_from_conventional_skips_wrong_shape_then_matches_later_pair(self, caplog):
        """A present-but-wrong-shape early pair is skipped (debug) and a valid later pair resolves.

        Test scenario:
            XLONG/XLAT (first name-pair) are present but wrong-shape -> one debug skip; xc/yc
            (last name-pair) are present and valid -> the loop continues past the skip and
            resolves + warns, proving the loop does not stop at the debug-skip.
        """
        xc, yc = _grid_2d(200.0), _grid_2d(45.0)
        fake = _FakeNC({"XLONG": _bad_2d(), "XLAT": _bad_2d(), "xc": xc, "yc": yc})
        with caplog.at_level(logging.DEBUG, logger="pyramids.netcdf._plot"):
            with pytest.warns(DeprecationWarning, match="model-specific"):
                pair = CurvilinearCoordResolver(fake)._from_conventional()
        assert pair is not None, "the valid later pair must resolve"
        assert pair[0] is xc, "x must be xc from the later pair"
        assert pair[1] is yc, "y must be yc from the later pair"
        debugs = [
            r
            for r in caplog.records
            if "don't match the data slice shape" in r.getMessage()
        ]
        assert debugs, (
            "the early wrong-shape pair must log a debug skip before the later pair resolves"
        )

    def test_conventional_pair_arrays_none_data_shape_returns_none(self):
        """A direct call with ``data_shape=None`` returns ``None`` (not TypeError), like the siblings."""
        fake = _FakeNC({"xc": _grid_2d(200.0), "yc": _grid_2d(45.0)}, shape=())
        result = CurvilinearCoordResolver(fake)._conventional_pair_arrays(
            "xc", "yc", True
        )
        assert result is None, (
            "a None data_shape must yield None, consistent with the other sources"
        )
