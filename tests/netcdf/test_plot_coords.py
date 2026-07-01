"""Plot tests: coordinate axes, curvilinear coords, and dim-resolver fallbacks."""

from __future__ import annotations

import logging
import types
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf import Selectors
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf._plot_helpers import _make_2d_nc, _make_3d_nc_anon_dim, _make_capture, _make_curvilinear_nc
from tests.netcdf.conftest import make_plot_3d_nc

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config
Config.set_matplotlib_backend("Agg")


class TestNetCDFPlotCoordAxes:
    """Tests for explicit curvilinear `coords=` validation."""

    def test_invalid_coords_x_raises(self):
        """`coords=("nope", "t2m")` is rejected because "nope" is unknown."""
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"coords x="):
            nc.plot(variable="t2m", coords=("nope", "t2m"))

    def test_invalid_coords_y_raises(self):
        """`coords=("t2m", "nope")` is rejected because "nope" is unknown."""
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"coords y="):
            nc.plot(variable="t2m", coords=("t2m", "nope"))

    def test_valid_coords_render(self):
        """`coords=(<valid>, <valid>)` passes validation and renders.

        Test scenario:
            On this in-memory NetCDF the only variable name is the
            data variable itself. Passing the same name on both axes
            exercises the variable-name lookup branch of
            ``_coerce_coord_spec`` without needing a separate coord
            variable; shape validation falls through to the
            geotransform-derived extent because the data variable's
            shape does not match the slice's 2-D shape.
        """
        nc = make_plot_3d_nc()
        result = nc.plot(variable="t2m", coords=("t2m", "t2m"))
        assert isinstance(result, ArrayGlyph), (
            f"Expected ArrayGlyph from valid-coords render, got {type(result).__name__}"
        )


class TestNetCDFPlotCoordAxesExtra:
    """Additional `coords=` validation coverage."""

    def test_invalid_coords_x_with_valid_y_raises_on_x_first(self):
        """`coords=("bogus", "t2m")` raises on the x axis first."""
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"coords x=") as exc_info:
            nc.plot(variable="t2m", coords=("bogus", "t2m"))
        assert "bogus" in str(
            exc_info.value
        ), f"Error must echo the bad name, got: {exc_info.value}"

    def test_coords_pair_both_valid_renders(self):
        """`coords=(<valid>, <valid>)` passes validation and renders.

        Test scenario:
            Pass the data variable's own name on both axes. The lookup
            via ``_coerce_coord_spec`` succeeds for both; shape
            validation then falls back through the auto-detection
            ladder, and the final render returns an ArrayGlyph.
        """
        nc = make_plot_3d_nc()
        result = nc.plot(variable="t2m", coords=("t2m", "t2m"))
        assert isinstance(result, ArrayGlyph), "coords=(<valid>, <valid>) should render"


class TestNetCDFPlotDimResolverFallbacks:
    """Coverage for the 2-D / fallback paths in the three dim resolvers."""

    def test_time_on_pure_2d_variable_raises(self):
        """``time=`` on a variable with no band dim raises a helpful ValueError.

        Test scenario:
            A 2-D ``(lat, lon)`` variable has empty
            ``_band_dim_names``; the resolver must short-circuit with
            a message that mentions the absent band dimension.
        """
        nc = _make_2d_nc()
        var = nc.get_variable("surface")
        with pytest.raises(ValueError, match=r"no band dimension"):
            var.plot(selectors=Selectors(time=0))

    def test_level_on_pure_2d_variable_raises(self):
        """``level=`` on a 2-D variable raises with the band-dim hint."""
        nc = _make_2d_nc()
        var = nc.get_variable("surface")
        with pytest.raises(ValueError, match=r"no band dimension"):
            var.plot(selectors=Selectors(level=500))

    def test_member_on_pure_2d_variable_raises(self):
        """``member=`` on a 2-D variable raises with the band-dim hint."""
        nc = _make_2d_nc()
        var = nc.get_variable("surface")
        with pytest.raises(ValueError, match=r"no band dimension"):
            var.plot(selectors=Selectors(member=0))

    def test_time_falls_back_to_primary_band_dim(self):
        """``time=`` returns the first band dim when no candidate name matches.

        Test scenario:
            Build a NetCDF whose band dim is ``alpha`` — none of the
            time-coded candidates (``time``, ``valid_time``, ``t``)
            are present. The resolver must fall back to
            ``_band_dim_names[0]`` and pin the slice via
            ``sel(alpha=...)``. Verify by capturing the band that
            reaches the renderer.
        """
        nc = _make_3d_nc_anon_dim()
        var = nc.get_variable("signal")
        expected = var.read_array()[1]
        captured: dict = {}

        with patch.object(
            type(var.analysis), "plot", autospec=True, side_effect=_make_capture(captured)
        ):
            nc.plot(variable="signal", selectors=Selectors(time=20))
        assert_array_equal(
            captured["data"],
            expected,
            err_msg="time= fallback must resolve to the first band dim (alpha)",
        )


class TestCurvilinearCoords:
    """PR-3 — curvilinear coord detection and ``kind=`` dispatch in NetCDF.plot.

    Each test renders a synthetic NetCDF whose parent container exposes
    2-D ``XLAT``/``XLONG``-style coord variables via patched
    ``variable_names`` / ``_read_variable``. The :meth:`NetCDF.plot`
    surface should resolve those coords, hand them to cleopatra as
    ``coords=(x, y)``, and let cleopatra route to ``pcolormesh``.
    """

    def test_explicit_kind_pcolormesh_renders(self):
        """`kind="pcolormesh"` plus 2-D curvilinear coords renders.

        Test scenario:
            Build a WRF-style NetCDF with 2-D ``XLAT``/``XLONG`` coord
            variables. The first call passes the kind explicitly; the
            returned ArrayGlyph wraps a 2-D array with shape
            ``(rows, cols)`` and exposes the resolved coords on
            ``cleo.coords``.
        """
        nc, x_2d, y_2d, _ = _make_curvilinear_nc(rows=6, cols=7)
        cleo = nc.plot(variable="CANWAT", kind="pcolormesh")
        assert isinstance(cleo, ArrayGlyph)
        assert cleo.coords is not None, "curvilinear coords must reach cleopatra"
        assert cleo.coords[0].shape == (6, 7)
        assert cleo.coords[1].shape == (6, 7)
        assert (
            cleo.extent is None
        ), "extent must be suppressed when curvilinear coords are present"

    def test_wrapping_curvilinear_grid_has_no_antimeridian_smear(self):
        """A real 0-360 wrapping curvilinear grid renders without a seam smear (#669).

        Test scenario:
            `none__4v__1d1-2d2-3d1__curv.nc` is a RASM-style converging-pole grid
            whose 2-D longitude wraps 0->360 (a ~360-degree row jump). Plotting
            `Tair` must unwrap the longitude upstream so no rendered `pcolormesh`
            quad spans the antimeridian (~178 degrees wide); the widest legitimate
            converging-pole cell is well under that.
        """
        nc = NetCDF.read_file("tests/data/netcdf/none__4v__1d1-2d2-3d1__curv.nc")
        cleo = nc.plot(variable="Tair")
        assert cleo.coords is not None, "curvilinear coords must reach cleopatra"
        mesh_lon = np.asarray(cleo.im.get_coordinates()[..., 0], dtype=float)
        max_quad_width = float(np.nanmax(np.abs(np.diff(mesh_lon, axis=1))))
        assert max_quad_width < 100.0, (
            f"seam smear: a quad spans {max_quad_width:.1f} deg of longitude "
            "(unwrap should keep every cell well under the ~178 deg antimeridian jump)"
        )

    def test_kind_auto_routes_to_pcolormesh_with_2d_coords(self):
        """`kind="auto"` (default) auto-routes when 2-D coords are detected.

        Test scenario:
            With WRF-style ``XLAT``/``XLONG`` available on the
            container, the auto-detection path picks them up and
            cleopatra's ``kind="auto"`` resolves to pcolormesh
            (verified via ``cleo.coords`` being populated).
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=6, cols=7)
        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (6, 7)

    def test_explicit_coords_by_name(self):
        """`coords=("XLONG", "XLAT")` looks up coord variables by name."""
        nc, x_2d, y_2d, _ = _make_curvilinear_nc(rows=5, cols=6)
        cleo = nc.plot(variable="CANWAT", coords=("XLONG", "XLAT"))
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)

    def test_explicit_coords_by_array(self):
        """`coords=(x_array, y_array)` passes arrays through untouched."""
        nc, x_2d, y_2d, _ = _make_curvilinear_nc(rows=4, cols=5)
        cleo = nc.plot(variable="CANWAT", coords=(x_2d, y_2d))
        assert cleo.coords is not None
        np.testing.assert_array_equal(cleo.coords[0], x_2d)
        np.testing.assert_array_equal(cleo.coords[1], y_2d)

    def test_invalid_coords_one_tuple_raises(self):
        """`coords=("nonexistent",)` (length-1) is rejected as malformed."""
        nc, _, _, _ = _make_curvilinear_nc()
        with pytest.raises(ValueError, match=r"length-2 sequence"):
            nc.plot(variable="CANWAT", coords=("nonexistent",))

    def test_coords_override_auto_detection(self):
        """`coords=("XLONG", "XLAT")` overrides auto-detection.

        Test scenario:
            With the curvilinear conventions in place auto-detection
            would normally pick them up; the test sets ``coords=``
            explicitly and verifies the same coords still reach
            cleopatra (i.e. the explicit path uses the same arrays).
        """
        nc, x_2d, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        cleo = nc.plot(variable="CANWAT", coords=("XLONG", "XLAT"))
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)

    def test_no_curvilinear_falls_back_to_extent(self):
        """A regular variable with no curvilinear coords keeps imshow extent.

        Test scenario:
            Build a plain 3-D ``(time, lat, lon)`` NetCDF with no
            curvilinear coord variables. The auto path returns None, so
            cleopatra renders via imshow with the geotransform-derived
            extent — verified by ``cleo.coords is None`` and a non-None
            ``cleo.extent``.
        """
        nc = make_plot_3d_nc()
        cleo = nc.plot(variable="t2m")
        assert cleo.coords is None
        assert cleo.extent is not None

    def test_roms_naming_convention_auto_detected(self):
        """ROMS-style `lat_rho`/`lon_rho` are auto-detected like WRF."""
        nc, _, _, _ = _make_curvilinear_nc(
            rows=5,
            cols=6,
            x_name="lon_rho",
            y_name="lat_rho",
        )
        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)

    def test_kind_contour_forwards(self):
        """`kind="contour"` is forwarded and renders."""
        nc, _, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        cleo = nc.plot(variable="CANWAT", kind="contour")
        assert isinstance(cleo, ArrayGlyph)

    def test_kind_contourf_forwards(self):
        """`kind="contourf"` is forwarded and renders."""
        nc, _, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        cleo = nc.plot(variable="CANWAT", kind="contourf")
        assert isinstance(cleo, ArrayGlyph)

    def test_kind_bogus_raises_value_error(self):
        """`kind="bogus"` propagates cleopatra's ValueError to the caller.

        Test scenario:
            cleopatra validates ``kind`` against
            :data:`cleopatra.array_glyph.VALID_PLOT_KINDS`. An unknown
            value triggers a ValueError that must propagate through
            pyramids unchanged so users see the same error message they
            would see calling ArrayGlyph directly.
        """
        nc, _, _, _ = _make_curvilinear_nc()
        with pytest.raises(ValueError, match=r"Invalid kind"):
            nc.plot(variable="CANWAT", kind="bogus")

    def test_cf_coordinates_attr_auto_detected(self):
        """CF `coordinates` attribute drives the auto-detection path.

        Test scenario:
            Build a NetCDF where the data variable's subset carries a
            CF ``coordinates`` attribute that lists ``"longitude
            latitude"`` (custom names, not in the well-known list). The
            CF-aware detection path should parse the attribute, resolve
            each name via ``_read_variable``, and pass them to
            cleopatra as curvilinear coords.
        """
        nc, _, _, _ = _make_curvilinear_nc(
            rows=5,
            cols=6,
            x_name="longitude",
            y_name="latitude",
            cf_attr="longitude latitude",
        )
        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)

    def test_nemo_naming_convention_auto_detected(self):
        """NEMO-style ``nav_lat``/``nav_lon`` are auto-detected like WRF."""
        nc, _, _, _ = _make_curvilinear_nc(
            rows=5,
            cols=6,
            x_name="nav_lon",
            y_name="nav_lat",
        )
        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)


class TestCurvilinearCoordsEdges:
    """PR-3 edge cases not covered by :class:`TestCurvilinearCoords`.

    These tests pin down the corner cases of curvilinear coord
    detection — CF attribute ordering, mixed coord-spec forms, shape
    validation, and ``kind=`` interaction with regular vs.
    curvilinear grids.
    """

    def test_explicit_coords_shape_mismatch_warns_and_falls_back(self, caplog):
        """Wrong-shaped `coords=` arrays → `logger.warning` + fall back to extent.

        Test scenario:
            M1 fix — when the caller passes explicit `coords=` whose
            arrays don't match the data slice shape, pyramids must not
            silently ignore them. It logs a WARNING on the
            ``pyramids.netcdf._plot`` logger naming the mismatched
            shapes, then falls through (no conventional coords on this
            NetCDF, so all the way to the geotransform-derived extent).
            The render still succeeds, just without curvilinear coords.
        """
        nc = _make_2d_nc()  # 5x5 `surface`, no XLONG/XLAT auto-detect names
        bad_x = np.zeros((3, 3), dtype=np.float64)
        bad_y = np.zeros((3, 3), dtype=np.float64)
        with caplog.at_level(logging.WARNING, logger="pyramids.netcdf._plot"):
            cleo = nc.plot(variable="surface", coords=(bad_x, bad_y))
        assert cleo.coords is None, (
            "mismatched explicit coords must be dropped, not used; "
            f"got {getattr(cleo, 'coords', None)!r}"
        )
        assert any(
            "don't match the data slice shape" in r.getMessage()
            and r.levelno == logging.WARNING
            for r in caplog.records
        ), f"expected a shape-mismatch WARNING, got: {[r.getMessage() for r in caplog.records]}"

    def test_cf_coordinates_lon_then_lat(self):
        """CF `coordinates="XLONG XLAT"` (lon-first) still resolves the pair.

        Test scenario:
            CF Conventions list auxiliary coord variables space-
            separated with no enforced order. The lon-first form must
            still be parsed: the lon/lat name heuristic identifies
            XLONG as the x candidate and XLAT as the y candidate
            regardless of the order in the attribute string.
        """
        nc, _, _, _ = _make_curvilinear_nc(
            rows=5,
            cols=6,
            cf_attr="XLONG XLAT",
        )
        cleo = nc.plot(variable="CANWAT")
        assert (
            cleo.coords is not None
        ), "lon-first CF attribute must still resolve curvilinear coords"
        assert cleo.coords[0].shape == (
            5,
            6,
        ), f"x array should be (5, 6), got {cleo.coords[0].shape}"

    def test_cf_coordinates_lat_then_lon(self):
        """CF `coordinates="XLAT XLONG"` (lat-first) is also accepted.

        Test scenario:
            With the names in the opposite order the same pair must
            resolve — the heuristic looks at the names, not the list
            position. Both axes still match the data slice shape.
        """
        nc, _, _, _ = _make_curvilinear_nc(
            rows=5,
            cols=6,
            cf_attr="XLAT XLONG",
        )
        cleo = nc.plot(variable="CANWAT")
        assert (
            cleo.coords is not None
        ), "lat-first CF attribute must still resolve curvilinear coords"
        assert cleo.coords[0].shape == (
            5,
            6,
        ), f"x array should still be (5, 6), got {cleo.coords[0].shape}"

    def test_cf_attribute_wins_over_well_known_naming(self):
        """CF `coordinates` takes priority over the WRF naming convention.

        Test scenario:
            The variable carries both a CF ``coordinates`` attribute
            that names a custom pair (``my_lon``/``my_lat``) AND the
            WRF-style ``XLONG``/``XLAT`` is available on the parent.
            CF detection runs first, so the custom pair wins. We assert
            on the actual coord arrays returned — the CF arrays differ
            from the WRF arrays because they are independent grids.
        """
        rng = np.random.default_rng(42)
        nc = NetCDF.create_from_array(
            arr=rng.random((5, 6)).astype(np.float32),
            geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
            epsg=4326,
            variable_name="CANWAT",
        )
        wrf_x = np.linspace(-110.0, -100.0, 6, dtype=np.float32)
        wrf_y = np.linspace(35.0, 45.0, 5, dtype=np.float32)
        wrf_x_2d, wrf_y_2d = np.meshgrid(wrf_x, wrf_y)
        cf_x_2d = wrf_x_2d + 100.0
        cf_y_2d = wrf_y_2d + 50.0
        extra_vars = {
            "XLONG": wrf_x_2d,
            "XLAT": wrf_y_2d,
            "my_lon": cf_x_2d,
            "my_lat": cf_y_2d,
        }
        spliced_names = list(nc.variable_names) + list(extra_vars)
        original_read = type(nc)._read_variable
        original_get_variable = type(nc).get_variable

        def _read(self_, var, window=None):
            if var in extra_vars:
                return extra_vars[var]
            return original_read(self_, var, window)

        def _get_variable(self_, name, x_dim=None, y_dim=None):
            subset = original_get_variable(self_, name, x_dim=x_dim, y_dim=y_dim)
            attrs = dict(getattr(subset, "_variable_attrs", {}) or {})
            attrs["coordinates"] = "my_lon my_lat"
            subset._variable_attrs = attrs
            return subset

        nc._read_variable = types.MethodType(_read, nc)
        nc.get_variable = types.MethodType(_get_variable, nc)
        nc_class = type(nc)
        subcls = type(
            f"{nc_class.__name__}WithBothCoordPairs",
            (nc_class,),
            {"variable_names": property(lambda _self: spliced_names)},
        )
        nc.__class__ = subcls

        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None, "CF coords must resolve"
        # The CF arrays (shifted by 100/50) should reach cleopatra, not
        # the WRF arrays. Check on the x axis (longitude shift = +100).
        np.testing.assert_array_equal(
            cleo.coords[0],
            cf_x_2d,
            err_msg="CF attribute pair must win over WRF naming convention",
        )

    def test_cf_attribute_wrong_shape_falls_back_to_extent(self):
        """A CF `coordinates` attr naming a wrong-shape coord falls back.

        Test scenario:
            The CF attribute names ``my_lon``/``my_lat`` but the
            arrays returned by ``_read_variable`` have a shape that
            does not match the data slice. The detector must silently
            skip and the render must succeed using the geotransform
            extent — i.e. no crash, ``cleo.coords is None``, and the
            extent is populated from the bbox.
        """
        rng = np.random.default_rng(43)
        nc = NetCDF.create_from_array(
            arr=rng.random((5, 6)).astype(np.float32),
            geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
            epsg=4326,
            variable_name="CANWAT",
        )
        bad_x = np.linspace(-1.0, 1.0, 99, dtype=np.float32)
        bad_y = np.linspace(0.0, 1.0, 99, dtype=np.float32)
        extra_vars = {"my_lon": bad_x, "my_lat": bad_y}
        spliced_names = list(nc.variable_names) + list(extra_vars)
        original_read = type(nc)._read_variable
        original_get_variable = type(nc).get_variable

        def _read(self_, var, window=None):
            if var in extra_vars:
                return extra_vars[var]
            return original_read(self_, var, window)

        def _get_variable(self_, name, x_dim=None, y_dim=None):
            subset = original_get_variable(self_, name, x_dim=x_dim, y_dim=y_dim)
            attrs = dict(getattr(subset, "_variable_attrs", {}) or {})
            attrs["coordinates"] = "my_lon my_lat"
            subset._variable_attrs = attrs
            return subset

        nc._read_variable = types.MethodType(_read, nc)
        nc.get_variable = types.MethodType(_get_variable, nc)
        nc_class = type(nc)
        subcls = type(
            f"{nc_class.__name__}WithBadCFShape",
            (nc_class,),
            {"variable_names": property(lambda _self: spliced_names)},
        )
        nc.__class__ = subcls

        cleo = nc.plot(variable="CANWAT")
        assert (
            cleo.coords is None
        ), "Wrong-shape CF coords must be skipped (no crash); got coords"
        assert (
            cleo.extent is not None
        ), "Renderer must fall back to extent when CF coords don't fit"

    def test_explicit_coords_missing_variable_name_raises(self):
        """`coords=("missing", "XLAT")` references a non-variable name.

        Test scenario:
            One of the two names doesn't exist in
            ``parent.variable_names``. The coord-spec coercer raises
            :class:`ValueError`, mentioning the bad name and listing
            available variables. The other valid name must not mask
            the error.
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        with pytest.raises(ValueError, match=r"missing") as exc_info:
            nc.plot(variable="CANWAT", coords=("missing", "XLAT"))
        assert "Available" in str(
            exc_info.value
        ), f"Error must list available variables, got: {exc_info.value}"

    def test_explicit_coords_mixed_string_array_forms(self):
        """`coords=(name, array)` mixed-form is accepted.

        Test scenario:
            The first element is a variable name (resolved via
            ``_read_variable``), the second is a raw numpy array. The
            coercer treats each element independently, so mixed forms
            must work. The resulting curvilinear coords must reach
            cleopatra.
        """
        nc, x_2d, y_2d, _ = _make_curvilinear_nc(rows=4, cols=5)
        cleo = nc.plot(variable="CANWAT", coords=("XLONG", y_2d))
        assert cleo.coords is not None, "Mixed-form coords must resolve"
        np.testing.assert_array_equal(cleo.coords[0], x_2d)
        np.testing.assert_array_equal(cleo.coords[1], y_2d)

    def test_explicit_coords_with_nan_values_propagates_matplotlib_error(self):
        """`coords=(x_nan, y_nan)` propagates matplotlib's non-finite-coords error.

        Test scenario:
            Pyramids does not validate coord *values* — only shapes.
            All-NaN coord arrays reach cleopatra which calls
            ``ax.pcolormesh``. Matplotlib rejects non-finite coords
            with a ValueError. The pyramids layer must not mask this
            error (no try/except around the render); it must
            propagate to the caller unchanged so the user can fix
            the upstream data.
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=4, cols=5)
        x_nan = np.full((4, 5), np.nan, dtype=np.float32)
        y_nan = np.full((4, 5), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match=r"non-finite"):
            nc.plot(variable="CANWAT", coords=(x_nan, y_nan))

    def test_kind_auto_no_curvilinear_uses_imshow_path(self):
        """`kind="auto"` on a regular grid leaves coords None (imshow path).

        Test scenario:
            A plain 3-D NetCDF with no curvilinear conventions has no
            coords to resolve. With ``kind="auto"`` (the default) the
            renderer should fall through to imshow — verified by
            ``cleo.coords is None`` and a populated extent.
        """
        nc = make_plot_3d_nc()
        cleo = nc.plot(variable="t2m", kind="auto")
        assert (
            cleo.coords is None
        ), "Regular grid + kind='auto' should keep coords None (imshow path)"
        assert cleo.extent is not None, "Imshow path must carry an extent"

    def test_kind_pcolormesh_without_explicit_coords_renders(self):
        """`kind="pcolormesh"` + coords=None — cleopatra auto-derives a grid.

        Test scenario:
            With no curvilinear coords and an explicit
            ``kind="pcolormesh"``, cleopatra falls back to an
            index-derived grid. The pyramids layer must forward the
            kind verbatim and not crash; cleo handles the rest.
        """
        nc = make_plot_3d_nc()
        cleo = nc.plot(variable="t2m", kind="pcolormesh")
        assert isinstance(
            cleo, ArrayGlyph
        ), "kind='pcolormesh' without coords must still produce an ArrayGlyph"

    def test_coords_1d_x_1d_y_correct_lengths(self):
        """`coords=(1D x of len cols, 1D y of len rows)` is accepted.

        Test scenario:
            cleopatra accepts 1-D coord pairs (x of length ``cols``,
            y of length ``rows``) and meshgrids them internally. The
            pyramids shape validator must accept this form: assert the
            returned cleo carries the original 1-D arrays.
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        x_1d = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
        y_1d = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        cleo = nc.plot(variable="CANWAT", coords=(x_1d, y_1d))
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (
            6,
        ), f"x should be 1-D of length 6, got {cleo.coords[0].shape}"
        assert cleo.coords[1].shape == (
            5,
        ), f"y should be 1-D of length 5, got {cleo.coords[1].shape}"

    def test_coords_1d_swapped_lengths_falls_back_to_extent(self):
        """`coords=(1D x of len rows, 1D y of len cols)` shapes mismatch.

        Test scenario:
            Swap the two arrays — now x has the row length and y has
            the col length. ``_coord_shapes_match`` returns False, so
            the explicit-coord branch rejects them. With no other
            curvilinear conventions on the container (plain NetCDF
            from :func:`make_plot_3d_nc`) the render falls back to
            extent. We assert no crash and ``cleo.coords is None``.
        """
        nc = make_plot_3d_nc(n_times=1, rows=5, cols=6)
        x_wrong = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
        y_wrong = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        cleo = nc.plot(variable="t2m", coords=(x_wrong, y_wrong))
        assert (
            cleo.coords is None
        ), "Swapped-length 1-D coords must skip and fall back to extent"
        assert cleo.extent is not None

    def test_coords_2d_x_1d_y_mixed_dims_accepted(self):
        """`coords=(2D x matching slice, 1D y of len rows)` mixed dims work.

        Test scenario:
            The shape validator accepts each axis independently — 2-D
            x matching the slice plus a 1-D y matching ``rows``
            satisfies both `x_ok` and `y_ok`. Verify the mixed-dim
            arrays reach cleopatra unchanged.
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        x_2d = np.random.default_rng(99).random((5, 6)).astype(np.float32)
        y_1d = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        cleo = nc.plot(variable="CANWAT", coords=(x_2d, y_1d))
        assert cleo.coords is not None, "Mixed (2D, 1D) coords must resolve"
        assert cleo.coords[0].shape == (5, 6)
        assert cleo.coords[1].shape == (5,)
