"""Shared in-memory NetCDF builders and render side-effects for the plot tests."""

from __future__ import annotations

import types

import numpy as np

from pyramids.netcdf.netcdf import NetCDF


def _make_capture(captured: dict):
    """Return a side-effect for patching Analysis.plot that stores the rendered slice."""
    def _inner(self_engine, **kw):
        captured["data"] = self_engine._ds.read_array(band=0)
        return "ok"
    return _inner


def _make_fake_render(captured: dict):
    """Return a side-effect for patching _render_array that stores the call kwargs."""
    def _inner(**kw):
        captured["kw"] = kw
        return "ok"
    return _inner


def _make_3d_nc_with_dates():
    """Build a 3-D variable with date-string time coords for label selection.

    Returns:
        tuple[NetCDF, list[str], NetCDF]: The container, the list of time
            coord values (strings) used for the time axis, and the ``t2m``
            variable subset (whose patched date coords the caller relies on).
    """
    rng = np.random.default_rng(1)
    times = ["2024-01-13", "2024-01-14", "2024-01-15", "2024-01-16"]
    arr = rng.random((len(times), 5, 5)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
        epsg=4326,
        variable_name="t2m",
        extra_dim_name="time",
        extra_dim_values=list(range(len(times))),
    )
    var = nc.get_variable("t2m")
    var._band_dim_values = list(times)
    var._band_dim_values_map = dict(var._band_dim_values_map)
    var._band_dim_values_map["time"] = list(times)
    return nc, times, var


def _make_4d_nc():
    """Build a 4-D (time, pressure_level, lat, lon) NetCDF in memory.

    Returns:
        NetCDF: Root MDIM container with a single variable ``temperature``.
    """
    nt, nl, ny, nx = 3, 2, 4, 5
    rng = np.random.default_rng(2)
    arr = rng.random((nt, nl, ny, nx)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, float(ny), 0, -1.0),
        epsg=4326,
        variable_name="temperature",
        extra_dims=[
            ("time", [0, 6, 12]),
            ("pressure_level", [1000, 500]),
        ],
    )
    return nc


def _make_2d_nc():
    """Build a 2-D (lat, lon) NetCDF in memory with no band dim.

    Returns:
        NetCDF: Container with a single 2-D variable ``surface``.
    """
    rng = np.random.default_rng(3)
    arr = rng.random((5, 5)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
        epsg=4326,
        variable_name="surface",
    )
    return nc


def _make_ensemble_nc():
    """Build a 3-D (member, lat, lon) NetCDF for ensemble selector tests.

    Returns:
        NetCDF: Container whose only variable has an ``ensemble`` dim.
    """
    rng = np.random.default_rng(4)
    arr = rng.random((3, 4, 4)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, 4.0, 0, -1.0),
        epsg=4326,
        variable_name="forecast",
        extra_dim_name="ensemble",
        extra_dim_values=[0, 1, 2],
    )
    return nc


def _make_3d_nc_anon_dim():
    """Build a 3-D NetCDF whose band dim is not a time-coded name.

    Returns:
        NetCDF: Container whose variable has a single ``alpha`` band dim
        (none of ``time`` / ``valid_time`` / ``t``).
    """
    rng = np.random.default_rng(5)
    arr = rng.random((3, 4, 4)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, 4.0, 0, -1.0),
        epsg=4326,
        variable_name="signal",
        extra_dim_name="alpha",
        extra_dim_values=[10, 20, 30],
    )
    return nc


def _attach_curvilinear_coords(
    nc,
    rows: int,
    cols: int,
    *,
    x_name: str = "XLONG",
    y_name: str = "XLAT",
    cf_attr: str | None = None,
    coord_ndim: int | tuple[int, int] = 2,
):
    """Splice synthetic curvilinear coord arrays (2-D by default, 1-D on request) onto the container.

    Helper used by :class:`TestCurvilinearCoords` to simulate a WRF-style
    NetCDF without authoring a real GDAL MDIM file. Patches are scoped
    to the instance — ``nc.__dict__`` is mutated with overrides for
    ``variable_names`` and ``_read_variable``, leaving other instances
    of :class:`NetCDF` untouched.

    Args:
        nc: NetCDF container to splice onto.
        rows: Number of rows for the synthetic coord grid.
        cols: Number of columns for the synthetic coord grid.
        x_name: Name of the synthetic x-coord variable.
        y_name: Name of the synthetic y-coord variable.
        cf_attr: When not None, the value is written as a CF
            ``coordinates`` attribute on the data variable's subset so
            the CF detection path fires.
        coord_ndim: ``2`` (default) installs 2-D meshgrid curvilinear
            coords; ``1`` installs 1-D axis vectors (``cols``-long x,
            ``rows``-long y) to simulate a projected rectilinear grid. A
            ``(x_ndim, y_ndim)`` tuple installs a per-axis mix, e.g.
            ``(2, 1)`` for a 2-D x with a 1-D y.

    Returns:
        tuple: ``(x_coord, y_coord)`` — the synthetic coord arrays that
        were installed (2-D by default, 1-D / mixed per ``coord_ndim``).
    """
    x_arr = np.linspace(-110.0, -100.0, cols, dtype=np.float32)
    y_arr = np.linspace(35.0, 45.0, rows, dtype=np.float32)
    x_2d, y_2d = np.meshgrid(x_arr, y_arr)
    x_ndim, y_ndim = (coord_ndim, coord_ndim) if isinstance(coord_ndim, int) else coord_ndim
    x_installed = x_arr if x_ndim == 1 else x_2d
    y_installed = y_arr if y_ndim == 1 else y_2d
    extra_vars = {x_name: x_installed, y_name: y_installed}
    base_names = list(nc.variable_names)
    spliced_names = base_names + [x_name, y_name]
    original_read = type(nc)._read_variable
    original_get_variable = type(nc).get_variable

    def _read(self_, var, window=None):
        if var in extra_vars:
            return extra_vars[var]
        return original_read(self_, var, window)

    def _get_variable(self_, name, x_dim=None, y_dim=None):
        subset = original_get_variable(self_, name, x_dim=x_dim, y_dim=y_dim)
        if cf_attr is not None and name in base_names:
            attrs = dict(getattr(subset, "_variable_attrs", {}) or {})
            attrs["coordinates"] = cf_attr
            subset._variable_attrs = attrs
        return subset

    # Instance-level patches: bind the closures to this `nc` only by
    # using `types.MethodType` so other NetCDF instances built inside
    # the same test session see the original implementations.
    nc._read_variable = types.MethodType(_read, nc)
    nc.get_variable = types.MethodType(_get_variable, nc)
    # `variable_names` is a property on the class. Override per-instance
    # by stuffing a simple object that returns the spliced list when the
    # property's descriptor falls back to `__dict__` (it doesn't).
    # Instead, monkeypatch the property via a subclass on this instance
    # using a small class trick: replace `__class__` on this instance
    # with a thin subclass that overrides only `variable_names`.
    nc_class = type(nc)
    subcls = type(
        f"{nc_class.__name__}WithSpliceCoords",
        (nc_class,),
        {"variable_names": property(lambda _self: spliced_names)},
    )
    nc.__class__ = subcls
    return x_installed, y_installed


def _make_curvilinear_nc(
    rows: int = 6,
    cols: int = 7,
    *,
    x_name: str = "XLONG",
    y_name: str = "XLAT",
    cf_attr: str | None = None,
    n_times: int | None = None,
    coord_ndim: int | tuple[int, int] = 2,
):
    """Build a NetCDF whose container advertises curvilinear coords.

    Args:
        rows: Number of latitude rows.
        cols: Number of longitude columns.
        x_name: Name for the synthetic x-coord variable
            (``"XLONG"`` for WRF, ``"lon_rho"`` for ROMS, etc.).
        y_name: Name for the synthetic y-coord variable.
        cf_attr: When not None, the value is written as a CF
            ``coordinates`` attribute on the data variable's subset so
            the CF detection path fires.
        n_times: When set, build a 3-D (time, lat, lon) variable;
            otherwise build a 2-D (lat, lon) variable.
        coord_ndim: ``2`` (default) attaches 2-D curvilinear coords;
            ``1`` attaches 1-D projected axis vectors; a
            ``(x_ndim, y_ndim)`` tuple attaches a per-axis mix.

    Returns:
        tuple: ``(nc, x_arr, y_arr, data_var_name)`` — the container,
        the synthetic coord arrays, and the data variable name.
    """
    rng = np.random.default_rng(7)
    data_var = "CANWAT"
    if n_times is None:
        arr = rng.random((rows, cols)).astype(np.float32)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=(0.0, 1.0, 0, float(rows), 0, -1.0),
            epsg=4326,
            variable_name=data_var,
        )
    else:
        arr = rng.random((n_times, rows, cols)).astype(np.float32)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=(0.0, 1.0, 0, float(rows), 0, -1.0),
            epsg=4326,
            variable_name=data_var,
            extra_dim_name="time",
            extra_dim_values=list(range(n_times)),
        )
    x_2d, y_2d = _attach_curvilinear_coords(
        nc,
        rows,
        cols,
        x_name=x_name,
        y_name=y_name,
        cf_attr=cf_attr,
        coord_ndim=coord_ndim,
    )
    return nc, x_2d, y_2d, data_var
