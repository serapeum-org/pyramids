"""The one-line facades `NetCDF` exposes for the selection engine's newest members.

`NetCDF.rolling` and its eight siblings do nothing but forward to the `Selection` method of the
same name. That is one line each, and the way a one-line facade goes wrong is silently: a
keyword dropped, a default restated differently from the engine's, or a positional passed in the
wrong order. This module pins the signatures against the engine's own and pins that calling the
facade answers what calling the engine directly answers.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines.selection import Selection

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
NT, NY, NX = 4, 3, 2
TIMES = [0.0, 6.0, 12.0, 18.0]

CALLS: dict[str, tuple[tuple, dict[str, Any]]] = {
    "rolling": (("time", 3), {"how": "max", "center": True, "min_periods": 1}),
    "diff": (("time", 2), {"label": "lower"}),
    "cumsum": (("time",), {"skipna": False}),
    "shift": (("time", -2), {"fill_value": 0.0}),
    "argmin": (("time",), {"skipna": False}),
    "argmax": (("time",), {"skipna": False}),
    "idxmin": (("time",), {"skipna": False}),
    "idxmax": (("time",), {"skipna": False}),
}
"""One call per along-dimension facade, every argument away from its default."""

MEMBERS = tuple(CALLS) + ("weighted",)


def _container() -> NetCDF:
    """A four-step container holding `v(time, y, x)`, one cell a gap.

    Returns:
        NetCDF: The container.
    """
    rng = np.random.default_rng(1337)
    values = np.round(rng.uniform(-5.0, 15.0, size=(NT, NY, NX)), 2)
    values[1, 0, 1] = -9999.0
    return NetCDF.from_array(
        values,
        geo_ref=GEO,
        variable_name="v",
        no_data_value=-9999.0,
        dims=ExtraDimensions(name="time", values=list(TIMES)),
    )


def _weights() -> np.ndarray:
    """One weight per step of `time`, none of them equal.

    Returns:
        np.ndarray: The weights.
    """
    return np.array([3.0, 1.0, 2.0, 4.0])


def _through_the_facade(member: str) -> NetCDF:
    """Call `member` on the container itself.

    Args:
        member: One of `MEMBERS`.

    Returns:
        NetCDF: The result.
    """
    container = _container()
    if member == "weighted":
        result = container.weighted(_weights(), "time", how="sum", skipna=False)
    else:
        args, kwargs = CALLS[member]
        result = getattr(container, member)(*args, **kwargs)
    return result


def _through_the_engine(member: str) -> NetCDF:
    """Call `member` on the container's selection engine, with the same arguments.

    Args:
        member: One of `MEMBERS`.

    Returns:
        NetCDF: The result.
    """
    container = _container()
    if member == "weighted":
        result = container.selection.weighted(
            _weights(), "time", how="sum", skipna=False
        )
    else:
        args, kwargs = CALLS[member]
        result = getattr(container.selection, member)(*args, **kwargs)
    return result


def _read(nc: NetCDF) -> np.ndarray:
    """The container's one variable as a float64 array.

    Args:
        nc: The container.

    Returns:
        np.ndarray: The values.
    """
    return np.asarray(nc.get_variable("v").read_array(), dtype="float64")


def _declared_gap(nc: NetCDF) -> tuple:
    """The no-data value of the container's one variable, NaN reported as a name.

    A NaN never equals itself, so the two routes' declarations are compared through a token
    that does.

    Args:
        nc: The container.

    Returns:
        tuple: One entry per band, `"nan"` wherever the band declares a NaN.
    """
    declared = nc.get_variable("v").no_data_value
    return tuple(
        "nan" if isinstance(value, float) and np.isnan(value) else value
        for value in declared
    )


class TestTheFacadesMatchTheEngine:
    """Each facade's signature is the engine method's, so no argument can be lost in between."""

    @pytest.mark.parametrize("member", MEMBERS)
    def test_the_parameters_are_the_same(self, member):
        """Names, kinds and order match, so a positional cannot land on another parameter.

        Args:
            member: The member whose two signatures are compared.
        """
        facade = list(inspect.signature(getattr(NetCDF, member)).parameters.values())
        engine = list(inspect.signature(getattr(Selection, member)).parameters.values())
        assert [(p.name, p.kind) for p in facade] == [
            (p.name, p.kind) for p in engine
        ], f"{member}() signatures differ"

    @pytest.mark.parametrize("member", MEMBERS)
    def test_the_defaults_are_the_same(self, member):
        """A facade restating a default differently would change behaviour without saying so.

        Args:
            member: The member whose two signatures are compared.
        """
        facade = inspect.signature(getattr(NetCDF, member)).parameters
        engine = inspect.signature(getattr(Selection, member)).parameters
        assert {n: p.default for n, p in facade.items()} == {
            n: p.default for n, p in engine.items()
        }, f"{member}() defaults differ"

    @pytest.mark.parametrize("member", MEMBERS)
    def test_the_facade_exists_on_the_class(self, member):
        """The facade is a method of `NetCDF`, not something reached only through `selection`.

        Args:
            member: The member that must be public on `NetCDF`.
        """
        assert callable(getattr(NetCDF, member, None)), (
            f"NetCDF.{member} is not callable"
        )


class TestTheFacadesForwardEveryArgument:
    """Calling the facade answers exactly what calling the engine method answers."""

    @pytest.mark.parametrize("member", MEMBERS)
    def test_the_values_match(self, member):
        """Every argument reaches the engine, so both routes compute the same array.

        Args:
            member: The member called both ways.

        Test scenario:
            Each call passes every argument away from its default, so a keyword the facade
            forgot to forward would leave the engine on its own default and the arrays would
            differ.
        """
        assert_array_equal(
            _read(_through_the_facade(member)),
            _read(_through_the_engine(member)),
            err_msg=f"{member}() answered differently through the facade",
        )

    @pytest.mark.parametrize("member", MEMBERS)
    def test_the_band_layout_matches(self, member):
        """The dimension the member keeps or removes is the same on both routes.

        Args:
            member: The member called both ways.
        """
        facade = _through_the_facade(member).get_variable("v")
        engine = _through_the_engine(member).get_variable("v")
        assert tuple(facade._band_dim_names) == tuple(engine._band_dim_names), (
            f"{member}() band dimensions differ"
        )

    @pytest.mark.parametrize("member", MEMBERS)
    def test_the_declared_gap_matches(self, member):
        """The no-data value the result declares is the same on both routes.

        Args:
            member: The member called both ways.
        """
        facade = _declared_gap(_through_the_facade(member))
        engine = _declared_gap(_through_the_engine(member))
        assert facade == engine, f"{member}() declared {facade!r} against {engine!r}"

    def test_rolling_forwards_its_quantile(self):
        """`q` is the one argument only `rolling` takes, and it must reach the engine.

        Test scenario:
            A facade that dropped `q` would raise from the engine's own check — `q=` is
            required with `how="quantile"` — rather than compute the quantile asked for.
        """
        through_engine = _container()
        facade = _container().rolling("time", 3, how="quantile", q=0.25, min_periods=1)
        engine = through_engine.selection.rolling(
            "time", 3, how="quantile", q=0.25, min_periods=1
        )
        assert_array_equal(_read(facade), _read(engine), err_msg="q= was not forwarded")

    def test_weighted_forwards_its_dims_positionally(self):
        """`dims` is the facade's second positional, so it must not land on a keyword.

        Test scenario:
            Weighting `time` keeps the grid while the default weights both spatial axes and
            reduces it to one cell, so a misplaced `dims` is visible in the result's shape.
        """
        variable = _container().weighted(_weights(), "time").get_variable("v")
        assert (variable.rows, variable.columns) == (NY, NX), (
            f"weighted() reduced the grid to {(variable.rows, variable.columns)}"
        )


class TestTheFacadesReachAVariable:
    """A variable is a `NetCDF` too, so the same facades answer a variable for a variable."""

    @pytest.mark.parametrize("member", MEMBERS)
    def test_a_variable_receiver_matches_the_engine(self, member):
        """Called on a variable the facade forwards to the engine bound to that variable.

        Args:
            member: The member called both ways.
        """
        held = _container()
        variable = held.get_variable("v")
        other = _container()
        through_engine = other.get_variable("v")
        if member == "weighted":
            facade = variable.weighted(_weights(), "time", how="sum", skipna=False)
            engine = through_engine.selection.weighted(
                _weights(), "time", how="sum", skipna=False
            )
        else:
            args, kwargs = CALLS[member]
            facade = getattr(variable, member)(*args, **kwargs)
            engine = getattr(through_engine.selection, member)(*args, **kwargs)
        assert_array_equal(
            np.asarray(facade.read_array(), dtype="float64"),
            np.asarray(engine.read_array(), dtype="float64"),
            err_msg=f"{member}() on a variable answered differently through the facade",
        )
