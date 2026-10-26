"""The ``pyramids.plot_backends`` conformance kit.

A plugin registers a render callable; the kit lets it assert, in its own test suite, that the
callable satisfies the ``fn(obj, *, backend, **kwargs)`` contract the seam dispatches through. These
tests pin what conforms and what does not, mirroring the shapes a real backend (digitalearth's
``quickmap`` adapter) and a broken one take.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pytest

from pyramids.base import plot_backend_conformance as module
from pyramids.base.plot_backend_conformance import (
    assert_plot_backend_conforms,
    check_plot_backend,
)

pytestmark = pytest.mark.core


def _good(obj: Any, *, backend: str, **kwargs: Any) -> Any:
    """A render callable with the exact seam call shape."""
    return obj, backend, kwargs


def _adapter(fn: Any, obj: Any, *, backend: str, **kwargs: Any) -> Any:
    """Stand-in for the seam's ``functools.partial`` quickmap adapter."""
    return fn(obj, backend=backend, **kwargs)


class TestConforms:
    """Callables that satisfy the render call shape."""

    def test_canonical_shape(self) -> None:
        """``fn(obj, *, backend, **kwargs)`` conforms with no problems."""
        assert check_plot_backend(_good) == []
        assert assert_plot_backend_conforms(_good) is None

    def test_lambda_shape(self) -> None:
        """A lambda of the same shape conforms."""
        assert check_plot_backend(lambda obj, *, backend, **kw: None) == []

    def test_partial_adapter_conforms(self) -> None:
        """A ``partial`` over the quickmap-adapter shape (what the seam registers) conforms."""
        backend = partial(_adapter, lambda obj, *, backend, **kw: obj)
        assert check_plot_backend(backend) == []

    def test_backend_absorbed_by_var_kwargs(self) -> None:
        """A callable without an explicit ``backend`` param still conforms via ``**kwargs``."""

        def render(obj: Any, **kwargs: Any) -> Any:
            return obj, kwargs

        assert check_plot_backend(render) == []

    def test_non_introspectable_callable_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A callable whose signature cannot be introspected is accepted on being callable alone."""

        def no_signature(_: Any) -> Any:
            raise ValueError("no signature found")

        monkeypatch.setattr(module.inspect, "signature", no_signature)
        assert check_plot_backend(_good) == []

    def test_callable_instance_conforms(self) -> None:
        """An instance whose ``__call__`` has the shape conforms (signature reads ``__call__``)."""

        class Renderer:
            def __call__(self, obj: Any, *, backend: str, **kwargs: Any) -> Any:
                return obj, backend, kwargs

        assert check_plot_backend(Renderer()) == []

    def test_positional_only_obj_conforms(self) -> None:
        """A positional-only ``obj`` conforms — the seam passes ``obj`` positionally."""

        def render(obj: Any, /, *, backend: str, **kwargs: Any) -> Any:
            return obj, backend

        assert check_plot_backend(render) == []

    def test_nested_partial_conforms(self) -> None:
        """A doubly-wrapped ``partial`` still introspects to the right shape."""
        inner = partial(_adapter, lambda obj, *, backend, **kw: obj)
        assert check_plot_backend(partial(inner)) == []


class TestDoesNotConform:
    """Objects the kit rejects, with a problem per reason."""

    @pytest.mark.parametrize("value", ["not-callable", 42, None, object()])
    def test_non_callable_is_rejected(self, value: Any) -> None:
        """A non-callable is the first-reported, discovery-rejecting problem."""
        problems = check_plot_backend(value)
        assert len(problems) == 1
        assert "not callable" in problems[0]

    def test_missing_var_kwargs_is_rejected(self) -> None:
        """Without ``**kwargs`` the callable cannot receive the forwarded arguments."""

        def too_rigid(obj: Any, *, backend: str) -> Any:
            return obj, backend

        problems = check_plot_backend(too_rigid)
        assert len(problems) == 1
        assert "**kwargs" in problems[0]

    def test_extra_required_positional_is_rejected(self) -> None:
        """A second required positional cannot be filled by ``fn(obj, backend=…)``."""

        def needs_two(first: Any, obj: Any, *, backend: str, **kwargs: Any) -> Any:
            return first, obj, backend

        problems = check_plot_backend(needs_two)
        assert len(problems) == 1
        assert "render call shape" in problems[0]

    def test_keyword_only_obj_is_rejected(self) -> None:
        """A keyword-only ``obj`` cannot be filled — the seam passes ``obj`` positionally."""

        def render(*, obj: Any, backend: str, **kwargs: Any) -> Any:
            return obj, backend

        problems = check_plot_backend(render)
        assert len(problems) == 1
        assert "render call shape" in problems[0]

    def test_callable_instance_without_var_kwargs_is_rejected(self) -> None:
        """An instance whose ``__call__`` lacks ``**kwargs`` cannot take forwarded arguments."""

        class Rigid:
            def __call__(self, obj: Any, *, backend: str) -> Any:
                return obj, backend

        problems = check_plot_backend(Rigid())
        assert len(problems) == 1
        assert "**kwargs" in problems[0]

    def test_param_named_like_probe_without_var_kwargs_is_rejected(self) -> None:
        """A param coinciding with the old sentinel name but no ``**kwargs`` is still rejected.

        The seam forwards real keys (`title`, `basemap`, …); a callable that binds one sentinel key
        but has no `**kwargs` would `TypeError` on any of them, so the kit must not be fooled by the
        parameter name.
        """

        def collides(
            obj: Any, *, backend: str, _pyramids_conformance_probe: Any = None
        ) -> Any:
            return obj, backend

        problems = check_plot_backend(collides)
        assert len(problems) == 1
        assert "**kwargs" in problems[0]

    def test_assert_raises_with_reason(self) -> None:
        """``assert_plot_backend_conforms`` raises ``AssertionError`` naming the problem."""
        with pytest.raises(
            AssertionError, match="does not conform to pyramids.plot_backends"
        ):
            assert_plot_backend_conforms("nope")
