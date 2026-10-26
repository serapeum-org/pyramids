"""A reusable conformance kit for ``pyramids.plot_backends`` plugins.

A downstream distribution that registers a plot backend (e.g. digitalearth's interactive / 3-D /
web tiers) runs these checks against its render *callable* in its own test suite — under any test
runner, since the kit raises plain :class:`AssertionError` and needs no pytest. It verifies the same
contract :mod:`pyramids.base._plot_backends` relies on at discovery and dispatch: the object the
entry point loads is **callable**, and its signature accepts the render call shape the seam uses —
``fn(obj, *, backend, **kwargs)`` — so a registered backend is never silently skipped and
:func:`~pyramids.base._plot_backends.render_via_backend` can always call it.

What it cannot check is the render itself: a backend turns a pyramids object into its own figure,
which needs that object and the backend's heavy drawing stack. The kit validates the *interface*,
not the picture.

Usage in a plugin's tests::

    from pyramids.base.plot_backend_conformance import assert_plot_backend_conforms
    from your_package.backends import interactive

    def test_backend_conforms():
        assert_plot_backend_conforms(interactive)

See the [Writing a plot backend](../../reference/writing-a-plot-backend.md) guide for the full
contract and the ``pyramids.plot_backends`` entry-point registration.
"""

from __future__ import annotations

import inspect
from typing import Any

__all__ = ["check_plot_backend", "assert_plot_backend_conforms"]


def check_plot_backend(backend: Any) -> list[str]:
    """Return every way a plot-backend callable departs from the contract.

    Args:
        backend: The render callable an entry point in the ``pyramids.plot_backends`` group loads —
            the callable itself, not its name.

    Returns:
        A list of human-readable problems; empty when the backend conforms. The first entry, if any,
        is the one discovery itself would reject on (a non-callable).

    Examples:
        - A well-formed backend callable has no problems:
            ```python
            >>> from pyramids.base.plot_backend_conformance import check_plot_backend
            >>> def interactive(obj, *, backend, **kwargs):
            ...     ...  # render obj and return the backend's own object
            >>> check_plot_backend(interactive)
            []

            ```
        - A non-callable is rejected (discovery would skip it with a warning):
            ```python
            >>> check_plot_backend("not-a-callable")
            ['object is not callable; the pyramids.plot_backends entry point must load a callable']

            ```
        - A callable without ``**kwargs`` cannot receive the forwarded arguments, and is reported:
            ```python
            >>> def too_rigid(obj, *, backend):  # no **kwargs
            ...     ...
            >>> problems = check_plot_backend(too_rigid)
            >>> len(problems)
            1
            >>> "**kwargs" in problems[0]
            True

            ```
    """
    problems: list[str] = []
    if not callable(backend):
        problems.append(
            "object is not callable; the pyramids.plot_backends entry point must load a callable"
        )
    else:
        try:
            signature: inspect.Signature | None = inspect.signature(backend)
        except (TypeError, ValueError):
            # A builtin or C callable with no introspectable signature; being callable is all
            # discovery can enforce, so accept it rather than report a false problem.
            signature = None
        if signature is not None:
            try:
                signature.bind("conformance-object", backend="matplotlib")
            except TypeError as error:
                problems.append(
                    "callable does not accept the render call shape "
                    f"fn(obj, *, backend, **kwargs): {error}"
                )
            else:
                has_var_keyword = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
                if not has_var_keyword:
                    problems.append(
                        "callable has no **kwargs, so it cannot receive the facade's forwarded "
                        "arguments"
                    )
    return problems


def assert_plot_backend_conforms(backend: Any) -> None:
    """Raise :class:`AssertionError` if a plot-backend callable departs from the contract.

    A thin wrapper over :func:`check_plot_backend` for use directly in a plugin's tests.

    Args:
        backend: The render callable to check.

    Raises:
        AssertionError: When the backend does not conform; the message lists every problem.

    Examples:
        - A conforming backend passes silently:
            ```python
            >>> from pyramids.base.plot_backend_conformance import assert_plot_backend_conforms
            >>> def web(obj, *, backend, **kwargs):
            ...     ...
            >>> assert_plot_backend_conforms(web) is None
            True

            ```
        - A non-conforming backend raises with the reason:
            ```python
            >>> try:
            ...     assert_plot_backend_conforms(42)
            ... except AssertionError as exc:
            ...     print(exc)
            plot backend does not conform to pyramids.plot_backends:
            - object is not callable; the pyramids.plot_backends entry point must load a callable

            ```
    """
    problems = check_plot_backend(backend)
    if problems:
        raise AssertionError(
            "plot backend does not conform to pyramids.plot_backends:\n- "
            + "\n- ".join(problems)
        )
