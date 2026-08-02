"""Typed cleopatra plot specs, re-exported for pyramids.

pyramids' ``plot`` and ``animate`` methods accept and forward the cleopatra
plotting spec objects introduced in cleopatra 0.27. Import them from here so a
call site never has to reach into ``cleopatra`` directly:

```python
>>> from pyramids.plot import ColorBar, PointOverlay      # doctest: +SKIP
>>> ds.plot(                                              # doctest: +SKIP
...     colorbar=ColorBar(location="bottom", label_color="black"),
...     points=PointOverlay(points, color="red"),
... )

```

Each name resolves lazily from the installed cleopatra (the ``[viz]`` extra), so
importing :mod:`pyramids.plot` never forces cleopatra to be installed; accessing
a spec without it raises
:class:`~pyramids.base._errors.OptionalPackageDoesNotExist` with the install hint.

Available specs:

- ``ColorBar`` — colour-bar placement and appearance, passed as ``colorbar=``.
- ``FrameLabel`` — per-frame animation label, passed as ``frame_label=``.
- ``PointOverlay`` — styled point overlay, passed as ``points=``.
- ``Basemap`` / ``Feature`` — shaded-relief / coastline reference layers, passed
  as ``basemap=<Basemap>`` (distinct from a web-tile ``basemap="<provider>"``).
"""

from typing import TYPE_CHECKING, Any

from pyramids.base._utils import require_optional

if TYPE_CHECKING:  # names for static type checkers / IDEs; resolved lazily at runtime
    from cleopatra.array_glyph import ColorBar, FrameLabel, PointOverlay
    from cleopatra.geo import Basemap, Feature

_CLEO_EXPORTS: dict[str, tuple[str, str]] = {
    "ColorBar": ("cleopatra.array_glyph", "ColorBar"),
    "FrameLabel": ("cleopatra.array_glyph", "FrameLabel"),
    "PointOverlay": ("cleopatra.array_glyph", "PointOverlay"),
    "Basemap": ("cleopatra.geo", "Basemap"),
    "Feature": ("cleopatra.geo", "Feature"),
}

__all__ = sorted(_CLEO_EXPORTS)

_VIZ_HINT = (
    "The pyramids plotting specs require cleopatra (the [viz] extra). "
    "Install with: pip install 'pyramids-gis[viz]'."
)


def __getattr__(name: str) -> Any:
    """Resolve a cleopatra plot spec lazily, or raise the [viz] install hint."""
    target = _CLEO_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = require_optional(module_name, _VIZ_HINT, return_module=True)
    return getattr(module, attribute)


def __dir__() -> list[str]:
    """List the re-exported spec names (for tab-completion)."""
    return __all__
