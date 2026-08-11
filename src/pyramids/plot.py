"""Typed cleopatra plot specs, re-exported for pyramids.

pyramids' ``plot`` and ``animate`` methods accept and forward the cleopatra
plotting spec objects (the ``[viz]`` extra requires cleopatra >= 0.30, which
carries the complete spec API). Import them from here so a call site never has
to reach into ``cleopatra`` directly:

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

- ``ColorBar`` — the complete colour-bar spec (caption ``label`` / ``length`` /
  orientation / placement / appearance), passed as ``colorbar=``.
- ``FrameLabel`` — per-frame animation label, passed as ``frame_label=``.
- ``PointOverlay`` — styled point overlay, passed as ``points=``.
- ``Basemap`` / ``Feature`` — shaded-relief / coastline reference layers, passed
  as ``basemap=<Basemap>`` (distinct from a web-tile ``basemap="<provider>"``).
- ``ColorScaling`` — the colour-scale spec (linear / power / sym-log / boundary /
  midpoint norm + colour-bar construction), passed as ``color=``. Its variant
  constructors are ``ColorScaling.power(gamma=...)`` etc.
- ``Contour`` — contour-line control (``levels`` / ``labels`` / ``label_kw``),
  passed as ``contour=``.
- ``CellValues`` — per-cell value annotation (``show`` / ``size`` /
  ``background_threshold``), passed as ``cells=``.
- ``DataStyle`` — data-style preset + relief (``style`` / ``hillshade`` / ``bands`` /
  ``alpha``), passed as ``data_style=``.
- ``Classify`` — value classification for vector glyphs (``scheme`` / ``k``), passed
  as ``classify=``.
- ``PanelLabels`` — per-panel coordinate labels for faceting (``col`` / ``row``),
  passed as ``labels=`` on the facet path.
"""

from typing import TYPE_CHECKING, Any

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import require_optional

if TYPE_CHECKING:  # names for static type checkers / IDEs; resolved lazily at runtime
    # Re-exported for typing only (runtime resolution is via ``__getattr__``), so
    # ruff's unused-import rule is silenced rather than deleting the typing intent.
    from cleopatra.basemap.geo import Basemap, Feature  # noqa: F401
    from cleopatra.glyphs.gridded.array_glyph import (  # noqa: F401
        FrameLabel,
        PanelLabels,
        PointOverlay,
    )
    from cleopatra.styling.colorbar import ColorBar  # noqa: F401
    from cleopatra.styling.params import (  # noqa: F401
        CellValues,
        Classify,
        Contour,
        DataStyle,
    )
    from cleopatra.styling.scaling import ColorScaling  # noqa: F401

_ARRAY_GLYPH = "cleopatra.glyphs.gridded.array_glyph"
_COLORBAR = "cleopatra.styling.colorbar"
_GEO = "cleopatra.basemap.geo"
_PARAMS = "cleopatra.styling.params"
_SCALING = "cleopatra.styling.scaling"

_CLEO_EXPORTS: dict[str, tuple[str, str]] = {
    "ColorBar": (_COLORBAR, "ColorBar"),
    "FrameLabel": (_ARRAY_GLYPH, "FrameLabel"),
    "PointOverlay": (_ARRAY_GLYPH, "PointOverlay"),
    "PanelLabels": (_ARRAY_GLYPH, "PanelLabels"),
    "Basemap": (_GEO, "Basemap"),
    "Feature": (_GEO, "Feature"),
    "ColorScaling": (_SCALING, "ColorScaling"),
    "Contour": (_PARAMS, "Contour"),
    "CellValues": (_PARAMS, "CellValues"),
    "DataStyle": (_PARAMS, "DataStyle"),
    "Classify": (_PARAMS, "Classify"),
}

__all__ = sorted(_CLEO_EXPORTS)

_VIZ_HINT = (
    "The pyramids plotting specs require cleopatra (the [viz] extra). "
    "Install with: pip install 'pyramids-gis[viz]'."
)


def __getattr__(name: str) -> Any:
    """Resolve a cleopatra plot spec lazily, or raise the [viz]/upgrade hint."""
    target = _CLEO_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = require_optional(module_name, _VIZ_HINT, return_module=True)
    try:
        spec = getattr(module, attribute)
    except AttributeError as exc:
        # cleopatra is importable but too old to carry this spec — surface the
        # branded upgrade hint instead of a bare AttributeError.
        raise OptionalPackageDoesNotExist(
            f"cleopatra is installed but too old to provide `{attribute}`. The "
            "pyramids plotting specs require cleopatra >= 0.30; upgrade with: "
            "pip install -U 'pyramids-gis[viz]'."
        ) from exc
    return spec


def __dir__() -> list[str]:
    """List the re-exported spec names (for tab-completion)."""
    return __all__
