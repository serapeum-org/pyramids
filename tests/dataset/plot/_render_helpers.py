"""Test adapter for calling ``render_array`` with the pre-RenderRequest kwarg style.

``render_array``'s 17 loose parameters were grouped into a :class:`RenderRequest`
(composing :class:`RgbSpec` and :class:`ModeSpec`) to clear SonarCloud
``python:S107`` — see issue #1006. These plot tests exercise ``render_array``'s
internal kwarg routing and predate that change, so this thin adapter rebuilds a
``RenderRequest`` from loose kwargs and forwards the styling kwargs verbatim.
New tests should construct :class:`RenderRequest` directly.
"""

from typing import Any

from pyramids.dataset._plot_helpers import (
    ModeSpec,
    RenderRequest,
    RgbSpec,
)
from pyramids.dataset._plot_helpers import render_array as _render_array

_REQUEST_FIELDS = frozenset(
    {"arr", "extent", "coords", "exclude_value", "ax", "fig", "basemap", "basemap_epsg"}
)
_RGB_FIELDS = frozenset({"rgb", "surface_reflectance", "cutoff", "percentile"})
_MODE_FIELDS = frozenset(
    {"mode", "animation_axis_values", "data_getter", "facet_kwargs"}
)


def render_array(**kwargs: Any) -> Any:
    """Build a ``RenderRequest`` from loose kwargs and forward styling as ``**kwargs``.

    Splits the caller's kwargs into the :class:`RenderRequest` fields, the
    :class:`RgbSpec` fields, and the :class:`ModeSpec` fields; everything left
    over is a styling kwarg forwarded verbatim to ``render_array``.
    """
    fields = {k: kwargs.pop(k) for k in list(kwargs) if k in _REQUEST_FIELDS}
    rgb = {k: kwargs.pop(k) for k in list(kwargs) if k in _RGB_FIELDS}
    mode = {k: kwargs.pop(k) for k in list(kwargs) if k in _MODE_FIELDS}
    request = RenderRequest(rgb=RgbSpec(**rgb), mode=ModeSpec(**mode), **fields)
    return _render_array(request, **kwargs)
