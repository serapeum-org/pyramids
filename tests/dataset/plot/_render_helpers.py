"""Test adapter for calling ``render_array`` with the pre-RenderRequest kwarg style.

``render_array``'s 17 loose parameters were grouped into a :class:`RenderRequest`
(composing :class:`RgbSpec` and :class:`ModeSpec`) to clear SonarCloud
``python:S107`` — see issue #1006. These plot tests exercise ``render_array``'s
internal kwarg routing and predate that change, so this thin adapter rebuilds a
``RenderRequest`` from loose kwargs and forwards the styling kwargs verbatim.
New tests should construct :class:`RenderRequest` directly.
"""

from dataclasses import fields
from typing import Any

from pyramids.dataset._plot_helpers import (
    ModeSpec,
    RenderRequest,
    RgbSpec,
)
from pyramids.dataset._plot_helpers import render_array as _render_array

# Derive the field-name sets from the dataclasses so the loose-kwarg split can never
# drift from RenderRequest / RgbSpec / ModeSpec when a field is added or renamed.
_RGB_FIELDS = frozenset(f.name for f in fields(RgbSpec))
_MODE_FIELDS = frozenset(f.name for f in fields(ModeSpec))
# The RenderRequest-level fields, minus the two sub-object fields built separately.
_REQUEST_FIELDS = frozenset(f.name for f in fields(RenderRequest)) - {"rgb", "mode"}


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
