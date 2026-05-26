"""Affine unit conversion for raster band values.

Backs :meth:`pyramids.dataset.Dataset.convert_units`. Rather than pulling in a
general unit-conversion dependency, pyramids ships a small affine lookup table for
the common meteorological/geophysical conversions (each entry is a ``(scale, offset)``
pair applied as ``value * scale + offset``). Extend :data:`_AFFINE` as new pairs are
needed. No new third-party dependencies.
"""

from __future__ import annotations

import numpy as np

# (source_unit, target_unit) -> (scale, offset); value_out = value_in * scale + offset.
# Reverse directions are listed explicitly so round-trips are exact.
_AFFINE: dict[tuple[str, str], tuple[float, float]] = {
    ("K", "celsius"): (1.0, -273.15),
    ("celsius", "K"): (1.0, 273.15),
    ("m s-1", "knots"): (1.943844, 0.0),
    ("knots", "m s-1"): (1.0 / 1.943844, 0.0),
    ("Pa", "hPa"): (0.01, 0.0),
    ("hPa", "Pa"): (100.0, 0.0),
    ("m", "mm"): (1000.0, 0.0),
    ("mm", "m"): (0.001, 0.0),
}


def supported_conversions() -> list[tuple[str, str]]:
    """List the ``(source, target)`` unit pairs the affine table can convert.

    Returns:
        A sorted list of ``(source_unit, target_unit)`` tuples accepted by
        :func:`convert_array`.

    Examples:
        - Inspect a few of the supported pairs:
            ```python
            >>> from pyramids.dataset.ops.units import supported_conversions
            >>> pairs = supported_conversions()
            >>> ("K", "celsius") in pairs
            True
            >>> ("Pa", "hPa") in pairs
            True

            ```
    """
    return sorted(_AFFINE.keys())


def convert_array(array: np.ndarray, source: str, target: str) -> np.ndarray:
    """Convert an array of values from ``source`` units to ``target`` units.

    Args:
        array: Values to convert.
        source: Source unit label (e.g. ``"K"``). Must be non-empty.
        target: Target unit label (e.g. ``"celsius"``).

    Returns:
        A new array of converted values. When ``source == target`` the input is
        returned unchanged.

    Raises:
        ValueError: ``source`` is empty (no unit to convert from), or the
            ``(source, target)`` pair is not in the affine table.

    Examples:
        - Convert Kelvin to Celsius:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset.ops.units import convert_array
            >>> convert_array(np.array([273.15, 283.15]), "K", "celsius").tolist()
            [0.0, 10.0]

            ```
        - Converting to the same unit is a no-op:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset.ops.units import convert_array
            >>> convert_array(np.array([5.0]), "Pa", "Pa").tolist()
            [5.0]

            ```
        - An unknown pair is rejected:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset.ops.units import convert_array
            >>> try:
            ...     convert_array(np.array([1.0]), "K", "furlongs")
            ... except ValueError as exc:
            ...     print("No unit conversion" in str(exc))
            True

            ```
    """
    if not source:
        raise ValueError(
            f"cannot convert a band with no source unit to {target!r}; set "
            "band_units on the dataset before calling convert_units."
        )
    if source == target:
        result = array
    else:
        key = (source, target)
        if key not in _AFFINE:
            raise ValueError(
                f"No unit conversion from {source!r} to {target!r}. Supported pairs: "
                f"{supported_conversions()}."
            )
        scale, offset = _AFFINE[key]
        result = array * scale + offset
    return result
