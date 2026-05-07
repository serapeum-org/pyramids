"""Engine classes for ``Dataset`` operations.

After the L-2 composition refactor, the seven public-API families on
``pyramids.dataset.Dataset`` live as engine instances accessed via
attribute (``ds.io``, ``ds.spatial``, ``ds.bands``, ``ds.analysis``,
``ds.cell``, ``ds.vectorize``, ``ds.cog``). Each engine owns the
methods of one family; ``Dataset`` exposes same-named facade methods
so ``ds.crop(mask)`` and ``ds.spatial.crop(mask)`` are equivalent.

Engines hold a weakref proxy back to the parent Dataset; the proxy
keeps GDAL handle release deterministic on Windows. See
:class:`pyramids.dataset.engines._base._Engine` for the contract.
"""

from pyramids.dataset.engines.analysis import Analysis
from pyramids.dataset.engines.bands import Bands
from pyramids.dataset.engines.cell import Cell
from pyramids.dataset.engines.cog import COG
from pyramids.dataset.engines.io import IO
from pyramids.dataset.engines.spatial import Spatial
from pyramids.dataset.engines.vectorize import Vectorize

__all__ = [
    "Analysis",
    "Bands",
    "COG",
    "Cell",
    "IO",
    "Spatial",
    "Vectorize",
]
