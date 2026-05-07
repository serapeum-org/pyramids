"""Base classes shared by every engine in :mod:`pyramids.dataset.engines`.

Defines :class:`_Engine`, the weakref-proxy holder that every public
engine class subclasses, plus the pickle placeholder used when a
caller pickles an engine directly. Also exposes the module-level
``logger`` that staticmethods (which have no ``self._ds`` to reach
the Dataset's logger through) fall back on.
"""

from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


# Module-level logger used by engine staticmethods that have no
# ``self._ds`` to reach the Dataset's logger through.
logger = logging.getLogger("pyramids.dataset.engines")


class _Placeholder:
    """Stand-in returned by `_Collaborator.__reduce__`.

    Exists only as the unpickle target for a directly-pickled
    collaborator. `Dataset.__init__` creates fresh collaborators
    on Dataset unpickle, overwriting any placeholder that would
    otherwise be attached. If user code ever observes a
    `_Placeholder` instance, the unpickle sequence has been
    interrupted — open a bug.
    """


def _recreate_placeholder() -> _Placeholder:
    return _Placeholder()


class _Engine:
    """Base class for every Dataset collaborator.

    Holds a **weak** back-reference to the parent `Dataset`. The
    weakref is essential: a strong `_ds` reference creates a cycle
    (`ds -> ds.spatial -> ds`) that the cycle collector eventually
    breaks but that delays GDAL handle release long enough to fail
    Windows file-unlink in tests (and to leak file descriptors in
    long-running processes). xarray uses the same pattern for
    accessors. `weakref.proxy` is transparent — `self._ds.crs`
    works as if `_ds` were a real reference — so collaborator
    method bodies don't need to know the back-reference is weak.

    Also overrides `__reduce__` so direct collaborator pickling
    (`pickle.dumps(ds.io)`) produces a placeholder rather than a
    circular pickle through `_ds`.
    """

    # Single-slot class. Forbids extra attributes (catches typos
    # and accidental state on engines) but still allows reassigning
    # the slot itself — which `Dataset._update_inplace` relies on
    # to re-bind the back-reference after `__dict__.update`.
    __slots__ = ("_ds",)

    def __init__(self, ds: Dataset) -> None:
        # `weakref.proxy` so the back-reference does not create a
        # strong cycle with the parent Dataset. See class docstring.
        self._ds = weakref.proxy(ds)

    def __reduce__(self) -> tuple[Any, tuple]:
        return (_recreate_placeholder, ())
