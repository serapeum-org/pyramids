"""Public pyramids exception hierarchy.

Every exception raised by pyramids inherits from :class:`PyramidsError`,
so a caller that wants to catch any pyramids failure can write::

    from pyramids import errors

    try:
        ds = Dataset.read_file(path)
    except errors.PyramidsError:
        ...

Finer-grained handlers stay useful: most concrete classes also inherit
from a matching Python builtin (`ValueError` / `RuntimeError`) so
existing `except ValueError:` / `except RuntimeError:` blocks keep
working without change.

The pyramids-emitted warning categories (`ContainerRasterWarning`,
`GeometryWarning`) are re-exported here too, so a caller who wants to
filter one writes ``warnings.filterwarnings("ignore",
category=errors.ContainerRasterWarning)`` against this public module.

The implementation lives in :mod:`pyramids.base._errors`; that module
is private and its signatures may change without notice. Always import
exception classes from here instead.
"""

from __future__ import annotations

from pyramids.base._errors import (
    AlignmentError,
    ContainerRasterWarning,
    CoverageError,
    CRSError,
    DatasetNotFoundError,
    DriverNotExistError,
    DtypeNarrowingWarning,
    FailedToSaveError,
    FeatureError,
    FileFormatNotSupportedError,
    GeometryWarning,
    InvalidGeometryError,
    NoDataValueError,
    OGCAPIError,
    OptionalPackageDoesNotExist,
    OutOfBoundsError,
    OverviewTargetError,
    ReadOnlyError,
    VectorDriverError,
    WCSError,
    WFSError,
    WMSError,
)
from pyramids.base._errors import _PyramidsError as PyramidsError

__all__ = [
    "PyramidsError",
    "AlignmentError",
    "ContainerRasterWarning",
    "DtypeNarrowingWarning",
    "CoverageError",
    "CRSError",
    "DatasetNotFoundError",
    "DriverNotExistError",
    "FailedToSaveError",
    "FeatureError",
    "FileFormatNotSupportedError",
    "GeometryWarning",
    "InvalidGeometryError",
    "NoDataValueError",
    "OGCAPIError",
    "OptionalPackageDoesNotExist",
    "OutOfBoundsError",
    "OverviewTargetError",
    "ReadOnlyError",
    "VectorDriverError",
    "WCSError",
    "WFSError",
    "WMSError",
]
