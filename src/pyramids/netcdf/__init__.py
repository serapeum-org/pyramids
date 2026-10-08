"""NetCDF subpackage for pyramids."""

from __future__ import annotations

from pyramids.netcdf._lazy_cube import LazyNetCDF
from pyramids.netcdf.array_options import (
    CFAttributes,
    Encoding,
    ExtraDimensions,
    GeoReference,
)
from pyramids.netcdf.engines.selection import CumulativeAccessor
from pyramids.netcdf.labeled import LabeledArray, LabeledDataset
from pyramids.netcdf.metadata import from_json, get_metadata, to_dict, to_json
from pyramids.netcdf.models import (
    CFInfo,
    DimensionInfo,
    GroupInfo,
    NetCDFMetadata,
    StructuralInfo,
    VariableInfo,
)
from pyramids.netcdf.netcdf import Container, NetCDF, Variable
from pyramids.netcdf.plot_options import (
    CoordinateSpec,
    FacetSpec,
    Selectors,
)
from pyramids.netcdf.ugrid import UgridDataset

__all__ = [
    "CFAttributes",
    "CFInfo",
    "Container",
    "CoordinateSpec",
    "CumulativeAccessor",
    "DimensionInfo",
    "Encoding",
    "ExtraDimensions",
    "FacetSpec",
    "from_json",
    "GeoReference",
    "get_metadata",
    "GroupInfo",
    "LabeledArray",
    "LabeledDataset",
    "LazyNetCDF",
    "NetCDF",
    "NetCDFMetadata",
    "Selectors",
    "StructuralInfo",
    "to_dict",
    "to_json",
    "UgridDataset",
    "Variable",
    "VariableInfo",
]
