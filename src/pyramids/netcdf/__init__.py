"""NetCDF subpackage for pyramids."""

from __future__ import annotations

from pyramids.netcdf.plot_options import ColourOpts, FacetSpec, Selectors
from pyramids.netcdf.metadata import from_json, get_metadata, to_dict, to_json
from pyramids.netcdf.models import (
    CFInfo,
    DimensionInfo,
    GroupInfo,
    NetCDFMetadata,
    StructuralInfo,
    VariableInfo,
)
from pyramids.netcdf.netcdf import NetCDF
from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.labeled import LabeledDataset

__all__ = [
    "NetCDF",
    "UgridDataset",
    "LabeledDataset",
    "NetCDFMetadata",
    "CFInfo",
    "DimensionInfo",
    "VariableInfo",
    "GroupInfo",
    "StructuralInfo",
    "Selectors",
    "ColourOpts",
    "FacetSpec",
    "get_metadata",
    "to_json",
    "from_json",
    "to_dict",
]
