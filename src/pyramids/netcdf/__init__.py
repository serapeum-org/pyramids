"""NetCDF subpackage for pyramids."""

from __future__ import annotations

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
from pyramids.netcdf.netcdf import NetCDF
from pyramids.netcdf.plot_options import (
    ColorOpts,
    ColourOpts,
    FacetSpec,
    Selectors,
)
from pyramids.netcdf.ugrid import UgridDataset

__all__ = [
    "NetCDF",
    "UgridDataset",
    "LabeledDataset",
    "LabeledArray",
    "NetCDFMetadata",
    "CFInfo",
    "DimensionInfo",
    "VariableInfo",
    "GroupInfo",
    "StructuralInfo",
    "Selectors",
    "ColorOpts",
    "ColourOpts",
    "FacetSpec",
    "get_metadata",
    "to_json",
    "from_json",
    "to_dict",
]
