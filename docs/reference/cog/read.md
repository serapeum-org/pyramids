# Reading, inspecting & validating

The read side of the COG surface: inspect structure without touching pixels, validate that
a file is a valid COG, and read only the pixels you need via overview-decimated partial
reads.

- **Inspect** — `cog_info` reads only headers/metadata (compression, predictor, blocksize,
  dtype, CRS/bounds/resolution, the overview pyramid, per-band tags, colour table) and
  returns a frozen `COGInfo`. Cheap even for a large remote COG.
- **Validate** — `validate` returns a `ValidationReport` (usable as a bool); `Dataset.is_cog`
  is a fast metadata-only probe and `Dataset.validate_cog` is the authoritative check.
- **Partial reads** — `read_part` / `preview` / `point` / `read_tile` request a smaller
  output size so GDAL serves from the nearest overview, fetching only the relevant byte
  ranges over `/vsicurl/`.

## Structured inspection

::: pyramids.dataset.cog.inspect
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["cog_info", "COGInfo", "OverviewLevel"]

## Overview-decimated reads

::: pyramids.dataset.engines.cog.COG
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["read_part", "preview", "point", "read_tile", "is_cog",
            "validate_cog", "info"]

## Validation

::: pyramids.dataset.cog.validate
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["validate", "ValidationReport"]
