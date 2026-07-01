# Cell & Coordinate Operations

Cell coordinate retrieval, cell polygons/points, and map-to-array coordinate conversions.

```mermaid
flowchart LR
    CE(("Cell<br/>ds.cell"))
    CE --> G["<b>cell geometry</b><br/>get_cell_coords<br/>get_cell_polygons · get_cell_points"]
    CE --> P["<b>pixel ↔ map</b><br/>array_to_map_coordinates<br/>map_to_array_coordinates"]
```

::: pyramids.dataset.engines.Cell
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
