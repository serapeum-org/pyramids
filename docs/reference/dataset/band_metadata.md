# Band Metadata & NoData

Band names, color tables, attribute tables, color interpretation, and no-data handling.

```mermaid
flowchart LR
    BA(("Bands<br/>ds.bands"))
    BA --> AT["<b>attribute tables</b><br/>get_attribute_table<br/>set_attribute_table"]
    BA --> CO["<b>colours</b><br/>band_color · get_band_by_color<br/>color_table"]
    BA --> BN["<b>bands & no-data</b><br/>add_band · change_no_data_value"]
```

::: pyramids.dataset.engines.Bands
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
