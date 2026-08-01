# Processing pipelines

`pyramids.processing` turns existing `Dataset` / `FeatureCollection` operations into named, self-describing
**tools** that can be chained into a serializable **pipeline** and run (batched) over one or many inputs — a
Whitebox/QGIS-Processing-style workflow layer built on the ops pyramids already owns.

The pieces:

- a **registry** of named tools, each with a parameter schema and a receiver type (`Dataset` vs
  `FeatureCollection`);
- a **`Pipeline`** — an ordered chain of `(tool, params)` steps, serializable to a portable YAML "model" file;
- a batch **`run`** with an error policy and optional process-pool parallelism;
- a **provenance** record per run (tool, params, timing) that can re-emit the exact pipeline that produced an
  output.

## v1 tool allowlist

v1 ships a curated, real-signature allowlist (see [ADR 0007](../adr/0007-processing-registry-approach.md) for why it
is hand-written rather than introspected). List it with `pyramids tools`:

| tool | receiver → returns |
|------|--------------------|
| `slope`, `aspect`, `hillshade` | `Dataset` → `Array`\* |
| `to_crs`, `resample` | `Dataset` → `Dataset` |
| `interpolate_to_raster` | `FeatureCollection` → `Dataset` |
| `to_h3` | `FeatureCollection` → `FeatureCollection` |

\* The terrain ops natively return a numpy array; inside a pipeline the runner materializes it back into a
single-band, georeferenced `Dataset` (carrying the source raster's geotransform/CRS), so the result is writable to
disk and can be chained into a further `Dataset` step.

The registry is extensible — register a `ToolSpec` to add a tool.

## Example (Python)

A cross-receiver pipeline: interpolate scattered points onto a raster, then compute its slope. `interpolate_to_raster`
is a `FeatureCollection` op that returns a `Dataset`, and `slope` runs on that `Dataset` — the runner dispatches each
step to the correct receiver automatically.

```python
from pyramids.feature import FeatureCollection
from pyramids.processing import Pipeline, run

gauges = FeatureCollection.read_file("samples/elevation_points.geojson")

pipe = Pipeline([
    ("interpolate_to_raster", {"column": "elevation", "cell_size": 1000.0}),
    ("slope", {}),
])
pipe.to_yaml("elevation_slope.yaml")            # the portable "model"

result = run(pipe, gauges)                       # batch over one or many inputs
slope_raster = result.outputs[0]                 # a georeferenced Dataset (slope array materialized)
print(result.provenance[0].total_seconds)        # per-run timing
```

## Example (CLI)

```bash
pyramids tools                                   # list registered tools
pyramids tool interpolate_to_raster              # show a tool's parameters
pyramids run elevation_slope.yaml \
    --inputs "samples/*.geojson" --out out/      # batch over a glob, write to out/
```

`run` writes each output into `--out`, named after its source; `--on-error skip` (default) collects failures and
continues, `--on-error raise` fails fast, and `--parallel` fans the batch across a process pool (file-path inputs
only — GDAL handles cannot cross process boundaries).

## API

::: pyramids.processing.Pipeline
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.processing.run
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.processing.RunResult
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.processing.Provenance
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.processing.ToolSpec
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.processing.ParamSpec
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.processing.resolve
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.processing.tool_names
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.processing.register
    options:
        show_root_heading: true
        heading_level: 3
