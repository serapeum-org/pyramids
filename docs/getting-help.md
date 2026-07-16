# Getting help

Stuck, hit a bug, or want a feature? Here is where to go.

## Before asking

- **Search the docs.** The search box (top of every page) covers tutorials, the ["How do I…?" index](examples/index.md),
  and the [API Reference](reference/dataset/index.md).
- **Check the common fixes.** [Troubleshooting](troubleshooting.md) and the
  [FAQ & install gotchas](how-to/faq.md) cover the frequent GDAL/format/install issues.
- **Confirm the API.** The per-class [reference pages](reference/dataset/index.md) list every method with its
  signature and docstring.

## Ask a question

- **GitHub Discussions** — usage questions, "how do I…", design/roadmap talk:
  <https://github.com/serapeum-org/pyramids/discussions>
- **Stack Overflow** / **GIS Stack Exchange** — tag your question `python` + `gdal` / `geospatial` so the wider
  community can find and answer it.

## Report a bug

Open an issue: <https://github.com/serapeum-org/pyramids/issues>. A good report is a *fast-fixed* report — please
include:

1. **A minimal, runnable reproduction** — the fewest lines that trigger it, ideally on data you can share (or one
   of the `examples/data/` fixtures).
2. **What you expected vs. what happened** — including the full traceback.
3. **Versions** — pyramids, GDAL, Python, and OS:

    ```python
    import pyramids
    from osgeo import gdal
    import sys, platform
    print("pyramids", pyramids.__version__, "| GDAL", gdal.__version__,
          "| Python", sys.version.split()[0], "|", platform.platform())
    ```

## Request a feature

Open an issue describing the use case and the data it applies to. pyramids stays a **generic GDAL/OGR toolkit**
— see [SCOPE](SCOPE.md) for what fits (generic raster/vector/datacube primitives) and what doesn't
(domain/sensor-specific logic).

!!! tip "Plotting requests go to cleopatra"
    All plotting delegates to [cleopatra](https://github.com/serapeum-org/cleopatra). Missing a chart type or
    basemap? File it on the cleopatra repo; pyramids wires up new cleopatra features through its `plot` methods.
