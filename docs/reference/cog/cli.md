# Command line — `pyramids cog`

pyramids ships a small command-line interface (the `pyramids` console script, registered on
`pip install pyramids-gis`) built on the standard-library `argparse` — no extra dependency.
The `cog` command group mirrors the common write / validate / inspect workflow.

```bash
# Write a COG (validates the output unless --no-validate)
pyramids cog create input.tif scene_cog.tif --profile zstd
pyramids cog create input.tif scene_cog.tif --compress DEFLATE --blocksize 256

# Validate (exit code 1 for a non-COG; --strict promotes warnings to errors)
pyramids cog validate scene_cog.tif --strict

# Print structured metadata + the overview pyramid
pyramids cog info scene_cog.tif
```

The functions are also callable in-process for scripting/testing:

```python
from pyramids.cli import main
exit_code = main(["cog", "info", "scene_cog.tif"])
```

## API

::: pyramids.cli
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["main"]
