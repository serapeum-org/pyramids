# Command line (`pyramids`)

Installing pyramids adds a `pyramids` command — a thin CLI over the same API, for quick jobs and scripts without
writing Python. Every command has `-h/--help`:

```bash
pyramids --help                 # list all commands
pyramids <command> --help       # options for one command
```

Commands that write a file refuse to clobber an existing output unless you pass **`--overwrite`**. Most accept a
CRS as an EPSG code, WKT, or PROJ string, and a `--resampling` method (`nearest`, `bilinear`, `cubic`,
`average`, `mode`, …).

## Inspect & edit metadata

```bash
pyramids info dem.tif                 # human-readable metadata (add --json for machine output)
pyramids info dem.tif --json
pyramids bounds dem.tif               # bounding box; --crs 4326 reprojects the corners; --json for JSON
pyramids edit-info dem.tif --crs 4326 --nodata -9999 --tag units=m   # edit CRS / nodata / tags in place
```

- **`info`** — print raster metadata (`--json` for a machine-readable dump).
- **`bounds`** — print the bounding box; `--crs` reprojects the corners; `--json` emits JSON.
- **`edit-info`** — set `--crs`, `--nodata`, and/or repeatable `--tag KEY=VALUE` on a raster **in place**.

## Convert, warp, clip, merge, overviews

```bash
pyramids convert in.asc out.tif                       # re-encode; --driver GTiff to force a driver
pyramids warp in.tif out.tif --crs 4326 --resampling bilinear
pyramids clip in.tif out.tif --bbox 440000 475000 480000 515000    # or: --vector aoi.geojson
pyramids merge tileA.tif tileB.tif tileC.tif mosaic.tif            # two or more inputs, last arg = output
pyramids overview dem.tif --levels 2 4 8 --resampling average      # build image pyramids IN PLACE
```

- **`convert`** — re-save a raster in another format (driver inferred from the extension, or `--driver`).
- **`warp`** — reproject to `--crs` with an optional `--resampling`.
- **`clip`** — crop by `--bbox MINX MINY MAXX MAXY` (in the raster CRS) **or** by a polygon `--vector`.
- **`merge`** — mosaic two-or-more rasters; the **last** positional argument is the output.
- **`overview`** — build power-of-two overviews (`--levels 2 4 8 …`) into the file itself.

## Cloud-Optimized GeoTIFF

```bash
pyramids cog create in.tif out.cog.tif --profile zstd --blocksize 512   # write + auto-validate
pyramids cog validate out.cog.tif --strict                              # --strict: warnings are errors
pyramids cog info out.cog.tif                                           # structured COG layout / overviews
```

`cog create` profiles: `deflate`, `zstd`, `lzw`, `webp`, `jpeg`, `lerc`, `lerc_deflate`, `lerc_zstd`,
`packbits`, `raw`. Pass `--compress`/`--blocksize` to override, `--no-validate` to skip the post-write check.
See the [COG CLI reference](cog/cli.md) for the full option set.

## Band math

```bash
pyramids calc '(A - B) / (A + B)' nir.tif red.tif ndvi.tif --dtype float32
```

- **`calc`** — evaluate an expression over inputs `A, B, …` (the operands, in order) into a new raster; the
  **last** operand is the output path. `--dtype` sets the output NumPy dtype.

## Georeferencing

```bash
# Fit a transform through ground-control points (pixel/line -> map x/y)
pyramids georeference raw.tif out.tif \
    --gcp 0 0 100000 500000 --gcp 512 0 105000 500000 --gcp 0 512 100000 495000 \
    --gcp-crs 32636 --transform polynomial --order 1 --to-crs 4326

# Orthorectify from the raster's RPC sensor model (needs a DEM or a constant height)
pyramids orthorectify scene.tif ortho.tif --dem dem.tif --to-crs 4326
```

- **`georeference`** — warp from repeatable `--gcp PIXEL LINE X Y` points (with `--gcp-crs`); `--transform`
  `polynomial` (order 1–3 via `--order`) or `tps`; optional `--to-crs`.
- **`orthorectify`** — apply the raster's RPC model using `--dem` (or a constant `--rpc-height`).

## Raster ↔ vector

```bash
pyramids rasterize parcels.geojson parcels.tif --cell-size 10 --column value   # or --like template.tif
pyramids shapes classes.tif classes.geojson --geometry polygon                 # vectorize (one feature/cell)
pyramids sample dem.tif --points "440000,510000;450000,505000" --json          # read values at points
```

- **`rasterize`** — burn a vector into a new raster; set `--cell-size` **or** adopt a template grid with
  `--like`; `--column` selects the attribute to burn (default: all non-geometry columns).
- **`shapes`** — vectorize a raster to a vector file, one feature per cell (`--geometry polygon|point`,
  `--driver`); guarded for huge rasters — pass `--allow-large` to override the ~4M-cell safety limit.
- **`sample`** — read band values at `--points` (`'x,y'` pairs separated by `;`); `--json` for JSON output.

!!! tip "The CLI mirrors the API"
    Each command maps to a `Dataset` / `FeatureCollection` method — e.g. `warp` → `to_crs`, `clip` → `crop`,
    `rasterize` → `Dataset.from_features`, `shapes` → `to_feature_collection`. Reach for Python when you need to
    compose steps or stay in-memory; reach for the CLI for one-off file jobs.
