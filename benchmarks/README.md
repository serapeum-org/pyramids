# Performance benchmarks

Micro-benchmarks for the hot raster/vector paths, built on
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/). They answer *"did this change make
operation X slower?"* — regression detection, not profiling. For *"why is X slow"* reach for a
profiler (`pyinstrument`, `scalene`) instead.

This directory is deliberately **outside `testpaths = ["tests"]`**, so it is never collected by the
`main`, `plot`, `interop-tests`, `test-all` or `test-fast` tasks — your normal test runtime is
unchanged. Benchmarks only run when you ask for them.

## Running

```bash
# the whole suite, with a compact stats table
pixi run -e dev benchmark

# equivalently / with options
pixi run -e dev pytest benchmarks --benchmark-only

# one module or one group
pixi run -e dev pytest benchmarks/test_bench_spatial.py
pixi run -e dev pytest benchmarks --benchmark-only -k reproject
```

## Catching a regression

```bash
# save a baseline on the branch you trust
pixi run -e dev pytest benchmarks --benchmark-autosave

# on a change, compare and fail if mean regresses >10%
pixi run -e dev pytest benchmarks --benchmark-compare --benchmark-compare-fail=mean:10%
```

Saved runs live under `.benchmarks/` (git-ignored).

## What is covered

| Module | Operations |
|---|---|
| `test_bench_raster_io.py` | `from_array`; `read_array` (full/bbox/`out_shape`); `read_file`; write; COG write; COG decimated reads (`read_part`/`preview`/`point`/`read_tile`) vs full read |
| `test_bench_spatial.py` | `crop`, `to_crs` (→ Web Mercator), `resample`, `align`, `warped_view` (lazy windowed read) |
| `test_bench_vectorize.py` | `to_polygons`, `contour`, `sieve`, `proximity` |
| `test_bench_collection.py` | `DatasetCollection`: `from_dataset`, `mean` (temporal), `apply`, `to_cog_stack`, `merge` (mosaic) |
| `test_bench_netcdf.py` | NetCDF multidim `read_array` **eager vs lazy (dask)**, `isel`, `sel`, `subset` |
| `test_bench_feature.py` | `FeatureCollection.fishnet`, rasterize (`Dataset.from_features`), `with_centroid`, `geodesic_area` |
| `test_bench_memory.py` | asserts the lazy/dask read stays **memory-bounded** (chunk-structural + `tracemalloc`) |

Every operation named in the tracking issue's hot-path table is covered. The COG and NetCDF read groups keep
the full-resolution / eager read alongside the decimated / lazy one in the same comparison group, so a run shows
the partial-read win directly.

## Methodology notes (so the numbers mean something)

- **GDAL cache is pinned** (`GDAL_CACHEMAX` = 256 MB) for the session in `conftest.py`, so results do not
  drift with the ambient block-cache size.
- **Data is synthetic**, built from a fixed seed at a representative size (2048×2048 float32 ≈ 16 MB), not the
  tiny committed test fixtures (too small to measure) and not a committed multi-GB raster (repo bloat). The COG
  fixture writes a tiled raster with overviews so decimated reads can be served from a coarse level; the NetCDF
  fixture is a synthetic `(time, lat, lon)` cube written through `NetCDF.from_xarray` (pyramids' own GDAL
  multidim writer — no `netCDF4` / `h5netcdf` backend needed).
- Read benchmarks reuse a **session-scoped** dataset, i.e. a **warm** GDAL block cache — a consistent regime.
  If you need cold-cache numbers, read a fresh file per round instead.
- Cloud/COG reads are **network-bound**: measure bytes transferred / range-request count (GDAL `/vsicurl/`
  stats), not wall-clock, and keep them behind the `live` marker — not added here.

## CI

Shared CI runners are too noisy to gate a build on absolute milliseconds, so these do **not** run in the normal
test matrix. Instead, `.github/workflows/benchmarks.yml` runs the suite **nightly** (and on `workflow_dispatch`)
and uploads the raw `--benchmark-json` plus the `.benchmarks/` autosave as a 90-day artifact. That gives a
history to track and a baseline to compare a branch against locally (see "Catching a regression" above) without
failing a PR on timing noise. (A deterministic, instruction-counted route — CodSpeed — was considered; it was
deferred because it needs an extra dev dependency and the CodSpeed GitHub app enabled on the repo.)
