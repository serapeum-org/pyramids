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
| `test_bench_raster_io.py` | `from_array`; `read_array` (full / bbox / `out_shape`); `read_file` from disk; GeoTIFF write |
| `test_bench_spatial.py` | `crop` (bbox), `to_crs` (WGS84 → Web Mercator), `resample` (coarsen, bilinear) |
| `test_bench_vectorize.py` | `to_polygons` on a class-label raster |

This is the first cut. The issue tracks the full target set (DatasetCollection reductions / `to_cog_stack`,
NetCDF multidim + eager-vs-lazy, COG decimated reads, FeatureCollection rasterize) and a memory-benchmark layer.

## Methodology notes (so the numbers mean something)

- **GDAL cache is pinned** (`GDAL_CACHEMAX` = 256 MB) for the session in `conftest.py`, so results do not
  drift with the ambient block-cache size.
- **Data is synthetic**, built from a fixed seed at a representative size (2048×2048 float32 ≈ 16 MB), not the
  tiny committed test fixtures (too small to measure) and not a committed multi-GB raster (repo bloat).
- Read benchmarks reuse a **session-scoped** dataset, i.e. a **warm** GDAL block cache — a consistent regime.
  If you need cold-cache numbers, read a fresh file per round instead.
- Cloud/COG reads are **network-bound**: measure bytes transferred / range-request count (GDAL `/vsicurl/`
  stats), not wall-clock, and keep them behind the `live` marker — not added here.

## CI

Shared CI runners are too noisy to gate a build on absolute milliseconds, so these do **not** run in the
normal matrix. The intended route (CodSpeed for deterministic, instruction-counted PR comparisons, or a
nightly job that stores `--benchmark-json`) is still an open decision — see the tracking issue.
