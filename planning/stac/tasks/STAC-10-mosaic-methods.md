# STAC-10 — Richer mosaic pixel-selection methods in `merge_rasters`

- **Gap:** A9 · **Priority:** P3 · **Effort:** M · **Depends on:** nothing ·
  **Feeds:** STAC-06/16 (grouped mosaics) · **Status:** ready

## Objective

Add `"count"` and `"mean"` (and, if it fits the memory model, `"median"`) to
`merge_rasters(method=...)`, so STAC mosaics and `groupby`/solar-day fusion can
composite beyond first/last/min/max/sum.

## Why

`merge_rasters` already supports first/last (VRT path) and min/max/sum (strip
reduce). rio-tiler's mosaic and odc's fusion additionally offer mean/count/
median. `"mean"` and `"count"` fit the existing strip-reduce accumulator cheaply.

## Files

- `src/pyramids/dataset/merge.py`.
- `tests/dataset/spatial/test_merge.py`.

## Exact current code

`src/pyramids/dataset/merge.py`:

```python
_VRT_METHODS = ("first", "last")
_REDUCE_METHODS = ("min", "max", "sum")
_MERGE_METHODS = _VRT_METHODS + _REDUCE_METHODS
# strip height so peak memory is O(strip) not O(grid)
_REDUCE_IDENTITY = {"min": np.inf, "max": -np.inf, "sum": 0.0}
```

The reduction runs one full-width strip at a time (`if method in
_REDUCE_METHODS:` at ~L1599), accumulating per strip and marking uncovered
pixels with NaN. Output dtype is promoted to Float64 for the reduce path
(~L1426-1431).

## Implementation steps

1. Extend `_REDUCE_METHODS` to include `"count"` and `"mean"`; extend
   `_MERGE_METHODS` accordingly.
2. `_REDUCE_IDENTITY`: `"count": 0.0`, `"mean": 0.0` (mean tracked as sum with a
   parallel count accumulator).
3. In the strip reduction, maintain a **count accumulator** (number of valid
   contributions per cell) alongside the value accumulator:
   - `"count"` → output the count accumulator (cast appropriately; uncovered = 0
     or nodata — decide and document; 0 is natural for a count).
   - `"mean"` → accumulate `sum` and `count`; at strip finalisation output
     `sum / count` where `count > 0`, else the nodata marker (NaN). Divide by the
     **valid** count, never the total overlap.
4. `"median"`: only if it fits O(strip) memory. It requires all overlapping
   values per cell, which the current one-pass accumulator does not hold. If it
   would need buffering every source per strip (O(n_sources · strip)), **defer
   it with a clear `NotImplementedError`/docstring note** rather than silently
   approximating.
5. Surface the new methods wherever `method=` is validated and documented, and
   thread them through `from_stac`/solar-day/`groupby` `method=`.

## Verified facts

- `_REDUCE_METHODS`, `_REDUCE_IDENTITY`, the strip loop, and the Float64 promotion
  are all in `merge.py` (line refs above).
- Default `method="last"` (do not change it).

## Pitfalls / regression risks

1. **`mean` divides by the valid count, not the overlap count.** A cell covered
   by 2 of 3 sources averages 2 values.
2. **Keep memory O(strip).** count/mean add one accumulator array — fine. median
   does not fit — defer.
3. **Uncovered cells**: for `count`, an uncovered cell is `0` (or nodata —
   document which); for `mean`, uncovered stays the NaN marker like the other
   reductions.
4. **Don't change existing methods** (first/last/min/max/sum) — regression-test
   them.
5. Float64 promotion already applies to the reduce path; count could be int but
   keep it consistent with the existing Float64 output unless a separate dtype is
   clearly better.

## Tests (`tests/dataset/spatial/test_merge.py`)

Two overlapping rasters with known values:
- `method="count"` → overlap cells == 2, single-cover == 1.
- `method="mean"` → overlap cell == mean of the two source values.
- Regression: `first/last/min/max/sum` outputs unchanged.
- `method="median"` → raises `NotImplementedError` (if deferred) with a clear
  message.

## Definition of Done

- [ ] `count` + `mean` implemented in the strip reduce (O(strip) memory).
- [ ] `mean` divides by valid count; `count` uncovered rule documented.
- [ ] `median` implemented or cleanly deferred with a clear error.
- [ ] New methods surfaced to `from_stac`/groupby `method=`.
- [ ] Existing methods unchanged; tests pass.
