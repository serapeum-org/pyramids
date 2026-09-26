# STAC-16 — Custom `fuse_func` for grouped/mosaic overlap

- **Gap:** odc `fuse_func` · **Priority:** P4 · **Effort:** M · **Depends on:**
  STAC-06 (grouped path) and ideally STAC-10 (methods) · **Status:** ready after
  its deps

## Objective

Allow a caller-supplied fusion callback when mosaicking overlapping items in a
group (solar-day / `groupby`), instead of only the built-in
`merge_rasters(method=...)`.

## Why

`from_stac`'s docstring currently declares `fuse_func` out of scope. odc-stac
lets users customise how overlapping pixels combine. This is the advanced escape
hatch beyond the fixed methods.

## Files

- `src/pyramids/dataset/_stac.py` — the grouped path (`_from_stac_grouped` from
  STAC-06, and `_from_stac_solar_day`).
- possibly `src/pyramids/dataset/merge.py` if the callback is applied there.
- `tests/dataset/stac/test_stac.py`.

## Contract (match odc, adapted)

odc's `fuse_func` is `(dst: np.ndarray, src: np.ndarray) -> None` — an in-place
merge of the next source `src` into the accumulator `dst` (copy only the pixels
to keep). Adopt the same contract for pyramids:

```python
def my_fuser(dst: np.ndarray, src: np.ndarray) -> None:
    """Merge src into dst in place (e.g. copy only where dst is still nodata)."""
    mask = np.isnan(dst)
    dst[mask] = src[mask]
```

## Implementation steps

1. `from_stac(..., fuse_func: Callable[[np.ndarray, np.ndarray], None] | None =
   None)`.
2. When `fuse_func` is provided (grouped/solar-day modes only), fuse each group's
   source arrays with the callback into one array per group instead of calling
   `merge_rasters(method=...)`. This requires reading each source onto a common
   grid (align to the first source, as the multi-asset path already does) and
   applying `fuse_func` pairwise in order.
3. `fuse_func` and `method` are mutually exclusive per group — if both given,
   raise a clear error (or document precedence).
4. Keep it single-asset (like the other grouped modes) unless explicitly scoped.

## Pitfalls / regression risks

1. **In-place contract** — the callback mutates `dst`; document it and don't rely
   on a return value.
2. **Consistent nodata/fill** — the arrays handed to `fuse_func` must share a
   known nodata/fill convention (NaN), matching the reduce path, so the callback
   can detect "still empty" cells.
3. **Alignment** — sources in a group may differ in grid; align to the first
   before fusing (reuse existing align machinery). Don't hand `fuse_func` arrays
   of different shapes.
4. **Default `None`** — behaviour unchanged (built-in `method`).
5. This is advanced/opt-in — keep the built-in `method=` path as the common case.

## Tests

```python
def test_fuse_func_last_valid(three_local_items):
    import numpy as np
    def keep_first(dst, src):        # copy only where dst is still NaN
        m = np.isnan(dst); dst[m] = src[m]
    coll = DatasetCollection.from_stac(
        three_local_items, asset="data", groupby="orbit", fuse_func=keep_first)
    assert coll.time_length == 2      # 2 orbit groups fused with the callback
```

## Definition of Done

- [ ] `fuse_func` (in-place `(dst, src) -> None`) supported in grouped/solar-day
  modes; aligned sources; NaN fill convention.
- [ ] Mutually-exclusive-with-`method` handling documented.
- [ ] Default `None` unchanged; from_stac docstring updated (removes the
  "out of scope" note).
- [ ] Tests pass.
