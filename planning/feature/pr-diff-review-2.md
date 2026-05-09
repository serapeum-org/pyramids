# PR review — `refactor/feature` vs `main` (delta since pr-diff-review-1)

**Scope:** 30 commits, 16 files changed, +2417 / -242 lines.
**Base:** `main` @ `f9fead0` (via merge-base with `958fae0` — the tip reviewed in
`planning/feature/pr-diff-review-1.md`).
**Head:** `refactor/feature` @ `2563fa3`.
**Reviewing only the delta since the last review.** The earlier 26 ARC commits already have a
review; this review covers the C2-C44 / D-H1-D-N4 fix sequence that landed in response to
`planning/feature/pr-review-merged.md`.

**Cross-reference — `pr-copilot-2.md` merge.** A second copilot review landed in
`planning/feature/pr-copilot-2.md` covering the `feat/dask` branch. 30+ of its findings
(C1 Zarr race, C2 zonal polygon validity, H2 FileManager cache, H3 xarray backend,
H5 STAC asset URLs, H6 open_arrow, H7 spatial_shuffle, M1 eviction callback, M2
to_kerchunk, M3 flox, M5 parquet storage_options, M7 read_file layer kwarg, L1/L2/L3
to_file compute, L4 focal_std, L6 STAC bbox, L7 dask-partitioning, N1/N3/N4/N5/N6/N7/N8/N9
docs for feat/dask-only APIs) point at files (`ops/_zarr.py`, `ops/_zonal.py`,
`ops/_focal.py`, `base/_file_manager.py`, `netcdf/_xarray_backend.py`,
`feature/_lazy_collection.py`, `dataset/_stac.py`, …) that **do not exist on
refactor/feature** and belong in a future `feat/dask` review. Findings from
pr-copilot-2 that genuinely apply to refactor/feature are merged into the tracker
below as **M4 / L6 / N6** (renumbered to avoid collision with my own pr-diff-review-2
IDs).

# Summary

- Implements 27 of the 40 issues from `pr-review-merged.md` (11 Nit / Low are marked N/A
  with rationale). Each fix is bounded, tested, and committed independently (one commit
  per issue, grouped into 7 batches). The pattern is clean: target the function, add a
  typed error path where appropriate, add ≥1 regression test per issue.
- Two subtle regressions introduced by the D-L6 refactor and by broad `except Exception`
  in C23 — both flagged below as Medium.
- Public API break (D-H2): `FeatureCollection.create_point` staticmethod is deleted and
  `create_polygon(coords, wkt=True)` raises `TypeError` now. The commit message calls this
  out explicitly, but no user-facing CHANGELOG entry surfaces in the diff.
- Tests are strong (+2175 test LOC across 8 files, including chained e2e scenarios) but
  have two specific gaps flagged below (M1, M2).

# Findings

## Critical

None.

## High

None.

## Medium

### M1 — `crs.py:174` catches `except Exception`, too broad

`src/pyramids/feature/crs.py:171-177` wraps the `Transformer.from_crs` call in a bare
`except Exception` and re-raises as `CRSError`. The actual pyproj failure type is
`pyproj.exceptions.CRSError` (a subclass of `Exception`), but the bare `except Exception`
also swallows:

- `TypeError` (e.g. passing a non-CRS-convertible object)
- `ValueError` (e.g. out-of-range EPSG integer)
- `AttributeError` from any downstream attribute lookup

…and surfaces them all as a single `CRSError("reproject_coordinates failed to parse CRS
…: {exc}")`. The caller loses the typed information of the real failure.

**Impact:** bug-hunting loses specificity. A pyproj upgrade that raises a new exception
type will also get silently swallowed.

**Suggested fix:** narrow the except:

```python
import pyproj.exceptions  # top-level or inline

try:
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
except pyproj.exceptions.CRSError as exc:
    raise _CRSError(
        f"reproject_coordinates failed to parse CRS "
        f"(from_crs={from_crs!r}, to_crs={to_crs!r}): {exc}"
    ) from exc
```

If you also want to catch `TypeError`/`ValueError` (e.g. for non-CRS inputs), name them
explicitly: `except (pyproj.exceptions.CRSError, TypeError, ValueError)`.

### M2 — `geometry.py` D-L6 refactor introduced eager attribute access

`src/pyramids/feature/geometry.py:58-62` (`get_point_coords`):

```python
accessor = {"x": geometry.x, "y": geometry.y}
if coord_type not in accessor:
    raise ValueError(...)
return float(accessor[coord_type])
```

Both `geometry.x` and `geometry.y` are evaluated on every call, eagerly. For an empty
shapely `Point`, either attribute raises `shapely.errors.GEOSException("getX called on
empty Point")` BEFORE the `coord_type not in accessor` guard. The pre-D-L6 code was
lazy — it only touched the attribute the caller asked for.

With C22 guarding empty geoms at `get_coords` (the parent), the public pyramids entry
points are shielded. But **direct callers of `get_point_coords`** now see a GEOS error
rather than the old cleaner `ValueError("coord_type can only have a value of 'x' or
'y'")`. More subtle: the D-L6 comment ("single-return via attribute dispatch") claims the
change is cosmetic — it's not.

Same concern applies to `get_xy_coords:39-41` where
`geometry.coords.xy` is eagerly materialised before the axis-index lookup.

**Impact:** internal callers may hit a worse error surface on empty geometries. Not a
user-facing regression today but a trap for future refactors.

**Suggested fix:** roll back D-L6 on these two functions (the multi-return was fine).
Dispatch-on-value helpers are the rare case where the CLAUDE.md rule hurts more than it
helps — note it in the rule. Or use a function dispatch:

```python
_ACCESSORS = {"x": attrgetter("x"), "y": attrgetter("y")}
if coord_type not in _ACCESSORS:
    raise ValueError(...)
return float(_ACCESSORS[coord_type](geometry))
```

### M4 — `Dataset.from_features` does not type-check `column_name` (from pr-copilot-2 #14)

`src/pyramids/dataset/dataset.py:784+` — my D-M2 added content validation (empty list,
unknown column) but did not type-check. A caller passing `column_name=123` (an int)
skips the `isinstance(column_name, list)` branch, lands in the `else` branch, hits
`column_name not in features.columns` (which evaluates `123 not in Index([...])` and
returns True), then raises a misleading "not in the FeatureCollection" ValueError rather
than a TypeError naming the real issue.

**Impact:** Medium. The user sees "column_name 123 is not in the FeatureCollection" and
may try to find a column named 123 rather than realise they passed the wrong type.

**Suggested fix:**

```python
if column_name is not None and not isinstance(column_name, (str, list)):
    raise TypeError(
        f"column_name must be str, list[str], or None; "
        f"got {type(column_name).__name__}."
    )
```

Place this check at the top of `from_features`, next to the existing `cell_size <= 0`
guard.

### M3 — `_ogr.py:18` keeps an unused `uuid` import behind `# noqa: F401`

`src/pyramids/feature/_ogr.py:18` — `import uuid  # noqa: F401 — preserved for downstream
compatibility`. But **no function in the module still uses `uuid.uuid4()`** after C35
swapped `_new_vsimem_path` to `time.time_ns() + random.randint(...)`. The comment claims
"downstream compatibility" but nothing downstream imports `uuid` from `pyramids.feature._ogr`
— it's a private module.

**Impact:** dead import silenced by a misleading `noqa`. Any pylint/ruff run will flag it.

**Suggested fix:** delete `import uuid` and the `noqa` comment.

## Low

### L1 — redundant `bytes(...)` copy in `_ogr.datasource_to_gdf`

`src/pyramids/feature/_ogr.py:320` — `data = bytes(gdal.VSIFReadL(1, size, vsi_file))`.
On GDAL ≥3.0, `gdal.VSIFReadL` already returns a Python `bytes` object; the `bytes(...)`
call forces a defensive second copy of the entire buffer. For a small feature collection
it doesn't matter; for a 100-MB polygonize output it's a wasted memory round-trip.

**Suggested fix:** `data = gdal.VSIFReadL(1, size, vsi_file)` — no wrapper.

### L2 — `epsg` equality fallback (C11) cost not documented

`src/pyramids/feature/collection.py:693-697` — the `cached_crs == crs` fallback calls
`pyproj.CRS.__eq__`, which internally compares WKT2 serialisations. Cheap once, but
**called on every `fc.epsg` access** whenever `self.crs` returns a fresh object (which
some pandas paths do via `_geometry_column_name` lookups). The cache-hit fast-path
(`cached_crs is crs`) skips it; the fallback doesn't.

**Impact:** rare hot-path regression if pandas internals stop returning the same CRS
object identity. No current bug.

**Suggested fix:** add a one-line note in the docstring acknowledging the equality
fallback is cheaper than `to_epsg()` but not free, and consider caching by
`cached_crs.to_wkt()` instead of identity if this ever dominates a profile.

### L3 — `list_layers` local-path check duplicates the URL-scheme list

`src/pyramids/feature/collection.py:795-802` hardcodes
`("/vsi", "http://", "https://", "s3://", "gs://", "az://", "abfs://", "file://")` to
distinguish local from remote. The authoritative list lives in `pyramids.base.remote`.

**Impact:** if a new scheme lands in `base.remote` (the feat/dask branch already added
dask-specific paths), `list_layers` silently treats it as local and raises
`FileNotFoundError` even though the driver could open it.

**Suggested fix:** import `pyramids.base.remote.is_remote` (or equivalent) and use it:

```python
from pyramids.base.remote import is_remote

if not is_remote(path_str) and not Path(path_str).exists():
    raise FileNotFoundError(...)
```

### L4 — `with_centroid` does two passes when one would do

`src/pyramids/feature/collection.py:1363-1387` — builds `coords_list` by iterating
`zip(avg_x, avg_y)`, calls `_geom.create_points(coords_list)`, then iterates the returned
`points` list in a second loop to substitute empty Points where the mask is `True`. A
single pass could build `cleaned` directly.

**Impact:** style nit; 2× the loop cost on the happy path.

**Suggested fix:** merge the loops:

```python
cleaned = []
for ax, ay, bad in zip(avg_x.tolist(), avg_y.tolist(), bad_mask):
    cleaned.append(Point() if bad else Point(ax, ay))
fc["center_point"] = cleaned
```

### L6 — `with_centroid` NaN warning uses plain `UserWarning` (from pr-copilot-2 #26)

`src/pyramids/feature/collection.py:with_centroid` (C18 fix) emits
``warnings.warn(..., UserWarning, stacklevel=2)``. The warning DOES have a stacklevel
(pr-copilot-2 got that detail wrong), but the category is plain `UserWarning`, which
means callers cannot suppress only the NaN-centroid warning without also silencing
every other `UserWarning` pyramids / geopandas / shapely might emit.

**Impact:** low — users who want to silence just the NaN warning must filter by
message substring, which is brittle.

**Suggested fix:** introduce a pyramids-specific category for geometry warnings and use
it here:

```python
# In pyramids.base._errors (or a new module):
class GeometryWarning(UserWarning):
    """Pyramids-emitted warning about geometry validity / degeneracy."""
```

And in `with_centroid`:

```python
warnings.warn(
    f"with_centroid: {len(bad_idx)} row(s) yielded NaN centroids ...",
    GeometryWarning,
    stacklevel=2,
)
```

Users can then do `warnings.filterwarnings("ignore", category=GeometryWarning)`.

### L5 — `concat` CRS-mismatch error one-sided

`src/pyramids/feature/collection.py:1304-1311` — the error message says "Call
`other.to_crs(self.crs)` before concatenating". The inverse direction (reproject `self`
to `other.crs`) may be what the caller wants; the message only hints at one option.

**Suggested fix:** "Reproject one side — ``other.to_crs(self.crs)`` or
``self.to_crs(other.crs)`` — before concatenating."

## Nit

### N1 — `# CN:` / `# D-NN:` comment markers will rot

Throughout the source, inline comments like `# C2:`, `# C14:`, `# D-H1:`, `# D-L6:` mark
which tracker issue the code addresses. In 6-12 months, post-merge, nobody remembers
what these codes mean. `planning/feature/pr-review-merged.md` is durable, but grep-ability
from code → plan is one-way.

**Suggested fix:** either link each marker to the tracker doc in a single README line,
or migrate the rationale into the docstring "Notes:" section so the code-level comment
can be dropped. Doing nothing is fine too; flagging for awareness.

### N2 — `from_records` has duplicated D-N4 comment block

`src/pyramids/feature/collection.py:338-345` and `:352-359` — identical 5-line comment
blocks in the two empty-input branches. DRY violation in comments.

**Suggested fix:** hoist into a single helper:

```python
def _empty_from_records_fc(geometry: str, crs: Any) -> FeatureCollection:
    """Empty-input branch shared by both orient modes."""
    return cls(gpd.GeoDataFrame({geometry: []}, geometry=geometry, crs=crs))
```

### N3 — D-H2 breakage not reflected in a CHANGELOG

`FeatureCollection.create_point` staticmethod deleted and
`FeatureCollection.create_polygon(coords, wkt=True)` raises `TypeError`. A user who wrote
these against refactor/feature pre-D-H2 sees a hard break. No `CHANGELOG.md` /
`docs/change-log.md` update surfaces in the diff.

**Impact:** pre-1.0 project so breakage is allowed, but a note under "Breaking changes"
in the changelog would help downstream migration.

**Suggested fix:** add an entry to `docs/change-log.md` under the next release:

```markdown
### Breaking
- `FeatureCollection.create_point(coords, epsg=...)` removed — use
  `FeatureCollection.create_points(coords)` for the list form or
  `FeatureCollection.point_collection(coords, crs=...)` for the FC form.
- `FeatureCollection.create_polygon(coords, wkt=True)` removed — use
  `FeatureCollection.polygon_wkt(coords)`.
```

### N4 — `_DEFAULT_ITER_BATCH_SIZE` comment is vague

`src/pyramids/feature/collection.py:46-49` — "1000 rows balances pyogrio overhead against
memory headroom on a typical development machine." That's a hand-wavy justification in
production code. Pick a concrete reason (benchmark on a specific file size?) or drop the
second half of the comment.

### N5 — `# noqa: F401` on uuid is misleading

Covered by **M3**; listed here again because the `noqa` suppresses what would otherwise be
a valid linter signal.

### N6 — D-H2 commit body doesn't name which legacy shims (from pr-copilot-2 #29)

The D-H2 commit message ("delete ARC-15 legacy shims outright") does reference ARC-15 but
doesn't name the functions or kwargs removed in the one-line summary. A reader of
`git log --oneline` has to click through to the body to see
``create_polygon_legacy`` / ``create_point_legacy`` / ``create_polygon(..., wkt=True)``
/ ``create_point(..., epsg=...)``. The body DOES cover all four; the nit is just about
the one-liner's information density.

**Impact:** trivial. Reviewer ergonomics only.

**Suggested fix:** for future breaking-change commits, name the shortest set of public
symbols removed in the summary line, e.g.:
``D-H2: delete create_polygon(wkt=), create_point(epsg=), legacy shims``.
No action on this commit (already merged).

# Tests

## Added (+2175 lines across 8 files)

- `tests/feature/test_feature_unit.py` (+306) — C3 metadata dedup subclass, C11 epsg
  cache equality fallback + None↔non-None transitions, C22 empty-geom raise
  (Point/LineString/Polygon), C23 pyproj → pyramids CRSError wrap, D-M5 branded
  ImportError for missing pyarrow, D-H1 explode_gdf non-mutation + expanded-row count,
  C33 read_parquet bbox kwarg routing.
- `tests/feature/test_from_features_and_stream.py` (+247) — C9 empty iter, C14
  include_index (per-feature + chunked + Python-bbox filter + chunksize=1 boundary),
  D-M3 engine="pyogrio" pin, C26 orient="list" + missing-geom + mismatched-lengths.
- `tests/feature/test_ogr_bridge.py` (+195) — C4 exception safety (None open, no unlink
  on serialization failure, VectorDriverError message shape, user-code raise cleanup),
  D-M4 no-temp-file assertion + empty DataSource round-trip, C35 stem format.
- `tests/feature/test_schema_and_layers.py` (+136) — C15 LRU cache (repeats, clear,
  separate entries, fresh-list-each-call), C29 missing-file FileNotFoundError, C30 crs
  in schema dict.
- `tests/feature/test_feature.py` (+244) — C18 with_centroid NaN warn + empty-Point
  substitution + multi-row warning, C21 ≥3-vertex guard + boundary, D-H2 shim deletion,
  C32 concat CRS-mismatch + None-side allowance.
- `tests/feature/test_pickle_safety.py` (+61) — C3 metadata dedup, subclass-still-deduped,
  copy-preserves-dedup.
- `tests/feature/test_to_file_options.py` (+102) — C8 pyogrio unknown-option raises,
  known-option accepted, case-insensitivity, mixed known+unknown, C28 engine pin.
- `tests/test_e2e_workflows.py` (+304) + `tests/feature/test_e2e_workflows.py` (+167) —
  four chained e2e pipelines (C2-C8 chain, C9-C18 chain, D-H1/C21/C22/C23 geometry
  hardening chain, D-M2/D-M3/D-M4/C26 records→rasterise chain).

## Gaps

1. **M1 not covered** — no test that a non-`pyproj.CRSError` exception (e.g. `TypeError`
   on a non-string from_crs) leaks through the wrapper with its typed cause. Because the
   catch is `Exception`, a test that passes something weird still sees `CRSError`
   regardless.
2. **M2 not covered** — `get_point_coords(Point(), "x")` direct-call on an empty Point
   would currently raise GEOSException. Once M2 is fixed, a test pinning "empty Point →
   `ValueError` (not GEOSException)" guards against a future revert.
3. **D-H2 docs breakage** — no test verifies the migration message steers callers to
   `create_points` / `point_collection`. One could assert the `AttributeError` message on
   `FeatureCollection.create_point` names the replacement, or that a `CHANGELOG.md` entry
   exists (less useful as a unit test).
4. **D-L2 performance claim** — the O(N²) → O(N) claim for `explode_gdf` is not backed
   by a benchmark. Not a test gap per se, but if performance regresses, no canary.

## Not run

The reviewer did not execute the full suite during this review. The author's commit log
reports `332 passed, 14 skipped` on the final post-batch-7 sweep.

# Questions and Assumptions

## Questions

- **Q1.** `feat/dask` already contains the D-M1 inline-import cleanup. Many of the new
  commits (`C5`, `C23`, C32, D-H1's test harness, D-M4, `create_polygon` guard, etc.)
  added **new** inline imports inside function bodies. When `feat/dask` rebases onto or
  merges with post-`refactor/feature` `main`, these will conflict cleanly but require
  re-hoisting each new inline import to module top. Is someone tracking that?
- **Q2.** D-H2 (legacy shims removed) is a BREAKING CHANGE flagged in the commit body.
  Does it warrant a changelog update before tagging a release? (Pre-1.0 excuses a lot,
  but silent hard-breaks hurt downstreams.)
- **Q3.** C11 equality fallback — did you measure the CRS `__eq__` cost on a hot
  `fc.epsg` loop? At worst it's a WKT2 string comparison; not free, but probably fine.
  Worth a single micro-benchmark for peace of mind.

## Assumptions

- **A1.** `refactor/feature` and `feat/dask` will merge to `main` in that order.
- **A2.** The `CloudConfig` / `base.remote._to_vsi` surface is stable enough that L3's
  hardcoded scheme list in `list_layers` is tolerable short-term.
- **A3.** Testing on this branch is exclusively under pyogrio (geopandas' default on
  1.0+). The engine pins on iter_features, read_file, to_file, and read_parquet
  assume that contract. Fiona remains a supported engine only via explicit override.
- **A4.** `gdal.VSIFReadL` on the pinned `GDAL >= 3.10` returns a Python `bytes` object
  (needed for L1 to be valid). Older GDAL returned a `SwigObject` that had to be
  wrapped — the pyproject-toml pin makes L1's suggestion safe.

# Residual Risks

- **R1** — M1's broad `except Exception` is the only Medium that could surprise users
  in production. Narrow it before merge.
- **R2** — M2's eager attribute access makes the old test suite pass but is a foot-gun
  for future refactors that bypass the C22 parent guard.
- **R3** — Inline-import accumulation on `refactor/feature` (C5, C23, C32, D-H1 test
  paths, D-M4, C21, D-N4's from_records) will conflict with `feat/dask`'s D-M1 cleanup.
  Resolvable but touchable.
- **R4** — No CHANGELOG entry for D-H2 breakage (N3). Soft risk, pre-1.0 scope lets it
  slide.
- **R5** — L1's redundant `bytes(...)` copy doubles memory for every polygonize. Fix
  is a one-line delete.

No blocking issues found apart from the two Mediums (M1 narrow-except, M2 eager
attribute access). Fix those, update a CHANGELOG for D-H2, and the delta is ready to
ride with the rest of `refactor/feature` into `main`.

# Issue Tracker

Source tag: **[own]** = this review; **[copilot-2]** = merged from
`planning/feature/pr-copilot-2.md` (only the subset that applies to refactor/feature;
feat/dask-only findings are in that file's tracker, not re-listed here).

Source: **own** = this review; **cp2** = merged from `planning/feature/pr-copilot-2.md`
(subset applying to refactor/feature only; feat/dask-only findings stay in that file).

| ID | Sev | State   | Description                                        | File                     | Src |
|----|-----|---------|----------------------------------------------------|--------------------------|-----|
| M1 | Med | Solved  | `except Exception` in reproject_coordinates broad  | `feature/crs.py`         | own |
| M2 | Med | Solved  | D-L6 made get_point_coords eager on empty geom     | `feature/geometry.py`    | own |
| M3 | Med | Solved  | Unused `uuid` hidden behind misleading `noqa`      | `feature/_ogr.py`        | own |
| M4 | Med | Solved  | `from_features` doesn't type-check `column_name`   | `dataset/dataset.py`     | cp2 |
| L1 | Low | Solved  | Redundant `bytes(...)` copy in datasource_to_gdf   | `feature/_ogr.py`        | own |
| L2 | Low | Solved  | epsg equality-fallback cost not documented         | `feature/collection.py`  | own |
| L3 | Low | Solved  | list_layers local-path check duplicates schemes    | `feature/collection.py`  | own |
| L4 | Low | Solved  | with_centroid does two loops when one would do     | `feature/collection.py`  | own |
| L5 | Low | Solved  | concat CRS-mismatch message one-sided              | `feature/collection.py`  | own |
| L6 | Low | Solved  | with_centroid NaN warning uses plain UserWarning   | `feature/collection.py`  | cp2 |
| N1 | Nit | Solved  | Inline `# CN:` / `# D-NN:` markers will rot        | `feature/*`, `dataset/*` | own |
| N2 | Nit | Solved  | from_records duplicated D-N4 comment block         | `feature/collection.py`  | own |
| N3 | Nit | Solved  | D-H2 breakage lacks a CHANGELOG entry              | `docs/change-log.md`     | own |
| N4 | Nit | Solved  | `_DEFAULT_ITER_BATCH_SIZE` comment is hand-wavy    | `feature/collection.py`  | own |
| N5 | Nit | Solved  | `# noqa: F401` on uuid silences dead-import signal | `feature/_ogr.py`        | own |
| N6 | Nit | Closed  | D-H2 summary line doesn't name the removed shims   | git history              | cp2 |

> **Resolution notes.**
>
> - **N5** was auto-resolved by the fix for **M3**: removing the unused
>   ``import uuid`` eliminated the ``# noqa: F401`` that was silencing the
>   dead-import signal.
> - **N6** is **Closed (won't-fix)**: the D-H2 commit is already merged
>   history and the concern is purely about its one-line summary, which
>   cannot be edited without rewriting history. The commit body *does*
>   enumerate every removed shim, and the new guidance (name the removed
>   symbols on the summary line) has been folded into future breaking
>   changes.
