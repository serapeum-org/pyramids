# STAC-00 — Prerequisites & shared infrastructure

- **Priority:** P0 (do first) · **Effort:** S · **Blocks:** STAC-05 (extra +
  marker + CI env), STAC-06 / STAC-08 (shared fixtures) · **Status:** ready

## Objective

Create the shared scaffolding the other tasks assume exists, so each of them is
truly self-contained: shared test fixtures, the new `[stac-parquet]` extra wired
through pixi + a pytest marker so its tests actually run in CI, and the docs/
CHANGELOG plumbing for new public APIs.

## Why

Several task specs reference infrastructure that does not exist yet
(a `three_local_items` fixture; a `[stac-parquet]` extra + marker + CI
environment). Without this task, the implementer has to invent them ad hoc — the
exact guessing we are trying to eliminate.

## 1. Shared test fixtures

There is **no** `tests/dataset/stac/conftest.py` today. Create one with the
fixtures the STAC tasks reuse. Match the existing style in
`tests/dataset/stac/test_to_stac_item.py` (`Dataset.from_array(..., geo_ref=
GeoReference(top_left_corner=..., cell_size=..., epsg=...))`, `pytest.mark.core`).

```python
# tests/dataset/stac/conftest.py
from __future__ import annotations
import numpy as np
import pytest
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset


@pytest.fixture
def packed_raster_path(tmp_path):
    """A local int16 GeoTIFF with a known scale so rescale tests are exact.

    stored: [[0,100],[200,300]], nodata 0, scale 0.01 -> physical [[_,1.0],[2.0,3.0]].
    """
    p = str(tmp_path / "packed.tif")
    ds = Dataset.from_array(
        np.array([[0, 100], [200, 300]], dtype="int16"),
        no_data_value=0,
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
    )
    ds.to_file(p)
    return p


@pytest.fixture
def three_local_items(tmp_path):
    """Three STAC item dicts over real local rasters, with distinct datetimes and
    an ``orbit`` property (two share orbit=1, one has orbit=2) for groupby tests."""
    items = []
    orbits = [1, 1, 2]
    for i, orbit in enumerate(orbits):
        p = str(tmp_path / f"scene{i}.tif")
        Dataset.from_array(
            np.ones((3, 3), "float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
        ).to_file(p)
        items.append({
            "type": "Feature",
            "id": f"scene{i}",
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]]},
            "bbox": [0.0, 0.0, 3.0, 3.0],
            "properties": {"datetime": "2023-06-0%dT00:00:00Z" % (i + 1),
                           "orbit": orbit, "eo:cloud_cover": 10 * i},
            "assets": {"data": {"href": p, "type": "image/tiff"}},
            "stac_extensions": [],
        })
    return items
```

(Confirm `Dataset.from_array` `geo_ref` is keyword-only — it is, `dataset.py:5996`.)

## 2. `[stac-parquet]` extra + pixi feature + marker (for STAC-05)

`pyproject.toml` today: the STAC deps live in `stac = ["pystac-client...",
"stac-asset..."]` (~L81); pixi has `[tool.pixi.feature.stac.dependencies]`
(~L456) and `[tool.pixi.feature.parquet.dependencies]` (~L450); environments
`dev`/`docs`/`parquet` pull features in (~L384-390); markers are registered in
`[tool.pytest.ini_options] markers = [...]` (~L331), which already has
`"parquet: tests requiring the [parquet] extra (pyarrow)"` (~L344).

Do all of:
1. **PyPI extra** — add:
   ```toml
   stac-parquet = ["stac-geoparquet>=0.8.0", "pyarrow>=16,!=19.0.0"]
   ```
2. **pixi feature** — add a `[tool.pixi.feature.stac-parquet.dependencies]` (or
   `.pypi-dependencies`) section for `stac-geoparquet`, and **include that
   feature** in the `dev` and `docs` environments (so the notebook/docs builds
   and the dev test run install it). Consider a dedicated test env if the CI
   matrix isolates extras (mirror how `parquet` is handled).
3. **pytest marker** — register:
   ```toml
   "stac_parquet: tests requiring the [stac-parquet] extra (stac-geoparquet)",
   ```
   Use `@pytest.mark.stac_parquet` on STAC-05's spec-round-trip tests (with a
   `pytest.importorskip("stac_geoparquet")` guard so a bare-core run skips
   cleanly).
4. **Verify** `pip install '.[parquet,stac-parquet]'` resolves — `[parquet]`
   allows `pyarrow>=10`; `[stac-parquet]` needs `>=16` — the union must pick
   `>=16,!=19.0.0` with no conflict.

## 3. Docs & CHANGELOG plumbing (for every task that adds public API)

- `mkdocs.yml` has a `nav:` and the reference lives under `docs/reference/stac/`.
  New public functions/kwargs must be reachable from the STAC reference and
  mentioned in `docs/tutorials/stac.md`. Add a short checklist line to each task's
  DoD (already present) and, where a new page is warranted (e.g. a `stac_cfg`
  guide for STAC-07), add it to `nav:`.
- The repo uses **commitizen** (`cz` in `pyproject.toml`); user-facing changes
  should land as conventional commits so the changelog generates. No manual
  CHANGELOG edit needed — just conventional commit messages
  (`feat(stac): ...` / `fix(stac): ...`).

## Pitfalls / regression risks

- Adding a feature to `pyproject.toml`'s pixi table without updating `pixi.lock`
  will fail CI's locked-install; re-lock (`pixi install` / the repo's lock task)
  as part of this task, or coordinate with a maintainer — **call this out in the
  PR** since `pixi.lock` is large and regenerated, not hand-edited.
- Don't add `stac-geoparquet` to the existing `stac` extra — keep it separate so
  `[stac]` (search/download) stays lean.
- The `core` marker is auto-applied when no extras marker is present, so plain
  STAC tests need no marker; only the extra-requiring ones do.

## Tests

- The new fixtures are exercised by STAC-04/05/06/08 tests; add a trivial
  `test_three_local_items_fixture` asserting it yields 3 items with the `orbit`
  property, so the fixture itself is covered.

## Definition of Done

- [ ] `tests/dataset/stac/conftest.py` with `packed_raster_path` and
  `three_local_items` (and a self-test).
- [ ] `[stac-parquet]` PyPI extra added; pixi feature added and included in
  `dev` + `docs` (and any extras test env); `pixi.lock` regenerated.
- [ ] `stac_parquet` pytest marker registered.
- [ ] Combined `[parquet,stac-parquet]` install resolves (no pyarrow conflict).
- [ ] Docs nav / tutorial touch-points identified; commitizen conventions noted.
- [ ] lint clean.
