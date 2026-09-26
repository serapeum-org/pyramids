# UGRID functionality gap analysis — pyramids vs xugrid, xarray, gridded, MDAL

What UGRID / unstructured-mesh functionality the four most common GIS/scientific-mesh
packages offer that **pyramids' `UgridDataset` does not** (as of pyramids `0.64.0`).

## Sourcing & method

- **pyramids** capabilities were read directly from the source in
  `src/pyramids/netcdf/ugrid/` (the nine modules: `dataset.py`, `mesh.py`,
  `connectivity.py`, `io.py`, `spatial.py`, `interpolation.py`, `models.py`,
  `plot.py`, `__init__.py`) — **not** from the docs, which can lag the code.
  Every "pyramids has / lacks" claim below was checked against a concrete symbol.
- **xugrid** was checked against its v0.15.3 source (`docs/api.rst`, `xugrid/__init__.py`,
  the `regrid/` and `ugrid/` modules).
- **xarray**, **gridded** (NOAA-ORR-ERD), and **MDAL** (Lutra Consulting) were catalogued
  from their source / C API headers / READMEs.

## The four packages at a glance

| Package | What it is | Role for UGRID |
|---------|-----------|----------------|
| **xugrid** | xarray extension for unstructured grids (Deltares) | The closest peer — full 1D/2D UGRID data model, regridding, topology ops |
| **xarray** | N-D labeled arrays (the netCDF data model in Python) | The *substrate*; not mesh-aware itself, but its whole selection/compute/lazy API is what xugrid inherits |
| **gridded** | Grid-agnostic model-results access layer (NOAA) | Point/time/depth value interpolation on UGRID + SGRID + regular grids |
| **MDAL** | C++ mesh IO abstraction (the "GDAL for meshes"), Python bindings | Format breadth — 20+ hydrodynamic mesh formats; the engine behind QGIS mesh layers |

## How to read this — the scope caveat

pyramids has an explicit boundary (`docs/SCOPE.md`): it owns **generic GIS primitives and
format support**, and deliberately excludes **domain value-semantics** (physics/ocean/atmos
unit systems, model-specific vertical coordinates, sensor conventions), which belong in a
downstream domain package. So a "missing" feature is not automatically a gap pyramids *should*
fill. Each item below is tagged:

- 🟢 **In-scope gap** — a generic mesh/GIS primitive or format support that fits pyramids' charter and is genuinely absent.
- 🟡 **Partial** — pyramids has something adjacent but weaker/narrower.
- 🔴 **Out of scope (by pyramids' own SCOPE.md)** — domain value-semantics; would live in a domain package (e.g. `earthlens`), reusing pyramids' generic bridges.

Legend for the matrices: ✓ built-in · ◐ partial / narrower · ✗ not provided.

---

## What pyramids' UGRID support already does (baseline)

So the gaps below are read in context — pyramids' `UgridDataset` **can**:

- Read UGRID-1.0 NetCDF via the GDAL multidim API (lazy variable arrays; windowed time-slab reads).
- Represent **2D face/node meshes** (`Mesh2d`), incl. mixed triangle/quad/ragged faces.
- Read connectivity (face_node, edge_node, face_edge, face_face, edge_face) and **derive**
  `edge_node` and `face_face` from faces.
- Compute face centroids, face areas (shoelace), fan triangulation, per-face polygons.
- Locate nearest node/face (cKDTree), points-in-bbox, and exact point-in-face (shapely STRtree).
- **Clip** to a polygon and **subset** by bbox (with full node/edge renumbering).
- **Interpolate mesh → regular raster** (`nearest` or scipy `linear`) → a pyramids `Dataset`.
- Convert to **GeoDataFrame** / **FeatureCollection** (faces→polygons, nodes→points, edges→lines).
- Detect CRS and **reproject** (`to_crs`).
- Select a time step or time range (`sel_time`, `sel_time_range`).
- Plot face data (tripcolor) / node data (tricontourf) / outline (via cleopatra).
- Write UGRID-1.0 / CF-1.8 NetCDF.

---

## Gap 1 — Topology dimensions (1D networks, 3D/layered)

| Capability | pyramids | xugrid | gridded | MDAL |
|---|---|---|---|---|
| 2D face/node mesh | ✓ `Mesh2d` | ✓ `Ugrid2d` | ✓ `Grid_U` | ✓ |
| **1D network** (nodes+edges, no faces) | ✗ | ✓ `Ugrid1d` | ◐ | ✓ (`DataOnEdges`) |
| **3D / layered volumes** | ✗ | ◐ layer dim over 2D | ✓ depth model | ✓ `DataOnVolumes` |
| Multiple topologies in one file | ◐ reads list, **loads only first** | ✓ multi-topology datasets | ◐ | ✓ |

- 🟢 **1D network meshes.** pyramids has no `Mesh1d`/network class; `read_file` always builds a
  `Mesh2d` and *requires* `face_node_connectivity` (raises otherwise), so a 1D-only UGRID file
  (river networks, 1D hydraulic models) cannot be loaded at all. `MeshTopologyInfo` carries
  `topology_dimension` (1/2/3) and `boundary_node_var`, but nothing consumes the 1D case.
  xugrid's `Ugrid1d` covers this fully (incl. directed connectivity, cyclic checks, network sort).
- 🟢 **Layered / 3D data.** pyramids correctly recognises a `(n_layers, n_face)` variable as
  *non-temporal*, but exposes only its raw array — **no layer-selection API**. xugrid treats layers
  as extra dims (works through the whole stack); MDAL has a first-class `DataOnVolumes` + vertical-level model.
  (The *physics* of a vertical coordinate is 🔴 out of scope — see Gap 8 — but generic layer *indexing* is a fair 🟢 gap.)
- 🟡 **Multi-topology files.** `parse_ugrid_topology` returns *all* topologies, but `read_file` uses
  `topologies[0]`. A file with several meshes silently drops the rest.

## Gap 2 — Regridding & interpolation between meshes/grids

pyramids interpolates **only mesh → regular raster**, with **only `nearest` and `linear`** methods,
and only that one direction. This is the single biggest capability gap versus xugrid.

| Capability | pyramids | xugrid |
|---|---|---|
| mesh → regular raster | ✓ `mesh_to_grid` (nearest/linear) | ✓ `rasterize` / `rasterize_like` |
| `rasterize_like(other)` (match an existing grid) | ✗ | ✓ |
| **Area-weighted (overlap) regridding** | ✗ | ✓ `OverlapRegridder` (mean/sum/min/max/mode/median/percentiles/harmonic/geometric) |
| **First-order conservative regridding** | ✗ | ✓ `RelativeOverlapRegridder` |
| **Barycentric interpolation** | ✗ | ✓ `BarycentricInterpolator` |
| **Centroid-locator regridding** | ✗ | ✓ `CentroidLocatorRegridder` |
| structured → unstructured, unstructured → unstructured | ✗ | ✓ |
| 1D network → 2D grid | ✗ | ✓ `NetworkGridder` |
| Reusable / serializable weights | ✗ | ✓ `.weights`, `from_weights`, `to_dataset` |
| **Point-value sampling with barycentric weights** | ◐ locates faces only | ✓ `sel_points` | (gridded: ✓ `interpolate_var_to_points`) |

- 🟢 **Overlap / conservative / barycentric regridders.** All absent. These are generic spatial
  primitives (the mesh analogue of `Dataset.resample`/`align`) and fit pyramids' charter.
- 🟢 **`rasterize_like` / reusable weights.** pyramids' `mesh_to_grid` takes `cell_size`+`bounds` but
  can't snap to an existing `Dataset`'s grid, and recomputes everything each call.
- 🟡 **Point-value sampling.** pyramids' `MeshSpatialIndex.locate_faces` returns the *containing face
  index* but not an interpolated **value at the point**. gridded (`interpolation_alphas` +
  `interpolate_var_to_points`, celltree/barycentric) and xugrid (`sel_points`) return sampled values.

## Gap 3 — NaN fill / interpolation *on* the mesh

| Capability | pyramids | xugrid | xarray |
|---|---|---|---|
| Fill NaNs by nearest on mesh | ✗ | ✓ `interpolate_na` | ◐ (structured only) |
| **Laplace interpolation** (solve ∇²=0 with data as BC) | ✗ | ✓ `laplace_interpolate` | ✗ |
| ffill / bfill / dropna along a dim | ✗ | (via xarray) | ✓ |

- 🟢 pyramids has no mesh-aware gap-filling at all. `laplace_interpolate` (fill holes by solving
  Laplace's equation) and `interpolate_na` are generic and heavily used for mesh data cleanup.

## Gap 4 — Connectivity / graph / topology operations

pyramids derives `edge_node` and `face_face` and computes geometry, but has **no graph algorithms**.

| Capability | pyramids | xugrid |
|---|---|---|
| `face_face`, `edge_node` derivation | ✓ | ✓ |
| node_node / node_face / node_edge / edge_edge connectivity | ✗ | ✓ (+ **directed** variants) |
| **Connected components** | ✗ | ✓ `connected_components` |
| **Reverse Cuthill–McKee** (bandwidth reduction) | ✗ | ✓ |
| **Binary dilation / erosion** (morphology on face masks) | ✗ | ✓ |
| **Voronoi / circumcenter tesselation**, `triangulate`, barycentric weights | ◐ fan-triangulation only | ✓ `voronoi_topology`, `circumcenters`, `tesselate_*` |
| Exterior edges/faces, **dissolved bounding polygon (with holes)** | ✗ | ✓ `exterior_edges`, `bounding_polygon` |
| `boundary_node_connectivity` | ◐ parsed, not built | ✓ |
| **Facet remapping** (`to_node`/`to_edge`/`to_face`) | ✗ | ✓ |

- 🟢 Connected components, RCM reordering, morphology, Voronoi tesselation, bounding-polygon,
  facet remapping — all generic mesh/graph primitives, all absent. (pyramids' connected-component
  *raster* `cluster` exists, but nothing equivalent on meshes.)

## Gap 5 — Vector-into-mesh & mesh-to-vector richness

| Capability | pyramids | xugrid |
|---|---|---|
| faces/nodes/edges → GeoDataFrame | ✓ `to_geodataframe` | ✓ |
| **Polygonize** (merge adjacent equal-value faces → polygons) | ✗ | ✓ `polygonize` |
| **Burn vector geometry into a mesh** | ✗ | ✓ `burn_vector_geometry` |
| **Earcut-triangulate polygons** into a mesh | ✗ | ✓ `earcut_triangulate_polygons` |
| **Line / cross-section extraction** (sample along a line) | ✗ | ✓ `intersect_line`, `intersect_linestring` |

- 🟢 pyramids can vectorise a mesh but cannot **polygonize by value**, **burn** vector data onto a
  mesh, or **extract a transect** along a line — all generic and in scope. (pyramids has these on the
  *raster* side; they're missing on the mesh side.)

## Gap 6 — Mesh construction, snapping, partitioning

| Capability | pyramids | xugrid | MDAL |
|---|---|---|---|
| Build mesh from arrays | ✓ `from_arrays` | ✓ `from_data` | ✓ `add_vertices/faces/edges` |
| **From structured / curvilinear** grid | ✗ | ✓ `from_structured2d` etc. | ✗ |
| From GeoDataFrame / shapely | ◐ (rasterize path) | ✓ `from_geodataframe`, `from_shapely` | ✗ |
| **Mesh generation / editing** (meshkernel) | ✗ | ✓ `from_meshkernel`, `.meshkernel` | ◐ structural add only |
| **Snapping** (nodes together, lines to grid edges) | ✗ | ✓ `snap_nodes`, `snap_to_grid` | ✗ |
| **METIS partitioning + merge** (domain decomposition) | ✗ | ✓ `partition`, `merge_partitions` | ✗ |
| **Periodic ↔ non-periodic** conversion | ✗ | ✓ `to_periodic` | ✗ |

- 🟢 `from_structured2d` (flatten a raster/curvilinear grid into a mesh) is a natural bridge for a
  GDAL-first library and is absent.
- 🟡 Snapping and partition/merge are generic but lean toward modelling-workflow tooling; reasonable
  candidates but lower priority against pyramids' charter.
- 🔴 Full **mesh generation** via meshkernel is arguably a domain/modelling concern, not a GDAL-style
  primitive — likely out of scope.

## Gap 7 — Format breadth (MDAL's domain)

pyramids reads UGRID **only** via NetCDF/GDAL. MDAL reads 20+ hydrodynamic mesh formats.

| Format family (MDAL) | pyramids | MDAL |
|---|---|---|
| UGRID NetCDF | ✓ | ✓ (R/W) |
| **2DM** (TUFLOW/BASEMENT/HYDRO_AS-2D) | ✗ | ✓ R/W |
| **Selafin/Serafin** (TELEMAC) | ✗ | ✓ R/W |
| **DFSU / DFS2** (DHI/MIKE) | ✗ | ✓ R |
| **XMDF / XDMF** | ✗ | ✓ R (lazy) |
| **FLO-2D, SWW (ANUGA), 3Di, H2i** | ✗ | ✓ R |
| **DAT** result files | ✗ | ✓ R/W |
| **TIN (XMS/Esri), PLY, Mike21** | ✗ | ✓ |
| GRIB / generic NetCDF meshes | ◐ (GDAL) | ✓ |

- 🟢 **Format support is explicitly in scope** per SCOPE.md ("Reading a format is never out of scope,
  even an exotic one"). So MDAL-style readers for 2DM / Selafin / DFSU / etc. are legitimate gaps —
  though realistically they'd be delivered by *wrapping MDAL* (or its Python binding) rather than
  re-implementing 20 parsers. This is the biggest breadth gap.
- Note: MDAL is purely an IO/abstraction layer — it does **no** interpolation, regridding, contouring,
  or reprojection. So beyond format reading, MDAL adds little that pyramids doesn't already exceed.

## Gap 8 — Value-semantics & domain features (mostly out of scope)

These exist in **gridded** (an ocean/model-results library) but are domain value-semantics.

| Capability | pyramids | gridded | Verdict |
|---|---|---|---|
| **Vertical coordinate transforms** (ROMS/FVCOM sigma, z-levels, depth interp) | ✗ | ✓ `S_Depth`, `ROMS_Depth`, `FVCOM_Depth`, `L_Depth` | 🔴 model-specific physics → domain package |
| **Vector fields** (u/v as one variable, vector interp) | ✗ | ✓ `VectorVariable` | 🟡 generic-ish; borderline |
| **Time interpolation** between steps (alpha blending) | ◐ selects a step only | ✓ `Time.interp_alpha` | 🟡 generic; pyramids only *selects*, doesn't interpolate in time |
| **SGRID** (staggered/curvilinear) support | ✗ | ✓ `Grid_S` | 🔴/🟡 different convention; format-support argument exists |
| Active/dry cell flags | ✗ | ✓ (MDAL `ACTIVE_INTEGER` too) | 🟡 format metadata passthrough |

- 🔴 The **sigma/vertical physics** (ROMS/FVCOM/z-level transforms) is the clearest out-of-scope case —
  it interprets what a coordinate *means* in an ocean model. By pyramids' own S1/S2 precedents this
  belongs downstream (e.g. `earthlens`), reusing pyramids' generic interpolation bridge.
- 🟡 **Time interpolation** and **vector variables** are generic enough to be reasonable, but neither
  exists today.

## Gap 9 — xarray-substrate operations (pyramids' UGRID layer wraps *no* xarray)

pyramids' `UgridDataset` is a standalone GDAL-backed object — it does **not** wrap xarray, so it
inherits none of xarray's labeled-array machinery that xugrid gets for free.

| Capability (xarray, inherited by xugrid) | pyramids UGRID |
|---|---|
| `.sel`/`.isel` label & positional selection, nearest-neighbour | ✗ (only spatial locators + time index) |
| `groupby` / `resample` / `rolling` / `coarsen` / `weighted` reductions | ✗ |
| `apply_ufunc` / `map_blocks` (arbitrary mesh-aware compute) | ✗ |
| `fillna` / `ffill` / `bfill` / `dropna` / `interpolate_na` | ✗ |
| `concat` / `merge` / `combine_by_coords` | ✗ |
| **Dask lazy/parallel** chunked arrays over mesh dims | ◐ lazy *load*, no chunked compute |
| `.dt` time accessor, CF calendar (cftime) decoding, calendar conversion | ◐ time values only |
| `stack`/`unstack`/`transpose`/`expand_dims` reshaping | ✗ |
| Accessor registration, faceted plotting | ✗ |
| **xarray interop / hand-off** (`to_xarray`/`from_xarray`) for the mesh | ✗ |
| **Zarr** write for UGRID | ✗ (NetCDF only) |

- 🟡/🟢 pyramids' *raster/NetCDF* side already has dask, xarray hand-off, groupby-ish reductions and
  Zarr — but the **UGRID layer specifically has none of them**. The highest-value, most in-character
  items here are **xarray interop for the mesh** (`UgridDataset.to_xarray`/`from_xarray`, mirroring
  `NetCDF.to_xarray`) and **Zarr write**, both of which fit the existing pattern.

---

## Priority summary (in-scope gaps, roughly high → low value)

1. **Regridding beyond nearest/linear** — overlap/area-weighted, conservative, barycentric; `rasterize_like`; reusable weights. *(Gap 2)*
2. **1D network meshes** and **generic layer indexing** for 3D/layered data. *(Gap 1)*
3. **Point-value sampling** (values at points, not just face indices) + **time interpolation**. *(Gaps 2, 8)*
4. **Mesh NaN-fill** — `interpolate_na`, Laplace fill. *(Gap 3)*
5. **Mesh↔vector richness** — polygonize-by-value, burn vector, transect/line extraction. *(Gap 5)*
6. **Connectivity/graph ops** — connected components, Voronoi tesselation, bounding polygon, facet remapping. *(Gap 4)*
7. **xarray interop + Zarr write** for the UGRID layer (mirror the NetCDF side). *(Gap 9)*
8. **Load all topologies** in a multi-mesh file, not just the first. *(Gap 1)*
9. **`from_structured2d`** (raster/curvilinear → mesh) bridge. *(Gap 6)*
10. **Format breadth via MDAL** — 2DM / Selafin / DFSU / XMDF readers (best delivered by wrapping MDAL). *(Gap 7)*

## Explicitly *out of scope* (don't add to pyramids core)

- ROMS/FVCOM/sigma **vertical-coordinate physics** and other model-specific value-semantics (→ domain package). *(Gap 8)*
- Full **mesh generation** via meshkernel (modelling concern rather than a GDAL-style primitive). *(Gap 6)*
- Rendering/analysis MDAL deliberately omits (contouring, streamlines) — pyramids already delegates viz to cleopatra.

---

*Baseline: pyramids `0.64.0`, `src/pyramids/netcdf/ugrid/` (code-verified). Peers: xugrid 0.15.3,
xarray (current), gridded (NOAA-ORR-ERD), MDAL (Lutra Consulting). "Out of scope" verdicts follow
`docs/SCOPE.md`.*

---
---

# Part 2 — Implementation plan

> **Audience:** the engineer/agent implementing these features. This part is **self-contained**:
> every task states the objective, the exact pyramids files and symbols to touch, the reference
> algorithm (with the source package + verified file:line), the correctness/regression traps, and a
> concrete Definition of Done with test guidance. **You should not need to guess anything.** Read
> §"Shared implementation context" first — it is prerequisite for every task and is not repeated
> per-task.
>
> All pyramids file:line anchors below were read from the code at commit on branch
> `claude/adoring-euler-r933jw` (pyramids `0.64.0`). If a line number has drifted, the symbol name is
> authoritative — grep for it.

## How to use this plan

- Tasks are grouped into **Epics (A–I)**. Each task has a stable id (e.g. `A2`), a **status**, an
  **objective**, a **DoD**, and full implementation notes.
- **Do tasks in dependency order** (the "Depends on" line). Within an epic, ascending order is safe.

### Task contract — what every task carries, and what it assumes (READ THIS)

Each task is written to be picked up **independently**, but "independent" means *self-contained
**together with this file's shared context***, not a standalone paragraph. Concretely:

**Every task body already contains** — objective; the exact pyramids **files/symbols** to create or
modify; the **public API** (method/class signatures); the **reference algorithm** with its verified
upstream source (xugrid/gridded/pyramids-NetCDF file:line); **correctness/regression traps**; a concrete
**Definition of Done**; and **test guidance** (fixtures, cases, location).

**Every task assumes (does NOT repeat) the shared context** — so **before starting ANY task, read, in
this order:**
1. **Part 1** — the Gap the task belongs to (the "why" + how peers do it). Each task's heading names its
   Gap number.
2. **§S.1–S.8 in full** — the mesh-dimension model (S.1), facade/engine mirror pattern (S.2),
   `MeshVariable` contract + the `from_arrays`/`dimensions` trap (S.3), rewrap/import-cycle rule (S.4),
   optional-dep guards (S.5), test conventions (S.6), geometry-dep policy (S.7), and the project
   workflow/commands/imports/ADR rules (S.8). **These are prerequisites for correctness — skipping them
   is how you introduce a regression.**
3. The task's **"Depends on"** tasks (their new helpers/APIs are your building blocks).
4. Then the task body itself.

If you are launching an agent on a single task, give it **this whole file** (or at minimum Part 1's
matching Gap + all of §S + the task + its dependencies) — never just the task paragraph. A per-task
standalone brief that inlines the needed §S excerpts can be generated on request; by default the context
is consolidated in §S to keep the plan DRY and single-source-of-truth.
- Every task ends **green** (see §S.8 for exact commands): `pixi run main` (or `pixi run test-fast`, or
  `pixi run -e dev pytest tests/ugrid`) passes, `pre-commit run -a` clean, `pixi run mypy` clean, new
  public symbols documented (Google docstrings + a `docs/reference/netcdf/ugrid/*.md` mkdocstrings stub
  wired into `mkdocs.yml` if a new module/class is added) and exported in `__init__.py`. **The changelog
  is generated by Commitizen from your Conventional-Commits message — do not hand-edit it** (§S.8).
- **Do not** widen scope into the 🔴 out-of-scope items (Part 1, Gap 8): no sigma/vertical physics, no
  meshkernel mesh generation. If a task tempts you toward them, stop and note it instead.

## Task index / status tracker

| Id | Task | Epic | Gap | Priority | Size | Status |
|----|------|------|-----|----------|------|--------|
| A1 | `isel` / `sel` (element + time selection) | A xarray-parity | 9 | High | M | ☐ Not started |
| A2 | `reduce` (time & spatial, grouped) | A | 9 | High | M | ☐ Not started |
| A3 | `rolling` (over time) | A | 9 | Med | S | ☐ Not started |
| A4 | `weighted` (area-weighted spatial mean) | A | 9 | High | M | ☐ Not started |
| A5 | `ffill`/`bfill`/`dropna`/`interpolate_na` (over time) | A | 9 | Med | M | ☐ Not started |
| A6 | `concat` / `merge` | A | 9 | Med | M | ☐ Not started |
| A7 | `diff`/`cumsum`/`cumprod`/`shift`/`argmin`/`argmax`/`squeeze` (over time) | A | 9 | Low | M | ☐ Not started |
| A8 | `stats` (per-variable min/max/mean/std) | A | 9 | Low | S | ☐ Not started |
| B1 | `mesh_to_grid` — `rasterize_like` + method additions | B regridding | 2 | High | S | ☐ Not started |
| B2 | Point-value sampling (`sample`) | B | 2 | High | M | ☐ Not started |
| B3 | Mesh↔mesh regridders (Overlap/Conservative/Barycentric/Centroid) | B | 2 | High | XL | ☐ Not started |
| B4 | Time interpolation between steps (`interp_time`) | B | 2/8 | Med | S | ☐ Not started |
| C1 | Load **all** topologies in a file | C topology | 1 | Med | M | ☐ Not started |
| C2 | Layer / vertical index selection | C | 1 | Med | M | ☐ Not started |
| C3 | 1D network mesh (`Network1d`) | C | 1 | High | XL | ☐ Not started |
| D1 | `interpolate_na` across the mesh (nearest fill) | D fill | 3 | High | S | ☐ Not started |
| D2 | `laplace_interpolate` | D | 3 | Med | L | ☐ Not started |
| E1 | `connected_components` | E graph | 4 | Med | S | ☐ Not started |
| E2 | `reverse_cuthill_mckee` | E | 4 | Low | S | ☐ Not started |
| E3 | `binary_dilation` / `binary_erosion` | E | 4 | Low | M | ☐ Not started |
| E4 | Voronoi tesselation + circumcenters | E | 4 | Low | L | ☐ Not started |
| E5 | Exterior edges/faces + `bounding_polygon` | E | 4 | Med | M | ☐ Not started |
| E6 | Extra connectivity builders + facet remap (`to_node`/`to_edge`/`to_face`) | E | 4 | Med | M | ☐ Not started |
| F1 | `polygonize` (merge equal-value faces) | F vector | 5 | Med | M | ☐ Not started |
| F2 | Transect / line extraction (`sample_line`) | F | 5 | Med | M | ☐ Not started |
| F3 | `burn` vector geometry onto faces | F | 5 | Low | L | ☐ Not started |
| G1 | `from_dataset` / `from_structured` (raster→mesh) | G construct | 6 | Med | M | ☐ Not started |
| G2 | `from_geodataframe` (polygons→mesh) | G | 6 | Low | M | ☐ Not started |
| H1 | `to_xarray` / `from_xarray` | H interop | 9 | High | L | ☐ Not started |
| H2 | `to_zarr` / `from_zarr` | H | 9 | Med | M | ☐ Not started |
| H3 | Lazy dask `chunks=` path for mesh variables | H | 9 | Med | L | ☐ Not started |
| I1 | `from_mdal` (2DM/Selafin/DFSU/… via mdal-python) | I formats | 7 | Low | L | ☐ Not started |

Size key: S ≈ ½–1 day, M ≈ 1–3 days, L ≈ 3–5 days, XL ≈ 1–2 weeks. Mark status ☐→◧ (in progress)→☑ (done) as you go.

---

## Shared implementation context (READ FIRST — prerequisite for all tasks)

### S.1 The mesh-dimension model (the single most important thing to get right)

A `UgridDataset` has **no structured (row, col) grid**. Every data field is a `MeshVariable` living on
exactly **one** mesh location — `node` / `face` / `edge` — identified by `MeshVariable.location`. The
element counts are `n_node = len(node_x)`, `n_face = face_node_connectivity.n_elements`,
`n_edge = edge_node_connectivity.n_elements` (0 when there is no edge connectivity).

`MeshVariable.data` array-shape invariants (verified in `models.py`):

| Variable kind | `data.shape` | element axis | time axis | notes |
|---|---|---|---|---|
| static face var | `(n_face,)` | `-1` (only axis) | — | `has_time == False` |
| temporal face var | `(n_time, n_face)` | `-1` | `0` | `has_time == True`, `time_index == 0` |
| layered (non-temporal) | `(n_layers, n_face)` | `-1` | — | `has_time == False`, `time_index is None` (ARC-19) |
| temporal + layered | `(n_time, n_layers, n_face)` | `-1` | `time_index` | rare; layer axis is the middle one |

**Rules every Epic-A task must obey:**

1. **The element axis is always the last axis (`-1`).** Use `arr[..., idx]` to index/slice elements so
   any leading time/layer axes are preserved — this is exactly how
   `spatial.py:_subset_mesh_by_face_indices` slices (see `spatial.py:414`).
2. **The time axis is `MeshVariable.time_index`** (0 for temporal vars, `None` otherwise). Never assume
   "extra axis == time": a `(n_layers, n_face)` variable is **not** temporal. Read `time_index`
   (`models.py:166`), which matches CF dim names (`time`/`t`/`time…`/`…_time`/`…_time_…`) with
   word-boundary logic to avoid `runtime`/`lifetime` false positives.
3. **Reducing/selecting over `"time"`** operates on axis `time_index` and yields a variable that is
   still on the mesh → return a **new `UgridDataset`**.
4. **Reducing over the element axis (`"face"`/`"node"`/`"edge"`, or `"space"`)** collapses the mesh →
   the result is **no longer mesh-located**. Return a tabular/array result (a `pandas.DataFrame`
   indexed by time, or a `dict[str, np.ndarray]`), **not** a `UgridDataset`. This mirrors how
   `NetCDF.weighted` on spatial axes collapses the grid.
5. **Only operate on variables for which the dim exists.** When a dataset-wide op names `dim="time"`,
   process temporal variables and **carry static/layered variables through unchanged** (do not error).
   Mirror `NetCDF`'s "carry others, warn if an aux var spans the reduced dim" behaviour.

### S.2 The pyramids facade/engine house pattern (mirror it — don't invent a new shape)

pyramids' NetCDF side already implements the entire xarray-parity surface **in pyramids** (numpy, no
xarray at runtime). The architecture, which you must mirror for the mesh:

- A **thin facade method** on the container (`NetCDF`) with a fully spelled-out signature + short
  Google docstring, delegating to…
- …an **engine** class/function holding the body, in `src/pyramids/netcdf/engines/`.
- **Grouped optional parameters** are `@dataclass(frozen=True)` value objects (see
  `netcdf/plot_options.py`, `netcdf/array_options.py`) — all fields defaulted, validation in
  `__post_init__`, an `as_dict()` where useful.

For the mesh, create engine modules under **`src/pyramids/netcdf/ugrid/engines/`** (new package;
mirror `netcdf/engines/`) and keep `UgridDataset` methods as thin facades. **Reuse the existing numpy
op bodies wherever the math is identical** — they are already mesh-agnostic per-axis kernels:

- `src/pyramids/netcdf/engines/_along_dim.py` — `_Reduction`, `_Rolling`, `_Diff`, `_CumSum`,
  `_Shift`, `_Extremum`, `_DropNa`, ffill/bfill/interp ops, and the `_apply_to_variable` loop. These
  operate on a numpy array + an axis; they do **not** import xarray. Prefer calling these over
  re-implementing. (If a helper is too `NetCDF`-coupled, lift the pure-numpy core into a shared
  `_reductions` helper rather than copy-pasting.)
- `src/pyramids/dataset/_reduce_ops.py:20` — `resolve_dask_op(op_name, *, skipna)` maps
  `mean/sum/min/max/std/var` → `dask.array.nan*`; reuse for the lazy path.
- `src/pyramids/netcdf/engines/_weighted.py` — weighted-mean math for A4.
- `src/pyramids/netcdf/engines/combine.py` — `concat`/`merge` structure for A6.

### S.3 `MeshVariable` construction contract (how to return derived data)

To produce a derived variable, **always** use `MeshVariable.with_data(new_array)` (`models.py:322`).
It returns an **eager** copy preserving `name`, `location`, `mesh_name`, `attributes`, `nodata`,
`units`, `standard_name`, `dimensions`, and sets `shape` from `new_array.shape`. **Do not** construct
`MeshVariable(...)` by hand in derived ops (you would drop metadata).

⚠️ **`dimensions` bookkeeping:** `with_data` copies `dimensions` verbatim. If your op **removes** an
axis (e.g. `reduce(dim="time")` drops the time axis, `sel_time(i)` already does), the returned
`dimensions` tuple will be **stale** (still lists `time`). Extend `with_data` with an optional
`dimensions=` override, OR set `_data`+`shape`+`dimensions` consistently. Add a regression test that
`result[var].dimensions` matches `result[var].data.ndim`. (Today `sel_time_range` keeps `dimensions`
because it keeps the time axis; `sel_time` returns a bare array via `MeshVariable.sel_time`, not a
`MeshVariable`, so it sidesteps this — your dataset-level ops won't.)

⚠️ **`from_arrays` does not set `dimensions` (verified, `dataset.py:800`).** Its signature is
`from_arrays(node_x, node_y, face_node_connectivity, data=None, data_locations=None, epsg=4326, mesh_name=...)`
— it builds each `MeshVariable` with `dimensions=()`. Consequence: **a variable built via `from_arrays`
has no dimension names, so `time_index` falls back to the leading-axis heuristic** (`models.py:191`:
`0 if ndim > 1 else None`). A `(n_layers, n_face)` array built through `from_arrays` will therefore be
**mis-detected as temporal** (axis 0 read as time). This bites C2 (layers) and any Epic-A test that
needs an unambiguous temporal-vs-layered distinction. **Two fixes, pick per task:**
(a) **Recommended foundational change:** add an optional `dimensions: dict[str, tuple[str, ...]] | None`
parameter to `from_arrays` so callers/tests can name axes (e.g. `{"salinity": ("nLayers", "nFaces")}`),
threaded into each `MeshVariable(dimensions=...)`. Do this once, early — it makes every downstream test
honest. (b) For a one-off test, construct the `MeshVariable(..., dimensions=("time","nFaces"))` directly
and assemble the `UgridDataset(mesh=..., data_variables={...}, global_attributes={...})` by hand.
File-read datasets are unaffected — `_read_data_variables` populates `dimensions` from the GDAL dims.

### S.4 Rewrapping into a `UgridDataset` (avoid the import cycle)

- Spatial/topology ops that **rebuild the mesh** live as standalone functions in `spatial.py` (or a
  new sibling module) and return a **`(Mesh2d, dict[str, MeshVariable])` tuple**, never a
  `UgridDataset` — this is a hard structural invariant (`TestNoImportCycle` asserts `spatial.py` has
  no module-scope import of `ugrid.dataset`). The facade then calls `self._wrap_subset(mesh, data_vars)`
  (`dataset.py:434`), which carries `global_attributes` / `topology_info` / `crs_wkt` onto the result
  and sets `file_name=None`.
- Ops that **keep the same mesh** and only transform variable arrays (all of Epic A over time) build a
  `new_data_vars: dict[str, MeshVariable]` and construct
  `UgridDataset(mesh=self._mesh, data_variables=new_data_vars, global_attributes=self._global_attributes, topology_info=self._topology_info, crs_wkt=self._crs_wkt)` directly (see how `sel_time`/`sel_time_range` do it around `dataset.py:599`).

### S.5 Optional dependencies & imports

- **scipy** and **shapely** are hard deps of the ugrid modules — import at module top (as
  `spatial.py`/`interpolation.py` already do: `from scipy.spatial import cKDTree`,
  `from scipy.interpolate import LinearNDInterpolator`).
- **scipy.sparse / scipy.sparse.csgraph / scipy.sparse.linalg** — hard-dep-safe to import at top; used
  by D2/E1/E2/E3/F1.
- **xarray, dask, zarr** are **optional** (`[lazy]`/interop extras). Guard exactly like the NetCDF side:
  `xr = import_xarray(msg)` / `import_dask(...)` from `src/pyramids/base/_utils.py` (`import_xarray` at
  `_utils.py:1493`). Never add a top-level `import xarray`/`import dask` to a ugrid module. Missing dep
  → raises `OptionalPackageDoesNotExist`.
- **cleopatra** (viz) — guarded lazily inside functions via `require_cleopatra(_CLEOPATRA_MSG)` then a
  function-local import (see `plot.py`).
- **numba_celltree / meshkernel / mapbox_earcut** — **new** optional deps; only B3/F3 (and optionally
  B2/F2) need them. See §S.7 before adding any.
- Errors: raise plain builtins (`ValueError`, `KeyError`, `IndexError`, `TypeError`) as the ugrid
  subsystem already does; the only pyramids-specific error reachable today is
  `OptionalPackageDoesNotExist`. Import errors from `pyramids.errors`, never `pyramids.base._errors`.
- Style: every module starts `from __future__ import annotations`; Google docstrings; PEP-604 unions
  (`X | None`); `np.typing.NDArray`; `Any` for GDAL/cleopatra/xarray objects.

### S.6 Test conventions (match these exactly)

- Tests live in **`tests/ugrid/`**; module-level `pytestmark = pytest.mark.core`; classes
  `Test<Feature>`, methods `test_<behavior>`, Google docstrings.
- Build fixtures **both** ways: tiny hand-built `Mesh2d(...)` + `Connectivity(...)` (see
  `tests/ugrid/conftest.py`: `triangle_mesh` = 5 nodes/2 tris `[[0,1,2],[1,4,3]]`, `mixed_mesh` = quad+2
  tris with `-1` fills), and full-dataset `UgridDataset.from_arrays(...)` (e.g. `unit_square_dataset`:
  9 nodes, 4 quads, face var `temperature=np.arange(4)`). A convention sample file exists at
  `tests/data/netcdf/ugrid/ugrid.nc` (8916 nodes, 8355 faces, 17270 edges; vars `mesh2d_node_z` (node),
  `mesh2d_edge_type` (edge)) via the `ugrid_convention_nc_path` session fixture.
- Assert arrays with `np.testing.assert_array_equal` / `assert_array_almost_equal` (`err_msg=`);
  warnings with `pytest.warns(UserWarning, match=...)`; errors with `pytest.raises(ValueError, match=...)`.
- For every derived op, add tests that (a) preserve `location`, `units`, `nodata`; (b) keep the element
  axis last and time/layer axes correct; (c) round-trip shapes; (d) leave the input unmutated.

### S.7 Dependency policy for the geometry-heavy tasks (B3/F3, optionally B2/F2/E4)

The overlap/conservative/barycentric regridders and polygon-burning depend on a **cell-tree +
polygon-clipping engine**. Two routes — **decide and record the choice in the task before coding**:

1. **Wrap `numba_celltree`** (what xugrid uses) as a new **optional** dep behind a `[mesh]` extra.
   Pro: correct, fast, battle-tested `intersect_faces`/`locate_points`/`intersect_edges`. Con: new
   heavy dep (numba). This is the recommended route for B3.
2. **Pure numpy/scipy** for the subset that doesn't need polygon clipping: nearest/barycentric **point**
   sampling (B2) needs only a KDTree/point-in-triangle test (gridded's closed-form alphas, §ref below);
   nearest-fill (D1), Laplace (D2), connectivity/graph ops (E1–E6), `from_structured` (G1), and
   `polygonize` (F1, needs shapely only) are **pure numpy/scipy/shapely — no cell tree**. Prefer this
   route wherever it suffices; it adds no dependency.

**Never** vendor a partial/incorrect polygon-intersection by hand for B3 — an approximate overlap area
silently corrupts conservative regridding. Use route 1 for true area-overlap.

### S.8 Project workflow & conventions (how to run, test, document, land — all verified)

This repo uses **pixi** (conda-forge) for envs and tasks, and **Commitizen** for versioning/changelog.
Exact facts (from `pyproject.toml`, `tests/_marks.py`, `docs/`):

**Environments** (`[tool.pixi.environments]`): `default` = `gdal,viz`; `dev` = `gdal,viz,dev,interop,lazy,stac,parquet` (use `dev` for implementation — it has every optional dep); `docs` env builds the site.

**Run tests** (pixi tasks, `[tool.pixi.tasks]`):
- `pixi run main` — main suite, marker filter `-m 'not plot and not interop'`, with coverage. **This is the default gate.**
- `pixi run test-fast` — same suite in parallel (`pytest -n auto --dist loadfile`), no coverage; fastest local loop.
- `pixi run plot` — viz/cleopatra tests (`-m plot`); `pixi run interop-tests` — xarray interop (`-m interop`); `pixi run test-all` — everything.
- `pixi run mypy` — type-check. Lint/format is **pre-commit** (`pre-commit run -a`; config `.pre-commit-config.yaml`).
- Target the ugrid suite directly while iterating: `pixi run -e dev pytest tests/ugrid -q` (add `-k <name>`).
- There is **no** `pixi run test`; don't invent task names.

**pytest markers** (`[tool.pytest.ini_options].markers` + `tests/_marks.py`): `core` is **auto-applied to
extras-free tests** — a pure-numpy/scipy/shapely mesh test needs **no** marker (it's `core`). Extra-gated
tests auto-skip when the dep is absent via `_has("<module>")` probes in `tests/_marks.py`
(`HAS_INTEROP=_has("xarray")`, `HAS_DASK`, `HAS_ZARR`, `HAS_CLEOPATRA`, …) mapped in `EXTRA_MARKERS`.
**When you add a new optional extra (`[mesh]` for B3/F3, `[mdal]` for I1) you MUST:**
1. add the extra to `[project.optional-dependencies]` in `pyproject.toml`;
2. register a marker line under `[tool.pytest.ini_options].markers` (e.g.
   `"mesh: tests requiring the [mesh] extra (numba_celltree)"`);
3. add a probe + mapping in `tests/_marks.py` (`HAS_NUMBA_CELLTREE = _has("numba_celltree")`, wire into
   `EXTRA_MARKERS`), and mark the tests `@pytest.mark.mesh`. Then they auto-skip on core installs.
Existing markers you'll reuse: `plot` (viz), `lazy` (dask/zarr — H2/H3), `interop` (xarray — H1).

**Helper import paths (exact — don't guess):**
- `from pyramids.base.crs import sr_from_epsg, crs_spec, crs_from_user_input` (`crs.py:69/737/1369`).
- `from pyramids.base.georeference import GeoReference` (`georeference.py:29`) — used by B1/G1 output.
- `from pyramids.base._utils import require_optional, extra_hint, import_xarray, import_dask, import_dask_geopandas`
  (`_utils.py:1218/1411/1493/1483/1463`); `require_cleopatra` is also in `_utils.py:1310`. Optional-dep
  guard idiom: `xr = import_xarray(extra_hint("xarray is required for <feature>.", "interop"))`.
- `from pyramids.errors import AlignmentError, OptionalPackageDoesNotExist` (never `base._errors`).

**Public-API exposure:** add every new public class/dataclass to
`src/pyramids/netcdf/ugrid/__init__.py` `__all__`; if it's user-facing at the package top (like
`ExtraDimensions` is on `netcdf`), also re-export from `src/pyramids/netcdf/__init__.py`.

**Reference docs (mkdocstrings):** each new module/class gets a stub
`docs/reference/netcdf/ugrid/<name>.md` following the existing pattern (see
`docs/reference/netcdf/ugrid/mesh.md`):
````
::: pyramids.netcdf.ugrid.<Symbol>
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
````
and wire the page into the nav in `mkdocs.yml`. Docstrings are **not** doctested in CI (no
`--doctest-modules`), but keep `>>>` examples correct — example **notebooks** under `docs/examples/` are
executed via `pixi run notebooks` (nbval); add a short ugrid example notebook for a big feature.

**Commits, changelog & versioning (CORRECTION — do NOT hand-edit a changelog):** the changelog is
**generated by Commitizen** (`pixi run cz-changelog`) from **Conventional-Commits** messages, and the
version is bumped by `pixi run cz-bump`. So the per-task deliverable is a well-formed conventional commit
— `feat(ugrid): add UgridDataset.reduce`, `fix(ugrid): …`, `docs(ugrid): …`, `test(ugrid): …` — **not** a
manual `docs/change-log.md` edit. Do not touch `docs/change-log.md` by hand.

**ADRs:** architectural decisions live in `docs/adr/` (0001–0007 exist; e.g. `0003-io-formats`,
`0007-processing-registry-approach`). The **big** tasks warrant a new ADR (next number, `0008+`) before
coding: **B3** (regridder subpackage + `numba_celltree`/`[mesh]` dep), **C3** (1D `Network1d` — changes
the `UgridDataset` topology contract), **H1** (the UGRID↔xarray encoding), the **`ugrid/engines/`**
package introduction (Epic A), and **I1** (`[mdal]` dep). Small pure-numpy tasks (most of A/D/E/F/G) do
not need an ADR.

**Read before starting:** `docs/contributing.md` (contribution rules), `docs/SCOPE.md` (the in/out-of-scope
boundary — re-check every task against it), and the ADR index `docs/adr/index.md`.

---

## Epic A — xarray-parity on the mesh (Gap 9) — the priority the user called out

**Goal:** give `UgridDataset` the labeled-array operations xarray/xugrid have, by **mirroring the
existing `NetCDF` facades/engines** onto the mesh element/time axes (§S.1, §S.2). These are
pure-numpy, no new deps, and the highest-confidence tasks.

**Epic-wide DoD:** each new method has a spelled-out facade on `UgridDataset`, an engine body under
`ugrid/engines/` (new package — introduce it in A1/A2 and consider an ADR, §S.8), Google docstrings,
`core`-marker tests in `tests/ugrid/` (no extras needed — all pure numpy), a reference-doc stub, and a
Conventional-Commits message (changelog auto-generated, §S.8). The `ugrid/engines/` package mirrors
`netcdf/engines/`; add it to `__init__.py` exports where symbols are public. Input dataset is never
mutated. `location`/`units`/`nodata`/`standard_name` are preserved on every derived variable (§S.3).

### A1 — `isel` / `sel` (element + time selection)
- **Status:** ☐ Not started **· Depends on:** none **· Size:** M
- **Objective:** positional and label selection over the two meaningful mesh axes: the **element** axis
  (by integer position) and the **time** axis (by position and by time label). There is no 1-D
  coordinate along the element axis, so element **label** selection is spatial and stays in Epic B
  (`sample`) / `crop`; do **not** fake a coordinate.
- **API (facades on `UgridDataset`):**
  - `isel(self, *, time: int | slice | Sequence[int] | None = None, element: int | slice | Sequence[int] | None = None) -> UgridDataset`
  - `sel(self, *, time=..., method: str | None = None, tolerance=None) -> UgridDataset` — `time` by label
    (int index or datetime-like matched against `time_values`); `method ∈ {None,"nearest"}`.
- **Reference:** pyramids `NetCDF.isel`/`sel` (`engines/selection.py:731`/`:874`) for the parameter
  rules (lists sorted+deduped, `bool` refused, N-d indexers refused; `tolerance` only with
  `method="nearest"`, `KeyError` on tolerance breach). Nearest primitive: `_label_select.nearest_indices`
  (`_label_select.py:526`) — **reuse it** for time-label nearest.
- **Implementation:**
  1. **time selection:** for each variable, if `has_time`, index axis `time_index` with the resolved
     indexer using `np.take`/basic slicing on that axis; if not temporal, **carry through unchanged**.
     Update `dimensions` (drop `time` on a scalar int selection; keep it on slice/sequence). Build
     `new_data_vars` and construct a `UgridDataset` on the same mesh (§S.4).
  2. **element selection:** this changes the mesh itself (fewer nodes/faces/edges) → it is a
     **topology subset**. Reuse the existing machinery: `element` by face position ⇒ call
     `_subset_mesh_by_face_indices(self, list(indices))` (`spatial.py:311`) then `_wrap_subset`.
     Restrict `isel(element=...)` to **face-located** meshes/vars in v1 and document that node/edge
     positional subsetting is not yet supported (node/edge subsetting can't rebuild a valid face
     topology). Raise `ValueError` if the dataset has non-face variables when `element` is given.
- **Correctness traps:** (a) element axis is `-1`, time axis is `time_index` — never transpose;
  (b) reject boolean and N-d indexers like `NetCDF.isel` does; (c) a scalar `time=i` must **drop** the
  time axis and update `dimensions` (§S.3 stale-dimensions trap); (d) don't mutate input arrays — slice
  produces views, wrap via `with_data(np.ascontiguousarray(...))` if you'll write later.
- **DoD:** both methods; time int/slice/list + nearest-by-label; `element` face subsetting delegates to
  the verified subsetter; tests cover temporal + static + layered vars in one dataset (statics carried
  through), nearest tolerance breach → `KeyError`, scalar-time dimension drop, input unmutated.

### A2 — `reduce` (aggregate over time or over the mesh)
- **Status:** ☐ Not started **· Depends on:** A1 (shares axis-resolution helper) **· Size:** M
- **Objective:** `reduce(dim, how="mean", *, groupby=None, skipna=True, q=None)` — aggregate a variable
  either along **time** (→ new `UgridDataset` on the same mesh) or along the **element** axis (→ tabular
  per-time result; §S.1 rule 4). Support grouped reduction via a per-step label array (this is
  pyramids' resample/groupby mechanism — there is no separate `.groupby()` object).
- **API:** `reduce(self, dim: str, how: str = "mean", *, groupby=None, skipna: bool = True, q: float | None = None) -> UgridDataset | pandas.DataFrame`
  where `dim ∈ {"time"}` → `UgridDataset`; `dim ∈ {"face","node","edge","space"}` → `DataFrame`
  (index = time or a single row; columns = variables).
- **`how` set:** `{mean,sum,min,max,std,var,median,prod,quantile(needs q),count,all,any}` — identical to
  `NetCDF.reduce` (`selection.py:2141`). **Reuse the numpy op bodies** in
  `engines/_along_dim.py:_Reduction` rather than re-deriving (§S.2).
- **Implementation:**
  1. Resolve axis: `dim=="time"` → `var.time_index` (skip/ carry non-temporal vars); element dims → `-1`.
  2. `groupby` (only meaningful with `dim=="time"`): given labels of length `n_time`, aggregate each
     group along the time axis, stack results back along a new time axis whose coordinate is the group
     key. Mirror `NetCDF.reduce(groupby=...)`.
  3. Reduction over time → `with_data` new array + drop time axis (or keep grouped time axis); construct
     `UgridDataset`. Reduction over element axis → produce `{name: reduced_array}` and assemble a
     `DataFrame` (time-indexed when temporal).
- **Correctness traps:** `skipna` uses `np.nanmean` etc.; `count` counts non-NaN; `quantile` requires
  `q` (else `ValueError`, like NetCDF); do not silently return a `UgridDataset` for an element-axis
  reduce (that would misrepresent a collapsed mesh as still-on-mesh — §S.1 rule 4).
- **DoD:** all `how` values; time-reduce → UgridDataset (mesh unchanged, statics carried); space-reduce
  → DataFrame; `groupby` labels; `q`-guard; parity test vs `numpy.nan*` on a hand-built temporal var.

### A3 — `rolling` (moving window over time)
- **Status:** ☐ Not started **· Depends on:** A2 **· Size:** S
- **Objective:** `rolling(dim="time", window, *, how="mean", center=False, min_periods=None, q=None)` —
  moving-window aggregation along the time axis, output keeps the time length. **Only `dim="time"`** is
  meaningful (a moving window over unstructured elements has no defined order) → raise `ValueError` for
  element dims with a clear message.
- **Reference/reuse:** `NetCDF.rolling` (`selection.py:2492`) + `engines/_along_dim.py:_Rolling`. Reuse
  the `_Rolling` numpy body on axis `time_index`.
- **DoD:** matches `NetCDF.rolling` semantics on the time axis; `center`/`min_periods` honored; static
  vars carried through; element-dim request rejected; test vs a hand-computed rolling mean.

### A4 — `weighted` (area-weighted spatial mean — the mesh-natural reduction)
- **Status:** ☐ Not started **· Depends on:** A2 **· Size:** M
- **Objective:** `weighted(weights="area", *, how="mean", skipna=True) -> pandas.DataFrame` — collapse
  the **element** axis with weights. For **face** variables the natural weight is `mesh.face_areas`
  (already implemented, shoelace, `mesh.py`); accept `weights="area"` (→ `face_areas`) or an explicit
  1-D array of length `n_face`/`n_node`/`n_edge` matching the variable's location.
- **Reference:** `NetCDF.weighted` (`selection.py:3266`, engine `_weighted.py`) — where weighting the
  spatial axes collapses the grid to one value; `how` includes `sum_of_weights`. Mesh analog: weighted
  reduce over axis `-1`.
- **Implementation:** `result = Σ(w·v)/Σ(w)` over axis `-1` with NaN-aware masking (skip NaN values and
  their weights when `skipna`); per time step for temporal vars → a time-indexed `DataFrame`. `how ∈
  {mean, sum, sum_of_weights, std}`.
- **Correctness traps:** weight length must equal the variable's `n_elements` for its location (validate,
  else `ValueError`); `"area"` is only defined for face vars (node/edge → require explicit weights);
  area weights come from `mesh.face_areas` which is fill-aware — don't recompute.
- **DoD:** `weights="area"` and explicit-array paths; per-time DataFrame; NaN handling; length
  validation; test that area-weighted mean of a constant field equals that constant.

### A5 — `ffill` / `bfill` / `dropna` / `interpolate_na` (along time)
- **Status:** ☐ Not started **· Depends on:** A2 **· Size:** M
- **Objective:** gap-handling **along the time axis** (distinct from D1/D2 which fill across mesh
  *space*). Names and semantics mirror `NetCDF` exactly:
  - `ffill(dim="time", *, limit=None)` / `bfill(...)` — `selection.py:3431`/`:3492`.
  - `dropna(dim="time", *, how="any", thresh=None)` — drops time steps; refuses to drop all
    (`ValueError`); `selection.py:3540`.
  - `interpolate_na(dim="time", method="linear", *, limit=None, use_coordinate=True)` — `method ∈
    {linear,nearest}`; leading/trailing gaps left alone; `selection.py:3610`.
- **Reuse:** the ops already exist in `engines/_along_dim.py`; apply on axis `time_index`.
- **Naming caution:** the mesh-space fill in Epic D **must** be named differently to avoid a collision —
  use `fill_na_spatial` / `laplace_interpolate` for D, and reserve `interpolate_na(dim=...)` here for the
  time axis (matching NetCDF). Document the distinction in both docstrings.
- **DoD:** four methods on the time axis; `dim!="time"` rejected with a clear message; static vars
  carried; parity tests vs NetCDF behaviour on a temporal face var with injected NaNs.

### A6 — `concat` / `merge`
- **Status:** ☐ Not started **· Depends on:** none **· Size:** M
- **Objective:** combine multiple `UgridDataset`s. `concat(objs, dim="time")` joins temporal cubes
  end-to-end (same mesh + same variables required); `merge(objs, *, compat="no_conflicts")` puts
  different variables side-by-side on the **same mesh**.
- **Reference:** `netcdf/engines/combine.py` (`concat` :38, `merge` :207) and the `@_joins_cubes`
  decorator (`netcdf.py:2111`) that makes them callable both as `UgridDataset.concat([a,b], dim)` and
  `a.concat([b], dim)`.
- **Mesh-specific check:** meshes must be **identical topology** to combine. Add a
  `Mesh2d.__eq__`/`is_congruent(other)` helper (compare `n_node`/`n_face`, node coords within tolerance,
  and `face_node_connectivity.data`) — there is none today; a naive identity check will wrongly reject
  equal meshes read twice. Raise `AlignmentError` (from `pyramids.errors`) on mismatch, matching NetCDF.
- **DoD:** `concat` along time (coords joined in order, CF units/calendar carried only if all agree);
  `merge` side-by-side with `compat ∈ {no_conflicts,override}`; congruent-mesh guard with a real
  tolerance; both call shapes; tests for happy path + mismatched-mesh rejection + variable-name conflict.

### A7 — `diff` / `cumsum` / `cumprod` / `shift` / `argmin` / `argmax` / `squeeze` (over time)
- **Status:** ☐ Not started **· Depends on:** A2 **· Size:** M
- **Objective:** the remaining per-axis transforms, all along the time axis, mirroring the identically
  named `NetCDF` methods (`selection.py` `diff`:2632, `cumsum`:2752, `cumprod`:2845, `shift`:2917,
  `argmin`:3026, `argmax`:3099, `squeeze`:1740). Reuse `engines/_along_dim.py` bodies (`_Diff`,
  `_CumSum`, `_Shift`, `_Extremum`).
- **Notes:** `argmin`/`argmax` over time return the **time index** of the extremum per element (static
  face var). `squeeze` drops length-1 time/layer axes and fixes `dimensions`. Group these into one
  task/PR since they share the engine wiring.
- **DoD:** each method on the time axis with NetCDF-matching semantics; `dimensions` kept consistent;
  tests per method on a hand-built temporal var.

### A8 — `stats` (per-variable summary)
- **Status:** ☐ Not started **· Depends on:** none **· Size:** S
- **Objective:** `stats(self, variable_name: str | None = None) -> pandas.DataFrame` — per-variable
  `[min, max, mean, std, count]` in physical units, one row per variable (or per time step if you want
  parity with `Dataset.stats`; keep v1 simple: one row per variable, over all elements & steps).
- **Reference:** `Dataset.stats` (`dataset/dataset.py:845` → `engines/analysis.py:443`) for the return
  shape/columns. Mesh version just uses numpy nan-stats over the whole `MeshVariable.data`.
- **DoD:** DataFrame with the standard columns; respects `nodata` (mask before stats); test on
  `unit_square_dataset`.

---

## Epic B — Regridding & interpolation (Gap 2)

**Goal:** go beyond the current single mesh→raster path (`interpolation.py:mesh_to_grid`, only
`nearest`/`linear`). Existing code to build on: `interpolation.py` (`mesh_to_grid`, `_get_source_data`
which picks face centroids / node coords / edge centers by `location`, `_interpolate_nearest` via
cKDTree, `_interpolate_linear` via scipy `LinearNDInterpolator`), and `UgridDataset.to_dataset`
(`dataset.py:269`) which wraps `mesh_to_grid` into a pyramids `Dataset`.

### B1 — `mesh_to_grid`: `rasterize_like` + method additions
- **Status:** ☐ Not started **· Depends on:** none **· Size:** S
- **Objective:** (a) add `UgridDataset.rasterize_like(reference: Dataset, variable_name, *, method="nearest", nodata=-9999.0) -> Dataset` that samples mesh data onto an **existing** `Dataset`'s grid
  (matching its geotransform/CRS/shape) instead of only `cell_size`+`bounds`; (b) surface the existing
  `nearest`/`linear` plus (optionally) an inverse-distance-weighted `idw` method in `mesh_to_grid`.
- **pyramids internals (verified):** `to_dataset` (`dataset.py:269`) computes a grid from
  `cell_size`/`bounds`, calls `mesh_to_grid(...)` → `(grid_array, geotransform)`, and builds the result
  via `Dataset.from_array(grid_array, no_data_value=nodata, geo_ref=GeoReference(geo=geotransform, epsg=target_epsg))`.
  For `rasterize_like`, read the reference `Dataset`'s accessors (all confirmed to exist in
  `dataset/dataset.py`): `geotransform` (used internally at `dataset.py:3547`), `rows` (:3290),
  `columns` (:3295), `shape` (:3300), `epsg` (:3305), `cell_size` (:3363). Generate cell-center
  coordinates from `geotransform`+`rows`+`columns`, sample with the existing cKDTree/`LinearNDInterpolator`
  path in `interpolation.py`, then build the output with
  `Dataset.from_array(grid, no_data_value=nodata, geo_ref=GeoReference(geo=reference.geotransform, epsg=reference.epsg))`
  so it is byte-for-byte co-registered with the reference. **Note `bounds` on `Dataset` returns a
  `GeoDataFrame` (`dataset.py:3919`), not a tuple** — don't use it as a bbox; use `geotransform`+shape.
  Reproject mesh coords to the reference CRS first if they differ (use `to_crs`). Temporal vars: follow
  the existing `to_dataset` convention — it rasterizes the **first** time step (`data[0]` when
  `var.has_time`); either match that (document it) or add an explicit `time_index=` param.
- **Correctness traps:** the reference grid may be in a different CRS — reproject mesh source points to
  match before sampling (don't assume same CRS); honor the reference's nodata; keep row/col orientation
  consistent with pyramids raster convention (row 0 = north).
- **DoD:** `rasterize_like` returns a `Dataset` co-registered with the reference (assert identical
  geotransform/shape/CRS); handles CRS mismatch; test against a `to_dataset` grid used as the reference.

### B2 — Point-value sampling (`sample`)
- **Status:** ☐ Not started **· Depends on:** none **· Size:** M
- **Objective:** `sample(self, x, y, variable_name, *, method="nearest") -> np.ndarray` — return the
  **value** of a variable at arbitrary points (not just the face index that `MeshSpatialIndex.locate_faces`
  gives today). This closes the "locates faces but not values" gap vs gridded/xugrid.
- **Reference algorithm (pure numpy + scipy — no cell tree needed):**
  - `method="nearest"`: for **face** vars, `locate_faces(x,y)` (STRtree containment, `spatial.py`)
    then take `data[..., face_index]`, with `-1` (outside) → `nodata`/NaN. For **node** vars, nearest
    node via `locate_nearest_node`. Straightforward.
  - `method="linear"` for **node-located** data on **triangular** faces: use gridded's closed-form
    **barycentric** weights (verified in gridded `pyugrid/ugrid.py`): locate the containing face, gather
    its 3 node coords, compute `denom = (lat3-lat1)(lon2-lon1) - (lon3-lon1)(lat2-lat1)` and the three
    sub-area alphas `α1,α2,α3` (each `/denom`), zero alphas for points where face index == -1, then
    `value = Σ αk · node_value[k]`. This is the light-weight path; document that it requires triangular
    faces (check `mesh.face_node_connectivity.is_triangular()`; fall back to nearest otherwise).
- **Correctness traps:** points outside the mesh → `nodata` (never silently 0); temporal vars → sample
  keeps the leading time axis (`data[..., idx]`); barycentric path is defined only for triangular faces
  and node data — guard and fall back; reuse the existing `MeshSpatialIndex` (don't build a second index).
- **DoD:** nearest for face+node; barycentric-linear for triangular node data (exact recovery of a
  linear field — test that a plane `z = a·x + b·y + c` is reproduced to `1e-10`); outside points →
  nodata; temporal support; no duplicate spatial index.

### B3 — Mesh↔mesh regridders (Overlap / RelativeOverlap-conservative / Barycentric / CentroidLocator)
- **Status:** ☐ Not started **· Depends on:** B2, and a §S.7 **dependency decision** **· Size:** XL
- **Objective:** area/overlap-weighted, first-order-conservative, barycentric, and centroid-locator
  regridding between meshes (and structured↔unstructured). This is the single biggest capability gap
  and the largest task.
- **Dependency decision (record before coding):** the overlap-area computation requires exact
  polygon–polygon intersection area. **Route 1 (recommended): add `numba_celltree` behind a new
  `[mesh]` optional extra** and call `celltree.intersect_faces(vertices, faces, fill_value)` → `(target_index, source_index, weights)` where `weights` is the clipped-intersection area (verified: xugrid
  `regrid/unstructured.py:109`). Do **not** hand-roll polygon clipping (§S.7). Route 2 (numpy-only) is
  acceptable **only** for CentroidLocator (point-in-face) and Barycentric-via-gridded-alphas; true
  conservative regridding needs Route 1.
- **Reference design (xugrid, verified):**
  - **Weight storage:** build COO triplets `(target_index, source_index, weight)`; convert to CSR
    (`core/sparse.py`, rows sorted by target). Row *t* = `indices[indptr[t]:indptr[t+1]]`. Persist/reuse
    weights (`from_weights`) — this is a first-class feature (weights computed once, applied to many
    time/layer slabs).
  - **Absolute vs relative:** RelativeOverlap divides each weight by the **source** face area
    (`weights /= source_area[source_index]`) → enables `first_order_conservative` (`Σ v·w`). Face area =
    shoelace (already in `mesh.face_areas`).
  - **Aggregation methods** (all NaN-aware, return NaN when `Σw==0`), from `regrid/reduce.py`:
    `mean = Σwv/Σw`, `sum`, `minimum`, `maximum`, `mode` (area-weighted, tie→largest value),
    `harmonic_mean`, `geometric_mean`, `max_overlap`, `first_order_conservative`, and
    percentiles/median (quickselect, ranks `1+(n-1)·p/100`).
  - **CentroidLocator:** `source_index = tree.locate_points(target_centroids)`; weight 1.0; pure copy
    (COO 1:1). Points outside → dropped.
  - **Barycentric:** build the centroidal-Voronoi tesselation of the source (see E4), then celltree
    barycentric weights of target centroids in the voronoi mesh, remapping voronoi vertices → source
    faces; exterior interpolated vertices redistributed to the two surrounding real nodes by
    inverse-distance (xugrid `regrid/unstructured.py:17` `replace_interpolated_weights`).
- **API sketch:** a `ugrid/regrid/` subpackage with `OverlapRegridder`, `RelativeOverlapRegridder`,
  `CentroidLocatorRegridder`, `BarycentricInterpolator`, each `__init__(source, target, ...)`, `.regrid(data_or_dataset)`, `.weights`, classmethod `.from_weights(weights, target)`. Facade:
  `UgridDataset.regrid(target, method=..., ) -> UgridDataset`.
- **Correctness traps:** overlap weights must be **true intersection areas** (Route 1); output pre-filled
  NaN, empty target rows stay NaN; conservative method must divide by source area (not target);
  broadcast over leading time/layer axes (loop the "extra" dims, apply the sparse matmul per slab); mode
  tie-break must be order-independent (largest value).
- **DoD:** the four regridders; reusable weights; conservative regridding **conserves the integral**
  (test: `Σ source_value·source_area ≈ Σ target_value·target_area` to `1e-8` on a refinement pair);
  overlap-mean of a constant field = that constant; outside targets = NaN; broadcasting over time. If
  Route 1 chosen, `numba_celltree` gated behind `[mesh]` extra with a clean `OptionalPackageDoesNotExist`
  when absent.

### B4 — Time interpolation between steps (`interp_time`)
- **Status:** ☐ Not started **· Depends on:** A1 **· Size:** S
- **Objective:** `interp_time(self, times, *, method="linear") -> UgridDataset` — interpolate temporal
  variables **between** existing time steps (today only exact `sel_time` selection exists). Mirrors
  gridded's `Time.interp_alpha` linear blending.
- **Implementation:** map requested `times` onto the source time coordinate (`time_values`), compute the
  bracketing indices and the linear alpha, `out = (1-α)·data[lo] + α·data[hi]` along the time axis;
  `method="nearest"` picks the closer step. Out-of-range → clamp or NaN (parameterize
  `bounds="nan"|"clamp"`, default `"nan"`).
- **Correctness traps:** operate on axis `time_index`; leave static/layered vars untouched; `time_values`
  may be a synthetic range fallback — require real time coords or raise.
- **DoD:** linear + nearest; midpoint interpolation of a linear-in-time field exact; out-of-range policy;
  static vars carried; test on a 3-step temporal var.

---

## Epic C — Topology dimensions: 1D networks, layers, multi-topology (Gap 1)

### C1 — Load all topologies in a multi-mesh file
- **Status:** ☐ Not started **· Depends on:** none **· Size:** M
- **Objective:** today `read_file` uses `topologies[0]` (`dataset.py:96` step 3), silently dropping
  extra meshes. `parse_ugrid_topology` (`io.py:100`) already returns **all** of them. Expose them.
- **Design options (pick the least disruptive):** keep `read_file` returning the **primary** mesh
  (backward-compatible), but add:
  - `UgridDataset.mesh_names -> list[str]` and a `read_file(..., mesh_name: str | None = None)` selector, or
  - a classmethod `read_all(path) -> dict[str, UgridDataset]` returning one dataset per topology.
  Prefer adding `mesh_name=` to `read_file` + a `read_all` helper; do **not** change the default return
  type of `read_file` (regression risk for every caller/test).
- **pyramids internals:** `_read_data_variables(rg, topo_info, path)` is already per-topology; call it
  per `MeshTopologyInfo`. `Mesh2d.from_gdal_group(rg, topo_info)` builds one mesh from one topo.
- **DoD:** multi-topology file exposes every mesh; `read_file` default unchanged (regression-safe);
  `read_all`/`mesh_name=` covered by a test using a 2-topology fixture (build one, or synthesize).

### C2 — Layer / vertical index selection
- **Status:** ☐ Not started **· Depends on:** A1 **· Size:** M
- **Objective:** give `(n_layers, n_face)` and `(n_time, n_layers, n_face)` variables a **layer
  selection** API (today the raw array is exposed with no accessor). This is generic **index** selection
  — **not** vertical-coordinate physics (that stays 🔴 out of scope, Gap 8).
- **API:** `isel_layer(self, index) -> UgridDataset` / `sel_layer(...)` selecting along the layer axis;
  and a `n_layers` property + `layer_values` (if a layer coordinate variable exists).
- **Implementation:** the layer axis is the non-time, non-last axis. Detect it: for a var, axes are
  `[time?(0), layer?(middle), element(-1)]`. If `var.data.ndim == 2 and not has_time` → axis 0 is layer;
  if `ndim == 3` → axis 1 is layer. Index that axis, update `dimensions`, `with_data`.
- **Correctness traps:** must not confuse layer with time (use `time_index`/`has_time`, §S.1 rule 2);
  update `dimensions` when a scalar layer index drops the axis (§S.3); carry non-layered vars through.
- **DoD:** `isel_layer`/`sel_layer`, `n_layers`; correct axis detection for 2-D layered and 3-D
  time+layer vars; `dimensions` consistency; test on a synthesized `(n_layers, n_face)` var.
  ⚠️ **Test-building trap (see §S.3):** `from_arrays` does **not** set `dimensions`, so a layered array
  passed through it is mis-flagged temporal. Either land the `from_arrays(dimensions=...)` enhancement
  first (recommended — this task is the natural place for it) and build the fixture with
  `dimensions={"salinity": ("nLayers","nFaces")}`, or construct the `MeshVariable(..., dimensions=("nLayers","nFaces"))`
  directly and assemble the `UgridDataset` by hand. Without a `dimensions` tuple, layer-vs-time detection
  is undefined for array-built vars — call that out in the docstring.

### C3 — 1D network mesh (`Network1d`)
- **Status:** ☐ Not started **· Depends on:** C1 **· Size:** XL
- **Objective:** support UGRID `topology_dimension == 1` (river networks / 1D hydraulic models): nodes +
  `edge_node_connectivity`, **no faces**. Today `read_file` always builds a `Mesh2d` and requires
  `face_node_connectivity` (`Mesh2d.from_gdal_group` raises without it) → 1D files can't be loaded at all.
- **Reference (xugrid `ugrid1d.py`):** the whole topology is `node_x`, `node_y`,
  `edge_node_connectivity (n_edge×2)`, `fill_value`. `core_dimension == edge`; no centroids/area/celltree
  faces; nearest via node/edge KDTree; `topology_subset` subsets edges + dedup/renumbers nodes.
- **Design:** add a `Network1d` class (sibling of `Mesh2d`) in a new `ugrid/network.py`, holding
  `node_x/node_y/edge_node_connectivity/crs` + `n_node`/`n_edge`/`bounds`/`get_edge_coords`/edge
  KDTree. Extend `io.parse_ugrid_topology` consumption: when `topo_info.topology_dimension == 1`, build a
  `Network1d`; `UgridDataset` must hold **either** a `Mesh2d` or a `Network1d` (introduce a small shared
  protocol or a `topology_dimension` discriminator). Data variables live on `node` or `edge` only.
  `to_geodataframe` → node points / edge LineStrings (edge path already exists via `_edge_linestrings`).
  `to_crs`, `sel_time`, `crop`(bbox+node/edge subset) should work; `to_dataset` (rasterize) uses node/edge
  source points.
- **Correctness traps:** don't break `Mesh2d`-only assumptions — audit every `self._mesh.face_*` access
  in `dataset.py` and guard on topology dimension; `n_face`/face properties should raise a clear error on
  a 1D dataset; keep `_wrap_subset` working for both. This is invasive — do it behind the
  `topology_dimension` discriminator and add a `TestNetwork1d` suite plus a 1D sample file fixture.
- **DoD:** load a 1D UGRID file into a `UgridDataset`; node/edge variables; `to_geodataframe`
  (points/lines), `to_crs`, time selection, bbox crop; `Mesh2d` path fully unregressed (all existing
  ugrid tests pass unchanged); new 1D test suite.

---

## Epic D — NaN fill across the mesh (Gap 3)

> Distinct from A5 (which fills along **time**). These fill missing values across mesh **space** using
> connectivity/geometry. Name them to avoid colliding with A5's `interpolate_na(dim="time")`.

### D1 — `fill_na_spatial` (nearest fill across the mesh)
- **Status:** ☐ Not started **· Depends on:** none **· Size:** S
- **Objective:** fill NaN/`nodata` elements from the nearest non-null element by geometric distance.
- **Reference (xugrid `ugrid2d.py:1619`, pure scipy):**
  ```
  coords = face_centroids (face vars) | node coords (node vars) | edge centers (edge vars)
  i_src = flatnonzero(~isnull); i_tgt = flatnonzero(isnull)
  tree = cKDTree(coords[i_src]); _, idx = tree.query(coords[i_tgt], distance_upper_bound=max_distance)
  keep = idx < len(i_src)                      # scipy returns len(i_src) when beyond max_distance
  out[i_tgt[keep]] = data[i_src[idx[keep]]]
  ```
- **API:** `fill_na_spatial(self, variable_name=None, *, max_distance=None) -> UgridDataset`.
- **pyramids internals:** reuse `MeshSpatialIndex` coordinate sources / `_get_source_data`
  (`interpolation.py`) for the location→coords mapping; cKDTree already imported in `spatial.py`.
- **Correctness traps:** define "null" as `isnan(data) | (data == var.nodata)`; broadcast over leading
  time/layer axes (fill each slab, or fill using per-slab null masks — decide and document: per-element
  across all steps vs per-step; recommend **per-slab** to match xugrid's apply-over-extra-dims);
  `max_distance=None` → unbounded.
- **DoD:** nearest fill for face/node/edge vars; `max_distance` cutoff leaves far gaps as NaN; temporal
  broadcasting; test that a single hole surrounded by a constant field fills to that constant.

### D2 — `laplace_interpolate`
- **Status:** ☐ Not started **· Depends on:** E1 (connected components), E6 (face_face / node_node
  connectivity + weights) **· Size:** L
- **Objective:** fill NaN cells by solving Laplace's equation with the known cells as Dirichlet boundary
  conditions — smooth interpolation over the mesh graph.
- **Reference (xugrid `ugrid/interpolate.py:207`, pure scipy.sparse):**
  1. `isnull/notnull`; error if all null; copy if none.
  2. Per connected component (`scipy.sparse.csgraph.connected_components`): a fully-null component stays
     NaN (avoids a singular system). `known = notnull & ~all_null`, `unknown = isnull & ~all_null`.
  3. Weight matrix `W` from the connectivity CSR (uniform `1.0` if `use_weights=False`, else
     inverse-distance normalized ~1.0 from centroid/node distances — see E6). Degree `D = W.sum(axis=1)`.
     Laplacian `L = diags(D) - W`.
  4. `A = L[unknown][:,unknown]`; `rhs = -L[unknown][:,known] · data[known]`.
  5. Jacobi diagonal scaling: `s = 1/sqrt(diag(A))`; `A_s = S A S`, `rhs_s = s·rhs` (conditioning).
  6. Solve: `direct_solve=True` → `scipy.sparse.linalg.spsolve`; else `scipy.sparse.linalg.cg`
     (`rtol/atol/maxiter`), optionally ILU-preconditioned; warn on non-convergence.
  7. `out[unknown] = s · x`.
- **API:** `laplace_interpolate(self, variable_name=None, *, use_weights=True, direct_solve=False, rtol=1e-5, atol=0.0, maxiter=500) -> UgridDataset`.
- **pyramids internals:** needs `face_face_connectivity` (derivable today via
  `Mesh2d.build_face_face_connectivity`, but it's a `Connectivity`, not a sparse adjacency — build a
  `scipy.sparse` matrix from it, see E1/E6). Node-data laplace uses node_node connectivity (E6). Edge
  laplace is forbidden (raise).
- **Correctness traps:** fully-null components must be skipped (singular matrix otherwise); the diagonal
  scaling is required for CG conditioning; keep the ILU preconditioner optional (plain CG works, just
  slower) — do **not** block the task on porting MODFLOW-6 ILU0; start with `spsolve`/plain CG and add
  ILU later. Broadcast over time/layer.
- **DoD:** laplace fill for face + node vars (edge raises); matches a known harmonic solution on a small
  mesh to `1e-6`; fully-null component stays NaN; direct + CG paths; time broadcasting.

---

## Epic E — Connectivity / graph / topology operations (Gap 4)

> All pure numpy/scipy — **no new dependency**. Shared prerequisite: a helper that turns a
> `Connectivity` (face_face or node_node) into a `scipy.sparse` adjacency matrix. Add it once in E1/E6
> and reuse in D2/E2/E3.

### E1 — `connected_components`
- **Status:** ☐ Not started **· Depends on:** E6 (needs a sparse `face_face` adjacency) **· Size:** S
- **Objective:** `connected_components(self) -> np.ndarray` (per-face component label) — mesh analog of
  the raster `Dataset.cluster`.
- **Reference (xugrid):** `scipy.sparse.csgraph.connected_components(face_face_adjacency)` → `(n, labels)`.
- **pyramids internals:** `Mesh2d.build_face_face_connectivity()` exists but yields a `Connectivity`
  (ragged neighbor index array), not a sparse matrix. Build the CSR adjacency from
  `edge_face_connectivity` (xugrid `connectivity.py:487`: keep edges with both faces valid, symmetric COO
  `(edge_index,(ij,ji))`, → CSR) — put this in a shared `ugrid/graph.py` `face_adjacency_matrix(mesh)`.
- **DoD:** returns per-face labels; two disconnected triangles → labels `[0,1]`; a connected mesh → all
  zeros; test on `mixed_mesh` and a 2-component fixture.

### E2 — `reverse_cuthill_mckee`
- **Status:** ☐ Not started **· Depends on:** E1 **· Size:** S
- **Objective:** `reverse_cuthill_mckee(self) -> UgridDataset` — reorder nodes/faces to reduce matrix
  bandwidth (useful before solvers). `scipy.sparse.csgraph.reverse_cuthill_mckee(adjacency)` → permutation;
  reindex mesh + data by it.
- **Correctness traps:** reindexing must renumber `face_node_connectivity`, node coords, **and** every
  face-located variable consistently — reuse the renumber logic from
  `spatial.py:_subset_mesh_by_face_indices` (searchsorted remap) rather than re-deriving; this is a
  permutation (not a subset) so all elements are kept.
- **DoD:** returns a congruent, reordered `UgridDataset`; data values follow their faces; round-trip
  (apply permutation, invert) recovers the original ordering; test on a small mesh.

### E3 — `binary_dilation` / `binary_erosion`
- **Status:** ☐ Not started **· Depends on:** E1 **· Size:** M
- **Objective:** morphology on boolean **face** masks across shared edges (grow/shrink True regions).
- **Reference (xugrid `connectivity.py:804` `_binary_iterate`, NOT scipy.ndimage):** from
  `face_face_connectivity.tocoo()` take `(i,j)`; `_mutate`: wherever `output[i] != output[j]` set both to
  `value` (True=dilation, False=erosion); reapply the fixed-cell `mask` each pass; first iteration also
  handles the exterior boundary (`exterior_faces`, see E5) when `value == border_value`; repeat
  `iterations`.
- **API:** `binary_dilation(self, variable_name, *, iterations=1, border_value=False) -> UgridDataset`
  and `binary_erosion(...)`.
- **DoD:** dilation grows / erosion shrinks a True region across edge neighbors by `iterations` rings;
  border handling; test on a hand-built adjacency with a known result.

### E4 — Voronoi tesselation + circumcenters
- **Status:** ☐ Not started **· Depends on:** E6 (node_face connectivity) **· Size:** L
- **Objective:** `voronoi_topology()` (centroidal Voronoi dual) and `circumcenters` (triangular meshes).
  Needed as a building block for B3's `BarycentricInterpolator` and useful standalone for plotting/duals.
- **Reference (xugrid `ugrid/voronoi.py`, pure numpy/scipy — portable):** interior: per node, gather
  centroids of connected faces (via `node_face_connectivity` CSR), order CCW by `arctan2(dy,dx)` around
  the node, group+order with `np.lexsort((angle, node_i))`. Exterior (the hard part,
  `exterior_topology`): orthogonally project face centroids onto exterior edges
  (`_project_centroid_on_edge`), add projected vertices; with `add_vertices`, insert linear interps to
  keep polygons convex; `skip_concave` swaps in the true vertex only when it enlarges the area.
- **Correctness traps:** exterior handling is subtle — port `voronoi.py` closely; test interior tesselation
  first, then exterior. `circumcenters` only defined for triangular faces (guard via `is_triangular`).
- **DoD:** interior Voronoi dual of a regular triangular mesh matches expected cells; circumcenters of a
  known triangle equal the analytic value; exterior handling covered by a boundary-node test.

### E5 — Exterior edges/faces + `bounding_polygon`
- **Status:** ☐ Not started **· Depends on:** E6 (edge_face connectivity) **· Size:** M
- **Objective:** `exterior_edges` / `exterior_faces` (boundary of the mesh) and `bounding_polygon()` (the
  dissolved outline, possibly with holes) — today `bounds` only gives a bbox.
- **Reference (xugrid):** exterior edges = edges whose `edge_face_connectivity` has a fill on one side
  (only one adjacent face). `bounding_polygon` = dissolve the boundary edges into polygon(s) via
  `shapely.polygonize` of the exterior edge linestrings, taking the largest-area ring(s).
- **pyramids internals:** needs `edge_face_connectivity` (read if present, else derive — E6). Reuse
  `_edge_linestrings` (`dataset.py`) for edge geometry; shapely is already a hard dep.
- **DoD:** `exterior_edges`/`exterior_faces` correct on `mixed_mesh`; `bounding_polygon` returns a shapely
  (Multi)Polygon whose area equals the mesh footprint; a mesh with a hole yields an interior ring.

### E6 — Extra connectivity builders + facet remapping
- **Status:** ☐ Not started **· Depends on:** none (foundational — do early; D2/E1/E4/E5 build on it)
  **· Size:** M
- **Objective:** derive the connectivity arrays pyramids doesn't build today and add facet remapping:
  - **builders:** `node_face_connectivity`, `node_node_connectivity`, `edge_face_connectivity`
    (pyramids has `face_face` + `edge_node` derivation only). Plus a shared
    `graph.py:sparse_adjacency(connectivity)` used by D2/E1/E2/E3.
  - **facet remap:** `to_node()` / `to_edge()` / `to_face()` — move data between locations by averaging
    contributing source elements (xugrid `.ugrid.to_node/to_edge/to_face`).
- **Reference (xugrid `ugrid/connectivity.py`):** `node_face_connectivity` = inverse of
  `face_node_connectivity` (CSR: for each node, the faces containing it). `edge_face_connectivity` =
  inverse of `face_edge_connectivity` (each edge → its ≤2 faces; fill = exterior). `node_node` from edges.
- **pyramids internals:** add as lazy cached properties on `Mesh2d` (mirror `_cached_face_centroids`
  pattern). `edge_face` requires `face_edge_connectivity` or `edge_node_connectivity` — derive edges first
  via existing `build_edge_connectivity()`.
- **Correctness traps:** fill/`-1` handling exactly as `Connectivity` (0-indexed internal, -1 fill);
  CSR (ragged) representations for node_face since node degree varies; facet remap must average only
  **valid** (non-fill) contributors and honor `location`.
- **DoD:** the three builders (validated against a hand-built mesh where the answer is known); shared
  `sparse_adjacency`; `to_node`/`to_edge`/`to_face` preserve totals sensibly (test face→node→face
  round-trip on a constant field returns the constant); fill-safe.

---

## Epic F — Mesh ↔ vector richness (Gap 5)

> pyramids has these on the **raster** side; port the mesh equivalents. F1 is pure shapely (no new dep);
> F3 needs the cell-tree/earcut engine (§S.7).

### F1 — `polygonize` (merge connected equal-value faces → polygons)
- **Status:** ☐ Not started **· Depends on:** E6 (edge_face connectivity) **· Size:** M
- **Objective:** `polygonize(self, variable_name) -> FeatureCollection` — dissolve adjacent faces with
  the same value into polygons (raster `to_feature_collection` analog for meshes).
- **Reference (xugrid `ugrid/polygonize.py`, pure numpy/scipy + shapely):**
  1. drop NaN faces; `i, j = edge_face_connectivity.T`.
  2. keep edges where both faces exist **and** `values[i] == values[j]`; symmetric COO with explicit
     `shape=(n,n)` (so isolated faces still get a label) → `connected_components` → `polygon_id`.
  3. boundary edges = where the two faces have different `polygon_id` (or a fill on one side).
  4. per label: `shapely.polygonize(shapely.linestrings(coords[boundary_edges]))`, take the largest-bbox
     geom (outer body; holes appear too). Value = the faces' shared value.
- **pyramids internals:** return a pyramids `FeatureCollection` (as `to_feature_collection` does), CRS set
  from `self.crs`. Reuse `edge_face_connectivity` from E6 and `_edge_linestrings` geometry.
- **Correctness traps:** the explicit COO `shape=(n,n)` is required or isolated faces are lost; intended
  for **few unique values** (document); float values need exact equality (document; suggest rounding/
  classifying first).
- **DoD:** a face field with 2–3 distinct values → correct dissolved polygons with the right attribute;
  isolated face preserved; CRS set; test on `unit_square_dataset` with a 2-value field.

### F2 — Transect / line extraction (`sample_line`)
- **Status:** ☐ Not started **· Depends on:** B2 **· Size:** M
- **Objective:** `sample_line(self, start, end, variable_name, *, n=None, step=None)` or
  `sample_linestring(self, linestring, ...)` — sample face values along a line (cross-sections).
- **Light-weight implementation (no cell tree):** densify the line into points (by `n` samples or `step`
  spacing), then reuse B2 `sample(x, y, method="nearest")` at those points; return values + distances
  along the line (and optionally the crossed face indices). This avoids the exact edge-intersection that
  xugrid's `intersect_line` does via celltree — acceptable and clearly documented as sampling, not exact
  segment clipping. (An exact `intersect_edges` version can be a later enhancement under the `[mesh]`
  extra.)
- **DoD:** returns per-sample `(distance, value)` along a line; endpoints inside the mesh sampled
  correctly; points off-mesh → nodata; test that a linear field sampled along a line is monotonic/linear.

### F3 — `burn` vector geometry onto faces
- **Status:** ☐ Not started **· Depends on:** §S.7 decision (points need only `locate_faces`; lines/
  polygons need cell-tree/earcut) **· Size:** L
- **Objective:** `burn(self, geometry, *, column=None, all_touched=False) -> UgridDataset` — rasterize
  vector features onto mesh faces (inverse of `polygonize`).
- **Reference (xugrid `ugrid/burn.py`):** points → `locate_points` containment → assign; lines → split
  into 2-point segments, `intersect_edges` → assign crossed faces; polygons → earcut-triangulate then
  `locate_faces` of the triangles (`all_touched`), or centroid-in-triangle when `all_touched=False`.
- **Staging:** v1 = **points only** (pure numpy via existing `locate_faces` — no new dep). v2 =
  lines/polygons behind the `[mesh]` extra (numba_celltree + mapbox_earcut). Split accordingly so the
  no-dep points path ships first.
- **DoD (v1):** burning point features with a value column sets the containing faces; others = fill;
  test on `unit_square_dataset`. (v2 gated behind `[mesh]`.)

---

## Epic G — Mesh construction from other data (Gap 6)

### G1 — `from_dataset` / `from_structured` (raster / curvilinear → mesh)
- **Status:** ☐ Not started **· Depends on:** none **· Size:** M
- **Objective:** build a `UgridDataset` (quad mesh) from a pyramids `Dataset`/`NetCDF` (structured grid),
  flattening cells into faces+nodes — a natural bridge for a GDAL-first library.
- **Reference (xugrid `ugrid2d.py:1894` `_from_intervals_helper`, pure numpy):** for a rectilinear grid,
  nodes = meshgrid of cell-edge coordinates (interval breaks from centers via `infer_interval_breaks1d`);
  faces = quads over the `(ny+1)×(nx+1)` node lattice with a **linear index** trick; **reverse the
  left/right or lower/upper slices when a coordinate is descending** to keep faces counter-clockwise:
  ```
  linear_index = arange((ny+1)*(nx+1)).reshape(ny+1, nx+1)
  # left,right = (:-1),(1:) ; lower,upper = (:-1),(1:) ; swap if node_x/ node_y descending
  face_nodes[:,0]=linear_index[lower,left]; [:,1]=[lower,right]; [:,2]=[upper,right]; [:,3]=[upper,left]
  ```
  No dedup needed for a regular lattice. Curvilinear/2-D-coord grids use `bounds2d_to_topology2d`
  (`conversion.py:297`) which **does** dedup (`np.unique(..., axis=0, return_inverse=True)`), drops
  degenerate/NaN cells, and CCW-sorts corners by `arctan2` about the centroid.
- **pyramids internals:** read the source `Dataset` cell-center coords / geotransform (see B1 accessors);
  produce `node_x/node_y/face_node_connectivity` + one face variable per band, then
  `UgridDataset.from_arrays(...)`. CRS carried from the source `Dataset.epsg`.
- **Correctness traps:** the CCW orientation reversal for descending axes is mandatory (pyramids rasters
  are typically north-up so `y` is descending — you **will** hit this); inactive/nodata cells should
  become dropped faces (curvilinear path) or nodata face values (rectilinear path) — decide and document.
- **DoD:** a small `Dataset` → `UgridDataset` with `n_face == rows*cols` (minus dropped nodata cells),
  faces CCW (positive `face_areas`), values matching the raster; CRS preserved; test round-tripping a 3×3
  raster and checking `rasterize_like` back onto the source grid recovers the values.

### G2 — `from_geodataframe` (polygons → mesh)
- **Status:** ☐ Not started **· Depends on:** none **· Size:** M
- **Objective:** `from_geodataframe(gdf) -> UgridDataset` — build a face mesh from polygon geometries
  (inverse of `to_geodataframe`).
- **Reference (xugrid `conversion.py`):** collect polygon exterior rings, dedup vertices via
  `np.unique(coords, axis=0, return_inverse=True)`, build `face_node_connectivity` from the inverse index
  (pad ragged rings with `-1` fill to `max_nodes`), CCW-sort, drop degenerate faces; attribute columns →
  face variables.
- **Correctness traps:** ragged faces (mixed polygon sizes) need `-1` fill and correct `max_nodes`;
  vertex dedup tolerance (xugrid assumes exact shared vertices — document); CRS from `gdf.crs`.
- **DoD:** a GeoDataFrame of polygons → mesh whose faces match the polygons; ragged (tri+quad) handled;
  attributes become face vars; CRS carried; round-trip `to_geodataframe` recovers geometries.

---

## Epic H — Interop: xarray, Zarr, lazy dask (Gap 9)

### H1 — `to_xarray` / `from_xarray`
- **Status:** ☐ Not started **· Depends on:** none **· Size:** L
- **Objective:** hand a mesh off to xarray and back, mirroring `NetCDF.to_xarray`/`from_xarray`. For real
  ecosystem value, **encode the topology per UGRID conventions** so the resulting `xarray.Dataset` is
  readable by **xugrid** (`xugrid.UgridDataset.from_dataset`) — i.e. a plain `xr.Dataset` carrying the
  mesh topology variable + connectivity + node coords as standard variables, plus the data variables on
  their location dims.
- **Reference (pyramids `Interop`, `netcdf/engines/interop.py`):** `to_xarray` builds
  `xr.Dataset(data_vars=..., coords=..., attrs=ds.global_attributes)` then promotes CF non-data arrays;
  `from_xarray` builds a multidim source and reads it back. Guard-import xarray via
  `import_xarray(_XARRAY_HINT.format(func="to_xarray"))` (`_utils.py:1493`) — **no top-level import**.
- **Implementation:**
  - `to_xarray(self) -> xr.Dataset`: create a dataset with dims `{mesh_nNodes, mesh_nFaces, mesh_nEdges?}`;
    add `node_x/node_y` (1-D on nodes, CF `standard_name`/`units`), `face_node_connectivity` (2-D
    `(nFaces, nMaxFaceNodes)`, with `start_index`/`_FillValue` restored to the on-disk convention — reuse
    the write-side reverse mapping in `io.py:_write_connectivity_array`), a scalar `mesh` topology var
    with the UGRID attrs (`cf_role=mesh_topology`, `topology_dimension`, `node_coordinates`,
    `face_node_connectivity`, …), each `MeshVariable` as a data var with `mesh`/`location` attrs, a CRS
    grid-mapping var, and `time` coord when present. This is essentially the `to_file` variable set
    (`io.py:write_ugrid_topology`/`write_ugrid_data_variable`) emitted as xarray objects instead of GDAL
    MDArrays — **factor the shared "what variables/attrs define the UGRID encoding" logic** so `to_file`
    and `to_xarray` can't drift.
  - `from_xarray(cls, dataset) -> UgridDataset`: read the topology var + connectivity + coords back
    (reuse `io.parse_ugrid_topology`-equivalent logic adapted for xarray, or write to a temp `.nc` and
    call `read_file` — the pragmatic first cut, exactly how `NetCDF.from_xarray` round-trips through a
    temp file at `interop.py:1395`).
- **Correctness traps:** connectivity must be written back with the **original `start_index` and file
  fill** (not the internal 0-index/-1) so xugrid/other readers interpret it correctly — this is the
  single most likely bug; reuse `_write_connectivity_array`'s reverse logic. Preserve `location`
  attributes so xugrid knows node/face/edge. xarray optional (guarded).
- **DoD:** `to_xarray` produces a dataset that xugrid `from_dataset` reads back to an equivalent mesh
  (assert node/face counts, connectivity, a data var round-trip); `from_xarray` inverts it; connectivity
  start_index/fill correct; xarray-missing → `OptionalPackageDoesNotExist`. Add a test **skipped when
  xugrid isn't installed** for the cross-package round-trip.

### H2 — `to_zarr` / `from_zarr`
- **Status:** ☐ Not started **· Depends on:** H1 **· Size:** M
- **Objective:** write/read the mesh as UGRID-encoded Zarr (today `to_file` is NetCDF-only). Simplest
  correct route: `to_zarr` = `to_xarray().to_zarr(store)` (via the guarded xarray/zarr import); `from_zarr`
  = `from_xarray(xr.open_zarr(store))`. Zarr is in the `[lazy]` extra.
- **DoD:** round-trip mesh + variables through a Zarr store; guarded deps; test skipped when zarr absent.

### H3 — Lazy dask `chunks=` path for mesh variables
- **Status:** ☐ Not started **· Depends on:** none **· Size:** L
- **Objective:** let mesh variables be dask-backed for out-of-core reads, mirroring
  `NetCDF.read_array(chunks=...)` / `build_lazy_array`. Today mesh vars are lazy-**loaded** (whole array
  on first `.data`) but never chunked.
- **Reference (pyramids `_lazy.py:build_lazy_array:595`):** wraps a `CachingFileManager` around a GDAL
  MDIM open and `dask.array.map_blocks` over a block grid, each chunk read via
  `md_arr.ReadAsArray(array_start_idx=starts, count=counts)`. `import_dask` guarded; `[lazy]` extra.
- **Implementation:** add `read_file(path, *, chunks=...)` and/or a per-variable
  `MeshVariable`-with-dask-loader: when `chunks` is set, the loader returns a `dask.array` built by
  `build_lazy_array(path, var_name, chunks, ...)` instead of an eager `ReadAsArray()`. The element axis
  chunking should default to native block size; time axis chunkable for windowed workflows.
- **Correctness traps:** most Epic-A/D ops call `np.nan*` — verify they work on dask arrays or add a
  `.compute()` boundary; keep the eager default (chunks=None) unchanged (regression); the existing
  time-slab windowing (`_read_time_slab`) is a different optimization — don't double-read.
- **DoD:** `chunks=` yields dask-backed `MeshVariable.data`; a reduction over such a var computes
  correctly; eager path unchanged; dask-missing → `OptionalPackageDoesNotExist`; test with a chunked read
  of the sample file.

---

## Epic I — Format breadth via MDAL (Gap 7)

### I1 — `from_mdal` (2DM / Selafin / DFSU / XMDF / … readers)
- **Status:** ☐ Not started **· Depends on:** none **· Size:** L
- **Objective:** read the 20+ hydrodynamic mesh formats MDAL supports (2DM, Selafin/Telemac, DFSU/DFS2,
  XMDF, FLO-2D, SWW, DAT, TIN, PLY, …) into a `UgridDataset`. Format reading is **explicitly in scope**
  (SCOPE.md: "reading a format is never out of scope"). Deliver by **wrapping `mdal-python`**, not by
  writing parsers.
- **Reference (mdal-python, ViRGIS-Team):** `Datasource(path).load(index) -> Mesh`; `Mesh.vertices`,
  `.faces`, `.edges` (numpy), `.projection`, `.extent`; `DatasetGroup` (`.location` ∈
  vertices/faces/edges/volumes, `.is_temporal`, `.data(i)` numpy, `.dataset_time(i)`). Map: vertices →
  `node_x/node_y`, faces → `face_node_connectivity` (watch MDAL's face-vertex layout & fill), dataset
  groups → `MeshVariable`s with `location` = node(vertices)/face/edge, temporal groups stacked to
  `(n_time, n_elem)`, projection → CRS.
- **Dependency:** `mdal-python` behind a new `[mdal]` optional extra; guarded import → clean error.
- **Correctness traps:** MDAL face-vertex arrays may be 1-indexed / padded differently than pyramids'
  `Connectivity` (normalize to 0-index/-1 fill exactly as `Connectivity.from_gdal_array` does); volume
  (3-D) datasets map onto the layer model (C2) — or raise "not yet supported" cleanly; MDAL only carries
  a CRS **string** (no reprojection) — set `crs_wkt`, don't transform.
- **DoD:** read at least one non-UGRID format (e.g. a 2DM + DAT pair, or a Selafin) into a `UgridDataset`
  with correct nodes/faces/variables/CRS; `[mdal]` extra + guarded import; test skipped when mdal-python
  absent, using a small sample mesh committed under `tests/data/`.

---

## Explicitly out of scope — do NOT implement in pyramids core (Gap 8)

Per `docs/SCOPE.md`, these interpret domain value-semantics and belong in a downstream domain package
(e.g. `earthlens`), reusing pyramids' generic bridges. If a task above tempts you toward them, stop and
add a note instead of code:

- **Vertical-coordinate physics** — ROMS/FVCOM sigma transforms, z-level depth interpolation (gridded's
  `S_Depth`/`ROMS_Depth`/`FVCOM_Depth`). Generic **layer indexing** (C2) is fine; interpreting a sigma
  coordinate into physical depth is not.
- **Full mesh generation** via meshkernel (orthogonalization, refinement, remeshing) — a modelling
  concern, not a GDAL-style primitive.
- **Model-specific grid recognizers** (the removed `grids/` HEALPix/ORCA/octahedral) — precedent S2.
- **Rendering/analysis MDAL omits** (contouring, streamlines) — pyramids delegates viz to cleopatra.

---

## Dependency map & suggested sequencing

**New optional extras introduced (keep them optional + guarded):**

| Extra | Packages | Needed by |
|-------|----------|-----------|
| `[mesh]` | `numba_celltree` (+ `mapbox_earcut` for polygon burn) | B3 (overlap/conservative regridding), F3 v2 (line/polygon burn) |
| `[lazy]` (exists) | `dask`, `zarr` | H2, H3 |
| interop (xarray) | `xarray` (already an optional interop dep on the NetCDF side) | H1, H2 |
| `[mdal]` | `mdal-python` | I1 |

Everything else (Epic A, B1/B2/B4, C1/C2, D1/D2, all of E, F1/F2, G1/G2) is **pure numpy/scipy/shapely —
no new dependency**, and should be prioritized.

**Foundational tasks to do first (unblock others):**
- **E6** (connectivity builders + `sparse_adjacency`) → unblocks D2, E1, E2, E3, E4, E5, F1.
- **A1** (axis-resolution helper + selection) → unblocks A2–A7, B4, C2.
- **B2** (point sampling) → unblocks F2 and the light path of B3/F3.

**Suggested milestones:**
1. **M1 — xarray parity (Epic A) + E6.** Highest user-requested value, all pure-numpy, mirrors proven
   NetCDF engines. Ship A1–A5 + A8 first, A6/A7 next.
2. **M2 — mesh fill & graph ops:** D1, D2, E1–E5. Pure scipy; high analytical value.
3. **M3 — sampling & light regridding:** B1, B2, B4, F1, F2; G1, G2. No new deps.
4. **M4 — interop:** H1, H2, H3. Unlocks the whole xarray/dask/zarr ecosystem for meshes.
5. **M5 — heavy regridding & formats (opt-in deps):** B3, F3 v2 (`[mesh]`), C3 (1D networks), I1 (`[mdal]`).

**Cross-cutting regression guardrails (verify after every task):**
- All existing `tests/ugrid/` pass unchanged — especially `TestNoImportCycle` (spatial subsetters return
  tuples, never `UgridDataset`) and `test_spatial_return_contract` (node/edge renumber + slice).
- `read_file`'s default return contract is unchanged (C1/H3 must not alter it).
- Every derived op preserves `location`/`units`/`nodata`/`standard_name`, keeps the element axis last,
  keeps `dimensions` consistent with `data.ndim` (§S.3), and never mutates the input dataset.
- Connectivity written to disk/xarray uses the **original** `start_index`/fill, never the internal
  0-index/-1 (H1 trap).

---

*Part 2 authored from three code-verified deep-dives: pyramids UGRID internals
(`src/pyramids/netcdf/ugrid/`), pyramids NetCDF facade/engine patterns
(`src/pyramids/netcdf/engines/`, `interop.py`, `_lazy.py`), and xugrid v0.15.3 + gridded reference
algorithms. Every pyramids symbol/line was read from the tree; every reference algorithm was read from
the named upstream source file. Refine line numbers by symbol name if they drift.*



