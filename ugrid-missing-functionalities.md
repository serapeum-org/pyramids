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
