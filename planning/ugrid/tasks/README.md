# UGRID implementation — standalone task briefs

One file per task. **Each `<ID>.md` is self-contained**: it carries the task's objective, API, reference
algorithm, pyramids internals, steps, correctness traps, Definition of Done, and tests, followed by the
**Common context** appendix (the shared §S.1–S.8 prerequisites) inlined verbatim. Hand a single brief to
an implementing agent and it needs nothing else in the repo to start.

- **Master plan / rationale:** `../ugrid-missing-functionalities.md` (Part 1 = gap analysis vs
  xugrid/xarray/gridded/MDAL; Part 2 = the same tasks in consolidated form + dependency map + milestones).
- **Shared context source of truth:** `_common-context.md` (inlined into every brief; edit there and
  regenerate briefs to keep them in lockstep).
- **Scope guard:** re-check every task against `../../../docs/SCOPE.md`.

## Suggested order (dependencies)

Foundational first: **E6** (connectivity builders + `sparse_adjacency`) unblocks D2/E1/E2/E3/E4/E5/F1;
**A1** unblocks A2–A7/B4/C2; **B2** unblocks F2 and the light path of B3/F3.

Milestones (from the master plan): **M1** xarray-parity (A1–A8) + E6 · **M2** mesh fill & graph
(D1,D2,E1–E5) · **M3** sampling & light regridding (B1,B2,B4,F1,F2,G1,G2) · **M4** interop (H1,H2,H3) ·
**M5** heavy/opt-in (B3,F3-v2,C3,I1).

## Index

| Id | Task | Epic | Gap | Priority | Size |
|----|------|------|-----|----------|------|
| [A1](A1.md) | `isel`/`sel` (element + time selection) | A xarray-parity | 9 | High | M |
| [A2](A2.md) | `reduce` (time & spatial, grouped) | A | 9 | High | M |
| [A3](A3.md) | `rolling` (over time) | A | 9 | Med | S |
| [A4](A4.md) | `weighted` (area-weighted spatial mean) | A | 9 | High | M |
| [A5](A5.md) | `ffill`/`bfill`/`dropna`/`interpolate_na` (time) | A | 9 | Med | M |
| [A6](A6.md) | `concat`/`merge` | A | 9 | Med | M |
| [A7](A7.md) | `diff`/`cumsum`/`cumprod`/`shift`/`argmin`/`argmax`/`squeeze` (time) | A | 9 | Low | M |
| [A8](A8.md) | `stats` (per-variable) | A | 9 | Low | S |
| [B1](B1.md) | `rasterize_like` + method additions | B regridding | 2 | High | S |
| [B2](B2.md) | Point-value sampling (`sample`) | B | 2 | High | M |
| [B3](B3.md) | Mesh↔mesh regridders | B | 2 | High | XL |
| [B4](B4.md) | Time interpolation (`interp_time`) | B | 2 | Med | S |
| [C1](C1.md) | Load all topologies | C topology | 1 | Med | M |
| [C2](C2.md) | Layer / vertical index selection | C | 1 | Med | M |
| [C3](C3.md) | 1D network mesh (`Network1d`) | C | 1 | High | XL |
| [D1](D1.md) | `fill_na_spatial` (nearest fill) | D fill | 3 | High | S |
| [D2](D2.md) | `laplace_interpolate` | D | 3 | Med | L |
| [E1](E1.md) | `connected_components` | E graph | 4 | Med | S |
| [E2](E2.md) | `reverse_cuthill_mckee` | E | 4 | Low | S |
| [E3](E3.md) | `binary_dilation`/`binary_erosion` | E | 4 | Low | M |
| [E4](E4.md) | Voronoi tesselation + circumcenters | E | 4 | Low | L |
| [E5](E5.md) | Exterior edges/faces + `bounding_polygon` | E | 4 | Med | M |
| [E6](E6.md) | Connectivity builders + facet remap | E | 4 | Med | M |
| [F1](F1.md) | `polygonize` | F vector | 5 | Med | M |
| [F2](F2.md) | Transect / line extraction (`sample_line`) | F | 5 | Med | M |
| [F3](F3.md) | `burn` vector geometry onto faces | F | 5 | Low | L |
| [G1](G1.md) | `from_dataset`/`from_structured` (raster→mesh) | G construct | 6 | Med | M |
| [G2](G2.md) | `from_geodataframe` (polygons→mesh) | G | 6 | Low | M |
| [H1](H1.md) | `to_xarray`/`from_xarray` | H interop | 9 | High | L |
| [H2](H2.md) | `to_zarr`/`from_zarr` | H | 9 | Med | M |
| [H3](H3.md) | Lazy dask `chunks=` path | H | 9 | Med | L |
| [I1](I1.md) | `from_mdal` (2DM/Selafin/DFSU/…) | I formats | 7 | Low | L |

_Out of scope (not tasks; see master plan Gap 8): sigma/vertical-coordinate physics, meshkernel mesh
generation, MDAL rendering/analysis._
