# Scope

This page defines what **pyramids** is for, what it deliberately leaves out, and the boundary for every
module in the package. It is the reference used to decide whether a proposed feature belongs in pyramids
or in a downstream package. It was distilled from the 2026-06 out-of-scope audit
(`planning/architecture-review/`), which classified every public surface against a single rule.

## 1. What pyramids is

pyramids is a **generic GDAL/OGR-backed geospatial data library**. It reads, writes, and transforms:

- **raster** data (GeoTIFF, COG, NetCDF, GRIB, Zarr, HDF, …) via `Dataset` / `NetCDF`,
- **vector** data (Shapefile, GeoJSON, GeoParquet, FlatGeobuf, …) via `FeatureCollection`,
- **multi-temporal datacubes** of co-registered rasters via `DatasetCollection`,

plus the spatial primitives that operate on them (reproject, crop, align, mosaic, resample, rasterize,
vectorize, zonal stats, focal/terrain analysis, interpolation) and the cloud/lazy I/O needed to do that
at scale.

Its peer group is **rasterio / rioxarray / fiona / geopandas** — the general-purpose Python geospatial
stack. pyramids competes in that niche: it is the GDAL-backed engine, not a domain application built on
top of one.

**Guiding principle:** pyramids owns *generic geospatial primitives and format support*. Anything whose
correctness depends on knowing the scientific domain, the sensor, the data provider, or the downstream
workflow belongs in a package that has that knowledge.

## 2. What pyramids is not

pyramids is **not**:

- a **science-domain library** — it does not encode the value semantics of a discipline (atmospheric
  physics, oceanography, hydrology unit systems, radiometric models). It moves and reshapes pixels; it
  does not decide what a pixel *means*.
- a **provider/EO SDK** — it does not hardcode specific Earth-observation catalogs, agency auth
  endpoints, or per-mission product layouts.
- a **plotting/cartography product** — it has a thin, optional viz layer that delegates to
  [cleopatra](https://github.com/serapeum-org/cleopatra); it is not a map-styling or basemap-data host.
- a **workflow/orchestration framework** — it provides primitives that downstream pipelines compose; it
  does not own end-to-end, opinionated pipelines for a particular consumer.

When a capability matches one of these, it is relocated to the package that owns the domain (e.g.
`earthlens` for EO, `cleopatra` for viz) rather than kept in pyramids.

## 3. The scope test

Every candidate surface is classified against one question:

> **Is this a generic GIS primitive (or support for a data format), or is it domain value-semantics,
> downstream-consumer orchestration, or a specialized non-GIS standard?**

This yields four buckets:

- **Generic primitive — in scope.** A spatial/array/IO operation any GIS user needs, parameterised the
  same way GDAL/rasterio expose it: reproject, crop, mosaic, resample, rasterize, vectorize, zonal
  stats, terrain analysis with cartographic defaults.
- **Format support — in scope.** Decoding/encoding a data format and passing through its *standard*
  metadata (GeoTIFF tags, NetCDF/CF attributes, GRIB WMO keys, STAC items). Reading a format is never
  out of scope, even an exotic one.
- **Domain value-semantics — out of scope.** Logic that interprets *what a value means* in a discipline
  — unit conversions tied to physics, sensor-specific band orders, model-specific variable naming,
  agency-specific auth. Relocate to the domain package, or keep behind an explicit domain extra.
- **Downstream orchestration — out of scope.** Opinionated, end-to-end convenience for one
  consumer/workflow that composes primitives but encodes a specific use case. It belongs in the
  consumer.

Two clarifying rules that resolve most edge cases:

- **Format support counts as in scope.** Recognising and decoding a file format — however
  domain-specific the format is — is the library's job. The line is between *decoding a format* (in)
  and *interpreting domain values once decoded* (out).
- **A generic mechanism reused by a domain feature stays; only the domain-specific part moves.** The
  scattered-point / mesh→grid interpolation bridge and the cloud-signing `Signer` protocol are generic
  and stay; the exotic-grid *recognisers* and the provider-specific *signers* that used them moved out.

## 4. Module-by-module scope

Source layout: `src/pyramids/`. Each module lists its **in-scope** responsibilities and any **explicit
exclusions** (with the audit finding ID where the boundary was drawn — see §5).

### `base/` — infrastructure

GDAL/OGR initialisation and the cross-cutting primitives every other module builds on.

- **In scope:** GDAL/OGR/config bootstrap (`config.py`); dtype and driver catalogs and conversions
  (`_utils.py`); the custom exception hierarchy (`_errors.py`); CRS helpers (`crs.py`); typing
  protocols (`protocols.py`); raster-metadata value objects (`_raster_meta.py`); file/lock/artifact
  management; and the **generic cloud I/O primitives** in `remote.py` — the `s3://`/`gs://`/`az://` →
  GDAL `/vsi*/` rewriter and the `CloudConfig` credential context manager for AWS/GS/Azure.
- **Out of scope:** none. This is provider-agnostic plumbing. Cloud *auth schemes* that hardcode a
  specific EO catalog do not live here (they would be EO-domain — see `stac/`).

### `_io.py` and `io/` — local & archived I/O

- **In scope:** compressed/archived raster handling through the GDAL virtual filesystem
  (`/vsizip/`, `/vsigzip/`, `/vsitar/`), URL-scheme rewriting, and format sniffing (`io/sniff.py`).
  This is format/transport support.
- **Out of scope:** none.

### `dataset/` — single-raster core

`Dataset` (and its `AbstractDataset` base) plus the spatial engines and operations.

- **In scope:** read/write/array-I/O; no-data handling; **spatial operations** (crop, reproject, align,
  mosaic via `merge.py`, resample); band-level operations and band metadata (`engines/bands.py`);
  **analysis** (`engines/analysis.py`); **vectorize** (`engines/vectorize.py` — connected-component
  labeling → polygons, not ML); **COG** support (`cog/`, `engines/cog.py`); and the `ops/` primitives:
  focal/terrain (`slope`, `aspect`, `hillshade` with cartographic defaults), zonal stats, proximity,
  sieve, contour, and **generic interpolation** (`ops/interpolate.py`, the scattered-point/`gdal.Grid`
  bridge). STAC entry points `from_stac` / `from_point` (`_stac.py`) are in scope as generic readers.
- **Out of scope / deprecated:**
  - **`convert_units`** (`ops/units.py`, `dataset.py`) — physical value-unit conversion
    (K↔°C, m/s↔knots, Pa↔hPa, m↔mm). Atmospheric/geophysical value-semantics, not a GIS primitive.
    Deprecated; keep unit *metadata* on `band_units`, convert values downstream. **(S1)**
  - The implicit **Sentinel-2 `[2, 1, 0]` RGB plot fallback** in `_resolve_plot_band` — a sensor
    assumption baked into the generic viz path. Pass an explicit `rgb=[...]`. **(S5, viz-only)**

### `dataset/collection.py` — datacube

- **In scope:** `DatasetCollection` — a time-series of co-registered rasters backed by a base
  `Dataset` template; alignment, stacking, temporal indexing, lazy/Dask reads.
- **Borderline, kept:** `from_stac(groupby="solar_day")` — overpass mosaicking for tiled optical EO.
  This is EO temporal semantics, flagged by the audit, but **kept** as a first-class convenience with
  hardened docs. **(S7)** Use `groupby=None` for non-overpass data.

### `netcdf/` — NetCDF / CF / UGRID

`NetCDF` extends `Dataset` with variable/subdataset handling, the time dimension, CF conventions
(`cf.py`), UGRID unstructured grids (`ugrid/`), labeled selection, lazy/multi-file/kerchunk reads, and
metadata.

- **In scope:** all of the above — this is **format support** for NetCDF/HDF/CF/UGRID, exactly the
  category that is in scope however specialised the format is. The UGRID mesh→raster bridge is generic.
- **Out of scope / deprecated:** the **WRF/ROMS/NEMO curvilinear coordinate-name heuristic** in
  `_plot.py` (`_CURVILINEAR_NAME_PAIRS`) — hardcoded model-specific variable names as a plotting
  fallback. Set the CF `coordinates` attribute or pass `coords=` explicitly. **(S6, plot-fallback
  only)** Reading the CF `coordinates` attribute itself is in scope (it is the standard).

### `feature/` — vector core

- **In scope:** `FeatureCollection` (wrapping `GeoDataFrame` + `ogr.DataSource`), rasterization,
  geometry operations (`geometry.py`), bounding-box helpers (`bbox.py`), OGR I/O (`_ogr.py`), and lazy
  vector reads (`_lazy_collection.py`). This is the fiona/geopandas niche — fully in scope.
- **Out of scope:** none identified.

### `stac/` — STAC access & signing

- **In scope:** generic STAC API access (`client.py`, `search.py`), asset loading and engine dispatch
  (`_loader.py`), lazy VRT mosaicking (`_vrt.py`), GeoParquet item serialization (`_geoparquet.py`),
  extension metadata reads (`_extensions.py`), downloads (`download.py`), and the **generic signing
  framework** in `signers.py`: the `Signer` protocol plus `AnonymousSigner`, `AWSRequesterPaysSigner`
  (generic S3 requester-pays), and `BearerTokenSigner` (provider-agnostic).
- **Out of scope / removed:** **provider-specific EO signers** — `PlanetaryComputerSigner`,
  `EarthdataSigner`, `CDSESigner` (and their shared base) — which hardcode specific agency OAuth/SAS
  endpoints. They implement the same `Signer` protocol but encode EO-provider domain knowledge, so they
  live in `earthlens.stac`. **(S3)**

### `basemap/` — web-tile basemaps

- **In scope:** `add_basemap` / `get_provider` — thin wrappers over `cleopatra.tiles` (the viz layer
  does the tile fetch/stitch/warp/render). pyramids only forwards.
- **Out of scope / removed:** **reference map data** — `natural_earth` (Natural Earth vector layers)
  and `relief` (hypsometric raster). This is plotting reference-data acquisition (the `contextily` /
  `cartopy.stock_img()` niche), not a GIS primitive; it now lives in `cleopatra.reference`. **(S4)**

### `grib.py` — GRIB reader

- **In scope:** the GRIB reader and the `_GRIB_FIELDS` glossary that maps **WMO-standard** GRIB metadata
  keys. This is format-level metadata passthrough — the same category as reading GeoTIFF or NetCDF
  tags — and is fully in scope. **(S8)**
- **Out of scope:** none. (Interpreting GRIB *values* in a meteorological model would be out of scope,
  but pyramids only decodes the format.)

### `cli.py` — command-line interface

- **In scope:** a thin CLI exposing in-scope operations (COG conversion, inspection, etc.). It must not
  grow domain-specific subcommands; each command should map to a generic primitive.

### Removed: `grids/`

The exotic model-grid adapters `from_healpix` / `from_octahedral` / `from_orca` (HEALPix cosmology grids,
ECMWF octahedral reduced-Gaussian grids, NEMO ORCA curvilinear ocean grids) **were removed**. The
*regridding* they performed is generic and stays (`ops/interpolate.py`, `ugrid` mesh→raster bridge);
only the domain-specific grid *recognition* moved to `earthlens.grids`. **(S2)**

## 5. Out-of-scope register (worked boundary cases)

The audit produced eight boundary findings. They are the canonical examples of where the §3 rule was
applied, and the precedent for future decisions.

**S1 — `Dataset.convert_units`.** Physical value-unit conversion (K/°C, m·s⁻¹/knots, Pa/hPa, m/mm) is
atmospheric/geophysical value-semantics, not a GIS primitive. → *Deprecated; destination TBD. Keep the
`band_units` metadata in pyramids and convert values downstream.*

**S2 — `grids.from_healpix` / `from_octahedral` / `from_orca`.** Recognising domain-specific scientific
grids (cosmology / NWP / ocean-model) is domain knowledge; the regridding underneath is generic and
stays. → *Removed → `earthlens.grids`.*

**S3 — `PlanetaryComputerSigner` / `EarthdataSigner` / `CDSESigner`.** They hardcode specific EO-agency
auth endpoints — provider domain knowledge. The generic signers and the `Signer` protocol stay. →
*Removed → `earthlens.stac`.*

**S4 — `basemap.natural_earth` / `relief`.** Reference-data download for plotting (the contextily
niche); a pure viz convenience, not a data primitive. → *Removed → `cleopatra.reference`.*

**S5 — implicit Sentinel-2 `[2, 1, 0]` RGB plot fallback.** A sensor-specific band-order assumption
baked into the generic plot path. → *Kept; make the RGB order caller-supplied (no sensor default).
Viz-only.*

**S6 — WRF/ROMS/NEMO curvilinear coord-name heuristic.** Hardcoded model-specific variable names as a
plotting fallback, tried only after the CF `coordinates` attribute. → *Kept as a documented heuristic;
prefer the CF attribute or an explicit `coords=`. Plot-fallback only.*

**S7 — `from_stac(groupby="solar_day")`.** Optical-satellite overpass temporal semantics (solar-day
mosaicking). → *Kept as a first-class EO convenience; documentation hardened.*

**S8 — GRIB `_GRIB_FIELDS` glossary.** WMO-standard GRIB metadata keys — format support, like any
format's tags. → *In scope, no action.*

Note: S5/S6 are low-urgency, viz-only convenience defaults; S7 is a deliberate maintainer exception kept
for EO ergonomics; S1 awaits a relocation destination. S2/S3/S4 are the clear leaks that were relocated.

## 6. Proposing a scope change

Before adding a surface, run it through §3:

1. **Is it format support or a generic spatial/array/IO primitive?** If yes, it is in scope — implement
   it the way GDAL/rasterio would expose it (no sensor/provider/model defaults).
2. **Does its correctness depend on a discipline, sensor, provider, or workflow?** If yes, it is domain
   logic — put it in the package that owns that domain, and have it reuse pyramids' generic primitives
   (the interpolation bridge, the `Signer` protocol, `Dataset`/`FeatureCollection`) rather than adding
   the domain knowledge here.
3. **If it is genuinely borderline**, prefer keeping the *generic mechanism* in pyramids and the
   *domain-specific configuration* out, exactly as S2/S3 split (generic bridge/protocol stays, exotic
   recogniser/provider-signer moves).

Dependency direction matters: pyramids must not depend on its domain consumers (`earthlens` →
pyramids, not the reverse), so EO/domain surfaces are *removed*, not shimmed. The viz layer is the one
exception pyramids may optionally depend on (`cleopatra`, via the `[viz]` extra).
