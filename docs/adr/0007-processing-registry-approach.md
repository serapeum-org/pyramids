# ADR 0007 — `pyramids.processing` tool-registry approach

**Status:** Accepted · **Date:** 2026-08-01 · **Issue:** #780 (task T1)

## Context

The declarative pipeline layer (#780) needs a **tool registry**: existing pyramids ops made addressable by name, each
with a parameter schema and a receiver type (`Dataset` vs `FeatureCollection`), so a serialized pipeline can reference
`"slope"` or `"interpolate_to_raster"` and the runner dispatches to the right object.

The blocker: most `Dataset` ops are exposed as bare `(*args, **kwargs)` facades that delegate to collaborator engines
(69 of 104 public `Dataset` callables as of `main` 0.47.0). `inspect.signature(Dataset.crop)` returns
`(self, *args, **kwargs)` — no parameter names, types, or defaults — so the registry **cannot** be built by
introspecting the public method signatures.

## Options

- **(a) Hand-written `ToolSpec` schemas for a curated allowlist.** Each registered tool declares its params
  explicitly (`Parameter`), independent of whether the underlying method has a real signature.
- **(b) Restore real signatures to the 69 facades first** (architecture-review ARC-121), then auto-introspect the
  registry from signatures.

## Decision

**Adopt (a) for v1.** Ship the registry with hand-written `ToolSpec`s over a **curated allowlist** of
real-signature, serialization-safe ops. This decouples the pipeline work from the 69-facade refactor, ships a working
v1 sooner, and follows the well-established practice of hand-authoring a per-tool manifest for each registered op.

Record (b)/ARC-121 as the follow-up that later lets the registry **auto-expand** beyond the allowlist by introspecting
restored signatures — at which point the hand-written specs for those ops can be generated instead of maintained.

## Consequences

- v1 scope is a curated allowlist, not the full ~136-member surface (see the registry module). Ops whose only useful
  parameters are non-serializable (e.g. `crop(mask=<FeatureCollection>)`, `apply(func=...)`) are excluded until the
  serialization model or ARC-121 lands.
- The `Parameter`/`ToolSpec` schema is the single source of truth for CLI help, pipeline validation, and
  serialization-safety — so it must carry `serializable` and `receiver`/`returns` metadata that a bare signature would
  not provide anyway. This makes (a) the right long-term shape even after ARC-121.
