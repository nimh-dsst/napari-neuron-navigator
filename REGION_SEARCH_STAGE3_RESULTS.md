# Regional Profile Stage 3 Results

Status: **Complete** on 2026-10-01. This document records the implementation
and acceptance evidence for Stage 3 of
`REGION_SEARCH_IMPLEMENTATION_PLAN.md`.

## Structured Regions-Tab Editor

The **Regions** tab now offers **Compound Region Query** without changing the
existing Atlas Regions, Custom Regions, or Mask Layer modes. The new editor:

- builds immutable nested AND, OR, and NOT groups;
- exposes only subject-applicable soma and non-soma neurite conditions;
- supports intersection plus cable-length, node-count, and terminus thresholds;
- retains an independent numeric-ID `RegionSelection` for every clause;
- continuously renders a readable canonical query;
- copies terminal IDs from the existing Custom Regions selection on request;
- uses one shared `RegionSelectorWidget` dialog rather than one atlas tree per
  clause; and
- remaps labels by numeric region ID on atlas changes, clearing selections that
  do not exist in the replacement atlas.

The active clause reuses the existing Regions preview path. Raw **Node types**
controls are hidden in compound mode because the profile's compartments are
Soma and all non-soma Neurite; returning to a simple mode restores the existing
controls and behavior.

## Background Profile and Query Work

**Build Regional Profile** runs the Stage-1 builder in a dedicated thread,
forwards phase progress, and offers **Cancel Build** at safe builder
checkpoints. A previous valid sidecar is not replaced by a cancelled build.

Compound execution snapshots the typed query, atlas, source path, and scope,
then validates and queries the sidecar in a worker-owned DuckDB connection. The
catalog scan also occurs in that worker. **Cancel Query** interrupts an active
DuckDB operation and reports cancellation without changing the Data table.
Results or errors from a stale source/atlas snapshot are discarded.

## Identity, Scope, and Result Handoff

Whole Parquet and Current Table use the existing scope resolver. Current Table
captures explicit `file_id` values, and an empty table fails before work starts.
All Boolean evaluation remains in the Stage-2 query engine and is keyed by
`file_id`; `neuron_id` and `subject` are display metadata only.

Completed results use the existing Data-table population path. Whole Parquet
replaces the prior query result. Current Table retains only matching members
while preserving existing row state. Status output includes the matched count,
scope, canonical query, profile digest, atlas identity, unavailable catalog
IDs, and runtime.

Missing sidecars, incomplete clauses, incompatible profiles, build failures,
query failures, cancellation, and source/atlas changes leave Data unchanged and
produce corrective status text. Profile compatibility remains centralized in
the Stage-1/Stage-2 core; the editor does not implement a second hierarchy or
profile validator.

## Acceptance Evidence

The immutable-editor tests construct both motivating queries without text:

```text
SOMA IN ANY(MOp5, MOs5)
AND NEURITE INTERSECTS CONTRALATERAL CP
```

```text
SOMA IN ANY(MOp5, MOs5)
AND NEURITE INTERSECTS IPSILATERAL pons
```

Their typed ASTs execute against the Stage-2 fixture semantics already recorded
in `REGION_SEARCH_STAGE2_RESULTS.md`. Additional tests cover nested grouping,
thresholds, subject-specific options, independent selections, atlas remapping,
invalid editor state, `file_id` result handoff, Current Table preservation,
profile progress/cancellation, worker-owned catalog loading, and explicit query
cancellation.

Automated results:

- `pixi run test` — 1,281 passed and 4 skipped (1,285 collected).
- Focused Stage-3 regression set — 218 passed.
- Ruff import/undefined-export checks on every changed Python file — passed.
- `git diff --check` — passed.

The four skipped tests are pre-existing Qt-dependent point-import widget tests.
The full napari workflow and visual control behavior have not been exercised
manually. UC-021 in `USE_CASES.md` therefore remains **Not run**, as required;
automated coverage is not treated as manual verification.
