# Regional Profile Stage 2 Results

Status: **Complete** on 2026-10-01. This document records the implementation
and acceptance evidence for Stage 2 of
`REGION_SEARCH_IMPLEMENTATION_PLAN.md`.

## Implemented Core

`analysis/region_query.py` now provides a Qt-independent Boolean query engine
over a validated regional-profile sidecar:

- immutable `AllOf`, `AnyOf`, `Not`, `SomaIn`, `NeuriteIntersects`, and
  `RegionalThreshold` query nodes;
- numeric-ID `RegionSelection` values with stable acronym/name display
  metadata and an explicit descendant-expansion flag;
- versioned canonical JSON serialization and strict deserialization;
- hierarchy resolution through the Stage-1 closure, with overlapping parent
  and child selections deduplicated before aggregation;
- parameterized DuckDB leaf predicates and `INTERSECT`, `UNION`, and
  scope-relative `EXCEPT` Boolean operations;
- Whole Parquet and explicit-`file_id` scopes, including a typed empty result
  for an empty explicit scope;
- standard `file_id`, `neuron_id`, and `subject` catalog results, with every
  set operation completed on `file_id` before display metadata is joined;
- result metadata containing the canonical query, scope, matched and returned
  counts, unavailable catalog IDs, sidecar content digest, atlas identity,
  and measured runtime; and
- a safe explanation model listing each condition's selected and resolved
  direct region IDs without exposing executable SQL.

The engine validates and materializes the sidecar through Stage 1 before it
accepts a query. Its query universe comes from the compact profile, not the
source node relation. The neuron catalog is copied into a narrow temporary
table during engine setup. Query execution therefore touches only those two
temporary tables.

## Pinned Query Semantics

`NeuriteIntersects` requires a neurite bucket with at least one node or
positive cable length. A soma-only bucket cannot satisfy it, and a neurite
bucket cannot satisfy `SomaIn`.

Threshold conditions sum the requested measurement over the complete,
deduplicated represented direct-region set before applying `HAVING`.
Thresholds start from the active query scope and left-join matching profile
rows, so an absent bucket has value zero. This is required for correct `<`,
`<=`, and `= 0` behavior; grouping only existing profile rows would silently
drop zero-valued neurons.

`QueryLaterality.EITHER` includes all four stored logical channels. Explicit
ipsilateral, contralateral, midline, and unknown restrictions match only that
channel. The Stage-2 mirror fixture confirms that lower- and upper-side somas
with mirrored contralateral projections produce the same query outcome.

Region IDs absent from the compatible atlas fail during explanation and
before DuckDB execution. Region acronyms, names, and all user-facing display
text are serialized for reproducibility but never interpolated into SQL.

## Deterministic Fixture Evidence

The Stage-2 fixture builds a real sidecar through the Stage-1 exact traversal
path. Its seven neurons cover:

- both motivating motor-soma plus CP/pons queries;
- mirrored lower- and upper-soma contralateral projections;
- a CP parent represented by two direct child regions;
- an overlapping parent-plus-child selection;
- cable crossing a region even when no node endpoint occupies that region;
- a soma-only pons neuron;
- repeated `neuron_id` values across distinct `file_id` values; and
- an intentionally incomplete display catalog.

The compatibility query combines soma node presence with neurite
`node_count > 0` over one direct region. It exactly matches the source-node
membership on the fixture, while cable-based `Intersects` correctly retains
its broader traversal-aware meaning.

## Performance Evidence

The Stage-0 reference benchmark remains the canonical-size physical-layout
measurement: its 5,000-neuron, 353,162-row temporary profile ran a warm
one-clause query in 1.61 ms and a warm two-clause parent-aggregation query in
4.49 ms, against budgets of 200 ms and 500 ms respectively.

The completed Stage-2 engine was also measured end to end on a deterministic
expansion of its validated fixture to 18,621 unique `file_id` values and
287,296 temporary profile rows. This expansion is a query-load fixture, not a
biological regional profile. After five warmups, medians over 20 executions
were:

| Query | Matches | Warm median |
| --- | ---: | ---: |
| Contralateral neurite intersection, one clause | 7,981 | 10.475 ms |
| Motor soma `AND` contralateral neurite intersection | 5,321 | 13.185 ms |

These timings include typed-query explanation, hierarchy resolution, SQL
compilation, execution, the display-catalog join, DataFrame creation, and
execution-metadata construction. They clear the Stage-0 budgets despite
returning more rows than the Stage-0 reference queries.

## Automated Evidence

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pixi run pytest -q
  tests/test_region_query_stage2.py` — 19 passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pixi run pytest -q
  tests/test_region_profile_stage0.py tests/test_region_profile_stage1.py
  tests/test_region_query_stage2.py tests/test_db.py` — 61 passed.
- `pixi run ruff check` on the Stage-2 core, exports, and tests — passed.

The headless test command disables third-party pytest plugin autoload because
napari's automatically loaded Qt test plugin aborts while importing Qt in the
Codex macOS sandbox. No Qt behavior changes in Stage 2, and Qt-dependent tests
are outside this stage. A normal full-suite attempt was therefore not treated
as reliable evidence in this environment.
