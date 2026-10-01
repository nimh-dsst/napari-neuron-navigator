# Regional Profile and Boolean Region Search Implementation Plan

> Planning status (2026-10-01): Stages 0, 1, and 2 complete; Stage 3 not started. This
> document divides the work into independently reviewable stages. Finish and
> validate each stage before beginning the next one; do not treat later-stage
> acceptance criteria as evidence that an earlier stage is complete. Stage 0
> decisions and measurements are recorded in
> `REGION_SEARCH_STAGE0_RESULTS.md`.

## Outcome

Build a fast, atlas-aware regional description of every neuron and use it as
the common foundation for:

- compound queries in the existing **Regions** tab;
- parent-region aggregation through the Allen hierarchy;
- ipsilateral and contralateral projection queries relative to each neuron's
  soma;
- sparse, fixed-column feature matrices for clustering, classification, and
  other machine-learning methods.

The initial user-facing queries include:

```text
SOMA IN ANY(MOp5, MOs5)
AND NEURITE INTERSECTS CONTRALATERAL CP
```

```text
SOMA IN ANY(MOp5, MOs5)
AND NEURITE INTERSECTS IPSILATERAL pons
```

The implementation must reuse the existing atlas hierarchy, Whole Parquet and
Current Table scopes, region previews, and Data-table result path. It must not
create an independent region-selection system or place this workflow in the
similar-neuron **Search** tab.

## Product Terminology

Call the stored representation a **regional profile**, not a distance metric.
It describes a neuron as regional measurements; a downstream algorithm decides
how profiles are compared.

Use **neurite** for all non-soma morphology. The UI may explain it as
"Projection (all non-soma neurites)," but it must not label `type = 2` data as
biological axon. The source annotations contain dendritic projections typed as
`2`, and their error rate is unknown.

The following terms have precise meanings throughout the implementation:

- **Direct region**: the `region_id` assigned by the atlas annotation at a
  node or cable-length portion. It is not an ancestor copied into the profile.
- **Aggregated region**: a selected Allen region plus its represented
  descendants, resolved from the loaded atlas hierarchy.
- **Ipsilateral**: on the same physical side of the atlas midline as the
  neuron's soma.
- **Contralateral**: on the opposite physical side from the soma.
- **Terminus**: a childless non-soma node found from the complete morphology
  graph before applying any region, side, or compartment restriction.
- **Intersects**: at least one qualifying node or a positive qualifying cable
  length in the resolved region set. Later stages may add a minimum evidence
  threshold without changing this shorthand.

## Non-Negotiable Correctness Contracts

### Neuron identity

`file_id` is the only per-neuron key. Every join, group, parent lookup,
terminus lookup, profile row, query result, cache identity, and ML matrix row
must remain scoped by `file_id`.

`neuron_id` and `subject` are display metadata only. In particular, never
group profile construction by `neuron_id`, and never join a parent node on
`node_id` without also joining on `file_id`.

### Complete-tree topology

Terminus detection must see every node in each selected `file_id`. Determine
childlessness first, then assign the resulting termini to compartment, region,
and laterality buckets. Do not filter nodes by region, side, or type before the
child lookup.

Node IDs must be treated as arbitrary identifiers. They need not be
contiguous, start at one, or follow parent-before-child order.

### Direct storage and hierarchical aggregation

The canonical profile stores direct atlas regions only. Parent rows are never
materialized alongside child rows in the same canonical table. Parent totals
are computed with a hierarchy closure mapping, preventing storage expansion
and silent double counting.

The closure includes `(region_id, region_id, depth=0)` so direct and parent
queries use the same path. It is built from `structure_id_path` and does not
assume `parent_structure_id` is present.

If a condition contains overlapping selected regions, such as a parent and
one of its children, resolve and deduplicate the represented direct IDs before
summing measurements.

### Relative laterality

Laterality is computed relative to the soma, not by hardcoding left or right.
Mirroring an entire neuron and its soma across the atlas midline must preserve
its ipsilateral/contralateral regional profile.

The profile uses four explicit logical values:

- `ipsilateral`
- `contralateral`
- `midline`
- `unknown`

A midline or indeterminate soma must not be silently assigned to a hemisphere.
The atlas left-right axis and midline used by the builder are recorded in
profile metadata.

### Atlas compatibility

A regional profile is valid only for the source node Parquet and the atlas
hierarchy/annotation used to build it. A different atlas name, version,
resolution, hierarchy digest, or annotation geometry invalidates the profile.
Do not combine or compare region IDs across incompatible atlas identities.

## Canonical Regional Profile

### Sparse long-form schema

Store one row for each non-empty:

```text
file_id x direct region_id x laterality x compartment
```

The version-1 logical schema is:

| Column | Type | Meaning |
| --- | --- | --- |
| `file_id` | string | Required neuron identity key. |
| `region_id` | int32 | Direct Allen atlas region; `0` means outside/unmapped. |
| `laterality` | dictionary string or compact enum | `ipsilateral`, `contralateral`, `midline`, or `unknown`. |
| `compartment` | dictionary string or compact enum | `soma` or `neurite`. |
| `cable_length_um` | float64 | Cable length allocated to the bucket. |
| `node_count` | int64 | Nodes whose coordinates fall in the bucket. |
| `terminus_count` | int64 | Complete-tree, childless non-soma nodes in the bucket. |

The profile deliberately adds `compartment` to the requested region/side
summary. Without it, a soma node alone could incorrectly satisfy "projection
intersects," and the two example queries could not be expressed faithfully.

Rows with `region_id = 0` are retained for quality control and conservation
checks but are hidden from ordinary atlas-region selection.

### Display catalog

Continue to use a separate one-row-per-`file_id` catalog containing display
metadata such as `neuron_id` and `subject`. Query results join the matching
`file_id` values to this catalog only after all Boolean operations are
complete.

### Cable-length allocation

The durable target is length measured against the voxelized atlas rather than
assigning an entire parent-child edge to one endpoint:

1. Join each non-root child to its parent on `(file_id, parent_id = node_id)`.
2. Treat the straight child-parent segment as the cable represented by that
   SWC edge.
3. Traverse the atlas voxels crossed by the segment using 3D grid traversal or
   an equivalent exact piecewise-linear method.
4. Split length where the atlas region changes and where the segment crosses
   the physical midline.
5. Assign every portion to the child's compartment. An edge from the soma to
   its first non-soma child is neurite cable.
6. Store outside-atlas portions under `region_id = 0` so total length remains
   auditable.

The implementation must record a versioned value such as
`atlas_voxel_traversal_v1` in `length_method`. If Stage 0 shows that exact voxel
traversal cannot meet the agreed build-time budget, stop and revise this plan
before substituting endpoint attribution or sampling. A faster approximation
must never be introduced under the same method name.

The builder must satisfy this conservation invariant for each `file_id`,
within a documented floating-point tolerance:

```text
sum(profile.cable_length_um) == sum(Euclidean child-parent edge lengths)
```

### Node and terminus allocation

Nodes are assigned from their coordinate's existing direct `region_id` and
physical side. A soma node contributes to `compartment = soma`; all other node
types, including undefined type `0`, contribute to `compartment = neurite`.

Termini are computed from the complete `(file_id, node_id, parent_id)` graph.
Only after that computation are childless non-soma nodes assigned to their
direct region and laterality bucket. No `type = 2` restriction is applied.

### Profile metadata and invalidation

Store versioned JSON metadata in the Parquet schema under a dedicated key such
as:

```text
napari_neuron_navigator.region_profile_json
```

The metadata contains at least:

- profile format version;
- builder algorithm version;
- source-Parquet fingerprint and row count;
- required source-column names and their types;
- atlas name, version when available, resolution, shape, and hierarchy digest;
- atlas left-right axis and midline in microns;
- cable-length method and numeric tolerance;
- compartment and terminus definitions;
- build timestamp and aggregate validation counts.

The source fingerprint policy is pinned in Stage 0. It must detect source
replacement reliably without requiring a full source scan every time the
Regions tab opens.

## Query Model

### Typed abstract syntax tree

All query entry points create the same typed abstract syntax tree. The core
must not accept raw SQL. The initial node types are conceptually:

```text
AllOf(children)
AnyOf(children)
Not(child)
SomaIn(region_selection)
NeuriteIntersects(region_selection, laterality)
RegionalThreshold(
    compartment,
    measurement,
    comparison,
    value,
    region_selection,
    laterality,
)
```

`region_selection` stores numeric atlas IDs, whether descendants are included,
and display acronyms/names for reproducibility. Numeric IDs are authoritative.

Supported threshold measurements are:

- `cable_length_um`
- `node_count`
- `terminus_count`

The first release supports `>`, `>=`, `<`, `<=`, and `=` after validating that
the threshold is finite and non-negative.

### SQL compilation

Each leaf predicate returns a distinct set of `file_id` values. Compile Boolean
nodes to DuckDB set operations:

- `AllOf` -> `INTERSECT`
- `AnyOf` -> `UNION`
- `Not` -> `EXCEPT` from the resolved query scope

Threshold predicates aggregate over the deduplicated represented direct
regions before applying `HAVING`. Parameters are bound values; region names,
acronyms, and user-entered text are never interpolated into SQL.

The Whole Parquet or Current Table scope is resolved before evaluation and
forms the universe for `Not`. Results always contain unique `file_id` values.

### Text language

The text language is a parser for the typed query model, not natural-language
generation and not a SQL pass-through. It uses case-insensitive keywords,
explicit region resolution, normal `NOT` > `AND` > `OR` precedence, and
parentheses.

Prefer an unambiguous multi-region form:

```text
SOMA IN ANY(MOp5, MOs5)
AND NEURITE INTERSECTS CONTRALATERAL CP
```

The UI must show the canonical parsed form before execution. Ambiguous or
unknown names, atlas mismatches, malformed parentheses, and unsupported terms
produce location-aware errors without running a partial query.

## Storage and Runtime Design

### Sidecar layout

Store the profile outside the node Parquet so old Parquets remain readable and
the query path scans only compact data:

```text
source.parquet
source.region_profile.parquet
```

Use a single Parquet sidecar by default rather than a directory containing
thousands of partitions. Write rows in an order favorable to the common
predicate columns, initially:

```text
region_id, laterality, compartment, file_id
```

Choose row-group size, compression, direct Parquet scanning versus an in-memory
DuckDB temporary table, and any indexing only after Stage 0 measurements.

### Bounded profile construction

Build profiles in complete-neuron batches. Every batch must include every node
for each participating `file_id`, keeping parent joins and childless detection
correct while bounding memory. The builder writes completed batch aggregates
incrementally and performs a final sort/compact pass without materializing the
entire node table in pandas.

Profile creation is a cancellable background operation with progress. A
cancelled or failed build leaves no sidecar that can be mistaken for a complete
profile; publish the final file with an atomic rename after validation.

### Interactive query path

Register a validated profile sidecar once per loaded source Parquet. Warm
queries must scan the sidecar or its measured faster representation, never the
full node Parquet. Cache resolved descendant-ID sets and compiled immutable
query inputs for the lifetime of the loaded atlas.

The existing raw-node region methods remain available for arbitrary SWC
node-type filtering and compatibility. The profile fast path owns Soma,
Neurite, laterality, length, node-count, and terminus predicates.

## Existing Integration Points

The work should extend these components rather than duplicate them:

- `src/napari_neuron_navigator/parquet.py`: canonical node schema and
  conversion pipeline.
- `src/napari_neuron_navigator/db.py`: DuckDB connection, neuron catalog, raw
  region queries, and standard result columns.
- `src/napari_neuron_navigator/terminals.py`: complete-tree childless
  semantics and batching lessons.
- `src/napari_neuron_navigator/widgets/region_selector.py`: atlas hierarchy,
  descendant expansion, and region selection.
- `src/napari_neuron_navigator/widgets/neuron_viewer.py`: Regions-tab scope,
  previews, status, and Data-table result handoff.
- `src/napari_neuron_navigator/workers.py`: cancellable background work and
  progress/error signaling.
- `src/napari_neuron_navigator/project_io.py`: later persistence of a completed
  query definition when project behavior is added.

Prefer new focused modules:

```text
src/napari_neuron_navigator/analysis/region_profile.py
src/napari_neuron_navigator/analysis/region_query.py
src/napari_neuron_navigator/widgets/region_query_editor.py
```

Names may change during implementation, but profile construction, query
evaluation, and Qt presentation must remain separately testable.

## Delivery Stages

### Stage 0: Pin Semantics, Fixtures, and Performance Budgets

#### Outcome

Remove algorithmic uncertainty before committing to a sidecar format. Produce
small deterministic fixtures that all later stages reuse, and measure the
candidate cable traversal and DuckDB layouts.

#### Work

1. Create synthetic morphologies covering:
   - a segment entirely inside one atlas voxel/region;
   - a segment crossing a region boundary;
   - a segment crossing the atlas midline;
   - a segment partly outside the annotation volume;
   - left- and right-soma mirror pairs;
   - duplicate `neuron_id` and duplicate `node_id` values across `file_id`;
   - non-contiguous and parent-after-child node IDs;
   - an interleaved node-type tree whose termini require a complete child
     lookup.
2. Prototype and verify exact voxel-grid segment traversal, including
   anisotropic voxel resolutions even if the first Allen fixture is isotropic.
3. Compare the traversal output with analytically known segment portions and a
   very fine independent sampling oracle used only in tests.
4. Measure profile-row cardinality and query latency for representative
   subsets of the canonical 18,621-neuron Parquet.
5. Compare sorted Parquet scanning with a DuckDB temporary materialization and
   document the chosen runtime layout.
6. Pin the source fingerprint and atomic sidecar publication policy.
7. Record hardware, input size, cold/warm state, and commands in the benchmark
   output so later numbers are comparable.

#### Provisional performance budgets to confirm or revise

- Opening an already valid sidecar must not scan all node rows.
- A warm one-clause query over the canonical dataset should complete within
  200 ms on the documented reference machine.
- A warm two-clause `AND` query with parent aggregation should complete within
  500 ms on that machine.
- Profile construction must use bounded memory and report throughput, even if
  the one-time exact traversal takes substantially longer than a query.

These are engineering budgets, not claims about current performance. Revise
them in this document using measured evidence before Stage 1 if they are not
realistic.

Stage 0 retained the 200 ms and 500 ms query budgets and pinned additional
registration, memory, and construction-throughput gates. The measured runtime
layout, exact figures, fingerprint policy, and atomic-publication policy are in
`REGION_SEARCH_STAGE0_RESULTS.md`.

#### Acceptance criteria

- Exact segment portions and conservation pass on every synthetic fixture.
- The mirror pair produces identical relative-laterality aggregates.
- A written benchmark result selects the storage/runtime layout.
- The cache fingerprint and invalidation contract are unambiguous.
- No production UI behavior changes in this stage.

### Stage 1: Regional Profile Core and Sidecar Builder

#### Outcome

Create, validate, load, and inspect a version-1 regional-profile sidecar from
an annotated neuron Parquet and a compatible loaded atlas.

#### Work

1. Add immutable profile metadata and build-summary models.
2. Implement atlas hierarchy digest and direct-region closure construction.
3. Validate required source columns and per-file graph invariants before or
   during batched construction.
4. Resolve each soma's physical hemisphere from its soma coordinates and the
   loaded atlas. Report missing, multiple, or midline soma cases explicitly.
5. Join parent-child edges by `(file_id, node_id)` and allocate their length
   through the Stage-0 traversal implementation.
6. Compute childless nodes from the complete tree and allocate all non-soma
   termini after topology is known.
7. Aggregate direct rows by
   `(file_id, region_id, laterality, compartment)`.
8. Retain unmapped rows, run per-file and whole-file conservation checks, and
   write versioned metadata.
9. Write through a temporary path and atomically publish only a complete,
   validated sidecar.
10. Add a read-only inspection API returning schema, metadata, aggregate
    counts, and compatibility status without loading all profile rows.
11. Add optional profile generation to the conversion workflow only after the
    standalone builder is stable. Existing conversion defaults must remain
    explicit and testable.

#### Automated coverage

- `file_id` keeps duplicate display IDs and node IDs separated.
- Arbitrary node ordering and non-contiguous IDs produce the expected edges.
- Dangling parents, duplicate `(file_id, node_id)`, missing somas, and invalid
  coordinates produce actionable validation results.
- A child's existence prevents its parent from being counted as a terminus
  regardless of region or node type.
- Undefined type `0` nodes contribute to neurite rows.
- Type `2` is never renamed or interpreted as biological axon.
- Cable length is conserved per file and globally.
- Node and terminus counts are conserved across direct-region buckets.
- Mirrored neurons retain the same ipsilateral/contralateral profile.
- Midline and unmapped cases remain explicit.
- An atlas/source/algorithm metadata mismatch rejects a stale sidecar.
- Cancellation and injected failure never publish a valid-looking partial
  sidecar.

#### Acceptance criteria

- The builder completes on a representative real subset within the Stage-0
  memory budget.
- The sidecar passes all conservation and compatibility validation.
- Loading metadata and registering the profile does not scan the source node
  table.
- No Regions-tab control depends on the new builder yet.

Stage 1 satisfied these criteria on 2026-10-01. The implementation, real-data
measurements, compatibility behavior, and automated evidence are recorded in
`REGION_SEARCH_STAGE1_RESULTS.md`.

### Stage 2: Core Boolean Query Engine

#### Outcome

Evaluate typed, hierarchy-aware regional queries against a validated sidecar
without Qt and return the repository's standard neuron catalog rows.

#### Work

1. Implement immutable query AST models and canonical serialization.
2. Implement region selections keyed by numeric atlas IDs with stable display
   metadata.
3. Resolve selected parents to deduplicated direct-region sets through the
   shared hierarchy closure.
4. Compile leaf conditions to parameterized DuckDB queries over the profile.
5. Compile Boolean nodes to `INTERSECT`, `UNION`, and scope-relative `EXCEPT`.
6. Support Whole Parquet and explicit `file_id` scopes; an empty explicit scope
   returns an empty result without invalid SQL.
7. Join final IDs to the neuron catalog for `file_id`, `neuron_id`, and
   `subject` display columns.
8. Return execution metadata including canonical query, scope, matched count,
   sidecar identity, atlas identity, and measured runtime.
9. Add an explain/debug representation that identifies the resolved direct
   region IDs without exposing executable raw SQL in the UI.

#### Automated coverage

- The two motivating queries return the expected fixture neurons.
- Parent-region predicates equal the union/sum of their represented direct
  descendants.
- Selecting a parent and its child in one condition does not double count.
- `AND`, `OR`, `NOT`, nesting, and scope-relative negation obey the AST.
- Soma predicates cannot be satisfied by a neurite row, and neurite predicates
  cannot be satisfied by a soma-only row.
- Ipsilateral and contralateral predicates remain correct for left- and
  right-soma neurons.
- Length thresholds aggregate all represented descendants before comparison.
- Unknown or incompatible region IDs fail before execution.
- Results are unique and keyed by `file_id` even when display IDs repeat.
- A simple "any node in these direct regions" compatibility query matches the
  existing raw-node query on a shared fixture.
- Query execution does not read the source node relation.

#### Acceptance criteria

- The query engine is usable from a unit test or small script without Qt.
- Warm representative queries meet the Stage-0 latency budgets.
- Query serialization round-trips without losing hierarchy or laterality
  semantics.

Stage 2 satisfied these criteria on 2026-10-01. The headless API, query
semantics, performance measurement, and automated evidence are recorded in
`REGION_SEARCH_STAGE2_RESULTS.md`.

### Stage 3: Structured Query Builder in the Regions Tab

#### Outcome

Users can construct and run compound regional-profile queries from the
existing **Regions** tab without learning the text grammar.

#### Product design

Keep the current simple Atlas Regions, Custom Regions, and Mask Layer queries
available. Add a clearly labeled **Compound Region Query** mode rather than
silently changing the meaning of existing controls.

Each condition row exposes only applicable controls:

- **Subject:** Soma or Projection (all non-soma neurites)
- **Condition:** In region, Intersects, Cable length, Node count, or Termini
- **Side:** Either, Ipsilateral, or Contralateral where applicable
- **Regions:** a compact summary plus an **Edit Regions...** action
- **Comparison** and **Value:** for threshold conditions
- **Combine with:** AND or OR
- add, remove, duplicate, and group controls

Do not instantiate a complete atlas tree for every condition row. Reuse one
region-picker dialog or a shared hierarchy model, loading each condition's
selection into it when edited. This keeps UI construction and atlas switching
bounded as queries grow.

The editor continuously displays the canonical structured query. The active
condition's region selection may drive existing mesh/segmentation previews;
the UI must clearly identify which clause is being previewed.

#### Work

1. Add the structured editor as a separately testable widget.
2. Reuse the existing atlas selector and Custom Region terminal IDs through a
   shared region-selection value model.
3. Reuse Whole Parquet and Current Table scope resolution.
4. Detect a missing, stale, or incompatible sidecar before query execution.
5. Offer a cancellable **Build Regional Profile** action with progress when a
   compatible sidecar is unavailable. Never block the Qt event loop for the
   build.
6. Run profile queries in a worker even when warm latency is expected to be
   short.
7. Send results through the existing Data-table population path. Whole Parquet
   replaces the current query result as the existing simple query does;
   Current Table restricts and preserves the existing table consistently with
   current behavior.
8. Report matched neurons, scope, canonical query, profile identity, and query
   runtime in status text/logging.
9. Preserve the current raw-node **Node types** control in simple-query mode.
   Disable or hide it in compound-profile mode with an explanation that the
   profile uses Soma versus all non-soma Neurite compartments.
10. Keep mask queries on their existing raw path. Arbitrary mask clauses are
    not inferred from atlas profile rows.

#### Automated coverage

- Control visibility and valid options follow subject/condition choices.
- Independent clauses retain independent region selections.
- Reopening the shared picker restores the selected clause exactly.
- Atlas switching clears or rejects incompatible clause IDs and profiles.
- Whole Parquet and Current Table produce the expected result membership and
  preservation behavior.
- Empty Current Table, no conditions, no selected regions, missing profile,
  stale profile, cancellation, and worker failure are actionable and do not
  alter the table.
- A completed query populates results through `file_id` and reports unavailable
  catalog rows without substituting `neuron_id`.
- Existing simple Atlas, Custom, and Mask queries retain their behavior.

#### Acceptance criteria

- Both motivating queries can be constructed without text entry and return
  correct fixture results.
- Profile construction and queries remain responsive and cancellable in
  napari.
- No second, inconsistent atlas hierarchy implementation is introduced.
- Existing Regions-tab automated tests remain green.

### Stage 4: Text Query Input and Canonical Parsing

#### Outcome

Users can type, validate, edit, and execute region queries while the structured
builder and text form remain two views of the same AST.

#### Grammar

Implement a small explicit grammar, for example:

```text
expression     := or_expression
or_expression := and_expression (OR and_expression)*
and_expression:= unary_expression (AND unary_expression)*
unary_expression := NOT unary_expression | '(' expression ')' | predicate
```

Initial predicate forms include:

```text
SOMA IN <region>
SOMA IN ANY(<region>, ...)
NEURITE INTERSECTS [EITHER|IPSILATERAL|CONTRALATERAL] <region>
NEURITE LENGTH [side] IN <region> >= <number> UM
NEURITE NODES [side] IN <region> >= <integer>
NEURITE TERMINI [side] IN <region> >= <integer>
```

Full region names containing spaces require quoting. Acronyms and unambiguous
full names resolve case-insensitively against the loaded atlas. If a token is
ambiguous, show all candidates and require the user to choose; never select the
first match silently.

#### Work

1. Implement a tokenizer and recursive-descent parser, or adopt a small parser
   dependency only after documenting why it is preferable.
2. Preserve source spans for location-aware syntax and resolution errors.
3. Parse to the existing typed AST, then run normal validation and SQL
   compilation.
4. Render every valid AST to one canonical text form.
5. Synchronize text and structured views only after successful parsing; invalid
   text must not destroy the last valid structured query.
6. Add examples and concise inline syntax help.
7. Record the canonical query and AST format version in any saved query state.

#### Automated coverage

- Keyword case, whitespace, quoted names, acronyms, and numeric thresholds.
- `NOT` > `AND` > `OR` precedence and nested parentheses.
- The motivating expressions parse to the intended AST.
- `SOMA IN ANY(MOp5, MOs5)` is equivalent to an explicit soma-region `OR`.
- Unknown, ambiguous, and incompatible regions identify the failing source
  span.
- Invalid numbers, units, comparisons, and missing parentheses fail without
  executing.
- AST -> canonical text -> AST is stable.
- Text and structured forms execute identically.
- Query text can never inject or alter SQL structure.

#### Acceptance criteria

- Both motivating queries can be entered as text, inspected canonically, and
  executed.
- Parser errors are actionable and leave the prior valid query intact.
- Text and structured editors cannot diverge semantically.

### Stage 5: Sparse Regional Features for Machine Learning

#### Outcome

Convert the same validated regional profiles and hierarchy semantics into
deterministic sparse feature matrices without introducing a second aggregation
implementation.

#### Feature model

Return a model containing at least:

```text
matrix: scipy.sparse.csr_matrix
file_ids: ordered tuple[str, ...]
features: ordered tuple[RegionalFeatureKey, ...]
metadata: atlas/profile/aggregation/transform provenance
```

A feature key contains:

- output region ID;
- laterality;
- compartment;
- measurement (`cable_length_um`, `node_count`, or `terminus_count`).

Soma location may be represented as a soma-region one-hot feature block from
the same profile rows.

#### Work

1. Add a core feature-matrix builder that accepts an explicit ordered
   `file_id` cohort and explicit region vocabulary or validated atlas cut.
2. Reuse the hierarchy closure and direct-region aggregation from the query
   engine.
3. Construct Arrow/NumPy coordinate triples and then CSR; do not pivot the
   complete atlas vocabulary through a dense pandas DataFrame.
4. Support versioned transformations without baking them into the profile:
   - raw values;
   - per-neuron cable-length fractions;
   - `log1p` counts or lengths;
   - optional downstream standardization metadata.
5. Keep feature blocks distinguishable so length, node count, and termini are
   not silently placed on a common scale.
6. Warn or reject by default when a requested ML vocabulary contains
   overlapping parent/child regions. Permit it only through an explicit option
   and record that the features overlap.
7. Expose node count as optional because reconstruction sampling density can
   dominate it; make cable length the recommended projection feature.
8. Add a portable export containing the sparse matrix plus ordered row/column
   metadata and complete profile/atlas provenance.
9. Do not add a clustering algorithm in this stage. Existing or future methods
   consume the representation explicitly.

#### Automated coverage

- Deterministic row and feature ordering.
- Exact agreement between query-time parent sums and ML feature values.
- Correct ipsilateral/contralateral feature channels for mirror fixtures.
- Stable empty-region and zero-valued behavior without dense expansion.
- Raw, fractional, and `log1p` transformations match hand calculations.
- Feature export/import preserves matrix values, `file_id` order, feature
  order, and provenance.
- Incompatible profiles or atlases cannot be combined.
- Duplicate `neuron_id` values remain separate matrix rows.

#### Acceptance criteria

- A caller can build a CSR matrix for all canonical neurons using bounded
  memory.
- The matrix uses the same direct-region, hierarchy, laterality, compartment,
  and terminus semantics as interactive queries.
- Exported features are self-describing enough to prevent incompatible runs
  from being compared as if they matched.

### Stage 6: Optimization, Persistence, and Documentation

#### Outcome

Harden the complete workflow on the canonical dataset, preserve useful query
state where appropriate, and document it as a repeatable user capability.

#### Work

1. Add `scripts/benchmark_region_profile.py` or an equivalently focused
   benchmark covering:
   - build throughput and peak memory;
   - sidecar row count and compressed size;
   - cold registration time;
   - warm simple, parent, Boolean, threshold, and scoped query latency;
   - sparse feature-matrix build time and memory.
2. Run DuckDB `EXPLAIN` checks during performance work to confirm queries use
   the profile sidecar rather than the source node table.
3. Optimize only measured bottlenecks: row-group layout, temporary
   materialization, descendant lookup caching, batch size, or SQL shape.
4. Persist the last valid compound query in projects only if the loaded
   project can verify its atlas/profile compatibility. Otherwise retain it as
   visible but unavailable state with an explanation.
5. Add query/profile provenance to relevant exports and logs.
6. Update README/MANUAL documentation after UI labels stabilize.
7. Add the next sequential entry to `USE_CASES.md` covering profile creation,
   both motivating queries, parent aggregation, side switching, Current Table
   scope, cancellation, stale-profile handling, and text/structured parity.
8. Leave the new use case at **Not run** until it is exercised manually in
   napari; code inspection and automated tests do not count as manual
   verification.
9. Record canonical benchmark results with the neuron count, source identity,
   atlas, hardware, and implementation versions.

#### Acceptance criteria

- The canonical 18,621-neuron dataset meets the Stage-0 query budgets or this
  document records justified revised budgets and measurements.
- Query execution performs no full node-Parquet scan after a valid sidecar is
  registered.
- Profile building remains bounded, cancellable, and recoverable.
- Project/export round trips reject or expose incompatible atlas/profile
  state rather than silently reusing it.
- Automated tests pass through `pixi run test` to the extent supported in the
  environment, and Qt-dependent limitations are reported explicitly.
- The complete manual use case has an honest verification status and date.

## Stage Status

| Stage | Deliverable | Status |
| --- | --- | --- |
| 0 | Semantics, fixtures, traversal prototype, budgets | Complete (2026-09-24) |
| 1 | Regional-profile sidecar builder | Complete (2026-10-01) |
| 2 | Core Boolean query engine | Complete (2026-10-01) |
| 3 | Structured Regions-tab query builder | Not started |
| 4 | Text query parser | Not started |
| 5 | Sparse ML feature matrices | Not started |
| 6 | Optimization, persistence, and documentation | Not started |

Update this table and the opening planning status only when a stage satisfies
all of its acceptance criteria. If a stage changes a pinned semantic or file
format, update the earlier contracts and version the affected persisted data.

## Deferred Beyond This Plan

- Inferring biological axon versus dendrite from unreliable SWC `type` values.
- Quantifying the source dataset's compartment-label error rate.
- Arbitrary mask-layer clauses inside the atlas regional-profile query AST.
- Natural-language or model-generated query interpretation beyond the explicit
  grammar.
- Training, selecting, or endorsing a particular ML model.
- Automatically combining regional profiles from different atlases.
- Materializing every ancestor as a canonical profile row.
- Replacing the source node Parquet with a nested per-neuron profile column.

These may become separate releases, but none is required to deliver fast,
hierarchical soma/projection queries and reusable regional feature matrices.
