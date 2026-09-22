# Similar-Neuron Search Implementation Plan

> Implementation status (2026-09-22): releases 1 and 2 are implemented.
> Automated tests cover search scoring and scope, versioned import/export,
> Data-table append and annotation, filtered heatmap construction and layer
> metadata, and worker paths. The expanded manual napari workflow in UC-020
> remains unverified, so its status is still **Not run**.

## Release 2: Scope, Annotation, and Distance Heatmaps

### Outcome

Extend the released Search tab so a completed search is a reusable, visible
cohort rather than only a ranked list. Users can restrict the candidate pool to
the whole Parquet, the current Data table, or selected Data rows; export both
the references and matches; add and annotate the complete search cohort in the
Data table; and create CCFv3 heatmaps whose colors communicate similarity to
the search reference.

None of these actions changes the Pearson scorer. References and per-neuron
data remain keyed only by `file_id`.

### Recommended Product Decisions

#### Candidate scope

- Add **Input neurons:** with the same choices and ordering as Analysis:
  **Whole Parquet**, **Current Table**, and **Selected Rows**.
- Scope restricts candidate neurons only. A reference may sit outside the
  candidate scope and must still contribute its complete filtered query vector.
  References that also occur in the scope remain excluded from ranked hits.
- Snapshot the Current Table or Selected Rows `file_id` values when **Run
  Search** is clicked. Later table or selection changes cannot affect a running
  search or the completed result.
- A scoped search needs at least one non-reference candidate. This differs from
  Analysis clustering, which needs at least two input neurons.
- Keep the single-reference selector backed by the whole-Parquet catalog. Do
  not silently narrow it when candidate scope changes.
- Keep the existing Search region and voxel-filter settings when scope changes,
  but clear an active dendrite-label cohort restriction. The coverage scan must
  be rerun against the new scope, just as it is in Analysis.
- Status and export metadata record the resolved scope label and candidate
  count. For example: `Current Table (37 rows; 35 non-reference candidates)`.

Use a Search-owned scope resolver rather than calling private Analysis widget
methods. `NeuronViewerWidget` should provide both the current-table and
selected-row callbacks to Search. The resolved IDs populate the existing
`VoxelSearchRequest.candidate_file_ids`; `None` continues to mean Whole
Parquet. The analysis core already unions references into the filtered source
view, so an out-of-scope reference remains available without becoming a hit.

#### Reference rows in CSV

Export references as actual CSV rows before the ranked hits:

```text
format_version,row_role,rank,file_id,neuron_id,subject,pearson_distance,...
2,reference,0,<file_id>,<neuron_id>,<subject>,,
2,search_result,1,<file_id>,<neuron_id>,<subject>,0.1234,
```

- Every captured reference gets `row_role=reference` and `rank=0`. Multiple
  aggregate members therefore legitimately share rank zero.
- Leave `pearson_distance` empty for reference rows. In aggregate mode an
  individual member is not distance zero from the summed query, so writing
  zero would be a false measurement.
- Hits use `row_role=search_result`, retain ranks `1..N`, and require a finite
  distance in `[0, 2]`.
- Bump the export format to version 2. The loader accepts both version 1 and
  version 2: version 1 imports as results with no recoverable references;
  version 2 restores the reference set and completed-run context.
- Reject a `file_id` duplicated within or across roles. Validate role/rank
  consistency rather than inferring role from rank alone.
- Preserve captured reference order in the exported row order. Aggregate
  summation is order-independent, so all reference ranks remain zero.
- Unknown reference and result IDs remain visible as unavailable after import
  and are never resolved through `neuron_id`.

Add enough compact run context to version 2 to reproduce annotations and
filtered heatmaps after reopening the CSV. A `search_context_json` column
repeats the same canonical, sorted JSON object on every row; this is redundant
but survives row reordering and filtering better than storing it on only the
first row. It should contain the candidate scope, resolution, region-filter
rules, voxel-node filter, metric/policy names, and deterministic filter tags.
The importer must require every non-empty context value in one file to match.
Do not treat the source path in the context as identity for the currently
loaded Parquet; availability is still checked by exact `file_id`.

Represent a loaded document explicitly instead of returning one undifferentiated
DataFrame. For example:

```python
@dataclass(frozen=True)
class SearchResultsDocument:
    references: pd.DataFrame
    hits: pd.DataFrame
    metadata: dict[str, object]
    format_version: int
```

The live `VoxelSearchResult` should likewise carry reference catalog rows, not
only `reference_file_ids`, so export never has to reverse-map a display ID.

#### Add and annotate the search cohort

Keep **Add Selected to Data** and **Add All Results to Data** unchanged. Add a
separate explicit action named **Add & Annotate Search in Data**. It performs
one parent-owned transaction:

1. Append every available reference, in captured order, followed by every
   available hit in rank order.
2. Apply any saved/project table state to newly appended rows.
3. Apply the current search annotations so this explicit action wins over old
   saved metadata.
4. Run the existing scene, heatmap-membership, cluster, and summary refreshes
   once.

Apply these exact annotations:

| Search role | Group | Label |
| --- | --- | --- |
| Reference | `reference` | Preserve the existing label |
| Ranked hit | `search result` | `Rank 1`, `Rank 2`, and so on |

All cohort rows receive deterministic tags derived from the completed run, not
from whatever controls happen to be visible later. Recommended tags are one
atomic table tag per active term:

```text
Search scope: Current Table
Search include: MOp (+10%)
Search exclude: VISp
Search node types: include 2|3
Search dendrite labels: required
Search soma distance: >=200 um
```

If no anatomical or voxel-node filter is active, add `Search filters: none`.
Scope is recorded separately because it changes the candidate cohort rather
than the vectors' row filtering.

The annotation operation is intentionally idempotent. Preserve all tags that
do not start with the managed `Search ` prefix, replace older managed Search
tags on affected rows with the current run's tags, and de-duplicate tags while
preserving order. The explicit action may replace a hit's existing **Group**
and **Label** as described above; it must not change color, visibility, notes,
cluster assignments, scene membership, or heatmap membership. A reference
takes precedence if malformed imported data ever assigns the same ID both
roles.

Add a public batch metadata API to `NeuronTableWidget` rather than reaching
into `_entries`, for example:

```python
def apply_metadata_updates(
    self,
    updates: Mapping[object, NeuronMetadataUpdate],
) -> MetadataUpdateSummary: ...
```

It should disable sorting while updating, refresh visible cells, emit one
`state_changed`, and return updated/missing counts. The Search-to-viewer signal
should carry an immutable annotation request containing references, ranked
hits, and filter tags rather than a bare list of IDs. The final status reports
appended, already present, annotated, and unavailable counts separately.

#### Pearson-distance heatmaps

Add an **Add Search Heatmaps** menu with two explicit modes, each offering:

- **Selected Results + Reference**
- **All Results + Reference**

The modes are:

- **Scored Voxels**: apply the completed search's immutable region and
  voxel-node filters so the layer shows exactly the node rows that contributed
  to Pearson distance.
- **Whole Neuron**: bypass those search filters and show every valid,
  in-atlas source voxel for each requested neuron. The layer color still
  represents the distance calculated from the scored voxels.

The action snapshots the completed result and creates one heatmap for each
requested hit. It also creates the corresponding query heatmap for visual
comparison:

- single-reference mode: one heatmap for that reference;
- aggregate mode: one combined heatmap containing the summed filtered counts
  of all reference members.

The aggregate layer is preferable to one layer per member because hits were
scored against the summed query, not against each member independently. Its
metadata lists every reference `file_id`. Individual aggregate-member
heatmaps can be considered later if users need to inspect contribution balance.

**Scored Voxels** must use the same immutable region and voxel-node filters as
the completed search. **Whole Neuron** is intentionally contextual rather than
metric-exact, so its layer name and metadata must say that search filters were
not applied. A reopened version-2 CSV may create either mode only when all
requested IDs are present in the loaded Parquet and its reference set can be
recovered. Version-1 imports lack recoverable references, so the action should
explain that the search must be rerun.

Each hit layer uses a transparent-to-fixed-color colormap: voxel count controls
brightness/opacity, while the fixed RGB color encodes Pearson distance. Normalize
against the observed minimum and maximum across the complete result table, even
when the user renders only selected rows. This uses the available color range
without making a selected subset change colors:

```text
normalized = 0 if table_max == table_min else
             clip((pearson_distance - table_min) / (table_max - table_min), 0, 1)
similarity = 1 - normalized
hot_coordinate = 0.25 + 0.75 * similarity
color = hot(hot_coordinate)
```

This deliberately reverses the distance direction: closer result neurons look
hotter/brighter, while distant result neurons are dark red. Sampling `hot` no
lower than `0.25` avoids the farthest result becoming black and effectively
invisible with additive blending. If all table distances are equal, every
result uses `hot(1.0)` because there is no relative distance difference to
encode. The query layer uses fixed magenta, outside the result colorscale, and
is named as a reference without claiming that aggregate members individually
have distance zero. Show the observed table range near the action and explain
that **magenta = reference; white/yellow = closest result; dark red = farthest
result**.

Use recognizable layer names such as:

```text
Search Reference (Scored Voxels) Heatmap
Search Rank 1 (d=0.1234; Whole Neuron) Heatmap
```

Store `file_ids`, role, rank, exact distance, voxel mode, whether filters were
applied, color mapping/domain, source search filters, and Search run context in
layer metadata. Continue using additive blending and the individual-heatmap
contrast policy so sparse projections remain visible. Do not recolor Data-table
swatches; the search heatmap color is metric state, not the neuron's persistent
display color.

Reuse the existing sequential selected-neuron heatmap queue, layer lifecycle,
and memory estimator where practical. Estimate retained memory for the query
layer plus every requested hit before starting. Above the existing 1 GiB
threshold, default the confirmation dialog to **Cancel**. Disable the Search
heatmap action for the full queue, stop pending work after an error, retain any
layers already completed, and report `completed/total`. Creating heatmaps must
not implicitly add or annotate Data-table rows.

### Implementation Design

#### 1. Search models and portable format

- Extend `VoxelSearchResult` with reference catalog rows and canonical run
  context, while retaining `reference_file_ids` for convenient identity checks.
- Add a pure filter-tag formatter shared by live results, CSV import/export,
  and Data annotation. It formats `type = 2` as **Axon-typed (type 2)** if a
  human-facing node-type name is used; never call it verified axon.
- Add version-2 export composition: rank-zero references first, then hits.
- Add a structured version-1/version-2 loader. Validate context JSON, roles,
  ranks, distances, duplicate `file_id`, and availability independently.
- Record the scope and exact snapshotted candidate IDs/counts in in-memory
  metadata. Do not put the potentially large candidate-ID list in every CSV
  row; the scope label and count are sufficient portable provenance.

#### 2. Search scope UI and request building

- Add `_current_table_file_ids_provider` and a setter to `SearchTabWidget`.
- Add the scope combo above **Top results** with the exact Analysis labels.
- Implement a Search-specific resolver returning candidate IDs, display label,
  and input count. De-duplicate by exact stringified `file_id` while preserving
  table/selection order.
- On **Run Search**, resolve scope, reject empty/no-non-reference cohorts, and
  construct one immutable request. Use that same candidate scope for preflight,
  dendrite-label scanning, scoring, status, tags, and export metadata.
- When scope changes, invalidate only scope-dependent dendrite coverage; do not
  erase unrelated filter controls or the last completed result. Result actions
  continue to use that result's stored context until a new search completes.

#### 3. Data-table annotation handoff

- Add immutable annotation/update models and the public batch table API.
- Connect a new Search signal in `NeuronViewerWidget`.
- Append available IDs, restore saved state, apply annotations, and perform one
  normal membership refresh in the parent widget.
- Return a structured summary to Search for user-visible counts. Unknown IDs
  from an imported CSV are counted and skipped.

#### 4. Search heatmap handoff

- Add a Search heatmap request model containing the query IDs, selected hit
  rows, exact filters, distance colors, and imported/live provenance.
- Let `NeuronViewerWidget` own worker queues and napari layer creation; the
  Search widget must not manipulate the viewer directly.
- Extend the heatmap build path to accept Search's region and voxel-node
  filters and to combine aggregate references before layer creation.
- Add a pure, independently tested `pearson_distance_to_hot_rgba()` helper.
- Tag layers distinctly from ordinary `selected_neurons` heatmaps so project
  persistence and the existing **Heatmap** column can restore membership
  without confusing their creation mode.

#### 5. Documentation and manual validation

- Expand UC-020 instead of creating a duplicate use case. Add scope,
  rank-zero CSV rows, annotation, and distance-heatmap actions and boundary
  behavior.
- Leave UC-020 at **Not run** until all paths are exercised in napari. Record
  the verification date and any partially tested OS/atlas combination.
- Update the Search section in user documentation after UI labels stabilize.

### Delivery Phases

1. **Portable result context:** reference catalog rows, version-2 CSV,
   version-1 compatibility, filter tags, and format tests.
2. **Candidate scopes:** providers, combo/resolver, scoped coverage scan,
   request snapshots, and scope-aware status.
3. **Data annotations:** table batch API, parent-owned add/annotate transaction,
   UI action, and project-state tests.
4. **Distance heatmaps:** filtered heatmap requests, hot mapping, aggregate
   reference layer, queue/memory handling, metadata, and UI action.
5. **Validation and documentation:** focused tests, full `pixi run test`, visual
   heatmap checks in napari, and UC-020/manual updates.

CSV/context work comes first because both later actions need a durable
distinction between references and ranked hits. Annotation precedes heatmaps
because it is smaller and pins the role/rank semantics before visualization
adds worker and memory concerns.

### Automated Test Plan for Release 2

#### Scope and scorer integration

- Whole Parquet preserves release-1 results.
- Current Table and Selected Rows pass exactly the snapshotted `file_id` cohort
  in stable order; selection changes after launch do not change the request.
- References outside the candidate scope still score correctly and never
  appear as hits.
- Empty scopes and scopes containing only references fail before a worker is
  launched with actionable messages.
- Repeated `neuron_id` values remain distinct because scope, query, and output
  all use `file_id`.
- Changing scope clears scope-dependent dendrite coverage, and the scan uses
  the same cohort as the eventual search.

#### CSV and context

- Single and aggregate references export as rank-zero `reference` rows before
  results, with blank reference distances.
- Version-2 round trip preserves reference order, hit ranks/distances, scope,
  filters, and availability.
- Version-1 CSVs remain loadable as result-only documents.
- Duplicate IDs across roles, bad role/rank pairs, non-empty reference
  distances, invalid result distances, and conflicting context JSON fail with
  corrective errors.
- String-like numeric `file_id` values retain leading zeros.

#### Data annotations

- The action appends references first and hits in rank order, then assigns the
  exact Group/Label values.
- Existing non-Search tags, notes, colors, visibility, cluster assignments,
  selection, scene state, and heatmap membership survive.
- Older managed Search tags are replaced, the current tags are de-duplicated,
  and repeating the action is idempotent.
- Reference precedence is deterministic; unavailable imported IDs are skipped
  and counted.
- Sorting is restored and only one table state-change notification is emitted.

#### Heatmaps

- The query/reference layer uses fixed magenta and is not sampled from the
  result-table distance colorscale.
- Scored Voxels applies the completed run's exact filters, while Whole Neuron
  bypasses them and retains all valid, in-atlas source voxels.
- The result-table minimum, midpoint, and maximum map to the expected `hot`
  samples, and color is monotonic from hotter/closer to darker/farther. A
  selected subset retains the complete table's domain.
- Each hit volume contains only its own nodes and follows its requested voxel
  mode; aggregate reference counts equal the sum of its reference members.
- Selected and All actions preserve rank order and always prepend one query
  layer.
- Layer names and metadata contain role, IDs, rank, exact distance, voxel mode,
  filter-application state, source filters, and color mapping.
- Memory estimation includes the query layer; cancelling above the threshold
  creates no layer. Mid-queue errors retain completed layers and stop pending
  requests with accurate status.
- Search heatmaps do not add rows or change Data-table colors. Version-1
  imports cannot silently create unfiltered heatmaps.

### Release 2 Acceptance Criteria

1. Search offers **Whole Parquet**, **Current Table**, and **Selected Rows** and
   scores only the snapshotted non-reference candidates from that scope.
2. Every exported reference is a rank-zero, explicitly typed CSV row, including
   every member of an aggregate reference; older CSVs still load.
3. One explicit action adds all available cohort neurons and assigns
   `reference`, `search result`, `Rank N`, and completed-run filter tags without
   damaging unrelated Data-table state.
4. Selected or all ranked hits can be rendered with the actual query heatmap
   as either metric-exact **Scored Voxels** or contextual **Whole Neuron**
   layers, using a documented, result-table-normalized hot-distance mapping
   where hotter means closer.
5. Large heatmap batches are guarded by an accurate memory estimate, and all
   background actions snapshot their inputs and report partial/error outcomes.
6. Automated tests pass through `pixi run test`, and the expanded UC-020 is
   manually exercised before its status changes from **Not run**.

## Release 1 Outcome (Implemented)

Add a top-level **Search** tab that ranks neurons by the same CCFv3 voxel-count
Pearson distance used by **Analysis** > **Voxel Correlation**. A user can use one
neuron as the reference or sum several selected neurons into one aggregate
voxel-count vector, apply the existing anatomical and node-row filters, inspect
the closest matches, append chosen matches to **Data** > **Selected Neurons**,
and save or reopen the ranked list.

The result is a nearest-neighbor-style search, not another clustering run. It
does not create cluster assignments or a pairwise distance matrix.

## Recommended First-Release Decisions

These decisions keep the first implementation useful without turning the
Search tab into a general query language.

- Search in **CCFv3 Coordinates** only. Flat map + Depth search should be added
  later through the same request/result interface after CCF behavior is pinned
  to Analysis with parity tests.
- Search the **Whole Parquet** candidate cohort. The reference neuron or neurons
  do not need to be removed from the Data table.
- Support two reference modes:
  - **Single neuron**: choose one neuron from a searchable whole-Parquet
    catalog, with the selected Data-table row available as a shortcut.
  - **Aggregate selected rows**: two or more selected Data-table neurons.
- Build an aggregate by summing the filtered per-voxel node counts of all
  selected reference neurons. This is the literal aggregate count vector. It
  intentionally gives a neuron with more retained nodes more influence; equal
  per-neuron weighting is a separate future mode.
- Exclude all reference `file_id` values from the returned matches. A self-match
  is mathematically useful for testing but not useful in the normal result list.
- Return the best **Top N** matches, defaulting to 100. Smaller Pearson distance
  means a more similar voxel-count pattern.
- Keep Search filter state independent from Analysis filter state. Reuse the
  same filter models and controls, but do not silently copy or synchronize
  settings between tabs.
- Adding search results appends new `file_id` values to the Data table,
  de-duplicates existing rows, and preserves all existing row state, rendered
  layers, colors, labels, notes, and cluster assignments.
- Use a versioned CSV as the portable result format. It is simple to inspect in
  other tools and simple to reopen in Search.

`file_id` is the only identity key throughout this feature. `neuron_id` and
`subject` may be displayed and exported for readability, but must never be used
to group, score, join, de-duplicate, import, or add neurons to the Data table.

## User Experience

Place **Search** next to **Analysis** in the main tab bar. A compact first
version can use four sections.

### 1. Reference Sample

- **Reference mode:** `Single neuron` / `Aggregate selected rows`
- In single mode, a searchable neuron selector populated from the loaded
  Parquet. Match text against `file_id`, `subject`, and display `neuron_id`, but
  store and resolve the choice only by `file_id`.
- **Use selected Data row** shortcut in single mode.
- **Use selected Data rows** button in aggregate mode.
- Read-only summary such as:
  - `Reference: subject-1 / neuron-17 (file_id: ... )`
  - `Aggregate: 4 neurons selected`
- A short explanation that aggregate node counts are summed before Pearson
  distance is calculated.

The tab should snapshot the selected `file_id` values when the user clicks
either Data-selection shortcut. Later changes to Data-table selection must not
mutate a running request or silently change the displayed results.

### 2. Search Filters

- **Region Filters**, using `RegionFilterEditorWidget` and the same independent
  include/exclude rule model as Analysis.
- **Voxel Node Filters**, with the same controls and semantics as Analysis:
  - node-type filter off / include selected / exclude selected;
  - optional whole-neuron dendrite-label coverage restriction;
  - optional **Exclude nodes within soma distance** in CCFv3 microns.
- Reuse the existing annotation and geometric-proxy warnings. Type `2` must be
  called **Axon-typed (type 2)**, never a verified axon compartment.

All filters apply identically to reference and candidate vectors. Applying a
different transformation to the two sides would make the distance meaningless.
Region filters restrict contributing node rows; they do not merely filter the
result list by soma location.

### 3. Search Controls

- **Top results:** integer control, default 100.
- **Run Search** button.
- Background progress and a result summary reporting:
  - requested reference count;
  - candidate neurons scanned;
  - candidates with usable filtered nodes;
  - candidates omitted because no usable nodes remained;
  - returned match count;
  - retained source-node count.

Use the existing large-input preflight convention before scanning more than
10,000,000 retained node rows. Top N limits the output and sorting work, but it
does not reduce the number of candidate rows that must be scanned.

### 4. Results

The visible table should contain:

| Column | Purpose |
| --- | --- |
| Rank | Stable one-based order |
| Neuron | Human-readable `subject` / `neuron_id`, with `file_id` available in the cell or tooltip |
| Pearson distance (1 - r) | The only similarity/distance value |

Sort initially by ascending distance and then by stringified `file_id` so ties
are deterministic. Distances are in `[0, 2]`: `0` is the closest possible,
`1` corresponds to `r = 0`, and `2` is the farthest possible.

Actions:

- **Add Selected to Data**
- **Add All Results to Data**
- **Save Results CSV...**
- **Load Results CSV...**

After an add operation, report separate counts for newly added, already
present, and unavailable IDs. Do not automatically render the added neurons.

## Computation Contract

The data flow is:

```text
Loaded Parquet + atlas
        |
        v
region rules + node types + dendrite cohort + soma-distance rule
        |
        v
filtered rows grouped by (file_id, voxel_id)
        |                         |
        |                         +--> candidate count vectors
        +--> sum reference rows ------> aggregate query vector
                                      |
                                      v
                       Pearson r, distance = 1 - r
                                      |
                                      v
                         exclude references, rank, Top N
```

For one reference neuron, every returned distance must equal that neuron's row
in the Analysis correlation/distance matrix when both runs use the same input
cohort and filters.

For multiple references, if `c_i(v)` is the retained node count for reference
neuron `i` in voxel `v`, the query is:

```text
q(v) = sum_i c_i(v)
```

Each candidate vector is correlated with `q` over the same occupied-voxel
universe. Build that universe before reference rows are omitted from the result
list, so reference-only occupied voxels still affect the distance exactly as
they do in the equivalent pairwise Analysis run. Clip numerical correlation
drift to `[-1, 1]` before computing `1 - r`.

### Existing CCF Edge Semantics

The current CCF Analysis path maps a pair with no shared occupied voxel, or an
undefined Pearson denominator, to `r = -1` when it densifies the correlation
matrix. Search must reproduce that behavior in v1 so its single-reference
distances agree exactly with Analysis. This is a compatibility rule, not a new
statistical claim: changing it later requires a coordinated Analysis/Search
migration and updated export metadata.

### Invalid Reference Inputs

Preflight must validate reference neurons separately from candidates.

- If any requested reference has no usable node after filters, stop and name
  the affected `file_id` values. Do not silently build an aggregate from only a
  subset of the requested sample.
- If the selected soma-distance rule encounters a reference without a valid
  soma, treat that reference as unusable and explain why.
- If no occupied voxel or no candidate remains, return an actionable empty
  state rather than an empty successful table.
- Candidate neurons with no usable nodes are omitted from the ranking and
  counted in the result summary.

## Analysis Layer Design

Add `src/napari_neuron_navigator/analysis/search.py`. Keep all scoring code
independent of Qt so it can be tested on small Parquets.

Suggested immutable models:

```python
@dataclass(frozen=True)
class VoxelSearchRequest:
    reference_file_ids: tuple[str, ...]
    candidate_file_ids: tuple[str, ...] | None
    region_filter: ClusterRegionFilter | None
    voxel_node_filter: VoxelNodeFilter | None
    resolution_um: float
    top_n: int = 100
    exclude_references: bool = True


@dataclass(frozen=True)
class VoxelSearchResult:
    hits: pd.DataFrame
    reference_file_ids: tuple[str, ...]
    input_candidate_count: int
    usable_candidate_count: int
    omitted_candidate_file_ids: tuple[str, ...]
    retained_node_count: int
    metadata: dict[str, object]
```

`hits` should use one row per `file_id` with `rank`, `file_id`, `neuron_id`,
`subject`, and `pearson_distance`.

Build on the current reusable pieces rather than reimplementing filter logic:

- `prepare_cluster_region_filter()` and `register_filtered_source_view()`
- `VoxelNodeFilter`, `prepare_voxel_node_filter()`, and
  `register_voxel_filtered_source_view()`
- the CCF voxel mapping in `analysis/correlation.py`

Refactor `_prepare_region_nodes_view()` into a narrowly named reusable helper
only if needed. Preserve its important ordering: dendrite coverage and soma
centroids are prepared from each complete neuron, while region and node-row
filters are applied to the rows that contribute to a vector.

### Linear-Time Scoring Query

Do not call `compute_pearson_correlation_matrix()` and then discard nearly the
whole matrix. In DuckDB:

1. Build counts grouped by `(file_id, voxel_id)` once.
2. Sum reference counts by `voxel_id` to form the query vector.
3. Compute the query sums and sum-of-squares once.
4. Compute each candidate's sums and sum-of-squares once.
5. Join candidate counts to query counts on `voxel_id` for cross-products.
6. Calculate one Pearson value per candidate, apply the compatibility rule,
   exclude references, order by distance and `file_id`, and apply `LIMIT`.

This changes the correlation stage from quadratic pair generation and an
`N x N` dense matrix to one query-to-candidate pass. Complexity remains linear
in the retained sparse node/count data plus result sorting.

## Worker and UI Integration

Add `SearchWorker` in `workers.py` with the same signal pattern as the Analysis
workers: `progress`, `finished`, and `error`. The widget snapshots a complete
request before starting the thread and disables controls while it runs.

Add `widgets/search_tab.py` with `SearchTabWidget`. In
`NeuronViewerWidget._setup_ui()`:

- create the widget next to Analysis;
- provide the loaded database/Parquet and atlas through `set_database()` and
  `set_atlas()`;
- populate the single-reference selector from a distinct neuron catalog keyed
  by `file_id`;
- provide `NeuronTableWidget.get_selected_file_ids` for capturing references;
- connect an `add_file_ids_requested` signal to a parent-owned Data-table
  append method.

Add a small `NeuronDatabase.get_neuron_catalog()` query for the single-reference
selector. It should return one row per `file_id` with `neuron_id` and `subject`
as display metadata and should validate or deterministically report conflicting
metadata within one file rather than grouping by the display fields.

The Search widget should not reach into Analysis widget internals. If the
node-filter controls need reuse, extract a small `VoxelNodeFilterWidget` used by
both tabs. The model returned by that widget remains `VoxelNodeFilter`.

## Data-Table Handoff

`NeuronTableWidget` currently replaces, retains, or removes membership but has
no append operation. Add a public append API, for example:

```python
def append_file_ids(self, file_ids: Iterable[object]) -> AppendSummary: ...
```

It must:

- de-duplicate by exact `file_id`;
- preserve existing `NeuronEntry` objects and their UI state;
- append new rows in ranked request order;
- initialize new colors without recoloring existing rows;
- populate active saved cluster assignments for the new rows;
- keep scene and heatmap membership synchronized through the existing
  `NeuronViewerWidget` refresh path;
- return added/already-present counts for Search status text.

Keep this integration in `NeuronViewerWidget`, not in the analysis layer, so
the same state-restoration and scene bookkeeping used by normal Data queries is
applied.

## CSV Round Trip

Use UTF-8 CSV with this version-1 schema:

```text
format_version,rank,file_id,neuron_id,subject,pearson_distance
```

Rules:

- `file_id` and `pearson_distance` are required on import.
- `file_id` is read as a string and never reconstructed from `neuron_id`.
- `neuron_id` and `subject` are optional display fields.
- Extra columns are ignored so collaborators can annotate a copy.
- Reject duplicate `file_id` rows, invalid/non-finite distances, unsupported
  format versions, and missing required columns with actionable messages.
- Re-sort valid imported rows by distance and `file_id`, then regenerate rank.
- Compare imported `file_id` values with the currently loaded Parquet. Unknown
  IDs may remain visible as unavailable rows but cannot be added to Data; report
  their count.
- Loading a result file replaces only the Search result table. It must not
  mutate Data until an Add action is used.

Full scientific provenance can later be added as a JSON sidecar or workbook
metadata sheet. It is not required for the first portable-list contract, but
the in-memory `VoxelSearchResult.metadata` should already record the source
Parquet, atlas/resolution, reference IDs, region rules, node filter, candidate
and retained counts, metric name, and compatibility policy.

## Delivery Phases

### Phase 0: Pin Existing Behavior

- Add a small synthetic CCF Parquet fixture.
- Record expected Analysis distances for shared voxels, disjoint voxels,
  filtered-out neurons, and an undefined denominator.
- Add a single-reference parity test that the future search scorer must pass.

### Phase 1: Analysis Search Core

- Add request/result models and CCF query-vector scoring.
- Reuse prepared region and voxel-node filters.
- Add exact preflight counts and actionable reference-validation errors.
- Add deterministic Top-N ranking.

### Phase 2: Search Tab and Background Worker

- Add reference capture, reusable filter controls, Top N, progress, and the
  sortable result table.
- Wire database and atlas lifecycle changes. Loading a new Parquet clears stale
  references and results.
- Keep a running request immutable if Data selection changes.

### Phase 3: Data-Table Append

- Add the table append API and result summary.
- Connect Add Selected/Add All without rendering automatically.
- Preserve table and scene state and make project saving include the appended
  membership through the existing table-state path.

### Phase 4: CSV Import and Export

- Add schema validation and atomic save behavior.
- Exercise a save/reopen/add round trip.
- Add warnings for rows unavailable in the loaded Parquet.

### Phase 5: Validation and Documentation

- Run the non-Qt suite with `pixi run test`.
- Run focused widget tests where the environment permits; do not treat a
  sandboxed Qt failure as manual napari validation.
- Exercise UC-020 in napari and update its manual verification status.
- Add a concise Search section to `README.md` or `MANUAL.MD` once labels and
  behavior are stable.

## Automated Test Plan

### Analysis tests

- Single-reference distance equals the corresponding Analysis matrix row.
- Aggregate counts are summed per voxel before correlation.
- Repeated `neuron_id` and repeated `node_id` values in different files remain
  separate because every operation uses `file_id`.
- Region include/exclude masks match Analysis, including exclude-wins overlap.
- Node-type include and exclude modes match Analysis.
- Dendrite-label coverage is computed from whole neurons and reports excluded
  candidate counts.
- Soma-distance exclusion includes the boundary and rejects missing-soma
  references with an actionable error.
- A reference with zero retained nodes blocks the run; a candidate with zero
  retained nodes is reported and omitted.
- No-overlap and undefined-denominator distances match the pinned CCF policy.
- Ties are ordered by `file_id`, references are omitted, and Top N is exact.

### Table and format tests

- Append preserves existing color, visibility, label, group, tags, notes,
  cluster assignments, added state, selection, and heatmap membership.
- Duplicate additions do not create duplicate rows.
- CSV export/import preserves order, `file_id`, and numeric distance.
- Malformed versions, missing columns, duplicate IDs, NaN/Infinity, and unknown
  IDs produce the intended errors or unavailable-row state.

### Widget/integration tests

- Buttons follow database/atlas/reference/busy/result state.
- The single-neuron catalog keeps duplicate display `neuron_id` values separate
  by `file_id`; its Data-row shortcut rejects zero or multiple selected rows.
- Aggregate mode rejects fewer than two selected rows.
- Changing Data selection during a run does not change the request.
- Loading a new Parquet clears stale results.
- Add Selected and Add All emit exact ranked `file_id` sequences.

## Acceptance Criteria

The first release is complete when:

1. A single-reference search reproduces Analysis Pearson distances under the
   same CCFv3 filters.
2. A multi-reference search uses one summed voxel-count vector and returns a
   deterministic ascending ranking.
3. Region, node-type, dendrite-label, and soma-distance filters have the same
   semantics and warnings as Analysis.
4. The UI clearly calls the metric **Pearson distance (1 - r)** and explains
   that lower is more similar.
5. Selected or all results append to Data without destroying existing state or
   duplicating `file_id` rows.
6. A saved CSV can be reopened and used to add the same available neurons.
7. Query/candidate exclusions and unusable neurons are reported rather than
   silently disappearing.
8. UC-020 has been exercised manually in napari and its status updated.

## Still Deferred After Release 2

- Flat map + Depth search and depth-collapsed flatmap search.
- Per-reference weights or equal-neuron normalization before aggregation.
- Boolean/nested query builders, saved named searches, and search histories.
- Distance thresholds in addition to Top N.
- Approximate-nearest-neighbor indexes and persistent vector caches.
- Searching across multiple Parquet datasets.
- One separate heatmap per aggregate-reference member; release 2 renders the
  summed query that candidates were actually scored against.
- Automatically mutating the Data table or creating heatmaps as a side effect
  of **Run Search**. Release 2 actions remain explicit.
- Automatically clustering or recoloring persistent neuron display state from
  search results.

The analysis API should leave room for these additions, but they are not part
of release 2.
