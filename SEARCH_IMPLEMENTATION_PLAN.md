# Similar-Neuron Search Implementation Plan

> Implementation status (2026-09-22): implemented. Automated tests cover the
> analysis, import/export, Data-table append, and worker paths; the manual
> napari workflow in UC-020 remains unverified.

## Outcome

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

## Explicitly Deferred

- Flat map + Depth search and depth-collapsed flatmap search.
- Candidate scopes other than Whole Parquet.
- Per-reference weights or equal-neuron normalization before aggregation.
- Boolean/nested query builders, saved named searches, and search histories.
- Distance thresholds in addition to Top N.
- Approximate-nearest-neighbor indexes and persistent vector caches.
- Searching across multiple Parquet datasets.
- Automatically rendering, clustering, tagging, or recoloring matches.

The analysis API should leave room for these additions, but none should block
the focused first release.
