# Preserve Tags and Set Membership

Issue: [#173](https://github.com/nimh-dsst/napari-neuron-navigator/issues/173),
subissue 1. Branch: `issue-173-preserve-tags-and-set-membership`.

Planning date: **2026-11-06**. Inspected baseline: `2c64a43`.
Implementation date: **2026-11-06**.
Status: annotation fixes and save-export resource limits implemented. Annotation
regressions, the synthetic napari GUI check, and full-suite verification passed,
including the save-resource follow-up. The original project that
showed empty sets immediately on loading is unavailable on this computer.

## Implementation results

Three annotation-loss paths were reproduced before their fixes:

1. Applying saved state to populated rows interpreted display text as a new
   user edit. The tag `Region query: TH, CP` became two tags after the first
   table refresh, leaving the live set for the original exact name empty.
   `NeuronTableWidget.apply_state()` now blocks table signals during the bulk
   update and restores their previous state afterward.
2. Search append/annotation and Current Table retention reapplied the entire
   original saved snapshot, reverting newer Labels and Tags. Retention now
   keeps survivor state; appending restores saved annotations only for new
   rows. The current shared cluster-assignment registry is preserved as well.
3. A Tags cell could remain uncommitted when Save Project As was clicked,
   especially with an inactive macOS window. `export_state()` now explicitly
   commits and closes the active editor before capturing the snapshot. Moving
   focus alone was insufficient in the visible docked napari check.

`tests/test_project_annotation_persistence.py` adds 13 real Qt/viewer regression
cases. They exercise disk persistence, fresh and previously populated viewers,
filtered saves, duplicate display IDs, exact tag names, live membership and
counts, Show Members, overlap, set-layer snapshots, repeated overwrite/Save As,
post-load edits, new-row initialization, enhanced Parquet, legacy projects, and
active/inactive cell-editor saves, and recovery from an export memory error.
Existing test doubles were updated for the
new editor-state query and new-row initialization arguments.

Serialization already retained the tested annotations and rules. The existing
project format is retained; `project_io.py` now also bounds export resources as
described in the save-crash follow-up below.
Immediate synthetic-project restoration passed; the original immediate-on-load
report cannot be attributed conclusively without its saved state. No recovery
of annotations already missing from an older saved file is claimed.

UC-027 now specifies exact tag, member, count, selection, overlap, overwrite,
and second-cycle expectations. Its complete manual scientific workflow remains
**Not run**; the synthetic GUI check below covers annotation persistence only.

## Save-crash follow-up (2026-11-06)

The user reported napari closing by itself after choosing a save destination,
at **9%**, with a leaked-semaphore warning at shutdown. The first export step
writes the filtered neuron Parquet, before layer saving. The abandoned staging
bundle contained a zero-byte Parquet, and DuckDB left
spill files in the repository's `.tmp` directory.

The canonical source contains **728,703,227 nodes** and is **17.92 GiB** on disk.
DuckDB 1.4.3's defaults on this machine allowed the export to use **28.7 GiB**
and **14 threads**, alongside napari's existing layers. This supports a
memory-pressure diagnosis; the shutdown warning alone does not prove the
operating-system termination cause. The earlier tiny fixtures did not exercise
this scale of export.

`export_filtered_project_neuron_parquet()` now uses a dedicated connection with
a **512 MiB working-memory budget**, **two threads**, and **65,536-row output
groups**. External-sort spills use a unique temporary directory beside the
output, cleaned after the connection closes, including on allocation failure.
The viewer's existing database settings are unaffected. Selected-neuron order,
node ordering, annotations, and schema metadata remain covered by tests.
DuckDB documents external sorting, spill-directory control, and reduced
parallelism in its [1.4 performance guide](https://duckdb.org/docs/1.4/guides/performance/how_to_tune_workloads.html).

New regressions verify the resource policy, actual external sorting of 500,000
nodes under a 64 MiB test budget, and preservation of an existing project after
a deliberately forced DuckDB allocation error. A real Qt test confirms that
save controls recover, annotations and set members remain intact, and the same
viewer can retry successfully after that error.

A read-only-source validation selected **17 neurons** from the first 16 source
row groups' file-ID bounds. It exported **1,166,902 nodes** successfully in
**12.91 seconds** to a **49.49 MiB** Parquet. Profiling recorded **512.21 MiB**
peak DuckDB buffers, **55.31 MiB** peak spill storage, and **1,296.86 MiB** peak
process RSS. The small transient buffer overrun shows why the configured budget
is not a strict whole-process cap. This measurement is for a standalone export,
not napari plus loaded layers or a full-cohort save. The source size and
modification time were unchanged; validation output was temporary and removed.

Only **8.7 GiB** was free on the source/output volume during diagnosis. A
full-source project copy was not attempted; large saves also require room for
their output and temporary sort files. No manual cleanup of crash leftovers
was performed.

## Required behavior

A project round trip preserves the complete saved Data cohort, each neuron's
Labels and Tags, and each set's name, identifier, and annotation rule. Evaluating
those rules after loading gives the same member `file_id` values and counts.
Subsequent annotation edits continue to update membership.

Use `file_id` throughout persistence, matching, selection, and comparisons.
Different neurons with the same display `neuron_id` must remain independent.
Temporary row filters and selected rows must not narrow the saved Data cohort
or redefine a set's membership.

Keep the existing live-set model: persist annotations and rules, then evaluate
membership against the complete restored cohort. A heatmap or mask's recorded
source membership remains its creation-time snapshot; restoring a live set
must not rewrite that provenance.

## Baseline inspection

- `NeuronTableWidget.export_state()` already exports per-row `tags` and `label`
  fields, plus `neuron_sets` definitions. `NeuronEntry.from_state()` parses the
  stored tag list. `import_state()` replaces entries before loading definitions.
- `NeuronSetStore` persists names, identifiers, and rules. Membership is computed
  by `AnnotationRule.members()` from the current table entries, independently
  of row visibility.
- `save_project_bundle()` writes `data/table_state.json` and embeds table state
  in the filtered `data/source_neurons.parquet`. The cluster-assignment filtering
  helper retains other table-state fields, including neuron sets.
- `_restore_project_bundle()` first calls `_load_parquet_path()`, which clears
  and restores optional enhanced metadata, then imports the bundle's table
  state. Project restoration therefore traverses two annotation-loading paths.
- `_saved_table_state` is reapplied by `_apply_saved_table_state_to_table()`
  after table population and Search additions. Investigate whether that stale
  snapshot overwrites annotations edited since project loading.
- `apply_state()` updates editable cells without blocking `itemChanged`.
  `_on_item_changed()` parses the displayed Tags text by commas. Test whether a
  programmatic restore can reinterpret an already serialized tag name that
  contains a comma. This is an inspection finding, not a demonstrated cause of
  the reported empty membership.
- Existing tests check serialized set definitions and widget state round trips,
  but do not exercise a complete `_restore_project_bundle()` round trip into a
  fresh viewer and verify annotations, members, and visible counts together.

At planning time, these observations did not establish whether the reported
loss occurred before saving, in the bundle, during restoration, or during a
later table refresh. Locate that boundary before choosing the patch.

## Implementation sequence

### 1. Reproduce and identify the first loss of state

Create a synthetic Parquet with at least four distinct `file_id` values,
including two subjects sharing a display `neuron_id`. Give three rows the
controlled annotations used by UC-027: A has TH and CP, B has TH, C has CP.
Leave D untagged. Include Label-based, overlapping, excluded-tag, exact-tag,
and genuinely empty sets. Include a separate tag containing punctuation and
a comma to check exact preservation of stored names.

Drive annotation edits through the real table widget. Record and compare:

1. `export_state()` immediately before Save Project As....
2. The saved `data/table_state.json` and the bundled Parquet's enhanced metadata
   and annotation columns.
3. The `ProjectBundle.table_state` returned by `load_project_bundle()`.
4. Entries and definitions after `_restore_project_bundle()` in a fresh viewer,
   after pending Qt events run, and after a subsequent table refresh.
5. Each live rule's sorted member `file_id` values, set-panel counts,
   Show Members selection, and membership-only overlap results.

Start without derived layers to isolate annotations. Then repeat with set
heatmap/mask metadata to check restoration callbacks and preserved source
snapshots. Use a Qt application and napari `ViewerModel` where possible to
exercise real widget/controller state without requiring an OpenGL canvas.
Stub unrelated atlas access or rendering only; do not mock the state export,
save/load, table import, or membership operations under investigation.

If this fixture does not reproduce the report, characterize the failing
workflow using its saved table state, set rules, annotation names, and action
sequence. Distinguish loss immediately after Load Project... from loss after a
query, annotation edit, or Search addition. Do not select a speculative fix
merely because the small round trip passes.

Deliverable: a failing regression at the first identified boundary, or a
recorded passing characterization with the remaining reproduction gap.

### 2. Repair the identified boundary

- If the exported snapshot is incomplete, repair snapshot capture so it uses
  the complete current Data cohort and current annotations, including an edit
  still being committed by the table editor when saving begins.
- If serialized state differs from the exported snapshot, repair the specific
  normalization or export step. Keep JSON and embedded metadata consistent.
- If restoration loses intact serialized state, establish the bundle's table
  JSON as the authoritative project snapshot. Restore row annotations and set
  definitions as one coherent update, then refresh annotation catalogs, set
  counts, selection, and overlap controls from the final state. Ensure Qt
  notifications cannot feed partially restored cell text back into the model.
- If a later population path reverts edits, constrain application of saved
  annotations to the rows and lifecycle that require restoration. Preserve
  edits to existing rows during append or retain operations. Loading a saved
  project must still deliberately restore its saved annotations.
- Retain the distinction between `import_state()` and `apply_state()`:
  importing a project replaces definitions, while applying row annotations
  must not resurrect a set the user deleted. An existing widget test guards
  this behavior.
- Keep older projects without a `neuron_sets` field loadable with no sets.
  Missing legacy tag fields remain empty. Do not require a format migration
  unless the demonstrated defect needs a new stored field; existing fields
  already represent the requested state.

Change the smallest set of functions supported by the reproduction. Likely
locations are `widgets/neuron_table.py` and the project restore/population paths
in `widgets/neuron_viewer.py`; modify `project_io.py` only if the serialization
comparison identifies loss there.

### 3. Add regression coverage through the real save/load path

Extend the existing suites rather than testing a second implementation of
persistence:

| Location | Required checks |
|---|---|
| `tests/test_project_io.py` | Exact Labels, Tags, rules, set IDs, and member IDs survive disk save/load, overwrite, Save As, and a second save/reload cycle. Bundled annotation columns agree with table JSON. |
| `tests/test_neuron_sets_widget.py` | Fresh-widget restoration shows correct counts, Show Members selects expected IDs, and edits still change live membership. Empty sets and hidden members behave correctly. |
| Viewer/controller integration, extending `tests/test_project_io.py` or a focused new test module | Call the real project restore path with real table state. Cover fresh and already-populated sessions, pending Qt events, conflicting old annotations, and the first post-load refresh. |
| Existing overlap widget tests | Membership-only overlap remains A=2, B=2, intersection=1, union=3 for the controlled TH/CP fixture after loading. |

Also cover duplicate display IDs across subjects, several tags per neuron,
tag names that must survive verbatim, active row filters at save time, and a
post-load edit followed by append/retain. Compare membership by `file_id`,
independently of table order. Preserve existing cluster-assignment and
creation-time layer snapshot behavior.

### 4. Verify the user workflow and document the result

Extend **UC-027 step 10** in `USE_CASES.md` with explicit before/after tag values,
member IDs/counts, Show Members selection, and overlap expectations. Include
Save Project As..., Save Project overwrite, Load Project..., a fresh napari
session, and another edit/save/reload cycle. Use UC-026's tag-persistence check
as a related control. Keep the existing use cases instead of adding duplicates.

Record the exact dataset path, cohort, atlas settings when used, tested
`file_id` values, outcome, and date. Leave manual verification as **Not run**
until the workflow has actually been exercised in napari.

## Validation

Planning baseline completed with:

```sh
pixi run test tests/test_neuron_sets.py tests/test_set_overlap.py tests/test_project_io.py -k 'live_sets or live_rules or store_roundtrip or membership_overlap or invalid_rule or enhanced_neuron_parquet_round_trips or save_project_bundle_writes_only_current_table_neurons'
```

Result: **8 passed, 58 deselected**. This selection checks rules, identities,
metadata persistence, and filtered export; it does not validate the reported
napari workflow. No Qt-dependent tests or manual napari checks were run during
planning.

The focused suites are run through pixi:

```sh
pixi run test tests/test_project_annotation_persistence.py tests/test_neuron_sets.py tests/test_set_overlap.py tests/test_project_io.py tests/test_neuron_table_widget.py tests/test_neuron_sets_widget.py tests/test_soma_projection.py
```

Run Qt-dependent checks outside the Codex sandbox with a working Qt environment.
Then run `pixi run test` for integration regressions and exercise UC-027 in
napari. Passing serialization tests alone do not complete this subissue.

Annotation implementation verification on **2026-11-06**, before the resource
follow-up:

- Full-suite run, `pixi run test -q --tb=short`: **1,657 passed, 8 skipped**
  in 99.67 seconds, outside the sandbox. This includes all **12** new persistence
  regression cases, including the inactive-window editor variant and preserving
  the live assignment object when new rows are restored.
- Ruff passes for the new persistence test file and `neuron_table.py`.
  `neuron_viewer.py` has the same 22 lint diagnostics as the inspected baseline;
  the change introduces no new diagnostics.
- A scripted check opened real napari windows in **three separate processes**:
  Save Project As, Load Project followed by an edit and Save Project/Overwrite,
  and Load Project again. It used a five-neuron synthetic source and saved the
  four-row Data cohort (`subject-a/shared`, `subject-b/shared`,
  `subject-c/shared`, `d/empty`), with no atlas loaded. TH/CP counts were 2/2
  after the first fresh reload and 1/2 after the edited project was overwritten
  and reopened. The comma-containing tag/set survived both reloads, empty sets
  stayed empty, and Show Members selected the expected full file IDs.
- GUI source, project, expected-state JSON, and screenshots are temporary local
  artifacts at `/private/tmp/nn-173-gui-se5blf_u/`. The input fixture is defined
  in the new test module. Visual inspection of `load.png` confirmed the restored
  set counts in napari; visual inspection of `verify.png` confirmed the edited
  counts after the second reload. This scripted
  check is separate from UC-027's full manual verification status.
- Exploration also identified an existing failure restoring a generic napari
  default Labels colormap. Plugin-generated binary masks use a direct colormap
  and pass the integration check; generic Labels colormap restoration remains
  a separate project-I/O issue.

Save-resource follow-up verification:

- Final full-suite run, `pixi run test -q --tb=short`: **1,661 passed, 8 skipped**
  in 98.83 seconds, outside the sandbox.
- `pixi run test tests/test_project_io.py tests/test_project_annotation_persistence.py --tb=short`:
  **53 passed**, outside the sandbox, including the 500,000-node spill test and
  real Qt allocation-failure/retry check.
- The standalone real-source export described above completed outside the
  sandbox using DuckDB 1.4.3 and left the input source unchanged.
- Ruff introduced no new diagnostics in `project_io.py` or `tests/test_project_io.py`;
  their respective 11 and 3 findings match the baseline. The new persistence
  test module passes Ruff, and `git diff --check` passes.
