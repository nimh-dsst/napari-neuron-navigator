# napari-neuron-navigator User Manual

## What this plugin does

`napari-neuron-navigator` is a napari plugin for:

- Opening `.swc` neuron morphology files directly
- Opening annotated `.parquet` neuron datasets
- Opening flatmap/depth `.nrrd` lookup files directly
- Filtering neurons by Allen brain regions
- Rendering neurons as lines, points, or both
- Showing Allen reference layers (template, outline, region meshes, segmentation)
- Running clustering and heatmap analysis
- Comparing saved cluster mappings in an interactive 1×1 through 4×4 board

The plugin adds:

- A file reader for `*.swc`, `*.parquet`, and flatmap/depth `*.nrrd`
- A dock widget named **Neuron Viewer**

## Prerequisites

- Python `>=3.10`
- napari `>=0.5`
- `pyqt6`
- BrainGlobe atlas API dependencies (for atlas-backed features)
- Internet access for first-time atlas downloads

This repo is configured to use Pixi.

## Installation and launch

From the repository root:

```bash
pixi run build
pixi run napari
```

When napari is open, launch the widget from the Plugins menu and choose **Neuron Viewer**.

To enable debug-level plugin logging for runtime diagnostics:

```bash
NAPARI_NEURON_NAVIGATOR_DEBUG=1 pixi run napari
```

Optional custom log file:

```bash
NAPARI_NEURON_NAVIGATOR_DEBUG=1 NAPARI_NEURON_NAVIGATOR_LOG_FILE=/tmp/napari-neuron-navigator.log pixi run napari
```

Without an override, logs are written to
`~/.napari-neuron-navigator/debug.log`. The file rotates at 10 MB and retains three
backups. Plugin DEBUG output is written to both the file and console. Focused
napari layer-slicer DEBUG output is captured in the file only.

Project bundles created by Neuron Navigator use `.nnproj`. Existing `.swcv`
project directories created before the rename can still be loaded and saved.

### Detached flatmap window diagnostics

1. Launch napari with `NAPARI_NEURON_NAVIGATOR_DEBUG=1` and a dedicated
   `NAPARI_NEURON_NAVIGATOR_LOG_FILE`.
2. Open the plugin, project data into the detached **Neuron Navigator Flatmap**
   window, use the operating-system control to close that window, and accept
   napari's close confirmation. Cancelling the confirmation must leave the
   existing viewer and layers usable.
3. Wait at least two seconds so the bounded post-close checkpoints run, then
   project again and confirm that exactly one fresh detached window opens.
   Close it again, then close the main napari window and retain the complete
   log.

The trace records `flatmap_viewer_lifecycle event=qt_close`, close checkpoints
at 0, 250, and 2000 ms, `event=qt_deferreddelete`, registered napari viewers,
retained plugin references, flatmap worker states, and Python thread names.
Before these close records, a successful first display records
`event=created_hidden`, `event=first_layer_ready`, `event=show_scheduled`, and
`event=shown`. The window should not become visible before `event=shown`, and
that record should report `pending_first_show=false`.
After destruction, three additional `event=post_destroy_checkpoint` records
capture matching QWidget and native QWindow state. Napari's
`_LayerSlicer.shutdown` records show when full viewer cleanup actually occurs.
A successful accepted close has `event=cleanup_complete`,
`cleanup_trigger=deferred_delete`, `cleanup_status=ok`,
`cleanup_status_thread=stopped` (napari's StatusChecker QThread is stopped
during teardown so the window is never destroyed with it still running),
`cleanup_qt_viewer=closed`, `layer_count=0`,
`napari_viewer_registered=false`, `owner_ref_is_viewer=false`,
`tab_ref_is_viewer=false`, and `slicer_executor_shutdown=true`. At the 2000 ms
post-destroy checkpoint, `qt_matching_top_level_widgets=empty` and
`qt_matching_native_windows=empty`. Any `event=cleanup_failure` record
identifies the teardown stage that could not be completed.

### macOS fullscreen diagnostics

On macOS napari reapplies its app-wide saved geometry — including fullscreen —
to every viewer shown through `Window.show()`, and, on close, runs a blocking
`showNormal`/`processEvents` cycle while the window is fullscreen. The plugin
works around both for the detached viewer. Each lifecycle snapshot reports
`qt_fullscreen`, `qt_window_state`, `napari_saved_fullscreen`, and
`fullscreen_close_state`.

- On show, the guarded normal-window path records `event=normal_show_requested`
  and `event=fullscreen_restore_suppressed`, and `event=shown` reports
  `show_path=normal_qt`. On other platforms `event=shown` reports
  `show_path=napari`. The detached window must open normally even when the main
  napari window is in macOS Full Screen.
- When a manually fullscreened detached window is closed, the close is deferred
  and retried after the window returns to normal: `event=fullscreen_close_deferred`
  → `event=fullscreen_exit_requested` → `event=fullscreen_exit_complete` →
  `event=fullscreen_close_retried`. The retry still presents napari's
  confirmation; cancelling leaves the viewer usable in normal window mode.
- `event=fullscreen_guard_failure` marks any guarded fallback — a missing
  Qt/napari private interface (`reason=confirm_need_unavailable`) or a
  fullscreen-exit timeout (`reason=exit_timed_out`); in the former case the
  window is left open in normal mode and must be closed again.

## Quick validation commands

Run from the repository root:

```bash
pixi run test
pixi run test-cov
```

If you need sample data used by tests/examples:

```bash
pixi run download-test-data
```

## Supported data inputs

### SWC files

- Open through napari File Open dialog (`.swc`)
- The reader renders parent-child segments as a `Shapes` layer (line segments)
- If no valid parent-child edges exist, it falls back to a `Points` layer

### Parquet files

- Open through napari File Open dialog (`.parquet`) or through the widget Data tab
- Initial reader view shows soma points in a red `Points` layer
- Full querying/rendering workflows are done in the **Neuron Viewer** widget

Recommended Parquet schema columns:

- `file_id`, `node_id`, `type`, `x`, `y`, `z`, `radius`, `parent_id`
- `region_id`, `region_name`, `region_acronym`
- `subject`, `neuron_id`

### Flatmap and depth NRRD files

- Open through napari File Open dialog (`.nrrd`) or drag-and-drop
- 4D flatmap lookup volumes with a length-2 coordinate axis are split into
  `Flatmap X: <stem>` and `Flatmap Y: <stem>` image layers
- 3D depth lookup volumes are loaded as `Depth: <stem>` image layers
- Direct NRRD image layers use voxel/pixel space to match the plugin's reference
  image layers
- Unsupported NRRD shapes are rejected instead of being guessed

### Precomputed flatmap Parquet files

- In **Flatmap**, use **Lookup directory...** to select a directory containing
  `flatmap_both_shaped.nrrd`, `flatmap_both_square.nrrd`, and `depth.nrrd`, then
  click **Prepare Whole Parquet...**
- Preprocessing reads every row from the source file, independent of the
  current query or table selection. It preserves custom columns and Arrow
  schema metadata and suggests `<source>_flatmap.parquet`.
- Replacing the source requires explicit confirmation. Work is published by an
  atomic same-directory replacement; cancellation and failure remove temporary
  artifacts and leave an existing destination untouched.
- The version-3 Parquet retains `x`, `y`, and `z`. It adds
  `x_flat_shaped`, `y_flat_shaped`, shaped validity/provenance columns,
  corresponding square columns, and shared `depth_um`, `depth_valid`,
  `depth_invalid_code`, and `depth_lookup_mode` columns.
- Shaped and square XY values always come from the original CCFv3 voxel. If
  direct depth is invalid, depth alone may be read from the voxel mirrored
  across the configured hemisphere axis. An invalid bilateral XY remains
  unmapped.
- Metadata records the lookup-set ID, source NRRD hashes, both style
  definitions, shared depth definition, canonical bounds, transform/resolution,
  column mapping, and algorithm versions.
- Command-line equivalent for the bilateral lookup set:

```bash
pixi run python scripts/add_flatmap_columns_to_parquet.py neurons.parquet neurons_flatmap.parquet --lookup-dir /path/to/lookups
```

The legacy `--flatmap ... --depth ...` single-style command remains supported
for version-1/2 workflows. If the NRRD headers do not contain a usable spatial
transform, bilateral preprocessing requires an explicit lookup resolution; it
does not assume 10 µm.

In **Data** > **Convert SWC to Parquet**, enable **Add bilateral flatmap/depth
columns** and select **Lookup directory...** to combine SWC conversion and
version-3 augmentation in one cancellable, atomic background operation.

### Allen Isocortex layer flatmap projection and stack

- In **Flatmap**, choose **Allen Layer Heatmap** under **Render**. The
  **Allen Layer Options** section starts with `L1`, `L2/3`, `L4`, `L5`, `L6a`,
  and `L6b` checked and **View** set to **2D stack**. Use **Select All** or
  **Clear** for quick selection changes.
- **2D stack** creates one raw node-count plane per checked layer, compacted in
  superficial-to-deep order. **Single 2D projection** sums the checked planes
  into one XY node-count image; it is exactly the selected stack summed over
  its plane axis.
- Load the Allen atlas used to annotate the input and provide a Parquet with
  `region_id`. Membership is derived from terminal regions in the loaded
  atlas's **Isocortex** hierarchy; numeric depth does not assign layers.
- Flatmap-invalid nodes, non-Isocortex annotations, parent regions,
  unannotated nodes, and non-laminar regions are excluded. Valid nodes in
  unchecked Allen layers are counted separately as unselected. Cortical areas
  without layer 4 remain empty on the `L4` plane.
- **Y bins**, shaped/square style, neuron selection, and heatmap color mode
  continue to apply. Depth-bin controls and cached surfaces/outlines are
  unavailable while this categorical mode is selected. If no layer is checked,
  projection, soma, and label actions are disabled until one is selected.
- With **Precomputed Parquet + Cache** and a compatible cache profile active,
  select **Atlas Regions** or **Custom Regions** and click **Show Region
  Labels** to create cache-backed labels matching the current selection and
  view. Stack labels use the same compact layer order and XY grid as the
  heatmap, so napari's first-axis slider changes both layers together.
  Projection labels are one rank-2 image named **Flatmap Region Labels Allen
  Projection**. Labels may also be shown before projecting neurons.
- Planar labels contain an Atlas selection's terminal Allen-layer descendants
  or the exact checked Custom Region terminal IDs. Cached source-voxel counts
  are summed across depth for each region/XY bin; greatest occupancy wins
  collisions, with smaller region ID as the tie-breaker. The viewing path does
  not load NRRDs or `atlas.annotation`.
- Single color creates one selected stack or projection. Individual and cluster
  color modes create one corresponding image per `file_id` or cluster.
- The flatmap window names what is on screen: the plane slider is captioned
  **Allen layer**, the upper-left canvas corner reads the layer of the plane
  under the slider (for example `Allen layer: L5  (plane 2 of 2)`, following
  the slider as it moves), and
  labelled **Flatmap X** / **Flatmap Y** axis arrows are drawn at the image
  origin. **Single 2D projection** is named **Isocortex Flatmap Allen Layer
  Projection** and has only the Flatmap Y/X axes, with no slider or plane
  caption. In **3D Heatmap** the readout reports the plane's depth range in
  microns and the caption reads **Depth bin**. The images are binned in index
  space, so the arrows orient and name the axes without asserting physical
  units or anatomical direction.
- CSV export is available for materialized NRRD recomputation and includes
  canonical `allen_layer_index`, `allen_layer_label`,
  `allen_layer_selected`, and compact `allen_layer_render_index` provenance.
  The fast precomputed, volume-only path remains non-exportable.

### Depth-free 2D flatmap modes

- In **Flatmap**, choose **2D Heatmap** or **2D Vector** under **Render** for a
  plain flatmap with no depth axis at all. Both open the flatmap window in 2D
  with only **Flatmap Y** / **Flatmap X** axes and no plane slider or plane
  caption.
- Collapsing removes the depth *axis*, not any node: **Exclude depth -1 nodes**
  still decides whether depth `-1` nodes are rendered, so a 2D Heatmap is
  exactly the matching **3D Heatmap** summed over its planes. The summary panel
  reports the same node counts plus `Depth: collapsed into one flatmap plane`.
  The depth-bin size control is disabled because there are no depth bins to
  size.
- **2D Heatmap** produces one raw node-count image over the flatmap XY grid.
  **Y bins**, shaped/square style, neuron selection, and **Heatmap colors** all
  apply, so Individual and cluster coloring create one tinted 2D image per
  neuron or cluster. With **Precomputed Parquet + Cache** the binning runs in
  DuckDB, matching the 3D heatmap's speed on whole-Parquet renders.
- **2D Vector** draws one line per parent-child edge, colored per neuron from
  the table, over the same pixel grid the 2D Heatmap uses. Edges are selected by
  the render's own validity, so the traces cover exactly the nodes the matching
  2D Heatmap counts. Vectors always use per-neuron colors, so **Heatmap colors**
  is disabled.
- 2D Vector is a per-node render and is limited to 250,000 segments. Above that
  it reports the count and the limit and draws nothing, rather than silently
  omitting whole neurons; select fewer neurons, or use 2D Heatmap for the whole
  set.
- Cached region labels and outlines are available in both 2D modes, collapsed
  into the same single plane as the render. Cached surfaces are not: they are 3D
  voxel shells with no 2D form. See "Flatmap region cache and overlays" below.

### Flatmap soma overlay

- **Add Soma** projects only soma nodes (`type == 1`) into the separate
  **Isocortex Flatmap Somas** point layer, leaving the main projection layer in
  place.
- The somas are built for the **Render** mode that is currently selected, so
  they land in the coordinate space of the visible render: depth bins in **3D
  Heatmap** / **3D Points**, selected compact planes or XY-only points in
  **Allen Layer Heatmap**, and a single flat plane in **2D Heatmap** / **2D
  Vector**. The
  flatmap window's display dimensionality, axis captions, and plane caption are
  preserved rather than reset.
- **Add Soma** in Allen Layer mode needs a Parquet with `region_id` so somas can
  be filtered to the same selected layers. Without it the action reports the
  requirement instead of falling back to depth bins, which would place the somas
  in a different coordinate space.
- Changing **Render**, the flatmap style, or the cache profile removes the soma
  layer, because its bin coordinates belong to the retired grid. Re-run **Add
  Soma** after switching.

### Flatmap region cache and overlays

- Cache generation records the BrainGlobe atlas name and version as profile
  identity, and requires its resolution and annotation shape to exactly match
  the lookup grid. It memory-maps the atlas annotation and lookup arrays and
  processes them in bounded chunks.
- In **Flatmap**, use **Build Cache Profile...** to choose a cache directory and
  build one shaped/square profile pair. The default fixed grid is 256 Y bins
  with 25 µm depth bins. The X bin count is derived per style from its flat map
  aspect ratio so bins are square, so the two styles in one profile have
  different X counts (512 for square, 491 for shaped).
- The root `flatmap-region-cache.json` manifest can reference multiple profiles.
  Profile identity includes the lookup-set ID, exact atlas provenance, grid
  settings, bounds, output shape, and algorithm versions.
- The manifest is at format version 2. A cache built before per-axis bin counts
  were introduced is refused with a message telling you to rebuild it: its
  arrays were binned on a grid that used one bin count for both flat map axes,
  so they cannot be reinterpreted on the square-bin grid. Rebuilding is a
  one-time cost per cache directory.
- Each style stores sparse region occupancy and source-voxel counts, closed
  voxel-faithful surface triangles, and per-depth XY perimeter vectors. Parent
  region geometry is precomputed as the union of its descendants. Depth-collapsed
  2D footprints are **not** stored; they are unioned from descendant occupancy on
  demand, which measures under 150 ms even for the whole isocortex.
- New arrays are published before the manifest is atomically replaced. A
  cancelled or failed build does not invalidate existing profiles.
- For viewing, choose **Precomputed Parquet + Cache**, click **Choose Cache
  Directory...**, and select a compatible profile. Profile selection locks the
  Y and X bin counts, depth bins, canonical bounds, and exclusion of the depth
  `-1` plane. The stored X count is used verbatim rather than re-derived, so a
  cache-backed render always lands on the profile's own grid.
- Flatmap region actions follow the active **Query source** and **Search
  scope**. For **Atlas Regions**, **Show Region Labels** uses the
  Include-child-expanded IDs while surfaces and outlines use each directly
  selected parent's cached descendant union. For **Custom Regions**, labels,
  surfaces, and outlines use exactly the checked terminal Allen IDs, with one
  acronym/ID-named, atlas-colored geometry layer per terminal region.
- In **3D Heatmap** mode, cached labels use the depth grid; recomputed labels
  also accept the active Atlas or Custom selection. In **Allen Layer Heatmap**
  mode, cached labels collapse the same occupancy arrays into the selected
  compact Allen planes or one projected label image. The planar modes need no
  rebuild of a format-version-2 cache, and surfaces/outlines remain unavailable
  because their geometry is depth-based.
- In **2D Heatmap** and **2D Vector** modes, **Show Region Labels** builds one
  `Y bins × X bins` image named **Flatmap Region Labels 2D**, and **Show Region
  Outlines** builds one **Flatmap Region Outlines 2D** vector layer per directly
  selected region. Both are derived at read time from the same occupancy
  arrays; a format-version-2 cache is reused unchanged and the manifest is not
  rewritten.
- Both 2D overlays sum source-voxel counts across depth **per selected region**
  before the majority vote. Selecting `Isocortex` therefore gives one solid
  footprint, and selecting `MOp` plus `SSp` gives an area map with a shared
  boundary. Collapsing the terminal *layer* regions instead would turn every
  cortical column into a contest between its own layers: measured on the 25 µm
  cache, 43,863 of 43,967 columns are contested and the thickest layer wins — L5
  in 53% of columns and L2/3 in 33%. That reads as a layer patchwork rather than
  anatomy, which is why **Allen Layer Heatmap** provides selectable layer stacks
  and projections for per-layer questions.
- Combining more than one directly selected region into a 2D overlay needs a
  loaded matching structure catalog, because the plugin has to decide which
  cached region belongs to which selection. A single selected region, and any
  selection of terminal regions that name their own labels, needs no catalog.
- **Show Region Surfaces** stays disabled in the 2D modes. The cached surface is
  a 3D exposed-face voxel shell with no 2D form; use **Show Region Labels** for a
  filled 2D region, which also gives per-pixel value readout.
- The 2D overlays cover depth-valid source voxels only, because every cache
  profile records `excludes_depth_minus_one_plane: true`. Selecting a cache
  profile also checks and locks **Exclude depth -1 nodes**, so the 2D heatmap
  under the overlay counts the same nodes.
- The 2D overlays set no layer `scale`, matching the 2D heatmap and 2D vector
  layers, so they land on the same pixels even though the flat map's X axis spans
  twice the range of its Y axis in the same number of bins.
- Region outlines are drawn as unstyled line segments in both the depth and
  collapsed modes, so a perimeter reads as a boundary rather than as a field of
  direction arrows.
- Label paths preserve majority-region collision handling with smaller-ID
  tie-breaking. **Mask Layer** is not an atlas-region source and reports an
  actionable error instead of reusing a retained Atlas selection.
- A loaded atlas with the same atlas/version structure catalog supplies region
  selection and colors during viewing. Its voxel resolution may differ because
  the precomputed path does not read `atlas.annotation` or meshes.
- Missing, corrupt, or incompatible profiles show the exact mismatch and never
  trigger conversion. Choose **Recompute from NRRDs** explicitly for the legacy
  runtime path.
- Project bundles store the external cache path and profile ID but do not copy
  cache arrays. Enhanced and project-filtered Parquets retain version-3 schema
  metadata.

### Standard point Parquet files

- Produced from raw CSVs using `scripts/convert_point_csv.py`
- Imported through the widget Data tab with `Import Point Parquet...`
- Required columns: `label`, `x`, `y`, `z`
- Optional columns: `region_name`, `acronym`, `id`, `hemisphere`
- Any additional columns are preserved in the standardized Parquet

## Typical workflow

### 1. Open Neuron Viewer

Open the **Neuron Viewer** dock widget.

Tabs:

- Data
- Regions
- Visualization
- Flatmap
- Reference
- Analysis
- Compare
- Tools
- Histogram

### 2. Load data (Data tab)

In **Parquet Data**:

1. Click `Load...`
2. Choose a `.parquet` file
3. Confirm stats appear (nodes/files/subjects/regions)

In **Atlas**:

1. Choose atlas (`allen_mouse_10um`, `25um`, or `50um`)
2. Click `Load Atlas`

In **Point Parquet Import**:

1. Click `Import Point Parquet...`
2. Choose a standardized point `.parquet` file
3. The app adds one heatmap image layer per unique `label`, using a different color for each layer
4. If optional atlas metadata columns are present, the app checks them against the selected atlas and warns on mismatches

Notes:

- Atlas features need a loaded atlas
- First atlas load may download data and take time
- The template toggle attempts to load atlas automatically
- If atlas download fails, see Troubleshooting below

### 3. Select regions (Regions tab)

1. Choose `Query source`: `Atlas Regions`, `Custom Regions`, or `Mask Layer`
2. For `Atlas Regions`, use search to find structures, check regions in the
   tree, and optionally enable/disable `Include child regions`
3. For `Custom Regions`, first load a compatible Allen atlas. Expand
   `Isocortex Layers`, then check the root, a layer group (`L1`, `L2/3`, `L4`,
   `L5`, `L6a`, or `L6b`), or individual terminal regions. Searches match the
   full atlas name, acronym, or numeric region ID. Group checks select exactly
   the terminal leaves shown beneath them; partially selected groups display a
   partial check state.
4. Choose `Search scope`: `Whole Parquet` or `Current Table`. Atlas and custom
   selections are kept separately for the two scopes.
5. Choose `Node types`; leave `All node types` selected for the previous
   any-node behavior, or select one or more of `Soma`, `Axon`,
   `Basal dendrite`, and `Apical dendrite`
6. For `Mask Layer`, check one or more generated mask layers
7. Click the query button for the active source

This fills the **Selected Neurons** table in the Data tab.

`Custom Regions` currently contains only the atlas-derived
`Isocortex Layers` hierarchy. Its leaves are the terminal Allen Isocortex
regions used by the flatmap layer classifier; parent atlas regions and
unlisted descendants are never added to the query. A `region_id` column is
required in the loaded Parquet. When `Custom Regions` is the active query
source, the **Reference** tab's selected-region controls use the active custom
scope: meshes are combined into at most one `Region: Custom <layer>` surface
per selected Isocortex layer, while `Region Segmentation` contains exactly the
checked terminal IDs. In **Flatmap**, the label, cached surface, and cached
outline actions also use those exact terminal IDs; geometry remains one
atlas-colored layer per terminal region rather than a synthetic layer union.
Flatmap overlays update only when their **Show** action is clicked. Switching
back to `Atlas Regions` restores include-child labels and parent-union cached
geometry without discarding either selector's retained checks.

### 4. Generate blurred heatmaps (Tools tab)

1. Select one or more eligible heatmap layers
2. Choose `Separate layers` or `Merged layer`
3. Set Gaussian sigma in atlas voxels
4. Click `Create Blurred Layer`
5. To isolate heatmaps after they already exist, use `Region Isolation`: choose
   `Separate layers` or `Merged layer`, select one or more atlas regions, choose
   whether child regions are included, and click `Create Isolated Heatmap`

Notes:

- Only app-created heatmaps on the native atlas voxel grid are eligible
- Depth-binned analysis heatmaps are excluded from mask creation in this version
- Generated blurred layers are also eligible native-grid sources in both the `Tools` and `Histogram` tabs
- Generated region-isolated heatmaps are new native-grid image layers and are also eligible sources in the `Tools` and `Histogram` tabs
- Gaussian sigma is in atlas voxels; for example, on `allen_mouse_25um`, `1.0` means `1 voxel = 25 µm`

### 5. Inspect histograms and build masks (Histogram tab)

1. Select one or more eligible heatmap or blurred heatmap layers
2. Review the overlaid interactive histogram traces for the selected layers
3. Optionally enable `Include zero-valued background` to inspect the zero spike
4. Choose `Separate layers` or `Merged layer` for mask creation
5. Set a `Lower bound`, optionally enable `Use upper bound`, and use either `Use Otsu Lower` or `Use Layer Contrast` to populate the bounds
6. Click `Create Mask Layer`

Notes:

- Generated masks are added as binary `Labels` layers and can be used from the `Regions` tab
- Mask-layer queries use the current nonzero voxels in the mask, including manual edits made after creation
- If the mask came from a heatmap with tracked source neurons, the `Regions` tab shows their count and can exclude them from mask-layer query results
- `Use Layer Contrast` requires exactly one selected heatmap or blurred heatmap layer
- Zero-valued background voxels are hidden from the histogram by default
- The histogram plot supports zooming and panning for tighter threshold inspection

### 6. Manage selected neurons (Data tab table)

In the table you can:

- Select rows to choose which neurons to render
- Toggle per-neuron visibility (`Vis`)
- Edit per-neuron color (color swatch)
- Inspect subject and every saved cluster-assignment column; blank assignment
  cells mean that neuron was outside the run or was not assigned

Cluster assignment controls:

- `Cluster assignment` chooses the active saved run used by the cluster filter,
  summary, rendered-neuron colors, flatmap grouping, and Analysis heatmaps
- `Rename...` changes the display name without changing its portable Parquet
  column name
- `Delete` removes only the saved assignment after confirmation; it never
  removes neuron rows, and the newest remaining assignment becomes active
- `Export Enhanced Parquet...` writes every assignment as a nullable integer
  column and keeps the active assignment in `cluster_assignment` for older
  readers

Buttons:

- `Add Full Neurons`: add currently selected neurons to the scene while keeping neurons already added.
- `Add Soma Only`: add currently selected neurons to the scene as soma-only, reusing the shared soma layers without full trace geometry.
- `Remove Selected`: remove only currently selected neurons from the scene.
- `Clear All`: remove all neuron layers.
- `Add Heatmap`: open a menu for creating native-grid node-count heatmaps from
  the currently selected table neurons.
  - `Single Heatmap`: create one `hot` heatmap containing all selected neurons.
  - `Individual Heatmaps`: create one Greek-named heatmap per selected neuron,
    colored to match that neuron. If two or more selected neurons initially
    have the same RGB color, they are first assigned distinct colors from the
    shared neuron palette; other table neurons are not recolored. Each individual layer initially
    displays a contrast range from zero to 20% of its maximum node count so
    lower-density structure remains visible; this can still be adjusted in
    napari's layer controls.
  - Individual heatmaps are built sequentially. If their estimated float32
    image data exceeds 1 GiB, a confirmation defaults to `Cancel` before any
    colors or layers change.
- `Remove Selected From Table`: remove currently selected neuron rows from the table while keeping rendered scene layers.
- `Remove Unselected from Table`: keep selected table rows and remove all other table rows, including rows hidden by active filters.
- `Clear Table`: clear the table while keeping rendered scene layers.
- Centering mode (`Same` | `Auto`): `Same` keeps the current depth slice after add/remove. `Auto` auto-centers once on first add/remove in the widget session.

### 7. Configure rendering (Visualization tab)

Controls:

- Render mode: `Lines`, `Points`, `Both`
- Point size and optional `Color by node type`
- Line width
- Opacity slider
- Shared 2D Slice Projection toggle (`Show in 2D slices`) and slice thickness (µm)

Rendering behavior:

- Batch rendering is used (single layers for performance): `Neuron Lines` (`Shapes`), `Neuron Points` (`Points`), and `Soma Labels` (`Points`, labeled by `neuron_id`)
- `Add Selected` contributes to the full-trace layers based on the current Visualization render mode, while `Add Soma Only` contributes only to the shared soma layers
- Mixed scenes are supported: some neurons can remain full traces while others are rendered soma-only at the same time
- In 2D mode, line/point layers are auto-hidden to keep navigation responsive
- Optional `Neuron Slice Projection` (vectors layer) shows line segments near current slice
- Optional `Soma Slice Projection` (points layer) shows raw soma/body nodes (`type = 1`) near the current slice using the same shared slice thickness
- In 2D with projection enabled, `Soma Labels` are hidden to avoid duplicate soma markers, and `Points` render mode suppresses raw soma-node duplicates from `Neuron Points`

### 8. Add reference context (Reference tab)

Available layers:

- `Show template` -> `Allen Template` image
- `Show brain outline` -> `Brain Outline` surface
- `Show selected region meshes` -> `Region: <acronym>` surfaces
- `Show selected region segmentation` -> `Region Segmentation` labels layer

Notes:

- Mesh display auto-switches napari to 3D mode
- Segmentation is intended for 2D context overlays
- With **Custom Regions** active, selected terminal meshes are grouped into at
  most six `Region: Custom <layer>` surfaces, preserving each terminal
  region's Allen color. Segmentation includes only the exact checked terminal
  IDs and does not expand descendants.

### 9. Run analysis (Analysis tab)

### Clustering

Method options:

- `Voxel Correlation`
- `Soma Location`

For Soma Location, algorithm options:

- `Hierarchical`
- `K-Means`
- `DBSCAN`

Common controls:

- Input neurons: `Whole Parquet`, `Current Table`, or `Selected Rows`
- Target region: required for CCFv3 `Whole Parquet`; optional for CCFv3
  `Current Table` and `Selected Rows`
- Dilation %
- Clusters (for non-DBSCAN modes)
- Linkage method (hierarchical modes)
- DBSCAN `eps` and `min_samples` when DBSCAN is selected

The three CCFv3 input scopes retain independent target-region selections.
When `Current Table` or `Selected Rows` shows `All regions (optional)`, the run
uses all nodes in that table cohort. Selecting a target region restricts that
cohort with the existing dilation setting. `Whole Parquet` does not run until
at least one represented target region is selected. `Selected Rows` requires
at least two visible selected table rows. Flat map + Depth clustering supports
all three input scopes but does not expose region controls.

Run:

1. Click `Run Clustering`. Each completed run creates and activates a new named
   assignment column, such as `Soma Location 1` or `Voxel Correlation 1`, while
   retaining earlier columns.
   Before clustering starts, the plugin counts the node rows that will actually
   contribute. Runs over 10,000,000 nodes show the exact formatted count and ask
   you to **Continue** or **Cancel**; cancellation creates no assignment.
2. Optional: expand `Clustermap` and click `Build Dendrogram` to render the dendrogram and cluster map
3. Click `Color Neurons by Cluster` to recolor rendered neuron layers

Projects and Enhanced Parquet preserve assignment labels, palettes, cohorts,
lineage, and run provenance. Projects do not store large clustering matrices,
so dendrogram and distance exports show `Rerun required` after reload; filters,
colors, grouping, and cluster heatmaps remain available.

### Comparison Board (Compare tab)

Use **Compare** → **Open Comparison Board** to open a separate lightweight
pyqtgraph window for side-by-side cluster comparisons. The board belongs to the
current project and supports 1×1 through 4×4 layouts. It coexists with the main
napari window and the detached flatmap viewer; it does not create mini-napari
viewers or duplicate full 3D scenes.

Each cell can show:

- **Flatmap somas** for one saved assignment
- **Flatmap arbor heatmap**, built in a background worker and cached by source
  Parquet signature, stable assignment ID, flatmap style, exact grid, and
  filters
- **CCFv3 somas** in coronal, sagittal, or horizontal view
- **Existing CCFv3 heatmap(s)** created in **Analysis**, either one layer or a
  metadata-compatible cluster set

Select a cell by clicking it, edit its **Title**, **Map**, **Assignment** or
**Heatmap source**, projection settings, point size, opacity, intensity range,
and navigation option, then click **Apply and Render**. **Add Cell**,
**Duplicate**, **Move Earlier**, **Move Later**, and **Remove** arrange recipes
within the current grid.

For CCFv3 cells, **Full projection** shows every soma or sums every heatmap
voxel through the hidden axis. **Slice / slab** uses **Slice position** and
**Slab thickness** in micrometers. Compatible slice cells share physical slice
position when their navigation is linked. Coronal, sagittal, and horizontal
cells use anatomical axes and retain physical spacing for binned, non-cubic
heatmap arrays.

Flatmap cells follow the same rectangular-grid policy as the Flatmap tab. The
user chooses **Y bins**; **X bins** is derived once from the stored bounds and
saved in the cell recipe. A restored recipe always reuses that exact X count.
For canonical bounds, 256 Y bins gives 491 X bins for **Both hemispheres,
shaped** and 512 for **Both hemispheres, square**.

Set **Reference** to the assignment whose saved colors should define the visual
comparison. Other assignments are matched by maximum overlap over shared
`file_id` membership. This affects display colors only: original cluster IDs,
saved palettes, and assignments remain unchanged, and the selected cell legend
shows the mapping. Unmatched split or sparse clusters receive separate colors.

The napari **Compare** tab also contains **Cluster Membership Comparison**.
Choose a **Reference assignment** and a different **Candidate assignment** to
see Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), optimally
matched agreement, cohort coverage, the cluster overlap matrix, and per-pair
Jaccard similarity. The tab's reference is synchronized with the board's
color-matching reference. Metrics align neurons only by `file_id`. ARI, NMI,
and matched agreement use the shared cohort assigned in both runs; separate
coverage counts show neurons assigned in only one run, unassigned in both, or
present in only one assignment cohort. Jaccard cluster sizes use the complete
shared cohort, so one-sided unassignment reduces the score.

Pan/zoom is linked automatically only between cells with compatible coordinate
provenance; flatmap and CCFv3 cells never link. Uncheck **Link when compatible**
for a per-cell navigation override. Comparable heatmaps share one count range
while **Share comparable intensity** is enabled. **Use per-cell maximum** opts
one cell out and adds an override badge.

Every cell reports assigned and omitted/unassigned neuron counts. Hover shows
coordinates and heatmap count. Point picking and brush selection are not
available in the first version; use the full napari and flatmap viewers for
neuron-level inspection.

Only newly generated Analysis heatmaps carry the stable source, assignment,
atlas, and normalized filter metadata needed by the board. If a referenced
assignment or heatmap is missing or incompatible, the cell shows **Source
unavailable** and is never silently rebound by layer or assignment name. Use
**Refresh Sources** after creating or removing Analysis heatmaps.

**Export Board...** writes a labeled PNG and a neighboring
`.comparison.json` sidecar containing the complete board recipe, source
signature, cohort counts, color-matching table, intensity maxima, and cell
provenance. The sidecar also stores one membership-comparison record per unique
non-reference assignment on the board, including ARI, NMI, matched agreement,
the overlap matrix, coverage counts, Jaccard matches, and contributing cell
IDs. Missing stable assignment IDs are recorded as unavailable rather than
replaced by a same-named assignment. Project bundles save the board recipe and
referenced Analysis heatmap layers. Older projects load with an empty board.
Enhanced Parquet does not store comparison-board state.

### Node Count Heatmap

Controls:

- Region filter (optional)
- Cluster filter (enabled after clustering)
- Node types (`All node types`, or any combination of soma, axon, basal dendrite, and apical dendrite)
- Soma radius (`0 µm` disables this filter; positive values count only selected node types within that distance of each neuron's soma centroid)
- Depth bin factor

Run:

1. Click `Build Heatmap Volume`
2. A heatmap image layer is added (cluster-specific or full population)

## SWC to Parquet conversion (inside plugin)

Data tab, **Convert SWC to Parquet**:

- `From Directory...` recursively finds `*.swc`
- `From Files...` selects explicit SWC files
- Set atlas annotation resolution (10-100 µm, default 25)
- Optionally enable **Add bilateral flatmap/depth columns** and select a
  **Lookup directory...** containing the shaped, square, and depth NRRDs
- Choose output `.parquet`

Conversion runs in a background worker with progress updates. With flatmap
augmentation enabled, both phases form one cancellable atomic pipeline.

## CSV to standard point Parquet conversion

This repo also provides a converter for atlas-registered point CSV files:

```bash
python scripts/convert_point_csv.py raw_points.csv mapping.json points.parquet
```

Mapping JSON format:

```json
{
  "label": "marker",
  "x": "atlas_x",
  "y": "atlas_y",
  "z": "atlas_z",
  "region_name": "region_name",
  "acronym": "acronym",
  "id": "id",
  "hemisphere": "hemisphere"
}
```

Notes:

- Mapping keys are the standard target column names
- `label`, `x`, `y`, and `z` mappings are required
- The converter preserves any unmapped source columns in the output Parquet
- The converter validates numeric `x`, `y`, `z`, and `id` values before writing output

## Command-line hemisphere utility

This repo also provides a script for hemisphere detection and coordinate flipping:

```bash
python scripts/flip_swc.py input.swc -o output_flipped.swc
```

Useful flags:

- `--atlas allen_mouse_10um`
- `--coord-axis 0|1|2` (default 2)
- `--midline <microns>`
- `--info-only`
- `--list-atlases`

## Troubleshooting

### “Atlas failed to load”

- Check internet access for initial atlas download
- Try another atlas resolution (e.g. `allen_mouse_25um`)
- Restart napari after dependency changes

### “No neurons found in selected regions”

- Confirm region acronyms exist in your atlas
- Verify the selected regions match dataset `region_id` / `region_acronym`
- Toggle `Include child regions` and retry
- Confirm parquet stats report non-zero nodes/files/regions

### Clustering errors or empty results

- Ensure both a Parquet file and atlas are loaded
- Use a region containing enough neurons/somas
- For DBSCAN, increase `eps` or reduce `min_samples` if clusters are not found

### “Point Parquet import failed”

- Confirm the file contains standardized `label`, `x`, `y`, `z` columns
- Re-run `scripts/convert_point_csv.py` if you started from a raw CSV
- Load the target atlas before importing point Parquet
- If the app warns about metadata mismatches, the heatmap layers are still imported; review optional `region_name`, `acronym`, `id`, and `hemisphere` values

### “No eligible heatmaps” in Tools or Histogram

- Only app-created native-grid heatmap and blurred heatmap image layers are listed
- The heatmap must match the currently loaded atlas
- Depth-binned analysis heatmaps are excluded because they are not on the native atlas voxel grid

### “No neurons found” for a mask query

- Confirm the selected mask layer contains nonzero voxels
- Try `Any node in mask` before `Soma in mask`
- Check that the mask was created from heatmaps for the same atlas resolution as the loaded dataset

### Slow 2D slicing

- Keep `Show in 2D slices` disabled unless needed for either neuron lines or soma/body-point projection
- Reduce rendered neuron count
- Use `Lines` or `Points` selectively instead of `Both`

## Notes on coordinates and scale

- Node coordinates are expected in microns
- Some layers are displayed in atlas voxel space and use internal scaling for alignment
- For analysis/annotation quality, SWC coordinates should already be aligned to Allen CCF space
