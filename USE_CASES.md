# Neuron Navigator Use Cases

This document describes what the Neuron Navigator plugin can do through concrete,
repeatable user workflows. Each use case also serves as a manual test case for
confirming that the documented capability works in napari.

Use cases are added as workflows are described and refined. They document
observable behavior rather than implementation details.

## Running the Manual Tests

Unless a use case says otherwise:

1. From the repository root, launch napari with the development build:

   ```bash
   pixi run napari
   ```

2. Open the **Neuron Navigator** plugin from napari's **Plugins** menu.
3. Follow the selected use case from a clean napari session.
4. Record the result in that use case's **Manual verification** section. A use
   case remains `Not run` until someone actually performs it in napari.

## Use-Case Index

| ID | Capability | Manual status |
| --- | --- | --- |
| [UC-001](#uc-001-download-an-allen-mouse-atlas) | Download and cache a supported Allen Mouse Brain Atlas | Not run |
| [UC-002](#uc-002-convert-swc-files-to-parquet) | Convert a folder or selected SWC files into one Parquet file | Not run |
| [UC-003](#uc-003-prepare-a-whole-neuron-parquet-for-flatmap-viewing) | Append bilateral shaped/square flatmap and depth coordinates to a whole Parquet | Not run |
| [UC-004](#uc-004-build-and-reuse-a-flatmap-region-cache) | Build, reopen, parse, and switch shaped/square region-cache data | Passed |
| [UC-005](#uc-005-view-selected-allen-isocortex-layers-as-a-flatmap-projection-or-stack) | View selected Allen-layer node counts as one projection or a compact stack | Not run |
| [UC-006](#uc-006-inspect-and-query-custom-isocortex-layer-regions) | Inspect and query exact terminal regions grouped by Isocortex layer | Not run |
| [UC-007](#uc-007-refine-and-save-multiple-cluster-assignments) | Preserve a soma clustering and refine selected neurons with a second method | Not run |
| [UC-008](#uc-008-create-combined-individual-and-enhanced-neuron-heatmaps) | Create combined or individual heatmaps and enhance fine projections in selected layers | Not run |
| [UC-009](#uc-009-save-and-overwrite-the-current-project) | Save changes back to the current Neuron Navigator project safely | Not run |
| [UC-010](#uc-010-identify-axon-termini-and-prune-neurons-lacking-them) | Locate termini as childless axon-typed nodes, then select and remove the neurons that have none (see the annotation caution) | Partially run |
| [UC-011](#uc-011-view-a-depth-free-2d-flatmap-and-per-neuron-vector-traces) | View a plain flatmap with no depth axis as a 2D heatmap or per-neuron vector traces, and place somas in the current render's space | Not run |
| [UC-012](#uc-012-balance-cortical-depth-against-flat-map-position-when-clustering-somas) | Cluster somas with an isotropic flatmap metric and an explicit cortical-depth weight | Partially run |
| [UC-013](#uc-013-overlay-cached-brain-regions-on-a-2d-flat-map) | Overlay cached region fills and outlines on depth-free flatmap renders | Not run |
| [UC-014](#uc-014-control-and-share-region-appearance-across-ccfv3-and-flatmap-views) | Assign, stage, share, save, and restore region colors and visibility across CCFv3 and flatmap overlays | Not run |
| [UC-015](#uc-015-compare-cluster-assignments-in-an-interactive-board) | Compare flatmap and CCFv3 cluster mappings side by side in a linked grid | Not run |

### UC-001: Download an Allen Mouse Atlas

**Capability**

The user can download the Allen Adult Mouse Brain Atlas at the resolution
needed for their data. The plugin supports the BrainGlobe atlas identifiers
`allen_mouse_10um`, `allen_mouse_25um`, `allen_mouse_50um`, and
`allen_mouse_100um`.

The underlying atlas is the Allen Institute's Adult Mouse Brain Atlas, which
provides the commonly used CCFv3 coordinate space. The plugin obtains it
through the [BrainGlobe Atlas API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html),
which standardizes the Allen data into a reference image, anatomical annotation
volume, region hierarchy, region meshes, and metadata. BrainGlobe distributes
the packaged atlas archives from its
[dedicated GIN atlas repository](https://gin.g-node.org/BrainGlobe/atlases).
The plugin downloads from that BrainGlobe repository rather than directly from
the Allen Institute during this workflow.

These files are not bundled with the plugin because the volumetric images and
meshes are large and users may need different resolutions. The plugin needs the
selected atlas locally to place registered SWC morphologies and point data in a
shared anatomical coordinate space, identify the brain region at a coordinate,
display reference images, outlines, and region meshes, and support region-aware
queries and analyses. BrainGlobe caches each downloaded atlas so later sessions
can reuse it without another download. By default, the cache is under
`~/.brainglobe`, although the location can be changed in the BrainGlobe
configuration; see BrainGlobe's
[atlas file documentation](https://brainglobe.info/documentation/brainglobe-atlasapi/usage/using-the-files-directly.html).

**Prerequisites**

- The computer has an internet connection and enough free disk space for the
  selected atlas.
- The user can write to the BrainGlobe cache directory.
- At least one of `allen_mouse_10um`, `allen_mouse_25um`,
  `allen_mouse_50um`, or `allen_mouse_100um` is not already present in the
  configured BrainGlobe cache.
- Start from a clean napari session with the **Neuron Navigator** plugin open.

**Steps and expected results**

1. **Action:** Open the **Data** tab and expand **Atlas**.
   **Expected:** The atlas selector offers `allen_mouse_10um`,
   `allen_mouse_25um`, `allen_mouse_50um`, and `allen_mouse_100um`, with a
   **Load Atlas** button and an atlas status area.
2. **Action:** Select an Allen mouse atlas that is not in the local BrainGlobe
   cache, then click **Load Atlas**.
   **Expected:** The plugin checks the cache, displays a text warning that the
   selected atlas is not cached, and asks whether to download it. The prompt
   identifies the selected atlas and explains that it will be downloaded via
   BrainGlobe.
3. **Action:** Cancel the download prompt.
   **Expected:** No atlas is downloaded or loaded. The Atlas controls remain
   available, and the status text makes it clear that the download was
   cancelled.
4. **Action:** Click **Load Atlas** again and confirm the download when
   prompted.
   **Expected:** The plugin starts the download through BrainGlobe. The Atlas
   controls are disabled during the operation, status text identifies the
   current phase, and the progress bar reports download progress before showing
   that the atlas is being installed in the local cache.
5. **Action:** Wait for the download and installation to finish.
   **Expected:** The Atlas controls are enabled again, the progress indicator
   closes, and the status identifies the loaded atlas and its number of
   structures. The plugin reports that the atlas is loaded and that its
   template, outline, or selected region meshes can be shown from the
   **Reference** tab.
6. **Action:** Close napari, start a new session, select the same atlas, and
   click **Load Atlas**.
   **Expected:** The plugin finds and loads the cached atlas without showing the
   download prompt or downloading the files again.

Repeat this use case for each supported resolution that needs to be available
locally.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: The uncached-atlas warning and confirmation prompt are tracked in
  `TODO.md`.

### UC-002: Convert SWC Files to Parquet

**Capability**

The user can combine multiple SWC morphology files into one Parquet file from
the **Data** tab. The Parquet stores each SWC node with its source filename,
node type and topology, coordinates, radius, filename-derived neuron metadata,
and Allen atlas region information. This format lets the plugin load and query
large groups of registered neurons as one dataset and supports its downstream
visualization and analysis tools.

The **Convert SWC to Parquet** section offers two ways to choose the input. Both
produce one Parquet file, but they differ in which SWCs are included:

| Input action | Files included | Use it when |
| --- | --- | --- |
| **From Directory...** | Every file with an `.swc` extension, including uppercase variants such as `.SWC`, in the selected folder and all of its subfolders | A folder represents the complete dataset to convert |
| **From Files...** | Only the files explicitly selected in the multi-file picker; other SWCs in the same folder or its subfolders are not discovered automatically | Only a particular group or subset of SWCs should be converted |

`Resolution (μm)` selects the Allen atlas resolution used to annotate node
coordinates with brain regions. `Hemisphere alignment` controls whether source
coordinates are kept as-is (`None`) or morphologies on the opposite side are
reflected into a common `Left` or `Right` hemisphere. The atlas resolution and
the coordinate space of the SWCs must match for region annotations and
hemisphere alignment to be meaningful.

The optional **Add bilateral flatmap/depth columns** setting chains flatmap
preprocessing onto the same background conversion. **Lookup directory...**
selects a directory containing `flatmap_both_shaped.nrrd`,
`flatmap_both_square.nrrd`, and `depth.nrrd`. When enabled, the output is a
version-3 Parquet that can be rendered without reading the NRRDs again.
**Lookup resolution** normally reads the transform from the NRRD headers; enter
an explicit micrometer value when those headers do not contain a usable
transform. **Cancel conversion** cooperatively stops either phase and prevents
a partial output from being published.

**Prerequisites**

- Complete UC-001 for `allen_mouse_25um` so the atlas needed by this test is
  available locally.
- Use valid SWC files registered to the Allen CCF in micrometers.
- To test the optional chained preprocessing, prepare a lookup directory whose
  shaped, square, and depth NRRDs have the same spatial grid and transform.
- Prepare this input layout, where each SWC has a distinct filename:

  ```text
  swc_batch/
  ├── cell_a.swc
  ├── cell_b.swc
  ├── notes.txt
  └── nested/
      └── cell_c.SWC
  ```

- Start from a clean napari session with the **Neuron Navigator** plugin open.

**Steps and expected results**

1. **Action:** Open the **Data** tab and expand **Convert SWC to Parquet**.
   **Expected:** The section shows **From Directory...**, **From Files...**,
   `Resolution (μm)`, `Hemisphere alignment`, **Add bilateral flatmap/depth
   columns**, **Lookup directory...**, `Lookup resolution`, **Cancel
   conversion**, a status area, and a progress indicator that is initially
   hidden.
2. **Action:** Set `Resolution (μm)` to `25` and `Hemisphere alignment` to
   `None`.
   **Expected:** Conversion will use the 25 μm Allen annotation while preserving
   the SWC coordinates.
3. **Action:** Click **From Directory...**, select `swc_batch`, and save the
   output as `directory.parquet` when **Save Parquet File** opens.
   **Expected:** The plugin recursively searches the selected folder, discovers
   `cell_a.swc`, `cell_b.swc`, and `nested/cell_c.SWC`, and ignores `notes.txt`.
   The progress indicator and status text report discovery and conversion of
   three files.
4. **Action:** Wait for directory conversion to finish.
   **Expected:** `directory.parquet` is created. The progress indicator closes,
   and the status reports `Done! Converted 3 file(s) -> directory.parquet`.
5. **Action:** Click **From Files...**, multi-select only `cell_a.swc` and
   `cell_b.swc`, and save the output as `selected_files.parquet`.
   **Expected:** The conversion starts with two explicitly selected files. It
   does not discover `nested/cell_c.SWC` or include `notes.txt`.
6. **Action:** Wait for selected-file conversion to finish.
   **Expected:** `selected_files.parquet` is created. The progress indicator
   closes, and the status reports
   `Done! Converted 2 file(s) -> selected_files.parquet`.
7. **Action:** Expand **SWC Parquet Data**, click **Load...**, and open each new
   Parquet in turn.
   **Expected:** The statistics for `directory.parquet` report `Files: 3`,
   while the statistics for `selected_files.parquet` report `Files: 2`. The
   difference confirms that directory selection is recursive and file selection
   is explicit.
8. **Action:** Start either input-selection flow again, but cancel the input
   picker or the **Save Parquet File** dialog.
   **Expected:** No conversion starts and no additional Parquet file is created.
9. **Action:** Enable **Add bilateral flatmap/depth columns**, click **Lookup
   directory...**, select the prepared lookup directory, leave `Lookup
   resolution` at `From NRRD header`, convert the three-file directory again,
   and save it as `directory_flatmap.parquet`.
   **Expected:** SWC conversion and flatmap augmentation run as one progress
   operation. The published Parquet contains all three neurons plus shaped and
   square bilateral flatmap columns and shared depth columns. No intermediate
   Parquet remains beside the output.
10. **Action:** Start the chained conversion once more and cancel while it is
    running.
    **Expected:** The worker stops, temporary artifacts are removed, any
    existing output remains unchanged, and all conversion controls become
    available again.

If an individual SWC cannot be parsed, the converter skips it, keeps processing
the valid files, and includes the skipped count in the completion status. If no
SWC files are found or none can be processed successfully, the status displays
an error instead of reporting successful conversion.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-003: Prepare a Whole Neuron Parquet for Flatmap Viewing

**Capability**

The user can preprocess every node in an existing neuron Parquet once and save
bilateral shaped-flatmap, square-flatmap, and shared depth coordinates in a
version-3 Parquet. The output retains the original CCFv3 coordinates, custom
columns, and schema metadata. Later flatmap viewing reads the stored columns
and does not need to load or query the NRRDs.

**Prerequisites**

- Prepare a neuron Parquet containing multiple neurons and at least one custom
  column. Load it through **Data** > **SWC Parquet Data** and use a Regions query
  so that only a subset of its neurons is in the visible data table.
- Prepare one lookup directory containing `flatmap_both_shaped.nrrd`,
  `flatmap_both_square.nrrd`, and `depth.nrrd`. All three files must describe
  the same spatial grid and transform.
- If the NRRD headers do not provide a usable spatial transform, know the
  lookup resolution in micrometers so it can be entered explicitly.
- Start from a clean napari session with the **Neuron Navigator** plugin open.

**Steps and expected results**

1. **Action:** Open the **Flatmap** tab, click **Lookup directory...**, and
   select the prepared directory.
   **Expected:** The selected lookup directory is shown. Validation and lookup
   set identification begin with **Prepare Whole Parquet...**, off the UI
   thread.
2. **Action:** Click **Prepare Whole Parquet...** and accept the suggested
   output name ending in `_flatmap.parquet`.
   **Expected:** A background, cancellable operation processes every source row,
   including neurons excluded from the current table query. Progress does not
   depend on which table rows are selected.
3. **Action:** Wait for preprocessing to finish, then load the output through
   **SWC Parquet Data**.
   **Expected:** The output has the same source row count and custom columns.
   It adds `x_flat_shaped`, `y_flat_shaped`, shaped validity/provenance columns,
   matching square columns, and shared `depth_um` validity/provenance columns.
   Its metadata identifies format version 3 and the lookup-set ID.
4. **Action:** In **Flatmap**, choose **Precomputed Parquet + Cache**, switch
   between **Both hemispheres, shaped** and **Both hemispheres, square**, and
   project the same table subset.
   **Expected:** Both styles render from the stored columns on their recorded
   canonical grids. The plugin does not ask for or load the NRRDs.
5. **Action:** Run **Prepare Whole Parquet...** again, choose the source file as
   the destination, and decline the replacement confirmation.
   **Expected:** Preprocessing does not start and the source is unchanged.
6. **Action:** Start preprocessing to a new output and cancel it while running.
   **Expected:** The temporary output is removed and no partial Parquet is
   published. If an output already existed, its previous contents are intact.
7. **Action:** Select a lookup directory whose files have mismatched grids or
   transforms.
   **Expected:** Validation reports the exact mismatch and does not create an
   output. A lookup set without a usable NRRD transform requests an explicit
   resolution instead of silently assuming one.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-004: Build and Reuse a Flatmap Region Cache

**Capability**

The user can build a versioned, reusable cache that projects BrainGlobe atlas
regions into the same fixed shaped and square flatmap grids used by a version-3
neuron Parquet. The cache stores label occupancy, closed region surfaces, and
depth-slice outlines. Reopening it avoids loading atlas annotation voxels,
projecting meshes, or converting region coordinates while viewing.

**Prerequisites**

- Complete UC-003 and keep its version-3 Parquet and lookup directory.
- Load the intended BrainGlobe atlas. Its annotation shape and voxel resolution
  must exactly match the lookup grid; its atlas name and version are recorded
  as cache-profile identity.
- Ensure there is enough writable disk space for both shaped and square cache
  profiles. Use the default 256 Y bins and 25 μm depth bins for this test. The
  X bin count is derived per style so bins are square, giving 512 X bins for
  the square style and 491 for the shaped style.
- Select at least one parent region with descendants in the **Regions** tab.
- Put at least two main-view layers in different visibility states and select
  one of them before testing that the flatmap window leaves the main viewer
  untouched.

**Steps and expected results**

1. **Action:** Open **Flatmap**, choose the lookup directory, click **Build
   Cache Profile...**, choose a new cache directory, and keep **New profile Y
   bins** at `256` and **Depth bin** at `25 um`.
   **Expected:** The plugin validates the exact atlas/lookup grid match and
   builds one profile containing a shaped/square pair. Progress covers
   annotation scanning, occupancy, surfaces, and outlines. The manifest is
   published only after all referenced arrays are complete.
2. **Action:** Change **New profile Y bins** to `272`, start a second build,
   and cancel it while it is running.
   **Expected:** Temporary files from that build are removed, the prior profile
   remains listed and usable, and the controls are restored.
3. **Action:** Close and reopen napari, load the version-3 Parquet and a
   BrainGlobe atlas with the same atlas/version structure catalog, choose
   **Precomputed Parquet + Cache**, set **Render** to **Heatmap** and its color
   mode to **Single color**, then click **Project to Flatmap**. After the
   heatmap appears in **Neuron Navigator Flatmap**, click **Choose Cache
   Directory...** and select the existing cache. Repeat the cache selection
   several times on both Windows and macOS when those systems are available.
   **Expected:** Cache validation runs without freezing either napari window.
   The main viewer stays visible and unchanged during computation. The separate
   flatmap viewer is first shown with its populated 3D heatmap; there is no
   blank intermediate window.
   The plugin parses `flatmap-region-cache.json`, memory-maps its arrays, and
   lists only profiles compatible with the Parquet lookup-set ID and selected
   style. A heatmap whose fixed grid matches the selected profile remains
   visible and is not recreated. Napari does not crash or show an operating
   system crash report, and the plugin does not access atlas annotation or
   region mesh data.
4. **Action:** Select the new cache profile.
   **Expected:** Y bins, depth-bin size, canonical bounds, and exclusion of the
   depth `-1` sentinel plane are set from the profile and locked. The profile's
   stored X bin count is used verbatim rather than re-derived. If the profile
   grid differs from an existing heatmap, that heatmap is hidden before being
   removed and the status asks the user to click **Project to Flatmap** again.
5. **Action:** Select a parent region, enable child regions, then click **Show
   Region Labels**, **Show Region Surfaces**, and **Show Region Outlines**.
   Then choose **Custom Regions**, select two terminal leaves, and click the
   same actions again.
   **Expected:** Labels apply the include-child-expanded selection. The surface
   and outlines use the selected parent's cached descendant union and atlas
   color. With Custom Regions active, labels and one atlas-colored
   surface/outline layer per exact terminal ID replace the Atlas selection.
   The resulting **Flatmap Region Labels**, **Flatmap Region Surfaces**, and
   **Flatmap Region Outlines** layers align with the neuron heatmap.
6. **Action:** Switch between **Both hemispheres, shaped** and **Both
   hemispheres, square** without changing the Parquet or cache directory, then
   project and show the desired cached region layers again.
   **Expected:** Stale layers from the previous style are cleared. Neurons and
   newly shown cached region layers use the matching profile grid and remain
   aligned, including when the queried neurons occupy only a small part of the
   global flatmap.
7. **Action:** Load a Parquet or atlas that is incompatible with every cached
   profile, or remove one referenced array and then reopen the cache. When
   testing a second directory, keep a valid cache and heatmap active first.
   **Expected:** The plugin reports the exact lookup, atlas, grid, missing-file,
   or validation mismatch. A failed candidate directory does not replace the
   active cache or clear its heatmap. It never starts NRRD recomputation
   automatically.
8. **Action:** Explicitly choose **Recompute from NRRDs**.
   **Expected:** The legacy runtime lookup workflow becomes available only for
   this explicit fallback choice.
9. **Action:** After a projection appears, use the flatmap window's
   operating-system close control. Verify that the main layers, order,
   visibility, active selection, dimensions, and camera never changed, then
   project again. Repeat three times. Finally, trigger a projection error or
   cancellation before any flatmap layer is committed.
   **Expected:** The Flatmap tab has no plugin-provided close button. On macOS
   the operating-system close control hides the visible Qt top-level and clears
   its transient layers, but its live viewer and canvas are reused by the next
   projection; `Viewer.close()` is not called on the visible secondary window.
   On Windows and Linux napari's public close API is used and the next
   projection creates one fresh viewer. A failed first render never exposes an
   empty window, while a failed re-render retains an already valid flatmap. The
   main viewer remains unchanged throughout.
10. **Action (macOS only):** Put the main napari window in macOS Full Screen,
    project, and dismiss/redisplay **Neuron Navigator Flatmap** three times using its
    operating-system close control. Then put the flatmap window itself in Full
    Screen and use the same close control.
    **Expected:** The main window remains fullscreen and responsive. The
    flatmap window opens populated, disappears without a ghost window, and can
    be reused. Dismissal produces no native crash or operating-system crash
    report. The macOS hide guard is the only private Qt integration; no
    slicer, status-thread, viewer-registry, or model cleanup is performed by the
    plugin.

**Manual verification**

- Status: Not run
- Last verified: 2026-07-22 (against the pre-square-bin cache format)
- Notes: **Re-run required.** On 2026-08-11 the flat map grid moved to per-axis
  bin counts: the control is now **Y bins** and the X count is derived from each
  style's aspect ratio, so one profile holds a 512-wide square grid and a
  491-wide shaped grid instead of two 256x256 grids. The region-cache manifest
  went to format version 2 and any cache built before that date is refused with a
  rebuild message rather than opened. Steps 1, 2, and 4 therefore describe
  different labels and different profile contents than the last passing run, and
  step 3's "reopen an existing cache" path must be re-exercised with a cache
  rebuilt under the new format. The earlier run below still stands for the
  previous cache format, but it does not verify napari 0.9 or the revised
  flatmap-window dismissal path.
  **Failed experiments:** On 2026-08-26, napari 0.9.0 on macOS 26.5.1 arm64
  crashed when **Close Flatmap Window** called `Viewer.close()` directly. It
  crashed again when the plugin first hid the window, disabled painting, waited
  50 ms, and then called `Viewer.close()`. Both macOS reports recorded
  `EXC_BAD_ACCESS` in `gleRunVertexSubmitImmediate` while Qt flushed a queued
  backing-store repaint. The resource-tracker semaphore warning after the
  second crash was shutdown fallout from `SIGSEGV`, not its cause.
  **Partial manual result:** The revised retain-and-reuse guard was stable when
  reached through the macOS operating-system close control, but invoking the
  same dismissal from the plugin button still caused a native crash. The plugin
  button and its callback were therefore removed; the complete repeated and
  fullscreen sequence in steps 9 and 10 remains `Not run`.
  **Superseded historical result:** macOS 26.5.1 arm64 with napari 0.6.6 and
  PyQt6 6.8.1 passed cache build/reopen/parse, shaped/square switching, and the
  former detached-window close/fullscreen checks without a crash or retained
  window. The napari 0.9 implementation now retains a macOS hide/reuse guard,
  so steps 3, 9, and 10 must be run again.

### UC-005: View Selected Allen Isocortex Layers as a Flatmap Projection or Stack

**Capability**

The user can choose any combination of the six Allen Isocortex layer groups:
`L1`, `L2/3`, `L4`, `L5`, `L6a`, and `L6b`. The chosen nodes can be rendered
as a **Single 2D projection**, which sums their raw node counts by flatmap XY
bin, or as a **2D stack** containing only the checked layers in superficial-to-
deep order. Layer membership comes from each node's Allen `region_id`, not
from numeric flatmap depth or SWC `type`.

All six layers and **2D stack** are selected by default, preserving the original
six-plane behavior. Single-color mode creates one image; Individual and Cluster
color modes create one corresponding image per `file_id` or cluster. The layer
selection also governs **Add Soma** and cache-backed region labels so every
visible layer uses the same coordinate space.

A stack has an **Allen layer** plane axis and an on-canvas caption naming the
current selected layer. A projection is a rank-2 image with only **Flatmap Y**
and **Flatmap X** axes, with no plane slider or plane caption.

The images are binned in index space, so the axis arrows name and orient the
axes without asserting physical units or anatomical direction.

**Prerequisites**

- Load a supported Allen mouse atlas in **Data** > **Atlas**.
- Load a neuron Parquet with Allen `region_id` annotations and valid flatmap
  coordinates. For **Precomputed Parquet + Cache**, use a version-3 Parquet
  produced by UC-003.
- Complete UC-004 and keep a compatible flatmap region cache. Select at least
  one Atlas Regions parent or Custom Regions terminal leaf so the planar label
  overlay can be checked.
- Include neurons with nodes in at least two Allen Isocortex layers. Include an
  agranular cortical area without layer 4 when test data is available.
- Start from a clean napari session with the **Neuron Navigator** plugin open.

**Steps and expected results**

1. **Action:** In the neuron table, select one or more rows. Open **Flatmap**,
   choose **Precomputed Parquet + Cache**, select **Both hemispheres, shaped**,
   choose the compatible cache profile, and set **Render** to **Allen Layer
   Heatmap**.
   **Expected:** **Allen Layer Options** appears with all six layers checked and
   **View** set to **2D stack**. **Y bins** remains available unless cache-
   locked. Numeric depth controls are disabled. Cached labels are available;
   cached surfaces and outlines remain unavailable because they are depth-based.
2. **Action:** Uncheck every layer except `L2/3` and `L5`, set **View** to
   **Single 2D projection**, keep **Heatmap colors** at **Single color**, and
   click **Project to Flatmap**.
   **Expected:** **Isocortex Flatmap Allen Layer Projection** is one rank-2
   node-count image. Every pixel equals the sum of the matching `L2/3` and `L5`
   stack pixels. Only **Flatmap Y** and **Flatmap X** axes appear; there is no
   plane slider or caption. The summary reports the two selected layer counts
   and separately counts nodes excluded because their Allen layer was unchecked.
3. **Action:** In **Regions**, select a cortical Atlas parent or Custom Region
   leaves spanning the chosen layers, then click **Show Region Labels**.
   **Expected:** **Flatmap Region Labels Allen Projection** is a rank-2 label
   image aligned with the heatmap and filtered to `L2/3` and `L5`. At XY
   collisions, greatest summed source-voxel occupancy wins and smaller region
   ID breaks ties. It reads the active cache rather than NRRDs or
   `atlas.annotation`.
4. **Action:** Click **Add Soma**.
   **Expected:** Only somas assigned by `region_id` to `L2/3` or `L5` appear,
   as XY-only points aligned with the projection. A selection with no matching
   soma reports that no soma mapped instead of falling back to numeric depth.
5. **Action:** Change **View** to **2D stack**, select `L1`, `L5`, and `L6b`,
   and project again.
   **Expected:** The projection, soma, and label layers from the previous
   coordinate space are removed immediately when the options change. The new
   **Isocortex Flatmap Allen Layers** image has exactly three planes in the
   order `L1`, `L5`, `L6b`; unchecked layers do not remain as blank planes.
6. **Action:** Click **Show Region Labels** and **Add Soma**, then move through
   all three planes.
   **Expected:** Heatmap, labels, and somas use the same compact plane indices.
   The canvas reads `Allen layer: L1  (plane 1 of 3)` through
   `Allen layer: L6b  (plane 3 of 3)` and each plane contains only nodes and
   regions belonging to its named Allen layer.
7. **Action:** Compare the summary and a recomputed CSV with independent
   `region_id` counts for the selected neurons.
   **Expected:** Rendered and per-selected-layer counts agree. Export retains
   canonical `allen_layer_index`/`allen_layer_label` and adds
   `allen_layer_selected` plus compact `allen_layer_render_index`. Flatmap-
   invalid, non-laminar, and valid but unchecked-layer nodes remain exported
   with `render_valid == false` and are reported separately.
8. **Action:** Use **Select All**, project the default six-plane stack, then
   repeat both views with **Both hemispheres, square**, **All table rows**,
   **Individual neurons**, and **Cluster**.
   **Expected:** Select All plus stack reproduces the original six-plane
   behavior. Shaped and square views use their recorded rectangular grids.
   Individual grouping remains by unique `file_id`, even when `neuron_id`
   repeats across subjects; Cluster uses the current assignment colors.
9. **Action:** Choose **Recompute from NRRDs**, select matching lookup files,
   and project the same layer subsets in both views.
   **Expected:** Materialized output matches the precomputed node-count and
   selection semantics. CSV export is available. Cache-backed labels, surfaces,
   and outlines are unavailable in this explicit fallback mode.
10. **Action:** Clear every Allen layer.
    **Expected:** **Project to Flatmap**, **Add Soma**, and **Show Region
    Labels** are disabled and the status reads **Select at least one Allen
    layer.** Selecting any layer enables the otherwise valid actions again.
11. **Action:** Switch **Render** to **3D Heatmap**, project, then switch to
    **3D Points**.
    **Expected:** Allen options hide, numeric-depth controls and compatible
    cached actions return, and no Allen layer filter affects the depth renders.
12. **Action:** Retry without a loaded atlas, with a Parquet missing
    `region_id`, with no flatmap-valid nodes in the selected Allen layers, and
    with selected regions that are either outside the chosen layers or absent
    from the active cache.
    **Expected:** Each attempt reports a corrective message, retains an already
    valid render when appropriate, and leaves no blank window or stale overlay.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: Selectable layers and the projection/stack choice were added on
  2026-09-04. Automated tests cover data parity, rank, metadata, compact plane
  indices, labels, somas, and rectangular grids, but the controls and visual
  alignment have not yet been exercised in napari.

### UC-006: Inspect and Query Custom Isocortex Layer Regions

**Capability**

The user can visually inspect the terminal Allen regions assigned to each of
the six Isocortex layer groups and query neurons using exactly those displayed
`region_id` values. The synthetic hierarchy is computed from the currently
loaded atlas and remains independent of the ordinary atlas-region selection
while retaining the same Allen identities. While Custom Regions is active, its
checked terminal regions can drive Reference-tab previews and flatmap labels,
surfaces, and outlines.

**Prerequisites**

- Load an Allen mouse atlas in **Data** > **Atlas**. For the count comparison,
  use `allen_mouse_25um`.
- Load a neuron Parquet annotated against that atlas. It must contain
  `file_id`, `type`, and `region_id`; include neurons from several Isocortex
  layers and at least one non-Isocortex region.
- Put a strict subset of the Parquet's neurons in **Selected Neurons** so the
  two search scopes can be distinguished.
- Complete UC-004 and keep its compatible version-3 Parquet and flatmap region
  cache so cached labels, surfaces, and outlines can be exercised.
- From the repository root, optionally run
  `pixi run python scripts/audit_allen_isocortex_layers.py` to print the
  independently inspectable region list and counts for `allen_mouse_25um`.

**Steps and expected results**

1. **Action:** Open **Regions** and inspect **Query source**.
   **Expected:** The menu is ordered **Atlas Regions**, **Custom Regions**,
   **Mask Layer**. Existing Atlas and Mask pages remain available.
2. **Action:** Select **Custom Regions** with **Search scope** set to
   **Whole Parquet**, then expand **Isocortex Layers**.
   **Expected:** The root starts expanded and the six initially collapsed
   groups appear in the order `L1`, `L2/3`, `L4`, `L5`, `L6a`, `L6b`.
   Root and group labels show terminal-region counts. Each leaf shows its full
   atlas name, acronym, and numeric ID and is alphabetized within its layer.
   For `allen_mouse_25um`, the displayed lists and counts match
   `scripts/audit_allen_isocortex_layers.py`; no parent or unrelated region is
   present.
3. **Action:** Check one layer group, uncheck one leaf beneath it, and then
   check that leaf again.
   **Expected:** Checking the group selects all and only its displayed terminal
   leaves. Unchecking a leaf makes the group and root partially checked;
   restoring it makes the layer fully checked. **Selected: N terminal
   regions** always reports the exact number of selected leaves.
4. **Action:** Clear the selection. Search in turn by part of a full name, an
   acronym, and a numeric region ID, then select one matching leaf and leaves
   from a second layer.
   **Expected:** Search retains matching leaves and their ancestors. **Clear**
   restores the full hierarchy. **Clear Selection** removes all checks.
   Selecting across layers yields a sorted, deduplicated set of the visible
   terminal IDs.
5. **Action:** Choose desired **Node types**, click **Find Neurons in Selected
   Custom Regions**, and compare the results with an exact `region_id` query of
   the Parquet.
   **Expected:** The **Selected Neurons** table contains precisely the matching
   neurons. The status reports the selected terminal-region count and node
   membership. No unlisted descendant or parent region contributes matches,
   and non-Isocortex rows do not match.
6. **Action:** Change **Search scope** to **Current Table**, make a different
   custom selection, and query. Switch back to **Whole Parquet**.
   **Expected:** Current Table querying is restricted to the exact file IDs
   already in the table and preserves nonmatching existing rows. The two
   scopes retain separate custom selections, and returning to Whole Parquet
   restores its selection.
7. **Action:** Keep **Custom Regions** active, select a full layer and a
   partial second layer, then enable **Show selected region meshes** and
   **Show selected region segmentation** in **Reference**.
   **Expected:** Napari switches to 3D for meshes and creates at most one
   `Region: Custom <layer>` Surface per selected canonical layer. Each
   terminal mesh retains its Allen color. `Region Segmentation` contains
   exactly the checked terminal IDs with no unlisted descendants. Changing
   Custom checks or Search scope refreshes both visible previews, and their
   opacity controls continue to apply.
8. **Action:** Switch **Query source** between **Atlas Regions** and
   **Custom Regions** while both Reference controls remain enabled.
   **Expected:** The existing preview layers are replaced using the active
   source and scope. Both selectors retain their own selections.
9. **Action:** Keep **Custom Regions** active, select two terminal leaves, open
   **Flatmap**, choose **Precomputed Parquet + Cache**, activate the compatible
   profile, set **Render** to **3D Heatmap**, and click **Show Region Labels**,
   **Show Region Surfaces**, and **Show Region Outlines**.
   **Expected:** **Flatmap Region Labels** contains exactly the checked
   terminal IDs. One atlas-colored surface and one outline layer are created
   per selected terminal region, with its acronym and numeric ID in the layer
   name. No unchecked descendant, synthetic layer union, or Atlas Regions
   selection contributes. Viewing reads cache arrays and the atlas structure
   catalog without loading NRRDs, `atlas.annotation`, or BrainGlobe meshes.
10. **Action:** Change the Custom Regions selection and click each **Show**
    action again; then switch between **Whole Parquet** and **Current Table**
    and repeat with the different retained selection.
    **Expected:** The existing Labels layer is updated instead of duplicated.
    Each geometry family is replaced with one layer per terminal ID from the
    active scope. Selection changes do not rebuild flatmap layers until the
    corresponding **Show** action is clicked.
11. **Action:** Set **Render** to **Allen Layer Heatmap**, choose **2D stack**,
    select the applicable layers, click **Show Region Labels**, then project
    neurons and move the first-axis slider.
    Repeat after switching between shaped and square styles.
    **Expected:** The selected custom terminal regions appear only on their
    corresponding `L1`, `L2/3`, `L4`, `L5`, `L6a`, or `L6b` planes. The
    labels and heatmap remain synchronized and aligned on each style's grid,
    and the canvas plane line names the Allen layer under the slider.
    Cached surfaces and outlines remain disabled because they are depth-based.
12. **Action:** Switch **Query source** to **Atlas Regions**, show its
    overlays, then switch to **Mask Layer** and try the same actions.
    **Expected:** Atlas actions retain include-child labels and parent-union
    geometry. Mask selections never fall back to a retained Atlas selection
    and instead report that flatmap atlas overlays require Atlas or Custom
    Regions.
13. **Action:** Exercise the error cases: query or render with no custom leaf
    selected; use an empty Current Table; load a Parquet without `region_id`;
    load an atlas whose structure catalog cannot produce all six Isocortex
    layers; and select a valid terminal region with no occupancy or geometry
    in the active cache.
    **Expected:** Each case reports an actionable message and does not perform
    a misleading query. An incompatible or missing atlas leaves the Custom
    Regions page in an explanatory empty state and does not open or download a
    second atlas. A valid query selection with no matching neurons reports zero
    results without error. A valid render selection with no cached data clears
    the stale overlay family and reports that nothing was represented.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-007: Refine and Save Multiple Cluster Assignments

**Capability**

The user can retain several named clustering results for the same neurons. A
first run can group neurons by soma location; one soma cluster can then be
selected and clustered by voxel correlation without removing other neurons or
overwriting the soma assignments. One assignment is active at a time for
filtering, colors, flatmap grouping, and Analysis heatmaps. Projects and
Enhanced Parquet exports preserve every assignment and its run provenance.

**Prerequisites**

- Load an Allen mouse atlas and a neuron Parquet containing at least six
  neurons with soma nodes and morphology nodes in one represented atlas region.
- To test Flat map + Depth selected-row clustering and flatmap grouping, use a
  version-3 Parquet with bilateral flatmap/depth columns.
- Populate **Selected Neurons** with the full test population and leave at
  least two neurons in each intended soma cluster.
- To verify the large-run guard, prepare cohorts whose post-filter clustering
  inputs contain exactly 10,000,000 and at least 10,000,001 contributing node
  rows.
- Start with no prior cluster assignments or record their names so the new
  columns can be distinguished.

**Steps and expected results**

1. **Action:** Open **Analysis** > **Clustering**, choose **CCFv3
   Coordinates**, **Soma Location**, and the desired soma algorithm. Set
   **Input neurons** to **Current Table**, click **Clear Selection** under
   **Select Target Region**, choose at least two clusters, and click **Run
   Clustering**.
   **Expected:** The run completes and creates a **Soma Location 1** column in
   the Data-tab table. **Cluster assignment** selects **Soma Location 1**, and
   every clustered neuron has an integer label. **Target region (optional)**
   reports **All regions (optional)**, and all eligible somas in the Current
   Table cohort contribute. The cluster filter, summary, rendered colors,
   flatmap cluster mode, and Analysis heatmap cluster choices use these labels.
2. **Action:** In **Data** > **Selected Neurons**, use **Cluster** to show one
   soma cluster and select all of its visible rows with Ctrl+A or Cmd+A.
   **Expected:** Only rows from that soma cluster are selected. Other neurons
   remain in the table and retain their soma labels.
3. **Action:** Return to **Analysis** > **Clustering**, choose **Voxel
   Correlation**, set **Input neurons** to **Selected Rows**, select the target
   region for this scope, and click **Run Clustering**. Repeat after clicking
   **Clear Selection** for this scope.
   **Expected:** Only the explicitly selected neuron IDs enter the run. A new
   **Voxel Correlation 1** column appears and becomes active. Its selected
   neurons receive local integer labels; every neuron outside the run is blank.
   **Soma Location 1** remains unchanged. The saved provenance records the
   selected cohort and its parent Soma Location assignment and cluster. The
   first run uses the selected region and dilation; the repeated run reports
   **All regions (optional)** and uses all finite CCFv3 nodes for those rows.
4. **Action:** Switch **Cluster assignment** between **Soma Location 1** and
   **Voxel Correlation 1**.
   **Expected:** The active header, cluster filter, summary, sort target,
   rendered colors, flatmap cluster grouping, and Analysis heatmap cluster
   choices all follow the selected assignment. Switching does not modify any
   saved labels. The voxel assignment's blank rows appear under
   **Unclustered**.
5. **Action:** Click **Rename...**, give the voxel assignment a descriptive
   name, and confirm. Then create another selected-row run.
   **Expected:** The table header and selector use the new display name while
   the next run creates another independent column. Existing Enhanced Parquet
   column identifiers remain stable.
6. **Action:** Select an assignment with live run data and build its
   dendrogram or save a single-run Analysis export.
   **Expected:** The output uses the active run. Other assignment columns are
   unaffected.
7. **Action:** In **Data** > **SWC Parquet Data**, click **Save Project As...**
   and **Export Enhanced Parquet...**. Close the session, load the saved
   project, and then load the Enhanced Parquet separately.
   **Expected:** Both reload paths restore every assignment name, sparse label
   map, active assignment, palette, cohort, parent context, and run parameters.
   Enhanced Parquet contains one nullable integer column per assignment and a
   backward-compatible `cluster_assignment` column matching the active set.
   After project reload, assignment-based filters, colors, flatmap grouping,
   and heatmaps work immediately; dendrogram and distance actions report
   **Rerun required** because matrices are not stored.
8. **Action:** With an assignment active, click **Delete**, cancel once, then
   confirm deletion.
   **Expected:** Cancelling changes nothing. Confirming removes only that
   assignment column and its saved result; no neuron rows are removed. The
   most recently created remaining assignment becomes active.
9. **Action:** Set **Input neurons** to **Selected Rows** with no selected
   rows, then with exactly one selected row, and click **Run Clustering** after
   each change. Set **Input neurons** to **Whole Parquet**, clear its target
   region selection, and try again. Repeat a valid selected-row run in **Flat
   map + Depth** for both **Soma Location** and **Voxel Correlation** when
   flatmap columns are available.
   **Expected:** Empty and one-row inputs produce actionable messages and do
   not launch a worker or create a column. Whole Parquet reports **Select at
   least one target region** and does not run. Valid flatmap runs use exactly
   the selected neuron IDs and create new sparse assignments like their CCFv3
   counterparts.
10. **Action:** Run clustering on the cohort with exactly 10,000,000 contributing
    node rows. Then run it on the cohort with at least 10,000,001 contributing
    rows; click **Cancel** in **Large Clustering Run**, repeat, and click
    **Continue**.
    **Expected:** The 10,000,000-node run starts without a warning. The larger
    run displays its exact comma-formatted node count and defaults to **Cancel**.
    Cancelling starts no clustering worker and creates no assignment; the
    repeated confirmation starts the snapshotted run without changing its
    cohort or region filter.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-008: Create Combined, Individual, and Enhanced Neuron Heatmaps

**Capability**

The user can turn selected rows in the Data-tab neuron table into either one
combined node-count heatmap or one color-matched heatmap per neuron. Individual
layers preserve the selected cohort captured when the run starts and make it
possible to compare neuron occupancy independently. A monochrome cohort is
automatically assigned distinct palette colors so its rendered neurons, table
swatches, and heatmaps remain visually associated. One or more heatmaps can be
selected in **Tools** and given the minimum supported gamma to make faint,
fine projections easier to see without changing the underlying voxel data.

**Prerequisites**

- Have an Allen mouse atlas available locally and a neuron Parquet containing
  at least four neurons with morphology nodes inside the atlas bounds.
- Know three neurons to use as the heatmap cohort and retain a fourth neuron as
  a non-selected color control.
- To exercise the memory warning, use an atlas resolution and selected-neuron
  count whose displayed estimate exceeds 1 GiB.
- Start from a clean napari session with the **Neuron Navigator** plugin open.

**Steps and expected results**

1. **Action:** Open **Data** > **Selected Neurons**, click **Add Heatmap**, and
   inspect its menu before loading a neuron Parquet.
   **Expected:** The menu contains exactly **Single Heatmap** and **Individual
   Heatmaps**. Choosing either option reports **Load a neuron Parquet before
   creating a heatmap** and creates no layer.
2. **Action:** Load the test neuron Parquet, add its four neurons to **Selected
   Neurons**, leave the atlas unloaded, select three rows, and choose **Add
   Heatmap** > **Single Heatmap**. Then clear the table selection and try the
   action again after loading the atlas.
   **Expected:** Without an atlas, the first attempt reports **Load an atlas
   before creating a neuron heatmap**. With no selected rows, the second attempt
   reports **Select at least one neuron row to create a heatmap**. Neither
   attempt creates a layer.
3. **Action:** Give the three cohort neurons visibly different colors with
   their color swatches, select their rows, and choose **Add Heatmap** >
   **Single Heatmap**.
   **Expected:** One Greek-named image layer such as **alpha Heatmap** is added
   with the `hot` colormap. Its node counts combine all three selected neurons.
   Each cohort row lists that layer in the **Heatmap** column, the fourth row
   does not, and choosing the layer under **Manual Heatmap** shows only its
   three source rows.
4. **Action:** Keep the same distinctly colored rows selected and choose **Add
   Heatmap** > **Individual Heatmaps**. Change the table selection while the
   sequential progress messages are visible.
   **Expected:** **Add Heatmap** is disabled for the full queue. Three new
   Greek-named layers are added in deterministic order, one for each originally
   selected neuron, regardless of the later selection change. Each layer uses
   a transparent-to-neuron-color colormap, contains only that neuron's counts,
   initially sets its upper contrast limit to 20% of its maximum node count,
   and appears only on that neuron's **Heatmap** cell and **Manual Heatmap**
   filter result. Lower-density structures are more visible than with the full
   contrast range, and the limit remains editable in napari. The existing
   neuron colors do not change.
5. **Action:** Select the same three cohort rows, set all three color swatches
   to the same RGB color, record the fourth neuron's color, and choose **Add
   Heatmap** > **Individual Heatmaps** again.
   **Expected:** The three selected neurons receive distinct palette colors in
   stable neuron-ID order, and rendered versions of those neurons update to the
   same colors. The fourth neuron's color is unchanged. Each newly created
   heatmap matches its source neuron's new color.
6. **Action:** Set one cohort neuron back to the same color as another while
   leaving the third different, then create individual heatmaps for all three.
   Repeat with only one selected neuron.
   **Expected:** A partially duplicated but non-monochrome cohort keeps its
   existing colors, including the duplicate. A one-neuron request creates one
   individual layer without recoloring that neuron.
7. **Action:** Select enough neurons for the projected individual heatmap data
   to exceed 1 GiB and choose **Add Heatmap** > **Individual Heatmaps**. Leave
   **Cancel** selected in **Large Individual Heatmap Request** and confirm.
   **Expected:** The warning shows the selected layer count and estimated GiB
   before rendering overhead. Cancelling starts no worker, adds no layers, and
   changes no neuron colors.
8. **Action:** Delete or rename one of the individual heatmap layers, then use
   the table's **Heatmap** column and **Manual Heatmap** selector.
   **Expected:** Layer membership and names stay synchronized. Deleting a layer
   removes its membership; renaming it updates the affected row and selector
   without changing the recorded source neuron.
9. **Action:** Open **Tools** > **Heatmap Sources**. With no source highlighted,
   inspect **Enhance Fine Projections** and **Reset Gamma**. Select one heatmap
   and click **Enhance Fine Projections**.
   **Expected:** Both buttons are disabled with no selection and enabled when a
   heatmap is selected. The selected layer's gamma becomes `0.20`, faint
   low-intensity projections brighten, other heatmaps do not change, and the
   status reports the number of layers updated. The heatmap voxel data and
   contrast limits are unchanged.
10. **Action:** Use Ctrl-click (Windows/Linux) or Command-click (macOS) to
    select multiple entries under **Heatmap Sources**, then click **Enhance Fine
    Projections**. Leave at least one heatmap unselected as a control. Click
    **Reset Gamma** with the same entries selected.
    **Expected:** Enhancement applies gamma `0.20` to every selected heatmap and
    does not affect the unselected control. Reset restores gamma `1.00` on every
    selected heatmap. If the project is saved and reopened, each saved
    heatmap's current gamma is restored.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-009: Save and Overwrite the Current Project

**Capability**

The user can create a Neuron Navigator project and then save later table, analysis,
and app-created layer changes back to that same project folder. Replacement is
confirmed explicitly and publishes a complete new bundle without retaining
stale files from the previous version.

**Prerequisites**

- Load a neuron Parquet containing at least two neurons into **Data** > **SWC
  Parquet Data**.
- Add both neurons to **Selected Neurons** and create one app-generated heatmap
  or mask layer that will be recognizable after a project reload.
- Choose a writable directory where a new `.nnproj` project folder can be
  created.

**Steps and expected results**

1. **Action:** With only the neuron Parquet loaded, inspect the project controls
   under **Data** > **SWC Parquet Data**.
   **Expected:** **Save Project** is disabled. **Save Project As...**, **Load
   Project...**, and **Export Enhanced Parquet...** remain available.
2. **Action:** Click **Save Project As...**, choose a new path named
   `overwrite_test.nnproj`, and complete the save.
   **Expected:** The project is created and the status reports **Saved project
   bundle: overwrite_test.nnproj**. **Save Project** becomes enabled and its
   tooltip identifies the new project's absolute path.
3. **Action:** Change a table label or note, remove the saved heatmap or mask
   layer, create a different app-generated layer, and click **Save Project**.
   In **Overwrite Neuron Navigator Project?**, click **Cancel**.
   **Expected:** The dialog shows the exact current project path, warns that all
   existing project-folder contents will be replaced, and defaults to
   **Cancel**. Cancelling starts no save and leaves the on-disk project
   unchanged.
4. **Action:** Click **Save Project** again and click **Overwrite**.
   **Expected:** Progress is shown while saving, both save actions are disabled
   during serialization, and completion reports **Saved project bundle:
   overwrite_test.nnproj**. No save-location picker is shown.
5. **Action:** Close the session, click **Load Project...**, select
   `overwrite_test.nnproj`, and inspect its table state and app-generated layers.
   **Expected:** The changed table metadata and replacement layer are restored.
   The layer removed before overwrite does not return, and no stale project
   files affect the loaded session.
6. **Action:** Load a standalone neuron Parquet through **Load...**.
   **Expected:** **Save Project** becomes disabled because the session is no
   longer associated with a current project. **Save Project As...** can create
   another project, and doing so enables **Save Project** for that new folder.
7. **Action:** Load `overwrite_test.nnproj`, then rename or move that folder
   outside napari and click **Save Project**.
   **Expected:** The plugin reports that the current project is unavailable or
   unrecognized, disables **Save Project**, and does not create or replace any
   folder. The user can still choose a new destination with **Save Project
   As...**.

**Manual verification**

- Status: Passed
- Last verified: 2026-08-05
- Notes: Verified by removing Heatmap layers. Two clusterings were retained.

### UC-010: Identify Axon Termini and Prune Neurons Lacking Them

**Capability**

The user can locate the tips of traced neurites. A terminus is a node no other
node claims as its parent, so restricting termini to axon-typed nodes yields the
tips of everything the source file typed as axon.

> **Caution: axon-typed does not mean axon.** Visual inspection on 2026-08-05
> found neurons in `isocortex_total_right_brainglobe_flatmap.parquet` whose
> **dendritic** projections are typed `2` (axon). Those dendrite tips are
> therefore reported as axon termini. This is a defect in the source SWC
> annotations, not in the detection: a reported node genuinely has no children,
> but the `type` on it cannot be trusted to identify the compartment. Treat every
> count below as **axon-typed** termini — an upper bound on true axon termini,
> contaminated by an unknown number of dendrite tips. The extent has not been
> quantified. Verify visually before drawing a biological conclusion from these
> points, and do not assume node types partition a neuron into clean axon and
> dendrite subtrees.

The other risk the node-type restriction carries is coverage. Reconstructions
whose neurites are all typed `Undefined` contain no axon-typed nodes, so they
contribute nothing. The plugin therefore always reports how many neurons were
skipped instead of quietly returning a partial answer.

Because those skipped neurons are exactly the ones worth excluding from a
clustering run, the section doubles as a data-quality gate on the neuron table:
after a run it can select either the neurons that produced termini or the ones
that did not, so the unwanted group can be removed with **Remove Selected From
Table** before clustering.

Two rules keep the count correct, and both follow from the childless test needing
to see the whole tree: the node-type restriction narrows only which termini are
reported, and any region restriction is applied after detection. Filtering nodes
away before the test would leave their parents looking childless.

**Prerequisites**

- A neuron Parquet with `file_id`, `node_id`, `parent_id`, `type`, `x`, `y`, and
  `z` columns loaded into **Data** > **SWC Parquet Data**.
- No atlas is required; terminus detection is pure topology.
- For the reference numbers below, use
  `isocortex_total_right_brainglobe_flatmap.parquet` (18,621 neurons,
  728,703,227 nodes). Useful individual neurons in that file:
  - `17099_002_reg.swc` — axon labelled, 295 axon termini, no dendrite nodes.
  - `192309_031_reg.swc` — axon and dendrite labelled, 91 axon termini and 89
    dendrite termini.
  - `212064_001.swc` — every neurite typed `Undefined`, so 0 axon termini.

**Steps and expected results**

1. **Action:** Open the **Data** tab, scroll below **Selected Neurons**, and
   expand **Termini**.
   **Expected:** The section explains that it finds childless nodes of the
   selected types, that neurons typed entirely `Undefined` are reported as
   skipped, and how to use the selection to prune the table. **Neurons:**
   defaults to **Current Table**, **Node types:** shows **Axon**, and **Point
   size:** shows `20.0`. **Find Termini** is enabled once a Parquet is loaded,
   with or without an atlas. **Select in Table** is disabled until a run
   finishes.
2. **Action:** Add `17099_002_reg.swc` to **Selected Neurons**, set **Neurons:**
   to **Selected Rows**, and click **Find Termini**.
   **Expected:** Progress is shown, then a **Termini (Axon)** points layer
   appears. The coverage line reads **295 termini (Axon) in 1 of 1 neurons** with
   no skipped-neuron clause, and **Copy Skipped Neuron IDs** stays disabled.
3. **Action:** Hide every layer except the neuron trace and the new points layer,
   then inspect the arbor in 3D.
   **Expected:** Points sit at the visible free ends of the axon branches, not
   mid-branch and not at branch points. The soma carries no point.
4. **Action:** Select `192309_031_reg.swc` alone and click **Find Termini** with
   **Node types:** still **Axon**.
   **Expected:** The coverage line reads **91 termini (Axon) in 1 of 1 neurons**,
   and it also reports that **89 childless nodes of other types** were not
   counted. Points land on the far-reaching axon branches, none on the compact
   arbor next to the soma.
5. **Action:** Change **Node types:** to **Basal dendrite** only and click **Find
   Termini** again.
   **Expected:** A **Termini (Basal dendrite)** layer replaces nothing — it is a
   separate layer — and reports **89 termini (Basal dendrite) in 1 of 1
   neurons**. For this particular neuron its points form a compact cluster close
   to the soma, distinct from the axon-typed termini of the previous step. Note
   that this separation is a property of how *this* file was annotated, not a
   guarantee: see the caution above and step 6.
6. **Action:** Repeat steps 2-4 on several other neurons, hiding all layers
   except the neuron trace and the **Termini (Axon)** points, and look
   specifically for axon-typed points sitting on short, soma-proximal,
   dendrite-shaped arbors rather than on long-range projections.
   **Expected:** At least some neurons show axon-typed termini on plainly
   dendritic branches. This is the annotation defect described in the caution
   above, not a detection error — the plugin faithfully reports the tips of
   whatever the file typed as axon. Record which neurons show it; the extent
   across the file has not been measured.
7. **Action:** Select `212064_001.swc` alone, set **Node types:** back to
   **Axon**, and click **Find Termini**.
   **Expected:** No points layer is added, and the coverage line still appears,
   reading **0 termini (Axon) in 0 of 1 neurons — 1 neurons skipped (no nodes of
   the selected types)** along with the count of childless nodes not counted.
   **Copy Skipped Neuron IDs** becomes enabled.
8. **Action:** Click **Copy Skipped Neuron IDs** and paste into a text editor.
   **Expected:** The status line reports how many IDs were copied, and the
   clipboard holds `212064_001.swc`.
9. **Action:** Set **Neurons:** to **Whole Parquet**, keep **Node types:** as
   **Axon**, and click **Find Termini**.
   **Expected:** Progress advances in neuron batches, reporting
   **Scanning neurons for termini (N/18,621)**, and napari stays responsive
   throughout. The run takes roughly 70 seconds on a workstation with the file on
   local disk. The coverage line then reads **3,207,618 termini (Axon) in 13,750
   of 18,621 neurons — 4,871 neurons skipped (no nodes of the selected types);
   1,521,610 childless nodes of other types not counted**. Because more than 200
   neurons were skipped, the line also states that only the first 200 skipped
   neuron IDs are listed. Memory use stays bounded: detection runs a batch of
   neurons at a time rather than the whole file in one query.
10. **Action:** Set **Neurons:** to **Selected Rows** with no table rows selected
    and click **Find Termini**.
    **Expected:** No worker starts and no layer is added. The status line reports
    that no table rows are selected and suggests switching to **Whole Parquet** or
    populating the table.
11. **Action:** Select the **Termini (Axon)** layer and inspect its metadata (for
    example through the napari console).
    **Expected:** `file_ids_per_point`, `node_ids`, and `point_types` are present
    and the same length as the point data, so each point traces back to its
    source node. `coverage_summary` and `skipped_file_ids` are also recorded.
12. **Action:** Populate **Selected Neurons** with a mix of neurons including
    `17099_002_reg.swc` and `212064_001.swc`, leave **Neurons:** on **Current
    Table** and **Node types:** on **Axon**, then click **Find Termini**.
    **Expected:** The coverage line reports one skipped neuron, and **Select in
    Table** becomes enabled.
13. **Action:** Set the selection dropdown to **Neurons lacking termini** and
    click **Select in Table**.
    **Expected:** Only `212064_001.swc` is selected in the table, and the status
    line reads **Selected 1 of N table neurons lacking termini.**
14. **Action:** Set the dropdown to **Neurons with termini** and click **Select
    in Table** again, without re-running detection.
    **Expected:** The complement is selected — every table neuron except
    `212064_001.swc` — and the status line reports that count. The dropdown
    switches groups without another detection run.
15. **Action:** Switch back to **Neurons lacking termini**, click **Select in
    Table**, then click **Remove Selected From Table** in **Selected Neurons**.
    **Expected:** `212064_001.swc` leaves the table, the table summary updates,
    and a clustering run started from the **Analysis** tab with **Current Table**
    scope no longer includes it.
16. **Action:** With the table still populated, set **Neurons:** to **Selected
    Rows**, select only a subset of rows, click **Find Termini**, then use
    **Select in Table** on either group.
    **Expected:** Only rows inside the analyzed subset are considered, and the
    status line additionally reports how many table rows were outside the
    analyzed scope, suggesting a re-run to cover them. Rows the run never
    examined are never selected on a guess.
17. **Action:** Clear the neuron table, leave **Neurons:** on **Current Table**,
    and click **Find Termini**.
    **Expected:** No worker starts and no layer is added. The status line reports
    that the current table is empty and suggests switching to **Whole Parquet**
    or populating the table.

**Manual verification**

- Status: Partially run
- Last verified: 2026-08-05
- Notes: **Step 6 found the annotation defect.** Visual inspection in napari
  confirmed that some neurons in
  `isocortex_total_right_brainglobe_flatmap.parquet` have dendritic projections
  typed `2` (axon), so their dendrite tips are reported as axon termini. The
  detection itself is behaving correctly — the source SWC types are wrong. The
  number of affected neurons was not counted; quantifying it is open work.
  Because of this, every "axon termini" count in this use case is really a count
  of **axon-typed** termini.
  The counts in steps 2, 4, 5, 7, and 9 were confirmed against that same file by
  running the detection functions over all 728,703,227 rows, but those UI steps
  have not been exercised. Steps 1 and 12-17 cover the move to the **Data** tab
  and the table-selection dropdown and have not been exercised in napari either.

### UC-011: View a Depth-Free 2D Flatmap and Per-Neuron Vector Traces

**Capability**

The user can view selected neuron morphology on a plain flatmap with no depth
axis at all, either as one 2D node-count image (**2D Heatmap**) or as per-neuron
line traces (**2D Vector**). This is for collaborators who want the flatmap
footprint of an arbor without cortical-depth structure.

Collapsing removes the depth *axis*, not any node: **Exclude depth -1 nodes**
still decides whether depth `-1` nodes render, so a **2D Heatmap** is exactly the
matching **3D Heatmap** summed over its planes and reports the same node counts.
The dedicated flatmap canvas uses 2D with only **Flatmap Y** / **Flatmap X**
axes, no plane slider, and no plane caption.

**2D Vector** draws one line per parent-child edge on the same pixel grid the
**2D Heatmap** uses, colored per neuron from the table. It is a per-node render
limited to 250,000 segments; above that it refuses rather than drawing an
incomplete subset of neurons.

**Add Soma** places somas in whichever coordinate space the current **Render**
mode uses, so soma points land on the visible render in all five modes.

The collapsed **Heatmap Appearance** section lists rendered flatmap heatmap
layers from the dedicated flatmap viewer. It can apply the minimum supported
gamma to one or several selected 3D, 2D, or Allen-layer heatmaps so faint
projections remain visible without changing their node-count data.

**Prerequisites**

- Load a neuron Parquet with valid flatmap coordinates. For **Precomputed
  Parquet + Cache**, use a version-3 Parquet produced by UC-003.
- Include neurons whose nodes span a range of cortical depths, so collapsing is
  visibly different from a single depth plane.
- To check the over-limit behavior, have a table large enough that all selected
  neurons together exceed 250,000 rendered nodes.
- Start from a clean napari session with the **Neuron Navigator** plugin open.

**Steps and expected results**

1. **Action:** Select a few neuron rows, open **Flatmap**, choose **Precomputed
   Parquet + Cache** and **Both hemispheres, shaped**, set **Render** to **2D
   Heatmap**, and click **Project to Flatmap**.
   **Expected:** **Neuron Navigator Flatmap** shows one image named **Isocortex Flatmap
   2D Heatmap** in 2D. There is no plane slider and no plane caption in the
   upper-left corner. Labelled **Flatmap X** / **Flatmap Y** axis arrows are
   drawn at the image origin. The summary panel ends with `Depth: collapsed into
   one flatmap plane`.
2. **Action:** Inspect the control row.
   **Expected:** **Y bins** is available unless locked by the cache profile.
   **Depth bin** is disabled because there are no depth bins to size, while
   **Exclude depth -1 nodes** stays enabled. **Show Region Labels**, cached
   surfaces, and outlines are all disabled because their geometry is depth-based.
3. **Action:** Note the rendered-node count, switch **Render** to **3D Heatmap**,
   click **Project to Flatmap**, and compare.
   **Expected:** The rendered-node and flatmap-valid counts match the 2D render
   exactly for the same **Exclude depth -1 nodes** setting. The depth-binned
   stack has a plane slider captioned **Depth bin**.
4. **Action:** Switch back to **2D Heatmap**, toggle **Exclude depth -1 nodes**,
   and project again.
   **Expected:** The rendered-node count changes by the number of depth `-1`
   nodes, confirming the checkbox still governs a depth-free render.
5. **Action:** Set **Heatmap colors** to **Individual neurons** and project.
   **Expected:** One tinted 2D image per neuron appears, each named `Isocortex
   Flatmap 2D Heatmap: <file id>`, and the images overlay additively.
6. **Action:** With a small selection, set **Render** to **2D Vector** and click
   **Project to Flatmap**.
   **Expected:** A layer named **Isocortex Flatmap 2D Vectors** draws each
   neuron's arbor as connected lines in its table color. **Heatmap colors** is
   disabled. The summary reports the rendered segment count.
7. **Action:** Re-project **2D Heatmap** so both layers are present, then zoom in
   on a soma and on a distal arbor tip.
   **Expected:** The vector lines sit **on** the lit heatmap pixels with no
   visible half-pixel offset in either direction. This is the check automated
   tests cannot make.
8. **Action:** Select every neuron in the table and project in **2D Vector**.
   **Expected:** The projection is refused with a message naming the segment
   count and the 250,000 limit and suggesting 2D Heatmap. No vector layer is
   added, the existing flatmap remains usable (or no window opens on a first
   failure), and both viewers stay responsive.
9. **Action:** In each of the five **Render** modes in turn — **3D Heatmap**, **3D
   Points**, **2D Heatmap**, **2D Vector**, **Allen Layer Heatmap** —
   project, then click **Add Soma**.
   **Expected:** In every mode the somas appear on the render that is on screen.
   In the two 2D modes and the Allen stack the flatmap canvas stays in 2D. In the
   Allen stack the plane caption (for example `Allen layer: L2/3  (plane 2 of 6)`) and
   the **Flatmap X** / **Flatmap Y** labels survive adding the somas, and moving
   the plane slider shows somas only on their own layer's plane.
10. **Action:** With a soma layer visible, change **Render** to a different mode.
    **Expected:** The **Isocortex Flatmap Somas** layer is removed, because its
    bin coordinates belong to the previous coordinate space. **Add Soma** can be
    clicked again to rebuild it for the new mode.
11. **Action:** With **Render** set to **Allen Layer Heatmap** and a
    Parquet that has no `region_id` column, click **Add Soma**.
    **Expected:** The action reports that `region_id` is required and names both
    remedies (regenerate the Parquet, or switch to a depth or 2D mode). No soma
    layer is added and no somas are silently placed on depth bins.
12. **Action:** Before rendering a heatmap, expand **Heatmap Appearance** in the
    **Flatmap** tab. Then render a **2D Heatmap** and inspect the section again.
    **Expected:** Initially the section reports that no rendered flatmap
    heatmaps are available and both gamma buttons are disabled. After rendering,
    **Isocortex Flatmap 2D Heatmap** appears in the section's layer list. Points,
    vectors, region labels, surfaces, and outlines never appear in this list.
13. **Action:** Render at least three heatmaps with **Heatmap colors** set to
    **Individual neurons**. Select two entries in **Heatmap Appearance** with
    Ctrl-click (Windows/Linux) or Command-click (macOS), leaving the third as a
    control. Click **Enhance Fine Projections**, then **Reset Gamma**.
    **Expected:** Enhancement applies gamma `0.20` to the two selected flatmap
    heatmaps, brightening their faint projections without changing their data,
    contrast limits, or the unselected control. Reset restores gamma `1.00` on
    both selected layers. Renaming or removing a heatmap in **Neuron Navigator
    Flatmap** refreshes the list. Using the flatmap window's operating-system
    close control clears the transient heatmaps and the list.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: Added on 2026-08-05 and not yet exercised in napari. On 2026-08-11 the
  grid gained per-axis bin counts, which makes step 7's vector/heatmap overlay
  check the most important one here: the vector builder and the heatmap now take
  two counts, and a single shared count would compress every vector about 2x
  along X. `tests/test_flatmap_rectangular_grid.py` asserts per-node
  co-registration on a deliberately non-square grid, but only the canvas shows
  whether the lines sit on lit pixels. The control is now labelled **Y bins**.
  Automated tests
  cover the collapse invariant, the pixel-centering math, the segment limit, and
  the per-mode soma coordinate space, and the vector/heatmap alignment was
  confirmed numerically against `AUDpo_left_brainglobe_flatmap.parquet` (interior
  vector endpoints round to exactly their heatmap bin). None of that shows
  whether the overlay reads correctly on the canvas, so step 7 still needs eyes.

### UC-012: Balance Cortical Depth Against Flat Map Position When Clustering Somas

**Capability**

The user can control how heavily cortical depth counts when clustering neurons
by soma location in **Flat map + Depth** space, and can ignore depth entirely —
in either **Soma Location** or **Voxel Correlation** — to cluster on flat map
position alone.

This exists because the raw Parquet columns mix units. `x_flat` and `y_flat` are
normalized floats — a hemisphere spans 1.0 — while `depth_um` is raw microns
spanning up to about 1,856. Clustering those together with an unweighted
Euclidean metric let depth supply over 99.99% of the variance, so results were a
laminar partition with no flat map contribution: measured on
`isocortex_total_right_brainglobe_flatmap.parquet`, every k-means cluster at
k=10 spanned essentially the whole hemisphere tangentially while carving depth
into contiguous bands, and the labels matched a depth-only clustering at
ARI 0.85.

Soma coordinates are now scaled before any distance is computed: **both flat
map axes are divided by the same number** (the `y` span, which is one
hemisphere tall), so flat map space is scaled without being distorted.
**Depth scale** then weights the depth axis:

- `1.0` (default) treats a full cortical thickness of depth separation as
  equivalent to one hemisphere height of tangential separation.
- Higher values weight depth **more**, pulling clusters toward cortical layers.
- Lower values weight depth **less**, pulling clusters toward flat map position.
- **Ignore depth (flat map X/Y only)** drops the depth axis outright, clustering
  in two dimensions.

**Depth scale is a ratio of axis fractions, not of physical distances.** The
flat map projection — produced by a separate research group, not this
repository — distorts the cortical surface, so flat map `x`/`y` have no reliable
conversion to microns. Measuring the local scale empirically against CCF
coordinates gives values that vary by roughly 2x across the map and drift
systematically with the separation being measured, which is the distortion
showing through. Do not convert flat map `x`/`y` to microns, and do not describe
a depth scale as equivalent to some number of microns of tangential distance.
Scaling both flat map axes by one shared divisor is what keeps the metric well
defined without making that claim.

One divisor serves both flat map axes, so equal distances in the metric mean
equal distances on the flat map whichever direction they run. This is the same
policy the voxel grid uses, where the derived `x` bin count makes a bin as wide
as it is tall; both take `y` as the reference axis because `y` spans one
hemisphere while `x` spans two.

Earlier versions divided each axis by its own span, which forced every style's
hemisphere into a square bounding box. The bilateral square style is an exact
2:1 map, so its divisors were already equal and its results are unchanged. The
bilateral shaped style is about 4% off square, so it carried a 4.2% anisotropy.
Removing it moves about 38% of shaped somas to a different cluster, measured on
all 18,518 somas in `isocortex_total_right_brainglobe_flatmap.parquet` with
hierarchical/ward at **Depth scale** `1.0` — about 37% at k=5 and about 39% at
k=10. Results from before the change are not comparable, and `distance_metric`
was renamed so they cannot be mistaken for each other.

Quote the algorithm and the neuron count whenever you cite one of those rates,
because nothing else reproduces them: k-means over the same somas moves about
1% at k=5, and ward over an 833-soma subset moves about 2%. Ward is
agglomerative, so removing a small distortion flips near-tied merges and the
reordering cascades; a larger table has more near-ties available to flip. A
rate measured on a partial table does not bound the full-dataset rate.

DBSCAN's **Eps** changes units with the coordinate space: microns in **CCFv3
Coordinates**, normalized flat map units in **Flat map + Depth**, where
1.0 is one hemisphere height. The radius is circular because both flat map
axes share one divisor. Each space remembers its own value.

**Ignore depth in Voxel Correlation** collapses the voxel grid's depth planes,
so nodes at one flat map position share a voxel whatever their depth. Two
neurons then correlate on flat map footprint regardless of which layers they
occupy. Depth still decides which nodes are counted, so the rendered node count
is unchanged and **Include depth -1 plane** keeps its meaning; **Depth bin**
greys out because a collapsed grid has no depth bins to size.

There is deliberately **no depth scale for Voxel Correlation.** That path
compares voxel occupancy with a Pearson correlation, which treats voxels as an
unordered set of categories — two neurons in adjacent voxels correlate exactly
as poorly as two in opposite corners of the volume, so there is no distance for a
weight to act on. Scaling depth would also be redundant: since
`depth_bin = floor(depth / depth_bin_um)`, multiplying depth by *k* is identical
to setting **Depth bin** to `depth_bin_um / k`. Bin *resolution* is the real knob
there, and it already exists.

**Prerequisites**

- A version-3 Parquet with flatmap and depth columns from UC-003; the canonical
  bounds in its metadata are what make the metric independent of which neurons
  are in scope. `isocortex_total_right_brainglobe_flatmap.parquet` works.
- Neurons whose somas span several cortical layers *and* a wide area of the flat
  map, so laminar and areal groupings are visibly different. A single small
  region will not show the difference.
- A Parquet lacking canonical bounds is still usable but falls back to observed
  data bounds; results are then comparable only within that fixed dataset.
- Start from a clean napari session with the **Neuron Navigator** plugin open.

**Steps and expected results**

1. **Action:** Open the **Analysis** tab, expand **Clustering**, and set
   **Coordinate space** to **Flat map + Depth** and **Method** to **Soma
   Location**.
   **Expected:** **Ignore depth (flat map X/Y only)** and **Depth scale** appear.
   **Depth scale** reads `1.00` and is enabled. **Y bins**, **Depth bin**, and
   **Include depth -1 plane** stay hidden, since binning applies only to voxel
   correlation.
2. **Action:** Hover over **Depth scale**.
   **Expected:** The tooltip states that higher values weight depth MORE
   (laminar grouping) and lower values weight depth LESS (areal grouping), and
   explains that 1.0 equates one cortical thickness with one hemisphere height,
   and states that both flat map axes share one divisor so the space is scaled
   without distortion.
3. **Action:** Set **Method** to **Voxel Correlation**, then back to **Soma
   Location**.
   **Expected:** **Depth scale** hides for voxel correlation and reappears for
   soma location, while **Ignore depth (flat map X/Y only)** stays visible for
   both. Switching **Coordinate space** to **CCFv3 Coordinates** hides both.
4. **Action:** With **Algorithm** set to **Hierarchical**, **Clusters** to 10,
   and **Depth scale** at `1.00`, click **Run Clustering**. Color the table by
   cluster and project the somas to the flatmap.
   **Expected:** Clusters are compact patches on the flat map that each still
   cover a range of depths — not hemisphere-wide laminar sheets. This is the
   check automated tests cannot make.
5. **Action:** Set **Depth scale** to `20.00` and re-run.
   **Expected:** Clusters become depth bands: each spans a wide area of the flat
   map but a narrow depth range, reproducing the old depth-dominated behavior on
   purpose.
6. **Action:** Set **Depth scale** back to `1.00`, check **Ignore depth (flat
   map X/Y only)**, and re-run.
   **Expected:** **Depth scale** greys out. Clusters are flat map patches that
   ignore layer entirely, so one cluster contains both superficial and deep
   somas at the same tangential position.
7. **Action:** Set **Algorithm** to **DBSCAN** and read the **Eps** row.
   **Expected:** The label reads **Eps (hemisphere fraction):** with no micron
   suffix and a value of `0.050`. Switching **Coordinate space** to **CCFv3
   Coordinates** restores **Eps (μm):** at `100.0`; switching back restores the
   normalized value. Editing one space's value and returning to the other leaves
   that other value intact.
8. **Action:** Run DBSCAN in flat map space at **Eps** `0.050`, then at `1.000`.
   **Expected:** `0.050` produces multiple clusters plus noise. `1.000` — one
   whole hemisphere height — collapses nearly everything into a single cluster,
   confirming the control now spans a useful range rather than saturating.
9. **Action:** Export the clustering result and inspect the run metadata.
   **Expected:** `distance_metric` reads
   `euclidean_flatmap_isotropic_plus_depth`, or
   `euclidean_flatmap_isotropic` when depth was ignored. A
   `flatmap_normalization` entry records the single `flatmap_divisor` shared by
   both flat map axes, the `depth_divisor_um`, `depth_scale`,
   `include_depth`, `axis_count`, and whether bounds came from `canonical`
   metadata or `observed` data.
10. **Action:** Set **Method** to **Voxel Correlation** with **Y bins** 128 and
    **Depth bin** 25 µm, note the reported rendered-node count from a run, then
    check **Ignore depth (flat map X/Y only)** and run again.
    **Expected:** **Depth bin** greys out while **Y bins** and **Include depth
    -1 plane** stay enabled. The rendered-node count is **identical** across the
    two runs — collapsing changes how nodes are grouped, not which are counted —
    while the occupied-voxel count drops sharply as depth planes merge.
11. **Action:** Color the table by cluster after the collapsed run and project
    the neurons to the flatmap in **2D Heatmap**.
    **Expected:** Clusters group neurons whose flat map footprints overlap, even
    when their arbors sit in different layers. Neurons that overlap in flat map
    projection but not in depth now cluster together, which the uncollapsed run
    cannot do.
12. **Action:** Export the collapsed correlation result and inspect the metadata.
    **Expected:** `distance_metric` reads `one_minus_pearson_r_flatmap_xy`,
    `flatmap_collapse_depth` is `true`, and `flatmap_volume_shape` has two
    entries instead of three.

**Manual verification**

- Status: Partially run — step 9 (export metadata) and the isotropy change
  verified in napari on 2026-08-13; steps 1-8 and 10-12 not run
- Last verified: 2026-08-13 (step 9 and the metric change only)
- Notes: Added on 2026-08-06 and not yet fully exercised in napari. On
  2026-08-11 the voxel-correlation control became **Y bins** with a derived X
  count, so step 10's grid is rectangular and its occupied-voxel counts differ
  from any earlier run. The same date also made the soma-clustering metric
  isotropic: both flat map axes now share one divisor, which removed the shaped
  style's 4.2% anisotropy. Steps 1-8 therefore still need re-running for the
  shaped style, and the **Depth scale** and **Eps** tooltips now say
  "hemisphere height" rather than "width".

  On 2026-08-13 the metric change itself was verified in napari by exporting
  paired clustering runs before and after the fix and comparing them with
  `scripts/compare_clustering_exports.py` (run it with
  `--dir anisotropy_fix_data`; that directory is gitignored, so the workbooks
  are local-only and must be regenerated to repeat this). Step 9's metadata was
  confirmed
  directly: `distance_metric` reads `euclidean_flatmap_isotropic_plus_depth`,
  and `flatmap_normalization` carries a single `flatmap_divisor` with no
  `x_divisor`/`y_divisor`, plus `bounds_source: canonical` and `axis_count: 3`.
  Twelve exports in six pairs were compared. Over all 18,518 somas with
  hierarchical/ward at
  **Depth scale** `1.00`, the shaped style moved 37.5% of somas at k=5 and 38.7%
  at k=10 (ARI 0.44 and 0.47). Two controls held exactly: the square style moved
  0.0% (ARI 1.000000, all ten cluster sizes identical) because its old divisors
  were already equal, and a voxel-correlation pair moved 0.0% because it never
  reaches the soma normalization. The square control is the stronger of the two,
  since it runs the same code path the fix changed under the most
  perturbation-sensitive settings available. `ANISOTROPY_FIX_RESULTS.MD` records
  the full analysis. Automated tests in
  `tests/test_flatmap_depth_normalization.py` cover the shared-divisor math,
  isotropy as a distance property on both styles, agreement with the voxel
  grid's reference axis, depth exclusion versus a zero weight, monotonic
  reweighting, the observed-bounds fallback, and provenance;
  `tests/test_flatmap_clustering_from_parquet.py` covers the collapsed
  correlation, including that collapsing preserves the node count while shrinking
  the grid, and that neurons sharing a footprint across layers correlate at 1.0
  once collapsed but below 0.5 before. The variance rebalance was confirmed
  numerically against `isocortex_total_right_brainglobe_flatmap.parquet`: depth's
  share of the feature-space variance drops from 99.9998% to 11.9% at **Depth
  scale** `1.00` for the bilateral square style, and falls monotonically as the
  scale is lowered. The collapse invariant was confirmed on a 40-neuron subset of
  the same file — 575,098 nodes counted either way, with occupied voxels falling
  from 11,321 to 1,538. None of that shows whether the resulting clusters read as
  sensible anatomy on the canvas, so steps 4-6 and 11 still need eyes. Choosing a
  defensible depth scale is a judgement about the biological question, not
  something the numbers settle — the flat map distortion rules out calibrating it
  against physical distance.

### UC-013: Overlay Cached Brain Regions on a 2D Flat Map

**Capability**

In the two depth-free renders (**2D Heatmap**, **2D Vector**) the user can draw
the regions selected in **Regions** directly onto the flat map, either as a
filled atlas-colored label image or as region outlines. Both are derived at read
time from an existing flatmap region cache, without rebuilding it and without
loading NRRDs, `atlas.annotation`, or BrainGlobe meshes. Before this, a 2D flat
map had no anatomical frame of reference at all.

The filled Labels overlay preserves descendant region identity after depth is
collapsed. Each source region's counts are first summed through depth; when
several descendants occupy the same flatmap column, the region with the largest
count supplies that pixel's label ID, with the smaller ID winning a tie. This
lets **Region Appearance** recolor a child independently while unchanged
children still inherit an overridden parent color. A 2D pixel can represent
only one of its depth-overlapping descendants, so this majority rule is an
explicit part of the view rather than a biological partition of the column.

Outlines keep different semantics: they union all selected descendants under
each directly selected parent. Selecting `Isocortex` therefore produces one
outer Isocortex perimeter even though its filled overlay retains the winning
descendant ID in each pixel.

**Prerequisites**

- Complete UC-003 and keep its version-3 flatmap Parquet.
- Complete UC-004 and keep a compatible flatmap region cache directory.
- A loaded BrainGlobe atlas whose name/version structure catalog matches the
  cache profile (`allen_mouse_25um` v1.2 for the UC-004 cache). Its voxel
  resolution may differ from the cache's.
- Neurons whose arbors span several isocortical areas, so area boundaries are
  visible. `isocortex_total_right_brainglobe_flatmap.parquet` works.
- Start from a clean napari session with the **Neuron Navigator** plugin open.

**Steps and expected results**

1. **Action:** Select a few neuron rows, open **Flatmap**, set **Source** to
   **Precomputed Parquet + Cache** and **Style** to **Both hemispheres, shaped**,
   click **Choose Cache Directory...** and pick the compatible profile, set
   **Render** to **2D Heatmap**, and click **Project to Flatmap**.
   **Expected:** An **Isocortex Flatmap 2D Heatmap** layer appears on the main
   canvas with no plane slider. In **Cached Regions**, **Show Region Labels**,
   **Show Region Outlines**, and **Clear Geometry** are enabled, while **Show
   Region Surfaces** and the **Atlas** combo are disabled. Hovering **Show Region
   Surfaces** explains that cached surfaces are 3D voxel shells and points to
   **Show Region Labels**.
2. **Action:** In **Regions**, set **Query source** to **Atlas Regions**, enable
   **Include child regions**, check only `Isocortex`, and click **Show Region
   Labels**.
   **Expected:** A layer named **Flatmap Region Labels 2D** draws the Isocortex
   footprint with descendant source-region IDs retained. Each pixel shows the
   descendant with the largest depth-summed occupancy in that flatmap column;
   equal counts choose the smaller ID. There is no plane slider or plane
   caption, and the **Flatmap X** / **Flatmap Y** axis captions are shown. The
   status line reports the collapsed bin count, the number of directly selected
   regions represented, and the profile ID.
3. **Action:** Zoom into a boundary between lit heatmap pixels and unlabeled
   background.
   **Expected:** The label edge sits exactly on heatmap pixel boundaries, with no
   half-pixel offset in either axis, even though the flat map's X axis spans twice
   the range of its Y axis in the same number of bins. This is the check
   automated tests cannot make.
4. **Action:** Click **Show Region Outlines**.
   **Expected:** A layer named **Flatmap Region Outlines 2D** draws a single
   closed Isocortex perimeter in the Isocortex color, with **no arrowheads**,
   tracing the same boundary as the label image and including interior holes. The
   camera stays where it was rather than re-centring.
5. **Action:** Uncheck `Isocortex`, check `MOp` and `SSp`, then click **Show
   Region Labels** followed by **Show Region Outlines**.
   **Expected:** Every nonzero label ID belongs to the selected `MOp` or `SSp`
   descendant sets, with overlaps resolved by depth-summed occupancy. Two
   parent-union outline layers appear, named
   `Flatmap Region Outlines 2D: MOp (985)` and
   `Flatmap Region Outlines 2D: SSp (322)`.
6. **Action:** Hover the cursor over a labeled pixel.
   **Expected:** napari's status bar reports the winning descendant/source ID
   stored in that pixel, such as `MOp5`, rather than rewriting every pixel to
   the directly selected parent ID (`985` or `322`).
7. **Action:** Switch **Query source** to **Custom Regions**, check two terminal
   layer regions, and click **Show Region Outlines**.
   **Expected:** One acronym/ID-named, atlas-colored outline layer appears per
   checked terminal region. No structure catalog beyond the loaded atlas is
   required, because terminal selections name their own labels.
8. **Action:** Switch **Render** to **2D Vector**, click **Project to Flatmap**,
   then click **Show Region Outlines**.
   **Expected:** The overlays built in the previous mode were removed when
   **Render** changed and have to be rebuilt. The rebuilt region perimeters and
   the neuron trace vectors sit on the same grid with no offset.
9. **Action:** Switch **Render** to **3D Heatmap** and click **Project to
   Flatmap**.
   **Expected:** The 2D overlays are gone. **Show Region Surfaces** becomes
   available, and **Show Region Outlines** now builds the per-depth 3D outline
   layers — also without arrowheads.
10. **Action:** Return to **2D Heatmap**, select **Recompute from NRRDs** as
    **Source**, and click **Show Region Labels**.
    **Expected:** The action reports that *recomputed* region labels are built on
    the depth grid and names both remedies — choose **Precomputed Parquet +
    Cache**, or switch to a 3D render. No layer is added.
11. **Action:** Return to **Precomputed Parquet + Cache**, select a region with no
    isocortical occupancy (for example a thalamic nucleus), and click **Show
    Region Labels**.
    **Expected:** The action reports that the selection has no occupancy in the
    active flatmap cache. No empty layer is left behind.
12. **Action:** Set **Query source** to **Mask Layer** and click **Show Region
    Labels**.
    **Expected:** The existing actionable error naming **Atlas Regions** and
    **Custom Regions**; no layer is added.
13. **Action:** Click **Clear Geometry**, then **Clear Region Labels**.
    **Expected:** Both 2D overlay families are removed. The **Isocortex Flatmap
    2D Heatmap** and **Isocortex Flatmap 2D Vectors** layers remain.
14. **Action:** Note the `flatmap-region-cache.json` modification time and the
    profile ID reported in the status line before step 2 and again after step 13.
    **Expected:** Both are unchanged. No profile was rebuilt and no manifest was
    rewritten — the 2D overlays are read-time derivations of the occupancy
    arrays.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: Added on 2026-08-10 and not yet exercised in napari. On 2026-08-11 the
  region cache moved to per-axis bin counts and manifest format version 2, so
  this use case needs a cache rebuilt under the new format; the 2D label image
  is now `Y bins x X bins` rather than square. Automated tests in
  `tests/test_flatmap_region_cache.py` cover the depth-collapse arithmetic,
  source-ID retention beneath a single selected root, majority and tie
  resolution, assignment of members to several outline roots from either an
  explicit descendant map or a structure catalog, the refusal when neither is
  available, reuse of the stored depth-free perimeter tracer, agreement between
  the flat and Allen-layer footprints, and that neither materializer changes
  `profile_id` or the manifest. `tests/test_flatmap_widget.py` covers the
  button-enable matrix in both flat modes, agreement between the two gating call
  sites across five render modes and two sources, the 2D layer names, descendant
  colormap IDs, axis captions, plane-mode metadata, `ndisplay`, absence of
  `scale`/`translate`, a single materializer call for a multi-region outline
  request, and overlay retirement. Verified numerically against
  `/Users/lawrimorejg/Downloads/flatmap_cache_25` (shaped, 240 leaf regions):
  the `Isocortex` union occupies 43,967 bins with a 1,506-segment perimeter in
  0.105 s; 43,863 columns contain competing descendant sources (L5 wins 53% and
  L2/3 33% under the majority rule measured before this UI change). Those
  existing measurements need to be rerun against the retained-ID Labels layer.
  None of this shows whether the overlays or child recoloring read correctly on
  the canvas, so steps 2-6 and 8 still need eyes.

### UC-014: Control and Share Region Appearance Across CCFv3 and Flatmap Views

**Capability**

The user can give atlas subregions a consistent visual identity across CCFv3
segmentation and mesh layers and flatmap label, surface, and outline layers.
The **Region Appearance** panel edits one atlas-scoped palette without changing
which region IDs are selected for neuron queries. Edits remain staged until
**Apply** is clicked, so several colors, fill/outline visibility settings, and
opacity values can be reviewed as one change.

Each region uses its own atlas color when no override exists. An explicit
parent setting is inherited by descendants; a child can override individual
properties, select **Use Atlas Color** to break inherited color, or select
**Inherit** to remove all of its explicit settings. Filled labels stay in one
combined napari Labels layer: hidden IDs become transparent colormap entries
rather than separate volumes. CCFv3 has no per-region contour layer, so outline
settings affect flatmap outlines only.

The depth-free 2D flatmap retains descendant/source IDs in that combined Labels
layer. Where descendants overlap in one XY column, depth-summed occupancy picks
one displayed ID by majority, with the smaller ID breaking a tie. Child colors
therefore remain independently addressable, but only the winning descendant is
visible at a given 2D pixel; the 3D layer continues to preserve depth planes.

**Prerequisites**

- A loaded Allen Mouse Brain Atlas. Use `allen_mouse_25um` v1.2 when following
  UC-003 and UC-004 with their example data and cache.
- A neuron Parquet with atlas `region_id` data, plus at least two selected atlas
  areas or custom terminal regions. `MOp` and `SSp` make a useful pair.
- To check both coordinate spaces, complete UC-003 and UC-004 and open the
  compatible flatmap region cache.
- Start from a clean napari session with the **Neuron Navigator** plugin open.

**Steps and expected results**

1. **Action:** Load the atlas and Parquet, open **Regions**, set **Query source**
   to **Atlas Regions**, select `MOp` and `SSp`, then expand **Region
   Appearance** at the bottom of the tab.
   **Expected:** The query buttons and status text remain above the collapsed
   section. Expanding **Region Appearance** shows a searchable tree containing
   the selected regions and their descendants. Each row shows a color swatch,
   **Fill**, **Fill %**, **Outline**, **Outline %**, and the setting source.
   Regions without overrides display their own atlas colors.
2. **Action:** From **Reference**, enable **Show selected region segmentation**
   and **Show selected region meshes**. In **Flatmap**, create cached region
   labels, surfaces, and outlines in render modes that support each overlay.
   **Expected:** The initial CCFv3 and flatmap fills and flatmap outlines agree
   on each region's atlas color. Filled regions share combined Labels layers;
   the plugin does not create one full annotation volume per region. The
   **Region Meshes** message explains that only directly selected top-level
   parents receive meshes; descendant meshes are omitted because of their
   higher loading and rendering cost, even though descendant IDs remain visible
   in segmentation.
3. **Action:** Select the `MOp` and `SSp` rows in **Region Appearance** and click
   **Assign Distinct Colors**.
   **Expected:** The two swatches change to deterministic colors suitable for
   napari's dark canvas, and the status reports unapplied changes. Existing
   overlays do not change yet, and the selected query regions and query results
   are unchanged.
4. **Action:** Choose a child row, click its swatch to set a custom color, set
   **Fill** to **Hide**, and give another row partial **Fill %** and **Outline
   %** values. Select a child of a custom-colored parent and click **Use Atlas
   Color**.
   **Expected:** The source indicators distinguish custom, inherited, and
   explicit atlas color. The atlas-color child keeps its own atlas color even
   though its parent has a custom color. All changes remain staged.
5. **Action:** Click **Apply**.
   **Expected:** Existing CCFv3 segmentation/mesh fills and flatmap
   label/surface fills update to the shared colors, fill visibility, and fill
   opacity. Existing flatmap outlines update to the same colors and their
   outline visibility/opacity. In the depth-free **Flatmap Region Labels 2D**
   layer, a child override recolors pixels whose retained label value is that
   child ID, while sibling IDs without an override inherit the parent setting.
   Hidden combined-label entries are transparent; visible IDs remain available
   as label values. No CCF contour layer appears. No atlas volume, NRRD,
   flatmap projection, mesh geometry, or region cache is rebuilt.
6. **Action:** Change a Reference opacity slider and toggle a styled napari
   layer's eye icon, then apply another region edit.
   **Expected:** The family opacity remains the global value and each per-region
   opacity remains a multiplier beneath it. A layer hidden with napari's eye
   icon stays globally hidden after **Apply**. Region visibility still does not
   alter query selection.
7. **Action:** Stage a different color or visibility value and click **Revert**.
   **Expected:** The staged controls return to the applied palette and every
   rendered layer remains unchanged.
8. **Action:** Click **Export Applied Palette...**, save the JSON file, clear or
   change several overrides with **Inherit** and **Apply**, then click **Import
   Palette...** and choose the exported file. Try **Merge**, repeat the import,
   and try **Replace**; click **Apply** after each accepted import.
   **Expected:** Import first summarizes matching overrides and offers
   **Merge**, **Replace**, and **Cancel**. **Merge** preserves unrelated staged
   overrides; **Replace** removes them. The imported palette remains staged
   until **Apply**, after which the two coordinate spaces again agree.
9. **Action:** Attempt to import a palette whose atlas name differs, then one
   with the same name but a different version, and one containing an unknown
   numeric region ID.
   **Expected:** An atlas-name mismatch is rejected. A version mismatch requires
   confirmation before matching IDs are imported. Unknown IDs are skipped and
   their count is reported.
10. **Action:** Stage an unapplied edit and click **Save Project As...**. Test
    **Cancel**, then repeat and test **Discard**, and finally repeat with an edit
    and test **Apply**.
    **Expected:** **Cancel** stops the save. **Discard** saves the previously
    applied palette and removes the draft. **Apply** first updates the overlays
    and saves that newly applied palette, so the project never silently records
    an ambiguous draft.
11. **Action:** Close the project, click **Load Project...**, and reopen the
    saved bundle. Also open a project bundle created before Region Appearance
    was added.
    **Expected:** The new project restores its applied palette and newly created
    overlays use it. The older project opens normally with an empty palette and
    atlas-color defaults.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: Added on 2026-08-20. Automated tests cover ancestry resolution,
  explicit atlas-color breaks, transparent combined-label entries, per-region
  mesh and flatmap styling, retained descendant IDs and inherited/overridden
  colors in a 2D Labels layer, in-place restyling without cache/NRRD work,
  palette JSON and project round trips, import filtering, and old-project
  compatibility. Qt-dependent interaction and visual co-registration still
  require this manual napari run.

### UC-015: Compare Cluster Assignments in an Interactive Board

**Capability**

The user can compare as many as 16 saved cluster mappings in a separate **SWC
Viewer Comparison Board** window without duplicating full napari viewers. Each
cell shows flatmap somas, flatmap arbor density, CCFv3 somas, or an existing
Analysis heatmap set. The board keeps each cell bound to a saved assignment ID,
matches display colors to a chosen reference by shared `file_id` membership,
and links navigation only where the cells share coordinate provenance.

This is intended for judging how alternate clustering runs split or preserve
the same neuron cohort. It is a visual comparison surface, not a replacement
for full napari inspection: v1 supports pan, zoom, and coordinate/value hover,
but no point picking or brush selection.

**Prerequisites**

- A loaded neuron Parquet containing CCFv3 coordinates. To test flatmap cells,
  use the v3-prepared Parquet from UC-003 with both shaped and square metadata.
- A loaded atlas matching the CCFv3 data, such as `allen_mouse_25um` v1.2.
- At least two saved assignment sets produced as in UC-007. Use assignments
  with partly overlapping cohorts or relabeled clusters to make matching
  visible.
- To test CCF heatmap cells, use **Analysis** to create one complete
  cluster-heatmap set for each assignment with identical region, node-type,
  soma-radius, depth-axis, and bin-factor settings.
- If those heatmaps filter `type = 2`, interpret the result as **axon-typed**,
  not confirmed axon: this dataset contains dendritic projections mislabeled
  as type 2, so a biological comparison still needs visual verification.
- Start with the **Neuron Navigator** plugin open. A 4×4 check needs enough screen
  space to keep titles and hover readouts legible.

**Steps and expected results**

1. **Action:** Open **Compare** and click **Open Comparison Board**.
   **Expected:** A separate **Neuron Navigator Comparison Board** window opens with a
   2×2 layout and a selected-cell inspector. Closing it hides the same board;
   clicking **Open Comparison Board** again restores its cells and camera state.
2. **Action:** Set **Rows** and **Columns** through the boundary layouts 1×1,
   1×4, 4×1, and 4×4. Use **Add Cell**, **Duplicate**, **Move Earlier**, **Move
   Later**, and **Remove**.
   **Expected:** The board accepts only values 1–4 on each axis and at most 16
   cells. It refuses to shrink below the current cell count and reports that
   cells must be removed first. Duplicates retain the source recipe but get a
   distinct title and board identity.
3. **Action:** Add two cells. For each, set **Map** to **Flatmap somas**, choose
   a different **Assignment**, set a useful **Title**, choose **Both
   hemispheres, shaped**, and click **Apply and Render**. Set **Reference** to
   the first run.
   **Expected:** Both cells show the assignments' soma centroids and report
   assigned and omitted/unassigned neuron counts. The reference keeps its saved
   palette. The other cell's legend reports the maximum-overlap mappings to
   reference clusters; split, sparse, and unmatched clusters remain separate
   without changing either saved assignment.
4. **Action:** Return to the napari **Compare** tab. Under **Cluster Membership
   Comparison**, choose the first run as **Reference assignment** and the
   second as **Candidate assignment**.
   **Expected:** The tab reports ARI, NMI, optimally matched agreement, shared
   and one-sided cohort counts, a reference-by-candidate cluster overlap
   matrix, and per-pair Jaccard values. Changing **Reference assignment** also
   changes the board's color-matching reference. All membership joins use
   `file_id`; neurons unassigned in either run are reported in coverage and
   excluded from ARI, NMI, and matched agreement.
5. **Action:** Pan and zoom either shaped flatmap cell, then change one cell to
   **Both hemispheres, square** and click **Apply and Render**.
   **Expected:** Shaped cells navigate together. The square cell does not join
   that camera group. No flatmap cell ever links to a CCFv3 cell.
6. **Action:** Change one cell to **Flatmap arbor heatmap**. Set **Y bins** to
   256 for shaped, apply, then repeat with square.
   **Expected:** The inspector shows the resolved **X bins** as 491 for the
   canonical shaped bounds and 512 for square. Each image is `Y × X`, aligns
   with the corresponding soma points without a transpose, and is not
   horizontally squashed. Returning to a saved cell or project uses its stored
   X count verbatim.
7. **Action:** Configure three cells as **CCFv3 somas**, using **Coronal**,
   **Sagittal**, and **Horizontal** under **CCF plane**, and select **Full
   projection**.
   **Expected:** Each point cloud is projected through the full hidden axis.
   Coronal displays left-right horizontally and dorsal-ventral vertically;
   sagittal displays rostral-caudal horizontally and dorsal-ventral vertically;
   horizontal displays left-right horizontally and rostral-caudal vertically.
8. **Action:** Set two same-plane CCF cells to **Slice / slab**, use the same
   **Slab thickness**, leave **Link when compatible** checked, and change
   **Slice position** in one cell before clicking **Apply and Render**.
   **Expected:** The physical slice position is copied to the compatible cell,
   both cells retain only somas in that slab, and their pan/zoom stays linked.
   Uncheck **Link when compatible** for one cell to opt it out.
9. **Action:** Set a cell to **Existing CCFv3 heatmap(s)** and choose a complete
   cluster set under **Heatmap source**. Repeat for the second assignment with
   the same geometry and filters.
   **Expected:** Cluster volumes are reduced with count-preserving sums through
   the chosen slab or the full projection axis. Compatible cells use the same
   displayed count maximum when **Share comparable intensity** is checked.
   A set with mismatched atlas, shape, scale, assignment, or filter provenance
   is not composed as one source.
10. **Action:** Enable **Use per-cell maximum** in one heatmap cell, set
   **Maximum**, and apply. Then uncheck **Share comparable intensity** globally.
   **Expected:** The per-cell maximum is used and an `intensity override` badge
   appears in that cell. Disabling the global option gives each non-overridden
   heatmap its own observed maximum.
11. **Action:** Hover over soma and heatmap cells, click between cells, and try
    dragging and scrolling in each plot.
    **Expected:** The footer reports physical/flatmap coordinates and heatmap
    count where applicable. Clicking selects the cell for the inspector. Drag
    and scroll pan/zoom; no neuron selection, brush, or napari layer is created.
12. **Action:** Delete an assignment used by a cell, or delete an Analysis
    heatmap layer referenced by a cell, then click **Refresh Sources**.
    **Expected:** The affected cell becomes an explanatory **Source
    unavailable** placeholder. A same-named assignment or layer is never used
    as a substitute for the missing stable ID; other cells still render.
13. **Action:** Click **Export Board...** and save `comparison.png`.
    **Expected:** The labeled grid is written as `comparison.png`, and
    `comparison.comparison.json` records the complete board recipe, source file
    signature, cohort counts, source IDs, original and display palettes,
    overlap mappings, intensity maxima, and per-cell provenance. Its
    `membership_comparisons` section contains one ARI/NMI/agreement, overlap
    matrix, coverage, and Jaccard record for each unique non-reference
    assignment represented on the board; repeated cells share one record and
    retain all source cell IDs.
14. **Action:** Save the project, close and reload it, then reopen **Compare**.
    Also load a project created before Comparison Board support.
    **Expected:** The new project restores its one board after assignments,
    heatmaps, atlas, and flatmap references are available. The legacy project
    loads normally with an empty 2×2 board. Enhanced Parquet export remains
    unchanged and does not contain board state.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: Added on 2026-08-25. Automated tests cover `file_id`-only overlap
  matching (including duplicate display `neuron_id` values), relabeled/split/
  sparse/unmatched clusters, CCF plane orientation and count-preserving slab
  reductions on non-cubic volumes, stored rectangular flatmap recipes, every
  layout through 4×4, cell recipe operations, shared-intensity grouping and
  overrides, stable heatmap source metadata, missing-source refusal, project
  and legacy-project round trips, cluster membership statistics and cohort
  accounting, and export-sidecar provenance. The 4×4 visual result, anatomical
  screen orientation, pyqtgraph navigation links, metrics-table layout, window
  close/reopen behavior, and PNG output still require this manual napari run.

### UC-016: Compare Multiple Flatmap Windows Side by Side

**Capability**

The user can keep several complete napari flatmap viewers open at once to
compare different neuron selections, flatmap styles, render modes, clusters, or
display settings side by side. **Project to Flatmap** continues to update the
newest flatmap viewer, while **Project in New Window** preserves it and renders
into another viewer. Older viewers remain independently navigable snapshots;
projection, soma, and cached-region controls in the plugin's **Flatmap** tab
target the newest viewer, while **Heatmap Appearance** can target any open
viewer.

**Prerequisites**

- Load a neuron Parquet with valid flatmap coordinates. For **Precomputed
  Parquet + Cache**, use a version-3 Parquet produced by UC-003.
- Select enough neurons or clusters to make two visibly different projections.
- Start from a clean napari session with the **Neuron Navigator** plugin open.

**Steps and expected results**

1. **Action:** In **Flatmap**, choose the first neuron selection and render
   options, then click **Project to Flatmap**.
   **Expected:** A populated viewer named **Neuron Navigator Flatmap** appears.
   The main napari viewer is unchanged.
2. **Action:** Change the selection, style, or render mode, then click **Project
   in New Window**.
   **Expected:** Changing **Render** does not remove or alter the first viewer.
   A second populated viewer named **Neuron Navigator Flatmap 2** appears after
   projection. The first viewer, all of its layers, and its camera remain
   unchanged, so the two projections can be arranged side by side.
3. **Action:** In the plugin tab, expand **Heatmap Appearance**, choose each
   viewer in **Window**, and select one or more of that viewer's heatmaps. Apply
   **Enhance Fine Projections** or **Reset Gamma**. Then use **Add Soma** or a
   cached region action.
   **Expected:** The heatmap list changes to show only the selected window's
   heatmaps, and the gamma action changes only the selected layers in that
   window. Returning to a window restores its heatmap selection. **Add Soma**
   and cached region actions continue to change only the newest flatmap viewer.
4. **Action:** Move the plane slider and camera independently in both flatmap
   viewers.
   **Expected:** Each viewer keeps its own dimensions and camera. Its on-canvas
   depth or Allen-layer caption follows that viewer's slider independently.
5. **Action:** Close the older viewer, leaving the newest open, then click
   **Project in New Window** again with another configuration.
   **Expected:** Closing the older viewer does not close or modify the newest
   viewer. The next successful projection opens another independently tracked
   flatmap viewer without exposing a blank window.
6. **Action:** Cause a validation error while using **Project in New Window**,
   such as selecting an incompatible input or render option.
   **Expected:** The failed target is never shown as an empty viewer. Every
   already populated comparison viewer remains open and unchanged, and the
   status reports the corrective error.

**Manual verification**

- Status: Partially run — render-mode transition and native stability confirmed
- Last verified: 2026-09-08
- Notes: Manually confirmed on a MacBook Pro with an M3 chip that changing
  **Render** after creating the initial flatmap window no longer shuts down
  napari. Automated tests cover retaining two populated viewers, active-viewer
  scoping, per-window heatmap selection and gamma changes, independent plane
  captions, retiring an older closed viewer without changing the active viewer,
  stale-render generation guards, preserving a scene while **Render** changes,
  and deferring incompatible GPU-layer removal outside Qt selection events. The
  remaining UC-016 steps have not been manually run.

## Use-Case Template

Copy this section when adding a use case. Remove guidance in parentheses and
replace `XXX` with the next sequential identifier.

### UC-XXX: Descriptive Workflow Name

**Capability**

(Explain what the plugin lets the user accomplish and why the workflow is
useful.)

**Prerequisites**

- (List required input files, atlas data, existing layers, configuration, or
  starting state.)

**Steps and expected results**

1. **Action:** (Describe one user action using the visible UI labels.)
   **Expected:** (Describe the observable result of that action.)
2. **Action:** (Continue until the workflow and its final outcome are
   complete.)
   **Expected:** (Describe the observable result.)

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None
