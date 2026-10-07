# Neuron Navigator Poster Workflow: MOs Soma and Projection Clustering

**Status:** Living working document  
**Created:** 2026-10-06  
**Purpose:** Record the exact steps, parameters, outputs, provenance, and interpretation used to build the Neuron Navigator poster demonstration requested by Chip. This document is intentionally procedural so it can later be converted into a user manual / repository use case.

---

## 1. Provenance tags used in this document

To keep observations separate from interpretation, each important statement should be traceable to one of these source classes:

- **[COLLABORATOR]** — workflow or analysis direction supplied by Chip.
- **[ABSTRACT]** — content from the submitted poster abstract, *Neuron Navigator: a napari Plugin for Interactive Cluster Analysis of Large Numbers of Neuron Reconstructions*.
- **[REPOSITORY]** — behavior documented or implemented in the `napari-neuron-navigator` repository.
- **[LITERATURE]** — biological background from published literature, especially Hooks et al. 2018.
- **[RUN RECORD]** — values observed during the actual analysis: neuron counts, parameters, screenshots, exports, dates, etc.
- **[INTERPRETATION]** — biological or methodological interpretation added after inspecting the results.

When this workflow is converted into a manual, the provenance tags can be removed from the user-facing steps but retained in the development notes.

---

## 2. Scientific goal

**[COLLABORATOR]** Demonstrate a hierarchical analysis of neurons whose somas lie in the murine secondary motor cortex (`MOs`):

1. Find all neurons with somas in MOs (~1,900 expected from the current dataset).
2. Cluster those neurons by soma proximity to establish a soma-based topographic organization.
3. Choose one spatially coherent soma cluster.
4. Re-cluster neurons in that local cluster by similarity of their long-range projection patterns outside MOs.
5. Divide the selected population into **5 projection clusters**.
6. Visualize where the neurons in each projection cluster project.

**[LITERATURE]** Biological motivation: Hooks et al. 2018 studied the organization of corticostriatal projections from motor cortex and discussed distinct layer-5 IT-type and PT-type projection neurons. In that paper, the vibrissal primary motor cortex (`vM1`) site corresponds to the Allen Reference Atlas `MOs` territory used in this demonstration.

**[COLLABORATOR — 2026-10-07]** The scientific demonstration will document our attempts to distinguish **IT-like and PT-like projection patterns** among MOs neurons using soma clustering followed by voxel-correlation clustering. Soma clustering selects a local population; projection clustering tests whether its reconstructed output patterns separate. IT and PT are the biological comparison of interest, not an exhaustive list of neurons in MOs or labels supplied to the clustering. No layer-5 restriction is implied by the MOs query; record any layer/depth filter actually applied.

**Working poster message [INTERPRETATION]:**

> Neuron Navigator links anatomical selection, local topographic clustering, and single-neuron projection clustering to test for distinct long-range output motifs among neurons occupying the same cortical territory.

Do **not** assume beforehand that the five data-driven projection clusters are five canonical biological cell types. The five-cluster solution is an analysis resolution requested for the demonstration; biological labels should be assigned only after inspecting the resulting target patterns.

---

## 3. Software provenance

### Repository

- **Repository [REPOSITORY]:** https://github.com/nimh-dsst/napari-neuron-navigator
- **Branch used [RUN RECORD]:** `main` / TODO
- **Git commit [RUN RECORD]:** TODO — record `git rev-parse HEAD` at the time the poster analysis is frozen.
- **Plugin/package version [RUN RECORD]:** TODO
- **Project bundle [RUN RECORD]:** TODO (`.nnproj` if saved)

### Environment

**[REPOSITORY]** The project uses **Pixi** as the source of truth for environment and dependency management. The current repository documentation specifies Python 3.11 or 3.12 and napari 0.9.x.

Record the actual environment used for the poster:

```text
Operating system: TODO
Python: TODO
napari: TODO
napari-neuron-navigator: TODO
pixi lock / environment date: TODO
```

Recommended reproducibility commands:

```bash
git rev-parse HEAD
pixi run napari
```

If a fully frozen environment record is needed later, add a `pixi list` or equivalent dependency export here.

---

## 4. Input-data provenance

### Reconstruction datasets

**[ABSTRACT]** Neuron Navigator was developed using **18,621 published murine cortical neuron reconstructions** registered to Allen CCFv3, stored in a Parquet-based database representation. The plugin is designed to accept any CCFv3-registered SWC files.

**[RUN RECORD / SOURCE DATA]** The 18,621-neuron development collection is assembled from two Brain Science Data Center (Chinese Academy of Sciences) datasets distributed as CCFv3-registered SWC reconstructions:

1. **12,264 single-neuron projectomes of mouse cortex**  
   DOI: https://doi.org/10.12412/BSDC.1747279998.20001  
   The previously published 6,357 prefrontal-cortex neurons are **not** included in this dataset.

2. **6,357 reconstructed single neurons in mouse prefrontal cortex**  
   DOI: https://doi.org/10.12412/BSDC.1690164952.20001  
   Of these, **1,920 neurons contain both dendrite and axon morphologies**; the remaining **4,437 contain only axon morphologies**.

Together these source collections account for the **18,621 reconstructions** used as the Neuron Navigator development dataset.

> **Important morphology-coverage note:** these source datasets are primarily single-neuron **projectome/axon** datasets. Dendrites are annotated only for a subset of the SWCs. Do **not** assume that absence of dendritic nodes means a biological absence of dendrites; in many reconstructions, dendrites were simply not included in the released morphology. Any dendrite-dependent analysis must first identify and restrict itself to reconstructions that actually contain dendritic annotations.

Record the exact file used for this poster analysis:

```text
Input Parquet filename: TODO
Absolute/source path: TODO
File size: TODO
SHA-256: TODO
Number of unique neurons (`file_id`): TODO
Original reconstruction source/publication/DOI: TODO
Coordinate space: Allen CCFv3
Hemisphere normalization or flipping applied: TODO
Flatmap/depth preprocessing present: TODO
```

### Identity rule

**[REPOSITORY — critical]** `file_id` is the only safe per-neuron key in the development dataset. `neuron_id` is not unique and is intended for display, not grouping or joins.

For every exported cluster assignment or derived table, preserve `file_id`.

### Morphology-coverage and node-type caveats

**[RUN RECORD / SOURCE DATA — critical]** Dendritic morphology is not available for every reconstruction. The prefrontal-cortex source dataset contains both dendrite and axon morphology for only **1,920 of 6,357 neurons**; the other **4,437 reconstructions are axon-only**. The 12,264-neuron source dataset should likewise be treated as a projectome dataset unless dendritic coverage has been explicitly verified for a given SWC. Therefore:

- do not compare dendritic morphology across the full 18,621-neuron dataset without first filtering for reconstructions that actually contain dendritic annotations;
- do not interpret a reconstruction lacking dendritic nodes as a neuron that biologically lacks dendrites;
- record the number of neurons with dendritic annotations for any dendrite-dependent analysis.

**[REPOSITORY — critical]** Separately, in the current development dataset, the SWC `type` field is not perfectly reliable as a biological compartment label. Some morphologically dendritic processes are typed `2` (axon), and some neurons contain no type-2 nodes at all. Therefore:

- describe filtered nodes as **“axon-typed nodes”** unless their biological identity has been independently verified;
- record how many neurons are excluded by any node-type filter;
- visually inspect representative neurons before making an IT/PT or axon-specific biological claim from `type` alone.

These caveats are particularly important for the projection-clustering portion of the poster: the workflow is well matched to an axon/projectome dataset, but dendrite-based interpretation requires explicit annotation checks.

---

## 5. Atlas and flatmap provenance

**Original flatmap method [LITERATURE]:** Bolaños-Puchet S, Teska A, Hernando JB, Lu H, Romani A, Schürmann F, Reimann MW. (2024). Enhancement of brain atlases with laminar coordinate systems: Flatmaps and barrel column annotations. *Imaging Neuroscience*, **2**, imag–2–00209. https://doi.org/10.1162/imag_a_00209

**Published lookup data [SOURCE DATA]:** Bolaños-Puchet S, Teska A, Reimann MW. (2024). *Enhanced atlases and flatmaps of rodent neocortex* (v4.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11218079. This version was published on 2024-05-20 and supplies the mouse lookup fields on a **10 µm CCFv3 grid**. The 25 µm files used locally are derived products; cite the paper and this source-data release when describing them.

**Code provenance [LITERATURE + REPOSITORY]:** The paper identifies [BlueBrain/atlas-enhancement](https://github.com/BlueBrain/atlas-enhancement) as its method implementation. The 25 µm conversion utility is [resample_ccf_coordinate_fields.py at commit ea735eacdfe8afc40d006d80caefd54d6b30fdd3](https://github.com/joshlawrimore/atlas-enhancement/blob/ea735eacdfe8afc40d006d80caefd54d6b30fdd3/resample_ccf_coordinate_fields.py) in the separate [joshlawrimore/atlas-enhancement](https://github.com/joshlawrimore/atlas-enhancement) fork, checked out locally at `/Users/lawrimorejg/repos/atlas-enhancement`. See Step 0F for the script hash, conversion parameters, and historical-run provenance limitations.

Record exactly what atlas and lookup assets were used.

```text
BrainGlobe / Allen atlas name: TODO
Atlas version: TODO
Voxel resolution: TODO
Allen CCF version: CCFv3
Flatmap style: TODO (`both_shaped` or `both_square` if applicable)
Flatmap lookup directory: TODO
Flatmap lookup-set ID / hashes: TODO
Depth lookup file/hash: TODO
Region-cache directory/profile: TODO
```

**[REPOSITORY]** Whole-Parquet flatmap preprocessing can store bilateral shaped/square flatmap coordinates and cortical depth in the Parquet metadata. If a precomputed flatmap region cache is used, record the cache profile because the cache fixes the render bounds and binning.

**[REPOSITORY]** Bilateral flatmaps span both hemispheres along the flatmap x-axis. Current code uses isotropic flatmap scaling rather than independently normalizing x and y. Therefore, for soma clustering, record the flatmap style, distance metric, and depth scaling exactly.

---

# SETUP WORKFLOW

## Step 0A — Assemble the source SWC collection

**Goal:** Place the SWC reconstructions from both Brain Science Data Center source datasets into one input directory for conversion by Neuron Navigator.

### Action

1. Download and extract the SWC files from both BSDC datasets listed in Section 4:
   - 12,264-neuron whole-cortex projectome dataset: `https://doi.org/10.12412/BSDC.1747279998.20001`
   - 6,357-neuron prefrontal-cortex projectome dataset: `https://doi.org/10.12412/BSDC.1690164952.20001`
2. Create a new directory that will serve as the combined raw-SWC input folder.
3. Copy the SWC files from both extracted datasets into that directory.
4. **Before allowing any overwrite, check for duplicate filenames.** Preserve the original distributed filenames whenever possible because filenames are part of the practical provenance trail back to the source archives.
5. Count the resulting SWC files and record the value below. If each released reconstruction is represented by one SWC, the expected combined count is **18,621** (`12,264 + 6,357`). Investigate any discrepancy rather than assuming the merge is complete.

### Record

```text
Whole-cortex source download/extraction path: TODO
Prefrontal source download/extraction path: TODO
Combined SWC directory: TODO
Whole-cortex SWCs copied: TODO
Prefrontal SWCs copied: TODO
Combined SWC count: TODO (expected 18,621)
Duplicate filenames encountered: TODO
Renamed files, if any: TODO
Date assembled: TODO
```

### Provenance note

**[RUN RECORD / SOURCE DATA]** The combined directory is a convenience working copy. The two BSDC datasets remain distinct source datasets and should continue to be cited separately. Do not treat the merged folder as a new independently published dataset.

**[RUN RECORD / SOURCE DATA — critical]** Most of these SWCs are projectome/axon reconstructions rather than complete neuron morphologies. Only a subset contains annotated dendrites; see Section 4 before using dendrite-dependent analyses.

---

## Step 0B — Clone the Neuron Navigator repository

**Goal:** Obtain the exact plugin source used for the analysis and preserve its Git revision for reproducibility.

### Action

From a terminal:

```bash
git clone https://github.com/nimh-dsst/napari-neuron-navigator.git
cd napari-neuron-navigator
```

Record the checked-out branch and commit:

```bash
git branch --show-current
git rev-parse HEAD
```

Unless the poster analysis intentionally uses another revision, work from the repository state that was current when the analysis was run and record the commit rather than relying only on the moving `main` branch name.

### Record

```text
Repository clone path: TODO
Branch: TODO
Commit SHA: TODO
Clone/update date: TODO
```

---

## Step 0C — Install Pixi

**Goal:** Install the package/environment manager used by the repository.

**[REPOSITORY]** The repository README specifies **Pixi** as the environment and dependency manager and points users to the official Pixi installation instructions. The repository lock file is used to resolve the supported Python/napari environment; do not create a separate ad-hoc `pip` or Conda environment for this workflow unless there is a documented reason.

### Linux / macOS

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

After installation, restart the terminal/shell if necessary so the updated `PATH` is visible, then verify:

```bash
pixi --version
```

Official installation documentation: `https://pixi.sh/latest/#installation`

### Record

```text
Pixi version: TODO
Installation date: TODO
Operating system: TODO
```

---

## Step 0D — Resolve the repository environment and launch Neuron Navigator

**Goal:** Use the repository-defined Pixi environment rather than installing dependencies manually.

From the cloned repository root, optionally build/install the plugin in editable mode:

```bash
pixi run build
```

Launch napari with Neuron Navigator available:

```bash
pixi run napari
```

**[REPOSITORY]** `pixi run build` installs the package in editable development mode. `pixi run napari` depends on the build task, so it can build the package automatically when needed.

### Record

```text
`pixi run build` completed: yes/no
`pixi run napari` completed: yes/no
napari version: TODO
Plugin version/commit: TODO
Setup errors or warnings: TODO
```

---

## Step 0E — Download the flatmap files

**Goal:** Obtain the published mouse lookup fields before preparing the local 25 µm derivative.

**[SOURCE DATA]** Use [Zenodo v4.1, DOI 10.5281/zenodo.11218079](https://doi.org/10.5281/zenodo.11218079), associated with the Bolaños-Puchet et al. (2024) paper cited in Section 5. The record supplies mouse and rat archives; this workflow uses `mouse_isocortex_enhanced.zip` (listed size: 2.9 GB).

### 0E.1 — Choose the working directories

**Prerequisites:** Git, Pixi, an internet connection, and the Neuron Navigator checkout/environment from Steps 0B–0D. These terminal examples use macOS/Linux shell syntax. Run Steps 0E–0G in the same terminal, in order, and stop if a command fails. If you reopen the terminal, rerun this variable-definition block first.

Edit `NN_REPO` to your existing Neuron Navigator checkout. Choose a fresh `FLATMAP_WORK` directory with space for the archive, extracted NRRDs, dependencies, Allen reference data, and derived files. All subsequent paths are derived from these two settings:

```bash
NN_REPO="/absolute/path/to/napari-neuron-navigator"
FLATMAP_WORK="$HOME/neuron-navigator-flatmap-setup"
FLATMAP_DOWNLOAD_DIR="$FLATMAP_WORK/downloads"
FLATMAP_SOURCE="$FLATMAP_WORK/source/mouse_isocortex_enhanced"
FLATMAP_CODE="$FLATMAP_WORK/atlas-enhancement"
FLATMAP_PREP_ENV="$FLATMAP_WORK/preparation-env"
FLATMAP_OUTPUT="$FLATMAP_WORK/resampled-25um"
FLATMAP_LOOKUPS="$FLATMAP_WORK/lookups-25um"
mkdir -p "$FLATMAP_DOWNLOAD_DIR"
```

### 0E.2 — Download the mouse archive

Open the [version-specific record](https://zenodo.org/records/11218079), locate **Files**, and download `mouse_isocortex_enhanced.zip` into `$FLATMAP_DOWNLOAD_DIR`. Alternatively, use this command to save the same archive there:

```bash
curl --fail --location --retry 3 \
  --output "$FLATMAP_DOWNLOAD_DIR/mouse_isocortex_enhanced.zip" \
  "https://zenodo.org/records/11218079/files/mouse_isocortex_enhanced.zip?download=1"
```

### 0E.3 — Verify and extract the three source fields

Verify the archive against Zenodo's published MD5 before extracting it. This command uses Neuron Navigator's Python environment:

```bash
pixi run --manifest-path "$NN_REPO/pixi.toml" python \
  - "$FLATMAP_DOWNLOAD_DIR/mouse_isocortex_enhanced.zip" <<'PY'
from pathlib import Path
import hashlib
import sys

archive = Path(sys.argv[1])
with archive.open("rb") as stream:
    actual = hashlib.file_digest(stream, "md5").hexdigest()
expected = "2a0223c1b8fd383a2f589202f1c6376d"
if actual != expected:
    raise SystemExit(f"Archive checksum mismatch: {actual}")
print(f"Verified MD5: {actual}")
PY
```

**Expected result:** `Verified MD5: 2a0223c1b8fd383a2f589202f1c6376d`. If it differs, resolve the download mismatch before proceeding.

Extract the required files, retaining the archive and original filenames. The member paths below were checked against the locally available archive. `unzip -n` leaves existing extracted files untouched; use a fresh workspace for a new preparation run.

```bash
unzip -n "$FLATMAP_DOWNLOAD_DIR/mouse_isocortex_enhanced.zip" \
  "mouse_isocortex_enhanced/flatmap_both_shaped.nrrd" \
  "mouse_isocortex_enhanced/flatmap_both_square.nrrd" \
  "mouse_isocortex_enhanced/depth.nrrd" \
  -d "$FLATMAP_WORK/source"
ls -lh "$FLATMAP_SOURCE"
```

**Expected result:** `$FLATMAP_SOURCE` contains these three **10 µm source NRRDs**:

| File | Meaning |
|---|---|
| `flatmap_both_shaped.nrrd` | Bilateral shaped flatmap coordinates |
| `flatmap_both_square.nrrd` | Bilateral square flatmap coordinates |
| `depth.nrrd` | Absolute cortical depth in µm |

The `flatmap_authalic_*_256.nrrd` comparison maps are separate products and are not inputs to this preparation sequence. Continue to Step 0F; the converter records the input NRRD SHA-256 hashes in its reports.

### Record

```text
Zenodo DOI: 10.5281/zenodo.11218079
Version/publication date: v4.1 / 2024-05-20
Archive: mouse_isocortex_enhanced.zip
Published archive MD5: 2a0223c1b8fd383a2f589202f1c6376d
Local archive checksum verification: TODO
Download date: TODO
Extracted 10 µm lookup directory: TODO
SHA-256 of each input NRRD: TODO (also recorded by the conversion report)
Download/extraction repeated for this documentation update: Not run
```

---

## Step 0F — Install our atlas-enhancement fork and create the 25 µm NRRDs

**Goal:** Evaluate the published continuous coordinate fields on the 25 µm Allen CCFv3 grid and assemble a lookup directory for Neuron Navigator. This changes the sampling grid; flatmap values remain flatmap coordinates and depth values remain µm.

Use the paths defined in Step 0E.1. The repository name is **`atlas-enhancement`** (singular). Its resampling script runs directly from the checkout after its dependencies are installed.

### 0F.1 — Download the fork at the recorded revision

Clone our fork into a new directory, then select the exact converter revision recorded for this workflow:

```bash
git clone https://github.com/joshlawrimore/atlas-enhancement.git "$FLATMAP_CODE"
git -C "$FLATMAP_CODE" checkout --detach ea735eacdfe8afc40d006d80caefd54d6b30fdd3
git -C "$FLATMAP_CODE" rev-parse HEAD
```

**Expected result:** the last command prints `ea735eacdfe8afc40d006d80caefd54d6b30fdd3`, and the checkout contains `resample_ccf_coordinate_fields.py`. A detached HEAD is expected because the workflow selects a fixed commit. Use a new clone for these commands so an existing development checkout is not changed.

### 0F.2 — Install the converter dependencies with Pixi

The converter needs NumPy, SciPy, pynrrd, and AllenSDK. Neuron Navigator's Pixi environment does not include AllenSDK, so create a separate **Pixi-managed preparation environment**. The scientific-package versions below follow the fork's committed requirements; installing the complete flatmap-generation toolchain is unnecessary for this resampling script.

```bash
pixi init --channel conda-forge "$FLATMAP_PREP_ENV"
pixi add --manifest-path "$FLATMAP_PREP_ENV/pixi.toml" \
  "python=3.10" "numpy=1.23.5" "scipy=1.10.1" "pynrrd=1.1.3"
pixi add --manifest-path "$FLATMAP_PREP_ENV/pixi.toml" --pypi "allensdk==2.16.2"
```

Check the imports and command-line interface before starting the conversion:

```bash
pixi run --manifest-path "$FLATMAP_PREP_ENV/pixi.toml" python -c \
  "import numpy, scipy, nrrd; from allensdk.core.reference_space_cache import ReferenceSpaceCache; print('Converter dependencies available')"
pixi run --manifest-path "$FLATMAP_PREP_ENV/pixi.toml" python \
  "$FLATMAP_CODE/resample_ccf_coordinate_fields.py" --help
pixi list --manifest-path "$FLATMAP_PREP_ENV/pixi.toml"
```

**Expected result:** the import check prints `Converter dependencies available`, and help lists `--target-resolution`, `--min-support`, and `--mirror-depth`. Preserve this environment's `pixi.toml`, `pixi.lock`, and package list with the run record. This installation recipe has not been executed as part of the documentation update; resolve any installation/import failure before proceeding. The import check matters because `--help` alone does not import AllenSDK.

### 0F.3 — Generate shaped and square 25 µm lookup fields

Run the converter twice, using a separate output subdirectory for each style. AllenSDK downloads or reuses the 25 µm Allen annotation and structure hierarchy in `$FLATMAP_OUTPUT/allen_input`; the converter also caches its Isocortex mask there.

```bash
for FLATMAP_STYLE in shaped square; do
  pixi run --manifest-path "$FLATMAP_PREP_ENV/pixi.toml" python \
    "$FLATMAP_CODE/resample_ccf_coordinate_fields.py" \
    --flatmap "$FLATMAP_SOURCE/flatmap_both_${FLATMAP_STYLE}.nrrd" \
    --depth "$FLATMAP_SOURCE/depth.nrrd" \
    --target-resolution 25 \
    --allen-cache "$FLATMAP_OUTPUT/allen_input" \
    --output-dir "$FLATMAP_OUTPUT/$FLATMAP_STYLE" \
    --min-support 0.5 \
    --mirror-depth || break
done
```

**Expected result:** both style runs finish and the following files exist:

```text
resampled-25um/
  allen_input/                         # Allen reference files and Isocortex mask
  shaped/
    flatmap_both_shaped_25.nrrd
    depth_both_25.nrrd
    resampling_report_25.json
  square/
    flatmap_both_square_25.nrrd
    depth_both_25.nrrd
    resampling_report_25.json
```

The command uses **minimum support 0.5, mirrored depth, and no harmonic filling**, matching the settings in the previously inspected strict-output reports. The converter refuses to overwrite existing outputs by default. If either run fails, stop and inspect the error; do not assemble a lookup set from a partially completed pair of runs.

### 0F.4 — Validate and assemble the Neuron Navigator lookup directory

1. Open both `resampling_report_25.json` files. Confirm `target_resolution_um = 25`, `minimum_valid_support = 0.5`, `depth_mirrored_to_left_hemisphere = true`, `fill_holes = null`, and an empty `validation.errors` list. Inspect coverage counts and invalid samples as well as the numerical checks. Completion does not establish full cortical coverage or visual agreement.
2. Confirm that both flatmaps and depth have matching 25 µm spatial transforms and grids. Compare the two depth arrays and their spatial metadata before choosing one. Their compressed-file hashes can differ even when their arrays agree. This recipe uses the shaped run's depth file.
3. Assemble a **new** directory with the three filenames Neuron Navigator expects. Preserve the original generated files and reports:

```bash
mkdir "$FLATMAP_LOOKUPS"
cp -n "$FLATMAP_OUTPUT/shaped/flatmap_both_shaped_25.nrrd" \
  "$FLATMAP_LOOKUPS/flatmap_both_shaped.nrrd"
cp -n "$FLATMAP_OUTPUT/square/flatmap_both_square_25.nrrd" \
  "$FLATMAP_LOOKUPS/flatmap_both_square.nrrd"
cp -n "$FLATMAP_OUTPUT/shaped/depth_both_25.nrrd" \
  "$FLATMAP_LOOKUPS/depth.nrrd"
```

**Expected result:** `$FLATMAP_LOOKUPS` contains `flatmap_both_shaped.nrrd`, `flatmap_both_square.nrrd`, and `depth.nrrd`, all at 25 µm. The copy operations change filenames only. Record the mapping and verify their SHA-256 hashes against the corresponding `outputs.sha256` entries in the two reports. Keep the original Zenodo archive, source NRRDs, Allen reference files, and both reports for provenance. Continue to Step 0G to use this directory with a neuron Parquet.

### Conversion code and inspected provenance

**[REPOSITORY — inspected 2026-10-07]** The utility is located at:

```text
Local checkout: /Users/lawrimorejg/repos/atlas-enhancement
Local script: resample_ccf_coordinate_fields.py
Branch: main
Checkout HEAD and local origin/main: ea735eacdfe8afc40d006d80caefd54d6b30fdd3
Commit title: Add CCF coordinate-field resampling with harmonic hole filling
Commit date: 2026-10-07
Script Git status: tracked; working copy matches committed contents
Committed script SHA-256: 3bd18b3d352fb6ca8495d28c87542d3d589aac07a02683f2e1ebbf1e297c6429
Neuron Navigator checkout at inspection: docs/poster
Neuron Navigator HEAD: 8251e537d38c945afbb320b8d48286e30517e6a1
```

**[RUN RECORD — provenance update 2026-10-07]** The user reported pushing the converter to the fork's `main` branch. Local `HEAD` and `origin/main` both identify the commit above, and the committed script is byte-identical to the working copy and to the SHA-256 recorded before the push. The earlier inspection found it untracked at checkout `dd52516a3b2f9e7ce983898c1c5a8b0018d7f7a1`; the new commit resolves that code-availability gap. Use the [commit-pinned script](https://github.com/joshlawrimore/atlas-enhancement/blob/ea735eacdfe8afc40d006d80caefd54d6b30fdd3/resample_ccf_coordinate_fields.py) for subsequent runs and record their environment. This does not establish which script revision produced the earlier NRRDs.

### Method and parameters

**[REPOSITORY]** The converter uses mask-normalized trilinear interpolation at target NRRD grid positions. Its target mask is Allen Isocortex (structure ID `315`, including descendants) from `annotation/ccf_2017`. It preserves the native target mask separately in each hemisphere. It streams the input NRRDs and does not apply an anti-alias filter or multiply coordinate/depth values by the 10-to-25 µm spacing ratio.

- Source spatial shape: `(1320, 800, 1140)` at 10 µm.
- Target spatial shape: `(528, 320, 456)` at 25 µm.
- Flatmap NRRDs have an additional leading two-component axis: `(2, 528, 320, 456)`; depth is scalar.
- `--min-support 0.5` requires at least half of the interpolation weight to come from valid source samples. This is the value in the existing strict-output reports inspected below.
- `--mirror-depth` extends the source right-hemisphere depth field into the left hemisphere while applying each hemisphere's target mask.
- Omit `--fill-holes` for the strict interpolation outputs. Unsupported samples retain the invalid background value `-1`.

### Optional inferred outputs

**[REPOSITORY]** Adding `--fill-holes harmonic` retains the strict products and writes additional `*_25_harmonic.nrrd` files. It extends coordinates into unsupported Isocortex voxels using a six-neighbor discrete Laplace solve, homologous transfer, and seeding where a disconnected component lacks supported boundary values. These are inferred coordinates and must be identified as such in provenance. The option requires `--mirror-depth`; it does not change `--min-support` automatically. Record the support threshold, fill mode, and exact strict/harmonic file choice for every lookup set.

### Existing artifacts inspected on 2026-10-07

**[RUN RECORD — file/report inspection only]** The local directory `/Users/lawrimorejg/Downloads/mouse_isocortex_enhanced_25_resample` contains the following files. Their SHA-256 hashes match the strict-output reports in `/Users/lawrimorejg/repos/atlas-enhancement/data/allen_ccf_25/output/`:

| Lookup file | Matching report/output | SHA-256 |
|---|---|---|
| `flatmap_both_shaped.nrrd` | `shaped/resampling_report_25.json`, flatmap | `70f36eb48525cb2c69b19cd9566c48c4bda11b2c1bd9023d80e39eb2f2cfbbb4` |
| `flatmap_both_square.nrrd` | `square/resampling_report_25.json`, flatmap | `708ee3c74bf25028edfd17a525dcbcb69e864c8f2af3615b611258c08bf6876b` |
| `depth.nrrd` | `shaped/resampling_report_25.json`, depth | `e83054671af4960967f3eec407e1f2639ab9f8185eedf75142a9fe8029308c85` |

The three headers specify 25 µm axis-aligned spacing, origin `(0, 0, 0)`, and the target shapes above. The matched reports record `minimum_valid_support = 0.5`, mirrored depth, and no harmonic filling. Their `validation.errors` lists are empty. These are observations of existing files and reports, not a new conversion or a napari visual validation.

**Parameter discrepancy to preserve:** the separate checkout's README example uses `--min-support 0.125`, whereas all four reports inspected, including those under a directory named `min125_harmonic`, record **0.5**. Use the JSON value as the record of those runs; directory names and example commands do not establish executed parameters. The commands above explicitly use 0.5 to match the strict reports, but reproducing their bytes also requires the original script/environment.

```text
New download/conversion during this documentation update: Not run
25 µm files selected for the final poster analysis: TODO
Historical conversion date and exact script revision: TODO
Preparation Pixi environment/lock and dependency versions: TODO
Source archive verification against Zenodo: TODO
Strict versus harmonic choice: TODO (inspected Downloads files are strict)
Final lookup directory, three SHA-256 hashes, and lookup-set ID: TODO
Napari visual verification and date: Not run for this update
```

---

## Step 0G — Use the NRRDs to add flatmap coordinates to a neuron Parquet

**Goal:** Prepare an existing CCFv3 neuron Parquet once, so later queries, flatmap displays, and flatmap-based analyses can use stored per-node coordinates.

### What the NRRDs do

The three NRRDs are **spatial lookup fields**. For each node's CCFv3 `(x, y, z)` position, Neuron Navigator uses the NRRD spatial transform to find its lookup voxel, reads shaped/square flatmap coordinates and cortical depth there, and writes those values into the output Parquet. The two resampling runs in Step 0F prepare reusable lookup fields; this step applies them to a particular neuron dataset.

The node rows, original CCFv3 coordinates, `file_id` identities, and custom columns are retained. The version-3 output adds `x_flat_shaped`, `y_flat_shaped`, `x_flat_square`, `y_flat_square`, and shared `depth_um`, together with validity, invalid-reason, and lookup-mode columns. Unsupported/out-of-cortex positions remain in the file with invalid lookup results; they are not silently deleted. Flatmap XY comes from the original voxel. If necessary, only depth is recovered from its mirrored voxel.

**25 µm refers to the lookup's three-dimensional CCF grid spacing.** It does not set the two-dimensional flatmap display bin counts. Leave the input neuron coordinates in their original CCFv3 micron units; do not divide them by 25 before running the micron-coordinate workflow below.

### 0G.1 — Prepare an existing Parquet in napari

1. Launch Neuron Navigator with `pixi run --manifest-path "$NN_REPO/pixi.toml" napari`.
2. Under **Data** → **SWC Parquet Data**, click **Load...** and select the source neuron Parquet. It must contain CCFv3 node coordinates in microns for this workflow.
3. Open **Flatmap**, expand **Flatmap Lookup Files**, and click **Lookup directory...**. Select `$FLATMAP_LOOKUPS` from Step 0F.4. The directory must contain all three canonical filenames; opening the NRRDs as napari image layers does not select them for this operation.
4. Leave **Lookup resolution:** at **From NRRD header** because the generated files contain 25 µm spatial transforms. An explicit value of `25` is only needed for lookup files without usable transforms. Changing this control does not resample a file.
5. Click **Prepare Whole Parquet...** and save to a new filename, such as `neurons_flatmap.parquet`. **Every source row is processed**, regardless of the current table query or selected neurons. Keep the source Parquet as provenance.
6. Wait for completion, record the status/counts, then load the resulting file through **Data** → **SWC Parquet Data** → **Load...**. Choosing a file as the output does not substitute for explicitly loading it for the subsequent analysis.

**Expected result:** a version-3 Parquet containing both bilateral flatmap styles and depth, with metadata recording the lookup-set ID, transforms, canonical bounds, and source hashes. Verify that the row count and distinct `file_id` count match the source, inspect validity/mapping counts, and visually check representative cortical neurons before using the file for poster results. Whole-Parquet preparation is covered by manual test [UC-003](../USE_CASES.md#uc-003-prepare-a-whole-neuron-parquet-for-flatmap-viewing); documenting these steps does not mark that test as passed.

### 0G.2 — Equivalent command-line preparation

Use the **Neuron Navigator** Pixi environment for this step. Set two different Parquet paths, with a new output filename:

```bash
NEURON_SOURCE_PARQUET="/absolute/path/to/neurons.parquet"
NEURON_FLATMAP_PARQUET="/absolute/path/to/neurons_flatmap.parquet"
pixi run --manifest-path "$NN_REPO/pixi.toml" python \
  "$NN_REPO/scripts/add_flatmap_columns_to_parquet.py" \
  "$NEURON_SOURCE_PARQUET" "$NEURON_FLATMAP_PARQUET" \
  --lookup-dir "$FLATMAP_LOOKUPS" \
  --coordinate-mode microns
```

**Expected result:** the command prints the rows written, direct/mirrored-depth/unmapped lookup counts, output path, and lookup-set ID. Save that output with the run record. These are **node-row counts**, not neuron counts; count neurons by distinct `file_id`. The command processes the whole Parquet. `--lookup-resolution 25` is a geometry fallback for missing NRRD transforms, not a downsampling operation, and is unnecessary for the files created in Step 0F.

### When the NRRDs are needed again

| Operation | Uses the NRRD lookup files? |
|---|---|
| Add flatmap/depth columns to an existing Parquet | Yes, during **Prepare Whole Parquet...** or the command above. |
| Convert SWCs with **Add bilateral flatmap/depth columns** enabled | Yes. Choose **Lookup directory...** in the conversion controls before selecting the SWCs; conversion and augmentation then run together. |
| Build a new flatmap region-cache profile | Yes. Use matching lookup files and the required atlas annotation to generate the region arrays. |
| View an already prepared Parquet with **Precomputed Parquet + Cache** and a compatible cache profile | No. The viewer reads stored neuron coordinates and cached region arrays; a matching atlas/version structure catalog is still required for region selection and colors. |
| Choose **Recompute from NRRDs** | Yes. This explicitly performs lookup-based projection again. |

Reuse the prepared Parquet for different MOs queries and cluster analyses; changing the selected neurons does not require repeating the resampling or augmentation. Prepare a new Parquet when the source node coordinates change or a different lookup set is adopted. Preserve the old Parquet and its lookup provenance so analyses remain attributable to the correct version. When changing lookups, build/select a compatible region-cache profile as well. Once preparation and cache building are complete, keep the NRRDs for reproducibility even though routine precomputed viewing does not load them.

### Record

```text
Input Parquet path/SHA-256: TODO
Output Parquet path/SHA-256: TODO
Lookup directory and three NRRD SHA-256 hashes: TODO
Lookup-set ID: TODO
Neuron Navigator commit and Pixi environment: TODO
Preparation date, command or UI actions: TODO
Source/output node-row counts: TODO
Source/output distinct file_id counts: TODO
Shaped/square/depth valid-node counts and unmapped-node counts: TODO
Neurons lacking usable coordinates for the intended analysis: TODO
Region-cache profile, if built: TODO
End-to-end execution / napari visual verification: Not run for this update
```

---

## Step 0H — Create the regional-profile sidecar Parquet for region search

**Goal:** Build the compact per-neuron regional summary used by **Regions** → **Compound Region Query**, then check it with a soma-in-MOs query.

### What the sidecar contains and when it is used

**[REPOSITORY]** A regional profile summarizes each neuron by direct atlas region, soma-relative laterality, and compartment. Each non-empty combination of `file_id`, `region_id`, `laterality`, and `compartment` has one row containing `cable_length_um`, `node_count`, and `terminus_count`. Compound queries use these summaries to combine soma location with regional neurite intersection, measurement thresholds, and ipsilateral/contralateral conditions. Selected parent regions can include their descendants at query time.

The two compartments are **Soma** (`type == 1`) and **Projection (all non-soma neurites)** (`type != 1`, including undefined type `0`). Here, “Projection” includes reconstructed axon and dendrite processes; it is not an axon-only result. Termini are childless non-soma nodes detected from the complete tree within each `file_id` before regional allocation. Cable length is allocated by tracing edges through the atlas voxels. Retain the source morphology-coverage and annotation caveats when interpreting these measurements.

| Prepared artifact | Purpose |
|---|---|
| `<source>.region_profile.parquet` | Regional summary for **Compound Region Query**. |
| Neuron Parquet containing flatmap/depth columns (Step 0G) | Per-node coordinates for flatmap viewing and analysis. |
| Flatmap region cache built with **Build Cache Profile...** | Cached atlas-region geometry for flatmap overlays. |

The regional-profile builder uses the **source neuron Parquet and loaded Allen atlas**. It does not require the flatmap NRRDs or flatmap region cache. A simple **Atlas Regions** query can still run without this sidecar; it is required for **Compound Region Query**.

### Prerequisites and build order

- Finish preparing the exact neuron Parquet that will be used for region searches. If using Step 0G, build the sidecar for its resulting `_flatmap.parquet`, after augmentation. The original Parquet and its augmented copy have different source fingerprints even when they contain the same neurons.
- The source must contain `file_id`, `node_id`, `parent_id`, `type`, `x`, `y`, `z`, and `region_id`, with CCFv3 coordinates in microns and region annotations compatible with the selected atlas. For each `file_id`, the builder validates finite coordinates, unique node IDs, one soma that is also the root, valid parent references, and a connected tree. Use the complete neuron reconstructions as input.
- Load the matching Allen atlas, for example `allen_mouse_25um` if that is the atlas used to annotate the source. Record the actual atlas name, version, and resolution; the flatmap lookup resolution alone does not establish which atlas matches the Parquet.
- The source directory must be writable and have room for the sidecar and temporary build files. The sidecar is saved automatically beside the source; there is no output-file picker in this GUI workflow.

### 0H.1 — Build the sidecar in napari

1. Launch Neuron Navigator with `pixi run napari` from its repository, or use the explicit manifest command from Step 0G.
2. Under **Data** → **SWC Parquet Data**, click **Load...** and select the exact source for the poster analysis, such as `neurons_flatmap.parquet`.
3. Under **Data** → **Atlas**, select the matching atlas and click **Load Atlas**. Wait until it is loaded. If either the Parquet or atlas is missing, the profile controls explain the missing prerequisite and disable building.
4. Open **Regions** and set **Query source:** to **Compound Region Query**. Locate the **Regional Profile** controls. A missing sidecar is reported there. If a file is already found, the message explains that source/atlas compatibility will be validated before a query; file existence alone is not validation.
5. Click **Build Regional Profile**. The build processes **every neuron in the loaded Parquet**, regardless of **Search scope** or the rows currently displayed in **Data**. It fingerprints the source, processes complete neuron graphs, constructs the regional summaries, and validates the result before publishing it. Progress runs in the background; **Cancel Build** becomes available.
6. Wait for **Regional profile ready: … neurons, … sparse rows in … s.** Record all three values. The sparse-row count is the number of regional-summary rows, not the number of neurons or source nodes. Building the sidecar does not itself select a population or replace the neuron Parquet.
7. Confirm that the sidecar exists next to the source, using the naming rule below. Keep the neuron Parquet loaded in **SWC Parquet Data**; the compound-query workflow discovers its sidecar automatically.

**Expected output naming:** the builder replaces the source's final `.parquet` suffix with `.region_profile.parquet`:

```text
neurons_flatmap.parquet
neurons_flatmap.region_profile.parquet
```

For the current development filename, the corresponding output would be `isocortex_total_right_brainglobe_flatmap.region_profile.parquet` in the same directory. This is an expected filename, not a claim that the poster sidecar has been built or verified.

The sidecar metadata records the source fingerprint, atlas identity (including annotation and hierarchy digests), profile/builder versions, measurement definitions, build timestamp, and build-summary counts. Preserve the file alongside the source and record its SHA-256 when freezing the analysis.

### 0H.2 — Check the sidecar with a soma-in-MOs query

1. Keep **Query source:** set to **Compound Region Query**, and set **Search scope:** to **Whole Parquet**.
2. Configure a single condition as **Soma** → **In region**. Remove any other conditions from earlier work so they do not restrict this check.
3. Click **Edit Regions...**, select `MOs`, enable **Include child regions** to include its layer subdivisions, and click **OK**. Check that **Canonical structured query:** expresses only the intended soma-in-MOs condition with descendants.
4. Click **Run Compound Region Query**. The plugin validates the sidecar against the loaded Parquet and atlas before evaluating the query.
5. On success, record the matched-neuron count, canonical query, runtime, and validated profile identity displayed by the plugin. **Whole Parquet** results replace the current Data-table query result. Preserve the returned `file_id` membership for the MOs workflow in Step 2; do not use display `neuron_id` as the identity key.

**Expected result:** the matching MOs soma population appears in **Data**, and the profile status reports **Validated regional profile … for …**. The collaborator's approximately 1,900-neuron expectation remains a planning estimate; record the actual result. If comparing with **Atlas Regions**, use the same source, atlas, **Soma** node selection, descendant inclusion, and scope, then compare `file_id` membership rather than counts alone.

**Search scope** affects the query, not the build: **Current Table** restricts the result to matching neurons already in Data and preserves their table state. An empty Current Table produces a message instead of starting a query. Use **Whole Parquet** for the initial MOs population.

### Cancellation, failures, and reuse

- **Cancel Build** requests cancellation at a safe checkpoint. A cancelled or failed rebuild does not replace an older valid sidecar; completed builds replace the destination only after validation. Record any graph, annotation, or I/O error and correct the source/configuration before rebuilding.
- **Cancel Query** leaves the existing Data table unchanged. Missing, stale, or incompatible sidecars block compound queries before changing Data and direct the user to rebuild.
- Reuse the sidecar when changing query regions, Boolean conditions, thresholds, or search scope. Build/rebuild for the exact source and atlas whenever their compatibility check fails. Flatmap augmentation, source edits, atlas changes, or a changed builder version can require a new profile.
- Keep the source and sidecar together with matching basenames so the GUI can discover the file. A sidecar built for a different source is not made compatible merely by renaming it.

### Record and verification status

```text
Source neuron Parquet path/SHA-256: TODO
Regional sidecar path/SHA-256: TODO
Atlas name/version/resolution: TODO
Atlas annotation/hierarchy digests: TODO (from sidecar metadata)
Neuron Navigator commit and Pixi environment: TODO
Build date/time and elapsed seconds: TODO
Source node rows / distinct file_id count: TODO
Reported build neuron count / sparse profile-row count: TODO
Profile format version / builder version: TODO (from sidecar metadata)
Validated profile identity: TODO
MOs check: Soma → In region; Include child regions enabled; Whole Parquet
Canonical query / matching-neuron count / runtime: TODO
Returned file_id export: TODO
Errors, cancellations, or rebuilds: TODO
Regional sidecar creation/search: Functional (user-confirmed 2026-10-07)
Poster-specific source/atlas/query and run counts: TODO (not supplied with confirmation)
```

The repeatable manual test is [UC-021](../USE_CASES.md#uc-021-build-and-run-a-compound-regional-profile-query). Its **Passed (core workflow)** status records the user's 2026-10-07 confirmation that sidecar creation and regional search are functional. Cancellation and the other documented edge cases were not confirmed in this update. The exact poster input, atlas, query, and result counts still need to be recorded above; the functional confirmation does not supply those run details.

---

# ANALYSIS WORKFLOW

## Step 1 — Load the neuron dataset

**Goal:** Open the exact Parquet dataset to be used for the poster.

### Action

1. Launch the plugin using the repository environment (`pixi run napari`).
2. Load the prepared neuron Parquet from Step 0G in Neuron Navigator if the demonstration uses flatmap/depth coordinates. For a CCFv3-only analysis, the original neuron Parquet can be used without that augmentation.
3. Load/select the matching Allen/BrainGlobe atlas required for region-based queries.
4. Confirm that the neuron table is populated and that `file_id` is available internally as the identity key.
5. If using **Compound Region Query**, use the regional-profile sidecar prepared for this exact Parquet/atlas in Step 0H. The simple **Atlas Regions** route does not require it.

### Record

```text
Date/time: TODO
Input filename: TODO
Total neurons loaded: TODO
Atlas selected: TODO
Screenshot: screenshots/01_dataset_loaded.png
Notes/errors: TODO
```

### Expected result

A neuron table representing the complete input collection is available for region selection and downstream analysis.

---

## Step 2 — Query neurons by soma location in MOs

**Goal:** Find neurons whose **soma** is located in the Allen `MOs` region.

The actions below describe the simple **Atlas Regions** route. For a sidecar-backed query, use the equivalent single-condition **Compound Region Query** in Step 0H.2 and record which route produced the poster population.

### Action

1. Open **Regions**, set **Query source:** to **Atlas Regions**, and choose **Search scope:** → **Whole Parquet**.
2. Select `MOs` (secondary motor area) and enable **Include child regions** to include the MOs layer subdivisions.
3. Set **Node types** to **Soma**, so that inclusion means the neuron's soma lies within MOs, rather than merely having a process that traverses MOs.
4. Click **Find Neurons in Selected Regions** and preserve the matching population in the working table.

### Record

```text
Atlas region acronym: MOs
Region full name: Secondary motor area
Node selection: soma
Query source: Atlas Regions / Compound Region Query
Include child regions: yes
Search scope: Whole Parquet
Regional-profile identity (if compound query): TODO / not used
Number of neurons returned: TODO (expected approximately 1,900)
Query/export filename: TODO
Screenshot: screenshots/02_MOs_soma_query.png
```

### Provenance

- **[COLLABORATOR]** MOs is the requested cortical territory.
- **[REPOSITORY]** Region queries can select neurons based on Allen atlas regions and node-type selections.
- **[LITERATURE]** Hooks et al. relate their vM1 experimental territory to Allen MOs.

### Quality check

Confirm that the resulting soma positions actually lie within the MOs atlas territory in the viewer/flatmap. If the count differs substantially from ~1,900, stop and record whether the difference is due to dataset version, atlas version, hemisphere handling, or query settings.

---

## Step 3 — Visualize all MOs somas

**Goal:** Establish the spatial distribution of the selected population before clustering.

### Action

1. Render or display the MOs population as **somas only**.
2. Capture a whole-region view in the most informative coordinate representation:
   - CCFv3 3D, and/or
   - cortical flatmap, optionally including cortical depth.
3. Preserve a screenshot before cluster recoloring.

### Record

```text
View: TODO
Flatmap style (if used): TODO
Depth shown: yes/no
Neuron count displayed: TODO
Screenshot: screenshots/03_MOs_somas_all.png
```

### Poster candidate

This image can become the first results panel: “MOs neurons selected by soma location.”

---

## Step 4 — Cluster MOs neurons by soma proximity

**Goal:** Establish a soma-based topographic clustering pattern within MOs.

### Action

1. Open the soma-clustering analysis.
2. Choose the clustering space:
   - CCFv3 soma coordinates, **or**
   - cortical flatmap coordinates, optionally incorporating cortical depth.
3. Choose the clustering algorithm and number of spatial clusters.
4. Run clustering and recolor the soma population by cluster.
5. Export the cluster assignments/metrics if available.

### Parameters to record

```text
Input neurons: TODO
Clustering space: CCFv3 / flatmap / flatmap+depth
Flatmap style: TODO
Algorithm: TODO (for example hierarchical/Ward or k-means)
Number of soma clusters (k): TODO
Distance metric: TODO
Depth included: yes/no
Depth scale: TODO
Random seed (if applicable): TODO
Export filename: TODO
Screenshot: screenshots/04_MOs_soma_clusters.png
```

**[REPOSITORY]** Current repository development notes distinguish hierarchical/Ward and seeded k-means behavior and record the current flatmap distance metric as isotropic. Do not omit the algorithm, k, neuron count, or map style when recording a clustering result.

### Quality check

Ask whether clusters correspond to contiguous or interpretable spatial territories rather than obvious rendering artifacts. Save the unclustered and clustered screenshots with the same orientation if possible.

---

## Step 5 — Choose one soma cluster for projection analysis

**Goal:** Hold local soma geography approximately constant and ask whether neurons occupying the same MOs neighborhood have different long-range output patterns.

### Action

1. Choose one spatial soma cluster for the second-stage analysis.
2. Prefer a cluster with enough neurons to support a five-way projection clustering and a compact enough spatial distribution to make the “same neighborhood, different outputs” concept clear.
3. Preserve the selected `file_id` list.

### Record

```text
Selected soma-cluster ID: TODO
Number of neurons: TODO
Reason for selection: TODO
file_id list/export: TODO
Screenshot: screenshots/05_selected_soma_cluster.png
```

**Important:** cluster numbers are arbitrary labels. Record the actual `file_id` membership rather than relying on “cluster 2” remaining stable across reruns.

---

## Step 6 — Define the long-range projection feature space

**Goal:** Make the second-stage clustering reflect projection pattern **outside MOs**, not dense local morphology around the soma.

### Requested analysis definition

**[COLLABORATOR]** Use the selected soma cluster, exclude the MOs territory from the projection comparison, and cluster the remaining projection patterns into five groups.

### Repository-supported similarity concept

**[ABSTRACT + REPOSITORY]** Projection similarity in Neuron Navigator is based on voxel-wise node-count distributions. Pairwise Pearson correlations are computed between per-neuron voxel-count vectors and converted to distances (`1 - r`). Region and node-type restrictions can be applied to analysis/search features.

### Parameters that must be made explicit

```text
Input neurons: TODO
Nodes included: TODO
MOs excluded from feature calculation: yes/no
How MOs exclusion was implemented: TODO
Additional included/excluded regions: TODO
Voxel/bin definition: TODO
Coordinate space: CCFv3 / flatmap
Similarity: Pearson correlation
Distance: 1 - Pearson r
Normalization: TODO / none / implicit in correlation
Zero-vector handling: TODO
Number of projection clusters: 5
Clustering algorithm/linkage: TODO
Random seed (if applicable): TODO
```

### Open implementation question

The scientific requirement is “outside MOs.” Before freezing the poster analysis, verify the exact current UI/code path used to express **region exclusion**, rather than assuming that a region-inclusion control is equivalent. Record the exact UI labels and implementation here after the workflow is run.

### Node-type decision

If clustering uses SWC `type == 2`, document it as **axon-typed-node projection clustering** unless the source annotations have been independently validated. Record the number of selected neurons that lack qualifying nodes.

---

## Step 7 — Cluster the selected local population into five projection groups

**Goal:** Identify five groups of spatially nearby MOs neurons with distinct long-range projection patterns.

### Action

1. Run the projection-similarity calculation using the feature definition from Step 6.
2. Cluster the distance representation into **k = 5** groups.
3. Export cluster memberships and analysis metrics.
4. Recolor the neurons by projection-cluster assignment.

### Record

```text
Number of neurons analyzed: TODO
Number excluded and why: TODO
k: 5
Algorithm/linkage: TODO
Distance metric: 1 - Pearson r
Cluster sizes: TODO
Analysis/export filename: TODO
Screenshot: screenshots/06_projection_clusters.png
```

### Quality checks

- Cluster membership export contains `file_id`.
- No local MOs voxels contributed to the intended long-range comparison.
- A cluster is not merely defined by missing/empty compartment annotations.
- Representative neurons from every cluster are visually inspected.

---

## Step 8 — Visualize the projection destination pattern of each cluster

**Goal:** Turn the abstract five-cluster solution into an anatomical result: where does each group go?

**[ABSTRACT]** Neuron Navigator can render reconstructions as line segments or heatmaps. Flatmap and atlas-region overlays can be used where appropriate.

### Preferred poster outputs

For each of the five clusters, generate one or more of the following using identical display settings across clusters:

1. **Whole-brain heatmap** showing aggregate reconstruction density.
2. **Whole-brain line rendering** for a smaller representative subset if individual trajectories remain legible.
3. **Cortical flatmap heatmap** for cortical projection patterns.
4. **Atlas-region overlay/labels** to help identify target structures.
5. A compact **region-by-cluster quantitative summary** if the plugin/export supports it or if it is generated downstream.

### Record for every cluster

| Projection cluster | n neurons | Heatmap/export | Screenshot | Dominant visible targets | Notes |
|---|---:|---|---|---|---|
| 1 | TODO | TODO | `screenshots/07_cluster1.png` | TODO | TODO |
| 2 | TODO | TODO | `screenshots/08_cluster2.png` | TODO | TODO |
| 3 | TODO | TODO | `screenshots/09_cluster3.png` | TODO | TODO |
| 4 | TODO | TODO | `screenshots/10_cluster4.png` | TODO | TODO |
| 5 | TODO | TODO | `screenshots/11_cluster5.png` | TODO | TODO |

### Display controls to keep fixed

```text
Heatmap voxel/bin size: TODO
Heatmap normalization: TODO
Colormap: TODO
Contrast limits: TODO
Opacity: TODO
Atlas overlays: TODO
Camera orientation / slice: TODO
Flatmap bounds/binning: TODO
```

Fixed rendering parameters are essential if the five cluster panels are to be compared visually.

### Region-overlay orientation verification — 2026-10-07

**Software verification, not a biological result.** The local compatibility
fix preserves atlas/source coordinates and corrects the displayed vertex order
of region meshes, grouped Custom Regions, the brain outline, and cached flatmap
surfaces. Napari remains **0.9.0**; the dependency specifications and Pixi lock
file were not changed. The source state was repository commit
`8251e537d38c945afbb320b8d48286e30517e6a1` plus the uncommitted surface-orientation
fix and its tests. Record the eventual fix commit before producing poster assets.

**Automated checks:** the following command passed **383 tests**, using synthetic
geometry and real napari layer models for the orientation regressions:

```bash
pixi run test tests/test_surface_compat.py tests/test_reference_layers_logging.py tests/test_flatmap_widget.py tests/test_flatmap_rectangular_grid.py tests/test_soma_projection.py -q --tb=short
```

**Scripted rendering check:** ran `pixi run python
/private/tmp/verify_nn_surface_axes.py` outside the sandbox with temporary,
non-persistent napari settings. This local diagnostic script and its PNGs are
temporary verification artifacts, not versioned poster inputs. It checked the
actual Qt/VisPy surface vertices for all six 3D axis orders and a 2D/3D transition
with both synchronous and asynchronous slicing enabled in turn.

The atlas check used cached **`allen_mouse_25um` v1.2**, accessed with
`BrainGlobeAtlas('allen_mouse_25um', check_latest=False)`. It loaded the reference
template and MOs segmentation in 2D, called `viewer.dims.transpose()`, switched
to `ndisplay = 3`, then added the MOs mesh, whole-brain outline, and one Custom
Region group containing `MOs1` and `MOp1`. It checked renderer coordinates and
captured the scene, then called `viewer.dims.roll()` and repeated the check.
Opacities were template **0.25**, segmentation **0.45**, MOs mesh and Custom group
**0.7**, and brain outline **0.08**; colors were atlas defaults and the camera
used napari defaults. No neuron Parquet was loaded and no clustering was run.

**Visual inspection:** inspected
`/private/tmp/nn_surface_axis_verification/mos_transposed.png` and
`mos_rolled.png`; the meshes and brain outline follow the template orientation
and overlay the corresponding anatomical territories. These diagnostic images
were not added to the poster. The complete interactive regression is
[UC-023](../USE_CASES.md#uc-023-keep-region-surfaces-aligned-when-transposing-axes),
whose manual status remains **Not run**. Check it before capturing final region
overlay panels, and record `viewer.dims.order`, `ndisplay`, and camera settings
with each poster screenshot.

---

## Step 9 — Biological interpretation relative to Hooks et al.

**Reference (poster reference 9):** Hooks BM, Papale AE, Paletzki RF, Feroze MW, Eastwood BS, Couey JJ, Winnubst J, Chandrashekar J, Gerfen CR. (2018). *Topographic precision in sensory and motor corticostriatal projections varies across cell type and cortical area.* Nature Communications, **9**, 3549. DOI: https://doi.org/10.1038/s41467-018-05780-7

### Questions to ask after seeing the five clusters

- Does a cluster show bilateral/callosal cortical or corticostriatal projections consistent with an **IT-like** pattern?
- Does a cluster show strong ipsilateral descending projections to thalamic, midbrain, pontine, or medullary targets consistent with a **PT/ET-like** pattern?
- Are there finer subdivisions within either broad pattern?
- Are projection differences mainly target selection, laterality, spatial extent, or total reconstruction density?
- Do nearby somas remain intermingled after they are recolored by projection cluster?

### Interpretation rule

Use terms such as **“IT-like projection pattern”** or **“PT-like projection pattern”** unless the available morphology/metadata directly establishes the cell class. The clustering itself does not assign molecular or transcriptomic identity.

---

# POSTER ASSET PLAN

## Introduction draft — added 2026-10-07

**Asset:** slide 1, **INTRODUCTION** panel of `neuron_navigator_poster_refs_updated.pptx`; editable text box named **MOs workflow introduction**. The three paragraphs below reproduce the poster draft; superscripts match the numbered poster bibliography.

> Mouse secondary motor cortex (Allen region MOs) contains neurons with distinct long-range projection patterns. Layer-5 intratelencephalic (IT) neurons project within the telencephalon, including cortex and striatum, whereas pyramidal tract (PT) neurons also innervate targets beyond it, including brainstem nuclei. Hooks et al. (2018) showed that these classes differ in corticostriatal topography, motivating analysis of projection diversity within a cortical territory.⁹
>
> Neuron Navigator is an open-source napari plugin for interactive exploration of CCFv3-registered neuron reconstructions.⁶ Developed using 18,621 published murine cortical neurons,¹˒² it links anatomical queries, soma-position clustering, and projection-similarity clustering with single-neuron visualization in whole-brain and cortical-flatmap views.
>
> Here we ask whether this workflow can help distinguish IT-like and PT-like projection patterns among MOs neurons. We will select neurons with somata in MOs, cluster them by soma proximity, and re-cluster a spatially compact group by projection similarity outside MOs, measured by Pearson correlations between voxel-wise node counts. Whole-brain inspection will assess whether the resulting groups show anatomy consistent with IT-like or PT-like projections. Cluster membership alone does not establish cell identity.

**Source and interpretation record:** paragraph 1 summarizes the biological background in [Hooks et al. (2018)](https://doi.org/10.1038/s41467-018-05780-7), especially the Introduction and Figures 1 and 7. Paragraph 2 describes the accepted abstract and repository capabilities; 18,621 is the development-collection count, not the MOs or clustered population size. Paragraph 3 states the user-requested analysis question and planned sequence, not a result. The plugin computes correlations from per-neuron node counts and converts them to distances (`1 - r`); Hooks et al.'s injection-based comparison used fluorescence intensities, so this is a related analytical approach, not a replication of their measurements. The paper's vM1/Allen-MOs correspondence motivates the region choice; the eventual atlas query is not proof of an identical experimental territory.

**Status:** draft introduction; the MOs soma/projection-clustering demonstration and IT/PT interpretation have not been verified in this update. Five projection clusters remain the planned demonstration resolution, not five established cell classes. Record raw screenshots, counts, filters, exports, and inspection findings during the user-led walkthrough using Steps 1–9 and the screenshot convention below. Retain the `file_id` identity rule and the morphology/type caveats in Section 4 when interpreting results.

**Poster verification — 2026-10-07:** opened a temporary copy in Microsoft PowerPoint, saved it as an Open XML presentation, and exported a PDF. Inspected the native Introduction and References rendering; the three introduction paragraphs fit inside their panel at 22 pt, and all nine references fit inside theirs. Checked that the introduction text matches this document and the citations match the bibliography. The repository PPTX is the PowerPoint-saved copy; `poster/neuron_poster_powerpoint_verified.pdf` is the updated full-poster proof. This verifies document rendering, not the planned scientific analysis.

## Step 0 provenance flowchart — added 2026-10-07

**Asset:** slide 1, **METHODS** panel of `neuron_navigator_poster_refs_updated.pptx`; editable PowerPoint group named **Provenance flowchart**. Short bubble labels use superscript reference numbers matching references 1–8 below and the poster's numbered REFERENCES list. Workflow step numbers are omitted from the figure; the table below retains their mapping for reproducibility.

| Bubble / preparation stage | Workflow provenance |
|---|---|
| Neuron Navigator⁶ + Pixi⁷ | Steps 0B–0D: obtain the plugin, install the environment manager, and prepare/launch the plugin. |
| PFC SWCs¹ + Cortex SWCs² | Step 0A: the two distinct BSDC reconstruction sources. |
| Neuron Parquet | Plugin SWC conversion, the input prerequisite for Step 0G; see [UC-002](../USE_CASES.md#uc-002-convert-swc-files-to-parquet). |
| Flatmaps³˒⁴ + Atlas fork⁵ → 25 µm NRRDs⁵ | Steps 0E–0F: published lookup fields and the pinned resampling utility; the output cites its conversion code. |
| Neuron Parquet + 25 µm NRRDs → Flatmap Parquet | Step 0G: append bilateral flatmap coordinates and cortical depth. |
| Flatmap Parquet + Allen atlas⁸ → Region sidecar | Step 0H: build the regional-profile Parquet for the source used in this poster workflow. |

This is a **provenance schematic**, not a new run result. The arrows summarize inputs and preparation dependencies. For this poster, the sidecar is built after flatmap augmentation so it matches the final source Parquet; regional-profile construction itself does not require flatmap columns or NRRDs. The Allen target grid also contributes to the 25 µm resampling, as documented in Step 0F; reference 8 identifies the atlas, while reference 5 identifies the resampling code. Detailed parameters, strict/harmonic distinctions, hashes, and verification status remain in Steps 0E–0H and the slide notes.

**Figure legend (poster text):**

**Figure 1. Provenance of the flatmap-enabled neuron Parquet and regional-profile sidecar.**
CCFv3-registered SWC reconstructions from the prefrontal cortex and whole-cortex datasets are consolidated into a neuron Parquet using Neuron Navigator. Published cortical flatmap and depth fields are resampled to a 25 µm atlas grid with the atlas-enhancement fork and used to append bilateral flatmap coordinates and cortical depth. Per-neuron summaries derived from Allen atlas annotations are stored in an associated regional-profile Parquet, linked by file_id, to support compound anatomical searches. Arrows indicate preparation dependencies; superscripts identify source references.

**Legend verification — 2026-10-07:** opened and saved the updated presentation in Microsoft PowerPoint, exported a PDF, and inspected the Methods panel. The full legend fits beneath the flowchart within the panel. Its title and body exactly match the text above, and citations 1–8 still match the bibliography. The native rendering proof was updated to include the legend.

**Production record:** generated as native grouped PowerPoint shapes, arrows, and text with `poster/build_provenance_flowchart.py`. The same layout specification produces `poster/provenance_flowchart_preview.png`, a layout proof rather than a PowerPoint-rendered screenshot. Regenerate from the repository root with:

```bash
pixi run python poster/build_provenance_flowchart.py
```

The script replaces only its named diagram group, maintains the numbered source-reference block, and adds explanatory slide notes. It saves a temporary backup of the presentation before writing. It checks source identifiers against both bibliographies and checks that step numbers do not appear in the diagram. Native diagram labels and geometry can also be edited directly in PowerPoint; update the script if those changes should survive regeneration. Figure review checks include readable labels/superscripts, reference-number agreement, dependency arrows, and placement within the Methods panel. ZIP/XML checks alone do not establish PowerPoint compatibility: open the generated presentation in PowerPoint when changing diagram geometry. No neuron counts, measurements, or completed-run claims were added to the diagram.

**PowerPoint repair and verification — 2026-10-07:** after the user reported that the first generated PPTX would not open correctly, restored the pre-flowchart backup and rebuilt the diagram using standard straight PowerPoint connectors in place of custom-path connectors. Removed the visible step labels and changed the 25 µm output citation to reference 5 (the resampling utility). Opened the rebuilt file in Microsoft PowerPoint, saved it as an Open XML presentation, and exported a PDF. Inspected the actual PDF rendering of the Methods and References panels; all 12 diagram text shapes and 21 connector segments survived the PowerPoint save, and all eight numbered source identifiers matched this bibliography. The repository PPTX is the copy saved by PowerPoint. `poster/provenance_flowchart_powerpoint.png` is a crop of that PDF rendering; the separate `provenance_flowchart_preview.png` remains the generator's layout proof.

## Candidate results sequence

1. **MOs query** — all somas within Allen MOs.
2. **Soma topography** — same population colored by spatial cluster.
3. **Local population selection** — one chosen soma cluster highlighted.
4. **Projection clustering** — local neurons recolored into five projection clusters.
5. **Five projection maps** — matched heatmaps/whole-brain views showing different target patterns.
6. **Hooks et al. context** — a small schematic/reference figure or concise text tying the demonstration to known L5 IT/PT projection organization.

### Screenshot provenance convention

Use filenames that preserve analysis order rather than descriptive names alone:

```text
01_dataset_loaded.png
02_MOs_soma_query.png
03_MOs_somas_all.png
04_MOs_soma_clusters.png
05_selected_soma_cluster.png
06_projection_clusters.png
07_cluster1.png
08_cluster2.png
09_cluster3.png
10_cluster4.png
11_cluster5.png
```

For each screenshot, keep a sidecar note or this document entry with:

- date/time;
- plugin commit;
- input dataset;
- filter/query;
- selected neuron count;
- clustering parameters;
- camera/view settings;
- display normalization/contrast;
- whether the screenshot was cropped or otherwise post-processed.

---

# REPRODUCIBILITY CHECKLIST

Before using any image or number in the final poster, verify that the following are recorded:

- [ ] Repository URL
- [ ] Git commit hash
- [ ] Plugin version
- [ ] Python / napari / environment versions
- [ ] Input Parquet filename and SHA-256
- [ ] Original reconstruction data source and DOI
- [ ] Allen/BrainGlobe atlas version and resolution
- [ ] Hemisphere/coordinate standardization
- [ ] Flatmap/depth lookup provenance, if used
- [ ] Original flatmap paper and version-specific Zenodo data DOI
- [ ] 25 µm conversion script revision/hash, environment, support threshold, mirroring, and fill mode
- [ ] Both resampling reports and renamed lookup-file hashes
- [ ] Regional-profile sidecar path/hash, source/atlas compatibility, builder version, and build counts (if used)
- [ ] Region-query mode, descendant inclusion, search scope, and validated profile identity (if used)
- [ ] MOs query definition and exact neuron count
- [ ] Soma clustering space, algorithm, k, metric, and depth scale
- [ ] Selected soma cluster `file_id` membership
- [ ] Projection node-type definition
- [ ] Exact implementation of “outside MOs”
- [ ] Projection voxel/bin settings
- [ ] Pearson-distance definition
- [ ] Projection clustering algorithm and k=5
- [ ] Cluster membership export with `file_id`
- [ ] Number of neurons excluded because of missing qualifying nodes
- [ ] Heatmap normalization and display settings
- [ ] Screenshot filenames and post-processing history
- [ ] Biological interpretation kept separate from software-generated labels

---

# CODEBASE NOTES RELEVANT TO THIS WORKFLOW

These notes summarize repository behavior that matters specifically for the poster analysis.

1. **Data model:** large SWC collections are aggregated into Parquet for interactive querying and analysis. [ABSTRACT/REPOSITORY]
2. **Neuron identity:** `file_id` is the per-neuron key; `neuron_id` is display metadata and is not unique. [REPOSITORY]
3. **Region querying:** Allen/BrainGlobe regions and node-type selections can be used to retrieve neurons. [REPOSITORY]
4. **Soma clustering:** soma positions can be clustered in CCFv3 or flatmap/depth space. [ABSTRACT/REPOSITORY]
5. **Projection clustering:** voxel-wise node-count vectors are compared by Pearson correlation; the analysis/search distance is `1 - r`. [ABSTRACT/REPOSITORY]
6. **Visualization:** reconstructions can be shown as line segments or heatmaps; flatmap displays use a separate Neuron Navigator Flatmap napari window. [ABSTRACT/REPOSITORY]
7. **Flatmap preprocessing:** the repository supports bilateral shaped and square flatmaps plus cortical depth and can store their provenance in Parquet metadata. [REPOSITORY]
8. **Project persistence:** current project bundles use `.nnproj`; large external flatmap region caches are referenced rather than copied into the project bundle. [REPOSITORY]
9. **Compartment warning:** source SWC `type` labels are not fully trustworthy in the development dataset. [REPOSITORY]
10. **Manual-test philosophy:** the repository's `USE_CASES.md` is intended to serve both as a capability catalog and as repeatable manual tests. This poster workflow is a good candidate for a future use-case entry after it has actually been run end-to-end. [REPOSITORY]
11. **Compound region search:** `analysis/region_profile.py` builds a separate `<source>.region_profile.parquet` summary; **Regions** → **Compound Region Query** validates it against the loaded source and atlas before searching. The summary includes all non-soma neurites, including undefined type-0 nodes. See Step 0H and UC-021. [REPOSITORY]

Confirmed repository implementation references worth tracing when this becomes formal documentation include:

- `src/napari_neuron_navigator/terminals.py` — topology/terminal logic and filtering cautions.
- `src/napari_neuron_navigator/analysis/region_profile.py` and `analysis/region_query.py` — regional sidecar construction, provenance/compatibility checks, and compound-query semantics.
- `flatmap_heatmap.py` — flatmap bin-count behavior referenced by developer notes.
- `analysis/flatmap_correlation.py` — flatmap/projection analysis behavior referenced by developer notes.
- `docs/cpd2_workflow.md` — existing end-to-end workflow cited by the README as an example covering conversion, atlas loading, region queries, soma clustering, and cluster heatmaps.
- `USE_CASES.md` — repository convention for repeatable user workflows/manual verification.

---

# REFERENCES / SOURCE PROVENANCE

References **1–9** use the same numbering as the poster REFERENCES block. The provenance flowchart cites references 1–8; the introduction additionally cites Hooks et al. as reference 9. Reference 10 is the submitted abstract, retained here as workflow context.

1. **PFC SWC dataset**  
   Wang X. (2023). *Single-neuron projectome of mouse prefrontal cortex (with dendrite).* Brain Science Data Center, Chinese Academy of Sciences. CSTR: 33145.11.BSDC.1689837400.1681922768243666945.  
   https://doi.org/10.12412/BSDC.1690164952.20001

2. **Cortex SWC dataset**  
   Gao L. (2025). *Single-neuron projectome of mouse whole cortex.* Brain Science Data Center, Chinese Academy of Sciences. CSTR: 33145.11.BSDC.1747278088.1922847323835588609.  
   https://doi.org/10.12412/BSDC.1747279998.20001

3. **Original flatmap paper**  
   Bolaños-Puchet S, Teska A, Hernando JB, Lu H, Romani A, Schürmann F, Reimann MW. (2024). Enhancement of brain atlases with laminar coordinate systems: Flatmaps and barrel column annotations. *Imaging Neuroscience*, **2**, imag–2–00209.  
   https://doi.org/10.1162/imag_a_00209

4. **Published flatmap/depth lookup data**  
   Bolaños-Puchet S, Teska A, Reimann MW. (2024). *Enhanced atlases and flatmaps of rodent neocortex* (v4.1) [Data set]. Zenodo.  
   https://doi.org/10.5281/zenodo.11218079

5. **Flatmap implementation and local preparation code**  
   Paper-linked implementation: https://github.com/BlueBrain/atlas-enhancement  
   Local fork origin: https://github.com/joshlawrimore/atlas-enhancement  
   Lawrimore J. (2026). CCF coordinate-field resampling utility, `resample_ccf_coordinate_fields.py`. Commit `ea735eacdfe8afc40d006d80caefd54d6b30fdd3`; script hash recorded in Step 0F.  
   [Committed conversion code](https://github.com/joshlawrimore/atlas-enhancement/blob/ea735eacdfe8afc40d006d80caefd54d6b30fdd3/resample_ccf_coordinate_fields.py)

6. **Neuron Navigator repository**  
   https://github.com/nimh-dsst/napari-neuron-navigator

7. **Pixi environment manager**  
   Pixi. *Package and environment management.* https://pixi.sh/

8. **Allen CCFv3 reference atlas**  
   Wang Q et al. (2020). The Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas. *Cell*, **181**(4), 936–953.e20.  
   https://doi.org/10.1016/j.cell.2020.04.007  
   Neuron Navigator obtains its atlas package through BrainGlobe; record the package name, version, and resolution separately in each run.

9. **Hooks et al. 2018 — biological background**  
   Hooks BM, Papale AE, Paletzki RF, Feroze MW, Eastwood BS, Couey JJ, Winnubst J, Chandrashekar J, Gerfen CR. (2018). *Topographic precision in sensory and motor corticostriatal projections varies across cell type and cortical area.* Nature Communications, **9**, 3549.  
   https://doi.org/10.1038/s41467-018-05780-7

10. **Poster abstract**  
    *Neuron Navigator: a napari Plugin for Interactive Cluster Analysis of Large Numbers of Neuron Reconstructions.* Josh Lawrimore, Dustin Moraczewski, Charles Gerfen, Adam Thomas. Submitted abstract supplied for this poster project.

---

# CHANGE LOG

| Date | Change | Source |
|---|---|---|
| 2026-10-06 | Initial workflow created from collaborator's requested MOs → soma cluster → five projection clusters demonstration; added repository-derived reproducibility constraints and provenance fields. | Collaborator discussion + abstract + repository documentation |
| 2026-10-07 | Added the original flatmap paper and Zenodo v4.1 citations to the workflow and poster; expanded the download instructions; documented 25 µm conversion in the separate atlas-enhancement checkout, matched existing lookup hashes to reports, and recorded unresolved script/environment provenance and the support-threshold discrepancy. No download, conversion, or napari validation was rerun. | User-supplied paper/DOIs + Zenodo record + local code, report, and NRRD inspection |
| 2026-10-07 | Pinned the converter provenance and poster software reference to commit `ea735eacdfe8afc40d006d80caefd54d6b30fdd3` after the user reported pushing to origin/main. Verified matching local HEAD/origin/main and unchanged script SHA-256; resolved the untracked-code gap while retaining the unknown historical-run revision/environment. | User-reported push + local Git objects, refs, and script hash |
| 2026-10-07 | Reworked Steps 0E–0F into a sequential download, checksum, extraction, fork-clone, Pixi-install, 25 µm resampling, and lookup-assembly recipe with shared paths. Added Step 0G explaining when the NRRDs are consumed, GUI/CLI Parquet augmentation, stored columns, and cache reuse. Installation and end-to-end execution remain unrun for this documentation update. | User request + pinned converter source + Neuron Navigator implementation + source ZIP member inspection |
| 2026-10-07 | Added Step 0H for building and reusing the regional-profile sidecar, checking a soma-in-MOs compound query, and recording source/atlas identity, counts, cancellation, and verification status. Distinguished the sidecar from flatmap preparation/cache artifacts and made the simple MOs query route explicit. No sidecar build or napari query was run for this update. | User request + regional-profile builder, worker, query editor, and UC-021 |
| 2026-10-07 | Recorded user confirmation that regional sidecar creation and search are functional; updated UC-021 to Passed (core workflow) and synchronized Step 0H. Specific poster run details and edge-case verification remain outstanding. | User confirmation |
| 2026-10-07 | Added an editable Step 0 provenance flowchart to the poster Methods panel, numbered its source citations 1–8, aligned the workflow bibliography, and recorded the generator, layout proof, and schematic scope. | User request + Steps 0A–0H + source/code citations + Allen CCFv3 paper |
| 2026-10-07 | Rebuilt the poster after the reported opening failure, replaced custom-path connectors with standard connectors, removed visible step numbers, and matched every source citation to the workflow. Verified opening, saving, and PDF export in Microsoft PowerPoint and inspected the rendered diagram and reference list. | User report + pre-flowchart backup + native PowerPoint save/export + bibliography checks |
| 2026-10-07 | Added a scientific figure legend beneath the provenance flowchart describing the neuron Parquet's bilateral flatmap/depth columns, the associated regional-profile Parquet, their per-neuron linkage, and the arrows and citation superscripts. Retained the exact text in this workflow and the regeneration script. | User request + Steps 0G–0H |
| 2026-10-07 | Added an editable introduction draft framing the MOs soma/projection-clustering workflow as an attempt to distinguish IT-like and PT-like projection patterns; added Hooks et al. as poster reference 9 and aligned the bibliography and flowchart generator. Recorded the exact draft, source distinctions, and pending user-led screenshot walkthrough. No new clustering results or biological classifications were asserted. | User request + Hooks et al. (2018) + abstract + correlation/clustering implementation |
| 2026-10-07 | Recorded the local surface-axis compatibility fix while retaining napari 0.9.0, 383 passing targeted tests, and scripted Qt/VisPy checks with inspected MOs atlas screenshots. Added UC-023 for the complete interactive regression, still Not run. No poster assets, neuron data, or biological results were changed. | Local tests + cached Allen 25 µm atlas + scripted rendering and visual inspection |
