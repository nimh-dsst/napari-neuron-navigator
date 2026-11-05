# Mask-query validation — 2026-11-05

The supplied Whole Parquet / any-node intersection search meets the requested
five-minute target with direct Arrow scanning. No spatial index was built;
the optional 3 GiB index allowance remains unused.

## Data and result

- Source: `isocortex_total_right_brainglobe_flatmap.parquet`, 19,241,887,289 bytes,
  728,703,227 rows in 2,915 row groups.
- Mask: `Intersection Mask Blurred Set HB+MB Heatmap ∩ Mask Blurred Set TH Heatmap.tif`
  in the user's Downloads folder. It contains 25,819 positive voxels on the
  native `(528, 320, 456)` grid at 25 µm resolution.
- TIFF SHA-256: `c0a203fa32f36dad89f246514e114e9a333c76e44f6dbe8d32c059d2c5013ddd`.
- Scope: Whole Parquet, any node, no source-neuron exclusions. The standalone
  TIFF does not contain source-set provenance, so exclusions cannot be inferred
  from it. The app continues to honor exclusions recorded in its mask layers.
- Result: **9,594 distinct file IDs**. Neuron display IDs are never used as keys.

## Measurements

| Run | Search elapsed | Peak process RSS | Query index/spill files |
|---|---:|---:|---|
| First guarded full scan | 26.6 s | 809 MiB | None |
| Full scan with independent verification | 25.6 s | 792 MiB | None |

Times include TIFF loading, mask preparation, scanning, and the final neuron
list, but exclude Python imports, independent verification, and napari rendering.
These runs used the machine's normal filesystem cache state; cold-cache timing
was not measured. Peak RSS measures the entire benchmark process, including
NumPy/Arrow allocations, rather than DuckDB's internal memory budget. Additional
atlas and scene layers already open in napari are outside that process.

The independent reference used Arrow Dataset predicate pushdown and checked
every candidate point, without the production scanner's early hit-file pruning.
It took another 26.0 s and confirmed exactly the same file IDs. Process output
blocks were zero before writing the small JSON report. System free-space changes
include activity from other applications and are not attributed to this query.

The supervisor stops a benchmark at 300 seconds, 1,536 MiB resident memory, or
less than 4 GiB free disk space. A preliminary SQL reference exceeded the memory
guard and was stopped; it is not used for verification. Both the production
search and the final reference avoid that SQL materialization path.

## Implementation and checks

The reader uses `ParquetFile.iter_batches` with 65,536-row batches, limited
columns, dictionary-encoded identities, and prefetch disabled. It checks points
on the native atlas grid with `floor(coordinate / resolution)`, including the
full extent of boundary voxels. It stores one metadata record per matched file
and skips later points for that file. Conservative row-group pruning is used
when statistics permit; all 728,703,227 rows were scanned for the supplied mask.

Mask union and bounds use slabs and axis reductions, avoiding a coordinate list
for every positive voxel. Search and preparation run in a worker thread with
progress and cancellation checks. UI progress signals are throttled to avoid a
backlog. Cancelled, edited-mask, or stale-source/cohort results cannot replace
the Data table. Worker resources are released when the widget closes.

Automated checks exercise result correctness, missing statistics, dictionaries,
null/nonfinite coordinates, scope/type/exclusion filters, empty masks, batch
boundaries, cancellation, responsive Qt event processing, stale results, and
shutdown. The complete manual napari workflow remains **Not run**; UC-027 in
`USE_CASES.md` is the repeatable checklist.

## Reproduce

From the repository root, using the provided TIFF path:

```sh
pixi run python scripts/benchmark_mask_query.py \
  --parquet isocortex_total_right_brainglobe_flatmap.parquet \
  --mask '/Users/lawrimorejg/Downloads/Intersection Mask Blurred Set HB+MB Heatmap ∩ Mask Blurred Set TH Heatmap.tif' \
  --resolution-um 25 \
  --output benchmark_data/mask-query-verified.json \
  --verify-reference
```

The JSON includes the complete matching file-ID list and resource measurements.
It is local benchmark data and is excluded from version control.
