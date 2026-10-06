# Regional Profile Stage 0 Results

Status: **Complete** on 2026-09-24. This document records the decisions and
measurements that gate Stage 1 of `REGION_SEARCH_IMPLEMENTATION_PLAN.md`.

## Pinned Geometry Semantics

`analysis/region_profile_traversal.py` implements the
`atlas_voxel_traversal_v1` geometry that Stage 1 will use:

- Atlas voxels have half-open physical bounds
  `[0, shape[axis] * resolution[axis])`.
- Internal boundaries belong to the voxel on their positive side. The upper
  outer boundary is outside the annotation.
- The implementation accepts an independent resolution for each of the three
  axes and splits at every crossed voxel plane and at the physical atlas
  midline.
- Region `0` voxels and portions outside the annotation both remain explicitly
  allocated to direct region `0`.
- A segment lying in the midline plane is `midline`; a crossing segment is
  divided into lower- and upper-axis portions. Stage 1 will translate those
  physical sides to laterality relative to the soma.
- A zero-length child-parent edge has no cable portions. Every positive-length
  edge conserves its Euclidean length, using `math.fsum` for validation.
- Node-region assignment continues to use the direct `region_id` already in
  the source Parquet. The half-open geometry above governs cable allocation;
  it does not reinterpret stored node annotations.

The physical midline must be resolved from the loaded atlas hemisphere labels
and voxel geometry, then recorded in profile metadata. For the cached
`allen_mouse_25um` atlas, hemisphere code 2 occupies left-right voxel indices
through 227 and code 1 begins at 228, pinning the boundary to `5700 um`.
`hemisphere.get_atlas_midline()` returns the voxel-center mirror point
`5687.5 um`; that value is useful for mirroring index centers but is not the
physical hemisphere boundary and must not be reused by the profile builder.

The shared fixtures in `tests/region_profile_stage0_fixtures.py` cover a
single voxel, a direct-region boundary, the midline, a partially out-of-volume
segment, mirrored neurons, repeated display and node IDs across `file_id`,
non-contiguous child-before-parent IDs, and an interleaved type tree. The tests
also compare the exact result with a fine, independent midpoint-sampling
oracle. The 50,000-segment benchmark's maximum length-conservation error was
`5.68e-14 um`.

## Storage and Runtime Decision

Use one Zstandard-compressed Parquet sidecar, ordered by:

```text
region_id, laterality, compartment, file_id
```

Use 122,880 rows per Parquet row group initially. On registration, validate
the sidecar and materialize its narrow columns once into a connection-local
DuckDB temporary table. Do not add an index in Stage 1. Keep direct sorted
Parquet scanning as a low-memory fallback, because it also clears the query
budgets by a wide margin.

The benchmark sidecar is explicitly a `node_bucket_proxy_v1`, not a regional
profile. It measures the target sparse keys and query shape without assigning
cable by endpoints or claiming an approximate cable-length method.

## Reference Benchmark

Command:

```bash
pixi run python scripts/benchmark_region_profile_stage0.py \
  --parquet isocortex_total_right_brainglobe_flatmap.parquet \
  --neuron-counts 100,1000,5000 \
  --warmups 3 --iterations 20 \
  --traversal-segments 50000 \
  --output-json /tmp/nnn-region-profile-stage0-5000/report.json
```

Reference machine and software:

- Apple arm64 macOS host; 14 logical CPUs; 36 GiB physical memory. The sandbox
  did not expose the processor's marketing model.
- Python 3.12.14, DuckDB 1.4.3, PyArrow 21.0.0, and NumPy 2.5.2.
- Source: 19,241,887,289 bytes, 728,703,227 node rows, 2,915 row groups, and
  18,621 neurons. The benchmark selected the first distinct `file_id` values
  in physical row-group order.
- "Cold" means the first execution in a fresh DuckDB connection; operating
  system caches were not dropped. "Warm" is the median of 20 executions after
  three explicit warmups.

Sparse proxy cardinality:

| Neurons | Source node rows | Sparse rows | Rows/neuron | Node/sparse ratio |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 4,175,111 | 6,793 | 67.93 | 614.62 |
| 1,000 | 42,786,093 | 65,718 | 65.72 | 651.06 |
| 5,000 | 233,891,141 | 353,162 | 70.63 | 662.28 |

The 5,000-neuron sorted proxy was 1,211,942 bytes. Materializing its 353,162
rows took 14.8 ms. The whole benchmark process peaked at 905,183,232 bytes RSS;
that includes `file_id` discovery, proxy construction, both layouts, and the
traversal benchmark rather than isolating registration alone.

Query timings:

| Runtime layout | One clause, warm median | Two clauses with aggregated direct IDs, warm median |
| --- | ---: | ---: |
| Sorted Parquet scan | 3.08 ms | 6.85 ms |
| DuckDB temporary table | 1.61 ms | 4.49 ms |

The queries matched the same 4,631 and 433 `file_id` values, respectively, in
both layouts. This selects temporary materialization as the normal runtime
layout while retaining sorted Parquet as the durable representation and
fallback.

Exact traversal of 50,000 deterministic anisotropic synthetic segments ran at
9,045 segments/s in one Python process, with 30.69 voxel portions per segment.
Naively extrapolating that deliberately long-segment workload to the
728,684,606 non-root edges in the canonical source is about 22.4 hours on one
core. This is not a measured profile build time: real SWC edges normally cross
far fewer voxels, and Stage 1 will use complete-neuron batches and parallel
workers. It does establish that a serial Python loop is not an acceptable
canonical builder.

## Confirmed Budgets for Stage 1

- Warm one-clause query: at most 200 ms on the reference machine.
- Warm two-clause `AND` query with parent aggregation: at most 500 ms.
- Profile registration: at most 2 seconds and 512 MiB incremental RSS for the
  canonical sidecar; it may read/materialize the sidecar but must not scan the
  source node relation.
- Construction: at most 4 GiB incremental RSS. Batches contain complete
  `file_id` trees and target at most 400 neurons or 5 million source rows;
  a single oversized neuron remains intact in its own batch.
- Exact traversal: Stage 1 targets at least 35,000 edges/s aggregate, equivalent
  to no more than about six hours of traversal for the canonical source before
  final compaction and validation. If a measured Stage-1 representative build
  cannot support that target, revise the build-time budget before proceeding;
  do not replace exact traversal with sampling or endpoint attribution.

## Source Fingerprint and Invalidation Policy

The profile records a `sha256_v1` streaming digest of the complete source
Parquet when it is built. It also records a fast validation token containing:

- file size, nanosecond modification time, and nanosecond change time where
  available;
- device and inode/file identifier where available;
- Parquet row count, row-group count, Arrow schema digest, and footer digest.

The resolved path is diagnostic, not part of content identity, so moving an
unchanged source and sidecar together does not invalidate them. During a
loaded-atlas session, a full SHA-256 result is cached under the fast token. On
later opens, an unchanged fast token and footer digest validate without
scanning node data. Any fast-token or footer change triggers a full streaming
SHA-256 comparison: equal content is accepted under its new locator metadata;
different content rejects the sidecar. The threat model is ordinary filesystem
replacement, not an adversary able to preserve inode, timestamps, size, and
the complete Parquet footer while changing only data pages.

Atlas compatibility is independent of the source token. Stage 1 will reject a
different atlas name, version, axis resolution, annotation shape or digest,
left-right axis/midline, hierarchy digest, profile version, builder version,
or cable-length method.

## Atomic Publication Policy

Stage 1 writes a uniquely named temporary file in the destination sidecar's
directory. It closes and flushes the writer, fsyncs the temporary file where
supported, reopens and fully validates schema, metadata, counts, and
conservation, and only then publishes with `os.replace`. It fsyncs the parent
directory where supported after replacement. Cancellation or failure before
replacement removes the temporary file and preserves any prior valid sidecar.
No manifest, success marker, or final filename is created early, so a partial
file cannot look complete.

## Acceptance Evidence

- `pixi run pytest -q tests/test_region_profile_stage0.py` — 15 passed.
- `pixi run pytest -q tests/test_region_profile_stage0.py tests/test_benchmark_region_profile_stage0_script.py`
  — 18 passed.
- `pixi run test` — 1,225 passed and 4 skipped.
- The benchmark above supplies the measured storage/runtime choice and records
  hardware, input identity, command, cold/warm definitions, and software
  versions in its JSON output.
- No production UI behavior changed in Stage 0.
