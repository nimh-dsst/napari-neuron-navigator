# Regional Profile Stage 1 Results

Status: **Complete** on 2026-10-01. This document records the implementation
and acceptance evidence for Stage 1 of
`REGION_SEARCH_IMPLEMENTATION_PLAN.md`.

## Implemented Core

`analysis/region_profile.py` now provides the headless regional-profile core:

- immutable atlas, source-fingerprint, build-summary, profile-metadata, and
  inspection models;
- a hierarchy closure derived from `structure_id_path`, including every
  region's depth-zero self link;
- physical-midline resolution from the loaded atlas's hemisphere-label volume;
- complete-`file_id` batching capped by both neuron and source-row counts;
- a cancellable, single-pass Arrow staging step for multi-batch builds, which
  avoids rescanning the source Parquet for every batch and retains only the
  eight required columns in temporary partitions;
- graph validation for duplicate nodes, roots, somas, dangling parents,
  disconnected cycles, coordinates, and atlas-region membership;
- exact child-parent cable allocation through `atlas_voxel_traversal_v1`;
- complete-tree non-soma terminus detection before region or side allocation;
- direct-region aggregation with explicit soma/neurite and
  ipsilateral/contralateral/midline/unknown buckets;
- per-neuron and global length, node, and terminus conservation checks;
- Zstandard sidecar compaction in
  `region_id, laterality, compartment, file_id` order;
- full source SHA-256 identity plus fast stat/schema/footer validation;
- atlas annotation and hierarchy digests;
- cancellation-safe temporary output, full pre-publication validation,
  `fsync`, and atomic replacement;
- metadata-only inspection and validated DuckDB temporary-table registration.

`AtlasVoxelTraverser` in `analysis/region_profile_traversal.py` prepares and
validates atlas geometry once per build. It preserves the Stage 0 traversal
semantics while avoiding repeated annotation and resolution validation for
every edge.

The SWC conversion API and `scripts/convert_swc_to_parquet.py` expose optional
regional-profile generation. It is off by default. The CLI's explicit
`--build-regional-profile` option enables region annotation and writes the
sidecar only after the source Parquet conversion succeeds.

No Regions-tab or other Qt control depends on the sidecar yet.

## Real-Data Build Measurement

The standalone builder was measured with the cached `allen_mouse_25um` atlas
and local `cpd2_left.parquet` source:

- 77 complete neurons;
- 607,381 source nodes;
- 607,304 non-root edges;
- 6.484 seconds end to end for `build_region_profile`;
- 93,659 edges/s, including source fingerprinting, source reads, exact
  traversal, aggregation, compaction, validation, and atomic publication;
- 439 sparse profile rows;
- 13,298-byte Zstandard sidecar;
- 1,001,177,088-byte process peak RSS, including the loaded atlas, Python,
  DuckDB, source batching, and profile construction.

This clears the Stage 0 construction target of at least 35,000 edges/s and the
4 GiB process-memory budget. The real edges are substantially shorter than the
deliberately long Stage 0 synthetic traversal workload, so the measured serial
builder already exceeds the aggregate target; multiprocessing was not added.

A forced four-batch run over the same complete cohort exercised the single-pass
staging path. It completed in 6.875 seconds at 88,336 edges/s with a
1,068,138,496-byte process peak RSS and produced the same 439 rows and
13,298-byte sidecar. This verifies that complete-neuron batching does not rely
on repeated full-source queries.

The same sidecar was compatibility-checked and registered into a fresh DuckDB
connection in 0.145 seconds. The process high-water mark increased by at most
14,532,608 bytes during registration, well below the 2-second and 512 MiB
budgets. Registration read the compact sidecar only; source validation used
its unchanged stat and Parquet-footer token and did not scan node data.

The real build conserved 607,304 edge lengths and all 607,381 node assignments.
The sidecar passed schema, metadata, uniqueness, measurement-total, source,
atlas, algorithm-version, and cable-method validation before publication.

## Correctness and Failure Behavior

The Stage 0 mirror and topology fixtures now run through the complete builder.
They demonstrate that repeated `neuron_id` and `node_id` values remain separate
by `file_id`, child-before-parent ordering is accepted, undefined type `0`
contributes to neurite rows, and a typed parent with a child is not reported as
a terminus.

Midline somas remain `midline`; out-of-volume somas remain `unknown`; direct
region `0` rows remain stored. Source relocation and timestamp-only changes
fall back to content SHA-256 and remain compatible, while source-content,
annotation, hierarchy, atlas, format, builder, and length-method differences
reject the sidecar.

Cancellation and injected compaction failures remove only uniquely named
temporary files. An existing destination remains byte-for-byte unchanged
unless a fully validated replacement reaches the final `os.replace` call.

## Automated Evidence

- `pixi run pytest -q tests/test_region_profile_stage0.py
  tests/test_region_profile_stage1.py tests/test_batch_swc_to_parquet.py` — 57
  passed.
- `pixi run ruff check` on the new Stage 1 core and tests — passed.
- `pixi run test` — 1,248 passed and 4 skipped.

The full suite includes the Stage 0 and Stage 1 tests and all existing plugin,
conversion, analysis, and UI tests.
