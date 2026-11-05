"""Measure a real mask query in an isolated process with resource limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--resolution-um", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--max-rss-mib", type=float, default=1536)
    parser.add_argument("--min-free-gib", type=float, default=4)
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        import numpy as np
        import pyarrow.parquet as pq
        import tifffile

        from napari_neuron_navigator.analysis.mask_query import query_neurons_in_masks

        start = time.perf_counter()
        free_before = shutil.disk_usage(args.output.parent).free
        mask = tifffile.imread(args.mask) > 0
        last = 0.0

        def progress(stage, completed, total):
            nonlocal last
            now = time.perf_counter()
            if now - last >= 10 or completed == total:
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "completed": completed,
                            "total": total,
                            "elapsed_seconds": now - start,
                        }
                    ),
                    flush=True,
                )
                last = now

        result = query_neurons_in_masks(
            args.parquet,
            [mask],
            atlas_shape=mask.shape,
            resolution_um=(args.resolution_um,) * 3,
            progress_callback=progress,
        )
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak *= 1024
        elapsed = time.perf_counter() - start
        reference = None
        if args.verify_reference:
            # Independent oracle: Dataset predicate pushdown and a full point
            # lookup, without the production scanner's early hit-file pruning.
            import pyarrow.dataset as ds

            reference_start = time.perf_counter()
            ranges = [
                np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != i)))
                for i in range(3)
            ]
            predicate = None
            for i, name in enumerate(("x", "y", "z")):
                clause = (
                    ds.field(name) >= float(ranges[i][0] * args.resolution_um)
                ) & (ds.field(name) < float((ranges[i][-1] + 1) * args.resolution_um))
                predicate = clause if predicate is None else predicate & clause
            ids = set()
            scanner = ds.dataset(args.parquet, format="parquet").scanner(
                columns=["file_id", "x", "y", "z"],
                filter=predicate,
                batch_size=65536,
                batch_readahead=0,
                fragment_readahead=0,
                use_threads=False,
                fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=False),
            )
            for batch in scanner.to_batches():
                frame = batch.to_pandas()
                voxels = np.floor(
                    frame[["x", "y", "z"]].to_numpy() / args.resolution_um
                ).astype(np.int64)
                inside = mask[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
                ids.update(frame.loc[inside, "file_id"].tolist())
                progress("Verifying independent reference", len(ids), 0)
            if ids != set(result.file_id):
                raise AssertionError(
                    f"Reference differs: {len(ids - set(result.file_id))} missing, {len(set(result.file_id) - ids)} extra"
                )
            reference = {
                "matched_neurons": len(ids),
                "identical_file_ids": True,
                "elapsed_seconds": time.perf_counter() - reference_start,
            }
        summary = {
            "parquet_path": str(args.parquet.resolve()),
            "parquet_bytes": args.parquet.stat().st_size,
            "parquet_rows": pq.ParquetFile(args.parquet).metadata.num_rows,
            "mask_path": str(args.mask.resolve()),
            "mask_sha256": hashlib.sha256(args.mask.read_bytes()).hexdigest(),
            "mask_shape": list(mask.shape),
            "resolution_um": args.resolution_um,
            "positive_mask_voxels": int(np.count_nonzero(mask)),
            "elapsed_seconds": elapsed,
            "peak_rss_mib": peak / 1024**2,
            "disk_free_before": free_before,
            "disk_free_after": shutil.disk_usage(args.output.parent).free,
            "process_output_blocks": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_oublock,
            "matched_neurons": len(result),
            "scan": result.attrs.get("mask_query"),
            "independent_reference": reference,
            "file_ids": result.file_id.tolist(),
        }
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in summary.items() if k != "file_ids"}),
            flush=True,
        )
        return 0

    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    child = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    def output():
        for line in child.stdout:
            print(line, end="", flush=True)

    reader = threading.Thread(target=output, daemon=True)
    reader.start()
    started = time.monotonic()
    process = psutil.Process(child.pid)
    reason = None
    try:
        while child.poll() is None:
            if time.monotonic() - started > args.max_seconds:
                reason = "Exceeded runtime budget"
            elif process.memory_info().rss > args.max_rss_mib * 1024**2:
                reason = "Exceeded process RSS budget"
            elif (
                shutil.disk_usage(args.output.parent).free < args.min_free_gib * 1024**3
            ):
                reason = "Free disk space fell below the guard"
            if reason:
                child.kill()
                break
            time.sleep(0.25)
    except psutil.NoSuchProcess:
        pass
    finally:
        if child.poll() is None:
            child.kill()
        code = child.wait()
        reader.join(timeout=2)
    if reason:
        print(reason, file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
