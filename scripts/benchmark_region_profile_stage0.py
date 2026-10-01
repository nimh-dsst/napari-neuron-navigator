#!/usr/bin/env python
"""Benchmark Stage-0 regional-profile traversal and candidate query layouts.

The generated Parquet is a *node-bucket proxy*, not a regional profile: it
uses node counts to reproduce the target sparse keys without pretending that
endpoint attribution is cable length.  Its only purpose is to measure row
cardinality and query-layout overhead before the sidecar builder exists.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from napari_neuron_navigator.analysis.region_profile_traversal import (
    sum_portion_lengths,
    traverse_atlas_voxels,
)

REQUIRED_COLUMNS = {
    "file_id",
    "node_id",
    "parent_id",
    "type",
    "x",
    "y",
    "z",
    "region_id",
}
AXIS_COLUMNS = ("x", "y", "z")


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _parse_neuron_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "neuron counts must be comma-separated integers"
        ) from error
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("neuron counts must be positive")
    return counts


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _elapsed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def _memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    return page_size * pages


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the BSD-derived Python documentation use
    # KiB.  Windows has no ``resource`` module and returns ``None`` above.
    return peak if sys.platform == "darwin" else peak * 1024


def _source_details(path: Path, parquet_file: pq.ParquetFile) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "row_count": parquet_file.metadata.num_rows,
        "row_group_count": parquet_file.metadata.num_row_groups,
        "columns": parquet_file.schema_arrow.names,
    }


def _first_file_ids(parquet_file: pq.ParquetFile, limit: int) -> list[str]:
    """Read leading file-id row groups until ``limit`` complete IDs are known."""
    selected: list[str] = []
    seen: set[str] = set()
    for row_group in range(parquet_file.metadata.num_row_groups):
        values = parquet_file.read_row_group(
            row_group, columns=["file_id"], use_threads=True
        ).column("file_id")
        for value in values.to_pylist():
            file_id = str(value)
            if file_id not in seen:
                seen.add(file_id)
                selected.append(file_id)
                if len(selected) == limit:
                    return selected
    return selected


def _benchmark_traversal(segment_count: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    annotation = rng.integers(0, 24, size=(32, 24, 20), dtype=np.int32)
    resolution = np.asarray((10.0, 20.0, 30.0))
    bounds = np.asarray(annotation.shape) * resolution
    starts = rng.uniform(-0.1 * bounds, 1.1 * bounds, size=(segment_count, 3))
    stops = rng.uniform(-0.1 * bounds, 1.1 * bounds, size=(segment_count, 3))
    midline = float(bounds[2] / 2.0)

    portion_count = 0
    max_conservation_error_um = 0.0
    started = time.perf_counter()
    for start, stop in zip(starts, stops, strict=True):
        portions = traverse_atlas_voxels(
            start,
            stop,
            annotation,
            resolution,
            left_right_axis=2,
            midline_um=midline,
        )
        expected_length = float(np.linalg.norm(stop - start))
        actual_length = sum_portion_lengths(portions)
        max_conservation_error_um = max(
            max_conservation_error_um, abs(expected_length - actual_length)
        )
        portion_count += len(portions)
    elapsed_s = time.perf_counter() - started
    return {
        "algorithm": "atlas_voxel_traversal_v1_prototype",
        "segment_count": segment_count,
        "portion_count": portion_count,
        "elapsed_s": elapsed_s,
        "segments_per_s": segment_count / elapsed_s,
        "portions_per_segment": portion_count / segment_count,
        "max_conservation_error_um": max_conservation_error_um,
        "annotation_shape": list(annotation.shape),
        "resolution_um": resolution.tolist(),
        "left_right_axis": 2,
        "midline_um": midline,
        "seed": seed,
    }


def _create_profile_proxy(
    connection: duckdb.DuckDBPyConnection,
    source_path: Path,
    selected_ids: Sequence[str],
    *,
    left_right_axis: int,
    midline_um: float,
) -> float:
    selected_table = pa.table(
        {
            "file_id": list(selected_ids),
            "sample_rank": np.arange(1, len(selected_ids) + 1, dtype=np.int32),
        }
    )
    connection.register("selected_ids", selected_table)
    source = f"read_parquet({_sql_string(source_path)})"
    lr_column = AXIS_COLUMNS[left_right_axis]

    def build() -> None:
        connection.execute(
            f"""
            CREATE TEMP TABLE selected_somas AS
            SELECT
                n.file_id,
                count(*) FILTER (WHERE n.type = 1)::BIGINT AS soma_count,
                avg(n.{lr_column}) FILTER (WHERE n.type = 1) AS soma_lr_um
            FROM {source} AS n
            INNER JOIN selected_ids AS selected USING (file_id)
            GROUP BY n.file_id
            """
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE profile_proxy AS
            SELECT
                n.file_id,
                coalesce(n.region_id, 0)::INTEGER AS region_id,
                CASE
                    WHEN soma.soma_count <> 1 OR soma.soma_lr_um IS NULL
                        THEN 'unknown'
                    WHEN soma.soma_lr_um = ? THEN 'unknown'
                    WHEN n.{lr_column} = ? THEN 'midline'
                    WHEN (n.{lr_column} < ?) = (soma.soma_lr_um < ?)
                        THEN 'ipsilateral'
                    ELSE 'contralateral'
                END AS laterality,
                CASE WHEN n.type = 1 THEN 'soma' ELSE 'neurite' END AS compartment,
                0.0::DOUBLE AS cable_length_um,
                count(*)::BIGINT AS node_count,
                0::BIGINT AS terminus_count
            FROM {source} AS n
            INNER JOIN selected_ids AS selected USING (file_id)
            INNER JOIN selected_somas AS soma USING (file_id)
            GROUP BY
                n.file_id,
                coalesce(n.region_id, 0),
                laterality,
                compartment
            """,
            [midline_um, midline_um, midline_um, midline_um],
        )

    _, elapsed_s = _elapsed(build)
    return elapsed_s


def _write_sorted_candidate(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    row_group_size: int,
) -> float:
    def write() -> None:
        connection.execute(
            f"""
            COPY (
                SELECT * FROM profile_proxy
                ORDER BY region_id, laterality, compartment, file_id
            ) TO {_sql_string(path)} (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE {int(row_group_size)}
            )
            """
        )

    _, elapsed_s = _elapsed(write)
    return elapsed_s


def _cardinalities(
    connection: duckdb.DuckDBPyConnection,
    neuron_counts: Sequence[int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for neuron_count in neuron_counts:
        profile_rows, source_rows = connection.execute(
            """
            SELECT
                count(*)::BIGINT,
                sum(profile.node_count)::BIGINT
            FROM profile_proxy AS profile
            INNER JOIN selected_ids AS selected USING (file_id)
            WHERE selected.sample_rank <= ?
            """,
            [int(neuron_count)],
        ).fetchone()
        results.append(
            {
                "neuron_count": int(neuron_count),
                "source_node_rows": int(source_rows),
                "profile_proxy_rows": int(profile_rows),
                "proxy_rows_per_neuron": profile_rows / neuron_count,
                "node_to_proxy_compression_ratio": source_rows / profile_rows,
            }
        )
    return results


def _query_inputs(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    neurite_regions = [
        int(row[0])
        for row in connection.execute(
            """
            SELECT region_id
            FROM profile_proxy
            WHERE compartment = 'neurite' AND region_id <> 0
            GROUP BY region_id
            ORDER BY count(DISTINCT file_id) DESC, region_id
            LIMIT 8
            """
        ).fetchall()
    ]
    soma_row = connection.execute(
        """
        SELECT region_id
        FROM profile_proxy
        WHERE compartment = 'soma' AND region_id <> 0
        GROUP BY region_id
        ORDER BY count(DISTINCT file_id) DESC, region_id
        LIMIT 1
        """
    ).fetchone()
    if not neurite_regions:
        raise ValueError("The selected source rows contain no mapped neurite regions")
    soma_region = int(soma_row[0]) if soma_row is not None else neurite_regions[0]
    return {
        "one_clause_region_id": neurite_regions[0],
        "soma_region_id": soma_region,
        "represented_direct_region_ids": neurite_regions,
    }


def _queries(relation: str, inputs: dict[str, Any]) -> tuple[tuple[str, list[Any]], ...]:
    represented_ids = list(inputs["represented_direct_region_ids"])
    placeholders = ", ".join("?" for _ in represented_ids)
    one_clause = (
        f"""
        SELECT DISTINCT file_id
        FROM {relation}
        WHERE region_id = ? AND compartment = 'neurite'
        """,
        [inputs["one_clause_region_id"]],
    )
    two_clause = (
        f"""
        SELECT file_id
        FROM {relation}
        WHERE region_id = ? AND compartment = 'soma'
        GROUP BY file_id
        INTERSECT
        SELECT file_id
        FROM {relation}
        WHERE region_id IN ({placeholders}) AND compartment = 'neurite'
        GROUP BY file_id
        HAVING sum(node_count) > 0
        """,
        [inputs["soma_region_id"], *represented_ids],
    )
    return one_clause, two_clause


def _time_queries(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    inputs: dict[str, Any],
    *,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    names = ("one_clause", "two_clause_parent_aggregation")
    results: dict[str, Any] = {}
    for name, (sql, parameters) in zip(names, _queries(relation, inputs), strict=True):
        cold_started = time.perf_counter()
        rows = connection.execute(sql, parameters).fetchall()
        cold_s = time.perf_counter() - cold_started
        for _ in range(warmups):
            connection.execute(sql, parameters).fetchall()
        warm_times: list[float] = []
        for _ in range(iterations):
            warm_started = time.perf_counter()
            connection.execute(sql, parameters).fetchall()
            warm_times.append(time.perf_counter() - warm_started)
        results[name] = {
            "matched_file_ids": len(rows),
            "cold_first_execution_s": cold_s,
            "warm_median_s": statistics.median(warm_times),
            "warm_min_s": min(warm_times),
            "warm_max_s": max(warm_times),
            "warm_iterations": iterations,
            "warmup_iterations": warmups,
        }
    return results


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.parquet).resolve()
    parquet_file = pq.ParquetFile(source_path)
    missing = sorted(REQUIRED_COLUMNS - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"Source Parquet is missing required columns: {', '.join(missing)}")

    requested_counts = tuple(args.neuron_counts)
    selected_ids, selection_elapsed_s = _elapsed(
        lambda: _first_file_ids(parquet_file, max(requested_counts))
    )
    if len(selected_ids) < max(requested_counts):
        raise ValueError(
            f"Requested {max(requested_counts)} neurons, but the source contains "
            f"only {len(selected_ids)} distinct file_id values"
        )

    workspace_context = (
        tempfile.TemporaryDirectory(prefix="region-profile-stage0-")
        if args.work_dir is None
        else None
    )
    work_dir = (
        Path(workspace_context.name)
        if workspace_context is not None
        else Path(args.work_dir).resolve()
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = work_dir / "sorted-profile-proxy.parquet"

    try:
        builder = duckdb.connect()
        try:
            build_elapsed_s = _create_profile_proxy(
                builder,
                source_path,
                selected_ids,
                left_right_axis=args.left_right_axis,
                midline_um=args.midline_um,
            )
            cardinalities = _cardinalities(builder, requested_counts)
            query_inputs = _query_inputs(builder)
            write_elapsed_s = _write_sorted_candidate(
                builder,
                candidate_path,
                row_group_size=args.row_group_size,
            )
        finally:
            builder.close()

        query_connection = duckdb.connect()
        try:
            query_connection.execute(
                f"CREATE VIEW profile_parquet AS SELECT * FROM "
                f"read_parquet({_sql_string(candidate_path)})"
            )
            parquet_queries = _time_queries(
                query_connection,
                "profile_parquet",
                query_inputs,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            _, materialization_elapsed_s = _elapsed(
                lambda: query_connection.execute(
                    "CREATE TEMP TABLE profile_memory AS SELECT * FROM profile_parquet"
                )
            )
            memory_queries = _time_queries(
                query_connection,
                "profile_memory",
                query_inputs,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            layout_peak_rss_bytes = _peak_rss_bytes()
        finally:
            query_connection.close()

        traversal = _benchmark_traversal(args.traversal_segments, args.seed)
        candidate_details = {
            "format": "node_bucket_proxy_v1_not_a_regional_profile",
            "sort_order": [
                "region_id",
                "laterality",
                "compartment",
                "file_id",
            ],
            "compression": "zstd",
            "row_group_size": args.row_group_size,
            "size_bytes": candidate_path.stat().st_size,
            "build_proxy_elapsed_s": build_elapsed_s,
            "write_sorted_parquet_elapsed_s": write_elapsed_s,
            "materialize_temp_table_elapsed_s": materialization_elapsed_s,
            "process_peak_rss_bytes_after_layout_benchmark": layout_peak_rss_bytes,
        }
    finally:
        if workspace_context is not None:
            workspace_context.cleanup()

    return {
        "benchmark": "regional_profile_stage0_v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": _memory_bytes(),
            "process_peak_rss_bytes": _peak_rss_bytes(),
        },
        "software": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "source": _source_details(source_path, parquet_file),
        "selection": {
            "method": "first_distinct_file_ids_in_physical_row_group_order",
            "requested_neuron_counts": list(requested_counts),
            "maximum_selected_neurons": len(selected_ids),
            "elapsed_s": selection_elapsed_s,
        },
        "candidate": candidate_details,
        "cardinality": cardinalities,
        "query_inputs": query_inputs,
        "query_timing_definition": {
            "cold": "first execution in one fresh DuckDB connection; OS cache uncontrolled",
            "warm": "median after explicit warmup executions in the same connection",
        },
        "layouts": {
            "sorted_parquet_scan": parquet_queries,
            "duckdb_temporary_table": memory_queries,
        },
        "traversal": traversal,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact voxel traversal and regional-profile query layouts."
        )
    )
    parser.add_argument("--parquet", required=True, help="Annotated node Parquet")
    parser.add_argument(
        "--neuron-counts",
        type=_parse_neuron_counts,
        default=(25, 100, 250),
        help="Comma-separated cumulative subset sizes (default: 25,100,250)",
    )
    parser.add_argument(
        "--left-right-axis",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Source coordinate axis used for physical laterality (default: 2)",
    )
    parser.add_argument(
        "--midline-um",
        type=float,
        default=5700.0,
        help="Physical atlas midline in microns (default: 5700)",
    )
    parser.add_argument("--warmups", type=_positive_int, default=3)
    parser.add_argument("--iterations", type=_positive_int, default=10)
    parser.add_argument("--traversal-segments", type=_positive_int, default=20_000)
    parser.add_argument("--row-group-size", type=_positive_int, default=122_880)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument(
        "--work-dir",
        help="Keep the candidate proxy in this directory instead of a temporary one",
    )
    parser.add_argument("--output-json", type=Path, help="Write the report as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args)
    except Exception as error:  # noqa: BLE001 - command reports actionable failures
        parser.exit(1, f"error: {error}\n")
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
