from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/benchmark_region_profile_stage0.py"


def _write_source(path: Path, *, neuron_count: int = 8) -> None:
    rows: list[dict[str, object]] = []
    for neuron_index in range(neuron_count):
        file_id = f"cell-{neuron_index:03d}.swc"
        soma_z = 25.0 if neuron_index % 2 == 0 else 75.0
        for node_id, node_type, parent_id, region_id, offset in (
            (10, 1, -1, 100 + neuron_index % 2, 0.0),
            (30, 0, 10, 200 + neuron_index % 3, 10.0),
            (20, 2, 30, 201 + neuron_index % 3, 20.0),
        ):
            rows.append(
                {
                    "file_id": file_id,
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "type": node_type,
                    "x": float(neuron_index + offset),
                    "y": 2.0,
                    "z": soma_z,
                    "region_id": region_id,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False, row_group_size=6)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_stage0_benchmark_help() -> None:
    result = _run(["--help"])

    assert result.returncode == 0
    assert "exact voxel traversal" in result.stdout
    assert "--neuron-counts" in result.stdout


def test_stage0_benchmark_reports_cardinality_layouts_and_traversal(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "report.json"
    work_dir = tmp_path / "candidates"
    _write_source(source_path)

    result = _run(
        [
            "--parquet",
            str(source_path),
            "--neuron-counts",
            "4,8",
            "--midline-um",
            "50",
            "--warmups",
            "1",
            "--iterations",
            "2",
            "--traversal-segments",
            "100",
            "--row-group-size",
            "2048",
            "--work-dir",
            str(work_dir),
            "--output-json",
            str(output_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text())
    assert report["benchmark"] == "regional_profile_stage0_v1"
    assert report["source"]["row_count"] == 24
    assert [item["neuron_count"] for item in report["cardinality"]] == [4, 8]
    assert report["cardinality"][-1]["source_node_rows"] == 24
    assert report["candidate"]["format"] == (
        "node_bucket_proxy_v1_not_a_regional_profile"
    )
    assert report["candidate"]["sort_order"] == [
        "region_id",
        "laterality",
        "compartment",
        "file_id",
    ]
    assert set(report["layouts"]) == {
        "sorted_parquet_scan",
        "duckdb_temporary_table",
    }
    for layout in report["layouts"].values():
        assert layout["one_clause"]["warm_iterations"] == 2
        assert "warm_median_s" in layout["two_clause_parent_aggregation"]
    assert report["traversal"]["segment_count"] == 100
    assert report["traversal"]["max_conservation_error_um"] < 1e-10
    assert (work_dir / "sorted-profile-proxy.parquet").is_file()


def test_stage0_benchmark_rejects_missing_columns(tmp_path: Path) -> None:
    source_path = tmp_path / "invalid.parquet"
    pd.DataFrame({"file_id": ["cell.swc"]}).to_parquet(source_path, index=False)

    result = _run(
        [
            "--parquet",
            str(source_path),
            "--neuron-counts",
            "1",
            "--traversal-segments",
            "1",
        ]
    )

    assert result.returncode == 1
    assert "missing required columns" in result.stderr
    assert "node_id" in result.stderr
