"""Tests for similar-neuron voxel-correlation search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.analysis.correlation import (
    compute_pearson_correlation_matrix,
    correlation_long_to_matrix,
)
from napari_neuron_navigator.analysis.search import (
    VoxelSearchRequest,
    compute_voxel_search,
    export_search_results_csv,
    load_search_results_csv,
)
from napari_neuron_navigator.analysis.voxel_filter import VoxelNodeFilter


def _write_search_parquet(path) -> None:
    rows: list[dict[str, object]] = []

    def add(file_id: str, neuron_id: str, subject: str, counts: list[int]) -> None:
        node_id = 10
        for voxel, count in enumerate(counts):
            for _ in range(count):
                rows.append(
                    {
                        "file_id": file_id,
                        "neuron_id": neuron_id,
                        "subject": subject,
                        "node_id": node_id,
                        "parent_id": -1 if node_id == 10 else 10,
                        "type": 2,
                        "x": float(voxel),
                        "y": 0.0,
                        "z": 0.0,
                    }
                )
                node_id += 7

    add("ref", "shared", "subject-a", [2, 1, 0, 0])
    add("similar", "similar", "subject-a", [4, 2, 0, 0])
    add("partial", "partial", "subject-b", [1, 0, 1, 0])
    add("disjoint", "shared", "subject-c", [0, 0, 0, 3])
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_single_reference_search_matches_analysis_distance_row(tmp_path) -> None:
    import duckdb

    path = tmp_path / "search.parquet"
    _write_search_parquet(path)
    conn = duckdb.connect()
    try:
        correlations = compute_pearson_correlation_matrix(
            conn,
            str(path),
            voxel_id_map=None,
            resolution=1.0,
        )
        matrix_frame, matrix = correlation_long_to_matrix(correlations)
        result = compute_voxel_search(
            conn,
            path,
            VoxelSearchRequest(
                reference_file_ids=("ref",),
                resolution_um=1.0,
                top_n=10,
            ),
        )
    finally:
        conn.close()

    reference_index = matrix_frame.index.get_loc("ref")
    expected = {
        str(file_id): float(1.0 - matrix[reference_index, column])
        for column, file_id in enumerate(matrix_frame.columns)
        if str(file_id) != "ref"
    }
    actual = result.hits.set_index("file_id")["pearson_distance"].to_dict()
    assert actual == pytest.approx(expected)
    assert actual["disjoint"] == pytest.approx(2.0)
    assert next(iter(result.hits["file_id"])) == "similar"
    assert result.input_candidate_count == 3
    assert result.usable_candidate_count == 3


def test_aggregate_search_sums_reference_voxel_counts(tmp_path) -> None:
    import duckdb

    path = tmp_path / "aggregate.parquet"
    rows: list[dict[str, object]] = []
    vectors = {
        "r1": [2, 0, 0],
        "r2": [0, 1, 0],
        "candidate": [2, 1, 0],
        "other": [0, 0, 2],
    }
    for file_id, counts in vectors.items():
        node_id = 100
        for voxel, count in enumerate(counts):
            for _ in range(count):
                rows.append(
                    {
                        "file_id": file_id,
                        "neuron_id": file_id,
                        "subject": "s",
                        "node_id": node_id,
                        "parent_id": -1,
                        "type": 2,
                        "x": float(voxel),
                        "y": 0.0,
                        "z": 0.0,
                    }
                )
                node_id += 11
    pd.DataFrame(rows).to_parquet(path, index=False)

    conn = duckdb.connect()
    try:
        result = compute_voxel_search(
            conn,
            path,
            VoxelSearchRequest(
                reference_file_ids=("r1", "r2"),
                resolution_um=1.0,
                top_n=10,
            ),
        )
    finally:
        conn.close()

    hits = result.hits.set_index("file_id")
    assert set(hits.index) == {"candidate", "other"}
    assert hits.loc["candidate", "pearson_distance"] == pytest.approx(0.0)
    assert hits.loc["other", "pearson_distance"] > 1.0
    assert result.reference_file_ids == ("r1", "r2")


def test_search_keeps_duplicate_display_ids_separate_by_file_id(tmp_path) -> None:
    import duckdb

    path = tmp_path / "search.parquet"
    _write_search_parquet(path)
    conn = duckdb.connect()
    try:
        result = compute_voxel_search(
            conn,
            path,
            VoxelSearchRequest(
                reference_file_ids=("similar",),
                resolution_um=1.0,
                top_n=10,
            ),
        )
    finally:
        conn.close()

    shared = result.hits[result.hits["neuron_id"] == "shared"]
    assert set(shared["file_id"]) == {"ref", "disjoint"}
    assert set(shared["subject"]) == {"subject-a", "subject-c"}


def test_search_blocks_reference_with_no_surviving_nodes(tmp_path) -> None:
    import duckdb

    path = tmp_path / "search.parquet"
    _write_search_parquet(path)
    conn = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="Reference neuron.*ref"):
            compute_voxel_search(
                conn,
                path,
                VoxelSearchRequest(
                    reference_file_ids=("ref",),
                    resolution_um=1.0,
                    voxel_node_filter=VoxelNodeFilter(
                        node_type_mode="include",
                        node_types=(3,),
                    ),
                ),
            )
    finally:
        conn.close()


def test_search_reports_candidates_omitted_by_filter(tmp_path) -> None:
    import duckdb

    path = tmp_path / "filtered.parquet"
    pd.DataFrame(
        {
            "file_id": ["ref", "ref", "kept", "removed"],
            "neuron_id": ["r", "r", "k", "x"],
            "subject": ["s"] * 4,
            "node_id": [1, 2, 1, 1],
            "parent_id": [-1, 1, -1, -1],
            "type": [2, 2, 2, 3],
            "x": [0.0, 1.0, 0.0, 2.0],
            "y": [0.0] * 4,
            "z": [0.0] * 4,
        }
    ).to_parquet(path, index=False)
    conn = duckdb.connect()
    try:
        result = compute_voxel_search(
            conn,
            path,
            VoxelSearchRequest(
                reference_file_ids=("ref",),
                resolution_um=1.0,
                voxel_node_filter=VoxelNodeFilter(
                    node_type_mode="include",
                    node_types=(2,),
                ),
            ),
        )
    finally:
        conn.close()

    assert result.omitted_candidate_file_ids == ("removed",)
    assert result.input_candidate_count == 2
    assert result.usable_candidate_count == 1
    assert result.hits["file_id"].tolist() == ["kept"]


def test_search_csv_round_trip_and_availability(tmp_path) -> None:
    output = tmp_path / "results.csv"
    hits = pd.DataFrame(
        {
            "rank": [1, 2],
            "file_id": ["001", "missing"],
            "neuron_id": ["n1", "n2"],
            "subject": ["s1", "s2"],
            "pearson_distance": [0.125, 1.5],
        }
    )

    export_search_results_csv(output, hits)
    loaded = load_search_results_csv(output, available_file_ids={"001"})

    assert loaded["file_id"].tolist() == ["001", "missing"]
    assert loaded["pearson_distance"].tolist() == pytest.approx([0.125, 1.5])
    assert loaded["available"].tolist() == [True, False]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns=["file_id"]), "missing required"),
        (
            lambda frame: pd.concat([frame, frame], ignore_index=True),
            "duplicate file_id",
        ),
        (
            lambda frame: frame.assign(pearson_distance=np.nan),
            "finite and between",
        ),
        (
            lambda frame: frame.assign(format_version=2),
            "Unsupported search result format",
        ),
    ],
)
def test_search_csv_rejects_invalid_rows(tmp_path, mutator, message) -> None:
    path = tmp_path / "invalid.csv"
    valid = pd.DataFrame(
        {
            "format_version": [1],
            "rank": [1],
            "file_id": ["n1"],
            "neuron_id": ["n1"],
            "subject": ["s"],
            "pearson_distance": [0.5],
        }
    )
    mutator(valid).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_search_results_csv(path)
