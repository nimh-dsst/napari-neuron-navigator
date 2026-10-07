"""Tests for CCF voxel-correlation input preparation."""

from __future__ import annotations

from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.analysis.clustering import (
    ClusterExclusionRule,
    ClusterRegionFilter,
)
from napari_neuron_navigator.analysis.correlation import (
    compute_pearson_correlation_matrix,
    correlation_long_to_matrix,
    count_correlation_input_nodes,
)
from napari_neuron_navigator.analysis.region_filter import PreparedClusterRegionFilter


def _write_count_vectors(path, counts):
    """Write independently specified vectors, with colliding display/node IDs."""
    rows = []
    for file_index, vector in enumerate(counts):
        node_id = 7
        for voxel, count in enumerate(vector):
            for _ in range(count):
                rows.append(
                    {
                        "file_id": f"file-{file_index}",
                        "neuron_id": "shared",
                        "subject": f"subject-{file_index}",
                        "node_id": node_id,
                        "x": (voxel + 0.25) * 25.0,
                        "y": 6.25,
                        "z": 6.25,
                    }
                )
                node_id += 11
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.mark.parametrize("use_map", [False, True])
@pytest.mark.parametrize("corrected", [True, False])
def test_sparse_pearson_matches_numpy_for_every_pair(tmp_path, use_map, corrected):
    counts = np.array(
        [
            [3, 1, 0, 0, 0, 0],
            [6, 2, 0, 0, 0, 0],
            [0, 0, 1, 3, 0, 0],
            [1, 0, 1, 0, 2, 0],
            # Out-of-scope neuron must not expand the occupied voxel universe.
            [0, 0, 0, 0, 0, 7],
        ]
    )
    path = tmp_path / "counts.parquet"
    _write_count_vectors(path, counts)
    ids = [f"file-{i}" for i in range(4)]
    voxel_map = np.arange(8, dtype=np.int32).reshape(8, 1, 1) if use_map else None
    with duckdb.connect() as conn:
        correlations = compute_pearson_correlation_matrix(
            conn,
            str(path),
            voxel_map,
            25.0,
            file_ids=ids,
            use_corrected_pearson=corrected,
        )
    frame, matrix = correlation_long_to_matrix(correlations)
    assert len(correlations) == len(ids) ** 2
    assert frame.index.tolist() == ids
    # Jointly empty columns are outside the existing occupied-union policy.
    expected = np.corrcoef(counts[:4, :5])
    if not corrected:
        expected[(counts[:4] @ counts[:4].T) == 0] = -1.0
    np.testing.assert_allclose(matrix, expected, atol=1e-7)
    assert (frame.loc["file-0", "file-2"] > -1.0) == corrected
    assert (
        correlations.attrs["correlation_metadata"]["use_corrected_pearson"] is corrected
    )


@pytest.mark.parametrize(
    "counts, invalid_ids",
    [
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]], ["file-3"]),
        ([[1], [3]], ["file-0", "file-1"]),
        ([[1, 1], [3, 3]], ["file-0", "file-1"]),
    ],
)
@pytest.mark.parametrize("corrected", [True, False])
def test_zero_variance_neurons_follow_selected_policy(
    tmp_path, counts, invalid_ids, corrected
):
    path = tmp_path / "constant.parquet"
    _write_count_vectors(path, counts)
    with duckdb.connect() as conn:
        correlations = compute_pearson_correlation_matrix(
            conn, str(path), None, 25.0, use_corrected_pearson=corrected
        )
    frame, matrix = correlation_long_to_matrix(correlations)
    assert set(frame.columns) == {f"file-{i}" for i in range(len(counts))} - set(
        invalid_ids if corrected else []
    )
    assert np.isfinite(matrix).all()
    assert (
        correlations.attrs["correlation_metadata"]["zero_variance_file_ids"]
        == invalid_ids
    )
    if not corrected:
        for file_id in invalid_ids:
            assert frame.loc[file_id, file_id] == 1.0
            assert (frame.loc[file_id].drop(file_id) == -1.0).all()


@pytest.mark.parametrize("bad_value", [None, np.nan, np.inf])
def test_matrix_rejects_missing_or_undefined_correlations(bad_value):
    records = [("a", "a", 1.0), ("b", "b", 1.0)]
    if bad_value is not None:
        records.extend([("a", "b", bad_value), ("b", "a", bad_value)])
    with pytest.raises(ValueError, match="complete.*finite"):
        correlation_long_to_matrix(
            pd.DataFrame(records, columns=["swc_id_1", "swc_id_2", "r"])
        )


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("corrected", [True, False])
def test_worker_reports_zero_variance_as_unclustered(tmp_path, scoped, corrected):
    from tests.test_workers import _import_workers_module

    path = tmp_path / "worker.parquet"
    _write_count_vectors(path, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])
    worker = _import_workers_module().CorrelationWorker(
        parquet_path=str(path),
        atlas=SimpleNamespace(resolution=(25.0,) * 3, atlas_name="test"),
        region_selection=None,
        file_ids=[f"file-{i}" for i in range(4)] if scoped else None,
        linkage_method="ward",
        n_clusters=2,
        use_corrected_pearson=corrected,
    )
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.run()
    assert errors == []
    assert len(finished) == 1
    result = finished[0]
    assert result.neuron_ids == [f"file-{i}" for i in range(3 if corrected else 4)]
    assert result.unassigned_neuron_ids == (["file-3"] if corrected else [])
    diagnostics = result.metadata.to_dict()["extra_metadata"]["correlation"]
    assert diagnostics["zero_variance_file_ids"] == ["file-3"]
    assert diagnostics["zero_variance_neuron_count"] == 1
    assert diagnostics["implementation"] == (
        "ccf_pearson_complete_pairs_v2" if corrected else "ccf_pearson_legacy_v1"
    )


@pytest.mark.parametrize("counts", [[[1], [3]], [[1, 0], [1, 1]], [[1]], [[1, 2]]])
@pytest.mark.parametrize("corrected", [True, False])
def test_worker_requires_two_neurons_eligible_under_policy(tmp_path, counts, corrected):
    from tests.test_workers import _import_workers_module

    path = tmp_path / "unclusterable.parquet"
    _write_count_vectors(path, counts)
    worker = _import_workers_module().CorrelationWorker(
        parquet_path=str(path),
        atlas=SimpleNamespace(resolution=(25.0,) * 3, atlas_name="test"),
        region_selection=None,
        use_corrected_pearson=corrected,
    )
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.run()
    if not corrected and len(counts) >= 2:
        assert errors == []
        assert len(finished) == 1
        assert finished[0].neuron_ids == ["file-0", "file-1"]
        return
    assert finished == []
    assert len(errors) == 1
    assert "at least 2 neurons" in errors[0]
    assert ("zero variance" in errors[0]) == corrected


@pytest.mark.parametrize("corrected", [True, False])
def test_genuine_anticorrelation_is_retained(tmp_path, corrected):
    path = tmp_path / "anticorrelated.parquet"
    _write_count_vectors(path, [[3, 1], [1, 3]])
    with duckdb.connect() as conn:
        correlations = compute_pearson_correlation_matrix(
            conn, str(path), None, 25.0, use_corrected_pearson=corrected
        )
    _, matrix = correlation_long_to_matrix(correlations)
    np.testing.assert_allclose(matrix, [[1, -1], [-1, 1]])


def test_count_correlation_input_nodes_supports_unfiltered_file_scope(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n2", "n2", "n3"],
            "x": [10.0, 30.0, 10.0, 55.0, np.nan],
            "y": [10.0, 30.0, 10.0, 55.0, 10.0],
            "z": [10.0, 30.0, 10.0, 55.0, 10.0],
        }
    ).to_parquet(path, index=False)
    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            None,
            25.0,
            file_ids=["n1", "n3"],
        )
    finally:
        conn.close()

    assert count == 2


def test_count_correlation_input_nodes_applies_region_lookup(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["inside", "outside"],
            "x": [10.0, 55.0],
            "y": [10.0, 55.0],
            "z": [10.0, 55.0],
        }
    ).to_parquet(path, index=False)
    voxel_id_map = np.full((4, 4, 4), -1, dtype=np.int32)
    voxel_id_map[0, 0, 0] = 0
    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            voxel_id_map,
            25.0,
        )
    finally:
        conn.close()

    assert count == 1


def test_count_correlation_input_nodes_applies_exclusion_and_keeps_out_of_bounds(
    tmp_path,
) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["kept", "excluded", "outside"],
            "x": [10.0, 30.0, 200.0],
            "y": [10.0, 30.0, 200.0],
            "z": [10.0, 30.0, 200.0],
        }
    ).to_parquet(path, index=False)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    rule = ClusterExclusionRule(
        region_id=10,
        acronym="EXC",
        node_types=(1,),
    )
    prepared = PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(exclude_rules=(rule,)),
        resolution_um=(25.0, 25.0, 25.0),
        atlas_shape=mask.shape,
        include_mask=None,
        exclude_masks=(mask,),
        exclude_mask=mask,
    )
    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            None,
            25.0,
            prepared_region_filter=prepared,
        )
    finally:
        conn.close()

    assert count == 2


def test_empty_correlation_table_returns_empty_matrix_for_actionable_worker_error() -> (
    None
):
    frame, matrix = correlation_long_to_matrix(
        pd.DataFrame(columns=["swc_id_1", "swc_id_2", "r"])
    )

    assert frame.empty
    assert matrix.shape == (0, 0)


def test_ccf_voxel_filter_omits_neuron_with_no_surviving_nodes(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n2", "n2", "n2", "n3", "n3", "n3"],
            "x": [10.0, 10.0, 30.0, 30.0, 60.0, 30.0, 60.0, 60.0],
            "y": [10.0, 10.0, 30.0, 30.0, 60.0, 30.0, 60.0, 60.0],
            "z": [10.0, 10.0, 30.0, 30.0, 60.0, 30.0, 60.0, 60.0],
        }
    ).to_parquet(path, index=False)
    excluded = np.zeros((4, 4, 4), dtype=bool)
    excluded[0, 0, 0] = True
    rule = ClusterExclusionRule(region_id=10, acronym="EXC")
    prepared = PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(exclude_rules=(rule,)),
        resolution_um=(25.0, 25.0, 25.0),
        atlas_shape=excluded.shape,
        include_mask=None,
        exclude_masks=(excluded,),
        exclude_mask=excluded,
    )
    conn = duckdb.connect()
    try:
        correlations = compute_pearson_correlation_matrix(
            conn,
            str(path),
            None,
            25.0,
            prepared_region_filter=prepared,
        )
    finally:
        conn.close()

    ids = set(correlations["swc_id_1"]) | set(correlations["swc_id_2"])
    assert ids == {"n2", "n3"}
