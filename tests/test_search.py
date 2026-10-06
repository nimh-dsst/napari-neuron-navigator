"""Tests for similar-neuron voxel-correlation search."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.analysis.correlation import (
    compute_pearson_correlation_matrix,
    correlation_long_to_matrix,
)
from napari_neuron_navigator.analysis.flatmap_correlation import (
    compute_flatmap_voxel_correlation_from_parquet,
    pearson_correlation_from_counts,
)
from napari_neuron_navigator.analysis.search import (
    SEARCH_HEATMAP_MODE_SCORED,
    SEARCH_HEATMAP_MODE_WHOLE,
    SEARCH_REFERENCE_HEATMAP_RGBA,
    SEARCH_SCOPE_CURRENT,
    SEARCH_SPACE_CCF,
    SEARCH_SPACE_FLATMAP,
    SearchResultsDocument,
    VoxelSearchRequest,
    build_filtered_search_heatmap_volume,
    build_search_filter_tags,
    compute_voxel_search,
    export_search_results_csv,
    load_search_results_csv,
    pearson_distance_color_domain,
    pearson_distance_to_hot_rgba,
)
from napari_neuron_navigator.analysis.voxel_filter import VoxelNodeFilter


@pytest.fixture
def search_region_atlas():
    return types.SimpleNamespace(
        structures={
            region_id: {"acronym": acronym, "structure_id_path": path}
            for region_id, acronym, path in (
                (997, "root", [997]),
                (315, "Isocortex", [997, 315]),
                (184, "FRP", [997, 315, 184]),
                (68, "FRP1", [997, 315, 184, 68]),
                (667, "FRP2/3", [997, 315, 184, 667]),
                (526, "FRP5", [997, 315, 184, 526]),
                (672, "CP", [997, 672]),
            )
        }
    )


def test_search_region_filters_show_dataset_leaves_and_ancestors(search_region_atlas):
    from napari_neuron_navigator.widgets.search_tab import SearchTabWidget

    editor = MagicMock()
    widget = types.SimpleNamespace(
        _atlas=search_region_atlas,
        _dataset_region_ids={68, 667, 999999},
        _region_filter_editor=editor,
    )

    SearchTabWidget._refresh_region_editor(widget)

    editor.set_atlas_and_allowed_ids.assert_called_once_with(
        search_region_atlas, {997, 315, 184, 68, 667}
    )
    editor.clear.assert_not_called()
    # Ancestors belong to the display tree, not the represented dataset IDs.
    assert widget._dataset_region_ids == {68, 667, 999999}


@pytest.mark.parametrize("missing", ["atlas", "regions"])
def test_search_region_filters_clear_when_inputs_are_missing(search_region_atlas, missing):
    from napari_neuron_navigator.widgets.search_tab import SearchTabWidget

    editor = MagicMock()
    widget = types.SimpleNamespace(
        _atlas=None if missing == "atlas" else search_region_atlas,
        _dataset_region_ids=set() if missing == "regions" else {68},
        _region_filter_editor=editor,
    )

    SearchTabWidget._refresh_region_editor(widget)

    editor.clear.assert_called_once_with()
    editor.set_atlas_and_allowed_ids.assert_not_called()


@pytest.mark.parametrize(
    ("selected_ids", "expected"),
    [
        ([184], [(68, "FRP1"), (667, "FRP2/3")]),
        ([315, 184, 68], [(68, "FRP1"), (667, "FRP2/3")]),
        ([68], [(68, "FRP1")]),
        ([672], []),
    ],
)
def test_search_parent_regions_resolve_only_represented_descendants(
    search_region_atlas, selected_ids, expected
):
    from napari_neuron_navigator.widgets.search_tab import SearchTabWidget

    widget = types.SimpleNamespace(
        _atlas=search_region_atlas,
        _dataset_region_ids={68, 667, 999999},
    )

    assert SearchTabWidget._represented_region_entries(widget, selected_ids) == expected


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


def _write_flatmap_search_parquet(
    path,
    vectors: dict[str, list[tuple[int, int, float]]],
) -> None:
    """Write nodes as ``(xy_position, count, depth_um)`` flatmap vectors."""
    # Bilateral x spans twice the single-hemisphere y extent, so the shared
    # aspect-ratio policy resolves twice as many x bins as y bins.
    positions = ((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0))
    rows: list[dict[str, object]] = []
    for file_index, (file_id, entries) in enumerate(vectors.items()):
        node_id = 10
        for position, count, depth_um in entries:
            x_flat, y_flat = positions[position]
            for _ in range(count):
                rows.append(
                    {
                        "file_id": file_id,
                        "neuron_id": file_id,
                        "subject": f"subject-{file_index}",
                        "node_id": node_id,
                        "parent_id": -1 if node_id == 10 else 10,
                        "type": 2,
                        "x": float(position),
                        "y": 0.0,
                        "z": 0.0,
                        "x_flat_square": x_flat,
                        "y_flat_square": y_flat,
                        "flatmap_square_valid": True,
                        "depth_um": depth_um,
                        "depth_valid": True,
                    }
                )
                node_id += 7
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.mark.parametrize("corrected", [True, False])
def test_single_reference_search_matches_analysis_distance_row(
    tmp_path, corrected
) -> None:
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
            use_corrected_pearson=corrected,
        )
        matrix_frame, matrix = correlation_long_to_matrix(correlations)
        result = compute_voxel_search(
            conn,
            path,
            VoxelSearchRequest(
                reference_file_ids=("ref",),
                resolution_um=1.0,
                top_n=10,
                use_corrected_pearson=corrected,
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
    expected_disjoint = 1.0 - np.corrcoef([2, 1, 0, 0], [0, 0, 0, 3])[0, 1]
    if not corrected:
        expected_disjoint = 2.0
    assert actual["disjoint"] == pytest.approx(expected_disjoint)
    assert next(iter(result.hits["file_id"])) == "similar"
    assert result.input_candidate_count == 3
    assert result.usable_candidate_count == 3
    assert result.reference_rows["file_id"].tolist() == ["ref"]


@pytest.mark.parametrize("corrected", [True, False])
def test_flatmap_search_ranks_sparse_binned_vectors_and_records_grid(
    tmp_path, corrected
) -> None:
    import duckdb

    path = tmp_path / "flatmap-search.parquet"
    _write_flatmap_search_parquet(
        path,
        {
            "ref": [(0, 2, 0.0), (1, 1, 0.0)],
            "similar": [(0, 4, 0.0), (1, 2, 0.0)],
            "partial": [(0, 1, 0.0), (2, 1, 0.0)],
            "disjoint": [(3, 3, 0.0)],
        },
    )
    conn = duckdb.connect()
    try:
        result = compute_voxel_search(
            conn,
            path,
            VoxelSearchRequest(
                reference_file_ids=("ref",),
                coordinate_space=SEARCH_SPACE_FLATMAP,
                flatmap_style="both_square",
                flatmap_y_bins=2,
                flatmap_collapse_depth=True,
                use_corrected_pearson=corrected,
            ),
        )
    finally:
        conn.close()

    _cluster_result, count_data, provenance = (
        compute_flatmap_voxel_correlation_from_parquet(
            str(path),
            style="both_square",
            y_bins=2,
            depth_bin_um=25.0,
            collapse_depth=True,
            n_clusters=2,
        )
    )
    correlations = pearson_correlation_from_counts(count_data.count_matrix)
    reference_index = count_data.neuron_ids.index("ref")
    expected = {
        file_id: float(1.0 - correlations[reference_index, index])
        for index, file_id in enumerate(count_data.neuron_ids)
        if file_id != "ref"
    }

    hits = result.hits.set_index("file_id")
    assert hits["pearson_distance"].to_dict() == pytest.approx(expected)
    assert hits.loc["similar", "pearson_distance"] == pytest.approx(0.0)
    assert result.metadata["coordinate_space"] == SEARCH_SPACE_FLATMAP
    assert result.metadata["flatmap_style"] == "both_square"
    assert result.metadata["flatmap_y_bins"] == 2
    assert result.metadata["flatmap_x_bins"] == 4
    assert result.metadata["flatmap_volume_shape"] == [2, 4]
    assert list(provenance.volume_shape) == result.metadata["flatmap_volume_shape"]
    assert result.metadata["flatmap_collapse_depth"] is True
    assert "use_corrected_pearson" not in result.metadata
    assert (
        result.metadata["missing_correlation_policy"]
        == "flatmap_dense_zero_filled_r_zero_if_undefined"
    )
    assert "Search space: Flat map + Depth" in result.metadata["filter_tags"]

    output = tmp_path / "flatmap-search.csv"
    export_search_results_csv(output, result)
    reopened = load_search_results_csv(output)
    assert reopened.metadata["coordinate_space"] == SEARCH_SPACE_FLATMAP
    assert reopened.metadata["flatmap_style"] == "both_square"
    assert reopened.metadata["flatmap_y_bins"] == 2
    assert reopened.metadata["flatmap_x_bins"] == 4
    assert reopened.metadata["flatmap_collapse_depth"] is True


def test_flatmap_search_can_include_or_collapse_cortical_depth(tmp_path) -> None:
    import duckdb

    path = tmp_path / "flatmap-depth-search.parquet"
    _write_flatmap_search_parquet(
        path,
        {
            "ref": [(0, 2, 0.0), (1, 1, 0.0)],
            "other-depth": [(0, 4, 50.0), (1, 2, 50.0)],
        },
    )
    common = {
        "reference_file_ids": ("ref",),
        "coordinate_space": SEARCH_SPACE_FLATMAP,
        "flatmap_style": "both_square",
        "flatmap_y_bins": 2,
        "flatmap_depth_bin_um": 25.0,
    }
    conn = duckdb.connect()
    try:
        with_depth = compute_voxel_search(
            conn,
            path,
            VoxelSearchRequest(**common, flatmap_collapse_depth=False),
        )
        collapsed = compute_voxel_search(
            conn,
            path,
            VoxelSearchRequest(**common, flatmap_collapse_depth=True),
        )
    finally:
        conn.close()

    assert with_depth.hits.iloc[0]["pearson_distance"] > 1.0
    assert collapsed.hits.iloc[0]["pearson_distance"] == pytest.approx(0.0)


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
            # Disjoint, nonconstant reference/candidate vectors after filtering.
            "x": [0.0, 0.0, 1.0, 2.0],
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


@pytest.mark.parametrize("corrected", [True, False])
def test_ccf_search_reports_zero_variance_candidates(tmp_path, corrected):
    import duckdb

    from napari_neuron_navigator.widgets.search_tab import SearchTabWidget
    from tests.test_correlation import _write_count_vectors

    path = tmp_path / "constant_candidate.parquet"
    _write_count_vectors(path, [[2, 1, 0], [0, 0, 3], [1, 1, 1]])
    with duckdb.connect() as conn:
        result = compute_voxel_search(
            conn,
            path,
            VoxelSearchRequest(
                reference_file_ids=("file-0",), use_corrected_pearson=corrected
            ),
        )
    assert result.hits["file_id"].tolist() == (
        ["file-1"] if corrected else ["file-1", "file-2"]
    )
    assert result.omitted_candidate_file_ids == (("file-2",) if corrected else ())
    assert result.input_candidate_count == 2
    assert result.usable_candidate_count == (1 if corrected else 2)
    assert result.metadata["zero_variance_candidate_count"] == int(corrected)
    assert result.metadata["missing_correlation_policy"] == (
        "zero_cross_product_omit_zero_variance" if corrected else "pearson_r_minus_one"
    )

    messages = []
    widget = types.SimpleNamespace(
        _set_result_frame=lambda frame: None,
        _status_label=types.SimpleNamespace(setText=messages.append),
        _corrected_pearson_cb=types.SimpleNamespace(isChecked=lambda: not corrected),
    )
    SearchTabWidget._on_search_finished(widget, result)
    assert ("1 candidate(s) were omitted" in messages[0]) == corrected
    assert ("Pearson correlation is undefined" in messages[0]) == corrected
    assert f"CCF Pearson: {'corrected' if corrected else 'legacy'}." in messages[0]
    output = tmp_path / "constant_candidate.csv"
    export_search_results_csv(output, result)
    restored = load_search_results_csv(output)
    assert (
        restored.metadata["missing_correlation_policy"]
        == result.metadata["missing_correlation_policy"]
    )
    assert restored.metadata["zero_variance_candidate_count"] == int(corrected)
    assert restored.metadata["use_corrected_pearson"] is corrected
    assert restored.metadata["implementation"] == result.metadata["implementation"]


@pytest.mark.parametrize(
    "counts, references, message",
    [
        ([[1, 1], [3, 0]], ("file-0",), "aggregate reference has zero variance"),
        (
            [[1, 0], [0, 1], [3, 0]],
            ("file-0", "file-1"),
            "aggregate reference has zero variance",
        ),
        ([[3, 0], [1, 1]], ("file-0",), "No candidate neurons have nonzero variance"),
    ],
)
@pytest.mark.parametrize("corrected", [True, False])
def test_ccf_search_rejects_undefined_correlations(
    tmp_path, counts, references, message, corrected
):
    import duckdb

    from tests.test_correlation import _write_count_vectors

    path = tmp_path / "undefined.parquet"
    _write_count_vectors(path, counts)
    with duckdb.connect() as conn:
        request = VoxelSearchRequest(
            reference_file_ids=references, use_corrected_pearson=corrected
        )
        if corrected:
            with pytest.raises(ValueError, match=message):
                compute_voxel_search(conn, path, request)
        else:
            result = compute_voxel_search(conn, path, request)
            assert not result.hits.empty
            assert (result.hits["pearson_distance"] == 2.0).all()
            assert result.omitted_candidate_file_ids == ()


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

    document = SearchResultsDocument(
        references=pd.DataFrame(
            {
                "file_id": ["ref"],
                "neuron_id": ["r"],
                "subject": ["s0"],
            }
        ),
        hits=hits,
        metadata={
            "candidate_scope": SEARCH_SCOPE_CURRENT,
            "resolution_um": 25.0,
            "region_filter": None,
            "voxel_node_filter": None,
            "filter_tags": ["Search scope: Current Table", "Search filters: none"],
        },
    )

    export_search_results_csv(output, document)
    exported = pd.read_csv(output)
    loaded = load_search_results_csv(output, available_file_ids={"ref", "001"})

    assert exported["row_role"].tolist() == [
        "reference",
        "search_result",
        "search_result",
    ]
    assert exported["rank"].tolist() == [0, 1, 2]
    assert np.isnan(exported.loc[0, "pearson_distance"])
    assert loaded.references["file_id"].tolist() == ["ref"]
    assert loaded.hits["file_id"].tolist() == ["001", "missing"]
    assert loaded.hits["pearson_distance"].tolist() == pytest.approx([0.125, 1.5])
    assert loaded.hits["available"].tolist() == [True, False]
    assert loaded.metadata["candidate_scope"] == SEARCH_SCOPE_CURRENT


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
            lambda frame: frame.assign(format_version=3),
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


def test_legacy_search_csv_still_loads_as_result_only_document(tmp_path) -> None:
    path = tmp_path / "legacy.csv"
    pd.DataFrame(
        {
            "format_version": [1],
            "rank": [9],
            "file_id": ["001"],
            "neuron_id": ["n1"],
            "subject": ["s"],
            "pearson_distance": [0.25],
        }
    ).to_csv(path, index=False)

    loaded = load_search_results_csv(path, available_file_ids={"001"})

    assert loaded.format_version == 1
    assert loaded.references.empty
    assert loaded.hits["rank"].tolist() == [1]
    assert loaded.hits["file_id"].tolist() == ["001"]


def test_candidate_scope_can_exclude_reference_from_candidate_input(tmp_path) -> None:
    import duckdb

    path = tmp_path / "search.parquet"
    _write_search_parquet(path)
    conn = duckdb.connect()
    try:
        result = compute_voxel_search(
            conn,
            path,
            VoxelSearchRequest(
                reference_file_ids=("ref",),
                candidate_file_ids=("similar", "disjoint"),
                candidate_scope=SEARCH_SCOPE_CURRENT,
                resolution_um=1.0,
            ),
        )
    finally:
        conn.close()

    assert result.hits["file_id"].tolist() == ["similar", "disjoint"]
    assert result.input_candidate_count == 2
    assert result.metadata["candidate_scope"] == SEARCH_SCOPE_CURRENT
    assert result.metadata["candidate_file_ids"] == ["similar", "disjoint"]
    assert result.metadata["candidate_scope_input_count"] == 2
    assert result.metadata["candidate_scope_label"] == (
        "Current Table (2 rows; 2 non-reference candidates)"
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.assign(format_version=2.5),
            "integer format version",
        ),
        (
            lambda frame: frame.assign(row_role="search_result"),
            "reference row",
        ),
        (
            lambda frame: frame.assign(rank=[1, 1, 2]),
            "reference row must have rank 0",
        ),
        (
            lambda frame: frame.assign(pearson_distance=[0.0, 0.25, 0.5]),
            "reference pearson_distance",
        ),
        (
            lambda frame: frame.assign(
                search_context_json=["{}", "{}", '{"different":true}']
            ),
            "conflicting search context",
        ),
    ],
)
def test_version_2_search_csv_validates_roles_ranks_and_context(
    tmp_path,
    mutator,
    message,
) -> None:
    path = tmp_path / "invalid-v2.csv"
    frame = pd.DataFrame(
        {
            "format_version": [2, 2, 2],
            "row_role": ["reference", "search_result", "search_result"],
            "rank": [0, 1, 2],
            "file_id": ["ref", "001", "002"],
            "neuron_id": ["r", "n1", "n2"],
            "subject": ["s", "s", "s"],
            "pearson_distance": [np.nan, 0.25, 0.5],
            "search_context_json": ["{}", "{}", "{}"],
        }
    )
    mutator(frame).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_search_results_csv(path)


def test_filter_tags_and_hot_distance_mapping_are_deterministic() -> None:
    tags = build_search_filter_tags(
        SEARCH_SCOPE_CURRENT,
        {
            "include_rules": [{"acronym": "MOp", "dilation_fraction": 0.1}],
            "exclude_rules": [],
        },
        {
            "node_type_mode": "include",
            "node_types": [2, 3],
            "require_dendrite_labels": True,
            "exclude_within_soma_um": 200.0,
        },
    )

    assert tags == (
        "Search scope: Current Table",
        "Search include: MOp (+10%)",
        "Search node types: include 2|3",
        "Search dendrite labels: required",
        "Search soma distance: >=200 um",
    )
    close = pearson_distance_to_hot_rgba(0.0)
    middle = pearson_distance_to_hot_rgba(1.0)
    far = pearson_distance_to_hot_rgba(2.0)
    assert close == (1.0, 1.0, 1.0, 1.0)
    assert far[0] > 0.0
    assert sum(close[:3]) > sum(middle[:3]) > sum(far[:3])

    observed_domain = pearson_distance_color_domain([0.4, 0.5, 0.8])
    assert observed_domain == pytest.approx((0.4, 0.8))
    observed_close = pearson_distance_to_hot_rgba(
        0.4,
        distance_domain=observed_domain,
    )
    observed_middle = pearson_distance_to_hot_rgba(
        0.6,
        distance_domain=observed_domain,
    )
    observed_far = pearson_distance_to_hot_rgba(
        0.8,
        distance_domain=observed_domain,
    )
    assert observed_close == close
    assert observed_far == far
    assert sum(observed_close[:3]) > sum(observed_middle[:3]) > sum(observed_far[:3])
    assert (
        pearson_distance_to_hot_rgba(
            0.5,
            distance_domain=(0.5, 0.5),
        )
        == close
    )


def test_selected_heatmaps_use_the_full_result_table_distance_domain() -> None:
    from napari_neuron_navigator.widgets.search_tab import SearchTabWidget

    references = pd.DataFrame(
        {
            "rank": [0],
            "file_id": ["ref"],
            "neuron_id": ["r"],
            "subject": ["s"],
            "available": [True],
        }
    )
    hits = pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "file_id": ["h1", "h2", "h3"],
            "neuron_id": ["n1", "n2", "n3"],
            "subject": ["s", "s", "s"],
            "pearson_distance": [0.4, 0.6, 0.8],
            "available": [True, True, True],
        }
    )
    document = SearchResultsDocument(
        references=references,
        hits=hits,
        metadata={
            "resolution_um": 25.0,
            "atlas_name": "test_atlas",
            "region_filter": None,
            "voxel_node_filter": {
                "node_type_mode": "include",
                "node_types": [2],
            },
        },
    )
    emitted = []
    widget = types.SimpleNamespace(
        _result_document=document,
        _result_frame=hits,
        _atlas=types.SimpleNamespace(
            resolution=(25.0, 25.0, 25.0),
            atlas_name="test_atlas",
        ),
        _heatmap_busy=False,
        _status_label=types.SimpleNamespace(setText=lambda _message: None),
        _update_button_states=lambda: None,
        search_heatmaps_requested=types.SimpleNamespace(emit=emitted.append),
    )
    widget._result_distance_domain = types.MethodType(
        SearchTabWidget._result_distance_domain,
        widget,
    )

    SearchTabWidget._emit_search_heatmap_request(widget, ["h2"])

    assert len(emitted) == 1
    request = emitted[0]
    assert request.voxel_mode == SEARCH_HEATMAP_MODE_SCORED
    assert request.voxel_node_filter is not None
    assert request.distance_color_domain == pytest.approx((0.4, 0.8))
    assert [layer.file_ids for layer in request.layers] == [("ref",), ("h2",)]
    assert request.layers[0].color == SEARCH_REFERENCE_HEATMAP_RGBA
    assert request.layers[1].color == pytest.approx(
        pearson_distance_to_hot_rgba(
            0.6,
            distance_domain=(0.4, 0.8),
        )
    )
    assert request.metadata["heatmap_distance_color_basis"] == (
        "completed_result_table"
    )
    assert request.metadata["heatmap_reference_color"] == "magenta"
    assert request.metadata["heatmap_reference_rgba"] == list(
        SEARCH_REFERENCE_HEATMAP_RGBA
    )
    assert request.metadata["heatmap_voxel_mode"] == SEARCH_HEATMAP_MODE_SCORED
    assert request.metadata["heatmap_filters_applied"] is True

    widget._heatmap_busy = False
    SearchTabWidget._emit_search_heatmap_request(
        widget,
        ["h2"],
        voxel_mode=SEARCH_HEATMAP_MODE_WHOLE,
    )

    whole_request = emitted[1]
    assert whole_request.voxel_mode == SEARCH_HEATMAP_MODE_WHOLE
    assert whole_request.region_filter is None
    assert whole_request.voxel_node_filter is None
    assert whole_request.metadata["heatmap_voxel_mode"] == (SEARCH_HEATMAP_MODE_WHOLE)
    assert whole_request.metadata["heatmap_filters_applied"] is False


def test_flatmap_search_result_cannot_emit_ccfv3_heatmap_request() -> None:
    from napari_neuron_navigator.widgets.search_tab import SearchTabWidget

    messages = []
    emitted = []
    widget = types.SimpleNamespace(
        _result_document=SearchResultsDocument(
            references=pd.DataFrame({"file_id": ["ref"]}),
            hits=pd.DataFrame(
                {"file_id": ["hit"], "rank": [1], "pearson_distance": [0.2]}
            ),
            metadata={"coordinate_space": SEARCH_SPACE_FLATMAP},
        ),
        _atlas=object(),
        _heatmap_busy=False,
        _status_label=types.SimpleNamespace(setText=messages.append),
        search_heatmaps_requested=types.SimpleNamespace(emit=emitted.append),
    )

    SearchTabWidget._emit_search_heatmap_request(widget, ["hit"])

    assert emitted == []
    assert "unavailable for flatmap-space results" in messages[-1]


def test_flatmap_search_results_emit_data_colors_over_full_distance_domain() -> None:
    from napari_neuron_navigator.widgets.search_tab import SearchTabWidget

    references = pd.DataFrame({"file_id": ["ref-1", "ref-2"]})
    hits = pd.DataFrame(
        {
            "file_id": ["h1", "h2", "h3"],
            "rank": [1, 2, 3],
            "pearson_distance": [0.4, 0.6, 0.8],
        }
    )
    document = SearchResultsDocument(
        references=references,
        hits=hits,
        metadata={"coordinate_space": SEARCH_SPACE_FLATMAP},
    )
    emitted = []
    widget = types.SimpleNamespace(
        _result_document=document,
        _result_frame=hits,
        _status_label=types.SimpleNamespace(setText=lambda _message: None),
        apply_search_colors_requested=types.SimpleNamespace(emit=emitted.append),
    )
    widget._result_distance_domain = types.MethodType(
        SearchTabWidget._result_distance_domain,
        widget,
    )

    SearchTabWidget._apply_search_colors_to_data(widget)

    assert len(emitted) == 1
    request = emitted[0]
    assert request.reference_file_ids == ("ref-1", "ref-2")
    assert request.distance_color_domain == pytest.approx((0.4, 0.8))
    assert [file_id for file_id, _color in request.colors_by_file_id] == [
        "ref-1",
        "ref-2",
        "h1",
        "h2",
        "h3",
    ]
    colors = dict(request.colors_by_file_id)
    assert colors["ref-1"] == SEARCH_REFERENCE_HEATMAP_RGBA
    assert colors["ref-2"] == SEARCH_REFERENCE_HEATMAP_RGBA
    assert colors["h1"] == pearson_distance_to_hot_rgba(
        0.4,
        distance_domain=(0.4, 0.8),
    )
    assert colors["h2"] == pearson_distance_to_hot_rgba(
        0.6,
        distance_domain=(0.4, 0.8),
    )
    assert colors["h3"] == pearson_distance_to_hot_rgba(
        0.8,
        distance_domain=(0.4, 0.8),
    )


def test_flatmap_results_enable_data_colors_while_heatmaps_stay_disabled() -> None:
    from napari_neuron_navigator.widgets.search_tab import SearchTabWidget

    class Control:
        def __init__(self, *, data=None, checked=False) -> None:
            self.data = data
            self.checked = checked
            self.enabled = None
            self.tooltip = None

        def currentData(self):
            return self.data

        def isChecked(self):
            return self.checked

        def setEnabled(self, enabled) -> None:
            self.enabled = bool(enabled)

        def setToolTip(self, tooltip) -> None:
            self.tooltip = str(tooltip)

    controls = {
        name: Control()
        for name in (
            "_run_btn",
            "_dendrite_scan_btn",
            "_add_all_btn",
            "_add_selected_btn",
            "_annotate_btn",
            "_apply_colors_btn",
            "_heatmap_btn",
            "_heatmap_scored_all_action",
            "_heatmap_whole_all_action",
            "_heatmap_scored_selected_action",
            "_heatmap_whole_selected_action",
            "_save_csv_btn",
            "_load_csv_btn",
            "_reference_section",
            "_filter_section",
            "_scope_combo",
            "_coordinate_space_combo",
            "_corrected_pearson_cb",
            "_flatmap_style_combo",
            "_flatmap_y_bins_spin",
            "_flatmap_depth_bin_spin",
            "_flatmap_include_depth_minus_one_cb",
            "_top_n_spin",
        )
    }
    controls["_reference_mode_combo"] = Control(data="single")
    controls["_flatmap_ignore_depth_cb"] = Control(checked=False)
    document = SearchResultsDocument(
        references=pd.DataFrame({"file_id": ["ref"]}),
        hits=pd.DataFrame({"file_id": ["hit"], "rank": [1], "pearson_distance": [0.2]}),
        metadata={"coordinate_space": SEARCH_SPACE_FLATMAP},
    )
    widget = types.SimpleNamespace(
        **controls,
        _db=object(),
        _atlas=object(),
        _result_document=document,
        _result_frame=document.hits,
        _heatmap_busy=False,
        _is_busy=lambda: False,
        _pending_request=None,
        _reference_file_ids=lambda: ("ref",),
        _available_dendrite_node_types=lambda: (),
        _selected_result_rows=list,
    )

    SearchTabWidget._update_button_states(widget)

    assert widget._apply_colors_btn.enabled is True
    assert widget._heatmap_btn.enabled is False
    assert "Apply Search Colors to Data" in widget._heatmap_btn.tooltip


@pytest.mark.parametrize("coordinate_space", [SEARCH_SPACE_FLATMAP, SEARCH_SPACE_CCF])
@pytest.mark.parametrize("corrected", [True, False])
def test_search_widget_snapshots_controls_into_request(
    coordinate_space, corrected
) -> None:
    from napari_neuron_navigator.widgets.search_tab import SearchTabWidget

    def control(value):
        return types.SimpleNamespace(value=lambda: value)

    def checked(value):
        return types.SimpleNamespace(isChecked=lambda: value)

    widget = types.SimpleNamespace(
        _reference_file_ids=lambda: ("ref",),
        _reference_mode_combo=types.SimpleNamespace(currentData=lambda: "single"),
        _resolve_candidate_scope=lambda _references: (("candidate",), "scope", 1),
        _region_filter_editor=types.SimpleNamespace(
            region_filter=lambda _resolver: None
        ),
        _represented_region_entries=lambda _ids: [],
        _selected_voxel_node_filter=lambda: None,
        _selected_coordinate_space=lambda: coordinate_space,
        _corrected_pearson_cb=checked(corrected),
        _selected_flatmap_style=lambda: "both_shaped",
        _atlas=types.SimpleNamespace(resolution=(25.0, 25.0, 25.0)),
        _top_n_spin=control(17),
        _selected_scope=lambda: SEARCH_SCOPE_CURRENT,
        _flatmap_y_bins_spin=control(64),
        _flatmap_depth_bin_spin=control(50.0),
        _flatmap_include_depth_minus_one_cb=checked(False),
        _flatmap_ignore_depth_cb=checked(True),
    )

    request = SearchTabWidget._build_request(widget)

    assert request.reference_file_ids == ("ref",)
    assert request.candidate_file_ids == ("candidate",)
    assert request.coordinate_space == coordinate_space
    assert request.flatmap_style == (
        "both_shaped" if coordinate_space == "flatmap" else None
    )
    assert request.use_corrected_pearson is (
        corrected if coordinate_space == SEARCH_SPACE_CCF else True
    )
    widget._corrected_pearson_cb = checked(not corrected)
    assert request.use_corrected_pearson is (
        corrected if coordinate_space == SEARCH_SPACE_CCF else True
    )
    assert request.flatmap_y_bins == 64
    assert request.flatmap_x_bins is None
    assert request.flatmap_depth_bin_um == 50.0
    assert request.flatmap_include_depth_minus_one is False
    assert request.flatmap_collapse_depth is True


@pytest.mark.parametrize("cancel", [True, False])
@pytest.mark.parametrize("corrected", [True, False])
def test_search_preflight_retains_policy_or_unlocks_on_cancel(
    monkeypatch, cancel, corrected
):
    from napari_neuron_navigator.widgets import search_tab

    request = VoxelSearchRequest(
        reference_file_ids=("ref",), use_corrected_pearson=corrected
    )
    preflight = types.SimpleNamespace(
        node_count=search_tab._LARGE_SEARCH_NODE_THRESHOLD + 1
    )
    continue_button, cancel_button = object(), object()
    prompt = MagicMock()
    prompt.addButton.side_effect = [continue_button, cancel_button]
    prompt.clickedButton.return_value = cancel_button if cancel else continue_button
    monkeypatch.setattr(search_tab, "QMessageBox", MagicMock(return_value=prompt))
    widget = types.SimpleNamespace(
        _pending_request=request,
        _pending_preflight=preflight,
        _release_thread=MagicMock(),
        _update_button_states=MagicMock(),
        _launch_search=MagicMock(),
        _status_label=MagicMock(),
        _corrected_pearson_cb=types.SimpleNamespace(isChecked=lambda: not corrected),
    )
    search_tab.SearchTabWidget._on_preflight_thread_finished(widget)
    if cancel:
        assert widget._pending_request is None
        assert widget._pending_preflight is None
        widget._launch_search.assert_not_called()
        widget._update_button_states.assert_called_once()
    else:
        widget._launch_search.assert_called_once_with(request, preflight)
        assert request.use_corrected_pearson is corrected


@pytest.mark.parametrize("corrected", [True, False, None])
def test_imported_search_policy_uses_recorded_metadata(
    tmp_path, monkeypatch, corrected
):
    from napari_neuron_navigator.analysis.pearson_policy import (
        ccf_pearson_policy_metadata,
    )
    from napari_neuron_navigator.widgets import search_tab

    policy = {} if corrected is None else ccf_pearson_policy_metadata(corrected)
    document = SearchResultsDocument(
        references=pd.DataFrame(
            {"file_id": ["ref"], "neuron_id": ["r"], "subject": ["s"]}
        ),
        hits=pd.DataFrame(
            {
                "file_id": ["hit"],
                "neuron_id": ["h"],
                "subject": ["s"],
                "rank": [1],
                "pearson_distance": [0.2],
            }
        ),
        metadata={"coordinate_space": SEARCH_SPACE_CCF, **policy},
    )
    output = tmp_path / "recorded.csv"
    export_search_results_csv(output, document)
    monkeypatch.setattr(
        search_tab,
        "QFileDialog",
        types.SimpleNamespace(
            getOpenFileName=lambda *args: (str(output), ""),
        ),
    )
    messages = []
    widget = types.SimpleNamespace(
        _db=object(),
        _available_file_ids={"ref", "hit"},
        _set_result_frame=MagicMock(),
        _status_label=types.SimpleNamespace(setText=messages.append),
        _corrected_pearson_cb=types.SimpleNamespace(isChecked=lambda: not corrected),
    )
    search_tab.SearchTabWidget._load_results_csv(widget)
    if corrected is None:
        assert "CCF Pearson:" not in messages[-1]
        assert "use_corrected_pearson" not in widget._result_document.metadata
    else:
        assert f"CCF Pearson: {'corrected' if corrected else 'legacy'}." in messages[-1]
        assert widget._result_document.metadata["use_corrected_pearson"] is corrected
    assert widget._result_document.hits["pearson_distance"].tolist() == [0.2]


def test_filtered_search_heatmap_combines_requested_reference_files(tmp_path) -> None:
    import duckdb

    path = tmp_path / "search.parquet"
    _write_search_parquet(path)
    conn = duckdb.connect()
    try:
        volume = build_filtered_search_heatmap_volume(
            conn,
            path,
            atlas_shape=(5, 2, 2),
            resolution_um=1.0,
            file_ids=("ref", "similar"),
            voxel_node_filter=VoxelNodeFilter(
                node_type_mode="include",
                node_types=(2,),
            ),
        )
    finally:
        conn.close()

    assert volume.dtype == np.float32
    assert volume[0, 0, 0] == pytest.approx(6.0)
    assert volume[1, 0, 0] == pytest.approx(3.0)
    assert float(volume.sum()) == pytest.approx(9.0)
