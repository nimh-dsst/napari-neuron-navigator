from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari_neuron_navigator.analysis.heatmap import build_node_counts_volume
from napari_neuron_navigator.analysis.mask import (
    build_binary_mask_from_heatmap,
    build_binary_mask_from_threshold_range,
    get_expanded_region_voxel_ids_for_regions,
    get_regions_mask,
    isolate_heatmap_volume_to_region_ids,
    merge_heatmap_volumes,
    otsu_threshold_positive,
    smooth_heatmap_volume,
)
from napari_neuron_navigator.db import NeuronDatabase


class FakeAtlas:
    def __init__(self) -> None:
        self.annotation = np.zeros((5, 5, 5), dtype=np.int32)
        self.resolution = (25.0, 25.0, 25.0)
        self.atlas_name = "fake_atlas"
        self.structures = {}


class _AnnotationAtlas:
    def __init__(self, annotation: np.ndarray) -> None:
        self.annotation = np.asarray(annotation, dtype=np.int32)
        self.resolution = (25.0, 25.0, 25.0)
        self.atlas_name = "annotation_atlas"
        self.structures = {}


class _UnionMaskAtlas:
    """Atlas-like object with explicit per-acronym masks."""

    def __init__(self, masks: dict[str, np.ndarray]) -> None:
        self._masks = {key: np.asarray(value, dtype=np.uint8) for key, value in masks.items()}
        self.resolution = (25.0, 25.0, 25.0)

    def get_structure_mask(self, acronym: str) -> np.ndarray:
        return self._masks[str(acronym)]


def test_smooth_heatmap_volume_sigma_zero_returns_copy() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 1] = 5.0

    smoothed = smooth_heatmap_volume(volume, sigma=0.0)

    assert np.array_equal(smoothed, volume)
    assert smoothed is not volume


def test_merge_heatmap_volumes_sums_inputs() -> None:
    left = np.zeros((3, 3, 3), dtype=np.float32)
    right = np.zeros((3, 3, 3), dtype=np.float32)
    left[1, 1, 1] = 2.0
    right[1, 1, 1] = 3.0
    right[2, 1, 1] = 1.0

    merged = merge_heatmap_volumes([left, right])

    assert float(merged[1, 1, 1]) == 5.0
    assert float(merged[2, 1, 1]) == 1.0


def test_isolate_heatmap_volume_to_region_ids_keeps_selected_region_values() -> None:
    annotation = np.array(
        [
            [[1, 2], [3, 1]],
            [[2, 3], [1, 0]],
        ],
        dtype=np.int32,
    )
    volume = np.arange(annotation.size, dtype=np.float32).reshape(annotation.shape)
    atlas = _AnnotationAtlas(annotation)

    isolated = isolate_heatmap_volume_to_region_ids(volume, atlas, [1])

    expected = np.where(annotation == 1, volume, 0.0).astype(np.float32)
    assert isolated.dtype == np.float32
    np.testing.assert_array_equal(isolated, expected)


def test_isolate_heatmap_volume_to_region_ids_supports_multiple_regions() -> None:
    annotation = np.array([[[1, 2], [3, 4]]], dtype=np.int32)
    volume = np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    atlas = _AnnotationAtlas(annotation)

    isolated = isolate_heatmap_volume_to_region_ids(volume, atlas, [2, 4])

    expected = np.array([[[0.0, 6.0], [0.0, 8.0]]], dtype=np.float32)
    np.testing.assert_array_equal(isolated, expected)


def test_isolate_heatmap_volume_to_region_ids_rejects_empty_selection() -> None:
    volume = np.zeros((2, 2, 2), dtype=np.float32)
    atlas = _AnnotationAtlas(np.zeros((2, 2, 2), dtype=np.int32))

    with pytest.raises(ValueError, match="At least one region ID"):
        isolate_heatmap_volume_to_region_ids(volume, atlas, [])


def test_isolate_heatmap_volume_to_region_ids_rejects_shape_mismatch() -> None:
    volume = np.zeros((2, 2, 2), dtype=np.float32)
    atlas = _AnnotationAtlas(np.zeros((2, 2, 3), dtype=np.int32))

    with pytest.raises(ValueError, match="does not match atlas annotation shape"):
        isolate_heatmap_volume_to_region_ids(volume, atlas, [1])


def test_get_regions_mask_unions_selected_region_masks() -> None:
    atlas = _UnionMaskAtlas(
        {
            "R1": np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
            "R2": np.array([[[0, 0], [1, 0]]], dtype=np.uint8),
        }
    )

    mask = get_regions_mask(atlas, ["R1", "R2"])

    assert mask.dtype == np.bool_
    assert int(mask.sum()) == 2
    assert bool(mask[0, 0, 0])
    assert bool(mask[0, 1, 0])


def test_get_expanded_region_voxel_ids_for_regions_dilates_after_union(monkeypatch) -> None:
    atlas = _UnionMaskAtlas(
        {
            "R1": np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
            "R2": np.array([[[0, 1], [0, 0]]], dtype=np.uint8),
        }
    )
    seen: dict[str, object] = {}

    def fake_dilate(mask, increase_fraction, voxel_spacing_um):
        seen["mask"] = np.asarray(mask, dtype=bool).copy()
        seen["increase_fraction"] = float(increase_fraction)
        seen["voxel_spacing_um"] = tuple(float(value) for value in voxel_spacing_um)
        return np.asarray(mask, dtype=bool)

    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.mask.dilate_mask_to_volume_increase",
        fake_dilate,
    )

    id_map = get_expanded_region_voxel_ids_for_regions(
        atlas,
        ["R1", "R2"],
        increase_fraction=0.2,
    )

    expected_union = np.array([[[True, True], [False, False]]], dtype=bool)
    assert np.array_equal(seen["mask"], expected_union)
    assert seen["increase_fraction"] == 0.2
    assert seen["voxel_spacing_um"] == (25.0, 25.0, 25.0)
    assert sorted(id_map[id_map >= 0].tolist()) == [0, 1]


def test_otsu_threshold_positive_ignores_zero_background() -> None:
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    volume.ravel()[:4] = 1.0
    volume.ravel()[4:8] = 10.0

    threshold = otsu_threshold_positive(volume)

    assert 1.0 <= threshold <= 10.0


def test_build_binary_mask_from_heatmap_all_zero_volume_stays_empty() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)

    mask, threshold, smoothed = build_binary_mask_from_heatmap(
        volume,
        sigma=1.0,
        threshold_mode="otsu",
    )

    assert threshold == 0.0
    assert smoothed.shape == volume.shape
    assert int(mask.sum()) == 0


def test_build_binary_mask_from_threshold_range_lower_only() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 1] = 5.0
    volume[1, 1, 2] = 1.0

    mask = build_binary_mask_from_threshold_range(
        volume,
        lower_threshold=2.0,
    )

    assert int(mask[1, 1, 1]) == 1
    assert int(mask[1, 1, 2]) == 0


def test_build_binary_mask_from_threshold_range_lower_and_upper() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 0] = 1.0
    volume[1, 1, 1] = 5.0
    volume[1, 1, 2] = 10.0

    mask = build_binary_mask_from_threshold_range(
        volume,
        lower_threshold=2.0,
        upper_threshold=6.0,
    )

    assert int(mask[1, 1, 0]) == 0
    assert int(mask[1, 1, 1]) == 1
    assert int(mask[1, 1, 2]) == 0


def test_build_binary_mask_from_threshold_range_rejects_upper_below_lower() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Upper threshold"):
        build_binary_mask_from_threshold_range(
            volume,
            lower_threshold=2.0,
            upper_threshold=1.0,
        )


def test_build_binary_mask_from_heatmap_manual_threshold() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 1] = 5.0
    volume[1, 1, 2] = 1.0

    mask, threshold, _smoothed = build_binary_mask_from_heatmap(
        volume,
        sigma=0.0,
        threshold_mode="manual",
        manual_threshold=2.0,
    )

    assert threshold == 2.0
    assert int(mask[1, 1, 1]) == 1
    assert int(mask[1, 1, 2]) == 0


def test_get_neurons_by_mask_supports_any_node_and_soma_only(tmp_path: Path) -> None:
    atlas = FakeAtlas()
    parquet_path = tmp_path / "neurons.parquet"
    df = pd.DataFrame(
        {
            "file_id": ["file_a", "file_a", "file_b", "file_c"],
            "neuron_id": ["n1", "n1", "n2", "n3"],
            "subject": ["s1", "s1", "s2", "s3"],
            "node_id": [1, 2, 1, 1],
            "parent_id": [-1, 1, -1, -1],
            "x": [0.0, 50.0, 100.0, 75.0],
            "y": [0.0, 75.0, 75.0, 75.0],
            "z": [0.0, 100.0, 100.0, 100.0],
            "type": [1, 2, 1, 1],
            "region_id": [0, 0, 0, 0],
            "region_name": ["", "", "", ""],
            "region_acronym": ["", "", "", ""],
        }
    )
    df.to_parquet(parquet_path, index=False)

    mask = np.zeros(atlas.annotation.shape, dtype=np.uint8)
    mask[2, 3, 4] = 1
    mask[4, 3, 4] = 1

    with NeuronDatabase(parquet_path) as db:
        any_node = db.get_neurons_by_mask(mask, atlas, soma_only=False)
        soma_only = db.get_neurons_by_mask(mask, atlas, soma_only=True)
        restricted_any_node = db.get_neurons_by_mask(
            mask,
            atlas,
            soma_only=False,
            file_ids=["file_a"],
        )
        restricted_soma_only = db.get_neurons_by_mask(
            mask,
            atlas,
            soma_only=True,
            file_ids=["file_a", "file_c"],
        )
        excluded_any_node = db.get_neurons_by_mask(
            mask,
            atlas,
            soma_only=False,
            exclude_file_ids=["file_a"],
        )

    assert any_node["file_id"].tolist() == ["file_a", "file_b"]
    assert soma_only["file_id"].tolist() == ["file_b"]
    assert restricted_any_node["file_id"].tolist() == ["file_a"]
    assert restricted_soma_only.empty
    assert excluded_any_node["file_id"].tolist() == ["file_b"]


def test_mask_query_reads_large_candidates_without_sql(tmp_path):
    """The mask scan must bypass eager SQL even on a DuckDB-backed database."""
    source = tmp_path / "large.parquet"
    with duckdb.connect() as connection:
        connection.execute("""
            COPY (
                SELECT
                    'file_' || lpad((i % 64)::VARCHAR, 4, '0') AS file_id,
                    'same_display_id' AS neuron_id,
                    'subject_' || (i % 64)::VARCHAR AS subject,
                    i AS node_id, -1 AS parent_id, 2 AS type,
                    25.0 AS x, 25.0 AS y, 25.0 AS z
                FROM range(1000000) t(i)
            ) TO ? (FORMAT PARQUET, ROW_GROUP_SIZE 16384)
        """, [str(source)])
    atlas = FakeAtlas()
    mask = np.ones(atlas.annotation.shape, dtype=np.uint8)
    with NeuronDatabase(source) as db:
        connection, db.conn = db.conn, None
        try:
            result = db.get_neurons_by_mask(mask, atlas)
        finally:
            db.conn = connection
        assert connection.execute("SELECT 42").fetchone() == (42,)
    assert result["file_id"].tolist() == [f"file_{i:04}" for i in range(64)]
    assert result["neuron_id"].unique().tolist() == ["same_display_id"]
    assert result["subject"].nunique() == 64


def test_mask_query_matches_later_batches_and_keeps_file_ids_separate(tmp_path, monkeypatch):
    import napari_neuron_navigator.db as db_module

    monkeypatch.setattr(db_module, "_MASK_QUERY_BATCH_SIZE", 2)
    source = tmp_path / "interleaved.parquet"
    ids = ["b/shared", "a/shared", "no_hit", "b/shared", "a/shared", "b/shared"]
    pd.DataFrame({
        "file_id": ids,
        "neuron_id": ["shared"] * len(ids),
        "subject": [file_id.split("/")[0] for file_id in ids],
        "type": [1, 1, 1, 2, 2, 2],
        "x": [25., 25., 25., 50., 0., 50.],
        "y": [25., 25., 25., 50., 0., 50.],
        "z": [25., 25., 25., 50., 0., 50.],
    }).to_parquet(source, index=False)
    atlas = FakeAtlas()
    mask = np.zeros(atlas.annotation.shape, dtype=np.uint8)
    mask[0, 0, 0] = mask[2, 2, 2] = 1
    with NeuronDatabase(source) as db:
        result = db.get_neurons_by_mask(mask, atlas)
        assert result.to_dict("records") == [
            {"file_id": "a/shared", "neuron_id": "shared", "subject": "a"},
            {"file_id": "b/shared", "neuron_id": "shared", "subject": "b"},
        ]
        assert db.get_neurons_by_mask(mask, atlas, soma_only=True).empty
        assert db.get_neurons_by_mask(mask, atlas, file_ids=[]).empty
        assert db.get_neurons_by_mask(
            mask, atlas, file_ids=["a/shared", "b/shared"],
            exclude_file_ids=["b/shared"], node_types=[2],
        )["file_id"].tolist() == ["a/shared"]


def test_mask_query_releases_reader_on_membership_error(tmp_path, monkeypatch):
    source = tmp_path / "neurons.parquet"
    pd.DataFrame({
        "file_id": ["a"], "neuron_id": ["a"], "subject": ["a"],
        "x": [25.], "y": [25.], "z": [25.],
    }).to_parquet(source, index=False)
    atlas = FakeAtlas()
    mask = np.ones(atlas.annotation.shape, dtype=np.uint8)
    def fail_membership(stage, completed, _total):
        if stage.startswith("Searching nodes") and completed:
            raise ValueError("test membership failure")

    with NeuronDatabase(source) as db:
        with pytest.raises(ValueError, match="test membership failure"):
            db.get_neurons_by_mask(mask, atlas, progress_callback=fail_membership)
        assert db.get_neurons_by_mask(mask, atlas)["file_id"].tolist() == ["a"]


def test_get_neurons_by_heatmap_mask_uses_swc_grid_for_soma_membership(
    tmp_path: Path,
) -> None:
    atlas = FakeAtlas()
    parquet_path = tmp_path / "neurons.parquet"
    df = pd.DataFrame(
        {
            "file_id": [
                "inside_soma",
                "outside_transposed",
                "neurite_only",
            ],
            "neuron_id": [
                "inside",
                "outside",
                "neurite",
            ],
            "subject": ["s1", "s2", "s3"],
            "node_id": [1, 1, 1],
            "parent_id": [-1, -1, -1],
            "x": [25.0, 75.0, 25.0],
            "y": [25.0, 25.0, 25.0],
            "z": [75.0, 25.0, 75.0],
            "type": [1, 1, 2],
            "region_id": [0, 0, 0],
            "region_name": ["", "", ""],
            "region_acronym": ["", "", ""],
        }
    )
    df.to_parquet(parquet_path, index=False)

    conn = duckdb.connect()
    try:
        volume = build_node_counts_volume(
            conn,
            str(parquet_path),
            atlas,
        )
    finally:
        conn.close()
    mask = build_binary_mask_from_threshold_range(volume, lower_threshold=2.0)

    assert float(volume[1, 1, 3]) == 2.0
    assert float(volume[3, 1, 1]) == 1.0
    assert int(mask[1, 1, 3]) == 1
    assert int(mask[3, 1, 1]) == 0

    with NeuronDatabase(parquet_path) as db:
        any_node = db.get_neurons_by_mask(mask, atlas, soma_only=False)
        soma_only = db.get_neurons_by_mask(mask, atlas, soma_only=True)

    assert any_node["file_id"].tolist() == ["inside_soma", "neurite_only"]
    assert soma_only["file_id"].tolist() == ["inside_soma"]


def test_get_neurons_by_region_supports_any_node_and_soma_only(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "neurons.parquet"
    df = pd.DataFrame(
        {
            "file_id": ["file_a", "file_a", "file_b", "file_c"],
            "neuron_id": ["n1", "n1", "n2", "n3"],
            "subject": ["s1", "s1", "s2", "s3"],
            "node_id": [1, 2, 1, 1],
            "parent_id": [-1, 1, -1, -1],
            "x": [0.0, 50.0, 100.0, 75.0],
            "y": [0.0, 75.0, 75.0, 75.0],
            "z": [0.0, 100.0, 100.0, 100.0],
            "type": [1, 2, 1, 1],
            "region_id": [202, 101, 101, 303],
            "region_name": ["Region Two", "Region One", "Region One", "Region Three"],
            "region_acronym": ["R2", "R1", "R1", "R3"],
        }
    )
    df.to_parquet(parquet_path, index=False)

    with NeuronDatabase(parquet_path) as db:
        any_node = db.get_neurons_by_region(["R1"], soma_only=False)
        soma_only = db.get_neurons_by_region(["R1"], soma_only=True)
        any_node_by_id = db.get_neurons_by_region_id([101], soma_only=False)
        soma_only_by_id = db.get_neurons_by_region_id([101], soma_only=True)
        restricted_any_node = db.get_neurons_by_region(
            ["R1"],
            soma_only=False,
            file_ids=["file_a"],
        )
        restricted_soma_only = db.get_neurons_by_region(
            ["R1"],
            soma_only=True,
            file_ids=["file_a", "file_c"],
        )
        restricted_any_node_by_id = db.get_neurons_by_region_id(
            [101],
            soma_only=False,
            file_ids=["file_b"],
        )
        restricted_soma_only_by_id = db.get_neurons_by_region_id(
            [101],
            soma_only=True,
            file_ids=["file_a"],
        )
        excluded_any_node = db.get_neurons_by_region(
            ["R1"],
            soma_only=False,
            exclude_file_ids=["file_a"],
        )
        excluded_any_node_by_id = db.get_neurons_by_region_id(
            [101],
            soma_only=False,
            exclude_file_ids=["file_b"],
        )

    assert any_node["file_id"].tolist() == ["file_a", "file_b"]
    assert soma_only["file_id"].tolist() == ["file_b"]
    assert any_node_by_id["file_id"].tolist() == ["file_a", "file_b"]
    assert soma_only_by_id["file_id"].tolist() == ["file_b"]
    assert restricted_any_node["file_id"].tolist() == ["file_a"]
    assert restricted_soma_only.empty
    assert restricted_any_node_by_id["file_id"].tolist() == ["file_b"]
    assert restricted_soma_only_by_id.empty
    assert excluded_any_node["file_id"].tolist() == ["file_b"]
    assert excluded_any_node_by_id["file_id"].tolist() == ["file_a"]


def test_get_neurons_by_mask_supports_explicit_node_type_filters(
    tmp_path: Path,
) -> None:
    atlas = FakeAtlas()
    parquet_path = tmp_path / "neurons.parquet"
    df = pd.DataFrame(
        {
            "file_id": ["axon_file", "basal_file", "apical_file", "outside_file"],
            "neuron_id": ["n1", "n2", "n3", "n4"],
            "subject": ["s1", "s2", "s3", "s4"],
            "node_id": [1, 1, 1, 1],
            "parent_id": [-1, -1, -1, -1],
            "x": [25.0, 25.0, 50.0, 100.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "type": [2, 3, 4, 3],
            "region_id": [0, 0, 0, 0],
            "region_name": ["", "", "", ""],
            "region_acronym": ["", "", "", ""],
        }
    )
    df.to_parquet(parquet_path, index=False)

    mask = np.zeros(atlas.annotation.shape, dtype=np.uint8)
    mask[1, 0, 0] = 1
    mask[2, 0, 0] = 1

    with NeuronDatabase(parquet_path) as db:
        axons = db.get_neurons_by_mask(mask, atlas, node_types=[2])
        dendrites = db.get_neurons_by_mask(mask, atlas, node_types=[3, 4])

    assert axons["file_id"].tolist() == ["axon_file"]
    assert dendrites["file_id"].tolist() == ["apical_file", "basal_file"]


def test_get_neurons_by_region_supports_explicit_node_type_filters(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "neurons.parquet"
    df = pd.DataFrame(
        {
            "file_id": ["axon_file", "basal_file", "apical_file", "other_region"],
            "neuron_id": ["n1", "n2", "n3", "n4"],
            "subject": ["s1", "s2", "s3", "s4"],
            "node_id": [1, 1, 1, 1],
            "parent_id": [-1, -1, -1, -1],
            "x": [0.0, 0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "type": [2, 3, 4, 4],
            "region_id": [101, 101, 101, 202],
            "region_name": [
                "Region One",
                "Region One",
                "Region One",
                "Region Two",
            ],
            "region_acronym": ["R1", "R1", "R1", "R2"],
        }
    )
    df.to_parquet(parquet_path, index=False)

    with NeuronDatabase(parquet_path) as db:
        axons = db.get_neurons_by_region(["R1"], node_types=[2])
        dendrites = db.get_neurons_by_region(["R1"], node_types=[3, 4])
        dendrites_by_id = db.get_neurons_by_region_id([101], node_types=[3, 4])

    assert axons["file_id"].tolist() == ["axon_file"]
    assert dendrites["file_id"].tolist() == ["apical_file", "basal_file"]
    assert dendrites_by_id["file_id"].tolist() == [
        "apical_file",
        "basal_file",
    ]
