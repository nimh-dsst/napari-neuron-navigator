"""Tests for NeuronDatabase query helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from napari_neuron_navigator.db import NeuronDatabase


def _write_source_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "a.swc", "b.swc"],
            "node_id": [1, 2, 3, 1],
            "type": [1, 3, 2, 1],
            "x": [10.0, 20.0, 30.0, 40.0],
            "y": [10.0, 20.0, 30.0, 40.0],
            "z": [10.0, 20.0, 30.0, 40.0],
            "parent_id": [-1, 1, 2, -1],
            "region_acronym": ["REG", "REG", "REG", "R2"],
            "neuron_id": ["a", "a", "a", "b"],
        }
    ).to_parquet(path, index=False)


def test_get_soma_nodes_for_rendering_returns_only_soma_rows(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    db = NeuronDatabase(path)
    try:
        rows = db.get_soma_nodes_for_rendering(["a.swc", "b.swc"])
    finally:
        db.close()

    # Only type == 1 rows survive, and every column is preserved.
    assert rows["type"].tolist() == [1, 1]
    assert rows["file_id"].tolist() == ["a.swc", "b.swc"]
    assert rows["node_id"].tolist() == [1, 1]
    assert {"parent_id", "x", "y", "z"}.issubset(rows.columns)


def test_get_soma_nodes_for_rendering_empty_file_ids(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    db = NeuronDatabase(path)
    try:
        rows = db.get_soma_nodes_for_rendering([])
    finally:
        db.close()
    assert rows.empty


def test_has_column_reports_parquet_schema_membership(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    db = NeuronDatabase(path)
    try:
        assert db.has_column("file_id") is True
        assert db.has_column("region_id") is False
    finally:
        db.close()


def test_get_unique_node_types_returns_dataset_codes_in_order(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    frame = pd.DataFrame(
        {
            "file_id": ["a", "a", "b", "b", "c"],
            "type": [2, 99, 0, 2, 1],
            "x": [0.0] * 5,
            "y": [0.0] * 5,
            "z": [0.0] * 5,
        }
    )
    frame.to_parquet(path, index=False)
    db = NeuronDatabase(path)
    try:
        values = db.get_unique_node_types()
    finally:
        db.close()

    assert values == (0, 1, 2, 99)


def test_get_neuron_catalog_groups_only_by_file_id(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    pd.DataFrame(
        {
            "file_id": ["a", "a", "b", "b"],
            "neuron_id": ["duplicate"] * 4,
            "subject": ["s1", "s1", "s2", "s2"],
            "type": [1, 2, 1, 2],
            "x": [0.0] * 4,
            "y": [0.0] * 4,
            "z": [0.0] * 4,
        }
    ).to_parquet(path, index=False)
    db = NeuronDatabase(path)
    try:
        catalog = db.get_neuron_catalog()
    finally:
        db.close()

    assert catalog.to_dict("records") == [
        {"file_id": "a", "neuron_id": "duplicate", "subject": "s1"},
        {"file_id": "b", "neuron_id": "duplicate", "subject": "s2"},
    ]


def test_get_dendrite_label_coverage_uses_file_id(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    pd.DataFrame(
        {
            "file_id": ["a", "a", "b", "b"],
            "neuron_id": ["duplicate"] * 4,
            "type": [1, 3, 1, 2],
            "x": [0.0] * 4,
            "y": [0.0] * 4,
            "z": [0.0] * 4,
        }
    ).to_parquet(path, index=False)
    db = NeuronDatabase(path)
    try:
        coverage = db.get_dendrite_label_coverage()
    finally:
        db.close()

    assert coverage.input_file_ids == ("a", "b")
    assert coverage.labeled_file_ids == ("a",)
