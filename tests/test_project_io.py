from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import napari_neuron_navigator.project_io as project_io_module
from napari_neuron_navigator.cluster_assignments import ClusterAssignmentStore
from napari_neuron_navigator.project_io import (
    ENHANCED_NEURON_COLUMNS,
    LEGACY_PROJECT_BUNDLE_FORMAT,
    LEGACY_PROJECT_METADATA_PREFIX,
    PROJECT_BUNDLE_FORMAT,
    export_enhanced_neuron_parquet,
    export_filtered_project_neuron_parquet,
    is_recognized_project_bundle,
    load_project_bundle,
    read_enhanced_parquet_metadata,
    save_project_bundle,
)


def _write_source_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n2"],
            "node_id": [1, 2, 1],
            "type": [1, 3, 1],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
            "radius": [1.0, 1.0, 1.0],
            "parent_id": [-1, 1, -1],
            "region_id": [1, 1, 2],
            "region_name": ["A", "A", "B"],
            "region_acronym": ["A", "A", "B"],
            "subject": ["s1", "s1", "s2"],
            "neuron_id": ["neuron1", "neuron1", "neuron2"],
        }
    ).to_parquet(path, index=False)


def _write_three_neuron_source_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n2", "n3", "n3"],
            "node_id": [1, 2, 1, 1, 2],
            "type": [1, 3, 1, 1, 3],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [6.0, 7.0, 8.0, 9.0, 10.0],
            "z": [11.0, 12.0, 13.0, 14.0, 15.0],
            "radius": [1.0, 1.0, 1.0, 2.0, 2.0],
            "parent_id": [-1, 1, -1, -1, 1],
            "region_id": [1, 1, 2, 3, 3],
            "region_name": ["A", "A", "B", "C", "C"],
            "region_acronym": ["A", "A", "B", "C", "C"],
            "subject": ["s1", "s1", "s2", "s3", "s3"],
            "neuron_id": ["neuron1", "neuron1", "neuron2", "neuron3", "neuron3"],
        }
    ).to_parquet(path, index=False)


def _add_flatmap_v3_schema_metadata(path: Path) -> dict[bytes, bytes]:
    """Attach representative v3 and custom Arrow metadata to a fixture."""
    table = pq.read_table(path)
    metadata = {
        b"napari_neuron_navigator.flatmap_projection_json": (
            b'{"version":3,"lookup_set_id":"lookup-test"}'
        ),
        b"custom.dataset_metadata": b"must-survive",
    }
    fields = [
        field.with_metadata({b"units": b"micrometer"}) if field.name == "x" else field
        for field in table.schema
    ]
    schema = pa.schema(fields, metadata=metadata)
    pq.write_table(pa.Table.from_arrays(table.columns, schema=schema), path)
    return metadata


def _table_state() -> dict[str, object]:
    return {
        "version": 1,
        "entries": [
            {
                "file_id": "n1",
                "subject": "s1",
                "label": "projection",
                "group": "A",
                "tags": ["axon", "reviewed"],
                "notes": "keep",
                "cluster_id": 3,
                "color": [0.1, 0.2, 0.3, 1.0],
                "visible": False,
            }
        ],
    }


def _subset_table_state() -> dict[str, object]:
    return {
        "version": 1,
        "entries": [
            {
                "file_id": "n3",
                "subject": "s3",
                "label": "target",
                "group": "GPe",
                "tags": ["filtered"],
                "notes": "save",
                "cluster_id": 7,
            },
            {
                "file_id": "n1",
                "subject": "s1",
                "label": "source",
                "group": "CTX",
                "tags": ["reviewed"],
                "notes": "keep",
                "cluster_id": 2,
            },
        ],
    }


def _multi_assignment_table_state() -> dict[str, object]:
    store = ClusterAssignmentStore()
    soma = store.add(
        name="Soma Location 1",
        assignments={"n1": 1, "n2": 1, "n3": 2},
        input_file_ids=["n1", "n2", "n3"],
        activate=True,
    )
    store.add(
        name="Voxel Correlation 1",
        assignments={"n1": 2, "n2": 1},
        input_file_ids=["n1", "n2"],
        parent_assignment_id=soma.assignment_id,
        parent_cluster_ids=[1],
        activate=True,
    )
    return {
        "version": 2,
        "entries": [
            {"file_id": "n1", "subject": "s1", "cluster_id": 2},
            {"file_id": "n2", "subject": "s2", "cluster_id": 1},
            {"file_id": "n3", "subject": "s3", "cluster_id": None},
        ],
        "cluster_assignments": store.to_state(),
    }


@dataclass
class _DummyLayer:
    name: str
    data: np.ndarray
    metadata: dict[str, object]
    opacity: float = 0.6
    visible: bool = True
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    blending: str = "additive"
    rendering: str = "mip"
    contrast_limits: tuple[float, float] = (0.0, 1.0)
    colormap: object = field(default_factory=lambda: SimpleNamespace(name="hot"))


class _DummyColormap:
    def __init__(self, name: str, colors: list[list[float]]) -> None:
        self.name = name
        self.colors = np.asarray(colors, dtype=np.float32)
        self.controls = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
        self.interpolation = "linear"
        self.low_color = None
        self.high_color = None
        self.nan_color = [0.0, 0.0, 0.0, 0.0]


def test_export_enhanced_neuron_parquet_round_trips_labels_and_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "enhanced.parquet"
    _write_source_parquet(source)

    export_enhanced_neuron_parquet(
        source,
        output,
        table_state=_table_state(),
        metadata={"atlas_name": "fake_atlas"},
    )

    loaded = (
        pd.read_parquet(output)
        .sort_values(["file_id", "node_id"])
        .reset_index(drop=True)
    )
    assert len(loaded) == 3
    assert all(column in loaded.columns for column in ENHANCED_NEURON_COLUMNS)
    assert loaded.loc[loaded["file_id"] == "n1", "neuron_label"].unique().tolist() == [
        "projection"
    ]
    cluster_values = (
        loaded.loc[loaded["file_id"] == "n1", "cluster_assignment"]
        .dropna()
        .unique()
        .tolist()
    )
    assert cluster_values == [3]
    assert loaded.loc[loaded["file_id"] == "n2", "cluster_assignment"].isna().all()

    payload = read_enhanced_parquet_metadata(output)
    assert payload["metadata"]["atlas_name"] == "fake_atlas"
    assert payload["table_state"]["entries"][0]["label"] == "projection"
    assert payload["enhanced_columns"] == [
        *ENHANCED_NEURON_COLUMNS,
        "cluster_imported_cluster_assignment",
    ]


def test_project_and_enhanced_parquet_preserve_live_sets_and_mask_snapshot(tmp_path):
    from napari_neuron_navigator.neuron_sets import AnnotationRule, NeuronSetStore

    source = tmp_path / "source.parquet"
    _write_source_parquet(source)
    state = _table_state()
    store = NeuronSetStore()
    item = store.save("Reviewed", AnnotationRule(tag_filter=("reviewed",)))
    state["neuron_sets"] = store.to_state()
    output = tmp_path / "enhanced.parquet"
    export_enhanced_neuron_parquet(source, output, table_state=state)
    assert read_enhanced_parquet_metadata(output)["table_state"]["neuron_sets"] == store.to_state()
    snapshot = {**item.to_state(), "file_ids": ["n1"]}
    layer = _DummyLayer("Region", np.ones((1, 2, 2), dtype=np.uint8), {
        "mask_query_source": True, "atlas_name": "fake_atlas",
        "neuron_set": snapshot, "source_file_ids": ["n1"],
        "sigma": 1.0, "lower_threshold": 0.25,
    })
    layer.rotate = np.eye(3)
    layer.shear = (0.0, 0.0, 0.0)
    layer.affine = SimpleNamespace(affine_matrix=np.eye(4))
    intersection = _DummyLayer(
        "Intersection: A ∩ B", np.array([[[0, 1], [0, 0]]], dtype=np.uint8),
        {"mask_query_source": True, "mask_operation": "intersection",
         "atlas_name": "fake_atlas", "source_file_ids": ["n1", "n2"],
         "source_masks": [
             {"name": "A", "metadata": layer.metadata},
             {"name": "B", "metadata": {"source_file_ids": ["n2"], "sigma": 2}},
         ]},
    )
    destination = tmp_path / "sets.nnproj"
    save_project_bundle(destination, source_parquet_path=source, table_state=state,
                        layers=[layer, intersection], atlas_name="fake_atlas")
    bundle = load_project_bundle(destination)
    assert bundle.table_state["neuron_sets"] == store.to_state()
    assert bundle.layers[0].metadata["metadata"]["neuron_set"] == snapshot
    assert bundle.layers[0].metadata["display"]["affine"] == np.eye(4).tolist()
    np.testing.assert_array_equal(bundle.layers[1].data, intersection.data)
    assert bundle.layers[1].metadata["metadata"] == intersection.metadata


def test_enhanced_parquet_writes_all_named_assignment_columns_and_active_mirror(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "enhanced.parquet"
    _write_three_neuron_source_parquet(source)

    export_enhanced_neuron_parquet(
        source,
        output,
        table_state=_multi_assignment_table_state(),
    )

    loaded = pd.read_parquet(output)
    assert set(loaded.loc[loaded["file_id"] == "n1", "cluster_soma_location_1"]) == {1}
    assert set(loaded.loc[loaded["file_id"] == "n3", "cluster_soma_location_1"]) == {2}
    assert set(
        loaded.loc[loaded["file_id"] == "n1", "cluster_voxel_correlation_1"]
    ) == {2}
    assert (
        loaded.loc[loaded["file_id"] == "n3", "cluster_voxel_correlation_1"]
        .isna()
        .all()
    )
    assert set(loaded.loc[loaded["file_id"] == "n2", "cluster_assignment"]) == {1}
    assert loaded.loc[loaded["file_id"] == "n3", "cluster_assignment"].isna().all()

    payload = read_enhanced_parquet_metadata(output)
    registry = payload["table_state"]["cluster_assignments"]
    assert registry["active_assignment_id"] == registry["sets"][1]["assignment_id"]
    assert [item["name"] for item in registry["sets"]] == [
        "Soma Location 1",
        "Voxel Correlation 1",
    ]


def test_enhanced_parquet_uses_collision_safe_assignment_column(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "enhanced.parquet"
    _write_three_neuron_source_parquet(source)
    source_df = pd.read_parquet(source)
    source_df["cluster_soma_location_1"] = 99
    source_df.to_parquet(source, index=False)

    export_enhanced_neuron_parquet(
        source,
        output,
        table_state=_multi_assignment_table_state(),
    )

    loaded = pd.read_parquet(output)
    assert set(loaded["cluster_soma_location_1"]) == {99}
    assert "cluster_soma_location_1_2" in loaded.columns
    payload = read_enhanced_parquet_metadata(output)
    assert (
        payload["table_state"]["cluster_assignments"]["sets"][0]["column_name"]
        == "cluster_soma_location_1_2"
    )


def test_enhanced_parquet_assignment_column_cannot_replace_active_mirror(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "enhanced.parquet"
    _write_source_parquet(source)
    store = ClusterAssignmentStore()
    store.add(
        name="Assignment",
        assignments={"n1": 7},
        input_file_ids=["n1", "n2"],
    )
    state = {
        "version": 2,
        "entries": [{"file_id": "n1"}, {"file_id": "n2"}],
        "cluster_assignments": store.to_state(),
    }

    export_enhanced_neuron_parquet(source, output, table_state=state)

    loaded = pd.read_parquet(output)
    assert "cluster_assignment" in loaded.columns
    assert "cluster_assignment_2" in loaded.columns
    assert set(loaded.loc[loaded["file_id"] == "n1", "cluster_assignment"]) == {7}


def test_enhanced_and_filtered_exports_preserve_flatmap_v3_schema_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    enhanced = tmp_path / "enhanced.parquet"
    filtered = tmp_path / "filtered.parquet"
    _write_three_neuron_source_parquet(source)
    original_metadata = _add_flatmap_v3_schema_metadata(source)

    export_enhanced_neuron_parquet(
        source,
        enhanced,
        table_state=_subset_table_state(),
    )
    export_filtered_project_neuron_parquet(
        source,
        filtered,
        table_state=_subset_table_state(),
    )

    enhanced_schema = pq.read_schema(enhanced)
    filtered_schema = pq.read_schema(filtered)
    for key, value in original_metadata.items():
        assert enhanced_schema.metadata[key] == value
        assert filtered_schema.metadata[key] == value
    assert enhanced_schema.field("x").metadata == {b"units": b"micrometer"}
    assert filtered_schema.field("x").metadata == {b"units": b"micrometer"}
    assert pd.read_parquet(filtered)["file_id"].tolist() == ["n3", "n3", "n1", "n1"]


def test_read_enhanced_parquet_metadata_accepts_canonical_parquet(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_source_parquet(source)

    payload = read_enhanced_parquet_metadata(source)

    assert payload["has_project_metadata"] is False
    assert payload["enhanced_columns"] == []
    assert payload["table_state"]["entries"] == []


def test_read_enhanced_parquet_metadata_imports_legacy_cluster_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-clusters.parquet"
    _write_source_parquet(source)
    frame = pd.read_parquet(source)
    frame["cluster_id"] = [4, 4, 9]
    frame.to_parquet(source, index=False)

    payload = read_enhanced_parquet_metadata(source)

    entries = {
        entry["file_id"]: entry["cluster_id"]
        for entry in payload["table_state"]["entries"]
    }
    assert entries == {"n1": 4, "n2": 9}


def test_read_enhanced_parquet_metadata_accepts_pre_rename_namespace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-namespace.parquet"
    _write_source_parquet(source)
    table = pq.read_table(source)
    metadata = dict(table.schema.metadata or {})
    metadata[f"{LEGACY_PROJECT_METADATA_PREFIX}metadata_json".encode()] = (
        json.dumps({"version": "2", "table_state": _table_state()}).encode()
    )
    pq.write_table(table.replace_schema_metadata(metadata), source)

    payload = read_enhanced_parquet_metadata(source)

    assert payload["has_project_metadata"] is True
    assert payload["table_state"] == _table_state()


def test_save_project_bundle_writes_only_current_table_neurons(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_three_neuron_source_parquet(source)

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_subset_table_state(),
        layers=[],
        atlas_name="fake_atlas",
    )

    bundle = load_project_bundle(bundle_path)
    saved = pd.read_parquet(bundle.source_parquet_path)
    assert saved["file_id"].tolist() == ["n3", "n3", "n1", "n1"]
    assert "n2" not in set(saved["file_id"])
    assert saved.loc[saved["file_id"] == "n3", "neuron_label"].unique().tolist() == [
        "target"
    ]
    assert saved.loc[
        saved["file_id"] == "n1", "cluster_assignment"
    ].unique().tolist() == [2]
    assert not list(tmp_path.glob(".saved.nnproj.*.staging"))


def test_filtered_project_export_limits_its_own_connection_and_cleans_spills(
    tmp_path, monkeypatch
):
    source, output = tmp_path / "source.parquet", tmp_path / "filtered.parquet"
    _write_three_neuron_source_parquet(source)
    connect = project_io_module.duckdb.connect
    settings = []

    def track_connection(*args, **kwargs):
        conn = connect(*args, **kwargs)
        settings.append(
            conn.execute(
                "SELECT current_setting('memory_limit'), current_setting('threads'), "
                "current_setting('temp_directory')"
            ).fetchone()
        )
        return conn

    # Resource policy must not mutate a viewer's already-open database.
    viewer_db = connect(config={"memory_limit": "1GiB", "threads": 1})
    try:
        monkeypatch.setattr(project_io_module.duckdb, "connect", track_connection)
        export_filtered_project_neuron_parquet(
            source, output, table_state=_subset_table_state()
        )
        assert len(settings) == 1
        memory_limit, threads, spill_dir = settings[0]
        assert memory_limit == "512.0 MiB"
        assert threads == 2
        assert Path(spill_dir).parent == tmp_path
        assert Path(spill_dir).name.startswith(".nn-project-export-")
        assert not Path(spill_dir).exists()
        assert viewer_db.execute(
            "SELECT current_setting('memory_limit'), current_setting('threads')"
        ).fetchone() == ("1.0 GiB", 1)
    finally:
        viewer_db.close()


def test_filtered_project_export_spills_large_sort_without_losing_order(
    tmp_path, monkeypatch
):
    source, output = tmp_path / "source.parquet", tmp_path / "filtered.parquet"
    profile_path = tmp_path / "export-profile.json"
    connect = project_io_module.duckdb.connect
    count = 500_000
    with connect(config={"memory_limit": "64MiB", "threads": 1}) as builder:
        builder.execute(f"""
            COPY (
                SELECT CASE WHEN i % 2 = 0 THEN 'a/shared' ELSE 'b/shared' END AS file_id,
                    {count} - i AS node_id, 'shared' AS neuron_id,
                    repeat(md5(CAST(i AS VARCHAR)), 4) AS payload
                FROM range({count}) AS rows(i)
            ) TO '{project_io_module._duckdb_path(source)}' (FORMAT PARQUET)
        """)

    def profile_connection(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.execute(
            f"SET profiling_output = '{project_io_module._duckdb_path(profile_path)}'"
        )
        conn.execute("SET enable_profiling = 'json'")
        return conn

    monkeypatch.setattr(project_io_module, "_PROJECT_EXPORT_MEMORY_LIMIT", "64MiB")
    monkeypatch.setattr(project_io_module, "_PROJECT_EXPORT_THREADS", 1)
    monkeypatch.setattr(project_io_module.duckdb, "connect", profile_connection)
    export_filtered_project_neuron_parquet(
        source,
        output,
        table_state={
            "entries": [
                {"file_id": "b/shared", "tags": ["B"]},
                {"file_id": "a/shared", "tags": ["A"]},
            ]
        },
    )
    profile = json.loads(profile_path.read_text())
    assert (
        profile["system_peak_temp_dir_size"] > 0
    )  # Actually exercised external sorting.
    # Profiling can include a transient buffer beyond the configured budget.
    assert profile["system_peak_buffer_memory"] <= 65 * 1024**2
    assert not list(tmp_path.glob(".nn-project-export-*"))
    saved = pq.read_table(output, columns=["file_id", "node_id"]).to_pandas()
    assert len(saved) == count
    assert saved.file_id.iloc[: count // 2].eq("b/shared").all()
    assert saved.file_id.iloc[count // 2 :].eq("a/shared").all()
    assert saved.node_id.iloc[: count // 2].is_monotonic_increasing
    assert saved.node_id.iloc[count // 2 :].is_monotonic_increasing


def test_project_export_memory_error_preserves_previous_bundle(tmp_path, monkeypatch):
    source, bundle_path = tmp_path / "source.parquet", tmp_path / "saved.nnproj"
    _write_three_neuron_source_parquet(source)
    save_project_bundle(
        bundle_path, source_parquet_path=source, table_state=_subset_table_state()
    )
    manifest_before = (bundle_path / "manifest.json").read_bytes()
    data_before = (bundle_path / "data/source_neurons.parquet").read_bytes()
    # Force a real DuckDB allocation failure, instead of exhausting this process.
    monkeypatch.setattr(project_io_module, "_PROJECT_EXPORT_MEMORY_LIMIT", "1KiB")
    with pytest.raises(project_io_module.duckdb.OutOfMemoryException):
        save_project_bundle(
            bundle_path,
            source_parquet_path=source,
            table_state=_subset_table_state(),
            overwrite=True,
        )
    assert (bundle_path / "manifest.json").read_bytes() == manifest_before
    assert (bundle_path / "data/source_neurons.parquet").read_bytes() == data_before
    assert is_recognized_project_bundle(bundle_path)
    assert not list(tmp_path.glob(".saved.nnproj.*.staging"))


def test_save_project_bundle_rejects_existing_destination_without_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_source_parquet(source)
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
    )
    sentinel = bundle_path / "keep.txt"
    sentinel.write_text("original")
    original_manifest = (bundle_path / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        save_project_bundle(
            bundle_path,
            source_parquet_path=source,
            table_state=_table_state(),
        )

    assert sentinel.read_text() == "original"
    assert (bundle_path / "manifest.json").read_bytes() == original_manifest


def test_save_project_bundle_overwrite_replaces_entire_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_source_parquet(source)
    stale_layer = _DummyLayer(
        name="Stale Heatmap",
        data=np.ones((2, 2, 2), dtype=np.float32),
        metadata={"heatmap_source": True, "file_ids": ["n1"]},
    )
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
        layers=[stale_layer],
    )
    (bundle_path / "stale.txt").write_text("remove me")

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
        layers=[],
        overwrite=True,
    )

    loaded = load_project_bundle(bundle_path)
    assert loaded.layers == ()
    assert not (bundle_path / "stale.txt").exists()
    assert not any((bundle_path / "layers").iterdir())
    assert not list(tmp_path.glob(".saved.nnproj.*.rollback"))


def test_save_project_bundle_overwrites_project_using_its_bundled_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_source_parquet(source)
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
    )

    bundled_source = bundle_path / "data" / "source_neurons.parquet"
    save_project_bundle(
        bundle_path,
        source_parquet_path=bundled_source,
        table_state=_table_state(),
        overwrite=True,
    )

    loaded = load_project_bundle(bundle_path)
    assert loaded.source_parquet_path == bundled_source
    assert pd.read_parquet(bundled_source)["file_id"].unique().tolist() == ["n1"]


def test_project_overwrite_rejects_unrecognized_file_and_directory_targets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_source_parquet(source)
    missing = tmp_path / "missing.nnproj"
    file_target = tmp_path / "file.nnproj"
    file_target.write_text("not a project")
    directory_target = tmp_path / "directory.nnproj"
    directory_target.mkdir()
    (directory_target / "keep.txt").write_text("unrelated")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        save_project_bundle(
            missing,
            source_parquet_path=source,
            table_state=_table_state(),
            overwrite=True,
        )
    with pytest.raises(ValueError, match="not a directory"):
        save_project_bundle(
            file_target,
            source_parquet_path=source,
            table_state=_table_state(),
            overwrite=True,
        )
    with pytest.raises(ValueError, match="no readable manifest"):
        save_project_bundle(
            directory_target,
            source_parquet_path=source,
            table_state=_table_state(),
            overwrite=True,
        )

    assert file_target.read_text() == "not a project"
    assert (directory_target / "keep.txt").read_text() == "unrelated"
    assert not is_recognized_project_bundle(directory_target)


def test_project_overwrite_rejects_symlink_target(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    symlink_path = tmp_path / "linked.nnproj"
    _write_source_parquet(source)
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
    )
    try:
        symlink_path.symlink_to(bundle_path, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="must not be a symlink"):
        save_project_bundle(
            symlink_path,
            source_parquet_path=source,
            table_state=_table_state(),
            overwrite=True,
        )

    assert is_recognized_project_bundle(bundle_path)
    assert not is_recognized_project_bundle(symlink_path)


def test_project_overwrite_staging_failure_keeps_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_source_parquet(source)
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
    )
    sentinel = bundle_path / "keep.txt"
    sentinel.write_text("original")

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("injected staging failure")

    monkeypatch.setattr(
        project_io_module,
        "export_filtered_project_neuron_parquet",
        fail_export,
    )
    with pytest.raises(RuntimeError, match="injected staging failure"):
        save_project_bundle(
            bundle_path,
            source_parquet_path=source,
            table_state=_table_state(),
            overwrite=True,
        )

    assert sentinel.read_text() == "original"
    assert is_recognized_project_bundle(bundle_path)
    assert not list(tmp_path.glob(".saved.nnproj.*.staging"))


def test_project_overwrite_publication_failure_restores_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_source_parquet(source)
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
    )
    sentinel = bundle_path / "keep.txt"
    sentinel.write_text("original")
    real_rename = Path.rename

    def fail_staged_publication(path: Path, target: str | Path) -> Path:
        if path.name.endswith(".staging") and Path(target) == bundle_path:
            raise OSError("injected publication failure")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staged_publication)
    with pytest.raises(OSError, match="injected publication failure"):
        save_project_bundle(
            bundle_path,
            source_parquet_path=source,
            table_state=_table_state(),
            overwrite=True,
        )

    assert sentinel.read_text() == "original"
    assert is_recognized_project_bundle(bundle_path)
    assert not list(tmp_path.glob(".saved.nnproj.*.staging"))
    assert not list(tmp_path.glob(".saved.nnproj.*.rollback"))


def test_load_project_bundle_accepts_version_one_manifest_and_table_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "legacy.nnproj"
    _write_source_parquet(source)
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
        layers=[],
    )
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = "1"
    manifest_path.write_text(json.dumps(manifest))

    bundle = load_project_bundle(bundle_path)

    assert bundle.manifest["version"] == "1"
    assert bundle.table_state["version"] == "1"
    assert bundle.table_state["entries"][0]["cluster_id"] == 3
    assert bundle.comparison_board is None


def test_load_and_overwrite_accept_pre_rename_project_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "legacy.swcv"
    _write_source_parquet(source)
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
        layers=[],
    )
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format"] = LEGACY_PROJECT_BUNDLE_FORMAT
    manifest_path.write_text(json.dumps(manifest))

    assert is_recognized_project_bundle(bundle_path)
    assert load_project_bundle(bundle_path).manifest["format"] == (
        LEGACY_PROJECT_BUNDLE_FORMAT
    )

    save_project_bundle(
        bundle_path,
        source_parquet_path=bundle_path / "data" / "source_neurons.parquet",
        table_state=_table_state(),
        layers=[],
        overwrite=True,
    )
    assert load_project_bundle(bundle_path).manifest["format"] == PROJECT_BUNDLE_FORMAT


def test_project_bundle_round_trips_comparison_board_and_heatmap_source_ids(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "comparison.nnproj"
    _write_source_parquet(source)
    heatmap = _DummyLayer(
        name="Cluster 3 Heatmap",
        data=np.ones((2, 3, 4), dtype=np.float32),
        metadata={
            "heatmap_source": True,
            "heatmap_kind": "analysis",
            "comparison_source_id": "heatmap-source-3",
            "comparison_assignment_id": "assignment-a",
            "comparison_assignment_name": "Run A",
            "comparison_filter_signature": {
                "atlas_name": "allen_mouse_25um",
                "region_ids": [1, 2],
                "node_types": [2],
                "soma_radius_um": None,
                "depth_axis": 0,
                "depth_bin_factor": 1,
            },
            "heatmap_cluster": 3,
            "source_file_ids": ["n1"],
        },
    )
    board = {
        "version": 1,
        "rows": 1,
        "columns": 2,
        "reference_assignment_id": "assignment-a",
        "shared_intensity": True,
        "cells": [
            {
                "cell_id": "cell-1",
                "title": "Run A cluster heatmap",
                "source_kind": "ccf_heatmap",
                "assignment_id": "assignment-a",
                "comparison_source_ids": ["heatmap-source-3"],
                "ccf_plane": "coronal",
                "reduction": "projection",
            }
        ],
    }

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
        layers=[heatmap],
        atlas_name="allen_mouse_25um",
        comparison_board=board,
    )

    loaded = load_project_bundle(bundle_path)
    assert loaded.comparison_board == board
    assert loaded.layers[0].metadata["metadata"]["comparison_source_id"] == (
        "heatmap-source-3"
    )
    assert loaded.layers[0].metadata["metadata"]["comparison_assignment_id"] == (
        "assignment-a"
    )
    assert loaded.layers[0].metadata["metadata"]["comparison_filter_signature"] == {
        "atlas_name": "allen_mouse_25um",
        "depth_axis": 0,
        "depth_bin_factor": 1,
        "node_types": [2],
        "region_ids": [1, 2],
        "soma_radius_um": None,
    }


def test_project_bundle_filters_v2_assignments_to_current_membership(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_three_neuron_source_parquet(source)
    state = _multi_assignment_table_state()
    state["entries"] = state["entries"][:2]

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=state,
        layers=[],
    )

    bundle = load_project_bundle(bundle_path)
    registry = bundle.table_state["cluster_assignments"]
    assert bundle.manifest["version"] == "2"
    assert [item["name"] for item in registry["sets"]] == [
        "Soma Location 1",
        "Voxel Correlation 1",
    ]
    for assignment in registry["sets"]:
        assert set(assignment["assignments"]) <= {"n1", "n2"}
        assert set(assignment["input_file_ids"]) <= {"n1", "n2"}


def test_project_bundle_keeps_collision_resolved_assignment_column_stable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    reexported = tmp_path / "reexported.parquet"
    _write_three_neuron_source_parquet(source)
    frame = pd.read_parquet(source)
    frame["cluster_soma_location_1"] = 99
    frame.to_parquet(source, index=False)

    state = _multi_assignment_table_state()
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=state,
        layers=[],
    )
    bundle = load_project_bundle(bundle_path)
    first_set = bundle.table_state["cluster_assignments"]["sets"][0]
    assert first_set["column_name"] == "cluster_soma_location_1_2"

    export_enhanced_neuron_parquet(
        bundle.source_parquet_path,
        reexported,
        table_state=bundle.table_state,
    )
    loaded = pd.read_parquet(reexported)
    assert set(loaded["cluster_soma_location_1"]) == {99}
    assert "cluster_soma_location_1_2" in loaded.columns
    assert "cluster_soma_location_1_2_2" not in loaded.columns


def test_save_project_bundle_replaces_existing_enhanced_columns(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_three_neuron_source_parquet(source)
    source_df = pd.read_parquet(source)
    source_df["neuron_label"] = "old"
    source_df["neuron_group"] = "old_group"
    source_df["neuron_tags_json"] = '["old"]'
    source_df["neuron_notes"] = "old notes"
    source_df["cluster_assignment"] = 99
    source_df.to_parquet(source, index=False)

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_subset_table_state(),
        layers=[],
    )

    schema = pq.read_schema(bundle_path / "data" / "source_neurons.parquet")
    for column in ENHANCED_NEURON_COLUMNS:
        assert schema.names.count(column) == 1

    saved = pd.read_parquet(bundle_path / "data" / "source_neurons.parquet")
    assert set(saved["neuron_label"]) == {"source", "target"}
    assert set(saved["neuron_group"]) == {"CTX", "GPe"}
    assert set(saved["cluster_assignment"]) == {2, 7}


def test_save_project_bundle_manifest_records_filtered_source_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_three_neuron_source_parquet(source)

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_subset_table_state(),
        layers=[],
    )

    bundle = load_project_bundle(bundle_path)
    source_info = bundle.manifest["source_parquet"]
    assert source_info["path"] == "data/source_neurons.parquet"
    assert source_info["is_filtered_subset"] is True
    assert source_info["filter"] == {
        "type": "current_table_file_ids",
        "file_id_count": 2,
    }
    assert source_info["original_path"] == str(source)
    assert source_info["original_size_bytes"] == source.stat().st_size
    assert source_info["original_mtime_ns"] == source.stat().st_mtime_ns
    assert "sha256" in source_info


def test_project_bundle_references_external_flatmap_cache_without_copying_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    cache_path = tmp_path / "shared-flatmap-cache"
    cache_path.mkdir()
    (cache_path / "flatmap-region-cache.json").write_text("{}\n")
    _write_three_neuron_source_parquet(source)

    reference = {
        "path": str(cache_path),
        "profile_id": "lookup-test__atlas-test__256x25",
    }
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_subset_table_state(),
        layers=[],
        flatmap_cache_reference=reference,
    )

    bundle = load_project_bundle(bundle_path)
    assert bundle.manifest["flatmap_cache"] == reference
    assert bundle.flatmap_cache_reference == reference
    assert not (bundle_path / "flatmap-region-cache.json").exists()


def test_project_bundle_round_trips_region_appearance(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_three_neuron_source_parquet(source)
    appearance = {
        "format": "napari_neuron_navigator.region_palette",
        "version": 1,
        "atlas": {"name": "allen_mouse_25um", "version": "1.2"},
        "overrides": {
            "101": {
                "color_mode": "custom",
                "color_rgb": [0.1, 0.2, 0.3],
                "fill_visible": False,
            }
        },
    }

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_subset_table_state(),
        layers=[],
        region_appearance=appearance,
    )

    bundle = load_project_bundle(bundle_path)
    assert bundle.manifest["region_appearance"] == appearance
    assert bundle.region_appearance == appearance


def test_save_project_bundle_rejects_empty_table(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "empty.nnproj"
    _write_source_parquet(source)

    with pytest.raises(ValueError, match="at least one neuron"):
        save_project_bundle(
            bundle_path,
            source_parquet_path=source,
            table_state={"version": 1, "entries": []},
            layers=[],
        )

    assert not (bundle_path / "data" / "source_neurons.parquet").exists()


def test_save_and_load_project_bundle_preserves_mask_array_and_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.nnproj"
    _write_source_parquet(source)
    progress_events: list[tuple[str, int, int]] = []

    heatmap = _DummyLayer(
        name="alpha Heatmap",
        data=np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        metadata={
            "heatmap_source": True,
            "heatmap_kind": "selected_neurons",
            "file_ids": ["n1"],
            "source_path": str(source),
            "color": [1.0, 0.0, 0.0, 1.0],
            "atlas_name": "fake_atlas",
        },
        colormap=_DummyColormap(
            "alpha_red",
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
        ),
    )
    heatmap.gamma = 0.2
    mask = _DummyLayer(
        name="Mask: alpha Heatmap",
        data=np.array([[[0, 1], [1, 0]]], dtype=np.uint8),
        metadata={
            "mask_query_source": True,
            "source_heatmap_layers": ["alpha Heatmap"],
            "query_excluded_file_ids": ["n1", "n2", "n1"],
            "lower_threshold": 0.5,
            "upper_threshold": 2.0,
            "merge_mode": "separate",
            "atlas_name": "fake_atlas",
        },
    )
    ignored = _DummyLayer(
        name="Neuron Lines",
        data=np.zeros((1, 2, 3), dtype=float),
        metadata={"file_ids": ["n1"]},
    )

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
        layers=[heatmap, mask, ignored],
        atlas_name="fake_atlas",
        progress_callback=lambda message, current, total: progress_events.append(
            (message, current, total)
        ),
    )

    assert progress_events[0] == ("Preparing project bundle...", 0, 6)
    assert progress_events[-1] == ("Done", 6, 6)
    assert any(
        "Saving layer 1/2: alpha Heatmap" in event[0] for event in progress_events
    )
    assert any(
        "Saving layer 2/2: Mask: alpha Heatmap" in event[0] for event in progress_events
    )

    bundle = load_project_bundle(bundle_path)
    assert bundle.manifest["format"] == PROJECT_BUNDLE_FORMAT
    assert bundle.source_parquet_path.exists()
    assert bundle.table_state["entries"][0]["file_id"] == "n1"
    assert len(bundle.layers) == 2
    restored_heatmap = next(
        layer for layer in bundle.layers if layer.metadata["layer_type"] == "image"
    )
    colormap = restored_heatmap.metadata["display"]["colormap"]
    assert colormap["name"] == "alpha_red"
    assert colormap["colors"] == [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
    assert restored_heatmap.metadata["display"]["gamma"] == pytest.approx(0.2)
    restored_mask = next(
        layer for layer in bundle.layers if layer.metadata["layer_type"] == "labels"
    )
    np.testing.assert_array_equal(restored_mask.data, mask.data)
    source_sets = restored_mask.metadata["source_neuron_sets"]
    assert source_sets[0]["count"] == 2
    assert source_sets[0]["file_ids"] == ["n1", "n2"]
    assert source_sets[0]["derivation"]["lower_threshold"] == 0.5

    bundled_schema = pq.read_schema(bundle.source_parquet_path)
    assert "neuron_label" in bundled_schema.names


def test_project_colormap_payload_restores_image_colormap_kwargs() -> None:
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    kwargs: dict[str, object] = {}
    restored = NeuronViewerWidget._apply_project_colormap_kwargs(
        SimpleNamespace(),
        kwargs,
        {
            "colormap": {
                "type": "colormap",
                "name": "alpha_red",
                "colors": [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
                "controls": [0.0, 1.0],
                "interpolation": "linear",
                "nan_color": [0.0, 0.0, 0.0, 0.0],
            }
        },
    )

    assert restored is True
    assert kwargs["colormap"].name == "alpha_red"
    np.testing.assert_allclose(
        np.asarray(kwargs["colormap"].colors),
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
    )


def test_project_colormap_payload_restores_labels_colormap_kwargs() -> None:
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    kwargs: dict[str, object] = {}
    restored = NeuronViewerWidget._apply_project_colormap_kwargs(
        SimpleNamespace(_project_label_key=NeuronViewerWidget._project_label_key),
        kwargs,
        {
            "colormap": {
                "type": "direct_label_colormap",
                "color_dict": [
                    {
                        "label": {"type": "none", "value": None},
                        "color": [0.0, 0.0, 0.0, 0.0],
                    },
                    {
                        "label": {"type": "int", "value": 1},
                        "color": [0.2, 0.4, 0.6, 1.0],
                    },
                ],
            }
        },
    )

    assert restored is True
    np.testing.assert_allclose(
        kwargs["colormap"].color_dict[1],
        [0.2, 0.4, 0.6, 1.0],
    )


def test_project_image_restore_applies_saved_gamma() -> None:
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    viewer = SimpleNamespace(add_image=MagicMock())
    widget = SimpleNamespace(
        viewer=viewer,
        _apply_project_colormap_kwargs=MagicMock(return_value=False),
    )
    volume = np.ones((2, 2, 2), dtype=np.float32)

    NeuronViewerWidget._restore_project_image_layer(
        widget,
        volume,
        {"name": "alpha Heatmap"},
        {"gamma": 0.2},
        {},
    )

    viewer.add_image.assert_called_once_with(
        volume,
        name="alpha Heatmap",
        gamma=0.2,
    )


def test_current_project_path_updates_save_control(tmp_path: Path) -> None:
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    save_button = MagicMock()
    widget = SimpleNamespace(_save_project_btn=save_button)
    project_path = tmp_path / "saved.nnproj"

    NeuronViewerWidget._set_current_project_path(widget, project_path)

    assert widget._current_project_path == project_path.resolve()
    save_button.setEnabled.assert_called_with(True)
    assert str(project_path.resolve()) in save_button.setToolTip.call_args.args[0]

    NeuronViewerWidget._set_current_project_path(widget, None)

    assert widget._current_project_path is None
    save_button.setEnabled.assert_called_with(False)


def test_loading_plain_parquet_clears_current_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    source = tmp_path / "source.parquet"
    new_db = MagicMock()
    new_db.get_statistics.return_value = {
        "n_nodes": 3,
        "n_files": 2,
        "n_subjects": 2,
        "n_regions": 2,
    }
    old_db = MagicMock()
    flatmap_tab = SimpleNamespace(
        invalidate_loaded_parquet_projection=MagicMock(),
        refresh_cache_profiles=MagicMock(),
    )
    widget = SimpleNamespace(
        _db=old_db,
        _file_label=MagicMock(),
        _stats_label=MagicMock(),
        _regions_status_label=MagicMock(),
        _analysis_tab=SimpleNamespace(set_database=MagicMock()),
        _termini_section_widget=SimpleNamespace(set_database=MagicMock()),
        _flatmap_tab=flatmap_tab,
        _set_current_project_path=MagicMock(),
        _load_enhanced_table_state=MagicMock(return_value=0),
        _load_flatmap_transform_status=MagicMock(return_value=""),
        _set_region_query_buttons_enabled=MagicMock(),
        _apply_saved_table_state_to_table=MagicMock(),
    )
    monkeypatch.setitem(
        NeuronViewerWidget._load_parquet_path.__globals__,
        "NeuronDatabase",
        lambda _path: new_db,
    )

    NeuronViewerWidget._load_parquet_path(widget, source)

    assert widget._db is new_db
    old_db.close.assert_called_once_with()
    widget._set_current_project_path.assert_called_once_with(None)


def test_save_current_project_cancel_does_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    project_path = tmp_path / "saved.nnproj"
    widget = SimpleNamespace(
        _current_project_path=project_path,
        _confirm_project_overwrite=MagicMock(return_value=False),
        _save_project_to_path=MagicMock(),
    )
    monkeypatch.setitem(
        NeuronViewerWidget._save_current_project.__globals__,
        "is_recognized_project_bundle",
        lambda _path: True,
    )

    NeuronViewerWidget._save_current_project(widget)

    widget._confirm_project_overwrite.assert_called_once_with(project_path)
    widget._save_project_to_path.assert_not_called()


def test_save_current_project_confirms_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    project_path = tmp_path / "saved.nnproj"
    widget = SimpleNamespace(
        _current_project_path=project_path,
        _confirm_project_overwrite=MagicMock(return_value=True),
        _save_project_to_path=MagicMock(),
    )
    monkeypatch.setitem(
        NeuronViewerWidget._save_current_project.__globals__,
        "is_recognized_project_bundle",
        lambda _path: True,
    )

    NeuronViewerWidget._save_current_project(widget)

    widget._save_project_to_path.assert_called_once_with(
        project_path,
        overwrite=True,
    )


def _project_save_widget(source: Path, current: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        _db=SimpleNamespace(parquet_path=source),
        _current_project_path=current,
        _project_progress=MagicMock(),
        _project_status_label=MagicMock(),
        _regions_status_label=MagicMock(),
        _flatmap_tab=SimpleNamespace(active_cache_reference=lambda: None),
        _set_project_save_in_progress=MagicMock(),
        _set_current_project_path=MagicMock(),
        _current_table_state=lambda: _table_state(),
        _iter_viewer_layers=lambda: [],
        _current_atlas_name=lambda: None,
        _analysis_project_metadata=lambda: {},
    )


def test_successful_save_as_sets_current_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    source = tmp_path / "source.parquet"
    project_path = tmp_path / "saved.nnproj"
    widget = _project_save_widget(source)
    method_globals = NeuronViewerWidget._save_project_to_path.__globals__
    save_calls: list[dict[str, object]] = []

    def fake_save(path, **kwargs):
        save_calls.append({"path": path, **kwargs})
        return project_path

    monkeypatch.setitem(method_globals, "save_project_bundle", fake_save)
    monkeypatch.setitem(
        method_globals,
        "QApplication",
        SimpleNamespace(processEvents=lambda: None),
    )
    monkeypatch.setitem(method_globals, "show_info", lambda _message: None)

    NeuronViewerWidget._save_project_to_path(
        widget,
        project_path,
        overwrite=False,
    )

    assert save_calls[0]["path"] == project_path
    assert save_calls[0]["overwrite"] is False
    widget._set_current_project_path.assert_called_once_with(project_path)
    assert widget._set_project_save_in_progress.call_args_list == [
        call(True),
        call(False),
    ]


def test_failed_overwrite_retains_recognized_current_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    source = tmp_path / "source.parquet"
    project_path = tmp_path / "saved.nnproj"
    widget = _project_save_widget(source, current=project_path)
    method_globals = NeuronViewerWidget._save_project_to_path.__globals__

    def fail_save(*_args, **_kwargs):
        raise OSError("save failed")

    monkeypatch.setitem(method_globals, "save_project_bundle", fail_save)
    monkeypatch.setitem(
        method_globals,
        "is_recognized_project_bundle",
        lambda _path: True,
    )
    monkeypatch.setitem(
        method_globals,
        "QApplication",
        SimpleNamespace(processEvents=lambda: None),
    )
    monkeypatch.setitem(method_globals, "show_warning", lambda _message: None)

    NeuronViewerWidget._save_project_to_path(
        widget,
        project_path,
        overwrite=True,
    )

    widget._set_current_project_path.assert_not_called()
    assert widget._set_project_save_in_progress.call_args_list == [
        call(True),
        call(False),
    ]
    assert "save failed" in widget._project_status_label.setText.call_args.args[0]
