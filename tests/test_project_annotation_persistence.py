"""Exercise annotation persistence through real project and viewer state paths."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.neuron_sets import AnnotationRule
from napari_neuron_navigator.project_io import (
    export_enhanced_neuron_parquet,
    load_project_bundle,
    read_enhanced_parquet_metadata,
    save_project_bundle,
)

FILE_IDS = ("subject-a/shared", "subject-b/shared", "subject-c/shared", "d/empty")
COMMA_TAG = "Region query: TH, CP"


@pytest.fixture
def project_source(tmp_path):
    path = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "file_id": [*FILE_IDS, "outside"],
            "subject": ["subject-a", "subject-b", "subject-c", "d", "outside"],
            "neuron_id": ["shared", "shared", "shared", "empty", "outside"],
            "node_id": [10] * 5,
            "parent_id": [-1] * 5,
            "type": [1] * 5,
            "x": [1.0] * 5,
            "y": [2.0] * 5,
            "z": [3.0] * 5,
            "radius": [1.0] * 5,
            "region_id": [1] * 5,
            "region_name": ["Test"] * 5,
            "region_acronym": ["TEST"] * 5,
        }
    ).to_parquet(path, index=False)
    return path


@pytest.fixture
def owner_factory(qt_app, monkeypatch):
    from napari.components import ViewerModel

    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    # Atlas rendering is unrelated to the table/project persistence path.
    monkeypatch.setattr(
        NeuronViewerWidget, "_start_cached_template_autoload", lambda self: None
    )
    owners = []

    def create():
        owner = NeuronViewerWidget(ViewerModel())
        owners.append(owner)
        return owner

    yield create

    for owner in reversed(owners):
        owner._on_neuron_viewer_destroyed()
        owner.viewer.layers.clear()
        if owner._db is not None:
            owner._db.close()
        owner.close()
        owner.deleteLater()
    qt_app.processEvents()


def _annotate(owner, source):
    from napari_neuron_navigator.widgets.neuron_table import NeuronMetadataUpdate

    owner._load_parquet_path(source)
    owner._populate_neuron_table(pd.DataFrame({"file_id": FILE_IDS}))
    table = owner._neuron_table
    table.apply_metadata_updates(
        {
            FILE_IDS[0]: NeuronMetadataUpdate(
                label="Target", tags=("TH", "CP", COMMA_TAG)
            ),
            FILE_IDS[1]: NeuronMetadataUpdate(label="Target", tags=("TH",)),
            FILE_IDS[2]: NeuronMetadataUpdate(label="Other", tags=("CP",)),
        }
    )
    rules = {
        "TH": AnnotationRule(tag_filter=("TH",)),
        "CP": AnnotationRule(tag_filter=("CP",)),
        "TH without CP": AnnotationRule(tag_filter=("TH",), excluded_tags=("CP",)),
        "Exact TH": AnnotationRule(tag_filter=("TH",), tag_match="exact"),
        "Target": AnnotationRule(label_filter="Target"),
        "Empty": AnnotationRule(tag_filter=("missing",)),
        "Comma": AnnotationRule(tag_filter=(COMMA_TAG,)),
    }
    for name, rule in rules.items():
        table.neuron_sets.save(name, rule)
    table._on_sets_changed()
    return table


def _annotations(table):
    return {
        entry["file_id"]: (entry["label"], entry["tags"])
        for entry in table.export_state()["entries"]
    }


def _members(table):
    return {
        item.name: item.rule.members(table._entries)
        for item in table.neuron_sets.sets()
    }


def _assert_sets(owner, expected):
    table = owner._neuron_table
    assert _members(table) == expected
    panel = table._sets_panel.summary_table
    counts = {
        panel.item(row, 0).text(): int(panel.item(row, 1).text())
        for row in range(panel.rowCount())
    }
    assert counts == {name: len(ids) for name, ids in expected.items()}
    th = next(item for item in table.neuron_sets.sets() if item.name == "TH")
    cp = next(item for item in table.neuron_sets.sets() if item.name == "CP")
    table.show_set_members(th.set_id)
    assert set(table.get_selected_file_ids()) == set(expected["TH"])
    overlap = owner._set_overlap
    for combo, item in zip(overlap.set_combos, (th, cp), strict=True):
        combo.setCurrentIndex(combo.findData(item.set_id))
    overlap.compare()
    assert [overlap.results.item(row, 1).text() for row in range(4)] == [
        "2",
        "2",
        "1",
        "3",
    ]


@pytest.mark.parametrize("with_layers", [False, True])
@pytest.mark.parametrize("existing_session", [False, True])
def test_project_annotations_and_live_members_survive_viewer_roundtrip(
    owner_factory, project_source, tmp_path, qt_app, existing_session, with_layers
):
    owner = owner_factory()
    table = _annotate(owner, project_source)
    before = _annotations(table)
    definitions = table.neuron_sets.to_state()
    members = _members(table)
    snapshots = {}
    if with_layers:
        from napari.utils import DirectLabelColormap

        th = table.neuron_sets.sets()[0]
        snapshot = {**th.to_state(), "file_ids": list(members["TH"])}
        for name, kind in (
            ("TH Heatmap", "heatmap_source"),
            ("TH Mask", "mask_query_source"),
        ):
            metadata = {
                kind: True,
                "neuron_set": deepcopy(snapshot),
                "source_file_ids": list(members["TH"]),
                "sigma": 1.0,
                "lower_threshold": 0.25,
            }
            snapshots[name] = metadata
            kwargs = {"name": name, "metadata": metadata}
            if kind == "heatmap_source":
                add = owner.viewer.add_image
            else:
                add = owner.viewer.add_labels
                kwargs["colormap"] = DirectLabelColormap(
                    color_dict={None: (0, 0, 0, 0), 0: (0, 0, 0, 0), 1: (1, 0, 0, 1)}
                )
            add(np.ones((2, 2, 2), dtype=np.uint8), **kwargs)
    # Saving while filtered/selected must still include the whole Data cohort.
    table.set_annotation_rule(AnnotationRule(tag_filter=("TH",)))
    table.select_file_ids([FILE_IDS[0]])
    path = tmp_path / "saved.nnproj"
    owner._save_project_to_path(path, overwrite=False)
    assert owner._current_project_path == path.resolve(), (
        owner._project_status_label.text()
    )
    saved = json.loads((path / "data/table_state.json").read_text())
    assert {
        row["file_id"]: (row["label"], row["tags"]) for row in saved["entries"]
    } == before
    embedded = read_enhanced_parquet_metadata(path / "data/source_neurons.parquet")
    assert embedded["table_state"] == saved
    columns = (
        pd.read_parquet(path / "data/source_neurons.parquet")
        .fillna("")
        .set_index("file_id")
    )
    for file_id, (label, tags) in before.items():
        assert (columns.loc[file_id, "neuron_label"] or "") == label
        assert json.loads(columns.loc[file_id, "neuron_tags_json"] or "[]") == tags

    restored = owner_factory()
    if existing_session:
        _annotate(restored, project_source)
        restored._neuron_table.clear()
        restored._neuron_table.populate(["outside", FILE_IDS[0]])
    restored._restore_project_bundle(load_project_bundle(path))
    qt_app.processEvents()
    assert _annotations(restored._neuron_table) == before
    assert restored._neuron_table.neuron_sets.to_state() == definitions
    _assert_sets(restored, members)
    for name, metadata in snapshots.items():
        assert restored.viewer.layers[name].metadata == metadata
    # Exercise the actual first table refresh after project loading.
    restored._populate_neuron_table(pd.DataFrame({"file_id": FILE_IDS}))
    qt_app.processEvents()
    assert _annotations(restored._neuron_table) == before
    _assert_sets(restored, members)

    # Overwrite using the bundled source, then Save As and reopen again.
    for destination, overwrite in ((path, True), (tmp_path / "copy.nnproj", False)):
        restored._save_project_to_path(destination, overwrite=overwrite)
        assert restored._current_project_path == destination.resolve()
        fresh = owner_factory()
        fresh._restore_project_bundle(load_project_bundle(destination))
        qt_app.processEvents()
        assert _annotations(fresh._neuron_table) == before
        assert fresh._neuron_table.neuron_sets.to_state() == definitions
        _assert_sets(fresh, members)
        for name, metadata in snapshots.items():
            assert fresh.viewer.layers[name].metadata == metadata
        restored = fresh


@pytest.mark.parametrize("operation", ["append", "retain", "annotate-search"])
def test_post_load_edits_survive_table_membership_operations(
    owner_factory, project_source, tmp_path, qt_app, operation
):
    from napari_neuron_navigator.analysis.search import SearchAnnotationRequest
    from napari_neuron_navigator.widgets.neuron_table import NeuronMetadataUpdate

    owner = owner_factory()
    _annotate(owner, project_source)
    path = tmp_path / "saved.nnproj"
    owner._save_project_to_path(path, overwrite=False)
    owner._restore_project_bundle(load_project_bundle(path))
    table = owner._neuron_table
    table.apply_metadata_updates(
        {
            FILE_IDS[1]: NeuronMetadataUpdate(
                label="Edited", remove_tags=("TH",), tags=("new",)
            )
        }
    )
    before = _annotations(table)
    if operation == "append":
        owner._add_search_file_ids_to_table([FILE_IDS[0]])
    elif operation == "retain":
        owner._populate_neuron_table(
            pd.DataFrame({"file_id": FILE_IDS}), preserve_existing=True
        )
    else:
        owner._annotate_search_in_table(
            SearchAnnotationRequest(
                reference_file_ids=(FILE_IDS[0],),
                ranked_hits=(),
                filter_tags=("Search test",),
            )
        )
        before[FILE_IDS[0]][1].append("Search test")
    qt_app.processEvents()
    assert _annotations(table) == before
    assert _members(table)["TH"] == (FILE_IDS[0],)

    # The new annotations, including removals, must become the next snapshot.
    owner._save_project_to_path(path, overwrite=True)
    restored = owner_factory()
    restored._restore_project_bundle(load_project_bundle(path))
    qt_app.processEvents()
    assert _annotations(restored._neuron_table) == before
    assert _members(restored._neuron_table)["TH"] == (FILE_IDS[0],)


def test_append_restores_only_new_saved_rows_and_keeps_edited_assignments(
    owner_factory, project_source, tmp_path, qt_app
):
    from napari_neuron_navigator.widgets.neuron_table import NeuronMetadataUpdate

    owner = owner_factory()
    table = _annotate(owner, project_source)
    assignment = owner._cluster_assignment_store.add(
        name="Original",
        assignments=dict.fromkeys(FILE_IDS, 1),
        input_file_ids=FILE_IDS,
        activate=True,
    )
    path = tmp_path / "saved.nnproj"
    owner._save_project_to_path(path, overwrite=False)
    owner._restore_project_bundle(load_project_bundle(path))
    owner._cluster_assignment_store.rename(
        assignment.assignment_id, "Edited assignment"
    )
    current_assignment = owner._cluster_assignment_store.active
    table.retain_file_ids(FILE_IDS[:2])
    table.apply_metadata_updates({FILE_IDS[1]: NeuronMetadataUpdate(tags=("edited",))})
    current = table._entries[FILE_IDS[1]].to_state()
    owner._add_search_file_ids_to_table([FILE_IDS[1], FILE_IDS[2]])
    qt_app.processEvents()
    assert table._entries[FILE_IDS[1]].to_state() == current
    assert table._entries[FILE_IDS[2]].tags == ("CP",)
    assert owner._cluster_assignment_store.active.name == "Edited assignment"
    assert owner._cluster_assignment_store.active is current_assignment
    assert set(table.file_ids()) == set(FILE_IDS[:3])


def test_enhanced_parquet_initial_population_preserves_exact_tags_and_sets(
    owner_factory, project_source, tmp_path, qt_app
):
    owner = owner_factory()
    table = _annotate(owner, project_source)
    before, members = _annotations(table), _members(table)
    output = tmp_path / "enhanced.parquet"
    export_enhanced_neuron_parquet(
        project_source, output, table_state=table.export_state()
    )
    restored = owner_factory()
    restored._load_parquet_path(output)
    restored._populate_neuron_table(pd.DataFrame({"file_id": FILE_IDS}))
    qt_app.processEvents()
    assert _annotations(restored._neuron_table) == before
    assert restored._neuron_table.neuron_sets.to_state() == table.neuron_sets.to_state()
    _assert_sets(restored, members)


def test_legacy_project_without_annotations_clears_previous_definitions(
    owner_factory, project_source, tmp_path, qt_app
):
    path = tmp_path / "legacy.nnproj"
    save_project_bundle(
        path,
        source_parquet_path=project_source,
        table_state={"version": 1, "entries": [{"file_id": FILE_IDS[0]}]},
    )
    owner = owner_factory()
    _annotate(owner, project_source)
    owner._restore_project_bundle(load_project_bundle(path))
    qt_app.processEvents()
    assert _annotations(owner._neuron_table) == {FILE_IDS[0]: ("", [])}
    assert owner._neuron_table.neuron_sets.sets() == ()


@pytest.mark.parametrize("active_window", [False, True])
def test_save_as_commits_the_active_tag_editor(
    owner_factory, project_source, tmp_path, qt_app, monkeypatch, active_window
):
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QFileDialog, QLineEdit

    from napari_neuron_navigator.widgets.neuron_table import COL_TAGS

    owner = owner_factory()
    table = _annotate(owner, project_source)
    table.show_metadata_columns()
    owner.show()
    qt_app.processEvents()
    qt_app.setActiveWindow(owner if active_window else None)
    row = table._file_id_to_row(FILE_IDS[1])
    item = table._table.item(row, COL_TAGS)
    table._table.setCurrentItem(item)
    table._table.editItem(item)
    editor = table._table.findChild(QLineEdit)
    assert editor is not None
    editor.setText("CP, edited")
    # Do not press Enter or otherwise commit before clicking Save Project As.
    path = tmp_path / "active-edit.nnproj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    QTest.mouseClick(owner._save_project_as_btn, Qt.LeftButton)
    qt_app.processEvents()
    saved = load_project_bundle(path)
    row = next(
        row for row in saved.table_state["entries"] if row["file_id"] == FILE_IDS[1]
    )
    assert row["tags"] == ["CP", "edited"]


def test_export_memory_failure_leaves_viewer_and_saved_project_usable(
    owner_factory, project_source, tmp_path, qt_app, monkeypatch
):
    from napari_neuron_navigator import project_io

    owner = owner_factory()
    table = _annotate(owner, project_source)
    path = tmp_path / "saved.nnproj"
    owner._save_project_to_path(path, overwrite=False)
    before = (path / "manifest.json").read_bytes()
    annotations, members = _annotations(table), _members(table)
    budget = project_io._PROJECT_EXPORT_MEMORY_LIMIT
    monkeypatch.setattr(project_io, "_PROJECT_EXPORT_MEMORY_LIMIT", "1KiB")
    owner._save_project_to_path(path, overwrite=True)
    qt_app.processEvents()
    assert "Out of Memory" in owner._project_status_label.text()
    assert owner._save_project_btn.isEnabled()
    assert owner._save_project_as_btn.isEnabled()
    assert not owner._project_progress.isVisible()
    assert owner._current_project_path == path.resolve()
    assert (path / "manifest.json").read_bytes() == before
    assert _annotations(table) == annotations
    _assert_sets(owner, members)
    # Retry through the same live widget after the allocation failure.
    monkeypatch.setattr(project_io, "_PROJECT_EXPORT_MEMORY_LIMIT", budget)
    owner._save_project_to_path(tmp_path / "retry.nnproj", overwrite=False)
    assert owner._current_project_path == (tmp_path / "retry.nnproj").resolve()
