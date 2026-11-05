"""Regional clustering Qt controls, cancellation, persistence and report handoff."""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from napari_neuron_navigator.analysis.region_profile import build_region_profile
from napari_neuron_navigator.analysis.regional_clustering import (
    RegionalClusteringSettings,
    RegionalClusterResult,
    build_regional_features,
)
from tests.test_region_profile_stage1 import _write_source
from tests.test_region_profile_stage1 import (
    profile_atlas as profile_atlas,  # noqa: PLC0414
)
from tests.test_region_report import report_context as report_context  # noqa: PLC0414


@pytest.fixture
def regional_widget(qt_app, report_context):
    module = importlib.import_module(
        "napari_neuron_navigator.widgets.regional_clustering"
    )
    connection, structures, ids = report_context
    cohorts = {"table": list(ids), "selected": [ids[0]]}
    annotations = {}
    widget = module.RegionalClusteringWidget(
        lambda scope: cohorts[scope],
        lambda: (10,),
        lambda file_ids, field: {key: annotations.get(key, "") for key in file_ids},
    )
    widget.set_context(
        "/private/tmp/regional-fixture.parquet",
        SimpleNamespace(structures=structures, atlas_name="fixture"),
    )
    settings = RegionalClusteringSettings(ids, n_clusters=2)
    features = build_regional_features(
        connection, structures=structures, settings=settings
    )
    # Feature/distance correctness is tested in the headless algorithm suite.
    # These fixed assignments isolate real Qt behavior from subprocess timing.
    result = RegionalClusterResult(
        features,
        features.retained_file_ids,
        features.matrix[features.retained_mask],
        np.array([1, 2]),
        np.array([[0, 1, 0.5, 2]]),
        np.array([1, 0]),
        {
            "settings": asdict(settings),
            "measurement_units": "neurite cable micrometres",
            "features": [asdict(c) for c in features.columns],
            "external_annotations_used_for_clustering": False,
        },
    )
    yield module, widget, result, cohorts, annotations
    widget.shutdown()
    widget.close()
    widget.deleteLater()
    qt_app.processEvents()


def _wait(app, predicate, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for a regional clustering worker")


def test_controls_preserve_separate_measurements_and_default_laterality(
    regional_widget,
):
    _, widget, _, _, _ = regional_widget
    assert widget.scope.currentData() == "table"
    assert widget.laterality.currentData() == "separate"
    assert [widget.mode.itemData(i) for i in range(widget.mode.count())] == [
        "cable_distribution",
        "regions_reached",
        "terminus_distribution",
    ]
    assert "average linkage" in widget.definition.text()
    assert not widget.save_group.isEnabled()
    assert widget.memory_limit.value() == 512
    widget.grouping.setCurrentIndex(1)
    widget.exclusions.exclude_selector.set_selected_ids((10,))
    settings = widget._settings()
    assert settings.group_region_ids == (10,)
    assert settings.excluded_region_ids == (10,)
    assert 0 in widget.exclusions.exclude_selector._items_by_id


def test_result_shows_heatmap_assignments_and_selects_only_chosen_cluster(
    regional_widget,
):
    _, widget, result, _, _ = regional_widget
    received = []
    widget.cluster_selected.connect(received.append)
    widget._prepared(widget._generation, result.features)
    widget._finished(widget._generation, result)
    assert widget.zero_table.rowCount() == 1
    assert widget.zero_table.item(0, 0).text() == "zero.swc"
    assert widget.cluster_combo.count() == 3
    assert widget._heat_axis is not None
    assert received == []
    widget.assignment_id = "saved-assignment"
    widget.cluster_combo.setCurrentIndex(1)
    assert received[-1][:2] == ("saved-assignment", (result.file_ids[0],))
    assert widget.save_group.isEnabled()


def test_annotation_overlay_is_post_hoc_and_does_not_change_clustering(regional_widget):
    _, widget, result, _, annotations = regional_widget
    widget._finished(widget._generation, result)
    matrix = result.matrix.copy()
    labels = result.labels.copy()
    annotations.update({result.file_ids[0]: "IT-like", result.file_ids[1]: "PT-like"})
    widget.overlay.setCurrentIndex(widget.overlay.findData("label"))
    assert not widget.overlay_counts.isHidden()
    assert widget.overlay_counts.rowCount() == 2
    assert widget.overlay_counts.columnCount() == 3
    assert (
        sum(
            int(widget.overlay_counts.item(r, c).text())
            for r in range(2)
            for c in (1, 2)
        )
        == 2
    )
    np.testing.assert_array_equal(result.matrix, matrix)
    np.testing.assert_array_equal(result.labels, labels)
    assert result.metadata["external_annotations_used_for_clustering"] is False


def test_save_result_and_neuron_group_are_portable(
    regional_widget, monkeypatch, tmp_path
):
    module, widget, result, _, _ = regional_widget
    widget._finished(widget._generation, result)
    path = tmp_path / "result.npz"
    monkeypatch.setattr(
        module.QFileDialog, "getSaveFileName", lambda *_: (str(path), "")
    )
    widget._save_result()
    with np.load(path, allow_pickle=False) as saved:
        assert tuple(saved["file_ids"]) == result.file_ids
    path = tmp_path / "group.json"
    widget.cluster_combo.setCurrentIndex(1)
    widget.group_name.setText("Unsupervised morphology group")
    groups = []
    widget.neuron_group_saved.connect(lambda ids, name: groups.append((ids, name)))
    widget._save_group()
    assert json.loads(path.read_text())["file_ids"] == [result.file_ids[0]]
    assert groups == [((result.file_ids[0],), "Unsupervised morphology group")]


def test_stale_results_are_discarded_and_source_changes_clear_inspection(
    regional_widget,
):
    _, widget, result, _, _ = regional_widget
    widget._finished(widget._generation - 1, result)
    assert widget.result is None
    widget._finished(widget._generation, result)
    widget.set_context(None, None)
    assert widget.result is None
    assert not widget.run_button.isEnabled()
    assert not widget.save_result.isEnabled()


def test_real_background_worker_can_cancel_before_publication(
    qt_app, profile_atlas, tmp_path
):
    from napari_neuron_navigator.workers import RegionalClusteringWorker

    source = tmp_path / "source.parquet"
    frame = _write_source(source)
    build_region_profile(source, profile_atlas)
    worker = RegionalClusteringWorker(
        4,
        source,
        profile_atlas,
        RegionalClusteringSettings(tuple(frame.file_id.unique()), n_clusters=2),
    )
    ready, cancelled = [], []
    worker.finished.connect(lambda *args: ready.append(args))
    worker.cancelled.connect(lambda *args: cancelled.append(args))
    worker.cancel()
    worker.run()
    assert ready == []
    assert cancelled[0][0] == 4


def test_cancel_discards_completion_already_queued_for_publication(regional_widget):
    _, widget, result, _, _ = regional_widget
    generation = widget._generation
    cancelled = []
    widget._worker = SimpleNamespace(cancel=lambda: cancelled.append(True))
    widget._running = True
    widget.cancel()
    widget._finished(generation, result)
    assert cancelled == [True]
    assert widget.result is None
    assert widget.cluster_combo.count() == 0
    widget._worker = None


def test_napari_assignment_report_and_saved_group_handoff(
    qt_app, viewer_model, profile_atlas, tmp_path, monkeypatch
):
    from qtpy.QtWidgets import QScrollArea

    from napari_neuron_navigator.db import NeuronDatabase
    from napari_neuron_navigator.widgets.neuron_table import NeuronMetadataUpdate
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    source = tmp_path / "source.parquet"
    frame = _write_source(source)
    build_region_profile(source, profile_atlas)
    database = NeuronDatabase(source)
    viewer = viewer_model
    owner = NeuronViewerWidget(viewer)
    try:
        owner._db, owner._atlas = database, profile_atlas
        owner._neuron_table.populate(list(frame.file_id.unique()))
        owner._refresh_regional_profile_status()
        widget = owner._regional_clustering
        widget.cluster_count.setValue(2)
        widget.section.set_expanded(True)
        owner._tabs.setCurrentIndex(1)
        owner.resize(1600, 1050)
        owner.show()
        widget.run()
        _wait(qt_app, lambda: widget._thread is None)
        assert widget.result is not None, widget.status.text()
        assignment = owner._cluster_assignment_store.active
        assert assignment.coordinate_space == "Regional Profile"
        assert set(assignment.assignments) == set(widget.result.file_ids)
        assert assignment.run_metadata["distance_metric"] == "cosine"
        # Supply inspection labels only after unsupervised clustering.
        ids = widget.result.file_ids
        owner._neuron_table.apply_metadata_updates(
            {
                ids[0]: NeuronMetadataUpdate(label="IT-like", tags=("keep",)),
                ids[1]: NeuronMetadataUpdate(label="PT-like", tags=("keep",)),
            }
        )
        widget.overlay.setCurrentIndex(widget.overlay.findData("label"))
        # Cluster inspection must expose its members even when a Data
        # annotation filter currently hides every neuron.
        tag_filter = owner._neuron_table.tag_filter_combo
        tag_filter.setCurrentIndex(tag_filter.findData(""))
        widget.cluster_combo.setCurrentIndex(1)
        assert tag_filter.currentData() is None
        selected = widget.result.file_ids_for_cluster(1)
        assert tuple(owner._neuron_table.get_selected_file_ids()) == selected
        report = owner._region_report
        _wait(qt_app, lambda: report._worker is None and report._report is not None)
        assert report._report.cohort_file_ids == selected
        path = tmp_path / "cluster_group.json"
        module = importlib.import_module(
            "napari_neuron_navigator.widgets.regional_clustering"
        )
        monkeypatch.setattr(
            module.QFileDialog, "getSaveFileName", lambda *_: (str(path), "")
        )
        widget.group_name.setText("Regional morphology group")
        widget._save_group()
        assert path.exists()
        assert (
            owner._neuron_table._entries[selected[0]].group
            == "Regional morphology group"
        )
        assert owner._neuron_table._entries[selected[0]].tags == ("keep",)
        assert {e.label for e in owner._neuron_table._entries.values()} == {
            "IT-like",
            "PT-like",
        }
        assert (
            widget.result.metadata["external_annotations_used_for_clustering"] is False
        )
        scroll = owner._tabs.widget(1).findChild(QScrollArea)
        scroll.ensureWidgetVisible(widget)
        qt_app.processEvents()
        preview = tmp_path / "regional_clustering.png"
        assert owner.grab().save(str(preview))
        print(f"Regional clustering UI preview: {preview}")
    finally:
        owner._on_neuron_viewer_destroyed()
        owner.close()
        owner.deleteLater()
        qt_app.processEvents()
        database.close()
