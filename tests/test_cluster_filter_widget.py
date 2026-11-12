"""Data cluster filters follow saved column identity and assigned membership."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from napari_neuron_navigator.neuron_table_ops import ClusterFilterSelection


@pytest.fixture
def filter_owner(qt_app, viewer_model, monkeypatch):
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    monkeypatch.setattr(
        NeuronViewerWidget, "_start_cached_template_autoload", lambda *_: None
    )
    owner = NeuronViewerWidget(viewer_model)
    owner._neuron_table.populate(
        ["subject-a/shared", "subject-b/shared", "subject-c/shared", "unassigned"]
    )
    owner._refresh_cluster_filter_controls()
    qt_app.processEvents()
    try:
        yield owner
    finally:
        owner._on_neuron_viewer_destroyed()
        owner.close()
        owner.deleteLater()
        qt_app.processEvents()


def _publish(owner, name, assignments):
    """Exercise the same assignment publication handler as a completed run."""
    result = SimpleNamespace(
        neuron_ids=list(assignments), labels=np.array(list(assignments.values()))
    )
    table_ids = owner._neuron_table.file_ids()
    assignment = owner._cluster_assignment_store.add(
        name=name,
        assignments=assignments,
        input_file_ids=table_ids,
        unassigned_neuron_ids=[fid for fid in table_ids if fid not in assignments],
        runtime_result=result,
    )
    owner._on_cluster_colors_updated(result, {})
    return assignment


def _switch(owner, assignment):
    owner._cluster_assignment_combo.setCurrentIndex(
        owner._cluster_assignment_combo.findData(assignment.assignment_id)
    )


def _toggle(owner, data):
    combo = owner._cluster_filter_combo
    index = _item_index(combo, data)
    combo._toggle_item_at_row(index)


def _item_index(combo, data):
    # Qt's QVariant comparison does not compare Python tuple payloads by value.
    return next(
        index for index in range(combo.count()) if combo.itemData(index) == data
    )


def _shown(owner):
    return set(owner._neuron_table.displayed_file_ids())


def test_new_column_defaults_to_all_assigned_and_switching_restores_each_filter(
    filter_owner, qt_app
):
    owner = filter_owner
    assert owner._cluster_filter_combo.currentText() == "All neurons"
    first = _publish(
        owner,
        "Soma Location 1",
        {"subject-a/shared": 1, "subject-b/shared": 2, "subject-c/shared": 2},
    )
    assert owner._selected_cluster_filter() == ClusterFilterSelection(all_clusters=True)
    assert _shown(owner) == {"subject-a/shared", "subject-b/shared", "subject-c/shared"}
    _toggle(owner, ("cluster", 2))
    assert _shown(owner) == {"subject-a/shared"}

    # Cluster 1 means different neurons in this new column. It must not
    # inherit the older column's Cluster 1 filter.
    second = _publish(
        owner, "Voxel Correlation 1", {"subject-a/shared": 2, "subject-b/shared": 1}
    )
    assert owner._cluster_filter_combo.currentText() == "All clusters"
    assert _shown(owner) == {"subject-a/shared", "subject-b/shared"}
    _toggle(owner, ("cluster", 2))
    assert _shown(owner) == {"subject-b/shared"}
    for _ in range(2):
        _switch(owner, first)
        qt_app.processEvents()
        assert _shown(owner) == {"subject-a/shared"}
        assert owner._selected_cluster_filter() == ClusterFilterSelection({1})
        _switch(owner, second)
        qt_app.processEvents()
        assert _shown(owner) == {"subject-b/shared"}
        assert owner._selected_cluster_filter() == ClusterFilterSelection({1})

    owner._cluster_assignment_store.rename(first.assignment_id, "Renamed soma run")
    owner._refresh_cluster_assignment_controls()
    _switch(owner, first)
    assert _shown(owner) == {"subject-a/shared"}
    assert owner._cluster_filter_assignment_id == first.assignment_id
    assert set(owner._neuron_table.file_ids()) == {
        "subject-a/shared",
        "subject-b/shared",
        "subject-c/shared",
        "unassigned",
    }


def test_all_clusters_bulk_selection_and_unclustered_controls_use_real_mouse(qt_app):
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest

    from napari_neuron_navigator.widgets.neuron_viewer import _ClusterFilterComboBox

    combo = _ClusterFilterComboBox()
    combo.set_cluster_options([0, 2, 10], include_unclustered=True)
    assert combo.currentText() == "All clusters"
    assert combo.cluster_filter_selection() == ClusterFilterSelection(all_clusters=True)
    for cluster_id in (0, 2, 10):
        assert combo._item_checked(_item_index(combo, ("cluster", cluster_id)))
    combo.show()
    combo.showPopup()
    qt_app.processEvents()

    def click(data):
        index = combo.model().index(_item_index(combo, data), 0)
        QTest.mouseClick(
            combo.view().viewport(),
            Qt.LeftButton,
            pos=combo.view().visualRect(index).center(),
        )

    try:
        click(("cluster", 2))
        assert combo.cluster_filter_selection() == ClusterFilterSelection({0, 10})
        assert combo.view().isVisible()
        click("all_clusters")
        assert combo.cluster_filter_selection() == ClusterFilterSelection(
            all_clusters=True
        )
        click("unclustered")
        assert combo.cluster_filter_selection() == ClusterFilterSelection(
            all_clusters=True, include_unclustered=True
        )
        click("all")
        assert combo.currentText() == "All neurons"
        assert combo.cluster_filter_selection().is_all
        click(("cluster", 0))
        assert combo.cluster_filter_selection() == ClusterFilterSelection({0})
        click("all_clusters")
        assert combo.cluster_filter_selection() == ClusterFilterSelection(
            all_clusters=True
        )
    finally:
        combo.hidePopup()
        combo.close()
        combo.deleteLater()
        qt_app.processEvents()


def test_empty_assignment_stays_empty_and_all_clusters_includes_new_labels(
    filter_owner, qt_app
):
    owner = filter_owner
    empty = _publish(owner, "No assigned neurons", {})
    for _ in range(2):
        owner._refresh_cluster_filter_controls()
        qt_app.processEvents()
        assert owner._cluster_filter_combo.currentText() == "All clusters"
        assert _shown(owner) == set()
    empty.assignments.update({"subject-a/shared": 0, "subject-b/shared": 9})
    owner._neuron_table.refresh_cluster_assignments()
    owner._refresh_cluster_filter_controls()
    assert _shown(owner) == {"subject-a/shared", "subject-b/shared"}
    _toggle(owner, "all")
    assert len(_shown(owner)) == 4
    _toggle(owner, "unclustered")
    assert _shown(owner) == {"subject-c/shared", "unassigned"}


def test_new_column_keeps_label_tag_and_manual_heatmap_filters(filter_owner, qt_app):
    from napari_neuron_navigator.widgets.neuron_table import NeuronMetadataUpdate

    owner = filter_owner
    table = owner._neuron_table
    table.apply_metadata_updates(
        {
            fid: NeuronMetadataUpdate(label="Keep", tags=("Keep",))
            for fid in table.file_ids()
            if fid != "subject-a/shared"
        }
    )
    qt_app.processEvents()
    table.label_filter_combo.setCurrentIndex(table.label_filter_combo.findData("Keep"))
    table.tag_filter_combo.set_selection("Keep")
    owner._manual_heatmap_combo.addItem(
        "Test heatmap", ("Test heatmap", ("subject-b/shared", "subject-c/shared"))
    )
    owner._manual_heatmap_combo.setCurrentIndex(owner._manual_heatmap_combo.count() - 1)
    _publish(owner, "Sparse run", {"subject-a/shared": 1, "subject-b/shared": 2})
    qt_app.processEvents()
    assert owner._selected_cluster_filter() == ClusterFilterSelection(all_clusters=True)
    assert _shown(owner) == {"subject-b/shared"}
    assert table.label_filter_combo.currentData() == "Keep"
    assert table.tag_filter_combo.selected_tags() == ("Keep",)


def test_deleted_columns_drop_filter_history_and_reloaded_source_resets_it(
    filter_owner, monkeypatch, tmp_path
):
    from napari_neuron_navigator.widgets import neuron_viewer

    owner = filter_owner
    first = _publish(owner, "First run", {"subject-a/shared": 1, "subject-b/shared": 2})
    _toggle(owner, ("cluster", 2))
    second = _publish(
        owner, "Second run", {"subject-a/shared": 2, "subject-b/shared": 1}
    )
    _toggle(owner, ("cluster", 2))
    owner._cluster_assignment_store.delete(second.assignment_id)
    owner._refresh_cluster_assignment_controls()
    owner._apply_active_cluster_assignment()
    assert _shown(owner) == {"subject-a/shared"}
    assert second.assignment_id not in owner._cluster_filter_by_assignment
    saved = owner._neuron_table.export_state()
    monkeypatch.setattr(
        neuron_viewer,
        "read_enhanced_parquet_metadata",
        lambda _path: {"table_state": saved},
    )
    owner._load_enhanced_table_state(tmp_path / "reloaded.parquet")
    owner._apply_saved_table_state_to_table()
    assert owner._cluster_assignment_store.active_assignment_id == first.assignment_id
    assert owner._cluster_filter_combo.currentText() == "All clusters"
    assert _shown(owner) == {"subject-a/shared", "subject-b/shared"}
