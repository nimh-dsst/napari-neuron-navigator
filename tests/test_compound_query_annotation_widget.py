"""Real Qt compound-query annotations on a cohort populated by a simple query."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from napari_neuron_navigator.analysis.region_profile import build_region_profile
from napari_neuron_navigator.analysis.region_query import RegionSelection
from tests.region_profile_stage0_fixtures import morphology_fixture_frame
from tests.test_region_profile_stage1 import (
    profile_atlas as profile_atlas,  # noqa: PLC0414
)


def _wait(app, owner):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        app.processEvents()
        if owner._regional_query_thread is None:
            return
        time.sleep(0.01)
    raise AssertionError("Compound query did not finish")


@pytest.fixture
def query_owner(qt_app, viewer_model, profile_atlas, tmp_path, monkeypatch):
    from napari_neuron_navigator.db import NeuronDatabase
    from napari_neuron_navigator.widgets.compound_region_query import (
        CONDITION_TERMINUS_COUNT,
        SUBJECT_NEURITE,
        StructuredRegionCondition,
        StructuredRegionGroup,
    )
    from napari_neuron_navigator.widgets.neuron_table import NeuronMetadataUpdate
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    frame = morphology_fixture_frame()
    lower_tip = (frame.file_id == "mirror-lower.swc") & (frame.node_id == 900)
    frame.loc[lower_tip, ["x", "region_id", "region_acronym"]] = [25, 22, "MID"]
    outside = frame[frame.file_id == "mirror-lower.swc"].copy()
    outside["file_id"], outside["subject"] = "outside-cohort.swc", "outside"
    outside["x"], outside["region_id"], outside["region_acronym"] = 25, 22, "MID"
    source = tmp_path / "source.parquet"
    pd.concat([frame, outside], ignore_index=True).to_parquet(source, index=False)
    build_region_profile(source, profile_atlas)
    database = NeuronDatabase(source)
    viewer = viewer_model
    owner = NeuronViewerWidget(viewer)
    try:
        owner._db, owner._atlas = database, profile_atlas
        owner._whole_parquet_region_selector.set_atlas(profile_atlas)
        owner._whole_parquet_region_selector.select_region_by_id(11)
        owner._query_neurons_by_region(soma_only=True)
        assert set(owner._current_table_file_ids()) == {
            "mirror-lower.swc",
            "mirror-upper.swc",
        }, owner._regions_status_label.text()
        table = owner._neuron_table
        table.apply_metadata_updates(
            {
                file_id: NeuronMetadataUpdate(
                    label="Existing label", group="Cohort", tags=("keep",)
                )
                for file_id in table.file_ids()
            }
        )
        table._entries["mirror-upper.swc"].notes = "Preserve this note"
        table.select_file_ids(["mirror-upper.swc"])
        owner._compound_region_query.set_atlas(profile_atlas)
        owner._compound_region_query.set_value(
            StructuredRegionGroup(
                children=(
                    StructuredRegionCondition(
                        subject=SUBJECT_NEURITE,
                        condition=CONDITION_TERMINUS_COUNT,
                        region_selection=RegionSelection.from_ids([22]),
                    ),
                )
            )
        )
        owner._region_query_source_combo.setCurrentText("Compound Region Query")
        owner._refresh_regional_profile_status()
        owner._set_region_query_buttons_enabled(True)
        monkeypatch.setattr(
            owner,
            "_populate_neuron_table",
            MagicMock(wraps=owner._populate_neuron_table),
        )
        yield owner
    finally:
        owner._on_neuron_viewer_destroyed()
        owner.close()
        owner.deleteLater()
        qt_app.processEvents()
        database.close()


@pytest.mark.parametrize("action", ["tag", "label"])
def test_simple_query_cohort_can_be_annotated_without_replacing_rows(
    query_owner, qt_app, action
):
    from napari_neuron_navigator.widgets.neuron_table import NeuronTableWidget

    owner = query_owner
    editor, table = owner._compound_region_query, owner._neuron_table
    before = {key: value.to_state() for key, value in table._entries.items()}
    assert owner._region_query_scope_combo.currentText() == "Whole Parquet"
    editor.result_action_combo.setCurrentIndex(
        editor.result_action_combo.findData(action)
    )
    assert owner._region_query_scope_combo.currentText() == "Current Table"
    assert not owner._region_query_scope_combo.isEnabled()
    editor.annotation_input.setText("MID termini")
    owner._region_query_find_btn.click()
    snapshot = owner._regional_query_snapshot
    assert set(snapshot["input_file_ids"]) == set(before)
    assert snapshot["result_action"] == action
    assert not editor._result_controls.isEnabled()
    # The action/text is captured before the worker starts.
    editor.annotation_input.setText("Changed after launch")
    _wait(qt_app, owner)
    assert set(table.file_ids()) == set(before)
    owner._populate_neuron_table.assert_not_called()
    assert table._entries["mirror-upper.swc"].to_state() == before["mirror-upper.swc"]
    expected = dict(before["mirror-lower.swc"])
    if action == "tag":
        expected["tags"] = ["keep", "MID termini"]
    else:
        expected["label"] = "MID termini"
    assert table._entries["mirror-lower.swc"].to_state() == expected
    assert table.get_selected_file_ids() == ["mirror-upper.swc"]
    assert table._current_table_view == "label"
    assert "retained all 2 table neurons" in owner._regions_status_label.text()
    assert not owner._region_query_scope_combo.isEnabled()
    assert editor._result_controls.isEnabled()

    restored = NeuronTableWidget()
    try:
        restored.import_state(table.export_state())
        assert restored._entries["mirror-lower.swc"].to_state() == expected
        assert set(restored.file_ids()) == set(before)
    finally:
        restored.close()
        restored.deleteLater()

    # Repeating the same annotation is safe and tags stay unique.
    editor.annotation_input.setText("MID termini")
    owner._region_query_find_btn.click()
    _wait(qt_app, owner)
    assert table._entries["mirror-lower.swc"].to_state() == expected


def test_zero_matches_leave_all_rows_and_annotations_unchanged(query_owner, qt_app):
    from napari_neuron_navigator.widgets.compound_region_query import (
        CONDITION_TERMINUS_COUNT,
        SUBJECT_NEURITE,
        StructuredRegionCondition,
        StructuredRegionGroup,
    )

    owner = query_owner
    editor, table = owner._compound_region_query, owner._neuron_table
    before = table.export_state()
    editor.result_action_combo.setCurrentIndex(
        editor.result_action_combo.findData("tag")
    )
    editor.annotation_input.setText("No matches")
    editor.set_value(
        StructuredRegionGroup(
            children=(
                StructuredRegionCondition(
                    subject=SUBJECT_NEURITE,
                    condition=CONDITION_TERMINUS_COUNT,
                    region_selection=RegionSelection.from_ids([33]),
                ),
            )
        )
    )
    owner._run_compound_region_query()
    _wait(qt_app, owner)
    assert table.export_state() == before
    owner._populate_neuron_table.assert_not_called()
    assert "Tagged 0 matching" in owner._regions_status_label.text()


def test_membership_change_during_query_discards_annotation_result(query_owner, qt_app):
    owner = query_owner
    editor, table = owner._compound_region_query, owner._neuron_table
    editor.result_action_combo.setCurrentIndex(
        editor.result_action_combo.findData("tag")
    )
    editor.annotation_input.setText("Must not apply")
    owner._run_compound_region_query()
    table.remove_file_ids(["mirror-upper.swc"])
    before = table.export_state()
    _wait(qt_app, owner)
    assert table.export_state() == before
    assert "membership changed" in owner._regions_status_label.text()
    owner._populate_neuron_table.assert_not_called()


def test_empty_annotation_and_empty_table_do_not_start_query(query_owner):
    owner = query_owner
    editor = owner._compound_region_query
    editor.result_action_combo.setCurrentIndex(
        editor.result_action_combo.findData("tag")
    )
    editor.annotation_input.setText("  ")
    owner._run_compound_region_query()
    assert owner._regional_query_thread is None
    assert "Enter annotation text" in owner._regions_status_label.text()
    editor.annotation_input.setText("MID termini")
    owner._neuron_table.populate([])
    owner._run_compound_region_query()
    assert owner._regional_query_thread is None
    assert "Current table is empty" in owner._regions_status_label.text()
    owner._populate_neuron_table.assert_not_called()


def test_cancel_discards_a_queued_completion_without_annotating(query_owner):
    from types import SimpleNamespace

    owner = query_owner
    before = owner._neuron_table.export_state()
    owner._regional_query_snapshot = {"result_action": "tag"}
    owner._regional_query_worker = SimpleNamespace(cancel=lambda: None)
    owner._cancel_regional_query()
    owner._on_regional_query_finished(SimpleNamespace())
    assert owner._neuron_table.export_state() == before
    owner._populate_neuron_table.assert_not_called()
    assert "cancelled" in owner._regions_status_label.text()
    owner._regional_query_worker = None


def test_result_actions_and_other_query_sources_keep_scope_controls_consistent(
    query_owner,
):
    owner = query_owner
    editor = owner._compound_region_query
    assert editor.result_action() == "replace"
    assert editor.annotation_input.isHidden()
    editor.result_action_combo.setCurrentIndex(
        editor.result_action_combo.findData("tag")
    )
    assert not editor.annotation_input.isHidden()
    assert owner._region_query_find_btn.text() == "Tag Matches in Current Table"
    owner._region_query_source_combo.setCurrentText("Atlas Regions")
    assert owner._region_query_scope_combo.isEnabled()
    assert owner._region_query_find_btn.text() == "Find Neurons in Selected Regions"
    owner._region_query_source_combo.setCurrentText("Compound Region Query")
    assert not owner._region_query_scope_combo.isEnabled()
    editor.result_action_combo.setCurrentIndex(
        editor.result_action_combo.findData("replace")
    )
    assert owner._region_query_scope_combo.isEnabled()
    assert owner._region_query_find_btn.text() == "Run Compound Region Query"


def test_query_tags_populate_data_filters_and_themed_controls(
    query_owner, qt_app, tmp_path
):
    import numpy as np
    from napari.qt import get_stylesheet
    from qtpy.QtWidgets import QScrollArea

    owner = query_owner
    editor, table = owner._compound_region_query, owner._neuron_table
    editor.result_action_combo.setCurrentIndex(
        editor.result_action_combo.findData("tag")
    )
    editor.annotation_input.setText("MID termini")
    owner._run_compound_region_query()
    _wait(qt_app, owner)
    table.tag_filter_combo.setCurrentIndex(
        table.tag_filter_combo.findData("MID termini")
    )
    table.label_filter_combo.setCurrentIndex(
        table.label_filter_combo.findData("Existing label")
    )
    shown = {
        file_id
        for row, file_id in table._iter_rows_with_file_ids()
        if not table._table.isRowHidden(row)
    }
    assert shown == {"mirror-lower.swc"}
    assert table._filter_summary_label.text() == "Showing 1 of 2 neurons"
    assert len(table.file_ids()) == 2
    table.tag_filter_combo.set_selection(("MID termini", "keep"))
    assert set(table.get_selected_file_ids()) <= {"mirror-lower.swc"}
    table.tag_tools.set_expanded(True)
    table.delete_tags_button.click()
    # Both checked names are removed from the displayed match only.
    assert table._entries["mirror-lower.swc"].tags == ()
    assert table._entries["mirror-upper.swc"].tags == ("keep",)
    table.undo_tags_button.click()
    assert set(table._entries["mirror-lower.swc"].tags) == {"MID termini", "keep"}
    ids = table.file_ids()
    layer = owner.viewer.add_points(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        name="Soma Labels",
        metadata={"file_ids": ids},
        face_color=[table.get_color(file_id) for file_id in ids],
    )
    owner._current_neuron_layers.append(layer)
    owner._highlighted_file_ids = None
    table.tag_action_scope.setCurrentIndex(table.tag_action_scope.findData("table"))
    table.color_tags_button.click()
    np.testing.assert_allclose(
        layer.face_color, [table.get_color(file_id) for file_id in ids]
    )
    assert table.get_color(ids[0]) != table.get_color(ids[1])
    owner._tabs.setCurrentIndex(0)
    owner.resize(1200, 1050)
    owner.show()
    scroll = owner._tabs.widget(0).findChild(QScrollArea)
    for theme in ("dark", "light"):
        owner.viewer.theme = theme
        owner.setStyleSheet(get_stylesheet(theme))
        qt_app.processEvents()
        scroll.ensureWidgetVisible(table)
        qt_app.processEvents()
        preview = tmp_path / f"data_annotation_filters_{theme}.png"
        assert owner.grab().save(str(preview))
        print(f"Data annotation filters {theme} preview: {preview}")
