"""Real Qt report interaction checks; run on a desktop outside the sandbox."""

from __future__ import annotations

import importlib
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from napari_neuron_navigator.analysis.region_profile import build_region_profile
from napari_neuron_navigator.analysis.region_report import (
    REPORT_COLUMNS,
    aggregate_region_report,
)
from tests.test_region_profile_stage1 import (
    _write_source,
)
from tests.test_region_profile_stage1 import (
    profile_atlas as profile_atlas,  # noqa: PLC0414
)
from tests.test_region_report import report_context as report_context  # noqa: PLC0414


@pytest.fixture(scope="module")
def qt_app():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def report_widget(qt_app, report_context):
    module = importlib.import_module("napari_neuron_navigator.widgets.region_report")
    connection, structures, cohort = report_context
    membership = {"table": list(cohort), "selected": [cohort[0]]}
    widget = module.RegionReportWidget(lambda scope: membership[scope])
    widget.set_context(
        "/private/tmp/report-fixture-source.parquet",
        SimpleNamespace(structures=structures, atlas_name="fixture"),
    )
    report = aggregate_region_report(connection, structures=structures, file_ids=cohort)
    widget._accept_report(widget._generation, report)
    yield widget, membership, report
    widget.shutdown()
    widget.close()
    widget.deleteLater()
    qt_app.processEvents()


def _item(tree, region_id, side="combined"):
    pending = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while pending:
        item = pending.pop()
        if item.group_key == (region_id, side):
            return item
        pending.extend(item.child(i) for i in range(item.childCount()))
    raise AssertionError(f"Missing report item {region_id}, {side}")


def test_defaults_have_requested_columns_and_only_anatomical_exclusions(report_widget):
    widget, _, _ = report_widget
    assert not widget.section.is_expanded()
    assert widget.cohort_combo.currentText() == "Data table"
    assert not widget.split_laterality.isChecked()
    assert [
        widget.tree.headerItem().text(i) for i in range(widget.tree.columnCount())
    ] == list(REPORT_COLUMNS)
    assert widget.tree.sortColumn() == 2
    editor = widget.exclusion_editor
    assert not hasattr(editor, "tabs")
    assert editor.exclude_table is None
    assert editor.include_table is None
    assert set(editor.exclude_selector._items_by_id) == {0, 1, 10, 11, 12}


def test_expanding_and_collapsing_changes_direct_rows_to_unique_neuron_aggregate(
    report_widget,
):
    widget, _, _ = report_widget
    parent = _item(widget.tree, 10)
    assert parent.isExpanded()
    assert parent.text(1) == ""
    direct = next(
        parent.child(i)
        for i in range(parent.childCount())
        if parent.child(i).text(0).endswith("[direct]")
    )
    assert direct.text(1) == "1"
    assert direct.text(3) == "0.300000"
    parent.setExpanded(False)
    assert parent.text(1) == "2"
    assert parent.text(2) == "66.67"
    assert parent.text(3) == "7.800000"
    assert parent.text(6) == "7"
    parent.setExpanded(True)
    assert parent.text(3) == ""
    widget._collapse_all()
    assert _item(widget.tree, 1).text(3) == "7.800000"
    widget._expand_all()
    assert _item(widget.tree, 1).text(3) == ""


def test_sorting_uses_numbers_instead_of_formatted_strings(report_widget):
    from qtpy.QtCore import Qt

    widget, _, report = report_widget
    base = report.nodes[0].aggregate[0]
    widget.tree.clear()
    small = widget._make_item("Small", replace(base, neurons_present=90), False)
    large = widget._make_item("Large", replace(base, neurons_present=1000), False)
    widget.tree.addTopLevelItems([small, large])
    widget.tree.sortByColumn(1, Qt.DescendingOrder)
    assert widget.tree.topLevelItem(0).text(1) == "1,000"


def test_split_adds_laterality_and_separates_side_metrics(
    report_widget, report_context
):
    widget, _, _ = report_widget
    widget.split_laterality.setChecked(True)
    connection, structures, cohort = report_context
    split = aggregate_region_report(
        connection, structures=structures, file_ids=cohort, split_laterality=True
    )
    widget._accept_report(widget._generation, split)
    assert widget.tree.columnCount() == 8
    assert widget.tree.headerItem().text(1) == "Laterality"
    assert widget.tree.sortColumn() == 3
    contra = _item(widget.tree, 11, "contralateral")
    assert contra.text(1) == "Contralateral"
    assert contra.text(2) == "1"
    assert contra.text(3) == "33.33"
    assert contra.text(4) == "0.500000"


def test_exclusion_picker_preserves_excluded_branches_and_cohort(
    report_widget, report_context
):
    widget, _, _ = report_widget
    widget.exclusion_editor.exclude_selector.select_regions(["R11"])
    assert widget.exclusion_editor.excluded_region_ids() == (11,)
    connection, structures, cohort = report_context
    report = aggregate_region_report(
        connection, structures=structures, file_ids=cohort, excluded_region_ids=(11,)
    )
    widget._accept_report(widget._generation, report)
    parent = _item(widget.tree, 10)
    parent.setExpanded(False)
    assert parent.text(3) == "4.300000"
    assert "Cohort: 3 neurons" in widget.status.text()
    assert 11 in widget.exclusion_editor.exclude_selector._items_by_id
    assert widget.exclusion_editor.excluded_region_ids() == (11,)


def test_selected_rows_are_independent_and_stale_results_are_discarded(report_widget):
    widget, membership, report = report_widget
    old_generation = widget._generation
    widget.cohort_combo.setCurrentIndex(1)
    assert widget._request_key[2] == tuple(membership["selected"])
    assert widget.tree.topLevelItemCount() == 0
    widget._accept_report(old_generation, report)
    assert widget.tree.topLevelItemCount() == 0
    membership["selected"] = []
    widget.invalidate()
    assert "Select neuron rows" in widget.status.text()
    assert widget.tree.topLevelItemCount() == 0


def test_missing_profile_has_build_action(report_widget):
    widget, _, _ = report_widget
    widget._start_report()
    assert "Click Build Regional Profile" in widget.status.text()
    assert widget.build_button.isEnabled()


def test_unmapped_exclusion_and_clustering_defaults(report_widget):
    widget, _, _ = report_widget
    widget.exclusion_editor.exclude_selector.set_selected_ids((0,))
    assert widget.exclusion_editor.excluded_region_ids() == (0,)
    module = importlib.import_module(
        "napari_neuron_navigator.widgets.region_filter_editor"
    )
    editor = module.RegionFilterEditorWidget()
    try:
        assert [editor.tabs.tabText(i) for i in range(editor.tabs.count())] == [
            "Include",
            "Exclude",
        ]
        assert editor.include_table.columnCount() == 2
        assert editor.exclude_table.columnCount() == 4
    finally:
        editor.deleteLater()


def test_regions_tab_tracks_actual_data_selection(qt_app, tmp_path, profile_atlas):
    import napari
    from qtpy.QtWidgets import QScrollArea

    from napari_neuron_navigator.db import NeuronDatabase
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    source = tmp_path / "source.parquet"
    frame = _write_source(source)
    build_region_profile(source, profile_atlas)
    database = NeuronDatabase(source)
    viewer = napari.Viewer(show=False)
    widget = NeuronViewerWidget(viewer)
    try:
        widget._db = database
        widget._atlas = profile_atlas
        cohort = list(frame.file_id.unique())
        widget._neuron_table.populate(cohort)
        widget._refresh_regional_profile_status()
        report_widget = widget._region_report
        assert set(report_widget._cohort_provider("table")) == set(cohort)
        widget._neuron_table._table.selectRow(0)
        assert len(report_widget._cohort_provider("selected")) == 1
        assert widget._tabs.tabText(1) == "Regions"
        assert widget._tabs.widget(1).isAncestorOf(report_widget)
        report_widget.section.set_expanded(True)
        widget._tabs.setCurrentIndex(1)
        widget.resize(1800, 1050)
        widget.show()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            qt_app.processEvents()
            if (
                report_widget._displayed_generation == report_widget._generation
                and report_widget._worker is None
            ):
                break
            time.sleep(0.01)
        assert report_widget.tree.topLevelItemCount() > 0, report_widget.status.text()
        assert f"Cohort: {len(cohort)} neurons" in report_widget.status.text()
        assert report_widget.tree.columnWidth(0) >= 300
        scroll = widget._tabs.widget(1).findChild(QScrollArea)
        scroll.ensureWidgetVisible(report_widget)
        qt_app.processEvents()
        # This widget is shown standalone by the check, so explicitly inherit
        # the real napari stylesheet it receives when hosted in the viewer.
        for theme in ("dark", "light"):
            viewer.theme = theme
            widget.setStyleSheet(viewer.window._qt_window.styleSheet())
            qt_app.processEvents()
            scroll.ensureWidgetVisible(report_widget)
            qt_app.processEvents()
            image_path = tmp_path / f"region_report_{theme}.png"
            assert widget.grab().save(str(image_path))
            print(f"Region report {theme} preview: {image_path}")
    finally:
        widget._on_neuron_viewer_destroyed()
        widget.close()
        widget.deleteLater()
        viewer.close()
        qt_app.processEvents()
        database.close()


def test_worker_validates_sidecar_and_rejects_changed_source(
    qt_app, profile_atlas, tmp_path
):
    module = importlib.import_module("napari_neuron_navigator.widgets.region_report")
    source = tmp_path / "neurons.parquet"
    frame = _write_source(source)
    metadata = build_region_profile(source, profile_atlas)
    ids = tuple(frame.file_id.unique())
    worker = module.RegionalReportWorker(7, str(source), profile_atlas, ids, False, ())
    ready, errors = [], []
    worker.ready.connect(lambda generation, result: ready.append((generation, result)))
    worker.error.connect(lambda generation, message: errors.append(message))
    worker.run()
    assert errors == []
    assert ready[0][0] == 7
    report = ready[0][1]
    direct_cable = sum(
        value.total_cable_mm for node in report.nodes for value in node.direct
    )
    assert direct_cable * 1000 == pytest.approx(
        metadata.build_summary.total_cable_length_um
    )
    frame.loc[0, "x"] += 0.01
    frame.to_parquet(source, index=False)
    worker.run()
    assert len(ready) == 1
    assert "source Parquet content differs" in errors[0]


def test_background_report_tracks_latest_cohort_and_stops_cleanly(
    qt_app, profile_atlas, tmp_path
):
    module = importlib.import_module("napari_neuron_navigator.widgets.region_report")
    source = tmp_path / "neurons.parquet"
    frame = _write_source(source)
    build_region_profile(source, profile_atlas)
    ids = list(frame.file_id.unique())
    widget = module.RegionReportWidget(lambda _scope: ids)
    widget.set_context(source, profile_atlas)
    widget.section.set_expanded(True)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            qt_app.processEvents()
            if (
                widget._displayed_generation == widget._generation
                and widget._worker is None
            ):
                break
            time.sleep(0.01)
        assert widget.tree.topLevelItemCount() > 0, widget.status.text()
        ids[:] = ids[:1]
        widget.invalidate()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            qt_app.processEvents()
            if (
                widget._displayed_generation == widget._generation
                and widget._worker is None
            ):
                break
            time.sleep(0.01)
        assert "Cohort: 1 neurons" in widget.status.text()
        assert widget.tree.topLevelItemCount() > 0
    finally:
        widget.shutdown()
        widget.close()
        widget.deleteLater()
        qt_app.processEvents()
