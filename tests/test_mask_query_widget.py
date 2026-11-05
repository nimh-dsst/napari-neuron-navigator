"""Real Qt thread, progress, cancellation, and publication checks for mask scans."""

import threading
import time
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


def _wait(app, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Mask worker did not reach the expected state")


@pytest.fixture
def mask_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import (
        QApplication,
        QComboBox,
        QLabel,
        QProgressBar,
        QPushButton,
        QWidget,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Using extra keyword arguments on `Field`.*"
        )
        from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    from napari_neuron_navigator.analysis import mask_query

    app = QApplication.instance() or QApplication([])
    source = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["a/shared", "b/shared", "outside"],
            "neuron_id": ["shared"] * 3,
            "subject": ["a", "b", "c"],
            "x": [25.0, 25.0, 0.0],
            "y": [25.0, 25.0, 0.0],
            "z": [25.0, 25.0, 0.0],
            "type": [1, 2, 1],
        }
    ).to_parquet(source, index=False)
    owner = NeuronViewerWidget.__new__(NeuronViewerWidget)
    QWidget.__init__(owner)
    owner._db = SimpleNamespace(parquet_path=source)
    owner._atlas = SimpleNamespace(
        annotation=np.zeros((3, 3, 3)), resolution=(25, 25, 25)
    )
    owner._regional_query_thread = owner._regional_query_worker = None
    owner._regional_profile_thread = None
    owner._regional_query_snapshot = None
    owner._region_query_source = "Mask Layer"
    owner._region_query_scope_combo = QComboBox(owner)
    owner._region_query_scope_combo.addItem("Whole Parquet", "whole")
    owner._region_query_scope_combo.addItem("Current Table", "current")
    owner._regions_status_label = QLabel(owner)
    owner._mask_query_progress = QProgressBar(owner)
    owner._region_query_cancel_btn = QPushButton(owner)
    owner._region_query_find_btn = QPushButton(owner)
    owner._neuron_table = SimpleNamespace(file_ids=lambda: ["a/shared", "b/shared"])
    owner._populate_neuron_table = MagicMock()
    mask = np.zeros((3, 3, 3), dtype=np.uint32)
    mask[1, 1, 1] = 1
    layer = SimpleNamespace(data=mask, name="Intersection", metadata={})
    owner._selected_mask_query_layers = lambda: [layer]
    started, release = threading.Event(), threading.Event()
    real_query = mask_query.query_neurons_in_masks

    def gated_query(**settings):
        started.set()
        while not release.wait(0.005):
            if settings["cancel_check"]():
                raise mask_query.MaskQueryCancelled(
                    "Mask search cancelled; the Data table was not changed."
                )
        return real_query(**settings)

    monkeypatch.setattr(mask_query, "query_neurons_in_masks", gated_query)
    ticks = []
    timer = QTimer(owner)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(1)
    try:
        yield app, owner, started, release, ticks, layer
    finally:
        owner._shutdown_mask_query()
        release.set()
        _wait(app, lambda: owner._regional_query_thread is None)
        timer.stop()
        owner.close()
        owner.deleteLater()
        app.processEvents()


def test_mask_search_runs_off_gui_thread_and_reports_progress(mask_owner):
    app, owner, started, release, ticks, _layer = mask_owner
    owner._query_neurons_by_mask()
    _wait(app, started.is_set)
    _wait(app, lambda: len(ticks) >= 3)
    assert owner._region_query_cancel_btn.isEnabled()
    assert not owner._region_query_find_btn.isEnabled()
    assert not owner._mask_query_progress.isHidden()
    assert not owner._populate_neuron_table.called
    release.set()
    _wait(app, lambda: owner._regional_query_thread is None)
    result = owner._populate_neuron_table.call_args.args[0]
    assert result.file_id.tolist() == ["a/shared", "b/shared"]
    assert owner._mask_query_progress.value() == 100
    assert owner._mask_query_progress.isHidden()
    assert owner._region_query_find_btn.isEnabled()
    assert "Found 2" in owner._regions_status_label.text()


def test_cancel_and_late_completion_never_replace_table(mask_owner):
    app, owner, started, _release, _ticks, _layer = mask_owner
    owner._query_neurons_by_mask()
    _wait(app, started.is_set)
    owner._cancel_regional_query()
    # Simulate a successful result queued just before the cancellation.
    owner._on_mask_query_finished(pd.DataFrame({"file_id": ["late"]}))
    assert not owner._populate_neuron_table.called
    _wait(app, lambda: owner._regional_query_thread is None)
    assert "cancelled" in owner._regions_status_label.text().lower()


@pytest.mark.parametrize("change", ["source", "cohort", "mask"])
def test_changed_inputs_discard_queued_results(mask_owner, change):
    app, owner, started, release, _ticks, layer = mask_owner
    owner._region_query_scope_combo.setCurrentIndex(1)
    owner._query_neurons_by_mask()
    _wait(app, started.is_set)
    if change == "source":
        owner._db = None
    elif change == "cohort":
        owner._neuron_table.file_ids = lambda: ["a/shared"]
    else:
        layer.data = layer.data.copy()
    release.set()
    _wait(app, lambda: owner._regional_query_thread is None)
    assert not owner._populate_neuron_table.called
    assert "discarded" in owner._regions_status_label.text()


def test_closing_widget_cancels_active_scan(mask_owner):
    app, owner, started, _release, _ticks, _layer = mask_owner
    owner._query_neurons_by_mask()
    _wait(app, started.is_set)
    owner.close()
    _wait(app, lambda: owner._regional_query_thread is None)
    assert not owner._populate_neuron_table.called


@pytest.mark.parametrize("invalid", ["empty", "shape"])
def test_bad_masks_report_error_without_changing_table(mask_owner, invalid):
    app, owner, _started, release, _ticks, layer = mask_owner
    layer.data = np.zeros((3, 3, 3) if invalid == "empty" else (2, 2, 2), dtype=np.uint8)
    owner._query_neurons_by_mask()
    release.set()
    _wait(app, lambda: owner._regional_query_thread is None)
    assert not owner._populate_neuron_table.called
    assert "failed" in owner._regions_status_label.text()
    assert owner._mask_query_progress.isHidden()
    assert owner._region_query_find_btn.isEnabled()


def test_current_cohort_and_source_exclusion_are_preserved(mask_owner):
    app, owner, _started, release, _ticks, layer = mask_owner
    owner._region_query_scope_combo.setCurrentIndex(1)
    layer.metadata["query_excluded_file_ids"] = ["a/shared"]
    owner._query_neurons_by_mask()
    release.set()
    _wait(app, lambda: owner._regional_query_thread is None)
    assert owner._populate_neuron_table.call_args.args[0].file_id.tolist() == ["b/shared"]
    assert owner._populate_neuron_table.call_args.kwargs == {"preserve_existing": True}
    assert "excluded 1 source neuron" in owner._regions_status_label.text()
