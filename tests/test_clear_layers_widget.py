"""Real Qt menu actions and independent clearing of napari neuron layers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def scene_owner(qt_app, viewer_model, monkeypatch):
    from napari_neuron_navigator.widgets.neuron_table import NeuronMetadataUpdate
    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    monkeypatch.setattr(
        NeuronViewerWidget, "_start_cached_template_autoload", lambda *_: None
    )
    owner = NeuronViewerWidget(viewer_model)
    file_ids = ["subject-a/shared", "subject-b/shared", "subject-c/shared"]

    def node_frame(ids):
        return pd.DataFrame(
            [
                {
                    "file_id": fid,
                    "neuron_id": "shared",
                    "type": node_type,
                    "x": 0.0,
                    "y": float(index),
                    "z": 0.0,
                }
                for fid in ids
                for index, node_type in enumerate((1, 2))
            ]
        )

    owner._db = MagicMock()
    owner._db.get_neurons_for_rendering.side_effect = node_frame
    owner._db.get_soma_locations.side_effect = lambda ids: node_frame(ids).query(
        "type == 1"
    )
    owner._db.get_soma_points.side_effect = lambda ids: node_frame(ids).query(
        "type == 1"
    )
    owner._db.get_neuron_lines_batch.side_effect = lambda ids: {
        fid: (np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([[0, 1]]))
        for fid in ids
    }
    table = owner._neuron_table
    table.populate(file_ids)
    table.apply_metadata_updates(
        {fid: NeuronMetadataUpdate(label="Keep", tags=("Keep",)) for fid in file_ids}
    )
    owner._render_mode_combo.setCurrentText("Both")
    owner._scene_render_modes = dict(zip(file_ids, ("full", "soma", "traces")))
    reference = viewer_model.add_image(np.zeros((3, 3, 3)), name="Reference")
    heatmap = viewer_model.add_image(np.ones((3, 3, 3)), name="Neuron Heatmap")
    owner._render_scene()
    qt_app.processEvents()
    table.select_file_ids([file_ids[1]])
    try:
        yield owner, file_ids, reference, heatmap
    finally:
        viewer_model.dims.ndisplay = 2
        owner._on_neuron_viewer_destroyed()
        owner.close()
        owner.deleteLater()
        qt_app.processEvents()


@pytest.mark.parametrize("ndisplay", [2, 3])
@pytest.mark.parametrize("choice", ["Neuron Traces", "Soma", "Both"])
def test_clear_layers_menu_preserves_other_category_and_table(
    scene_owner, qt_app, ndisplay, choice
):
    owner, (full_id, soma_id, trace_id), reference, heatmap = scene_owner
    table, viewer = owner._neuron_table, owner.viewer
    viewer.dims.ndisplay = ndisplay
    qt_app.processEvents()
    before = {fid: entry.to_state() for fid, entry in table._entries.items()}
    before_layers = {layer.name: layer for layer in owner._current_neuron_layers}
    expected_modes = {
        "Neuron Traces": {full_id: "soma", soma_id: "soma"},
        "Soma": {full_id: "traces", trace_id: "traces"},
        "Both": {},
    }[choice]
    menu = owner._clear_neurons_btn.menu()
    assert owner._clear_neurons_btn.text() == "Clear Layers"
    assert [action.text() for action in menu.actions()] == [
        "Neuron Traces",
        "Soma",
        "Both",
    ]
    action = next(action for action in menu.actions() if action.text() == choice)

    action.trigger()
    qt_app.processEvents()

    assert owner._scene_render_modes == expected_modes
    assert reference in viewer.layers and heatmap in viewer.layers
    assert table.get_selected_file_ids() == [soma_id]
    for fid, entry in table._entries.items():
        assert entry.to_state() == dict(
            before[fid], added_to_scene=fid in expected_modes
        )
    if choice == "Neuron Traces":
        assert owner._current_neuron_layers == [before_layers["Soma Labels"]]
        assert set(owner._soma_slice_projector._source_data) == {full_id, soma_id}
        assert not owner._slice_projector._source_data
    elif choice == "Soma":
        assert owner._current_neuron_layers == [
            before_layers["Neuron Lines"],
            before_layers["Neuron Points"],
        ]
        assert not owner._soma_slice_projector._source_data
        assert set(owner._slice_projector._source_data) == {full_id, trace_id}
        # Soma markers embedded in the raw point cloud also stay suppressed.
        points = before_layers["Neuron Points"]
        types = np.asarray(points.metadata["point_types"])
        assert np.all(points.face_color[types == 1, 3] == 0)
    else:
        assert owner._current_neuron_layers == []
        assert not owner._slice_projector._source_data
        assert not owner._soma_slice_projector._source_data
    # Repeating a clear, including with no matching layers, is harmless.
    action.trigger()
    assert owner._scene_render_modes == expected_modes


@pytest.mark.parametrize("choice", ["Neuron Traces", "Soma"])
def test_adding_selected_neurons_does_not_restore_cleared_layers_for_others(
    scene_owner, qt_app, choice
):
    owner, (full_id, soma_id, trace_id), _reference, _heatmap = scene_owner
    next(
        action
        for action in owner._clear_layers_menu.actions()
        if action.text() == choice
    ).trigger()
    owner._neuron_table.select_file_ids([soma_id])
    owner._render_selected_neurons()
    qt_app.processEvents()

    if choice == "Neuron Traces":
        assert owner._scene_render_modes == {full_id: "soma", soma_id: "full"}
        assert set(owner._slice_projector._source_data) == {soma_id}
        assert set(owner._soma_slice_projector._source_data) == {full_id, soma_id}
    else:
        assert owner._scene_render_modes == {
            full_id: "traces",
            trace_id: "traces",
            soma_id: "full",
        }
        assert set(owner._slice_projector._source_data) == {full_id, soma_id, trace_id}
        assert set(owner._soma_slice_projector._source_data) == {soma_id}
        soma = next(
            layer
            for layer in owner._current_neuron_layers
            if layer.name == "Soma Labels"
        )
        assert soma.metadata["file_ids"] == [soma_id]
        # Color and slice-mode changes must not reveal cleared soma markers.
        owner._neuron_table.update_colors({full_id: [0.2, 0.4, 0.6, 1.0]})
        owner._toggle_slice_projection(False)
        owner.viewer.dims.ndisplay = 3
        qt_app.processEvents()
        points = next(
            layer
            for layer in owner._current_neuron_layers
            if layer.name == "Neuron Points"
        )
        for index, (fid, node_type) in enumerate(
            zip(points.metadata["file_ids_per_point"], points.metadata["point_types"])
        ):
            if node_type == 1:
                assert (points.face_color[index, 3] > 0) == (fid == soma_id)


def test_cleared_soma_can_be_replaced_with_selected_rows(scene_owner, qt_app):
    owner, (full_id, soma_id, _trace_id), _reference, _heatmap = scene_owner
    next(
        action
        for action in owner._clear_layers_menu.actions()
        if action.text() == "Soma"
    ).trigger()
    owner._neuron_table.select_file_ids([soma_id])
    owner._render_selected_soma_only()
    qt_app.processEvents()
    assert owner._scene_render_modes[full_id] == "traces"
    assert set(owner._soma_slice_projector._source_data) == {soma_id}
    soma = next(
        layer for layer in owner._current_neuron_layers if layer.name == "Soma Labels"
    )
    assert soma.metadata["file_ids"] == [soma_id]
