"""Exercise live set filtering and overlap controls with real Qt widgets."""

import importlib
import json
from types import SimpleNamespace

import numpy as np

from napari_neuron_navigator.neuron_sets import AnnotationRule
from tests.test_neuron_table_widget import (
    real_table_widget as real_table_widget,  # noqa: PLC0414
)


def _cohort(widget):
    widget.import_state(
        {
            "entries": [
                {"file_id": "a/shared", "tags": ["TH", "CP"]},
                {"file_id": "b/shared", "tags": ["TH"]},
                {"file_id": "c/shared", "tags": ["CP"]},
            ]
        }
    )
    widget._refresh_annotation_filters()


def test_set_view_live_membership_and_show_members_use_whole_cohort(real_table_widget):
    module, widget = real_table_widget
    _cohort(widget)
    widget._table_view_combo.setCurrentIndex(widget._table_view_combo.findData("sets"))
    assert not widget._sets_panel.isHidden()
    assert not widget._table.isColumnHidden(module.COL_TAGS)
    widget.set_annotation_rule(
        AnnotationRule(tag_filter=("TH",), excluded_tags=("CP",))
    )
    panel = widget._sets_panel
    panel.name_edit.setText("TH without CP")
    panel.create_set()
    item = widget.neuron_sets.sets()[0]
    assert item.rule.members(widget._entries) == ("b/shared",)
    assert panel.summary_table.item(0, 1).text() == "1"

    # The live set updates even when another row filter currently hides members.
    widget.apply_filters(None, heatmap_file_ids=["c/shared"])
    widget._entries["a/shared"].tags = ("TH",)
    widget._refresh_annotation_filters()
    assert panel.summary_table.item(0, 1).text() == "2"
    panel.show_members()
    assert set(widget.displayed_file_ids()) == {"a/shared", "b/shared"}
    assert set(widget.get_selected_file_ids()) == {"a/shared", "b/shared"}
    assert len(widget.file_ids()) == 3

    # Snapshot captures membership for the heatmap, not a dynamic reference.
    requests = []
    widget.set_heatmap_requested.connect(requests.append)
    panel.create_heatmap()
    assert requests[0]["set_id"] == item.set_id
    assert requests[0]["file_ids"] == ["a/shared", "b/shared"]
    widget._entries["b/shared"].tags = ()
    assert requests[0]["file_ids"] == ["a/shared", "b/shared"]


def test_set_state_roundtrip_rules_survive_tag_changes_and_deletion(real_table_widget):
    _module, widget = real_table_widget
    _cohort(widget)
    item = widget.neuron_sets.save("TH", AnnotationRule(tag_filter=("TH",)))
    widget._on_sets_changed()
    state = json.loads(json.dumps(widget.export_state()))
    widget.import_state(state)
    assert widget.neuron_sets.get(item.set_id) == item
    widget._entries["c/shared"].tags = ("TH",)
    assert len(item.rule.members(widget._entries)) == 3
    widget._sets_panel.summary_table.selectRow(0)
    widget._sets_panel.delete_set()
    # Reapplying saved row annotations must not resurrect deleted definitions.
    widget.apply_state(state)
    assert widget.neuron_sets.sets() == ()
    widget.import_state({"entries": []})
    assert widget.neuron_sets.sets() == ()


def test_overlap_widget_displays_membership_and_spatial_results(real_table_widget):
    _module, widget = real_table_widget
    _cohort(widget)
    left = widget.neuron_sets.save("TH", AnnotationRule(tag_filter=("TH",)))
    right = widget.neuron_sets.save("CP", AnnotationRule(tag_filter=("CP",)))
    layers = []
    for item, data in ((left, [1, 1, 0, 0]), (right, [0, 1, 1, 0])):
        members = list(item.rule.members(widget._entries))
        layers.append(
            SimpleNamespace(
                name=item.name,
                data=np.array(data).reshape(1, 2, 2),
                metadata={
                    "atlas_name": "test",
                    "mask_query_source": True,
                    "source_file_ids": members,
                    "neuron_set": {**item.to_state(), "file_ids": members},
                },
            )
        )
    overlap_module = importlib.import_module(
        "napari_neuron_navigator.widgets.set_overlap"
    )
    created = []

    def create_mask(mask, sources):
        created.append((mask, sources))
        return SimpleNamespace(name="Intersection")

    panel = overlap_module.SetOverlapWidget(
        widget, lambda: layers, lambda: "test", intersection_mask_creator=create_mask
    )
    assert not panel.create_intersection_button.isEnabled()
    panel.region_combos[0].setCurrentIndex(1)
    assert not panel.create_intersection_button.isEnabled()
    panel.region_combos[1].setCurrentIndex(1)
    assert panel.create_intersection_button.isEnabled()
    # Creation uses the current masks without requiring a previous Compare Sets.
    panel.create_intersection_button.click()
    np.testing.assert_array_equal(created[0][0], np.array([[[0, 1], [0, 0]]]))
    assert created[0][1] == layers
    assert "1 intersecting voxels" in panel.status.text()
    layers[1].data[:] = 0
    panel.create_intersection_button.click()
    assert not created[1][0].any()
    assert "intersection is empty" in panel.status.text()
    layers[1].data[:] = np.array([[[0, 1], [1, 0]]])
    panel.compare()
    assert panel.results.item(2, 1).text() == "1"
    assert panel.results.item(2, 2).text() == "1"
    assert panel.results.item(8, 2).text() == "33.33%"
    widget._entries["c/shared"].tags = ("TH", "CP")
    panel.refresh()
    panel.compare()
    assert "membership has changed" in panel.status.text()
    assert panel.results.item(2, 2).text() == "N/A"
    panel.create_intersection_button.click()
    assert "membership has changed" in panel.status.text()
    assert len(created) == 2
    panel.close()
