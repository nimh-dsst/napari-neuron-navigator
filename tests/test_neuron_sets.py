"""Live cohorts retain file identity and remain reproducible after persistence."""

import json
from types import SimpleNamespace

import pytest

from napari_neuron_navigator.neuron_sets import (
    AnnotationRule,
    NeuronSetStore,
    membership_overlap,
)


def test_live_rules_include_exclude_and_preserve_duplicate_display_ids():
    entries = {
        "subject-a/shared": SimpleNamespace(label="MOs", tags=("TH", "CP")),
        "subject-b/shared": SimpleNamespace(label="MOs", tags=("TH",)),
        "subject-c/shared": SimpleNamespace(label="MOs", tags=("CP",)),
        "other": SimpleNamespace(label="Other", tags=()),
    }
    rule = AnnotationRule(
        tag_filter=("TH", "CP"), tag_match="any", excluded_tags=("CP",)
    )
    assert rule.members(entries) == ("subject-b/shared",)
    entries["subject-a/shared"].tags = ("TH",)
    assert rule.members(entries) == ("subject-a/shared", "subject-b/shared")
    entries["added"] = SimpleNamespace(label="MOs", tags=("TH",))
    del entries["subject-b/shared"]
    assert rule.members(entries) == ("added", "subject-a/shared")
    assert AnnotationRule(tag_filter="").members(entries) == ("other",)
    assert AnnotationRule(
        tag_filter=("TH",), tag_match="exact", label_filter="MOs"
    ).members(entries) == rule.members(entries)


def test_store_roundtrip_update_delete_and_name_validation():
    store = NeuronSetStore()
    item = store.save("TH set", AnnotationRule(tag_filter=("TH",)))
    state = json.loads(json.dumps(store.to_state()))
    restored = NeuronSetStore()
    restored.load_state(state)
    assert restored.sets() == (item,)
    with pytest.raises(ValueError, match="already exists"):
        restored.save("th SET", AnnotationRule())
    with pytest.raises(ValueError, match="nonempty"):
        restored.save(" ", AnnotationRule())
    changed = restored.save(
        "Updated", AnnotationRule(excluded_tags=("CP",)), set_id=item.set_id
    )
    assert changed.set_id == item.set_id
    assert len(restored.sets()) == 1
    restored.remove(item.set_id)
    assert restored.sets() == ()
    restored.load_state({})
    assert restored.sets() == ()


def test_membership_overlap_counts_ids_once_and_empty_rates_are_undefined():
    result = membership_overlap(
        ["a/shared", "b/shared", "a/shared"], ["b/shared", "c/shared"]
    )
    assert result == {
        "a": 2,
        "b": 2,
        "intersection": 1,
        "union": 3,
        "a_only": 1,
        "b_only": 1,
        "jaccard": 1 / 3,
        "dice": 0.5,
        "fraction_a": 0.5,
        "fraction_b": 0.5,
    }
    assert membership_overlap([], ["x"])["jaccard"] == 0
    assert membership_overlap([], [])["jaccard"] is None
    assert membership_overlap([], [])["dice"] is None


@pytest.mark.parametrize("rule", [{"tag_match": "bad"}, {"tag_count": "bad"}])
def test_invalid_rule_rejected_even_without_members(rule):
    with pytest.raises(ValueError):
        AnnotationRule(**rule)
