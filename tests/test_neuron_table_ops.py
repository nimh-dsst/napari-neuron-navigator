"""Tests for pure neuron-table helper logic."""

import pytest

from napari_neuron_navigator.neuron_palette import neuron_palette
from napari_neuron_navigator.neuron_table_ops import (
    GRAY_RGBA,
    ClusterFilterSelection,
    NeuronTableSummary,
    added_flags,
    cluster_filter_matches,
    cluster_ids_available,
    cluster_sort_value,
    distinct_colors_for_file_ids,
    has_unclustered_entries,
    label_sort_key,
    recolor_cluster_distinct,
    summarize_neuron_table,
    visibility_for_selected_cluster,
)


def test_added_flags_transition() -> None:
    """Added flags track current scene membership only."""
    file_ids = ["n1", "n2", "n3"]

    initial = added_flags(file_ids, set())
    after_add = added_flags(file_ids, {"n1", "n3"})
    after_clear = added_flags(file_ids, set())

    assert initial == {"n1": False, "n2": False, "n3": False}
    assert after_add == {"n1": True, "n2": False, "n3": True}
    assert after_clear == {"n1": False, "n2": False, "n3": False}


def test_added_flags_tolerates_mixed_id_types() -> None:
    """Added flags match even when IDs differ only by Python type."""
    file_ids = [1, 2, 3]
    in_scene = {"1", "3"}

    result = added_flags(file_ids, in_scene)

    assert result == {1: True, 2: False, 3: True}


def test_cluster_sort_is_numeric() -> None:
    """Cluster sorting uses numeric values (e.g. -1, 2, 10)."""
    values = [10, -1, 2, None]
    sorted_values = sorted(values, key=cluster_sort_value)
    assert sorted_values == [-1, 2, 10, None]


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (
            ["Rank 100", "Rank 2", "Rank 11", "Rank 1", "Rank 99", "Rank 10", "Rank 9"],
            ["Rank 1", "Rank 2", "Rank 9", "Rank 10", "Rank 11", "Rank 99", "Rank 100"],
        ),
        (
            ["Sample 10", "", "Sample 2", "reference", "2", "10", "Alpha"],
            ["", "2", "10", "Alpha", "reference", "Sample 2", "Sample 10"],
        ),
        (
            ["sample 2", "Sample 2", "Sample 02", "sample 10", "Sample 002"],
            ["Sample 002", "Sample 02", "Sample 2", "sample 2", "sample 10"],
        ),
        (
            ["Sample 10 part 1", "Sample 2 part 10", "Sample 2 part 2"],
            ["Sample 2 part 2", "Sample 2 part 10", "Sample 10 part 1"],
        ),
    ],
)
def test_label_sort_uses_numbers_and_preserves_text(labels, expected) -> None:
    assert sorted(labels, key=label_sort_key) == expected
    assert sorted(labels, key=label_sort_key, reverse=True) == expected[::-1]


def test_cluster_filter_matches() -> None:
    """Filtering returns only selected cluster rows, or all for None."""
    cluster_by_file = {"n1": 1, "n2": 2, "n3": 1}
    only_cluster_1 = cluster_filter_matches(cluster_by_file, 1)
    all_rows = cluster_filter_matches(cluster_by_file, None)

    assert only_cluster_1 == {"n1": True, "n2": False, "n3": True}
    assert all(all_rows.values())


def test_cluster_filter_selection_matches_multiple_groups() -> None:
    """Filtering supports multiple clusters plus unclustered rows."""
    cluster_by_file = {"n1": 1, "n2": 2, "n3": None, "n4": 3}
    selection = ClusterFilterSelection({1, 3}, include_unclustered=True)

    matches = cluster_filter_matches(cluster_by_file, selection)

    assert matches == {"n1": True, "n2": False, "n3": True, "n4": True}
    assert selection.is_all is False


def test_cluster_filter_selection_all_rows_when_empty() -> None:
    """An empty explicit selection means all rows."""
    cluster_by_file = {"n1": 1, "n2": None}
    selection = ClusterFilterSelection()

    matches = cluster_filter_matches(cluster_by_file, selection)

    assert selection.is_all is True
    assert matches == {"n1": True, "n2": True}


def test_visibility_for_selected_cluster_and_show_all() -> None:
    """Visibility map hides non-cluster members and restores all for None."""
    cluster_by_file = {"n1": 1, "n2": 2, "n3": 1}
    hidden = visibility_for_selected_cluster(cluster_by_file, 1)
    shown = visibility_for_selected_cluster(cluster_by_file, None)

    assert hidden == {"n1": True, "n2": False, "n3": True}
    assert shown == {"n1": True, "n2": True, "n3": True}


def test_visibility_for_cluster_selection_with_unclustered() -> None:
    """Visibility can keep selected clusters and unclustered rows visible."""
    cluster_by_file = {"n1": 1, "n2": 2, "n3": None, "n4": 3}
    selection = ClusterFilterSelection({2}, include_unclustered=True)

    visibility = visibility_for_selected_cluster(cluster_by_file, selection)

    assert visibility == {"n1": False, "n2": True, "n3": True, "n4": False}


def test_recolor_cluster_distinct_grays_others() -> None:
    """Selected cluster gets palette colors while non-selected become gray."""
    cluster_by_file = {"n10": 2, "n2": 1, "n1": 1}
    colors = recolor_cluster_distinct(cluster_by_file, 1, gray_others=True)

    # Deterministic order is by file_id ascending within the selected cluster.
    expected = neuron_palette(2)
    assert colors["n1"] == expected[0]
    assert colors["n2"] == expected[1]
    assert colors["n10"] == list(GRAY_RGBA)


def test_recolor_cluster_distinct_handles_multi_selection_and_unclustered() -> None:
    """Recoloring draws one palette across all selected groups together."""
    cluster_by_file = {"n3": None, "n2": 2, "n1": 1, "n4": 3}
    selection = ClusterFilterSelection({1, 2}, include_unclustered=True)

    colors = recolor_cluster_distinct(cluster_by_file, selection, gray_others=True)

    for file_id, expected_color in zip(["n1", "n2", "n3"], neuron_palette(3)):
        assert colors[file_id] == expected_color
    assert colors["n4"] == list(GRAY_RGBA)


def test_distinct_colors_for_file_ids_is_stable_and_deduplicates() -> None:
    colors = distinct_colors_for_file_ids(["n2", "n10", "n2", "n1"])

    expected_ids = ["n1", "n10", "n2"]
    assert list(colors) == expected_ids
    for file_id, expected_color in zip(expected_ids, neuron_palette(3)):
        assert colors[file_id] == expected_color


def test_distinct_colors_for_file_ids_handles_empty_and_singleton_inputs() -> None:
    assert distinct_colors_for_file_ids([]) == {}
    assert distinct_colors_for_file_ids(["only"]) == {"only": neuron_palette(1)[0]}


def test_recolor_cluster_distinct_uses_the_shared_palette() -> None:
    cluster_by_file = {"n3": 2, "n2": 1, "n1": 1}

    colors = recolor_cluster_distinct(cluster_by_file, 1, gray_others=False)

    assert colors == distinct_colors_for_file_ids(["n1", "n2"])


def test_available_cluster_ids_sorted_unique() -> None:
    """Cluster IDs are unique and sorted ascending, excluding None."""
    cluster_by_file = {"a": 3, "b": None, "c": -1, "d": 3, "e": 2}
    assert cluster_ids_available(cluster_by_file) == [-1, 2, 3]


def test_has_unclustered_entries() -> None:
    """Unclustered availability detects rows without cluster assignment."""
    assert has_unclustered_entries({"a": 1, "b": None}) is True
    assert has_unclustered_entries({"a": 1, "b": 2}) is False


def test_summarize_neuron_table_empty() -> None:
    """Empty table summary reports zeros and no clusters."""
    summary = summarize_neuron_table({}, {}, {})

    assert summary == NeuronTableSummary(
        table_count=0,
        added_count=0,
        visible_count=0,
        cluster_counts=(),
    )


def test_summarize_neuron_table_tracks_counts_and_cluster_breakdown() -> None:
    """Summary reflects table membership, added flags, visibility, and clusters."""
    summary = summarize_neuron_table(
        {"n1": 2, "n2": None, "n3": 1, "n4": 2},
        {"n1": True, "n2": False, "n3": True, "n4": False},
        {"n1": True, "n2": False, "n3": True, "n4": True},
    )

    assert summary.table_count == 4
    assert summary.added_count == 2
    assert summary.visible_count == 3
    assert summary.cluster_counts == ((1, 1), (2, 2), (None, 1))
