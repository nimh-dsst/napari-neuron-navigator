"""Regional feature definitions, distances, safeguards, and durable results."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from napari_neuron_navigator.analysis import regional_clustering as module
from napari_neuron_navigator.analysis.regional_clustering import (
    RegionalClusteringCancelled,
    RegionalClusteringSettings,
    RegionalFeature,
    RegionalFeatures,
    RegionalMode,
    build_regional_features,
    cluster_regional_features,
    estimate_regional_memory,
    save_regional_cluster_result,
    save_regional_neuron_group,
)
from napari_neuron_navigator.cluster_assignments import ClusterAssignmentStore
from tests.test_region_report import report_context as report_context  # noqa: PLC0414


def _features(context, **kwargs):
    connection, structures, ids = context
    settings = RegionalClusteringSettings(
        file_ids=kwargs.pop("file_ids", ids),
        n_clusters=kwargs.pop("n_clusters", 2),
        **kwargs,
    )
    return build_regional_features(connection, structures=structures, settings=settings)


def _value(features, file_id, region_id, side="combined"):
    index = next(
        i
        for i, column in enumerate(features.columns)
        if column.region_id == region_id and column.laterality == side
    )
    return features.matrix[features.settings.file_ids.index(file_id), index]


def test_zero_filling_and_file_id_identity(report_context):
    a, b, zero = report_context[2]
    features = _features(
        report_context, laterality="combined", file_ids=(a, b, a, zero)
    )
    assert features.settings.file_ids == tuple(sorted((a, b, zero)))
    assert features.matrix.shape == (3, 4)
    assert _value(features, a, 11) == 1500
    assert _value(features, b, 11) == 2000
    assert _value(features, a, 0) == 0
    assert _value(features, b, 0) == 250
    assert not features.matrix[features.settings.file_ids.index(zero)].any()
    assert features.zero_feature_file_ids == (zero,)
    assert features.retained_file_ids == (a, b)


def test_binary_intersections_include_cable_only_and_node_only_buckets(report_context):
    a, b, _ = report_context[2]
    features = _features(
        report_context, mode=RegionalMode.REACHED, laterality="combined"
    )
    assert features.matrix.dtype == np.bool_
    assert _value(features, b, 11)  # cable > 0, node_count == 0
    assert _value(features, b, 12)  # cable == 0, node_count > 0
    assert _value(features, a, 11)
    assert not _value(features, a, 0)
    assert {column.region_id for column in features.columns} == {0, 10, 11, 12}


def test_laterality_is_separate_by_default_and_combination_is_explicit(report_context):
    a, b, _ = report_context[2]
    separate = _features(report_context)
    assert separate.settings.laterality == "separate"
    assert _value(separate, a, 11, "ipsilateral") == 1000
    assert _value(separate, a, 11, "contralateral") == 500
    assert _value(separate, b, 11, "contralateral") == 0
    assert _value(separate, b, 0, "unknown") == 250
    ipsi = _features(report_context, laterality="ipsilateral")
    assert {column.laterality for column in ipsi.columns} == {"ipsilateral"}
    combined = _features(report_context, laterality="combined")
    assert _value(combined, a, 11) == 1500


@pytest.mark.parametrize("groups", [(10,), (10, 11), (1, 10, 11)])
def test_hierarchy_partition_never_counts_parent_and_child_twice(
    report_context, groups
):
    a, b, _ = report_context[2]
    features = _features(report_context, laterality="combined", group_region_ids=groups)
    group = 1 if 1 in groups else 10
    assert _value(features, a, group) == 5800
    assert _value(features, b, group) == 2000
    assert _value(features, b, 0) == 250  # unmapped stays outside the atlas root
    allocated = [r for column in features.columns for r in column.direct_region_ids]
    assert len(allocated) == len(set(allocated))
    assert np.sum(features.matrix) == 8050


def test_partial_grouping_and_exclusion_keep_remaining_direct_contributions(
    report_context,
):
    a, b, _ = report_context[2]
    features = _features(
        report_context,
        laterality="combined",
        group_region_ids=(10,),
        excluded_region_ids=(11,),
    )
    assert _value(features, a, 10) == 4300
    assert _value(features, b, 10) == 0
    column = next(c for c in features.columns if c.region_id == 10)
    assert column.direct_region_ids == (10, 12)
    assert features.excluded_direct_region_ids == (11,)


def test_source_parent_exclusion_and_unmapped_exclusion_are_supported(report_context):
    a, b, zero = report_context[2]
    features = _features(report_context, excluded_region_ids=(10,))
    assert {column.region_id for column in features.columns} == {0}
    assert features.zero_feature_file_ids == (a, zero)
    assert features.retained_file_ids == (b,)
    empty = _features(report_context, excluded_region_ids=(1, 0))
    assert empty.matrix.shape == (3, 0)
    assert empty.zero_feature_file_ids == tuple(sorted((a, b, zero)))
    with pytest.raises(ValueError, match="0 neurons have nonzero"):
        cluster_regional_features(empty)


def test_terminus_distribution_uses_reconstructed_counts(report_context):
    a, b, _ = report_context[2]
    features = _features(
        report_context,
        mode=RegionalMode.TERMINI,
        laterality="combined",
        group_region_ids=(10,),
    )
    assert _value(features, a, 10) == 6
    assert _value(features, b, 10) == 1
    assert _value(features, b, 0) == 1


def test_grouped_regions_reached_is_binary_union_not_sum(report_context):
    a, b, _ = report_context[2]
    features = _features(
        report_context,
        mode=RegionalMode.REACHED,
        laterality="combined",
        group_region_ids=(10,),
    )
    assert _value(features, a, 10) == 1
    assert _value(features, b, 10) == 1
    assert features.matrix.max() == 1


@pytest.mark.parametrize("mode", list(RegionalMode))
def test_distances_average_linkage_and_zero_exclusion(report_context, mode):
    features = _features(report_context, mode=mode)
    result = cluster_regional_features(
        features, provenance={"source_sha256": "fixture"}
    )
    expected = pdist(
        features.matrix[features.retained_mask].astype(float),
        metric="jaccard" if mode is RegionalMode.REACHED else "cosine",
    )
    assert result.linkage_matrix[0, 2] == pytest.approx(expected[0])
    assert set(result.labels) == {1, 2}
    assert len(result.file_ids) == 2
    assert result.features.zero_feature_file_ids == (report_context[2][2],)
    assert result.metadata["linkage_method"] == "average"
    assert result.metadata["source_sha256"] == "fixture"
    assert result.metadata["external_annotations_used_for_clustering"] is False


def test_average_linkage_recovers_two_clear_patterns():
    settings = RegionalClusteringSettings(("a", "b", "c", "d", "zero"), n_clusters=2)
    matrix = np.array([[10, 0], [20, 0], [0, 5], [0, 10], [0, 0]], dtype=float)
    features = RegionalFeatures(
        settings,
        (
            RegionalFeature(11, "A", "A", "ipsilateral", (11,)),
            RegionalFeature(12, "B", "B", "ipsilateral", (12,)),
        ),
        matrix,
        ("zero",),
        (),
    )
    result = cluster_regional_features(features)
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[0] != result.labels[2]
    ordered = result.labels[result.leaf_order]
    assert np.count_nonzero(np.diff(ordered)) == 1


def test_memory_guard_runs_before_pairwise_allocation(report_context, monkeypatch):
    features = _features(report_context, memory_limit_bytes=1024**2)

    def forbidden(*args, **kwargs):
        raise AssertionError("Distances were allocated before the memory guard")

    monkeypatch.setattr(module.np, "memmap", forbidden)
    with pytest.raises(MemoryError, match="Estimated peak"):
        cluster_regional_features(features)
    estimate = estimate_regional_memory(18621)
    assert estimate.condensed_bytes == 18621 * 18620 // 2 * 8
    assert estimate.estimated_peak_bytes > 4 * estimate.condensed_bytes


def test_feature_memory_guard_and_requested_cluster_count(report_context):
    with pytest.raises(MemoryError, match="Feature matrix"):
        _features(report_context, memory_limit_bytes=1)
    features = _features(report_context, n_clusters=3)
    with pytest.raises(ValueError, match="2 neurons have nonzero"):
        cluster_regional_features(features)


def test_cancellation_before_features_and_during_distances(report_context):
    connection, structures, ids = report_context
    with pytest.raises(RegionalClusteringCancelled):
        build_regional_features(
            connection,
            structures=structures,
            settings=RegionalClusteringSettings(ids),
            cancel_check=lambda: True,
        )
    features = _features(report_context)
    cancelled = [False]

    def progress(*args):
        cancelled[0] = True

    with pytest.raises(RegionalClusteringCancelled):
        cluster_regional_features(
            features, cancel_check=lambda: cancelled[0], progress_callback=progress
        )


def test_cancellation_terminates_an_active_linkage_process(tmp_path, monkeypatch):
    class Process:
        alive = False
        terminated = False
        closed = False

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            pass

        def terminate(self):
            self.terminated = True
            self.alive = False

        def close(self):
            self.closed = True

    process = Process()
    monkeypatch.setattr(
        module.mp,
        "get_context",
        lambda *_: SimpleNamespace(Process=lambda **_: process),
    )
    with pytest.raises(RegionalClusteringCancelled):
        module._cancellable_linkage("unused", 2, tmp_path, lambda: process.alive)
    assert process.terminated and process.closed


def test_saved_result_group_and_assignment_preserve_definitions(
    report_context, tmp_path
):
    features = _features(report_context, laterality="combined", group_region_ids=(10,))
    result = cluster_regional_features(features)
    path = tmp_path / "result.npz"
    save_regional_cluster_result(path, result)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["matrix"], result.matrix)
        assert tuple(saved["file_ids"]) == result.file_ids
        metadata = json.loads(str(saved["metadata_json"]))
        assert metadata["settings"]["file_ids"] == list(features.settings.file_ids)
        assert metadata["features"][1]["direct_region_ids"] == [10, 11, 12]
        assert metadata["zero_feature_file_ids"] == [report_context[2][2]]
    group_path = tmp_path / "group.json"
    save_regional_neuron_group(group_path, result, 1, "Unlabelled morphology group")
    group = json.loads(group_path.read_text())
    assert group["file_ids"] == list(result.file_ids_for_cluster(1))
    assert group["regional_clustering"]["distance_metric"] == "cosine"
    store = ClusterAssignmentStore()
    store.add(
        name="Regional",
        assignments=dict(zip(result.file_ids, result.labels)),
        input_file_ids=features.settings.file_ids,
        unassigned_neuron_ids=features.zero_feature_file_ids,
        run_metadata=result.metadata,
        coordinate_space="Regional Profile",
    )
    restored = ClusterAssignmentStore.from_state(
        json.loads(json.dumps(store.to_state()))
    )
    assert restored.active.input_file_ids == features.settings.file_ids
    assert restored.active.unassigned_neuron_ids == features.zero_feature_file_ids
    assert restored.active.run_metadata["features"] == metadata["features"]


def test_invalid_or_foreign_cohorts_and_regions_are_rejected(report_context):
    with pytest.raises(ValueError, match="absent from the regional profile"):
        _features(report_context, file_ids=("foreign",))
    with pytest.raises(ValueError, match="absent from the compatible atlas"):
        _features(report_context, group_region_ids=(999,))
    with pytest.raises(ValueError, match="nonempty"):
        _features(report_context, file_ids=())
    with pytest.raises(ValueError, match="nonempty file_id"):
        RegionalClusteringSettings((None,))
