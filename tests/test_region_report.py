"""Cohort denominators, hierarchy unions, exclusions and laterality reports."""

from __future__ import annotations

import duckdb
import pandas as pd
import pyarrow as pa
import pytest

from napari_neuron_navigator.analysis.region_report import (
    RegionReportSelection,
    aggregate_region_report,
)


@pytest.fixture
def report_context():
    paths = {
        1: [1],
        10: [1, 10],
        11: [1, 10, 11],
        12: [1, 10, 12],
        20: [1, 20],
        21: [1, 21],
    }
    structures = {
        key: {
            "id": key,
            "name": f"Region {key}",
            "acronym": f"R{key}",
            "structure_id_path": path,
        }
        for key, path in paths.items()
    }
    # These neurons have the same display ID, in different subjects. Only their
    # file_id values distinguish them. The zero neuron has only a soma.
    a, b, zero = "subject-a/shared.swc", "subject-b/shared.swc", "zero.swc"
    rows = [
        (a, 10, "ipsilateral", "neurite", 300.0, 1, 0),
        (a, 11, "ipsilateral", "neurite", 1000.0, 2, 1),
        (a, 11, "contralateral", "neurite", 500.0, 2, 2),
        (b, 11, "ipsilateral", "neurite", 2000.0, 0, 0),  # cable only
        (a, 12, "ipsilateral", "neurite", 4000.0, 4, 3),
        (b, 12, "ipsilateral", "neurite", 0.0, 3, 1),  # nodes only
        (b, 0, "unknown", "neurite", 250.0, 1, 1),
        ("outside-cohort", 20, "ipsilateral", "neurite", 99000.0, 1, 1),
        (zero, 21, "ipsilateral", "soma", 0.0, 1, 0),
    ]
    rows.extend((file_id, 10, "ipsilateral", "soma", 0.0, 1, 0) for file_id in (a, b))
    frame = pd.DataFrame(
        rows,
        columns=[
            "file_id",
            "region_id",
            "laterality",
            "compartment",
            "cable_length_um",
            "node_count",
            "terminus_count",
        ],
    )
    frame["neuron_id"] = "shared"
    connection = duckdb.connect()
    connection.register(
        "input_profile", pa.Table.from_pandas(frame, preserve_index=False)
    )
    connection.execute(
        "CREATE TEMP TABLE region_profile AS SELECT * FROM input_profile"
    )
    try:
        yield connection, structures, (a, b, zero)
    finally:
        connection.close()


def _report(context, **kwargs):
    connection, structures, cohort = context
    return aggregate_region_report(
        connection,
        structures=structures,
        file_ids=kwargs.pop("file_ids", cohort),
        **kwargs,
    )


def _node(report, region_id):
    return next(node for node in report.nodes if node.region_id == region_id)


def _metrics(report, region_id, side="combined", *, direct=False):
    node = _node(report, region_id)
    return next(
        value
        for value in (node.direct if direct else node.aggregate)
        if value.laterality == side
    )


def test_full_cohort_denominator_includes_zero_neurons_and_cable_only(report_context):
    report = _report(report_context)
    row = _metrics(report, 11)
    assert report.cohort_count == 3
    assert row.neurons_present == 2
    assert row.cohort_present_percent == pytest.approx(200 / 3)
    assert row.total_cable_mm == pytest.approx(3.5)
    assert row.mean_cable_per_cohort_neuron_mm == pytest.approx(3.5 / 3)
    assert row.neurons_with_termini_percent == pytest.approx(100 / 3)
    assert row.total_termini == 3
    assert {node.region_id for node in report.nodes} == {0, 1, 10, 11, 12}
    assert _metrics(report, 12).neurons_present == 2  # includes the zero-cable neuron


def test_parent_unions_file_ids_across_children_and_preserves_direct_measurements(
    report_context,
):
    report = _report(report_context)
    parent = _metrics(report, 10)
    assert parent.neurons_present == 2  # child counts sum to 4; that would be wrong
    assert parent.total_cable_mm == pytest.approx(7.8)
    assert parent.total_termini == 7
    assert parent.neurons_with_termini_percent == pytest.approx(200 / 3)
    direct = _metrics(report, 10, direct=True)
    assert direct.neurons_present == 1
    assert direct.total_cable_mm == pytest.approx(0.3)
    assert _node(report, 11).parent_id == 10
    assert _node(report, 10).parent_id == 1


def test_laterality_split_keeps_cohort_denominator_and_combined_deduplicates(
    report_context,
):
    split = _report(report_context, split_laterality=True)
    assert _metrics(split, 11, "ipsilateral").neurons_present == 2
    contra = _metrics(split, 11, "contralateral")
    assert contra.neurons_present == 1
    assert contra.cohort_present_percent == pytest.approx(100 / 3)
    assert contra.mean_cable_per_cohort_neuron_mm == pytest.approx(0.5 / 3)
    assert contra.total_termini == 2
    combined = _report(report_context)
    assert _metrics(combined, 11).neurons_present == 2
    assert _metrics(combined, 11).total_cable_mm == pytest.approx(
        sum(value.total_cable_mm for value in _node(split, 11).aggregate)
    )


def test_child_exclusion_changes_parent_totals_without_changing_cohort(report_context):
    report = _report(report_context, excluded_region_ids=(11,))
    assert 11 not in {node.region_id for node in report.nodes}
    assert 11 in report.available_region_ids  # can be restored in the picker
    assert report.cohort_count == 3
    parent = _metrics(report, 10)
    assert parent.total_cable_mm == pytest.approx(4.3)
    assert parent.mean_cable_per_cohort_neuron_mm == pytest.approx(4.3 / 3)
    assert parent.neurons_present == 2
    assert parent.total_termini == 4


def test_parent_exclusion_removes_descendants_and_overlapping_exclusions_are_idempotent(
    report_context,
):
    report = _report(report_context, excluded_region_ids=(10,))
    overlap = _report(report_context, excluded_region_ids=(10, 11, 12, 10))
    assert report == overlap
    assert [node.region_id for node in report.nodes] == [0]
    assert report.cohort_count == 3


def test_unmapped_is_separate_from_atlas_root_and_can_be_excluded(report_context):
    report = _report(report_context)
    assert _node(report, 0).parent_id is None
    assert _metrics(report, 0).total_cable_mm == pytest.approx(0.25)
    assert _metrics(report, 1).total_cable_mm == pytest.approx(7.8)
    excluded = _report(report_context, excluded_region_ids=(0, 1))
    assert excluded.nodes == ()
    assert excluded.cohort_count == 3


def test_selected_rows_deduplicate_file_ids_and_keep_distinct_neurons_with_shared_display_id(
    report_context,
):
    a, b, zero = report_context[2]
    report = _report(report_context, file_ids=(a, a, b, zero))
    assert report.cohort_count == 3
    assert _metrics(report, 11).neurons_present == 2
    selected = _report(report_context, file_ids=(a,))
    assert selected.cohort_count == 1
    assert _metrics(selected, 11).total_cable_mm == pytest.approx(1.5)
    assert _metrics(selected, 11).cohort_present_percent == 100


def test_empty_and_soma_only_cohorts_have_no_report_rows(report_context):
    assert _report(report_context, file_ids=()).cohort_count == 0
    zero = report_context[2][2]
    report = _report(report_context, file_ids=(zero,))
    assert report.cohort_count == 1
    assert report.nodes == ()
    assert report.available_region_ids == frozenset()


def test_selected_rows_match_file_ids_including_cable_only_contributors(report_context):
    report = _report(report_context)
    a, b, zero = report_context[2]
    matches = report.match_rows((RegionReportSelection(11),))
    assert matches[0].label == "R11"
    assert matches[0].file_ids == {a, b}
    assert zero not in matches[0].file_ids


def test_selected_parent_and_direct_row_have_different_memberships(report_context):
    report = _report(report_context)
    a, b, _ = report_context[2]
    parent, direct = report.match_rows(
        (
            RegionReportSelection(10),
            RegionReportSelection(10, include_descendants=False),
        )
    )
    assert parent.file_ids == {a, b}
    assert direct.label == "R10 [direct]"
    assert direct.file_ids == {a}


def test_selected_rows_respect_laterality_and_exclusions(report_context):
    a, b, _ = report_context[2]
    split = _report(report_context, split_laterality=True)
    matches = split.match_rows(
        (
            RegionReportSelection(11, "contralateral"),
            RegionReportSelection(0, "unknown"),
        )
    )
    assert {match.label: match.file_ids for match in matches} == {
        "R11 (contralateral)": frozenset({a}),
        "unmapped (unknown)": frozenset({b}),
    }
    excluded = _report(report_context, excluded_region_ids=(11, 12))
    assert excluded.match_rows((RegionReportSelection(10),))[0].file_ids == {a}


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("exclusions", [(), (11,), (10,)])
def test_report_membership_matches_displayed_neuron_counts(
    report_context, split, exclusions
):
    report = _report(
        report_context, split_laterality=split, excluded_region_ids=exclusions
    )
    for node in report.nodes:
        for row in node.aggregate:
            match = report.match_rows(
                (RegionReportSelection(node.region_id, row.laterality),)
            )[0]
            assert len(match.file_ids) == row.neurons_present
        for row in node.direct:
            match = report.match_rows(
                (
                    RegionReportSelection(
                        node.region_id, row.laterality, include_descendants=False
                    ),
                )
            )[0]
            assert len(match.file_ids) == row.neurons_present


def test_empty_duplicate_and_stale_report_selections(report_context):
    report = _report(report_context)
    assert report.match_rows(()) == ()
    selection = RegionReportSelection(11)
    assert len(report.match_rows((selection, selection))) == 1
    with pytest.raises(ValueError, match="region is no longer available"):
        report.match_rows((RegionReportSelection(20),))
    with pytest.raises(ValueError, match="laterality is no longer available"):
        report.match_rows((RegionReportSelection(11, "midline"),))


def test_missing_cohort_members_and_foreign_exclusions_are_actionable(report_context):
    with pytest.raises(ValueError, match="missing from the regional profile"):
        _report(report_context, file_ids=("stale.swc",))
    with pytest.raises(ValueError, match="absent from the atlas"):
        _report(report_context, excluded_region_ids=(999,))
    with pytest.raises(ValueError, match="empty file_id"):
        _report(report_context, file_ids=(None,))


def test_repeated_reports_release_temporary_relations(report_context):
    _report(report_context)
    _report(report_context, split_laterality=True)
    tables = {row[0] for row in report_context[0].execute("SHOW TABLES").fetchall()}
    assert (
        not {
            "_report_cohort",
            "_report_closure",
            "_report_excluded",
            "_report_neurites",
        }
        & tables
    )
