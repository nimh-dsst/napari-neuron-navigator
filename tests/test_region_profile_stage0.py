"""Stage-0 contracts for exact regional-profile cable allocation."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest

from napari_neuron_navigator.analysis.region_profile_traversal import (
    AtlasAxisSide,
    sum_portion_lengths,
    traverse_atlas_voxels,
)
from napari_neuron_navigator.terminals import childless_mask
from tests.region_profile_stage0_fixtures import (
    ANISOTROPIC_RESOLUTION_UM,
    LEFT_RIGHT_AXIS,
    MIDLINE_UM,
    analytic_segment_fixtures,
    morphology_fixture_frame,
    stage0_annotation,
)


def _region_lengths(portions) -> dict[int, float]:
    totals: defaultdict[int, float] = defaultdict(float)
    for portion in portions:
        totals[portion.region_id] += portion.length_um
    return dict(totals)


def _side_lengths(portions) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for portion in portions:
        totals[portion.axis_side.value] += portion.length_um
    return dict(totals)


@pytest.mark.parametrize("fixture", analytic_segment_fixtures(), ids=lambda item: item.name)
def test_exact_traversal_matches_analytic_segment_portions(fixture) -> None:
    portions = traverse_atlas_voxels(
        fixture.start_um,
        fixture.stop_um,
        stage0_annotation(),
        ANISOTROPIC_RESOLUTION_UM,
        left_right_axis=LEFT_RIGHT_AXIS,
        midline_um=MIDLINE_UM,
    )

    assert _region_lengths(portions) == pytest.approx(
        fixture.expected_region_lengths, abs=1e-12
    )
    assert _side_lengths(portions) == pytest.approx(
        fixture.expected_side_lengths, abs=1e-12
    )
    expected_length = math.dist(fixture.start_um, fixture.stop_um)
    assert sum_portion_lengths(portions) == pytest.approx(expected_length, abs=1e-12)


def _sampled_allocation(
    start: np.ndarray,
    stop: np.ndarray,
    annotation: np.ndarray,
    *,
    sample_count: int = 50_000,
) -> dict[tuple[int, str], float]:
    """Independent fine midpoint-sampling oracle used only by this test."""
    sample_parameters = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    points = start[None, :] + sample_parameters[:, None] * (stop - start)[None, :]
    voxel_indices = np.floor(
        points / np.asarray(ANISOTROPIC_RESOLUTION_UM)[None, :]
    ).astype(np.int64)
    in_bounds = np.all(
        (voxel_indices >= 0)
        & (voxel_indices < np.asarray(annotation.shape)[None, :]),
        axis=1,
    )
    region_ids = np.zeros(sample_count, dtype=np.int32)
    region_ids[in_bounds] = annotation[tuple(voxel_indices[in_bounds].T)]
    sides = np.where(points[:, LEFT_RIGHT_AXIS] < MIDLINE_UM, "lower", "upper")
    sample_length = math.dist(start, stop) / sample_count
    totals: defaultdict[tuple[int, str], float] = defaultdict(float)
    for region_id, side in zip(region_ids.tolist(), sides.tolist(), strict=True):
        totals[(int(region_id), str(side))] += sample_length
    return dict(totals)


def test_exact_traversal_agrees_with_independent_fine_sampling_oracle() -> None:
    annotation = stage0_annotation()
    rng = np.random.default_rng(20260924)
    exact_totals: defaultdict[tuple[int, str], float] = defaultdict(float)
    sampled_totals: defaultdict[tuple[int, str], float] = defaultdict(float)
    total_length = 0.0

    for _ in range(12):
        start = rng.uniform((-12.0, -8.0, -20.0), (48.0, 48.0, 140.0))
        stop = rng.uniform((-12.0, -8.0, -20.0), (48.0, 48.0, 140.0))
        portions = traverse_atlas_voxels(
            start,
            stop,
            annotation,
            ANISOTROPIC_RESOLUTION_UM,
            left_right_axis=LEFT_RIGHT_AXIS,
            midline_um=MIDLINE_UM,
        )
        length = math.dist(start, stop)
        total_length += length
        assert sum_portion_lengths(portions) == pytest.approx(length, abs=2e-12)
        for portion in portions:
            exact_totals[(portion.region_id, portion.axis_side.value)] += (
                portion.length_um
            )
        for bucket, bucket_length in _sampled_allocation(
            start, stop, annotation
        ).items():
            sampled_totals[bucket] += bucket_length

    # Midpoint sampling can miss at most a small fraction near each exact
    # crossing.  It is deliberately independent of the traversal breakpoints.
    for bucket in exact_totals.keys() | sampled_totals.keys():
        assert sampled_totals[bucket] == pytest.approx(
            exact_totals[bucket], abs=total_length * 2e-4
        )


def test_traversal_is_symmetric_when_edge_direction_is_reversed() -> None:
    annotation = stage0_annotation()
    start = (-5.0, 5.0, 10.0)
    stop = (45.0, 35.0, 110.0)
    forward = traverse_atlas_voxels(
        start,
        stop,
        annotation,
        ANISOTROPIC_RESOLUTION_UM,
        left_right_axis=LEFT_RIGHT_AXIS,
        midline_um=MIDLINE_UM,
    )
    reverse = traverse_atlas_voxels(
        stop,
        start,
        annotation,
        ANISOTROPIC_RESOLUTION_UM,
        left_right_axis=LEFT_RIGHT_AXIS,
        midline_um=MIDLINE_UM,
    )

    assert _region_lengths(forward) == pytest.approx(_region_lengths(reverse))
    assert _side_lengths(forward) == pytest.approx(_side_lengths(reverse))
    assert [portion.voxel_index for portion in forward] == [
        portion.voxel_index for portion in reversed(reverse)
    ]


def test_segment_on_midline_remains_explicit() -> None:
    portions = traverse_atlas_voxels(
        (2.0, 2.0, MIDLINE_UM),
        (28.0, 2.0, MIDLINE_UM),
        stage0_annotation(),
        ANISOTROPIC_RESOLUTION_UM,
        left_right_axis=LEFT_RIGHT_AXIS,
        midline_um=MIDLINE_UM,
    )

    assert {portion.axis_side for portion in portions} == {AtlasAxisSide.MIDLINE}
    assert sum_portion_lengths(portions) == pytest.approx(26.0)


def test_zero_length_segment_has_no_cable_portions() -> None:
    assert traverse_atlas_voxels(
        (2.0, 2.0, 2.0),
        (2.0, 2.0, 2.0),
        stage0_annotation(),
        ANISOTROPIC_RESOLUTION_UM,
        left_right_axis=LEFT_RIGHT_AXIS,
        midline_um=MIDLINE_UM,
    ) == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_um": (math.nan, 0.0, 0.0)}, "start_um coordinates"),
        ({"resolution_um": (10.0, 0.0, 30.0)}, "finite and positive"),
        ({"annotation": stage0_annotation().astype(float)}, "integer dtype"),
        ({"left_right_axis": 3}, "0, 1, or 2"),
        ({"midline_um": math.inf}, "midline_um must be finite"),
    ],
)
def test_traversal_rejects_invalid_geometry(kwargs, message: str) -> None:
    arguments = {
        "start_um": (0.0, 0.0, 0.0),
        "stop_um": (1.0, 1.0, 1.0),
        "annotation": stage0_annotation(),
        "resolution_um": ANISOTROPIC_RESOLUTION_UM,
        "left_right_axis": LEFT_RIGHT_AXIS,
        "midline_um": MIDLINE_UM,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        traverse_atlas_voxels(**arguments)


def _relative_profile_for_file(frame, file_id: str) -> dict[tuple[int, str], float]:
    neuron = frame.loc[frame["file_id"] == file_id]
    soma_rows = neuron.loc[neuron["type"] == 1]
    assert len(soma_rows) == 1
    soma_side = "lower" if float(soma_rows.iloc[0]["z"]) < MIDLINE_UM else "upper"
    nodes = neuron.set_index("node_id", drop=False)
    totals: defaultdict[tuple[int, str], float] = defaultdict(float)
    for child in neuron.itertuples(index=False):
        if child.parent_id == -1:
            continue
        parent = nodes.loc[child.parent_id]
        portions = traverse_atlas_voxels(
            (child.x, child.y, child.z),
            (parent.x, parent.y, parent.z),
            stage0_annotation(),
            ANISOTROPIC_RESOLUTION_UM,
            left_right_axis=LEFT_RIGHT_AXIS,
            midline_um=MIDLINE_UM,
        )
        for portion in portions:
            relative_side = (
                "ipsilateral"
                if portion.axis_side.value == soma_side
                else "contralateral"
            )
            totals[(portion.region_id, relative_side)] += portion.length_um
    return dict(totals)


def test_mirror_pair_has_identical_relative_laterality_aggregates() -> None:
    frame = morphology_fixture_frame()

    assert _relative_profile_for_file(frame, "mirror-lower.swc") == pytest.approx(
        _relative_profile_for_file(frame, "mirror-upper.swc")
    )


def test_fixture_keeps_duplicate_identifiers_separate_and_topology_complete() -> None:
    frame = morphology_fixture_frame()
    assert frame["neuron_id"].nunique() == 1
    assert frame["file_id"].nunique() == 2

    termini: dict[str, list[int]] = {}
    for file_id, neuron in frame.groupby("file_id", sort=False):
        mask = childless_mask(
            neuron["node_id"].to_numpy(), neuron["parent_id"].to_numpy()
        )
        termini[str(file_id)] = neuron.loc[mask, "node_id"].tolist()

    assert termini == {
        "mirror-lower.swc": [900],
        "mirror-upper.swc": [900],
    }
    assert set(frame.loc[frame["node_id"] == 900, "type"]) == {0}
