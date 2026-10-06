"""Deterministic regional-profile fixtures shared by staged implementation tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANISOTROPIC_RESOLUTION_UM = (10.0, 20.0, 30.0)
LEFT_RIGHT_AXIS = 2
MIDLINE_UM = 60.0


def stage0_annotation() -> np.ndarray:
    """Return a small direct-region annotation with deliberate zero voxels."""
    annotation = np.empty((4, 2, 4), dtype=np.int32)
    annotation[0, :, :] = 11
    annotation[1, :, :] = 22
    annotation[2, :, :] = 22
    annotation[3, :, :] = 33
    annotation[2, 1, 3] = 0
    return annotation


@dataclass(frozen=True)
class SegmentFixture:
    """One segment with analytically known allocation totals."""

    name: str
    start_um: tuple[float, float, float]
    stop_um: tuple[float, float, float]
    expected_region_lengths: dict[int, float]
    expected_side_lengths: dict[str, float]


def analytic_segment_fixtures() -> tuple[SegmentFixture, ...]:
    """Cover one voxel, region, midline, and annotation-bound crossings."""
    return (
        SegmentFixture(
            name="one_voxel",
            start_um=(2.0, 4.0, 6.0),
            stop_um=(8.0, 12.0, 18.0),
            expected_region_lengths={11: np.sqrt(244.0)},
            expected_side_lengths={"lower": np.sqrt(244.0)},
        ),
        SegmentFixture(
            name="region_boundary",
            start_um=(5.0, 10.0, 15.0),
            stop_um=(25.0, 10.0, 15.0),
            expected_region_lengths={11: 5.0, 22: 15.0},
            expected_side_lengths={"lower": 20.0},
        ),
        SegmentFixture(
            name="midline_crossing",
            start_um=(5.0, 10.0, 45.0),
            stop_um=(5.0, 10.0, 75.0),
            expected_region_lengths={11: 30.0},
            expected_side_lengths={"lower": 15.0, "upper": 15.0},
        ),
        SegmentFixture(
            name="partly_outside",
            start_um=(-5.0, 10.0, 15.0),
            stop_um=(15.0, 10.0, 15.0),
            expected_region_lengths={0: 5.0, 11: 10.0, 22: 5.0},
            expected_side_lengths={"lower": 20.0},
        ),
    )


def morphology_fixture_frame() -> pd.DataFrame:
    """Return trees exposing identity, ordering, type, and mirror traps.

    Both mirror files reuse ``neuron_id`` and every ``node_id``.  Rows are in
    child-before-parent order, identifiers are non-contiguous, and the final
    child changes from type 2 to type 0.  A topology implementation that
    filters by type before finding children will therefore invent a terminus at
    node 250.
    """
    rows: list[dict[str, object]] = []
    for file_id, subject, z_values in (
        ("mirror-lower.swc", "subject-lower", (25.0, 35.0, 45.0)),
        ("mirror-upper.swc", "subject-upper", (95.0, 85.0, 75.0)),
    ):
        for node_id, node_type, z, parent_id in (
            (900, 0, z_values[0], 250),
            (250, 2, z_values[1], 100),
            (100, 1, z_values[2], -1),
        ):
            rows.append(
                {
                    "file_id": file_id,
                    "neuron_id": "duplicate-display-id",
                    "subject": subject,
                    "node_id": node_id,
                    "type": node_type,
                    "x": 5.0,
                    "y": 10.0,
                    "z": z,
                    "radius": 1.0,
                    "parent_id": parent_id,
                    "region_id": 11,
                    "region_name": "Fixture region",
                    "region_acronym": "FIX",
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "ANISOTROPIC_RESOLUTION_UM",
    "LEFT_RIGHT_AXIS",
    "MIDLINE_UM",
    "SegmentFixture",
    "analytic_segment_fixtures",
    "morphology_fixture_frame",
    "stage0_annotation",
]
