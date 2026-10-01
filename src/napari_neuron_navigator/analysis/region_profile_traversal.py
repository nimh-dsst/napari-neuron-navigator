"""Exact atlas-voxel allocation for regional-profile cable segments.

The regional profile assigns child-parent cable to every annotation voxel it
crosses.  This module contains the geometry kernel only; profile aggregation
and persistence deliberately live elsewhere.  Atlas voxels use half-open
physical bounds ``[0, shape * resolution)`` on each axis.  Portions outside
those bounds are retained with region id ``0``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AtlasAxisSide(str, Enum):
    """Physical side of the atlas left-right coordinate axis."""

    LOWER = "lower"
    UPPER = "upper"
    MIDLINE = "midline"


@dataclass(frozen=True)
class VoxelSegmentPortion:
    """One exact, constant-region portion of a straight cable segment."""

    t_start: float
    t_stop: float
    length_um: float
    region_id: int
    axis_side: AtlasAxisSide
    voxel_index: tuple[int, int, int] | None


@dataclass(frozen=True)
class AtlasVoxelTraverser:
    """Validated atlas geometry reusable across many cable segments."""

    annotation: NDArray[np.integer]
    resolution_um: float | Sequence[float] | NDArray[np.floating]
    left_right_axis: int
    midline_um: float

    def __post_init__(self) -> None:
        volume = np.asarray(self.annotation)
        if volume.ndim != 3:
            raise ValueError("annotation must be a three-dimensional array")
        if not np.issubdtype(volume.dtype, np.integer):
            raise ValueError("annotation labels must have an integer dtype")
        axis = int(self.left_right_axis)
        if not 0 <= axis < 3:
            raise ValueError("left_right_axis must be 0, 1, or 2")
        midline = float(self.midline_um)
        if not math.isfinite(midline):
            raise ValueError("midline_um must be finite")
        object.__setattr__(self, "annotation", volume)
        object.__setattr__(
            self, "resolution_um", _resolution_vector(self.resolution_um)
        )
        object.__setattr__(self, "left_right_axis", axis)
        object.__setattr__(self, "midline_um", midline)

    def traverse(
        self,
        start_um: Sequence[float] | NDArray[np.floating],
        stop_um: Sequence[float] | NDArray[np.floating],
    ) -> tuple[VoxelSegmentPortion, ...]:
        """Validate and split one segment using the prepared atlas geometry."""
        start = _coordinate_vector(start_um, name="start_um")
        stop = _coordinate_vector(stop_um, name="stop_um")
        return self.traverse_validated(start, stop)

    def traverse_validated(
        self,
        start_um: NDArray[np.float64],
        stop_um: NDArray[np.float64],
    ) -> tuple[VoxelSegmentPortion, ...]:
        """Split finite ``(3,)`` float arrays without repeating atlas checks."""
        return _traverse_prepared(
            start_um,
            stop_um,
            self.annotation,
            self.resolution_um,
            left_right_axis=self.left_right_axis,
            midline_um=self.midline_um,
        )


def _coordinate_vector(
    value: Sequence[float] | NDArray[np.floating],
    *,
    name: str,
) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three coordinates")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} coordinates must all be finite")
    return vector


def _resolution_vector(
    resolution_um: float | Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float64]:
    resolution = np.asarray(resolution_um, dtype=np.float64)
    if resolution.ndim == 0:
        resolution = np.repeat(resolution, 3)
    if resolution.shape != (3,):
        raise ValueError("resolution_um must be a scalar or three values")
    if not np.all(np.isfinite(resolution)) or np.any(resolution <= 0.0):
        raise ValueError("resolution_um values must be finite and positive")
    return resolution


def _strictly_internal_plane_parameters(
    start: NDArray[np.float64],
    delta: NDArray[np.float64],
    resolution: NDArray[np.float64],
    shape: tuple[int, int, int],
) -> list[float]:
    """Return parameters where a segment crosses a finite atlas voxel plane."""
    parameters: list[float] = []
    for axis in range(3):
        step = float(delta[axis])
        if step == 0.0:
            continue

        coordinate_start = float(start[axis])
        coordinate_stop = coordinate_start + step
        lower = min(coordinate_start, coordinate_stop)
        upper = max(coordinate_start, coordinate_stop)
        voxel_size = float(resolution[axis])

        # Only planes forming the finite annotation volume are relevant.  The
        # strict coordinate check excludes a segment endpoint that lies on a
        # plane, avoiding zero-length portions without an arbitrary epsilon.
        first_plane = max(0, math.ceil(lower / voxel_size))
        last_plane = min(shape[axis], math.floor(upper / voxel_size))
        for plane_index in range(first_plane, last_plane + 1):
            plane = plane_index * voxel_size
            if lower < plane < upper:
                parameter = (plane - coordinate_start) / step
                if 0.0 < parameter < 1.0:
                    parameters.append(float(parameter))
    return parameters


def _deduplicate_parameters(parameters: list[float]) -> list[float]:
    """Sort crossing parameters and merge roundoff at edge/corner crossings."""
    parameters.sort()
    unique: list[float] = []
    for parameter in parameters:
        tolerance = (
            16.0 * max(math.ulp(parameter), math.ulp(unique[-1])) if unique else 0.0
        )
        if not unique or parameter - unique[-1] > tolerance:
            unique.append(parameter)
        else:
            # An exact corner should have one canonical boundary.  Averaging
            # near-equal values keeps the choice symmetric across axes.
            unique[-1] = (unique[-1] + parameter) / 2.0
    return unique


def _axis_side(
    coordinate: float,
    *,
    segment_on_midline: bool,
    midline_um: float,
) -> AtlasAxisSide:
    if segment_on_midline:
        return AtlasAxisSide.MIDLINE
    if coordinate < midline_um:
        return AtlasAxisSide.LOWER
    return AtlasAxisSide.UPPER


def traverse_atlas_voxels(
    start_um: Sequence[float] | NDArray[np.floating],
    stop_um: Sequence[float] | NDArray[np.floating],
    annotation: NDArray[np.integer],
    resolution_um: float | Sequence[float] | NDArray[np.floating],
    *,
    left_right_axis: int,
    midline_um: float,
) -> tuple[VoxelSegmentPortion, ...]:
    """Split a straight segment at every atlas voxel and midline crossing.

    Parameters
    ----------
    start_um, stop_um
        Segment endpoints in atlas-axis order and microns.
    annotation
        Three-dimensional direct-region annotation.  Label ``0`` is retained
        as outside/unmapped, as are portions beyond the annotation bounds.
    resolution_um
        Atlas voxel size in microns, either isotropic or one value per axis.
    left_right_axis
        Coordinate axis used to split physical sides.
    midline_um
        Physical midline on ``left_right_axis`` in microns.

    Returns
    -------
    tuple[VoxelSegmentPortion, ...]
        Ordered positive-length portions whose lengths conserve the Euclidean
        endpoint distance.  A zero-length edge returns an empty tuple.

    Notes
    -----
    Boundaries follow half-open array geometry: a point on an internal voxel
    plane belongs to the voxel on the positive side, and a point on the upper
    outer plane is outside the annotation.  Node-region lookup remains the
    responsibility of the annotated source Parquet; this convention applies
    only to physical cable allocation.
    """
    traverser = AtlasVoxelTraverser(
        annotation=annotation,
        resolution_um=resolution_um,
        left_right_axis=left_right_axis,
        midline_um=midline_um,
    )
    return traverser.traverse(start_um, stop_um)


def _traverse_prepared(
    start: NDArray[np.float64],
    stop: NDArray[np.float64],
    volume: NDArray[np.integer],
    resolution: NDArray[np.float64],
    *,
    left_right_axis: int,
    midline_um: float,
) -> tuple[VoxelSegmentPortion, ...]:
    """Split one segment after atlas geometry and coordinates are validated."""
    delta = stop - start
    total_length = float(np.linalg.norm(delta))
    if total_length == 0.0:
        return ()

    shape = tuple(int(size) for size in volume.shape)
    parameters = [0.0, 1.0]
    parameters.extend(
        _strictly_internal_plane_parameters(start, delta, resolution, shape)
    )

    lr_axis = int(left_right_axis)
    lr_delta = float(delta[lr_axis])
    segment_on_midline = float(start[lr_axis]) == float(midline_um) and lr_delta == 0.0
    if lr_delta != 0.0:
        midline_parameter = (float(midline_um) - float(start[lr_axis])) / lr_delta
        if 0.0 < midline_parameter < 1.0:
            parameters.append(float(midline_parameter))

    boundaries = _deduplicate_parameters(parameters)
    portions: list[VoxelSegmentPortion] = []
    for t_start, t_stop in pairwise(boundaries):
        if t_stop <= t_start:
            continue
        midpoint_t = (t_start + t_stop) / 2.0
        midpoint = start + midpoint_t * delta
        voxel = np.floor(midpoint / resolution).astype(np.int64)
        in_bounds = all(0 <= int(voxel[axis]) < shape[axis] for axis in range(3))
        voxel_index = tuple(int(index) for index in voxel) if in_bounds else None
        region_id = int(volume[voxel_index]) if voxel_index is not None else 0
        portions.append(
            VoxelSegmentPortion(
                t_start=float(t_start),
                t_stop=float(t_stop),
                length_um=float((t_stop - t_start) * total_length),
                region_id=region_id,
                axis_side=_axis_side(
                    float(midpoint[lr_axis]),
                    segment_on_midline=segment_on_midline,
                    midline_um=float(midline_um),
                ),
                voxel_index=voxel_index,
            )
        )
    return tuple(portions)


def sum_portion_lengths(
    portions: Sequence[VoxelSegmentPortion],
) -> float:
    """Return the numerically stable sum of allocated cable lengths."""
    return math.fsum(portion.length_um for portion in portions)


__all__ = [
    "AtlasAxisSide",
    "AtlasVoxelTraverser",
    "VoxelSegmentPortion",
    "sum_portion_lengths",
    "traverse_atlas_voxels",
]
