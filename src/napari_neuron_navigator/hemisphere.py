"""Hemisphere detection and coordinate flipping using BrainGlobe Atlas API.

This module provides functionality to:
1. Detect which hemisphere SWC coordinates are in using BrainGlobe atlases
2. Flip coordinates from one hemisphere to the other

All coordinate operations use numpy vectorized operations for efficient
processing of large SWC files (10,000+ nodes).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas

from .swc import SWCData

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Hemisphere(Enum):
    """Brain hemisphere identifiers."""

    LEFT = "left"
    RIGHT = "right"
    MIDLINE = "midline"


# Mapping from BrainGlobeAtlas hemisphere_from_coords return values to Hemisphere enum
# hemisphere_from_coords returns: 0=outside brain, 1=left, 2=right
_ATLAS_HEMISPHERE_MAP = {
    1: Hemisphere.LEFT,
    2: Hemisphere.RIGHT,
}


def _validate_hemisphere_with_atlas(
    coords: tuple[float, float, float],
    atlas: BrainGlobeAtlas,
    midline_result: Hemisphere,
) -> None:
    """Validate midline-based hemisphere detection against atlas.

    Parameters
    ----------
    coords : tuple[float, float, float]
        Coordinates in microns (x, y, z).
    atlas : BrainGlobeAtlas
        The atlas instance to validate against.
    midline_result : Hemisphere
        The hemisphere determined by midline calculation.

    Raises
    ------
    ValueError
        If the midline-based result differs from the atlas's hemisphere_from_coords.
    """
    # Get hemisphere from atlas using its built-in method
    atlas_hemisphere_code = atlas.hemisphere_from_coords(coords, microns=True)

    # 0 means outside the brain - skip validation in this case
    if atlas_hemisphere_code == 0:
        return

    atlas_hemisphere = _ATLAS_HEMISPHERE_MAP.get(atlas_hemisphere_code)

    # If midline result is MIDLINE, it could match either hemisphere at boundary
    if midline_result == Hemisphere.MIDLINE:
        return

    # Check if results match
    if atlas_hemisphere is not None and atlas_hemisphere != midline_result:
        raise ValueError(
            f"Hemisphere mismatch: midline calculation returned {midline_result.value}, "
            f"but atlas.hemisphere_from_coords returned {atlas_hemisphere.value} "
            f"(code={atlas_hemisphere_code}) for coordinates {coords}"
        )


def get_atlas_midline(atlas: BrainGlobeAtlas, coord_axis: int = 2) -> float:
    """Get the midline coordinate (in microns) for an atlas along a given axis.

    The midline is at the center of the atlas along the specified axis,
    calculated as the midpoint between the first and last voxel positions.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        A BrainGlobe atlas instance.
    coord_axis : int, default=2
        Which coordinate axis (0=x, 1=y, 2=z) to get the midline for.

    Returns
    -------
    float
        The midline coordinate in microns.
    """
    # Voxel indices go from 0 to (shape - 1)
    # The center is at (shape - 1) / 2 in voxel coordinates
    # Convert to microns by multiplying by resolution
    shape_voxels = atlas.shape[coord_axis]
    resolution_um = atlas.resolution[coord_axis]
    midline_um = ((shape_voxels - 1) * resolution_um) / 2.0
    return midline_um


def detect_hemisphere(
    coords: NDArray[np.float64],
    atlas: BrainGlobeAtlas | None = None,
    atlas_name: str = "allen_mouse_10um",
    midline: float | None = None,
    coord_axis: int = 2,
    validate: bool = True,
) -> Hemisphere:
    """Detect which hemisphere a coordinate or set of coordinates is in.

    Parameters
    ----------
    coords : NDArray[np.float64]
        Coordinates to check. Can be a single point (3,) or multiple points (N, 3).
    atlas : BrainGlobeAtlas, optional
        Pre-loaded atlas instance. If None, will load using atlas_name.
    atlas_name : str, default="allen_mouse_10um"
        Name of the BrainGlobe atlas to use if atlas is not provided.
    midline : float, optional
        Override the atlas midline with a custom value in microns.
        When provided, validation against atlas is skipped.
    coord_axis : int, default=2
        Which coordinate axis (0=x, 1=y, 2=z) corresponds to the left-right axis.
    validate : bool, default=True
        If True, validate the result against atlas.hemisphere_from_coords.
        Validation is skipped when a custom midline is provided.

    Returns
    -------
    Hemisphere
        The hemisphere the coordinates are in (LEFT, RIGHT, or MIDLINE).

    Raises
    ------
    ValueError
        If validation is enabled and the midline-based result differs from
        the atlas's hemisphere_from_coords method.

    Notes
    -----
    For multiple coordinates, returns the hemisphere of the centroid.
    Coordinates exactly at the midline return MIDLINE.
    """
    coords = np.atleast_2d(coords)
    custom_midline_provided = midline is not None

    # Get midline value
    if midline is None:
        if atlas is None:
            atlas = BrainGlobeAtlas(atlas_name)
        midline = get_atlas_midline(atlas, coord_axis)

    # Calculate mean position along left-right axis
    mean_coords = np.mean(coords, axis=0)
    mean_lr = mean_coords[coord_axis]

    # Determine hemisphere with small tolerance for midline
    tolerance = 1.0  # 1 micron tolerance
    if abs(mean_lr - midline) < tolerance:
        result = Hemisphere.MIDLINE
    elif mean_lr < midline:
        result = Hemisphere.LEFT
    else:
        result = Hemisphere.RIGHT

    # Validate against atlas if enabled and no custom midline was provided
    if validate and not custom_midline_provided and atlas is not None:
        _validate_hemisphere_with_atlas(
            tuple(mean_coords),
            atlas,
            result,
        )

    return result


def detect_soma_hemisphere(
    swc_data: SWCData,
    atlas: BrainGlobeAtlas | None = None,
    atlas_name: str = "allen_mouse_10um",
    midline: float | None = None,
    coord_axis: int = 2,
    validate: bool = True,
) -> Hemisphere:
    """Detect which hemisphere the soma of an SWC morphology is in.

    Parameters
    ----------
    swc_data : SWCData
        Parsed SWC morphology data.
    atlas : BrainGlobeAtlas, optional
        Pre-loaded atlas instance.
    atlas_name : str, default="allen_mouse_10um"
        Name of the BrainGlobe atlas to use.
    midline : float, optional
        Override the atlas midline with a custom value.
        When provided, validation against atlas is skipped.
    coord_axis : int, default=2
        Which coordinate axis corresponds to left-right.
    validate : bool, default=True
        If True, validate the result against atlas.hemisphere_from_coords.

    Returns
    -------
    Hemisphere
        The hemisphere the soma is in.

    Raises
    ------
    ValueError
        If no soma nodes are found in the SWC data, or if validation fails.
    """
    soma_coords = swc_data.soma_coords
    if len(soma_coords) == 0:
        raise ValueError("No soma nodes found in SWC data")

    return detect_hemisphere(
        soma_coords,
        atlas=atlas,
        atlas_name=atlas_name,
        midline=midline,
        coord_axis=coord_axis,
        validate=validate,
    )


def flip_coordinates(
    coords: NDArray[np.float64],
    atlas: BrainGlobeAtlas | None = None,
    atlas_name: str = "allen_mouse_10um",
    midline: float | None = None,
    coord_axis: int = 2,
) -> NDArray[np.float64]:
    """Flip coordinates from one hemisphere to the other.

    This function mirrors coordinates across the atlas midline along the
    specified axis. Uses vectorized numpy operations for efficient processing
    of large coordinate arrays.

    Parameters
    ----------
    coords : NDArray[np.float64]
        Coordinates to flip. Shape (N, 3) or (3,).
    atlas : BrainGlobeAtlas, optional
        Pre-loaded atlas instance.
    atlas_name : str, default="allen_mouse_25um"
        Name of the BrainGlobe atlas to use.
    midline : float, optional
        Override the atlas midline with a custom value.
    coord_axis : int, default=0
        Which coordinate axis (0=x, 1=y, 2=z) to flip across.

    Returns
    -------
    NDArray[np.float64]
        Flipped coordinates with the same shape as input.

    Notes
    -----
    The flip operation is: new_coord = 2 * midline - old_coord
    This mirrors the coordinate across the midline plane.

    This operation is fully vectorized and can efficiently process
    arrays with 10,000+ coordinates.
    """
    coords = np.asarray(coords, dtype=np.float64)
    original_shape = coords.shape
    coords = np.atleast_2d(coords)

    # Get midline value
    if midline is None:
        if atlas is None:
            atlas = BrainGlobeAtlas(atlas_name)
        midline = get_atlas_midline(atlas, coord_axis)

    # Vectorized flip: reflect across midline
    # new_x = midline - (old_x - midline) = 2 * midline - old_x
    flipped = coords.copy()
    flipped[:, coord_axis] = 2.0 * midline - coords[:, coord_axis]

    # Return with original shape
    if len(original_shape) == 1:
        return flipped[0]
    return flipped


def flip_swc(
    swc_data: SWCData,
    atlas: BrainGlobeAtlas | None = None,
    atlas_name: str = "allen_mouse_10um",
    midline: float | None = None,
    coord_axis: int = 2,
    in_place: bool = False,
) -> SWCData:
    """Flip all coordinates in an SWC morphology to the opposite hemisphere.

    Parameters
    ----------
    swc_data : SWCData
        The SWC morphology data to flip.
    atlas : BrainGlobeAtlas, optional
        Pre-loaded atlas instance.
    atlas_name : str, default="allen_mouse_25um"
        Name of the BrainGlobe atlas to use.
    midline : float, optional
        Override the atlas midline with a custom value.
    coord_axis : int, default=0
        Which coordinate axis to flip across.
    in_place : bool, default=False
        If True, modify the input SWCData. If False, return a copy.

    Returns
    -------
    SWCData
        SWC data with flipped coordinates.

    Notes
    -----
    This operation uses vectorized numpy operations and can efficiently
    process SWC files with 10,000+ nodes in a single array operation.
    """
    if in_place:
        result = swc_data
    else:
        result = swc_data.copy()

    # Flip all coordinates in a single vectorized operation
    result.coords = flip_coordinates(
        result.coords,
        atlas=atlas,
        atlas_name=atlas_name,
        midline=midline,
        coord_axis=coord_axis,
    )

    return result


def flip_swc_batch(
    swc_data_list: list[SWCData],
    atlas: BrainGlobeAtlas | None = None,
    atlas_name: str = "allen_mouse_10um",
    midline: float | None = None,
    coord_axis: int = 2,
) -> list[SWCData]:
    """Flip multiple SWC morphologies to the opposite hemisphere.

    This function pre-loads the atlas once and reuses it for all
    morphologies, providing better performance for batch operations.

    Parameters
    ----------
    swc_data_list : list[SWCData]
        List of SWC morphology data to flip.
    atlas : BrainGlobeAtlas, optional
        Pre-loaded atlas instance.
    atlas_name : str, default="allen_mouse_25um"
        Name of the BrainGlobe atlas to use.
    midline : float, optional
        Override the atlas midline with a custom value.
    coord_axis : int, default=0
        Which coordinate axis to flip across.

    Returns
    -------
    list[SWCData]
        List of SWC data with flipped coordinates.
    """
    # Load atlas once for all operations
    if atlas is None and midline is None:
        atlas = BrainGlobeAtlas(atlas_name)
        midline = get_atlas_midline(atlas, coord_axis)

    return [
        flip_swc(
            swc_data,
            atlas=atlas,
            midline=midline,
            coord_axis=coord_axis,
        )
        for swc_data in swc_data_list
    ]
