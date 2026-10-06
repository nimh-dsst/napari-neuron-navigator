"""Brain region mapping using BrainGlobe Atlas API.

This module provides functionality to:
1. Load annotation volumes from BrainGlobe atlases
2. Map coordinates to brain regions using the Allen CCF
3. Get hierarchical region information from the structure tree
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Allen CCF resolution options in microns
RESOLUTIONS = [10, 25, 50, 100]

# Mapping from resolution to atlas name
ATLAS_NAMES = {
    10: "allen_mouse_10um",
    25: "allen_mouse_25um",
    50: "allen_mouse_50um",
    100: "allen_mouse_100um",
}


def get_dataset_region_structure_ids(
    atlas: BrainGlobeAtlas | None,
    region_ids: Iterable[int],
) -> set[int]:
    """Return represented atlas regions and their ancestors for a selector tree.

    Parquet region IDs may contain only leaves. Include their hierarchy paths
    so parent regions remain visible and selectable, without adding unrelated
    atlas branches. Unknown dataset IDs and invalid path entries are ignored.
    """
    if atlas is None:
        return set()

    allowed_ids: set[int] = set()
    for region_id in region_ids:
        struct = atlas.structures.get(int(region_id))
        if struct is None:
            continue
        allowed_ids.add(int(region_id))
        for path_id in struct.get("structure_id_path", []) or []:
            try:
                allowed_ids.add(int(path_id))
            except (TypeError, ValueError):
                continue
    return allowed_ids


def setup_allen_sdk(
    resolution: int = 25,
    cache_dir: Path | str | None = None,
) -> tuple:
    """Load annotation volume and structure data using BrainGlobe Atlas API.

    This function provides a compatible interface with the original Allen SDK
    setup, but uses BrainGlobe Atlas API which has better dependency compatibility.

    Parameters
    ----------
    resolution : int, default=25
        Resolution in microns. Must be one of [10, 25, 50, 100].
    cache_dir : Path or str, optional
        Not used (BrainGlobe manages its own cache). Kept for API compatibility.

    Returns
    -------
    tuple
        (atlas, annotation_volume, structure_tree) where:
        - atlas: BrainGlobeAtlas instance
        - annotation_volume: 3D array of region IDs
        - structure_tree: BrainGlobeStructureTree wrapper for region metadata
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Resolution must be one of {RESOLUTIONS}, got {resolution}")

    atlas_name = ATLAS_NAMES[resolution]
    logger.info(f"Loading BrainGlobe atlas: {atlas_name}")

    atlas = BrainGlobeAtlas(atlas_name)
    annotation_volume = atlas.annotation

    # Create a structure tree wrapper that provides compatible interface
    structure_tree = BrainGlobeStructureTree(atlas)

    logger.info(
        f"Annotation volume shape: {annotation_volume.shape}, "
        f"Structure tree loaded with {len(atlas.structures)} structures"
    )

    return atlas, annotation_volume, structure_tree


class BrainGlobeStructureTree:
    """Wrapper around BrainGlobe atlas structures to provide Allen SDK-like interface."""

    def __init__(self, atlas: BrainGlobeAtlas):
        self.atlas = atlas
        self._structures_by_id: dict[int, dict] = {}
        self._children: dict[int, list[int]] = {}

        # Build lookup tables - extract only the fields we need without triggering mesh loading
        for key, struct in atlas.structures.items():
            if isinstance(key, int):
                # Manually extract fields to avoid triggering mesh loading via dict()
                struct_dict = {
                    "id": key,
                    "name": struct["name"],
                    "acronym": struct["acronym"],
                    "structure_id_path": struct["structure_id_path"],
                    "rgb_triplet": struct["rgb_triplet"],
                }
                # Safely get optional fields
                if "parent_structure_id" in struct.data:
                    struct_dict["parent_structure_id"] = struct["parent_structure_id"]
                if "color_hex_triplet" in struct.data:
                    struct_dict["color_hex_triplet"] = struct["color_hex_triplet"]

                self._structures_by_id[key] = struct_dict

                # Build children mapping
                parent_id = struct_dict.get("parent_structure_id")
                if parent_id is not None:
                    if parent_id not in self._children:
                        self._children[parent_id] = []
                    self._children[parent_id].append(key)

    def get_structures_by_id(self, ids: list[int]) -> list[dict]:
        """Get structures by their IDs."""
        return [self._structures_by_id[i] for i in ids if i in self._structures_by_id]

    def get_structures_by_set_id(self, set_ids: list) -> list[dict]:
        """Get all structures (set_ids is ignored, returns all)."""
        return list(self._structures_by_id.values())

    def child_ids(self, parent_ids: list[int]) -> list[list[int]]:
        """Get child IDs for each parent ID."""
        return [self._children.get(pid, []) for pid in parent_ids]


@lru_cache(maxsize=4)
def _get_cached_atlas(resolution: int = 25) -> tuple:
    """Get cached atlas data. Returns same tuple as setup_allen_sdk."""
    return setup_allen_sdk(resolution)


def _coords_to_annotation_indices(
    coords: NDArray[np.float64] | tuple[float, float, float],
    resolution: int | tuple[float, float, float],
) -> NDArray[np.int_]:
    """Convert micron coordinates to atlas indices using BrainGlobe semantics.

    BrainGlobeAtlas.structure_from_coords divides by the atlas resolution and
    then applies ``int(...)`` component-wise, which truncates toward zero.
    """
    resolution_array = np.asarray(resolution, dtype=float)
    coords_array = np.asarray(coords, dtype=float)
    return np.trunc(coords_array / resolution_array).astype(int)


def get_region_at_coords(
    coords: NDArray[np.float64] | tuple[float, float, float],
    annotation_volume: NDArray[np.int32],
    structure_tree: BrainGlobeStructureTree,
    resolution: int = 25,
) -> dict | None:
    """Get brain region information for a single coordinate.

    Parameters
    ----------
    coords : array-like
        Coordinate in microns (x, y, z) or PIR format matching the atlas.
    annotation_volume : NDArray[np.int32]
        3D annotation volume.
    structure_tree : BrainGlobeStructureTree
        Structure tree for region metadata.
    resolution : int, default=25
        Resolution of the annotation volume in microns.

    Returns
    -------
    dict or None
        Region info with keys: id, name, acronym, structure_id_path, color_hex_triplet
        Returns None if coordinate is outside the brain.
    """
    # Match BrainGlobeAtlas.structure_from_coords for micron input.
    voxel_coords = _coords_to_annotation_indices(coords, resolution)

    # Check bounds
    if not all(0 <= voxel_coords[i] < annotation_volume.shape[i] for i in range(3)):
        return None

    # Get region ID from annotation volume
    region_id = annotation_volume[voxel_coords[0], voxel_coords[1], voxel_coords[2]]

    if region_id == 0:
        return None  # Outside brain

    # Get region info from structure tree
    structures = structure_tree.get_structures_by_id([int(region_id)])
    if not structures:
        return None

    structure = structures[0]
    return {
        "id": int(region_id),
        "name": structure.get("name", ""),
        "acronym": structure.get("acronym", ""),
        "structure_id_path": structure.get("structure_id_path", []),
        "color_hex_triplet": structure.get("color_hex_triplet", ""),
    }


def get_regions_for_coords(
    coords: NDArray[np.float64],
    annotation_volume: NDArray[np.int32],
    structure_tree: BrainGlobeStructureTree,
    resolution: int = 25,
) -> list[dict | None]:
    """Get brain region information for multiple coordinates.

    Parameters
    ----------
    coords : NDArray[np.float64]
        Array of coordinates in microns, shape (N, 3).
    annotation_volume : NDArray[np.int32]
        3D annotation volume.
    structure_tree : BrainGlobeStructureTree
        Structure tree for region metadata.
    resolution : int, default=25
        Resolution of the annotation volume in microns.

    Returns
    -------
    list[dict | None]
        List of region info dicts, one per coordinate.
    """
    coords = np.atleast_2d(coords)
    return [
        get_region_at_coords(coord, annotation_volume, structure_tree, resolution)
        for coord in coords
    ]


def get_region_ids_vectorized(
    coords: NDArray[np.float64],
    annotation_volume: NDArray[np.int32],
    resolution: int = 25,
) -> NDArray[np.int32]:
    """Get region IDs for multiple coordinates using vectorized operations.

    Parameters
    ----------
    coords : NDArray[np.float64]
        Array of coordinates in microns, shape (N, 3).
    annotation_volume : NDArray[np.int32]
        3D annotation volume.
    resolution : int, default=25
        Resolution of the annotation volume in microns.

    Returns
    -------
    NDArray[np.int32]
        Array of region IDs, 0 for coordinates outside the brain.
    """
    coords = np.atleast_2d(coords)
    n_coords = len(coords)

    # Match BrainGlobeAtlas.structure_from_coords for micron input.
    voxel_coords = _coords_to_annotation_indices(coords, resolution)

    # Initialize result with zeros (outside brain)
    region_ids = np.zeros(n_coords, dtype=np.int32)

    # Check bounds for all coordinates
    in_bounds = np.all(
        (voxel_coords >= 0) & (voxel_coords < np.array(annotation_volume.shape)),
        axis=1,
    )

    # Get region IDs for in-bounds coordinates
    valid_voxels = voxel_coords[in_bounds]
    if len(valid_voxels) > 0:
        region_ids[in_bounds] = annotation_volume[
            valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]
        ]

    return region_ids


def build_region_lookup(structure_tree: BrainGlobeStructureTree) -> dict[int, dict]:
    """Build a lookup table from region ID to region info.

    Parameters
    ----------
    structure_tree : BrainGlobeStructureTree
        Structure tree wrapper.

    Returns
    -------
    dict[int, dict]
        Mapping from region ID to dict with name, acronym, etc.
    """
    all_structures = structure_tree.get_structures_by_set_id([])
    return {
        s["id"]: {
            "id": s["id"],
            "name": s.get("name", ""),
            "acronym": s.get("acronym", ""),
            "structure_id_path": s.get("structure_id_path", []),
            "color_hex_triplet": s.get("color_hex_triplet", ""),
            "parent_structure_id": s.get("parent_structure_id"),
        }
        for s in all_structures
    }


def get_region_hierarchy(structure_tree: BrainGlobeStructureTree) -> dict[int, list[int]]:
    """Get the full hierarchy of regions.

    Parameters
    ----------
    structure_tree : BrainGlobeStructureTree
        Structure tree wrapper.

    Returns
    -------
    dict[int, list[int]]
        Mapping from region ID to list of child region IDs.
    """
    return structure_tree._children.copy()


def get_all_descendant_ids(
    structure_tree: BrainGlobeStructureTree,
    region_id: int,
) -> set[int]:
    """Get all descendant region IDs for a given region.

    Parameters
    ----------
    structure_tree : BrainGlobeStructureTree
        Structure tree wrapper.
    region_id : int
        The region ID to get descendants for.

    Returns
    -------
    set[int]
        Set of all descendant region IDs, including the input region.
    """
    descendants = {region_id}
    children = structure_tree.child_ids([region_id])[0]

    for child_id in children:
        descendants.update(get_all_descendant_ids(structure_tree, child_id))

    return descendants


def get_structures_with_ancestors(
    structure_tree: BrainGlobeStructureTree,
    region_ids: list[int] | set[int],
) -> dict[int, list[dict]]:
    """Get structure info with full ancestor path for multiple regions.

    Parameters
    ----------
    structure_tree : BrainGlobeStructureTree
        Structure tree wrapper.
    region_ids : list or set of int
        Region IDs to get ancestry for.

    Returns
    -------
    dict[int, list[dict]]
        Mapping from region ID to list of ancestor structures
        (from root to leaf).
    """
    all_structures = structure_tree.get_structures_by_id(list(region_ids))
    lookup = {s["id"]: s for s in all_structures}

    result = {}
    for region_id in region_ids:
        if region_id not in lookup:
            continue
        structure = lookup[region_id]
        path_ids = structure.get("structure_id_path", [])
        ancestors = structure_tree.get_structures_by_id(path_ids)
        result[region_id] = sorted(
            ancestors, key=lambda x: len(x.get("structure_id_path", []))
        )

    return result
