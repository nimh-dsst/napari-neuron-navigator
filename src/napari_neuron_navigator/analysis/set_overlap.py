"""Spatial overlap of thresholded set heatmaps on a common voxel grid."""

from __future__ import annotations

import numpy as np


def source_set_metadata(layers):
    """Propagate a set only when every input represents the same snapshot."""
    snapshots = [getattr(layer, "metadata", {}).get("neuron_set") for layer in layers]
    if snapshots and snapshots[0] and all(s == snapshots[0] for s in snapshots):
        return {"neuron_set": snapshots[0]}
    return {}


def _transform(layer):
    # data_to_world includes scale, translation, rotation, shear and extra affine.
    transform = getattr(layer, "data_to_world", None)
    if callable(transform):
        points = np.vstack([np.zeros(3), np.eye(3)])
        return np.asarray([transform(point) for point in points], dtype=float)
    scale = np.asarray(getattr(layer, "scale", np.ones(3)), dtype=float)
    translate = np.asarray(getattr(layer, "translate", np.zeros(3)), dtype=float)
    return np.vstack([translate, np.diag(scale) + translate])


def validate_spatial_grids(layers):
    """Voxelwise operations require equal shape and the full world transform."""
    first, *rest = layers
    shape = np.shape(first.data)
    if len(shape) != 3 or any(np.shape(layer.data) != shape for layer in rest):
        raise ValueError("Regions must have the same 3D voxel-grid shape.")
    transform = _transform(first)
    if (
        not np.isfinite(transform).all()
        or np.linalg.matrix_rank(transform[1:] - transform[0]) != 3
    ):
        raise ValueError("Regions need a finite, invertible spatial transform.")
    if any(
        not np.allclose(transform, _transform(layer), rtol=0, atol=1e-8)
        for layer in rest
    ):
        raise ValueError("Regions have different spatial transforms; align them first.")


def _validated_set_regions(layer_a, layer_b, *, set_a, set_b, atlas_name):
    """Check the current source cohorts and grid before using mask data."""
    for layer, current in ((layer_a, set_a), (layer_b, set_b)):
        metadata = layer.metadata
        if not metadata.get("mask_query_source"):
            raise ValueError("Choose thresholded mask layers from Histogram.")
        if not atlas_name or metadata.get("atlas_name") != atlas_name:
            raise ValueError("Both regions must use the currently loaded atlas.")
        snapshot = metadata.get("neuron_set") or {}
        if snapshot.get("set_id") != current["set_id"]:
            raise ValueError("The region does not belong to the chosen neuron set.")
        if set(snapshot.get("file_ids", [])) != set(current["file_ids"]):
            raise ValueError(
                "Set membership has changed. Recreate its heatmap, blur, and mask."
            )
        actual_ids = metadata.get("source_file_ids", metadata.get("file_ids"))
        if actual_ids is None or set(map(str, actual_ids)) != set(current["file_ids"]):
            raise ValueError("The region's source neurons do not match the chosen set.")
    validate_spatial_grids([layer_a, layer_b])
    return np.asarray(layer_a.data), np.asarray(layer_b.data)


def build_set_intersection_mask(layer_a, layer_b, *, set_a, set_b, atlas_name):
    """Return a binary mask of shared nonzero voxels without changing inputs."""
    a, b = _validated_set_regions(
        layer_a, layer_b, set_a=set_a, set_b=set_b, atlas_name=atlas_name
    )
    mask = np.empty(a.shape, dtype=np.uint8)
    for start in range(0, a.shape[0], 8):
        slab = slice(start, start + 8)
        mask[slab] = (a[slab] != 0) & (b[slab] != 0)
    return mask


def compare_set_regions(layer_a, layer_b, *, set_a, set_b, atlas_name):
    """Compare binary regions; reject stale cohorts and incompatible coordinates.

    ``set_a`` and ``set_b`` contain set_id and the current file_ids. Counts are
    accumulated in slabs to avoid allocating full-size boolean atlas copies.
    Empty denominators return None, never an invented perfect agreement.
    """
    a, b = _validated_set_regions(
        layer_a, layer_b, set_a=set_a, set_b=set_b, atlas_name=atlas_name
    )
    count_a = count_b = shared = 0
    for start in range(0, a.shape[0], 8):
        left = a[start : start + 8] != 0
        right = b[start : start + 8] != 0
        count_a += int(np.count_nonzero(left))
        count_b += int(np.count_nonzero(right))
        shared += int(np.count_nonzero(left & right))
    union = count_a + count_b - shared
    return {
        "a": count_a,
        "b": count_b,
        "intersection": shared,
        "union": union,
        "a_only": count_a - shared,
        "b_only": count_b - shared,
        "jaccard": shared / union if union else None,
        "dice": 2 * shared / (count_a + count_b) if count_a + count_b else None,
        "fraction_a": shared / count_a if count_a else None,
        "fraction_b": shared / count_b if count_b else None,
    }
