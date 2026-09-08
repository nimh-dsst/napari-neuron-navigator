"""CSV export helpers for projected isocortex flatmap coordinates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FLATMAP_EXPORT_COLUMNS = [
    "file_id",
    "neuron_id",
    "subject",
    "node_id",
    "parent_id",
    "type",
    "x_um",
    "y_um",
    "z_um",
    "voxel_i",
    "voxel_j",
    "voxel_k",
    "x_flat",
    "y_flat",
    "depth_um",
    "valid",
    "invalid_reason",
    "region_id",
    "region_acronym",
    "flatmap_style",
    "coordinate_mode",
    "flatmap_lookup_mode",
    "flatmap_valid",
    "depth_valid",
    "render_valid",
    "x_flat_bin",
    "y_flat_bin",
    "depth_bin",
    "depth_bin_label",
    "allen_layer_index",
    "allen_layer_label",
    "allen_layer_selected",
    "allen_layer_render_index",
]


def export_projected_nodes_csv(
    projected_nodes: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write a node-level flatmap projection table to CSV."""
    if projected_nodes is None or projected_nodes.empty:
        raise ValueError("No projected nodes are available to export.")

    table = projected_nodes.copy()
    for column in FLATMAP_EXPORT_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, FLATMAP_EXPORT_COLUMNS].to_csv(output, index=False)
    return output
