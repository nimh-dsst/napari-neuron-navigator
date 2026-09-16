"""Spatial analysis pipeline for neuron clustering and heatmap generation.

Ported from the swc-mapper repository, adapted to use BrainGlobe Atlas API
instead of Allen SDK.
"""

from .clustering import (
    ClusterExclusionRule,
    ClusterRegionFilter,
    ClusterRegionRule,
    ClusterRegionSelection,
    ClusterResult,
    ClusterRunMetadata,
    cluster_somas_dbscan,
    cluster_somas_hierarchical,
    cluster_somas_kmeans,
    compute_clustermap_data,
    compute_linkage,
    compute_linkage_from_condensed,
)
from .correlation import (
    compute_pearson_correlation_matrix,
    correlation_long_to_matrix,
)
from .heatmap import build_node_counts_volume
from .mask import (
    build_binary_mask_from_heatmap,
    build_binary_mask_from_threshold_range,
    dilate_mask_to_volume_increase,
    get_expanded_region_voxel_ids,
    get_expanded_region_voxel_ids_for_regions,
    get_region_mask,
    get_regions_mask,
    isolate_heatmap_volume_to_region_ids,
    merge_heatmap_volumes,
    otsu_threshold_positive,
    smooth_heatmap_volume,
)
from .region_filter import (
    PreparedClusterRegionFilter,
    prepare_cluster_region_filter,
)

__all__ = [
    "ClusterExclusionRule",
    "ClusterRegionFilter",
    "ClusterRegionRule",
    "ClusterRegionSelection",
    "ClusterResult",
    "ClusterRunMetadata",
    "PreparedClusterRegionFilter",
    "build_binary_mask_from_heatmap",
    "build_binary_mask_from_threshold_range",
    "build_node_counts_volume",
    "cluster_somas_dbscan",
    "cluster_somas_hierarchical",
    "cluster_somas_kmeans",
    "compute_clustermap_data",
    "compute_linkage",
    "compute_linkage_from_condensed",
    "compute_pearson_correlation_matrix",
    "correlation_long_to_matrix",
    "dilate_mask_to_volume_increase",
    "get_expanded_region_voxel_ids",
    "get_expanded_region_voxel_ids_for_regions",
    "get_region_mask",
    "get_regions_mask",
    "isolate_heatmap_volume_to_region_ids",
    "merge_heatmap_volumes",
    "otsu_threshold_positive",
    "prepare_cluster_region_filter",
    "smooth_heatmap_volume",
]
