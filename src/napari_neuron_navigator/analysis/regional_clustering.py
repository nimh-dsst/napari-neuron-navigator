"""Unsupervised clustering of neurite measurements in regional sidecars."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import scipy
from scipy.cluster.hierarchy import cut_tree, leaves_list, linkage
from scipy.spatial.distance import cdist

from .region_profile import build_hierarchy_closure
from .region_report import REPORT_LATERALITIES

logger = logging.getLogger(__name__)
DEFAULT_MEMORY_LIMIT = 512 * 1024**2


class RegionalMode(str, Enum):
    CABLE = "cable_distribution"
    REACHED = "regions_reached"
    TERMINI = "terminus_distribution"


MODE_LABELS = {
    RegionalMode.CABLE: "Neurite cable distribution",
    RegionalMode.REACHED: "Regions reached",
    RegionalMode.TERMINI: "Reconstructed terminus distribution",
}


class RegionalClusteringCancelled(RuntimeError):
    """A cancelled computation must not publish assignments."""


@dataclass(frozen=True)
class RegionalClusteringSettings:
    file_ids: tuple[str, ...]
    mode: RegionalMode = RegionalMode.CABLE
    scope: str = "table"
    group_region_ids: tuple[int, ...] = ()
    excluded_region_ids: tuple[int, ...] = ()
    laterality: str = "separate"
    n_clusters: int = 5
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT

    def __post_init__(self):
        if any(value is None or not str(value).strip() for value in self.file_ids):
            raise ValueError("Every cohort member must have a nonempty file_id.")
        object.__setattr__(
            self, "file_ids", tuple(sorted({str(v) for v in self.file_ids}))
        )
        object.__setattr__(self, "mode", RegionalMode(self.mode))
        for field in ("group_region_ids", "excluded_region_ids"):
            object.__setattr__(
                self, field, tuple(sorted({int(v) for v in getattr(self, field)}))
            )
        if self.laterality not in {
            "separate",
            "combined",
            "ipsilateral",
            "contralateral",
        }:
            raise ValueError("Unsupported regional laterality setting.")
        if self.scope not in {"table", "selected"}:
            raise ValueError(
                "Regional clustering uses the Data table or Selected rows."
            )
        if self.n_clusters < 1 or self.memory_limit_bytes < 1:
            raise ValueError("Cluster count and memory limit must be positive.")


@dataclass(frozen=True)
class RegionalFeature:
    region_id: int
    acronym: str
    name: str
    laterality: str
    direct_region_ids: tuple[int, ...]

    @property
    def label(self):
        return f"{self.acronym or self.name} / {self.laterality}"


@dataclass(frozen=True)
class RegionalFeatures:
    settings: RegionalClusteringSettings
    columns: tuple[RegionalFeature, ...]
    matrix: np.ndarray
    zero_feature_file_ids: tuple[str, ...]
    excluded_direct_region_ids: tuple[int, ...]

    @property
    def retained_mask(self):
        return np.any(self.matrix != 0, axis=1)

    @property
    def retained_file_ids(self):
        return tuple(
            v
            for v, keep in zip(self.settings.file_ids, self.retained_mask, strict=True)
            if keep
        )


@dataclass(frozen=True)
class RegionalMemoryEstimate:
    neuron_count: int
    condensed_bytes: int
    estimated_peak_bytes: int


def estimate_regional_memory(n: int, feature_bytes: int = 0) -> RegionalMemoryEstimate:
    """Allow for condensed distances, linkage workspace, and feature copies.

    No square distance matrix is allocated. The conservative working estimate
    includes four condensed-sized buffers and three feature matrices, plus 64 MiB
    for the isolated linkage process. It is a guard, not an RSS guarantee.
    """
    if n < 0 or feature_bytes < 0:
        raise ValueError("Memory-estimate dimensions must be nonnegative.")
    condensed = n * (n - 1) // 2 * 8
    return RegionalMemoryEstimate(
        n, condensed, 4 * condensed + 3 * feature_bytes + 64 * 1024**2
    )


def _check_cancelled(check):
    if check is not None and check():
        raise RegionalClusteringCancelled("Regional-profile clustering was cancelled.")


def _partition_regions(structures, observed, group_ids, excluded_ids):
    closure = build_hierarchy_closure(structures)
    known = set(structures) | {0}
    foreign = (set(group_ids) | set(excluded_ids) | set(observed)) - known
    if foreign:
        raise ValueError(
            f"Region IDs absent from the compatible atlas: {sorted(foreign)}"
        )
    descendants = {
        region_id: set(closure.represented_direct_ids(region_id))
        for region_id in set(group_ids) | set(excluded_ids)
    }
    descendants[0] = {0}
    excluded = set().union(*(descendants[r] for r in excluded_ids))
    # Overlapping selected grouping ancestors resolve to the outermost group.
    # Every retained direct ID maps to exactly one feature group.
    outer = [
        g
        for g in group_ids
        if not any(g != p and g in descendants[p] for p in group_ids)
    ]
    mapping = {}
    for direct in sorted(set(observed) - excluded):
        mapping[direct] = next((g for g in outer if direct in descendants[g]), direct)
    return mapping, tuple(sorted(excluded))


def build_regional_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    structures: Mapping[int, Mapping[str, Any]],
    settings: RegionalClusteringSettings,
    cancel_check: Callable[[], bool] | None = None,
) -> RegionalFeatures:
    """Build a zero-filled file_id × nonoverlapping region/side matrix.

    The connection must contain the validated ``region_profile`` sidecar.
    Globally zero measurement columns are omitted; their values would be zero
    for every cohort neuron. All cohort rows, including zero rows, are retained
    here so exclusion is explicit at the clustering boundary.
    """
    _check_cancelled(cancel_check)
    if not settings.file_ids:
        raise ValueError("Choose a nonempty Data-table or Selected rows cohort.")
    connection.register("_regional_cohort", pa.table({"file_id": settings.file_ids}))
    try:
        missing = connection.execute("""
            SELECT file_id FROM _regional_cohort EXCEPT SELECT file_id FROM region_profile
        """).fetchall()
        if missing:
            raise ValueError(
                f"{len(missing)} cohort file_id(s) are absent from the regional profile."
            )
        observed = [
            int(r[0])
            for r in connection.execute("""
            SELECT DISTINCT region_id FROM region_profile p
            JOIN _regional_cohort c USING (file_id) WHERE compartment = 'neurite'
        """).fetchall()
        ]
        mapping, excluded = _partition_regions(
            structures,
            observed,
            settings.group_region_ids,
            settings.excluded_region_ids,
        )
        connection.register(
            "_regional_groups",
            pa.table(
                {
                    "region_id": pa.array(list(mapping), pa.int32()),
                    "group_id": pa.array(list(mapping.values()), pa.int32()),
                }
            ),
        )
        side = "'combined'" if settings.laterality == "combined" else "p.laterality"
        restriction = (
            " AND p.laterality = ?"
            if settings.laterality in {"ipsilateral", "contralateral"}
            else ""
        )
        params = [settings.laterality] if restriction else []
        value = {
            RegionalMode.CABLE: "sum(p.cable_length_um)",
            RegionalMode.TERMINI: "sum(p.terminus_count)",
            RegionalMode.REACHED: "max(CASE WHEN p.node_count > 0 OR p.cable_length_um > 0 THEN 1 ELSE 0 END)",
        }[settings.mode]
        rows = connection.execute(
            f"""
            SELECT p.file_id, g.group_id, {side} AS side, {value} AS value
            FROM region_profile p JOIN _regional_cohort c USING (file_id)
            JOIN _regional_groups g ON g.region_id = p.region_id
            WHERE p.compartment = 'neurite'{restriction}
            GROUP BY p.file_id, g.group_id, side HAVING value != 0
        """,
            params,
        ).fetchall()
    finally:
        connection.unregister("_regional_cohort")
        connection.unregister("_regional_groups")
    _check_cancelled(cancel_check)
    side_order = {s: i for i, s in enumerate((*REPORT_LATERALITIES, "combined"))}
    keys = sorted(
        {(int(r[1]), str(r[2])) for r in rows}, key=lambda k: (k[0], side_order[k[1]])
    )
    dtype = np.bool_ if settings.mode is RegionalMode.REACHED else np.float64
    size = len(settings.file_ids) * len(keys) * np.dtype(dtype).itemsize
    if size * 2 > settings.memory_limit_bytes:
        raise MemoryError(
            f"Feature matrix needs at least {size * 2 / 1024**2:.1f} MiB including workspace; reduce the cohort or grouping detail."
        )
    matrix = np.zeros((len(settings.file_ids), len(keys)), dtype=dtype)
    row_index = {v: i for i, v in enumerate(settings.file_ids)}
    column_index = {v: i for i, v in enumerate(keys)}
    for file_id, region_id, laterality, value in rows:
        matrix[row_index[file_id], column_index[int(region_id), laterality]] = value
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Regional measurements must be finite and nonnegative.")
    columns = []
    for region_id, laterality in keys:
        structure = structures.get(region_id, {})
        columns.append(
            RegionalFeature(
                region_id,
                str(structure.get("acronym", "unmapped" if region_id == 0 else "")),
                str(
                    structure.get(
                        "name",
                        "Outside / unmapped"
                        if region_id == 0
                        else f"Region {region_id}",
                    )
                ),
                laterality,
                tuple(d for d, g in mapping.items() if g == region_id),
            )
        )
    zero = tuple(
        v
        for v, keep in zip(settings.file_ids, np.any(matrix != 0, axis=1), strict=True)
        if not keep
    )
    matrix.setflags(write=False)
    return RegionalFeatures(settings, tuple(columns), matrix, zero, excluded)


@dataclass(frozen=True)
class RegionalClusterResult:
    features: RegionalFeatures
    file_ids: tuple[str, ...]
    matrix: np.ndarray
    labels: np.ndarray
    linkage_matrix: np.ndarray
    leaf_order: np.ndarray
    metadata: dict[str, Any]

    def file_ids_for_cluster(self, label):
        return tuple(
            v
            for v, c in zip(self.file_ids, self.labels, strict=True)
            if int(c) == int(label)
        )


def _average_linkage_job(distance_path: str, output_path: str, error_path: str, n: int):
    """Qt-free child process, allowing cancellation inside SciPy linkage."""
    try:
        distances = np.memmap(
            distance_path, dtype=np.float64, mode="r", shape=(n * (n - 1) // 2,)
        )
        np.save(
            output_path, linkage(distances, method="average", optimal_ordering=False)
        )
    except Exception as exc:
        logger.exception("Regional average linkage failed")
        Path(error_path).write_text(str(exc), encoding="utf-8")
        raise


def _cancellable_linkage(path, n, directory, check):
    output = str(Path(directory) / "linkage.npy")
    error = str(Path(directory) / "error.txt")
    process = mp.get_context("spawn").Process(
        target=_average_linkage_job, args=(str(path), output, error, n)
    )
    _check_cancelled(check)
    process.start()
    try:
        while process.is_alive():
            _check_cancelled(check)
            process.join(0.1)
        _check_cancelled(check)
        if process.exitcode != 0:
            detail = (
                Path(error).read_text()
                if Path(error).exists()
                else f"exit code {process.exitcode}"
            )
            raise RuntimeError(f"Regional average linkage failed: {detail}")
        return np.load(output, allow_pickle=False)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()


def cluster_regional_features(
    features: RegionalFeatures,
    *,
    provenance: Mapping[str, Any] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> RegionalClusterResult:
    """Cosine/Jaccard distances followed by average-linkage clustering."""
    _check_cancelled(cancel_check)
    ids = features.retained_file_ids
    n = len(ids)
    k = features.settings.n_clusters
    if n < 2:
        raise ValueError(
            f"{n} neurons have nonzero features; {len(features.zero_feature_file_ids)} have zero features. "
            "At least two nonzero neurons are required; broaden the cohort or relax exclusions."
        )
    if k > n:
        raise ValueError(
            f"{n} neurons have nonzero features; {len(features.zero_feature_file_ids)} have zero features. "
            f"Choose between 1 and {n} clusters."
        )
    estimate = estimate_regional_memory(n, features.matrix.size * 8)
    if estimate.estimated_peak_bytes > features.settings.memory_limit_bytes:
        raise MemoryError(
            f"Estimated peak {estimate.estimated_peak_bytes / 1024**2:.1f} MiB "
            f"(pairwise distances {estimate.condensed_bytes / 1024**2:.1f} MiB) exceeds "
            f"the {features.settings.memory_limit_bytes / 1024**2:.0f} MiB limit. "
            "Reduce the cohort or raise the memory limit before running."
        )
    matrix = features.matrix[features.retained_mask].copy()
    metric = "jaccard" if features.settings.mode is RegionalMode.REACHED else "cosine"
    # Cosine is invariant to positive row scaling. Keep intermediate dot
    # products finite without changing the raw, preserved measurement matrix.
    values = (
        matrix if metric == "jaccard" else matrix / matrix.max(axis=1, keepdims=True)
    )
    with tempfile.TemporaryDirectory(prefix="regional-clustering-") as directory:
        path = Path(directory) / "distances.bin"
        distances = np.memmap(
            path, dtype=np.float64, mode="w+", shape=(n * (n - 1) // 2,)
        )
        try:
            offset = 0
            for i in range(n - 1):
                _check_cancelled(cancel_check)
                row = cdist(values[i : i + 1], values[i + 1 :], metric=metric)[0]
                if not np.isfinite(row).all():
                    raise ValueError(
                        "Pairwise regional distances contain nonfinite values."
                    )
                distances[offset : offset + len(row)] = np.clip(row, 0.0, 1.0)
                offset += len(row)
                if progress_callback is not None and i % max(1, n // 100) == 0:
                    progress_callback(f"Computing {metric} distances...", i + 1, n)
            distances.flush()
        finally:
            distances._mmap.close()
        _check_cancelled(cancel_check)
        if progress_callback is not None:
            progress_callback("Average linkage (cancellable)...", n, n)
        tree = _cancellable_linkage(path, n, directory, cancel_check)
    _check_cancelled(cancel_check)
    labels = (cut_tree(tree, n_clusters=[k])[:, 0] + 1).astype(np.int32)
    order = leaves_list(tree).astype(np.int64)
    metadata = {
        **dict(provenance or {}),
        "format_version": 1,
        "analysis": "regional_profile_clustering",
        "algorithm_version": "regional_profile_clustering_v1",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "created_utc": datetime.now(UTC).isoformat(),
        "settings": asdict(features.settings),
        "features": [asdict(column) for column in features.columns],
        "feature_column_policy": "observed nonzero measurement columns; absent buckets zero",
        "measurement_units": {
            RegionalMode.CABLE: "neurite cable micrometres",
            RegionalMode.REACHED: "binary neurite intersection",
            RegionalMode.TERMINI: "reconstructed childless non-soma nodes",
        }[features.settings.mode],
        "intersection_definition": "node_count > 0 OR cable_length_um > 0",
        "compartment": "neurite (all non-soma types)",
        "distance_metric": metric,
        "linkage_method": "average",
        "cluster_label_base": 1,
        "zero_feature_file_ids": list(features.zero_feature_file_ids),
        "excluded_direct_region_ids": list(features.excluded_direct_region_ids),
        "memory_estimate": asdict(estimate),
        "external_annotations_used_for_clustering": False,
    }
    for array in (matrix, labels, tree, order):
        array.setflags(write=False)
    return RegionalClusterResult(features, ids, matrix, labels, tree, order, metadata)


def save_regional_cluster_result(
    path: str | Path, result: RegionalClusterResult
) -> None:
    """Save raw features, assignments, tree, order, and JSON provenance without pickle."""
    with Path(path).open("wb") as stream:
        np.savez_compressed(
            stream,
            file_ids=np.asarray(result.file_ids, dtype=str),
            matrix=result.matrix,
            labels=result.labels,
            linkage=result.linkage_matrix,
            leaf_order=result.leaf_order,
            metadata_json=np.asarray(json.dumps(result.metadata, allow_nan=False)),
        )


def save_regional_neuron_group(path, result, cluster_label, name):
    """Save an explicit file_id group with the clustering definitions attached."""
    ids = result.file_ids_for_cluster(cluster_label)
    if not str(name).strip() or not ids:
        raise ValueError(
            "Choose a named, nonempty cluster before saving a neuron group."
        )
    Path(path).write_text(
        json.dumps(
            {
                "format_version": 1,
                "kind": "regional_profile_neuron_group",
                "name": str(name).strip(),
                "file_ids": list(ids),
                "cluster_label": int(cluster_label),
                "regional_clustering": result.metadata,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
