"""Reusable node-row filters for voxel-correlation clustering.

The filters in this module are deliberately independent from anatomical region
filters.  Region membership, source node types, and physical distance from the
soma answer different questions and are composed only at the final source-view
boundary used by voxel correlation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa

from ..swc import NodeType, node_type_labels, normalize_node_types

if TYPE_CHECKING:
    import duckdb


NodeTypeFilterMode = Literal["all", "include", "exclude"]
logger = logging.getLogger(__name__)


def _provenance_node_type_labels(values: tuple[int, ...]) -> list[str]:
    labels = node_type_labels(values)
    return [
        "Axon-typed (type 2)" if value == NodeType.AXON else label
        for value, label in zip(values, labels)
    ]


@dataclass(frozen=True)
class VoxelNodeFilter:
    """Immutable node-filter settings for one voxel-correlation run."""

    node_type_mode: NodeTypeFilterMode = "all"
    node_types: tuple[int, ...] = ()
    require_dendrite_labels: bool = False
    exclude_within_soma_um: float | None = None
    dendrite_node_types: tuple[int, ...] = (
        NodeType.BASAL_DENDRITE,
        NodeType.APICAL_DENDRITE,
    )

    def __post_init__(self) -> None:
        mode = str(self.node_type_mode).lower()
        if mode not in {"all", "include", "exclude"}:
            raise ValueError(f"Unknown node-type filter mode: {self.node_type_mode!r}")
        object.__setattr__(self, "node_type_mode", mode)
        object.__setattr__(
            self,
            "node_types",
            normalize_node_types(self.node_types) or (),
        )
        object.__setattr__(
            self,
            "dendrite_node_types",
            normalize_node_types(self.dendrite_node_types) or (),
        )
        if mode != "all" and not self.node_types:
            raise ValueError("Select at least one node type for node-type filtering.")
        radius = self.exclude_within_soma_um
        if radius is not None:
            radius = float(radius)
            if not np.isfinite(radius) or radius < 0.0:
                raise ValueError(
                    "Soma exclusion distance must be finite and non-negative."
                )
            object.__setattr__(self, "exclude_within_soma_um", radius)
        if self.require_dendrite_labels and not self.dendrite_node_types:
            raise ValueError(
                "No dendrite node types are available for coverage filtering."
            )

    @property
    def is_empty(self) -> bool:
        """Return whether this specification leaves voxel nodes unfiltered."""
        return (
            self.node_type_mode == "all"
            and not self.require_dendrite_labels
            and self.exclude_within_soma_um is None
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe, versioned filter provenance."""
        return {
            "version": 1,
            "node_type_mode": self.node_type_mode,
            "node_types": [int(value) for value in self.node_types],
            "node_type_labels": _provenance_node_type_labels(self.node_types),
            "require_dendrite_labels": bool(self.require_dendrite_labels),
            "dendrite_node_types": [int(value) for value in self.dendrite_node_types],
            "dendrite_node_type_labels": _provenance_node_type_labels(
                self.dendrite_node_types
            ),
            "exclude_within_soma_um": self.exclude_within_soma_um,
            "soma_distance_boundary": (
                "exclude_less_than_or_equal"
                if self.exclude_within_soma_um is not None
                else None
            ),
            "soma_coordinate_space": (
                "ccfv3_xyz_um" if self.exclude_within_soma_um is not None else None
            ),
        }


@dataclass(frozen=True)
class DendriteLabelCoverage:
    """Whole-neuron dendrite-label coverage, always keyed by ``file_id``."""

    input_file_ids: tuple[str, ...]
    labeled_file_ids: tuple[str, ...]
    dendrite_node_types: tuple[int, ...]

    @property
    def input_neuron_count(self) -> int:
        return len(self.input_file_ids)

    @property
    def labeled_neuron_count(self) -> int:
        return len(self.labeled_file_ids)

    @property
    def excluded_neuron_count(self) -> int:
        return self.input_neuron_count - self.labeled_neuron_count


@dataclass(frozen=True)
class PreparedVoxelNodeFilter:
    """Small lookup tables prepared once and reused by preflight and execution."""

    settings: VoxelNodeFilter
    input_file_ids: tuple[str, ...]
    dendrite_labeled_file_ids: tuple[str, ...] | None = None
    soma_file_ids: tuple[str, ...] | None = None
    soma_centroids_xyz: np.ndarray | None = None

    @property
    def missing_soma_neuron_count(self) -> int:
        if self.soma_file_ids is None:
            return 0
        return len(set(self.input_file_ids) - set(self.soma_file_ids))

    def metadata(self) -> dict[str, object]:
        """Return the settings plus neuron-level preparation counts."""
        payload = self.settings.to_dict()
        payload.update(
            {
                "input_neuron_count": len(self.input_file_ids),
                "dendrite_labeled_neuron_count": (
                    None
                    if self.dendrite_labeled_file_ids is None
                    else len(self.dendrite_labeled_file_ids)
                ),
                "dendrite_unlabeled_neuron_count": (
                    None
                    if self.dendrite_labeled_file_ids is None
                    else len(self.input_file_ids) - len(self.dendrite_labeled_file_ids)
                ),
                "valid_soma_neuron_count": (
                    None if self.soma_file_ids is None else len(self.soma_file_ids)
                ),
                "missing_soma_neuron_count": (
                    None
                    if self.soma_file_ids is None
                    else self.missing_soma_neuron_count
                ),
            }
        )
        return payload


def _source_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _unique_strings(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def prepare_voxel_node_filter(
    conn: duckdb.DuckDBPyConnection,
    raw_source_sql: str,
    settings: VoxelNodeFilter,
    *,
    file_ids: list[str] | tuple[str, ...] | None = None,
) -> PreparedVoxelNodeFilter:
    """Prepare dendrite coverage and/or soma centroids from unfiltered rows.

    ``raw_source_sql`` must be the original Parquet relation.  In particular,
    it must not be a region-filtered view: coverage and soma location are
    properties of the complete neuron.
    """
    scoped_ids = None if file_ids is None else _unique_strings(file_ids)
    if scoped_ids == ():
        return PreparedVoxelNodeFilter(
            settings=settings,
            input_file_ids=(),
            dendrite_labeled_file_ids=() if settings.require_dendrite_labels else None,
            soma_file_ids=() if settings.exclude_within_soma_um is not None else None,
            soma_centroids_xyz=(
                np.empty((0, 3), dtype=np.float64)
                if settings.exclude_within_soma_um is not None
                else None
            ),
        )
    scope_name = "voxel_filter_prepare_scope"
    scope_join = ""
    registered_scope = scoped_ids is not None
    if registered_scope:
        conn.register(scope_name, pa.table({"file_id": list(scoped_ids)}))
        scope_join = (
            f"JOIN {scope_name} scope ON CAST(p.file_id AS VARCHAR) = scope.file_id"
        )

    select_parts = ["CAST(p.file_id AS VARCHAR) AS file_id"]
    if settings.require_dendrite_labels:
        dendrite_values = ", ".join(
            str(int(value)) for value in settings.dendrite_node_types
        )
        select_parts.append(
            f"BOOL_OR(p.type IN ({dendrite_values})) AS has_dendrite_label"
        )
    if settings.exclude_within_soma_um is not None:
        finite_soma = (
            "p.type = 1 AND p.x IS NOT NULL AND isfinite(p.x) "
            "AND p.y IS NOT NULL AND isfinite(p.y) "
            "AND p.z IS NOT NULL AND isfinite(p.z)"
        )
        select_parts.extend(
            [
                f"AVG(p.x) FILTER (WHERE {finite_soma}) AS soma_x",
                f"AVG(p.y) FILTER (WHERE {finite_soma}) AS soma_y",
                f"AVG(p.z) FILTER (WHERE {finite_soma}) AS soma_z",
                f"COUNT(*) FILTER (WHERE {finite_soma}) AS soma_node_count",
            ]
        )

    try:
        frame = conn.execute(
            f"""
            SELECT {", ".join(select_parts)}
            FROM {raw_source_sql} p
            {scope_join}
            GROUP BY p.file_id
            ORDER BY file_id
            """
        ).fetchdf()
    finally:
        if registered_scope:
            try:
                conn.unregister(scope_name)
            except Exception:
                logger.debug("Could not unregister voxel-filter scope", exc_info=True)

    input_ids = _unique_strings(frame.get("file_id", ()))
    dendrite_ids: tuple[str, ...] | None = None
    if settings.require_dendrite_labels:
        dendrite_ids = _unique_strings(
            frame.loc[
                frame["has_dendrite_label"].fillna(False).astype(bool),
                "file_id",
            ]
        )

    soma_ids: tuple[str, ...] | None = None
    soma_centroids: np.ndarray | None = None
    if settings.exclude_within_soma_um is not None:
        valid_soma = frame["soma_node_count"].fillna(0).astype(int) > 0
        soma_frame = frame.loc[valid_soma, ["file_id", "soma_x", "soma_y", "soma_z"]]
        soma_ids = _unique_strings(soma_frame["file_id"])
        soma_centroids = soma_frame[["soma_x", "soma_y", "soma_z"]].to_numpy(
            dtype=np.float64,
            copy=True,
        )

    return PreparedVoxelNodeFilter(
        settings=settings,
        input_file_ids=input_ids,
        dendrite_labeled_file_ids=dendrite_ids,
        soma_file_ids=soma_ids,
        soma_centroids_xyz=soma_centroids,
    )


def prepare_voxel_node_filter_from_parquet(
    parquet_path: str | Path,
    settings: VoxelNodeFilter,
    *,
    file_ids: list[str] | tuple[str, ...] | None = None,
) -> PreparedVoxelNodeFilter:
    """Open a short-lived DuckDB connection and prepare a voxel filter."""
    import duckdb

    conn = duckdb.connect()
    try:
        source_sql = f"read_parquet('{_source_path(parquet_path)}')"
        return prepare_voxel_node_filter(
            conn,
            source_sql,
            settings,
            file_ids=file_ids,
        )
    finally:
        conn.close()


def query_dendrite_label_coverage(
    parquet_path: str | Path,
    *,
    file_ids: list[str] | tuple[str, ...] | None = None,
    dendrite_node_types: tuple[int, ...] = (
        NodeType.BASAL_DENDRITE,
        NodeType.APICAL_DENDRITE,
    ),
) -> DendriteLabelCoverage:
    """Return which scoped neurons contain at least one dendrite-typed node."""
    settings = VoxelNodeFilter(
        require_dendrite_labels=True,
        dendrite_node_types=dendrite_node_types,
    )
    prepared = prepare_voxel_node_filter_from_parquet(
        parquet_path,
        settings,
        file_ids=file_ids,
    )
    return DendriteLabelCoverage(
        input_file_ids=prepared.input_file_ids,
        labeled_file_ids=prepared.dendrite_labeled_file_ids or (),
        dendrite_node_types=settings.dendrite_node_types,
    )


def register_voxel_filtered_source_view(
    conn: duckdb.DuckDBPyConnection,
    source_sql: str,
    prepared_filter: PreparedVoxelNodeFilter | None,
    *,
    view_name: str = "voxel_node_filtered_source",
) -> tuple[str, tuple[str, ...]]:
    """Wrap ``source_sql`` in a view applying all configured row filters."""
    if prepared_filter is None or prepared_filter.settings.is_empty:
        return source_sql, ()

    settings = prepared_filter.settings
    registered: list[str] = []
    joins: list[str] = []
    clauses: list[str] = []

    if settings.require_dendrite_labels:
        name = f"{view_name}_dendrite_ids"
        conn.register(
            name,
            pa.table(
                {"file_id": list(prepared_filter.dendrite_labeled_file_ids or ())}
            ),
        )
        registered.append(name)
        joins.append(f"JOIN {name} dend ON CAST(p.file_id AS VARCHAR) = dend.file_id")

    if settings.exclude_within_soma_um is not None:
        name = f"{view_name}_soma_centroids"
        soma_ids = prepared_filter.soma_file_ids or ()
        centroids = prepared_filter.soma_centroids_xyz
        if centroids is None:
            centroids = np.empty((0, 3), dtype=np.float64)
        conn.register(
            name,
            pa.table(
                {
                    "file_id": list(soma_ids),
                    "soma_x": centroids[:, 0],
                    "soma_y": centroids[:, 1],
                    "soma_z": centroids[:, 2],
                }
            ),
        )
        registered.append(name)
        joins.append(f"JOIN {name} soma ON CAST(p.file_id AS VARCHAR) = soma.file_id")
        radius_squared = float(settings.exclude_within_soma_um) ** 2
        clauses.append(
            "(POW(p.x - soma.soma_x, 2) + POW(p.y - soma.soma_y, 2) "
            f"+ POW(p.z - soma.soma_z, 2)) > {radius_squared!r}"
        )

    if settings.node_type_mode != "all":
        values = ", ".join(str(int(value)) for value in settings.node_types)
        if settings.node_type_mode == "include":
            clauses.append(f"p.type IN ({values})")
        else:
            # NULL has no selectable type and therefore is not one of the
            # explicitly excluded values.
            clauses.append(f"(p.type IS NULL OR p.type NOT IN ({values}))")

    where_sql = "" if not clauses else "WHERE " + " AND ".join(clauses)
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        SELECT p.*
        FROM {source_sql} p
        {" ".join(joins)}
        {where_sql}
        """
    )
    return view_name, tuple(registered)


def cleanup_voxel_filtered_source_view(
    conn: duckdb.DuckDBPyConnection,
    view_name: str,
    registered_relations: tuple[str, ...],
) -> None:
    """Remove one filtered view and its registered lookup relations."""
    try:
        conn.execute(f"DROP VIEW IF EXISTS {view_name}")
    except Exception:
        logger.debug("Could not drop voxel-filter view %s", view_name, exc_info=True)
    for name in registered_relations:
        try:
            conn.unregister(name)
        except Exception:
            logger.debug(
                "Could not unregister voxel-filter relation %s",
                name,
                exc_info=True,
            )
