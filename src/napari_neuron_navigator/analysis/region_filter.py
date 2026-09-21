"""Reusable anatomical include/exclude rules for clustering pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

from ..atlas_utils import mask_to_swc_xyz_bounds
from .clustering import ClusterRegionFilter
from .mask import dilate_mask_to_volume_increase, get_region_mask

if TYPE_CHECKING:
    import duckdb
    from brainglobe_atlasapi import BrainGlobeAtlas
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedClusterRegionFilter:
    """Atlas masks prepared once in preflight and reused by a clustering run."""

    region_filter: ClusterRegionFilter
    resolution_um: tuple[float, ...]
    atlas_shape: tuple[int, int, int]
    include_mask: NDArray[np.bool_] | None
    exclude_masks: tuple[NDArray[np.bool_], ...]
    exclude_mask: NDArray[np.bool_] | None

    @property
    def is_empty(self) -> bool:
        """Return whether no effective include or exclude masks exist."""
        return self.include_mask is None and self.exclude_mask is None


def prepare_cluster_region_filter(
    atlas: BrainGlobeAtlas,
    region_filter: ClusterRegionFilter | None,
) -> PreparedClusterRegionFilter | None:
    """Build every direct rule mask independently, then union by rule role."""
    if region_filter is None or region_filter.is_empty:
        return None

    resolution = tuple(float(value) for value in atlas.resolution)
    shape = tuple(int(value) for value in atlas.annotation.shape)
    cache: dict[tuple[int, float], np.ndarray] = {}

    def prepare_rule(rule) -> np.ndarray:
        key = (int(rule.region_id), float(rule.dilation_fraction))
        cached = cache.get(key)
        if cached is not None:
            return cached
        mask = np.asarray(get_region_mask(atlas, rule.acronym), dtype=bool)
        if mask.shape != shape:
            raise ValueError(
                f"Region {rule.acronym!r} mask shape {mask.shape} does not match "
                f"atlas shape {shape}."
            )
        if rule.dilation_fraction > 0.0:
            mask = dilate_mask_to_volume_increase(
                mask,
                increase_fraction=rule.dilation_fraction,
                voxel_spacing_um=resolution,
            )
        cache[key] = mask
        return mask

    include_masks = tuple(prepare_rule(rule) for rule in region_filter.include_rules)
    exclude_masks = tuple(prepare_rule(rule) for rule in region_filter.exclude_rules)
    include_union = np.logical_or.reduce(include_masks) if include_masks else None
    exclude_union = np.logical_or.reduce(exclude_masks) if exclude_masks else None
    return PreparedClusterRegionFilter(
        region_filter=region_filter,
        resolution_um=resolution,
        atlas_shape=shape,
        include_mask=include_union,
        exclude_masks=exclude_masks,
        exclude_mask=exclude_union,
    )


def points_in_mask(
    coords_xyz: np.ndarray,
    mask: np.ndarray,
    resolution_um: tuple[float, ...] | list[float] | np.ndarray,
) -> np.ndarray:
    """Return mask membership for SWC/Parquet XYZ micron coordinates."""
    coords = np.asarray(coords_xyz, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords_xyz must be an (N, 3) array; got {coords.shape}.")
    volume = np.asarray(mask, dtype=bool)
    resolution = np.asarray(resolution_um, dtype=float)
    finite = np.all(np.isfinite(coords), axis=1)
    voxels = np.zeros((len(coords), 3), dtype=np.int64)
    if finite.any():
        voxels[finite] = np.floor(coords[finite] / resolution).astype(np.int64)
    in_bounds = finite & np.all(
        (voxels >= 0) & (voxels < np.asarray(volume.shape, dtype=np.int64)),
        axis=1,
    )
    matched = np.zeros(len(coords), dtype=bool)
    valid = voxels[in_bounds]
    if len(valid):
        matched[in_bounds] = volume[valid[:, 0], valid[:, 1], valid[:, 2]]
    return matched


def filter_soma_frame(
    soma_frame: pd.DataFrame,
    prepared_filter: PreparedClusterRegionFilter | None,
) -> pd.DataFrame:
    """Apply include-mask soma eligibility to an averaged soma dataframe."""
    if soma_frame.empty:
        return soma_frame.copy()
    retained = np.ones(len(soma_frame), dtype=bool)
    if prepared_filter is not None and prepared_filter.include_mask is not None:
        coords = soma_frame[["x", "y", "z"]].to_numpy(dtype=float, copy=False)
        retained = points_in_mask(
            coords,
            prepared_filter.include_mask,
            prepared_filter.resolution_um,
        )
    return soma_frame.loc[retained].reset_index(drop=True)


def filter_voxel_frame(
    node_frame: pd.DataFrame,
    prepared_filter: PreparedClusterRegionFilter | None,
) -> pd.DataFrame:
    """Apply voxel include/exclude masks to an in-memory projected-node table."""
    if prepared_filter is None or prepared_filter.is_empty or node_frame.empty:
        return node_frame.copy()
    missing = [name for name in ("x", "y", "z") if name not in node_frame]
    if missing:
        raise ValueError(
            "Anatomical region filtering requires projected nodes to retain "
            f"their CCF coordinate column(s); missing {missing}."
        )

    coords = node_frame[["x", "y", "z"]].to_numpy(dtype=float, copy=False)
    retained = np.ones(len(node_frame), dtype=bool)
    if prepared_filter.include_mask is not None:
        retained &= points_in_mask(
            coords,
            prepared_filter.include_mask,
            prepared_filter.resolution_um,
        )
    if prepared_filter.exclude_mask is not None:
        retained &= ~points_in_mask(
            coords,
            prepared_filter.exclude_mask,
            prepared_filter.resolution_um,
        )
    return node_frame.loc[retained].reset_index(drop=True)


def _source_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def excluded_soma_file_ids(
    parquet_path: str | Path,
    prepared_filter: PreparedClusterRegionFilter | None,
    *,
    file_ids: list[str] | tuple[str, ...] | None = None,
) -> set[str]:
    """Return file IDs rejected by any per-region soma exclusion rule."""
    if prepared_filter is None or not prepared_filter.exclude_masks:
        return set()
    if file_ids is not None and not file_ids:
        return set()

    import duckdb

    scoped_ids = None if file_ids is None else [str(value) for value in file_ids]
    excluded: set[str] = set()
    conn = duckdb.connect()
    try:
        source_sql = f"read_parquet('{_source_path(parquet_path)}')"
        for rule, mask in zip(
            prepared_filter.region_filter.exclude_rules,
            prepared_filter.exclude_masks,
        ):
            bounds = mask_to_swc_xyz_bounds(
                mask,
                prepared_filter.resolution_um,
            )
            if bounds is None:
                continue
            lower, upper = bounds
            clauses = [
                "x IS NOT NULL AND isfinite(x)",
                "y IS NOT NULL AND isfinite(y)",
                "z IS NOT NULL AND isfinite(z)",
                "x >= ? AND x <= ?",
                "y >= ? AND y <= ?",
                "z >= ? AND z <= ?",
            ]
            params: list[object] = [
                float(lower[0]),
                float(upper[0]),
                float(lower[1]),
                float(upper[1]),
                float(lower[2]),
                float(upper[2]),
            ]
            if rule.node_types is not None:
                placeholders = ", ".join("?" for _ in rule.node_types)
                clauses.append(f"type IN ({placeholders})")
                params.extend(int(value) for value in rule.node_types)
            if scoped_ids is not None:
                placeholders = ", ".join("?" for _ in scoped_ids)
                clauses.append(f"CAST(file_id AS VARCHAR) IN ({placeholders})")
                params.extend(scoped_ids)

            frame = conn.execute(
                f"""
                SELECT CAST(file_id AS VARCHAR) AS file_id, x, y, z
                FROM {source_sql}
                WHERE {" AND ".join(clauses)}
                """,
                params,
            ).fetchdf()
            if frame.empty:
                continue
            inside = points_in_mask(
                frame[["x", "y", "z"]].to_numpy(dtype=float, copy=False),
                mask,
                prepared_filter.resolution_um,
            )
            counts = frame.loc[inside, "file_id"].astype(str).value_counts()
            excluded.update(
                str(file_id)
                for file_id, count in counts.items()
                if int(count) >= rule.minimum_node_count
            )
    finally:
        conn.close()
    return excluded


def _mask_index_table(mask: np.ndarray) -> pa.Table:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool).ravel()).astype(
        np.int64,
        copy=False,
    )
    return pa.table({"voxel_index": indices})


def register_filtered_source_view(
    conn: duckdb.DuckDBPyConnection,
    source_sql: str,
    prepared_filter: PreparedClusterRegionFilter | None,
    *,
    view_name: str = "cluster_region_filtered_source",
) -> tuple[str, tuple[str, ...]]:
    """Create a DuckDB view applying voxel include/exclude masks to node rows.

    The returned relation names must be passed to
    :func:`cleanup_filtered_source_view` after the query finishes.
    """
    if prepared_filter is None or prepared_filter.is_empty:
        return source_sql, ()

    registered: list[str] = []
    include_join = ""
    if prepared_filter.include_mask is not None:
        include_name = "cluster_include_voxels"
        conn.register(include_name, _mask_index_table(prepared_filter.include_mask))
        registered.append(include_name)
        include_join = (
            f"JOIN {include_name} inc ON inc.voxel_index = mapped.voxel_index"
        )

    exclude_join = ""
    exclude_where = ""
    if prepared_filter.exclude_mask is not None:
        exclude_name = "cluster_exclude_voxels"
        conn.register(exclude_name, _mask_index_table(prepared_filter.exclude_mask))
        registered.append(exclude_name)
        exclude_join = (
            f"LEFT JOIN {exclude_name} exc ON exc.voxel_index = mapped.voxel_index"
        )
        exclude_where = "WHERE exc.voxel_index IS NULL"

    a_size, b_size, c_size = prepared_filter.atlas_shape
    ra, rb, rc = prepared_filter.resolution_um
    finite = (
        "p.x IS NOT NULL AND isfinite(p.x) "
        "AND p.y IS NOT NULL AND isfinite(p.y) "
        "AND p.z IS NOT NULL AND isfinite(p.z)"
    )
    ai = f"CAST(FLOOR(p.x / {float(ra)}) AS BIGINT)"
    bi = f"CAST(FLOOR(p.y / {float(rb)}) AS BIGINT)"
    ci = f"CAST(FLOOR(p.z / {float(rc)}) AS BIGINT)"
    in_bounds = (
        f"{ai} >= 0 AND {ai} < {a_size} "
        f"AND {bi} >= 0 AND {bi} < {b_size} "
        f"AND {ci} >= 0 AND {ci} < {c_size}"
    )
    linear = f"({ai} * {b_size * c_size}::BIGINT + {bi} * {c_size}::BIGINT + {ci})"
    voxel_index = f"CASE WHEN ({finite}) AND ({in_bounds}) THEN {linear} ELSE NULL END"
    conn.execute(f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        WITH mapped AS (
            SELECT p.*, {voxel_index} AS voxel_index
            FROM {source_sql} p
        )
        SELECT mapped.* EXCLUDE (voxel_index)
        FROM mapped
        {include_join}
        {exclude_join}
        {exclude_where}
    """)
    return view_name, tuple(registered)


def cleanup_filtered_source_view(
    conn: duckdb.DuckDBPyConnection,
    view_name: str,
    registered_relations: tuple[str, ...],
) -> None:
    """Remove a filtered source view and its registered Arrow relations."""
    if registered_relations:
        try:
            conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        except Exception:
            logger.debug("Could not drop clustering region view", exc_info=True)
    for name in registered_relations:
        try:
            conn.unregister(name)
        except Exception:
            logger.debug("Could not unregister %s", name, exc_info=True)
