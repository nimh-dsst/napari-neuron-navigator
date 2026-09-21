"""Pearson cross-correlation matrix computation using DuckDB.

Ported from swc-mapper/pearson_cross_correlation_matrix_egpe_counts.py and
swc-mapper/corr_full_to_matrix.py.

Computes pairwise Pearson correlations between neurons based on their node
counts per voxel, optionally restricted to a target brain region.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

if TYPE_CHECKING:
    import duckdb
    from numpy.typing import NDArray

    from .region_filter import PreparedClusterRegionFilter
    from .voxel_filter import PreparedVoxelNodeFilter, VoxelNodeFilter

logger = logging.getLogger(__name__)


def _prepare_region_nodes_view(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    voxel_id_map: NDArray[np.int32] | None,
    resolution: float,
    file_ids: list[str] | tuple[str, ...] | None,
    prepared_region_filter: PreparedClusterRegionFilter | None = None,
    voxel_node_filter: VoxelNodeFilter | None = None,
    prepared_voxel_filter: PreparedVoxelNodeFilter | None = None,
) -> tuple[
    bool,
    bool,
    str | None,
    tuple[str, ...],
    str | None,
    tuple[str, ...],
]:
    """Create the scoped node-to-voxel view used by counting and correlation."""
    parquet_path_escaped = str(parquet_path).replace("\\", "/").replace("'", "''")
    scoped_file_ids = None
    if file_ids is not None:
        scoped_file_ids = list(dict.fromkeys(str(file_id) for file_id in file_ids))
        if not scoped_file_ids:
            conn.execute(
                "CREATE OR REPLACE TEMP VIEW region_nodes AS "
                "SELECT CAST(NULL AS VARCHAR) AS swc_id, "
                "CAST(NULL AS BIGINT) AS voxel_id WHERE FALSE"
            )
            return False, False, None, (), None, ()

    scope_registered = scoped_file_ids is not None
    if scope_registered:
        conn.register(
            "scope_ids",
            pa.table({"swc_id": np.asarray(scoped_file_ids, dtype=object)}),
        )
    scope_join = (
        "JOIN scope_ids s ON CAST(p.file_id AS VARCHAR) = s.swc_id"
        if scope_registered
        else ""
    )

    raw_source_sql = f"read_parquet('{parquet_path_escaped}')"
    source_sql = raw_source_sql
    filter_view_name = None
    filter_relations: tuple[str, ...] = ()
    if prepared_region_filter is not None:
        from .region_filter import register_filtered_source_view

        filter_view_name = "cluster_region_filtered_ccf_source"
        source_sql, filter_relations = register_filtered_source_view(
            conn,
            source_sql,
            prepared_region_filter,
            view_name=filter_view_name,
        )

    voxel_filter_view_name = None
    voxel_filter_relations: tuple[str, ...] = ()
    if voxel_node_filter is not None and not voxel_node_filter.is_empty:
        from .voxel_filter import (
            prepare_voxel_node_filter,
            register_voxel_filtered_source_view,
        )

        if prepared_voxel_filter is None:
            prepared_voxel_filter = prepare_voxel_node_filter(
                conn,
                raw_source_sql,
                voxel_node_filter,
                file_ids=file_ids,
            )
        elif prepared_voxel_filter.settings != voxel_node_filter:
            raise ValueError(
                "Prepared voxel filter does not match the requested settings."
            )
        voxel_filter_view_name = "cluster_voxel_filtered_ccf_source"
        source_sql, voxel_filter_relations = register_voxel_filtered_source_view(
            conn,
            source_sql,
            prepared_voxel_filter,
            view_name=voxel_filter_view_name,
        )

    conn.execute(f"""
        CREATE OR REPLACE TEMP VIEW base_nodes AS
        SELECT
            CAST(p.file_id AS VARCHAR) AS swc_id,
            CAST(FLOOR(z / {float(resolution)}) AS BIGINT) AS xi,
            CAST(FLOOR(y / {float(resolution)}) AS BIGINT) AS yi,
            CAST(FLOOR(x / {float(resolution)}) AS BIGINT) AS zi
        FROM {source_sql} p
        {scope_join}
        WHERE x IS NOT NULL AND y IS NOT NULL AND z IS NOT NULL
          AND isfinite(x) AND isfinite(y) AND isfinite(z)
    """)

    if voxel_id_map is None:
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE occupied_voxels AS
            SELECT
                zi,
                yi,
                xi,
                ROW_NUMBER() OVER (ORDER BY zi, yi, xi) - 1 AS voxel_id
            FROM (SELECT DISTINCT zi, yi, xi FROM base_nodes)
        """)
        conn.execute("""
            CREATE OR REPLACE TEMP VIEW region_nodes AS
            SELECT b.swc_id, v.voxel_id
            FROM base_nodes b
            JOIN occupied_voxels v USING (zi, yi, xi)
        """)
        return (
            False,
            scope_registered,
            filter_view_name,
            filter_relations,
            voxel_filter_view_name,
            voxel_filter_relations,
        )

    Z, Y, X = voxel_id_map.shape
    conn.register(
        "lut",
        pa.table(
            {
                "idx": np.arange(voxel_id_map.size, dtype=np.int64),
                "val": voxel_id_map.ravel(),
            }
        ),
    )
    conn.execute(f"""
        CREATE OR REPLACE TEMP VIEW region_nodes AS
        WITH mapped AS (
            SELECT b.swc_id, l.val AS voxel_id
            FROM base_nodes b
            JOIN lut l
                ON l.idx = (b.zi * ({Y} * {X})::BIGINT + b.yi * {X}::BIGINT + b.xi)
            WHERE b.xi >= 0 AND b.xi < {X}
              AND b.yi >= 0 AND b.yi < {Y}
              AND b.zi >= 0 AND b.zi < {Z}
        )
        SELECT swc_id, CAST(voxel_id AS BIGINT) AS voxel_id
        FROM mapped
        WHERE voxel_id >= 0
    """)
    return (
        True,
        scope_registered,
        filter_view_name,
        filter_relations,
        voxel_filter_view_name,
        voxel_filter_relations,
    )


def _cleanup_region_nodes_view(
    conn: duckdb.DuckDBPyConnection,
    *,
    lut_registered: bool,
    scope_registered: bool,
    filter_view_name: str | None = None,
    filter_relations: tuple[str, ...] = (),
    voxel_filter_view_name: str | None = None,
    voxel_filter_relations: tuple[str, ...] = (),
) -> None:
    """Remove temporary relations and Arrow registrations."""
    for relation in ("region_nodes", "base_nodes", "occupied_voxels"):
        try:
            conn.execute(f"DROP VIEW IF EXISTS {relation}")
            conn.execute(f"DROP TABLE IF EXISTS {relation}")
        except Exception:
            pass
    if lut_registered:
        try:
            conn.unregister("lut")
        except Exception:
            pass
    if scope_registered:
        try:
            conn.unregister("scope_ids")
        except Exception:
            pass
    if voxel_filter_view_name is not None:
        from .voxel_filter import cleanup_voxel_filtered_source_view

        cleanup_voxel_filtered_source_view(
            conn,
            voxel_filter_view_name,
            voxel_filter_relations,
        )
    if filter_view_name is not None:
        from .region_filter import cleanup_filtered_source_view

        cleanup_filtered_source_view(
            conn,
            filter_view_name,
            filter_relations,
        )


def count_correlation_input_nodes(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    voxel_id_map: NDArray[np.int32] | None,
    resolution: float,
    file_ids: list[str] | tuple[str, ...] | None = None,
    prepared_region_filter: PreparedClusterRegionFilter | None = None,
    voxel_node_filter: VoxelNodeFilter | None = None,
    prepared_voxel_filter: PreparedVoxelNodeFilter | None = None,
) -> int:
    """Return the exact node-row count used by CCF voxel correlation."""
    (
        lut_registered,
        scope_registered,
        filter_view_name,
        filter_relations,
        voxel_filter_view_name,
        voxel_filter_relations,
    ) = _prepare_region_nodes_view(
        conn,
        parquet_path,
        voxel_id_map,
        resolution,
        file_ids,
        prepared_region_filter,
        voxel_node_filter,
        prepared_voxel_filter,
    )
    try:
        row = conn.execute("SELECT COUNT(*) FROM region_nodes").fetchone()
        return int(row[0] or 0) if row is not None else 0
    finally:
        _cleanup_region_nodes_view(
            conn,
            lut_registered=lut_registered,
            scope_registered=scope_registered,
            filter_view_name=filter_view_name,
            filter_relations=filter_relations,
            voxel_filter_view_name=voxel_filter_view_name,
            voxel_filter_relations=voxel_filter_relations,
        )


def compute_pearson_correlation_matrix(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    voxel_id_map: NDArray[np.int32] | None,
    resolution: float,
    file_ids: list[str] | tuple[str, ...] | None = None,
    progress_callback: callable | None = None,
    prepared_region_filter: PreparedClusterRegionFilter | None = None,
    voxel_node_filter: VoxelNodeFilter | None = None,
    prepared_voxel_filter: PreparedVoxelNodeFilter | None = None,
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of neuron node counts per voxel.

    For each neuron, counts nodes in each occupied voxel, optionally restricted
    by ``voxel_id_map``, then computes Pearson r between all neuron pairs.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        An open DuckDB connection.
    parquet_path : str
        Path to the neuron parquet file.
    voxel_id_map : NDArray[np.int32] or None
        Optional 3D target-region voxel ID map. When omitted, all finite CCF
        coordinates in the selected file scope contribute.
    resolution : float
        Voxel resolution in microns (e.g., 25.0).
    file_ids : list[str] or tuple[str, ...], optional
        Restrict the computation to this working set of neuron ``file_id`` values.
    progress_callback : callable, optional
        Called with (step_name: str, step_number: int, total_steps: int).

    Returns
    -------
    pd.DataFrame
        Long-form correlation table with columns: swc_id_1, swc_id_2, r.
        Includes both triangles plus the diagonal (r=1).
    """

    def _progress(name: str, step: int, total: int = 7) -> None:
        logger.info(f"Correlation step {step}/{total}: {name}")
        if progress_callback is not None:
            progress_callback(name, step, total)

    _progress("Preparing voxel lookup", 1)
    (
        lut_registered,
        scope_registered,
        filter_view_name,
        filter_relations,
        voxel_filter_view_name,
        voxel_filter_relations,
    ) = _prepare_region_nodes_view(
        conn,
        parquet_path,
        voxel_id_map,
        resolution,
        file_ids,
        prepared_region_filter,
        voxel_node_filter,
        prepared_voxel_filter,
    )

    _progress("Mapping nodes to voxel IDs", 2)

    # Count nodes per neuron per voxel
    _progress("Counting nodes per voxel", 3)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE counts_by_voxel AS
        SELECT
            swc_id,
            voxel_id,
            COUNT(*)::BIGINT AS c
        FROM region_nodes
        GROUP BY swc_id, voxel_id
    """)

    # Compute voxel universe size
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE voxel_universe AS
        SELECT COUNT(DISTINCT voxel_id)::BIGINT AS V
        FROM counts_by_voxel
    """)

    # Per-neuron sums for Pearson formula
    _progress("Computing per-neuron statistics", 4)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE per_neuron AS
        SELECT
            swc_id,
            SUM(c)::DOUBLE AS Sx,
            SUM(c * c)::DOUBLE AS Sxx
        FROM counts_by_voxel
        GROUP BY swc_id
    """)

    # Assign numeric IDs for ordering (to compute only upper triangle)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE swc_numeric AS
        SELECT
            swc_id,
            ROW_NUMBER() OVER (ORDER BY swc_id) AS swc_num
        FROM (SELECT DISTINCT swc_id FROM counts_by_voxel)
    """)

    # Pairwise cross-products via voxel join (upper triangle only)
    _progress("Computing pairwise cross-products", 5)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE pairwise_xy AS
        SELECT
            a.swc_id AS i,
            b.swc_id AS j,
            SUM(a.c * b.c)::DOUBLE AS Sxy
        FROM counts_by_voxel a
        JOIN counts_by_voxel b
            ON a.voxel_id = b.voxel_id
        JOIN swc_numeric a_num ON a.swc_id = a_num.swc_id
        JOIN swc_numeric b_num ON b.swc_id = b_num.swc_id
        WHERE a_num.swc_num < b_num.swc_num
        GROUP BY a.swc_id, b.swc_id
    """)

    # Pearson correlation: r = (V*Sxy - Sx*Sy) / sqrt((V*Sxx - Sx^2)(V*Syy - Sy^2))
    _progress("Computing Pearson correlations", 6)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE corr_pairs AS
        WITH VU AS (SELECT V FROM voxel_universe)
        SELECT
            p.i,
            p.j,
            (VU.V * p.Sxy - x.Sx * y.Sx)
            / NULLIF(
                SQRT(
                    (VU.V * x.Sxx - x.Sx * x.Sx)
                    * (VU.V * y.Sxx - y.Sx * y.Sx)
                ),
                0
            ) AS r
        FROM pairwise_xy p
        JOIN per_neuron x ON p.i = x.swc_id
        JOIN per_neuron y ON p.j = y.swc_id
        CROSS JOIN VU
    """)

    # Build symmetric matrix (both triangles + diagonal)
    _progress("Building symmetric correlation table", 7)
    result_df = conn.execute("""
        SELECT i AS swc_id_1, j AS swc_id_2, r FROM corr_pairs
        UNION ALL
        SELECT j AS swc_id_1, i AS swc_id_2, r FROM corr_pairs
        UNION ALL
        SELECT swc_id AS swc_id_1, swc_id AS swc_id_2, 1.0 AS r FROM per_neuron
    """).fetchdf()

    # Clean up temp tables
    for table in [
        "counts_by_voxel",
        "voxel_universe",
        "per_neuron",
        "swc_numeric",
        "pairwise_xy",
        "corr_pairs",
    ]:
        try:
            conn.execute(f"DROP VIEW IF EXISTS {table}")
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass

    _cleanup_region_nodes_view(
        conn,
        lut_registered=lut_registered,
        scope_registered=scope_registered,
        filter_view_name=filter_view_name,
        filter_relations=filter_relations,
        voxel_filter_view_name=voxel_filter_view_name,
        voxel_filter_relations=voxel_filter_relations,
    )

    n_neurons = result_df["swc_id_1"].nunique()
    logger.info(
        f"Correlation matrix computed: {n_neurons} neurons, {len(result_df)} entries"
    )
    return result_df


def correlation_long_to_matrix(
    corr_df: pd.DataFrame,
) -> tuple[pd.DataFrame, NDArray[np.float32]]:
    """Pivot long-form correlation DataFrame to a square matrix.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Long-form table with columns: swc_id_1, swc_id_2, r.

    Returns
    -------
    tuple[pd.DataFrame, NDArray[np.float32]]
        (mat_df, mat) where mat_df is a square DataFrame with neuron IDs
        as index and columns, and mat is the dense float32 array.
    """
    ids = pd.Index(
        pd.unique(pd.concat([corr_df["swc_id_1"], corr_df["swc_id_2"]]))
    ).sort_values()

    mat_df = corr_df.pivot(index="swc_id_1", columns="swc_id_2", values="r").reindex(
        index=ids, columns=ids
    )

    # Fill any missing pairs with -1 (uncorrelated/missing)
    mat_df.fillna(-1.0, inplace=True)

    mat = mat_df.to_numpy(dtype=np.float32)
    if mat.size == 0:
        logger.info("Correlation matrix: 0x0")
        return mat_df, mat

    # Sanity checks
    if not np.allclose(mat, mat.T, equal_nan=True):
        logger.warning(
            "Correlation matrix is not perfectly symmetric; forcing symmetry"
        )
        mat = (mat + mat.T) / 2.0

    if not np.allclose(np.diag(mat), 1.0):
        logger.warning("Diagonal is not all 1.0; forcing diagonal to 1.0")
        np.fill_diagonal(mat, 1.0)

    logger.info(
        f"Correlation matrix: {mat.shape[0]}x{mat.shape[1]}, "
        f"range [{mat.min():.3f}, {mat.max():.3f}]"
    )
    return mat_df, mat
