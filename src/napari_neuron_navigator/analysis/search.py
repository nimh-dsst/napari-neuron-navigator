"""Rank neurons against a single or aggregate voxel-count reference."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

if TYPE_CHECKING:
    import duckdb

    from .clustering import ClusterRegionFilter
    from .region_filter import PreparedClusterRegionFilter
    from .voxel_filter import PreparedVoxelNodeFilter, VoxelNodeFilter

logger = logging.getLogger(__name__)


SEARCH_RESULTS_FORMAT_VERSION = 1
SEARCH_RESULTS_COLUMNS = (
    "format_version",
    "rank",
    "file_id",
    "neuron_id",
    "subject",
    "pearson_distance",
)


def _unique_strings(values) -> tuple[str, ...]:
    """Return stable, unique string values."""
    return tuple(dict.fromkeys(str(value) for value in values))


@dataclass(frozen=True)
class VoxelSearchRequest:
    """Immutable specification for one CCF voxel-correlation search."""

    reference_file_ids: tuple[str, ...]
    candidate_file_ids: tuple[str, ...] | None = None
    region_filter: ClusterRegionFilter | None = None
    voxel_node_filter: VoxelNodeFilter | None = None
    resolution_um: float = 25.0
    top_n: int = 100
    exclude_references: bool = True

    def __post_init__(self) -> None:
        references = _unique_strings(self.reference_file_ids)
        if not references:
            raise ValueError("Select at least one reference neuron.")
        object.__setattr__(self, "reference_file_ids", references)

        if self.candidate_file_ids is not None:
            object.__setattr__(
                self,
                "candidate_file_ids",
                _unique_strings(self.candidate_file_ids),
            )
        resolution = float(self.resolution_um)
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("resolution_um must be finite and positive.")
        object.__setattr__(self, "resolution_um", resolution)
        top_n = int(self.top_n)
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        object.__setattr__(self, "top_n", top_n)


@dataclass(frozen=True)
class VoxelSearchResult:
    """Ranked search hits plus counts and reproducibility metadata."""

    hits: pd.DataFrame
    reference_file_ids: tuple[str, ...]
    input_candidate_count: int
    usable_candidate_count: int
    omitted_candidate_file_ids: tuple[str, ...]
    retained_node_count: int
    metadata: dict[str, object] = field(default_factory=dict)


def _source_sql(parquet_path: str | Path) -> str:
    escaped = str(parquet_path).replace("\\", "/").replace("'", "''")
    return f"read_parquet('{escaped}')"


def _source_columns(
    conn: duckdb.DuckDBPyConnection,
    source_sql: str,
) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(f"DESCRIBE SELECT * FROM {source_sql}").fetchall()
    }


def query_neuron_catalog(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str | Path,
) -> pd.DataFrame:
    """Return one deterministic display row per unique ``file_id``."""
    source_sql = _source_sql(parquet_path)
    columns = _source_columns(conn, source_sql)
    if "file_id" not in columns:
        raise ValueError("The loaded Parquet does not contain a file_id column.")

    neuron_expr = (
        "COALESCE(MIN(CAST(neuron_id AS VARCHAR)), '')"
        if "neuron_id" in columns
        else "''"
    )
    subject_expr = (
        "COALESCE(MIN(CAST(subject AS VARCHAR)), '')"
        if "subject" in columns
        else "''"
    )
    return conn.execute(f"""
        SELECT
            CAST(file_id AS VARCHAR) AS file_id,
            {neuron_expr} AS neuron_id,
            {subject_expr} AS subject
        FROM {source_sql}
        WHERE file_id IS NOT NULL
        GROUP BY CAST(file_id AS VARCHAR)
        ORDER BY file_id
    """).fetchdf()


def _empty_hits() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "rank",
            "file_id",
            "neuron_id",
            "subject",
            "pearson_distance",
        ]
    )


def compute_voxel_search(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str | Path,
    request: VoxelSearchRequest,
    *,
    voxel_id_map: np.ndarray | None = None,
    prepared_region_filter: PreparedClusterRegionFilter | None = None,
    prepared_voxel_filter: PreparedVoxelNodeFilter | None = None,
    progress_callback=None,
) -> VoxelSearchResult:
    """Rank candidate neurons by Pearson distance from an aggregate query.

    The CCF voxel universe and missing-correlation policy intentionally match
    :func:`compute_pearson_correlation_matrix` plus
    :func:`correlation_long_to_matrix`. In particular, a candidate with no
    occupied voxel shared with the query, or an undefined Pearson denominator,
    receives ``r = -1`` and therefore distance ``2``.
    """
    from .correlation import _cleanup_region_nodes_view, _prepare_region_nodes_view

    def progress(message: str, current: int, total: int = 5) -> None:
        if progress_callback is not None:
            progress_callback(message, current, total)

    progress("Reading neuron catalog...", 1)
    catalog = query_neuron_catalog(conn, parquet_path)
    catalog_ids = tuple(catalog["file_id"].astype(str).tolist())
    catalog_set = set(catalog_ids)
    missing_references = [
        file_id
        for file_id in request.reference_file_ids
        if file_id not in catalog_set
    ]
    if missing_references:
        raise ValueError(
            "Reference file_id value(s) are not present in the loaded Parquet: "
            + ", ".join(missing_references)
        )

    if request.candidate_file_ids is None:
        requested_candidate_ids = catalog_ids
        scope_file_ids = None
    else:
        requested_candidate_ids = tuple(
            file_id
            for file_id in request.candidate_file_ids
            if file_id in catalog_set
        )
        scope_file_ids = list(
            dict.fromkeys((*requested_candidate_ids, *request.reference_file_ids))
        )

    reference_set = set(request.reference_file_ids)
    result_candidate_ids = tuple(
        file_id
        for file_id in requested_candidate_ids
        if not request.exclude_references or file_id not in reference_set
    )
    if not result_candidate_ids:
        raise ValueError("No non-reference candidate neurons are available.")

    progress("Applying region and node filters...", 2)
    prepared = _prepare_region_nodes_view(
        conn,
        str(parquet_path),
        voxel_id_map,
        request.resolution_um,
        scope_file_ids,
        prepared_region_filter,
        request.voxel_node_filter,
        prepared_voxel_filter,
    )
    (
        lut_registered,
        scope_registered,
        filter_view_name,
        filter_relations,
        voxel_filter_view_name,
        voxel_filter_relations,
    ) = prepared

    registered_reference_ids = False
    registered_candidate_ids = False
    try:
        progress("Counting nodes per neuron and voxel...", 3)
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE search_counts_by_voxel AS
            SELECT
                CAST(swc_id AS VARCHAR) AS file_id,
                voxel_id,
                COUNT(*)::DOUBLE AS c
            FROM region_nodes
            GROUP BY file_id, voxel_id
        """)
        retained_row = conn.execute(
            "SELECT COALESCE(SUM(c), 0)::BIGINT FROM search_counts_by_voxel"
        ).fetchone()
        retained_node_count = int(retained_row[0] or 0) if retained_row else 0
        usable_ids = {
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT file_id FROM search_counts_by_voxel"
            ).fetchall()
        }
        unusable_references = [
            file_id
            for file_id in request.reference_file_ids
            if file_id not in usable_ids
        ]
        if unusable_references:
            detail = ""
            if (
                request.voxel_node_filter is not None
                and request.voxel_node_filter.exclude_within_soma_um is not None
            ):
                detail = " They may lack a valid soma for the active soma-distance filter."
            raise ValueError(
                "Reference neuron(s) have no usable nodes after applying the "
                "selected filters: "
                + ", ".join(unusable_references)
                + "."
                + detail
            )

        omitted = tuple(
            file_id for file_id in result_candidate_ids if file_id not in usable_ids
        )
        usable_candidate_ids = tuple(
            file_id for file_id in result_candidate_ids if file_id in usable_ids
        )
        if not usable_candidate_ids:
            raise ValueError(
                "No candidate neurons have usable nodes after applying the "
                "selected filters."
            )

        conn.register(
            "search_reference_ids",
            pa.table({"file_id": list(request.reference_file_ids)}),
        )
        registered_reference_ids = True
        conn.register(
            "search_candidate_ids",
            pa.table({"file_id": list(usable_candidate_ids)}),
        )
        registered_candidate_ids = True

        progress("Building the aggregate reference vector...", 4)
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE search_query_counts AS
            SELECT c.voxel_id, SUM(c.c)::DOUBLE AS c
            FROM search_counts_by_voxel c
            JOIN search_reference_ids r USING (file_id)
            GROUP BY c.voxel_id
        """)
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE search_candidate_stats AS
            SELECT
                c.file_id,
                SUM(c.c)::DOUBLE AS sx,
                SUM(c.c * c.c)::DOUBLE AS sxx
            FROM search_counts_by_voxel c
            JOIN search_candidate_ids candidates USING (file_id)
            GROUP BY c.file_id
        """)
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE search_cross_products AS
            SELECT
                c.file_id,
                SUM(c.c * q.c)::DOUBLE AS sxy
            FROM search_counts_by_voxel c
            JOIN search_candidate_ids candidates USING (file_id)
            JOIN search_query_counts q USING (voxel_id)
            GROUP BY c.file_id
        """)

        progress("Computing and ranking Pearson distances...", 5)
        score_frame = conn.execute(
            """
            WITH
            voxel_universe AS (
                SELECT COUNT(DISTINCT voxel_id)::DOUBLE AS v
                FROM search_counts_by_voxel
            ),
            query_stats AS (
                SELECT
                    SUM(c)::DOUBLE AS sy,
                    SUM(c * c)::DOUBLE AS syy
                FROM search_query_counts
            )
            SELECT
                cstats.file_id,
                COALESCE(
                    (
                        vu.v * xprod.sxy - cstats.sx * qstats.sy
                    ) / NULLIF(
                        SQRT(
                            (vu.v * cstats.sxx - cstats.sx * cstats.sx)
                            * (vu.v * qstats.syy - qstats.sy * qstats.sy)
                        ),
                        0
                    ),
                    -1.0
                ) AS pearson_r
            FROM search_candidate_stats cstats
            LEFT JOIN search_cross_products xprod USING (file_id)
            CROSS JOIN voxel_universe vu
            CROSS JOIN query_stats qstats
            """
        ).fetchdf()

        if score_frame.empty:
            hits = _empty_hits()
        else:
            correlations = np.clip(
                pd.to_numeric(score_frame["pearson_r"], errors="coerce")
                .fillna(-1.0)
                .to_numpy(dtype=np.float64),
                -1.0,
                1.0,
            )
            score_frame["pearson_distance"] = 1.0 - correlations
            score_frame["file_id"] = score_frame["file_id"].astype(str)
            score_frame = score_frame.sort_values(
                ["pearson_distance", "file_id"],
                kind="mergesort",
            ).head(request.top_n)
            hits = score_frame[["file_id", "pearson_distance"]].merge(
                catalog,
                on="file_id",
                how="left",
                validate="one_to_one",
            )
            hits.insert(0, "rank", np.arange(1, len(hits) + 1, dtype=np.int64))
            hits = hits[
                [
                    "rank",
                    "file_id",
                    "neuron_id",
                    "subject",
                    "pearson_distance",
                ]
            ].reset_index(drop=True)

        voxel_filter_metadata = None
        if request.voxel_node_filter is not None:
            voxel_filter_metadata = (
                prepared_voxel_filter.metadata()
                if prepared_voxel_filter is not None
                else request.voxel_node_filter.to_dict()
            )
        metadata: dict[str, object] = {
            "format_version": 1,
            "analysis_method": "voxel_similarity_search",
            "coordinate_space": "ccfv3",
            "distance_metric": "one_minus_pearson_r",
            "missing_correlation_policy": "pearson_r_minus_one",
            "aggregate_mode": "sum_voxel_counts",
            "source_parquet_path": str(Path(parquet_path)),
            "resolution_um": float(request.resolution_um),
            "reference_file_ids": list(request.reference_file_ids),
            "top_n": int(request.top_n),
            "exclude_references": bool(request.exclude_references),
            "region_filter": (
                None
                if request.region_filter is None
                else request.region_filter.to_dict()
            ),
            "voxel_node_filter": voxel_filter_metadata,
        }
        return VoxelSearchResult(
            hits=hits,
            reference_file_ids=request.reference_file_ids,
            input_candidate_count=len(result_candidate_ids),
            usable_candidate_count=len(usable_candidate_ids),
            omitted_candidate_file_ids=omitted,
            retained_node_count=retained_node_count,
            metadata=metadata,
        )
    finally:
        for relation in (
            "search_cross_products",
            "search_candidate_stats",
            "search_query_counts",
            "search_counts_by_voxel",
        ):
            try:
                conn.execute(f"DROP TABLE IF EXISTS {relation}")
            except Exception:
                logger.debug("Could not drop search relation %s", relation, exc_info=True)
        if registered_candidate_ids:
            try:
                conn.unregister("search_candidate_ids")
            except Exception:
                logger.debug(
                    "Could not unregister search candidate IDs",
                    exc_info=True,
                )
        if registered_reference_ids:
            try:
                conn.unregister("search_reference_ids")
            except Exception:
                logger.debug(
                    "Could not unregister search reference IDs",
                    exc_info=True,
                )
        _cleanup_region_nodes_view(
            conn,
            lut_registered=lut_registered,
            scope_registered=scope_registered,
            filter_view_name=filter_view_name,
            filter_relations=filter_relations,
            voxel_filter_view_name=voxel_filter_view_name,
            voxel_filter_relations=voxel_filter_relations,
        )


def export_search_results_csv(
    output_path: str | Path,
    result: VoxelSearchResult | pd.DataFrame,
) -> Path:
    """Atomically export ranked results using the versioned CSV schema."""
    path = Path(output_path)
    hits = result.hits if isinstance(result, VoxelSearchResult) else result
    required = {"file_id", "pearson_distance"}
    missing = required - set(hits.columns)
    if missing:
        raise ValueError(
            "Search results are missing required column(s): "
            + ", ".join(sorted(missing))
        )

    frame = hits.copy()
    frame["file_id"] = frame["file_id"].astype(str)
    if "rank" not in frame:
        frame["rank"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    for column in ("neuron_id", "subject"):
        if column not in frame:
            frame[column] = ""
    frame["format_version"] = SEARCH_RESULTS_FORMAT_VERSION
    frame = frame[list(SEARCH_RESULTS_COLUMNS)]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            prefix=f".{path.stem}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            frame.to_csv(handle, index=False)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def load_search_results_csv(
    input_path: str | Path,
    *,
    available_file_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Load, validate, rank, and mark availability for a Search CSV."""
    frame = pd.read_csv(
        input_path,
        dtype={"file_id": "string", "neuron_id": "string", "subject": "string"},
    )
    required = {"format_version", "file_id", "pearson_distance"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Search CSV is missing required column(s): "
            + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise ValueError("Search CSV contains no result rows.")

    versions = pd.to_numeric(frame["format_version"], errors="coerce")
    if versions.isna().any() or not np.all(
        versions.to_numpy(dtype=float) == SEARCH_RESULTS_FORMAT_VERSION
    ):
        raise ValueError(
            "Unsupported search result format version; expected version "
            f"{SEARCH_RESULTS_FORMAT_VERSION}."
        )
    file_ids = frame["file_id"].astype("string")
    if file_ids.isna().any() or (file_ids.str.strip() == "").any():
        raise ValueError("Search CSV contains an empty file_id.")
    frame["file_id"] = file_ids.astype(str)
    duplicates = frame.loc[frame["file_id"].duplicated(), "file_id"].unique()
    if len(duplicates):
        raise ValueError(
            "Search CSV contains duplicate file_id value(s): "
            + ", ".join(str(value) for value in duplicates[:10])
        )

    distances = pd.to_numeric(frame["pearson_distance"], errors="coerce")
    finite = np.isfinite(distances.to_numpy(dtype=float))
    in_range = distances.between(0.0, 2.0, inclusive="both").to_numpy()
    if not np.all(finite & in_range):
        raise ValueError(
            "Search CSV pearson_distance values must be finite and between 0 and 2."
        )
    frame["pearson_distance"] = distances.astype(float)
    for column in ("neuron_id", "subject"):
        if column not in frame:
            frame[column] = ""
        else:
            frame[column] = frame[column].fillna("").astype(str)

    frame = frame.sort_values(
        ["pearson_distance", "file_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    if available_file_ids is None:
        frame["available"] = True
    else:
        available = {str(value) for value in available_file_ids}
        frame["available"] = frame["file_id"].isin(available)
    return frame[
        [
            "rank",
            "file_id",
            "neuron_id",
            "subject",
            "pearson_distance",
            "available",
        ]
    ]
