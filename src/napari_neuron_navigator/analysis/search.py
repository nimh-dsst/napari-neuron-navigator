"""Rank neurons against a single or aggregate voxel-count reference."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Mapping
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


SEARCH_RESULTS_FORMAT_VERSION = 2
SEARCH_RESULTS_LEGACY_FORMAT_VERSION = 1
SEARCH_ROW_REFERENCE = "reference"
SEARCH_ROW_RESULT = "search_result"
SEARCH_REFERENCE_HEATMAP_RGBA = (1.0, 0.0, 1.0, 1.0)
SEARCH_HEATMAP_MODE_SCORED = "scored_voxels"
SEARCH_HEATMAP_MODE_WHOLE = "whole_neuron"
SEARCH_HEATMAP_MODE_LABELS = {
    SEARCH_HEATMAP_MODE_SCORED: "Scored Voxels",
    SEARCH_HEATMAP_MODE_WHOLE: "Whole Neuron",
}
SEARCH_SCOPE_WHOLE = "whole_parquet"
SEARCH_SCOPE_CURRENT = "current_table"
SEARCH_SCOPE_SELECTED = "selected_rows"
SEARCH_SCOPE_LABELS = {
    SEARCH_SCOPE_WHOLE: "Whole Parquet",
    SEARCH_SCOPE_CURRENT: "Current Table",
    SEARCH_SCOPE_SELECTED: "Selected Rows",
}
SEARCH_SPACE_CCF = "ccfv3"
SEARCH_SPACE_FLATMAP = "flatmap"
SEARCH_SPACE_LABELS = {
    SEARCH_SPACE_CCF: "CCFv3 Coordinates",
    SEARCH_SPACE_FLATMAP: "Flat map + Depth",
}
SEARCH_RESULTS_COLUMNS = (
    "format_version",
    "row_role",
    "rank",
    "file_id",
    "neuron_id",
    "subject",
    "pearson_distance",
    "search_context_json",
)


def _unique_strings(values) -> tuple[str, ...]:
    """Return stable, unique string values."""
    return tuple(dict.fromkeys(str(value) for value in values))


def _format_percent(value: object) -> str:
    percent = float(value or 0.0) * 100.0
    return f"{percent:g}%"


def build_search_filter_tags(
    candidate_scope: str,
    region_filter: Mapping[str, object] | None,
    voxel_node_filter: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Return deterministic, human-readable tags for completed Search input."""
    tags = [
        f"Search scope: {SEARCH_SCOPE_LABELS.get(candidate_scope, candidate_scope)}"
    ]
    active_filter = False
    region_payload = region_filter or {}
    for role, prefix in (("include_rules", "include"), ("exclude_rules", "exclude")):
        rules = region_payload.get(role, ())
        if not isinstance(rules, list | tuple):
            continue
        for raw_rule in rules:
            if not isinstance(raw_rule, Mapping):
                continue
            acronym = str(raw_rule.get("acronym", "")).strip()
            if not acronym:
                continue
            dilation = float(raw_rule.get("dilation_fraction", 0.0) or 0.0)
            suffix = f" (+{_format_percent(dilation)})" if dilation > 0.0 else ""
            tags.append(f"Search {prefix}: {acronym}{suffix}")
            active_filter = True

    voxel_payload = voxel_node_filter or {}
    mode = str(voxel_payload.get("node_type_mode", "all"))
    if mode in {"include", "exclude"}:
        raw_types = voxel_payload.get("node_types", ())
        try:
            type_text = "|".join(str(int(value)) for value in raw_types)
        except TypeError:
            type_text = ""
        if type_text:
            tags.append(f"Search node types: {mode} {type_text}")
            active_filter = True
    if bool(voxel_payload.get("require_dendrite_labels", False)):
        tags.append("Search dendrite labels: required")
        active_filter = True
    radius = voxel_payload.get("exclude_within_soma_um")
    if radius is not None:
        tags.append(f"Search soma distance: >={float(radius):g} um")
        active_filter = True
    if not active_filter:
        tags.append("Search filters: none")
    return tuple(tags)


def _search_context(metadata: Mapping[str, object]) -> dict[str, object]:
    """Return the compact, portable subset of Search run metadata."""
    keys = (
        "analysis_method",
        "coordinate_space",
        "distance_metric",
        "missing_correlation_policy",
        "aggregate_mode",
        "resolution_um",
        "flatmap_style",
        "flatmap_y_bins",
        "flatmap_x_bins",
        "flatmap_depth_bin_um",
        "flatmap_include_depth_minus_one",
        "flatmap_collapse_depth",
        "flatmap_volume_shape",
        "flatmap_x_bounds",
        "flatmap_y_bounds",
        "flatmap_depth_range_um",
        "candidate_scope",
        "candidate_scope_label",
        "candidate_scope_input_count",
        "input_candidate_count",
        "region_filter",
        "voxel_node_filter",
        "filter_tags",
        "atlas_name",
        "atlas_resolution_um",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _canonical_context_json(metadata: Mapping[str, object]) -> str:
    return json.dumps(
        _search_context(metadata),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def cluster_region_filter_from_dict(
    payload: Mapping[str, object] | None,
) -> ClusterRegionFilter | None:
    """Reconstruct a region filter from exported Search context."""
    if not payload:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("Search region-filter context must contain an object.")
    from .clustering import (
        ClusterExclusionRule,
        ClusterRegionFilter,
        ClusterRegionRule,
    )

    def common(raw: Mapping[str, object]) -> dict[str, object]:
        return {
            "region_id": int(raw["region_id"]),
            "acronym": str(raw["acronym"]),
            "represented_region_ids": tuple(
                int(value) for value in raw.get("represented_region_ids", ())
            ),
            "represented_region_acronyms": tuple(
                str(value) for value in raw.get("represented_region_acronyms", ())
            ),
            "dilation_fraction": float(raw.get("dilation_fraction", 0.0)),
        }

    include_rules = tuple(
        ClusterRegionRule(**common(raw))
        for raw in payload.get("include_rules", ())
        if isinstance(raw, Mapping)
    )
    exclude_rules = tuple(
        ClusterExclusionRule(
            **common(raw),
            node_types=(
                None
                if raw.get("node_types") is None
                else tuple(int(value) for value in raw.get("node_types", ()))
            ),
            minimum_node_count=int(raw.get("minimum_node_count", 1)),
        )
        for raw in payload.get("exclude_rules", ())
        if isinstance(raw, Mapping)
    )
    result = ClusterRegionFilter(
        include_rules=include_rules,
        exclude_rules=exclude_rules,
    )
    return None if result.is_empty else result


def voxel_node_filter_from_dict(
    payload: Mapping[str, object] | None,
) -> VoxelNodeFilter | None:
    """Reconstruct voxel-node settings from exported Search context."""
    if not payload:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("Search voxel-filter context must contain an object.")
    from .voxel_filter import VoxelNodeFilter

    result = VoxelNodeFilter(
        node_type_mode=str(payload.get("node_type_mode", "all")),
        node_types=tuple(int(value) for value in payload.get("node_types", ())),
        require_dendrite_labels=bool(payload.get("require_dendrite_labels", False)),
        exclude_within_soma_um=payload.get("exclude_within_soma_um"),
        dendrite_node_types=tuple(
            int(value) for value in payload.get("dendrite_node_types", (3, 4))
        ),
    )
    return None if result.is_empty else result


def pearson_distance_color_domain(values) -> tuple[float, float]:
    """Return the finite observed Pearson-distance range for result rows."""
    distances = pd.to_numeric(pd.Series(values, copy=False), errors="coerce").to_numpy(
        dtype=float
    )
    if distances.size == 0 or not np.isfinite(distances).all():
        raise ValueError(
            "Search heatmap colors require at least one finite Pearson distance."
        )
    if np.any((distances < 0.0) | (distances > 2.0)):
        raise ValueError("Pearson distances must be between 0 and 2.")
    return float(distances.min()), float(distances.max())


def pearson_distance_to_hot_rgba(
    pearson_distance: float,
    *,
    distance_domain: tuple[float, float] = (0.0, 2.0),
) -> tuple[float, float, float, float]:
    """Map a distance domain to a visible reversed-hot RGBA color."""
    distance = float(pearson_distance)
    if not np.isfinite(distance):
        raise ValueError("Pearson distance must be finite.")
    try:
        distance_min, distance_max = (
            float(distance_domain[0]),
            float(distance_domain[1]),
        )
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(
            "Pearson distance color domain must contain two finite values."
        ) from error
    if not np.isfinite(distance_min) or not np.isfinite(distance_max):
        raise ValueError(
            "Pearson distance color domain must contain two finite values."
        )
    if distance_max < distance_min:
        raise ValueError(
            "Pearson distance color domain maximum must not be below its minimum."
        )
    if distance_max == distance_min:
        similarity = 1.0
    else:
        normalized = (distance - distance_min) / (distance_max - distance_min)
        similarity = float(np.clip(1.0 - normalized, 0.0, 1.0))
    coordinate = 0.25 + 0.75 * similarity
    red = min(1.0, 3.0 * coordinate)
    green = min(1.0, max(0.0, 3.0 * coordinate - 1.0))
    blue = min(1.0, max(0.0, 3.0 * coordinate - 2.0))
    return (red, green, blue, 1.0)


@dataclass(frozen=True)
class VoxelSearchRequest:
    """Immutable specification for one CCF or flatmap voxel search."""

    reference_file_ids: tuple[str, ...]
    candidate_file_ids: tuple[str, ...] | None = None
    region_filter: ClusterRegionFilter | None = None
    voxel_node_filter: VoxelNodeFilter | None = None
    resolution_um: float = 25.0
    top_n: int = 100
    exclude_references: bool = True
    candidate_scope: str = SEARCH_SCOPE_WHOLE
    coordinate_space: str = SEARCH_SPACE_CCF
    flatmap_style: str | None = None
    flatmap_y_bins: int = 256
    flatmap_x_bins: int | None = None
    flatmap_depth_bin_um: float = 25.0
    flatmap_include_depth_minus_one: bool = True
    flatmap_collapse_depth: bool = False

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
        scope = str(self.candidate_scope)
        if scope not in SEARCH_SCOPE_LABELS:
            raise ValueError(f"Unknown search candidate scope: {scope!r}")
        object.__setattr__(self, "candidate_scope", scope)
        coordinate_space = str(self.coordinate_space)
        if coordinate_space not in SEARCH_SPACE_LABELS:
            raise ValueError(f"Unknown search coordinate space: {coordinate_space!r}")
        object.__setattr__(self, "coordinate_space", coordinate_space)
        if coordinate_space == SEARCH_SPACE_FLATMAP:
            if not self.flatmap_style:
                raise ValueError("Select a flatmap style for flatmap search.")
            y_bins = int(self.flatmap_y_bins)
            if y_bins < 1:
                raise ValueError("flatmap_y_bins must be at least 1.")
            object.__setattr__(self, "flatmap_y_bins", y_bins)
            if self.flatmap_x_bins is not None:
                x_bins = int(self.flatmap_x_bins)
                if x_bins < 1:
                    raise ValueError("flatmap_x_bins must be at least 1.")
                object.__setattr__(self, "flatmap_x_bins", x_bins)
            depth_bin_um = float(self.flatmap_depth_bin_um)
            if not np.isfinite(depth_bin_um) or depth_bin_um <= 0.0:
                raise ValueError("flatmap_depth_bin_um must be finite and positive.")
            object.__setattr__(self, "flatmap_depth_bin_um", depth_bin_um)


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
    reference_rows: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class SearchResultsDocument:
    """References, ranked hits, and context loaded from one Search CSV."""

    references: pd.DataFrame
    hits: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)
    format_version: int = SEARCH_RESULTS_FORMAT_VERSION


@dataclass(frozen=True)
class SearchAnnotationRequest:
    """Immutable Data-table annotation request for one completed search."""

    reference_file_ids: tuple[str, ...]
    ranked_hits: tuple[tuple[str, int], ...]
    filter_tags: tuple[str, ...]
    unavailable_file_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchTableColorRequest:
    """Exact Search colors to apply to matching neurons already in Data."""

    colors_by_file_id: tuple[tuple[str, tuple[float, float, float, float]], ...]
    reference_file_ids: tuple[str, ...] = ()
    distance_color_domain: tuple[float, float] = (0.0, 2.0)


@dataclass(frozen=True)
class SearchHeatmapLayerRequest:
    """One reference or ranked-result volume requested from Search."""

    file_ids: tuple[str, ...]
    role: str
    rank: int
    pearson_distance: float | None
    color: tuple[float, float, float, float]


@dataclass(frozen=True)
class SearchHeatmapRequest:
    """Immutable Search heatmap batch with an explicit voxel-display mode."""

    layers: tuple[SearchHeatmapLayerRequest, ...]
    region_filter: ClusterRegionFilter | None
    voxel_node_filter: VoxelNodeFilter | None
    resolution_um: float
    metadata: dict[str, object] = field(default_factory=dict)
    distance_color_domain: tuple[float, float] = (0.0, 2.0)
    voxel_mode: str = SEARCH_HEATMAP_MODE_SCORED

    def __post_init__(self) -> None:
        voxel_mode = str(self.voxel_mode)
        if voxel_mode not in SEARCH_HEATMAP_MODE_LABELS:
            raise ValueError(f"Unknown Search heatmap voxel mode: {voxel_mode!r}")
        object.__setattr__(self, "voxel_mode", voxel_mode)
        try:
            distance_min, distance_max = (
                float(self.distance_color_domain[0]),
                float(self.distance_color_domain[1]),
            )
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                "Search heatmap distance color domain must contain two values."
            ) from error
        if (
            not np.isfinite(distance_min)
            or not np.isfinite(distance_max)
            or distance_min < 0.0
            or distance_max > 2.0
            or distance_max < distance_min
        ):
            raise ValueError(
                "Search heatmap distance color domain must be finite, ordered, "
                "and within 0..2."
            )
        object.__setattr__(
            self,
            "distance_color_domain",
            (distance_min, distance_max),
        )


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
        "COALESCE(MIN(CAST(subject AS VARCHAR)), '')" if "subject" in columns else "''"
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

    CCF searches match :func:`compute_pearson_correlation_matrix`; flatmap
    searches use the same rectangular grid and bin expressions as Analysis.
    CCF preserves its historical ``r = -1`` fallback for no shared occupancy
    or an undefined denominator. Flatmap follows Analysis's dense zero-filled
    vectors: no-overlap cross-products are zero and an undefined denominator
    produces ``r = 0``.
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
        file_id for file_id in request.reference_file_ids if file_id not in catalog_set
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
            file_id for file_id in request.candidate_file_ids if file_id in catalog_set
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

    lut_registered = False
    scope_registered = False
    filter_view_name: str | None = None
    filter_relations: tuple[str, ...] = ()
    voxel_filter_view_name: str | None = None
    voxel_filter_relations: tuple[str, ...] = ()
    flatmap_scope_registered = False
    flatmap_scope_view_created = False
    flatmap_provenance: dict[str, object] = {}
    registered_reference_ids = False
    registered_candidate_ids = False
    try:
        progress("Applying region and node filters...", 2)
        if request.coordinate_space == SEARCH_SPACE_CCF:
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
            progress("Counting nodes per neuron and CCFv3 voxel...", 3)
            conn.execute("""
                CREATE OR REPLACE TEMP TABLE search_counts_by_voxel AS
                SELECT
                    CAST(swc_id AS VARCHAR) AS file_id,
                    voxel_id,
                    COUNT(*)::DOUBLE AS c
                FROM region_nodes
                GROUP BY file_id, voxel_id
            """)
        else:
            from ..flatmap_heatmap import (
                MAX_FLATMAP_HEATMAP_VOXELS,
                _depth_bin_count,
                _duckdb_column_names,
                _flatmap_sql_expressions,
                _nondegenerate_bounds,
                _resolve_axis_bin_counts,
                _style_suffix,
            )
            from .flatmap_correlation import _resolve_flatmap_render_bounds
            from .region_filter import register_filtered_source_view
            from .voxel_filter import (
                prepare_voxel_node_filter,
                register_voxel_filtered_source_view,
            )

            style = str(request.flatmap_style)
            suffix = _style_suffix(style)
            x_bounds, y_bounds, depth_range = _resolve_flatmap_render_bounds(
                str(parquet_path), style, suffix
            )
            x_lower, x_upper = _nondegenerate_bounds(*x_bounds)
            y_lower, y_upper = _nondegenerate_bounds(*y_bounds)
            depth_lower, depth_upper = _nondegenerate_bounds(*depth_range)
            y_bins, x_bins, depth_bin_um = _resolve_axis_bin_counts(
                x_bounds=(x_lower, x_upper),
                y_bounds=(y_lower, y_upper),
                y_bins=request.flatmap_y_bins,
                x_bins=request.flatmap_x_bins,
                depth_bin_um=request.flatmap_depth_bin_um,
            )
            valid_depth_bins = _depth_bin_count(
                (depth_lower, depth_upper), depth_bin_um
            )
            sentinel_offset = 1 if request.flatmap_include_depth_minus_one else 0
            total_depth_bins = valid_depth_bins + sentinel_offset
            volume_shape: tuple[int, ...] = (
                (y_bins, x_bins)
                if request.flatmap_collapse_depth
                else (total_depth_bins, y_bins, x_bins)
            )
            if int(np.prod(volume_shape)) > MAX_FLATMAP_HEATMAP_VOXELS:
                shape_text = "x".join(str(int(size)) for size in volume_shape)
                raise ValueError(
                    f"Flatmap voxel grid is too large: {shape_text} voxels. "
                    "Use fewer Y bins or a larger depth bin."
                )

            raw_source_sql = _source_sql(parquet_path)
            source_sql = raw_source_sql
            filter_view_name = "search_flatmap_region_filtered_source"
            source_sql, filter_relations = register_filtered_source_view(
                conn,
                source_sql,
                prepared_region_filter,
                view_name=filter_view_name,
            )
            voxel_filter_view_name = "search_flatmap_voxel_filtered_source"
            if (
                request.voxel_node_filter is not None
                and not request.voxel_node_filter.is_empty
            ):
                if prepared_voxel_filter is None:
                    prepared_voxel_filter = prepare_voxel_node_filter(
                        conn,
                        raw_source_sql,
                        request.voxel_node_filter,
                        file_ids=scope_file_ids,
                    )
                elif prepared_voxel_filter.settings != request.voxel_node_filter:
                    raise ValueError(
                        "Prepared voxel filter does not match the requested settings."
                    )
                source_sql, voxel_filter_relations = (
                    register_voxel_filtered_source_view(
                        conn,
                        source_sql,
                        prepared_voxel_filter,
                        view_name=voxel_filter_view_name,
                    )
                )
            if scope_file_ids is not None:
                conn.register(
                    "search_flatmap_scope_ids",
                    pa.table({"file_id": [str(value) for value in scope_file_ids]}),
                )
                flatmap_scope_registered = True
                conn.execute(f"""
                    CREATE OR REPLACE TEMP VIEW search_flatmap_scoped_source AS
                    SELECT p.*
                    FROM {source_sql} p
                    JOIN search_flatmap_scope_ids scope_ids
                      ON CAST(p.file_id AS VARCHAR) = scope_ids.file_id
                """)
                source_sql = "search_flatmap_scoped_source"
                flatmap_scope_view_created = True

            expressions = _flatmap_sql_expressions(
                _duckdb_column_names(conn, source_sql),
                suffix=suffix,
                x_lower=x_lower,
                x_upper=x_upper,
                y_lower=y_lower,
                y_upper=y_upper,
                depth_lower=depth_lower,
                depth_bin_um=depth_bin_um,
                y_bins=y_bins,
                x_bins=x_bins,
                valid_depth_bins=valid_depth_bins,
                sentinel_offset=sentinel_offset,
                include_depth_minus_one=request.flatmap_include_depth_minus_one,
            )
            xy_voxel = (
                f"(({expressions['y_bin']}) * {int(x_bins)}::BIGINT "
                f"+ ({expressions['x_bin']}))"
            )
            voxel_id = (
                xy_voxel
                if request.flatmap_collapse_depth
                else (
                    f"(({expressions['depth_bin']}) * "
                    f"{int(y_bins * x_bins)}::BIGINT + {xy_voxel})"
                )
            )
            progress("Counting nodes per neuron and flatmap voxel...", 3)
            conn.execute(f"""
                CREATE OR REPLACE TEMP TABLE search_counts_by_voxel AS
                SELECT
                    CAST(file_id AS VARCHAR) AS file_id,
                    {voxel_id} AS voxel_id,
                    COUNT(*)::DOUBLE AS c
                FROM {source_sql}
                WHERE {expressions["render_where"]}
                GROUP BY file_id, voxel_id
            """)
            flatmap_provenance = {
                "flatmap_style": style,
                "flatmap_y_bins": int(y_bins),
                "flatmap_x_bins": int(x_bins),
                "flatmap_depth_bin_um": float(depth_bin_um),
                "flatmap_include_depth_minus_one": bool(
                    request.flatmap_include_depth_minus_one
                ),
                "flatmap_collapse_depth": bool(request.flatmap_collapse_depth),
                "flatmap_volume_shape": [int(size) for size in volume_shape],
                "flatmap_x_bounds": [float(x_lower), float(x_upper)],
                "flatmap_y_bounds": [float(y_lower), float(y_upper)],
                "flatmap_depth_range_um": [float(depth_lower), float(depth_upper)],
            }
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
                detail = (
                    " They may lack a valid soma for the active soma-distance filter."
                )
            raise ValueError(
                "Reference neuron(s) have no usable nodes after applying the "
                "selected filters: " + ", ".join(unusable_references) + "." + detail
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
        cross_product = (
            "xprod.sxy"
            if request.coordinate_space == SEARCH_SPACE_CCF
            else "COALESCE(xprod.sxy, 0.0)"
        )
        undefined_correlation = (
            -1.0 if request.coordinate_space == SEARCH_SPACE_CCF else 0.0
        )
        score_frame = conn.execute(
            f"""
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
                        vu.v * {cross_product} - cstats.sx * qstats.sy
                    ) / NULLIF(
                        SQRT(
                            (vu.v * cstats.sxx - cstats.sx * cstats.sx)
                            * (vu.v * qstats.syy - qstats.sy * qstats.sy)
                        ),
                        0
                    ),
                    {undefined_correlation!r}
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
                .fillna(undefined_correlation)
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
        region_filter_metadata = (
            None if request.region_filter is None else request.region_filter.to_dict()
        )
        filter_tags = list(
            build_search_filter_tags(
                request.candidate_scope,
                region_filter_metadata,
                voxel_filter_metadata,
            )
        )
        if request.coordinate_space == SEARCH_SPACE_FLATMAP:
            depth_text = (
                "collapsed"
                if request.flatmap_collapse_depth
                else f"{flatmap_provenance['flatmap_depth_bin_um']:g} um bins"
            )
            filter_tags[1:1] = [
                "Search space: Flat map + Depth",
                f"Search flatmap style: {flatmap_provenance['flatmap_style']}",
                (
                    "Search flatmap grid: "
                    f"{flatmap_provenance['flatmap_y_bins']} Y x "
                    f"{flatmap_provenance['flatmap_x_bins']} X; depth {depth_text}"
                ),
            ]
        metadata: dict[str, object] = {
            "format_version": SEARCH_RESULTS_FORMAT_VERSION,
            "analysis_method": "voxel_similarity_search",
            "coordinate_space": request.coordinate_space,
            "distance_metric": "one_minus_pearson_r",
            "missing_correlation_policy": (
                "pearson_r_minus_one"
                if request.coordinate_space == SEARCH_SPACE_CCF
                else "flatmap_dense_zero_filled_r_zero_if_undefined"
            ),
            "aggregate_mode": "sum_voxel_counts",
            "source_parquet_path": str(Path(parquet_path)),
            "resolution_um": float(request.resolution_um),
            "reference_file_ids": list(request.reference_file_ids),
            "top_n": int(request.top_n),
            "exclude_references": bool(request.exclude_references),
            "candidate_scope": request.candidate_scope,
            "candidate_scope_label": (
                f"{SEARCH_SCOPE_LABELS[request.candidate_scope]} "
                f"({len(requested_candidate_ids):,} rows; "
                f"{len(result_candidate_ids):,} non-reference candidates)"
            ),
            "candidate_scope_input_count": len(requested_candidate_ids),
            "input_candidate_count": len(result_candidate_ids),
            # Kept in memory for reproducibility. The compact CSV context omits
            # this potentially large list and retains the scope/count instead.
            "candidate_file_ids": list(requested_candidate_ids),
            "region_filter": region_filter_metadata,
            "voxel_node_filter": voxel_filter_metadata,
            "filter_tags": filter_tags,
        }
        metadata.update(flatmap_provenance)
        catalog_by_file_id = catalog.set_index("file_id", drop=False)
        reference_rows = pd.DataFrame(
            [
                catalog_by_file_id.loc[file_id].to_dict()
                for file_id in request.reference_file_ids
            ]
        ).reset_index(drop=True)
        return VoxelSearchResult(
            hits=hits,
            reference_file_ids=request.reference_file_ids,
            input_candidate_count=len(result_candidate_ids),
            usable_candidate_count=len(usable_candidate_ids),
            omitted_candidate_file_ids=omitted,
            retained_node_count=retained_node_count,
            metadata=metadata,
            reference_rows=reference_rows,
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
                logger.debug(
                    "Could not drop search relation %s", relation, exc_info=True
                )
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
        if request.coordinate_space == SEARCH_SPACE_CCF:
            _cleanup_region_nodes_view(
                conn,
                lut_registered=lut_registered,
                scope_registered=scope_registered,
                filter_view_name=filter_view_name,
                filter_relations=filter_relations,
                voxel_filter_view_name=voxel_filter_view_name,
                voxel_filter_relations=voxel_filter_relations,
            )
        else:
            if flatmap_scope_view_created:
                try:
                    conn.execute("DROP VIEW IF EXISTS search_flatmap_scoped_source")
                except Exception:
                    logger.debug(
                        "Could not drop flatmap Search scope view", exc_info=True
                    )
            if flatmap_scope_registered:
                try:
                    conn.unregister("search_flatmap_scope_ids")
                except Exception:
                    logger.debug(
                        "Could not unregister flatmap Search scope", exc_info=True
                    )
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


def export_search_results_csv(
    output_path: str | Path,
    result: VoxelSearchResult | SearchResultsDocument | pd.DataFrame,
) -> Path:
    """Atomically export a current or legacy Search result document."""
    path = Path(output_path)
    output_version = SEARCH_RESULTS_FORMAT_VERSION
    if isinstance(result, VoxelSearchResult):
        references = result.reference_rows.copy()
        hits = result.hits.copy()
        metadata = result.metadata
    elif isinstance(result, SearchResultsDocument):
        references = result.references.copy()
        hits = result.hits.copy()
        metadata = result.metadata
        output_version = int(result.format_version)
    else:
        references = pd.DataFrame(columns=["file_id", "neuron_id", "subject"])
        hits = result.copy()
        metadata = {}
        # Preserve the release-1 DataFrame API without producing a malformed
        # version-2 file whose reference rows cannot be recovered.
        output_version = SEARCH_RESULTS_LEGACY_FORMAT_VERSION
    if output_version not in {
        SEARCH_RESULTS_LEGACY_FORMAT_VERSION,
        SEARCH_RESULTS_FORMAT_VERSION,
    }:
        raise ValueError(f"Unsupported Search export format: {output_version}")
    required = {"file_id", "pearson_distance"}
    missing = required - set(hits.columns)
    if missing:
        raise ValueError(
            "Search results are missing required column(s): "
            + ", ".join(sorted(missing))
        )

    hits["file_id"] = hits["file_id"].astype(str)
    if "rank" not in hits:
        hits["rank"] = np.arange(1, len(hits) + 1, dtype=np.int64)
    hits["rank"] = pd.to_numeric(hits["rank"], errors="raise").astype(np.int64)
    if (hits["rank"] < 1).any() or hits["rank"].duplicated().any():
        raise ValueError("Search result ranks must be unique positive integers.")
    distances = pd.to_numeric(hits["pearson_distance"], errors="coerce")
    if not np.all(
        np.isfinite(distances.to_numpy(dtype=float))
        & distances.between(0.0, 2.0, inclusive="both").to_numpy()
    ):
        raise ValueError(
            "Search result pearson_distance values must be finite and between 0 and 2."
        )
    hits["pearson_distance"] = distances.astype(float)

    if output_version == SEARCH_RESULTS_LEGACY_FORMAT_VERSION:
        for column in ("neuron_id", "subject"):
            if column not in hits:
                hits[column] = ""
            hits[column] = hits[column].fillna("").astype(str)
        frame = hits[
            ["rank", "file_id", "neuron_id", "subject", "pearson_distance"]
        ].copy()
        frame.insert(0, "format_version", SEARCH_RESULTS_LEGACY_FORMAT_VERSION)
        return _atomic_write_search_csv(path, frame)

    if hits.empty:
        raise ValueError(
            "Version-2 Search exports require at least one search result row."
        )
    if not references.empty and "file_id" not in references:
        raise ValueError("Search references are missing required column: file_id")
    if "file_id" not in references:
        references["file_id"] = pd.Series(dtype="string")
    if references.empty:
        raise ValueError("Version-2 Search exports require at least one reference row.")
    references["file_id"] = references["file_id"].astype(str)
    for frame in (references, hits):
        for column in ("neuron_id", "subject"):
            if column not in frame:
                frame[column] = ""
            frame[column] = frame[column].fillna("").astype(str)

    duplicated = pd.concat([references["file_id"], hits["file_id"]], ignore_index=True)
    duplicate_ids = duplicated[duplicated.duplicated()].unique()
    if len(duplicate_ids):
        raise ValueError(
            "Search export contains duplicate file_id value(s): "
            + ", ".join(str(value) for value in duplicate_ids[:10])
        )

    references = references[["file_id", "neuron_id", "subject"]].copy()
    references.insert(0, "rank", np.zeros(len(references), dtype=np.int64))
    references["pearson_distance"] = np.nan
    references.insert(0, "row_role", SEARCH_ROW_REFERENCE)

    hits = hits[["rank", "file_id", "neuron_id", "subject", "pearson_distance"]].copy()
    hits.insert(0, "row_role", SEARCH_ROW_RESULT)
    frame = pd.concat([references, hits], ignore_index=True)
    frame.insert(0, "format_version", SEARCH_RESULTS_FORMAT_VERSION)
    frame["search_context_json"] = _canonical_context_json(metadata)
    frame = frame[list(SEARCH_RESULTS_COLUMNS)]

    return _atomic_write_search_csv(path, frame)


def _atomic_write_search_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Write one Search CSV atomically in the destination directory."""

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
) -> SearchResultsDocument:
    """Load and validate a legacy or current Search CSV document."""
    frame = pd.read_csv(
        input_path,
        dtype={
            "file_id": "string",
            "neuron_id": "string",
            "subject": "string",
            "row_role": "string",
            "search_context_json": "string",
        },
    )
    required = {"format_version", "file_id", "pearson_distance"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Search CSV is missing required column(s): " + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise ValueError("Search CSV contains no result rows.")

    versions = pd.to_numeric(frame["format_version"], errors="coerce")
    if versions.isna().any() or not np.all(np.equal(versions, np.floor(versions))):
        raise ValueError(
            "Search CSV must contain one supported integer format version."
        )
    unique_versions = set(versions.dropna().astype(int).tolist())
    if len(unique_versions) != 1:
        raise ValueError("Search CSV must contain one supported format version.")
    version = unique_versions.pop()
    if version not in {
        SEARCH_RESULTS_LEGACY_FORMAT_VERSION,
        SEARCH_RESULTS_FORMAT_VERSION,
    }:
        raise ValueError(
            "Unsupported search result format version; expected version 1 or "
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

    for column in ("neuron_id", "subject"):
        if column not in frame:
            frame[column] = ""
        else:
            frame[column] = frame[column].fillna("").astype(str)

    available = None
    if available_file_ids is None:
        available = set(frame["file_id"])
    else:
        available = {str(value) for value in available_file_ids}

    if version == SEARCH_RESULTS_LEGACY_FORMAT_VERSION:
        distances = pd.to_numeric(frame["pearson_distance"], errors="coerce")
        finite = np.isfinite(distances.to_numpy(dtype=float))
        in_range = distances.between(0.0, 2.0, inclusive="both").to_numpy()
        if not np.all(finite & in_range):
            raise ValueError(
                "Search CSV pearson_distance values must be finite and between 0 and 2."
            )
        frame["pearson_distance"] = distances.astype(float)
        hits = frame.sort_values(
            ["pearson_distance", "file_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        hits["rank"] = np.arange(1, len(hits) + 1, dtype=np.int64)
        hits["available"] = hits["file_id"].isin(available)
        return SearchResultsDocument(
            references=pd.DataFrame(
                columns=["rank", "file_id", "neuron_id", "subject", "available"]
            ),
            hits=hits[
                [
                    "rank",
                    "file_id",
                    "neuron_id",
                    "subject",
                    "pearson_distance",
                    "available",
                ]
            ],
            metadata={},
            format_version=version,
        )

    current_required = {"row_role", "rank", "search_context_json"}
    current_missing = current_required - set(frame.columns)
    if current_missing:
        raise ValueError(
            "Search CSV is missing version-2 column(s): "
            + ", ".join(sorted(current_missing))
        )
    roles = frame["row_role"].fillna("").astype(str)
    if not roles.isin({SEARCH_ROW_REFERENCE, SEARCH_ROW_RESULT}).all():
        raise ValueError("Search CSV contains an unknown row_role value.")
    ranks = pd.to_numeric(frame["rank"], errors="coerce")
    if ranks.isna().any() or not np.all(np.equal(ranks, np.floor(ranks))):
        raise ValueError("Search CSV ranks must be integers.")
    frame["rank"] = ranks.astype(np.int64)
    reference_mask = roles == SEARCH_ROW_REFERENCE
    result_mask = roles == SEARCH_ROW_RESULT
    if not reference_mask.any():
        raise ValueError(
            "Version-2 Search CSV must contain at least one reference row."
        )
    if not result_mask.any():
        raise ValueError(
            "Version-2 Search CSV must contain at least one search_result row."
        )
    if not (frame.loc[reference_mask, "rank"] == 0).all():
        raise ValueError("Every Search reference row must have rank 0.")
    result_ranks = frame.loc[result_mask, "rank"]
    expected_ranks = list(range(1, len(result_ranks) + 1))
    if sorted(result_ranks.tolist()) != expected_ranks:
        raise ValueError("Search result ranks must be unique and contiguous from 1.")

    raw_distances = frame["pearson_distance"]
    if raw_distances.loc[reference_mask].notna().any():
        raise ValueError("Search reference pearson_distance values must be empty.")
    result_distances = pd.to_numeric(raw_distances.loc[result_mask], errors="coerce")
    if not np.all(
        np.isfinite(result_distances.to_numpy(dtype=float))
        & result_distances.between(0.0, 2.0, inclusive="both").to_numpy()
    ):
        raise ValueError(
            "Search result pearson_distance values must be finite and between 0 and 2."
        )
    frame.loc[result_mask, "pearson_distance"] = result_distances.astype(float)

    context_strings = frame["search_context_json"].fillna("").astype(str)
    parsed_contexts: list[dict[str, object]] = []
    canonical_contexts: set[str] = set()
    for raw_context in context_strings:
        if not raw_context.strip():
            raise ValueError(
                "Every version-2 Search CSV row must contain search context JSON."
            )
        try:
            parsed = json.loads(raw_context)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Search CSV contains invalid context JSON: {error}"
            ) from error
        if not isinstance(parsed, dict):
            raise TypeError("Search CSV context JSON must contain an object.")
        parsed_contexts.append(parsed)
        canonical_contexts.add(
            json.dumps(
                parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
    if len(canonical_contexts) > 1:
        raise ValueError("Search CSV rows contain conflicting search context JSON.")
    metadata = parsed_contexts[0] if parsed_contexts else {}

    references = frame.loc[
        reference_mask,
        ["rank", "file_id", "neuron_id", "subject"],
    ].copy()
    references["available"] = references["file_id"].isin(available)
    hits = frame.loc[
        result_mask,
        ["rank", "file_id", "neuron_id", "subject", "pearson_distance"],
    ].copy()
    hits["pearson_distance"] = hits["pearson_distance"].astype(float)
    hits = hits.sort_values("rank", kind="mergesort").reset_index(drop=True)
    hits["available"] = hits["file_id"].isin(available)
    return SearchResultsDocument(
        references=references.reset_index(drop=True),
        hits=hits,
        metadata=metadata,
        format_version=version,
    )


def build_filtered_search_heatmap_volume(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str | Path,
    *,
    atlas_shape: tuple[int, int, int],
    resolution_um: float,
    file_ids: tuple[str, ...],
    voxel_node_filter: VoxelNodeFilter | None = None,
    prepared_region_filter: PreparedClusterRegionFilter | None = None,
    prepared_voxel_filter: PreparedVoxelNodeFilter | None = None,
) -> np.ndarray:
    """Build a dense count volume from the exact filtered Search node rows."""
    from .correlation import _cleanup_region_nodes_view, _prepare_region_nodes_view

    shape = tuple(int(value) for value in atlas_shape)
    volume = np.zeros(shape, dtype=np.float32)
    prepared = _prepare_region_nodes_view(
        conn,
        str(parquet_path),
        voxel_id_map=None,
        resolution=float(resolution_um),
        file_ids=file_ids,
        prepared_region_filter=prepared_region_filter,
        voxel_node_filter=voxel_node_filter,
        prepared_voxel_filter=prepared_voxel_filter,
    )
    (
        lut_registered,
        scope_registered,
        filter_view_name,
        filter_relations,
        voxel_filter_view_name,
        voxel_filter_relations,
    ) = prepared
    try:
        counts = conn.execute(
            """
            SELECT zi, yi, xi, COUNT(*)::FLOAT AS node_count
            FROM base_nodes
            WHERE zi >= 0 AND zi < ?
              AND yi >= 0 AND yi < ?
              AND xi >= 0 AND xi < ?
            GROUP BY zi, yi, xi
            """,
            [shape[0], shape[1], shape[2]],
        ).fetchdf()
        if not counts.empty:
            zi = counts["zi"].to_numpy(dtype=np.intp)
            yi = counts["yi"].to_numpy(dtype=np.intp)
            xi = counts["xi"].to_numpy(dtype=np.intp)
            volume[zi, yi, xi] = counts["node_count"].to_numpy(dtype=np.float32)
        return volume
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
