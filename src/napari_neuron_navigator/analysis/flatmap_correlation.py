"""Flatmap-space voxel correlation for clustered neurons."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..flatmap_heatmap import FlatmapLookupStats
from .clustering import ClusterResult, compute_clustermap_data

logger = logging.getLogger(__name__)

#: Depth axis weight that makes one full cortical thickness count for exactly
#: one hemisphere *height* of tangential flat map distance.  Height, not width:
#: both flat map axes are divided by the ``y`` span, which is the axis that
#: spans one hemisphere.
DEFAULT_FLATMAP_DEPTH_SCALE = 1.0

#: DBSCAN neighbourhood radius for flat map soma clustering, in normalized flat
#: map units rather than microns.  One unit is one hemisphere height, so this is
#: 5% of that in any direction -- the radius is circular because both flat map
#: axes share one divisor.  The CCF soma clustering keeps its own micron-valued
#: default; the two spaces are not interchangeable.
DEFAULT_FLATMAP_SOMA_DBSCAN_EPS = 0.05


@dataclass(frozen=True)
class FlatmapVoxelCorrelationSource:
    """Binned flatmap heatmap data used as a clustering source."""

    projected_nodes: pd.DataFrame
    volume_shape: tuple[int, int, int]
    input_file_ids: tuple[str, ...]
    y_bins: int
    x_bins: int | None
    depth_bin_um: float
    include_depth_minus_one: bool
    flatmap_style: str | None = None
    coordinate_mode: str | None = None
    flatmap_path: str | None = None
    depth_path: str | None = None
    invalid_zero_sentinel: bool = False
    invalid_negative_one_sentinel: bool = True
    lookup_stats: FlatmapLookupStats | None = None
    mirror_depth_fallback: bool = True
    mirror_coord_axis: int = 2
    cache_dir: str | None = None
    cache_profile_id: str | None = None
    cache_style: str | None = None


@dataclass(frozen=True)
class FlatmapCountMatrix:
    """Dense count matrix and provenance for flatmap voxel correlation."""

    neuron_ids: list[str]
    voxel_ids: np.ndarray
    count_matrix: np.ndarray
    rendered_node_count: int
    unassigned_neuron_ids: list[str]


def _unique_strings_in_order(values) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return tuple(unique)


def _validate_region_mask(
    region_mask: np.ndarray | None,
    volume_shape: tuple[int, int, int],
) -> np.ndarray | None:
    if region_mask is None:
        return None
    mask = np.asarray(region_mask, dtype=bool)
    if mask.shape != tuple(volume_shape):
        raise ValueError(
            "region_mask shape must match the flatmap heatmap volume; "
            f"got {mask.shape} and {tuple(volume_shape)}."
        )
    return mask


def _rendered_binned_nodes(
    projected_nodes: pd.DataFrame,
    volume_shape: tuple[int, int, int],
    *,
    region_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    required = ("file_id", "render_valid", "depth_bin", "y_flat_bin", "x_flat_bin")
    missing = [column for column in required if column not in projected_nodes.columns]
    if missing:
        raise ValueError(f"Projected nodes are missing render column(s): {missing}")
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must be 3D; got {volume_shape}.")

    depth_size, y_size, x_size = (int(size) for size in volume_shape)
    mask = _validate_region_mask(region_mask, volume_shape)
    table = projected_nodes.copy()

    render_valid = table["render_valid"].fillna(False).astype(bool).to_numpy()
    depth_bins = pd.to_numeric(table["depth_bin"], errors="coerce").to_numpy(
        dtype=float
    )
    y_bins = pd.to_numeric(table["y_flat_bin"], errors="coerce").to_numpy(dtype=float)
    x_bins = pd.to_numeric(table["x_flat_bin"], errors="coerce").to_numpy(dtype=float)
    finite_bins = np.isfinite(depth_bins) & np.isfinite(y_bins) & np.isfinite(x_bins)
    in_bounds = (
        finite_bins
        & (depth_bins >= 0)
        & (depth_bins < depth_size)
        & (y_bins >= 0)
        & (y_bins < y_size)
        & (x_bins >= 0)
        & (x_bins < x_size)
    )
    valid = render_valid & in_bounds
    if mask is not None and valid.any():
        d = depth_bins[valid].astype(np.int64)
        y = y_bins[valid].astype(np.int64)
        x = x_bins[valid].astype(np.int64)
        region_valid = np.zeros(len(table), dtype=bool)
        region_valid[np.flatnonzero(valid)] = mask[d, y, x]
        valid &= region_valid

    rendered = table.loc[valid].copy()
    if rendered.empty:
        return rendered

    for column in ("depth_bin", "y_flat_bin", "x_flat_bin"):
        rendered.loc[:, column] = pd.to_numeric(
            rendered[column], errors="coerce"
        ).astype(np.int64)
    rendered.loc[:, "_flatmap_file_id"] = rendered["file_id"].map(str)
    rendered.loc[:, "_flatmap_linear_voxel"] = (
        rendered["depth_bin"].to_numpy(dtype=np.int64) * y_size * x_size
        + rendered["y_flat_bin"].to_numpy(dtype=np.int64) * x_size
        + rendered["x_flat_bin"].to_numpy(dtype=np.int64)
    )
    return rendered


def build_flatmap_count_matrix(
    source: FlatmapVoxelCorrelationSource,
    *,
    region_mask: np.ndarray | None = None,
) -> FlatmapCountMatrix:
    """Build per-neuron node-count vectors in flatmap heatmap voxel space."""
    rendered = _rendered_binned_nodes(
        source.projected_nodes,
        source.volume_shape,
        region_mask=region_mask,
    )
    input_file_ids = tuple(source.input_file_ids) or _unique_strings_in_order(
        source.projected_nodes["file_id"].tolist()
        if "file_id" in source.projected_nodes.columns
        else []
    )

    if rendered.empty:
        return FlatmapCountMatrix(
            neuron_ids=[],
            voxel_ids=np.empty(0, dtype=np.int64),
            count_matrix=np.empty((0, 0), dtype=np.float32),
            rendered_node_count=0,
            unassigned_neuron_ids=list(input_file_ids),
        )

    rendered_ids = set(rendered["_flatmap_file_id"].tolist())
    neuron_ids = [file_id for file_id in input_file_ids if file_id in rendered_ids]
    if not neuron_ids:
        neuron_ids = list(
            _unique_strings_in_order(rendered["_flatmap_file_id"].tolist())
        )

    unassigned = [file_id for file_id in input_file_ids if file_id not in rendered_ids]
    counts = (
        rendered.groupby(["_flatmap_file_id", "_flatmap_linear_voxel"], sort=False)
        .size()
        .reset_index(name="count")
    )
    voxel_ids = np.sort(counts["_flatmap_linear_voxel"].unique().astype(np.int64))
    row_index = {file_id: index for index, file_id in enumerate(neuron_ids)}
    col_index = {int(voxel_id): index for index, voxel_id in enumerate(voxel_ids)}
    matrix = np.zeros((len(neuron_ids), len(voxel_ids)), dtype=np.float32)
    for file_id, voxel_id, count in counts.itertuples(index=False):
        row = row_index.get(str(file_id))
        if row is None:
            continue
        matrix[row, col_index[int(voxel_id)]] = float(count)

    return FlatmapCountMatrix(
        neuron_ids=neuron_ids,
        voxel_ids=voxel_ids,
        count_matrix=matrix,
        rendered_node_count=int(len(rendered)),
        unassigned_neuron_ids=unassigned,
    )


def pearson_correlation_from_counts(count_matrix: np.ndarray) -> np.ndarray:
    """Return a stable Pearson correlation matrix from count vectors."""
    counts = np.asarray(count_matrix, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"count_matrix must be 2D; got shape {counts.shape}.")
    n_neurons = counts.shape[0]
    if n_neurons == 0:
        return np.empty((0, 0), dtype=np.float32)

    centered = counts - counts.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=1))
    denom = np.outer(norm, norm)
    corr = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    valid = denom > 0.0
    corr[valid] = (centered @ centered.T)[valid] / denom[valid]
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def query_flatmap_soma_coordinates_and_count(
    parquet_path: str,
    *,
    style: str,
    file_ids: list[str] | None = None,
    prepared_region_filter=None,
    excluded_file_ids: set[str] | None = None,
) -> tuple[list[str], np.ndarray, int]:
    """Return per-neuron flatmap soma coordinates and contributing row count.

    Averages the flatmap ``x``/``y`` and ``depth_um`` of each neuron's soma
    nodes (``type == 1``) that have valid flatmap and depth projections, using
    the version-3 column family selected by ``style``.  Returns ``(file_ids,
    coords)`` where ``coords`` has one ``(x_flat, y_flat, depth_um)`` row per
    file, ordered to match ``file_ids``.
    """
    import duckdb

    from ..flatmap_heatmap import (
        _combine_where,
        _duckdb_column_names,
        _duckdb_source_path,
        _file_id_filter,
        _flatmap_sql_expressions,
        _sql_identifier,
        _style_suffix,
    )

    suffix = _style_suffix(style)
    file_filter = _file_id_filter(file_ids)
    if file_filter is None:
        return [], np.empty((0, 3), dtype=float), 0
    file_filter_sql, params = file_filter

    conn = duckdb.connect()
    try:
        source_sql = f"read_parquet('{_duckdb_source_path(parquet_path)}')"
        column_names = _duckdb_column_names(conn, source_sql)
        # Only the flatmap_valid / depth_valid expressions are used here; the
        # bin math is irrelevant for soma coordinates, so pass placeholder
        # bounds and bins that keep _flatmap_sql_expressions valid.
        expressions = _flatmap_sql_expressions(
            column_names,
            suffix=suffix,
            x_lower=0.0,
            x_upper=1.0,
            y_lower=0.0,
            y_upper=1.0,
            depth_lower=0.0,
            depth_bin_um=1.0,
            y_bins=1,
            x_bins=1,
            valid_depth_bins=1,
            sentinel_offset=0,
            include_depth_minus_one=False,
        )
        x_ref = _sql_identifier(f"x_flat_{suffix}")
        y_ref = _sql_identifier(f"y_flat_{suffix}")
        projected_soma = (
            f"({expressions['flatmap_valid']}) "
            f"AND ({expressions['depth_valid']})"
        )
        where_sql = _combine_where("type = 1", file_filter_sql)
        query = f"""
            SELECT
                file_id,
                AVG(CASE WHEN {projected_soma} THEN {x_ref} END) AS x_flat,
                AVG(CASE WHEN {projected_soma} THEN {y_ref} END) AS y_flat,
                AVG(CASE WHEN {projected_soma} THEN depth_um END) AS depth_um,
                AVG(x) AS x,
                AVG(y) AS y,
                AVG(z) AS z,
                COUNT(CASE WHEN {projected_soma} THEN 1 END)::BIGINT
                    AS soma_node_count
            FROM {source_sql}
            WHERE {where_sql}
            GROUP BY file_id
            HAVING COUNT(CASE WHEN {projected_soma} THEN 1 END) > 0
            ORDER BY file_id
        """
        soma_df = (
            conn.execute(query, params).fetchdf()
            if params
            else conn.execute(query).fetchdf()
        )
    finally:
        conn.close()

    if soma_df.empty:
        return [], np.empty((0, 3), dtype=float), 0

    if prepared_region_filter is not None:
        from .region_filter import excluded_soma_file_ids, filter_soma_frame

        soma_df = filter_soma_frame(soma_df, prepared_region_filter)
        if excluded_file_ids is None:
            excluded_file_ids = excluded_soma_file_ids(
                parquet_path,
                prepared_region_filter,
                file_ids=file_ids,
            )
        if excluded_file_ids:
            soma_df = soma_df[
                ~soma_df["file_id"].astype(str).isin(excluded_file_ids)
            ].reset_index(drop=True)
        if soma_df.empty:
            return [], np.empty((0, 3), dtype=float), 0

    coords = soma_df[["x_flat", "y_flat", "depth_um"]].to_numpy(dtype=float)
    finite = np.all(np.isfinite(coords), axis=1)
    coords = coords[finite]
    ids = soma_df["file_id"].astype(str).to_numpy()[finite].tolist()
    node_count = int(soma_df.loc[finite, "soma_node_count"].sum())
    return ids, coords, node_count


def query_flatmap_soma_coordinates(
    parquet_path: str,
    *,
    style: str,
    file_ids: list[str] | None = None,
    prepared_region_filter=None,
    excluded_file_ids: set[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return per-neuron soma coordinates in flatmap + depth space."""
    ids, coords, _node_count = query_flatmap_soma_coordinates_and_count(
        parquet_path,
        style=style,
        file_ids=file_ids,
        prepared_region_filter=prepared_region_filter,
        excluded_file_ids=excluded_file_ids,
    )
    return ids, coords


@dataclass(frozen=True)
class FlatmapDepthNormalization:
    """Divisors that put flatmap + depth coordinates on a common scale.

    The raw Parquet columns mix units: ``x_flat``/``y_flat`` are normalized
    floats spanning the *bilateral* flat map while ``depth_um`` is raw microns
    spanning the full cortical thickness.  Clustering those together with an
    unweighted Euclidean metric lets depth contribute >99.99% of the variance,
    which reduces the result to a 1-D laminar partition.

    **The two flat map axes share one divisor**, so flat map space stays
    isotropic: equal distances in the metric mean equal distances on the flat
    map, whichever direction they run.  This is the same policy the voxel grid
    uses -- :func:`napari_neuron_navigator.flatmap_heatmap.resolve_flatmap_bin_counts`
    derives the ``x`` bin count from the aspect ratio so a bin is as wide as it
    is tall, which is exactly one shared unit per axis.  Both take ``y`` as the
    reference axis, since ``y`` spans one hemisphere while ``x`` spans two.

    Giving each axis its own span instead would force every style's hemisphere
    into a square bounding box, stretching whichever axis is shorter.  For
    ``both_shaped`` that put 4.2% anisotropy into the metric; ``both_square`` is
    an exact 2:1 map, so it was and remains unaffected.  Since the projection's
    true surface geometry is unknowable (flat map x/y have no reliable micron
    conversion), the conservative choice is to add no distortion at all.

    ``depth_um`` keeps its own divisor because it is a genuinely different
    quantity in different units; ``depth_scale`` weights it against flat map
    distance.  No offset is subtracted: Euclidean distance is translation
    invariant, and leaving the origin alone keeps the two hemispheres adjacent
    rather than superimposed.
    """

    style: str
    flatmap_divisor: float
    depth_divisor_um: float
    depth_scale: float = DEFAULT_FLATMAP_DEPTH_SCALE
    include_depth: bool = True
    bounds_source: str = "canonical"

    @property
    def axis_count(self) -> int:
        """Return the dimensionality of the normalized feature space."""
        return 3 if self.include_depth else 2

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping for export metadata."""
        return {
            "style": self.style,
            # One divisor, not one per axis: recording two numbers that must
            # always be equal is how the old anisotropy hid in plain sight.
            "flatmap_divisor": float(self.flatmap_divisor),
            "depth_divisor_um": float(self.depth_divisor_um),
            "depth_scale": float(self.depth_scale),
            "include_depth": bool(self.include_depth),
            "bounds_source": self.bounds_source,
            "axis_count": self.axis_count,
        }


def resolve_flatmap_depth_normalization(
    parquet_path: str,
    *,
    style: str,
    depth_scale: float = DEFAULT_FLATMAP_DEPTH_SCALE,
    include_depth: bool = True,
) -> FlatmapDepthNormalization:
    """Return the shared flat map divisor and depth divisor for one style.

    Both flat map axes are divided by the ``y`` span, which keeps flat map space
    isotropic and matches how the voxel grid derives its bin counts.  ``y`` is
    the reference axis because it spans one hemisphere while ``x`` spans two, so
    the ``x`` extent is not a per-hemisphere quantity and must not set the unit.

    Because ``x`` no longer sets a divisor, the bilateral halving this function
    used to apply to canonical bounds -- and the matching exception for observed
    bounds -- are both gone.  ``y`` spans one hemisphere either way, so there is
    nothing left to special-case.

    Prefers the canonical bounds recorded in version-3 Parquet metadata so the
    metric does not depend on which neurons happen to be in scope.  Falls back
    to data-derived bounds for Parquets that never recorded canonical bounds,
    which is reproducible only within a fixed dataset — the fallback is recorded
    in ``bounds_source`` so exports stay honest about it.
    """
    from ..flatmap_heatmap import _style_suffix

    if depth_scale < 0.0:
        raise ValueError(f"depth_scale must be non-negative; got {depth_scale!r}.")

    grid = _flatmap_canonical_grid_spec(parquet_path, style)
    if grid is not None:
        bounds_source = "canonical"
        y_bounds = grid.y_bounds
        depth_bounds = grid.depth_bounds_um
    else:
        bounds_source = "observed"
        logger.warning(
            "Parquet %s records no canonical flat map bounds for style %r; "
            "normalizing soma coordinates against observed data bounds instead. "
            "The resulting metric is comparable only within this dataset.",
            parquet_path,
            style,
        )
        _x_bounds, y_bounds, depth_bounds = _resolve_flatmap_render_bounds(
            parquet_path,
            style,
            _style_suffix(style),
        )
    flatmap_divisor = float(y_bounds[1] - y_bounds[0])
    depth_divisor = float(depth_bounds[1] - depth_bounds[0])

    for name, value in (
        ("y", flatmap_divisor),
        ("depth_um", depth_divisor),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Flat map style {style!r} has a non-positive {name} span "
                f"({value!r}); cannot normalize soma coordinates."
            )

    return FlatmapDepthNormalization(
        style=style,
        flatmap_divisor=flatmap_divisor,
        depth_divisor_um=depth_divisor,
        depth_scale=float(depth_scale),
        include_depth=bool(include_depth),
        bounds_source=bounds_source,
    )


def _flatmap_canonical_grid_spec(parquet_path: str, style: str):
    """Return the canonical v3 grid spec for a style, or ``None`` if absent."""
    from ..flatmap_parquet import read_flatmap_parquet_transform_info

    try:
        info = read_flatmap_parquet_transform_info(parquet_path)
    except Exception:  # noqa: BLE001 - fall back to observed bounds
        return None
    return info.grid_spec(style)


def normalize_flatmap_soma_coordinates(
    coords: np.ndarray,
    normalization: FlatmapDepthNormalization,
) -> np.ndarray:
    """Scale raw ``(x_flat, y_flat, depth_um)`` rows onto a common scale.

    Both flat map axes are divided by the same number, so the flat map plane is
    scaled without being distorted; only depth is weighted separately.

    Returns an ``(N, 3)`` array when ``normalization.include_depth`` is set and
    an ``(N, 2)`` array otherwise, so excluding depth genuinely clusters on flat
    map position rather than merely down-weighting the depth axis to zero.
    """
    raw = np.asarray(coords, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(
            "coords must be an (N, 3) array of (x_flat, y_flat, depth_um) rows; "
            f"got shape {raw.shape}."
        )

    scaled_xy = raw[:, :2] / normalization.flatmap_divisor
    if not normalization.include_depth:
        return scaled_xy

    scaled_depth = (
        raw[:, 2] / normalization.depth_divisor_um
    ) * normalization.depth_scale
    return np.column_stack([scaled_xy, scaled_depth])


@dataclass(frozen=True)
class FlatmapParquetCorrelationProvenance:
    """Binning provenance for a parquet-driven flatmap correlation run."""

    style: str
    y_bins: int
    x_bins: int
    depth_bin_um: float
    include_depth_minus_one: bool
    volume_shape: tuple[int, ...]
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    depth_range_um: tuple[float, float]
    collapse_depth: bool = False


def _resolve_flatmap_render_bounds(
    parquet_path: str,
    style: str,
    suffix: str,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return canonical (x, y, depth) render bounds for one flatmap style.

    Prefers the canonical bounds recorded in version-3 Parquet metadata so
    partial queries stay aligned to the same grid.  Falls back to data-derived
    min/max for legacy Parquets that never recorded canonical bounds.
    """
    import duckdb

    from ..flatmap_heatmap import _duckdb_source_path
    from ..flatmap_parquet import read_flatmap_parquet_transform_info

    info = read_flatmap_parquet_transform_info(parquet_path)
    grid = info.grid_spec(style)
    if grid is not None:
        return grid.x_bounds, grid.y_bounds, grid.depth_bounds_um

    x_column = f"x_flat_{suffix}"
    y_column = f"y_flat_{suffix}"
    conn = duckdb.connect()
    try:
        source_sql = f"read_parquet('{_duckdb_source_path(parquet_path)}')"
        row = conn.execute(
            f"""
            SELECT
                MIN("{x_column}") FILTER (WHERE isfinite("{x_column}")),
                MAX("{x_column}") FILTER (WHERE isfinite("{x_column}")),
                MIN("{y_column}") FILTER (WHERE isfinite("{y_column}")),
                MAX("{y_column}") FILTER (WHERE isfinite("{y_column}")),
                MIN(depth_um) FILTER (WHERE isfinite(depth_um) AND depth_um >= 0.0),
                MAX(depth_um) FILTER (WHERE isfinite(depth_um) AND depth_um >= 0.0)
            FROM {source_sql}
            """
        ).fetchone()
    finally:
        conn.close()
    if row is None or any(value is None for value in row):
        raise ValueError(
            "Parquet does not contain valid flatmap/depth coordinates for style "
            f"{style!r}; cannot resolve render bounds."
        )
    return (
        (float(row[0]), float(row[1])),
        (float(row[2]), float(row[3])),
        (float(row[4]), float(row[5])),
    )


def build_flatmap_count_matrix_from_bin_counts(
    counts: pd.DataFrame,
    volume_shape: tuple[int, ...],
    *,
    input_file_ids: tuple[str, ...],
) -> FlatmapCountMatrix:
    """Build per-neuron node-count vectors from DuckDB per-file bin counts.

    ``counts`` must have ``file_id``, ``y_bin``, ``x_bin`` and ``node_count``
    columns, exactly as produced by
    :func:`napari_neuron_navigator.flatmap_heatmap._query_flatmap_bin_counts` with
    ``include_file_id=True``.

    A 2-length ``volume_shape`` selects a depth-collapsed matrix, whose counts
    carry no ``depth_bin`` column: every node at one flat map position lands in
    the same voxel regardless of its depth.  As in
    :func:`napari_neuron_navigator.flatmap_heatmap._rendered_binned_nodes`, the rank of
    the volume decides that -- never the presence of a column -- so a stray
    ``depth_bin`` cannot silently change the grouping.
    """
    if len(volume_shape) not in (2, 3):
        raise ValueError(f"volume_shape must be 2D or 3D; got {volume_shape}.")
    has_depth_axis = len(volume_shape) == 3
    # Only the trailing y/x extents enter the linear voxel id; the depth extent
    # is implied by whichever depth bins the counts actually contain.
    y_size, x_size = (int(size) for size in volume_shape[-2:])

    required = ("file_id", "y_bin", "x_bin", "node_count")
    if has_depth_axis:
        required = (*required, "depth_bin")
    missing = [column for column in required if column not in counts.columns]
    if missing:
        raise ValueError(f"Bin counts are missing required column(s): {missing}")

    ordered_inputs = tuple(str(file_id) for file_id in input_file_ids)
    if counts.empty:
        return FlatmapCountMatrix(
            neuron_ids=[],
            voxel_ids=np.empty(0, dtype=np.int64),
            count_matrix=np.empty((0, 0), dtype=np.float32),
            rendered_node_count=0,
            unassigned_neuron_ids=list(ordered_inputs),
        )

    file_ids = counts["file_id"].map(str).to_numpy()
    linear = counts["y_bin"].to_numpy(dtype=np.int64) * x_size + counts[
        "x_bin"
    ].to_numpy(dtype=np.int64)
    if has_depth_axis:
        linear = counts["depth_bin"].to_numpy(dtype=np.int64) * y_size * x_size + linear
    node_counts = counts["node_count"].to_numpy(dtype=np.float64)

    rendered_ids = set(file_ids.tolist())
    neuron_ids = [file_id for file_id in ordered_inputs if file_id in rendered_ids]
    if not neuron_ids:
        neuron_ids = list(_unique_strings_in_order(file_ids.tolist()))
    unassigned = [file_id for file_id in ordered_inputs if file_id not in rendered_ids]

    voxel_ids = np.unique(linear.astype(np.int64))
    row_index = {file_id: index for index, file_id in enumerate(neuron_ids)}
    col_index = {int(voxel_id): index for index, voxel_id in enumerate(voxel_ids)}
    matrix = np.zeros((len(neuron_ids), len(voxel_ids)), dtype=np.float32)
    for file_id, voxel_id, count in zip(file_ids, linear, node_counts):
        row = row_index.get(str(file_id))
        if row is None:
            continue
        matrix[row, col_index[int(voxel_id)]] += float(count)

    return FlatmapCountMatrix(
        neuron_ids=neuron_ids,
        voxel_ids=voxel_ids,
        count_matrix=matrix,
        rendered_node_count=int(node_counts.sum()),
        unassigned_neuron_ids=unassigned,
    )


def count_flatmap_voxel_correlation_nodes(
    parquet_path: str,
    *,
    style: str,
    y_bins: int,
    x_bins: int | None = None,
    depth_bin_um: float,
    include_depth_minus_one: bool = True,
    file_ids: list[str] | None = None,
    collapse_depth: bool = False,
    prepared_region_filter=None,
) -> int:
    """Return the exact rendered-node count for flatmap voxel correlation.

    ``collapse_depth`` only shrinks the voxel grid that the size guard checks;
    the node count itself is unchanged, because collapsing removes the depth
    axis without changing which nodes are counted.
    """
    import duckdb

    from ..flatmap_heatmap import (
        MAX_FLATMAP_HEATMAP_VOXELS,
        _combine_where,
        _depth_bin_count,
        _duckdb_column_names,
        _duckdb_source_path,
        _file_id_filter,
        _flatmap_sql_expressions,
        _nondegenerate_bounds,
        _resolve_axis_bin_counts,
        _style_suffix,
    )

    suffix = _style_suffix(style)
    x_bounds, y_bounds, depth_range = _resolve_flatmap_render_bounds(
        parquet_path, style, suffix
    )
    x_lower, x_upper = _nondegenerate_bounds(*x_bounds)
    y_lower, y_upper = _nondegenerate_bounds(*y_bounds)
    depth_lower, depth_upper = _nondegenerate_bounds(*depth_range)
    y_bins, x_bins, depth_bin_um = _resolve_axis_bin_counts(
        x_bounds=(x_lower, x_upper),
        y_bounds=(y_lower, y_upper),
        y_bins=y_bins,
        x_bins=x_bins,
        depth_bin_um=depth_bin_um,
    )
    valid_depth_bins = _depth_bin_count((depth_lower, depth_upper), depth_bin_um)
    sentinel_offset = 1 if include_depth_minus_one else 0
    total_depth_bins = valid_depth_bins + sentinel_offset
    grid_shape: tuple[int, ...] = (
        (y_bins, x_bins) if collapse_depth else (total_depth_bins, y_bins, x_bins)
    )
    if int(np.prod(grid_shape)) > MAX_FLATMAP_HEATMAP_VOXELS:
        shape_text = "x".join(str(int(size)) for size in grid_shape)
        raise ValueError(
            f"Flatmap voxel grid is too large: {shape_text} voxels. "
            "Use fewer Y bins or a larger depth bin."
        )

    file_filter = _file_id_filter(file_ids)
    if file_filter is None:
        return 0
    file_filter_sql, params = file_filter

    conn = duckdb.connect()
    filter_view_name = "cluster_region_filtered_flatmap_count"
    filter_relations: tuple[str, ...] = ()
    try:
        source_sql = f"read_parquet('{_duckdb_source_path(parquet_path)}')"
        if prepared_region_filter is not None:
            from .region_filter import register_filtered_source_view

            source_sql, filter_relations = register_filtered_source_view(
                conn,
                source_sql,
                prepared_region_filter,
                view_name=filter_view_name,
            )
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
            include_depth_minus_one=include_depth_minus_one,
        )
        where_sql = _combine_where(expressions["render_where"], file_filter_sql)
        query = f"SELECT COUNT(*) FROM {source_sql} WHERE {where_sql}"
        row = (
            conn.execute(query, params).fetchone()
            if params
            else conn.execute(query).fetchone()
        )
    finally:
        if filter_relations:
            from .region_filter import cleanup_filtered_source_view

            cleanup_filtered_source_view(
                conn,
                filter_view_name,
                filter_relations,
            )
        conn.close()
    return int(row[0] or 0) if row is not None else 0


def compute_flatmap_voxel_correlation_from_parquet(
    parquet_path: str,
    *,
    style: str,
    y_bins: int,
    x_bins: int | None = None,
    depth_bin_um: float,
    include_depth_minus_one: bool = True,
    method: str = "average",
    n_clusters: int = 5,
    file_ids: list[str] | None = None,
    collapse_depth: bool = False,
    prepared_region_filter=None,
) -> tuple[ClusterResult, FlatmapCountMatrix, FlatmapParquetCorrelationProvenance]:
    """Cluster neurons by flatmap-space voxel correlation straight from Parquet.

    Reads only the precomputed flatmap/depth columns via DuckDB and bins them
    with a ``GROUP BY`` (mirroring the heatmap fast path), so no rendered
    heatmap is required.  Requires the version-3 bilateral ``*_shaped`` /
    ``*_square`` coordinate columns.

    ``collapse_depth`` drops the depth *axis* from the voxel grid, so two
    neurons correlate on flat map footprint regardless of which layer their
    nodes occupy.  Depth still decides which nodes are counted, so
    ``include_depth_minus_one`` keeps its meaning and the rendered node count is
    unchanged.  This is not reachable by widening ``depth_bin_um``: collapsing
    changes voxel identity rather than voxel resolution, and a single wide depth
    bin still separates the depth ``-1`` sentinel plane from the valid one.
    """
    import duckdb

    from ..flatmap_heatmap import (
        MAX_FLATMAP_HEATMAP_VOXELS,
        _combine_where,
        _depth_bin_count,
        _duckdb_column_names,
        _duckdb_source_path,
        _file_id_filter,
        _flatmap_sql_expressions,
        _nondegenerate_bounds,
        _query_flatmap_bin_counts,
        _resolve_axis_bin_counts,
        _style_suffix,
    )

    suffix = _style_suffix(style)
    x_bounds, y_bounds, depth_range = _resolve_flatmap_render_bounds(
        parquet_path, style, suffix
    )
    x_lower, x_upper = _nondegenerate_bounds(x_bounds[0], x_bounds[1])
    y_lower, y_upper = _nondegenerate_bounds(y_bounds[0], y_bounds[1])
    depth_lower, depth_upper = _nondegenerate_bounds(depth_range[0], depth_range[1])
    y_bins, x_bins, depth_bin_um = _resolve_axis_bin_counts(
        x_bounds=(x_lower, x_upper),
        y_bounds=(y_lower, y_upper),
        y_bins=y_bins,
        x_bins=x_bins,
        depth_bin_um=depth_bin_um,
    )

    valid_depth_bins = _depth_bin_count((depth_lower, depth_upper), depth_bin_um)
    sentinel_offset = 1 if include_depth_minus_one else 0
    total_depth_bins = valid_depth_bins + sentinel_offset
    volume_shape: tuple[int, ...] = (
        (y_bins, x_bins) if collapse_depth else (total_depth_bins, y_bins, x_bins)
    )

    voxel_count = int(np.prod(volume_shape))
    if voxel_count > MAX_FLATMAP_HEATMAP_VOXELS:
        shape_text = "x".join(str(int(size)) for size in volume_shape)
        raise ValueError(
            f"Flatmap voxel grid is too large: {shape_text} voxels. "
            "Use fewer Y bins or a larger depth bin."
        )

    file_filter = _file_id_filter(file_ids)
    if file_filter is None:
        raise ValueError(
            "Flatmap voxel correlation requires at least one neuron; the "
            "selected file set is empty."
        )
    file_filter_sql, file_params = file_filter

    conn = duckdb.connect()
    filter_view_name = "cluster_region_filtered_flatmap_source"
    filter_relations: tuple[str, ...] = ()
    all_input_file_ids: tuple[str, ...] | None = None
    try:
        raw_source_sql = f"read_parquet('{_duckdb_source_path(parquet_path)}')"
        if file_ids is None:
            all_input_file_ids = tuple(
                str(row[0])
                for row in conn.execute(
                    f"SELECT DISTINCT file_id FROM {raw_source_sql} ORDER BY file_id"
                ).fetchall()
            )
        source_sql = raw_source_sql
        if prepared_region_filter is not None:
            from .region_filter import register_filtered_source_view

            source_sql, filter_relations = register_filtered_source_view(
                conn,
                source_sql,
                prepared_region_filter,
                view_name=filter_view_name,
            )
        column_names = _duckdb_column_names(conn, source_sql)
        expressions = _flatmap_sql_expressions(
            column_names,
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
            include_depth_minus_one=include_depth_minus_one,
        )
        where_sql = _combine_where(expressions["render_where"], file_filter_sql)
        counts = _query_flatmap_bin_counts(
            conn,
            source_sql,
            expressions,
            where_sql,
            file_params,
            include_file_id=True,
            include_depth_bin=not collapse_depth,
        )
    finally:
        if filter_relations:
            from .region_filter import cleanup_filtered_source_view

            cleanup_filtered_source_view(
                conn,
                filter_view_name,
                filter_relations,
            )
        conn.close()

    if file_ids is None:
        input_file_ids = all_input_file_ids or _unique_strings_in_order(
            counts["file_id"].tolist()
        )
    else:
        input_file_ids = tuple(str(file_id) for file_id in file_ids)

    count_data = build_flatmap_count_matrix_from_bin_counts(
        counts,
        volume_shape,
        input_file_ids=input_file_ids,
    )
    if len(count_data.neuron_ids) < 2:
        raise ValueError(
            "Flatmap voxel correlation requires at least 2 neurons with usable "
            "nodes after applying region and coordinate filters."
        )
    if count_data.count_matrix.shape[1] == 0:
        raise ValueError(
            "Flatmap voxel correlation requires at least one occupied voxel."
        )

    corr = pearson_correlation_from_counts(count_data.count_matrix)
    result = compute_clustermap_data(
        corr,
        count_data.neuron_ids,
        method=method,
        n_clusters=n_clusters,
    )
    result.unassigned_neuron_ids = list(count_data.unassigned_neuron_ids)
    provenance = FlatmapParquetCorrelationProvenance(
        style=style,
        y_bins=int(y_bins),
        x_bins=int(x_bins),
        depth_bin_um=float(depth_bin_um),
        include_depth_minus_one=bool(include_depth_minus_one),
        volume_shape=volume_shape,
        x_bounds=(x_lower, x_upper),
        y_bounds=(y_lower, y_upper),
        depth_range_um=(depth_lower, depth_upper),
        collapse_depth=bool(collapse_depth),
    )
    return result, count_data, provenance


def compute_flatmap_voxel_correlation_result(
    source: FlatmapVoxelCorrelationSource,
    *,
    method: str = "average",
    n_clusters: int = 5,
    region_mask: np.ndarray | None = None,
) -> tuple[ClusterResult, FlatmapCountMatrix]:
    """Cluster neurons by Pearson correlation in flatmap heatmap voxel space."""
    count_data = build_flatmap_count_matrix(source, region_mask=region_mask)
    if len(count_data.neuron_ids) < 2:
        raise ValueError(
            "Flatmap voxel correlation requires at least 2 rendered neurons "
            "after applying the selected filters."
        )
    if count_data.count_matrix.shape[1] == 0:
        raise ValueError(
            "Flatmap voxel correlation requires at least one occupied voxel."
        )

    corr = pearson_correlation_from_counts(count_data.count_matrix)
    result = compute_clustermap_data(
        corr,
        count_data.neuron_ids,
        method=method,
        n_clusters=n_clusters,
    )
    result.unassigned_neuron_ids = list(count_data.unassigned_neuron_ids)
    return result, count_data
