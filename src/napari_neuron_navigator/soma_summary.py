"""Summarize hemisphere locations for soma nodes stored in Parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

DEFAULT_COORD_AXIS = 2
DEFAULT_MIDLINE = 5695.0
DEFAULT_TOLERANCE = 1.0

_COORD_COLUMNS = {
    0: "x",
    1: "y",
    2: "z",
}


@dataclass(frozen=True)
class SomaHemisphereSummary:
    """Node-level hemisphere summary for soma rows in a Parquet file."""

    parquet_path: Path
    coord_axis: int
    coord_column: str
    midline: float
    tolerance: float
    total_soma_nodes: int
    neurons_with_soma: int
    left_count: int
    midline_count: int
    right_count: int
    coord_min: float | None
    coord_mean: float | None
    coord_max: float | None


def _coord_column(coord_axis: int) -> str:
    """Return the parquet column name for the selected coordinate axis."""
    try:
        return _COORD_COLUMNS[coord_axis]
    except KeyError as exc:
        raise ValueError("coord_axis must be one of 0, 1, or 2") from exc


def summarize_soma_hemispheres(
    parquet_path: Path | str,
    *,
    coord_axis: int = DEFAULT_COORD_AXIS,
    midline: float = DEFAULT_MIDLINE,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SomaHemisphereSummary:
    """Summarize hemisphere counts for all soma rows in a Parquet file."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    coord_column = _coord_column(coord_axis)
    parquet_sql_path = str(path.resolve()).replace("\\", "/").replace("'", "''")

    query = f"""
        SELECT
            COUNT(*) AS total_soma_nodes,
            COUNT(DISTINCT file_id) AS neurons_with_soma,
            COALESCE(SUM(CASE WHEN hemisphere = 'left' THEN 1 ELSE 0 END), 0) AS left_count,
            COALESCE(SUM(CASE WHEN hemisphere = 'midline' THEN 1 ELSE 0 END), 0) AS midline_count,
            COALESCE(SUM(CASE WHEN hemisphere = 'right' THEN 1 ELSE 0 END), 0) AS right_count,
            MIN(coord) AS coord_min,
            AVG(coord) AS coord_mean,
            MAX(coord) AS coord_max
        FROM (
            SELECT
                file_id,
                {coord_column} AS coord,
                CASE
                    WHEN ABS({coord_column} - ?) < ? THEN 'midline'
                    WHEN {coord_column} < ? THEN 'left'
                    ELSE 'right'
                END AS hemisphere
            FROM read_parquet('{parquet_sql_path}')
            WHERE type = 1
        )
    """

    with duckdb.connect() as conn:
        row = conn.execute(query, [midline, tolerance, midline]).fetchone()

    return SomaHemisphereSummary(
        parquet_path=path.resolve(),
        coord_axis=coord_axis,
        coord_column=coord_column,
        midline=midline,
        tolerance=tolerance,
        total_soma_nodes=int(row[0]),
        neurons_with_soma=int(row[1]),
        left_count=int(row[2]),
        midline_count=int(row[3]),
        right_count=int(row[4]),
        coord_min=row[5],
        coord_mean=row[6],
        coord_max=row[7],
    )


def format_soma_hemisphere_summary(summary: SomaHemisphereSummary) -> str:
    """Render a compact human-readable summary."""

    def _line(label: str, count: int) -> str:
        total = summary.total_soma_nodes
        pct = 0.0 if total == 0 else (100.0 * count / total)
        return f"  {label}: {count} ({pct:.2f}%)"

    lines = [
        f"Parquet: {summary.parquet_path}",
        f"Soma node rows: {summary.total_soma_nodes}",
        f"Neurons with soma: {summary.neurons_with_soma}",
        f"Axis: {summary.coord_column} ({summary.coord_axis})",
        f"Midline: {summary.midline:.2f} um",
        f"Tolerance: {summary.tolerance:.2f} um",
        "",
        "Hemisphere summary:",
        _line("left", summary.left_count),
        _line("midline", summary.midline_count),
        _line("right", summary.right_count),
    ]

    if summary.coord_min is not None:
        lines.extend(
            [
                "",
                f"{summary.coord_column.upper()} coordinate summary:",
                f"  min: {summary.coord_min:.2f} um",
                f"  mean: {summary.coord_mean:.2f} um",
                f"  max: {summary.coord_max:.2f} um",
            ]
        )

    return "\n".join(lines)
