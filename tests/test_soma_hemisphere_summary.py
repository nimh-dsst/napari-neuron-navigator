"""Tests for soma hemisphere summary helpers."""

from __future__ import annotations

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from napari_neuron_navigator.soma_summary import (
    format_soma_hemisphere_summary,
    summarize_soma_hemispheres,
)


def test_summarize_soma_hemispheres_counts_and_formats_output(tmp_path) -> None:
    """Soma rows are classified by axis coordinate and formatted for display."""
    parquet_path = tmp_path / "sample.parquet"
    table = pa.table(
        {
            "file_id": ["a.swc", "b.swc", "c.swc", "d.swc", "ignored.swc"],
            "type": [1, 1, 1, 1, 2],
            "z": [10.0, 49.5, 50.2, 90.0, 999.0],
        }
    )
    pq.write_table(table, parquet_path)

    summary = summarize_soma_hemispheres(
        parquet_path,
        coord_axis=2,
        midline=50.0,
        tolerance=1.0,
    )

    assert summary.total_soma_nodes == 4
    assert summary.neurons_with_soma == 4
    assert summary.left_count == 1
    assert summary.midline_count == 2
    assert summary.right_count == 1
    assert summary.coord_min == 10.0
    assert summary.coord_mean == pytest.approx(49.925)
    assert summary.coord_max == 90.0

    rendered = format_soma_hemisphere_summary(summary)
    assert "Soma node rows: 4" in rendered
    assert "left: 1 (25.00%)" in rendered
    assert "midline: 2 (50.00%)" in rendered
    assert "right: 1 (25.00%)" in rendered
