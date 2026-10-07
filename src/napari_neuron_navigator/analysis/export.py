"""Export helpers for analysis clustering results."""

from __future__ import annotations

import io
import json
import zlib
from pathlib import Path
from typing import Callable, Mapping, Sequence

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.cluster.hierarchy import dendrogram

from .clustering import ClusterResult
from .clustermap_contrast import ClustermapContrast

PARQUET_EXPORT_VERSION = "1"
PARQUET_METADATA_PREFIX = "napari_neuron_navigator.analysis_export."
LEGACY_PARQUET_METADATA_PREFIX = "napari_swc_viewer.analysis_export."
DEFAULT_PREVIEW_HEATMAP_SIZE = 1024
MIN_EXPORT_HEATMAP_SIZE = 512
DENDROGRAM_LINEWIDTH = 0.5
CLUSTERMAP_WIDTH_RATIOS = [0.18, 0.03, 0.75, 0.04]
CLUSTERMAP_HEIGHT_RATIOS = [0.36, 0.03, 0.61]
CLUSTERMAP_FIGURE_XLABEL_Y = 0.04


def rgba_to_hex(rgba: Sequence[float]) -> str:
    """Convert an RGBA float sequence in [0, 1] to a hex color string."""
    values = np.clip(np.asarray(rgba, dtype=float), 0.0, 1.0)
    rgb = np.rint(values[:3] * 255.0).astype(np.uint8)
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _progress(
    callback: Callable[[str, int, int], None] | None,
    message: str,
    step: int,
    total: int,
) -> None:
    """Emit one optional export progress update."""
    if callback is not None:
        callback(message, step, total)


def _source_parquet_path(
    result: ClusterResult,
    source_parquet_path: str | Path | None = None,
) -> Path:
    """Resolve the source parquet path from an override or result metadata."""
    if source_parquet_path is not None:
        return Path(source_parquet_path)

    metadata = result.metadata
    if metadata is None or not metadata.source_parquet_path:
        raise ValueError("Cluster result metadata does not include a source parquet path.")

    return Path(metadata.source_parquet_path)


def _ordered_cluster_data(
    result: ClusterResult,
) -> tuple[list[str], list[int], np.ndarray]:
    """Return ordered neuron IDs, labels, and distances in leaf order."""
    order = [int(index) for index in result.reorder_indices.tolist()]
    ordered_ids = [result.neuron_ids[index] for index in order]
    ordered_labels = [int(result.labels[index]) for index in order]
    ordered_distance = np.asarray(
        result.distance_matrix[np.ix_(result.reorder_indices, result.reorder_indices)],
        dtype=np.float32,
    )
    return ordered_ids, ordered_labels, ordered_distance


def _cluster_label_colors(
    result: ClusterResult,
    cluster_color_map: Mapping[str, Sequence[float]],
) -> dict[int, Sequence[float]]:
    """Return one RGBA color per cluster label."""
    label_colors: dict[int, Sequence[float]] = {}
    default_color = [0.5, 0.5, 0.5, 1.0]
    for neuron_id, label in zip(result.neuron_ids, result.labels):
        label_colors.setdefault(int(label), cluster_color_map.get(neuron_id, default_color))
    return label_colors


def _leaf_order_sample_indices(
    leaf_order: np.ndarray,
    max_render_size: int,
) -> np.ndarray:
    """Return result-order indices sampled along dendrogram leaf order."""
    if leaf_order.ndim != 1:
        leaf_order = np.asarray(leaf_order, dtype=np.intp).reshape(-1)
    n_items = int(leaf_order.size)
    if n_items == 0:
        return np.array([], dtype=np.intp)

    limit = max(1, int(max_render_size))
    if n_items <= limit:
        return np.asarray(leaf_order, dtype=np.intp)

    positions = np.linspace(0, n_items - 1, num=limit, dtype=np.intp)
    return np.asarray(leaf_order[positions], dtype=np.intp)


def _effective_render_heatmap_size(
    n_neurons: int,
    *,
    max_render_size: int | None,
    figsize: tuple[float, float],
    dpi: int | None,
) -> int:
    """Return the heatmap side length to render for one figure."""
    if max_render_size is not None:
        return max(1, min(int(max_render_size), int(n_neurons)))

    base_dpi = int(dpi) if dpi is not None else 200
    pixel_target = int(max(figsize) * base_dpi)
    target = max(DEFAULT_PREVIEW_HEATMAP_SIZE, pixel_target)
    return max(1, min(target, int(n_neurons)))


def _sampled_heatmap_inputs(
    result: ClusterResult,
    cluster_color_map: Mapping[str, Sequence[float]] | None,
    *,
    max_render_size: int | None,
    figsize: tuple[float, float],
    dpi: int | None,
) -> tuple[np.ndarray, list[Sequence[float]] | None]:
    """Return a sampled heatmap matrix and optional cluster colors."""
    leaf_order = np.asarray(result.reorder_indices, dtype=np.intp).reshape(-1)
    render_size = _effective_render_heatmap_size(
        len(result.neuron_ids),
        max_render_size=max_render_size,
        figsize=figsize,
        dpi=dpi,
    )
    sampled_indices = _leaf_order_sample_indices(leaf_order, render_size)
    sampled_heatmap = np.asarray(
        result.distance_matrix[np.ix_(sampled_indices, sampled_indices)],
        dtype=np.float32,
    )

    if not cluster_color_map:
        return sampled_heatmap, None

    sampled_colors = [
        list(cluster_color_map.get(result.neuron_ids[int(index)], [0.5, 0.5, 0.5, 1.0]))[:3]
        for index in sampled_indices.tolist()
    ]
    return sampled_heatmap, sampled_colors


def _distance_colorbar_label(result: ClusterResult) -> str:
    """Return a human-readable colorbar label for the plotted distance."""
    metadata = result.metadata
    if metadata is None:
        return "Distance"

    metric = metadata.distance_metric
    if metric == "one_minus_pearson_r":
        return "Distance (1 - Pearson r)"
    if metric == "euclidean_um":
        return "Distance (um)"
    return "Distance"


def build_clustermap_figure(
    result: ClusterResult,
    cluster_color_map: Mapping[str, Sequence[float]] | None = None,
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    figsize: tuple[float, float] = (6.0, 6.0),
    max_render_size: int | None = None,
    dpi: int | None = None,
    contrast: ClustermapContrast | None = None,
):
    """Build a clustermap; omitted contrast uses cached automatic limits."""
    figure = plt.Figure(figsize=figsize)
    populate_clustermap_figure(
        figure,
        result,
        cluster_color_map,
        title=title,
        x_label=x_label,
        y_label=y_label,
        figsize=figsize,
        max_render_size=max_render_size,
        dpi=dpi,
        contrast=contrast,
    )
    return figure


def populate_clustermap_figure(
    figure,
    result: ClusterResult,
    cluster_color_map: Mapping[str, Sequence[float]] | None = None,
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    figsize: tuple[float, float] = (6.0, 6.0),
    max_render_size: int | None = None,
    dpi: int | None = None,
    contrast: ClustermapContrast | None = None,
):
    """Populate one existing figure with the clustermap layout."""
    statistics = result.clustermap_contrast_statistics
    contrast = contrast if contrast is not None else statistics.resolve()
    figure.clear()
    grid = GridSpec(
        3,
        4,
        figure=figure,
        width_ratios=CLUSTERMAP_WIDTH_RATIOS,
        height_ratios=CLUSTERMAP_HEIGHT_RATIOS,
        wspace=0.02,
        hspace=0.02,
    )

    ax_top = figure.add_subplot(grid[0, 2])
    ax_left = figure.add_subplot(grid[2, 0])
    ax_top_colors = figure.add_subplot(grid[1, 2])
    ax_left_colors = figure.add_subplot(grid[2, 1])
    ax_heatmap = figure.add_subplot(grid[2, 2])
    ax_colorbar = figure.add_subplot(grid[2, 3])

    dendrogram(
        result.linkage_matrix,
        ax=ax_top,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="black",
    )
    dendrogram(
        result.linkage_matrix,
        ax=ax_left,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="black",
    )
    ax_left.invert_yaxis()
    for axis in (ax_top, ax_left):
        for collection in axis.collections:
            collection.set_linewidth(DENDROGRAM_LINEWIDTH)

    for axis in (ax_top, ax_left):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    heatmap_data, cluster_colors = _sampled_heatmap_inputs(
        result,
        cluster_color_map,
        max_render_size=max_render_size,
        figsize=figsize,
        dpi=dpi,
    )
    image = ax_heatmap.imshow(
        heatmap_data,
        cmap="coolwarm",
        interpolation="nearest",
        origin="upper",
        aspect="auto",
        vmin=contrast.minimum,
        vmax=contrast.maximum,
    )
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    colorbar = figure.colorbar(
        image,
        cax=ax_colorbar,
        extend=statistics.extension(contrast),
        ticks=MaxNLocator(nbins=5),
        format=FuncFormatter(lambda value, _pos: contrast.format_value(value)),
    )
    colorbar.set_label(_distance_colorbar_label(result))
    ax_colorbar.set_title(contrast.mode.capitalize(), fontsize=8)
    explanation = statistics.explanation(contrast)
    if explanation:
        figure.text(0.5, 0.01, explanation, ha="center", fontsize=8)

    if cluster_colors:
        cluster_array = np.asarray(cluster_colors, dtype=np.float32)
        ax_top_colors.imshow(
            cluster_array[np.newaxis, :, :],
            interpolation="nearest",
            origin="lower",
            aspect="auto",
        )
        ax_left_colors.imshow(
            cluster_array[:, np.newaxis, :],
            interpolation="nearest",
            origin="upper",
            aspect="auto",
        )
    for axis in (ax_top_colors, ax_left_colors):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        if not cluster_colors:
            axis.set_visible(False)

    if title:
        figure.subplots_adjust(top=0.94)
        figure.suptitle(title)
    if x_label:
        figure.supxlabel(x_label, y=CLUSTERMAP_FIGURE_XLABEL_Y)

    return figure


def save_dendrogram_figure(
    output_path: str | Path,
    result: ClusterResult,
    cluster_color_map: Mapping[str, Sequence[float]] | None = None,
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    dpi: int = 300,
    contrast: ClustermapContrast | None = None,
) -> Path:
    """Save the clustermap-style dendrogram figure as a raster image."""
    render_size = max(
        MIN_EXPORT_HEATMAP_SIZE,
        int(max((6.0, 6.0)) * int(dpi)),
    )
    figure = build_clustermap_figure(
        result,
        cluster_color_map,
        title=title,
        x_label=x_label,
        y_label=y_label,
        dpi=dpi,
        max_render_size=render_size,
        contrast=contrast,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


def _load_clustered_neuron_summary(
    parquet_path: str | Path,
    neuron_ids: Sequence[str],
) -> pd.DataFrame:
    """Load one summary row per clustered neuron from the source parquet."""
    if not neuron_ids:
        return pd.DataFrame(
            columns=[
                "file_id",
                "neuron_id",
                "subject",
                "soma_x_um",
                "soma_y_um",
                "soma_z_um",
                "soma_region_acronym",
            ]
        )

    path = Path(parquet_path)
    path_str = str(path.resolve()).replace("\\", "/").replace("'", "''")
    placeholders = ", ".join(["?"] * len(neuron_ids))

    conn = duckdb.connect()
    try:
        return conn.execute(
            f"""
            SELECT
                file_id,
                MAX(neuron_id) AS neuron_id,
                MAX(subject) AS subject,
                AVG(CASE WHEN type = 1 THEN x END) AS soma_x_um,
                AVG(CASE WHEN type = 1 THEN y END) AS soma_y_um,
                AVG(CASE WHEN type = 1 THEN z END) AS soma_z_um,
                MAX(CASE WHEN type = 1 THEN region_acronym END) AS soma_region_acronym
            FROM read_parquet('{path_str}')
            WHERE file_id IN ({placeholders})
            GROUP BY file_id
            ORDER BY file_id
            """,
            list(neuron_ids),
        ).fetchdf()
    finally:
        conn.close()


def _metadata_frame(
    result: ClusterResult,
    *,
    figure_title: str = "",
    x_label: str = "",
    y_label: str = "",
    dpi: int | None = None,
) -> pd.DataFrame:
    """Return a human-readable key/value metadata sheet."""
    metadata = result.metadata.to_dict() if result.metadata is not None else {}
    ordered_ids, ordered_labels, _ordered_distance = _ordered_cluster_data(result)
    rows: list[tuple[str, object]] = [
        ("figure_title", figure_title),
        ("figure_x_label", x_label),
        ("figure_y_label", y_label),
        ("figure_dpi", dpi),
        ("neuron_ids_in_result_order", json.dumps(list(result.neuron_ids))),
        ("cluster_labels_in_result_order", json.dumps([int(value) for value in result.labels.tolist()])),
        ("neuron_ids_in_dendrogram_order", json.dumps(ordered_ids)),
        ("cluster_labels_in_dendrogram_order", json.dumps(ordered_labels)),
    ]
    rows.extend((key, json.dumps(value) if isinstance(value, (list, dict)) else value) for key, value in metadata.items())
    return pd.DataFrame(rows, columns=["field", "value"])


def _autosize_dataframe_columns(
    worksheet,
    dataframe: pd.DataFrame,
    *,
    start_col: int = 0,
) -> None:
    """Size worksheet columns to the data being written."""
    for column_offset, column_name in enumerate(dataframe.columns, start=start_col):
        series = dataframe[column_name]
        width = max(
            len(str(column_name)),
            *(len(str(value)) for value in series.tolist()),
        )
        worksheet.set_column(column_offset, column_offset, min(max(width + 2, 12), 40))


def export_cluster_workbook(
    output_path: str | Path,
    result: ClusterResult,
    cluster_color_map: Mapping[str, Sequence[float]],
    *,
    source_parquet_path: str | Path | None = None,
    figure_title: str = "",
    x_label: str = "",
    y_label: str = "",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Write a human-readable workbook of clustered neurons and metadata."""
    total = 3
    source_path = _source_parquet_path(result, source_parquet_path)
    _progress(progress_callback, "Loading clustered neuron metadata...", 1, total)

    ordered_ids, ordered_labels, _ordered_distance = _ordered_cluster_data(result)
    summary_df = _load_clustered_neuron_summary(source_path, ordered_ids)
    summary_by_id = summary_df.set_index("file_id").to_dict(orient="index")

    rows: list[dict[str, object]] = []
    for order_index, (file_id, label) in enumerate(zip(ordered_ids, ordered_labels), start=1):
        summary = summary_by_id.get(file_id, {})
        color = cluster_color_map.get(file_id, [0.5, 0.5, 0.5, 1.0])
        rows.append(
            {
                "dendrogram_order": order_index,
                "file_id": file_id,
                "neuron_id": summary.get("neuron_id"),
                "subject": summary.get("subject"),
                "cluster_assignment": int(label),
                "cluster_color_hex": rgba_to_hex(color),
                "soma_x_um": summary.get("soma_x_um"),
                "soma_y_um": summary.get("soma_y_um"),
                "soma_z_um": summary.get("soma_z_um"),
                "soma_region_acronym": summary.get("soma_region_acronym"),
            }
        )

    clusters_df = pd.DataFrame(rows)
    metadata_df = _metadata_frame(
        result,
        figure_title=figure_title,
        x_label=x_label,
        y_label=y_label,
    )

    _progress(progress_callback, "Writing Excel workbook...", 2, total)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        clusters_df.to_excel(writer, sheet_name="Clusters", index=False)
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

        workbook = writer.book
        clusters_sheet = writer.sheets["Clusters"]
        metadata_sheet = writer.sheets["Metadata"]
        label_colors = _cluster_label_colors(result, cluster_color_map)

        cluster_formats = {
            label: workbook.add_format({"bg_color": rgba_to_hex(color)})
            for label, color in label_colors.items()
        }

        cluster_col = clusters_df.columns.get_loc("cluster_assignment")
        color_col = clusters_df.columns.get_loc("cluster_color_hex")
        for row_index, label in enumerate(clusters_df["cluster_assignment"].tolist(), start=1):
            cell_format = cluster_formats.get(int(label))
            clusters_sheet.write_number(row_index, cluster_col, int(label), cell_format)
            clusters_sheet.write_string(
                row_index,
                color_col,
                str(clusters_df.iloc[row_index - 1]["cluster_color_hex"]),
                cell_format,
            )

        clusters_sheet.freeze_panes(1, 0)
        clusters_sheet.autofilter(0, 0, len(clusters_df), len(clusters_df.columns) - 1)
        metadata_sheet.freeze_panes(1, 0)
        _autosize_dataframe_columns(clusters_sheet, clusters_df)
        _autosize_dataframe_columns(metadata_sheet, metadata_df)

    _progress(progress_callback, "Done", 3, total)
    return path


def export_distance_workbook(
    output_path: str | Path,
    result: ClusterResult,
    cluster_color_map: Mapping[str, Sequence[float]],
    *,
    figure_title: str = "",
    x_label: str = "",
    y_label: str = "",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Write a human-readable workbook of dendrogram-ordered distances."""
    total = 2
    _progress(progress_callback, "Preparing ordered distance matrix...", 1, total)

    ordered_ids, ordered_labels, ordered_distance = _ordered_cluster_data(result)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        distances_sheet = workbook.add_worksheet("Distances")
        writer.sheets["Distances"] = distances_sheet

        metadata_df = _metadata_frame(
            result,
            figure_title=figure_title,
            x_label=x_label,
            y_label=y_label,
        )
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        metadata_sheet = writer.sheets["Metadata"]

        label_colors = _cluster_label_colors(result, cluster_color_map)
        header_format = workbook.add_format({"bold": True, "border": 1})
        numeric_format = workbook.add_format({"num_format": "0.0000"})
        cluster_formats = {
            label: workbook.add_format({"bg_color": rgba_to_hex(color), "border": 1})
            for label, color in label_colors.items()
        }

        distances_sheet.write(0, 0, "Neuron ID", header_format)
        distances_sheet.write(0, 1, "Cluster", header_format)
        for column_index, (neuron_id, label) in enumerate(zip(ordered_ids, ordered_labels), start=2):
            distances_sheet.write(0, column_index, neuron_id, cluster_formats.get(int(label), header_format))

        for row_index, (neuron_id, label, row_values) in enumerate(
            zip(ordered_ids, ordered_labels, ordered_distance.tolist()),
            start=1,
        ):
            row_format = cluster_formats.get(int(label), header_format)
            distances_sheet.write(row_index, 0, neuron_id, row_format)
            distances_sheet.write_number(row_index, 1, int(label), row_format)
            for column_index, value in enumerate(row_values, start=2):
                distances_sheet.write_number(row_index, column_index, float(value), numeric_format)

        distances_sheet.freeze_panes(1, 2)
        distances_sheet.set_column(0, 0, min(max(len(max(ordered_ids, key=len, default="Neuron ID")) + 2, 12), 40))
        distances_sheet.set_column(1, 1, 12)
        for column_index, neuron_id in enumerate(ordered_ids, start=2):
            distances_sheet.set_column(
                column_index,
                column_index,
                min(max(len(neuron_id) + 2, 12), 32),
            )

        metadata_sheet.freeze_panes(1, 0)
        _autosize_dataframe_columns(metadata_sheet, metadata_df)

    _progress(progress_callback, "Done", 2, total)
    return path


def _encoded_array_payload(array: np.ndarray) -> bytes:
    """Encode one numeric array as compressed ``.npy`` bytes."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return zlib.compress(buffer.getvalue())


def _decoded_array_payload(payload: bytes) -> np.ndarray:
    """Decode one compressed ``.npy`` payload."""
    return np.load(io.BytesIO(zlib.decompress(payload)), allow_pickle=False)


def _parquet_export_json_payload(result: ClusterResult) -> dict[str, object]:
    """Build the human-readable JSON metadata stored in the export parquet."""
    ordered_ids, ordered_labels, _ordered_distance = _ordered_cluster_data(result)
    return {
        "version": PARQUET_EXPORT_VERSION,
        "run_metadata": result.metadata.to_dict() if result.metadata is not None else None,
        "neuron_ids_in_result_order": list(result.neuron_ids),
        "cluster_labels_in_result_order": [int(value) for value in result.labels.tolist()],
        "neuron_ids_in_dendrogram_order": ordered_ids,
        "cluster_labels_in_dendrogram_order": ordered_labels,
    }


def export_extended_parquet(
    output_path: str | Path,
    result: ClusterResult,
    *,
    source_parquet_path: str | Path | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Write an extended parquet with cluster assignments and export metadata."""
    total = 3
    source_path = _source_parquet_path(result, source_parquet_path)
    _progress(progress_callback, "Loading source parquet...", 1, total)

    table = pq.read_table(source_path)
    cluster_by_file = {
        str(file_id): int(label)
        for file_id, label in zip(result.neuron_ids, result.labels.tolist())
    }
    cluster_values = [cluster_by_file.get(str(file_id)) for file_id in table.column("file_id").to_pylist()]
    cluster_array = pa.array(cluster_values, type=pa.int32())

    if "cluster_assignment" in table.column_names:
        column_index = table.column_names.index("cluster_assignment")
        table = table.set_column(column_index, "cluster_assignment", cluster_array)
    else:
        table = table.append_column("cluster_assignment", cluster_array)

    _progress(progress_callback, "Writing extended parquet...", 2, total)

    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[f"{PARQUET_METADATA_PREFIX}version".encode("utf-8")] = PARQUET_EXPORT_VERSION.encode("utf-8")
    schema_metadata[f"{PARQUET_METADATA_PREFIX}metadata_json".encode("utf-8")] = json.dumps(
        _parquet_export_json_payload(result),
        sort_keys=True,
    ).encode("utf-8")
    schema_metadata[f"{PARQUET_METADATA_PREFIX}distance_matrix_npy_zlib".encode("utf-8")] = _encoded_array_payload(
        np.asarray(result.distance_matrix, dtype=np.float32)
    )
    schema_metadata[f"{PARQUET_METADATA_PREFIX}linkage_matrix_npy_zlib".encode("utf-8")] = _encoded_array_payload(
        np.asarray(result.linkage_matrix, dtype=np.float64)
    )

    table = table.replace_schema_metadata(schema_metadata)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")

    _progress(progress_callback, "Done", 3, total)
    return path


def read_extended_parquet_analysis_metadata(
    parquet_path: str | Path,
) -> dict[str, object]:
    """Read back analysis export metadata for tests or inspection."""
    metadata = dict(pq.read_schema(parquet_path).metadata or {})
    prefix = PARQUET_METADATA_PREFIX
    if f"{prefix}metadata_json".encode("utf-8") not in metadata:
        prefix = LEGACY_PARQUET_METADATA_PREFIX
    json_key = f"{prefix}metadata_json".encode("utf-8")
    distance_key = f"{prefix}distance_matrix_npy_zlib".encode("utf-8")
    linkage_key = f"{prefix}linkage_matrix_npy_zlib".encode("utf-8")

    payload = json.loads(metadata[json_key].decode("utf-8"))
    payload["distance_matrix"] = _decoded_array_payload(metadata[distance_key])
    payload["linkage_matrix"] = _decoded_array_payload(metadata[linkage_key])
    return payload
