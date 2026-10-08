"""Standard point Parquet import, heatmap generation, and CSV normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .atlas_utils import world_coords_xyz_to_atlas_voxels
from .hemisphere import get_atlas_midline

REQUIRED_POINT_COLUMNS = ("label", "x", "y", "z")
OPTIONAL_POINT_COLUMNS = ("region_name", "acronym", "id", "hemisphere")
STANDARD_POINT_COLUMNS = REQUIRED_POINT_COLUMNS + OPTIONAL_POINT_COLUMNS
POINT_PARQUET_ORIGIN_NOT_RECORDED = "(not recorded)"

BLTR_STANDARD_MAPPING = {
    "label": "marker",
    "x": "atlas_x",
    "y": "atlas_y",
    "z": "atlas_z",
    "region_name": "region_name",
    "acronym": "region_acronym",
    "id": "region_id",
    "hemisphere": "region_hemisphere",
}
BLTR_EXTRA_COLUMNS = (
    "experiment_x",
    "experiment_y",
    "experiment_z",
    "strength",
    "channel_mono_channel",
)

_STRING_OPTIONAL_COLUMNS = ("region_name", "acronym", "hemisphere")


class PointImportError(ValueError):
    """Raised when a point import file or mapping is invalid."""


@dataclass(frozen=True)
class AtlasValidationSummary:
    """Summary of optional metadata validation against an atlas."""

    total_points: int
    checked_fields: tuple[str, ...]
    mismatch_counts: dict[str, int]
    mismatches: pd.DataFrame

    @property
    def total_mismatched_rows(self) -> int:
        return int(len(self.mismatches))

    @property
    def has_mismatches(self) -> bool:
        return self.total_mismatched_rows > 0


@dataclass
class BatchPointParquetConversionSummary:
    """Summary statistics for converting one or more point CSV files to Parquet."""

    discovered_files: int = 0
    processed_files: int = 0
    rows_written: int = 0


@dataclass(frozen=True)
class PointParquetAppendSummary:
    """Summary statistics for appending rows into an existing point Parquet."""

    appended_rows: int
    total_rows: int


def _empty_string_series(length: int) -> pd.Series:
    return pd.Series(pd.array([pd.NA] * length, dtype="string"))


def _empty_int_series(length: int) -> pd.Series:
    return pd.Series(pd.array([pd.NA] * length, dtype="Int64"))


def _normalize_string_series(series: pd.Series) -> pd.Series:
    def normalize(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        return str(value)

    return series.map(normalize).astype("string")


def _nonempty_string_mask(series: pd.Series) -> pd.Series:
    normalized = _normalize_string_series(series)
    return normalized.notna() & normalized.str.strip().ne("")


def _normalize_region_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("/", ", ")
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def _compare_string_series(left: pd.Series, right: pd.Series) -> pd.Series:
    left_norm = _normalize_string_series(left).str.strip()
    right_norm = _normalize_string_series(right).str.strip()
    return left_norm == right_norm


def _compare_region_name_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.map(_normalize_region_name) == right.map(_normalize_region_name)


def _compare_hemisphere_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return (
        _normalize_string_series(left).str.strip().str.lower()
        == _normalize_string_series(right).str.strip().str.lower()
    )


def _path_exists_message(path: Path, kind: str) -> str:
    return f"{kind} not found: {path}"


def _escape_duckdb_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _read_standard_point_parquet_schema(parquet_path: str | Path) -> pa.Schema:
    """Read and lightly validate a standardized point Parquet schema."""

    path = Path(parquet_path)
    try:
        schema = pq.read_schema(path)
    except FileNotFoundError as exc:
        raise PointImportError(_path_exists_message(path, "Point Parquet file")) from exc
    except Exception as exc:
        raise PointImportError(f"Failed to read Point Parquet schema: {path}") from exc

    missing_required = [
        column for column in REQUIRED_POINT_COLUMNS if column not in schema.names
    ]
    if missing_required:
        columns = ", ".join(missing_required)
        raise PointImportError(f"Missing required point column(s): {columns}")
    return schema


def point_parquet_has_origin_csv(parquet_path: str | Path) -> bool:
    """Return whether a point Parquet stores provenance in ``origin_csv``."""

    return "origin_csv" in _read_standard_point_parquet_schema(parquet_path).names


def _dataframe_arrow_schema(df: pd.DataFrame) -> pa.Schema:
    """Return an Arrow schema for a dataframe without pandas metadata."""

    return pa.Table.from_pandas(df, preserve_index=False).schema.remove_metadata()


def _validate_append_schema_compatibility(
    incoming_df: pd.DataFrame,
    target_schema: pa.Schema,
) -> pa.Schema:
    """Validate strict append compatibility against an existing schema."""

    incoming_schema = _dataframe_arrow_schema(incoming_df)
    return _validate_point_schema_compatibility(incoming_schema, target_schema)


def _is_utf8_arrow_type(data_type: pa.DataType) -> bool:
    """Return whether an Arrow type stores UTF-8 text values."""

    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return True
    is_string_view = getattr(pa.types, "is_string_view", None)
    return bool(is_string_view is not None and is_string_view(data_type))


def _point_field_types_are_compatible(
    incoming_type: pa.DataType,
    target_type: pa.DataType,
) -> bool:
    """Return whether two Arrow field types are append-compatible."""

    return incoming_type == target_type or (
        _is_utf8_arrow_type(incoming_type) and _is_utf8_arrow_type(target_type)
    )


def _validate_point_schema_compatibility(
    incoming_schema: pa.Schema,
    target_schema: pa.Schema,
) -> pa.Schema:
    """Validate strict append compatibility between two Arrow schemas."""

    target_schema = target_schema.remove_metadata()
    incoming_schema = incoming_schema.remove_metadata()
    incoming_names = list(incoming_schema.names)
    target_names = list(target_schema.names)

    if incoming_names != target_names:
        missing_columns = [name for name in target_names if name not in incoming_names]
        if missing_columns:
            columns = ", ".join(missing_columns)
            raise PointImportError(
                f"Point Parquet schema mismatch: missing column(s): {columns}"
            )

        extra_columns = [name for name in incoming_names if name not in target_names]
        if extra_columns:
            columns = ", ".join(extra_columns)
            raise PointImportError(
                f"Point Parquet schema mismatch: extra column(s): {columns}"
            )

        raise PointImportError(
            "Point Parquet schema mismatch: column order does not match existing file."
        )

    for incoming_field, target_field in zip(incoming_schema, target_schema):
        if not _point_field_types_are_compatible(
            incoming_field.type,
            target_field.type,
        ):
            raise PointImportError(
                "Point Parquet schema mismatch: "
                f"column '{target_field.name}' has type {incoming_field.type} "
                f"but existing file uses {target_field.type}."
            )

    return target_schema


def _align_table_to_schema(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
    """Return a table aligned to the target schema, preserving schema metadata."""

    target_without_metadata = target_schema.remove_metadata()
    if table.schema.remove_metadata() != target_without_metadata:
        table = table.cast(target_without_metadata)
    return table.replace_schema_metadata(target_schema.metadata)


def _write_point_parquet_with_appended_tables(
    existing_tables: Iterable[pa.Table],
    append_tables: Iterable[pa.Table],
    output_path: Path,
    target_schema: pa.Schema,
) -> Path:
    """Write appended point Parquet data to a temp file and return its path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    writer: pq.ParquetWriter | None = None

    try:
        writer = pq.ParquetWriter(
            temp_output_path,
            target_schema,
            compression="snappy",
        )
        for row_group_table in existing_tables:
            writer.write_table(_align_table_to_schema(row_group_table, target_schema))
        for append_table in append_tables:
            writer.write_table(_align_table_to_schema(append_table, target_schema))
    except Exception:
        if writer is not None:
            writer.close()
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise
    else:
        writer.close()

    return temp_output_path


def _normalize_origin_csv_value(value: Any) -> str:
    if pd.isna(value):
        return POINT_PARQUET_ORIGIN_NOT_RECORDED
    text = str(value).strip()
    return text if text else POINT_PARQUET_ORIGIN_NOT_RECORDED


def _ensure_origin_csv_provenance(
    df: pd.DataFrame,
    default_origin_csv: str,
) -> pd.DataFrame:
    """Return a dataframe with a normalized ``origin_csv`` provenance column."""

    normalized = df.copy()
    default_values = pd.Series(
        pd.array([default_origin_csv] * len(normalized), dtype="string")
    )

    if "origin_csv" not in normalized.columns:
        normalized["origin_csv"] = default_values
        return normalized

    existing = _normalize_string_series(normalized["origin_csv"]).str.strip()
    missing_origin = existing.isna() | existing.eq("")
    normalized["origin_csv"] = existing.where(~missing_origin, default_values)
    return normalized


def _schema_with_origin_csv(schema: pa.Schema) -> pa.Schema:
    """Return a schema that includes ``origin_csv`` as the final string field."""

    if "origin_csv" in schema.names:
        return schema
    return pa.schema([*schema, pa.field("origin_csv", pa.string())], metadata=schema.metadata)


def _table_with_origin_csv(table: pa.Table, origin_csv: str) -> pa.Table:
    """Return a table with ``origin_csv`` appended when it is missing."""

    if "origin_csv" in table.column_names:
        return table
    origin_values = pa.array([origin_csv] * table.num_rows, type=pa.string())
    return table.append_column("origin_csv", origin_values)


def _normalize_point_selection_keys(
    selections: Iterable[tuple[str, str | None]],
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for label, origin_csv in selections:
        label_text = str(label).strip()
        if not label_text:
            continue
        normalized_key = (label_text, _normalize_origin_csv_value(origin_csv))
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        normalized.append(normalized_key)
    return normalized


def _normalize_bltr_header_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.startswith("Unnamed:"):
        return ""
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()


def _flatten_bltr_columns(columns: pd.Index) -> list[str]:
    """Flatten the observed BLTR two-row header into canonical column names."""

    current_group = ""
    flattened: list[str] = []
    for top_level, sub_level in columns.to_flat_index():
        top_token = _normalize_bltr_header_token(top_level)
        sub_token = _normalize_bltr_header_token(sub_level)
        if top_token:
            current_group = top_token
        group = current_group

        if group == "marker":
            name = "marker"
        elif group == "experiment":
            name = "strength" if sub_token == "strength" else f"experiment_{sub_token}"
        elif group == "channels":
            name = sub_token
        elif group == "atlas":
            name = f"atlas_{sub_token}"
        elif group == "region":
            name = f"region_{sub_token}"
        elif group:
            name = f"{group}_{sub_token}" if sub_token else group
        else:
            name = sub_token

        if not name:
            raise PointImportError("Failed to flatten BLTR CSV header.")
        flattened.append(name)

    return flattened


def load_bltr_point_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a BLTR-format two-row-header CSV into canonical source columns."""

    path = Path(csv_path)
    try:
        raw_df = pd.read_csv(path, header=[0, 1], low_memory=False)
    except FileNotFoundError as exc:
        raise PointImportError(_path_exists_message(path, "CSV file")) from exc
    except Exception as exc:
        raise PointImportError(f"Failed to read BLTR CSV file: {path}") from exc

    flattened_columns = _flatten_bltr_columns(raw_df.columns)
    raw_df.columns = flattened_columns

    missing_columns = sorted(
        set(BLTR_STANDARD_MAPPING.values()) - set(flattened_columns)
    )
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise PointImportError(f"BLTR CSV is missing expected column(s): {columns}")

    missing_extra_columns = sorted(set(BLTR_EXTRA_COLUMNS) - set(flattened_columns))
    if missing_extra_columns:
        columns = ", ".join(missing_extra_columns)
        raise PointImportError(f"BLTR CSV is missing expected BLTR column(s): {columns}")

    return raw_df


def load_column_mapping(mapping_path: str | Path) -> dict[str, str]:
    """Load and validate a target-to-source JSON column mapping."""

    path = Path(mapping_path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise PointImportError(f"Mapping file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PointImportError(f"Invalid JSON in mapping file: {path}") from exc

    if not isinstance(data, dict):
        raise PointImportError("Mapping file must contain a single JSON object.")

    unknown_targets = sorted(set(data) - set(STANDARD_POINT_COLUMNS))
    if unknown_targets:
        targets = ", ".join(unknown_targets)
        raise PointImportError(f"Unknown mapping target column(s): {targets}")

    mapping: dict[str, str] = {}
    for target, source in data.items():
        if not isinstance(source, str) or not source.strip():
            raise PointImportError(
                f"Mapping for '{target}' must be a non-empty source column name."
            )
        mapping[str(target)] = source.strip()

    missing_required = [
        column for column in REQUIRED_POINT_COLUMNS if column not in mapping
    ]
    if missing_required:
        columns = ", ".join(missing_required)
        raise PointImportError(f"Missing required mapping(s): {columns}")

    duplicate_sources = sorted(
        source
        for source in set(mapping.values())
        if list(mapping.values()).count(source) > 1
    )
    if duplicate_sources:
        sources = ", ".join(duplicate_sources)
        raise PointImportError(
            f"Source columns cannot be mapped more than once: {sources}"
        )

    return mapping


def validate_standard_point_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a standardized point dataframe."""

    missing_required = [
        column for column in REQUIRED_POINT_COLUMNS if column not in df.columns
    ]
    if missing_required:
        columns = ", ".join(missing_required)
        raise PointImportError(f"Missing required point column(s): {columns}")

    normalized = df.copy()
    extras = [
        column for column in normalized.columns if column not in STANDARD_POINT_COLUMNS
    ]

    labels = _normalize_string_series(normalized["label"]).str.strip()
    invalid_labels = labels.isna() | labels.eq("")
    if invalid_labels.any():
        count = int(invalid_labels.sum())
        raise PointImportError(f"Column 'label' has {count} empty value(s).")
    normalized["label"] = labels

    for column in ("x", "y", "z"):
        values = pd.to_numeric(normalized[column], errors="coerce")
        invalid = values.isna()
        if invalid.any():
            count = int(invalid.sum())
            raise PointImportError(f"Column '{column}' has {count} invalid value(s).")
        normalized[column] = values.astype(float)

    for column in _STRING_OPTIONAL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = _empty_string_series(len(normalized))
        else:
            normalized[column] = _normalize_string_series(normalized[column])

    if "id" not in normalized.columns:
        normalized["id"] = _empty_int_series(len(normalized))
    else:
        id_series = normalized["id"]
        if pd.api.types.is_string_dtype(id_series) or id_series.dtype == object:
            blanks = _normalize_string_series(id_series).str.strip().eq("")
            id_series = id_series.where(~blanks, pd.NA)
        values = pd.to_numeric(id_series, errors="coerce")
        invalid = pd.Series(False, index=normalized.index)
        invalid |= values.isna() & id_series.notna()
        if invalid.any():
            count = int(invalid.sum())
            raise PointImportError(f"Column 'id' has {count} invalid value(s).")
        normalized["id"] = values.astype("Int64")

    ordered_columns = [*STANDARD_POINT_COLUMNS, *extras]
    return normalized[ordered_columns]


def standardize_point_dataframe(
    raw_df: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    """Apply a target-to-source mapping to produce a standardized dataframe."""

    missing_sources = sorted(set(mapping.values()) - set(raw_df.columns))
    if missing_sources:
        columns = ", ".join(missing_sources)
        raise PointImportError(f"Mapped source column(s) not found in CSV: {columns}")

    mapped_sources = set(mapping.values())
    conflicting_extras = sorted(
        column
        for column in raw_df.columns
        if column not in mapped_sources and column in STANDARD_POINT_COLUMNS
    )
    if conflicting_extras:
        columns = ", ".join(conflicting_extras)
        raise PointImportError(
            "Unmapped source columns conflict with standardized column names: "
            f"{columns}"
        )

    standardized = pd.DataFrame(index=raw_df.index)
    for column in REQUIRED_POINT_COLUMNS:
        standardized[column] = raw_df[mapping[column]]

    for column in OPTIONAL_POINT_COLUMNS:
        if column in mapping:
            standardized[column] = raw_df[mapping[column]]
        elif column == "id":
            standardized[column] = _empty_int_series(len(raw_df))
        else:
            standardized[column] = _empty_string_series(len(raw_df))

    for column in raw_df.columns:
        if column not in mapped_sources:
            standardized[column] = raw_df[column]

    return validate_standard_point_dataframe(standardized)


def load_raw_point_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a raw point CSV file."""

    path = Path(csv_path)
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError as exc:
        raise PointImportError(_path_exists_message(path, "CSV file")) from exc
    except Exception as exc:
        raise PointImportError(f"Failed to read CSV file: {path}") from exc


def load_and_standardize_point_csv(
    csv_path: str | Path,
    mapping_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load and standardize a point CSV via mapping or known header formats."""

    if mapping_path is not None:
        raw_df = load_raw_point_csv(csv_path)
        mapping = load_column_mapping(mapping_path)
        return standardize_point_dataframe(raw_df, mapping), "mapping JSON"

    try:
        raw_df = load_raw_point_csv(csv_path)
        return validate_standard_point_dataframe(raw_df), "standardized headers"
    except PointImportError:
        pass

    try:
        raw_df = load_bltr_point_csv(csv_path)
        return (
            standardize_point_dataframe(raw_df, BLTR_STANDARD_MAPPING),
            "BLTR headers",
        )
    except PointImportError as exc:
        raise PointImportError(
            "Could not infer point CSV columns from headers. "
            "Provide a mapping JSON."
        ) from exc


def convert_point_csv_to_parquet(
    csv_path: str | Path,
    mapping_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Convert a raw point CSV plus mapping JSON into standardized Parquet."""

    standardized, _source = load_and_standardize_point_csv(csv_path, mapping_path)
    standardized = _ensure_origin_csv_provenance(standardized, Path(csv_path).name)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    standardized.to_parquet(output_path, index=False)
    return standardized


def _load_point_csv_for_batch_conversion(
    csv_path: str | Path,
    mapping_path: str | Path | None,
) -> pd.DataFrame:
    """Load one point CSV for batch conversion, falling back to mapping if needed."""

    try:
        standardized, _source = load_and_standardize_point_csv(csv_path)
        return standardized
    except PointImportError:
        if mapping_path is None:
            raise
    standardized, _source = load_and_standardize_point_csv(csv_path, mapping_path)
    return standardized


def convert_point_csv_files_to_parquet(
    csv_paths: Sequence[str | Path],
    output_path: str | Path,
    mapping_path: str | Path | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> BatchPointParquetConversionSummary:
    """Convert one or more point CSV files into a standardized Parquet."""

    normalized_paths = [Path(path) for path in csv_paths]
    if not normalized_paths:
        raise PointImportError("No point CSV files were provided.")

    summary = BatchPointParquetConversionSummary(discovered_files=len(normalized_paths))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    try:
        for current, csv_path in enumerate(normalized_paths, start=1):
            if progress_callback is not None:
                progress_callback(
                    f"Processing point CSV {current}/{len(normalized_paths)}: {csv_path.name}",
                    current - 1,
                    len(normalized_paths),
                )

            standardized = _load_point_csv_for_batch_conversion(csv_path, mapping_path)
            standardized = _ensure_origin_csv_provenance(standardized, csv_path.name)

            if schema is None:
                table = pa.Table.from_pandas(
                    standardized,
                    preserve_index=False,
                )
                schema = table.schema
                writer = pq.ParquetWriter(
                    temp_output_path,
                    schema,
                    compression="snappy",
                )
            else:
                schema = _validate_append_schema_compatibility(standardized, schema)
                table = pa.Table.from_pandas(
                    standardized,
                    schema=schema,
                    preserve_index=False,
                )

            assert writer is not None
            writer.write_table(table.replace_schema_metadata(schema.metadata))
            summary.processed_files += 1
            summary.rows_written += len(standardized)
    except Exception:
        if writer is not None:
            writer.close()
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise

    if writer is None:
        raise PointImportError("No point CSV rows were written.")

    writer.close()
    temp_output_path.replace(output_path)
    if progress_callback is not None:
        progress_callback(
            f"Finalized point Parquet: {output_path.name}",
            len(normalized_paths),
            len(normalized_paths),
        )
    return summary


def append_point_csv_to_parquet(
    csv_path: str | Path,
    mapping_path: str | Path | None,
    parquet_path: str | Path,
    output_path: str | Path | None = None,
) -> PointParquetAppendSummary:
    """Append a raw point CSV into a point Parquet and write the result."""

    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    output_path = Path(output_path) if output_path is not None else parquet_path
    target_schema = _schema_with_origin_csv(_read_standard_point_parquet_schema(parquet_path))

    standardized, _source = load_and_standardize_point_csv(csv_path, mapping_path)
    standardized = _ensure_origin_csv_provenance(standardized, csv_path.name)
    target_schema = _validate_append_schema_compatibility(standardized, target_schema)
    existing_parquet = pq.ParquetFile(parquet_path)
    temp_output_path: Path | None = None
    try:
        append_table = pa.Table.from_pandas(
            standardized,
            schema=target_schema,
            preserve_index=False,
        )

        legacy_row_groups: list[pa.Table] = []
        for row_group_index in range(existing_parquet.num_row_groups):
            row_group_table = existing_parquet.read_row_group(row_group_index)
            legacy_row_groups.append(
                _table_with_origin_csv(
                    row_group_table,
                    POINT_PARQUET_ORIGIN_NOT_RECORDED,
                )
            )

        temp_output_path = _write_point_parquet_with_appended_tables(
            existing_tables=legacy_row_groups,
            append_tables=[append_table],
            output_path=output_path,
            target_schema=target_schema,
        )
        total_rows = int(existing_parquet.metadata.num_rows + len(standardized))
    finally:
        existing_parquet.close()

    if temp_output_path is None:
        raise PointImportError("Failed to write appended point Parquet.")

    temp_output_path.replace(output_path)
    return PointParquetAppendSummary(
        appended_rows=len(standardized),
        total_rows=total_rows,
    )


def append_point_parquet_to_parquet(
    input_parquet_path: str | Path,
    parquet_path: str | Path,
    output_path: str | Path | None = None,
) -> PointParquetAppendSummary:
    """Append one standardized point Parquet into another with exact schema matching."""

    input_parquet_path = Path(input_parquet_path)
    parquet_path = Path(parquet_path)
    if input_parquet_path.resolve() == parquet_path.resolve():
        raise PointImportError("Input point Parquet must differ from the target file.")

    output_path = Path(output_path) if output_path is not None else parquet_path
    target_schema = _read_standard_point_parquet_schema(parquet_path)
    input_schema = _read_standard_point_parquet_schema(input_parquet_path)
    target_schema = _validate_point_schema_compatibility(input_schema, target_schema)

    existing_parquet = pq.ParquetFile(parquet_path)
    input_parquet = pq.ParquetFile(input_parquet_path)
    temp_output_path: Path | None = None
    try:
        existing_tables = (
            existing_parquet.read_row_group(row_group_index)
            for row_group_index in range(existing_parquet.num_row_groups)
        )
        append_tables = [
            input_parquet.read_row_group(row_group_index)
            for row_group_index in range(input_parquet.num_row_groups)
        ]
        temp_output_path = _write_point_parquet_with_appended_tables(
            existing_tables=existing_tables,
            append_tables=append_tables,
            output_path=output_path,
            target_schema=target_schema,
        )
        appended_rows = int(input_parquet.metadata.num_rows)
        total_rows = int(existing_parquet.metadata.num_rows + input_parquet.metadata.num_rows)
    finally:
        input_parquet.close()
        existing_parquet.close()

    if temp_output_path is None:
        raise PointImportError("Failed to write appended point Parquet.")

    temp_output_path.replace(output_path)
    return PointParquetAppendSummary(
        appended_rows=appended_rows,
        total_rows=total_rows,
    )


def append_point_file_to_parquet(
    input_path: str | Path,
    parquet_path: str | Path,
    output_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> PointParquetAppendSummary:
    """Append a point CSV or point Parquet into an existing point Parquet."""

    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return append_point_csv_to_parquet(
            input_path,
            mapping_path,
            parquet_path,
            output_path,
        )
    if suffix == ".parquet":
        if mapping_path is not None:
            raise PointImportError("Mapping JSON is only supported when appending CSV input.")
        return append_point_parquet_to_parquet(
            input_path,
            parquet_path,
            output_path,
        )

    raise PointImportError(
        f"Unsupported point input file type: {input_path.suffix or '(none)'}"
    )


def convert_bltr_point_csv_directory_to_parquet(
    input_dir: str | Path,
    output_path: str | Path,
) -> BatchPointParquetConversionSummary:
    """Convert a directory of BLTR CSV files into one standardized Parquet."""

    input_path = Path(input_dir)
    if not input_path.exists():
        raise PointImportError(_path_exists_message(input_path, "Input directory"))
    if not input_path.is_dir():
        raise PointImportError(f"Input path must be a directory: {input_path}")

    csv_paths = sorted(path for path in input_path.glob("*.csv") if path.is_file())
    summary = BatchPointParquetConversionSummary(discovered_files=len(csv_paths))
    if not csv_paths:
        raise PointImportError(f"No BLTR CSV files found in directory: {input_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None

    try:
        for csv_path in csv_paths:
            raw_df = load_bltr_point_csv(csv_path)
            standardized = standardize_point_dataframe(raw_df, BLTR_STANDARD_MAPPING)
            standardized["origin_csv"] = csv_path.name

            table = pa.Table.from_pandas(
                standardized,
                schema=schema,
                preserve_index=False,
            )
            if schema is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    temp_output_path,
                    schema,
                    compression="snappy",
                )

            assert writer is not None
            writer.write_table(table)
            summary.processed_files += 1
            summary.rows_written += len(standardized)
    except Exception:
        if writer is not None:
            writer.close()
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise

    if writer is None:
        raise PointImportError("No BLTR CSV rows were written.")

    writer.close()
    temp_output_path.replace(output_path)
    return summary


def summarize_standard_point_parquet_groups(parquet_path: str | Path) -> pd.DataFrame:
    """Return a grouped preview of labels, origin CSV, and point counts."""

    path = Path(parquet_path)
    has_origin_csv = point_parquet_has_origin_csv(path)
    conn = duckdb.connect()
    try:
        path_str = _escape_duckdb_path(path)
        if has_origin_csv:
            query = f"""
                SELECT
                    TRIM(CAST(label AS VARCHAR)) AS label,
                    CASE
                        WHEN TRIM(COALESCE(CAST(origin_csv AS VARCHAR), '')) = ''
                            THEN ?
                        ELSE CAST(origin_csv AS VARCHAR)
                    END AS origin_csv,
                    COUNT(*) AS point_count
                FROM read_parquet('{path_str}')
                GROUP BY 1, 2
                ORDER BY origin_csv, label
            """
            summary = conn.execute(
                query,
                [POINT_PARQUET_ORIGIN_NOT_RECORDED],
            ).fetchdf()
        else:
            query = f"""
                SELECT
                    TRIM(CAST(label AS VARCHAR)) AS label,
                    ? AS origin_csv,
                    COUNT(*) AS point_count
                FROM read_parquet('{path_str}')
                GROUP BY 1
                ORDER BY origin_csv, label
            """
            summary = conn.execute(
                query,
                [POINT_PARQUET_ORIGIN_NOT_RECORDED],
            ).fetchdf()
    except duckdb.Error as exc:
        raise PointImportError(f"Failed to summarize Point Parquet file: {path}") from exc
    finally:
        conn.close()

    if summary.empty:
        empty = pd.DataFrame(columns=["label", "origin_csv", "point_count"])
        empty.attrs["has_origin_csv"] = has_origin_csv
        return empty

    summary["label"] = _normalize_string_series(summary["label"]).str.strip()
    summary["origin_csv"] = (
        _normalize_string_series(summary["origin_csv"])
        .str.strip()
        .fillna(POINT_PARQUET_ORIGIN_NOT_RECORDED)
    )
    summary["point_count"] = (
        pd.to_numeric(summary["point_count"], errors="raise").astype(int)
    )
    summary = summary[["label", "origin_csv", "point_count"]]
    summary.attrs["has_origin_csv"] = has_origin_csv
    return summary


def load_standard_point_parquet_selection(
    parquet_path: str | Path,
    selections: Sequence[tuple[str, str | None]],
) -> pd.DataFrame:
    """Load only the selected label/origin rows from a point Parquet file."""

    path = Path(parquet_path)
    has_origin_csv = point_parquet_has_origin_csv(path)
    normalized_selections = _normalize_point_selection_keys(selections)
    if not normalized_selections:
        columns = [*STANDARD_POINT_COLUMNS]
        if has_origin_csv:
            columns.append("origin_csv")
        return validate_standard_point_dataframe(pd.DataFrame(columns=columns))

    conn = duckdb.connect()
    try:
        path_str = _escape_duckdb_path(path)
        params: list[str] = []
        if has_origin_csv:
            clauses: list[str] = []
            for label, origin_csv in normalized_selections:
                if origin_csv == POINT_PARQUET_ORIGIN_NOT_RECORDED:
                    clauses.append(
                        "("
                        "TRIM(CAST(label AS VARCHAR)) = ? AND "
                        "TRIM(COALESCE(CAST(origin_csv AS VARCHAR), '')) = ''"
                        ")"
                    )
                    params.append(label)
                else:
                    clauses.append(
                        "("
                        "TRIM(CAST(label AS VARCHAR)) = ? AND "
                        "CAST(origin_csv AS VARCHAR) = ?"
                        ")"
                    )
                    params.extend([label, origin_csv])

            query = (
                f"SELECT * FROM read_parquet('{path_str}') WHERE "
                + " OR ".join(clauses)
            )
        else:
            labels = [label for label, _origin_csv in normalized_selections]
            placeholders = ", ".join(["?"] * len(labels))
            params = labels
            query = (
                f"SELECT * FROM read_parquet('{path_str}') "
                f"WHERE TRIM(CAST(label AS VARCHAR)) IN ({placeholders})"
            )

        df = conn.execute(query, params).fetchdf()
    except duckdb.Error as exc:
        raise PointImportError(f"Failed to query Point Parquet file: {path}") from exc
    finally:
        conn.close()

    return validate_standard_point_dataframe(df)


def load_standard_point_parquet(parquet_path: str | Path) -> pd.DataFrame:
    """Load and validate a standardized point Parquet file."""

    path = Path(parquet_path)
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise PointImportError(_path_exists_message(path, "Point Parquet file")) from exc
    except Exception as exc:
        raise PointImportError(f"Failed to read Point Parquet file: {path}") from exc
    return validate_standard_point_dataframe(df)


def _build_region_lookup(atlas: Any) -> dict[int, dict[str, Any]]:
    """Build a simple region metadata lookup from a BrainGlobe atlas."""

    lookup: dict[int, dict[str, Any]] = {}
    structures = getattr(atlas, "structures", {})
    for key, structure in structures.items():
        if not isinstance(key, int):
            continue
        try:
            name = structure["name"]
        except Exception:
            name = getattr(structure, "name", "")
        try:
            acronym = structure["acronym"]
        except Exception:
            acronym = getattr(structure, "acronym", "")
        lookup[int(key)] = {
            "id": int(key),
            "name": str(name),
            "acronym": str(acronym),
        }
    return lookup


def _world_coords_to_atlas_region_ids(
    coords_xyz: np.ndarray,
    atlas: Any,
) -> np.ndarray:
    """Map world-space XYZ micron coordinates to atlas annotation region IDs."""

    annotation = np.asarray(atlas.annotation)
    voxel_coords = world_coords_xyz_to_atlas_voxels(coords_xyz, atlas)

    region_ids = np.zeros(len(voxel_coords), dtype=np.int64)
    in_bounds = np.all(
        (voxel_coords >= 0) & (voxel_coords < np.asarray(annotation.shape)),
        axis=1,
    )
    valid = voxel_coords[in_bounds]
    if len(valid) > 0:
        region_ids[in_bounds] = annotation[valid[:, 0], valid[:, 1], valid[:, 2]]
    return region_ids


def validate_point_metadata_against_atlas(
    df: pd.DataFrame,
    atlas: Any,
) -> AtlasValidationSummary:
    """Validate optional point metadata columns against atlas-derived values."""

    standardized = validate_standard_point_dataframe(df)
    coords = standardized[["x", "y", "z"]].to_numpy(dtype=float, copy=False)

    region_ids = _world_coords_to_atlas_region_ids(coords, atlas)
    lookup = _build_region_lookup(atlas)
    derived_region_name = pd.Series(
        [lookup.get(int(region_id), {}).get("name", "") for region_id in region_ids],
        dtype="string",
    )
    derived_acronym = pd.Series(
        [lookup.get(int(region_id), {}).get("acronym", "") for region_id in region_ids],
        dtype="string",
    )

    midline = get_atlas_midline(atlas, coord_axis=2)
    hemisphere_values = np.full(len(standardized), "midline", dtype=object)
    hemisphere_values[coords[:, 2] < midline - 1.0] = "left"
    hemisphere_values[coords[:, 2] > midline + 1.0] = "right"
    derived_hemisphere = pd.Series(hemisphere_values, dtype="string")

    compared_fields: list[str] = []
    mismatch_counts: dict[str, int] = {}
    row_mask = pd.Series(False, index=standardized.index)
    mismatch_fields: dict[int, list[str]] = {}

    def record_mismatches(field: str, mask: pd.Series) -> None:
        mismatch_counts[field] = int(mask.sum())
        if not mask.any():
            return
        for index in standardized.index[mask]:
            mismatch_fields.setdefault(int(index), []).append(field)

    if "id" in standardized.columns:
        supplied = standardized["id"].notna()
        if supplied.any():
            compared_fields.append("id")
            mismatch = supplied & (standardized["id"].astype("Int64") != region_ids)
            row_mask |= mismatch
            record_mismatches("id", mismatch)

    if "acronym" in standardized.columns:
        supplied = _nonempty_string_mask(standardized["acronym"])
        if supplied.any():
            compared_fields.append("acronym")
            mismatch = supplied & ~_compare_string_series(
                standardized["acronym"],
                derived_acronym,
            )
            row_mask |= mismatch
            record_mismatches("acronym", mismatch)

    if "region_name" in standardized.columns:
        supplied = _nonempty_string_mask(standardized["region_name"])
        if supplied.any():
            compared_fields.append("region_name")
            mismatch = supplied & ~_compare_region_name_series(
                standardized["region_name"],
                derived_region_name,
            )
            row_mask |= mismatch
            record_mismatches("region_name", mismatch)

    if "hemisphere" in standardized.columns:
        supplied = _nonempty_string_mask(standardized["hemisphere"])
        if supplied.any():
            compared_fields.append("hemisphere")
            mismatch = supplied & ~_compare_hemisphere_series(
                standardized["hemisphere"],
                derived_hemisphere,
            )
            row_mask |= mismatch
            record_mismatches("hemisphere", mismatch)

    mismatch_df = standardized.loc[row_mask].copy()
    mismatch_df["atlas_region_name"] = derived_region_name[row_mask].to_numpy()
    mismatch_df["atlas_acronym"] = derived_acronym[row_mask].to_numpy()
    mismatch_df["atlas_id"] = region_ids[row_mask.to_numpy()]
    mismatch_df["atlas_hemisphere"] = derived_hemisphere[row_mask].to_numpy()
    mismatch_df["mismatch_fields"] = [
        ",".join(mismatch_fields[int(index)])
        for index in mismatch_df.index
    ]

    return AtlasValidationSummary(
        total_points=len(standardized),
        checked_fields=tuple(compared_fields),
        mismatch_counts=mismatch_counts,
        mismatches=mismatch_df,
    )


def format_atlas_validation_summary(
    summary: AtlasValidationSummary,
    max_examples: int = 5,
) -> str:
    """Format a concise user-facing summary of atlas validation mismatches."""

    if not summary.has_mismatches:
        return (
            f"Atlas validation checked {summary.total_points} point(s) and found "
            "no mismatches."
        )

    count_bits = ", ".join(
        f"{field}: {count}"
        for field, count in summary.mismatch_counts.items()
        if count > 0
    )
    message = (
        f"Atlas validation found {summary.total_mismatched_rows} mismatched point(s) "
        f"out of {summary.total_points}"
    )
    if count_bits:
        message += f" ({count_bits})"
    message += "."

    examples: list[str] = []
    for index, row in summary.mismatches.head(max_examples).iterrows():
        label = row["label"]
        fields = row["mismatch_fields"]
        examples.append(f"row {int(index) + 1} label={label} [{fields}]")

    if examples:
        message += " Examples: " + "; ".join(examples)

    return message


def dataframe_to_point_properties(df: pd.DataFrame) -> dict[str, list[Any]]:
    """Convert a point dataframe into napari point properties."""

    properties: dict[str, list[Any]] = {}
    for column in df.columns:
        if column in {"x", "y", "z"}:
            continue
        series = df[column]
        values = series.astype(object).where(series.notna(), None).tolist()
        properties[column] = values
    return properties


def _normalized_groupby_series(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column]
    if column == "origin_csv":
        normalized = _normalize_string_series(series).str.strip()
        return normalized.map(_normalize_origin_csv_value).astype("string")
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return _normalize_string_series(series).str.strip()
    return series


def build_grouped_point_heatmap_volumes(
    df: pd.DataFrame,
    atlas: Any,
    group_columns: Sequence[str],
) -> dict[tuple[Any, ...], np.ndarray]:
    """Build one dense atlas-space count volume per requested group key."""

    if not group_columns:
        raise ValueError("group_columns must contain at least one column.")

    standardized = validate_standard_point_dataframe(df)
    missing_group_columns = [
        column for column in group_columns if column not in standardized.columns
    ]
    if missing_group_columns:
        columns = ", ".join(missing_group_columns)
        raise PointImportError(f"Grouping column(s) not found in dataframe: {columns}")

    coords = standardized[["x", "y", "z"]].to_numpy(dtype=float, copy=False)
    voxel_coords = world_coords_xyz_to_atlas_voxels(coords, atlas)
    atlas_shape = np.asarray(atlas.annotation.shape)
    in_bounds = np.all((voxel_coords >= 0) & (voxel_coords < atlas_shape), axis=1)

    group_frame = pd.DataFrame(index=standardized.index)
    for column in group_columns:
        group_frame[column] = _normalized_groupby_series(standardized, column)

    volumes: dict[tuple[Any, ...], np.ndarray] = {}
    for raw_key, positions in group_frame.groupby(list(group_columns), sort=False).indices.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        volume = np.zeros(atlas.annotation.shape, dtype=np.float32)
        group_positions = np.asarray(positions, dtype=int)
        valid_positions = group_positions[in_bounds[group_positions]]
        if len(valid_positions) > 0:
            group_voxels = voxel_coords[valid_positions]
            np.add.at(
                volume,
                (group_voxels[:, 0], group_voxels[:, 1], group_voxels[:, 2]),
                1.0,
            )
        volumes[key] = volume

    return volumes


def build_label_heatmap_volumes(
    df: pd.DataFrame,
    atlas: Any,
) -> dict[str, np.ndarray]:
    """Build one dense atlas-space count volume per label."""
    grouped = build_grouped_point_heatmap_volumes(df, atlas, ("label",))
    return {str(key[0]): volume for key, volume in grouped.items()}
