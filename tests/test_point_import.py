from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import napari_neuron_navigator.point_import as point_import
from napari_neuron_navigator.point_import import (
    append_point_file_to_parquet,
    append_point_parquet_to_parquet,
    append_point_csv_to_parquet,
    build_grouped_point_heatmap_volumes,
    build_label_heatmap_volumes,
    convert_bltr_point_csv_directory_to_parquet,
    convert_point_csv_files_to_parquet,
    PointParquetAppendSummary,
    PointImportError,
    POINT_PARQUET_ORIGIN_NOT_RECORDED,
    convert_point_csv_to_parquet,
    format_atlas_validation_summary,
    load_and_standardize_point_csv,
    load_bltr_point_csv,
    load_column_mapping,
    load_raw_point_csv,
    load_standard_point_parquet,
    load_standard_point_parquet_selection,
    summarize_standard_point_parquet_groups,
    standardize_point_dataframe,
    validate_point_metadata_against_atlas,
)


class FakeAtlas:
    def __init__(self) -> None:
        self.annotation = np.zeros((5, 5, 5), dtype=np.int32)
        self.annotation[4, 3, 2] = 101
        self.annotation[2, 2, 1] = 202
        self.resolution = (25.0, 25.0, 25.0)
        self.shape = self.annotation.shape
        self.atlas_name = "fake_atlas"
        self.structures = {
            101: {"name": "Region, One", "acronym": "R1"},
            202: {"name": "Region Two", "acronym": "R2"},
        }


@pytest.fixture
def fake_atlas() -> FakeAtlas:
    return FakeAtlas()


def _write_bltr_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a BLTR-format CSV fixture with the observed two-row header shape."""

    columns = pd.MultiIndex.from_tuples(
        [
            ("Marker", ""),
            ("Experiment", "x"),
            ("", "y"),
            ("", "z"),
            ("", "strength"),
            ("Channels", "channel Mono Channel"),
            ("Atlas", "x"),
            ("", "y"),
            ("", "z"),
            ("Region", "name"),
            ("", "acronym"),
            ("", "id"),
            ("", "hemisphere"),
        ]
    )
    frame = pd.DataFrame(
        [
            [
                row["marker"],
                row["experiment_x"],
                row["experiment_y"],
                row["experiment_z"],
                row["strength"],
                row["channel_mono_channel"],
                row["atlas_x"],
                row["atlas_y"],
                row["atlas_z"],
                row["region_name"],
                row["region_acronym"],
                row["region_id"],
                row["region_hemisphere"],
            ]
            for row in rows
        ],
        columns=columns,
    )
    frame.to_csv(path, index=False)


def _write_point_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_point_mapping(
    path: Path,
    *,
    extra_mappings: dict[str, str] | None = None,
) -> None:
    mapping = {
        "label": "marker",
        "x": "atlas_x",
        "y": "atlas_y",
        "z": "atlas_z",
    }
    if extra_mappings is not None:
        mapping.update(extra_mappings)
    path.write_text(json.dumps(mapping))


def _typed_optional_point_columns() -> dict[str, pd.Series]:
    return {
        "region_name": pd.Series([pd.NA], dtype="string"),
        "acronym": pd.Series([pd.NA], dtype="string"),
        "id": pd.Series([pd.NA], dtype="Int64"),
        "hemisphere": pd.Series([pd.NA], dtype="string"),
    }


class _DummyEvent:
    def connect(self, *_args, **_kwargs) -> None:
        return None

    def disconnect(self, *_args, **_kwargs) -> None:
        return None


class _DummyDims:
    def __init__(self) -> None:
        self.ndisplay = 3
        self.not_displayed = (0,)
        self.point = (0.0, 0.0, 0.0)
        self.order = (0, 1, 2)
        self.events = types.SimpleNamespace(
            ndisplay=_DummyEvent(),
            order=_DummyEvent(),
            current_step=_DummyEvent(),
        )


class _DummyPointsLayer:
    def __init__(self, data: np.ndarray, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = kwargs["name"]
        self.scale = np.asarray(kwargs.get("scale"))
        self.properties = {
            key: np.asarray(value, dtype=object)
            for key, value in kwargs.get("properties", {}).items()
        }
        self.metadata = kwargs.get("metadata", {})


class _DummyImageLayer:
    def __init__(self, data: np.ndarray, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = kwargs["name"]
        self.opacity = kwargs.get("opacity")
        self.metadata = kwargs.get("metadata", {})


class _DummyLabelsLayer:
    def __init__(self, data: np.ndarray, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = kwargs["name"]
        self.opacity = kwargs.get("opacity")
        self.metadata = kwargs.get("metadata", {})


class _DummyViewer:
    def __init__(self) -> None:
        self.layers: list[_DummyPointsLayer] = []
        self.dims = _DummyDims()

    def add_points(self, data: np.ndarray, **kwargs) -> _DummyPointsLayer:
        layer = _DummyPointsLayer(data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_image(self, data: np.ndarray, **kwargs) -> _DummyImageLayer:
        layer = _DummyImageLayer(data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_labels(self, data: np.ndarray, **kwargs) -> _DummyLabelsLayer:
        layer = _DummyLabelsLayer(data, **kwargs)
        self.layers.append(layer)
        return layer


def _import_neuron_viewer_widget():
    import importlib

    from qtpy.QtWidgets import QApplication, QWidget

    app = QApplication.instance()
    if app is None:
        QApplication([])

    class _DummySignal:
        def connect(self, *_args, **_kwargs) -> None:
            return None

    class _FakeAnalysisTabWidget(QWidget):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            self.cluster_colors_updated = _DummySignal()

        def set_slice_projector(self, *_args, **_kwargs) -> None:
            return None

        def set_database(self, *_args, **_kwargs) -> None:
            return None

        def set_atlas(self, *_args, **_kwargs) -> None:
            return None

        def apply_cluster_colors(self) -> None:
            return None

    fake_analysis_module = types.ModuleType(
        "napari_neuron_navigator.widgets.analysis_tab"
    )
    fake_analysis_module.AnalysisTabWidget = _FakeAnalysisTabWidget
    sys.modules["napari_neuron_navigator.widgets.analysis_tab"] = fake_analysis_module
    sys.modules.pop("napari_neuron_navigator.widgets.neuron_viewer", None)
    sys.modules.pop("napari_neuron_navigator.widgets", None)

    module = importlib.import_module("napari_neuron_navigator.widgets.neuron_viewer")
    return module.NeuronViewerWidget


def test_load_column_mapping_requires_required_targets(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({"label": "marker", "x": "atlas_x"}))

    with pytest.raises(PointImportError, match="Missing required mapping"):
        load_column_mapping(mapping_path)


def test_load_raw_point_csv_disables_low_memory_chunked_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "points.csv"
    csv_path.write_text("label,x,y,z\nA,1,2,3\n")
    calls: dict[str, object] = {}

    def fake_read_csv(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return pd.DataFrame({"label": ["A"], "x": [1], "y": [2], "z": [3]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    loaded = load_raw_point_csv(csv_path)

    assert Path(calls["path"]) == csv_path
    assert calls["kwargs"]["low_memory"] is False
    assert loaded["label"].tolist() == ["A"]


def test_convert_point_csv_to_parquet_preserves_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "points.csv"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "marker": ["A", "B"],
            "atlas_x": [50.0, 25.0],
            "atlas_y": [75.0, 50.0],
            "atlas_z": [100.0, 50.0],
            "score": [0.5, 0.9],
            "region_acronym": ["R1", "R2"],
        }
    ).to_csv(csv_path, index=False)
    mapping_path.write_text(
        json.dumps(
            {
                "label": "marker",
                "x": "atlas_x",
                "y": "atlas_y",
                "z": "atlas_z",
                "acronym": "region_acronym",
            }
        )
    )

    standardized = convert_point_csv_to_parquet(csv_path, mapping_path, output_path)
    loaded = pd.read_parquet(output_path)

    assert list(standardized.columns) == [
        "label",
        "x",
        "y",
        "z",
        "region_name",
        "acronym",
        "id",
        "hemisphere",
        "score",
        "origin_csv",
    ]
    assert list(loaded.columns) == list(standardized.columns)
    assert loaded["score"].tolist() == [0.5, 0.9]
    assert loaded["acronym"].tolist() == ["R1", "R2"]
    assert loaded["region_name"].isna().all()
    assert loaded["origin_csv"].tolist() == ["points.csv", "points.csv"]


def test_convert_point_csv_to_parquet_rejects_invalid_coordinates(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "points.csv"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "marker": ["A"],
            "atlas_x": ["bad"],
            "atlas_y": [75.0],
            "atlas_z": [100.0],
        }
    ).to_csv(csv_path, index=False)
    mapping_path.write_text(
        json.dumps(
            {"label": "marker", "x": "atlas_x", "y": "atlas_y", "z": "atlas_z"}
        )
    )

    with pytest.raises(PointImportError, match="Column 'x' has 1 invalid value"):
        convert_point_csv_to_parquet(csv_path, mapping_path, output_path)


def test_load_and_standardize_point_csv_accepts_standardized_headers(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "points.csv"
    pd.DataFrame(
        {
            "label": ["A"],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            "score": [0.5],
        }
    ).to_csv(csv_path, index=False)

    standardized, source_description = load_and_standardize_point_csv(csv_path)

    assert source_description == "standardized headers"
    assert standardized["label"].tolist() == ["A"]
    assert standardized["score"].tolist() == [0.5]


def test_load_and_standardize_point_csv_accepts_bltr_headers(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "bltr.csv"
    _write_bltr_csv(
        csv_path,
        [
            {
                "marker": "A",
                "experiment_x": 1.0,
                "experiment_y": 2.0,
                "experiment_z": 3.0,
                "strength": 0.5,
                "channel_mono_channel": 7,
                "atlas_x": 50.0,
                "atlas_y": 75.0,
                "atlas_z": 100.0,
                "region_name": "Region One",
                "region_acronym": "R1",
                "region_id": 101,
                "region_hemisphere": "right",
            }
        ],
    )

    standardized, source_description = load_and_standardize_point_csv(csv_path)

    assert source_description == "BLTR headers"
    assert standardized["label"].tolist() == ["A"]
    assert standardized["acronym"].tolist() == ["R1"]


def test_append_point_csv_to_parquet_accepts_standardized_headers_without_mapping(
    tmp_path: Path,
) -> None:
    existing_csv_path = tmp_path / "existing.csv"
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    _write_point_csv(
        existing_csv_path,
        [
            {
                "marker": "A",
                "atlas_x": 10.0,
                "atlas_y": 20.0,
                "atlas_z": 30.0,
            }
        ],
    )
    _write_point_mapping(mapping_path)
    convert_point_csv_to_parquet(existing_csv_path, mapping_path, parquet_path)

    pd.DataFrame(
        {
            "label": ["B"],
            "x": [40.0],
            "y": [50.0],
            "z": [60.0],
        }
    ).to_csv(append_csv_path, index=False)

    summary = append_point_csv_to_parquet(
        append_csv_path,
        None,
        parquet_path,
    )

    loaded = pd.read_parquet(parquet_path)

    assert summary.appended_rows == 1
    assert summary.total_rows == 2
    assert loaded["label"].tolist() == ["A", "B"]
    assert loaded["origin_csv"].tolist() == ["existing.csv", "append.csv"]


def test_append_point_csv_to_parquet_writes_new_output_file(
    tmp_path: Path,
) -> None:
    existing_csv_path = tmp_path / "existing.csv"
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"
    output_path = tmp_path / "points_combined.parquet"

    _write_point_csv(
        existing_csv_path,
        [
            {
                "marker": "A",
                "atlas_x": 10.0,
                "atlas_y": 20.0,
                "atlas_z": 30.0,
            }
        ],
    )
    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 40.0,
                "atlas_y": 50.0,
                "atlas_z": 60.0,
            }
        ],
    )
    _write_point_mapping(mapping_path)
    convert_point_csv_to_parquet(existing_csv_path, mapping_path, parquet_path)

    summary = append_point_csv_to_parquet(
        append_csv_path,
        mapping_path,
        parquet_path,
        output_path,
    )

    original_loaded = pd.read_parquet(parquet_path)
    combined_loaded = pd.read_parquet(output_path)

    assert summary.appended_rows == 1
    assert summary.total_rows == 2
    assert original_loaded["label"].tolist() == ["A"]
    assert combined_loaded["label"].tolist() == ["A", "B"]
    assert original_loaded["origin_csv"].tolist() == ["existing.csv"]
    assert combined_loaded["origin_csv"].tolist() == ["existing.csv", "append.csv"]


def test_append_point_csv_to_parquet_upgrades_legacy_parquet_with_origin_csv(
    tmp_path: Path,
) -> None:
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "label": ["A"],
            "x": [10.0],
            "y": [20.0],
            "z": [30.0],
            **_typed_optional_point_columns(),
        }
    ).to_parquet(parquet_path, index=False)
    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 40.0,
                "atlas_y": 50.0,
                "atlas_z": 60.0,
            }
        ],
    )
    _write_point_mapping(mapping_path)

    summary = append_point_csv_to_parquet(
        append_csv_path,
        mapping_path,
        parquet_path,
    )

    loaded = pd.read_parquet(parquet_path)

    assert summary.appended_rows == 1
    assert summary.total_rows == 2
    assert loaded["label"].tolist() == ["A", "B"]
    assert loaded["origin_csv"].tolist() == [
        POINT_PARQUET_ORIGIN_NOT_RECORDED,
        "append.csv",
    ]


def test_append_point_csv_to_parquet_accepts_large_string_origin_csv_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "label": ["A"],
            "x": [10.0],
            "y": [20.0],
            "z": [30.0],
            **_typed_optional_point_columns(),
        }
    ).to_parquet(parquet_path, index=False)
    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 40.0,
                "atlas_y": 50.0,
                "atlas_z": 60.0,
            }
        ],
    )
    _write_point_mapping(mapping_path)

    legacy_schema = pq.read_schema(parquet_path).remove_metadata()
    incoming_schema = pa.schema([*legacy_schema, pa.field("origin_csv", pa.large_string())])
    monkeypatch.setattr(
        point_import,
        "_dataframe_arrow_schema",
        lambda _df: incoming_schema,
    )

    summary = append_point_csv_to_parquet(
        append_csv_path,
        mapping_path,
        parquet_path,
    )

    loaded = pd.read_parquet(parquet_path)

    assert summary.appended_rows == 1
    assert summary.total_rows == 2
    assert loaded["label"].tolist() == ["A", "B"]
    assert loaded["origin_csv"].tolist() == [
        POINT_PARQUET_ORIGIN_NOT_RECORDED,
        "append.csv",
    ]


def test_append_point_csv_to_parquet_injects_origin_csv_when_present(
    tmp_path: Path,
) -> None:
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "label": pd.Series(["A"], dtype="string"),
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            **_typed_optional_point_columns(),
            "origin_csv": pd.Series(["existing.csv"], dtype="string"),
        }
    ).to_parquet(parquet_path, index=False)

    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 4.0,
                "atlas_y": 5.0,
                "atlas_z": 6.0,
            }
        ],
    )
    _write_point_mapping(mapping_path)

    summary = append_point_csv_to_parquet(
        append_csv_path,
        mapping_path,
        parquet_path,
    )
    loaded = pd.read_parquet(parquet_path)

    assert summary.appended_rows == 1
    assert summary.total_rows == 2
    assert loaded["origin_csv"].tolist() == ["existing.csv", "append.csv"]


def test_append_point_parquet_to_parquet_accepts_exact_matching_schema(
    tmp_path: Path,
) -> None:
    first_csv_path = tmp_path / "first.csv"
    second_csv_path = tmp_path / "second.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"
    input_parquet_path = tmp_path / "points_in.parquet"
    output_path = tmp_path / "points_out.parquet"

    _write_point_csv(
        first_csv_path,
        [
            {
                "marker": "A",
                "atlas_x": 1.0,
                "atlas_y": 2.0,
                "atlas_z": 3.0,
            }
        ],
    )
    _write_point_csv(
        second_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 4.0,
                "atlas_y": 5.0,
                "atlas_z": 6.0,
            }
        ],
    )
    _write_point_mapping(mapping_path)
    convert_point_csv_to_parquet(first_csv_path, mapping_path, parquet_path)
    convert_point_csv_to_parquet(second_csv_path, mapping_path, input_parquet_path)

    summary = append_point_parquet_to_parquet(
        input_parquet_path,
        parquet_path,
        output_path,
    )

    loaded = pd.read_parquet(output_path)

    assert summary.appended_rows == 1
    assert summary.total_rows == 2
    assert loaded["label"].tolist() == ["A", "B"]
    assert loaded["origin_csv"].tolist() == ["first.csv", "second.csv"]


def test_append_point_parquet_to_parquet_rejects_extra_columns(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "points.parquet"
    input_parquet_path = tmp_path / "points_in.parquet"

    pd.DataFrame(
        {
            "label": pd.Series(["A"], dtype="string"),
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            **_typed_optional_point_columns(),
            "origin_csv": pd.Series(["first.csv"], dtype="string"),
        }
    ).to_parquet(parquet_path, index=False)
    pd.DataFrame(
        {
            "label": pd.Series(["B"], dtype="string"),
            "x": [4.0],
            "y": [5.0],
            "z": [6.0],
            **_typed_optional_point_columns(),
            "origin_csv": pd.Series(["second.csv"], dtype="string"),
            "score": [0.9],
        }
    ).to_parquet(input_parquet_path, index=False)

    with pytest.raises(PointImportError, match="extra column\\(s\\): score"):
        append_point_parquet_to_parquet(input_parquet_path, parquet_path)


def test_append_point_parquet_to_parquet_casts_equivalent_utf8_types(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "points.parquet"
    input_parquet_path = tmp_path / "points_in.parquet"

    pq.write_table(
        pa.table(
            {
                "label": pa.array(["A"], type=pa.string()),
                "x": pa.array([1.0], type=pa.float64()),
                "y": pa.array([2.0], type=pa.float64()),
                "z": pa.array([3.0], type=pa.float64()),
                "region_name": pa.array([None], type=pa.string()),
                "acronym": pa.array([None], type=pa.string()),
                "id": pa.array([None], type=pa.int64()),
                "hemisphere": pa.array([None], type=pa.string()),
                "origin_csv": pa.array(["first.csv"], type=pa.string()),
            }
        ),
        parquet_path,
    )
    pq.write_table(
        pa.table(
            {
                "label": pa.array(["B"], type=pa.large_string()),
                "x": pa.array([4.0], type=pa.float64()),
                "y": pa.array([5.0], type=pa.float64()),
                "z": pa.array([6.0], type=pa.float64()),
                "region_name": pa.array([None], type=pa.large_string()),
                "acronym": pa.array([None], type=pa.large_string()),
                "id": pa.array([None], type=pa.int64()),
                "hemisphere": pa.array([None], type=pa.large_string()),
                "origin_csv": pa.array(["second.csv"], type=pa.large_string()),
            }
        ),
        input_parquet_path,
    )

    summary = append_point_parquet_to_parquet(input_parquet_path, parquet_path)

    loaded = pd.read_parquet(parquet_path)
    schema = pq.read_schema(parquet_path).remove_metadata()

    assert summary.appended_rows == 1
    assert summary.total_rows == 2
    assert loaded["label"].tolist() == ["A", "B"]
    assert loaded["origin_csv"].tolist() == ["first.csv", "second.csv"]
    assert schema.field("origin_csv").type == pa.string()


def test_append_point_csv_to_parquet_closes_source_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "label": pd.Series(["A"], dtype="string"),
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            **_typed_optional_point_columns(),
            "origin_csv": pd.Series(["existing.csv"], dtype="string"),
        }
    ).to_parquet(parquet_path, index=False)
    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 4.0,
                "atlas_y": 5.0,
                "atlas_z": 6.0,
            }
        ],
    )
    _write_point_mapping(mapping_path)

    original_parquet_file = point_import.pq.ParquetFile
    tracked_files: list[TrackingParquetFile] = []

    class TrackingParquetFile:
        def __init__(self, *args, **kwargs) -> None:
            self._delegate = original_parquet_file(*args, **kwargs)
            self.closed = False
            tracked_files.append(self)

        def close(self) -> None:
            self._delegate.close()
            self.closed = True

        def __getattr__(self, name: str) -> object:
            return getattr(self._delegate, name)

    original_replace = Path.replace

    def guarded_replace(self: Path, target: str | Path) -> Path:
        if any(not tracked_file.closed for tracked_file in tracked_files):
            raise PermissionError("simulated windows file lock")
        return original_replace(self, target)

    monkeypatch.setattr(point_import.pq, "ParquetFile", TrackingParquetFile)
    monkeypatch.setattr(Path, "replace", guarded_replace)

    summary = append_point_csv_to_parquet(
        append_csv_path,
        mapping_path,
        parquet_path,
    )

    loaded = pd.read_parquet(parquet_path)

    assert summary.appended_rows == 1
    assert loaded["origin_csv"].tolist() == ["existing.csv", "append.csv"]


def test_append_point_parquet_to_parquet_closes_sources_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_path = tmp_path / "points.parquet"
    input_parquet_path = tmp_path / "points_in.parquet"

    pd.DataFrame(
        {
            "label": pd.Series(["A"], dtype="string"),
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            **_typed_optional_point_columns(),
            "origin_csv": pd.Series(["first.csv"], dtype="string"),
        }
    ).to_parquet(parquet_path, index=False)
    pd.DataFrame(
        {
            "label": pd.Series(["B"], dtype="string"),
            "x": [4.0],
            "y": [5.0],
            "z": [6.0],
            **_typed_optional_point_columns(),
            "origin_csv": pd.Series(["second.csv"], dtype="string"),
        }
    ).to_parquet(input_parquet_path, index=False)

    original_parquet_file = point_import.pq.ParquetFile
    tracked_files: list[TrackingParquetFile] = []

    class TrackingParquetFile:
        def __init__(self, *args, **kwargs) -> None:
            self._delegate = original_parquet_file(*args, **kwargs)
            self.closed = False
            tracked_files.append(self)

        def close(self) -> None:
            self._delegate.close()
            self.closed = True

        def __getattr__(self, name: str) -> object:
            return getattr(self._delegate, name)

    original_replace = Path.replace

    def guarded_replace(self: Path, target: str | Path) -> Path:
        if any(not tracked_file.closed for tracked_file in tracked_files):
            raise PermissionError("simulated windows file lock")
        return original_replace(self, target)

    monkeypatch.setattr(point_import.pq, "ParquetFile", TrackingParquetFile)
    monkeypatch.setattr(Path, "replace", guarded_replace)

    summary = append_point_parquet_to_parquet(input_parquet_path, parquet_path)

    loaded = pd.read_parquet(parquet_path)

    assert summary.appended_rows == 1
    assert loaded["origin_csv"].tolist() == ["first.csv", "second.csv"]


def test_append_point_file_to_parquet_routes_parquet_input(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "points.parquet"
    input_parquet_path = tmp_path / "points_in.parquet"

    pd.DataFrame(
        {
            "label": pd.Series(["A"], dtype="string"),
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            **_typed_optional_point_columns(),
            "origin_csv": pd.Series(["first.csv"], dtype="string"),
        }
    ).to_parquet(parquet_path, index=False)
    pd.DataFrame(
        {
            "label": pd.Series(["B"], dtype="string"),
            "x": [4.0],
            "y": [5.0],
            "z": [6.0],
            **_typed_optional_point_columns(),
            "origin_csv": pd.Series(["second.csv"], dtype="string"),
        }
    ).to_parquet(input_parquet_path, index=False)

    summary = append_point_file_to_parquet(input_parquet_path, parquet_path)
    loaded = pd.read_parquet(parquet_path)

    assert summary.appended_rows == 1
    assert loaded["origin_csv"].tolist() == ["first.csv", "second.csv"]


def test_append_point_csv_to_parquet_rejects_extra_columns(
    tmp_path: Path,
) -> None:
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "label": ["A"],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            "region_name": [pd.NA],
            "acronym": [pd.NA],
            "id": [pd.NA],
            "hemisphere": [pd.NA],
        }
    ).to_parquet(parquet_path, index=False)

    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 4.0,
                "atlas_y": 5.0,
                "atlas_z": 6.0,
                "score": 0.9,
            }
        ],
    )
    _write_point_mapping(mapping_path)

    with pytest.raises(PointImportError, match="extra column\\(s\\): score"):
        append_point_csv_to_parquet(append_csv_path, mapping_path, parquet_path)


def test_append_point_csv_to_parquet_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "label": ["A"],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            "region_name": [pd.NA],
            "acronym": [pd.NA],
            "id": [pd.NA],
            "hemisphere": [pd.NA],
            "score": [0.1],
        }
    ).to_parquet(parquet_path, index=False)

    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 4.0,
                "atlas_y": 5.0,
                "atlas_z": 6.0,
            }
        ],
    )
    _write_point_mapping(mapping_path)

    with pytest.raises(PointImportError, match="missing column\\(s\\): score"):
        append_point_csv_to_parquet(append_csv_path, mapping_path, parquet_path)


def test_append_point_csv_to_parquet_rejects_type_mismatch(
    tmp_path: Path,
) -> None:
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "label": pd.Series(["A"], dtype="string"),
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            **_typed_optional_point_columns(),
            "score": pd.Series(["high"], dtype="string"),
        }
    ).to_parquet(parquet_path, index=False)

    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 4.0,
                "atlas_y": 5.0,
                "atlas_z": 6.0,
                "score": 0.9,
            }
        ],
    )
    _write_point_mapping(mapping_path)

    with pytest.raises(PointImportError, match="column 'score' has type double"):
        append_point_csv_to_parquet(append_csv_path, mapping_path, parquet_path)


def test_append_point_csv_to_parquet_keeps_original_file_on_failure(
    tmp_path: Path,
) -> None:
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    original = pd.DataFrame(
        {
            "label": pd.Series(["A"], dtype="string"),
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            **_typed_optional_point_columns(),
            "score": pd.Series(["high"], dtype="string"),
        }
    )
    original.to_parquet(parquet_path, index=False)

    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 4.0,
                "atlas_y": 5.0,
                "atlas_z": 6.0,
                "score": 0.9,
            }
        ],
    )
    _write_point_mapping(mapping_path)

    with pytest.raises(PointImportError):
        append_point_csv_to_parquet(append_csv_path, mapping_path, parquet_path)

    loaded = pd.read_parquet(parquet_path)
    assert loaded["label"].tolist() == ["A"]
    assert loaded["score"].tolist() == ["high"]
    assert len(loaded) == 1
    assert not parquet_path.with_suffix(".parquet.tmp").exists()


def test_convert_point_csv_files_to_parquet_combines_multiple_files_with_origin_csv(
    tmp_path: Path,
) -> None:
    first_csv_path = tmp_path / "one.csv"
    second_csv_path = tmp_path / "two.csv"
    output_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {"label": ["A"], "x": [1.0], "y": [2.0], "z": [3.0]}
    ).to_csv(first_csv_path, index=False)
    pd.DataFrame(
        {"label": ["B"], "x": [4.0], "y": [5.0], "z": [6.0]}
    ).to_csv(second_csv_path, index=False)

    summary = convert_point_csv_files_to_parquet(
        [first_csv_path, second_csv_path],
        output_path,
    )
    loaded = pd.read_parquet(output_path)

    assert summary.discovered_files == 2
    assert summary.processed_files == 2
    assert summary.rows_written == 2
    assert loaded["label"].tolist() == ["A", "B"]
    assert loaded["origin_csv"].tolist() == ["one.csv", "two.csv"]


def test_convert_point_csv_files_to_parquet_uses_mapping_for_raw_headers(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "points.csv"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "points.parquet"

    _write_point_csv(
        csv_path,
        [
            {
                "marker": "A",
                "atlas_x": 1.0,
                "atlas_y": 2.0,
                "atlas_z": 3.0,
            }
        ],
    )
    _write_point_mapping(mapping_path)

    summary = convert_point_csv_files_to_parquet(
        [csv_path],
        output_path,
        mapping_path,
    )
    loaded = pd.read_parquet(output_path)

    assert summary.discovered_files == 1
    assert summary.processed_files == 1
    assert summary.rows_written == 1
    assert loaded["label"].tolist() == ["A"]
    assert loaded["origin_csv"].tolist() == ["points.csv"]


def test_load_bltr_point_csv_flattens_expected_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bltr.csv"
    _write_bltr_csv(
        csv_path,
        [
            {
                "marker": "A",
                "experiment_x": 1.0,
                "experiment_y": 2.0,
                "experiment_z": 3.0,
                "strength": 0.5,
                "channel_mono_channel": 7,
                "atlas_x": 50.0,
                "atlas_y": 75.0,
                "atlas_z": 100.0,
                "region_name": "Region One",
                "region_acronym": "R1",
                "region_id": 101,
                "region_hemisphere": "right",
            }
        ],
    )

    loaded = load_bltr_point_csv(csv_path)

    assert list(loaded.columns) == [
        "marker",
        "experiment_x",
        "experiment_y",
        "experiment_z",
        "strength",
        "channel_mono_channel",
        "atlas_x",
        "atlas_y",
        "atlas_z",
        "region_name",
        "region_acronym",
        "region_id",
        "region_hemisphere",
    ]
    assert loaded.iloc[0]["marker"] == "A"
    assert float(loaded.iloc[0]["atlas_z"]) == 100.0


def test_convert_bltr_point_csv_directory_to_parquet_combines_files(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "bltr"
    input_dir.mkdir()
    output_path = tmp_path / "combined.parquet"

    _write_bltr_csv(
        input_dir / "a.csv",
        [
            {
                "marker": "A",
                "experiment_x": 1.0,
                "experiment_y": 2.0,
                "experiment_z": 3.0,
                "strength": 0.5,
                "channel_mono_channel": 7,
                "atlas_x": 50.0,
                "atlas_y": 75.0,
                "atlas_z": 100.0,
                "region_name": "Region One",
                "region_acronym": "R1",
                "region_id": 101,
                "region_hemisphere": "right",
            }
        ],
    )
    _write_bltr_csv(
        input_dir / "b.csv",
        [
            {
                "marker": "B",
                "experiment_x": 4.0,
                "experiment_y": 5.0,
                "experiment_z": 6.0,
                "strength": 0.9,
                "channel_mono_channel": 11,
                "atlas_x": 25.0,
                "atlas_y": 50.0,
                "atlas_z": 50.0,
                "region_name": "Region Two",
                "region_acronym": "R2",
                "region_id": 202,
                "region_hemisphere": "left",
            }
        ],
    )

    summary = convert_bltr_point_csv_directory_to_parquet(input_dir, output_path)
    loaded = pd.read_parquet(output_path)

    assert summary.discovered_files == 2
    assert summary.processed_files == 2
    assert summary.rows_written == 2
    assert list(loaded.columns) == [
        "label",
        "x",
        "y",
        "z",
        "region_name",
        "acronym",
        "id",
        "hemisphere",
        "experiment_x",
        "experiment_y",
        "experiment_z",
        "strength",
        "channel_mono_channel",
        "origin_csv",
    ]
    assert loaded["label"].tolist() == ["A", "B"]
    assert loaded["origin_csv"].tolist() == ["a.csv", "b.csv"]


def test_convert_bltr_point_csv_directory_to_parquet_removes_partial_output(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "bltr"
    input_dir.mkdir()
    output_path = tmp_path / "combined.parquet"

    _write_bltr_csv(
        input_dir / "a.csv",
        [
            {
                "marker": "A",
                "experiment_x": 1.0,
                "experiment_y": 2.0,
                "experiment_z": 3.0,
                "strength": 0.5,
                "channel_mono_channel": 7,
                "atlas_x": 50.0,
                "atlas_y": 75.0,
                "atlas_z": 100.0,
                "region_name": "Region One",
                "region_acronym": "R1",
                "region_id": 101,
                "region_hemisphere": "right",
            }
        ],
    )
    (input_dir / "b.csv").write_text("bad\n1\n")

    with pytest.raises(PointImportError):
        convert_bltr_point_csv_directory_to_parquet(input_dir, output_path)

    assert not output_path.exists()
    assert not output_path.with_suffix(".parquet.tmp").exists()


def test_summarize_standard_point_parquet_groups_with_origin_csv(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "points.parquet"
    pd.DataFrame(
        {
            "label": ["A", "A", "B", "A"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "z": [1.0, 2.0, 3.0, 4.0],
            "origin_csv": ["one.csv", "one.csv", "two.csv", "two.csv"],
        }
    ).to_parquet(parquet_path, index=False)

    summary = summarize_standard_point_parquet_groups(parquet_path)

    assert summary.attrs["has_origin_csv"] is True
    assert summary.to_dict("records") == [
        {"label": "A", "origin_csv": "one.csv", "point_count": 2},
        {"label": "A", "origin_csv": "two.csv", "point_count": 1},
        {"label": "B", "origin_csv": "two.csv", "point_count": 1},
    ]


def test_summarize_standard_point_parquet_groups_without_origin_csv(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "points.parquet"
    pd.DataFrame(
        {
            "label": ["A", "A", "B"],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "z": [1.0, 2.0, 3.0],
        }
    ).to_parquet(parquet_path, index=False)

    summary = summarize_standard_point_parquet_groups(parquet_path)

    assert summary.attrs["has_origin_csv"] is False
    assert summary.to_dict("records") == [
        {
            "label": "A",
            "origin_csv": POINT_PARQUET_ORIGIN_NOT_RECORDED,
            "point_count": 2,
        },
        {
            "label": "B",
            "origin_csv": POINT_PARQUET_ORIGIN_NOT_RECORDED,
            "point_count": 1,
        },
    ]


def test_load_standard_point_parquet_selection_filters_by_label_and_origin(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "points.parquet"
    pd.DataFrame(
        {
            "label": ["A", "A", "B"],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "z": [1.0, 2.0, 3.0],
            "origin_csv": ["one.csv", "two.csv", "two.csv"],
        }
    ).to_parquet(parquet_path, index=False)

    selected = load_standard_point_parquet_selection(
        parquet_path,
        [("A", "two.csv"), ("B", "two.csv")],
    )

    assert selected["label"].tolist() == ["A", "B"]
    assert selected["origin_csv"].tolist() == ["two.csv", "two.csv"]


def test_load_standard_point_parquet_selection_filters_by_label_without_origin(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "points.parquet"
    pd.DataFrame(
        {
            "label": ["A", "A", "B"],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "z": [1.0, 2.0, 3.0],
        }
    ).to_parquet(parquet_path, index=False)

    selected = load_standard_point_parquet_selection(
        parquet_path,
        [("B", POINT_PARQUET_ORIGIN_NOT_RECORDED)],
    )

    assert selected["label"].tolist() == ["B"]


def test_validate_point_metadata_against_atlas_uses_world_xyz_order(
    fake_atlas: FakeAtlas,
) -> None:
    df = pd.DataFrame(
        {
            "label": ["A"],
            "x": [50.0],
            "y": [75.0],
            "z": [100.0],
            "region_name": ["Region/One"],
            "acronym": ["R1"],
            "id": [101],
            "hemisphere": ["right"],
        }
    )

    summary = validate_point_metadata_against_atlas(df, fake_atlas)

    assert not summary.has_mismatches
    assert summary.checked_fields == ("id", "acronym", "region_name", "hemisphere")


def test_validate_point_metadata_against_atlas_reports_field_counts(
    fake_atlas: FakeAtlas,
) -> None:
    df = pd.DataFrame(
        {
            "label": ["A"],
            "x": [50.0],
            "y": [75.0],
            "z": [100.0],
            "region_name": ["Wrong Region"],
            "acronym": ["BAD"],
            "id": [999],
            "hemisphere": ["left"],
        }
    )

    summary = validate_point_metadata_against_atlas(df, fake_atlas)
    message = format_atlas_validation_summary(summary)

    assert summary.total_mismatched_rows == 1
    assert summary.mismatch_counts == {
        "id": 1,
        "acronym": 1,
        "region_name": 1,
        "hemisphere": 1,
    }
    assert "row 1 label=A" in message


def test_build_label_heatmap_volumes_groups_counts_by_label(
    fake_atlas: FakeAtlas,
) -> None:
    df = pd.DataFrame(
        {
            "label": ["A", "A", "B"],
            "x": [50.0, 50.0, 25.0],
            "y": [75.0, 75.0, 50.0],
            "z": [100.0, 100.0, 50.0],
        }
    )

    volumes = build_label_heatmap_volumes(df, fake_atlas)

    assert set(volumes) == {"A", "B"}
    assert volumes["A"].shape == fake_atlas.annotation.shape
    assert float(volumes["A"][4, 3, 2]) == 2.0
    assert float(volumes["B"][2, 2, 1]) == 1.0
    assert int((volumes["A"] > 0).sum()) == 1
    assert int((volumes["B"] > 0).sum()) == 1


def test_build_grouped_point_heatmap_volumes_separates_same_label_by_origin(
    fake_atlas: FakeAtlas,
) -> None:
    df = pd.DataFrame(
        {
            "label": ["A", "A"],
            "x": [50.0, 25.0],
            "y": [75.0, 50.0],
            "z": [100.0, 50.0],
            "origin_csv": ["one.csv", "two.csv"],
        }
    )

    volumes = build_grouped_point_heatmap_volumes(
        df,
        fake_atlas,
        ("label", "origin_csv"),
    )

    assert set(volumes) == {("A", "one.csv"), ("A", "two.csv")}
    assert float(volumes[("A", "one.csv")][4, 3, 2]) == 1.0
    assert float(volumes[("A", "two.csv")][2, 2, 1]) == 1.0


def test_standardize_point_dataframe_rejects_conflicting_extra_columns() -> None:
    raw_df = pd.DataFrame(
        {
            "marker": ["A"],
            "atlas_x": [1.0],
            "atlas_y": [2.0],
            "atlas_z": [3.0],
            "label": ["conflict"],
        }
    )

    with pytest.raises(PointImportError, match="conflict with standardized"):
        standardize_point_dataframe(
            raw_df,
            {"label": "marker", "x": "atlas_x", "y": "atlas_y", "z": "atlas_z"},
        )


def test_convert_point_csv_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "points.csv"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "marker": ["A"],
            "atlas_x": [50.0],
            "atlas_y": [75.0],
            "atlas_z": [100.0],
        }
    ).to_csv(csv_path, index=False)
    mapping_path.write_text(
        json.dumps(
            {"label": "marker", "x": "atlas_x", "y": "atlas_y", "z": "atlas_z"}
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/convert_point_csv.py",
            str(csv_path),
            str(mapping_path),
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    loaded = load_standard_point_parquet(output_path)
    assert loaded["label"].tolist() == ["A"]
    assert loaded["origin_csv"].tolist() == ["points.csv"]


def test_convert_point_csv_script_append(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    existing_csv_path = tmp_path / "existing.csv"
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    _write_point_csv(
        existing_csv_path,
        [
            {
                "marker": "A",
                "atlas_x": 1.0,
                "atlas_y": 2.0,
                "atlas_z": 3.0,
            }
        ],
    )
    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 4.0,
                "atlas_y": 5.0,
                "atlas_z": 6.0,
            }
        ],
    )
    _write_point_mapping(mapping_path)
    convert_point_csv_to_parquet(existing_csv_path, mapping_path, parquet_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/convert_point_csv.py",
            "--append",
            str(append_csv_path),
            str(mapping_path),
            str(parquet_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Appended 1 standardized point rows" in result.stdout
    loaded = load_standard_point_parquet(parquet_path)
    assert loaded["label"].tolist() == ["A", "B"]
    assert loaded["origin_csv"].tolist() == ["existing.csv", "append.csv"]


def test_convert_point_csv_script_append_reports_schema_mismatch(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    append_csv_path = tmp_path / "append.csv"
    mapping_path = tmp_path / "mapping.json"
    parquet_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "label": ["A"],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            "region_name": [pd.NA],
            "acronym": [pd.NA],
            "id": [pd.NA],
            "hemisphere": [pd.NA],
        }
    ).to_parquet(parquet_path, index=False)
    _write_point_csv(
        append_csv_path,
        [
            {
                "marker": "B",
                "atlas_x": 4.0,
                "atlas_y": 5.0,
                "atlas_z": 6.0,
                "score": 0.9,
            }
        ],
    )
    _write_point_mapping(mapping_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/convert_point_csv.py",
            "--append",
            str(append_csv_path),
            str(mapping_path),
            str(parquet_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "schema mismatch" in result.stderr


def test_convert_bltr_point_csv_directory_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = tmp_path / "bltr"
    input_dir.mkdir()
    output_path = tmp_path / "combined.parquet"

    _write_bltr_csv(
        input_dir / "a.csv",
        [
            {
                "marker": "A",
                "experiment_x": 1.0,
                "experiment_y": 2.0,
                "experiment_z": 3.0,
                "strength": 0.5,
                "channel_mono_channel": 7,
                "atlas_x": 50.0,
                "atlas_y": 75.0,
                "atlas_z": 100.0,
                "region_name": "Region One",
                "region_acronym": "R1",
                "region_id": 101,
                "region_hemisphere": "right",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/convert_bltr_point_csv_directory.py",
            str(input_dir),
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    loaded = pd.read_parquet(output_path)
    assert loaded["origin_csv"].tolist() == ["a.csv"]


@pytest.mark.skip(reason="Qt runtime is unavailable in the current test sandbox.")
def test_widget_selected_neuron_controls_show_updated_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    NeuronViewerWidget = _import_neuron_viewer_widget()
    monkeypatch.setattr(NeuronViewerWidget, "_toggle_template", lambda self, state: None)

    viewer = _DummyViewer()
    widget = NeuronViewerWidget(viewer)

    assert widget._render_btn.text() == "Add Full Neurons"
    assert widget._selected_neurons_hint_label.text() == (
        "Only Neurons highlighted in the above table will be added to scene. "
        "Use cmd+A or ctrl+A to select all."
    )


@pytest.mark.skip(reason="Qt runtime is unavailable in the current test sandbox.")
def test_widget_point_parquet_preview_loads_without_atlas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from qtpy.QtCore import QItemSelectionModel

    NeuronViewerWidget = _import_neuron_viewer_widget()
    monkeypatch.setattr(NeuronViewerWidget, "_toggle_template", lambda self, state: None)

    viewer = _DummyViewer()
    widget = NeuronViewerWidget(viewer)

    parquet_path = tmp_path / "points.parquet"
    pd.DataFrame({"label": ["A"], "x": [1.0], "y": [2.0], "z": [3.0]}).to_parquet(
        parquet_path,
        index=False,
    )

    widget._load_point_parquet_file(str(parquet_path))
    selection_model = widget._point_preview_table.selectionModel()
    selection_model.select(
        widget._point_preview_table.model().index(0, 0),
        QItemSelectionModel.Select | QItemSelectionModel.Rows,
    )
    widget._update_point_import_controls()

    assert widget._point_preview_table.rowCount() == 1
    assert "Load an atlas to enable Import Selected Heatmaps." in (
        widget._point_import_status_label.text()
    )
    assert not widget._import_selected_point_heatmaps_btn.isEnabled()
    assert len(viewer.layers) == 0


@pytest.mark.skip(reason="Qt runtime is unavailable in the current test sandbox.")
def test_widget_import_point_parquet_populates_preview_and_imports_selected_layers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_atlas: FakeAtlas,
) -> None:
    from qtpy.QtCore import QItemSelectionModel

    NeuronViewerWidget = _import_neuron_viewer_widget()
    monkeypatch.setattr(NeuronViewerWidget, "_toggle_template", lambda self, state: None)
    warnings: list[str] = []
    monkeypatch.setattr(
        "napari_neuron_navigator.widgets.neuron_viewer.show_warning",
        lambda message: warnings.append(message),
    )

    viewer = _DummyViewer()
    widget = NeuronViewerWidget(viewer)
    widget._atlas = fake_atlas

    parquet_path = tmp_path / "points.parquet"
    pd.DataFrame(
        {
            "label": ["A", "A", "B"],
            "x": [50.0, 25.0, 50.0],
            "y": [75.0, 50.0, 75.0],
            "z": [100.0, 50.0, 100.0],
            "region_name": ["Region/One", pd.NA, "Wrong Region"],
            "acronym": ["R1", pd.NA, "BAD"],
            "id": [101, pd.NA, 999],
            "hemisphere": ["right", pd.NA, "left"],
            "origin_csv": ["one.csv", "one.csv", "two.csv"],
            "score": [0.1, 0.2, 0.3],
        }
    ).to_parquet(parquet_path, index=False)

    widget._load_point_parquet_file(str(parquet_path))

    assert widget._point_preview_table.rowCount() == 2
    assert widget._point_import_status_label.text().startswith("Loaded 3 point(s)")

    selection_model = widget._point_preview_table.selectionModel()
    selection_model.select(
        widget._point_preview_table.model().index(0, 0),
        QItemSelectionModel.Select | QItemSelectionModel.Rows,
    )
    selection_model.select(
        widget._point_preview_table.model().index(1, 0),
        QItemSelectionModel.Select | QItemSelectionModel.Rows,
    )
    widget._update_point_import_controls()

    assert widget._import_selected_point_heatmaps_btn.isEnabled()

    widget._import_selected_point_heatmaps()

    layer_names = [layer.name for layer in viewer.layers]
    assert layer_names == ["Points Heatmap: A [one.csv]", "Points Heatmap: B [two.csv]"]
    assert warnings
    assert "mismatched point(s)" in warnings[0]
    assert "3 selected point(s) into 2 heatmap layer(s)" in widget._point_import_status_label.text()
    assert "mismatched row(s)" in widget._point_import_status_label.text()

    layer_a = viewer.layers[0]
    assert float(layer_a.data[4, 3, 2]) == 2.0
    assert layer_a.metadata["point_count"] == 2
    assert layer_a.metadata["origin_csv"] == "one.csv"


@pytest.mark.skip(reason="Qt runtime is unavailable in the current test sandbox.")
def test_widget_point_parquet_append_refreshes_preview(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    NeuronViewerWidget = _import_neuron_viewer_widget()
    monkeypatch.setattr(NeuronViewerWidget, "_toggle_template", lambda self, state: None)

    viewer = _DummyViewer()
    widget = NeuronViewerWidget(viewer)

    parquet_path = tmp_path / "points.parquet"
    pd.DataFrame(
        {
            "label": ["A", "B"],
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
            "z": [1.0, 2.0],
        }
    ).to_parquet(parquet_path, index=False)

    widget._on_point_append_finished(
        str(parquet_path),
        PointParquetAppendSummary(appended_rows=1, total_rows=2),
    )

    assert widget._point_preview_table.rowCount() == 2
    assert widget._point_file_label.text() == "points.parquet"
    assert widget._point_import_status_label.text() == (
        "Saved points.parquet with 1 added point(s) (2 total)."
    )
