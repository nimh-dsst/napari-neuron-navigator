"""Tests for background workers."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from napari_neuron_navigator.analysis.clustering import ClusterRegionSelection, ClusterResult
from napari_neuron_navigator.analysis.flatmap_correlation import (
    FlatmapVoxelCorrelationSource,
)
from napari_neuron_navigator.isocortex_layers import AllenIsocortexLayerMap
from napari_neuron_navigator.point_import import PointParquetAppendSummary
from napari_neuron_navigator.parquet import BatchParquetConversionSummary


class _BoundSignal:
    """Minimal Qt-like signal implementation for worker tests."""

    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class _Signal:
    """Descriptor that mimics ``qtpy.QtCore.Signal`` enough for tests."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._storage_name = ""

    def __set_name__(self, _owner, name: str) -> None:
        self._storage_name = f"__signal_{name}"

    def __get__(self, instance, _owner):
        if instance is None:
            return self
        if self._storage_name not in instance.__dict__:
            instance.__dict__[self._storage_name] = _BoundSignal()
        return instance.__dict__[self._storage_name]


class _QObject:
    """Minimal QObject stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


def _import_workers_module():
    """Import the workers module with a stubbed QtCore dependency."""
    qtpy_module = types.ModuleType("qtpy")
    qtcore_module = types.ModuleType("qtpy.QtCore")
    qtcore_module.QObject = _QObject
    qtcore_module.Signal = _Signal
    qtpy_module.QtCore = qtcore_module
    previous_qtpy = sys.modules.get("qtpy")
    previous_qtcore = sys.modules.get("qtpy.QtCore")
    previous_workers = sys.modules.get("napari_neuron_navigator.workers")

    try:
        sys.modules["qtpy"] = qtpy_module
        sys.modules["qtpy.QtCore"] = qtcore_module
        sys.modules.pop("napari_neuron_navigator.workers", None)
        return importlib.import_module("napari_neuron_navigator.workers")
    finally:
        if previous_workers is None:
            sys.modules.pop("napari_neuron_navigator.workers", None)
        else:
            sys.modules["napari_neuron_navigator.workers"] = previous_workers

        if previous_qtpy is None:
            sys.modules.pop("qtpy", None)
        else:
            sys.modules["qtpy"] = previous_qtpy

        if previous_qtcore is None:
            sys.modules.pop("qtpy.QtCore", None)
        else:
            sys.modules["qtpy.QtCore"] = previous_qtcore


def _make_cluster_result(neuron_ids: list[str], labels: list[int]) -> ClusterResult:
    """Build a small cluster result fixture."""
    n_neurons = len(neuron_ids)
    return ClusterResult(
        correlation_matrix=np.eye(n_neurons, dtype=np.float32),
        distance_matrix=np.eye(n_neurons, dtype=np.float32),
        linkage_matrix=np.zeros((max(n_neurons - 1, 1), 4), dtype=np.float64),
        neuron_ids=list(neuron_ids),
        reorder_indices=np.arange(n_neurons - 1, -1, -1, dtype=np.intp),
        labels=np.asarray(labels, dtype=np.int32),
    )


def test_cached_brainglobe_atlas_dir_finds_single_cache(tmp_path):
    """Cache lookup should only return an unambiguous local atlas directory."""
    workers = _import_workers_module()
    atlas_dir = tmp_path / "allen_mouse_25um_v1.2"
    atlas_dir.mkdir()

    assert (
        workers.cached_brainglobe_atlas_dir(
            "allen_mouse_25um",
            brainglobe_dir=tmp_path,
        )
        == atlas_dir
    )
    assert (
        workers.cached_brainglobe_atlas_dir(
            "allen_mouse_10um",
            brainglobe_dir=tmp_path,
        )
        is None
    )

    (tmp_path / "allen_mouse_25um_v1.3").mkdir()
    assert (
        workers.cached_brainglobe_atlas_dir(
            "allen_mouse_25um",
            brainglobe_dir=tmp_path,
        )
        is None
    )


def test_region_cache_open_worker_emits_validated_cache(monkeypatch, tmp_path):
    workers = _import_workers_module()
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    cache = object()
    opened = []

    def fake_open(path):
        opened.append(Path(path))
        return cache

    monkeypatch.setattr(cache_module, "open_region_cache", fake_open)
    worker = workers.RegionCacheOpenWorker(tmp_path)
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert opened == [tmp_path]
    assert finished == [cache]
    assert errors == []


def test_region_cache_open_worker_emits_validation_error(monkeypatch, tmp_path):
    workers = _import_workers_module()
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    def fail_open(_path):
        raise ValueError("manifest is corrupt")

    monkeypatch.setattr(cache_module, "open_region_cache", fail_open)
    worker = workers.RegionCacheOpenWorker(tmp_path)
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert finished == []
    assert errors == ["manifest is corrupt"]


def test_cached_atlas_load_worker_uses_local_loader(monkeypatch, tmp_path):
    """CachedAtlasLoadWorker should emit the locally loaded atlas object."""
    workers = _import_workers_module()
    atlas = types.SimpleNamespace(atlas_name="fake_atlas")
    calls = []

    def fake_load_cached(atlas_name, atlas_dir):
        calls.append((atlas_name, Path(atlas_dir)))
        return atlas

    monkeypatch.setattr(workers, "load_cached_brainglobe_atlas", fake_load_cached)
    worker = workers.CachedAtlasLoadWorker("fake_atlas", tmp_path)
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert calls == [("fake_atlas", tmp_path)]
    assert finished == [atlas]
    assert errors == []


def test_atlas_load_worker_reports_cached_load(monkeypatch, tmp_path):
    """AtlasLoadWorker should report and load an existing BrainGlobe cache."""
    workers = _import_workers_module()
    atlas_dir = tmp_path / "allen_mouse_25um_v1.2"
    atlas_dir.mkdir()
    atlas = types.SimpleNamespace(
        atlas_name="allen_mouse_25um",
        structures={1: {"acronym": "R1"}},
    )
    calls = []

    def fake_load(atlas_name, **kwargs):
        calls.append((atlas_name, kwargs))
        return atlas

    monkeypatch.setattr(workers, "load_brainglobe_atlas", fake_load)
    worker = workers.AtlasLoadWorker(
        "allen_mouse_25um",
        brainglobe_dir=tmp_path,
        interm_download_dir=tmp_path,
    )
    statuses = []
    progress = []
    finished = []
    errors = []
    worker.status.connect(statuses.append)
    worker.progress.connect(lambda *args: progress.append(args))
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert calls
    assert calls[0][0] == "allen_mouse_25um"
    assert calls[0][1]["brainglobe_dir"] == tmp_path
    assert "fn_update" in calls[0][1]
    assert any("Found cached allen_mouse_25um" in status for status in statuses)
    assert progress[0] == (0, 0, 0)
    assert progress[-1] == (0, 100, 100)
    assert finished == [atlas]
    assert errors == []


def test_atlas_load_worker_reports_cache_miss_and_download_progress(
    monkeypatch,
    tmp_path,
):
    """Cache misses should show download destination and byte progress."""
    workers = _import_workers_module()
    atlas = types.SimpleNamespace(
        atlas_name="allen_mouse_25um",
        structures={1: {"acronym": "R1"}},
    )
    calls = []

    def fake_load(atlas_name, **kwargs):
        calls.append((atlas_name, kwargs))
        kwargs["fn_update"](25, 100)
        kwargs["fn_update"](100, 100)
        return atlas

    monkeypatch.setattr(workers, "load_brainglobe_atlas", fake_load)
    worker = workers.AtlasLoadWorker(
        "allen_mouse_25um",
        brainglobe_dir=tmp_path,
        interm_download_dir=tmp_path,
    )
    statuses = []
    progress = []
    finished = []
    errors = []
    worker.status.connect(statuses.append)
    worker.progress.connect(lambda *args: progress.append(args))
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert calls
    assert any(
        "was not found in the local BrainGlobe cache" in status
        and f"to {tmp_path}" in status
        for status in statuses
    )
    assert any("Installing allen_mouse_25um" in status for status in statuses)
    assert (0, 100, 25) in progress
    assert (0, 100, 100) in progress
    assert (0, 0, 0) in progress
    assert finished == [atlas]
    assert errors == []


def test_load_brainglobe_atlas_disables_latest_check(monkeypatch, tmp_path):
    """BrainGlobe loads should avoid the remote latest-version check."""
    workers = _import_workers_module()
    calls = []
    atlas = object()

    class FakeBrainGlobeAtlas:
        def __new__(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return atlas

    fake_module = types.ModuleType("brainglobe_atlasapi")
    fake_module.BrainGlobeAtlas = FakeBrainGlobeAtlas
    monkeypatch.setitem(sys.modules, "brainglobe_atlasapi", fake_module)

    result = workers.load_brainglobe_atlas(
        "allen_mouse_25um",
        brainglobe_dir=tmp_path,
        interm_download_dir=tmp_path,
        config_dir=tmp_path / "config.conf",
        fn_update=lambda *_args: None,
    )

    assert result is atlas
    assert calls == [
        (
            ("allen_mouse_25um",),
            {
                "brainglobe_dir": tmp_path,
                "interm_download_dir": tmp_path,
                "check_latest": False,
                "config_dir": tmp_path / "config.conf",
                "fn_update": calls[0][1]["fn_update"],
            },
        )
    ]


def test_atlas_load_worker_emits_error_without_finished(monkeypatch, tmp_path):
    """AtlasLoadWorker should emit errors without reporting success."""
    workers = _import_workers_module()

    def fake_load(_atlas_name, **_kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(workers, "load_brainglobe_atlas", fake_load)
    worker = workers.AtlasLoadWorker(
        "allen_mouse_25um",
        brainglobe_dir=tmp_path,
        interm_download_dir=tmp_path,
    )
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert finished == []
    assert errors == ["download failed"]


class _FakeDuckConnection:
    """Very small DuckDB connection stub used for worker tests."""

    def __init__(self, dataframe: pd.DataFrame | None = None):
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()

    def execute(self, _query):
        return self

    def fetchdf(self):
        return self._dataframe

    def close(self) -> None:
        return None


class _FakeAtlasStruct:
    """Minimal BrainGlobe structure record for cached atlas tests."""

    def __init__(self, data: dict) -> None:
        self.data = data

    def __getitem__(self, key):
        return self.data[key]


class _FakeCachedAtlas:
    """Small atlas object exposing fields used by conversion caching."""

    atlas_name = "allen_mouse_25um"
    shape = (8, 8, 8)
    resolution = (25.0, 25.0, 25.0)

    def __init__(self) -> None:
        self.annotation = np.zeros((8, 8, 8), dtype=np.int32)
        self.structures = {
            5: _FakeAtlasStruct(
                {
                    "name": "Cached Region",
                    "acronym": "CR",
                    "structure_id_path": [5],
                    "rgb_triplet": [1, 2, 3],
                }
            )
        }


def test_convert_worker_uses_batch_conversion_with_alignment(monkeypatch, tmp_path):
    """ConvertWorker should delegate to the batch helper with UI-selected options."""
    workers = _import_workers_module()
    ConvertWorker = workers.ConvertWorker
    calls: dict[str, object] = {}

    def fake_batch_convert_swc_to_parquet(
        input_path,
        output_path,
        *,
        recursive,
        hemisphere,
        atlas_name,
        coord_axis,
        annotate_regions,
        resolution,
        n_workers,
        progress_callback,
        **_kwargs,
    ):
        calls["input_path"] = input_path
        calls["output_path"] = output_path
        calls["recursive"] = recursive
        calls["hemisphere"] = hemisphere
        calls["atlas_name"] = atlas_name
        calls["coord_axis"] = coord_axis
        calls["annotate_regions"] = annotate_regions
        calls["resolution"] = resolution
        calls["n_workers"] = n_workers
        progress_callback("Processing a.swc...", 0, 2)
        return BatchParquetConversionSummary(
            discovered_files=2,
            processed_files=2,
            flipped_files=1,
            rows_written=4,
        )

    monkeypatch.setattr(
        "napari_neuron_navigator.parquet.batch_convert_swc_to_parquet",
        fake_batch_convert_swc_to_parquet,
    )

    output_path = tmp_path / "neurons.parquet"
    worker = ConvertWorker(
        ["a.swc", "b.swc"],
        str(output_path),
        resolution=25,
        hemisphere="right",
        atlas_name="allen_mouse_10um",
    )

    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, BatchParquetConversionSummary]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append(
            (message, current, total)
        )
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert [Path(p).name for p in calls["input_path"]] == ["a.swc", "b.swc"]
    assert Path(calls["output_path"]) == output_path
    assert calls["recursive"] is False
    assert calls["hemisphere"] == "right"
    assert calls["atlas_name"] == "allen_mouse_10um"
    assert calls["coord_axis"] == 2
    assert calls["annotate_regions"] is True
    assert calls["resolution"] == 25
    assert calls["n_workers"] == 1
    assert progress_events[0] == ("Preparing SWC-to-Parquet conversion...", 0, 2)
    assert progress_events[-1] == ("Finalizing Parquet...", 2, 2)
    assert not errors
    assert finished[0][0] == str(output_path)
    assert finished[0][1].flipped_files == 1


def test_convert_worker_passes_cached_atlas_inputs(monkeypatch, tmp_path):
    """ConvertWorker should pass cached atlas-derived inputs to the batch helper."""
    workers = _import_workers_module()
    ConvertWorker = workers.ConvertWorker
    cached_atlas = _FakeCachedAtlas()
    calls: dict[str, object] = {}

    def fake_batch_convert_swc_to_parquet(
        input_path,
        output_path,
        *,
        midline,
        annotation_volume,
        region_lookup,
        progress_callback,
        **_kwargs,
    ):
        calls["input_path"] = input_path
        calls["output_path"] = output_path
        calls["midline"] = midline
        calls["annotation_volume"] = annotation_volume
        calls["region_lookup"] = region_lookup
        progress_callback("Processing a.swc...", 0, 1)
        return BatchParquetConversionSummary(
            discovered_files=1,
            processed_files=1,
            rows_written=2,
        )

    monkeypatch.setattr(
        "napari_neuron_navigator.parquet.batch_convert_swc_to_parquet",
        fake_batch_convert_swc_to_parquet,
    )

    worker = ConvertWorker(
        ["a.swc"],
        str(tmp_path / "cached.parquet"),
        hemisphere="left",
        cached_atlas=cached_atlas,
        use_cached_annotation=True,
    )

    finished: list[tuple[str, BatchParquetConversionSummary]] = []
    errors: list[str] = []
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert calls["midline"] == 87.5
    assert calls["annotation_volume"] is cached_atlas.annotation
    assert calls["region_lookup"][5]["acronym"] == "CR"
    assert not errors
    assert finished[0][1].processed_files == 1


def test_convert_worker_chains_atomic_v3_flatmap_preparation(
    monkeypatch,
    tmp_path,
) -> None:
    workers = _import_workers_module()
    calls: dict[str, object] = {}

    def fake_batch(_input, output_path, *, progress_callback, **_kwargs):
        staged = Path(output_path)
        calls["conversion_output"] = staged
        staged.write_bytes(b"intermediate")
        progress_callback("Converted SWCs", 1, 1)
        return BatchParquetConversionSummary(
            discovered_files=1,
            processed_files=1,
            rows_written=2,
        )

    lookup_set = object()

    def fake_discover(path, **kwargs):
        calls["lookup_dir"] = Path(path)
        calls["lookup_resolution_um"] = kwargs["lookup_resolution_um"]
        return lookup_set

    def fake_augment(source, output, received_lookup_set, **kwargs):
        calls["augment_source"] = Path(source)
        calls["augment_output"] = Path(output)
        calls["lookup_set"] = received_lookup_set
        assert Path(source).read_bytes() == b"intermediate"
        assert kwargs["cancel_callback"]() is False
        Path(output).write_bytes(b"version-3")
        return types.SimpleNamespace(rows=2, output_parquet=Path(output))

    monkeypatch.setattr(
        "napari_neuron_navigator.parquet.batch_convert_swc_to_parquet",
        fake_batch,
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.flatmap_profiles.discover_flatmap_lookup_set",
        fake_discover,
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.flatmap_parquet.augment_neuron_parquet_with_flatmaps",
        fake_augment,
    )
    output = tmp_path / "neurons.parquet"
    worker = workers.ConvertWorker(
        ["a.swc"],
        str(output),
        flatmap_lookup_dir=tmp_path / "lookups",
        flatmap_lookup_resolution_um=10.0,
    )
    errors: list[str] = []
    worker.error.connect(errors.append)

    worker.run()

    assert not errors
    assert output.read_bytes() == b"version-3"
    assert calls["conversion_output"] != output
    assert calls["augment_source"] == calls["conversion_output"]
    assert calls["augment_output"] == output
    assert calls["lookup_dir"] == tmp_path / "lookups"
    assert calls["lookup_resolution_um"] == 10.0
    assert calls["lookup_set"] is lookup_set
    assert not Path(calls["conversion_output"]).exists()


def test_flatmap_preparation_worker_reuses_cache_and_finishes_after_publication_cancel(
    monkeypatch,
    tmp_path,
) -> None:
    workers = _import_workers_module()
    output = tmp_path / "neurons_flatmap.parquet"
    expected_cache_dir = tmp_path / ".flatmap-lookup-arrays"
    lookup_set = object()
    summary = types.SimpleNamespace(rows=2, output_parquet=output)
    calls: dict[str, object] = {}
    worker = workers.FlatmapParquetPreparationWorker(
        tmp_path / "neurons.parquet",
        output,
        tmp_path / "lookups",
        lookup_resolution_um=10.0,
    )

    def fake_discover(path, **kwargs):
        cache_dir = Path(kwargs["npy_cache_dir"])
        cache_dir.mkdir()
        (cache_dir / "reuse.marker").write_text("ready")
        calls["discover_lookup_dir"] = Path(path)
        calls["discover_cache_dir"] = cache_dir
        assert kwargs["cancel_callback"]() is False
        return lookup_set

    def fake_augment(source, destination, received_lookup_set, **kwargs):
        cache_dir = Path(kwargs["npy_cache_dir"])
        calls["augment_source"] = Path(source)
        calls["augment_destination"] = Path(destination)
        calls["augment_cache_dir"] = cache_dir
        assert received_lookup_set is lookup_set
        assert (cache_dir / "reuse.marker").read_text() == "ready"
        Path(destination).write_bytes(b"published-v3")
        # The destination has already been atomically published. A late cancel
        # request must not convert this completed run into an error.
        worker.cancel()
        return summary

    monkeypatch.setattr(
        "napari_neuron_navigator.flatmap_profiles.discover_flatmap_lookup_set",
        fake_discover,
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.flatmap_parquet.augment_neuron_parquet_with_flatmaps",
        fake_augment,
    )
    finished: list[object] = []
    errors: list[str] = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert output.read_bytes() == b"published-v3"
    assert calls["discover_lookup_dir"] == tmp_path / "lookups"
    assert calls["discover_cache_dir"] == expected_cache_dir
    assert calls["augment_cache_dir"] == expected_cache_dir
    assert calls["augment_source"] == tmp_path / "neurons.parquet"
    assert calls["augment_destination"] == output
    assert finished == [summary]
    assert errors == []


def test_heatmap_worker_forwards_node_type_and_radius_filters(monkeypatch):
    """HeatmapWorker should pass node-type and soma-radius filters through."""
    workers = _import_workers_module()
    HeatmapWorker = workers.HeatmapWorker
    calls: dict[str, object] = {}

    class _FakeConn:
        def close(self) -> None:
            calls["closed"] = True

    def fake_build_node_counts_volume(conn, parquet_path, atlas, **kwargs):
        calls["conn"] = conn
        calls["parquet_path"] = parquet_path
        calls["atlas"] = atlas
        calls.update(kwargs)
        return np.ones((2, 2, 2), dtype=np.float32)

    monkeypatch.setattr("duckdb.connect", lambda: _FakeConn())
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.heatmap.build_node_counts_volume",
        fake_build_node_counts_volume,
    )

    atlas = types.SimpleNamespace(annotation=np.zeros((2, 2, 2)), resolution=(25.0,))
    worker = HeatmapWorker(
        parquet_path="neurons.parquet",
        atlas=atlas,
        region_ids=[101],
        file_ids=["n1"],
        node_types=[3, 4],
        soma_radius_um=100.0,
        depth_bin_factor=2,
        depth_axis=1,
    )
    finished: list[np.ndarray] = []
    errors: list[str] = []
    worker.finished.connect(lambda volume: finished.append(volume))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert not errors
    assert finished and finished[0].shape == (2, 2, 2)
    assert calls["parquet_path"] == "neurons.parquet"
    assert calls["atlas"] is atlas
    assert calls["region_ids"] == [101]
    assert calls["file_ids"] == ["n1"]
    assert calls["node_types"] == [3, 4]
    assert calls["soma_radius_um"] == 100.0
    assert calls["depth_bin_factor"] == 2
    assert calls["depth_axis"] == 1
    assert calls["closed"] is True


def test_convert_worker_accepts_directory_source(monkeypatch, tmp_path):
    """Directory conversion should be discovered inside the worker, not the UI."""
    workers = _import_workers_module()
    ConvertWorker = workers.ConvertWorker
    calls: dict[str, object] = {}

    def fake_batch_convert_swc_to_parquet(
        input_path,
        output_path,
        *,
        recursive,
        hemisphere,
        atlas_name,
        coord_axis,
        annotate_regions,
        resolution,
        n_workers,
        progress_callback,
        **_kwargs,
    ):
        calls["input_path"] = input_path
        calls["output_path"] = output_path
        calls["recursive"] = recursive
        calls["hemisphere"] = hemisphere
        calls["atlas_name"] = atlas_name
        calls["coord_axis"] = coord_axis
        calls["annotate_regions"] = annotate_regions
        calls["resolution"] = resolution
        calls["n_workers"] = n_workers
        progress_callback("Discovered 3 SWC file(s).", 0, 3)
        return BatchParquetConversionSummary(
            discovered_files=3,
            processed_files=3,
            rows_written=6,
        )

    monkeypatch.setattr(
        "napari_neuron_navigator.parquet.batch_convert_swc_to_parquet",
        fake_batch_convert_swc_to_parquet,
    )

    input_dir = tmp_path / "swcs"
    output_path = tmp_path / "neurons.parquet"
    worker = ConvertWorker(
        str(input_dir),
        str(output_path),
        resolution=25,
        recursive=True,
    )

    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, BatchParquetConversionSummary]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append(
            (message, current, total)
        )
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert calls["input_path"] == input_dir
    assert Path(calls["output_path"]) == output_path
    assert calls["recursive"] is True
    assert calls["annotate_regions"] is True
    assert calls["n_workers"] == 1
    assert progress_events[0] == ("Searching for SWC files...", 0, 0)
    assert ("Discovered 3 SWC file(s).", 0, 3) in progress_events
    assert progress_events[-1] == ("Finalizing Parquet...", 3, 3)
    assert not errors
    assert finished[0][0] == str(output_path)


def test_convert_worker_logs_files_source_timing(monkeypatch, tmp_path, caplog):
    """ConvertWorker should time explicit file-list conversions in debug logs."""
    workers = _import_workers_module()
    ConvertWorker = workers.ConvertWorker
    calls: dict[str, object] = {}

    def fake_batch_convert_swc_to_parquet(
        input_path,
        output_path,
        *,
        recursive,
        source_mode,
        progress_callback,
        **_kwargs,
    ):
        calls["input_path"] = input_path
        calls["output_path"] = output_path
        calls["recursive"] = recursive
        calls["source_mode"] = source_mode
        progress_callback("Processing a.swc...", 0, 2)
        return BatchParquetConversionSummary(
            discovered_files=2,
            processed_files=2,
            rows_written=4,
        )

    monkeypatch.setattr(
        "napari_neuron_navigator.parquet.batch_convert_swc_to_parquet",
        fake_batch_convert_swc_to_parquet,
    )

    worker = ConvertWorker(
        ["a.swc", "b.swc"],
        str(tmp_path / "files.parquet"),
        source_mode="files",
    )

    with caplog.at_level(logging.DEBUG, logger=workers.logger.name):
        worker.run()

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert calls["source_mode"] == "files"
    assert "swc_conversion_worker_start source_mode=files" in messages
    assert "file_count=2" in messages
    assert "swc_conversion_worker_batch_ok source_mode=files elapsed_s=" in messages
    assert "swc_conversion_worker_finished source_mode=files elapsed_s=" in messages


def test_convert_worker_logs_directory_source_timing(monkeypatch, tmp_path, caplog):
    """ConvertWorker should time directory conversions in debug logs."""
    workers = _import_workers_module()
    ConvertWorker = workers.ConvertWorker
    calls: dict[str, object] = {}

    def fake_batch_convert_swc_to_parquet(
        input_path,
        output_path,
        *,
        recursive,
        source_mode,
        progress_callback,
        **_kwargs,
    ):
        calls["input_path"] = input_path
        calls["output_path"] = output_path
        calls["recursive"] = recursive
        calls["source_mode"] = source_mode
        progress_callback("Discovered 26 SWC file(s).", 0, 26)
        return BatchParquetConversionSummary(
            discovered_files=26,
            processed_files=26,
            rows_written=52,
        )

    monkeypatch.setattr(
        "napari_neuron_navigator.parquet.batch_convert_swc_to_parquet",
        fake_batch_convert_swc_to_parquet,
    )

    input_dir = tmp_path / "swcs"
    worker = ConvertWorker(
        str(input_dir),
        str(tmp_path / "directory.parquet"),
        recursive=True,
        source_mode="directory",
    )

    with caplog.at_level(logging.DEBUG, logger=workers.logger.name):
        worker.run()

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert calls["input_path"] == input_dir
    assert calls["recursive"] is True
    assert calls["source_mode"] == "directory"
    assert "swc_conversion_worker_start source_mode=directory" in messages
    assert f"source={input_dir}" in messages
    assert "swc_conversion_worker_batch_ok source_mode=directory elapsed_s=" in messages
    assert "discovered=26 processed=26 failed=0 rows=52" in messages


def test_correlation_worker_uses_multi_region_mask_and_attaches_metadata(monkeypatch):
    """CorrelationWorker should union selected regions and persist run metadata."""
    workers = _import_workers_module()
    CorrelationWorker = workers.CorrelationWorker
    calls: dict[str, object] = {}

    def fake_get_expanded_region_voxel_ids_for_regions(
        atlas, acronyms, increase_fraction
    ):
        calls["atlas"] = atlas
        calls["acronyms"] = list(acronyms)
        calls["increase_fraction"] = float(increase_fraction)
        return np.zeros((2, 2, 2), dtype=np.int32)

    def fake_compute_pearson_correlation_matrix(
        conn,
        parquet_path,
        voxel_id_map,
        resolution,
        file_ids=None,
    ):
        calls["parquet_path"] = parquet_path
        calls["voxel_id_map_shape"] = voxel_id_map.shape
        calls["resolution"] = resolution
        calls["file_ids"] = file_ids
        return pd.DataFrame({"swc_id_1": [], "swc_id_2": [], "r": []})

    def fake_correlation_long_to_matrix(_corr_df):
        return pd.DataFrame([[1.0, 0.4], [0.4, 1.0]], columns=["n1", "n2"]), np.array(
            [[1.0, 0.4], [0.4, 1.0]],
            dtype=np.float32,
        )

    def fake_compute_clustermap_data(mat, neuron_ids, method, n_clusters):
        calls["linkage_method"] = method
        calls["n_clusters"] = n_clusters
        assert list(neuron_ids) == ["n1", "n2"]
        return _make_cluster_result(["n1", "n2"], [1, 2])

    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        fake_get_expanded_region_voxel_ids_for_regions,
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.correlation.compute_pearson_correlation_matrix",
        fake_compute_pearson_correlation_matrix,
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.correlation.correlation_long_to_matrix",
        fake_correlation_long_to_matrix,
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.clustering.compute_clustermap_data",
        fake_compute_clustermap_data,
    )
    monkeypatch.setattr("duckdb.connect", lambda: _FakeDuckConnection())

    atlas = types.SimpleNamespace(
        resolution=(25.0, 25.0, 25.0),
        atlas_name="fake_atlas",
    )
    region_selection = ClusterRegionSelection(
        selected_region_ids=[184, 500],
        selected_region_acronyms=["FRP", "CP"],
        represented_region_ids=[68, 500],
        represented_region_acronyms=["FRP1", "CP"],
    )
    worker = CorrelationWorker(
        parquet_path="neurons.parquet",
        atlas=atlas,
        region_selection=region_selection,
        dilation_fraction=0.3,
        linkage_method="average",
        n_clusters=4,
        file_ids=["n1", "n2"],
    )

    finished: list[ClusterResult] = []
    errors: list[str] = []
    worker.finished.connect(lambda result: finished.append(result))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert not errors
    assert calls["acronyms"] == ["FRP", "CP"]
    assert calls["increase_fraction"] == 0.3
    assert calls["voxel_id_map_shape"] == (2, 2, 2)
    assert calls["resolution"] == 25.0
    assert calls["file_ids"] == ["n1", "n2"]
    metadata = finished[0].metadata
    assert metadata is not None
    assert metadata.analysis_method == "voxel_correlation"
    assert metadata.clustering_algorithm == "hierarchical"
    assert metadata.distance_metric == "one_minus_pearson_r"
    assert metadata.clustering_linkage == "average"
    assert metadata.dendrogram_linkage == "average"
    assert metadata.selected_region_ids == [184, 500]
    assert metadata.represented_region_acronyms == ["FRP1", "CP"]
    assert metadata.requested_cluster_count == 4
    assert metadata.actual_cluster_count == 2
    assert metadata.atlas_name == "fake_atlas"
    assert metadata.dendrogram_leaf_order == [1, 0]


def test_correlation_worker_allows_unfiltered_ccf_scope(monkeypatch):
    workers = _import_workers_module()
    CorrelationWorker = workers.CorrelationWorker
    calls: dict[str, object] = {}

    def fake_compute_pearson(
        conn,
        parquet_path,
        voxel_id_map,
        resolution,
        file_ids=None,
    ):
        calls["voxel_id_map"] = voxel_id_map
        calls["file_ids"] = file_ids
        return pd.DataFrame(
            {
                "swc_id_1": ["n1", "n1", "n2", "n2"],
                "swc_id_2": ["n1", "n2", "n1", "n2"],
                "r": [1.0, 0.5, 0.5, 1.0],
            }
        )

    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.correlation.compute_pearson_correlation_matrix",
        fake_compute_pearson,
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.clustering.compute_clustermap_data",
        lambda mat, neuron_ids, method, n_clusters: _make_cluster_result(
            list(neuron_ids), [1, 2]
        ),
    )
    monkeypatch.setattr("duckdb.connect", lambda: _FakeDuckConnection())

    worker = CorrelationWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(
            resolution=(25.0, 25.0, 25.0), atlas_name="fake_atlas"
        ),
        region_selection=None,
        file_ids=["n1", "n2"],
        n_clusters=2,
    )
    finished: list[ClusterResult] = []
    worker.finished.connect(finished.append)
    worker.run()

    assert calls == {"voxel_id_map": None, "file_ids": ["n1", "n2"]}
    assert finished[0].metadata is not None
    assert finished[0].metadata.selected_region_ids == []
    assert finished[0].metadata.dilation_fraction == 0.0


def test_clustering_preflight_counts_with_reusable_region_map(monkeypatch):
    workers = _import_workers_module()
    voxel_id_map = np.zeros((3, 3, 3), dtype=np.int32)
    mask_builder = MagicMock(return_value=voxel_id_map)
    count_helper = MagicMock(return_value=500_001)
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        mask_builder,
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.correlation.count_correlation_input_nodes",
        count_helper,
    )
    monkeypatch.setattr("duckdb.connect", lambda: _FakeDuckConnection())
    selection = ClusterRegionSelection(
        selected_region_ids=[184],
        selected_region_acronyms=["FRP"],
        represented_region_ids=[68],
        represented_region_acronyms=["FRP1"],
    )
    worker = workers.ClusteringPreflightWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(resolution=(25.0, 25.0, 25.0)),
        coordinate_space="ccf",
        clustering_method="voxel",
        region_selection=selection,
        dilation_fraction=0.2,
        file_ids=["n1", "n2"],
    )
    finished: list = []
    worker.finished.connect(finished.append)

    worker.run()

    mask_builder.assert_called_once_with(
        worker._atlas,
        ["FRP"],
        0.2,
    )
    assert count_helper.call_args.args[2] is voxel_id_map
    assert count_helper.call_args.kwargs["file_ids"] == ["n1", "n2"]
    assert finished[0].node_count == 500_001
    assert finished[0].voxel_id_map is voxel_id_map


def test_flatmap_correlation_worker_projects_region_mask_with_sentinel_plane(
    monkeypatch,
) -> None:
    """Flatmap region filters should align with heatmaps that include depth -1."""
    workers = _import_workers_module()
    source = FlatmapVoxelCorrelationSource(
        projected_nodes=pd.DataFrame(),
        volume_shape=(2, 2, 2),
        input_file_ids=("n1", "n2"),
        y_bins=2,
        x_bins=2,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
        flatmap_path="flatmap.nrrd",
        depth_path="depth.nrrd",
    )
    atlas = types.SimpleNamespace(annotation=np.zeros((1, 1, 1), dtype=np.int32))
    selection = ClusterRegionSelection(
        selected_region_ids=[184],
        selected_region_acronyms=["FRP"],
        represented_region_ids=[68],
        represented_region_acronyms=["FRP1"],
    )

    import napari_neuron_navigator.flatmap_labels as labels_module
    import napari_neuron_navigator.flatmap_loader as loader_module

    monkeypatch.setattr(
        loader_module,
        "load_flatmap_volume_set",
        lambda _flatmap_path, _depth_path: types.SimpleNamespace(
            flatmap=np.zeros((1, 1, 1, 2), dtype=np.float32),
            depth=np.zeros((1, 1, 1), dtype=np.float32),
        ),
    )
    captured: dict[str, object] = {}

    def fake_build_labels(*_args, **kwargs):
        captured["selected_region_ids"] = kwargs["selected_region_ids"]
        captured["mirror_depth_fallback"] = kwargs["mirror_depth_fallback"]
        captured["mirror_coord_axis"] = kwargs["mirror_coord_axis"]
        return types.SimpleNamespace(
            labels=np.array([[[1, 0], [0, 1]]], dtype=np.int32),
            summary=types.SimpleNamespace(
                labeled_voxels=2,
                valid_source_voxels=2,
                collision_voxels=0,
            ),
            represented_region_ids=[68],
        )

    monkeypatch.setattr(
        labels_module,
        "build_flatmap_region_label_volume",
        fake_build_labels,
    )
    worker = workers.FlatmapCorrelationWorker(
        source=source,
        atlas=atlas,
        parquet_path="neurons.parquet",
        region_selection=selection,
    )

    mask, metadata = worker._build_region_mask()

    assert captured["selected_region_ids"] == [68]
    assert captured["mirror_depth_fallback"] is True
    assert captured["mirror_coord_axis"] == 2
    assert mask.shape == (2, 2, 2)
    assert mask[0].sum() == 0
    assert bool(mask[1, 0, 0]) is True
    assert metadata["flatmap_region_labeled_voxels"] == 2
    assert metadata["flatmap_region_mirrored_depth_source_voxels"] == 0


def test_flatmap_correlation_worker_uses_cache_without_nrrd_or_annotation(
    monkeypatch,
) -> None:
    workers = _import_workers_module()
    source = FlatmapVoxelCorrelationSource(
        projected_nodes=pd.DataFrame(),
        volume_shape=(1, 2, 2),
        input_file_ids=("n1", "n2"),
        y_bins=2,
        x_bins=2,
        depth_bin_um=25.0,
        include_depth_minus_one=False,
        coordinate_mode="parquet_columns",
        cache_dir="cache",
        cache_profile_id="profile",
        cache_style="both_square",
    )

    class _AtlasWithoutAnnotation:
        atlas_name = "allen_mouse_25um"

        @property
        def annotation(self):
            raise AssertionError("precomputed analysis must not access annotation")

    selection = ClusterRegionSelection(
        selected_region_ids=[184],
        selected_region_acronyms=["FRP"],
        represented_region_ids=[68],
        represented_region_acronyms=["FRP1"],
    )
    import napari_neuron_navigator.flatmap_loader as loader_module
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    monkeypatch.setattr(
        loader_module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("precomputed analysis must not load NRRDs")
        ),
    )
    profile = object()
    cache_closed: list[bool] = []

    cached_labels = np.array([[[68, 0], [0, 0]]], dtype=np.int32)

    def close_cache():
        cache_closed.append(True)
        cached_labels.fill(0)

    cache = types.SimpleNamespace(
        profile=lambda _profile_id: profile,
        close=close_cache,
    )
    monkeypatch.setattr(
        cache_module,
        "open_region_cache",
        lambda _path: cache,
    )
    monkeypatch.setattr(
        cache_module,
        "materialize_region_selection",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            labels=cached_labels,
            represented_region_ids=(68,),
            summary=types.SimpleNamespace(
                labeled_bins=1,
                collision_bins=0,
                source_voxel_count=3,
            ),
        ),
    )
    worker = workers.FlatmapCorrelationWorker(
        source=source,
        atlas=_AtlasWithoutAnnotation(),
        parquet_path="neurons.parquet",
        region_selection=selection,
    )

    mask, metadata = worker._build_region_mask()

    assert mask.tolist() == [[[True, False], [False, False]]]
    assert metadata["flatmap_region_source"] == "precomputed_cache"
    assert metadata["flatmap_region_cache_profile_id"] == "profile"
    assert cache_closed == [True]


def test_flatmap_correlation_worker_counts_lookup_modes_for_metadata() -> None:
    workers = _import_workers_module()

    counts = workers.FlatmapCorrelationWorker._lookup_mode_counts(
        pd.DataFrame(
            {
                "flatmap_lookup_mode": [
                    "direct",
                    "mirrored_depth",
                    "mirrored",
                    "mirrored",
                    "unmapped",
                ]
            }
        )
    )

    assert counts == {
        "flatmap_direct_lookup_node_count": 1,
        "flatmap_mirrored_depth_lookup_node_count": 1,
        "flatmap_mirrored_lookup_node_count": 2,
        "flatmap_unmapped_lookup_node_count": 1,
    }


def test_soma_cluster_worker_hierarchical_attaches_true_linkage(monkeypatch):
    """Hierarchical soma clustering should record the chosen linkage for clustering and dendrograms."""
    workers = _import_workers_module()
    SomaClusterWorker = workers.SomaClusterWorker

    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        lambda atlas, acronyms, increase_fraction: np.zeros((4, 4, 4), dtype=np.int32),
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.clustering.cluster_somas_hierarchical",
        lambda coords, neuron_ids, method, n_clusters: _make_cluster_result(
            neuron_ids, [1, 2]
        ),
    )
    monkeypatch.setattr(
        "duckdb.connect",
        lambda: _FakeDuckConnection(
            pd.DataFrame(
                {
                    "file_id": ["n1", "n2"],
                    "x": [0.0, 25.0],
                    "y": [0.0, 25.0],
                    "z": [0.0, 25.0],
                }
            )
        ),
    )

    worker = SomaClusterWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(
            resolution=(25.0, 25.0, 25.0), atlas_name="fake_atlas"
        ),
        region_selection=ClusterRegionSelection(
            selected_region_ids=[184],
            selected_region_acronyms=["FRP"],
            represented_region_ids=[68],
            represented_region_acronyms=["FRP1"],
        ),
        dilation_fraction=0.2,
        algorithm="hierarchical",
        linkage_method="ward",
        n_clusters=3,
    )
    finished: list[ClusterResult] = []
    worker.finished.connect(lambda result: finished.append(result))

    worker.run()

    metadata = finished[0].metadata
    assert metadata is not None
    assert metadata.clustering_algorithm == "hierarchical"
    assert metadata.clustering_linkage == "ward"
    assert metadata.dendrogram_linkage == "ward"
    assert metadata.requested_cluster_count == 3
    assert metadata.distance_metric == "euclidean_um"


def test_soma_cluster_worker_filters_to_current_table_file_ids(monkeypatch):
    """Soma clustering should respect an optional current-table file-id subset."""
    workers = _import_workers_module()
    SomaClusterWorker = workers.SomaClusterWorker
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        lambda atlas, acronyms, increase_fraction: np.zeros((4, 4, 4), dtype=np.int32),
    )

    def fake_cluster_somas_hierarchical(coords, neuron_ids, method, n_clusters):
        observed["neuron_ids"] = list(neuron_ids)
        observed["coords_shape"] = tuple(coords.shape)
        return _make_cluster_result(list(neuron_ids), [1, 2])

    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.clustering.cluster_somas_hierarchical",
        fake_cluster_somas_hierarchical,
    )
    monkeypatch.setattr(
        "duckdb.connect",
        lambda: _FakeDuckConnection(
            pd.DataFrame(
                {
                    "file_id": ["n1", "n2", "n3"],
                    "x": [0.0, 25.0, 50.0],
                    "y": [0.0, 25.0, 50.0],
                    "z": [0.0, 25.0, 50.0],
                }
            )
        ),
    )

    worker = SomaClusterWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(
            resolution=(25.0, 25.0, 25.0),
            atlas_name="fake_atlas",
        ),
        region_selection=ClusterRegionSelection(
            selected_region_ids=[184],
            selected_region_acronyms=["FRP"],
            represented_region_ids=[68],
            represented_region_acronyms=["FRP1"],
        ),
        algorithm="hierarchical",
        linkage_method="ward",
        n_clusters=2,
        file_ids=["n1", "n2"],
    )

    finished: list[ClusterResult] = []
    worker.finished.connect(lambda result: finished.append(result))

    worker.run()

    assert observed["neuron_ids"] == ["n1", "n2"]
    assert observed["coords_shape"] == (2, 3)
    assert finished[0].neuron_ids == ["n1", "n2"]


def test_soma_cluster_worker_allows_unfiltered_ccf_scope(monkeypatch):
    workers = _import_workers_module()
    SomaClusterWorker = workers.SomaClusterWorker
    mask_builder = MagicMock()
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        mask_builder,
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.clustering.query_ccf_soma_coordinates",
        lambda parquet_path, resolution, file_ids, voxel_id_map: (
            ["n1", "n2"],
            np.array([[0.0, 0.0, 0.0], [25.0, 25.0, 25.0]]),
            2,
        ),
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.clustering.cluster_somas_hierarchical",
        lambda coords, neuron_ids, method, n_clusters: _make_cluster_result(
            list(neuron_ids), [1, 2]
        ),
    )

    worker = SomaClusterWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(
            resolution=(25.0, 25.0, 25.0), atlas_name="fake_atlas"
        ),
        region_selection=None,
        algorithm="hierarchical",
        n_clusters=2,
        file_ids=["n1", "n2"],
    )
    finished: list[ClusterResult] = []
    worker.finished.connect(finished.append)
    worker.run()

    mask_builder.assert_not_called()
    assert finished[0].metadata is not None
    assert finished[0].metadata.selected_region_ids == []
    assert finished[0].metadata.dilation_fraction == 0.0


def test_soma_cluster_worker_kmeans_uses_synthesized_dendrogram_linkage(monkeypatch):
    """K-means soma clustering should keep clustering linkage empty and dendrogram linkage synthetic."""
    workers = _import_workers_module()
    SomaClusterWorker = workers.SomaClusterWorker

    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        lambda atlas, acronyms, increase_fraction: np.zeros((4, 4, 4), dtype=np.int32),
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.clustering.cluster_somas_kmeans",
        lambda coords, neuron_ids, n_clusters: _make_cluster_result(neuron_ids, [1, 1]),
    )
    monkeypatch.setattr(
        "duckdb.connect",
        lambda: _FakeDuckConnection(
            pd.DataFrame(
                {
                    "file_id": ["n1", "n2"],
                    "x": [0.0, 25.0],
                    "y": [0.0, 25.0],
                    "z": [0.0, 25.0],
                }
            )
        ),
    )

    worker = SomaClusterWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(
            resolution=(25.0, 25.0, 25.0), atlas_name="fake_atlas"
        ),
        region_selection=ClusterRegionSelection(
            selected_region_ids=[184, 500],
            selected_region_acronyms=["FRP", "CP"],
            represented_region_ids=[68, 500],
            represented_region_acronyms=["FRP1", "CP"],
        ),
        algorithm="kmeans",
        n_clusters=2,
    )
    finished: list[ClusterResult] = []
    worker.finished.connect(lambda result: finished.append(result))

    worker.run()

    metadata = finished[0].metadata
    assert metadata is not None
    assert metadata.clustering_algorithm == "kmeans"
    assert metadata.clustering_linkage is None
    assert metadata.dendrogram_linkage == "average"
    assert metadata.requested_cluster_count == 2


def test_soma_cluster_worker_dbscan_records_dbscan_parameters(monkeypatch):
    """DBSCAN soma clustering should keep clustering linkage empty and persist DBSCAN params."""
    workers = _import_workers_module()
    SomaClusterWorker = workers.SomaClusterWorker

    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        lambda atlas, acronyms, increase_fraction: np.zeros((4, 4, 4), dtype=np.int32),
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.clustering.cluster_somas_dbscan",
        lambda coords, neuron_ids, eps, min_samples: _make_cluster_result(
            neuron_ids, [1, 2]
        ),
    )
    monkeypatch.setattr(
        "duckdb.connect",
        lambda: _FakeDuckConnection(
            pd.DataFrame(
                {
                    "file_id": ["n1", "n2"],
                    "x": [0.0, 25.0],
                    "y": [0.0, 25.0],
                    "z": [0.0, 25.0],
                }
            )
        ),
    )

    worker = SomaClusterWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(
            resolution=(25.0, 25.0, 25.0), atlas_name="fake_atlas"
        ),
        region_selection=ClusterRegionSelection(
            selected_region_ids=[184],
            selected_region_acronyms=["FRP"],
            represented_region_ids=[68],
            represented_region_acronyms=["FRP1"],
        ),
        algorithm="dbscan",
        eps=150.0,
        min_samples=7,
    )
    finished: list[ClusterResult] = []
    worker.finished.connect(lambda result: finished.append(result))

    worker.run()

    metadata = finished[0].metadata
    assert metadata is not None
    assert metadata.clustering_algorithm == "dbscan"
    assert metadata.clustering_linkage is None
    assert metadata.dendrogram_linkage == "average"
    assert metadata.requested_cluster_count is None
    assert metadata.dbscan_eps == 150.0
    assert metadata.dbscan_min_samples == 7


def test_append_point_file_worker_routes_csv_to_append_helper(monkeypatch, tmp_path):
    """AppendPointFileWorker should route CSV input to the CSV append helper."""
    workers = _import_workers_module()
    AppendPointFileWorker = workers.AppendPointFileWorker
    calls: dict[str, object] = {}

    def fake_append_point_csv_to_parquet(
        csv_path,
        mapping_path,
        parquet_path,
        output_path,
    ):
        calls["csv_path"] = csv_path
        calls["mapping_path"] = mapping_path
        calls["parquet_path"] = parquet_path
        calls["output_path"] = output_path
        return PointParquetAppendSummary(appended_rows=2, total_rows=5)

    monkeypatch.setattr(
        "napari_neuron_navigator.point_import.append_point_csv_to_parquet",
        fake_append_point_csv_to_parquet,
    )

    worker = AppendPointFileWorker(
        str(tmp_path / "points.csv"),
        str(tmp_path / "mapping.json"),
        str(tmp_path / "points.parquet"),
        str(tmp_path / "points_out.parquet"),
    )
    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, PointParquetAppendSummary]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append(
            (message, current, total)
        )
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert Path(calls["csv_path"]).name == "points.csv"
    assert Path(calls["mapping_path"]).name == "mapping.json"
    assert Path(calls["parquet_path"]).name == "points.parquet"
    assert Path(calls["output_path"]).name == "points_out.parquet"
    assert progress_events[0] == ("Validating point CSV and target Parquet...", 1, 3)
    assert progress_events[-1] == ("Done", 3, 3)
    assert not errors
    assert finished[0][0].endswith("points_out.parquet")
    assert finished[0][1].appended_rows == 2
    assert finished[0][1].total_rows == 5


def test_append_point_file_worker_routes_parquet_to_append_helper(
    monkeypatch, tmp_path
):
    """AppendPointFileWorker should route Parquet input to the Parquet append helper."""
    workers = _import_workers_module()
    AppendPointFileWorker = workers.AppendPointFileWorker
    calls: dict[str, object] = {}

    def fake_append_point_parquet_to_parquet(
        input_parquet_path,
        parquet_path,
        output_path,
    ):
        calls["input_parquet_path"] = input_parquet_path
        calls["parquet_path"] = parquet_path
        calls["output_path"] = output_path
        return PointParquetAppendSummary(appended_rows=3, total_rows=8)

    monkeypatch.setattr(
        "napari_neuron_navigator.point_import.append_point_parquet_to_parquet",
        fake_append_point_parquet_to_parquet,
    )

    worker = AppendPointFileWorker(
        str(tmp_path / "points_in.parquet"),
        None,
        str(tmp_path / "points.parquet"),
        str(tmp_path / "points_out.parquet"),
    )
    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, PointParquetAppendSummary]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append(
            (message, current, total)
        )
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert Path(calls["input_parquet_path"]).name == "points_in.parquet"
    assert Path(calls["parquet_path"]).name == "points.parquet"
    assert Path(calls["output_path"]).name == "points_out.parquet"
    assert progress_events[0] == ("Validating point Parquet schemas...", 1, 3)
    assert not errors
    assert finished[0][1].appended_rows == 3
    assert finished[0][1].total_rows == 8


def test_append_point_file_worker_emits_error(monkeypatch, tmp_path):
    """AppendPointFileWorker should surface append failures through error."""
    workers = _import_workers_module()
    AppendPointFileWorker = workers.AppendPointFileWorker

    def fake_append_point_csv_to_parquet(
        _csv_path,
        _mapping_path,
        _parquet_path,
        _output_path,
    ):
        raise ValueError("schema mismatch")

    monkeypatch.setattr(
        "napari_neuron_navigator.point_import.append_point_csv_to_parquet",
        fake_append_point_csv_to_parquet,
    )

    worker = AppendPointFileWorker(
        str(tmp_path / "points.csv"),
        str(tmp_path / "mapping.json"),
        str(tmp_path / "points.parquet"),
        str(tmp_path / "points_out.parquet"),
    )
    finished: list[tuple[str, PointParquetAppendSummary]] = []
    errors: list[str] = []
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert not finished
    assert errors == ["schema mismatch"]


def test_convert_point_csv_worker_delegates_to_batch_helper(monkeypatch, tmp_path):
    """ConvertPointCSVWorker should call the batch point CSV helper."""
    workers = _import_workers_module()
    ConvertPointCSVWorker = workers.ConvertPointCSVWorker
    calls: dict[str, object] = {}

    def fake_convert_point_csv_files_to_parquet(
        csv_paths,
        output_path,
        mapping_path,
        progress_callback,
    ):
        calls["csv_paths"] = csv_paths
        calls["output_path"] = output_path
        calls["mapping_path"] = mapping_path
        progress_callback("Processing point CSV 1/2: one.csv", 0, 2)
        return types.SimpleNamespace(
            discovered_files=2,
            processed_files=2,
            rows_written=4,
        )

    monkeypatch.setattr(
        "napari_neuron_navigator.point_import.convert_point_csv_files_to_parquet",
        fake_convert_point_csv_files_to_parquet,
    )

    worker = ConvertPointCSVWorker(
        [str(tmp_path / "one.csv"), str(tmp_path / "two.csv")],
        str(tmp_path / "points.parquet"),
        str(tmp_path / "mapping.json"),
    )
    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, object]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append(
            (message, current, total)
        )
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert [Path(path).name for path in calls["csv_paths"]] == ["one.csv", "two.csv"]
    assert Path(calls["output_path"]).name == "points.parquet"
    assert Path(calls["mapping_path"]).name == "mapping.json"
    assert progress_events[0] == ("Processing point CSV 1/2: one.csv", 0, 2)
    assert not errors
    assert finished[0][0].endswith("points.parquet")
    assert finished[0][1].processed_files == 2
    assert finished[0][1].rows_written == 4


def test_convert_point_csv_worker_emits_error(monkeypatch, tmp_path):
    """ConvertPointCSVWorker should surface point conversion failures."""
    workers = _import_workers_module()
    ConvertPointCSVWorker = workers.ConvertPointCSVWorker

    def fake_convert_point_csv_files_to_parquet(
        _csv_paths,
        _output_path,
        _mapping_path,
        progress_callback=None,
    ):
        raise ValueError("bad headers")

    monkeypatch.setattr(
        "napari_neuron_navigator.point_import.convert_point_csv_files_to_parquet",
        fake_convert_point_csv_files_to_parquet,
    )

    worker = ConvertPointCSVWorker(
        [str(tmp_path / "one.csv")],
        str(tmp_path / "points.parquet"),
        None,
    )
    finished: list[tuple[str, object]] = []
    errors: list[str] = []
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert not finished
    assert errors == ["bad headers"]


def _write_flatmap_v3_parquet(path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.uniform(0.0, 100.0, n).astype(np.float32)
    y = rng.uniform(0.0, 80.0, n).astype(np.float32)
    depth = rng.uniform(0.0, 700.0, n).astype(np.float32)
    frame = pd.DataFrame(
        {
            "file_id": [f"neuron_{i % 4}" for i in range(n)],
            "node_id": np.arange(n, dtype=np.int32),
            "parent_id": np.full(n, -1, dtype=np.int32),
            "type": np.full(n, 3, dtype=np.int32),
            "x_flat_shaped": x,
            "y_flat_shaped": y,
            "x_flat_square": x,
            "y_flat_square": y,
            "depth_um": depth,
            "flatmap_shaped_valid": np.ones(n, dtype=bool),
            "flatmap_square_valid": np.ones(n, dtype=bool),
            "depth_valid": np.ones(n, dtype=bool),
        }
    )
    frame.to_parquet(path, index=False)


def test_flatmap_heatmap_worker_emits_single_volume(tmp_path):
    workers = _import_workers_module()
    path = tmp_path / "flatmap_v3.parquet"
    _write_flatmap_v3_parquet(path)

    worker = workers.FlatmapHeatmapWorker(
        str(path),
        style_key="both_shaped",
        color_mode="single",
        x_bounds=(0.0, 100.0),
        y_bounds=(0.0, 80.0),
        depth_range_um=(0.0, 700.0),
        y_bins=16,
        depth_bin_um=50.0,
        include_depth_minus_one=False,
    )
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert len(finished) == 1
    result = finished[0]
    assert result.volume is not None
    assert result.render_summary.rendered_nodes == 1500
    assert float(result.volume.sum()) == 1500.0


def test_flatmap_heatmap_worker_emits_collapsed_volume(tmp_path):
    workers = _import_workers_module()
    path = tmp_path / "flatmap_v3.parquet"
    _write_flatmap_v3_parquet(path)

    def _run(plane_mode):
        worker = workers.FlatmapHeatmapWorker(
            str(path),
            style_key="both_shaped",
            color_mode="single",
            x_bounds=(0.0, 100.0),
            y_bounds=(0.0, 80.0),
            depth_range_um=(0.0, 700.0),
            y_bins=16,
            depth_bin_um=50.0,
            include_depth_minus_one=False,
            plane_mode=plane_mode,
        )
        finished = []
        errors = []
        worker.finished.connect(finished.append)
        worker.error.connect(errors.append)
        worker.run()
        assert errors == []
        assert len(finished) == 1
        return finished[0]

    flat = _run("flat")
    depth = _run("depth")

    # x(0,100)/y(0,80) is ratio 1.25, so 16 y bins derive 20 x bins.
    assert flat.volume.shape == (16, 20)
    assert flat.volume_shape == (16, 20)
    assert float(flat.volume.sum()) == 1500.0
    assert flat.render_summary.rendered_nodes == 1500
    assert flat.render_summary.depth_bins == 0
    # The flat plane mode drops only the depth axis, not any node.
    np.testing.assert_array_equal(flat.volume, depth.volume.sum(axis=0))


def test_flatmap_heatmap_worker_derives_bounds_when_missing(tmp_path):
    workers = _import_workers_module()
    path = tmp_path / "flatmap_v3.parquet"
    _write_flatmap_v3_parquet(path)

    worker = workers.FlatmapHeatmapWorker(
        str(path),
        style_key="both_shaped",
        color_mode="single",
        x_bounds=None,
        y_bounds=None,
        depth_range_um=None,
        y_bins=16,
        depth_bin_um=50.0,
        include_depth_minus_one=False,
    )
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert len(finished) == 1
    assert float(finished[0].volume.sum()) == 1500.0


def test_flatmap_heatmap_worker_reports_error(tmp_path):
    workers = _import_workers_module()
    worker = workers.FlatmapHeatmapWorker(
        str(tmp_path / "does_not_exist.parquet"),
        style_key="both_shaped",
        color_mode="single",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
        depth_range_um=(0.0, 1.0),
        y_bins=8,
        depth_bin_um=10.0,
        include_depth_minus_one=False,
    )
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert finished == []
    assert len(errors) == 1


def test_flatmap_heatmap_worker_emits_allen_layer_stack(tmp_path):
    workers = _import_workers_module()
    path = tmp_path / "flatmap_v3.parquet"
    _write_flatmap_v3_parquet(path)
    frame = pd.read_parquet(path)
    region_ids = (101, 102, 103, 104, 105, 106)
    frame["region_id"] = np.resize(region_ids, len(frame))
    frame.to_parquet(path, index=False)
    layer_map = AllenIsocortexLayerMap(
        atlas_name="allen_mouse_25um",
        isocortex_region_id=315,
        region_to_layer_index={
            region_id: index for index, region_id in enumerate(region_ids)
        },
        region_ids_by_layer=tuple((region_id,) for region_id in region_ids),
    )

    worker = workers.FlatmapHeatmapWorker(
        str(path),
        style_key="both_shaped",
        color_mode="single",
        x_bounds=(0.0, 100.0),
        y_bounds=(0.0, 80.0),
        depth_range_um=None,
        y_bins=16,
        depth_bin_um=50.0,
        include_depth_minus_one=False,
        plane_mode="allen_layers",
        allen_layer_map=layer_map,
    )
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert len(finished) == 1
    result = finished[0]
    # Six Allen layer planes on the derived rectangular grid.
    assert result.volume.shape == (6, 16, 20)
    assert (result.summary.y_bins, result.summary.x_bins) == (16, 20)
    assert result.summary.rendered_nodes == 1500
    assert float(result.volume.sum()) == 1500.0


def test_flatmap_heatmap_worker_emits_selected_allen_layer_projection(tmp_path):
    workers = _import_workers_module()
    path = tmp_path / "flatmap_v3.parquet"
    _write_flatmap_v3_parquet(path)
    frame = pd.read_parquet(path)
    region_ids = (101, 102, 103, 104, 105, 106)
    frame["region_id"] = np.resize(region_ids, len(frame))
    frame.to_parquet(path, index=False)
    layer_map = AllenIsocortexLayerMap(
        atlas_name="allen_mouse_25um",
        isocortex_region_id=315,
        region_to_layer_index={
            region_id: index for index, region_id in enumerate(region_ids)
        },
        region_ids_by_layer=tuple((region_id,) for region_id in region_ids),
    )
    worker = workers.FlatmapHeatmapWorker(
        str(path),
        style_key="both_shaped",
        color_mode="single",
        x_bounds=(0.0, 100.0),
        y_bounds=(0.0, 80.0),
        depth_range_um=None,
        y_bins=16,
        depth_bin_um=50.0,
        include_depth_minus_one=False,
        plane_mode="allen_layer_projection",
        allen_layer_map=layer_map,
        allen_layer_indices=(1, 4),
        allen_layer_output_mode="projection",
    )
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert len(finished) == 1
    result = finished[0]
    assert result.volume.shape == (16, 20)
    assert result.summary.selected_layer_indices == (1, 4)
    assert result.summary.layer_labels == ("L2/3", "L6a")
    assert result.summary.output_mode == "projection"
    assert result.summary.rendered_nodes == 500
    assert float(result.volume.sum()) == 500.0


def _write_terminus_parquet(path: Path) -> None:
    """One axon-labelled cell with two tips, plus an undefined-only cell."""
    pd.DataFrame(
        {
            "file_id": ["a.swc"] * 4 + ["u.swc"] * 3,
            "node_id": [1, 2, 3, 4, 1, 2, 3],
            "parent_id": [-1, 1, 2, 2, -1, 1, 2],
            "type": [1, 2, 2, 2, 1, 0, 0],
            "x": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0],
            "y": [0.0] * 7,
            "z": [0.0] * 7,
            "neuron_id": ["a"] * 4 + ["u"] * 3,
        }
    ).to_parquet(path, index=False)


def test_terminus_worker_emits_termini_and_coverage(tmp_path):
    """The worker returns axon tips plus the report of skipped neurons."""
    workers = _import_workers_module()
    path = tmp_path / "neurons.parquet"
    _write_terminus_parquet(path)

    worker = workers.TerminusWorker(
        parquet_path=str(path), file_ids=None, node_types=[2]
    )
    results = []
    errors = []
    progress = []
    worker.finished.connect(lambda frame, cov: results.append((frame, cov)))
    worker.error.connect(errors.append)
    worker.progress.connect(lambda *args: progress.append(args))

    worker.run()

    assert errors == []
    assert len(results) == 1
    frame, coverage = results[0]
    assert list(zip(frame["file_id"], frame["node_id"])) == [
        ("a.swc", 3),
        ("a.swc", 4),
    ]
    # u.swc has a childless node but no axon-typed nodes, so it is reported.
    assert coverage.cells_without_selected_types == 1
    assert coverage.file_ids_without == ["u.swc"]
    assert progress[-1][0] == "Done"


def test_terminus_worker_honours_file_ids_and_node_types(tmp_path):
    workers = _import_workers_module()
    path = tmp_path / "neurons.parquet"
    _write_terminus_parquet(path)

    worker = workers.TerminusWorker(
        parquet_path=str(path),
        file_ids=["u.swc"],
        node_types=[0],
    )
    results = []
    worker.finished.connect(lambda frame, cov: results.append((frame, cov)))
    worker.run()

    frame, coverage = results[0]
    assert list(zip(frame["file_id"], frame["node_id"])) == [("u.swc", 3)]
    assert coverage.cells_requested == 1
    assert coverage.cells_without_selected_types == 0


def test_terminus_worker_reports_errors(tmp_path):
    workers = _import_workers_module()
    worker = workers.TerminusWorker(
        parquet_path=str(tmp_path / "missing.parquet"),
        file_ids=None,
        node_types=[2],
    )
    errors = []
    results = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda frame, cov: results.append(frame))

    worker.run()

    assert results == []
    assert len(errors) == 1
