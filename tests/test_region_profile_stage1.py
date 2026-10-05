"""Stage-1 regional-profile construction, validation, and persistence tests."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from napari_neuron_navigator.analysis import region_profile as profile_module
from napari_neuron_navigator.analysis.region_profile import (
    BUILDER_ALGORITHM_VERSION,
    LENGTH_METHOD,
    PROFILE_FORMAT_VERSION,
    REGION_PROFILE_METADATA_KEY,
    RegionalProfileAtlas,
    RegionProfileBuildCancelled,
    RegionProfileValidationError,
    build_hierarchy_closure,
    build_region_profile,
    default_region_profile_path,
    inspect_region_profile,
    register_region_profile,
    resolve_physical_midline,
)
from tests.region_profile_stage0_fixtures import (
    ANISOTROPIC_RESOLUTION_UM,
    LEFT_RIGHT_AXIS,
    MIDLINE_UM,
    morphology_fixture_frame,
    stage0_annotation,
)


def _hemisphere_annotation() -> np.ndarray:
    labels = np.empty(stage0_annotation().shape, dtype=np.uint8)
    labels[:, :, :2] = 2
    labels[:, :, 2:] = 1
    return labels


def _structures() -> dict[int, dict[str, object]]:
    return {
        1: {"id": 1, "acronym": "root", "structure_id_path": [1]},
        11: {"id": 11, "acronym": "FIX", "structure_id_path": [1, 11]},
        22: {"id": 22, "acronym": "MID", "structure_id_path": [1, 22]},
        33: {"id": 33, "acronym": "END", "structure_id_path": [1, 33]},
    }


@pytest.fixture
def profile_atlas() -> RegionalProfileAtlas:
    midline, lower_code, upper_code = resolve_physical_midline(
        _hemisphere_annotation(),
        ANISOTROPIC_RESOLUTION_UM,
        left_right_axis=LEFT_RIGHT_AXIS,
    )
    return RegionalProfileAtlas(
        atlas_name="stage1_fixture",
        atlas_version="1.0",
        annotation=stage0_annotation(),
        resolution_um=ANISOTROPIC_RESOLUTION_UM,
        structures=_structures(),
        left_right_axis=LEFT_RIGHT_AXIS,
        midline_um=midline,
        lower_hemisphere_code=lower_code,
        upper_hemisphere_code=upper_code,
    )


def _write_source(path: Path, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    source = morphology_fixture_frame() if frame is None else frame
    source.to_parquet(path, index=False)
    return source


def _read_profile(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame.sort_values(
        ["file_id", "region_id", "laterality", "compartment"]
    ).reset_index(drop=True)


def test_physical_midline_uses_shared_voxel_boundary() -> None:
    midline, lower_code, upper_code = resolve_physical_midline(
        _hemisphere_annotation(),
        ANISOTROPIC_RESOLUTION_UM,
        left_right_axis=LEFT_RIGHT_AXIS,
    )

    assert midline == MIDLINE_UM
    assert (lower_code, upper_code) == (2, 1)
    # The older voxel-center mirror point would be 45 um for this fixture.
    voxel_center_midpoint = (
        (_hemisphere_annotation().shape[LEFT_RIGHT_AXIS] - 1)
        * ANISOTROPIC_RESOLUTION_UM[LEFT_RIGHT_AXIS]
        / 2
    )
    assert midline != voxel_center_midpoint


def test_profile_atlas_factory_uses_loaded_hemisphere_labels() -> None:
    class FakeAtlas:
        atlas_name = "fixture"
        local_version = (1, 2)
        annotation = stage0_annotation()
        resolution = ANISOTROPIC_RESOLUTION_UM
        hemispheres = _hemisphere_annotation()
        left_hemisphere_value = 1
        right_hemisphere_value = 2
        structures = _structures()

    atlas = RegionalProfileAtlas.from_atlas(FakeAtlas())

    assert atlas.atlas_name == "fixture"
    assert atlas.atlas_version == "1.2"
    assert atlas.midline_um == MIDLINE_UM
    assert atlas.lower_hemisphere_code == 2
    assert atlas.upper_hemisphere_code == 1


def test_midline_resolution_rejects_overlapping_or_gapped_labels() -> None:
    overlapping = _hemisphere_annotation()
    overlapping[0, 0, 2] = 2
    with pytest.raises(ValueError, match="overlap"):
        resolve_physical_midline(
            overlapping,
            ANISOTROPIC_RESOLUTION_UM,
            left_right_axis=LEFT_RIGHT_AXIS,
        )

    gapped = np.zeros((1, 1, 5), dtype=np.uint8)
    gapped[:, :, :2] = 2
    gapped[:, :, 3:] = 1
    with pytest.raises(ValueError, match="do not meet"):
        resolve_physical_midline(
            gapped,
            (1.0, 1.0, 1.0),
            left_right_axis=2,
        )


def test_hierarchy_closure_uses_paths_and_includes_self() -> None:
    structures = {
        1: {"id": 1, "structure_id_path": [1]},
        10: {"id": 10, "structure_id_path": [1, 10]},
        11: {"id": 11, "structure_id_path": [1, 10, 11]},
        12: {"id": 12, "structure_id_path": "/1/10/12/"},
    }

    closure = build_hierarchy_closure(structures)

    assert closure.represented_direct_ids(1) == (1, 10, 11, 12)
    assert closure.represented_direct_ids(10) == (10, 11, 12)
    assert closure.represented_direct_ids(11) == (11,)
    assert closure.digest == build_hierarchy_closure(structures).digest
    self_link = [
        link
        for link in closure.links
        if link.ancestor_region_id == 11 and link.direct_region_id == 11
    ]
    assert self_link[0].depth == 0


def test_builder_writes_sparse_profile_and_conserves_all_totals(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
) -> None:
    source = tmp_path / "neurons.parquet"
    frame = _write_source(source)
    sidecar = tmp_path / "neurons.region_profile.parquet"
    progress: list[tuple[str, int, int]] = []

    metadata = build_region_profile(
        source,
        profile_atlas,
        output_path=sidecar,
        max_batch_neurons=1,
        max_batch_rows=2,
        row_group_size=2,
        progress_callback=lambda message, current, total: progress.append(
            (message, current, total)
        ),
    )

    assert sidecar.exists()
    assert metadata.profile_format_version == PROFILE_FORMAT_VERSION
    assert metadata.builder_algorithm_version == BUILDER_ALGORITHM_VERSION
    assert metadata.length_method == LENGTH_METHOD
    assert metadata.build_summary.neuron_count == 2
    assert metadata.build_summary.batch_count == 2
    assert metadata.build_summary.source_row_count == len(frame)
    assert metadata.build_summary.node_count == len(frame)
    assert metadata.build_summary.terminus_count == 2
    assert metadata.build_summary.total_edge_count == 4
    assert metadata.build_summary.total_cable_length_um == pytest.approx(40.0)
    assert metadata.build_summary.allocated_cable_length_um == pytest.approx(40.0)
    assert metadata.build_summary.soma_laterality_counts == {"ipsilateral": 2}
    assert progress[-1][1] == progress[-1][2]

    result = _read_profile(sidecar)
    assert set(result["file_id"]) == {"mirror-lower.swc", "mirror-upper.swc"}
    assert set(result["laterality"].astype(str)) == {"ipsilateral"}
    assert set(result["compartment"].astype(str)) == {"soma", "neurite"}
    for file_id in ("mirror-lower.swc", "mirror-upper.swc"):
        neuron = result.loc[result["file_id"] == file_id]
        soma = neuron.loc[neuron["compartment"].astype(str) == "soma"].iloc[0]
        neurite = neuron.loc[neuron["compartment"].astype(str) == "neurite"].iloc[0]
        assert soma["node_count"] == 1
        assert soma["cable_length_um"] == 0
        assert neurite["node_count"] == 2
        assert neurite["terminus_count"] == 1
        assert neurite["cable_length_um"] == pytest.approx(20.0)

    schema_metadata = pq.read_schema(sidecar).metadata or {}
    assert REGION_PROFILE_METADATA_KEY in schema_metadata
    inspection = inspect_region_profile(
        sidecar,
        source_path=source,
        atlas=profile_atlas,
        validate_contents=True,
    )
    assert inspection.valid
    assert inspection.compatible
    assert inspection.issues == ()
    assert not list(tmp_path.glob(".neurons.region_profile.parquet.source_batches.*"))


def test_default_sidecar_name() -> None:
    assert default_region_profile_path("folder/source.parquet") == Path(
        "folder/source.region_profile.parquet"
    )


@pytest.mark.parametrize("max_batch_neurons", [1, 1024], ids=["staged", "single"])
@pytest.mark.parametrize("custom_output", [False, True], ids=["default", "custom"])
def test_sidecar_create_rebuild_and_release_files(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
    max_batch_neurons: int,
    custom_output: bool,
) -> None:
    """Exercise real file handles and native paths, including on Windows CI."""
    folder = tmp_path / "researcher's data" / "régions"
    folder.mkdir(parents=True)
    source = folder / "neurons.parquet"
    frame = _write_source(source)
    destination = (
        folder / "output" / "custom.parquet"
        if custom_output
        else default_region_profile_path(source)
    )
    kwargs = {"output_path": destination} if custom_output else {}

    # Rebuild with a different source so merely retaining the old sidecar
    # cannot pass. Both original neurons share neuron_id and node_id values.
    for expected_neurons in (2, 1):
        if expected_neurons == 1:
            frame = frame.loc[frame["file_id"] == "mirror-lower.swc"].copy()
            _write_source(source, frame)
        source_bytes = source.read_bytes()
        metadata = build_region_profile(
            source,
            profile_atlas,
            max_batch_neurons=max_batch_neurons,
            row_group_size=2,
            **kwargs,
        )

        inspection = inspect_region_profile(
            destination,
            source_path=source,
            atlas=profile_atlas,
            validate_contents=True,
        )
        assert inspection.valid and inspection.compatible, inspection.issues
        assert inspection.metadata == metadata
        assert metadata.build_summary.neuron_count == expected_neurons
        assert metadata.build_summary.batch_count == (
            expected_neurons if max_batch_neurons == 1 else 1
        )
        result = _read_profile(destination)
        assert set(result["file_id"]) == set(frame["file_id"])
        assert result.groupby("file_id")["node_count"].sum().to_dict() == {
            file_id: 3 for file_id in frame["file_id"].unique()
        }
        assert result["terminus_count"].sum() == expected_neurons
        assert result["cable_length_um"].sum() == pytest.approx(20 * expected_neurons)
        assert source.read_bytes() == source_bytes
        assert set(destination.parent.iterdir()) == (
            {destination} if custom_output else {source, destination}
        )

    # Windows refuses these operations if the builder/inspector leaked a
    # handle that denies deletion. Test both the input and the published file.
    for path in (source, destination):
        moved = path.with_suffix(".moved.parquet")
        path.rename(moved)
        moved.unlink()


def test_fsync_file_requires_write_access_without_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce Windows' writable-handle requirement on macOS/Linux too."""
    path = tmp_path / "pending.parquet"
    _write_source(path)
    original = path.read_bytes()
    real_fsync = os.fsync
    synced = []

    def require_writable_descriptor(descriptor: int) -> None:
        # A zero-byte write checks access without changing the file. With the
        # old read-only open this raises EBADF even on a POSIX developer host.
        os.write(descriptor, b"")
        real_fsync(descriptor)
        synced.append(descriptor)

    monkeypatch.setattr(profile_module.os, "fsync", require_writable_descriptor)
    profile_module._fsync_file(path)

    assert len(synced) == 1
    assert path.read_bytes() == original
    with pytest.raises(OSError) as closed:
        os.fstat(synced[0])
    assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("error_number", [errno.EBADF, errno.EIO])
def test_fsync_failure_preserves_valid_sidecar_and_cleans_up(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    source = tmp_path / "neurons.parquet"
    _write_source(source)
    destination = default_region_profile_path(source)
    build_region_profile(source, profile_atlas)
    original = destination.read_bytes()

    def fail_fsync(_descriptor: int) -> None:
        raise OSError(error_number, "injected file flush failure")

    # Inject at the OS call, after the real writers have completed. A failed
    # flush must propagate; suppressing it would publish an unflushed file.
    with monkeypatch.context() as patch:
        patch.setattr(profile_module.os, "fsync", fail_fsync)
        with pytest.raises(OSError, match="injected file flush failure") as failure:
            build_region_profile(source, profile_atlas, max_batch_neurons=1)

    assert failure.value.errno == error_number
    assert destination.read_bytes() == original
    assert set(tmp_path.iterdir()) == {source, destination}
    inspection = inspect_region_profile(
        destination, source_path=source, atlas=profile_atlas, validate_contents=True
    )
    assert inspection.compatible, inspection.issues
    # Retry with real fsync to verify cleanup left no locked staging files.
    build_region_profile(source, profile_atlas, max_batch_neurons=1)
    assert set(tmp_path.iterdir()) == {source, destination}


def test_builder_uses_relative_laterality_and_retains_region_zero(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "file_id": "crossing.swc",
                "node_id": 30,
                "type": 0,
                "x": 5.0,
                "y": 10.0,
                "z": 130.0,
                "parent_id": 20,
                "region_id": 0,
            },
            {
                "file_id": "crossing.swc",
                "node_id": 20,
                "type": 2,
                "x": 5.0,
                "y": 10.0,
                "z": 75.0,
                "parent_id": 10,
                "region_id": 11,
            },
            {
                "file_id": "crossing.swc",
                "node_id": 10,
                "type": 1,
                "x": 5.0,
                "y": 10.0,
                "z": 45.0,
                "parent_id": -1,
                "region_id": 11,
            },
        ]
    )
    source = tmp_path / "crossing.parquet"
    _write_source(source, frame)
    sidecar = tmp_path / "crossing.profile.parquet"

    build_region_profile(source, profile_atlas, output_path=sidecar)
    result = _read_profile(sidecar)

    assert 0 in set(result["region_id"])
    assert {"ipsilateral", "contralateral"} <= set(result["laterality"].astype(str))
    outside = result.loc[result["region_id"] == 0]
    assert outside["node_count"].sum() == 1
    assert outside["terminus_count"].sum() == 1
    assert outside["cable_length_um"].sum() == pytest.approx(10.0)


def test_midline_and_out_of_bounds_somas_remain_explicit(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
) -> None:
    rows: list[dict[str, object]] = []
    for file_id, soma_z, child_z in (
        ("midline.swc", MIDLINE_UM, MIDLINE_UM),
        ("unknown.swc", 150.0, 90.0),
    ):
        rows.extend(
            [
                {
                    "file_id": file_id,
                    "node_id": 2,
                    "type": 0,
                    "x": 5.0,
                    "y": 10.0,
                    "z": child_z,
                    "parent_id": 1,
                    "region_id": 0 if child_z >= 120 else 11,
                },
                {
                    "file_id": file_id,
                    "node_id": 1,
                    "type": 1,
                    "x": 5.0,
                    "y": 10.0,
                    "z": soma_z,
                    "parent_id": -1,
                    "region_id": 0 if soma_z >= 120 else 11,
                },
            ]
        )
    source = tmp_path / "special.parquet"
    _write_source(source, pd.DataFrame(rows))
    sidecar = tmp_path / "special.profile.parquet"

    metadata = build_region_profile(source, profile_atlas, output_path=sidecar)
    result = _read_profile(sidecar)

    assert metadata.build_summary.soma_laterality_counts == {
        "midline": 1,
        "unknown": 1,
    }
    assert set(
        result.loc[result["file_id"] == "midline.swc", "laterality"].astype(str)
    ) == {"midline"}
    assert set(
        result.loc[result["file_id"] == "unknown.swc", "laterality"].astype(str)
    ) == {"unknown"}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.assign(node_id=[250, 250, 100, 900, 250, 100]),
            "duplicate node_id",
        ),
        (
            lambda frame: frame.assign(parent_id=[999, 100, -1, 250, 100, -1]),
            "dangling parent_id",
        ),
        (
            lambda frame: frame.assign(type=frame["type"].replace({1: 0})),
            "exactly one soma",
        ),
        (
            lambda frame: frame.assign(type=1),
            "exactly one soma",
        ),
        (
            lambda frame: frame.assign(x=[np.nan, 5, 5, 5, 5, 5]),
            "coordinates must all be finite",
        ),
        (
            lambda frame: frame.assign(parent_id=[250, 900, -1, 250, 100, -1]),
            "not connected to the root",
        ),
        (
            lambda frame: frame.assign(region_id=[44, 11, 11, 11, 11, 11]),
            "absent from atlas hierarchy",
        ),
    ],
)
def test_builder_reports_actionable_graph_validation_errors(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
    mutate,
    message: str,
) -> None:
    source = tmp_path / "invalid.parquet"
    _write_source(source, mutate(morphology_fixture_frame()))

    with pytest.raises(RegionProfileValidationError, match=message):
        build_region_profile(source, profile_atlas)

    assert not default_region_profile_path(source).exists()


def test_whole_tree_terminus_lookup_does_not_invent_typed_parent_termini(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
) -> None:
    source = tmp_path / "neurons.parquet"
    _write_source(source)
    sidecar = tmp_path / "profile.parquet"

    build_region_profile(source, profile_atlas, output_path=sidecar)
    result = _read_profile(sidecar)

    # Each file has one true childless type-0 node.  Node 250 is type 2 but
    # has that child, and must not become a false terminus.
    assert result["terminus_count"].sum() == 2
    neurite = result.loc[result["compartment"].astype(str) == "neurite"]
    assert neurite["node_count"].sum() == 4


def test_cancel_and_failure_preserve_existing_sidecar(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "neurons.parquet"
    _write_source(source)
    destination = tmp_path / "profile.parquet"
    sentinel = b"existing-sidecar"
    destination.write_bytes(sentinel)

    with pytest.raises(RegionProfileBuildCancelled):
        build_region_profile(
            source,
            profile_atlas,
            output_path=destination,
            cancel_check=lambda: True,
        )
    assert destination.read_bytes() == sentinel
    assert not list(tmp_path.glob(".profile.parquet.*.parquet"))

    checks = 0

    def cancel_during_staging() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 4

    with pytest.raises(RegionProfileBuildCancelled):
        build_region_profile(
            source,
            profile_atlas,
            output_path=destination,
            max_batch_neurons=1,
            cancel_check=cancel_during_staging,
        )
    assert destination.read_bytes() == sentinel
    assert not list(tmp_path.glob(".profile.parquet.source_batches.*"))

    def fail_compaction(*args, **kwargs):
        raise RuntimeError("injected compaction failure")

    monkeypatch.setattr(profile_module, "_write_sorted_profile", fail_compaction)
    with pytest.raises(RuntimeError, match="injected"):
        build_region_profile(
            source,
            profile_atlas,
            output_path=destination,
            max_batch_neurons=1,
        )
    assert destination.read_bytes() == sentinel
    assert not list(tmp_path.glob(".profile.parquet.*.parquet"))
    assert not list(tmp_path.glob(".profile.parquet.source_batches.*"))


def test_source_relocation_and_metadata_only_change_remain_compatible(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
) -> None:
    source = tmp_path / "neurons.parquet"
    _write_source(source)
    sidecar = tmp_path / "profile.parquet"
    build_region_profile(source, profile_atlas, output_path=sidecar)

    relocated = tmp_path / "moved.parquet"
    shutil.copyfile(source, relocated)
    relocated_inspection = inspect_region_profile(
        sidecar, source_path=relocated, atlas=profile_atlas
    )
    assert relocated_inspection.compatible

    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    touched_inspection = inspect_region_profile(
        sidecar, source_path=source, atlas=profile_atlas
    )
    assert touched_inspection.compatible


def test_source_and_atlas_content_changes_reject_stale_sidecar(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
) -> None:
    source = tmp_path / "neurons.parquet"
    original = _write_source(source)
    sidecar = tmp_path / "profile.parquet"
    build_region_profile(source, profile_atlas, output_path=sidecar)

    changed = original.copy()
    changed.loc[0, "region_id"] = 22
    _write_source(source, changed)
    source_inspection = inspect_region_profile(
        sidecar, source_path=source, atlas=profile_atlas
    )
    assert not source_inspection.compatible
    assert any(
        "source Parquet content differs" in issue for issue in source_inspection.issues
    )

    changed_annotation = profile_atlas.annotation.copy()
    changed_annotation[0, 0, 0] = 22
    changed_atlas = RegionalProfileAtlas(
        atlas_name=profile_atlas.atlas_name,
        atlas_version=profile_atlas.atlas_version,
        annotation=changed_annotation,
        resolution_um=profile_atlas.resolution_um,
        structures=profile_atlas.structures,
        left_right_axis=profile_atlas.left_right_axis,
        midline_um=profile_atlas.midline_um,
    )
    atlas_inspection = inspect_region_profile(sidecar, atlas=changed_atlas)
    assert not atlas_inspection.compatible
    assert any("annotation_digest" in issue for issue in atlas_inspection.issues)


def test_registration_materializes_only_the_validated_sidecar(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "neurons.parquet"
    _write_source(source)
    sidecar = tmp_path / "profile.parquet"
    metadata = build_region_profile(source, profile_atlas, output_path=sidecar)

    def fail_source_scan(*args, **kwargs):
        raise AssertionError("unchanged source validation must not scan node data")

    monkeypatch.setattr(profile_module, "_streaming_sha256", fail_source_scan)
    connection = duckdb.connect()
    try:
        inspection = register_region_profile(
            connection,
            sidecar,
            source_path=source,
            atlas=profile_atlas,
            table_name="loaded_profile",
        )
        rows = connection.execute(
            "SELECT COUNT(*), SUM(node_count) FROM loaded_profile"
        ).fetchone()
    finally:
        connection.close()

    assert inspection.compatible
    assert rows == (metadata.build_summary.profile_row_count, 6)


def test_registration_rejects_invalid_table_name(
    tmp_path: Path,
    profile_atlas: RegionalProfileAtlas,
) -> None:
    source = tmp_path / "neurons.parquet"
    _write_source(source)
    sidecar = tmp_path / "profile.parquet"
    build_region_profile(source, profile_atlas, output_path=sidecar)
    connection = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="simple SQL identifier"):
            register_region_profile(
                connection,
                sidecar,
                source_path=source,
                atlas=profile_atlas,
                table_name="profile; DROP TABLE profile",
            )
    finally:
        connection.close()
