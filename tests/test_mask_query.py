"""Compare bounded mask scans with an independent point-by-point oracle."""

from threading import Event

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_neuron_navigator.analysis.mask_query import (
    MaskQueryCancelled,
    prepare_query_mask,
    query_neurons_in_masks,
)


def _source(tmp_path, *, dictionary=True, statistics=True):
    rng = np.random.default_rng(13)
    count = 1031
    ids = np.array(["a/shared", "b/shared", "apostrophe's neuron", "other"])
    frame = pd.DataFrame(
        {
            "file_id": ids[np.arange(count) % 4],
            "neuron_id": ["shared"] * count,
            "subject": np.array(["a", "b", "c", "d"])[np.arange(count) % 4],
            "type": rng.choice([0, 1, 2, 3], count),
            "x": rng.uniform(-1, 100, count),
            "y": rng.uniform(-1, 90, count),
            "z": rng.uniform(-1, 80, count),
        }
    )
    frame.loc[:3, "x"] = np.array([np.nan, np.inf, -np.inf, np.nan])
    path = tmp_path / "nodes.parquet"
    pq.write_table(
        pa.Table.from_pandas(frame),
        path,
        row_group_size=29,
        use_dictionary=dictionary,
        write_statistics=statistics,
    )
    return path, frame


@pytest.mark.parametrize("dictionary,statistics", [(True, True), (False, False)])
@pytest.mark.parametrize("batch_size", [1, 37, 65536])
@pytest.mark.parametrize(
    "types,scope,excluded",
    [
        (None, None, ()),
        ((1,), None, ()),
        ((0, 2), ("a/shared", "b/shared"), ("a/shared",)),
    ],
)
def test_scan_agrees_with_point_oracle(
    tmp_path, dictionary, statistics, batch_size, types, scope, excluded
):
    path, frame = _source(tmp_path, dictionary=dictionary, statistics=statistics)
    mask = np.zeros((5, 6, 7), dtype=bool)
    mask[1:4, 2:5, 1:3] = True
    mask[4, 5, 6] = True
    resolution = np.array([20.0, 15.0, 10.0])
    expected = set()
    for row in frame.itertuples():
        xyz = np.array([row.x, row.y, row.z])
        if not np.isfinite(xyz).all() or (types is not None and row.type not in types):
            continue
        if row.file_id in excluded or (scope is not None and row.file_id not in scope):
            continue
        voxel = np.floor(xyz / resolution).astype(int)
        if (voxel >= 0).all() and (voxel < mask.shape).all() and mask[tuple(voxel)]:
            expected.add(row.file_id)
    progress = []
    result = query_neurons_in_masks(
        path,
        [mask],
        atlas_shape=mask.shape,
        resolution_um=resolution,
        node_types=types,
        file_ids=scope,
        exclude_file_ids=excluded,
        batch_size=batch_size,
        progress_callback=lambda *args: progress.append(args),
    )
    assert result.file_id.tolist() == sorted(expected)
    assert result.file_id.is_unique
    scans = [
        (done, total)
        for stage, done, total in progress
        if stage.startswith("Searching nodes")
    ]
    assert scans[-1] == (len(frame), len(frame))
    assert [done for done, _total in scans] == sorted(done for done, _total in scans)


def test_mask_snapshot_union_and_bounds_never_enumerate_positive_voxels(monkeypatch):
    a, b = np.zeros((19, 5, 7), dtype=np.uint32), np.zeros((19, 5, 7), dtype=np.uint8)
    a[0, 1, 2] = 1
    b[18, 4, 6] = 2
    monkeypatch.setattr(
        np, "argwhere", lambda *_a: pytest.fail("Do not enumerate mask coordinates")
    )
    mask, bounds = prepare_query_mask([a, b], a.shape)
    assert mask.sum() == 2
    np.testing.assert_array_equal(bounds[0], [0, 1, 2])
    np.testing.assert_array_equal(bounds[1], [19, 5, 7])
    a[:] = 0
    assert mask[0, 1, 2]


def test_full_voxel_extent_is_included_and_next_voxel_is_excluded(tmp_path):
    path = tmp_path / "edge.parquet"
    pd.DataFrame(
        {
            "file_id": ["inside", "next", "below"],
            "neuron_id": ["same"] * 3,
            "subject": ["a", "b", "c"],
            "x": [49.99, 50.0, 24.99],
            "y": [25.0] * 3,
            "z": [25.0] * 3,
        }
    ).to_parquet(path)
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    result = query_neurons_in_masks(
        path, [mask], atlas_shape=mask.shape, resolution_um=(25,) * 3
    )
    assert result.file_id.tolist() == ["inside"]


@pytest.mark.parametrize("phase", ["before", "prepare", "scan"])
def test_cancellation_never_returns_partial_results(tmp_path, phase):
    path, _frame = _source(tmp_path)
    event = Event()
    if phase == "before":
        event.set()

    def progress(stage, done, _total):
        if (phase == "prepare" and stage == "Preparing mask") or (
            phase == "scan" and stage.startswith("Searching nodes") and done
        ):
            event.set()

    with pytest.raises(MaskQueryCancelled):
        query_neurons_in_masks(
            path,
            [np.ones((5, 6, 7), dtype=bool)],
            atlas_shape=(5, 6, 7),
            resolution_um=(20, 15, 10),
            cancel_check=event.is_set,
            progress_callback=progress,
            batch_size=7,
        )


def test_pruning_and_empty_masks(tmp_path):
    path, _frame = _source(tmp_path)
    empty = np.zeros((5, 6, 7), dtype=bool)
    result = query_neurons_in_masks(
        path, [empty], atlas_shape=empty.shape, resolution_um=(1, 1, 1)
    )
    assert result.empty and result.attrs["empty_mask"]
    with pytest.raises(ValueError, match="shape"):
        query_neurons_in_masks(
            path, [empty], atlas_shape=(2, 2, 2), resolution_um=(1, 1, 1)
        )
    for scope, types in (([], None), (None, [])):
        assert query_neurons_in_masks(
            path,
            [~empty],
            atlas_shape=empty.shape,
            resolution_um=(1, 1, 1),
            file_ids=scope,
            node_types=types,
        ).empty
